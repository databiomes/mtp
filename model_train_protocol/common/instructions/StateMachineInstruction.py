from typing import List, Union

from model_train_protocol.errors import InstructionError, InstructionTypeError
from .BaseInstruction import BaseInstruction, Sample
from .input.StateMachineInput import StateMachineInput
from .output.StateMachineOutput import StateMachineOutput
from ..constants import NON_TOKEN, STATE_MACHINE_MINIMUM_INSTRUCTION_SAMPLES, PER_FINAL_TOKEN_SAMPLE_MINIMUM
from ..guardrails import Guardrail
from ..tokens.TokenSet import TokenSet, Snippet
from ... import Token


class StateSample(Sample):
    """A Sample whose output is a state Token. The output string is the token's key."""

    def __init__(self, input: List[str], state: Token, numbers: List[List[int]],
                 number_lists: List[List[List[int]]]):
        self.input: List[str] = input
        self.state: Token = state
        self.prompt = None
        self.numbers: List[List[int]] = numbers
        self.number_lists: List[List[List[int]]] = number_lists
        self.result = NON_TOKEN
        self.value = None

    @property
    def output(self) -> str:
        """The state token's key, resolved at read time so protocol-assigned keys are used."""
        return self.state.key if self.state.key is not None else self.state.value


class StateMachineInstruction(BaseInstruction):
    """
    Special Instruction for State Machine Protocols.

    The model responds with exactly one state Token, followed by the default final token.
    """
    input: StateMachineInput
    output: StateMachineOutput

    minimum_samples: int = STATE_MACHINE_MINIMUM_INSTRUCTION_SAMPLES

    def __init__(self, input: StateMachineInput, states: List[Token]):
        """
        Initializes an Instruction instance.

        :param input: StateMachineInput defining the input TokenSets. This precedes the model's response.
        :param states: Tokens the model can respond with.
        """
        state_token: Token = Token("State",
                                   desc=f"The response is exactly one of these state tokens: {[s.value for s in states]}.")
        instruction_output: StateMachineOutput = StateMachineOutput(
            tokenset=TokenSet(tokens=[state_token]),
            states=states,
        )
        super().__init__(input=input, output=instruction_output, context=[], name="StateMachineInstruction")
        if not isinstance(self.output, StateMachineOutput):
            raise InstructionTypeError(f"Output must be an instance of StateMachineOutput. Got: {type(self.output)}")
        self._validate_input_snippets()

    def _validate_snippets_match(self, inputs: List[Snippet]):
        """Validates that all input snippets match their expected token sets."""
        for snippet, token_set in zip(inputs, self.input.tokensets):
            self._validate_snippet_matches_set(snippet=snippet, expected_token_set=token_set)

    def _validate_minimum_samples(self):
        """Validates the instruction-wide minimum and that every state token has enough samples."""
        super()._validate_minimum_samples()
        state_counts: dict[Token, int] = {state: 0 for state in self.output.states}
        for sample in self.samples:
            state_counts[sample.state] += 1
        for state, count in state_counts.items():
            if count < PER_FINAL_TOKEN_SAMPLE_MINIMUM:
                raise InstructionError(
                    f"Instruction '{self.name}' has only {count} samples for state token '{state.value}'. "
                    f"Each state token must have at least {PER_FINAL_TOKEN_SAMPLE_MINIMUM} samples."
                )

    @property
    def last_tokenset(self) -> TokenSet:
        """Returns the last TokenSet in the Instruction, which is the response TokenSet."""
        return self.output.tokenset

    @property
    def has_guardrails(self) -> bool:
        """Indicates whether the Instruction has any guardrails defined."""
        return len(self.input.guardrails) > 0

    def get_guardrails(self) -> list[Guardrail]:
        """Gets the guardrails attached to the instruction"""
        return list(self.input.guardrails.values())

    def get_tokens(self) -> List[Token]:
        """Returns all tokens in the instruction, including the state tokens."""
        return super().get_tokens() + list(self.output.states)

    def get_states(self) -> list[str]:
        """Returns the state token keys used by the samples, plus each guardrail's text output."""
        states: list[str] = list(dict.fromkeys(sample.output for sample in self.samples))
        for guardrail in self.get_guardrails():
            if guardrail.bad_output not in states:
                states.append(guardrail.bad_output)
        return states

    # noinspection PyMethodOverriding
    def add_sample(self, input_snippets: List[Union[str | Snippet]], state: Token):
        """
        Add a sample to the Instruction.

        :param input_snippets: List of context snippets or strings that will be added to the Instruction.
        :param state: The state token the model responds with.
        """
        input_snippets: List[Snippet] = self._enforce_input_snippets(inputs=input_snippets)
        self.output.validate_sample(state=state)
        self._assert_input_snippet_count(inputs=input_snippets)
        self._validate_snippets_match(inputs=input_snippets)
        self._validate_snippet_length(inputs=input_snippets, response_snippet=self.output.tokenset.create_snippet(
            string=state.value))
        self.samples.append(StateSample(
            input=[snippet.string for snippet in input_snippets],
            state=state,
            numbers=[snippet.numbers for snippet in input_snippets] + [[]],
            number_lists=[snippet.number_lists for snippet in input_snippets] + [[]],
        ))

    def add_guardrail(self, guardrail: Guardrail, tokenset_index: int):
        """
        Adds a guardrail to the Instruction.

        :param guardrail: The Guardrail instance to add.
        :param tokenset_index: The index of the TokenSet the guardrail applies to.
        """
        if len(guardrail.samples) < 3:
            raise InstructionError(
                "Guardrail must have at least 3 samples of bad inputs before being added to an Instruction.")

        self.input.add_guardrail(guardrail=guardrail, tokenset_index=tokenset_index)
