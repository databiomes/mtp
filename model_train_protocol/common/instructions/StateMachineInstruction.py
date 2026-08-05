from typing import List, Union

from model_train_protocol.errors import InstructionError, InstructionTypeError
from .BaseInstruction import BaseInstruction, Sample
from .input.StateMachineInput import StateMachineInput
from .output.StateMachineOutput import StateMachineOutput
from ..constants import NON_TOKEN, STATE_MACHINE_MINIMUM_INSTRUCTION_SAMPLES
from ..guardrails import Guardrail
from ..tokens.FinalToken import FinalToken
from ..tokens.TokenSet import TokenSet, Snippet
from ... import Token


class StateMachineInstruction(BaseInstruction):
    """
    Special Instruction for State Machine Protocols.

    StateMachineInstructions do not have final tokens.
    """
    input: StateMachineInput
    output: StateMachineOutput

    minimum_samples: int = STATE_MACHINE_MINIMUM_INSTRUCTION_SAMPLES

    def __init__(self, input: StateMachineInput, states: List[str]):
        """
        Initializes an Instruction instance.

        :param input: List of tuples containing Token instances that define the input structure. This precedes the model's response.
        :param states: List of possible states that the model can respond with. These will be added to the response TokenSet.
        """
        state_token: Token = Token("State", desc=f"The responses that are acceptable: {states}.")
        state_tokenset = TokenSet(tokens=[state_token])
        instruction_output: StateMachineOutput = StateMachineOutput(
            tokenset=state_tokenset,
        )
        super().__init__(input=input, output=instruction_output, context=[], name="StateMachineInstruction")
        if not isinstance(self.output, StateMachineOutput):
            raise InstructionTypeError(f"Output must be an instance of StateMachineOutput. Got: {type(self.output)}")
        self._validate_input_snippets()

    def _validate_snippets_match(self, inputs: List[Snippet], response_snippet: Snippet):
        """Validates that all snippets in the samples match their expected token sets."""
        all_snippets: List[Snippet] = inputs + [response_snippet]
        all_token_sets: List[TokenSet] = self.get_token_sets()

        for i in range(len(all_snippets)):
            self._validate_snippet_matches_set(snippet=all_snippets[i], expected_token_set=all_token_sets[i])

        # Validate output snippet set matches output token set
        self._validate_snippet_matches_set(snippet=response_snippet, expected_token_set=self.output.tokenset)

    def get_token_sets(self) -> List[TokenSet]:
        """Returns all tokens in the instruction as a list of tuples."""
        all_tokens_sets: List = []
        for token_set in self.input.tokensets:
            all_tokens_sets.append(token_set)
        all_tokens_sets.append(self.output.tokenset)
        return all_tokens_sets

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

    def get_states(self) -> list[str]:
        """Returns the list of states defined in the TokenSet."""
        # Deduplicated in sample order rather than through a set, so the states (and every file generated from them)
        # come out the same on every run.
        states: list[str] = list(dict.fromkeys(sample.output for sample in self.samples))
        if self.has_guardrails:
            states.append("GUARDRAIL")
        return states

    # noinspection PyMethodOverriding
    def add_sample(self, input_snippets: List[Union[str | Snippet]], state: str):
        f"""
        Add a sample to the Instruction.

        :param input_snippets: List of context snippets or strings that will be added to the Instruction.
        :param state: The model's response state.
        """
        input_snippets: List[Snippet] = self._enforce_input_snippets(inputs=input_snippets)
        output_snippet: Snippet = self._enforce_response_snippet(state)
        final: FinalToken = NON_TOKEN
        self.output.validate_sample(snippet=output_snippet)
        self._assert_input_snippet_count(inputs=input_snippets)
        self._validate_snippets_match(inputs=input_snippets, response_snippet=output_snippet)
        self._validate_snippet_length(inputs=input_snippets, response_snippet=output_snippet)
        sample: Sample = self._create_sample(inputs=input_snippets, response_snippet=output_snippet, final=final)
        self.samples.append(sample)

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
