from typing import List, Union

from .BaseInstruction import BaseInstruction, Sample
from .input.InstructionInput import InstructionInput
from .output.InstructionOutput import InstructionOutput
from ..guardrails import Guardrail
from ..tokens.FinalToken import FinalToken
from ..tokens.TokenSet import TokenSet, Snippet
from model_train_protocol.errors import InstructionError, InstructionTypeError


class Instruction(BaseInstruction):
    """
    Instructions are provided to the model to guide its behavior.

    It includes context Tokens that define the input structure and a response TokenSet that defines the expected output.

    Samples must be added to the Instruction to provide context for the model.
    A minimum of 3 samples must be added to an Instruction.
    """
    output: InstructionOutput
    input: InstructionInput

    def __init__(self, name: str, input: InstructionInput, output: InstructionOutput, context: List[str] | None = None):
        """
        Initializes an Instruction instance.

        :param name: Name for the Instruction. Must be unique across all instructions.
        :param input: InstructionInput instance containing the input structure.
        :param output: InstructionOutput instance containing the output structure.
        :param context: A list of strings providing background context for the instruction.
        """
        super().__init__(input=input, output=output, context=context, name=name)
        if not isinstance(self.output, InstructionOutput):
            raise InstructionTypeError(f"Output must be an instance of Output. Got: {type(self.output)}")
        self._validate_input_snippets()

    def _validate_snippets_match(self, inputs: List[Snippet], response_snippet: Snippet):
        """Validates that all snippets in the samples match their expected token sets."""
        all_snippets: List[Snippet] = inputs + [response_snippet]
        all_token_sets: List[TokenSet] = self.get_token_sets()

        for i in range(len(all_snippets)):
            self._validate_snippet_matches_set(snippet=all_snippets[i], expected_token_set=all_token_sets[i])

        if not isinstance(self.output, InstructionOutput):
            raise InstructionTypeError(f"Output must be an instance of Output. Got: {type(self.output)}")

        # Validate output snippet set matches output token set
        self._validate_snippet_matches_set(snippet=response_snippet, expected_token_set=self.output.tokenset)

    # noinspection PyMethodOverriding
    def add_sample(self, input_snippets: List[Union[str | Snippet]], output_snippet: Snippet | str,
                   output_value: Union[int, float, List[Union[int, float]], None] = None, final: FinalToken | None = None):
        """
        Add a sample to the Instruction.

        :param input_snippets: List of context snippets or strings that will be added to the Instruction.
        :param output_snippet: The model's response snippet.
        :param output_value: Optional value ascribed to the final Instruction output IF the final Token output is a number.
        :param final: Optional Token instance designating th e final action by the model. Defaults to a non-action Token designated {self.output.default_final}.
        """
        input_snippets: List[Snippet] = self._enforce_input_snippets(inputs=input_snippets)
        output_snippet: Snippet = self._enforce_response_snippet(output_snippet)
        final: FinalToken = self._assign_final_token(final=final)
        self.output.validate_sample(snippet=output_snippet, value=output_value, final=final)
        self._assert_input_snippet_count(inputs=input_snippets)
        self._validate_snippets_match(inputs=input_snippets, response_snippet=output_snippet)
        self._validate_snippet_length(inputs=input_snippets, response_snippet=output_snippet)

        sample: Sample = self._create_sample(inputs=input_snippets, response_snippet=output_snippet,
                                             value=output_value, final=final)
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
