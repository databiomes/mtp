from typing import Dict
from typing import List, Union

from model_train_protocol import Token
from model_train_protocol.common.instructions.output.MultiClassifierOutput import MultiClassifierOutput
from model_train_protocol.errors import InstructionTypeError
from .BaseInstruction import BaseInstruction, Sample
from .input.InstructionInput import InstructionInput
from ..tokens.FinalToken import FinalToken
from ..tokens.TokenSet import TokenSet, Snippet


class MultiClassifierInstruction(BaseInstruction):
    """
    Instruction that responds with multiple classification outputs in JSON format.

    Each classification output is represented as a key-value pair in the JSON object,
    where the key is the classification label and the value is the corresponding classification result.
    """
    input: InstructionInput
    output: MultiClassifierOutput

    def __init__(self, input: InstructionInput, state_map: Dict[str, List[str]]):
        """
        Initializes an Instruction instance.

        :param input: List of tuples containing Token instances that define the input structure. This precedes the model's response.
        :param state_map: A dictionary mapping classification labels (keys) to their corresponding acceptable values (list of strings).
        """
        state_token: Token = Token("States",
                                   desc=f"Acceptable responses must contain exactly 1 key and 1 value for every key in this map, in any combination: {state_map}.")
        state_tokenset = TokenSet(tokens=[state_token])
        instruction_output: MultiClassifierOutput = MultiClassifierOutput(
            tokenset=state_tokenset,
            required_keys=list(state_map.keys()),
        )
        super().__init__(input=input, output=instruction_output, context=[], name="MultiClassifierInstruction")
        if not isinstance(self.output, MultiClassifierOutput):
            raise InstructionTypeError(f"Output must be an instance of MultiClassifierOutput. Got: {type(self.output)}")
        self._validate_input_snippets()

    # noinspection PyMethodOverriding
    def add_sample(self, input_snippets: List[Union[str | Snippet]], output_snippet: Snippet | str,
                   output_value: Union[int, float, List[Union[int, float]], None] = None,
                   final: FinalToken | None = None):
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
        self.output.validate_sample(snippet=output_snippet)
        self._assert_input_snippet_count(inputs=input_snippets)
        self._validate_snippets_match(inputs=input_snippets, response_snippet=output_snippet)
        self._validate_snippet_length(inputs=input_snippets, response_snippet=output_snippet)

        sample: Sample = self._create_sample(inputs=input_snippets, response_snippet=output_snippet,
                                             value=output_value, final=final)
        self.samples.append(sample)


    def _validate_snippets_match(self, inputs: List[Snippet], response_snippet: Snippet):
        """Validates that all snippets in the samples match their expected token sets."""
        all_snippets: List[Snippet] = inputs + [response_snippet]
        all_token_sets: List[TokenSet] = self.get_token_sets()

        for i in range(len(all_snippets)):
            self._validate_snippet_matches_set(snippet=all_snippets[i], expected_token_set=all_token_sets[i])

        if not isinstance(self.output, MultiClassifierOutput):
            raise InstructionTypeError(f"Output must be an instance of MultiClassifierOutput. Got: {type(self.output)}")

        # Validate output snippet set matches output token set
        self._validate_snippet_matches_set(snippet=response_snippet, expected_token_set=self.output.tokenset)