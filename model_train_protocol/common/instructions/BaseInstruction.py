import abc
import warnings
from abc import ABC
from typing import List, Optional, Union

from .input.BaseInput import BaseInput
from .output.BaseOutput import BaseOutput
from ..constants import MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION, \
    RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE, \
    RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET, GENERAL_MINIMUM_INSTRUCTION_SAMPLES
from ..guardrails import Guardrail
from ..tokens.FinalToken import FinalToken
from ..tokens.Token import Token
from ..tokens.TokenSet import TokenSet, Snippet
from model_train_protocol.errors import InstructionError, InstructionTypeError


class Sample:
    """A Sample is a single example of input and output for an Instruction."""

    def __init__(self, input: List[str], output: str, prompt: Optional[str], numbers: List[List[int]],
                 number_lists: List[List[List[int]]],
                 result: FinalToken,
                 value: Union[int, float, None]):
        self.input: List[str] = input
        self.output: str = output
        self.prompt: Optional[str] = prompt
        self.numbers: List[List[int]] = numbers
        self.number_lists: List[List[List[int]]] = number_lists
        self.result: FinalToken = result
        self.value: Union[int, float, None] = value

    @property
    def strings(self) -> List[str]:
        """Returns all strings in the sample as a list."""
        return self.input + [self.output]

    def to_dict(self) -> dict:
        return {
            'strings': self.strings,
            'prompt': self.prompt,
            'numbers': self.numbers,
            'number_lists': self.number_lists,
            'result': self.result.value,  # We only need the value of the result token
            'value': self.value
        }

    def __repr__(self):
        """String representation of the Sample."""
        result_str = self.result.value
        if self.value is not None:
            result_str += f"{self.value}"
        return f"Sample(Context: {self.input}, Output: {self.output}, Result: {result_str})"


class BaseInstruction(ABC):
    """
    An Instruction is a set of tokens that show possible input combinations for a model.

    Samples must be added to the Instruction to provide context for the model.
    A minimum of 3 samples must be added to an Instruction.

    Example:
        context = TokenSet(
        context = [
                 [ Token("SentenceLength", num=True), Token("Greeting") ],
                 [ Token("CurtResponse") ],
                 [ Token("SentenceLength", num=True), Token("Goodbye") ],
                 ]
        response = TokenSet( Token("SentenceLength", num=True), Token("PoliteResponse") )
        final = Token("End")
        instruction = Instruction(context=context, response=response, final=final, name="example_instruction")
    """
    minimum_samples: int = GENERAL_MINIMUM_INSTRUCTION_SAMPLES

    def __init__(self, name: str, input: BaseInput, output: BaseOutput, context: List[str] | None = None):
        """
        Initializes the common attributes to all Instructions.
        
        :param name: Name for the Instruction. Must be unique across all instructions.
        :param input: BaseInput instance containing the input structure.
        :param output: BaseOutput instance containing the output structure.
        :param context: A list of strings providing background context for the instruction.
        """
        self.name: str = name
        self.input: BaseInput = input
        self.output: BaseOutput = output
        if context is None:
            context = []
        self.context: List[str] = context
        self.samples: List[Sample] = []
        self.samples: list[Sample] = []
        if not isinstance(input, BaseInput):
            raise InstructionTypeError("Context must be a sequence of TokenSet instances.")
        if not all(isinstance(ts, TokenSet) for ts in input.tokensets):
            raise InstructionTypeError("All items in context must be instances of TokenSet.")
        self._validate_context()

    def get_token_sets(self) -> List[TokenSet]:
        """Returns all tokens in the instruction as a list of tuples."""
        all_tokens_sets: List = []
        for token_set in self.input.tokensets:
            all_tokens_sets.append(token_set)
        all_tokens_sets.append(self.output.tokenset)
        return all_tokens_sets

    @abc.abstractmethod
    def add_sample(self):
        """Add a sample to the Instruction."""
        raise NotImplementedError("Subclasses must implement add_sample method.")

    @property
    def example_final_token(self) -> FinalToken:
        """Returns an example final token from the response."""
        if self.output.final:
            return self.output.final[0]
        return self.output.default_final

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

    def validate_instruction(self):
        """Validates the Instruction meets required Protocol standards."""
        self._validate_input_snippets()
        self._validate_minimum_samples()
        self._validate_context()
        for sample in self.samples:
            all_snippet_strings: List[str] = sample.strings
            self.___warn_on_recommended_max_chars(all_snippet_strings)

    def add_context(self, context: str):
        """Adds context to the Instruction."""
        self.context.append(context)

    @classmethod
    def _validate_snippet_length(cls, inputs: List[Snippet], response_snippet: Snippet):
        """Warns if any snippet in the samples exceeds the recommended max length"""
        all_snippets: List[Snippet] = inputs + [response_snippet]
        all_snippet_strings: List[str] = [snippet.string for snippet in all_snippets]
        cls.___warn_on_recommended_max_chars(all_snippet_strings)

    @classmethod
    def ___warn_on_recommended_max_chars(cls, snippet_strings: List[str]):
        """Warns if any snippet string exceeds the recommended max length"""
        for snippet_string in snippet_strings:
            if len(snippet_string) > RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET:
                warnings.warn(
                    f"Snippet length {len(snippet_string)} exceeds recommended maximum length of "
                    f"{RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET} characters for snippet: {snippet_string}",
                    UserWarning,
                    stacklevel=3,
                )

    def get_tokens(self) -> List[Token]:
        """Returns all tokens in the instruction as a flat list."""
        all_tokens: List[Token] = []
        for token_set in self.get_token_sets():
            all_tokens.extend(token_set.tokens)
        for sample in self.samples:
            all_tokens.append(sample.result)
        return all_tokens

    def serialize_samples(self) -> List[dict]:
        """Serializes the Instruction samples"""
        serialized_samples: List[dict] = []
        for sample in self.samples:
            serialized_samples.append(sample.to_dict())

        return serialized_samples

    def serialize_guardrails(self) -> List[Guardrail]:
        """Serialize the Instruction guardrails."""
        guardrails = []
        for index, guardrail in self.input.guardrails.items():
            guardrail_dict: dict = guardrail.to_dict()
            guardrail_dict['index'] = index
            guardrails.append(guardrail_dict)
        return guardrails

    def serialize_ppo(self) -> List[dict]:
        """Serialize the Instruction for PPO training."""
        # To be implemented when ppo introduced
        # ppo = []
        # ppo_strings = [sample['strings'] for sample in self.ppo]
        # ppo_prompts = [sample['prompt'] for sample in self.ppo]
        # ppo_numbers = [sample['number'] for sample in self.ppo]
        # ppo_results = [sample['result'].value for sample in self.ppo]
        # ppo_values = [sample['value'] for sample in self.ppo]
        # ppo_a_samples = [sample['a_sample'] for sample in self.ppo]
        # ppo_b_samples = [sample['b_sample'] for sample in self.ppo]
        # ppo_pref = [sample['pref'] for sample in self.ppo]
        # for s, p, n, r, v, a, b, pr in zip(ppo_strings, ppo_prompts, ppo_numbers, ppo_results, ppo_values,
        #                                    ppo_a_samples, ppo_b_samples, ppo_pref):
        #     ppo.append({'sample': s, 'prompt': p, 'number': n, 'result': r, 'value': v, 'a': a, 'b': b, 'pref': pr})
        # TODO: implement PPO training
        return []

    def serialize_memory_set(self) -> List[List[str]]:
        """Serialize the Instruction token memory set training."""
        memory_set = []
        for token_set in self.get_token_sets():
            token_strings = [t.value for t in token_set.tokens]
            memory_set.append(token_strings)
        return memory_set

    def _create_sample(self, inputs: List[Snippet], response_snippet: Snippet, final: FinalToken,
                       value: Union[int, float, List[Union[int, float]], None] = None) -> Sample:
        """Create a base sample dictionary without a prompt."""
        all_snippets: List[Snippet] = inputs + [response_snippet]

        # format sample
        numbers: List[List[int]] = []
        for snippet in all_snippets:
            numbers.append(snippet.numbers)

        number_lists: List[List[List[int]]] = []
        for snippet in all_snippets:
            number_lists.append(snippet.number_lists)

        return Sample(
            input=[snippet.string for snippet in inputs],
            output=response_snippet.string,
            prompt=None,
            numbers=numbers,
            number_lists=number_lists,
            result=final,
            value=value
        )

    def _validate_context(self):
        """Validates the total context lines and the length of each context line."""
        if len(self.context) > MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION:
            raise ValueError(f"Context exceeds maximum allowed lines of {MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION}. "
                             f"Current lines: {len(self.context)}")

        for i, line in enumerate(self.context):
            if len(line) > RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE:
                warnings.warn(
                    f"Context line {i} exceeds recommended maximum length of "
                    f"{RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE} characters. "
                    f"Current length: {len(line)}",
                    UserWarning,
                    stacklevel=3,
                )

    def _validate_input_snippets(self):
        """Validates that input snippets do not contain any final tokens."""
        for token_set in self.input.tokensets:
            for token in token_set:
                if isinstance(token, FinalToken):
                    raise InstructionError(
                        f"Context TokenSet cannot contain FinalToken instances. Found: {token}"
                    )

    def _validate_minimum_samples(self):
        """Validates that each instruction has at least 3 samples for each FinalToken"""
        if len(self.samples) < self.minimum_samples:
                raise InstructionError(
                    f"Instruction '{self.name}' has only {len(self.samples)} samples. "
                    f"Each instruction must have at least {self.minimum_samples} samples."
                )

        # Enforce there are 3 samples for each FinalToken in the response
        final_token_counts: dict[FinalToken, int] = {final_token: 0 for final_token in self.output.final}
        for sample in self.samples:
            final_token_counts[sample.result] += 1

        for final_token, count in final_token_counts.items():
            if count < 3:
                raise InstructionError(
                    f"Instruction '{self.name}' has only {count} samples for final token '{final_token.value}'. "
                    f"Each final token must have at least 3 samples."
                )

    def _enforce_input_snippets(self, inputs: List[Union[Snippet, str]]) -> List[Snippet]:
        """Converts regular strings to snippets if provided as a list of strings."""
        if len(inputs) != len(self.input.tokensets):
            raise InstructionError(
                f"Number of context snippets ({len(inputs)}) must match number of context token sets ({len(self.input.tokensets)}).")

        for i, snippet in enumerate(inputs):
            if isinstance(snippet, str):
                associated_tokenset: TokenSet = self.input.tokensets[i]
                try:
                    inputs[i] = associated_tokenset.create_snippet(string=snippet)
                except Exception as e:
                    raise InstructionError(
                        f"Failed to create Snippet from string '{snippet}' for TokenSet {associated_tokenset}.\n"
                        f"Create a Snippet from the tokenset and add associated information: {e}")

        return inputs

    def _enforce_response_snippet(self, snippet: Union[Snippet, str]) -> Snippet:
        """Converts a regular string to a snippet if provided as a string."""
        if isinstance(snippet, str):
            associated_tokenset: TokenSet = self.output.tokenset
            try:
                snippet = associated_tokenset.create_snippet(string=snippet)
            except Exception as e:
                raise InstructionError(
                    f"Failed to create Snippet from string '{snippet}' for TokenSet {associated_tokenset}.\n"
                    f"Create a Snippet from the tokenset and add associated information: {e}")
        return snippet

    @classmethod
    def _validate_snippet_matches_set(cls, snippet: Snippet, expected_token_set: TokenSet):
        """Validates that the snippet matches the expected token set."""
        if snippet.token_set_key != expected_token_set.key:
            raise InstructionError(
                f"Snippet {snippet} does not match expected token set {expected_token_set}."
            )

    def _assert_input_snippet_count(self, inputs: List[Snippet]):
        """Assert the number of input snippets matches the number of context token sets."""
        if len(inputs) != len(self.input.tokensets):
            raise InstructionError(
                f"Number of context snippets ({len(inputs)}) must match number of context token sets ({len(self.input.tokensets)}).")

    def _assign_final_token(self, final: Optional[FinalToken]) -> FinalToken:
        """
        Validate the final token if provided.

        The following behaviour is observed:

        If a final token is provided directly, always use it.

        If no final token is provided in the sample, and no final token is defined in the response, use the default <NON> final token.

        If no final token is provided in the sample, and only one final token is defined in the response, use that final token.

        If no final token is provided in the sample, and multiple final tokens are defined in the response, raise an error requiring clarification.
        """
        if final is not None and not isinstance(final, FinalToken):
            raise InstructionTypeError("Final must be a FinalToken instance or None.")

        if final is not None:  # Use the specified final token if provided
            return final

        if len(self.output.final) == 0:  # Return default final if no finals are defined
            return self.output.default_final

        if len(self.output.final) == 1:  # If we only have one final token, use it
            return self.output.final[0]

        raise InstructionError(
            "Multiple final tokens are allowed in the Output. Specify which final token to use for this sample.")

    def __str__(self) -> str:
        """String representation of the Instruction."""
        tokens_str: str = ', '.join(
            [''.join([token.key for token in token_set.tokens]) for token_set in self.get_token_sets()])
        samples_str: str = ',\n'.join([str(sample) for sample in self.samples])
        return f"Token Set(Tokens: {tokens_str}, Result: {self.output.default_final.key}, Samples:\n{samples_str})"

    def __hash__(self) -> int:
        """Hash based on the token sets of the Instruction. Instructions with the same TokenSets in the same order
        will have the same hash."""
        return hash(str(self.get_token_sets()))

    def __eq__(self, other) -> bool:
        """
        Defines equality based on the attributes of the Instruction.
        Returns True if the other object is an Instruction and its attributes match this Instruction's attributes.
        Includes the 'name' field in comparison.
        """
        if not isinstance(other, BaseInstruction):
            return False

        attrs_to_compare = ['name', 'context', 'response', 'final', 'samples']
        for attr in attrs_to_compare:
            try:
                self_val = getattr(self, attr, None)
                other_val = getattr(other, attr, None)
                if self_val != other_val:
                    return False
            except AttributeError:
                return False

        return True

    def __dict__(self) -> dict:
        """Dictionary representation of the Instruction."""
        return self.to_dict()

    def to_dict(self) -> dict:
        """Convert the Instruction to a dictionary representation."""
        return {
            'name': self.name,
            'tokens': [[token.to_dict() for token in token_set.tokens] for token_set in self.get_token_sets()],
            'result': self.output.default_final.to_dict() if self.output.default_final else None,
            'samples': self.samples
        }
