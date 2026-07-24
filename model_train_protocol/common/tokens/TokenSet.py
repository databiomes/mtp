from dataclasses import dataclass
from typing import List, Sequence, Collection, Union

from . import NumListToken
from .NumToken import NumToken
from .Token import Token
from model_train_protocol.errors import TokenSetError, TokenSetTypeError


@dataclass
class Snippet:
    string: str
    token_set_key: str
    numbers: List[Union[int, float]]
    number_lists: List[List[Union[int, float]]]

class TokenSet:
    """A set of Tokens representing a combination of input types."""

    def __init__(self, tokens: Union[Sequence[Token], Token]):
        """Initializes a TokenSet instance."""
        if isinstance(tokens, Token):
            tokens = [tokens]
        self.tokens: Sequence[Token] = tokens
        self.key: str = ''.join(token.value for token in
                                tokens)  # Note this key is based on the value of the tokens and not the keys of the tokens
        self._num_tokens: List[NumToken] = [token for token in tokens if isinstance(token, NumToken)]
        self._num_list_tokens: List[NumListToken] = [token for token in tokens if isinstance(token, NumListToken)]

    def _validate_num_tokens(self, numbers: Collection[Union[int, float]]):
        """Validates the numbers against the TokenSet requirements."""
        required_numtoken_numbers: int = sum(
            token.num for token in self.tokens if isinstance(token, NumToken))  # Count of NumToken
        if len(numbers) != required_numtoken_numbers:
            raise TokenSetError(
                f"{self} requires {required_numtoken_numbers} numbers but {len(numbers or [])} were provided.")
        for (i, number) in enumerate(numbers):
            corresponding_token = self._num_tokens[i]
            if not (corresponding_token.min_value <= number <= corresponding_token.max_value):
                raise TokenSetError(
                    f"Number at index {i} with value {number} is out of bounds for token {corresponding_token}. Must be between {corresponding_token.min_value} and {corresponding_token.max_value}.")

    def _validate_numlist_tokens(self, number_lists: Collection[Collection[Union[int, float]]]):
        """Validates the number lists against the TokenSet requirements."""
        required_numlists: List[int] = [
            token.num_list for token in self.tokens if isinstance(token, NumListToken)]
        if len(number_lists) != len(required_numlists):
            raise TokenSetError(
                f"{self} requires {len(required_numlists)} number lists but {len(number_lists or [])} lists were provided.")
        if len(self._num_list_tokens) != len(required_numlists):
            raise TokenSetError(
                f"Number of NumListToken instances ({len(self._num_list_tokens)}) does not match required number lists ({len(required_numlists)}).")
        for (i, required_length) in enumerate(required_numlists):
            if len(number_lists[i]) != required_length:
                raise TokenSetError(
                    f"Number list at index {i} must be of length {required_length} but is of length {len(number_lists[i])}.")
        for (i, number_list) in enumerate(number_lists):
            corresponding_token = self._num_list_tokens[i]
            for number in number_list:
                if not (corresponding_token.min_value <= number <= corresponding_token.max_value):
                    raise TokenSetError(
                        f"Number {number} in number list at index {i} is out of bounds for token {corresponding_token}. Must be between {corresponding_token.min_value} and {corresponding_token.max_value}.")

    def create_snippet(self, string: str,
                       numbers: Union[Collection[Union[int, float]], int, float, None] = None,
                       number_lists: Union[Collection[Union[int, float, Collection[Union[int, float]]]], None] = None) -> Snippet:
        """Create a snippet for the TokenSet"""
        if not isinstance(string, str):
            raise TokenSetTypeError("String must be of type str.")

        if numbers is None:
            numbers = []
        elif isinstance(numbers, int):
            numbers = [numbers]
        elif isinstance(numbers, Collection):
            numbers = list(numbers)
        else:
            raise TokenSetTypeError("Numbers must be an int, an Collection of ints, or None.")

        if number_lists is None:
            number_lists = []
        # if number lists is a single list of numbers, wrap it in another list
        elif isinstance(number_lists, Collection) and all(isinstance(nl, (float, int)) for nl in number_lists):
            number_lists = [number_lists]
        elif isinstance(number_lists, Collection) and all(isinstance(nl, Collection) for nl in number_lists):
            pass
        else:
            raise TokenSetTypeError(
                "Number lists must be an Collection of numbers or Collection of Collections or None."
            )

        self._validate_num_tokens(numbers=numbers)
        self._validate_numlist_tokens(number_lists=number_lists)

        # Combine numbers and number_lists into single input for Snippet
        numbers_index = 0
        number_lists_index = 0
        # final_numbers: list[int | float] = []
        # final_number_lists: list[Collection[int | float]] = []
        # combined_numbers: list[int | float | Collection[int | float]] = []  # Combined list of numbers and number lists
        # for index, token in enumerate(self.tokens):
        #     if not isinstance(token, NumToken):
        #         continue
        #
        #     if token.num == 1:
        #         combined_numbers.append(numbers[numbers_index])
        #         numbers_index += 1
        #     elif token.num > 1:
        #         combined_numbers.append(number_lists[number_lists_index])
        #         number_lists_index += 1
        #
        # for index, token in enumerate(self.tokens):
        #
        #     if not isinstance(token, NumListToken):
        #         continue
        #
        #     number_lists.append(numbers[number_lists_index])

        return Snippet(string=string, numbers=numbers, number_lists=number_lists, token_set_key=self.key)

    def get_token_key_set(self) -> str:
        """Returns a string representing the combined token keys of the individual Tokens in the TokenSet."""
        token_key_set = ''
        for token in self.tokens:
            token_key_set += token.key
        return token_key_set

    @property
    def has_num_tokens(self) -> bool:
        """Returns True if the TokenSet contains any NumTokens."""
        return len(self._num_tokens) > 0

    @property
    def has_num_list_tokens(self) -> bool:
        """Returns True if the TokenSet contains any NumListTokens."""
        return len(self._num_list_tokens) > 0

    def __eq__(self, other):
        """Equality comparison for TokenSet."""
        if not isinstance(other, TokenSet):
            return False
        return self.key == other.key and all(st == ot for (st, ot) in zip(self.tokens, other.tokens))

    def __hash__(self):
        """Hash based on the string representation of the TokenSet."""
        return hash(str(self))

    def __repr__(self):
        """String representation of the TokenSet."""
        return f"TokenSet([{self.key}])"

    def __iter__(self):
        """Iterator over the tokens in the TokenSet."""
        return iter(self.tokens)
