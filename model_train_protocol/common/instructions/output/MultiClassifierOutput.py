from model_train_protocol.errors import MultiClassifierError, OutputTypeError
from .BaseOutput import BaseOutput
from ...constants import NON_TOKEN
from ...tokens.TokenSet import TokenSet, Snippet

import ast
import json


def parse_multi_classifier_output(string: str) -> dict:
    """
    Parses a MultiClassifier output string into a dictionary.

    The canonical on-disk form uses single quotes (e.g. "{'emotion': 'confused'}") so that the snippet can be embedded
    inside a double-quoted JSON string, but callers may also supply strict JSON (e.g. json.dumps output). Both forms are
    accepted here, as are values that themselves contain quote characters.

    :param string: The output string to parse.
    :raises MultiClassifierError: If the string does not represent a mapping of keys to values.
    """
    parsed: object = None
    try:
        parsed = json.loads(string)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(string)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            raise MultiClassifierError(
                f"MultiClassifier output must be a valid JSON object or Python dict literal. Got: {string}")

    if not isinstance(parsed, dict):
        raise MultiClassifierError(f"MultiClassifier output must be an object mapping keys to values. Got: {string}")

    for key in parsed:
        if not isinstance(key, str):
            raise MultiClassifierError(f"MultiClassifier output keys must be strings. Got: {key!r} in {string}")
    return parsed


def format_multi_classifier_output(output: dict) -> str:
    """
    Renders a MultiClassifier output dictionary in the canonical single-quoted form, e.g.
    "{'emotion': 'confused', 'intent': 'question'}".

    Quotes and backslashes inside keys and values are escaped rather than replaced, so a value such as "it's" survives
    the round trip (it is rendered with double quotes for that entry, which parse_multi_classifier_output accepts).

    :param output: The output dictionary to render.
    """
    return "{" + ", ".join(f"{key!r}: {value!r}" for key, value in output.items()) + "}"


class MultiClassifierOutput(BaseOutput):
    """Defines the output of MultiClassifierOutput."""

    def __init__(self, tokenset: TokenSet, required_keys: list[str]):
        """
        Initializes a MultiClassifierOutput instance.

        :param tokenset: The TokenSet associated with the model's response.
        :param required_keys: A list of required keys that must be present in the response.
        """
        super().__init__(tokenset=tokenset, final=NON_TOKEN)
        self.required_keys = required_keys

    # noinspection PyMethodOverriding
    def validate_sample(self, snippet: Snippet):
        """
        Validates the snippet against the response definition.

        :param snippet: The snippet to validate.
        """
        if not isinstance(snippet, Snippet):
            raise OutputTypeError(f"Snippet must be an instance of Snippet. Got: {type(snippet)}")

        # Validate that snippet.string represents a classification object
        snippet_dict: dict[str, str] = parse_multi_classifier_output(snippet.string)

        # Validate all required keys are present in the snippet
        for key in self.required_keys:
            if key not in snippet_dict:
                raise MultiClassifierError(
                    f"MultiClassifier Snippet must contain the key '{key}'. Got: {snippet_dict.keys()}")

        if len(snippet_dict) != len(self.required_keys):
            raise MultiClassifierError(
                f"MultiClassifier Snippet must contain exactly {len(self.required_keys)} keys. Got: {len(snippet_dict)} keys.")
