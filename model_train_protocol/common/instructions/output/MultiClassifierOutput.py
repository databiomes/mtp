from model_train_protocol.errors import OutputTypeError
from .BaseOutput import BaseOutput
from ...constants import NON_TOKEN
from ...tokens.TokenSet import TokenSet, Snippet

import json


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

        # Validate all keys in the snippet are present in the required keys
        snippet_dict: dict[str, str] = json.loads(snippet.string)

        for key in self.required_keys:
            if key not in snippet_dict:
                raise OutputTypeError(
                    f"MultiClassifier Snippet must contain the key '{key}'. Got: {snippet_dict.keys()}")

        if len(snippet_dict) != len(self.required_keys):
            raise OutputTypeError(
                f"MultiClassifier Snippet must contain exactly {len(self.required_keys)} keys. Got: {len(snippet_dict)} keys.")

        # Validate that snippet.string is valid json
        try:
            json.loads(snippet.string)
        except json.JSONDecodeError:
            raise OutputTypeError(f"MultiClassifier Snippet string must be valid JSON. Got: {snippet.string}")
