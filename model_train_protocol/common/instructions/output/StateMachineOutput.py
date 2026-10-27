import re
from typing import List

from model_train_protocol.errors import OutputError, OutputTypeError
from .BaseOutput import BaseOutput
from ...constants import NON_TOKEN
from ...tokens.FinalToken import FinalToken
from ...tokens.Token import Token
from ...tokens.TokenSet import TokenSet


def state_token_from_string(state: str) -> Token:
    """Creates a state Token from a string, replacing characters that are invalid in token values."""
    return Token(re.sub(r"\W", "_", state.strip()))


class StateMachineOutput(BaseOutput):
    """Defines the output of StateMachineInstructions. The model responds with one of a fixed set of state Tokens."""

    def __init__(self, tokenset: TokenSet, states: List[Token]):
        """
        Initializes a StateMachineOutput instance.

        :param tokenset: The TokenSet that prompts the model's response.
        :param states: Tokens the model can respond with.
        """
        super().__init__(tokenset=tokenset, final=NON_TOKEN)
        for state in states:
            if not isinstance(state, Token) or isinstance(state, FinalToken):
                raise OutputTypeError(f"States must be Token instances (not FinalTokens). Got: {type(state)}")
            if state.num or state.num_list:
                raise OutputError(f"State tokens cannot be numeric. Got: {state.value}")
        if len({state.value for state in states}) != len(states):
            raise OutputError("State tokens must be unique.")
        self.states: List[Token] = list(states)

    # noinspection PyMethodOverriding
    def validate_sample(self, state: Token):
        """
        Validates that the sample state is one of the defined state tokens.

        :param state: The state token to validate.
        """
        if not isinstance(state, Token):
            raise OutputTypeError(f"State must be a Token instance. Got: {type(state)}")
        if state not in self.states:
            raise OutputError(
                f"State '{state.value}' is not one of the defined states: {[s.value for s in self.states]}.")
