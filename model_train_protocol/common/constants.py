from enum import Enum

from model_train_protocol.common.tokens import SpecialToken, Token, NumToken, FinalToken, FinalNumToken, NumListToken
from model_train_protocol.common.tokens.SpecialFinalToken import SpecialFinalToken

NON_TOKEN: SpecialFinalToken = SpecialFinalToken(value="<NON>", key="<NON>", special="none")
BOS_TOKEN: SpecialToken = SpecialToken(value="<BOS>", key="<BOS>", special="start")
EOS_TOKEN: SpecialFinalToken = SpecialFinalToken(value="<EOS>", key="<EOS>", special="end")
RUN_TOKEN: SpecialToken = SpecialToken(value="<RUN>", key="<RUN>", special="infer")
PAD_TOKEN: SpecialToken = SpecialToken(value="<PAD>", key="<PAD>", special="pad")
UNK_TOKEN: SpecialToken = SpecialToken(value="<UNK>", key="<UNK>", special="unknown")

MINIMUM_TOTAL_CONTEXT_LINES = 10
PER_FINAL_TOKEN_SAMPLE_MINIMUM = 3

MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION: int = 100_000 # Arbitrary large number to allow as many context lines as needed

# String lengths are unbounded as of bloom 2.0.1. These are recommendations only: exceeding them
# warns rather than raising.
RECOMMENDED_MAXIMUM_CHARACTERS_PER_MODEL_CONTEXT_LINE: int = 3000
RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE: int = 3000
RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET: int = 3000

GENERAL_MINIMUM_INSTRUCTION_SAMPLES: int = 3
STATE_MACHINE_MINIMUM_INSTRUCTION_SAMPLES: int = 10

MIN_SAMPLES_PER_GUARDRAIL: int = 3

TokenTypeEnum: dict = {
    "Token": Token,
    "SpecialToken": SpecialToken,
    "SpecialFinalToken": SpecialFinalToken,
    "NumToken": NumToken,
    "FinalToken": FinalToken,
    "FinalNumToken": FinalNumToken,
    "NumListToken": NumListToken
}


class ModelType(str, Enum):
    """Enumeration for model types."""
    GENERATIVE = "generative"
    STATE_MACHINE = "state_machine"
    MULTI_CLASSIFICATION = "multi_classifier"
