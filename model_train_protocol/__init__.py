"""
Model Train Protocol (MTP) - A Python package for creating custom Language Model training protocols.

MTP is an open-source protocol for training custom Language Models on Databiomes. 
MTP contains all the data that a model is trained on.
"""
from .common.instructions.input.InstructionInput import InstructionInput
from .common.instructions.input.StateMachineInput import StateMachineInput
from .common.tokens import Token, NumToken, NumListToken, FinalToken, Snippet, TokenSet, FinalNumToken
from .common.instructions.output import InstructionOutput, ExtendedResponse
from .common.instructions import Instruction, ExtendedInstruction
from .common.instructions.StateMachineInstruction import StateMachineInstruction
from .common.instructions.output.StateMachineOutput import StateMachineOutput
from .common.instructions.MultiClassifierInstruction import MultiClassifierInstruction
from .common.instructions.output.MultiClassifierOutput import MultiClassifierOutput
from .common.guardrails import Guardrail
# `Protocol` aliases the current protocol version. V1 (bloom 1.2.x, `state_machine`) stays
# importable from model_train_protocol.v1 for reading legacy bloom files.
from model_train_protocol.v2.protocol.protocol_v2 import ProtocolV2 as Protocol
from .errors import (
    MTPError,
    MTPValueError,
    MTPTypeError,
    MTPKeyError,
    InstructionInputError,
    GuardrailIndexError,
    DuplicateGuardrailError,
    ProtocolFileError,
    ProtocolFileLayerDepthError,
    TemplateFileError,
    GuardrailError,
    GuardrailTypeError,
    TokenError,
    TokenTypeError,
    TokenSetError,
    TokenSetTypeError,
    InstructionError,
    InstructionTypeError,
    OutputError,
    OutputTypeError,
    MultiClassifierError,
    ProtocolError,
    ProtocolTypeError,
    ProviderError,
    StateMachineError,
)

__all__ = [
    "Protocol",
    "Token",
    "FinalToken",
    "FinalNumToken",
    "NumToken",
    "NumListToken",
    "TokenSet",
    "Snippet",
    "Instruction",
    "InstructionInput",
    "StateMachineInput",
    "ExtendedInstruction",
    "InstructionOutput",
    "StateMachineInstruction",
    "StateMachineOutput",
    "MultiClassifierInstruction",
    "MultiClassifierOutput",
    "ExtendedResponse",
    "Guardrail",
    "MTPError",
    "MTPValueError",
    "MTPTypeError",
    "MTPKeyError",
    "InstructionInputError",
    "GuardrailIndexError",
    "DuplicateGuardrailError",
    "ProtocolFileError",
    "ProtocolFileLayerDepthError",
    "TemplateFileError",
    "GuardrailError",
    "GuardrailTypeError",
    "TokenError",
    "TokenTypeError",
    "TokenSetError",
    "TokenSetTypeError",
    "InstructionError",
    "InstructionTypeError",
    "OutputError",
    "OutputTypeError",
    "MultiClassifierError",
    "ProtocolError",
    "ProtocolTypeError",
    "ProviderError",
    "StateMachineError",
]
