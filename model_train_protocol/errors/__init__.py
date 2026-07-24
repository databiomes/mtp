"""Custom errors for the model_train_protocol package."""

from .base import (
    MTPError,
    MTPKeyError,
    MTPTypeError,
    MTPValueError,
)
from .conversion import ConversionError
from .guardrails import GuardrailError, GuardrailTypeError
from .instruction_input import DuplicateGuardrailError, GuardrailIndexError, InstructionInputError
from .instructions import InstructionError, InstructionTypeError
from .outputs import OutputError, OutputTypeError
from .multi_classifier import MultiClassifierError
from .protocol import ProtocolError, ProtocolTypeError
from .protocol_file import ProtocolFileError, ProtocolFileLayerDepthError
from .providers import ProviderError
from .template_file import TemplateFileError
from .tokens import TokenError, TokenSetError, TokenSetTypeError, TokenTypeError
from .state_machine import StateMachineError

__all__ = [
    "MTPError",
    "MTPValueError",
    "MTPTypeError",
    "MTPKeyError",
    "ConversionError",
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
    "StateMachineError"
]

