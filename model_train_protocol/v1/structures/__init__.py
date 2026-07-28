"""
Pydantic structures describing the bloom 1.2.x / template 1.2.x file formats.

These are pinned copies of the models that model_train_protocol_schemas shipped while the
1.2.x formats were current. The schemas package only ever exposes the *latest* structures, so
a versioned protocol that imported them directly would silently start validating against a
newer format (see the 1.2.x `state_machine` -> 2.x `model_type` change). V1 owns its own
definitions instead; the schemas package remains the source of truth for the JSON Schema
documents themselves.
"""

from model_train_protocol.v1.structures.protocol import (
    Guardrail,
    Instruction,
    InstructionSet,
    Protocol,
    Sample,
    TokenInfo,
)
from model_train_protocol.v1.structures.template import (
    ExampleUsage,
    InstructionDefinition,
    Template,
    Tokens,
)

__all__ = [
    "Guardrail",
    "Instruction",
    "InstructionSet",
    "Protocol",
    "Sample",
    "TokenInfo",
    "ExampleUsage",
    "InstructionDefinition",
    "Template",
    "Tokens",
]
