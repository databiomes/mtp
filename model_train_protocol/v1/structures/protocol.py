"""
Pydantic models for Protocol JSON structure.
"""

from typing import Annotated, List, Dict, Any, Optional, Union, Literal

from pydantic import BaseModel, ConfigDict, Field

from model_train_protocol_schemas.constants import (
    MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE,
    MAXIMUM_CHARACTERS_PER_MODEL_CONTEXT_LINE,
    MAXIMUM_CHARACTERS_PER_SNIPPET,
    MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION,
    MIN_SAMPLES_PER_GUARDRAIL,
    PER_FINAL_TOKEN_SAMPLE_MINIMUM,
)

# String types with length limits for JSON Schema (no root json_schema_extra to avoid overwriting schema).
ContextLine = Annotated[str, Field(max_length=MAXIMUM_CHARACTERS_PER_MODEL_CONTEXT_LINE)]
InstructionContextLine = Annotated[str, Field(max_length=MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE)]
SnippetStr = Annotated[str, Field(max_length=MAXIMUM_CHARACTERS_PER_SNIPPET)]

# Token type values emitted by Token.__class__.__name__
TokenTypeLiteral = Literal[
    "Token",
    "SpecialToken",
    "SpecialFinalToken",
    "NumToken",
    "FinalToken",
    "FinalNumToken",
    "NumListToken",
]


class TokenInfo(BaseModel):
    """Model for individual token information."""
    key: str = Field(..., min_length=1, description="Token key; alphanumeric, underscore, and emoji allowed.")
    num: bool
    num_list: int = Field(..., ge=0, description="Number of list dimensions for num_list tokens.")
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    length: Optional[int] = None
    desc: Optional[str] = None
    special: Optional[str] = None
    type: TokenTypeLiteral


class Sample(BaseModel):
    """Model for instruction samples."""
    strings: List[SnippetStr] = Field(
        ...,
        description="Snippet strings per input/response.",
        min_length=1,
    )
    prompt: Optional[Union[str]] = None
    numbers: Union[int, List[int], List[List[int]]]
    number_lists: Union[int, List[int], List[List[int]], List[List[List[int]]]]
    result: str = Field(..., min_length=1, description="Final token value for this sample.")
    value: Optional[Union[str, int, float, List[int], List[float]]] = None  # Can be string, int, float, or list


class Guardrail(BaseModel):
    """Model for guardrails configuration."""
    index: int = Field(..., ge=0)
    good_prompt: str = Field(..., min_length=1)
    bad_prompt: str = Field(..., min_length=1)
    bad_output: str = Field(..., min_length=1)
    bad_examples: List[str] = Field(
        default_factory=lambda: [],
        min_length=MIN_SAMPLES_PER_GUARDRAIL,
        description=f"At least {MIN_SAMPLES_PER_GUARDRAIL} bad examples per guardrail.",
    )


class InstructionSet(BaseModel):
    """Model for instruction sets."""
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Name of the instruction set."
    )
    guardrails: List[Guardrail] = Field(default_factory=list)
    context: List[InstructionContextLine] = Field(
        default_factory=list,
        max_length=MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION,
        description="Instruction context lines.",
    )
    set: List[List[SnippetStr]] = Field(..., description="Token set rows.")
    samples: List[Sample] = Field(
        ...,
        min_length=PER_FINAL_TOKEN_SAMPLE_MINIMUM,
        description="At least three samples per instruction set.",
    )
    ppo: List[Dict[str, Any]] = Field(default_factory=list)


class Instruction(BaseModel):
    """Model for instruction configuration."""
    memory: int = Field(..., ge=1, description="Input lines plus response (e.g. inputs + 1).")
    sets: List[InstructionSet] = Field(..., min_length=1)


def _number_model_json_schema_extra(schema: dict, model_class) -> None:
    """Customize JSON schema to specify that additionalProperties must be strings."""
    schema["additionalProperties"] = {
        "type": "string",
        "description": "Number rule value (must be a string)"
    }


class Protocol(BaseModel):
    """Main model for MTP Protocol JSON structure."""
    name: str = Field(..., min_length=1)
    inputs: int = Field(...)
    state_machine: bool
    encrypted: bool
    valid: bool
    context: List[ContextLine] = Field(
        default_factory=list,
        description="Protocol-level context lines.",
    )
    tokens: Dict[str, TokenInfo]
    special_tokens: List[str] = Field(default_factory=list)
    instruction: Instruction

    model_config = ConfigDict(extra="allow")
