"""
Pydantic models for Template JSON structure.
"""

from typing import Dict, List, Literal

from pydantic import BaseModel


class Tokens(BaseModel):
    """Model for tokens in template (input and output mappings)."""
    input: Dict[str, str]  # Maps token values to token keys with template representations
    output: Dict[str, str]  # Maps token values to token keys


class InstructionDefinition(BaseModel):
    """Model for a single instruction definition in the template."""
    type: Literal["basic", "extended", "state_machine"]  # Instruction type
    input: List[str]  # List of input strings with token keys and placeholders
    output: List[str]  # List of output strings with token keys and placeholders


def _example_usage_json_schema_extra(schema: dict, model_class) -> None:
    """Customize JSON schema to specify that additionalProperties must be strings."""
    schema["additionalProperties"] = {
        "type": "string",
        "description": "Example usage string (e.g., instruction_input, valid_model_output, guardrail_model_output)"
    }


class ExampleUsage(BaseModel):
    """Model for example usage in template."""
    
    instruction_input: str
    valid_model_output: str
    guardrail_model_output: str


class Template(BaseModel):
    """Main model for MTP Template JSON structure."""
    encrypt: bool
    state_machine: bool
    inputs: int
    tokens: Tokens
    instructions: Dict[str, InstructionDefinition]  # Instruction name -> instruction definition
    states: List[str]
    example_usage: ExampleUsage
