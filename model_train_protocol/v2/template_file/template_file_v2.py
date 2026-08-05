import random
from dataclasses import dataclass
from enum import Enum
from typing import Union, List

from model_train_protocol import Instruction, ExtendedInstruction, StateMachineInstruction, MultiClassifierInstruction
from model_train_protocol.common.constants import BOS_TOKEN, RUN_TOKEN, EOS_TOKEN, UNK_TOKEN, NON_TOKEN, ModelType
from model_train_protocol.common.instructions import BaseInstruction
from model_train_protocol.common.instructions.BaseInstruction import Sample
from model_train_protocol_schemas.structures.template import (
    Template as TemplateModel,
    Tokens as TokensModel,
    InstructionDefinition,
    ExampleUsage,
)
from model_train_protocol_schemas.utils import get_template_schema_url

from model_train_protocol.common.tokens import FinalToken
from model_train_protocol.common.tokens import NumToken, NumListToken
from model_train_protocol.errors import TemplateFileError


class InstructionTypeEnum(Enum):
    """Enumeration for instruction types in the template."""

    BASIC = "basic"
    EXTENDED = "extended"
    STATE_MACHINE = "state_machine"
    MULTI_CLASSIFICATION_MACHINE = "multi_classification_machine"

    @classmethod
    def get_instruction_type_by_class(cls, instruction: BaseInstruction) -> 'InstructionTypeEnum':
        """Returns the InstructionTypeEnum corresponding to the given instruction class."""
        if isinstance(instruction, Instruction):
            return cls.BASIC
        elif isinstance(instruction, ExtendedInstruction):
            return cls.EXTENDED
        elif isinstance(instruction, StateMachineInstruction):
            return cls.STATE_MACHINE
        elif isinstance(instruction, MultiClassifierInstruction):
            return cls.MULTI_CLASSIFICATION_MACHINE
        else:
            raise TemplateFileError("Unknown instruction type.")


class TemplateFileV2:
    """Manages the model.json file for model training protocols."""

    @dataclass
    class ExampleUsage:
        """Stores example usages of the template."""

        input: str
        output: str

    class Tokens:
        """Represents all tokens used in the template."""

        def __init__(self):
            self.instructions_list: list[BaseInstruction] = []

        def add_tokens_from_instructions(self, instructions: list[BaseInstruction]):
            """Stores instruction data for later token extraction."""

            self.instructions_list = instructions

        def to_json(self) -> dict[str, dict[str, str]]:
            """Extracts tokens from stored instructions and converts to JSON-serializable dictionary."""

            input_token_mapping: dict[str, str] = {}
            output_token_mapping: dict[str, str] = {}

            for instruction in self.instructions_list:
                for token_set in instruction.get_token_sets():
                    token_value = "".join([t.value for t in token_set])
                    token_key = "".join([
                        t.key + t.template_representation for t in token_set
                    ])

                    # Check if any token in the token_set is a FinalToken
                    has_final_token = any(isinstance(t, FinalToken) for t in token_set)

                    if has_final_token:
                        output_token_mapping[token_value] = token_key
                    else:
                        input_token_mapping[token_value] = token_key

                for sample in instruction.samples:
                    output_token_mapping[sample.result.value] = sample.result.key

                # Add UNK token if Guardrails:
                if instruction.has_guardrails:
                    output_token_mapping[UNK_TOKEN.value] = UNK_TOKEN.key

            return {
                "input": dict(sorted(input_token_mapping.items())),
                "output": dict(sorted(output_token_mapping.items()))
            }

    class Instructions:
        """Represents the instruction set of the template."""

        def __init__(self):
            self.instructions_list: list[BaseInstruction] = []

        def add_inputs_from_instructions(self, instructions: list[BaseInstruction]):
            """Stores instruction data for later JSON conversion."""

            self.instructions_list = instructions

        def to_json(self):
            """Converts stored instructions to JSON-serializable dictionary."""

            instructions_dict: dict[str, dict] = {}

            for instruction in self.instructions_list:
                input_list: list[str] = [BOS_TOKEN.key + '\n']
                for idx, token_set in enumerate(instruction.get_token_sets()):
                    token_key = "".join([
                        t.key + t.template_representation for t in token_set
                    ])

                    is_last_context = idx == len(instruction.get_token_sets()) - 1
                    is_extended_instruction_extra_string = isinstance(instruction,
                                                                      ExtendedInstruction) and is_last_context

                    if is_extended_instruction_extra_string:
                        token_key += "<string>\n"

                    token_key += "\n"

                    if not is_last_context:
                        token_key += "<string>\n"

                    input_list.append(token_key)

                input_list.append(RUN_TOKEN.key)

                output_strs: list[str] = [
                    "<string>\n" + sample.result.key + "\n" + EOS_TOKEN.key
                    for sample in instruction.samples
                ]

                # add guardrail outputs if guardrails present
                if instruction.has_guardrails:
                    outputs: list[str] = [token.key for token in instruction.output.final]
                    if outputs is None:
                        outputs: list[str] = [NON_TOKEN.key + '_']
                    # Edge case for CSV conversion - if NON TOKEN + UNK TOKEN we need to add underscore to NON
                    if outputs == [NON_TOKEN.key] and len(outputs) == 1:
                        output_strs.append(f"<string>\n{NON_TOKEN.key}_{UNK_TOKEN.key}_\n{EOS_TOKEN.key}")
                    else:
                        for output in outputs:
                            output_strs.append(f"<string>\n{output}{UNK_TOKEN.key}_\n{EOS_TOKEN.key}")

                instructions_dict[instruction.name] = {
                    "type": InstructionTypeEnum.get_instruction_type_by_class(instruction).value,
                    "input": input_list,
                    "output": list(set(output_strs))
                }

            return instructions_dict

    def __init__(self, inputs: int, instructions: list[BaseInstruction], encrypt: bool, has_guardrails: bool,
                 model_type: ModelType):
        """Initializes the template"""

        self.tokens: TemplateFileV2.Tokens = TemplateFileV2.Tokens()
        self.instructions: TemplateFileV2.Instructions = TemplateFileV2.Instructions()
        self.inputs: int = inputs
        self.instructions_list: list[BaseInstruction] = instructions
        self.encrypt: bool = encrypt
        self.model_type: ModelType = model_type
        self.has_guardrails: bool = has_guardrails
        self._add_io_from_instructions()

    def _add_io_from_instructions(self):
        """Adds input and output sequences from the instructions."""

        self.tokens.add_tokens_from_instructions(self.instructions_list)
        self.instructions.add_inputs_from_instructions(self.instructions_list)

    @classmethod
    def _format_token_set_with_sample(cls, token_set, sample_string: str, is_extended_last: bool = False) -> str:
        """Formats a token set with actual sample data, including numbers and number lists.
        
        :param token_set: The TokenSet to format
        :param sample_string: The actual sample string to use
        :param is_extended_last: If True, format for extended instruction's last token set (no newline before string)
        """
        token_keys = "".join([token.key for token in token_set])

        # Add template representation for NumTokens and NumListTokens to the token key
        for token in token_set:
            if isinstance(token, NumToken):
                example_number: int = random.randint(token.min_value, token.max_value)
                token_keys += str(example_number)
            elif isinstance(token, NumListToken):
                example_list: list[Union[int, float]] = [
                    random.randint(token.min_value, token.max_value)
                    for _ in range(token.length)
                ]
                token_keys += str(example_list)

        if is_extended_last:
            # For extended instruction's last token set: token_key<string>\n (no newline before string)
            formatted_string = token_keys + sample_string + "\n"
        else:
            # For regular format: token_key\n<string>\n
            formatted_string = token_keys + "\n"
            formatted_string += sample_string + "\n"

        return formatted_string

    @classmethod
    def _create_sample_model_output(cls, instruction: BaseInstruction, sample: 'Sample') -> str:
        """Creates a sample model output string for a given instruction using actual sample data."""

        sample_output = sample.output + "\n"
        sample_output += instruction.example_final_token.key + "\n"
        sample_output += EOS_TOKEN.key
        return sample_output

    def _select_example_instructions(self) -> tuple[BaseInstruction | None, BaseInstruction | None]:
        """
        Selects one basic instruction and one extended instruction with samples for example creation.

        Puts a preference on instructions that include numeric tokens.
        """

        basic_instruction: BaseInstruction | None = None
        extended_instruction: BaseInstruction | None = None

        numeric_instruction_exists: bool = any(
            any(isinstance(t, (NumToken, NumListToken)) for ts in instr.get_token_sets() for t in ts)
            for instr in self.instructions.instructions_list if isinstance(instr, Instruction)
        )
        numeric_extended_instruction_exists: bool = any(
            any(isinstance(t, (NumToken, NumListToken)) for ts in instr.get_token_sets() for t in ts)
            for instr in self.instructions.instructions_list if isinstance(instr, ExtendedInstruction)
        )

        for instr in self.instructions.instructions_list:
            if isinstance(instr, Instruction) and instr.samples:
                if basic_instruction is None:
                    if numeric_instruction_exists:
                        if any(isinstance(t, (NumToken, NumListToken)) for ts in instr.get_token_sets() for t in ts):
                            basic_instruction = instr
                    else:
                        basic_instruction = instr
            elif isinstance(instr, ExtendedInstruction) and instr.samples:
                if extended_instruction is None:
                    if numeric_extended_instruction_exists:
                        if any(isinstance(t, (NumToken, NumListToken)) for ts in instr.get_token_sets() for t in ts):
                            extended_instruction = instr
                    else:
                        extended_instruction = instr
            elif isinstance(instr, StateMachineInstruction) and instr.samples:
                if basic_instruction is None:
                    basic_instruction = instr

            if basic_instruction and extended_instruction:
                break

        if basic_instruction is None and extended_instruction is None:
            raise TemplateFileError(
                "Could not select an instruction to build the template's example usage from."
                f"{sorted({type(instr).__name__ for instr in self.instructions.instructions_list})}."
            )

        return basic_instruction, extended_instruction

    def _create_examples(self) -> dict[str, str]:
        """Creates example usages of the template using actual sample data from instructions."""

        examples: dict[str, str] = dict()
        instruction, extended_instruction = self._select_example_instructions()

        if instruction and instruction.samples:
            # Use the first sample for the example
            sample = instruction.samples[0]

            instruction_input = BOS_TOKEN.key + "\n"

            # Map input strings to input token sets
            for idx, token_set in enumerate(instruction.input.tokensets):
                if idx < len(sample.input):
                    sample_string = sample.input[idx]
                    instruction_input += self._format_token_set_with_sample(token_set, sample_string)

            instruction_input += instruction.last_tokenset.key + "\n"

            instruction_input += RUN_TOKEN.key + "\n"
            examples["instruction_input"] = instruction_input

        # if extended_instruction and extended_instruction.samples:
        #     # Use the first sample for the example
        #     sample = extended_instruction.samples[0]
        #
        #     extended_instruction_input = BOS_TOKEN.key + "\n"
        #
        #     # Map input strings to input token sets
        #     for idx, token_set in enumerate(extended_instruction.input.tokensets):
        #         if idx < len(sample.input):
        #             sample_string = sample.input[idx]
        #             extended_instruction_input += self._format_token_set_with_sample(token_set, sample_string)
        #
        #     last_tokenset = extended_instruction.last_tokenset  # This is the last TokenSet in the original input
        #     prompt_string = sample.prompt if sample.prompt else ""
        #     extended_instruction_input += self._format_token_set_with_sample(last_tokenset, prompt_string, is_extended_last=True)
        #
        #     extended_instruction_input += RUN_TOKEN.key + "\n"
        #     examples["extended_instruction_input"] = extended_instruction_input

        first_instruction: BaseInstruction = instruction or extended_instruction
        if first_instruction and first_instruction.samples:
            sample = first_instruction.samples[0]
            examples["valid_model_output"] = self._create_sample_model_output(first_instruction, sample)

        if self.has_guardrails:
            guardrailed_instruction: BaseInstruction = next(
                instruction for instruction in self.instructions_list if instruction.has_guardrails)
            guardrail_full_output: str = f"{list(guardrailed_instruction.input.guardrails.values()).pop().bad_output}\n{guardrailed_instruction.example_final_token.key}{UNK_TOKEN.key}_\n{EOS_TOKEN.key}"
            # Edge case for CSV conversion - if NON TOKEN + UNK TOKEN we need to add underscore to NON
            if NON_TOKEN.key in guardrail_full_output and NON_TOKEN.key + "_" not in guardrail_full_output:
                guardrail_full_output = guardrail_full_output.replace(NON_TOKEN.key, NON_TOKEN.key + "_")
            examples["guardrail_model_output"] = guardrail_full_output

        if "guardrail_model_output" not in examples:
            examples["guardrail_model_output"] = ""

        if "instruction_input" not in examples:
            examples["instruction_input"] = ""

        if "valid_model_output" not in examples:
            examples["valid_model_output"] = ""

        return examples

    def to_json(self) -> dict:
        """Converts the entire template to a JSON-serializable dictionary."""
        tokens_dict: dict[str, dict[str, str]] = self.tokens.to_json()
        tokens: TokensModel = TokensModel(
            input=tokens_dict["input"],
            output=tokens_dict["output"],
        )

        instructions_dict: dict[str, dict[str, object]] = self.instructions.to_json()
        instruction_models: dict[str, InstructionDefinition] = {
            name: InstructionDefinition(**instruction_definition)
            for name, instruction_definition in instructions_dict.items()
        }

        example_usage_dict: dict[str, str] = self._create_examples()
        example_usage: ExampleUsage = ExampleUsage(**example_usage_dict)

        states: List[str] = []
        if self.model_type == ModelType.STATE_MACHINE:
            state_machine_instruction: BaseInstruction = self.instructions_list[0]
            if not isinstance(state_machine_instruction, StateMachineInstruction):
                raise TemplateFileError(
                    "For state machine templates, the provided instruction must be a StateMachineInstruction.")
            states = state_machine_instruction.get_states()

        template: TemplateModel = TemplateModel(
            encrypt=self.encrypt,
            model_type=self.model_type.value,
            states=states,
            inputs=self.inputs,
            tokens=tokens,
            instructions=instruction_models,
            example_usage=example_usage,
        )

        # Hotfix: if state machine, replace the <NON> output token with <NON>_<UNK>_
        if self.has_guardrails and self.model_type == ModelType.STATE_MACHINE:
            non_unk_combined: str = NON_TOKEN.key + "_" + UNK_TOKEN.key + "_"
            template.tokens.output[non_unk_combined] = non_unk_combined
            del template.tokens.output[UNK_TOKEN.key]

        json_dict: dict[str, object] = template.model_dump()
        final_json: dict[str, object] = {"$schema": get_template_schema_url()}
        final_json.update(json_dict)
        return final_json
