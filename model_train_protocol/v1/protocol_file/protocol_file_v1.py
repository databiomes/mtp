from dataclasses import dataclass, field
from typing import Collection, List, Dict, Set

from packaging.version import Version

from model_train_protocol import Token, NumToken
from model_train_protocol.common.instructions import BaseInstruction
from model_train_protocol_schemas.structures.protocol import Instruction, TokenInfo, Sample, \
    InstructionSet, Guardrail
from model_train_protocol_schemas.structures.protocol import Protocol
from model_train_protocol_schemas.utils import get_bloom_schema_url
from model_train_protocol.common.constants import ModelType
from model_train_protocol.common.tokens import SpecialToken
from model_train_protocol.errors import ProtocolFileLayerDepthError


class ProtocolFileV1:
    """Manages the model.json file for model training protocols."""

    @dataclass
    class ProtocolInstruction:
        """Represents an instruction in the template."""

        inputs: int
        sets: List = field(default_factory=list)

    @dataclass
    class ProtocolInstructionSet:
        """Represents an instruction set in the template."""
        name: str
        guardrails: List[Guardrail]
        context: List[str]
        set: List[List[str]]
        samples: List
        ppo: List

    @dataclass
    class Batches:
        """Represents batches in the template."""

        pretrain: List = field(default_factory=list)
        instruct: List = field(default_factory=list)
        judge: List = field(default_factory=list)
        ppo: List = field(default_factory=list)

    def __init__(self, name: str, context: List[str], inputs: int, encrypted: bool, valid: bool,
                 model_type: ModelType,
                 tokens: Collection[Token], special_tokens: Collection[Token],
                 instructions: Collection[BaseInstruction], bloom_version: Version):
        """Initializes the Template with a name and context."""
        self.bloom_version: Version = bloom_version
        self.name: str = name
        self.inputs: int = inputs
        self.context: List[str] = context
        self.encrypted: bool = encrypted
        self.valid: bool = valid
        self.model_type: ModelType = model_type
        self.tokens: Dict[str, dict] = {}
        self.special_token_keys: Set[str] = set()
        self.instruction_token_keys: Set[str] = set()
        self.instruction: ProtocolFileV1.ProtocolInstruction = ProtocolFileV1.ProtocolInstruction(
            inputs=inputs)
        self.numbers: Dict[str, str] = {}
        self.batches: ProtocolFileV1.Batches = ProtocolFileV1.Batches()

        # Add regular tokens
        self.add_tokens(tokens)

        # Add special tokens
        self.add_tokens(special_tokens)

        # Add instructions
        self.add_instructions(instructions)

    def add_tokens(self, tokens: Collection[Token]):
        """Adds tokens to the template."""
        for token in tokens:
            token_dict: Dict[str, dict] = token.to_dict()
            token_dict.pop("value")
            self.tokens[token.value] = token_dict

            # Add numbers to the numbers dictionary
            if isinstance(token, NumToken):
                self.numbers[token.value] = token.template_representation

            # Add special tokens to the special tokens set
            if isinstance(token, SpecialToken):
                self.special_token_keys.add(token.key)

    def add_instructions(self, instructions: Collection[BaseInstruction]):
        """Adds instructions to the template."""
        for instruction in instructions:
            instruction_set: ProtocolFileV1.ProtocolInstructionSet = ProtocolFileV1.ProtocolInstructionSet(
                name=instruction.name,
                guardrails=instruction.serialize_guardrails(),
                context=instruction.context,
                set=instruction.serialize_memory_set(),
                samples=instruction.serialize_samples(),
                ppo=instruction.serialize_ppo(),
            )
            self.instruction.sets.append(instruction_set)

            # Add instruction token keys
            for token_set in instruction.get_token_sets():
                self._add_instruction_token_key(token_set.get_token_key_set())

            # Add the result token in each sample as a special token and to tokens dictionary
            for sample in instruction.samples:
                result_token = sample.result
                # Add to instruction token keys
                self._add_instruction_token_key(result_token.key)
                # Add to tokens dictionary if not already present
                if result_token.value not in self.tokens:
                    token_dict = result_token.to_dict()
                    token_dict.pop("value")
                    self.tokens[result_token.value] = token_dict

    def _add_instruction_token_key(self, key: str):
        """Adds an instruction token key to the template."""
        self.instruction_token_keys.add(key)

    def _get_special_token_keys(self):
        """
        Returns a sorted list of tokens that should be under 'special_tokens' in the JSON.

        Guarantees that <BOS> and <EOS> are always the first two tokens in the list, followed by other special tokens sorted alphabetically.

        :return: A sorted list of special token keys.
        """
        all_keys = self.special_token_keys | self.instruction_token_keys

        def sort_priority(key):
            if key == "<BOS>": return 0, key
            if key == "<EOS>": return 1, key
            return 2, key

        return sorted(list(all_keys), key=sort_priority)

    @classmethod
    def _alphabetize_dicts_by_keys_after_layer_n(cls, data: dict, n: int = 1):
        """
        Alphabetizes the keys in the protocol JSON after a specified layer depth.
        :param n: The layer depth after which to alphabetize keys. Default is 1.
        """
        if n < 1:
            raise ProtocolFileLayerDepthError("Layer depth n must be at least 1.")

        def _recursively_alphabetize(data, current_layer):
            if isinstance(data, dict):
                if current_layer >= n:
                    # Alphabetize keys at this layer
                    sorted_dict = {}
                    for key in sorted(data.keys()):
                        sorted_dict[key] = _recursively_alphabetize(data[key], current_layer + 1)
                    return sorted_dict
                else:
                    # Do not alphabetize keys at this layer
                    return {key: _recursively_alphabetize(value, current_layer + 1) for key, value in data.items()}
            elif isinstance(data, list):
                return [_recursively_alphabetize(item, current_layer) for item in data]
            else:
                return data

        return _recursively_alphabetize(data, 0)

    @classmethod
    def _alphabetize_list_of_dict_by_key_value(cls, data: List, key: str) -> List:
        """
        Alphabetizes a list of dictionaries by the value of a specific key.
        
        :param data: List of dictionaries to alphabetize
        :param key: The key to use for alphabetization
        :return: Alphabetized list of dictionaries
        """
        if not isinstance(data, list):
            return data

        # Filter out non-dictionary items and items without the key
        dict_items = [item for item in data if isinstance(item, dict) and key in item]
        non_dict_items = [item for item in data if not isinstance(item, dict) or key not in item]

        # Sort dictionary items by the specified key value
        sorted_dict_items = sorted(dict_items, key=lambda x: str(x[key]))

        # Combine sorted dictionary items with non-dictionary items
        # Non-dictionary items are placed at the end
        return sorted_dict_items + non_dict_items

    def to_json(self) -> dict:
        """Converts the template to a JSON-compatible dictionary using Pydantic models."""

        # Create TokenInfo objects for each token
        token_info_dict = {}
        for token_value, token_dict in self.tokens.items():
            token_info = TokenInfo(
                key=token_dict['key'],
                num=token_dict['num'],
                num_list=token_dict['num_list'],
                min_value=token_dict['min_value'],
                max_value=token_dict['max_value'],
                length=token_dict['length'],
                desc=token_dict['desc'],
                special=token_dict['special'],
                type=token_dict['type']
            )
            token_info_dict[token_value] = token_info

        # Create InstructionSet objects
        instruction_sets = []
        for instruction_set in self.instruction.sets:
            # Create Sample objects
            samples = []
            for sample_data in instruction_set.samples:
                sample = Sample(**sample_data)
                samples.append(sample)

            instruction_set_obj = InstructionSet(
                    name=instruction_set.name,
                    guardrails=instruction_set.guardrails,
                    context=instruction_set.context,
                    set=instruction_set.set,
                    samples=samples,
                    ppo=instruction_set.ppo
                )
            instruction_sets.append(instruction_set_obj)

        # Create Instruction object
        instruction = Instruction(
            memory=self.instruction.inputs + 1,  # +1 for the response line
            sets=instruction_sets
        )

        # Create ProtocolModel
        protocol = Protocol(
            name=self.name,
            context=self.context,
            model_type=self.model_type.value,
            inputs=self.inputs,
            encrypted=self.encrypted,
            valid=self.valid,
            tokens=token_info_dict,
            special_tokens=self._get_special_token_keys(),
            instruction=instruction
        )

        # Convert to JSON and apply backwards compatibility transformations
        json_dict = protocol.model_dump(by_alias=True)
        json_dict = self._alphabetize_dicts_by_keys_after_layer_n(json_dict, n=1)

        # Apply list alphabetization to relevant lists of dictionaries
        if "instruction" in json_dict and "sets" in json_dict["instruction"]:
            # Alphabetize instruction sets by "result" key
            json_dict["instruction"]["sets"] = self._alphabetize_list_of_dict_by_key_value(
                json_dict["instruction"]["sets"], "result"
            )

            # Alphabetize samples within each instruction set by "result" key
            for instruction_set in json_dict["instruction"]["sets"]:
                if "samples" in instruction_set:
                    instruction_set["samples"] = self._alphabetize_list_of_dict_by_key_value(
                        instruction_set["samples"], "result"
                    )

        # Alphabetize batch lists if they contain dictionaries
        if "batches" in json_dict:
            for batch_key in ["pretrain", "instruct", "judge", "ppo"]:
                if batch_key in json_dict["batches"]:
                    json_dict["batches"][batch_key] = self._alphabetize_list_of_dict_by_key_value(
                        json_dict["batches"][batch_key], "result"
                    )

        # Reconstruct the dictionary with $schema at the top
        final_json = {"$schema": get_bloom_schema_url(version=self.bloom_version)}
        final_json.update(json_dict)

        return final_json
