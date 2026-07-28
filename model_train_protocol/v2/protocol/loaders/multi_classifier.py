from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List

from model_train_protocol import Token, FinalToken, InstructionInput, Snippet
from model_train_protocol.common.instructions.BaseInstruction import Sample
from model_train_protocol.common.instructions.MultiClassifierInstruction import MultiClassifierInstruction
from model_train_protocol.common.tokens import TokenSet
from model_train_protocol.errors import MultiClassifierError
from model_train_protocol.v2.protocol.loaders.bloom_utils import BloomUtils

if TYPE_CHECKING:
    from model_train_protocol.v2.protocol.protocol_v2 import ProtocolV2


def _build_state_map(samples: List[Sample]) -> Dict[str, List[str]]:
    """
    Builds the classification state map from the union of all sample outputs.

    Each sample output is a JSON object mapping classification labels (keys) to a classification value. The state map
    is the set of values observed across all samples for each key, so it is reconstructed by parsing every sample
    output and accumulating the values into a set per key.
    """
    state_sets: Dict[str, set] = {}
    key_order: List[str] = []
    for sample in samples:
        try:
            output_dict: Dict[str, str] = json.loads(sample.output)
        except json.JSONDecodeError:
            raise MultiClassifierError(
                f"MultiClassifier sample output must be valid JSON. Got: {sample.output}")
        for key, value in output_dict.items():
            if key not in state_sets:
                state_sets[key] = set()
                key_order.append(key)
            state_sets[key].add(value)
    return {key: sorted(state_sets[key]) for key in key_order}


def load_multi_classifier_protocol(protocol_file: dict, protocol: "ProtocolV2",
                                   tokens: Dict[str, Token]) -> "ProtocolV2":
    """Loads a multi classifier protocol from its bloom (model.json) representation."""
    # Add tokens
    BloomUtils.add_tokens(protocol_file=protocol_file, protocol=protocol, tokens=tokens)

    instruction_info = protocol_file["instruction"]
    for instruction in instruction_info["sets"]:
        context: List[str] = instruction["context"]
        tokensets: List[TokenSet] = []
        for token_set in instruction["set"]:
            tokensets.append(TokenSet([tokens[token_value] for token_value in token_set]))

        samples: List[Sample] = []
        for sample in instruction["samples"]:
            input_lines: List[str] = sample["strings"][:-1]
            output_line: str = sample["strings"][-1]
            result_token: FinalToken = tokens[sample["result"]]  # type: ignore
            samples.append(Sample(input=input_lines, output=output_line, prompt=None, numbers=sample["numbers"],
                                  number_lists=sample["number_lists"], result=result_token, value=sample["value"]))

        # Build the state map from the union of all sample outputs, collecting the set of values seen per key.
        state_map: Dict[str, List[str]] = _build_state_map(samples)

        instr_input: InstructionInput = InstructionInput(
            tokensets=tokensets[:-1],
        )

        protocol_instruction: MultiClassifierInstruction = MultiClassifierInstruction(
            input=instr_input,
            state_map=state_map,
            context=context,
        )
        # Reuse the 'States' token from the bloom file so its token is not re-added as a duplicate.
        protocol_instruction.output.tokenset = tokensets[-1]

        for sample in samples:
            inputs_snippets: List[Snippet] = []
            for i, sample_input in enumerate(sample.input):
                inputs_snippets.append(
                    tokensets[i].create_snippet(string=sample_input, number_lists=sample.number_lists[i] if len(
                        sample.number_lists[i]) > 0 else None,
                                                numbers=sample.numbers[i] if len(sample.numbers[i]) > 0 else None))

            outputs_snippet: Snippet = protocol_instruction.output.tokenset.create_snippet(
                string=sample.output,
                number_lists=sample.number_lists[-1] if len(sample.number_lists[-1]) > 0 else None,
                numbers=sample.numbers[-1] if len(sample.numbers[-1]) > 0 else None
            )

            protocol_instruction.add_sample(
                input_snippets=inputs_snippets,
                output_snippet=outputs_snippet,
                output_value=sample.value,
                final=sample.result,
            )

        # Add guardrails
        BloomUtils.add_guardrails_to_instruction(protocol_instruction=protocol_instruction, instruction=instruction)
        protocol.add_instruction(protocol_instruction)

    return protocol
