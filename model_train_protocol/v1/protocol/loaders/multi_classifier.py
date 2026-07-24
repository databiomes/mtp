from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Dict, List

from model_train_protocol import Token, FinalToken, InstructionInput, Snippet
from model_train_protocol.common.instructions.BaseInstruction import Sample
from model_train_protocol.common.instructions.MultiClassifierInstruction import MultiClassifierInstruction
from model_train_protocol.common.tokens import TokenSet
from model_train_protocol.errors import MultiClassifierError
from model_train_protocol.v1.protocol.loaders.bloom_utils import BloomUtils

if TYPE_CHECKING:
    from model_train_protocol.v1.protocol.protocol_v1 import ProtocolV1


def _recover_state_map(state_token: Token) -> Dict[str, List[str]]:
    """
    Recovers the classification state map from the description of the classifier's output ('States') token.

    The state map is embedded in the token description when a MultiClassifierInstruction is created, so it can be
    parsed back out to rebuild an equivalent instruction.
    """
    desc: str = state_token.desc or ""
    start: int = desc.find("{")
    end: int = desc.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise MultiClassifierError(
            f"Unable to recover the classification state map from token '{state_token.value}'.")
    return ast.literal_eval(desc[start:end + 1])


def load_multi_classifier_protocol(protocol_file: dict, protocol: "ProtocolV1",
                                   tokens: Dict[str, Token]) -> "ProtocolV1":
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

        # The final tokenset holds the classifier's 'States' token, whose description encodes the state map.
        state_map: Dict[str, List[str]] = _recover_state_map(tokensets[-1].tokens[0])

        instr_input: InstructionInput = InstructionInput(
            tokensets=tokensets[:-1],
        )

        protocol_instruction: MultiClassifierInstruction = MultiClassifierInstruction(
            input=instr_input,
            state_map=state_map,
        )
        protocol_instruction.context = context

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
