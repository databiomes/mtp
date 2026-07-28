from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from model_train_protocol import Token, FinalToken, Instruction, InstructionInput, InstructionOutput, Snippet
from model_train_protocol.common.instructions.BaseInstruction import Sample
from model_train_protocol.common.tokens import TokenSet
from model_train_protocol.v2.protocol.loaders.bloom_utils import BloomUtils

if TYPE_CHECKING:
    from model_train_protocol.v2.protocol.protocol_v2 import ProtocolV2


def load_generative_protocol(protocol_file: dict, protocol: "ProtocolV2",
                             tokens: Dict[str, Token]) -> "ProtocolV2":
    """Loads a generative protocol from its bloom (model.json) representation."""
    # Add tokens
    BloomUtils.add_tokens(protocol_file=protocol_file, protocol=protocol, tokens=tokens)

    # Add instructions
    instruction_info = protocol_file["instruction"]
    for i, instruction in enumerate(instruction_info["sets"]):
        if "name" in instruction:
            instruction_name = instruction["name"]
        else:
            instruction_name = f"Instruction_{i}"
        context: List[str] = instruction["context"]
        tokensets: List[TokenSet] = []
        final_tokens: List[FinalToken] = []
        for token_set in instruction["set"]:
            tokensets.append(TokenSet([tokens[token_value] for token_value in token_set]))

        samples: List[Sample] = []
        for sample in instruction["samples"]:
            input_lines: List[str] = sample["strings"][:-1]
            output_line: str = sample["strings"][-1]
            result_token: FinalToken = tokens[sample["result"]]  # type: ignore
            final_tokens.append(result_token)
            samples.append(Sample(input=input_lines, output=output_line, prompt=None, numbers=sample["numbers"],
                                  number_lists=sample["number_lists"], result=result_token, value=sample["value"]))

        instr_input: InstructionInput = InstructionInput(
            tokensets=tokensets[:-1],
        )

        instr_output: InstructionOutput = InstructionOutput(
            tokenset=tokensets[-1],
            final=final_tokens,
        )

        protocol_instruction: Instruction = Instruction(
            name=instruction_name,
            input=instr_input,
            output=instr_output,
            context=context
        )

        for sample in samples:
            inputs_snippets: List[Snippet] = []
            for i, sample_input in enumerate(sample.input):
                inputs_snippets.append(
                    tokensets[i].create_snippet(string=sample_input, number_lists=sample.number_lists[i] if len(
                        sample.number_lists[i]) > 0 else None,
                                                numbers=sample.numbers[i] if len(sample.numbers[i]) > 0 else None))

            outputs_snippet: Snippet = tokensets[-1].create_snippet(
                string=sample.output,
                number_lists=sample.number_lists[-1] if len(sample.number_lists[-1]) > 0 else None
                , numbers=sample.numbers[-1] if len(sample.numbers[-1]) > 0 else None
            )

            final_token: FinalToken = sample.result

            protocol_instruction.add_sample(
                input_snippets=inputs_snippets,
                output_snippet=outputs_snippet,
                output_value=sample.value,
                final=final_token,
            )

        # Add guardrails
        BloomUtils.add_guardrails_to_instruction(protocol_instruction=protocol_instruction, instruction=instruction)
        protocol.add_instruction(protocol_instruction)

    return protocol
