from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from model_train_protocol import Token, FinalToken, Snippet
from model_train_protocol.common.instructions.BaseInstruction import Sample
from model_train_protocol.common.instructions.StateMachineInstruction import StateMachineInstruction
from model_train_protocol.common.instructions.input.StateMachineInput import StateMachineInput
from model_train_protocol.common.tokens import TokenSet
from model_train_protocol.v1.protocol.loaders.bloom_utils import BloomUtils

if TYPE_CHECKING:
    from model_train_protocol.v1.protocol.protocol_v1 import ProtocolV1


def load_state_machine_protocol(protocol_file: dict, protocol: "ProtocolV1",
                                tokens: Dict[str, Token]) -> "ProtocolV1":
    """Loads a state machine protocol from its bloom (model.json) representation."""
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

        instr_input: StateMachineInput = StateMachineInput(
            tokensets=tokensets[:-1],
        )

        states: list[str] = list(dict.fromkeys([sample.output for sample in samples]))
        protocol_instruction: StateMachineInstruction = StateMachineInstruction(
            input=instr_input,
            states=states,
        )
        protocol_instruction.output.tokenset = tokensets[-1]
        protocol_instruction.context = context

        for sample in samples:
            inputs_snippets: List[Snippet] = []
            for i, sample_input in enumerate(sample.input):
                inputs_snippets.append(
                    tokensets[i].create_snippet(string=sample_input, number_lists=sample.number_lists[i] if len(
                        sample.number_lists[i]) > 0 else None,
                                                numbers=sample.numbers[i] if len(sample.numbers[i]) > 0 else None))

            protocol_instruction.add_sample(
                input_snippets=inputs_snippets,
                state=sample.output,
            )

        # Add guardrails
        BloomUtils.add_guardrails_to_instruction(protocol_instruction=protocol_instruction, instruction=instruction)
        protocol.add_instruction(protocol_instruction)

    return protocol
