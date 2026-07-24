from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Union

from model_train_protocol import Token, Guardrail, Instruction
from model_train_protocol.common.constants import TokenTypeEnum
from model_train_protocol.common.instructions.StateMachineInstruction import StateMachineInstruction

if TYPE_CHECKING:
    from model_train_protocol.v1.protocol.protocol_v1 import ProtocolV1


class BloomUtils:
    """Helper class for converting bloom files into Protocol objects."""

    @classmethod
    def add_guardrails_to_instruction(cls, protocol_instruction: Union[Instruction, StateMachineInstruction],
                                      instruction: dict):
        """Adds guardrails defined in a bloom instruction set to a protocol instruction."""
        for guardrail_set in instruction["guardrails"]:
            guardrail: Guardrail = Guardrail(
                good_prompt=guardrail_set["good_prompt"],
                bad_prompt=guardrail_set["bad_prompt"],
                bad_output=guardrail_set["bad_output"]
            )

            for sample in guardrail_set["bad_examples"]:
                guardrail.add_sample(sample)

            protocol_instruction.add_guardrail(guardrail=guardrail, tokenset_index=guardrail_set["index"])

    @classmethod
    def add_tokens(cls, protocol_file: dict, protocol: "ProtocolV1", tokens: Dict[str, Token]):
        """Adds all tokens defined in a bloom file to the protocol and the given token lookup."""
        for token_value, token_info in protocol_file["tokens"].items():
            token_value = token_value[:-1] if token_value[-1] == "_" else token_value
            token_class: type[Token] = TokenTypeEnum[token_info["type"]]
            token: Token = token_class(value=token_value, **token_info)
            protocol._add_token(token)
            tokens[token.value] = token
