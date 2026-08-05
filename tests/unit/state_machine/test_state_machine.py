"""
Unit tests for state machine protocol requirements.
"""
from typing import List

import pytest

import model_train_protocol as mtp


def _add_context_lines(protocol: mtp.Protocol, total_lines: int = 10) -> None:
    for i in range(total_lines):
        protocol.add_context(f"State machine context line {i + 1}")


def _build_state_machine_instruction(sample_count: int, token_prefix: str) -> mtp.StateMachineInstruction:
    machine_token: mtp.Token = mtp.Token(f"{token_prefix}Machine")
    event_token: mtp.Token = mtp.Token(f"{token_prefix}Event")

    state_tokenset: mtp.TokenSet = mtp.TokenSet(tokens=machine_token)
    event_tokenset: mtp.TokenSet = mtp.TokenSet(tokens=event_token)

    instruction_input: mtp.StateMachineInput = mtp.StateMachineInput(
        tokensets=[state_tokenset, event_tokenset],
    )

    states: List[str] = [f"Action {i}" for i in range(sample_count)]

    instruction: mtp.StateMachineInstruction = mtp.StateMachineInstruction(
        input=instruction_input,
        states=states
    )

    for i in range(sample_count):
        instruction.add_sample(
            input_snippets=[f"State {i}", f"Event {i}"],
            state=f"Action {i}",
        )

    return instruction


def _build_basic_instruction() -> mtp.Instruction:
    token_one: mtp.Token = mtp.Token("InputOne")
    token_two: mtp.Token = mtp.Token("InputTwo")
    token_out: mtp.Token = mtp.Token("Output")
    final_token: mtp.FinalToken = mtp.FinalToken("Done")

    input_one_set: mtp.TokenSet = mtp.TokenSet(tokens=(token_one,))
    input_two_set: mtp.TokenSet = mtp.TokenSet(tokens=(token_two,))
    output_set: mtp.TokenSet = mtp.TokenSet(tokens=(token_out,))

    instruction_input: mtp.InstructionInput = mtp.InstructionInput(tokensets=[input_one_set, input_two_set])
    instruction_output: mtp.InstructionOutput = mtp.InstructionOutput(tokenset=output_set, final=final_token)

    return mtp.Instruction(name='instruction_1', input=instruction_input, output=instruction_output)


class TestStateMachine:
    """Test cases for state machine protocol requirements."""

    def test_state_machine_protocol_no_instructions_invalid(
            self,
            empty_state_machine_protocol: mtp.Protocol,
    ) -> None:
        valid, error_msg = empty_state_machine_protocol.validate_protocol()

        assert not valid
        assert error_msg is not None
        assert "No instructions have been added" in error_msg

    def test_state_machine_protocol_single_instruction_valid(
            self,
            state_machine_protocol: mtp.Protocol,
    ) -> None:
        valid, error_msg = state_machine_protocol.validate_protocol()

        assert valid
        assert error_msg is None

    def test_state_machine_protocol_two_instructions_rejected(
            self,
            state_machine_instruction_with_samples: mtp.Instruction,
    ) -> None:
        protocol: mtp.Protocol = mtp.Protocol("state_machine_two_instructions", inputs=2, encrypt=False)
        _add_context_lines(protocol)
        protocol.add_instruction(state_machine_instruction_with_samples)

        second_instruction: mtp.StateMachineInstruction = _build_state_machine_instruction(
            sample_count=10,
            token_prefix="SM2",
        )
        with pytest.raises(mtp.StateMachineError, match="A state machine protocol can only have one instruction."):
            protocol.add_instruction(second_instruction)

    def test_state_machine_protocol_rejects_non_state_machine_instruction(
            self,
            state_machine_instruction_with_samples: mtp.Instruction,
    ) -> None:
        """A protocol is a state machine because of its StateMachineInstruction, which then excludes other types."""
        protocol: mtp.Protocol = mtp.Protocol("state_machine", inputs=2, encrypt=False)
        _add_context_lines(protocol)
        protocol.add_instruction(state_machine_instruction_with_samples)

        with pytest.raises(mtp.StateMachineError, match="only train one type of model"):
            protocol.add_instruction(_build_basic_instruction())

    def test_state_machine_protocol_rejects_numeric_output_tokens(self) -> None:
        protocol: mtp.Protocol = mtp.Protocol("state_machine", inputs=2, encrypt=False)
        instruction: mtp.StateMachineInstruction = _build_state_machine_instruction(
            sample_count=0,
            token_prefix="SM",
        )
        instruction.output.final = [mtp.FinalNumToken("Score", min_value=0, max_value=10)]

        with pytest.raises(mtp.StateMachineError, match="cannot have a generated numeric output"):
            protocol.add_instruction(instruction)

    def test_state_machine_instruction_minimum_samples(self) -> None:
        protocol: mtp.Protocol = mtp.Protocol("state_machine", inputs=2, encrypt=False)
        _add_context_lines(protocol)
        instruction: mtp.StateMachineInstruction = _build_state_machine_instruction(
            sample_count=9,
            token_prefix="SM",
        )
        protocol.add_instruction(instruction)

        valid, error_msg = protocol.validate_protocol()
        assert not valid
        assert error_msg is not None
        assert "has only 9 samples" in error_msg

    def test_state_machine_instruction_add_sample_wrong_input_count(self) -> None:
        instruction: mtp.StateMachineInstruction = _build_state_machine_instruction(
            sample_count=0,
            token_prefix="SM",
        )

        with pytest.raises(mtp.InstructionError, match="Number of context snippets"):
            instruction.add_sample(
                input_snippets=["Only one input"],
                state="Action",
            )
