"""
State machine protocol fixtures for testing state machine requirements.
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


@pytest.fixture
def state_machine_instruction_with_samples() -> mtp.StateMachineInstruction:
    return _build_state_machine_instruction(sample_count=10, token_prefix="SM")


@pytest.fixture
def state_machine_instruction_with_few_samples() -> mtp.StateMachineInstruction:
    return _build_state_machine_instruction(sample_count=9, token_prefix="SM")


@pytest.fixture
def empty_state_machine_protocol() -> mtp.Protocol:
    protocol: mtp.Protocol = mtp.Protocol("empty_state_machine", inputs=2, encrypt=False)
    _add_context_lines(protocol)
    return protocol


@pytest.fixture
def state_machine_protocol(state_machine_instruction_with_samples: mtp.StateMachineInstruction) -> mtp.Protocol:
    protocol: mtp.Protocol = mtp.Protocol("state_machine_protocol", inputs=2, encrypt=False)
    _add_context_lines(protocol)
    protocol.add_instruction(state_machine_instruction_with_samples)
    return protocol
