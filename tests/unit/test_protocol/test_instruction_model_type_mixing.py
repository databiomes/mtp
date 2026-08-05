"""
Tests that a protocol trains exactly one type of model.

The instruction classes a protocol holds are what decide the model type, so a caller never has to declare it: adding a
MultiClassifierInstruction makes a multi classifier, adding a StateMachineInstruction makes a state machine, and
anything else is generative. Because the type is inferred, instructions from two different types cannot be mixed into
one protocol — that would describe a model that cannot exist — so add_instruction rejects the second family.
"""
import pytest

import model_train_protocol as mtp
from model_train_protocol.common.constants import ModelType
from model_train_protocol.errors import ProtocolError, ProtocolTypeError, StateMachineError
from model_train_protocol.v2.protocol.protocol_v2 import ProtocolV2
from tests.fixtures.model_type_protocols import (
    build_extended_instruction,
    build_generative_instruction,
    build_generative_protocol,
    build_multi_classifier_instruction,
    build_multi_classifier_protocol,
    build_state_machine_instruction,
    build_state_machine_protocol,
)


def _other_generative_instruction() -> mtp.Instruction:
    """A second generative instruction that does not collide with the first one's name or tokens."""
    return build_generative_instruction(action="Shout", name="alice_cat_shout")


class TestInstructionsDecideTheModelType:
    """The model type follows from the instruction classes, with no need to declare it."""

    def test_generative_instructions_make_a_generative_protocol(self):
        assert build_generative_protocol().get_model_type() == ModelType.GENERATIVE

    def test_multi_classifier_instruction_makes_a_multi_classifier_protocol(self):
        assert build_multi_classifier_protocol().get_model_type() == ModelType.MULTI_CLASSIFICATION

    def test_state_machine_instruction_makes_a_state_machine_protocol(self):
        """A StateMachineInstruction is the only thing needed; there is no model type to declare."""
        protocol = mtp.Protocol(name="inferred_state_machine", inputs=1, encrypt=False)
        protocol.add_context("The Cheshire Cat labels each line Alice speaks.")

        protocol.add_instruction(build_state_machine_instruction())

        assert protocol.get_model_type() == ModelType.STATE_MACHINE

    def test_state_machine_rules_follow_the_instruction(self):
        """The one-instruction rule comes from the instruction type, with nothing declared at construction."""
        protocol = mtp.Protocol(name="inferred_state_machine_two", inputs=1, encrypt=False)
        protocol.add_context("The Cheshire Cat labels each line Alice speaks.")
        protocol.add_instruction(build_state_machine_instruction())

        with pytest.raises(StateMachineError, match="can only have one instruction"):
            protocol.add_instruction(build_state_machine_instruction(action="Shout"))

    def test_empty_protocol_is_generative(self):
        """With no instructions there is nothing to infer from, so a protocol starts out generative."""
        assert mtp.Protocol(name="empty", inputs=1, encrypt=False).get_model_type() == ModelType.GENERATIVE

    def test_the_model_type_cannot_be_declared(self):
        """The old state_machine flag is gone: the instructions are the only thing that decides the model type."""
        with pytest.raises(TypeError):
            mtp.Protocol(name="declared", inputs=1, encrypt=False, state_machine=True)

    @pytest.mark.parametrize("build_instruction, expected_model_type", [
        (build_generative_instruction, ModelType.GENERATIVE),
        (build_extended_instruction, ModelType.GENERATIVE),
        (build_multi_classifier_instruction, ModelType.MULTI_CLASSIFICATION),
        (build_state_machine_instruction, ModelType.STATE_MACHINE),
    ])
    def test_instruction_classification(self, build_instruction, expected_model_type):
        assert ProtocolV2.get_model_type_for_instruction(build_instruction()) == expected_model_type

    def test_non_instruction_is_rejected(self):
        with pytest.raises(ProtocolTypeError, match="BaseInstruction"):
            ProtocolV2.get_model_type_for_instruction("not an instruction")


class TestGenerativeInstructionsMixWithEachOther:
    """Basic and extended instructions both train a generative model, so they may share a protocol."""

    def test_basic_and_extended_instructions_can_be_mixed(self):
        protocol = build_generative_protocol()

        protocol.add_instruction(build_extended_instruction())

        assert protocol.get_model_type() == ModelType.GENERATIVE
        assert len(protocol.instructions) == 2

    def test_two_basic_instructions_can_be_mixed(self):
        protocol = build_generative_protocol()

        protocol.add_instruction(_other_generative_instruction())

        assert protocol.get_model_type() == ModelType.GENERATIVE
        assert len(protocol.instructions) == 2


class TestModelTypesCannotBeMixed:
    """Every cross-type combination is rejected, in both orders."""

    @pytest.mark.parametrize("build_protocol, build_instruction, expected_error", [
        (build_generative_protocol, build_multi_classifier_instruction, ProtocolTypeError),
        (build_generative_protocol, build_state_machine_instruction, StateMachineError),
        (build_multi_classifier_protocol, _other_generative_instruction, ProtocolTypeError),
        (build_multi_classifier_protocol, build_state_machine_instruction, StateMachineError),
        (build_state_machine_protocol, _other_generative_instruction, StateMachineError),
        (build_state_machine_protocol, build_multi_classifier_instruction, StateMachineError),
    ])
    def test_mixing_instruction_types_is_rejected(self, build_protocol, build_instruction, expected_error):
        protocol = build_protocol()

        with pytest.raises(expected_error):
            protocol.add_instruction(build_instruction())

    def test_every_mixing_error_is_a_protocol_error(self):
        """Callers that only catch ProtocolError still see the mixing failures (StateMachineError subclasses it)."""
        protocol = build_generative_protocol()

        with pytest.raises(ProtocolError):
            protocol.add_instruction(build_state_machine_instruction())

    def test_rejection_names_both_model_types(self):
        protocol = build_generative_protocol()

        with pytest.raises(ProtocolTypeError, match="generative.*multi_classifier"):
            protocol.add_instruction(build_multi_classifier_instruction())

    def test_a_rejected_instruction_is_not_added(self):
        """A failed add leaves the protocol exactly as it was."""
        protocol = build_multi_classifier_protocol()
        instructions_before = set(protocol.instructions)
        tokens_before = set(protocol.tokens)

        with pytest.raises(ProtocolTypeError):
            protocol.add_instruction(_other_generative_instruction())

        assert set(protocol.instructions) == instructions_before
        assert set(protocol.tokens) == tokens_before
        assert protocol.get_model_type() == ModelType.MULTI_CLASSIFICATION

    def test_a_rejected_state_machine_instruction_does_not_flip_the_protocol(self):
        """Rejecting a StateMachineInstruction must not turn a generative protocol into a state machine."""
        protocol = build_generative_protocol()

        with pytest.raises(StateMachineError):
            protocol.add_instruction(build_state_machine_instruction())

        assert protocol.get_model_type() == ModelType.GENERATIVE
