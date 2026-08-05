"""
Tests for the example_usage section of generated template files.

The example_usage block is built from a single "example" instruction picked out of the protocol. That selection used to
only recognise Instruction, ExtendedInstruction and StateMachineInstruction, so multi classifier protocols silently
produced a template with every example_usage field blank. These tests lock in that each model type produces a populated
example.
"""
import pytest

import model_train_protocol as mtp
from model_train_protocol.common.constants import BOS_TOKEN, EOS_TOKEN, NON_TOKEN, RUN_TOKEN
from model_train_protocol.common.instructions import BaseInstruction
from model_train_protocol.errors import TemplateFileError
from model_train_protocol.v2.protocol.protocol_v2 import ProtocolV2
from tests.fixtures.model_type_protocols import (
    MULTI_CLASSIFIER_SAMPLES,
    STATE_MAP,
    build_multi_classifier_protocol,
    build_state_machine_protocol,
)


class TestMultiClassifierTemplateExampleUsage:
    """The multi classifier template must contain a real example, not empty strings."""

    @pytest.fixture
    def template(self) -> dict:
        return build_multi_classifier_protocol().get_template_file().to_json()

    def test_example_usage_fields_are_not_empty(self, template):
        """Regression: every multi classifier template used to have a blank example_usage."""
        example_usage = template["example_usage"]

        assert example_usage["instruction_input"] != ""
        assert example_usage["valid_model_output"] != ""

    def test_instruction_input_uses_a_real_sample(self, template):
        """The example input frames an actual sample line between the protocol's tokens."""
        instruction_input = template["example_usage"]["instruction_input"]
        first_sample_line = MULTI_CLASSIFIER_SAMPLES[0][0]

        assert instruction_input.startswith(BOS_TOKEN.key + "\n")
        assert first_sample_line in instruction_input
        assert "States_\n" in instruction_input
        assert instruction_input.endswith(RUN_TOKEN.key + "\n")

    def test_valid_model_output_is_a_parseable_classification(self, template):
        """The example output is the JSON classification, then the final token, then <EOS>."""
        classification_line, final_token_line, eos_line = template["example_usage"]["valid_model_output"].split("\n")

        assert final_token_line == NON_TOKEN.key
        assert eos_line == EOS_TOKEN.key

        classification = mtp.common.instructions.output.MultiClassifierOutput.parse_multi_classifier_output(
            classification_line)
        assert set(classification.keys()) == set(STATE_MAP.keys())
        for key, value in classification.items():
            assert value in STATE_MAP[key]

    def test_valid_model_output_matches_declared_instruction_output(self, template):
        """The example output must be shaped like one of the instruction's declared outputs."""
        declared_outputs = template["instructions"]["MultiClassifierInstruction"]["output"]
        example_output = template["example_usage"]["valid_model_output"]

        expected = "<string>\n" + "\n".join(example_output.split("\n")[1:])
        assert expected in declared_outputs

    def test_example_usage_survives_a_bloom_round_trip(self, template):
        """A protocol reloaded from its model.json produces the same populated example_usage."""
        protocol_file = build_multi_classifier_protocol().get_protocol_file(valid=True).to_json()
        reloaded_template = ProtocolV2.from_json(protocol_file).get_template_file().to_json()

        assert reloaded_template["example_usage"]["instruction_input"] != ""
        assert reloaded_template["example_usage"]["valid_model_output"] != ""
        assert reloaded_template["example_usage"] == template["example_usage"]


class TestUnselectableInstructionRaises:
    """An instruction type the selector does not handle must fail loudly, not yield a blank example_usage."""

    def test_unknown_instruction_type_raises(self):
        protocol = build_multi_classifier_protocol()
        template_file = protocol.get_template_file()

        class UnhandledInstruction(BaseInstruction):
            """Stands in for a future instruction type nobody taught the selector about."""

            def add_sample(self, *args, **kwargs):
                raise NotImplementedError

        unhandled = object.__new__(UnhandledInstruction)
        unhandled.samples = [object()]
        template_file.instructions.instructions_list = [unhandled]

        with pytest.raises(TemplateFileError, match="UnhandledInstruction"):
            template_file._select_example_instructions()

    def test_sampleless_instructions_raise(self):
        template_file = build_multi_classifier_protocol().get_template_file()
        for instruction in template_file.instructions.instructions_list:
            instruction.samples = []

        with pytest.raises(TemplateFileError):
            template_file._select_example_instructions()


class TestOtherModelTypesStillHaveExampleUsage:
    """Guards that fixing the multi classifier case did not regress the other model types."""

    def test_state_machine_example_usage_is_populated(self):
        template = build_state_machine_protocol().get_template_file().to_json()

        assert template["example_usage"]["instruction_input"] != ""
        assert template["example_usage"]["valid_model_output"] != ""
