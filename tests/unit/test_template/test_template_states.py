"""
Tests for the `states` field of generated template files.

`states` tells a consumer what the model is allowed to answer with, and its shape follows the model type:

* generative      -> an empty list (no constrained answers)
* state_machine   -> a flat list of states, one of which the model picks
* multi_classifier -> a mapping of classification label to the values acceptable for that label

The mapping shape arrived in template schema 2.1.0; before that multi classifier templates reported an empty list and
gave the consumer no way to know the valid classifications.
"""
import json

import pytest
from model_train_protocol_schemas.structures.template import Template as TemplateModel

import model_train_protocol as mtp
from model_train_protocol.common.constants import ModelType
from model_train_protocol.errors import TemplateFileError
from model_train_protocol.v2.protocol.protocol_v2 import ProtocolV2
from model_train_protocol.v2.template_file.template_file_v2 import TemplateFileV2
from tests.unit.test_template.builders import (
    MULTI_CLASSIFIER_SAMPLES,
    STATE_MACHINE_STATES,
    STATE_MAP,
    build_generative_protocol,
    build_multi_classifier_instruction,
    build_multi_classifier_protocol,
    build_state_machine_protocol,
)


class TestMultiClassifierStates:
    """A multi classifier reports its states as label -> acceptable values."""

    @pytest.fixture
    def template(self) -> dict:
        return build_multi_classifier_protocol().get_template_file().to_json()

    def test_states_is_the_declared_state_map(self, template):
        assert template["states"] == STATE_MAP

    def test_states_is_a_mapping_not_a_list(self, template):
        """Regression: multi classifier templates used to report a bare empty list."""
        assert isinstance(template["states"], dict)
        assert template["states"] != []

    def test_states_covers_every_key_the_output_requires(self, template):
        """Every classification key the model must answer with is described in states."""
        instruction = list(build_multi_classifier_protocol().instructions)[0]

        assert set(template["states"].keys()) == set(instruction.output.required_keys)

    def test_example_output_values_are_all_listed_in_states(self, template):
        """The template's own example must be answerable from the states it advertises."""
        classification_line = template["example_usage"]["valid_model_output"].split("\n")[0]
        classification = mtp.common.instructions.output.MultiClassifierOutput.parse_multi_classifier_output(
            classification_line)

        for label, value in classification.items():
            assert value in template["states"][label]

    def test_states_are_rebuilt_from_samples_on_a_bloom_round_trip(self, template):
        """
        A protocol reloaded from its model.json rebuilds the state map out of the samples.

        The bloom file records the samples, not the originally declared state map, so a value that no sample ever uses
        ("amused") is not recoverable. Every observed value must still be present.
        """
        protocol_file = build_multi_classifier_protocol().get_protocol_file(valid=True).to_json()
        reloaded_states = ProtocolV2.from_json(protocol_file).get_template_file().to_json()["states"]

        assert set(reloaded_states.keys()) == set(STATE_MAP.keys())
        for _, classification in MULTI_CLASSIFIER_SAMPLES:
            for label, value in classification.items():
                assert value in reloaded_states[label]

    def test_states_merge_across_multiple_classifier_instructions(self):
        """Labels shared by two instructions list the union of both instructions' values."""
        template_file = TemplateFileV2(
            inputs=1,
            instructions=[
                build_multi_classifier_instruction(
                    state_map={"emotion": ["curious"], "intent": ["question"]},
                    samples=[(line, {"emotion": "curious", "intent": "question"})
                             for line, _ in MULTI_CLASSIFIER_SAMPLES],
                ),
                build_multi_classifier_instruction(
                    state_map={"emotion": ["afraid"], "mood": ["dark"]},
                    action="Whisper",
                    samples=[(line, {"emotion": "afraid", "mood": "dark"}) for line, _ in MULTI_CLASSIFIER_SAMPLES],
                ),
            ],
            encrypt=False,
            has_guardrails=False,
            model_type=ModelType.MULTI_CLASSIFICATION,
        )

        # These instructions were never added to a protocol, so their tokens carry no keys and only the states can be
        # rendered.
        assert template_file._get_states() == {
            "emotion": ["curious", "afraid"],
            "intent": ["question"],
            "mood": ["dark"],
        }

    def test_missing_classifier_instruction_raises(self):
        """A multi classifier template with no MultiClassifierInstruction cannot describe its states."""
        template_file = TemplateFileV2(
            inputs=1,
            instructions=list(build_generative_protocol().instructions),
            encrypt=False,
            has_guardrails=False,
            model_type=ModelType.MULTI_CLASSIFICATION,
        )

        with pytest.raises(TemplateFileError, match="MultiClassifierInstruction"):
            template_file.to_json()


class TestStateMachineAndGenerativeStates:
    """The other model types keep the list shape they have always had."""

    def test_state_machine_states_are_a_flat_list(self):
        states = build_state_machine_protocol().get_template_file().to_json()["states"]

        assert isinstance(states, list)
        assert sorted(states) == sorted(STATE_MACHINE_STATES)

    def test_generative_states_are_empty(self):
        assert build_generative_protocol().get_template_file().to_json()["states"] == []

    def test_state_machine_template_requires_a_state_machine_instruction(self):
        template_file = TemplateFileV2(
            inputs=1,
            instructions=list(build_generative_protocol().instructions),
            encrypt=False,
            has_guardrails=False,
            model_type=ModelType.STATE_MACHINE,
        )

        with pytest.raises(TemplateFileError, match="StateMachineInstruction"):
            template_file.to_json()


class TestStatesValidateAgainstTheTemplateSchema:
    """Both shapes must satisfy the Template pydantic model that generates the published JSON Schema."""

    @pytest.mark.parametrize("build_protocol", [
        build_generative_protocol,
        build_state_machine_protocol,
        build_multi_classifier_protocol,
    ])
    def test_template_round_trips_through_the_schema_model(self, build_protocol):
        template = build_protocol().get_template_file().to_json()
        template_without_schema_url = {key: value for key, value in template.items() if key != "$schema"}

        validated = TemplateModel(**template_without_schema_url)

        assert validated.states == template["states"]
        # The template must also survive being written out and read back as plain JSON.
        assert json.loads(json.dumps(template))["states"] == template["states"]
