"""
Unit tests for the MultiClassifierInstruction class.

These tests mirror the structure of the other instruction test suites
(e.g. test_simple_instruction.py, test_samples.py). The key difference for
the MultiClassifier is that the model's response snippet must be valid JSON
containing exactly the required classification keys, so those cases are
tested explicitly.
"""
import json

import pytest

from model_train_protocol.common.instructions.MultiClassifierInstruction import MultiClassifierInstruction
from model_train_protocol.common.instructions.output.MultiClassifierOutput import MultiClassifierOutput
from model_train_protocol.common.instructions.input.InstructionInput import InstructionInput
from model_train_protocol.errors import MultiClassifierError
from tests.fixtures.tokens import SIMPLE_TOKENSET, USER_TOKENSET


# Default state map used across the tests: two classification labels, each with
# a set of acceptable values.
DEFAULT_STATE_MAP = {
    "sentiment": ["positive", "negative", "neutral"],
    "topic": ["sports", "politics"],
}


def _make_instruction(state_map=None, input_tokensets=None) -> MultiClassifierInstruction:
    """Helper to build a MultiClassifierInstruction for testing."""
    state_map = DEFAULT_STATE_MAP if state_map is None else state_map
    input_tokensets = [SIMPLE_TOKENSET] if input_tokensets is None else input_tokensets
    instruction_input = InstructionInput(tokensets=input_tokensets)
    return MultiClassifierInstruction(input=instruction_input, state_map=state_map)


class TestMultiClassifierInstruction:
    """Test cases for constructing the MultiClassifierInstruction class."""

    def test_construction(self):
        """Test creating a MultiClassifierInstruction with a basic state map."""
        instruction = _make_instruction()

        assert isinstance(instruction, MultiClassifierInstruction)
        assert len(instruction.input.tokensets) == 1
        assert instruction.name == "MultiClassifierInstruction"

    def test_output_is_multi_classifier_output(self):
        """Test that the instruction's output is a MultiClassifierOutput instance."""
        instruction = _make_instruction()

        assert isinstance(instruction.output, MultiClassifierOutput)

    def test_required_keys_match_state_map(self):
        """Test that the output's required keys match the state map keys."""
        instruction = _make_instruction()

        assert instruction.output.required_keys == list(DEFAULT_STATE_MAP.keys())

    def test_single_key_state_map(self):
        """Test creating a MultiClassifierInstruction with a single classification key."""
        instruction = _make_instruction(state_map={"sentiment": ["positive", "negative"]})

        assert instruction.output.required_keys == ["sentiment"]

    def test_multiple_input_tokensets(self):
        """Test creating a MultiClassifierInstruction with multiple input tokensets."""
        instruction = _make_instruction(input_tokensets=[SIMPLE_TOKENSET, USER_TOKENSET])

        assert len(instruction.input.tokensets) == 2


class TestMultiClassifierJSONValidation:
    """Test cases for JSON validity of the MultiClassifier response snippet."""

    def test_valid_json_output_succeeds(self):
        """Test that a valid JSON response with all required keys is accepted."""
        instruction = _make_instruction()

        input_snippet = SIMPLE_TOKENSET.create_snippet("The cat sits in the tree")
        output_string = json.dumps({"sentiment": "positive", "topic": "sports"})
        output_snippet = instruction.output.tokenset.create_snippet(output_string)

        instruction.add_sample(
            input_snippets=[input_snippet],
            output_snippet=output_snippet,
        )

        assert len(instruction.samples) == 1

    def test_valid_json_single_key_succeeds(self):
        """Test that a valid single-key JSON response is accepted."""
        instruction = _make_instruction(state_map={"sentiment": ["positive", "negative"]})

        input_snippet = SIMPLE_TOKENSET.create_snippet("The cat sits in the tree")
        output_string = json.dumps({"sentiment": "positive"})
        output_snippet = instruction.output.tokenset.create_snippet(output_string)

        instruction.add_sample(
            input_snippets=[input_snippet],
            output_snippet=output_snippet,
        )

        assert len(instruction.samples) == 1

    def test_invalid_json_raises_error(self):
        """Test that a malformed JSON response snippet raises an error."""
        instruction = _make_instruction()

        input_snippet = SIMPLE_TOKENSET.create_snippet("The cat sits in the tree")
        # Not valid JSON - missing closing brace and quotes.
        output_snippet = instruction.output.tokenset.create_snippet("{sentiment: positive, topic: sports")

        with pytest.raises(MultiClassifierError, match="valid JSON"):
            instruction.add_sample(
                input_snippets=[input_snippet],
                output_snippet=output_snippet,
            )

    def test_plain_text_not_json_raises_error(self):
        """Test that a plain-text (non-JSON) response snippet raises an error."""
        instruction = _make_instruction()

        input_snippet = SIMPLE_TOKENSET.create_snippet("The cat sits in the tree")
        output_snippet = instruction.output.tokenset.create_snippet("positive and sports")

        with pytest.raises(MultiClassifierError, match="valid JSON"):
            instruction.add_sample(
                input_snippets=[input_snippet],
                output_snippet=output_snippet,
            )

    def test_missing_required_key_raises_error(self):
        """Test that a valid JSON response missing a required key raises an error."""
        instruction = _make_instruction()

        input_snippet = SIMPLE_TOKENSET.create_snippet("The cat sits in the tree")
        # Valid JSON but missing the "topic" key.
        output_string = json.dumps({"sentiment": "positive"})
        output_snippet = instruction.output.tokenset.create_snippet(output_string)

        with pytest.raises(MultiClassifierError, match="must contain the key"):
            instruction.add_sample(
                input_snippets=[input_snippet],
                output_snippet=output_snippet,
            )

    def test_extra_key_raises_error(self):
        """Test that a valid JSON response with too many keys raises an error."""
        instruction = _make_instruction()

        input_snippet = SIMPLE_TOKENSET.create_snippet("The cat sits in the tree")
        # Valid JSON with all required keys plus an unexpected extra key.
        output_string = json.dumps({"sentiment": "positive", "topic": "sports", "extra": "value"})
        output_snippet = instruction.output.tokenset.create_snippet(output_string)

        with pytest.raises(MultiClassifierError, match="exactly"):
            instruction.add_sample(
                input_snippets=[input_snippet],
                output_snippet=output_snippet,
            )

    def test_wrong_key_names_raises_error(self):
        """Test that valid JSON with the right count but wrong key names raises an error."""
        instruction = _make_instruction()

        input_snippet = SIMPLE_TOKENSET.create_snippet("The cat sits in the tree")
        # Correct number of keys but wrong names.
        output_string = json.dumps({"mood": "positive", "subject": "sports"})
        output_snippet = instruction.output.tokenset.create_snippet(output_string)

        with pytest.raises(MultiClassifierError, match="must contain the key"):
            instruction.add_sample(
                input_snippets=[input_snippet],
                output_snippet=output_snippet,
            )
