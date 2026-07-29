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
from model_train_protocol.common.instructions.output.MultiClassifierOutput import (
    MultiClassifierOutput,
    format_multi_classifier_output,
    parse_multi_classifier_output,
)
from model_train_protocol.common.instructions.input.InstructionInput import InstructionInput
from model_train_protocol.errors import MultiClassifierError
from tests.fixtures.tokens import SIMPLE_TOKENSET, USER_TOKENSET


# Default state map used across the tests: two classification labels, each with
# a set of acceptable values.
DEFAULT_STATE_MAP = {
    "sentiment": ["positive", "negative", "neutral"],
    "topic": ["sports", "politics"],
}


def _make_instruction(state_map=None, input_tokensets=None, context=None) -> MultiClassifierInstruction:
    """Helper to build a MultiClassifierInstruction for testing."""
    state_map = DEFAULT_STATE_MAP if state_map is None else state_map
    input_tokensets = [SIMPLE_TOKENSET] if input_tokensets is None else input_tokensets
    instruction_input = InstructionInput(tokensets=input_tokensets)
    return MultiClassifierInstruction(input=instruction_input, state_map=state_map, context=context)


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

    def test_context_defaults_to_empty(self):
        """Test that omitting context leaves the instruction context empty."""
        instruction = _make_instruction()

        assert instruction.context == []

    def test_context_parameter(self):
        """Test creating a MultiClassifierInstruction with context provided at construction."""
        context = ["Classify each message by sentiment and topic.", "Responses are JSON objects."]
        instruction = _make_instruction(context=context)

        assert instruction.context == context

    def test_add_context_appends_to_constructor_context(self):
        """Test that add_context appends to context provided at construction."""
        instruction = _make_instruction(context=["Classify each message by sentiment and topic."])

        instruction.add_context("Responses are JSON objects.")

        assert instruction.context == [
            "Classify each message by sentiment and topic.",
            "Responses are JSON objects.",
        ]


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


class TestMultiClassifierOutputParsing:
    """Test cases for the accepted input formats of a MultiClassifier output string."""

    @pytest.mark.parametrize("output_string", [
        '{"sentiment": "positive", "topic": "sports"}',  # strict JSON, as produced by json.dumps
        "{'sentiment': 'positive', 'topic': 'sports'}",  # canonical single-quoted form stored in the bloom file
        '{"sentiment": \'positive\', "topic": \'sports\'}',  # mixed quoting
        '{ "sentiment" : "positive" ,\n  "topic" : "sports" }',  # extra whitespace and newlines
    ])
    def test_accepted_formats_parse_to_same_dict(self, output_string):
        """Test that each accepted output format parses to the same dictionary."""
        assert parse_multi_classifier_output(output_string) == {"sentiment": "positive", "topic": "sports"}

    @pytest.mark.parametrize("output_string", [
        '{"sentiment": "it\'s positive"}',  # single quote inside a value
        '{"sentiment": "say \\"positive\\""}',  # escaped double quotes inside a value
        '{"sentiment": "both \' and \\" quotes"}',  # both quote characters inside a value
        '{"sentiment": "back\\\\slash"}',  # backslash inside a value
        "{'sentiment': \"it's positive\"}",  # canonical rendering of a value containing a single quote
    ])
    def test_nested_quotes_survive_round_trip(self, output_string):
        """Test that values containing quote characters survive parse -> format -> parse."""
        parsed = parse_multi_classifier_output(output_string)

        assert parse_multi_classifier_output(format_multi_classifier_output(parsed)) == parsed

    def test_format_uses_canonical_single_quoted_form(self):
        """Test that formatting produces the documented single-quoted form."""
        formatted = format_multi_classifier_output({"emotion": "confused", "intent": "question"})

        assert formatted == "{'emotion': 'confused', 'intent': 'question'}"

    def test_format_is_idempotent(self):
        """Test that re-formatting an already canonical string does not change it."""
        canonical = "{'sentiment': 'positive', 'topic': 'sports'}"

        assert format_multi_classifier_output(parse_multi_classifier_output(canonical)) == canonical

    def test_format_preserves_key_order(self):
        """Test that formatting preserves the key order of the parsed output."""
        parsed = parse_multi_classifier_output('{"topic": "sports", "sentiment": "positive"}')

        assert format_multi_classifier_output(parsed) == "{'topic': 'sports', 'sentiment': 'positive'}"

    @pytest.mark.parametrize("output_string", [
        '["positive", "sports"]',  # a list, not an object
        '"positive"',  # a bare string
        "42",  # a bare number
        "null",  # JSON null
    ])
    def test_non_object_output_raises_error(self, output_string):
        """Test that an output string that is not a key-value object raises an error."""
        with pytest.raises(MultiClassifierError, match="mapping keys to values"):
            parse_multi_classifier_output(output_string)

    @pytest.mark.parametrize("output_string", [
        "{sentiment: positive, topic: sports",  # unbalanced and unquoted
        "positive and sports",  # plain text
        "",  # empty string
    ])
    def test_unparseable_output_raises_error(self, output_string):
        """Test that an output string that is neither JSON nor a dict literal raises an error."""
        with pytest.raises(MultiClassifierError, match="valid JSON"):
            parse_multi_classifier_output(output_string)

    def test_non_string_keys_raise_error(self):
        """Test that a dict literal with non-string keys raises an error."""
        with pytest.raises(MultiClassifierError, match="keys must be strings"):
            parse_multi_classifier_output("{1: 'positive'}")


class TestMultiClassifierSampleNormalization:
    """Test cases for how add_sample normalizes the stored output format."""

    @staticmethod
    def _add(instruction, output_string, input_string="The cat sits in the tree"):
        """Adds a single sample and returns its stored output string."""
        instruction.add_sample(
            input_snippets=[SIMPLE_TOKENSET.create_snippet(input_string)],
            output_snippet=instruction.output.tokenset.create_snippet(output_string),
        )
        return instruction.samples[-1].output

    @pytest.mark.parametrize("output_string", [
        '{"sentiment": "positive", "topic": "sports"}',
        "{'sentiment': 'positive', 'topic': 'sports'}",
        '{ "sentiment": "positive",  "topic": "sports" }',
        '{"sentiment": \'positive\', "topic": "sports"}',
    ])
    def test_stored_output_is_canonical(self, output_string):
        """Test that every accepted input format is stored in the canonical single-quoted form."""
        instruction = _make_instruction()

        assert self._add(instruction, output_string) == "{'sentiment': 'positive', 'topic': 'sports'}"

    def test_plain_string_output_snippet_is_canonical(self):
        """Test that passing the output as a plain string (not a Snippet) is normalized the same way."""
        instruction = _make_instruction()

        instruction.add_sample(
            input_snippets=["The cat sits in the tree"],
            output_snippet=json.dumps({"sentiment": "positive", "topic": "sports"}),
        )

        assert instruction.samples[0].output == "{'sentiment': 'positive', 'topic': 'sports'}"

    def test_value_with_single_quote_is_not_mangled(self):
        """Test that a value containing an apostrophe is escaped rather than corrupted."""
        instruction = _make_instruction(state_map={"sentiment": ["it's positive"], "topic": ["sports"]})

        stored = self._add(instruction, json.dumps({"sentiment": "it's positive", "topic": "sports"}))

        assert parse_multi_classifier_output(stored) == {"sentiment": "it's positive", "topic": "sports"}

    def test_value_with_double_quote_is_not_mangled(self):
        """Test that a value containing a double quote is escaped rather than corrupted."""
        instruction = _make_instruction(state_map={"sentiment": ['say "positive"'], "topic": ["sports"]})

        stored = self._add(instruction, json.dumps({"sentiment": 'say "positive"', "topic": "sports"}))

        assert parse_multi_classifier_output(stored) == {"sentiment": 'say "positive"', "topic": "sports"}

    def test_stored_output_is_reparseable_as_a_new_sample(self):
        """Test that a stored output can be fed back into add_sample, as the protocol loader does."""
        instruction = _make_instruction()
        stored = self._add(instruction, json.dumps({"sentiment": "positive", "topic": "sports"}))

        reloaded = _make_instruction()

        assert self._add(reloaded, stored) == stored
