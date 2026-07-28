"""
Tests for validating the inner structure of guardrails in JSON output.
"""
import pytest
from model_train_protocol.v1 import ProtocolV1


class TestGuardrailJSONStructure:
    """Test cases for validating guardrail structure in JSON output."""

    def _get_json_output(self, protocol: ProtocolV1) -> dict:
        """Helper method to get JSON output from protocol."""
        import json
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            protocol.save(path=os.path.dirname(temp_path))
            filename = os.path.join(os.path.dirname(temp_path), f"{protocol.name}_model.json")
            with open(filename, 'r', encoding='utf-8') as file:
                return json.load(file)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            json_file = os.path.join(os.path.dirname(temp_path), f"{protocol.name}_model.json")
            if os.path.exists(json_file):
                os.remove(json_file)

    def test_guardrail_required_fields(self, basic_user_protocol_with_guardrail):
        """Test that each guardrail has all required fields."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        guardrail_found = False
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                guardrail_found = True
                # Check all required fields exist
                assert "index" in guardrail, "Guardrail must have 'index' field"
                assert "good_prompt" in guardrail, "Guardrail must have 'good_prompt' field"
                assert "bad_prompt" in guardrail, "Guardrail must have 'bad_prompt' field"
                assert "bad_output" in guardrail, "Guardrail must have 'bad_output' field"
                assert "bad_examples" in guardrail, "Guardrail must have 'bad_examples' field"
        
        assert guardrail_found, "Expected to find at least one guardrail"

    def test_guardrail_field_types(self, basic_user_protocol_with_guardrail):
        """Test that all guardrail fields have correct data types."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                # Check field types
                assert isinstance(guardrail["index"], int), "Guardrail 'index' must be an integer"
                assert isinstance(guardrail["good_prompt"], str), "Guardrail 'good_prompt' must be a string"
                assert isinstance(guardrail["bad_prompt"], str), "Guardrail 'bad_prompt' must be a string"
                assert isinstance(guardrail["bad_output"], str), "Guardrail 'bad_output' must be a string"
                assert isinstance(guardrail["bad_examples"], list), "Guardrail 'bad_examples' must be a list"
                
                # Check that bad_examples contains only strings
                for example in guardrail["bad_examples"]:
                    assert isinstance(example, str), f"Each item in 'bad_examples' must be a string, got {type(example)}"

    def test_guardrail_index_validity(self, basic_user_protocol_with_guardrail):
        """Test that guardrail index is a non-negative integer."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                assert guardrail["index"] >= 0, f"Guardrail index must be non-negative, got {guardrail['index']}"
                assert isinstance(guardrail["index"], int), f"Guardrail index must be an integer, got {type(guardrail['index'])}"

    def test_guardrail_string_fields_non_empty(self, basic_user_protocol_with_guardrail):
        """Test that all string fields in guardrails are non-empty."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                assert len(guardrail["good_prompt"]) > 0, "Guardrail 'good_prompt' must not be empty"
                assert len(guardrail["bad_prompt"]) > 0, "Guardrail 'bad_prompt' must not be empty"
                assert len(guardrail["bad_output"]) > 0, "Guardrail 'bad_output' must not be empty"

    def test_guardrail_bad_examples_minimum_count(self, basic_user_protocol_with_guardrail):
        """Test that guardrail bad_examples has at least 3 items."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                assert len(guardrail["bad_examples"]) >= 3, \
                    f"Guardrail 'bad_examples' must have at least 3 items, got {len(guardrail['bad_examples'])}"

    def test_guardrail_bad_examples_all_strings(self, basic_user_protocol_with_guardrail):
        """Test that all items in bad_examples are strings."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                for i, example in enumerate(guardrail["bad_examples"]):
                    assert isinstance(example, str), \
                        f"bad_examples[{i}] must be a string, got {type(example)}: {example}"

    def test_guardrail_bad_examples_non_empty_strings(self, basic_user_protocol_with_guardrail):
        """Test that all items in bad_examples are non-empty strings."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                for i, example in enumerate(guardrail["bad_examples"]):
                    assert len(example) > 0, \
                        f"bad_examples[{i}] must not be empty"

    def test_guardrail_no_extra_fields(self, basic_user_protocol_with_guardrail):
        """Test that guardrails don't have unexpected extra fields."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        expected_fields = {"index", "good_prompt", "bad_prompt", "bad_output", "bad_examples"}
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                actual_fields = set(guardrail.keys())
                extra_fields = actual_fields - expected_fields
                assert len(extra_fields) == 0, \
                    f"Guardrail has unexpected fields: {extra_fields}. Expected only: {expected_fields}"

    def test_guardrail_structure_completeness(self, basic_user_protocol_with_guardrail):
        """Test that guardrail structure is complete with all fields present."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                # Verify all required fields are present
                required_fields = ["index", "good_prompt", "bad_prompt", "bad_output", "bad_examples"]
                for field in required_fields:
                    assert field in guardrail, f"Guardrail missing required field: {field}"
                
                # Verify no None values
                assert guardrail["index"] is not None, "Guardrail 'index' must not be None"
                assert guardrail["good_prompt"] is not None, "Guardrail 'good_prompt' must not be None"
                assert guardrail["bad_prompt"] is not None, "Guardrail 'bad_prompt' must not be None"
                assert guardrail["bad_output"] is not None, "Guardrail 'bad_output' must not be None"
                assert guardrail["bad_examples"] is not None, "Guardrail 'bad_examples' must not be None"

    def test_multiple_guardrails_different_indexes(self, comprehensive_protocol):
        """Test that multiple guardrails on different indexes are correctly structured."""
        json_output = self._get_json_output(comprehensive_protocol)
        
        sets = json_output["instruction"]["sets"]
        guardrail_indices = set()
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                # Validate structure
                assert "index" in guardrail
                assert isinstance(guardrail["index"], int)
                assert guardrail["index"] >= 0
                
                # Track indices to ensure they're unique within an instruction set
                guardrail_indices.add(guardrail["index"])
                
                # Validate all other fields
                assert "good_prompt" in guardrail
                assert "bad_prompt" in guardrail
                assert "bad_output" in guardrail
                assert "bad_examples" in guardrail
                assert len(guardrail["bad_examples"]) >= 3

    def test_guardrail_in_empty_instruction_set(self, basic_simple_protocol):
        """Test that instruction sets without guardrails have empty guardrails list."""
        json_output = self._get_json_output(basic_simple_protocol)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            assert "guardrails" in instruction_set, "Instruction set must have 'guardrails' field"
            assert isinstance(instruction_set["guardrails"], list), "Guardrails must be a list"
            # For basic_simple_protocol, guardrails should be empty
            assert len(instruction_set["guardrails"]) == 0, \
                "Basic simple protocol should have no guardrails"

    def test_guardrail_index_matches_tokenset_index(self, basic_user_protocol_with_guardrail):
        """Test that guardrail index correctly corresponds to tokenset index."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            set_tokensets = instruction_set["set"]
            for guardrail in instruction_set["guardrails"]:
                guardrail_index = guardrail["index"]
                # Verify index is within valid range
                assert 0 <= guardrail_index < len(set_tokensets), \
                    f"Guardrail index {guardrail_index} is out of range for tokensets (0-{len(set_tokensets)-1})"

    def test_guardrail_with_exactly_three_examples(self, basic_user_protocol_with_guardrail):
        """Test that guardrails with exactly 3 examples are valid."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                # Guardrails should have at least 3 examples
                assert len(guardrail["bad_examples"]) >= 3
                
                # If it has exactly 3, verify they're all valid
                if len(guardrail["bad_examples"]) == 3:
                    for example in guardrail["bad_examples"]:
                        assert isinstance(example, str)
                        assert len(example) > 0

    def test_guardrail_with_more_than_three_examples(self, comprehensive_protocol):
        """Test that guardrails with more than 3 examples are correctly structured."""
        json_output = self._get_json_output(comprehensive_protocol)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                if len(guardrail["bad_examples"]) > 3:
                    # Verify all examples are valid strings
                    for i, example in enumerate(guardrail["bad_examples"]):
                        assert isinstance(example, str), \
                            f"bad_examples[{i}] must be a string when there are more than 3 examples"
                        assert len(example) > 0, \
                            f"bad_examples[{i}] must not be empty"

    def test_guardrail_string_fields_are_actual_strings(self, basic_user_protocol_with_guardrail):
        """Test that string fields are actual strings, not other types that might serialize as strings."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)
        
        sets = json_output["instruction"]["sets"]
        
        for instruction_set in sets:
            for guardrail in instruction_set["guardrails"]:
                # Verify these are actual strings, not numbers or other types
                assert isinstance(guardrail["good_prompt"], str), \
                    f"good_prompt must be str, got {type(guardrail['good_prompt'])}"
                assert isinstance(guardrail["bad_prompt"], str), \
                    f"bad_prompt must be str, got {type(guardrail['bad_prompt'])}"
                assert isinstance(guardrail["bad_output"], str), \
                    f"bad_output must be str, got {type(guardrail['bad_output'])}"
                
                # Verify they're not empty after stripping
                assert guardrail["good_prompt"].strip() != "", "good_prompt must not be empty or whitespace"
                assert guardrail["bad_prompt"].strip() != "", "bad_prompt must not be empty or whitespace"
                assert guardrail["bad_output"].strip() != "", "bad_output must not be empty or whitespace"



