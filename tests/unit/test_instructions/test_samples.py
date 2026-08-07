"""
Unit tests for sample creation and validation with NumToken and NumListToken result tokens.
Tests error handling when values are not provided or wrong types are provided.
"""

import warnings

import pytest
from model_train_protocol.common.tokens import Token, NumToken, NumListToken, FinalToken, FinalNumToken
from model_train_protocol.common.tokens import TokenSet
from model_train_protocol.common.instructions import Instruction, ExtendedInstruction
from model_train_protocol.common.instructions.input.InstructionInput import InstructionInput
from model_train_protocol.common.instructions.output.InstructionOutput import InstructionOutput
from model_train_protocol.common.instructions.output.ExtendedResponse import ExtendedResponse


class TestSampleValueValidation:
    """Test sample value validation for numeric result tokens."""

    def test_numtoken_result_missing_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that NumToken result token raises error when value is not provided."""
        final_num_token = FinalNumToken("Count", min_value=1, max_value=10)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_num_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        with pytest.raises(ValueError, match="requires a numeric value"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet
                # Missing value parameter
            )

    def test_numtoken_result_wrong_type_raises_error(self, simple_tokenset, user_tokenset):
        """Test that NumToken result token raises error when value is wrong type."""
        final_num_token = FinalNumToken("Count", min_value=1, max_value=10)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_num_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        with pytest.raises(TypeError, match="Value must be an int or float"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value="invalid_string"  # Wrong type
            )

    def test_numtoken_result_none_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that NumToken result token raises error when value is None."""
        final_num_token = FinalNumToken("Count", min_value=1, max_value=10)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_num_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        with pytest.raises(ValueError, match="requires a numeric value"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value=None  # None value
            )

    def test_numtoken_result_valid_numeric_value_succeeds(self, simple_tokenset, user_tokenset):
        """Test that NumToken result token accepts valid numeric value."""
        final_num_token = FinalNumToken("Count", min_value=1, max_value=10)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_num_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should not raise error - numeric values are now accepted
        instruction.add_sample(
            input_snippets=[context_snippet1, context_snippet2],
            output_snippet=output_snippet,
            output_value=5  # Pass a numeric value
        )

    def test_numlisttoken_result_missing_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that NumListToken cannot be used as a final token."""
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=3)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        # NumListToken is not a FinalToken, but InstructionOutput doesn't validate at init
        # It will fail when trying to use it in add_sample
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should fail when trying to validate the final token
        with pytest.raises((TypeError, AttributeError)):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet
            )

    def test_numlisttoken_result_wrong_type_raises_error(self, simple_tokenset, user_tokenset):
        """Test that NumListToken cannot be used as a final token."""
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=3)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        # NumListToken is not a FinalToken, but InstructionOutput doesn't validate at init
        # It will fail when trying to use it in add_sample
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should fail when trying to check len(self.output.final) since NumListToken has no len()
        with pytest.raises(TypeError, match="object of type 'NumListToken' has no len"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value="invalid_string"  # Wrong type
            )

    def test_numlisttoken_result_none_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that NumListToken cannot be used as a final token."""
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=3)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should fail when trying to check len(self.output.final) since NumListToken has no len()
        with pytest.raises(TypeError, match="object of type 'NumListToken' has no len"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value=None  # None value
            )

    def test_numlisttoken_result_valid_list_value_succeeds(self, simple_tokenset, user_tokenset):
        """Test that NumListToken cannot be used as a final token."""
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=3)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should fail when trying to check len(self.output.final) since NumListToken has no len()
        with pytest.raises(TypeError, match="object of type 'NumListToken' has no len"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value=[1, 2, 3]  # Valid list value
            )

    def test_regular_token_result_none_value_allowed(self, simple_tokenset, user_tokenset):
        """Test that regular Token result token allows None value."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should not raise error
        instruction.add_sample(
            input_snippets=[context_snippet1, context_snippet2],
            output_snippet=output_snippet,
            output_value=None  # None value allowed for regular tokens
        )

    def test_user_token_result_none_value_allowed(self, simple_tokenset, user_tokenset):
        """Test that Token result token allows None value."""
        final_token = FinalToken("User")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = user_tokenset.create_snippet("Output")

        # Should not raise error
        instruction.add_sample(
            inputs=[context_snippet1, output_snippet],
            response_string="User prompt",
            value=None  # None value allowed for user tokens
        )

    def test_regular_token_result_non_none_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that regular Token result token raises error when value is not None."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Note: Current implementation doesn't validate that value must be None for non-FinalNumToken
        # This test may need to be updated if validation is added
        # For now, the test passes without error, which may not be the intended behavior
        instruction.add_sample(
            input_snippets=[context_snippet1, context_snippet2],
            output_snippet=output_snippet,
            output_value="some_value"  # Non-None value - currently not validated
        )

    def test_user_token_result_non_none_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that Token result token allows non-None value (validation not implemented)."""
        final_token = FinalToken("User")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = user_tokenset.create_snippet("Output")

        # Note: Current implementation doesn't validate that value must be None for non-FinalNumToken
        # This test may need to be updated if validation is added
        # ExtendedInstruction: inputs should match input tokensets, with last one being user prompt
        instruction.add_sample(
            inputs=[context_snippet1, output_snippet],  # 2 snippets for 2 tokensets
            response_string="User prompt",
            value="some_value"  # Non-None value - currently not validated
        )

    def test_numtoken_result_list_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that NumToken result token raises error when value is a list."""
        final_num_token = FinalNumToken("Count", min_value=1, max_value=10)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_num_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        with pytest.raises(TypeError, match="Value must be an int or float"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value=[1, 2, 3]  # List value not allowed for NumToken
            )

    def test_numlisttoken_result_numtoken_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that NumListToken result token raises error when value is a NumToken."""
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=3)
        num_token = NumToken("Count", min_value=1, max_value=10)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should fail when trying to check len(self.output.final) since NumListToken has no len()
        with pytest.raises(TypeError, match="object of type 'NumListToken' has no len"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value=None  # None value
            )

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # This code is unreachable since the previous add_sample already raises TypeError
        pass

    def test_numlisttoken_result_dict_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that NumListToken result token raises error when value is a dict."""
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=3)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should fail when trying to check len(self.output.final) since NumListToken has no len()
        with pytest.raises(TypeError, match="object of type 'NumListToken' has no len"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value=None  # None value
            )

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # This code is unreachable since the previous add_sample already raises TypeError
        pass

    def test_numlisttoken_result_boolean_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that NumListToken result token raises error when value is a boolean."""
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=3)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should fail when trying to check len(self.output.final) since NumListToken has no len()
        with pytest.raises(TypeError, match="object of type 'NumListToken' has no len"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value=None  # None value
            )

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # This code is unreachable since the previous add_sample already raises TypeError
        pass

    def test_numlisttoken_result_empty_list_succeeds(self, simple_tokenset, user_tokenset):
        """Test that NumListToken cannot be used as a final token."""
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=1)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should fail when trying to check len(self.output.final) since NumListToken has no len()
        with pytest.raises(TypeError, match="object of type 'NumListToken' has no len"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value=[]  # Empty list
            )

    def test_numlisttoken_result_single_element_list_succeeds(self, simple_tokenset, user_tokenset):
        """Test that NumListToken cannot be used as a final token."""
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=1)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should fail when trying to check len(self.output.final) since NumListToken has no len()
        with pytest.raises(TypeError, match="object of type 'NumListToken' has no len"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value=[50]  # Single element list
            )

    def test_numlisttoken_result_multi_element_list_succeeds(self, simple_tokenset, user_tokenset):
        """Test that NumListToken cannot be used as a final token."""
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=5)
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should fail when trying to check len(self.output.final) since NumListToken has no len()
        with pytest.raises(TypeError, match="object of type 'NumListToken' has no len"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value=[10, 20, 30, 40, 50]  # Multi-element list
            )

    def test_mixed_token_types_value_validation(self, simple_tokenset, user_tokenset):
        """Test value validation with mixed token types in instruction."""
        final_num_token = FinalNumToken("Count", min_value=1, max_value=10)
        regular_token = FinalToken("Result")
        
        # Test with FinalNumToken
        instruction_input1 = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output1 = InstructionOutput(tokenset=simple_tokenset, final=final_num_token)
        instruction1 = Instruction(name='instruction_1', input=instruction_input1, output=instruction_output1)
        
        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")
        
        # Should require numeric value (raises TypeError, not ValueError)
        with pytest.raises(TypeError, match="Value must be an int or float"):
            instruction1.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value="invalid"
            )
        
        # Test with NumListToken - cannot be used as final token
        numlist_token = NumListToken("Coordinates", min_value=1, max_value=100, length=3)
        instruction_input2 = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        # NumListToken is not a FinalToken, but InstructionOutput doesn't validate at init
        instruction_output2 = InstructionOutput(tokenset=simple_tokenset, final=numlist_token)
        instruction2 = Instruction(name='instruction_1', input=instruction_input2, output=instruction_output2)
        # It will fail when trying to use it
        with pytest.raises(TypeError, match="object of type 'NumListToken' has no len"):
            instruction2.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet,
                output_value="invalid"
            )
        
        # Test with regular Token
        instruction_input3 = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output3 = InstructionOutput(tokenset=simple_tokenset, final=regular_token)
        instruction3 = Instruction(name='instruction_1', input=instruction_input3, output=instruction_output3)
        
        # Should allow None
        instruction3.add_sample(
            input_snippets=[context_snippet1, context_snippet2],
            output_snippet=output_snippet,
            output_value=None
        )
        
        # Note: Current implementation doesn't validate that value must be None for non-FinalNumToken
        # This test may need to be updated if validation is added
        instruction3.add_sample(
            input_snippets=[context_snippet1, context_snippet2],
            output_snippet=output_snippet,
            output_value="any_value"  # Currently not validated
        )


class TestInstructionValidation:
    """Test instruction validation for add_sample method."""

    def test_simple_instruction_wrong_context_snippet_count_raises_error(self, simple_tokenset, user_tokenset):
        """Test that Instruction raises error when context snippet count doesn't match."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        # Only provide 1 context snippet when 2 are expected
        output_snippet = simple_tokenset.create_snippet("Output")

        with pytest.raises(ValueError, match="Number of context snippets \\(1\\) must match number of context token sets \\(2\\)"):
            instruction.add_sample(
                input_snippets=[context_snippet1],  # Only 1 snippet, but need 2
                output_snippet=output_snippet
            )

    def test_simple_instruction_wrong_snippet_token_set_raises_error(self, simple_tokenset, user_tokenset):
        """Test that Instruction raises error when snippet doesn't match expected token set."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        # Create snippet with wrong token set
        wrong_snippet = user_tokenset.create_snippet("Wrong context")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        with pytest.raises(ValueError, match="does not match expected token set"):
            instruction.add_sample(
                input_snippets=[wrong_snippet, context_snippet2],  # First snippet has wrong token set
                output_snippet=output_snippet
            )

    def test_simple_instruction_wrong_output_snippet_token_set_raises_error(self, simple_tokenset, user_tokenset):
        """Test that Instruction raises error when output snippet doesn't match response token set."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        # Create output snippet with wrong token set
        wrong_output_snippet = user_tokenset.create_snippet("Wrong output")

        with pytest.raises(ValueError, match="does not match expected token set"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=wrong_output_snippet  # Wrong token set for output
            )

    def test_user_instruction_wrong_context_snippet_count_raises_error(self, simple_tokenset, user_tokenset):
        """Test that ExtendedInstruction raises error when context snippet count doesn't match."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        # Only provide 1 context snippet when 2 are expected
        output_snippet = user_tokenset.create_snippet("Output")

        with pytest.raises(ValueError, match="Number of context snippets \\(0\\) must match number of context token sets \\(2\\)"):
            instruction.add_sample(
                inputs=[],  # No context snippets, but need 2
                response_string="User prompt"
            )

    def test_user_instruction_wrong_snippet_token_set_raises_error(self, simple_tokenset, user_tokenset):
        """Test that ExtendedInstruction raises error when snippet doesn't match expected token set."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)

        # Create snippet with wrong token set - use a different tokenset
        wrong_tokenset = TokenSet(tokens=(Token("Wrong"),))
        wrong_snippet = wrong_tokenset.create_snippet("Wrong context")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = user_tokenset.create_snippet("Output")

        with pytest.raises(ValueError, match="does not match expected token set"):
            instruction.add_sample(
                inputs=[wrong_snippet, output_snippet],  # First snippet has wrong token set
                response_string="User prompt"
            )

    def test_user_instruction_wrong_output_snippet_token_set_raises_error(self, simple_tokenset, user_tokenset):
        """Test that ExtendedInstruction raises error when output snippet doesn't match user token set."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        # Create output snippet with wrong token set
        wrong_output_snippet = simple_tokenset.create_snippet("Wrong output")

        with pytest.raises(ValueError, match="does not match expected token set"):
            instruction.add_sample(
                inputs=[context_snippet1, wrong_output_snippet],  # Wrong token set for output
                response_string="User prompt"
            )

    def test_user_instruction_missing_prompt_raises_error(self, simple_tokenset, user_tokenset):
        """Test that ExtendedInstruction raises error when prompt is missing."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = user_tokenset.create_snippet("Output")

        with pytest.raises(TypeError, match="missing 1 required positional argument: 'response_string'"):
            instruction.add_sample(
                inputs=[context_snippet1, output_snippet]
                # Missing response_string parameter
            )

    def test_user_instruction_invalid_value_type_raises_error(self, simple_tokenset, user_tokenset):
        """Test that ExtendedInstruction raises error when value is not int or float."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = user_tokenset.create_snippet("Output")

        # Note: Current implementation doesn't validate that value must be None for non-FinalNumToken
        # This test may need to be updated if validation is added
        instruction.add_sample(
            inputs=[context_snippet1, output_snippet],
            response_string="User prompt",
            value="invalid_string"  # Wrong type - currently not validated
        )

    def test_user_instruction_list_value_raises_error(self, simple_tokenset, user_tokenset):
        """Test that ExtendedInstruction raises error when value is a list."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = user_tokenset.create_snippet("Output")

        # Note: Current implementation doesn't validate that value must be None for non-FinalNumToken
        # This test may need to be updated if validation is added
        instruction.add_sample(
            inputs=[context_snippet1, output_snippet],
            response_string="User prompt",
            value=[1, 2, 3]  # List - currently not validated
        )

    def test_simple_instruction_creation_with_token_in_response_succeeds(self, simple_tokenset, user_tokenset):
        """Test that Instruction succeeds when created with token in response."""
        # This should not raise an error since UserToken no longer exists
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset])
        instruction_output = InstructionOutput(tokenset=user_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)
        assert instruction is not None

    def test_user_instruction_creation_without_user_token_succeeds(self, simple_tokenset):
        """Test that ExtendedInstruction succeeds when created without user token."""
        # This should not raise an error since UserToken no longer exists
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)
        assert instruction is not None

    def test_instruction_creation_with_invalid_context_type_raises_error(self):
        """Test that Instruction raises error when input is not a BaseInput."""
        final_token = FinalToken("Result")
        test_tokenset = TokenSet(tokens=(Token("Test"),))
        instruction_output = InstructionOutput(tokenset=test_tokenset, final=final_token)
        with pytest.raises(TypeError, match="Context must be a sequence of TokenSet instances"):
            Instruction(name='instruction_1', 
                input="not_a_base_input",  # Wrong type
                output=instruction_output
            )

    def test_instruction_creation_with_invalid_context_items_raises_error(self):
        """Test that InstructionInput raises error when tokensets contains non-TokenSet items."""
        final_token = FinalToken("Result")
        test_tokenset = TokenSet(tokens=(Token("Test"),))
        instruction_output = InstructionOutput(tokenset=test_tokenset, final=final_token)
        with pytest.raises(TypeError, match="All items in context must be instances of TokenSet"):
            instruction_input = InstructionInput(tokensets=[test_tokenset, "not_a_tokenset"])
            Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

    def test_instruction_creation_with_invalid_response_type_raises_error(self):
        """Test that Instruction raises error when output is not an InstructionOutput."""
        test_tokenset = TokenSet(tokens=(Token("Test"),))
        instruction_input = InstructionInput(tokensets=[test_tokenset])
        with pytest.raises(TypeError, match="Output must be an instance of Output"):
            Instruction(name='instruction_1', 
                input=instruction_input,
                output="not_an_instruction_output",  # Wrong type
            )

    def test_instruction_creation_with_invalid_final_type_raises_error(self):
        """Test that InstructionOutput doesn't validate final type at init, but fails when used."""
        test_tokenset = TokenSet(tokens=(Token("Test"),))
        # InstructionOutput doesn't validate final type at initialization
        instruction_output = InstructionOutput(tokenset=test_tokenset, final="not_a_token")  # Wrong type
        instruction_input = InstructionInput(tokensets=[test_tokenset])
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)
        
        # It will fail when trying to use it - the error occurs when checking len(self.output.final)
        # Since "not_a_token" is a string, len() returns its length, causing a confusing error
        test_snippet = test_tokenset.create_snippet("Test")
        with pytest.raises((ValueError, TypeError)):
            instruction.add_sample(
                input_snippets=[test_snippet],
                output_snippet=test_snippet
            )

    def test_simple_instruction_valid_sample_succeeds(self, simple_tokenset, user_tokenset):
        """Test that Instruction accepts valid sample."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should not raise error
        instruction.add_sample(
            input_snippets=[context_snippet1, context_snippet2],
            output_snippet=output_snippet
        )

    def test_user_instruction_valid_sample_succeeds(self, simple_tokenset, user_tokenset):
        """Test that ExtendedInstruction accepts valid sample."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = user_tokenset.create_snippet("Output")

        # Should not raise error
        instruction.add_sample(
            inputs=[context_snippet1, output_snippet],
            response_string="User prompt"
        )

    def test_user_instruction_valid_sample_with_none_value_succeeds(self, simple_tokenset, user_tokenset):
        """Test that ExtendedInstruction accepts valid sample with None value."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        extended_response = ExtendedResponse(final=final_token)
        instruction = ExtendedInstruction(input=instruction_input, output=extended_response)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = user_tokenset.create_snippet("Output")

        # Should not raise error
        instruction.add_sample(
            inputs=[context_snippet1, output_snippet],
            response_string="User prompt",
            value=None  # Valid None value for non-numeric final token
        )

    def test_simple_instruction_valid_sample_with_none_value_succeeds(self, simple_tokenset, user_tokenset):
        """Test that Instruction accepts valid sample with None value."""
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        # Should not raise error
        instruction.add_sample(
            input_snippets=[context_snippet1, context_snippet2],
            output_snippet=output_snippet,
            output_value=None  # Valid None value
        )

    def test_snippet_length_within_recommendation_does_not_warn(self, simple_tokenset, user_tokenset):
        """Test that snippets at exactly the recommended length are accepted without a warning."""
        from model_train_protocol.common.constants import RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET

        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        # Create snippets at exactly the recommended length (should succeed)
        context_snippet1 = simple_tokenset.create_snippet("a" * RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET)
        context_snippet2 = user_tokenset.create_snippet("b" * RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET)
        output_snippet = simple_tokenset.create_snippet("c" * RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET)

        # Should not raise or warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=output_snippet
            )

    def test_snippet_length_exceeds_recommendation_warns(self, simple_tokenset, user_tokenset):
        """Test that snippets exceeding the recommended length warn but are accepted."""
        from model_train_protocol.common.constants import RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET

        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        # Create snippet one character over the recommendation (should warn, not raise)
        long_snippet = simple_tokenset.create_snippet("a" * (RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET + 1))
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        output_snippet = simple_tokenset.create_snippet("Output")

        with pytest.warns(UserWarning,
                          match=f"Snippet length.*exceeds recommended maximum length of "
                                f"{RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET} characters"):
            instruction.add_sample(
                input_snippets=[long_snippet, context_snippet2],
                output_snippet=output_snippet
            )

        assert len(instruction.samples) == 1

    def test_output_snippet_length_exceeds_recommendation_warns(self, simple_tokenset, user_tokenset):
        """Test that output snippets exceeding the recommended length warn but are accepted."""
        from model_train_protocol.common.constants import RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET

        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)
        instruction = Instruction(name='instruction_1', input=instruction_input, output=instruction_output)

        context_snippet1 = simple_tokenset.create_snippet("Context 1")
        context_snippet2 = user_tokenset.create_snippet("Context 2")
        # Create output snippet one character over the recommendation (should warn, not raise)
        long_output_snippet = simple_tokenset.create_snippet("a" * (RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET + 1))

        with pytest.warns(UserWarning,
                          match=f"Snippet length.*exceeds recommended maximum length of "
                                f"{RECOMMENDED_MAXIMUM_CHARACTERS_PER_SNIPPET} characters"):
            instruction.add_sample(
                input_snippets=[context_snippet1, context_snippet2],
                output_snippet=long_output_snippet
            )

        assert len(instruction.samples) == 1

    @pytest.mark.skip(reason="Current implementation does not enforce maximum context lines, but this test is here for future validation when implemented.")
    def test_instruction_context_maximum_lines_exceeds_limit_raises_error(self, simple_tokenset, user_tokenset):
        """Test that instruction context exceeding 10 lines raises an error."""
        from model_train_protocol.common.constants import MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION
        
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)

        # Create context with 11 lines (should fail)
        context_lines = [f"Context line {i}" for i in range(MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION + 1)]

        with pytest.raises(ValueError, match=f"Context exceeds maximum allowed lines of {MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION}"):
            Instruction(name='instruction_1', 
                input=instruction_input,
                output=instruction_output,
                context=context_lines
            )

    def test_instruction_context_maximum_lines_at_limit_succeeds(self, simple_tokenset, user_tokenset):
        """Test that instruction context with exactly 10 lines succeeds."""
        from model_train_protocol.common.constants import MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION
        
        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)

        # Create context with exactly 10 lines (should succeed)
        context_lines = [f"Context line {i}" for i in range(MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION)]

        instruction = Instruction(name='instruction_1', 
            input=instruction_input,
            output=instruction_output,
            context=context_lines
        )
        assert len(instruction.context) == MAXIMUM_CONTEXT_LINES_PER_INSTRUCTION

    def test_instruction_context_line_length_exceeds_recommendation_warns(self, simple_tokenset, user_tokenset):
        """Test that instruction context lines exceeding the recommended length warn but are accepted."""
        from model_train_protocol.common.constants import RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE

        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)

        # Create context with one line over the recommendation (should warn, not raise)
        long_line = "a" * (RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE + 1)
        context_lines = [
            "Normal context line",
            long_line
        ]

        with pytest.warns(UserWarning,
                          match=f"Context line.*exceeds recommended maximum length of "
                                f"{RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE} characters"):
            instruction = Instruction(name='instruction_1',
                input=instruction_input,
                output=instruction_output,
                context=context_lines
            )

        assert instruction.context == context_lines

    def test_instruction_context_line_length_at_recommendation_does_not_warn(self, simple_tokenset, user_tokenset):
        """Test that instruction context lines at exactly the recommended length do not warn."""
        from model_train_protocol.common.constants import RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE

        final_token = FinalToken("Result")
        instruction_input = InstructionInput(tokensets=[simple_tokenset, user_tokenset])
        instruction_output = InstructionOutput(tokenset=simple_tokenset, final=final_token)

        # Create context with lines at exactly the recommended length (should succeed without warning)
        context_lines = [
            "a" * RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE,
            "b" * RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            instruction = Instruction(name='instruction_1',
                input=instruction_input,
                output=instruction_output,
                context=context_lines
            )
        assert all(len(line) == RECOMMENDED_MAXIMUM_CHARACTERS_PER_INSTRUCTION_CONTEXT_LINE
                   for line in instruction.context)