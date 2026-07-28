"""
Integration tests for complete protocol workflows.
"""
import pytest

from model_train_protocol.v1 import ProtocolV1


class TestProtocolWorkflow:
    """Integration tests for complete protocol workflows."""

    def test_complete_protocol_creation(self, simple_workflow_instruction_with_samples, user_workflow_instruction_with_samples):
        """Test creating a complete protocol with all components using fixtures."""
        protocol = ProtocolV1("integration_test", inputs=2)
        
        # Add context (minimum 10 lines total required)
        protocol.add_context("This is a test context line 1")
        protocol.add_context("This is a test context line 2")
        protocol.add_context("This is a test context line 3")
        protocol.add_context("This is a test context line 4")
        protocol.add_context("This is a test context line 5")
        protocol.add_context("This is a test context line 6")
        protocol.add_context("This is a test context line 7")
        protocol.add_context("This is a test context line 8")
        protocol.add_context("This is a test context line 9")
        protocol.add_context("This is a test context line 10")
        
        # Add instructions to protocol
        protocol.add_instruction(simple_workflow_instruction_with_samples)
        protocol.add_instruction(user_workflow_instruction_with_samples)
        
        # Verify protocol state
        assert len(protocol.instructions) == 2
        assert len(protocol.tokens) >= 6  # Should include all tokens
        assert len(protocol.context) == 10
        assert protocol.name == "integration_test"
        assert protocol.input_count == 2

    def test_protocol_with_numeric_tokens(self, simple_numtoken_workflow_instruction_with_samples):
        """Test creating a protocol with numeric tokens using fixtures."""
        protocol = ProtocolV1("numeric_test", inputs=2)
        
        # Add context (minimum 10 lines total required)
        protocol.add_context("This protocol uses numeric tokens.")
        protocol.add_context("Numbers are important for calculations.")
        protocol.add_context("This is a third context line.")
        protocol.add_context("This is a fourth context line.")
        protocol.add_context("This is a fifth context line.")
        protocol.add_context("This is a sixth context line.")
        protocol.add_context("This is a seventh context line.")
        protocol.add_context("This is an eighth context line.")
        protocol.add_context("This is a ninth context line.")
        protocol.add_context("This is a tenth context line.")
        
        protocol.add_instruction(simple_numtoken_workflow_instruction_with_samples)
        
        # Verify protocol state
        assert len(protocol.instructions) == 1
        assert len(protocol.tokens) >= 3
        assert protocol.name == "numeric_test"

    def test_protocol_save_and_load_workflow(self, temp_directory, simple_workflow_instruction_with_samples):
        """Test complete save and load workflow using fixtures."""
        protocol = ProtocolV1("save_test", inputs=2)
        
        # Add context (minimum 10 lines total required)
        protocol.add_context("Context line 1")
        protocol.add_context("Context line 2")
        protocol.add_context("Context line 3")
        protocol.add_context("Context line 4")
        protocol.add_context("Context line 5")
        protocol.add_context("Context line 6")
        protocol.add_context("Context line 7")
        protocol.add_context("Context line 8")
        protocol.add_context("Context line 9")
        protocol.add_context("Context line 10")
        
        protocol.add_instruction(simple_workflow_instruction_with_samples)
        
        # Save protocol
        protocol.save(path=str(temp_directory))
        
        # Create template
        protocol.template(path=str(temp_directory))
        
        # Check files were created
        model_file = temp_directory / "save_test_model.json"
        template_file = temp_directory / "save_test_template.json"
        
        assert model_file.exists()
        assert template_file.exists()
        
        # Clean up
        model_file.unlink()
        template_file.unlink()

    def test_protocol_with_guardrails_workflow(self, user_workflow_instruction_with_samples):
        """Test protocol workflow with guardrails using fixtures."""
        protocol = ProtocolV1("guardrail_test", inputs=2)
        
        # Add context (minimum 10 lines total required)
        protocol.add_context("This protocol uses guardrails.")
        protocol.add_context("Guardrails provide safety mechanisms.")
        protocol.add_context("This is a third context line.")
        protocol.add_context("This is a fourth context line.")
        protocol.add_context("This is a fifth context line.")
        protocol.add_context("This is a sixth context line.")
        protocol.add_context("This is a seventh context line.")
        protocol.add_context("This is an eighth context line.")
        protocol.add_context("This is a ninth context line.")
        protocol.add_context("This is a tenth context line.")
        
        protocol.add_instruction(user_workflow_instruction_with_samples)
        
        # Verify protocol state
        assert len(protocol.instructions) == 1
        assert len(protocol.tokens) >= 3

    def test_protocol_encryption_workflow(self, simple_workflow_instruction_with_samples):
        """Test protocol workflow with encryption using fixtures."""
        protocol = ProtocolV1("encryption_test", inputs=2, encrypt=True)
        
        # Add context (minimum 10 lines total required)
        protocol.add_context("This protocol uses encryption.")
        protocol.add_context("Keys are generated automatically.")
        protocol.add_context("This is a third context line.")
        protocol.add_context("This is a fourth context line.")
        protocol.add_context("This is a fifth context line.")
        protocol.add_context("This is a sixth context line.")
        protocol.add_context("This is a seventh context line.")
        protocol.add_context("This is an eighth context line.")
        protocol.add_context("This is a ninth context line.")
        protocol.add_context("This is a tenth context line.")
        
        protocol.add_instruction(simple_workflow_instruction_with_samples)
        
        # Verify protocol state
        assert len(protocol.instructions) == 1
        assert len(protocol.tokens) >= 3
        assert protocol.encrypt is True

    def test_protocol_unencrypted_workflow(self, simple_workflow_instruction_with_samples):
        """Test protocol workflow without encryption using fixtures."""
        protocol = ProtocolV1("unencrypted_test", inputs=2, encrypt=False)
        
        # Add context (minimum 10 lines total required)
        protocol.add_context("This protocol does not use encryption.")
        protocol.add_context("Keys match token values.")
        protocol.add_context("This is a third context line.")
        protocol.add_context("This is a fourth context line.")
        protocol.add_context("This is a fifth context line.")
        protocol.add_context("This is a sixth context line.")
        protocol.add_context("This is a seventh context line.")
        protocol.add_context("This is an eighth context line.")
        protocol.add_context("This is a ninth context line.")
        protocol.add_context("This is a tenth context line.")
        
        protocol.add_instruction(simple_workflow_instruction_with_samples)
        
        # Verify protocol state
        assert len(protocol.instructions) == 1
        assert len(protocol.tokens) >= 3
        assert protocol.encrypt is False

    def test_protocol_multiple_instructions_workflow(self, simple_workflow_instruction_with_samples, user_workflow_instruction_with_samples):
        """Test protocol workflow with multiple instructions using fixtures."""
        protocol = ProtocolV1("multi_instruction_test", inputs=2)
        
        # Add context (minimum 10 lines total required)
        protocol.add_context("This protocol has multiple instructions.")
        protocol.add_context("Each instruction handles different scenarios.")
        protocol.add_context("This is a third context line.")
        protocol.add_context("This is a fourth context line.")
        protocol.add_context("This is a fifth context line.")
        protocol.add_context("This is a sixth context line.")
        protocol.add_context("This is a seventh context line.")
        protocol.add_context("This is an eighth context line.")
        protocol.add_context("This is a ninth context line.")
        protocol.add_context("This is a tenth context line.")
        
        # Add instructions to protocol
        protocol.add_instruction(simple_workflow_instruction_with_samples)
        protocol.add_instruction(user_workflow_instruction_with_samples)
        
        # Verify protocol state
        assert len(protocol.instructions) == 2
        assert len(protocol.tokens) >= 6

    def test_protocol_complex_workflow(self, simple_workflow_instruction_with_samples, user_workflow_instruction_with_samples, simple_numtoken_workflow_instruction_with_samples):
        """Test complex protocol workflow with all features using fixtures."""
        protocol = ProtocolV1("complex_test", inputs=2)
        
        # Add context (minimum 10 lines total required)
        protocol.add_context("This is a complex protocol with all features.")
        protocol.add_context("It includes tokens, instructions, and guardrails.")
        protocol.add_context("This is a third context line.")
        protocol.add_context("This is a fourth context line.")
        protocol.add_context("This is a fifth context line.")
        protocol.add_context("This is a sixth context line.")
        protocol.add_context("This is a seventh context line.")
        protocol.add_context("This is an eighth context line.")
        protocol.add_context("This is a ninth context line.")
        protocol.add_context("This is a tenth context line.")
        
        # Add instructions to protocol
        protocol.add_instruction(simple_workflow_instruction_with_samples)
        protocol.add_instruction(user_workflow_instruction_with_samples)
        protocol.add_instruction(simple_numtoken_workflow_instruction_with_samples)
        
        # Verify protocol state
        assert len(protocol.instructions) == 3
        assert len(protocol.tokens) >= 8
        assert len(protocol.context) == 10
        assert protocol.name == "complex_test"
        assert protocol.input_count == 2

    def test_protocol_error_handling_workflow(self):
        """Test protocol workflow with error handling."""
        protocol = ProtocolV1("error_test", inputs=2)
        
        # Add context (minimum 10 lines total required)
        protocol.add_context("This protocol tests error handling.")
        protocol.add_context("It should handle errors gracefully.")
        protocol.add_context("This is a third context line.")
        protocol.add_context("This is a fourth context line.")
        protocol.add_context("This is a fifth context line.")
        protocol.add_context("This is a sixth context line.")
        protocol.add_context("This is a seventh context line.")
        protocol.add_context("This is an eighth context line.")
        protocol.add_context("This is a ninth context line.")
        protocol.add_context("This is a tenth context line.")
        
        # Test adding instruction without samples - create a simple instruction without samples
        from model_train_protocol import Token, TokenSet, Instruction, FinalToken
        from model_train_protocol.common.instructions.input.InstructionInput import InstructionInput
        from model_train_protocol.common.instructions.output.InstructionOutput import InstructionOutput
        
        token1 = Token("Token1", key="🔑")
        token2 = Token("Token2", key="🔧")
        context_set = TokenSet(tokens=(token1,))
        response_set = TokenSet(tokens=(token2,))
        
        # Create second context set
        context_set2 = TokenSet(tokens=(token2,))
        
        final_token = FinalToken(token2.value)
        instruction_input = InstructionInput(tokensets=[context_set, context_set2])
        instruction_output = InstructionOutput(tokenset=response_set, final=final_token)

        instruction: Instruction = Instruction(name='instruction_1', 
            input=instruction_input,
            output=instruction_output
        )

        # Attempt to add instruction without samples
        with pytest.raises(ValueError, match="Instruction must have at least three samples."):
            protocol.add_instruction(instruction)

    def test_protocol_validation_workflow(self, user_workflow_instruction_with_samples):
        """Test protocol workflow with validation using fixtures."""
        protocol = ProtocolV1("validation_test", inputs=2)
        
        # Add context (minimum 10 lines total required)
        protocol.add_context("This protocol tests validation.")
        protocol.add_context("It should validate all components.")
        protocol.add_context("This is a third context line.")
        protocol.add_context("This is a fourth context line.")
        protocol.add_context("This is a fifth context line.")
        protocol.add_context("This is a sixth context line.")
        protocol.add_context("This is a seventh context line.")
        protocol.add_context("This is an eighth context line.")
        protocol.add_context("This is a ninth context line.")
        protocol.add_context("This is a tenth context line.")
        
        protocol.add_instruction(user_workflow_instruction_with_samples)
        
        # Verify protocol state
        assert len(protocol.instructions) == 1
        assert len(protocol.tokens) >= 3
        assert protocol.name == "validation_test"
        assert protocol.input_count == 2

