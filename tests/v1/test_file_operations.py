"""
Integration tests for file operations.
"""
import json
from pathlib import Path

import pytest

from model_train_protocol.v1 import ProtocolV1


class TestFileOperations:
    """Integration tests for file operations."""

    def test_protocol_save_basic(self, temp_directory, simple_workflow_instruction_with_samples):
        """Test basic protocol saving."""
        protocol: ProtocolV1 = ProtocolV1(name="file_test", inputs=2, encrypt=False)
        # Add context (minimum 10 lines total required)
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        protocol.add_instruction(simple_workflow_instruction_with_samples)

        # Save protocol
        protocol.save(name="test_save", path=str(temp_directory))

        # Check files were created
        model_file = temp_directory / "test_save_model.json"
        assert model_file.exists()

        # Check file content
        with open(model_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["name"] == "file_test"
        
        # Clean up
        model_file.unlink()

    def test_protocol_template_basic(self, temp_directory, simple_workflow_instruction_with_samples):
        """Test basic protocol templating."""
        protocol: ProtocolV1 = ProtocolV1(name="file_test", inputs=2, encrypt=False)
        # Add context (minimum 10 lines total required)
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        protocol.add_instruction(simple_workflow_instruction_with_samples)

        # Create template
        protocol.template(path=str(temp_directory))

        # Check files were created
        template_file = temp_directory / "file_test_template.json"
        assert template_file.exists()

        # Check file content
        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Template files have different structure - check for expected keys
        assert "example_usage" in data
        assert "tokens" in data
        assert "instructions" in data
        
        # Clean up
        template_file.unlink()

    def test_protocol_template_non_state_machine_states_empty(
        self,
        temp_directory,
        simple_workflow_instruction_with_samples,
    ):
        """Test non-state machine protocol template has empty states."""
        protocol: ProtocolV1 = ProtocolV1(name="non_state_machine", inputs=2, encrypt=False)
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        protocol.add_instruction(simple_workflow_instruction_with_samples)

        protocol.template(path=str(temp_directory))

        template_file = temp_directory / "non_state_machine_template.json"
        assert template_file.exists()

        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["state_machine"] is False
        assert data["states"] == []

        template_file.unlink()

    def test_protocol_template_state_machine_includes_states(self, temp_directory, state_machine_protocol):
        """Test state machine protocol templating includes states."""
        state_machine_protocol.template(path=str(temp_directory))

        template_file = temp_directory / "state_machine_protocol_template.json"
        assert template_file.exists()

        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["state_machine"] is True
        assert data["states"]
        assert "Action 0" in data["states"]

        template_file.unlink()

    def test_protocol_template(self, temp_directory, simple_workflow_instruction_with_samples, user_workflow_instruction_with_samples):
        """Test protocol template with guardrails."""
        protocol = ProtocolV1("guardrail_test", inputs=2)
        # Add context (minimum 10 lines total required)
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        protocol.add_instruction(simple_workflow_instruction_with_samples)
        protocol.add_instruction(user_workflow_instruction_with_samples)

        # Create template
        protocol.template(path=str(temp_directory))

        # Check file was created
        template_file = temp_directory / "guardrail_test_template.json"
        assert template_file.exists()

        # Check file content
        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Template files have different structure - check for expected keys
        assert "example_usage" in data
        assert "tokens" in data
        assert "instructions" in data
        
        # Clean up
        template_file.unlink()

    def test_protocol_save_with_numeric_tokens(self, temp_directory, simple_workflow_instruction_with_samples):
        """Test protocol saving with numeric tokens."""
        protocol = ProtocolV1("numeric_test", inputs=2)
        # Add context (minimum 10 lines total required)
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        protocol.add_instruction(simple_workflow_instruction_with_samples)

        # Save protocol
        protocol.save(name="numeric_save", path=str(temp_directory))

        # Check file was created
        model_file = temp_directory / "numeric_save_model.json"
        assert model_file.exists()

        # Check file content
        with open(model_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["name"] == "numeric_test"
        assert "numbers" not in data
        
        # Clean up
        model_file.unlink()

    def test_protocol_save_encrypted(self, temp_directory, simple_workflow_instruction_with_samples):
        """Test protocol saving with encryption."""
        protocol = ProtocolV1("encrypted_test", inputs=2, encrypt=True)
        # Add context (minimum 10 lines total required)
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        protocol.add_instruction(simple_workflow_instruction_with_samples)

        # Save protocol
        protocol.save(name="encrypted_save", path=str(temp_directory))

        # Check file was created
        model_file = temp_directory / "encrypted_save_model.json"
        assert model_file.exists()

        # Check file content
        with open(model_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["name"] == "encrypted_test"
        assert len(data["tokens"]) >= 2
        
        # Clean up
        model_file.unlink()

    def test_protocol_save_unencrypted(self, temp_directory, simple_workflow_instruction_with_samples):
        """Test protocol saving without encryption."""
        protocol = ProtocolV1("unencrypted_test", inputs=2, encrypt=False)
        # Add context (minimum 10 lines total required)
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        protocol.add_instruction(simple_workflow_instruction_with_samples)

        # Save protocol
        protocol.save(name="unencrypted_save", path=str(temp_directory))

        # Check file was created
        model_file = temp_directory / "unencrypted_save_model.json"
        assert model_file.exists()

        # Check file content
        with open(model_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["name"] == "unencrypted_test"
        assert len(data["tokens"]) >= 2
        
        # Clean up
        model_file.unlink()

    def test_protocol_save_multiple_instructions(self, temp_directory, user_workflow_instruction_with_samples, simple_workflow_instruction_with_samples):
        """Test protocol saving with multiple instructions."""
        protocol = ProtocolV1("multi_instruction_test", inputs=2)
        # Add context (minimum 10 lines total required)
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        protocol.add_instruction(user_workflow_instruction_with_samples)
        protocol.add_instruction(simple_workflow_instruction_with_samples)

        # Save protocol
        protocol.save(name="multi_save", path=str(temp_directory))

        # Check file was created
        model_file = temp_directory / "multi_save_model.json"
        assert model_file.exists()

        # Check file content
        with open(model_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["name"] == "multi_instruction_test"
        assert len(data["instruction"]) == 2
        
        # Clean up
        model_file.unlink()

    def test_protocol_save_default_path(self, simple_workflow_instruction_with_samples):
        """Test protocol saving with default path."""
        protocol = ProtocolV1("default_path_test", inputs=2)
        # Add context (minimum 10 lines total required)
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        protocol.add_instruction(simple_workflow_instruction_with_samples)

        # Save protocol with default path
        protocol.save(name="default_path_save")

        # Check file was created in current directory
        model_file = Path("default_path_save_model.json")
        assert model_file.exists()

        # Clean up
        model_file.unlink()

    def test_protocol_template_default_path(self, simple_workflow_instruction_with_samples):
        """Test protocol templating with default path."""
        protocol = ProtocolV1("default_template_test", inputs=2)
        # Add context (minimum 10 lines total required)
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        protocol.add_instruction(simple_workflow_instruction_with_samples)

        # Create template with default path
        protocol.template()

        # Check file was created in current directory
        template_file = Path("default_template_test_template.json")
        assert template_file.exists()

        # Clean up
        template_file.unlink()

    def test_protocol_save_error_handling(self, temp_directory):
        """Test protocol saving error handling."""
        protocol = ProtocolV1("error_test", inputs=2)

        # Add context but no instructions
        protocol.add_context("Context line 1")
        protocol.add_context("Context line 2")

        # Should raise error when trying to save without instructions
        with pytest.raises(ValueError, match="No instructions have been added"):
            protocol.save(name="error_save", path=str(temp_directory))

    def test_protocol_template_error_handling(self, temp_directory):
        """Test protocol templating error handling."""
        protocol = ProtocolV1("template_error_test", inputs=2)

        # Add context but no instructions
        protocol.add_context("Context line 1")
        protocol.add_context("Context line 2")

        # Should raise error when trying to create template without instructions
        with pytest.raises(ValueError, match="No instructions have been added"):
            protocol.template(path=str(temp_directory))
