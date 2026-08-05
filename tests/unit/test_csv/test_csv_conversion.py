"""
Unit tests for CSV conversion functionality.
"""

from model_train_protocol import StateMachineInstruction
from model_train_protocol.common.constants import ModelType
from model_train_protocol.common.instructions import BaseInstruction
from model_train_protocol.csv.conversion import CSVConversion, CSVLine
from model_train_protocol.errors.conversion import ConversionError
from model_train_protocol.errors.protocol import ProtocolError
from model_train_protocol.v2 import ProtocolV2
from tests.fixtures.csv_fixtures import *


class TestCSVLine:
    """Test cases for the CSVLine class."""

    def test_csvline_initialization(self):
        """Test CSVLine initialization."""
        line = CSVLine(
            input_str="Hello", 
            output_str="greeting", 
            context_str="A greeting"
        )
        assert line.input_str == "Hello"
        assert line.output_str == "greeting"
        assert line.context_str == "A greeting"

    def test_csvline_is_guardrail_true(self):
        """Test CSVLine guardrail detection when output is GUARDRAIL."""
        line = CSVLine(
            input_str="Bad input", 
            output_str="GUARDRAIL", 
            context_str="Bad context"
        )
        assert line.is_guardrail is True

    def test_csvline_is_guardrail_false(self):
        """Test CSVLine guardrail detection when output is not GUARDRAIL."""
        line = CSVLine(
            input_str="Good input", 
            output_str="normal_state", 
            context_str="Normal context"
        )
        assert line.is_guardrail is False


class TestCSVConversion:
    """Test cases for the CSVConversion class."""

    def test_initialization_basic(self, valid_csv_data):
        """Test basic CSVConversion initialization."""
        conversion = CSVConversion(valid_csv_data, "Test Protocol")
        
        assert conversion.protocol.name == "Test Protocol"
        assert conversion.protocol.input_count == 1
        assert conversion.protocol.encrypt is False
        assert len(conversion.ordered_lines) == 12

    def test_initialization_default_protocol_name(self, valid_csv_data):
        """Test CSVConversion initialization with default protocol name."""
        conversion = CSVConversion(valid_csv_data)
        assert conversion.protocol.name == "CSV Protocol"

    def test_required_columns_constant(self):
        """Test that required columns match expected values."""
        expected_columns = ["Input", "Output", "Reference"]
        assert CSVConversion.REQUIRED_COLUMNS == expected_columns

    def test_process_dataframe_removes_empty_inputs(self):
        """Test that dataframe processing removes rows with NaN inputs."""
        data = pd.DataFrame({
            'Input': ['Valid input', '', pd.NA, 'Another valid'],
            'Output': ['state1', 'state2', 'state3', 'state4'],
            'Reference': ['ref1', 'ref2', 'ref3', 'ref4']
        })
        
        conversion = CSVConversion(data)
        processed_data = conversion.csv_data
        
        # Should only have 3 rows (only NaN inputs removed, empty strings kept)
        assert len(processed_data) == 3
        assert processed_data.iloc[0]['Input'] == 'Valid input'
        assert processed_data.iloc[1]['Input'] == ''  # Empty string is kept
        assert processed_data.iloc[2]['Input'] == 'Another valid'
        
        # Test full conversion works with processed data
        protocol = conversion.to_mtp()
        assert isinstance(protocol, ProtocolV2)

    def test_format_lines_basic(self, valid_csv_data):
        """Test basic line formatting."""
        conversion = CSVConversion(valid_csv_data)
        lines = conversion.ordered_lines
        
        assert len(lines) == 12
        assert lines[0].input_str == "Hello, how are you?"
        assert lines[0].output_str == "greeting"
        assert lines[0].context_str == "A friendly greeting to start conversation"
        
        # Test full conversion works with formatted lines
        protocol = conversion.to_mtp()
        assert isinstance(protocol, ProtocolV2)

    def test_format_line_empty_output_inherits_previous(self, csv_with_empty_outputs):
        """Test that empty outputs inherit from previous line."""
        conversion = CSVConversion(csv_with_empty_outputs)
        lines = conversion.ordered_lines
        
        assert lines[0].output_str == "first_state"
        assert lines[1].output_str == "first_state"  # Inherited
        assert lines[2].output_str == "first_state"  # Inherited
        assert lines[3].output_str == "fourth_state"
        
        # Test full conversion works with inherited outputs
        protocol = conversion.to_mtp()
        assert isinstance(protocol, ProtocolV2)

    def test_format_line_nan_output_inherits_previous(self, csv_with_nan_outputs):
        """Test that NaN outputs inherit from previous line."""
        conversion = CSVConversion(csv_with_nan_outputs)
        lines = conversion.ordered_lines
        
        assert lines[0].output_str == "state1"
        assert lines[1].output_str == "state1"  # Inherited from previous
        assert lines[2].output_str == "state2"
        
        # Test full conversion works with NaN output inheritance
        protocol = conversion.to_mtp()
        assert isinstance(protocol, ProtocolV2)

    def test_format_line_first_row_empty_output_raises_error(self, csv_empty_output_first_row):
        """Test that empty output in first row raises ConversionError."""
        with pytest.raises(ConversionError, match="The first line of the CSV cannot have an empty output"):
            CSVConversion(csv_empty_output_first_row)

    def test_get_unique_states_excludes_guardrail(self, valid_csv_with_guardrails):
        """Test that unique states excludes GUARDRAIL."""
        conversion = CSVConversion(valid_csv_with_guardrails)
        unique_states = conversion.unique_states
        
        expected_states = {"greeting", "name_request", "joke_request", "help_offer", "time_request", 
                          "location_request", "assistance_request", "weather_request", 
                          "direction_request", "advice_request", "farewell"}
        assert unique_states == expected_states
        assert "GUARDRAIL" not in unique_states
        
        # Test full conversion works with guardrail exclusion
        protocol = conversion.to_mtp()
        assert isinstance(protocol, ProtocolV2)

    def test_get_unique_states_handles_empty_and_nan(self, csv_with_context_variations):
        """Test that unique states excludes empty and NaN values."""
        conversion = CSVConversion(csv_with_context_variations)
        unique_states = conversion.unique_states
        
        expected_states = {"state1", "state2", "state3", "state4"}
        assert unique_states == expected_states
        
        # Test full conversion works with context variations
        protocol = conversion.to_mtp()
        assert isinstance(protocol, ProtocolV2)

    def test_to_mtp_basic_conversion(self, valid_csv_data):
        """Test basic CSV to MTP conversion."""
        conversion = CSVConversion(valid_csv_data, "Test Protocol")
        protocol = conversion.to_mtp()
        
        assert isinstance(protocol, ProtocolV2)
        assert protocol.name == "Test Protocol"
        assert len(protocol.instructions) == 1
        
        # Check the instruction
        instruction: BaseInstruction = list(protocol.instructions)[0]
        assert isinstance(instruction, StateMachineInstruction)
        assert len(instruction.get_states()) == 12  # 12 regular states
        assert len(instruction.samples) == 12

    def test_to_mtp_with_guardrails(self, valid_csv_with_guardrails):
        """Test CSV to MTP conversion with valid guardrails."""
        conversion = CSVConversion(valid_csv_with_guardrails)
        protocol = conversion.to_mtp()
        
        instruction = list(protocol.instructions)[0]
        
        # Should have 11 regular states + GUARDRAIL = 12 total states
        assert isinstance(instruction, StateMachineInstruction)
        assert len(instruction.get_states()) == 12

        # Should have guardrail with 3 samples
        assert instruction.has_guardrails
        guardrails = instruction.get_guardrails()
        assert len(guardrails) == 1
        guardrail = guardrails[0]
        assert len(guardrail.samples) == 3
        assert guardrail.bad_output == "GUARDRAIL"

    def test_to_mtp_insufficient_guardrails_no_error(self, csv_insufficient_guardrails):
        """Test that insufficient guardrail samples doesn't raise error but no guardrails are added."""
        conversion = CSVConversion(csv_insufficient_guardrails)
        protocol = conversion.to_mtp()
        
        # Should succeed but with no guardrails added due to insufficient samples
        instruction = list(protocol.instructions)[0]
        assert not instruction.has_guardrails  # No guardrails should be added
        assert isinstance(instruction, StateMachineInstruction)
        assert len(instruction.get_states()) == 3  # Only regular states

    def test_to_mtp_with_proper_guardrails_success(self, csv_with_proper_guardrails_but_wrong_context):
        """Test that sufficient guardrail samples are added correctly."""
        conversion = CSVConversion(csv_with_proper_guardrails_but_wrong_context)
        protocol = conversion.to_mtp()
        
        # Should succeed with guardrails added
        instruction = list(protocol.instructions)[0]
        assert instruction.has_guardrails
        assert len(instruction.get_guardrails()) == 1
        assert len(instruction.get_guardrails()[0].samples) == 3
        assert isinstance(instruction, StateMachineInstruction)
        assert len(instruction.get_states()) == 4  # 3 regular states + GUARDRAIL

    def test_to_mtp_adds_context(self, csv_with_context_variations):
        """Test that non-empty context values are added to instruction."""
        conversion = CSVConversion(csv_with_context_variations)
        protocol = conversion.to_mtp()
        
        instruction = list(protocol.instructions)[0]
        # Should have context from rows 1 and 4 (rows 2 and 3 have empty/NaN context)
        assert len(instruction.context) == 2
        assert "Context 1" in instruction.context
        assert "Context 4" in instruction.context

    def test_to_mtp_adds_samples_correctly(self, valid_csv_data):
        """Test that samples are added correctly to instruction.""" 
        conversion = CSVConversion(valid_csv_data)
        protocol = conversion.to_mtp()
        
        instruction = list(protocol.instructions)[0]
        samples = instruction.samples
        
        # Check that all samples are present
        sample_inputs = [sample.input[0] for sample in samples]
        sample_states = [sample.output for sample in samples]
        
        assert "Hello, how are you?" in sample_inputs
        assert "greeting" in sample_states
        assert "What is your name?" in sample_inputs
        assert "name_request" in sample_states

    def test_large_dataset_performance(self, large_valid_csv):
        """Test conversion with large dataset."""
        conversion = CSVConversion(large_valid_csv)
        protocol = conversion.to_mtp()
        
        assert isinstance(protocol, ProtocolV2)
        assert len(conversion.ordered_lines) == 100
        
        instruction = list(protocol.instructions)[0]
        assert len(instruction.samples) == 100
        assert isinstance(instruction, StateMachineInstruction)
        assert len(instruction.get_states()) == 10  # 10 unique states

    def test_assign_latest_helper_method(self):
        """Test the _assign_latest helper method."""
        column = pd.Series(['value1', '', 'value3', pd.NA])
        
        # Test with valid value
        result = CSVConversion._assign_latest(column, 0, 'default')
        assert result == 'value1'
        
        # Test with empty value
        result = CSVConversion._assign_latest(column, 1, 'default')
        assert result == 'default'
        
        # Test with NaN value  
        result = CSVConversion._assign_latest(column, 3, 'default')
        assert result == 'default'
        
        # Test with invalid index
        result = CSVConversion._assign_latest(column, 99, 'default')
        assert result == 'default'


class TestCSVConversionEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame({
            'Input': [],
            'Output': [],
            'Reference': []
        })
        
        conversion = CSVConversion(empty_df)
        assert len(conversion.ordered_lines) == 0
        
        # Should raise ProtocolError because we need at least 3 samples
        with pytest.raises(ProtocolError, match="Instruction must have at least three samples"):
            conversion.to_mtp()

    def test_single_row_dataframe(self):
        """Test conversion with single row raises error due to minimum sample requirement."""
        single_row_df = pd.DataFrame({
            'Input': ['Single input'],
            'Output': ['single_state'],
            'Reference': ['Single reference']
        })
        
        conversion = CSVConversion(single_row_df)
        
        # Should raise ProtocolError because we need at least 3 samples
        with pytest.raises(ProtocolError, match="Instruction must have at least three samples"):
            conversion.to_mtp()

    def test_duplicate_states(self):
        """Test handling of duplicate states."""
        duplicate_states_df = pd.DataFrame({
            'Input': ['Input 1', 'Input 2', 'Input 3'],
            'Output': ['state1', 'state1', 'state2'],
            'Reference': ['Ref 1', 'Ref 2', 'Ref 3']
        })
        
        conversion = CSVConversion(duplicate_states_df)
        protocol = conversion.to_mtp()
        
        instruction = list(protocol.instructions)[0]
        assert isinstance(instruction, StateMachineInstruction)
        assert len(instruction.get_states()) == 2  # Only unique states
        assert set(instruction.get_states()) == {'state1', 'state2'}
        assert len(instruction.samples) == 3  # All samples preserved

    def test_only_guardrails(self):
        """Test dataset with only guardrail entries raises ProtocolError."""
        only_guardrails_df = pd.DataFrame({
            'Input': ['Bad 1', 'Bad 2', 'Bad 3', 'Bad 4'],
            'Output': ['GUARDRAIL', 'GUARDRAIL', 'GUARDRAIL', 'GUARDRAIL'],
            'Reference': ['Bad ref 1', 'Bad ref 2', 'Bad ref 3', 'Bad ref 4']
        })
        
        conversion = CSVConversion(only_guardrails_df)
        
        # Should raise ProtocolError because we have 0 regular samples (need at least 3)
        with pytest.raises(ProtocolError, match="Instruction must have at least three samples"):
            conversion.to_mtp()

    def test_context_with_special_characters(self):
        """Test context handling with special characters and formatting."""
        special_chars_df = pd.DataFrame({
            'Input': ['Input with "quotes"', 'Input with\nnewlines', 'Input with\ttabs'],
            'Output': ['state1', 'state2', 'state3'],
            'Reference': ['Context with "quotes"', 'Context with\nnewlines', 'Context with\ttabs']
        })
        
        conversion = CSVConversion(special_chars_df)
        protocol = conversion.to_mtp()
        
        instruction = list(protocol.instructions)[0]
        # All contexts should be preserved with special characters
        assert len(instruction.context) == 3


class TestCSVConversionValidation:
    """Test CSV conversion protocol validation requirements."""

    def test_valid_csv_protocol_save_success(self, valid_csv_data):
        """Test that valid CSV with sufficient context lines passes validation."""
        conversion = CSVConversion(valid_csv_data)
        protocol = conversion.to_mtp()
        
        # Should succeed - we have 12 context lines which is > 10
        try:
            protocol.save()
            # If we get here, save succeeded
            assert True
        except Exception as e:
            pytest.fail(f"Protocol save should have succeeded but failed with: {e}")
    
    def test_insufficient_context_lines_validation_fails(self, csv_insufficient_context_lines):
        """Test that CSV with insufficient context lines fails validation."""
        conversion = CSVConversion(csv_insufficient_context_lines)
        protocol = conversion.to_mtp()
        
        # Should fail validation due to insufficient context lines (4 < 10)
        try:
            protocol.save()
            pytest.fail("Protocol save should have failed due to insufficient context lines")
        except Exception as e:
            # Expect validation error mentioning context lines
            error_message = str(e)
            assert "context lines" in error_message.lower()
            assert "minimum required of 10" in error_message
    
    def test_valid_csv_with_guardrails_save_success(self, valid_csv_with_guardrails):
        """Test that valid CSV with guardrails and sufficient context passes validation."""
        conversion = CSVConversion(valid_csv_with_guardrails)
        protocol = conversion.to_mtp()
        
        # Should succeed - we have 14 context lines which is > 10
        try:
            protocol.save()
            # If we get here, save succeeded
            assert True
        except Exception as e:
            pytest.fail(f"Protocol save should have succeeded but failed with: {e}")


class TestCSVConversionInstructionNaming:
    """Test CSV conversion instruction naming behavior."""

    def test_csv_conversion_creates_state_machine_instruction(self, valid_csv_data):
        """Test that CSV conversion creates a StateMachineInstruction instance."""
        conversion = CSVConversion(valid_csv_data)
        protocol = conversion.to_mtp()
        
        assert len(protocol.instructions) == 1
        instruction = list(protocol.instructions)[0]
        
        # Should be a StateMachineInstruction instance
        assert isinstance(instruction, StateMachineInstruction)
        assert issubclass(type(instruction), BaseInstruction)

    def test_csv_conversion_instruction_default_name(self, valid_csv_data):
        """Test that CSV conversion instructions get default name 'StateMachineInstruction'."""
        conversion = CSVConversion(valid_csv_data)
        protocol = conversion.to_mtp()
        
        instruction = list(protocol.instructions)[0]
        assert instruction.name == "StateMachineInstruction"

    def test_csv_conversion_instruction_naming_with_guardrails(self, valid_csv_with_guardrails):
        """Test instruction naming with guardrails present."""
        conversion = CSVConversion(valid_csv_with_guardrails)
        protocol = conversion.to_mtp()
        
        instruction = list(protocol.instructions)[0]
        assert isinstance(instruction, StateMachineInstruction)
        assert instruction.name == "StateMachineInstruction"
        assert instruction.has_guardrails

    def test_csv_conversion_instruction_naming_different_protocols(self, csv_with_proper_guardrails_but_wrong_context):
        """Test instruction naming consistency across different CSV configurations."""
        conversion = CSVConversion(csv_with_proper_guardrails_but_wrong_context, "Custom Protocol Name")
        protocol = conversion.to_mtp()
        
        # Protocol name can be custom
        assert protocol.name == "Custom Protocol Name"
        
        # But instruction name should still be default
        instruction = list(protocol.instructions)[0]
        assert isinstance(instruction, StateMachineInstruction)
        assert instruction.name == "StateMachineInstruction"

    def test_csv_conversion_instruction_naming_edge_cases(self, csv_with_empty_outputs):
        """Test instruction naming with edge case data."""
        conversion = CSVConversion(csv_with_empty_outputs)
        protocol = conversion.to_mtp()
        
        instruction = list(protocol.instructions)[0]
        assert isinstance(instruction, StateMachineInstruction)
        assert instruction.name == "StateMachineInstruction"

    def test_csv_conversion_multiple_instructions_same_type(self, valid_csv_data):
        """Test that even if we could have multiple instructions, they would be StateMachineInstructions."""
        conversion = CSVConversion(valid_csv_data)
        protocol = conversion.to_mtp()
        
        # CSV conversion should only create one instruction for state machine protocols
        assert len(protocol.instructions) == 1
        
        instruction = list(protocol.instructions)[0] 
        assert isinstance(instruction, StateMachineInstruction)
        assert instruction.name == "StateMachineInstruction"
        
        # Verify it's a state machine protocol
        assert protocol.get_model_type() == ModelType.STATE_MACHINE