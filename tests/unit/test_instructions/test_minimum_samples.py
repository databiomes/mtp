"""
Unit tests for minimum sample validation in Instructions.
"""
import pytest

from model_train_protocol import Instruction, FinalToken
from model_train_protocol.common.instructions.input.InstructionInput import InstructionInput
from model_train_protocol.common.instructions.output.InstructionOutput import InstructionOutput
from model_train_protocol.v2 import ProtocolV2
from tests.fixtures.tokens import SIMPLE_TOKENSET


class TestMinimumSamples:
    """Test cases for minimum sample validation."""

    def test_instruction_with_less_than_minimum_samples_fails(self):
        """Test that adding an instruction with less than 3 samples raises ValueError."""
        protocol = ProtocolV2("test_protocol", inputs=2)
        
        # Add minimum context lines
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        
        # Create instruction
        final_token = FinalToken("Continue")
        instruction_input = InstructionInput(tokensets=[SIMPLE_TOKENSET, SIMPLE_TOKENSET])
        instruction_output = InstructionOutput(tokenset=SIMPLE_TOKENSET, final=final_token)
        instruction = Instruction(name='instruction_1', 
            input=instruction_input,
            output=instruction_output
        )
        
        # Add only 2 samples (less than minimum of 3)
        instruction.add_sample(
            input_snippets=["Context 1", "Context 2"],
            output_snippet="Output 1"
        )
        instruction.add_sample(
            input_snippets=["Context 3", "Context 4"],
            output_snippet="Output 2"
        )
        
        # Attempting to add instruction with less than 3 samples should fail
        with pytest.raises(ValueError, match="Instruction must have at least three samples"):
            protocol.add_instruction(instruction)

    def test_instruction_with_minimum_samples_passes(self):
        """Test that adding an instruction with exactly 3 samples succeeds."""
        protocol = ProtocolV2("test_protocol", inputs=2)
        
        # Add minimum context lines
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        
        # Create instruction
        final_token = FinalToken("Continue")
        instruction_input = InstructionInput(tokensets=[SIMPLE_TOKENSET, SIMPLE_TOKENSET])
        instruction_output = InstructionOutput(tokenset=SIMPLE_TOKENSET, final=final_token)
        instruction = Instruction(name='instruction_1', 
            input=instruction_input,
            output=instruction_output,
        )
        
        # Add exactly 3 samples (minimum required)
        instruction.add_sample(
            input_snippets=["Context 1", "Context 2"],
            output_snippet="Output 1"
        )
        instruction.add_sample(
            input_snippets=["Context 3", "Context 4"],
            output_snippet="Output 2"
        )
        instruction.add_sample(
            input_snippets=["Context 5", "Context 6"],
            output_snippet="Output 3"
        )
        
        # Adding instruction with exactly 3 samples should succeed
        protocol.add_instruction(instruction)
        assert len(protocol.instructions) == 1

    def test_instruction_with_more_than_minimum_samples_passes(self):
        """Test that adding an instruction with more than 3 samples succeeds."""
        protocol = ProtocolV2("test_protocol", inputs=2)
        
        # Add minimum context lines
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        
        # Create instruction
        final_token = FinalToken("Continue")
        instruction_input = InstructionInput(tokensets=[SIMPLE_TOKENSET, SIMPLE_TOKENSET])
        instruction_output = InstructionOutput(tokenset=SIMPLE_TOKENSET, final=final_token)
        instruction = Instruction(name='instruction_1', 
            input=instruction_input,
            output=instruction_output
        )
        
        # Add 5 samples (more than minimum of 3)
        for i in range(5):
            instruction.add_sample(
                input_snippets=[f"Context {i*2+1}", f"Context {i*2+2}"],
                output_snippet=f"Output {i+1}"
            )
        
        # Adding instruction with more than 3 samples should succeed
        protocol.add_instruction(instruction)
        assert len(protocol.instructions) == 1
        assert len(instruction.samples) == 5

    def test_instruction_with_multiple_final_tokens_insufficient_samples_fails(self):
        """Test that adding an instruction where a FinalToken has less than 3 samples raises ValueError."""
        protocol = ProtocolV2("test_protocol", inputs=2)
        
        # Add minimum context lines
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        
        # Create instruction with multiple final tokens
        final_token_1 = FinalToken("Appear")
        final_token_2 = FinalToken("Vanish")
        instruction_input = InstructionInput(tokensets=[SIMPLE_TOKENSET, SIMPLE_TOKENSET])
        instruction_output = InstructionOutput(tokenset=SIMPLE_TOKENSET, final=[final_token_1, final_token_2])
        instruction = Instruction(name='instruction_1', 
            input=instruction_input,
            output=instruction_output
        )
        
        # Add 3 samples for final_token_1 (minimum)
        instruction.add_sample(
            input_snippets=["Context 1", "Context 2"],
            output_snippet="Output 1",
            final=final_token_1
        )
        instruction.add_sample(
            input_snippets=["Context 3", "Context 4"],
            output_snippet="Output 2",
            final=final_token_1
        )
        instruction.add_sample(
            input_snippets=["Context 5", "Context 6"],
            output_snippet="Output 3",
            final=final_token_1
        )
        
        # Add only 2 samples for final_token_2 (less than minimum of 3)
        instruction.add_sample(
            input_snippets=["Context 7", "Context 8"],
            output_snippet="Output 4",
            final=final_token_2
        )
        instruction.add_sample(
            input_snippets=["Context 9", "Context 10"],
            output_snippet="Output 5",
            final=final_token_2
        )
        
        # Attempting to add instruction where final_token_2 has less than 3 samples should fail
        with pytest.raises(ValueError, match="Missing minimum 3 samples for each FinalToken"):
            protocol.add_instruction(instruction)

    def test_instruction_with_multiple_final_tokens_sufficient_samples_passes(self):
        """Test that adding an instruction where all FinalTokens have at least 3 samples succeeds."""
        protocol = ProtocolV2("test_protocol", inputs=2)
        
        # Add minimum context lines
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        
        # Create instruction with multiple final tokens
        final_token_1 = FinalToken("Appear")
        final_token_2 = FinalToken("Vanish")
        instruction_input = InstructionInput(tokensets=[SIMPLE_TOKENSET, SIMPLE_TOKENSET])
        instruction_output = InstructionOutput(tokenset=SIMPLE_TOKENSET, final=[final_token_1, final_token_2])
        instruction = Instruction(name='instruction_1', 
            input=instruction_input,
            output=instruction_output
        )
        
        # Add 3 samples for final_token_1 (minimum)
        instruction.add_sample(
            input_snippets=["Context 1", "Context 2"],
            output_snippet="Output 1",
            final=final_token_1
        )
        instruction.add_sample(
            input_snippets=["Context 3", "Context 4"],
            output_snippet="Output 2",
            final=final_token_1
        )
        instruction.add_sample(
            input_snippets=["Context 5", "Context 6"],
            output_snippet="Output 3",
            final=final_token_1
        )
        
        # Add 3 samples for final_token_2 (minimum)
        instruction.add_sample(
            input_snippets=["Context 7", "Context 8"],
            output_snippet="Output 4",
            final=final_token_2
        )
        instruction.add_sample(
            input_snippets=["Context 9", "Context 10"],
            output_snippet="Output 5",
            final=final_token_2
        )
        instruction.add_sample(
            input_snippets=["Context 11", "Context 12"],
            output_snippet="Output 6",
            final=final_token_2
        )
        
        # Adding instruction where all FinalTokens have at least 3 samples should succeed
        protocol.add_instruction(instruction)
        assert len(protocol.instructions) == 1

    def test_instruction_with_multiple_final_tokens_more_than_minimum_samples_passes(self):
        """Test that adding an instruction where FinalTokens have more than 3 samples succeeds."""
        protocol = ProtocolV2("test_protocol", inputs=2)
        
        # Add minimum context lines
        for i in range(10):
            protocol.add_context(f"Context line {i+1}")
        
        # Create instruction with multiple final tokens
        final_token_1 = FinalToken("Appear")
        final_token_2 = FinalToken("Vanish")
        instruction_input = InstructionInput(tokensets=[SIMPLE_TOKENSET, SIMPLE_TOKENSET])
        instruction_output = InstructionOutput(tokenset=SIMPLE_TOKENSET, final=[final_token_1, final_token_2])
        instruction = Instruction(name='instruction_1', 
            input=instruction_input,
            output=instruction_output
        )
        
        # Add 5 samples for final_token_1 (more than minimum)
        for i in range(5):
            instruction.add_sample(
                input_snippets=[f"Context {i*2+1}", f"Context {i*2+2}"],
                output_snippet=f"Output {i+1}",
                final=final_token_1
            )
        
        # Add 4 samples for final_token_2 (more than minimum)
        for i in range(4):
            instruction.add_sample(
                input_snippets=[f"Context {i*2+11}", f"Context {i*2+12}"],
                output_snippet=f"Output {i+6}",
                final=final_token_2
            )
        
        # Adding instruction where all FinalTokens have more than 3 samples should succeed
        protocol.add_instruction(instruction)
        assert len(protocol.instructions) == 1
        assert len(instruction.samples) == 9

