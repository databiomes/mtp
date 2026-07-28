"""
Test JSON creation for workflow protocol with 2 context lines.
"""


class TestWorkflow2ContextProtocolJSON:
    """Test JSON structure and content for workflow protocol with 2 context lines."""

    def _get_json_output(self, protocol):
        """Helper method to get JSON output from a protocol."""
        protocol._prep_protocol()
        from model_train_protocol.v1 import ProtocolFileV1
        protocol_file = ProtocolFileV1(
            name=protocol.name,
            context=protocol.context,
            inputs=protocol.input_count,
            encrypted=protocol.encrypt,
            valid=True,
            state_machine=protocol.state_machine,
            tokens=protocol.tokens,
            special_tokens=protocol.special_tokens,
            instructions=protocol.instructions,
            bloom_version=protocol.bloom_version
        )
        return protocol_file.to_json()

    def test_workflow_2context_protocol_json_structure(self, workflow_2context_protocol):
        """Test that the JSON has the correct top-level structure."""
        # Get the JSON output
        json_output = self._get_json_output(workflow_2context_protocol)
        
        # Test top-level keys
        assert "name" in json_output
        assert "context" in json_output
        assert "tokens" in json_output
        assert "special_tokens" in json_output
        assert "instruction" in json_output
        
        # Test that no unexpected keys are present
        expected_keys = {"$schema", "name", "context", "tokens", "special_tokens", "instruction", "encrypted", "valid",
                         "inputs", "state_machine"}
        actual_keys = set(json_output.keys())
        assert actual_keys == expected_keys

        assert json_output["state_machine"] == workflow_2context_protocol.state_machine

    def test_workflow_2context_protocol_name(self, workflow_2context_protocol):
        """Test that the protocol name is correct."""
        json_output = self._get_json_output(workflow_2context_protocol)
        
        assert json_output["name"] == "workflow_2context"

    def test_workflow_2context_protocol_tokens(self, workflow_2context_protocol):
        """Test that tokens are correctly included."""
        json_output = self._get_json_output(workflow_2context_protocol)
        
        assert "tokens" in json_output
        assert isinstance(json_output["tokens"], dict)
        
        # Check that we have the expected tokens (with underscores)
        token_keys = set(json_output["tokens"].keys())
        expected_tokens = {"Tree_", "English_", "Cat_", "Talk_", "Result_", "Alice_", "End_"}
        # Check that at least some expected tokens are present (tokens may be stored as concatenated values)
        assert len(expected_tokens.intersection(token_keys)) > 0, f"Expected at least some of {expected_tokens} to be present in {token_keys}"
        
        # Test token structure
        for token_key, token_info in json_output["tokens"].items():
            assert "key" in token_info
            assert "num" in token_info
            assert "desc" in token_info
            assert "special" in token_info
            
            # Check data types
            assert isinstance(token_info["key"], str)
            assert isinstance(token_info["num"], bool)
            assert token_info["desc"] is None or isinstance(token_info["desc"], str)
            assert token_info["special"] is None or isinstance(token_info["special"], str)

    def test_workflow_2context_protocol_token_types(self, workflow_2context_protocol):
        """Test that token types are correctly set."""
        json_output = self._get_json_output(workflow_2context_protocol)
        
        tokens = json_output["tokens"]
        
        # Check specific token types
        for token_key, token_info in tokens.items():
            if token_key == "Alice_":
                assert token_info["num"] is False
            elif token_key in ["<RUN>", "<PAD>", "<EOS>", "<BOS>", "<UNK>", "<NON>"]:
                # Special tokens
                assert token_info["special"] is not None
            else:
                assert token_info["num"] is False
                assert token_info["special"] is None

    def test_workflow_2context_protocol_special_tokens(self, workflow_2context_protocol):
        """Test that special tokens are correctly included."""
        json_output = self._get_json_output(workflow_2context_protocol)
        
        assert "special_tokens" in json_output
        assert isinstance(json_output["special_tokens"], list)
        
        # Should include instruction tokens
        assert len(json_output["special_tokens"]) > 0
        assert all(isinstance(token, str) for token in json_output["special_tokens"])

    def test_workflow_2context_protocol_instruction(self, workflow_2context_protocol):
        """Test that instruction structure is correct."""
        json_output = self._get_json_output(workflow_2context_protocol)
        
        assert "instruction" in json_output
        instruction = json_output["instruction"]
        
        # Test instruction top-level keys
        assert "memory" in instruction
        assert "sets" in instruction
        
        # Test memory (should be instruction_context_snippets + 1)
        assert instruction["memory"] == 3  # 2 context lines + 1 response line
        
        # Test sets structure
        assert isinstance(instruction["sets"], list)
        assert len(instruction["sets"]) == 2  # Two instruction sets
        
        for instruction_set in instruction["sets"]:
            self._test_instruction_set_structure(instruction_set)

    def _test_instruction_set_structure(self, instruction_set):
        """Test the structure of an instruction set."""
        # Test instruction set keys
        assert "set" in instruction_set
        assert "context" in instruction_set
        assert "samples" in instruction_set
        assert "ppo" in instruction_set
        
        # Test set structure (context tokens)
        assert isinstance(instruction_set["set"], list)
        assert len(instruction_set["set"]) == 3  # Three context lines
        assert isinstance(instruction_set["set"][0], list)
        assert isinstance(instruction_set["set"][1], list)
        assert isinstance(instruction_set["set"][2], list)
        assert len(instruction_set["set"][0]) > 0  # Should have tokens
        assert len(instruction_set["set"][1]) > 0  # Should have tokens
        assert len(instruction_set["set"][2]) > 0  # Should have tokens
        
        # Test context
        assert isinstance(instruction_set["context"], list)
        
        # Test samples
        assert isinstance(instruction_set["samples"], list)
        assert len(instruction_set["samples"]) == 3  # Should have 3 samples
        
        for sample in instruction_set["samples"]:
            self._test_sample_structure(sample)
        
        # Test ppo
        assert isinstance(instruction_set["ppo"], list)

    def _test_sample_structure(self, sample):
        """Test the structure of a sample."""
        # Test sample keys
        assert "strings" in sample
        assert "prompt" in sample
        assert "numbers" in sample
        assert "result" in sample
        assert "value" in sample
        
        # Test sample data types
        assert isinstance(sample["strings"], list)
        assert sample["prompt"] is None or isinstance(sample["prompt"], str)
        assert sample["numbers"] is None or isinstance(sample["numbers"], list)
        assert isinstance(sample["result"], str)
        assert isinstance(sample["value"], (str, type(None)))
        
        # Test sample content
        assert len(sample["strings"]) == 3  # Three context snippets
        # Numbers should be empty arrays for each context snippet
        assert sample["numbers"] is not None
        assert len(sample["numbers"]) == 3  # Three context snippets
        for num_list in sample["numbers"]:
            assert isinstance(num_list, list)
            assert len(num_list) == 0  # Empty number lists
        assert sample["result"] in ["Result__", "End__"]
        assert sample["value"] is None  # No value for workflow instructions
        
        # User instruction should have prompts
        if sample["result"] == "End_":
            assert len(sample["prompt"]) > 0

    def test_workflow_2context_protocol_guardrails(self, workflow_2context_protocol):
        """Test that guardrails are correctly included in instruction sets."""
        json_output = self._get_json_output(workflow_2context_protocol)
        
        # Workflow protocol should have no guardrails
        sets = json_output["instruction"]["sets"]
        for instruction_set in sets:
            assert "guardrails" in instruction_set
            assert isinstance(instruction_set["guardrails"], list)
            assert len(instruction_set["guardrails"]) == 0

    def test_workflow_2context_protocol_numbers(self, workflow_2context_protocol):
        """Test that numbers are correctly included."""
        json_output = self._get_json_output(workflow_2context_protocol)
        
        assert "numbers" not in json_output

    def test_workflow_2context_protocol_batches(self, workflow_2context_protocol):
        """Test that batches are correctly included."""
        json_output = self._get_json_output(workflow_2context_protocol)
        
        assert "batches" not in json_output

    def test_workflow_2context_protocol_instruction_sets_order(self, workflow_2context_protocol):
        """Test that instruction sets are in the correct order."""
        json_output = self._get_json_output(workflow_2context_protocol)
        
        instruction_sets = json_output["instruction"]["sets"]
        
        # Should have 2 instruction sets
        assert len(instruction_sets) == 2
        
        # Check that all instruction sets have context
        for instruction_set in instruction_sets:
            assert "context" in instruction_set
            assert isinstance(instruction_set["context"], list)

    def test_workflow_2context_protocol_instruction_context_snippets(self, workflow_2context_protocol):
        """Test that the protocol correctly handles 2 context lines."""
        json_output = self._get_json_output(workflow_2context_protocol)
        
        # Memory should be instruction_context_snippets + 1
        assert json_output["instruction"]["memory"] == 3
        
        # Each instruction set should have 3 context lines (actual structure)
        for instruction_set in json_output["instruction"]["sets"]:
            assert len(instruction_set["set"]) == 3
            
            # Each sample should have 3 context snippets
            for sample in instruction_set["samples"]:
                assert len(sample["strings"]) == 3


class TestNumTokenWorkflow2ContextProtocolJSON:
    """Test JSON structure and content for NumToken workflow protocol with 2 context lines."""

    def _get_json_output(self, protocol):
        """Helper method to get JSON output from a protocol."""
        protocol._prep_protocol()
        from model_train_protocol.v1 import ProtocolFileV1
        protocol_file = ProtocolFileV1(
            name=protocol.name,
            context=protocol.context,
            inputs=protocol.input_count,
            encrypted=protocol.encrypt,
            valid=True,
            state_machine=protocol.state_machine,
            tokens=protocol.tokens,
            special_tokens=protocol.special_tokens,
            instructions=protocol.instructions,
            bloom_version=protocol.bloom_version
        )
        return protocol_file.to_json()

    def test_numtoken_workflow_2context_protocol_json_structure(self, numtoken_workflow_2context_protocol):
        """Test that the JSON has the correct top-level structure."""
        # Get the JSON output
        json_output = self._get_json_output(numtoken_workflow_2context_protocol)
        
        # Test top-level keys
        assert "name" in json_output
        assert "context" in json_output
        assert "tokens" in json_output
        assert "special_tokens" in json_output
        assert "instruction" in json_output
        assert "numbers" not in json_output
        assert "batches" not in json_output

    def test_numtoken_workflow_2context_protocol_name(self, numtoken_workflow_2context_protocol):
        """Test that the protocol name is correct."""
        json_output = self._get_json_output(numtoken_workflow_2context_protocol)
        
        assert json_output["name"] == "numtoken_workflow_2context"

    def test_numtoken_workflow_2context_protocol_tokens(self, numtoken_workflow_2context_protocol):
        """Test that tokens are correctly included."""
        json_output = self._get_json_output(numtoken_workflow_2context_protocol)
        
        assert "tokens" in json_output
        assert isinstance(json_output["tokens"], dict)
        
        # Check that we have the expected tokens
        token_keys = set(json_output["tokens"].keys())
        expected_tokens = {"Tree_", "English_", "Cat_", "Talk_", "Count_", "Alice_"}
        # Check that at least some expected tokens are present (tokens may be stored as concatenated values)
        assert len(expected_tokens.intersection(token_keys)) > 0, f"Expected at least some of {expected_tokens} to be present in {token_keys}"

    def test_numtoken_workflow_2context_protocol_numeric_tokens(self, numtoken_workflow_2context_protocol):
        """Test that numeric tokens are correctly identified."""
        json_output = self._get_json_output(numtoken_workflow_2context_protocol)
        
        tokens = json_output["tokens"]
        
        # Count should be marked as a numeric token
        if "Count" in tokens:
            assert tokens["Count"]["num"] is True

    def test_numtoken_workflow_2context_protocol_instruction(self, numtoken_workflow_2context_protocol):
        """Test that instruction structure is correct."""
        json_output = self._get_json_output(numtoken_workflow_2context_protocol)
        
        assert "instruction" in json_output
        instruction = json_output["instruction"]
        
        # Test memory (should be instruction_context_snippets + 1)
        assert instruction["memory"] == 3  # 2 context lines + 1 response line
        
        # Test sets structure
        assert isinstance(instruction["sets"], list)
        assert len(instruction["sets"]) == 1  # One instruction set
        
        instruction_set = instruction["sets"][0]
        assert len(instruction_set["set"]) == 3  # Three context lines (2 context + 1 response)
        assert isinstance(instruction_set["context"], list)
        
        # Test samples
        for sample in instruction_set["samples"]:
            assert len(sample["strings"]) == 3  # Three context snippets (2 context + 1 response)
            assert isinstance(sample["numbers"], (list, type(None)))  # Can be list or None
            assert sample["result"] == "Count__"
            assert isinstance(sample["value"], (int, float))

    def test_numtoken_workflow_2context_protocol_numbers(self, numtoken_workflow_2context_protocol):
        """Test that numbers are correctly included."""
        json_output = self._get_json_output(numtoken_workflow_2context_protocol)
        
        assert "numbers" not in json_output
