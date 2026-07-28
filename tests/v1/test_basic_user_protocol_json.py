"""
Test JSON creation for basic user protocol.
"""


class TestBasicUserProtocolJSON:
    """Test JSON structure and content for basic user protocol."""

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

    def test_basic_user_protocol_json_structure(self, basic_user_protocol):
        """Test that the JSON has the correct top-level structure."""
        # Get the JSON output
        json_output = self._get_json_output(basic_user_protocol)
        
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

        assert json_output["state_machine"] == basic_user_protocol.state_machine

    def test_basic_user_protocol_name(self, basic_user_protocol):
        """Test that the protocol name is correct."""
        json_output = self._get_json_output(basic_user_protocol)
        
        assert json_output["name"] == "basic_user"

    def test_basic_user_protocol_context(self, basic_user_protocol):
        """Test that the context is correctly included."""
        json_output = self._get_json_output(basic_user_protocol)
        
        assert "context" in json_output
        assert isinstance(json_output["context"], list)
        assert len(json_output["context"]) == 10
        assert json_output["context"][0] == "This is a user context line."

    def test_basic_user_protocol_tokens(self, basic_user_protocol):
        """Test that tokens are correctly included."""
        json_output = self._get_json_output(basic_user_protocol)
        
        assert "tokens" in json_output
        assert isinstance(json_output["tokens"], dict)
        
        # Check that we have the expected tokens
        token_keys = set(json_output["tokens"].keys())
        expected_tokens = {"Tree_", "English_", "Alice_", "Talk_", "Result_"}
        special_tokens = {"<BOS>", "<EOS>", "<PAD>", "<RUN>", "<UNK>"}
        
        # Check that we have at least some expected tokens (not all may be present)
        assert len(expected_tokens.intersection(token_keys)) > 0, f"Expected at least some of {expected_tokens} to be present in {token_keys}"
        # Check that at least some special tokens are present
        assert len(special_tokens.intersection(token_keys)) > 0, f"Expected at least some of {special_tokens} to be present in {token_keys}"
        
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

    def test_basic_user_protocol_token_types(self, basic_user_protocol):
        """Test that token types are correctly identified."""
        json_output = self._get_json_output(basic_user_protocol)
        
        tokens = json_output["tokens"]
        
        # Alice should be a regular token now
        if "Alice_" in tokens:
            assert tokens["Alice_"]["num"] is False
        
        # Other tokens should be regular tokens
        for token_key, token_info in tokens.items():
            if token_key not in ["<BOS>", "<EOS>", "<PAD>", "<RUN>", "<UNK>"]:
                assert token_info["num"] is False

    def test_basic_user_protocol_special_tokens(self, basic_user_protocol):
        """Test that special tokens are correctly included."""
        json_output = self._get_json_output(basic_user_protocol)
        
        assert "special_tokens" in json_output
        assert isinstance(json_output["special_tokens"], list)
        
        # Should include instruction tokens
        assert len(json_output["special_tokens"]) > 0
        assert all(isinstance(token, str) for token in json_output["special_tokens"])

    def test_basic_user_protocol_instruction(self, basic_user_protocol):
        """Test that instruction structure is correct."""
        json_output = self._get_json_output(basic_user_protocol)
        
        assert "instruction" in json_output
        instruction = json_output["instruction"]
        
        # Test instruction top-level keys
        assert "memory" in instruction
        assert "sets" in instruction
        
        # Test memory (should be instruction_context_snippets + 1)
        assert instruction["memory"] == 3  # 2 context lines + 1 response line
        
        # Test sets structure
        assert isinstance(instruction["sets"], list)
        assert len(instruction["sets"]) == 1  # One instruction set
        
        instruction_set = instruction["sets"][0]
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
        assert len(instruction_set["set"]) == 3  # Three context lines (2 context + 1 response)
        assert isinstance(instruction_set["set"][0], list)
        assert len(instruction_set["set"][0]) > 0  # Should have tokens
        
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
        assert "number_lists" in sample
        assert "result" in sample
        assert "value" in sample
        
        # Test sample data types
        assert isinstance(sample["strings"], list)
        assert isinstance(sample["prompt"], str)
        assert isinstance(sample["numbers"], (list, type(None)))
        assert isinstance(sample["number_lists"], (list, type(None)))
        assert isinstance(sample["result"], str)
        assert isinstance(sample["value"], (str, type(None)))
        
        # Test sample content
        assert len(sample["strings"]) == 3  # Three context snippets (2 context + 1 response)
        assert sample["result"] == "End__"
        assert sample["value"] is None  # No value for user instruction
        
        # User instruction should have prompts
        assert len(sample["prompt"]) > 0

    def test_basic_user_protocol_empty_guardrails(self, basic_user_protocol):
        """Test that guardrails are correctly included in instruction sets."""
        json_output = self._get_json_output(basic_user_protocol)
        
        # Basic user protocol should have no guardrails
        sets = json_output["instruction"]["sets"]
        for instruction_set in sets:
            assert "guardrails" in instruction_set
            assert isinstance(instruction_set["guardrails"], list)
            assert len(instruction_set["guardrails"]) == 0

    def test_basic_user_protocol_one_guardrail(self, basic_user_protocol_with_guardrail):
        """Test that guardrails are correctly included in instruction sets."""
        json_output = self._get_json_output(basic_user_protocol_with_guardrail)

        # Basic user protocol should have guardrails
        sets = json_output["instruction"]["sets"]
        guardrails_found = False
        for instruction_set in sets:
            assert "guardrails" in instruction_set
            assert isinstance(instruction_set["guardrails"], list)
            if len(instruction_set["guardrails"]) > 0:
                guardrails_found = True
                # Check guardrail structure
                guardrail = instruction_set["guardrails"][0]
                assert "index" in guardrail
                assert "bad_output" in guardrail
                assert "bad_prompt" in guardrail
                assert "good_prompt" in guardrail
                assert "bad_examples" in guardrail
                assert isinstance(guardrail["bad_examples"], list)
                assert len(guardrail["bad_examples"]) >= 3
        assert guardrails_found, "Expected to find at least one guardrail in instruction sets"

    def test_basic_user_protocol_numbers(self, basic_user_protocol):
        """Test that numbers are correctly included."""
        json_output = self._get_json_output(basic_user_protocol)
        
        assert "numbers" not in json_output

    def test_basic_user_protocol_batches(self, basic_user_protocol):
        """Test that batches are correctly included."""
        json_output = self._get_json_output(basic_user_protocol)
        
        assert "batches" not in json_output

    def test_basic_user_protocol_token_descriptions(self, basic_user_protocol):
        """Test that token descriptions are correctly included."""
        json_output = self._get_json_output(basic_user_protocol)
        
        tokens = json_output["tokens"]
        
        # Check specific token descriptions
        if "Tree" in tokens:
            assert tokens["Tree"]["desc"] == "A tree token"
        if "English" in tokens:
            assert tokens["English"]["desc"] == "English language token"
        if "Alice" in tokens:
            assert tokens["Alice"]["desc"] == "User token"
        if "Talk" in tokens:
            assert tokens["Talk"]["desc"] == "A talk token"
        if "Result" in tokens:
            assert tokens["Result"]["desc"] == "Result token"

    def test_basic_user_protocol_token_types_final(self, basic_user_protocol):
        """Test that token types are correctly set."""
        json_output = self._get_json_output(basic_user_protocol)
        
        tokens = json_output["tokens"]
        
        # Alice should be a regular token, others should be regular tokens
        for token_key, token_info in tokens.items():
            if token_key in ["<BOS>", "<EOS>", "<PAD>", "<RUN>", "<UNK>", "<NON>"]:
                # Special tokens
                assert token_info["special"] is not None
            else:
                assert token_info["num"] is False
                assert token_info["special"] is None
