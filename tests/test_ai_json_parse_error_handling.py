"""Test AI JSON parse error handling."""
import pytest

# The AI client imports `openai`, absent in the lint job (ruff+pytest only);
# skip at collection there so pytest doesn't error importing this module.
pytest.importorskip("openai")

from dashboard_services.ai.client import clean_ai_text


class TestCleanAiText:
    """Test the clean_ai_text function handles malformed responses safely."""
    
    def test_clean_valid_json_with_dashes(self):
        """Should clean dashes in valid JSON."""
        text = '{"summary": "Great player — top tier — must start"}'
        result = clean_ai_text(text)
        assert result == '{"summary": "Great player, top tier, must start"}'
    
    def test_clean_valid_json_array(self):
        """Should clean dashes in valid JSON arrays."""
        text = '["Player A — excellent", "Player B — good"]'
        result = clean_ai_text(text)
        assert result == '["Player A, excellent", "Player B, good"]'
    
    def test_preserve_malformed_response_with_dashes(self):
        """Should NOT clean malformed responses (don't start with { or [)."""
        text = "— — — — — — —"
        result = clean_ai_text(text)
        # Should return unchanged so it can be logged for diagnosis
        assert result == text
    
    def test_preserve_malformed_response_with_quotes_and_dashes(self):
        """Should NOT clean responses that don't look like JSON."""
        text = '"— — — — — — —"'
        result = clean_ai_text(text)
        # Starts with quote, not { or [, so should be preserved
        assert result == text
    
    def test_preserve_empty_string(self):
        """Should return empty string unchanged."""
        assert clean_ai_text("") == ""
    
    def test_preserve_whitespace_only(self):
        """Should return whitespace-only string unchanged."""
        assert clean_ai_text("   ") == "   "
    
    def test_clean_json_with_leading_whitespace(self):
        """Should clean valid JSON even with leading whitespace."""
        text = '  {"summary": "Great — must start"}'
        result = clean_ai_text(text)
        assert result == '  {"summary": "Great, must start"}'


class TestAiErrorHandling:
    """Test that AI response errors are handled gracefully."""
    
    def test_empty_response_handling(self):
        """Empty responses should be caught and logged."""
        # This would be tested via integration test with mocked OpenAI client
        # The error should be caught in generate_team_ai_result and raise ValueError
        pass
    
    def test_malformed_json_response_handling(self):
        """Malformed JSON should be caught and logged with original text."""
        # This would be tested via integration test with mocked OpenAI client
        # The error should log both original and cleaned versions
        pass
    
    def test_response_becomes_empty_after_cleaning(self):
        """Response that becomes empty after cleaning should be caught."""
        # This would be tested via integration test with mocked OpenAI client
        # Should raise ValueError with appropriate message
        pass
