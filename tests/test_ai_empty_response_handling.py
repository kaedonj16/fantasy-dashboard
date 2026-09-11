"""Test that AI functions handle empty responses gracefully."""
import pytest
from unittest.mock import Mock, patch
from dashboard_services.ai.prompts import (
    generate_team_ai_result,
    generate_trade_analysis_result,
    generate_power_rankings_result,
    generate_trade_suggestions_result,
)
from dashboard_services.ai.history_recap import generate_season_recap_result
from dashboard_services.ai.weekly_recap import generate_weekly_recap_result


@pytest.fixture
def mock_empty_response():
    """Mock OpenAI response with empty output_text."""
    mock_resp = Mock()
    mock_resp.output_text = ""
    return mock_resp


@pytest.fixture
def mock_whitespace_response():
    """Mock OpenAI response with whitespace-only output_text."""
    mock_resp = Mock()
    mock_resp.output_text = "   \n\t  "
    return mock_resp


class TestEmptyResponseHandling:
    """Test that all AI functions validate empty responses before JSON parsing."""

    def test_gm_memo_empty_response(self, mock_empty_response):
        """Test generate_team_ai_result handles empty response for gm_memo."""
        with patch("dashboard_services.ai.prompts.get_ai_client") as mock_client:
            mock_client.return_value.responses.create.return_value = mock_empty_response
            
            with pytest.raises(ValueError, match="OpenAI API returned empty response for gm_memo"):
                generate_team_ai_result({"scoring_type": "dynasty"}, mode="gm_memo")

    def test_front_office_empty_response(self, mock_empty_response):
        """Test generate_team_ai_result handles empty response for front_office_briefing."""
        with patch("dashboard_services.ai.prompts.get_ai_client") as mock_client:
            mock_client.return_value.responses.create.return_value = mock_empty_response
            
            with pytest.raises(ValueError, match="OpenAI API returned empty response for front_office_briefing"):
                generate_team_ai_result({"scoring_type": "dynasty"}, mode="front_office_briefing")

    def test_trade_analysis_empty_response(self, mock_empty_response):
        """Test generate_trade_analysis_result handles empty response."""
        with patch("dashboard_services.ai.prompts.get_ai_client") as mock_client:
            mock_client.return_value.responses.create.return_value = mock_empty_response
            
            with pytest.raises(ValueError, match="OpenAI API returned empty response for trade_analysis"):
                generate_trade_analysis_result({})

    def test_power_rankings_empty_response(self, mock_empty_response):
        """Test generate_power_rankings_result handles empty response."""
        with patch("dashboard_services.ai.prompts.get_ai_client") as mock_client:
            mock_client.return_value.responses.create.return_value = mock_empty_response
            
            with pytest.raises(ValueError, match="OpenAI API returned empty response for power_rankings"):
                generate_power_rankings_result({"teams": []}, is_redraft=False)

    def test_trade_suggestions_empty_response(self, mock_empty_response):
        """Test generate_trade_suggestions_result handles empty response."""
        with patch("dashboard_services.ai.prompts.get_ai_client") as mock_client:
            mock_client.return_value.responses.create.return_value = mock_empty_response
            
            with pytest.raises(ValueError, match="OpenAI API returned empty response for trade_suggestions"):
                generate_trade_suggestions_result({})

    def test_season_recap_empty_response(self, mock_empty_response):
        """Test generate_season_recap_result handles empty response."""
        with patch("dashboard_services.ai.history_recap.get_ai_client") as mock_client:
            mock_client.return_value.responses.create.return_value = mock_empty_response
            
            with pytest.raises(ValueError, match="OpenAI API returned empty response for season_recap"):
                generate_season_recap_result({})

    def test_weekly_recap_empty_response(self, mock_empty_response):
        """Test generate_weekly_recap_result handles empty response."""
        with patch("dashboard_services.ai.weekly_recap.get_ai_client") as mock_client:
            mock_client.return_value.responses.create.return_value = mock_empty_response
            
            with pytest.raises(ValueError, match="OpenAI API returned empty response for weekly_recap"):
                generate_weekly_recap_result({})

    def test_whitespace_only_response(self, mock_whitespace_response):
        """Test that whitespace-only responses are treated as empty."""
        with patch("dashboard_services.ai.prompts.get_ai_client") as mock_client:
            mock_client.return_value.responses.create.return_value = mock_whitespace_response
            
            with pytest.raises(ValueError, match="OpenAI API returned empty response for gm_memo"):
                generate_team_ai_result({"scoring_type": "dynasty"}, mode="gm_memo")

    def test_valid_json_response_still_works(self):
        """Test that valid JSON responses still work correctly."""
        mock_resp = Mock()
        mock_resp.output_text = '{"team_identity": "test", "outlook": "test", "strength": "test", "weakness": "test", "next_move": "test", "trade_posture": "test", "verdict": "BUY"}'
        
        with patch("dashboard_services.ai.prompts.get_ai_client") as mock_client:
            mock_client.return_value.responses.create.return_value = mock_resp
            
            result = generate_team_ai_result({"scoring_type": "dynasty"}, mode="gm_memo")
            assert isinstance(result, dict)
            assert "team_identity" in result
