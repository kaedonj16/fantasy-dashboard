"""Test that 404 errors for future weeks are handled gracefully without noisy logging."""
import logging
from unittest.mock import Mock, patch
import requests

from dashboard_services.matchups import build_matchup_preview


def test_404_logged_at_debug_level(caplog):
    """404 errors (future weeks) should log at DEBUG, not WARNING."""
    # Create a mock 404 HTTPError
    mock_response = Mock()
    mock_response.status_code = 404
    http_error = requests.exceptions.HTTPError(response=mock_response)
    
    with patch('dashboard_services.matchups.get_matchups', side_effect=http_error):
        with patch('dashboard_services.matchups.get_users', return_value=[]):
            with patch('dashboard_services.matchups.get_rosters', return_value=[]):
                with caplog.at_level(logging.DEBUG):
                    result = build_matchup_preview(
                        league_id="test_league",
                        week=10,
                        roster_map={},
                        players_map={},
                        season="2026",
                        platform="sleeper"
                    )
    
    # Should return empty list (synthesized)
    assert result == []
    
    # Should have a DEBUG log, not WARNING
    debug_logs = [r for r in caplog.records if r.levelname == 'DEBUG']
    warning_logs = [r for r in caplog.records if r.levelname == 'WARNING']
    
    assert len(debug_logs) == 1
    assert len(warning_logs) == 0
    assert "404 (future week)" in debug_logs[0].message
    assert "synthesizing" in debug_logs[0].message


def test_non_404_errors_logged_at_warning_level(caplog):
    """Non-404 errors should still log at WARNING with traceback."""
    # Create a mock 500 HTTPError
    mock_response = Mock()
    mock_response.status_code = 500
    http_error = requests.exceptions.HTTPError(response=mock_response)
    
    with patch('dashboard_services.matchups.get_matchups', side_effect=http_error):
        with patch('dashboard_services.matchups.get_users', return_value=[]):
            with patch('dashboard_services.matchups.get_rosters', return_value=[]):
                with caplog.at_level(logging.DEBUG):
                    result = build_matchup_preview(
                        league_id="test_league",
                        week=1,
                        roster_map={},
                        players_map={},
                        season="2026",
                        platform="sleeper"
                    )
    
    # Should return empty list (synthesized)
    assert result == []
    
    # Should have a WARNING log with traceback
    warning_logs = [r for r in caplog.records if r.levelname == 'WARNING']
    
    assert len(warning_logs) == 1
    assert "get_matchups failed" in warning_logs[0].message
    assert warning_logs[0].exc_info is not None  # Traceback included


def test_generic_exception_logged_at_warning_level(caplog):
    """Generic exceptions should log at WARNING with traceback."""
    with patch('dashboard_services.matchups.get_matchups', side_effect=ValueError("test error")):
        with patch('dashboard_services.matchups.get_users', return_value=[]):
            with patch('dashboard_services.matchups.get_rosters', return_value=[]):
                with caplog.at_level(logging.DEBUG):
                    result = build_matchup_preview(
                        league_id="test_league",
                        week=1,
                        roster_map={},
                        players_map={},
                        season="2026",
                        platform="sleeper"
                    )
    
    # Should return empty list (synthesized)
    assert result == []
    
    # Should have a WARNING log with traceback
    warning_logs = [r for r in caplog.records if r.levelname == 'WARNING']
    
    assert len(warning_logs) == 1
    assert "get_matchups failed" in warning_logs[0].message
    assert warning_logs[0].exc_info is not None  # Traceback included
