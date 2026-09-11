"""Test that Sleeper API layer handles 404s gracefully for future weeks.

This tests the fix at the API layer (dashboard_services.api) for matchups,
transactions, and brackets, rather than the service layer.
"""
import pytest
from unittest.mock import patch, Mock

# Heavy deps aren't installed in the lint job (ruff+pytest only); skip there so
# collection doesn't hard-fail. Mirrors tests/test_matchup_404_handling.py.
requests = pytest.importorskip("requests")

from dashboard_services.api import get_matchups, get_transactions, get_bracket


# ---- Matchups Tests ----

def test_get_matchups_returns_empty_list_on_404():
    """get_matchups should return [] for 404 (future weeks), not raise."""
    mock_response = Mock()
    mock_response.status_code = 404
    http_error = requests.HTTPError(response=mock_response)
    
    with patch('dashboard_services.api.fetch_json', side_effect=http_error):
        result = get_matchups("887776065", 16)
        assert result == []


def test_get_matchups_raises_on_500_error():
    """get_matchups should still raise for non-404 HTTP errors."""
    mock_response = Mock()
    mock_response.status_code = 500
    http_error = requests.HTTPError(response=mock_response)
    
    with patch('dashboard_services.api.fetch_json', side_effect=http_error):
        with pytest.raises(requests.HTTPError):
            get_matchups("887776065", 1)


def test_get_matchups_returns_data_on_success():
    """get_matchups should return matchup data when API succeeds."""
    expected_data = [{"roster_id": 1, "matchup_id": 1}]
    
    with patch('dashboard_services.api.fetch_json', return_value=expected_data):
        result = get_matchups("887776065", 1)
        assert result == expected_data


def test_get_matchups_raises_on_network_error():
    """get_matchups should raise for network errors (not HTTPError)."""
    with patch('dashboard_services.api.fetch_json', side_effect=requests.ConnectionError()):
        with pytest.raises(requests.ConnectionError):
            get_matchups("887776065", 1)


# ---- Transactions Tests ----

def test_get_transactions_returns_empty_list_on_404():
    """get_transactions should return [] for 404 (future weeks or no transactions), not raise."""
    mock_response = Mock()
    mock_response.status_code = 404
    http_error = requests.HTTPError(response=mock_response)
    
    with patch('dashboard_services.api.fetch_json', side_effect=http_error):
        result = get_transactions("887776065", 16)
        assert result == []


def test_get_transactions_raises_on_500_error():
    """get_transactions should still raise for non-404 HTTP errors."""
    mock_response = Mock()
    mock_response.status_code = 500
    http_error = requests.HTTPError(response=mock_response)
    
    with patch('dashboard_services.api.fetch_json', side_effect=http_error):
        with pytest.raises(requests.HTTPError):
            get_transactions("887776065", 1)


def test_get_transactions_returns_data_on_success():
    """get_transactions should return transaction data when API succeeds."""
    expected_data = [{"type": "trade", "status": "complete"}]
    
    with patch('dashboard_services.api.fetch_json', return_value=expected_data):
        result = get_transactions("887776065", 1)
        assert result == expected_data


# ---- Bracket Tests ----

def test_get_bracket_returns_empty_list_on_404():
    """get_bracket should return [] for 404 (no playoffs yet), not raise."""
    mock_response = Mock()
    mock_response.status_code = 404
    http_error = requests.HTTPError(response=mock_response)
    
    with patch('dashboard_services.api.fetch_json', side_effect=http_error):
        result = get_bracket("887776065", "winners")
        assert result == []


def test_get_bracket_raises_on_500_error():
    """get_bracket should still raise for non-404 HTTP errors."""
    mock_response = Mock()
    mock_response.status_code = 500
    http_error = requests.HTTPError(response=mock_response)
    
    with patch('dashboard_services.api.fetch_json', side_effect=http_error):
        with pytest.raises(requests.HTTPError):
            get_bracket("887776065", "winners")


def test_get_bracket_returns_data_on_success():
    """get_bracket should return bracket data when API succeeds."""
    expected_data = [{"r": 1, "m": 1, "t1": 1, "t2": 2}]
    
    with patch('dashboard_services.api.fetch_json', return_value=expected_data):
        result = get_bracket("887776065", "winners")
        assert result == expected_data
