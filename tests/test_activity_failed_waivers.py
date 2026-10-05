"""Failed waiver claims must not appear as successful adds in League Activity."""
import pytest


@pytest.fixture
def _svc(monkeypatch):
    pd = pytest.importorskip("pandas")
    from dashboard_services import service as svc

    # Stub out the pieces build_week_activity needs beyond transactions
    monkeypatch.setattr(
        svc, "build_roster_display_maps", lambda *a, **k: ({"1": "Team A", "2": "Team B"}, {})
    )
    monkeypatch.setattr(svc, "_activity_sweep_weeks", lambda season: [3])
    return svc, pd


def _tx(pid, rid, status):
    return {
        "type": "waiver",
        "status": status,
        "status_updated": 1759500000000,
        "adds": {pid: rid},
        "drops": {},
        "roster_ids": [rid],
    }


def test_failed_waiver_claims_excluded(monkeypatch, _svc):
    svc, pd = _svc
    txs = [
        _tx("p1", "1", "complete"),  # Team A got the player
        _tx("p1", "2", "failed"),    # Team B lost the claim
    ]
    monkeypatch.setattr(
        svc, "get_transactions_by_week", lambda *a, **k: {3: txs}
    )
    players = {"p1": {"name": "Ollie Gordon", "pos": "RB", "team": "MIA"}}
    df = svc.build_week_activity("lg1", "sleeper", 2026, players_map=players)
    rows = df.to_dict("records")
    assert len(rows) == 1
    assert rows[0]["data"]["name"] == "Team A"
    assert rows[0]["data"]["adds"][0]["name"] == "Ollie Gordon"


def test_complete_waiver_without_status_still_shown(monkeypatch, _svc):
    svc, pd = _svc
    tx = _tx("p1", "1", "complete")
    del tx["status"]  # providers that omit status must not be filtered
    monkeypatch.setattr(svc, "get_transactions_by_week", lambda *a, **k: {3: [tx]})
    players = {"p1": {"name": "Ollie Gordon", "pos": "RB", "team": "MIA"}}
    df = svc.build_week_activity("lg1", "sleeper", 2026, players_map=players)
    assert len(df.to_dict("records")) == 1


def test_failed_trade_excluded(monkeypatch, _svc):
    svc, pd = _svc
    tx = {
        "type": "trade",
        "status": "failed",
        "status_updated": 1759500000000,
        "adds": {"p1": "1"},
        "drops": {"p2": "2"},
        "draft_picks": [],
        "roster_ids": ["1", "2"],
    }
    monkeypatch.setattr(svc, "get_transactions_by_week", lambda *a, **k: {3: [tx]})
    players = {
        "p1": {"name": "Player One", "pos": "RB", "team": "MIA"},
        "p2": {"name": "Player Two", "pos": "WR", "team": "DAL"},
    }
    df = svc.build_week_activity("lg1", "sleeper", 2026, players_map=players)
    assert df.to_dict("records") == []
