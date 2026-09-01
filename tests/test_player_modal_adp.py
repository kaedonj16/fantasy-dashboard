"""Player-modal ADP range: consensus shares the source-dot pick scale.

The range chart plots Sleeper / BR Fantasy / … as dots and Consensus as a
marker on one overall-pick axis. Consensus is the arithmetic mean of raw
source ADPs, so every plotted value must be that same raw avg_pick — not the
1..N board rank rankings uses for the BR Fantasy column. Ranking BR Fantasy
here puts its dot near pick 1 while consensus still sits at the mean-pick
floor, so Cons pins to the right of every visible source.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _player_adp_route_src() -> str:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def api_player_adp")
    end = src.find("def api_player_game_logs", start)
    assert start > 0 and end > start
    return src[start:end]


def test_player_adp_api_does_not_ordinal_rank_brfantasy():
    body = _player_adp_route_src()
    assert 'as_rank=(_source == "brfantasy")' not in body
    assert "as_rank=False" in body
    assert "fallback=False" in body


def test_player_modal_range_treats_consensus_as_same_scale_as_dots():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    # Scale is min/max of source dots AND consensus, then both are positioned
    # with the same `pos()` so Cons sits among the dots when they share a scale.
    assert "const all = pts.map(p => p.v).concat(cons != null ? [cons] : []);" in js
    assert "style=\"left:${pos(p.v).toFixed(1)}%" in js
    assert "style=\"left:${pos(cons).toFixed(1)}%" in js
    assert "Average of the source ADPs on this axis" in js


def test_api_player_adp_returns_raw_brfantasy_not_board_rank(monkeypatch):
    """Regression: a top player's BR Fantasy mean-pick (~7.7) must not become
    rank 1.0 / 4.0 on the modal; consensus 6.0 is the mean of that raw value
    and Sleeper, so the orange dot has to be the raw number."""
    try:
        from app import app as flask_app
    except Exception as exc:
        pytest.skip(f"app not importable ({type(exc).__name__})")

    from dashboard_services.adp_service import ordinal_rank_adp

    pid = "4046"
    br_raw = {pid: 7.7}

    def fake_resolve(season, is_sf, scoring_type="redraft", source="consensus",
                     as_rank=False, fallback=True, **kwargs):
        if source == "brfantasy":
            return ordinal_rank_adp(br_raw) if as_rank else dict(br_raw)
        if source == "consensus":
            return {pid: 6.0}
        return {}

    monkeypatch.setattr(
        "dashboard_services.adp_service.resolve_market_adp", fake_resolve,
    )
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as client:
        resp = client.get(f"/api/player-adp/{pid}?season=2026")
    assert resp.status_code == 200
    payload = resp.get_json()
    by_label = {row["label"]: row["vals"] for row in (payload.get("sources") or [])}
    assert by_label["BR Fantasy"]["dynasty_1qb"] == 7.7
    assert by_label["Consensus"]["dynasty_1qb"] == 6.0
    # Ranked, a one-player map would be 1.0 — that is the bug this guards.
    assert by_label["BR Fantasy"]["dynasty_1qb"] != 1.0
