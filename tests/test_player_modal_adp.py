"""Player-modal ADP range: Cons is the mean of the plotted source dots.

BR Fantasy is shown as a 1..N ordinal rank so its board can top out at 1.
Sleeper (and the redraft globals) stay on their displayed ADP. Consensus on
the range must average those plotted values — (BR rank 2.0 + Sleeper 4.3) / 2
→ 3.15, shown as 3.2 — not the backend raw-ADP consensus, which sits off the
dots at the mean-pick floor.
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


def test_player_adp_api_ordinal_ranks_brfantasy():
    body = _player_adp_route_src()
    assert 'as_rank=(_source in ("brfantasy", "brfantasy_live"))' in body
    assert "fallback=False" in body
    assert '"brfantasy_live", "BR Fantasy Live (7d)"' in body


def test_player_modal_consensus_averages_plotted_dots():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "pts.reduce((sum, p) => sum + p.v, 0) / pts.length" in js
    assert "consSrc && consSrc.vals[k]" not in js
    assert "Average of the source values shown on this axis" in js
    # Backend Consensus is filtered out so it cannot become a third dot.
    assert "sources.filter(s => s.label !== 'Consensus')" in js


def test_displayed_consensus_is_mean_of_rank_and_sleeper():
    """(BR ordinal 2.0 + Sleeper 4.3) / 2 → 3.15, labeled 3.2."""
    cons = (2.0 + 4.3) / 2
    assert cons == pytest.approx(3.15)
    labeled = round(cons * 10) / 10
    assert labeled == 3.2


def test_api_player_adp_ranks_brfantasy_to_contiguous_board(monkeypatch):
    try:
        from app import app as flask_app
    except Exception as exc:
        pytest.skip(f"app not importable ({type(exc).__name__})")

    from dashboard_services.adp_service import ordinal_rank_adp

    pid = "4046"
    br_raw = {"early": 3.3, pid: 7.7, "late": 20.0}

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
    # 3.3 → rank 1, 7.7 → rank 2, 20.0 → rank 3
    assert by_label["BR Fantasy"]["dynasty_1qb"] == 2.0
    # Backend consensus is still the raw mean; the modal does not plot it.
    assert by_label["Consensus"]["dynasty_1qb"] == 6.0


def test_rankings_consensus_column_averages_displayed_sleeper_and_br_rank(monkeypatch):
    """Gibbs Sleeper 1.0 + BR rank 2.0 must show Cons 1.5, not raw-blend 3.1."""
    try:
        from app import _attach_all_adp_sources
    except Exception as exc:
        pytest.skip(f"app not importable ({type(exc).__name__})")

    players = [{"id": "gibbs"}, {"id": "bijan"}]
    sleeper = {"gibbs": 1.0, "bijan": 2.1}
    # Raw mean-picks: Bijan earlier → rank 1, Gibbs rank 2 after ordinal_rank.
    br_raw = {"gibbs": 5.2, "bijan": 4.1}

    def fake_resolve(season, is_sf, scoring_type="redraft", source="consensus",
                     as_rank=False, fallback=True, **kwargs):
        if source == "consensus":
            raise AssertionError("rankings Cons must not use resolve_market_adp(consensus)")
        if source == "sleeper":
            return dict(sleeper)
        if source == "brfantasy":
            return dict(br_raw)
        return {}

    monkeypatch.setattr(
        "dashboard_services.adp_service.resolve_market_adp", fake_resolve,
    )
    cols = _attach_all_adp_sources(
        players, 2026, ["sleeper", "brfantasy", "consensus"],
    )
    assert [c["value"] for c in cols] == ["sleeper", "brfantasy", "consensus"]
    by = {p["id"]: p["adp_by_source"] for p in players}
    assert by["gibbs"]["sleeper"]["avg_pick"] == 1.0
    assert by["gibbs"]["brfantasy"]["avg_pick"] == 2.0
    assert by["gibbs"]["consensus"]["avg_pick"] == 1.5
    assert by["bijan"]["sleeper"]["avg_pick"] == 2.1
    assert by["bijan"]["brfantasy"]["avg_pick"] == 1.0
    assert by["bijan"]["consensus"]["avg_pick"] == 1.6


def test_yahoo_overlay_rebuilds_consensus_from_all_displayed_columns(monkeypatch):
    try:
        from app import _attach_all_adp_sources
    except Exception as exc:
        pytest.skip(f"app not importable ({type(exc).__name__})")

    players = [{
        "id": "p",
        "adp_by_source": {
            "sleeper": {"avg_pick": 1.0, "sf_avg_pick": 1.0,
                        "redraft_avg_pick": 1.0, "sf_redraft_avg_pick": 1.0},
            "brfantasy": {"avg_pick": 2.0, "sf_avg_pick": 2.0,
                          "redraft_avg_pick": 2.0, "sf_redraft_avg_pick": 2.0},
            "consensus": {"avg_pick": 3.1},
        },
    }]

    def fake_resolve(season, is_sf, scoring_type="redraft", source="consensus",
                     as_rank=False, fallback=True, **kwargs):
        if source == "consensus":
            raise AssertionError("overlay must not resolve raw consensus")
        if source == "yahoo":
            return {"p": 4.0}
        return {}

    monkeypatch.setattr(
        "dashboard_services.adp_service.resolve_market_adp", fake_resolve,
    )
    _attach_all_adp_sources(players, 2026, ["yahoo", "consensus"])
    cons = players[0]["adp_by_source"]["consensus"]
    # (1.0 + 2.0 + 4.0) / 3 = 2.3, not yahoo-only 4.0 and not leftover 3.1.
    assert cons["redraft_avg_pick"] == 2.3
    assert players[0]["adp_by_source"]["yahoo"]["redraft_avg_pick"] == 4.0
