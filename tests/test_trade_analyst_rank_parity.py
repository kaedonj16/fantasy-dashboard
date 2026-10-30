"""Trade Analyst rank parity with the calculator chips.

The chips on the trade cards rank by the calibrated served values (the
board payload recomputes pos_rank_label from them). /api/trade-eval used
to hand the analyst the label stored in player_values at daily save time
(ranked on the pre-calibration model ordering, and always the 1QB label),
so the analyst prose cited different ranks than the cards above it
(chip WR10, prose WR12). These tests pin:

- trade-eval assets carry labels recomputed from the same table the
  endpoint prices from, in the viewer's format (1QB / SF / redraft);
- the analyst prose scrub rewrites rank mentions bound to a traded
  player to that player's supplied label, and leaves loose role
  language alone.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("openai")

from dashboard_services.ai.prompts import enforce_trade_rank_labels  # noqa: E402

# Stored labels are deliberately stale nonsense ("WR9" everywhere): the
# endpoint must recompute them from the values, not trust the column.
_TABLE = [
    {"id": "w1", "name": "Alpha One", "position": "WR", "team": "NYJ",
     "age": 26.2, "value": 500.0, "sf_value": 480.0,
     "redraft_value_1qb": 50.0, "redraft_value_sf": 50.0,
     "pos_rank_label": "WR9", "sf_pos_rank_label": "WR9",
     "rank_change_7d": 1},
    {"id": "w2", "name": "Beta Two", "position": "WR", "team": "TB",
     "age": 24.0, "value": 400.0, "sf_value": 520.0,
     "redraft_value_1qb": 90.0, "redraft_value_sf": 90.0,
     "pos_rank_label": "WR9", "sf_pos_rank_label": "WR9",
     "rank_change_7d": 2},
    {"id": "w3", "name": "Gamma Three", "position": "WR", "team": "DAL",
     "age": 25.0, "value": 300.0, "sf_value": 300.0,
     "redraft_value_1qb": 70.0, "redraft_value_sf": 70.0,
     "pos_rank_label": "WR9", "sf_pos_rank_label": "WR9",
     "rank_change_7d": 0},
]


def _run_trade_eval(offline_client, monkeypatch, **overrides):
    import app
    import dashboard_services.player_value_history as pvh

    captured = {}
    # The endpoint's id map is a module-level cache keyed on the value
    # source's refresh timestamp, which a stubbed loader does not advance.
    # Give each test a fresh cache (restored at teardown) so this file's
    # fixture table can never leak into another file's trade-eval tests
    # (test_trade_eval_ungraded_veto.py failed in full-suite order when
    # this cache carried the table below).
    monkeypatch.setattr(app, "_TRADE_EVAL_IDMAP_CACHE", {})
    monkeypatch.setattr(app, "has_premium_for_viewer", lambda *a, **k: True)
    monkeypatch.setattr(app, "get_league_ctx_from_cache",
                        lambda **k: {"rosters": []})
    monkeypatch.setattr(app, "load_pick_value_table", lambda **k: {})
    monkeypatch.setattr(app, "_tier_thresholds_cached", lambda *a, **k: [])
    monkeypatch.setattr(pvh, "load_current_values_from_db",
                        lambda: [dict(r) for r in _TABLE])

    def fake_analysis(**kwargs):
        captured.update(kwargs)
        return "<ai>"

    monkeypatch.setattr(app, "get_trade_ai_analysis", fake_analysis)

    payload = {
        "league_id": "league-1", "viewer_roster_id": "7",
        "league_type": "1qb", "scoring_type": "dynasty",
        "side_a_players": ["w1"], "side_b_players": ["w2", "w3"],
    }
    payload.update(overrides)
    response = offline_client.post("/api/trade-eval", json=payload)
    assert response.status_code == 200
    labels = {}
    for side_key in ("side_a", "side_b"):
        for asset in captured[side_key]["assets"]:
            if asset.get("position") != "PICK":
                labels[asset["id"]] = asset["pos_rank_label"]
    return labels


def test_trade_eval_dynasty_1qb_labels_recomputed_from_served_values(
        offline_client, monkeypatch):
    labels = _run_trade_eval(offline_client, monkeypatch)
    # By value: Alpha 500 > Beta 400 > Gamma 300.
    assert labels == {"w1": "WR1", "w2": "WR2", "w3": "WR3"}


def test_trade_eval_dynasty_sf_labels_follow_sf_values(
        offline_client, monkeypatch):
    labels = _run_trade_eval(offline_client, monkeypatch, league_type="sf")
    # By sf_value: Beta 520 > Alpha 480 > Gamma 300. The old code always
    # sent the 1QB label, so SF analysts cited 1QB ranks.
    assert labels == {"w1": "WR2", "w2": "WR1", "w3": "WR3"}


def test_trade_eval_redraft_labels_follow_redraft_values(
        offline_client, monkeypatch):
    labels = _run_trade_eval(offline_client, monkeypatch,
                             scoring_type="redraft")
    # By redraft value: Beta 90 > Gamma 70 > Alpha 50.
    assert labels == {"w1": "WR3", "w2": "WR1", "w3": "WR2"}


_PAYLOAD = {
    "trade": {
        "viewer_gets": {"assets": [
            {"name": "Garrett Wilson", "position": "WR",
             "pos_rank_label": "WR10"},
        ]},
        "viewer_gives": {"assets": [
            {"name": "Emeka Egbuka", "position": "WR",
             "pos_rank_label": "WR18"},
        ]},
    },
}


def test_scrub_rewrites_appositive_ranks_to_supplied_labels():
    data = {
        "verdict": "DECLINE",
        "summary": (
            "You would be turning Emeka Egbuka, a young WR15 who is still "
            "part of Tampa Bay's core, into Garrett Wilson, a proven WR12 "
            "in his prime."
        ),
        "helps": [], "risks": [], "counter": "", "confidence": "high",
    }
    out = enforce_trade_rank_labels(data, _PAYLOAD)
    assert "WR18" in out["summary"]
    assert "WR10" in out["summary"]
    assert "WR15" not in out["summary"]
    assert "WR12" not in out["summary"]


def test_scrub_rewrites_parenthetical_and_prefix_ranks():
    data = {
        "summary": "WR12 Garrett Wilson (WR12) is the prize.",
        "helps": ["Egbuka (WR15) plus a first is a lot to send."],
        "risks": [], "counter": "", "confidence": "high",
    }
    out = enforce_trade_rank_labels(data, _PAYLOAD)
    assert out["summary"] == "WR10 Garrett Wilson (WR10) is the prize."
    assert out["helps"] == ["Egbuka (WR18) plus a first is a lot to send."]


def test_scrub_leaves_role_language_and_correct_ranks_alone():
    data = {
        "summary": "Garrett Wilson, the WR10 overall, is a clean upgrade.",
        "helps": [
            "Wilson has a longer runway as a WR1 than Egbuka.",
            "In a 1QB league, a single premium WR can stabilize a contender.",
        ],
        "risks": [], "counter": "", "confidence": "high",
    }
    out = enforce_trade_rank_labels(data, _PAYLOAD)
    assert out["summary"] == data["summary"]
    assert out["helps"] == data["helps"]


def test_scrub_no_labels_is_a_noop():
    data = {"summary": "Garrett Wilson, a proven WR12, headlines it."}
    assert enforce_trade_rank_labels(data, {"trade": {}}) == data
    assert enforce_trade_rank_labels(data, {}) == data
