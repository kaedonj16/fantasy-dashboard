"""Missed tackles are broken tackles: the concept wiring tests.

Live failure (2026-10-15): "what missed tackle metrics do I have?"
answered "That metric is not tracked on the site." The site tracks the
concept as broken tackles (missed tackles forced by the ball carrier,
PFR charting): Broken Tackles/Carry on rush attempts and Broken
Tackles / Catch on receptions, both in the Advanced Metrics catalog.
These tests pin the semantic resolution (aliases), the family pack
(build_metric names every member), the open-layer definition tool
(sibling carried with its leader), and the router keyword fallback
(the hint survives even without the model router).
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import aliases
from dashboard_services.assistant import evidence
from dashboard_services.assistant import tools as tools_mod
from dashboard_services.assistant.router import _keyword_route


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": {"raw_scoring_settings": {"rec": 1.0},
                      "rosters": [], "roster_positions": []},
              "roster": {"players": [], "starters": []},
              "roster_player_ids": set(), "has_pro": True}
    bundle.update(over)
    return bundle


_GAMES_VOL = {"col": "games", "opts": [4, 8, 12, 16], "label": "Min Games"}
_CARRIES_VOL = {"col": "carries", "opts": [50, 100], "label": "Min Carries"}
_RECS_VOL = {"col": "receptions", "opts": [25, 50], "label": "Min Receptions"}
# Labels and descriptions copied from the real catalog entries in
# data_building/advanced_metrics.py (LEADERBOARD_METRICS).
CONFIG = {
    "avoided_tackles_per_carry": {
        "label": "Broken Tackles/Carry", "category": "Rushing",
        "positions": ["RB"], "min_vol": dict(_CARRIES_VOL),
        "desc": "Broken tackles per carry on rush attempts (PFR charting via nflverse)."},
    "rec_broken_tackles_per_reception": {
        "label": "Broken Tackles / Catch", "category": "Receiving",
        "positions": ["WR", "RB", "TE"], "min_vol": dict(_RECS_VOL),
        "desc": "Broken tackles per reception on receiving plays only (Pro Football Reference charting)."},
    "target_share": {
        "label": "Target Share", "positions": ["WR", "TE", "RB"],
        "min_vol": dict(_GAMES_VOL), "desc": "Share of team pass attempts."},
}


def _prow(pid, name, team, pos, value, games=6):
    return {"player_id": pid, "name": name, "team": team, "position": pos,
            "value": value, "games": games}


ROWS = {
    "avoided_tackles_per_carry": [
        _prow("gibbs", "Jahmyr Gibbs", "DET", "RB", 0.19),
        _prow("irving", "Bucky Irving", "TB", "RB", 0.18),
        _prow("conner", "James Conner", "ARI", "RB", 0.17)],
    "rec_broken_tackles_per_reception": [
        _prow("deebo", "Deebo Samuel", "WAS", "WR", 0.31),
        _prow("kraft", "Tucker Kraft", "GB", "TE", 0.29)],
}


class _Policy:
    games_min = 4
    provisional = False

    def minimum(self, col, full):
        return full


@pytest.fixture()
def boards(monkeypatch):
    monkeypatch.setattr(evidence, "_metric_config", lambda: dict(CONFIG))
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: {})
    monkeypatch.setattr(evidence, "qualification_policy_for",
                        lambda season, week_start=None, week_end=None: _Policy())
    monkeypatch.setattr(evidence, "pro_metric_keys", lambda: {"wopr"})
    monkeypatch.setattr(evidence, "premium_metric_keys", lambda: set())
    monkeypatch.setattr(evidence, "premium_metrics_exposed", lambda: False)

    def fake_leaderboard(metric, position=None, limit=500, season=None,
                         min_vol=None, combine=False):
        rows = ROWS.get(metric, [])
        if position:
            rows = [r for r in rows if r["position"] == position]
        return list(rows)

    monkeypatch.setattr(evidence, "metric_leaderboard", fake_leaderboard)
    monkeypatch.setattr("data_building.advanced_metrics.get_metric_leaderboard",
                        lambda metric, **k: list(ROWS.get(metric, [])))
    return monkeypatch


# -- aliases -----------------------------------------------------------------

def test_alias_missed_tackles_resolve_to_broken_tackle_keys():
    r = aliases.resolve_phrase("missed tackles")
    assert r["kind"] == "metric" and r["key"] == "avoided_tackles_per_carry"
    r = aliases.resolve_phrase("who forces the most missed tackles")
    assert r["kind"] == "metric" and r["key"] == "avoided_tackles_per_carry"
    r = aliases.resolve_phrase("broken tackles per catch leaders")
    assert r["kind"] == "metric"
    assert r["key"] == "rec_broken_tackles_per_reception"
    r = aliases.resolve_phrase("broken tackles per carry")
    assert r["kind"] == "metric" and r["key"] == "avoided_tackles_per_carry"


def test_alias_elusiveness_resolves_to_broken_tackles():
    r = aliases.resolve_phrase("elusiveness")
    assert r["kind"] == "metric" and r["key"] == "avoided_tackles_per_carry"
    r = aliases.resolve_phrase("most elusive running back")
    assert r["kind"] == "metric" and r["key"] == "avoided_tackles_per_carry"


def test_metric_family_for_names_both_tracked_rates():
    fam = aliases.metric_family_for("what missed tackle metrics do I have")
    assert fam["keys"] == ("avoided_tackles_per_carry",
                           "rec_broken_tackles_per_reception")
    assert "missed tackles forced" in fam["blurb"]
    assert aliases.metric_family_for("target share") is None


# -- evidence: the family pack ------------------------------------------------

def test_build_metric_missed_tackles_answers_with_the_family(boards):
    pack, _ = evidence.build_metric(_bundle(), {"metric_hint": "missed tackles"})
    assert pack["available"] is True
    assert pack["metric_key"] == "avoided_tackles_per_carry"
    assert pack["label"] == "Broken Tackles/Carry"
    assert [e["metric_key"] for e in pack["family_metrics"]] == [
        "avoided_tackles_per_carry", "rec_broken_tackles_per_reception"]
    assert "Broken Tackles/Carry" in pack["definition"]
    assert "Broken Tackles / Catch" in pack["definition"]
    assert "missed tackles forced" in pack["definition"]
    assert pack["leaderboard_top5"][0]["name"] == "Jahmyr Gibbs"


def test_build_metric_specific_rate_keeps_its_own_board(boards):
    pack, _ = evidence.build_metric(
        _bundle(), {"metric_hint": "broken tackles per catch"})
    assert pack["available"] is True
    assert pack["metric_key"] == "rec_broken_tackles_per_reception"
    assert pack["leaderboard_top5"][0]["name"] == "Deebo Samuel"


def test_build_metric_plain_metric_has_no_family(boards):
    pack, _ = evidence.build_metric(_bundle(), {"metric_hint": "target share"})
    assert pack["available"] is True
    assert "family_metrics" not in pack


# -- router keyword fallback ----------------------------------------------------

def test_keyword_missed_tackle_question_keeps_its_hint():
    r = _keyword_route("what missed tackle metrics do I have?")
    assert r["intent"] == "metric"
    assert r["metric_hint"] == "missed tackle"


# -- open layer tools -----------------------------------------------------------

def test_tool_metric_definition_missed_tackles_carries_the_sibling(boards):
    entry = tools_mod.run_tool(
        "metric_definition", {"metric": "missed tackles"}, _bundle())
    assert entry["ok"] is True
    assert entry["metric_key"] == "avoided_tackles_per_carry"
    assert entry["leader"]["name"] == "Jahmyr Gibbs"
    assert "missed tackles forced" in entry["concept"]
    also = entry["also_tracked"]
    assert [e["metric_key"] for e in also] == ["rec_broken_tackles_per_reception"]
    assert also[0]["label"] == "Broken Tackles / Catch"
    assert also[0]["leader"]["name"] == "Deebo Samuel"


def test_tool_top_board_missed_tackles(boards):
    entry = tools_mod.run_tool(
        "top_board", {"stat": "missed tackles", "top_n": 3}, _bundle())
    assert entry["ok"] is True
    assert entry["metric_key"] == "avoided_tackles_per_carry"
    assert entry["board"][0]["name"] == "Jahmyr Gibbs"
