"""Playoff Path section in the Front Office Report (PR A).

The section is built from the same two payloads the Standings Playoff
Path tab reads: the shared cached sim rows (``ctx["playoff_odds"]``)
and ``compute_scenarios`` assembled from ``build_sim_state`` exactly as
``/api/playoff-scenarios`` assembles it. Fixtures use the REAL scenario
enumerator over a crafted sim state (only ``build_sim_state`` is
stubbed): a 10-team league, 6 playoff spots, 2 byes, Weeks 13-14 left.
The viewer (Nunky, roster 4) is alive with magic number 1 and a
"Win and you're in" week.
"""
import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

import data_building.simulate_playoff_odds as spo
from dashboard_services.ai import front_office_report as fmod
from dashboard_services.ai.prompts import build_front_office_prompt_payload

TEAMS = [
    {"roster_id": 1, "name": "Couch",  "wins": 8, "losses": 4, "ties": 0, "pf": 1427.6},
    {"roster_id": 2, "name": "Pilots", "wins": 9, "losses": 3, "ties": 0, "pf": 1460.0},
    {"roster_id": 3, "name": "Fire",   "wins": 8, "losses": 4, "ties": 0, "pf": 1390.0},
    {"roster_id": 4, "name": "Nunky",  "wins": 7, "losses": 5, "ties": 0, "pf": 1330.0},
    {"roster_id": 5, "name": "Trust",  "wins": 6, "losses": 6, "ties": 0, "pf": 1440.0},
    {"roster_id": 6, "name": "Quon",   "wins": 6, "losses": 6, "ties": 0, "pf": 1435.0},
    {"roster_id": 7, "name": "Veiny",  "wins": 6, "losses": 6, "ties": 0, "pf": 1450.0},
    {"roster_id": 8, "name": "Never",  "wins": 5, "losses": 7, "ties": 0, "pf": 1180.0},
    {"roster_id": 9, "name": "Kam",    "wins": 4, "losses": 8, "ties": 0, "pf": 1100.0},
    {"roster_id": 10, "name": "Jerry", "wins": 3, "losses": 9, "ties": 0, "pf": 1020.0},
]

STATE = {
    "teams": TEAMS,
    "playoff_teams": 6,
    "matchups": {
        13: [(1, 2), (3, 4), (5, 9), (6, 10), (7, 8)],
        14: [(1, 3), (2, 9), (4, 10), (5, 7), (6, 8)],
    },
}

# Six regular-season weeks left: the scenarios window is shut.
STATE_EARLY = {
    "teams": TEAMS,
    "playoff_teams": 6,
    "matchups": {w: [(1, 2), (3, 4), (5, 9), (6, 10), (7, 8)]
                 for w in range(8, 14)},
}

ROW = {
    "roster_id": 4, "team_name": "Nunky",
    "playoff_pct": 62.4, "bye_pct": 3.1, "first_seed_pct": 0.4,
    "impact_week": 13,
    "playoff_if_win": 99.1, "playoff_if_lose": 27.2,
    "bye_if_win": 8.0, "bye_if_lose": 0.5,
    "first_seed_if_win": 1.2, "first_seed_if_lose": 0.1,
    "swing": 71.9,
    "root_for": [
        {"a": 1, "b": 2, "playoff_if_a": 55.0, "playoff_if_b": 68.0,
         "bye_if_a": None, "bye_if_b": None},
        {"a": 5, "b": 9, "playoff_if_a": 60.1, "playoff_if_b": 63.3,
         "bye_if_a": None, "bye_if_b": None},
        {"a": 6, "b": 10, "playoff_if_a": 61.0, "playoff_if_b": 61.5,
         "bye_if_a": None, "bye_if_b": None},
        {"a": 7, "b": 8, "playoff_if_a": 70.2, "playoff_if_b": 41.0,
         "bye_if_a": None, "bye_if_b": None},
    ],
}


def _ctx(**over):
    ctx = {
        "platform": "sleeper",
        "league_settings": {"playoff_teams": 6, "playoff_week_start": 15},
        "rosters": [{"roster_id": t["roster_id"], "settings": {}}
                    for t in TEAMS],
        "playoff_odds": [ROW],
    }
    ctx.update(over)
    return ctx


@pytest.fixture
def sim_state(monkeypatch):
    monkeypatch.setattr(
        spo, "build_sim_state", lambda ctx, platform="sleeper": STATE)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

def test_build_playoff_path_in_window(sim_state):
    pp = fmod._build_playoff_path(_ctx(), "4")
    assert pp is not None
    # Posture: standings-order seed (Pilots, Couch, Fire ahead), the sim
    # row's odds, and the enumerator's magic number + summary verbatim.
    assert pp["seed"] == 4
    assert pp["playoff_pct"] == 62.4
    assert pp["bye_pct"] == 3.1
    assert pp["magic_number"] == 1
    assert pp["status"] == "alive"
    assert pp["summary"] == "Win and you're in"
    assert pp["can_clinch_this_week"] is True
    assert pp["earliest_clinch_week"] == 13
    # Stakes: the viewer's fork, all three metrics, vs the scenario
    # next-game opponent.
    assert pp["stakes"]["opponent"] == "Fire"
    assert pp["stakes"]["week"] == 13
    assert pp["stakes"]["fork"] == [
        {"label": "Playoffs", "if_win": 99.1, "if_lose": 27.2},
        {"label": "Bye", "if_win": 8.0, "if_lose": 0.5},
        {"label": "1 seed", "if_win": 1.2, "if_lose": 0.1},
    ]
    # Root for: top 3 by swing, rooted side is the higher-odds side.
    assert [g["root_name"] for g in pp["root_for"]] == [
        "Veiny", "Pilots", "Kam"]
    assert pp["root_for"][0]["delta"] == 29.2
    assert pp["root_for"][0]["other_name"] == "Never"


def test_build_playoff_path_out_of_window_returns_none(monkeypatch):
    monkeypatch.setattr(
        spo, "build_sim_state", lambda ctx, platform="sleeper": STATE_EARLY)
    assert fmod._build_playoff_path(_ctx(), "4") is None


def test_build_playoff_path_cold_sim_cache_returns_none(sim_state):
    ctx = _ctx()
    del ctx["playoff_odds"]
    assert fmod._build_playoff_path(ctx, "4") is None


def test_build_playoff_path_bye_week_viewer_keeps_posture(sim_state):
    bye_row = dict(ROW)
    for key in ("playoff_if_win", "playoff_if_lose", "bye_if_win",
                "bye_if_lose", "first_seed_if_win", "first_seed_if_lose",
                "swing"):
        bye_row[key] = None
    pp = fmod._build_playoff_path(_ctx(playoff_odds=[bye_row]), "4")
    assert pp is not None
    assert pp["stakes"] is None
    assert pp["magic_number"] == 1
    assert len(pp["root_for"]) == 3


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------

def _path_block():
    return {
        "seed": 4, "playoff_pct": 62.4, "bye_pct": 3.1,
        "magic_number": 1, "status": "alive",
        "summary": "Win and you're in",
        "can_clinch_this_week": True, "earliest_clinch_week": 13,
        "stakes": {
            "week": 13, "opponent": "Fire",
            "fork": [
                {"label": "Playoffs", "if_win": 99.1, "if_lose": 27.2},
                {"label": "Bye", "if_win": 8.0, "if_lose": 0.5},
            ],
        },
        "root_for": [
            {"a_name": "Veiny", "b_name": "Never", "root_name": "Veiny",
             "other_name": "Never", "root_pct": 70.2, "other_pct": 41.0,
             "delta": 29.2},
        ],
    }


def _data(**over):
    data = {
        "team_name": "Nunky", "week": 13, "record": "7-5",
        "playoff_pct": 62.4, "grades": [], "roster_rows": [],
        "injury_rows": [], "trade_targets": [],
        "potential_trade_targets": [], "waiver_targets": [],
        "cut_candidates": [], "drop_add_pairs": [],
        "last_week": {"week": 12, "result": "W", "pf": 121.0,
                      "pa": 100.0, "opponent": "Trust"},
        "risers_7d": [{"id": "1", "name": "Ris Ing", "position": "WR",
                       "trend_7d": 12}],
        "fallers_7d": [],
        "playoff_path": _path_block(),
    }
    data.update(over)
    return data


_AI = {"verdict": "HOLD", "headline": "Alive and in control.",
       "posture": "Win this week.", "top_move": "", "gm_alert": "",
       "trade_notes": {}, "waiver_notes": {}}


def test_render_fallback_layout_section_before_since_last_week():
    out = fmod.render_front_office_report_html(_data(), _AI)
    assert "id='forPlayoffPath'" in out
    assert "Playoff Path" in out
    assert "Magic number 1" in out
    assert "Win and you" in out  # summary verbatim (apostrophe escaped)
    assert "This week" in out and "vs Fire" in out
    assert "Root for" in out and "Veiny" in out
    assert out.index("id='forPlayoffPath'") < out.index("Since last week")


def test_render_board_layout_section_and_jump_button():
    data = _data(decision_moves=[
        {"rank": 1, "line": "Set the lineup", "timing": "Before waivers",
         "why": "", "after": "", "dep": ""},
    ])
    out = fmod.render_front_office_report_html(data, _AI)
    assert "data-for-jump='forPlayoffPath'" in out
    assert out.index("id='forBoard'") < out.index("id='forPlayoffPath'")
    assert out.index("id='forPlayoffPath'") < out.index("id='forLastWeek'")
    # The jump button order follows the section order on the page.
    assert out.index("data-for-jump='forPlayoffPath'") < \
        out.index("data-for-jump='forLastWeek'")


def test_render_without_path_block_has_no_section_or_button():
    data = _data(playoff_path=None, decision_moves=[
        {"rank": 1, "line": "Set the lineup", "timing": "Before waivers",
         "why": "", "after": "", "dep": ""},
    ])
    out = fmod.render_front_office_report_html(data, _AI)
    assert "forPlayoffPath" not in out
    assert "Playoff Path" not in out


# ---------------------------------------------------------------------------
# prompt payload + cache version
# ---------------------------------------------------------------------------

def test_prompt_payload_carries_playoff_path():
    payload = build_front_office_prompt_payload(_data())
    pp = payload["playoff_path"]
    assert pp["magic_number"] == 1
    assert pp["scenario"] == "Win and you're in"
    assert pp["this_week_opponent"] == "Fire"
    assert pp["fork"][0] == {"label": "Playoffs", "if_win": 99.1,
                             "if_lose": 27.2}
    assert pp["root_for"][0]["root_for"] == "Veiny"


def test_prompt_payload_omits_path_out_of_window():
    payload = build_front_office_prompt_payload(_data(playoff_path=None))
    assert "playoff_path" not in payload


def test_cache_version_bumped_for_new_section():
    assert fmod.CACHE_VERSION == "v9"
