"""Compose contract fixes: forced receipts follow the question's subject,
and the mention appender speaks the right sentence for live/draft kinds.

Covers the 2026-11-06 gap audit (PR C):
- live_team / draft_clock / draft_best mentions fell through to the
  schedule-stretch sentence ("That stretch runs through ...").
- playoffs receipts never forced the first-round bye odds the pinned
  follow-up asks about, and the wins receipt hid that its number is
  measured against the projected cut line.
- schedule receipts led with the playoff-stretch index ahead of the
  remaining-schedule index.
- standings / activity / metric receipts were stapled regardless of
  the question's subject (my seed on a last-place question, the latest
  trade's totals on a waiver question, a leader's value on a
  definition question).
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod


def _labels(rows):
    return [r["label"] for r in rows]


# -- mention appender: live and draft kinds --------------------------------

LIVE_PACK = {
    "type": "live", "available": True,
    "matchup": {"is_mine": True,
                "team_a": {"team_name": "Caleb's Casting Couch",
                           "current_points": 84.5, "projected": 121.4,
                           "win_prob_pct": 62.0},
                "team_b": {"team_name": "Bijan Mustard",
                           "current_points": 79.2, "projected": 118.9,
                           "win_prob_pct": 38.0}},
}

DRAFT_PACK = {
    "type": "draft", "available": True,
    "current_pick": {"team_name": "Hurts So Good", "pick_no": 13},
    "best_available": [{"name": "Ja'Marr Chase", "position": "WR"}],
}


def test_live_mentions_use_game_sentence_not_stretch():
    data = {"verdict": "You lead", "text": "You lead late.",
            "caveat": None, "receipts": [], "followups": []}
    out = compose_mod._ensure_required_mentions("live", LIVE_PACK, data, [])
    assert "That game involves Caleb's Casting Couch and Bijan Mustard." \
        in out["text"]
    assert "That stretch" not in out["text"]


def test_live_single_missing_team_uses_game_sentence():
    data = {"verdict": "Caleb's Casting Couch leads",
            "text": "Caleb's Casting Couch leads late.",
            "caveat": None, "receipts": [], "followups": []}
    out = compose_mod._ensure_required_mentions("live", LIVE_PACK, data, [])
    assert "That game involves Bijan Mustard." in out["text"]
    assert "That stretch" not in out["text"]


def test_draft_mentions_use_clock_and_best_available_sentences():
    data = {"verdict": "Round 2", "text": "The board is thinning.",
            "caveat": None, "receipts": [], "followups": []}
    out = compose_mod._ensure_required_mentions("draft", DRAFT_PACK, data, [])
    assert "The team on the clock is Hurts So Good." in out["text"]
    assert "The best available player is Ja'Marr Chase." in out["text"]
    assert "That stretch" not in out["text"]


def test_schedule_stretch_mentions_keep_stretch_sentence():
    pack = {"type": "schedule", "available": True,
            "playoff_stretch": {"me": {"opponents": [
                {"opponent": "Pittsburgh Pilots"},
                {"opponent": "Lamb Chop"}]}}}
    data = {"verdict": "Tough finish", "text": "The finish is tough.",
            "caveat": None, "receipts": [], "followups": []}
    out = compose_mod._ensure_required_mentions("schedule", pack, data, [])
    assert "That stretch runs through Pittsburgh Pilots and Lamb Chop." \
        in out["text"]


# -- playoffs receipts: bye odds + cut-line label ----------------------------

PLAYOFFS_PACK = {
    "type": "playoffs", "available": True,
    "me": {"team_name": "Caleb's Casting Couch", "playoff_pct": 71.5,
           "bye_pct": 18.2, "wins_still_needed": 3.9},
    "scenario": None, "weeks_remaining": 7, "odds_trajectory": None,
}


def test_playoffs_receipts_include_bye_odds_and_cut_line_label():
    rows = compose_mod._required_receipts("playoffs", PLAYOFFS_PACK)
    assert _labels(rows) == ["Your playoff odds (%)",
                             "First-round bye odds (%)",
                             "Wins still needed (projected cut line)",
                             "Weeks remaining"]
    assert rows[1]["value"] == "18.2"
    assert "Wins still needed (projected)" not in _labels(rows)


def test_playoffs_receipts_without_bye_pct_have_no_bye_row():
    pack = dict(PLAYOFFS_PACK)
    pack["me"] = {"team_name": "Caleb's Casting Couch", "playoff_pct": 71.5,
                  "wins_still_needed": 3.9}
    rows = compose_mod._required_receipts("playoffs", pack)
    assert "First-round bye odds (%)" not in _labels(rows)
    assert "Wins still needed (projected cut line)" in _labels(rows)


# -- schedule receipts: remaining slate first ---------------------------------

SCHEDULE_PACK = {
    "type": "schedule", "available": True,
    "me": {"roster_id": "3", "team_name": "Caleb's Casting Couch",
           "opp_index": 99.8},
    "playoff_stretch": {"me": {"opp_index": 104.3}},
    "easiest": {"roster_id": "6", "team_name": "Lamb Chop",
                "opp_index": 91.2},
}


def test_schedule_receipts_remaining_index_leads():
    rows = compose_mod._required_receipts("schedule", SCHEDULE_PACK)
    assert _labels(rows) == ["Your remaining opponent index",
                             "Your playoff stretch opponent index",
                             "Lamb Chop opponent index"]
    assert rows[0]["value"] == "99.8"
    assert rows[1]["value"] == "104.3"


# -- standings receipts: subject-aware -----------------------------------------

def _standings_pack():
    teams = [
        {"seed": 1, "roster_id": "1", "team_name": "Pittsburgh Pilots",
         "pf": 845.2, "is_me": False},
        {"seed": 3, "roster_id": "3", "team_name": "Caleb's Casting Couch",
         "pf": 812.4, "is_me": True},
        {"seed": 12, "roster_id": "12", "team_name": "Burrowed Time",
         "pf": 640.5, "is_me": False},
    ]
    return {"type": "standings", "available": True, "teams": teams,
            "me": teams[1],
            "points_leader": {"team_name": "Pittsburgh Pilots", "pf": 845.2}}


def test_standings_last_place_question_receipts_are_the_last_place_row():
    rows = compose_mod._required_receipts(
        "standings", _standings_pack(), "Who is in last place?")
    assert _labels(rows) == ["Burrowed Time seed", "Burrowed Time points for"]
    assert [r["value"] for r in rows] == ["12", "640.5"]
    assert "Your seed" not in _labels(rows)


def test_standings_points_leader_question_receipts_are_the_leader_row():
    rows = compose_mod._required_receipts(
        "standings", _standings_pack(), "Who is the points leader?")
    assert _labels(rows) == ["Pittsburgh Pilots seed",
                             "Pittsburgh Pilots points for"]
    assert "Your seed" not in _labels(rows)


def test_standings_named_team_question_receipts_are_that_team():
    rows = compose_mod._required_receipts(
        "standings", _standings_pack(), "Where do the Pittsburgh Pilots sit?")
    assert _labels(rows) == ["Pittsburgh Pilots seed",
                             "Pittsburgh Pilots points for"]


def test_standings_default_and_my_questions_keep_viewer_receipts():
    for question in (None, "What do the standings look like?",
                     "What seed am I?", "How are my points?"):
        rows = compose_mod._required_receipts(
            "standings", _standings_pack(), question)
        assert _labels(rows) == ["Your seed", "Your points for",
                                 "Pittsburgh Pilots points for"], question


# -- activity receipts: subject-aware -------------------------------------------

ACTIVITY_PACK = {
    "type": "activity", "available": True,
    "counts": {"trades": 6, "trades_this_week": 1, "waiver_adds": 23,
               "injuries": 1},
    "latest_trade": {"value_gap": 70.0, "teams": [
        {"team_name": "Pittsburgh Pilots", "gets_value": 4680.0},
        {"team_name": "Caleb's Casting Couch", "gets_value": 4750.0}]},
    "recent_adds": [],
}


def test_activity_waiver_question_receipts_are_counts_not_trade_totals():
    rows = compose_mod._required_receipts(
        "activity", ACTIVITY_PACK, "What waiver moves happened this week?")
    assert rows == [{"label": "Waiver adds this season", "value": "23"}]
    assert "Fairness gap" not in _labels(rows)


def test_activity_trade_question_keeps_latest_trade_receipts():
    for question in (None, "What trades have gone down recently?",
                     "Any trades and waiver moves this week?"):
        rows = compose_mod._required_receipts(
            "activity", ACTIVITY_PACK, question)
        assert _labels(rows) == [
            "Pittsburgh Pilots received (total value)",
            "Caleb's Casting Couch received (total value)",
            "Fairness gap"], question


# -- metric receipts: definitions force no leader value -------------------------

METRIC_PACK = {
    "type": "metric", "available": True, "week": 7,
    "metric_key": "target_share", "label": "Target Share",
    "definition": "Share of team pass attempts.",
    "leaderboard_top5": [
        {"player_id": "jc", "name": "Ja'Marr Chase", "team": "CIN",
         "position": "WR", "value": 0.31, "on_my_roster": False}],
    "player_ranks": [],
}


def test_metric_definition_question_forces_no_receipts():
    rows = compose_mod._required_receipts(
        "metric", METRIC_PACK, "What is target share?")
    assert rows == []


def test_metric_leader_questions_keep_leader_receipt():
    for question in (None, "Who leads in target share?",
                     "What is target share and who leads?"):
        rows = compose_mod._required_receipts(
            "metric", METRIC_PACK, question)
        assert _labels(rows) == ["Ja'Marr Chase"], question


def test_metric_definition_with_named_player_keeps_receipts():
    pack = dict(METRIC_PACK)
    pack["player_ranks"] = [{"name": "Ja'Marr Chase", "value": 0.31}]
    rows = compose_mod._required_receipts(
        "metric", pack, "What is target share for Ja'Marr Chase?")
    assert rows  # a named player is in play: today's set applies


# -- end to end: the retry and injection use the same subject-aware set --------

def _canned(monkeypatch, draft):
    calls = {"n": 0}

    def _fn(*a, **k):
        calls["n"] += 1
        return dict(draft)

    monkeypatch.setattr(compose_mod, "call_model_json", _fn)
    return calls


def test_compose_standings_last_place_answer_carries_last_place_receipts(monkeypatch):
    _canned(monkeypatch, {
        "verdict": "Burrowed Time sits last",
        "text": "Burrowed Time is in last place in the league.",
        "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_mod.compose_answer(
        "standings", "Who is in last place?", _standings_pack())
    labels = _labels(answer["receipts"])
    assert "Burrowed Time seed" in labels
    assert "Burrowed Time points for" in labels
    assert "Your seed" not in labels
    assert "Your points for" not in labels


def test_compose_activity_waiver_answer_has_no_trade_totals(monkeypatch):
    _canned(monkeypatch, {
        "verdict": "Waiver wire stayed busy",
        "text": "The waiver wire stayed busy this week.",
        "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_mod.compose_answer(
        "activity", "What waiver moves happened this week?", ACTIVITY_PACK)
    labels = _labels(answer["receipts"])
    assert labels == ["Waiver adds this season"]
    assert not any("received" in label for label in labels)


def test_compose_metric_definition_answer_gets_no_leader_receipt(monkeypatch):
    _canned(monkeypatch, {
        "verdict": "Target share defined",
        "text": "Target share is the share of team pass attempts a "
                "player earns.",
        "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_mod.compose_answer(
        "metric", "What is target share?", METRIC_PACK)
    assert answer["receipts"] == []
    assert "Ja'Marr Chase" not in answer["text"]
