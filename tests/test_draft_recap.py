"""Tests for the Draft Recap feature.

Pure-helper tests (utils/draft_recap.py) run anywhere: verdict bands,
competition ranks, the median team leaderboard, the first-half bust gate,
and the curated story lists. Payload/API tests import the app lazily inside
each test (guarded by importorskip) and drive _draft_recap_payload with
fakes shaped like the real Sleeper payloads, so the class-vs-pool rank math
is verified end to end without network or a database.
"""
from types import SimpleNamespace

import pytest

from utils.draft_recap import (
    first_half_bust_pool,
    injury_exemption,
    injury_span_exemption,
    pick_superlatives,
    rank_values,
    story_lists,
    team_leaderboard,
    verdict_for,
)


# ── Pure helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("delta,label", [
    (20, "Steal"), (8, "Steal"),
    (7, "Hit"), (3, "Hit"),
    (2, "As drafted"), (0, "As drafted"), (-2, "As drafted"),
    (-3, "Reach"), (-7, "Reach"),
    (-8, "Bust"), (-121, "Bust"),
    (None, "No stats"),
])
def test_verdict_bands(delta, label):
    assert verdict_for(delta)[0] == label


def test_rank_values_competition_ties():
    ranks = rank_values({"a": 5.0, "b": 5.0, "c": 9.0, "d": 1.0})
    assert ranks == {"a": 2, "b": 2, "c": 1, "d": 4}
    assert rank_values({}) == {}


def _row(pid, pick_no, delta, team="T", roster_id=1, total=10.0, rnd=1,
         delta_adp=None, pos="WR"):
    return {"player_id": pid, "pick_no": pick_no, "round": rnd,
            "roster_id": roster_id, "team": team, "name": pid, "pos": pos,
            "total": total, "delta_pick": delta, "delta_adp": delta_adp}


def test_bust_gate_excludes_late_flier():
    rows = [
        _row("early_bust", 3, -5),
        _row("late_flier", 18, -30),   # worse delta, but a last-round flier
        _row("fine", 1, 0),
    ]
    pool = first_half_bust_pool(rows, total_picks=20)
    assert [r["player_id"] for r in pool] == ["early_bust", "fine"]
    sup = pick_superlatives(rows, total_picks=20)
    assert sup["bust"]["player_id"] == "early_bust"
    assert sup["steal"]["player_id"] == "fine"


def test_team_leaderboard_median_not_mean():
    rows = [
        _row("a1", 1, 10, team="Alpha", roster_id=1, total=50.0),
        _row("a2", 2, -30, team="Alpha", roster_id=1, total=5.0),
        _row("a3", 3, 10, team="Alpha", roster_id=1, total=40.0),
        _row("b1", 4, 0, team="Beta", roster_id=2, total=30.0),
        _row("b2", 5, 0, team="Beta", roster_id=2, total=30.0),
    ]
    board = team_leaderboard(rows)
    # Alpha's MEAN is -3.3 (the -30 outlier), but its MEDIAN is +10, so it
    # still wins the leaderboard; Beta's flat 0 median ranks second.
    assert [t["team"] for t in board] == ["Alpha", "Beta"]
    alpha = board[0]
    assert alpha["median"] == 10.0
    assert alpha["draft_points"] == 95.0
    assert alpha["hits"] == 2 and alpha["graded"] == 3
    assert alpha["best"]["player_id"] in ("a1", "a3")
    assert alpha["worst"]["player_id"] == "a2"


# ── Injury exemption (pure) ──────────────────────────────────────────────────

def test_injury_exemption_current_designation():
    assert injury_exemption("IR", "Hamstring", None, None, 6) == "IR · Hamstring"
    assert injury_exemption("PUP", "", None, None, 6) == "PUP"
    assert injury_exemption("ir", None, None, None, 6) == "IR"
    # Lesser designations never exempt on their own.
    assert injury_exemption("Questionable", "Knee", None, None, 6) is None
    assert injury_exemption("Out", "Knee", None, None, 6) is None
    assert injury_exemption("", "", None, None, 6) is None
    assert verdict_for(None)[0] == "No stats"  # unchanged default


def test_injury_exemption_history_thresholds():
    # Out for 5 of 6 completed weeks exempts; 4 of 6 does not.
    assert injury_exemption(None, None, {1, 2, 3, 4, 5}, None, 6) == "Out W1-W5"
    assert injury_exemption(None, None, {1, 2, 3, 4}, None, 6) is None
    assert injury_exemption(None, None, {2, 3, 4, 5, 6}, None, 6) == "Out W2-W6"
    # Reserve weeks (weekly rosters) work the same way.
    assert injury_exemption(None, None, None, {1, 2}, 2) == "Reserve W1-W2"
    # Scattered coverage is described, not ranged.
    assert injury_exemption(None, None, {1, 3, 4, 5, 6}, None, 6) == "Out in 5 of 6 weeks"
    # History unavailable: the arm cannot fire, even for a full-looking set,
    # and one completed week is never enough for the history arm.
    assert injury_exemption(None, None, None, None, 6) is None
    assert injury_exemption(None, None, {1}, None, 1) is None


def test_injury_span_exemption_duration_gate():
    # Appeared-but-scoreless picks exempt on DURATION alone: the Out or
    # Reserve span must cover at least completed_weeks - 1 weeks.
    assert injury_span_exemption(None, {2, 3, 4, 5, 6}, 6, "Ankle") == \
        "Reserve · missed 5 of 6 · Ankle"
    assert injury_span_exemption({1, 2, 3, 4, 5}, None, 6) == \
        "Out · missed 5 of 6"
    # 4 of 6 is not "injured all but one week": no exemption.
    assert injury_span_exemption(None, {3, 4, 5, 6}, 6) is None
    assert injury_span_exemption({1, 2, 3, 4}, None, 6) is None
    # One completed week, or unavailable history, can never fire the gate.
    assert injury_span_exemption(None, {1}, 1) is None
    assert injury_span_exemption(None, None, 6) is None


def test_injury_history_loader_ignores_exe(monkeypatch):
    # The weekly-roster EXE (exempt list / personal leave) status is not an
    # injury absence: it must never enter the reserve span (personal leave
    # still busts). Only RES weeks and Out report weeks count.
    from dashboard_services import nfl_game_data as ngd
    inj = [{"gsis_id": "00-1", "week": "3", "report_status": "Out"},
           {"gsis_id": "00-2", "week": "3", "report_status": "Questionable"}]
    rosters = [
        {"gsis_id": "00-1", "sleeper_id": "9001", "week": "3", "status": "RES"},
        {"gsis_id": "00-3", "sleeper_id": "9003", "week": "3", "status": "EXE"},
        {"gsis_id": "00-3", "sleeper_id": "9003", "week": "4", "status": "EXE"},
    ]
    monkeypatch.setattr(
        ngd, "_nflverse_release_csv_rows",
        lambda url, filename: inj if "injuries" in filename else rosters)
    ngd._INJURY_HISTORY_CACHE.clear()
    try:
        hist = ngd.nflverse_injury_history(2099)
    finally:
        ngd._INJURY_HISTORY_CACHE.clear()
    assert hist["out"] == {"00-1": {3}}
    assert hist["res"] == {"00-1": {3}, "9001": {3}}


def test_story_lists_rookie_per_round_and_redraft_groups():
    rows = [
        _row("r1best", 1, 5, rnd=1), _row("r1ok", 2, 1, rnd=1),
        _row("r2best", 3, 9, rnd=2), _row("r2ok", 4, -1, rnd=2),
    ]
    lists = story_lists(rows, "rookie", total_picks=4)
    assert [g["label"] for g in lists["by_round"]] == ["Round 1", "Round 2"]
    assert lists["by_round"][0]["row"]["player_id"] == "r1best"
    assert lists["by_round"][1]["row"]["player_id"] == "r2best"
    assert lists["steals"][0]["player_id"] == "r2best"

    rows_rd = [
        _row("early", 1, 4, rnd=2), _row("mid", 40, 6, rnd=7),
        _row("late", 90, 8, rnd=13), _row("worst", 3, -9, rnd=1),
    ]
    lists_rd = story_lists(rows_rd, "redraft", total_picks=100)
    assert [g["label"] for g in lists_rd["by_round"]] == [
        "Rounds 1-4", "Rounds 5-10", "Rounds 11+"]
    assert lists_rd["busts"][0]["player_id"] == "worst"


# ── Payload math with fakes shaped like the real payloads ────────────────────

_NAMES = {
    "101": ("Alpha Back", "RB"), "102": ("Bravo Wide", "WR"),
    "103": ("Charlie Wide", "WR"), "104": ("Delta Back", "RB"),
    "105": ("Echo Wide", "WR"), "106": ("Foxtrot Back", "RB"),
    "107": ("Golf Wide", "WR"), "108": ("Hotel Back", "RB"),
    "201": ("Pool Wide", "WR"), "202": ("Pool Back", "RB"),
}
_WEEKLY = {
    "101": [10.0, 10.0], "102": [30.0, 30.0], "103": [5.0, 5.0],
    "104": [20.0, 20.0], "105": [15.0, 15.0], "107": [25.0, 25.0],
    "108": [1.0, 1.0], "201": [30.0], "202": [40.0],
    # "106" never played: absent from the weekly files entirely.
}
_PICKS = [  # (pick_no, round, slot, roster_id, pid)
    (1, 1, 1, 1, "101"), (2, 1, 2, 2, "102"), (3, 1, 3, 3, "103"),
    (4, 1, 4, 4, "104"), (5, 2, 1, 3, "105"), (6, 2, 2, 4, "106"),
    (7, 2, 3, 1, "107"), (8, 2, 4, 2, "108"),
]
_ROOKIE_ADP = {  # the crawler-shaped league-ADP map (avg_pick per player)
    "101": {"avg_pick": 3.0}, "102": {"avg_pick": 8.0},
    "103": {"avg_pick": 1.5}, "104": {"avg_pick": 2.0},
    "105": {"avg_pick": 4.0}, "106": {"avg_pick": 5.0},
    "107": {"avg_pick": 2.5}, "108": {"avg_pick": 6.0},
}


def _fake_app(monkeypatch, rounds=2, draft_type="linear", weekly=None,
              players_feed=None, injury_history=None):
    pytest.importorskip("flask")
    pytest.importorskip("pandas")
    pytest.importorskip("openai")
    import app as appmod
    from dashboard_services import adp_service

    draft = {"draft_id": "D1", "type": draft_type, "season": 2026,
             "settings": {"rounds": rounds, "teams": 4},
             "slot_to_roster_id": {"1": 1, "2": 2, "3": 3, "4": 4}}
    picks = [{"pick_no": pn, "round": rd, "draft_slot": sl, "roster_id": rid,
              "player_id": pid,
              "metadata": {"first_name": _NAMES[pid][0].split()[0],
                           "last_name": _NAMES[pid][0].split()[1],
                           "position": _NAMES[pid][1], "team": "BUF"}}
             for pn, rd, sl, rid, pid in _PICKS]
    league = {"name": "Fake League", "scoring_settings": {"rec": 1.0},
              "roster_positions": ["QB", "RB", "WR", "TE", "FLEX"]}
    rosters = [{"roster_id": i, "owner_id": f"u{i}",
                "metadata": {"team_name": f"Team {i}"}} for i in (1, 2, 3, 4)]
    index = {pid: {"name": name, "pos": pos, "team": "BUF", "espnID": ""}
             for pid, (name, pos) in _NAMES.items()}

    monkeypatch.setattr(appmod, "get_drafts", lambda *a, **k: [draft])
    monkeypatch.setattr(appmod, "get_draft_picks", lambda *a, **k: picks)
    monkeypatch.setattr(appmod, "get_league", lambda *a, **k: league)
    monkeypatch.setattr(appmod, "get_rosters", lambda *a, **k: rosters)
    monkeypatch.setattr(appmod, "get_users", lambda *a, **k: [])
    monkeypatch.setattr(appmod, "load_players_index", lambda *a, **k: index)
    monkeypatch.setattr(appmod, "_load_season_weekly_points",
                        lambda *a, **k: dict(_WEEKLY if weekly is None else weekly))
    # Injury-rule seams: no current designations and history unavailable
    # unless a test wires them up (the payload must fail open, not crash).
    monkeypatch.setattr(appmod, "get_players_global",
                        lambda *a, **k: dict(players_feed or {}))
    monkeypatch.setattr(appmod, "_draft_recap_injury_history",
                        lambda *a, **k: injury_history)
    monkeypatch.setattr(appmod, "_fetch_league_adp_from_db",
                        lambda *a, **k: dict(_ROOKIE_ADP))
    monkeypatch.setattr(adp_service, "resolve_market_adp", lambda *a, **k: {})
    import utils.season_qualification as sq
    monkeypatch.setattr(sq, "qualification_policy",
                        lambda season: SimpleNamespace(completed_weeks=(1, 2)))
    appmod._DRAFT_RECAP_CACHE.clear()
    return appmod


def test_payload_rookie_grades_within_class(monkeypatch):
    appmod = _fake_app(monkeypatch, rounds=2)
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    assert payload["status"] == "ok" and payload["mode"] == "rookie"
    assert payload["picks_total"] == 8 and payload["picks_graded"] == 8
    assert payload["through_week"] == 2
    by_pid = {r["player_id"]: r for r in payload["picks"]}

    # Golf Wide was the 4th WR drafted but finished WR2 *in the class* (+2),
    # even though the non-drafted Pool Wide pushes his whole-pool finish to 3.
    golf = by_pid["107"]
    assert golf["draft_pos_rank"] == 4
    assert golf["class_finish"] == 2
    assert golf["finish_ppg"] == 3
    assert golf["delta_pick"] == 2

    # Alpha Back: drafted RB1, class RB2 (-1); whole-pool RB3 behind Pool Back.
    alpha = by_pid["101"]
    assert alpha["class_finish"] == 2 and alpha["finish_ppg"] == 3
    assert alpha["delta_pick"] == -1

    # Foxtrot Back never played and has no injury exemption (no current
    # designation, history unavailable): he grades at the bottom of the
    # drafted RB class (3 RBs appeared, so he finishes RB4) instead of
    # sitting ungraded. Drafted RB3, delta -1.
    fox = by_pid["106"]
    assert fox["graded"] is True and fox["injury_exempt"] is False
    assert fox["class_finish"] == 4 and fox["delta_pick"] == -1
    assert fox["verdict"] == "As drafted"
    # History was consulted for him and was unavailable: flagged, not faked.
    assert payload["injury_history_available"] is False

    # ADP from the crawler-shaped map, ranked within the drafted pool:
    # Bravo Wide's market ADP (8.0) made him the 4th WR by ADP; he finished
    # WR1 in class, so his vs-ADP delta is +3 and he is the Best vs ADP card.
    bravo = by_pid["102"]
    assert bravo["adp"] == 8.0 and bravo["adp_pos_rank"] == 4
    assert bravo["delta_adp"] == 3
    assert payload["superlatives"]["best_vs_adp"]["player_id"] == "102"
    assert payload["superlatives"]["steal"]["player_id"] == "107"
    # Slot-owner headers resolve through the roster map.
    assert payload["slot_teams"]["1"] == "Team 1"


def test_payload_redraft_grades_against_whole_pool(monkeypatch):
    appmod = _fake_app(monkeypatch, rounds=12, draft_type="snake")
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    assert payload["status"] == "ok" and payload["mode"] == "redraft"
    by_pid = {r["player_id"]: r for r in payload["picks"]}
    # Same player, redraft basis: whole-pool finish (WR3), not class (WR2).
    assert by_pid["107"]["delta_pick"] == 4 - 3
    assert by_pid["101"]["delta_pick"] == 1 - 3
    # Foxtrot's bottom grade is against the whole pool too: 4 pool RBs
    # appeared (101/104/108 drafted + 202 undrafted), so he finishes RB5.
    assert by_pid["106"]["finish_ppg"] == 5
    assert by_pid["106"]["delta_pick"] == 3 - 5


# ── Injury exemption (payload level) ─────────────────────────────────────────

def test_payload_injury_exemption_current_ir(monkeypatch):
    feed = {"106": {"injury_status": "IR", "injury_body_part": "Hamstring",
                    "gsis_id": "00-0099999"}}
    appmod = _fake_app(monkeypatch, rounds=2, players_feed=feed)
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    by_pid = {r["player_id"]: r for r in payload["picks"]}
    fox = by_pid["106"]
    assert fox["injury_exempt"] is True
    assert fox["injury_evidence"] == "IR · Hamstring"
    assert fox["verdict"] == "Injured"
    assert fox["graded"] is False and fox["delta_pick"] is None
    assert payload["picks_graded"] == 7
    # Arm (a) settled it, so history was never consulted: still "available".
    assert payload["injury_history_available"] is True
    # Excluded from the leaderboard exactly like a No stats pick: Team 4's
    # graded count covers only its other pick.
    team4 = [t for t in payload["teams"] if str(t["roster_id"]) == "4"][0]
    assert team4["graded"] == 1


def test_payload_injury_exemption_out_history(monkeypatch):
    feed = {"106": {"injury_status": "", "gsis_id": "00-0099999"}}
    history = {"out": {"00-0099999": {1, 2}}, "res": {}}
    appmod = _fake_app(monkeypatch, rounds=2, players_feed=feed,
                       injury_history=history)
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    fox = {r["player_id"]: r for r in payload["picks"]}["106"]
    assert fox["injury_exempt"] is True
    assert fox["injury_evidence"] == "Out W1-W2"
    assert fox["verdict"] == "Injured" and fox["graded"] is False
    assert payload["injury_history_available"] is True


def test_payload_injury_exemption_reserve_history(monkeypatch):
    # Reserve weeks keyed by Sleeper id in the roster file (no gsis needed).
    history = {"out": {}, "res": {"106": {1, 2}}}
    appmod = _fake_app(monkeypatch, rounds=2, players_feed={},
                       injury_history=history)
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    fox = {r["player_id"]: r for r in payload["picks"]}["106"]
    assert fox["injury_exempt"] is True
    assert fox["injury_evidence"] == "Reserve W1-W2"


def _scoreless_hotel_weekly():
    # Hotel Back (108) dresses for both weeks and scores exactly 0.
    return {**_WEEKLY, "108": [0.0, 0.0]}


def test_payload_scoreless_played_exempt_on_span(monkeypatch):
    # Appeared (2 games) but scoreless, and on reserve for the whole
    # completed stretch: exempt on duration, with the span as evidence.
    feed = {"108": {"injury_status": "IR", "injury_body_part": "Ankle",
                    "gsis_id": "00-0010808"}}
    history = {"out": {}, "res": {"108": {1, 2}}}
    appmod = _fake_app(monkeypatch, rounds=2,
                       weekly=_scoreless_hotel_weekly(),
                       players_feed=feed, injury_history=history)
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    by_pid = {r["player_id"]: r for r in payload["picks"]}
    hotel = by_pid["108"]
    assert hotel["g"] == 2 and hotel["total"] == 0.0
    assert hotel["injury_exempt"] is True
    assert hotel["injury_evidence"] == "Reserve · missed 2 of 2 · Ankle"
    assert hotel["verdict"] == "Injured"
    assert hotel["graded"] is False and hotel["delta_pick"] is None
    assert payload["picks_graded"] == 7
    # Off the leaderboard's graded math like any other exempt pick.
    team2 = [t for t in payload["teams"] if str(t["roster_id"]) == "2"][0]
    assert team2["graded"] == 1


def test_payload_scoreless_played_no_span_grades(monkeypatch):
    # The control case (a healthy backup who dressed and never scored):
    # history is available but holds no Out/reserve weeks for him, so the
    # duration gate cannot fire and he grades on his 0.0 PPG like anyone.
    history = {"out": {}, "res": {}}
    appmod = _fake_app(monkeypatch, rounds=2,
                       weekly=_scoreless_hotel_weekly(),
                       players_feed={}, injury_history=history)
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    hotel = {r["player_id"]: r for r in payload["picks"]}["108"]
    assert hotel["injury_exempt"] is False
    assert hotel["graded"] is True
    # RB class: 104 (20 PPG) 1st, 101 (10) 2nd, Hotel (0.0) 3rd; he was the
    # 4th RB drafted, so the bottom grade is +1, not a bust.
    assert hotel["class_finish"] == 3 and hotel["delta_pick"] == 1
    assert hotel["verdict"] == "As drafted"
    assert payload["injury_history_available"] is True


def test_payload_scoreless_played_current_tag_alone_does_not_exempt(monkeypatch):
    # For an APPEARED pick, a current IR tag without the duration span is
    # not enough: the approved gate is "injured all but one week".
    feed = {"108": {"injury_status": "IR", "injury_body_part": "Knee",
                    "gsis_id": "00-0010808"}}
    history = {"out": {}, "res": {}}
    appmod = _fake_app(monkeypatch, rounds=2,
                       weekly=_scoreless_hotel_weekly(),
                       players_feed=feed, injury_history=history)
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    hotel = {r["player_id"]: r for r in payload["picks"]}["108"]
    assert hotel["injury_exempt"] is False and hotel["graded"] is True


def test_payload_scored_played_never_exemption_tested(monkeypatch):
    # A player with any points grades on what he did even with a full
    # reserve span on file: the exemption test never runs for him.
    history = {"out": {}, "res": {"108": {1, 2}}}
    appmod = _fake_app(monkeypatch, rounds=2, players_feed={},
                       injury_history=history)
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    hotel = {r["player_id"]: r for r in payload["picks"]}["108"]
    assert hotel["total"] == 2.0
    assert hotel["injury_exempt"] is False and hotel["graded"] is True
    assert hotel["delta_pick"] == 1


def test_payload_zero_appearance_bottom_grade_and_bust_gate(monkeypatch):
    # Bravo Wide (pick 2, the first WR taken) also never plays in this
    # variant. Healthy scratches share the bottom class rank.
    weekly = {k: v for k, v in _WEEKLY.items() if k not in ("102", "106")}
    appmod = _fake_app(monkeypatch, rounds=2, weekly=weekly)
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    by_pid = {r["player_id"]: r for r in payload["picks"]}
    bravo, fox = by_pid["102"], by_pid["106"]
    # WR class: 3 appeared (103/105/107), so both... only Bravo is a WR:
    # bottom WR finish = 4. RB class: 3 appeared, Foxtrot bottom = 4 too.
    assert bravo["class_finish"] == 4 and bravo["delta_pick"] == 1 - 4
    assert fox["class_finish"] == 4
    # Bravo is a first-half pick and now the worst graded one: he takes the
    # bust card (the first-half gate still applies to bottom grades).
    assert payload["superlatives"]["bust"]["player_id"] == "102"
    assert payload["lists"]["busts"][0]["player_id"] == "102"
    # Team 2's median includes the bottom grade (Bravo -3, Hotel +1).
    team2 = [t for t in payload["teams"] if str(t["roster_id"]) == "2"][0]
    assert team2["graded"] == 2 and team2["median"] == -1.0

    # Same draft with Bravo on IR instead: he is exempt, leaves the bust
    # gate entirely, and Team 2's median is Hotel's +1 alone.
    feed = {"102": {"injury_status": "IR", "injury_body_part": "Ankle"}}
    appmod = _fake_app(monkeypatch, rounds=2, weekly=weekly, players_feed=feed)
    payload = appmod._draft_recap_payload("sleeper", "FAKE", 2026)
    by_pid = {r["player_id"]: r for r in payload["picks"]}
    assert by_pid["102"]["verdict"] == "Injured"
    assert payload["superlatives"]["bust"]["player_id"] != "102"
    assert all(r["player_id"] != "102" for r in payload["lists"]["busts"])
    team2 = [t for t in payload["teams"] if str(t["roster_id"]) == "2"][0]
    assert team2["graded"] == 1 and team2["median"] == 1.0


# ── API honest states ────────────────────────────────────────────────────────

def test_api_no_draft_is_404(offline_client, monkeypatch):
    appmod = _fake_app(monkeypatch)
    monkeypatch.setattr(appmod, "get_drafts", lambda *a, **k: [])
    appmod._DRAFT_RECAP_CACHE.clear()
    resp = offline_client.get("/api/draft-recap?league_id=NONE&season=2026")
    assert resp.status_code == 404
    assert resp.get_json()["error"]


def test_api_auction_is_honest_unsupported(offline_client, monkeypatch):
    appmod = _fake_app(monkeypatch, draft_type="auction")
    appmod._DRAFT_RECAP_CACHE.clear()
    resp = offline_client.get("/api/draft-recap?league_id=FAKE&season=2026")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "unsupported" and body["supported"] is False


def test_api_requires_league_id(offline_client):
    pytest.importorskip("flask")
    resp = offline_client.get("/api/draft-recap")
    assert resp.status_code == 400


def test_page_renders_curated_sections(monkeypatch):
    _fake_app(monkeypatch, rounds=2)
    from dashboard_services.pages.draft_recap_page import build_draft_recap_body
    body = build_draft_recap_body("FAKE", 2026, "sleeper")
    assert "Draft Recap" in body
    assert "Team leaderboard" in body
    assert "Show full board" in body
    assert "Golf Wide" in body
    # No initials fallback anywhere on this page: the avatar renderer has no
    # initials code path, and no initials text is emitted for any player.
    assert ">GW<" not in body and "drc-av-init" not in body
