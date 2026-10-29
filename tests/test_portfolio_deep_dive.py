"""Tests for /api/portfolio-deep-dive (the Front Office board's
"Generate full report").

The report used to be composed from aggregates alone (posture tallies,
moves-by-posture, the top move's title), so a 5-move board produced
three restatements of the same counts and no per-move reasoning. The
builder now walks the board's own moves in rank order; every line must
trace to a payload field. Calendar nudges (kind="calendar") are dates,
not actions: they are never numbered or counted, they render as
reminder lines after the plan. Round 3 adds a timeline roll-up in
place of the posture ordering line, reasons that never restate their
title, done lines that carry titles, a coverage line for the leagues
that need nothing, and labeled sections alongside the flat bullet
list. There is deliberately NO cross-league conflict callout (an
earlier draft flagged the same player claimed in two leagues and
told the user to "prioritize one claim"; Kaedon corrected that
premise on 2026-10-29: separate leagues have independent wires,
FAAB and rosters, so a same-player claim in two leagues, or a claim
in one league and a drop in another, is an ordinary per-league
decision). The report text carries no em dashes anywhere (Kaedon's
standing UI rule: heads join titles with a colon, sentences with
periods). These tests pin that contract against a fixture board
shaped exactly like the real payload, plus the honest degradation
paths (partial scan refused, empty board)."""
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _move(rank, kind, title, detail="", evidence=None, timing=None,
          done=None):
    return {
        "rank": rank, "kind": kind, "title": title, "detail": detail,
        "evidence": evidence or [],
        "timing": timing or {"label": "This week", "locked": False,
                             "lock_label": ""},
        "done": done, "move_key": f"pf:{kind}:{rank}",
        "platform": "sleeper", "league_id": "x", "season": 2026,
    }


def _board_payload(scan_incomplete=False):
    """Five open actions plus a done action over four leagues, and a
    trade-deadline calendar nudge riding in the Dynasty Rebuild group,
    from an 11-league scan: the shape behind the 2026-10-29 report
    Kaedon flagged as 'super weak', with the payload substance filled
    in. The calendar nudge is a reminder, not a move: it takes no
    number and no slot in the count. Woody Marks is the add target in
    two leagues (board ranks 3 and 5): NOT a conflict, since separate
    leagues run independent wires; the fixture pins that both claims
    render as ordinary steps and no Watch line fires."""
    groups = [
        {
            "key": "sleeper|111", "platform": "sleeper",
            "league_id": "111", "league_name": "FantasyPros",
            "meta": {"record_line": "6-1 · 1st of 12",
                     "posture": "Contender", "odds_pct": 78.0},
            "moves": [
                _move(1, "lineup", "Ka'imi Fairbairn on bye", detail="",
                      timing={"label": "Before Sunday", "locked": False,
                              "lock_label": ""}),
                _move(3, "waiver", "Add Woody Marks (RB)",
                      detail=("RB24 on the season and the wire's best "
                              "remaining back · Drop Dameon Pierce (RB)"),
                      evidence=[
                          {"k": "Add", "v": "Woody Marks · value 1,240"},
                          {"k": "Why", "v": "RB24 on the season and the "
                                            "wire's best remaining back"},
                          {"k": "Drop", "v": "Dameon Pierce · RB"},
                      ],
                      timing={"label": "Before waivers", "locked": False,
                              "lock_label": ""}),
            ],
        },
        {
            "key": "sleeper|222", "platform": "sleeper",
            "league_id": "222", "league_name": "Blackedraw",
            "meta": {"record_line": "5-2 · 2nd of 12",
                     "posture": "Contender", "odds_pct": 71.0},
            "moves": [
                _move(2, "lineup", "Start Jaylen Waddle over Tee Higgins",
                      detail=("Jaylen Waddle projects 14.2 vs "
                              "Tee Higgins's 9.1"),
                      evidence=[
                          {"k": "Start", "v": "Jaylen Waddle · proj 14.2"},
                          {"k": "Bench", "v": "Tee Higgins · proj 9.1"},
                          {"k": "Gain", "v": "+5.1 projected pts"},
                      ],
                      timing={"label": "Locks 1:00 PM", "locked": False,
                              "lock_label": ""}),
                _move(5, "waiver", "Add Woody Marks (RB)",
                      detail=("RB24 on the season and the wire's best "
                              "remaining back"),
                      evidence=[
                          {"k": "Add", "v": "Woody Marks · value 1,240"},
                          {"k": "Why", "v": "RB24 on the season and the "
                                            "wire's best remaining back"},
                      ],
                      timing={"label": "Before waivers", "locked": False,
                              "lock_label": ""}),
            ],
        },
        {
            "key": "sleeper|333", "platform": "sleeper",
            "league_id": "333", "league_name": "Dynasty Rebuild",
            "meta": {"record_line": "2-5 · 9th of 12",
                     "posture": "Rebuilding", "odds_pct": 4.0},
            "moves": [
                _move(4, "injury", "Move Trey Benson to IR",
                      detail=("1 of 2 IR slots used · Approx return "
                              "Week 11"),
                      evidence=[
                          {"k": "Status", "v": "Out"},
                          {"k": "Time out", "v": "Approx return Week 11"},
                          {"k": "IR room", "v": "1 of 2 slots used"},
                      ],
                      timing={"label": "Do now", "locked": False,
                              "lock_label": ""}),
                _move(7, "calendar", "Trade deadline in 1 week",
                      detail=("Deadline is Week 10. Line up any deals "
                              "before the wire closes.")),
            ],
        },
        {
            "key": "sleeper|444", "platform": "sleeper",
            "league_id": "444", "league_name": "League of Greatness",
            "meta": {"record_line": "4-3 · 6th of 12",
                     "posture": "Fringe", "odds_pct": 38.0},
            "moves": [
                _move(6, "lineup", "Start Brock Bowers over Dalton Kincaid",
                      detail=("Brock Bowers projects 12.6 vs Dalton "
                              "Kincaid's 8.4"),
                      evidence=[
                          {"k": "Start", "v": "Brock Bowers · proj 12.6"},
                          {"k": "Bench", "v": "Dalton Kincaid · proj 8.4"},
                          {"k": "Gain", "v": "+4.2 projected pts"},
                      ],
                      done={"verified": True, "label": "Verified done"}),
            ],
        },
    ]
    return {
        "ok": True,
        "scan": {"total": 11, "scanned": 11, "skipped_cold": 0,
                 "skipped_no_roster": 0, "incomplete": scan_incomplete},
        "board": {"groups": groups, "week": 9},
        "week": 9, "season": 2026,
    }


_DONE_LINE = ("Already marked done: Move 6, Start Brock Bowers over "
              "Dalton Kincaid (League of Greatness).")
_TIMELINE_LINE = ("Timeline: Move 3 and Move 5 go before waivers run. "
                  "Move 2 locks 1:00 PM. Move 1 and Move 4 have no "
                  "hard deadline.")
class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def get_json(self):
        return self._payload


@pytest.fixture
def dd_client(offline_client, monkeypatch):
    import routes.user_pages_bp as bp

    def _stub(payload):
        monkeypatch.setattr(bp, "api_portfolio_actions",
                            lambda: _Resp(payload))
        return offline_client

    return _stub


def _post(client):
    resp = client.post("/api/portfolio-deep-dive", json={})
    assert resp.status_code == 200
    return resp.get_json()


def test_report_walks_every_open_move_with_its_reason(dd_client):
    data = _post(dd_client(_board_payload()))
    assert data["ok"] is True
    bullets = data["bullets"]
    steps = bullets[:5]
    # The plan walks the open moves in the board's rank order, each
    # step naming its league and its move, renumbered 1-5 (open and
    # done share one sequence in board-rank order, so the done action
    # at board rank 6 is report Move 6; board ranks never leak).
    assert steps[0].startswith("Move 1 · FantasyPros")
    assert "Ka'imi Fairbairn on bye" in steps[0]
    assert steps[1].startswith("Move 2 · Blackedraw")
    assert "Start Jaylen Waddle over Tee Higgins" in steps[1]
    assert steps[2].startswith("Move 3 · FantasyPros")
    assert "Add Woody Marks (RB)" in steps[2]
    assert steps[3].startswith("Move 4 · Dynasty Rebuild")
    assert "Move Trey Benson to IR" in steps[3]
    assert steps[4].startswith("Move 5 · Blackedraw")
    assert "Add Woody Marks (RB)" in steps[4]
    # Per-move substance from the payload, not aggregates: the swap's
    # projected gain (compressed, see the restatement test), the
    # waiver's named drop, the injury's return window, and a reason
    # for the detail-less bye move.
    assert "Worth +5.1 projected pts in the swap." in steps[1]
    assert "Drop Dameon Pierce (RB)" in steps[2]
    assert "Approx return Week 11" in steps[3]
    assert "on a bye this week" in steps[0]
    # League context (record line + posture) rides the first step of
    # each league only: Blackedraw's second step does not repeat it.
    assert "6-1 · 1st of 12 · Contender" in steps[0]
    assert "2-5 · 9th of 12 · Rebuilding" in steps[3]
    assert "5-2 · 2nd of 12" not in steps[4]
    # The headline frames scope against the full scan and names the
    # first action instead of a slogan. The count covers true actions
    # only (5, across the 4 leagues that hold them); the deadline is
    # acknowledged as a reminder, not a sixth move.
    assert "5 moves across 4 of your 11 leagues" in data["headline"]
    assert "plus 1 reminder" in data["headline"]
    assert "Ka'imi Fairbairn on bye" in data["headline"]


def test_report_has_no_count_only_tautology_bullets(dd_client):
    data = _post(dd_client(_board_payload()))
    for b in data["bullets"]:
        assert not re.match(
            r"^\d+ contending · \d+ fringe · \d+ rebuilding", b), b
        assert "moves sit in contending leagues" not in b
    assert "Here is the order that protects your contenders" \
        not in data["headline"]
    joined = "\n".join(data["bullets"])
    # The standalone waiver roundup line stays cut (the step itself
    # carries the claim's named drop, pinned in the walk test), and
    # the posture ordering line is gone too: the plan's rank order
    # already sequences the moves, and the timeline roll-up below
    # carries the when-information it used to gesture at.
    assert "Waiver claims to place" not in joined
    assert "Order inside the plan" not in joined
    assert _TIMELINE_LINE in data["bullets"]
    # The done move is reported as done, not walked as a to-do, and
    # the done line carries its title and league now (round 3), not
    # just a bare number.
    assert _DONE_LINE in data["bullets"]
    assert not any(b.startswith("Move 6 ·") for b in data["bullets"])
    assert "Move 7" not in joined
    assert "Move 7" not in data["headline"]


def test_trade_deadline_is_a_reminder_not_a_move(dd_client):
    data = _post(dd_client(_board_payload()))
    bullets = data["bullets"]
    # The deadline takes no Move number anywhere: not as a plan step,
    # not in the count, not in the timeline.
    assert not any(b.startswith("Move 6 ·") for b in bullets)
    assert "Trade deadline" not in data["headline"]
    assert data["headline"].startswith("5 moves across")
    # It renders once, after the plan, as a reminder naming its league
    # and its timeframe from the payload itself.
    reminders = [b for b in bullets if b.startswith("Reminder:")]
    assert len(reminders) == 1
    assert "trade deadline in 1 week" in reminders[0]
    assert "Dynasty Rebuild" in reminders[0]
    assert "Deadline is Week 10" in reminders[0]
    plan_last = max(i for i, b in enumerate(bullets)
                    if b.startswith("Move "))
    assert bullets.index(reminders[0]) > plan_last
    # Renumbering is consistent: the open walk is Moves 1-5 in rank
    # order and the done action is Move 6 in the one line naming it.
    assert data["sections"][2]["items"] == [_DONE_LINE]


def test_done_calendar_nudge_drops_out_entirely(dd_client):
    payload = _board_payload()
    # Mark the deadline acknowledged: an awareness item the user has
    # already handled is neither a reminder nor a "done move".
    cal = payload["board"]["groups"][2]["moves"][1]
    cal["done"] = {"verified": False, "label": "Marked done"}
    data = _post(dd_client(payload))
    joined = "\n".join(data["bullets"])
    assert "Reminder:" not in joined
    assert "Trade deadline" not in joined
    assert "plus 1 reminder" not in data["headline"]
    assert _DONE_LINE in data["bullets"]


def test_reason_never_restates_the_title(dd_client):
    data = _post(dd_client(_board_payload()))
    step = data["bullets"][1]
    # The swap's detail ("Jaylen Waddle projects 14.2 vs Tee
    # Higgins's 9.1") merely repeats the title's claim, so the reason
    # compresses to the one added fact, the Gain row's delta, without
    # re-naming both players or re-quoting both projections.
    assert "Worth +5.1 projected pts in the swap." in step
    assert "projects 14.2" not in step
    assert step.count("Jaylen Waddle") == 1
    assert step.count("Tee Higgins") == 1


def test_detail_less_swap_prefers_matchup_evidence(dd_client):
    payload = _board_payload()
    swap = payload["board"]["groups"][1]["moves"][0]
    swap["detail"] = ""
    swap["evidence"] = [
        {"k": "Start", "v": "Jaylen Waddle · proj 14.2"},
        {"k": "Bench", "v": "Tee Higgins · proj 9.1"},
        {"k": "Matchup", "v": "Waddle draws the #27 pass defense"},
    ]
    data = _post(dd_client(payload))
    step = data["bullets"][1]
    # With no detail and no Gain row, the reason is the evidence item
    # that adds a new fact (the matchup), not a projection row.
    assert "Waddle draws the #27 pass defense" in step
    assert "proj 14.2" not in step


def test_same_player_claimed_in_two_leagues_is_not_a_conflict(dd_client):
    data = _post(dd_client(_board_payload()))
    # Regression (Kaedon, 2026-10-29): the fixture claims Woody Marks
    # in BOTH FantasyPros and Blackedraw. Separate leagues have
    # independent wires, FAAB and rosters, so this is not a conflict:
    # no Watch section, no conflict/prioritize language anywhere, and
    # both claims still render as ordinary plan steps.
    joined = "\n".join(data["bullets"]) + data["headline"]
    assert not any(b.startswith("Watch:") for b in data["bullets"])
    assert "Prioritize" not in joined
    assert "add target in both" not in joined
    assert all(s["label"] != "Watch" for s in data["sections"])
    steps = data["bullets"][:5]
    assert steps[2].startswith("Move 3 · FantasyPros")
    assert "Add Woody Marks (RB)" in steps[2]
    assert steps[4].startswith("Move 5 · Blackedraw")
    assert "Add Woody Marks (RB)" in steps[4]


def test_add_in_one_league_drop_in_another_is_not_flagged(dd_client):
    payload = _board_payload()
    # Blackedraw's claim becomes a different player whose drop is
    # Woody Marks: Marks is now added in FantasyPros and dropped in
    # Blackedraw. Also not a conflict: each league's roster context
    # is its own, so no Watch line fires for this either.
    claim = payload["board"]["groups"][1]["moves"][1]
    claim["title"] = "Add Rico Dowdle (RB)"
    claim["detail"] = ("Top remaining back on the wire · "
                       "Drop Woody Marks (RB)")
    claim["evidence"] = [
        {"k": "Add", "v": "Rico Dowdle · value 980"},
        {"k": "Why", "v": "Top remaining back on the wire"},
        {"k": "Drop", "v": "Woody Marks · RB"},
    ]
    data = _post(dd_client(payload))
    assert not any(b.startswith("Watch:") for b in data["bullets"])
    assert "while dropping" not in "\n".join(data["bullets"])
    assert all(s["label"] != "Watch" for s in data["sections"])


def test_timeline_collapses_to_one_clause_for_one_bucket(dd_client):
    payload = _board_payload()
    # Strip every hard deadline: claims become plain lineup moves and
    # every label loses its lock/waiver wording, so all five open
    # moves land in the no-deadline bucket and the roll-up is a
    # single clause.
    for g in payload["board"]["groups"]:
        for mv in g["moves"]:
            if mv["kind"] == "waiver":
                mv["kind"] = "lineup"
            if not mv.get("done"):
                mv["timing"] = {"label": "Do now", "locked": False,
                                "lock_label": ""}
    data = _post(dd_client(payload))
    timelines = [b for b in data["bullets"] if b.startswith("Timeline:")]
    assert timelines == [
        "Timeline: Move 1, Move 2, Move 3, Move 4 and Move 5 have "
        "no hard deadline."]


def test_sections_group_the_same_lines(dd_client):
    data = _post(dd_client(_board_payload()))
    sections = data["sections"]
    assert [s["label"] for s in sections] == [
        "The plan", "Reminders", "Already done", ""]
    # The flat bullet list is exactly the sections flattened, so any
    # reader of either shape sees the same report.
    assert data["bullets"] == [
        item for s in sections for item in s["items"]]
    plan = sections[0]["items"]
    assert len(plan) == 6  # five steps + the timeline roll-up
    assert plan[-1] == _TIMELINE_LINE
    assert sections[1]["items"][0].startswith("Reminder:")
    assert sections[2]["items"] == [_DONE_LINE]
    # The trailing unlabeled section is the coverage line: the scan
    # scanned all 11 leagues and 4 produced board groups, so the
    # other 7 were checked and need nothing.
    assert sections[3]["items"] == [
        "The other 7 leagues need nothing this week."]
    assert data["bullets"][-1] == sections[3]["items"][0]


def test_coverage_line_counts_only_scanned_leagues(dd_client):
    payload = _board_payload()
    # Only 6 of the 11 leagues were actually scanned (5 skipped
    # cold): the quiet count is scanned minus grouped (6 - 4 = 2).
    # The 5 skipped leagues were never checked and are never counted
    # as needing nothing.
    payload["scan"] = {"total": 11, "scanned": 6, "skipped_cold": 5,
                        "skipped_no_roster": 0, "incomplete": False}
    data = _post(dd_client(payload))
    line = "The other 2 leagues need nothing this week."
    assert data["bullets"][-1] == line
    # The coverage line rides last, in its own unlabeled section.
    assert data["sections"][-1] == {"label": "", "items": [line]}


def test_report_text_has_no_em_dashes(dd_client):
    data = _post(dd_client(_board_payload()))
    # Kaedon's standing UI rule: no em dashes in the report. The
    # step head joins its title with a colon and the reminder joins
    # its detail with a period; nothing the composer emits (as
    # opposed to payload detail text it passes through) may carry
    # the dash in any output shape.
    assert "—" not in data["headline"]
    for b in data["bullets"]:
        assert "—" not in b, b
    for s in data["sections"]:
        for item in s["items"]:
            assert "—" not in item, item
    steps = [b for b in data["bullets"] if b.startswith("Move ")]
    assert steps[0].startswith(
        "Move 1 · FantasyPros (6-1 · 1st of 12 · Contender): "
        "Ka'imi Fairbairn on bye.")
    reminder = [b for b in data["bullets"]
                if b.startswith("Reminder:")][0]
    assert reminder == (
        "Reminder: trade deadline in 1 week in Dynasty Rebuild. "
        "Deadline is Week 10. Line up any deals before the wire "
        "closes.")



def test_partial_scan_is_refused_not_narrated(dd_client):
    data = _post(dd_client(_board_payload(scan_incomplete=True)))
    assert data["ok"] is False
    assert data["error"] == "scan_incomplete"
    assert "headline" not in data
    assert "bullets" not in data


def test_empty_board_still_errors(dd_client):
    payload = _board_payload()
    payload["board"] = {"groups": [], "week": 9}
    data = _post(dd_client(payload))
    assert data["ok"] is False
    assert data["error"] == "no_board"
