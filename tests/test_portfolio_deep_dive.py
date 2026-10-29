"""Tests for /api/portfolio-deep-dive, composed as The Brief.

The report used to walk the board's moves as a labeled step list;
Kaedon rejected that twice over (it narrated only the moves, in the
board's own shape, and said nothing about the leagues themselves).
The Brief is a short prose report with a league layer: a standfirst
splitting every scanned league by posture (the payload's leagues
array, one entry per scanned league, not just leagues holding
moves), a portfolio paragraph (flagship and deep end by record,
the featured trade-deadline league, the quiet count), a moves
paragraph (biggest projected gain, then the waiver claims as one
clock sentence), a clock paragraph (fills, anytime moves, locks,
calendar facts), and end matter (done moves by short title, then
coverage). Deterministic templates, every sentence conditional on
its facts existing.

Pinned contracts carried over from the step-list era: calendar
nudges are dates, not moves (never counted, and a done nudge drops
out); there is NO cross-league conflict callout (Kaedon corrected
that premise on 2026-10-29: separate leagues have independent
wires, FAAB and rosters, so the same player claimed in two leagues
is stated plainly, never as something to prioritize); no em dashes
anywhere (standing UI rule); a partial scan is refused, not
narrated. Paragraphs return as span arrays; strong marks player
names and digit-bearing tokens only."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _move(rank, kind, title, detail="", evidence=None, timing=None,
          done=None, **extra):
    mv = {
        "rank": rank, "kind": kind, "title": title, "detail": detail,
        "evidence": evidence or [],
        "timing": timing or {"label": "This week", "locked": False,
                             "lock_label": ""},
        "done": done, "move_key": f"pf:{kind}:{rank}",
        "platform": "sleeper", "league_id": "x", "season": 2026,
    }
    mv.update(extra)
    return mv


def _league(key, name, record_line, posture, wins, losses, rank,
            team_count=12, pf=None, odds=None, **extra):
    ent = {
        "key": key, "platform": "sleeper",
        "league_id": key.split("|")[1], "league_name": name,
        "record_line": record_line, "posture": posture,
        "wins": wins, "losses": losses, "ties": 0, "rank": rank,
        "team_count": team_count,
    }
    if pf is not None:
        ent["points_for"] = pf
    if odds is not None:
        ent["odds_pct"] = odds
    ent.update(extra)
    return ent


def _board_payload(scan_incomplete=False):
    """The mock-shaped board behind the approved Brief: five open
    actions plus a done action over four leagues, a trade-deadline
    calendar nudge in Dynasty Rebuild, and seven quiet leagues, from
    an 11-league scan. Blackedraw (6-1, 1st) is the flagship, League
    of Greatness (1-6, last, rebuilding) the deep end. Woody Marks
    is the add target in two leagues: NOT a conflict, since separate
    leagues run independent wires; the brief states both claims
    plainly."""
    groups = [
        {
            "key": "sleeper|111", "platform": "sleeper",
            "league_id": "111", "league_name": "FantasyPros",
            "meta": {"record_line": "5-2 · 2nd of 12",
                     "posture": "Contender", "odds_pct": 71.0},
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
            "meta": {"record_line": "6-1 · 1st of 12",
                     "posture": "Contender", "odds_pct": 78.0},
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
                              "before the wire closes."),
                      due_week=10, due_in_weeks=1),
            ],
        },
        {
            "key": "sleeper|444", "platform": "sleeper",
            "league_id": "444", "league_name": "League of Greatness",
            "meta": {"record_line": "1-6 · 12th of 12",
                     "posture": "Rebuilding", "odds_pct": 2.0},
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
    leagues = [
        _league("sleeper|111", "FantasyPros", "5-2 · 2nd of 12",
                "Contender", 5, 2, 2, pf=790.0, odds=71.0),
        _league("sleeper|222", "Blackedraw", "6-1 · 1st of 12",
                "Contender", 6, 1, 1, pf=812.5, odds=78.0),
        _league("sleeper|333", "Dynasty Rebuild", "2-5 · 9th of 12",
                "Rebuilding", 2, 5, 9, pf=640.0, odds=4.0,
                trade_deadline_week=10, playoff_week_start=15),
        _league("sleeper|444", "League of Greatness", "1-6 · 12th of 12",
                "Rebuilding", 1, 6, 12, pf=590.0, odds=2.0),
        _league("sleeper|555", "Josh Allen's Twitter", "5-2 · 3rd of 12",
                "Contender", 5, 2, 3, pf=780.0),
        _league("sleeper|666", "KC fantasy-yearly", "5-2 · 4th of 12",
                "Contender", 5, 2, 4, pf=760.0),
        _league("sleeper|777", "BLITZ THE LEAGUE", "4-3 · 5th of 12",
                "Fringe", 4, 3, 5, pf=720.0),
        _league("sleeper|888", "All American All Star League",
                "4-3 · 6th of 12", "Fringe", 4, 3, 6, pf=710.5),
        _league("sleeper|999", "Revenge of the Picks", "3-4 · 8th of 12",
                "Fringe", 3, 4, 8, pf=680.0),
        _league("sleeper|101", "The Fellow of the Fender",
                "3-4 · 9th of 12", "Fringe", 3, 4, 9, pf=660.0),
        _league("sleeper|102", "Yahoo H2H-Pts 1788569",
                "2-5 · 10th of 12", "Full Rebuild", 2, 5, 10, pf=655.0),
    ]
    return {
        "ok": True,
        "scan": {"total": 11, "scanned": 11, "skipped_cold": 0,
                 "skipped_no_roster": 0, "incomplete": scan_incomplete},
        "board": {"groups": groups, "week": 9},
        "week": 9, "season": 2026,
        "leagues": leagues,
    }


_P1 = ("Blackedraw is the flagship: 6-1 and 1st, with two moves open. "
       "League of Greatness is the deep end at 1-6 and last, already "
       "playing for next year, and its one move is done. "
       "Dynasty Rebuild is 2-5 and rebuilding, and its trade deadline "
       "lands in Week 10, a week out, so it is the one league where "
       "the calendar matters more than the wire. "
       "Seven leagues need nothing this week.")
_P2 = ("The biggest gain on the board is the Waddle swap in "
       "Blackedraw: start Jaylen Waddle over Tee Higgins for +5.1 "
       "projected pts, and it locks at 1:00 PM. "
       "Two claims run against the waiver clock: add Woody Marks in "
       "FantasyPros (dropping Dameon Pierce) and in Blackedraw, both "
       "before waivers run.")
_P3 = ("The Fairbairn bye in FantasyPros needs a fill before Sunday, "
       "the Benson IR move in Dynasty Rebuild can happen anytime, "
       "and the Dynasty Rebuild trade deadline lands in 1 week, "
       "Week 10, so deals there get lined up now.")
_END = ["One move is already done: Bowers over Kincaid in "
        "League of Greatness.",
        "The other 7 leagues need nothing this week."]


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


def _text(spans):
    return "".join(s.get("text", "") for s in spans or [])


def _strongs(spans):
    return [s["text"] for s in spans or [] if s.get("strong")]


def _brief(data):
    b = data["brief"]
    return {
        "kicker": b["kicker"],
        "standfirst": _text(b["standfirst"]),
        "paragraphs": [_text(p) for p in b["paragraphs"]],
        "end_matter": [_text(line) for line in b["end_matter"]],
    }


def _all_text(data):
    b = _brief(data)
    return "\n".join([b["kicker"], b["standfirst"], *b["paragraphs"],
                      *b["end_matter"]])


def test_brief_reproduces_the_approved_mock(dd_client):
    data = _post(dd_client(_board_payload()))
    assert data["ok"] is True
    # The old step-list shape is gone; the Brief is the only shape.
    assert "headline" not in data
    assert "bullets" not in data
    assert "sections" not in data
    b = _brief(data)
    assert b["kicker"] == "Week 9 · Portfolio Brief"
    assert b["standfirst"] == (
        "Eleven leagues, split 4 contending, 4 on the bubble and "
        "3 rebuilding.")
    assert b["paragraphs"] == [_P1, _P2, _P3]
    assert b["end_matter"] == _END


def test_strong_marks_names_and_numbers_only(dd_client):
    data = _post(dd_client(_board_payload()))
    raw = data["brief"]
    # Standfirst: only the digit counts are strong; the spelled-out
    # total stays plain.
    assert _strongs(raw["standfirst"]) == ["4", "4", "3"]
    p1 = raw["paragraphs"][0]
    assert "6-1" in _strongs(p1)
    assert "1st" in _strongs(p1)
    # League names and spelled-out counts are never strong.
    assert "Blackedraw" not in _strongs(p1)
    assert "two" not in _strongs(p1)
    p2 = raw["paragraphs"][1]
    for name in ("Jaylen Waddle", "Tee Higgins", "Woody Marks",
                 "Dameon Pierce", "Waddle"):
        assert name in _strongs(p2), name
    assert "+5.1" in _strongs(p2)
    assert "FantasyPros" not in _strongs(p2)
    assert "Blackedraw" not in _strongs(p2)
    # End matter: the done swap's surnames are strong.
    assert _strongs(raw["end_matter"][0]) == ["Bowers", "Kincaid"]


def test_same_player_claims_carry_no_conflict_language(dd_client):
    data = _post(dd_client(_board_payload()))
    text = _all_text(data)
    # Regression (Kaedon, 2026-10-29): Woody Marks is claimed in
    # both FantasyPros and Blackedraw. Separate leagues have
    # independent wires, FAAB and rosters: the brief states both
    # claims plainly and never frames them as a conflict.
    assert "add Woody Marks in FantasyPros" in text
    assert "and in Blackedraw" in text
    assert "Prioritize" not in text
    assert "prioritize" not in text
    assert "conflict" not in text.lower()


def test_brief_text_has_no_em_dashes(dd_client):
    data = _post(dd_client(_board_payload()))
    assert "—" not in _all_text(data)


def test_no_swap_gain_leads_with_the_first_move(dd_client):
    payload = _board_payload()
    swap = payload["board"]["groups"][1]["moves"][0]
    swap["evidence"] = [r for r in swap["evidence"]
                        if r["k"] != "Gain"]
    data = _post(dd_client(payload))
    b = _brief(data)
    # No gain anywhere: the moves paragraph leads with the board's
    # first move and its reason, and that move is not repeated as a
    # clock clause. The swap still lands in the clock paragraph as
    # a lock clause.
    assert b["paragraphs"][1].startswith(
        "The first move on the board is in FantasyPros: "
        "Ka'imi Fairbairn on bye. The starter is on a bye this week, "
        "so the slot needs a fill.")
    assert "Fairbairn" not in b["paragraphs"][2]
    assert b["paragraphs"][2] == (
        "The Waddle swap in Blackedraw locks at 1:00 PM, "
        "the Benson IR move in Dynasty Rebuild can happen anytime, "
        "and the Dynasty Rebuild trade deadline lands in 1 week, "
        "Week 10, so deals there get lined up now.")


def test_one_claim_grammar(dd_client):
    payload = _board_payload()
    # Drop Blackedraw's claim: one claim left, and the flagship's
    # open count drops to one.
    payload["board"]["groups"][1]["moves"] = \
        payload["board"]["groups"][1]["moves"][:1]
    data = _post(dd_client(payload))
    b = _brief(data)
    assert ("One claim runs against the waiver clock: add Woody "
            "Marks in FantasyPros (dropping Dameon Pierce), before "
            "waivers run.") in b["paragraphs"][1]
    assert "with one move open" in b["paragraphs"][0]


def test_three_claims_grammar(dd_client):
    payload = _board_payload()
    payload["board"]["groups"][2]["moves"].insert(1, _move(
        8, "waiver", "Add Rico Dowdle (RB)",
        detail=("Top remaining back on the wire · "
                "Drop Roschon Johnson (RB)"),
        evidence=[
            {"k": "Add", "v": "Rico Dowdle · value 980"},
            {"k": "Why", "v": "Top remaining back on the wire"},
            {"k": "Drop", "v": "Roschon Johnson · RB"},
        ],
        timing={"label": "Before waivers", "locked": False,
                "lock_label": ""}))
    data = _post(dd_client(payload))
    b = _brief(data)
    assert ("Three claims run against the waiver clock: add Woody "
            "Marks in FantasyPros (dropping Dameon Pierce), in "
            "Blackedraw and add Rico Dowdle in Dynasty Rebuild "
            "(dropping Roschon Johnson), all before waivers run."
            ) in b["paragraphs"][1]


def test_no_deadline_no_deadline_sentences(dd_client):
    payload = _board_payload()
    payload["board"]["groups"][2]["moves"] = \
        payload["board"]["groups"][2]["moves"][:1]
    data = _post(dd_client(payload))
    text = _all_text(data)
    assert "trade deadline" not in text
    b = _brief(data)
    assert b["paragraphs"][2] == (
        "The Fairbairn bye in FantasyPros needs a fill before "
        "Sunday, and the Benson IR move in Dynasty Rebuild can "
        "happen anytime.")


def test_done_calendar_nudge_drops_out_entirely(dd_client):
    payload = _board_payload()
    cal = payload["board"]["groups"][2]["moves"][1]
    cal["done"] = {"verified": False, "label": "Marked done"}
    data = _post(dd_client(payload))
    # An acknowledged deadline is neither narrated as a league fact
    # nor listed as a done move.
    assert "trade deadline" not in _all_text(data)


def test_no_done_moves_end_matter_is_coverage_only(dd_client):
    payload = _board_payload()
    payload["board"]["groups"][3]["moves"][0]["done"] = None
    data = _post(dd_client(payload))
    b = _brief(data)
    assert b["end_matter"] == [
        "The other 7 leagues need nothing this week."]
    # The now-open Bowers swap joins the clock paragraph instead.
    assert ("the Bowers swap in League of Greatness comes due "
            "before Sunday") in b["paragraphs"][2]


def test_all_moves_done(dd_client):
    payload = _board_payload()
    for g in payload["board"]["groups"]:
        for mv in g["moves"]:
            if mv["kind"] != "calendar":
                mv["done"] = {"verified": False, "label": "Marked done"}
    data = _post(dd_client(payload))
    b = _brief(data)
    # No moves paragraph at all; the clock paragraph is the lone
    # deadline sentence, and the flagship counts done moves.
    assert len(b["paragraphs"]) == 2
    assert "and its two moves are done" in b["paragraphs"][0]
    assert b["paragraphs"][1] == (
        "The Dynasty Rebuild trade deadline lands in 1 week, "
        "Week 10, so deals there get lined up now.")
    assert b["end_matter"][0] == (
        "Six moves are already done: Ka'imi Fairbairn on bye in "
        "FantasyPros, Waddle over Higgins in Blackedraw, "
        "Add Woody Marks (RB) in FantasyPros, Move Trey Benson to "
        "IR in Dynasty Rebuild, Add Woody Marks (RB) in Blackedraw "
        "and Bowers over Kincaid in League of Greatness.")


def test_zero_bubble_standfirst_omits_the_bucket(dd_client):
    payload = _board_payload()
    for ent in payload["leagues"]:
        if ent["posture"] == "Fringe":
            ent["posture"] = ("Contender" if ent["league_id"] in
                              ("777", "888") else "Rebuilding")
    data = _post(dd_client(payload))
    b = _brief(data)
    assert b["standfirst"] == (
        "Eleven leagues, split 6 contending and 5 rebuilding.")


def test_missing_league_entry_degrades_the_split(dd_client):
    payload = _board_payload()
    # One scanned league's meta never made the payload: the split
    # describes only the leagues with known state, and nothing is
    # invented for the missing one.
    payload["leagues"] = [e for e in payload["leagues"]
                          if e["league_id"] != "666"]
    data = _post(dd_client(payload))
    b = _brief(data)
    assert b["standfirst"] == (
        "Ten leagues, split 3 contending, 4 on the bubble and "
        "3 rebuilding.")
    assert b["paragraphs"][0].startswith(
        "Blackedraw is the flagship: 6-1 and 1st")


def test_no_leagues_array_falls_back_to_group_meta(dd_client):
    payload = _board_payload()
    # A payload assembled before the leagues array existed: the
    # league layer degrades to the leagues holding board moves,
    # whose group meta carries the same record lines and postures.
    del payload["leagues"]
    data = _post(dd_client(payload))
    b = _brief(data)
    assert b["standfirst"] == (
        "Four leagues, split 2 contending and 2 rebuilding.")
    assert b["paragraphs"][0].startswith(
        "Blackedraw is the flagship: 6-1 and 1st, with two moves "
        "open. League of Greatness is the deep end at 1-6 and last, "
        "already playing for next year, and its one move is done.")


def test_partial_scan_is_refused_not_narrated(dd_client):
    data = _post(dd_client(_board_payload(scan_incomplete=True)))
    assert data["ok"] is False
    assert data["error"] == "scan_incomplete"
    assert "brief" not in data


def test_empty_board_still_errors(dd_client):
    payload = _board_payload()
    payload["board"] = {"groups": [], "week": 9}
    data = _post(dd_client(payload))
    assert data["ok"] is False
    assert data["error"] == "no_board"
