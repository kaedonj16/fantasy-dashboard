"""Next Steps waiver actions run the waiver-suggestions v2 gate.

Regression for the live bug (2026-10-15): the BR Assistant's this_week
answer and the dashboard Next Steps queue recommended claims by value
movement ("up N spots this week") because _next_steps_waiver_actions
parsed the Waiver Wire Targets card rows, an older risers-based path
that never called rank_team_waiver_suggestions. Kaedon (Blackedraw,
1-QB) was told to claim a fourth QB. These tests pin the fix: claims
come from the shared selector with per-week projection maps, calendar
need, and drop bars, so a covered QB room surfaces no QB claim, while
a real QB hole (starter on bye, backup out) still does.
"""
import pytest

pd = pytest.importorskip("pandas")

import app as app_module  # noqa: E402


# --------------------------------------------------------------------------
# Fixture league: dynasty, 1-QB. The viewer rosters three healthy QBs
# (a strong starter plus two backups). On the wire: Baker Mayfield,
# a starting QB whose value beats the worst rostered QB and who is
# "up 34 spots this week", plus an RB who genuinely improves the lineup.
# --------------------------------------------------------------------------

TEAMS = {
    "q1": "BUF", "q2": "NYJ", "q3": "CLE",
    "rb1": "DET", "rb2": "SF", "rb3": "SEA", "rb4": "DAL",
    "wr1": "CIN", "wr2": "MIA", "wr3": "LAR", "wr4": "DEN", "wr5": "PHI",
    "te1": "KC", "te2": "BAL",
    "mayf": "TB", "wrb": "HOU", "wwr": "ATL",
    "o1": "GB", "o2": "MIN", "o3": "CHI",
}
NAMES = {
    "q1": "Josh Allen", "q2": "Justin Fields", "q3": "Sam Howell",
    "rb1": "Jahmyr Gibbs", "rb2": "Breece Hall", "rb3": "Kenneth Walker",
    "rb4": "Tony Pollard",
    "wr1": "Ja'Marr Chase", "wr2": "Tyreek Hill", "wr3": "Puka Nacua",
    "wr4": "Courtland Sutton", "wr5": "Diontae Johnson",
    "te1": "Travis Kelce", "te2": "David Njoku",
    "mayf": "Baker Mayfield", "wrb": "Rising Back", "wwr": "Wire Wideout",
    "o1": "Opp One", "o2": "Opp Two", "o3": "Opp Three",
}
POS = {
    "q1": "QB", "q2": "QB", "q3": "QB",
    "rb1": "RB", "rb2": "RB", "rb3": "RB", "rb4": "RB",
    "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR", "wr5": "WR",
    "te1": "TE", "te2": "TE",
    "mayf": "QB", "wrb": "RB", "wwr": "WR",
    "o1": "RB", "o2": "WR", "o3": "TE",
}
VALUES = {
    "q1": 5200, "q2": 1800, "q3": 950,
    "rb1": 4600, "rb2": 2400, "rb3": 1200, "rb4": 700,
    "wr1": 4200, "wr2": 3600, "wr3": 2800, "wr4": 1500, "wr5": 650,
    "te1": 2600, "te2": 500,
    # Mayfield (2600) beats the worst rostered QB (950) by far more than
    # the x1.05 drop-bar margin: the old value-only gate waved him in.
    "mayf": 2600, "wrb": 2100, "wwr": 900,
    "o1": 3000, "o2": 3000, "o3": 3000,
}
LABELS = {"mayf": "QB8", "wrb": "RB18", "wwr": "WR48",
          "q1": "QB1", "q2": "QB14", "q3": "QB27"}
PROJ = {
    "q1": 24.0, "q2": 19.0, "q3": 15.0,
    "rb1": 16.0, "rb2": 11.0, "rb3": 8.0, "rb4": 5.0,
    "wr1": 15.0, "wr2": 14.0, "wr3": 12.5, "wr4": 9.0, "wr5": 5.0,
    "te1": 10.0, "te2": 4.0,
    # Mayfield projects well, just never above the starter: he cannot
    # crack a 1-QB lineup, so he adds zero starter points every week.
    "mayf": 20.5, "wrb": 13.5, "wwr": 8.0,
}
ROSTER = ["q1", "q2", "q3", "rb1", "rb2", "rb3", "rb4",
          "wr1", "wr2", "wr3", "wr4", "wr5", "te1", "te2"]
ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                    "BN", "BN", "BN", "BN", "BN", "BN"]


def _schedule(bye_teams=()):
    teams = sorted({t for t in TEAMS.values() if t not in bye_teams})
    return [{"away": teams[i], "home": teams[i + 1]}
            for i in range(0, len(teams) - 1, 2)]


def _fixture(monkeypatch, *, starter_bye=False, backup_out=False,
             roster=None):
    pidx = {
        pid: {"position": POS[pid], "pos": POS[pid], "team": TEAMS[pid],
              "full_name": NAMES[pid],
              "injury_status": "Out" if (backup_out and pid == "q2") else ""}
        for pid in TEAMS
    }
    rows = [{
        "id": pid, "name": NAMES[pid], "position": POS[pid],
        "team": TEAMS[pid], "value": VALUES[pid], "age": 31 if pid == "mayf" else 25,
        "pos_rank_label": LABELS.get(pid, ""),
        "rank_change_7d": 34 if pid == "mayf" else 112 if pid == "wrb" else 0,
    } for pid in TEAMS]
    ctx = {
        "platform": "sleeper", "league_id": "L1", "season": 2026,
        "current_season": 2026, "current_week": 7,
        "league": {"type": 2}, "settings": {"type": 2},
        "scoring_settings": {}, "raw_scoring_settings": {},
        "players_index": pidx,
        "roster_positions": list(ROSTER_POSITIONS),
        "rosters": [
            {"roster_id": 1, "players": list(roster or ROSTER)},
            {"roster_id": 2, "players": ["o1", "o2", "o3"]},
        ],
        "viewer": {"viewer_roster_id": 1},
    }

    def fake_proj(season, week):
        out = {}
        for pid, pts in PROJ.items():
            if starter_bye and pid == "q1" and week == 7:
                continue  # starter's team is on bye in Week 7
            if backup_out and pid == "q2":
                continue  # hurt: no projection at all
            out[pid] = pts
        return out

    def fake_sched(season, week):
        if starter_bye and week == 7:
            return _schedule(bye_teams={"BUF"})
        return _schedule()

    monkeypatch.setattr("utils.utils.load_week_projection", fake_proj)
    monkeypatch.setattr("utils.utils.load_week_schedule", fake_sched)
    return ctx, rows


# --------------------------------------------------------------------------
# The Mayfield regression
# --------------------------------------------------------------------------

def test_covered_qb_room_gets_no_qb_claim(monkeypatch):
    ctx, rows = _fixture(monkeypatch)
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    blob = " ".join(f"{a['action']} {a['why']}" for a in actions)
    assert "Mayfield" not in blob
    assert not any(a["action"].startswith("Claim Baker") for a in actions)
    # Value movement is never the reason on this surface.
    assert "spots" not in blob


def test_real_lineup_gain_claim_still_surfaces(monkeypatch):
    ctx, rows = _fixture(monkeypatch)
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    rb_claims = [a for a in actions if a["player_id"] == "wrb"]
    assert len(rb_claims) == 1
    claim = rb_claims[0]
    assert claim["action"] == "Claim Rising Back"
    assert claim["outcome"] in ("add", "add_drop")
    assert "lineup pts" in claim["why"]
    assert claim["impact"] and "projected points" in claim["impact"]


def test_qb_claim_surfaces_when_starter_on_bye_and_backup_out(monkeypatch):
    # Positive control: only two QBs rostered, the starter is on bye in
    # Week 7 and the backup is Out, so the wire QB starts this week and
    # fills a real calendar hole.
    roster = [p for p in ROSTER if p != "q3"]
    ctx, rows = _fixture(monkeypatch, starter_bye=True, backup_out=True,
                         roster=roster)
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    qb_claims = [a for a in actions if a["player_id"] == "mayf"]
    assert len(qb_claims) == 1
    claim = qb_claims[0]
    assert claim["action"] == "Claim Baker Mayfield"
    assert claim["outcome"] in ("add", "add_drop")
    assert "lineup pts" in claim["why"]
    assert "Week 7" in claim["why"]  # the calendar hole is the reason


def test_no_viewer_roster_no_claims(monkeypatch):
    ctx, rows = _fixture(monkeypatch)
    assert app_module._next_steps_waiver_actions(ctx, None, rows) == []
    assert app_module._next_steps_waiver_actions(ctx, 99, rows) == []


# --------------------------------------------------------------------------
# Wiring pins: the queue no longer parses the risers card.
# --------------------------------------------------------------------------

def test_actions_source_is_the_v2_selector():
    from pathlib import Path
    src = Path(app_module.__file__).read_text(encoding="utf-8")
    start = src.index("def _next_steps_waiver_claims")
    end = src.index("def _next_steps_trade_actions", start)
    body = src[start:end]
    assert "rank_team_waiver_suggestions" in body
    assert "weekly_availability_from_roster" in body
    actions_start = body.index("def _next_steps_waiver_actions")
    actions_body = body[actions_start:]
    assert "_next_steps_waiver_claims" in actions_body
    assert "_build_waiver_targets_rows" not in actions_body
    assert "spots this week" not in actions_body
