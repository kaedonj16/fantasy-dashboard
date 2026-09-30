"""Weekly digest improvements: Thursday alerts, league activity, playoff stakes, preheader."""
from __future__ import annotations

import importlib.util

import utils.weekly_email as we
from utils.digest_sections import (
    email_shell,
    league_activity_html,
    league_summary_html,
    thursday_alert_html,
)

# 2026-10-01 is a Thursday. 8:15 PM ET == 2026-10-02 00:15 UTC, which a naive
# UTC weekday check would misclassify as Friday.
THURSDAY_GAMES = [
    {"gameTime": "2026-10-01T20:15:00-04:00", "home": "KC", "away": "BUF"},
    {"gameTime": "2026-10-04T13:00:00-04:00", "home": "DET", "away": "MIN"},
]
PIDX = {
    "1": {"full_name": "Patrick Mahomes", "team": "KC"},
    "2": {"full_name": "Jahmyr Gibbs", "team": "DET"},
}


def test_thursday_night_starters_flags_thursday_players():
    out = we.thursday_night_starters(
        starters=["1", "2"], pidx=PIDX, season=2026, week=5, games=THURSDAY_GAMES,
    )
    assert out == [{
        "player_id": "1", "name": "Patrick Mahomes",
        "team": "KC", "kickoff": "8:15 PM ET",
    }]


def test_thursday_night_starters_detects_defense_by_abbreviation():
    out = we.thursday_night_starters(
        starters=["BUF"], pidx={}, season=2026, week=5, games=THURSDAY_GAMES,
    )
    assert len(out) == 1
    assert out[0]["team"] == "BUF"
    assert out[0]["name"] == "BUF"


def test_thursday_night_starters_utc_midnight_kickoff_is_thursday():
    games = [{"gameTime": "2026-10-02T00:15:00Z", "home": "KC", "away": "BUF"}]
    out = we.thursday_night_starters(
        starters=["1"], pidx=PIDX, season=2026, week=5, games=games,
    )
    assert len(out) == 1
    assert out[0]["kickoff"] == "8:15 PM ET"


def test_thursday_night_starters_empty_without_schedule():
    assert we.thursday_night_starters(
        starters=["1"], pidx=PIDX, season=2026, week=5, games=[],
    ) == []
    assert we.thursday_night_starters(
        starters=[], pidx=PIDX, season=2026, week=5, games=THURSDAY_GAMES,
    ) == []


def test_thursday_alert_html_renders_card():
    html = thursday_alert_html([{
        "player_id": "1", "name": "Patrick Mahomes",
        "team": "KC", "kickoff": "8:15 PM ET",
    }])
    assert "Thursday night" in html
    assert "Patrick Mahomes (KC)" in html
    assert "8:15 PM ET" in html
    assert thursday_alert_html([]) == ""


def test_thursday_alert_html_compact_mode():
    html = thursday_alert_html([{"name": "Patrick Mahomes"}], compact=True)
    assert "Thursday night:" in html
    assert "Patrick Mahomes" in html


_TXNS = [
    {"transaction_id": "t1", "type": "waiver", "status": "complete", "created": 300,
     "roster_ids": [2], "adds": {"10": 2}, "drops": {"11": 2}},
    {"transaction_id": "t2", "type": "trade", "status": "complete", "created": 200,
     "roster_ids": [2, 3], "adds": {"20": 2, "21": 3}, "draft_picks": [{"a": 1}]},
    {"transaction_id": "t3", "type": "waiver", "status": "failed", "created": 400,
     "roster_ids": [2], "adds": {"30": 2}},
    {"transaction_id": "t4", "type": "waiver", "status": "complete", "created": 100,
     "roster_ids": [1], "adds": {"40": 1}},
]
_ROSTERS = [
    {"roster_id": 1, "owner_id": "u1"},
    {"roster_id": 2, "owner_id": "u2"},
    {"roster_id": 3, "owner_id": "u3"},
]
_USERS = [
    {"user_id": "u1", "display_name": "Kaedon"},
    {"user_id": "u2", "display_name": "Jayden"},
    {"user_id": "u3", "username": "caleb"},
]
_TXN_PIDX = {
    "10": {"full_name": "Rome Odunze"},
    "11": {"full_name": "Old Guy"},
    "20": {"full_name": "Derrick Henry"},
    "21": {"full_name": "Davante Adams"},
}


def _activity(**kw):
    base = {"platform": "sleeper", "league_id": "x", "week": 5,
            "rosters": _ROSTERS, "users": _USERS, "pidx": _TXN_PIDX,
            "my_roster_id": "1", "transactions": _TXNS}
    base.update(kw)
    return we.recent_league_activity(**base)


def test_recent_league_activity_waiver_and_trade():
    bullets = _activity()
    assert bullets == [
        "Jayden picked up Rome Odunze (dropped Old Guy)",
        "Trade: Jayden gets Derrick Henry; caleb gets Davante Adams; plus 1 draft pick",
    ]


def test_recent_league_activity_skips_own_moves_and_failed():
    bullets = _activity()
    assert not any("Kaedon" in b for b in bullets)
    assert not any("30" in b for b in bullets)


def test_recent_league_activity_non_sleeper_is_empty():
    assert _activity(platform="espn") == []


def test_recent_league_activity_respects_limit():
    assert len(_activity(limit=1)) == 1


def test_league_activity_html():
    html = league_activity_html(["Jayden picked up Rome Odunze"], href="https://x/waivers")
    assert "Around your league" in html
    assert "Jayden picked up Rome Odunze" in html
    assert "https://x/waivers" in html
    assert league_activity_html([]) == ""


def _stakes(rank, wins, wins_list, week=5):
    rosters = [{"settings": {"wins": w}} for w in wins_list]
    league = {"settings": {"playoff_teams": 4, "playoff_week_start": 15}}
    return we.playoff_stakes_line(
        rank=rank, wins=wins, rosters=rosters, league=league, current_week=week,
    )


def test_playoff_stakes_behind_cut_line():
    assert _stakes(5, 2, [5, 4, 4, 3, 2, 1]) == \
        "You're 1 game back of the final playoff spot with 9 weeks to go."


def test_playoff_stakes_clear_of_cut_line():
    assert _stakes(1, 5, [5, 4, 4, 3, 2, 1]) == \
        "You hold a playoff spot, 3 games clear of the cut line with 9 weeks to go."


def test_playoff_stakes_tiebreak_and_tie():
    assert _stakes(3, 2, [5, 4, 2, 2, 2, 1]) == \
        "You hold a playoff spot on the tiebreak with 9 weeks to go."
    assert _stakes(5, 3, [5, 4, 4, 3, 3, 1]) == \
        "You're tied for the final playoff spot with 9 weeks to go."


def test_playoff_stakes_empty_when_unknown():
    assert _stakes(3, 4, [5, 4, 4, 3, 2, 1], week=15) == ""  # playoffs started
    assert we.playoff_stakes_line(
        rank=3, wins=4, rosters=[{"settings": {"wins": 4}}] * 6,
        league={}, current_week=5) == ""  # no settings
    assert we.playoff_stakes_line(
        rank=None, wins=4, rosters=[{"settings": {"wins": 4}}] * 6,
        league={"settings": {"playoff_teams": 4, "playoff_week_start": 15}},
        current_week=5) == ""  # no rank


def test_playoff_stakes_in_summary():
    html = league_summary_html(
        league_name="Blackedraw", rank=5, wins=2, losses=3,
        format_label="Redraft",
        stakes_line="You're 1 game back of the final playoff spot with 9 weeks to go.",
    )
    assert "1 game back of the final playoff spot" in html
    html2 = league_summary_html(league_name="Blackedraw", rank=5, wins=2, losses=3)
    assert "final playoff spot" not in html2


def test_choose_subject_prefers_thursday():
    subject = we.choose_subject(
        "Blackedraw", {"type": "redraft"}, rank=2, wins=3, losses=1,
        lineup_note={"title": "Empty slot", "body": "Your RB2 slot is empty."},
        thursday=[{"name": "Patrick Mahomes"}],
    )
    assert subject == "Blackedraw: Set your lineup before Thursday night"


def test_choose_preheader_most_actionable_first():
    assert we.choose_preheader({
        "lineup_note": {"title": "Empty slot", "body": "Your RB2 slot is empty."},
        "thursday": [{"name": "Patrick Mahomes"}],
        "league_name": "Blackedraw",
    }) == "Empty slot: Your RB2 slot is empty."
    assert we.choose_preheader({"thursday": [{"name": "Patrick Mahomes"}]}) == \
        "Set your lineup before Thursday kickoff. Patrick Mahomes"
    assert we.choose_preheader({"matchup": {"opponent_name": "Jayden", "win_prob": 0.7}}) == \
        "You're favored vs Jayden this week."
    assert we.choose_preheader({"matchup": {"opponent_name": "Jayden", "win_prob": 0.3}}) == \
        "Tough one vs Jayden this week."
    assert we.choose_preheader({"waivers": [{"name": "Rome Odunze"}]}) == \
        "Top waiver target: Rome Odunze."
    assert we.choose_preheader({"league_name": "Blackedraw"}) == \
        "Your weekly fantasy digest for Blackedraw."
    assert we.choose_preheader({}) == "Your weekly fantasy digest."


def test_choose_preheader_clips_long_text():
    ph = we.choose_preheader({"lineup_note": {"title": "X", "body": "y" * 200}})
    assert len(ph) <= 113
    assert ph.endswith("...")


def test_email_shell_preheader():
    html = email_shell("<p>body</p>", subtitle="Test", preheader="Set your lineup tonight")
    assert "Set your lineup tonight" in html
    assert "display:none" in html
    body_pos = html.find("<p>body</p>")
    pre_pos = html.find("Set your lineup tonight")
    assert 0 < pre_pos < body_pos


def test_email_shell_no_preheader_by_default():
    html = email_shell("<p>body</p>", subtitle="Test")
    assert "display:none" not in html


def _load_trigger_script():
    path = "scripts/trigger_notifications.py"
    spec = importlib.util.spec_from_file_location("trigger_notifications", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trigger_retries_weekly_on_524(monkeypatch):
    mod = _load_trigger_script()
    import urllib.error

    calls = []

    class FakeResp:
        status = 202

        def read(self):
            return b'{"sent": 3, "breakdown": {"weekly": 3}}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                req.full_url, 524, "timeout", {}, None)
        return FakeResp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setenv("APP_URL", "https://www.example.com")
    monkeypatch.setenv("CRON_SECRET", "secret")
    assert mod.trigger("weekly") == 0
    assert len(calls) == 2


def test_trigger_does_not_retry_non_524_or_non_weekly(monkeypatch):
    mod = _load_trigger_script()
    import urllib.error

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setenv("APP_URL", "https://www.example.com")
    monkeypatch.setenv("CRON_SECRET", "secret")
    assert mod.trigger("weekly") == 1
    assert len(calls) == 1
    assert mod.trigger("hourly") == 1
    assert len(calls) == 2


def test_connected_leagues_covers_eleven_leagues(monkeypatch):
    """Regression: an 11-league portfolio must not be silently capped at 8."""
    linked = [
        {"platform": "sleeper", "league_id": f"L{i}", "season": 2026,
         "name": f"League {i}", "team_id": str(i)}
        for i in range(11)
    ]
    monkeypatch.setattr(
        "dashboard_services.accounts.list_user_leagues",
        lambda aid: linked,
    )

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            class R:
                def fetchone(self):
                    return None
            return R()

    monkeypatch.setattr("dashboard_services.db.get_conn", lambda: _Conn())
    leagues = we.connected_leagues_for_account(
        42,
        primary_platform="sleeper",
        primary_league_id="L0",
        primary_season=2026,
        primary_roster_id="0",
        primary_name="League 0",
    )
    assert len(leagues) == 11
    assert leagues[0]["league_id"] == "L0"  # primary first
    assert {lg["league_id"] for lg in leagues} == {f"L{i}" for i in range(11)}
