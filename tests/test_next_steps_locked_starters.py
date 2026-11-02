"""Next Steps must never suggest benching a locked starter.

Regression for Kaedon's 2026-11-02 report: Jeremiyah Love got hurt
mid-game, his live projection cratered to 0.0, and Next Steps told her
"Start Quinshon Judkins over Jeremiyah Love / Judkins projects 10.8 vs
Love's 0.0" - an impossible move, because she had already started Love
and his game had kicked off. Every other projection_upgrades caller
(digest, push, hub hint, portfolio) already passes locked_pids; the
Next Steps lineup actions were the one surface that did not.
"""
import time

import pytest

pd = pytest.importorskip("pandas")

import app as app_module  # noqa: E402

NOW = time.time()

PLAYERS = {
    # pid: (name, team, position)
    "love": ("Jeremiyah Love", "ARI", "RB"),
    "judkins": ("Quinshon Judkins", "CLE", "RB"),
    "rb_ok": ("Steady Back", "DET", "RB"),
}
PROJ = {"love": 0.0, "judkins": 10.8, "rb_ok": 12.0}


def _ctx():
    return {
        "platform": "sleeper",
        "league_id": "L1",
        "current_season": 2026,
        "current_week": 9,
        "rosters": [{
            "roster_id": 1,
            "starters": ["love", "rb_ok"],
            "players": ["love", "rb_ok", "judkins"],
            "reserve": [],
            "taxi": [],
        }],
        "players_map": {
            pid: {"name": name, "team": team, "pos": pos}
            for pid, (name, team, pos) in PLAYERS.items()
        },
        "roster_positions": ["RB", "RB", "BN", "BN"],
        "proj_by_week": {9: {"projections": dict(PROJ)}},
    }


def _fixture(monkeypatch, *, love_kickoff, love_status=""):
    full = {
        pid: {
            "full_name": name, "team": team, "position": pos,
            "injury_status": love_status if pid == "love" else "",
        }
        for pid, (name, team, pos) in PLAYERS.items()
    }
    monkeypatch.setattr(app_module, "get_players_global", lambda: full)

    def fake_schedule(season, week):
        return [
            {"home": "ARI", "away": "SF", "gameTime_epoch": love_kickoff},
            {"home": "CLE", "away": "DET", "gameTime_epoch": NOW + 3600},
        ]

    monkeypatch.setattr(app_module, "load_week_schedule", fake_schedule)


def _actions(monkeypatch, **kw):
    _fixture(monkeypatch, **kw)
    with app_module.app.test_request_context("/"):
        return app_module._next_steps_lineup_actions(_ctx(), 1)


def test_mid_game_injured_locked_starter_gets_no_swap_nudge(monkeypatch):
    # Love's game kicked off an hour ago and he is hurt (proj 0.0):
    # no "Start Judkins over Love" action may be produced.
    actions = _actions(monkeypatch, love_kickoff=NOW - 3600)
    blob = " ".join(a["action"] for a in actions)
    assert "Judkins" not in blob
    assert actions == []


def test_same_swap_is_suggested_before_kickoff(monkeypatch):
    # Control: with Love's game still to come, the identical projection
    # gap IS a legitimate, actionable suggestion.
    actions = _actions(monkeypatch, love_kickoff=NOW + 3600)
    assert len(actions) == 1
    assert actions[0]["action"] == "Start Quinshon Judkins over Jeremiyah Love"
    assert actions[0]["why"] == (
        "Quinshon Judkins projects 10.8 vs Jeremiyah Love's 0.0."
    )


def test_locked_starter_gets_no_address_alert_when_listed_out(monkeypatch):
    # If the mid-game injury also flips his designation to Out, the
    # generic "Address:" alert is just as unactionable once locked.
    actions = _actions(monkeypatch, love_kickoff=NOW - 3600, love_status="Out")
    assert not any(a["action"].startswith("Address:") for a in actions)


def test_unlocked_out_starter_still_surfaces(monkeypatch):
    # Before kickoff an Out starter is a real, fixable problem: he must
    # still surface (here as the swap-out side, which also covers the
    # generic Address alert for the same player).
    actions = _actions(monkeypatch, love_kickoff=NOW + 3600, love_status="Out")
    assert any("Jeremiyah Love" in a["action"] for a in actions)
