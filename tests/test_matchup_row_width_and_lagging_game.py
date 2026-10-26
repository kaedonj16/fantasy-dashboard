"""Matchup row width + lagging-game stats (Kaedon's screenshot, 2026-10-25).

Three reports from one phone screenshot of the matchup board:

1. The live pulsing dot is clipped. .live-dot's pulse ring scales to
   2.6x the 8px dot (~6.4px past it on every side) and .mb-game's
   overflow:hidden sliced the ring flat on the left and bottom.
2. The drive bar is wider than the text. The bar (and the stat row)
   span the player cell PLUS the score column, while the name / team /
   game text lived only in the cell column: the bar stuck out past the
   text on the outer edge and the game line ellipsized a score-column
   early ("Q4 11:30 14-2...", "Sun 9:30 am @ ..."). Everything on a
   side now shares the bar's box.
3. G. Bernard (and J. Warren -- both PIT) showed real points but no
   game stats, and a pregame "Sun 9:30 am @ ..." game line at 3:30pm.
   Their 9:30am game's schedule row still read statusCode "0"
   (scheduled) hours after kickoff: game_has_started only forgave a
   past *date*, format_team_game_line only corrected a past date or a
   future same-day kickoff, and player_bits let the schedule verdict
   suppress the box score (and block the Sleeper gap fill) even though
   the pid status already said FINAL.
"""
import pathlib
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from dashboard_services import matchups as mmod
from utils.game_status import game_has_started

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------- width / dot

def test_text_and_bar_share_one_width_per_side():
    # Text insets match the drive bar's 1px inset on the outer edge.
    assert ".m-combo .mb-cell { padding-inline: 1px; }" in CSS
    assert ".m-combo .mb-stat-row { padding-inline: 1px; }" in CSS
    # The game line extends across the score column (36px + 7px gap),
    # the same span the drive bar and stat row already used.
    assert ".m-combo .mb-cell-l .mb-game { margin-right: -43px; max-width: none; }" in CSS
    assert ".m-combo .mb-cell-r .mb-game { margin-left: -43px; max-width: none; width: calc(100% + 43px); }" in CSS
    # The pre-existing base rules the older pins assert stay untouched;
    # the unification is an override scoped to the slide board.
    assert ".mb-cell { padding: 8px 12px; }" in CSS
    assert ".mb-stat-row { min-width: 0; padding: 0 9px; margin-top: -3px; }" in CSS


def test_live_dot_pulse_gets_headroom_inside_the_clip_box():
    assert ".m-combo .mb-game:has(.live-dot) {" in CSS
    assert "padding-block: 5px;" in CSS
    assert "margin-block: -5px;" in CSS
    assert "padding-left: 8px;" in CSS
    assert "margin-left: -8px;" in CSS
    # Right cell: headroom is padding-only; its margin stays at the -43px
    # extension so the right edge stays pinned to the bar's right edge.
    assert ".m-combo .mb-cell-r .mb-game:has(.live-dot) { margin-left: -43px; }" in CSS


def test_service_worker_bumped():
    assert "br-fantasy-v98" in SW


# ------------------------------------------------------- lagging schedule game

def _today_str():
    return date.today().strftime("%Y%m%d")


def test_started_when_code0_but_todays_kickoff_passed_epoch():
    now = datetime(2026, 10, 25, 15, 30, tzinfo=ET)
    game = {
        "gameStatusCode": "0",
        "gameDate": "20261025",
        "gameTime_epoch": (now - timedelta(hours=6)).timestamp(),
    }
    assert game_has_started(game, now=now.astimezone(timezone.utc)) is True


def test_not_started_when_code0_and_todays_kickoff_is_future():
    now = datetime(2026, 10, 25, 15, 30, tzinfo=ET)
    game = {
        "gameStatusCode": "0",
        "gameDate": "20261025",
        "gameTime_epoch": (now + timedelta(hours=2)).timestamp(),
    }
    assert game_has_started(game, now=now.astimezone(timezone.utc)) is False


def test_started_when_code0_today_with_tank_time_string():
    now = datetime(2026, 10, 25, 15, 30, tzinfo=ET)
    game = {"gameStatusCode": "0", "gameDate": "20261025", "gameTime": "9:30a"}
    assert game_has_started(game, now=now.astimezone(timezone.utc)) is True
    early = datetime(2026, 10, 25, 8, 0, tzinfo=ET)
    assert game_has_started(game, now=early.astimezone(timezone.utc)) is False


def test_code0_past_and_future_dates_unchanged():
    now = datetime(2026, 10, 25, 15, 30, tzinfo=ET)
    past = {"gameStatusCode": "0", "gameDate": "20261024"}
    future = {"gameStatusCode": "0", "gameDate": "20261026", "gameTime": "9:30a"}
    assert game_has_started(past, now=now.astimezone(timezone.utc)) is True
    assert game_has_started(future, now=now.astimezone(timezone.utc)) is False


def _bernard_slide(monkeypatch, fake_clock=False):
    """The screenshot's PIT row: pid status FINAL with real points, the
    schedule game still 'scheduled' for today at 9:30am."""
    today = _today_str()
    game = {
        "home": "MIN", "away": "PIT",
        "gameStatusCode": "0", "gameDate": today, "gameTime": "9:30a",
        "gameID": f"{today}@MIN",
    }
    weekly = {"PIT": {"WR": {"g bernard": {
        "rec": 3, "tgt": 5, "rec_yds": 27, "rec_td": 0, "_src": "tank",
    }}}}
    monkeypatch.setattr(mmod, "load_teams_index", lambda: {})
    monkeypatch.setattr(mmod, "build_offense_rankings", lambda *_: {})
    monkeypatch.setattr(mmod, "load_week_stats", lambda *_: weekly)
    monkeypatch.setattr(mmod, "load_week_schedule", lambda *_: [game])
    monkeypatch.setattr(
        mmod, "build_team_schedule_lookup", lambda *_: {"PIT": game, "MIN": game}
    )
    if fake_clock:
        fixed = datetime(2026, 10, 25, 15, 30)

        class _DT(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed

        class _D(date):
            @classmethod
            def today(cls):
                return fixed.date()

        monkeypatch.setattr(mmod, "datetime", _DT)
        monkeypatch.setattr(mmod, "date", _D)
        monkeypatch.setattr(mmod, "get_nfl_scores_for_date", lambda *_: {})
        game["gameDate"] = "20261025"
    matchup = {
        "left": {"name": "Left", "roster_id": "1", "starters": [], "pts_total": 0.0},
        "right": {
            "name": "Right", "roster_id": "2", "pts_total": 5.7,
            "starters": [{
                "pid": "10", "name": "G Bernard", "pos": "WR",
                "nfl": "PIT", "pts": 5.7,
            }],
        },
    }
    return mmod.render_matchup_slide(
        "2026", matchup, 8, 8, {"10": "final"}, {}, {}, {}, {},
        roster_positions=["WR"],
    )


def test_final_players_stats_survive_a_lagging_schedule_game(monkeypatch):
    html = _bernard_slide(monkeypatch)
    # The box score renders (before the fix the not-started schedule
    # verdict suppressed it and blocked the Sleeper gap fill too).
    assert "REC 3/5 27 yds" in html
    assert "Stats unavailable" not in html


def test_pregame_player_gets_no_stats_unavailable_marker(monkeypatch):
    # The absent-status-map fallback can label an unplayed player FINAL;
    # the "Stats unavailable" marker must still wait for the GAME to
    # confirm it was played (here: tomorrow's kickoff), or every pregame
    # row would carry it. The status-wins stats gate above must not leak
    # into the marker.
    tomorrow = (date.today() + timedelta(days=1)).strftime("%Y%m%d")
    game = {
        "home": "PIT", "away": "GB",
        "gameStatusCode": "0", "gameDate": tomorrow, "gameTime": "1:00p",
        "gameID": f"{tomorrow}@PIT",
    }
    monkeypatch.setattr(mmod, "load_teams_index", lambda: {})
    monkeypatch.setattr(mmod, "build_offense_rankings", lambda *_: {})
    monkeypatch.setattr(mmod, "load_week_stats", lambda *_: {})
    monkeypatch.setattr(mmod, "load_week_schedule", lambda *_: [game])
    monkeypatch.setattr(
        mmod, "build_team_schedule_lookup", lambda *_: {"GB": game, "PIT": game}
    )
    matchup = {
        "left": {"name": "Left", "roster_id": "1", "starters": [], "pts_total": 0.0},
        "right": {
            "name": "Right", "roster_id": "2", "pts_total": 0.0,
            "starters": [{
                "pid": "20", "name": "Jordan Love", "pos": "QB",
                "nfl": "GB", "pts": 0.0,
            }],
        },
    }
    html = mmod.render_matchup_slide(
        "2026", matchup, 8, 8, {"20": "final"}, {}, {}, {}, {},
        roster_positions=["QB"],
    )
    assert "Stats unavailable" not in html
    assert "1:00 pm" in html  # still a pregame kickoff line


def test_lagging_game_line_corrects_to_final_after_the_window(monkeypatch):
    html = _bernard_slide(monkeypatch, fake_clock=True)
    # 9:30am kickoff read at 3:30pm: past the 3h window, so the line is
    # a final line, not the pregame kickoff the screenshot showed.
    assert "mb-final-tag" in html
    assert "9:30 am" not in html
    assert "REC 3/5 27 yds" in html
