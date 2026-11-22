"""Focused tests for the ScoreZone fix-all batch.

Covers the two backend gaps that were still open after PR #2102 (which
already landed the per-player play-store key, the cold-cache ``pending``
contract, and the week-scoped moments read):

- DST association in the ESPN PBP extractor: a defensive TD / interception
  must emit a row keyed by the fantasy DST pid, the same way other starters
  are matched, so the moments query can surface it for a DST starter.
- Poller fallback: ``maybe_fallback_collect`` only fires while games are
  live AND the store is stale, throttled per process, and the collect runs
  under the poller's advisory leader lock.
- ``poll_once`` passes a team -> DST pid map built from the player index
  into the PBP fetch.
"""
from __future__ import annotations

from typing import ClassVar

import utils.scorezone_store as rs
from utils.scorezone import extract_espn_pbp_plays, fetch_alt_pbp_plays


def _espn_payload(plays, drive_team="DET"):
    """ESPN gamepackage shape, mirroring tests/test_scorezone_alt_pbp.py."""
    return {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {"team": {"abbreviation": drive_team}, "plays": plays}
                ]
            }
        }
    }


def _play(pid, text, ptype, scoring=False, seq="1"):
    return {
        "id": seq,
        "text": text,
        "clock": {"displayValue": "2:41"},
        "period": {"number": 4},
        "start": {"down": 2, "distance": 7, "possessionText": "GB 14"},
        "type": {"text": ptype},
        "scoringPlay": scoring,
    }


_GAME = "20261025_DET@GB"  # away DET (drive team), home GB
_DEF_MAP = {"GB": "GB", "DET": "DET"}
_NAMES = {"jared goff": "4046", "jahmyr gibbs": "5500"}


def test_pick_six_emits_dst_row_for_defending_team():
    payload = _espn_payload([
        _play("1", "(Shotgun) J.Goff pass short middle INTERCEPTED by "
                   "K.Byard for 24 yards, TOUCHDOWN.", "Passing Touchdown",
              scoring=True),
    ])
    plays = extract_espn_pbp_plays(
        payload, _GAME, name_to_pid=_NAMES, team_to_def_pid=_DEF_MAP,
    )
    dst = [p for p in plays if p["pid"] == "GB"]
    assert len(dst) == 1
    assert dst[0]["stat_line"] == {"def_td": 1}
    assert dst[0]["is_td"] is True
    assert dst[0]["team"] == "GB"
    # The QB keeps his own row for the same play (per-player keying).
    assert any(p["pid"] == "4046" for p in plays)


def test_interception_emits_dst_def_int_row():
    payload = _espn_payload([
        _play("1", "J.Goff pass deep right INTERCEPTED by K.Byard at the "
                   "GB 40 for no gain.", "Pass"),
    ])
    plays = extract_espn_pbp_plays(
        payload, _GAME, name_to_pid=_NAMES, team_to_def_pid=_DEF_MAP,
    )
    dst = [p for p in plays if p["pid"] == "GB"]
    assert len(dst) == 1
    assert dst[0]["stat_line"] == {"def_int": 1}
    assert dst[0]["is_td"] is False


def test_offensive_td_emits_no_dst_row():
    payload = _espn_payload([
        _play("1", "J.Gibbs up the middle for 14 yards, TOUCHDOWN.",
              "Rushing Touchdown", scoring=True),
    ])
    plays = extract_espn_pbp_plays(
        payload, _GAME, name_to_pid=_NAMES, team_to_def_pid=_DEF_MAP,
    )
    assert not [p for p in plays if p["pid"] in ("GB", "DET")]
    assert any(p["pid"] == "5500" for p in plays)


def test_dst_rows_require_def_map():
    payload = _espn_payload([
        _play("1", "J.Goff pass deep right INTERCEPTED by K.Byard at the "
                   "GB 40 for no gain.", "Pass"),
    ])
    plays = extract_espn_pbp_plays(payload, _GAME, name_to_pid=_NAMES)
    assert not [p for p in plays if p["pid"] == "GB"]


def test_fetch_alt_forwards_team_to_def_pid(monkeypatch):
    seen = {}

    def fake_extract(payload, game_id, **kw):
        seen.update(kw)
        return [{"play_id": "p"}]

    monkeypatch.setattr(
        "utils.scorezone.fetch_espn_event_id", lambda **kw: "event-1")
    monkeypatch.setattr(
        "utils.scorezone.fetch_espn_pbp", lambda eid, ttl=0.0: {"ok": 1})
    monkeypatch.setattr("utils.scorezone.extract_espn_pbp_plays", fake_extract)
    fetch_alt_pbp_plays(
        _GAME, season=2026, week=8, team_to_def_pid=_DEF_MAP,
        providers=("espn",),
    )
    assert seen.get("team_to_def_pid") == _DEF_MAP


def test_build_def_pid_map_only_takes_def_entries():
    players = {
        "GB": {"full_name": "Green Bay Packers", "position": "DEF",
               "team": "GB"},
        "4046": {"full_name": "Jared Goff", "position": "QB", "team": "DET"},
        "DET": {"full_name": "Detroit Lions", "position": "DEF",
                "team": "DET"},
    }
    assert rs._build_def_pid_map(players) == {"GB": "GB", "DET": "DET"}


# ---------------------------------------------------------------------------
# Poller fallback gating
# ---------------------------------------------------------------------------

class _FakeThread:
    started: ClassVar[list] = []

    def __init__(self, target=None, name=None, daemon=None):
        self._target = target

    def start(self):
        _FakeThread.started.append(self._target)


def _gate(monkeypatch, *, stale, live):
    monkeypatch.setattr(rs, "store_is_stale", lambda *a, **k: stale)
    monkeypatch.setattr(
        rs, "discover_live_games",
        lambda current_week=None: [{"game_id": "g", "live": live,
                                    "final": not live, "week": 8}],
    )
    import threading

    monkeypatch.setattr(threading, "Thread", _FakeThread)
    _FakeThread.started = []
    rs._fallback_attempts.clear()


def test_fallback_fires_when_live_and_stale(monkeypatch):
    _gate(monkeypatch, stale=True, live=True)
    assert rs.maybe_fallback_collect(2026, 8, now=1000.0) is True
    assert _FakeThread.started == [rs._fallback_collect_once]


def test_fallback_skips_when_store_fresh(monkeypatch):
    _gate(monkeypatch, stale=False, live=True)
    assert rs.maybe_fallback_collect(2026, 8, now=1000.0) is False
    assert _FakeThread.started == []


def test_fallback_skips_when_nothing_live(monkeypatch):
    _gate(monkeypatch, stale=True, live=False)
    assert rs.maybe_fallback_collect(2026, 8, now=1000.0) is False
    assert _FakeThread.started == []


def test_fallback_is_throttled(monkeypatch):
    _gate(monkeypatch, stale=True, live=True)
    assert rs.maybe_fallback_collect(2026, 8, now=1000.0) is True
    assert rs.maybe_fallback_collect(2026, 8, now=1030.0) is False
    assert len(_FakeThread.started) == 1
    # After the throttle window it may fire again.
    assert rs.maybe_fallback_collect(2026, 8, now=1000.0 + 121) is True


def test_store_is_stale_threshold(monkeypatch):
    monkeypatch.setattr(rs, "latest_observed_ts", lambda season, week=None: 1000.0)
    assert rs.store_is_stale(2026, 8, now=1000.0 + 179) is False
    assert rs.store_is_stale(2026, 8, now=1000.0 + 181) is True
    monkeypatch.setattr(rs, "latest_observed_ts", lambda season, week=None: 0.0)
    assert rs.store_is_stale(2026, 8, now=1000.0) is True


def test_poll_once_passes_def_map_to_pbp(monkeypatch):
    import sys
    import types

    seen = {}

    fake_api = types.ModuleType("dashboard_services.api")
    fake_api.get_nfl_state = lambda: {"season": 2026, "week": 8,
                                      "season_type": "reg"}
    fake_api.get_nfl_players = lambda: {
        "GB": {"full_name": "Green Bay Packers", "position": "DEF",
               "team": "GB"},
        "4046": {"full_name": "Jared Goff", "position": "QB", "team": "DET"},
    }
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)

    fake_alt = types.ModuleType("utils.scorezone")
    fake_alt.fetch_alt_pbp_plays = (
        lambda gid, **kw: seen.update(kw) or [{"play_id": "p1"}]
    )
    fake_alt.parse_game_id = (
        lambda gid: (gid.split("_", 1)[0], "DET", "GB")
    )
    fake_alt._normalize_name = lambda name: name
    fake_alt._extract_first_initial_last = lambda name: ""
    monkeypatch.setitem(sys.modules, "utils.scorezone", fake_alt)

    monkeypatch.setattr(rs, "discover_live_games", lambda current_week=None: [
        {"game_id": _GAME, "live": True, "final": False, "week": 8},
    ])
    monkeypatch.setattr(rs, "upsert_plays", lambda *a, **k: 1)

    stats = rs.poll_once()
    assert stats["games"] == 1
    assert seen.get("team_to_def_pid") == {"GB": "GB"}


# ── TEN@DAL 2026-11-22: the eligible-reporter TD play ─────────────────────
# Real booth text captured live from the game. "J.Williams" is ambiguous in
# the two-team index (James Williams Sr. and Joshua Williams, both TEN) but
# unique on DAL, and the play opens with an eligibility report for a DAL
# lineman. Before the fix the rushing TD went uncredited, Javonte Williams
# got no row, and the rows for the lineman and the penalized defender both
# carried the TD flag, so the client headlined the lineman's 0-point card.

_TEN_DAL_NFL = {
    "7588": {"full_name": "Javonte Williams", "team": "DAL", "position": "RB"},
    "11847": {"full_name": "James Williams Sr.", "team": "TEN", "position": "DB"},
    "8470": {"full_name": "Joshua Williams", "team": "TEN", "position": "CB"},
    "10884": {"full_name": "Broderick Jones", "team": "DAL", "position": "OL"},
    "5466": {"full_name": "John Franklin-Myers", "team": "TEN", "position": "DT"},
    "3294": {"full_name": "Dak Prescott", "team": "DAL", "position": "QB"},
    "8170": {"full_name": "James Mitchell", "team": "DAL", "position": "TE"},
    "9999": {"full_name": "George Pickens", "team": "DAL", "position": "WR"},
}

_TD_TEXT = (
    "B.Jones reported in as eligible.  J.Williams up the middle for 1 yard, "
    "TOUCHDOWN.PENALTY on TEN-J.Franklin-Myers, Unnecessary Roughness, 1 "
    "yard, enforced between downs. TWO-POINT CONVERSION ATTEMPT. D.Prescott "
    "pass to G.Pickens is incomplete. ATTEMPT FAILS."
)


def _ten_dal_maps():
    return rs._build_name_maps(_TEN_DAL_NFL, {"TEN", "DAL"})


def _raw_play(pid, text, ptype, scoring=False, seq="1", clock="8:23"):
    return {
        "id": seq,
        "text": text,
        "clock": {"displayValue": clock},
        "period": {"number": 1},
        "start": {"down": 2, "distance": 1, "possessionText": "TEN 1"},
        "type": {"text": ptype},
        "scoringPlay": scoring,
    }


def test_eligible_reporter_td_credits_the_rusher_not_the_lineman():
    name_to_pid, meta = _ten_dal_maps()
    payload = _espn_payload(
        [_raw_play("401873073442", _TD_TEXT, "Rushing Touchdown", scoring=True)],
        drive_team="DAL",
    )
    plays = extract_espn_pbp_plays(
        payload, "20261122_TEN@DAL", name_to_pid=name_to_pid,
        player_meta_by_pid=meta, team_to_def_pid={"TEN": "TEN", "DAL": "DAL"},
    )
    by_pid = {p["pid"]: p for p in plays}
    # The rusher is credited and is the ONLY touchdown row.
    assert by_pid["7588"]["stat_line"].get("rush_td") == 1
    assert by_pid["7588"]["stat_line"].get("rush_yds") == 1
    assert [p["pid"] for p in plays if p["is_td"]] == ["7588"]
    # The eligibility reporter gets no row at all.
    assert "10884" not in by_pid
    # The penalized defender keeps his mention row but not the TD flag.
    assert by_pid["5466"]["is_td"] is False


def test_eligible_reporter_completion_play_has_no_lineman_row():
    name_to_pid, meta = _ten_dal_maps()
    text = (
        "B.Jones reported in as eligible.  D.Prescott pass short left to "
        "J.Mitchell to TEN 47 for 13 yards (K.Winston)."
    )
    payload = _espn_payload(
        [_raw_play("401873073113", text, "Pass", clock="13:49")],
        drive_team="DAL",
    )
    plays = extract_espn_pbp_plays(
        payload, "20261122_TEN@DAL", name_to_pid=name_to_pid,
        player_meta_by_pid=meta, team_to_def_pid={"TEN": "TEN", "DAL": "DAL"},
    )
    by_pid = {p["pid"]: p for p in plays}
    assert "10884" not in by_pid
    assert by_pid["3294"]["stat_line"].get("pass_yds") == 13
    assert by_pid["8170"]["stat_line"].get("rec_yds") == 13
    assert not any(p["is_td"] for p in plays)


def test_injury_update_note_names_no_actor():
    # Real booth text from IND@HOU 2026-11-20: ESPN appended an injury note
    # about a DB who had just returned, and he surfaced as a touchdown row
    # on the TD pass he was nowhere near.
    nfl = {
        "9764": {"full_name": "C.J. Stroud", "team": "HOU", "position": "QB"},
        "12500": {"full_name": "Marlin Klein", "team": "HOU", "position": "TE"},
        "11371": {"full_name": "Ka'imi Fairbairn", "team": "HOU", "position": "K"},
        "12468": {"full_name": "Kapena Gushiken", "team": "IND", "position": "DB"},
    }
    name_to_pid, meta = rs._build_name_maps(nfl, {"IND", "HOU"})
    text = (
        "(Shotgun) C.Stroud pass short right to M.Klein for 13 yards, "
        "TOUCHDOWN. ** Injury Update: IND-K.Gushiken has returned to the "
        "game. K.Fairbairn extra point is GOOD, Center-A.Brinkman, "
        "Holder-K.Kroeger."
    )
    payload = _espn_payload(
        [_raw_play("4018730662675", text, "Passing Touchdown", scoring=True)],
        drive_team="HOU",
    )
    plays = extract_espn_pbp_plays(
        payload, "20261120_IND@HOU", name_to_pid=name_to_pid,
        player_meta_by_pid=meta, team_to_def_pid={"IND": "IND", "HOU": "HOU"},
    )
    by_pid = {p["pid"]: p for p in plays}
    assert "12468" not in by_pid
    assert by_pid["12500"]["stat_line"].get("rec_td") == 1
    assert by_pid["12500"]["is_td"] is True
    assert by_pid["9764"]["is_td"] is True
    assert by_pid["11371"]["is_td"] is False
