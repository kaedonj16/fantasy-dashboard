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
    fake_alt.parse_tank_game_id = (
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
