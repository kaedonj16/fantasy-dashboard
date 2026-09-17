"""The K/IDP/DEF overlay must find the *non-dated* Sleeper stats file.

Both the live in-season fetcher and the history backfill write
``cache/sleeper_stats/sleeper_stats_s{Y}_w{W}.json`` (no date suffix). The
overlay used to glob only for a dated ``..._w{W}_{date}.json`` variant and fall
back to a path in the wrong directory, so it matched none of the 180+ real
files and silently no-oped -- leaving every kicker and defense as
"Stats unavailable" on completed weeks.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("requests")
pytest.importorskip("bs4")

import utils.utils as umod


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_overlay_finds_non_dated_file_and_populates_k_and_idp(tmp_path, monkeypatch):
    monkeypatch.setattr(umod, "CACHE_DIR", tmp_path)
    # Non-dated name, exactly what fetch_week_stats / the backfill write.
    _write(
        tmp_path / "sleeper_stats" / "sleeper_stats_s2026_w1.json",
        {
            "100": {"idp_sack": 2, "idp_tkl": 6},   # an IDP defender
            "200": {"fgm": 3, "fga": 3, "xpm": 2, "xpa": 2},  # a kicker
        },
    )
    monkeypatch.setattr(
        umod, "load_idp_index",
        lambda: {"100": {"name": "Rush Man", "team": "KC", "pos": "LB"}},
    )
    monkeypatch.setattr(
        umod, "load_players_index",
        lambda: {"200": {"name": "Boot Foot", "team": "KC", "pos": "PK"}},
    )

    lws: dict = {}
    umod.overlay_idp_and_k_stats_from_sleeper(
        league_week_stats=lws, season=2026, week=1, teams_index={"KC": {}},
    )

    assert "rush man" in (lws.get("KC", {}).get("IDP") or {})
    assert "boot foot" in (lws.get("KC", {}).get("K") or {})


def test_overlay_no_ops_cleanly_when_file_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(umod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(umod, "load_idp_index", lambda: {"100": {"name": "X", "team": "KC", "pos": "LB"}})
    monkeypatch.setattr(umod, "load_players_index", lambda: {})

    lws: dict = {"KC": {"QB": {"someone": {"pass_yds": 1}}}}
    umod.overlay_idp_and_k_stats_from_sleeper(
        league_week_stats=lws, season=2026, week=1, teams_index={"KC": {}},
    )
    # Untouched: no crash, no spurious buckets.
    assert lws == {"KC": {"QB": {"someone": {"pass_yds": 1}}}}
