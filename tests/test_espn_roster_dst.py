"""ESPN get_rosters must canonicalize D/ST the same way matchups do.

Negative ESPN defense ids (``-160xx``) are never in the espnID crosswalk, so a
plain ``canon_pid`` lookup drops every defense from team modals and the Teams
page. Matchups already special-cased this; rosters must too.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dashboard_services.providers import espn_api as E


def test_dst_canonical_id_accepts_string_pro_team():
    bp = SimpleNamespace(proTeam="JAX", proTeamId=None)
    assert E._dst_canonical_id(bp, -16030) == "JAX"


def test_dst_canonical_id_wsh_to_was():
    bp = SimpleNamespace(proTeam="WSH")
    assert E._dst_canonical_id(bp, -16028) == "WAS"


def test_dst_canonical_id_falls_back_to_negative_id_math():
    assert E._dst_canonical_id(None, -16030) == "JAX"
    assert E._dst_canonical_id(SimpleNamespace(), -16012) == "KC"


def test_resolve_espn_player_id_maps_dst_and_skill():
    canon = {"4039057": "4046"}
    assert E.resolve_espn_player_id(-16030, canon, player=SimpleNamespace(proTeam="JAX")) == "JAX"
    assert E.resolve_espn_player_id("4039057", canon) == "4046"
    assert E.resolve_espn_player_id("999999", canon) is None


def test_get_rosters_includes_dst(monkeypatch):
    dst = SimpleNamespace(
        playerId=-16030,
        name="Jaguars D/ST",
        lineupSlot="D/ST",
        slot_position=None,
        slotPosition=None,
        proTeam="JAX",
        proTeamId=None,
    )
    skill = SimpleNamespace(
        playerId=3117251,
        name="Christian McCaffrey",
        lineupSlot="RB",
        slot_position=None,
        slotPosition=None,
        proTeam="SF",
    )
    team = SimpleNamespace(
        team_id=1,
        team_name="Trilogy",
        name="Trilogy",
        owners=[{"id": "{AAA}", "displayName": "mgr"}],
        wins=1,
        losses=0,
        ties=0,
        outcomes=["W"],
        points_for=100.5,
        points_against=90.0,
        roster=[skill, dst],
    )
    lg = SimpleNamespace(teams=[team])

    monkeypatch.setattr(E, "_league", lambda season, league_id: lg)
    monkeypatch.setattr(E, "_espn_to_canon_cached", lambda: {"3117251": "4034"})

    rosters = E.get_rosters(2025, "1")
    assert len(rosters) == 1
    r = rosters[0]
    assert "4034" in r["players"]
    assert "JAX" in r["players"]
    assert "JAX" in r["starters"]
    assert r["metadata"].get("team_name") == "Trilogy"
    assert r["owner_id"] == "{AAA}"
