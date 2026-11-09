"""ScoreZone team labels for Yahoo leagues (Kaedon report 2026-11-08:
every team name in ScoreZone for her Yahoo league read "Sunday alex").

Root cause: Yahoo's viewer-personalized teams payload stamps the API
caller's guid as the manager guid on every team (verified live 2026-10-22:
no usable is_current_login flags), so every roster's owner_id is the same
value and every user's user_id is the same value. The ScoreZone client
resolved a roster's label with find(user_id === owner_id), which lands on
the FIRST user -- team 1, "Sunday alex" -- for every roster. The collect
payload also stripped each roster down to roster_id/owner_id, leaving the
client no per-team label to fall back on.

Fix pinned here:
- ``_scorezone_collect`` carries a per-roster ``team_name`` resolved by
  ``_rz_roster_team_names`` (roster metadata first, then the user matched
  by roster_id -- never by the collapsible guid).
- The shipped ``_ownerName`` in static/scorezone.js uses that team name
  whenever another roster in the same league scope shares the owner_id
  (driven for real under Node in tests/scorezone_owner_name_harness.mjs).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _yahoo_shaped():
    """Real Yahoo shape for league 1307110: the caller's guid (and, in the
    worst case, one collapsed display entry) on every team, while each
    team's own name stays distinct and public."""
    names = {1: "Sunday alex", 4: "Dicker? I hardly know her", 6: "Sixth Sense"}
    rosters = [
        {"roster_id": rid, "owner_id": "CALLER-GUID",
         "metadata": {"team_name": name}, "players": [], "starters": []}
        for rid, name in names.items()
    ]
    users = [
        {"user_id": "CALLER-GUID", "roster_id": rid,
         "display_name": "Sunday alex",
         "metadata": {"team_name": name}, "team_name": name}
        for rid, name in names.items()
    ]
    return rosters, users


def test_roster_team_names_are_per_team_despite_shared_guid():
    rosters, users = _yahoo_shaped()
    out = app._rz_roster_team_names(rosters, users)
    assert out == {
        "1": "Sunday alex",
        "4": "Dicker? I hardly know her",
        "6": "Sixth Sense",
    }


def test_roster_team_names_from_roster_metadata_without_user_roster_id():
    # Sleeper/ESPN shape: users carry no roster_id; the roster's own
    # metadata team_name still resolves.
    rosters = [{"roster_id": 2, "owner_id": "u2",
                "metadata": {"team_name": "Metadata Name"}}]
    users = [{"user_id": "u2", "display_name": "Owner Name"}]
    assert app._rz_roster_team_names(rosters, users) == {"2": "Metadata Name"}


def test_roster_team_names_empty_when_no_label_anywhere():
    rosters = [{"roster_id": 3, "owner_id": "u3"}]
    assert app._rz_roster_team_names(rosters, []) == {}


def test_collect_payload_carries_team_name_per_roster():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '"team_name": _rz_team_names.get(str(r.get("roster_id")), "")' in src


def test_owner_name_harness_passes():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the JS owner-name harness")
    harness = ROOT / "tests" / "scorezone_owner_name_harness.mjs"
    result = subprocess.run(
        [node, str(harness)], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL SCOREZONE OWNER NAME HARNESS CHECKS PASSED" in result.stdout
