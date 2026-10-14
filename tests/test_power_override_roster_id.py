"""Standings power card: the blended-score override must join on roster_id.

Regression for the Blackedraw bug: Nunky Figgas' Sleeper team name carries
a trailing space ("Nunky Figgas ") while team_stats owner is stripped, so
the old team-name-keyed override in render_power_and_playoffs missed and
silently replaced his blended PowerScore with 0.0, dropping him a spot and
promoting the team below him. The override (and the week-over-week
movement lookup) now key on roster_id, and a team that matches neither
keeps its performance PowerScore instead of being zeroed.
"""
import re

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")

import app as appmod


ROSTER_MAP = {
    "1": "Alpha",
    "2": "Beta",
    "3": "Nunky Figgas ",  # trailing space, exactly as Sleeper sends it
    "4": "Delta",
}


def _team_stats():
    return pd.DataFrame([
        {"owner": "Alpha", "avatar": "", "Wins": 4, "G": 5, "Ties": 0,
         "PF": 800.0, "PowerScore": 0.50},
        {"owner": "Beta", "avatar": "", "Wins": 4, "G": 5, "Ties": 0,
         "PF": 780.0, "PowerScore": 0.40},
        # team_stats owner is stripped: no trailing space here.
        {"owner": "Nunky Figgas", "avatar": "", "Wins": 3, "G": 5, "Ties": 0,
         "PF": 790.0, "PowerScore": 0.53},
        {"owner": "Delta", "avatar": "", "Wins": 3, "G": 5, "Ties": 0,
         "PF": 700.0, "PowerScore": 0.90},
    ])


def _power_rows(html):
    """[(pos, name, score)] parsed from the rendered power card rows."""
    lead = re.search(
        r"pwr-lead[ '].{0,900}?data-rk-key='([^']+)'", html, re.S)
    lead_score = re.search(r"pwr-lead-score'><b>([^<]+)</b>", html)
    rows = [(1, lead.group(1), lead_score.group(1))] \
        if lead and lead_score else []
    for chunk in re.split(r"<div class='pwr-row", html)[1:]:
        pos = re.search(r"pwr-pos'>(\d+)</span>", chunk)
        key = re.search(r"data-rk-key='([^']+)'", chunk)
        score = re.search(r"pwr-score'>([^<]+)</div>", chunk)
        if pos and key and score:
            rows.append((int(pos.group(1)), key.group(1), score.group(1)))
    return sorted(rows)


def _render(power_rankings, prev_power_ranks=None):
    return appmod.render_power_and_playoffs(
        _team_stats(), ROSTER_MAP, "league1", "sleeper", 2026,
        bracket_override=[], seed_map_override={},
        power_rankings=power_rankings,
        prev_power_ranks=prev_power_ranks,
    )


def test_trailing_space_team_keeps_its_blended_score():
    blended = [
        {"roster_id": "1", "team_name": "Alpha", "power_score": 1.4, "rank": 1},
        {"roster_id": "2", "team_name": "Beta", "power_score": 1.2, "rank": 2},
        {"roster_id": "3", "team_name": "Nunky Figgas ",
         "power_score": 1.1, "rank": 3},
        {"roster_id": "4", "team_name": "Delta", "power_score": 0.3, "rank": 4},
    ]
    rows = _power_rows(_render(blended))
    assert rows == [
        (1, "Alpha", "1.4"),
        (2, "Beta", "1.2"),
        (3, "Nunky Figgas", "1.1"),  # not the fabricated 0.0
        (4, "Delta", "0.3"),
    ]


def test_unmatched_team_keeps_performance_score_instead_of_zero():
    blended = [
        {"roster_id": "1", "team_name": "Alpha", "power_score": 1.4, "rank": 1},
        {"roster_id": "2", "team_name": "Beta", "power_score": 1.2, "rank": 2},
        {"roster_id": "3", "team_name": "Nunky Figgas ",
         "power_score": 1.1, "rank": 3},
        # Delta missing from the blended list entirely.
    ]
    rows = _power_rows(_render(blended))
    by_name = {name: (pos, score) for pos, name, score in rows}
    # Delta falls back to its team_stats PowerScore (0.9), never 0.0.
    assert by_name["Delta"] == (4, "0.9")


def test_movement_arrow_resolves_by_roster_id():
    blended = [
        {"roster_id": "1", "team_name": "Alpha", "power_score": 1.4, "rank": 1},
        {"roster_id": "2", "team_name": "Beta", "power_score": 1.2, "rank": 2},
        {"roster_id": "3", "team_name": "Nunky Figgas ",
         "power_score": 1.1, "rank": 3},
        {"roster_id": "4", "team_name": "Delta", "power_score": 0.3, "rank": 4},
    ]
    html = _render(blended, prev_power_ranks={"3": 5, "1": 1})
    # Nunky was 5th last week, is 3rd now: his row must show the up arrow
    # even though his name carries the trailing space on one side.
    nunky_chunk = re.split(r"<div class='pwr-row", html)[2]
    assert "Up 2 since last week" in nunky_chunk


def test_legacy_name_keyed_prev_ranks_still_resolve():
    blended = [
        {"roster_id": "1", "team_name": "Alpha", "power_score": 1.4, "rank": 1},
        {"roster_id": "2", "team_name": "Beta", "power_score": 1.2, "rank": 2},
    ]
    html = _render(blended, prev_power_ranks={"Alpha": 2})
    assert "Up 1 since last week" in html
