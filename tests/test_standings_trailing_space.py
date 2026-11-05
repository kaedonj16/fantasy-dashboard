"""Trailing-space team names must never drop a team from the seed map.

League of Greatness (Sleeper 1312054460171718656) has two team names with a
trailing space in Sleeper's user metadata: "I Chase Brown Kids " (roster 4)
and "Butter Boys " (roster 3). build_roster_map kept the space while
build_tables strips team_stats owners via public_owner_label, so
build_standings_map's exact-string join dropped both teams: its seed map had
8 entries instead of 10 and Rebuild from Hell (roster 10, truly 9th at
1-7 / 782.38 PF through Week 8) read 8th on its My Leagues card.

The standings fixture below is that league's verified Week 8 table
(records / PF / PA recomputed from the Sleeper matchup feed, matching
Sleeper's roster settings exactly). True flat order: 9, 5, 8, 1, 2, 7, 4,
6, 10, 3 by roster_id.
"""
import pytest

from utils.league_payload import build_roster_map

# (roster_id, team name as Sleeper stores it, wins, losses, PF, PA)
_GREATNESS = [
    (1, "Super Cena", 5, 3, 1290.42, 1066.84),
    (2, "Kirk Patullos", 4, 4, 999.28, 1104.44),
    (3, "Butter Boys ", 1, 7, 752.08, 990.98),
    (4, "I Chase Brown Kids ", 3, 5, 1019.54, 1053.52),
    (5, "JiggyJay30", 7, 1, 1162.28, 923.06),
    (6, "the foolies", 2, 6, 954.42, 1084.48),
    (7, "Lisan Al Caleb", 4, 4, 895.96, 1021.06),
    (8, "jmoneybags91", 6, 2, 1192.02, 982.42),
    (9, "roundsd34", 7, 1, 1181.22, 926.54),
    (10, "Rebuild from Hell", 1, 7, 782.38, 1076.26),
]

_EXPECTED_SEEDS = {9: 1, 5: 2, 8: 3, 1: 4, 2: 5, 7: 6, 4: 7, 6: 8, 10: 9, 3: 10}


def _sleeper_payloads():
    """Sleeper-shaped users/rosters: team_name lives in user metadata and
    roster metadata carries none, exactly as the live league returns it."""
    users, rosters = [], []
    for rid, name, *_ in _GREATNESS:
        uid = str(1000 + rid)
        users.append({
            "user_id": uid,
            "display_name": f"user{rid}",
            "metadata": {"team_name": name},
        })
        rosters.append({"roster_id": rid, "owner_id": uid, "metadata": {}})
    return users, rosters


def _team_stats(strip_owners):
    pd = pytest.importorskip("pandas")
    rows = []
    for rid, name, wins, losses, pf, pa in _GREATNESS:
        rows.append({
            "owner": name.strip() if strip_owners else name,
            "Wins": wins, "Losses": losses, "Ties": 0, "PF": pf, "PA": pa,
        })
    return pd.DataFrame(rows)


def _seeds(team_stats, roster_map):
    pytest.importorskip("flask")
    pytest.importorskip("pandas")
    from dashboard_services.service import build_standings_map
    return build_standings_map(team_stats, roster_map)


# ---- build_roster_map strips ----------------------------------------------

def test_roster_map_strips_user_metadata_team_name():
    users, rosters = _sleeper_payloads()
    roster_map = build_roster_map(users, rosters)
    assert roster_map["4"] == "I Chase Brown Kids"
    assert roster_map["3"] == "Butter Boys"
    assert roster_map["10"] == "Rebuild from Hell"
    assert len(roster_map) == 10


def test_roster_map_strips_roster_metadata_team_name_and_display_name():
    users = [{"user_id": "1", "display_name": "  padded owner  "}]
    rosters = [{"roster_id": 7, "owner_id": "1",
                "metadata": {"team_name": "Spacey Team "}}]
    assert build_roster_map(users, rosters) == {"7": "Spacey Team"}
    rosters = [{"roster_id": 7, "owner_id": "1", "metadata": {}}]
    assert build_roster_map(users, rosters) == {"7": "padded owner"}


# ---- build_standings_map joins whitespace-insensitively --------------------

def test_standings_map_seeds_every_team_with_unstripped_roster_map():
    # Pre-fix production combination: stripped team_stats owners against a
    # roster_map that still carries the trailing spaces.
    roster_map = {str(rid): name for rid, name, *_ in _GREATNESS}
    seeds = _seeds(_team_stats(strip_owners=True), roster_map)
    assert seeds == _EXPECTED_SEEDS
    assert seeds[10] == 9  # Rebuild from Hell's true rank, not 8th
    assert len(seeds) == 10


def test_standings_map_seeds_every_team_with_unstripped_team_stats():
    # The mirror mismatch (space on the team_stats side) resolves the same.
    roster_map = {str(rid): name.strip() for rid, name, *_ in _GREATNESS}
    seeds = _seeds(_team_stats(strip_owners=False), roster_map)
    assert seeds == _EXPECTED_SEEDS


def test_end_to_end_sleeper_payloads_seed_rebuild_ninth():
    users, rosters = _sleeper_payloads()
    roster_map = build_roster_map(users, rosters)
    seeds = _seeds(_team_stats(strip_owners=True), roster_map)
    assert seeds == _EXPECTED_SEEDS
    assert len(seeds) == 10
