"""Crawler metadata extraction for the Draft Database (migration 056).

Fixtures use the real Sleeper payload shapes the crawler already fetches:
draft objects carry settings / slot_to_roster_id / metadata / type;
league objects carry name / settings (type + scoring_settings);
rosters carry roster_id + owner_id; users carry user_id + display_name +
metadata.team_name.
"""
from __future__ import annotations

from data_building.trade_intel.draft_adp_crawler import extract_draft_metadata

DRAFT = {
    "draft_id": "998877",
    "league_id": "112233",
    "season": "2026",
    "status": "complete",
    "type": "snake",
    "settings": {"rounds": 15, "teams": 12},
    "slot_to_roster_id": {"1": 4, "2": 9},
    "metadata": {"scoring_type": "ppr"},
}

LEAGUE = {
    "league_id": "112233",
    "name": "Alpha League",
    "settings": {"type": 2, "scoring_settings": {"rec": 1.0}},
}

USERS = [
    {"user_id": "u4", "display_name": "Owner Four",
     "metadata": {"team_name": "Four Play"}},
    {"user_id": "u9", "display_name": "Owner Nine", "metadata": {}},
]

ROSTERS = [
    {"roster_id": 4, "owner_id": "u4"},
    {"roster_id": 9, "owner_id": "u9"},
    {"roster_id": 7, "owner_id": None},
]


def test_extract_full_metadata():
    md = extract_draft_metadata(DRAFT, LEAGUE, USERS, ROSTERS)
    assert md["league_name"] == "Alpha League"
    assert md["scoring_type"] == "ppr"
    assert md["league_format"] == "dynasty"
    assert md["slot_to_roster_id"] == {"1": "4", "2": "9"}
    assert md["draft_order_type"] == "snake"
    # team_name wins over display_name; roster 7 has no owner and is omitted.
    assert md["roster_names"] == {"4": "Four Play", "9": "Owner Nine"}
    assert md["roster_owners"] == {"4": "u4", "9": "u9"}


def test_extract_scoring_from_league_settings_when_draft_silent():
    draft = {**DRAFT, "metadata": {}}
    league = {**LEAGUE, "settings": {"type": 0,
                                     "scoring_settings": {"rec": 0.5}}}
    md = extract_draft_metadata(draft, league, USERS, ROSTERS)
    assert md["scoring_type"] == "half"
    assert md["league_format"] == "redraft"


def test_extract_keeper_format_from_league_type_arg():
    md = extract_draft_metadata(DRAFT, None, None, None, league_type=1)
    assert md["league_format"] == "keeper"


def test_extract_unknowns_stay_none():
    md = extract_draft_metadata({"draft_id": "x"}, None, None, None)
    assert md["league_name"] is None
    assert md["scoring_type"] is None
    assert md["league_format"] is None
    assert md["slot_to_roster_id"] is None
    assert md["draft_order_type"] is None
    assert md["roster_names"] is None
    assert md["roster_owners"] is None


def test_extract_half_ppr_metadata_normalized():
    draft = {**DRAFT, "metadata": {"scoring_type": "half_ppr"}}
    md = extract_draft_metadata(draft, LEAGUE, USERS, ROSTERS)
    assert md["scoring_type"] == "half"
