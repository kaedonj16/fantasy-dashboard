"""Resolver tests: name normalization and the namesake policy.

Pure Python (explicit index fixtures, no app import), so this module
collects in every CI shard. The fixtures mirror the real players index
shapes: pid -> {name, team, pos}, including the Tank01 index's stored
spelling for Jaxon Smith-Njigba and the two Justin Jeffersons.
"""
from dashboard_services.assistant.resolve import _normalize, resolve_player

INDEX = {
    "6794": {"name": "Justin Jefferson", "team": "MIN", "pos": "WR"},
    "13524": {"name": "Justin Jefferson", "team": "CLE", "pos": "LB"},
    "9488": {"name": "Jaxon Smith-Njigba", "team": "SEA", "pos": "WR"},
    "7564": {"name": "Ja'Marr Chase", "team": "CIN", "pos": "WR"},
    "100": {"name": "Chris Smith", "team": "NYG", "pos": "WR"},
    "101": {"name": "Chris Smith", "team": "DAL", "pos": "RB"},
    "102": {"name": "DeVonta Smith", "team": "PHI", "pos": "WR"},
}


def test_normalize_apostrophes_join_and_hyphens_split():
    assert _normalize("Ja'Marr Chase") == "jamarr chase"
    assert _normalize("Ja’Marr Chase") == "jamarr chase"
    assert _normalize("Smith-Njigba") == _normalize("Smith Njigba")
    assert _normalize("Brian Robinson Jr.") == "brian robinson"


def test_apostrophe_variants_resolve_to_same_player():
    for typed in ("Ja'Marr Chase", "Jamarr Chase", "JaMarr Chase", "Ja’Marr Chase"):
        r = resolve_player(typed, index=INDEX)
        assert r["status"] == "resolved" and r["player_id"] == "7564", typed


def test_hyphen_and_space_forms_resolve():
    for typed in ("Jaxon Smith-Njigba", "Jaxon Smith Njigba"):
        r = resolve_player(typed, index=INDEX)
        assert r["status"] == "resolved" and r["player_id"] == "9488", typed


def test_one_letter_misspelling_resolves_via_fuzzy():
    # Kaedon's spelling: "Jaxson" for the stored "Jaxon".
    r = resolve_player("Jaxson Smith Njigba", index=INDEX)
    assert r["status"] == "resolved" and r["player_id"] == "9488"
    r = resolve_player("Jaxson Smith-Njigba", index=INDEX)
    assert r["status"] == "resolved" and r["player_id"] == "9488"


def test_fuzzy_does_not_invent_matches():
    assert resolve_player("Jaxson Smith", index=INDEX)["status"] == "missing"
    assert resolve_player("Justin Jeffersen", index=INDEX)["status"] in ("resolved", "ambiguous")
    assert resolve_player("Zebra Unicorn", index=INDEX)["status"] == "missing"


def test_squashed_token_boundary_resolves():
    r = resolve_player("De Vonta Smith", index=INDEX)
    assert r["status"] == "resolved" and r["player_id"] == "102"


def test_idp_namesake_never_triggers_clarification_with_footprint():
    # The live defect: WR Justin Jefferson vs LB Justin Jefferson. With
    # the value-table footprint naming only the WR, the name resolves
    # silently to the WR.
    r = resolve_player("Justin Jefferson", index=INDEX, footprint_ids={"6794"})
    assert r["status"] == "resolved"
    assert r["player_id"] == "6794"
    assert r["player"]["position"] == "WR"


def test_namesake_without_footprint_info_stays_ambiguous():
    # No footprint data at all: nothing distinguishes the namesakes.
    r = resolve_player("Justin Jefferson", index=INDEX, footprint_ids=set())
    assert r["status"] == "ambiguous"
    assert {c["player_id"] for c in r["candidates"]} == {"6794", "13524"}


def test_two_plausible_namesakes_stay_ambiguous_with_plausible_only():
    # Both Chris Smiths are fantasy-pool players with footprints: the
    # clarification survives, and it offers the plausible pair.
    r = resolve_player("Chris Smith", index=INDEX, footprint_ids={"100", "101"})
    assert r["status"] == "ambiguous"
    assert {c["player_id"] for c in r["candidates"]} == {"100", "101"}


def test_genuine_ambiguity_no_footprint_still_clarifies():
    r = resolve_player("Chris Smith", index=INDEX, footprint_ids=set())
    assert r["status"] == "ambiguous"
    assert len(r["candidates"]) == 2


def test_roster_tiebreak_still_wins():
    r = resolve_player("Chris Smith", index=INDEX, footprint_ids=set(),
                       roster_player_ids={"101"})
    assert r["status"] == "resolved" and r["player_id"] == "101"
