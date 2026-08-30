"""Unit tests for utils.redzone_user (cross-platform My Leagues portfolio)."""
from utils.redzone_user import (
    match_viewer_roster,
    portfolio_from_account_leagues,
    portfolio_from_sleeper_leagues,
)


def test_match_viewer_prefers_stored_team_id():
    rosters = [
        {"roster_id": 1, "owner_id": "u-alice"},
        {"roster_id": 7, "owner_id": "u-bob"},
    ]
    hit = match_viewer_roster(rosters, team_id="7", owner_id="u-alice")
    assert hit["roster_id"] == 7


def test_match_viewer_falls_back_to_owner_id():
    rosters = [
        {"roster_id": 1, "owner_id": "u-alice"},
        {"roster_id": 2, "owner_id": "u-bob"},
    ]
    hit = match_viewer_roster(rosters, team_id="", owner_id="u-bob")
    assert hit["roster_id"] == 2


def test_match_viewer_miss_returns_none():
    assert match_viewer_roster([{"roster_id": 1, "owner_id": "x"}], team_id="9") is None


def test_match_viewer_team_id_can_be_an_owner_id():
    rosters = [
        {"roster_id": 1, "owner_id": "u-alice"},
        {"roster_id": 7, "owner_id": "{SW-OWNER}"},
    ]
    hit = match_viewer_roster(rosters, team_id="{SW-OWNER}")
    assert hit["roster_id"] == 7


def test_match_viewer_owner_ids_select_the_platform_identity():
    rosters = [
        {"roster_id": 1, "owner_id": "sleeper-user"},
        {"roster_id": 2, "owner_id": "{ESPN-SWID}"},
    ]
    assert match_viewer_roster(rosters, owner_ids=["sleeper-user"])["roster_id"] == 1
    assert match_viewer_roster(rosters, owner_ids=["{ESPN-SWID}"])["roster_id"] == 2


def test_swid_brace_variants_match():
    from utils.redzone_user import owner_id_variants
    raw = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert "{" + raw + "}" in owner_id_variants(raw)
    assert raw in owner_id_variants("{" + raw + "}")


def test_account_portfolio_is_cross_platform_and_capped():
    saved = [
        {"platform": "espn", "league_id": "111", "season": 2025, "team_id": "3", "name": "ESPN Keepers"},
        {"platform": "sleeper", "league_id": "aaa", "season": 2026, "team_id": "1", "name": "Sleeper Dyno"},
        {"platform": "yahoo", "league_id": "222", "season": 2026, "team_id": "4", "name": "Yahoo Redraft"},
        {"platform": "espn", "league_id": "111", "season": 2026, "team_id": "3", "name": "dup"},
    ]
    out = portfolio_from_account_leagues(saved, season=2026)
    assert [x["platform"] for x in out] == ["espn", "sleeper", "yahoo"]
    # ESPN saved season rolls forward to the current year.
    assert out[0]["season"] == 2026
    assert out[0]["team_id"] == "3"


def test_sleeper_portfolio_normalizes_name_and_platform():
    raw = [{"league_id": "L1", "name": "Alpha"}, {"league_id": "L2"}]
    out = portfolio_from_sleeper_leagues(raw, season=2026)
    assert out[0] == {
        "platform": "sleeper", "league_id": "L1", "name": "Alpha",
        "season": 2026, "team_id": "",
    }
    assert out[1]["name"] == "League 2"


def test_resolve_portfolio_viewer_ignores_espn_session_owner_on_sleeper():
    """Viewing Redzone from an ESPN league leaves an ESPN SWID in session.

    That id must not be used to match Sleeper rosters (portfolio contract), or
    My Leagues collapses to only the current ESPN league.
    """
    from utils.redzone_user import resolve_portfolio_viewer_roster

    sleeper_rosters = [
        {"roster_id": 1, "owner_id": "sleeper-alice"},
        {"roster_id": 2, "owner_id": "sleeper-bob"},
    ]
    # ESPN SWID in session, no stored team — must miss, not invent a hit.
    assert resolve_portfolio_viewer_roster(
        sleeper_rosters,
        platform="sleeper",
        team_id="",
        session_owner_id="{ESPN-SWID-AAAA}",
    ) is None

    # Stored team_id still wins.
    hit = resolve_portfolio_viewer_roster(
        sleeper_rosters,
        platform="sleeper",
        team_id="2",
        session_owner_id="{ESPN-SWID-AAAA}",
    )
    assert hit["roster_id"] == 2

    # Account-resolved roster id wins even without team_id.
    hit = resolve_portfolio_viewer_roster(
        sleeper_rosters,
        platform="sleeper",
        team_id="",
        session_owner_id="{ESPN-SWID-AAAA}",
        account_roster_id="1",
    )
    assert hit["roster_id"] == 1


def test_resolve_portfolio_viewer_espn_uses_account_owner_ids_not_sleeper_session():
    from utils.redzone_user import resolve_portfolio_viewer_roster

    espn_rosters = [
        {"roster_id": 3, "owner_id": "{ESPN-SWID-AAAA}"},
        {"roster_id": 9, "owner_id": "{OTHER}"},
    ]
    # Sleeper session owner must not match ESPN rosters.
    assert resolve_portfolio_viewer_roster(
        espn_rosters,
        platform="espn",
        team_id="",
        session_owner_id="sleeper-alice",
    ) is None

    hit = resolve_portfolio_viewer_roster(
        espn_rosters,
        platform="espn",
        team_id="",
        session_owner_id="sleeper-alice",
        account_owner_ids=["ESPN-SWID-AAAA"],  # brace variants expand
    )
    assert hit["roster_id"] == 3


def test_redzone_fetch_user_uses_portfolio_viewer_resolver():
    """Source contract: My Leagues must not pass session owner into every platform."""
    from pathlib import Path
    src = Path("app.py").read_text(encoding="utf-8")
    fn = src[src.index("def _redzone_fetch_user"): src.index("def page_redzone")]
    assert "resolve_portfolio_viewer_roster" in fn
    assert "resolve_account_viewer_for_league" in fn
    # Old bug: match_viewer_roster(..., owner_id=viewer_uid) for every platform.
    assert "match_viewer_roster(\n            d[\"rosters\"], team_id=lg.get(\"team_id\"), owner_id=viewer_uid" not in fn
    assert "session_owner_id=viewer_uid" in fn
