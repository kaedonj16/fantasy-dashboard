from __future__ import annotations

from utils.cross_league_actions import make_action
from utils.digest_sections import league_overview_card_html, matchup_one_liner
from utils.weekly_email import (
    MAX_DIGEST_LEAGUES,
    build_multi_league_digest,
    choose_multi_league_subject,
    compact_league_blurb,
    connected_leagues_for_account,
    cross_league_digest_html,
    multi_league_sections_html,
    other_leagues_for_account,
)


def test_compact_league_blurb_minimal(monkeypatch):
    monkeypatch.setattr(
        "utils.weekly_email._canonical_standing",
        lambda *a, **k: (3, 5, 2),
    )
    html = compact_league_blurb(
        platform="espn", season=2025, league_id="99",
        roster_id="1", league_name="Second League",
        base_url="https://brfantasy.com",
    )
    assert "Second League" in html
    assert "#3" in html
    assert "5-2" in html
    assert "/espn/2025/99/dashboard" in html


def test_compact_league_blurb_empty_league_id():
    assert compact_league_blurb(platform="sleeper", season=2025, league_id="") == ""


def test_other_leagues_skips_primary(monkeypatch):
    monkeypatch.setattr(
        "dashboard_services.accounts.list_user_leagues",
        lambda aid: [
            {"platform": "sleeper", "league_id": "A", "season": 2025, "name": "Primary", "team_id": "1"},
            {"platform": "espn", "league_id": "B", "season": 2025, "name": "Other", "team_id": "2"},
            {"platform": "yahoo", "league_id": "C", "season": 2025, "name": "Third", "team_id": "3"},
        ],
    )

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k):
            class R:
                def fetchone(self): return None
            return R()

    monkeypatch.setattr("dashboard_services.db.get_conn", lambda: _Conn())
    rows = other_leagues_for_account(
        7, primary_platform="sleeper", primary_league_id="A", primary_season=2025, limit=2,
    )
    assert [r["league_id"] for r in rows] == ["B", "C"]
    assert all(r["league_id"] != "A" for r in rows)


def test_multi_league_sections_empty_when_no_others(monkeypatch):
    monkeypatch.setattr("utils.weekly_email.other_leagues_for_account", lambda *a, **k: [])
    assert multi_league_sections_html(
        1, primary_platform="sleeper", primary_league_id="x", primary_season=2025,
    ) == ""


def test_multi_league_sections_header(monkeypatch):
    monkeypatch.setattr(
        "utils.weekly_email.other_leagues_for_account",
        lambda *a, **k: [{"platform": "espn", "league_id": "9", "season": 2025, "roster_id": "", "name": "Alt"}],
    )
    monkeypatch.setattr(
        "utils.weekly_email.compact_league_blurb",
        lambda **kw: f"<div>{kw['league_name']}</div>",
    )
    html = multi_league_sections_html(
        1, primary_platform="sleeper", primary_league_id="x", primary_season=2025,
    )
    assert "Your other leagues" in html
    assert "Alt" in html


def test_cross_league_digest_html_ranks_and_formats():
    actions = [
        make_action(
            kind="injury", platform="espn", season=2025, league_id="2",
            league_name="Beta", title="Stash: X", detail="Approx return ~3 wk",
        ),
        make_action(
            kind="lineup", platform="sleeper", season=2025, league_id="1",
            league_name="Alpha", title="Empty starting slot", detail="QB open",
            severity=1.0,
        ),
    ]
    html = cross_league_digest_html(actions, base_url="https://brfantasy.com", limit=3)
    assert "This week's moves" in html
    assert "Empty starting slot" in html
    assert "Alpha" in html
    # Lineup ranks above injury — empty slot appears first.
    assert html.index("Empty starting slot") < html.index("Stash: X")
    assert "https://brfantasy.com/sleeper/2025/1/waivers" in html
    assert " — " not in html
    assert "—" not in html
    assert "–" not in html


def test_multi_league_includes_cross_league_actions(monkeypatch):
    monkeypatch.setattr("utils.weekly_email.other_leagues_for_account", lambda *a, **k: [])
    actions = [
        make_action(
            kind="lineup", platform="espn", season=2025, league_id="9",
            league_name="Alt League", title="Starter on bye", severity=0.7,
        ),
    ]
    html = multi_league_sections_html(
        1, primary_platform="sleeper", primary_league_id="x", primary_season=2025,
        base_url="https://brfantasy.com",
        actions=actions,
    )
    assert "This week's moves" in html
    assert "Starter on bye" in html
    assert "Alt League" in html


def test_connected_leagues_primary_first_then_others(monkeypatch):
    monkeypatch.setattr(
        "utils.weekly_email.other_leagues_for_account",
        lambda *a, **k: [
            {"platform": "espn", "league_id": "B", "season": 2026, "roster_id": "2", "name": "The Gridiron"},
            {"platform": "yahoo", "league_id": "C", "season": 2026, "roster_id": "3", "name": "Yahoo H2H"},
        ],
    )
    rows = connected_leagues_for_account(
        7,
        primary_platform="sleeper",
        primary_league_id="A",
        primary_season=2026,
        primary_roster_id="1",
        primary_name="blackedraw",
        limit=8,
    )
    assert [r["league_id"] for r in rows] == ["A", "B", "C"]
    assert rows[0]["name"] == "blackedraw"
    assert MAX_DIGEST_LEAGUES >= 3


def test_connected_leagues_respects_cap(monkeypatch):
    extras = [
        {"platform": "espn", "league_id": str(i), "season": 2026, "roster_id": "1", "name": f"L{i}"}
        for i in range(12)
    ]
    monkeypatch.setattr("utils.weekly_email.other_leagues_for_account", lambda *a, **k: extras[: k.get("limit", 99)])
    rows = connected_leagues_for_account(
        1, primary_platform="sleeper", primary_league_id="A", primary_season=2026, limit=4,
    )
    assert len(rows) == 4
    assert rows[0]["league_id"] == "A"


def test_league_overview_card_includes_standing_matchup_and_waiver():
    html = league_overview_card_html(
        league_name="blackedraw",
        format_label="1QB · Dynasty",
        rank=3, wins=5, losses=2,
        dash_url="https://brfantasy.com/sleeper/2026/A/dashboard",
        matchup={"opponent_name": "Gridiron FC", "user_proj": 120.4, "opp_proj": 110.1, "win_prob": 0.62},
        waiver={"name": "Gabe Davis", "pos": "WR", "reason": "Available"},
        top_asset={"name": "Jeremiyah Love", "pos": "RB", "value": 773},
    )
    assert "blackedraw" in html
    assert "1QB · Dynasty" in html
    assert "#3" in html
    assert "5-2" in html
    assert "Gridiron FC" in html
    assert "Gabe Davis" in html
    assert "Jeremiyah Love" in html
    assert "Open league" in html
    assert "/sleeper/2026/A/dashboard" in html
    assert "—" not in html
    assert "–" not in html


def test_matchup_one_liner_omits_empty():
    assert matchup_one_liner(None) == ""
    assert matchup_one_liner({}) == ""
    line = matchup_one_liner({"opponent_name": "Them", "win_prob": 0.4})
    assert "vs Them" in line
    assert "40%" in line


def test_choose_multi_league_subject_prefers_lineup():
    snaps = [
        {
            "league_name": "blackedraw",
            "fmt": {"is_dynasty": True},
            "rank": 3, "wins": 5, "losses": 2,
            "lineup_note": {"title": "Start/Sit · empty slot", "body": "1 empty starting slot"},
            "matchup": None, "waivers": [], "my_risers": [], "pidx": {},
        },
        {
            "league_name": "The Gridiron",
            "fmt": {"is_dynasty": False},
            "rank": 1, "wins": 6, "losses": 1,
            "lineup_note": None,
            "matchup": {"win_prob": 0.7},
            "waivers": [{"name": "Waive Me"}],
            "my_risers": [], "pidx": {},
        },
    ]
    assert choose_multi_league_subject(snaps, 2) == "blackedraw: Fix your lineup before Sunday"


def test_choose_multi_league_subject_portfolio_fallback():
    snaps = [
        {"league_name": "A", "fmt": {}, "rank": None, "wins": 0, "losses": 0,
         "lineup_note": None, "matchup": None, "waivers": [], "my_risers": [], "pidx": {}},
    ]
    assert choose_multi_league_subject(snaps, 3) == "Your 3 leagues this week"


def _league_bundle(name, roster_id, players, settings, fmt, wins=3, losses=1):
    roster = {
        "roster_id": roster_id,
        "players": players,
        "settings": {"wins": wins, "losses": losses},
    }
    return {
        "league": {"name": name, "settings": settings, "roster_positions": ["QB", "RB", "WR", "TE", "FLEX"]},
        "rosters": [roster],
        "format": fmt,
        "uid_name": {},
        "owned_ids": set(players),
        "roster_by_id": {str(roster_id): roster},
        "matchups": [],
        "week": 4,
    }


def test_build_multi_league_digest_covers_every_connected_league(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    bundles = {
        "A": _league_bundle(
            "blackedraw", "1", ["4046"], {"type": 2},
            {"is_dynasty": True, "is_superflex": False, "type": "dynasty", "is_redraft": False},
            wins=5, losses=2,
        ),
        "B": _league_bundle(
            "The Gridiron", "2", ["6794"], {"type": 0},
            {"is_dynasty": False, "is_superflex": False, "type": "redraft", "is_redraft": True},
            wins=6, losses=1,
        ),
        "C": _league_bundle(
            "Yahoo H2H-Pts 1307110", "3", ["1"], {"type": 2},
            {"is_dynasty": True, "is_superflex": False, "type": "dynasty", "is_redraft": False},
            wins=2, losses=5,
        ),
    }
    standings = {
        "A": (3, 5, 2),
        "B": (1, 6, 1),
        "C": (8, 2, 5),
    }

    def _standing(plat, lid, season, rid):
        return standings.get(str(lid), (None, 0, 0))

    def _actions(**kw):
        lid = str(kw.get("league_id") or "")
        if lid == "B":
            return [{"kind": "waiver", "targets": [
                {"name": "Gabe Davis", "pos": "WR", "reason": "Available"},
            ]}]
        if lid == "C":
            return [{"kind": "lineup", "title": "Start/Sit · empty slot",
                     "body": "1 empty starting slot"}]
        return []

    from utils.digest_context import DigestRunCache
    cache = DigestRunCache()
    cache.nfl_state = {"season_type": "reg", "week": 4, "season": 2026}
    cache.league_bundle = lambda plat, season, lid: bundles.get(str(lid))

    monkeypatch.setattr("utils.digest_context.DigestRunCache.load_shared", lambda self: None)
    monkeypatch.setattr("utils.weekly_email._canonical_standing", _standing)
    monkeypatch.setattr("utils.digest_actions.gather_digest_action_items", _actions)
    monkeypatch.setattr(
        "utils.digest_context.matchup_for_roster",
        lambda bundle, rid, cache: (
            {"opponent_name": "Rival", "user_proj": 118.0, "opp_proj": 109.0, "win_prob": 0.61}
            if (bundle.get("league") or {}).get("name") == "The Gridiron" else None
        ),
    )
    out = build_multi_league_digest(
        [
            {"platform": "sleeper", "league_id": "A", "season": 2026, "roster_id": "1", "name": "blackedraw"},
            {"platform": "espn", "league_id": "B", "season": 2026, "roster_id": "2", "name": "The Gridiron"},
            {"platform": "yahoo", "league_id": "C", "season": 2026, "roster_id": "3", "name": "Yahoo H2H-Pts 1307110"},
        ],
        first_name="Kaedon",
        run_cache=cache,
    )

    assert out is not None
    html = out["html"]
    assert "Hey Kaedon" in html
    assert "3 connected league" in html
    assert "Your leagues" in html
    assert "Your other leagues" not in html
    assert "blackedraw" in html
    assert "The Gridiron" in html
    assert "Yahoo H2H-Pts 1307110" in html
    assert "#3" in html and "5-2" in html
    assert "#1" in html and "6-1" in html
    assert "Gabe Davis" in html
    assert "Rival" in html
    assert "empty starting slot" in html
    assert "Open your leagues" in html
    assert "https://brfantasy.com/portfolio" in html
    assert "sleeper/2026/A/dashboard" in html
    assert "espn/2026/B/dashboard" in html
    assert "yahoo/2026/C/dashboard" in html
    assert "—" not in html
    assert "–" not in html
    assert out["league_count"] == 3
    assert "multi-league" in out["tags"]
    assert "weekly-digest" in out["tags"]
