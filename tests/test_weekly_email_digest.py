"""Weekly digest deep links, format-aware subjects, and section rendering."""
from __future__ import annotations

from unittest import mock

import utils.weekly_email as we


def test_player_deep_link_opens_dashboard_modal():
    url = we.player_deep_link(
        "https://brfantasy.com", "sleeper", 2026, "L1", "4046", "Justin Jefferson",
    )
    assert url.startswith("https://brfantasy.com/sleeper/2026/L1/dashboard?")
    assert "player=4046" in url
    assert "player_name=Justin%20Jefferson" in url


def _digest_mocks(rosters, league, standing=(2, 3, 1)):
    return mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
        mock.patch("dashboard_services.platform_api.get_league", return_value=league), \
        mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
        mock.patch.object(we, "_canonical_standing", return_value=standing), \
        mock.patch("utils.digest_actions.gather_digest_action_items", return_value=[]), \
        mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None)


def test_build_digest_links_mover_rows(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    movers = {
        "risers": [{"player_id": "4046", "delta": 120.0}],
        "fallers": [{"player_id": "6794", "delta": -80.0}],
    }
    pidx = {
        "4046": {"full_name": "Justin Jefferson"},
        "6794": {"full_name": "Ja'Marr Chase"},
    }
    rosters = [{"roster_id": "7", "players": ["4046", "6794"], "settings": {"wins": 3, "losses": 1}}]
    league = {"name": "Test League", "settings": {"type": 2}}

    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league", return_value=league), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(2, 3, 1)), \
         mock.patch("utils.digest_actions.gather_digest_action_items", return_value=[]), \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("utils.digest_context.DigestRunCache.league_bundle", return_value=None):
        out = we.build_digest(
            "sleeper", "L1", 2026, "7",
            first_name="Sam", movers=movers, pidx=pidx,
        )

    assert out is not None
    html = out["html"]
    assert 'href="https://brfantasy.com/sleeper/2026/L1/dashboard?player=4046' in html
    assert "Justin Jefferson" in html
    assert 'href="https://brfantasy.com/sleeper/2026/L1/dashboard?player=6794' in html
    assert "{UNSUB}" in html
    assert "Test League" in html
    assert "Justin Jefferson" in out["subject"]
    assert out["subject"] == "Test League: #2 · Justin Jefferson ▲120"


def test_build_digest_subject_fallback_without_risers(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    movers = {
        "risers": [],
        "fallers": [{"player_id": "6794", "delta": -8.0}],
    }
    pidx = {"6794": {"full_name": "Ja'Marr Chase"}}
    rosters = [{"roster_id": "7", "players": ["6794"], "settings": {"wins": 1, "losses": 2}}]
    league = {"name": "Fall League", "settings": {"type": 2}}

    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league", return_value=league), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(5, 1, 2)), \
         mock.patch("utils.digest_actions.gather_digest_action_items", return_value=[]), \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("utils.digest_context.DigestRunCache.league_bundle", return_value=None):
        out = we.build_digest(
            "sleeper", "L1", 2026, "7", movers=movers, pidx=pidx,
        )

    assert out is not None
    assert "dynasty digest" not in out["subject"].lower() or "weekly" in out["subject"].lower()
    assert out["subject"] in (
        "Fall League: #5 · 1-2",
        "Fall League: your weekly fantasy digest",
        "Fall League: your weekly dynasty digest",
    )
    assert "Fall League:" in out["subject"]


def test_build_digest_appends_action_sections(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    movers = {"risers": [{"player_id": "1", "delta": 80.0}], "fallers": []}
    pidx = {"1": {"full_name": "A"}}
    rosters = [{"roster_id": "7", "players": ["1"]}]
    section = {"kind": "waiver", "html": '<div class="act">Waiver wire</div>',
               "targets": [{"name": "Free Agent", "pos": "RB"}]}

    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league",
                    return_value={"name": "L", "settings": {"type": 2}}), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(None, 0, 0)), \
         mock.patch("utils.digest_actions.gather_digest_action_items", return_value=[section]), \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("utils.digest_context.DigestRunCache.league_bundle", return_value=None):
        out = we.build_digest("sleeper", "L1", 2026, "999", movers=movers, pidx=pidx)

    assert out is not None
    assert "Waiver wire" in out["html"] or "Free Agent" in out["html"]
    assert "▲" in out["subject"] or "waiver" in out["subject"].lower() or "A ▲" in out["subject"]


def test_redraft_digest_leads_with_matchup_not_dynasty_movers(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    movers = {"risers": [{"player_id": "4046", "delta": 400.0}], "fallers": []}
    pidx = {"4046": {"full_name": "Justin Jefferson"}}
    rosters = [{
        "roster_id": "7", "players": ["4046"], "starters": ["4046"],
        "settings": {"wins": 4, "losses": 1},
    }]
    league = {
        "name": "Redraft League",
        "settings": {"type": 0},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        "scoring_settings": {"rec": 1, "bonus_rec_te": 0},
    }
    matchup = {
        "opponent_name": "The Other Team",
        "user_proj": 118.4, "opp_proj": 109.1, "margin": 9.3, "win_prob": 0.62,
    }

    from utils.digest_context import DigestRunCache
    cache = DigestRunCache()
    cache.nfl_state = {"season_type": "reg", "week": 4, "season": 2026}

    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league", return_value=league), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(1, 4, 1)), \
         mock.patch("utils.digest_actions.gather_digest_action_items", return_value=[]), \
         mock.patch.object(type(cache), "load_shared", lambda self: None), \
         mock.patch.object(type(cache), "league_bundle", return_value=None), \
         mock.patch("utils.digest_context.matchup_for_roster", return_value=matchup):
        out = we.build_digest(
            "sleeper", "L1", 2026, "7", movers=movers, pidx=pidx, run_cache=cache,
        )

    assert out is not None
    assert out["format"]["is_redraft"] is True
    assert "This week's matchup" in out["html"]
    assert "The Other Team" in out["html"]
    assert "118.4" in out["html"]
    assert "Your risers this week" not in out["html"]
    assert "favored" in out["subject"].lower()
    assert "dynasty digest" not in out["html"].lower()


def test_dynasty_digest_keeps_movers_and_omits_empty_matchup(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    movers = {"risers": [{"player_id": "4046", "delta": 220.0}], "fallers": []}
    pidx = {"4046": {"full_name": "Justin Jefferson", "position": "WR"}}
    rosters = [{"roster_id": "7", "players": ["4046"], "settings": {"wins": 3, "losses": 1}}]
    league = {"name": "Dynasty Home", "settings": {"type": 2},
              "roster_positions": ["QB", "RB", "WR", "WR", "TE", "SUPER_FLEX"]}

    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league", return_value=league), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(2, 3, 1)), \
         mock.patch("utils.digest_actions.gather_digest_action_items", return_value=[]), \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("utils.digest_context.DigestRunCache.league_bundle", return_value=None), \
         mock.patch("utils.digest_context.in_season", return_value=False), \
         mock.patch("utils.digest_context.matchup_for_roster", return_value=None):
        out = we.build_digest("sleeper", "L1", 2026, "7", movers=movers, pidx=pidx)

    assert out is not None
    assert out["format"]["is_dynasty"] is True
    assert out["format"]["is_superflex"] is True
    assert "Your risers this week" in out["html"]
    assert "This week's matchup" not in out["html"]
    assert "Justin Jefferson" in out["html"]


def test_no_data_omits_digest(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    rosters = [{"roster_id": "7", "players": [], "settings": {}}]
    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league",
                    return_value={"name": "Empty", "settings": {"type": 2}}), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(None, 0, 0)), \
         mock.patch("utils.digest_actions.gather_digest_action_items", return_value=[]), \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("utils.digest_context.DigestRunCache.league_bundle", return_value=None), \
         mock.patch("utils.digest_context.in_season", return_value=False):
        out = we.build_digest("sleeper", "L1", 2026, "7", movers={"risers": [], "fallers": []}, pidx={})
    assert out is None


def test_keeper_digest_includes_value_movers_not_zero_record(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    movers = {"risers": [{"player_id": "4046", "delta": 180.0}], "fallers": []}
    pidx = {"4046": {"full_name": "Justin Jefferson", "position": "WR"}}
    rosters = [{"roster_id": "7", "players": ["4046"], "settings": {"wins": 0, "losses": 0}}]
    league = {"name": "BLITZ THE LEAGUE", "settings": {"type": 1},
              "roster_positions": ["QB", "RB", "WR", "WR", "TE", "FLEX"]}
    waivers = {"kind": "waiver", "targets": [
        {"player_id": "w1", "name": "Parker Washington", "pos": "WR", "reason": "Breakout"},
        {"player_id": "w2", "name": "Luther Burden III", "pos": "WR", "reason": "Breakout"},
        {"player_id": "w3", "name": "A Running Back", "pos": "RB", "reason": "RB need"},
    ]}

    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league", return_value=league), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(6, 0, 0)), \
         mock.patch("utils.digest_actions.gather_digest_action_items", return_value=[waivers]), \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("utils.digest_context.DigestRunCache.league_bundle", return_value=None), \
         mock.patch("utils.digest_context.in_season", return_value=False):
        out = we.build_digest("sleeper", "L1", 2026, "7", first_name="Kaedon", movers=movers, pidx=pidx)

    assert out is not None
    assert out["format"]["is_keeper"] is True
    html = out["html"]
    assert "BLITZ THE LEAGUE" in html
    assert "at 0-0" not in html
    assert "Your risers this week" in html
    assert "Justin Jefferson" in html
    assert "Parker Washington" in html
    assert "Luther Burden III" not in html
    assert "A Running Back" in html
    assert "Top waiver targets:" not in html
    assert out["subject"] != "BLITZ THE LEAGUE: #6 · 0-0"


def test_unsubscribe_token_roundtrip(monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-test-secret")
    token = we.make_unsub_token(42)
    assert we.verify_unsub_token(token) == 42
    assert we.verify_unsub_token("42.deadbeef") is None
    assert we.verify_unsub_token("nope") is None


def test_choose_subject_hierarchy_lineup_beats_waiver():
    subject = we.choose_subject(
        "League A", {"is_dynasty": False},
        rank=2, lineup_note={"title": "Start/Sit · empty slot", "body": "1 empty starting slot"},
        matchup={"win_prob": 0.7}, waivers=[{"name": "Waive Me"}],
    )
    assert subject == "League A: Fix your lineup before Sunday"


def test_preview_digest_writes_html(tmp_path, monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    dest = tmp_path / "preview.html"
    rosters = [{"roster_id": "1", "players": ["4046"], "settings": {"wins": 1, "losses": 0}}]
    league = {"name": "Preview LG", "settings": {"type": 2}}
    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league", return_value=league), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(3, 1, 0)), \
         mock.patch("utils.digest_actions.gather_digest_action_items", return_value=[]), \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("utils.digest_context.DigestRunCache.league_bundle", return_value=None):
        out = we.preview_digest(
            platform="sleeper", league_id="L1", season=2026, roster_id="1",
            first_name="Sam", out_path=str(dest),
        )
    assert out is not None
    assert dest.exists()
    html = dest.read_text()
    assert "#unsubscribe" in html
    assert "{UNSUB}" not in html
