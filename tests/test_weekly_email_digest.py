"""Weekly digest deep links (R12.1) and subject preview."""
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


def test_build_digest_links_mover_rows(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    movers = {
        "risers": [{"player_id": "4046", "delta": 12.0}],
        "fallers": [{"player_id": "6794", "delta": -8.0}],
    }
    pidx = {
        "4046": {"full_name": "Justin Jefferson"},
        "6794": {"full_name": "Ja'Marr Chase"},
    }
    rosters = [{"roster_id": "7", "players": ["4046", "6794"], "settings": {"wins": 3, "losses": 1}}]

    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league",
                    return_value={"name": "Test League"}), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(2, 3, 1)), \
         mock.patch("utils.digest_actions.gather_digest_actions", return_value=[]):
        out = we.build_digest(
            "sleeper", "L1", 2026, "7",
            first_name="Sam", movers=movers, pidx=pidx,
        )

    assert out is not None
    html = out["html"]
    assert 'href="https://brfantasy.com/sleeper/2026/L1/dashboard?player=4046' in html
    assert "Justin%20Jefferson" in html or "Justin Jefferson" in html
    assert 'href="https://brfantasy.com/sleeper/2026/L1/dashboard?player=6794' in html
    assert "{UNSUB}" in html
    assert "Justin Jefferson" in out["subject"]
    assert "▲12" in out["subject"] or "▲ 12" in out["subject"] or "▲12" in out["subject"].replace(" ", "")
    assert out["subject"] == "Test League: #2 · Justin Jefferson ▲12"


def test_build_digest_subject_fallback_without_risers(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    movers = {
        "risers": [],
        "fallers": [{"player_id": "6794", "delta": -8.0}],
    }
    pidx = {"6794": {"full_name": "Ja'Marr Chase"}}
    rosters = [{"roster_id": "7", "players": ["6794"], "settings": {"wins": 1, "losses": 2}}]

    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league",
                    return_value={"name": "Fall League"}), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(5, 1, 2)), \
         mock.patch("utils.digest_actions.gather_digest_actions", return_value=[]):
        out = we.build_digest(
            "sleeper", "L1", 2026, "7", movers=movers, pidx=pidx,
        )

    assert out is not None
    assert out["subject"] == "Fall League: your weekly dynasty digest"


def test_build_digest_appends_action_sections(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasy.com")
    movers = {"risers": [{"player_id": "1", "delta": 5.0}], "fallers": []}
    pidx = {"1": {"full_name": "A"}}
    rosters = [{"roster_id": "7", "players": ["1"]}]
    section = '<div class="act">Waiver wire</div>'

    with mock.patch("dashboard_services.platform_api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.platform_api.get_league",
                    return_value={"name": "L"}), \
         mock.patch("dashboard_services.platform_api.get_users", return_value=[]), \
         mock.patch.object(we, "_canonical_standing", return_value=(None, 0, 0)), \
         mock.patch("utils.digest_actions.gather_digest_actions", return_value=[section]):
        out = we.build_digest("sleeper", "L1", 2026, "999", movers=movers, pidx=pidx)

    assert out is not None
    assert "Waiver wire" in out["html"]
    assert out["subject"] == "L: A ▲5"
