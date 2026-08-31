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
         mock.patch.object(we, "_canonical_standing", return_value=(2, 3, 1)):
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
