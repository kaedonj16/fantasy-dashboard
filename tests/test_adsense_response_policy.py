"""Behavioral contracts for ownership, response-level ads, and crawl metadata."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


PUBLISHER = "9164153092633845"


def test_initial_public_html_owns_site_and_ids_match(offline_client):
    response = offline_client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert f'<meta name="google-adsense-account" content="ca-pub-{PUBLISHER}">' in html
    assert f'data-ad-client="ca-pub-{PUBLISHER}"' in html

    ads = offline_client.get("/ads.txt")
    assert ads.status_code == 200
    assert ads.mimetype == "text/plain"
    assert f"google.com, pub-{PUBLISHER}, DIRECT, f08c47fec0942fa0" in ads.get_data(as_text=True)


def test_substantive_page_is_eligible_but_premium_session_is_not(offline_client, monkeypatch):
    eligible = offline_client.get("/guides/dynasty-trade-value").get_data(as_text=True)
    assert 'data-ad-eligible="true"' in eligible
    assert "ins class=\"adsbygoogle\"" in eligible

    import app
    monkeypatch.setattr(app, "has_premium_for_viewer", lambda *args, **kwargs: True)
    premium = offline_client.get("/guides/dynasty-trade-value").get_data(as_text=True)
    assert 'data-ad-eligible="false"' in premium
    assert "adsbygoogle" not in premium


def test_missing_trade_is_noindex_and_ad_free(offline_client):
    response = offline_client.get("/t/definitely-missing-share")
    assert response.status_code == 404
    html = response.get_data(as_text=True)
    assert 'name="robots" content="noindex, follow"' in html
    assert 'data-ad-eligible="false"' in html
    assert "adsbygoogle" not in html


def test_unknown_player_is_real_ad_free_404(offline_client, monkeypatch):
    import routes.seo_pages_bp as seo
    monkeypatch.setattr(seo, "get_player_slug_index", lambda: {})
    response = offline_client.get("/player/not-a-real-player/trade-value")
    assert response.status_code == 404
    html = response.get_data(as_text=True)
    assert "Player not found" in html
    assert 'name="robots" content="noindex, follow"' in html
    assert "adsbygoogle" not in html


def test_player_provider_failure_is_retryable_and_ad_free(offline_client, monkeypatch):
    import routes.seo_pages_bp as seo
    monkeypatch.setattr(seo, "get_player_slug_index", lambda: {"test-player": "p1"})
    monkeypatch.setattr(seo, "get_nfl_state", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    response = offline_client.get("/player/test-player/trade-value")
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "300"
    html = response.get_data(as_text=True)
    assert "temporarily unavailable" in html
    assert 'data-ad-eligible="false"' in html
    assert "adsbygoogle" not in html


def test_soft_navigation_carries_response_eligibility(offline_client):
    eligible = offline_client.get(
        "/guides/dynasty-trade-value", headers={"X-Soft-Nav": "1"}
    ).get_data(as_text=True)
    ineligible = offline_client.get(
        "/privacy", headers={"X-Soft-Nav": "1"}
    ).get_data(as_text=True)
    assert 'data-ad-eligible="true"' in eligible
    assert 'data-ad-eligible="false"' in ineligible
    # Swap payloads never initialize or duplicate slots themselves.
    assert "adsbygoogle" not in eligible
    assert "adsbygoogle" not in ineligible

    source = __import__("pathlib").Path("static/app.js").read_text(encoding="utf-8")
    assert "ad eligibility requires full navigation" in source
    assert "document.querySelectorAll('.ad-container')" in source


def test_empty_rankings_are_retryable_and_ad_free(offline_client, monkeypatch):
    import routes.seo_pages_bp as seo
    monkeypatch.setattr(seo, "get_model_value_table_cached", lambda: [])
    response = offline_client.get("/rankings/dynasty")
    assert response.status_code == 503
    html = response.get_data(as_text=True)
    assert "temporarily unavailable" in html
    assert 'name="robots" content="noindex, follow"' in html
    assert "adsbygoogle" not in html


def test_player_ppg_labels_basis_and_preserves_missing_as_na():
    from dashboard_services.pages.player_page import build_player_page_body

    common = dict(
        player_id="p1", name="Test Player", position="WR", team="BUF", age=25,
        headshot=None, value_1qb=5000, sf_value=4900, pos_rank_label="WR12",
        ovr_rank=24, sf_pos_rank_label="WR13", sf_ovr_rank=28,
        value_history=[], season=2026, similar_players=[],
    )
    available = build_player_page_body(ppg=17.25, **common)
    missing = build_player_page_body(ppg=None, **common)
    assert "Full-PPR PPG" in available
    assert "2026 completed appearances" in available
    assert "17.2" in available
    assert "Full-PPR PPG" in missing
    assert ">N/A<" in missing


def test_canonical_robots_and_sitemap_share_production_origin(offline_client, monkeypatch):
    import app
    import routes.public_bp as public
    monkeypatch.setattr(app, "PRIMARY_DOMAIN", "brfantasyfootball.com")
    monkeypatch.setattr(app, "_WWW_HOST", "www.brfantasyfootball.com")
    monkeypatch.setenv("PRIMARY_DOMAIN", "brfantasyfootball.com")

    home = offline_client.get("/").get_data(as_text=True)
    canonical = re.search(r'<link rel="canonical" href="([^"]+)">', home).group(1)
    robots = offline_client.get("/robots.txt").get_data(as_text=True)
    sitemap_response = offline_client.get("/sitemap.xml")
    root = ET.fromstring(sitemap_response.get_data(as_text=True))
    locations = [node.text for node in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]

    assert canonical == "https://www.brfantasyfootball.com/"
    assert "Sitemap: https://www.brfantasyfootball.com/sitemap.xml" in robots
    assert locations
    assert all(url.startswith("https://www.brfantasyfootball.com/") or url == "https://www.brfantasyfootball.com" for url in locations)


def test_missing_trade_outcome_is_noindex_and_ad_free(offline_client):
    # /o/<id> (short URL) 404s through render_page like the /t/ trade shares.
    response = offline_client.get("/o/definitely-missing-share")
    assert response.status_code == 404
    html = response.get_data(as_text=True)
    assert 'name="robots" content="noindex, follow"' in html
    assert 'data-ad-eligible="false"' in html
    assert "adsbygoogle" not in html

    # The standalone card itself carries a noindex meta tag (raw HTML page).
    card = offline_client.get("/trade-outcome-card/definitely-missing-share")
    assert card.status_code == 404
    card_html = card.get_data(as_text=True)
    assert 'name="robots" content="noindex"' in card_html
