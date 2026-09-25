"""SEO canonical normalization: canonical tags, og:url, and sitemap locs
must advertise the www origin (the site serves on www; apex 301s to www)."""
from __future__ import annotations

import pytest

pytest.importorskip("flask")


@pytest.fixture
def www_domain(monkeypatch):
    monkeypatch.setenv("PRIMARY_DOMAIN", "brfantasyfootball.com")
    import app as appmod
    import importlib
    # PRIMARY_DOMAIN / _WWW_HOST are read at app import; recompute for the test env.
    monkeypatch.setattr(appmod, "PRIMARY_DOMAIN", "brfantasyfootball.com")
    monkeypatch.setattr(appmod, "_WWW_HOST", "www.brfantasyfootball.com")
    return appmod


def test_seo_origin_is_www(www_domain):
    from routes.public_bp import _seo_origin
    with www_domain.app.test_request_context("/"):
        assert _seo_origin() == "https://www.brfantasyfootball.com"


def test_seo_origin_www_input_not_doubled(www_domain, monkeypatch):
    monkeypatch.setenv("PRIMARY_DOMAIN", "www.brfantasyfootball.com")
    from routes.public_bp import _seo_origin
    with www_domain.app.test_request_context("/"):
        assert _seo_origin() == "https://www.brfantasyfootball.com"


def test_site_origin_is_www(www_domain):
    with www_domain.app.test_request_context("/"):
        assert www_domain._site_origin() == "https://www.brfantasyfootball.com"


def test_sitemap_locs_use_www(offline_client, www_domain):
    resp = offline_client.get("/sitemap.xml")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "<loc>https://www.brfantasyfootball.com/" in body
    assert "<loc>https://brfantasyfootball.com/" not in body


def test_robots_sitemap_line_uses_www(offline_client, www_domain):
    resp = offline_client.get("/robots.txt")
    assert resp.status_code == 200
    assert "Sitemap: https://www.brfantasyfootball.com/sitemap.xml" in resp.get_data(as_text=True)


def test_nfl_teams_has_h1_and_absolute_canonical(offline_client, www_domain):
    resp = offline_client.get("/nfl-teams")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "<h1>NFL Team Rankings</h1>" in body
    assert '<link rel="canonical" href="https://www.brfantasyfootball.com/nfl-teams">' in body


def test_relative_canonical_absolutized(www_domain):
    tags = www_domain._build_seo_meta_tags("d", "/dynasty-trade-value-chart", False, False)
    assert 'href="https://www.brfantasyfootball.com/dynasty-trade-value-chart"' in tags


def test_default_canonical_uses_www(www_domain):
    with www_domain.app.test_request_context("/trade"):
        tags = www_domain._build_seo_meta_tags("d", None, False, False)
    assert 'href="https://www.brfantasyfootball.com/trade"' in tags


def test_player_page_og_set(offline_client, www_domain):
    idx = www_domain.get_player_slug_index() or {}
    assert idx, "player slug index empty"
    slug = next(iter(idx))
    resp = offline_client.get(f"/player/{slug}", follow_redirects=True)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert f"<meta property='og:url' content='https://www.brfantasyfootball.com/player/{slug}'>" in body
    assert "<meta name='twitter:card' content='summary_large_image'>" in body


def test_default_social_tags_og_url_www(www_domain):
    with www_domain.app.test_request_context("/trade"):
        tags = www_domain._default_social_tags("T", "D")
    assert 'property="og:url" content="https://www.brfantasyfootball.com/trade"' in tags
    assert 'property="og:image" content="https://www.brfantasyfootball.com/static/og-default.png' in tags
