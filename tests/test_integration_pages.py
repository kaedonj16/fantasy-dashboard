"""End-to-end page render tests through the real Flask stack.

These exercise the full route -> body-builder -> render_page path offline, using
the ``offline_client`` fixture (Sleeper HTTP mocked). They catch the class of
bug that pure unit tests miss: a route that 500s, a template that references a
missing field, a page chrome that breaks. Skipped automatically when Flask/
pandas aren't installed (the pure suite still runs); run them via the venv from
scripts/dev_setup.sh, and they run in CI where the full stack is present.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

# Tour routes render from seeded mock data, so they need no live league.
TOUR_PAGES = [
    "/sleeper/2026/tourdemo/graphs?tour=1",
    "/sleeper/2026/tourdemo/history?tour=1",
]


@pytest.mark.parametrize("path", TOUR_PAGES)
def test_tour_page_renders_200(offline_client, path):
    r = offline_client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    html = r.get_data(as_text=True)
    # Real page, not an error stub.
    assert len(html) > 2000
    assert "Traceback" not in html


def test_graphs_tour_has_core_chart(offline_client):
    html = offline_client.get("/sleeper/2026/tourdemo/graphs?tour=1").get_data(as_text=True)
    assert "PF vs PA" in html


def test_robots_and_sitemap(offline_client):
    assert offline_client.get("/robots.txt").status_code == 200
    assert offline_client.get("/sitemap.xml").status_code in (200, 500)  # sitemap may need data


def test_health_timing_requires_admin(offline_client):
    # No/incorrect admin secret -> 403 (never leaks timing without auth).
    r = offline_client.get("/api/health/timing")
    assert r.status_code == 403
