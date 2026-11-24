"""SEO shell tests for /prospects (site improvement, 2026-11-24).

The board relaunched with model 3.7 grades but was invisible to search:
absent from the sitemap, a bare "Prospect Rankings" title with the
site-wide boilerplate description, and zero server-rendered prospect
names in the served HTML. These tests pin the fix: the sitemap entry,
the class-year title and prospect-specific description, and the
server-rendered top 50 rows (same markup and enriched fields the
client board produces from /api/prospects/rankings).
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _html(client, path):
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    return r.get_data(as_text=True)


def test_prospects_in_sitemap(offline_client):
    xml = _html(offline_client, "/sitemap.xml")
    assert "/prospects</loc>" in xml


def test_prospects_title_and_meta(offline_client):
    from data_building.rookie_pipeline.pipeline import get_active_rookie_class

    year = get_active_rookie_class()
    html = _html(offline_client, "/prospects")
    assert f"<title>{year} NFL Draft Prospect Rankings | BR Fantasy</title>" in html
    assert f"{year} NFL Draft prospect rankings with grades" in html
    # The bare title and the generic boilerplate no longer describe the page.
    assert "<title>Prospect Rankings</title>" not in html


# ── Server-rendered rows ────────────────────────────────────────────────────

def _fixture_rows(n=55):
    """Board-shaped rows (the enriched fields rookie_api attaches)."""
    rows = []
    for i in range(1, n + 1):
        score = round(95.0 - i * 0.7, 2)
        band = "elite" if score >= 92 else "great" if score >= 80 else \
            "average" if score >= 70 else "day3"
        rows.append({
            "player_id": f"FIX_{i}",
            "name": f"Fixture Player {i}",
            "position": "WR",
            "position_rank": i,
            "school": "Fixture State",
            "overall_rank": i,
            "rank_delta": 0,
            "prospect_score": score,
            "percentiles": {"production": 90, "efficiency": 80,
                            "breakout": None},
            "tag": {"label": "x", "band": band},
        })
    rows[0]["rank_delta"] = 3
    rows[1]["rank_delta"] = -2
    return rows


@pytest.fixture()
def ssr_rows(monkeypatch):
    import dashboard_services.rookie_api as rookie_api
    import dashboard_services.pages.rookies_page as rookies_page

    rows = _fixture_rows()
    monkeypatch.setattr(rookie_api, "get_board_rows", lambda year: rows)
    rookies_page._ssr_cache.clear()
    yield rows
    rookies_page._ssr_cache.clear()


def test_ssr_renders_top_50_with_names_and_grades(ssr_rows):
    from dashboard_services.pages.rookies_page import build_prospects_body

    body = build_prospects_body(False)
    # Top of the board is in the served HTML with its grade (one decimal,
    # the client's toFixed(1) format).
    assert "Fixture Player 1" in body
    assert '<span class="pr-grade">94.3</span>' in body
    assert "Fixture Player 50" in body
    # The cap holds: row 51 waits for the client fetch.
    assert "Fixture Player 51" not in body
    assert body.count('data-pid="FIX_') == 50
    # Header and count render in their post-load states; skeleton hidden.
    assert 'id="rkTableHeader" class="pr-grid-row">' in body
    assert 'id="rkCount">Showing 50 of 55 prospects</div>' in body
    assert 'id="rkLoading" class="sk-list" style="display:none;margin-top:4px;"' in body


def test_ssr_row_markup_matches_client_shape(ssr_rows):
    from dashboard_services.pages.rookies_page import _rk_board_row_html

    row = _rk_board_row_html(ssr_rows[0])
    assert 'data-pid="FIX_1"' in row
    assert '<span class="pr-rank">#1' in row
    # Movement triangle for a +3 delta, client entity and wording.
    assert 'pr-rank-delta up' in row and "&#9650;3" in row
    assert '<span class="pos-badge WR">WR</span>' in row
    assert "WR1 &middot; Fixture State" in row
    # Percentiles print as plain numbers; a missing one is a muted dash.
    assert '<span class="rk-pct ">90</span>' in row
    assert '<span class="rk-pct pr-eff">80</span>' in row
    assert "rk-pct-dash" in row
    # No score renders the muted dash grade cell, never a placeholder.
    bare = dict(ssr_rows[0], prospect_score=None)
    assert '<span class="pr-grade none">-</span>' in _rk_board_row_html(bare)


def test_ssr_band_dividers_follow_band_changes(ssr_rows):
    from dashboard_services.pages.rookies_page import _prospects_ssr

    html, shown, total = _prospects_ssr()
    assert (shown, total) == (50, 55)
    # First divider (Elite) precedes the first row, like the client board.
    assert html.index("pr-tier-divider") < html.index("Fixture Player 1")
    assert "Elite" in html and "Great" in html


def test_ssr_fails_soft_to_the_plain_shell(monkeypatch):
    import dashboard_services.rookie_api as rookie_api
    import dashboard_services.pages.rookies_page as rookies_page

    def _boom(year):
        raise RuntimeError("db down")

    monkeypatch.setattr(rookie_api, "get_board_rows", _boom)
    rookies_page._ssr_cache.clear()
    body = rookies_page.build_prospects_body(False)
    rookies_page._ssr_cache.clear()
    # Exactly the pre-SSR shell: empty list, skeleton visible, header hidden.
    assert '<div id="rkList"></div>' in body
    assert 'id="rkLoading" class="sk-list" style="margin-top:4px;"' in body
    assert 'id="rkTableHeader" class="pr-grid-row" style="display:none">' in body
