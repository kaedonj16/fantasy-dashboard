"""Unit tests for the persistent league + week chrome labels."""
from pathlib import Path

from utils.league_chrome import build_league_chrome, format_label, week_label

_PAGES = Path(__file__).resolve().parents[1] / "dashboard_services" / "pages"


def test_format_label_includes_size_and_qb_type():
    assert format_label(12, True) == "12tm SF"
    assert format_label(10, False) == "10tm 1QB"
    assert format_label(0, False) == "1QB"
    assert format_label(1, True) == "SF"


def test_week_label_regular_and_offseason():
    assert week_label(14) == "Week 14"
    assert week_label(0) == ""
    assert week_label(3, offseason=True) == "Offseason"
    assert week_label(2, season_type="pre") == "Preseason · Wk 2"
    assert week_label(0, season_type="pre") == "Preseason"
    assert week_label(18, season_type="off") == "Offseason"


def test_build_chrome_superflex_from_slots():
    meta = build_league_chrome(
        name="  Dynasty Warriors  ",
        size=12,
        roster_positions=["QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX"],
        week=14,
        season_type="regular",
    )
    assert meta["name"] == "Dynasty Warriors"
    assert meta["raw_name"] == "Dynasty Warriors"
    assert meta["format"] == "12tm SF"
    assert meta["week_label"] == "Week 14"
    assert meta["sf"] is True


def test_build_chrome_fallback_name_and_1qb():
    meta = build_league_chrome(name="", size=10, roster_positions=["QB", "RB", "WR"], week=1)
    assert meta["name"] == "This league"
    assert meta["raw_name"] == ""
    assert meta["format"] == "10tm 1QB"
    assert meta["week_label"] == "Week 1"


def test_hub_page_titles_do_not_restate_week():
    """Season/Offseason Hub H1s stay just the hub name; week lives in chrome."""
    src = (
        (_PAGES / "dashboard_page.py").read_text(encoding="utf-8")
        + (_PAGES / "offseason_dashboard_page.py").read_text(encoding="utf-8")
    )
    assert '<h1 class="os-hero-title">Season Hub</h1>' in src
    assert '<h1 class="os-hero-title">Offseason Hub</h1>' in src
    assert "Viewing {season}" not in src
    assert "Viewing {html.escape(str(season))}" not in src
    assert 'class="os-hero-kicker"' not in src
