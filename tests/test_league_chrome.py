"""Unit tests for the persistent league + week chrome labels."""
from pathlib import Path

from utils.league_chrome import (
    build_league_chrome,
    fields_from_provider_league,
    format_label,
    merge_chrome_sources,
    week_label,
)

_PAGES = Path(__file__).resolve().parents[1] / "dashboard_services" / "pages"


def test_format_label_includes_size_and_qb_type():
    assert format_label(12, True) == "12tm SF"
    assert format_label(10, False) == "10tm 1QB"
    assert format_label(0, False) == "1QB"
    assert format_label(1, True) == "SF"


def test_week_label_is_hidden_for_all_season_states():
    assert week_label(14) == ""
    assert week_label(0) == ""
    assert week_label(3, offseason=True) == ""
    assert week_label(2, season_type="pre") == ""
    assert week_label(0, season_type="pre") == ""
    assert week_label(3, season_type="pre", offseason=True) == ""
    assert week_label(18, season_type="off") == ""


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
    assert meta["week_label"] == ""
    assert meta["sf"] is True


def test_build_chrome_fallback_name_and_1qb():
    meta = build_league_chrome(name="", size=10, roster_positions=["QB", "RB", "WR"], week=1)
    assert meta["name"] == "This league"
    assert meta["raw_name"] == ""
    assert meta["format"] == "10tm 1QB"
    assert meta["week_label"] == ""


def test_build_chrome_omits_format_when_slots_unknown():
    meta = build_league_chrome(name="", size=0)
    assert meta["name"] == "This league"
    assert meta["format"] == ""
    assert meta["sf"] is False


def test_merge_chrome_uses_sleeper_league_name_and_superflex():
    meta = merge_chrome_sources(
        ctx={},
        saved_name="",
        provider_league={
            "name": "KC fantasy-yearly",
            "total_rosters": 8,
            "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX", "BN"],
            "settings": {"slots_qb": 1},
        },
        week=1,
    )
    assert meta["name"] == "KC fantasy-yearly"
    assert meta["raw_name"] == "KC fantasy-yearly"
    assert meta["format"] == "8tm SF"
    assert meta["sf"] is True
    assert meta["size"] == 8


def test_merge_chrome_detects_sf_from_settings_slots():
    meta = merge_chrome_sources(
        ctx={"league": {"name": "Deep Ball"}},
        provider_league={"settings": {"slots_super_flex": 1, "num_teams": 12}},
    )
    assert meta["name"] == "Deep Ball"
    assert meta["format"] == "12tm SF"
    assert meta["sf"] is True


def test_merge_chrome_uses_saved_name_when_cache_and_provider_empty():
    meta = merge_chrome_sources(ctx={}, saved_name="KC fantasy-yearly", provider_league=None)
    assert meta["name"] == "KC fantasy-yearly"
    assert meta["format"] == ""


def test_fields_from_provider_league_reads_sleeper_payload():
    fields = fields_from_provider_league({
        "name": "  KC fantasy-yearly  ",
        "total_rosters": 8,
        "roster_positions": ["QB", "SUPER_FLEX"],
    })
    assert fields["name"] == "KC fantasy-yearly"
    assert fields["size"] == 8
    assert fields["is_sf"] is True
    assert fields["has_format"] is True


def test_league_chrome_meta_falls_back_to_live_sleeper():
    src = Path(__file__).resolve().parents[1] / "app.py"
    text = src.read_text(encoding="utf-8")
    start = text.index("def _league_chrome_meta")
    block = text[start:start + 1800]
    assert "merge_chrome_sources" in block
    assert "_provider_league_for_chrome" in block
    assert "_saved_league_chrome_name" in block


def test_hub_page_titles_do_not_restate_week():
    """Season/Offseason Hub H1s stay just the hub name."""
    src = (
        (_PAGES / "dashboard_page.py").read_text(encoding="utf-8")
        + (_PAGES / "offseason_dashboard_page.py").read_text(encoding="utf-8")
    )
    assert '<h1 class="os-hero-title">Season Hub</h1>' in src
    assert '<h1 class="os-hero-title">Offseason Hub</h1>' in src
    assert "Viewing {season}" not in src
    assert "Viewing {html.escape(str(season))}" not in src
    assert 'class="os-hero-kicker"' not in src
