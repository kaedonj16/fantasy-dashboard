"""Season Hub playoff-odds display: never round an undecided value up.

Regression coverage for the Hub tile reading "100%" for a team whose sim
odds were 99.9%. The simulator caps undecided odds at one decimal
(99.9 / 0.1) and reserves exact 100 / 0 for clinched / eliminated; the
Hub used to throw that away with integer rounding in three places
(``_playoff_tile_from_cache``, the dashboard client fill, and the
trade-window card). These tests exec the REAL helpers out of app.py so
they track the shipped code, plus source contracts for the JS mirror
and the trade-window odds line.
"""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
DASH_PY = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(
    encoding="utf-8"
)


def _load_helpers():
    # The formatter and the tile builder sit back to back in app.py, right
    # before the share-card cache block.
    start = APP_PY.index("def _fmt_playoff_pct_display")
    end = APP_PY.index("# Cache for share card HTML", start)
    ns = {"math": math}
    exec(APP_PY[start:end], ns)
    return ns["_fmt_playoff_pct_display"], ns["_playoff_tile_from_cache"]


FMT, TILE = _load_helpers()


# ---------------------------------------------------------------------------
# _fmt_playoff_pct_display
# ---------------------------------------------------------------------------

def test_fmt_undecided_values_read_at_one_decimal():
    assert FMT(99.9) == "99.9"
    assert FMT(0.1) == "0.1"
    assert FMT(42.24) == "42.2"
    assert FMT(50) == "50.0"
    assert FMT(7.5) == "7.5"


def test_fmt_never_rounds_up():
    # Defensive inputs past the simulator's own 99.9 cap: formatting
    # truncates to tenths instead of rounding.
    assert FMT(99.99) == "99.9"
    assert FMT(99.95) == "99.9"
    assert FMT(42.25) == "42.2"  # round-half-up would give 42.3
    assert FMT(42.29) == "42.2"


def test_fmt_exact_values_only_for_settled_odds():
    assert FMT(100) == "100"
    assert FMT(100.0) == "100"
    assert FMT(0) == "0"
    assert FMT(0.0) == "0"


# ---------------------------------------------------------------------------
# _playoff_tile_from_cache
# ---------------------------------------------------------------------------

def test_tile_99_9_never_displays_100():
    pct, sub = TILE(
        [{"roster_id": 1, "playoff_pct": 99.9, "first_seed_pct": 40.0, "bye_pct": 0}], 1
    )
    assert pct == "99.9"
    assert sub == "40.0% top seed"


def test_tile_defensive_99_99_truncates_to_99_9():
    assert TILE([{"roster_id": 1, "playoff_pct": 99.99}], 1) == (
        "99.9",
        "to make the playoffs",
    )


def test_tile_floor_of_range_reads_0_1():
    assert TILE([{"roster_id": 1, "playoff_pct": 0.1}], 1) == (
        "0.1",
        "to make the playoffs",
    )


def test_tile_exact_100_and_0_are_settled_statuses():
    assert TILE([{"roster_id": 1, "playoff_pct": 100, "is_complete": True}], 1) == (
        "100",
        "Clinched",
    )
    assert TILE([{"roster_id": 1, "playoff_pct": 0, "is_complete": True}], 1) == (
        "0",
        "Eliminated",
    )


def test_tile_status_uses_raw_value_not_display():
    # A completed season with a mid raw value is neither clinched nor
    # eliminated, whatever the display string looks like.
    assert TILE([{"roster_id": 1, "playoff_pct": 55, "is_complete": True}], 1) == (
        "55.0",
        "Playoff bound",
    )


def test_tile_42_24_floors_to_42_2():
    pct, _sub = TILE([{"roster_id": 1, "playoff_pct": 42.24}], 1)
    assert pct == "42.2"


def test_tile_first_seed_subtitle_uses_formatter():
    assert TILE(
        [{"roster_id": 1, "playoff_pct": 99.9, "first_seed_pct": 99.9, "bye_pct": 88.8}],
        1,
    ) == ("99.9", "99.9% top seed")


def test_tile_bye_subtitle_uses_formatter():
    assert TILE(
        [{"roster_id": 1, "playoff_pct": 95.0, "first_seed_pct": 0, "bye_pct": 99.9}],
        1,
    ) == ("95.0", "99.9% first-round bye")


def test_tile_sub_value_truncating_to_zero_falls_through():
    # 0.04% top seed displays as 0.0, so the subtitle must skip it and use
    # the bye line instead of printing "0.0% top seed".
    assert TILE(
        [{"roster_id": 1, "playoff_pct": 50.0, "first_seed_pct": 0.04, "bye_pct": 12.34}],
        1,
    ) == ("50.0", "12.3% first-round bye")
    assert TILE(
        [{"roster_id": 1, "playoff_pct": 50.0, "first_seed_pct": 0.04, "bye_pct": 0.02}],
        1,
    ) == ("50.0", "to make the playoffs")


def test_tile_missing_roster_returns_none():
    assert TILE([{"roster_id": 1, "playoff_pct": 20}], 9) is None
    assert TILE([], 1) is None
    assert TILE([{"roster_id": 1, "playoff_pct": 20}], None) is None
    assert TILE([{"roster_id": 1, "playoff_pct": "n/a"}], 1) is None


def test_tile_projected_offseason_variant():
    rows = [
        {
            "roster_id": 3,
            "playoff_pct": 87.65,
            "first_seed_pct": 12.34,
            "is_projected": True,
        }
    ]
    assert TILE(rows, 3, projected=True) == ("87.6", "Projected · 12.3% top seed")
    # A non-projection row never fills the offseason tile.
    assert TILE([{"roster_id": 3, "playoff_pct": 87.65}], 3, projected=True) is None
    # A top-seed chance that truncates to 0.0 falls back to the plain line.
    rows[0]["first_seed_pct"] = 0.04
    assert TILE(rows, 3, projected=True) == ("87.6", "Projected from current rosters")


# ---------------------------------------------------------------------------
# Source contracts: no integer rounding left on any Hub odds surface
# ---------------------------------------------------------------------------

def test_tile_helper_has_no_integer_rounding():
    start = APP_PY.index("def _fmt_playoff_pct_display")
    end = APP_PY.index("# Cache for share card HTML", start)
    helper_src = APP_PY[start:end]
    assert "int(round(" not in helper_src
    assert "_fmt_playoff_pct_display" in helper_src


def test_trade_window_card_uses_display_formatter():
    start = APP_PY.index("def _next_steps_trade_window_action")
    end = APP_PY.index("\ndef ", start + 1)
    card_src = APP_PY[start:end]
    assert "{pct:.0f}" not in card_src
    assert "_fmt_playoff_pct_display(pct)" in card_src
    assert "playoff odds" in card_src


def test_dashboard_client_fill_mirrors_the_formatter():
    # No Math.round on any of the three odds fields in the tile fill.
    assert "Math.round(row.playoff_pct" not in DASH_PY
    assert "Math.round(row.first_seed_pct" not in DASH_PY
    assert "Math.round(row.bye_pct" not in DASH_PY
    # The floor-to-tenths JS mirror is present, brCountUp gets the floored
    # value with dp picked by settled-ness, and status reads the raw pct.
    assert "function poFloorTenths" in DASH_PY
    assert "function poFmtPct" in DASH_PY
    assert "Math.floor(raw * 10 + 1e-9) / 10" in DASH_PY
    assert "dp: settled ? 0 : 1" in DASH_PY
    assert "rawPct >= 100 ? 'Clinched'" in DASH_PY
