"""Pick pricing unification: every surface prices picks from the one
canonical WLS table (dashboard_services.picks.load_pick_value_table).

Regression contracts:
- trade_intel analytics no longer uses its private hardcoded table
  (2027 1st mid at 650, SF at a flat x1.5, unknown combos at an
  invented 10): values come from the canonical table per format axis,
  and an unpriceable pick is 0.0, never a fabricated constant.
- the Activity page no longer fishes early/late prices for a pick
  whose seed bucket is unknown: mid is the only default bucket, and a
  pick the table can't price that way renders as "No current value".
"""
from __future__ import annotations

import os
import sys
import types

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")


@pytest.fixture
def analytics(monkeypatch):
    """Import trade_intel analytics with dashboard_services.db stubbed
    when the psycopg driver is unavailable (same pattern as
    test_trade_analytics_streaming; stub scoped via monkeypatch)."""
    if "dashboard_services.db" not in sys.modules:
        try:
            import psycopg  # noqa: F401
        except Exception:
            _stub = types.ModuleType("dashboard_services.db")
            _stub.get_conn = lambda *a, **k: None
            if "dashboard_services" not in sys.modules:
                _pkg = types.ModuleType("dashboard_services")
                _pkg.__path__ = [os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "dashboard_services")]
                monkeypatch.setitem(sys.modules, "dashboard_services", _pkg)
            monkeypatch.setitem(sys.modules, "dashboard_services.db", _stub)
    from data_building.trade_intel import analytics as _analytics
    return _analytics


def test_analytics_pick_value_uses_canonical_table(analytics, monkeypatch):
    tables = {
        "1qb": {"2027_1_mid": 301.1},
        "sf": {"2027_1_mid": 303.4},
    }
    monkeypatch.setattr(analytics, "_pick_table", lambda fmt: tables["sf" if fmt == "sf" else "1qb"])
    asset = {"pick_season": 2027, "pick_round": 1, "pick_order": "mid"}
    # Canonical table values, not the old hardcoded 650 / 650*1.5.
    assert analytics._pick_value(asset, "1qb") == 301.1
    assert analytics._pick_value(asset, "sf") == 303.4


def test_analytics_pick_value_unpriced_is_zero_never_invented(analytics, monkeypatch):
    monkeypatch.setattr(analytics, "_pick_table", lambda fmt: {})
    asset = {"pick_season": 2027, "pick_round": 3, "pick_order": "weird"}
    # The old code returned a fabricated 10 for combos outside its table.
    assert analytics._pick_value(asset, "1qb") == 0.0
    assert analytics._pick_value(asset, "sf") == 0.0


def test_analytics_pick_value_season_falls_back_to_trade_season(analytics, monkeypatch):
    monkeypatch.setattr(analytics, "_pick_table", lambda fmt: {"2026_2_mid": 120.0})
    asset = {"pick_round": 2, "pick_order": "mid"}
    assert analytics._pick_value(asset, "1qb", season=2026) == 120.0
    assert analytics._pick_value(asset, "1qb") == 0.0


def _render_activity(monkeypatch, pick_values, trades):
    import app as appmod

    monkeypatch.setattr(appmod, "ensure_activity_bits", lambda ctx: None)
    monkeypatch.setattr(appmod, "get_users", lambda *a, **k: [])
    monkeypatch.setattr(
        appmod, "load_pick_value_table", lambda **_kw: dict(pick_values)
    )
    monkeypatch.setattr(appmod, "resolve_exact_pick_slot", lambda *a, **k: None)

    from dashboard_services.pages.activity_page import build_activity_body

    rows = [
        {
            "kind": "trade", "week": 1,
            "ts": pd.Timestamp("2026-08-12", tz="UTC"),
            "data": {"teams": teams, "draft_picks": picks},
        }
        for teams, picks in trades
    ]
    ctx = {
        "league_id": "L1",
        "resolved_league_id": "L1",
        "platform": "sleeper",
        "season": 2026,
        "activity_df": pd.DataFrame(rows),
        "injury_df": pd.DataFrame(),
        "standings_map": {1: 1, 2: 2},
        "model_value_table": [
            {"id": "8146", "name": "Garrett Wilson", "search_name": "garrett wilson",
             "position": "WR", "team": "NYJ", "value": 517.4, "pos_rank_label": "WR10"},
            {"id": "11631", "name": "Brian Thomas Jr.", "search_name": "brian thomas",
             "position": "WR", "team": "JAX", "value": 167.6, "pos_rank_label": "WR43"},
        ],
        "scoring_settings": {},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        "rosters": [{"roster_id": i} for i in range(1, 11)],
        "users": [],
    }
    return build_activity_body(ctx)


def test_activity_pick_without_seed_never_fishes_other_buckets(monkeypatch):
    wilson = {"name": "Garrett Wilson", "pos": "WR", "team": "NYJ", "pid": "8146"}
    thomas = {"name": "Brian Thomas Jr.", "pos": "WR", "team": "JAX", "pid": "11631"}
    teams = [
        {"roster_id": 1, "name": "Team One", "avatar": "", "gets": [wilson], "sends": [thomas]},
        {"roster_id": 2, "name": "Team Two", "avatar": "", "gets": [thomas], "sends": [wilson]},
    ]
    # previous_owner_id 99 is not in the standings map, so no seed bucket
    # exists. The table prices ONLY the early bucket for this pick.
    pick = {"season": 2027, "round": 1, "owner_id": 1, "previous_owner_id": 99, "roster_id": 1}
    html = _render_activity(monkeypatch, {"2027_1_early": 300.0}, [(teams, [pick])])
    # Old behavior fished the early price (300.0) for an unknown-bucket
    # pick; the unified rule prices mid-or-nothing, so the pick is
    # honestly shown as unvalued.
    assert "No current value" in html
    assert "300.0" not in html
