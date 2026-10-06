"""Trade intel cards: 90d heat + buyer-split extras.

Covers the module-level helpers behind the reframed trade intel cards
(``trade_intel_card_extras`` in app.py): side classification, package labels,
velocity trend rules, and the best-effort contract (DB failure -> {}).

Skipped when Flask/pandas aren't installed; runs in CI with the full stack.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from datetime import datetime, timedelta, timezone

import app
import dashboard_services.db as db_module
import utils.data_cache as data_cache_module


# ---------------------------------------------------------------------------
# _ti_classify_side
# ---------------------------------------------------------------------------

def test_classify_side_contender():
    got = app._ti_classify_side({"wins": 9, "losses": 2, "ties": 0})
    assert got == {"record": "9-2", "class": "contender"}


def test_classify_side_rebuilder():
    got = app._ti_classify_side({"wins": 2, "losses": 8, "ties": 0})
    assert got == {"record": "2-8", "class": "rebuilder"}


def test_classify_side_mid_pack():
    got = app._ti_classify_side({"wins": 5, "losses": 5, "ties": 1})
    assert got["class"] == "mid-pack"
    assert got["record"] == "5-5-1"


def test_classify_side_none_without_games():
    assert app._ti_classify_side({"wins": 0, "losses": 0}) is None
    assert app._ti_classify_side(None) is None
    assert app._ti_classify_side("nope") is None


# ---------------------------------------------------------------------------
# _ti_package_label
# ---------------------------------------------------------------------------

def _values():
    return {
        "p1": {"position": "WR", "value": 640},   # tier 3 -> WR3
        "p2": {"position": "RB", "value": 900},   # tier 2 -> RB2
    }


def test_package_label_players_and_picks():
    assets = [
        {"asset_type": "player", "sent_player_id": "p1", "pick_round": None, "pick_season": None},
        {"asset_type": "pick", "sent_player_id": None, "pick_round": 1, "pick_season": "2026"},
    ]
    label = app._ti_package_label(assets, _values())
    assert label == "2026 1st + WR3"


def test_package_label_counts_duplicates():
    assets = [
        {"asset_type": "pick", "sent_player_id": None, "pick_round": 2, "pick_season": "2026"},
        {"asset_type": "pick", "sent_player_id": None, "pick_round": 2, "pick_season": "2026"},
    ]
    assert app._ti_package_label(assets, _values()) == "2× 2026 2nd"


def test_package_label_none_when_unlabelable():
    assert app._ti_package_label([], _values()) is None
    assert app._ti_package_label(
        [{"asset_type": "player", "sent_player_id": "unknown", "pick_round": None,
          "pick_season": None}], _values()) is None


# ---------------------------------------------------------------------------
# _ti_velocity_trend
# ---------------------------------------------------------------------------

def test_velocity_trend_rules():
    assert app._ti_velocity_trend(6, 2) == "heating_up"
    assert app._ti_velocity_trend(0, 5) == "cooling_off"
    assert app._ti_velocity_trend(4, 4) == "steady"
    assert app._ti_velocity_trend(2, 1) is None      # fewer than 4 total
    assert app._ti_velocity_trend(3, 0) is None      # fewer than 4 total
    assert app._ti_velocity_trend(0, 0) is None


# ---------------------------------------------------------------------------
# trade_intel_card_extras
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, _sql, _params=None):
        return _FakeResult(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, * _a):
        return False


def _ctx(w, l):
    return {"a": {"wins": w, "losses": l, "ties": 0},
            "b": {"wins": 12 - w, "losses": w, "ties": 0}}


def _row(pid, tid, days_ago, buyer_wins, sent):
    return {
        "focus_pid": pid,
        "trade_id": tid,
        "tctx": _ctx(buyer_wins, 11 - buyer_wins),
        "recv_side": "a",
        "tcreated": datetime.now(timezone.utc) - timedelta(days=days_ago),
        "asset_type": sent["kind"],
        "sent_pid": sent.get("pid"),
        "pick_round": sent.get("rnd"),
        "pick_season": sent.get("season"),
    }


def _run_extras(monkeypatch, rows):
    monkeypatch.setattr(db_module, "get_conn", lambda: _FakeConn(rows))
    monkeypatch.setattr(
        data_cache_module, "load_model_value_table",
        lambda *a, **k: [
            {"id": "p1", "position": "WR", "value": 640},
            {"id": "p2", "position": "RB", "value": 900},
        ],
    )
    # Reset the TTL cache so the monkeypatched table is picked up.
    app._TI_CARD_VALUES_CACHE["at"] = 0.0
    app._TI_CARD_VALUES_CACHE["by_id"] = {}
    return app.trade_intel_card_extras


def test_card_extras_db_failure_returns_empty(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(db_module, "get_conn", _boom)
    assert app.trade_intel_card_extras(["p1"]) == {}
    assert app.trade_intel_card_extras([]) == {}


def test_card_extras_velocity_and_buyer_split(monkeypatch):
    rows = []
    # 5 recent trades (10-80d ago), buyer 9-2 contender, sent 2026 1st each
    for i in range(5):
        rows.append(_row("p1", f"t{i}", 10 + i * 15, 9,
                         {"kind": "pick", "rnd": 1, "season": "2026"}))
    # 1 older trade (120d ago), buyer 2-9 rebuilder, sent WR p1
    rows.append(_row("p1", "t5", 120, 2,
                     {"kind": "player", "pid": "p1"}))
    extras = _run_extras(monkeypatch, rows)(["p1"])
    got = extras["p1"]
    assert got["velocity"]["recent_90d"] == 5
    assert got["velocity"]["prior_90d"] == 1
    assert got["velocity"]["trend"] == "heating_up"
    split = got["buyer_split"]
    assert split["contender"]["label"] == "2026 1st"
    assert split["contender"]["count"] == 5
    assert split["contender"]["total"] == 5
    assert split["rebuilder"]["label"] == "WR3"
    assert split["rebuilder"]["count"] == 1


def test_card_extras_skips_midpack_and_unlabelable(monkeypatch):
    rows = [
        # mid-pack buyer (6-5): excluded from the split but counts for velocity
        _row("p9", "m0", 10, 6, {"kind": "pick", "rnd": 1, "season": "2026"}),
        _row("p9", "m1", 20, 6, {"kind": "pick", "rnd": 1, "season": "2026"}),
        _row("p9", "m2", 30, 6, {"kind": "pick", "rnd": 1, "season": "2026"}),
        _row("p9", "m3", 40, 6, {"kind": "pick", "rnd": 1, "season": "2026"}),
    ]
    extras = _run_extras(monkeypatch, rows)(["p9"])
    got = extras["p9"]
    assert got["velocity"]["recent_90d"] == 4
    assert got["velocity"]["trend"] == "heating_up"
    assert got["buyer_split"] == {}
