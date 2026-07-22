"""Unit tests for the shared market-ADP resolver (dashboard_services.adp_service).

adp_service imports lightly (network calls are lazy), so these run in the pure
suite with the source fetchers monkeypatched.
"""
import sys

from dashboard_services import adp_service as A


# ── consensus_adp ────────────────────────────────────────────────────────────

def test_consensus_single_source_kept_raw():
    assert A.consensus_adp([{"a": 3.0, "b": 20.0}]) == {"a": 3.0, "b": 20.0}
    assert A.consensus_adp([{}, {"a": 5.0}]) == {"a": 5.0}
    assert A.consensus_adp([]) == {}
    assert A.consensus_adp([{}, {}]) == {}


def test_consensus_blends_by_rank():
    # Sources disagree on a vs b; consensus averages their ranks.
    s = {"a": 1.0, "b": 2.0, "c": 3.0}   # ranks a1 b2 c3
    y = {"a": 2.0, "b": 1.0, "c": 3.0}   # ranks a2 b1 c3
    c = A.consensus_adp([s, y])
    assert c["a"] == 1.5 and c["b"] == 1.5 and c["c"] == 3.0


def test_consensus_scale_invariant():
    # Second source on a wildly different numeric scale must not dominate.
    s = {"a": 1.0, "b": 2.0}
    big = {"a": 500.0, "b": 100.0}       # ranks b1 a2
    c = A.consensus_adp([s, big])
    assert c["a"] == c["b"] == 1.5        # a(1,2) and b(2,1) both average 1.5


# ── resolve_market_adp ───────────────────────────────────────────────────────

def test_resolve_sleeper_field_by_format(monkeypatch):
    monkeypatch.setattr(A, "fetch_sleeper_adp", lambda season: {"1": {"adp_ppr": 3.0, "adp_2qb": 2.0}})
    assert A.resolve_market_adp(2026, False, "redraft", "sleeper") == {"1": 3.0}   # 1QB -> adp_ppr
    assert A.resolve_market_adp(2026, True, "redraft", "sleeper") == {"1": 2.0}    # SF  -> adp_2qb


def test_resolve_dynasty_uses_dynasty_field(monkeypatch):
    monkeypatch.setattr(A, "fetch_sleeper_adp", lambda season: {"1": {"adp_ppr": 3.0, "adp_dynasty_ppr": 8.0}})
    assert A.resolve_market_adp(2026, False, "dynasty", "sleeper") == {"1": 8.0}


def test_resolve_yahoo_is_redraft_only_and_falls_back(monkeypatch):
    monkeypatch.setattr(A, "fetch_sleeper_adp", lambda season: {"1": {"adp_dynasty_ppr": 8.0}})
    monkeypatch.setattr(A, "fetch_fc_startup_adp", lambda is_sf: {})
    # yahoo requested on the dynasty axis is invalid -> resolver falls back to sleeper dynasty.
    assert A.resolve_market_adp(2026, False, "dynasty", "yahoo") == {"1": 8.0}


def test_resolve_consensus_blends_sleeper_and_fc(monkeypatch):
    monkeypatch.setattr(A, "fetch_sleeper_adp", lambda season: {"a": {"adp_ppr": 1.0}, "b": {"adp_ppr": 2.0}})
    monkeypatch.setattr(A, "fetch_fc_redraft_adp", lambda is_sf: {"a": {"avg_pick": 2.0}, "b": {"avg_pick": 1.0}})
    # No token, so the yahoo source contributes nothing.
    c = A.resolve_market_adp(2026, False, "redraft", "consensus")
    assert c["a"] == 1.5 and c["b"] == 1.5


def test_resolve_empty_when_no_sources(monkeypatch):
    monkeypatch.setattr(A, "fetch_sleeper_adp", lambda season: {})
    monkeypatch.setattr(A, "fetch_fc_redraft_adp", lambda is_sf: {})
    assert A.resolve_market_adp(2026, False, "redraft", "consensus") == {}


# ── ordinal_rank_adp (option 2 display transform) ────────────────────────────

def test_ordinal_rank_makes_contiguous_board():
    # Raw ADP that never hits 1.0 (crawler-style floor) -> clean 1,2,3.
    assert A.ordinal_rank_adp({"a": 3.3, "b": 5.1, "c": 4.0}) == {"a": 1.0, "c": 2.0, "b": 3.0}


def test_ordinal_rank_ties_break_by_id_stable():
    assert A.ordinal_rank_adp({"b": 2.0, "a": 2.0}) == {"a": 1.0, "b": 2.0}


def test_ordinal_rank_empty():
    assert A.ordinal_rank_adp({}) == {}


def test_resolve_as_rank_applies_ordinal(monkeypatch):
    monkeypatch.setattr(A, "fetch_sleeper_adp",
                        lambda season: {"a": {"adp_dynasty_ppr": 3.3}, "b": {"adp_dynasty_ppr": 9.9}})
    monkeypatch.setattr(A, "_crawler_adp_source", lambda *a, **k: {})
    monkeypatch.setattr(A, "fetch_fc_startup_adp", lambda is_sf: {})
    out = A.resolve_market_adp(2026, False, "dynasty", "sleeper", as_rank=True)
    assert out == {"a": 1.0, "b": 2.0}


# ── crawler source (size-combined dynasty/rookie ADP) ────────────────────────

class _FakeCur:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, handler):
        self._handler = handler

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params):
        return _FakeCur(self._handler(sql, params))


def _install_fake_db(monkeypatch, handler):
    import types as _t
    mod = _t.ModuleType("dashboard_services.db")
    mod.get_conn = lambda: _FakeConn(handler)
    monkeypatch.setitem(sys.modules, "dashboard_services.db", mod)


def test_crawler_redraft_yields_nothing():
    # crawler only sees startup/rookie drafts; redraft is off-axis.
    assert A.fetch_crawler_adp(2026, False, "redraft") == {}


def test_crawler_scales_norm_round_to_ref_size(monkeypatch):
    # SQL returns size-normalized round position; wrapper rescales to a 12-team pick.
    def handler(sql, params):
        return [{"player_id": "1", "norm_round": 0.5, "n": 30},   # half a round in -> pick 6
                {"player_id": "2", "norm_round": 2.0, "n": 30}]   # two rounds in -> pick 24
    _install_fake_db(monkeypatch, handler)
    assert A.fetch_crawler_adp(2026, False, "dynasty") == {"1": 6.0, "2": 24.0}


def test_crawler_falls_back_to_latest_season(monkeypatch):
    calls = {"seasons": []}

    def handler(sql, params):
        if "MAX(season)" in sql:
            return [{"s": 2025}]
        season = params[1]
        calls["seasons"].append(season)
        if season == 2026:
            return []                       # requested season empty
        return [{"player_id": "9", "norm_round": 1.0, "n": 50}]   # latest season has data
    _install_fake_db(monkeypatch, handler)
    out = A.fetch_crawler_adp(2026, True, "rookie")
    assert out == {"9": 12.0}
    assert calls["seasons"] == [2026, 2025]   # tried requested, then fell back


def test_resolve_dynasty_consensus_includes_crawler(monkeypatch):
    monkeypatch.setattr(A, "fetch_sleeper_adp", lambda season: {"a": {"adp_dynasty_ppr": 1.0},
                                                                "b": {"adp_dynasty_ppr": 2.0}})
    monkeypatch.setattr(A, "fetch_fc_startup_adp", lambda is_sf: {})
    monkeypatch.setattr(A, "_crawler_adp_source", lambda season, is_sf, st: {"a": 24.0, "b": 6.0})
    # sleeper says a<b, crawler says b<a -> consensus rank-average is a tie at 1.5.
    c = A.resolve_market_adp(2026, False, "dynasty", "consensus")
    assert c["a"] == 1.5 and c["b"] == 1.5
