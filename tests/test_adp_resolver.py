"""Unit tests for the shared market-ADP resolver (dashboard_services.adp_service).

adp_service imports lightly (network calls are lazy), so these run in the pure
suite with the source fetchers monkeypatched.
"""
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
