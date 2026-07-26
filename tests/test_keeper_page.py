"""Tests for keeper_page data helpers: ADP fallback and draft selection.

keeper_page imports lightly (no pandas/flask at module load), so these run in
the pure unit suite.
"""
import sys
import types

from dashboard_services.pages import keeper_page as kp


def _fake_adp_service(resolve):
    m = types.ModuleType("dashboard_services.adp_service")
    m.resolve_market_adp = resolve
    return m


def test_adp_delegates_to_resolver_redraft(monkeypatch):
    # _adp_map is a thin redraft wrapper over the shared resolver; field/source
    # selection lives in test_adp_resolver.py. Here we only assert the delegation:
    # keeper decisions always ask the redraft axis and pass the requested source.
    seen = {}

    def resolve(season, is_sf, scoring_type="consensus", source="consensus"):
        seen.update(season=season, is_sf=is_sf, scoring_type=scoring_type, source=source)
        return {"1": 3.0, "2": 20.0}

    monkeypatch.setitem(sys.modules, "dashboard_services.adp_service", _fake_adp_service(resolve))
    assert kp._adp_map(is_sf=True, season=2026, source="sleeper") == {"1": 3.0, "2": 20.0}
    assert seen == {"season": 2026, "is_sf": True, "scoring_type": "redraft", "source": "sleeper"}


def test_adp_returns_empty_when_resolver_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setitem(sys.modules, "dashboard_services.adp_service", _fake_adp_service(boom))
    assert kp._adp_map(is_sf=False, season=2026) == {}


def test_value_rank_ranks_by_value_and_skips_zero():
    vr = kp._value_rank_map({"a": 1000.0, "b": 800.0, "c": 500.0, "d": 0.0})
    assert vr == {"a": 1.0, "b": 2.0, "c": 3.0}   # zero-value player omitted


def test_candidates_use_value_rank_when_adp_missing():
    vals = {"a": 1000.0, "b": 800.0}
    vr = kp._value_rank_map(vals)
    cands = kp._candidates_for_ids(
        ["a", "b"], {"a": {"name": "A", "pos": "WR"}}, vals, adp={}, drafted={}, value_rank=vr,
    )
    assert cands[0].adp_overall == 1.0 and cands[1].adp_overall == 2.0


def test_candidates_prefer_real_adp_over_rank():
    vals = {"a": 1000.0}
    vr = kp._value_rank_map(vals)   # a -> 1
    cands = kp._candidates_for_ids(["a"], {}, vals, adp={"a": 42.0}, drafted={}, value_rank=vr)
    assert cands[0].adp_overall == 42.0   # market ADP wins when present


def test_best_draft_prefers_completed_with_most_rounds():
    drafts = [
        {"draft_id": "rook", "status": "complete", "settings": {"rounds": 3}},
        {"draft_id": "startup", "status": "complete", "settings": {"rounds": 15}},
        {"draft_id": "pre", "status": "pre_draft", "settings": {"rounds": 20}},
    ]
    assert kp._best_draft(drafts)["draft_id"] == "startup"   # completed + most rounds


def test_num_rounds_yahoo_from_deepest_drafted_round():
    # Yahoo has no round count in its draft list, so derive it from the picks.
    assert kp._num_rounds("yahoo", "L", drafted={"a": 1, "b": 16, "c": 9}) == 16
    # A tiny/empty draft falls back to the standard default depth.
    assert kp._num_rounds("yahoo", "L", drafted={"a": 3}) == 15
    assert kp._num_rounds("yahoo", "L", drafted={}) == 15


def test_drafted_round_map_other_platform_empty():
    assert kp._drafted_round_map("espn", "L", 2026) == {}


# ── Sleeper season chain (the offseason "everyone costs R15" bug) ────────────

def _fake_sleeper_api(drafts_by_league, picks_by_draft, history=None):
    m = types.ModuleType("dashboard_services.api")
    m.get_drafts = lambda lid: drafts_by_league.get(str(lid), [])
    m.get_draft_picks = lambda did: picks_by_draft.get(str(did), [])
    m.build_league_history_map = lambda plat, lid, season: (history or {})
    return m


def test_sleeper_falls_back_to_previous_season_draft(monkeypatch):
    # Offseason: the current league's draft is scheduled but has no picks, so the
    # rounds live under last season's league. Without the chain walk every player
    # looked undrafted and got the flat last-round cost.
    drafts = {
        "2026": [{"draft_id": "d26", "status": "pre_draft", "settings": {"rounds": 15}}],
        "2025": [{"draft_id": "d25", "status": "complete", "settings": {"rounds": 15}}],
    }
    picks = {
        "d26": [],
        "d25": [{"player_id": "a", "round": 1}, {"player_id": "b", "round": 9}],
    }
    monkeypatch.setitem(sys.modules, "dashboard_services.api",
                        _fake_sleeper_api(drafts, picks, {2026: "2026", 2025: "2025"}))
    drafted, deepest = kp._sleeper_draft_history("2026", 2026)
    assert drafted == {"a": 1, "b": 9}
    assert deepest == 15


def test_sleeper_prefers_most_recent_season_for_a_player(monkeypatch):
    # A player taken in both seasons keeps the newer round (re-drafted since).
    drafts = {
        "2026": [{"draft_id": "d26", "status": "complete", "settings": {"rounds": 12}}],
        "2025": [{"draft_id": "d25", "status": "complete", "settings": {"rounds": 15}}],
    }
    picks = {
        "d26": [{"player_id": "a", "round": 2}],
        "d25": [{"player_id": "a", "round": 11}, {"player_id": "b", "round": 4}],
    }
    monkeypatch.setitem(sys.modules, "dashboard_services.api",
                        _fake_sleeper_api(drafts, picks, {2026: "2026", 2025: "2025"}))
    drafted, deepest = kp._sleeper_draft_history("2026", 2026)
    assert drafted["a"] == 2      # newest season wins
    assert drafted["b"] == 4      # only in the older draft
    assert deepest == 15          # scale from the deepest completed draft


def test_sleeper_draft_context_uses_deepest_for_rounds(monkeypatch):
    drafts = {"L": [{"draft_id": "d", "status": "complete", "settings": {"rounds": 16}}]}
    picks = {"d": [{"player_id": "a", "round": 3}]}
    monkeypatch.setitem(sys.modules, "dashboard_services.api",
                        _fake_sleeper_api(drafts, picks, {2026: "L"}))
    drafted, num_rounds = kp._draft_context("sleeper", "L", 2026)
    assert drafted == {"a": 3}
    assert num_rounds == 16


def test_sleeper_rookie_only_draft_keeps_standard_depth(monkeypatch):
    # A 3-round rookie draft must not become the cost scale (undrafted would
    # otherwise cost R3, making every waiver add look like a steal).
    drafts = {"L": [{"draft_id": "r", "status": "complete", "settings": {"rounds": 3}}]}
    picks = {"r": [{"player_id": "rk", "round": 1}]}
    monkeypatch.setitem(sys.modules, "dashboard_services.api",
                        _fake_sleeper_api(drafts, picks, {2026: "L"}))
    drafted, num_rounds = kp._draft_context("sleeper", "L", 2026)
    assert drafted == {"rk": 1}
    assert num_rounds == 15   # standard depth, not 3


def test_best_draft_falls_back_when_none_complete():
    drafts = [{"draft_id": "x", "status": "pre_draft", "settings": {"rounds": 12}}]
    assert kp._best_draft(drafts)["draft_id"] == "x"
    assert kp._best_draft([]) is None
