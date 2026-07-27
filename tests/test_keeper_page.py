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


def _stub_body_deps(monkeypatch):
    """Minimal stubs so build_keeper_body renders without the DB/player index."""
    for name, attrs in (
        ("dashboard_services.player_value_history",
         {"load_current_values_from_db": lambda: [{"id": "a", "redraft_value_1qb": 1000.0}]}),
        ("utils.utils", {"load_players_index": lambda: {"a": {"name": "A", "pos": "RB"}}}),
    ):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.setattr(kp, "_adp_map", lambda is_sf, season, source="consensus": {"a": 1.0})
    monkeypatch.setattr(kp, "_draft_context", lambda plat, lid, season: ({"a": 5}, 15))


_BODY_CTX = {"total_rosters": 10, "rosters": [{"roster_id": 1, "players": ["a"]}],
             "roster_positions": ["RB"]}


def test_draft_handoff_link_uses_route_season_when_ctx_has_none(monkeypatch):
    """The "Open in Draft Room" button is the only way to carry keepers into the
    draft room. Building its link from ctx alone meant a cached ctx without a
    season produced an empty link, which dropped the button entirely."""
    _stub_body_deps(monkeypatch)
    html = kp.build_keeper_body(dict(_BODY_CTX), viewer_roster_id="1",
                                platform="sleeper", league_id="L123", season=2026)
    assert "kpr-to-draft" in html
    assert "/sleeper/2026/L123/draft?keepers=1" in html


def test_draft_handoff_link_prefers_ctx_season(monkeypatch):
    _stub_body_deps(monkeypatch)
    html = kp.build_keeper_body(dict(_BODY_CTX, season=2025), viewer_roster_id="1",
                                platform="sleeper", league_id="L123", season=2026)
    assert "/sleeper/2025/L123/draft?keepers=1" in html


def _league_ctx_for_limits(monkeypatch):
    vals = [{"id": c, "redraft_value_1qb": v} for c, v in
            (("a", 1200), ("b", 1100), ("c", 1000), ("d", 900), ("e", 800), ("f", 700))]
    idx = {c: {"name": c.upper(), "pos": "WR"} for c in "abcdef"}
    for name, attrs in (
        ("dashboard_services.player_value_history", {"load_current_values_from_db": lambda: vals}),
        ("utils.utils", {"load_players_index": lambda: idx}),
    ):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.setattr(kp, "_adp_map",
                        lambda is_sf, season, source="consensus": {c: i + 1 for i, c in enumerate("abcdef")})
    monkeypatch.setattr(kp, "_draft_context", lambda plat, lid, season: ({}, 15))
    return {"season": 2026, "total_rosters": 2,
            "rosters": [{"roster_id": 1, "players": ["a", "b", "c"]},
                        {"roster_id": 2, "players": ["d", "e", "f"]}],
            "roster_positions": ["WR"]}


def test_keeper_limit_override_applies_to_every_team(monkeypatch):
    """The user's "Keep up to N" is handed off from the keeper page. Without it
    rival teams were projected against the league default, so a user keeping 3
    saw everyone else holding fewer."""
    ctx = _league_ctx_for_limits(monkeypatch)
    out = kp.compute_league_keepers(ctx, platform="sleeper", league_id="L",
                                    viewer_roster_id="1", limit_override=3)
    assert out["limit"] == 3
    assert out["byTeam"]["1"] == ["a", "b", "c"]
    assert out["byTeam"]["2"] == ["d", "e", "f"]


def test_keeper_limit_falls_back_to_league_default(monkeypatch):
    ctx = _league_ctx_for_limits(monkeypatch)
    out = kp.compute_league_keepers(ctx, platform="sleeper", league_id="L",
                                    viewer_roster_id="1", limit_override=None)
    assert out["limit"] == 2
    assert out["byTeam"]["2"] == ["d", "e"]


def test_undrafted_cost_override_repricing(monkeypatch):
    """With no drafted rounds (a dynasty roster), every player prices at the
    undrafted default - the deepest round the league ever drafted. The keeper
    page's "Undrafted cost" now rides along so the user can reprice them."""
    ctx = _league_ctx_for_limits(monkeypatch)
    monkeypatch.setattr(kp, "_draft_context", lambda plat, lid, season: ({}, 23))
    # one_per_round off here so this isolates the undrafted repricing (otherwise
    # the three undrafted keepers would bump to unique rounds).
    default = kp.compute_league_keepers(ctx, platform="sleeper", league_id="L",
                                        viewer_roster_id="1", limit_override=3,
                                        rules_override={"one_per_round": False})
    assert {k["costRound"] for k in default["kept"]} == {23}
    tuned = kp.compute_league_keepers(ctx, platform="sleeper", league_id="L",
                                      viewer_roster_id="1", limit_override=3,
                                      rules_override={"undrafted_round": 15, "one_per_round": False})
    assert {k["costRound"] for k in tuned["kept"]} == {15}


def test_rules_override_is_bounded_to_real_rounds(monkeypatch):
    """Rules arrive from query params, so an absurd undrafted round must clamp
    into the league's actual draft depth rather than through it."""
    ctx = _league_ctx_for_limits(monkeypatch)
    monkeypatch.setattr(kp, "_draft_context", lambda plat, lid, season: ({}, 20))
    out = kp.compute_league_keepers(ctx, platform="sleeper", league_id="L",
                                    viewer_roster_id="1", limit_override=3,
                                    rules_override={"undrafted_round": 999, "one_per_round": False})
    assert {k["costRound"] for k in out["kept"]} == {20}


def test_keeper_limit_override_is_bounded(monkeypatch):
    """The value arrives from a query param, so it must not be trusted raw."""
    ctx = _league_ctx_for_limits(monkeypatch)
    out = kp.compute_league_keepers(ctx, platform="sleeper", league_id="L",
                                    viewer_roster_id="1", limit_override=9999)
    assert out["limit"] == 25


# ── Dynasty leagues: the tool doesn't apply ──────────────────────────────────

_DYN_BASE = {"season": 2026, "total_rosters": 12,
             "rosters": [{"roster_id": 1, "players": ["a"]}],
             "roster_positions": ["RB"]}


def test_dynasty_without_keeper_limit_is_detected():
    # Sleeper league type: 0 redraft, 1 keeper, 2 dynasty.
    assert kp.is_dynasty_without_keepers(dict(_DYN_BASE, league_settings={"type": 2})) is True


def test_dynasty_with_a_keeper_limit_keeps_the_tool():
    """A dynasty league that configures keepers is a real keeper league."""
    ctx = dict(_DYN_BASE, league_settings={"type": 2, "max_keepers": 2})
    assert kp.is_dynasty_without_keepers(ctx) is False


def test_non_dynasty_types_keep_the_tool():
    for t in (0, 1):
        ctx = dict(_DYN_BASE, league_settings={"type": t})
        assert kp.is_dynasty_without_keepers(ctx) is False, t


def test_unknown_league_type_keeps_the_tool():
    """ESPN and Yahoo publish no dynasty flag, so they must not be hidden."""
    assert kp.is_dynasty_without_keepers(dict(_DYN_BASE, league_settings={})) is False
    assert kp.is_dynasty_without_keepers(dict(_DYN_BASE)) is False


def test_dynasty_renders_the_notice_instead_of_the_table(monkeypatch):
    _stub_body_deps(monkeypatch)
    html = kp.build_keeper_body(dict(_DYN_BASE, league_settings={"type": 2}),
                                viewer_roster_id="1", platform="sleeper",
                                league_id="L", season=2026)
    assert "kpr-dyn" in html and "You keep everyone" in html
    assert "kpr-tbody" not in html          # no placeholder-cost table
    assert "Show it anyway" in html         # escape hatch for informal keepers


def test_dynasty_force_shows_the_tool(monkeypatch):
    _stub_body_deps(monkeypatch)
    html = kp.build_keeper_body(dict(_DYN_BASE, league_settings={"type": 2}),
                                viewer_roster_id="1", platform="sleeper",
                                league_id="L", season=2026, force=True)
    assert "kpr-tbody" in html and "kpr-dyn" not in html


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


# ── One keeper per round + new controls ──────────────────────────────────────

def test_body_renders_undrafted_dropdown_and_one_per_round_toggle(monkeypatch):
    _stub_body_deps(monkeypatch)
    html = kp.build_keeper_body(dict(_BODY_CTX), viewer_roster_id="1",
                                platform="sleeper", league_id="L123", season=2026)
    # Undrafted cost is a dropdown (was a bare number input) with a "Last round"
    # default and explicit rounds.
    assert '<select id="kpr-undr">' in html
    assert ">Last round<" in html
    # One-keeper-per-round toggle, on by default (in the UI and the seed).
    assert 'id="kpr-opr"' in html and "checked" in html
    assert '"onePerRound":true' in html.replace(" ", "")


def test_one_per_round_resolves_collisions_in_projection(monkeypatch):
    """Two viewer keepers drafted in the same round collide; with one-per-round
    (default) the projection bumps the weaker to a neighbouring round so no two
    share a cost round. Turning it off leaves the raw duplicate."""
    ctx = _league_ctx_for_limits(monkeypatch)
    # a & b both drafted R5 (a is the higher-value, so it holds R5); c at R9.
    monkeypatch.setattr(kp, "_draft_context",
                        lambda plat, lid, season: ({"a": 5, "b": 5, "c": 9}, 15))

    def _viewer_cost_rounds(rules_override):
        out = kp.compute_league_keepers(ctx, platform="sleeper", league_id="L",
                                        viewer_roster_id="1", limit_override=3,
                                        rules_override=rules_override)
        return sorted(k["costRound"] for k in out["kept"] if str(k["rosterId"]) == "1")

    # Default (no override) -> one_per_round on -> unique rounds, weaker bumped earlier.
    assert _viewer_cost_rounds(None) == [4, 5, 9]
    # Explicitly off -> the raw collision remains (two at R5).
    assert _viewer_cost_rounds({"one_per_round": False}) == [5, 5, 9]
