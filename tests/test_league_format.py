from __future__ import annotations

from utils.league_format import (
    auction_budget,
    detect_league_format,
    is_auction_draft,
    is_best_ball,
)


def test_sleeper_auction_draft_type():
    assert is_auction_draft({"type": "auction", "settings": {"budget": 200}})
    assert not is_auction_draft({"type": "snake", "settings": {"rounds": 15}})


def test_snake_draft_ignores_league_auction_budget():
    """ESPN may expose auctionBudget even on snake leagues — draft type wins."""
    league = {"settings": {"draftSettings": {"type": "SNAKE", "auctionBudget": 200}}}
    assert not is_auction_draft({"type": "snake"}, league=league)
    assert not is_auction_draft(league=league)


def test_espn_auction_budget_signal():
    league = {"settings": {"draftSettings": {"type": "AUCTION", "auctionBudget": 200}}}
    assert is_auction_draft(league=league)
    assert auction_budget(league=league) == 200


def test_espn_numeric_type_needs_budget():
    assert not is_auction_draft(league={"settings": {"draftSettings": {"type": "2"}}})
    assert is_auction_draft(
        league={"settings": {"draftSettings": {"type": "2", "auctionBudget": 100}}}
    )


def test_best_ball_settings_flag():
    assert is_best_ball({"settings": {"best_ball": 1}})
    assert is_best_ball({"settings": {"bestBall": True}})
    assert not is_best_ball({"settings": {"best_ball": 0}, "name": "Best Ball Bros"})


def test_detect_league_format_combined():
    fmt = detect_league_format(
        league={"name": "X", "settings": {"best_ball": 1}},
        drafts=[{"type": "auction", "settings": {"budget": 150}}],
    )
    assert fmt["is_auction"] is True
    assert fmt["is_best_ball"] is True
    assert fmt["auction_budget"] == 150
    assert fmt["draft_type"] == "auction"


def test_detect_league_format_prefers_full_snake_over_mock_auction():
    fmt = detect_league_format(
        drafts=[
            {"type": "auction", "status": "complete", "settings": {"rounds": 3, "budget": 200}},
            {"type": "snake", "status": "complete", "settings": {"rounds": 15}},
        ],
    )
    assert fmt["is_auction"] is False
    assert fmt["draft_type"] == "snake"


def test_classify_dynasty_superflex_tep():
    from utils.league_format import classify_league_roster_format
    fmt = classify_league_roster_format(
        league={
            "settings": {"type": 2},
            "roster_positions": ["QB", "RB", "WR", "TE", "SUPER_FLEX"],
            "scoring_settings": {"bonus_rec_te": 1.0},
        },
        platform="sleeper",
    )
    assert fmt["is_dynasty"] is True
    assert fmt["is_superflex"] is True
    assert fmt["is_tep"] is True
    assert fmt["qb_format"] == "sf"


def test_classify_redraft_1qb_non_tep():
    from utils.league_format import classify_league_roster_format
    fmt = classify_league_roster_format(
        league={
            "settings": {"type": 0},
            "roster_positions": ["QB", "RB", "WR", "TE", "FLEX"],
            "scoring_settings": {"rec": 1.0, "bonus_rec_te": 0},
        },
    )
    assert fmt["is_redraft"] is True
    assert fmt["is_superflex"] is False
    assert fmt["is_tep"] is False


def test_classify_keeper():
    from utils.league_format import classify_league_roster_format
    fmt = classify_league_roster_format(league={"settings": {"type": 1}})
    assert fmt["is_keeper"] is True
    assert fmt["is_dynasty"] is False


def test_espn_platform_is_redraft_even_if_type_says_dynasty():
    from utils.league_format import classify_league_roster_format
    fmt = classify_league_roster_format(
        league={"settings": {"type": 2}}, platform="espn",
    )
    assert fmt["is_redraft"] is True


def test_yahoo_unlabelled_league_is_redraft():
    """Yahoo's get_league returns only {league_id, season, name} -- no
    settings -- so the unknown-format dynasty default priced every Yahoo
    league on dynasty values (redraft Yahoo league 1307110 got 'losing
    dynasty value' digest alerts, 2026-10-31). Yahoo is redraft unless a
    caller supplies an explicit format signal."""
    from utils.league_format import classify_league_roster_format
    fmt = classify_league_roster_format(
        league={"league_id": "1307110", "season": 2026,
                "name": "Yahoo H2H-Pts 1307110"},
        platform="yahoo",
    )
    assert fmt["is_redraft"] is True
    assert fmt["is_dynasty"] is False


def test_yahoo_explicit_format_signals_still_win():
    from utils.league_format import classify_league_roster_format
    keeper = classify_league_roster_format(
        league={"settings": {"type": 1}}, platform="yahoo")
    assert keeper["is_keeper"] is True
    dynasty = classify_league_roster_format(
        league={"settings": {"type": 2}}, platform="yahoo")
    assert dynasty["is_dynasty"] is True
    labelled = classify_league_roster_format(
        league={"league_type": "keeper"}, platform="yahoo")
    assert labelled["is_keeper"] is True


def test_unknown_sleeper_league_still_defaults_dynasty():
    from utils.league_format import classify_league_roster_format
    fmt = classify_league_roster_format(league={}, platform="sleeper")
    assert fmt["is_dynasty"] is True


# ── enrich_league_with_globals: bare Yahoo/ESPN shapes ───────────────────────

_YAHOO_GLOBALS = {
    "league_settings": {"type": 0, "league_type": "redraft", "num_teams": 12},
    "roster_positions": ["QB", "RB", "WR", "TE", "SUPER_FLEX"],
    "total_rosters": 12,
    "scoring_settings": {"rec": 1.0},
}


def _stub_globals(monkeypatch, payload):
    import dashboard_services.platform_api as papi
    calls = []

    def _globals(platform, league_id, season):
        calls.append((platform, league_id, season))
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr(papi, "get_league_globals", _globals)
    return calls


def test_enrich_bare_yahoo_league_merges_globals(monkeypatch):
    from utils.league_format import (
        classify_league_roster_format, enrich_league_with_globals,
    )
    calls = _stub_globals(monkeypatch, dict(_YAHOO_GLOBALS))
    bare = {"league_id": "1307110", "season": 2026, "name": "Yahoo H2H-Pts"}
    out = enrich_league_with_globals("yahoo", "1307110", 2026, bare)
    assert calls == [("yahoo", "1307110", 2026)]
    assert out["total_rosters"] == 12
    assert out["settings"]["type"] == 0
    fmt = classify_league_roster_format(league=out, platform="yahoo")
    assert fmt["is_redraft"] is True
    assert fmt["is_superflex"] is True
    assert fmt["qb_format"] == "sf"
    # The caller's dict is not mutated.
    assert "settings" not in bare


def test_enrich_complete_league_costs_no_globals_call(monkeypatch):
    from utils.league_format import enrich_league_with_globals
    calls = _stub_globals(monkeypatch, dict(_YAHOO_GLOBALS))
    full = {"settings": {"type": 2}, "total_rosters": 10,
            "roster_positions": ["QB", "RB"]}
    out = enrich_league_with_globals("sleeper", "L1", 2026, full)
    assert calls == []
    assert out is full


def test_enrich_never_overwrites_present_fields(monkeypatch):
    from utils.league_format import enrich_league_with_globals
    _stub_globals(monkeypatch, dict(_YAHOO_GLOBALS))
    partial = {"settings": {"type": 2}, "total_rosters": 10}
    out = enrich_league_with_globals("yahoo", "L9", 2026, partial)
    assert out["settings"] == {"type": 2}          # present: untouched
    assert out["total_rosters"] == 10              # present: untouched
    assert out["roster_positions"] == _YAHOO_GLOBALS["roster_positions"]


def test_enrich_globals_failure_returns_league_unchanged(monkeypatch):
    from utils.league_format import enrich_league_with_globals
    _stub_globals(monkeypatch, RuntimeError("provider down"))
    bare = {"league_id": "1307110", "season": 2026}
    out = enrich_league_with_globals("yahoo", "1307110", 2026, bare)
    assert out == bare
