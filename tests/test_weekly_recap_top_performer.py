from pathlib import Path

from dashboard_services.pages.recap_page import _top_performers_by_roster


ROOT = Path(__file__).resolve().parents[1]


def _team(rid, starters, *, historical=True, bench=None):
    return {
        "roster_id": rid,
        "starters": starters,
        "bench": bench or [],
        "lineup_is_historical": historical,
    }


def _player(pid, name, points):
    return {"pid": pid, "name": name, "pts": points}


def test_top_performer_uses_starters_not_higher_scoring_bench_player():
    matchups = [{
        "left": _team("1", [_player("start", "Starter", 12.4)],
                      bench=[_player("bench", "Bench Boom", 40.0)]),
        "right": _team("2", [_player("other", "Other", 8.0)]),
    }]

    result = _top_performers_by_roster(matchups)

    assert result["1"] == [{"pid": "start", "name": "Starter", "pts": 12.4}]


def test_top_performer_preserves_provider_league_scoring_and_ties():
    # These values stand in for custom/TE-premium points already calculated by
    # the provider normalization used by Matchups; Recap must not recalculate.
    matchups = [{
        "left": _team("1", [
            _player("te", "Zed Tight End", 28.75),
            _player("wr", "Alpha Receiver", 28.75),
        ]),
        "right": _team("2", [_player("qb", "Quarterback", 20.0)]),
    }]

    result = _top_performers_by_roster(matchups)

    assert [p["pid"] for p in result["1"]] == ["wr", "te"]
    assert all(p["pts"] == 28.75 for p in result["1"])


def test_top_performer_preserves_zero_and_negative_scores():
    matchups = [{
        "left": _team("zero", [_player("a", "Zero", 0), _player("b", "Minus", -1.2)]),
        "right": _team("negative", [_player("c", "Less Negative", -2),
                                     _player("d", "More Negative", -9.5)]),
    }]

    result = _top_performers_by_roster(matchups)

    assert result["zero"][0]["pts"] == 0.0
    assert result["negative"][0]["pts"] == -2.0


def test_missing_points_and_current_roster_fallback_produce_na():
    matchups = [{
        "left": _team("missing", [_player("a", "Unknown", None)]),
        # A normalized matchup may use the current roster for display when a
        # provider has no historical lineup. Recap must not award from it.
        "right": _team("fallback", [_player("new", "Current Addition", 30)], historical=False),
    }]

    result = _top_performers_by_roster(matchups)

    assert "missing" not in result
    assert "fallback" not in result


def test_scoreboard_contract_includes_modal_context_and_protected_mobile_score_lane():
    page = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    css = (ROOT / "static/dashboard.css").read_text()

    assert "Top performer: " in page and "N/A" in page
    assert 'data-league-id=' in page and 'data-platform=' in page and 'data-season=' in page
    assert "recap-top-performer-players" in page
    assert "white-space: normal" in css
    assert "recap-matchup-main" in page
    # The performer is physically inside its team block on every breakpoint.
    assert "{top_performer_html(rid)}" in page
    assert "grid-template-columns:1fr" in css
    assert ".recap-team-score" in css
    assert "overflow-wrap:anywhere" in css


def test_completed_recap_cache_version_is_refreshed():
    source = (ROOT / "dashboard_services/ai/weekly_recap.py").read_text()
    assert "v13_story" in source
    assert "score_revision" in source


def test_recap_route_hydrates_matchups_before_rendering():
    """The recap must have normalized weekly data even on a direct page visit."""
    source = (ROOT / "routes/league_pages_bp.py").read_text()
    route = source.split("def page_recap", 1)[1].split("def ", 1)[0]

    assert "ensure_weekly_bits(ctx)" in route
    assert route.index("ensure_weekly_bits(ctx)") < route.index("build_recap_body(")
