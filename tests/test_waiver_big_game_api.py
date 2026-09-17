"""League-context curation for the unexpected-performance strip."""
from routes.waiver_api_bp import _curate_big_game_discoveries


def _row(pid, pos, surprise, sustain, absolute, value=500, confirmed=True):
    return {"player_id": pid, "position": pos, "performance_surprise": surprise,
            "role_sustainability": sustain, "absolute_score": absolute,
            "role_confirmed": confirmed, "value": value, "cautions": []}


def test_one_qb_results_are_not_dominated_by_replacement_passers():
    rows = [_row(f"q{i}", "QB", .7, .45, .5) for i in range(5)]
    rows += [_row("wr", "WR", .55, .8, .45), _row("te", "TE", .6, .7, .6)]
    result = _curate_big_game_discoveries(rows, superflex=False)
    assert [r["position"] for r in result].count("QB") <= 1
    assert {r["player_id"] for r in result} >= {"wr", "te"}


def test_superflex_preserves_starting_qb_relevance():
    rows = [_row("q1", "QB", .7, .7, .6), _row("q2", "QB", .65, .7, .55),
            _row("wr", "WR", .6, .6, .5)]
    result = _curate_big_game_discoveries(rows, superflex=True)
    assert {"q1", "q2"}.issubset({r["player_id"] for r in result})


def test_curated_order_rewards_sustainability_not_raw_surprise_alone():
    fluke = _row("fluke", "WR", .95, .1, .9, confirmed=False)
    fluke["cautions"] = ["td_dependent", "role_unconfirmed"]
    real = _row("real", "TE", .62, .9, .6, value=2500)
    result = _curate_big_game_discoveries([fluke, real])
    assert result[0]["player_id"] == "real"
