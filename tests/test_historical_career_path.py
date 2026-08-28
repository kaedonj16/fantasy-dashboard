"""Career-path overlay: prior elite + current situation, not last-year comps."""
from dashboard_services.historical.aggregates_store import _merge_career_path_overlay
from dashboard_services.historical.board import build_deep_panel, build_hist_panel_copy
from dashboard_services.historical.career_path import (
    apply_career_path_history,
    is_bounce_back_query,
    is_bounce_back_row,
)
from dashboard_services.historical.signals import lookup_history_probability


def _pct_rate(pct, n=40):
    return {
        "display_pct": pct,
        "sample_size": n,
        "successes": int(round(pct * n / 100.0)),
        "raw_rate": pct / 100.0,
        "smoothed_rate": pct / 100.0,
        "confidence": "moderate",
    }


def test_bounce_back_detects_prior_elite_after_a_down_year():
    assert is_bounce_back_query({
        "prior_top12_count": 1,
        "previous_season_finish": 42,
        "years_experience": 2,
    }) is True
    assert is_bounce_back_query({
        "prior_top12_count": 0,
        "previous_season_finish": 42,
        "years_experience": 2,
    }) is False
    assert is_bounce_back_query({
        "prior_top12_count": 1,
        "previous_season_finish": 8,
        "years_experience": 2,
    }) is False
    assert is_bounce_back_row({
        "prior_top12_count": 1,
        "previous_season_finish": 42,
        "ppr_positional_finish": 10,
    }) is True
    assert is_bounce_back_row({
        "prior_top12_count": 1,
        "previous_season_finish": 8,
        "ppr_positional_finish": 10,
    }) is False


def test_overlay_merge_stamps_counts_without_dropping_other_rates():
    data = {
        "preseason_profiles": {"by_player": {"11631": {"position": "WR"}}},
        "repeat_and_breakout": {"WR": {"n_prev_top12": 9}},
    }
    overlay = {
        "prior_top12_count": {"11631": 1},
        "bounce_back": {
            "WR": {
                "n_bounce_back": 40,
                "bounce_back": {"top_12": _pct_rate(31, 40)},
            }
        },
    }
    _merge_career_path_overlay(data, overlay)
    assert data["preseason_profiles"]["by_player"]["11631"]["prior_top12_count"] == 1
    assert data["repeat_and_breakout"]["WR"]["n_bounce_back"] == 40
    assert data["repeat_and_breakout"]["WR"]["n_prev_top12"] == 9


def test_hist_headline_uses_bounce_back_not_last_year_lookalikes():
    bounce = {
        "top_5": _pct_rate(14, 20),
        "top_12": _pct_rate(28, 20),
        "top_24": _pct_rate(45, 20),
    }
    aggs = {
        "preseason_profiles": {
            "by_player": {
                "btj": {
                    "position": "WR",
                    "years_experience": 2,
                    "draft_capital_bucket": "round_1",
                    "previous_season_finish": 42,
                    "prior_top12_count": 1,
                }
            }
        },
        "comps": {"by_position": {"WR": {"leaves": [], "baseline": {}}}},
        "repeat_and_breakout": {
            "WR": {
                "bounce_back": {
                    "top_5": _pct_rate(12, 50),
                    "top_12": _pct_rate(31, 50),
                    "top_24": _pct_rate(48, 50),
                },
                "n_bounce_back": 50,
                "bounce_back_by_stage": {"year_3": bounce},
                "bounce_back_by_capital": {},
            }
        },
        "age_curves": {},
    }
    panel = build_deep_panel("btj", aggs, extra={"position": "WR"})
    assert panel["history"]["career_path"] == "bounce_back"
    assert panel["history"]["career_path_rate"] == "stage"
    assert panel["history"].get("examples") == []
    headline = str(panel["copy"].get("headline") or "").lower()
    assert "already been top-12" in headline
    assert "year 3" in headline
    assert "outside the top 36 last year" in headline
    assert "never" not in headline
    top12 = next(row for row in panel["copy"]["hit_rates"] if row["tier"] == "top_12")
    assert top12["pct"] == 28
    assert "Career elite" in [row["label"] for row in panel["copy"]["profile"]]
    assert "this player's historical chance" in panel["copy"]["cohort_note"].lower()
    hist = lookup_history_probability(
        {
            "position": "WR",
            "years_experience": 2,
            "draft_capital_bucket": "round_1",
            "previous_season_finish": 42,
            "prior_top12_count": 1,
        },
        aggs,
    )
    assert hist["source"] == "career_path"
    assert hist["p_top_12"] == 0.28


def test_never_elite_keeps_comps_and_does_not_use_bounce_back():
    looked = {
        "position": "WR",
        "n": 80,
        "rates": {"top_12": _pct_rate(4, 80)},
        "key_used": {"position": "WR", "prior_finish": "outside_36"},
        "source": "comps",
    }
    query = {
        "position": "WR",
        "prior_top12_count": 0,
        "previous_season_finish": 42,
        "years_experience": 2,
    }
    aggs = {
        "repeat_and_breakout": {
            "WR": {
                "bounce_back": {"top_12": _pct_rate(31, 50)},
                "n_bounce_back": 50,
            }
        }
    }
    out = apply_career_path_history(query, looked, aggs)
    assert out.get("career_path") is None
    assert out["rates"]["top_12"]["display_pct"] == 4


def test_bounce_back_panel_copy_names_career_elite():
    copy = build_hist_panel_copy({
        "n": 20,
        "career_path": "bounce_back",
        "career_path_rate": "stage",
        "key_used": {
            "position": "WR",
            "career_stage": "year_3",
            "draft_capital": "round_1",
            "prior_finish": "outside_36",
            "prior_elite": "has_been",
        },
        "profile_key": {
            "position": "WR",
            "career_stage": "year_3",
            "draft_capital": "round_1",
            "prior_finish": "outside_36",
            "prior_elite": "has_been",
        },
        "rates": {
            "top_5": _pct_rate(14, 20),
            "top_12": _pct_rate(28, 20),
            "top_24": _pct_rate(45, 20),
        },
    })
    assert "already been top-12" in copy["headline"].lower()
    assert "Career elite" in [row["label"] for row in copy["profile"]]
    assert any(row["value"] == "Had been top-12 before" for row in copy["profile"])
    assert "prior top-12" in copy["cohort_note"]
