"""Fumble-lost scoring in ScoreZone.

Regression test: Drake Maye's sack+fumble (Week 4 vs BUF) rendered 0 pts
in ScoreZone but should be -2 under standard fum_lost scoring.
"""
import pytest

from utils.scorezone_pbp import _fumble_lost_fumbler, extract_pbp_plays
from utils.scorezone_stats import rz_stat_line_from_ps

MAYE_TEXT = (
    "D.Maye sacked at BUF 30 for -7 yards (E.Oliver). "
    "FUMBLES (E.Oliver) [E.Oliver], RECOVERED by BUF-G.Gaines at BUF 30."
)


def test_rz_stat_line_extracts_fum_lost_flat():
    line = rz_stat_line_from_ps({"fumbles": 1})
    assert line["fum_lost"] == 1.0


def test_rz_stat_line_extracts_fum_lost_nested():
    line = rz_stat_line_from_ps({"Rushing": {"fumblesLost": 2}})
    assert line["fum_lost"] == 2.0


def test_rz_stat_line_fum_lost_defaults_zero():
    line = rz_stat_line_from_ps({"Passing": {"passYds": 84}})
    assert line["fum_lost"] == 0.0


def test_fumble_lost_fumbler_opponent_recovery():
    assert _fumble_lost_fumbler(MAYE_TEXT, "NE") == "D.Maye"


def test_fumble_lost_own_recovery_not_lost():
    text = (
        "D.Maye sacked at NE 30 for -7 yards (E.Oliver). FUMBLES (E.Oliver), "
        "RECOVERED by NE-D.Maye at NE 30."
    )
    assert _fumble_lost_fumbler(text, "NE") == ""


def test_fumble_out_of_bounds_not_lost():
    text = "(J.Bates). FUMBLES (J.Bates), ball out of bounds at GB 42."
    assert _fumble_lost_fumbler(text, "GB") == ""


def test_fumble_lost_no_fumble_in_text():
    assert (
        _fumble_lost_fumbler("D.Maye pass short left to K.Boutte for 8 yards.", "NE")
        == ""
    )


def _maye_box():
    return {
        "allPlayByPlay": [
            {
                "playId": "1",
                "playText": MAYE_TEXT,
                "playerStats": {
                    "maye": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Rushing": {"rushYds": -7, "carries": 1},
                    }
                },
            }
        ]
    }


def test_pbp_fallback_credits_fum_lost():
    """Booth text names the fumble but Tank01 playerStats omit it."""
    plays = extract_pbp_plays(_maye_box(), "NE@BUF")
    maye = [p for p in plays if p.get("name") == "Drake Maye"]
    assert maye, "expected a Drake Maye play row"
    assert maye[0]["stat_line"].get("fum_lost") == 1.0


def test_pbp_no_double_count_when_feed_has_fumble():
    """When Tank01 already ships the fumble, the text fallback must not add another."""
    box = _maye_box()
    box["allPlayByPlay"][0]["playerStats"]["maye"]["Rushing"]["fumbles"] = 1
    plays = extract_pbp_plays(box, "NE@BUF")
    maye = [p for p in plays if p.get("name") == "Drake Maye"]
    assert maye[0]["stat_line"].get("fum_lost") == 1.0


def test_pbp_out_of_bounds_fumble_not_credited():
    box = {
        "allPlayByPlay": [
            {
                "playId": "1",
                "playText": "(J.Bates). FUMBLES (J.Bates), ball out of bounds at GB 42.",
                "playerStats": {
                    "kraft": {
                        "longName": "Tucker Kraft",
                        "teamAbv": "GB",
                        "Receiving": {"recYds": 4, "receptions": 1},
                    }
                },
            }
        ]
    }
    plays = extract_pbp_plays(box, "GB@ATL")
    kraft = [p for p in plays if p.get("name") == "Tucker Kraft"]
    assert kraft, "expected a Tucker Kraft play row"
    assert kraft[0]["stat_line"].get("fum_lost") in (0, 0.0, None)
