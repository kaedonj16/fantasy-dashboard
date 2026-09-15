"""Regression coverage for two-point conversion parsing and scoring.

A single provider play can pack a touchdown, the two-point attempt, and its
result into one booth line. The conversion must be scored with its own canonical
keys (pass_2pt / rush_2pt / rec_2pt), attributed to the conversion actors (never
the TD player), and it must never count conversion yardage as scrimmage yardage.
"""
from __future__ import annotations

from utils.redzone_pbp import (
    extract_pbp_plays,
    parse_two_point_conversion,
    two_point_attempted,
    two_point_succeeded,
    _two_point_segments,
)
from utils.redzone_alt_pbp import parse_pbp_play_stats


PASS_SUCCESS = (
    "A.Jones left end for 3 yards, TOUCHDOWN. "
    "TWO-POINT CONVERSION ATTEMPT. "
    "C.Wentz pass to J.Jefferson is complete. ATTEMPT SUCCEEDS."
)
RUSH_SUCCESS = (
    "J.Hurts up the middle for 1 yard, TOUCHDOWN. "
    "TWO-POINT CONVERSION ATTEMPT. J.Hurts up the middle. ATTEMPT SUCCEEDS."
)
PASS_FAIL = (
    "A.Jones left end for 3 yards, TOUCHDOWN. "
    "TWO-POINT CONVERSION ATTEMPT. "
    "C.Wentz pass to J.Jefferson is incomplete. ATTEMPT FAILS."
)
INCOMPLETE = (
    "A.Jones left end for 3 yards, TOUCHDOWN. "
    "TWO-POINT CONVERSION ATTEMPT. C.Wentz pass to J.Jefferson is incomplete."
)


# ── Pure segmentation / detection ────────────────────────────────────────────

def test_segments_split_touchdown_from_conversion():
    main, conv = _two_point_segments(PASS_SUCCESS)
    assert "TOUCHDOWN" in main
    assert "TWO-POINT" not in main
    assert "SUCCEEDS" in conv
    assert two_point_attempted(PASS_SUCCESS)
    assert not two_point_attempted("A.Jones left end for 3 yards, TOUCHDOWN.")


def test_success_and_failure_detection():
    _, conv_ok = _two_point_segments(PASS_SUCCESS)
    _, conv_bad = _two_point_segments(PASS_FAIL)
    _, conv_inc = _two_point_segments(INCOMPLETE)
    assert two_point_succeeded(conv_ok)
    assert not two_point_succeeded(conv_bad)
    assert not two_point_succeeded(conv_inc)


# ── parse_two_point_conversion (shared, provider-agnostic) ───────────────────

def test_passing_conversion_actors_and_keys():
    contribs = parse_two_point_conversion(PASS_SUCCESS)
    by_role = {c["role"]: c for c in contribs}
    assert by_role["conv_passer"]["name"] == "C.Wentz"
    assert by_role["conv_passer"]["stat_line"] == {"pass_2pt": 1}
    assert by_role["conv_receiver"]["name"] == "J.Jefferson"
    assert by_role["conv_receiver"]["stat_line"] == {"rec_2pt": 1}
    # No conversion yardage leaks in as scrimmage yardage.
    for c in contribs:
        assert "pass_yds" not in c["stat_line"]
        assert "rec_yds" not in c["stat_line"]


def test_rushing_conversion_actor_and_key():
    contribs = parse_two_point_conversion(RUSH_SUCCESS)
    assert len(contribs) == 1
    assert contribs[0]["role"] == "conv_rusher"
    assert contribs[0]["stat_line"] == {"rush_2pt": 1}


def test_failed_and_incomplete_conversions_score_nothing():
    assert parse_two_point_conversion(PASS_FAIL) == []
    assert parse_two_point_conversion(INCOMPLETE) == []


def test_revision_extraction_ignores_success_gate():
    # For tombstones, we need the actors regardless of the (now nullified) result.
    contribs = parse_two_point_conversion(PASS_FAIL, require_success=False)
    roles = {c["role"] for c in contribs}
    assert roles == {"conv_passer", "conv_receiver"}


# ── Alt (ESPN/Sleeper) booth-line parsing ────────────────────────────────────

def test_alt_path_scores_td_and_conversion_separately():
    out = parse_pbp_play_stats(PASS_SUCCESS)
    # TD actor keeps his rushing TD; conversion actors get 2PT keys only.
    assert out["a.jones"]["rush_td"] == 1
    assert out["a.jones"]["rush_yds"] == 3
    assert out["c.wentz"] == {"pass_2pt": 1}
    assert out["j.jefferson"] == {"rec_2pt": 1}
    # Jefferson must NOT be credited an ordinary reception or the TD's yardage.
    assert "rec" not in out["j.jefferson"]
    assert "rec_yds" not in out["j.jefferson"]


def test_alt_path_failed_conversion_only_td():
    out = parse_pbp_play_stats(PASS_FAIL)
    assert out["a.jones"]["rush_td"] == 1
    assert "c.wentz" not in out
    assert "j.jefferson" not in out


# ── End-to-end Tank01 extraction ─────────────────────────────────────────────

_PLAYER_META = {
    "jones": {"name": "Aaron Jones", "team": "MIN", "position": "RB"},
    "wentz": {"name": "Carson Wentz", "team": "MIN", "position": "QB"},
    "jeff": {"name": "Justin Jefferson", "team": "MIN", "position": "WR"},
}
_NAME_TO_PID = {
    "aaron jones": "jones", "a jones": "jones",
    "carson wentz": "wentz", "c wentz": "wentz",
    "justin jefferson": "jeff", "j jefferson": "jeff",
}


def _combined_box(play_status: str = ""):
    play = {
        "playId": "55",
        "play": PASS_SUCCESS,
        "playerStats": {
            "a": {"longName": "Aaron Jones", "teamAbv": "MIN",
                  "Rushing": {"carries": "1", "rushYds": "3"}},
        },
    }
    if play_status:
        play["playStatus"] = play_status
    return {"allPlayByPlay": [play]}


def test_tank01_combined_td_and_conversion_contributions():
    plays = extract_pbp_plays(
        _combined_box(), "20260101_MIN@GB",
        name_to_pid=_NAME_TO_PID, player_meta_by_pid=_PLAYER_META,
    )
    by_pid = {p["pid"]: p for p in plays}
    assert set(by_pid) == {"jones", "wentz", "jeff"}
    # TD actor stays independent from the conversion actors.
    assert by_pid["jones"]["is_td"] is True
    assert by_pid["jones"]["stat_line"]["rush_td"] == 1
    assert "pass_2pt" not in by_pid["jones"]["stat_line"]
    # Conversion actors: 2PT keys only, never flagged as a TD (no TD alert).
    assert by_pid["wentz"]["stat_line"] == {"pass_2pt": 1}
    assert by_pid["wentz"]["is_td"] is False
    assert by_pid["wentz"]["contrib_role"] == "conv_passer"
    assert by_pid["jeff"]["stat_line"] == {"rec_2pt": 1}
    assert by_pid["jeff"]["is_td"] is False
    assert by_pid["jeff"]["contrib_role"] == "conv_receiver"


def test_tank01_all_contributions_share_one_canonical_play():
    plays = extract_pbp_plays(
        _combined_box(), "20260101_MIN@GB",
        name_to_pid=_NAME_TO_PID, player_meta_by_pid=_PLAYER_META,
    )
    assert {p["play_id"] for p in plays} == {"55"}


def test_cross_surface_server_and_client_score_2pt_identically():
    """The RedZone event delta / player total / modal all run the client
    _lineToPts; the matchup total and the device-push body run the server
    week_stats_line_points. Both must resolve the SAME 2PT value from the league
    dictionary, or the surfaces would disagree (a visual +2 over a wrong total).
    """
    import json
    import shutil
    import subprocess

    from utils.fantasy_scoring import week_stats_line_points

    if shutil.which("node") is None:
        import pytest as _pt
        _pt.skip("Node.js not available")

    rz = (__import__("pathlib").Path(__file__).resolve().parents[1] / "static" / "redzone.js").read_text()
    scoring_js = rz[rz.index("function _n(x)"):rz.index("function _scoringForPid")]
    league = {"pass_2pt": 2, "rush_2pt": 2, "rec_2pt": 2, "rush_td": 6, "rush_yd": 0.1}
    cases = {
        "wentz": ({"pass_2pt": 1}, "QB"),
        "jeff": ({"rec_2pt": 1}, "WR"),
        "jones": ({"rush_yds": 3, "rush_td": 1}, "RB"),
    }
    js = scoring_js + "var lg=" + json.dumps(league) + ";console.log(JSON.stringify({" + ",".join(
        f'{k}:_lineToPts({json.dumps(line)},lg,{json.dumps(pos)})' for k, (line, pos) in cases.items()
    ) + "}));"
    client = json.loads(subprocess.check_output(["node", "-e", js], text=True))
    for k, (line, pos) in cases.items():
        server = round(float(week_stats_line_points(line, league, pos)), 4)
        assert round(float(client[k]), 4) == server, f"{k}: client={client[k]} server={server}"
    # The screenshot scenario resolves to real points on every surface.
    assert client["jones"] == 6.3 and client["wentz"] == 2 and client["jeff"] == 2


def test_tank01_nullified_conversion_emits_zero_tombstones():
    plays = extract_pbp_plays(
        _combined_box(play_status="overturned"), "20260101_MIN@GB",
        name_to_pid=_NAME_TO_PID, player_meta_by_pid=_PLAYER_META,
        emit_revisions=True,
    )
    by_pid = {p["pid"]: p for p in plays}
    # Every actor (TD + both conversion roles) gets an identity-preserving zero
    # so a client can reverse the points it applied — no scoring left behind.
    assert set(by_pid) == {"jones", "wentz", "jeff"}
    for pid in ("jones", "wentz", "jeff"):
        assert by_pid[pid]["stat_line"] == {}
        assert by_pid[pid]["play_state"] == "OVERTURNED"
    assert by_pid["wentz"]["contrib_role"] == "conv_passer"
    assert by_pid["jeff"]["contrib_role"] == "conv_receiver"
