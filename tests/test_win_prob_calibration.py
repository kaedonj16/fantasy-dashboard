"""Win-probability variance calibration tests.

compute_win_prob (matchup page) models pending players as normals with a
per-position sigma = a*proj + b, and floors team pending variance at
(TEAM_CV_FLOOR * remaining projection)^2. The constants were fit 2026-09-29
from 2023-2025 cached game logs (774 player-seasons, >=10 played games):
per-position OLS of weekly std on mean ppg, and a direct bootstrap of
synthetic lineup team totals for the CV floor. See
dashboard_services/matchups.py::_WINPROB_SIGMA.

The consistency test drives the REAL Lineup Lab Monte Carlo engine (node,
the exact code the browser runs) against the recalibrated page model on
synthetic pre-game matchups built from real 2025 player profiles. The two
models keep their own machinery (analytic erf vs 2,000 sims); this asserts
they rhyme within 5 points on the same lineup and projections.
"""
import re

import pytest

from dashboard_services.matchups import (
    TEAM_CV_FLOOR,
    _WINPROB_SIGMA,
    _winprob_sigma,
    compute_win_prob,
    STATUS_NOT_STARTED,
    STATUS_FINAL,
    STATUS_IN_PROGRESS,
)


# ---------------------------------------------------------------------------
# Unit: fitted sigma values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pos,proj,expected", [
    ("QB", 20.0, 7.5),    # 0.10*20 + 5.5
    ("RB", 13.0, 6.9),    # 0.30*13 + 3.0
    ("WR", 13.0, 7.28),   # 0.36*13 + 2.6
    ("TE", 10.0, 5.9),    # 0.41*10 + 1.8
    ("K", 8.0, 4.0),
    ("DEF", 8.0, 5.5),
])
def test_sigma_fitted_values(pos, proj, expected):
    assert _winprob_sigma(pos, proj) == pytest.approx(expected, abs=0.01)


def test_sigma_floor_for_degenerate_projections():
    # Guardrail only: projections clamp at 0, so only TE's 1.8 intercept can
    # trip the 2.0 floor. TE at 0 -> 1.8 -> floored to 2.0.
    assert _winprob_sigma("TE", 0.0) == pytest.approx(2.0, abs=0.01)
    assert _winprob_sigma("WR", -2.0) == pytest.approx(2.6, abs=0.01)
    # Sane projections never touch the floor.
    assert _winprob_sigma("WR", 4.0) == pytest.approx(4.04, abs=0.01)


def test_sigma_unknown_pos_falls_back_to_default():
    assert _winprob_sigma("FLEX", 13.0) == pytest.approx(
        _winprob_sigma("WR", 13.0), abs=1e-9)
    # Case-insensitive like the rest of the codebase.
    assert _winprob_sigma("qb", 20.0) == pytest.approx(
        _winprob_sigma("QB", 20.0), abs=1e-9)


def test_team_cv_floor_value():
    # Bootstrap of synthetic 9-man lineups from 2023-2025 game logs: 0.19.
    assert TEAM_CV_FLOOR == pytest.approx(0.20)


def test_sigma_model_covers_all_skill_positions():
    assert set(_WINPROB_SIGMA) >= {"QB", "RB", "WR", "TE", "K", "DEF"}


# ---------------------------------------------------------------------------
# Existing behavior: finished games and in-progress handling
# ---------------------------------------------------------------------------

def _one_qb_side(pid, pts):
    return {"starters": [{"pid": pid, "pos": "QB", "pts": pts}]}


def test_finished_game_edges():
    st = {"1": STATUS_FINAL, "2": STATUS_FINAL}
    assert compute_win_prob(_one_qb_side("1", 120.0), _one_qb_side("2", 100.0),
                            st, {}) == 1.0
    assert compute_win_prob(_one_qb_side("1", 100.0), _one_qb_side("2", 120.0),
                            st, {}) == 0.0
    assert compute_win_prob(_one_qb_side("1", 110.0), _one_qb_side("2", 110.0),
                            st, {}) == 0.5


def test_in_progress_without_frac_locks_at_actual():
    # Prior behavior preserved: no frac_lookup -> current points are banked.
    st = {"1": STATUS_IN_PROGRESS, "2": STATUS_NOT_STARTED}
    proj = {"1": 20.0, "2": 20.0}
    p = compute_win_prob(_one_qb_side("1", 15.0), _one_qb_side("2", 0.0),
                         st, proj)
    # A locked at 15 vs B projected 20 -> A is the underdog.
    assert 0.0 < p < 0.5


def test_in_progress_frac_zero_matches_finished():
    # Game over (frac 0): banked score, zero remaining variance == final.
    proj = {"1": 20.0, "2": 20.0}
    st = {"1": STATUS_IN_PROGRESS, "2": STATUS_FINAL}
    frac = lambda p: 0.0  # noqa: E731
    p_frac = compute_win_prob(_one_qb_side("1", 18.0), _one_qb_side("2", 22.0),
                              st, proj, frac_lookup=frac)
    st2 = {"1": STATUS_FINAL, "2": STATUS_FINAL}
    p_final = compute_win_prob(_one_qb_side("1", 18.0), _one_qb_side("2", 22.0),
                               st2, proj)
    assert p_frac == pytest.approx(p_final, abs=1e-9)


def test_in_progress_frac_one_matches_pregame():
    # Game not started (frac 1): identical to a not-started player at 0 pts.
    proj = {"1": 20.0, "2": 20.0}
    st_live = {"1": STATUS_IN_PROGRESS, "2": STATUS_NOT_STARTED}
    frac = lambda p: 1.0  # noqa: E731
    p_live = compute_win_prob(_one_qb_side("1", 0.0), _one_qb_side("2", 0.0),
                              st_live, proj, frac_lookup=frac)
    st_pre = {"1": STATUS_NOT_STARTED, "2": STATUS_NOT_STARTED}
    p_pre = compute_win_prob(_one_qb_side("1", 0.0), _one_qb_side("2", 0.0),
                             st_pre, proj)
    assert p_live == pytest.approx(p_pre, abs=1e-9)


def test_in_progress_banks_hot_start():
    # Mid-game, a hot start moves the needle vs the pre-game coin flip.
    proj = {"1": 20.0, "2": 20.0}
    st_live = {"1": STATUS_IN_PROGRESS, "2": STATUS_NOT_STARTED}
    frac = lambda p: 0.5  # noqa: E731
    p_live = compute_win_prob(_one_qb_side("1", 15.0), _one_qb_side("2", 0.0),
                              st_live, proj, frac_lookup=frac)
    st_pre = {"1": STATUS_NOT_STARTED, "2": STATUS_NOT_STARTED}
    p_pre = compute_win_prob(_one_qb_side("1", 0.0), _one_qb_side("2", 0.0),
                             st_pre, proj)
    assert p_pre == pytest.approx(0.5, abs=0.02)
    assert p_live > 0.6


# ---------------------------------------------------------------------------
# Consistency: matchup page vs the Lab's Monte Carlo engine (node)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lab_js():
    from dashboard_services.pages.waivers_page import build_waivers_body
    page = build_waivers_body("sleeper", 2026, "12345", {})
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    s = [x for x in scripts if "function wvLoad(" in x]
    assert s, "waivers inline script not found"
    script = s[0]
    start = script.index("// ── Lineup Lab (Start/Sit tab) ──")
    end = script.index("if (!window.__brctx)", start)
    return script[start:end]


@pytest.fixture(scope="module")
def real_profiles():
    """Real 2025 player profiles for 4 distinct 9-man teams."""
    from data_building.player_distributions import (
        _week_files, _weekly_scores, build_profiles,
    )
    import json
    players = json.load(open("cache/players_index.json"))

    def pos_of(pid):
        p = players.get(str(pid)) or {}
        return (p.get("pos") or p.get("position") or "").upper()

    ws = _week_files(2025)
    by_pos = {}
    for pid in ws:
        pos = pos_of(pid)
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        pts, _, _ = _weekly_scores(pid, 2025)
        if len(pts) >= 12:
            by_pos.setdefault(pos, []).append(
                (sum(pts), str(pid), sum(pts) / len(pts)))
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    teams = []
    for t in range(4):
        off = t * 8  # spacing exceeds max take per position: no dupes
        team = []
        team.append(("QB",) + by_pos["QB"][off][1:])
        for pos, n in (("RB", 2), ("WR", 2), ("TE", 1)):
            for i in range(n):
                team.append((pos,) + by_pos[pos][off + i][1:])
        flex_candidates = (
            [("RB",) + by_pos["RB"][off + 2][1:]] +
            [("WR",) + by_pos["WR"][off + 2][1:]] +
            [("TE",) + by_pos["TE"][off + 1][1:]])
        flex_candidates.sort(key=lambda e: -e[2])
        team.append(flex_candidates[0])
        # K/DEF are absent from game logs; synthetic baseline entries,
        # matching production (baseline_only profiles).
        k_proj = 8.5 - 0.5 * t
        d_proj = 9.0 - 0.5 * t
        team.append(("K", f"SYN_K_{t}", k_proj))
        team.append(("DEF", f"SYN_D_{t}", d_proj))
        teams.append([(pos, pid, round(mean, 1)) for pos, pid, mean in team])

    reqs = [{"player_id": pid, "pos": pos, "mean": mean}
            for team in teams for pos, pid, mean in team]
    profiles = build_profiles(reqs, 2025, 18)
    return teams, profiles


def _lab_win_pct(lab_js, teams, profiles, a_idx, b_idx, sims=2000):
    """Run the real browser Lab engine in node; return P(A wins) in %."""
    import subprocess
    import tempfile
    import os
    from data_building.player_distributions import correlation_pairs
    from data_building.lineup_lab import _team_std_from_profiles

    team_a, team_b = teams[a_idx], teams[b_idx]
    a_pids = [pid for _, pid, _ in team_a]
    b_pids = [pid for _, pid, _ in team_b]
    pairs = correlation_pairs(a_pids + b_pids, 2025)
    corr_js = "{" + ",".join(
        '"%s:%s":%.3f' % (a, b, r) for (a, b), r in pairs.items()) + "}"
    opp_profiles = {pid: profiles[pid] for pid in b_pids}
    opp_mean = round(sum(p["mean"] for p in opp_profiles.values()), 1)
    opp_pairs = {(a, b): r for (a, b), r in pairs.items()
                 if a in opp_profiles and b in opp_profiles}
    opp_std = round(_team_std_from_profiles(b_pids, opp_profiles, opp_pairs), 1)

    def entry(pid):
        pr = profiles[pid]
        return ('{player_id:"%s",profile:{mean:%s,std:%s,skew_alpha:%s,'
                'dud_risk:%s}}' % (pid, pr["mean"], pr["std"],
                                   pr["skew_alpha"], pr["dud_risk"]))

    lineup_js = "[" + ",".join(entry(pid) for pid in a_pids) + "]"
    pids_js = "[" + ",".join('"%s"' % p for p in a_pids + b_pids) + "]"
    harness = lab_js + (
        "\nvar WV_LAB_SIMS = %d;\n"
        "wvLabData = { corr: %s };\n"
        "wvLabBase = wvLabBuildBase(%s, WV_LAB_SIMS, 42);\n"
        "wvLabLineup = %s;\n"
        "var rand = wvLabRng(42 + 7);\n"
        "var sp = wvLabSkewParams(%s, %s, 2.0);\n"
        "wvLabOppDraws = new Float64Array(WV_LAB_SIMS);\n"
        "for (var s = 0; s < WV_LAB_SIMS; s++) {\n"
        "  var a = rand() + 1e-12, b = rand();\n"
        "  var r = Math.sqrt(-2 * Math.log(a)), ang = 2 * Math.PI * b;\n"
        "  wvLabOppDraws[s] = Math.max(0, sp.xi + sp.omega * "
        "(sp.delta * Math.abs(r*Math.cos(ang)) + sp.w2 * r * Math.sin(ang)));\n"
        "}\n"
        "var res = wvLabEvaluate(wvLabLineup);\n"
        "console.log('LAB_WINPCT=' + res.winPct.toFixed(4));\n"
        % (sims, corr_js, pids_js, lineup_js, opp_mean, opp_std))
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        out = subprocess.run(["node", path], capture_output=True, text=True,
                             timeout=120)
    finally:
        os.unlink(path)
    assert out.returncode == 0, "node harness failed: %s" % out.stderr[-2000:]
    m = re.search(r"LAB_WINPCT=([\d.]+)", out.stdout)
    assert m, "no LAB_WINPCT in node output: %s" % out.stdout[-500:]
    return float(m.group(1)) * 100.0


def _page_win_pct(teams, a_idx, b_idx):
    def side(team):
        return {"starters": [{"pid": pid, "pos": pos, "pts": 0.0}
                             for pos, pid, _ in team]}
    status, proj_map = {}, {}
    for team in (teams[a_idx], teams[b_idx]):
        for pos, pid, mean in team:
            status[pid] = STATUS_NOT_STARTED
            proj_map[pid] = mean
    return compute_win_prob(side(teams[a_idx]), side(teams[b_idx]),
                            status, proj_map) * 100.0


@pytest.mark.parametrize("a_idx,b_idx", [(0, 1), (1, 2), (2, 3)])
def test_page_and_lab_agree_within_five_points(lab_js, real_profiles,
                                               a_idx, b_idx):
    """Same lineup, same projections, pre-game: the two models must rhyme."""
    teams, profiles = real_profiles
    page_pct = _page_win_pct(teams, a_idx, b_idx)
    lab_pct = _lab_win_pct(lab_js, teams, profiles, a_idx, b_idx)
    diff = abs(page_pct - lab_pct)
    assert diff <= 5.0, (
        "page=%.1f%% lab=%.1f%% (diff %.1f) on teams %d vs %d"
        % (page_pct, lab_pct, diff, a_idx, b_idx))
