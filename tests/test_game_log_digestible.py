"""Digestible game log: verdict block, weekly finish Rank, pool-derived
washes, and the Detailed toggle.

Covers the server-side finish-rank helper (app._weekly_finish_ranks),
the renderer's column order / wash-step assignment / Detailed behavior
(driven through Node against the real _buildStatsHTML), and the source
contracts tying the JS wash constants to scripts/derive_game_log_tiers.py.
"""
from __future__ import annotations

import glob
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PM_SRC = (ROOT / "static" / "player_modal.js").read_text()
CSS_SRC = (ROOT / "static" / "dashboard.css").read_text()


# ── Source contracts ────────────────────────────────────────────────────

def test_verdict_block_column_order_in_spec():
    fant = PM_SRC[PM_SRC.index("const _FANT = ["):PM_SRC.index("const _PASS = [")]
    order = [fant.index(f"'{k}'") for k in ("pts", "rank", "snppct", "xfp")]
    assert order == sorted(order), "verdict block must lead Pts, Rank, SNP%, xFP"
    passing = PM_SRC[PM_SRC.index("const _PASS = ["):PM_SRC.index("const _RUSH = [")]
    assert passing.index("'patt'") < passing.index("'pcmp'") < passing.index("'pyd'")


def test_wash_constants_match_derivation():
    # Spot pins from scripts/derive_game_log_tiers.py output (Sleeper
    # weekly pool, 2025 + 2026 wks 1-6). If the tables are re-derived,
    # update these pins from the script's --js output in the same change.
    for pin in (
        "car: [21, 17, 9, 7]",              # RB carries
        "rypa: [6.43, 5.11, 3.12, 2.42]",   # RB yards per carry
        "snppct: [81.2, 70.9, 42.4, 31.9]",  # RB snap share
        "patt: [42, 37, 25, 20]",           # QB attempts
        "tar: [9, 6, 3, 2]",                # TE targets
        "rtd: [2, 1, null, null]",          # RB rushing TDs (top-tail only)
    ):
        assert pin in PM_SRC, pin
    # Rank finish bands and Kaedon's fixed xFP calibration bands.
    assert "v <= 5 ? 1 : v <= 12 ? 2 : v <= 24 ? 3 : 4" in PM_SRC
    assert "v >= 18 ? 1 : v >= 14 ? 2 : v < 5.5 ? 4 : v < 9 ? 3 : 0" in PM_SRC


def test_detailed_toggle_markup_and_css():
    start = PM_SRC.index('<span class="pm-section-label">Game Logs</span>')
    header = PM_SRC[start:start + 2000]
    # Detailed sits immediately before the Averages/Totals pill.
    assert header.index("st-detail-toggle") < header.index("gl-toggle")
    assert 'onchange="glSetDetailed(this)"' in PM_SRC
    assert "br-gamelog-detailed" in PM_SRC
    assert "data-cs-all" in PM_SRC and "data-cs-core" in PM_SRC
    assert ".game-log-table:not(.gl-show-detail) .gl-deep" in CSS_SRC
    assert ".gl-head-controls" in CSS_SRC
    assert ".grp-pill" in CSS_SRC and 'class="grp-pill"' in PM_SRC
    assert "#f4f6f8" in CSS_SRC  # group band tint from the approved mock


def test_wash_css_uses_schedule_rankings_palette():
    for step, hexv in ((1, "#22c55e"), (2, "#84cc16"), (3, "#f59e0b"), (4, "#ef4444")):
        rule = re.search(rf"\.gl-adv td\.gl-w{step} \{{([^}}]+)\}}", CSS_SRC)
        assert rule, f"gl-w{step} rule missing"
        assert f"color: {hexv}" in rule.group(1)
        assert f"background-color: {hexv}18" in rule.group(1)


def test_derivation_script_documents_pool():
    src = (ROOT / "scripts" / "derive_game_log_tiers.py").read_text()
    assert "sleeper_stats" in src and "mid-percentile" in src
    assert "RENDERED" in src


@pytest.mark.skipif(
    not glob.glob(str(ROOT / "cache" / "sleeper_stats" / "sleeper_stats_s2025_w*.json"))
    or not (ROOT / "cache" / "players_index.json").exists(),
    reason="weekly stats cache not present in this checkout",
)
def test_derivation_script_runs_on_checkout_cache():
    # Smoke test only: the exact cutoffs depend on the cache's fetch
    # vintage (week files land over a season), so value-for-value
    # reproduction is pinned in test_wash_constants_match_derivation
    # against the derivation the constants were taken from. Here the
    # script just has to run and emit a full four-position table set.
    out = subprocess.run(
        ["python3", str(ROOT / "scripts" / "derive_game_log_tiers.py"), "--js"],
        capture_output=True, text=True, timeout=120, check=True,
    ).stdout
    assert out.startswith("const _GL_WASH = {")
    for pos in ("QB:", "RB:", "WR:", "TE:"):
        assert pos in out


# ── Finish-rank helper ──────────────────────────────────────────────────

SCORING = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
    "rush_yd": 0.1, "rush_td": 6.0, "rec": 1.0,
    "rec_yd": 0.1, "rec_td": 6.0, "fum_lost": -2.0,
}


@pytest.fixture()
def rank_env(monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("flask")
    pytest.importorskip("openai")
    import app as appmod

    rows = {
        "a1": {"rush_yd": 100, "rush_td": 1},   # 16.0
        "a2": {"rush_yd": 160},                  # 16.0 (tie)
        "a3": {"rush_yd": 50},                   # 5.0
        "a4": None,                              # played, zero line -> 0.0
        "q1": {"pass_yd": 300, "pass_td": 2},    # 20.0
        "q2": {"pass_yd": 100},                  # 4.0
        "k1": {"rush_yd": 500},                  # K: never ranked
    }
    calls = {"n": 0}

    def fake_rows(season, week):
        calls["n"] += 1
        return rows

    monkeypatch.setattr(appmod, "_week_stat_index_rows", fake_rows)
    appmod._WEEK_FINISH_RANK_CACHE.clear()
    positions = {"a1": "RB", "a2": "RB", "a3": "RB", "a4": "RB",
                 "q1": "QB", "q2": "QB", "k1": "K"}
    return appmod, positions, calls


def test_finish_ranks_order_ties_and_pools(rank_env):
    appmod, positions, _ = rank_env
    ranks = appmod._weekly_finish_ranks(2026, 5, SCORING, positions)
    # Independent expectation, hand-scored from the fixture stats with
    # SCORING's rates (0.1/rush yd, 6/rush TD, 0.04/pass yd, 4/pass TD):
    # RB pool 16.0 / 16.0 / 5.0 / 0.0 -> competition ranks 1, 1, 3, 4.
    assert ranks == {"a1": 1, "a2": 1, "a3": 3, "a4": 4, "q1": 1, "q2": 2}


def test_finish_ranks_cached_per_scoring(rank_env):
    appmod, positions, calls = rank_env
    appmod._weekly_finish_ranks(2026, 5, SCORING, positions)
    appmod._weekly_finish_ranks(2026, 5, SCORING, positions)
    assert calls["n"] == 1
    other = dict(SCORING, rec=0.5)
    appmod._weekly_finish_ranks(2026, 5, other, positions)
    assert calls["n"] == 2


def test_finish_ranks_missing_week_is_empty(rank_env, monkeypatch):
    appmod, positions, _ = rank_env
    monkeypatch.setattr(appmod, "_week_stat_index_rows", lambda s, w: {})
    appmod._WEEK_FINISH_RANK_CACHE.clear()
    assert appmod._weekly_finish_ranks(2026, 6, SCORING, positions) == {}


# ── Renderer behavior (Node) ────────────────────────────────────────────

pytestmark_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available")


def _gamelog_slice():
    start = PM_SRC.index("window._glAvgTot = window._glAvgTot || 'avg';")
    end_marker = "window.glSetAvgTot = function (mode, el) {"
    end = PM_SRC.index(end_marker)
    depth, i = 0, end
    while True:
        if PM_SRC[i] == "{":
            depth += 1
        elif PM_SRC[i] == "}":
            depth -= 1
            if depth == 0:
                i += 1
                break
        i += 1
    return PM_SRC[start:i]


def _zero_stats(**over):
    stats = {"pass_yd": 0, "pass_td": 0, "pass_int": 0, "pass_att": 0,
             "pass_cmp": 0, "rush_att": 0, "rush_yd": 0, "rush_td": 0,
             "rec": 0, "rec_tgt": 0, "rec_yd": 0, "rec_td": 0, "fum_lost": 0}
    stats.update(over)
    return stats


RB_GAMES = [
    {"week": 2, "date": "20250913", "opponent": "NO", "fantasy_pts": 33.6,
     "rank": 3,
     "stats": _zero_stats(rush_att=29, rush_yd=156, rush_td=1,
                          rec=4, rec_tgt=5, rec_yd=40),
     "adv": {"snap_pct": 74, "target_share": 14.3, "carry_share": 100,
             "rz_targets": 1, "rz_carries": 6, "adot": 3.1, "ybc": 2.4,
             "xfp": 29.1, "btk": 3, "wopr": 0.52}},
    {"week": 10, "date": "20251108", "opponent": "@MIA", "fantasy_pts": 13.4,
     "rank": 27,
     "stats": _zero_stats(rush_att=15, rush_yd=46, rec_tgt=1),
     "adv": {"snap_pct": 69, "target_share": 3.0, "carry_share": 59,
             "rz_targets": 0, "rz_carries": 0, "adot": -1.0, "ybc": 1.1,
             "xfp": 15.8, "btk": 0, "wopr": 0.2}},
    {"week": 8, "date": "20251026", "opponent": "BYE", "fantasy_pts": None,
     "stats": None, "is_bye": True},
    {"week": 11, "date": "20251115", "opponent": "NE", "fantasy_pts": 24.3,
     "stats": None, "is_projection": True},
]

QB_GAMES = [
    {"week": 9, "date": "20251101", "opponent": "MIN", "fantasy_pts": 24.8,
     "rank": 8,
     "stats": _zero_stats(pass_att=38, pass_cmp=27, pass_yd=290, pass_td=2,
                          pass_int=1, rush_att=3, rush_yd=12),
     "adv": {"snap_pct": 100, "xfp": 22.7}},
]


def _render():
    driver = """
global.window = global;
global.localStorage = { _s: {}, getItem: function (k) { return k in this._s ? this._s[k] : null; },
  setItem: function (k, v) { this._s[k] = String(v); } };
global.document = { addEventListener: function () {}, querySelectorAll: function () { return []; } };
global.requestAnimationFrame = function () { return 0; };
global.fmtPts = function (v) { return (Math.round(v * 10) / 10).toFixed(1); };
""" + _gamelog_slice() + """
const RB = { "2026": %s };
const QB = { "2026": %s };
const out = {};
out.rbOff = _buildStatsHTML(RB, false, 'RB', {}, {});
window._glDetailed = true;
out.rbOn = _buildStatsHTML(RB, false, 'RB', {}, {});
window._glDetailed = false;
out.qb = _buildStatsHTML(QB, false, 'QB', {}, {});
process.stdout.write(JSON.stringify(out));
""" % (json.dumps(RB_GAMES), json.dumps(QB_GAMES))
    res = subprocess.run(["node", "-e", driver], capture_output=True,
                         text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _grp_headers(html):
    m = re.search(r'<tr class="grp">(.*?)</tr>', html, re.S)
    ths = re.findall(r'<th colspan="(\d+)"(?: class="([^"]*)")?([^>]*)>', m.group(1))
    return [(int(cs), cls, attrs) for cs, cls, attrs in ths]


def _col_headers(html):
    m = re.search(r'<tr class="cols">(.*?)</tr>', html, re.S)
    return [re.sub(r"<[^>]+>", "", t).strip()
            for t in re.findall(r"<th[^>]*>(.*?)</th>", m.group(1), re.S)]


def _body_rows(html):
    tbody = re.search(r"<tbody>(.*?)</tbody>", html, re.S).group(1)
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", tbody, re.S):
        cells = [(cls, re.sub(r"<[^>]+>", "", txt).strip())
                 for cls, txt in re.findall(
                     r'<td[^>]*class="([^"]*)"[^>]*>(.*?)</td>', row, re.S)]
        rows.append(cells)
    return rows


def _wash(cls):
    m = re.search(r"gl-w(\d)", cls)
    return int(m.group(1)) if m else 0


@pytestmark_node
def test_rb_verdict_order_and_washes():
    html = _render()["rbOff"]
    assert _col_headers(html) == [
        "Date", "Opp", "Pts *", "Rank", "SNP%", "xFP",
        "Car", "Yd", "Yd/A", "Car%", "TD", "BTK", "RZ Car", "YBC",
        "Tar", "Rec", "Yd", "TD", "YPT", "TS%", "ADOT", "RZ Tar", "WOPR",
    ]
    rows = _body_rows(html)
    great, bad, bye, proj = rows[0], rows[1], rows[2], rows[3]
    # Verdict block: Pts / Rank / SNP% / xFP steps.
    assert _wash(great[2][0]) == 1 and great[2][1] == "33.6"     # Pts
    assert _wash(great[3][0]) == 1 and great[3][1] == "3"        # Rank 3
    assert _wash(great[4][0]) == 2 and great[4][1] == "74"       # SNP% 74
    assert _wash(great[5][0]) == 1 and great[5][1] == "29.1"     # xFP
    assert _wash(great[6][0]) == 1 and great[6][1] == "29"       # Car 29
    assert _wash(great[8][0]) == 2 and great[8][1] == "5.38"     # Yd/A
    assert _wash(great[10][0]) == 2 and great[10][1] == "1"      # Rush TD 1
    assert _wash(great[17][0]) == 0 and great[17][1] == "0"      # Rec TD 0: top-tail only
    assert _wash(bad[2][0]) == 3                                 # Pts 13.4
    assert _wash(bad[3][0]) == 4 and bad[3][1] == "27"           # Rank 27
    assert _wash(bad[4][0]) == 0 and bad[4][1] == "69"           # SNP% mid: plain
    assert _wash(bad[6][0]) == 0 and bad[6][1] == "15"           # Car mid: plain
    assert _wash(bad[8][0]) == 3 and bad[8][1] == "3.07"         # Yd/A bad tail
    assert _wash(bad[14][0]) == 3 and bad[14][1] == "1"          # Tar 1
    assert _wash(bad[15][0]) == 4 and bad[15][1] == "0"          # Rec 0
    assert _wash(bad[16][0]) == 3 and bad[16][1] == "0"          # Rec Yd 0
    assert all(_wash(c) == 0 for c, _ in bye), "bye row must be unwashed"
    assert all(_wash(c) == 0 for c, _ in proj), "projected row must be unwashed"
    assert proj[3][1] == "-", "projected row has no finish rank"


@pytestmark_node
def test_rb_detailed_toggle_states():
    out = _render()
    off, on = out["rbOff"], out["rbOn"]
    assert '<table class="game-log-table gl-adv">' in off
    assert '<table class="game-log-table gl-adv gl-show-detail">' in on
    grp_off = _grp_headers(off)
    grp_on = _grp_headers(on)
    assert [g[0] for g in grp_off] == [2, 4, 4, 4, 1]
    assert [g[0] for g in grp_on] == [2, 4, 8, 8, 1]
    # Rushing header carries the swap values; the all-deep Usage header
    # hides with its columns while Detailed is off.
    assert 'data-cs-all="8"' in grp_off[2][2] and 'data-cs-core="4"' in grp_off[2][2]
    assert "gl-deep" in grp_off[4][1]
    assert "gl-deep" not in grp_on[4][1]
    # A deep body cell (Car%) is stamped gl-deep in both states.
    assert "gl-deep" in _body_rows(off)[0][9][0]


@pytestmark_node
def test_qb_verdict_and_passing_columns():
    html = _render()["qb"]
    assert _col_headers(html) == [
        "Date", "Opp", "Pts", "Rank", "SNP%", "xFP",
        "Att", "Cmp", "Yd", "Y/A", "TD", "INT",
        "Car", "Yd", "Yd/A", "TD",
    ]
    row = _body_rows(html)[0]
    assert _wash(row[3][0]) == 2 and row[3][1] == "8"      # Rank 8 -> step 2
    assert _wash(row[4][0]) == 0 and row[4][1] == "100"    # QB SNP%: no table, plain
    assert _wash(row[5][0]) == 1 and row[5][1] == "22.7"   # xFP
    assert _wash(row[6][0]) == 2 and row[6][1] == "38"     # Att
    assert _wash(row[7][0]) == 2 and row[7][1] == "27"     # Cmp
    assert _wash(row[8][0]) == 2 and row[8][1] == "290"    # Pass Yd
    assert _wash(row[9][0]) == 0                           # Y/A mid: plain
    assert _wash(row[11][0]) == 0 and row[11][1] == "1"    # INT 1: plain
    # Rank has no season total.
    tfoot = re.search(r"<tfoot>(.*?)</tfoot>", html, re.S).group(1)
    tcells = [re.sub(r"<[^>]+>", "", t).strip()
              for t in re.findall(r"<td[^>]*>(.*?)</td>", tfoot, re.S)]
    assert tcells[3] == "-"
