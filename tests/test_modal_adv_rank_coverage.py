"""Pos-rank coverage for the player modal's Advanced tab.

Kaedon asked for the positional ranks back on EVERY advanced metric in the
player modal. The plumbing (ranks endpoint -> _rankSub -> .pm-comp-sub) was
intact, but coverage had holes on both sides:

- Frontend: the Slot Rate / Wide Rate / Inline Rate / Block Rate / YAC
  (season) tiles were built with no key and no rank sub at all, so they
  could never show a rank.
- Backend: get_player_metric_ranks never ranked slot_rate, wide_rate,
  inline_rate, pass_block_rate, yards_after_catch, or wopr (a stored
  column absent from the SQL rank list), so the endpoint had nothing to
  give the WOPR tile either.
- The ranks fetches also shared the generic 12s abort budget; the endpoint
  computes whole-position ranks live on a cold cache, so a slow first hit
  silently resolved to null and the tab rendered rankless.

These tests pin the restored coverage. TD/INT Ratio, Usage Trend and Eff
Trend intentionally stay rankless: they are JS-derived presentations with
no stored metric for the endpoint to rank.
"""

import inspect
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
MODAL_JS = (ROOT / "static" / "player_modal.js").read_text()
DASH_CSS = (ROOT / "static" / "dashboard.css").read_text()


def test_supp_gates_rank_the_modal_only_metrics():
    import data_building.advanced_metrics as am

    src = inspect.getsource(am.get_player_metric_ranks)
    for entry in (
        '"slot_rate": ("total_targets", 15)',
        '"wide_rate": ("total_targets", 15)',
        '"inline_rate": ("total_targets", 15)',
        '"pass_block_rate": ("games", 4)',
        '"yards_after_catch": ("total_receptions", 10)',
        '"wopr": ("games", 4)',
    ):
        assert entry in src, entry


def test_ranks_function_returns_the_new_metric_ranks(monkeypatch):
    """Behavioral: fake-conn with real-shape snapshot/supplement rows."""
    import data_building.advanced_metrics as am

    players = ["101", "102", "103", "104", "105"]
    snap_cols = (
        "total_carries total_targets total_receptions total_touches "
        "total_rush_tds total_rec_tds total_pass_tds total_tds "
        "carries_per_game targets_per_game receptions_per_game "
        "touches_per_game rush_tds_per_game rec_tds_per_game "
        "pass_tds_per_game total_tds_per_game role_score snap_share "
        "grades_offense opportunity_share red_zone_usage catch_rate"
    ).split()

    class Cur:
        def __init__(self, rows=None, one=None):
            self._rows = rows or []
            self._one = one

        def fetchone(self):
            return self._one

        def fetchall(self):
            return self._rows

    def snap_row(pid):
        d = {"player_id": pid}
        for c in snap_cols:
            d[c] = 3
        return d

    def supp_row(pid, i):
        # Player 101 leads every newly-ranked metric; values step down by i.
        return {
            "player_id": pid, "as_of_date": "2026-11-08", "games": 9,
            "total_targets": 100 - i * 10, "total_receptions": 60 - i * 5,
            "total_carries": 5, "total_touches": 105 - i * 10,
            "total_snaps": 500, "total_pass_att": 0, "total_rec_tds": 5,
            "total_tds": 5, "yards_per_reception": 12.0,
            "avoided_tackles": 3, "explosive_runs_10_plus": 1,
            "ngs_avg_separation": 3.0 + i * 0.1,
            "slot_rate": 0.60 - i * 0.05, "wide_rate": 0.30 + i * 0.05,
            "inline_rate": 0.10 + i * 0.02, "pass_block_rate": 0.05,
            "yards_after_catch": 400 - i * 40, "wopr": 0.60 - i * 0.04,
        }

    class FakeConn:
        def execute(self, sql, params=None):
            if "SELECT position, season FROM player_advanced_metrics" in sql:
                return Cur(one={"position": "WR", "season": 2026})
            if "WITH snapshot AS" in sql:
                return Cur(rows=[snap_row(p) for p in players])
            if "SELECT * FROM player_advanced_metrics" in sql:
                return Cur(rows=[supp_row(p, i) for i, p in enumerate(players)])
            raise AssertionError("unexpected SQL: " + sql[:80])

    @contextmanager
    def fake_conn(*a, **k):
        yield FakeConn()

    monkeypatch.setattr(am, "get_conn", fake_conn)
    am._POSITION_RANKS_CACHE.clear()
    am._POSITION_BOUNDS_CACHE.clear()

    out = am.get_player_metric_ranks("101", season=2026)
    ranks = out["ranks"]
    assert ranks["slot_rate"] == 1
    assert ranks["wopr"] == 1
    assert ranks["yards_after_catch"] == 1
    assert ranks["wide_rate"] == 5  # lowest slot share = highest wide share here
    assert ranks["inline_rate"] == 5
    assert ranks["pass_block_rate"] == 1  # tied across the pool, RANK semantics


def test_modal_defs_carry_rank_subs_for_every_tile():
    fn = MODAL_JS[MODAL_JS.index("function buildAdvancedMetricsHTML"):]
    for frag in (
        "key: 'slot_rate', sub: _rankSub('slot_rate')",
        "key: 'wide_rate', sub: _rankSub('wide_rate')",
        "key: 'inline_rate', sub: _rankSub('inline_rate')",
        "key: 'pass_block_rate', sub: _rankSub('pass_block_rate')",
        "key: 'yards_after_catch', sub: _rankSub('yards_after_catch')",
    ):
        assert frag in fn, frag


def test_ranks_fetches_get_a_30s_budget():
    # Only the two ranks fetches inside loadAdvancedMetrics; each is the
    # whole-position live compute that a 12s abort used to silently kill.
    assert "_advFetch(_rUrl, 30000)" in MODAL_JS
    assert "_advFetch(rankUrl, 30000)" in MODAL_JS


def test_advanced_section_header_links_to_am_page():
    # The header carries an explicit page link next to the (linked) title,
    # built from the same URL: the AM page reads pos/preset/search on load
    # and its search deep link filters the leaderboard to this player.
    assert "pm-adv-page-link" in MODAL_JS
    assert "const _advPageUrl" in MODAL_JS
    assert "Advanced Metrics page &#8250;" in MODAL_JS
    assert ".pm-adv-page-link" in DASH_CSS


@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
def test_node_render_shows_rank_subs_on_new_tiles(tmp_path):
    """Execute the real buildAdvancedMetricsHTML and check the subs render."""
    script = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
function sliceBetween(a, b) {
  const i = src.indexOf(a);
  const j = src.indexOf(b, i + a.length);
  if (i < 0 || j < 0) throw new Error('slice failed: ' + a);
  return src.slice(i, j);
}
eval(
  sliceBetween('const _ADV_METRIC_DESCS', '\nfunction buildAdvancedMetricsHTML') + '\n' +
  sliceBetween('function _pmAdvEmptyHTML', '\n// Per-category') + '\n' +
  sliceBetween('function buildAdvancedMetricsHTML', '\nfunction toggleGameLogYear')
);
const metricsData = { position: 'WR', season: 2026, metrics: {
  games: 9, snap_share: 0.82, target_share: 0.27, wopr: 0.55,
  slot_rate: 0.42, wide_rate: 0.55, yards_after_catch: 380,
  total_targets: 96, total_receptions: 61, total_touches: 102 } };
const ranks = { snap_share: 18, target_share: 7, wopr: 9,
  slot_rate: 12, wide_rate: 20, yards_after_catch: 6 };
const cfg = { snap_share: {label: 'Snap Share', category: 'General', basic: true},
  target_share: {label: 'Target Share', category: 'Receiving', basic: true},
  wopr: {label: 'WOPR', category: 'Receiving', basic: true} };
const html = buildAdvancedMetricsHTML(metricsData, ranks, cfg, false, {}, {}, null);
for (const want of ['(#18)', '(#7)', '(#9)', '(#12)', '(#20)', '(#6)']) {
  if (!html.includes(want)) { console.error('missing ' + want); process.exit(1); }
}
if (!html.includes('Ranked among qualified')) { console.error('no rank note'); process.exit(1); }
console.log('ok');
"""
    js_file = tmp_path / "probe.js"
    js_file.write_text(script)
    out = subprocess.run(
        ["node", str(js_file), str(ROOT / "static" / "player_modal.js")],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
