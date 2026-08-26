"""Compare modal / page must follow league type the same way pos rank does."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

# Rebased onto main for fresh CI.

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")


def _slice(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i + 1)
    return src[i:j]


def test_compare_helpers_exist():
    assert "function _cmpIsSf()" in APP_JS
    assert "function _cmpHistValue(h)" in APP_JS
    assert "function _cmpPlayerDetailsUrl(pid)" in APP_JS
    assert "function _cmpPlayersSearchUrl(q)" in APP_JS
    assert "function _cmpDisplayValue(p)" in APP_JS
    assert "function _cmpLeagueParams()" in APP_JS


def test_compare_fetches_pass_league_type():
    details = _slice(APP_JS, "function _cmpPlayerDetailsUrl(pid)", "function _cmpPlayersSearchUrl")
    assert "_cmpLeagueParams()" in details
    assert "league_id=" in details

    search = _slice(APP_JS, "function _cmpPlayersSearchUrl(q)", "function _cmpDisplayValue")
    assert "/api/players?q=" in search
    assert "_cmpLeagueParams()" in search

    assert "fetch(_cmpPlayerDetailsUrl(pid))" in APP_JS
    assert "fetch(_cmpPlayersSearchUrl(q))" in APP_JS
    assert "fetch('/api/player-details/' + encodeURIComponent(pid))" not in APP_JS
    assert "fetch(`/api/players?q=${encodeURIComponent(q)}&limit=20`)" not in APP_JS


def test_compare_hero_and_adp_follow_league_type():
    hero = _slice(APP_JS, "function _buildCompareHeroHTML(p, other)", "function _hiResHeadshot")
    assert "const isSf = _cmpIsSf();" in hero
    assert "pm-hero-primary" in hero
    assert "${isSf ? cardSf + card1qb : card1qb + cardSf}" in hero
    assert "const dynPri = isSf ? dynSf : dyn1" in hero
    assert "const rdrPri = isSf ? rdrSf : rdr1" in hero
    assert "const adpSubLbl = isSf ? '1QB' : 'SF';" in hero


def test_compare_search_shows_league_type_value():
    search = _slice(APP_JS, "function openCompareSearch(player1Data)", "function _computeSeasonStats")
    assert "const rightVal = _cmpDisplayValue(p);" in search
    assert "p.value || '-'" not in search
    assert "p.stats.value" not in search or "_cmpDisplayValue" in search


def test_compare_charts_and_triple_follow_league_type():
    chart = _slice(APP_JS, "function _compareWireView(p1, p2)", "function renderCompareInline")
    assert "_cmpHistValue(h)" in chart
    assert "p.sf_value_band || p.value_band" in chart
    assert "_cmpIsSf()" in chart

    triple = _slice(APP_JS, "function renderCompareTriple(d1, d2, d3, hostEl)", "function _renderTripleValueChart")
    assert "const isSf = _cmpIsSf();" in triple
    assert "st(p).sf_value_ovr_rank" in triple
    assert "sf_pos_rank_label" in triple
    assert "st(p).adp.dynasty_sf" in triple
    assert "st(p).adp.redraft_sf" in triple
    assert "st(p).adp.dynasty_1qb" in triple

    triple_chart = _slice(APP_JS, "function _renderTripleValueChart(players)", "function openComparisonView")
    assert "_cmpHistValue(h)" in triple_chart


def test_compare_baselines_include_sf_value_band():
    baselines = _slice(APP_PY, "def _compute_compare_baselines():", "def api_player_details")
    assert '"sf_value_band"' in baselines
    assert '_val(i, "sf_value")' in baselines


def test_api_players_sorts_by_league_type():
    api = _slice(APP_PY, "def api_players():", "def api_sparklines")
    assert 'request.args.get("league_type"' in api
    assert '"sf_pos_rank_label"' in api
    assert 'sort_key = "sf_value" if is_sf else "value"' in api


@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
def test_cmp_helpers_switch_on_league_type():
    helpers = _slice(APP_JS, "function _cmpIsSf()", "function _buildCompareHeroHTML")
    driver = r"""
let leagueType = '1qb';
function brLeagueType() { return leagueType; }
function _wlLeagueParams() {
  return 'league_type=' + encodeURIComponent(brLeagueType()) + '&league_size=10';
}
const window = {
  location: { pathname: '/sleeper/2026/lg123/compare', search: '' },
  __brctx: { leagueType: '1qb', leagueId: 'lg123', platform: 'sleeper', season: '2026' },
};
%s
const hist = { value: 10, value_1qb: 11, value_sf: 22 };
const player = { value: 100, sf_value: 250, stats: { value: 101, sf_value: 251 } };
const baseline = { stats: { value: 80, sf_value: 180 } };

leagueType = '1qb';
const oneQb = {
  isSf: _cmpIsSf(),
  hist: _cmpHistValue(hist),
  val: _cmpDisplayValue(player),
  base: _cmpDisplayValue(baseline),
  details: _cmpPlayerDetailsUrl('4046'),
  search: _cmpPlayersSearchUrl('mahomes'),
};
leagueType = 'sf';
const sf = {
  isSf: _cmpIsSf(),
  hist: _cmpHistValue(hist),
  val: _cmpDisplayValue(player),
  base: _cmpDisplayValue(baseline),
  details: _cmpPlayerDetailsUrl('4046'),
  search: _cmpPlayersSearchUrl('mahomes'),
};
process.stdout.write(JSON.stringify({ oneQb, sf }));
""" % helpers
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)

    assert out["oneQb"]["isSf"] is False
    assert out["oneQb"]["hist"] == 11
    assert out["oneQb"]["val"] == 100
    assert out["oneQb"]["base"] == 80
    assert "league_type=1qb" in out["oneQb"]["details"]
    assert "league_id=lg123" in out["oneQb"]["details"]
    assert "league_type=1qb" in out["oneQb"]["search"]

    assert out["sf"]["isSf"] is True
    assert out["sf"]["hist"] == 22
    assert out["sf"]["val"] == 250
    assert out["sf"]["base"] == 180
    assert "league_type=sf" in out["sf"]["details"]
    assert "league_type=sf" in out["sf"]["search"]
    assert "/api/player-details/4046?" in out["sf"]["details"]
