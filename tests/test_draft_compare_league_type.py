"""Draft Room compare/preview stats follow league type the same way pos rank does."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
DRAFT_JS = (ROOT / "static" / "draft_room.js").read_text(encoding="utf-8")


def _slice(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i + 1)
    return src[i:j]


def test_draft_player_facts_switch_sf_fields():
    facts = _slice(DRAFT_JS, "function draftPlayerFacts(p)", "function fmtSigned")
    assert "var isSf = !!(state && state.sf);" in facts
    assert "var isRedraft = state && state.type === 'redraft';" in facts
    assert "valOf(p)" in facts
    assert "adpOf(p)" in facts
    assert "vorOf(p)" in facts
    assert "var pr = posRankOf(p);" in facts
    assert "posRank: pr.label" in facts
    assert "p.sf_pos_rank_label" not in facts
    assert "p.sf_vorp != null ? p.sf_vorp : p.vorp" in facts
    assert "p.sf_market_vs_adp != null ? p.sf_market_vs_adp : p.market_vs_adp" in facts
    assert "isRedraft ? null : (isSf ? p.sf_adp_n : p.adp_n)" in facts


def test_pos_rank_uses_board_value_not_dynasty_labels():
    helper = _slice(DRAFT_JS, "function refreshPosRankMap()", "function posRankOf")
    assert "valOf(b) - valOf(a)" in helper
    assert "state.sf ? 'sf' : '1qb'" in helper
    facts = _slice(DRAFT_JS, "function draftPlayerFacts(p)", "function fmtSigned")
    assert "posRankOf(p)" in facts
    cmp = _slice(DRAFT_JS, "function openCompare()", "function pickScore")
    assert "statRow('Pos Rank'" in cmp
    assert "p.pos_rank_label" not in cmp


def test_compare_modal_reads_draft_player_facts():
    cmp = _slice(DRAFT_JS, "function openCompare()", "function pickScore")
    assert "var f = draftPlayerFacts(p);" in cmp
    assert "statRow('Value', f.value" in cmp
    assert "statRow('ADP', f.adp" in cmp
    assert "statRow('VORP', f.vorp" in cmp
    assert "statRow('Pos Rank'" in cmp
    assert "statRow('Mkt vs ADP', f.market" in cmp
    # Raw 1QB fields must not bypass the league-type snapshot.
    assert "p.value" not in cmp or "draftPlayerFacts" in cmp
    assert "p.avg_pick" not in cmp
    assert "p.pos_rank_label" not in cmp


def test_draft_room_fetches_league_type():
    load = _slice(DRAFT_JS, "function loadPlayers()", "function applyKeepers")
    assert "params.push('league_type=' + (state && state.sf ? 'sf' : '1qb'))" in load
    assert "params.push('scoring_type='" in load
    assert "state.type === 'redraft' ? 'redraft'" in load
    assert "params.push('league_size='" in load


def test_league_players_stamps_sf_vorp_and_sf_market():
    overlay = _slice(APP_PY, "def _build_league_players_payload_uncached", "def api_market_intel_health")
    assert 'starters={"QB": 2.0}' in overlay
    assert '_player["sf_vorp"]' in overlay

    route = _slice(APP_PY, "def api_league_players():", "def api_teams():")
    assert 'request.args.get("league_type"' in route
    assert 'request.args.get("scoring_type"' in route
    assert "scoring_type=_mi_scoring" in route
    assert '["sf_market_vs_adp"]' in route
    assert '["market_vs_adp_1qb"]' in route
    assert "is_superflex=False" in route
    assert "is_superflex=True" in route
