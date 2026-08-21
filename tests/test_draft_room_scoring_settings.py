"""Rendering guards for Draft Room scoring controls."""

from dashboard_services.pages.draft_room_page import build_draft_room_body
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_draft_room_offers_four_and_six_point_passing_touchdowns():
    body = build_draft_room_body(None, None, None, is_guest=True)

    assert 'id="drPassTd"' in body
    assert '<option value="4" selected>4 points</option>' in body
    assert '<option value="6">6 points</option>' in body


def test_league_scoring_is_available_to_live_and_mock_drafts():
    body = build_draft_room_body(
        "league", 2026, "sleeper", scoring={"ppr": 0.5, "tep": 0, "passTd": 6},
    )
    match = re.search(r"window\.__draftCfg = (.*?);</script>", body)
    assert match
    assert json.loads(match.group(1))["scoring"] == {"ppr": 0.5, "tep": 0, "passTd": 6}


def test_setup_source_and_draft_pick_pills_match_canonical_chip_styles():
    body = build_draft_room_body(None, None, None, is_guest=True)

    assert ".dr-roster-src-tag, .dr-cap-pill {" in body
    assert "background:var(--row); border:1px solid var(--grid); border-radius:6px; padding:2px 8px;" in body
    assert "color:var(--text-muted); font-size:11px; font-weight:700; line-height:1.45; white-space:nowrap;" in body
    assert ".dr-roster-src-tag { text-transform:none; letter-spacing:normal; }" in body
    assert "rgba(168,85,247,.14)" not in body


def test_player_load_failure_exposes_api_error_and_retry_control():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "Player API HTTP " in source
    assert "Player API returned non-JSON" in source
    assert "retry.addEventListener('click', loadPlayers)" in source
    assert "console.error('[draft-room] loadPlayers failed', err)" in source


def test_pick_reason_uses_its_own_current_pick_variable():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    match = re.search(r"function pickReason\(p, counts\)\{(.*?)\n  \}", source, re.DOTALL)

    assert match
    body = match.group(1)
    assert "var pickNo = (state && state.current) || 1;" in body
    assert "_pn" not in body


def test_board_offers_pick_score_sort_instead_of_steals():
    body = build_draft_room_body(None, None, None, is_guest=True)
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert 'data-val="pickscore">Pick Score</button>' in body
    assert 'data-val="steals"' not in body
    assert "if (sortBy === 'pickscore'){ return (b._ps || 0) - (a._ps || 0); }" in source
    assert "pickscore: 'Pick Score'" in source
    assert body.index('data-val="ps">Recommendation</button>') < body.index('data-val="pickscore">Pick Score</button>')
    assert body.index('data-val="pickscore">Pick Score</button>') < body.index('data-val="value">Value</button>')


def test_glossary_explains_live_recommendation_logic():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "{ term: 'Recommendation'" in source
    assert "starter or FLEX spot" in source
    assert "required slots and picks remaining" in source
    assert "expected availability at your next pick" in source
    assert "shown as a rank rather than a grade" in source


def test_recommendation_rows_use_compact_rank_and_reason_copy():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    body = build_draft_room_body(None, None, None, is_guest=True)

    assert "Decision ' + p._ds + ' · recommendation #" not in source
    assert 'dr-ba-recchip">#' in source
    assert "ppgNum.toFixed(1) + ' proj'" in source
    assert ".dr-ba-recchip {" in body


def test_pick_score_sort_controls_the_visible_score_chip():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "showPickScore: sortBy === 'pickscore'" in source
    assert "<small>PS</small>" in source
    assert "<small>REC</small>" in source
    assert "' · PS ' + ps" in source
    assert "' · Pick ' + ps" not in source


def test_recommendation_is_a_rank_not_a_declining_numeric_grade():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "var _isRec = opts.rank && p._ds != null;" in source
    assert "#' + opts.rank + '<small>REC</small>" in source
    assert "prepareDecisionDisplay" not in source


def test_position_filters_preserve_all_player_recommendation_rank():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "function rankedRecommendationPool()" in source
    assert "recommendationRanks[String(p.id)] = i + 1" in source
    assert "rank: recommendationRanks[String(p.id)]" in source
    assert "rank: i + 1 }\n        : { showPickScore" not in source


def test_cpu_never_drafts_kicker_or_defense_past_roster_capacity():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "(pos === 'K' || pos === 'DEF') && (t <= 0 || have >= t)" in source
    assert "c.w = 0; c.ds = -1; return;" in source
    assert "cands = cands.filter(function(c){ return c.w > 0; });" in source
    assert "function _cpuKDefMustFill(pool, counts, remaining)" in source
    assert "if (mustFillKDef) return mustFillKDef;" in source


def test_cpu_respects_format_aware_tight_end_roster_limit():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "DraftBoardCore.positionRosterLimit(pos, _rs" in source
    assert "if (have >= rosterLimit){ c.w = 0; c.ds = -1; return; }" in source


def test_likely_next_pick_survivors_pay_current_pick_opportunity_cost():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "var LIVE_WAIT_TUNING = { threshold: 50, maxPenalty: 10 };" in source
    assert "var returnProb = nextPick ? availProb(p, nextPick) : null;" in source
    assert "c.demandByPos = _demandBeforeNext(next);" in source
    assert "var effectiveReturnProb = returnProb == null ? null : returnProb * (1 - demandRisk);" in source
    assert "waitPenalty: waitPenalty" in source


def test_tier_cliff_urgency_is_suppressed_during_round_one():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    match = re.search(r"function isTierCliff\(p, pickNo\)\{(.*?)\n  \}", source, re.DOTALL)

    assert match
    assert "if (pn <= ((state && state.teams) || 12)) return false;" in match.group(1)
    assert "isTierCliff: isTierCliff(p, _pn)" in source


def test_roster_setup_has_editable_platform_and_dynasty_presets():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    body = build_draft_room_body(None, None, None, is_guest=True)

    assert "espn:    { label:'ESPN',       QB:1,SF:0,RB:2,WR:2,TE:1,FLEX:1,K:1,DEF:1,BN:8 }" in source
    assert "sleeper: { label:'Sleeper'" in source
    assert "yahoo:   { label:'Yahoo'" in source
    assert "dynasty: { label:'Dynasty SF'" in source
    assert "data-roster-preset" in source
    assert ".dr-roster-preset {" in body


def test_roster_source_sits_outside_immediately_above_slot_grid():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "var html = presetHtml + srcHtml + '<div class=\"dr-setup-roster\">';" in source
