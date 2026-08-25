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


def test_cpu_kicker_and_defense_timing_varies_by_team_plan():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    core = (REPO / "static" / "draft_board_core.js").read_text(encoding="utf-8")

    assert "function _simKDefPlan(slot)" in source
    assert "state.simKDefPlans[slot]" in source
    assert "var _kdPlan = _simKDefPlan(slot);" in source
    assert "DraftBoardCore.specialTeamsFillPos" in source
    assert "function specialTeamsFillPos(needK, needDef, plan)" in core
    assert "var _delayOther = _kdPlan.split && _alreadyHasOther && cpuCtx.remaining > 1;" in source
    assert "if ((pos === 'K' || pos === 'DEF') && (t > 0) && (have < t) && _remainRds <= 3){" not in source
    assert "w *= 8;" not in source
    assert "candidates.sort(function(a,b){ return lineupScore(b) - lineupScore(a); });\n    return candidates[0] || null;" not in source


def test_cpu_respects_format_aware_tight_end_roster_limit():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "DraftBoardCore.positionRosterLimit(pos, _rs" in source
    assert "if (have >= rosterLimit){ c.w = 0; c.ds = -1; return; }" in source


def test_mock_draft_uses_legal_fallback_instead_of_silently_stopping():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "function _fallbackLegalPick(pool, counts)" in source
    assert "if (!ap) ap = _fallbackLegalPick(availablePool(), myPosCounts());" in source
    assert "if (!p) p = _fallbackLegalPick(availablePool(), teamCounts(" in source
    assert "No legal players remain before the draft is complete" in source


def test_likely_next_pick_survivors_pay_current_pick_opportunity_cost():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "var LIVE_WAIT_TUNING = { threshold: 50, maxPenalty: 10 };" in source
    assert "var returnProb = nextPick ? availProb(p, nextPick) : null;" in source
    assert "c.demandByPos = _demandBeforeNext(next);" in source
    assert "var effectiveReturnProb = returnProb == null ? null : returnProb * (1 - demandRisk);" in source
    assert "waitPenalty: waitPenalty" in source


def test_autodraft_uses_shared_need_multiplier_instead_of_uncapped_starter_boost():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    core = (REPO / "static" / "draft_board_core.js").read_text(encoding="utf-8")

    assert "DraftBoardCore.autoDraftNeedMultiplier" in source
    assert "else if (sSlots > 0 && have < sSlots) s *= 1.35;" not in source
    assert "function autoDraftNeedMultiplier(o)" in core
    assert "var AUTO_WAIT_TURN = 0.35;" in core


def test_tier_cliff_urgency_is_suppressed_during_round_one():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    match = re.search(r"function isTierCliff\(p, pickNo\)\{(.*?)\n  \}", source, re.DOTALL)

    assert match
    assert "if (pn <= ((state && state.teams) || 12)) return false;" in match.group(1)
    # Live board still uses isTierCliff(); grading uses at-pick map / override.
    assert "else _cliff = isTierCliff(p, _pn);" in source
    assert "_buildGradeCliffs" in source
    assert "map[pn] = (pn > teams) && t != null && left <= 2;" in source


def test_roster_setup_has_editable_platform_and_dynasty_presets():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    body = build_draft_room_body(None, None, None, is_guest=True)

    # Assert on the meaningful preset content (label + roster slots), decoupled
    # from cosmetic whitespace and the type/ppr fields presets also carry, so a
    # reformat doesn't break the test while the presets are intact.
    assert "label:'ESPN'" in source
    assert "QB:1,SF:0,RB:2,WR:2,TE:1,FLEX:1,K:1,DEF:1,BN:8" in source  # ESPN slots
    assert "label:'Sleeper'" in source
    assert "label:'Yahoo'" in source
    assert "label:'Dynasty SF'" in source
    assert "data-roster-preset" in source
    assert ".dr-roster-preset {" in body


def test_roster_source_sits_outside_immediately_above_slot_grid():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "var html = presetHtml + srcHtml + '<div class=\"dr-setup-roster\">';" in source


def test_deep_dive_value_vs_adp_uses_consensus():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    body = build_draft_room_body(None, None, None, is_guest=True)

    assert "function consensusAdpOf(p)" in source
    assert "p.adp_by_source && p.adp_by_source.consensus" in source
    assert "var consAdp = consensusAdpOf(full);" in source
    assert "function ddTlDelta(p){ return p.consDiff != null ? p.consDiff : p.diff; }" in source
    assert "'<small class=\"dd-h-sub\">Consensus ADP</small>'" in source
    assert "Each pick against consensus ADP." in source
    assert ".dd-h-sub { display:inline-block; margin-left:8px;" in body


def test_pick_ledger_formats_adp_delta_to_one_decimal():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "function fmtAdpDelta(n)" in source
    assert "var dtxt = fmtAdpDelta(p.diff);" in source
    assert "(p.diff > 0 ? '+' : '') + p.diff;" not in source
    assert "fmtAdpDelta(netValue)" in source
    assert 'td class="r num"><span class="dd-diff' in source


def test_pick_ledger_adp_column_is_right_aligned_tabular():
    body = build_draft_room_body(None, None, None, is_guest=True)

    assert ".dd-ledger thead th.r, .dd-ledger tbody td.r { text-align:center; font-variant-numeric:tabular-nums; }" in body
    assert '.dd-diff { display:inline-block; min-width:6.2ch; text-align:right; font-weight:800;' in body
    assert 'font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1; }' in body


def test_summary_modal_keeps_footer_visible_and_roster_scrollable():
    body = build_draft_room_body(None, None, None, is_guest=True)

    assert "display:flex; flex-direction:column;" in body
    assert "max-height:min(620px, calc(100dvh - 48px));" in body
    assert ".dr-sum-body-wrap { padding:0 16px 4px; flex:1 1 auto; min-height:0; overflow-y:auto;" in body
    assert ".dr-sum-footer { display:flex; gap:8px; padding:12px 16px 14px; flex-shrink:0;" in body
    assert "max-height:min(78dvh, calc(100dvh - 16px));" in body
    assert ".dr-summary-overlay { position:fixed; inset:0; z-index:1001; background:rgba(0,0,0,.6);\n    display:flex; align-items:center; justify-content:center; overflow:hidden;" in body


def test_starters_meter_shows_percent_of_league_average():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "meter('Starters', '100% = league-average lineup', starterPct, 100, sRank, { unit: '% of avg', vsAvg: true })" in source
    assert "meter('Starters', 'lineup vs league average', g.tier, m.tier, sRank)" not in source
    assert "x.strength != null ? x.strength : x.tier" in source
    assert "Tied values share a rank" in source


def test_draft_capital_percentages_use_finite_numeric_value():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "function finiteVal(v)" in source
    assert "return finiteVal(v);" in source
    assert "function capPct(part, tot)" in source
    assert "var v = valOf(playersById[String(x.p.id)] || x.p) || 0;" not in source
    assert "lgByPos = lgCount; lgTot = lgN;" in source


def test_edit_setup_opens_a_modal_instead_of_leaving_the_board():
    body = build_draft_room_body(None, None, None, is_guest=True)
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert 'id="drEditTitle">Edit Setup</h2>' in body
    assert 'id="drEditApply">Apply Settings</button>' in body
    assert 'id="drEditReset">' in body
    assert 'id="drEditCancel">Cancel</button>' in body
    assert 'id="drSetupStartCta"' in body
    assert 'id="drSetupEditCta"' in body
    assert ".dr-setup-is-modal {" in body
    assert "function openEditSetup()" in source
    assert "function applyEditedSetup()" in source
    assert "function closeEditSetup()" in source
    assert "document.getElementById('drEdit').addEventListener('click', openEditSetup);" in source
    assert "document.getElementById('drEdit').addEventListener('click', showSetup);" not in source
    assert "document.getElementById('drEditReset').addEventListener('click', resetDraft);" in source
    assert "state = null;\n      showSetup();" in source
    assert "This wipes every pick and returns to setup." in source
    assert "if (!state || !state.teams || state.mode === 'live') return;" in source


def test_statusbar_shows_league_settings_chips():
    body = build_draft_room_body(None, None, None, is_guest=True)
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert 'id="drLeagueMeta"' in body
    assert ".dr-league-meta {" in body
    assert ".dr-lm-chip {" in body
    assert "function leagueMetaParts()" in source
    assert "function renderLeagueMeta()" in source
    assert "renderLeagueMeta();" in source
    assert "document.getElementById('drLeagueMeta').addEventListener('click'" in source
    assert "state.teams + '-team ' + (state.sf ? 'SF' : '1QB')" in source
    assert "el.classList.toggle('is-editable', canEdit);" in source
    # Live drafts lock settings; the chips still render but do not open Edit Setup.
    assert "var canEdit = state.mode !== 'live';" in source
