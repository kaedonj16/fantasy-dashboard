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


def test_draft_room_scoring_defaults_to_full_ppr():
    body = build_draft_room_body(None, None, None, is_guest=True)
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert '<option value="1" selected>Full PPR</option>' in body
    assert '<option value="0.5">Half PPR</option>' in body
    # Roster presets may change the league shape, but should retain the Draft
    # Room's full-PPR default. The explicitly named Standard preset is the only
    # intentional non-PPR exception.
    for key in ("espn", "sleeper", "yahoo", "sfredraft", "bestball", "dynasty", "dynasty1q"):
        assert re.search(rf"{key}:\s+\{{[^\n]+ppr:1,", source)
    assert re.search(r"standard:\s+\{[^\n]+ppr:0,", source)


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


def test_keeper_banner_pages_five_and_counts_yours_by_roster():
    """Keepers Details lists 5 per page; 'yours' is ownership (viewer roster),
    not the old !projected flag that left assistant-mode keepers at 0 yours."""
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    css = (REPO / "static" / "dashboard.css").read_text(encoding="utf-8")

    assert "var KEEPER_PAGE_SIZE = 5;" in source
    assert "function isMyKeeper(k)" in source
    assert "keeperSet.filter(isMyKeeper)" in source
    assert "drKeeperPrev" in source and "drKeeperNext" in source
    assert "dr-keeper-pager" in source
    assert "dr-keeper-items" in css
    # Ownership must not rely solely on !k.projected (that counted every
    # assistant-projected own-roster keeper as projected).
    banner = re.search(r"function renderKeeperBanner\(\)\{(.*?)\n  \}", source, re.DOTALL)
    assert banner
    assert "filter(function(k){ return !k.projected; })" not in banner.group(1)
    assert "isMyKeeper" in banner.group(1)


def test_player_load_failure_exposes_api_error_and_retry_control():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "Player API HTTP " in source
    assert "Player API returned non-JSON" in source
    assert "retry.addEventListener('click', loadPlayers)" in source
    assert "console.error('[draft-room] loadPlayers failed', err)" in source


def test_pick_reason_uses_its_own_current_pick_variable():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    match = re.search(r"function pickReason\(p, counts, opts\)\{(.*?)\n  \}", source, re.DOTALL)

    assert match
    body = match.group(1)
    assert "var pickNo = (state && state.current) || 1;" in body
    assert "_pn" not in body
    # Steal math stays on the live clock so a pick-9 look-ahead does not call
    # ADP 1.7 an "Elite steal" before anyone has been drafted.
    assert "Math.round(pickNo - adp)" in body
    # "Best available" is the #1 rec, not the fallback for every row.
    assert "if (opts.rank === 1)" in body
    assert "return 'Strong remaining value';" in body
    assert "Gone before #" in body
    assert "Best available at #" in body
    assert "1st-round talent" in body


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
    assert "When it is not your turn, the order is for your next owned pick" in source


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


def test_compare_modal_uses_relative_pick_score():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    match = re.search(
        r"function openCompare\(\)\{(.*?)document\.getElementById\('drCompare'\)\.style\.display",
        source,
        re.DOTALL,
    )

    assert match
    body = match.group(1)
    assert "function psRelLive(p)" in source
    assert "return psDisplay(p._ps != null ? p._ps : pickScoreFor(p));" in source
    assert "var ps = psRelLive(p);" in body
    assert "psAbs(" not in body
    assert "class=\"dr-cmp-player\"" in body
    assert "statRow('VOR'" in body
    assert "statRow('VORP'" in body
    assert "p.vorp != null || other.vorp != null" not in body
    assert "not a count of which compare rows you win" in body
    # Sidebar rows share the same relative chip so the modal cannot disagree.
    row = re.search(r"function playerRowHtml\(p, opts\)\{(.*?)function renderQueue", source, re.DOTALL)
    assert row
    assert "var ps = psRelLive(p);" in row.group(1)


def test_deep_dive_avg_pick_score_uses_relative():
    """Deep Dive 'Avg pick score' must average relPS chips, not absolute kernel."""
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    match = re.search(
        r"function gradePicks\(mine\)\{(.*?)\n  // At-pick tier-cliff map",
        source,
        re.DOTALL,
    )
    assert match
    body = match.group(1)
    assert "var relVals = mine.map(function(m){ return relPS(m.p, m.pn); })" in body
    assert "var avgPs = relVals.length" in body
    # Absolute kernel scores still feed the letter-grade composite — don't
    # accidentally average those for the display chip again.
    assert "picks.map(function(x){ return x.ps; })" not in body
    assert "{ v: g.avgPs != null ? g.avgPs : '—', l: 'Avg pick score' }" in source
    assert "Deep Dive’s Avg pick score, use the same relative scale" in source


def test_preview_modal_uses_relative_pick_score():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    match = re.search(
        r"function openPreview\(id\)\{(.*?)document\.getElementById\('drPreview'\)\.style\.display",
        source,
        re.DOTALL,
    )

    assert match
    body = match.group(1)
    assert "ps = psRelLive(p);" in body
    assert "psAbs(" not in body
    assert "function psAbs" not in source
    assert "dr-prev-score-hero" in body
    assert "function draftPlayerFacts(p)" in source
    facts = source[source.index("function draftPlayerFacts(p)"):source.index("function fmtSigned")]
    assert "var isSf = !!(state && state.sf);" in facts
    assert "p.sf_vorp != null ? p.sf_vorp : p.vorp" in facts
    assert "p.sf_market_vs_adp != null ? p.sf_market_vs_adp : p.market_vs_adp" in facts
    assert "var pr = posRankOf(p);" in facts
    assert "posRank: pr.label" in facts
    assert "var tradeVor = vorOf(p);" in facts
    assert "p.vorp != null ? Number(p.vorp) : vorOf(p)" not in source
    assert "statBox('Proj PPG'" in body
    assert "statBox('REC'" in body
    assert "statBox('Bye'" in body
    assert "statRow('vs ADP'" in source
    assert "statRow('Survive'" in source


def test_league_players_uses_sleeper_only_for_proj_ppg():
    source = (REPO / "app.py").read_text(encoding="utf-8")

    assert "fetch_sleeper_season_projections" in source
    assert "fetch_sleeper_season_ppg_variants" in source
    assert "Sleeper projected PPG fill skipped" in source
    assert '_player["proj_ppg_by"]' in source
    assert "unprojected_season_injury" in source
    assert '_player["proj_ppg"] = 0.0' in source
    assert "fp_projections_" not in source.split("def _build_league_players_payload_uncached")[1].split("def api_league_players")[0]


def test_draft_room_shows_scoring_adjusted_proj_ppg():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    body = build_draft_room_body(None, None, None, is_guest=True)

    assert "function scoringProjPpg(p)" in source
    assert "DraftBoardCore.scoringProjPpg(full, scoringCfg())" in source
    assert "DraftBoardCore.ppgOf(full, scoringCfg())" in source
    assert "var proj = scoringProjPpg(p);" in source
    assert "var ppgNum = scoringProjPpg(p);" in source
    assert "select the projected PPG variant" in source
    assert 'title="Projected PPG uses this reception scoring' in body
    assert 'title="Adjusts quarterback projected PPG, recommendations, and pick grades"' in body


def test_draft_room_roster_projection_uses_sleeper_proj_ppg_only():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "function _pPpg(p){ return scoringProjPpg(p); }" in source
    assert "var _ppgv = scoringProjPpg(s.p);" in source
    assert "p.proj_ppg != null ? Number(p.proj_ppg) : (p.ppg != null ? Number(p.ppg) : null)" not in source
    assert "p.ppg != null ? Number(p.ppg) : scoringProjPpg" not in source


def test_draft_summary_proj_ppg_is_starting_lineup_not_full_roster():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    match = re.search(
        r"function openSummary\(\)\{(.*?)function closeSummary",
        source,
        re.DOTALL,
    )
    assert match
    body = match.group(1)
    assert "starters.forEach(function(s){" in body
    assert "var _ppgv = scoringProjPpg(s.p);" in body
    assert "if (_ppgv != null){ sumProjTotal += _ppgv; sumProjCount++; }" in body
    assert "l: 'Proj PPG'" in body
    # Full-roster sum was inflating a start-9 to ~20 PPG per starter.
    assert "var _ppgv = scoringProjPpg(p);" not in body
    assert "mine.forEach(function(p){\n        var _ppgv = scoringProjPpg(p);" not in body


def test_recommendation_is_a_rank_not_a_declining_numeric_grade():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "var _isRec = opts.rank && p._ds != null;" in source
    assert "#' + opts.rank + '<small>REC</small>" in source
    assert "prepareDecisionDisplay" not in source


def test_position_filters_preserve_all_player_recommendation_rank():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "function rankedRecommendationPool()" in source
    assert "recommendationRanks[String(p.id)] = i + 1" in source
    assert "_rank = recommendationRanks[String(p.id)]" in source
    assert "rank: recommendationRanks[String(p.id)]" not in source
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
    # Wait target is the pick AFTER the rec pick, so waiting for #9 does not
    # treat pick 9 itself as "can wait until then".
    assert "var nextPick = recWaitPickNo();" in source
    assert "function recWaitPickNo()" in source
    assert "function recommendationPickNo()" in source
    assert "DraftBoardCore.futurePickDecisionScore(score, availProb(p, recPn))" in source


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
    # Consensus is the 'consensus' case of the generalized per-source lookup that
    # also backs the setup's "CPU drafts from" selector; it still reads the
    # blended column from the per-source payload.
    assert "var by = p.adp_by_source && p.adp_by_source[source];" in source
    assert "function consensusAdpOf(p){ return adpBySource(p, 'consensus'); }" in source
    assert "var consAdp = consensusAdpOf(full);" in source
    assert "if (p.consBoardDiff != null) return p.consBoardDiff;" in source
    assert "'<small class=\"dd-h-sub\">Consensus ADP</small>'" in source
    assert "Each pick against consensus ADP." in source
    assert ".dd-h-sub { display:inline-block; margin-left:8px;" in body


def test_cpu_drafts_from_source_selector():
    """The setup card exposes a 'CPU drafts from' selector (default consensus)
    and the CPU pick engine resolves that source before falling back."""
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    body = build_draft_room_body(None, None, None, is_guest=True)

    # Setup control renders with consensus as the default option.
    assert 'id="drCpuAdpSource"' in body
    assert '<option value="consensus" selected>Consensus (all platforms)</option>' in body
    for src in ("sleeper", "brfantasy", "espn", "mfl", "yahoo"):
        assert '<option value="%s">' % src in body

    # Setup reads the field into state and hydrates it back for the Edit modal.
    assert "cpuAdpSource: (document.getElementById('drCpuAdpSource') || {}).value || 'consensus'" in source
    assert "syncCpuAdpSourceOptions(state.cpuAdpSource);" in source

    # simAdp honors the chosen source, then consensus, then adpOf().
    assert "var src = (state && state.cpuAdpSource) || 'consensus';" in source
    assert "var a = adpBySource(p, src);" in source
    assert "if (a == null && src !== 'consensus') a = consensusAdpOf(p);" in source


def test_cpu_drafts_from_options_filtered_by_draft_type():
    """The CPU-source selector is filtered per draft type: redraft offers the
    global platforms, dynasty/rookie only Sleeper + BR Fantasy, consensus always."""
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    # Static fallback mirrors the server's ADP_SOURCES (redraft gets the globals;
    # dynasty/rookie do not); keeper maps to redraft; consensus leads every list.
    assert "startup: ['consensus', 'sleeper', 'brfantasy']" in source
    assert "rookie:  ['consensus', 'sleeper', 'brfantasy']" in source
    assert "redraft: ['consensus', 'sleeper', 'espn', 'yahoo', 'mfl', 'brfantasy']" in source
    assert "if (t === 'keeper') t = 'redraft';" in source
    # Prefers the payload's season-gated list when a pool has loaded.
    assert "adpSourceOptions && adpSourceOptions[t] && adpSourceOptions[t].length" in source
    # Rebuilt on draft-type change and on init.
    assert "syncCpuAdpSourceOptions();   // valid CPU sources depend on the draft type" in source
    assert "syncCpuAdpSourceOptions();   // filter CPU-source options to the initial draft type" in source


def test_random_pick_slot_option_and_resolution():
    """The pick selector offers Random, which resolves to a concrete slot at
    draft start and seeds that seat's natural draft capital."""
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    # fillSlotOptions prepends a Random option and never silently switches a
    # numbered pick to Random on a team-count change.
    assert "rnd.value = 'random'; rnd.textContent = '\U0001f3b2 Random';" in source
    assert "sel.value = (prev === 'random' || (pn >= 1 && pn <= teams)) ? prev : '1';" in source

    # readSetup resolves Random to a concrete 1..teams slot and flags it.
    assert "var randomSlot = slotRaw === 'random';" in source
    assert "(1 + Math.floor(Math.random() * teams))" in source
    assert "randomSlot: randomSlot," in source

    # Both start paths seed the resolved seat's capital via the shared helper.
    assert "function _seedOwnedForRandomSlot(st){" in source
    assert source.count("_seedOwnedForRandomSlot(state);") == 2

    # The capital editor shows a note (not Pick 1's picks) while Random is chosen.
    assert "Your pick is random" in source


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


def test_deep_dive_reach_uses_remaining_adp_and_survival():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    core = (REPO / "static" / "draft_board_core.js").read_text(encoding="utf-8")

    assert "var ADP_REACH_SURVIVE = 20;" in core
    assert "var ADP_REACH_CLUSTER = 1.0;" in core
    assert "function adpDeltaVerdict(o)" in core
    assert "function adpBoardDelta(o)" in core
    assert "function isRemainingAdpBpa(playerAdp, bestRemainingAdp, cluster)" in core
    assert "Core.bestRemainingAdp(remPool, taken, adpOf)" in source
    assert "ddSurvivePct(full, nextOwnedPickAfter(pn))" in source
    assert "ddVerdict(p).cls === 'reach'" in source
    assert "under 20% to last to your next pick" in source
    assert "Best remaining ADP and players under 20%" in source


def test_deep_dive_includes_descriptive_historical_trends():
    """Redraft Deep Dive shows Hist vs ADP-bucket rates without mixing into grades."""
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    body = build_draft_room_body(None, None, None, is_guest=True)
    grade = source.split("function gradePicks(mine){")[1].split("function gradeAllTeams")[0]

    assert "function ddHistHtml" in source
    assert "html += ddHistHtml(picks)" in source
    assert "Two groups per pick: players like this, and anyone taken in that ADP round." in source
    assert "Early ADP is a high bar, not a miss." in source
    assert "Lead with the gap between them." not in source
    assert "They are not averaged into one chance." not in source
    assert "Descriptive only. Not a ranking, Pick Score, or Draft Grade input." not in source
    assert "Hist below ADP bucket" not in source
    assert "Historical miss vs market" not in source
    assert "ADP round is a higher bar" in source
    assert "Hist group higher" in source
    assert "Avg pts vs ADP" in source
    assert "Avg vs ADP (info)" not in source
    assert "Avg vs ADP bucket" not in source
    assert "Strong profiles (25%+)" not in source
    assert "DD_HIST_STRONG" not in source
    assert "DD_HIST_EDGE" not in source
    assert "dd-hist-stats" in source
    assert "dd-hist-callout" in source
    assert "dd-hist-compare" in source
    assert "function ddHistVsClass" in source
    assert "is-bar" in source
    assert 'data-k="hist"' in source
    assert "ddHistPct(p)" in source
    assert "p_hit_pct" in source
    assert "historicalAvailable" in source
    assert "p_hit_pct" not in grade
    assert "historical.p_hit" not in grade
    assert ".dd-hist-pct" in body
    assert ".dd-hist-stats" in body
    assert ".dd-hist-callout.is-bar" in body
    assert ".dd-hist-vs.is-up" in body
    assert "{ term: 'Hist'" in source
    from dashboard_services.changelog import CHANGELOG
    entry = next(
        item for item in CHANGELOG
        if "deep dive" in item.get("text", "").lower()
        and "historical top-12" in item.get("text", "").lower()
    )
    assert entry["tag"] == "update"
    assert entry["link"] == "/draft"
    assert "descriptive only" in entry["text"].lower()
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]
    groups = next(
        item for item in CHANGELOG
        if "lower hist than the adp round is not painted as a miss" in item.get("text", "").lower()
    )
    assert groups["tag"] == "fix"
    assert groups["link"] == "/draft"
    assert "—" not in groups["text"]
    assert "–" not in groups["text"]
