from __future__ import annotations

import hashlib
import html
import logging
import os

from dashboard_services.ai.cache import (
    build_ai_cache_key,
    load_cached_ai_text,
    save_cached_ai_text,
)
from dashboard_services.ai.client import AIRateLimitError, AIUnavailableError
from dashboard_services.ai.context_builders import (
    _ctx_is_sf,
    build_model_value_lookup,
    build_power_rankings_context,
    build_team_gm_context,
    build_trade_suggestions_context,
    calculate_roster_grade,
    ctx_scoring_type,
    redraft_window_label,
    summarize_roster_players,
)
from dashboard_services.ai.html_sanitize import sanitize_ai_html
from dashboard_services.ai.prompts import (
    generate_power_rankings_result,
    generate_team_ai_result,
    generate_trade_ai_result,
    generate_trade_suggestions_result,
    normalize_trade_scoring_type,
)
from dashboard_services.ai.prose import (
    scrub_ai_prose_field_names,
    scrub_ai_result_strings,
)
from dashboard_services.providers.espn_api import safe_float
from dashboard_services.rank_medals import rank_mark

logger = logging.getLogger(__name__)

AI_ENABLED = os.getenv("AI_ENABLED", "true").lower() == "true"


def _emit_ai_html(raw: str) -> str:
    """Final gate for every AI HTML string returned to the browser."""
    return sanitize_ai_html(raw)


def ai_available() -> bool:
    return AI_ENABLED and bool(os.getenv("OPENAI_API_KEY"))


def _ai_error_notice(reason: str = "") -> str:
    msg = "AI analysis temporarily unavailable"
    if reason:
        msg += f" ({reason})"
    return (
        f"<div class='ai-copy ai-error-notice'>"
        f"<span class='ai-error-icon'>&#x26A0;&#xFE0F;</span> {html.escape(msg)}. "
        f"Showing data-based summary below.</div>"
    )


def _gm_memo_fallback_html(team_ctx: dict) -> str:
    top_assets = ", ".join(
        p["name"] for p in (team_ctx.get("top_assets") or [])[:4]
    ) or "None"
    name = html.escape(str(team_ctx.get("team_name") or "This team"))
    direction = html.escape(str(team_ctx.get("direction") or "bubble"))
    bits = []
    pct = team_ctx.get("playoff_pct")
    if pct is not None:
        bits.append(f"Playoff odds {html.escape(str(pct))}%.")
    if team_ctx.get("season_phase") == "preseason":
        bits.append("Preseason -- no games yet.")
    else:
        rec = str(team_ctx.get("record") or "").strip()
        if rec:
            bits.append(f"Record: {html.escape(rec)}.")
        pf = team_ctx.get("points_for")
        pa = team_ctx.get("points_against")
        if pf not in (None, "", 0, 0.0) or pa not in (None, "", 0, 0.0):
            bits.append(
                f"PF: {safe_float(pf):.1f} | PA: {safe_float(pa):.1f}."
            )
    weak = team_ctx.get("weakest_positions") or []
    if weak:
        bits.append("Weakest rooms: " + html.escape(", ".join(str(w) for w in weak)) + ".")
    extra = " ".join(bits)
    extra_p = f"<p>{extra}</p>" if extra else ""
    return f"""
        <div class="ai-copy">
          <p><strong>{name}</strong> profiles as a <strong>{direction}</strong> team.</p>
          <p>Top assets: {html.escape(top_assets)}.</p>
          {extra_p}
        </div>
        """


def _fo_brief_fallback_html(team_ctx: dict) -> str:
    pos = team_ctx.get("position_strength") or {}
    ranked = sorted(
        (
            (p, safe_float((info or {}).get("top_3_sum")))
            for p, info in pos.items()
            if p in ("QB", "RB", "WR", "TE")
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    strong = ", ".join(p for p, _ in ranked[:2] if ranked) or "None"
    weak = ", ".join(str(w) for w in (team_ctx.get("weakest_positions") or [])) or (
        ", ".join(p for p, _ in ranked[-2:] if ranked) or "None"
    )
    direction = html.escape(str(team_ctx.get("direction") or "bubble"))
    bits = [f"<p><strong>Direction:</strong> {direction}</p>"]
    pct = team_ctx.get("playoff_pct")
    if pct is not None:
        bits.append(f"<p><strong>Playoff odds:</strong> {html.escape(str(pct))}%</p>")
    return f"""
        <div class="ai-copy">
          <p><strong>Strongest rooms:</strong> {html.escape(strong)}</p>
          <p><strong>Weakest rooms:</strong> {html.escape(weak)}</p>
          {''.join(bits)}
        </div>
        """


def render_team_ai_result(result: dict, mode: str = "gm_memo") -> str:
    """
    Render the AI-generated team analysis result as HTML.
    mode: 'gm_memo' or 'front_office_briefing'
    """
    result = scrub_ai_result_strings(result or {})
    if mode == "gm_memo":
        team_identity = html.escape(str(result.get("team_identity") or ""))
        outlook = html.escape(str(result.get("outlook") or ""))
        strength = html.escape(str(result.get("strength") or ""))
        weakness = html.escape(str(result.get("weakness") or ""))
        next_move = html.escape(str(result.get("next_move") or ""))
        trade_posture = html.escape(str(result.get("trade_posture") or ""))
        verdict = html.escape(str(result.get("verdict") or "HOLD").upper())

        return f"""
        <div class="ai-copy">
          <p><strong>{team_identity}</strong></p>
          <p>{outlook}</p>
          <ul>
            <li><strong>Biggest strength:</strong> {strength}</li>
            <li><strong>Biggest weakness:</strong> {weakness}</li>
            <li><strong>Best next move:</strong> {next_move}</li>
          </ul>
          <p>{trade_posture}</p>
        </div>
        """
    else:  # front_office_briefing
        headline = html.escape(str(result.get("headline") or ""))
        posture = html.escape(str(result.get("posture") or ""))
        strongest_room = html.escape(str(result.get("strongest_room") or ""))
        weakest_room = html.escape(str(result.get("weakest_room") or ""))
        next_move = html.escape(str(result.get("next_move") or ""))
        gm_alert = html.escape(str(result.get("gm_alert") or ""))

        return f"""
        <div class="ai-copy">
          <p><strong>{headline}</strong></p>
          <p>{posture}</p>
          <ul>
            <li><strong>Strongest room:</strong> {strongest_room}</li>
            <li><strong>Weakest room:</strong> {weakest_room}</li>
            <li><strong>Most important next move:</strong> {next_move}</li>
          </ul>
          <p><strong>GM Alert:</strong> {gm_alert}</p>
        </div>
        """


def _ctx_with_playoff_odds(ctx: dict, *, block: bool = False) -> dict:
    """Attach a warm playoff-odds snapshot when the sim cache already has one.

    Defaults to ``block=False``: a cold Monte Carlo (projection fetches + 10k
    sims) plus an OpenAI call regularly exceeds the edge proxy budget and the
    browser reports a generic "Network error" on Refresh Report. Callers that
    truly need odds before continuing can pass ``block=True``.
    """
    if not ctx or ctx.get("playoff_odds"):
        return ctx
    try:
        from app import _playoff_sim_cached
        rows = _playoff_sim_cached(
            ctx, str(ctx.get("platform") or "sleeper"), block=block,
        ) or []
    except Exception:
        return ctx
    if not rows:
        return ctx
    out = dict(ctx)
    out["playoff_odds"] = rows
    return out


def _trade_room_slice(team_ctx: dict, *, is_redraft: bool) -> dict:
    """Compact roster card for trade analysis (viewer or opponent)."""
    pos = team_ctx.get("position_strength") or {}
    ranked = sorted(
        (
            (p, safe_float((info or {}).get("top_3_sum")))
            for p, info in pos.items()
            if p in ("QB", "RB", "WR", "TE")
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    weak = list(team_ctx.get("weakest_positions") or [])
    if not weak:
        weak = [p for p, _ in ranked[-2:]]
    strong = [p for p, _ in ranked[:2]]
    out = {
        "team_name": team_ctx.get("team_name"),
        "direction": team_ctx.get("direction"),
        "season_phase": team_ctx.get("season_phase"),
        "record": team_ctx.get("record"),
        "playoff_pct": team_ctx.get("playoff_pct"),
        "playoff_status": team_ctx.get("playoff_status"),
        "playoff_rank": team_ctx.get("playoff_rank"),
        "weakest_positions": weak,
        "strong_positions": strong,
        "points_for": team_ctx.get("points_for"),
        "points_against": team_ctx.get("points_against"),
    }
    if not is_redraft:
        out["future_picks"] = team_ctx.get("future_picks") or []
    return out


def get_team_gm_memo(ctx: dict, viewer_roster_id: str, force_refresh: bool = False) -> str:
    # Never block this request on a cold playoff Monte Carlo. Refresh Report
    # already pays for an OpenAI round-trip; stacking a full sim behind the
    # same HTTP call trips gateway timeouts and the UI shows "Network error".
    # Odds attribution is fixed at the sim layer (real roster_ids); warm cache
    # is enough, and a miss just omits odds for this generation while a
    # background sim fills the cache for the next one.
    team_ctx = build_team_gm_context(_ctx_with_playoff_odds(ctx, block=False), viewer_roster_id)
    if not team_ctx:
        return ""

    # v8: Week-1 seeded standings must not attach mis-keyed playoff odds
    # (RangeIndex roster_ids → FO report citing another team's 0.1%).
    cache_key = build_ai_cache_key("gm_memo", team_ctx, "v8")
    if not force_refresh:
        cached = load_cached_ai_text(cache_key)
        if cached:
            return _emit_ai_html(scrub_ai_prose_field_names(cached))

    if not team_ctx.get("playoff_pct") and ctx is not None:
        # Kick a background sim so the next refresh can cite odds.
        try:
            from app import _playoff_sim_cached
            _playoff_sim_cached(ctx, str(ctx.get("platform") or "sleeper"), block=False)
        except Exception:
            logger.debug("[ai gm_memo] background odds warm failed", exc_info=True)

    if not ai_available():
        html_out = _gm_memo_fallback_html(team_ctx)
        save_cached_ai_text(cache_key, html_out)
        return _emit_ai_html(html_out)

    try:
        result = generate_team_ai_result(team_ctx, mode="gm_memo")
        html_out = render_team_ai_result(result, mode="gm_memo")
    except (AIRateLimitError, AIUnavailableError) as e:
        reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
        logger.warning("[ai gm_memo] %s: %s", reason, e)
        html_out = _ai_error_notice(reason) + _gm_memo_fallback_html(team_ctx)
    except Exception as e:
        logger.exception("[ai gm_memo] unexpected error: %s", e)
        html_out = _ai_error_notice() + _gm_memo_fallback_html(team_ctx)

    save_cached_ai_text(cache_key, html_out)
    return _emit_ai_html(html_out)


def get_front_office_briefing(ctx: dict, viewer_roster_id: str) -> str:
    team_ctx = build_team_gm_context(_ctx_with_playoff_odds(ctx), viewer_roster_id)
    if not team_ctx:
        return ""

    cache_key = build_ai_cache_key("front_office_briefing", team_ctx, "v6")
    cached = load_cached_ai_text(cache_key)
    if cached:
        return _emit_ai_html(scrub_ai_prose_field_names(cached))

    if not ai_available():
        html_out = _fo_brief_fallback_html(team_ctx)
        save_cached_ai_text(cache_key, html_out)
        return _emit_ai_html(html_out)

    try:
        result = generate_team_ai_result(team_ctx, mode="front_office_briefing")
        html_out = render_team_ai_result(result, mode="front_office_briefing")
    except (AIRateLimitError, AIUnavailableError) as e:
        reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
        logger.warning("[ai front_office] %s: %s", reason, e)
        html_out = _ai_error_notice(reason) + _fo_brief_fallback_html(team_ctx)
    except Exception as e:
        logger.exception("[ai front_office] unexpected error: %s", e)
        html_out = _ai_error_notice() + _fo_brief_fallback_html(team_ctx)

    save_cached_ai_text(cache_key, html_out)
    return _emit_ai_html(html_out)


# ──────────────────────────────────────────────────────────────────────────────
# Trade Analyst verdict guard
#
# Bug (Kaedon, Week 11 2026, Blackedraw): the calculator showed a +226.3
# depth-adjusted value win (Burrow + Cook in; Caleb Williams + Egbuka +
# Judkins out) and the AI still returned DECLINE / High confidence, purely
# on age-curve grounds ("young dynasty insulation" for "two older, proven
# pieces"). The system prompt already forbids this (market delta is factor
# 1; factors 2-5, incl. direction / age curve, must never override a
# decisive delta), but a prompt is not an enforcement mechanism. This guard
# is: it runs server-side on every AI verdict, dynasty and redraft alike,
# and mirrors the prompt's Section 3 / Section 8 thresholds exactly.
#
# v2 (same PR): the bands are percentage hybrids, not flat numbers. The
# fair band IS the site's shared fair_value_band (max of 7% of the larger
# side, 25) so the Analyst can never disagree with the calculator about
# what "fair" means; "decisive" needs BOTH a +50 absolute floor AND the
# fair band cleared. A decisive override must also survive the raw delta:
# the depth-adjusted delta alone can manufacture a win out of the roster
# slot penalty (a 2-for-3 consolidation is docked far less than the side
# receiving 3 bodies), so a trade whose raw delta is a real loss gets at
# most fair-band treatment, never decisive treatment. The catastrophe
# exception now counts starter-CALIBER bodies (league starter thresholds,
# IR/Out excluded) and applies across the whole accept band, not just at
# decisive wins. And the verdict is assigned deterministically BEFORE the
# model writes (payload trade.assigned_verdict), so the narrative is
# written for the verdict instead of being overwritten after the fact;
# this guard remains as the backstop when the model disobeys anyway.
# ──────────────────────────────────────────────────────────────────────────────

_TRADE_GUARD_SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
_TRADE_GUARD_UNAVAILABLE = ("IR", "OUT", "O", "PUP")


def trade_delta_bands(larger_side) -> dict:
    """Percentage-hybrid verdict bands for a trade's larger side.

    fair: the site's shared fair_value_band (max of 7% of the larger
    side, 25). decisive: at least +50 AND at least the fair band, so a
    +60 delta on a ~3000-value blockbuster (2%) is a slight win, not a
    decisive one.
    """
    from utils.trade import fair_value_band

    try:
        larger = float(larger_side or 0.0)
    except (TypeError, ValueError):
        larger = 0.0
    fair = fair_value_band(larger)
    return {"fair": fair, "decisive": max(50.0, fair)}


def _asset_is_unavailable(asset: dict) -> bool:
    status = str((asset or {}).get("injury_status") or "").strip().upper()
    return status in _TRADE_GUARD_UNAVAILABLE


def starter_caliber_counts(
        roster_players: list[dict],
        thresholds: dict,
        injury_status_by_id: dict | None = None,
) -> dict[str, int]:
    """Per-position count of starter-caliber, available bodies.

    A body counts only if its value clears the league's starter threshold
    for its position (utils.trade.derive_league_thresholds) and it is not
    flagged IR/Out. A low-value backup QB is a roster body, not a starter.
    """
    counts = {pos: 0 for pos in _TRADE_GUARD_SKILL_POSITIONS}
    injuries = injury_status_by_id or {}
    for p in roster_players or []:
        pos = str((p or {}).get("position") or "").upper()
        if pos not in counts:
            continue
        status = str(injuries.get(str((p or {}).get("id") or "")) or "").strip().upper()
        if status in _TRADE_GUARD_UNAVAILABLE:
            continue
        try:
            value = float((p or {}).get("value") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value >= float((thresholds or {}).get(pos) or 0.0):
            counts[pos] += 1
    return counts


def post_trade_caliber_counts(
        pre_trade_caliber: dict,
        gets_assets: list[dict],
        gives_assets: list[dict],
        thresholds: dict,
) -> dict[str, int]:
    """Starter-caliber counts after the trade, from pre-trade caliber
    counts. Only assets that are themselves starter-caliber and available
    move the counts; picks never do."""
    counts = {
        pos: int((pre_trade_caliber or {}).get(pos) or 0)
        for pos in _TRADE_GUARD_SKILL_POSITIONS
    }

    def _caliber(asset: dict) -> str:
        pos = str((asset or {}).get("position") or "").upper()
        if pos not in counts or _asset_is_unavailable(asset):
            return ""
        try:
            value = float((asset or {}).get("value") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        return pos if value >= float((thresholds or {}).get(pos) or 0.0) else ""

    for asset in gives_assets or []:
        pos = _caliber(asset)
        if pos:
            counts[pos] -= 1
    for asset in gets_assets or []:
        pos = _caliber(asset)
        if pos:
            counts[pos] += 1
    return counts


def trade_verdict_assessment(
        *,
        adjusted_delta: float,
        raw_delta: float | None = None,
        larger_side: float | None = None,
        caliber_counts: dict | None = None,
        pos_counts: dict | None = None,
        slot_counts: dict | None = None,
) -> dict:
    """Deterministic read of a trade before any model narrative exists.

    decisive: adjusted delta clears the decisive bar AND the raw delta is
    not a loss (a slot-penalty-only "win" is never decisive).
    catastrophe: the post-trade roster cannot field a starter-caliber
    player at a required position (caliber counts when available, body
    counts as the fallback).
    assigned_verdict: "ACCEPT" when the framework leaves no honest room
    for another verdict (inside the accept band, no catastrophe, raw
    delta not a real loss); otherwise None and the model decides, with
    the guard as backstop.
    """
    try:
        adj = float(adjusted_delta or 0.0)
    except (TypeError, ValueError):
        adj = 0.0
    raw = adj if raw_delta is None else float(raw_delta or 0.0)
    bands = trade_delta_bands(larger_side)
    fair, decisive_bar = bands["fair"], bands["decisive"]
    counts = caliber_counts if caliber_counts is not None else (pos_counts or {})
    catastrophe = trade_structural_catastrophe(counts, slot_counts or {})
    decisive = adj >= decisive_bar and raw >= 0
    in_accept_band = adj >= -fair
    raw_real_loss = raw < -fair
    assigned = None
    if in_accept_band and not catastrophe and not raw_real_loss:
        assigned = "ACCEPT"
    return {
        "adjusted_delta": round(adj, 1),
        "raw_delta": round(raw, 1),
        "larger_side": round(float(larger_side or 0.0), 1),
        "fair_band": round(fair, 1),
        "decisive_bar": round(decisive_bar, 1),
        "decisive": decisive,
        "catastrophe": catastrophe,
        "assigned_verdict": assigned,
    }


def trade_deterministic_confidence(
        verdict: str,
        adjusted_delta: float,
        fair_band: float,
        decisive_bar: float,
        decisive_ok: bool = True,
) -> str:
    """Confidence from the final verdict and the delta's magnitude only.

    Never inherited from the model: an overridden ACCEPT must not carry
    the DECLINE's "high confidence" (the reported card did exactly that).
    "high" additionally requires decisive_ok -- a win that exists only in
    the adjusted delta (raw delta negative) never earns it.
    """
    try:
        mag = abs(float(adjusted_delta or 0.0))
    except (TypeError, ValueError):
        mag = 0.0
    if mag >= decisive_bar and decisive_ok:
        return "high"
    if mag > fair_band:
        return "medium"
    return "low"


def partner_acceptance_label(adjusted_delta: float, larger_side=None) -> str:
    """How likely the partner is to accept this deal as constructed,
    read deterministically off the partner's side of the delta.

    Separate from the verdict on purpose: the verdict answers "should
    you take this if it is offered to you", this answers "will they say
    yes if you send it". A partner asked to lose a decisive amount is a
    Long shot no matter how good the deal is for the viewer.
    """
    try:
        adj = float(adjusted_delta or 0.0)
    except (TypeError, ValueError):
        adj = 0.0
    bands = trade_delta_bands(larger_side)
    if adj <= bands["fair"]:
        return "Likely"
    if adj < bands["decisive"]:
        return "Possible"
    return "Long shot"


def partner_acceptance_note(label: str, adjusted_delta: float) -> str:
    try:
        adj = float(adjusted_delta or 0.0)
    except (TypeError, ValueError):
        adj = 0.0
    if label == "Long shot":
        return (
            f"Partner acceptance: Long shot. They would give up {adj:.1f} more in "
            "depth-adjusted value than they get back, so expect a counter or a no "
            "if you send it as-is."
        )
    if label == "Possible":
        return (
            f"Partner acceptance: Possible. They would give up {adj:.1f} more in "
            "depth-adjusted value than they get back, inside the range partners "
            "sometimes eat for fit."
        )
    return (
        "Partner acceptance: Likely. The value does not ask them to take a "
        "real loss, so fit and need decide it."
    )


def post_trade_position_counts(
        pre_trade_counts: dict,
        gets_assets: list[dict],
        gives_assets: list[dict],
) -> dict[str, int]:
    """Skill-position body counts after the trade, from pre-trade counts.

    Picks are not roster bodies and never move these counts.
    """
    counts = {
        pos: int((pre_trade_counts or {}).get(pos) or 0)
        for pos in _TRADE_GUARD_SKILL_POSITIONS
    }
    for asset in gives_assets or []:
        pos = str(asset.get("position") or "").upper()
        if pos in counts:
            counts[pos] -= 1
    for asset in gets_assets or []:
        pos = str(asset.get("position") or "").upper()
        if pos in counts:
            counts[pos] += 1
    return counts


def trade_structural_catastrophe(
        pos_counts: dict,
        slot_counts: dict,
) -> bool:
    """True only for the prompt's one deterministic decline excuse: the
    post-trade roster cannot field a starting lineup at all.

    The counts are whatever the caller counted -- starter-caliber bodies
    when thresholds are available (the live path), plain bodies as the
    fallback. Either way the test is the same: fewer counted players than
    the direct QB/RB/WR/TE starter slots, or not enough flex-eligible
    players left over for the FLEX / SUPER_FLEX slots. Anything softer
    (thinner depth, older starters, a weaker bench) is a narrative
    concern, not a catastrophe, and cannot veto a value win. Unknown
    counts or slots prove nothing, so they return False.
    """
    if not pos_counts or not slot_counts:
        return False
    counts = {pos: int(pos_counts.get(pos) or 0) for pos in _TRADE_GUARD_SKILL_POSITIONS}
    direct = {pos: int(slot_counts.get(pos) or 0) for pos in _TRADE_GUARD_SKILL_POSITIONS}
    if any(counts[pos] < direct[pos] for pos in _TRADE_GUARD_SKILL_POSITIONS):
        return True
    flex_pool = sum(counts[pos] - direct[pos] for pos in ("RB", "WR", "TE"))
    flex_needed = int(slot_counts.get("FLEX") or 0)
    if flex_pool < flex_needed:
        return True
    superflex_needed = int(slot_counts.get("SUPER_FLEX") or 0)
    superflex_pool = flex_pool - flex_needed + (counts["QB"] - direct["QB"])
    return superflex_pool < superflex_needed


def apply_trade_verdict_guard(
        result: dict,
        *,
        market_delta: float,
        gets_assets: list[dict],
        gives_assets: list[dict],
        pos_counts: dict,
        slot_counts: dict,
        raw_delta: float | None = None,
        larger_side: float | None = None,
        caliber_counts: dict | None = None,
) -> dict:
    """Backstop: force the AI verdict back inside the deterministic
    framework (trade_verdict_assessment) when the model disobeys the
    assigned verdict it was given.

    - Inside the accept band (adjusted delta >= -fair band), with no
      structural catastrophe and no real raw-delta loss, the verdict is
      ACCEPT. The catastrophe exception applies across the whole band:
      a fair or small-win trade that leaves a required position with no
      starter-caliber player is NOT forced to ACCEPT.
    - Below the accept band the model is in COUNTER / DECLINE territory
      already; its verdict stands untouched.

    When the guard fires it replaces summary / helps / risks / counter
    wholesale: keeping the model's decline narrative under an ACCEPT badge
    would contradict the verdict on the card the user actually reads.
    The replacement follows the existing deterministic fallback style and
    still names the real cost (the outgoing package) as the con.
    Confidence on an override is deterministic, never the model's.
    Every override (and every catastrophe-blocked override) is logged
    with both deltas so the bands can be calibrated against real trades.
    """
    if not isinstance(result, dict):
        return result
    verdict = str(result.get("verdict") or "").upper()
    if verdict == "ACCEPT":
        return result
    try:
        delta = float(market_delta or 0.0)
    except (TypeError, ValueError):
        return result
    assessment = trade_verdict_assessment(
        adjusted_delta=delta,
        raw_delta=raw_delta,
        larger_side=larger_side,
        caliber_counts=caliber_counts,
        pos_counts=pos_counts,
        slot_counts=slot_counts,
    )
    if assessment["assigned_verdict"] != "ACCEPT":
        if assessment["catastrophe"] and delta >= -assessment["fair_band"]:
            logger.info(
                "[trade-guard] override blocked by catastrophe pre=%s "
                "raw_delta=%.1f adjusted_delta=%.1f larger_side=%.1f "
                "pct=%.3f catastrophe=%s",
                verdict, assessment["raw_delta"], assessment["adjusted_delta"],
                assessment["larger_side"],
                (assessment["adjusted_delta"] / assessment["larger_side"])
                if assessment["larger_side"] else 0.0,
                assessment["catastrophe"],
            )
        return result

    gets_names = ", ".join(
        str(a.get("name")) for a in (gets_assets or []) if a.get("name")) or "the incoming package"
    gives_names = ", ".join(
        str(a.get("name")) for a in (gives_assets or []) if a.get("name")) or "the outgoing package"
    logger.info(
        "[trade-guard] override pre=%s post=ACCEPT raw_delta=%.1f "
        "adjusted_delta=%.1f larger_side=%.1f pct=%.3f catastrophe=%s",
        verdict, assessment["raw_delta"], assessment["adjusted_delta"],
        assessment["larger_side"],
        (assessment["adjusted_delta"] / assessment["larger_side"])
        if assessment["larger_side"] else 0.0,
        assessment["catastrophe"],
    )
    guarded = dict(result)
    guarded["verdict"] = "ACCEPT"
    guarded["summary"] = (
        f"You win this deal on value, receiving {gets_names} for {gives_names}. "
        f"The depth-adjusted market delta is +{delta:.1f} in your favor, and your "
        "post-trade roster still fields a starter at every position, so the "
        "age and depth cost does not outweigh the value and production gained."
    )
    guarded["helps"] = [
        f"Receiving {gets_names} wins the depth-adjusted value by {delta:.1f}.",
        "Your post-trade roster still has a starter at every position.",
    ]
    guarded["risks"] = [
        f"You give up {gives_names}, so your bench gets thinner and younger upside leaves the roster.",
    ]
    guarded["counter"] = ""
    guarded["confidence"] = trade_deterministic_confidence(
        "ACCEPT", delta, assessment["fair_band"], assessment["decisive_bar"],
        decisive_ok=assessment["decisive"])
    return guarded


def get_trade_ai_analysis(
        ctx: dict,
        viewer_roster_id: str,
        viewer_side: str,
        side_a: dict,
        side_b: dict,
        opponent_roster_id: str = "",
        scoring_type: str = "dynasty",
) -> str:
    ctx = _ctx_with_playoff_odds(ctx)
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx or not isinstance(team_ctx, dict):
        return ""

    scoring_type = normalize_trade_scoring_type(scoring_type)
    is_redraft = scoring_type == "redraft"

    viewer_side = (viewer_side or "a").lower().strip()

    viewer_gets = side_a if viewer_side == "a" else side_b
    viewer_gives = side_b if viewer_side == "a" else side_a

    # Injury data from Sleeper player objects stored in ctx
    players_full = ctx.get("players") or {}

    def clean_asset(a: dict) -> dict:
        if not isinstance(a, dict):
            return {
                "id": "",
                "name": "Unknown",
                "position": "?",
                "team": "",
                "age": None,
                "value": 0.0,
            }
        pid = str(a.get("id") or "")
        full_p = players_full.get(pid) or {}
        raw_status = (full_p.get("injury_status") or full_p.get("status") or "").strip().upper()
        injury_status = "" if raw_status in ("", "ACTIVE", "ACT") else raw_status
        injury_body_part = (full_p.get("injury_body_part") or "").strip().lower()
        return {
            "id": pid,
            "name": a.get("name") or "Unknown",
            "position": str(a.get("position") or a.get("pos") or "?").upper(),
            "team": a.get("team") or "",
            "age": a.get("age"),
            "value": safe_float(a.get("value")),
            "pos_rank_label": a.get("pos_rank_label") or "",
            "pos_tier_label": a.get("pos_tier_label") or "",
            "rank_change_7d": a.get("rank_change_7d"),
            "injury_status": injury_status,
            "injury_body_part": injury_body_part,
        }

    def summarize_pick_ids(pick_ids: list) -> dict:
        summary = {
            "count": 0,
            "firsts": 0,
            "seconds": 0,
            "thirds_plus": 0,
            "display": [],
        }
        for raw in pick_ids or []:
            pk = str(raw or "").strip()
            if not pk:
                continue
            summary["count"] += 1
            summary["display"].append(pk.replace("_", " "))
            try:
                parts = pk.split("_")
                if len(parts) >= 2:
                    rnd = int(parts[1])
                    if rnd == 1:
                        summary["firsts"] += 1
                    elif rnd == 2:
                        summary["seconds"] += 1
                    elif rnd >= 3:
                        summary["thirds_plus"] += 1
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
        return summary

    def pos_totals(assets: list[dict]) -> dict[str, float]:
        out: dict[str, float] = {}
        for a in assets:
            pos = str(a.get("position") or "?").upper()
            out[pos] = out.get(pos, 0.0) + safe_float(a.get("value"))
        return {k: round(v, 1) for k, v in out.items()}

    gets_assets = [clean_asset(a) for a in (viewer_gets.get("assets") or viewer_gets.get("breakdown") or [])]
    gives_assets = [clean_asset(a) for a in (viewer_gives.get("assets") or viewer_gives.get("breakdown") or [])]

    gets_pick_summary = summarize_pick_ids(viewer_gets.get("pick_ids") or [])
    gives_pick_summary = summarize_pick_ids(viewer_gives.get("pick_ids") or [])

    gets_pos = pos_totals(gets_assets)
    gives_pos = pos_totals(gives_assets)

    market_delta = round(
        safe_float(viewer_gets.get("effective_total")) - safe_float(viewer_gives.get("effective_total")),
        1,
    )
    # Raw (unadjusted) delta: the guard's second check. Sides built by the
    # calculator carry raw_total; when a caller only carries effective
    # totals the two are the same number by construction there.
    _gets_raw = viewer_gets.get("raw_total")
    _gives_raw = viewer_gives.get("raw_total")
    raw_delta = round(
        safe_float(_gets_raw if _gets_raw is not None else viewer_gets.get("effective_total"))
        - safe_float(_gives_raw if _gives_raw is not None else viewer_gives.get("effective_total")),
        1,
    )
    larger_side = max(
        safe_float(viewer_gets.get("effective_total")),
        safe_float(viewer_gives.get("effective_total")),
        1.0,
    )

    # Post-trade roster: remove traded-away players, add incoming players
    gives_ids = {a.get("id") for a in gives_assets}
    current_top = team_ctx.get("top_assets") or []
    remaining = [a for a in current_top if str(a.get("id") or "") not in gives_ids]
    for a in gets_assets:
        if a.get("position") not in ("PICK", None, "?"):
            remaining.append(a)
    remaining.sort(key=lambda x: safe_float(x.get("value")), reverse=True)
    post_trade_roster = [
        {"name": a.get("name"), "position": a.get("position"), "value": round(safe_float(a.get("value")), 1)}
        for a in remaining[:8]
    ]

    # Pick-to-prospect mapping using rookie ADP
    from utils.lineup_slots import (
        canonicalize_slots,
        count_lineup_slots,
        is_superflex_lineup,
    )
    current_season = ctx.get("current_season") or ctx.get("season") or 2026
    num_teams = len(ctx.get("rosters") or []) or 12
    _roster_positions = canonicalize_slots(ctx.get("roster_positions") or [])
    _slots = count_lineup_slots(_roster_positions)
    is_sf = bool(ctx.get("is_sf") or is_superflex_lineup(_roster_positions))
    # Starter format the analyst should reason from (QB value swings hard on this).
    _qb_slots = _slots.get("QB", 0)
    _sf_slots = _slots.get("SUPER_FLEX", 0)
    if is_redraft:
        _format_note = (
            "REDRAFT league: players are owned for this NFL season only. Draft picks "
            "cannot be traded. Never recommend a pick as a counter or sweetener. "
            "Market values in this JSON are redraft (this-season) values."
        )
        if is_sf:
            _format_note += (
                " Superflex/2QB: quarterbacks carry premium value this season. The "
                "market values already reflect the format."
            )
        else:
            _format_note += (
                " 1QB: standard single-QB scarcity. The market values already "
                "reflect the league format."
            )
    else:
        _format_note = (
            "Superflex/2QB league: quarterbacks carry premium value; weight QB assets and QB "
            "rookie picks up. The market values in this JSON already reflect the league format."
            if is_sf else
            "1QB league: standard single-QB scarcity. The market values in this JSON already "
            "reflect the league format."
        )
    league_format = {
        "scoring_type": scoring_type,
        "picks_tradable": not is_redraft,
        "qb_format": "Superflex/2QB" if is_sf else "1QB",
        "superflex": is_sf,
        "qb_starter_slots": _qb_slots + _sf_slots,
        "starting_lineup": _roster_positions or None,
        "note": _format_note,
    }
    all_pick_ids = list(viewer_gets.get("pick_ids") or []) + list(viewer_gives.get("pick_ids") or [])

    pick_prospects: dict = {}
    if (not is_redraft) and all_pick_ids:
        try:
            from dashboard_services.adp_service import (
                build_model_adp_fallback,
                fetch_league_adp_from_db,
            )
            adp_raw = fetch_league_adp_from_db(is_sf=is_sf, season=current_season, draft_type="rookie") or {}
            if not adp_raw:
                adp_raw = build_model_adp_fallback(is_sf=is_sf, season=current_season) or {}
            name_by_id = {str(p.get("id")): p.get("name") for p in (ctx.get("model_value_table") or []) if p.get("id")}
            prospects_sorted = sorted(
                [
                    {"sid": sid, "name": name_by_id.get(str(sid)) or "", "avg_pick": float(info.get("avg_pick") or info.get("adp_rank") or 999), "position": info.get("position") or ""}
                    for sid, info in adp_raw.items()
                    if name_by_id.get(str(sid))
                ],
                key=lambda x: x["avg_pick"],
            )
            for pk in all_pick_ids:
                parts = str(pk).split("_")
                if len(parts) < 3:
                    continue
                try:
                    yr, rnd = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                if yr != current_season or not prospects_sorted:
                    continue
                third = parts[2]
                try:
                    slot = int(third)
                except ValueError:
                    slot = {"early": max(1, round(num_teams * 0.2)), "mid": round(num_teams * 0.5), "late": round(num_teams * 0.8)}.get(third, round(num_teams / 2))
                overall = (rnd - 1) * num_teams + slot
                closest = min(prospects_sorted, key=lambda p: abs(p["avg_pick"] - overall))
                if abs(closest["avg_pick"] - overall) <= num_teams // 2 + 1:
                    pick_prospects[pk] = {"name": closest["name"], "position": closest["position"], "adp_overall": round(closest["avg_pick"], 1)}
        except Exception:
            logger.debug("suppressed exception", exc_info=True)

    # Opponent team context. Prefer the opponent the UI actually bound (reliable
    # even when the return is picks-only or the platform's roster ids don't match
    # the calculator's player ids); only fall back to inferring the partner from
    # the acquired players' owner when no explicit opponent was passed.
    opponent_ctx: dict = {}
    try:
        _roster_ids = {str(r.get("roster_id") or "") for r in (ctx.get("rosters") or [])}
        opp_roster_id = None
        _explicit = str(opponent_roster_id or "").strip()
        if _explicit and _explicit != str(viewer_roster_id) and _explicit in _roster_ids:
            opp_roster_id = _explicit
        else:
            opponent_side = side_b if viewer_side == "a" else side_a
            opp_player_ids = {str(a.get("id")) for a in (opponent_side.get("assets") or []) if a.get("id")}
            for roster in (ctx.get("rosters") or []):
                rid = str(roster.get("roster_id") or "")
                if rid == str(viewer_roster_id):
                    continue
                if opp_player_ids & {str(p) for p in (roster.get("players") or [])}:
                    opp_roster_id = rid
                    break
        if opp_roster_id:
            opp_team_ctx = build_team_gm_context(ctx, opp_roster_id)
            if opp_team_ctx:
                opponent_ctx = _trade_room_slice(opp_team_ctx, is_redraft=is_redraft)
                opponent_ctx["top_assets"] = [
                    {
                        "name": a.get("name"),
                        "position": a.get("position"),
                        "value": round(safe_float(a.get("value")), 1),
                    }
                    for a in (opp_team_ctx.get("top_assets") or [])[:5]
                ]
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    # Deterministic verdict inputs (guard v2): starter thresholds for this
    # league, the viewer's pre-trade starter-caliber counts, and the
    # post-trade caliber counts. Any failure leaves caliber_counts None
    # and the assessment falls back to plain body counts.
    _caliber_post = None
    _thresholds = {}
    try:
        from utils.trade import derive_league_thresholds

        _thresholds, _depth_floor = derive_league_thresholds(
            _roster_positions, num_teams, is_sf=is_sf)
        _viewer_roster = next(
            (r for r in (ctx.get("rosters") or [])
             if str(r.get("roster_id")) == str(viewer_roster_id)),
            None,
        )
        if _viewer_roster is not None:
            _injury_by_id = {}
            for _pid, _full in (players_full or {}).items():
                _st = str((_full or {}).get("injury_status")
                          or (_full or {}).get("status") or "").strip()
                if _st:
                    _injury_by_id[str(_pid)] = _st
            _pre_caliber = starter_caliber_counts(
                summarize_roster_players(
                    _viewer_roster,
                    ctx.get("players_index") or {},
                    ctx.get("players_map") or {},
                    build_model_value_lookup(
                        ctx.get("model_value_table") or [],
                        is_sf=_ctx_is_sf(ctx),
                        scoring_type=ctx_scoring_type(ctx),
                    ),
                ),
                _thresholds,
                _injury_by_id,
            )
            _caliber_post = post_trade_caliber_counts(
                _pre_caliber, gets_assets, gives_assets, _thresholds)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        _caliber_post = None

    _strength = team_ctx.get("position_strength") or {}
    _pre_counts = {
        pos: (_strength.get(pos) or {}).get("count") or 0
        for pos in _TRADE_GUARD_SKILL_POSITIONS
    }
    _post_counts = post_trade_position_counts(_pre_counts, gets_assets, gives_assets)
    assessment = trade_verdict_assessment(
        adjusted_delta=market_delta,
        raw_delta=raw_delta,
        larger_side=larger_side,
        caliber_counts=_caliber_post,
        pos_counts=_post_counts,
        slot_counts=_slots,
    )
    _partner_label = partner_acceptance_label(market_delta, larger_side)

    team_slice = _trade_room_slice(team_ctx, is_redraft=is_redraft)
    if not is_redraft:
        team_slice["pick_summary"] = team_ctx.get("pick_summary") or {}
        team_slice["market_profile"] = team_ctx.get("market_profile") or {}
        team_slice["starter_profile"] = team_ctx.get("starter_profile") or {}
        team_slice["bench_profile"] = team_ctx.get("bench_profile") or {}
    payload = {
        "team_context": team_slice,
        "trade": {
            "viewer_side": viewer_side,
            "viewer_gets": {
                "assets": gets_assets,
                "pick_ids": viewer_gets.get("pick_ids") or [],
                "effective_total": safe_float(viewer_gets.get("effective_total")),
                "position_totals": gets_pos,
                "pick_summary": gets_pick_summary,
            },
            "viewer_gives": {
                "assets": gives_assets,
                "pick_ids": viewer_gives.get("pick_ids") or [],
                "effective_total": safe_float(viewer_gives.get("effective_total")),
                "position_totals": gives_pos,
                "pick_summary": gives_pick_summary,
            },
            "post_trade_roster": post_trade_roster,
            "pick_prospects": pick_prospects,
            "market_delta": market_delta,
            "market_delta_raw": raw_delta,
            "fair_band": assessment["fair_band"],
            "decisive_bar": assessment["decisive_bar"],
            "structural_catastrophe": assessment["catastrophe"],
            "assigned_verdict": assessment["assigned_verdict"],
            "partner_acceptance": _partner_label,
        },
        "league_format": league_format,
        "scoring_type": scoring_type,
        "opponent_team": opponent_ctx or None,
    }

    # Build cache key for trade analysis. v14: see the verdict-guard note
    # at the key below. v13: the rank/tier label prompt
    # was rewritten (two-construction whitelist, per-trade worked examples,
    # self-check line), so v12 analyses generated under the old note must
    # not hit. v12: the tier-label scrubber now
    # also collapses the spaced/tier role forms ("RB18 type", "WR9 tier
    # asset") and tied exact ranks at any rank, so v11 analyses carrying
    # those forms must not hit. (v11: assets now also carry a
    # pos_tier_label and prose "-type" language is pinned to the tier, not
    # the exact rank; cached v10 analyses can still say "WR23-type", so
    # they must not hit. v10: asset rank labels became the recomputed
    # format labels and prose ranks were pinned to them.)
    # v14: the deterministic verdict guard (apply_trade_verdict_guard) now
    # overrides DECLINE/COUNTER verdicts that contradict a decisive value
    # win, so v13 analyses carrying those verdicts must not hit.
    # v15: guard v2 -- percentage-hybrid bands, raw+adjusted double check,
    # starter-caliber catastrophe, verdict-first assigned_verdict in the
    # payload, deterministic confidence, and the partner-acceptance split,
    # so v14 analyses carrying flat-band verdicts must not hit.
    cache_key = build_ai_cache_key("trade_analysis", payload, "v15")

    # Try to get from cache first
    cached = load_cached_ai_text(cache_key)
    if cached:
        return _emit_ai_html(cached)

    if not ai_available():
        verdict = "ACCEPT" if market_delta > 40 else "DECLINE" if market_delta < -40 else "COUNTER"
        fallback = {
            "verdict": verdict,
            "summary": f"This is a fallback analysis for a {team_ctx.get('direction') or 'balanced'} team. Market delta: {market_delta:.1f}.",
            "helps": ["Uses market value and roster context as baseline inputs."],
            "risks": ["AI is currently disabled, so this is not a model-generated explanation."],
            "counter": "Enable AI to get a fuller front-office style recommendation.",
            "confidence": "low",
        }
        html_out = render_trade_ai_html(fallback)
        save_cached_ai_text(cache_key, html_out)
        return _emit_ai_html(html_out)

    try:
        result = generate_trade_ai_result(payload)
        result = apply_trade_verdict_guard(
            result,
            market_delta=market_delta,
            gets_assets=gets_assets,
            gives_assets=gives_assets,
            pos_counts=_post_counts,
            slot_counts=_slots,
            raw_delta=raw_delta,
            larger_side=larger_side,
            caliber_counts=_caliber_post,
        )
        # Deterministic finalize: confidence comes from the final verdict
        # and the delta's magnitude, never from the model (an overridden
        # ACCEPT must not inherit a DECLINE's confidence), and partner
        # acceptance rides as its own labeled read, separate from the
        # verdict.
        if isinstance(result, dict):
            result = dict(result)
            result["confidence"] = trade_deterministic_confidence(
                str(result.get("verdict") or ""),
                market_delta,
                assessment["fair_band"],
                assessment["decisive_bar"],
                decisive_ok=assessment["decisive"],
            )
            result["partner_acceptance"] = _partner_label
            result["partner_acceptance_note"] = partner_acceptance_note(
                _partner_label, market_delta)
        html_out = render_trade_ai_html(result)
        save_cached_ai_text(cache_key, html_out)
        return _emit_ai_html(html_out)
    except (AIRateLimitError, AIUnavailableError) as e:
        reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
        logger.warning("[trade-ai] %s: %s", reason, e)
        verdict = "ACCEPT" if market_delta > 40 else "DECLINE" if market_delta < -40 else "COUNTER"
        fallback = {
            "verdict": verdict,
            "summary": f"AI analysis {reason}. Market delta: {market_delta:.1f} - verdict based on value differential only.",
            "helps": [f"Market value calculation is based on current {scoring_type} rankings."],
            "risks": [f"AI is {reason}; this is a data-only estimate without roster context."],
            "counter": "Try again shortly for a full AI-powered front-office recommendation.",
            "confidence": "low",
        }
        html_out = _ai_error_notice(reason) + render_trade_ai_html(fallback)
        save_cached_ai_text(cache_key, html_out)
        return _emit_ai_html(html_out)
    except Exception as e:
        logger.exception("[trade-ai] unexpected error: %s", e)
        verdict = "ACCEPT" if market_delta > 40 else "DECLINE" if market_delta < -40 else "COUNTER"
        fallback = {
            "verdict": verdict,
            "summary": f"The trade is evaluated for a {team_ctx.get('direction') or 'balanced'} roster profile. Market delta: {market_delta:.1f}.",
            "helps": ["The return may line up with your current roster direction."],
            "risks": ["AI call failed; this is a simplified market-value estimate."],
            "counter": "Try adjusting a secondary player if the deal feels close." if is_redraft else "Try adjusting the pick side or a secondary asset if the deal feels close.",
            "confidence": "low",
        }
        html_out = _ai_error_notice() + render_trade_ai_html(fallback)
        save_cached_ai_text(cache_key, html_out)
        return _emit_ai_html(html_out)


def _trade_ai_takeaway(summary: str) -> str:
    """Extract the first 1-2 sentences from an analysis summary for the takeaway box."""
    import re
    text = (summary or "").strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    takeaway = " ".join(s for s in sentences[:2] if s).strip()
    return takeaway or text


# Verdict -> (CSS modifier suffix, glyph). The color itself lives in
# dashboard.css under .trade-ai-verdict-badge-<suffix> /
# .trade-ai-conf-dot-<suffix>: the AI HTML sanitizer strips inline style
# attributes, so a color carried only in style="background:..." never
# reaches the page (the badge rendered as white text on no background).
_TRADE_AI_VERDICTS = {
    "ACCEPT": ("accept", "&#10003;"),
    "DECLINE": ("decline", "&#10005;"),
    "COUNTER": ("counter", "&#8646;"),
}


def render_trade_ai_html(result: dict) -> str:
    raw_verdict = str(result.get("verdict") or "COUNTER").upper()
    verdict = html.escape(raw_verdict)
    verdict_mod, verdict_glyph = _TRADE_AI_VERDICTS.get(raw_verdict, _TRADE_AI_VERDICTS["COUNTER"])
    helps = result.get("helps") or []
    risks = result.get("risks") or []
    counter = html.escape(str(result.get("counter") or ""))
    confidence_raw = str(result.get("confidence") or "").strip()

    helps_html = "".join(
        f"<li>{html.escape(str(x))}</li>" for x in helps[:4]) or "<li>No specific edge identified.</li>"
    risks_html = "".join(
        f"<li>{html.escape(str(x))}</li>" for x in risks[:4]) or "<li>No specific risk identified.</li>"

    counter_html = ""
    if counter:
        counter_html = f"""
        <div class="trade-ai-block">
          <div class="trade-ai-label">Advice</div>
          <div class="trade-ai-copy-line">{counter}</div>
        </div>
        """

    # Partner acceptance: a separate deterministic read (will they say
    # yes if you send it), never merged into the verdict above, which
    # answers whether YOU should take the deal.
    partner_note = html.escape(str(result.get("partner_acceptance_note") or ""))
    partner_html = ""
    if partner_note:
        partner_html = f"""
        <div class="trade-ai-block">
          <div class="trade-ai-label">Partner acceptance</div>
          <div class="trade-ai-copy-line">{partner_note}</div>
        </div>
        """

    # Verdict fragment: the client moves this into the panel header slot.
    conf_label = f"{confidence_raw[:1].upper()}{confidence_raw[1:]} confidence" if confidence_raw else ""
    conf_html = ""
    if conf_label:
        conf_html = (
            f'<div class="trade-ai-conf-pill">'
            f'<span class="trade-ai-conf-dot trade-ai-conf-dot-{verdict_mod}"></span>'
            f"{html.escape(conf_label)}</div>"
        )
    verdict_frag = f"""
      <div class="trade-ai-verdict-frag" data-verdict="{verdict}">
        <div class="trade-ai-verdict-badge trade-ai-verdict-badge-{verdict_mod}">{verdict_glyph} {verdict}</div>
        {conf_html}
      </div>
    """

    takeaway = _trade_ai_takeaway(str(result.get("summary") or ""))
    body_html = ""
    if takeaway:
        body_html = f'<p class="trade-ai-body">{html.escape(takeaway)}</p>'

    return f"""
    <div class="ai-copy trade-ai-wrap">
      {verdict_frag}
      {body_html}
      <div class="trade-ai-pc-grid">
        <div class="trade-ai-pc-col trade-ai-pc-pros">
          <div class="trade-ai-pc-label">Pros</div>
          <ul class="trade-ai-list">{helps_html}</ul>
        </div>

        <div class="trade-ai-pc-col trade-ai-pc-cons">
          <div class="trade-ai-pc-label">Cons</div>
          <ul class="trade-ai-list">{risks_html}</ul>
        </div>
      </div>

      {counter_html}
      {partner_html}
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Roster Grade
# ──────────────────────────────────────────────────────────────────────────────

def get_roster_grade(ctx: dict, viewer_roster_id: str) -> dict:
    """Return grade data for a single roster."""
    ctx = _ctx_with_playoff_odds(ctx)
    rosters = ctx.get("rosters") or []
    roster = next((r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)), None)
    if not roster:
        return {"grade": "N/A", "score": 0, "win_window": "Unknown", "breakdown": {}}

    scoring_type = ctx_scoring_type(ctx)
    model_value_lookup = build_model_value_lookup(
        ctx.get("model_value_table") or [],
        is_sf=_ctx_is_sf(ctx),
        scoring_type=scoring_type,
    )
    players_summary = summarize_roster_players(
        roster=roster,
        players_index=ctx.get("players_index") or {},
        players_map=ctx.get("players_map") or {},
        model_value_lookup=model_value_lookup,
    )
    is_redraft = scoring_type == "redraft"
    future_picks = [] if is_redraft else (
        ctx.get("picks_by_roster", {}).get(str(viewer_roster_id), [])
    )
    grade = calculate_roster_grade(
        players_summary, future_picks, scoring_type=scoring_type,
    )
    if is_redraft:
        playoff_pct = None
        for row in (ctx.get("playoff_odds") or []):
            if str((row or {}).get("roster_id")) == str(viewer_roster_id):
                playoff_pct = (row or {}).get("playoff_pct")
                break
        bd = grade.get("breakdown") or {}
        grade["win_window"] = redraft_window_label(
            playoff_pct=playoff_pct, redraft_pct=bd.get("redraft_pct"),
        )
        grade["scoring_type"] = "redraft"
    return grade


def render_roster_grade_badge(grade_data: dict, scoring_type: str = "") -> str:
    grade = html.escape(str(grade_data.get("grade") or "?"))
    win_window = html.escape(str(grade_data.get("win_window") or ""))
    score = grade_data.get("score") or 0
    bd = grade_data.get("breakdown") or {}
    avg_age = bd.get("avg_age", 0)
    elite_count = bd.get("elite_count", 0)
    scoring = (scoring_type or grade_data.get("scoring_type") or "dynasty").strip().lower()

    grade_class = "grade-a" if grade.startswith("A") else "grade-b" if grade.startswith("B") else "grade-c" if grade.startswith("C") else "grade-d"
    extras = f"Score: {score:.0f}/100"
    if scoring != "redraft":
        extras += f" &bull; Age: {avg_age:.1f}"
    extras += f" &bull; Elite: {elite_count}"
    # Disclose when the grade came from a fallback blend (no league context)
    # rather than the documented 40/25/15/12/8 formula.
    formula_id = str(grade_data.get("formula_id") or "")
    _formula_tip = ""
    if formula_id and formula_id != "dynasty_ctx":
        _formula_tip = (
            f' title="Grade blend: {html.escape(formula_id)} '
            f'(estimated without full league context)"'
        )

    return f"""
    <div class="roster-grade-wrap"{_formula_tip}>
      <div class="roster-grade-badge {grade_class}">{grade}</div>
      <div class="roster-grade-meta">
        <div class="roster-grade-window">{win_window}</div>
        <div class="roster-grade-score">{extras}</div>
      </div>
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Power Rankings
# ──────────────────────────────────────────────────────────────────────────────

def _power_rankings_cheap_key(ctx: dict) -> str | None:
    """Build a cache key for power rankings WITHOUT building the full context.

    build_power_rankings_context() computes blended team scores, playoff odds,
    momentum, SoS across the league. On a cache hit we skip it entirely. The
    precise key uses week/season/team IDs, all cheaply available from ctx.
    """
    try:
        rosters = ctx.get("rosters") or []
        team_ids = sorted(str(r.get("roster_id")) for r in rosters if r.get("roster_id") is not None)
        if not team_ids:
            return None
        return build_ai_cache_key(
            "power_rankings",
            {
                # League identity is part of the key (Class A audit): roster
                # ids are 1..N in nearly every league, so a week/season/teams
                # key alone served one league's cached rankings HTML to a
                # different league with the same roster-id set.
                "league_id": str(ctx.get("league_id") or ""),
                "platform": str(ctx.get("platform") or ""),
                "week": ctx.get("week") or ctx.get("current_week"),
                "season": ctx.get("season"),
                "teams": team_ids,
            },
            "v6",
        )
    except Exception:
        return None


def get_power_rankings_html(ctx: dict) -> str:
    # Fast path: check cache with a cheap key before building the expensive
    # full rankings context (blended scores, playoff odds, momentum, SoS).
    _cheap_key = _power_rankings_cheap_key(ctx)
    if _cheap_key:
        _cached = load_cached_ai_text(_cheap_key)
        if _cached:
            return _emit_ai_html(_cached)

    rankings_ctx = build_power_rankings_context(_ctx_with_playoff_odds(ctx))
    teams = rankings_ctx.get("teams") or []
    if not teams:
        return _emit_ai_html("<p>Not enough data for power rankings.</p>")

    cache_key = build_ai_cache_key("power_rankings", {"league_id": str(ctx.get("league_id") or ""), "platform": str(ctx.get("platform") or ""), "week": rankings_ctx.get("week"), "season": rankings_ctx.get("season"), "teams": [t["roster_id"] for t in teams]}, "v6")
    cached = load_cached_ai_text(cache_key)
    if cached:
        return _emit_ai_html(cached)

    # Build fallback narrative map
    fallback_narratives: dict[str, str] = {}
    for t in teams:
        direction = t.get("playoff_status") or t.get("direction") or "balanced"
        top = (t.get("top_assets") or [{}])[0].get("name") or "their core"
        pct = t.get("playoff_pct")
        odds = f" at {pct}% playoff odds" if pct is not None else ""
        fallback_narratives[t["roster_id"]] = (
            f"Led by {top}, this {direction} team sits at #{t['rank']}{odds}."
        )

    if not ai_available():
        html_out = _render_power_rankings_html_from_data(teams, fallback_narratives)
        save_cached_ai_text(cache_key, html_out)
        if _cheap_key and _cheap_key != cache_key:
            save_cached_ai_text(_cheap_key, html_out)
        return _emit_ai_html(html_out)

    try:
        # Trim context for AI - only send what's needed
        scoring_type = rankings_ctx.get("scoring_type") or "dynasty"
        is_redraft = str(scoring_type).strip().lower() == "redraft"
        ai_input = {
            "season": rankings_ctx.get("season"),
            "week": rankings_ctx.get("week"),
            "season_phase": rankings_ctx.get("season_phase"),
            "scoring_type": scoring_type,
            "playoff_teams": rankings_ctx.get("playoff_teams"),
            "teams": [
                {
                    "roster_id": t["roster_id"],
                    "team_name": t["team_name"],
                    "rank": t["rank"],
                    "wins": t["wins"],
                    "losses": t["losses"],
                    "pf": round(t["pf"], 1),
                    "playoff_pct": t.get("playoff_pct"),
                    "playoff_status": t.get("playoff_status") or "",
                    "win_window": (
                        (t.get("playoff_status") or t.get("direction") or "bubble")
                        if is_redraft
                        else (t.get("win_window") or t.get("direction") or "balanced")
                    ),
                    "avg_age": None if is_redraft else t.get("avg_age"),
                    "first_round_picks": 0 if is_redraft else t.get("first_round_picks", 0),
                    "position_strengths": t.get("position_strengths") or {},
                    "top_assets": [{"name": p["name"], "position": p["position"], "value": p["value"]} for p in (t.get("top_assets") or [])[:3]],
                }
                for t in teams
            ],
        }
        result = generate_power_rankings_result(ai_input)
        narratives = {r["roster_id"]: r["narrative"] for r in (result.get("rankings") or [])}
        momentums = {r["roster_id"]: r.get("momentum", "steady") for r in (result.get("rankings") or [])}
        html_out = _render_power_rankings_html_from_data(teams, narratives, momentums)
    except (AIRateLimitError, AIUnavailableError) as e:
        reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
        logger.warning("[power-rankings-ai] %s: %s", reason, e)
        html_out = _ai_error_notice(reason) + _render_power_rankings_html_from_data(teams, fallback_narratives)
    except Exception as e:
        logger.exception("[power-rankings-ai] unexpected error: %s", e)
        html_out = _render_power_rankings_html_from_data(teams, fallback_narratives)

    save_cached_ai_text(cache_key, html_out)
    if _cheap_key and _cheap_key != cache_key:
        save_cached_ai_text(_cheap_key, html_out)
    return _emit_ai_html(html_out)


def _render_power_rankings_html_from_data(
        teams: list[dict],
        narratives: dict[str, str],
        momentums: dict[str, str] | None = None,
) -> str:
    if momentums is None:
        momentums = {}

    rows_html = ""
    for t in teams:
        rid = t["roster_id"]
        rank = t.get("rank", "?")
        team_name = html.escape(str(t.get("team_name") or ""))
        wins = t.get("wins", 0)
        losses = t.get("losses", 0)
        pf = t.get("pf", 0)
        narrative = html.escape(str(narratives.get(rid) or ""))
        momentum = (momentums.get(rid) or "steady").lower()
        momentum_icon = {"rising": "↑", "falling": "↓", "steady": "→"}.get(momentum, "→")
        momentum_class = f"momentum-{momentum}"
        # All-play stays visible: it is the foundation the ranking sorts by.
        ap_pct = t.get("all_play_pct")
        ap_chip = ""
        if ap_pct is not None:
            ap_chip = (
                f"<span class='pr-allplay' title='All-play win rate: how this team "
                f"fares against every other team each week. This is what the power "
                f"rankings sort by.'>{ap_pct:.1%} All-Play</span>"
            )

        rows_html += f"""
        <div class="pr-row">
          <div class="pr-rank">{rank_mark(rank)}</div>
          <div class="pr-body">
            <div class="pr-team-line">
              <span class="pr-team-name team-clickable" data-roster-id="{rid}" data-team-name="{team_name}">{team_name}</span>
              <span class="pr-record">{wins}-{losses}</span>
              <span class="pr-pf">{pf:.1f} PF</span>
              {ap_chip}
              <span class="pr-momentum {momentum_class}">{momentum_icon}</span>
            </div>
            <div class="pr-narrative">{narrative}</div>
          </div>
        </div>
        """

    return f"""
    <div class="power-rankings-wrap">
      {rows_html}
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Trade Suggestions
# ──────────────────────────────────────────────────────────────────────────────

def _trade_suggestions_cheap_key(ctx: dict, viewer_roster_id: str) -> str | None:
    """Build a cache key for trade suggestions WITHOUT building the full context.

    The full build_trade_suggestions_context() is expensive (model value lookup,
    per-roster positional totals across the league, partner analysis). On a cache
    hit we can skip it entirely. The cheap key covers the inputs that determine
    the output: roster composition + week/season. Needs/surplus/direction are
    deterministic functions of these, so a cheap-key hit implies the precise key
    would also hit.
    """
    try:
        rosters = ctx.get("rosters") or []
        roster = next(
            (r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)),
            None,
        )
        if not roster:
            return None
        pids = sorted(str(pid) for pid in (roster.get("players") or []))
        roster_hash = hashlib.md5(",".join(pids).encode()).hexdigest()[:12]
        week = ctx.get("week") or ctx.get("current_week")
        season = ctx.get("season")
        # v15: suggestion prose now runs through the tier-language scrub
        # (spaced/tier role forms collapse); v14 cards can still carry an
        # exact rank as role language ("RB18 type") and must not hit.
        return build_ai_cache_key(
            "trade_suggestions",
            {
                "roster_id": str(viewer_roster_id),
                "roster_hash": roster_hash,
                "week": week,
                "season": season,
            },
            "v15",
        )
    except Exception:
        return None


def get_trade_suggestions_html(ctx: dict, viewer_roster_id: str) -> str:
    # Fast path: check cache with a cheap key before building the expensive
    # full context. build_trade_suggestions_context() does a model value lookup
    # plus per-roster positional totals across the league; skipping it on a
    # cache hit saves significant CPU and memory per request.
    _cheap_key = _trade_suggestions_cheap_key(ctx, viewer_roster_id)
    if _cheap_key:
        _cached = load_cached_ai_text(_cheap_key)
        if _cached:
            return _emit_ai_html(_cached)

    suggestions_ctx = build_trade_suggestions_context(
        _ctx_with_playoff_odds(ctx), viewer_roster_id,
    )
    if not suggestions_ctx:
        return _emit_ai_html("<p>Could not build trade suggestions context.</p>")

    # Roster hash: needs/surplus/direction alone can't see a same-position
    # swap (trading WR2 for a different WR2), so hash the actual roster or
    # the cache serves a stale card after a trade.
    _rids = ",".join(suggestions_ctx.get("viewer_roster_ids") or [])
    _roster_hash = hashlib.md5(_rids.encode()).hexdigest()[:12]
    # v14: same tier-language scrub as the cheap key above; v13 cards can
    # still carry an exact rank as role language and must not hit.
    cache_key = build_ai_cache_key(
        "trade_suggestions",
        {
            "roster_id": viewer_roster_id,
            "roster_hash": _roster_hash,
            "needs": suggestions_ctx.get("viewer_needs"),
            "surplus": suggestions_ctx.get("viewer_surplus"),
            "ceiling_needs": suggestions_ctx.get("viewer_ceiling_needs"),
            "direction": suggestions_ctx.get("viewer_direction"),
        },
        "v14",
    )
    cached = load_cached_ai_text(cache_key)
    if cached:
        return _emit_ai_html(cached)

    if not ai_available() or (not suggestions_ctx.get("top_partners") and not suggestions_ctx.get("pick_trade_partners")):
        html_out = _render_trade_suggestions_fallback(suggestions_ctx)
        save_cached_ai_text(cache_key, html_out)
        if _cheap_key and _cheap_key != cache_key:
            save_cached_ai_text(_cheap_key, html_out)
        return _emit_ai_html(html_out)

    try:
        result = generate_trade_suggestions_result(suggestions_ctx)
        html_out = _render_trade_suggestions_from_data(
            result.get("suggestions") or [],
            suggestions_ctx.get("top_partners") or [],
        )
    except (AIRateLimitError, AIUnavailableError) as e:
        reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
        logger.warning("[trade-suggestions-ai] %s: %s", reason, e)
        html_out = _ai_error_notice(reason) + _render_trade_suggestions_fallback(suggestions_ctx)
    except Exception as e:
        logger.exception("[trade-suggestions-ai] unexpected error: %s", e)
        html_out = _render_trade_suggestions_fallback(suggestions_ctx)

    save_cached_ai_text(cache_key, html_out)
    if _cheap_key and _cheap_key != cache_key:
        save_cached_ai_text(_cheap_key, html_out)
    return _emit_ai_html(html_out)


def _fmt_pick_label(pk: dict) -> str:
    season = pk.get("season", "")
    rnd    = int(pk.get("round") or 0)
    slot   = pk.get("slot")
    if slot:
        return f"{season} {rnd}.{int(slot):02d}"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
    return f"{season} {rnd}{suffix} (Mid)"


def _fairness_label(fairness, value_you_get=None, value_you_give=None) -> str:
    """Plain-language fairness: a 0.82 means nothing to a user, so translate
    the 0-1 score into words, with direction when it's close but not even."""
    try:
        f = float(fairness)
    except (TypeError, ValueError):
        return ""
    if f >= 0.85:
        return "Fair deal"
    if f >= 0.70:
        try:
            if float(value_you_get) >= float(value_you_give):
                return "Slight edge to you"
            return "Slight edge to them"
        except (TypeError, ValueError):
            return "Roughly fair"
    return "Lopsided"


def _tier_label(tier: str) -> str:
    return {"elite": "Elite target", "buy_low": "Buy-low", "realistic": "Realistic"}.get(
        str(tier or "").lower(), ""
    )


def _partner_motivation(p: dict) -> str:
    """Two-sided reasoning: why the PARTNER says yes. Built from their
    record, direction, and needs plus what the viewer sends back."""
    rec = str(p.get("partner_record") or "").strip()
    direction = str(p.get("partner_direction") or "").strip()
    needs = [str(n) for n in (p.get("partner_needs") or []) if n]
    sends = p.get("targets_viewer_sends") or []
    send_names = ", ".join(t.get("name") for t in sends[:2] if t.get("name"))
    if rec and direction:
        head = f"They're {rec} ({direction})"
    elif rec:
        head = f"They're {rec}"
    elif direction:
        head = f"They're in {direction} mode"
    else:
        head = "They"
    if needs and send_names:
        return f"{head}. They need {', '.join(needs)}, and your {send_names} fill that hole."
    if needs:
        return f"{head}. They need {', '.join(needs)}."
    if send_names:
        return f"{head} could use your {send_names}."
    return f"{head} are open to deals."


def _acceptance_line(p: dict) -> str:
    rate = p.get("acceptance_rate")
    try:
        r = int(rate)
    except (TypeError, ValueError):
        return ""
    if r <= 0:
        return ""
    return f"~{r}% chance they accept"


def _render_trade_suggestions_fallback(ctx: dict) -> str:
    needs = ctx.get("viewer_needs") or []
    surplus = ctx.get("viewer_surplus") or []
    partners = ctx.get("top_partners") or []
    pick_partners = ctx.get("pick_trade_partners") or []
    projected_picks = ctx.get("projected_picks") or []

    needs_str = html.escape(", ".join(needs) if needs else "None identified")
    surplus_str = html.escape(", ".join(surplus) if surplus else "None identified")

    partner_rows = ""
    for p in partners[:3]:
        pname = html.escape(str(p.get("team_name") or ""))
        targets = p.get("targets_they_have") or []
        sends = p.get("targets_viewer_sends") or []
        is_pkg = p.get("is_package_trade", False)
        target_names = html.escape(", ".join(t["name"] for t in targets[:2]) or "players at your needed positions")
        send_names = html.escape(", ".join(t["name"] for t in sends[:2]))
        # Surface target tiers (elite / buy-low) next to names; realistic stays plain.
        disp_names = []
        for t in targets[:2]:
            tl = _tier_label(t.get("tier"))
            nm = t.get("name") or ""
            disp_names.append(f"{nm} ({tl.lower()})" if tl and tl != "Realistic" else nm)
        target_line = html.escape(", ".join(disp_names) or "players at your needed positions")
        if is_pkg and len(sends) >= 2:
            title = f"Package Deal: {pname}"
            reasoning = f"Package {send_names} to acquire {target_line} - converts surplus depth into an elite upgrade."
        else:
            title = f"Target: {pname}"
            send_part = f" - offer {send_names}" if send_names else ""
            reasoning = f"They have depth at {html.escape(', '.join(p.get('partner_surplus') or []))} - target {target_line}{send_part}."

        # Two-sided: why the partner says yes, not just why you want it.
        reasoning += " " + _partner_motivation(p)

        # Higher positional fit + a fairer deal = a more actionable suggestion.
        fairness = p.get("fairness") or 0
        urgency_cls, urgency_txt = ("urgency-low", "worth exploring")
        if p.get("match_score", 0) >= 4 and fairness >= 0.80:
            urgency_cls, urgency_txt = ("urgency-high", "strong fit")
        elif p.get("match_score", 0) >= 2 and fairness >= 0.72:
            urgency_cls, urgency_txt = ("urgency-medium", "good fit")
        fair_txt = _fairness_label(fairness, p.get("value_you_get"), p.get("value_you_give"))
        acpt_txt = _acceptance_line(p)
        meta_bits = [b for b in [fair_txt, acpt_txt] if b]
        meta_html = (
            f"<div class=\"suggestion-meta-line\">{html.escape(' · '.join(meta_bits))}</div>"
            if meta_bits else ""
        )
        partner_rows += f"""
        <div class="suggestion-card">
          <div class="suggestion-title">{html.escape(title)}</div>
          <div class="suggestion-reasoning">{html.escape(reasoning)}</div>
          <div class="suggestion-urgency {urgency_cls}">{urgency_txt}</div>
          {meta_html}
        </div>
        """

    # Pick-for-player suggestions
    for p in pick_partners[:3]:
        pname = html.escape(str(p.get("team_name") or ""))
        targets = p.get("targets_they_have") or []
        picks = p.get("picks_you_offer") or []
        target_names = html.escape(", ".join(t["name"] for t in targets[:2]))
        pick_labels = ", ".join(_fmt_pick_label(pk) for pk in picks[:2])
        if not target_names or not pick_labels:
            continue
        partner_rows += f"""
        <div class="suggestion-card">
          <div class="suggestion-title">Pick Trade: {pname}</div>
          <div class="suggestion-reasoning">Offer {html.escape(pick_labels)} to acquire {target_names}.</div>
          <div class="suggestion-urgency urgency-low">pick offer</div>
        </div>
        """

    if not partner_rows:
        partner_rows = "<p>No strong trade partners identified based on positional fit.</p>"

    return f"""
    <div class="trade-suggestions-wrap">
      <div class="suggestion-meta">
        <span>Needs: <strong>{needs_str}</strong></span>
        <span>Surplus: <strong>{surplus_str}</strong></span>
      </div>
      {partner_rows}
    </div>
    """


def _render_trade_suggestions_from_data(suggestions: list[dict], partners: list[dict] | None = None) -> str:
    if not suggestions:
        return "<p>No specific trade suggestions generated.</p>"

    # Enrich AI cards with computed context: match each suggestion back to its
    # partner by team name for fairness, acceptance, record/direction, tiers.
    _pmap = {str(p.get("team_name") or ""): p for p in (partners or [])}

    cards = ""
    for s in suggestions:
        title = html.escape(str(s.get("title") or "Trade Idea"))
        partner = html.escape(str(s.get("partner_team") or ""))
        reasoning = html.escape(str(s.get("reasoning") or ""))
        urgency = (s.get("urgency") or "medium").lower()
        you_give_raw = [str(x) for x in (s.get("you_give") or [])]
        you_get_raw = [str(x) for x in (s.get("you_get") or [])]

        if not you_give_raw or not you_get_raw:
            continue  # skip incomplete suggestions (no named players on one side)

        _p = _pmap.get(str(s.get("partner_team") or ""))
        _tier_by_name = {}
        if _p:
            for t in (_p.get("targets_they_have") or []):
                _tier_by_name[str(t.get("name") or "").lower()] = _tier_label(t.get("tier"))

        def _asset_span(raw: str, side: str) -> str:
            x = html.escape(raw)
            tl = _tier_by_name.get(raw.lower(), "") if side == "get" else ""
            badge = (
                f" <span class=\"suggestion-tier\" style=\"font-size:11px;font-weight:700;"
                f"padding:1px 5px;border-radius:3px;background:#f1f5f9;color:#475569;\">"
                f"{html.escape(tl.lower())}</span>"
                if tl and tl != "Realistic" else ""
            )
            return f"<span class='suggestion-asset {side}'>{x}{badge}</span>"

        give_html = "".join(_asset_span(x, "give") for x in you_give_raw)
        get_html = "".join(_asset_span(x, "get") for x in you_get_raw)

        trade_type = (s.get("trade_type") or "swap").lower()
        _type_labels = {
            "up_tier":   ("Up Tier", "#047857"),
            "down_tier": ("Down Tier", "#b45309"),
            "swap":      ("Swap", "#0369a1"),
        }
        type_label, type_color = _type_labels.get(trade_type, ("Swap", "#0369a1"))
        type_badge = (
            f"<span style='font-size:11px;font-weight:700;padding:2px 7px;border-radius:4px;"
            f"background:{type_color}18;color:{type_color};border:1px solid {type_color}40;'>"
            f"{html.escape(type_label)}</span>"
        )

        # Partner context line: record + direction so the "why they'd say yes"
        # is visible even when the AI reasoning stays one-sided.
        _pline = f"<div class=\"suggestion-partner\">Partner: {partner}</div>"
        _meta_bits: list = []
        if _p:
            _rec = str(_p.get("partner_record") or "").strip()
            _dir = str(_p.get("partner_direction") or "").strip()
            _rd = " ".join(x for x in [_rec, f"({_dir})" if _dir else ""] if x).strip()
            if _rd:
                _pline = (
                    f"<div class=\"suggestion-partner\">Partner: {partner} "
                    f"<span style=\"color:var(--text-muted);font-size:11px;\">"
                    f"{html.escape(_rd)}</span></div>"
                )
            _fair = _fairness_label(_p.get("fairness"), _p.get("value_you_get"), _p.get("value_you_give"))
            if _fair:
                _meta_bits.append(_fair)
            _acpt = _acceptance_line(_p)
            if _acpt:
                _meta_bits.append(_acpt)
        _meta_html = (
            f"<div class=\"suggestion-meta-line\" style=\"font-size:11px;color:var(--text-muted);"
            f"margin-top:6px;\">{html.escape(' · '.join(_meta_bits))}</div>"
            if _meta_bits else ""
        )

        cards += f"""
        <div class="suggestion-card">
          <div class="suggestion-header">
            <div class="suggestion-title">{title}</div>
            <div style="display:flex;gap:5px;align-items:center;">{type_badge}<div class="suggestion-urgency urgency-{urgency}">{urgency}</div></div>
          </div>
          {_pline}
          <div class="suggestion-assets">
            <div class="suggestion-side"><div class="suggestion-side-label">You give</div>{give_html}</div>
            <div class="suggestion-arrow">⇄</div>
            <div class="suggestion-side"><div class="suggestion-side-label">You get</div>{get_html}</div>
          </div>
          <div class="suggestion-reasoning">{reasoning}</div>
          {_meta_html}
        </div>
        """

    return f"""<div class="trade-suggestions-wrap">{cards}</div>"""
