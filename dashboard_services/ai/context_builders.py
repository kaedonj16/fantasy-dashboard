from __future__ import annotations

from typing import Any, Union

# Core fantasy positions used for scarcity analysis
_SCARCITY_POSITIONS = {"QB", "RB", "WR", "TE"}


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _ctx_is_sf(ctx: dict) -> bool:
    """Return True if the league context indicates a SuperFlex format."""
    rp = ctx.get("roster_positions") or []
    if isinstance(rp, list):
        return any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in rp)
    return False


def build_model_value_lookup(model_value_table: list[dict], is_sf: bool = False) -> dict[str, dict]:
    """Return pid→row lookup. When is_sf=True, rewrites each row's 'value' to the
    sf_value so all downstream callers automatically use the right format."""
    out: dict[str, dict] = {}
    for row in model_value_table or []:
        pid = str(row.get("id") or row.get("player_id") or "")
        if not pid:
            continue
        if is_sf and row.get("sf_value") is not None:
            row = {**row, "value": row["sf_value"]}
        out[pid] = row
    return out


def summarize_roster_players(
        roster: dict,
        players_index: dict,
        players_map: dict,
        model_value_lookup: dict[str, dict],
) -> list[dict]:
    out: list[dict] = []
    for pid in roster.get("players") or []:
        spid = str(pid)
        meta = players_index.get(spid) or players_map.get(spid) or {}
        mv = model_value_lookup.get(spid) or {}

        name = meta.get("full_name") or meta.get("name") or mv.get("name") or "Unknown"
        pos = meta.get("position") or meta.get("pos") or mv.get("position") or "?"
        team = meta.get("team") or mv.get("team") or "FA"
        age = meta.get("age") or mv.get("age")
        value = _safe_float(mv.get("value") or mv.get("sf_value") or mv.get("model_value") or mv.get("trade_value"), 0.0)

        out.append({
            "id": spid,
            "name": name,
            "position": pos,
            "team": team,
            "age": age,
            "value": round(value, 1),
        })

    out.sort(key=lambda x: x["value"], reverse=True)
    return out


def group_position_strength(players: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[float]] = {}
    for p in players:
        pos = p.get("position") or "?"
        buckets.setdefault(pos, []).append(_safe_float(p.get("value")))

    out: dict[str, dict] = {}
    for pos, vals in buckets.items():
        vals = sorted(vals, reverse=True)
        out[pos] = {
            "count": len(vals),
            "top_3_sum": round(sum(vals[:3]), 1),
            "top_5_sum": round(sum(vals[:5]), 1),
            "best": round(vals[0], 1) if vals else 0.0,
        }
    return out


def detect_team_direction(players: list[dict], future_picks: list[dict]) -> str:
    ages = [
        _safe_float(p.get("age"))
        for p in players
        if p.get("age") not in (None, "")
    ]
    avg_age = sum(ages) / len(ages) if ages else 0.0

    elite_assets = sum(1 for p in players if _safe_float(p.get("value")) >= 750)
    strong_assets = sum(1 for p in players if _safe_float(p.get("value")) >= 550)
    firsts = sum(1 for p in future_picks if "1." in str(p.get("display") or ""))

    if avg_age and avg_age <= 28.5:
        if elite_assets >= 3:
            return "contender"
        if elite_assets >= 2 and strong_assets >= 5:
            return "contender"
    if firsts >= 2 and avg_age and avg_age >= 26.8 and elite_assets < 2:
        return "retool"
    if firsts >= 3 and strong_assets <= 3:
        return "rebuild"
    return "balanced"


def build_team_gm_context(ctx: dict, viewer_roster_id: str) -> Union[dict, None]:
    rosters = ctx.get("rosters") or []
    roster = next((r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)), None)
    if not roster:
        return None

    roster_map = ctx.get("roster_map") or {}
    team_name = roster_map.get(str(viewer_roster_id)) or f"Roster {viewer_roster_id}"

    model_value_lookup = build_model_value_lookup(ctx.get("model_value_table") or [], is_sf=_ctx_is_sf(ctx))
    roster_players = summarize_roster_players(
        roster=roster,
        players_index=ctx.get("players_index") or {},
        players_map=ctx.get("players_map") or {},
        model_value_lookup=model_value_lookup,
    )

    position_strength = group_position_strength(roster_players)
    future_picks = ctx.get("picks_by_roster", {}).get(str(viewer_roster_id), [])

    top_assets = roster_players[:8]
    aging_assets = [
        p for p in roster_players
        if p.get("age") not in (None, "") and _safe_float(p.get("age")) >= 28 and _safe_float(p.get("value")) >= 300
    ][:5]

    direction = detect_team_direction(roster_players, future_picks)

    standing = (ctx.get("standings_map") or {}).get(str(viewer_roster_id), {})
    record = standing.get("record") or standing.get("display_record") or ""
    pf = _safe_float(standing.get("PF"))
    pa = _safe_float(standing.get("PA"))

    return {
        "league_id": ctx.get("league_id"),
        "season": ctx.get("current_season"),
        "week": ctx.get("current_week"),
        "viewer_roster_id": str(viewer_roster_id),
        "team_name": team_name,
        "record": record,
        "points_for": round(pf, 1),
        "points_against": round(pa, 1),
        "direction": direction,
        "top_assets": top_assets,
        "aging_assets": aging_assets,
        "future_picks": future_picks,
        "position_strength": position_strength,
        "roster_size": len(roster_players),
    }


def _safe_str(v, default: str = "") -> str:
    if v is None:
        return default
    return str(v).strip() or default


def _format_pick_display(pk: str) -> str:
    """
    Converts:
      2026_1_04 -> 2026 1.04
      2026_1_early -> 2026 Early 1st
      fallback -> original string
    """
    raw = _safe_str(pk)
    if not raw:
        return ""

    parts = raw.split("_")
    if len(parts) == 3:
        year, rnd, slot = parts
        if slot.isdigit():
            return f"{year} {rnd}.{slot.zfill(2)}"
        bucket_label = slot.capitalize()
        suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(_safe_int(rnd), f"{rnd}th")
        return f"{year} {suffix} ({bucket_label})"

    if len(parts) == 2:
        year, rnd = parts
        suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(_safe_int(rnd), f"{rnd}th")
        return f"{year} {suffix} (Mid)"

    return raw


def _find_roster(ctx: dict, roster_id: str) -> Union[dict, None]:
    for r in ctx.get("rosters") or []:
        if str(r.get("roster_id")) == str(roster_id):
            return r
    return None


def _top_position_edges(position_strength: dict[str, dict]) -> tuple[str, str]:
    if not position_strength:
        return "Unknown", "Unknown"

    scored = []
    for pos, meta in position_strength.items():
        scored.append((pos, _safe_float(meta.get("top_3_sum")), _safe_float(meta.get("best"))))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best = scored[0][0]
    worst = scored[-1][0]
    return best, worst


def build_front_office_brief_context(ctx: dict, viewer_roster_id: str) -> Union[dict, None]:
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx:
        return None

    pos_strength = team_ctx.get("position_strength") or {}
    best_pos, worst_pos = _top_position_edges(pos_strength)

    top_assets = team_ctx.get("top_assets") or []
    future_picks = team_ctx.get("future_picks") or []

    return {
        "league_id": team_ctx.get("league_id"),
        "season": team_ctx.get("season"),
        "week": team_ctx.get("week"),
        "team_name": team_ctx.get("team_name"),
        "record": team_ctx.get("record"),
        "direction": team_ctx.get("direction"),
        "points_for": team_ctx.get("points_for"),
        "points_against": team_ctx.get("points_against"),
        "best_position": best_pos,
        "weakest_position": worst_pos,
        "top_assets": top_assets[:6],
        "aging_assets": (team_ctx.get("aging_assets") or [])[:4],
        "future_picks": future_picks[:8],
        "position_strength": pos_strength,
    }


def build_trade_ai_context(
        ctx: dict,
        viewer_roster_id: str,
        viewer_side: str,
        side_a: dict,
        side_b: dict,
) -> Union[dict, None]:
    """
    side_a / side_b are the objects your /api/trade-eval route already builds.
    """
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx:
        return None

    viewer_side = (viewer_side or "a").lower().strip()
    viewer_gets = side_a if viewer_side == "a" else side_b
    viewer_gives = side_b if viewer_side == "a" else side_a

    def clean_asset(asset: dict) -> dict:
        return {
            "id": _safe_str(asset.get("id")),
            "name": _safe_str(asset.get("name")),
            "position": _safe_str(asset.get("position")),
            "team": _safe_str(asset.get("team")),
            "age": asset.get("age"),
            "value": round(_safe_float(asset.get("value")), 1),
        }

    def clean_pick(pk: Any) -> dict:
        raw = _safe_str(pk)
        return {
            "id": raw,
            "display": _format_pick_display(raw),
        }

    return {
        "viewer_team": {
            "roster_id": str(viewer_roster_id),
            "team_name": team_ctx.get("team_name"),
            "direction": team_ctx.get("direction"),
            "record": team_ctx.get("record"),
            "points_for": team_ctx.get("points_for"),
            "points_against": team_ctx.get("points_against"),
            "top_assets": (team_ctx.get("top_assets") or [])[:6],
            "aging_assets": (team_ctx.get("aging_assets") or [])[:4],
            "future_picks": (team_ctx.get("future_picks") or [])[:8],
            "position_strength": team_ctx.get("position_strength") or {},
        },
        "viewer_side": viewer_side,
        "viewer_gets": {
            "players": [clean_asset(a) for a in (viewer_gets.get("assets") or []) if str(a.get("position")) != "PICK"],
            "picks": [clean_pick(pk) for pk in (viewer_gets.get("pick_ids") or [])],
            "raw_total": round(_safe_float(viewer_gets.get("raw_total")), 1),
            "effective_total": round(_safe_float(viewer_gets.get("effective_total")), 1),
            "adjustment": round(_safe_float(viewer_gets.get("adjustment")), 1),
        },
        "viewer_gives": {
            "players": [clean_asset(a) for a in (viewer_gives.get("assets") or []) if str(a.get("position")) != "PICK"],
            "picks": [clean_pick(pk) for pk in (viewer_gives.get("pick_ids") or [])],
            "raw_total": round(_safe_float(viewer_gives.get("raw_total")), 1),
            "effective_total": round(_safe_float(viewer_gives.get("effective_total")), 1),
            "adjustment": round(_safe_float(viewer_gives.get("adjustment")), 1),
        },
        "net_effective_delta": round(
            _safe_float(viewer_gets.get("effective_total")) - _safe_float(viewer_gives.get("effective_total")),
            1,
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Roster Grade
# ──────────────────────────────────────────────────────────────────────────────

_GRADE_THRESHOLDS = [
    (90, "A+"), (82, "A"), (74, "A-"),
    (67, "B+"), (60, "B"), (52, "B-"),
    (44, "C+"), (36, "C"), (28, "C-"),
    (0, "D"),
]

def _compute_win_window(
    avg_age: float,
    firsts: int,
    dynasty_pct: float,
    redraft_pct: float,
    dr_ratio: float,
) -> str:
    """
    Classify a team's competitive window purely from roster construction.

    No current-season results (wins, PF, standings) are used — those are noisy
    and miss the point of a dynasty window. Signals:

    dynasty_pct  – 0-1 percentile of total dynasty value vs league (long-term strength)
    redraft_pct  – 0-1 percentile of total redraft value vs league (scoring projection NOW)
    dr_ratio     – avg(dynasty / redraft) for top starters
                   >1.2 = future-heavy (young, upside not yet realized)
                   <0.9 = win-now (aging/peaking, redraft ≥ dynasty)
    avg_age      – top-2-weighted average age of top-8 players
    firsts       – future 1st round picks owned
    """
    young  = avg_age <= 26.5
    prime  = 26.5 < avg_age <= 28.5
    aging  = avg_age > 28.5

    win_now_roster = dr_ratio <= 0.90   # players more valuable now than long-term
    future_roster  = dr_ratio >= 1.20   # players more valuable long-term than now

    # Value tier flags
    dynasty_elite  = dynasty_pct >= 0.70   # top 30% of league
    dynasty_strong = dynasty_pct >= 0.55   # top 45%
    dynasty_avg    = dynasty_pct >= 0.40   # above average

    redraft_elite  = redraft_pct >= 0.70   # projecting to score in top 30%
    redraft_strong = redraft_pct >= 0.55   # top 45%
    redraft_avg    = redraft_pct >= 0.42   # above average

    # ── Transitional / negative ───────────────────────────────────────────────

    # Full Rebuild — deliberate asset accumulation: many 1sts + weak current roster
    if firsts >= 3 and dynasty_pct <= 0.45:
        return "Full Rebuild"

    # Retooling — have picks AND aging/declining profile (trading away the peak)
    if firsts >= 2 and (aging or (prime and win_now_roster)):
        return "Retooling"

    # Rebuilding — weak on both dynasty and projected scoring, no tank capital
    if dynasty_pct <= 0.35 and redraft_pct <= 0.40:
        return "Rebuilding"

    # ── Win-now tier (projecting well NOW, window is short) ───────────────────

    # Win-Now — elite projected scoring + aging/peaking roster + decent dynasty
    # These teams need to capitalize immediately; their window is closing
    if redraft_elite and dynasty_strong and (aging or win_now_roster):
        return "Win-Now"

    # Aging Contender — strong projected scoring but aging roster, dynasty declining
    if redraft_strong and aging and dynasty_avg:
        return "Aging Contender"

    # ── Prime contender tier (strong on both axes) ────────────────────────────

    # Contender — elite dynasty AND strong projected scoring (best of both worlds)
    if dynasty_elite and redraft_strong:
        return "Contender"

    # ── Building toward peak (dynasty strong, scoring still developing) ───────

    # Contender Window — elite dynasty + young profile; peak scoring is coming
    # These rosters have everything, they just haven't hit their ceiling yet
    if dynasty_elite and (young or prime) and not win_now_roster:
        return "Contender Window"

    # Also Contender Window for strong dynasty + above-avg projected scoring + young
    if dynasty_strong and redraft_avg and young:
        return "Contender Window"

    # 2-3 Year Window — solid dynasty + future-heavy (upside not yet realized)
    if dynasty_strong and (future_roster or young):
        return "2-3 Year Window"

    # Rising — decent dynasty + very young + future-heavy roster
    if dynasty_avg and young and future_roster:
        return "Rising"

    return "Holding Pattern"


def calculate_roster_grade(
    players: list[dict],
    future_picks: list[dict],
    position_ranks: dict | None = None,
    num_teams: int = 12,
    # League-context signals — passed from the team page when available
    dynasty_pct_val: float = -1.0,   # 0-1 percentile of team's dynasty value (-1 = not provided)
    redraft_pct_val: float = 0.5,    # 0-1 percentile of team's redraft value (scoring projection)
    dr_ratio: float = 1.0,           # avg(dynasty/redraft) for top starters
    # Deprecated season-result params kept for call-site compatibility but not used in window logic
    pf_pct_val: float = 0.5,
    win_rate: float | None = None,
    standings_rank: int = 0,
    offseason: bool = False,
) -> dict:
    """
    Score a dynasty roster and compute a competitive window label.

    The window is based purely on roster construction:
      dynasty_pct_val  — long-term value percentile vs league
      redraft_pct_val  — projected scoring percentile vs league (not actual PF)
      dr_ratio         — dynasty/redraft spread (future-heavy vs win-now)
      avg_age          — top-2-weighted age of top-8 players
      firsts           — future first-round picks

    When called without league context (AI renderer, trade strategy), falls back
    to age/depth/capital/elite scoring with positional rank proxy.
    """
    top8 = players[:8]

    # Age — weight the top-2 players at 2× since they define the window
    ages_all = [_safe_float(p.get("age")) for p in top8 if p.get("age") not in (None, "")]
    if ages_all:
        weights = [2.0 if i < 2 else 1.0 for i in range(len(ages_all))]
        avg_age = sum(a * w for a, w in zip(ages_all, weights)) / sum(weights)
    else:
        avg_age = 28.0
    if avg_age <= 23:
        age_score = 95
    elif avg_age <= 25:
        age_score = 85
    elif avg_age <= 27:
        age_score = 70
    elif avg_age <= 28.5:
        age_score = 55
    elif avg_age <= 30:
        age_score = 38
    else:
        age_score = 22

    # Depth Score - positions with 2+ players worth ≥350
    pos_counts: dict[str, int] = {}
    for p in players:
        if _safe_float(p.get("value")) >= 350:
            pos = str(p.get("position") or "?").upper()
            if pos in ("QB", "RB", "WR", "TE"):
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
    deep_positions = sum(1 for cnt in pos_counts.values() if cnt >= 2)
    depth_score = min(deep_positions * 25, 100)

    # Capital Score - future picks weighted
    capital = 0
    firsts = 0
    for pk in future_picks:
        display = str(pk.get("display") or pk.get("id") or "")
        raw = str(pk.get("id") or "")
        parts = raw.split("_") if "_" in raw else []
        try:
            rnd = int(parts[1]) if len(parts) >= 2 else (1 if "1st" in display or "1." in display else 2)
        except Exception:
            rnd = 3
        if rnd == 1:
            capital += 100
            firsts += 1
        elif rnd == 2:
            capital += 40
        else:
            capital += 10
    capital_score = min(int(capital / 3), 100)

    # Elite Core Score - players above 700
    elite_count = sum(1 for p in players if _safe_float(p.get("value")) >= 700)
    elite_score = min(elite_count * 33, 100)

    # Rank Score (used only in fallback grade when no league context)
    rank_score = 50.0
    if position_ranks and num_teams > 1:
        ranks = [v for v in position_ranks.values() if v is not None]
        if ranks:
            avg_rank = sum(ranks) / len(ranks)
            rank_score = round(100.0 * (1.0 - (avg_rank - 1.0) / (num_teams - 1.0)), 1)
            rank_score = max(0.0, min(100.0, rank_score))

    # ── Grade score ──────────────────────────────────────────────────────────────
    has_league_ctx = dynasty_pct_val >= 0.0
    if has_league_ctx:
        dynasty_score = dynasty_pct_val * 100
        redraft_score = redraft_pct_val * 100
        total = round(
            dynasty_score * 0.40   # long-term roster quality
            + redraft_score * 0.25  # projected scoring NOW
            + age_score     * 0.15  # window length
            + elite_score   * 0.12  # franchise-player quality
            + capital_score * 0.08, # rebuild resources
            1,
        )
    elif position_ranks:
        total = round(
            age_score    * 0.18
            + depth_score  * 0.13
            + capital_score * 0.07
            + elite_score  * 0.12
            + rank_score   * 0.50,
            1,
        )
    else:
        total = round(
            age_score    * 0.36
            + depth_score  * 0.26
            + capital_score * 0.14
            + elite_score  * 0.24,
            1,
        )

    grade = "D"
    for threshold, letter in _GRADE_THRESHOLDS:
        if total >= threshold:
            grade = letter
            break

    # ── Win window ───────────────────────────────────────────────────────────────
    if has_league_ctx:
        win_window = _compute_win_window(
            avg_age=avg_age,
            firsts=firsts,
            dynasty_pct=dynasty_pct_val,
            redraft_pct=redraft_pct_val,
            dr_ratio=dr_ratio,
        )
    else:
        # Fallback for callers without league context (AI renderer, trade strategy)
        # Approximate using positional rank score as a redraft proxy
        _young = avg_age <= 27.0
        _aging = avg_age > 28.5
        _dr = dr_ratio  # may still carry meaning even in fallback
        _firsts = firsts
        # Map rank_score (0-100) to 0-1 percentile proxy
        _rank_pct = rank_score / 100.0
        win_window = _compute_win_window(
            avg_age=avg_age,
            firsts=_firsts,
            dynasty_pct=_rank_pct,    # positional rank as dynasty proxy
            redraft_pct=_rank_pct,    # same proxy for both axes in fallback
            dr_ratio=_dr,
        )

    return {
        "score": total,
        "grade": grade,
        "win_window": win_window,
        "breakdown": {
            "age_score": age_score,
            "depth_score": depth_score,
            "capital_score": capital_score,
            "elite_score": elite_score,
            "rank_score": rank_score,
            "avg_age": round(avg_age, 1),
            "elite_count": elite_count,
            "deep_positions": deep_positions,
            "dynasty_pct": round(dynasty_pct_val, 3) if has_league_ctx else None,
            "redraft_pct": round(redraft_pct_val, 3) if has_league_ctx else None,
            "dr_ratio": round(dr_ratio, 3) if has_league_ctx else None,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Roster Depth Warning (replaces league-wide positional scarcity)
# ──────────────────────────────────────────────────────────────────────────────

# Fallback thresholds used when no league context is available
_STARTER_THRESHOLD = {"QB": 500, "RB": 350, "WR": 350, "TE": 400}
_DEPTH_FLOOR = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}


def _derive_league_thresholds(
    roster_positions: list[str],
    num_teams: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Derive starter-caliber value thresholds and depth floors from actual
    league settings.

    Depth floor  = number of that position in the starting lineup
                   (including FLEX, split evenly across RB/WR).
    Value threshold scales down with league size: larger leagues spread
    talent thinner, so a lower absolute value still constitutes a starter.
    """
    pos_counts: dict[str, int] = {}
    flex_count = 0
    for slot in roster_positions:
        s = str(slot).upper()
        if s in ("QB", "RB", "WR", "TE"):
            pos_counts[s] = pos_counts.get(s, 0) + 1
        elif s in ("FLEX", "RB_WR_FLEX", "RB_WR_TE", "WR_RB", "WR_TE", "RB_WR"):
            flex_count += 1

    # Distribute FLEX evenly across RB and WR (most common usage)
    rb_flex = flex_count // 2
    wr_flex = flex_count - rb_flex
    floor: dict[str, int] = {
        "QB": max(1, pos_counts.get("QB", 1)),
        "RB": max(1, pos_counts.get("RB", 1) + rb_flex),
        "WR": max(1, pos_counts.get("WR", 1) + wr_flex),
        "TE": max(1, pos_counts.get("TE", 1)),
    }

    # Value threshold: base on 12-team league, scale linearly with team count.
    # More teams = talent diluted = lower bar to be a starter.
    scale = 12 / max(num_teams, 6)
    threshold: dict[str, int] = {
        "QB": round(500 * scale),
        "RB": round(350 * scale),
        "WR": round(350 * scale),
        "TE": round(400 * scale),
    }
    return threshold, floor


def calculate_roster_depth_warning(
        viewer_roster: dict,
        model_value_lookup: dict[str, dict],
        sending_assets: list[dict],
        receiving_assets: list[dict],
        roster_positions: list[str] | None = None,
        num_teams: int = 12,
) -> dict[str, dict]:
    """
    Warn when a trade leaves the viewer dangerously thin at a position.

    Simulates the post-trade roster: removes sent players, adds received players,
    then counts starter-caliber players at each affected position.

    Returns {pos: {before, after, warning, severity}} only for positions
    where the trade changes depth or triggers a warning.
    severity: 'danger' | 'caution' | None
    """
    # Derive league-specific thresholds
    if roster_positions:
        starter_threshold, depth_floor = _derive_league_thresholds(roster_positions, num_teams)
    else:
        starter_threshold, depth_floor = _STARTER_THRESHOLD, _DEPTH_FLOOR

    # Build current roster value map
    roster_values: dict[str, dict] = {}
    for pid in viewer_roster.get("players") or []:
        spid = str(pid)
        mv = model_value_lookup.get(spid) or {}
        pos = str(mv.get("position") or "").upper()
        val = _safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value"))
        if pos in starter_threshold and val > 0:
            roster_values[spid] = {"pos": pos, "val": val, "name": str(mv.get("name") or spid)}

    sending_ids = {str(a.get("id") or "") for a in sending_assets if a.get("id")}
    receiving_map: dict[str, dict] = {}
    for a in receiving_assets:
        pid = str(a.get("id") or "")
        pos = str(a.get("position") or "").upper()
        val = _safe_float(a.get("value"))
        if pid and pos in _STARTER_THRESHOLD and val > 0:
            receiving_map[pid] = {"pos": pos, "val": val, "name": str(a.get("name") or pid)}

    # Identify positions actually touched by this trade
    touched_positions: set[str] = set()
    for a in sending_assets:
        pos = str(a.get("position") or "").upper()
        if pos in _STARTER_THRESHOLD:
            touched_positions.add(pos)
    for a in receiving_assets:
        pos = str(a.get("position") or "").upper()
        if pos in _STARTER_THRESHOLD:
            touched_positions.add(pos)

    result: dict[str, dict] = {}
    for pos in touched_positions:
        threshold = _STARTER_THRESHOLD[pos]
        floor = _DEPTH_FLOOR[pos]

        # Count before
        before = sum(
            1 for pid, info in roster_values.items()
            if info["pos"] == pos and info["val"] >= threshold
        )

        # Simulate: remove sent, add received
        post_roster = {
            pid: info for pid, info in roster_values.items()
            if pid not in sending_ids
        }
        for pid, info in receiving_map.items():
            if info["pos"] == pos:
                post_roster[pid] = info

        after = sum(1 for info in post_roster.values() if info["pos"] == pos and info["val"] >= threshold)

        # Only warn when THIS trade actually reduces depth at the position.
        # Acquiring (or not touching) a position never triggers a depth warning,
        # even if it's already thin — that's a pre-existing condition, not caused
        # by this trade. Prevents "low RB depth" alerts when you're receiving an RB.
        if after >= before:
            continue

        if after == 0:
            warning = f"You'll have no starter-caliber {pos} after this trade"
            severity = "danger"
        elif after < floor:
            warning = f"Leaves you with only {after} starter-caliber {pos} (need {floor})"
            severity = "caution"
        else:
            warning = f"{pos} depth drops from {before} to {after} starters"
            severity = "caution"

        result[pos] = {
            "before": before,
            "after": after,
            "warning": warning,
            "severity": severity,
        }

    return result


def _fetch_real_trade_patterns(
    viewer_player_ids: list[str],
    is_sf: bool,
    num_teams: int,
    lookback_days: int = 180,
) -> dict[str, list[dict]]:
    """
    Query trade_intel to find what viewer's players have recently been traded for
    in comparable leagues (dynasty, same superflex format, similar size ±2).

    Returns {player_id: [{asset_type, player_id, pick_season, pick_round,
                          pick_order, frequency, last_seen}, ...]}
    sorted by frequency desc per player.  Only included when enough trades exist.
    """
    if not viewer_player_ids:
        return {}
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                """
                WITH relevant AS (
                    SELECT DISTINCT t.id AS trade_id,
                           a_out.player_id AS sent_player,
                           a_out.side      AS sent_side
                    FROM trade_intel_trades t
                    JOIN trade_intel_leagues l ON l.league_id = t.league_id
                    JOIN trade_intel_assets a_out
                         ON a_out.trade_id = t.id
                        AND a_out.asset_type = 'player'
                        AND a_out.player_id = ANY(%s)
                    WHERE l.league_type = 2
                      AND COALESCE(l.is_superflex, FALSE) = %s
                      AND COALESCE(l.num_teams, 12) BETWEEN %s AND %s
                      AND t.created_at > NOW() - INTERVAL '%s days'
                )
                SELECT
                    r.sent_player,
                    a_in.asset_type,
                    a_in.player_id   AS recv_player_id,
                    a_in.pick_season,
                    a_in.pick_round,
                    a_in.pick_order,
                    COUNT(*)         AS frequency,
                    MAX(t.created_at) AS last_seen
                FROM relevant r
                JOIN trade_intel_trades t ON t.id = r.trade_id
                JOIN trade_intel_assets a_in
                     ON a_in.trade_id = r.trade_id
                    AND a_in.side != r.sent_side
                GROUP BY r.sent_player, a_in.asset_type, a_in.player_id,
                         a_in.pick_season, a_in.pick_round, a_in.pick_order
                HAVING COUNT(*) >= 2
                ORDER BY r.sent_player, COUNT(*) DESC
                LIMIT 400
                """,
                (viewer_player_ids, is_sf, num_teams - 2, num_teams + 2, lookback_days),
            ).fetchall()

        patterns: dict[str, list[dict]] = {}
        for row in rows:
            pid = str(row["sent_player"])
            patterns.setdefault(pid, []).append({
                "asset_type":  row["asset_type"],
                "player_id":   row["recv_player_id"],
                "pick_season": row["pick_season"],
                "pick_round":  row["pick_round"],
                "pick_order":  row["pick_order"],
                "frequency":   int(row["frequency"]),
                "last_seen":   row["last_seen"].isoformat() if row["last_seen"] else None,
            })
        return patterns
    except Exception:
        return {}


def _enrich_trade_patterns(
    patterns: dict[str, list[dict]],
    model_value_lookup: dict,
) -> list[dict]:
    """
    Convert raw DB patterns into suggestion-ready dicts with player names.

    Groups picks of the same round into a single label.  Returns list sorted
    by combined frequency of the sending player's patterns.
    """
    out = []
    for sent_pid, assets in patterns.items():
        if not assets:
            continue
        mv = model_value_lookup.get(sent_pid) or {}
        sent_name = str(mv.get("name") or sent_pid)
        sent_value = _safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value"))

        total_freq = sum(a["frequency"] for a in assets[:10])

        # Build the "received" summary: top individual assets
        received_summary = []
        seen: set[str] = set()
        for a in assets[:8]:
            if a["asset_type"] == "player" and a["player_id"]:
                r_pid = str(a["player_id"])
                if r_pid in seen:
                    continue
                seen.add(r_pid)
                rmv = model_value_lookup.get(r_pid) or {}
                received_summary.append({
                    "type":      "player",
                    "id":        r_pid,
                    "name":      str(rmv.get("name") or r_pid),
                    "position":  str(rmv.get("position") or "").upper(),
                    "value":     _safe_float(rmv.get("value") or rmv.get("model_value") or 0),
                    "frequency": a["frequency"],
                })
            elif a["asset_type"] == "pick":
                rnd = a["pick_round"]
                order = a["pick_order"] or ""
                label = f"{a['pick_season'] or 'Future'} {order.capitalize()} {rnd}{'st' if rnd == 1 else 'nd' if rnd == 2 else 'rd' if rnd == 3 else 'th'}" if rnd else "Future Pick"
                key = label
                if key in seen:
                    continue
                seen.add(key)
                received_summary.append({
                    "type":      "pick",
                    "label":     label,
                    "frequency": a["frequency"],
                })

        if not received_summary:
            continue

        out.append({
            "sent_player_id":    sent_pid,
            "sent_player_name":  sent_name,
            "sent_player_value": round(sent_value, 1),
            "received_assets":   received_summary[:4],
            "total_trades":      total_freq,
        })

    out.sort(key=lambda x: x["total_trades"], reverse=True)
    return out[:8]


# ──────────────────────────────────────────────────────────────────────────────
# Trade Suggestions Context
# ──────────────────────────────────────────────────────────────────────────────

def build_trade_suggestions_context(
        ctx: dict,
        viewer_roster_id: str,
) -> dict | None:
    """
    Build context for proactive trade suggestions.

    Identifies the viewer's positional needs/surplus, then finds leaguemates
    with complementary surpluses/needs and specific player targets.
    """
    rosters = ctx.get("rosters") or []
    roster = next((r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)), None)
    if not roster:
        return None

    model_value_lookup = build_model_value_lookup(ctx.get("model_value_table") or [], is_sf=_ctx_is_sf(ctx))
    roster_map = ctx.get("roster_map") or {}
    picks_by_roster = ctx.get("picks_by_roster") or {}

    def _roster_pos_totals(r: dict) -> dict[str, float]:
        totals: dict[str, float] = {pos: 0.0 for pos in _SCARCITY_POSITIONS}
        for pid in r.get("players") or []:
            spid = str(pid)
            mv = model_value_lookup.get(spid) or {}
            pos = str(mv.get("position") or "").upper()
            if pos in _SCARCITY_POSITIONS:
                totals[pos] += _safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value"))
        return totals

    def _roster_top_players(r: dict, pos: str, exclude_ids: set[str] | None = None) -> list[dict]:
        out = []
        for pid in r.get("players") or []:
            spid = str(pid)
            mv = model_value_lookup.get(spid) or {}
            p_pos = str(mv.get("position") or "").upper()
            if p_pos != pos:
                continue
            if exclude_ids and spid in exclude_ids:
                continue
            val = _safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value"))
            if val > 0:
                out.append({"id": spid, "name": str(mv.get("name") or spid), "value": round(val, 1), "position": pos})
        return sorted(out, key=lambda x: x["value"], reverse=True)

    n_teams = max(len(rosters), 1)
    viewer_totals = _roster_pos_totals(roster)
    viewer_player_ids = {str(pid) for pid in roster.get("players") or []}

    # Rank every roster by positional total (1 = best)
    roster_totals_map = {str(r.get("roster_id") or ""): _roster_pos_totals(r) for r in rosters}

    # Project the viewer's upcoming picks to actual rookies using the live rankings table.
    # Pick slot = standings rank of original owner (worst record = 1.01, etc.).
    # Supports both 1QB (value_1qb) and SF (value_sf).
    viewer_rid = str(viewer_roster_id)
    viewer_picks_list = picks_by_roster.get(viewer_rid, [])
    from datetime import datetime as _dt
    _cur_yr = _dt.now().year
    _league_type = str(ctx.get("league_type") or "1qb").lower()

    # Sorted rookie rankings from ctx (already ordered by overall_rank asc)
    _rookie_rankings: list[dict] = sorted(
        ctx.get("rookie_rankings") or [],
        key=lambda r: int(r.get("overall_rank") or 999),
    )
    # For SF use sf value, otherwise 1qb value
    _val_key = "value_sf" if _league_type == "sf" else "value_1qb"

    # Build pick-slot map: for each season+round, rank original owners by record
    # (worst record = slot 1 = earliest pick in that round)
    _standings_map = ctx.get("standings_map") or {}

    def _win_pct(rid: str) -> float:
        s = _standings_map.get(str(rid)) or {}
        w = _safe_float(s.get("wins") or s.get("W") or s.get("Wins") or 0)
        l = _safe_float(s.get("losses") or s.get("L") or s.get("Losses") or 0)
        return w / (w + l) if (w + l) > 0 else 0.5

    # Group all picks (across all rosters) by season+round to assign slots
    _all_picks_flat: list[dict] = []
    for _rid, _plist in picks_by_roster.items():
        for _pk in _plist:
            _all_picks_flat.append({**_pk, "current_owner": _rid})

    _slot_cache: dict[tuple, dict[str, int]] = {}  # (season,round) -> {original_owner: slot}

    def _get_slot(season: int, rnd: int, original_owner: str) -> int:
        key = (season, rnd)
        if key not in _slot_cache:
            picks_for_rnd = [p for p in _all_picks_flat
                             if p.get("season") == season and p.get("round") == rnd]
            unique_owners = list({str(p.get("original_owner", "")) for p in picks_for_rnd})
            # Sort worst record first → they get the earliest pick slot
            unique_owners.sort(key=lambda r: _win_pct(r))
            _slot_cache[key] = {o: i + 1 for i, o in enumerate(unique_owners)}
        return _slot_cache[key].get(str(original_owner), len(rosters))

    _pick_pos_credits: dict[str, float] = {}
    _projected_picks: list[dict] = []
    for _rnd in [1, 2]:
        _rnd_picks = sorted(
            [p for p in viewer_picks_list
             if p.get("round") == _rnd and int(p.get("season", 0)) <= _cur_yr + 1],
            key=lambda p: (p.get("season", 9999), _get_slot(
                int(p.get("season", _cur_yr)), _rnd, p.get("original_owner", "")
            )),
        )
        for _pk in _rnd_picks:
            _season = int(_pk.get("season", _cur_yr))
            _slot   = _get_slot(_season, _rnd, _pk.get("original_owner", ""))
            # For R1 the rookie index = slot-1; for R2 offset past all R1 rookies
            _rookie_base = 0 if _rnd == 1 else len(rosters)
            _rookie_idx  = _rookie_base + _slot - 1
            if _rookie_idx < len(_rookie_rankings):
                _proj = _rookie_rankings[_rookie_idx]
                _pos  = str(_proj.get("position", "")).upper()
                _val  = float(_proj.get(_val_key) or _proj.get("value_1qb") or 0)
                if _pos in _SCARCITY_POSITIONS:
                    _pick_pos_credits[_pos] = _pick_pos_credits.get(_pos, 0.0) + _val
                _projected_picks.append({
                    "season":    _season,
                    "round":     _rnd,
                    "slot":      _slot,
                    "proj_name": _proj.get("name", ""),
                    "proj_pos":  _pos,
                    "proj_val":  round(_val, 1),
                })
            else:
                _projected_picks.append({"season": _season, "round": _rnd, "slot": _slot})

    if _pick_pos_credits:
        viewer_totals = {
            pos: viewer_totals.get(pos, 0.0) + _pick_pos_credits.get(pos, 0.0)
            for pos in _SCARCITY_POSITIONS
        }
        roster_totals_map[viewer_rid] = viewer_totals
    pos_rank_map: dict[str, dict[str, int]] = {}  # rid -> {pos -> rank}
    for pos in _SCARCITY_POSITIONS:
        sorted_rids = sorted(
            roster_totals_map.keys(),
            key=lambda rid: roster_totals_map[rid].get(pos, 0.0),
            reverse=True,
        )
        for i, rid in enumerate(sorted_rids, start=1):
            pos_rank_map.setdefault(rid, {})[pos] = i

    viewer_ranks = pos_rank_map.get(viewer_rid, {})

    # Need = bottom 35% of league; Surplus = top 30% of league (rank-based)
    need_cutoff = max(1, round(n_teams * 0.35))
    surplus_cutoff = max(1, round(n_teams * 0.30))

    viewer_needs = [
        pos for pos in _SCARCITY_POSITIONS
        if viewer_ranks.get(pos, n_teams) > n_teams - need_cutoff
    ]
    viewer_surplus = [
        pos for pos in _SCARCITY_POSITIONS
        if viewer_ranks.get(pos, n_teams) <= surplus_cutoff
    ]

    # League averages for context only
    league_avg: dict[str, float] = {}
    for pos in _SCARCITY_POSITIONS:
        vals = [roster_totals_map[rid].get(pos, 0.0) for rid in roster_totals_map]
        league_avg[pos] = sum(vals) / len(vals) if vals else 0.0

    # Find best trade partners
    partners = []
    for r in rosters:
        rid = str(r.get("roster_id") or "")
        if rid == viewer_rid:
            continue

        partner_ranks = pos_rank_map.get(rid, {})
        partner_needs = [
            pos for pos in _SCARCITY_POSITIONS
            if partner_ranks.get(pos, n_teams) > n_teams - need_cutoff
        ]
        partner_surplus = [
            pos for pos in _SCARCITY_POSITIONS
            if partner_ranks.get(pos, n_teams) <= surplus_cutoff
        ]

        # Score compatibility: viewer's surplus matches partner's need and vice versa
        match_score = 0
        mutual_surplus_need = []
        for pos in viewer_surplus:
            if pos in partner_needs:
                match_score += 2
                mutual_surplus_need.append(pos)
        for pos in partner_surplus:
            if pos in viewer_needs:
                match_score += 2

        if match_score == 0:
            continue

        # Find specific player targets (partner's surplus positions that viewer needs)
        targets_they_have = []
        for pos in partner_surplus:
            if pos in viewer_needs:
                top = _roster_top_players(r, pos, exclude_ids=viewer_player_ids)[:2]
                targets_they_have.extend(top)

        # Surplus-liquidation path: viewer has surplus to give but no explicit need.
        # Partner can offer any high-value player at a position viewer is weak/neutral at.
        is_package_trade = False
        if not targets_they_have and mutual_surplus_need:
            is_package_trade = True
            # Positions viewer isn't in surplus at, prioritized by weakest rank first
            weaker_positions = sorted(
                [pos for pos in _SCARCITY_POSITIONS if pos not in viewer_surplus],
                key=lambda pos: viewer_ranks.get(pos, n_teams), reverse=True,
            )
            if not weaker_positions:
                weaker_positions = list(_SCARCITY_POSITIONS)
            for pos in weaker_positions:
                top = _roster_top_players(r, pos, exclude_ids=viewer_player_ids)[:1]
                targets_they_have.extend(p for p in top if p.get("value", 0) >= 350)
            targets_they_have.sort(key=lambda x: x["value"], reverse=True)
            targets_they_have = targets_they_have[:1]

        # Find what viewer could send (viewer's surplus that partner needs).
        # Surplus positions: keep ≥1 (they have a starter and can trade depth).
        # Non-surplus positions: keep ≥2 (don't gut a weak spot).
        # Last resort: allow sending 1 if getting that position back.
        getting_positions = {str(p.get("position", "")).upper() for p in targets_they_have}
        partner_player_ids = {str(p.get("id")) for p in targets_they_have}
        targets_viewer_sends = []
        for pos in viewer_surplus:
            if pos not in partner_needs:
                continue
            pos_depth = sum(
                1 for pid in (roster.get("players") or [])
                if str(model_value_lookup.get(str(pid), {}).get("position", "")).upper() == pos
            )
            min_keep = 1  # surplus positions can trade down to 1 starter
            max_sendable = max(0, pos_depth - min_keep)
            if max_sendable == 0:
                if pos in getting_positions:
                    max_sendable = 1
                else:
                    continue
            top = _roster_top_players(roster, pos, exclude_ids=partner_player_ids)[:min(2, max_sendable)]
            targets_viewer_sends.extend(top)

        # Package trade fallback: if viewer has a surplus at multiple positions and needs
        # to package them to match the target's value, pull from all surplus positions.
        if is_package_trade and not targets_viewer_sends and targets_they_have:
            target_val = targets_they_have[0]["value"]
            all_surplus_sendable = []
            for pos in viewer_surplus:
                pos_depth = sum(
                    1 for pid in (roster.get("players") or [])
                    if str(model_value_lookup.get(str(pid), {}).get("position", "")).upper() == pos
                )
                if pos_depth >= 1:
                    top = _roster_top_players(roster, pos)
                    # Send 2nd player onward at surplus positions (keep 1 starter)
                    all_surplus_sendable.extend(top[1:2] if pos_depth > 1 else top[:1])
            all_surplus_sendable.sort(key=lambda x: x["value"], reverse=True)
            # Build package until combined value reaches ~80% of target
            running = 0.0
            for p in all_surplus_sendable:
                if running >= target_val * 0.80:
                    break
                targets_viewer_sends.append(p)
                running += p["value"]

        # Only include partners where both sides have named players (avoids TBD suggestions)
        if not targets_they_have or not targets_viewer_sends:
            continue

        # Compute value sums and classify the trade type
        value_you_get = round(sum(p["value"] for p in targets_they_have[:3]), 1)
        value_you_give = round(sum(p["value"] for p in targets_viewer_sends[:3]), 1)

        if value_you_give > 0 and value_you_get / value_you_give >= 1.20:
            trade_type_hint = "up_tier"
        elif value_you_get > 0 and value_you_give / value_you_get >= 1.20:
            trade_type_hint = "down_tier"
        else:
            trade_type_hint = "swap"

        partners.append({
            "roster_id": rid,
            "team_name": roster_map.get(rid) or f"Team {rid}",
            "match_score": match_score,
            "partner_needs": partner_needs,
            "partner_surplus": partner_surplus,
            "targets_they_have": targets_they_have[:3],
            "targets_viewer_sends": targets_viewer_sends[:3],
            "value_you_get": value_you_get,
            "value_you_give": value_you_give,
            "trade_type_hint": trade_type_hint,
            "is_package_trade": is_package_trade,
        })

    partners.sort(key=lambda x: x["match_score"], reverse=True)

    # Pick-for-player suggestions: viewer offers a pick instead of a player.
    # Don't require partner_surplus - any team with a good player at a needed
    # position could be a pick trade target.
    pick_trade_partners = []
    if viewer_needs:
        # Include all upcoming picks as potential offers (with or without rookie projection)
        all_picks_as_offers = [
            p for p in _projected_picks
            if p.get("round") in (1, 2)
        ] or [
            {"season": pk.get("season"), "round": pk.get("round"), "proj_name": "TBD", "proj_pos": "TBD", "proj_val": 0}
            for pk in viewer_picks_list
            if pk.get("round") in (1, 2) and int(pk.get("season", 0)) <= _cur_yr + 1
        ]
        if all_picks_as_offers:
            for r in rosters:
                rid = str(r.get("roster_id") or "")
                if rid == viewer_rid:
                    continue
                targets = []
                for pos in viewer_needs:
                    top = _roster_top_players(r, pos, exclude_ids=viewer_player_ids)[:1]
                    targets.extend(p for p in top if p.get("value", 0) >= 250)
                if not targets:
                    continue
                pick_trade_partners.append({
                    "roster_id":         rid,
                    "team_name":         roster_map.get(rid) or f"Team {rid}",
                    "targets_they_have": targets[:2],
                    "picks_you_offer":   all_picks_as_offers[:2],
                })
        pick_trade_partners.sort(key=lambda x: max((t.get("value", 0) for t in x["targets_they_have"]), default=0), reverse=True)
        pick_trade_partners = pick_trade_partners[:3]

    viewer_team_ctx = build_team_gm_context(ctx, viewer_roster_id) or {}

    return {
        "viewer_team": viewer_team_ctx.get("team_name") or f"Roster {viewer_roster_id}",
        "viewer_direction": viewer_team_ctx.get("direction") or "balanced",
        "viewer_needs": viewer_needs,
        "viewer_surplus": viewer_surplus,
        "viewer_pos_ranks": {pos: viewer_ranks.get(pos, n_teams) for pos in _SCARCITY_POSITIONS},
        "league_size": n_teams,
        "viewer_pos_totals": {pos: round(v, 1) for pos, v in viewer_totals.items()},
        "league_avg_pos_totals": {pos: round(v, 1) for pos, v in league_avg.items()},
        "top_partners": partners[:5],
        "projected_picks": _projected_picks,
        "pick_trade_partners": pick_trade_partners,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Power Rankings Context
# ──────────────────────────────────────────────────────────────────────────────

def build_power_rankings_context(ctx: dict) -> dict:
    """
    Build context for AI-generated power rankings.

    For each roster, compute a PowerScore:
      0.30 * Z(PF) + 0.40 * Z(Win%) + 0.30 * Z(avg_roster_value)
    Then pass top_assets and direction to AI for narrative generation.
    """
    rosters = ctx.get("rosters") or []
    standings_map = ctx.get("standings_map") or {}
    roster_map = ctx.get("roster_map") or {}
    model_value_lookup = build_model_value_lookup(ctx.get("model_value_table") or [], is_sf=_ctx_is_sf(ctx))

    team_data = []
    for roster in rosters:
        rid = str(roster.get("roster_id") or "")
        settings = roster.get("settings") or {}
        wins = _safe_int(settings.get("wins"))
        losses = _safe_int(settings.get("losses"))
        total_games = wins + losses
        win_pct = wins / total_games if total_games > 0 else 0.0

        fpts = _safe_float(settings.get("fpts")) + _safe_float(settings.get("fpts_decimal")) / 100.0
        standing = standings_map.get(rid) or {}
        pf = _safe_float(standing.get("PF") or fpts)

        # Compute roster value
        roster_players_vals = []
        for pid in roster.get("players") or []:
            mv = model_value_lookup.get(str(pid)) or {}
            val = _safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value"))
            roster_players_vals.append(val)
        avg_value = sum(roster_players_vals) / len(roster_players_vals) if roster_players_vals else 0.0

        players_summary = summarize_roster_players(
            roster=roster,
            players_index=ctx.get("players_index") or {},
            players_map=ctx.get("players_map") or {},
            model_value_lookup=model_value_lookup,
        )
        future_picks = ctx.get("picks_by_roster", {}).get(rid, [])
        direction = detect_team_direction(players_summary, future_picks)
        pos_strength = group_position_strength(players_summary)

        ages = [_safe_float(p.get("age")) for p in players_summary if p.get("age") not in (None, "")]
        avg_age = round(sum(ages) / len(ages), 1) if ages else None
        first_round_picks = sum(1 for p in future_picks if "1." in str(p.get("display") or ""))

        team_data.append({
            "roster_id": rid,
            "team_name": roster_map.get(rid) or f"Team {rid}",
            "wins": wins,
            "losses": losses,
            "win_pct": win_pct,
            "pf": pf,
            "avg_value": round(avg_value, 1),
            "avg_age": avg_age,
            "direction": direction,
            "top_assets": players_summary[:5],
            "future_picks": future_picks[:4],
            "first_round_picks": first_round_picks,
            "position_strengths": {
                pos: {"top3": s["top_3_sum"], "best": s["best"], "count": s["count"]}
                for pos, s in pos_strength.items()
                if pos in ("QB", "RB", "WR", "TE")
            },
        })

    if not team_data:
        return {"teams": []}

    # Z-score normalization
    def _z_scores(values: list[float]) -> list[float]:
        if len(values) < 2:
            return [0.0] * len(values)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        if std == 0:
            return [0.0] * len(values)
        return [(v - mean) / std for v in values]

    pf_vals = [t["pf"] for t in team_data]
    win_vals = [t["win_pct"] for t in team_data]
    val_vals = [t["avg_value"] for t in team_data]

    pf_z = _z_scores(pf_vals)
    win_z = _z_scores(win_vals)
    val_z = _z_scores(val_vals)

    for i, team in enumerate(team_data):
        team["power_score"] = round(0.30 * pf_z[i] + 0.40 * win_z[i] + 0.30 * val_z[i], 3)

    team_data.sort(key=lambda t: t["power_score"], reverse=True)

    # Assign rank and momentum hint (prior rank not available without history; use score quartile)
    for rank, team in enumerate(team_data, start=1):
        team["rank"] = rank

    return {
        "season": ctx.get("current_season"),
        "week": ctx.get("current_week"),
        "teams": team_data,
    }
