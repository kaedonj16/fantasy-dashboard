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


def build_model_value_lookup(model_value_table: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in model_value_table or []:
        pid = str(row.get("id") or row.get("player_id") or "")
        if pid:
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
        value = _safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value"), 0.0)

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

    if elite_assets >= 3 and avg_age and avg_age <= 27.5:
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

    model_value_lookup = build_model_value_lookup(ctx.get("model_value_table") or [])
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
        bucket = slot.lower()
        bucket_label = bucket.capitalize()
        suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(_safe_int(rnd), f"{rnd}th")
        return f"{year} {bucket_label} {suffix}"

    if len(parts) == 2:
        year, rnd = parts
        suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(_safe_int(rnd), f"{rnd}th")
        return f"{year} {suffix}"

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
    (92, "A+"), (85, "A"), (78, "A-"),
    (72, "B+"), (65, "B"), (58, "B-"),
    (50, "C+"), (42, "C"), (35, "C-"),
    (0, "D"),
]

_WIN_WINDOW_LABELS = {
    ("contender", True): "Win-Now Window",
    ("contender", False): "Aging Contender",
    ("balanced", True): "Rising Contender",
    ("balanced", False): "2-3 Year Window",
    ("retool", True): "Retooling",
    ("retool", False): "Retooling",
    ("rebuild", True): "Full Rebuild",
    ("rebuild", False): "Full Rebuild",
}


def calculate_roster_grade(
    players: list[dict],
    future_picks: list[dict],
    position_ranks: dict | None = None,
    num_teams: int = 12,
) -> dict:
    """
    Score a dynasty roster across five dimensions and return a letter grade.

    Dimensions:
      Age Score (25%)  — based on avg age of top-8 players by value
      Depth Score (20%) — positions with 2+ starters above 300 value
      Capital Score (20%) — weighted pick capital (1st = 100, 2nd = 40, 3rd = 10)
      Elite Core (15%) — count players above 700 value
      Rank Score (20%) — positional rankings relative to rest of league

    Returns dict with score (0-100), grade (A+ ... D), win_window label, breakdown.
    """
    top8 = players[:8]

    # Age Score
    ages = [_safe_float(p.get("age")) for p in top8 if p.get("age") not in (None, "")]
    avg_age = sum(ages) / len(ages) if ages else 28.0
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

    # Depth Score — positions with 2+ players worth >300
    pos_counts: dict[str, int] = {}
    for p in players:
        if _safe_float(p.get("value")) >= 350:
            pos = str(p.get("position") or "?").upper()
            if pos in ("QB", "RB", "WR", "TE"):
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
    deep_positions = sum(1 for cnt in pos_counts.values() if cnt >= 2)
    depth_score = min(deep_positions * 25, 100)

    # Capital Score — future picks weighted
    capital = 0
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
        elif rnd == 2:
            capital += 40
        else:
            capital += 10
    capital_score = min(int(capital / 3), 100)

    # Elite Core Score — players above 700 (top-tier on 0-999 scale)
    elite_count = sum(1 for p in players if _safe_float(p.get("value")) >= 700)
    elite_score = min(elite_count * 33, 100)

    # Rank Score — positional rankings vs rest of league (1=best → 100, last → 0)
    if position_ranks and num_teams > 1:
        ranks = [v for v in position_ranks.values() if v is not None]
        if ranks:
            avg_rank = sum(ranks) / len(ranks)
            rank_score = round(100.0 * (1.0 - (avg_rank - 1.0) / (num_teams - 1.0)), 1)
            rank_score = max(0.0, min(100.0, rank_score))
        else:
            rank_score = 50.0
    else:
        rank_score = 50.0  # neutral when no ranking data available

    if position_ranks:
        total = (
            age_score * 0.25
            + depth_score * 0.20
            + capital_score * 0.20
            + elite_score * 0.15
            + rank_score * 0.20
        )
    else:
        total = (
            age_score * 0.30
            + depth_score * 0.25
            + capital_score * 0.25
            + elite_score * 0.20
        )
    total = round(total, 1)

    grade = "D"
    for threshold, letter in _GRADE_THRESHOLDS:
        if total >= threshold:
            grade = letter
            break

    direction = detect_team_direction(players, future_picks)
    young = avg_age <= 26.5
    win_window = _WIN_WINDOW_LABELS.get((direction, young), "Balanced")

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
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Roster Depth Warning (replaces league-wide positional scarcity)
# ──────────────────────────────────────────────────────────────────────────────

# Minimum value to count as "starter-caliber" at each position (0-999 scale)
_STARTER_THRESHOLD = {"QB": 500, "RB": 350, "WR": 350, "TE": 400}

# How many starter-caliber players you need to feel safe at each position
_DEPTH_FLOOR = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}


def calculate_roster_depth_warning(
        viewer_roster: dict,
        model_value_lookup: dict[str, dict],
        sending_assets: list[dict],
        receiving_assets: list[dict],
) -> dict[str, dict]:
    """
    Warn when a trade leaves the viewer dangerously thin at a position.

    Simulates the post-trade roster: removes sent players, adds received players,
    then counts starter-caliber players at each affected position.

    Returns {pos: {before, after, warning, severity}} only for positions
    where the trade changes depth or triggers a warning.
    severity: 'danger' | 'caution' | None
    """
    # Build current roster value map
    roster_values: dict[str, dict] = {}
    for pid in viewer_roster.get("players") or []:
        spid = str(pid)
        mv = model_value_lookup.get(spid) or {}
        pos = str(mv.get("position") or "").upper()
        val = _safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value"))
        if pos in _STARTER_THRESHOLD and val > 0:
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

        if after == before and after > floor:
            continue  # No meaningful change, skip

        warning = None
        severity = None

        if after == 0:
            warning = f"You'll have no starter-caliber {pos} after this trade"
            severity = "danger"
        elif after < floor:
            warning = f"Leaves you with only {after} starter-caliber {pos} (need {floor})"
            severity = "danger" if after == 0 else "caution"
        elif after < before:
            warning = f"{pos} depth drops from {before} to {after} starters"
            severity = "caution"

        result[pos] = {
            "before": before,
            "after": after,
            "warning": warning,
            "severity": severity,
        }

    return result


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

    model_value_lookup = build_model_value_lookup(ctx.get("model_value_table") or [])
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

    # Project the viewer's upcoming picks to actual rookie positions/values using
    # the live rookie rankings, so needs detection reflects what they're likely to draft.
    # e.g. if they hold 1.01 and the top rookie is an RB, suppress RB as a need.
    viewer_rid = str(viewer_roster_id)
    viewer_picks_list = picks_by_roster.get(viewer_rid, [])
    from datetime import datetime as _dt
    _cur_yr = _dt.now().year

    top_rookies = sorted(
        [v for v in model_value_lookup.values()
         if v.get("is_rookie") and str(v.get("position", "")).upper() in _SCARCITY_POSITIONS],
        key=lambda r: float(r.get("value") or 0),
        reverse=True,
    )
    _rookie_idx = 0
    _pick_pos_credits: dict[str, float] = {}
    _projected_picks: list[dict] = []  # for context output
    for _rnd in [1, 2]:
        _rnd_picks = sorted(
            [p for p in viewer_picks_list
             if p.get("round") == _rnd and int(p.get("season", 0)) <= _cur_yr + 1],
            key=lambda p: p.get("season", 9999),
        )
        for _pk in _rnd_picks:
            if _rookie_idx < len(top_rookies):
                _proj = top_rookies[_rookie_idx]
                _pos  = str(_proj.get("position", "")).upper()
                _val  = float(_proj.get("value") or 0)
                if _pos:
                    _pick_pos_credits[_pos] = _pick_pos_credits.get(_pos, 0.0) + _val
                _projected_picks.append({
                    "season":    _pk.get("season"),
                    "round":     _rnd,
                    "proj_name": _proj.get("name", ""),
                    "proj_pos":  _pos,
                    "proj_val":  round(_val, 1),
                })
                _rookie_idx += 1
            else:
                _projected_picks.append({"season": _pk.get("season"), "round": _rnd})

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

        # Find what viewer could send (viewer's surplus that partner needs).
        # Rules:
        #  - Always keep at least 2 at each position (starter + backup).
        #  - If keeping 2 would leave nothing sendable, allow 1 only if the partner
        #    is sending that same position back (positional balance).
        getting_positions = {str(p.get("position", "")).upper() for p in targets_they_have}
        targets_viewer_sends = []
        for pos in viewer_surplus:
            if pos not in partner_needs:
                continue
            pos_depth = sum(
                1 for pid in (roster.get("players") or [])
                if str(model_value_lookup.get(str(pid), {}).get("position", "")).upper() == pos
            )
            max_sendable = max(0, pos_depth - 2)  # keep at least 2
            if max_sendable == 0:
                if pos in getting_positions:
                    max_sendable = 1  # can spare 1 when getting that position back
                else:
                    continue  # don't strip down to 0 at a position
            top = _roster_top_players(roster, pos)[:min(2, max_sendable)]
            targets_viewer_sends.extend(top)

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
        })

    partners.sort(key=lambda x: x["match_score"], reverse=True)

    # Pick-for-player suggestions: viewer offers a pick instead of a player.
    # Don't require partner_surplus — any team with a good player at a needed
    # position could be a pick trade target.
    pick_trade_partners = []
    if viewer_needs:
        # Include all upcoming picks as potential offers (with or without rookie projection)
        all_picks_as_offers = [
            p for p in _projected_picks
            if p.get("round") in (1, 2)
        ] or [
            {"season": pk.get("season"), "round": pk.get("round"), "proj_name": "", "proj_pos": "", "proj_val": 0}
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
    model_value_lookup = build_model_value_lookup(ctx.get("model_value_table") or [])

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

        team_data.append({
            "roster_id": rid,
            "team_name": roster_map.get(rid) or f"Team {rid}",
            "wins": wins,
            "losses": losses,
            "win_pct": win_pct,
            "pf": pf,
            "avg_value": round(avg_value, 1),
            "direction": direction,
            "top_assets": players_summary[:5],
            "future_picks": future_picks[:4],
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
