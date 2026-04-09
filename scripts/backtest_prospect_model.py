"""
Backtest the rookie prospect model against 5 years of actual NFL outcomes (2021-2025).

For each draft class, we reconstruct what the model would have scored each player
using the ACTUAL draft pick as draft capital (the closest historical proxy for
pre-draft consensus). College stats from CFBD and combine athleticism are fetched
where available.

Output: ranked table showing model rank, player, position, model score,
        actual NFL PPR fantasy points (Y1, Y2, cumulative), and delta between
        model rank and actual performance rank.

Usage:
    cd /home/user/fantasy-dashboard
    python -m scripts.backtest_prospect_model
  or:
    python scripts/backtest_prospect_model.py
"""
from __future__ import annotations

import math
import statistics
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

from pandas import read_csv

from utils.utils import read_json

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline.historical_calibration import (
    _fetch_csv,
    _NFLVERSE_ROSTER,
    _NFLVERSE_BASE,
    _NFLVERSE_COMBINE,
    _safe_float,
    _calc_ppr_points,
)
from data_building.rookie_pipeline.prospect_model import score_all_prospects
from data_building.rookie_pipeline.mock_draft_consensus import (
    build_mock_draft_consensus,
    pick_to_draft_capital_score,
)
from data_building.rookie_pipeline.ingestion import fetch_cfbd_college_stats

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DRAFT_YEARS  = [2021, 2022, 2023, 2024, 2025]
SKILL_POS    = {"QB", "RB", "WR", "TE"}
NFL_LOOKBACK = 4   # seasons of NFL data to collect per player

# How many top-N players per draft class to show in the table
TOP_N_PER_CLASS = 20

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Load nflverse roster data to get draft picks
# ─────────────────────────────────────────────────────────────────────────────

def _load_draft_class(draft_year: int) -> List[Dict[str, Any]]:
    """
    Return skill-position players drafted in `draft_year`.

    Uses nflverse roster CSV.  Deduplicates by gsis_id, keeps best row
    (non-empty draft_number).

    Returns list of dicts:
        name, position, gsis_id, draft_pick, college, birth_date
    """
    rows = _fetch_csv(_NFLVERSE_ROSTER.format(year=draft_year))
    seen: Dict[str, Dict] = {}  # gsis_id → best row

    for row in rows:
        if row.get("rookie_year") != str(draft_year):
            continue
        pos = (row.get("position") or "").upper()
        if pos not in SKILL_POS:
            continue
        gid = row.get("gsis_id") or ""
        pick_raw = row.get("draft_number") or ""
        pick = int(float(pick_raw)) if pick_raw.strip() else 0

        if gid not in seen or (pick and not seen[gid].get("draft_pick")):
            seen[gid] = {
                "name":       (row.get("full_name") or row.get("football_name") or "").strip(),
                "position":   pos,
                "gsis_id":    gid,
                "draft_pick": pick,
                "college":    (row.get("college") or "").strip(),
                "birth_date": (row.get("birth_date") or "").strip(),
            }

    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Load combine athleticism
# ─────────────────────────────────────────────────────────────────────────────

def _load_combine_athleticism(draft_year: int) -> Dict[str, Dict[str, Any]]:
    """
    Return {name_lower: athleticism_dict} from nflverse combine.csv.
    Only includes players from the given draft year.
    """
    rows = _fetch_csv(_NFLVERSE_COMBINE)
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("draft_year") != str(draft_year):
            continue
        pos = (row.get("pos") or row.get("position") or "").upper()
        if pos not in SKILL_POS:
            continue
        name = (row.get("player_name") or row.get("name") or "").strip().lower()
        if not name:
            continue

        def _f(k: str) -> Optional[float]:
            v = row.get(k, "")
            try:
                return float(v) if v not in ("", "NA", "NULL", None) else None
            except (TypeError, ValueError):
                return None

        # Keys must match what calc_athleticism_score() reads:
        #   forty_yard, vertical_inches, broad_jump_in, three_cone, short_shuttle,
        #   ras_score, weight_lbs
        # nflverse combine CSV uses: forty, vertical, broad_jump, cone, shuttle
        result[name] = {
            "forty_yard":     _f("forty_yard") or _f("forty"),
            "vertical_inches":_f("vertical"),          # was wrongly keyed as "vertical"
            "broad_jump_in":  _f("broad_jump"),        # was wrongly keyed as "broad_jump"
            "bench_reps":     _f("bench_reps") or _f("bench"),
            "three_cone":     _f("three_cone") or _f("cone"),   # CSV column is "cone"
            "short_shuttle":  _f("shuttle"),            # was wrongly keyed as "shuttle"
            "ras_score":      _f("ras_score"),
            "height_inches":  _f("ht") or _f("height"),
            "weight_lbs":     _f("wt") or _f("weight"),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Build actual NFL PPR outcomes per player
# ─────────────────────────────────────────────────────────────────────────────

def _build_nfl_ppr_per_player(
    gsis_ids: List[str],
    draft_year: int,
    nfl_data_years: int = NFL_LOOKBACK,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-player aggregate PPR fantasy points for up to `nfl_data_years`
    seasons starting from `draft_year`.

    Returns: {gsis_id: {"ppr_y1": float, "ppr_y2": float, ..., "ppr_peak": float,
                         "ppr_cum": float, "seasons_available": int}}
    """
    gid_set = set(gsis_ids)

    # gsis_id → list of seasonal PPR totals (index 0 = year 1, etc.)
    season_pts: Dict[str, List[float]] = {gid: [] for gid in gid_set}

    for offset in range(nfl_data_years):
        nfl_yr = draft_year + offset
        if nfl_yr == 2025:
            read_csv("cache/stats_player_reg_2025.csv")
        else:
            stat_rows = _fetch_csv(_NFLVERSE_BASE.format(year=nfl_yr))

        # Accumulate PPR per gsis_id for this season (stats are weekly)
        yr_pts: Dict[str, float] = {}
        for sr in stat_rows:
            gid = sr.get("player_id") or ""
            if gid not in gid_set:
                continue
            if sr.get("season_type", "REG").upper() != "REG":
                continue
            yr_pts[gid] = yr_pts.get(gid, 0.0) + _calc_ppr_points(sr)

        for gid in gid_set:
            season_pts[gid].append(yr_pts.get(gid, 0.0))

    result: Dict[str, Dict[str, Any]] = {}
    for gid, pts in season_pts.items():
        if not pts:
            continue
        available = sum(1 for x in pts if x > 0)
        result[gid] = {
            "ppr_y1":             pts[0] if len(pts) >= 1 else 0.0,
            "ppr_y2":             pts[1] if len(pts) >= 2 else 0.0,
            "ppr_y3":             pts[2] if len(pts) >= 3 else 0.0,
            "ppr_y4":             pts[3] if len(pts) >= 4 else 0.0,
            "ppr_peak":           max(pts) if pts else 0.0,
            "ppr_cum":            sum(pts),
            "seasons_available":  available,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Build prospect dicts for the scorer
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Load CFBD college stats
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """
    Lowercase and strip suffixes/punctuation for fuzzy name matching between
    nflverse (full_name) and CFBD (player) name fields.

    Examples:
      "Travis Etienne Jr." → "travis etienne"
      "Wan'Dale Robinson"  → "wandale robinson"
      "Bijan Robinson"     → "bijan robinson"
    """
    import re
    name = name.lower().strip()
    # Remove generational suffixes
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\s*$", "", name).strip()
    # Remove punctuation that differs between sources (apostrophes, hyphens, periods)
    name = re.sub(r"['\-\.]", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _load_cfbd_college_stats(
    draft_year: int,
    draft_class: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch CFBD college stats for `draft_year` and match by name to the nflverse
    draft class.  Returns {player_id: [season_dict, ...]} using the prospect's
    player_id (gsis_id or name-slug) as key.

    Gracefully returns {} if CFBD_API_KEY is not set or the API is unreachable.
    """
    print(f"[backtest]   Fetching CFBD college stats for {draft_year} class…")
    cfbd_raw = fetch_cfbd_college_stats(draft_year)
    if not cfbd_raw:
        print("[backtest]   CFBD returned no data (key missing or API unreachable)")
        return {}

    print(f"[backtest]   CFBD returned stats for {len(cfbd_raw)} players")

    # Build a normalised-name → player_id map from the draft class
    pid_by_norm: Dict[str, str] = {}
    for p in draft_class:
        pid  = p["gsis_id"] or p["name"].lower().replace(" ", "-")
        norm = _normalize_name(p["name"])
        pid_by_norm[norm] = pid

    # Match CFBD names → player_ids
    result: Dict[str, List[Dict]] = {}
    matched = 0
    for cfbd_name, seasons in cfbd_raw.items():
        norm = _normalize_name(cfbd_name)
        pid  = pid_by_norm.get(norm)
        if pid:
            result[pid] = seasons
            matched += 1

    print(f"[backtest]   Matched {matched}/{len(cfbd_raw)} CFBD players to draft class "
          f"({matched}/{len(draft_class)} draftees have college stats)")
    return result


def _age_at_draft(birth_date_str: str, draft_year: int) -> Optional[float]:
    """Calculate age on draft day (≈ April 25)."""
    if not birth_date_str:
        return None
    try:
        from datetime import date
        parts = birth_date_str.split("-")
        if len(parts) != 3:
            return None
        bdate = date(int(parts[0]), int(parts[1]), int(parts[2]))
        ddate = date(draft_year, 4, 25)
        return round((ddate - bdate).days / 365.25, 2)
    except (ValueError, OverflowError):
        return None


def _build_prospect_dicts(
    draft_class: List[Dict[str, Any]],
    athleticism: Dict[str, Dict[str, Any]],
    draft_year: int,
    college_stats: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Convert raw draft-class entries into the format expected by score_all_prospects.
    Also returns a consensus_map with actual draft picks as projected_pick.

    Args:
        college_stats: {player_id: [season_dict, ...]} from _load_cfbd_college_stats.
                       When provided, activates 52% of the model that otherwise
                       defaults to neutral (production, breakout, competition, etc.).

    Returns: (prospects, consensus_map)
    """
    if college_stats is None:
        college_stats = {}

    prospects: List[Dict[str, Any]] = []
    consensus_map: Dict[str, Dict[str, Any]] = {}

    for p in draft_class:
        player_id = p["gsis_id"] or p["name"].lower().replace(" ", "-")
        name_lower = p["name"].lower().strip()
        ath = athleticism.get(name_lower, {})

        # Height / weight from combine if available, else defaults
        height = ath.pop("height_inches", None)
        weight = ath.pop("weight_lbs", None)

        age = _age_at_draft(p["birth_date"], draft_year)

        # College seasons from CFBD (already in the format score_all_prospects expects)
        seasons = college_stats.get(player_id, [])

        # Extract conference from most recent CFBD season (improves competition_score)
        conference = None
        if seasons:
            latest = max(seasons, key=lambda s: s.get("season", 0))
            conference = latest.get("conference")

        prospect = {
            "player_id":        player_id,
            "name":             p["name"],
            "position":         p["position"],
            "school":           p["college"],
            "conference":       conference,
            "draft_class_year": draft_year,
            "age":              age,
            "height_inches":    height,
            "weight_lbs":       weight,
            "early_declare":    False,
            "transfer_history": False,
            "seasons":          seasons,
            "athleticism":      ath,
        }
        prospects.append(prospect)

        # Build consensus entry from actual draft pick (position-adjusted score)
        pick     = p["draft_pick"] or 300
        position = p.get("position", "WR").upper()
        consensus_map[player_id] = {
            "player_id":                    player_id,
            "projected_pick":               pick,
            "projected_round":              ((pick - 1) // 32) + 1,
            "projected_pick_low":           max(1, pick - 5),
            "projected_pick_high":          pick + 5,
            "projected_draft_capital_score": pick_to_draft_capital_score(pick, position),
            "num_mocks_used":               1,
            "consensus_confidence":         100.0,  # actual pick = certainty
            "is_actual_pick":               True,
        }

    return prospects, consensus_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Run one draft class
# ─────────────────────────────────────────────────────────────────────────────

def _run_draft_class_backtest(
    draft_year: int,
) -> List[Dict[str, Any]]:
    """
    Score a historical draft class and attach actual NFL outcomes.
    Returns list of result rows, sorted by model overall_rank.
    """
    print(f"\n[backtest] ──── {draft_year} Draft Class ────")

    print(f"[backtest]   Loading draft picks from nflverse…")
    draft_class = _load_draft_class(draft_year)
    print(f"[backtest]   {len(draft_class)} skill-position draftees found")
    if not draft_class:
        return []

    print(f"[backtest]   Loading combine athleticism…")
    athleticism = _load_combine_athleticism(draft_year)
    print(f"[backtest]   Combine data for {len(athleticism)} players")

    college_stats = _load_cfbd_college_stats(draft_year, draft_class)

    prospects, consensus_map = _build_prospect_dicts(
        draft_class, athleticism, draft_year, college_stats
    )

    n_with_stats = sum(1 for p in prospects if p.get("seasons"))
    print(f"[backtest]   {n_with_stats}/{len(prospects)} prospects have college stats "
          f"({'full model' if n_with_stats > 0 else 'draft capital + athleticism only'})")

    print(f"[backtest]   Scoring {len(prospects)} prospects…")
    scores = score_all_prospects(prospects, consensus_map)

    # Build lookup maps
    p_by_id  = {p["player_id"]: p for p in prospects}
    dc_entry = {p["gsis_id"] or p["name"].lower().replace(" ", "-"): p for p in draft_class}

    gsis_ids = [p["gsis_id"] for p in draft_class if p["gsis_id"]]
    print(f"[backtest]   Fetching NFL stats for up to {NFL_LOOKBACK} seasons…")
    nfl_ppr = _build_nfl_ppr_per_player(gsis_ids, draft_year)
    print(f"[backtest]   NFL PPR data for {len(nfl_ppr)} players")

    rows = []
    for sc in scores:
        pid  = sc["player_id"]
        p    = p_by_id.get(pid, {})
        dc   = dc_entry.get(pid, {})
        gid  = dc.get("gsis_id", "")
        ppr  = nfl_ppr.get(gid, {})

        rows.append({
            "draft_year":       draft_year,
            "model_rank":       sc["overall_rank"],
            "pos_rank":         sc["position_rank"],
            "name":             p.get("name", pid),
            "position":         p.get("position", ""),
            "college":          p.get("school", ""),
            "draft_pick":       dc.get("draft_pick", 0),
            "model_score":      sc["prospect_score"],
            "dc_score":         sc["projected_draft_capital_score"],
            "ath_score":        sc["athleticism_score"],
            "prod_score":       sc["production_score"],
            "age_score":        sc["age_score"],
            "ppr_y1":           ppr.get("ppr_y1", 0.0),
            "ppr_y2":           ppr.get("ppr_y2", 0.0),
            "ppr_y3":           ppr.get("ppr_y3", 0.0),
            "ppr_y4":           ppr.get("ppr_y4", 0.0),
            "ppr_peak":         ppr.get("ppr_peak", 0.0),
            "ppr_cum":          ppr.get("ppr_cum", 0.0),
            "seasons_avail":    ppr.get("seasons_available", 0),
            "has_cfbd":         bool(p_by_id.get(pid, {}).get("seasons")),
            "breakout_score":   sc["breakout_profile_score"],
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Summary stats
# ─────────────────────────────────────────────────────────────────────────────

def _pearson_r(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx  = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy  = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy * n)


def _rank_corr(rows: List[Dict], metric: str = "ppr_cum") -> float:
    """Spearman rank correlation between model_rank and actual PPR rank."""
    valid = [r for r in rows if r.get(metric, 0) > 0]
    if len(valid) < 5:
        return float("nan")
    model_ranks = [r["model_rank"] for r in valid]
    ppr_sorted  = sorted(valid, key=lambda x: x[metric], reverse=True)
    ppr_rank    = {r["name"]: i + 1 for i, r in enumerate(ppr_sorted)}
    actual_ranks = [ppr_rank[r["name"]] for r in valid]
    return _pearson_r(model_ranks, actual_ranks)


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(val: float, mx: float = 400.0, width: int = 20) -> str:
    filled = int(round(val / mx * width)) if mx > 0 else 0
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _print_class_table(
    rows: List[Dict],
    draft_year: int,
    seasons_note: str,
    top_n: int = TOP_N_PER_CLASS,
) -> None:
    print(f"\n{'=' * 110}")
    print(f"  {draft_year} DRAFT CLASS — Top {top_n} by Model Score  ({seasons_note})")
    print(f"{'=' * 110}")
    header = (
        f"{'Rnk':>3}  {'Player':<25} {'Pos':>3}  {'Pick':>4}  "
        f"{'Score':>5}  {'PPR-Y1':>6}  {'PPR-Y2':>6}  {'PPR-Y3':>6}  "
        f"{'Cum PPR':>7}  NFL Bar"
    )
    print(header)
    print("-" * 110)

    for r in rows[:top_n]:
        # Annotate players with no NFL data
        nfl_note = " (no NFL data)" if r["seasons_avail"] == 0 and r["draft_pick"] else ""
        # Current year players may have limited data
        ppr_y3_str = f"{r['ppr_y3']:>6.0f}" if r.get("ppr_y3", 0) > 0 else "     -"
        print(
            f"{r['model_rank']:>3}.  "
            f"{r['name']:<25} {r['position']:>3}  "
            f"#{r['draft_pick']:>3}  "
            f"{r['model_score']:>5.1f}  "
            f"{r['ppr_y1']:>6.0f}  "
            f"{r['ppr_y2']:>6.0f}  "
            f"{ppr_y3_str}  "
            f"{r['ppr_cum']:>7.0f}  "
            f"{_bar(r['ppr_cum'])}{nfl_note}"
        )

    # Bottom of table: top-10 actual performers vs model rank (only when data exists)
    has_actual = [r for r in rows if r["ppr_cum"] > 0]
    if has_actual:
        print(f"\n  ── Actual Top-10 PPR performers in {draft_year} class (ranked by cum PPR) ──")
        ranked_by_ppr = sorted(has_actual, key=lambda x: x["ppr_cum"], reverse=True)
        for i, r in enumerate(ranked_by_ppr[:10], 1):
            delta = r["model_rank"] - i
            arrow = "↑" if delta > 3 else ("↓" if delta < -3 else "≈")
            delta_str = f"{arrow}{abs(delta):>3}" if delta != 0 else "  ="
            print(
                f"  PPR#{i:>2}  {r['name']:<25} {r['position']:>3}  "
                f"Model#{r['model_rank']:>3}  ({delta_str})  "
                f"cum={r['ppr_cum']:>7.0f}  Y1={r['ppr_y1']:>6.0f}  Y2={r['ppr_y2']:>6.0f}"
            )
    else:
        print(f"\n  (No NFL data available yet for {draft_year} class)")


def _print_summary(all_rows: List[Dict]) -> None:
    print(f"\n{'=' * 110}")
    print("  OVERALL BACKTEST SUMMARY  (2021–2025 draft classes)")
    print("  Note: ranks below are cross-year (all classes pooled); per-class ranks are in tables above")
    print(f"{'=' * 110}")

    # Group by draft year
    by_year: Dict[int, List[Dict]] = {}
    for r in all_rows:
        by_year.setdefault(r["draft_year"], []).append(r)

    overall_valid: List[Tuple[float, float]] = []  # (model_score, ppr_cum)
    overall_valid_cfbd: List[Tuple[float, float]] = []  # only rows with CFBD data
    print(f"\n  {'Year':>4}  {'Players':>7}  {'w/NFL data':>10}  {'CFBD hit%':>9}  {'Spearman-ρ':>10}  {'Top5 hit':>8}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*8}")

    for yr in sorted(by_year.keys()):
        rows = by_year[yr]
        with_data = [r for r in rows if r["ppr_cum"] > 0]
        rc = _rank_corr(rows, "ppr_cum")
        rc_str = f"{rc:+.3f}" if not math.isnan(rc) else "  n/a"

        cfbd_n = sum(1 for r in rows if r.get("has_cfbd"))
        cfbd_pct = f"{cfbd_n/len(rows)*100:.0f}%" if rows else "n/a"

        # Top-5 hit rate: how many of model's top 5 are in actual top 10?
        top5_model   = {r["name"] for r in rows[:5]}
        top10_actual = {r["name"] for r in sorted(with_data, key=lambda x: x["ppr_cum"], reverse=True)[:10]}
        hit_n = len(top5_model & top10_actual)
        hit_str = f"{hit_n}/5" if with_data else "n/a"

        print(f"  {yr:>4}  {len(rows):>7}  {len(with_data):>10}  {cfbd_pct:>9}  {rc_str:>10}  {hit_str:>8}")

        for r in with_data:
            overall_valid.append((r["model_score"], r["ppr_cum"]))
            if r.get("has_cfbd"):
                overall_valid_cfbd.append((r["model_score"], r["ppr_cum"]))

    # ── Pearson r by data-completeness tier ────────────────────────────────────
    # 2024 has only 1 NFL season; 2023 has 2.  Including them pools clean signal
    # with noisy early data, deflating the overall r.  Show r at each tier so
    # the model's real predictive quality is visible.
    print()
    tiers = [
        ("≥3 complete seasons (2021–2022)", lambda r: r["draft_year"] <= 2022 and r["ppr_cum"] > 0),
        ("≥2 complete seasons (2021–2023)", lambda r: r["draft_year"] <= 2023 and r["ppr_cum"] > 0),
        ("all years incl. partial (2021–2024)", lambda r: r["ppr_cum"] > 0),
    ]
    print(f"  {'Data tier':<42}  {'n':>4}  {'Pearson-r':>9}")
    print(f"  {'-'*42}  {'-'*4}  {'-'*9}")
    for label, pred in tiers:
        subset = [r for r in all_rows if pred(r)]
        if len(subset) < 10:
            continue
        xs = [r["model_score"] for r in subset]
        ys = [r["ppr_cum"] for r in subset]
        rv = _pearson_r(xs, ys)
        flag = "✓" if rv > 0.45 else ("~" if rv > 0.30 else "✗")
        print(f"  {label:<42}  {len(subset):>4}  {rv:>+9.3f}  {flag}")

    # Show CFBD lift
    if len(overall_valid_cfbd) >= 5:
        xs_c = [x for x, _ in overall_valid_cfbd]
        ys_c = [y for _, y in overall_valid_cfbd]
        r_cfbd = _pearson_r(xs_c, ys_c)
        rest_pairs = [(x, y) for (x, y) in overall_valid
                      if (x, y) not in set(overall_valid_cfbd)]
        r_no_cfbd = (
            _pearson_r([x for x, _ in rest_pairs], [y for _, y in rest_pairs])
            if len(rest_pairs) >= 5 else float("nan")
        )
        print(f"\n  CFBD college stats coverage:")
        print(f"    WITH stats  (n={len(overall_valid_cfbd)}): r={r_cfbd:+.3f}")
        if not math.isnan(r_no_cfbd):
            print(f"    WITHOUT stats (n={len(rest_pairs)}): r={r_no_cfbd:+.3f}  lift={r_cfbd - r_no_cfbd:+.3f}")

    # ── Per-position Pearson r (skill positions only) ──────────────────────────
    # Cross-position mixing distorts r: QBs score high PPR but model discounts
    # them via POSITION_FANTASY_MULT=0.90 and dc_multiplier=0.65. Separating
    # by position shows the true within-group predictive accuracy.
    print(f"\n  Per-position Pearson r (model score vs cum PPR):")
    print(f"  {'Pos':>4}  {'n':>4}  {'Pearson-r':>9}  signal")
    print(f"  {'-'*4}  {'-'*4}  {'-'*9}  {'-'*30}")
    for pos in ("WR", "RB", "TE", "QB"):
        pos_rows = [r for r in all_rows if r["position"] == pos and r["ppr_cum"] > 0]
        if len(pos_rows) < 5:
            continue
        px = [r["model_score"] for r in pos_rows]
        py = [r["ppr_cum"] for r in pos_rows]
        r_pos = _pearson_r(px, py)
        signal = (
            "strong" if r_pos > 0.45 else
            "moderate" if r_pos > 0.28 else
            "weak" if r_pos > 0.10 else
            "noise"
        )
        print(f"  {pos:>4}  {len(pos_rows):>4}  {r_pos:>+9.3f}  {signal}")

    print()

    # Best and worst model calls across all years
    has_ppr = [r for r in all_rows if r["ppr_cum"] > 50]
    if has_ppr:
        # Use PPR per season (avg) for cross-year comparisons.
        # Cumulative unfairly favours older classes that simply have more seasons
        # of data — a 2021 player at 250/season looks worse than a 2021 player at
        # 200/season with 4 years banked.  Per-season average is position-neutral
        # and class-neutral.
        for r in has_ppr:
            r["ppr_avg"] = r["ppr_cum"] / max(r.get("seasons_avail", 1), 1)

        actual_sorted   = sorted(has_ppr, key=lambda x: x["ppr_avg"], reverse=True)
        model_sorted    = sorted(has_ppr, key=lambda x: x["model_score"], reverse=True)
        model_rank_map  = {r["name"]: i + 1 for i, r in enumerate(model_sorted)}
        actual_rank_map = {r["name"]: i + 1 for i, r in enumerate(actual_sorted)}

        # Best calls: low model_rank AND low actual_rank (both close to 1 = both good)
        hits = sorted(
            has_ppr,
            key=lambda r: actual_rank_map.get(r["name"], 999) + model_rank_map.get(r["name"], 999),
        )[:10]
        print("  ── Best model calls (model ranked high AND performed well) ──")
        for r in hits[:10]:
            mr = model_rank_map.get(r["name"], 999)
            ar = actual_rank_map.get(r["name"], 999)
            print(
                f"    {r['draft_year']}  {r['name']:<25} {r['position']:>3}  "
                f"Model#{mr:>3}  Actual#{ar:>3}  "
                f"avg={r['ppr_avg']:>5.0f}/season  peak={r['ppr_peak']:>5.0f}  (n={r.get('seasons_avail', 1)})"
            )

        # Biggest overrates: model ranked HIGH but actual averaged poorly
        overrates = sorted(
            has_ppr,
            key=lambda r: actual_rank_map.get(r["name"], 999) - model_rank_map.get(r["name"], 999),
            reverse=True,
        )[:5]
        print("\n  ── Model's biggest overrates (model ranked high, actual averaged poorly) ──")
        for r in overrates[:5]:
            mr = model_rank_map.get(r["name"], 999)
            ar = actual_rank_map.get(r["name"], 999)
            delta = ar - mr
            print(
                f"    {r['draft_year']}  {r['name']:<25} {r['position']:>3}  "
                f"Model#{mr:>3}  Actual#{ar:>3}  (fell {delta:+d})  "
                f"avg={r['ppr_avg']:>5.0f}/season  peak={r['ppr_peak']:>5.0f}  (n={r.get('seasons_avail', 1)})"
            )

        # Biggest underrates: model ranked LOW but actual averaged well
        underrates = sorted(
            has_ppr,
            key=lambda r: model_rank_map.get(r["name"], 999) - actual_rank_map.get(r["name"], 999),
            reverse=True,
        )[:5]
        print("\n  ── Model's biggest underrates (model ranked low, actual averaged well) ──")
        for r in underrates[:5]:
            mr = model_rank_map.get(r["name"], 999)
            ar = actual_rank_map.get(r["name"], 999)
            delta = mr - ar
            print(
                f"    {r['draft_year']}  {r['name']:<25} {r['position']:>3}  "
                f"Model#{mr:>3}  Actual#{ar:>3}  (rose {delta:+d})  "
                f"avg={r['ppr_avg']:>5.0f}/season  peak={r['ppr_peak']:>5.0f}  (n={r.get('seasons_avail', 1)})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(draft_years: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Run the full backtest. Returns all result rows (one per draftee).
    """
    if draft_years is None:
        draft_years = DRAFT_YEARS

    from datetime import date
    current_year = date.today().year

    all_rows: List[Dict[str, Any]] = []

    for dy in draft_years:
        # How many complete NFL seasons are available?
        completed_seasons = max(0, current_year - dy - 1)
        seasons_note = (
            f"{completed_seasons} complete NFL seasons available"
            if completed_seasons > 0
            else "No complete NFL season data yet (current class)"
        )
        rows = _run_draft_class_backtest(dy)
        if rows:
            all_rows.extend(rows)
            _print_class_table(rows, dy, seasons_note)

    if all_rows:
        _print_summary(all_rows)

    return all_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest rookie prospect model (2021-2025)")
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=DRAFT_YEARS,
        help="Draft years to include (default: 2021 2022 2023 2024 2025)",
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N_PER_CLASS,
        help=f"Rows per draft class (default: {TOP_N_PER_CLASS})",
    )
    args = parser.parse_args()
    TOP_N_PER_CLASS = args.top_n
    run_backtest(args.years)
