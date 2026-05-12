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
import os
import statistics
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

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

DRAFT_YEARS  = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
SKILL_POS    = {"QB", "RB", "WR", "TE"}
NFL_LOOKBACK = 4   # seasons of NFL data to collect per player

# How many top-N players per draft class to show in the table
TOP_N_PER_CLASS = 10

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
    cfbd_raw = fetch_cfbd_college_stats(draft_year, skip_sagarin=True)
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
    scores = score_all_prospects(prospects, consensus_map, skip_sagarin=True)

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

        # Extract college benchmark features from the latest available season
        latest_s: Dict[str, Any] = {}
        if p.get("seasons"):
            latest_s = max(p["seasons"], key=lambda s: s.get("season", 0))

        def _sf(k: str) -> Optional[float]:
            v = latest_s.get(k)
            try:
                f = float(v) if v not in (None, "", "NA") else None
                return f if f else None
            except (TypeError, ValueError):
                return None

        gp           = max(float(latest_s.get("games_played") or 12), 1.0)
        rec_yds      = float(latest_s.get("receiving_yards") or 0)
        rec_tds      = float(latest_s.get("receiving_tds")   or 0)
        rush_yds     = float(latest_s.get("rush_yards")      or 0)
        rush_tds     = float(latest_s.get("rush_tds")        or 0)
        pass_yds     = float(latest_s.get("pass_yards")      or 0)
        team_pass_yds = float(latest_s.get("team_pass_yards") or 0)

        col_pass_share = (rec_yds / team_pass_yds) if team_pass_yds > 0 else None

        row = {
            "draft_year":         draft_year,
            "model_rank":         sc["overall_rank"],
            "pos_rank":           sc["position_rank"],
            "name":               p.get("name", pid),
            "position":           p.get("position", ""),
            "college":            p.get("school", ""),
            "draft_pick":         dc.get("draft_pick", 0),
            "model_score":        sc["prospect_score"],
            "dc_score":           sc["projected_draft_capital_score"],
            "ath_score":          sc["athleticism_score"],
            "prod_score":         sc["production_score"],
            "age_score":          sc["age_score"],
            "ppr_y1":             ppr.get("ppr_y1", 0.0),
            "ppr_y2":             ppr.get("ppr_y2", 0.0),
            "ppr_y3":             ppr.get("ppr_y3", 0.0),
            "ppr_y4":             ppr.get("ppr_y4", 0.0),
            "ppr_peak":           ppr.get("ppr_peak", 0.0),
            "ppr_cum":            ppr.get("ppr_cum", 0.0),
            "seasons_avail":      ppr.get("seasons_available", 0),
            "has_cfbd":           bool(p.get("seasons")),
            "breakout_score":     sc["breakout_profile_score"],
            # College benchmark features (None = data unavailable)
            "col_rec_yds_pg":     rec_yds / gp if latest_s else None,
            "col_rec_yds_season": rec_yds      if latest_s else None,
            "col_rec_tds_pg":     rec_tds / gp if latest_s else None,
            "col_rush_yds_pg":    rush_yds / gp if latest_s else None,
            "col_pass_yds_pg":    pass_yds / gp if latest_s else None,
            "col_tds_pg":         (rec_tds + rush_tds) / gp if latest_s else None,
            "col_dominator":      _sf("dominator_rating"),
            "col_pass_share":     col_pass_share,
            "col_yac_per_rec":    _sf("yards_after_catch_per_reception"),
            "col_ypc":            _sf("yds_per_carry"),
            "col_completion_pct": _sf("completion_pct"),
            "col_ypa":            _sf("yds_per_attempt"),
            "col_td_int":         _sf("td_int_ratio"),
            "col_age_at_draft":   p.get("age"),
            # Full dicts attached for ML training feature extraction
            "_prospect":          p,
            "_consensus":         consensus_map.get(pid, {}),
        }
        rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Summary stats
# ─────────────────────────────────────────────────────────────────────────────

# PPR-peak threshold that approximates a "hit" season per position
_HIT_PPR_THRESHOLD: Dict[str, float] = {
    "QB": 310.0,   # ≈ top-6 QB season
    "WR": 220.0,   # ≈ top-12 WR season
    "RB": 240.0,   # ≈ top-12 RB season
    "TE": 175.0,   # ≈ top-6 TE season
}

# (display_label, row_key, operator, threshold)
_POS_BENCHMARKS: Dict[str, List[Tuple[str, str, str, float]]] = {
    "WR": [
        ("Rec yds/game ≥80",      "col_rec_yds_pg",     ">=", 80.0),
        ("Rec yds/game ≥65",      "col_rec_yds_pg",     ">=", 65.0),
        ("Season rec yds ≥1000",  "col_rec_yds_season", ">=", 1000.0),
        ("Season rec yds ≥800",   "col_rec_yds_season", ">=", 800.0),
        ("Dominator rating ≥30%", "col_dominator",      ">=", 0.30),
        ("Dominator rating ≥25%", "col_dominator",      ">=", 0.25),
        ("Pass share ≥24%",       "col_pass_share",     ">=", 0.24),
        ("YAC/rec ≥5.5",          "col_yac_per_rec",    ">=", 5.5),
        ("Draft age ≤21.5",       "col_age_at_draft",   "<=", 21.5),
        ("TDs/game ≥0.7",         "col_rec_tds_pg",     ">=", 0.7),
        ("1st-round pick",        "draft_pick",         "<=", 32),
        ("2nd-round or earlier",  "draft_pick",         "<=", 64),
    ],
    "RB": [
        ("Rush yds/game ≥100",    "col_rush_yds_pg",    ">=", 100.0),
        ("Rush yds/game ≥80",     "col_rush_yds_pg",    ">=", 80.0),
        ("Dominator rating ≥30%", "col_dominator",      ">=", 0.30),
        ("Dominator rating ≥20%", "col_dominator",      ">=", 0.20),
        ("YPC ≥5.5",              "col_ypc",            ">=", 5.5),
        ("Rec yds/game ≥20",      "col_rec_yds_pg",     ">=", 20.0),
        ("Draft age ≤21.5",       "col_age_at_draft",   "<=", 21.5),
        ("TDs/game ≥1.0",         "col_tds_pg",         ">=", 1.0),
        ("1st-round pick",        "draft_pick",         "<=", 32),
    ],
    "QB": [
        ("Completion% ≥70%",      "col_completion_pct", ">=", 70.0),
        ("Completion% ≥65%",      "col_completion_pct", ">=", 65.0),
        ("YPA ≥8.0",              "col_ypa",            ">=", 8.0),
        ("YPA ≥7.5",              "col_ypa",            ">=", 7.5),
        ("TD:INT ratio ≥3.0",     "col_td_int",         ">=", 3.0),
        ("Pass yds/game ≥280",    "col_pass_yds_pg",    ">=", 280.0),
        ("Rush yds/game ≥40",     "col_rush_yds_pg",    ">=", 40.0),
        ("Top-10 pick",           "draft_pick",         "<=", 10),
        ("1st-round pick",        "draft_pick",         "<=", 32),
    ],
    "TE": [
        ("Rec yds/game ≥55",      "col_rec_yds_pg",     ">=", 55.0),
        ("Rec yds/game ≥40",      "col_rec_yds_pg",     ">=", 40.0),
        ("Season rec yds ≥800",   "col_rec_yds_season", ">=", 800.0),
        ("Dominator rating ≥20%", "col_dominator",      ">=", 0.20),
        ("Dominator rating ≥15%", "col_dominator",      ">=", 0.15),
        ("Pass share ≥15%",       "col_pass_share",     ">=", 0.15),
        ("YAC/rec ≥4.0",          "col_yac_per_rec",    ">=", 4.0),
        ("TDs/game ≥0.4",         "col_rec_tds_pg",     ">=", 0.4),
        ("Draft age ≤22",         "col_age_at_draft",   "<=", 22.0),
        ("1st-round pick",        "draft_pick",         "<=", 32),
    ],
}

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


def _precision_recall_at_k(rows: List[Dict], k: int = 10) -> Tuple[float, float, int]:
    """
    Compute Precision@k and Recall@k using top-k cumulative PPR as positives.

    Returns (precision, recall, n_actual_positives).
    """
    with_data = [r for r in rows if r.get("ppr_cum", 0) > 0]
    if not with_data or k <= 0:
        return float("nan"), float("nan"), 0

    k_eff = min(k, len(with_data))
    by_model = sorted(with_data, key=lambda x: x.get("model_score", 0.0), reverse=True)
    by_actual = sorted(with_data, key=lambda x: x.get("ppr_cum", 0.0), reverse=True)
    pred_topk = {r["name"] for r in by_model[:k_eff]}
    actual_topk = {r["name"] for r in by_actual[:k_eff]}
    hits = len(pred_topk & actual_topk)
    precision = hits / k_eff
    recall = hits / len(actual_topk) if actual_topk else float("nan")
    return precision, recall, len(actual_topk)


def _ndcg_at_k(rows: List[Dict], k: int = 10, metric: str = "ppr_cum") -> float:
    """
    Compute NDCG@k where relevance is normalized actual `metric` value.
    """
    with_data = [r for r in rows if r.get(metric, 0) > 0]
    if len(with_data) < 2 or k <= 0:
        return float("nan")

    k_eff = min(k, len(with_data))
    max_rel = max(r.get(metric, 0.0) for r in with_data)
    if max_rel <= 0:
        return float("nan")

    rel = {r["name"]: (r.get(metric, 0.0) / max_rel) for r in with_data}
    by_model = sorted(with_data, key=lambda x: x.get("model_score", 0.0), reverse=True)[:k_eff]
    by_actual = sorted(with_data, key=lambda x: x.get(metric, 0.0), reverse=True)[:k_eff]

    def _dcg(items: List[Dict]) -> float:
        score = 0.0
        for idx, row in enumerate(items, 1):
            score += rel.get(row["name"], 0.0) / math.log2(idx + 1)
        return score

    ideal = _dcg(by_actual)
    if ideal <= 0:
        return float("nan")
    return _dcg(by_model) / ideal


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
    print(f"  {draft_year} DRAFT CLASS - Top {top_n} by Model Score  ({seasons_note})")
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


def _print_positional_rankings(rows: List[Dict], draft_year: int) -> None:
    """
    For a single draft class, show within-position model rank vs actual PPR rank.
    Displays QB1/QB2, WR1/WR2/WR3/WR4, RB1/RB2/RB3, TE1/TE2 labels.
    """
    pos_order = ["WR", "RB", "QB", "TE"]
    pos_top_n = {"WR": 8, "RB": 6, "QB": 4, "TE": 4}

    print(f"\n  ── {draft_year} Positional Rankings (Model vs Actual) ──")
    header = f"  {'Pos-Rank':<8}  {'Player':<25} {'Pick':>4}  {'Score':>5}  {'PPR-Y1':>6}  {'PPR-Y2':>6}  {'PPR-Cum':>7}  {'Actual':<8}"
    print(header)
    print(f"  {'-'*90}")

    for pos in pos_order:
        pos_rows = [r for r in rows if r["position"] == pos]
        if not pos_rows:
            continue

        # Sort by model score (descending) for model rank within position
        pos_by_model  = sorted(pos_rows, key=lambda x: x["model_score"], reverse=True)
        # Sort by cum PPR for actual rank within position
        has_ppr = [r for r in pos_rows if r["ppr_cum"] > 0]
        pos_by_actual = sorted(has_ppr, key=lambda x: x["ppr_cum"], reverse=True)
        actual_rank   = {r["name"]: i + 1 for i, r in enumerate(pos_by_actual)}

        top_n = pos_top_n.get(pos, 5)
        for model_pos_rank, r in enumerate(pos_by_model[:top_n], 1):
            ar   = actual_rank.get(r["name"])
            label = f"{pos}{model_pos_rank}"   # e.g. WR1, WR2

            if ar is not None:
                actual_label = f"{pos}{ar}"
                delta = model_pos_rank - ar
                # Arrow: ↑ = did better than model expected, ↓ = did worse
                arrow = "↑" if delta > 1 else ("↓" if delta < -1 else "≈")
                rank_str = f"{actual_label} ({arrow}{abs(delta):d})" if delta != 0 else f"{actual_label} (=)"
            else:
                rank_str = "no NFL data"

            ppr_y3_str = f"{r['ppr_y3']:>6.0f}" if r.get("ppr_y3", 0) > 0 else "     -"
            print(
                f"  {label:<8}  {r['name']:<25} #{r['draft_pick']:>3}  "
                f"{r['model_score']:>5.1f}  "
                f"{r['ppr_y1']:>6.0f}  {r['ppr_y2']:>6.0f}  "
                f"{r['ppr_cum']:>7.0f}  {rank_str}"
            )
        print()


def _print_positional_summary(all_rows: List[Dict]) -> None:
    """
    Cross-year positional analysis: how well does the model predict each position's
    within-position rank?  Shows Spearman-ρ per position per year.
    """
    pos_order = ["WR", "RB", "QB", "TE"]
    by_year: Dict[int, List[Dict]] = {}
    for r in all_rows:
        by_year.setdefault(r["draft_year"], []).append(r)

    print(f"\n{'=' * 90}")
    print("  POSITIONAL RANK ACCURACY  (model pos-rank vs actual PPR pos-rank, per year)")
    print(f"{'=' * 90}")

    years = sorted(by_year.keys())
    # Header
    year_cols = "  ".join(f"{y}" for y in years)
    print(f"  {'Pos':<5}  {year_cols}   Overall")
    print(f"  {'-'*80}")

    for pos in pos_order:
        year_rs = []
        for yr in years:
            yr_rows = [r for r in by_year[yr] if r["position"] == pos]
            has_ppr = [r for r in yr_rows if r["ppr_cum"] > 0]
            if len(has_ppr) < 3:
                year_rs.append("n/a")
                continue
            # Within-position model rank
            by_model  = sorted(yr_rows, key=lambda x: x["model_score"], reverse=True)
            model_pos_rank = {r["name"]: i + 1 for i, r in enumerate(by_model)}
            # Within-position actual rank
            by_actual = sorted(has_ppr, key=lambda x: x["ppr_cum"], reverse=True)
            actual_pos_rank = {r["name"]: i + 1 for i, r in enumerate(by_actual)}

            paired = [(model_pos_rank[r["name"]], actual_pos_rank[r["name"]])
                      for r in has_ppr if r["name"] in model_pos_rank]
            if len(paired) < 3:
                year_rs.append("n/a")
                continue
            mr_list  = [x for x, _ in paired]
            ar_list  = [y for _, y in paired]
            rho = _pearson_r(mr_list, ar_list)
            year_rs.append(f"{rho:+.2f}" if not math.isnan(rho) else " n/a")

        # Overall across all years for this position
        all_pos = [r for r in all_rows if r["position"] == pos and r["ppr_cum"] > 0]
        if len(all_pos) >= 5:
            # Need to build cross-year within-position ranks per class
            by_yr_pos: Dict[int, List] = {}
            for r in all_pos:
                by_yr_pos.setdefault(r["draft_year"], []).append(r)
            mr_all: List[float] = []
            ar_all: List[float] = []
            for yr, yr_pos_rows in by_yr_pos.items():
                by_model  = sorted(yr_pos_rows, key=lambda x: x["model_score"], reverse=True)
                model_pos_rank = {r["name"]: i + 1 for i, r in enumerate(by_model)}
                by_actual = sorted(yr_pos_rows, key=lambda x: x["ppr_cum"], reverse=True)
                actual_pos_rank = {r["name"]: i + 1 for i, r in enumerate(by_actual)}
                for r in yr_pos_rows:
                    if r["name"] in model_pos_rank and r["name"] in actual_pos_rank:
                        mr_all.append(model_pos_rank[r["name"]])
                        ar_all.append(actual_pos_rank[r["name"]])
            overall_r = _pearson_r(mr_all, ar_all) if len(mr_all) >= 5 else float("nan")
            overall_str = f"{overall_r:+.2f}" if not math.isnan(overall_r) else " n/a"
        else:
            overall_str = " n/a"

        year_col_str = "  ".join(f"{r:>5}" for r in year_rs)
        print(f"  {pos:<5}  {year_col_str}   {overall_str}")

    print()


def _print_benchmark_hit_rates(all_rows: List[Dict]) -> None:
    """
    For each position, compute the hit rate (% who reached top-6/12 PPR-peak)
    for every defined college benchmark threshold.
    Only players who have at least one season of NFL data (ppr_peak > 0) are counted.
    """
    print(f"\n{'=' * 105}")
    print("  BENCHMARK HIT RATES  -  college threshold  →  top-6 (QB/TE) or top-12 (WR/RB) fantasy season")
    print("  Approximate PPR-peak thresholds: QB ≥325 pts  |  WR ≥175 pts  |  RB ≥175 pts  |  TE ≥110 pts")
    print("  'N meet' = players with that stat + NFL data  |  'Hit%' = % who peaked above threshold")
    print("  'vs base' = hit-rate delta vs. all players at that position  |  col_ features require CFBD data")
    print(f"{'=' * 105}")

    for pos in ("WR", "RB", "QB", "TE"):
        benchmarks = _POS_BENCHMARKS.get(pos, [])
        hit_thresh = _HIT_PPR_THRESHOLD[pos]
        tier_label = "TOP-6" if pos in ("QB", "TE") else "TOP-12"

        # Only players with real NFL data
        with_data = [r for r in all_rows if r["position"] == pos and r.get("ppr_peak", 0) > 0]
        if not with_data:
            continue

        base_hits = sum(1 for r in with_data if r["ppr_peak"] >= hit_thresh)
        base_rate = base_hits / len(with_data) * 100

        print(f"\n  ── {pos}  ({tier_label}: PPR-peak ≥ {hit_thresh:.0f} pts)  "
              f"[{len(with_data)} draftees with NFL data, baseline {base_rate:.0f}% hit rate] ──")
        print(f"  {'Benchmark':<26}  {'N meet':>6}  {'N hit':>5}  {'Hit%':>5}  {'vs base':>8}  Bar (5% per block)")
        print(f"  {'-'*26}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*25}")

        results = []
        for lbl, key, op, thresh in benchmarks:
            eligible = []
            for r in with_data:
                val = r.get(key)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if   op == ">=" and val >= thresh:
                    eligible.append(r)
                elif op == "<=" and val <= thresh and val > 0:
                    eligible.append(r)
                elif op == ">"  and val >  thresh:
                    eligible.append(r)

            if len(eligible) < 3:
                results.append((lbl, None, None, None, None))
                continue

            hits = sum(1 for r in eligible if r["ppr_peak"] >= hit_thresh)
            rate = hits / len(eligible) * 100
            delta = rate - base_rate
            results.append((lbl, len(eligible), hits, rate, delta))

        # Sort by hit rate descending, n/a at bottom
        results.sort(key=lambda x: (x[3] is None, -(x[3] or 0)))

        for lbl, n_meet, n_hit, rate, delta in results:
            if rate is None:
                print(f"  {lbl:<26}  {'<3':>6}  {'-':>5}  {'-':>5}  {'-':>8}")
                continue
            delta_str = f"{delta:+.1f}%"
            bar_filled = min(int(rate / 5), 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            marker = " ◄" if abs(delta) >= 15 else ""
            print(f"  {lbl:<26}  {n_meet:>6}  {n_hit:>5}  {rate:>4.0f}%  {delta_str:>8}  {bar}{marker}")

        # Print baseline row at the bottom
        base_bar = "█" * min(int(base_rate / 5), 20) + "░" * (20 - min(int(base_rate / 5), 20))
        print(f"  {'(All - baseline)':<26}  {len(with_data):>6}  {base_hits:>5}  {base_rate:>4.0f}%  {'baseline':>8}  {base_bar}")


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
    print(
        f"\n  {'Year':>4}  {'Players':>7}  {'w/NFL data':>10}  {'CFBD hit%':>9}  "
        f"{'Spearman-ρ':>10}  {'Top10 hit':>8}  {'P@10':>6}  {'R@10':>6}  {'NDCG@10':>8}  {'NDCG@25':>8}"
    )
    print(f"  {'-'*4}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}")

    for yr in sorted(by_year.keys()):
        rows = by_year[yr]
        with_data = [r for r in rows if r["ppr_cum"] > 0]
        rc = _rank_corr(rows, "ppr_cum")
        rc_str = f"{rc:+.3f}" if not math.isnan(rc) else "  n/a"

        cfbd_n = sum(1 for r in rows if r.get("has_cfbd"))
        cfbd_pct = f"{cfbd_n/len(rows)*100:.0f}%" if rows else "n/a"

        # Top-10 hit rate: how many of model.s top 10 are in actual top 10?
        top10_model   = {r["name"] for r in rows[:10]}
        top10_actual = {r["name"] for r in sorted(with_data, key=lambda x: x["ppr_cum"], reverse=True)[:10]}
        hit_n = len(top10_model & top10_actual)
        hit_str = f"{hit_n}/10" if with_data else "n/a"
        p10, r10, _ = _precision_recall_at_k(rows, k=10)
        ndcg10 = _ndcg_at_k(rows, k=10, metric="ppr_cum")
        ndcg25 = _ndcg_at_k(rows, k=25, metric="ppr_cum")

        p10_str = f"{p10:.2f}" if not math.isnan(p10) else " n/a"
        r10_str = f"{r10:.2f}" if not math.isnan(r10) else " n/a"
        ndcg10_str = f"{ndcg10:.3f}" if not math.isnan(ndcg10) else "   n/a"
        ndcg25_str = f"{ndcg25:.3f}" if not math.isnan(ndcg25) else "   n/a"

        print(
            f"  {yr:>4}  {len(rows):>7}  {len(with_data):>10}  {cfbd_pct:>9}  {rc_str:>10}  {hit_str:>8}  "
            f"{p10_str:>6}  {r10_str:>6}  {ndcg10_str:>8}  {ndcg25_str:>8}"
        )

        for r in with_data:
            overall_valid.append((r["model_score"], r["ppr_cum"]))
            if r.get("has_cfbd"):
                overall_valid_cfbd.append((r["model_score"], r["ppr_cum"]))

    # Pooled decision-quality metrics should be averaged across classes, not
    # computed on one giant merged table (that collapses to just 10 positives).
    pooled_p10: List[float] = []
    pooled_r10: List[float] = []
    pooled_nd10: List[float] = []
    pooled_nd25: List[float] = []
    for yr in sorted(by_year.keys()):
        yr_rows = [r for r in by_year[yr] if r.get("ppr_cum", 0) > 0]
        if len(yr_rows) < 10:
            continue
        p10, r10, _ = _precision_recall_at_k(yr_rows, k=10)
        nd10 = _ndcg_at_k(yr_rows, k=10, metric="ppr_cum")
        nd25 = _ndcg_at_k(yr_rows, k=25, metric="ppr_cum")
        if not math.isnan(p10):
            pooled_p10.append(p10)
        if not math.isnan(r10):
            pooled_r10.append(r10)
        if not math.isnan(nd10):
            pooled_nd10.append(nd10)
        if not math.isnan(nd25):
            pooled_nd25.append(nd25)

    if pooled_p10:
        overall_p10 = statistics.mean(pooled_p10)
        overall_r10 = statistics.mean(pooled_r10) if pooled_r10 else float("nan")
        overall_ndcg10 = statistics.mean(pooled_nd10) if pooled_nd10 else float("nan")
        overall_ndcg25 = statistics.mean(pooled_nd25) if pooled_nd25 else float("nan")
        p10_str = f"{overall_p10:.2f}" if not math.isnan(overall_p10) else "n/a"
        r10_str = f"{overall_r10:.2f}" if not math.isnan(overall_r10) else "n/a"
        nd10_str = f"{overall_ndcg10:.3f}" if not math.isnan(overall_ndcg10) else "n/a"
        nd25_str = f"{overall_ndcg25:.3f}" if not math.isnan(overall_ndcg25) else "n/a"
        print(
            f"\n  Decision-quality ranking metrics (all classes pooled): "
            f"P@10={p10_str}  R@10={r10_str}  NDCG@10={nd10_str}  NDCG@25={nd25_str}"
        )

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
        # of data - a 2021 player at 250/season looks worse than a 2021 player at
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
# All-time top-10 tables
# ─────────────────────────────────────────────────────────────────────────────

def _print_all_time_top10(all_rows: List[Dict], top_n: int = 10) -> None:
    """
    Print top-N across all tested classes, balanced and per position.

    The "overall" table shows the top-N/4 from each position by model score
    (i.e. top-3 per position for top_n=10, rounded up).  Raw cross-position
    score comparison is misleading without CFBD data because draft capital
    alone drives 30% of the score and WRs are systematically drafted earlier
    than RBs/TEs - making busts like Jalen Reagor look better than any RB
    in the class purely on pick number.

    Uses PPR-per-season average for fair cross-year comparison.
    """
    with_nfl = [r for r in all_rows if r.get("ppr_cum", 0) > 0]
    no_nfl   = [r for r in all_rows if r.get("ppr_cum", 0) == 0]

    for r in with_nfl:
        r["ppr_avg"] = round(r["ppr_cum"] / max(r.get("seasons_avail", 1), 1), 0)

    header = (
        f"  {'#':>2}  {'Year':>4}  {'Player':<24} {'Pos':>3}  {'Pick':>4}  "
        f"{'Score':>5}  {'PPR-Y1':>6}  {'PPR-Y2':>6}  {'PPR/seas':>8}  {'Peak':>6}  Match"
    )
    divider = f"  {'─' * 95}"

    def _match(model_rank: int, actual_rank: int) -> str:
        delta = actual_rank - model_rank
        if abs(delta) <= 2:
            return f"≈ (Δ{delta:+d})"
        elif delta > 0:
            return f"↓ overrated  (actual #{actual_rank})"
        else:
            return f"↑ underrated (actual #{actual_rank})"

    # ── Balanced overall top-N: pick ceiling(top_n / 4) from each position ──
    per_pos = max(1, math.ceil(top_n / 4))
    positions_ordered = ["WR", "RB", "QB", "TE"]

    balanced: List[Dict] = []
    for pos in positions_ordered:
        pos_rows = sorted(
            [r for r in with_nfl if r["position"] == pos],
            key=lambda x: x["model_score"],
            reverse=True,
        )
        balanced.extend(pos_rows[:per_pos])

    # Sort the combined balanced list by model score for display
    balanced_sorted = sorted(balanced, key=lambda x: x["model_score"], reverse=True)

    by_actual_overall  = sorted(with_nfl, key=lambda x: x["ppr_avg"], reverse=True)
    actual_rank_overall = {r["name"]: i + 1 for i, r in enumerate(by_actual_overall)}

    cfbd_pct = sum(1 for r in with_nfl if r.get("has_cfbd")) / max(len(with_nfl), 1) * 100

    print(f"\n{'═' * 100}")
    print(
        f"  ALL-TIME TOP {per_pos} PER POSITION BY MODEL SCORE  "
        f"(balanced; {len(balanced_sorted)} players shown)"
    )
    if cfbd_pct < 30:
        print(
            f"  ⚠  Only {cfbd_pct:.0f}% of players have CFBD college stats - scores are draft capital + "
            f"athleticism only."
        )
        print(
            f"     Without production data the model cannot separate busts from stars at the same pick."
        )
        print(
            f"     Add CFBD_API_KEY to enable the full model (production, breakout, efficiency components)."
        )
    print(f"{'═' * 100}")
    print(header)
    print(divider)

    for i, r in enumerate(balanced_sorted, 1):
        ar = actual_rank_overall.get(r["name"], 999)
        y2 = f"{r['ppr_y2']:>6.0f}" if r.get("ppr_y2", 0) > 0 else "     -"
        print(
            f"  {i:>2}.  {r['draft_year']:>4}  {r['name']:<24} {r['position']:>3}  "
            f"#{r['draft_pick']:>3}  {r['model_score']:>5.1f}  "
            f"{r['ppr_y1']:>6.0f}  {y2}  "
            f"{r['ppr_avg']:>8.0f}  {r['ppr_peak']:>6.0f}  "
            f"{_match(i, ar)}"
        )

    # Players with high model score but no NFL data yet
    no_nfl_top = sorted(no_nfl, key=lambda x: x["model_score"], reverse=True)[:3]
    if no_nfl_top:
        print(f"\n  (Players with high model score but no NFL data yet:)")
        for r in no_nfl_top:
            print(f"       {r['draft_year']}  {r['name']:<24} {r['position']:>3}  "
                  f"#{r['draft_pick']:>3}  score={r['model_score']:.1f}")

    # ── Per-position top-N ───────────────────────────────────────────────────
    for pos in positions_ordered:
        pos_with_nfl = [r for r in with_nfl if r["position"] == pos]
        if not pos_with_nfl:
            continue

        by_model_pos  = sorted(pos_with_nfl, key=lambda x: x["model_score"], reverse=True)
        by_actual_pos = sorted(pos_with_nfl, key=lambda x: x["ppr_avg"], reverse=True)
        actual_rank_pos = {r["name"]: i + 1 for i, r in enumerate(by_actual_pos)}

        print(f"\n{'═' * 100}")
        print(f"  ALL-TIME TOP {top_n} {pos}s BY MODEL SCORE")
        print(f"{'═' * 100}")
        print(header)
        print(divider)

        for i, r in enumerate(by_model_pos[:top_n], 1):
            ar = actual_rank_pos.get(r["name"], 999)
            y2 = f"{r['ppr_y2']:>6.0f}" if r.get("ppr_y2", 0) > 0 else "     -"
            print(
                f"  {i:>2}.  {r['draft_year']:>4}  {r['name']:<24} {r['position']:>3}  "
                f"#{r['draft_pick']:>3}  {r['model_score']:>5.1f}  "
                f"{r['ppr_y1']:>6.0f}  {y2}  "
                f"{r['ppr_avg']:>8.0f}  {r['ppr_peak']:>6.0f}  "
                f"{_match(i, ar)}"
            )

        # Show actual top-N for this position so you can compare who the model missed
        print(f"\n  Actual top {top_n} {pos}s by PPR/season:")
        print(f"  {'#':>2}  {'Year':>4}  {'Player':<24} {'Pick':>4}  {'Model#':>6}  {'PPR/seas':>8}  {'Peak':>6}")
        print(f"  {'─' * 65}")
        for i, r in enumerate(by_actual_pos[:top_n], 1):
            model_pos_rank = next(
                (j + 1 for j, x in enumerate(by_model_pos) if x["name"] == r["name"]), 999
            )
            delta = model_pos_rank - i
            arrow = "≈" if abs(delta) <= 2 else ("↓" if delta > 0 else "↑")
            print(
                f"  {i:>2}.  {r['draft_year']:>4}  {r['name']:<24} #{r['draft_pick']:>3}  "
                f"Model#{model_pos_rank:>3} {arrow}  {r['ppr_avg']:>8.0f}  {r['ppr_peak']:>6.0f}"
            )

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Save to DB
# ─────────────────────────────────────────────────────────────────────────────

def _tier_from_score(score: float) -> int:
    if score >= 88: return 1
    if score >= 80: return 2
    if score >= 70: return 3
    if score >= 60: return 4
    if score >= 50: return 5
    return 6


_TIER_LABELS = {
    1: "Elite Prospect",
    2: "High-End Starter",
    3: "Solid Starter",
    4: "Rotational",
    5: "Depth/Developmental",
    6: "Long Shot",
}


def save_grades_to_db(all_rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Upsert all backtest rows into historical_prospect_grades.
    Returns (written, failed).
    """
    import re as _re
    try:
        from dashboard_services.db import get_conn
    except ImportError:
        print("[backtest] WARNING: dashboard_services.db not available - skipping DB save")
        return 0, 0

    def _slug(name: str) -> str:
        return _re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")

    written = failed = 0
    for r in all_rows:
        name  = r.get("name", "")
        pos   = r.get("position", "")
        year  = r.get("draft_year", 0)
        if not name or not pos or not year:
            continue

        hist_id = f"HIST_{year}_{_slug(name)}"
        score   = float(r.get("model_score") or 0)
        tier    = _tier_from_score(score)

        payload = {
            "player_id":             hist_id,
            "name":                  name,
            "position":              pos,
            "draft_class_year":      year,
            "school":                r.get("college") or None,
            "prospect_score":        round(score, 2),
            "tier":                  tier,
            "tier_label":            _TIER_LABELS.get(tier, ""),
            "overall_rank":          r.get("model_rank"),
            "position_rank":         r.get("pos_rank"),
            "production_score":      round(float(r.get("prod_score") or 0), 2),
            "efficiency_score":      round(float(r.get("efficiency_score") or 0), 2),
            "age_score":             round(float(r.get("age_score") or 0), 2),
            "breakout_profile_score":round(float(r.get("breakout_score") or 0), 2),
            "athleticism_score":     round(float(r.get("ath_score") or 0), 2),
            "competition_score":     round(float(r.get("competition_score") or 0), 2),
            "draft_capital_score":   round(float(r.get("dc_score") or 0), 2),
            "confidence_score":      round(float(r.get("confidence_score") or 0), 2),
            "actual_pick":           r.get("draft_pick") or None,
            "actual_round":          None,
            "actual_nfl_team":       None,
            "headshot_url":          None,
        }

        try:
            with get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO historical_prospect_grades (
                        player_id, name, position, draft_class_year, school,
                        prospect_score, tier, tier_label, overall_rank, position_rank,
                        production_score, efficiency_score, age_score,
                        breakout_profile_score, athleticism_score,
                        competition_score, draft_capital_score, confidence_score,
                        actual_pick, actual_round, actual_nfl_team, headshot_url
                    ) VALUES (
                        %(player_id)s, %(name)s, %(position)s, %(draft_class_year)s, %(school)s,
                        %(prospect_score)s, %(tier)s, %(tier_label)s, %(overall_rank)s, %(position_rank)s,
                        %(production_score)s, %(efficiency_score)s, %(age_score)s,
                        %(breakout_profile_score)s, %(athleticism_score)s,
                        %(competition_score)s, %(draft_capital_score)s, %(confidence_score)s,
                        %(actual_pick)s, %(actual_round)s, %(actual_nfl_team)s, %(headshot_url)s
                    )
                    ON CONFLICT (player_id) DO UPDATE SET
                        prospect_score          = EXCLUDED.prospect_score,
                        tier                    = EXCLUDED.tier,
                        tier_label              = EXCLUDED.tier_label,
                        overall_rank            = EXCLUDED.overall_rank,
                        position_rank           = EXCLUDED.position_rank,
                        production_score        = EXCLUDED.production_score,
                        efficiency_score        = EXCLUDED.efficiency_score,
                        age_score               = EXCLUDED.age_score,
                        breakout_profile_score  = EXCLUDED.breakout_profile_score,
                        athleticism_score       = EXCLUDED.athleticism_score,
                        competition_score       = EXCLUDED.competition_score,
                        draft_capital_score     = EXCLUDED.draft_capital_score,
                        confidence_score        = EXCLUDED.confidence_score,
                        actual_pick             = EXCLUDED.actual_pick
                    """,
                    payload,
                )
                conn.commit()
            written += 1
        except Exception as e:
            print(f"  [db] Failed to save {name}: {e}")
            failed += 1

    return written, failed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(draft_years: Optional[List[int]] = None, save_grades: bool = True) -> List[Dict[str, Any]]:
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
            _print_positional_rankings(rows, dy)

    if all_rows:
        _print_positional_summary(all_rows)
        _print_benchmark_hit_rates(all_rows)
        _print_summary(all_rows)
        _print_all_time_top10(all_rows, top_n=TOP_N_PER_CLASS)

        if save_grades:
            print(f"\n[backtest] Saving {len(all_rows)} grades to historical_prospect_grades…")
            written, failed = save_grades_to_db(all_rows)
            print(f"[backtest] Saved {written} rows" + (f" ({failed} failed)" if failed else "") + ".")

    return all_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest rookie prospect model (2016-2025)")
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=DRAFT_YEARS,
        help="Draft years to include (default: 2021 2022 2023 2024 2025)",
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N_PER_CLASS,
        help=f"Rows per draft class (default: {TOP_N_PER_CLASS})",
    )
    parser.add_argument(
        "--benchmark-profile",
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Benchmark boost profile to use during scoring (default: conservative)",
    )
    parser.add_argument(
        "--no-save-grades",
        action="store_true",
        help="Skip writing grades to historical_prospect_grades table",
    )
    args = parser.parse_args()
    os.environ["ROOKIE_BENCHMARK_PROFILE"] = args.benchmark_profile
    TOP_N_PER_CLASS = args.top_n
    run_backtest(args.years, save_grades=not args.no_save_grades)
