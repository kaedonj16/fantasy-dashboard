"""Prospect accuracy report: compare rookie prospect grades vs actual NFL outcomes.

Closes the feedback loop on the rookie prospect grading model by:
1. Collecting actual NFL performance (PPR, games) for each graded prospect
   at Y+1, Y+2, Y+3 from nflverse data
2. Computing hit/miss against position-specific PPR thresholds
3. Aggregating hit rates by grade tier, position, and draft class
4. Identifying systematic biases (e.g. overgrading Day 3 WRs)

Usage:
    python scripts/prospect_accuracy_report.py [--year 2026] [--draft-class 2023]
    python scripts/prospect_accuracy_report.py --out /tmp/accuracy.md

Scheduled: annually in February via cron_daily.py (after the NFL season ends).
This is separate from the rookie pipeline schedule (which is paused during
the CFB season).

Tables:
    prospect_nfl_outcomes     - per-player NFL performance + hit flag
    prospect_accuracy_reports - precomputed aggregates by tier/position/year
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services.db import get_conn
from data_building.rookie_pipeline.historical_calibration import (
    _fetch_csv,
    _NFLVERSE_BASE,
    _NFLVERSE_ROSTER,
    _calc_ppr_points,
    _safe_float,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# PPR-peak threshold approximating a "hit" season per position.
# QB/TE: top-6 positional finish. WR/RB: top-12 positional finish.
# (Matches scripts/backtest_prospect_model.py::_HIT_PPR_THRESHOLD)
HIT_PPR_THRESHOLD: Dict[str, float] = {
    "QB": 310.0,
    "WR": 220.0,
    "RB": 240.0,
    "TE": 175.0,
}

SKILL_POS = {"QB", "RB", "WR", "TE"}
NFL_SEASONS = 3  # Y+1, Y+2, Y+3

# Minimum players for a bias finding to be reported
MIN_N_FOR_BIAS = 8


# ─────────────────────────────────────────────────────────────────────────────
# NFL outcome collection (from nflverse)
# ─────────────────────────────────────────────────────────────────────────────

def _build_gsis_to_sleeper() -> Dict[str, str]:
    """Build gsis_id -> sleeper_id crosswalk from nflverse roster data."""
    try:
        from data_building.external_data.nflverse_metrics import _gsis_to_sleeper
        return _gsis_to_sleeper()
    except Exception:
        return {}


def _fetch_nfl_outcomes(
    draft_year: int,
    gsis_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Fetch per-player NFL PPR + games for Y+1..Y+3 from nflverse.

    Returns: {gsis_id: {ppr_y1, ppr_y2, ppr_y3, ppr_peak, ppr_cum,
                        games_y1, games_y2, games_y3, seasons_with_data}}
    Processes one season at a time to bound memory.
    """
    gid_set = set(gsis_ids)
    # gsis_id -> [ppr_y1, ppr_y2, ppr_y3], [games_y1, games_y2, games_y3]
    ppr: Dict[str, List[float]] = {gid: [0.0, 0.0, 0.0] for gid in gid_set}
    games: Dict[str, List[int]] = {gid: [0, 0, 0] for gid in gid_set}
    seen_weeks: Dict[str, List[set]] = {gid: [set(), set(), set()] for gid in gid_set}

    for offset in range(NFL_SEASONS):
        nfl_yr = draft_year + offset
        stat_rows = _fetch_csv(_NFLVERSE_BASE.format(year=nfl_yr), quiet_404=True)
        if not stat_rows:
            continue
        for sr in stat_rows:
            gid = sr.get("player_id") or ""
            if gid not in gid_set:
                continue
            if (sr.get("season_type") or "REG").upper() != "REG":
                continue
            ppr[gid][offset] += _calc_ppr_points(sr)
            wk = sr.get("week") or ""
            if wk not in seen_weeks[gid][offset]:
                seen_weeks[gid][offset].add(wk)
                games[gid][offset] += 1

    result = {}
    for gid in gid_set:
        pts = ppr[gid]
        gms = games[gid]
        available = sum(1 for x in pts if x > 0)
        result[gid] = {
            "ppr_y1": round(pts[0], 2),
            "ppr_y2": round(pts[1], 2),
            "ppr_y3": round(pts[2], 2),
            "ppr_peak": round(max(pts), 2),
            "ppr_cum": round(sum(pts), 2),
            "games_y1": gms[0],
            "games_y2": gms[1],
            "games_y3": gms[2],
            "seasons_with_data": available,
        }
    return result


def _load_draft_class_gsis(draft_year: int) -> Dict[str, Dict[str, str]]:
    """Load gsis_id -> {name, position} for drafted skill players in a class."""
    rows = _fetch_csv(_NFLVERSE_ROSTER.format(year=draft_year), quiet_404=True)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row.get("rookie_year") != str(draft_year):
            continue
        pos = (row.get("position") or "").upper()
        if pos not in SKILL_POS:
            continue
        gid = row.get("gsis_id") or ""
        if not gid or not (row.get("draft_number") or "").strip():
            continue
        # Keep first occurrence per gsis_id
        if gid not in out:
            out[gid] = {
                "name": row.get("full_name") or "",
                "position": pos,
            }
    return out


def _normalize_name(name: str) -> str:
    """Normalize a name for fuzzy matching."""
    import re
    import unicodedata
    n = unicodedata.normalize("NFKD", name or "")
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"[^a-z ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    # Strip common suffixes
    for suf in (" jr", " sr", " iii", " ii", " iv", " v"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Grade loading + matching
# ─────────────────────────────────────────────────────────────────────────────

def _load_graded_prospects(
    conn, draft_class_year: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load prospects with computed grades from historical_prospect_grades."""
    with conn.cursor() as cur:
        if draft_class_year:
            cur.execute(
                """
                select player_id, sleeper_id, name, position, draft_class_year,
                       prospect_score, tier, tier_label, overall_rank,
                       position_rank, actual_pick, actual_round
                from historical_prospect_grades
                where draft_class_year = %s
                  and prospect_score is not null
                """,
                (draft_class_year,),
            )
        else:
            cur.execute(
                """
                select player_id, sleeper_id, name, position, draft_class_year,
                       prospect_score, tier, tier_label, overall_rank,
                       position_rank, actual_pick, actual_round
                from historical_prospect_grades
                where prospect_score is not null
                order by draft_class_year, position_rank nulls last
                """
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _match_prospects_to_nfl(
    prospects: List[Dict[str, Any]],
    gsis_to_sleeper: Dict[str, str],
    draft_class: Dict[str, Dict[str, str]],
) -> List[Tuple[Dict[str, Any], Optional[str]]]:
    """Match each graded prospect to an nflverse gsis_id.

    Strategy: (1) sleeper_id via gsis->sleeper crosswalk (reversed),
    (2) normalized name + position match against the draft class roster.
    Returns list of (prospect, gsis_id or None).
    """
    sleeper_to_gsis = {v: k for k, v in gsis_to_sleeper.items()}

    # Build name index for fallback matching
    name_idx: Dict[Tuple[str, str], str] = {}
    for gid, info in draft_class.items():
        key = (_normalize_name(info["name"]), info["position"])
        if key not in name_idx:
            name_idx[key] = gid

    matched = []
    for p in prospects:
        gid = None
        sid = p.get("sleeper_id")
        if sid and sid in sleeper_to_gsis:
            gid = sleeper_to_gsis[sid]
        if not gid:
            key = (_normalize_name(p["name"]), (p["position"] or "").upper())
            gid = name_idx.get(key)
        matched.append((p, gid))
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Hit computation + storage
# ─────────────────────────────────────────────────────────────────────────────

def compute_hit(position: str, ppr_peak: float) -> Tuple[bool, Optional[int], List[float], float]:
    """Determine if a prospect is a hit.

    Returns (is_hit, hit_season, [ppr_y1..y3], ppr_peak).
    hit_season is the first season where the threshold was met (1-based).
    """
    thresh = HIT_PPR_THRESHOLD.get((position or "").upper(), 0)
    return ppr_peak >= thresh


def _upsert_outcomes(
    conn,
    rows: List[Dict[str, Any]],
) -> int:
    """Upsert NFL outcomes into prospect_nfl_outcomes. Returns row count."""
    n = 0
    with conn.cursor() as cur:
        for r in rows:
            ppr = [r["ppr_y1"], r["ppr_y2"], r["ppr_y3"]]
            thresh = HIT_PPR_THRESHOLD.get((r["position"] or "").upper(), 0)
            is_hit = r["ppr_peak"] >= thresh
            hit_season = None
            for i, pts in enumerate(ppr):
                if pts >= thresh:
                    hit_season = i + 1
                    break
            cur.execute(
                """
                insert into prospect_nfl_outcomes
                    (player_id, draft_class_year, position,
                     ppr_y1, ppr_y2, ppr_y3, ppr_peak, ppr_cumulative,
                     games_y1, games_y2, games_y3, seasons_with_data,
                     is_hit, hit_season, updated_at)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                on conflict (player_id) do update set
                    ppr_y1 = excluded.ppr_y1,
                    ppr_y2 = excluded.ppr_y2,
                    ppr_y3 = excluded.ppr_y3,
                    ppr_peak = excluded.ppr_peak,
                    ppr_cumulative = excluded.ppr_cumulative,
                    games_y1 = excluded.games_y1,
                    games_y2 = excluded.games_y2,
                    games_y3 = excluded.games_y3,
                    seasons_with_data = excluded.seasons_with_data,
                    is_hit = excluded.is_hit,
                    hit_season = excluded.hit_season,
                    updated_at = now()
                """,
                (
                    r["player_id"], r["draft_class_year"], r["position"],
                    r["ppr_y1"], r["ppr_y2"], r["ppr_y3"],
                    r["ppr_peak"], r["ppr_cum"],
                    r["games_y1"], r["games_y2"], r["games_y3"],
                    r["seasons_with_data"], is_hit, hit_season,
                ),
            )
            n += 1
    conn.commit()
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy aggregates
# ─────────────────────────────────────────────────────────────────────────────

def _compute_aggregates(conn, report_year: int) -> List[Dict[str, Any]]:
    """Compute hit-rate aggregates by (draft_class_year, position, tier).

    Also computes position='ALL' rollups per year/tier.
    Returns list of aggregate dicts.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            select g.draft_class_year, g.position, g.tier,
                   g.prospect_score, o.is_hit
            from historical_prospect_grades g
            join prospect_nfl_outcomes o on o.player_id = g.player_id
            where g.prospect_score is not null
            """
        )
        rows = cur.fetchall()

    # (year, pos, tier) -> stats
    from collections import defaultdict
    buckets: Dict[Tuple[int, str, Optional[int]], Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "hits": 0, "scores": []}
    )
    for year, pos, tier, score, is_hit in rows:
        for key_pos in (pos, "ALL"):
            for key_tier in (tier, None):
                b = buckets[(year, key_pos, key_tier)]
                b["n"] += 1
                if is_hit:
                    b["hits"] += 1
                if score is not None:
                    b["scores"].append(float(score))

    aggregates = []
    for (year, pos, tier), b in sorted(buckets.items()):
        n = b["n"]
        hit_rate = round(b["hits"] / n * 100, 2) if n else 0.0
        avg_score = (
            round(sum(b["scores"]) / len(b["scores"]), 2) if b["scores"] else None
        )
        aggregates.append({
            "report_year": report_year,
            "draft_class_year": year,
            "position": pos,
            "tier": tier,
            "n_players": n,
            "n_with_nfl_data": n,  # all joined rows have outcome data
            "n_hits": b["hits"],
            "hit_rate": hit_rate,
            "avg_prospect_score": avg_score,
        })
    return aggregates


def _store_aggregates(conn, aggregates: List[Dict[str, Any]]) -> int:
    n = 0
    with conn.cursor() as cur:
        for a in aggregates:
            cur.execute(
                """
                insert into prospect_accuracy_reports
                    (report_year, draft_class_year, position, tier,
                     n_players, n_with_nfl_data, n_hits, hit_rate,
                     avg_prospect_score, created_at)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                on conflict (report_year, draft_class_year, position, tier)
                do update set
                    n_players = excluded.n_players,
                    n_with_nfl_data = excluded.n_with_nfl_data,
                    n_hits = excluded.n_hits,
                    hit_rate = excluded.hit_rate,
                    avg_prospect_score = excluded.avg_prospect_score,
                    created_at = now()
                """,
                (
                    a["report_year"], a["draft_class_year"], a["position"],
                    a["tier"], a["n_players"], a["n_with_nfl_data"],
                    a["n_hits"], a["hit_rate"], a["avg_prospect_score"],
                ),
            )
            n += 1
    conn.commit()
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Bias detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_biases(conn) -> List[str]:
    """Identify systematic grading biases. Returns human-readable findings."""
    findings: List[str] = []
    with conn.cursor() as cur:
        # 1. Tier calibration: does hit rate increase monotonically with tier?
        cur.execute(
            """
            select g.tier, count(*), sum(case when o.is_hit then 1 else 0 end)
            from historical_prospect_grades g
            join prospect_nfl_outcomes o on o.player_id = g.player_id
            where g.tier is not null and g.prospect_score is not null
            group by g.tier order by g.tier
            """
        )
        tier_rows = cur.fetchall()
        if len(tier_rows) >= 3:
            rates = []
            for tier, n, hits in tier_rows:
                if n >= MIN_N_FOR_BIAS:
                    rates.append((tier, hits / n * 100))
            # Check monotonicity: tier 1 should out-hit tier 2, etc.
            for i in range(len(rates) - 1):
                t1, r1 = rates[i]
                t2, r2 = rates[i + 1]
                if r1 < r2 - 5:  # lower tier out-hits higher tier by >5pts
                    findings.append(
                        f"Tier inversion: Tier {t1} hit rate ({r1:.0f}%) is "
                        f"below Tier {t2} ({r2:.0f}%). The tier boundaries may "
                        f"need recalibration."
                    )

        # 2. Position accuracy: which positions are we best/worst at grading?
        cur.execute(
            """
            select g.position, count(*),
                   sum(case when o.is_hit then 1 else 0 end),
                   avg(g.prospect_score)
            from historical_prospect_grades g
            join prospect_nfl_outcomes o on o.player_id = g.player_id
            where g.prospect_score is not null
            group by g.position
            """
        )
        pos_rows = cur.fetchall()
        pos_rates = []
        for pos, n, hits, avg_score in pos_rows:
            if n >= MIN_N_FOR_BIAS:
                pos_rates.append((pos, hits / n * 100, n))
        if len(pos_rates) >= 2:
            pos_rates.sort(key=lambda x: x[1])
            worst, best = pos_rates[0], pos_rates[-1]
            if best[1] - worst[1] >= 15:
                findings.append(
                    f"Position gap: {best[0]} grades hit at {best[1]:.0f}% "
                    f"(n={best[2]}) vs {worst[0]} at {worst[1]:.0f}% "
                    f"(n={worst[2]}). Consider reweighting {worst[0]} components."
                )

        # 3. Draft capital vs model: do late picks we liked actually hit?
        cur.execute(
            """
            select
                case when g.actual_round >= 3 then 'Day 3+' else 'Day 1-2' end as bucket,
                count(*),
                sum(case when o.is_hit then 1 else 0 end),
                avg(g.prospect_score)
            from historical_prospect_grades g
            join prospect_nfl_outcomes o on o.player_id = g.player_id
            where g.prospect_score is not null and g.actual_round is not null
            group by 1
            """
        )
        for bucket, n, hits, avg_score in cur.fetchall():
            if n >= MIN_N_FOR_BIAS and bucket == "Day 3+":
                rate = hits / n * 100
                if rate >= 20:
                    findings.append(
                        f"Day 3+ value: {rate:.0f}% hit rate (n={n}) on late "
                        f"picks we graded above baseline. The Day-3 penalty "
                        f"may be too harsh."
                    )
                elif rate <= 5:
                    findings.append(
                        f"Day 3+ overgrading: only {rate:.0f}% hit rate (n={n}) "
                        f"on Day 3+ picks. Consider strengthening the Day-3 penalty."
                    )

        # 4. Score calibration: do high scores actually hit more?
        cur.execute(
            """
            select
                case
                    when g.prospect_score >= 85 then '85+'
                    when g.prospect_score >= 72 then '72-84'
                    when g.prospect_score >= 60 then '60-71'
                    else '<60'
                end as band,
                count(*),
                sum(case when o.is_hit then 1 else 0 end)
            from historical_prospect_grades g
            join prospect_nfl_outcomes o on o.player_id = g.player_id
            where g.prospect_score is not null
            group by 1
            """
        )
        bands = {band: (n, hits) for band, n, hits in cur.fetchall()}
        if "85+" in bands and "<60" in bands:
            n_hi, h_hi = bands["85+"]
            n_lo, h_lo = bands["<60"]
            if n_hi >= 5 and n_lo >= MIN_N_FOR_BIAS:
                r_hi = h_hi / n_hi * 100
                r_lo = h_lo / n_lo * 100
                if r_hi - r_lo < 20:
                    findings.append(
                        f"Weak discrimination: 85+ scores hit at {r_hi:.0f}% "
                        f"vs <60 at {r_lo:.0f}%. The model struggles to "
                        f"separate elite from replacement-level."
                    )

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_markdown(
    conn, report_year: int, biases: List[str]
) -> str:
    """Generate the accuracy report as markdown."""
    with conn.cursor() as cur:
        cur.execute(
            """
            select draft_class_year, position, tier, n_players,
                   n_hits, hit_rate, avg_prospect_score
            from prospect_accuracy_reports
            where report_year = %s and position = 'ALL' and tier is null
            order by draft_class_year
            """,
            (report_year,),
        )
        year_rows = cur.fetchall()

        cur.execute(
            """
            select position, tier, sum(n_players), sum(n_hits),
                   avg(hit_rate), avg(avg_prospect_score)
            from prospect_accuracy_reports
            where report_year = %s and position != 'ALL'
            group by position, tier order by position, tier
            """,
            (report_year,),
        )
        tier_rows = cur.fetchall()

        cur.execute("select count(*) from prospect_nfl_outcomes")
        total_outcomes = cur.fetchone()[0]
        cur.execute(
            "select count(*) from prospect_nfl_outcomes where is_hit"
        )
        total_hits = cur.fetchone()[0]

    lines = [
        f"# Prospect Grading Accuracy Report ({report_year})",
        "",
        f"Generated after the {report_year - 1} NFL season. Compares prospect "
        f"grades against actual NFL performance (Y+1 through Y+3).",
        "",
        f"**Coverage:** {total_outcomes} graded prospects with NFL outcomes, "
        f"{total_hits} hits.",
        "",
        "Hit definition: PPR peak season >= position threshold "
        "(QB 310, WR 220, RB 240, TE 175).",
        "",
        "## Hit Rate by Draft Class",
        "",
        "| Class | Players | Hits | Hit Rate | Avg Score |",
        "|-------|---------|------|----------|-----------|",
    ]
    for year, pos, tier, n, hits, rate, avg in year_rows:
        lines.append(
            f"| {year} | {n} | {hits} | {rate:.1f}% | {avg:.1f} |"
            if avg is not None else
            f"| {year} | {n} | {hits} | {rate:.1f}% | n/a |"
        )

    lines += ["", "## Hit Rate by Position and Tier", "",
              "| Pos | Tier | Players | Hits | Hit Rate | Avg Score |",
              "|-----|------|---------|------|----------|-----------|"]
    for pos, tier, n, hits, rate, avg in tier_rows:
        tier_lbl = str(tier) if tier is not None else "All"
        lines.append(
            f"| {pos} | {tier_lbl} | {n} | {hits} | {rate:.1f}% | "
            f"{avg:.1f} |" if avg is not None else
            f"| {pos} | {tier_lbl} | {n} | {hits} | {rate:.1f}% | n/a |"
        )

    lines += ["", "## Systematic Biases", ""]
    if biases:
        for b in biases:
            lines.append(f"- {b}")
    else:
        lines.append("No significant biases detected at current sample sizes.")
    lines += ["", "_Thresholds: minimum 8 players per group for bias findings._", ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_accuracy_report(
    draft_class_year: Optional[int] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full accuracy loop. Returns summary stats."""
    import datetime
    report_year = datetime.date.today().year

    stats: Dict[str, Any] = {
        "report_year": report_year,
        "classes_processed": [],
        "outcomes_upserted": 0,
        "aggregates_stored": 0,
        "biases": [],
    }

    with get_conn() as conn:
        prospects = _load_graded_prospects(conn, draft_class_year)
        if not prospects:
            print("[accuracy] No graded prospects found. "
                  "Run scripts/populate_historical_grades.py first.")
            return stats

        # Group by draft class for batched NFL fetching
        by_year: Dict[int, List[Dict[str, Any]]] = {}
        for p in prospects:
            by_year.setdefault(p["draft_class_year"], []).append(p)

        gsis_to_sleeper = _build_gsis_to_sleeper()
        all_outcome_rows: List[Dict[str, Any]] = []

        for year in sorted(by_year):
            # Skip classes too recent to have meaningful outcomes
            # (need at least Y+1 complete)
            if year > report_year - 1:
                print(f"[accuracy] Skipping {year}: season not complete yet")
                continue

            class_prospects = by_year[year]
            print(f"[accuracy] Processing {year}: {len(class_prospects)} graded prospects")

            draft_class = _load_draft_class_gsis(year)
            matched = _match_prospects_to_nfl(
                class_prospects, gsis_to_sleeper, draft_class
            )
            matched_ids = [gid for _, gid in matched if gid]
            n_matched = len(matched_ids)
            print(f"[accuracy]   matched {n_matched}/{len(class_prospects)} to NFL data")

            if not matched_ids:
                continue

            outcomes = _fetch_nfl_outcomes(year, matched_ids)

            for p, gid in matched:
                if not gid or gid not in outcomes:
                    continue
                o = outcomes[gid]
                all_outcome_rows.append({
                    "player_id": p["player_id"],
                    "draft_class_year": year,
                    "position": (p["position"] or "").upper(),
                    **o,
                })

            stats["classes_processed"].append(year)

        if all_outcome_rows:
            n = _upsert_outcomes(conn, all_outcome_rows)
            stats["outcomes_upserted"] = n
            print(f"[accuracy] Upserted {n} NFL outcomes")

        aggregates = _compute_aggregates(conn, report_year)
        n_agg = _store_aggregates(conn, aggregates)
        stats["aggregates_stored"] = n_agg
        print(f"[accuracy] Stored {n_agg} aggregate rows")

        biases = _detect_biases(conn)
        stats["biases"] = biases

        report_md = _generate_markdown(conn, report_year, biases)
        if out_path:
            with open(out_path, "w") as f:
                f.write(report_md)
            print(f"[accuracy] Report written to {out_path}")
        else:
            print()
            print(report_md)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prospect grading accuracy report"
    )
    parser.add_argument("--draft-class", type=int, default=None,
                        help="Only process a single draft class year")
    parser.add_argument("--out", type=str, default=None,
                        help="Write markdown report to this path (default: stdout)")
    args = parser.parse_args()
    run_accuracy_report(
        draft_class_year=args.draft_class, out_path=args.out
    )


if __name__ == "__main__":
    main()
