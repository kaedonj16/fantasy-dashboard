#!/usr/bin/env python3
"""Walk-forward backtest of the live blended PowerScore vs simple baselines.

Question
--------
At week N of a *completed* season, does ``blended_team_scores`` (the production
formula in ``dashboard_services/power_score.py``) predict the future better than
all-play win %, points-per-game, or actual win %?

Targets (Spearman, higher = better agreement):
  * final  — week-N ranking vs regular-season finishing order (wins, then PF)
  * ros    — week-N ranking vs rest-of-season actual H2H win rate (weeks N+1+)

The ROS target cannot leak past results; it is the headline number.

Method
------
Inputs are assembled the same way the live board does
(``build_power_rankings_context``), then scored with the real
``blended_team_scores`` / ``PHASE_WEIGHTS`` path. Do not reimplement the blend.

Historical week-N roster values are not stored in this repo or its database, so
``starter_value`` is held at 0 for every team (the value z-score is then 0 and
rank-equivalent to dropping that term). Playoff % is a leak-free Monte Carlo
of the remaining published schedule using each team's scoring mean/std through
week N — not future Sleeper projections.

Data
----
Completed-season weekly scores are loaded from, in order:

  1. ``--from-json`` files (one league-season per JSON object / JSONL line)
  2. ``--leagues platform:league_id:season`` (fetched from the platform API)
  3. Seed Sleeper league IDs that already appear in this repo's tests, walked
     via ``previous_league_id`` and (optionally) co-owners' other completed
     seasons. Matchup rows come from the same Sleeper ``/matchups/{week}``
     endpoint ``build_tables`` uses. Nothing is synthesized.

If none of those sources yield a completed regular season, the script exits
with a missing-data report instead of fabricating a result.

Usage
-----
    python scripts/backtest_power_rankings.py
    python scripts/backtest_power_rankings.py --max-leagues 12 --min-week 4
    python scripts/backtest_power_rankings.py --leagues sleeper:123:2025
    python scripts/backtest_power_rankings.py --from-json /tmp/league.json
    python scripts/backtest_power_rankings.py --no-fetch   # local files/DB only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# League IDs that already appear in committed tests. Used only as a discovery
# seed for the public Sleeper API — not as synthetic weekly scores.
REPO_SEED_LEAGUE_IDS = (
    "1312067280816832512",  # tests/test_team_modal_schedule.py (2026 in-season)
    "1236152168919072768",  # 2025 complete predecessor of the above
)

METHODS = ("blend", "all_play", "ppg", "win_pct")
COMPONENTS = ("pf", "record", "momentum", "sos", "ros", "playoff")
PHASES = ("early", "mid", "late")

# Weeks N in this range are scored. N=4 is the first week momentum is defined
# in production (needs a season baseline vs last-3).
DEFAULT_MIN_WEEK = 4


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _rank_avg_ties(vals: Sequence[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman ρ: Pearson on average-tie ranks. None when undefined."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    return pearson(_rank_avg_ties(xs), _rank_avg_ties(ys))


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt(x: Optional[float], digits: int = 3, width: int = 8) -> str:
    if x is None:
        return f"{'n/a':>{width}}"
    return f"{x:+{width}.{digits}f}"


# ---------------------------------------------------------------------------
# League-season container
# ---------------------------------------------------------------------------


@dataclass
class LeagueSeason:
    """One completed regular season of weekly fantasy scores."""

    platform: str
    league_id: str
    season: int
    name: str
    n_teams: int
    playoff_week_start: int
    playoff_teams: int
    rows: list[dict]  # week, roster_id (str), matchup_id, points
    roster_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.roster_ids:
            self.roster_ids = sorted({str(r["roster_id"]) for r in self.rows})

    @property
    def last_reg_week(self) -> int:
        return int(self.playoff_week_start) - 1

    def weeks(self) -> list[int]:
        return sorted({int(r["week"]) for r in self.rows if int(r["week"]) <= self.last_reg_week})

    def rows_through(self, week_n: int) -> list[dict]:
        return [r for r in self.rows if int(r["week"]) <= int(week_n)]

    def rows_after(self, week_n: int) -> list[dict]:
        return [
            r for r in self.rows
            if int(week_n) < int(r["week"]) <= self.last_reg_week
        ]

    def to_json(self) -> dict:
        return {
            "platform": self.platform,
            "league_id": self.league_id,
            "season": self.season,
            "name": self.name,
            "n_teams": self.n_teams,
            "playoff_week_start": self.playoff_week_start,
            "playoff_teams": self.playoff_teams,
            "rows": self.rows,
        }

    @classmethod
    def from_json(cls, blob: dict) -> "LeagueSeason":
        return cls(
            platform=str(blob.get("platform") or "sleeper"),
            league_id=str(blob["league_id"]),
            season=int(blob["season"]),
            name=str(blob.get("name") or blob["league_id"]),
            n_teams=int(blob.get("n_teams") or 0),
            playoff_week_start=int(blob.get("playoff_week_start") or 15),
            playoff_teams=int(blob.get("playoff_teams") or 6),
            rows=list(blob.get("rows") or []),
            roster_ids=list(blob.get("roster_ids") or []),
        )


def missing_data_report(*, tried: Sequence[str] = ()) -> str:
    lines = [
        "MISSING DATA: no completed-season weekly fantasy scores were found.",
        "",
        "The live PowerScore blend needs per-roster weekly points and matchup_id",
        "pairings (the df_weekly frame build_tables builds from /matchups/{week}).",
        "This environment does not have that table in Postgres, and the repo cache",
        "only stores NFL-player weekly stats (cache/sleeper_stats/), not league",
        "team scores.",
        "",
        "Needed to run the backtest:",
        "  • at least one completed regular season with weekly H2H scores for",
        "    every roster (weeks 1..playoff_week_start-1)",
        "  • matchup_id (or equivalent pairing) so SoS / ROS / actual W-L can",
        "    be reconstructed without leaking future weeks",
        "",
        "Optional but used by the live board and neutralized here when absent:",
        "  • week-N roster snapshots + contemporaneous redraft values",
        "    (starter_value). Not stored historically; held at 0 so the value",
        "    term does not leak end-of-season rosters.",
        "",
        "How to supply data:",
        "  python scripts/backtest_power_rankings.py --leagues sleeper:<id>:<season>",
        "  python scripts/backtest_power_rankings.py --from-json season.json",
        "  (JSON shape: league_id, season, playoff_week_start, playoff_teams,",
        "   rows: [{week, roster_id, matchup_id, points}, ...])",
    ]
    if tried:
        lines.append("")
        lines.append("Sources tried:")
        lines.extend(f"  • {t}" for t in tried)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Records / schedule from weekly rows
# ---------------------------------------------------------------------------


def _h2h_pairs(rows: Sequence[dict]) -> list[tuple[int, Any, str, float, str, float]]:
    """(week, matchup_id, rid_a, pts_a, rid_b, pts_b) for two-team matchups."""
    buckets: dict[tuple[int, Any], list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        mid = r.get("matchup_id")
        if mid is None:
            continue
        buckets[(int(r["week"]), mid)].append(
            (str(r["roster_id"]), float(r.get("points") or 0.0))
        )
    out = []
    for (wk, mid), teams in buckets.items():
        if len(teams) != 2:
            continue
        (a, pa), (b, pb) = teams
        out.append((wk, mid, a, pa, b, pb))
    return out


def h2h_records(rows: Sequence[dict]) -> dict[str, dict]:
    """Actual W-L-T and PF through the given rows (ties ignored in win_pct, matching production)."""
    rec: dict[str, dict] = {}

    def bucket(rid: str) -> dict:
        row = rec.get(rid)
        if row is None:
            row = {"wins": 0, "losses": 0, "ties": 0, "pf": 0.0, "games": 0}
            rec[rid] = row
        return row

    seen_pts: set[tuple[int, str]] = set()
    for r in rows:
        rid = str(r["roster_id"])
        key = (int(r["week"]), rid)
        if key in seen_pts:
            continue
        seen_pts.add(key)
        b = bucket(rid)
        b["pf"] += float(r.get("points") or 0.0)

    for _wk, _mid, a, pa, b, pb in _h2h_pairs(rows):
        ra, rb = bucket(a), bucket(b)
        ra["games"] += 1
        rb["games"] += 1
        if pa > pb:
            ra["wins"] += 1
            rb["losses"] += 1
        elif pb > pa:
            rb["wins"] += 1
            ra["losses"] += 1
        else:
            ra["ties"] += 1
            rb["ties"] += 1
    return rec


def ros_win_pct(rows_after: Sequence[dict]) -> dict[str, float]:
    rec = h2h_records(rows_after)
    out: dict[str, float] = {}
    for rid, r in rec.items():
        denom = r["wins"] + r["losses"]
        if denom <= 0:
            continue
        out[rid] = r["wins"] / denom
    return out


def final_standings_score(rows: Sequence[dict]) -> dict[str, float]:
    """Higher is a better finish: wins, then PF. Ties in W-L-T are ignored like production win%."""
    rec = h2h_records(rows)
    out: dict[str, float] = {}
    for rid, r in rec.items():
        out[rid] = r["wins"] + r["pf"] / 100_000.0
    return out


def matchups_by_week_left_right(rows: Sequence[dict]) -> dict[int, list[dict]]:
    """Production ``matchups_by_week`` shape used for ROS ease."""
    by_week: dict[int, list[dict]] = defaultdict(list)
    for wk, _mid, a, _pa, b, _pb in _h2h_pairs(rows):
        by_week[wk].append({"left": {"roster_id": a}, "right": {"roster_id": b}})
    return dict(by_week)


def remaining_pairs(
    rows: Sequence[dict], week_n: int, last_reg: int,
) -> dict[int, list[tuple[int, int]]]:
    """``_run_mc`` schedule: {week: [(rid, rid), ...]} with int roster ids."""
    by_week: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for wk, _mid, a, _pa, b, _pb in _h2h_pairs(rows):
        if int(week_n) < wk <= int(last_reg):
            try:
                by_week[wk].append((int(a), int(b)))
            except (TypeError, ValueError):
                continue
    return dict(by_week)


# ---------------------------------------------------------------------------
# Playoff % as of week N (hist scoring only — no future projections)
# ---------------------------------------------------------------------------


def playoff_pct_as_of(
    rec: dict[str, dict],
    rows_through: Sequence[dict],
    remaining: dict[int, list[tuple[int, int]]],
    *,
    playoff_teams: int,
    n_sims: int,
    seed: int,
) -> dict[str, float]:
    """Monte Carlo remaining H2H using each team's mean/std through week N."""
    if not remaining:
        return {}
    from data_building.simulate_playoff_odds import _MIN_STD, _run_mc

    pts_by_rid: dict[str, list[float]] = defaultdict(list)
    for r in rows_through:
        pts_by_rid[str(r["roster_id"])].append(float(r.get("points") or 0.0))

    teams = []
    for rid, row in rec.items():
        try:
            irid = int(rid)
        except (TypeError, ValueError):
            continue
        arr = pts_by_rid.get(rid) or [0.0]
        mean = sum(arr) / len(arr)
        if len(arr) >= 2:
            var = sum((x - mean) ** 2 for x in arr) / len(arr)
            std = max(var ** 0.5, _MIN_STD)
        else:
            std = _MIN_STD
        games = max(1, int(row["wins"] + row["losses"] + row["ties"]))
        teams.append({
            "roster_id": irid,
            "name": f"Team {rid}",
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "ties": int(row["ties"]),
            "pf": float(row["pf"]),
            "avg": float(mean if mean > 0 else row["pf"] / games),
            "std": float(std),
        })
    if len(teams) < 4 or not remaining:
        return {}
    result = _run_mc(teams, remaining, {}, int(playoff_teams), int(n_sims), int(seed))
    return {str(r["roster_id"]): float(r.get("playoff_pct") or 0.0) for r in result}


# ---------------------------------------------------------------------------
# Production ranking at week N
# ---------------------------------------------------------------------------


def _install_light_stubs() -> None:
    """Skip narrative/grade helpers; we only need the PowerScore math."""
    import dashboard_services.ai.context_builders as cb

    cb.summarize_roster_players = lambda **_k: []
    cb.detect_team_direction = lambda *_a, **_k: "balanced"
    cb.group_position_strength = lambda _x: {}
    cb.calculate_roster_grade = lambda *_a, **_k: {"win_window": "balanced"}
    cb.build_model_value_lookup = lambda tbl, is_sf=False, **_k: {
        str(r.get("player_id") or r.get("id") or ""): r for r in (tbl or [])
    }


def _df_weekly(rows: Sequence[dict]):
    import pandas as pd

    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    df["roster_id"] = df["roster_id"].astype(str)
    df["week"] = df["week"].astype(int)
    df["points"] = df["points"].astype(float)
    df["finalized"] = True
    return df


def rank_week(
    league: LeagueSeason,
    week_n: int,
    *,
    n_sims: int = 2000,
    include_playoff: bool = True,
) -> dict[str, dict]:
    """Return {roster_id: team_dict} from the live context builder + blend.

    ``starter_value`` is 0 for every roster (no historical week-N values).
    """
    import dashboard_services.ai.context_builders as cb
    from dashboard_services.power_score import season_phase_from_progress

    _install_light_stubs()

    through = league.rows_through(week_n)
    rec = h2h_records(through)
    full_schedule_rows = [r for r in league.rows if int(r["week"]) <= league.last_reg_week]
    mbw = matchups_by_week_left_right(full_schedule_rows)

    playoff_rows: list[dict] = []
    if include_playoff:
        remaining = remaining_pairs(full_schedule_rows, week_n, league.last_reg_week)
        seed = (int(league.season) * 10_000 + int(week_n)) % (2 ** 31)
        try:
            pct = playoff_pct_as_of(
                rec, through, remaining,
                playoff_teams=league.playoff_teams,
                n_sims=n_sims,
                seed=seed,
            )
            playoff_rows = [
                {"roster_id": rid, "playoff_pct": p} for rid, p in pct.items()
            ]
        except Exception:
            playoff_rows = []

    rosters = []
    standings_map = {}
    roster_map = {}
    for rid in league.roster_ids:
        r = rec.get(rid) or {"wins": 0, "losses": 0, "ties": 0, "pf": 0.0}
        # Integer PF + hundredths in fpts_decimal, matching Sleeper settings.
        pf = float(r["pf"])
        fpts = int(math.floor(pf))
        fpts_decimal = int(round((pf - fpts) * 100))
        rosters.append({
            "roster_id": rid,
            "players": [],
            "settings": {
                "wins": int(r["wins"]),
                "losses": int(r["losses"]),
                "ties": int(r["ties"]),
                "fpts": fpts,
                "fpts_decimal": fpts_decimal,
            },
        })
        standings_map[rid] = {"PF": pf}
        roster_map[rid] = f"Team {rid}"

    ctx = {
        "rosters": rosters,
        "standings_map": standings_map,
        "roster_map": roster_map,
        "model_value_table": [],
        "picks_by_roster": {},
        "df_weekly": _df_weekly(through),
        "matchups_by_week": mbw,
        "current_week": int(week_n),
        "current_season": int(league.season),
        "league_type": "1qb",
        "league_settings": {
            "playoff_teams": int(league.playoff_teams),
            "playoff_week_start": int(league.playoff_week_start),
        },
        "playoff_odds": playoff_rows,
        "roster_positions": [],
    }
    built = cb.build_power_rankings_context(ctx)
    teams = {str(t["roster_id"]): t for t in (built.get("teams") or [])}
    games = max((t["wins"] + t["losses"] for t in teams.values()), default=week_n)
    phase = season_phase_from_progress(games_played=games, current_week=week_n)
    for t in teams.values():
        t["_phase"] = phase
        t["_all_play"] = t.get("all_play_pct")
        t["_ppg"] = t.get("avg")
        t["_win_pct"] = t.get("win_pct")
        t["_blend"] = t.get("power_score")
    return teams


# ---------------------------------------------------------------------------
# Evaluate one league-season
# ---------------------------------------------------------------------------


@dataclass
class SnapshotRow:
    season: int
    league_id: str
    week: int
    phase: str
    method: str
    spearman_final: Optional[float]
    spearman_ros: Optional[float]
    n_teams: int


def evaluate_league(
    league: LeagueSeason,
    *,
    min_week: int = DEFAULT_MIN_WEEK,
    n_sims: int = 2000,
    include_playoff: bool = True,
) -> tuple[list[SnapshotRow], list[dict]]:
    """Score every week N. Returns (method rows, per-team component rows)."""
    weeks = [w for w in league.weeks() if w >= min_week]
    last = league.last_reg_week
    final_rows = [r for r in league.rows if int(r["week"]) <= last]
    final_score = final_standings_score(final_rows)
    method_rows: list[SnapshotRow] = []
    component_rows: list[dict] = []

    for week_n in weeks:
        ranked = rank_week(
            league, week_n, n_sims=n_sims, include_playoff=include_playoff,
        )
        if len(ranked) < 4:
            continue
        rids = [rid for rid in league.roster_ids if rid in ranked]
        if len(rids) < 4:
            continue
        y_final = [final_score.get(rid, 0.0) for rid in rids]
        ros = ros_win_pct(league.rows_after(week_n))
        # ROS needs at least a couple of remaining H2H results.
        ros_ok = sum(1 for rid in rids if rid in ros) >= 4
        y_ros = [ros.get(rid, 0.0) for rid in rids] if ros_ok else None

        methods = {
            "blend": [float(ranked[rid].get("_blend") or 0.0) for rid in rids],
            "all_play": [float(ranked[rid].get("_all_play") or 0.0) for rid in rids],
            "ppg": [float(ranked[rid].get("_ppg") or 0.0) for rid in rids],
            "win_pct": [float(ranked[rid].get("_win_pct") or 0.0) for rid in rids],
        }
        phase = str(next(iter(ranked.values())).get("_phase") or "mid")
        for method, xs in methods.items():
            method_rows.append(SnapshotRow(
                season=league.season,
                league_id=league.league_id,
                week=week_n,
                phase=phase,
                method=method,
                spearman_final=spearman(xs, y_final),
                spearman_ros=spearman(xs, y_ros) if y_ros is not None else None,
                n_teams=len(rids),
            ))

        for rid in rids:
            comps = dict(ranked[rid].get("power_components") or {})
            row = {
                "season": league.season,
                "league_id": league.league_id,
                "week": week_n,
                "phase": phase,
                "roster_id": rid,
            }
            for k in COMPONENTS:
                if k in comps and comps[k] is not None:
                    row[k] = float(comps[k])
            component_rows.append(row)

    return method_rows, component_rows


# ---------------------------------------------------------------------------
# Aggregation / report
# ---------------------------------------------------------------------------


def _group_mean(
    rows: Sequence[SnapshotRow],
    method: str,
    attr: str,
    *,
    phase: Optional[str] = None,
    season: Optional[int] = None,
) -> Optional[float]:
    xs = []
    for r in rows:
        if r.method != method:
            continue
        if phase is not None and r.phase != phase:
            continue
        if season is not None and r.season != season:
            continue
        xs.append(getattr(r, attr))
    return _mean(xs)


def pairwise_corr(component_rows: Sequence[dict], keys: Sequence[str]) -> dict[tuple[str, str], Optional[float]]:
    out: dict[tuple[str, str], Optional[float]] = {}
    for i, a in enumerate(keys):
        for b in keys[i:]:
            xs, ys = [], []
            for row in component_rows:
                if a in row and b in row:
                    xs.append(float(row[a]))
                    ys.append(float(row[b]))
            out[(a, b)] = spearman(xs, ys) if len(xs) >= 8 else None
            if a != b:
                out[(b, a)] = out[(a, b)]
    return out


def component_target_spearman(
    component_rows: Sequence[dict],
    method_rows: Sequence[SnapshotRow],
    leagues: Sequence[LeagueSeason],
) -> dict[str, dict[str, Optional[float]]]:
    """Each live component's Spearman vs the two targets (diagnostic, not a baseline)."""
    by_key = {(r.league_id, r.season): r for r in leagues}
    # Rebuild targets per snapshot from stored component rows' league/week.
    buckets: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in component_rows:
        buckets[(str(row["league_id"]), int(row["season"]), int(row["week"]))].append(row)

    acc: dict[str, dict[str, list]] = {k: {"final": [], "ros": []} for k in COMPONENTS}
    for (lid, season, week), rows in buckets.items():
        league = by_key.get((lid, season))
        if league is None:
            continue
        last = league.last_reg_week
        y_final = final_standings_score([r for r in league.rows if int(r["week"]) <= last])
        y_ros = ros_win_pct(league.rows_after(week))
        rids = [r["roster_id"] for r in rows]
        for key in COMPONENTS:
            if any(key not in r for r in rows):
                continue
            xs = [float(r[key]) for r in rows]
            ys_f = [y_final.get(rid, 0.0) for rid in rids]
            acc[key]["final"].append(spearman(xs, ys_f))
            if sum(1 for rid in rids if rid in y_ros) >= 4:
                ys_r = [y_ros.get(rid, 0.0) for rid in rids]
                acc[key]["ros"].append(spearman(xs, ys_r))
    return {
        k: {"final": _mean(v["final"]), "ros": _mean(v["ros"])}
        for k, v in acc.items()
    }


def headline(rows: Sequence[SnapshotRow]) -> tuple[str, dict]:
    blend_ros = _group_mean(rows, "blend", "spearman_ros")
    ap_ros = _group_mean(rows, "all_play", "spearman_ros")
    blend_fin = _group_mean(rows, "blend", "spearman_final")
    ap_fin = _group_mean(rows, "all_play", "spearman_final")
    stats = {
        "blend_ros": blend_ros, "all_play_ros": ap_ros,
        "blend_final": blend_fin, "all_play_final": ap_fin,
        "ros_delta": None if blend_ros is None or ap_ros is None else blend_ros - ap_ros,
        "final_delta": None if blend_fin is None or ap_fin is None else blend_fin - ap_fin,
    }
    # "Clearly beat" = strictly better on ROS (the honest target) by ≥ 0.02,
    # and not worse on final standings. A dead heat is a loss for the blend.
    clear = 0.02
    if blend_ros is None or ap_ros is None:
        msg = "Not enough ROS snapshots to compare the blend to all-play%."
    elif blend_ros >= ap_ros + clear and (blend_fin is None or ap_fin is None or blend_fin >= ap_fin - 0.01):
        msg = (
            f"The full blend beats all-play% out of sample on rest-of-season "
            f"win rate ({blend_ros:+.3f} vs {ap_ros:+.3f}, Δ={blend_ros - ap_ros:+.3f})."
        )
    else:
        delta = blend_ros - ap_ros
        msg = (
            f"The full blend does not clearly beat all-play% out of sample. "
            f"ROS Spearman: blend {blend_ros:+.3f} vs all-play {ap_ros:+.3f} "
            f"(Δ={delta:+.3f}). All-play win % is earning its keep; the extra "
            f"hand-tuned terms are not."
        )
    return msg, stats


def recommend(msg: str, pair: dict, comp_vs: dict) -> str:
    redundant = []
    for a, b in (("pf", "record"), ("record", "playoff"), ("pf", "playoff"),
                 ("sos", "ros"), ("record", "momentum")):
        rho = pair.get((a, b))
        if rho is not None and abs(rho) >= 0.75:
            redundant.append(f"{a}/{b} ρ={rho:+.2f}")
    weak = []
    for k, vals in comp_vs.items():
        ros = vals.get("ros")
        if ros is not None and ros < 0.05:
            weak.append(f"{k} ({ros:+.2f} vs ROS)")
    lines = [msg]
    if "does not clearly beat" in msg:
        lines.append(
            "Recommendation: do not keep the current PHASE_WEIGHTS as a claim of "
            "added accuracy. Prefer all-play% (optionally with PPG as a tie-break) "
            "until the extra terms are re-fit out of sample."
        )
        if redundant:
            lines.append("Highly redundant pairs to drop or decorrelate: " + ", ".join(redundant) + ".")
        if weak:
            lines.append("Components with little ROS signal: " + ", ".join(weak) + ".")
        lines.append(
            "If you keep a blend, re-fit weights on ROS Spearman (or a proper "
            "walk-forward loss) rather than hand-tuning; freeze the fit and "
            "re-evaluate on a later season."
        )
    else:
        lines.append(
            "Recommendation: keep a blend, but re-check whether every positive-"
            "weight term is pulling its weight. Drop or shrink any component "
            "that is both redundant (ρ≥0.75 with record/pf) and weak vs ROS."
        )
        if redundant:
            lines.append("Redundant pairs: " + ", ".join(redundant) + ".")
        if weak:
            lines.append("Weak vs ROS: " + ", ".join(weak) + ".")
    lines.append(
        "starter_value was neutralized (0 for all teams) because week-N roster "
        "values are not in the historical store. Re-run when snapshot history exists "
        "before judging the value term."
    )
    return "\n".join(lines)


def format_report(
    leagues: Sequence[LeagueSeason],
    rows: Sequence[SnapshotRow],
    component_rows: Sequence[dict],
    *,
    limitations: Sequence[str] = (),
) -> str:
    n_snap = len({(r.league_id, r.season, r.week) for r in rows})
    seasons = sorted({lg.season for lg in leagues})
    lines: list[str] = []
    lines.append("PowerScore blend backtest")
    lines.append("=" * 72)
    lines.append(
        f"Completed seasons: {len(leagues)} league-seasons "
        f"({min(seasons) if seasons else '?'}–{max(seasons) if seasons else '?'}); "
        f"{n_snap} week-snapshots; methods {', '.join(METHODS)}."
    )
    for lg in leagues:
        lines.append(
            f"  • {lg.season} {lg.name} ({lg.platform}:{lg.league_id}) "
            f"{lg.n_teams}tm  reg weeks 1–{lg.last_reg_week}  "
            f"playoff spots {lg.playoff_teams}"
        )
    lines.append("")

    def table(attr: str, title: str) -> None:
        lines.append(title)
        hdr = f"  {'method':<10} {'overall':>8} {'early':>8} {'mid':>8} {'late':>8}"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for method in METHODS:
            overall = _group_mean(rows, method, attr)
            cells = [overall] + [_group_mean(rows, method, attr, phase=ph) for ph in PHASES]
            lines.append(
                f"  {method:<10} " + " ".join(_fmt(c) for c in cells)
            )
        lines.append("")

    table("spearman_final", "Mean Spearman vs final regular-season standings")
    table("spearman_ros", "Mean Spearman vs rest-of-season H2H win rate (honest target)")

    # Per-season ROS overall for blend vs all-play
    lines.append("ROS Spearman by season (blend vs all-play)")
    for season in seasons:
        b = _group_mean(rows, "blend", "spearman_ros", season=season)
        a = _group_mean(rows, "all_play", "spearman_ros", season=season)
        n = len({(r.league_id, r.week) for r in rows if r.season == season and r.method == "blend"})
        lines.append(f"  {season}: blend {_fmt(b)}   all-play {_fmt(a)}   snapshots={n}")
    lines.append("")

    pair = pairwise_corr(component_rows, COMPONENTS)
    lines.append("Pairwise Spearman of live components (pooled team-weeks)")
    hdr = "          " + "".join(f"{k:>10}" for k in COMPONENTS)
    lines.append(hdr)
    for a in COMPONENTS:
        row = f"  {a:<8}" + "".join(_fmt(pair.get((a, b)), digits=2, width=10) for b in COMPONENTS)
        lines.append(row)
    lines.append("")

    by_lg = {(lg.league_id, lg.season): lg for lg in leagues}
    comp_vs = component_target_spearman(component_rows, rows, list(by_lg.values()))
    lines.append("Each component alone vs the two targets (diagnostic)")
    lines.append(f"  {'component':<12} {'vs final':>10} {'vs ROS':>10}")
    for k in COMPONENTS:
        lines.append(
            f"  {k:<12} {_fmt(comp_vs[k]['final']):>10} {_fmt(comp_vs[k]['ros']):>10}"
        )
    lines.append("")

    msg, _stats = headline(rows)
    lines.append("HEADLINE")
    lines.append(msg)
    lines.append("")
    lines.append("RECOMMENDATION")
    lines.append(recommend(msg, pair, comp_vs))
    if limitations:
        lines.append("")
        lines.append("LIMITATIONS")
        lines.extend(f"  • {x}" for x in limitations)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_json_path(path: Path) -> list[LeagueSeason]:
    text = path.read_text(encoding="utf-8")
    blobs: list[dict]
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            blobs = [x for x in parsed if isinstance(x, dict)]
        elif isinstance(parsed, dict):
            blobs = [parsed]
        else:
            blobs = []
    except json.JSONDecodeError:
        blobs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            blobs.append(json.loads(line))
    out = []
    for blob in blobs:
        lg = LeagueSeason.from_json(blob)
        if lg.rows:
            out.append(lg)
    return out


def _sleeper_get(path: str):
    from dashboard_services.api import fetch_json
    return fetch_json(path)


def fetch_sleeper_league_season(
    league_id: str,
    season: Optional[int] = None,
) -> Optional[LeagueSeason]:
    meta = _sleeper_get(f"/league/{league_id}") or {}
    if not meta:
        return None
    status = str(meta.get("status") or "")
    lg_season = int(meta.get("season") or season or 0)
    if season is not None and lg_season != int(season):
        # Caller asked for a specific year; this id is a different season.
        return None
    settings = meta.get("settings") or {}
    playoff_week_start = int(settings.get("playoff_week_start") or 15)
    playoff_teams = int(settings.get("playoff_teams") or 6)
    n_teams = int(meta.get("total_rosters") or 0)
    last_reg = playoff_week_start - 1
    if last_reg < DEFAULT_MIN_WEEK + 2:
        return None
    # Completed seasons only. "complete" is the Sleeper flag; also accept
    # in_season ids whose season is already in the past relative to NFL state.
    if status not in ("complete", "pre_draft", "drafting"):
        # in_season for a past year still has all scores; current year does not.
        try:
            state = _sleeper_get("/state/nfl") or {}
            live = int(state.get("season") or 0)
        except Exception:
            live = 0
        if live and lg_season >= live and status != "complete":
            return None
    elif status in ("pre_draft", "drafting"):
        return None

    weeks = list(range(1, last_reg + 1))

    def _week(w: int) -> list[dict]:
        try:
            data = _sleeper_get(f"/league/{league_id}/matchups/{w}") or []
        except Exception:
            return []
        rows = []
        for m in data:
            if not isinstance(m, dict):
                continue
            rid = m.get("roster_id")
            if rid is None:
                continue
            pts = m.get("points")
            if pts is None:
                continue
            rows.append({
                "week": w,
                "roster_id": str(rid),
                "matchup_id": m.get("matchup_id"),
                "points": float(pts or 0.0),
            })
        return rows

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(8, len(weeks))) as pool:
        futs = {pool.submit(_week, w): w for w in weeks}
        for fut in as_completed(futs):
            rows.extend(fut.result())

    # Require a real regular season: most weeks have a full (or near-full) slate
    # with non-zero scoring.
    by_week: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_week[int(r["week"])].append(r)
    scored_weeks = []
    for w, wr in by_week.items():
        nonzero = sum(1 for r in wr if float(r["points"]) > 0)
        if n_teams and nonzero >= max(4, n_teams - 2):
            scored_weeks.append(w)
    if len(scored_weeks) < max(8, DEFAULT_MIN_WEEK + 3):
        return None
    if status != "complete":
        # Past-season leftover: treat as complete only if the last two reg weeks scored.
        if not (last_reg in scored_weeks and (last_reg - 1) in scored_weeks):
            return None

    return LeagueSeason(
        platform="sleeper",
        league_id=str(league_id),
        season=lg_season,
        name=str(meta.get("name") or league_id),
        n_teams=n_teams or len({r["roster_id"] for r in rows}),
        playoff_week_start=playoff_week_start,
        playoff_teams=playoff_teams,
        rows=rows,
    )


def walk_previous_ids(league_id: str, limit: int = 8) -> list[str]:
    ids = []
    cursor = str(league_id)
    seen = set()
    for _ in range(limit):
        if not cursor or cursor in seen or cursor in ("0", "None"):
            break
        seen.add(cursor)
        ids.append(cursor)
        try:
            meta = _sleeper_get(f"/league/{cursor}") or {}
        except Exception:
            break
        cursor = str(meta.get("previous_league_id") or "")
    return ids


def discover_sleeper_completed(
    seeds: Sequence[str],
    *,
    max_leagues: int,
    bfs_owners: bool = True,
) -> list[LeagueSeason]:
    """Walk seed ids → previous seasons → co-owner leagues; fetch completed years."""
    queue: list[str] = []
    seen: set[str] = set()
    for s in seeds:
        for lid in walk_previous_ids(str(s)):
            if lid not in seen:
                seen.add(lid)
                queue.append(lid)

    found: list[LeagueSeason] = []
    owner_frontier: list[str] = []

    def consider(lid: str) -> None:
        if len(found) >= max_leagues:
            return
        try:
            lg = fetch_sleeper_league_season(lid)
        except Exception as exc:
            print(f"  [skip] {lid}: {exc}", file=sys.stderr)
            return
        if lg is None:
            return
        key = (lg.league_id, lg.season)
        if any((x.league_id, x.season) == key for x in found):
            return
        found.append(lg)
        print(
            f"  [ok] {lg.season} {lg.name} {lg.n_teams}tm "
            f"weeks={len(lg.weeks())} id={lg.league_id}",
            file=sys.stderr,
        )

    for lid in list(queue):
        consider(lid)
        if len(found) >= max_leagues:
            return found
        if bfs_owners:
            try:
                rosters = _sleeper_get(f"/league/{lid}/rosters") or []
            except Exception:
                rosters = []
            for r in rosters:
                oid = r.get("owner_id") if isinstance(r, dict) else None
                if oid:
                    owner_frontier.append(str(oid))

    if bfs_owners and len(found) < max_leagues:
        seasons_wanted = sorted({lg.season for lg in found} | {2024, 2025})
        for uid in dict.fromkeys(owner_frontier):
            if len(found) >= max_leagues:
                break
            for yr in seasons_wanted:
                try:
                    lgs = _sleeper_get(f"/user/{uid}/leagues/nfl/{yr}") or []
                except Exception:
                    continue
                time.sleep(0.05)
                for meta in lgs:
                    if not isinstance(meta, dict):
                        continue
                    lid = str(meta.get("league_id") or "")
                    if not lid or lid in seen:
                        continue
                    seen.add(lid)
                    n_teams = int(meta.get("total_rosters") or 0)
                    if n_teams and n_teams < 8:
                        continue
                    consider(lid)
                    if len(found) >= max_leagues:
                        break
    return found


def try_db_league_ids() -> list[str]:
    ids: list[str] = []
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            for sql in (
                "SELECT DISTINCT league_id FROM user_leagues",
                "SELECT DISTINCT league_id FROM trade_intel_leagues",
                "SELECT DISTINCT league_id FROM playoff_odds",
                "SELECT DISTINCT league_id FROM power_rank_history",
            ):
                try:
                    rows = conn.execute(sql).fetchall()
                except Exception:
                    continue
                for r in rows:
                    if isinstance(r, dict):
                        val = r.get("league_id")
                    else:
                        val = r[0]
                    if val:
                        ids.append(str(val))
    except Exception:
        pass
    return list(dict.fromkeys(ids))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_league_specs(specs: Sequence[str]) -> list[tuple[str, str, Optional[int]]]:
    out = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) == 3:
            out.append((parts[0], parts[1], int(parts[2])))
        elif len(parts) == 2:
            out.append((parts[0], parts[1], None))
        else:
            raise SystemExit(f"bad --leagues entry '{spec}', expected platform:league_id[:season]")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leagues", nargs="+", default=[],
                    help="platform:league_id[:season] (Sleeper completed seasons)")
    ap.add_argument("--from-json", nargs="+", default=[],
                    help="JSON/JSONL files of LeagueSeason objects")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Do not hit Sleeper; only JSON / DB ids already fetched")
    ap.add_argument("--max-leagues", type=int, default=12)
    ap.add_argument("--min-week", type=int, default=DEFAULT_MIN_WEEK)
    ap.add_argument("--sims", type=int, default=2000, help="Playoff MC iterations per snapshot")
    ap.add_argument("--no-playoff", action="store_true",
                    help="Leave playoff_pct missing (weights redistribute)")
    ap.add_argument("--no-owner-bfs", action="store_true")
    ap.add_argument("--scratch", default="",
                    help="Directory for fetched JSON (default: /tmp/power-rankings-backtest)")
    ap.add_argument("--json-out", default="", help="Write the text report to this path too")
    args = ap.parse_args(list(argv) if argv is not None else None)

    scratch = Path(args.scratch or os.environ.get("POWER_RANK_BACKTEST_SCRATCH") or "/tmp/power-rankings-backtest")
    scratch.mkdir(parents=True, exist_ok=True)

    tried: list[str] = []
    leagues: list[LeagueSeason] = []

    for path_s in args.from_json:
        path = Path(path_s)
        tried.append(f"json {path}")
        if path.is_file():
            leagues.extend(load_json_path(path))

    if args.leagues and not args.no_fetch:
        for platform, lid, season in parse_league_specs(args.leagues):
            tried.append(f"cli {platform}:{lid}:{season}")
            if platform != "sleeper":
                print(f"  [skip] only sleeper fetch is implemented ({platform}:{lid})", file=sys.stderr)
                continue
            lg = fetch_sleeper_league_season(lid, season)
            if lg:
                leagues.append(lg)

    db_ids = try_db_league_ids()
    tried.append(f"postgres league ids ({len(db_ids)} unique)")

    seed_ids = list(dict.fromkeys(
        [lid for _, lid, _ in parse_league_specs(args.leagues)]
        + db_ids
        + list(REPO_SEED_LEAGUE_IDS)
    ))

    if not args.no_fetch and len(leagues) < args.max_leagues:
        tried.append(
            f"sleeper API from {len(seed_ids)} seed league ids "
            f"(repo tests + CLI + DB), previous_league_id walk"
            + ("" if args.no_owner_bfs else " + owner BFS")
        )
        print("Discovering completed Sleeper seasons…", file=sys.stderr)
        found = discover_sleeper_completed(
            seed_ids,
            max_leagues=args.max_leagues,
            bfs_owners=not args.no_owner_bfs,
        )
        have = {(lg.league_id, lg.season) for lg in leagues}
        for lg in found:
            if (lg.league_id, lg.season) not in have:
                leagues.append(lg)
                have.add((lg.league_id, lg.season))

    # Dedupe
    uniq: dict[tuple[str, int], LeagueSeason] = {}
    for lg in leagues:
        uniq[(lg.league_id, lg.season)] = lg
    leagues = list(uniq.values())
    leagues.sort(key=lambda lg: (lg.season, lg.name, lg.league_id))

    if not leagues:
        print(missing_data_report(tried=tried))
        return 2

    for lg in leagues:
        dest = scratch / f"{lg.platform}_{lg.league_id}_{lg.season}.json"
        dest.write_text(json.dumps(lg.to_json()), encoding="utf-8")

    limitations = [
        "starter_value held at 0 for every team (no week-N roster/value snapshots in repo or DB).",
        "playoff_pct is a hist-only Monte Carlo of the remaining published schedule "
        f"({args.sims} sims); live board also blends in that week's Sleeper player projections.",
        "Final-standings target is regular-season W-L then PF, not playoff-bracket finish.",
        "Sample is public Sleeper leagues reachable from repo seed IDs, not a census of all leagues.",
    ]

    print(f"Evaluating {len(leagues)} league-season(s)…", file=sys.stderr)
    all_rows: list[SnapshotRow] = []
    all_comps: list[dict] = []
    for lg in leagues:
        print(f"  {lg.season} {lg.name}…", file=sys.stderr)
        rows, comps = evaluate_league(
            lg,
            min_week=args.min_week,
            n_sims=args.sims,
            include_playoff=not args.no_playoff,
        )
        all_rows.extend(rows)
        all_comps.extend(comps)

    if not all_rows:
        print(missing_data_report(tried=tried + ["fetched leagues had too few scored weeks"]))
        return 2

    report = format_report(leagues, all_rows, all_comps, limitations=limitations)
    print(report)
    (scratch / "report.txt").write_text(report + "\n", encoding="utf-8")
    payload = {
        "leagues": [lg.to_json() | {"rows": f"{len(lg.rows)} rows"} for lg in leagues],
        "snapshots": [r.__dict__ for r in all_rows],
        "headline": headline(all_rows)[0],
    }
    (scratch / "summary.json").write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
