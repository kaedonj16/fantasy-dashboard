"""Build a real-draft roster-construction benchmark from the production DB.

Run this from a Render shell, where ``DATABASE_URL`` is already configured:

    python scripts/audit_draft_realism.py --output artifacts/draft-realism.md

The audit is read-only.  It groups completed drafts into redraft/dynasty and
1QB/Superflex cohorts, then reports the real manager behavior that should guide
Draft Room and CPU tuning (QB2/QB3 and TE2/TE3 timing, final position counts,
K/DEF timing, repeated-position picks, and position share by phase).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")
NFL_TEAM_CODES = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
    # Historical Sleeper defense identifiers can remain in older draft rows.
    "OAK", "SD", "STL",
})


@dataclass
class CohortSummary:
    draft_type: str
    format: str
    drafts: int
    teams: int
    picks: int
    resolved_pct: float
    median_rounds: float
    position_counts: dict[str, float]
    qb_rounds: dict[str, float | None]
    te_rounds: dict[str, float | None]
    k_first_round: float | None
    def_first_round: float | None
    consecutive_position_pct: float
    phase_shares: dict[str, dict[str, float]]


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.median(values), 2) if values else None


def load_position_map() -> dict[str, str]:
    """Load Sleeper positions without booting Flask or querying another API."""
    candidates = (
        REPO_ROOT / "cache" / "players_index.json",
        REPO_ROOT / "cache" / "players_index_relevant.json",
    )
    result: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for player_id, player in (json.load(handle) or {}).items():
                pos = str(player.get("pos") or player.get("position") or "").upper()
                if pos == "PK":
                    pos = "K"
                elif pos in {"DST", "D/ST"}:
                    pos = "DEF"
                if pos in SKILL_POSITIONS:
                    result[str(player_id)] = pos
    # Sleeper represents team defenses by their abbreviation (for example
    # player_id="BAL"); those entries are not part of players_index.json.
    for team in NFL_TEAM_CODES:
        result.setdefault(team, "DEF")
    return result


def summarize_cohort(drafts: list[dict], picks: list[dict], positions: dict[str, str]) -> CohortSummary:
    """Summarize one homogeneous draft-type/format cohort."""
    draft_ids = {str(d["draft_id"]) for d in drafts}
    draft_meta = {str(d["draft_id"]): d for d in drafts}
    rosters: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    resolved = 0
    relevant_picks = 0
    for pick in picks:
        draft_id = str(pick["draft_id"])
        if draft_id not in draft_ids:
            continue
        relevant_picks += 1
        pos = positions.get(str(pick["player_id"]))
        if not pos:
            continue
        resolved += 1
        pick_no = int(pick.get("pick_no") or 0)
        num_teams = int(draft_meta[draft_id].get("num_teams") or 12)
        round_no = int(pick.get("round") or ((pick_no - 1) // num_teams + 1))
        roster_id = str(pick.get("roster_id") or pick.get("pick_in_round") or "?")
        rosters[(draft_id, roster_id)].append((pick_no, round_no, pos))

    final_counts: dict[str, list[int]] = defaultdict(list)
    nth_rounds: dict[str, dict[int, list[int]]] = {
        "QB": defaultdict(list), "TE": defaultdict(list)
    }
    first_rounds: dict[str, list[int]] = {"K": [], "DEF": []}
    phases: dict[str, Counter] = defaultdict(Counter)
    consecutive = transitions = 0
    for roster in rosters.values():
        roster.sort()
        counts = Counter(pos for _, _, pos in roster)
        for pos in SKILL_POSITIONS:
            final_counts[pos].append(counts[pos])
        for pos in ("QB", "TE"):
            rounds = [rnd for _, rnd, picked_pos in roster if picked_pos == pos]
            for nth in (1, 2, 3):
                if len(rounds) >= nth:
                    nth_rounds[pos][nth].append(rounds[nth - 1])
        for pos in ("K", "DEF"):
            rounds = [rnd for _, rnd, picked_pos in roster if picked_pos == pos]
            if rounds:
                first_rounds[pos].append(rounds[0])
        for index, (_, rnd, pos) in enumerate(roster):
            phase = "early (1-4)" if rnd <= 4 else "middle (5-9)" if rnd <= 9 else "late (10+)"
            phases[phase][pos] += 1
            if index:
                transitions += 1
                consecutive += int(pos == roster[index - 1][2])

    dtype = str(drafts[0]["draft_type"])
    return CohortSummary(
        draft_type="dynasty " + dtype if dtype in {"startup", "rookie"} else dtype,
        format="Superflex" if drafts[0].get("is_superflex") else "1QB",
        drafts=len(drafts), teams=len(rosters), picks=relevant_picks,
        resolved_pct=round(100 * resolved / max(1, relevant_picks), 1),
        median_rounds=_median(float(d.get("rounds") or 0) for d in drafts) or 0,
        position_counts={p: _median(v) or 0 for p, v in final_counts.items()},
        qb_rounds={f"QB{n}": _median(nth_rounds["QB"][n]) for n in (1, 2, 3)},
        te_rounds={f"TE{n}": _median(nth_rounds["TE"][n]) for n in (1, 2, 3)},
        k_first_round=_median(first_rounds["K"]), def_first_round=_median(first_rounds["DEF"]),
        consecutive_position_pct=round(100 * consecutive / max(1, transitions), 1),
        phase_shares={phase: {p: round(100 * count / max(1, sum(counter.values())), 1)
                              for p, count in counter.items()}
                      for phase, counter in phases.items()},
    )


def render_markdown(summaries: list[CohortSummary], filters: str) -> str:
    lines = ["# Real Draft Roster-Construction Audit", "", filters, "",
             "> Read-only production-DB benchmark. Medians describe what real managers did; "
             "they are calibration evidence, not hard drafting rules.", ""]
    for summary in summaries:
        lines += [f"## {summary.draft_type.title()} — {summary.format}", "",
                  f"- **Sample:** {summary.drafts:,} drafts, {summary.teams:,} teams, "
                  f"{summary.picks:,} picks ({summary.resolved_pct:.1f}% position-resolved)",
                  f"- **Draft length:** median {summary.median_rounds:g} rounds",
                  f"- **Repeated-position picks:** {summary.consecutive_position_pct:.1f}% of transitions",
                  "", "| Metric | Median round |", "|---|---:|"]
        timing = {**summary.qb_rounds, **summary.te_rounds,
                  "First K": summary.k_first_round, "First DEF": summary.def_first_round}
        lines += [f"| {label} | {'—' if value is None else f'{value:g}'} |"
                  for label, value in timing.items()]
        lines += ["", "| Position | Median final count |", "|---|---:|"]
        lines += [f"| {pos} | {summary.position_counts.get(pos, 0):g} |" for pos in SKILL_POSITIONS]
        lines += ["", "### Pick share by draft phase", "", "| Phase | QB | RB | WR | TE | K | DEF |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for phase in ("early (1-4)", "middle (5-9)", "late (10+)"):
            share = summary.phase_shares.get(phase, {})
            lines.append("| " + phase + " | " + " | ".join(f"{share.get(p, 0):.1f}%" for p in SKILL_POSITIONS) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _draft_filters(seasons: list[int] | None, draft_types: list[str], alias: str = "") -> tuple[str, list[object]]:
    """Return the cohort WHERE clause.

    Do not filter on ``status`` here.  The crawler only persists completed
    drafts, while older/legacy rows can legitimately have a NULL status.  The
    previous status predicate accidentally hid those otherwise valid rows.
    """
    prefix = f"{alias}." if alias else ""
    clauses = [f"{prefix}draft_type = ANY(%s)"]
    params: list[object] = [draft_types]
    if seasons:
        clauses.append(f"{prefix}season = ANY(%s)")
        params.append(seasons)
    return " AND ".join(clauses), params


def fetch_rows(seasons: list[int] | None, draft_types: list[str],
               max_drafts: int = 500) -> tuple[list[dict], list[dict]]:
    """Fetch a bounded, recent sample per cohort and its picks.

    Production contains a large historical corpus. Pulling every pick over the
    Render DB connection can take minutes and previously gave no indication
    that work was happening. A 500-draft sample per type/format is ample for
    stable medians while keeping an interactive audit quick. Pass 0 to opt into
    the full corpus.
    """
    from dashboard_services.db import get_conn

    where, params = _draft_filters(seasons, draft_types)
    with get_conn(autocommit=True) as conn:
        if max_drafts > 0:
            drafts = list(conn.execute(
                "WITH ranked AS ("
                " SELECT draft_id, season, draft_type, num_teams, is_superflex, rounds,"
                " ROW_NUMBER() OVER (PARTITION BY draft_type, is_superflex "
                " ORDER BY season DESC, crawled_at DESC, draft_id) AS cohort_rank"
                f" FROM draft_adp_drafts WHERE {where}"
                ") SELECT draft_id, season, draft_type, num_teams, is_superflex, rounds "
                "FROM ranked WHERE cohort_rank <= %s", [*params, max_drafts]).fetchall())
        else:
            drafts = list(conn.execute(
                f"SELECT draft_id, season, draft_type, num_teams, is_superflex, rounds "
                f"FROM draft_adp_drafts WHERE {where}", params).fetchall())
        if not drafts:
            return [], []
        draft_ids = [str(draft["draft_id"]) for draft in drafts]
        picks = list(conn.execute(
            "SELECT p.draft_id, p.player_id, p.pick_no, p.round, p.pick_in_round, p.roster_id "
            "FROM draft_adp_picks p WHERE p.draft_id = ANY(%s) "
            "ORDER BY p.draft_id, p.pick_no", (draft_ids,)).fetchall())
    return drafts, picks


def fetch_inventory() -> list[dict]:
    """Describe available cohorts so an empty filter is immediately actionable."""
    from dashboard_services.db import get_conn

    with get_conn(autocommit=True) as conn:
        return list(conn.execute(
            "SELECT season, draft_type, is_superflex, COUNT(*) AS drafts "
            "FROM draft_adp_drafts GROUP BY season, draft_type, is_superflex "
            "ORDER BY season DESC, draft_type, is_superflex"
        ).fetchall())


def render_inventory(rows: list[dict]) -> str:
    if not rows:
        return "The draft_adp_drafts table is empty. Run the draft ADP crawler first."
    lines = ["Available stored cohorts:", "  season  type      format      drafts"]
    for row in rows:
        fmt = "Superflex" if row.get("is_superflex") else "1QB"
        lines.append(f"  {row.get('season', '?')!s:<6}  {row.get('draft_type', '?')!s:<8}  "
                     f"{fmt:<10}  {int(row.get('drafts') or 0):>6}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, action="append", dest="seasons",
                        help="Season to include; repeat for multiple. Default: all stored seasons.")
    parser.add_argument("--draft-type", action="append", choices=("redraft", "startup", "rookie"),
                        dest="draft_types", help="Cohort to include; repeatable. Default: all.")
    parser.add_argument("--min-drafts", type=int, default=10,
                        help="Hide cohorts smaller than this (default: 10).")
    parser.add_argument("--max-drafts", type=int, default=500,
                        help="Recent drafts sampled per type/format (default: 500; 0 means all).")
    parser.add_argument("--output", type=Path,
                        help="Also write the Markdown report to this path (it is still printed).")
    parser.add_argument("--json", type=Path, dest="json_output", help="Also write machine-readable JSON.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print the Markdown report; useful for scheduled jobs.")
    args = parser.parse_args(argv)

    if not os.getenv("DATABASE_URL"):
        parser.error("DATABASE_URL is not set; run this in a Render shell or export it locally")
    draft_types = args.draft_types or ["redraft", "startup", "rookie"]
    sample_text = "all matching drafts" if args.max_drafts == 0 else f"up to {args.max_drafts:,} drafts per cohort"
    print(f"Querying production DB ({sample_text})...", file=sys.stderr, flush=True)
    drafts, picks = fetch_rows(args.seasons, draft_types, args.max_drafts)
    if not drafts:
        print("No drafts matched the requested season/type filters.\n", file=sys.stderr)
        print(render_inventory(fetch_inventory()), file=sys.stderr)
        print("\nRetry without --season to audit every stored season, or choose seasons/types "
              "shown above.", file=sys.stderr)
        return 1
    positions = load_position_map()
    grouped: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    draft_cohorts: dict[str, tuple[str, bool]] = {}
    for draft in drafts:
        key = (str(draft["draft_type"]), bool(draft["is_superflex"]))
        grouped[key].append(draft)
        draft_cohorts[str(draft["draft_id"])] = key
    picks_by_cohort: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for pick in picks:
        key = draft_cohorts.get(str(pick["draft_id"]))
        if key is not None:
            picks_by_cohort[key].append(pick)
    print(f"Loaded {len(drafts):,} drafts and {len(picks):,} picks; building report...",
          file=sys.stderr, flush=True)
    summaries = [summarize_cohort(group, picks_by_cohort[key], positions)
                 for key, group in grouped.items() if len(group) >= args.min_drafts]
    summaries.sort(key=lambda s: (s.draft_type, s.format))
    if not summaries:
        print(f"No cohort met --min-drafts={args.min_drafts}.", file=sys.stderr)
        return 1
    season_text = "all stored seasons" if not args.seasons else "seasons " + ", ".join(map(str, args.seasons))
    report = render_markdown(summaries, f"Filters: {season_text}; types {', '.join(draft_types)}.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr, flush=True)
    if not args.quiet:
        print(report, end="")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps([asdict(s) for s in summaries], indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_output}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
