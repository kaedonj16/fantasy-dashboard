#!/usr/bin/env python3
"""
Build the graded reference population for prospect percentiles.

The board/modal percentiles are computed within position against a
reference population of graded prospects. The stored classes alone
are a thin pool (classes 2025-2027, ~34 WRs), so the best player in
the pool prints 100 on every component. This script grades the
DRAFTED classes 2016-2024 through the CURRENT production model (the
model is imported from main at runtime; no version is pinned here)
and persists one reference row per drafted QB/RB/WR/TE into
prospect_reference_population (migration 050): component scores plus
the advanced marks, season rates, and testing marks the percentile
surfaces pool over. Rows are REFERENCE ONLY: percentile anchors,
never prospects; no board/values/rankings query reads them.

Machinery is reused, not reimplemented: draft classes, combine
athleticism, and CFBD college stats come from the backtest loaders
(scripts/backtest_prospect_model.py, which fetches through the
pipeline's own ingestion), scoring is score_all_prospects, and the
advanced marks / season rates are computed by the same functions the
live modal uses (dashboard_services.rookie_api._build_advanced_metrics
and _season_primary_rate), so a stored mark can never drift from the
definition the modal renders.

Resumable per class year: a year whose stored row count already
covers its drafted skill class is skipped unless --force. Re-runs
upsert on (position, draft_class_year, player_name).

Targets note: CFBD targets come from parsing play-by-play text
(fetch_cfbd_targets_from_plays), which 400s / crawls on old seasons
(the established backtest wrapper disables it for the same reason).
It is disabled here by default so the run takes backtest time, not
targets-backfill time (hours); pass --targets to leave it enabled.
Consequence when disabled: the Tgt/Gm and Catch% marks carry only
whatever the structured season fetch provides, and those two
distributions stay thin (the percentile surfaces hide thin metrics
rather than fake them).

Usage (Render web shell, after migration 050 has run):
    python scripts/run_migrations.py            # applies 050
    nohup python scripts/build_reference_population.py \
        > /tmp/refpop.log 2>&1 &
    tail -f /tmp/refpop.log                     # per-year progress
Then restart / redeploy the web service so its per-process
population caches pick up the enlarged pool.

    python scripts/build_reference_population.py --years 2020 2021
    python scripts/build_reference_population.py --force
    python scripts/build_reference_population.py --dry-run   # fixtures only

Fails closed (exit 2, nothing written) when CFBD_API_KEY or
DATABASE_URL is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))

DEFAULT_YEARS = list(range(2016, 2025))
SKILL_POS = {"QB", "RB", "WR", "TE"}
# Testing fields the modal's Testing table pools over (rookie_api
# _TEST_SPECS keys), as carried in the combine loader's dicts.
_TEST_FIELDS = ("forty_yard", "vertical_inches", "broad_jump_in",
                "three_cone", "short_shuttle", "bench_reps", "ras_score")

_UPSERT_SQL = """
INSERT INTO prospect_reference_population (
    player_name, position, draft_class_year, actual_pick,
    production_score, efficiency_score, breakout_profile_score,
    competition_score, athleticism_score, tested,
    advanced_marks, season_rates, testing_marks, model_version,
    updated_at)
VALUES (
    %(player_name)s, %(position)s, %(draft_class_year)s, %(actual_pick)s,
    %(production_score)s, %(efficiency_score)s, %(breakout_profile_score)s,
    %(competition_score)s, %(athleticism_score)s, %(tested)s,
    %(advanced_marks)s::jsonb, %(season_rates)s::jsonb,
    %(testing_marks)s::jsonb, %(model_version)s, NOW())
ON CONFLICT (position, draft_class_year, player_name) DO UPDATE SET
    actual_pick            = EXCLUDED.actual_pick,
    production_score       = EXCLUDED.production_score,
    efficiency_score       = EXCLUDED.efficiency_score,
    breakout_profile_score = EXCLUDED.breakout_profile_score,
    competition_score      = EXCLUDED.competition_score,
    athleticism_score      = EXCLUDED.athleticism_score,
    tested                 = EXCLUDED.tested,
    advanced_marks         = EXCLUDED.advanced_marks,
    season_rates           = EXCLUDED.season_rates,
    testing_marks          = EXCLUDED.testing_marks,
    model_version          = EXCLUDED.model_version,
    updated_at             = NOW()
"""


# ── Pure row assembly (shared by the real run, the dry run, tests) ──

def _advanced_marks(position: str, seasons: List[Dict],
                    athleticism: Dict, row: Dict) -> Dict[str, float]:
    """{label: value} from the live modal's own advanced-metrics code
    path; component-score rows and valueless rows are excluded, exactly
    as the stored-prospects distribution builder excludes them."""
    from dashboard_services.rookie_api import _build_advanced_metrics
    out: Dict[str, float] = {}
    for a in _build_advanced_metrics(position, seasons, athleticism,
                                     row, wepa_score=None):
        if a.get("component") or a.get("value") is None:
            continue
        try:
            out[a["label"]] = float(a["value"])
        except (TypeError, ValueError):
            continue
    return out


def _season_rates(position: str, seasons: List[Dict]) -> List[float]:
    """Completed-season primary per-game rates, by the modal's own
    definitions (_season_primary_rate, GP floor, in-season excluded)."""
    from dashboard_services.rookie_api import (
        _SEASON_MIN_GP, _season_primary_rate, is_in_season)
    rates: List[float] = []
    for s in seasons or []:
        gp = s.get("games_played")
        try:
            gp = float(gp) if gp is not None else None
        except (TypeError, ValueError):
            gp = None
        if not gp or gp < _SEASON_MIN_GP:
            continue
        if is_in_season(s.get("season"), gp):
            continue
        rate = _season_primary_rate(position, s)
        if rate is not None:
            rates.append(rate)
    return sorted(rates)


def build_reference_rows(
    draft_year: int,
    draft_class: List[Dict[str, Any]],
    prospects: List[Dict[str, Any]],
    scores: List[Dict[str, Any]],
    model_version: str,
) -> List[Dict[str, Any]]:
    """Assemble prospect_reference_population rows for one class.

    draft_class / prospects / scores are the backtest machinery's own
    shapes (_load_draft_class, _build_prospect_dicts,
    score_all_prospects). One row per drafted skill player who was
    scored. The athleticism score is stored ONLY for players with
    recorded testing (forty or RAS): an untested player's model score
    is a floor, and floors never anchor the athleticism scale.
    """
    dc_by_id = {
        (p["gsis_id"] or p["name"].lower().replace(" ", "-")): p
        for p in draft_class
    }
    p_by_id = {p["player_id"]: p for p in prospects}
    rows: List[Dict[str, Any]] = []
    for sc in scores:
        pid = sc.get("player_id")
        p = p_by_id.get(pid, {})
        dc = dc_by_id.get(pid, {})
        pos = (sc.get("position") or p.get("position") or "").upper()
        if pos not in SKILL_POS:
            continue
        name = (p.get("name") or dc.get("name") or "").strip()
        if not name:
            continue
        ath = p.get("athleticism") or {}
        tested = (ath.get("forty_yard") is not None
                  or ath.get("ras_score") is not None)
        testing = {f: ath[f] for f in _TEST_FIELDS if ath.get(f) is not None}
        marks_row = {
            "age": p.get("age"),
            "weight_lbs": p.get("weight_lbs"),
            "efficiency_score": sc.get("efficiency_score"),
            "production_score": sc.get("production_score"),
        }
        seasons = p.get("seasons") or []
        pick = dc.get("draft_pick") or None
        rows.append({
            "player_name": name,
            "position": pos,
            "draft_class_year": int(draft_year),
            "actual_pick": int(pick) if pick else None,
            "production_score": sc.get("production_score"),
            "efficiency_score": sc.get("efficiency_score"),
            "breakout_profile_score": sc.get("breakout_profile_score"),
            "competition_score": sc.get("competition_score"),
            "athleticism_score": (sc.get("athleticism_score")
                                  if tested else None),
            "tested": tested,
            "advanced_marks": _advanced_marks(pos, seasons, ath, marks_row),
            "season_rates": _season_rates(pos, seasons),
            "testing_marks": testing,
            "model_version": model_version,
        })
    return rows


def year_is_complete(stored_count: int, expected_count: int) -> bool:
    """Resumability rule: a year is done when its stored rows already
    cover every drafted skill player in it."""
    return expected_count > 0 and stored_count >= expected_count


# ── DB side ─────────────────────────────────────────────────────────

def _stored_count(draft_year: int) -> int:
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM prospect_reference_population "
            "WHERE draft_class_year = %s",
            (draft_year,),
        ).fetchone()
    return int(row["n"]) if row else 0


def _upsert_rows(rows: List[Dict[str, Any]]) -> int:
    from dashboard_services.db import get_conn
    written = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(_UPSERT_SQL, {
                    **r,
                    "advanced_marks": json.dumps(r["advanced_marks"]),
                    "season_rates": json.dumps(r["season_rates"]),
                    "testing_marks": json.dumps(r["testing_marks"]),
                })
                written += 1
        conn.commit()
    return written


# ── Per-year run ────────────────────────────────────────────────────

def _disable_plays_targets() -> None:
    """Skip the play-by-play targets parse (400s / crawls on old
    seasons; the backtest wrapper disables it identically)."""
    try:
        from data_building.rookie_pipeline import ingestion
        ingestion.fetch_cfbd_targets_from_plays = lambda *a, **k: {}
    except Exception:
        pass


def _run_year(draft_year: int, force: bool = False) -> int:
    from scripts.backtest_prospect_model import (
        _build_prospect_dicts, _load_cfbd_college_stats,
        _load_combine_athleticism, _load_draft_class,
    )
    from data_building.rookie_pipeline.prospect_model import (
        MODEL_VERSION, score_all_prospects,
    )

    print(f"\n[refpop] ── {draft_year} ─────────────────────────────")
    draft_class = _load_draft_class(draft_year)
    if not draft_class:
        print(f"[refpop] {draft_year}: no nflverse draft class, skipping.")
        return 0
    expected = len(draft_class)
    if not force:
        stored = _stored_count(draft_year)
        if year_is_complete(stored, expected):
            print(f"[refpop] {draft_year}: {stored} rows stored, covering "
                  f"all {expected} drafted skill players; skipping "
                  f"(pass --force to re-grade).")
            return stored
        if stored:
            print(f"[refpop] {draft_year}: {stored}/{expected} rows "
                  f"stored; completing the year.")

    athleticism = _load_combine_athleticism(draft_year)
    college_stats = _load_cfbd_college_stats(draft_year, draft_class)
    prospects, consensus_map = _build_prospect_dicts(
        draft_class, athleticism, draft_year, college_stats)
    n_stats = sum(1 for p in prospects if p.get("seasons"))
    print(f"[refpop] {draft_year}: scoring {len(prospects)} drafted "
          f"skill players ({n_stats} with college stats) with the "
          f"current model (version {MODEL_VERSION})…")
    scores = score_all_prospects(prospects, consensus_map,
                                 skip_sagarin=True)
    rows = build_reference_rows(draft_year, draft_class, prospects,
                                scores, MODEL_VERSION)
    written = _upsert_rows(rows)
    by_pos: Dict[str, int] = {}
    for r in rows:
        by_pos[r["position"]] = by_pos.get(r["position"], 0) + 1
    print(f"[refpop] {draft_year}: wrote {written} reference rows "
          f"({', '.join(f'{k} {v}' for k, v in sorted(by_pos.items()))}).")
    return written


# ── Dry run: the same assembly over the repo fixtures, no DB/key ────

def run_dry_run() -> List[Dict[str, Any]]:
    import copy

    try:
        import test_prospect_v32 as fx
    except BaseException as exc:  # fixture module skips without pandas
        sys.exit(f"--dry-run needs the repo test fixtures importable "
                 f"(tests/test_prospect_v32.py): {exc!r}")
    from data_building.rookie_pipeline.prospect_model import (
        MODEL_VERSION, score_all_prospects,
    )
    from data_building.rookie_pipeline.mock_draft_consensus import (
        pick_to_draft_capital_score,
    )

    picks = {"T_smith": 2, "T_green": 20, "T_mestemaker": 8}
    fixtures = [copy.deepcopy(fx.SMITH), copy.deepcopy(fx.GREEN),
                copy.deepcopy(fx.MESTEMAKER)]
    consensus_map = {}
    draft_class = []
    for p in fixtures:
        pick = picks[p["player_id"]]
        pos = p["position"]
        consensus_map[p["player_id"]] = {
            "projected_pick": pick,
            "projected_round": ((pick - 1) // 32) + 1,
            "projected_draft_capital_score":
                pick_to_draft_capital_score(pick, pos),
            "num_mocks_used": 5,
            "is_actual_pick": True,
        }
        draft_class.append({
            "name": p["name"], "position": pos, "gsis_id": p["player_id"],
            "draft_pick": pick, "college": "", "birth_date": "",
        })
    scores = score_all_prospects(fixtures, consensus_map,
                                 skip_sagarin=True)
    rows = build_reference_rows(2027, draft_class, fixtures, scores,
                                MODEL_VERSION)
    for r in rows:
        print(f"  [DRY] {r['player_name']:<22} {r['position']:>2}  "
              f"pick={r['actual_pick']}  prod={r['production_score']}  "
              f"eff={r['efficiency_score']}  marks={len(r['advanced_marks'])}"
              f"  rates={len(r['season_rates'])}  tested={r['tested']}")
    print(f"[refpop] dry run: {len(rows)} fixture rows assembled; "
          f"nothing written (no DB, no key needed).")
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the graded reference population (drafted "
                    "classes 2016-2024) for prospect percentiles.")
    parser.add_argument("--years", nargs="+", type=int,
                        default=DEFAULT_YEARS,
                        help="Draft years to process (default: 2016-2024)")
    parser.add_argument("--force", action="store_true",
                        help="Re-grade years already fully stored")
    parser.add_argument("--targets", action="store_true",
                        help="Leave the play-by-play targets parse "
                             "enabled (slow on old seasons)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the machinery on repo fixtures; "
                             "no DB writes, no CFBD key needed")
    args = parser.parse_args(argv)

    if args.dry_run:
        run_dry_run()
        return 0

    missing = [k for k in ("CFBD_API_KEY", "DATABASE_URL")
               if not os.environ.get(k)]
    if missing:
        print(f"[refpop] FAIL CLOSED: {', '.join(missing)} not set in "
              f"the environment; nothing was written. Run where both "
              f"are present (the Render web shell), or use --dry-run.")
        return 2

    if not args.targets:
        _disable_plays_targets()

    total = 0
    for year in sorted(args.years):
        total += _run_year(year, force=args.force)
    print(f"\n[refpop] Done: {total} reference rows across "
          f"{len(set(args.years))} class year(s). Restart / redeploy "
          f"the web service so its population caches pick up the "
          f"enlarged pool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
