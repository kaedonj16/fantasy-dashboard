#!/usr/bin/env python3
"""
derive_v4_reference.py - build the v4 model's reference artifact.

The v4 model (data_building/rookie_pipeline/prospect_model_v4.py)
prices every season-stat leg against RECORDED DISTRIBUTIONS of
drafted prospects. This script derives those distributions and
writes ONE versioned JSON artifact:

    data_building/rookie_pipeline/v4_reference.json

RUN IT ON RENDER (web or daily-job shell), where CFBD_API_KEY lives:

    python scripts/derive_v4_reference.py
    # optional: --classes 2016-2024 --out path/to/v4_reference.json

What it captures, for the drafted population of classes 2016-2024
(nflverse draft list; college seasons from CFBD via the pipeline's
own ingestion, the same source and season format the model grades):

  (a) COMPONENT PERCENTILE TABLES: the population run through the
      CURRENT model (v3.3); per position, the recorded distribution
      of production / efficiency / breakout / athleticism component
      scores, as percentile anchor tables for display + receipts.
  (b) LEG KNOT MARKS: per position, the 11 tier marks (worst / p1 /
      p5 / p10 / p25 / p50 / p75 / p90 / p97 / p99 / best) for every
      season-stat leg input in the round 15 pre-registered design
      (pools: seasons with GP >= 4; breakout pools are per-player
      peak values), plus age-at-draft marks (performance order),
      recruiting-composite marks (CFBD recruiting pull, best
      effort), and competition-index marks. Athleticism marks are
      the recorded drafted-tester tiers already carried by v3.3.
  (c) PROVENANCE + POPULATIONS: source, generated-at, the exact
      population definition, and the n behind every table.

FAIL CLOSED: without CFBD_API_KEY the real run exits before doing
anything. --dry-run validates the whole machinery WITHOUT the key
against the repo's three surviving test fixtures (Smith / Green /
Mestemaker in tests/test_prospect_v32.py) and writes an artifact
whose provenance block marks it FIXTURE SAMPLE data. A dry-run
artifact must never be renamed into production use; the v4 model
stamps fixture-anchored receipts as such.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))

SCRIPT_VERSION = "1.0"
POPULATION_DEFINITION = (
    "Drafted skill-position players (QB/RB/WR/TE), classes {classes}, "
    "from the nflverse draft list; college seasons from CFBD via the "
    "pipeline's ingestion (season < draft year); season-leg pools "
    "require GP >= 4; breakout pools are per-player peak values over "
    "seasons with GP >= 1."
)
KNOT_PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 97, 99, 100]
DISPLAY_PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 97, 99, 100]
COMPONENT_FIELDS = ["production_score", "efficiency_score",
                    "breakout_profile_score", "athleticism_score"]
POSITIONS = ("QB", "RB", "WR", "TE")


# ─────────────────────────────────────────────────────────────────────
# Percentile machinery (pure stdlib; the round 15 method)
# ─────────────────────────────────────────────────────────────────────

def _percentile(sorted_vals: List[float], p: float) -> float:
    n = len(sorted_vals)
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (rank - lo)


def knot_marks(values: List[float]) -> Optional[List[float]]:
    """11 tier marks at KNOT_PERCENTILES. Ties stay tied: discrete
    stats mass at one value in the recorded population, and the
    model's knot scorer prices a value on a tied mark at the
    highest tier that mark represents."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return [round(_percentile(vals, p), 4) for p in KNOT_PERCENTILES]


def percentile_table(values: List[float]) -> Optional[List[List[float]]]:
    """[[percentile, mark], ...] at DISPLAY_PERCENTILES."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return [[p, round(_percentile(vals, p), 4)] for p in DISPLAY_PERCENTILES]


# ─────────────────────────────────────────────────────────────────────
# Artifact assembly (shared by the real run and the dry run, so the
# machinery CI validates is the machinery that ships)
# ─────────────────────────────────────────────────────────────────────

def build_reference(population: List[Dict[str, Any]],
                    component_rows: Dict[str, Dict[str, Any]],
                    athleticism_marks: Dict[str, Any],
                    provenance: Dict[str, Any]) -> Dict[str, Any]:
    """population: prospect dicts (with seasons/age/recruiting).
    component_rows: {player_id: current-model result dict}."""
    from data_building.rookie_pipeline import prospect_model_v4 as v4

    anchors: Dict[str, Dict[str, List[float]]] = {p: {} for p in POSITIONS}
    percentiles: Dict[str, Dict[str, List[List[float]]]] = {p: {} for p in POSITIONS}
    populations: Dict[str, Any] = {
        "players": len(population),
        "players_by_position": {},
        "season_legs": {}, "breakout": {}, "components": {},
        "age": {}, "recruiting": {}, "competition": {},
    }

    by_pos: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POSITIONS}
    for prospect in population:
        pos = (prospect.get("position") or "").upper()
        if pos in by_pos:
            by_pos[pos].append(prospect)
    for pos, rows in by_pos.items():
        populations["players_by_position"][pos] = len(rows)

    # (a) component percentile tables, from the current model's scores
    for pos, rows in by_pos.items():
        populations["components"][pos] = {}
        for field in COMPONENT_FIELDS:
            vals = [component_rows[r["player_id"]].get(field)
                    for r in rows
                    if component_rows.get(r["player_id"], {}).get(field)
                    is not None]
            table = percentile_table([float(v) for v in vals])
            if table is not None:
                percentiles[pos][field] = table
            populations["components"][pos][field] = len(vals)

    # (b) season-leg pools (GP >= 4), breakout pools (per-player peak),
    # age, recruiting, competition index
    # Season-leg pools cover exactly the model's leg inputs per
    # position (the union of its production + efficiency sub-legs).
    def _pos_leg_keys(pos: str) -> List[str]:
        keys = {k for specs in (v4.PRODUCTION_SPECS, v4.EFFICIENCY_SPECS)
                for k, _w in specs.get(pos, [])}
        return sorted(keys)
    for pos, rows in by_pos.items():
        pools: Dict[str, List[float]] = {k: [] for k in _pos_leg_keys(pos)}
        peak_pool: List[float] = []
        traj_pool: List[float] = []
        age_pool: List[float] = []
        rec_pool: List[float] = []
        comp_pool: List[float] = []
        for prospect in rows:
            seasons = prospect.get("seasons") or []
            for s in seasons:
                m = v4.season_metrics(s, pos)
                if m["games_played"] is not None and m["games_played"] >= 4:
                    for k in pools:
                        if m.get(k) is not None:
                            pools[k].append(float(m[k]))
            peak_val, _ = v4.breakout_peak_value(seasons, pos)
            if peak_val is not None:
                peak_pool.append(float(peak_val))
            growth = v4.traj_growth(seasons)
            if growth is not None:
                traj_pool.append(float(growth))
            if prospect.get("age") is not None:
                age_pool.append(float(prospect["age"]))
            rec = prospect.get("recruiting") or {}
            if rec.get("composite_rating") is not None:
                rec_pool.append(float(rec["composite_rating"]))
            idx = v4.competition_index(seasons)
            if idx is not None:
                comp_pool.append(float(idx))
        populations["season_legs"][pos] = {}
        for k, vals in pools.items():
            marks = knot_marks(vals)
            if marks is not None:
                anchors[pos][k] = marks
            populations["season_legs"][pos][k] = len(vals)
        for name, vals in (("breakout_dom", peak_pool),
                           ("traj_growth", traj_pool)):
            marks = knot_marks(vals)
            if marks is not None:
                anchors[pos][name] = marks
            populations["breakout"].setdefault(pos, {})[name] = len(vals)
        age_marks = knot_marks(age_pool)
        if age_marks is not None:
            # Performance order for age: oldest is worst, so the
            # stored marks run oldest -> youngest.
            anchors[pos]["age_years"] = list(reversed(age_marks))
        populations["age"][pos] = len(age_pool)
        rec_marks = knot_marks(rec_pool)
        if rec_marks is not None:
            anchors[pos]["recruiting_composite"] = rec_marks
        populations["recruiting"][pos] = len(rec_pool)
        comp_marks = knot_marks(comp_pool)
        if comp_marks is not None:
            anchors[pos]["competition_index"] = comp_marks
        populations["competition"][pos] = len(comp_pool)

    return {
        "artifact_version": 1,
        "provenance": provenance,
        "populations": populations,
        "anchors": anchors,
        "percentiles": percentiles,
        "athleticism": athleticism_marks,
    }


def print_summary(artifact: Dict[str, Any], out_path: str) -> None:
    pops = artifact["populations"]
    print(f"\nReference artifact written: {out_path}")
    print(f"  players: {pops['players']} "
          f"({pops['players_by_position']})")
    for pos in POSITIONS:
        legs = artifact["anchors"].get(pos, {})
        print(f"  {pos}: {len(legs)} anchor tables, "
              f"component tables: "
              f"{list(artifact['percentiles'].get(pos, {}).keys())}")
        thin = [k for k, n in pops["season_legs"].get(pos, {}).items()
                if n < 30]
        if thin:
            print(f"      thin season-leg pools (n<30): {thin}")
    prov = artifact["provenance"]
    print(f"  provenance: {prov.get('source')}")


# ─────────────────────────────────────────────────────────────────────
# Real run (Render): population + current-model scores from the
# existing backtest machinery; recruiting composites from CFBD.
# ─────────────────────────────────────────────────────────────────────

def _norm_name(name: str) -> str:
    import re
    s = (name or "").lower().strip()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _pull_recruiting_composites(years: List[int]) -> Dict[str, float]:
    """Best-effort {normalized_name: 247 composite} from CFBD's
    recruiting endpoint, for the classes' recruiting years. Coverage
    is partial by nature; the artifact records the n it produced."""
    key = os.environ.get("CFBD_API_KEY", "")
    out: Dict[str, float] = {}
    for yr in years:
        req = urllib.request.Request(
            f"https://api.collegefootballdata.com/recruiting/players"
            f"?year={yr}&classification=high_school",
            headers={"Authorization": f"Bearer {key}",
                     "User-Agent": "derive-v4-reference/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            print(f"  recruiting {yr}: pull failed ({exc}); continuing")
            continue
        for r in rows:
            rating = r.get("rating")
            if rating is not None:
                out.setdefault(_norm_name(r.get("name")), float(rating))
        time.sleep(0.2)
    return out


def run_real(classes: List[int], out_path: str) -> Dict[str, Any]:
    if not os.environ.get("CFBD_API_KEY"):
        sys.exit("CFBD_API_KEY is not set in this environment. "
                 "Run on Render, or use --dry-run to validate the "
                 "machinery against the repo fixtures.")
    import backtest_prospect_model as bt
    from data_building.rookie_pipeline import prospect_model as pm

    population: List[Dict[str, Any]] = []
    component_rows: Dict[str, Dict[str, Any]] = {}
    recruiting_years = sorted({y + off for y in classes
                               for off in (-5, -4, -3)})
    composites = _pull_recruiting_composites(recruiting_years)
    print(f"recruiting composites pulled: {len(composites)}")

    for year in classes:
        print(f"\n[{year}] loading drafted class...")
        draft_class = bt._load_draft_class(year)
        athleticism = bt._load_combine_athleticism(year)
        college_stats = bt._load_cfbd_college_stats(year, draft_class)
        prospects, consensus = bt._build_prospect_dicts(
            draft_class, athleticism, year, college_stats)
        for p in prospects:
            if not p.get("recruiting"):
                rating = composites.get(_norm_name(p.get("name")))
                if rating is not None:
                    p["recruiting"] = {"composite_rating": rating}
        scores = pm.score_all_prospects(prospects, consensus,
                                        skip_sagarin=True)
        for row in scores:
            component_rows[row["player_id"]] = row
        population.extend(prospects)
        print(f"[{year}] {len(prospects)} prospects scored")

    provenance = {
        "source": ("CFBD (pipeline ingestion) + nflverse draft list; "
                   "component scores from prospect model "
                   f"{pm.MODEL_VERSION}; athleticism marks are v3.3's "
                   "recorded drafted-tester tiers"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "population": POPULATION_DEFINITION.format(
            classes=f"{classes[0]}-{classes[-1]}"),
        "classes": classes,
        "script": f"scripts/derive_v4_reference.py v{SCRIPT_VERSION}",
        "fixture_sample": False,
    }
    artifact = build_reference(population, component_rows,
                               dict(pm._KNOT_MARKS), provenance)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    return artifact


# ─────────────────────────────────────────────────────────────────────
# Dry run: the same assembly, validated on the repo fixtures.
# ─────────────────────────────────────────────────────────────────────

def run_dry(out_path: str) -> Dict[str, Any]:
    try:
        import test_prospect_v32 as fx
    except BaseException as exc:  # fixture module skips without pandas
        sys.exit(f"--dry-run needs the repo test fixtures importable "
                 f"(tests/test_prospect_v32.py): {exc!r}")
    import copy
    from data_building.rookie_pipeline import prospect_model as pm
    from data_building.rookie_pipeline.mock_draft_consensus import (
        pick_to_draft_capital_score)

    def _dc(pos: str, pick: int) -> Dict[str, Any]:
        return {"projected_pick": pick,
                "projected_round": ((pick - 1) // 32) + 1,
                "projected_draft_capital_score":
                    pick_to_draft_capital_score(pick, pos),
                "num_mocks_used": 5, "is_actual_pick": False}

    fixtures = [(fx.SMITH, _dc("WR", 2)), (fx.GREEN, _dc("TE", 20)),
                (fx.MESTEMAKER, _dc("QB", 8))]
    population, component_rows = [], {}
    for fixture, dc in fixtures:
        population.append(copy.deepcopy(fixture))
        component_rows[fixture["player_id"]] = pm.score_prospect(
            copy.deepcopy(fixture), dc, skip_sagarin=True)
    # A small recorded subset so the athleticism path is exercised:
    # v3.3's WR forty/vertical tiers (real recorded marks; the
    # fixtures themselves carry no testing).
    athleticism_marks = {"WR": {
        "forty": pm._KNOT_MARKS["WR"]["forty"],
        "vertical": pm._KNOT_MARKS["WR"]["vertical"]}}
    provenance = {
        "source": ("FIXTURE SAMPLE: derived from the repo's three "
                   "surviving test fixtures (Smith / Green / "
                   "Mestemaker, tests/test_prospect_v32.py). NOT "
                   "recorded population data; machinery validation "
                   "only. The WR forty/vertical athleticism marks are "
                   "v3.3's recorded drafted-tester tiers, included "
                   "so tests can exercise the athleticism path."),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "population": "repo test fixtures (n=3)",
        "classes": [],
        "script": f"scripts/derive_v4_reference.py v{SCRIPT_VERSION}",
        "fixture_sample": True,
    }
    artifact = build_reference(population, component_rows,
                               athleticism_marks, provenance)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    return artifact


def _parse_classes(spec: str) -> List[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the machinery on repo fixtures, "
                         "no CFBD key or network needed")
    ap.add_argument("--classes", default="2016-2024",
                    help="draft classes, e.g. 2016-2024 (default)")
    ap.add_argument("--out", default=None,
                    help="output path (default: the model's "
                         "v4_reference.json; dry-run defaults to "
                         "/tmp/v4_reference_dry_run.json)")
    args = ap.parse_args()
    default_out = os.path.join(
        _REPO_ROOT, "data_building", "rookie_pipeline", "v4_reference.json")
    if args.dry_run:
        out = args.out or "/tmp/v4_reference_dry_run.json"
        artifact = run_dry(out)
    else:
        out = args.out or default_out
        artifact = run_real(_parse_classes(args.classes), out)
    print_summary(artifact, out)


if __name__ == "__main__":
    main()
