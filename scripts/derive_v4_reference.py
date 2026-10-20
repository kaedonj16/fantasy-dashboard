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
  (a2) ADVANCED-METRIC PERCENTILE TABLES: per position, the recorded
      distribution of the modal Advanced tab's metrics (Dominator,
      Breakout Age, Mkt Share, TD Share, Tgt/Gm, Catch%, Rec Yds/Gm,
      Yds/Rec, Career Yds/Rec, Best Dominator, and every other
      non-component row), valued by the API's own builder
      (dashboard_services.rookie_api._build_advanced_metrics) over
      each population player, so the recorded distribution and the
      displayed row can never drift apart. Lower-is-better metrics
      (Breakout Age) are recorded in advanced_lower_better. Nothing
      consumes these tables yet; they land with the artifact so the
      Advanced tab's percentiles can move from the thin current-class
      pool to the recorded drafted pool.
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

SPEED: the real run disables the pipeline's play-by-play targets
parse (ingestion.fetch_cfbd_targets_from_plays) before loading any
class, identically to build_reference_population.py and the
backtest wrapper: that crawl walks every team x season, 400s on
old seasons, and takes hours, while targets enter this artifact
only through the same thin distributions the other tools accept.

RESUMABLE: each class, once loaded and scored, is cached to
derive_cache/v4/<year>.json holding exactly the per-player fields
build_reference() consumes. A later run loads cached classes from
disk with zero CFBD calls; --no-cache forces live loads. Classes
whose stat coverage is below the gate floor are never cached, so
a rerun re-fetches only the classes that need it.

COVERAGE GATE: if any class's stat-matched fraction (prospects
with college seasons / prospects in the class) is below 0.50, the
run prints the starved classes and exits nonzero WITHOUT writing
the artifact, so a rate-limited fetch can never bake distorted
distributions into the anchors.
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
from decimal import Decimal
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))

SCRIPT_VERSION = "1.2"
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
        "advanced": {},
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

    # (a2) advanced-metric percentile tables, from the API's own
    # advanced-metrics builder run over every population player (the
    # same construction the live _advanced_distributions uses against
    # the current-class pool). Component-score rows are excluded by
    # their flag; a metric enters the pool only when the builder
    # produced a value for that player.
    from dashboard_services.rookie_api import (
        _ADV_LOWER_BETTER, _build_advanced_metrics)
    advanced: Dict[str, Dict[str, List[List[float]]]] = {
        p: {} for p in POSITIONS}
    advanced_lower: Dict[str, List[str]] = {p: [] for p in POSITIONS}
    for pos, rows in by_pos.items():
        populations["advanced"][pos] = {}
        vals_by_label: Dict[str, List[float]] = {}
        for prospect in rows:
            comp = component_rows.get(prospect["player_id"], {})
            brow = {"age": prospect.get("age"),
                    "weight_lbs": prospect.get("weight_lbs"),
                    "efficiency_score": comp.get("efficiency_score"),
                    "production_score": comp.get("production_score")}
            for a in _build_advanced_metrics(
                    pos, prospect.get("seasons") or [],
                    prospect.get("athleticism") or {}, brow, None):
                if a.get("component") or a.get("value") is None:
                    continue
                vals_by_label.setdefault(a["label"], []).append(
                    float(a["value"]))
        for label, vals in vals_by_label.items():
            table = percentile_table(vals)
            if table is not None:
                advanced[pos][label] = table
            populations["advanced"][pos][label] = len(vals)
        advanced_lower[pos] = sorted(
            label for label in vals_by_label
            if label in _ADV_LOWER_BETTER)

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
        "advanced": advanced,
        "advanced_lower_better": advanced_lower,
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
        adv = artifact.get("advanced", {}).get(pos, {})
        print(f"      advanced tables: {len(adv)} "
              f"({', '.join(sorted(adv))})")
        thin_adv = [k for k, n in pops["advanced"].get(pos, {}).items()
                    if n < 30]
        if thin_adv:
            print(f"      thin advanced pools (n<30): {thin_adv}")
    prov = artifact["provenance"]
    print(f"  provenance: {prov.get('source')}")


# ─────────────────────────────────────────────────────────────────────
# Per-class derive cache + coverage gate (script v1.2)
#
# The cache holds, per class, exactly the per-player fields
# build_reference() consumes, JSON-normalized at load time on the
# live path too, so a cached run feeds build_reference() inputs
# identical to a live run's and produces the same artifact.
# ─────────────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(_REPO_ROOT, "derive_cache", "v4")
CACHE_VERSION = 1
COVERAGE_FLOOR = 0.50
# The prospect fields build_reference() reads. "seasons" and
# "athleticism" are cached whole: their consumers (the v4 leg
# helpers and rookie_api._build_advanced_metrics) read a wide key
# set inside them, so pruning keys would risk silent divergence
# between the live and cached paths.
PROSPECT_CACHE_FIELDS = ("player_id", "position", "age", "weight_lbs",
                         "seasons", "athleticism", "recruiting")
# The component-row fields build_reference() reads (== COMPONENT_FIELDS).
COMPONENT_CACHE_FIELDS = tuple(COMPONENT_FIELDS)


def _jsonable(value: Any) -> Any:
    """Recursively convert a value to plain JSON types (dict keys
    stringified, tuples -> lists, Decimal -> float). Anything that
    cannot be represented faithfully raises, at cache-write time,
    rather than silently corrupting the cached inputs."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)  # numpy scalars and other numerics
    except (TypeError, ValueError):
        raise TypeError(
            f"derive cache cannot serialize {type(value)!r}") from None


def _extract_prospect(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _jsonable(p.get(k)) for k in PROSPECT_CACHE_FIELDS}


def _extract_component_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {f: _jsonable(row.get(f)) for f in COMPONENT_CACHE_FIELDS}


def _matched_fraction(prospects: List[Dict[str, Any]]) -> float:
    """Share of the class with college stats (non-empty seasons)."""
    if not prospects:
        return 0.0
    matched = sum(1 for p in prospects if p.get("seasons"))
    return matched / len(prospects)


def _cache_path(cache_dir: str, year: int) -> str:
    return os.path.join(cache_dir, f"{year}.json")


def _write_class_cache(cache_dir: str, year: int,
                       prospects: List[Dict[str, Any]],
                       component_rows: Dict[str, Dict[str, Any]]) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, year)
    payload = {"cache_version": CACHE_VERSION, "year": year,
               "prospects": prospects, "component_rows": component_rows}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def _read_class_cache(cache_dir: str,
                      year: int) -> Optional[Any]:
    """(prospects, component_rows) for a cached class, else None.
    A missing, unreadable, stale-version, or sub-floor cache is a
    miss: sub-floor entries in particular must never short-circuit
    the coverage gate, so the class re-fetches live instead."""
    path = _cache_path(cache_dir, year)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    if (payload.get("cache_version") != CACHE_VERSION
            or payload.get("year") != year):
        return None
    prospects = payload.get("prospects") or []
    if _matched_fraction(prospects) < COVERAGE_FLOOR:
        return None
    return prospects, payload.get("component_rows") or {}


def _disable_plays_targets() -> None:
    """Skip the play-by-play targets parse (400s / crawls on old
    seasons; build_reference_population.py and the backtest wrapper
    disable it identically)."""
    try:
        from data_building.rookie_pipeline import ingestion
        ingestion.fetch_cfbd_targets_from_plays = lambda *a, **k: {}
    except Exception:
        pass


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


def _load_class_live(year: int, bt: Any, pm: Any,
                     composites: Dict[str, float]) -> Any:
    """Load + score one class live from CFBD, returned in the
    extracted (cache-shaped) form build_reference() consumes."""
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
    rows = {row["player_id"]: _extract_component_row(row)
            for row in scores}
    return [_extract_prospect(p) for p in prospects], rows


def run_real(classes: List[int], out_path: str, use_cache: bool = True,
             cache_dir: Optional[str] = None) -> Dict[str, Any]:
    if not os.environ.get("CFBD_API_KEY"):
        sys.exit("CFBD_API_KEY is not set in this environment. "
                 "Run on Render, or use --dry-run to validate the "
                 "machinery against the repo fixtures.")
    # Before any class loads: kill the play-by-play targets crawl
    # (see the module docstring's SPEED note).
    _disable_plays_targets()
    import backtest_prospect_model as bt
    from data_building.rookie_pipeline import prospect_model as pm

    cache_dir = cache_dir or CACHE_DIR
    population: List[Dict[str, Any]] = []
    component_rows: Dict[str, Dict[str, Any]] = {}
    coverage: Dict[int, float] = {}
    recruiting_years = sorted({y + off for y in classes
                               for off in (-5, -4, -3)})
    composites: Optional[Dict[str, float]] = None

    for year in classes:
        cached = _read_class_cache(cache_dir, year) if use_cache else None
        if cached is not None:
            prospects, rows = cached
            print(f"[{year}] from derive cache: {len(prospects)} prospects")
        else:
            if composites is None:
                composites = _pull_recruiting_composites(recruiting_years)
                print(f"recruiting composites pulled: {len(composites)}")
            print(f"\n[{year}] loading drafted class...")
            prospects, rows = _load_class_live(year, bt, pm, composites)
            print(f"[{year}] {len(prospects)} prospects scored")
            if _matched_fraction(prospects) >= COVERAGE_FLOOR:
                _write_class_cache(cache_dir, year, prospects, rows)
                print(f"[{year}] derive cache written")
            else:
                print(f"[{year}] derive cache skipped: stat coverage "
                      f"below the {COVERAGE_FLOOR:.0%} floor; a rerun "
                      f"re-fetches this class")
        population.extend(prospects)
        component_rows.update(rows)
        coverage[year] = _matched_fraction(prospects)

    starved = [(y, coverage[y]) for y in classes
               if coverage[y] < COVERAGE_FLOOR]
    if starved:
        print("\n[coverage gate] refusing to write the artifact: "
              "these classes scored with too little college data, "
              "and their distributions would distort the anchors:")
        for year, frac in starved:
            print(f"  [{year}] stat-matched fraction {frac:.3f} "
                  f"< {COVERAGE_FLOOR:.2f}")
        print("Classes above the floor keep their derive cache; "
              "rerun when CFBD is healthy to re-fetch only the "
              "starved classes.")
        sys.exit(1)

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
        "class_coverage": {str(y): round(coverage[y], 4)
                           for y in classes},
        "derive_cache": ("per-class cache under derive_cache/v4 "
                         f"(cache_version {CACHE_VERSION}); cached "
                         "classes load from disk with zero CFBD calls "
                         "unless --no-cache; classes below the "
                         "coverage floor are never cached"),
        "coverage_gate": ("artifact refused when any class's "
                          "stat-matched fraction (prospects with "
                          "college seasons / class prospects) is "
                          f"below {COVERAGE_FLOOR:.2f}"),
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
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the per-class derive cache and load "
                         "every class live from CFBD")
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
        artifact = run_real(_parse_classes(args.classes), out,
                            use_cache=not args.no_cache)
    print_summary(artifact, out)


if __name__ == "__main__":
    main()
