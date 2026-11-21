"""Prospect model v5: a learned outcome model (design:
~/workspace/prospect-model-v5-design.md, approved 2026-11-20).

Where v4 hand-priced weighted legs against knot ladders, v5 learns
the mapping from a prospect's raw college profile to his NFL
outcome from the recorded drafted population (2016-2024), one
gradient-boosted model per position (sklearn
HistGradientBoostingRegressor). The grade is the predicted outcome
(PPR per season over the first four NFL seasons) expressed as a
percentile of the recorded outcome distribution at the position,
mapped to the 0-100 scale through fixed absolute anchors that keep
the v4 scale semantics: 92+ elite, 80-92 great, 70-80 average,
below 70 day-three/fringe. 99+ is attainable only by a prediction
beyond the recorded outcome maximum.

Principles carried from the design's hardening pass:

- Leakage discipline: everything data-derived (the model, the
  outcome ECDF) is fit on training rows only. The percentile-to-
  grade anchors are fixed scale constants, not fit to data.
- Missing data is honest: untested athleticism and absent seasons
  stay missing (NaN); HistGradientBoosting handles them natively.
  No floors, no invented scores.
- Abstention: a prospect with no college season data gets no
  grade. The model never emits a capital-only guess wearing a
  talent grade's clothes.
- Capital is a feature whose value the fit learns. The receipt
  still prints a capital-blind grade (capital feature set to
  missing) and the capital delta, preserving v4's two-number
  transparency.
- Determinism: fixed seed everywhere; the artifact records a
  training fingerprint (population count + hash, class list,
  ordered feature lists, sklearn version, seed).
- Recruiting is deliberately NOT a feature: the recorded training
  population carries no recruiting data (the derive pools are
  empty), so the model could never learn it; a live-only feature
  would be pure train/serve skew.

Artifact: a dict {format, model_version, models, feature_lists,
outcome_ecdf, permutation_importance, fingerprint}, serialized
with pickle into Postgres (migration 057,
prospect_v5_model_artifact) by scripts/train_prospect_model_v5.py.
Grading fails closed (V5ArtifactError) when no artifact exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .prospect_model_v4 import (
    band_for,
    competition_index,
    detect_lifecycle_state,
    season_metrics,
    select_priced_seasons,
)

MODEL_VERSION = "5.0"
ARTIFACT_FORMAT = "prospect_v5_artifact"
RANDOM_SEED = 20261120

POSITIONS = ("QB", "RB", "WR", "TE")
_POSITION_ALIASES = {"FB": "RB"}

# ─────────────────────────────────────────────────────────────────────
# Grade anchors: predicted-outcome percentile -> grade. FIXED scale
# constants (design section 3), set once from the scale semantics,
# never fit to data and never tuned per class. The median drafted
# player grades 70 (the floor of the "average prospect" band); the
# elite band opens at the 97.5th outcome percentile; a 99 requires
# ~the 99.97th percentile, reachable only past the recorded max.
# ─────────────────────────────────────────────────────────────────────

GRADE_ANCHORS: List[Tuple[float, float]] = [
    (0.0, 30.0), (10.0, 45.0), (25.0, 58.0), (50.0, 70.0),
    (75.0, 80.0), (90.0, 87.0), (95.0, 90.0), (97.5, 92.0),
    (99.0, 95.0), (99.9, 98.0), (100.0, 99.5),
]
_ANCHOR_P = np.array([p for p, _ in GRADE_ANCHORS], dtype=float)
_ANCHOR_G = np.array([g for _, g in GRADE_ANCHORS], dtype=float)


def grade_for_percentile(percentile: Optional[float]) -> Optional[float]:
    """The 0-100 grade for a predicted-outcome percentile."""
    if percentile is None:
        return None
    return float(np.interp(float(percentile), _ANCHOR_P, _ANCHOR_G))


# ─────────────────────────────────────────────────────────────────────
# Features. One shared extractor over the prospect dict (the same
# dict shape the live pipeline and the backtest harness produce);
# per-position ordered lists select the columns each model sees.
# Monotone signs are the design's firm domain calls: production and
# efficiency rates up, age at draft down, capital up, timed speed
# marks down (lower is better). Everything else is unconstrained.
# ─────────────────────────────────────────────────────────────────────

MONOTONE: Dict[str, int] = {
    "primary_pg_peak": 1, "primary_pg_latest": 1,
    "efficiency_peak": 1, "efficiency_latest": 1,
    "td_pg_peak": 1, "rec_pg_peak": 1,
    "dominator_peak": 1, "market_share_peak": 1, "pass_share_peak": 1,
    "completion_pct_peak": 1, "td_int_peak": 1, "rush_pg_peak": 1,
    "vertical": 1, "broad": 1, "bench": 1, "ras": 1, "speed_score": 1,
    "capital_score": 1,
    "age": -1, "breakout_age": -1,
    "forty": -1, "cone": -1, "shuttle": -1,
}

FEATURES_BY_POSITION: Dict[str, List[str]] = {
    "QB": [
        "primary_pg_peak", "primary_pg_latest", "primary_pg_trend",
        "efficiency_peak", "efficiency_latest", "td_pg_peak",
        "completion_pct_peak", "td_int_peak", "qb_share_peak",
        "rush_pg_peak", "attempts_peak", "peak_games", "career_games",
        "age", "breakout_age", "competition_index", "cast_quality",
        "forty", "vertical", "broad", "speed_score", "weight",
        "tested", "capital_score",
    ],
    "RB": [
        "primary_pg_peak", "primary_pg_latest", "primary_pg_trend",
        "efficiency_peak", "efficiency_latest", "td_pg_peak",
        "rec_pg_peak", "dominator_peak", "market_share_peak",
        "touches_peak", "peak_games", "career_games", "n_seasons",
        "age", "breakout_age", "competition_index", "cast_quality",
        "forty", "vertical", "broad", "cone", "shuttle", "ras",
        "speed_score", "weight", "bmi", "tested", "capital_score",
    ],
    "WR": [
        "primary_pg_peak", "primary_pg_latest", "primary_pg_trend",
        "efficiency_peak", "efficiency_latest", "td_pg_peak",
        "rec_pg_peak", "dominator_peak", "market_share_peak",
        "pass_share_peak", "touches_peak", "peak_games", "career_games",
        "n_seasons", "age", "breakout_age", "competition_index",
        "cast_quality", "forty", "vertical", "broad", "cone",
        "shuttle", "ras", "speed_score", "height", "weight",
        "tested", "capital_score",
    ],
    "TE": [
        "primary_pg_peak", "primary_pg_latest", "primary_pg_trend",
        "efficiency_peak", "efficiency_latest", "td_pg_peak",
        "rec_pg_peak", "dominator_peak", "market_share_peak",
        "pass_share_peak", "touches_peak", "peak_games", "career_games",
        "n_seasons", "age", "breakout_age", "competition_index",
        "cast_quality", "forty", "vertical", "broad", "cone",
        "shuttle", "ras", "speed_score", "weight", "bmi",
        "tested", "capital_score",
    ],
}

# Disciplined capacity per position (hardening: small samples).
# QB is tightest: ~106 recorded outcomes.
_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "QB": dict(max_leaf_nodes=15, min_samples_leaf=20,
               l2_regularization=5.0, learning_rate=0.06, max_iter=300),
    "RB": dict(max_leaf_nodes=31, min_samples_leaf=20,
               l2_regularization=1.0, learning_rate=0.06, max_iter=400),
    "WR": dict(max_leaf_nodes=31, min_samples_leaf=20,
               l2_regularization=1.0, learning_rate=0.06, max_iter=400),
    "TE": dict(max_leaf_nodes=15, min_samples_leaf=15,
               l2_regularization=2.5, learning_rate=0.06, max_iter=300),
}

MIN_TRAIN_ROWS = 30


def _f(value: Any) -> Optional[float]:
    """Float or None: missing stays missing, never a default."""
    if value is None or value == "" or value == "NA":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def normalize_position(position: Any) -> Optional[str]:
    pos = (position or "").upper()
    pos = _POSITION_ALIASES.get(pos, pos)
    return pos if pos in POSITIONS else None


def _primary_rate(season: Dict, pos: str,
                  metrics: Dict[str, Optional[float]]) -> Optional[float]:
    """The position's primary per-game production rate for a season
    (the same definition v4's season selection peaks on)."""
    if pos == "QB":
        if metrics.get("pass_yds_pg") is not None:
            return metrics["pass_yds_pg"]
        gp = metrics.get("games_played")
        if gp:
            return ((_f(season.get("pass_yards")) or 0.0)
                    + (_f(season.get("rush_yards")) or 0.0)) / gp
        return None
    if pos == "RB":
        return metrics.get("scrimmage_yds_pg")
    return metrics.get("rec_yds_pg")


def _efficiency(metrics: Dict[str, Optional[float]],
                pos: str) -> Optional[float]:
    if pos == "QB":
        return metrics.get("ypa")
    if pos == "RB":
        return metrics.get("ypc")
    return metrics.get("ypr")


def has_college_data(prospect: Dict[str, Any]) -> bool:
    """True when at least one stored season carries stats (the
    abstention rule prices exactly this)."""
    pos = normalize_position(prospect.get("position")) or "WR"
    sel = select_priced_seasons(prospect.get("seasons") or [], pos)
    return sel.get("peak") is not None


def _capital_feature(prospect: Dict[str, Any],
                     consensus_entry: Optional[Dict]) -> Optional[float]:
    """The capital feature: the consensus map's precomputed
    position-adjusted score when present, else derived from the
    projected pick by the same helper v4's capital layer uses."""
    entry = consensus_entry or prospect.get("draft_capital") or {}
    pre = _f(entry.get("projected_draft_capital_score"))
    if pre is not None:
        return pre
    pick = entry.get("projected_pick")
    if pick is None:
        return None
    try:
        from .mock_draft_consensus import pick_to_draft_capital_score
    except ImportError:
        from data_building.rookie_pipeline.mock_draft_consensus import (
            pick_to_draft_capital_score)
    pos = normalize_position(prospect.get("position")) or "WR"
    return _f(pick_to_draft_capital_score(int(pick), pos))


def extract_features(
    prospect: Dict[str, Any],
    consensus_entry: Optional[Dict] = None,
    cast_quality: Optional[float] = None,
    partial_year: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """The shared feature extractor. Works on the prospect dicts
    produced by the backtest harness's _build_prospect_dicts and by
    the live pipeline (the same shape v4 consumes). Every value is
    a raw, uncapped measurement or None: missing stays missing."""
    pos = normalize_position(prospect.get("position")) or "WR"
    seasons = prospect.get("seasons") or []
    sel = select_priced_seasons(seasons, pos, partial_year=partial_year)
    peak, latest = sel.get("peak"), sel.get("latest")
    feats: Dict[str, Optional[float]] = {}

    def season_block(season: Optional[Dict]):
        if season is None:
            return None, None
        m = season_metrics(season, pos)
        return m, _primary_rate(season, pos, m)

    peak_m, peak_primary = season_block(peak)
    latest_m, latest_primary = season_block(latest)

    # Earliest season with a priceable primary rate (trajectory base).
    first_primary = None
    for s in sorted(seasons, key=lambda s: _f(s.get("season")) or 0):
        m = season_metrics(s, pos)
        r = _primary_rate(s, pos, m)
        if r is not None:
            first_primary = r
            break

    feats["primary_pg_peak"] = peak_primary
    feats["primary_pg_latest"] = latest_primary
    feats["primary_pg_first"] = first_primary
    feats["primary_pg_trend"] = (
        latest_primary - first_primary
        if latest_primary is not None and first_primary is not None
        else None)
    feats["efficiency_peak"] = _efficiency(peak_m, pos) if peak_m else None
    feats["efficiency_latest"] = (
        _efficiency(latest_m, pos) if latest_m else None)
    feats["td_pg_peak"] = peak_m.get("tds_pg") if peak_m else None
    feats["rec_pg_peak"] = peak_m.get("rec_pg") if peak_m else None
    feats["dominator_peak"] = (
        peak_m.get("dominator_rating") if peak_m else None)
    feats["market_share_peak"] = (
        peak_m.get("market_share_yards") if peak_m else None)
    feats["pass_share_peak"] = peak_m.get("pass_share") if peak_m else None
    feats["completion_pct_peak"] = (
        peak_m.get("completion_pct") if peak_m else None)
    feats["td_int_peak"] = peak_m.get("td_int") if peak_m else None
    feats["qb_share_peak"] = peak_m.get("qb_share") if peak_m else None
    feats["rush_pg_peak"] = peak_m.get("rush_yds_pg") if peak_m else None
    feats["attempts_peak"] = peak_m.get("pass_attempts") if peak_m else None
    feats["touches_peak"] = peak_m.get("touches") if peak_m else None
    feats["peak_games"] = peak_m.get("games_played") if peak_m else None

    career_gp = 0.0
    n_seasons = 0
    for s in seasons:
        gp = _f(s.get("games_played"))
        if gp:
            career_gp += gp
            n_seasons += 1
    feats["career_games"] = career_gp if n_seasons else None
    feats["n_seasons"] = float(n_seasons) if n_seasons else None

    age = _f(prospect.get("age"))
    feats["age"] = age
    peak_year = _f((peak or {}).get("season"))
    draft_year = _f(prospect.get("draft_class_year"))
    if draft_year is None and latest is not None:
        ly = _f(latest.get("season"))
        draft_year = ly + 1 if ly is not None else None
    if age is not None and peak_year is not None and draft_year is not None:
        # Age during the peak (fall) season: the draft is the spring
        # after the final season, ~0.4y after that season's fall.
        feats["breakout_age"] = age - (draft_year - peak_year - 0.4)
    else:
        feats["breakout_age"] = None

    feats["competition_index"] = competition_index(seasons)
    feats["cast_quality"] = cast_quality

    ath = prospect.get("athleticism") or {}
    forty = _f(ath.get("forty_yard"))
    weight = _f(ath.get("weight_lbs")) or _f(prospect.get("weight_lbs"))
    height = _f(ath.get("height_inches")) or _f(prospect.get("height_inches"))
    feats["forty"] = forty
    feats["vertical"] = _f(ath.get("vertical_inches"))
    feats["broad"] = _f(ath.get("broad_jump_in"))
    feats["cone"] = _f(ath.get("three_cone"))
    feats["shuttle"] = _f(ath.get("short_shuttle"))
    feats["bench"] = _f(ath.get("bench_reps"))
    feats["ras"] = _f(ath.get("ras_score"))
    feats["speed_score"] = (
        weight * 200.0 / forty ** 4 if (forty and weight) else None)
    feats["weight"] = weight
    feats["height"] = height
    feats["bmi"] = (
        703.0 * weight / height ** 2 if (weight and height) else None)
    feats["tested"] = 1.0 if any(
        feats[k] is not None
        for k in ("forty", "vertical", "broad", "cone", "shuttle")) else 0.0

    feats["capital_score"] = _capital_feature(prospect, consensus_entry)
    return feats


def compute_cast_quality(
        prospects: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Raw supporting-cast quality per prospect, from the SAME
    RUN's teammates (shared team-seasons), built from raw season
    production only (no model output), so training and serving
    compute it identically.

    QB: receiving-yards-weighted mean of his RB/WR/TE teammates'
    shared-season receiving yards per game. Others: pass-attempts-
    weighted mean of his QB teammates' shared-season passing yards
    per game. Absent when no teammate link exists."""
    season_maps: Dict[str, Dict[Tuple[str, int], Dict]] = {}
    for p in prospects:
        m: Dict[Tuple[str, int], Dict] = {}
        for s in (p.get("seasons") or []):
            yr = _f(s.get("season"))
            if yr is None:
                continue
            m[((s.get("team") or "").lower().strip(), int(yr))] = s
        season_maps[p.get("player_id")] = m

    out: Dict[str, Optional[float]] = {}
    for p in prospects:
        pos = normalize_position(p.get("position"))
        pid = p.get("player_id")
        mine = season_maps.get(pid) or {}
        if pos is None or not mine:
            out[pid] = None
            continue
        acc = 0.0
        total_w = 0.0
        for other in prospects:
            if other.get("player_id") == pid:
                continue
            opos = normalize_position(other.get("position"))
            if opos is None:
                continue
            shared = None
            for s in (other.get("seasons") or []):
                yr = _f(s.get("season"))
                if yr is None:
                    continue
                key = ((s.get("team") or "").lower().strip(), int(yr))
                if key in mine:
                    shared = s
                    break
            if shared is None:
                continue
            m = season_metrics(shared, opos)
            if pos == "QB" and opos in ("RB", "WR", "TE"):
                rate = m.get("rec_yds_pg")
                w = _f(shared.get("receiving_yards")) or _f(
                    shared.get("receptions")) or 1.0
            elif pos in ("RB", "WR", "TE") and opos == "QB":
                rate = m.get("pass_yds_pg")
                w = _f(shared.get("pass_attempts")) or 1.0
            else:
                continue
            if rate is None:
                continue
            acc += float(rate) * float(w)
            total_w += float(w)
        out[pid] = acc / total_w if total_w > 0 else None
    return out


def feature_vector(feats: Dict[str, Optional[float]],
                   pos: str) -> np.ndarray:
    """One model input row in the position's ordered feature list;
    None becomes NaN (native missing, never imputed)."""
    return np.array(
        [np.nan if feats.get(name) is None else float(feats[name])
         for name in FEATURES_BY_POSITION[pos]], dtype=float)


# ─────────────────────────────────────────────────────────────────────
# Outcome percentile: a predicted PPR/season against the recorded
# outcome ECDF (fit on training rows only, stored in the artifact).
# ─────────────────────────────────────────────────────────────────────


def _ecdf_curve(sorted_outcomes: List[float]):
    """(unique outcome values, plotting-position percentiles) with
    ties averaged: p_i = (rank + 0.5) / n * 100."""
    xs: List[float] = []
    ps: List[float] = []
    n = len(sorted_outcomes)
    i = 0
    arr = sorted(float(v) for v in sorted_outcomes)
    while i < n:
        j = i
        while j + 1 < n and arr[j + 1] == arr[i]:
            j += 1
        xs.append(arr[i])
        ps.append((0.5 * (i + j) + 0.5) / n * 100.0)
        i = j + 1
    return xs, ps


def outcome_percentile(sorted_outcomes: List[float],
                       value: float) -> Optional[float]:
    """Percentile of `value` in the recorded outcome distribution.
    Inside the range, the ECDF interpolates; beyond either end the
    endpoint segment's slope extrapolates (clamped to [0, 100]), so
    a prediction past the recorded maximum can approach 100 while
    an ordinary one cannot."""
    if not sorted_outcomes or value is None:
        return None
    xs, ps = _ecdf_curve(sorted_outcomes)
    if len(xs) == 1:
        return 50.0 if value == xs[0] else (100.0 if value > xs[0] else 0.0)
    if value <= xs[0]:
        slope = (ps[1] - ps[0]) / (xs[1] - xs[0])
        return float(max(0.0, ps[0] + slope * (value - xs[0])))
    if value >= xs[-1]:
        slope = (ps[-1] - ps[-2]) / (xs[-1] - xs[-2])
        return float(min(100.0, ps[-1] + slope * (value - xs[-1])))
    return float(np.interp(value, xs, ps))


# ─────────────────────────────────────────────────────────────────────
# Training. Rows are dicts: {player_id, position, draft_class_year,
# features (the extractor's superset dict), outcome (PPR/season)}.
# ─────────────────────────────────────────────────────────────────────


def _fit_position(pos: str, rows: List[Dict[str, Any]]):
    from sklearn.ensemble import HistGradientBoostingRegressor

    names = FEATURES_BY_POSITION[pos]
    X = np.array([feature_vector(r["features"], pos) for r in rows])
    y = np.array([float(r["outcome"]) for r in rows], dtype=float)
    mono = [MONOTONE.get(name, 0) for name in names]
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        monotonic_cst=mono,
        random_state=RANDOM_SEED,
        **_HYPERPARAMS[pos])
    model.fit(X, y)
    return model, X, y


def _permutation_importance(model, X, y, names) -> Dict[str, float]:
    try:
        from sklearn.inspection import permutation_importance
        res = permutation_importance(
            model, X, y, n_repeats=5, random_state=RANDOM_SEED)
        return {name: round(float(v), 4)
                for name, v in zip(names, res.importances_mean)}
    except Exception:
        return {}


def population_fingerprint(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The training fingerprint recorded in the artifact: who was
    trained on, in a form that changes iff the population does."""
    import sklearn
    lines = sorted(
        f"{r.get('player_id')}|{r.get('position')}|"
        f"{float(r['outcome']):.4f}" for r in rows)
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {
        "population_rows": len(rows),
        "population_hash": digest,
        "classes": sorted({int(r["draft_class_year"]) for r in rows
                           if r.get("draft_class_year") is not None}),
        "feature_lists": {pos: list(names)
                          for pos, names in FEATURES_BY_POSITION.items()},
        "sklearn_version": sklearn.__version__,
        "random_seed": RANDOM_SEED,
        "model_version": MODEL_VERSION,
    }


def fit_artifact(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train every position model on `rows` and assemble the
    artifact (models + outcome ECDFs + importances + fingerprint).
    Positions with fewer than MIN_TRAIN_ROWS rows are left out of
    the artifact; scoring a prospect at such a position abstains."""
    by_pos: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        pos = normalize_position(r.get("position"))
        if pos is not None:
            by_pos.setdefault(pos, []).append(r)
    models: Dict[str, Any] = {}
    ecdfs: Dict[str, List[float]] = {}
    importances: Dict[str, Dict[str, float]] = {}
    for pos, pos_rows in by_pos.items():
        if len(pos_rows) < MIN_TRAIN_ROWS:
            continue
        model, X, y = _fit_position(pos, pos_rows)
        models[pos] = model
        ecdfs[pos] = sorted(float(v) for v in y)
        importances[pos] = _permutation_importance(
            model, X, y, FEATURES_BY_POSITION[pos])
    return {
        "format": ARTIFACT_FORMAT,
        "model_version": MODEL_VERSION,
        "models": models,
        "feature_lists": {pos: list(FEATURES_BY_POSITION[pos])
                          for pos in models},
        "outcome_ecdf": ecdfs,
        "permutation_importance": importances,
        "fingerprint": population_fingerprint(rows),
    }


# ─────────────────────────────────────────────────────────────────────
# Artifact serialization + Postgres storage (migration 057).
# ─────────────────────────────────────────────────────────────────────


class V5ArtifactError(RuntimeError):
    """Raised when v5 is asked to grade without a usable trained
    artifact. v5 fails closed: no artifact, no grades."""


def artifact_to_bytes(artifact: Dict[str, Any]) -> bytes:
    return pickle.dumps(artifact, protocol=4)


def artifact_from_bytes(blob: bytes) -> Dict[str, Any]:
    artifact = pickle.loads(blob)
    if not isinstance(artifact, dict) or artifact.get("format") != ARTIFACT_FORMAT:
        raise V5ArtifactError("stored bytes are not a v5 model artifact")
    if artifact.get("model_version") != MODEL_VERSION:
        raise V5ArtifactError(
            f"artifact model_version {artifact.get('model_version')!r} "
            f"!= {MODEL_VERSION!r}")
    return artifact


_ARTIFACT_DDL = """
CREATE TABLE IF NOT EXISTS prospect_v5_model_artifact (
    id int PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    artifact bytea NOT NULL,
    fingerprint jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
)
"""

_ARTIFACT_UPSERT_SQL = """
INSERT INTO prospect_v5_model_artifact (id, artifact, fingerprint)
VALUES (1, %s, %s::jsonb)
ON CONFLICT (id) DO UPDATE SET
    artifact = EXCLUDED.artifact,
    fingerprint = EXCLUDED.fingerprint,
    created_at = now()
"""

_ARTIFACT_SELECT_SQL = """
SELECT artifact FROM prospect_v5_model_artifact WHERE id = 1
"""


def _connect():
    """Fresh psycopg connection from DATABASE_URL (isolated so
    tests can substitute a fake)."""
    import psycopg

    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise V5ArtifactError(
            "DATABASE_URL is not set: no v5 artifact store reachable")
    return psycopg.connect(url)


def save_artifact_to_db(artifact: Dict[str, Any], conn=None) -> None:
    """Upsert the trained artifact as the single stored row."""
    blob = artifact_to_bytes(artifact)
    fingerprint = json.dumps(artifact.get("fingerprint") or {})
    own = conn is None
    conn = conn or _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_ARTIFACT_DDL)
            cur.execute(_ARTIFACT_UPSERT_SQL, (blob, fingerprint))
        conn.commit()
    finally:
        if own:
            conn.close()


def load_artifact_from_db(conn=None) -> Dict[str, Any]:
    """The stored artifact, or V5ArtifactError when none exists.
    Grading never falls back to an untrained model."""
    own = conn is None
    conn = conn or _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_ARTIFACT_SELECT_SQL)
            row = cur.fetchone()
    except Exception as exc:
        raise V5ArtifactError(
            f"could not read prospect_v5_model_artifact ({exc!r})") from exc
    finally:
        if own:
            conn.close()
    if row is None:
        raise V5ArtifactError(
            "no v5 model artifact is stored yet "
            "(run scripts/train_prospect_model_v5.py first)")
    blob = row[0]
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    return artifact_from_bytes(bytes(blob))


def _resolve_artifact(model_artifact) -> Dict[str, Any]:
    if model_artifact is None:
        return load_artifact_from_db()
    if isinstance(model_artifact, (bytes, bytearray, memoryview)):
        return artifact_from_bytes(bytes(model_artifact))
    if isinstance(model_artifact, dict):
        if model_artifact.get("format") != ARTIFACT_FORMAT:
            raise V5ArtifactError("model_artifact dict is not a v5 artifact")
        return model_artifact
    raise V5ArtifactError(f"unusable model_artifact: {type(model_artifact)!r}")


# ─────────────────────────────────────────────────────────────────────
# Scoring.
# ─────────────────────────────────────────────────────────────────────


def _predict(model, row: np.ndarray) -> float:
    return float(model.predict(row.reshape(1, -1))[0])


def _grade_bundle(artifact: Dict[str, Any], pos: str,
                  feats: Dict[str, Optional[float]]):
    """(grade, talent_grade, capital_delta, pred_outcome, percentile)
    for one feature set under the artifact. talent = capital-blind
    (capital feature set to missing)."""
    model = artifact["models"][pos]
    ecdf = artifact["outcome_ecdf"][pos]
    x = feature_vector(feats, pos)
    pred = max(0.0, _predict(model, x))
    pct = outcome_percentile(ecdf, pred)
    grade = grade_for_percentile(pct)
    blind_feats = dict(feats)
    blind_feats["capital_score"] = None
    if feats.get("capital_score") is None:
        blind_grade = grade
    else:
        blind_pred = max(0.0, _predict(model, feature_vector(blind_feats, pos)))
        blind_grade = grade_for_percentile(outcome_percentile(ecdf, blind_pred))
    delta = (grade - blind_grade) if (grade is not None
                                      and blind_grade is not None) else 0.0
    return grade, blind_grade, delta, pred, pct, x


def _top_features(artifact: Dict[str, Any], pos: str,
                  feats: Dict[str, Optional[float]],
                  grade: float) -> List[Dict[str, Any]]:
    """Top-3 contributing features by drop-one delta: re-grade with
    each present feature set to missing; the delta is how much of
    the grade that feature carries (signed)."""
    model = artifact["models"][pos]
    ecdf = artifact["outcome_ecdf"][pos]
    names = FEATURES_BY_POSITION[pos]
    x = feature_vector(feats, pos)
    deltas = []
    for j, name in enumerate(names):
        if np.isnan(x[j]):
            continue
        x2 = x.copy()
        x2[j] = np.nan
        g2 = grade_for_percentile(
            outcome_percentile(ecdf, max(0.0, _predict(model, x2))))
        if g2 is not None:
            deltas.append({"feature": name, "delta": round(grade - g2, 2)})
    deltas.sort(key=lambda d: -abs(d["delta"]))
    return deltas[:3]


def _score_one(prospect, consensus_entry, artifact, cast_quality,
               cycle_complete, receipts=True):
    pos = normalize_position(prospect.get("position"))
    state = detect_lifecycle_state(prospect, None, cycle_complete)
    base: Dict[str, Any] = {
        "player_id": prospect.get("player_id"),
        "name": prospect.get("name"),
        "position": (prospect.get("position") or "").upper(),
        "model_version": MODEL_VERSION,
        "lifecycle_state": state,
        "in_season": state == "in_season",
        "prospect_score": None,
        "talent_score": None,
        "capital_delta": 0.0,
        "projected_draft_capital_score": None,
        "band": None,
        "band_label": None,
        "production_score": None,
        "efficiency_score": None,
        "breakout_profile_score": None,
        "age_score": None,
        "athleticism_score": None,
        "competition_score": None,
        "recruiting_score": None,
        "durability_score": None,
        "weights_used": {},
        "legs_missing": [],
        "seasons_priced": {"peak": None, "latest": None},
        "cast": {},
        "athleticism_receipt": {},
        "recruiting_receipt": {},
        "display_percentiles": {},
        "confidence_score": None,
        "reference_kind": "trained_model",
        "top_features": [],
        "predicted_ppr_season": None,
        "predicted_outcome_percentile": None,
        "reasons": [],
    }
    if pos is None or pos not in (artifact.get("models") or {}):
        base["reasons"] = [
            "Unscored under v5: no trained model for this position"]
        return base
    sel = select_priced_seasons(prospect.get("seasons") or [], pos)
    if sel.get("peak") is None:
        base["reasons"] = [
            "Unscored under v5: no college season data (abstention)"]
        return base
    base["seasons_priced"] = {
        "peak": (sel["peak"] or {}).get("season"),
        "latest": (sel["latest"] or {}).get("season"),
    }
    feats = extract_features(prospect, consensus_entry,
                             cast_quality=cast_quality)
    grade, talent, delta, pred, pct, _x = _grade_bundle(artifact, pos, feats)
    band_key, band_label = band_for(grade)
    base.update({
        "position": pos,
        "prospect_score": None if grade is None else round(grade, 2),
        "talent_score": None if talent is None else round(talent, 2),
        "capital_delta": round(delta, 2),
        "projected_draft_capital_score": (
            None if feats.get("capital_score") is None
            else round(float(feats["capital_score"]), 2)),
        "band": band_key,
        "band_label": band_label,
        "cast": ({"cast_quality": round(float(cast_quality), 2)}
                 if cast_quality is not None else {}),
        "athleticism_receipt": {
            "source": "tested" if feats.get("tested") else "not_tested"},
        "predicted_ppr_season": round(pred, 2),
        "predicted_outcome_percentile": (
            None if pct is None else round(pct, 2)),
    })
    confidence = 100.0
    if feats.get("capital_score") is None:
        confidence -= 6.0
    if not feats.get("tested"):
        confidence -= 4.0
    if cast_quality is None:
        confidence -= 3.0
    for core in ("primary_pg_peak", "efficiency_peak"):
        if feats.get(core) is None:
            confidence -= 2.0
    base["confidence_score"] = int(max(5.0, min(100.0, confidence)))

    top = _top_features(artifact, pos, feats, grade) if receipts else []
    base["top_features"] = top
    reasons = [f"{band_label} band ({state.replace('_', '-')})"]
    reasons.append(
        f"Talent (capital-blind) {talent:.2f}; capital "
        f"{delta:+.2f} to {grade:.2f}")
    if top:
        reasons.append("Top drivers: " + ", ".join(
            f"{t['feature']} ({t['delta']:+.2f})" for t in top))
    base["reasons"] = reasons
    return base


def score_all_v5(
    prospects: List[Dict[str, Any]],
    consensus_map: Optional[Dict[str, Dict]] = None,
    model_artifact: Optional[Any] = None,
    cycle_complete: bool = False,
    receipts: bool = True,
) -> List[Dict[str, Any]]:
    """Grade a run of prospects with the trained v5 artifact.

    Mirrors score_all_v4's contract: one result row per prospect,
    sorted by grade descending with overall_rank / position_rank
    stamped. Abstains (prospect_score None) on prospects with no
    college season data and on positions with no trained model.
    Fails closed (V5ArtifactError) when no artifact is supplied or
    stored."""
    artifact = _resolve_artifact(model_artifact)
    consensus_map = consensus_map or {}
    cast = compute_cast_quality(prospects)
    results = []
    for p in prospects:
        results.append(_score_one(
            p, consensus_map.get(p.get("player_id")), artifact,
            cast.get(p.get("player_id")), cycle_complete,
            receipts=receipts))
    results.sort(key=lambda x: (x["prospect_score"] is None,
                                -(x["prospect_score"] or 0.0)))
    pos_counters: Dict[str, int] = {}
    for i, r in enumerate(results):
        r["overall_rank"] = i + 1
        pos_counters[r["position"]] = pos_counters.get(r["position"], 0) + 1
        r["position_rank"] = pos_counters[r["position"]]
    return results
