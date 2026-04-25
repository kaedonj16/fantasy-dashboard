"""
XGBoost rookie prospect scorer.

Replaces all hand-crafted rules in prospect_model.py with gradient-boosted
trees trained on every available college stat, combine metric, and draft
position.  One model per position (WR, RB, QB, TE).

Features (32 total):
  Draft capital : draft_pick, log_draft_pick
  Age           : age_at_draft
  College stats : rec_yds_pg, rec_tds_pg, rec_pg, rush_yds_pg, rush_tds_pg,
                  all_yds_pg, tds_pg, pass_yds_pg, pass_tds_pg,
                  completion_pct, ypa, td_int_ratio, dominator_rating,
                  pass_share, yac_per_rec, ypc
  Context       : conf_quality, num_seasons
  Combine       : forty_yard, vertical_inches, broad_jump_in, three_cone,
                  short_shuttle, weight_lbs, height_inches, ras_score

Target: PPR fantasy points per active NFL season (normalized to 0-100).

Training
--------
    python scripts/train_ml_model.py

Inference
---------
    from data_building.rookie_pipeline.ml_model import score_all_prospects_ml
    results = score_all_prospects_ml(prospects, consensus_map)
"""
from __future__ import annotations

import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.impute import SimpleImputer

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _XGB = False

_MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "ml_prospect_models.pkl")

POSITIONS = ["WR", "RB", "QB", "TE"]

# ── Conference quality ─────────────────────────────────────────────────────────
_CQ: Dict[str, float] = {
    "sec": 1.00, "southeastern": 1.00,
    "big ten": 1.00, "big 10": 1.00,
    "big 12": 0.90, "big twelve": 0.90,
    "acc": 0.88, "atlantic coast": 0.88,
    "pac-12": 0.89, "pac 12": 0.89, "pacific-12": 0.89,
    "notre dame": 0.94,
    "american": 0.78, "american athletic": 0.78,
    "mountain west": 0.70,
    "sun belt": 0.66,
    "mac": 0.60, "mid-american": 0.60,
    "cusa": 0.56, "conference usa": 0.56,
    "fcs": 0.48,
}

def _cq(conf: Optional[str]) -> float:
    if not conf:
        return 0.72
    cl = conf.lower()
    for k, v in _CQ.items():
        if k in cl:
            return v
    return 0.72


def _sf(v: Any, d: float = 0.0) -> float:
    try:
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


# ── Feature schema (order must match between training and inference) ───────────
FEATURE_NAMES: List[str] = [
    "draft_pick",
    "log_draft_pick",
    "age_at_draft",
    "rec_yds_pg",
    "rec_tds_pg",
    "rec_pg",
    "rush_yds_pg",
    "rush_tds_pg",
    "all_yds_pg",
    "tds_pg",
    "pass_yds_pg",
    "pass_tds_pg",
    "completion_pct",
    "ypa",
    "td_int_ratio",
    "dominator_rating",
    "pass_share",
    "yac_per_rec",
    "ypc",
    "conf_quality",
    "num_seasons",
    "forty_yard",
    "vertical_inches",
    "broad_jump_in",
    "three_cone",
    "short_shuttle",
    "weight_lbs",
    "height_inches",
    "ras_score",
]

N_FEATURES = len(FEATURE_NAMES)
_FIDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(
    prospect: Dict[str, Any],
    consensus: Optional[Dict[str, Any]],
) -> np.ndarray:
    """
    Convert a prospect dict + consensus entry into a (1, N_FEATURES) array.
    Missing values become NaN and are handled by the model's imputer.
    """
    seasons = prospect.get("seasons") or []
    ath     = prospect.get("athleticism") or {}

    latest: Dict = {}
    if seasons:
        latest = max(seasons, key=lambda s: _sf(s.get("season"), 0))

    gp = max(_sf(latest.get("games_played"), 12.0), 1.0)

    pick = _sf((consensus or {}).get("projected_pick"), 300.0) or 300.0

    rec_yds  = _sf(latest.get("receiving_yards"))
    rec_tds  = _sf(latest.get("receiving_tds"))
    rec_cnt  = _sf(latest.get("receptions"))
    rush_yds = _sf(latest.get("rush_yards"))
    rush_tds = _sf(latest.get("rush_tds"))
    pass_yds = _sf(latest.get("pass_yards"))
    pass_tds = _sf(latest.get("pass_tds"))
    team_py  = _sf(latest.get("team_pass_yards"))

    def _opt(key: str) -> float:
        v = latest.get(key)
        return _sf(v) if v is not None else float("nan")

    vec = np.full(N_FEATURES, np.nan, dtype=np.float32)

    def _s(name: str, val: float) -> None:
        if name in _FIDX:
            vec[_FIDX[name]] = val

    _s("draft_pick",      pick)
    _s("log_draft_pick",  math.log(max(pick, 1)))
    _s("age_at_draft",    _sf(prospect.get("age")) if prospect.get("age") else float("nan"))
    _s("rec_yds_pg",      rec_yds / gp)
    _s("rec_tds_pg",      rec_tds / gp)
    _s("rec_pg",          rec_cnt / gp)
    _s("rush_yds_pg",     rush_yds / gp)
    _s("rush_tds_pg",     rush_tds / gp)
    _s("all_yds_pg",      (rec_yds + rush_yds) / gp)
    _s("tds_pg",          (rec_tds + rush_tds) / gp)
    _s("pass_yds_pg",     pass_yds / gp)
    _s("pass_tds_pg",     pass_tds / gp)
    _s("completion_pct",  _opt("completion_pct"))
    _s("ypa",             _opt("yds_per_attempt"))
    _s("td_int_ratio",    _opt("td_int_ratio"))
    _s("dominator_rating",_opt("dominator_rating"))
    _s("pass_share",      (rec_yds / team_py) if team_py > 0 else float("nan"))
    _s("yac_per_rec",     _opt("yards_after_catch_per_reception"))
    _s("ypc",             _opt("yds_per_carry"))
    _s("conf_quality",    _cq(latest.get("conference") or prospect.get("conference")))
    _s("num_seasons",     float(len(seasons)))

    # ── Fallback: fill missing college stats from rookie_profile.metrics ─────
    # When CFBD data is unavailable (no API key), use PFF/Sportradar metrics
    # stored in rookie_profile.metrics.  These use different units/scales than
    # CFBD but the imputer will fill remaining NaNs with training medians.
    if not seasons:
        rp      = prospect.get("rookie_profile") or {}
        pm      = rp.get("metrics") or {}

        def _pmv(key: str) -> float:
            m = pm.get(key)
            if m is None:
                return float("nan")
            return _sf(m.get("value") if isinstance(m, dict) else m)

        pm_gp      = _pmv("games_played")
        pm_routes  = _pmv("routes_run")
        pm_yprr    = _pmv("yprr")   # yards per route run
        pm_tprr    = _pmv("tprr")   # targets per route run
        pm_yac     = _pmv("yac_per_att")   # yac per target ≈ per catch
        pm_snap    = _pmv("snap_counts")
        pm_sos     = _pmv("player_level_sos")  # 0-1 strength-of-schedule

        if not math.isnan(pm_gp) and pm_gp > 0:
            gp = pm_gp

        # Reconstruct per-game production from route-level stats
        if not (math.isnan(pm_yprr) or math.isnan(pm_routes)):
            est_rec_yds = pm_yprr * pm_routes
            if not math.isnan(pm_gp) and pm_gp > 0:
                _s("rec_yds_pg", est_rec_yds / pm_gp)
                _s("all_yds_pg", est_rec_yds / pm_gp)

        if not (math.isnan(pm_tprr) or math.isnan(pm_routes)):
            est_recs = pm_tprr * pm_routes
            if not math.isnan(pm_gp) and pm_gp > 0:
                _s("rec_pg", est_recs / pm_gp)

        if not math.isnan(pm_yac):
            _s("yac_per_rec", pm_yac)

        # Strength-of-schedule → conf_quality proxy (SOS 1.0 = toughest)
        if not math.isnan(pm_sos) and pm_sos > 0:
            _s("conf_quality", min(1.0, pm_sos))

        # Snap-based num_seasons proxy: 1 if any snaps, else 0
        if not math.isnan(pm_snap) and pm_snap > 0:
            _s("num_seasons", 1.0)

    # Combine
    for feat, keys in [
        ("forty_yard",      ["forty_yard", "forty"]),
        ("vertical_inches", ["vertical_inches", "vertical"]),
        ("broad_jump_in",   ["broad_jump_in", "broad_jump"]),
        ("three_cone",      ["three_cone", "cone"]),
        ("short_shuttle",   ["short_shuttle", "shuttle"]),
        ("weight_lbs",      ["weight_lbs", "weight"]),
        ("height_inches",   ["height_inches", "height"]),
        ("ras_score",       ["ras_score"]),
    ]:
        for k in keys:
            v = ath.get(k)
            if v is not None:
                _s(feat, _sf(v))
                break

    return vec.reshape(1, -1)


# ── Per-position model bundle ──────────────────────────────────────────────────

class _PositionModel:
    """
    One XGBoost (or GBR fallback) + imputer + calibration stats per position.
    """
    def __init__(self, position: str) -> None:
        self.position   = position
        self.imputer    = SimpleImputer(strategy="median")
        self.model      = self._make_model()
        self.p5: float  = 0.0    # 5th-percentile training target (maps → score 0)
        self.p95: float = 300.0  # 95th-percentile (maps → score 100)
        self.trained    = False

    def _make_model(self):
        if _XGB:
            return xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_lambda=1.5,
                random_state=42,
                verbosity=0,
                tree_method="hist",
            )
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        X_imp = self.imputer.fit_transform(X)
        if _XGB:
            self.model.fit(X_imp, y, sample_weight=sample_weight)
        else:
            self.model.fit(X_imp, y, sample_weight=sample_weight)
        self.p5  = float(np.percentile(y, 5))
        self.p95 = float(np.percentile(y, 95))
        self.trained = True

    def predict_score(self, X: np.ndarray) -> float:
        """Return 0-100 score for a single prospect."""
        X_imp = self.imputer.transform(X)
        raw = float(self.model.predict(X_imp)[0])
        span = self.p95 - self.p5
        if span <= 0:
            return 50.0
        score = (raw - self.p5) / span * 100.0
        return max(0.0, min(100.0, score))

    def feature_importance(self) -> List[Tuple[str, float]]:
        if _XGB:
            imps = self.model.feature_importances_
        else:
            imps = self.model.feature_importances_
        pairs = sorted(
            zip(FEATURE_NAMES, imps),
            key=lambda x: x[1], reverse=True
        )
        return [(n, float(v)) for n, v in pairs if v > 0.001]


# ── Scorer ─────────────────────────────────────────────────────────────────────

class MLProspectScorer:
    """
    Loads (or holds in memory) per-position XGBoost models and scores prospects.
    """
    def __init__(self) -> None:
        self.models: Dict[str, _PositionModel] = {
            pos: _PositionModel(pos) for pos in POSITIONS
        }

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = _MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[ml_model] Models saved → {path}")

    @classmethod
    def load(cls, path: str = _MODEL_PATH) -> "MLProspectScorer":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(path, "rb") as f:
                obj = pickle.load(f)
        # Patch sklearn version skew: 1.6→1.8 renamed _fit_dtype→_fill_dtype
        for pos_model in obj.models.values():
            imp = pos_model.imputer
            if hasattr(imp, "_fit_dtype") and not hasattr(imp, "_fill_dtype"):
                imp._fill_dtype = imp._fit_dtype
        return obj

    @classmethod
    def is_trained(cls, path: str = _MODEL_PATH) -> bool:
        return os.path.exists(path)

    # ── training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        training_rows: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train per-position models from a list of training rows.

        Each row must have:
            prospect   : Dict (full prospect dict from pipeline)
            consensus  : Dict (consensus/pick entry)
            ppr_avg    : float (PPR per season — the target)
            seasons_avail : int (used for sample weighting)

        Returns: dict with per-position training stats.
        """
        by_pos: Dict[str, Tuple[List, List, List]] = {
            pos: ([], [], []) for pos in POSITIONS
        }

        for row in training_rows:
            pos = (row.get("position") or "").upper()
            if pos not in POSITIONS:
                continue
            ppr_avg = row.get("ppr_avg")
            if ppr_avg is None or ppr_avg < 0:
                continue

            prospect  = row.get("_prospect") or {}
            consensus = row.get("_consensus") or {}
            n_seasons = max(row.get("seasons_avail", 1), 1)

            feats = extract_features(prospect, consensus)
            by_pos[pos][0].append(feats)
            by_pos[pos][1].append(float(ppr_avg))
            # Weight by seasons of NFL data: more seasons = more reliable signal
            w = min(4.0, float(n_seasons)) / 4.0
            by_pos[pos][2].append(w)

        stats: Dict[str, Any] = {}
        for pos in POSITIONS:
            Xs, ys, ws = by_pos[pos]
            if len(Xs) < 20:
                if verbose:
                    print(f"[ml_model] {pos}: only {len(Xs)} samples — skipping")
                continue
            X = np.vstack(Xs)
            y = np.array(ys, dtype=np.float32)
            w = np.array(ws, dtype=np.float32)
            self.models[pos].fit(X, y, sample_weight=w)
            if verbose:
                print(f"[ml_model] {pos}: trained on {len(ys)} samples  "
                      f"target=[{y.min():.0f}, {y.max():.0f}]  "
                      f"p5={self.models[pos].p5:.0f}  p95={self.models[pos].p95:.0f}")
                top5 = self.models[pos].feature_importance()[:5]
                print(f"           top features: " +
                      "  ".join(f"{n}={v:.3f}" for n, v in top5))
            stats[pos] = {"n": len(ys), "p5": self.models[pos].p5, "p95": self.models[pos].p95}
        return stats

    # ── inference ────────────────────────────────────────────────────────────

    def score_prospects(
        self,
        prospects: List[Dict[str, Any]],
        consensus_map: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Score a list of prospect dicts.  Returns the same format as
        score_all_prospects() in prospect_model.py so it's a drop-in replacement.
        """
        # Sort by position so within-position ranks are computed correctly
        by_pos: Dict[str, List] = {pos: [] for pos in POSITIONS}
        for p in prospects:
            pos = (p.get("position") or "WR").upper()
            by_pos.setdefault(pos, []).append(p)

        results: List[Dict[str, Any]] = []
        overall_scores: List[Tuple[str, float]] = []  # (player_id, score)

        for pos in POSITIONS:
            model = self.models[pos]
            pos_prospects = by_pos.get(pos, [])
            if not pos_prospects:
                continue

            scored = []
            for p in pos_prospects:
                pid  = p.get("player_id", p.get("name", "unknown"))
                cons = consensus_map.get(pid)
                feats = extract_features(p, cons)

                if model.trained:
                    score = model.predict_score(feats)
                else:
                    # Model not trained yet — fall back to draft capital proxy
                    pick  = _sf((cons or {}).get("projected_pick"), 200.0)
                    base_score = max(0.0, 100.0 - (pick / 260.0) * 100.0)
                    
                    # Data completeness check: prevent artificial inflation from incomplete data
                    seasons = p.get("seasons", [])
                    meaningful_components = 0
                    if seasons:
                        latest = max(seasons, key=lambda s: _sf(s.get("season", 0)))
                        # Check for meaningful production data
                        if _sf(latest.get("receiving_yards", 0)) > 200: meaningful_components += 1
                        if _sf(latest.get("rush_yards", 0)) > 300: meaningful_components += 1
                        if _sf(latest.get("pass_yards", 0)) > 1000: meaningful_components += 1
                        if _sf(latest.get("dominator_rating", 0)) > 0.15: meaningful_components += 1
                    
                    # Only allow high scores for complete data profiles
                    if meaningful_components >= 2:
                        score = min(100.0, base_score)
                    else:
                        # Incomplete data: cap at 85.0
                        score = min(85.0, base_score * 0.9)

                scored.append((p, cons, score))
                overall_scores.append((pid, score))

            # Within-position rank
            scored.sort(key=lambda x: x[2], reverse=True)
            for pos_rank, (p, cons, score) in enumerate(scored, 1):
                pid = p.get("player_id", p.get("name", "unknown"))
                results.append(_build_result(p, cons, score, pos_rank))

        # Overall rank
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        overall_rank_map = {pid: rank for rank, (pid, _) in enumerate(overall_scores, 1)}
        for r in results:
            r["overall_rank"] = overall_rank_map.get(r["player_id"], 999)

        results.sort(key=lambda r: r["overall_rank"])
        return results


def _build_result(
    prospect: Dict[str, Any],
    consensus: Optional[Dict[str, Any]],
    score: float,
    pos_rank: int,
) -> Dict[str, Any]:
    """Build a result dict matching the format expected by the pipeline and DB."""
    seasons = prospect.get("seasons") or []
    ath     = prospect.get("athleticism") or {}

    latest: Dict = {}
    if seasons:
        latest = max(seasons, key=lambda s: _sf(s.get("season"), 0))
    gp = max(_sf(latest.get("games_played"), 12.0), 1.0)

    rec_yds = _sf(latest.get("receiving_yards"))
    conf_q  = _cq(latest.get("conference") or prospect.get("conference"))
    ras     = _sf(ath.get("ras_score")) if ath.get("ras_score") is not None else None

    # Map the single ML score back to a confidence value (higher score = more confident)
    confidence = max(30.0, min(90.0, score * 0.8 + 10.0))

    return {
        # Identity
        "player_id":                    prospect.get("player_id", ""),
        "name":                         prospect.get("name", ""),
        "position":                     (prospect.get("position") or "").upper(),
        "draft_class_year":             prospect.get("draft_class_year"),
        # Ranks — overall_rank filled in after full sort
        "overall_rank":                 0,
        "position_rank":                pos_rank,
        # Primary score
        "prospect_score":               round(score, 2),
        "tier":                         _tier(score),
        "confidence_score":             round(confidence, 1),
        # Component scores — ML model doesn't decompose; store None so the DB
        # stores NULL rather than stale rule-based values.
        "projected_draft_capital_score": _sf((consensus or {}).get("projected_draft_capital_score")),
        "production_score":             None,
        "efficiency_score":             None,
        "age_score":                    None,
        "breakout_profile_score":       None,
        "athleticism_score":            round(ras * 10.0, 1) if ras is not None else None,
        "competition_score":            round(conf_q * 100.0, 1),
        "utilization_score":            None,
        "environment_adjustment":       None,
        "durability_score":             None,
        "fantasy_translation_score":    None,
        "key_reasons":                  None,
        # Dynasty value (kept simple — translate_all will overwrite with full calc)
        "rookie_value":                 round(score, 2),
        "dynasty_value":                round(score, 2),
    }


def _tier(score: float) -> int:
    if score >= 85:  return 1
    if score >= 70:  return 2
    if score >= 55:  return 3
    if score >= 40:  return 4
    return 5


# ── Drop-in replacement for score_all_prospects ────────────────────────────────

_cached_scorer: Optional[MLProspectScorer] = None


def score_all_prospects_ml(
    prospects: List[Dict[str, Any]],
    consensus_map: Dict[str, Any],
    skip_sagarin: bool = True,
    position_weights_override: Optional[Dict] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Drop-in replacement for prospect_model.score_all_prospects().

    Loads the trained ML model on first call and caches it.  Falls back
    to the rule-based model if the pkl hasn't been created yet.

    `position_weights_override` and extra kwargs are forwarded to the
    rule-based fallback only (the ML model ignores them — it learned
    weights from data).
    """
    global _cached_scorer

    if not MLProspectScorer.is_trained():
        from data_building.rookie_pipeline.prospect_model import score_all_prospects
        return score_all_prospects(
            prospects, consensus_map,
            skip_sagarin=skip_sagarin,
            position_weights_override=position_weights_override,
        )

    if _cached_scorer is None:
        try:
            _cached_scorer = MLProspectScorer.load()
            print("[ml_model] ML model loaded ✓")
        except Exception as e:
            print(f"[ml_model] Failed to load model ({e}) — falling back to rule-based")
            from data_building.rookie_pipeline.prospect_model import score_all_prospects
            return score_all_prospects(
                prospects, consensus_map,
                skip_sagarin=skip_sagarin,
                position_weights_override=position_weights_override,
            )

    return _cached_scorer.score_prospects(prospects, consensus_map)
