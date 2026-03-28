# dashboard_services/value_model_training.py

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pickle
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Dict, List, Optional

from dashboard_services.api import get_nfl_state
from dashboard_services.picks import load_pick_value_table
from data_building.external_data.player_history import load_player_history_df, build_player_history_features
from data_building.external_data.player_investment import load_player_investment_context
from utils.paths import DATA_DIR
from utils.utils import load_teams_index, bucket_for_slot, normalize_name, load_players_index

# ------------------------------------------------
# Paths / constants
# ------------------------------------------------

MODEL_PATH = DATA_DIR / "trade_value_model.pkl"
DYNASTYPROCESS_VALUES_PATH = DATA_DIR / f"dynastyprocess_values_{date.today().isoformat()}.csv"
FANTASYCALC_VALUES_PATH = DATA_DIR / f"fantasycalc_api_values_{date.today().isoformat()}.csv"
ENGINE_VALUES_PATH = DATA_DIR / f"engine_values_{date.today().isoformat()}.csv"

FANTASYCALC_URL = (
    "https://api.fantasycalc.com/values/current"
    "?isDynasty=true&numQbs=1&numTeams=10&ppr=1"
)

CORE_POSITIONS = {"QB", "RB", "WR", "TE"}


# ------------------------------------------------
# Trained model bundle
# ------------------------------------------------

@dataclass
class TrainedModelBundle:
    pipeline: Pipeline
    scale_min: float
    scale_max: float
    feature_columns: List[str]


# ------------------------------------------------
# Small helpers
# ------------------------------------------------

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, str) and not x.strip():
            return float(default)
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, str) and not x.strip():
            return int(default)
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _current_season_from_state() -> int:
    state = get_nfl_state() or {}
    season = state.get("season")
    if season is None:
        return date.today().year
    return _safe_int(season, date.today().year)


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_columns(df: pd.DataFrame, cols: List[str], fill_value=0.0) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


# ------------------------------------------------
# Season helpers
# ------------------------------------------------

def get_training_seasons(current_season: int, num_past_seasons: int = 2) -> list[int]:
    current_season = int(current_season)
    start = current_season - num_past_seasons
    return list(range(start, current_season + 1))


# ------------------------------------------------
# History feature loader
# ------------------------------------------------

def load_history_feature_df(current_season: int) -> pd.DataFrame:
    seasons = get_training_seasons(current_season, 2)

    frames = []
    for season in seasons:
        try:
            season_df = load_player_history_df(season)
            if season_df is not None and not season_df.empty:
                frames.append(season_df)
        except FileNotFoundError:
            continue

    if not frames:
        return pd.DataFrame()

    history_df = pd.concat(frames, ignore_index=True)
    return build_player_history_features(history_df)


def load_player_investment_df() -> pd.DataFrame:
    df = load_player_investment_context()
    if df is None or df.empty:
        return pd.DataFrame()

    keep_cols = [
        "sleeper_id",
        "draft_year",
        "draft_round",
        "draft_pick",
        "draft_capital_score",
        "draft_capital_pos_pct",
        "contract_total_value",
        "contract_apy",
        "guaranteed_money",
        "fully_guaranteed_money",
        "guaranteed_pct",
        "contract_score",
        "team_investment_score",
        "years_to_fa",
        "contract_apy_pos_pct",
        "guaranteed_money_pos_pct",
        "guaranteed_pct_pos_pct",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    out = df[keep_cols].copy()
    if "sleeper_id" in out.columns:
        out["sleeper_id"] = out["sleeper_id"].astype(str)
    return out


# ------------------------------------------------
# Internal stats loader
# ------------------------------------------------

def load_internal_stats_df() -> pd.DataFrame:
    value_path = DATA_DIR / f"usage_table_{date.today().isoformat()}.json"

    if not value_path.exists():
        raise FileNotFoundError(f"No internal value table found at {value_path}")

    with value_path.open("r", encoding="utf-8") as f:
        players = json.load(f)

    df = pd.json_normalize(players)

    rename_map = {
        "id": "sleeper_id",
        "value": "internal_value_raw",
        "usage.games": "games",
        "usage.avg_off_snap_pct": "avg_off_snap_pct",
        "usage.avg_off_snaps": "avg_off_snaps",
        "usage.avg_targets": "avg_targets",
        "usage.avg_receptions": "avg_receptions",
        "usage.avg_rec_yards": "avg_rec_yards",
        "usage.avg_rec_tds": "avg_rec_tds",
        "usage.avg_carries": "avg_carries",
        "usage.avg_rush_yards": "avg_rush_yards",
        "usage.avg_rush_tds": "avg_rush_tds",
        "usage.ppr_ppg": "ppr_ppg",
        "usage.half_ppr_ppg": "half_ppr_ppg",
        "usage.std_scoring_ppg": "std_scoring_ppg",
        "usage.std_ppg": "std_ppg",
        "usage.rec_rz_tgt_pg": "rec_rz_tgt_pg",
        "usage.rush_rz_att_pg": "rush_rz_att_pg",
        "usage.avg_pass_att": "avg_pass_att",
        "usage.avg_pass_cmp": "avg_pass_cmp",
        "usage.avg_pass_yds": "avg_pass_yds",
        "usage.avg_pass_tds": "avg_pass_tds",
        "usage.avg_pass_int": "avg_pass_int",
        "usage.target_share": "target_share",
        "usage.target_share_pct": "target_share_pct",
        "usage.total_targets": "total_targets",
    }

    df = df.rename(columns=rename_map)

    if "sleeper_id" not in df.columns:
        raise ValueError("usage_table is missing 'id'/'sleeper_id'")

    df["sleeper_id"] = df["sleeper_id"].astype(str)
    df["internal_value"] = df.get("internal_value_raw", np.nan)

    teams_index = load_teams_index() or {}

    team_rows = []
    for abbr, meta in teams_index.items():
        team_rows.append(
            {
                "team": abbr,
                "team_pass_att_pg": meta.get("pass_att_pg"),
                "team_off_snaps_pg": meta.get("off_snaps_pg"),
                "team_games_tracked": meta.get("games_tracked"),
                "team_rush_att_pg": meta.get("rush_att_pg"),
                "team_rush_yds_pg": meta.get("rush_yds_pg"),
                "team_pass_yds_pg": meta.get("pass_yds_pg"),
            }
        )

    team_df = pd.DataFrame(team_rows)
    internal_df = df.merge(team_df, on="team", how="left")

    return internal_df


# ------------------------------------------------
# FantasyCalc loader
# ------------------------------------------------

def load_fantasycalc_df(path: Path = FANTASYCALC_VALUES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected FantasyCalc CSV at {path}. "
            "Export api data into fantasycalc_api_values.csv first."
        )

    df_raw = pd.read_csv(path)

    sid_col = _pick_col(df_raw, [
        "sleeper_id", "id", "player_id", "sleeperId", "player.sleeperId"
    ])
    val_col = _pick_col(df_raw, [
        "fc_value", "value", "dynasty_value", "player_value"
    ])

    if sid_col is None or val_col is None:
        raise ValueError(
            f"Could not find sleeper_id/value columns in {path}. "
            f"Columns present: {list(df_raw.columns)}"
        )

    name_col = _pick_col(df_raw, ["name", "player", "player_name", "player.name"])
    pos_col = _pick_col(df_raw, ["position", "pos", "player.position"])
    team_col = _pick_col(df_raw, ["team", "maybeTeam", "player.team"])
    rank_col = _pick_col(df_raw, ["fc_rank", "overallRank", "rank", "overall_rank"])
    age_col = _pick_col(df_raw, ["fc_age", "maybeAge", "age", "player_age"])

    rows = []
    for _, r in df_raw.iterrows():
        sid = r.get(sid_col)
        if pd.isna(sid):
            continue

        rows.append(
            {
                "sleeper_id": str(sid),
                "name": r.get(name_col) if name_col else None,
                "position": r.get(pos_col) if pos_col else None,
                "team": r.get(team_col) if team_col else None,
                "fc_value": r.get(val_col),
                "fc_rank": r.get(rank_col) if rank_col else None,
                "fc_age": r.get(age_col) if age_col else None,
            }
        )

    df = pd.DataFrame(rows)
    return df


# ------------------------------------------------
# DynastyProcess loader
# ------------------------------------------------

def _suffix(rnd: int) -> str:
    if rnd == 1:
        return "st"
    if rnd == 2:
        return "nd"
    if rnd == 3:
        return "rd"
    return "th"


def load_dynastyprocess_df(
        path: Path = DYNASTYPROCESS_VALUES_PATH,
        pick_value_lookup: Dict[str, float] = None,
        years=(2025, 2026, 2027, 2028),
        rounds=(1, 2, 3),
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing DP CSV at {path}")

    df = pd.read_csv(path)

    dp_value_col = "value_1qb"
    if dp_value_col not in df.columns:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No numeric value columns found in {path}")
        dp_value_col = "value" if "value" in numeric_cols else numeric_cols[0]

    name_col = "player" if "player" in df.columns else (
        "Player" if "Player" in df.columns else df.columns[0]
    )
    pos_col = "pos" if "pos" in df.columns else (
        "position" if "position" in df.columns else None
    )
    team_col = "team" if "team" in df.columns else None

    names = df[name_col].astype(str)
    positions = df[pos_col].astype(str) if pos_col else pd.Series([""] * len(df))

    def looks_like_pick(name: str):
        s = name.lower().strip()
        return bool(re.match(r"^\d{4}\s+(early|mid|late)\s+\d+(st|nd|rd|th)$", s))

    pick_mask = names.apply(looks_like_pick)
    positions = positions.where(~pick_mask, other="PICK")

    out = pd.DataFrame({
        "dp_name": names,
        "dp_position": positions,
        "dp_team": df[team_col] if team_col else None,
        "dp_value_raw": df[dp_value_col],
    })

    synthetic_rows = []
    if pick_value_lookup is None:
        pick_value_lookup = {}

    for yr in years:
        for rnd in rounds:
            for tier in ("Early", "Mid", "Late"):
                pick_name = f"{yr} {tier} {rnd}{_suffix(rnd)}"
                synthetic_rows.append({
                    "dp_name": pick_name,
                    "dp_position": "PICK",
                    "dp_team": None,
                    "dp_value_raw": float(pick_value_lookup.get(pick_name, 0.0)),
                })

    picks_df = pd.DataFrame(synthetic_rows)
    out = pd.concat([out, picks_df], ignore_index=True)

    return out


# ------------------------------------------------
# Engine loader
# ------------------------------------------------

def load_engine_df(path: Path = ENGINE_VALUES_PATH) -> pd.DataFrame:
    if not path.exists():
        print(f"[value_model] Engine CSV not found at {path}; continuing without engine values.")
        return pd.DataFrame(columns=["sleeper_id", "engine_value"])

    df_raw = pd.read_csv(path)

    if df_raw.empty:
        print("[value_model] Engine CSV is empty; continuing without engine values.")
        return pd.DataFrame(columns=["sleeper_id", "engine_value"])

    sid_col = _pick_col(df_raw, [
        "sleeper_id", "id", "player_id", "sleeperId"
    ])
    val_col = _pick_col(df_raw, [
        "engine_value", "value", "val", "score"
    ])

    if sid_col is None or val_col is None:
        raise ValueError(
            f"Could not find sleeper_id/value columns in engine CSV: {path}. "
            f"Columns present: {list(df_raw.columns)}"
        )

    # CRITICAL FIX: Also load sf_engine_value if present
    out = pd.DataFrame({
        "sleeper_id": df_raw[sid_col].astype(str),
        "engine_value": pd.to_numeric(df_raw[val_col], errors="coerce"),
    })

    if "sf_engine_value" in df_raw.columns:
        out["sf_engine_value"] = pd.to_numeric(df_raw["sf_engine_value"], errors="coerce")
    else:
        out["sf_engine_value"] = None

    out = out.dropna(subset=["sleeper_id", "engine_value"]).copy()
    return out


# ------------------------------------------------
# Draft values helpers
# ------------------------------------------------

def _load_fantasycalc(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(int)
    df["bucket"] = df["bucket"].str.lower().str.strip()
    return df


def _load_dynastyprocess(csv_path: str, num_teams: int = 10) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(int)
    df["pick"] = df["pick"].astype(int)
    df["bucket"] = df["pick"].apply(lambda s: bucket_for_slot(int(s), num_teams=num_teams))

    grouped = (
        df.groupby(["year", "round", "bucket"], as_index=False)["value"]
        .mean()
    )
    return grouped


# ------------------------------------------------
# Training data builder
# ------------------------------------------------

def build_training_dataframe() -> pd.DataFrame:
    current_season = _current_season_from_state()

    fc_df = load_fantasycalc_df()
    dp_df = load_dynastyprocess_df()
    engine_df = load_engine_df()
    history_features_df = load_history_feature_df(current_season)
    investment_df = load_player_investment_df()

    df = fc_df.merge(engine_df, on="sleeper_id", how="left")

    if history_features_df is not None and not history_features_df.empty:
        history_features_df = history_features_df.copy()
        history_features_df["sleeper_id"] = history_features_df["sleeper_id"].astype(str)
        df = df.merge(history_features_df, on="sleeper_id", how="left", suffixes=("", "_hist"))

    if investment_df is not None and not investment_df.empty:
        df = df.merge(investment_df, on="sleeper_id", how="left")

    if "dp_name" in dp_df.columns:
        df["name_lower"] = df["name"].astype(str).str.lower().str.strip()
        dp_df["dp_name_lower"] = dp_df["dp_name"].astype(str).str.lower().str.strip()

        df = df.merge(
            dp_df[["dp_name_lower", "dp_position", "dp_value_raw"]],
            left_on="name_lower",
            right_on="dp_name_lower",
            how="left",
        ).drop(columns=["dp_name_lower"], errors="ignore")

        df.rename(columns={"dp_value_raw": "dp_value"}, inplace=True)
    else:
        df["dp_value"] = np.nan

    if "age" not in df.columns and "fc_age" in df.columns:
        df["age"] = pd.to_numeric(df["fc_age"], errors="coerce")

    df = df[~df["fc_value"].isna()].copy()
    return df


# ------------------------------------------------
# Target normalization helper
# ------------------------------------------------

def _normalize_series_0_1(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mask = s.notna()

    if not mask.any():
        return pd.Series(np.nan, index=s.index)

    vmin = s[mask].min()
    vmax = s[mask].max()

    if vmax <= vmin:
        out = pd.Series(0.5, index=s.index)
        out[~mask] = np.nan
        return out

    out = (s - vmin) / (vmax - vmin)
    out[~mask] = np.nan
    return out


# ------------------------------------------------
# Model training
# ------------------------------------------------

def train_trade_value_model(
        test_size: float = 0.2,
        random_state: int = 42,
) -> TrainedModelBundle:
    df = build_training_dataframe()

    if df.empty:
        raise ValueError("[value_model] Training dataframe is empty.")

    # -----------------------------
    # Target from consensus sources
    # -----------------------------
    fc_val = pd.to_numeric(df["fc_value"], errors="coerce")
    dp_val = pd.to_numeric(
        df.get("dp_value", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    engine_val = pd.to_numeric(
        df.get("engine_value", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )

    fc_norm = _normalize_series_0_1(fc_val)
    dp_norm = _normalize_series_0_1(dp_val)
    engine_norm = _normalize_series_0_1(engine_val)

    # NOTE: Vendor values used for TARGET only, not as features
    # (They're not available during inference from usage_table.json)

    weights = np.vstack([
        np.where(~np.isnan(fc_norm.values), 0.35, 0.0),
        np.where(~np.isnan(dp_norm.values), 0.25, 0.0),
        np.where(~np.isnan(engine_norm.values), 0.40, 0.0),
    ])

    vals = np.vstack([fc_norm.values, dp_norm.values, engine_norm.values])
    numerator = np.nansum(vals * weights, axis=0)
    denominator = np.nansum(weights, axis=0)

    y_norm = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )

    df["target_vendor_norm"] = y_norm
    df["target_value"] = df["target_vendor_norm"] * 1000.0
    df = df.dropna(subset=["target_value"]).copy()

    if df.empty:
        raise ValueError("[value_model] No rows remain after target construction.")

    # -----------------------------
    # Feature columns
    # -----------------------------
    numeric_cols: List[str] = []

    # NOTE: Vendor values NOT used as features (only for target calculation)
    # They're not available during inference

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        if "fc_age" in df.columns:
            df["age"] = df["age"].fillna(pd.to_numeric(df["fc_age"], errors="coerce"))
        numeric_cols.append("age")
    elif "fc_age" in df.columns:
        df["fc_age"] = pd.to_numeric(df["fc_age"], errors="coerce")
        numeric_cols.append("fc_age")

    candidate_usage_cols = [
        "games",
        "avg_off_snap_pct",
        "avg_off_snaps",
        "avg_targets",
        "avg_receptions",
        "avg_rec_yards",
        "avg_rec_tds",
        "avg_carries",
        "avg_rush_yards",
        "avg_rush_tds",
        "ppr_ppg",
        "half_ppr_ppg",
        "std_scoring_ppg",
        "std_ppg",
        "rec_rz_tgt_pg",
        "rush_rz_att_pg",
        "avg_pass_att",
        "avg_pass_cmp",
        "avg_pass_yds",
        "avg_pass_tds",
        "avg_pass_int",
        "target_share",
        "target_share_pct",
        "total_targets",
        # CRITICAL FIX: Rolling window features (captures breakouts, role changes)
        "last_4_weeks_ppg",
        "last_4_weeks_snap_pct",
        "ppg_acceleration",
    ]

    candidate_history_cols = [
        "last_year_ppg",
        "prev_year_ppg",
        "three_year_weighted_ppg",
        "career_best_ppg",
        "career_avg_ppg",
        "last_year_snap_pct",
        "three_year_weighted_snap_pct",
        "last_year_target_share",
        "three_year_weighted_target_share",
        "ppg_trend_1yr",
        "ppg_trend_2yr",
        "target_share_trend_1yr",
        "games_last_year",
        "games_last_3yr",
        "seasons_played",
    ]

    team_feature_cols = [
        "team_pass_att_pg",
        "team_off_snaps_pg",
        "team_rush_att_pg",
        "team_rush_yds_pg",
        "team_pass_yds_pg",
        "team_games_tracked",
    ]

    candidate_investment_cols = [
        "draft_capital_score",
        "draft_capital_pos_pct",
        "contract_apy",
        "guaranteed_money",
        "guaranteed_pct",
        "contract_score",
        "team_investment_score",
        "years_to_fa",
        "contract_apy_pos_pct",
        "guaranteed_money_pos_pct",
        "guaranteed_pct_pos_pct",
    ]

    for col in (
            candidate_usage_cols
            + candidate_history_cols
            + team_feature_cols
            + candidate_investment_cols
    ):
        if col in df.columns:
            numeric_cols.append(col)

    numeric_cols = list(dict.fromkeys(numeric_cols))
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    cat_cols = ["position"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    df_model = df.dropna(subset=["position"]).copy()
    if df_model.empty:
        raise ValueError("[value_model] No rows remain after requiring position.")

    for col in numeric_cols:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    for col in cat_cols:
        df_model[col] = df_model[col].fillna("UNK").astype(str)

    feature_columns = numeric_cols + cat_cols
    if not feature_columns:
        raise ValueError("[value_model] No usable feature columns found.")

    X = df_model[feature_columns].copy()
    y = df_model["target_value"].values

    if len(df_model) < 25:
        raise ValueError(f"[value_model] Not enough training rows: {len(df_model)}")

    # -----------------------------
    # Split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # -----------------------------
    # Preprocessing
    # -----------------------------
    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_cols))

    if cat_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # -----------------------------
    # Model
    # -----------------------------
    gbr = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        random_state=random_state,
        subsample=0.85,
        min_samples_leaf=6,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", gbr),
        ]
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Validation
    # -----------------------------
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)

    # -----------------------------
    # Scaling range for inference
    # -----------------------------
    # CRITICAL FIX: Use fixed scale (0-1000) since target_value is constructed in that range
    # This ensures outputs reach full 999.9 scale even if model regresses to mean
    scale_min = 0.0
    scale_max = 1000.0

    bundle = TrainedModelBundle(
        pipeline=model,
        scale_min=scale_min,
        scale_max=scale_max,
        feature_columns=feature_columns,
    )

    with MODEL_PATH.open("wb") as f:
        pickle.dump(bundle, f)

    debug_cols = [
        "name",
        "position",
        "draft_capital_score",
        "draft_capital_pos_pct",
        "contract_apy",
        "contract_score",
        "team_investment_score",
    ]

    return bundle


# ------------------------------------------------
# Inference helpers
# ------------------------------------------------

def load_trained_bundle(path: Path = MODEL_PATH) -> TrainedModelBundle:
    if not path.exists():
        return train_trade_value_model()

    try:
        with path.open("rb") as f:
            bundle: TrainedModelBundle = pickle.load(f)
    except (AttributeError, ModuleNotFoundError):
        print("[value_model] Incompatible pickle found. Deleting and retraining…")
        try:
            path.unlink()
        except OSError:
            pass
        return train_trade_value_model()

    if not isinstance(bundle, TrainedModelBundle):
        print("[value_model] Invalid model bundle found. Retraining…")
        try:
            path.unlink()
        except OSError:
            pass
        return train_trade_value_model()

    return bundle


def predict_scaled_value_from_row(bundle: TrainedModelBundle, row: pd.Series) -> float:
    model = bundle.pipeline
    scale_min = bundle.scale_min
    scale_max = bundle.scale_max

    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    for col in bundle.feature_columns:
        if col not in row_dict:
            row_dict[col] = "UNK" if col == "position" else 0.0
        else:
            val = row_dict[col]
            if col == "position":
                if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
                    row_dict[col] = "UNK"
            else:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    row_dict[col] = 0.0

    X_row = pd.DataFrame([{col: row_dict[col] for col in bundle.feature_columns}])

    raw_pred = model.predict(X_row)[0]

    # CRITICAL FIX: Model trained on 0-1000 scale, normalize to 0-1 then scale to 999.9
    # Using fixed scale (0-1000) ensures predictions reach full range
    if scale_max <= scale_min:
        return 0.0

    s01 = (raw_pred - scale_min) / (scale_max - scale_min)
    s01 = max(0.0, min(1.0, s01))
    return round(s01 * 999.9, 1)


def build_inference_dataframe() -> pd.DataFrame:
    current_season = _current_season_from_state()

    internal_df = load_internal_stats_df()
    history_features_df = load_history_feature_df(current_season)
    investment_df = load_player_investment_df()

    df = internal_df.copy()

    if history_features_df is not None and not history_features_df.empty:
        history_features_df = history_features_df.copy()
        history_features_df["sleeper_id"] = history_features_df["sleeper_id"].astype(str)
        df = df.merge(
            history_features_df,
            on="sleeper_id",
            how="left",
            suffixes=("", "_hist"),
        )

    if investment_df is not None and not investment_df.empty:
        df = df.merge(investment_df, on="sleeper_id", how="left")

    if "age" not in df.columns:
        if "fc_age" in df.columns:
            df["age"] = pd.to_numeric(df["fc_age"], errors="coerce")
        elif "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")

    return df


def build_ml_value_table() -> Dict[str, float]:
    bundle = load_trained_bundle()

    df = build_inference_dataframe()
    if df.empty:
        return {}

    if "position" in df.columns:
        df = df[df["position"].isin(list(CORE_POSITIONS))].copy()
    else:
        return {}

    for col in bundle.feature_columns:
        if col not in df.columns:
            df[col] = "UNK" if col == "position" else 0.0

    values: Dict[str, float] = {}
    for _, row in df.iterrows():
        pid = str(row["sleeper_id"])
        values[pid] = predict_scaled_value_from_row(bundle, row)

    return values


# ------------------------------------------------
# Rewrite usage table with model outputs
# ------------------------------------------------

def rewrite_value_table_with_model() -> Path:
    date_str = date.today().isoformat()
    source_path = DATA_DIR / f"usage_table_{date_str}.json"
    if not source_path.exists():
        raise FileNotFoundError(f"No usage table file at {source_path}")

    with source_path.open("r", encoding="utf-8") as f:
        players = json.load(f)

    bundle = load_trained_bundle()
    inference_df = build_inference_dataframe()

    # CRITICAL FIX: Load vendor values to use directly when available
    fc_df = load_fantasycalc_df()
    dp_df = load_dynastyprocess_df()

    # Build vendor value lookup for 1QB
    vendor_values: dict[str, float] = {}

    # Get FC values normalized to 999.9 scale
    if not fc_df.empty and "fc_value" in fc_df.columns and "sleeper_id" in fc_df.columns:
        fc_max = fc_df["fc_value"].max()
        for _, row in fc_df.iterrows():
            pid = str(row.get("sleeper_id"))
            fc_val = row.get("fc_value")
            if pid and pd.notna(fc_val):
                vendor_values[pid] = float(fc_val) / fc_max * 999.9

    # Build Superflex vendor value lookup using sf_engine_value + value_2qb
    sf_vendor_values: dict[str, float] = {}

    # Load engine_values CSV (contains both 1QB + SF values for all league sizes)
    engine_values_path = DATA_DIR / f"engine_values_{date.today().isoformat()}.csv"
    sf_engine_map: dict[str, float] = {}
    # Per-league-size engine maps: {size: {pid: value}}
    engine_size_map: dict[int, dict[str, float]] = {}
    sf_engine_size_map: dict[int, dict[str, float]] = {}
    LEAGUE_SIZES = [8, 10, 12, 14]
    for n in LEAGUE_SIZES:
        engine_size_map[n] = {}
        sf_engine_size_map[n] = {}
    if engine_values_path.exists():
        try:
            engine_df = pd.read_csv(engine_values_path)
            for _, row in engine_df.iterrows():
                pid = str(row.get("player_id"))
                if not pid:
                    continue
                sf_eng_val = row.get("sf_engine_value")
                if pd.notna(sf_eng_val):
                    sf_engine_map[pid] = float(sf_eng_val)
                for n in LEAGUE_SIZES:
                    col_1qb = f"engine_value_{n}"
                    col_sf = f"sf_engine_value_{n}"
                    if col_1qb in engine_df.columns and pd.notna(row.get(col_1qb)):
                        engine_size_map[n][pid] = float(row[col_1qb])
                    if col_sf in engine_df.columns and pd.notna(row.get(col_sf)):
                        sf_engine_size_map[n][pid] = float(row[col_sf])
        except Exception as e:
            print(f"[ERROR] Failed to load engine_values: {e}")

    # Load value_2qb from dynastyprocess (need to match by name+team)
    dp_2qb_map: dict[tuple[str, str], float] = {}  # (name, team) -> value_2qb
    if not dp_df.empty:
        try:
            dp_raw = pd.read_csv(DATA_DIR / f"dynastyprocess_values_{date.today().isoformat()}.csv")
            if "player" in dp_raw.columns and "value_2qb" in dp_raw.columns:
                for _, row in dp_raw.iterrows():
                    name = str(row.get("player", "")).strip()
                    team = str(row.get("team", "")).strip()
                    val_2qb = row.get("value_2qb")
                    if name and pd.notna(val_2qb):
                        dp_2qb_map[(name, team)] = float(val_2qb)
        except Exception as e:
            print(f"[ERROR] Failed to load value_2qb from dynastyprocess: {e}")

    # Calculate Superflex vendor values: blend FC (50%), DP 2QB (35%), SF Engine (15%)
    # First, normalize DP 2QB values to 999.9 scale
    dp_2qb_max = max(dp_2qb_map.values()) if dp_2qb_map else 1.0

    # Build sf_vendor_values for each player
    players_index = load_players_index() or {}
    for pid_key, meta in players_index.items():
        pid = str(pid_key)
        name = meta.get("name", "").strip()
        team = meta.get("team", "").strip()

        # Get FC value (same for 1QB and SF)
        fc_val_norm = vendor_values.get(pid, 0.0)

        # Get SF engine value
        sf_eng_val = sf_engine_map.get(pid, 0.0)

        # Get DP 2QB value (normalized)
        dp_2qb_raw = dp_2qb_map.get((name, team), 0.0)
        dp_2qb_norm = (dp_2qb_raw / dp_2qb_max * 999.9) if dp_2qb_max > 0 else 0.0

        # Blend: 50% FC, 35% DP 2QB, 15% SF Engine
        if fc_val_norm > 0 or sf_eng_val > 0 or dp_2qb_norm > 0:
            sf_value = (0.50 * fc_val_norm) + (0.35 * dp_2qb_norm) + (0.15 * sf_eng_val)

            # CRITICAL FIX: Boost QBs significantly in Superflex so top QBs reach ~999
            # In Superflex, elite QBs should be valued like elite RBs/WRs
            pos = meta.get("pos", "").upper()
            if pos == "QB":
                sf_value = sf_value * 1.5  # 50% boost for QBs

            sf_vendor_values[pid] = sf_value

    df_by_id: dict[str, pd.Series] = {}
    for _, row in inference_df.iterrows():
        pid = str(row.get("sleeper_id"))
        if pid:
            df_by_id[pid] = row

    cleaned_assets: list[dict] = []

    for player in players:
        pid = str(player.get("id"))
        row = df_by_id.get(pid)

        # CRITICAL FIX: Use vendor value directly if available, else ML model
        if pid in vendor_values:
            final_value = vendor_values[pid]
        else:
            final_value = predict_scaled_value_from_row(bundle, row) if row is not None else 0.0

        # Calculate Superflex value
        if pid in sf_vendor_values:
            sf_value = sf_vendor_values[pid]
        else:
            # Fallback to ML model for SF (same as 1QB for now)
            sf_value = predict_scaled_value_from_row(bundle, row) if row is not None else 0.0

        # Position-specific adjustments: TEs capped at ~800
        position = player.get("position")
        if position == "TE":
            # Apply 1.35x multiplier to TEs (allows top TEs to reach ~800), then cap
            final_value = min(final_value * 1.35, 800.0)
            sf_value = min(sf_value * 1.35, 800.0)

        age = player.get("age")
        if age is None and row is not None:
            if "age" in row and not pd.isna(row["age"]):
                age = row["age"]
            elif "fc_age" in row and not pd.isna(row["fc_age"]):
                age = row["fc_age"]

        name = player.get("name")

        # Per-league-size values: scale the blended model value by the ratio
        # of engine values between the target size and the default 10-team size.
        # This preserves vendor-consensus anchoring while adjusting for scarcity.
        eng_base = engine_size_map[10].get(pid) or 0.0
        sf_eng_base = sf_engine_size_map[10].get(pid) or 0.0
        size_values: dict[str, float] = {}
        sf_size_values: dict[str, float] = {}
        for n in LEAGUE_SIZES:
            if n == 10:
                continue
            eng_n = engine_size_map[n].get(pid) or 0.0
            sf_eng_n = sf_engine_size_map[n].get(pid) or 0.0
            ratio = (eng_n / eng_base) if eng_base > 0 else 1.0
            sf_ratio = (sf_eng_n / sf_eng_base) if sf_eng_base > 0 else 1.0
            size_values[f"value_{n}"] = round(min(float(final_value) * ratio, 999.9), 1)
            sf_size_values[f"sf_value_{n}"] = round(min(float(sf_value) * sf_ratio, 999.9), 1)

        asset = {
            "id": player.get("id"),
            "name": name,
            "team": player.get("team"),
            "position": player.get("position"),
            "age": age,
            "value": float(final_value),
            "sf_value": float(sf_value),
            "search_name": normalize_name(name) if name else "",
            "pos_rank": None,
            "pos_rank_label": None,
            "sf_pos_rank": None,
            "sf_pos_rank_label": None,
        }
        asset.update(size_values)
        asset.update(sf_size_values)
        cleaned_assets.append(asset)

    pick_values = load_pick_value_table() or {}

    for key, val in pick_values.items():
        parts = key.split("_")

        name = None

        # Exact slotted pick format: YYYY_R_PPPOS  ->  2026 1.01
        # Example keys:
        #   2026_1_01
        #   2026_2_04
        if len(parts) == 3 and parts[2].isdigit():
            year_str, rnd_str, pick_str = parts

            try:
                year = int(year_str)
                rnd = int(rnd_str)
                pick_in_round = int(pick_str)
            except ValueError:
                continue

            name = f"{year} {rnd}.{pick_in_round:02d}"

        # Bucketed future pick format: YYYY_R_bucket  ->  2027 1st (Early)
        # Example keys:
        #   2027_1_early
        #   2027_2_mid
        elif len(parts) == 3:
            year_str, rnd_str, bucket = parts

            try:
                year = int(year_str)
                rnd = int(rnd_str)
            except ValueError:
                continue

            suffix = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
            bucket_label = bucket.lower().capitalize()
            name = f"{year} {rnd}{suffix} ({bucket_label})"

        # Plain round-only format: YYYY_R  ->  2027 1st
        elif len(parts) == 2:
            year_str, rnd_str = parts

            try:
                year = int(year_str)
                rnd = int(rnd_str)
            except ValueError:
                continue

            suffix = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
            name = f"{year} {rnd}{suffix}"

        else:
            continue

        pick_asset = {
            "id": key,
            "name": name,
            "team": "Pick",
            "position": "PICK",
            "age": None,
            "value": float(val),
            "sf_value": float(val),  # Picks have same value in 1QB and Superflex
            "search_name": normalize_name(name),
            "pos_rank": None,
            "pos_rank_label": None,
            "sf_pos_rank": None,
            "sf_pos_rank_label": None,
        }
        # Draft picks are not scarcity-sensitive to league size — same value in all sizes
        for n in LEAGUE_SIZES:
            if n != 10:
                pick_asset[f"value_{n}"] = float(val)
                pick_asset[f"sf_value_{n}"] = float(val)
        cleaned_assets.append(pick_asset)

    pos_to_indices: dict[str, list[int]] = {}

    for idx, asset in enumerate(cleaned_assets):
        pos = str(asset.get("position") or "").upper()
        if not pos or pos == "PICK":
            continue
        pos_to_indices.setdefault(pos, []).append(idx)

    for pos, indices in pos_to_indices.items():
        indices.sort(key=lambda i: float(cleaned_assets[i].get("value") or 0.0), reverse=True)

        rank = 1
        for i in indices:
            cleaned_assets[i]["pos_rank"] = rank
            cleaned_assets[i]["pos_rank_label"] = f"{pos}{rank}"
            rank += 1

    # Calculate Superflex position ranks based on sf_value
    sf_pos_to_indices: dict[str, list[int]] = {}

    for idx, asset in enumerate(cleaned_assets):
        pos = str(asset.get("position") or "").upper()
        if not pos or pos == "PICK":
            continue
        sf_pos_to_indices.setdefault(pos, []).append(idx)

    for pos, indices in sf_pos_to_indices.items():
        indices.sort(key=lambda i: float(cleaned_assets[i].get("sf_value") or 0.0), reverse=True)

        rank = 1
        for i in indices:
            cleaned_assets[i]["sf_pos_rank"] = rank
            cleaned_assets[i]["sf_pos_rank_label"] = f"{pos}{rank}"
            rank += 1

    today = date.today()
    yesterday = today - timedelta(days=1)
    yesterday_file = DATA_DIR / f"model_values_{yesterday.isoformat()}.json"

    if yesterday_file.exists():
        print(f"[model_values] Removing yesterday's value file: {yesterday_file.name}")
        try:
            yesterday_file.unlink()
        except Exception as e:
            print(f"[model_values] Failed to remove yesterday's file: {e}")

    out_path = DATA_DIR / f"model_values_{date_str}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_assets, f, ensure_ascii=False, indent=2)

    return out_path
