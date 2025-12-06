# dashboard_services/value_model_training.py

from __future__ import annotations

import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import re
import requests
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
from typing import Dict, Iterable, List, Optional

from dashboard_services.picks import load_pick_value_table
from dashboard_services.utils import load_teams_index, bucket_for_slot

# ------------------------------------------------
# Paths / constants
# ------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = DATA_DIR / "trade_value_model.pkl"
DYNASTYPROCESS_VALUES_PATH = DATA_DIR / f"dynastyprocess_values_{date.today().isoformat()}.csv"
FANTASYCALC_VALUES_PATH = DATA_DIR / f"fantasycalc_api_values_{date.today().isoformat()}.csv"
ENGINE_VALUES_PATH = DATA_DIR / f"engine_values_{date.today().isoformat()}.csv"

FANTASYCALC_URL = (
    "https://api.fantasycalc.com/values/current"
    "?isDynasty=true&numQbs=1&numTeams=10&ppr=1"
)

CURRENT_SEASON = 2025


# ------------------------------------------------
# Trained model bundle
# ------------------------------------------------

@dataclass
class TrainedModelBundle:
    """
    What we store on disk in trade_value_model.pkl
    """
    pipeline: Pipeline
    scale_min: float
    scale_max: float
    feature_columns: List[str]


# ------------------------------------------------
# Internal stats loader
# ------------------------------------------------

def load_internal_stats_df() -> pd.DataFrame:
    """
    Load your internal value table + usage + team context into a DataFrame.

    Expects data/usage_table_{YYYY-MM-DD}.json to look like:
      [
        {
          "id": "9488",
          "name": "Jaxon Smith-Njigba",
          "team": "SEA",
          "position": "WR",
          "age": null,
          "value": 983.0,
          "usage": { ... }
        },
        ...
      ]
    And teams_index to have pass/rush/snaps context.
    """
    value_path = DATA_DIR / f"usage_table_{date.today().isoformat()}.json"

    if not value_path.exists():
        raise FileNotFoundError(f"No internal value table found at {value_path}")

    with value_path.open() as f:
        players = json.load(f)  # list[dict]

    # Flatten players + usage
    df = pd.json_normalize(players)

    # Normalize column names
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
    }
    df = df.rename(columns=rename_map)
    df["sleeper_id"] = df["sleeper_id"].astype(str)

    df["internal_value"] = df.get("internal_value_raw", np.nan)

    # Attach team-level context from teams_index.json
    teams_index = load_teams_index()  # { "ARI": {...}, ... }

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
# FantasyCalc API loader
# ------------------------------------------------

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_fantasycalc_df(path: Path = FANTASYCALC_VALUES_PATH) -> pd.DataFrame:
    """
    Load FantasyCalc dynasty values from a local CSV exported previously.

    Expects a CSV that has at least:
      - some kind of sleeper id column
      - some kind of value column

    We try to be robust about column names and normalize to:

      sleeper_id, name, position, team, fc_value, fc_rank, fc_age
    """
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
    print(f"[value_model] FantasyCalc rows (from CSV): {len(df)}")
    return df


# ------------------------------------------------
# DynastyProcess CSV loader
# ------------------------------------------------

def load_dynastyprocess_df(
        path: Path = DYNASTYPROCESS_VALUES_PATH,
        pick_value_lookup: Dict[str, float] = None,
        years=(2025, 2026, 2027, 2028),
        rounds=(1, 2, 3),
) -> pd.DataFrame:
    """
    Load DynastyProcess CSV AND insert synthetic draft picks.

    pick_value_lookup should be something like:
        {
          "2026 Early 1st": 475,
          "2026 Mid 1st": 440,
          "2026 Late 1st": 410,
          ...
        }

    Returns DataFrame with columns:
        dp_name, dp_position, dp_team, dp_value_raw
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing DP CSV at {path}")

    df = pd.read_csv(path)

    # ----- Detect value column -----
    dp_value_col = "value_1qb"
    if dp_value_col not in df.columns:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        dp_value_col = "value" if "value" in numeric_cols else numeric_cols[0]

    # ----- Detect name/pos/team columns -----
    name_col = "player" if "player" in df.columns else \
        "Player" if "Player" in df.columns else df.columns[0]

    pos_col = "pos" if "pos" in df.columns else \
        "position" if "position" in df.columns else None

    team_col = "team" if "team" in df.columns else None

    names = df[name_col].astype(str)
    positions = df[pos_col].astype(str) if pos_col else pd.Series([""] * len(df))

    # ----- Detect pick-like rows -----
    def looks_like_pick(name: str):
        s = name.lower().strip()
        return bool(re.match(r"^\d{4}\s+(early|mid|late)\s+\d+(st|nd|rd|th)$", s))

    pick_mask = names.apply(looks_like_pick)

    # force pos="PICK" for all detected
    positions = positions.where(~pick_mask, other="PICK")

    # ----- Build base frame -----
    out = pd.DataFrame({
        "dp_name": names,
        "dp_position": positions,
        "dp_team": df[team_col] if team_col else None,
        "dp_value_raw": df[dp_value_col],
    })

    # ============================================================
    #                   ADD SYNTHETIC DRAFT PICKS
    # ============================================================

    synthetic_rows = []

    if pick_value_lookup is None:
        pick_value_lookup = {}  # empty fallback; user fills it later

    for yr in years:
        for rnd in rounds:
            for tier in ("Early", "Mid", "Late"):
                pick_name = f"{yr} {tier} {rnd}{_suffix(rnd)}"

                synthetic_rows.append({
                    "dp_name": pick_name,
                    "dp_position": "PICK",
                    "dp_team": None,
                    "dp_value_raw": float(pick_value_lookup.get(pick_name, 0.0))
                })

    picks_df = pd.DataFrame(synthetic_rows)

    # Append synthetic picks to the real DP data
    out = pd.concat([out, picks_df], ignore_index=True)

    return out


def _suffix(rnd: int) -> str:
    """Return st/nd/rd/th for round numbers."""
    if rnd == 1: return "st"
    if rnd == 2: return "nd"
    if rnd == 3: return "rd"
    return "th"


def load_engine_df(path: Path = ENGINE_VALUES_PATH) -> pd.DataFrame:
    """
    Load your engine's dynasty values from CSV.

    Expects a CSV that has at least:
      - a sleeper / player id column
      - a numeric value column

    Normalizes to:
      sleeper_id, engine_value
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected engine values CSV at {path}."
        )

    df_raw = pd.read_csv(path)

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

    out = pd.DataFrame(
        {
            "sleeper_id": df_raw[sid_col].astype(str),
            "engine_value": df_raw[val_col].astype(float),
        }
    )

    print(f"[value_model] Engine rows: {len(out)}")
    return out


# ------------------------------------------------
# Training data builder
# ------------------------------------------------

def build_training_dataframe(
) -> pd.DataFrame:
    """
    Combine:
      - FantasyCalc values (by sleeper_id)
      - DynastyProcess values (by fuzzy name)
      - Engine values (by sleeper_id)
      - Internal value_table_{date} (usage + team context) as features only
    """
    fc_df = load_fantasycalc_df()  # sleeper_id, name, position, team, fc_value, fc_rank, fc_age
    dp_df = load_dynastyprocess_df()  # dp_name, dp_position, dp_team, dp_value_raw
    engine_df = load_engine_df()  # sleeper_id, engine_value
    internal_df = load_internal_stats_df()  # sleeper_id, usage + team stats + age, etc.
    # --- 1) Merge FC + Engine + Internal on sleeper_id ---
    df = fc_df.merge(
        engine_df,
        on="sleeper_id",
        how="left",
    )

    df = df.merge(
        internal_df,
        on="sleeper_id",
        how="left",
        suffixes=("", "_int"),
    )

    # --- 2) Attach DP via name match (fuzzy-ish on lowercase name) ---
    if "dp_name" in dp_df.columns:
        df["name_lower"] = df["name"].astype(str).str.lower()
        dp_df["dp_name_lower"] = dp_df["dp_name"].astype(str).str.lower()

        df = df.merge(
            dp_df[["dp_name_lower", "dp_position", "dp_value_raw"]],
            left_on="name_lower",
            right_on="dp_name_lower",
            how="left",
        ).drop(columns=["dp_name_lower"])

        df.rename(columns={"dp_value_raw": "dp_value"}, inplace=True)
    else:
        df["dp_value"] = np.nan

    # We DO NOT create internal_value anymore (no internal vendor)
    # Keep players that have an FC value (base universe)
    df = df[~df["fc_value"].isna()].copy()

    return df


# dashboard_services/draft_values.py

def _load_fantasycalc(csv_path: str) -> pd.DataFrame:
    """
    Expected columns (tweak as needed):
      year, round, bucket, value
    Where bucket in {'early','mid','late'}.
    """
    df = pd.read_csv(csv_path)
    # normalize
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(int)
    df["bucket"] = df["bucket"].str.lower().str.strip()
    return df


def _load_dynastyprocess(csv_path: str, num_teams: int = 10) -> pd.DataFrame:
    """
    Expected something like:
      year, round, pick, value
    where 'pick' is 1..num_teams.
    If your DP CSV uses a single pick column like 1.01/1.02/etc,
    you can split that into (round, pick) before this step.
    """
    df = pd.read_csv(csv_path)

    # adjust these column names to your actual CSV
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(int)
    df["pick"] = df["pick"].astype(int)

    # derive bucket from pick within round
    df["bucket"] = df["pick"].apply(lambda s: bucket_for_slot(int(s), num_teams=num_teams))

    # now group to early/mid/late per round/year
    # e.g. average of all mid picks in that round for that class
    grouped = (
        df.groupby(["year", "round", "bucket"], as_index=False)["value"]
        .mean()
    )
    return grouped


# ------------------------------------------------
# Target normalization helper
# ------------------------------------------------

def _normalize_series_0_1(s: pd.Series) -> pd.Series:
    """Normalize a series to [0,1], keeping NaNs as NaN."""
    s = s.astype(float)
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
    print("[value_model] Building training dataframe…")
    df = build_training_dataframe()

    # --- Build target from 3 vendors: FC, DP, ENGINE ---

    fc_val = df["fc_value"].astype(float)
    dp_val = df.get("dp_value", pd.Series(np.nan, index=df.index)).astype(float)
    engine_val = df.get("engine_value", pd.Series(np.nan, index=df.index)).astype(float)

    fc_norm = _normalize_series_0_1(fc_val)
    dp_norm = _normalize_series_0_1(dp_val)
    engine_norm = _normalize_series_0_1(engine_val)

    weights = np.vstack([
        np.where(~np.isnan(fc_norm.values), 0.4, 0.0),  # FC weight
        np.where(~np.isnan(dp_norm.values), 0.4, 0.0),  # DP weight
        np.where(~np.isnan(engine_norm.values), 0.2, 0.0),  # Engine weight
    ])

    vals = np.vstack([fc_norm.values, dp_norm.values, engine_norm.values])
    numerator = np.nansum(vals * weights, axis=0)
    denominator = np.nansum(weights, axis=0)
    y_norm = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    df["target_vendor_norm"] = y_norm
    df["target_value"] = df["target_vendor_norm"] * 1000.0
    df = df[~df["target_value"].isna()].copy()

    # ------------------------------------------------
    # Features
    # ------------------------------------------------
    numeric_cols: List[str] = []

    # Age from internal stats if present, else fc_age
    if "age" in df.columns:
        df["age"] = df["age"].fillna(df.get("fc_age"))
        numeric_cols.append("age")
    elif "fc_age" in df.columns:
        numeric_cols.append("fc_age")

    candidate_usage_cols = [
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
        "total_targets",
    ]

    team_feature_cols = [
        "team_pass_att_pg",
        "team_off_snaps_pg",
        "team_rush_att_pg",
        "team_rush_yds_pg",
        "team_pass_yds_pg",
        "team_games_tracked",
    ]

    for col in candidate_usage_cols + team_feature_cols:
        if col in df.columns:
            numeric_cols.append(col)

    numeric_cols = list(dict.fromkeys([c for c in numeric_cols if c in df.columns]))
    cat_cols = ["position"]

    df_model = df.dropna(subset=["target_value", "position"]).copy()

    # Ensure numeric cols exist
    for col in numeric_cols:
        if col not in df_model.columns:
            df_model[col] = 0.0
    df_model[numeric_cols] = df_model[numeric_cols].astype(float)

    # Ensure categorical cols exist
    for col in cat_cols:
        if col not in df_model.columns:
            df_model[col] = "UNK"
    df_model[cat_cols] = df_model[cat_cols].fillna("UNK")

    X = df_model[numeric_cols + cat_cols]
    y = df_model["target_value"].values

    # ------------------------------------------------
    # Train / validation split
    # ------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ------------------------------------------------
    # Preprocessing
    # ------------------------------------------------
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    # ------------------------------------------------
    # Gradient Boosting model
    # ------------------------------------------------
    gbr = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=3,
        random_state=random_state,
        subsample=0.9,
        min_samples_leaf=5,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", gbr),
        ]
    )

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    print(f"[value_model] Validation MAE (3-vendor consensus): {mae:.2f} value points")

    # ------------------------------------------------
    # Capture prediction range for scaling in your app
    # ------------------------------------------------
    y_pred_full = model.predict(X)
    scale_min = float(y_pred_full.min())
    scale_max = float(y_pred_full.max())

    bundle = TrainedModelBundle(
        pipeline=model,
        scale_min=scale_min,
        scale_max=scale_max,
        feature_columns=numeric_cols + cat_cols,
    )

    with MODEL_PATH.open("wb") as f:
        pickle.dump(bundle, f)

    return bundle


# ------------------------------------------------
# Inference helpers
# ------------------------------------------------

def load_trained_bundle(path: Path = MODEL_PATH) -> TrainedModelBundle:
    """
    Load the trained model bundle from disk.

    If the existing pickle is incompatible with the current code
    (e.g. AttributeError: Can't get attribute 'TrainedModelBundle' on __main__),
    delete it and retrain.
    """
    if not path.exists():
        bundle = train_trade_value_model()

        return bundle

    try:
        with path.open("rb") as f:
            bundle: TrainedModelBundle = pickle.load(f)
    except (AttributeError, ModuleNotFoundError):
        print(
            "[value_model] Existing model pickle is incompatible with the "
            "current code. Deleting and retraining…"
        )
        try:
            path.unlink()
        except OSError:
            pass

        bundle = train_trade_value_model()

        return bundle

    if not isinstance(bundle, TrainedModelBundle):
        print(
            "[value_model] Loaded object is not TrainedModelBundle. "
            "Retraining model to refresh the pickle…"
        )
        try:
            path.unlink()
        except OSError:
            pass

        bundle = train_trade_value_model()

        return bundle

    return bundle


def predict_scaled_value_from_row(bundle: TrainedModelBundle, row: pd.Series) -> float:
    """
    Given a single player row (with same feature columns as training),
    return the scaled trade value 0–999.9.
    """
    model = bundle.pipeline
    scale_min = bundle.scale_min
    scale_max = bundle.scale_max

    X_row = row[bundle.feature_columns].to_frame().T

    raw_pred = model.predict(X_row)[0]
    if scale_max <= scale_min:
        return 0.0

    s01 = (raw_pred - scale_min) / (scale_max - scale_min)
    s01 = max(0.0, min(1.0, s01))
    return round(s01 * 999.9, 1)


def build_ml_value_table() -> Dict[str, float]:
    """
    Use the trained model to build a {sleeper_id: value} table
    for your trade calculator.
    """
    bundle = load_trained_bundle()

    df = load_internal_stats_df()
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()

    for col in bundle.feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    df[bundle.feature_columns] = df[bundle.feature_columns].fillna(0)

    values: Dict[str, float] = {}
    for _, row in df.iterrows():
        pid = str(row["sleeper_id"])
        v = predict_scaled_value_from_row(bundle, row)
        values[pid] = v

    return values




def normalize_name(name: str) -> str:
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

    if not name:
        return ""
    s = name.lower()

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # drop suffix tokens like jr, sr, ii, iii, etc.
    parts = s.split(" ")
    parts = [p for p in parts if p not in suffixes]

    return " ".join(parts)


# ------------------------------------------------
# Rewrite value_table_{date}.json with model outputs
# ------------------------------------------------

def rewrite_value_table_with_model() -> Path:
    """
    Load value_table_{today}.json, recompute the model value for each player,
    and write a NEW file:
        model_values_{today}.json

    Output schema per asset:
      {
        id,
        name,
        team,
        position,
        age,
        value,
        search_name,
        pos_rank,        # int or None
        pos_rank_label,  # e.g. "WR1" or None
      }

    Includes both:
      - real players (QB/RB/WR/TE/etc.)
      - draft picks (position='PICK', team='Pick')
    """
    date_str = date.today().isoformat()
    source_path = DATA_DIR / f"usage_table_{date_str}.json"
    if not source_path.exists():
        raise FileNotFoundError(f"No usage table file at {source_path}")

    with source_path.open("r", encoding="utf-8") as f:
        players = json.load(f)  # list[dict]

    bundle = load_trained_bundle()
    internal_df = load_internal_stats_df()

    # Map sleeper_id -> row for quick lookup
    df_by_id = {str(row["sleeper_id"]): row for _, row in internal_df.iterrows()}

    cleaned_assets: list[dict] = []

    # ---------- 1) Players with model values ----------
    for player in players:
        pid = str(player.get("id"))
        row = df_by_id.get(pid)

        ml_value = predict_scaled_value_from_row(bundle, row) if row is not None else 0.0

        # Prefer age from JSON; fall back to internal stats / fc if missing
        age = player.get("age")
        if age is None and row is not None:
            if "age" in row:
                age = row["age"]
            elif "fc_age" in row:
                age = row["fc_age"]

        cleaned_assets.append({
            "id": player.get("id"),
            "name": player.get("name"),
            "team": player.get("team"),
            "position": player.get("position"),
            "age": age,
            "value": float(ml_value),
            "search_name": normalize_name(player.get("name")),
            # filled in later
            "pos_rank": None,
            "pos_rank_label": None,
        })

    # ---------- 2) Draft picks from pick value table ----------
    pick_values = load_pick_value_table() or {}  # { "2026_1_early": value, "2026_1": value, ... }

    for key, val in pick_values.items():
        parts = key.split("_")

        # Expect either "2026_1_early" or "2026_1"
        if len(parts) == 3:
            year_str, rnd_str, bucket = parts
            bucket = bucket.lower()
        elif len(parts) == 2:
            year_str, rnd_str = parts
            bucket = None
        else:
            # Weird key like "foo" – skip
            continue

        try:
            year = int(year_str)
            rnd = int(rnd_str)
        except ValueError:
            continue

        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")

        if bucket:
            bucket_label = bucket.capitalize()  # early/mid/late -> Early/Mid/Late
            name = f"{year} {rnd}{suffix} ({bucket_label})"
        else:
            name = f"{year} {rnd}{suffix}"

        cleaned_assets.append({
            "id": key,           # trade calculator uses this as pick id
            "name": name,        # display string
            "team": "Pick",      # keeps schema consistent
            "position": "PICK",  # lets UI / logic distinguish picks
            "age": None,         # no age for picks
            "value": float(val),
            "search_name": normalize_name(name),
            "pos_rank": None,
            "pos_rank_label": None,
        })

    # ---------- 3) Compute per-position ranks (players only) ----------
    # Group indexes by position
    pos_to_indices: dict[str, list[int]] = {}

    for idx, asset in enumerate(cleaned_assets):
        pos = str(asset.get("position") or "").upper()
        if not pos or pos == "PICK":
            continue  # don't rank picks
        pos_to_indices.setdefault(pos, []).append(idx)

    # For each position, sort by value desc, assign rank
    for pos, indices in pos_to_indices.items():
        # sort indices by value
        indices.sort(key=lambda i: float(cleaned_assets[i].get("value") or 0.0), reverse=True)

        rank = 1
        for i in indices:
            # optional: if you want ties to share rank, uncomment this block
            # if last_val is not None and val < last_val:
            #     rank += 1
            # last_val = val

            # simpler: strict ordering (1,2,3,...)
            cleaned_assets[i]["pos_rank"] = rank
            cleaned_assets[i]["pos_rank_label"] = f"{pos}{rank}"
            rank += 1

    today = date.today()
    yesterday = today - timedelta(days=1)

    pattern = f"model_values_{yesterday.isoformat()}.json"
    yesterday_file = DATA_DIR / pattern

    if yesterday_file.exists():
        print(f"[model_values] Removing yesterday's value file: {yesterday_file.name}")
        try:
            yesterday_file.unlink()
        except Exception as e:
            print(f"[model_values] Failed to remove yesterday's file: {e}")

    # ---------- 4) Write combined table ----------
    out_path = DATA_DIR / f"model_values_{date_str}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_assets, f, ensure_ascii=False, indent=2)

    print(f"[value_model] Wrote model values (players + picks) → {out_path}")

    return out_path



# ------------------------------------------------
# CLI entrypoint
# ------------------------------------------------

if __name__ == "__main__":
    bundle = train_trade_value_model(
    )

    rewrite_value_table_with_model(
    )
