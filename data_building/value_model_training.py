# dashboard_services/value_model_training.py

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dashboard_services.api import get_nfl_state
from dashboard_services.picks import load_pick_value_table
from data_building.external_data.player_history import load_player_history_df, build_player_history_features
from data_building.external_data.player_investment import load_player_investment_context
from utils.paths import DATA_DIR
from utils.utils import load_teams_index, bucket_for_slot, normalize_name, load_players_index
from utils.coerce import safe_int as _safe_int

# ------------------------------------------------
# Paths / constants
# ------------------------------------------------

DYNASTYPROCESS_VALUES_PATH = DATA_DIR / "dynastyprocess_values.csv"
FANTASYCALC_VALUES_PATH    = DATA_DIR / "fantasycalc_api_values.csv"
ENGINE_VALUES_PATH         = DATA_DIR / "engine_values.csv"

# Cache file that persists the smoothed 1QB normalization scale across runs.
# NOTE: Render's web/cron services run on an ephemeral filesystem (no persistent
# disk in render.yaml), so this file is wiped on every deploy/restart and the
# two services never share it. The scale is therefore persisted in Postgres
# (pipeline_state table) as the durable source of truth; the file is kept only
# as a local-dev / DB-unavailable fallback. Losing the smoothed scale forces a
# fall back to the raw scale, which makes the whole player pool jump in value
# whenever the top non-QB anchor drifts — the bug this persistence prevents.
_SCALE_CACHE_PATH = DATA_DIR.parent / "cache" / "value_scale_1qb.json"
_SCALE_STATE_KEY = "value_scale_1qb"
# Weight given to TODAY's raw scale when blending with yesterday's smoothed scale.
# 0.15 means a true 10% shift in the top player's value only moves the scale ~1.5%
# today; it takes ~2 weeks to fully propagate — preventing overnight team-value swings.
_SCALE_EMA_ALPHA = 0.15

# --- 1QB ceiling anchor (basket) -------------------------------------------
# The 1QB scale pins the top non-QB to 999.9. Anchoring to a SINGLE player (the
# max) means that player's day-to-day drift drags every other player inversely:
# when the anchor ticks up, the scale shrinks and the whole board declines even
# though nothing about those players changed. To break that, the anchor is a
# basket — the mean of the top N non-QBs — so one player moving shifts it ~N×
# less. A separate slowly-moving "headroom" ratio (top-1 / basket) keeps the #1
# sitting at ~999.9 without letting the #1 re-introduce single-player sensitivity.
_ANCHOR_BASKET_N        = 5
_BASKET_STATE_KEY       = "value_basket_1qb"      # smoothed basket mean
_HEADROOM_STATE_KEY     = "value_headroom_1qb"    # smoothed top1/basket ratio
_BASKET_EMA_ALPHA       = 0.15   # basket tracks genuine tier drift over ~2 weeks
_HEADROOM_EMA_ALPHA     = 0.04   # headroom is near-fixed: a #1 pulling away just
                                 # rides the 999.9 cap rather than compressing all


def _load_persisted_scale() -> float:
    """Return the last smoothed 1QB normalization scale.

    Prefers the durable Postgres store (survives deploys and is shared by the
    web + cron services); falls back to the ephemeral file cache only when the
    DB is unavailable (local dev / outage). Returns 0.0 when no prior scale
    exists, which signals the caller to use today's raw scale unsmoothed.
    """
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    key TEXT PRIMARY KEY,
                    value DOUBLE PRECISION,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            row = conn.execute(
                "SELECT value FROM pipeline_state WHERE key = %s",
                (_SCALE_STATE_KEY,),
            ).fetchone()
        if row and row.get("value"):
            return float(row["value"])
    except Exception:
        pass
    try:
        return float(json.loads(_SCALE_CACHE_PATH.read_text()).get("scale", 0) or 0)
    except Exception:
        return 0.0


def _persist_scale(scale: float) -> None:
    """Persist the smoothed scale durably to Postgres and to the file fallback."""
    rounded = round(float(scale), 6)
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    key TEXT PRIMARY KEY,
                    value DOUBLE PRECISION,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute(
                """
                INSERT INTO pipeline_state (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = excluded.value, updated_at = NOW()
                """,
                (_SCALE_STATE_KEY, rounded),
            )
    except Exception:
        pass
    try:
        _SCALE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SCALE_CACHE_PATH.write_text(json.dumps({"scale": rounded}))
    except Exception:
        pass


def _load_state(key: str) -> float:
    """Read a single durable float from pipeline_state. Returns 0.0 if absent."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    key TEXT PRIMARY KEY,
                    value DOUBLE PRECISION,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            row = conn.execute(
                "SELECT value FROM pipeline_state WHERE key = %s", (key,)
            ).fetchone()
        if row and row.get("value"):
            return float(row["value"])
    except Exception:
        pass
    return 0.0


def _save_state(key: str, value: float) -> None:
    """Write a single durable float to pipeline_state (best-effort)."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    key TEXT PRIMARY KEY,
                    value DOUBLE PRECISION,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute(
                """
                INSERT INTO pipeline_state (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = excluded.value, updated_at = NOW()
                """,
                (key, round(float(value), 6)),
            )
    except Exception:
        pass

FANTASYCALC_URL = (
    "https://api.fantasycalc.com/values/current"
    "?isDynasty=true&numQbs=1&numTeams=10&ppr=1"
)

CORE_POSITIONS = {"QB", "RB", "WR", "TE"}


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
# Advanced metrics loader
# ------------------------------------------------

def load_advanced_metrics_df() -> pd.DataFrame:
    """
    Load advanced efficiency metrics from the database.

    Returns dataframe with player_id (renamed to sleeper_id) and all metrics.
    Uses most recent available data (current or previous season).
    Falls back to empty dataframe if metrics aren't available.
    """
    try:
        from data_building.advanced_metrics import get_player_metrics
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            # Get latest date with metrics (regardless of season)
            latest = conn.execute("""
                SELECT MAX(as_of_date) as max_date
                FROM player_advanced_metrics
            """).fetchone()

            if not latest or not latest["max_date"]:
                print("[value_model] No advanced metrics available yet")
                return pd.DataFrame()

            latest_date = latest["max_date"]

            # Load all metrics for latest date, including rookie evaluation columns.
            # rookie_eval_* columns are null for established NFL players (default 0 in model)
            # and populated for current draft-class prospects.
            rows = conn.execute("""
                SELECT
                    player_id,
                    yards_per_target,
                    catch_rate,
                    yards_per_reception,
                    target_quality_score,
                    yards_per_carry,
                    yards_per_touch,
                    rush_td_rate,
                    yards_per_attempt,
                    completion_pct,
                    td_rate,
                    int_rate,
                    snap_share,
                    opportunity_share,
                    red_zone_usage,
                    role_score,
                    usage_trend,
                    efficiency_trend,
                    -- Rookie evaluation columns (null for non-rookies; filled with 0 below)
                    rookie_eval_routes_run,
                    rookie_eval_yprr,
                    rookie_eval_tprr,
                    rookie_eval_yac_per_att,
                    rookie_eval_mtf_per_att,
                    rookie_eval_explosive_run_rate,
                    rookie_eval_adjusted_comp_pct,
                    rookie_eval_twp_rate,
                    rookie_eval_player_level_sos,
                    rookie_eval_perf_vs_top_def,
                    CASE WHEN rookie_eval_true_early_declare THEN 1 ELSE 0 END
                        AS rookie_eval_true_early_declare_flag,
                    rookie_eval_draft_class_year,
                    rookie_eval_completeness,
                    rookie_eval_prospect_score,
                    CASE WHEN rookie_eval_is_rookie THEN 1 ELSE 0 END
                        AS rookie_eval_is_rookie_flag
                FROM player_advanced_metrics
                WHERE as_of_date = %s
            """, (latest_date,)).fetchall()

            df = pd.DataFrame([dict(row) for row in rows])

            if df.empty:
                print("[value_model] Advanced metrics table is empty")
                return pd.DataFrame()

            # Rename player_id to sleeper_id for joining
            df = df.rename(columns={"player_id": "sleeper_id"})
            df["sleeper_id"] = df["sleeper_id"].astype(str)

            # Check if metrics are from previous season
            from datetime import datetime, date as dt_date
            today = dt_date.today()
            metrics_date = datetime.strptime(str(latest_date), "%Y-%m-%d").date()
            days_old = (today - metrics_date).days

            if days_old > 30:
                print(
                    f"[value_model] Using advanced metrics from {latest_date} ({days_old} days old - likely previous season)")
            else:
                print(f"[value_model] Loaded {len(df)} players with current advanced metrics")

            return df

    except Exception as e:
        print(f"[value_model] Failed to load advanced metrics: {e}")
        return pd.DataFrame()


# ------------------------------------------------
# Internal stats loader
# ------------------------------------------------

def load_internal_stats_df() -> pd.DataFrame:
    value_path = DATA_DIR / "usage_table.json"
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

# ------------------------------------------------
# Draft values helpers
# ------------------------------------------------

# ------------------------------------------------
# Training data builder
# ------------------------------------------------

# ------------------------------------------------
# Target normalization helper
# ------------------------------------------------

# ------------------------------------------------
# Model training
# ------------------------------------------------

# ------------------------------------------------
# Inference helpers
# ------------------------------------------------

def build_inference_dataframe() -> pd.DataFrame:
    current_season = _current_season_from_state()

    internal_df = load_internal_stats_df()
    history_features_df = load_history_feature_df(current_season)
    investment_df = load_player_investment_df()
    advanced_metrics_df = load_advanced_metrics_df()

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

    # ADVANCED METRICS: Merge efficiency metrics
    if advanced_metrics_df is not None and not advanced_metrics_df.empty:
        df = df.merge(advanced_metrics_df, on="sleeper_id", how="left")

    if "age" not in df.columns:
        if "fc_age" in df.columns:
            df["age"] = pd.to_numeric(df["fc_age"], errors="coerce")
        elif "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")

    return df


# ------------------------------------------------
# Rewrite usage table with model outputs
# ------------------------------------------------

def rewrite_value_table_with_model() -> Path:
    source_path = DATA_DIR / "usage_table.json"
    if not source_path.exists():
        raise FileNotFoundError(f"No usage table file at {source_path}")

    with source_path.open("r", encoding="utf-8") as f:
        players = json.load(f)

    inference_df = build_inference_dataframe()

    # Load WLS calibrated SF values (produced by trade_value_model.py).
    # These are market-derived QB SF values from real dynasty trades.
    # Used instead of a hardcoded boost so QB SF values reflect actual trade markets.
    wls_sf_values: dict[str, float] = {}
    try:
        from dashboard_services.db import get_conn as _get_conn
        with _get_conn() as _conn:
            _wls_rows = _conn.execute(
                "SELECT player_id, calibrated_value_sf FROM player_values "
                "WHERE calibrated_value_sf IS NOT NULL AND calibrated_value_sf > 0"
            ).fetchall()
        wls_sf_values = {str(r["player_id"]): float(r["calibrated_value_sf"]) for r in _wls_rows}
        print(f"[rewrite_value_table] Loaded {len(wls_sf_values)} WLS SF values from DB")
    except Exception as _wls_err:
        print(f"[rewrite_value_table] WLS SF load skipped: {_wls_err}")

    # CRITICAL FIX: Load vendor values to use directly when available
    fc_df = load_fantasycalc_df()
    dp_df = load_dynastyprocess_df()

    # Build vendor value lookup for 1QB
    vendor_values: dict[str, float] = {}

    # Get FC values normalized to a 0-999.9 scale (lowest FC player → 0, highest → 999.9)
    if not fc_df.empty and "fc_value" in fc_df.columns and "sleeper_id" in fc_df.columns:
        fc_values_nonzero = fc_df["fc_value"][fc_df["fc_value"] > 0]
        fc_max = fc_values_nonzero.max()
        fc_min = fc_values_nonzero.min()
        fc_range = max(fc_max - fc_min, 1.0)

        for _, row in fc_df.iterrows():
            pid = str(row.get("sleeper_id"))
            fc_val = row.get("fc_value")
            if pid and pd.notna(fc_val) and float(fc_val) > 0:
                # Min-max normalization to 0-999.9 (consistent with engine + DP sources).
                vendor_values[pid] = (float(fc_val) - fc_min) / fc_range * 999.9
    else:
        print("[DEBUG] vendor_values section SKIPPED due to missing conditions")

    # Load FC SF (numQbs=2) values — gives QBs their proper SF market premium.
    # Min-max normalized to 0-999.9 (same scale as the 1QB FC values).
    fc_sf_by_sid: dict[str, float] = {}
    try:
        from data_building.external_data.external_values_scraper import load_fantasycalc_sf_api_values
        _fc_sf_rows = load_fantasycalc_sf_api_values() or []
        if _fc_sf_rows:
            _fc_sf_vals = [float(r["value"]) for r in _fc_sf_rows if r.get("value") and float(r["value"]) > 0]
            if _fc_sf_vals:
                _fc_sf_max = max(_fc_sf_vals)
                _fc_sf_min = min(_fc_sf_vals)
                _fc_sf_range = max(_fc_sf_max - _fc_sf_min, 1.0)
                for _r in _fc_sf_rows:
                    _sid = str(_r.get("sleeper_id") or "").strip()
                    _v = _r.get("value")
                    if _sid and _v and float(_v) > 0:
                        fc_sf_by_sid[_sid] = (float(_v) - _fc_sf_min) / _fc_sf_range * 999.9
            print(f"[rewrite_value_table] Loaded {len(fc_sf_by_sid)} FC SF values (numQbs=2)")
        else:
            print("[rewrite_value_table] FC SF CSV missing — SF QB premium will use DP 2QB only")
    except Exception as _e:
        print(f"[rewrite_value_table] FC SF load failed: {_e}")

    # Build Superflex vendor value lookup using sf_engine_value + value_2qb
    sf_vendor_values: dict[str, float] = {}

    # Load engine_values CSV (contains both 1QB + SF values for all league sizes)
    engine_values_path = DATA_DIR / "engine_values.csv"
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

    # Load base 10-team engine values for blending with FC vendor values (1QB)
    engine_1qb_map: dict[str, float] = {}
    if engine_values_path.exists():
        try:
            engine_df_base = pd.read_csv(engine_values_path)
            for _, row in engine_df_base.iterrows():
                pid = str(row.get("player_id"))
                eng_val = row.get("engine_value_10")
                if pid and pd.notna(eng_val):
                    engine_1qb_map[pid] = float(eng_val)
        except Exception as e:
            print(f"[ERROR] Failed to load engine_1qb_map: {e}")

    # Load value_2qb from dynastyprocess (need to match by name+team)
    dp_2qb_map: dict[tuple[str, str], float] = {}  # (name, team) -> value_2qb
    dp_df_full = pd.DataFrame()  # Full DP dataframe for outlier detection
    # Always try to load DP dataframe for outlier detection
    try:
        dp_raw = pd.read_csv(DATA_DIR / "dynastyprocess_values.csv")
        if "player" in dp_raw.columns and "value_1qb" in dp_raw.columns:
            dp_df_full = dp_raw
    except Exception as e:
        print(f"[ERROR] Failed to load dp_df_full: {e}")

    # Pre-compute DP normalisation bounds (floor-at-100: min→100, max→999.9)
    _dp_1qb_vals = dp_df_full["value_1qb"].dropna() if not dp_df_full.empty and "value_1qb" in dp_df_full.columns else pd.Series(dtype=float)
    _dp_1qb_vals = _dp_1qb_vals[_dp_1qb_vals > 0]
    DP_1QB_MIN: float = float(_dp_1qb_vals.min()) if len(_dp_1qb_vals) else 1.0
    DP_1QB_MAX: float = float(_dp_1qb_vals.max()) if len(_dp_1qb_vals) else 10256.0
    DP_1QB_RANGE: float = max(DP_1QB_MAX - DP_1QB_MIN, 1.0)

    # NOTE: dp_df_full intentionally kept from the load above - it is used below
    # to look up per-player DP value_1qb for vendor consensus.  Do NOT reset it here.
    try:
        dp_raw = pd.read_csv(DATA_DIR / "dynastyprocess_values.csv")
        dp_df_full = dp_raw.copy()  # Populate dp_df_full with the actual data
        if "player" in dp_raw.columns and "value_2qb" in dp_raw.columns:
            for _, row in dp_raw.iterrows():
                name = str(row.get("player", "")).strip()
                team = str(row.get("team", "")).strip()
                val_2qb = row.get("value_2qb")
                if name and pd.notna(val_2qb):
                    dp_2qb_map[(name, team)] = float(val_2qb)
    except Exception as e:
        print(f"[ERROR] Failed to load value_2qb from dynastyprocess: {e}")

    # Calculate Superflex vendor values: blend FC SF (50%), DP 2QB (30%), SF Engine (20%).
    # Min-max normalize DP 2QB to 0-999.9 so it shares the scale of the other sources.
    _dp_2qb_vals = [v for v in dp_2qb_map.values() if v > 0]
    dp_2qb_max = max(_dp_2qb_vals) if _dp_2qb_vals else 1.0
    dp_2qb_min = min(_dp_2qb_vals) if _dp_2qb_vals else 0.0
    dp_2qb_range = max(dp_2qb_max - dp_2qb_min, 1.0)

    # Build sf_vendor_values for each player
    players_index = load_players_index() or {}
    for pid_key, meta in players_index.items():
        pid = str(pid_key)
        name = meta.get("name", "").strip()
        team = meta.get("team", "").strip()

        # Use FC SF (numQbs=2) as the vendor signal for SF blending.
        # Falls back to 1QB FC only when SF CSV is unavailable.
        fc_sf_norm = fc_sf_by_sid.get(pid, 0.0)
        fc_for_sf  = fc_sf_norm if fc_sf_norm > 0 else vendor_values.get(pid, 0.0)

        # Get SF engine value
        sf_eng_val = sf_engine_map.get(pid, 0.0)

        # Get DP 2QB value (min-max normalized to 0-999.9)
        dp_2qb_raw = dp_2qb_map.get((name, team), 0.0)
        dp_2qb_norm = ((dp_2qb_raw - dp_2qb_min) / dp_2qb_range * 999.9) if dp_2qb_raw > 0 else 0.0

        # Superflex blend: 50% FC SF, 30% DP 2QB, 20% SF engine.
        # FC SF (numQbs=2) is now the primary signal — it directly encodes QB scarcity.
        # Renormalize when a source is missing so values aren't deflated.
        if fc_for_sf > 0 or sf_eng_val > 0 or dp_2qb_norm > 0:
            SF_W_VENDOR, SF_W_DP, SF_W_ENGINE = 0.50, 0.30, 0.20
            sf_wsum = 0.0
            sf_wtot = 0.0
            if fc_for_sf > 0:
                sf_wsum += SF_W_VENDOR * fc_for_sf; sf_wtot += SF_W_VENDOR
            if dp_2qb_norm > 0:
                sf_wsum += SF_W_DP * dp_2qb_norm;   sf_wtot += SF_W_DP
            if sf_eng_val > 0:
                sf_wsum += SF_W_ENGINE * sf_eng_val; sf_wtot += SF_W_ENGINE
            sf_value = sf_wsum / sf_wtot if sf_wtot > 0 else 0.0

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

        # Gather all three value sources, now all on the same 0-999.9 scale:
        #   FC (vendor), DP (vendor, non-TEs only), internal engine.
        # They are combined with the fixed weighted blend below (renormalized when a
        # source is missing); there is no separate outlier rule.
        player_position = str(player.get("position") or "").upper()

        # Resolve DP value for this player (matched by name + team, no sleeper_id in DP)
        dp_val_raw = 0.0
        player_row = df_by_id.get(pid)
        if player_row is not None:
            p_name = str(player_row.get('name', '')).strip()
            p_team = str(player_row.get('team', '')).strip()
            if not dp_df_full.empty:
                dp_match = dp_df_full[
                    (dp_df_full['player'].str.lower() == p_name.lower()) &
                    (dp_df_full['team'].str.lower() == p_team.lower())
                ]
                if not dp_match.empty:
                    dp_val_raw = float(dp_match.iloc[0]['value_1qb'])
        # Normalise DP to 0-999.9 (min → 0, max → 999.9), matching the other sources.
        dp_norm = ((dp_val_raw - DP_1QB_MIN) / DP_1QB_RANGE * 999.9) if dp_val_raw > 0 else 0.0

        fc_val  = vendor_values.get(pid, 0.0)
        # DP undervalues TEs vs market consensus; exclude for that position.
        dp_val  = dp_norm if (dp_norm > 0 and player_position != "TE") else 0.0
        # Use None to distinguish "no data" (rookie/prospect) from "0 production" (known bad).
        # Players in the engine table with 0 production should have that zero count against them.
        eng_val = float(engine_1qb_map[pid]) if pid in engine_1qb_map else None

        # Fixed weights: 40% vendor (FantasyCalc), 40% engine, 20% DP.
        # DP and FC are dropped (renormalized) when missing - they may simply not cover a player.
        # Engine is dropped only when the player has NO engine record (pure prospect with no NFL data).
        # If a player IS in the engine table with 0 production, that zero is included in the blend
        # so FC hype can't inflate them past what their usage actually supports.
        W_VENDOR, W_ENGINE, W_DP = 0.40, 0.40, 0.20
        weighted_sum = 0.0
        total_weight = 0.0
        if fc_val > 0:
            weighted_sum += W_VENDOR * fc_val
            total_weight += W_VENDOR
        if eng_val is not None:
            weighted_sum += W_ENGINE * eng_val
            total_weight += W_ENGINE
        if dp_val > 0:
            weighted_sum += W_DP * dp_val
            total_weight += W_DP

        if total_weight > 0:
            final_value = weighted_sum / total_weight
        else:
            # No vendor (FC/DP) and no engine record for this player — there's no
            # signal to value them, so leave them at 0 (previously a rarely-used
            # ML fallback).
            final_value = 0.0

        # Calculate Superflex value - use engine values as primary source
        if pid in sf_engine_map:
            sf_value = sf_engine_map[pid]
        elif pid in sf_vendor_values:
            sf_value = sf_vendor_values[pid]
        else:
            # No SF engine value and no SF vendor blend → no SF signal.
            sf_value = 0.0

        # Non-QB players are not less valuable in SF - QBs go up, everyone else stays the same.
        # Floor non-QB sf_value at their 1QB value to prevent the DP 2QB blend from pulling them down.
        position = player.get("position")
        if position != "QB":
            sf_value = max(sf_value, final_value)
        elif pid in wls_sf_values:
            # WLS market value can only improve (never downgrade) the engine's SF value.
            # The sf_engine already gives top QBs (Allen: 999.9) their correct SF premium.
            # WLS refines this with real trade data but shouldn't override a higher engine value.
            sf_value = max(sf_value, wls_sf_values[pid])

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
            # A missing size-specific engine value means "no scarcity data," which
            # should leave the player unadjusted (ratio 1.0) — not zero them out.
            # Guard the numerator too, else eng_n == 0 collapses value_{n} to 0.
            ratio = (eng_n / eng_base) if (eng_base > 0 and eng_n > 0) else 1.0
            sf_ratio = (sf_eng_n / sf_eng_base) if (sf_eng_base > 0 and sf_eng_n > 0) else 1.0
            size_values[f"value_{n}"] = round(min(float(final_value) * ratio, 999.9), 1)
            sf_size_values[f"sf_value_{n}"] = round(min(float(sf_value) * sf_ratio, 999.9), 1)

        asset = {
            "id": player.get("id"),
            "name": name,
            "team": player.get("team"),
            "position": player.get("position"),
            "age": age,
            "value": round(min(float(final_value), 999.9), 1),
            "sf_value": round(min(float(sf_value), 999.9), 1),
            "search_name": normalize_name(name) if name else "",
            "pos_rank": None,
            "pos_rank_label": None,
            "sf_pos_rank": None,
            "sf_pos_rank_label": None,
            "rank_change_7d": None,
            "pos_rank_change_7d": None,
        }
        asset.update(size_values)
        asset.update(sf_size_values)
        cleaned_assets.append(asset)

    pick_values = load_pick_value_table() or {}

    # Track which (year, round) pairs have slot entries (YYYY_R_NN where NN is numeric).
    # For those pairs, skip bucket picks — slots are more precise.
    # Also track bucket pairs to skip redundant plain YYYY_R generic keys.
    slot_pairs: set[tuple[int, int]] = set()
    bucket_pairs: set[tuple[int, int]] = set()
    for key in pick_values:
        parts = key.split("_")
        if len(parts) == 3:
            try:
                yr, rnd = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if parts[2].isdigit():
                slot_pairs.add((yr, rnd))
            else:
                bucket_pairs.add((yr, rnd))

    for key, val in pick_values.items():
        parts = key.split("_")

        name = None

        # Exact slotted pick format: YYYY_R_PP  ->  2026 1.01
        if len(parts) == 3 and parts[2].isdigit():
            try:
                year, rnd, pick_in_round = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if rnd > 5:
                continue
            name = f"{year} {rnd}.{pick_in_round:02d}"

        # Bucketed format: YYYY_R_bucket  ->  2027 1st (Early)
        # Skip if slot picks already exist for this year/round (slots are more precise)
        elif len(parts) == 3:
            try:
                year, rnd = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if rnd > 5:
                continue
            if (year, rnd) in slot_pairs:
                continue
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
            name = f"{year} {rnd}{suffix} ({parts[2].capitalize()})"

        # Plain round-only format: YYYY_R — skip if slot or bucket entries exist for this pair
        elif len(parts) == 2:
            try:
                year, rnd = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if rnd > 5:
                continue
            if (year, rnd) in slot_pairs or (year, rnd) in bucket_pairs:
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
            "rank_change_7d": None,
            "pos_rank_change_7d": None,
        }
        # Draft picks are not scarcity-sensitive to league size - same value in all sizes
        for n in LEAGUE_SIZES:
            if n != 10:
                pick_asset[f"value_{n}"] = float(val)
                pick_asset[f"sf_value_{n}"] = float(val)
        cleaned_assets.append(pick_asset)

    # Normalize 1QB value scale so the top non-QB player always equals 999.9.
    # In 1QB leagues QBs are not premium, so the best RB/WR/TE should anchor the
    # ceiling.  This replaces the old _max_sf > _max_1qb approach which silently
    # skipped normalization whenever the QB SF engine value failed to load.
    #
    # The raw scale is EMA-smoothed against the previous run's scale so that
    # day-to-day noise in trade data (different trades sampled each day) doesn't
    # cause the entire roster to jump in value overnight.
    _SKILL_POS   = {"QB", "RB", "WR", "TE"}
    _NON_QB_POS  = {"RB", "WR", "TE"}
    _1qb_keys    = ["value", "value_8", "value_12", "value_14"]
    _non_qb_vals = sorted(
        (float(a.get("value") or 0) for a in cleaned_assets
         if str(a.get("position") or "").upper() in _NON_QB_POS),
        reverse=True,
    )
    _max_non_qb = _non_qb_vals[0] if _non_qb_vals else 0.0
    # Run whenever there are non-QBs. The old `< 999.9` guard skipped the basket
    # scaling entirely once the top player hit the 999.9 base cap (lines ~1016),
    # which pinned the #1 to a flat 999.9 and never let them float above it.
    if _max_non_qb > 0:
        # Basket anchor: the mean of the top N non-QBs is the ceiling estimate.
        # One player drifting moves this ~N× less than it moves the single max,
        # so the rest of the board no longer slides when the top player ticks up.
        _topN   = _non_qb_vals[:_ANCHOR_BASKET_N]
        _basket = sum(_topN) / len(_topN)
        _raw_headroom = _max_non_qb / _basket if _basket > 0 else 1.0

        # Smooth the basket (tracks genuine tier drift over ~2 weeks) and the
        # headroom ratio (near-fixed, so a single #1 pulling away rides the cap
        # instead of compressing everyone). Both persist durably in Postgres.
        _prev_basket   = _load_state(_BASKET_STATE_KEY)
        _prev_headroom = _load_state(_HEADROOM_STATE_KEY)
        _smoothed_basket = (
            _BASKET_EMA_ALPHA * _basket + (1.0 - _BASKET_EMA_ALPHA) * _prev_basket
            if _prev_basket > 0 else _basket
        )
        _smoothed_headroom = (
            _HEADROOM_EMA_ALPHA * _raw_headroom + (1.0 - _HEADROOM_EMA_ALPHA) * _prev_headroom
            if _prev_headroom > 0 else _raw_headroom
        )
        _save_state(_BASKET_STATE_KEY, _smoothed_basket)
        _save_state(_HEADROOM_STATE_KEY, _smoothed_headroom)

        # Anchor the BASKET MEAN (mean of top-N non-QBs) to 999.9 instead of
        # pinning the single #1. This lets the #1 float above 999.9 proportional
        # to how far ahead of the basket they are — their sparkline moves — while
        # the rest of the board no longer compresses whenever the anchor shifts.
        # No hard ceiling: the smoothed basket provides a soft cap naturally.
        _1qb_scale = 999.9 / _smoothed_basket if _smoothed_basket > 0 else 1.0
        _persist_scale(_1qb_scale)  # kept for back-compat / the reset script
        for _a in cleaned_assets:
            if str(_a.get("position") or "").upper() not in _SKILL_POS:
                continue
            for _k in _1qb_keys:
                if _a.get(_k) is not None:
                    _a[_k] = round(float(_a[_k]) * _1qb_scale, 1)

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

    # Calculate 7-day rank changes before writing JSON
    from data_building.update_player_values_with_rankings import _load_historical_ranks

    # Calculate current overall ranks
    player_assets = [a for a in cleaned_assets if a.get("position") != "PICK"]
    player_assets.sort(key=lambda a: float(a.get("value") or 0.0), reverse=True)

    # Build current rank maps
    current_overall_ranks = {}
    for overall_rank, asset in enumerate(player_assets, start=1):
        pid = asset.get("id")
        if pid:
            current_overall_ranks[str(pid)] = overall_rank

    # Calculate position ranks
    current_pos_ranks = {}
    pos_sorted_assets = {}
    for asset in player_assets:
        pos = asset.get("position")
        if pos:
            pos_sorted_assets.setdefault(pos, []).append(asset)

    for pos, assets in pos_sorted_assets.items():
        assets.sort(key=lambda a: float(a.get("value") or 0.0), reverse=True)
        for pos_rank, asset in enumerate(assets, start=1):
            pid = asset.get("id")
            if pid:
                current_pos_ranks.setdefault(pos, {})[str(pid)] = pos_rank

    # Load historical ranks from 7 days ago
    seven_days_ago = date.today() - timedelta(days=7)
    hist_ranks = _load_historical_ranks(seven_days_ago)

    # Add rank changes to all assets
    for asset in cleaned_assets:
        pid = str(asset.get("id") or "")
        if asset.get("position") == "PICK":
            # Draft picks don't have rank changes
            asset["rank_change_7d"] = None
            asset["pos_rank_change_7d"] = None
        else:
            cur_overall = current_overall_ranks.get(pid)
            cur_pos = current_pos_ranks.get(asset.get("position"), {}).get(pid)

            hist = hist_ranks.get(pid)
            if hist and cur_overall is not None:
                # Positive = moved up (lower rank number is better)
                asset["rank_change_7d"] = hist["overall_rank"] - cur_overall
            else:
                asset["rank_change_7d"] = None

            if hist and cur_pos is not None:
                asset["pos_rank_change_7d"] = hist["pos_rank"] - cur_pos
            else:
                asset["pos_rank_change_7d"] = None

    out_path = DATA_DIR / "model_values.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_assets, f, ensure_ascii=False, indent=2)

    return out_path
