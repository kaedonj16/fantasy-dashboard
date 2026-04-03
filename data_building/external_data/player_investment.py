from __future__ import annotations

import re
import time
from datetime import date
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import requests

from cache.paths import PLAYER_INVESTMENT_DIR
from utils.utils import normalize_name, load_players_index

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# nflverse release files
NFLVERSE_PLAYERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
)
NFLVERSE_DRAFT_PICKS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.parquet"
)

OTC_POSITION_URLS = {
    "QB": "https://overthecap.com/position/quarterback",
    "RB": "https://overthecap.com/position/running-back",
    "WR": "https://overthecap.com/position/wide-receiver",
    "TE": "https://overthecap.com/position/tight-end",
}

DRAFT_HISTORY_PATH = PLAYER_INVESTMENT_DIR / "draft_history.parquet"
CONTRACTS_LATEST_PATH = PLAYER_INVESTMENT_DIR / "contracts_latest.parquet"
PLAYER_INVESTMENT_LATEST_PATH = PLAYER_INVESTMENT_DIR / "player_investment_latest.parquet"


def _normalize_name_key(name: str) -> str:
    if not name:
        return ""
    return normalize_name(str(name))


def _money_to_float(val) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except Exception:
        return 0.0


def _safe_get(url: str, sleep_s: float = 0.4) -> bytes:
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=45)
    resp.raise_for_status()
    if sleep_s > 0:
        time.sleep(sleep_s)
    return resp.content


def _draft_capital_score(round_num: Optional[int], pick_num: Optional[int]) -> float:
    """
    0..1 scale.
    Earlier picks score much higher.
    """
    if round_num is None or pick_num is None:
        return 0.0

    try:
        round_num = int(round_num)
        pick_num = int(pick_num)
    except Exception:
        return 0.0

    score = 1.0 / (1.0 + (max(pick_num, 1) - 1) / 32.0)

    if round_num == 1:
        score *= 1.00
    elif round_num == 2:
        score *= 0.92
    elif round_num == 3:
        score *= 0.84
    elif round_num <= 5:
        score *= 0.68
    else:
        score *= 0.52

    return max(0.0, min(1.0, float(score)))


def _coalesce_col(df: pd.DataFrame, candidates: list[str], default=None):
    for col in candidates:
        if col in df.columns:
            return df[col]
    if default is None:
        return pd.Series([None] * len(df), index=df.index)
    return pd.Series([default] * len(df), index=df.index)


def load_nflverse_players_draft_history() -> pd.DataFrame:
    """
    Preferred draft loader.

    Uses nflverse players.parquet, which already includes player-level draft info.
    This is much more stable than scraping PFR.
    """
    print("[player_investment] loading nflverse players draft data")
    raw = _safe_get(NFLVERSE_PLAYERS_URL, sleep_s=0.0)
    df = pd.read_parquet(BytesIO(raw))

    if df.empty:
        return pd.DataFrame()

    name_series = _coalesce_col(df, ["display_name", "player_name", "full_name", "name"])
    team_series = _coalesce_col(df, ["team", "recent_team", "current_team"])
    pos_series = _coalesce_col(df, ["position", "pos"])
    draft_year_series = _coalesce_col(df, ["draft_year"])
    draft_round_series = _coalesce_col(df, ["draft_round"])
    draft_pick_series = _coalesce_col(df, ["draft_pick"])
    draft_team_series = _coalesce_col(df, ["draft_team"])

    out = pd.DataFrame({
        "name": name_series.astype(str).str.strip(),
        "name_key": name_series.astype(str).map(_normalize_name_key),
        "draft_year": pd.to_numeric(draft_year_series, errors="coerce").astype("Int64"),
        "draft_round": pd.to_numeric(draft_round_series, errors="coerce").astype("Int64"),
        "draft_pick": pd.to_numeric(draft_pick_series, errors="coerce").astype("Int64"),
        "draft_team": draft_team_series,
        "draft_position": pos_series,
        "current_team_hint": team_series,
    })

    out = out[out["name_key"] != ""].copy()

    out["draft_capital_score"] = out.apply(
        lambda r: _draft_capital_score(r.get("draft_round"), r.get("draft_pick")),
        axis=1,
    )

    # keep best / most informative row per player
    out = out.sort_values(
        by=["draft_year", "draft_pick"],
        ascending=[False, True],
        na_position="last",
    ).drop_duplicates(subset=["name_key"], keep="first")

    return out.reset_index(drop=True)


def load_nflverse_draft_picks_history(
        start_year: int = 2014,
        end_year: int | None = None,
) -> pd.DataFrame:
    """
    Optional fallback / secondary source.

    Uses nflverse draft_picks.parquet.
    """
    print("[player_investment] loading nflverse draft_picks history")
    raw = _safe_get(NFLVERSE_DRAFT_PICKS_URL, sleep_s=0.0)
    df = pd.read_parquet(BytesIO(raw))

    if df.empty:
        return pd.DataFrame()

    if end_year is None:
        end_year = date.today().year

    season_series = _coalesce_col(df, ["season", "year", "draft_year"])
    name_series = _coalesce_col(df, ["player_name", "name", "display_name"])
    round_series = _coalesce_col(df, ["round", "draft_round"])
    pick_series = _coalesce_col(df, ["pick", "overall", "draft_pick"])
    team_series = _coalesce_col(df, ["team", "team_abbr", "draft_team"])
    pos_series = _coalesce_col(df, ["position", "pos"])

    out = pd.DataFrame({
        "name": name_series.astype(str).str.strip(),
        "name_key": name_series.astype(str).map(_normalize_name_key),
        "draft_year": pd.to_numeric(season_series, errors="coerce").astype("Int64"),
        "draft_round": pd.to_numeric(round_series, errors="coerce").astype("Int64"),
        "draft_pick": pd.to_numeric(pick_series, errors="coerce").astype("Int64"),
        "draft_team": team_series,
        "draft_position": pos_series,
    })

    out = out[out["name_key"] != ""].copy()
    out = out[
        out["draft_year"].notna()
        & (out["draft_year"] >= start_year)
        & (out["draft_year"] <= end_year)
        ].copy()

    out["draft_capital_score"] = out.apply(
        lambda r: _draft_capital_score(r.get("draft_round"), r.get("draft_pick")),
        axis=1,
    )

    out = out.sort_values(
        by=["draft_year", "draft_pick"],
        ascending=[False, True],
        na_position="last",
    ).drop_duplicates(subset=["name_key"], keep="first")

    return out.reset_index(drop=True)


def load_draft_history(
        start_year: int = 2014,
        end_year: int | None = None,
) -> pd.DataFrame:
    """
    Main draft loader:
    1) nflverse players dataset
    2) fallback to nflverse draft_picks dataset
    """
    try:
        players_df = load_nflverse_players_draft_history()
        if not players_df.empty:
            return players_df
    except Exception as e:
        print(f"[player_investment] nflverse players draft load failed: {e}")

    try:
        draft_df = load_nflverse_draft_picks_history(start_year=start_year, end_year=end_year)
        if not draft_df.empty:
            return draft_df
    except Exception as e:
        print(f"[player_investment] nflverse draft_picks load failed: {e}")

    return pd.DataFrame()


def finalize_contract_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    needed_cols = [
        "contract_total_value",
        "guaranteed_money",
        "fully_guaranteed_money",
        "contract_apy",
        "contract_years",
        "free_agency_year",
    ]
    for col in needed_cols:
        if col not in out.columns:
            out[col] = 0.0 if col != "free_agency_year" else np.nan

    out["contract_total_value"] = out["contract_total_value"].map(_money_to_float)
    out["guaranteed_money"] = out["guaranteed_money"].map(_money_to_float)
    out["fully_guaranteed_money"] = out["fully_guaranteed_money"].map(_money_to_float)
    out["contract_apy"] = out["contract_apy"].map(_money_to_float)
    out["contract_years"] = pd.to_numeric(out["contract_years"], errors="coerce").fillna(0.0)
    out["free_agency_year"] = pd.to_numeric(out["free_agency_year"], errors="coerce")

    current_year = date.today().year

    # Fallback 1:
    # If contract years missing but FA year exists, estimate years remaining.
    years_mask = (
            (out["contract_years"] <= 0)
            & out["free_agency_year"].notna()
    )
    out.loc[years_mask, "contract_years"] = (
            out.loc[years_mask, "free_agency_year"] - current_year
    ).clip(lower=0)

    # Optional improvement:
    # count current season too so 2027 FA in 2026 implies ~2 contract seasons left
    inclusive_years_mask = (
            (out["contract_years"] <= 1)
            & out["free_agency_year"].notna()
            & (out["contract_total_value"] > 0)
    )
    out.loc[inclusive_years_mask, "contract_years"] = (
            out.loc[inclusive_years_mask, "free_agency_year"] - current_year + 1
    ).clip(lower=1)

    # Fallback 2:
    # If APY missing but total + years exist, compute APY.
    apy_mask = (
            (out["contract_apy"] <= 0)
            & (out["contract_total_value"] > 0)
            & (out["contract_years"] > 0)
    )
    out.loc[apy_mask, "contract_apy"] = (
            out.loc[apy_mask, "contract_total_value"] / out.loc[apy_mask, "contract_years"]
    )

    out["guaranteed_pct"] = np.where(
        out["contract_total_value"] > 0,
        out["guaranteed_money"] / out["contract_total_value"],
        0.0,
    )

    return out


def scrape_otc_contracts() -> pd.DataFrame:
    rows: list[dict] = []

    for pos, url in OTC_POSITION_URLS.items():
        print(f"[player_investment] scraping OTC contracts: {pos}")
        try:
            tables = pd.read_html(url)
        except Exception as e:
            print(f"[player_investment] failed OTC scrape for {pos}: {e}")
            continue

        if not tables:
            continue

        df = tables[0].copy()

        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            flat_cols = []
            for col in df.columns:
                parts = [
                    str(x).strip()
                    for x in col
                    if str(x).strip() and str(x).strip().lower() != "nan"
                ]
                flat_cols.append(" ".join(parts))
            df.columns = flat_cols
        else:
            df.columns = [str(c).strip() for c in df.columns]

        print(f"[player_investment] OTC {pos} raw columns: {list(df.columns)}")

        normalized = {}
        for c in df.columns:
            lc = str(c).strip().lower()

            if lc == "player":
                normalized[c] = "name"
            elif lc == "team":
                normalized[c] = "team"
            elif lc == "age":
                normalized[c] = "age"

            elif (
                    "total value" in lc
                    or "contract value" in lc
                    or lc == "value"
            ):
                normalized[c] = "contract_total_value"

            elif (
                    lc == "apy"
                    or lc == "aav"
                    or "avg/year" in lc
                    or "avg / year" in lc
                    or "average per year" in lc
                    or "per year" in lc
                    or "annual value" in lc
                    or "apy/aav" in lc
            ):
                normalized[c] = "contract_apy"

            elif (
                    lc == "years"
                    or lc == "year"
                    or "contract years" in lc
                    or "length" in lc
                    or "term" in lc
                    or "yrs" in lc
            ):
                normalized[c] = "contract_years"

            elif "fully guaranteed" in lc:
                normalized[c] = "fully_guaranteed_money"

            elif "guaranteed at signing" in lc:
                normalized[c] = "guaranteed_money"

            elif "total guaranteed" in lc or ("guaranteed" in lc and "fully" not in lc):
                normalized[c] = "guaranteed_money"

            elif "free agency" in lc:
                normalized[c] = "free_agency"

        df = df.rename(columns=normalized)

        if "name" not in df.columns:
            print(f"[player_investment] OTC {pos}: could not find player column after normalization")
            continue

        keep_cols = [
            c for c in [
                "name",
                "team",
                "age",
                "contract_total_value",
                "contract_apy",
                "contract_years",
                "guaranteed_money",
                "fully_guaranteed_money",
                "free_agency",
            ]
            if c in df.columns
        ]

        df = df[keep_cols].copy()
        df["position"] = pos
        df["name_key"] = df["name"].apply(_normalize_name_key)

        for col in [
            "contract_total_value",
            "contract_apy",
            "contract_years",
            "guaranteed_money",
            "fully_guaranteed_money",
            "age",
        ]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].apply(_money_to_float)

        if "free_agency" in df.columns:
            def parse_fa_year(v):
                m = re.search(r"(20\d{2})", str(v))
                return int(m.group(1)) if m else None

            df["free_agency_year"] = df["free_agency"].apply(parse_fa_year)
        else:
            df["free_agency_year"] = None

        # Fallback 1: infer years from FA year
        current_year = date.today().year
        years_mask = (
                (df["contract_years"] <= 0)
                & df["free_agency_year"].notna()
                & (df["contract_total_value"] > 0)
        )
        df.loc[years_mask, "contract_years"] = (
                df.loc[years_mask, "free_agency_year"] - current_year + 1
        ).clip(lower=1)

        # Fallback 2: infer APY from total / years
        apy_mask = (
                (df["contract_apy"] <= 0)
                & (df["contract_total_value"] > 0)
                & (df["contract_years"] > 0)
        )
        df.loc[apy_mask, "contract_apy"] = (
                df.loc[apy_mask, "contract_total_value"] / df.loc[apy_mask, "contract_years"]
        )

        print(
            f"[player_investment] OTC {pos}: "
            f"rows={len(df)} "
            f"with_total={(df['contract_total_value'] > 0).sum()} "
            f"with_apy={(df['contract_apy'] > 0).sum()} "
            f"with_years={(df['contract_years'] > 0).sum()}"
        )

        rows.extend(df.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).drop_duplicates(subset=["name_key", "position"], keep="first")
    return out


def build_player_investment_context(
        start_draft_season: int = 2014,
        end_draft_season: Optional[int] = None,
) -> pd.DataFrame:
    players_index = load_players_index() or {}
    relevant_rows = []

    for pid, meta in players_index.items():
        pos = str(meta.get("position") or meta.get("pos") or "").upper()
        if pos not in {"QB", "RB", "WR", "TE"}:
            continue

        relevant_rows.append({
            "sleeper_id": str(pid),
            "name": meta.get("name"),
            "name_key": _normalize_name_key(meta.get("name")),
            "team": meta.get("team"),
            "position": pos,
            "age": meta.get("age"),
        })

    players_df = pd.DataFrame(relevant_rows)
    if players_df.empty:
        return pd.DataFrame()

    draft_df = load_draft_history(
        start_year=start_draft_season,
        end_year=end_draft_season,
    )
    contract_df = scrape_otc_contracts()

    if not draft_df.empty:
        draft_keep = [
            c for c in [
                "name_key",
                "draft_year",
                "draft_round",
                "draft_pick",
                "draft_team",
                "draft_capital_score",
            ]
            if c in draft_df.columns
        ]
        players_df = players_df.merge(
            draft_df[draft_keep],
            on="name_key",
            how="left",
        )

    if not contract_df.empty:
        contract_keep = [
            c for c in [
                "name_key",
                "position",
                "contract_total_value",
                "contract_apy",
                "contract_years",
                "guaranteed_money",
                "fully_guaranteed_money",
                "free_agency_year",
            ]
            if c in contract_df.columns
        ]
        players_df = players_df.merge(
            contract_df[contract_keep],
            on=["name_key", "position"],
            how="left",
        )

    players_df = finalize_contract_df(players_df)

    if "draft_capital_score" not in players_df.columns:
        players_df["draft_capital_score"] = 0.0
    players_df["draft_capital_score"] = pd.to_numeric(
        players_df["draft_capital_score"], errors="coerce"
    ).fillna(0.0)

    for col in ["draft_year", "draft_round", "draft_pick"]:
        if col not in players_df.columns:
            players_df[col] = pd.NA

    if "free_agency_year" not in players_df.columns:
        players_df["free_agency_year"] = np.nan
    players_df["free_agency_year"] = pd.to_numeric(
        players_df["free_agency_year"], errors="coerce"
    )

    current_year = date.today().year
    players_df["years_to_fa"] = np.where(
        players_df["free_agency_year"].notna(),
        players_df["free_agency_year"] - current_year,
        np.nan,
    )

    # Positional percentiles for contract signals
    for metric in ["contract_apy", "guaranteed_money", "guaranteed_pct"]:
        if metric not in players_df.columns:
            players_df[metric] = 0.0
        players_df[metric] = pd.to_numeric(players_df[metric], errors="coerce").fillna(0.0)

        rank_col = f"{metric}_pos_pct"
        players_df[rank_col] = (
            players_df.groupby("position")[metric]
            .rank(method="average", pct=True)
            .fillna(0.0)
        )

    # New: positional draft capital percentile
    players_df["draft_capital_pos_pct"] = (
        players_df.groupby("position")["draft_capital_score"]
        .rank(method="average", pct=True)
        .fillna(0.0)
    )

    # Contract score stays contract-specific
    players_df["contract_score"] = (
            0.45 * players_df["contract_apy_pos_pct"] +
            0.35 * players_df["guaranteed_money_pos_pct"] +
            0.20 * players_df["guaranteed_pct_pos_pct"]
    ).clip(0.0, 1.0)

    # New: investment score blends raw + positional draft context
    players_df["team_investment_score"] = (
            0.35 * players_df["contract_score"] +
            0.25 * players_df["draft_capital_score"] +
            0.40 * players_df["draft_capital_pos_pct"]
    ).clip(0.0, 1.0)

    return players_df


def save_player_investment_context(df: pd.DataFrame) -> None:
    if df.empty:
        print("[player_investment] nothing to save")
        return

    df.to_parquet(PLAYER_INVESTMENT_LATEST_PATH, index=False)
    print(f"[player_investment] saved merged -> {PLAYER_INVESTMENT_LATEST_PATH}")

    draft_cols = [c for c in [
        "sleeper_id", "name", "name_key", "position",
        "draft_year", "draft_round", "draft_pick", "draft_team", "draft_capital_score"
    ] if c in df.columns]
    contract_cols = [c for c in [
        "sleeper_id", "name", "name_key", "position",
        "contract_total_value", "contract_apy", "contract_years",
        "guaranteed_money", "fully_guaranteed_money", "free_agency_year",
        "guaranteed_pct", "years_to_fa", "contract_score", "team_investment_score"
    ] if c in df.columns]

    if draft_cols:
        df[draft_cols].to_parquet(DRAFT_HISTORY_PATH, index=False)
        print(f"[player_investment] saved draft -> {DRAFT_HISTORY_PATH}")

    if contract_cols:
        df[contract_cols].to_parquet(CONTRACTS_LATEST_PATH, index=False)
        print(f"[player_investment] saved contracts -> {CONTRACTS_LATEST_PATH}")


def load_player_investment_context() -> pd.DataFrame:
    if PLAYER_INVESTMENT_LATEST_PATH.exists():
        return pd.read_parquet(PLAYER_INVESTMENT_LATEST_PATH)
    return pd.DataFrame()


def rebuild_player_investment_context(
        start_draft_season: int = 2014,
        end_draft_season: Optional[int] = None,
) -> pd.DataFrame:
    df = build_player_investment_context(
        start_draft_season=start_draft_season,
        end_draft_season=end_draft_season,
    )
    save_player_investment_context(df)
    return df


if __name__ == "__main__":
    df = rebuild_player_investment_context()
