"""
Update player_values table with current rankings and save to player_value_history.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import List, Dict, Any
import pandas as pd

from utils.utils import load_model_value_table
from data_building.save_player_values import save_daily_values_to_db


def _load_fc_redraft_values() -> dict[str, tuple[float | None, float | None]]:
    """
    Build a sleeper_id -> (redraft_value_1qb, redraft_value_sf) map from the
    latest FantasyCalc CSV.  FC only exposes one redraft value per API call
    (1QB by default), so redraft_value_sf is populated from a separate scrape
    when available; otherwise we reuse the 1QB value as a placeholder.
    Returns raw FC values (0-10000 scale); normalization is applied separately.
    """
    try:
        from data_building.external_data.external_values_scraper import load_fantasycalc_api_values
        rows = load_fantasycalc_api_values() or []
        result: dict[str, tuple[float | None, float | None]] = {}
        for row in rows:
            sid = str(row.get("sleeper_id") or "").strip()
            if not sid:
                continue
            try:
                v1 = float(row["redraft_value"]) if row.get("redraft_value") not in (None, "", "None") else None
            except (TypeError, ValueError):
                v1 = None
            result[sid] = (v1, v1)  # sf placeholder = 1qb until SF scrape added
        return result
    except Exception as e:
        print(f"[update_player_values] Could not load FC redraft values: {e}")
        return {}


def _normalize_redraft_values(
    fc_redraft: dict[str, tuple[float | None, float | None]],
    df: "pd.DataFrame",
) -> dict[str, tuple[float | None, float | None]]:
    """
    Normalize raw FantasyCalc redraft values (0-10000 scale) to the site's
    0-999.9 scale using the same top-5 anchor logic as dynasty values.

    The anchor players are chosen by the site's dynasty value rank (top-5
    non-QB skill positions), not by raw FC redraft rank. Using FC redraft rank
    would inflate the anchor whenever rookies or other players have artificially
    high FC redraft values, pushing the established elite below 999.9.

    Only the 1QB value is scaled here; the SF value is recomputed per-player
    from the normalized 1QB value using the dynasty SF/1QB ratio.
    """
    _ANCHOR_N = 5
    _NON_QB = {"RB", "WR", "TE"}

    # Rank players by dynasty value and pick the top-N non-QB as the anchor.
    # These are the known-elite players whose redraft values should sit near
    # 999.9, regardless of what FC assigns to unproven prospects.
    anchor_ids = (
        df[df["position"].str.upper().isin(_NON_QB)]
        .nlargest(_ANCHOR_N, "value")["id"]
        .astype(str)
        .tolist()
    )

    anchor_vals = [
        fc_redraft[pid][0]
        for pid in anchor_ids
        if pid in fc_redraft and fc_redraft[pid][0] is not None
    ]

    if not anchor_vals:
        print("[update_player_values] Redraft anchor: no anchor players found, skipping normalization")
        return fc_redraft

    avg = sum(anchor_vals) / len(anchor_vals)
    scale = 999.9 / avg
    print(
        f"[update_player_values] Redraft anchor: {len(anchor_vals)} dynasty-top players, "
        f"avg FC redraft={avg:.1f}, scale={scale:.5f}"
    )

    return {
        pid: (round(v1 * scale, 2) if v1 is not None else None, sf_raw)
        for pid, (v1, sf_raw) in fc_redraft.items()
    }


def _load_historical_ranks(target_date: date) -> Dict[str, Dict[str, int]]:
    """
    Load per-player overall_rank and pos_rank from the closest snapshot on or
    before target_date using player_value_history.

    Returns dict keyed by player_id: {'overall_rank': int, 'pos_rank': int}
    """
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url or any(t in db_url for t in ("USER", "PASSWORD", "HOST")):
        return {}
    try:
        from dashboard_services.db import get_conn
    except Exception:
        return {}

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Grab the single snapshot date closest to (but not after) target_date
                cur.execute(
                    """
                    SELECT DISTINCT as_of_date
                    FROM player_value_history
                    WHERE as_of_date <= %s AND source = 'model'
                    ORDER BY as_of_date DESC
                    LIMIT 1
                    """,
                    (target_date,),
                )
                row = cur.fetchone()
                if not row:
                    return {}
                snap_date = row["as_of_date"] if isinstance(row, dict) else row[0]

                cur.execute(
                    """
                    SELECT player_id, position, value
                    FROM player_value_history
                    WHERE as_of_date = %s AND source = 'model'
                      AND position IN ('QB', 'RB', 'WR', 'TE')
                    """,
                    (snap_date,),
                )
                rows = cur.fetchall()

        if not rows:
            return {}

        if isinstance(rows[0], dict):
            hist = pd.DataFrame(rows)
        else:
            hist = pd.DataFrame(rows, columns=["player_id", "position", "value"])

        hist["value"] = pd.to_numeric(hist["value"], errors="coerce").fillna(0)
        hist["overall_rank"] = hist["value"].rank(ascending=False, method="min").astype(int)
        hist["pos_rank"] = (
            hist.groupby("position")["value"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        return {
            str(r["player_id"]): {
                "overall_rank": int(r["overall_rank"]),
                "pos_rank": int(r["pos_rank"]),
            }
            for _, r in hist.iterrows()
        }
    except Exception as e:
        print(f"[update_player_values] Could not load historical ranks: {e}")
        return {}


def update_player_values_with_rankings() -> int:
    """
    Update player_values table with current rankings and save to player_value_history.

    Returns:
        Number of players updated
    """
    # Load raw model values only - must NOT use calibrated values here or
    # they would be written back into player_values.value_1qb, corrupting
    # the model prior that calibration depends on.
    value_table = load_model_value_table(apply_calibration=False)
    if not value_table:
        print("[update_player_values] No value table available")
        return 0

    df = pd.DataFrame(value_table)

    # Add rankings to each player
    df['overall_rank'] = df['value'].rank(ascending=False, method='min')

    # Player-only rank (QB/RB/WR/TE) used for rank_change_7d to match display pool.
    # Picks and other asset types are excluded so movement arrows reflect actual
    # player-vs-player movement, not pool composition changes.
    _player_mask = df['position'].isin({'QB', 'RB', 'WR', 'TE'})
    df['player_rank'] = df['value'].where(_player_mask).rank(ascending=False, method='min')
    # Load calibration overrides to get calibrated values for ranking
    try:
        from dashboard_services.player_value_history import load_calibration_overrides
        calibration_overrides = load_calibration_overrides()

        df['calibrated_value'] = df['id'].apply(
            lambda x: calibration_overrides.get(str(x), {}).get('value',
                df.loc[df['id'] == x, 'value'].iloc[0])
        )
        df['calibrated_sf_value'] = df['id'].apply(
            lambda x: calibration_overrides.get(str(x), {}).get('sf_value',
                df.loc[df['id'] == x, 'sf_value'].iloc[0])
        )

        df['pos_rank']    = df.groupby('position')['calibrated_value'].rank(ascending=False, method='min')
        df['sf_pos_rank'] = df.groupby('position')['calibrated_sf_value'].rank(ascending=False, method='min')
        print("[update_player_values] Position ranks calculated based on calibrated values (1QB + SF)")
    except Exception as e:
        print(f"[update_player_values] Failed to load calibrated values for ranking: {e}")
        df['pos_rank']    = df.groupby('position')['value'].rank(ascending=False, method='min')
        df['sf_pos_rank'] = df.groupby('position')['sf_value'].rank(ascending=False, method='min')
        print("[update_player_values] Position ranks calculated based on raw values (fallback)")

    # Apply smoothing to reduce steep drop-offs
    df_smoothed = apply_smoothing(df)

    # Load historical ranks from 7 days ago for movement indicators
    hist_ranks = _load_historical_ranks(date.today() - timedelta(days=7))

    # Load FC redraft values and normalize to the site's 0-999.9 scale.
    # Anchor players are selected by dynasty rank so established elites
    # sit near 999.9 regardless of inflated FC values for unproven rookies.
    fc_redraft = _load_fc_redraft_values()
    fc_redraft = _normalize_redraft_values(fc_redraft, df)

    # Convert back to list of dicts
    updated_players = []
    for _, row in df_smoothed.iterrows():
        pid      = str(row['id'])
        position = row['position']
        cur_overall    = int(row['overall_rank'])
        cur_pos        = int(row['pos_rank'])
        cur_sf_pos     = int(row.get('sf_pos_rank', cur_pos))
        _pr = row.get('player_rank')
        cur_player_rank = int(_pr) if (_pr is not None and not pd.isna(_pr)) else None

        hist = hist_ranks.get(pid)
        rank_change_7d     = (hist['overall_rank'] - cur_player_rank) if (hist and cur_player_rank is not None) else None
        pos_rank_change_7d = (hist['pos_rank'] - cur_pos) if hist else None

        rd_1qb, rd_sf_raw = fc_redraft.get(pid, (None, None))

        # Compute a real SF redraft value using the dynasty SF/1QB ratio as a position-aware
        # scaler.  QBs benefit most (2-3x boost); skill positions are nearly unchanged.
        if rd_1qb is not None:
            raw_val = float(row.get('value') or 1)
            sf_val  = float(row.get('sf_value') or raw_val)
            ratio   = sf_val / max(raw_val, 1.0)
            # Cap the ratio: QBs top out ~2.2x; skill positions stay near 1.0
            capped_ratio = min(ratio, 2.2) if position == 'QB' else min(ratio, 1.15)
            rd_sf = round(rd_1qb * capped_ratio, 2)
        else:
            rd_sf = rd_sf_raw

        updated_players.append({
            'id': pid,
            'name': row['name'],
            'position': position,
            'team': row['team'],
            'age': row['age'],
            'value': round(row['value'], 2),
            'sf_value': round(row['sf_value'], 2),
            'redraft_value_1qb': rd_1qb,
            'redraft_value_sf':  rd_sf,
            'overall_rank': cur_overall,
            'pos_rank': cur_pos,
            'pos_rank_label': f"{position}{cur_pos}",
            'sf_pos_rank': cur_sf_pos,
            'sf_pos_rank_label': f"{position}{cur_sf_pos}",
            'search_name': row.get('search_name', ''),
            'rank_change_7d': rank_change_7d,
            'pos_rank_change_7d': pos_rank_change_7d,
        })
    
    # Picks come from model_values.json exclusively; don't persist them in the DB
    # to avoid stale/duplicate entries from old name formats accumulating.
    players_only = [p for p in updated_players if p.get("position") != "PICK"]

    # Save to player_values table
    saved_count = save_daily_values_to_db(players_only)

    # Save to player_value_history table
    history_count = save_to_player_value_history(players_only)

    print(f"[update_player_values] Updated {saved_count} players with rankings")
    print(f"[update_player_values] Saved {history_count} entries to player_value_history")

    return saved_count


def apply_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply smoothing to reduce steep drop-offs between elite players.
    Creates specific spread pattern: 999, 982, 969, 958, 937 for top 5.
    Increases TE compression to lower TE values.
    """
    df_smoothed = df.copy()
    
    # Sort by value to get ranking
    df_sorted = df.sort_values('value', ascending=False)
    
    if len(df_sorted) < 2:
        return df_smoothed
    
    # Let the value formula naturally determine the spread
    # No forced values - allow the underlying model to create the distribution
    
    # TE compression is now handled in the core value formula (_apply_te_market_compression)
    # No additional compression needed here
    
    def _smooth_col(col: str) -> None:
        if col not in df_smoothed.columns:
            return
        non_te = df_smoothed[df_smoothed['position'] != 'TE']
        elite = non_te[non_te[col] >= 900.0].sort_values(col, ascending=False)
        remaining = elite.iloc[5:] if len(elite) > 5 else pd.DataFrame()
        if len(remaining) < 2:
            return
        vals = remaining[col].tolist()
        for i in range(len(vals) - 1):
            cur, nxt = vals[i], vals[i + 1]
            if cur - nxt > 80:
                df_smoothed.loc[df_smoothed[col] == cur, col] = nxt + (cur - nxt) * 0.7

    _smooth_col('value')
    _smooth_col('sf_value')

    return df_smoothed


def save_to_player_value_history(players: List[Dict[str, Any]]) -> int:
    """
    Save player values to player_value_history table.
    """
    # Only import if DATABASE_URL is set
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url or any(token in db_url for token in ("USER", "PASSWORD", "HOST")):
        print("[save_to_player_value_history] DATABASE_URL not configured, skipping save")
        return 0
    
    try:
        from dashboard_services.db import get_conn
    except Exception as e:
        print(f"[save_to_player_value_history] Database not available: {e}")
        return 0
    
    snapshot_date = date.today()
    saved_count = 0

    # Values are already model-normalized (top-N basket mean = 999.9; elite players
    # float above it). Re-deriving a per-day scale here would re-anchor to a different
    # population and, worse for a *history* table, silently rescale every player
    # whenever the daily leaguewide max moves — manufacturing day-over-day "changes"
    # out of a flat value. Preserve the model's numbers; clamp defensively at a high
    # ceiling only (NOT 999.9, which would re-flatten the top players' sparklines).
    def _clamp(v):
        return round(min(max(float(v or 0), 0.0), 9999.0), 2)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for player in players:
                    player_id = player.get("id")
                    if not player_id:
                        continue

                    _val    = _clamp(player.get("value", 0))
                    _sf_val = _clamp(player.get("sf_value", 0) or player.get("value", 0))

                    # Insert into player_value_history
                    cur.execute(
                        """
                        INSERT INTO player_value_history (
                            as_of_date,
                            player_id,
                            name,
                            position,
                            team,
                            value,
                            sf_value,
                            source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (as_of_date, player_id, source)
                        DO UPDATE SET
                            name     = EXCLUDED.name,
                            position = EXCLUDED.position,
                            team     = EXCLUDED.team,
                            value    = EXCLUDED.value,
                            sf_value = COALESCE(player_value_history.sf_value, EXCLUDED.sf_value)
                        """,
                        (
                            snapshot_date,
                            str(player_id),
                            player.get("name", ""),
                            player.get("position", ""),
                            player.get("team", ""),
                            _val,
                            _sf_val,
                            "model"
                        ),
                    )
                    saved_count += 1
            
            print(f"[save_to_player_value_history] Successfully saved {saved_count} player values to history for {snapshot_date}")
            conn.commit()
            
    except Exception as e:
        print(f"[save_to_player_value_history] Error saving to database: {e}")
        import traceback
        traceback.print_exc()
        return 0
    
    return saved_count


if __name__ == "__main__":
    print("Updating player values with rankings...")
    count = update_player_values_with_rankings()
    print(f"Updated {count} players successfully")
