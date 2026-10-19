"""
Reset phantom dynasty values: players the board prices far above BOTH
external markets (or prices highly with no market coverage at all).

The Phil Mafah case (Oct 2026): the board served 329.6 (RB20) for a player
FantasyCalc priced at ~9/999 normalized and DynastyProcess at 3 raw, with
0.0 PPG. The inflated state is self-sustaining once it exists — the WLS
calibrated track, the EMA history and the ±10%/day clamp baseline all read
each other's output — so a code fix alone unwinds it at 10%/day for weeks.
This script breaks the state in one pass for the whole cohort:

  * player_values: value_1qb / value_sf (and size variants) are set to the
    recomputed external vendor blend (the same FC 0.40 / DP 0.20 blend the
    board builder uses, 0 when no market covers the player), and every
    calibrated column is NULLed so the served COALESCE falls back to the
    corrected raw value and the next WLS run clamps against nothing.
  * player_value_history: the player's rows are DELETED, so the EMA carry,
    the board clamp baseline and the modal chart all restart from the
    corrected value on the next daily run instead of blending back toward
    the phantom.

Cohort rule (per track, 1QB and SF separately): the SERVED value
(COALESCE(calibrated, raw)) is >= 100 AND either no external market covers
the player, or it sits >= 2.5x the normalized market consensus with a gap
>= 100 — the board's own over-market guardrail criteria, applied to the
served track the guardrail never sees. A player is reset if EITHER track
qualifies; both tracks are corrected.

Dry-run by default; pass --apply to write.

Usage (Render shell, repo root):
    python data_building/reset_phantom_values.py            # report only
    python data_building/reset_phantom_values.py --apply    # write the reset
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_building.value_guardrails import (
    OVERMARKET_MIN_GAP,
    OVERMARKET_TRIGGER,
    sf_nonqb_floor,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SCALE = 999.9
# Board-builder blend weights (value_model_training): 1QB FC .40 / DP .20,
# SF FC-SF .50 / DP-2QB .30, each renormalized over the sources present.
_W_1QB = (("fc", 0.40), ("dp", 0.20))
_W_SF = (("fc", 0.50), ("dp", 0.30))


def minmax_norm(raw: float, lo: float, hi: float) -> float:
    """Normalize a raw external value to the board's 0-999.9 scale."""
    if raw is None or float(raw) <= 0 or hi <= lo:
        return 0.0
    return (float(raw) - lo) / (hi - lo) * SCALE


def _blend(parts: dict, weights) -> float:
    num = 0.0
    den = 0.0
    for key, w in weights:
        v = parts.get(key) or 0.0
        if v > 0:
            num += w * v
            den += w
    return num / den if den > 0 else 0.0


def corrected_values(fc_norm: float, dp_norm: float,
                     fc_sf_norm: float, dp2_norm: float,
                     position: str) -> tuple[float, float]:
    """The value the external markets actually support, board scale.

    The plain vendor blend with no WLS input and no smoothing state: this is
    what the board builder would produce for the player from a clean slate.
    """
    v1 = _blend({"fc": fc_norm, "dp": dp_norm}, _W_1QB)
    vsf = _blend({"fc": fc_sf_norm, "dp": dp2_norm}, _W_SF)
    if str(position or "").upper() != "QB":
        vsf = sf_nonqb_floor(vsf, v1)
    return round(v1, 1), round(vsf, 1)


def is_phantom(served: float, consensus_norm: float, *,
               trigger: float = OVERMARKET_TRIGGER,
               min_gap: float = OVERMARKET_MIN_GAP) -> bool:
    """True when a served value has no honest market behind it.

    Mirrors the board guardrail (overmarket_capped) plus its former
    no-coverage exemption, closed: with zero market coverage, any served
    value at/above the min-gap stake is a phantom.
    """
    if served is None or float(served) < min_gap:
        return False
    ref = float(consensus_norm or 0.0)
    if ref <= 0:
        return True
    return float(served) >= trigger * max(ref, 1.0) and (float(served) - ref) >= min_gap


def _bounds(values) -> tuple[float, float]:
    vals = [float(v) for v in values if v is not None and float(v) > 0]
    return (min(vals), max(vals)) if vals else (0.0, 1.0)


def find_cohort() -> list[dict]:
    """Players whose served value is a phantom on either track."""
    from dashboard_services.db import get_conn
    from data_building.external_data.external_values_scraper import (
        load_dynastyprocess_values,
        load_fantasycalc_api_values,
        load_fantasycalc_sf_api_values,
    )
    from utils.utils import load_players_index, normalize_name

    fc_rows = load_fantasycalc_api_values() or []
    fc_sf_rows = load_fantasycalc_sf_api_values() or []
    dp_rows = load_dynastyprocess_values() or []

    fc_lo, fc_hi = _bounds(r.get("value") for r in fc_rows)
    fcs_lo, fcs_hi = _bounds(r.get("value") for r in fc_sf_rows)
    dp_lo, dp_hi = _bounds(r.get("value_1qb") for r in dp_rows)
    dp2_lo, dp2_hi = _bounds(r.get("value_2qb") for r in dp_rows)

    fc_by_sid = {str(r.get("sleeper_id") or ""): r for r in fc_rows}
    fcs_by_sid = {str(r.get("sleeper_id") or ""): r for r in fc_sf_rows}
    # DP has no sleeper id; match by normalized name, dropping ambiguous names.
    dp_by_name: dict[str, dict] = {}
    dp_ambiguous: set[str] = set()
    for r in dp_rows:
        nm = normalize_name(str(r.get("player") or ""))
        if not nm:
            continue
        if nm in dp_by_name:
            dp_ambiguous.add(nm)
        dp_by_name[nm] = r
    for nm in dp_ambiguous:
        dp_by_name.pop(nm, None)

    index = load_players_index() or {}

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT player_id, position, team,
                   value_1qb, value_sf,
                   calibrated_value_1qb, calibrated_value_sf
            FROM player_values
            WHERE position IS NULL OR position != 'PICK'
            """
        ).fetchall()

    cohort = []
    for r in rows:
        pid = str(r["player_id"])
        meta = index.get(pid) or {}
        name = meta.get("name") or pid
        dp = dp_by_name.get(normalize_name(name)) or {}

        fc_n = minmax_norm(float((fc_by_sid.get(pid) or {}).get("value") or 0), fc_lo, fc_hi)
        dp_n = minmax_norm(float(dp.get("value_1qb") or 0), dp_lo, dp_hi)
        fcs_n = minmax_norm(float((fcs_by_sid.get(pid) or {}).get("value") or 0), fcs_lo, fcs_hi)
        dp2_n = minmax_norm(float(dp.get("value_2qb") or 0), dp2_lo, dp2_hi)

        raw1 = float(r["value_1qb"] or 0)
        served1 = float(r["calibrated_value_1qb"] or raw1)
        served_sf = float(r["calibrated_value_sf"] or r["value_sf"] or raw1)

        if not (is_phantom(served1, max(fc_n, dp_n))
                or is_phantom(served_sf, fcs_n)):
            continue
        corr1, corr_sf = corrected_values(fc_n, dp_n, fcs_n, dp2_n, r["position"])
        cohort.append({
            "player_id": pid, "name": name, "position": r["position"],
            "team": r["team"] or meta.get("team"),
            "served_1qb": round(served1, 1), "served_sf": round(served_sf, 1),
            "fc_norm": round(fc_n, 1), "dp_norm": round(dp_n, 1),
            "corrected_1qb": corr1, "corrected_sf": corr_sf,
        })
    return sorted(cohort, key=lambda c: -c["served_1qb"])


def apply_reset(cohort: list[dict]) -> tuple[int, int]:
    """Write corrected values + null calibrations; delete history rows."""
    from dashboard_services.db import get_conn

    updated = 0
    deleted = 0
    with get_conn() as conn, conn.cursor() as cur:
        for c in cohort:
            cur.execute(
                """
                UPDATE player_values SET
                    value_1qb = %(v1)s, value_sf = %(vsf)s,
                    value_8 = %(v1)s, value_12 = %(v1)s, value_14 = %(v1)s,
                    sf_value_8 = %(vsf)s, sf_value_12 = %(vsf)s,
                    sf_value_14 = %(vsf)s,
                    calibrated_value_1qb = NULL, calibrated_value_sf = NULL,
                    calibrated_value_8 = NULL, calibrated_value_12 = NULL,
                    calibrated_value_14 = NULL,
                    calibrated_sf_value_8 = NULL,
                    calibrated_sf_value_12 = NULL,
                    calibrated_sf_value_14 = NULL,
                    calibration_source = NULL, calibration_weight = NULL,
                    calibration_backing = 0, calibration_backing_sf = 0
                WHERE player_id = %(pid)s
                """,
                {"v1": c["corrected_1qb"], "vsf": c["corrected_sf"],
                 "pid": c["player_id"]},
            )
            updated += cur.rowcount or 0
            cur.execute(
                "DELETE FROM player_value_history WHERE player_id = %s",
                (c["player_id"],),
            )
            deleted += cur.rowcount or 0
    return updated, deleted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the reset (default is a dry-run report)")
    args = ap.parse_args()

    cohort = find_cohort()
    if not cohort:
        logger.info("No phantom values found — board looks clean.")
        return 0
    logger.info("Found %d phantom-value players:", len(cohort))
    for c in cohort:
        logger.info(
            "  %-26s %-2s %-4s served %7.1f / SF %7.1f  (FC %5.1f, DP %5.1f)"
            "  -> corrected %.1f / SF %.1f",
            c["name"], c["position"], c["team"], c["served_1qb"],
            c["served_sf"], c["fc_norm"], c["dp_norm"],
            c["corrected_1qb"], c["corrected_sf"],
        )
    if not args.apply:
        logger.info("Dry run only — re-run with --apply to reset these players.")
        return 0
    updated, deleted = apply_reset(cohort)
    logger.info("Reset %d player_values rows; deleted %d history rows.",
                updated, deleted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
