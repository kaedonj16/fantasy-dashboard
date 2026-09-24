"""Daily precomputed start/sit score inputs.

``compute_start_score`` (utils.start_sit_score) is a pure function of per-player
inputs that are almost entirely league-independent: weekly projection, recent
and season form, usage trend, Vegas implied total, weather, O-line quality,
expected plays, role confidence, and consistency. Only the *fantasy-point
scoring* of projections and actuals varies by league, and that collapses to one
of the seven canonical projection variants (``ppr | half_ppr | std | tep |
6pt_ppr | 6pt_half | 6pt_tep``) already used by the projection cache
(``data_building.fetch_projections.PROJ_VARIANTS``,
``utils.proj_variant.pick_proj_variant``).

This module builds one input bundle per (season, week, player, variant) in the
daily cron. Request paths (``/api/start-sit-options``) then become a DB lookup
plus a live injury overlay plus the pure scoring function, instead of
re-reading multi-MB stat files, rebuilding the season usage map, and
re-scoring every week per request.

Exactness contract: a league uses the precomputed bundles only when its scoring
settings are *exactly* covered by one canonical variant
(``variant_for_exact_scoring``); anything exotic (fractional PPR, point-per-
first-down, custom yardage rates, ...) falls back to the live per-request
computation, so no league ever gets approximate numbers silently.

Injury status is deliberately NOT trusted from the bundle: Q tags change
intraday, so the request always overlays the live status from the player map
before calling ``compute_start_score``. The bundle stores the batch-time status
for reference only.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical scoring per projection variant.
#
# Chosen so pick_proj_variant(VARIANT_SCORING[v]) == v for every variant (the
# TE-premium bonus 0.5 matches the projection file's TEP_BONUS). A league only
# uses the fast path when its own settings match one of these exactly on the
# variant dimensions (rec / pass_td / bonus_rec_te) and are otherwise standard.
# ---------------------------------------------------------------------------
VARIANT_SCORING: Dict[str, Dict[str, float]] = {
    "ppr":      {"rec": 1.0, "pass_td": 4.0, "bonus_rec_te": 0.0},
    "half_ppr": {"rec": 0.5, "pass_td": 4.0, "bonus_rec_te": 0.0},
    "std":      {"rec": 0.0, "pass_td": 4.0, "bonus_rec_te": 0.0},
    "tep":      {"rec": 1.0, "pass_td": 4.0, "bonus_rec_te": 0.5},
    "6pt_ppr":  {"rec": 1.0, "pass_td": 6.0, "bonus_rec_te": 0.0},
    "6pt_half": {"rec": 0.5, "pass_td": 6.0, "bonus_rec_te": 0.0},
    "6pt_tep":  {"rec": 1.0, "pass_td": 6.0, "bonus_rec_te": 0.5},
}

# Fantasy positions the Start/Sit advisor ranks.
_BUNDLE_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")

_TABLE_READY = False


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def variant_for_exact_scoring(scoring_settings: Optional[dict]) -> Optional[str]:
    """Return the precompute variant covering this league's scoring, or None.

    None means the league's scoring is exotic relative to the seven canonical
    variants (fractional reception points, non-4/6 pass-TD value, a TE premium
    other than 0.5, custom yardage rates, milestone/first-down bonuses, ...).
    Callers must fall back to live per-request scoring for those leagues so
    values stay exact.
    """
    from utils.fantasy_scoring import _DEFAULT_RATES
    from utils.proj_variant import pick_proj_variant

    s = scoring_settings or {}
    # "rec" has inconsistent defaults between the variant picker (1.0) and the
    # stat scorer (0.0), so require it explicitly rather than guessing.
    if "rec" not in s:
        return None
    variant = pick_proj_variant(s)
    canon = VARIANT_SCORING.get(variant)
    if not canon:
        return None
    if _num(s.get("rec"), 1.0) != canon["rec"]:
        return None
    if _num(s.get("pass_td"), 4.0) != canon["pass_td"]:
        return None
    if _num(s.get("bonus_rec_te"), 0.0) != canon["bonus_rec_te"]:
        return None
    for key, standard_rate in _DEFAULT_RATES.items():
        if key in ("rec", "pass_td"):
            # Variant dimensions, already checked exactly against the canonical
            # variant above (rec / pass_td / bonus_rec_te).
            continue
        if key in s and _num(s.get(key), standard_rate) != float(standard_rate):
            return None
    for key, value in s.items():
        if key == "bonus_rec_te":
            continue
        if not (str(key).startswith("bonus_") or key in ("pass_fd", "rush_fd", "rec_fd")):
            continue
        try:
            if float(value or 0) != 0:
                return None
        except (TypeError, ValueError):
            return None
    return variant


# ---------------------------------------------------------------------------
# Postgres table
# ---------------------------------------------------------------------------

def init_start_score_db() -> None:
    """Create the bundle table if needed. Never raises."""
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS start_score_bundle (
                    season    INTEGER NOT NULL,
                    week      INTEGER NOT NULL,
                    player_id TEXT    NOT NULL,
                    variant   TEXT    NOT NULL,
                    bundle    JSONB   NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (season, week, player_id, variant)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ssb_lookup "
                "ON start_score_bundle (season, week, variant)"
            )
        _TABLE_READY = True
    except Exception:
        logger.debug("start_score_bundle init failed", exc_info=True)


def _sleeper_stats_week_num(path: str) -> int:
    m = re.search(r"_w(\d+)", os.path.basename(str(path)))
    try:
        return int(m.group(1)) if m else -1
    except (TypeError, ValueError):
        return -1


def _stat_files(season: int) -> List[str]:
    from utils.paths import CACHE_DIR
    pattern = os.path.join(CACHE_DIR, "sleeper_stats", f"sleeper_stats_s{int(season)}_w*.json")
    return sorted(glob.glob(pattern), key=_sleeper_stats_week_num)


def _variant_week_points(season: int, scoring: dict):
    """Score one season's weekly stat files under canonical variant scoring.

    Returns ``(weekstat, weekly)`` mirroring the two app.py loaders the
    Start/Sit page uses:

    - ``weekstat``: ``{pid: {"sum":, "n":, "last4": [...]}}`` — all files in
      glob order, ``> 0`` filter, ``week_stat_points`` scorer with no TE flag
      (the page calls ``_season_weekstat_points(season, scoring)`` without a
      position, so TE premium never applies here).
    - ``weekly``: ``{pid: [pts...]}`` in completed-week order, ``score_stats``
      scorer *with* the stat line's position (the page's
      ``_load_season_weekly_points``), no ``> 0`` filter — every stat line
      counts as a game.

    Never raises; returns empty maps on failure.
    """
    from utils.fantasy_scoring import score_stats, week_stat_points
    from utils.season_qualification import qualification_policy

    weekstat: Dict[str, dict] = {}
    weekly: Dict[str, list] = {}
    try:
        completed = set(qualification_policy(int(season)).completed_weeks or ())
    except Exception:
        completed = set()
    try:
        files = _stat_files(season)
        last_idx = list(range(max(0, len(files) - 4), len(files)))
        for widx, wf in enumerate(files):
            try:
                with open(wf) as f:
                    wdata = json.load(f)
            except Exception:
                continue
            if not isinstance(wdata, dict):
                continue
            wnum = _sleeper_stats_week_num(wf)
            in_last4 = widx in last_idx
            for pid, st in wdata.items():
                if not isinstance(st, dict):
                    continue
                pid = str(pid)
                pos = str(st.get("position") or st.get("pos") or "")
                try:
                    pts_ws = float(week_stat_points(st, scoring, "") or 0)
                except Exception:
                    pts_ws = 0.0
                if pts_ws > 0:
                    rec = weekstat.setdefault(pid, {"sum": 0.0, "n": 0, "last4": {}})
                    rec["sum"] += pts_ws
                    rec["n"] += 1
                    if in_last4:
                        rec["last4"][widx] = pts_ws
                if wnum in completed:
                    try:
                        pts_w = float(score_stats(st, scoring, pos))
                    except Exception:
                        continue
                    weekly.setdefault(pid, []).append(pts_w)
        # Normalize last4 to a dense 4-list in file order (None = no line).
        for rec in weekstat.values():
            by_idx = rec.pop("last4")
            rec["last4"] = [by_idx.get(wi) for wi in last_idx]
            while len(rec["last4"]) < 4:
                rec["last4"].append(None)
    except Exception:
        logger.debug("start_score_bundle weekly scoring failed", exc_info=True)
    return weekstat, weekly


def _season_ppg_recent(weekstat: dict, pid: str):
    """(season_ppg, recent_ppg) exactly as the Start/Sit page derives them."""
    rec = weekstat.get(str(pid))
    if not rec or not rec.get("n"):
        return 0.0, 0.0
    season_ppg = round(rec["sum"] / rec["n"], 1)
    vals = [v for v in (rec.get("last4") or []) if v is not None]
    recent_ppg = round(sum(vals) / len(vals), 1) if vals else 0.0
    return season_ppg, recent_ppg


def _oline_index_for(season: int, ratings: dict, team: str, pos: str) -> Optional[float]:
    """Position-relevant 0-100 O-line index, mirroring app._oline_for_player."""
    try:
        from data_building.oline_ratings import _norm_team
        team = _norm_team(team)
    except Exception:
        team = (team or "").upper().strip()
    if not team:
        return None
    row = (ratings or {}).get(team)
    if not row:
        return None
    pos = (pos or "").upper().strip()
    key = ("run_block" if pos == "RB"
           else "pass_block" if pos in ("QB", "WR", "TE")
           else "composite")
    try:
        return float(row.get(key)) if row.get(key) is not None else None
    except (TypeError, ValueError):
        return None


def _load_oline_ratings(season: int) -> dict:
    try:
        path = os.path.join("cache", f"oline_ratings_s{int(season)}.json")
        if os.path.exists(path):
            with open(path) as f:
                return (json.load(f) or {}).get("ratings") or {}
    except Exception:
        pass
    # Fall back to the newest built season (same policy as the page).
    try:
        newest = None
        for fn in os.listdir("cache"):
            if fn.startswith("oline_ratings_s") and fn.endswith(".json"):
                try:
                    yr = int(fn[len("oline_ratings_s"):-len(".json")])
                except ValueError:
                    continue
                newest = yr if newest is None else max(newest, yr)
        if newest is not None:
            with open(os.path.join("cache", f"oline_ratings_s{newest}.json")) as f:
                return (json.load(f) or {}).get("ratings") or {}
    except Exception:
        pass
    return {}

# ---------------------------------------------------------------------------
# Batch build
# ---------------------------------------------------------------------------

def build_start_score_bundles(season: int = None, week: int = None) -> dict:
    """Build one input bundle per (player, variant) for the current week.

    Gathers every ``compute_start_score`` input the Start/Sit page uses, once,
    and upserts it into ``start_score_bundle``. Runs in the daily cron after
    the weekly projections step. Prunes bundles for older (season, week) so the
    table only ever holds the current week. Never raises; returns a summary.
    """
    from dashboard_services.api import get_nfl_state
    from dashboard_services.db import get_conn
    from dashboard_services.team_play_volume import load_team_play_volume
    from data_building.fetch_projections import PROJ_VARIANTS, fetch_sleeper_season_ppg_variants
    from data_building.weekly_metrics import get_usage_trends
    from utils.consistency import BLEND_FULL_SEASON, blended_consistency_profile
    from utils.fantasy_scoring import weekly_projection_points
    from utils.game_conditions import build_week_conditions
    from utils.league_scoring import stamp_scoring_aliases
    from utils.start_sit_context import expected_plays_context, role_confidence_from_trend
    from utils.utils import load_players_index, load_week_projection, load_week_sched

    t0 = time.time()
    summary: dict = {"ok": False, "season": season, "week": week}
    try:
        state = get_nfl_state() or {}
        season = int(season or state.get("season") or 0)
        week = int(week or state.get("week") or 0)
        if not season or not week:
            summary["error"] = "no season/week"
            return summary
        summary["season"], summary["week"] = season, week

        players_index = load_players_index() or {}

        # ── Schedule: opponent map + (home, away, gameDate) for conditions ──
        opponent_map: Dict[str, str] = {}
        week_games: list = []
        try:
            for game in (load_week_sched(season, week) or []):
                home = str(game.get("home") or "").upper()
                away = str(game.get("away") or "").upper()
                if home and away:
                    opponent_map[home] = away
                    opponent_map[away] = home
                    week_games.append((home, away, str(game.get("gameDate") or "")))
        except Exception:
            logger.debug("start_score_bundle schedule failed", exc_info=True)

        # ── League-independent inputs, loaded once ──────────────────────────
        try:
            usage_trends = get_usage_trends(season) or {}
        except Exception:
            usage_trends = {}
        try:
            game_conditions = build_week_conditions(season, week, week_games) if week_games else {}
        except Exception:
            game_conditions = {}
        try:
            _tpv_blob = load_team_play_volume(season) or {}
            team_play_volume = _tpv_blob.get("teams") or {}
            tpv_nfl_avg = _tpv_blob.get("nfl_avg_plays_faced_pg")
        except Exception:
            team_play_volume, tpv_nfl_avg = {}, None
        oline_ratings = _load_oline_ratings(season)
        try:
            week_proj_map = load_week_projection(season, week) or {}
        except Exception:
            week_proj_map = {}
        try:
            season_ppg_variants = fetch_sleeper_season_ppg_variants(season, players_index) or {}
        except Exception:
            season_ppg_variants = {}

        # ── Player universe: index fantasy positions + projection entries ───
        universe: Dict[str, dict] = {}
        for pid, meta in players_index.items():
            pos = str((meta or {}).get("pos") or (meta or {}).get("position") or "").upper()
            if pos in _BUNDLE_POSITIONS:
                universe[str(pid)] = {
                    "pos": pos,
                    "team": str((meta or {}).get("team") or "").upper(),
                    "injury_status": str((meta or {}).get("injury_status")
                                         or (meta or {}).get("status") or ""),
                }
        for pid, entry in (week_proj_map or {}).items():
            pid = str(pid)
            if pid not in universe and isinstance(entry, dict):
                meta = players_index.get(pid) or {}
                pos = str(meta.get("pos") or meta.get("position") or "").upper()
                if pos in _BUNDLE_POSITIONS:
                    universe[pid] = {
                        "pos": pos,
                        "team": str(meta.get("team") or "").upper(),
                        "injury_status": str(meta.get("injury_status")
                                             or meta.get("status") or ""),
                    }

        # ── Per-variant scoring-dependent inputs ────────────────────────────
        variants = [v for v in PROJ_VARIANTS if v in VARIANT_SCORING]
        per_variant: dict = {}
        for variant in variants:
            scoring = stamp_scoring_aliases(VARIANT_SCORING[variant])
            weekstat, weekly = _variant_week_points(season, scoring)
            # Prior-season crossfade for consistency (same BLEND_FULL_SEASON
            # gate as the page: only while the current sample is short).
            prior_weekly: Dict[str, list] = {}
            try:
                if weekly:
                    for py in (season - 1, season - 2):
                        _, pweekly = _variant_week_points(py, scoring)
                        if pweekly and max((len(v) for v in pweekly.values()), default=0) >= 3:
                            prior_weekly = pweekly
                            break
            except Exception:
                prior_weekly = {}
            per_variant[variant] = {
                "scoring": scoring, "weekstat": weekstat, "weekly": weekly,
                "prior_weekly": prior_weekly,
            }

        # ── Assemble bundles ────────────────────────────────────────────────
        rows: list = []
        for pid, u in universe.items():
            pos = u["pos"]
            team = u["team"]
            opponent = opponent_map.get(team, "")
            on_bye = bool(opponent_map) and team not in opponent_map
            ut = usage_trends.get(pid) or {}
            cond = (game_conditions.get(team) or {}) if not on_bye else {}
            implied_total = cond.get("implied_total")
            wx = cond.get("weather") or {}
            weather_kind = wx.get("kind") if isinstance(wx, dict) else None
            oline_index = _oline_index_for(season, oline_ratings, team, pos)
            pace = expected_plays_context(team_play_volume, team, opponent, tpv_nfl_avg) or {}
            role_conf = role_confidence_from_trend(ut)
            week_entry = week_proj_map.get(pid)
            if not isinstance(week_entry, dict):
                week_entry = week_proj_map.get(str(pid))
            for variant in variants:
                pv = per_variant[variant]
                scoring = pv["scoring"]
                # Projection: weekly file variant value first (same
                # weekly_projection_points selection the page uses), season
                # median PPG fallback.
                proj_pts = 0.0
                try:
                    if isinstance(week_entry, dict):
                        proj_pts = float(weekly_projection_points(
                            week_entry, scoring, pos) or 0.0)
                except Exception:
                    proj_pts = 0.0
                if not proj_pts:
                    try:
                        proj_pts = float((season_ppg_variants.get(pid) or {}).get(variant) or 0.0)
                    except (TypeError, ValueError):
                        proj_pts = 0.0
                season_ppg, recent_ppg = _season_ppg_recent(pv["weekstat"], pid)
                bust_rate = None
                try:
                    cur = pv["weekly"].get(pid) or []
                    prior = pv["prior_weekly"].get(pid) or []
                    cons = None
                    if cur or prior:
                        cons = blended_consistency_profile(cur, prior, pos)
                    if cons and not cons.get("small_sample"):
                        bust_rate = cons.get("bust_rate")
                except Exception:
                    bust_rate = None
                bundle = {
                    "pos": pos,
                    "team": team,
                    "opponent": opponent or None,
                    "on_bye": bool(on_bye),
                    "proj_pts": round(proj_pts, 1),
                    "recent_ppg": recent_ppg,
                    "season_ppg": season_ppg,
                    "bust_rate": bust_rate,
                    "usage_delta": ut.get("delta"),
                    "usage_season_avg": ut.get("season_avg"),
                    "implied_total": implied_total,
                    "weather_kind": weather_kind,
                    "oline_index": oline_index,
                    "expected_team_plays": pace.get("expected_team_plays"),
                    "league_average_plays": pace.get("league_average_plays"),
                    "role_confidence": role_conf,
                    # Batch-time status for reference; requests always overlay
                    # the live status before scoring.
                    "injury_status": u.get("injury_status") or None,
                }
                rows.append((season, week, pid, variant, json.dumps(bundle)))

        init_start_score_db()
        with get_conn() as conn:
            # Prune older weeks first so the table only holds the current one.
            conn.execute(
                "DELETE FROM start_score_bundle WHERE NOT (season = %s AND week = %s)",
                (season, week),
            )
            for i in range(0, len(rows), 500):
                chunk = rows[i:i + 500]
                conn.executemany(
                    """
                    INSERT INTO start_score_bundle
                        (season, week, player_id, variant, bundle, updated_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, now())
                    ON CONFLICT (season, week, player_id, variant)
                    DO UPDATE SET bundle = EXCLUDED.bundle, updated_at = now()
                    """,
                    chunk,
                )
        summary.update({
            "ok": True,
            "players": len(universe),
            "variants": len(variants),
            "rows": len(rows),
            "seconds": round(time.time() - t0, 1),
        })
        logger.info("start_score_bundle built: %s", summary)
        return summary
    except Exception as exc:
        logger.warning("build_start_score_bundles failed", exc_info=True)
        summary["error"] = str(exc)[:200]
        return summary


# ---------------------------------------------------------------------------
# Request-time read path
# ---------------------------------------------------------------------------

def load_start_score_bundles(season: int, week: int, variant: str,
                             player_ids: list) -> Dict[str, dict]:
    """{player_id: bundle} for one (season, week, variant). Never raises.

    Returns {} when the table is missing, the batch hasn't run yet, or the DB
    is unreachable — callers fall back to live per-request computation.
    """
    if not player_ids or not variant:
        return {}
    try:
        from dashboard_services.db import get_conn
        pids = [str(p) for p in player_ids]
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT player_id, bundle FROM start_score_bundle
                WHERE season = %s AND week = %s AND variant = %s
                  AND player_id = ANY(%s)
                """,
                (int(season), int(week), str(variant), pids),
            ).fetchall()
        out: Dict[str, dict] = {}
        for r in rows or []:
            pid = str(r["player_id"])
            bundle = r["bundle"]
            if isinstance(bundle, str):
                try:
                    bundle = json.loads(bundle)
                except Exception:
                    continue
            if isinstance(bundle, dict):
                out[pid] = bundle
        return out
    except Exception:
        logger.debug("load_start_score_bundles failed", exc_info=True)
        return {}


def start_score_bundles_ready(season: int, week: int, variant: str,
                              player_ids: list) -> bool:
    """True when every requested player has a bundle for (season, week, variant)."""
    if not player_ids:
        return False
    try:
        from dashboard_services.db import get_conn
        pids = [str(p) for p in player_ids]
        with get_conn() as conn:
            n = conn.execute(
                """
                SELECT COUNT(*) AS n FROM start_score_bundle
                WHERE season = %s AND week = %s AND variant = %s
                  AND player_id = ANY(%s)
                """,
                (int(season), int(week), str(variant), pids),
            ).fetchone()
        count = int((n or {}).get("n") or 0)
        return count >= len(set(pids))
    except Exception:
        logger.debug("start_score_bundles_ready check failed", exc_info=True)
        return False
