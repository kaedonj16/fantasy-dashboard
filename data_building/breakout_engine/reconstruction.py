"""As-of reconstruction of weekly breakout boards under the current scorer.

Weeks originally scored under an older SCORING_VERSION can be re-scored
under the current version with the inputs as they stood on the original
run's date, instead of today's live feeds (which would leak everything
that happened since). The reconstruction is published as an ordinary
weekly run whose detail is flagged ``reconstructed``; the store's serving
rules keep the ORIGINAL snapshot serving the week selector, the live
track record excludes reconstructed calls, and the sidebar reports them
as a separate, labeled backtest line. The existing grader grades the
reconstructed calls like any other stored call.

As-of inputs come from nflverse (the same source family the metrics
pipeline already uses):

- ``roster_weekly`` for week W: team, position, years_exp, rookie_year,
  entry_year (draft-year proxy) and birth_date (age as of the run date).
  A roster status of RES in week W or W+1 maps to Sleeper's "IR".
- ``injuries`` reports: the week W report designation (Out / Doubtful /
  Questionable), falling back to the week W+1 designation, which is where
  an injury sustained during week W's games first shows up. That mirrors
  what the live Sleeper feed showed when the original board was
  published. Questionable never counts as out, matching the live
  injury-context rule.
- ``depth_charts``: the latest snapshot dated on or before the original
  run's as_of_date gives ``depth_chart_order`` (pos_rank within team and
  position).

Documented gaps: career_games, career_starts, career_seasons,
prior_fantasy_ppg and draft_round are not carried by any nflverse file
in an as-of form, so they are left unset (the scorer's veteran gate
still fires on years_exp and age, which ARE reconstructed). A starter
hurt in week W who is neither RES nor designated on the W+1 report gets
no injury context; a practice injury during week W+1 can conversely be
picked up by the W+1 fallback. Both are small and bounded to one week.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
INJURIES_URL = _RELEASES + "/injuries/injuries_{season}.csv"
ROSTERS_URL = _RELEASES + "/weekly_rosters/roster_weekly_{season}.csv"
DEPTH_CHARTS_URL = _RELEASES + "/depth_charts/depth_charts_{season}.csv"

_SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}
# Sleeper scores fullbacks as running backs.
_POSITION_ALIASES = {"FB": "RB"}

NOT_RECONSTRUCTED_FIELDS = (
    "career_games, career_starts, career_seasons, prior_fantasy_ppg and "
    "draft_round are not available as-of from nflverse sources and were "
    "left unset"
)


# =============================================================================
# downloads (cached, mirroring nflverse_metrics' PFR helper)
# =============================================================================

def _download_csv(url: str, filename: str,
                  max_age_hours: float = 6.0) -> Optional[str]:
    """Cached download of one nflverse release CSV into the shared cache.

    Returns the local path, or None when the download fails and no cached
    copy exists. A stale cache beats nothing.
    """
    import time
    import urllib.request
    from pathlib import Path

    from utils.paths import CACHE_DIR

    dest = Path(CACHE_DIR) / filename
    try:
        if dest.exists() and (time.time() - dest.stat().st_mtime) < max_age_hours * 3600:
            return str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "fantasy-dashboard"})
        tmp = dest.with_name(dest.name + ".tmp")
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
        return str(dest)
    except Exception as exc:  # noqa: BLE001 - stale cache beats nothing
        logger.warning("reconstruction: download of %s failed (%s)", filename, exc)
        return str(dest) if dest.exists() else None


def _read_csv_rows(path: str, keep: Optional[Tuple[str, ...]] = None) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if keep is None:
            return [dict(r) for r in reader]
        return [{k: (r.get(k) or "") for k in keep} for r in reader]


def load_reconstruction_rows(season: int) -> Dict[str, List[Dict[str, str]]]:
    """Download (cached) and parse the three nflverse files for a season.

    Raises RuntimeError when any file is unavailable: a reconstruction
    built from partial inputs would silently mis-score, so the backfill
    stops loudly instead. Depth-chart rows are projected to the columns
    the builder needs (the full file is tens of MB).
    """
    season = int(season)
    paths = {
        "injuries": _download_csv(INJURIES_URL.format(season=season),
                                  f"nflverse_injuries_{season}.csv"),
        "rosters": _download_csv(ROSTERS_URL.format(season=season),
                                 f"nflverse_roster_weekly_{season}.csv"),
        "depth": _download_csv(DEPTH_CHARTS_URL.format(season=season),
                               f"nflverse_depth_charts_{season}.csv"),
    }
    missing = [name for name, path in paths.items() if not path]
    if missing:
        raise RuntimeError(
            f"reconstruction inputs unavailable for {season}: {', '.join(missing)}")
    return {
        "injuries": _read_csv_rows(paths["injuries"]),
        "rosters": _read_csv_rows(paths["rosters"]),
        "depth": _read_csv_rows(paths["depth"], keep=(
            "dt", "team", "gsis_id", "pos_abb", "pos_rank", "player_name")),
    }


def default_gsis_to_sleeper() -> Dict[str, str]:
    """The nflverse gsis -> Sleeper crosswalk the metrics pipeline uses."""
    try:
        from data_building.external_data.nflverse_metrics import _gsis_to_sleeper
        return _gsis_to_sleeper() or {}
    except Exception:
        logger.warning("reconstruction: gsis crosswalk unavailable", exc_info=True)
        return {}


# =============================================================================
# as-of override construction (pure given parsed rows)
# =============================================================================

def _clean_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _sleeper_id(value: Any) -> str:
    """Normalize a Sleeper id that may arrive as a float string ('4034.0')."""
    text = _clean_id(value)
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _parse_day(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _age_on(birth_date: Any, as_of: date) -> Optional[float]:
    born = _parse_day(birth_date)
    if born is None:
        return None
    years = (as_of - born).days / 365.25
    return round(years, 1) if years > 0 else None


def _depth_orders(depth_rows: List[Dict[str, str]], as_of_date: date) -> Dict[str, int]:
    """{gsis_id: depth_chart_order} from the latest depth snapshot dated
    on or before ``as_of_date``: pos_rank within (team, position group).
    A player listed at several slots keeps his best (lowest) rank."""
    snapshot: Optional[date] = None
    for row in depth_rows:
        day = _parse_day(row.get("dt"))
        if day is not None and day <= as_of_date and (snapshot is None or day > snapshot):
            snapshot = day
    if snapshot is None:
        return {}
    orders: Dict[str, int] = {}
    for row in depth_rows:
        if _parse_day(row.get("dt")) != snapshot:
            continue
        gsis = _clean_id(row.get("gsis_id"))
        rank = _int_or_none(row.get("pos_rank"))
        if not gsis or rank is None:
            continue
        if gsis not in orders or rank < orders[gsis]:
            orders[gsis] = rank
    return orders


def _injury_statuses(roster_rows: List[Dict[str, str]],
                     injury_rows: List[Dict[str, str]],
                     week: int) -> Dict[str, str]:
    """{gsis_id: Sleeper-shaped injury_status} as of the week-W board.

    RES on the week W or W+1 roster means "IR". Otherwise the injury
    report designation stands: week W first, then week W+1 (where an
    injury sustained in week W's games first appears)."""
    reserve = set()
    for row in roster_rows:
        if _int_or_none(row.get("week")) in (week, week + 1) \
                and str(row.get("status") or "").strip().upper() == "RES":
            gsis = _clean_id(row.get("gsis_id"))
            if gsis:
                reserve.add(gsis)
    report: Dict[Tuple[str, int], str] = {}
    for row in injury_rows:
        gsis = _clean_id(row.get("gsis_id"))
        wk = _int_or_none(row.get("week"))
        status = str(row.get("report_status") or "").strip()
        if gsis and wk is not None and status:
            report[(gsis, wk)] = status
    out: Dict[str, str] = {}
    for gsis in reserve | {g for g, _wk in report}:
        if gsis in reserve:
            out[gsis] = "IR"
        elif (gsis, week) in report:
            out[gsis] = report[(gsis, week)]
        elif (gsis, week + 1) in report:
            out[gsis] = report[(gsis, week + 1)]
    return out


def build_asof_overrides(
    *,
    roster_rows: List[Dict[str, str]],
    injury_rows: List[Dict[str, str]],
    depth_rows: List[Dict[str, str]],
    gsis_to_sleeper: Dict[str, str],
    week: int,
    as_of_date: date,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Build (players_index_override, full_players_override) for week W.

    Both dicts are keyed by Sleeper player id and shaped exactly like the
    live inputs ``run_weekly_breakout`` consumes, so the runner's own
    injury-context map and the pure scorer work unchanged.
    """
    week = int(week)
    depth = _depth_orders(depth_rows, as_of_date)
    injuries = _injury_statuses(roster_rows, injury_rows, week)

    week_rows: Dict[str, Dict[str, str]] = {}
    for row in roster_rows:
        if _int_or_none(row.get("week")) != week:
            continue
        gsis = _clean_id(row.get("gsis_id"))
        if gsis:
            week_rows[gsis] = row  # last row wins on a mid-week team change

    index_override: Dict[str, Dict[str, Any]] = {}
    feed_override: Dict[str, Dict[str, Any]] = {}
    for gsis, row in week_rows.items():
        pos = str(row.get("position") or "").strip().upper()
        pos = _POSITION_ALIASES.get(pos, pos)
        team = str(row.get("team") or "").strip()
        name = str(row.get("full_name") or "").strip()
        if pos not in _SKILL_POSITIONS or not team or not name:
            continue
        sleeper_id = _sleeper_id((gsis_to_sleeper or {}).get(gsis)) \
            or _sleeper_id(row.get("sleeper_id"))
        if not sleeper_id:
            continue
        index_override[sleeper_id] = {
            "name": name, "full_name": name,
            "pos": pos, "position": pos, "team": team,
        }
        entry: Dict[str, Any] = {
            "full_name": name, "team": team, "position": pos,
            "years_exp": _int_or_none(row.get("years_exp")),
            "rookie_year": _int_or_none(row.get("rookie_year")),
            # entry_year is the year the player entered the league: the
            # draft-year proxy the scorer's rookie logic needs.
            "draft_year": _int_or_none(row.get("entry_year")),
            "age": _age_on(row.get("birth_date"), as_of_date),
            "injury_status": injuries.get(gsis, ""),
        }
        if gsis in depth:
            entry["depth_chart_order"] = depth[gsis]
        feed_override[sleeper_id] = entry
    return index_override, feed_override


# =============================================================================
# backfill driver
# =============================================================================

def _detail_of(run: Dict[str, Any]) -> Dict[str, Any]:
    detail = (run or {}).get("detail")
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except ValueError:
            detail = {}
    return detail if isinstance(detail, dict) else {}


def reconstruct_week(
    season: int,
    week: int,
    *,
    rows: Optional[Dict[str, List[Dict[str, str]]]] = None,
    crosswalk: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Re-score one week under the current SCORING_VERSION with as-of
    inputs and publish it flagged as a reconstruction.

    Skips weeks that already have a completed original current-version
    run, and weeks with no completed original run at all (there is no
    original board to reconstruct). Re-running for a week that only has a
    reconstruction replaces it (publish is idempotent per week/version).
    """
    from data_building.breakout_engine import weekly_runner, weekly_store
    from data_building.breakout_engine.weekly_breakout import SCORING_VERSION

    season, week = int(season), int(week)
    base = {"season": season, "as_of_week": week}
    weekly_store.init_weekly_breakout_db()
    serving = weekly_store.get_serving_run(season, week)
    if serving is None:
        return {**base, "status": "skipped",
                "reason": "no completed original run for this week"}
    detail = _detail_of(serving)
    if serving.get("scoring_version") == SCORING_VERSION \
            and not detail.get("reconstructed"):
        return {**base, "status": "already_current",
                "reason": "week already has an original current-version run"}
    as_of = serving.get("as_of_date")
    if isinstance(as_of, str):
        as_of = _parse_day(as_of)
    if as_of is None:
        return {**base, "status": "skipped",
                "reason": "original run has no as_of_date to reconstruct against"}
    if rows is None:
        rows = load_reconstruction_rows(season)
    if crosswalk is None:
        crosswalk = default_gsis_to_sleeper()
    if not crosswalk:
        return {**base, "status": "failed",
                "reason": "gsis to Sleeper crosswalk unavailable"}
    index_override, feed_override = build_asof_overrides(
        roster_rows=rows["rosters"], injury_rows=rows["injuries"],
        depth_rows=rows["depth"], gsis_to_sleeper=crosswalk,
        week=week, as_of_date=as_of)
    if not index_override:
        return {**base, "status": "failed",
                "reason": "as-of overrides came out empty; inputs look wrong"}
    context = weekly_runner.ScoringContext(
        season=season, mode=weekly_runner.MODE_WEEKLY, as_of_date=as_of,
        cutoff_week=week, completed_weeks=list(range(1, week + 1)),
        reason="current-version reconstruction from as-of nflverse inputs",
    )
    detail_extra = {
        "reconstructed": True,
        "as_of_week": week,
        "injury_source": "nflverse",
        "original_scoring_version": serving.get("scoring_version"),
        "original_run_id": serving.get("id"),
        "not_reconstructed_fields": NOT_RECONSTRUCTED_FIELDS,
    }
    summary = weekly_runner.run_weekly_breakout(
        context, refresh=False,
        players_index_override=index_override,
        full_players_override=feed_override,
        run_detail_extra=detail_extra,
    )
    return {**base, **summary}


def reconstruct_season(season: int,
                       weeks: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Reconstruct every completed week of a season that lacks an
    original current-version run, oldest first (so each week's lifecycle
    chains off the previous week's reconstruction, exactly like the live
    pipeline). Downloads the nflverse inputs once for the whole pass."""
    from data_building.breakout_engine import weekly_store

    season = int(season)
    if weeks is None:
        weeks = [int(w["as_of_week"])
                 for w in weekly_store.list_completed_weeks(season)]
    rows = load_reconstruction_rows(season)
    crosswalk = default_gsis_to_sleeper()
    return [
        reconstruct_week(season, week, rows=rows, crosswalk=crosswalk)
        for week in sorted({int(w) for w in weeks})
    ]
