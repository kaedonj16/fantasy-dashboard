"""Sleeper-only projection helpers.

Weekly and season-level projection context comes exclusively from Sleeper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_CACHE_DIR = Path(__file__).parent.parent / "cache"

# Matches utils.proj_variant.pick_proj_variant keys and the weekly projection
# payload (ppr / half_ppr / std plus TE-premium and 6-point passing-TD layers).
PROJ_VARIANTS = ("ppr", "half_ppr", "std", "tep", "6pt_ppr", "6pt_half", "6pt_tep")

# Sleeper drops the weekly stat line (proj PPG 0) once a player is out for the
# year. Displayed point projections must not refill those players from any
# other source (FantasyPros preseason files, last-season actuals, etc.).
_SEASON_ENDING_INJURY = frozenset({"IR", "PUP", "NFI"})


def unprojected_season_injury(injury_status: Optional[str], sleeper_ppg) -> bool:
    """True when Sleeper has zeroed/omitted a player who is out for the year.

    PUP/IR players Sleeper still projects (expected return) keep that number.
    """
    status = str(injury_status or "").strip().upper()
    if status not in _SEASON_ENDING_INJURY:
        return False
    try:
        ppg = float(sleeper_ppg or 0)
    except (TypeError, ValueError):
        ppg = 0.0
    return ppg <= 0


# Plain PPR/half/std variants Sleeper publishes a precomputed total for.
_PUBLISHED_PTS_KEYS = {"ppr": "pts_ppr", "half_ppr": "pts_half_ppr", "std": "pts_std"}


def _published_base_points(raw_stats) -> dict[str, float]:
    """Sleeper's own published totals for the plain PPR/half/std variants.

    The weekly projection cache preserves the raw stat line, which carries
    Sleeper's precomputed pts_ppr/pts_half_ppr/pts_std alongside the raw stats.
    Preferring those makes the aggregated season PPG match the Sleeper app for
    standard scoring instead of a recompute that can drift on rounding or
    category coverage. TE-premium and 6-pt passing-TD variants have no published
    equivalent, so those keep the computed value.
    """
    if not isinstance(raw_stats, dict):
        return {}
    out: dict[str, float] = {}
    for variant, pts_key in _PUBLISHED_PTS_KEYS.items():
        val = raw_stats.get(pts_key)
        if val is None:
            continue
        try:
            out[variant] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def weekly_variant_values(weekly_maps) -> dict[str, dict[str, list[float]]]:
    """Collect positive weekly projection values per player and scoring variant.

    ``weekly_maps`` is an iterable of week dicts as stored on disk:
    ``{pid: {ppr, half_ppr, std, tep, 6pt_ppr, 6pt_half, 6pt_tep, raw_stats}}``.
    When the cached row preserved the raw stat line, Sleeper's own published
    pts_ppr/pts_half_ppr/pts_std are used for the plain PPR/half/std variants so
    the season PPG matches the Sleeper app. A missing variant falls back to that
    week's ``ppr`` value so a sparse week does not drop the player from a
    scoring-specific season average.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for week_map in weekly_maps or []:
        if not isinstance(week_map, dict):
            continue
        for pid, row in week_map.items():
            pid = str(pid)
            if isinstance(row, dict):
                ppr_fallback = row.get("ppr")
                published = _published_base_points(row.get("raw_stats"))
                for key in PROJ_VARIANTS:
                    value = published.get(key)
                    if value is None:
                        value = row.get(key)
                    if value is None:
                        value = ppr_fallback
                    try:
                        value = float(value or 0)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        out.setdefault(pid, {}).setdefault(key, []).append(value)
            else:
                try:
                    value = float(row or 0)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    out.setdefault(pid, {}).setdefault("ppr", []).append(value)
    return out


def season_ppg_from_weekly(weekly_maps) -> dict[str, dict[str, float]]:
    """Median-of-weeks PPG for every scoring variant: ``{pid: {variant: ppg}}``."""
    from statistics import median

    result: dict[str, dict[str, float]] = {}
    for pid, by_var in weekly_variant_values(weekly_maps).items():
        result[pid] = {
            key: round(float(median(vals)), 2)
            for key, vals in by_var.items()
            if vals
        }
    return result


def _year_weekly_values(year: int) -> dict[str, dict[str, list[float]]]:
    """Load weeks 1-18 and collect positive weekly values per scoring variant."""
    from utils.utils import load_week_projection

    weeks = [load_week_projection(year, week) or {} for week in range(1, 19)]
    return weekly_variant_values(weeks)


# Sleeper season-feed stat keys we can turn into PPG when weekly files omit a
# player. TE-premium / 6-pt passing-TD layers are weekly-only.
_SEASON_PTS_KEYS = {
    "ppr": "pts_ppr",
    "half_ppr": "pts_half_ppr",
    "std": "pts_std",
}


def _season_games_for_ppg(gp, season_games: float = 17.0) -> float:
    """Active-week divisor for a Sleeper season total.

    The season feed often stamps ``gp: 18`` (bye included). Use that only when
    it looks like actual games (1-17); otherwise the standard 17-game season.
    """
    try:
        games = float(gp or 0)
    except (TypeError, ValueError):
        games = 0.0
    if 1.0 <= games <= 17.0:
        return games
    return float(season_games)


def load_sleeper_season_stat_lines(year: int) -> dict[str, dict]:
    """Sleeper season projection totals: pid → {pts_ppr, pts_half_ppr, pts_std, gp}.

    Same public feed the ADP helper already hits. Cached for the calendar day.
    Returns {} on any failure so weekly aggregation still stands on its own.
    """
    import json
    from datetime import date

    cache_path = _CACHE_DIR / f"sleeper_season_proj_{int(year)}_{date.today().isoformat()}.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            pass

    url = (
        f"https://api.sleeper.com/projections/nfl/{int(year)}"
        "?season_type=regular&order_by=adp_ppr"
        "&position[]=QB&position[]=RB&position[]=WR&position[]=TE"
        "&position[]=K&position[]=DEF"
    )
    try:
        import requests
        resp = requests.get(url, timeout=20, headers={"User-Agent": "fantasy-dashboard/1.0"})
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return {}

    out: dict[str, dict] = {}
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("player_id") or "")
        stats = item.get("stats") or {}
        if not pid or not isinstance(stats, dict):
            continue
        row: dict = {}
        for key in ("pts_ppr", "pts_half_ppr", "pts_std", "gp"):
            val = stats.get(key)
            if val is None:
                continue
            try:
                row[key] = float(val)
            except (TypeError, ValueError):
                continue
        if row.get("pts_ppr") or row.get("pts_half_ppr") or row.get("pts_std"):
            out[pid] = row
    if out:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(out), encoding="utf-8")
            tmp.replace(cache_path)
        except Exception:
            pass
    return out


def fill_missing_from_season_totals(
    weekly_result: dict,
    season_lines: dict,
    *,
    scoring: str = "ppr",
    players_index: Optional[dict] = None,
    season_games: float = 17.0,
) -> dict:
    """Add Sleeper season-feed players that weekly files omitted.

    Weekly medians stay the source of truth when present. Preseason weekly
    caches sometimes omit a projected starter even though Sleeper's season
    feed has a full-year line (the A.J. Brown compare-modal N/A).
    """
    players_index = players_index or {}
    pts_key = _SEASON_PTS_KEYS.get(scoring, "pts_ppr")
    result = dict(weekly_result or {})
    for pid, stats in (season_lines or {}).items():
        pid = str(pid)
        if pid in result:
            continue
        if not isinstance(stats, dict):
            continue
        try:
            pts = float(stats.get(pts_key) or stats.get("pts_ppr") or 0)
        except (TypeError, ValueError):
            continue
        if pts <= 0:
            continue
        games = _season_games_for_ppg(stats.get("gp"), season_games)
        result[pid] = {
            "pos": str((players_index.get(pid) or {}).get("pos") or "").upper(),
            "season_pts": round(pts, 1),
            "ppg": round(pts / games, 2),
        }
    return result


def fill_missing_ppg_variants(
    weekly_variants: dict,
    season_lines: dict,
    season_games: float = 17.0,
) -> dict:
    """Add ppr/half_ppr/std PPG from season totals for players weekly files omitted."""
    result = {str(pid): dict(by_var) for pid, by_var in (weekly_variants or {}).items()}
    for pid, stats in (season_lines or {}).items():
        pid = str(pid)
        if not isinstance(stats, dict):
            continue
        if pid in result:
            continue
        games = _season_games_for_ppg(stats.get("gp"), season_games)
        if games <= 0:
            continue
        row = {}
        changed = False
        for variant, pts_key in _SEASON_PTS_KEYS.items():
            if row.get(variant):
                continue
            try:
                pts = float(stats.get(pts_key) or 0)
            except (TypeError, ValueError):
                continue
            if pts <= 0:
                continue
            row[variant] = round(pts / games, 2)
            changed = True
        if changed:
            result[pid] = row
    return result


def fetch_sleeper_season_ppg_variants(
    year: int,
    players_index: Optional[dict] = None,
) -> dict[str, dict[str, float]]:
    """Season PPG for every scoring variant, keyed by Sleeper player id.

    Shape: ``{pid: {ppr: 18.5, half_ppr: 16.2, 6pt_ppr: 20.1, ...}}``.
    PPG is the median of positive weekly values so byes do not dilute it.
    Players omitted from weekly files are filled from Sleeper's season feed.
    ``players_index`` is accepted so callers can share kwargs with the PPR helper.
    """
    collected = _year_weekly_values(year)
    from statistics import median
    result: dict[str, dict[str, float]] = {}
    for pid, by_var in collected.items():
        result[pid] = {
            key: round(float(median(vals)), 2)
            for key, vals in by_var.items()
            if vals
        }
    return fill_missing_ppg_variants(result, load_sleeper_season_stat_lines(year))


def fetch_sleeper_season_projections(
    year: int,
    scoring: str = "ppr",
    players_index: Optional[dict] = None,
) -> dict[str, dict]:
    """Aggregate Sleeper weekly projections into a season PPG baseline.

    The median positive weekly projection is used for PPG so bye weeks and
    missing weekly rows do not dilute a player's expected active-week output.
    Players the weekly files omit are filled from Sleeper's season projection
    feed (still Sleeper-only — not FantasyPros or last-season actuals).
    The returned shape matches the old season-projection helper, which lets all
    consumers use one Sleeper-only source without maintaining parallel models.
    """
    from statistics import median
    from utils.utils import load_players_index

    players_index = players_index or load_players_index() or {}
    variant = scoring if scoring in PROJ_VARIANTS else "ppr"
    collected = _year_weekly_values(year)

    result = {}
    for pid, by_var in collected.items():
        weekly = by_var.get(variant) or by_var.get("ppr") or []
        if not weekly:
            continue
        result[pid] = {
            "pos": str((players_index.get(pid) or {}).get("pos") or "").upper(),
            "season_pts": round(sum(weekly), 1),
            "ppg": round(float(median(weekly)), 2),
        }
    return fill_missing_from_season_totals(
        result,
        load_sleeper_season_stat_lines(year),
        scoring=variant,
        players_index=players_index,
    )


# ---------------------------------------------------------------------------
# Sleeper weekly projections (in-season)
# ---------------------------------------------------------------------------

def _sleeper_proj_cache_path(year: int, week: int) -> Path:
    return _CACHE_DIR / f"sleeper_proj_{year}_w{week:02d}.json"


def _raw_to_pts(st: dict, pos: str, scoring: str) -> float:
    """Compute fantasy points for one player from Sleeper raw projected stats."""
    pass_yd  = float(st.get("pass_yd")  or 0)
    pass_td  = float(st.get("pass_td")  or 0)
    pass_int = float(st.get("pass_int") or 0)
    rush_yd  = float(st.get("rush_yd")  or 0)
    rush_td  = float(st.get("rush_td")  or 0)
    rec      = float(st.get("rec")      or 0)
    rec_yd   = float(st.get("rec_yd")   or 0)
    rec_td   = float(st.get("rec_td")   or 0)
    fum_lost = float(st.get("fum_lost") or 0)

    if (pass_yd + pass_td + rush_yd + rush_td + rec + rec_yd + rec_td) == 0:
        return 0.0

    base = (
        pass_yd  * 0.04  + pass_td  * 4.0  + pass_int * -2.0
        + rush_yd * 0.1  + rush_td  * 6.0
        + rec_yd  * 0.1  + rec_td   * 6.0
        + fum_lost * -2.0
    )
    rec_pts  = 1.0 if scoring in ("ppr", "tep", "6pt_ppr", "6pt_tep") else (
        0.5 if scoring in ("half_ppr", "6pt_half") else 0.0)
    tep_pts  = 0.5 if (scoring in ("tep", "6pt_tep") and pos == "TE") else 0.0
    td6_pts  = pass_td * 2.0 if scoring in ("6pt_ppr", "6pt_half", "6pt_tep") else 0.0
    return round(base + rec * rec_pts + rec * tep_pts + td6_pts, 2)
