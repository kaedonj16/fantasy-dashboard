"""Live demo data for the guest landing page.

Supplies the two playable landing demos ("Who's more valuable?" and the
start/sit call) with the site's own numbers:

* Value pairs are generated from the site's dynasty values — the same rows
  the values board serves (``load_current_values_from_db``), falling back to
  ``data/model_values.json``. Pairs are picked algorithmically (closest
  superflex values among the stars) so the demo tracks the board as values
  move; nothing is hand-picked.
* Start/sit verdicts pair the site's own upcoming-week projections (the
  cached Sleeper stat lines priced by the site's variant converter, PPR)
  with the site's matchup ratings (``cache/matchup_ratings_s<season>.json``):
  the reasoning line is computed from the ratings file, never hand-written.
  No projections, no demo: the pairs are never backfilled from a snapshot.

Stdlib-only imports so this module (and its tests) run anywhere. Every
loader fails soft: missing data yields empty lists, never an exception.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_VALUES_PATH = ROOT_DIR / "data" / "model_values.json"
CACHE_DIR = ROOT_DIR / "cache"


def _find_ratings_path() -> Path:
    """Newest plain matchup-ratings file (profile-hash variants excluded)."""
    try:
        cands = sorted(CACHE_DIR.glob("matchup_ratings_s[0-9][0-9][0-9][0-9].json"))
        if cands:
            return cands[-1]
    except Exception:
        pass
    return CACHE_DIR / "matchup_ratings_s2026.json"


MATCHUP_RATINGS_PATH = _find_ratings_path()

_POSITIONS = ("QB", "RB", "WR", "TE")


# ---------------------------------------------------------------------------
# Dynasty values
# ---------------------------------------------------------------------------

def _db_value_rows() -> list:
    """The runtime values path: the same loader the values board uses."""
    try:
        from dashboard_services.player_value_history import (
            load_current_values_from_db,
        )

        return load_current_values_from_db() or []
    except Exception:
        return []


def _file_value_rows() -> list:
    try:
        data = json.loads(MODEL_VALUES_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _load_value_rows() -> list:
    rows = _db_value_rows()
    if rows:
        return rows
    return _file_value_rows()


def _clean_players(rows) -> list:
    """Normalize value rows to the fields the demo needs (SF values)."""
    out = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or "").strip()
        team = str(r.get("team") or "").strip()
        pos = str(r.get("position") or "").strip().upper()
        label = str(r.get("sf_pos_rank_label") or "").strip()
        try:
            sf = float(r.get("sf_value") or 0)
        except (TypeError, ValueError):
            continue
        if not name or not team or pos not in _POSITIONS or not label or sf <= 0:
            continue
        out.append(
            {"name": name, "team": team, "position": pos, "label": label, "sf": sf}
        )
    return out


def _value_pairs(rows) -> list:
    """Closest-value star pairs. Winner is always side "a" (pick 0); the
    client flips sides randomly so the answer position varies."""
    stars = sorted(
        (p for p in _clean_players(rows) if p["sf"] >= 660),
        key=lambda p: -p["sf"],
    )[:24]
    cands = []
    for i in range(len(stars)):
        for j in range(i + 1, len(stars)):
            gap = round(stars[i]["sf"]) - round(stars[j]["sf"])
            if 5 <= gap <= 60:
                cands.append((gap, stars[i], stars[j]))
    cands.sort(key=lambda c: c[0])

    used: dict = {}
    pairs = []
    for elite_only in (True, False):
        for gap, w, l in cands:
            if len(pairs) >= 8:
                break
            if elite_only and min(w["sf"], l["sf"]) < 750:
                continue
            if used.get(w["name"], 0) >= 2 or used.get(l["name"], 0) >= 2:
                continue
            used[w["name"]] = used.get(w["name"], 0) + 1
            used[l["name"]] = used.get(l["name"], 0) + 1
            wv, lv = round(w["sf"]), round(l["sf"])
            if w["position"] == l["position"]:
                q = f"Two of the top {w['position']}s on our board. Who is more valuable?"
            else:
                q = f"{w['position']} vs {l['position']} in superflex. Who is more valuable?"
            pairs.append(
                {
                    "q": q,
                    # Rank labels never appear in q/m: they would spoil the pick.
                    "a": {"n": w["name"], "m": f"{w['position']} · {w['team']}", "v": wv},
                    "b": {"n": l["name"], "m": f"{l['position']} · {l['team']}", "v": lv},
                    "pick": 0,
                    "why": (
                        f"<b>{w['name']}, {wv:,} to {lv:,}.</b> "
                        f"{w['label']} vs {l['label']} on our values board: "
                        f"a {gap}-point gap."
                    ),
                }
            )
    return pairs


# ---------------------------------------------------------------------------
# Matchup ratings + start/sit
# ---------------------------------------------------------------------------

def _load_ratings_doc():
    try:
        doc = json.loads(MATCHUP_RATINGS_PATH.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else None
    except Exception:
        return None


def _fpts_rank(ratings: dict, opp: str, pos: str):
    """(fpts allowed, rank) for a defense vs a position; rank 1 = most allowed."""
    rows = []
    for team, by_pos in (ratings or {}).items():
        try:
            rows.append((team, float((by_pos or {}).get(pos, {}).get("fpts"))))
        except (TypeError, ValueError, AttributeError):
            continue
    rows.sort(key=lambda t: -t[1])
    for i, (team, fpts) in enumerate(rows, 1):
        if team == opp:
            return fpts, i
    return None, None


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _demo_season_week():
    """(season, week) the demo should price, from the site's NFL state.

    Regular season only; anything else yields (None, None) and the
    start/sit demo stays hidden rather than showing a stale week.
    """
    try:
        from dashboard_services.api import get_nfl_state

        state = get_nfl_state() or {}
    except Exception:
        return None, None
    try:
        season = int(state.get("season") or 0)
        week = int(state.get("week") or 0)
    except (TypeError, ValueError):
        return None, None
    phase = str(state.get("season_type") or "").strip().lower()
    if season >= 2020 and week >= 1 and phase not in ("post", "off", "pre"):
        return season, week
    return None, None


def _live_projection_rows(season: int, week: int) -> dict:
    """Upcoming-week starters from the site's own projection path.

    Reads the same cached weekly projections every league page uses
    (Sleeper stat lines priced by the site's variant converter), joined
    to the players index for name/team/position and to the week schedule
    for opponent and home/away. Returns {pos: [(name, team, opp, loc,
    ppr_pts), ...]} sorted by projection, best first. Players on a bye
    have no schedule entry and drop out; unprojected players (0 pts)
    drop out. Any failure yields empty pools.
    """
    rows = {pos: [] for pos in _POSITIONS}
    try:
        from utils.data_cache import (
            get_week_projections_cached,
            load_players_index,
            load_week_schedule,
        )
        from utils.projections import fetch_week_projections

        proj = get_week_projections_cached(season, week, fetch_week_projections) or {}
        players = load_players_index() or {}
        schedule = load_week_schedule(season, week) or []
    except Exception:
        return rows
    opp_map = {}
    for game in schedule:
        if not isinstance(game, dict):
            continue
        home, away = game.get("home"), game.get("away")
        if home and away:
            opp_map[home] = (away, "vs")
            opp_map[away] = (home, "@")
    for pid, variants in proj.items():
        if not isinstance(variants, dict):
            continue
        info = players.get(str(pid)) or {}
        pos = str(info.get("pos") or "").strip().upper()
        if pos not in rows:
            continue
        try:
            pts = float(variants.get("ppr") or 0)
        except (TypeError, ValueError):
            continue
        name = str(info.get("name") or "").strip()
        team = str(info.get("team") or "").strip()
        opp_loc = opp_map.get(team)
        if pts <= 0 or not name or not team or not opp_loc:
            continue
        rows[pos].append((name, team, opp_loc[0], opp_loc[1], pts))
    for pos in rows:
        rows[pos].sort(key=lambda r: -r[4])
    return rows


def _closest_pair(pool: list, used: set):
    """The most debatable pair in a projection-sorted pool.

    Considers the top 8 unused players and returns the (hi, lo) pair
    with the smallest projection gap inside [0.2, 4.0] points: wide
    enough to have a verdict, tight enough to be a real question.
    Returns None when no pair qualifies.
    """
    cand = [r for r in pool[:8] if r[0] not in used]
    best = None
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            gap = round(cand[i][4] - cand[j][4], 2)
            if gap < 0.2:
                continue
            if gap > 4.0:
                break
            if best is None or gap < best[0]:
                best = (gap, cand[i], cand[j])
    if best is None:
        return None
    return best[1], best[2]


def _start_sit_pairs(ratings: dict, week_label: str, season: int, week: int) -> list:
    rows = _live_projection_rows(season, week)
    pairs = []
    used: set = set()
    plan = [("QB", rows.get("QB") or []), ("RB", rows.get("RB") or []),
            ("WR", rows.get("WR") or []), ("TE", rows.get("TE") or []),
            ("SF", rows.get("QB") or [])]
    for pos, pool in plan:
        picked = _closest_pair(pool, used)
        if picked is None:
            continue
        (na, ta, oa, la, pa), (nb, tb, ob, lb, pb) = picked
        used.add(na)
        used.add(nb)
        pick = 0 if pa >= pb else 1
        win, lose = ((na, oa, pa), (nb, ob, pb)) if pick == 0 else ((nb, ob, pb), (na, oa, pa))
        rpos = "QB" if pos == "SF" else pos
        fw, rw = _fpts_rank(ratings, win[1], rpos)
        fl, rl = _fpts_rank(ratings, lose[1], rpos)
        if fw is None or fl is None:
            continue
        why = (
            f"<b>Start {win[0]}.</b> +{abs(pa - pb):.1f} projected. "
            f"In our ratings {win[1]} allows {fw:.1f} {rpos} pts/g "
            f"({_ordinal(rw)}-most) and {lose[1]} allows {fl:.1f} "
            f"({_ordinal(rl)}-most)."
        )
        if pos == "SF":
            q = f"{week_label} · Your superflex spot. Who do you start?"
        else:
            q = f"{week_label} · One {pos} spot. Who do you start?"
        pairs.append(
            {
                "q": q,
                "a": {"n": na, "m": f"{ta} · {la} {oa}", "pr": f"{pa:.1f}"},
                "b": {"n": nb, "m": f"{tb} · {lb} {ob}", "pr": f"{pb:.1f}"},
                "pick": pick,
                "why": why,
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# Week label
# ---------------------------------------------------------------------------

def _nfl_state_week():
    """Current regular-season week from the site's NFL-state helper."""
    try:
        from dashboard_services.api import get_nfl_state

        state = get_nfl_state() or {}
    except Exception:
        return None
    try:
        week = int(state.get("week") or 0)
    except (TypeError, ValueError):
        return None
    phase = str(state.get("season_type") or "").strip().lower()
    if week >= 1 and phase not in ("post", "off", "pre"):
        return week
    return None


def _week_label(ratings_doc) -> str:
    tw = (ratings_doc or {}).get("through_week")
    if isinstance(tw, int) and tw >= 1:
        return f"Week {tw}"
    week = _nfl_state_week()
    if week:
        return f"Week {week}"
    return "This week"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_landing_demo_data() -> dict:
    """Demo payload for the guest landing page. Never raises."""
    try:
        value_pairs = _value_pairs(_load_value_rows())
    except Exception:
        value_pairs = []
    ratings_doc = _load_ratings_doc()
    season, week = _demo_season_week()
    start_sit_pairs = []
    if ratings_doc and season and week:
        try:
            start_sit_pairs = _start_sit_pairs(
                ratings_doc.get("ratings") or {}, f"Week {week}", season, week
            )
        except Exception:
            start_sit_pairs = []
    # The label names the week the pairs are priced for; only when no
    # live pairs exist does it fall back to the ratings/NFL-state label.
    week_label = f"Week {week}" if start_sit_pairs else _week_label(ratings_doc)
    return {
        "value_pairs": value_pairs,
        "start_sit_pairs": start_sit_pairs,
        "week_label": week_label,
    }
