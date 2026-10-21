"""Live demo data for the guest landing page.

Supplies the two playable landing demos ("Who's more valuable?" and the
start/sit call) with the site's own numbers:

* Value pairs are generated from the site's dynasty values — the same rows
  the values board serves (``load_current_values_from_db``), falling back to
  ``data/model_values.json``. Pairs are picked algorithmically (closest
  superflex values among the stars) so the demo tracks the board as values
  move; nothing is hand-picked.
* Start/sit verdicts pair the guest preview board's projections with the
  site's matchup ratings (``cache/matchup_ratings_s<season>.json``): the
  reasoning line is computed from the ratings file, never hand-written.

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


# The guest preview board's own five matchups (projections as published on
# the landing hero board). loc is "vs" (home) or "@" (away) per player.
_BOARD = [
    ("QB", "Josh Allen", "BUF", "MIA", 31.2, "vs", "Lamar Jackson", "BAL", "KC", 29.4, "@"),
    ("RB", "Jahmyr Gibbs", "DET", "CLE", 19.8, "vs", "Derrick Henry", "BAL", "KC", 17.6, "@"),
    ("WR", "Drake London", "ATL", "WAS", 17.4, "vs", "Ja'Marr Chase", "CIN", "PIT", 19.1, "vs"),
    ("TE", "Sam LaPorta", "DET", "CLE", 12.6, "vs", "Mark Andrews", "BAL", "KC", 11.4, "@"),
    ("SF", "Justin Herbert", "LAC", "DEN", 20.1, "vs", "Joe Burrow", "CIN", "PIT", 21.0, "vs"),
]


def _start_sit_pairs(ratings: dict, week_label: str) -> list:
    pairs = []
    for pos, na, ta, oa, pa, la, nb, tb, ob, pb, lb in _BOARD:
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
    week_label = _week_label(ratings_doc)
    start_sit_pairs = []
    if ratings_doc:
        try:
            start_sit_pairs = _start_sit_pairs(
                ratings_doc.get("ratings") or {}, week_label
            )
        except Exception:
            start_sit_pairs = []
    return {
        "value_pairs": value_pairs,
        "start_sit_pairs": start_sit_pairs,
        "week_label": week_label,
    }
