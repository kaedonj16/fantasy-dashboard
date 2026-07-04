"""Deterministic play-by-play simulation for the Redzone demo page.

Extracted from app.py so the simulation is unit-testable. Every player's
"game" is scripted from a seeded LCG keyed on the player id, so the demo is
stable across page loads while still looking like live football: folding the
script up to a time t yields that player's cumulative stat line and points.
"""
from typing import Callable

DEMO_GAME_SECONDS = 600  # sim seconds = "full game"

DEMO_SCORING = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
    "rush_yd": 0.1, "rush_td": 6.0,
    "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0,
}


def demo_rng(seed: int) -> Callable[[], float]:
    """Tiny seeded LCG returning floats in [0, 1). Deterministic per seed."""
    s = seed & 0x7FFFFFFF or 1

    def nxt():
        nonlocal s
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        return s / 0x7FFFFFFF
    return nxt


def demo_script(pid: str, pos: str, game_seconds: int = DEMO_GAME_SECONDS) -> list:
    """Deterministic per-player list of plays across the simulated game."""
    r = demo_rng(sum(ord(c) for c in pid) * 2654435761)
    plays, tt = [], 18 + int(r() * 40)
    while tt < game_seconds:
        roll = r()
        if pos == "QB":
            if roll < 0.76:
                plays.append({"t": tt, "kind": "pass", "yds": 4 + int(r() * 28),
                              "td": 1 if r() < 0.11 else 0})
            elif roll < 0.90:
                plays.append({"t": tt, "kind": "rush", "yds": 1 + int(r() * 12),
                              "td": 1 if r() < 0.08 else 0})
            else:
                plays.append({"t": tt, "kind": "int"})
            tt += 32 + int(r() * 40)
        elif pos == "RB":
            if roll < 0.66:
                plays.append({"t": tt, "kind": "rush", "yds": int(r() * 14),
                              "td": 1 if r() < 0.07 else 0})
            elif roll < 0.86:
                plays.append({"t": tt, "kind": "rec", "yds": 2 + int(r() * 11),
                              "td": 1 if r() < 0.05 else 0})
            else:
                plays.append({"t": tt, "kind": "target"})
            tt += 46 + int(r() * 54)
        else:  # WR / TE
            if roll < 0.50:
                plays.append({"t": tt, "kind": "rec", "yds": 3 + int(r() * 22),
                              "td": 1 if r() < 0.09 else 0})
            else:
                plays.append({"t": tt, "kind": "target"})
            tt += 50 + int(r() * 70)
    return plays


def demo_fold(plays: list, t: float) -> dict:
    """Cumulative stat line from the plays that have happened by sim-time t."""
    L = {"pass_yds": 0.0, "pass_td": 0.0, "int": 0.0, "carries": 0.0,
         "rush_yds": 0.0, "rush_td": 0.0, "rec": 0.0, "rec_yds": 0.0,
         "rec_td": 0.0, "targets": 0.0}
    for p in plays:
        if p["t"] > t:
            break
        k = p["kind"]
        if k == "rush":
            L["carries"] += 1; L["rush_yds"] += p["yds"]; L["rush_td"] += p.get("td", 0)
        elif k == "rec":
            L["rec"] += 1; L["targets"] += 1; L["rec_yds"] += p["yds"]; L["rec_td"] += p.get("td", 0)
        elif k == "target":
            L["targets"] += 1
        elif k == "pass":
            L["pass_yds"] += p["yds"]; L["pass_td"] += p.get("td", 0)
        elif k == "int":
            L["int"] += 1
    return L


def demo_pts(L: dict, scoring: dict = None) -> float:
    """Fantasy points for a folded stat line under the demo scoring rules."""
    s = scoring or DEMO_SCORING
    return round(
        L["pass_yds"] * s["pass_yd"] + L["pass_td"] * s["pass_td"] + L["int"] * s["pass_int"]
        + L["rush_yds"] * s["rush_yd"] + L["rush_td"] * s["rush_td"]
        + L["rec"] * s["rec"] + L["rec_yds"] * s["rec_yd"] + L["rec_td"] * s["rec_td"],
        2,
    )
