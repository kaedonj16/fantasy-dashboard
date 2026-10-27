"""Portfolio insights: Bye Watch coverage and the Rooting Guide.

Pure builders for two My Leagues insight cards, fed by the insights API
from cached league contexts (never a fresh platform scan):

- Bye Watch judges next week's bye coverage from the user's own bench with
  the site's startability bar (the waiver selector's base value floor),
  never from a guess: a hole is called only when no bench player at an
  eligible position clears the bar, and the fix names the actual top free
  agent at that position when one exists.
- The Rooting Guide groups the user's starters by NFL game window and
  game, in ScoreZone's normalized game shape, with live scores and each
  starter's actual fantasy points when games are under way. Need lines for
  the final window are computed from live matchup margins only.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

# The waiver selector's base value floor (utils/digest.py WEIGHTS.min_value):
# a bench player below it is a stash, not coverage.
STARTABLE_VALUE = 25.0
FLEX_POSITIONS = {"RB", "WR", "TE"}
OUT_OF_REACH_MARGIN = 25.0


# ── Bye Watch ────────────────────────────────────────────────────────────────

def _eligible_positions(out_entry: dict) -> set:
    pos = str(out_entry.get("pos") or "").upper()
    slot = str(out_entry.get("slot") or "").upper()
    eligible = {pos} if pos else set()
    if slot in ("FLEX", "WRRB_FLEX", "REC_FLEX"):
        eligible |= FLEX_POSITIONS
    elif slot in ("SUPER_FLEX", "OP", "QB_FLEX"):
        eligible |= FLEX_POSITIONS | {"QB"}
    return eligible


def bye_row(league: dict) -> dict:
    """One league's Bye Watch row.

    Input: {name, platform, league_id, out: [{pid, name, pos, slot}],
    bench: [{pid, name, pos, value}], top_fa: {POS: {pid, name, value}}}.
    Verdicts: "Full strength" (nobody out), "Covered" (every out starter
    has a startable bench cover), else "Hole · POS" for the first uncovered
    position, with a fix naming the top free agent there when one exists."""
    out = list(league.get("out") or [])
    bench = list(league.get("bench") or [])
    base = {
        "name": league.get("name") or "",
        "platform": league.get("platform") or "",
        "league_id": league.get("league_id") or "",
        "out": out,
        "out_count": len(out),
        "hole_pos": None,
        "fix": None,
        "cover_note": "",
    }
    if not out:
        base["verdict"] = "Full strength"
        # cover_note stays empty: the row already says "Full strength"
        # once, and a second "Nobody on bye." line only repeats it.
        return base
    used = set()
    covers = []
    hole = None
    for entry in out:
        eligible = _eligible_positions(entry)
        cover = None
        for b in bench:
            if str(b.get("pid")) in used:
                continue
            if str(b.get("pos") or "").upper() not in eligible:
                continue
            try:
                value = float(b.get("value") or 0)
            except (TypeError, ValueError):
                value = 0.0
            if value >= STARTABLE_VALUE:
                cover = b
                break
        if cover is not None:
            used.add(str(cover.get("pid")))
            covers.append((entry, cover))
        elif hole is None:
            hole = entry
    if hole is None:
        base["verdict"] = "Covered"
        if covers:
            entry, cover = covers[0]
            base["cover_note"] = (
                f"{cover.get('name')} covers {entry.get('pos')} from your bench.")
        return base
    hole_pos = str(hole.get("pos") or "").upper()
    base["verdict"] = f"Hole · {hole_pos}"
    base["hole_pos"] = hole_pos
    fa = (league.get("top_fa") or {}).get(hole_pos)
    if fa:
        base["fix"] = {
            "text": f"Fix: stream {fa.get('name')}, the top free agent {hole_pos}.",
            "move": {
                "platform": league.get("platform") or "",
                "league_id": league.get("league_id") or "",
                "season": league.get("season"),
                "pos": hole_pos,
                "pid": str(fa.get("pid") or ""),
                "name": fa.get("name") or "",
            },
        }
    else:
        base["fix"] = {
            "text": f"Fix: no startable free agent {hole_pos} right now. "
                    "Watch the waiver wire.",
            "move": None,
        }
    return base


def bye_watch_rows(leagues) -> List[dict]:
    return [bye_row(lg) for lg in leagues or []]


# ── Rooting Guide ────────────────────────────────────────────────────────────

def normalize_game(gd: dict) -> dict:
    """Tank01 game dict -> ScoreZone's normalized game shape."""
    ls = gd.get("lineScore") if isinstance(gd.get("lineScore"), dict) else {}
    code = str(gd.get("gameStatusCode") or "")
    status = {"0": "pre", "1": "in", "2": "final"}.get(code, "pre")
    epoch = gd.get("gameTime_epoch") or gd.get("gameTimeEpoch")
    try:
        epoch = float(epoch) if epoch else None
    except (TypeError, ValueError):
        epoch = None
    return {
        "home": gd.get("home") or "",
        "away": gd.get("away") or "",
        "home_pts": gd.get("homePts") if gd.get("homePts") is not None else gd.get("homeScore"),
        "away_pts": gd.get("awayPts") if gd.get("awayPts") is not None else gd.get("awayScore"),
        "status": status,
        "game_status": gd.get("gameStatus") or "",
        "game_quarter": ls.get("period") or "",
        "game_clock": gd.get("gameClock") or "",
        "epoch": epoch,
    }


def window_for(epoch) -> Optional[dict]:
    """The viewing window for a kickoff: Sun 1 PM / Sun 4 PM / Mon night...

    Buckets follow the mock: Sunday early (before 3 PM ET), Sunday late
    (3 PM to 7:30 PM ET), then the night games by day."""
    if not epoch:
        return None
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromtimestamp(float(epoch), ZoneInfo("America/New_York"))
    except Exception:
        return None
    day = dt.strftime("%a")
    minutes = dt.hour * 60 + dt.minute
    if day == "Sun":
        if minutes < 15 * 60:
            key, label = "sun-early", "Sun 1 PM"
        elif minutes < 19 * 60 + 30:
            key, label = "sun-late", "Sun 4 PM"
        else:
            key, label = "sun-night", "Sun night"
    elif day == "Mon":
        key, label = "mon", "Mon night"
    elif day == "Thu":
        key, label = "thu", "Thu night"
    elif day == "Sat":
        key, label = "sat", "Saturday"
    elif day == "Fri":
        key, label = "fri", "Fri night"
    else:
        key, label = day.lower(), day
    return {"key": key, "label": label, "epoch": float(epoch)}


def game_status_line(game: dict) -> str:
    """Centered status text for a game card: kickoff for upcoming games,
    quarter and clock while live (ScoreZone grammar), Final when done."""
    status = game.get("status")
    if status == "final":
        return "Final"
    if status == "in":
        q = str(game.get("game_quarter") or "").strip()
        clock = str(game.get("game_clock") or "").strip()
        qtxt = q if q.upper().startswith("Q") else (f"Q{q}" if q else "")
        return " · ".join(p for p in (qtxt, clock) if p) or "Live"
    epoch = game.get("epoch")
    if epoch:
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.fromtimestamp(float(epoch), ZoneInfo("America/New_York"))
            return f"{dt.strftime('%a')} · {dt.strftime('%-I:%M %p')}"
        except Exception:
            pass
    return str(game.get("game_status") or "Upcoming")


def group_rooting(games, starters, team_records: Optional[dict] = None,
                  team_colors: Optional[dict] = None) -> List[dict]:
    """Group starters under their NFL games, games under viewing windows.

    ``games`` are normalized game dicts (normalize_game); ``starters`` are
    {pid, name, pos, team, league_key, league_name, points, proj}. Games
    with no user starter are dropped; windows sort by earliest kickoff."""
    by_teams = {}
    for g in games or []:
        by_teams[(g.get("away"), g.get("home"))] = g
    starters_by_game: Dict[tuple, List[dict]] = {}
    for s in starters or []:
        team = s.get("team")
        for key, g in by_teams.items():
            if team in key:
                starters_by_game.setdefault(key, []).append(s)
                break
    windows: Dict[str, dict] = {}
    for (away, home), game in by_teams.items():
        game_starters = starters_by_game.get((away, home)) or []
        if not game_starters:
            continue
        win = window_for(game.get("epoch"))
        if win is None:
            continue
        sides = {
            "away": [s for s in game_starters if s.get("team") == away],
            "home": [s for s in game_starters if s.get("team") == home],
        }
        card = {
            "away": away, "home": home,
            "away_rec": (team_records or {}).get(away, ""),
            "home_rec": (team_records or {}).get(home, ""),
            "away_color": (team_colors or {}).get(away, ""),
            "home_color": (team_colors or {}).get(home, ""),
            "away_pts": game.get("away_pts"), "home_pts": game.get("home_pts"),
            "status": game.get("status") or "pre",
            "status_line": game_status_line(game),
            "epoch": game.get("epoch"),
            "sides": sides,
            "starter_count": len(game_starters),
            "league_keys": sorted({s.get("league_key") for s in game_starters if s.get("league_key")}),
        }
        w = windows.setdefault(win["key"], {
            "key": win["key"], "label": win["label"],
            "epoch": win["epoch"], "games": [],
        })
        w["epoch"] = min(w["epoch"], win["epoch"])
        w["games"].append(card)
    out = []
    for w in sorted(windows.values(), key=lambda x: x["epoch"]):
        w["games"].sort(key=lambda c: c.get("epoch") or 0)
        n_starters = sum(c["starter_count"] for c in w["games"])
        n_leagues = len({k for c in w["games"] for k in c["league_keys"]})
        w["stakes"] = (f"{n_starters} starter{'s' if n_starters != 1 else ''} "
                       f"across {n_leagues} of your leagues.")
        w["starter_count"] = n_starters
        out.append(w)
    return out


def need_lines(margins: Dict[str, dict]) -> List[str]:
    """Final-window need lines from live matchup margins.

    ``margins`` maps league_key -> {league_name, my_score, opp_score,
    remaining_names, final}. A league behind with starters still to play in
    the window gets a flip line; a league behind by a lot with nobody left
    on either side is called out of reach. Leading or finished leagues are
    omitted rather than padded."""
    lines = []
    hopeless = []
    for key in sorted(margins or {}):
        m = margins[key] or {}
        try:
            mine = float(m.get("my_score") or 0)
            theirs = float(m.get("opp_score") or 0)
        except (TypeError, ValueError):
            continue
        if m.get("final"):
            continue
        deficit = theirs - mine
        names = [n for n in (m.get("remaining_names") or []) if n]
        name = m.get("league_name") or key
        if deficit > 0 and names:
            need = deficit + 0.1
            who = " and ".join(names[:2]) if len(names) <= 2 else (
                ", ".join(names[:2]) + " and more")
            lines.append(
                f"{name} needs {need:.1f} combined from {who} to flip their game.")
        elif deficit >= OUT_OF_REACH_MARGIN and not names:
            hopeless.append(
                f"{name} has nobody left in this window and is out of reach.")
    return (lines + hopeless)[:3]
