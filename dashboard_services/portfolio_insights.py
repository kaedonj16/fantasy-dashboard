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
  starter's actual fantasy points when games are under way. Outlook lines
  for EVERY league's matchup are computed from live matchup margins:
  what the user needs from their remaining starters to flip a game, what
  the opponent needs to flip one back, when a game is locked up, and the
  final result once every starter's NFL game is done.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

# The waiver selector's base value floor (utils/digest.py WEIGHTS.min_value):
# a bench player below it is a stash, not coverage.
STARTABLE_VALUE = 25.0
FLEX_POSITIONS = {"RB", "WR", "TE"}


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
        # A window whose games are ALL final can never change again: the
        # client collapses it by default so live windows (and the
        # outlook lines above them) are what the user lands on.
        w["final"] = all(c.get("status") == "final" for c in w["games"])
        out.append(w)
    return out


def _who(names) -> str:
    names = [n for n in (names or []) if n]
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:2]) + " and more"


def _proj_clause(total) -> str:
    """The remaining group's combined projection, when it is known:
    the context that turns a raw need number into a judgement call."""
    try:
        total = float(total)
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    return f" The remaining players are projected for {total:.1f} combined."


def outlook_lines(margins: Dict[str, dict]) -> List[dict]:
    """One win-outlook line per league, from live matchup margins.

    ``margins`` maps league_key -> {league_name, my_score, opp_score,
    final, my_remaining_names, my_remaining_proj, opp_remaining_names,
    opp_remaining_proj}. Remaining means the starter's NFL game is not
    final yet, on EITHER side: the old final-window need lines only ever
    answered "what do I need" for leagues behind in the last window, and
    never answered the other half of the question (how few points the
    opponent needs to flip a game the user leads, whether a lead is
    already locked, or what the final result was).

    Returns [{league_key, league_name, state, line}] sorted with live
    games first by closeness (either direction), finals last. States:
    behind / ahead / tied / locked (ahead, opponent has nobody left) /
    gone (behind, user has nobody left) / final_won / final_lost /
    final_tied."""
    entries = []
    for key in sorted(margins or {}):
        m = margins[key] or {}
        try:
            mine = float(m.get("my_score") or 0)
            theirs = float(m.get("opp_score") or 0)
        except (TypeError, ValueError):
            continue
        name = m.get("league_name") or key
        my_names = [n for n in (m.get("my_remaining_names") or []) if n]
        opp_names = [n for n in (m.get("opp_remaining_names") or []) if n]
        diff = mine - theirs
        if m.get("final"):
            if diff > 0:
                state = "final_won"
                line = (f"{name}: won by {diff:.1f}, "
                        f"{mine:.1f} to {theirs:.1f}.")
            elif diff < 0:
                state = "final_lost"
                line = (f"{name}: lost by {abs(diff):.1f}, "
                        f"{mine:.1f} to {theirs:.1f}.")
            else:
                state = "final_tied"
                line = f"{name}: tied at {mine:.1f}."
            entries.append({"league_key": key, "league_name": name,
                            "state": state, "line": line,
                            "_sort": (1, 0.0)})
            continue
        if diff < 0:
            need = abs(diff) + 0.1
            if my_names:
                state = "behind"
                line = (f"{name} needs {need:.1f} combined from "
                        f"{_who(my_names)} to flip their game."
                        + _proj_clause(m.get("my_remaining_proj")))
            else:
                state = "gone"
                line = (f"{name} is down {abs(diff):.1f} and you have "
                        f"nobody left to play: this one is gone unless "
                        f"stat corrections flip it.")
        elif diff > 0:
            if opp_names:
                state = "ahead"
                need = diff + 0.1
                verb = "needs" if len(opp_names) == 1 else "need"
                line = (f"{name} is up {diff:.1f}. {_who(opp_names)} "
                        f"{verb} {need:.1f} combined to flip it."
                        + _proj_clause(m.get("opp_remaining_proj")))
            else:
                state = "locked"
                line = (f"{name} is up {diff:.1f} and your opponent has "
                        f"nobody left to play: only stat corrections "
                        f"can change this one.")
        else:
            state = "tied"
            if my_names:
                line = (f"{name} is tied: any points from "
                        f"{_who(my_names)} flip it your way.")
            elif opp_names:
                line = (f"{name} is tied and you have nobody left: "
                        f"any points from {_who(opp_names)} flip it "
                        f"the other way.")
            else:
                line = f"{name} is tied with nobody left on either side."
        entries.append({"league_key": key, "league_name": name,
                        "state": state, "line": line,
                        "_sort": (0, abs(diff))})
    entries.sort(key=lambda e: e["_sort"])
    for e in entries:
        e.pop("_sort", None)
    return entries
