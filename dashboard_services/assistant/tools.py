"""Site-data tool layer for the BR Assistant's open answering path.

Deterministic, read-only wrappers over the site's existing data paths
(the same sources the evidence builders use: the players index, the
model value table, advanced metrics, the week-stat cache, value history,
and the league-context builders). Every tool returns a small evidence
dict of distilled rows: only fields an answer may cite, never a raw
payload. Tools never write anything, never raise (a failing source
degrades to ok=False with a missing note), and enforce the site's PRO
gate INSIDE the tool: a gated metric comes back labeled gated with no
numbers at all, so no downstream composer can leak them.

Evidence entry shape:
    {"tool": <name>, "ok": bool, ... distilled data ...,
     "players": [{"player_id", "name"}], "receipts": [{"label", "value"}],
     "link": {"href", "label"} | {"player_id", "player_name", "label"} | None}

Callers (open_layer) collect entries into an evidence pack; the pack is
the grounding set for the composed answer, exactly like a builder pack.
"""
from __future__ import annotations

import logging

from dashboard_services.assistant import aliases
from dashboard_services.assistant import evidence as ev
from dashboard_services.assistant.resolve import resolve_player

logger = logging.getLogger(__name__)

_POSITIONS = ("QB", "RB", "WR", "TE")


def _n(value):
    """Evidence number: int when integral, else rounded to 1 decimal."""
    return ev._num(value)


def _resolve(bundle, name):
    """(pid, info, ambiguous_candidates) for a player name."""
    if not name:
        return None, {}, None
    # A name the thread already settled (a clarification pick, carried
    # on the bundle by the service) resolves to the settled player
    # without re-asking.
    forced = (bundle or {}).get("forced_player_ids") or {}
    if forced:
        from dashboard_services.assistant.resolve import _normalize
        pid = forced.get(_normalize(str(name)))
        if pid:
            return str(pid), ev._player_info(str(pid), ev._index()), None
    r = resolve_player(str(name),
                       roster_player_ids=bundle.get("roster_player_ids") or set())
    if r["status"] == "ambiguous":
        return None, {}, r.get("candidates") or []
    if r["status"] != "resolved":
        return None, {}, None
    pid = r["player_id"]
    return pid, ev._player_info(pid, ev._index()), None


def _base(tool: str) -> dict:
    return {"tool": tool, "ok": True, "players": [], "receipts": [], "link": None}


def _modal_link(info: dict) -> dict | None:
    if not info.get("player_id"):
        return None
    return {"player_id": str(info["player_id"]),
            "player_name": str(info.get("name") or ""), "label": "View player"}


def _scalars(row, keys):
    out = {}
    for k in keys:
        v = (row or {}).get(k)
        if isinstance(v, (str, int, float, bool)) or v is None:
            if v is not None:
                out[k] = v
    return out


# -- player tools ---------------------------------------------------------------

def _tool_player_profile(bundle, args) -> dict:
    out = _base("player_profile")
    pid, info, ambiguous = _resolve(bundle, args.get("player"))
    if ambiguous:
        out.update({"ok": False, "ambiguous": ambiguous,
                    "missing": "more than one player matches that name"})
        return out
    if not pid:
        out.update({"ok": False, "missing": "no resolved player"})
        return out
    pack, _tools = ev.build_player(bundle, {}, [pid])
    player = dict(pack.get("player") or info)
    out["player"] = _scalars(player, ("player_id", "name", "team", "position",
                                      "injury_status", "value"))
    role = pack.get("role") or {}
    if role:
        out["role"] = {k: _n(v) for k, v in role.items()
                       if isinstance(v, (int, float))}
    ranks = pack.get("metric_ranks") or {}
    if isinstance(ranks, dict) and ranks:
        rows = []
        for key, entry in ranks.items():
            if not isinstance(entry, dict):
                continue
            rows.append({"metric": key,
                         "label": entry.get("label") or key,
                         "value": _n(entry.get("value")),
                         "rank": entry.get("rank")})
        rows = [r for r in rows if r["rank"] is not None]
        rows.sort(key=lambda r: r["rank"])
        out["top_metric_ranks"] = rows[:4]
    if pack.get("recent_ppg") is not None:
        out["recent_ppg"] = pack["recent_ppg"]
    form = pack.get("recent_form") or []
    if form:
        out["recent_form"] = [_scalars(f, ("week", "points", "matchup"))
                              for f in form]
    owners = ev._owner_by_pid(bundle.get("ctx") or {})
    if owners.get(str(pid)):
        out["owner_team"] = owners[str(pid)]
    news = pack.get("news") or []
    if news:
        out["news"] = [n.get("title") for n in news if n.get("title")][:2]
    out["players"] = [{"player_id": str(pid), "name": player.get("name") or info.get("name")}]
    if out["player"].get("value") is not None:
        out["receipts"].append({"label": f"{out['player'].get('name')} value",
                                "value": str(_n(out["player"]["value"]))})
    if role.get("ppg") is not None:
        out["receipts"].append({"label": "Points per game",
                                "value": str(_n(role["ppg"]))})
    out["link"] = _modal_link(out["player"])
    return out


def _tool_player_stats(bundle, args) -> dict:
    out = _base("player_stats")
    pid, info, ambiguous = _resolve(bundle, args.get("player"))
    if ambiguous:
        out.update({"ok": False, "ambiguous": ambiguous,
                    "missing": "more than one player matches that name"})
        return out
    if not pid:
        out.update({"ok": False, "missing": "no resolved player"})
        return out
    line = ev.player_stat_line(bundle, pid)
    out["player"] = _scalars(info, ("player_id", "name", "team", "position",
                                    "injury_status"))
    totals = {}
    for key in ("games", "fantasy_points") + tuple(aliases.STAT_DEFS):
        if line.get(key) is not None:
            totals[key] = line[key]
    out["totals"] = totals
    log = line.get("game_log") or []
    out["recent_games"] = log[-4:]
    if log:
        pts = [g["points"] for g in log if g.get("points") is not None]
        if pts:
            out["best_week_points"] = max(pts)
            out["worst_week_points"] = min(pts)
    out["players"] = [{"player_id": str(pid), "name": info.get("name")}]
    if totals.get("fantasy_points") is not None:
        out["receipts"].append({"label": "Fantasy points (your scoring)",
                                "value": str(_n(totals["fantasy_points"]))})
    headline = {"QB": "pass_yd", "RB": "rush_yd"}.get(info.get("position"), "rec_yd")
    if totals.get(headline) is not None:
        label = aliases.STAT_DEFS[headline]["label"]
        out["receipts"].append({"label": label, "value": str(_n(totals[headline]))})
    out["link"] = _modal_link(out["player"])
    return out


def _tool_player_metrics(bundle, args) -> dict:
    out = _base("player_metrics")
    pid, info, ambiguous = _resolve(bundle, args.get("player"))
    if ambiguous:
        out.update({"ok": False, "ambiguous": ambiguous,
                    "missing": "more than one player matches that name"})
        return out
    if not pid:
        out.update({"ok": False, "missing": "no resolved player"})
        return out
    out["player"] = _scalars(info, ("player_id", "name", "team", "position"))
    has_pro = ev._caller_has_pro(bundle)
    try:
        pro_keys = set(ev.pro_metric_keys() or set())
        premium_keys = set(ev.premium_metric_keys() or set())
    except Exception:
        pro_keys, premium_keys = set(), set()
    gated_keys = pro_keys | premium_keys
    config = {}
    try:
        config = ev._metric_config()
    except Exception:
        config = {}
    metrics, _tools = ev._metrics_for(pid, bundle.get("season"))
    if metrics:
        out["role"] = {k: _n(v) for k, v in metrics.items()
                       if isinstance(v, (int, float))}
    try:
        from data_building.advanced_metrics import get_player_metric_ranks
        ranks = get_player_metric_ranks(str(pid), season=bundle.get("season"))
    except Exception:
        ranks = None
    rows, gated = [], []
    if isinstance(ranks, dict) and isinstance(ranks.get("ranks"), dict):
        for key, entry in ranks["ranks"].items():
            if not isinstance(entry, dict):
                continue
            label = (config.get(key) or {}).get("label") or entry.get("label") or key
            if key in gated_keys and not has_pro:
                gated.append(str(label))
                continue
            rows.append({"metric": key, "label": str(label),
                         "value": _n(entry.get("value")),
                         "rank": entry.get("rank")})
    rows = [r for r in rows if r["rank"] is not None]
    rows.sort(key=lambda r: r["rank"])
    out["metric_ranks"] = rows[:8]
    if gated:
        # The gate is enforced here: PRO metric names may be listed (the
        # site shows they exist) but their values and ranks never enter
        # the evidence for a caller without PRO.
        out["gated_metrics"] = sorted(set(gated))
    out["players"] = [{"player_id": str(pid), "name": info.get("name")}]
    for r in rows[:3]:
        if r["value"] is not None:
            out["receipts"].append({"label": f"{r['label']} (rank {r['rank']})",
                                    "value": str(r["value"])})
    out["link"] = _modal_link(out["player"])
    return out


# -- board tools ------------------------------------------------------------------

def _tool_season(bundle, args):
    """(season, historical) a board tool was asked for: an explicit
    season arg (a year, or 'last') resolved against the bundle's current
    season. (None, False) is the current-season board. A season at or
    after the current one is not historical here; a future season comes
    back as (season, False) with season > current so the caller can
    decline it instead of silently serving the current board."""
    current = bundle.get("season")
    try:
        current = int(current) if current else None
    except (TypeError, ValueError):
        current = None
    raw = args.get("season")
    if raw is None:
        return None, False
    if isinstance(raw, str) and raw.strip().lower() in (
            "last", "last season", "previous", "previous season"):
        return (current - 1 if current else None), True
    try:
        season = int(raw)
    except (TypeError, ValueError):
        return None, False
    if current is not None and season < current:
        return season, True
    return season, False


def _tool_top_board(bundle, args) -> dict:
    out = _base("top_board")
    phrase = str(args.get("stat") or args.get("metric") or "").strip()
    position = str(args.get("position") or "").strip().upper() or None
    if position not in _POSITIONS:
        position = None
    try:
        top_n = int(args.get("top_n") or 5)
    except (TypeError, ValueError):
        top_n = 5
    top_n = max(1, min(10, top_n))
    season_arg, historical = _tool_season(bundle, args)
    current = bundle.get("season")
    if season_arg is not None and current and season_arg > int(current):
        out.update({"ok": False,
                    "missing": "that season has not been played yet"})
        return out
    board_season = season_arg if season_arg is not None else current
    resolution = aliases.resolve_phrase(phrase) if phrase else None
    config = {}
    try:
        config = ev._metric_config()
    except Exception:
        config = {}
    # A past season's board never borrows the current roster's
    # membership: nobody is "on my roster" in a finished season.
    roster_ids = set() if historical else (
        bundle.get("roster_player_ids") or set())
    if resolution is None and phrase:
        key = ev._resolve_metric_key(phrase.lower().replace(" ", "_"), config)
        if key:
            resolution = {"kind": "metric", "key": key,
                          "label": (config.get(key) or {}).get("label") or key,
                          "family": None, "phrase": phrase, "position": None}
    if resolution is None:
        out.update({"ok": False, "missing": "that stat is not tracked",
                    "nearest": aliases.neighbors_for(phrase)})
        return out
    if resolution.get("position") and not position:
        position = resolution["position"]
    if resolution["kind"] == "stat":
        stat_bundle = bundle
        if historical:
            # The week-stat cache is per-season: a past season's totals
            # read that season's own weeks in full.
            stat_bundle = {**bundle, "season": season_arg,
                           "week": ev._completed_weeks_for(
                               bundle, season_arg) + 1,
                           "roster_player_ids": set()}
        rows, completed = ev.stat_board_rows(stat_bundle, resolution["key"],
                                             position, None,
                                             limit=max(top_n, 10))
        if not rows:
            out.update({"ok": False, "missing": "no recorded games yet for that stat"})
            return out
        board = []
        for i, r in enumerate(rows[:top_n]):
            row = {"rank": i + 1, "player_id": r.get("player_id"),
                   "name": r.get("name"), "team": r.get("team"),
                   "position": r.get("position"), "value": r.get("value"),
                   "games": r.get("games"),
                   "on_my_roster": bool(r.get("player_id") in roster_ids)}
            for fk in aliases.FAMILIES.get(resolution.get("family") or "",
                                           {}).get("stats", ()):
                if r.get(fk) is not None:
                    row[fk] = r[fk]
            board.append(row)
        out.update({"board": board, "label": resolution["label"],
                    "stat_key": resolution["key"], "stat_board": True,
                    "season": board_season,
                    "basis": (f"Final {season_arg} season totals" if historical
                              else f"Season totals through Week {completed}")})
        out["players"] = [{"player_id": r["player_id"], "name": r["name"]}
                          for r in board if r.get("player_id")]
        for r in board[:3]:
            out["receipts"].append({"label": f"{r['rank']}. {r['name']}",
                                    "value": str(_n(r["value"]))})
        out["link"] = {"href": "/metrics", "label": "View on Advanced Metrics"}
        return out
    key = resolution["key"]
    spec = config.get(key) or {}
    gate = ev._gate_for(bundle, key)
    if gate == "gated":
        out.update({"gated": True, "label": spec.get("label") or key,
                    "metric_key": key})
        return out
    if gate == "unavailable":
        out.update({"ok": False, "missing": "metric not available"})
        return out
    rows, _qual = ev._season_board(key, spec, board_season, position)
    board = [ev._board_row(i + 1, r, roster_ids)
             for i, r in enumerate(rows[:top_n]) if isinstance(r, dict)]
    if not board:
        out.update({"ok": False, "missing": "no qualified players for that board"})
        return out
    out.update({"board": board, "label": spec.get("label") or key,
                "metric_key": key, "season": board_season,
                "basis": (f"Final {season_arg} season board, "
                          f"{len(rows)} qualified players" if historical
                          else f"Season board, {len(rows)} qualified players")})
    out["players"] = [{"player_id": r["player_id"], "name": r["name"]}
                      for r in board if r.get("player_id")]
    for r in board[:3]:
        out["receipts"].append({"label": f"{r['rank']}. {r['name']}",
                                "value": str(_n(r["value"]))})
    base = ""
    league = bundle.get("league_id"), bundle.get("platform"), bundle.get("season")
    if all(league):
        base = f"/{league[1]}/{league[2]}/{league[0]}"
    out["link"] = {"href": f"{base}/metrics?metric={key}",
                   "label": "Open Advanced Metrics"}
    return out


def _tool_breadth_board(bundle, args) -> dict:
    """Cross-metric breadth: who places in the top N of the most metrics
    in one family. A thin distillation of the leaderboard breadth pack
    (evidence.build_leaderboard), so the open path and the fixed intent
    can never disagree; PRO-gated metrics are already excluded from the
    pack's count for callers without PRO."""
    out = _base("breadth_board")
    family = str(args.get("family") or "").strip().lower()
    if family not in ("receiving", "rushing", "passing", "fantasy"):
        family = ev._breadth_family_for(family) or "receiving"
    position = str(args.get("position") or "").strip().upper() or None
    if position not in _POSITIONS:
        position = None
    try:
        top_n = int(args.get("top_n") or 10)
    except (TypeError, ValueError):
        top_n = 10
    top_n = max(1, min(25, top_n))
    season_arg, _hist = _tool_season(bundle, args)
    pack, _tools = ev.build_leaderboard(
        bundle, {"board_kind": "breadth", "family": family,
                 "position": position, "top_n": top_n,
                 "season": season_arg}, [])
    if pack.get("available") is False:
        out.update({"ok": False,
                    "missing": pack.get("missing") or "no qualified players"})
        return out
    board = []
    for r in pack.get("board") or []:
        row = {"rank": r.get("rank"), "player_id": r.get("player_id"),
               "name": r.get("name"), "team": r.get("team"),
               "position": r.get("position"),
               "finish_count": r.get("finish_count"),
               "top_finishes": [
                   {"label": f.get("label"), "rank": f.get("rank"),
                    "value": f.get("value")}
                   for f in (r.get("top_finishes") or [])[:6]]}
        board.append(row)
    out.update({"board": board, "label": pack.get("label"),
                "family": pack.get("family"),
                "breadth_top_n": pack.get("breadth_top_n"),
                "metric_count": pack.get("metric_count"),
                "basis": pack.get("qualification_text")})
    out["players"] = [{"player_id": r["player_id"], "name": r["name"]}
                      for r in board if r.get("player_id")]
    n = pack.get("breadth_top_n") or top_n
    for r in board[:3]:
        out["receipts"].append({
            "label": f"{r['rank']}. {r['name']}",
            "value": f"top {n} in {r['finish_count']} metrics"})
    out["link"] = {"href": "/metrics", "label": "View on Advanced Metrics"}
    return out


_DEEP_BOARD_FIELDS = ("games", "ppg", "floor", "ceiling", "finishes",
                      "delta", "baseline_value", "slate_avg_rank",
                      "slate_games", "gap", "second_name", "second_value",
                      "climb", "old_rank", "new_rank", "current_value",
                      "age")


def _tool_deep_board(bundle, args) -> dict:
    """Detailed-question boards: one call into the leaderboard builder
    with a composed kind (finishes / floor / ceiling / value / slate /
    gap / rank_movers / defenses), distilled to the ranked rows, their
    basis, and receipts. PRO gating happens inside the builder, exactly
    as the fixed leaderboard path does it."""
    out = _base("deep_board")
    kind = str(args.get("kind") or "").strip().lower()
    if kind not in ("finishes", "floor", "ceiling", "value", "slate",
                    "gap", "rank_movers", "defenses"):
        out.update({"ok": False, "missing": "unknown board kind"})
        return out
    phrase = str(args.get("stat") or args.get("metric") or "").strip()
    position = str(args.get("position") or "").strip().upper() or None
    if position not in _POSITIONS:
        position = None
    team = str(args.get("team") or "").strip().upper() or None
    try:
        top_n = int(args.get("top_n") or 5)
    except (TypeError, ValueError):
        top_n = 5
    top_n = max(1, min(10, top_n))
    try:
        max_age = int(args.get("max_age") or 0) or None
    except (TypeError, ValueError):
        max_age = None
    pool = str(args.get("pool") or "").strip().lower()
    if pool not in ("waivers", "roster"):
        pool = None
    season_arg, _hist = _tool_season(bundle, args)
    ent = {"board_kind": kind, "top_n": top_n, "position": position,
           "metric_hint": phrase or None, "max_age": max_age, "pool": pool,
           "season": season_arg, "team": team}
    pack, _used = ev.build_leaderboard(bundle, ent, [])
    if pack.get("gated"):
        out.update({"gated": True, "label": pack.get("label"),
                    "metric_key": pack.get("metric_key")})
        return out
    board = []
    for r in pack.get("board") or []:
        row = {"rank": r.get("rank"), "player_id": r.get("player_id"),
               "name": r.get("name"), "team": r.get("team"),
               "position": r.get("position"), "value": r.get("value"),
               "on_my_roster": bool(r.get("on_my_roster"))}
        for fk in _DEEP_BOARD_FIELDS:
            if r.get(fk) is not None:
                row[fk] = r[fk]
        board.append(row)
    if not board:
        entry = {"ok": False,
                 "missing": pack.get("missing") or "no qualified players for that board"}
        if pack.get("nearest"):
            entry["nearest"] = pack["nearest"]
        out.update(entry)
        return out
    out.update({"board": board, "label": pack.get("label"),
                "board_kind": kind,
                "basis": pack.get("qualification_text")})
    if pack.get("focus"):
        # A named subject the board was built around (a team defense):
        # its rank and line ride along even when it sits below the
        # returned rows, so the composer can answer for it directly.
        out["focus"] = pack["focus"]
    out["players"] = [{"player_id": r["player_id"], "name": r["name"]}
                      for r in board if r.get("player_id")]
    for r in board[:3]:
        out["receipts"].append({"label": f"{r['rank']}. {r['name']}",
                                "value": str(_n(r["value"]))})
    out["link"] = {"href": "/metrics", "label": "View on Advanced Metrics"}
    return out


def _tool_metric_definition(bundle, args) -> dict:
    out = _base("metric_definition")
    phrase = str(args.get("metric") or args.get("stat") or "").strip()
    config = {}
    try:
        config = ev._metric_config()
    except Exception:
        config = {}
    key = ev._resolve_metric_key(phrase.lower().replace(" ", "_"), config) if phrase else None
    if not key:
        resolved = aliases.resolve_phrase(phrase)
        if resolved and resolved.get("kind") == "metric":
            key = resolved["key"]
    if not key or key not in config:
        out.update({"ok": False, "missing": "that metric is not tracked",
                    "nearest": aliases.neighbors_for(phrase)})
        return out
    spec = config[key] or {}
    out.update({"metric_key": key, "label": spec.get("label") or key,
                "definition": spec.get("desc") or spec.get("label") or key,
                "positions": spec.get("positions") or []})
    gate = ev._gate_for(bundle, key)
    try:
        out["pro"] = key in (set(ev.pro_metric_keys() or set())
                             | set(ev.premium_metric_keys() or set()))
    except Exception:
        out["pro"] = gate is not None
    if gate == "gated":
        # The definition is glossary text the site shows everyone; the
        # board numbers are what PRO withholds, so no leader is fetched.
        out["gated_board"] = True
        return out
    rows, _qual = ev._season_board(key, spec, bundle.get("season"), None)
    if rows:
        top = rows[0]
        info = ev._player_info(str(top.get("player_id") or top.get("id") or ""),
                               ev._index())
        out["leader"] = {"name": top.get("name") or info.get("name"),
                         "value": _n(top.get("value")
                                     if top.get("value") is not None
                                     else top.get(key))}
        out["receipts"].append({"label": f"{out['leader']['name']} {out['label']}",
                                "value": str(out["leader"]["value"])})
    # Concept families ("missed tackles" is the broken-tackles family):
    # the phrase names a family, so the entry also carries every sibling
    # metric with its own definition and leader; the answer can then
    # enumerate what the site tracks instead of naming one rate.
    family = aliases.metric_family_for(phrase)
    if family and key in family["keys"]:
        out["concept"] = family["blurb"]
        also = []
        for fk in family["keys"]:
            if fk == key or fk not in config:
                continue
            fspec = config[fk] or {}
            entry = {"metric_key": fk, "label": fspec.get("label") or fk,
                     "definition": fspec.get("desc") or fspec.get("label") or fk,
                     "positions": fspec.get("positions") or []}
            if ev._gate_for(bundle, fk) != "gated":
                try:
                    frows, _fq = ev._season_board(fk, fspec,
                                                  bundle.get("season"), None)
                except Exception:
                    frows = []
                if frows:
                    ftop = frows[0]
                    entry["leader"] = {
                        "name": ftop.get("name"),
                        "value": _n(ftop.get("value")
                                    if ftop.get("value") is not None
                                    else ftop.get(fk))}
                    out["receipts"].append(
                        {"label": f"{entry['leader']['name']} {entry['label']}",
                         "value": str(entry["leader"]["value"])})
            also.append(entry)
        if also:
            out["also_tracked"] = also
    out["link"] = {"href": f"/metrics?metric={key}",
                   "label": "Open Advanced Metrics"}
    return out


# -- team and league tools ---------------------------------------------------------

def _team_rows(bundle):
    """[(roster_id, team_name, roster)] for the session league."""
    ctx = bundle.get("ctx") or {}
    try:
        names = ev._team_names(ctx)
    except Exception:
        names = {}
    rows = []
    for roster in ctx.get("rosters") or []:
        if not isinstance(roster, dict):
            continue
        rid = roster.get("roster_id")
        name = None
        if isinstance(names, dict):
            name = names.get(rid) or names.get(str(rid))
        rows.append((rid, name or roster.get("team_name") or f"Team {rid}", roster))
    return rows


def _tool_team_roster(bundle, args) -> dict:
    out = _base("team_roster")
    ctx = bundle.get("ctx") or {}
    if not ctx:
        out.update({"ok": False, "missing": "no league context loaded"})
        return out
    want = str(args.get("team") or "").strip().lower()
    rows = _team_rows(bundle)
    if not rows:
        out.update({"ok": False, "missing": "no rosters in this league"})
        return out
    mine = bundle.get("roster") or {}
    target = None
    if want in ("", "mine", "my team", "my roster", "me"):
        target = next((r for r in rows
                       if str(r[0]) == str(mine.get("roster_id"))), rows[0])
    else:
        target = next((r for r in rows if want in str(r[1]).lower()), None)
        if target is None:
            out.update({"ok": False, "missing": "no team matches that name",
                        "teams": [r[1] for r in rows]})
            return out
    rid, team_name, roster = target
    index = ev._index()
    values = ev._value_rows()
    starters = {str(p) for p in (roster.get("starters") or [])}

    def _entry(pid):
        info = ev._player_info(pid, index)
        return {"player_id": str(pid), "name": info.get("name"),
                "team": info.get("team"), "position": info.get("position"),
                "injury_status": info.get("injury_status"),
                "value": _n(ev._row_value(values.get(str(pid))))}

    players = [_entry(p) for p in (roster.get("players") or [])]
    players = [p for p in players if p["name"]]
    players.sort(key=lambda p: -(p["value"] or 0))
    out["team"] = {"team_name": team_name, "roster_id": rid,
                   "is_mine": str(rid) == str(mine.get("roster_id"))}
    out["starters"] = [p for p in players if p["player_id"] in starters]
    out["bench"] = [p for p in players if p["player_id"] not in starters]
    hurt = [p for p in players
            if str(p.get("injury_status") or "").strip().upper()
            not in ("", "NONE", "HEALTHY", "ACTIVE")]
    if hurt:
        out["injuries"] = [{"name": p["name"],
                            "injury_status": p["injury_status"]} for p in hurt]
    total = sum(p["value"] or 0 for p in players)
    out["roster_value_total"] = _n(total) if players else None
    try:
        pack, _t = ev.build_standings(bundle, {}, [])
        for row in pack.get("teams") or []:
            if str(row.get("team_name")) == str(team_name):
                out["standing"] = _scalars(row, ("seed", "wins", "losses",
                                                 "ties", "pf", "pa"))
                break
    except Exception:
        logger.debug("[assistant] team_roster standings lookup failed",
                     exc_info=True)
    out["players"] = [{"player_id": p["player_id"], "name": p["name"]}
                      for p in players[:8]]
    if players and players[0].get("value") is not None:
        out["receipts"].append({"label": f"{players[0]['name']} value",
                                "value": str(players[0]["value"])})
    if out.get("standing", {}).get("pf") is not None:
        out["receipts"].append({"label": f"{team_name} points for",
                                "value": str(_n(out["standing"]["pf"]))})
    base = ""
    if bundle.get("platform") and bundle.get("season") and bundle.get("league_id"):
        base = f"/{bundle['platform']}/{bundle['season']}/{bundle['league_id']}"
    out["link"] = {"href": f"{base}/teams", "label": "View on Teams"}
    return out


_LEAGUE_KINDS = ("standings", "power_rankings", "playoffs", "schedule")


def _tool_league_lookup(bundle, args) -> dict:
    kind = str(args.get("kind") or "standings").strip().lower()
    if kind not in _LEAGUE_KINDS:
        kind = "standings"
    out = _base("league_lookup")
    out["kind"] = kind
    builder = ev.BUILDERS.get(kind)
    if builder is None:
        out.update({"ok": False, "missing": "that league view is not available"})
        return out
    pack, _tools = builder(bundle, {}, [])
    if pack.get("available") is False:
        out.update({"ok": False, "missing": pack.get("missing")
                    or "that league data is not available"})
        return out
    if isinstance(pack.get("me"), dict):
        out["me"] = {k: _n(v) if isinstance(v, (int, float)) else v
                     for k, v in pack["me"].items()
                     if isinstance(v, (str, int, float, bool))}
    for field in ("leader", "points_leader", "easiest", "hardest"):
        if isinstance(pack.get(field), dict):
            out[field] = {k: _n(v) if isinstance(v, (int, float)) else v
                          for k, v in pack[field].items()
                          if isinstance(v, (str, int, float, bool))}
    table = pack.get("teams") or pack.get("field") or []
    out["table"] = [{k: _n(v) if isinstance(v, (int, float)) else v
                     for k, v in row.items()
                     if isinstance(v, (str, int, float, bool))}
                    for row in table[:5] if isinstance(row, dict)]
    me = out.get("me") or {}
    for field, label in (("seed", "Your seed"), ("rank", "Your power rank"),
                         ("playoff_pct", "Your playoff odds (%)"),
                         ("pf", "Your points for")):
        if me.get(field) is not None:
            out["receipts"].append({"label": label, "value": str(_n(me[field]))})
            break
    leader = out.get("leader") or out.get("points_leader") or {}
    if leader.get("team_name") and (leader.get("pf") is not None
                                    or leader.get("power_score") is not None):
        val = leader.get("pf") if leader.get("pf") is not None \
            else leader.get("power_score")
        out["receipts"].append({"label": f"{leader['team_name']} headline",
                                "value": str(_n(val))})
    page = {"standings": "standings", "power_rankings": "standings",
            "playoffs": "dashboard", "schedule": "schedule"}[kind]
    if bundle.get("platform") and bundle.get("season") and bundle.get("league_id"):
        out["link"] = {"href": f"/{bundle['platform']}/{bundle['season']}"
                               f"/{bundle['league_id']}/{page}",
                       "label": "View in your league"}
    return out


# -- comparison, value, depth ------------------------------------------------------

def _profile_lite(bundle, pid):
    info = ev._player_info(pid, ev._index())
    values = ev._value_rows()
    metrics, _t = ev._metrics_for(pid, bundle.get("season"))
    form = ev._recent_form(bundle, pid, info.get("team"), info.get("position"),
                           limit=4)
    recent_ppg = (round(sum(f["points"] for f in form) / len(form), 1)
                  if form else None)
    return {"player_id": str(pid), "name": info.get("name"),
            "team": info.get("team"), "position": info.get("position"),
            "injury_status": info.get("injury_status"),
            "value": _n(ev._row_value(values.get(str(pid)))),
            "ppg": _n(metrics.get("ppg")),
            "expected_ppg": _n(metrics.get("expected_ppr_per_game")),
            "target_share": metrics.get("target_share"),
            "snap_share": metrics.get("snap_share"),
            "touches_per_game": _n(metrics.get("touches_per_game")),
            "games": metrics.get("games"),
            "recent_ppg": recent_ppg}


def _tool_compare_players(bundle, args) -> dict:
    out = _base("compare_players")
    pid_a, info_a, amb_a = _resolve(bundle, args.get("player_a") or args.get("player"))
    pid_b, info_b, amb_b = _resolve(bundle, args.get("player_b") or args.get("other"))
    ambiguous = amb_a or amb_b
    if ambiguous:
        out.update({"ok": False, "ambiguous": ambiguous,
                    "missing": "more than one player matches that name"})
        return out
    if not pid_a or not pid_b:
        out.update({"ok": False, "missing": "no resolved player"})
        return out
    a, b = _profile_lite(bundle, pid_a), _profile_lite(bundle, pid_b)
    out["players_detail"] = [a, b]
    out["players"] = [{"player_id": a["player_id"], "name": a["name"]},
                      {"player_id": b["player_id"], "name": b["name"]}]
    for p in (a, b):
        if p.get("value") is not None:
            out["receipts"].append({"label": f"{p['name']} value",
                                    "value": str(p["value"])})
        if p.get("ppg") is not None and len(out["receipts"]) < 4:
            out["receipts"].append({"label": f"{p['name']} points per game",
                                    "value": str(p["ppg"])})
    if bundle.get("platform") and bundle.get("season") and bundle.get("league_id"):
        out["link"] = {"href": f"/{bundle['platform']}/{bundle['season']}"
                               f"/{bundle['league_id']}/compare"
                               f"?p1={a['player_id']}&p2={b['player_id']}",
                       "label": "View head-to-head in Compare"}
    else:
        out["link"] = {"href": f"/compare?p1={a['player_id']}&p2={b['player_id']}",
                       "label": "View head-to-head in Compare"}
    return out


def _mover_row(r):
    return {"player_id": str(r.get("player_id") or r.get("id") or ""),
            "name": r.get("name"), "team": r.get("team"),
            "position": r.get("position"),
            "value": _n(r.get("new_value")), "delta": _n(r.get("delta")),
            "baseline_value": _n(r.get("old_value"))}


def _tool_value_lookup(bundle, args) -> dict:
    out = _base("value_lookup")
    payload = {}
    try:
        payload = ev.get_top_movers(days=7, limit=60, min_baseline_value=10,
                                    min_current_value=20.0) or {}
    except Exception:
        logger.debug("[assistant] value lookup movers failed", exc_info=True)
        payload = {}
    if isinstance(payload, list):
        risers = [r for r in payload if isinstance(r, dict)
                  and float(r.get("delta") or 0) > 0]
        fallers = [r for r in payload if isinstance(r, dict)
                   and float(r.get("delta") or 0) < 0]
    else:
        risers = payload.get("risers") or []
        fallers = payload.get("fallers") or []
    name = str(args.get("player") or "").strip()
    if name:
        pid, info, ambiguous = _resolve(bundle, name)
        if ambiguous:
            out.update({"ok": False, "ambiguous": ambiguous,
                        "missing": "more than one player matches that name"})
            return out
        if not pid:
            out.update({"ok": False, "missing": "no resolved player"})
            return out
        values = ev._value_rows()
        out["player"] = {"player_id": str(pid), "name": info.get("name"),
                         "team": info.get("team"),
                         "position": info.get("position"),
                         "value": _n(ev._row_value(values.get(str(pid))))}
        move = None
        for r in list(risers) + list(fallers):
            if str(r.get("player_id") or r.get("id") or "") == str(pid):
                move = _mover_row(r)
                break
        out["move_7d"] = move
        out["players"] = [{"player_id": str(pid), "name": info.get("name")}]
        if out["player"].get("value") is not None:
            out["receipts"].append({"label": f"{info.get('name')} value",
                                    "value": str(out["player"]["value"])})
        if move and move.get("delta") is not None:
            out["receipts"].append({"label": "7-day value move",
                                    "value": str(move["delta"])})
        out["link"] = _modal_link(out["player"]) or {
            "href": "/top-movers", "label": "View on Top Movers"}
        return out
    out["risers"] = [_mover_row(r) for r in risers[:5] if isinstance(r, dict)]
    out["fallers"] = [_mover_row(r) for r in fallers[:5] if isinstance(r, dict)]
    out["players"] = [{"player_id": r["player_id"], "name": r["name"]}
                      for r in (out["risers"] + out["fallers"])
                      if r.get("player_id")]
    for r in (out["risers"][:1] + out["fallers"][:1]):
        if r.get("delta") is not None:
            out["receipts"].append({"label": f"{r['name']} 7-day move",
                                    "value": str(r["delta"])})
    out["link"] = {"href": "/top-movers", "label": "View on Top Movers"}
    return out


def _tool_next_up(bundle, args) -> dict:
    out = _base("next_up")
    pid, info, ambiguous = _resolve(bundle, args.get("player"))
    if ambiguous:
        out.update({"ok": False, "ambiguous": ambiguous,
                    "missing": "more than one player matches that name"})
        return out
    if not pid:
        out.update({"ok": False, "missing": "no resolved player"})
        return out
    pack, _tools = ev.build_handcuff(bundle, {}, [pid])
    if pack.get("available") is False:
        out.update({"ok": False, "missing": pack.get("missing")
                    or "no one behind him on the depth chart"})
        return out
    out["depth_basis"] = pack.get("depth_basis")
    for field in ("subject", "next_up", "also_next"):
        if isinstance(pack.get(field), dict):
            out[field] = _scalars(pack[field], ("player_id", "name", "team",
                                                "position", "value",
                                                "location", "owner_team"))
    mine = pack.get("my_options") or []
    if mine:
        out["my_options"] = [_scalars(m, ("player_id", "name", "value"))
                             for m in mine[:3] if isinstance(m, dict)]
    named = [out.get("subject"), out.get("next_up")]
    out["players"] = [{"player_id": str(p["player_id"]), "name": p["name"]}
                      for p in named if p and p.get("player_id")]
    nxt = out.get("next_up") or {}
    if nxt.get("value") is not None:
        out["receipts"].append({"label": f"{nxt.get('name')} value",
                                "value": str(_n(nxt["value"]))})
    if (out.get("next_up") or {}).get("location") == "waivers":
        out["link"] = {"href": "/waivers", "label": "View on Waivers"}
    else:
        out["link"] = _modal_link(out.get("next_up") or out.get("subject") or {})
    return out


# -- live + draft tools ------------------------------------------------------------

def _tool_live_state(bundle, args) -> dict:
    out = _base("live_state")
    entities = {}
    if args.get("team"):
        entities["team_names"] = [str(args["team"])]
    pack, _tools = ev.build_live(bundle, entities, [])
    if pack.get("available") is False:
        out.update({"ok": False,
                    "missing": pack.get("missing") or "no live data right now"})
        return out
    mu = pack.get("matchup") or {}
    a, b = mu.get("team_a") or {}, mu.get("team_b") or {}
    side_keys = ("team_name", "current_points", "projected", "win_prob_pct")
    out["state"] = pack.get("state")
    out["matchup"] = {"team_a": _scalars(a, side_keys),
                      "team_b": _scalars(b, side_keys),
                      "is_mine": bool(mu.get("is_mine"))}
    top_lines = sorted((a.get("players") or []) + (b.get("players") or []),
                       key=lambda p: -(p.get("points") or 0))[:4]
    out["top_scorers"] = [_scalars(p, ("name", "position", "points", "status"))
                          for p in top_lines]
    need = pack.get("need") or {}
    if need:
        out["need"] = {"deficit": need.get("deficit"), "lead": need.get("lead"),
                       "my_remaining": need.get("my_remaining"),
                       "opp_remaining": need.get("opp_remaining")}
    if a.get("current_points") is not None:
        out["receipts"].append({"label": f"{a.get('team_name')} points so far",
                                "value": str(_n(a["current_points"]))})
    if b.get("current_points") is not None:
        out["receipts"].append({"label": f"{b.get('team_name')} points so far",
                                "value": str(_n(b["current_points"]))})
    if bundle.get("platform") and bundle.get("season") and bundle.get("league_id"):
        out["link"] = {"href": f"/{bundle['platform']}/{bundle['season']}"
                               f"/{bundle['league_id']}/scorezone",
                       "label": "Open ScoreZone"}
    return out


def _tool_draft_state(bundle, args) -> dict:
    out = _base("draft_state")
    entities = {}
    if args.get("pick"):
        entities["draft_slot"] = str(args["pick"])
    pack, _tools = ev.build_draft(bundle, entities, [])
    if pack.get("available") is False:
        out.update({"ok": False,
                    "missing": pack.get("missing") or "no draft found for this league"})
        return out
    out["status"] = pack.get("status")
    out["draft_kind"] = pack.get("draft_kind")
    out["picks_made"] = pack.get("picks_made")
    cur = pack.get("current_pick") or {}
    if cur:
        out["on_the_clock"] = _scalars(cur, ("team_name", "pick_no", "round"))
    nxt = pack.get("my_next_pick") or {}
    if nxt:
        out["my_next_pick"] = _scalars(nxt, ("pick_no", "round"))
    best = pack.get("best_available") or []
    out["best_available"] = [_scalars(r, ("name", "position", "team", "value"))
                             for r in best[:5]]
    out["players"] = [{"player_id": str(r.get("player_id")), "name": r.get("name")}
                      for r in best[:5] if r.get("player_id")]
    needs = [n for n in (pack.get("my_needs") or []) if n.get("open_slots")]
    if needs:
        out["my_needs"] = needs
    prep = pack.get("prep") or {}
    if prep:
        out["prep"] = prep
    if best and best[0].get("value") is not None:
        out["receipts"].append({"label": f"{best[0].get('name')} value",
                                "value": str(_n(best[0]["value"]))})
    if bundle.get("platform") and bundle.get("season") and bundle.get("league_id"):
        out["link"] = {"href": f"/{bundle['platform']}/{bundle['season']}"
                               f"/{bundle['league_id']}/draft",
                       "label": "Open Draft Room"}
    return out


def _tool_nfl_schedule(bundle, args) -> dict:
    out = _base("nfl_schedule")
    entities = {}
    if args.get("day"):
        entities["day_hint"] = str(args["day"])
    if args.get("team"):
        entities["team"] = str(args["team"])
    pack, _tools = ev.build_nfl_schedule(bundle, entities, [])
    if pack.get("available") is False:
        out.update({"ok": False,
                    "missing": pack.get("missing") or "the NFL schedule is not loaded"})
        return out
    out["scope"] = pack.get("scope")
    game_keys = ("away_name", "home_name", "date_label", "kickoff_label",
                 "network", "status")
    out["games"] = [_scalars(g, game_keys)
                    for g in (pack.get("games") or [])[:8]]
    if pack.get("team"):
        out["team"] = pack["team"]
        out["bye"] = bool(pack.get("bye"))
    nxt = pack.get("next_game") or {}
    if nxt:
        out["next_game"] = _scalars(nxt, game_keys)
    for g in (pack.get("games") or [])[:4]:
        out["receipts"].append({
            "label": f"{g.get('away_name')} at {g.get('home_name')} kickoff",
            "value": f"{g.get('date_label')}, {g.get('kickoff_label')}"})
    if bundle.get("platform") and bundle.get("season") and bundle.get("league_id"):
        out["link"] = {"href": f"/{bundle['platform']}/{bundle['season']}"
                               f"/{bundle['league_id']}/scorezone",
                       "label": "Open ScoreZone"}
    return out


# -- registry ------------------------------------------------------------------------

TOOL_FUNCS = {
    "player_profile": _tool_player_profile,
    "player_stats": _tool_player_stats,
    "player_metrics": _tool_player_metrics,
    "top_board": _tool_top_board,
    "breadth_board": _tool_breadth_board,
    "deep_board": _tool_deep_board,
    "team_roster": _tool_team_roster,
    "league_lookup": _tool_league_lookup,
    "compare_players": _tool_compare_players,
    "value_lookup": _tool_value_lookup,
    "metric_definition": _tool_metric_definition,
    "next_up": _tool_next_up,
    "live_state": _tool_live_state,
    "draft_state": _tool_draft_state,
    "nfl_schedule": _tool_nfl_schedule,
}

# The planner-facing catalog: name, what it fetches, its arguments.
TOOL_CATALOG = [
    {"name": "player_profile",
     "description": "One player: identity, NFL team, position, trade value, role line (PPG, target share), top metric ranks, recent form, owning fantasy team, latest headlines.",
     "args": {"player": "player name"}},
    {"name": "player_stats",
     "description": "One player's season counting stats (receiving/rushing/passing totals by position) and fantasy points under the league's scoring, plus his recent game log.",
     "args": {"player": "player name"}},
    {"name": "player_metrics",
     "description": "One player's advanced metrics and his rank in each. PRO metrics are withheld for callers without PRO and reported as gated names only.",
     "args": {"player": "player name"}},
    {"name": "top_board",
     "description": "A ranked top-N board by any tracked metric or raw stat (receiving yards, targets, carries, passing yards, fantasy points, target share, any Advanced Metrics stat). Counting-stat phrases resolve to season totals; per-game phrases to the per-game board.",
     "args": {"stat": "the stat or metric phrase", "position": "QB/RB/WR/TE or null",
              "top_n": "how many rows (default 5)",
              "season": "a season year like 2024 when the question names one, else null"}},
    {"name": "breadth_board",
     "description": "Which players place in the top N of the MOST metrics in one family (receiving, rushing, passing, or overall fantasy): counts each player's top-N finishes across the family's Advanced Metrics boards and ranks by that count, with each player's finishes listed. Use for 'top 10 in the most receiving metrics', 'most top-10 finishes', 'ranked in the most categories'.",
     "args": {"family": "receiving | rushing | passing | fantasy",
              "position": "QB/RB/WR/TE or null",
              "top_n": "the N in top N (default 10)"}},
    {"name": "deep_board",
     "description": "A composed analytical board: kind finishes (most top-12 weekly finishes at a position), floor (highest worst week), ceiling (highest best week), value (ranked by trade value, optional max_age or pool filter), slate (a metric's leaders re-ranked by easiest upcoming schedule), gap (who beats his own team's No. 2 by the most in a metric), rank_movers (biggest positional-rank climbers by value over the last month), defenses (team defenses ranked by season fantasy points; team is the defense's NFL abbreviation like DEN when one defense is named).",
     "args": {"kind": "finishes | floor | ceiling | value | slate | gap | rank_movers | defenses",
              "stat": "the metric phrase for slate/gap kinds, else null",
              "position": "QB/RB/WR/TE or null",
              "team": "NFL team abbreviation for the defenses kind, else null",
              "top_n": "how many rows (default 5)",
              "max_age": "age ceiling or null",
              "pool": "waivers | roster | null"}},
    {"name": "team_roster",
     "description": "A fantasy team's roster in this league: starters and bench with values, injuries, roster value total, and its standing line. team is a team name, or 'mine' for the user's own team.",
     "args": {"team": "team name or 'mine'"}},
    {"name": "league_lookup",
     "description": "A league-wide view: kind standings (records, points for, seeds), power_rankings, playoffs (odds and the field), or schedule (remaining strength).",
     "args": {"kind": "standings | power_rankings | playoffs | schedule"}},
    {"name": "compare_players",
     "description": "Two players side by side: value, PPG vs expected, target share, snap share, recent form.",
     "args": {"player_a": "first player name", "player_b": "second player name"}},
    {"name": "value_lookup",
     "description": "Trade values and 7-day value movement. With a player: his value and his 7-day move. Without: the top risers and fallers in value this week.",
     "args": {"player": "player name or null"}},
    {"name": "metric_definition",
     "description": "What a metric or stat means (from the Advanced Metrics catalog), its label, and who currently leads it when the board is not PRO-gated.",
     "args": {"metric": "the metric name or phrase"}},
    {"name": "next_up",
     "description": "Who is next up behind a player on his NFL team's depth chart (his handcuff), where that player is rostered, and what the user already holds at the position.",
     "args": {"player": "player name"}},
    {"name": "live_state",
     "description": "Live state of this week's games in the user's league: current scores, win probability, top scorers so far, and what the user still needs (remaining players on each side).",
     "args": {"team": "a team name, or null for the user's own game"}},
    {"name": "draft_state",
     "description": "The league's draft: status, who is on the clock, the best players still available by site value, the user's next pick and positional needs, and for a named pick like 1.03 a prep target slotted to that pick (the highest-ranked rookie presumed still available there, assuming the board goes in order, plus the next names behind him and the headliner above the slot who is worth taking if he falls).",
     "args": {"pick": "a draft slot like '1.03', or null"}},
    {"name": "nfl_schedule",
     "description": "The real NFL schedule: which NFL teams play tonight, today, tomorrow, on a named weekday, or this week; when a named NFL team plays; kickoff times in Eastern and the TV network when the schedule carries one. Use for NFL fixture questions, never for the user's fantasy matchup.",
     "args": {"day": "tonight | today | tomorrow | a lowercase weekday | this_week, or null",
              "team": "an NFL team abbreviation like KC, or null"}},
]


def run_tool(name: str, args: dict, bundle: dict) -> dict:
    """Execute one tool and return its evidence entry. Never raises."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"tool": str(name), "ok": False, "missing": "unknown tool",
                "players": [], "receipts": [], "link": None}
    try:
        entry = fn(bundle, args or {})
    except Exception:
        logger.debug("[assistant] tool %s failed", name, exc_info=True)
        return {"tool": name, "ok": False, "missing": "that data is not available",
                "players": [], "receipts": [], "link": None}
    entry.setdefault("tool", name)
    entry.setdefault("ok", True)
    entry.setdefault("players", [])
    entry.setdefault("receipts", [])
    entry.setdefault("link", None)
    return entry
