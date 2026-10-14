"""Evidence pack builders, one per core intent.

Every builder returns (pack, tools_called). Packs contain only fields an
answer may cite, plus as_of info where the source carries it. Builders never
raise: a failing source degrades to a smaller pack (or available=false),
never to invented numbers.

Reality notes vs the workshop tool map (verified in this repo):
- utils/start_sit.py exposes compute_start_score(proj_pts, ...) - a scoring
  function, not a per-player verdict entry point; the site verdict path lives
  in the monolithic start/sit view in app.py. So start_sit here uses
  compute_start_score when a projection is present on the value/details data,
  and otherwise falls back to projections + advanced-metric role data (see
  build_start_sit). No score is invented when no projection exists.
- utils/trade_value.py has no public defs at module level in this checkout
  (values come from get_model_value_table_cached / player_value_history).
- utils/lineup_issues.py is a shim re-exporting utils.lineups.find_lineup_issues.
- The waiver candidates view (api_waiver_candidates in routes/waiver_bp.py)
  is monolithic; waivers here are built from its primitives: the model value
  table, ctx free agents (league players minus rostered), the positional gate
  from the workshop spec, and load_do_not_drop from routes.waiver_bp.
- Player game logs have no single importable builder outside the
  player-details view (api_player_details in app.py is a view function), so
  the player pack is composed from get_player_metrics /
  get_player_metric_ranks (data_building.advanced_metrics, the functions
  behind /api/player-advanced-metrics and /api/player-metric-ranks),
  dashboard_services.news.get_player_news, and value history.
- Start/sit schedule/matchup evidence mirrors the site's start/sit view
  (_build_start_sit_options in app.py): weekly projections come from
  utils.utils.load_week_projection scored by
  utils.fantasy_scoring.weekly_projection_points under the league's own
  scoring settings; the opponent comes from utils.utils.load_week_sched;
  the defense rank comes from app._matchup_rank_table (rank 1 = easiest).
  The view's live Vegas/weather conditions are request-scoped there and
  are not reproduced here; compute_start_score treats them as optional.
- Recent form (game logs) is built from the same Sleeper week-stat cache
  the player modal reads: utils.utils.load_sleeper_week_stats scored by
  utils.projections.week_stat_points under league settings.
- Trade partner fit and balancing use the same machinery as the trade
  targets view in app.py: per-roster values by position, league position
  ranks from utils.trade.weighted_pos_strength, thresholds/floors from
  utils.trade.derive_league_thresholds, needs/surplus from
  utils.trade.classify_position_needs / detect_surplus_positions.
- Waiver trending adds come from routes.waiver_bp._sleeper_trending_adds
  (Sleeper's league-wide most-added feed). FAAB guidance uses
  utils.waiver_score.faab_recommendation over a utils.waiver_score
  .waiver_pickup_score computed from the candidate's own value and
  rest-of-season PPG, gated on routes.waiver_bp._waiver_faab_context
  confirming the league actually uses FAAB.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# -- lazy shims (monkeypatchable in tests) ------------------------------------

def get_model_value_table_cached(*args, **kwargs):
    from app import get_model_value_table_cached as _fn
    return _fn(*args, **kwargs)


def get_players_index_global(*args, **kwargs):
    from app import get_players_index_global as _fn
    return _fn(*args, **kwargs)


def load_week_projection(*args, **kwargs):
    from utils.utils import load_week_projection as _fn
    return _fn(*args, **kwargs)


def load_week_sched(*args, **kwargs):
    from utils.utils import load_week_sched as _fn
    return _fn(*args, **kwargs)


def load_sleeper_week_stats(*args, **kwargs):
    from utils.utils import load_sleeper_week_stats as _fn
    return _fn(*args, **kwargs)


def matchup_rank_table(*args, **kwargs):
    from app import _matchup_rank_table as _fn
    return _fn(*args, **kwargs)


def sleeper_trending_adds(*args, **kwargs):
    from routes.waiver_bp import _sleeper_trending_adds as _fn
    return _fn(*args, **kwargs)


def waiver_faab_context(*args, **kwargs):
    from routes.waiver_bp import _waiver_faab_context as _fn
    return _fn(*args, **kwargs)


def _index():
    try:
        return get_players_index_global() or {}
    except Exception:
        return {}


def _player_info(pid, index=None):
    index = index if index is not None else _index()
    p = {}
    if isinstance(index, dict):
        p = index.get(str(pid)) or index.get(pid) or {}
    if not isinstance(p, dict):
        p = {}
    name = p.get("full_name") or p.get("name") or ""
    if not name:
        name = ((p.get("first_name") or "") + " " + (p.get("last_name") or "")).strip()
    return {"player_id": str(pid), "name": name or str(pid),
            "team": p.get("team") or "", "position": p.get("position") or p.get("pos") or "",
            "injury_status": p.get("injury_status") or p.get("status") or None}


def _value_rows():
    try:
        rows = get_model_value_table_cached() or []
    except Exception:
        return {}
    out = {}
    for r in rows:
        if isinstance(r, dict) and (r.get("id") or r.get("player_id")):
            out[str(r.get("id") or r.get("player_id"))] = r
    return out


def _row_value(row) -> float | None:
    if not row:
        return None
    for k in ("value", "sf_value", "redraft_value", "trade_value"):
        v = row.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                continue
    return None


def _metrics_for(pid, season=None):
    tools = []
    try:
        from data_building.advanced_metrics import get_player_metrics_by_season, get_player_metrics
        m = None
        if season:
            m = get_player_metrics_by_season(str(pid), int(season))
        if not m:
            m = get_player_metrics(str(pid))
        if isinstance(m, dict):
            keep = {k: m[k] for k in (
                "snap_share", "target_share", "targets_per_game", "touches_per_game",
                "opportunity_share", "expected_ppr_per_game", "ppg", "games",
                "team", "position", "as_of_date",
            ) if m.get(k) is not None}
            return keep, ["advanced_metrics.get_player_metrics"]
    except Exception:
        logger.debug("[assistant] metrics lookup failed", exc_info=True)
    return {}, tools


def _base_pack(intent, bundle):
    return {
        "type": intent,
        "available": True,
        "league": {
            "platform": bundle.get("platform"), "league_id": bundle.get("league_id"),
            "season": bundle.get("season"), "league_name": bundle.get("league_name"),
            "team_name": bundle.get("team_name"),
        },
        "week": bundle.get("week"),
        "as_of": {"week": bundle.get("week"), "season": bundle.get("season")},
    }


def _scoring(bundle) -> dict:
    ctx = bundle.get("ctx") or {}
    return ctx.get("raw_scoring_settings") or ctx.get("scoring_settings") or {}


def _schedule_games(season, week):
    """The week's NFL schedule as a games list ([] when not loaded)."""
    if not season or not week:
        return []
    try:
        games = load_week_sched(int(season), int(week)) or []
    except Exception:
        return []
    return [g for g in games if isinstance(g, dict)] if isinstance(games, (list, tuple)) else []


def _opponent_for(team, games):
    """(opponent, label, on_bye) for an NFL team in a loaded schedule.

    Label follows the modal convention: "vs LV" at home, "@ LV" away. A team
    absent from a LOADED schedule is on bye; with no schedule loaded, nothing
    is claimed (opponent None, on_bye False).
    """
    if not team or not games:
        return None, None, False
    t = str(team).upper()
    for g in games:
        home = str(g.get("home") or "").upper()
        away = str(g.get("away") or "").upper()
        if t == home and away:
            return away, f"vs {away}", False
        if t == away and home:
            return home, f"@ {home}", False
    return None, "BYE", True


def _recent_form(bundle, pid, team, position, limit=4):
    """Game-log form: the player's last played weeks, oldest first.

    Points are scored with the league's settings via the site's own
    week_stat_points over the Sleeper week-stat cache (the same files the
    player modal's game logs read). Weeks with no stat line are skipped:
    the player did not record a game that week (bye, inactive, or DNP).
    """
    season, week = bundle.get("season"), bundle.get("week")
    if not season or not week:
        return []
    scoring = _scoring(bundle)
    entries = []
    try:
        from utils.projections import week_stat_points
        for w in range(int(week) - 1, 0, -1):
            if len(entries) >= limit:
                break
            try:
                stats = load_sleeper_week_stats(int(season), w) or {}
            except Exception:
                continue
            line = stats.get(str(pid)) if isinstance(stats, dict) else None
            if not isinstance(line, dict):
                continue
            pts = week_stat_points(line, scoring, position or "")
            opp, label, _bye = _opponent_for(team, _schedule_games(season, w))
            entries.append({"week": w, "points": round(float(pts), 1),
                            "opponent": opp, "matchup": label})
    except Exception:
        logger.debug("[assistant] recent form failed", exc_info=True)
        return []
    entries.reverse()
    return entries


def _upcoming_schedule(bundle, team, count=4):
    """The team's next games from the current week: [{week, opponent, matchup}].

    A week where the schedule is loaded and the team is absent is a BYE
    entry. Scanning stops at the first week whose schedule is not loaded.
    """
    season, week = bundle.get("season"), bundle.get("week")
    if not season or not week or not team:
        return []
    out = []
    for w in range(int(week), 19):
        if len(out) >= count:
            break
        games = _schedule_games(season, w)
        if not games:
            break
        opp, label, on_bye = _opponent_for(team, games)
        out.append({"week": w, "opponent": opp, "matchup": label, "bye": bool(on_bye)})
    return out


# -- start / sit ---------------------------------------------------------------

def build_start_sit(bundle, entities, player_ids) -> tuple[dict, list[str]]:
    pack = _base_pack("start_sit", bundle)
    tools: list[str] = []
    values = _value_rows()
    if values:
        tools.append("model_value_table")
    index = _index()
    season, week = bundle.get("season"), bundle.get("week")
    scoring = _scoring(bundle)
    # The site's start/sit evidence: this week's projection map scored under
    # league settings, the week's schedule for opponents/byes, and the
    # matchup rank table (rank 1 = easiest defense for the position).
    week_map = None
    if season and week:
        try:
            week_map = load_week_projection(int(season), int(week))
            if week_map:
                tools.append("utils.load_week_projection")
        except Exception:
            week_map = None
    games = _schedule_games(season, week)
    if games:
        tools.append("utils.load_week_sched")
    rank_tables: dict = {}

    def _def_rank(position, opponent):
        if not position or not opponent or not season:
            return None, None
        if position not in rank_tables:
            try:
                table = matchup_rank_table(int(season), position)
                rank_tables[position] = (table[0] or {}, table[1] or 32)
                tools.append("app._matchup_rank_table")
            except Exception:
                rank_tables[position] = (None, None)
        rmap, total = rank_tables[position]
        if not rmap:
            return None, None
        rank = rmap.get(opponent) or rmap.get(str(opponent).upper())
        return (rank, total) if rank is not None else (None, None)

    players = []
    for pid in player_ids[:3]:
        info = _player_info(pid, index)
        entry = dict(info)
        val = _row_value(values.get(str(pid)))
        if val is not None:
            entry["value"] = val
        metrics, mtools = _metrics_for(pid, bundle.get("season"))
        tools.extend(mtools)
        if metrics:
            entry["role"] = metrics
            if metrics.get("as_of_date"):
                pack["as_of"]["metrics_as_of"] = metrics["as_of_date"]
        # Projection from the week's projection map (the same source the
        # start/sit page scores), not from the value row.
        proj = None
        if week_map:
            try:
                from utils.fantasy_scoring import weekly_projection_points
                pts = weekly_projection_points(week_map, str(pid), scoring, entry.get("position") or "")
                if pts is not None:
                    proj = round(float(pts), 1)
            except Exception:
                proj = None
        opponent, matchup_label, on_bye = _opponent_for(entry.get("team"), games)
        if opponent:
            entry["opponent"] = opponent
            entry["matchup"] = matchup_label
        if on_bye:
            entry["on_bye"] = True
        def_rank = def_total = None
        if opponent and not on_bye:
            def_rank, def_total = _def_rank(entry.get("position"), opponent)
            if def_rank is not None:
                entry["def_rank_vs_position"] = def_rank
                entry["def_rank_total"] = def_total
        form = _recent_form(bundle, pid, entry.get("team"), entry.get("position"), limit=4)
        recent_ppg = None
        if form:
            entry["recent_form"] = form
            tools.append("utils.load_sleeper_week_stats")
            recent_ppg = round(sum(f["points"] for f in form) / len(form), 1)
            entry["recent_ppg"] = recent_ppg
        if proj is not None:
            entry["projection"] = proj
            try:
                from utils.start_sit import compute_start_score
                score, factors, _demotion = compute_start_score(
                    proj, on_bye=bool(on_bye),
                    recent_ppg=recent_ppg or 0.0,
                    season_ppg=float(metrics.get("ppg") or 0.0) if metrics else 0.0,
                    def_rank=def_rank, def_total=def_total,
                    injury_status=entry.get("injury_status"),
                    position=entry.get("position") or None,
                )
                entry["start_score"] = round(float(score), 2)
                tools.append("utils.start_sit.compute_start_score")
            except Exception:
                logger.debug("[assistant] start score failed", exc_info=True)
        players.append(entry)
    pack["players"] = players
    scored = [p for p in players if p.get("start_score") is not None]
    key = "start_score" if scored else "value"
    ranked = sorted(players, key=lambda p: (p.get(key) is not None, p.get(key) or 0), reverse=True)
    if ranked:
        pack["pick_player_id"] = ranked[0]["player_id"]
    if len(ranked) >= 2 and ranked[0].get(key) is not None and ranked[1].get(key) is not None:
        gap = abs(float(ranked[0][key]) - float(ranked[1][key]))
        base = max(abs(float(ranked[0][key])), 1.0)
        pack["margin"] = {"gap": round(gap, 2), "basis": key, "thin": bool(gap / base < 0.10)}
    else:
        pack["margin"] = {"thin": False}
    if not players:
        pack["available"] = False
        pack["missing"] = "no resolved players"
    return pack, tools


# -- trade evaluation ------------------------------------------------------------

def build_trade_eval(bundle, entities, resolved_ids) -> tuple[dict, list[str]]:
    pack = _base_pack("trade_eval", bundle)
    tools: list[str] = ["model_value_table"]
    values = _value_rows()
    index = _index()
    give_ids = (entities or {}).get("give_player_ids") or []
    recv_ids = (entities or {}).get("receive_player_ids") or []
    if not give_ids and not recv_ids:
        # No explicit sides: first resolved player is the give side, rest receive.
        ids = list(resolved_ids or [])
        give_ids, recv_ids = ids[:1], ids[1:]
    def _side(ids):
        out, total, unpriced = [], 0.0, []
        for pid in ids or []:
            info = _player_info(pid, index)
            v = _row_value(values.get(str(pid)))
            if v is None:
                unpriced.append(info["name"])
            else:
                total += v
            out.append({**info, "value": v})
        return out, round(total, 1), unpriced
    give, give_total, give_un = _side(give_ids)
    recv, recv_total, recv_un = _side(recv_ids)
    pack["give"] = {"players": give, "total_value": give_total}
    pack["receive"] = {"players": recv, "total_value": recv_total}
    pack["fairness_gap"] = round(recv_total - give_total, 1)
    pack["unpriced"] = give_un + recv_un
    # Picks: value table rows with position PICK carry values in this repo
    # (get_model_value_table_cached explicitly keeps PICK rows). Anything not
    # found there is marked unpriced, never invented.
    picks = (entities or {}).get("picks") or []
    if picks:
        pick_rows = []
        for pk in picks:
            found = None
            for pid, row in values.items():
                if str(row.get("position") or "").upper() == "PICK" and (str(pk).lower() in str(row.get("name") or "").lower() or str(pk) == str(pid)):
                    found = _row_value(row)
                    break
            pick_rows.append({"pick": pk, "value": found, "priced": found is not None})
            if found is None and pk not in pack["unpriced"]:
                pack["unpriced"].append(str(pk))
        pack["picks"] = pick_rows
        tools.append("model_value_table.picks")
    # Roster fit, partner fit, and a balancing asset, all from the trade
    # targets machinery (utils.trade): league position ranks from weighted
    # position strength, thresholds/floors derived from the league's own
    # roster settings, needs/surplus classified per roster.
    ctx = bundle.get("ctx") or {}
    roster = bundle.get("roster") or {}
    rosters = [r for r in (ctx.get("rosters") or []) if isinstance(r, dict)]
    if rosters:
        try:
            fit = _league_fit(ctx, rosters, values, index)
        except Exception:
            logger.debug("[assistant] trade fit failed", exc_info=True)
            fit = None
        if fit:
            tools.append("utils.trade.roster_fit")
            viewer_rid = str(roster.get("roster_id") or "")
            if viewer_rid in fit["by_roster"]:
                mine = fit["by_roster"][viewer_rid]
                pack["roster_fit"] = {
                    "surplus_positions": mine["surplus"],
                    "needed_positions": mine["needs"],
                }
            partner = _identify_partner(bundle, entities, fit, give_ids, recv_ids)
            if partner is not None:
                pack["partner"] = {
                    "team_name": partner["team_name"],
                    "needed_positions": partner["needs"],
                    "surplus_positions": partner["surplus"],
                }
                tools.append("utils.trade.partner_fit")
                balancing = _balancing_asset(pack["fairness_gap"], partner, fit,
                                             values, index, give_ids, recv_ids, viewer_rid)
                if balancing:
                    pack["balancing_asset"] = balancing
                    tools.append("utils.trade.balancing_asset")
    base = max(give_total, recv_total, 1.0)
    pack["margin"] = {"thin": bool(abs(pack["fairness_gap"]) / base < 0.10)}
    return pack, tools


def _league_fit(ctx, rosters, values, index):
    """Per-roster needs/surplus plus raw inputs, mirroring the trade targets
    view: position ranks across the league from weighted_pos_strength,
    thresholds/floors from derive_league_thresholds on the league settings."""
    from utils.trade import (
        POSITIONS, classify_position_needs, derive_league_thresholds,
        detect_surplus_positions, weighted_pos_strength,
    )
    roster_positions = [str(s) for s in (ctx.get("roster_positions") or [])]
    slot_counts: dict[str, int] = {}
    for slot in roster_positions:
        slot_counts[slot] = slot_counts.get(slot, 0) + 1
    is_sf = any("SUPER" in slot.upper() for slot in roster_positions)
    num_teams = len(rosters)
    thresholds, floors = derive_league_thresholds(roster_positions, num_teams, is_sf=is_sf)
    users = {str(u.get("user_id")): u for u in (ctx.get("users") or []) if isinstance(u, dict)}
    vals_by_roster: dict[str, dict[str, list[float]]] = {}
    strength: dict[str, dict[str, float]] = {}
    team_names: dict[str, str] = {}
    for r in rosters:
        rid = str(r.get("roster_id"))
        vals: dict[str, list[float]] = {pos: [] for pos in POSITIONS}
        for pid in r.get("players") or []:
            info = _player_info(pid, index)
            v = _row_value(values.get(str(pid)))
            if info.get("position") in POSITIONS and v is not None:
                vals[info["position"]].append(v)
        vals_by_roster[rid] = vals
        strength[rid] = {pos: weighted_pos_strength(vals[pos], pos, slot_counts) for pos in POSITIONS}
        name = str((r.get("metadata") or {}).get("team_name") or "")
        if not name:
            user = users.get(str(r.get("owner_id") or "")) or {}
            name = str(user.get("display_name") or user.get("username") or "")
        team_names[rid] = name or f"Roster {rid}"
    by_roster: dict[str, dict] = {}
    for rid in vals_by_roster:
        ranks = {}
        for pos in POSITIONS:
            order = sorted(strength, key=lambda other: strength[other][pos], reverse=True)
            ranks[pos] = order.index(rid) + 1 if rid in order else num_teams
        by_roster[rid] = {
            "team_name": team_names[rid],
            "needs": [pos for pos, _kind in classify_position_needs(
                ranks, vals_by_roster[rid], num_teams, thresholds, floors)],
            "surplus": detect_surplus_positions(
                ranks, vals_by_roster[rid], num_teams, thresholds, floors),
            "player_ids": [str(p) for p in next(
                (r.get("players") or [] for r in rosters if str(r.get("roster_id")) == rid), [])],
        }
    return {"by_roster": by_roster, "num_teams": num_teams}


def _identify_partner(bundle, entities, fit, give_ids, recv_ids):
    """Which league roster is on the other side of this deal, if knowable.

    First the router's other-team hint matched against team names, then
    ownership: the roster (not the viewer's) holding the receive-side
    players. Returns the fit entry for that roster or None.
    """
    by_roster = fit["by_roster"]
    viewer_rid = str((bundle.get("roster") or {}).get("roster_id") or "")
    hint = str((entities or {}).get("other_team_hint") or "").strip().lower()
    if hint:
        for rid, entry in by_roster.items():
            if rid == viewer_rid:
                continue
            name = entry["team_name"].lower()
            if hint in name or name in hint:
                return entry
    for pid in recv_ids or []:
        for rid, entry in by_roster.items():
            if rid != viewer_rid and str(pid) in entry["player_ids"]:
                return entry
    return None


def _balancing_asset(gap, partner, fit, values, index, give_ids, recv_ids, viewer_rid):
    """One concrete piece that closes the fairness gap, from real rosters.

    gap < 0 (the user gives more): the partner adds a piece. gap > 0: the
    user could add a piece to get it accepted. The piece is the rostered
    player outside the deal whose value lands closest to the gap, preferring
    the adding side's surplus positions. Nothing is suggested when the gap
    is thin or no real piece lands near it.
    """
    if not gap or abs(gap) < 1:
        return None
    deal_ids = {str(p) for p in (list(give_ids or []) + list(recv_ids or []))}
    if gap < 0:
        side_entry, side = partner, "partner_adds"
    else:
        side_entry = fit["by_roster"].get(viewer_rid)
        side = "you_add"
    if not side_entry:
        return None
    target = abs(float(gap))
    surplus = set(side_entry.get("surplus") or [])
    best = None
    for pid in side_entry.get("player_ids") or []:
        if pid in deal_ids:
            continue
        v = _row_value(values.get(str(pid)))
        if v is None or v <= 0:
            continue
        if not (target * 0.35 <= v <= target * 1.5):
            continue
        info = _player_info(pid, index)
        score = (abs(v - target), 0 if info.get("position") in surplus else 1)
        if best is None or score < best[0]:
            best = (score, {**info, "value": v})
    if best is None:
        return None
    player = best[1]
    return {"side": side, "player": player,
            "closes_gap_to": round(float(gap) + (player["value"] if side == "partner_adds" else -player["value"]), 1)}


# -- buy / sell / hold -------------------------------------------------------------

def build_buy_sell_hold(bundle, entities, player_ids) -> tuple[dict, list[str]]:
    pack = _base_pack("buy_sell_hold", bundle)
    tools: list[str] = []
    pid = player_ids[0] if player_ids else None
    if not pid:
        pack["available"] = False
        pack["missing"] = "no resolved player"
        return pack, tools
    index = _index()
    info = _player_info(pid, index)
    pack["player"] = info
    values = _value_rows()
    v = _row_value(values.get(str(pid)))
    if v is not None:
        pack["player"]["value"] = v
        tools.append("model_value_table")
    # Value trend
    try:
        from dashboard_services.player_value_history import get_player_value_history
        hist = get_player_value_history(str(pid), days=30) or []
        tools.append("player_value_history.get_player_value_history")
        vals = []
        for row in hist:
            if isinstance(row, dict):
                for k in ("value", "sf_value", "model_value"):
                    if row.get(k) is not None:
                        try:
                            vals.append(float(row[k]))
                        except Exception:
                            pass
                        break
        if len(vals) >= 2 and vals[0]:
            pack["value_trend_30d"] = {
                "start": vals[0], "current": vals[-1],
                "delta_pct": round((vals[-1] - vals[0]) / abs(vals[0]) * 100, 1),
            }
            pack["as_of"]["value_history_points"] = len(vals)
        elif vals:
            pack["value_trend_30d"] = {"current": vals[-1], "thin_sample": True}
    except Exception:
        logger.debug("[assistant] value history failed", exc_info=True)
    metrics, mtools = _metrics_for(pid, bundle.get("season"))
    tools.extend(mtools)
    if metrics:
        pack["role"] = metrics
    # Breakout board membership
    try:
        from dashboard_services.breakout_api import get_breakout_candidates
        board = get_breakout_candidates(season=bundle.get("season")) or {}
        tools.append("breakout_api.get_breakout_candidates")
        cands = board.get("candidates") or board.get("players") or []
        on_board = any(str(c.get("player_id") or c.get("id") or "") == str(pid) for c in cands if isinstance(c, dict))
        pack["breakout_board"] = {"member": bool(on_board)}
    except Exception:
        logger.debug("[assistant] breakout lookup failed", exc_info=True)
    # Signal synthesis (deterministic, from pack numbers only)
    delta = (pack.get("value_trend_30d") or {}).get("delta_pct")
    if delta is not None:
        pack["signal"] = "buy" if delta >= 5 else ("sell" if delta <= -5 else "hold")
        pack["margin"] = {"thin": bool(abs(delta) < 8)}
    return pack, tools


# -- player -------------------------------------------------------------------------

def build_player(bundle, entities, player_ids) -> tuple[dict, list[str]]:
    pack = _base_pack("player", bundle)
    tools: list[str] = []
    pid = player_ids[0] if player_ids else None
    if not pid:
        pack["available"] = False
        pack["missing"] = "no resolved player"
        return pack, tools
    index = _index()
    info = _player_info(pid, index)
    pack["player"] = info
    values = _value_rows()
    v = _row_value(values.get(str(pid)))
    if v is not None:
        pack["player"]["value"] = v
        tools.append("model_value_table")
    metrics, mtools = _metrics_for(pid, bundle.get("season"))
    tools.extend(mtools)
    if metrics:
        pack["role"] = metrics
        if metrics.get("as_of_date"):
            pack["as_of"]["metrics_as_of"] = metrics["as_of_date"]
    try:
        from data_building.advanced_metrics import get_player_metric_ranks
        ranks = get_player_metric_ranks(str(pid), season=bundle.get("season"))
        if isinstance(ranks, dict) and ranks.get("ranks"):
            pack["metric_ranks"] = ranks["ranks"]
            tools.append("advanced_metrics.get_player_metric_ranks")
    except Exception:
        logger.debug("[assistant] metric ranks failed", exc_info=True)
    try:
        from dashboard_services.news import get_player_news
        news = get_player_news(info["name"], limit=3) or {}
        items = news.get("items") or news.get("news") or (news if isinstance(news, list) else [])
        if items:
            pack["news"] = [{"title": (i.get("title") or i.get("headline") or "")} for i in items[:3] if isinstance(i, dict)]
            tools.append("news.get_player_news")
    except Exception:
        logger.debug("[assistant] news failed", exc_info=True)
    # Game logs (recent form) and the upcoming schedule, from the same
    # week-stat cache and schedule files the modal reads.
    form = _recent_form(bundle, pid, info.get("team"), info.get("position"), limit=4)
    if form:
        pack["recent_form"] = form
        pack["recent_ppg"] = round(sum(f["points"] for f in form) / len(form), 1)
        tools.append("utils.load_sleeper_week_stats")
    upcoming = _upcoming_schedule(bundle, info.get("team"), count=4)
    if upcoming:
        pack["upcoming_schedule"] = upcoming
        tools.append("utils.load_week_sched")
    return pack, tools


# -- metric ---------------------------------------------------------------------------

KNOWN_MISSING = {
    "route_rate": ("routes not tracked", ["target_share", "snap_share", "opportunity_share"]),
    "routes_run": ("routes not tracked", ["target_share", "snap_share"]),
    "yards_per_route_run": ("routes not tracked", ["target_share", "yards_per_target"]),
}


def _metric_config():
    from data_building.advanced_metrics import LEADERBOARD_METRICS
    return LEADERBOARD_METRICS


def build_metric(bundle, entities, player_ids=None) -> tuple[dict, list[str]]:
    pack = _base_pack("metric", bundle)
    tools: list[str] = ["advanced_metrics.config"]
    hint = str((entities or {}).get("metric_hint") or "").strip().lower().replace(" ", "_")
    pack["metric_key"] = hint or None
    try:
        config = _metric_config()
    except Exception:
        config = {}
    # Resolve hint to a config key by key, label, or substring.
    key = None
    if hint:
        if hint in config:
            key = hint
        else:
            for k, spec in config.items():
                label = str(spec.get("label") or "").lower().replace(" ", "_")
                if hint in (k, label) or hint in k or hint in label:
                    key = k
                    break
    if not key:
        missing_note, nearest = KNOWN_MISSING.get(hint, ("metric not in the site config", []))
        if not nearest:
            nearest = [k for k in list(config.keys())[:5]]
        pack["available"] = False
        pack["missing"] = missing_note
        pack["nearest"] = nearest
        pack["metric_key"] = hint or None
        return pack, tools
    spec = config[key]
    pack["metric_key"] = key
    pack["definition"] = spec.get("desc") or spec.get("label") or key
    pack["label"] = spec.get("label") or key
    pack["positions"] = spec.get("positions") or []
    try:
        from data_building.advanced_metrics import get_metric_leaderboard
        rows = get_metric_leaderboard(key, season=bundle.get("season")) or []
        tools.append("advanced_metrics.get_metric_leaderboard")
        top = []
        index = _index()
        for r in rows[:5]:
            if not isinstance(r, dict):
                continue
            pid = str(r.get("player_id") or r.get("id") or "")
            info = _player_info(pid, index) if pid else {}
            top.append({"player_id": pid, "name": r.get("name") or info.get("name") or pid,
                        "team": r.get("team") or info.get("team") or "",
                        "position": r.get("position") or info.get("position") or "",
                        "value": r.get("value") if r.get("value") is not None else r.get(key)})
        pack["leaderboard_top5"] = top
        roster_ids = bundle.get("roster_player_ids") or set()
        pack["leaderboard_top5"] = [
            {**t, "on_my_roster": bool(t["player_id"] in roster_ids)} for t in top
        ]
    except Exception:
        logger.debug("[assistant] leaderboard failed", exc_info=True)
    return pack, tools


# -- waivers ----------------------------------------------------------------------------

def build_waivers(bundle, entities, player_ids=None) -> tuple[dict, list[str]]:
    pack = _base_pack("waivers", bundle)
    tools: list[str] = ["model_value_table"]
    ctx = bundle.get("ctx") or {}
    roster = bundle.get("roster") or {}
    values = _value_rows()
    index = _index()
    roster_ids = {str(p) for p in (roster.get("players") or [])} or set(bundle.get("roster_player_ids") or set())
    # Free agents: everyone rostered anywhere in the league is owned.
    owned = set(roster_ids)
    for r in ctx.get("rosters") or []:
        if isinstance(r, dict):
            owned.update(str(p) for p in (r.get("players") or []))
    # My worst rostered player per position (the positional gate bar).
    worst: dict[str, dict] = {}
    for pid in roster_ids:
        info = _player_info(pid, index)
        v = _row_value(values.get(pid))
        if not info.get("position") or v is None:
            continue
        cur = worst.get(info["position"])
        if cur is None or v < cur["value"]:
            worst[info["position"]] = {**info, "value": v}
    pack["my_worst_by_position"] = worst
    # Roster holes: starter slots with no rostered player at the position.
    holes = []
    try:
        have_pos = {_player_info(p, index).get("position") for p in roster_ids}
        # A position with zero rostered players is a hole by definition here;
        # slot-level holes need lineup machinery beyond this pack.
        for pos in ("QB", "RB", "WR", "TE"):
            if pos not in have_pos and roster_ids:
                holes.append(pos)
    except Exception:
        holes = []
    pack["roster_holes"] = holes
    # Do-not-drop protected ids (cleared drops are rostered players NOT in it).
    protected = set()
    try:
        from routes.waiver_bp import load_do_not_drop
        from dashboard_services.assistant.context import resolve_account_id
        protected = {str(p) for p in (load_do_not_drop(resolve_account_id(), bundle.get("platform"), bundle.get("league_id")) or set())}
        tools.append("waiver_bp.load_do_not_drop")
    except Exception:
        protected = set()
    pack["do_not_drop_ids"] = sorted(protected)
    candidates = []
    for pid, row in values.items():
        if pid in owned:
            continue
        v = _row_value(row)
        if v is None or v <= 0:
            continue
        info = _player_info(pid, index)
        pos = info.get("position")
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        bar = worst.get(pos)
        passes = False
        receipt = None
        if pos in holes:
            passes = True
            receipt = f"fills your {pos} hole"
        elif bar is not None:
            passes = v > bar["value"] * 1.05
            receipt = f"candidate value {v} vs your worst {pos} {bar['name']} value {bar['value']}"
        candidates.append({**info, "value": v, "passes_gate": passes, "gate_receipt": receipt})
    candidates.sort(key=lambda c: c["value"], reverse=True)
    pack["candidates"] = [c for c in candidates if c["passes_gate"]][:8] or candidates[:0]
    pack["gate_passing_count"] = len([c for c in candidates if c["passes_gate"]])
    pack["cleared_drops"] = [
        {**_player_info(pid, index), "value": _row_value(values.get(pid))}
        for pid in roster_ids if pid not in protected
    ][:8]
    if not pack["candidates"]:
        pack["wire_thin"] = True
    # Trending adds (Sleeper's league-wide most-added feed, the same source
    # the waiver page annotates): counts on shown candidates, plus the top
    # trending free agents whether or not they clear the gate.
    try:
        trend_counts = {}
        for row in sleeper_trending_adds(limit=50) or []:
            if isinstance(row, dict) and row.get("player_id"):
                trend_counts[str(row["player_id"])] = int(row.get("count") or 0)
        if trend_counts:
            tools.append("waiver_bp._sleeper_trending_adds")
            for cand in pack["candidates"]:
                if cand["player_id"] in trend_counts:
                    cand["trending_adds_48h"] = trend_counts[cand["player_id"]]
            trending_fa = []
            for pid, count in trend_counts.items():
                if pid in owned:
                    continue
                info = _player_info(pid, index)
                if info.get("position") in ("QB", "RB", "WR", "TE"):
                    trending_fa.append({**info, "value": _row_value(values.get(pid)),
                                        "trending_adds_48h": count})
            trending_fa.sort(key=lambda c: c["trending_adds_48h"], reverse=True)
            pack["trending_free_agents"] = trending_fa[:5]
    except Exception:
        logger.debug("[assistant] trending adds failed", exc_info=True)
    # FAAB bid guidance, only when the league actually uses FAAB (the same
    # context check the waiver page gates its bid chips on). The bid comes
    # from the candidate's own pickup score: model value plus rest-of-season
    # PPG from advanced metrics, through the site's faab_recommendation.
    if ctx and pack["candidates"]:
        try:
            fx = waiver_faab_context(ctx)
        except Exception:
            fx = None
        if fx and fx.get("enabled"):
            try:
                from utils.waiver_score import faab_recommendation, waiver_pickup_score
                pack["faab"] = {"enabled": True, "waiver_type": fx.get("waiver_type"),
                                 "budget_total": fx.get("total")}
                tools.append("waiver_bp._waiver_faab_context")
                for cand in pack["candidates"][:3]:
                    metrics, _mt = _metrics_for(cand["player_id"], bundle.get("season"))
                    ros_ppg = metrics.get("expected_ppr_per_game") or metrics.get("ppg")
                    score = waiver_pickup_score(
                        {"value": cand.get("value") or 0, "position": cand.get("position"),
                         "player_id": cand["player_id"], "ros_ppg": ros_ppg,
                         "age": (index.get(str(cand["player_id"])) or {}).get("age")},
                        {},
                    )
                    rec = faab_recommendation(
                        score, budget_total=fx.get("total"), budget_remaining=None,
                        waiver_type=fx.get("waiver_type") or "faab",
                        season_phase=fx.get("phase") or "mid",
                    )
                    cand["faab"] = {k: rec.get(k) for k in (
                        "mode", "pct_low", "pct_target", "pct_high",
                        "pct_denominator", "rationale") if rec.get(k) is not None}
                tools.append("waiver_score.faab_recommendation")
            except Exception:
                logger.debug("[assistant] faab guidance failed", exc_info=True)
    return pack, tools


# -- this week ----------------------------------------------------------------------------

def build_this_week(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    pack = _base_pack("this_week", bundle)
    tools: list[str] = []
    actions: list[dict] = []
    ctx = bundle.get("ctx") or {}
    roster = bundle.get("roster") or {}
    # Lineup issues target (utils/lineup_issues.py -> utils.lineups).
    try:
        from utils.lineup_issues import find_lineup_issues
        index = _index()
        starters = [str(p) for p in (roster.get("starters") or [])]
        info = {pid: _player_info(pid, index) for pid in starters}
        issues = find_lineup_issues(starters, info) or []
        tools.append("utils.lineup_issues.find_lineup_issues")
        for iss in issues[:4]:
            if isinstance(iss, dict):
                actions.append({
                    "tag": "lineup", "headline": iss.get("detail") or f"Lineup issue: {iss.get('name') or iss.get('pid')}",
                    "detail": iss.get("detail") or "", "key_number": None,
                    "act_link": "/waivers?tab=startsit",
                })
    except Exception:
        logger.debug("[assistant] lineup issues failed", exc_info=True)
    # Digest action items: gather_digest_action_items needs roster + pidx and
    # many optional inputs; call it with what ctx provides, best-effort.
    try:
        from utils.digest import gather_digest_action_items
        items = gather_digest_action_items(
            platform=bundle.get("platform") or "sleeper",
            season=int(bundle.get("season") or 0),
            league_id=str(bundle.get("league_id") or ""),
            roster=roster or {},
            pidx=_index(),
            base_url="",
            nfl_state={"week": bundle.get("week"), "season": bundle.get("season"), "season_type": "reg"} if bundle.get("week") else None,
        ) or []
        tools.append("utils.digest.gather_digest_action_items")
        for item in items[:6]:
            if isinstance(item, dict):
                headline = item.get("headline") or item.get("title") or item.get("tag") or ""
                if not headline and item.get("html"):
                    continue  # HTML-only items are not renderable as action rows
                if headline:
                    actions.append({
                        "tag": item.get("tag") or "digest", "headline": str(headline),
                        "detail": str(item.get("detail") or ""), "key_number": item.get("key_number"),
                        "act_link": item.get("act_link") or item.get("href") or "/",
                    })
    except Exception:
        logger.debug("[assistant] digest items failed", exc_info=True)
    pack["actions"] = actions[:6]
    pack["available"] = True
    return pack, tools


BUILDERS = {
    "start_sit": build_start_sit,
    "trade_eval": build_trade_eval,
    "buy_sell_hold": build_buy_sell_hold,
    "player": build_player,
    "metric": build_metric,
    "waivers": build_waivers,
    "this_week": build_this_week,
}
