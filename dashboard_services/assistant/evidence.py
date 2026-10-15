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
- The this-week digest draws from the dashboard's own Next Steps queue
  (the _next_steps_*_actions builders in app.py over the live league ctx),
  the same source of truth the site renders. It previously called
  utils.digest.gather_digest_action_items (the weekly EMAIL digest) with
  almost none of that function's enrichment inputs, and mapped fields the
  digest items do not carry (headline/tag), so waiver and injury items
  were silently dropped and a real team got an empty digest.
- The trade pack carries trade_verdict, derived from the side totals via
  utils.trade_value.fairness_label / fair_value_band: the shared fairness
  definition the trade calculator (api_trade_eval) and the trade outcome
  analyzer use, so the assistant cannot contradict the calculator.
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


def metric_leaderboard(*args, **kwargs):
    from data_building.advanced_metrics import get_metric_leaderboard as _fn
    return _fn(*args, **kwargs)


def weekly_range_leaderboard(*args, **kwargs):
    from data_building.advanced_metrics import get_weekly_range_leaderboard as _fn
    return _fn(*args, **kwargs)


def adv_weekly_range_leaderboard(*args, **kwargs):
    from data_building.advanced_metrics import get_adv_weekly_range_leaderboard as _fn
    return _fn(*args, **kwargs)


def situational_leaderboard(*args, **kwargs):
    from data_building.advanced_metrics import get_situational_leaderboard as _fn
    return _fn(*args, **kwargs)


def adv_weekly_supported(metric) -> bool:
    from data_building.advanced_metrics import adv_weekly_metric_supported
    return bool(adv_weekly_metric_supported(metric))


def weekly_metric_keys():
    from data_building.advanced_metrics import _WEEKLY_METRICS
    return _WEEKLY_METRICS


def situational_metric_keys():
    from data_building.advanced_metrics import SITUATIONAL_METRICS
    return SITUATIONAL_METRICS


def qualification_policy_for(season, week_start=None, week_end=None):
    from utils.season_qualification import qualification_policy
    return qualification_policy(season, week_start=week_start, week_end=week_end)


def pro_metric_keys():
    from data_building.advanced_metrics import PRO_METRICS
    return PRO_METRICS


def premium_metric_keys():
    from data_building.advanced_metrics import PREMIUM_METRICS
    return PREMIUM_METRICS


def premium_metrics_exposed() -> bool:
    from data_building.advanced_metrics import premium_metrics_exposed as _fn
    return bool(_fn())


def request_has_pro(bundle) -> bool:
    """Per-user PRO check for assistant callers, mirroring
    routes/advanced_metrics_bp._request_has_premium: the same
    has_premium_for_viewer call with the session's viewer identity and this
    bundle's league/platform/season. Fail closed."""
    try:
        from flask import session
        from dashboard_services.subscriptions import has_premium_for_viewer
        return bool(has_premium_for_viewer(
            session.get("viewer_username"), session.get("viewer_user_id"),
            (bundle or {}).get("league_id"), (bundle or {}).get("platform", "sleeper"),
            (bundle or {}).get("season"),
        ))
    except Exception:
        return False


def _caller_has_pro(bundle) -> bool:
    if isinstance(bundle, dict) and bundle.get("has_pro") is not None:
        return bool(bundle["has_pro"])
    return request_has_pro(bundle or {})


def sleeper_trending_adds(*args, **kwargs):
    from routes.waiver_bp import _sleeper_trending_adds as _fn
    return _fn(*args, **kwargs)


def waiver_faab_context(*args, **kwargs):
    from routes.waiver_bp import _waiver_faab_context as _fn
    return _fn(*args, **kwargs)


def next_steps_actions(ctx, roster_id, model_rows):
    """The dashboard's Next Steps queue, aggregated like _render_next_steps_queue.

    Same builders (app.py), same ctx, same score-ranked order as the queue
    the site renders under the hero. Each builder is best-effort internally.
    """
    from app import (
        _next_steps_bench_check_actions,
        _next_steps_lineup_actions,
        _next_steps_roster_actions,
        _next_steps_trade_actions,
        _next_steps_waiver_actions,
    )
    actions: list[dict] = []
    actions.extend(_next_steps_lineup_actions(ctx, roster_id) or [])
    actions.extend(_next_steps_waiver_actions(ctx, roster_id, model_rows or []) or [])
    actions.extend(_next_steps_trade_actions(ctx, roster_id) or [])
    actions.extend(_next_steps_roster_actions(ctx, roster_id) or [])
    actions.extend(_next_steps_bench_check_actions(
        ctx, roster_id, (ctx or {}).get("last_final_week")) or [])
    actions.sort(key=lambda a: -(a.get("score") or 0))
    return actions


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
    # The verdict is code-derived from the totals via the trade analyzer's
    # shared fairness band (utils.trade_value), never left to the model:
    # strong_win/strong_loss are from the viewer's side of the delta
    # (fairness_gap is receive minus give).
    try:
        from utils.trade_value import fairness_label
        label = fairness_label(pack["fairness_gap"], base)
        pack["trade_verdict"] = {"strong_win": "ACCEPT", "strong_loss": "DECLINE",
                                 "fair": "FAIR"}[label]
        tools.append("utils.trade_value.fairness_label")
    except Exception:
        logger.debug("[assistant] trade verdict failed", exc_info=True)
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
    # Player-in-pool framing for target share: the player's rank inside the
    # QUALIFIED position pool (the same leaderboard pool the AM page uses).
    if metrics.get("target_share") is not None and info.get("position"):
        try:
            config = _metric_config()
            spec = config.get("target_share") or {}
            qrows, _qual = _season_board("target_share", spec,
                                          bundle.get("season"), info["position"])
            for i, r in enumerate(qrows):
                if isinstance(r, dict) and str(r.get("player_id") or r.get("id") or "") == str(pid):
                    pool = len(qrows)
                    rank = i + 1
                    pack["pool_framing"] = {
                        "metric_key": "target_share", "label": "Target Share",
                        "rank": rank, "pool_size": pool,
                        "percentile": (round(100 * (pool - rank) / max(pool - 1, 1))
                                       if pool > 1 else 100),
                    }
                    tools.append("advanced_metrics.get_metric_leaderboard")
                    break
        except Exception:
            logger.debug("[assistant] player pool framing failed", exc_info=True)
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


def _resolve_metric_key(hint, config) -> str | None:
    """Resolve a free-text metric hint to a LEADERBOARD_METRICS key: exact
    key, then label or substring match (the build_metric resolution rule,
    shared by the leaderboard builder)."""
    if not hint:
        return None
    if hint in config:
        return hint
    for k, spec in config.items():
        label = str(spec.get("label") or "").lower().replace(" ", "_")
        if hint in (k, label) or hint in k or hint in label:
            return k
    return None


def _r3(value):
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _vol_label(spec) -> str:
    mv = (spec or {}).get("min_vol") or {}
    label = str(mv.get("label") or "").strip().lower()
    if label.startswith("min "):
        label = label[4:]
    return label or str(mv.get("col") or "games")


def _season_board(key, spec, season, position):
    """(rows, qualification) for a season board, using the SAME pool
    definition as the Advanced Metrics leaderboard route: the metric's own
    min_vol default scaled by the season qualification policy
    (routes/advanced_metrics_bp.py computes min_vol exactly this way)."""
    min_vol = None
    games_min = None
    provisional = False
    if isinstance(season, int):
        mv = (spec or {}).get("min_vol") or {}
        opts = mv.get("opts") or []
        full_default = int(opts[0]) if opts else 1
        try:
            policy = qualification_policy_for(season)
            min_vol = policy.minimum(mv.get("col") or "games", full_default)
            games_min = policy.games_min
            provisional = bool(policy.provisional)
        except Exception:
            min_vol = None
    rows = metric_leaderboard(key, position=position, season=season,
                              min_vol=min_vol, limit=500) or []
    qual = {"vol_col": ((spec or {}).get("min_vol") or {}).get("col") or "games",
            "vol_label": _vol_label(spec), "min_vol": min_vol,
            "games_min": games_min, "provisional": provisional}
    return rows, qual


def _range_capable(key) -> bool:
    """Week-range boards exist for weekly, advanced-weekly, and situational
    metrics: the same dispatch the AM route uses."""
    try:
        if key in (weekly_metric_keys() or {}):
            return True
    except Exception:
        pass
    try:
        if adv_weekly_supported(key):
            return True
    except Exception:
        pass
    try:
        if key in (situational_metric_keys() or {}):
            return True
    except Exception:
        pass
    return False


def _range_board(key, spec, season, position, week_start, week_end):
    """Rows for one week window, dispatched exactly like the AM route's
    week-filtered leaderboard (weekly, then adv-weekly, then situational),
    with the qualification-scaled min_vol the route computes."""
    min_vol = None
    if isinstance(season, int):
        mv = (spec or {}).get("min_vol") or {}
        opts = mv.get("opts") or []
        full_default = int(opts[0]) if opts else 1
        try:
            policy = qualification_policy_for(season, week_start=week_start,
                                              week_end=week_end)
            min_vol = policy.minimum(mv.get("col") or "games", full_default)
        except Exception:
            min_vol = None
    if key in (weekly_metric_keys() or {}):
        rows = weekly_range_leaderboard(key, position=position, season=season,
                                        week_start=week_start, week_end=week_end,
                                        min_vol=min_vol, limit=500)
    elif adv_weekly_supported(key):
        rows = adv_weekly_range_leaderboard(key, position=position, season=season,
                                            week_start=week_start, week_end=week_end,
                                            min_vol=min_vol, limit=500)
    elif key in (situational_metric_keys() or {}):
        rows = situational_leaderboard(key, position=position, season=season,
                                       week_start=week_start, week_end=week_end,
                                       limit=500)
    else:
        rows = []
    return rows or []


def _gate_for(bundle, key) -> str | None:
    """PRO/PREMIUM gating with parity to routes/advanced_metrics_bp.py:
    premium metrics that are not exposed answer 'metric not available';
    PRO metrics answer gated for callers without PRO. Returns 'gated',
    'unavailable', or None."""
    try:
        if key in (premium_metric_keys() or set()) and not premium_metrics_exposed():
            return "unavailable"
    except Exception:
        pass
    try:
        if key in (pro_metric_keys() or set()) and not _caller_has_pro(bundle):
            return "gated"
    except Exception:
        pass
    return None


def _season_qual_text(qual, pool_size, position, team) -> str:
    pool_what = f"{position} players" if position else "players"
    text = f"Qualified pool: {pool_size} {pool_what}"
    if qual.get("min_vol") is not None:
        text += f", minimum {qual['min_vol']} {qual.get('vol_label') or 'games'}"
    text += "."
    if qual.get("provisional"):
        text += " Early season: the pool is still provisional and ranks will move."
    if team:
        text += f" Showing {team} only."
    return text


def _board_row(rank, r, roster_ids, value=None):
    pid = str(r.get("player_id") or r.get("id") or "")
    return {"rank": rank, "player_id": pid,
            "name": r.get("name") or pid,
            "team": r.get("team") or "",
            "position": r.get("position") or "",
            "value": _r3(value if value is not None else
                         (r.get("value") if r.get("value") is not None else None)),
            "games": r.get("games"),
            "on_my_roster": bool(pid and pid in roster_ids)}


def _gated_pack(pack, key, label):
    pack["available"] = False
    pack["gated"] = True
    pack["metric_key"] = key
    pack["label"] = label
    pack["missing"] = "pro_only"
    return pack


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
    key = _resolve_metric_key(hint, config)
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
    # Player-in-pool framing: where the asked-about player sits in the
    # QUALIFIED pool (the AM page's qualification minimums), not just the
    # raw list above.
    if player_ids:
        try:
            qrows, _qual = _season_board(key, spec, bundle.get("season"), None)
            pack["pool_size"] = len(qrows)
            target = str(player_ids[0])
            for i, r in enumerate(qrows):
                if isinstance(r, dict) and str(r.get("player_id") or r.get("id") or "") == target:
                    pool = len(qrows)
                    rank = i + 1
                    pack["player_rank"] = {
                        "player_id": target, "rank": rank, "pool_size": pool,
                        "percentile": (round(100 * (pool - rank) / max(pool - 1, 1))
                                       if pool > 1 else 100),
                        "value": _r3(r.get("value") if r.get("value") is not None else r.get(key)),
                    }
                    break
        except Exception:
            logger.debug("[assistant] metric pool framing failed", exc_info=True)
    return pack, tools


# -- leaderboard (deep cuts) ------------------------------------------------------------

def _no_board(pack, missing, nearest):
    pack["available"] = False
    pack["missing"] = missing
    pack["nearest"] = nearest
    return pack


def _build_matchup_board(pack, bundle, position, top_n, tools):
    """Defenses ranked by the site's matchup ratings (the same
    app._matchup_rank_table the start/sit path uses): rank 1 is the easiest
    matchup; the value shown is raw fantasy points allowed per game."""
    if not position:
        return _no_board(pack, "a position to rank defenses against", []), tools
    season = bundle.get("season")
    rows = []
    total = 0
    if season:
        try:
            table = matchup_rank_table(int(season), position)
            tools.append("app._matchup_rank_table")
            ranks = table[0] or {}
            total = table[1] or len(ranks)
            info = table[2] if len(table) > 2 else {}
            for team, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
                fpts = (info.get(team) or {}).get("fpts")
                if fpts is None:
                    continue
                try:
                    rows.append({"rank": int(rank), "team": team, "name": team,
                                 "player_id": "", "position": "",
                                 "value": round(float(fpts), 1),
                                 "on_my_roster": False})
                except (TypeError, ValueError):
                    continue
        except Exception:
            logger.debug("[assistant] matchup board failed", exc_info=True)
            rows = []
    if not rows:
        return _no_board(pack, "matchup ratings are not available yet", []), tools
    pack["board"] = rows[:top_n]
    pack["board_count"] = len(pack["board"])
    pack["pool_size"] = int(total) if total else len(rows)
    pack["metric_key"] = "defense_points_allowed"
    pack["label"] = f"Fantasy points allowed to {position}"
    pack["qualification_text"] = (
        f"All {pack['pool_size']} defenses, ranked easiest to hardest by the "
        f"site's opponent-adjusted matchup ratings. Values are raw fantasy "
        f"points allowed per game to {position}.")
    return pack, tools


def build_leaderboard(bundle, entities, player_ids=None) -> tuple[dict, list[str]]:
    """Pool-wide boards ("deep cuts"): leaderboards, intersections and dual
    ranks, movers, and matchup boards. Every rank and value is computed here
    from the Advanced Metrics leaderboards (one pool definition, the AM
    page's own qualification minimums) or the matchup-rank table; the model
    only phrases an intro and follow-ups around the finished board."""
    pack = _base_pack("leaderboard", bundle)
    tools: list[str] = ["advanced_metrics.config"]
    ent = entities or {}
    kind = str(ent.get("board_kind") or "board").strip().lower()
    if kind not in ("board", "intersection", "dual", "movers", "matchup"):
        kind = "board"
    pack["board_kind"] = kind
    try:
        top_n = int(ent.get("top_n") or 10)
    except (TypeError, ValueError):
        top_n = 10
    top_n = max(1, min(25, top_n))
    pack["requested_n"] = top_n
    position = str(ent.get("position") or "").strip().upper() or None
    if position not in ("QB", "RB", "WR", "TE"):
        position = None
    pack["position_filter"] = position
    team = str(ent.get("team") or "").strip().upper() or None
    pack["team_filter"] = team
    season = bundle.get("season")
    roster_ids = bundle.get("roster_player_ids") or set()

    if kind == "matchup":
        return _build_matchup_board(pack, bundle, position, top_n, tools)

    try:
        config = _metric_config()
    except Exception:
        config = {}
    hint = str(ent.get("metric_hint") or "").strip().lower().replace(" ", "_")
    pack["metric_hint"] = hint or None
    key = _resolve_metric_key(hint, config)
    trend_default = False
    if not key and kind == "movers" and not hint:
        # "Biggest risers" with no metric named is the site's own movers
        # strip: the Usage Trend board (PRO-gated, like the strip itself).
        key = "opportunity_trend" if "opportunity_trend" in config else None
        trend_default = bool(key)
    if not key:
        missing_note, nearest = KNOWN_MISSING.get(hint, ("metric not in the site config", []))
        if not nearest:
            nearest = [k for k in list(config.keys())[:5]]
        return _no_board(pack, missing_note, nearest), tools
    spec = config.get(key) or {}
    pack["metric_key"] = key
    pack["label"] = spec.get("label") or key
    pack["definition"] = spec.get("desc") or spec.get("label") or key
    gate = _gate_for(bundle, key)
    if gate == "gated":
        return _gated_pack(pack, key, pack["label"]), tools
    if gate == "unavailable":
        return _no_board(pack, "metric not available", []), tools

    second_key = None
    second_spec = None
    if kind in ("intersection", "dual"):
        shint = str(ent.get("second_metric_hint") or "").strip().lower().replace(" ", "_")
        second_key = _resolve_metric_key(shint, config) if shint else None
        if second_key:
            second_spec = config.get(second_key) or {}
            gate2 = _gate_for(bundle, second_key)
            if gate2 == "gated":
                return _gated_pack(pack, second_key,
                                   second_spec.get("label") or second_key), tools
            if gate2 == "unavailable":
                return _no_board(pack, "metric not available", []), tools
            pack["second_metric_key"] = second_key
            pack["second_label"] = second_spec.get("label") or second_key
        else:
            # One resolvable metric: degrade to the plain board.
            kind = "board"
            pack["board_kind"] = "board"

    try:
        window_weeks = int(ent.get("window_weeks") or 0)
    except (TypeError, ValueError):
        window_weeks = 0
    week = bundle.get("week")
    completed = (int(week) - 1) if week else 0
    direction = str(ent.get("direction") or "risers").strip().lower()
    if direction not in ("risers", "fallers"):
        direction = "risers"

    if kind == "movers":
        return _build_movers(pack, bundle, tools, key, spec, trend_default,
                             direction, window_weeks, completed, position,
                             team, top_n, roster_ids)

    # -- season or windowed boards (board / intersection / dual) ----------------
    window = {"kind": "season", "applied": False,
              "requested_weeks": window_weeks or None}
    range_ok = bool(window_weeks) and _range_capable(key) and (
        second_key is None or _range_capable(second_key))
    if range_ok and completed >= 1:
        ws = max(1, completed - window_weeks + 1)
        we = completed
        rows = _range_board(key, spec, season, position, ws, we)
        tools.append("advanced_metrics.range_leaderboard")
        second_rows = (_range_board(second_key, second_spec, season, position, ws, we)
                       if second_key else None)
        window = {"kind": "weeks", "week_start": ws, "week_end": we,
                  "applied": True, "requested_weeks": window_weeks}
        pool_size = len(rows)
        qual_text = (f"Weeks {ws} to {we} only. Qualified pool: {pool_size} "
                     f"players.")
        if team:
            qual_text += f" Showing {team} only."
    else:
        rows, qual = _season_board(key, spec, season, position)
        tools.append("advanced_metrics.get_metric_leaderboard")
        second_rows = None
        if second_key:
            srows, _sq = _season_board(second_key, second_spec, season, position)
            second_rows = srows
        pool_size = len(rows)
        pack["qualification"] = qual
        qual_text = _season_qual_text(qual, pool_size, position, team)
    pack["window"] = window
    pack["pool_size"] = pool_size
    pack["qualification_text"] = qual_text

    def _filtered(rs):
        if not team:
            return list(rs)
        return [r for r in rs if str(r.get("team") or "").upper() == team]

    rows = _filtered(rows)
    board = _shape_board(kind, rows, second_rows, top_n, roster_ids)
    if not board:
        return _no_board(pack, "no qualified players match those filters", []), tools
    pack["board"] = board
    pack["board_count"] = len(board)
    return pack, tools


def _shape_board(kind, rows, second_rows, top_n, roster_ids):
    """Rank-order rows into the pack board. Source order is the board's own
    ranking (the leaderboard layer sorts by value, honoring lower-is-better
    metrics); this never re-sorts."""
    second_by_id = {}
    if second_rows:
        for i, r in enumerate(second_rows):
            if isinstance(r, dict):
                pid = str(r.get("player_id") or r.get("id") or "")
                if pid:
                    second_by_id[pid] = (i + 1, r)
    board = []
    if kind == "intersection":
        primary_top = rows[:top_n]
        second_top_ids = {str(r.get("player_id") or r.get("id") or "")
                          for r in (second_rows or [])[:top_n]
                          if isinstance(r, dict)}
        for i, r in enumerate(primary_top):
            if not isinstance(r, dict):
                continue
            pid = str(r.get("player_id") or r.get("id") or "")
            if pid not in second_top_ids:
                continue
            row = _board_row(len(board) + 1, r, roster_ids)
            srank, srow = second_by_id[pid]
            row["rank_primary"] = i + 1
            row["secondary"] = {"value": _r3(srow.get("value")),
                                "rank": srank}
            board.append(row)
        return board
    for i, r in enumerate(rows[:top_n]):
        if not isinstance(r, dict):
            continue
        row = _board_row(i + 1, r, roster_ids)
        if kind == "dual" and second_rows is not None:
            pid = str(r.get("player_id") or r.get("id") or "")
            hit = second_by_id.get(pid)
            row["secondary"] = ({"value": _r3(hit[1].get("value")), "rank": hit[0]}
                                if hit else None)
        board.append(row)
    return board


def _build_movers(pack, bundle, tools, key, spec, trend_default, direction,
                  window_weeks, completed, position, team, top_n, roster_ids):
    season = bundle.get("season")
    label = pack["label"]
    if trend_default:
        # The Usage Trend board itself is the movement ranking: its value is
        # recent usage versus the season baseline.
        rows, qual = _season_board(key, spec, season, position)
        tools.append("advanced_metrics.get_metric_leaderboard")
        pack["pool_size"] = len(rows)
        pack["qualification"] = qual
        pack["movers_basis"] = "trend_metric"
        pack["direction"] = direction
        pack["qualification_text"] = _season_qual_text(
            qual, len(rows), position, team)
        ordered = list(rows)
        if direction == "fallers":
            ordered.reverse()
        ordered = [r for r in ordered
                   if not team or str(r.get("team") or "").upper() == team]
        board = [_board_row(i + 1, r, roster_ids)
                 for i, r in enumerate(ordered[:top_n]) if isinstance(r, dict)]
        if not board:
            return _no_board(pack, "no qualified players match those filters", []), tools
        pack["board"] = board
        pack["board_count"] = len(board)
        return pack, tools
    if not _range_capable(key):
        nearest = [k for k in ("target_share", "snap_share") if k != key]
        return _no_board(
            pack, f"{label} does not have week-by-week history here", nearest), tools
    if completed < 4:
        return _no_board(
            pack, "not enough completed weeks yet to compare windows", []), tools
    recent_len = min(window_weeks or 4, completed - 2)
    rs, re_ = completed - recent_len + 1, completed
    bs, be = 1, rs - 1
    recent = _range_board(key, spec, season, position, rs, re_)
    base = _range_board(key, spec, season, position, bs, be)
    tools.append("advanced_metrics.range_leaderboard")
    base_by_id = {str(r.get("player_id") or r.get("id") or ""): r
                  for r in base if isinstance(r, dict)}
    joined = []
    for r in recent:
        if not isinstance(r, dict):
            continue
        pid = str(r.get("player_id") or r.get("id") or "")
        brow = base_by_id.get(pid)
        if brow is None:
            continue
        rv, bv = _r3(r.get("value")), _r3(brow.get("value"))
        if rv is None or bv is None:
            continue
        joined.append((r, rv, bv, round(rv - bv, 3)))
    joined.sort(key=lambda t: t[3], reverse=(direction == "risers"))
    pack["pool_size"] = len(joined)
    pack["direction"] = direction
    pack["movers_basis"] = "window_delta"
    pack["window"] = {"kind": "movers", "recent_start": rs, "recent_end": re_,
                      "recent_weeks": recent_len, "base_start": bs,
                      "base_end": be, "applied": True}
    pack["qualification_text"] = (
        f"Recent window: weeks {rs} to {re_}. Baseline: weeks {bs} to {be}. "
        f"Only players with a {label} in both windows are ranked; "
        f"{len(joined)} qualify.")
    if team:
        pack["qualification_text"] += f" Showing {team} only."
    board = []
    for r, rv, bv, delta in joined:
        if team and str(r.get("team") or "").upper() != team:
            continue
        row = _board_row(len(board) + 1, r, roster_ids, value=rv)
        row["baseline_value"] = bv
        row["delta"] = delta
        board.append(row)
        if len(board) >= top_n:
            break
    if not board:
        return _no_board(pack, "no qualified players match those filters", []), tools
    pack["board"] = board
    pack["board_count"] = len(board)
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
    """The weekly digest: the dashboard's own Next Steps queue, as a pack.

    The bundle already carries the live league ctx and the viewer's roster,
    which is exactly what the site's queue builders read. Each queue action
    maps onto the pack's action shape: tag, headline (the queue's action
    text), detail (its why line), key_number (its impact string, e.g.
    "+1.9 projected points", or None), act_link (its CTA url), and the
    CTA label it ships with.
    """
    pack = _base_pack("this_week", bundle)
    tools: list[str] = []
    actions: list[dict] = []
    ctx = bundle.get("ctx") or {}
    roster = bundle.get("roster") or {}
    roster_id = roster.get("roster_id")
    if ctx and roster_id is not None:
        try:
            model_rows = get_model_value_table_cached() or []
        except Exception:
            model_rows = []
        try:
            queue = next_steps_actions(ctx, roster_id, model_rows)
            tools.append("app.next_steps_queue")
        except Exception:
            logger.debug("[assistant] next steps queue failed", exc_info=True)
            queue = []
        for a in queue[:6]:
            if not isinstance(a, dict):
                continue
            headline = str(a.get("action") or "").strip()
            if not headline:
                continue
            actions.append({
                "tag": a.get("tag") or a.get("kind") or "action",
                "headline": headline,
                "detail": str(a.get("why") or ""),
                "key_number": a.get("impact"),
                "act_link": a.get("cta_url") or "/",
                "cta_label": a.get("cta_label"),
            })
    pack["actions"] = actions
    pack["available"] = True
    return pack, tools


BUILDERS = {
    "start_sit": build_start_sit,
    "trade_eval": build_trade_eval,
    "buy_sell_hold": build_buy_sell_hold,
    "player": build_player,
    "metric": build_metric,
    "leaderboard": build_leaderboard,
    "waivers": build_waivers,
    "this_week": build_this_week,
}
