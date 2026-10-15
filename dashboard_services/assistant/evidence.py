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
- Roster-scoped start/sit (2026-10-15): build_start_sit's named-player
  path is unchanged, but a start_sit question that resolves NO players
  (a slot question like "who should I start at FLEX", a who-to-sit
  mirror, or a bare "who should I start this week") no longer declines:
  _roster_start_sit builds the candidate set from the asking user's own
  roster in the pack's league, running the site's lineup solver
  (utils.lineups assign_optimal_lineup / assign_fixed_lineup) over the
  same week projection map, with each candidate scored by the same
  per-player path as named questions (weekly projection plus
  utils.start_sit compute_start_score). The pack carries roster_mode,
  slot/slot_label, question_mode (start|sit), decisions (per-slot
  in/out/gain), recommended_player_id, and sit_player_id on sit
  questions; entries carry current_starter and slot_occupant flags.
  The router carries the slot on both paths: the model schema has
  slot/start_mode entities, and the keyword fallback parses them from
  the message (_slot_in / _start_mode_in).
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
- Trade FINDING runs the site's own suggestion machinery behind the site's
  own PRO gate: subscriptions.has_premium_for_viewer with the same
  arguments /api/trade-suggestions passes (that route 403s before building
  anything). Need mode reads ai.context_builders
  .build_trade_suggestions_context, the data path behind
  get_trade_suggestions_html, whose top_partners are complementary fits
  (partner needs x viewer surplus), fairness-checked and depth-checked in
  that builder. Sell and specific-target modes match packages with
  trade_acquire_packages.value_matched_acquire_packages (the matcher the
  Trade Hub targets flow uses) and judge plausibility with _league_fit
  needs/surplus plus the fair_value_band above.
  archetype_engine.get_archetype_suggestions was verified but is NOT wired
  into this synchronous path: it is a Monte Carlo strategy-slate pipeline
  (the Trade Hub itself loads it progressively, slate then sim phases)
  keyed by strategy archetypes, not a per-question partner scan.
- League fit is the Teams page machinery, not a local mirror: positional
  strengths and ranks come from utils.trade.rank_rosters_by_position over
  league_format_value_lookup values (the same lookup My Leagues and the
  Teams page share), and roster grades come from
  dashboard_services.ai.context_builders.calculate_roster_grade with the
  same percentile inputs the Teams page passes. Needs/surplus still come
  from utils.trade.classify_position_needs / detect_surplus_positions.
- Draft picks are priced AND attributed: ownership comes from the league
  ctx's picks_by_roster (built by app.py from the provider's traded picks;
  platform_api.get_traded_picks is the fallback source), slots follow the
  site convention (original owner's standings order, worst record picks
  first), and values come from dashboard_services.picks
  .load_pick_value_table via utils.draft.pick_value_from_table. Pack picks
  carry canonical pick ids (2027_1_03) per side as give_picks /
  receive_picks, the shape the trade calculator's deep link stages.
- The breakout board builder mirrors the board route's non-premium
  preview rule exactly: a caller without a paid plan sees the first 3
  candidates plus the locked count, never the names beyond the preview.
- Keeper answers use the Keeper Assistant's own machinery
  (dashboard_services.pages.keeper_page): league_keeper_limit gates
  applicability (a redraft league with no keeper settings gets the no-data
  shape, never a default-limit invention), compute_league_keepers runs the
  surplus optimizer for the projected set and cost rounds, and a full
  dynasty league (is_dynasty_without_keepers) gets dynasty framing from
  the roster's stored values and ages instead of meaningless last-round
  costs, the same call the keeper page itself makes.
- Rookie answers read the prospect pipeline's stored rows exactly as
  stored (rookie_api._get_rankings over get_rookie_rankings_from_db, plus
  _get_prospect_detail for one prospect's college seasons): scores are
  never recomputed or rescaled, and the pack carries the rows'
  model_version as score_version so the stored set is identifiable.
- Buy-low / sell-high boards (leaderboard kinds buy_low / sell_high) run
  the site's own movers computation (player_value_history.get_top_movers,
  the /api/value-movers source, with that route's displayed-value screen)
  and keep only movers whose production supports the move: PPG vs
  expected PPG from the same per-player metrics the buy/sell pack ships.
  A pool-wide buy/sell question names no player, so it is a board, never
  a single-player buy_sell_hold decline.
- Cross-league answers fan out over the account's leagues with the
  portfolio page's own builder (portfolio_summary.build_league_summary,
  fed by accounts.list_user_leagues and the assistant's league ctx
  loader). Each league is a labeled slice with its own numbers; nothing
  is blended across leagues. The fan-out runs inside one ask(), so the
  turn still counts once against the monthly cap.
- League activity answers read the Activity page's own data: the ctx
  activity frame that app.ensure_activity_bits populates from
  dashboard_services.service.build_week_activity (trades, waiver adds,
  and serious-injury rows, newest first). Trade side totals are priced
  from the same model value table the trade packs use, and the fairness
  call uses utils.trade.fairness_label over the shared fair_value_band,
  so an activity recap can never contradict the calculator.
- Watchlist answers are read-only over the Watchlist tool's own storage:
  routes.watchlist_bp._rows_for under the same acct:<id> key the tool
  uses (its _user_key reads the same session), with 7-day deltas from
  the same movers board /api/watchlist-alerts reads
  (player_value_history.get_top_movers) and the alert call made by
  utils.watchlist_alerts.is_value_alert, the threshold the in-app and
  push alerts share, plus any real injury designation from the players
  index. The assistant never adds or removes a watched player.
- Front Office answers carry the report's computed DATA only
  (ai/front_office_report.build_front_office_data: direction, playoff
  odds, weakest positions, the deterministic top moves, trade targets,
  waiver targets, cut candidates), distilled to names and totals. The
  report's written memo is never quoted; the model presents the data in
  its own words. The pack sits behind the same PRO gate /api/gm-memo
  applies (subscriptions.has_premium_for_viewer with the session viewer
  identity) before any report data is built.
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


def get_top_movers(*args, **kwargs):
    from data_building.player_value_history import get_top_movers as _fn
    return _fn(*args, **kwargs)


def get_player_weekly_adv_series(*args, **kwargs):
    from data_building.advanced_metrics import (
        get_player_weekly_adv_series as _fn)
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


def has_premium_for_viewer(*args, **kwargs):
    from dashboard_services.subscriptions import has_premium_for_viewer as _fn
    return _fn(*args, **kwargs)


def build_trade_suggestions_context(*args, **kwargs):
    from dashboard_services.ai.context_builders import (
        build_trade_suggestions_context as _fn,
    )
    return _fn(*args, **kwargs)


def value_matched_acquire_packages(*args, **kwargs):
    from dashboard_services.trade_acquire_packages import (
        value_matched_acquire_packages as _fn,
    )
    return _fn(*args, **kwargs)


def sleeper_trending_adds(*args, **kwargs):
    from routes.waiver_bp import _sleeper_trending_adds as _fn
    return _fn(*args, **kwargs)


def ensure_activity_bits(*args, **kwargs):
    from app import ensure_activity_bits as _fn
def watchlist_user_key(*args, **kwargs):
    from routes.watchlist_bp import _user_key as _fn
    return _fn(*args, **kwargs)


def watchlist_rows(*args, **kwargs):
    from routes.watchlist_bp import _rows_for as _fn
    return _fn(*args, **kwargs)


def watchlist_top_movers(*args, **kwargs):
    from dashboard_services.player_value_history import get_top_movers as _fn
def front_office_data(*args, **kwargs):
    from dashboard_services.ai.front_office_report import build_front_office_data as _fn
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


def breakout_board_candidates(*args, **kwargs):
    from dashboard_services.breakout_api import get_breakout_board_candidates as _fn
    return _fn(*args, **kwargs)


def breakout_track_record_data(*args, **kwargs):
    from dashboard_services.breakout_api import get_breakout_track_record as _fn
    return _fn(*args, **kwargs)


def breakout_call_history_for(*args, **kwargs):
    from dashboard_services.breakout_api import breakout_call_history as _fn
    return _fn(*args, **kwargs)


def breakout_week_detail(*args, **kwargs):
    from dashboard_services.breakout_api import get_historical_weekly_breakout_detail as _fn
    return _fn(*args, **kwargs)


def pick_value_table(*args, **kwargs):
    from dashboard_services.picks import load_pick_value_table as _fn
    return _fn(*args, **kwargs)


def traded_picks_for(*args, **kwargs):
    from dashboard_services.platform_api import get_traded_picks as _fn
    return _fn(*args, **kwargs)


def league_format_values(ctx, force_scoring_type=None):
    from dashboard_services.ai.context_builders import league_format_value_lookup as _fn
    return _fn(ctx, force_scoring_type=force_scoring_type)


def roster_grade(players, future_picks, **kwargs):
    from dashboard_services.ai.context_builders import calculate_roster_grade as _fn
    return _fn(players, future_picks, **kwargs)


def rank_positions(team_pos_values, slot_counts, positions=None):
    from utils.trade import rank_rosters_by_position as _fn
    if positions:
        return _fn(team_pos_values, slot_counts, positions=positions)
    return _fn(team_pos_values, slot_counts)
def compute_league_keepers(*args, **kwargs):
    from dashboard_services.pages.keeper_page import compute_league_keepers as _fn
    return _fn(*args, **kwargs)


def keeper_league_limit(ctx):
    from dashboard_services.pages.keeper_page import league_keeper_limit as _fn
    return _fn(ctx)


def dynasty_without_keepers(ctx):
    from dashboard_services.pages.keeper_page import is_dynasty_without_keepers as _fn
    return _fn(ctx)


def rookie_rankings(draft_year):
    from dashboard_services.rookie_api import _get_rankings as _fn
    return _fn(draft_year)


def prospect_detail(player_id, row):
    from dashboard_services.rookie_api import _get_prospect_detail as _fn
    return _fn(player_id, row)


def active_rookie_class():
    from data_building.rookie_pipeline.pipeline import get_active_rookie_class as _fn
    return _fn()


def list_account_leagues(account_id):
    from dashboard_services.accounts import list_user_leagues as _fn
    return _fn(account_id)


def league_summary(account_id, membership, context_loader):
    from dashboard_services.portfolio_summary import build_league_summary as _fn
    return _fn(account_id, membership, context_loader)


def league_ctx_loader(platform, league_id, season):
    from dashboard_services.assistant.context import get_league_ctx_from_cache as _fn
    return _fn(platform, str(league_id), season)


def resolve_account_id():
    from dashboard_services.assistant.context import resolve_account_id as _fn
    return _fn()


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

    def _raw_proj(pid, position):
        if not week_map:
            return None
        try:
            from utils.fantasy_scoring import weekly_projection_points
            pts = weekly_projection_points(week_map, str(pid), scoring, position or "")
            return round(float(pts), 1) if pts is not None else None
        except Exception:
            return None

    def _entry_for(pid):
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
        proj = _raw_proj(pid, entry.get("position"))
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
        return entry

    if player_ids:
        players = [_entry_for(pid) for pid in player_ids[:3]]
    else:
        # No players named: the question is about the asking user's own
        # lineup (a slot, or the lineup as a whole), so the candidates
        # come from their roster, not from a decline.
        players = _roster_start_sit(bundle, entities, pack, tools,
                                    _entry_for, _raw_proj)
    pack["players"] = players
    scored = [p for p in players if p.get("start_score") is not None]
    key = "start_score" if scored else "value"
    ranked = sorted(players, key=lambda p: (p.get(key) is not None, p.get(key) or 0), reverse=True)
    if ranked:
        pack["pick_player_id"] = ranked[0]["player_id"]
    # Roster mode: the lineup solver's pick at the asked slot outranks the
    # raw score order (it sees the whole lineup's slot constraints).
    rec = pack.get("recommended_player_id")
    if rec and any(p.get("player_id") == rec for p in players):
        pack["pick_player_id"] = rec
    if len(ranked) >= 2 and ranked[0].get(key) is not None and ranked[1].get(key) is not None:
        gap = abs(float(ranked[0][key]) - float(ranked[1][key]))
        base = max(abs(float(ranked[0][key])), 1.0)
        pack["margin"] = {"gap": round(gap, 2), "basis": key, "thin": bool(gap / base < 0.10)}
    else:
        pack["margin"] = {"thin": False}
    if not players and pack.get("available") is not False:
        pack["available"] = False
        pack["missing"] = "no resolved players"
    return pack, tools


def _roster_start_sit(bundle, entities, pack, tools, entry_for, raw_proj):
    """Roster-scoped start/sit: candidates from the asking user's roster.

    The lineup math is the site's own solver (utils.lineups
    assign_optimal_lineup / assign_fixed_lineup, the machinery behind
    projection_upgrades, the Lineup Lab, and the recap) run over this
    week's projection map, the same map the named-player path scores
    from. Slot questions scope to that slot's real decision set (the
    current occupant plus the bench players who could take it; for a
    base position slot, every rostered player at the position). A
    who-to-sit question mirrors it: the starter the solver benches for
    the biggest gain. Returns candidate entries in start-score order,
    or [] with the pack marked unavailable when no roster answer exists.
    """
    entities = entities or {}
    roster = bundle.get("roster") or {}
    ctx = bundle.get("ctx") or {}
    roster_positions = ctx.get("roster_positions") or []
    if hasattr(roster_positions, "tolist"):
        roster_positions = list(roster_positions)
    starters = [str(p) for p in (roster.get("starters") or [])
                if str(p) not in ("", "0")]
    reserve = {str(p) for p in (roster.get("reserve") or [])}
    taxi = {str(p) for p in (roster.get("taxi") or [])}
    eligible = [str(p) for p in (roster.get("players") or [])
                if str(p) not in reserve and str(p) not in taxi]
    if not eligible or not starters or not roster_positions:
        pack["available"] = False
        pack["missing"] = ("no linked team: link your team and I can make "
                           "lineup calls from your roster")
        return []
    index = _index()
    pos_map: dict[str, str] = {}
    proj_map: dict[str, float] = {}
    for pid in eligible:
        pos_map[pid] = _player_info(pid, index).get("position") or ""
        proj = raw_proj(pid, pos_map[pid])
        if proj is not None:
            proj_map[pid] = proj
    if not proj_map:
        pack["available"] = False
        pack["missing"] = "no projections are available for your roster this week"
        return []
    try:
        from utils.lineup_slots import canonicalize_slot, slot_eligible_positions
        from utils.lineups import assign_fixed_lineup, assign_optimal_lineup
        optimal = assign_optimal_lineup(proj_map, pos_map, roster_positions,
                                        list(proj_map), prefer_starters=starters)
        actual = assign_fixed_lineup(starters, pos_map, roster_positions) or []
        tools.append("utils.lineups.assign_optimal_lineup")
    except Exception:
        logger.debug("[assistant] roster start/sit solver failed", exc_info=True)
        pack["available"] = False
        pack["missing"] = "lineup data is not available for your roster right now"
        return []
    slots = optimal.get("slots") or []
    opt_assign = optimal.get("assignment") or []
    starter_set = set(starters)
    mode = "sit" if str(entities.get("start_mode") or "").lower() == "sit" else "start"
    pack["roster_mode"] = True
    pack["question_mode"] = mode

    def _at(assign, i):
        return assign[i] if 0 <= i < len(assign) else None

    # Per-slot decisions: the optimal lineup benches a current starter
    # for a bench player at that slot.
    decisions = []
    for i, slot in enumerate(slots):
        inc = _at(opt_assign, i)
        out = _at(actual, i)
        if inc and inc not in starter_set:
            decisions.append({
                "slot": slot, "in_player_id": inc, "out_player_id": out,
                "gain": round(proj_map.get(inc, 0.0)
                              - (proj_map.get(out, 0.0) if out else 0.0), 1),
            })
    decisions.sort(key=lambda d: -d["gain"])

    slot_raw = str(entities.get("slot") or "").strip().upper().replace(" ", "_")
    target_idx = None
    occupant = recommended = None
    if slot_raw:
        digits = ""
        while slot_raw and slot_raw[-1].isdigit():
            digits = slot_raw[-1] + digits
            slot_raw = slot_raw[:-1]
        canon = canonicalize_slot(slot_raw) or slot_raw
        pack["slot_label"] = canon + digits
        matches = [i for i, s in enumerate(slots) if s == canon]
        if matches:
            if digits and 1 <= int(digits) <= len(matches):
                target_idx = matches[int(digits) - 1]
            else:
                target_idx = matches[-1]

    if target_idx is not None:
        slot = slots[target_idx]
        pack["slot"] = slot
        occupant = _at(actual, target_idx)
        recommended = _at(opt_assign, target_idx)
        elig = slot_eligible_positions(slot)
        if slot in ("FLEX", "SUPER_FLEX", "RB_WR", "WR_TE", "RB_TE"):
            # The flex decision set: the bench players who could take
            # the slot, plus the player currently in it.
            cand_ids = [pid for pid in eligible
                        if pid not in starter_set
                        and pos_map.get(pid) in elig and pid in proj_map]
            if occupant:
                cand_ids.append(occupant)
        else:
            cand_ids = [pid for pid in eligible
                        if pos_map.get(pid) in elig and pid in proj_map]
        pack["decisions"] = [d for d in decisions if d["slot"] == slot]
    else:
        pack["slot"] = None
        involved: list[str] = []
        for d in decisions:
            for pid in (d["in_player_id"], d["out_player_id"]):
                if pid and pid not in involved:
                    involved.append(pid)
        cand_ids = involved or list(starters)
        pack["decisions"] = decisions
        if decisions:
            recommended = decisions[0]["in_player_id"]
            occupant = decisions[0]["out_player_id"]
            pack["slot"] = decisions[0]["slot"]
            pack["slot_label"] = decisions[0]["slot"]

    if mode == "sit":
        sit_pid = None
        if target_idx is not None:
            if occupant and recommended and recommended != occupant:
                sit_pid = occupant
        elif decisions:
            sit_pid = decisions[0]["out_player_id"]
        if sit_pid:
            pack["sit_player_id"] = sit_pid
            for d in pack["decisions"]:
                if d["out_player_id"] == sit_pid:
                    recommended = d["in_player_id"]
                    break
    if recommended:
        pack["recommended_player_id"] = recommended

    entries = {}
    for pid in cand_ids:
        if pid in proj_map:
            entries[pid] = entry_for(pid)
    for pid, entry in entries.items():
        entry["current_starter"] = pid in starter_set
        if occupant and pid == occupant:
            entry["slot_occupant"] = True

    def _rank_key(pid):
        entry = entries[pid]
        for field in ("start_score", "projection", "value"):
            if entry.get(field) is not None:
                return (1, float(entry[field]))
        return (0, 0.0)

    ordered = sorted(entries, key=_rank_key, reverse=True)
    keep = ordered[:5]
    for pid in (recommended, occupant):
        if pid and pid in entries and pid not in keep:
            keep.append(pid)
    if not keep:
        pack["available"] = False
        pack["missing"] = "no startable players were found on your roster for that slot"
        return []
    return [entries[pid] for pid in keep]


# -- league machinery: the Teams page's strengths, ranks, and grades -----------

CORE_POS = ("QB", "RB", "WR", "TE")


def _team_name(ctx, roster, users=None):
    """Team display name: ctx roster_map, then roster metadata, then owner."""
    rid = roster.get("roster_id")
    rmap = (ctx or {}).get("roster_map") or {}
    name = rmap.get(str(rid)) or rmap.get(rid)
    if name:
        return str(name)
    name = str((roster.get("metadata") or {}).get("team_name") or "")
    if name:
        return name
    if users is None:
        users = {str(u.get("user_id")): u for u in (ctx.get("users") or []) if isinstance(u, dict)}
    user = users.get(str(roster.get("owner_id") or "")) or {}
    return str(user.get("display_name") or user.get("username") or "") or f"Roster {rid}"


def _slot_counts(ctx):
    counts: dict[str, int] = {}
    for slot in (ctx or {}).get("roster_positions") or []:
        s = str(slot)
        counts[s] = counts.get(s, 0) + 1
    return counts


def _league_is_redraft(ctx) -> bool:
    try:
        from dashboard_services.ai.context_builders import ctx_scoring_type
        return ctx_scoring_type(ctx or {}) == "redraft"
    except Exception:
        return False


def _resolve_mode(ctx, requested) -> str:
    """Teams grading mode: a redraft league has only This Season; a dynasty
    league defaults to Dynasty (the Teams page's default tab)."""
    if _league_is_redraft(ctx):
        return "this_season"
    return "this_season" if requested == "this_season" else "dynasty"


def _mode_values(ctx, mode):
    """pid -> value row for one Teams mode, via league_format_value_lookup.

    Dynasty mode is the lookup's natural output for a dynasty league; This
    Season forces redraft values, exactly like the Teams page toggle. Falls
    back to the app value-table cache when the ctx carries no table.
    """
    eff = dict(ctx or {})
    if not eff.get("model_value_table"):
        rows = list(_value_rows().values())
        if rows:
            eff["model_value_table"] = rows
    if not eff.get("model_value_table"):
        return {}
    force = "redraft" if mode == "this_season" else None
    try:
        return league_format_values(eff, force_scoring_type=force) or {}
    except Exception:
        logger.debug("[assistant] league format values failed", exc_info=True)
        return {}


def _league_position_data(ctx, rosters, mode, values=None, index=None):
    """Positional strengths, ranks, and roster grades for every roster.

    The Teams page pipeline: per-roster position value lists from the
    mode's league-format values, strengths and ranks from
    rank_rosters_by_position, grades from calculate_roster_grade with the
    page's percentile inputs (top-8 dynasty/redraft totals, top-10
    dynasty/redraft ratio). When the league-format lookup is unavailable
    (no model table anywhere), strengths and ranks fall back to the raw
    value rows the caller passed, and grades are omitted rather than
    computed from partial inputs.
    """
    ctx = ctx or {}
    num_teams = len(rosters or [])
    if not num_teams:
        return None
    users = {str(u.get("user_id")): u for u in (ctx.get("users") or []) if isinstance(u, dict)}
    team_names = {str(r.get("roster_id")): _team_name(ctx, r, users) for r in rosters}
    slot_counts = _slot_counts(ctx)
    by_id = _mode_values(ctx, mode)
    source = "league_format" if by_id else "raw_values"
    vals_by_roster: dict[str, dict[str, list[float]]] = {}
    flat_by_roster: dict[str, list[dict]] = {}
    for r in rosters:
        rid = str(r.get("roster_id"))
        vals: dict[str, list[float]] = {pos: [] for pos in CORE_POS}
        flat: list[dict] = []
        for pid in r.get("players") or []:
            spid = str(pid)
            if by_id:
                row = by_id.get(spid) or {}
                pos = str(row.get("position") or row.get("pos") or "").upper()
                if not pos:
                    pos = str(_player_info(spid, index if index is not None else _index()).get("position") or "").upper()
                try:
                    v = float(row.get("value") or 0)
                except (TypeError, ValueError):
                    v = 0.0
                age = row.get("age")
            else:
                info = _player_info(spid, index if index is not None else _index())
                pos = str(info.get("position") or "").upper()
                v = _row_value((values or {}).get(spid)) or 0.0
                age = None
            if pos in CORE_POS and v > 0:
                vals[pos].append(v)
                flat.append({"position": pos, "value": v, "age": age})
        flat.sort(key=lambda x: x["value"], reverse=True)
        vals_by_roster[rid] = vals
        flat_by_roster[rid] = flat
    try:
        strengths_raw, ranks_raw = rank_positions(vals_by_roster, slot_counts, positions=list(CORE_POS))
        strengths = {str(rid): {pos: round(float((per or {}).get(pos) or 0), 1)
                                for pos in CORE_POS}
                     for rid, per in (strengths_raw or {}).items()}
        ranks = {pos: {str(rid): int(rk) for rid, rk in (per or {}).items()}
                 for pos, per in (ranks_raw or {}).items()}
    except Exception:
        logger.debug("[assistant] position ranks failed", exc_info=True)
        return None
    grades: dict[str, dict] = {}
    if source == "league_format":
        try:
            grades = _roster_grades(ctx, rosters, by_id, flat_by_roster, ranks, mode, num_teams)
        except Exception:
            logger.debug("[assistant] roster grades failed", exc_info=True)
            grades = {}
    return {"team_names": team_names, "strengths": strengths, "ranks": ranks,
            "vals_by_roster": vals_by_roster, "grades": grades, "mode": mode,
            "source": source, "num_teams": num_teams}


def _roster_grades(ctx, rosters, by_id, flat_by_roster, ranks, mode, num_teams):
    """calculate_roster_grade per roster with the Teams page's inputs.

    Percentile totals are computed from the league's natural (dynasty)
    values in both modes, exactly as teams_page does; only the graded
    player values, the position ranks, and the scoring_type follow the
    mode. This Season grades carry no draft capital, like the page.
    """
    from utils.lineup_slots import is_superflex_lineup
    roster_positions = (ctx or {}).get("roster_positions") or []
    is_sf = is_superflex_lineup(roster_positions)
    redraft_key = "redraft_value_sf" if is_sf else "redraft_value_1qb"
    natural = by_id if mode == "dynasty" else _mode_values(ctx, "dynasty")
    dyn_totals: dict[str, float] = {}
    red_totals: dict[str, float] = {}
    dr_ratios: dict[str, float] = {}
    for r in rosters:
        rid = str(r.get("roster_id"))
        pairs = []
        for pid in r.get("players") or []:
            row = (natural or {}).get(str(pid)) or {}
            pos = str(row.get("position") or row.get("pos") or "").upper()
            if pos not in CORE_POS:
                continue
            try:
                dval = float(row.get("value") or 0)
                rval = float(row.get(redraft_key) or 0)
            except (TypeError, ValueError):
                continue
            pairs.append((dval, rval))
        pairs.sort(reverse=True)
        dyn_totals[rid] = sum(d for d, _ in pairs[:8])
        red_totals[rid] = sum(rv for _, rv in pairs[:8])
        ratios = [d / max(rv, 1) for d, rv in pairs[:10] if d > 50 or rv > 50]
        dr_ratios[rid] = round(sum(ratios) / len(ratios), 3) if ratios else 1.0

    def _pct(totals, rid):
        vals = sorted(totals.values())
        denom = max(len(vals) - 1, 1)
        return sum(1 for v in vals if v < totals.get(rid, 0.0)) / denom

    picks_by_roster = (ctx or {}).get("picks_by_roster") or {}
    league_redraft = _league_is_redraft(ctx)
    grades: dict[str, dict] = {}
    for r in rosters:
        rid = str(r.get("roster_id"))
        picks = [] if (mode == "this_season" or league_redraft) else picks_by_roster.get(rid, [])
        p_ranks = {pos: (ranks.get(pos) or {}).get(rid, num_teams) for pos in CORE_POS}
        g = roster_grade(
            flat_by_roster.get(rid, []), picks,
            position_ranks=p_ranks, num_teams=num_teams,
            dynasty_pct_val=_pct(dyn_totals, rid), redraft_pct_val=_pct(red_totals, rid),
            dr_ratio=dr_ratios.get(rid, 1.0),
            scoring_type="redraft" if (mode == "this_season" or league_redraft) else "dynasty",
        )
        if isinstance(g, dict) and g.get("grade") is not None:
            grades[rid] = {"grade": g.get("grade"),
                           "score": round(float(g.get("score") or 0), 1),
                           "win_window": g.get("win_window"),
                           "formula_id": g.get("formula_id")}
    return grades


# -- trade evaluation ------------------------------------------------------------

_ROUND_WORDS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
                "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}


def _parse_pick_text(text):
    """Parse a pick mention into {season, round, slot} or None.

    Handles "2027 1st", "my 2027 first-round pick", "2027 1.03", and the
    canonical id form "2027_1_03" / "2027_1_early". Slot stays None unless
    the text itself carries one (canonical id or 1.03 style).
    """
    import re as _re
    s = str(text or "").strip().lower()
    if not s:
        return None
    slot = None
    m = _re.search(r"(20\d{2})_(\d{1,2})_(\d{1,2}|early|mid|late)", s)
    if m:
        third = m.group(3)
        return {"season": int(m.group(1)), "round": int(m.group(2)),
                "slot": int(third) if third.isdigit() else None}
    m = _re.search(r"(20\d{2})", s)
    if not m:
        return None
    season = int(m.group(1))
    rnd = None
    m2 = _re.search(r"\b(\d)\.(\d{1,2})\b", s)
    if m2:
        rnd, slot = int(m2.group(1)), int(m2.group(2))
    else:
        m3 = _re.search(r"\bround\s*(\d)\b", s) or _re.search(r"\b(\d)(?:st|nd|rd|th)\b", s)
        if m3:
            rnd = int(m3.group(1))
        else:
            for word, num in _ROUND_WORDS.items():
                if _re.search(rf"\b{word}\b", s):
                    rnd = num
                    break
    if rnd is None:
        return None
    return {"season": season, "round": rnd, "slot": slot}


def _pick_ownership(bundle):
    """Draft-pick ownership for the bundle's league, or None.

    Primary source: the ctx's picks_by_roster (app.py builds it from the
    provider's traded picks, so it already reflects every trade). Fallback:
    platform_api.get_traded_picks fed through build_picks_by_roster, the
    same builder app.py uses. Returns {by_pick, owned, team_names,
    num_teams, source}: by_pick maps (season, round, original_roster_id)
    to the current owner's roster id; owned maps an owner roster id to its
    pick entries {season, round, original_owner}.
    """
    ctx = bundle.get("ctx") or {}
    pbr = ctx.get("picks_by_roster") or {}
    source = "ctx.picks_by_roster"
    if not pbr and bundle.get("league_id"):
        try:
            traded = traded_picks_for(bundle.get("platform"), bundle.get("league_id"),
                                      bundle.get("season")) or []
            if traded:
                from dashboard_services.service import build_picks_by_roster
                league = ctx.get("league") or {"season": bundle.get("season")}
                pbr = build_picks_by_roster(
                    num_future_seasons=3, league=league,
                    rosters=ctx.get("rosters") or [], traded=traded,
                    draft_ended=bool(bundle.get("week"))) or {}
                source = "platform_api.get_traded_picks"
        except Exception:
            logger.debug("[assistant] traded picks fallback failed", exc_info=True)
            pbr = {}
    if not pbr:
        return None
    rosters = [r for r in (ctx.get("rosters") or []) if isinstance(r, dict)]
    users = {str(u.get("user_id")): u for u in (ctx.get("users") or []) if isinstance(u, dict)}
    team_names = {str(r.get("roster_id")): _team_name(ctx, r, users) for r in rosters}
    by_pick: dict = {}
    owned: dict[str, list[dict]] = {}
    for owner_rid, plist in pbr.items():
        for pk in plist or []:
            if not isinstance(pk, dict):
                continue
            try:
                season, rnd = int(pk.get("season") or 0), int(pk.get("round") or 0)
            except (TypeError, ValueError):
                continue
            if not season or not rnd:
                continue
            original = str(pk.get("original_owner") or "")
            by_pick[(season, rnd, original)] = str(owner_rid)
            owned.setdefault(str(owner_rid), []).append(
                {"season": season, "round": rnd, "original_owner": original})
    if not by_pick:
        return None
    return {"by_pick": by_pick, "owned": owned, "team_names": team_names,
            "num_teams": len(rosters) or len(pbr), "source": source}


def _pick_slots(ctx, ownership, season, rnd):
    """{original_roster_id: slot} for one season + round.

    The site convention (Front Office pick projection): original owners
    ordered by record, worst record picks first (slot 1). Records come
    from roster settings; the standings seed map is the fallback (seed 1
    is the best team). Ties break by roster id so the order is stable.
    """
    originals = {orig for (s, r, orig) in (ownership or {}).get("by_pick") or {}
                 if s == season and r == rnd and orig}
    if not originals:
        return {}
    ctx = ctx or {}
    settings = {str(r.get("roster_id")): (r.get("settings") or {})
                for r in (ctx.get("rosters") or []) if isinstance(r, dict)}
    standings = ctx.get("standings_map") or {}
    n = max(len(originals), 1)

    def _win_pct(rid):
        st = settings.get(str(rid)) or {}
        try:
            w, l = float(st.get("wins") or 0), float(st.get("losses") or 0)
        except (TypeError, ValueError):
            w, l = 0.0, 0.0
        if w + l > 0:
            return w / (w + l)
        seed = standings.get(str(rid))
        if seed is None and str(rid).isdigit():
            seed = standings.get(int(rid))
        if isinstance(seed, (int, float)) and seed:
            return (n - float(seed)) / n
        return 0.5

    ordered = sorted(originals, key=lambda rid: (_win_pct(rid), str(rid)))
    return {rid: i + 1 for i, rid in enumerate(ordered)}


def _pick_entry(bundle, ownership, table, parsed, *, side, current_owner_rid, original_rid):
    """One attributed, priced pick for a pack.

    The canonical pick id carries the ORIGINAL owner's slot (the site
    convention); when no slot can be derived the id uses the mid bucket,
    the same default the site's pick_label falls back to. Value comes from
    the pick value table via pick_value_from_table (exact slot, then
    bucket, then round average). Unpriced picks stay in the pack with
    priced False and value None, never an invented number.
    """
    from utils.draft import pick_label, pick_value_from_table
    season, rnd = parsed["season"], parsed["round"]
    ctx = bundle.get("ctx") or {}
    slot = parsed.get("slot")
    if slot is None and ownership and original_rid:
        slot = _pick_slots(ctx, ownership, season, rnd).get(str(original_rid))
    num_teams = (ownership or {}).get("num_teams") or len(ctx.get("rosters") or []) or 10
    pick_id = f"{season}_{rnd}_{int(slot):02d}" if slot else f"{season}_{rnd}_mid"
    value = None
    if table:
        try:
            v = pick_value_from_table(table, season, rnd, slot, num_teams=num_teams)
            if v:
                value = round(float(v), 1)
        except Exception:
            value = None
    team_names = (ownership or {}).get("team_names") or {}
    return {
        "pick_id": pick_id, "label": pick_label(season, rnd, slot),
        "season": season, "round": rnd, "slot": slot, "side": side,
        "original_owner_team": team_names.get(str(original_rid)) if original_rid else None,
        "current_owner_team": team_names.get(str(current_owner_rid)) if current_owner_rid else None,
        "value": value, "priced": value is not None,
    }


def _attribute_pick(bundle, ownership, parsed, text, *, side, viewer_rid, partner_rid):
    """Resolve (original_rid, current_owner_rid) for one mentioned pick.

    Current owner is definitional for a proposed deal: give-side picks are
    the viewer's, receive-side picks are the partner's. Original owner: a
    team named in the mention wins; otherwise the current owner's holdings
    for that season and round decide (their own pick preferred when they
    hold several); with no holdings data the pick is the current owner's
    own. Without ownership data at all, only the side's owner is known.
    """
    low = str(text or "").lower()
    current = viewer_rid if side == "give" else partner_rid
    if not ownership:
        return current, current
    team_names = ownership.get("team_names") or {}
    for rid, name in team_names.items():
        if name and len(name) >= 3 and name.lower() in low:
            return rid, current
    if current:
        holdings = [p for p in (ownership.get("owned") or {}).get(str(current), [])
                    if p["season"] == parsed["season"] and p["round"] == parsed["round"]]
        if len(holdings) == 1:
            return holdings[0]["original_owner"] or current, current
        if len(holdings) > 1:
            own = [p for p in holdings if p["original_owner"] == str(current)]
            chosen = (own or holdings)[0]
            return chosen["original_owner"] or current, current
        return current, current
    # No current owner (receive side, partner unknown): a lone league-wide
    # candidate for that season and round is still attributable.
    cands = [orig for (s, r, orig) in ownership.get("by_pick") or {}
             if s == parsed["season"] and r == parsed["round"] and orig]
    if len(cands) == 1:
        return cands[0], (ownership.get("by_pick") or {}).get(
            (parsed["season"], parsed["round"], cands[0]))
    return None, None


def _side_picks(bundle, entities, ownership, table, *, viewer_rid, partner_rid, partner_name):
    """Attributed pick entries per side for a trade pack.

    Router give_picks / receive_picks come first; un-sided mentions in the
    flat picks list are attributed by phrasing ("my" gives, "his/their" or
    the partner's name receives, otherwise the give side, where pick
    mentions in a proposal normally sit). Unparseable mentions are
    returned as unpriced labels so the answer can flag them, never priced.
    """
    import re as _re
    ent = entities or {}
    plan: dict[str, list[str]] = {"give": list(ent.get("give_picks") or []),
                                  "receive": list(ent.get("receive_picks") or [])}
    for text in ent.get("picks") or []:
        low = str(text or "").lower()
        if _re.search(r"\bmy\b", low):
            plan["give"].append(text)
        elif partner_rid and (_re.search(r"\bhis\b|\bher\b|\btheir\b|\bhim\b", low)
                              or (partner_name and partner_name.lower() in low)):
            plan["receive"].append(text)
        else:
            plan["give"].append(text)
    out: dict[str, list[dict]] = {"give": [], "receive": []}
    unpriced_labels: list[str] = []
    for side in ("give", "receive"):
        for text in plan[side]:
            parsed = _parse_pick_text(text)
            if parsed is None:
                unpriced_labels.append(str(text))
                continue
            original, current = _attribute_pick(
                bundle, ownership, parsed, text, side=side,
                viewer_rid=viewer_rid, partner_rid=partner_rid)
            entry = _pick_entry(bundle, ownership, table, parsed, side=side,
                                current_owner_rid=current, original_rid=original)
            out[side].append(entry)
            if not entry["priced"]:
                unpriced_labels.append(entry["label"])
    return out, unpriced_labels


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
    # Roster fit, partner fit, and a balancing asset, all from the trade
    # targets machinery (utils.trade): league position ranks from weighted
    # position strength, thresholds/floors derived from the league's own
    # roster settings, needs/surplus classified per roster.
    ctx = bundle.get("ctx") or {}
    roster = bundle.get("roster") or {}
    rosters = [r for r in (ctx.get("rosters") or []) if isinstance(r, dict)]
    viewer_rid = str(roster.get("roster_id") or "")
    partner = None
    partner_rid = None
    fit = None
    if rosters:
        try:
            fit = _league_fit(ctx, rosters, values, index)
        except Exception:
            logger.debug("[assistant] trade fit failed", exc_info=True)
            fit = None
        if fit:
            tools.append("utils.trade.roster_fit")
            if viewer_rid in fit["by_roster"]:
                mine = fit["by_roster"][viewer_rid]
                pack["roster_fit"] = {
                    "surplus_positions": mine["surplus"],
                    "needed_positions": mine["needs"],
                }
                if mine.get("grade") is not None:
                    pack["roster_fit"]["grade"] = mine["grade"]
                    pack["roster_fit"]["grade_score"] = mine["grade_score"]
                    pack["roster_fit"]["grade_mode"] = mine.get("grade_mode")
            partner = _identify_partner(bundle, entities, fit, give_ids, recv_ids)
            if partner is not None:
                partner_rid = next((rid for rid, e in fit["by_roster"].items() if e is partner), None)
                pack["partner"] = {
                    "team_name": partner["team_name"],
                    "needed_positions": partner["needs"],
                    "surplus_positions": partner["surplus"],
                }
                if partner.get("grade") is not None:
                    pack["partner"]["grade"] = partner["grade"]
                    pack["partner"]["grade_score"] = partner["grade_score"]
                tools.append("utils.trade.partner_fit")
    # Picks: priced AND attributed. Ownership (whose pick it originally is,
    # who holds it now) comes from the league's pick ownership data; values
    # from the pick value table. Priced pick values count in the side
    # totals, so the fairness verdict sees the whole deal. The per-side
    # lists carry canonical pick ids: the exact shape the trade
    # calculator's deep link stages (receive_picks -> ap, give_picks -> bp).
    ent = entities or {}
    if ent.get("picks") or ent.get("give_picks") or ent.get("receive_picks"):
        try:
            ownership = _pick_ownership(bundle)
        except Exception:
            logger.debug("[assistant] pick ownership failed", exc_info=True)
            ownership = None
        table = None
        try:
            from utils.lineup_slots import is_superflex_lineup
            table = pick_value_table(league_teams=len(rosters) or 10,
                                     is_sf=is_superflex_lineup(ctx.get("roster_positions") or []))
            if table:
                tools.append("dashboard_services.picks.load_pick_value_table")
        except Exception:
            logger.debug("[assistant] pick value table failed", exc_info=True)
            table = None
        if ownership:
            tools.append(ownership["source"])
        side_picks, unpriced_pick_labels = _side_picks(
            bundle, ent, ownership, table, viewer_rid=viewer_rid or None,
            partner_rid=partner_rid, partner_name=(partner or {}).get("team_name"))
        pack["give_picks"] = side_picks["give"]
        pack["receive_picks"] = side_picks["receive"]
        pack["picks"] = side_picks["give"] + side_picks["receive"]
        for label in unpriced_pick_labels:
            if label not in pack["unpriced"]:
                pack["unpriced"].append(label)
        give_total = round(give_total + sum(p["value"] for p in side_picks["give"] if p["value"]), 1)
        recv_total = round(recv_total + sum(p["value"] for p in side_picks["receive"] if p["value"]), 1)
        pack["give"]["total_value"] = give_total
        pack["receive"]["total_value"] = recv_total
        pack["fairness_gap"] = round(recv_total - give_total, 1)
    # The balancing asset is computed last, on the final gap (picks included).
    if partner is not None and fit:
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
    """Per-roster needs/surplus plus grades, from the Teams machinery.

    Strengths and ranks come from _league_position_data
    (rank_rosters_by_position over league-format values); needs/surplus are
    classified from those ranks with thresholds/floors from
    derive_league_thresholds on the league settings, as before. Roster
    grades (calculate_roster_grade) ride along when the league-format
    values are available.
    """
    from utils.trade import (
        POSITIONS, classify_position_needs, derive_league_thresholds,
        detect_surplus_positions,
    )
    roster_positions = [str(s) for s in (ctx.get("roster_positions") or [])]
    is_sf = any("SUPER" in slot.upper() for slot in roster_positions)
    num_teams = len(rosters)
    thresholds, floors = derive_league_thresholds(roster_positions, num_teams, is_sf=is_sf)
    mode = _resolve_mode(ctx, None)
    data = _league_position_data(ctx, rosters, mode, values=values, index=index)
    if data is None:
        return None
    by_roster: dict[str, dict] = {}
    for r in rosters:
        rid = str(r.get("roster_id"))
        ranks = {pos: (data["ranks"].get(pos) or {}).get(rid, num_teams) for pos in POSITIONS}
        entry = {
            "team_name": data["team_names"].get(rid) or f"Roster {rid}",
            "needs": [pos for pos, _kind in classify_position_needs(
                ranks, data["vals_by_roster"][rid], num_teams, thresholds, floors)],
            "surplus": detect_surplus_positions(
                ranks, data["vals_by_roster"][rid], num_teams, thresholds, floors),
            "player_ids": [str(p) for p in (r.get("players") or [])],
        }
        grade = (data.get("grades") or {}).get(rid)
        if grade:
            entry["grade"] = grade["grade"]
            entry["grade_score"] = grade["score"]
            entry["grade_mode"] = mode
        by_roster[rid] = entry
    return {"by_roster": by_roster, "num_teams": num_teams, "grade_mode": mode}


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


# -- trade finding -------------------------------------------------------------

def _finding_player(row, index):
    """Slim a value-table row or matcher asset to the finding player shape."""
    pid = str(row.get("player_id") or row.get("id") or "")
    info = _player_info(pid, index) if pid else {}
    return {
        "player_id": pid,
        "name": row.get("name") or info.get("name") or pid,
        "team": row.get("team") or info.get("team") or "",
        "position": row.get("position") or info.get("position") or "",
        "value": _row_value(row),
    }


def _asset_pool(roster, values, index):
    """A roster's priced players in the shape value_matched_acquire_packages
    consumes (and returns as package assets), plus per-position counts."""
    pool, counts = [], {}
    for pid in (roster or {}).get("players") or []:
        pid = str(pid)
        entry = _finding_player({**(values.get(pid) or {}), "player_id": pid}, index)
        if not entry["value"]:
            continue
        pool.append({"player_id": pid, "id": pid, "name": entry["name"],
                     "position": entry["position"], "value": entry["value"]})
        counts[entry["position"]] = counts.get(entry["position"], 0) + 1
    return pool, counts


def _finding(partner_roster_id, partner_team, give, receive, partner_info,
             viewer_needs, fairness=None, acceptance_rate=None):
    give_total = round(sum(p["value"] or 0 for p in give), 1)
    receive_total = round(sum(p["value"] or 0 for p in receive), 1)
    partner_needs = (partner_info or {}).get("needs") or []
    partner_surplus = (partner_info or {}).get("surplus") or []
    return {
        "partner_roster_id": str(partner_roster_id),
        "partner_team": partner_team,
        "give": give, "receive": receive,
        "give_total": give_total, "receive_total": receive_total,
        "fairness_gap": round(receive_total - give_total, 1),
        "fairness": fairness,
        "acceptance_rate": acceptance_rate,
        "partner_needs": partner_needs, "partner_surplus": partner_surplus,
        "you_fill": sorted({p["position"] for p in give if p.get("position")}
                           & set(partner_needs)),
        "they_fill": sorted({p["position"] for p in receive if p.get("position")}
                            & set(viewer_needs or [])),
    }


def _need_findings(ctx, bundle, target_position, index):
    """Need mode: the Suggestions engine's own computed partner packages."""
    tools = ["ai.context_builders.build_trade_suggestions_context"]
    try:
        sugg = build_trade_suggestions_context(ctx, str(bundle["roster"]["roster_id"]))
    except Exception:
        sugg = None
    if not sugg:
        return [], None, tools, [], []
    viewer_needs = sugg.get("viewer_needs") or []
    viewer_surplus = sugg.get("viewer_surplus") or []
    roster_map = ctx.get("roster_map") or {}

    def _shape(p):
        give = [_finding_player(t, index) for t in (p.get("targets_viewer_sends") or [])]
        receive = [_finding_player(t, index) for t in (p.get("targets_they_have") or [])]
        f = _finding(p.get("roster_id"), p.get("team_name") or roster_map.get(str(p.get("roster_id"))),
                     give, receive,
                     {"needs": p.get("partner_needs") or [],
                      "surplus": p.get("partner_surplus") or []},
                     viewer_needs, fairness=p.get("fairness"),
                     acceptance_rate=p.get("acceptance_rate"))
        # The engine's own side totals are the analyzer's numbers; they can
        # differ from a re-sum when it priced a piece the slim row dropped.
        if p.get("value_you_give") is not None:
            f["give_total"] = float(p["value_you_give"])
        if p.get("value_you_get") is not None:
            f["receive_total"] = float(p["value_you_get"])
        f["fairness_gap"] = round(f["receive_total"] - f["give_total"], 1)
        return f

    partners = [p for p in (sugg.get("top_partners") or [])
                if (p.get("targets_they_have") or []) and (p.get("targets_viewer_sends") or [])]
    shaped = [_shape(p) for p in partners]
    if not target_position:
        return shaped, None, tools, viewer_needs, viewer_surplus
    wanted = {"RB", "WR", "TE"} if target_position == "FLEX" else {target_position}
    matched = [f for f in shaped
               if any(p.get("position") in wanted for p in f["receive"])]
    miss = shaped[0] if (not matched and shaped) else None
    return matched, miss, tools, viewer_needs, viewer_surplus


def _sell_findings(ctx, bundle, sell_entry, values, index):
    """Sell mode: what each team that needs the player's position could send
    back, value-matched by the acquire-package matcher and held to the
    analyzer's fairness band."""
    from utils.trade_value import fair_value_band
    tools = ["trade_acquire_packages.value_matched_acquire_packages",
             "utils.trade.classify_position_needs",
             "utils.trade.detect_surplus_positions",
             "utils.trade_value.fair_value_band"]
    rosters = ctx.get("rosters") or []
    fit = _league_fit(ctx, rosters, values, index)
    by_roster = fit.get("by_roster") or {}
    viewer_rid = str((bundle.get("roster") or {}).get("roster_id"))
    viewer_info = by_roster.get(viewer_rid) or {}
    sell_value = float(sell_entry.get("value") or 0)
    sell_pos = sell_entry.get("position")
    if not sell_value or not sell_pos:
        return [], None, tools, viewer_info.get("needs") or [], viewer_info.get("surplus") or []
    findings, miss = [], None
    for r in rosters:
        rid = str(r.get("roster_id"))
        if rid == viewer_rid:
            continue
        info = by_roster.get(rid) or {}
        if sell_pos not in (info.get("needs") or []):
            continue
        pool, counts = _asset_pool(r, values, index)
        if not pool:
            continue
        try:
            pkgs = value_matched_acquire_packages(
                sell_value, pool, [], max_options=4,
                league_size=fit.get("num_teams") or 12, min_results=1) or []
        except Exception:
            pkgs = []
        for pkg in pkgs:
            assets = [a for a in (pkg.get("assets") or []) if not a.get("is_pick")]
            if not assets or len(assets) != len(pkg.get("assets") or []):
                continue
            # Never strip the partner's only player at a position.
            if any(counts.get(a.get("position"), 0) < 2 for a in assets):
                continue
            receive = [_finding_player(a, index) for a in assets]
            f = _finding(rid, info.get("team_name"), [sell_entry], receive, info,
                         viewer_info.get("needs") or [])
            if abs(f["fairness_gap"]) <= fair_value_band(
                    max(f["receive_total"], f["give_total"])):
                findings.append(f)
                break
            if miss is None or abs(f["fairness_gap"]) < abs(miss["fairness_gap"]):
                miss = f
    findings.sort(key=lambda f: abs(f["fairness_gap"]))
    return findings, miss, tools, viewer_info.get("needs") or [], viewer_info.get("surplus") or []


def _target_findings(ctx, bundle, target_entry, owner_roster, values, index):
    """Specific-target mode: value-matched offers from the viewer's own
    roster for one named player, surfaced only when the owner's needs or
    surplus make the deal plausible for them too."""
    from utils.trade_value import fair_value_band
    tools = ["trade_acquire_packages.value_matched_acquire_packages",
             "utils.trade.classify_position_needs",
             "utils.trade.detect_surplus_positions",
             "utils.trade_value.fair_value_band"]
    rosters = ctx.get("rosters") or []
    fit = _league_fit(ctx, rosters, values, index)
    by_roster = fit.get("by_roster") or {}
    viewer_rid = str((bundle.get("roster") or {}).get("roster_id"))
    viewer_info = by_roster.get(viewer_rid) or {}
    owner_info = by_roster.get(str(owner_roster.get("roster_id"))) or {}
    focus = float(target_entry.get("value") or 0)
    if not focus:
        return [], None, tools, viewer_info.get("needs") or [], viewer_info.get("surplus") or []
    pool, _counts = _asset_pool(bundle.get("roster") or {}, values, index)
    pool = [p for p in pool if p["player_id"] != target_entry.get("player_id")]
    if not pool:
        return [], None, tools, viewer_info.get("needs") or [], viewer_info.get("surplus") or []
    try:
        pkgs = value_matched_acquire_packages(
            focus, pool, [], max_options=6,
            league_size=fit.get("num_teams") or 12, min_results=1) or []
    except Exception:
        pkgs = []
    owner_needs = set(owner_info.get("needs") or [])
    spare = target_entry.get("position") in set(owner_info.get("surplus") or [])
    findings, miss = [], None
    for pkg in pkgs:
        assets = [a for a in (pkg.get("assets") or []) if not a.get("is_pick")]
        if not assets or len(assets) != len(pkg.get("assets") or []):
            continue
        give = [_finding_player(a, index) for a in assets]
        f = _finding(owner_roster.get("roster_id"), owner_info.get("team_name"),
                     give, [target_entry], owner_info, viewer_info.get("needs") or [])
        plausible = spare or any(p["position"] in owner_needs for p in give)
        if plausible and abs(f["fairness_gap"]) <= fair_value_band(
                max(f["receive_total"], f["give_total"])):
            findings.append(f)
        elif miss is None or abs(f["fairness_gap"]) < abs(miss["fairness_gap"]):
            miss = f
    findings.sort(key=lambda f: abs(f["fairness_gap"]))
    return findings, miss, tools, viewer_info.get("needs") or [], viewer_info.get("surplus") or []


_ROUND_LABELS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def _pick_equivalents(bundle, ctx, sell_value, num_teams):
    """What a player's value buys in next year's picks, priced by the
    site's own pick value table (dashboard_services.picks, the table the
    trade calculator prices picks from).

    Returns single picks and two-pick pairs whose table value lands
    inside the analyzer's fair band around the player's value, closest
    first, or None when the table cannot price the year.
    """
    if not sell_value:
        return None
    try:
        from utils.lineup_slots import is_superflex_lineup
        from utils.trade_value import fair_value_band
        table = pick_value_table(
            league_teams=num_teams or 10,
            is_sf=is_superflex_lineup(ctx.get("roster_positions") or []))
    except Exception:
        return None
    if not table:
        return None
    season = bundle.get("season")
    if not season:
        return None
    year = int(season) + 1
    band = fair_value_band(float(sell_value))
    singles = []
    for rnd in (1, 2, 3, 4):
        for bucket in ("early", "mid", "late"):
            v = table.get(f"{year}_{rnd}_{bucket}")
            if v:
                singles.append({"label": f"{year} {bucket} {_ROUND_LABELS[rnd]}",
                                "value": round(float(v), 1), "kind": "single",
                                "round": rnd})
    if not singles:
        try:
            from utils.draft import pick_value_from_table
            for rnd in (1, 2, 3, 4):
                v = pick_value_from_table(table, year, rnd, num_teams=num_teams)
                if v:
                    singles.append({"label": f"{year} {_ROUND_LABELS[rnd]}",
                                    "value": round(float(v), 1), "kind": "single",
                                    "round": rnd})
        except Exception:
            singles = []
    if not singles:
        return None
    singles.sort(key=lambda s: abs(s["value"] - sell_value))
    in_band = [s for s in singles if abs(s["value"] - sell_value) <= band]
    pairs = []
    for i, s1 in enumerate(singles):
        for s2 in singles[i + 1:]:
            if s2["round"] <= s1["round"]:
                continue
            total = round(s1["value"] + s2["value"], 1)
            if abs(total - sell_value) <= band:
                pairs.append({"label": f"{s1['label']} + {s2['label']}",
                              "value": total, "kind": "pair"})
    pairs.sort(key=lambda p: abs(p["value"] - sell_value))
    equivalents = (in_band[:3] + pairs[:2])[:4]
    return {"year": year, "equivalents": equivalents,
            "nearest_single": singles[0]}


def _shop_findings(ctx, bundle, values, index):
    """Shop-the-pool: my most valuable player at a surplus position (the
    league fit's own surplus read), with real sell findings for him.

    Returns None when I hold no surplus-position player worth shopping,
    so the caller falls back to need mode instead of inventing a shop
    candidate.
    """
    rosters = ctx.get("rosters") or []
    roster = bundle.get("roster") or {}
    try:
        fit = _league_fit(ctx, rosters, values, index)
    except Exception:
        return None
    by_roster = (fit or {}).get("by_roster") or {}
    viewer = by_roster.get(str(roster.get("roster_id"))) or {}
    surplus = viewer.get("surplus") or []
    if not surplus:
        return None
    mine = []
    for pid in roster.get("players") or []:
        entry = _finding_player({**(values.get(str(pid)) or {}),
                                 "player_id": str(pid)}, index)
        if entry.get("value") and entry.get("position") in surplus:
            mine.append(entry)
    if not mine:
        return None
    mine.sort(key=lambda e: -e["value"])
    sell_entry = mine[0]
    findings, miss, tools, needs, surplus_out = _sell_findings(
        ctx, bundle, sell_entry, values, index)
    return {"findings": findings, "miss": miss, "tools": tools,
            "needs": needs, "surplus": surplus_out,
            "sell_player": sell_entry, "shop_candidates": mine[:3]}


def build_trade_finding(bundle, entities, player_ids) -> tuple[dict, list[str]]:
    """Trade FINDING pack: ranked packages computed by the site's suggestion
    machinery, behind the same PRO gate as the site's Suggestions surface.

    Gate first: /api/trade-suggestions calls has_premium_for_viewer with the
    session viewer identity and returns a 403 paywall before any suggestion
    is built. The same call runs here before any scan, and a caller the site
    would paywall gets a gated pack with no findings and no scan numbers.
    """
    pack = _base_pack("trade_finding", bundle)
    tools: list[str] = []
    pack.update({"mode": None, "target_position": None, "sell_player": None,
                 "target_player": None, "viewer_needs": [], "viewer_surplus": [],
                 "findings": [], "no_fit": False, "closest_miss": None,
                 "margin": {"thin": False}})
    ctx = bundle.get("ctx") or {}
    roster = bundle.get("roster") or {}
    if not ctx or roster.get("roster_id") is None:
        pack["available"] = False
        pack["missing"] = "your roster in this league"
        pack["nearest"] = []
        return pack, tools

    tools.append("subscriptions.has_premium_for_viewer")
    viewer = bundle.get("viewer") or {}
    try:
        premium = bool(has_premium_for_viewer(
            viewer.get("viewer_username") or viewer.get("username"),
            viewer.get("viewer_user_id") or viewer.get("user_id"),
            bundle.get("league_id"), bundle.get("platform") or "sleeper",
            bundle.get("season")))
    except Exception:
        premium = False
    if not premium:
        pack["gated"] = True
        return pack, tools

    entities = entities or {}
    target_position = str(entities.get("target_position") or "").upper() or None
    if target_position not in ("QB", "RB", "WR", "TE", "FLEX"):
        target_position = None
    pack["target_position"] = target_position
    index = _index()
    values = _value_rows()
    my_ids = {str(p) for p in (roster.get("players") or [])}
    named = [str(pid) for pid in (player_ids or []) if pid]
    rosters = ctx.get("rosters") or []

    # A named player decides the mode: mine to move is a sell, another
    # team's player is a specific target, an unrostered player is a waiver
    # add rather than a trade.
    named_entry, owner_roster = None, None
    if named:
        pid = named[0]
        named_entry = _finding_player({**(values.get(pid) or {}), "player_id": pid}, index)
        if pid not in my_ids:
            for r in rosters:
                if pid in {str(p) for p in (r.get("players") or [])}:
                    owner_roster = r
                    break
    shop_done = False
    if (entities or {}).get("shop_pool") and not named:
        shop = _shop_findings(ctx, bundle, values, index)
        if shop is not None:
            findings, miss, used, needs, surplus = (
                shop["findings"], shop["miss"], shop["tools"],
                shop["needs"], shop["surplus"])
            pack["mode"] = "shop"
            pack["sell_player"] = shop["sell_player"]
            pack["shop_candidates"] = shop["shop_candidates"]
            shop_done = True
    if shop_done:
        pass
    elif named_entry and named[0] in my_ids:
        pack["mode"] = "sell"
        pack["sell_player"] = named_entry
        findings, miss, used, needs, surplus = _sell_findings(
            ctx, bundle, named_entry, values, index)
    elif named_entry and owner_roster is not None:
        pack["mode"] = "target"
        pack["target_player"] = named_entry
        findings, miss, used, needs, surplus = _target_findings(
            ctx, bundle, named_entry, owner_roster, values, index)
    elif named_entry:
        pack["mode"] = "target"
        pack["target_player"] = named_entry
        pack["target_unrostered"] = True
        findings, miss, used, needs, surplus = [], None, [], [], []
    else:
        pack["mode"] = "need"
        findings, miss, used, needs, surplus = _need_findings(
            ctx, bundle, target_position, index)
    tools += used
    # Worth in picks: price the sell player's value against next year's
    # pick table and carry the equivalents the answer may cite.
    if (entities or {}).get("value_in_picks") and pack["mode"] == "sell" \
            and pack.get("sell_player"):
        equiv = _pick_equivalents(bundle, ctx,
                                  float(pack["sell_player"].get("value") or 0),
                                  len(rosters))
        if equiv:
            pack["pick_equivalents"] = equiv
            tools.append("dashboard_services.picks.load_pick_value_table")
    pack["viewer_needs"] = needs
    pack["viewer_surplus"] = surplus
    pack["findings"] = findings[:3]
    pack["no_fit"] = not pack["findings"]
    pack["closest_miss"] = miss if pack["no_fit"] else None
    if pack["no_fit"]:
        pack["nearest_alternative"] = "waivers"
    return pack, tools


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


# -- replacement / succession ---------------------------------------------------

def _player_age(pid, index=None):
    """Age from the players index entry (the same field the waiver FAAB
    path reads), as a float, or None when the index carries none."""
    index = index if index is not None else _index()
    entry = {}
    if isinstance(index, dict):
        entry = index.get(str(pid)) or index.get(pid) or {}
    if not isinstance(entry, dict):
        return None
    try:
        age = float(entry.get("age"))
    except (TypeError, ValueError):
        return None
    return age if age > 0 else None


def _owner_by_pid(ctx):
    """pid -> owning team's display name, from the league ctx rosters."""
    out = {}
    users = {str(u.get("user_id")): u for u in (ctx.get("users") or [])
             if isinstance(u, dict)}
    for r in ctx.get("rosters") or []:
        if not isinstance(r, dict):
            continue
        team = _team_name(ctx, r, users)
        for pid in r.get("players") or []:
            out[str(pid)] = team
    return out


def _located(entry, pid, owners, my_ids):
    """A player entry plus where he lives: my roster, another named
    roster, or the waiver wire."""
    out = dict(entry)
    if str(pid) in my_ids:
        out["location"] = "your roster"
        out["on_my_roster"] = True
    elif str(pid) in owners:
        out["location"] = owners[str(pid)]
        out["owner_team"] = owners[str(pid)]
        out["on_my_roster"] = False
    else:
        out["location"] = "waivers"
        out["on_my_roster"] = False
    return out


def build_replacement(bundle, entities, player_ids) -> tuple[dict, list[str]]:
    """Succession pack: same-position players in a comparable value band
    with a real age gap in the asked direction (or, for cheaper, a real
    value gap), each tagged with where he is rostered.

    Ages and values are the site's own (players index age, model value
    table); the band is the comparability rule, not a new metric: a
    replacement must be within 0.35x to 1.5x of the subject's value and
    at least 2 years younger/older, or at most 0.65x of his value when
    the ask is cheaper.
    """
    pack = _base_pack("replacement", bundle)
    tools: list[str] = []
    pid = player_ids[0] if player_ids else None
    if not pid:
        pack["available"] = False
        pack["missing"] = "no resolved player"
        return pack, tools
    mode = str((entities or {}).get("replace_mode") or "younger").lower()
    if mode not in ("younger", "older", "cheaper"):
        mode = "younger"
    pack["mode"] = mode
    index = _index()
    values = _value_rows()
    if values:
        tools.append("model_value_table")
    info = _player_info(pid, index)
    subject_value = _row_value(values.get(str(pid)))
    subject_age = _player_age(pid, index)
    subject = {**info, "value": subject_value, "age": subject_age}
    pack["subject"] = subject
    if subject_value is None:
        pack["available"] = False
        pack["missing"] = f"no model value for {info.get('name') or 'that player'}"
        return pack, tools
    if mode in ("younger", "older") and subject_age is None:
        pack["available"] = False
        pack["missing"] = f"no age on file for {info.get('name') or 'that player'}"
        return pack, tools
    ctx = bundle.get("ctx") or {}
    owners = _owner_by_pid(ctx)
    my_ids = {str(p) for p in ((bundle.get("roster") or {}).get("players") or [])}
    if not my_ids:
        my_ids = {str(p) for p in (bundle.get("roster_player_ids") or set())}
    position = info.get("position")
    pool = []
    for pid2, row in values.items():
        if str(pid2) == str(pid):
            continue
        info2 = _player_info(pid2, index)
        if position and info2.get("position") != position:
            continue
        v2 = _row_value(row)
        if v2 is None or v2 <= 0:
            continue
        age2 = _player_age(pid2, index)
        entry = {**info2, "value": v2, "age": age2}
        if age2 is not None and subject_age is not None:
            entry["age_delta"] = round(abs(subject_age - age2), 1)
        pool.append(entry)
    if mode == "younger":
        cands = [e for e in pool if e["age"] is not None
                 and e["age"] <= subject_age - 2.0
                 and 0.35 * subject_value <= e["value"] <= 1.5 * subject_value]
        cands.sort(key=lambda e: (abs(e["value"] - subject_value), e["age"]))
    elif mode == "older":
        cands = [e for e in pool if e["age"] is not None
                 and e["age"] >= subject_age + 2.0
                 and 0.35 * subject_value <= e["value"] <= 1.5 * subject_value]
        cands.sort(key=lambda e: (abs(e["value"] - subject_value), -e["age"]))
    else:  # cheaper: a real value gap down, production band implied by value
        cands = [e for e in pool
                 if 0.15 * subject_value <= e["value"] <= 0.65 * subject_value]
        cands.sort(key=lambda e: -e["value"])
    pack["candidates"] = [_located(e, e["player_id"], owners, my_ids)
                          for e in cands[:4]]
    if not pack["candidates"]:
        pack["available"] = False
        pack["missing"] = (f"no {mode} replacement for {info.get('name')} in a "
                           "comparable value band")
    return pack, tools


# -- handcuff / next up ---------------------------------------------------------

def build_handcuff(bundle, entities, player_ids) -> tuple[dict, list[str]]:
    """Next-up pack: who sits behind a named player on his own NFL team's
    depth chart, and what the viewer already holds at that position.

    Depth order is the Sleeper depth chart carried in the players index
    (utils.waivers.build_depth_index, the waiver engine's own grouping).
    When the index carries no depth order for the room, the room is
    ordered by model value and the pack says so (depth_basis), so the
    answer never claims a chart it did not read.
    """
    pack = _base_pack("handcuff", bundle)
    tools: list[str] = []
    pid = player_ids[0] if player_ids else None
    if not pid:
        pack["available"] = False
        pack["missing"] = "no resolved player"
        return pack, tools
    index = _index()
    values = _value_rows()
    if values:
        tools.append("model_value_table")
    info = _player_info(pid, index)
    subject = {**info, "value": _row_value(values.get(str(pid))),
               "age": _player_age(pid, index)}
    pack["subject"] = subject
    team = str(info.get("team") or "").upper()
    position = str(info.get("position") or "").upper()
    ctx = bundle.get("ctx") or {}
    owners = _owner_by_pid(ctx)
    my_ids = {str(p) for p in ((bundle.get("roster") or {}).get("players") or [])}
    if not my_ids:
        my_ids = {str(p) for p in (bundle.get("roster_player_ids") or set())}

    room: list[dict] = []
    basis = "value"
    if team and position:
        try:
            from utils.waivers import build_depth_index
            group = (build_depth_index(index) or {}).get((team, position)) or []
            tools.append("waivers.build_depth_index")
        except Exception:
            group = []
        ordered = [g for g in group if g.get("depth_order") is not None]
        if group and ordered:
            basis = "depth_chart"
            group = sorted(group, key=lambda g: (g.get("depth_order") is None,
                                                 g.get("depth_order") or 99))
            room = [str(g.get("pid")) for g in group]
    if basis != "depth_chart":
        # Value-ordered room: everyone on the same NFL team at the position
        # with a model value, best first; the subject's place in it is his
        # value rank, so "next" is the next value down.
        room = [str(p2) for p2, row in values.items()
                if str(_player_info(p2, index).get("team") or "").upper() == team
                and str(_player_info(p2, index).get("position") or "").upper() == position
                and _row_value(row) is not None]
        room.sort(key=lambda p2: -(_row_value(values.get(p2)) or 0))
    pack["depth_basis"] = basis
    next_ups = []
    if str(pid) in room:
        behind = room[room.index(str(pid)) + 1:]
        for p2 in behind[:2]:
            e = {**_player_info(p2, index),
                 "value": _row_value(values.get(p2)),
                 "age": _player_age(p2, index)}
            next_ups.append(_located(e, p2, owners, my_ids))
    pack["next_up"] = next_ups[0] if next_ups else None
    pack["also_next"] = next_ups[1] if len(next_ups) > 1 else None
    my_options = []
    for p2 in my_ids:
        if str(p2) == str(pid):
            continue
        info2 = _player_info(p2, index)
        if position and str(info2.get("position") or "").upper() != position:
            continue
        my_options.append({**info2, "value": _row_value(values.get(p2)),
                           "age": _player_age(p2, index)})
    my_options.sort(key=lambda e: -(e["value"] or 0))
    pack["my_options"] = my_options[:3]
    if not next_ups and not my_options:
        pack["available"] = False
        pack["missing"] = (f"no one behind {info.get('name') or 'that player'} "
                           f"on the {team} depth chart")
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
        missing_note, nearest = KNOWN_MISSING.get(hint, ("that metric is not tracked", []))
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


# -- buy-low / sell-high boards ---------------------------------------------
#
# The v1 screen ranked the site's value movers by raw delta behind a
# one-dimensional season test (buy-low: season PPG at or over expected;
# sell-high: season PPG under it). That admits falling knives (value and
# role collapsing together), cannot tell a TD-propped spike from an
# earned rise, and ranks a 5% dip on a star over a 25% dip on a mid-tier
# player. The v2 logic ranks by MISPRICING: how far the price moved
# against what the player's recent production and opportunity line, his
# same-position peers, and his context say it should have.
#
# Signals (all from data the site already computes; no new metrics):
# - price: player_value_history.get_top_movers (the /api/value-movers
#   source, called with that route's arguments), scored as the RELATIVE
#   move delta / baseline value, never the raw delta.
# - recent line: the last 4 PLAYED weeks of the weekly advanced-metrics
#   series. expected_ppr is that week's opportunity-based expectation
#   and actual points = expected_ppr + ppr_over_expected, so the pair is
#   internally consistent. Season PPG vs expected PPG (the _metrics_for
#   line the single-player pack ships) is context and the fallback when
#   the series cannot verify a recent window.
# - opportunity: weekly volume weights from the same series (targets +
#   carries per game; pass attempts + carries for QBs), recent vs season
#   pace (opp_ratio). Production down WITH opportunity down is a falling
#   knife and is excluded from buy-low no matter how far value fell;
#   production down with opportunity holding or rising is the regression
#   buy-low. A rise backed by real opportunity growth is earned and is
#   excluded from sell-high, unless the surplus is TD-propped
#   (td_over_expected over the window), which is not role growth.
# - peers: same-position players in the examined mover pool whose
#   governing production and opportunity per game sit within a tolerance
#   band of the candidate's; the candidate's price is measured against
#   the peer median value (peer_gap). Production at peer level priced
#   well under the peer median is the cleanest buy-low; the mirror is
#   the cleanest sell-high.
# - context: injury status from the player index (the long-term
#   designations IR / PUP / NFI are excluded outright: an out-for-season
#   player is never a buy-low; a short absence explains a dip instead of
#   a role loss), the upcoming slate's average matchup rank from the
#   site's rank table as a why-line signal, and in dynasty leagues an
#   age weighting on the score (utils.waivers.WAIVER_PRIME_MAX cliffs).
#
# PRO gating: expected points and the volume weights are free fields.
# The PRO-gated derived metrics that summarize some of the same ideas
# (ppr_over_expected_per_game, opportunity_trend, xfp_trend) are NOT
# read; every signal below is computed from the free components.
#
# Score (so the board order is explainable from the pack):
#   rel_move = delta / baseline_value (signed)
#   fair_gap = (governing xPPG - governing PPG) / governing xPPG
#              (+ = production running under the opportunity line)
#   buy-low   score = discount * (1 + headroom) * (1 - 0.5 * over)
#                     + 0.03 * clamp(opp_ratio - 1, -1, 1) + 0.6 * peer_gap
#     discount = -rel_move; headroom = clamp(fair_gap, 0, 0.6);
#     over = clamp(-fair_gap, 0, 0.6): production over the line tempers
#     the discount, because part of that fall is justified;
#     peer_gap = (peer_median - value) / peer_median.
#   sell-high score = premium * (1 + propped)
#                     + 0.03 * clamp(1 - opp_ratio, -1, 1) + 0.6 * peer_gap
#     premium = rel_move; propped = clamp(-fair_gap, 0, 0.6);
#     peer_gap = (value - peer_median) / peer_median.
#   Sort: score desc, then opportunity ratio (buy-low desc, sell-high
#   asc), then |delta| desc. Dynasty age weighting multiplies the score
#   last: buy-low x1.08 at 24 or younger, x0.85 at the position's prime
#   cliff, x0.72 three years past it; sell-high inverts the cliff terms
#   (past-cliff spikes are the ones to cash out).

_VALUE_RECENT_WEEKS = 4
_VALUE_MIN_MOVE_PCT = 0.04
_VALUE_KNIFE_OPP_RATIO = 0.80
_VALUE_EARNED_OPP_RATIO = 1.15
_VALUE_TD_PROPPED_MIN = 0.75
_VALUE_SEASON_ENDING = {"IR", "PUP", "NFI"}
_VALUE_SIDELINED = {"O", "OUT", "D", "DOUBTFUL"}
_VALUE_PEER_MIN = 2
_VALUE_PEER_PROD_TOL = 0.25
_VALUE_PEER_OPP_TOL = 0.30
_VALUE_PEER_GAP_NAMED = 0.15


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _weekly_line(pid, season, position):
    """Recent-window line from the weekly advanced-metrics series.

    Played weeks are the rows carrying an expected_ppr (a DNP or bye
    week has no expectation row). Returns {} when the series has no
    played weeks; individual fields stay None when unverifiable so the
    caller can fall back to the season line without inventing a trend.
    """
    if not season:
        return {}
    try:
        rows = get_player_weekly_adv_series(str(pid), int(season)) or []
    except Exception:
        logger.debug("[assistant] weekly series failed", exc_info=True)
        return {}
    played = [r for r in rows
              if isinstance(r, dict) and r.get("expected_ppr") is not None]
    if not played:
        return {}
    played.sort(key=lambda r: r.get("week") or 0)

    def _opp(r):
        keys = (("w_pass_att", "w_carries") if position == "QB"
                else ("w_targets", "w_carries"))
        tot, seen = 0.0, False
        for k in keys:
            if r.get(k) is not None:
                tot += float(r[k])
                seen = True
        return tot if seen else None

    def _line(sub):
        n = len(sub)
        xppg = sum(float(r["expected_ppr"]) for r in sub) / n
        ppg = sum(float(r["expected_ppr"])
                  + float(r.get("ppr_over_expected") or 0.0)
                  for r in sub) / n
        opps = [o for o in (_opp(r) for r in sub) if o is not None]
        return {"games": n, "ppg": ppg, "xppg": xppg,
                "opp_pg": (sum(opps) / len(opps)) if opps else None,
                "opp_weeks": len(opps)}

    season_line = _line(played)
    recent_rows = played[-_VALUE_RECENT_WEEKS:]
    recent_line = _line(recent_rows) if len(recent_rows) >= 2 else None
    opp_ratio = None
    if (recent_line and recent_line["opp_pg"] is not None
            and recent_line["opp_weeks"] >= 2
            and season_line["opp_pg"] and season_line["opp_weeks"] >= 3):
        opp_ratio = recent_line["opp_pg"] / season_line["opp_pg"]
    td_over = None
    tds = [float(r["td_over_expected"]) for r in recent_rows
           if r.get("td_over_expected") is not None]
    if tds:
        td_over = sum(tds)
    return {"season": season_line, "recent": recent_line,
            "opp_ratio": opp_ratio, "td_over_recent": td_over,
            "last_played_week": played[-1].get("week")}


def _peer_context(cand, pool, kind):
    """Peer-relative pricing for one candidate, or None.

    Peers are same-position players from the examined mover pool whose
    governing production AND opportunity per game sit within the
    tolerance band of the candidate's. Returns the peer median value,
    the signed gap to it (fraction), and the anchor peer the why-line
    names: the most expensive peer for buy-low (the star whose line the
    candidate matches), the cheapest peer for sell-high (the same
    production available for far less).
    """
    cp, co = cand.get("_prod"), cand.get("_opp")
    if not cp or not co or cp <= 0 or co <= 0:
        return None
    peers = []
    for o in pool:
        if o["pid"] == cand["pid"] or o.get("position") != cand.get("position"):
            continue
        op, oo, ov = o.get("_prod"), o.get("_opp"), o.get("value")
        if not op or not oo or not ov:
            continue
        if (abs(op - cp) <= _VALUE_PEER_PROD_TOL * cp
                and abs(oo - co) <= _VALUE_PEER_OPP_TOL * co):
            peers.append(o)
    if len(peers) < _VALUE_PEER_MIN:
        return None
    values = sorted(float(p["value"]) for p in peers)
    mid = len(values) // 2
    median = (values[mid] if len(values) % 2
              else (values[mid - 1] + values[mid]) / 2.0)
    if median <= 0:
        return None
    value = float(cand["value"])
    if kind == "buy_low":
        gap = (median - value) / median
        anchor = max(peers, key=lambda p: float(p["value"]))
    else:
        gap = (value - median) / median
        anchor = min(peers, key=lambda p: float(p["value"]))
    anchor_value = float(anchor["value"])
    anchor_gap = None
    if anchor_value > 0 and value > 0:
        anchor_gap = ((anchor_value - value) / anchor_value if kind == "buy_low"
                      else (value - anchor_value) / anchor_value)
    return {"median": median, "gap": gap, "count": len(peers),
            "anchor_name": anchor.get("name"), "anchor_value": anchor_value,
            "anchor_gap": anchor_gap}


def _value_age_factor(kind, dynasty, age, position):
    """Dynasty age weighting on the mispricing score (1.0 in redraft,
    in keeper formats, without an age, or inside the prime window).
    Cliffs are the site's own WAIVER_PRIME_MAX ages."""
    if not dynasty or age is None:
        return 1.0
    try:
        from utils.waivers import WAIVER_PRIME_MAX
        prime = WAIVER_PRIME_MAX.get(position or "")
    except Exception:
        prime = None
    if not prime:
        return 1.0
    if kind == "buy_low":
        if age <= 24:
            return 1.08
        if age >= prime + 3:
            return 0.72
        if age >= prime:
            return 0.85
        return 1.0
    if age >= prime:
        return 1.10
    if age <= 24:
        return 0.95
    return 1.0


def _build_value_board(pack, bundle, tools, kind, position, team, top_n,
                       roster_ids):
    """Buy-low / sell-high candidate boards ("who should I buy low?").

    Candidates are the site's own value movers (see the design note
    above _weekly_line): the price move is scored against the recent
    production and opportunity line, the same-position peer median,
    slate and age context, with falling knives and season-ending
    injuries excluded before scoring.
    """
    season = bundle.get("season")
    ctx = bundle.get("ctx") or {}
    is_sf = False
    try:
        from utils.lineup_slots import is_superflex_lineup
        is_sf = bool(is_superflex_lineup(ctx.get("roster_positions") or []))
    except Exception:
        is_sf = False
    league_type = "sf" if is_sf else "1qb"
    num_teams = len(ctx.get("rosters") or [])
    league_size = num_teams if num_teams in (8, 10, 12, 14) else 10
    try:
        dynasty = bool(dynasty_without_keepers(ctx))
    except Exception:
        dynasty = False
    pack["dynasty"] = dynasty
    current_values = {}
    for pid, row in _value_rows().items():
        raw = row.get("sf_value") if is_sf else row.get("value")
        try:
            current_values[pid] = float(raw or 0)
        except (TypeError, ValueError):
            current_values[pid] = 0.0
    payload = {}
    try:
        payload = get_top_movers(
            days=7, limit=60, league_type=league_type, league_size=league_size,
            min_baseline_value=10, min_current_value=20.0,
            current_values=current_values) or {}
        tools.append("player_value_history.get_top_movers")
    except Exception:
        logger.debug("[assistant] value movers failed", exc_info=True)
        payload = {}
    if isinstance(payload, list):
        risers = [r for r in payload
                  if isinstance(r, dict) and float(r.get("delta") or 0) > 0]
        fallers = [r for r in payload
                   if isinstance(r, dict) and float(r.get("delta") or 0) < 0]
        used_days = None
    else:
        risers = payload.get("risers") or []
        fallers = payload.get("fallers") or []
        used_days = payload.get("used_days")
    pack["value_days"] = int(used_days) if used_days else 7
    pack["direction"] = "fallers" if kind == "buy_low" else "risers"
    pack["label"] = ("Buy-low candidates" if kind == "buy_low"
                     else "Sell-high candidates")
    pack["movers_basis"] = "value_window"
    pack["window"] = {"kind": "value", "days": pack["value_days"],
                      "applied": True}
    source = fallers if kind == "buy_low" else risers
    pack["movers_examined"] = len(source)

    # Pass 1: a verified line for every examined mover (both directions:
    # peers are drawn from the whole pool, not just this board's side).
    index = {}
    try:
        index = _index()
    except Exception:
        index = {}
    pool = {}
    for direction, moved_rows in (("fallers", fallers), ("risers", risers)):
        for r in moved_rows:
            if not isinstance(r, dict):
                continue
            pid = str(r.get("player_id") or r.get("id") or "")
            if not pid or pid in pool:
                continue
            pos = str(r.get("position") or "").upper()
            if position and pos != position:
                continue
            if team and str(r.get("team") or "").upper() != team:
                continue
            metrics, mtools = _metrics_for(pid, season)
            for t in mtools:
                if t not in tools:
                    tools.append(t)
            weekly = _weekly_line(pid, season, pos)
            if weekly and ("advanced_metrics.get_player_weekly_adv_series"
                           not in tools):
                tools.append("advanced_metrics.get_player_weekly_adv_series")
            recent = weekly.get("recent") if weekly else None
            prod = None
            if recent:
                prod = recent["ppg"]
            elif metrics.get("ppg") is not None:
                prod = float(metrics["ppg"])
            opp = None
            if weekly:
                opp = (recent["opp_pg"] if recent and recent.get("opp_pg")
                       is not None else weekly["season"]["opp_pg"])
            info = _player_info(pid, index) if pid else {}
            pool[pid] = {
                "pid": pid, "row": r, "direction": direction,
                "name": r.get("name"), "position": pos, "metrics": metrics,
                "weekly": weekly, "recent": recent, "value": None,
                "_prod": prod, "_opp": opp,
                "injury_status": info.get("injury_status"),
                "age": _fnum((index.get(pid) or {}).get("age")),
            }
            try:
                pool[pid]["value"] = float(r.get("new_value"))
            except (TypeError, ValueError):
                pool[pid]["value"] = current_values.get(pid)

    # Pass 2: gates, then the mispricing score.
    week = bundle.get("week")
    rank_tables: dict = {}
    qualified = []
    want = "fallers" if kind == "buy_low" else "risers"
    for entry in pool.values():
        r = entry["row"]
        if entry["direction"] != want:
            continue
        pid = entry["pid"]
        metrics = entry["metrics"]
        ppg, xppg, games = (metrics.get("ppg"), metrics.get("expected_ppr_per_game"),
                            metrics.get("games"))
        if ppg is None or xppg is None:
            continue  # no verifiable production line, no candidacy
        if games is not None and games < 3:
            continue
        try:
            delta = float(r.get("delta"))
            baseline = float(r.get("old_value"))
        except (TypeError, ValueError):
            continue
        if baseline <= 0:
            continue
        rel_move = delta / baseline
        if kind == "buy_low" and (delta >= 0 or -rel_move < _VALUE_MIN_MOVE_PCT):
            continue
        if kind == "sell_high" and (delta <= 0 or rel_move < _VALUE_MIN_MOVE_PCT):
            continue
        status = str(entry["injury_status"] or "").strip().upper()
        if status in _VALUE_SEASON_ENDING:
            continue  # out long-term: never a candidate on either board
        weekly, recent = entry["weekly"], entry["recent"]
        if recent:
            g_ppg, g_xppg = recent["ppg"], recent["xppg"]
        else:
            g_ppg, g_xppg = float(ppg), float(xppg)
        if not g_xppg:
            continue
        over_gov = g_ppg - g_xppg
        fair_gap = -over_gov / g_xppg
        opp_ratio = weekly.get("opp_ratio") if weekly else None
        td_over = weekly.get("td_over_recent") if weekly else None
        td_propped = bool(td_over is not None and td_over >= _VALUE_TD_PROPPED_MIN)
        if kind == "buy_low":
            # Falling knife: production AND opportunity falling together.
            # The exclusion is supreme: no peer gap or discount rescues it.
            if (opp_ratio is not None and opp_ratio <= _VALUE_KNIFE_OPP_RATIO
                    and over_gov < 0):
                continue
            if (recent and float(ppg) > 0
                    and recent["ppg"] <= 0.55 * float(ppg)
                    and (opp_ratio is None or opp_ratio < 1.0)):
                continue
            discount = -rel_move
            score = (discount * (1 + _clamp(fair_gap, 0.0, 0.6))
                     * (1 - 0.5 * _clamp(-fair_gap, 0.0, 0.6)))
            if opp_ratio is not None:
                score += 0.03 * _clamp(opp_ratio - 1.0, -1.0, 1.0)
        else:
            # Earned rise: real opportunity growth behind production at
            # or over the line is not mispricing, unless TDs prop it.
            if (opp_ratio is not None and opp_ratio >= _VALUE_EARNED_OPP_RATIO
                    and over_gov >= 0 and not td_propped):
                continue
            premium = rel_move
            score = premium * (1 + _clamp(-fair_gap, 0.0, 0.6))
            if opp_ratio is not None:
                score += 0.03 * _clamp(1.0 - opp_ratio, -1.0, 1.0)
        peers = _peer_context(entry, list(pool.values()), kind)
        if peers:
            score += 0.6 * peers["gap"]
        score *= _value_age_factor(kind, dynasty, entry["age"],
                                   entry["position"])
        entry["_score"] = score
        entry["_opp_ratio"] = opp_ratio

        # Mechanism: the why-line names the driver, in gate order.
        missed = None
        if week and weekly and weekly.get("last_played_week"):
            missed = (int(week) - 1) - int(weekly["last_played_week"])
        if kind == "buy_low":
            if status in _VALUE_SIDELINED:
                mechanism = "priced down while sidelined short term"
            elif missed is not None and missed >= 2:
                mechanism = "priced down during a short absence, now back"
            elif opp_ratio is not None and opp_ratio >= 1.10 and over_gov < 0:
                mechanism = "role rising while production lags the opportunity"
            elif over_gov < 0:
                mechanism = "opportunity holding while production lags it"
            else:
                mechanism = "still producing at or over expectation at a lower price"
        else:
            if td_propped:
                mechanism = "scoring propped by touchdowns over expectation"
            elif over_gov > 0:
                mechanism = "production running over his opportunity line"
            else:
                mechanism = "price ran ahead of the production line"

        row = _board_row(len(qualified) + 1, r, roster_ids,
                         value=r.get("new_value"))
        row["delta"] = _r3(delta)
        row["baseline_value"] = _r3(r.get("old_value"))
        row["ppg"] = _r3(ppg)
        row["expected_ppg"] = _r3(xppg)
        row["over_expected"] = round(float(ppg) - float(xppg), 2)
        row["games"] = games
        row["rel_move_pct"] = round(rel_move * 100, 1)
        if recent:
            row["recent_ppg"] = _r3(recent["ppg"])
            row["recent_expected_ppg"] = _r3(recent["xppg"])
            row["recent_over_expected"] = round(over_gov, 2)
            row["recent_games"] = recent["games"]
        if opp_ratio is not None:
            row["opp_ratio"] = round(opp_ratio, 2)
        if td_over is not None:
            row["td_over_expected_recent"] = _r3(td_over)
        row["age"] = entry["age"]
        row["injury_status"] = entry["injury_status"]
        row["mechanism"] = mechanism
        if peers:
            row["peer_median_value"] = _r3(peers["median"])
            row["peer_gap_pct"] = round(peers["gap"] * 100, 1)
            row["peer_count"] = peers["count"]
            if (peers["anchor_gap"] is not None
                    and peers["anchor_gap"] >= _VALUE_PEER_GAP_NAMED):
                row["peer_anchor_name"] = peers["anchor_name"]
                row["peer_anchor_value"] = _r3(peers["anchor_value"])
                row["peer_anchor_gap_pct"] = round(peers["anchor_gap"] * 100, 1)
        row["score"] = round(score, 4)
        row["_opp_sort"] = opp_ratio
        qualified.append(row)

    if kind == "buy_low":
        qualified.sort(key=lambda row: (
            -row["score"],
            -(row["_opp_sort"] if row["_opp_sort"] is not None else 1.0),
            -abs(row["delta"])))
    else:
        qualified.sort(key=lambda row: (
            -row["score"],
            (row["_opp_sort"] if row["_opp_sort"] is not None else 1.0),
            -abs(row["delta"])))
    for i, row in enumerate(qualified):
        row["rank"] = i + 1
        # Slate context for the why-line: the average matchup rank of
        # the next few opponents (rank 1 = easiest defense for the
        # position), from the same rank table the start/sit path uses.
        try:
            games_ahead = _upcoming_schedule(bundle, row.get("team"), 3)
            opps = [g for g in games_ahead
                    if not g.get("bye") and g.get("opponent")]
            pos = row.get("position")
            if opps and pos and season:
                if pos not in rank_tables:
                    table = matchup_rank_table(int(season), pos)
                    rank_tables[pos] = table[0] or {}
                    if "app._matchup_rank_table" not in tools:
                        tools.append("app._matchup_rank_table")
                rmap = rank_tables[pos]
                ranks = [rmap[o["opponent"]] for o in opps
                         if rmap.get(o["opponent"]) is not None]
                if ranks:
                    avg = sum(ranks) / len(ranks)
                    row["slate_avg_rank"] = round(avg, 1)
                    row["slate_games"] = len(ranks)
                    if avg <= 11:
                        row["slate_note"] = "soft slate ahead"
                    elif avg >= 22:
                        row["slate_note"] = "tough slate ahead"
        except Exception:
            logger.debug("[assistant] value board slate failed",
                         exc_info=True)
        row.pop("_opp_sort", None)
    pack["pool_size"] = len(qualified)
    board = qualified[:top_n]
    if not board:
        return _no_board(
            pack,
            "no buy-low candidates right now" if kind == "buy_low"
            else "no sell-high candidates right now", []), tools
    pack["board"] = board
    pack["board_count"] = len(board)
    if kind == "buy_low":
        rule = ("ranked by mispricing against the recent production and "
                "opportunity line (last 4 played weeks), peer prices at "
                "the position, and slate and age context; falling knives "
                "(production and opportunity both falling) and "
                "season-ending injuries are excluded")
        moved = "fallers"
    else:
        rule = ("ranked by mispricing against the recent production and "
                "opportunity line (last 4 played weeks), peer prices at "
                "the position, and slate and age context; rises earned by "
                "real opportunity growth are excluded")
        moved = "risers"
    pack["qualification_text"] = (
        f"Value movement over the last {pack['value_days']} days from the "
        f"site's value history; {rule}. {len(qualified)} of "
        f"{pack['movers_examined']} {moved} qualify.")
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
    if kind not in ("board", "intersection", "dual", "movers", "matchup",
                    "buy_low", "sell_high"):
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

    if kind in ("buy_low", "sell_high"):
        return _build_value_board(pack, bundle, tools, kind, position, team,
                                  top_n, roster_ids)

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
        missing_note, nearest = KNOWN_MISSING.get(hint, ("that metric is not tracked", []))
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
    # Ask shape: a position filter ("best available RB") narrows the gate
    # list to that position, and the mode decides which extra ranking the
    # pack carries (stash score, drop ranking, this-week streamers).
    ent = entities or {}
    pack["waiver_mode"] = (str(ent.get("waiver_mode") or "").lower()
                           if str(ent.get("waiver_mode") or "").lower()
                           in ("stash", "drop", "stream") else None)
    pos_filter = None
    for key in ("position", "position_hint", "target_position"):
        raw = str(ent.get(key) or "").upper()
        if raw in ("QB", "RB", "WR", "TE"):
            pos_filter = raw
            break
    pack["position_filter"] = pos_filter
    if pos_filter:
        candidates = [c for c in candidates if c.get("position") == pos_filter]
    candidates.sort(key=lambda c: c["value"], reverse=True)
    pack["candidates"] = [c for c in candidates if c["passes_gate"]][:8] or candidates[:0]
    pack["gate_passing_count"] = len([c for c in candidates if c["passes_gate"]])
    # Drops ranked weakest first: the drop question is answered off this
    # order, and the do-not-drop list is the only protection. (This list
    # was previously an unordered slice of the roster set, so "most
    # droppable" had no grounded answer.)
    cleared = []
    for pid in roster_ids:
        if pid in protected:
            continue
        v = _row_value(values.get(pid))
        if v is None:
            continue
        cleared.append({**_player_info(pid, index), "value": v})
    cleared.sort(key=lambda c: c["value"])
    pack["cleared_drops"] = cleared[:8]
    depth_at_pos: dict[str, int] = {}
    for pid in roster_ids:
        p = _player_info(pid, index).get("position")
        if p:
            depth_at_pos[p] = depth_at_pos.get(p, 0) + 1
    pack["drop_ranking"] = [
        {**c, "roster_depth_at_position": depth_at_pos.get(c.get("position"), 0)}
        for c in cleared[:5]
    ]
    # A named drop candidate ("can I drop X"): where he sits in the
    # ranking and whether he is protected, instead of a generic list.
    named = [str(p) for p in (player_ids or []) if p]
    if named:
        target = named[0]
        tinfo = _player_info(target, index)
        asked = {**tinfo, "value": _row_value(values.get(target)),
                 "on_my_roster": target in roster_ids,
                 "protected": target in protected, "drop_rank": None}
        for i, c in enumerate(cleared):
            if c["player_id"] == target:
                asked["drop_rank"] = i + 1
                break
        pack["asked_drop"] = asked
    # Stash ranking: the waiver engine's own pickup score under its stash
    # horizon weights (utils.waivers), over the free-agent pool. Stashes
    # do not need to clear the add gate; that is the point of a stash.
    if pack["waiver_mode"] == "stash":
        try:
            from utils.waivers import horizon_weights, waiver_pickup_score
            weights = horizon_weights("stash", dynasty=not _league_is_redraft(ctx))
        except Exception:
            weights = None
        stash_pool = []
        for pid, row in values.items():
            if pid in owned:
                continue
            v = _row_value(row)
            if v is None or v <= 0:
                continue
            info = _player_info(pid, index)
            if info.get("position") not in ("QB", "RB", "WR", "TE"):
                continue
            if pos_filter and info.get("position") != pos_filter:
                continue
            age = _player_age(pid, index)
            score = v
            if weights is not None:
                try:
                    score = float(waiver_pickup_score(
                        {"value": v, "age": age, "position": info.get("position"),
                         "player_id": pid}, {}, w=weights))
                except Exception:
                    score = v
            stash_pool.append({**info, "value": v, "age": age,
                               "stash_score": round(score, 2)})
        stash_pool.sort(key=lambda c: -c["stash_score"])
        pack["stash_candidates"] = stash_pool[:5]
        if stash_pool:
            tools.append("waivers.waiver_pickup_score")
    # Streamers: free agents at the asked position ranked by THIS week's
    # projection (the start/sit evidence source), not by season value.
    if pack["waiver_mode"] == "stream":
        season, week = bundle.get("season"), bundle.get("week")
        streamers = []
        if season and week:
            try:
                week_map = load_week_projection(int(season), int(week))
            except Exception:
                week_map = None
            if week_map:
                tools.append("utils.load_week_projection")
                scoring = _scoring(bundle)
                games = _schedule_games(season, week)
                try:
                    from utils.fantasy_scoring import weekly_projection_points
                except Exception:
                    weekly_projection_points = None
                for pid, row in values.items():
                    if weekly_projection_points is None:
                        break
                    if pid in owned:
                        continue
                    info = _player_info(pid, index)
                    pos = info.get("position")
                    if pos not in ("QB", "RB", "WR", "TE"):
                        continue
                    if pos_filter and pos != pos_filter:
                        continue
                    try:
                        pts = weekly_projection_points(week_map, str(pid), scoring, pos)
                    except Exception:
                        pts = None
                    if pts is None or float(pts) <= 0:
                        continue
                    opp, label, on_bye = _opponent_for(info.get("team"), games)
                    if on_bye:
                        continue
                    streamers.append({**info, "value": _row_value(row),
                                      "projection": round(float(pts), 1),
                                      "opponent": opp, "matchup": label})
                streamers.sort(key=lambda c: -c["projection"])
        pack["streamers"] = streamers[:4]
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


# -- league context: playoffs / power / standings / matchup / schedule ---------
#
# These five builders wire the site's league-context surfaces (workshop
# section 9, P1): the playoff odds sim and exact scenarios, the power
# rankings context builder, the standings machinery (finalize_team_stats via
# build_standings_as_of_week, so median-match and division rules are the
# site's own, never recomputed here), the matchup preview + win probability
# model, and remaining schedule strength. Numbers are computed by those
# engines; the packs only carry what an answer may cite.

def playoff_sim_cached(ctx, platform):
    """The app's TTL-cached playoff sim (shared with the odds API)."""
    from app import _playoff_sim_cached as _fn
    return _fn(ctx, platform)


def power_rankings_context(ctx):
    from dashboard_services.ai.context_builders import build_power_rankings_context as _fn
    return _fn(ctx)


def standings_as_of_week(ctx, week):
    from app import build_standings_as_of_week as _fn
    return _fn(ctx, week)


def live_matchups(platform, league_id, season, week, ctx):
    """(matchups, status_by_pid, proj_map) via the portfolio card's builder."""
    from routes.user_pages_bp import _build_live_matchups as _fn
    return _fn(platform, league_id, season, week, ctx)


def projected_starter_avgs(rosters, ctx, season, current_week):
    from routes.schedule_api_bp import _projected_starter_avgs as _fn
    return _fn(rosters, ctx, season, current_week)


def _f(value, nd=1):
    """Rounded float, or None for missing/NaN (packs never carry NaN)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return round(f, nd)


def _fnum(value):
    """Coerce a stored numeric (Decimal from psycopg rows included) to a
    plain float/int for the pack. Type normalization only: the stored
    number itself is never recomputed or rescaled."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else f




def _f0(value, nd=1):
    out = _f(value, nd)
    return 0.0 if out is None else out


def _record_str(wins, losses, ties=0):
    rec = f"{int(wins or 0)}-{int(losses or 0)}"
    if ties:
        rec += f"-{int(ties)}"
    return rec


def _team_names(ctx) -> dict:
    """roster_id (str) -> team name, from the ctx roster_map."""
    out = {}
    for rid, name in ((ctx or {}).get("roster_map") or {}).items():
        if name:
            out[str(rid)] = str(name)
    return out


def _viewer_rid(bundle) -> str:
    return str((bundle.get("roster") or {}).get("roster_id") or "")


def _finalized_weeks(ctx) -> list:
    """Ascending finalized weeks in the ctx weekly frame (duck-typed)."""
    df = (ctx or {}).get("df_weekly")
    if df is None:
        return []
    try:
        if getattr(df, "empty", True):
            return []
        fin = df[df["finalized"] == True]
        return sorted({int(w) for w in fin["week"].tolist()})
    except Exception:
        return []


def build_playoffs(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """Playoff path: Monte Carlo odds (the app's cached sim) plus, inside
    the end-of-season window, the exact clinch/elimination scenarios.

    Early in the season the odds are soft (workshop section 4: unstable
    before the stretch): the pack flags odds_soft and carries the
    wins-needed math (the projected cut line) so the answer can lead with
    that instead of false precision.
    """
    pack = _base_pack("playoffs", bundle)
    tools: list[str] = []
    ctx = bundle.get("ctx") or {}
    if not ctx:
        pack["available"] = False
        pack["missing"] = "no league context loaded"
        return pack, tools
    try:
        odds = playoff_sim_cached(ctx, bundle.get("platform") or "sleeper") or []
        tools.append("simulate_playoff_odds.simulate_playoff_odds")
    except Exception:
        logger.debug("[assistant] playoff sim failed", exc_info=True)
        odds = []
    if not odds:
        pack["available"] = False
        pack["missing"] = "playoff odds are not available for this league yet"
        return pack, tools
    settings = ctx.get("league_settings") or {}
    try:
        playoff_teams = int(settings.get("playoff_teams") or 6)
    except (TypeError, ValueError):
        playoff_teams = 6
    try:
        playoff_week_start = int(settings.get("playoff_week_start") or 15)
    except (TypeError, ValueError):
        playoff_week_start = 15
    current_week = int(bundle.get("week") or ctx.get("current_week") or 0)
    names = _team_names(ctx)
    rows = []
    for row in odds:
        if not isinstance(row, dict) or row.get("roster_id") is None:
            continue
        rid = str(row["roster_id"])
        rows.append({
            "roster_id": rid,
            "team_name": names.get(rid) or row.get("team_name") or f"Roster {rid}",
            "wins": int(row.get("wins") or 0), "losses": int(row.get("losses") or 0),
            "ties": int(row.get("ties") or 0),
            "record": _record_str(row.get("wins"), row.get("losses"), row.get("ties")),
            "playoff_pct": _f0(row.get("playoff_pct")),
            "bye_pct": _f0(row.get("bye_pct")),
            "avg_final_wins": _f(row.get("avg_final_wins")),
        })
    rows.sort(key=lambda r: r["playoff_pct"], reverse=True)
    pack["field"] = rows
    pack["playoff_teams"] = playoff_teams
    pack["playoff_week_start"] = playoff_week_start
    pack["current_week"] = current_week
    pack["is_complete"] = bool(odds[0].get("is_complete")) if odds else False
    weeks_remaining = None
    try:
        from data_building.simulate_playoff_odds import remaining_regular_season_weeks
        weeks_remaining = int(remaining_regular_season_weeks(
            odds, playoff_week_start, current_week=current_week))
    except Exception:
        played = max((r["wins"] + r["losses"] + r["ties"] for r in rows), default=0)
        weeks_remaining = max(0, playoff_week_start - 1 - played)
    pack["weeks_remaining"] = weeks_remaining
    cut = rows[playoff_teams - 1] if len(rows) >= playoff_teams else None
    first_out = rows[playoff_teams] if len(rows) > playoff_teams else None
    cut_line = {}
    if cut:
        cut_line["last_in"] = {"team_name": cut["team_name"],
                               "playoff_pct": cut["playoff_pct"],
                               "avg_final_wins": cut["avg_final_wins"]}
    if first_out:
        cut_line["first_out"] = {"team_name": first_out["team_name"],
                                 "playoff_pct": first_out["playoff_pct"],
                                 "avg_final_wins": first_out["avg_final_wins"]}
    pack["cut_line"] = cut_line
    projected_cut_wins = cut["avg_final_wins"] if cut else None
    pack["projected_cut_wins"] = projected_cut_wins
    me = next((r for r in rows if r["roster_id"] == _viewer_rid(bundle)), None)
    if me is not None:
        me = dict(me)
        if projected_cut_wins is not None:
            me["wins_still_needed"] = round(max(0.0, projected_cut_wins - me["wins"]), 1)
        pack["me"] = me
    # Soft-odds window: with 8+ regular-season weeks left the sim is mostly
    # projection noise, so the answer must say so and lean on wins math.
    pack["odds_soft"] = bool(weeks_remaining is not None and weeks_remaining >= 8)
    pack["margin"] = {"thin": bool(pack["odds_soft"] or (me and 20.0 <= me["playoff_pct"] <= 80.0))}
    # Exact scenarios (clinch / elimination / magic numbers), the same
    # assembly /api/playoff-scenarios serves. Only surfaces inside its
    # end-of-season window (compute_scenarios returns show=False earlier).
    pack["scenario"] = None
    try:
        from data_building.playoff_scenarios import compute_scenarios, scenario_summary
        from data_building.simulate_playoff_odds import _n_byes, build_sim_state
        state = build_sim_state(ctx, bundle.get("platform") or "sleeper")
        if state:
            divisions = int(settings.get("divisions") or 0) >= 2
            division_map: dict = {}
            if divisions:
                for r in ctx.get("rosters") or []:
                    if isinstance(r, dict) and r.get("roster_id") is not None:
                        try:
                            div = int((r.get("settings") or {}).get("division") or 0)
                        except (TypeError, ValueError):
                            div = 0
                        if div:
                            division_map[int(r["roster_id"])] = div
            sim_teams = state.get("teams") or []
            scen_teams = [
                {"roster_id": t["roster_id"], "wins": t.get("wins", 0),
                 "ties": t.get("ties", 0), "pf": t.get("pf", 0.0),
                 "division": division_map.get(int(t["roster_id"]))}
                for t in sim_teams
            ]
            scen = compute_scenarios(
                scen_teams, state.get("matchups") or {},
                int(state.get("playoff_teams") or playoff_teams),
                n_byes=_n_byes(int(state.get("playoff_teams") or playoff_teams)),
                divisions=divisions,
            )
            tools.append("playoff_scenarios.compute_scenarios")
            if scen.get("show"):
                name_by_rid = {int(t["roster_id"]): t.get("name") for t in sim_teams}
                entry = None
                for rid_key, e in (scen.get("teams") or {}).items():
                    if str(rid_key) == _viewer_rid(bundle):
                        entry = e
                        break
                if entry:
                    scenario = {
                        "status": entry.get("status"),
                        "summary": scenario_summary(entry),
                        "controls_destiny": bool(entry.get("controls_destiny")),
                        "wins_to_clinch": entry.get("wins_to_clinch"),
                        "clinch_if_win_next": bool(entry.get("clinch_if_win_next")),
                        "out_if_lose_next": bool(entry.get("out_if_lose_next")),
                        "needs_help": bool(entry.get("needs_help")),
                        "best_seed": entry.get("best_seed"),
                        "worst_seed": entry.get("worst_seed"),
                    }
                    nxt = entry.get("next_game") or {}
                    if nxt.get("opp") is not None:
                        scenario["next_opponent"] = name_by_rid.get(int(nxt["opp"]))
                    pack["scenario"] = scenario
    except Exception:
        logger.debug("[assistant] playoff scenarios failed", exc_info=True)
    return pack, tools


def build_power_rankings(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """Power rankings from build_power_rankings_context (the standings
    page's own builder; PowerScore from blended_team_scores). Movement is
    the standings page's method: last finalized week's ranking recomputed
    from the week-capped ctx and diffed by roster id."""
    pack = _base_pack("power_rankings", bundle)
    tools: list[str] = []
    ctx = bundle.get("ctx") or {}
    if not ctx:
        pack["available"] = False
        pack["missing"] = "no league context loaded"
        return pack, tools
    try:
        pr = power_rankings_context(ctx) or {}
        tools.append("ai.context_builders.build_power_rankings_context")
    except Exception:
        logger.debug("[assistant] power rankings failed", exc_info=True)
        pr = {}
    teams = [t for t in (pr.get("teams") or []) if isinstance(t, dict)]
    if not teams:
        pack["available"] = False
        pack["missing"] = "power rankings are not available for this league yet"
        return pack, tools

    def _row(t, rank):
        return {
            "rank": rank,
            "roster_id": str(t.get("roster_id")),
            "team_name": t.get("team_name") or f"Roster {t.get('roster_id')}",
            "power_score": _f(t.get("power_score")),
            "wins": int(t.get("wins") or 0), "losses": int(t.get("losses") or 0),
            "record": _record_str(t.get("wins"), t.get("losses")),
            "ppg": _f(t.get("avg")),
            "all_play_pct": _f(t.get("all_play_pct"), 3),
            "playoff_pct": _f(t.get("playoff_pct")),
            "momentum_label": t.get("momentum_label"),
            "is_me": str(t.get("roster_id")) == _viewer_rid(bundle),
        }

    ordered = sorted(teams, key=lambda t: (t.get("rank") is None, t.get("rank") or 0,
                                           -(t.get("power_score") or 0)))
    rows = [_row(t, int(t.get("rank") or (i + 1))) for i, t in enumerate(ordered)]
    rows.sort(key=lambda r: r["rank"])
    # Week-over-week movement, best effort (same method as the standings page).
    try:
        weeks = _finalized_weeks(ctx)
        if len(weeks) >= 2:
            capped = standings_as_of_week(ctx, weeks[-2])
            prev = power_rankings_context(capped) or {}
            prev_ranks = {str(t.get("roster_id")): int(t.get("rank") or (i + 1))
                          for i, t in enumerate(prev.get("teams") or [])
                          if isinstance(t, dict) and t.get("roster_id") is not None}
            if prev_ranks:
                tools.append("app.build_standings_as_of_week")
                for r in rows:
                    if r["roster_id"] in prev_ranks:
                        r["rank_change"] = prev_ranks[r["roster_id"]] - r["rank"]
    except Exception:
        logger.debug("[assistant] power movement failed", exc_info=True)
    pack["teams"] = rows
    pack["leader"] = rows[0] if rows else None
    pack["me"] = next((r for r in rows if r["is_me"]), None)
    pack["season_phase"] = pr.get("season_phase")
    return pack, tools


def build_standings(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """Standings from the site's own machinery: team_stats recomputed
    through the last finalized week by build_standings_as_of_week (which
    runs finalize_team_stats with the league's median-match setting), and
    seeds from build_standings_map (division winners first, then wild
    cards). Records are never recomputed in the assistant."""
    pack = _base_pack("standings", bundle)
    tools: list[str] = []
    ctx = bundle.get("ctx") or {}
    if not ctx:
        pack["available"] = False
        pack["missing"] = "no league context loaded"
        return pack, tools
    team_stats = None
    weeks = _finalized_weeks(ctx)
    if weeks:
        try:
            capped = standings_as_of_week(ctx, weeks[-1]) or {}
            team_stats = capped.get("team_stats")
            if team_stats is not None:
                tools.append("app.build_standings_as_of_week")
        except Exception:
            logger.debug("[assistant] standings cap failed", exc_info=True)
            team_stats = None
    if team_stats is None:
        team_stats = ctx.get("team_stats")
    if team_stats is None:
        pack["available"] = False
        pack["missing"] = "standings are not available for this league yet"
        return pack, tools
    tools.append("service.finalize_team_stats")
    roster_map = ctx.get("roster_map") or {}
    owner_to_rid = {}
    for rid, owner in roster_map.items():
        if owner is not None:
            owner_to_rid[str(owner)] = rid
            owner_to_rid[str(owner).strip()] = rid
    names = _team_names(ctx)
    div_by_rid: dict = {}
    div_records = None
    try:
        from utils.standings import division_records_for_ctx, roster_division_map
        div_by_rid = roster_division_map(
            ctx.get("rosters") or [],
            settings=ctx.get("league_settings") or (ctx.get("league") or {}).get("settings") or {},
        ) or {}
        div_records = division_records_for_ctx(ctx)
    except Exception:
        logger.debug("[assistant] standings divisions failed", exc_info=True)
        div_by_rid, div_records = {}, None
    seeds: dict = {}
    try:
        from dashboard_services.service import build_standings_map
        seeds = build_standings_map(
            team_stats, roster_map,
            division_by_rid=div_by_rid or None,
            division_records_by_rid=div_records,
        ) or {}
        if seeds:
            tools.append("service.build_standings_map")
    except Exception:
        logger.debug("[assistant] standings seeds failed", exc_info=True)
        seeds = {}
    raw_rows = []
    if hasattr(team_stats, "iterrows"):
        raw_rows = [dict(r) for _, r in team_stats.iterrows()]
    elif isinstance(team_stats, (list, tuple)):
        raw_rows = [dict(r) for r in team_stats if isinstance(r, dict)]
    rows = []
    for row in raw_rows:
        owner = row.get("owner")
        rid = owner_to_rid.get(str(owner)) if owner is not None else None
        if rid is None and owner is not None:
            rid = owner_to_rid.get(str(owner).strip())
        rid_s = str(rid) if rid is not None else None
        wins, losses, ties = (int(_f0(row.get("Wins"), 0)), int(_f0(row.get("Losses"), 0)),
                              int(_f0(row.get("Ties"), 0)))
        div_rec = None
        if div_records and rid_s is not None:
            rec = div_records.get(int(rid)) if str(rid).isdigit() else div_records.get(rid)
            if rec:
                div_rec = _record_str(rec[0], rec[1], rec[2] if len(rec) > 2 else 0)
        rows.append({
            "seed": None,
            "roster_id": rid_s,
            "team_name": (names.get(rid_s) if rid_s else None) or (str(owner) if owner else "Team"),
            "wins": wins, "losses": losses, "ties": ties,
            "record": _record_str(wins, losses, ties),
            "pf": _f0(row.get("PF")), "pa": _f0(row.get("PA")),
            "ppg": _f(row.get("AVG")),
            "streak": (str(row.get("Streak")) if row.get("Streak") else None),
            "division": (div_by_rid.get(int(rid)) if rid is not None and str(rid).isdigit()
                         else div_by_rid.get(rid)) if div_by_rid else None,
            "division_record": div_rec,
            "is_me": bool(rid_s and rid_s == _viewer_rid(bundle)),
        })
    if not rows:
        pack["available"] = False
        pack["missing"] = "standings are not available for this league yet"
        return pack, tools
    for r in rows:
        seed = None
        if r["roster_id"] is not None:
            seed = seeds.get(int(r["roster_id"])) if r["roster_id"].isdigit() else seeds.get(r["roster_id"])
            if seed is None:
                seed = seeds.get(r["roster_id"])
        r["seed"] = int(seed) if seed is not None else None
    if all(r["seed"] is not None for r in rows):
        rows.sort(key=lambda r: r["seed"])
    else:
        rows.sort(key=lambda r: (-r["wins"], -r["pf"]))
        for i, r in enumerate(rows):
            r["seed"] = i + 1
    pack["teams"] = rows
    pack["me"] = next((r for r in rows if r["is_me"]), None)
    leader_pf = max(rows, key=lambda r: r["pf"])
    pack["points_leader"] = {"team_name": leader_pf["team_name"], "pf": leader_pf["pf"]}
    pack["has_divisions"] = len({r["division"] for r in rows if r["division"]}) >= 2
    pack["division_leaders"] = []
    if pack["has_divisions"]:
        for div in sorted({r["division"] for r in rows if r["division"]}):
            top = next(r for r in rows if r["division"] == div)
            pack["division_leaders"].append({
                "division": div, "team_name": top["team_name"], "record": top["record"],
                "pf": top["pf"],
            })
    return pack, tools


def build_matchup(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """Head-to-head for one week: build_matchup_preview pairings, live
    totals and compute_win_prob, assembled exactly like the portfolio
    matchup card (_build_live_matchups). Any pairing in the league can be
    targeted with team_names; the default target is the caller's game."""
    pack = _base_pack("matchup", bundle)
    tools: list[str] = []
    ctx = bundle.get("ctx") or {}
    if not ctx:
        pack["available"] = False
        pack["missing"] = "no league context loaded"
        return pack, tools
    entities = entities or {}
    try:
        week = int(entities.get("week") or bundle.get("week") or 0)
    except (TypeError, ValueError):
        week = int(bundle.get("week") or 0)
    pack["week"] = week
    league_id = str(ctx.get("resolved_league_id") or bundle.get("league_id") or "")
    try:
        matchups, status_by_pid, proj_map = live_matchups(
            bundle.get("platform") or "sleeper", league_id,
            bundle.get("season"), week, ctx)
        tools.append("matchups.build_matchup_preview")
    except Exception:
        logger.debug("[assistant] matchup preview failed", exc_info=True)
        matchups, status_by_pid, proj_map = [], {}, {}
    if not matchups:
        pack["available"] = False
        pack["missing"] = f"no scheduled matchup for Week {week}"
        pack["nearest"] = []
        return pack, tools
    try:
        from dashboard_services.matchups import (
            _proj_value_for_pid, compute_win_prob, team_live_totals)
    except Exception:
        logger.debug("[assistant] matchup model import failed", exc_info=True)
        pack["available"] = False
        pack["missing"] = "matchup data is not available right now"
        return pack, tools

    def _side(team, win_pct):
        actual, proj = team_live_totals(team, status_by_pid, proj_map)
        starters = []
        for p in (team.get("starters") or []):
            if not p:
                continue
            try:
                pproj = _proj_value_for_pid(proj_map, p.get("pid"), pos=p.get("pos") or "")
            except Exception:
                pproj = 0.0
            starters.append({"name": p.get("name"), "position": p.get("pos"),
                             "projection": _f(pproj)})
        starters.sort(key=lambda s: -(s["projection"] or 0))
        return {
            "roster_id": str(team.get("roster_id")),
            "team_name": team.get("name") or f"Roster {team.get('roster_id')}",
            "record": team.get("record"),
            "current_points": _f(actual),
            "projected": _f(proj),
            "win_prob_pct": win_pct,
            "top_starters": starters[:3],
        }

    def _pair(m):
        left, right = m.get("left") or {}, m.get("right") or {}
        wp = None
        try:
            raw = compute_win_prob(left, right, status_by_pid, proj_map)
            if raw is not None:
                wp = round(float(raw) * 100.0, 1)
        except Exception:
            wp = None
        a = _side(left, wp)
        b = _side(right, (round(100.0 - wp, 1) if wp is not None else None))
        return a, b

    pairs = []
    for m in matchups:
        if isinstance(m, dict):
            pairs.append((m, *_pair(m)))
    tools.append("matchups.compute_win_prob")
    pack["board"] = [
        {"team_a": a["team_name"], "team_b": b["team_name"],
         "a_projected": a["projected"], "b_projected": b["projected"],
         "a_win_prob_pct": a["win_prob_pct"]}
        for _m, a, b in pairs
    ]
    wanted = [str(t).strip().lower() for t in (entities.get("team_names") or []) if str(t).strip()]
    hint = str(entities.get("other_team_hint") or "").strip().lower()
    if hint and hint not in wanted:
        wanted.append(hint)

    def _matches(team_name, want):
        n = str(team_name or "").lower()
        return bool(n) and (want in n or n in want)

    target = None
    if len(wanted) >= 2:
        for _m, a, b in pairs:
            if ((_matches(a["team_name"], wanted[0]) and _matches(b["team_name"], wanted[1]))
                    or (_matches(a["team_name"], wanted[1]) and _matches(b["team_name"], wanted[0]))):
                target = (a, b)
                break
    if target is None and wanted:
        for _m, a, b in pairs:
            if _matches(a["team_name"], wanted[0]) or _matches(b["team_name"], wanted[0]):
                target = (a, b)
                break
    viewer = _viewer_rid(bundle)
    is_mine = False
    if target is None:
        for _m, a, b in pairs:
            if viewer and viewer in (a["roster_id"], b["roster_id"]):
                target = (a, b) if a["roster_id"] == viewer else (b, a)
                is_mine = True
                break
    if target is None and pairs:
        target = (pairs[0][1], pairs[0][2])
    elif target is not None and viewer and viewer in (target[0]["roster_id"], target[1]["roster_id"]):
        is_mine = True
        if target[1]["roster_id"] == viewer:
            target = (target[1], target[0])
    a, b = target
    pack["matchup"] = {"week": week, "is_mine": is_mine, "team_a": a, "team_b": b}
    wp = a.get("win_prob_pct")
    pack["margin"] = {"thin": bool(wp is not None and 45.0 <= wp <= 55.0)}
    return pack, tools


def build_schedule(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """Remaining schedule strength from service.remaining_schedule_strength
    (the Schedule Assistant / schedule-rankings machinery): opponent
    strength indexed to 100 = league average, higher is a tougher slate.
    The playoff stretch re-runs the same function windowed to the playoff
    weeks, so both numbers come from one definition."""
    pack = _base_pack("schedule", bundle)
    tools: list[str] = []
    ctx = bundle.get("ctx") or {}
    if not ctx:
        pack["available"] = False
        pack["missing"] = "no league context loaded"
        return pack, tools
    rosters = [r for r in (ctx.get("rosters") or []) if isinstance(r, dict)]
    if not rosters:
        pack["available"] = False
        pack["missing"] = "schedule data is not available for this league yet"
        return pack, tools
    try:
        from dashboard_services.service import (
            regular_season_length, remaining_schedule_strength)
    except Exception:
        pack["available"] = False
        pack["missing"] = "schedule data is not available for this league yet"
        return pack, tools
    settings = ctx.get("league_settings") or (ctx.get("league") or {}).get("settings") or {}
    try:
        reg_weeks = int(regular_season_length(settings))
    except Exception:
        reg_weeks = 14
    try:
        playoff_week_start = int(settings.get("playoff_week_start") or reg_weeks + 1)
    except (TypeError, ValueError):
        playoff_week_start = reg_weeks + 1
    stretch_end = min(playoff_week_start + 2, 18)
    current_week = int(bundle.get("week") or ctx.get("current_week") or 0)
    matchups_by_week = ctx.get("matchups_by_week") or {}
    if not matchups_by_week:
        try:
            from dashboard_services.platform_api import get_matchups as _pf_matchups
            matchups_by_week = {}
            for w in range(1, max(reg_weeks, stretch_end) + 1):
                try:
                    matchups_by_week[w] = _pf_matchups(
                        bundle.get("platform") or "sleeper", bundle.get("league_id"),
                        w, bundle.get("season")) or []
                except Exception:
                    matchups_by_week[w] = []
        except Exception:
            matchups_by_week = {}
    names = _team_names(ctx)
    proj = None
    try:
        proj = projected_starter_avgs(rosters, ctx, bundle.get("season"), current_week) or None
    except Exception:
        proj = None

    def _rows(current, reg):
        results, source = remaining_schedule_strength(
            rosters, matchups_by_week, current_week=current,
            regular_season_weeks=reg, roster_names=names,
            projected_avg_by_rid=proj)
        out = []
        for r in results or []:
            if not isinstance(r, dict):
                continue
            out.append({
                "roster_id": str(r.get("roster_id")),
                "team_name": r.get("team_name") or names.get(str(r.get("roster_id"))) or "",
                "opp_index": _f(r.get("avg_opp_points")),
                "games_remaining": int(r.get("games_remaining") or 0),
                "my_avg_points": _f(r.get("my_avg_points")),
                "is_me": str(r.get("roster_id")) == _viewer_rid(bundle),
            })
        out.sort(key=lambda r: (r["opp_index"] is None, r["opp_index"]))
        for i, r in enumerate(out):
            r["ease_rank"] = i + 1  # 1 = easiest remaining slate
        return out, source

    try:
        rows, source = _rows(current_week, reg_weeks)
        tools.append("service.remaining_schedule_strength")
    except Exception:
        logger.debug("[assistant] schedule strength failed", exc_info=True)
        rows, source = [], None
    if not rows:
        pack["available"] = False
        pack["missing"] = "schedule data is not available for this league yet"
        return pack, tools
    pack["teams"] = rows
    pack["source"] = source
    pack["easiest"] = rows[0]
    pack["hardest"] = rows[-1]
    pack["me"] = next((r for r in rows if r["is_me"]), None)
    pack["weeks_remaining"] = max((r["games_remaining"] for r in rows), default=0)
    # Playoff stretch: the same machinery windowed to the playoff weeks.
    pack["playoff_stretch"] = None
    if playoff_week_start > current_week:
        try:
            srows, _src = _rows(playoff_week_start - 1, stretch_end)
            sme = next((r for r in srows if r["is_me"]), None)
            if sme and sme["games_remaining"] > 0:
                by_rid = {r["roster_id"]: r for r in srows}
                opponents = []
                viewer = _viewer_rid(bundle)
                for w in range(playoff_week_start, stretch_end + 1):
                    pair = {}
                    for m in (matchups_by_week.get(w) or []):
                        if not isinstance(m, dict):
                            continue
                        mid = m.get("matchup_id")
                        rid = str(m.get("roster_id"))
                        if mid is not None and rid:
                            pair.setdefault(mid, []).append(rid)
                    for rids in pair.values():
                        if viewer in rids and len(rids) == 2:
                            opp = next(x for x in rids if x != viewer)
                            opponents.append({
                                "week": w,
                                "opponent": names.get(opp) or f"Roster {opp}",
                                "opp_index": (by_rid.get(opp) or {}).get("opp_index"),
                            })
                pack["playoff_stretch"] = {
                    "weeks": [playoff_week_start, stretch_end],
                    "me": {"opp_index": sme["opp_index"], "ease_rank": sme["ease_rank"],
                           "opponents": opponents},
                    "easiest": {"team_name": srows[0]["team_name"],
                                "opp_index": srows[0]["opp_index"]},
                    "hardest": {"team_name": srows[-1]["team_name"],
                                "opp_index": srows[-1]["opp_index"]},
                }
        except Exception:
            logger.debug("[assistant] playoff stretch failed", exc_info=True)
            pack["playoff_stretch"] = None
    return pack, tools


# -- breakout board / track record / call history --------------------------------

def _board_candidate(c, rank):
    return {
        "rank": rank,
        "player_id": str(c.get("player_id") or ""),
        "name": c.get("player_name") or c.get("name") or str(c.get("player_id") or ""),
        "team": c.get("team") or "",
        "position": c.get("position") or "",
        "breakout_score": c.get("breakout_score") if c.get("breakout_score") is not None else c.get("score"),
        "classification_label": c.get("classification_label"),
        "confidence": c.get("confidence"),
    }


def build_breakout_board(bundle, entities, player_ids=None) -> tuple[dict, list[str]]:
    """This week's breakout calls, from the board's own candidate pipeline.

    The non-premium preview rule is mirrored from the board route exactly:
    a caller without a paid plan gets the first 3 candidates plus the
    locked count, and the pack carries nothing beyond the preview, so no
    answer can cite a name a free user could not see.
    """
    pack = _base_pack("breakout_board", bundle)
    tools: list[str] = []
    try:
        board = breakout_board_candidates(bundle.get("season")) or {}
        tools.append("breakout_api.get_breakout_board_candidates")
    except Exception:
        logger.debug("[assistant] breakout board failed", exc_info=True)
        board = {}
    if not board or board.get("data_available") is False:
        pack["available"] = False
        pack["missing"] = "the breakout board is not available right now"
        return pack, tools
    cands = [c for c in (board.get("candidates") or []) if isinstance(c, dict)]
    as_of = board.get("as_of_week")
    pack["as_of_week"] = as_of
    if isinstance(as_of, int):
        pack["week_label"] = f"Week {as_of + 1}"
    pack["total_calls"] = len(cands)
    preview = not bundle.get("is_paid")
    pack["preview"] = preview
    shown = cands[:3] if preview else cands
    if preview:
        pack["locked_count"] = max(0, len(cands) - 3)
    pack["candidates"] = [_board_candidate(c, i + 1) for i, c in enumerate(shown)]
    return pack, tools


def build_breakout_track_record(bundle, entities, player_ids=None) -> tuple[dict, list[str]]:
    """How good BR's breakout calls have been, from the board's own track
    record payload: the same surfaced-calls pool, graded calls, hit rate,
    and by-week split the breakout page sidebar renders."""
    pack = _base_pack("breakout_track_record", bundle)
    tools: list[str] = []
    try:
        tr = breakout_track_record_data(bundle.get("season")) or {}
        tools.append("breakout_api.get_breakout_track_record")
    except Exception:
        logger.debug("[assistant] breakout track record failed", exc_info=True)
        tr = {}
    weekly = tr.get("weekly") or {}
    overall = weekly.get("overall") or {}
    if not weekly.get("available") or not overall:
        pack["available"] = False
        pack["missing"] = "no breakout track record yet"
        return pack, tools
    pack["scoring_version"] = weekly.get("scoring_version")
    pack["overall"] = {k: overall.get(k) for k in ("calls", "graded", "hit", "partial", "miss")}
    hr = overall.get("hit_rate")
    pack["overall"]["hit_rate"] = hr
    pack["overall"]["hit_rate_pct"] = round(float(hr) * 100, 1) if hr is not None else None
    pack["pending"] = not overall.get("graded")
    by_week = []
    for w in tr.get("by_week") or []:
        if not isinstance(w, dict):
            continue
        row = {k: w.get(k) for k in ("week", "as_of_week", "calls", "graded", "hit", "partial", "miss", "hit_rate")
               if w.get(k) is not None}
        if row.get("hit_rate") is not None:
            row["hit_rate_pct"] = round(float(row["hit_rate"]) * 100, 1)
        by_week.append(row)
    pack["by_week"] = by_week

    def _slim(entries):
        out = []
        for e in entries or []:
            if not isinstance(e, dict):
                continue
            out.append({"name": e.get("player_name") or e.get("name"),
                        "week": e.get("as_of_week") or e.get("week"),
                        "label": e.get("label")})
        return out[:3]

    pack["biggest_hits"] = _slim(tr.get("hits"))
    pack["biggest_misses"] = _slim(tr.get("misses"))
    return pack, tools


def build_breakout_history(bundle, entities, player_ids) -> tuple[dict, list[str]]:
    """Was this player ever a breakout call? The earliest surfaced call
    (the same history the player modal's Breakout tab serves) plus that
    call's finished grade when one exists."""
    pack = _base_pack("breakout_history", bundle)
    tools: list[str] = []
    pid = player_ids[0] if player_ids else None
    if not pid:
        pack["available"] = False
        pack["missing"] = "no resolved player"
        return pack, tools
    index = _index()
    pack["player"] = _player_info(pid, index)
    season = bundle.get("season")
    history = None
    try:
        history = breakout_call_history_for(pid, season)
        tools.append("breakout_api.breakout_call_history")
    except Exception:
        logger.debug("[assistant] breakout history failed", exc_info=True)
        history = None
    current = None
    board_week = None
    try:
        board = breakout_board_candidates(season) or {}
        board_week = board.get("as_of_week")
        for i, c in enumerate(board.get("candidates") or []):
            if isinstance(c, dict) and str(c.get("player_id")) == str(pid):
                current = _board_candidate(c, i + 1)
                break
        if board:
            tools.append("breakout_api.get_breakout_board_candidates")
    except Exception:
        logger.debug("[assistant] breakout board (history) failed", exc_info=True)
    pack["on_current_board"] = current is not None
    if current is not None:
        pack["current_call"] = current
    if history:
        pack["was_call"] = True
        earliest = {"as_of_week": history.get("as_of_week"),
                    "label": history.get("label"),
                    "season": history.get("season")}
        pack["earliest_call"] = earliest
        try:
            detail = breakout_week_detail(str(pid), history.get("season") or season,
                                          history.get("as_of_week")) or {}
            tools.append("breakout_api.get_historical_weekly_breakout_detail")
            if detail.get("grade"):
                pack["grade"] = detail["grade"]
            if detail.get("breakout_score") is not None:
                earliest["breakout_score"] = detail["breakout_score"]
        except Exception:
            logger.debug("[assistant] breakout week detail failed", exc_info=True)
    elif current is not None:
        pack["was_call"] = True
        pack["earliest_call"] = {
            "as_of_week": board_week,
            "label": f"Week {board_week + 1}" if isinstance(board_week, int) else None,
            "season": season,
            "breakout_score": current.get("breakout_score"),
        }
    else:
        pack["was_call"] = False
    return pack, tools


# -- teams (league roster strength and grades) -------------------------------------

_POS_HINTS = {"QB": "QB", "QUARTERBACK": "QB", "RB": "RB", "RUNNING BACK": "RB",
              "WR": "WR", "WIDE RECEIVER": "WR", "WIDEOUT": "WR",
              "TE": "TE", "TIGHT END": "TE"}


def _position_focus(entities) -> str | None:
    hint = str((entities or {}).get("position_hint") or "").strip().upper()
    if hint in CORE_POS:
        return hint
    for word, pos in _POS_HINTS.items():
        if word in hint:
            return pos
    return None


def build_teams(bundle, entities, player_ids=None) -> tuple[dict, list[str]]:
    """League-wide roster strength: every team's grade and per-position
    strength and rank, from the Teams page machinery, in the mode the
    question asks for (Dynasty by default, This Season on redraft values)."""
    pack = _base_pack("teams", bundle)
    tools: list[str] = []
    ctx = bundle.get("ctx") or {}
    rosters = [r for r in (ctx.get("rosters") or []) if isinstance(r, dict)]
    if not ctx or not rosters:
        pack["available"] = False
        pack["missing"] = "no league context"
        return pack, tools
    mode = _resolve_mode(ctx, (entities or {}).get("teams_mode"))
    pack["mode"] = mode
    pack["mode_label"] = "This Season" if mode == "this_season" else "Dynasty"
    data = _league_position_data(ctx, rosters, mode)
    if data is None:
        pack["available"] = False
        pack["missing"] = "team strengths are not available for this league"
        return pack, tools
    tools.append("utils.trade.rank_rosters_by_position")
    if data.get("grades"):
        tools.append("context_builders.calculate_roster_grade")
    viewer_rid = str((bundle.get("roster") or {}).get("roster_id") or "")
    teams = []
    for r in rosters:
        rid = str(r.get("roster_id"))
        g = (data.get("grades") or {}).get(rid) or {}
        teams.append({
            "roster_id": rid,
            "team_name": data["team_names"].get(rid),
            "is_viewer": rid == viewer_rid,
            "grade": g.get("grade"), "grade_score": g.get("score"),
            "win_window": g.get("win_window"),
            "pos_strength": data["strengths"].get(rid),
            "pos_rank": {pos: (data["ranks"].get(pos) or {}).get(rid) for pos in CORE_POS},
        })
    teams.sort(key=lambda t: (t["grade_score"] is None, -(t["grade_score"] or 0),
                              t["team_name"] or ""))
    pack["teams"] = teams
    graded = [t for t in teams if t["grade_score"] is not None]
    if graded:
        top, bottom = graded[0], graded[-1]
        pack["strongest"] = {"team_name": top["team_name"], "grade": top["grade"],
                             "grade_score": top["grade_score"]}
        pack["weakest"] = {"team_name": bottom["team_name"], "grade": bottom["grade"],
                           "grade_score": bottom["grade_score"]}
    by_position = {}
    for pos in CORE_POS:
        ranked = sorted(teams, key=lambda t: (t["pos_rank"].get(pos) or 999))
        if not ranked:
            continue
        first, last = ranked[0], ranked[-1]
        by_position[pos] = {
            "strongest": {"team_name": first["team_name"],
                          "strength": (first["pos_strength"] or {}).get(pos)},
            "weakest": {"team_name": last["team_name"],
                        "strength": (last["pos_strength"] or {}).get(pos)},
        }
    pack["by_position"] = by_position
    focus = _position_focus(entities)
    if focus:
        pack["position_focus"] = focus
        ranked = sorted(teams, key=lambda t: (t["pos_rank"].get(focus) or 999))
        pack["position_ranking"] = [
            {"rank": t["pos_rank"].get(focus), "team_name": t["team_name"],
             "strength": (t["pos_strength"] or {}).get(focus)} for t in ranked]
    return pack, tools


# -- pick ownership ------------------------------------------------------------------

def build_pick_ownership(bundle, entities, player_ids=None) -> tuple[dict, list[str]]:
    """Who owns a given draft pick: the pick's original owner, its current
    owner after every trade, its canonical id, and its value."""
    pack = _base_pack("pick_ownership", bundle)
    tools: list[str] = []
    ent = entities or {}
    texts = list(ent.get("picks") or []) + list(ent.get("give_picks") or []) \
        + list(ent.get("receive_picks") or [])
    parsed, raw_text = None, None
    for t in texts:
        parsed = _parse_pick_text(t)
        if parsed is not None:
            raw_text = t
            break
    if parsed is None:
        pack["available"] = False
        pack["missing"] = "which pick (season and round) you mean"
        return pack, tools
    try:
        ownership = _pick_ownership(bundle)
    except Exception:
        logger.debug("[assistant] pick ownership failed", exc_info=True)
        ownership = None
    if not ownership:
        pack["available"] = False
        pack["missing"] = "pick ownership is not tracked for this league"
        return pack, tools
    tools.append(ownership["source"])
    viewer_rid = str((bundle.get("roster") or {}).get("roster_id") or "")
    team_names = ownership.get("team_names") or {}
    original = None
    low = str(raw_text or "").lower()
    for rid, name in team_names.items():
        if name and len(name) >= 3 and name.lower() in low:
            original = rid
            break
    if original is None and ent.get("other_team_hint"):
        hint = str(ent["other_team_hint"]).lower()
        for rid, name in team_names.items():
            if name and (hint in name.lower() or name.lower() in hint):
                original = rid
                break
    if original is None:
        original = viewer_rid or None
    current = None
    if original:
        current = (ownership.get("by_pick") or {}).get(
            (parsed["season"], parsed["round"], str(original)))
    table = None
    try:
        from utils.lineup_slots import is_superflex_lineup
        ctx = bundle.get("ctx") or {}
        table = pick_value_table(league_teams=ownership.get("num_teams") or 10,
                                 is_sf=is_superflex_lineup(ctx.get("roster_positions") or []))
        if table:
            tools.append("dashboard_services.picks.load_pick_value_table")
    except Exception:
        table = None
    entry = _pick_entry(bundle, ownership, table, parsed, side=None,
                        current_owner_rid=current, original_rid=original)
    pack["pick"] = entry
    pack["tracked"] = current is not None
    pack["owned_by_me"] = bool(current and viewer_rid and str(current) == viewer_rid)
    pack["original_owner_team"] = entry["original_owner_team"]
    pack["current_owner_team"] = entry["current_owner_team"]
    return pack, tools


def build_keeper(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """Keeper decisions from the Keeper Assistant's own machinery.

    Applicability is the keeper page's own call: a full dynasty league
    (is_dynasty_without_keepers) gets dynasty framing (the roster carries
    over; the numbers are stored values and ages, never keeper costs),
    and a league with no detected keeper limit gets the no-data shape
    (redraft leagues draft fresh; there is no keeper decision to make).
    Real keeper leagues run compute_league_keepers for the projected set
    and cost rounds, joined to the model value table for the value math.
    """
    entities = entities or {}
    pack = _base_pack("keeper", bundle)
    tools: list[str] = []
    ctx = bundle.get("ctx") or {}
    if not bundle.get("league_id") or not ctx:
        pack["available"] = False
        pack["missing"] = "keepers do not apply here: no league is linked for this conversation"
        pack["nearest"] = []
        return pack, tools
    roster = bundle.get("roster") or {}
    roster_id = roster.get("roster_id")
    values = _value_rows()
    index = _index()

    def _val(pid):
        return _row_value(values.get(str(pid)))

    if dynasty_without_keepers(ctx):
        tools.append("keeper_page.is_dynasty_without_keepers")
        pack["mode"] = "dynasty"
        pack["dynasty"] = True
        pack["keeper_limit"] = None
        core = []
        total = 0.0
        ages = []
        roster_ids = [str(p) for p in (roster.get("players") or [])]
        for pid in roster_ids:
            v = _val(pid)
            if v is not None:
                total += v
            age = _fnum((index.get(pid) or {}).get("age"))
            if age is not None:
                ages.append(age)
            info = _player_info(pid, index)
            core.append({"player_id": pid, "name": info["name"],
                         "position": info["position"], "age": age, "value": v})
        core = [c for c in core if c["value"] is not None]
        core.sort(key=lambda c: -c["value"])
        pack["roster_core"] = core[:8]
        pack["roster_size"] = len(roster_ids)
        pack["roster_value_total"] = round(total, 1) if core else None
        pack["roster_avg_age"] = round(sum(ages) / len(ages), 1) if ages else None
        return pack, tools

    limit = keeper_league_limit(ctx)
    tools.append("keeper_page.league_keeper_limit")
    if not limit or int(limit) <= 0:
        pack["available"] = False
        pack["missing"] = ("keepers do not apply in this league: it drafts fresh "
                           "every year and has no keeper settings")
        pack["nearest"] = []
        return pack, tools

    result = None
    try:
        result = compute_league_keepers(
            ctx, bundle.get("platform") or "sleeper",
            str(bundle.get("league_id") or ""), viewer_roster_id=roster_id)
        tools.append("keeper_page.compute_league_keepers")
    except Exception:
        logger.debug("[assistant] compute_league_keepers failed", exc_info=True)
    if not isinstance(result, dict):
        pack["available"] = False
        pack["missing"] = "keeper data is not available for this league right now"
        pack["nearest"] = []
        return pack, tools

    pack["mode"] = "keeper"
    pack["keeper_limit"] = result.get("limit") or int(limit)
    kept = [k for k in (result.get("kept") or []) if isinstance(k, dict)]
    mine = [k for k in kept if roster_id is not None
            and str(k.get("rosterId")) == str(roster_id)]
    kept_ids = {str(k.get("id")) for k in mine}
    my_keepers = []
    for k in mine:
        pid = str(k.get("id"))
        my_keepers.append({
            "player_id": pid, "name": k.get("name") or _player_info(pid, index)["name"],
            "position": k.get("pos") or _player_info(pid, index)["position"],
            "cost_round": k.get("costRound"), "adp_overall": _fnum(k.get("adpOverall")),
            "value": _val(pid),
        })
    my_keepers.sort(key=lambda c: -(c["value"] or 0))
    pack["my_keepers"] = my_keepers
    pack["league_kept_total"] = len(kept)
    # The cut line: the viewer's most valuable players the optimizer did
    # not keep, i.e. where the last keeper spot was decided.
    cut = []
    for pid in (roster.get("players") or []):
        pid = str(pid)
        if pid in kept_ids:
            continue
        v = _val(pid)
        if v is None:
            continue
        info = _player_info(pid, index)
        cut.append({"player_id": pid, "name": info["name"],
                    "position": info["position"], "value": v})
    cut.sort(key=lambda c: -c["value"])
    pack["cut_line"] = cut[:3]
    asked = None
    for pid in (player_ids or []):
        pid = str(pid)
        info = _player_info(pid, index)
        entry = next((k for k in my_keepers if k["player_id"] == pid), None)
        asked = {"player_id": pid, "name": info["name"], "position": info["position"],
                 "value": _val(pid), "projected_keeper": entry is not None,
                 "cost_round": entry["cost_round"] if entry else None}
        break
    pack["asked_player"] = asked
    return pack, tools


def _prospect_row_out(row: dict) -> dict:
    """One stored rankings row, trimmed to answerable fields, numbers as
    stored (type-normalized only)."""
    out = {
        "player_id": str(row.get("player_id") or ""),
        "sleeper_id": str(row.get("sleeper_id")) if row.get("sleeper_id") else None,
        "name": row.get("name") or "",
        "position": (row.get("position") or ""),
        "school": row.get("school") or "",
        "age": _fnum(row.get("age")),
        "overall_rank": _fnum(row.get("overall_rank")),
        "position_rank": _fnum(row.get("position_rank")),
        "prospect_score": _fnum(row.get("prospect_score")),
        "tier_label": row.get("tier_label") or row.get("tier") or None,
        "projected_round": _fnum(row.get("projected_round")),
        "projected_pick": _fnum(row.get("projected_pick")),
    }
    return out


def build_rookie(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """Prospect and draft-class answers from the pipeline's stored rows.

    Scores are the stored prospect scores, used exactly as stored (never
    recomputed or rescaled); the pack carries the rows' model_version as
    score_version. A named prospect resolves against the rankings rows
    themselves (by sleeper id when the router resolved an NFL id, else by
    normalized name), because the upcoming class is not in the NFL
    players index.
    """
    entities = entities or {}
    pack = _base_pack("rookie", bundle)
    tools: list[str] = []
    year = entities.get("draft_year")
    if not year:
        try:
            year = active_rookie_class()
            tools.append("rookie_pipeline.get_active_rookie_class")
        except Exception:
            year = None
    if not year and bundle.get("season"):
        year = int(bundle["season"]) + 1
    pack["draft_year"] = int(year) if year else None
    if not year:
        pack["available"] = False
        pack["missing"] = "no draft class is active for prospect rankings right now"
        pack["nearest"] = []
        return pack, tools
    try:
        rows = rookie_rankings(int(year)) or []
        tools.append("rookie_api._get_rankings")
    except Exception:
        logger.debug("[assistant] rookie rankings failed", exc_info=True)
        rows = []
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        pack["available"] = False
        pack["missing"] = f"no prospect rankings are stored for the {int(year)} class yet"
        pack["nearest"] = []
        return pack, tools
    versions = [str(r.get("model_version")) for r in rows if r.get("model_version")]
    if versions:
        pack["score_version"] = max(set(versions), key=versions.count)
    pack["class_size"] = len(rows)

    # Named prospect: sleeper-id match first, then normalized name match.
    named = None
    id_set = {str(p) for p in (player_ids or [])}
    if id_set:
        named = next((r for r in rows if r.get("sleeper_id") and str(r["sleeper_id"]) in id_set), None)
        if named is None:
            named = next((r for r in rows if str(r.get("player_id")) in id_set), None)
    if named is None:
        from dashboard_services.assistant.resolve import _normalize
        for name in (entities.get("player_names") or []):
            target = _normalize(name)
            if not target:
                continue
            named = next((r for r in rows if _normalize(r.get("name") or "") == target), None)
            if named is not None:
                break
    if named is not None:
        out = _prospect_row_out(named)
        try:
            detail = prospect_detail(str(named.get("player_id")), named) or {}
            tools.append("rookie_api._get_prospect_detail")
        except Exception:
            detail = {}
        seasons = [s for s in (detail.get("seasons") or []) if isinstance(s, dict)]
        if seasons:
            latest = max(seasons, key=lambda s: int(s.get("season") or 0))
            line = {"season": _fnum(latest.get("season")), "team": latest.get("team"),
                    "games_played": _fnum(latest.get("games_played"))}
            for k in ("pass_yards", "pass_tds", "rush_attempts", "rush_yards", "rush_tds",
                      "receptions", "targets", "receiving_yards", "receiving_tds",
                      "dominator_rating"):
                v = _fnum(latest.get(k))
                if v is not None:
                    line[k] = v
            out["college_latest"] = line
        ath = detail.get("athleticism") or {}
        ath_out = {k: _fnum(ath.get(k)) for k in ("forty_yard", "ras_score")
                   if _fnum(ath.get(k)) is not None}
        if ath_out:
            out["athleticism"] = ath_out
        reasons = named.get("key_reasons")
        if isinstance(reasons, str):
            try:
                import json as _json
                reasons = _json.loads(reasons)
            except Exception:
                reasons = [reasons]
        if isinstance(reasons, list) and reasons:
            out["key_reasons"] = [str(r) for r in reasons[:3]]
        pack["prospect"] = out
        pack["position_filter"] = None
        return pack, tools

    pos = str(entities.get("position") or "").upper() or None
    if pos not in ("QB", "RB", "WR", "TE"):
        pos = None
    pack["position_filter"] = pos
    board = rows
    if pos:
        board = [r for r in rows if str(r.get("position") or "").upper() == pos]
    pack["prospects"] = [_prospect_row_out(r) for r in board[:10 if not pos else 8]]
    return pack, tools


def build_cross_league(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """Cross-league answers: one labeled slice per linked league, from the
    portfolio page's own summary builder. Numbers are never blended across
    leagues (different scoring); each slice carries its league's name.
    Runs inside a single ask(), so the turn counts once against the cap
    however many leagues it fans out to. The thread's active league (the
    bundle) is untouched, so follow-ups still default to it.
    """
    entities = entities or {}
    pack = _base_pack("cross_league", bundle)
    tools: list[str] = []
    account_id = bundle.get("account_id")
    if not account_id:
        try:
            account_id = resolve_account_id()
        except Exception:
            account_id = None
    if not account_id:
        pack["available"] = False
        pack["missing"] = "no account leagues are linked for cross-league answers"
        pack["nearest"] = []
        return pack, tools
    try:
        memberships = list_account_leagues(int(account_id)) or []
        tools.append("accounts.list_user_leagues")
    except Exception:
        logger.debug("[assistant] list_user_leagues failed", exc_info=True)
        memberships = []
    if not memberships:
        pack["available"] = False
        pack["missing"] = "no leagues are linked to this account yet"
        pack["nearest"] = []
        return pack, tools
    pack["leagues_total"] = len(memberships)
    slices = []
    for m in memberships[:8]:
        if not isinstance(m, dict):
            continue
        slice_ = {
            "league_name": m.get("name") or "Unknown league",
            "platform": m.get("platform"), "league_id": str(m.get("league_id") or ""),
            "season": m.get("season"),
            "active": str(m.get("league_id") or "") == str(bundle.get("league_id") or ""),
            "state": "unavailable",
        }
        try:
            summary = league_summary(int(account_id), m, league_ctx_loader)
            if "portfolio_summary.build_league_summary" not in tools:
                tools.append("portfolio_summary.build_league_summary")
        except Exception:
            logger.debug("[assistant] league summary failed for %s",
                         m.get("league_id"), exc_info=True)
            summary = None
        if isinstance(summary, dict):
            slice_["league_name"] = summary.get("name") or slice_["league_name"]
            slice_["state"] = summary.get("state") or "ready"
            slice_["team_name"] = summary.get("team_name")
            if summary.get("state") not in ("reconnect_required", "team_not_linked"):
                slice_.update({
                    "record": summary.get("record"),
                    "wins": summary.get("wins"), "losses": summary.get("losses"),
                    "rank": summary.get("rank"), "total_teams": summary.get("total_teams"),
                    "points_for": _fnum(summary.get("pf")),
                    "total_value": _fnum(summary.get("total_value")),
                    "streak": summary.get("streak"),
                    "pos_ranks": {k: v for k, v in (summary.get("pos_user_rank") or {}).items()
                                  if v is not None},
                })
            elif summary.get("message"):
                slice_["note"] = summary.get("message")
        slices.append(slice_)
    pack["leagues"] = slices
    pack["leagues_count"] = len(slices)
    return pack, tools


def _activity_players(players, values):
    """Activity-frame player entries -> pack lines priced from the value table."""
    out, total, unpriced = [], 0.0, 0
    for p in players or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("pid") or p.get("player_id") or "")
        v = _row_value(values.get(pid)) if pid else None
        if v is None:
            unpriced += 1
        else:
            total += v
        out.append({"name": p.get("name") or pid, "position": p.get("pos") or p.get("position") or "",
                    "value": v})
    return out, round(total, 1), unpriced


def build_activity(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """League activity: the Activity page's own frame (trades, waiver adds,
    serious injuries), newest first, with completed trades priced from the
    model value table and judged by the shared fairness band."""
    tools: list[str] = []
    pack = _base_pack("activity", bundle)
    ctx = bundle.get("ctx") or {}
    if not ctx:
        pack["available"] = False
        pack["missing"] = "no league context"
        return pack, tools
    try:
        ensure_activity_bits(ctx)
        tools.append("app.ensure_activity_bits")
    except Exception:
        logger.debug("[assistant] ensure_activity_bits failed", exc_info=True)
    df = ctx.get("activity_df")
    if df is None:
        pack["available"] = False
        pack["missing"] = "activity is not available for this league yet"
        return pack, tools
    tools.append("service.build_week_activity")
    try:
        records = df.to_dict("records") if hasattr(df, "to_dict") else list(df)
    except Exception:
        records = []
    values = _value_rows()
    if values:
        tools.append("model_value_table")
    from utils.trade import fairness_label
    viewer_rid = str((bundle.get("roster") or {}).get("roster_id") or "")
    week_now = bundle.get("week")
    trades, adds, injuries = [], [], []
    counts = {"trades": 0, "trades_this_week": 0, "waiver_adds": 0, "injuries": 0}
    for row in records:
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        data = row.get("data") or {}
        wk = row.get("week")
        if kind == "trade":
            counts["trades"] += 1
            if week_now is not None and wk == week_now:
                counts["trades_this_week"] += 1
            if len(trades) >= 5:
                continue
            teams, totals = [], []
            for t in data.get("teams") or []:
                if not isinstance(t, dict):
                    continue
                gets, gets_total, gets_un = _activity_players(t.get("gets"), values)
                sends, sends_total, sends_un = _activity_players(t.get("sends"), values)
                rid = str(t.get("roster_id") or t.get("rid") or "")
                teams.append({
                    "team_name": t.get("name") or f"Roster {rid}",
                    "roster_id": rid,
                    "involves_you": bool(viewer_rid and rid == viewer_rid),
                    "gets": gets, "sends": sends,
                    "gets_value": gets_total if gets else None,
                    "sends_value": sends_total if sends else None,
                    "unpriced": gets_un + sends_un,
                })
                totals.append(gets_total if gets else None)
            entry = {"week": wk, "teams": teams,
                     "picks_moved": len(data.get("draft_picks") or [])}
            priced = [x for x in totals if x is not None]
            if len(teams) == 2 and len(priced) == 2:
                gap = round(abs(priced[0] - priced[1]), 1)
                entry["value_gap"] = gap
                entry["ahead_team"] = teams[0]["team_name"] if priced[0] >= priced[1] \
                    else teams[1]["team_name"]
                entry["fairness"] = "fair" if fairness_label(
                    priced[0] - priced[1], max(priced)) == "fair" else "lopsided"
            trades.append(entry)
        elif kind == "waiver":
            adds_players, _t, _u = _activity_players(data.get("adds"), values)
            drops_players, _t2, _u2 = _activity_players(data.get("drops"), values)
            counts["waiver_adds"] += len(adds_players)
            if len(adds) < 8 and (adds_players or drops_players):
                adds.append({"week": wk, "team_name": data.get("name") or "",
                             "adds": [p["name"] for p in adds_players],
                             "drops": [p["name"] for p in drops_players]})
        elif kind == "injury":
            counts["injuries"] += 1
            if len(injuries) < 8:
                pl = data.get("player") or {}
                injuries.append({"player": pl.get("name") or "",
                                 "status": data.get("status") or "",
                                 "team_name": data.get("team_name") or ""})
    pack["counts"] = counts
    pack["recent_trades"] = trades
    pack["latest_trade"] = trades[0] if trades else None
    pack["recent_adds"] = adds
    pack["injuries"] = injuries
    return pack, tools


def build_watchlist(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """The user's watchlist, read-only: each watched player with his current
    value, 7-day move, alert state (the shared watchlist_alerts threshold
    or a real injury designation), and roster state in this league."""
    tools: list[str] = []
    pack = _base_pack("watchlist", bundle)
    try:
        user_key = watchlist_user_key()
    except Exception:
        user_key = None
    if not user_key:
        pack["available"] = False
        pack["missing"] = "no signed-in account for this chat"
        return pack, tools
    try:
        rows = watchlist_rows(user_key) or []
        tools.append("watchlist_bp._rows_for")
    except Exception:
        logger.debug("[assistant] watchlist rows failed", exc_info=True)
        pack["available"] = False
        pack["missing"] = "your watchlist could not be loaded"
        return pack, tools
    values = _value_rows()
    if values:
        tools.append("model_value_table")
    index = _index()
    delta_map: dict = {}
    try:
        movers = watchlist_top_movers(days=7, limit=2000) or {}
        tools.append("player_value_history.get_top_movers")
        for p in (movers.get("risers") or []) + (movers.get("fallers") or []):
            if not isinstance(p, dict):
                continue
            pid = str(p.get("player_id") or p.get("id") or "")
            if pid and p.get("delta") is not None:
                try:
                    delta_map[pid] = round(float(p["delta"]), 1)
                except (TypeError, ValueError):
                    continue
    except Exception:
        logger.debug("[assistant] watchlist movers failed", exc_info=True)
    from utils.watchlist_alerts import is_value_alert
    ctx = bundle.get("ctx") or {}
    rostered: set = set()
    for r in ctx.get("rosters") or []:
        if isinstance(r, dict):
            rostered.update(str(p) for p in (r.get("players") or []))
    mine = {str(p) for p in (bundle.get("roster_player_ids") or set())}
    players = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("player_id") or "")
        if not pid:
            continue
        info = _player_info(pid, index)
        value = _row_value(values.get(pid))
        delta = delta_map.get(pid)
        injury = info.get("injury_status")
        injury_active = bool(injury) and str(injury).upper() not in ("ACTIVE", "HEALTHY", "NA")
        players.append({
            "player_id": pid,
            "name": row.get("name") or info.get("name") or pid,
            "position": row.get("position") or info.get("position") or "",
            "team": row.get("team") or info.get("team") or "",
            "note": row.get("note") or "",
            "value": value,
            "delta7": delta,
            "injury_status": injury if injury_active else None,
            "alert": bool(is_value_alert(delta, value) or injury_active),
            "on_my_roster": pid in mine,
            "rostered": pid in rostered,
        })
    players.sort(key=lambda p: (not p["alert"],
                                -(abs(p["delta7"]) if p["delta7"] is not None else 0.0)))
    pack["count"] = len(players)
    pack["players"] = players[:12]
    pack["alerts"] = [p["name"] for p in players if p["alert"]]
    movers_only = [p for p in players if p["delta7"] is not None]
    pack["top_mover"] = max(movers_only, key=lambda p: abs(p["delta7"])) if movers_only else None
    return pack, tools
def _fo_players(rows) -> list[dict]:
    out = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        out.append({"name": r.get("name"), "position": r.get("position"),
                    "value": r.get("value") if r.get("value") is not None
                    else r.get("model_value")})
    return out


def _fo_target(t) -> dict | None:
    if not isinstance(t, dict):
        return None
    gets = _fo_players(t.get("gets"))
    gives = _fo_players(t.get("gives"))

    def _total(rows):
        vals = [r["value"] for r in rows if isinstance(r.get("value"), (int, float))]
        return round(sum(vals), 1) if vals else None

    return {"partner": t.get("partner"), "gets": gets, "gives": gives,
            "get_total": _total(gets), "give_total": _total(gives),
            "fairness_label": t.get("fairness_label")}


def build_front_office(bundle, entities=None, player_ids=None) -> tuple[dict, list[str]]:
    """Front Office pack: the GM report's computed data (direction, playoff
    odds, weakest positions, deterministic top moves, trade/waiver targets,
    cut candidates), PRO-gated exactly like /api/gm-memo. Data only: the
    report's written memo is never part of the pack."""
    tools: list[str] = []
    pack = _base_pack("front_office", bundle)
    ctx = bundle.get("ctx") or {}
    roster = bundle.get("roster") or {}
    if not ctx or roster.get("roster_id") is None:
        pack["available"] = False
        pack["missing"] = "your roster in this league"
        return pack, tools
    tools.append("subscriptions.has_premium_for_viewer")
    viewer = bundle.get("viewer") or {}
    try:
        premium = bool(has_premium_for_viewer(
            viewer.get("viewer_username") or viewer.get("username"),
            viewer.get("viewer_user_id") or viewer.get("user_id"),
            bundle.get("league_id"), bundle.get("platform") or "sleeper",
            bundle.get("season")))
    except Exception:
        premium = False
    if not premium:
        pack["gated"] = True
        return pack, tools
    try:
        data = front_office_data(ctx, str(roster["roster_id"]))
        tools.append("ai.front_office_report.build_front_office_data")
    except Exception:
        logger.debug("[assistant] front office data failed", exc_info=True)
        data = None
    if not data:
        pack["available"] = False
        pack["missing"] = "front office data is not available for your team yet"
        return pack, tools
    pack["direction"] = data.get("direction")
    pack["record"] = data.get("record")
    pack["playoff_pct"] = data.get("playoff_pct")
    pack["weakest_positions"] = list(data.get("weakest_positions") or [])
    pack["top_moves"] = [
        {"line": m.get("line"), "type": m.get("type"), "why": m.get("why")}
        for m in (data.get("top_moves") or [])[:3] if isinstance(m, dict)]
    pack["trade_targets"] = [t for t in
                             (_fo_target(t) for t in (data.get("trade_targets") or [])[:3]) if t]
    pack["waiver_targets"] = _fo_players((data.get("waiver_targets") or [])[:3])
    pack["cut_candidates"] = _fo_players((data.get("cut_candidates") or [])[:3])
    return pack, tools


BUILDERS = {
    "start_sit": build_start_sit,
    "trade_eval": build_trade_eval,
    "trade_finding": build_trade_finding,
    "buy_sell_hold": build_buy_sell_hold,
    "player": build_player,
    "metric": build_metric,
    "leaderboard": build_leaderboard,
    "waivers": build_waivers,
    "this_week": build_this_week,
    "playoffs": build_playoffs,
    "power_rankings": build_power_rankings,
    "standings": build_standings,
    "matchup": build_matchup,
    "schedule": build_schedule,
    "breakout_board": build_breakout_board,
    "breakout_track_record": build_breakout_track_record,
    "breakout_history": build_breakout_history,
    "teams": build_teams,
    "pick_ownership": build_pick_ownership,
    "keeper": build_keeper,
    "rookie": build_rookie,
    "cross_league": build_cross_league,
    "replacement": build_replacement,
    "handcuff": build_handcuff,
    "activity": build_activity,
    "watchlist": build_watchlist,
    "front_office": build_front_office,
}
