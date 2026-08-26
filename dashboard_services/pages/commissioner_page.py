"""League Health / commissioner dashboard.

Moved from app.py. App helpers (model values, pick value, superflex) are
lazy-imported at request time to avoid a circular import.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# COMMISSIONER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

_COMMISH_HISTORY_CACHE: dict = {}
_COMMISH_HISTORY_TTL = 6 * 3600  # 6h - prior-season activity barely changes


def _commissioner_history_layer(platform, league_id, season, lookback=3):
    """
    Multi-season league context for the commissioner tool, keyed by owner
    user_id (stable across seasons; roster_id is not). Walks previous_league_id
    for up to `lookback` prior seasons and computes, per prior season, each
    owner's move count + inactivity and the league's total moves/trades.

    Sleeper-only (ESPN has no previous_league_id chain). Cached, and any failure
    returns None so the page degrades to the current-season view.
    """
    import time
    platform = (platform or "sleeper").lower()
    if platform != "sleeper" or not league_id:
        return None

    key = f"{platform}:{league_id}:{season}"
    now = time.time()
    cached = _COMMISH_HISTORY_CACHE.get(key)
    if cached and now - cached["ts"] < _COMMISH_HISTORY_TTL:
        return cached["layer"]

    from dashboard_services.api import build_league_history_map, get_league
    from dashboard_services.platform_api import get_rosters, get_users
    from dashboard_services.service import get_transactions_by_week

    layer = None
    try:
        hist = build_league_history_map(platform, league_id, season) or {}
        prior = sorted(s for s in hist if int(s) < int(season))[-lookback:]
        seasons_out = []
        owners: dict = {}
        for ps in prior:
            lid = hist[ps]
            try:
                rosters = get_rosters(platform, lid, ps) or []
                users = get_users(platform, lid, ps) or []
                league = get_league(lid) or {}
            except Exception:
                continue
            playoff_start = int((league.get("settings") or {}).get("playoff_week_start") or 14)
            try:
                txw = get_transactions_by_week(lid, range(1, playoff_start), platform=platform, season=ps) or {}
            except Exception:
                txw = {}

            moves_by_rid: dict = {}
            trades = 0
            for _wk, txns in txw.items():
                for tx in (txns or []):
                    rids = {str(r) for r in (tx.get("roster_ids") or [])}
                    rids |= {str(v) for v in (tx.get("adds") or {}).values()}
                    rids |= {str(v) for v in (tx.get("drops") or {}).values()}
                    if tx.get("type") == "trade":
                        trades += 1
                    for rid in rids:
                        moves_by_rid[rid] = moves_by_rid.get(rid, 0) + 1

            name_by_uid = {u.get("user_id"): (u.get("display_name") or u.get("username") or u.get("user_id"))
                           for u in users}
            season_moves = 0
            inactive_owners = 0
            for r in rosters:
                rid = str(r.get("roster_id"))
                uid = r.get("owner_id")
                if not uid:
                    continue
                st = r.get("settings") or {}
                games = int(st.get("wins") or 0) + int(st.get("losses") or 0)
                mv = moves_by_rid.get(rid, 0)
                season_moves += mv
                inactive = mv == 0 and games > 3
                inactive_owners += 1 if inactive else 0
                o = owners.setdefault(uid, {"name": "Unknown", "seasons_present": 0, "inactive_seasons": 0})
                o["name"] = name_by_uid.get(uid, o["name"])
                o["seasons_present"] += 1
                o["inactive_seasons"] += 1 if inactive else 0

            seasons_out.append({"season": int(ps), "trades": trades,
                                "moves": season_moves, "inactive_owners": inactive_owners,
                                "n_teams": len(rosters)})

        if seasons_out:
            layer = {"seasons": seasons_out, "owners": owners}
    except Exception:
        layer = None

    _COMMISH_HISTORY_CACHE[key] = {"ts": now, "layer": layer}
    return layer


def _league_activity_targets(layer,
                             fallback_moves_pt=5.0, fallback_trades_pt=1.5,
                             floor_moves_pt=3.0, floor_trades_pt=0.5):
    """Full-season, per-team activity the health score treats as "healthy" for
    THIS league — derived from what it typically does, not a league-agnostic
    constant. Takes the median of each prior season's moves-per-team and
    trades-per-team (per-team so league-size changes don't skew it), floored so
    a chronically dead league can't set an ultra-low bar for itself, and with no
    upper cap so a hyperactive league is judged against its own high pace.

    Returns (moves_per_team, trades_per_team, from_history). Falls back to fixed
    defaults when there's no usable history (ESPN, a league's first season, or a
    lookup failure), so the score degrades to a sensible league-agnostic bar.
    """
    import statistics
    mv_pts, tr_pts = [], []
    for s in (layer or {}).get("seasons") or []:
        nt = s.get("n_teams") or 0
        if nt <= 0:
            continue
        mv_pts.append((s.get("moves") or 0) / nt)
        tr_pts.append((s.get("trades") or 0) / nt)
    if not mv_pts:
        return fallback_moves_pt, fallback_trades_pt, False
    return (max(floor_moves_pt, statistics.median(mv_pts)),
            max(floor_trades_pt, statistics.median(tr_pts)),
            True)


def _render_commissioner_history(layer, current_season, current_moves, current_trades, current_inactive_uids,
                                 current_week=0, playoff_start=14):
    """Render the multi-season 'Chronic / Trend' panel. Current season is the
    headline (computed elsewhere); this is diagnosis context, not a blended score."""
    seasons = sorted((layer.get("seasons") or []), key=lambda s: s["season"])
    # Separate prior (completed) data from current in-progress season
    prior_moves_series = [(s["season"], s["moves"]) for s in seasons]
    prior_trades_series = [(s["season"], s["trades"]) for s in seasons]
    # Season is complete when playoffs have started (or it's a prior year entirely)
    season_complete = int(current_week) >= int(playoff_start)
    moves_series = prior_moves_series + [(int(current_season), int(current_moves))]
    trades_series = prior_trades_series + [(int(current_season), int(current_trades))]

    def _dir(series):
        # Trend is always computed from completed seasons only - never compare
        # a partial/preseason year against full seasons.
        completed = prior_moves_series if series is moves_series else prior_trades_series
        vals = [v for _, v in completed]
        if len(vals) < 2:
            return ("→", "var(--muted)", "flat")
        delta = vals[-1] - vals[0]
        thresh = max(2.0, 0.10 * abs(vals[0] or 1))
        if delta > thresh:
            return ("▲", "#22c55e", "up")
        if delta < -thresh:
            return ("▼", "#ef4444", "down")
        return ("→", "var(--muted)", "flat")

    def _series_html(series):
        parts = []
        last_idx = len(series) - 1
        for i, (yr, val) in enumerate(series):
            is_last = i == last_idx
            if is_last and not season_complete:
                # Current season still in progress - muted style with dashed border
                chip_bg = "var(--row,rgba(127,127,127,.08))"
                chip_fg = "var(--muted)"
                chip_border = "var(--muted)"
                border_style = "dashed"
                yr_label = f"{yr} ↻"
            elif is_last:
                chip_bg = "var(--accent,#3b82f6)"
                chip_fg = "#fff"
                chip_border = "transparent"
                border_style = "solid"
                yr_label = str(yr)
            else:
                chip_bg = "var(--row,rgba(127,127,127,.08))"
                chip_fg = "var(--text)"
                chip_border = "var(--border)"
                border_style = "solid"
                yr_label = str(yr)
            parts.append(
                f"<span class='msh-chip' style='background:{chip_bg};color:{chip_fg};"
                f"border-color:{chip_border};border-style:{border_style};'>"
                f"<span class='msh-chip-yr'>{yr_label}</span>"
                f"<span class='msh-chip-val'>{val}</span></span>"
            )
        connector = "<span class='msh-arrow'>›</span>"
        return connector.join(parts)

    m_arrow, m_color, m_dir = _dir(moves_series)
    t_arrow, t_color, t_dir = _dir(trades_series)

    def _trend_badge(arrow, color, direction):
        label = {"up": "Rising", "down": "Falling", "flat": "Steady"}[direction]
        return (
            f"<span class='msh-trend' style='color:{color};background:{color}1a;'>"
            f"{arrow} {label}</span>"
        )

    owners = layer.get("owners") or {}
    chronic = []
    for uid, o in owners.items():
        windows = o["seasons_present"] + 1  # + current
        inact = o["inactive_seasons"] + (1 if uid in current_inactive_uids else 0)
        if inact >= 2:
            chronic.append((o["name"], inact, windows))
    chronic.sort(key=lambda x: -x[1])

    chronic_rows = "".join(
        f"<div class='msh-chronic-row'>"
        f"<span class='msh-chronic-name'>{name}</span>"
        f"<span class='msh-chronic-flag'>inactive {inact} of last {windows}</span></div>"
        for name, inact, windows in chronic
    ) or (
                       "<div class='msh-chronic-ok'>"
                       "<span class='msh-chronic-ok-icon'>✓</span>"
                       "No chronic inactivity - owners stay engaged across seasons.</div>"
                   )

    n_prior = len(seasons)
    return f"""
<style>
  .msh-card {{ padding:20px; margin-bottom:20px; }}
  .msh-head {{ display:flex; align-items:baseline; gap:8px; margin-bottom:18px; }}
  .msh-head-title {{ font-size:12px; font-weight:800; letter-spacing:.06em; color:var(--text); text-transform:uppercase; }}
  .msh-head-sub {{ font-size:12px; color:var(--muted); }}
  .msh-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin-bottom:18px; }}
  .msh-stat {{ border:1px solid var(--border); border-radius:14px; padding:14px 16px; background:var(--row,rgba(127,127,127,.03)); }}
  .msh-stat-head {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }}
  .msh-stat-label {{ font-size:11px; font-weight:700; letter-spacing:.04em; color:var(--muted); text-transform:uppercase; }}
  .msh-trend {{ font-size:11px; font-weight:700; padding:3px 9px; border-radius:8px; white-space:nowrap; }}
  .msh-series {{ display:flex; align-items:center; flex-wrap:wrap; gap:2px; }}
  .msh-chip {{ display:inline-flex; flex-direction:column; align-items:center; gap:1px; padding:5px 11px; border-radius:10px; border:1px solid; line-height:1.1; }}
  .msh-chip-yr {{ font-size:10px; font-weight:600; opacity:.75; }}
  .msh-chip-val {{ font-size:15px; font-weight:800; }}
  .msh-arrow {{ color:var(--muted); font-size:16px; font-weight:700; padding:0 2px; opacity:.5; }}
  .msh-chronic-label {{ font-size:11px; font-weight:700; letter-spacing:.04em; color:var(--muted); text-transform:uppercase; margin-bottom:8px; }}
  .msh-chronic-row {{ display:flex; align-items:center; justify-content:space-between; padding:9px 12px; border-radius:10px; background:rgba(239,68,68,.06); border:1px solid rgba(239,68,68,.18); margin-bottom:6px; font-size:13px; }}
  .msh-chronic-name {{ font-weight:600; color:var(--text); }}
  .msh-chronic-flag {{ color:#ef4444; font-weight:700; font-size:12px; }}
  .msh-chronic-ok {{ display:flex; align-items:center; gap:8px; padding:11px 14px; border-radius:10px; background:rgba(34,197,94,.07); border:1px solid rgba(34,197,94,.2); color:var(--text); font-size:13px; }}
  .msh-chronic-ok-icon {{ display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px; border-radius:50%; background:#22c55e; color:#fff; font-size:11px; font-weight:800; flex-shrink:0; }}
</style>
<div class="card msh-card">
  <div class="msh-head">
    <span class="msh-head-title">Multi-Season Health</span>
    <span class="msh-head-sub">last {n_prior} prior season{'s' if n_prior != 1 else ''}</span>
  </div>
  <div class="msh-grid">
    <div class="msh-stat">
      <div class="msh-stat-head">
        <span class="msh-stat-label">Total Moves / Season</span>
        {_trend_badge(m_arrow, m_color, m_dir)}
      </div>
      <div class="msh-series">{_series_html(moves_series)}</div>
    </div>
    <div class="msh-stat">
      <div class="msh-stat-head">
        <span class="msh-stat-label">Trades / Season</span>
        {_trend_badge(t_arrow, t_color, t_dir)}
      </div>
      <div class="msh-series">{_series_html(trades_series)}</div>
    </div>
  </div>
  <div class="msh-chronic-label">Chronic Inactivity</div>
  {chronic_rows}
</div>"""


def commissioner_is_inactive(txns, games_played) -> bool:
    """Zero transactions after more than 3 games have been played."""
    return int(txns or 0) == 0 and int(games_played or 0) > 3


def commissioner_value_share_pct(team_value, league_total) -> float:
    """Percent of league value this team holds."""
    total = float(league_total or 0) or 1.0
    return round(float(team_value or 0) / total * 100, 1)


def build_commissioner_body(ctx):
    from dashboard_services.service import get_transactions_by_week
    from dashboard_services.picks import load_pick_value_table
    from app import (
        get_model_value_table_cached,
        _is_superflex_lineup,
        _team_pick_value,
        _safe_int,
    )

    platform = ctx.get("platform") or "sleeper"
    season = ctx.get("season") or datetime.now().year
    league_id = ctx.get("league_id") or ""
    rosters = ctx.get("rosters") or []
    users = ctx.get("users") or []
    roster_map = ctx.get("roster_map") or {}
    traded = ctx.get("traded") or []
    # current_week is on ctx directly; ctx has no "state" key (see build_optimal_body).
    current_week = int(ctx.get("current_week") or 0)
    settings = (ctx.get("league") or {}).get("settings") or {}
    playoff_start = int(settings.get("playoff_week_start") or 14)
    model_vals = ctx.get("model_value_table") or get_model_value_table_cached() or []
    is_sf = _is_superflex_lineup(ctx.get("roster_positions") or [])
    val_key = "sf_value" if is_sf else "value"
    val_by_pid = {str(p.get("id") or ""): float(p.get(val_key) or p.get("value") or 0)
                  for p in model_vals if p.get("id")}

    owner_by_rid = {}
    for u in users:
        uid = u.get("user_id")
        name = u.get("display_name") or u.get("username") or uid
        for r in rosters:
            if r.get("owner_id") == uid:
                owner_by_rid[str(r.get("roster_id"))] = name

    # ── 1. Fetch full-season transactions for real activity data ──────────
    reg_season_weeks = min(current_week or playoff_start - 1, playoff_start - 1)
    tx_by_week: dict = {}
    try:
        tx_by_week = get_transactions_by_week(
            league_id, range(1, max(reg_season_weeks + 1, 2)), platform=platform, season=season
        ) or {}
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    # Per-roster move counts from actual transactions (not roster snapshot)
    txn_by_rid: dict = {}
    trade_count_by_rid: dict = {}
    for wk, txns in tx_by_week.items():
        for tx in (txns or []):
            rids = {str(r) for r in (tx.get("roster_ids") or [])}
            rids |= {str(v) for v in (tx.get("adds") or {}).values()}
            rids |= {str(v) for v in (tx.get("drops") or {}).values()}
            for rid in rids:
                if tx.get("type") == "trade":
                    trade_count_by_rid[rid] = trade_count_by_rid.get(rid, 0) + 1
                else:
                    txn_by_rid[rid] = txn_by_rid.get(rid, 0) + 1

    # ── 2. Roster values & activity ───────────────────────────────────────
    # Roster value share must match the Standings value board exactly: it sums
    # player value (model cache `value`) PLUS draft-pick value as a share of the
    # league total. (val_by_pid stays SF-aware for the trade-fairness section.)
    _share_val_by_pid = {str(p.get("id") or ""): float(p.get("value") or 0)
                         for p in model_vals if p.get("id")}
    _picks_by_roster = ctx.get("picks_by_roster") or {}
    _pick_val_by_key = load_pick_value_table() or {}
    _pv_league_id = str(ctx.get("resolved_league_id") or league_id or "")
    roster_infos = []
    for r in rosters:
        rid = str(r.get("roster_id"))
        pids = [str(p) for p in (r.get("players") or [])]
        _picks = _picks_by_roster.get(rid, []) if isinstance(_picks_by_roster, dict) else []
        val = (sum(_share_val_by_pid.get(pid, 0) for pid in pids)
               + _team_pick_value(_picks, _pick_val_by_key, platform=platform,
                                  league_id=_pv_league_id, season=_safe_int(season, 0)))
        r_st = r.get("settings") or {}
        wins = r_st.get("wins", 0)
        losses = r_st.get("losses", 0)
        # Prefer transaction-API counts; fall back to roster snapshot
        txns = txn_by_rid.get(rid) or int(r_st.get("total_moves") or 0)
        trades = trade_count_by_rid.get(rid, 0)
        games_played = wins + losses
        inactive = commissioner_is_inactive(txns, games_played)
        roster_infos.append({
            "rid": rid, "name": roster_map.get(rid, f"Team {rid}"),
            "owner": owner_by_rid.get(rid, "Unknown"),
            "wins": wins, "losses": losses,
            "value": round(val, 0), "txns": txns, "trades": trades,
            "inactive": inactive,
        })

    # Value share % relative to league total (matches teams page display)
    league_val_total = sum(r["value"] for r in roster_infos) or 1.0
    fair_share = 100.0 / max(len(roster_infos), 1)
    for r in roster_infos:
        r["value_pct"] = commissioner_value_share_pct(r["value"], league_val_total)
    roster_infos.sort(key=lambda x: -x["value_pct"])

    def _val_bar(pct: float) -> str:
        width = min(pct / (fair_share * 2) * 100, 100)
        color = "var(--accent)" if pct >= fair_share else "var(--text-muted)"
        return (f'<div style="height:6px;border-radius:3px;background:var(--border);width:100%;min-width:60px;">'
                f'<div style="height:6px;border-radius:3px;background:{color};width:{width:.1f}%"></div></div>')

    # ── 3. Trade fairness ─────────────────────────────────────────────────
    # Use the same pick value table and logic as the activity page.
    # In Sleeper draft_picks: owner_id = receiver, previous_owner_id = sender,
    # roster_id = original pick owner (not always the sender).
    _pick_tbl = load_pick_value_table() or {}
    _standings_map = ctx.get("standings_map") or {}
    _num_teams = len(rosters) or 10

    def _pick_val_comm(pick: dict) -> float:
        year = int(pick.get("season") or 0)
        rnd = int(pick.get("round") or 0)
        if not year or not rnd:
            return 0.0
        exact_slot = resolve_exact_pick_slot(
            platform=platform, root_league_id=league_id,
            current_season=season, pick=pick,
        )
        if exact_slot is not None:
            k = f"{year}_{rnd}_{exact_slot:02d}"
            if k in _pick_tbl:
                return float(_pick_tbl[k])
        prev = pick.get("previous_owner_id")
        seed = None
        try:
            if prev is not None:
                seed = _standings_map.get(int(prev))
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
        if seed is not None:
            if 1 <= seed <= 3:
                bucket = "early"
            elif seed <= 7:
                bucket = "mid"
            else:
                bucket = "late"
            k = f"{year}_{rnd}_{bucket}"
            if k in _pick_tbl:
                return float(_pick_tbl[k])
        for b in ("mid", "early", "late"):
            k = f"{year}_{rnd}_{b}"
            if k in _pick_tbl:
                return float(_pick_tbl[k])
        k = f"{year}_{rnd}"
        return float(_pick_tbl[k]) if k in _pick_tbl else 0.0

    trade_txns = [tx for txns in tx_by_week.values() for tx in (txns or [])
                  if tx.get("type") == "trade"]
    trade_rows = []
    seen_txn_ids: set = set()
    for tx in trade_txns:
        tx_id = tx.get("transaction_id") or tx.get("id")
        if tx_id:
            if tx_id in seen_txn_ids:
                continue
            seen_txn_ids.add(tx_id)
        adds = tx.get("adds") or {}
        draft_picks = tx.get("draft_picks") or []
        # involved = teams who sent or received something (not the original pick owner)
        involved = {str(rid) for rid in adds.values()}
        for pick in draft_picks:
            for key in ("owner_id", "previous_owner_id"):
                v = str(pick.get(key) or "")
                if v and v != "0":
                    involved.add(v)
        if len(involved) < 2:
            continue
        # Value received by each team: player adds + picks (owner_id = receiver)
        received: dict = {rid: 0.0 for rid in involved}
        for pid, rid in adds.items():
            received[str(rid)] = received.get(str(rid), 0) + val_by_pid.get(str(pid), 0)
        for pick in draft_picks:
            rid = str(pick.get("owner_id") or "")
            if rid and rid in received:
                received[rid] = received.get(rid, 0) + _pick_val_comm(pick)
        vals = sorted(received.items(), key=lambda x: -x[1])
        if len(vals) >= 2:
            (r1, v1), (r2, v2) = vals[0], vals[1]
            diff = abs(v1 - v2)
            # Lopsided is judged by the *ratio* of the two sides (bigger haul /
            # smaller), against a band that TIGHTENS as the trade gets bigger: a
            # 10% gap is trivial on a cheap swap but a real fleece on a
            # blockbuster. The band shrinks linearly from 1.20 on small trades to
            # 1.05 on big ones (calibrated so ~1000-for-900 trips it). A small
            # absolute floor keeps trivial swaps from tripping it.
            total_val = v1 + v2
            _t = max(0.0, min(1.0, (total_val - 200.0) / (1800.0 - 200.0)))
            band = 1.20 + (1.05 - 1.20) * _t  # 1.20 (small) → 1.05 (large)
            lo_val, hi_val = min(v1, v2), max(v1, v2)
            ratio = (hi_val / lo_val) if lo_val > 0 else float("inf")
            trade_rows.append({
                "team_a": roster_map.get(r1, f"Team {r1}"),
                "team_b": roster_map.get(r2, f"Team {r2}"),
                "val_a": round(v1, 0), "val_b": round(v2, 0),
                "diff": round(diff, 0),
                "lopsided": ratio >= band and diff > 15,
            })
    trade_rows.sort(key=lambda x: -x["diff"])

    # ── 4. League health (based on full season activity) ──────────────────
    n = len(roster_infos) or 1
    inactive_count = sum(1 for r in roster_infos if r["inactive"])
    lopsided_count = sum(1 for t in trade_rows if t["lopsided"])
    total_txns = sum(r["txns"] for r in roster_infos)
    total_trades = sum(r["trades"] for r in roster_infos) // 2  # each trade counted per team
    avg_txns = total_txns / n
    trades_per_team = total_trades / n

    # Activity targets are relative to THIS league's own norm, not a fixed bar:
    # a hyperactive dynasty is held to its high pace, a casual league to its low
    # one. Targets are full-season per-team figures from prior seasons; pro-rate
    # them by how far the current season has run so a mid-season league isn't
    # compared against a full year of history.
    try:
        _hist_layer = _commissioner_history_layer(platform, league_id, season)
    except Exception:
        _hist_layer = None
    target_moves_pt, target_trades_pt, targets_from_history = _league_activity_targets(_hist_layer)
    progress = min(1.0, current_week / playoff_start) if playoff_start else 1.0
    progress = max(progress, 0.05)  # guard week 0 / preseason
    exp_moves_pt = target_moves_pt * progress
    exp_trades_pt = target_trades_pt * progress
    moves_ratio = (avg_txns / exp_moves_pt) if exp_moves_pt else 1.0
    trades_ratio = (trades_per_team / exp_trades_pt) if exp_trades_pt else 1.0

    # Score: an *engaged* league is a healthy league. Build up from a baseline
    # by rewarding activity relative to the league's own pace, then dock only for
    # genuine health problems — dead teams and a high *rate* of lopsided trades.
    # Raw trade/move volume is never punished, so a busy league scores high.
    activity_score = 40
    # Moves: full marks at (or above) the league's typical pace so far.
    activity_score += min(30, moves_ratio * 30)
    # Trades: full marks at (or above) the league's typical trade pace so far.
    activity_score += min(30, trades_ratio * 30)
    # Inactive (dead) teams are a real health problem — dock per dead team.
    activity_score -= inactive_count / n * 40
    # Lopsided trades: penalise the *share* of trades that are lopsided, not the
    # count, so trading a lot doesn't cost points — only systematically unfair
    # trading does (max 15-point hit if every trade is lopsided).
    lopsided_ratio = (lopsided_count / total_trades) if total_trades else 0
    activity_score -= lopsided_ratio * 15
    activity_score = round(max(0, min(100, activity_score)), 0)
    score_color = "#22c55e" if activity_score >= 80 else ("#f59e0b" if activity_score >= 60 else "#ef4444")
    score_label = "Healthy" if activity_score >= 80 else ("Watch" if activity_score >= 60 else "At Risk")

    # Composite card: the health score as a hero with a 0-100 track, then the
    # four contributing factors as labelled horizontal bars beneath it. Each bar
    # fill is meaningful and ties back to the score — inactive = share of teams
    # dead, lopsided = share of trades lopsided, trades/moves = pace vs the
    # league's own norm. Green fills read as the healthy direction, red/orange
    # as the unhealthy one.
    _GOOD, _WARN, _BAD = "#22c55e", "#f59e0b", "#ef4444"

    def _lh_bar(label: str, value, fill_pct: float, fill_color: str, val_color: str) -> str:
        fill_pct = max(0, min(100, round(fill_pct)))
        return f"""
    <div class="lh-bar-row">
      <div class="lh-bar-top">
        <span class="lh-bar-label">{label}</span>
        <span class="lh-bar-val" style="color:{val_color};">{value}</span>
      </div>
      <div class="lh-bar-track"><div class="lh-bar-fill" style="width:{fill_pct}%;background:{fill_color};"></div></div>
    </div>"""

    _bars_html = "".join([
        _lh_bar("Inactive Teams", inactive_count, inactive_count / n * 100,
                _BAD, _BAD if inactive_count else _GOOD),
        _lh_bar("Trades This Season", total_trades, min(1.0, trades_ratio) * 100,
                _GOOD, "var(--text)"),
        _lh_bar("Lopsided Trades", lopsided_count, lopsided_ratio * 100,
                _WARN, _WARN if lopsided_count else _GOOD),
        _lh_bar("Total Moves", total_txns, min(1.0, moves_ratio) * 100,
                _GOOD, "var(--text)"),
    ])

    # Ring geometry + a one-line verdict, both from values already computed above
    # (this is presentation only — no new metric is calculated).
    import math as _math
    _ring_r = 54
    _ring_circ = 2 * _math.pi * _ring_r
    _ring_off = _ring_circ * (1 - min(100, max(0, activity_score)) / 100)
    _verdict_bits = [f"{total_trades} trade" + ("s" if total_trades != 1 else ""),
                     f"{total_txns} move" + ("s" if total_txns != 1 else "") + " this season"]
    if inactive_count:
        _verdict_bits.append(f"{inactive_count} inactive team" + ("s" if inactive_count != 1 else ""))
    _verdict = f"{score_label} &middot; " + ", ".join(_verdict_bits) + "."

    health_html = f"""
<style>
  .lh-card {{ padding:20px; margin-bottom:20px; }}
  .lh-hero2 {{ display:grid; grid-template-columns:auto 1fr; gap:22px; align-items:center; }}
  @media (max-width:560px) {{ .lh-hero2 {{ grid-template-columns:1fr; text-align:center; justify-items:center; }} }}
  .lh-ring {{ position:relative; width:128px; height:128px; flex-shrink:0; }}
  .lh-ring svg {{ transform:rotate(-90deg); }}
  .lh-ring-mid {{ position:absolute; inset:0; display:grid; place-content:center; text-align:center; }}
  .lh-ring-num {{ font-size:40px; font-weight:800; line-height:1; }}
  .lh-ring-cap {{ font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin-top:3px; }}
  .lh-pill {{ display:inline-flex; align-items:center; gap:6px; font-size:11.5px; font-weight:800; letter-spacing:.03em; padding:4px 10px; border-radius:8px; text-transform:uppercase; }}
  .lh-verdict {{ font-size:15px; color:var(--muted); margin-top:10px; max-width:48ch; }}
  .lh-verdict b {{ color:var(--text); }}
  .lh-wrap {{ max-width:1000px; margin:0 auto; }}
  .lh-bars {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px 34px; }}
  @media (max-width:640px) {{ .lh-bars {{ grid-template-columns:1fr; gap:15px; }} }}
  .lh-bar-row {{ display:flex; flex-direction:column; gap:7px; }}
  .lh-bar-top {{ display:flex; justify-content:space-between; align-items:baseline; }}
  .lh-bar-label {{ font-size:13px; font-weight:600; color:var(--text); }}
  .lh-bar-val {{ font-size:16px; font-weight:800; line-height:1; }}
  .lh-bar-track {{ height:8px; border-radius:99px; background:var(--border); overflow:hidden; }}
  .lh-bar-fill {{ height:100%; border-radius:99px; min-width:2px; transition:width .3s ease; }}
  .lh-parity {{ position:relative; height:58px; margin:4px 4px 2px; }}
  .lh-parity-axis {{ position:absolute; left:0; right:0; top:33px; height:2px; background:var(--border); border-radius:2px; }}
  .lh-parity-fair {{ position:absolute; top:21px; bottom:6px; width:2px; background:var(--accent); opacity:.55; }}
  .lh-parity-fair span {{ position:absolute; top:-15px; left:50%; transform:translateX(-50%); font-size:10px; font-weight:700; color:var(--muted); white-space:nowrap; }}
  .lh-pdot {{ position:absolute; top:26px; width:14px; height:14px; border-radius:50%; transform:translateX(-50%); border:2px solid var(--card); box-shadow:0 1px 3px rgba(0,0,0,.28); }}
  .lh-parity-scale {{ display:flex; justify-content:space-between; font-size:10.5px; color:var(--muted); margin:2px 4px 12px; }}
</style>
<div class="card lh-card">
  <div class="lh-hero2">
    <div class="lh-ring">
      <svg width="128" height="128" viewBox="0 0 128 128" aria-hidden="true">
        <circle cx="64" cy="64" r="{_ring_r}" fill="none" stroke="var(--border)" stroke-width="10"></circle>
        <circle cx="64" cy="64" r="{_ring_r}" fill="none" stroke="{score_color}" stroke-width="10" stroke-linecap="round" stroke-dasharray="{_ring_circ:.1f}" stroke-dashoffset="{_ring_off:.1f}"></circle>
      </svg>
      <div class="lh-ring-mid"><div class="lh-ring-num" style="color:{score_color};">{int(activity_score)}</div><div class="lh-ring-cap">Health Score</div></div>
    </div>
    <div>
      <span class="lh-pill" style="color:{score_color};background:color-mix(in srgb,{score_color} 15%,transparent);">{score_label}</span>
      <div class="lh-verdict">{_verdict}</div>
    </div>
  </div>
</div>
<div class="card" style="padding:18px 20px;margin-bottom:20px;">
  <div class="card-header" style="margin-bottom:6px;"><h3>What's driving the score</h3></div>
  <div class="lh-bars">{_bars_html}
  </div>
</div>"""

    # ── Roster table with value bar ───────────────────────────────────────
    roster_rows = ""
    for r in roster_infos:
        inactive_badge = (" <span style='background:#ef444420;color:#ef4444;font-size:10px;"
                          "padding:2px 5px;border-radius:4px;'>INACTIVE</span>") if r["inactive"] else ""
        roster_rows += f"""
<tr style="border-bottom:1px solid var(--border);">
  <td style="padding:10px 14px;">
    <div style="font-weight:600;">{html.escape(r['name'])}{inactive_badge}</div>
    <div style="font-size:11px;color:var(--muted);">{html.escape(r['owner'])}</div>
  </td>
  <td style="padding:10px 14px;text-align:center;">{r['wins']}-{r['losses']}</td>
  <td style="padding:10px 14px;min-width:140px;">
    <div style="display:flex;align-items:center;gap:8px;">
      {_val_bar(r['value_pct'])}
      <span style="font-size:12px;font-weight:700;min-width:38px;text-align:right;">{r['value_pct']:.1f}%</span>
    </div>
  </td>
  <td style="padding:10px;text-align:center;">{r['txns']}</td>
  <td style="padding:10px;text-align:center;">{r['trades']}</td>
</tr>"""

    # Parity strip: plot each team's existing value_pct along a thin↔stacked axis
    # with the fair share marked. Pure visualization of the same numbers — no new
    # metric is computed.
    _shares = [r["value_pct"] for r in roster_infos] or [fair_share]
    _p_lo = min(min(_shares), fair_share)
    _p_hi = max(max(_shares), fair_share)
    _p_span = (_p_hi - _p_lo) or 1.0
    _p_lo -= _p_span * 0.12
    _p_hi += _p_span * 0.12
    _p_span = (_p_hi - _p_lo) or 1.0

    def _p_x(s):
        return max(0.0, min(100.0, (s - _p_lo) / _p_span * 100))

    _p_dots = ""
    for r in roster_infos:
        _pc = "#ef4444" if r["inactive"] else ("#22c55e" if abs(r["value_pct"] - fair_share) <= 2 else "#f59e0b")
        _p_dots += (f'<div class="lh-pdot" title="{html.escape(r["name"])} &middot; {r["value_pct"]:.1f}%" '
                    f'style="left:{_p_x(r["value_pct"]):.1f}%;background:{_pc};"></div>')
    parity_html = (
        f'<div style="padding:14px 16px 0;">'
        f'<div class="lh-parity"><div class="lh-parity-axis"></div>'
        f'<div class="lh-parity-fair" style="left:{_p_x(fair_share):.1f}%;"><span>fair {fair_share:.1f}%</span></div>'
        f'{_p_dots}</div>'
        f'<div class="lh-parity-scale"><span>Thin roster</span><span>Fair share</span><span>Stacked roster</span></div>'
        f'</div>'
    )

    roster_table = f"""
<div class="card" style="overflow:auto;margin-bottom:20px;">
  <div class="card-header"><h3>Team Overview</h3></div>
  {parity_html}
  <table style="width:100%;border-collapse:collapse;">
    <thead><tr style="border-bottom:2px solid var(--border);">
      <th style="padding:10px 14px;text-align:left;font-size:12px;color:var(--muted);">TEAM</th>
      <th style="padding:10px;text-align:center;font-size:12px;color:var(--muted);">RECORD</th>
      <th style="padding:10px 14px;text-align:left;font-size:12px;color:var(--muted);">ROSTER VALUE SHARE</th>
      <th style="padding:10px;text-align:center;font-size:12px;color:var(--muted);">MOVES</th>
      <th style="padding:10px;text-align:center;font-size:12px;color:var(--muted);">TRADES</th>
    </tr></thead>
    <tbody>{roster_rows}</tbody>
  </table>
</div>"""

    # ── Trade fairness ────────────────────────────────────────────────────
    if trade_rows:
        trade_items = ""
        for t in trade_rows[:25]:
            diff_color = "#ef4444" if t["lopsided"] else ("#f59e0b" if t["diff"] > 40 else "#22c55e")
            diff_label = "LOPSIDED" if t["lopsided"] else ("UNEVEN" if t["diff"] > 40 else "FAIR")
            max_val = max(t["val_a"], t["val_b"], 1)
            pct_a = t["val_a"] / max_val * 100
            pct_b = t["val_b"] / max_val * 100
            trade_items += f"""
<div style="display:flex;align-items:center;gap:12px;padding:14px 16px;border-bottom:1px solid var(--border);border-left:3px solid {diff_color};">
  <div style="flex:1;min-width:0;">
    <div style="font-size:10px;color:var(--muted);font-weight:600;letter-spacing:.04em;margin-bottom:3px;">SIDE A</div>
    <div style="font-weight:700;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(t['team_a'])}</div>
    <div style="font-size:22px;font-weight:800;color:var(--text);line-height:1.15;margin:2px 0;">{int(t['val_a']):,}</div>
    <div style="height:3px;background:var(--border);border-radius:2px;overflow:hidden;">
      <div style="height:3px;background:var(--accent);border-radius:2px;width:{pct_a:.0f}%;"></div>
    </div>
  </div>
  <div style="text-align:center;flex-shrink:0;">
    <div style="font-size:15px;color:var(--muted);margin-bottom:5px;">&#8644;</div>
    <div style="background:{diff_color}22;color:{diff_color};font-size:12px;font-weight:700;padding:3px 10px;border-radius:10px;white-space:nowrap;">&#177;{int(t['diff']):,}</div>
    <div style="font-size:9px;color:{diff_color};margin-top:3px;font-weight:700;letter-spacing:.06em;">{diff_label}</div>
  </div>
  <div style="flex:1;min-width:0;text-align:right;">
    <div style="font-size:10px;color:var(--muted);font-weight:600;letter-spacing:.04em;margin-bottom:3px;">SIDE B</div>
    <div style="font-weight:700;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(t['team_b'])}</div>
    <div style="font-size:22px;font-weight:800;color:var(--text);line-height:1.15;margin:2px 0;">{int(t['val_b']):,}</div>
    <div style="height:3px;background:var(--border);border-radius:2px;overflow:hidden;display:flex;justify-content:flex-end;">
      <div style="height:3px;background:var(--accent);border-radius:2px;width:{pct_b:.0f}%;"></div>
    </div>
  </div>
</div>"""
        trade_card = f"""
<div class="card" style="overflow:hidden;">
  <div class="card-header">
    <h3>Trade Fairness Log</h3>
    <span style="font-size:12px;color:var(--muted);">Received value per side &middot; &#177;75 = lopsided</span>
  </div>
  {trade_items}
</div>"""
    else:
        trade_card = ("<div class='card'><div class='card-body' style='padding:20px;color:var(--muted);'>"
                      "No trades recorded yet.</div></div>")

    # ── 5. Multi-season layer (chronic inactivity + engagement trend) ─────
    # Current season stays the headline; history is diagnosis, not a blend.
    history_panel = ""
    try:
        layer = _commissioner_history_layer(platform, league_id, season)
        if layer:
            rid_to_uid = {str(r.get("roster_id")): r.get("owner_id") for r in rosters}
            inactive_uids_now = {
                rid_to_uid.get(r["rid"]) for r in roster_infos
                if r["inactive"] and rid_to_uid.get(r["rid"])
            }
            history_panel = _render_commissioner_history(
                layer, current_season=season,
                current_moves=total_txns, current_trades=total_trades,
                current_inactive_uids=inactive_uids_now,
                current_week=current_week, playoff_start=playoff_start,
            )
    except Exception:
        history_panel = ""

    return f'<div class="lh-wrap"><p class="lh-readonly" style="margin:0 0 12px;font-size:13px;color:var(--text-muted);">Read-only analytics for commissioners — nothing here changes the league.</p>{health_html}{history_panel}{roster_table}{trade_card}</div>'


# /<...>/league_health and /<...>/commissioner are served by routes/league_pages_bp.py.
