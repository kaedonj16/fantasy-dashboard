"""League activity feed HTML builder.

Moved from app.py so the Flask monolith can keep shrinking. Helpers that still
live in app.py are lazy-imported inside the builder (request time).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

def build_activity_body(ctx: dict) -> str:
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        _activity_your_players_block,
        _rel_time,
        _safe_int,
        apply_te_premium,
        apply_tier_stack_adjustment,
        format_pick_display_label,
        get_users,
        html,
        load_pick_value_table,
        logger,
        pd,
        pill,
        render_injury_watch,
        resolve_exact_pick_slot,
        te_premium_from_settings,
    )

    league_id = ctx["league_id"]
    resolved_league_id = ctx.get("resolved_league_id", league_id)
    activity_df = ctx["activity_df"]
    injury_df = ctx["injury_df"]
    standings_map = ctx["standings_map"]
    platform = ctx["platform"]
    season = _safe_int(ctx["season"], 0)

    players_values_raw = ctx.get("model_value_table") or []
    _tep = te_premium_from_settings(ctx.get("scoring_settings"))
    player_val_by_key: Dict[Tuple[str, str, str], float] = {}
    player_val_by_key_np: Dict[Tuple[str, str], float] = {}
    rank_label_by_name: Dict[str, str] = {}

    if isinstance(players_values_raw, list):
        for row in players_values_raw:
            if not isinstance(row, dict):
                continue
            raw_name = str(row.get("search_name") or "").strip()
            if not raw_name:
                continue
            name_lower = raw_name.lower()
            pos = str(row.get("position") or row.get("pos") or "").strip().upper()
            team = str(row.get("team") or "").strip().upper()
            if not pos:
                continue
            try:
                val = float(row.get("value") or 0.0)
            except Exception:
                val = 0.0
            val = apply_te_premium(val, pos, _tep)

            player_val_by_key[(name_lower, pos, team)] = val
            player_val_by_key_np[(name_lower, pos)] = val

            lbl = row.get("pos_rank_label") or pos
            rank_label_by_name[name_lower] = str(lbl)

    def player_value(p: dict) -> tuple[float, str]:
        name = str(p.get("name") or "").strip()
        pos = str(p.get("pos") or p.get("position") or "").strip().upper()
        team = str(p.get("team") or "").strip().upper()
        if not name or not pos:
            return 0.0, ""

        # Normalize to match search_name format (no periods, lowercase)
        name_lower = name.lower().replace(".", "")

        val = float(
            player_val_by_key.get((name_lower, pos, team))
            or player_val_by_key_np.get((name_lower, pos), 0.0)
        )

        rank_label = rank_label_by_name.get(name_lower, pos)
        return val, rank_label

    pick_values = load_pick_value_table() or {}

    def pick_bucket_from_seed(seed: Optional[int], num_teams: int = 10) -> Optional[str]:
        if seed is None:
            return None
        if 1 <= seed <= 3:
            return "early"
        if 4 <= seed <= 7:
            return "mid"
        if 8 <= seed <= num_teams:
            return "late"
        return None

    def pick_value(pick: Dict, standings_map: Dict[int, int], num_teams: int = 10) -> float:
        """
        Prefer exact historical slot when available, then fall back to bucketed values.
        """
        year = _safe_int(pick.get("season"), 0)
        rnd = _safe_int(pick.get("round"), 0)
        if not year or not rnd:
            return 0.0

        exact_slot = resolve_exact_pick_slot(
            platform=platform,
            root_league_id=league_id,
            current_season=season,
            pick=pick,
        )

        if exact_slot is not None:
            exact_key = f"{year}_{rnd}_{exact_slot:02d}"
            if exact_key in pick_values:
                return float(pick_values[exact_key])

        prev_owner = pick.get("previous_owner_id")
        seed = None
        try:
            if prev_owner is not None:
                seed = standings_map.get(int(prev_owner))
        except Exception:
            seed = None

        bucket = pick_bucket_from_seed(seed, num_teams=num_teams)

        if bucket:
            key_bucket = f"{year}_{rnd}_{bucket}"
            if key_bucket in pick_values:
                return float(pick_values[key_bucket])

            key_generic = f"{year}_{rnd}"
            if key_generic in pick_values:
                return float(pick_values[key_generic])

        for b in ("mid", "early", "late"):
            key = f"{year}_{rnd}_{b}"
            if key in pick_values:
                return float(pick_values[key])

        key_generic = f"{year}_{rnd}"
        if key_generic in pick_values:
            return float(pick_values[key_generic])

        return 0.0

    def pick_subline(pick: dict, rid_to_name: dict, users: list, num_teams: int = 10) -> str:
        prev_owner = pick.get("previous_owner_id")
        seed = None
        try:
            if prev_owner is not None:
                seed = standings_map.get(int(prev_owner))
        except Exception:
            seed = None

        exact_slot = resolve_exact_pick_slot(
            platform=platform,
            root_league_id=league_id,
            current_season=season,
            pick=pick,
        )

        bucket = pick_bucket_from_seed(seed, num_teams=num_teams)
        bucket_label = None

        if exact_slot is not None:
            bucket_label = f"Pick {pick.get('round')}.{int(exact_slot):02d}"
        elif bucket:
            bucket_label = bucket.capitalize()

        orig_rid = pick.get("roster_id")
        orig_team = rid_to_name.get(orig_rid, f"User {orig_rid}") if orig_rid is not None else "Unknown"
        orig_name = next(
            (
                u.get("display_name")
                for u in users
                if u.get("metadata", {}).get("team_name") == orig_team
            ),
            None
        )

        owner_txt = f"from {orig_name}" if orig_name else "Traded Pick"
        return f"{bucket_label} • {owner_txt}" if bucket_label else owner_txt

    def verdict_from_net(net_total: float, baseline: float = 300.0) -> tuple[str, str]:
        # Dynamic fair band that scales with trade size - matches /api/trade-eval logic.
        if baseline >= 600:
            fair_pct = 0.05
        elif baseline >= 300:
            fair_pct = 0.07
        else:
            fair_pct = 0.10
        fair = max(baseline * fair_pct, 25.0)
        abs_net = abs(net_total)
        if abs_net <= fair:
            return "bract-verdict-even", "Fair"
        if net_total > 0:
            return ("bract-verdict-win", "Strong win") if abs_net > fair * 2 else ("bract-verdict-win", "Slight win")
        return ("bract-verdict-loss", "Strong loss") if abs_net > fair * 2 else ("bract-verdict-loss", "Slight loss")

    trade_count = 0
    waiver_count = 0
    most_active_counts: Dict[str, int] = {}
    traded_asset_counts: Dict[str, int] = {}
    # Parallel maps so the Snapshot leaders (most-active team, most-moved asset)
    # can be made clickable, same as every other name on the page.
    most_active_rid: Dict[str, str] = {}
    traded_asset_pid: Dict[str, str] = {}
    biggest_trade_label = "No trade data"
    biggest_trade_delta = 0.0

    activity_html = ""
    snapshot_html = ""
    if activity_df is not None and not activity_df.empty:

        # ---- League Pulse compact-row helpers -------------------------------
        def _pos_cls(p) -> tuple:
            raw = str(p.get("pos") or p.get("position") or "").strip().upper()
            for cand in ("QB", "RB", "WR", "TE"):
                if raw.startswith(cand):
                    return cand, cand.lower()
            if raw.startswith("K"):
                return "K", "pk"
            if "DEF" in raw or raw.startswith("DST"):
                return "DEF", "pk"
            return (raw[:3] or "FLX"), "pk"

        def _player_chip(p, io) -> str:
            name = html.escape(str(p.get("name") or "").strip())
            if not name:
                return ""
            pos_txt, pos_cls = _pos_cls(p)
            sign = "+" if io == "add" else "−"
            drop = "" if io == "add" else " act-chip-drop"
            # Make the chip open the player modal (same delegated handler as every
            # other player name on the site). pid is always present for real
            # players; draft-pick chips have none, so they stay non-clickable.
            pid = str(p.get("pid") or p.get("player_id") or "").strip()
            click_cls = " player-clickable" if pid else ""
            click_attrs = (
                f" style='cursor:pointer;'"
                f" data-player-id='{html.escape(pid, quote=True)}' data-player-name='{name}'"
                if pid else ""
            )
            return (
                f"<span class='act-chip{drop}{click_cls}'{click_attrs}><span class='act-sign'>{sign}</span>"
                f"<span class='act-pos {pos_cls}'>{html.escape(pos_txt)}</span>{name}</span>"
            )

        def _tm_img(tm) -> str:
            av = tm.get("avatar") or ""
            return (
                f"<img class='avatar' src='{av}' alt='' loading='lazy' decoding='async' "
                "onerror=\"this.style.display='none'\">"
                if av else ""
            )

        def _av_disc(name, img_html) -> str:
            if img_html:
                return f"<span class='act-av'>{img_html}</span>"
            initials = "".join(w[0] for w in str(name or "").split()[:2]).upper() or "?"
            hue = sum(ord(c) for c in str(name or "")) % 360  # deterministic per team
            return f"<span class='act-av act-av-ph' style='--h:{hue}'>{html.escape(initials)}</span>"

        def html_trade(txrow):
            nonlocal trade_count, biggest_trade_label, biggest_trade_delta, season

            data = txrow["data"]
            teams = data["teams"]
            users = get_users(platform, resolved_league_id, season)

            rid_to_name = {}
            for tm in teams:
                rid = tm.get("roster_id")
                if rid is not None:
                    rid_to_name[rid] = tm.get("name") or f"Team {rid}"
                team_name = tm.get("name") or f"Team {rid}"
                most_active_counts[team_name] = most_active_counts.get(team_name, 0) + 1
                if rid is not None:
                    most_active_rid.setdefault(team_name, str(rid))

            trade_count += 1

            def render_player_row(p, io_class):
                name = str(p.get("name") or "").strip()
                if name:
                    traded_asset_counts[name] = traded_asset_counts.get(name, 0) + 1
                    _rp_pid = str(p.get("pid") or p.get("player_id") or "").strip()
                    if _rp_pid:
                        traded_asset_pid.setdefault(name, _rp_pid)

                val, pos_rank_label = player_value(p)
                val_txt = f"{val:.1f}" if val > 0 else ""
                val_html = f'<div class="player-trade-value">{val_txt}</div>' if val_txt else ""

                pid = p.get("pid", "")
                clickable_attrs = (
                    f" class='player-clickable' style='cursor:pointer;font-weight:600;'"
                    f" data-player-id='{pid}' data-player-name='{name}'"
                    if pid else " style='font-weight:600'"
                )

                return (
                    "<div class='player-activity'>"
                    "<div style='gap: 10px;display: flex;align-items: center;'>"
                    f"<span class='io {io_class}'>"
                    f"{'+' if io_class == 'add' else '−'}</span>"
                    "<div>"
                    f"  <div{clickable_attrs}>{name}</div>"
                    f"  <div style='color:var(--text-muted);font-size:12px'>{pos_rank_label} • {p.get('team', '')}</div>"
                    "</div></div>"
                    f"{val_html}</div>"
                )

            def render_pick_row(pick, io_class):
                import json as _json
                traded_asset_counts["Draft Pick"] = traded_asset_counts.get("Draft Pick", 0) + 1

                pick_label = format_pick_display_label(
                    platform=platform,
                    root_league_id=league_id,
                    current_season=season,
                    pick=pick,
                )
                subline = pick_subline(pick, rid_to_name, users)
                val = pick_value(pick, standings_map)
                val_txt = f"{val:.1f}" if val > 0 else ""
                val_html = f'<div class="player-trade-value">{val_txt}</div>' if val_txt else ""

                yr = _safe_int(pick.get("season"), 0)
                rnd = _safe_int(pick.get("round"), 0)
                _pv = pick_values
                tier_vals = {
                    "early": float(_pv.get(f"{yr}_{rnd}_early") or _pv.get(f"{yr}_{rnd}") or 0),
                    "mid": float(_pv.get(f"{yr}_{rnd}_mid") or _pv.get(f"{yr}_{rnd}") or 0),
                    "late": float(_pv.get(f"{yr}_{rnd}_late") or _pv.get(f"{yr}_{rnd}") or 0),
                }
                pick_data = _json.dumps({
                    "label": pick_label,
                    "season": yr,
                    "round": rnd,
                    "value": round(val, 1),
                    "tiers": tier_vals,
                }, separators=(",", ":"))
                pick_data_attr = pick_data.replace('"', '&quot;')

                return (
                    "<div class='player-activity'>"
                    "<div style='gap: 10px;display: flex;align-items: center;'>"
                    f"<span class='io {io_class}'>"
                    f"{'+' if io_class == 'add' else '−'}</span>"
                    "<div>"
                    f"  <div style='font-weight:600'>{pick_label}</div>"
                    f"  <div style='color:var(--text-muted);font-size:12px'>{subline}</div>"
                    "</div></div>"
                    f"{val_html}</div>"
                )

            draft_picks = data.get("draft_picks", []) or []
            picks_by_receiver = {}
            picks_by_sender = {}
            for dp in draft_picks:
                recv = dp.get("owner_id")
                send = dp.get("previous_owner_id")
                if recv is not None:
                    picks_by_receiver.setdefault(recv, []).append(dp)
                if send is not None:
                    picks_by_sender.setdefault(send, []).append(dp)

            side_map: Dict[int, Dict] = {}
            for tm in teams:
                rid = tm.get("roster_id")
                if rid is None:
                    continue
                in_players = tm.get("gets") or []
                in_picks = picks_by_receiver.get(rid, []) or []

                in_player_pairs = [player_value(p) for p in in_players]
                in_player_vals = [v for (v, _label) in in_player_pairs]
                in_pick_vals = [pick_value(pk, standings_map) for pk in in_picks]

                raw_players_total = sum(in_player_vals)
                raw_picks_total = sum(in_pick_vals)
                raw_total = raw_players_total + raw_picks_total

                side_map[rid] = {
                    "raw_total": raw_total,
                    "raw_players_total": raw_players_total,
                    "raw_picks_total": raw_picks_total,
                    "player_values": in_player_vals,
                    "asset_count": len(in_players) + len(in_picks),
                    "breakdown": [],
                    "adjustment": 0.0,
                    "effective_total": raw_total,
                }

            if len(side_map) == 2:
                rid_list = list(side_map.keys())
                side_a = side_map[rid_list[0]]
                side_b = side_map[rid_list[1]]
                # Match /api/trade-eval: only apply depth penalty when both sides
                # have assets and one side sends more than the other. A one-sided
                # trade has no comparison, and penalising both sides of an
                # equal-count trade just zeroes out the adjustment and produces
                # misleadingly negative totals for both teams.
                if (side_a["asset_count"] > 0 and side_b["asset_count"] > 0
                        and side_a["asset_count"] != side_b["asset_count"]):
                    apply_tier_stack_adjustment(side_a, side_b)
                # Pre-compute zero-sum net values so both sides are mirrors of
                # each other - exactly how trade-eval reports the result.
                a_eff = side_a["effective_total"]
                b_eff = side_b["effective_total"]
                baseline_val = max(a_eff, b_eff, 1.0)
                side_a["net_total"] = a_eff - b_eff
                side_b["net_total"] = b_eff - a_eff
                side_a["baseline"] = baseline_val
                side_b["baseline"] = baseline_val

            net_values = []

            cols = []
            for tm in teams:
                roster_id = tm.get("roster_id")

                gets_parts = []
                for p in (tm.get("gets") or []):
                    gets_parts.append(render_player_row(p, "add"))
                gets_players = "".join(gets_parts)

                gets_pick_parts = []
                if roster_id is not None:
                    for pick in picks_by_receiver.get(roster_id, []):
                        gets_pick_parts.append(render_pick_row(pick, "add"))
                gets_picks = "".join(gets_pick_parts)
                gets = gets_players + gets_picks
                if not gets:
                    gets = "<div class='bract-empty-mini'>No incoming assets</div>"

                sends_parts = []
                for p in (tm.get("sends") or []):
                    sends_parts.append(render_player_row(p, "drop"))
                sends_players = "".join(sends_parts)

                sends_pick_parts = []
                if roster_id is not None:
                    for pick in picks_by_sender.get(roster_id, []):
                        sends_pick_parts.append(render_pick_row(pick, "drop"))
                sends_picks = "".join(sends_pick_parts)
                sends = sends_players + sends_picks

                side_info = side_map.get(roster_id)
                net_total = side_info["net_total"] if side_info else 0.0
                baseline = side_info["baseline"] if side_info else 300.0
                net_values.append((tm.get("name", ""), net_total))

                verdict_cls, verdict_txt = verdict_from_net(net_total, baseline)
                net_num_cls = (
                    "bract-net-pos" if net_total > 0 else
                    "bract-net-neg" if net_total < 0 else
                    "bract-net-even"
                )

                total_html = (
                    "<div class='trade-total-row bract-total-row'>"
                    "<hr style='margin-top:8px;margin-bottom:8px;border:none;border-top:1px solid #e2e8f0;'>"
                    "<div class='bract-total-head'>"
                    "<span>Total Value</span>"
                    f"<span class='{net_num_cls}'>{net_total:.0f}</span>"
                    "</div>"
                    f"<div class='bract-verdict {verdict_cls}'>{verdict_txt}</div>"
                    "</div>"
                )

                avatar = tm.get("avatar") or ""
                img = (
                    f"<img class='avatar' src='{avatar}' alt='' loading='lazy' decoding='async' "
                    "onerror=\"this.style.display='none'\">"
                    if avatar else ""
                )
                team_name = tm.get('name', '')
                roster_id = tm.get('roster_id', '')
                esc_name = html.escape(team_name)
                esc_name_attr = html.escape(team_name, quote=True)
                cols.append(
                    "<div class='team-col'>"
                    f"  <header>{img}<div class='team-name team-clickable' style='cursor:pointer;' data-roster-id='{roster_id}' data-team-name='{esc_name_attr}'>{esc_name}</div></header>"
                    f"  <div class='plist'>{gets}{sends}{total_html}</div>"
                    "</div>"
                )

            if len(net_values) == 2:
                delta = abs(net_values[0][1] - net_values[1][1])
                if delta > biggest_trade_delta:
                    biggest_trade_delta = delta
                    biggest_trade_label = f"{net_values[0][0]} vs {net_values[1][0]}"

            when = _rel_time(txrow["ts"]) if pd.notna(txrow["ts"]) else ""
            # Build data payload for outcome check (sent/received per team)
            trade_date_str = ""
            if pd.notna(txrow["ts"]):
                trade_date_str = txrow["ts"].strftime("%Y-%m-%d")

            outcome_data = []
            for tm in teams:
                rid = tm.get("roster_id")

                # Include players
                gets_pids = [{"id": str(p.get("pid") or ""), "name": str(p.get("name") or "")} for p in
                             (tm.get("gets") or []) if p.get("pid")]
                sends_pids = [{"id": str(p.get("pid") or ""), "name": str(p.get("name") or "")} for p in
                              (tm.get("sends") or []) if p.get("pid")]

                # Include picks with asset_type and pick details
                gets_picks = []
                for pick in picks_by_receiver.get(rid, []):
                    season = pick.get('season', '')
                    round_num = pick.get('round', '')
                    roster_id = pick.get('roster_id', '')

                    # Try to resolve exact slot from roster_id
                    exact_slot = None
                    if roster_id:
                        try:
                            exact_slot = resolve_exact_pick_slot(platform, resolved_league_id, int(season), pick)
                        except Exception:
                            logger.debug("suppressed exception", exc_info=True)

                    # Use exact slot if available, otherwise fall back to mid
                    if exact_slot:
                        pick_id = f"{season} {round_num}.{exact_slot:02d}"
                        display_name = pick_id
                        slot_value = exact_slot
                    else:
                        pick_id = f"{season} {round_num}.{roster_id}" if roster_id else f"{season} {round_num}.XX"
                        display_name = f"{season} {round_num} (Mid)"
                        slot_value = None

                    gets_picks.append({
                        "id": pick_id,
                        "name": display_name,
                        "asset_type": "pick",
                        "pick_season": season,
                        "pick_round": round_num,
                        "pick_order": pick.get("order"),
                        "pick_slot": slot_value,
                    })

                sends_picks = []
                for pick in picks_by_sender.get(rid, []):
                    season = pick.get('season', '')
                    round_num = pick.get('round', '')
                    roster_id = pick.get('roster_id', '')

                    # Try to resolve exact slot from roster_id
                    exact_slot = None
                    if roster_id:
                        try:
                            exact_slot = resolve_exact_pick_slot(platform, resolved_league_id, int(season), pick)
                        except Exception:
                            logger.debug("suppressed exception", exc_info=True)

                    # Use exact slot if available, otherwise fall back to mid
                    if exact_slot:
                        pick_id = f"{season} {round_num}.{exact_slot:02d}"
                        display_name = pick_id
                        slot_value = exact_slot
                    else:
                        pick_id = f"{season} {round_num}.{roster_id}" if roster_id else f"{season} {round_num}.XX"
                        display_name = f"{season} {round_num} (Mid)"
                        slot_value = None

                    sends_picks.append({
                        "id": pick_id,
                        "name": display_name,
                        "asset_type": "pick",
                        "pick_season": season,
                        "pick_round": round_num,
                        "pick_order": pick.get("order"),
                        "pick_slot": slot_value,
                    })

                # Combine players and picks
                all_gets = gets_pids + gets_picks
                all_sends = sends_pids + sends_picks

                outcome_data.append(
                    {"roster_id": rid, "team_name": tm.get("name", ""), "gets": all_gets, "sends": all_sends})

            import json as _json
            outcome_json = _json.dumps(outcome_data).replace('"', '&quot;')
            outcome_btn = (
                f"<button class='outcome-check-btn' "
                f"data-trade-teams='{outcome_json}' "
                f"data-trade-date='{trade_date_str}' "
                f"onclick='checkTradeOutcome(this)'>Check Outcome</button>"
            )
            outcome_result_id = f"outcome_{trade_count}"

            # Compact League Pulse summary row (expands to the full breakdown).
            swing_html = ""
            if len(net_values) == 2:
                _sd = abs(net_values[0][1] - net_values[1][1])
                if _sd > 0:
                    swing_html = f"<span class='act-swing'>Δ {_sd:.0f}</span>"
            team_a = teams[0] if teams else {}
            team_b = teams[1] if len(teams) > 1 else {}
            a_gets = "".join(_player_chip(p, "add") for p in (team_a.get("gets") or []))
            a_sends = "".join(_player_chip(p, "drop") for p in (team_a.get("sends") or []))
            _sep = "<span class='act-arrow'>⇄</span>" if (a_gets and a_sends) else ""
            trade_chips = a_gets + _sep + a_sends
            name_a = html.escape(str(team_a.get("name") or "Team A"))
            name_b = html.escape(str(team_b.get("name") or "Team B"))
            return (
                "<details class='tx activity-item act-row act-trade' data-kind='trade'>"
                "  <summary class='act-trade-sum'>"
                f"    {_av_disc(team_a.get('name'), _tm_img(team_a))}"
                "    <div class='act-rmain'>"
                "      <div class='act-rtop'>"
                "        <span class='act-kindtag trade'>Trade</span>"
                f"        <span class='act-tm'>{name_a}</span><span class='act-verb trade'>⇄</span><span class='act-tm'>{name_b}</span>"
                "      </div>"
                f"      <div class='act-chips'>{trade_chips}</div>"
                "    </div>"
                f"    <div class='act-rend'>{swing_html}<span class='act-caret'>▾</span></div>"
                "  </summary>"
                "  <div class='act-trade-body'>"
                f"    <div class='meta'>{pill('Trade completed')} • {when}{outcome_btn}</div>"
                f"    <div class='teams'>{''.join(cols)}</div>"
                f"    <div id='{outcome_result_id}' class='trade-outcome-result' style='display:none;'></div>"
                "  </div>"
                "</details>"
            )

        def html_waiver(txrow):
            nonlocal waiver_count

            d = txrow["data"]
            team_name = d.get("name") or "Unknown Team"
            # Waiver rows carry the roster id under "rid" (trades use "roster_id");
            # fall back so the team name actually opens the team modal.
            roster_id = d.get("roster_id") or d.get("rid") or ""
            most_active_counts[team_name] = most_active_counts.get(team_name, 0) + 1
            if roster_id != "":
                most_active_rid.setdefault(team_name, str(roster_id))
            waiver_count += 1

            avatar = d.get("avatar") or ""
            img = (
                f"<img class='avatar' src='{avatar}' alt='' loading='lazy' decoding='async' "
                "onerror=\"this.style.display='none'\">"
                if avatar else ""
            )
            chip_parts = []
            total_val = 0.0
            for p in d.get("adds", []):
                name = str(p.get("name") or "").strip()
                if name:
                    traded_asset_counts[name] = traded_asset_counts.get(name, 0) + 1
                    _wp_pid = str(p.get("pid") or p.get("player_id") or "").strip()
                    if _wp_pid:
                        traded_asset_pid.setdefault(name, _wp_pid)
                val, _lbl = player_value(p)
                total_val += val
                chip_parts.append(_player_chip(p, "add"))
            chips = "".join(c for c in chip_parts if c) or "<span class='act-chip'>No adds recorded</span>"
            val_txt = f"{total_val:.1f}" if total_val > 0 else ""
            val_html = f"<span class='act-val'>{val_txt}</span>" if val_txt else ""

            esc_wv_name = html.escape(team_name)
            esc_wv_name_attr = html.escape(team_name, quote=True)
            return (
                "<div class='tx activity-item act-row' data-kind='waiver'>"
                f"  {_av_disc(team_name, img)}"
                "  <div class='act-rmain'>"
                "    <div class='act-rtop'>"
                "      <span class='act-kindtag waiver'>Waiver</span>"
                f"      <span class='act-tm team-clickable' style='cursor:pointer;' data-roster-id='{roster_id}' data-team-name='{esc_wv_name_attr}'>{esc_wv_name}</span>"
                "      <span class='act-verb'>claimed</span>"
                "    </div>"
                f"    <div class='act-chips'>{chips}</div>"
                "  </div>"
                f"  <div class='act-rend'>{val_html}</div>"
                "</div>"
            )

        def _day_label(ts) -> str:
            """Human day header for the timeline: Today / Yesterday / weekday / date."""
            try:
                if pd.isna(ts):
                    return "Earlier"
                d = pd.Timestamp(ts)
            except Exception:
                return "Earlier"
            try:
                if d.tzinfo is not None:
                    d = d.tz_localize(None)
            except Exception:
                pass
            days = (pd.Timestamp.now().normalize() - d.normalize()).days
            if days <= 0:
                return "Today"
            if days == 1:
                return "Yesterday"
            if days < 7:
                return d.strftime("%A")
            return d.strftime("%b %d").replace(" 0", " ")

        def _ts_key(ts) -> int:
            try:
                if pd.isna(ts):
                    return -1
                return int(pd.Timestamp(ts).value)
            except Exception:
                return -1

        # Merge trades + waivers into one stream, newest first, grouped under day
        # headers so the feed reads as a chronological timeline (League Pulse).
        dated_cards = []
        for _, row in activity_df.iterrows():
            _card = html_trade(row) if row["kind"] == "trade" else html_waiver(row)
            dated_cards.append((row.get("ts"), _card))
        dated_cards.sort(key=lambda it: _ts_key(it[0]), reverse=True)

        _day_groups: list = []
        for _ts, _card in dated_cards:
            _lbl = _day_label(_ts)
            if not _day_groups or _day_groups[-1][0] != _lbl:
                _day_groups.append((_lbl, []))
            _day_groups[-1][1].append(_card)
        cards_html = "".join(
            f"<div class='act-daygroup'><div class='act-dayhdr'>{html.escape(lbl)}</div>{''.join(cs)}</div>"
            for lbl, cs in _day_groups
        )

        most_active_team = max(most_active_counts.items(), key=lambda x: x[1])[0] if most_active_counts else "None"
        most_moved_asset = max(traded_asset_counts.items(), key=lambda x: x[1])[0] if traded_asset_counts else "None"

        # Make the two Snapshot leaders open their modals, same as the feed.
        def _snap_team_html(nm: str) -> str:
            rid = most_active_rid.get(nm)
            esc = html.escape(nm)
            if not rid or nm == "None":
                return esc
            return (
                f"<span class='team-clickable' style='cursor:pointer;' "
                f"data-roster-id='{html.escape(str(rid), quote=True)}' "
                f"data-team-name='{html.escape(nm, quote=True)}'>{esc}</span>"
            )

        def _snap_player_html(nm: str) -> str:
            pid = traded_asset_pid.get(nm)
            esc = html.escape(nm)
            if not pid or nm in ("None", "Draft Pick"):
                return esc
            return (
                f"<span class='player-clickable' style='cursor:pointer;' "
                f"data-player-id='{html.escape(str(pid), quote=True)}' "
                f"data-player-name='{html.escape(nm, quote=True)}'>{esc}</span>"
            )

        # The four big stat tiles + spotlight banner collapse into one compact
        # rail card, so the timeline is the first thing you see.
        _swing = f" · {round(biggest_trade_delta, 1)}" if biggest_trade_delta > 0 else ""
        snapshot_html = (
            "<div class='card small act-snapshot'>"
            "  <div class='card-header'><h3>Snapshot</h3></div>"
            "  <div class='act-snap-body'>"
            f"    <div class='act-snap-stat'><span class='act-snap-n'>{trade_count}</span><span class='act-snap-l'>Trades</span></div>"
            f"    <div class='act-snap-stat'><span class='act-snap-n'>{waiver_count}</span><span class='act-snap-l'>Waivers</span></div>"
            "  </div>"
            f"  <div class='act-snap-line'><span>Biggest swing</span><strong>{html.escape(biggest_trade_label)}{_swing}</strong></div>"
            f"  <div class='act-snap-line'><span>Most active</span><strong>{_snap_team_html(most_active_team)}</strong></div>"
            f"  <div class='act-snap-line'><span>Most-moved asset</span><strong>{_snap_player_html(most_moved_asset)}</strong></div>"
            "</div>"
        )

        activity_html = (
            "<div class='card activity-card' data-section='activity'>"
            "  <div class='card-header-row'>"
            "    <div>"
            "      <h2>League activity</h2>"
            "    </div>"
            "  </div>"
            "  <div class='scroll-box'>"
            "    <div class='feed'>"
            f"      {cards_html}"
            "    </div>"
            "  </div>"
            "</div>"
        )

    injury_html = ""
    if injury_df is not None and not injury_df.empty:
        injury_html = render_injury_watch(injury_df)
        # Float the signed-in viewer's own injured players to the top with a badge
        # (client-side, cache-safe — see helper). Skipped silently on any issue.
        try:
            _act_roster_pids = {
                str(r.get("roster_id")): [str(p) for p in (r.get("players") or [])]
                for r in (ctx.get("rosters") or [])
            }
            if _act_roster_pids:
                injury_html += _activity_your_players_block(_act_roster_pids)
        except Exception:
            logger.debug("activity your-players block skipped", exc_info=True)
    else:
        injury_html = (
            "<div class='card'>"
            "  <div class='card-body'>"
            "    <div class='bract-empty-state'>"
            "      <div class='bract-empty-icon'><i class='fa-solid fa-shield-halved' style='font-size:28px;color:var(--muted);opacity:.5;'></i></div>"
            "      <div class='bract-empty-title'>No injury updates right now</div>"
            "      <div class='bract-empty-copy'>Either the feed is quiet or there are no currently tracked injury updates for this view.</div>"
            "    </div>"
            "  </div>"
            "</div>"
        )

    if not activity_html:
        activity_html = (
            "<div class='card'>"
            "  <div class='card-body'>"
            "    <div class='bract-empty-state'>"
            "      <div class='bract-empty-icon'><i class='fa-solid fa-arrows-rotate' style='font-size:28px;color:var(--muted);opacity:.5;'></i></div>"
            "      <div class='bract-empty-title'>No recent activity yet</div>"
            "      <div class='bract-empty-copy'>When trades and waiver claims come through, they'll show up here with value context and team-by-team breakdowns.</div>"
            "    </div>"
            "  </div>"
            "</div>"
        )

    return f"""
    <style>
      /* Three columns like the Dashboard: NFL News · feed · Snapshot+Injuries. */
      .activity-page.act-pulse.page-layout {{ grid-template-columns: 300px minmax(0, 1fr) 320px; gap: 20px; align-items: start; }}
      @media (max-width: 1100px) {{
        /* Drop the news rail below the feed; keep feed + right rail side by side. */
        .activity-page.act-pulse.page-layout {{ grid-template-columns: minmax(0, 1fr) 300px; }}
        .act-news-col {{ grid-column: 1 / -1; order: 3; }}
        .act-pulse-main {{ order: 1; }}
        .act-pulse-rail {{ order: 2; }}
      }}
      @media (max-width: 760px) {{
        .activity-page.act-pulse.page-layout {{ grid-template-columns: 1fr; }}
      }}
      /* Injury filter now sits with its list (was in the feed toolbar). */
      .act-injfilter {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 2px 2px 10px; }}
      /* ── Mobile tab switcher: Activity / News / League ─────────────────────
         On phones the three columns stacked (news landed on top); instead show a
         segmented tab bar and reveal one section at a time. Desktop is untouched. */
      .act-mtabs {{ display: none; }}
      @media (max-width: 760px) {{
        .act-mtabs {{
          display: flex; gap: 4px; padding: 4px; margin: 0 0 14px;
          background: var(--accent-soft); border: 1px solid var(--border); border-radius: 12px;
        }}
        .act-mtab {{
          flex: 1; border: 0; background: transparent; color: var(--text-muted);
          font-weight: 700; font-size: 13.5px; padding: 9px 8px; border-radius: 9px; cursor: pointer;
          -webkit-tap-highlight-color: transparent;
          transition: background .15s, color .15s, box-shadow .15s;
        }}
        .act-mtab:active {{ transform: scale(.97); }}
        .act-mtab.active {{ background: var(--card); color: var(--text); box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
        /* Reveal only the active tab's section. */
        .act-pulse[data-mtab="feed"]   .act-news-col,
        .act-pulse[data-mtab="feed"]   .act-pulse-rail,
        .act-pulse[data-mtab="news"]   .act-pulse-main,
        .act-pulse[data-mtab="news"]   .act-pulse-rail,
        .act-pulse[data-mtab="league"] .act-pulse-main,
        .act-pulse[data-mtab="league"] .act-news-col {{ display: none; }}
        /* Drop the desktop sticky/own-scroll rail behaviour on phones. */
        .act-news-col, .act-pulse-rail {{ position: static; max-height: none; overflow: visible; }}
        .act-news-col .card-body {{ max-height: none; overflow: visible; }}
      }}
      @media (prefers-reduced-motion: reduce) {{ .act-mtab:active {{ transform: none; }} }}
      /* Left news rail — sticky, own scroll, like the dashboard's snapshot rail. */
      @media (min-width: 1101px) {{
        .act-news-col {{ position: sticky; top: 90px; }}
      }}
      .act-news-col .card-body {{ max-height: calc(100vh - 190px); overflow-y: auto; overscroll-behavior: contain; }}

      /* ── Filter toolbar ─────────────────────────────────────────────── */
      .act-toolbar {{
        display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
        padding: 12px 14px; background: var(--card); border: 1px solid var(--border);
        border-radius: 12px; margin-bottom: 14px;
      }}
      .act-tb-label {{
        font-size: 10px; font-weight: 800; letter-spacing: .11em; text-transform: uppercase;
        color: var(--text-subtle, var(--text-muted));
      }}
      .act-tb-div {{ width: 1px; align-self: stretch; background: var(--border); margin: 2px 0; }}
      /* Show: filled multi-select toggles with icon + count */
      .act-tgl {{
        display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--border);
        background: var(--card); color: var(--text-muted); font-weight: 700; font-size: 13px;
        padding: 7px 12px 7px 11px; border-radius: 10px; cursor: pointer;
        transition: background .15s, border-color .15s, color .15s, box-shadow .15s;
      }}
      .act-tgl:hover {{ border-color: var(--accent); color: var(--text); }}
      .act-tgl svg {{ width: 15px; height: 15px; flex: 0 0 auto; }}
      .act-tgl .act-tgl-cnt {{
        font-size: 11px; font-weight: 800; background: var(--accent-soft); color: var(--text-muted);
        border-radius: 12px; padding: 1px 7px; font-variant-numeric: tabular-nums; line-height: 1.5;
      }}
      .act-tgl.active {{ background: var(--accent); border-color: var(--accent); color: #fff; box-shadow: 0 3px 10px rgba(18,45,75,.20); }}
      .act-tgl.active .act-tgl-cnt {{ background: rgba(255,255,255,.22); color: #fff; }}
      /* Injuries: connected segmented control */
      .act-segctrl {{ display: inline-flex; background: var(--accent-soft); border: 1px solid var(--border); border-radius: 11px; padding: 3px; gap: 2px; }}
      .act-segctrl button {{
        border: 0; background: transparent; color: var(--text-muted); font-weight: 700; font-size: 12.5px;
        padding: 6px 11px; border-radius: 8px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
        transition: background .15s, color .15s, box-shadow .15s;
      }}
      .act-segctrl button:hover {{ color: var(--text); }}
      .act-segctrl button.active {{ background: var(--card); color: var(--text); box-shadow: 0 1px 2px rgba(0,0,0,.10); }}
      .act-segctrl .act-sdot {{ width: 8px; height: 8px; border-radius: 50%; }}
      .act-tb-count {{ margin-left: auto; font-size: 12px; color: var(--text-subtle, var(--text-muted)); font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }}
      .act-pulse .act-daygroup + .act-daygroup {{ margin-top: 2px; }}
      .act-pulse .act-dayhdr {{
        font-size: 11px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase;
        color: var(--text-muted); padding: 14px 2px 8px; display: flex; align-items: center; gap: 10px;
      }}
      .act-pulse .act-dayhdr::after {{ content: ""; flex: 1; height: 1px; background: var(--border); opacity: .6; }}
      .act-pulse-rail {{ gap: 12px; }}
      /* The rail is sticky; when its content (a long injury watch + news) is
         taller than the viewport, let it scroll internally so the news card at
         the bottom stays reachable instead of being pinned off-screen. */
      @media (min-width: 901px) {{
        .act-pulse-rail {{ max-height: calc(100vh - 110px); overflow-y: auto; overscroll-behavior: contain; }}
      }}
      .act-snapshot .act-snap-body {{ display: flex; gap: 10px; padding: 4px 14px 10px; }}
      .act-snap-stat {{
        flex: 1; background: var(--card-soft, var(--bg-alt)); border: 1px solid var(--border);
        border-radius: 10px; padding: 9px 11px; display: flex; flex-direction: column;
      }}
      .act-snap-n {{ font-size: 22px; font-weight: 800; line-height: 1; color: var(--text); font-variant-numeric: tabular-nums; }}
      .act-snap-l {{ font-size: 10px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; color: var(--text-muted); margin-top: 4px; }}
      .act-snap-line {{
        display: flex; justify-content: space-between; gap: 12px; font-size: 12px;
        padding: 8px 14px; border-top: 1px solid var(--border);
      }}
      .act-snap-line span {{ color: var(--text-muted); flex: 0 0 auto; }}
      .act-snap-line strong {{ color: var(--text); font-weight: 700; text-align: right; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

      /* Compact timeline rows */
      .act-pulse .act-row {{ display: grid; grid-template-columns: auto 1fr auto; gap: 11px; align-items: center; padding: 9px 10px; border-radius: 10px; }}
      .act-pulse .act-daygroup .act-row + .act-row {{ margin-top: 6px; }}
      .act-pulse .act-row:hover {{ background: var(--row, var(--bg-alt)); }}
      .act-pulse .act-av {{ width: 30px; height: 30px; border-radius: 9px; flex: 0 0 auto; display: grid; place-items: center; overflow: hidden; font-size: 11px; font-weight: 800; }}
      .act-pulse .act-av img {{ width: 100%; height: 100%; object-fit: cover; border-radius: inherit; }}
      .act-pulse .act-av-ph {{ color: #fff; background: hsl(var(--h, 210), 42%, 46%); }}
      .act-pulse .act-rmain {{ min-width: 0; }}
      .act-pulse .act-rtop {{ display: flex; align-items: center; gap: 7px; flex-wrap: wrap; font-size: 12.5px; }}
      .act-pulse .act-tm {{ font-weight: 700; color: var(--text); }}
      .act-pulse .act-tm.team-clickable:hover {{ color: var(--accent); text-decoration: underline; }}
      .act-pulse .act-verb {{ color: var(--text-muted); font-weight: 600; }}
      .act-pulse .act-verb.trade {{ color: var(--accent); }}
      .act-pulse .act-kindtag {{ font-size: 9.5px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; padding: 2px 6px; border-radius: 5px; }}
      .act-pulse .act-kindtag.waiver {{ color: var(--accent); background: var(--accent-soft); }}
      .act-pulse .act-kindtag.trade {{ color: var(--orange, #b45309); background: color-mix(in srgb, var(--orange, #f59e0b) 15%, transparent); }}
      .act-pulse .act-chips {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px; }}
      .act-pulse .act-chip {{ display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px; font-weight: 600; padding: 3px 8px; border-radius: 7px; background: var(--card-soft, var(--bg-alt)); border: 1px solid var(--border); color: var(--text); white-space: nowrap; max-width: 100%; }}
      .act-pulse .act-chip-drop {{ opacity: .72; }}
      .act-pulse .act-sign {{ font-weight: 800; color: var(--win, #15803d); }}
      .act-pulse .act-chip-drop .act-sign {{ color: var(--loss, #b91c1c); }}
      .act-pulse .act-pos {{ font-size: 9px; font-weight: 800; letter-spacing: .03em; padding: 1px 4px; border-radius: 4px; color: #fff; }}
      .act-pulse .act-pos.qb {{ background: #3b82f6; }}
      .act-pulse .act-pos.rb {{ background: #22c55e; }}
      .act-pulse .act-pos.wr {{ background: #f59e0b; }}
      .act-pulse .act-pos.te {{ background: #8b5cf6; }}
      .act-pulse .act-pos.pk {{ background: #64748b; }}
      .act-pulse .act-arrow {{ color: var(--text-subtle, var(--text-muted)); font-weight: 700; padding: 0 1px; }}
      .act-pulse .act-rend {{ display: flex; align-items: center; gap: 8px; flex: 0 0 auto; }}
      .act-pulse .act-val {{ font-size: 12.5px; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums; }}
      .act-pulse .act-swing {{ font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 12px; color: var(--win, #15803d); background: color-mix(in srgb, var(--win, #16a34a) 14%, transparent); font-variant-numeric: tabular-nums; }}
      /* Trade rows expand to the full breakdown via native <details> */
      .act-pulse details.act-trade {{ display: block; padding: 0; }}
      .act-pulse .act-trade-sum {{ display: grid; grid-template-columns: auto 1fr auto; gap: 11px; align-items: center; padding: 9px 10px; border-radius: 10px; cursor: pointer; list-style: none; }}
      .act-pulse .act-trade-sum::-webkit-details-marker {{ display: none; }}
      .act-pulse .act-trade-sum:hover {{ background: var(--row, var(--bg-alt)); }}
      .act-pulse .act-caret {{ color: var(--text-muted); font-size: 11px; transition: transform .15s ease; }}
      .act-pulse details[open] .act-caret {{ transform: rotate(180deg); }}
      .act-pulse .act-trade-body {{ padding: 2px 10px 12px; }}
      @media (prefers-reduced-motion: reduce) {{ .act-pulse .act-caret {{ transition: none; }} }}

      /* Injury watch (rail) */
      .act-injwatch-list {{ display: flex; flex-direction: column; padding: 4px 6px 8px; }}
      .act-injrow {{ display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; padding: 8px; }}
      .act-injrow + .act-injrow {{ border-top: 1px solid var(--border); }}
      .act-injrow-l {{ min-width: 0; }}
      .act-injteam {{ font-size: 12.5px; font-weight: 700; color: var(--text); display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
      .act-injmeter {{ height: 5px; border-radius: 3px; background: var(--border); overflow: hidden; margin-top: 6px; }}
      .act-injmeter i {{ display: block; height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--orange, #f59e0b), var(--loss, #b91c1c)); }}
      .act-injrow-r {{ display: flex; align-items: center; gap: 8px; flex: 0 0 auto; }}
      .act-injdots {{ display: inline-flex; gap: 3px; flex-wrap: wrap; max-width: 72px; justify-content: flex-end; }}
      .act-injdot {{ width: 7px; height: 7px; border-radius: 50%; display: block; background: var(--text-muted); }}
      /* Canonical injury severity colors (match the player-badge-inj-* chips):
         IR/OUT red, doubtful orange, questionable amber. */
      .act-injdot.ir {{ background: var(--loss, #ef4444); }}
      .act-injdot.out {{ background: var(--loss, #ef4444); }}
      .act-injdot.dbt {{ background: var(--orange, #ea580c); }}
      .act-injdot.q {{ background: var(--inj-q, #ca8a04); }}
      .act-injcount {{ font-size: 12px; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums; min-width: 16px; text-align: right; }}
    </style>
    <div class="page-layout activity-page act-pulse" data-mtab="feed">
      <div class="act-mtabs" role="tablist" aria-label="Activity sections">
        <button type="button" class="act-mtab active" data-mtab="feed" role="tab" aria-selected="true">Activity</button>
        <button type="button" class="act-mtab" data-mtab="news" role="tab" aria-selected="false">News</button>
        <button type="button" class="act-mtab" data-mtab="league" role="tab" aria-selected="false">League</button>
      </div>
      <aside class="page-sidebar act-news-col">
        <div class="card small" id="nflNewsCard">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;">
            <h3>Fantasy News</h3>
            <span style="font-size:10px;color:var(--text-muted);font-weight:500;">via ESPN &amp; more</span>
          </div>
          <div id="nflNewsList" class="card-body" style="padding:0;">
            <div style="padding:16px 14px;display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text-muted);"><div class="loading-spinner" style="width:14px;height:14px;margin:0;flex-shrink:0;"></div>Loading…</div>
          </div>
        </div>
      </aside>

      <main class="page-main act-pulse-main">
        <div class="act-toolbar">
          <span class="act-tb-label">Show</span>
          <button class="act-tgl act-toggle active" data-kind="trade" type="button" aria-pressed="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3l4 4-4 4"/><path d="M21 7H3"/><path d="M7 21l-4-4 4-4"/><path d="M3 17h18"/></svg>
            Trades <span class="act-tgl-cnt">{trade_count}</span>
          </button>
          <button class="act-tgl act-toggle active" data-kind="waiver" type="button" aria-pressed="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>
            Waivers <span class="act-tgl-cnt">{waiver_count}</span>
          </button>

          <span class="act-tb-count">{trade_count + waiver_count} events</span>
        </div>
        {activity_html}
      </main>

      <aside class="page-sidebar act-pulse-rail">
        {snapshot_html}
        <div class="act-injfilter">
          <span class="act-tb-label">Injuries</span>
          <div class="act-segctrl" role="tablist">
            <button class="inj-toggle active" data-status="all" type="button" role="tab" aria-selected="true">All</button>
            <button class="inj-toggle" data-status="IR" type="button" role="tab" aria-selected="false"><span class="act-sdot" style="background:var(--loss,#b91c1c)"></span>IR</button>
            <button class="inj-toggle" data-status="OUT" type="button" role="tab" aria-selected="false"><span class="act-sdot" style="background:var(--orange,#f59e0b)"></span>Out</button>
            <button class="inj-toggle" data-status="QUESTIONABLE" type="button" role="tab" aria-selected="false"><span class="act-sdot" style="background:var(--inj-q, #ca8a04)"></span>Q</button>
          </div>
        </div>
        {injury_html}
      </aside>
    </div>

    <script>
    (function() {{
      function loadNflNews() {{
        var list = document.getElementById('nflNewsList');
        if (!list) return;
        fetch('/api/nfl-news?limit=12')
          .then(function(r) {{ return r.json(); }})
          .then(function(data) {{
            var items = data.news || [];
            if (!items.length) {{
              list.innerHTML = '<div style="padding:12px 14px;font-size:13px;color:var(--text-muted);">No news available.</div>';
              return;
            }}
            list.innerHTML = items.map(function(n) {{
              var linkOpen = n.url ? '<a href="' + n.url + '" target="_blank" rel="noopener" class="act-news-link">' : '<span>';
              var linkClose = n.url ? '</a>' : '</span>';
              return '<div class="act-news-item">' +
                '<div class="act-news-headline">' + linkOpen + n.headline + linkClose + '</div>' +
                (n.description ? '<div class="act-news-desc">' + n.description + '</div>' : '') +
                '<div class="act-news-meta">' + [n.source, n.age].filter(Boolean).join(' · ') + '</div>' +
              '</div>';
            }}).join('');
          }})
          .catch(function() {{ /* fail silently */ }});
      }}

      if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', loadNflNews);
      }} else {{
        loadNflNews();
      }}
    }})();
    </script>

    <style>
      .act-news-item {{
        padding: 10px 14px;
        border-bottom: 1px solid var(--border);
      }}
      .act-news-item:last-child {{ border-bottom: none; }}
      .act-news-headline {{ font-size: 12px; font-weight: 600; color: var(--text); line-height: 1.4; margin-bottom: 3px; }}
      .act-news-link {{ color: var(--text); text-decoration: none; }}
      .act-news-link:hover {{ text-decoration: underline; color: #3b82f6; }}
      .act-news-desc {{ font-size: 11px; color: var(--text-muted); line-height: 1.35; margin-bottom: 3px; }}
      .act-news-meta {{ font-size: 10px; color: var(--text-muted); opacity: .7; }}

      .bract-summary-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 0 0 12px 0;
      }}

      .bract-summary-card {{
        border: 1px solid var(--border);
        background: var(--card-soft);
        border-radius: 12px;
        padding: 12px 14px;
      }}

      .bract-summary-label {{
        font-size: 11px;
        line-height: 1.2;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 6px;
        font-weight: 700;
      }}

      .bract-summary-value {{
        font-size: 24px;
        line-height: 1.1;
        font-weight: 800;
        color: var(--text);
      }}

      .bract-summary-text {{
        font-size: 12px;
        line-height: 1.3;
      }}

      .bract-spotlight {{
        border: 1px solid var(--border);
        background: var(--accent-soft);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 14px;
      }}

      .bract-spotlight-title {{
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 4px;
      }}

      .bract-spotlight-copy {{
        font-size: 14px;
        color: var(--text);
      }}

      .bract-total-row {{
        padding-bottom: 2px;
      }}

      .bract-total-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        font-size: 14px;
        font-weight: 700;
        color: var(--text);
      }}

      .bract-net-pos {{
        color: #15803d;
      }}

      .bract-net-neg {{
        color: #b91c1c;
      }}

      .bract-net-even {{
        color: #475569;
      }}

      .bract-verdict {{
        display: inline-flex;
        align-items: center;
        margin-top: 8px;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
      }}

      .bract-verdict-win {{
        background: #dcfce7;
        color: #166534;
      }}

      .bract-verdict-loss {{
        background: #fee2e2;
        color: #991b1b;
      }}

      .bract-verdict-even {{
        background: #e2e8f0;
        color: #334155;
      }}

      @media (max-width: 900px) {{
        /* 2x2 instead of four tall stacked cards — far less vertical space. */
        .bract-summary-grid {{
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }}
      }}
    </style>

    <script>
    (function() {{
      // Mobile tab switcher: flip the container's data-mtab so the CSS reveals
      // the chosen section (Activity / News / League). No-op on desktop, where
      // the tab bar is hidden and all three columns show at once.
      document.querySelectorAll('.act-mtab').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          var tab = btn.getAttribute('data-mtab');
          var container = document.querySelector('.act-pulse');
          if (container) container.setAttribute('data-mtab', tab);
          document.querySelectorAll('.act-mtab').forEach(function(b) {{
            var on = b === btn;
            b.classList.toggle('active', on);
            b.setAttribute('aria-selected', on ? 'true' : 'false');
          }});
          window.scrollTo(0, 0);
        }});
      }});

      document.querySelectorAll('.act-toggle').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          this.classList.toggle('active');
          const activeKinds = Array.from(document.querySelectorAll('.act-toggle.active'))
            .map(b => b.getAttribute('data-kind'));

          document.querySelectorAll('.activity-item').forEach(function(item) {{
            const k = item.getAttribute('data-kind');
            item.style.display = activeKinds.length === 0 || activeKinds.includes(k)
              ? ''
              : 'none';
          }});
        }});
      }});

      document.querySelectorAll('.inj-toggle').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          document.querySelectorAll('.inj-toggle').forEach(b => b.classList.remove('active'));
          this.classList.add('active');

          const status = this.getAttribute('data-status');
          const rows = document.querySelectorAll('.inj-row');

          rows.forEach(function(row) {{
            if (status === 'all') {{
              row.style.display = '';
              return;
            }}
            const chips = row.querySelectorAll('.chip');
            let matched = false;
            chips.forEach(function(c) {{
              if (c.textContent.trim().toUpperCase() === status) {{
                matched = true;
              }}
            }});
            row.style.display = matched ? '' : 'none';
          }});
        }});
      }});
    }})();
    </script>
    """

