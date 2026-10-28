"""Teams-parity window posture for My Leagues cards + board headers.

Kaedon's 2026-10-26 directive: the league-card tag must be the Teams
page tag for the same roster. Dynasty cards take the win-window
classifier's label (via calculate_roster_grade with league context, the
exact call the Teams page makes); redraft leagues take the Teams redraft
labels (Contend / Bubble / Long Shot) and can never be tagged Rebuild.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from dashboard_services import portfolio_board as pb

NEW_LABELS = {
    "Powerhouse", "Contender", "Win-Now", "Contender Window", "Rising",
    "Crossroads", "Treadmill", "Retooling", "Rebuilding", "Full Rebuild",
}
REBUILD_FAMILY = {"Rebuild", "Rebuilding", "Full Rebuild", "Retooling",
                  "Fringe"}
REDRAFT_LABELS = {"Contend", "Bubble", "Long Shot"}

_POS_CYCLE = ["QB", "RB", "WR", "TE", "RB", "WR", "WR", "RB"]

# (dynasty value, redraft value, age) x8 per team, strongest first.
_TEAM_SPECS = {
    "1": [(900, 880, 25), (850, 860, 24), (800, 820, 25), (750, 700, 26),
          (700, 680, 24), (650, 640, 25), (600, 590, 26), (550, 540, 25)],
    "2": [(650, 640, 27), (600, 610, 27), (580, 570, 28), (500, 480, 27),
          (450, 460, 26), (420, 430, 27), (380, 370, 28), (340, 350, 27)],
    "3": [(450, 470, 29), (400, 430, 28), (380, 400, 29), (320, 340, 30),
          (300, 310, 28), (280, 290, 29), (240, 250, 30), (200, 210, 29)],
    "4": [(300, 320, 30), (260, 280, 31), (240, 260, 30), (200, 220, 31),
          (180, 190, 30), (160, 170, 31), (140, 150, 30), (120, 130, 31)],
}


def _build_league(scoring):
    """Fake ctx + rosters + value lookup in the real adapter shapes.

    ``values`` mirrors league_format_value_lookup output (model row with
    ``value`` rewritten for the league type, raw redraft columns kept);
    ages ride on the model table's search_name rows like production.
    """
    rosters, values, model_rows = [], {}, []
    for rid, specs in _TEAM_SPECS.items():
        pids = []
        for i, (dyn, red, age) in enumerate(specs):
            pid = f"t{rid}p{i}"
            name = f"Team{rid} Player{i}"
            pids.append(pid)
            values[pid] = {
                "position": _POS_CYCLE[i],
                "value": red if scoring == "redraft" else dyn,
                "redraft_value_1qb": red,
                "name": name,
                "search_name": name.lower(),
            }
            model_rows.append({
                "id": pid, "name": name, "search_name": name.lower(),
                "position": _POS_CYCLE[i], "age": age,
            })
        rosters.append({
            "roster_id": int(rid), "owner_id": f"u{rid}",
            "players": pids, "settings": {"wins": 5, "losses": 2},
        })
    picks = {rid: [] for rid in _TEAM_SPECS}
    picks["4"] = [{"id": "2027_1_01", "display": "2027 1st",
                   "season": 2027, "round": 1}]
    ctx = {
        "platform": "espn" if scoring == "redraft" else "sleeper",
        "league_settings": {"type": 0 if scoring == "redraft" else 2},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX"],
        "total_rosters": 4,
        "picks_by_roster": picks,
        "model_value_table": model_rows,
        "players_index": {},
        "league": {"name": "Fake League", "season": 2026},
        "league_id": "L1",
        "season": 2026,
    }
    return ctx, rosters, values


def _teams_page_grade(ctx, rosters, values, vid):
    """The Teams page's own grade for one roster, rebuilt from the shared
    scaffolding teams_page uses (value axes, percentiles, capital context,
    positional ranks) -- the parity reference."""
    from app import count_roster_positions
    from dashboard_services.ai.context_builders import (
        _team_value_axes, calculate_roster_grade, league_capital_context,
    )
    from utils.roster_strength import (
        rank_rosters_by_position, roster_pos_value_lists,
    )

    totals_d, totals_r, ratios = {}, {}, {}
    for r in rosters:
        pairs = []
        for pid in r.get("players") or []:
            row = values.get(str(pid)) or {}
            if str(row.get("position") or "").upper() in ("QB", "RB", "WR", "TE"):
                pairs.append((float(row.get("value") or 0),
                              float(row.get("redraft_value_1qb") or 0)))
        d, rr, dr = _team_value_axes(pairs)
        totals_d[str(r["roster_id"])] = d
        totals_r[str(r["roster_id"])] = rr
        ratios[str(r["roster_id"])] = dr

    def _pct(totals, rid):
        ordered = sorted(totals.values())
        return sum(1 for v in ordered if v < totals[rid]) / max(len(ordered) - 1, 1)

    slots = count_roster_positions(ctx.get("roster_positions") or [])
    pos_values = roster_pos_value_lists(
        rosters, values, players_index={}, rid_cast=str)
    _s, ranks = rank_rosters_by_position(pos_values, slots)
    p_ranks = {pos: (ranks.get(pos) or {}).get(vid)
               for pos in ("QB", "RB", "WR", "TE")}
    capital_pct, surplus = league_capital_context(ctx["picks_by_roster"])
    roster = next(r for r in rosters if str(r["roster_id"]) == vid)
    age_by_name = {
        str(r.get("search_name") or "").lower(): float(r["age"])
        for r in ctx["model_value_table"] if r.get("age") is not None
    }
    flat = []
    for pid in roster.get("players") or []:
        row = values.get(str(pid)) or {}
        flat.append({
            "position": str(row.get("position") or "").upper(),
            "value": float(row.get("value") or 0),
            "age": age_by_name.get(str(row.get("name") or "").lower()),
        })
    flat.sort(key=lambda p: p["value"], reverse=True)
    return calculate_roster_grade(
        flat, ctx["picks_by_roster"].get(vid, []),
        position_ranks=p_ranks, num_teams=4,
        dynasty_pct_val=_pct(totals_d, vid),
        redraft_pct_val=_pct(totals_r, vid),
        dr_ratio=ratios[vid],
        capital_pct_val=capital_pct.get(vid),
        surplus_firsts_val=surplus.get(vid),
        scoring_type="dynasty",
    )


# ── (a) dynasty parity ──────────────────────────────────────────────────────

def test_dynasty_card_label_equals_teams_classifier():
    ctx, rosters, values = _build_league("dynasty")
    win = pb.window_posture_for(ctx, rosters, values, "1")
    assert win is not None
    expected = _teams_page_grade(ctx, rosters, values, "1")
    assert win["label"] == expected["win_window"]
    assert win["label"] in NEW_LABELS
    assert win["label"] == "Powerhouse"
    assert win["shape"] == expected["roster_shape"]
    assert win["cls"] == "win"
    assert win["color"] == pb.WINDOW_COLORS[expected["win_window"]]


def test_dynasty_card_label_equals_teams_classifier_for_weak_roster():
    ctx, rosters, values = _build_league("dynasty")
    win = pb.window_posture_for(ctx, rosters, values, "3")
    expected = _teams_page_grade(ctx, rosters, values, "3")
    assert win is not None
    assert win["label"] == expected["win_window"]
    assert win["label"] in NEW_LABELS


def test_window_posture_none_without_values():
    ctx, rosters, _values = _build_league("dynasty")
    assert pb.window_posture_for(ctx, rosters, {}, "1") is None
    assert pb.window_posture_for(ctx, rosters, None, "1") is None
    assert pb.window_posture_for(ctx, [], {"x": {}}, "1") is None


# ── (b) redraft can never be Rebuild ────────────────────────────────────────

def test_redraft_weakest_roster_is_long_shot_not_rebuild():
    ctx, rosters, values = _build_league("redraft")
    win = pb.window_posture_for(ctx, rosters, values, "4", odds_pct=12.0)
    assert win is not None
    assert win["label"] == "Long Shot"
    assert win["cls"] == "longshot"
    # No odds either: the redraft-value percentile fallback is still a
    # redraft label, never a rebuild label.
    win2 = pb.window_posture_for(ctx, rosters, values, "4", odds_pct=None)
    assert win2["label"] == "Long Shot"


def test_redraft_never_rebuild_at_any_odds_or_record():
    ctx, rosters, values = _build_league("redraft")
    for odds in [0, 5, 20, 34.9, 35, 55, 69.9, 70, 90, 100, None]:
        for rid in ("1", "2", "3", "4"):
            win = pb.window_posture_for(ctx, rosters, values, rid, odds_pct=odds)
            assert win is not None
            assert win["label"] in REDRAFT_LABELS
            assert win["label"] not in REBUILD_FAMILY
    # Odds-first parity with the Teams page: even the weakest roster is
    # Contend at 85% playoff odds.
    win = pb.window_posture_for(ctx, rosters, values, "4", odds_pct=85.0)
    assert win["label"] == "Contend"
    win = pb.window_posture_for(ctx, rosters, values, "4", odds_pct=50.0)
    assert win["label"] == "Bubble"


def test_keeper_league_uses_redraft_labels():
    # Sleeper type 1 (keeper) classifies as redraft everywhere in the
    # codebase, including the Teams page, so its card does too.
    ctx, rosters, values = _build_league("redraft")
    ctx["platform"] = "sleeper"
    ctx["league_settings"] = {"type": 1}
    win = pb.window_posture_for(ctx, rosters, values, "4", odds_pct=5.0)
    assert win["label"] == "Long Shot"


# ── (c) summary (cards) and board headers agree ─────────────────────────────

def test_summary_posture_matches_board_style_call(monkeypatch):
    import dashboard_services.accounts as accounts
    import dashboard_services.portfolio_summary as summaries

    ctx, rosters, values = _build_league("dynasty")
    ctx.update({
        "rosters": rosters,
        "users": [{"user_id": "u1", "display_name": "Viewer",
                   "metadata": {"team_name": "Viewer Team"}}],
        "offseason_mode": False,
    })
    rosters[0]["owner_id"] = "u1"
    monkeypatch.setattr(accounts, "resolve_account_viewer_for_league",
                        lambda *a, **k: {"viewer_roster_id": "1"})
    monkeypatch.setattr(summaries, "_store_persistent", lambda *a: None)
    monkeypatch.setattr(
        "dashboard_services.ai.context_builders.portfolio_record_and_rank",
        lambda *a: (5, 2, 0, 500.0, 2))
    monkeypatch.setattr(
        "dashboard_services.ai.context_builders.league_format_value_lookup",
        lambda *a, **k: values)

    result = summaries.build_league_summary(
        7, {"platform": "sleeper", "league_id": "L1", "season": 2026},
        lambda *a: ctx,
    )
    assert result["posture"] in NEW_LABELS
    # Board headers call the same helper with their own lookup instance
    # and the same cache-only odds: the label must be identical.
    board_side = pb.window_posture_for(
        ctx, rosters, dict(values), "1",
        odds_pct=result.get("playoff_odds_pct"))
    assert board_side is not None
    assert result["posture"] == board_side["label"]
    assert result["posture_cls"] == board_side["cls"]
    assert result["posture_color"] == board_side["color"]
    assert result["posture_shape"] == board_side["shape"]


def test_all_posture_call_sites_use_window_helper():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    summary_src = (root / "dashboard_services" / "portfolio_summary.py").read_text()
    assert "window_posture_for" in summary_src
    routes_src = (root / "routes" / "user_pages_bp.py").read_text()
    inline = routes_src.split("def _league_summary")[1].split(
        "# Stale-while-revalidate")[0]
    assert "window_posture_for" in inline
    board = routes_src.split("scan_skipped_no_roster += 1")[1].split(
        '"meta": metadata')[0]
    assert "window_posture_for" in board
    assert "is_redraft=" in board  # degraded fallback stays redraft-safe


# ── Card render ─────────────────────────────────────────────────────────────

def _render_card(**kw):
    from app import app, build_portfolio_body

    lg = {
        "league_id": "1", "name": "Alpha", "platform": "sleeper",
        "season": 2026, "wins": 5, "losses": 2, "ties": 0, "record": "5-2",
        "rank": 2, "total_teams": 10, "pf": 812.4, "total_value": 4200.0,
        "all_players": {}, "streak": ["W", "W", "L"], "urgency": 3,
        "pos_user_vals": {}, "pos_league_avgs": {}, "pos_user_pctile": {},
        "pos_user_rank": {"QB": 4, "RB": 9, "WR": 2, "TE": 6},
        "offseason": False, "team_name": "Team Alpha", "is_favorite": False,
    }
    lg.update(kw)
    with app.test_request_context("/"):
        return build_portfolio_body(
            "kaedon", [lg], [lg], 2026, num_leagues=1, holdings=[],
            nfl_exposure=[], cross_pos={}, total_wins=5, total_losses=2,
            total_ties=0, current_week=8,
        )


def test_card_renders_window_chip_and_shape():
    html = _render_card(posture="Contender Window", posture_cls="win",
                        posture_color="#3b82f6", posture_shape="DEEP")
    assert "pf-posture win" in html
    assert "--wc:#3b82f6" in html
    assert ">Contender Window</span>" in html
    assert "data-posture-shape" in html
    assert ">DEEP</span>" in html
    # Window chip CSS ships with the page.
    assert ".pf-posture.win{" in html
    assert ".pf-shape{" in html


def test_card_renders_redraft_long_shot_never_rebuild():
    html = _render_card(posture="Long Shot", posture_cls="longshot",
                        posture_color="#94a3b8", posture_shape="")
    assert "pf-posture longshot" in html
    assert ">Long Shot</span>" in html
    assert "pf-posture rebuild" not in html


def test_js_hydration_applies_window_color_and_shape():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    seg = src.split("var postureEl = card.querySelector('[data-posture]');")[1]
    seg = seg.split("var stamp = data.last_successful_sync_at")[0]
    assert "data.posture_color" in seg
    assert "setProperty('--wc'" in seg
    assert "data.posture_shape" in seg
    assert "data-posture-shape" in seg


def test_changelog_entry_for_posture_parity():
    from dashboard_services.changelog import CHANGELOG

    top = CHANGELOG[24]  # shifted +1 by the Weekly Hub dedupe refresh prepend, +1 by the duplicate Tuesday dot removal, +1 by the phone tabs + card odds removal, +1 by the phone notifications panel + More-dot fix, +1 by the recap glance username fix, +1 by the Weekly Hub revamp, +1 by the graphs overhaul, +1 by the holdings sort + insights speed prepend, +1 by the rooting/bye cleanup, and +1 earlier by the Front Office scan fix
    assert top["date"] == "2026-10-26"
    assert top["link"] == "/portfolio"
    assert "Teams page" in top["text"]
    assert "\u2014" not in top["text"]  # no em dashes in UI copy
