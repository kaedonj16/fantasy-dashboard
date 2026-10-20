"""My Leagues pending-card viewer matching guards."""

import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_uses_account_viewer_resolution():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    fn = source.split("def _league_summary")[1].split("\n    leagues_data")[0]
    assert "resolve_account_viewer_for_league" in fn
    assert "match_viewer_roster" in fn
    assert "sleeper_owner_id_for_account" in fn
    assert "owner_id=sleeper_owner" in fn
    assert 'str(r.get("owner_id")) == str(viewer_user_id)' not in fn
    assert "team_label_from_user" in fn
    assert '"team_name": team_name' in fn


def test_sleeper_owner_id_ignores_unlinked_session_viewer(monkeypatch):
    pytest.importorskip("flask")
    from routes.user_pages_bp import sleeper_owner_id_for_account
    import dashboard_services.accounts as accounts

    monkeypatch.setattr(accounts, "list_account_platform_ids", lambda *a, **k: ["linked-sleeper"])
    assert sleeper_owner_id_for_account(42, "linked-sleeper", "sleeper") == "linked-sleeper"
    assert sleeper_owner_id_for_account(42, "1020439", "sleeper") is None
    assert sleeper_owner_id_for_account(42, "1020439", "fleaflicker") is None
    assert sleeper_owner_id_for_account(None, "sleeper-only", "sleeper") == "sleeper-only"


def test_portfolio_undrafted_leagues_use_startup_draft_phase():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    summary = source.split("def _league_summary")[1].split("\n    leagues_data")[0]
    assert "startup_draft_phase" in summary
    assert "draft_start_ms" in summary
    assert '"draft_phase": draft_phase' in summary
    assert 'draft_phase != "drafted"' in summary
    # Thin pre-draft shells must not fall through to positional ranks.
    assert "pos_user_rank" in summary
    phase_gate = summary.split('draft_phase != "drafted"')[0]
    ranks_after = summary.split('draft_phase != "drafted"')[1]
    assert "pos_user_rank" in ranks_after
    assert "pos_user_rank" not in phase_gate


def test_portfolio_positional_strength_uses_in_league_percentiles():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    summary = source.split("def _league_summary")[1].split("\n    leagues_data")[0]
    assert "rank_rosters_by_position" in summary
    assert "league_format_value_lookup" in summary
    assert "from utils.roster_strength import" in summary
    assert "strength_percentile" in summary
    assert "pos_user_pctile[pos] = strength_percentile(" in summary
    assert '"pos_user_pctile": pos_user_pctile' in summary

    blend = source.split("Cross-league positional strength")[1].split("valid_leagues.sort")[0]
    assert "average_league_percentiles" in blend
    assert "pos_user_pctile" in blend
    # Ratio-vs-median blend is what made stacked leagues read negative.
    assert "u / a" not in blend


def test_portfolio_actions_api_wired():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    assert '@user_pages_bp.route("/api/portfolio-actions")' in source
    assert "rank_cross_league_actions" in source
    assert "lineup_actions_from_issues" in source
    assert "select_waiver_add" in source
    assert "already_on_ir=" in source
    assert 'viewer_roster.get("reserve")' in source
    assert "_portfolio_viewer_has_pro" in source
    assert '"paywall": True' in source


def test_portfolio_body_moves_card_pro_gated():
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    assert "showPaywall" in fn
    assert "__brctx.isPremium" in fn or "isPremium" in fn


def test_portfolio_body_includes_moves_card():
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    assert "pfMovesCard" in fn
    assert "/api/portfolio-actions" in fn
    assert "moves_card" in fn
    assert "top_strip + moves_card + league_card" in fn
    assert "pf-move-row" in fn
    assert "pf-moves-list" in fn
    # The digest is lazy-loaded after page load (idle), not fetched inline, so it
    # never competes with the initial render.
    assert "__pfMovesGo" in fn
    assert "requestIdleCallback" in fn
    assert "addEventListener('load'" in fn


def test_portfolio_empty_digest_shows_all_caught_up_not_hidden():
    """A PRO user whose leagues have nothing to do must still see the card with
    an all-caught-up note, not have it vanish (which reads as 'missing')."""
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    # The old behavior hid the card on an empty digest; that must be gone.
    assert "if(!acts.length){card.hidden=true;return;}" not in fn
    # New empty state renders inside a visible card.
    assert "pf-moves-empty" in fn
    assert "all caught up" in fn.lower()
    assert ".pf-moves-empty{" in source  # style is defined


def test_changelog_announces_portfolio_moves_empty_state():
    from dashboard_services.changelog import CHANGELOG

    entry = next(
        e for e in CHANGELOG
        if "this week's moves" in e.get("text", "").lower()
        and "empty" in e.get("text", "").lower()
    )
    assert entry["tag"] == "fix"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]


def test_portfolio_record_and_rank_accepts_seed_int_standings_map():
    from dashboard_services.ai.context_builders import portfolio_record_and_rank

    lctx = {
        "standings_map": {1: 3, 2: 1},
        "roster_map": {"1": "Team A", "2": "Team B"},
        "rosters": [
            {
                "roster_id": 1,
                "settings": {
                    "wins": 2, "losses": 1, "ties": 0,
                    "fpts": 120, "fpts_decimal": 50,
                },
            },
            {"roster_id": 2, "settings": {"wins": 3, "losses": 0, "fpts": 140}},
        ],
    }
    wins, losses, ties, pf, rank = portfolio_record_and_rank(lctx, "1", lctx["rosters"][0])
    assert wins == 2 and losses == 1 and ties == 0
    assert pf == pytest.approx(120.5)
    assert rank == 3


def test_portfolio_record_and_rank_accepts_dict_standings_map():
    from dashboard_services.ai.context_builders import portfolio_record_and_rank

    lctx = {
        "standings_map": {
            "1": {"wins": 5, "losses": 2, "ties": 0, "pf": 800.0},
            "2": {"wins": 4, "losses": 3, "ties": 0, "pf": 750.0},
        },
        "rosters": [
            {"roster_id": 1, "settings": {}},
            {"roster_id": 2, "settings": {}},
        ],
    }
    wins, losses, ties, pf, rank = portfolio_record_and_rank(lctx, "1", lctx["rosters"][0])
    assert wins == 5 and losses == 2
    assert pf == pytest.approx(800.0)
    assert rank == 1


def test_portfolio_record_and_rank_falls_back_to_roster_settings():
    from dashboard_services.ai.context_builders import portfolio_record_and_rank

    lctx = {
        "standings_map": {},
        "rosters": [
            {"roster_id": 1, "settings": {"wins": 2, "losses": 4, "fpts": 90, "fpts_decimal": 0}},
            {"roster_id": 2, "settings": {"wins": 5, "losses": 1, "fpts": 140, "fpts_decimal": 0}},
            {"roster_id": 3, "settings": {"wins": 5, "losses": 1, "fpts": 150, "fpts_decimal": 0}},
        ],
    }
    wins, losses, ties, pf, rank = portfolio_record_and_rank(lctx, "1", lctx["rosters"][0])
    assert wins == 2 and losses == 4
    assert rank == 3


# ── /api/portfolio-actions: honest scan status + swap parity ────────────────
#
# The moves card used to paint "You're all caught up" for any empty actions
# list, including scans that checked nothing (cold league caches). These
# tests drive the real endpoint with stubbed league contexts.

_PF_PLAYERS = {
    "s1": {"full_name": "Starter Guy", "team": "BUF", "position": "RB",
           "injury_status": ""},
    "b1": {"full_name": "Bench Guy", "team": "MIA", "position": "RB",
           "injury_status": ""},
}
_PF_LEAGUES = [
    {"league_id": "L1", "platform": "sleeper", "season": 2026,
     "name": "Test League"},
]
_FUTURE = 9999999999


def _pf_ctx(projections=None, owner_id="owner-1"):
    ctx = {
        "rosters": [{
            "roster_id": 1, "owner_id": owner_id, "starters": ["s1"],
            "players": ["s1", "b1"], "reserve": [], "taxi": [],
        }],
        "league": {"name": "Test League", "settings": {}},
        "users": [],
        "roster_positions": ["RB", "BN"],
        "players_index": {
            "s1": {"position": "RB", "name": "Starter Guy", "team": "BUF"},
            "b1": {"position": "RB", "name": "Bench Guy", "team": "MIA"},
        },
        "model_value_table": [],
    }
    if projections is not None:
        ctx["proj_by_week"] = {7: {"projections": projections}}
    return ctx


def _pf_client(monkeypatch, ctx_by_lid, leagues=None, *, locked_teams=(),
               fallback_proj=None):
    pytest.importorskip("flask")
    from flask import Flask

    import dashboard_services.accounts
    import dashboard_services.api
    import dashboard_services.subscriptions
    import routes.user_pages_bp as bp_mod
    import utils.utils as uu
    accounts = dashboard_services.accounts
    ds_api = dashboard_services.api
    subs = dashboard_services.subscriptions

    games = []
    for home, away in (("BUF", "NYJ"), ("MIA", "NE")):
        games.append({
            "home": home, "away": away,
            "gameTime_epoch": 1 if home in locked_teams else _FUTURE,
        })
    monkeypatch.setattr(
        bp_mod, "get_nfl_state",
        lambda *a, **k: {"season": 2026, "week": 7, "season_type": "reg"})
    monkeypatch.setattr(
        bp_mod, "get_league_ctx_from_cache",
        lambda plat, lid, season, allow_build=False: ctx_by_lid.get(lid, {}))
    monkeypatch.setattr(
        bp_mod, "get_model_value_table_cached", lambda *a, **k: [])
    monkeypatch.setattr(
        bp_mod, "build_projections_by_week",
        lambda *a, **k: fallback_proj or {})
    monkeypatch.setattr(
        accounts, "resolve_my_leagues",
        lambda uid, aid, season: (leagues or _PF_LEAGUES, 2026))
    monkeypatch.setattr(subs, "has_premium_for_viewer", lambda *a, **k: True)
    monkeypatch.setattr(ds_api, "get_nfl_players", lambda *a, **k: dict(_PF_PLAYERS))
    monkeypatch.setattr(uu, "load_week_schedule", lambda season, week: games)

    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(bp_mod.user_pages_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "owner-1"
    return client


def _pf_get(client):
    resp = client.get("/api/portfolio-actions")
    assert resp.status_code == 200
    return resp.get_json()


def test_portfolio_actions_cold_cache_reports_incomplete_scan(monkeypatch):
    """Every league cold: empty actions must carry scan.incomplete, not a
    bare empty the client would paint as all-caught-up."""
    data = _pf_get(_pf_client(monkeypatch, {}))
    assert data["actions"] == []
    scan = data["scan"]
    assert scan["total"] == 1
    assert scan["scanned"] == 0
    assert scan["skipped_cold"] == scan["total"]
    assert scan["incomplete"] is True


def test_portfolio_actions_complete_scan_reports_incomplete_false(monkeypatch):
    ctx = _pf_ctx(projections={"s1": 15.0, "b1": 9.0})
    data = _pf_get(_pf_client(monkeypatch, {"L1": ctx}))
    assert data["actions"] == []
    assert data["scan"] == {
        "total": 1, "scanned": 1, "skipped_cold": 0,
        "skipped_no_roster": 0, "incomplete": False,
    }


def test_portfolio_actions_unmatched_roster_counted_skipped(monkeypatch):
    ctx = _pf_ctx(projections={"s1": 9.0, "b1": 15.0}, owner_id="someone-else")
    data = _pf_get(_pf_client(monkeypatch, {"L1": ctx}))
    assert data["actions"] == []
    assert data["scan"]["skipped_no_roster"] == 1
    assert data["scan"]["scanned"] == 0
    assert data["scan"]["incomplete"] is True


def test_portfolio_actions_partial_scan_keeps_actions_and_flags_incomplete(monkeypatch):
    leagues = _PF_LEAGUES + [
        {"league_id": "L2", "platform": "sleeper", "season": 2026,
         "name": "Cold League"},
    ]
    ctx = _pf_ctx(projections={"s1": 9.0, "b1": 15.0})
    data = _pf_get(_pf_client(monkeypatch, {"L1": ctx}, leagues))
    assert data["scan"]["total"] == 2
    assert data["scan"]["scanned"] == 1
    assert data["scan"]["skipped_cold"] == 1
    assert data["scan"]["incomplete"] is True
    assert any(a["title"] == "Start Bench Guy over Starter Guy"
               for a in data["actions"])


def test_portfolio_actions_emits_bench_swap_alert(monkeypatch):
    """Swap parity: a bench player out-projecting a starter by >= 2 produces
    the same Start-X-over-Y action push / Next Steps generate."""
    ctx = _pf_ctx(projections={"s1": 9.0, "b1": 15.0})
    data = _pf_get(_pf_client(monkeypatch, {"L1": ctx}))
    swaps = [a for a in data["actions"]
             if a["title"] == "Start Bench Guy over Starter Guy"]
    assert len(swaps) == 1
    swap = swaps[0]
    assert swap["kind"] == "lineup"
    assert swap["detail"] == "Bench Guy projects 15.0 vs Starter Guy's 9.0"
    assert swap["dismiss_key"] == "swap:s1>b1"
    assert swap["href"].endswith("/waivers?tab=startsit")


def test_portfolio_actions_swap_from_shared_fallback_projections(monkeypatch):
    """No proj_by_week on the ctx: the shared build_projections_by_week
    fallback (the push path's source) still yields the swap."""
    ctx = _pf_ctx(projections=None)
    fallback = {7: {"projections": {"s1": 9.0, "b1": 15.0}}}
    data = _pf_get(_pf_client(monkeypatch, {"L1": ctx},
                              fallback_proj=fallback))
    assert any(a["title"] == "Start Bench Guy over Starter Guy"
               for a in data["actions"])


def test_portfolio_actions_swap_skipped_when_starter_game_locked(monkeypatch):
    """A starter whose game already kicked off cannot be benched: no swap."""
    ctx = _pf_ctx(projections={"s1": 9.0, "b1": 15.0})
    data = _pf_get(_pf_client(monkeypatch, {"L1": ctx}, locked_teams=("BUF",)))
    assert data["actions"] == []
    assert data["scan"]["incomplete"] is False


def test_portfolio_actions_no_projections_no_swap_no_crash(monkeypatch):
    ctx = _pf_ctx(projections=None)
    data = _pf_get(_pf_client(monkeypatch, {"L1": ctx}))
    assert data["actions"] == []
    assert data["scan"]["scanned"] == 1
    assert data["scan"]["incomplete"] is False


def _pf_leagues(n):
    return [
        {"league_id": f"L{i}", "platform": "sleeper", "season": 2026,
         "name": f"League {i}"}
        for i in range(1, n + 1)
    ]


def test_portfolio_actions_scan_total_covers_all_linked_leagues(monkeypatch):
    """11 linked leagues, only 4 cached: total is the full 11, not the
    attempted count, so the client's missed count (total - scanned)
    includes the 7 cold leagues."""
    leagues = _pf_leagues(11)
    ctx = _pf_ctx(projections={"s1": 15.0, "b1": 9.0})
    cached = {f"L{i}": ctx for i in range(1, 5)}
    data = _pf_get(_pf_client(monkeypatch, cached, leagues))
    scan = data["scan"]
    assert scan["total"] == 11
    assert scan["scanned"] == 4
    assert scan["skipped_cold"] == 7
    assert scan["skipped_no_roster"] == 0
    # Buckets sum to the ATTEMPTED count (here all 11: the slice is 12).
    assert scan["scanned"] + scan["skipped_cold"] + scan["skipped_no_roster"] == 11
    assert scan["incomplete"] is True


def test_portfolio_actions_all_eleven_leagues_scanned(monkeypatch):
    """All 11 linked leagues are attempted (the slice is 12, not 8)."""
    leagues = _pf_leagues(11)
    ctx = _pf_ctx(projections={"s1": 15.0, "b1": 9.0})
    cached = {f"L{i}": ctx for i in range(1, 12)}
    data = _pf_get(_pf_client(monkeypatch, cached, leagues))
    assert data["scan"]["total"] == 11
    assert data["scan"]["scanned"] == 11
    assert data["scan"]["incomplete"] is False


def test_portfolio_actions_leagues_past_slice_count_toward_total(monkeypatch):
    """13 linked, attempt slice 12: the 13th is never attempted but still
    counts toward total, so the scan stays honestly incomplete."""
    leagues = _pf_leagues(13)
    ctx = _pf_ctx(projections={"s1": 15.0, "b1": 9.0})
    cached = {f"L{i}": ctx for i in range(1, 14)}
    data = _pf_get(_pf_client(monkeypatch, cached, leagues))
    scan = data["scan"]
    assert scan["total"] == 13
    assert scan["scanned"] == 12
    assert scan["skipped_cold"] == 0
    assert scan["incomplete"] is True


def test_portfolio_moves_card_retries_on_ladder_and_shows_incomplete_state():
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    # Bounded retry ladder while the scan reports incomplete: cold league
    # builds take far longer than one short retry. (No "},5000);" literal:
    # test_portfolio_leagues pins it out of this function to keep the old
    # 5s poll-and-reload pattern from returning.)
    assert "__pfMovesGo(attempt)" in fn
    assert "scan.incomplete" in fn
    assert "var PF_RETRY=[6000,15000,30000];" in fn
    assert "attempt<PF_RETRY.length" in fn
    assert "__pfMovesGo(attempt+1)" in fn
    assert "PF_RETRY[attempt]" in fn
    # The single-retry shape from #2739 is gone.
    assert "__pfMovesGo(1)" not in fn
    # Honest incomplete states: under the list and instead of all-caught-up.
    assert "Could not check" in fn
    assert "some moves may be missing" in fn
    assert "still loading" in fn
    # The true-empty state keeps its exact prior copy.
    assert "all caught up" in fn.lower()
    assert "No lineup, waiver, roster, or deadline moves need your attention" in fn


def test_changelog_announces_moves_scan_honesty():
    from dashboard_services.changelog import CHANGELOG

    entry = next(
        e for e in CHANGELOG
        if "this week's moves" in e.get("text", "").lower()
        and "could not check" in e.get("text", "").lower()
    )
    assert entry["tag"] == "fix"
    assert "swap" in entry["text"].lower()
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]
