"""Wave 2 PR 3: the shared partner checks wired into both engines.

Engine 1 (build_trade_suggestions_context): a Superflex partner whose
QB room only *looks* deep on value — the QB2 is OUT and the QB3 has no
starter projection — cannot sell his QB1 (role-aware surplus, spec
2.5/3.2 check 1). With a genuinely startable room, the same deal is
pitched.

Engine 2 (/api/trade-targets): the Burrow fixture — an SF partner with
exactly two startable QBs is not listed when the viewer cannot return
a QB; with a third startable QB he is.
"""
import pytest

pytest.importorskip("pandas")

import dashboard_services.ai.context_builders as cb

_SF_LINEUP = ["QB", "SUPER_FLEX", "RB", "RB", "WR", "WR", "TE"]


def _mv(pid, name, pos, val):
    return {"player_id": pid, "name": name, "position": pos,
            "value": val, "sf_value": val}


def _sf_ctx(rodgers_out: bool):
    mvt = []
    rosters = []
    roster_map = {}

    def team(rid, rows):
        pids = []
        for pid, name, pos, val in rows:
            mvt.append(_mv(pid, name, pos, val))
            pids.append(pid)
        rosters.append({"roster_id": rid, "players": pids})
        roster_map[rid] = f"Team {rid}"

    # Viewer: 1 startable QB (SF hole), 3 startable WRs (surplus).
    team("1", [
        ("v_qb1", "Viewer QB1", "QB", 1600),
        ("v_wr1", "Viewer WR1", "WR", 2400),
        ("v_wr2", "Viewer WR2", "WR", 2200),
        ("v_wr3", "Viewer WR3", "WR", 1900),
        ("v_rb1", "Viewer RB1", "RB", 1000),
        ("v_rb2", "Viewer RB2", "RB", 900),
        ("v_te1", "Viewer TE1", "TE", 800),
    ])
    # Partner: 3 QBs above the value bar, 1 startable WR (hole).
    team("2", [
        ("burrow", "Joe Burrow", "QB", 2000),
        ("rodgers", "Aaron Rodgers", "QB", 1500),
        ("qb3", "Partner QB3", "QB", 1300),
        ("p_wr1", "Partner WR1", "WR", 1500),
        ("p_wr2", "Partner WR2", "WR", 100),
        ("p_rb1", "Partner RB1", "RB", 1000),
        ("p_rb2", "Partner RB2", "RB", 900),
        ("p_te1", "Partner TE1", "TE", 800),
    ])
    # Filler teams: balanced, no surplus/need vs the viewer.
    for i in range(3, 11):
        team(str(i), [
            (f"f{i}_qb1", f"F{i} QB1", "QB", 1400),
            (f"f{i}_qb2", f"F{i} QB2", "QB", 1200),
            (f"f{i}_rb1", f"F{i} RB1", "RB", 1100),
            (f"f{i}_rb2", f"F{i} RB2", "RB", 1000),
            (f"f{i}_wr1", f"F{i} WR1", "WR", 1300),
            (f"f{i}_wr2", f"F{i} WR2", "WR", 1200),
            (f"f{i}_te1", f"F{i} TE1", "TE", 800),
        ])

    players = {
        "burrow": {"position": "QB", "team": "CIN"},
        "rodgers": {"position": "QB", "team": "PIT",
                    **({"injury_status": "OUT"} if rodgers_out else {})},
        "qb3": {"position": "QB", "team": "NYJ"},
    }
    projections = {"burrow": 24.0, "rodgers": 22.0,
                   # QB3 projects below the QB starter bar (14): with
                   # Rodgers out the room has ONE startable QB, not 3.
                   "qb3": 8.0 if rodgers_out else 20.0}
    return {
        "rosters": rosters,
        "model_value_table": mvt,
        "roster_map": roster_map,
        "picks_by_roster": {},
        "standings_map": {r["roster_id"]: {"wins": 5, "losses": 5}
                          for r in rosters},
        "rookie_rankings": [],
        "league_type": "sf",
        "roster_positions": _SF_LINEUP,
        "players": players,
        "projections": projections,
        "current_week": 9,
    }


def _run_suggestions(monkeypatch, ctx):
    monkeypatch.setattr(
        cb, "build_team_gm_context",
        lambda c, rid: {"team_name": "Viewer", "direction": "balanced"})
    return cb.build_trade_suggestions_context(ctx, "1")


def test_engine1_burrow_blocked_when_qb_room_is_thin_on_role(monkeypatch):
    res = _run_suggestions(monkeypatch, _sf_ctx(rodgers_out=True))
    assert res is not None
    for partner in res["top_partners"]:
        target_ids = {str(t.get("id")) for t in partner["targets_they_have"]}
        assert "burrow" not in target_ids


def test_engine1_burrow_pitched_when_room_is_genuinely_deep(monkeypatch):
    res = _run_suggestions(monkeypatch, _sf_ctx(rodgers_out=False))
    assert res is not None
    pitched = {
        str(t.get("id"))
        for partner in res["top_partners"]
        for t in partner["targets_they_have"]
    }
    assert "burrow" in pitched
    entry = next(
        p for p in res["top_partners"]
        if any(str(t.get("id")) == "burrow" for t in p["targets_they_have"]))
    # The conditioned acceptance + contention ride along on the entry.
    assert entry["acceptance_rate"] is not None
    assert entry["partner_contention"] in ("contender", "bubble", "seller")


# --- Engine 2: /api/trade-targets ---------------------------------------------

import app as app_mod  # noqa: E402


def _row(pid, name, pos, value, age=25):
    return {"id": pid, "name": name, "position": pos, "value": value,
            "sf_value": value, "age": age, "team": "BUF",
            "pos_rank_label": f"{pos}1", "rank_change_7d": 0}


def _targets_ctx(partner_qbs):
    viewer = {"roster_id": 1, "players": [
        "qb-mine", "wr-mine", "rb-mine", "te-mine"]}
    partner = {"roster_id": 2, "players": list(partner_qbs) + [
        "wr-theirs", "rb-theirs", "te-theirs"]}
    # Values are scaled to the 2-roster threshold scale the handler
    # derives (SF QB starter bar = 1600 with two rosters in the ctx).
    rows = [
        _row("qb-mine", "My QB", "QB", 1700, age=26),
        _row("wr-mine", "My WR", "WR", 1600),
        _row("rb-mine", "My RB", "RB", 1500),
        _row("te-mine", "My TE", "TE", 800),
        _row("wr-theirs", "Their WR", "WR", 1200),
        _row("rb-theirs", "Their RB", "RB", 1100),
        _row("te-theirs", "Their TE", "TE", 700),
    ]
    qb_rows = {
        "burrow": _row("burrow", "Joe Burrow", "QB", 2900, age=27),
        "rodgers": _row("rodgers", "Aaron Rodgers", "QB", 2400, age=40),
        "qb-third": _row("qb-third", "Third QB", "QB", 2100, age=24),
    }
    rows += [qb_rows[q] for q in partner_qbs]
    return {
        "platform": "fleaflicker", "league_id": "92916", "season": 2026,
        "rosters": [viewer, partner],
        "roster_map": {"1": "Mine", "2": "American Dream"},
        "model_value_table": rows,
        "players_index": {},
        "roster_positions": ["QB", "SUPER_FLEX", "RB", "RB", "WR", "WR",
                             "TE", "K", "DEF"],
        "picks_by_roster": {},
        "league_settings": {"type": 2},
    }


def _get_targets(monkeypatch, ctx):
    monkeypatch.setattr(app_mod, "has_premium_for_viewer",
                        lambda *a, **k: True)
    monkeypatch.setattr(app_mod, "load_pick_value_table",
                        lambda *a, **k: {})
    monkeypatch.setattr(app_mod, "get_league_ctx_from_cache",
                        lambda *a, **k: ctx)
    client = app_mod.app.test_client()
    resp = client.get(
        "/api/trade-targets?platform=fleaflicker&league_id=92916"
        "&season=2026&viewer_roster_id=1&league_type=sf&league_size=10")
    assert resp.status_code == 200, resp.get_data(as_text=True)[:400]
    return resp.get_json()


def _surfaced_ids(payload):
    ids = set()

    def walk(node):
        if isinstance(node, dict):
            pid = node.get("player_id")
            if pid:
                ids.add(str(pid))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for key in ("targets", "by_position", "all_positions"):
        walk(payload.get(key))
    return ids


def test_engine2_burrow_not_listed_with_exactly_two_startable_qbs(monkeypatch):
    payload = _get_targets(monkeypatch, _targets_ctx(["burrow", "rodgers"]))
    # Viewer fields only one QB and cannot return one: Burrow (and the
    # partner's QB2) are not pitchable targets in Superflex.
    assert "burrow" not in _surfaced_ids(payload)


def test_engine2_burrow_listed_with_real_qb_surplus(monkeypatch):
    payload = _get_targets(
        monkeypatch, _targets_ctx(["burrow", "rodgers", "qb-third"]))
    assert "burrow" in _surfaced_ids(payload)
