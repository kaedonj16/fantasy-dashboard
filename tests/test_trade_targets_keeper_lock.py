"""Wave 1 PR 2: /api/trade-targets keeper legality.

In a league whose designated keepers are known, a locked player must
never be surfaced as an acquisition target (the partner cannot trade
him) and never offered as one of the viewer's chips (the viewer cannot
trade him either). With designations unknown, output is unchanged.
"""
import pytest

pytest.importorskip("pandas")

import app as app_mod


def _row(pid, name, pos, value, age=25):
    return {"id": pid, "name": name, "position": pos, "value": value,
            "age": age, "team": "BUF", "pos_rank_label": f"{pos}1",
            "rank_change_7d": 0}


def _ctx(with_keepers=True):
    viewer = {"roster_id": 1,
              "players": ["qb-mine", "wr-chip", "rb-mine", "te-mine"]}
    partner = {"roster_id": 2,
               "players": ["qb-star", "qb-ok", "wr-theirs", "rb-theirs"]}
    if with_keepers:
        viewer["keepers"] = ["wr-chip"]
        partner["keepers"] = ["qb-star"]
    return {
        "platform": "fleaflicker", "league_id": "92916", "season": 2026,
        "rosters": [viewer, partner],
        "roster_map": {"1": "Mine", "2": "Theirs"},
        "model_value_table": [
            _row("qb-mine", "My QB", "QB", 300, age=26),
            _row("wr-chip", "Chip WR", "WR", 800),
            _row("rb-mine", "My RB", "RB", 500),
            _row("te-mine", "My TE", "TE", 200),
            _row("qb-star", "Star QB", "QB", 900, age=27),
            _row("qb-ok", "OK QB", "QB", 250),
            _row("wr-theirs", "Their WR", "WR", 400),
            _row("rb-theirs", "Their RB", "RB", 350),
        ],
        "players_index": {},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                             "K", "DEF"],
        "picks_by_roster": {},
        "league_settings": {"type": 2},
    }


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
        "&season=2026&viewer_roster_id=1&league_type=1qb&league_size=10")
    assert resp.status_code == 200, resp.get_data(as_text=True)[:400]
    return resp.get_json()


def test_locked_target_and_chip_never_surface(monkeypatch):
    payload = _get_targets(monkeypatch, _ctx(with_keepers=True))
    surfaced = _surfaced_ids(payload)
    assert surfaced, "fixture should still surface unlocked targets"
    # The partner's best QB is designated-keeper locked: not a target.
    assert "qb-star" not in surfaced
    # The viewer's best chip is locked too: not offered as a chip.
    chip_ids = {c["player_id"] for c in payload.get("top_chips") or []}
    assert "wr-chip" not in chip_ids
    assert chip_ids, "unlocked chips still surface"


def test_unknown_designations_leave_output_unchanged(monkeypatch):
    payload = _get_targets(monkeypatch, _ctx(with_keepers=False))
    surfaced = _surfaced_ids(payload)
    # Control: absent keeper data, the same players DO surface, proving
    # the exclusions above come from the lock and nothing else.
    assert "qb-star" in surfaced
    chip_ids = {c["player_id"] for c in payload.get("top_chips") or []}
    assert "wr-chip" in chip_ids
