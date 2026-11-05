"""Wave 3 PR 3: claims funding, blockers, asset-conflict sequencing.

Spec: Wave 3 sections 3.5, 3.6 and 8 PR 3 (unit tests; the full
league-92916 fixture lives in test_gm_brief_92916.py).
"""
import pytest

pytest.importorskip("pandas")

from dashboard_services import gm_brief as gb


def _player(team, pos, name, status=""):
    return {"full_name": name, "team": team, "position": pos,
            "injury_status": status}


def _claims_ctx(**over):
    players = {
        "claim_a": _player("AAA", "QB", "Claim Alpha"),
        "claim_b": _player("BBB", "QB", "Claim Bravo"),
        "harvey": _player("DEN", "RB", "Hurt Back", status="IR"),
        "bench_low": _player("CCC", "TE", "Bench Low"),
        "bench_mid": _player("DDD", "WR", "Bench Mid"),
        "starter": _player("EEE", "RB", "Star Starter"),
    }
    ctx = {
        "current_week": 9,
        "season": 2026,
        "current_season": 2026,
        "players": players,
        "rosters": [{
            "roster_id": "1",
            "players": ["harvey", "bench_low", "bench_mid", "starter"],
            "starters": ["starter"],
            "reserve": [],
            "settings": {"waiver_position": 5},
        }],
        "roster_map": {"1": "Viewer Team"},
        "league": {"settings": {"reserve_slots": 1}},
        "league_rules": {"source": "user_confirmed",
                         "waiver_type": "rolling",
                         "waiver_process_time": "03:00",
                         "waiver_process_days": "daily",
                         "waiver_process_tz": "America/New_York",
                         "playoff_week_start": 15},
        "model_value_table": [
            {"id": "bench_low", "value": 50},
            {"id": "bench_mid", "value": 300},
            {"id": "starter", "value": 900},
            {"id": "harvey", "value": 400},
        ],
    }
    ctx.update(over)
    return ctx


def _claim(pid, name, score=80):
    return {"kind": "waiver", "tag": "Waivers", "action": f"Claim {name}",
            "why": "Gate reason.", "cta_label": "Claim now",
            "cta_url": "#", "score": score, "player_id": pid}


def test_claims_funding_ir_move_first_then_named_drop():
    actions = [_claim("claim_a", "Claim Alpha", 80),
               _claim("claim_b", "Claim Bravo", 78)]
    brief = gb.build_gm_brief(_claims_ctx(), "1", actions, now=0)
    first, second = brief["claims"]
    assert first["funding"]["kind"] == "ir_move"
    assert first["funding"]["name"] == "Hurt Back"
    assert second["funding"]["kind"] == "drop"
    assert second["funding"]["name"] == "Bench Low"
    assert first["waiver_priority"] == 5
    assert "3:00 AM ET" in (first["process_label"] or "")
    assert first["blocker"] is None


def test_claim_without_funding_gets_blocker_not_cta():
    # Every roster spot is a starter: no IR move, no droppable bench.
    ctx = _claims_ctx()
    ctx["rosters"][0]["players"] = ["starter"]
    ctx["rosters"][0]["starters"] = ["starter"]
    ctx["league"] = {"settings": {"reserve_slots": 0}}
    brief = gb.build_gm_brief(ctx, "1", [_claim("claim_a", "Claim Alpha")],
                              now=0)
    claim = brief["claims"][0]
    assert claim["funding"] is None
    assert claim["blocker"] and "No legal funding move" in claim["blocker"]
    action = brief["actions"][0]
    assert action["blocker"] == claim["blocker"]


def test_blocked_claim_renders_blocker_instead_of_claim_cta(monkeypatch):
    pytest.importorskip("flask")
    import app as appmod

    ctx = _claims_ctx()
    ctx["rosters"][0]["players"] = ["starter"]
    ctx["rosters"][0]["starters"] = ["starter"]
    ctx["league"] = {"settings": {"reserve_slots": 0}}
    claim = _claim("claim_a", "Claim Alpha")
    monkeypatch.setattr(appmod, "_next_steps_lineup_actions",
                        lambda c, r: [])
    monkeypatch.setattr(appmod, "_next_steps_waiver_actions",
                        lambda c, r, m: [claim])
    monkeypatch.setattr(appmod, "_next_steps_trade_actions",
                        lambda c, r: [])
    monkeypatch.setattr(appmod, "_next_steps_roster_actions",
                        lambda c, r: [])
    monkeypatch.setattr(appmod, "_next_steps_bench_check_actions",
                        lambda c, r, w: [])
    monkeypatch.setattr(appmod, "_next_steps_bench_check_note",
                        lambda c, r, w: "")
    html_out = appmod._render_next_steps_queue(
        ctx, "1", [], season=2026, current_week=9)
    assert "No legal funding move" in html_out
    assert ">Claim now<" not in html_out
    # The claims block groups it with the blocker named too.
    assert "Claims" in html_out


# ---------------------------------------------------------------------------
# sequencing (3.6)
# ---------------------------------------------------------------------------

def _trade(partner, target, sends, score, **over):
    action = {"kind": "trade", "tag": "Trade", "partner": partner,
              "action": f"Offer {' + '.join(sends)} for {target}",
              "why": f"{partner} fit.", "cta_label": "Propose trade",
              "cta_url": "#", "score": score,
              "send_ids": list(sends), "receive_ids": [target]}
    action.update(over)
    return action


def test_shared_send_asset_second_deal_is_contingent_and_demoted():
    first = _trade("Goodfellas", "caleb", ["watson", "mccarthy"], 80)
    second = _trade("Cheeseheads", "love", ["watson", "dobbins"], 75)
    other = _trade("Others", "burrow", ["moore"], 70)
    brief = gb.build_gm_brief(_claims_ctx(), "1",
                              [second, other, first], now=0)
    by_partner = {a["partner"]: a for a in brief["actions"]
                  if a.get("kind") == "trade"}
    assert by_partner["Goodfellas"]["sequence"]["step"] == 1
    assert by_partner["Goodfellas"]["sequence"]["note"] == "Send first"
    love = by_partner["Cheeseheads"]
    assert love["sequence"]["step"] == 2
    assert love["why"].startswith("If the Goodfellas deal dies:")
    # Demoted below every step-1 action.
    assert love["score"] < by_partner["Goodfellas"]["score"]
    assert love["score"] < by_partner["Others"]["score"] or \
        by_partner["Others"].get("sequence", {}).get("step") != 1
    order = [a["action"] for a in brief["actions"]]
    assert order.index(love["action"]) > order.index(
        by_partner["Goodfellas"]["action"])


def test_picks_count_as_assets():
    first = _trade("A Team", "caleb", [], 80,
                   send_picks=[{"season": 2027, "round": 1,
                                "original_owner": "3"}])
    second = _trade("B Team", "love", [], 75,
                    send_picks=[{"season": 2027, "round": 1,
                                 "original_owner": "3"}])
    brief = gb.build_gm_brief(_claims_ctx(), "1", [first, second], now=0)
    seq = {a["partner"]: a["sequence"]["step"] for a in brief["actions"]}
    assert seq == {"A Team": 1, "B Team": 2}


def test_funding_drop_in_package_is_same_conflict():
    # Claim Bravo (second claim) is funded by dropping Bench Low; a
    # trade that also sends Bench Low is the same conflict: the deal
    # becomes contingent on the claim.
    actions = [_claim("claim_a", "Claim Alpha", 90),
               _claim("claim_b", "Claim Bravo", 88),
               _trade("Goodfellas", "caleb", ["bench_mid"], 80)]
    brief = gb.build_gm_brief(_claims_ctx(), "1", actions, now=0)
    assert brief["claims"][1]["funding"]["name"] == "Bench Low"
    # bench_mid is not the drop, so no conflict on this deal.
    trade_out = next(a for a in brief["actions"] if a["kind"] == "trade")
    assert "sequence" not in trade_out

    actions = [_claim("claim_a", "Claim Alpha", 90),
               _claim("claim_b", "Claim Bravo", 88),
               _trade("Goodfellas", "caleb", ["bench_low"], 80)]
    brief = gb.build_gm_brief(_claims_ctx(), "1", actions, now=0)
    trade_out = next(a for a in brief["actions"] if a["kind"] == "trade")
    assert trade_out["sequence"]["step"] == 2
    assert "claim dies:" in trade_out["why"]
