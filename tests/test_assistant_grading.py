"""Stage 2 outcome grading tests: no DB, no network.

store.get_conn is faked with an in-memory pattern-matching fake (same
approach as tests/test_assistant_service.py); the graders' data sources
(Sleeper week stats, NFL state, league ctx, value history, metrics) are
monkeypatched through grading's lazy shims.
"""
import re
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.integration

from dashboard_services.assistant import grading, review_packet, store

UTC = timezone.utc
NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


# -- in-memory fake DB ---------------------------------------------------------

class FakeCursor:
    def __init__(self, rows=None, row=None, rowcount=0):
        self._rows = rows or []
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        if self._row is not None:
            return self._row
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        pass

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        st = self.state
        if s.startswith("CREATE"):
            return FakeCursor()
        if "UPDATE assistant_advice_log SET grade" in s:
            grade, grader, detail, log_id = params
            for r in st["logs"]:
                if r.get("log_id") == log_id and r.get("grade") is None:
                    r["grade"] = grade
                    r["grader"] = grader
                    r["outcome_detail"] = detail
                    r["graded_at"] = NOW
                    return FakeCursor(rowcount=1)
            return FakeCursor(rowcount=0)
        if "FROM assistant_advice_log" in s and "grade IS NULL" in s:
            rows = [dict(r) for r in st["logs"]
                    if r.get("intent") == params[0] and r.get("grade") is None]
            rows.sort(key=lambda r: r["log_id"])
            return FakeCursor(rows=rows[: params[1]])
        if "FROM assistant_advice_log" in s and "graded_at >=" in s:
            rows = [dict(r) for r in st["logs"]
                    if r.get("graded_at") is not None and params[0] <= r["graded_at"] < params[1]]
            return FakeCursor(rows=rows)
        if "FROM assistant_advice_log" in s and "created_at >=" in s:
            rows = [dict(r) for r in st["logs"] if params[0] <= r["created_at"] < params[1]]
            return FakeCursor(rows=rows)
        raise AssertionError(f"fake DB cannot handle SQL: {s}")


def _row(**kw):
    base = {
        "log_id": 1, "created_at": NOW - timedelta(days=40), "account_key": "acct:1",
        "league_id": "L1", "platform": "sleeper", "season": 2026, "week": 6,
        "question_text": "Start Odunze or Mooney?", "intent": "start_sit",
        "entities": {}, "evidence_pack": {}, "answer_text": "Start Odunze.",
        "verdict": {"intent": "start_sit", "call": "START Odunze"},
        "numbers_shown": {}, "thumb": None, "correction_text": None,
        "corrected_claim": None, "correction_category": None, "reask_flag": False,
        "proactive": False, "trigger": None, "grade": None, "grader": None,
        "graded_at": None, "outcome_detail": None,
    }
    base.update(kw)
    return base


def _start_sit_pack(pick="1", players=None):
    return {
        "type": "start_sit", "available": True, "week": 6,
        "league": {"platform": "sleeper", "league_id": "L1", "season": 2026},
        "pick_player_id": pick,
        "players": players if players is not None else [
            {"player_id": "1", "name": "Rome Odunze", "position": "WR"},
            {"player_id": "2", "name": "Darnell Mooney", "position": "WR"},
        ],
    }


@pytest.fixture()
def env(monkeypatch):
    state = {"logs": []}
    conn = FakeConn(state)
    monkeypatch.setattr(store, "_TABLE_READY", True)
    monkeypatch.setattr(store, "get_conn", lambda *a, **k: conn)
    monkeypatch.setattr(grading, "get_nfl_state",
                        lambda: {"season": 2026, "week": 8, "display_week": 8})
    monkeypatch.setattr(grading, "get_league_ctx_from_cache",
                        lambda *a, **k: {"raw_scoring_settings": {"rec": 1.0}})
    monkeypatch.setattr(grading, "week_stat_points",
                        lambda line, scoring, pos: float(line.get("pts", 0.0)))
    return state


# -- pure banding ---------------------------------------------------------------

def test_classify_start_sit_bands():
    assert grading.classify_start_sit(24.0, 24.0) == "hit"      # tie goes to the call
    assert grading.classify_start_sit(25.0, 20.0) == "hit"
    assert grading.classify_start_sit(20.0, 21.5) == "partial"  # inside the 10% margin
    assert grading.classify_start_sit(10.0, 25.0) == "miss"


def test_classify_value_drift_bands():
    assert grading.classify_value_drift("buy", 12.0) == "hit"
    assert grading.classify_value_drift("buy", 0.0) == "partial"
    assert grading.classify_value_drift("buy", -9.0) == "miss"
    assert grading.classify_value_drift("sell", -8.0) == "hit"
    assert grading.classify_value_drift("sell", 6.0) == "miss"
    assert grading.classify_value_drift("hold", 3.0) == "hit"
    assert grading.classify_value_drift("hold", 10.0) == "partial"
    assert grading.classify_value_drift("hold", 22.0) == "miss"


# -- start / sit grader -----------------------------------------------------------

def test_start_sit_hit_writes_grade_and_detail(env, monkeypatch):
    env["logs"].append(_row(evidence_pack=_start_sit_pack()))
    monkeypatch.setattr(grading, "load_sleeper_week_stats",
                        lambda s, w: {"1": {"pts": 24.5}, "2": {"pts": 18.0}})
    summary = grading.run_graders(now=NOW)
    assert summary["start_sit"]["graded"] == 1
    assert summary["start_sit"]["hit"] == 1
    row = env["logs"][0]
    assert row["grade"] == "hit"
    assert row["grader"] == grading.START_SIT_GRADER
    import json
    detail = json.loads(row["outcome_detail"])
    assert detail["pick"] == {"player_id": "1", "name": "Rome Odunze", "points": 24.5}
    assert detail["best_alternative"]["player_id"] == "2"
    assert detail["delta"] == 6.5


def test_start_sit_unfinalized_week_is_not_graded(env, monkeypatch):
    env["logs"].append(_row(evidence_pack=_start_sit_pack(), week=8))
    monkeypatch.setattr(grading, "load_sleeper_week_stats",
                        lambda s, w: {"1": {"pts": 30.0}, "2": {"pts": 1.0}})
    summary = grading.run_graders(now=NOW)
    assert summary["start_sit"]["graded"] == 0
    assert summary["start_sit"]["skipped"] == 1
    assert env["logs"][0]["grade"] is None


def test_start_sit_missing_alternative_or_stats_skips(env, monkeypatch):
    env["logs"].append(_row(log_id=1, evidence_pack=_start_sit_pack(
        players=[{"player_id": "1", "name": "Rome Odunze", "position": "WR"}])))
    env["logs"].append(_row(log_id=2, evidence_pack=_start_sit_pack()))
    stats = {"empty": True}
    monkeypatch.setattr(grading, "load_sleeper_week_stats",
                        lambda s, w: {} if stats["empty"] else {"1": {"pts": 9.0}, "2": {"pts": 9.0}})
    summary = grading.run_graders(now=NOW)
    assert summary["start_sit"]["graded"] == 0
    assert summary["start_sit"]["skipped"] == 2
    assert all(r["grade"] is None for r in env["logs"])


def test_start_sit_grading_is_idempotent(env, monkeypatch):
    env["logs"].append(_row(evidence_pack=_start_sit_pack()))
    monkeypatch.setattr(grading, "load_sleeper_week_stats",
                        lambda s, w: {"1": {"pts": 5.0}, "2": {"pts": 20.0}})
    first = grading.run_graders(now=NOW)
    assert first["start_sit"]["miss"] == 1
    second = grading.run_graders(now=NOW)
    assert second["start_sit"]["considered"] == 0  # graded rows are not re-fetched
    assert env["logs"][0]["grade"] == "miss"
    # The store guard itself refuses a rewrite even when asked directly.
    assert store.write_grade(1, "hit", "someone-else-v9", {}) is False
    assert env["logs"][0]["grade"] == "miss"


def test_start_sit_dry_run_writes_nothing(env, monkeypatch):
    env["logs"].append(_row(evidence_pack=_start_sit_pack()))
    monkeypatch.setattr(grading, "load_sleeper_week_stats",
                        lambda s, w: {"1": {"pts": 22.0}, "2": {"pts": 10.0}})
    summary = grading.run_graders(now=NOW, dry_run=True)
    assert summary["start_sit"]["graded"] == 1
    assert env["logs"][0]["grade"] is None


# -- buy / sell / hold grader -------------------------------------------------------

def _buy_sell_row(**kw):
    pack = {
        "type": "buy_sell_hold", "available": True,
        "player": {"player_id": "9", "name": "Test Player", "value": 4200},
        "signal": "buy", "role": {"ppg": 12.0},
        "league": {"platform": "sleeper", "league_id": "L1", "season": 2026},
    }
    base = dict(intent="buy_sell_hold", week=None,
                verdict={"intent": "buy_sell_hold", "call": "BUY"},
                evidence_pack=pack, question_text="Is Test Player a buy?")
    base.update(kw)
    return _row(**base)


def _history(points):
    return [{"as_of_date": d, "value": v} for d, v in points]


def test_buy_sell_hit_at_four_weeks_with_ppg_detail(env, monkeypatch):
    env["logs"].append(_buy_sell_row())  # created 40 days before NOW
    monkeypatch.setattr(grading, "get_player_value_history",
                        lambda pid, days=30: _history([("2026-10-05", 4000.0),
                                                       ("2026-11-14", 4600.0)]))
    monkeypatch.setattr(grading, "get_player_metrics", lambda pid: {"ppg": 15.5})
    summary = grading.run_graders(now=NOW)
    assert summary["buy_sell_hold"]["hit"] == 1
    row = env["logs"][0]
    assert row["grade"] == "hit"
    assert row["grader"] == grading.VALUE_DRIFT_GRADER
    import json
    detail = json.loads(row["outcome_detail"])
    assert detail["call"] == "buy"
    assert detail["value_at_answer"] == 4000.0
    assert detail["value_now"] == 4600.0
    assert detail["value_delta_pct"] == 15.0
    assert detail["ppg_delta"] == 3.5


def test_buy_sell_too_young_is_not_graded(env, monkeypatch):
    env["logs"].append(_buy_sell_row(created_at=NOW - timedelta(days=10)))
    monkeypatch.setattr(grading, "get_player_value_history",
                        lambda pid, days=30: _history([("2026-11-01", 4000.0),
                                                       ("2026-11-14", 9000.0)]))
    summary = grading.run_graders(now=NOW)
    assert summary["buy_sell_hold"]["graded"] == 0
    assert env["logs"][0]["grade"] is None


def test_buy_sell_missing_history_is_ungraded_not_miss(env, monkeypatch):
    env["logs"].append(_buy_sell_row(log_id=1))
    env["logs"].append(_buy_sell_row(log_id=2))
    calls = {"n": 0}

    def fake_history(pid, days=30):
        calls["n"] += 1
        if calls["n"] == 1:
            return []  # no history at all
        # history exists but starts after the answer: no baseline
        return _history([("2026-11-10", 5000.0), ("2026-11-14", 5100.0)])

    monkeypatch.setattr(grading, "get_player_value_history", fake_history)
    summary = grading.run_graders(now=NOW)
    assert summary["buy_sell_hold"]["graded"] == 0
    assert summary["buy_sell_hold"]["miss"] == 0
    assert all(r["grade"] is None for r in env["logs"])


def test_buy_sell_sell_miss_and_hold_hit(env, monkeypatch):
    sell = _buy_sell_row(log_id=1)
    sell["evidence_pack"] = {**sell["evidence_pack"], "signal": "sell"}
    hold = _buy_sell_row(log_id=2)
    hold["evidence_pack"] = {**hold["evidence_pack"], "signal": "hold"}
    env["logs"] += [sell, hold]
    monkeypatch.setattr(grading, "get_player_value_history",
                        lambda pid, days=30: _history([("2026-10-05", 4000.0),
                                                       ("2026-11-14", 4400.0)]))
    summary = grading.run_graders(now=NOW)
    # Value rose 10% over the window: the sell was wrong (miss), and the
    # hold was not flat enough to hit but inside the partial band.
    assert summary["buy_sell_hold"]["miss"] == 1
    assert summary["buy_sell_hold"]["partial"] == 1
    assert summary["buy_sell_hold"]["hit"] == 0


# -- weekly review packet -------------------------------------------------------------

def _packet_rows():
    current = [
        _row(log_id=1, created_at=NOW - timedelta(days=2), thumb="down",
             grade="miss", graded_at=NOW - timedelta(days=1),
             outcome_detail={"pick": {"name": "Rome Odunze", "points": 4.0},
                             "best_alternative": {"name": "Darnell Mooney", "points": 21.0},
                             "delta": -17.0},
             numbers_shown={"projection_gap": "2.1"},
             evidence_pack=_start_sit_pack()),
        _row(log_id=2, created_at=NOW - timedelta(days=3), intent="player",
             question_text="How is Odunze doing?", proactive=False),
        _row(log_id=3, created_at=NOW - timedelta(days=1), intent="out_of_scope",
             question_text="Who wins the Super Bowl?"),
        _row(log_id=4, created_at=NOW - timedelta(days=1), intent="this_week",
             proactive=True, trigger="digest", question_text=None),
    ]
    # three corrections on the same player and claim: a draft data ticket
    for i in range(3):
        current.append(_row(
            log_id=10 + i, created_at=NOW - timedelta(days=2), intent="player",
            correction_category="wrong_data", corrected_claim="target_share",
            correction_text="his target share is 24%, not 19%",
            evidence_pack={"type": "player", "player": {"player_id": "1", "name": "Rome Odunze"}},
        ))
    current.append(_row(log_id=20, created_at=NOW - timedelta(days=4),
                        intent="metric", correction_category="bad_reasoning",
                        corrected_claim=None, evidence_pack={"type": "metric"}))
    graded = [r for r in current if r.get("grade")]
    graded.append(_row(log_id=30, created_at=NOW - timedelta(days=30),
                       grade="hit", graded_at=NOW - timedelta(days=2),
                       outcome_detail={"pick": {"name": "A", "points": 20.0},
                                       "best_alternative": {"name": "B", "points": 9.0},
                                       "delta": 11.0},
                       evidence_pack=_start_sit_pack()))
    prior = [_row(log_id=40, grade="miss", graded_at=NOW - timedelta(days=9),
                  outcome_detail={}, evidence_pack=_start_sit_pack())]
    return current, graded, prior


def test_packet_contents_from_seeded_log():
    current, graded, prior = _packet_rows()
    packet = review_packet.build_packet(
        current, graded, prior, generated_at=NOW,
        window_start=NOW - timedelta(days=7), window_end=NOW,
        ungraded_backlog={"start_sit": 2, "buy_sell_hold": 0})
    assert packet["volume_by_type"]["start_sit"]["count"] == 1
    assert packet["volume_by_type"]["start_sit"]["down"] == 1
    assert packet["graded_by_type"]["start_sit"]["graded"] == 2
    assert packet["graded_by_type"]["start_sit"]["hit"] == 1
    assert packet["graded_by_type"]["start_sit"]["miss"] == 1
    assert packet["graded_by_type"]["start_sit"]["hit_rate"] == 0.5
    assert packet["graded_by_type"]["start_sit"]["prior_hit_rate"] == 0.0
    assert packet["graded_by_type"]["start_sit"]["trend"] == 0.5
    assert len(packet["worst_misses"]) == 1
    assert "Rome Odunze scored 4.0" in packet["worst_misses"][0]["outcome"]
    assert packet["downvoted"][0]["grade"] == "miss"
    assert packet["corrections_by_kind"] == {"wrong_data": 3, "bad_reasoning": 1}
    assert len(packet["draft_tickets"]) == 1
    ticket = packet["draft_tickets"][0]
    assert ticket["player"] == "Rome Odunze" and ticket["claim"] == "target_share"
    assert ticket["count"] == 3
    assert packet["unanswered"]["out_of_scope"] == 1
    assert packet["proactive"] == {"total": 1, "by_trigger": {"digest": 1}}
    assert packet["grading_debt_flags"] == []  # grades were written this cycle

    md = review_packet.render_markdown(packet)
    assert "BR Assistant weekly review packet" in md
    assert "Draft data tickets" in md
    assert "target_share" in md
    assert "—" not in md  # no em dashes in generated copy


def test_packet_grading_debt_flag():
    packet = review_packet.build_packet(
        [], [], [], generated_at=NOW,
        window_start=NOW - timedelta(days=7), window_end=NOW,
        ungraded_backlog={"start_sit": 5, "buy_sell_hold": 0})
    assert len(packet["grading_debt_flags"]) == 1
    assert "start_sit" in packet["grading_debt_flags"][0]


def test_generate_packet_writes_file(env, tmp_path):
    current, graded, prior = _packet_rows()
    for r in current + graded + prior:
        env["logs"].append(r)
    path, packet = review_packet.generate_packet(now=NOW, out_dir=str(tmp_path))
    assert path.endswith(".md")
    text = open(path, encoding="utf-8").read()
    assert "weekly review packet" in text
    assert packet["proactive"]["total"] == 1
