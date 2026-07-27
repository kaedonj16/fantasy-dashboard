"""JS↔Python parity for the keeper math.

static/keeper.js re-implements utils/keeper_value.py so the keeper page recomputes
live as the manager tweaks rules — and a comment there says "keep the math here in
sync with the Python engine." Nothing enforced that until this test: it extracts
the shipped JS math (cost round, market round, verdict, collision resolution) and
runs it through Node against outputs computed by the real Python engine, over a
case table. If either side's math drifts, the two disagree and this fails.

Skipped when Node.js isn't available (same guard as the other JS tests).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

import pytest

from utils.keeper_value import (
    KeeperRules, KeeperCandidate, analyze, evaluate,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KEEPER_JS = os.path.join(_REPO_ROOT, "static", "keeper.js")

# Pull the pure math out of keeper.js verbatim: clamp / costRound / marketRound /
# verdict / resolveCollisions, i.e. everything from `function clamp` up to (but
# not including) `function compute`. These depend only on the module vars
# numRounds / leagueSize / lastBumps, which the harness supplies.
_MATH_RE = re.compile(r"function clamp\(n, lo, hi\).*?(?=\n  function compute\(\))", re.DOTALL)

_NUM_ROUNDS = 15
_LEAGUE_SIZE = 12


def _extract_math() -> str:
    src = open(_KEEPER_JS, encoding="utf-8").read()
    m = _MATH_RE.search(src)
    assert m, "could not locate the keeper math block in static/keeper.js"
    return m.group(0)


def _rules(**kw) -> KeeperRules:
    base = dict(league_size=_LEAGUE_SIZE, num_rounds=_NUM_ROUNDS,
                round_offset=0, escalation=1, undrafted_round=_NUM_ROUNDS,
                keep_at=2, pass_at=0)
    base.update(kw)
    return KeeperRules(**base)


def _js_rule(r: KeeperRules) -> dict:
    return {"roundOffset": r.round_offset, "escalation": r.escalation,
            "undraftedRound": r.undrafted_round, "keepAt": r.keep_at, "passAt": r.pass_at}


def _unit_cases():
    """(candidate, rules) pairs spanning cost/market/verdict behavior."""
    return [
        # drafted, no escalation
        (KeeperCandidate("1", "A", "TE", 11, 0, 30, 985), _rules()),
        # escalation: kept 2 yrs -> 2 rounds earlier
        (KeeperCandidate("2", "B", "WR", 11, 2, 18, 900), _rules(escalation=1)),
        # heavier escalation
        (KeeperCandidate("3", "C", "RB", 11, 3, 4, 800), _rules(escalation=2)),
        # undrafted -> undrafted_round
        (KeeperCandidate("4", "D", "WR", None, 0, 60, 500), _rules(undrafted_round=13)),
        # round offset earlier / later
        (KeeperCandidate("5", "E", "RB", 5, 0, 40, 400), _rules(round_offset=-1)),
        (KeeperCandidate("6", "F", "QB", 5, 0, 40, 400), _rules(round_offset=1)),
        # clamp at the shallow end
        (KeeperCandidate("7", "G", "RB", 1, 5, 4, 300), _rules(escalation=1)),
        # off-board (unknown ADP) -> no surplus, PASS
        (KeeperCandidate("8", "H", "WR", None, 0, None, 100), _rules()),
        # negative surplus -> PASS
        (KeeperCandidate("9", "I", "WR", 5, 0, 78, 690), _rules()),
        # exact market-round boundaries
        (KeeperCandidate("10", "J", "RB", 8, 0, 12, 700), _rules()),   # adp 12 -> R1
        (KeeperCandidate("11", "K", "RB", 8, 0, 13, 700), _rules()),   # adp 13 -> R2
    ]


def _build_unit_payload():
    out = []
    for c, r in _unit_cases():
        analyze(c, r)
        out.append({
            "p": {"draftedRound": c.drafted_round, "yearsKept": c.years_kept,
                  "adpOverall": c.adp_overall},
            "r": _js_rule(r),
            "expect": {"cost": c.cost_round, "mkt": c.market_round,
                       "surplus": c.surplus, "verdict": c.verdict},
        })
    return out


def _build_collision_payload():
    """Two keepers colliding on R5 (+ one clear keeper): parity on which round
    each ends up at after one-per-round resolution."""
    cands = [
        KeeperCandidate("a", "A", "RB", 5, 0, 4, 900),    # cost R5, +4
        KeeperCandidate("b", "B", "WR", 5, 0, 18, 800),   # cost R5, +3 -> bumps to R4
        KeeperCandidate("c", "C", "TE", 9, 0, 30, 700),   # cost R9, +6
    ]
    r = _rules()
    # Pre-resolution state (what keeper.js's compute() hands resolveCollisions).
    pre = evaluate([KeeperCandidate(*(c.player_id, c.name, c.position, c.drafted_round,
                                      c.years_kept, c.adp_overall, c.value)) for c in cands],
                   _rules(one_per_round=False), limit=3)
    rows = [{"id": c.player_id, "value": c.value, "cost": c.cost_round,
             "mkt": c.market_round, "surplus": c.surplus, "keep": c.keep} for c in pre]
    # Resolved truth from the engine.
    post = evaluate([KeeperCandidate(*(c.player_id, c.name, c.position, c.drafted_round,
                                       c.years_kept, c.adp_overall, c.value)) for c in cands],
                    _rules(one_per_round=True), limit=3)
    expect_cost = {c.player_id: c.cost_round for c in post if c.keep}
    return {"rule": _js_rule(r), "rows": rows, "expectCost": expect_cost}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_keeper_math_matches_python():
    cfg = {"numRounds": _NUM_ROUNDS, "leagueSize": _LEAGUE_SIZE}
    unit = _build_unit_payload()
    coll = _build_collision_payload()

    harness = (
        "var numRounds = %d, leagueSize = %d;\n" % (_NUM_ROUNDS, _LEAGUE_SIZE)
        + _extract_math() + "\n"
        + "var unit = " + json.dumps(unit) + ";\n"
        + "var coll = " + json.dumps(coll) + ";\n"
        + "var bad = [];\n"
        + "unit.forEach(function (t, i) {\n"
        + "  var cost = costRound(t.p, t.r);\n"
        + "  var mkt = marketRound(t.p);\n"
        + "  var surplus = mkt == null ? null : (cost - mkt);\n"
        + "  var v = verdict(surplus, t.r);\n"
        + "  if (cost !== t.expect.cost) bad.push('unit ' + i + ' cost ' + cost + ' != ' + t.expect.cost);\n"
        + "  if (mkt !== t.expect.mkt) bad.push('unit ' + i + ' mkt ' + mkt + ' != ' + t.expect.mkt);\n"
        + "  if (surplus !== t.expect.surplus) bad.push('unit ' + i + ' surplus ' + surplus + ' != ' + t.expect.surplus);\n"
        + "  if (v !== t.expect.verdict) bad.push('unit ' + i + ' verdict ' + v + ' != ' + t.expect.verdict);\n"
        + "});\n"
        + "var rows = coll.rows.map(function (r) {\n"
        + "  return { p: { id: r.id, name: String(r.id), value: r.value },\n"
        + "           cost: r.cost, mkt: r.mkt, surplus: r.surplus, keep: r.keep, verdict: '' };\n"
        + "});\n"
        + "resolveCollisions(rows, coll.rule);\n"
        + "rows.forEach(function (row) {\n"
        + "  if (!row.keep) return;\n"
        + "  var exp = coll.expectCost[row.p.id];\n"
        + "  if (row.cost !== exp) bad.push('collision ' + row.p.id + ' cost ' + row.cost + ' != ' + exp);\n"
        + "});\n"
        + "if (bad.length) { console.error(bad.join('\\n')); process.exit(1); }\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "keeper_parity.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True)
    assert res.returncode == 0, "keeper.js math diverged from Python engine:\n" + res.stderr
