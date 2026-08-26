"""Deep Dive reach verdict: remaining-board ADP + survival < 20%.

Historical ADP is a mean, so the best remaining player at pick 9 often has
ADP 11.0. That leftover-ADP gap is not a reach when the pick is remaining-board
BPA (or in a tight cluster) or the player would not last to the next pick.

Skips cleanly when Node isn't available.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CORE_JS = REPO / "static" / "draft_board_core.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def _run(expr: str):
    driver = (
        "global.self = global;\n"
        "const C = require(%s);\n"
        "process.stdout.write(JSON.stringify(%s));\n"
        % (json.dumps(str(CORE_JS)), expr)
    )
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_constants():
    out = _run("{cluster:C.ADP_REACH_CLUSTER,survive:C.ADP_REACH_SURVIVE}")
    assert out["cluster"] == 1.0
    assert out["survive"] == 20


def test_cook_at_9_is_remaining_bpa_not_a_reach():
    """James Cook ADP 11.0 at pick 9, next pick 16, Lamb clustered at 11.1."""
    out = _run(
        """(() => {
          const pool = [
            {id:'chase', adp:1.8},
            {id:'cook', adp:11.0},
            {id:'lamb', adp:11.1},
            {id:'later', adp:18.0},
          ];
          const taken = {chase:true};
          const best = C.bestRemainingAdp(pool, taken, p => p.adp);
          const cookDiff = 9 - 11.0;
          const sigma = Math.max(0.5, Math.min(10, 0.35 + 0.055 * 11));
          const survive = C.availabilityProbability({
            center:11, pick:16, sigma, draftType:'redraft', sf:false
          });
          return {
            best,
            cookBpa: C.isRemainingAdpBpa(11.0, best),
            lambBpa: C.isRemainingAdpBpa(11.1, best),
            laterBpa: C.isRemainingAdpBpa(18.0, best),
            survive,
            cookVerdict: C.adpDeltaVerdict({diff:cookDiff, isBpa:true, survivePct:survive}),
            cookBoard: C.adpBoardDelta({diff:cookDiff, isBpa:true, survivePct:survive}),
            lambVerdict: C.adpDeltaVerdict({diff:9-11.1, isBpa:true, survivePct:survive}),
          };
        })()"""
    )
    assert out["best"] == 11.0
    assert out["cookBpa"] is True
    assert out["lambBpa"] is True
    assert out["laterBpa"] is False
    assert out["survive"] < 20
    assert out["cookVerdict"] == {"label": "Fair", "cls": "fair"}
    assert out["cookBoard"] == 0
    assert out["lambVerdict"]["cls"] == "fair"


def test_true_reach_still_flags_when_skipping_better_adp():
    out = _run(
        """C.adpDeltaVerdict({diff: 9-40, isBpa:false, survivePct:90})"""
    )
    assert out == {"label": "Reach", "cls": "reach"}
    board = _run("C.adpBoardDelta({diff: 9-40, isBpa:false, survivePct:90})")
    assert board == 9 - 40


def test_survival_below_20_exempts_reach_at_20_does_not():
    low = _run("C.adpDeltaVerdict({diff:-8, isBpa:false, survivePct:10})")
    edge = _run("C.adpDeltaVerdict({diff:-8, isBpa:false, survivePct:19})")
    at = _run("C.adpDeltaVerdict({diff:-8, isBpa:false, survivePct:20})")
    high = _run("C.adpDeltaVerdict({diff:-8, isBpa:false, survivePct:50})")
    assert low["cls"] == "fair"
    assert edge["cls"] == "fair"
    assert at["cls"] == "reach"
    assert high["cls"] == "reach"
    assert _run("C.adpBoardDelta({diff:-8, isBpa:false, survivePct:19})") == 0
    assert _run("C.adpBoardDelta({diff:-8, isBpa:false, survivePct:20})") == -8


def test_bpa_exempts_even_when_raw_delta_is_a_big_reach():
    """Best remaining ADP 15.0 taken at pick 9: leftover-ADP, still BPA."""
    out = _run("C.adpDeltaVerdict({diff:9-15, isBpa:true, survivePct:40})")
    assert out["cls"] == "fair"
    assert _run("C.adpBoardDelta({diff:9-15, isBpa:true, survivePct:40})") == 0


def test_steal_and_value_unchanged_for_bpa():
    steal = _run("C.adpDeltaVerdict({diff:10, isBpa:true, survivePct:0})")
    value = _run("C.adpDeltaVerdict({diff:4, isBpa:false, survivePct:80})")
    fair = _run("C.adpDeltaVerdict({diff:-2, isBpa:false, survivePct:80})")
    assert steal == {"label": "Steal", "cls": "steal"}
    assert value == {"label": "Value", "cls": "value"}
    assert fair == {"label": "Fair", "cls": "fair"}


def test_cluster_boundary():
    assert _run("C.isRemainingAdpBpa(12.0, 11.0)") is True
    assert _run("C.isRemainingAdpBpa(12.1, 11.0)") is False
    assert _run("C.isRemainingAdpBpa(null, 11.0)") is False
