"""Parity guard for the shared draft-board kernels (static/draft_board_core.js).

The Draft Room and the Draft Cheat Sheet now derive replacement level, the PPG
production scale and roster-need targets from ONE module. This test pins that
module's math to a Python reference so an edit that changes the shared kernel
(and would silently move both surfaces) fails CI. It is the drift guard that
makes "the cheat sheet and the draft room agree" a fact rather than a promise.

Skips cleanly when Node isn't available.
"""
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PICK_JS = REPO / "static" / "pick_score.js"
CORE_JS = REPO / "static" / "draft_board_core.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def _js_round(x: float) -> int:
    """Math.round: round-half-up toward +inf (not Python's banker's rounding)."""
    return math.floor(x + 0.5)


# ── Python reference implementations of the shared kernels ────────────────────

def _ref_replacement(pool, starters, teams):
    by_pos = {"QB": [], "RB": [], "WR": [], "TE": []}
    for p in pool:
        pos = (p.get("position") or "").upper()
        if pos in by_pos:
            by_pos[pos].append(p.get("value") or 0)
    r = {}
    for pos, arr in by_pos.items():
        arr = sorted(arr, reverse=True)
        if not arr:
            r[pos] = 0
            continue
        idx = _js_round(teams * (starters.get(pos) or 1)) - 1
        idx = max(0, min(idx, len(arr) - 1))
        r[pos] = arr[idx]
    return r


def _ppg_of(p):
    if p.get("proj_ppg") is not None:
        return float(p["proj_ppg"])
    if p.get("ppg") is not None:
        return float(p["ppg"])
    return None


def _ref_ppg_scale(pool, starters, teams):
    by_pos = {"QB": [], "RB": [], "WR": [], "TE": []}
    for p in pool:
        pos = (p.get("position") or "").upper()
        v = _ppg_of(p)
        if pos in by_pos and v is not None:
            by_pos[pos].append(v)
    out = {}
    for pos, arr in by_pos.items():
        if not arr:
            continue
        arr = sorted(arr, reverse=True)
        top_n = max(1, min(3, len(arr)))
        elite = sum(arr[:top_n]) / top_n
        idx = _js_round(teams * (starters.get(pos) or 1)) - 1
        idx = max(0, min(idx, len(arr) - 1))
        out[pos] = {"repl": arr[idx], "elite": elite}
    return out


def _clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _ref_ppg_norm(p, scale):
    pos = (p.get("position") or "").upper()
    v = _ppg_of(p)
    sc = scale.get(pos)
    if v is None or not sc:
        return None
    span = sc["elite"] - sc["repl"]
    if span <= 0:
        return _clamp01(v / max(sc["elite"], 1))
    return _clamp01((v - sc["repl"]) / span)


def _ref_targets(rc, tep):
    flex = rc.get("FLEX", 0); sf = rc.get("SF", 0); bn = rc.get("BN", 0)
    bench_eff = min(bn, 7)
    t = {
        "QB": (rc.get("QB", 0)) + sf + _js_round(bench_eff * 0.10),
        "RB": (rc.get("RB", 0)) + flex + _js_round(bench_eff * 0.35),
        "WR": (rc.get("WR", 0)) + _js_round(bench_eff * 0.40),
        "TE": (rc.get("TE", 0)) + (_js_round(bench_eff * 0.15) if tep > 0 else 0),
    }
    if tep > 0:
        t["TE"] += 1
    cap = {"QB": 3 if sf else 2, "RB": 6, "WR": 6, "TE": 3 if tep > 0 else 2}
    for k, c in cap.items():
        if t[k] > c:
            t[k] = c
    if rc.get("K"):
        t["K"] = rc["K"]
    if rc.get("DEF"):
        t["DEF"] = rc["DEF"]
    return t


# ── build random cases and run the JS core over them ──────────────────────────

def _build():
    rng = random.Random(2026)
    positions = ["QB", "RB", "WR", "TE"]
    repl_cases, ppg_cases, target_cases = [], [], []
    for _ in range(120):
        teams = rng.choice([8, 10, 12, 14])
        pool = [{"position": rng.choice(positions),
                 "value": round(rng.uniform(0, 9000), 1),
                 "proj_ppg": round(rng.uniform(0, 22), 1)}
                for _ in range(rng.randint(20, 80))]
        starters = {"QB": round(rng.uniform(1, 2), 2), "RB": round(rng.uniform(2, 3), 2),
                    "WR": round(rng.uniform(2, 3), 2), "TE": 1}
        repl_cases.append({"pool": pool, "starters": starters, "teams": teams})
        ppg_cases.append({"pool": pool, "starters": starters, "teams": teams})
    for _ in range(60):
        target_cases.append({"rc": {k: rng.randint(0, 4) for k in ("QB", "RB", "WR", "TE", "FLEX", "SF", "BN")},
                             "tep": rng.choice([0, 0.5, 1])})
    return {"repl": repl_cases, "ppg": ppg_cases, "targets": target_cases}


def _run_js(inp):
    # Input goes through a temp file, not argv (the pools are too large for
    # `node -e`'s argument length limit).
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    try:
        json.dump(inp, tmp)
        tmp.close()
        driver = (
            "global.self = global;\n"
            "const fs = require('fs');\n"
            "const PS = require(%s); global.BRPickScore = PS;\n"
            "const C = require(%s);\n"
            "const inp = JSON.parse(fs.readFileSync(%s, 'utf8'));\n"
            "const out = {\n"
            "  repl: inp.repl.map(t => C.computeReplacement(t.pool, p => p.value || 0, t.starters, t.teams)),\n"
            "  ppg: inp.ppg.map(t => { const s = C.computePpgScale(t.pool, C.ppgOf, t.starters, t.teams);\n"
            "        return t.pool.map(p => C.ppgNorm(p, s, C.ppgOf)); }),\n"
            "  targets: inp.targets.map(t => C.posTargets(t.rc, t.tep)),\n"
            "};\n"
            "process.stdout.write(JSON.stringify(out));\n"
            % (json.dumps(str(PICK_JS)), json.dumps(str(CORE_JS)), json.dumps(tmp.name))
        )
        res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, res.stderr
        return json.loads(res.stdout)
    finally:
        os.unlink(tmp.name)


def test_shared_kernels_match_reference():
    inp = _build()
    out = _run_js(inp)

    for case, js in zip(inp["repl"], out["repl"]):
        py = _ref_replacement(case["pool"], case["starters"], case["teams"])
        assert {k: float(v) for k, v in js.items()} == {k: float(v) for k, v in py.items()}, \
            f"computeReplacement mismatch: js={js} py={py}"

    for case, js_rows in zip(inp["ppg"], out["ppg"]):
        scale = _ref_ppg_scale(case["pool"], case["starters"], case["teams"])
        for p, jv in zip(case["pool"], js_rows):
            pv = _ref_ppg_norm(p, scale)
            if jv is None or pv is None:
                assert jv == pv, f"ppgNorm None mismatch: js={jv} py={pv}"
            else:
                assert abs(float(jv) - float(pv)) < 1e-9, f"ppgNorm mismatch: js={jv} py={pv}"

    for case, js in zip(inp["targets"], out["targets"]):
        py = _ref_targets(case["rc"], case["tep"])
        assert {k: int(v) for k, v in js.items()} == {k: int(v) for k, v in py.items()}, \
            f"posTargets mismatch for {case['rc']} tep={case['tep']}: js={js} py={py}"
