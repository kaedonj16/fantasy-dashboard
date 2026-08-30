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
    flex = rc.get("FLEX", 0)
    sf = rc.get("SF", 0)
    bn = rc.get("BN", 0)
    bench_eff = min(bn, 8)
    rb_depth = math.ceil(bench_eff * 0.45)
    wr_depth = math.floor(bench_eff * 0.45)
    t = {
        "QB": (rc.get("QB", 0)) + sf + (1 if sf and bench_eff >= 5 else 0),
        "RB": (rc.get("RB", 0)) + flex + rb_depth,
        "WR": (rc.get("WR", 0)) + wr_depth,
        "TE": (rc.get("TE", 0)) + (1 if tep > 0 and bench_eff >= 5 else 0),
    }
    cap = {"QB": 4 if sf else max(1, rc.get("QB", 0)), "RB": 7, "WR": 7,
           "TE": max(3, rc.get("TE", 0)) if tep > 0 else max(1, rc.get("TE", 0))}
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


def test_roster_economics_respect_format_and_flex():
    script = (
        "global.self=global; global.BRPickScore=require(%s); const C=require(%s);"
        "const standard={QB:1,RB:2,WR:2,TE:1,FLEX:2,BN:7,K:1,DEF:1};"
        "const counts={QB:1,RB:2,WR:2,TE:1,K:0,DEF:0};"
        "process.stdout.write(JSON.stringify({targets:C.posTargets(standard,0),"
        "roles:['QB','RB','WR','TE'].map(p=>C.rosterRole(p,counts,standard,false)),"
        "utils:['QB','RB','WR','TE'].map(p=>C.rosterSlotUtility(p,counts,standard,{sf:false,tep:0,draftType:'redraft'})),"
        "specialUtil:[C.rosterSlotUtility('K',{K:0},standard,{sf:false}),"
        "C.rosterSlotUtility('K',{K:1},standard,{sf:false}),"
        "C.rosterSlotUtility('DEF',{DEF:1},standard,{sf:false})],"
        "dynDepth:[C.rosterSlotUtility('QB',{QB:2},standard,{sf:false,draftType:'startup'}),"
        "C.rosterSlotUtility('QB',{QB:3},standard,{sf:false,draftType:'startup'}),"
        "C.rosterSlotUtility('QB',{QB:3},{QB:1,SF:1},{sf:true,draftType:'startup'}),"
        "C.rosterSlotUtility('QB',{QB:4},{QB:1,SF:1},{sf:true,draftType:'startup'}),"
        "C.rosterSlotUtility('TE',{TE:2},standard,{sf:false,draftType:'startup',role:'bench2'}),"
        "C.rosterSlotUtility('TE',{TE:4},standard,{sf:false,draftType:'startup',role:'bench2'})],"
        "ob:C.remainingObligations(counts,standard,5,false),"
        "sf:C.rosterRole('QB',{QB:1},{QB:1,SF:1,RB:2,WR:2,TE:1,FLEX:1,BN:7},true),"
        "twoTe:C.rosterRole('TE',{TE:1},{QB:1,RB:2,WR:2,TE:2,FLEX:1,BN:7},false),"
        "flexUpgrade:C.candidateRosterRole('WR',.9,[{pos:'RB',quality:.8},{pos:'RB',quality:.7},{pos:'RB',quality:.6},"
        "{pos:'WR',quality:.8},{pos:'WR',quality:.7},{pos:'WR',quality:.2},{pos:'TE',quality:.7}],standard,false),"
        "scores:{qb2:C.decisionScore({base:97,utility:.30,bench:true,quality:.8,required:3,freePicks:1,waitLoss:4,recentPenalty:7}),"
        "flex:C.decisionScore({base:84,utility:.96,bench:false,waitLoss:18}),"
        "fallen:C.decisionScore({base:99,utility:.30,bench:true,quality:1,required:0,freePicks:4,waitLoss:18,exceptional:1}),"
        "qb3:C.decisionScore({base:92,utility:.18,bench:true,deepBench:true,quality:.8,required:2,freePicks:1})},"
        "waiting:[C.decisionScore({base:90,utility:1,waitPenalty:8}),C.decisionScore({base:87,utility:1,waitPenalty:0})],"
        "handcuff:[C.decisionScore({base:80,utility:1}),C.decisionScore({base:80,utility:1,handcuffBonus:5}),"
        "C.decisionScore({base:80,utility:1,handcuffBonus:50})],"
        "ceiling:[C.decisionScore({base:96,utility:1,waitLoss:30}),"
        "C.decisionScore({base:94,utility:1,waitLoss:30})],"
        "band:C.decisionBand([{id:'best',ds:90,weight:1},{id:'close',ds:87,weight:1},{id:'bad',ds:72,weight:9}],3,.8).map(x=>x.id),"
        "selected:C.selectDecisionCandidate([{id:'best',ds:90,weight:2},{id:'close',ds:87,weight:1}],3,.8,()=>0).id,"
        "availability:[C.availabilityProbability({adp:50,pick:40,sigma:5}),C.availabilityProbability({adp:50,pick:60,sigma:5})],"
        "limits:[C.positionRosterLimit('TE',{TE:1},{draftType:'redraft',tep:0}),"
        "C.positionRosterLimit('TE',{TE:1},{draftType:'redraft',tep:1}),"
        "C.positionRosterLimit('TE',{TE:1},{draftType:'startup',tep:0})]}));"
        % (json.dumps(str(PICK_JS)), json.dumps(str(CORE_JS)))
    )
    res = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["targets"]["QB"] == 1 and out["targets"]["TE"] == 1
    assert out["roles"] == ["bench1", "flex", "flex", "flex"]
    assert out["utils"][1] > out["utils"][0] and out["utils"][2] > out["utils"][3]
    assert out["specialUtil"] == [1, 0.06, 0.06]
    assert out["dynDepth"] == [0.55, 0.1, 0.55, 0.18, 0.44, 0.12]
    assert out["ob"]["missing"] == {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 1, "DEF": 1, "FLEX": 2}
    assert out["ob"]["freePicks"] == 1
    assert out["ob"]["lineupHoles"] == 2
    assert out["sf"] == "starter"
    assert out["twoTe"] == "starter"
    assert out["flexUpgrade"] == "flex"
    assert out["scores"]["flex"] > out["scores"]["qb2"]
    assert out["scores"]["fallen"] > out["scores"]["qb2"]
    assert out["scores"]["qb3"] < out["scores"]["qb2"]
    assert out["waiting"][1] > out["waiting"][0]
    # Handcuff insurance (now applied here, not in the pick-score kernel) nudges a
    # pick up, but is bounded so it can never leap a player past a real tier.
    assert out["handcuff"][1] > out["handcuff"][0]
    assert out["handcuff"][2] - out["handcuff"][0] <= 8
    assert out["ceiling"][0] < 99 and out["ceiling"][0] > out["ceiling"][1]
    assert out["band"] == ["best", "close"]
    assert out["selected"] == "best"
    assert out["availability"][0] >= 50 and out["availability"][1] < 10
    assert out["limits"] == [3, 4, 5]


def _run_need_cases(expression):
    script = (
        "global.self=global; global.BRPickScore=require(%s); const C=require(%s);"
        "process.stdout.write(JSON.stringify(%s));"
        % (json.dumps(str(PICK_JS)), json.dumps(str(CORE_JS)), expression)
    )
    res = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_position_need_utility_tracks_actual_lineup_shape():
    out = _run_need_cases("""(() => {
      const counts={QB:0,RB:0,WR:0,TE:0};
      const u=(p,rc,o={})=>C.positionNeedUtility(p,counts,rc,Object.assign({draftType:'redraft'},o));
      const twoWr={QB:1,RB:2,WR:2,TE:1,FLEX:1};
      const threeWr={QB:1,RB:2,WR:3,TE:1,FLEX:1};
      const twoFlex={QB:1,RB:2,WR:2,TE:1,FLEX:2};
      return {twoWr:u('WR',twoWr),threeWr:u('WR',threeWr),oneQb:u('QB',twoWr),
        superflex:u('QB',Object.assign({},twoWr,{SF:1}),{sf:true}),
        oneFlex:u('WR',twoWr),twoFlex:u('WR',twoFlex),
        teNormal:u('TE',twoFlex,{tep:0}),tePremium:u('TE',twoFlex,{tep:1})};
    })()""")

    assert out["threeWr"] > out["twoWr"] > out["oneQb"]
    assert out["superflex"] > out["oneQb"]
    assert out["twoFlex"] > out["oneFlex"]
    assert out["tePremium"] > out["teNormal"]


def test_live_decision_pressure_balances_wr_need_and_qb_timing():
    out = _run_need_cases("""(() => {
      const rc={QB:1,RB:2,WR:3,TE:1,FLEX:1,BN:7};
      const counts={QB:0,RB:3,WR:1,TE:0};
      const util=p=>C.positionNeedUtility(p,counts,rc,{draftType:'redraft'});
      const score=(base,p,waitLoss,waitPenalty)=>C.decisionScore({
        base:base,utility:util(p),waitLoss:waitLoss,waitPenalty:waitPenalty});
      return {utility:{qb:util('QB'),wr:util('WR'),te:util('TE'),rb:util('RB')},
        balanced:{dj:score(88,'WR',5,1),jayden:score(89,'QB',7,3),caleb:score(87,'QB',3,7),
          otherWr:score(84,'WR',3,2),te:score(85,'TE',4,3),rb4:score(87,'RB',4,3)},
        urgentJayden:{dj:score(88,'WR',4,2),jayden:score(92,'QB',18,0)}};
    })()""")

    # Two missing dedicated WR slots create more pressure than the lone QB slot;
    # a good WR stays near the top and ordinary RB4 depth does not jump starters.
    assert out["utility"]["wr"] > out["utility"]["qb"]
    assert out["balanced"]["dj"] > out["balanced"]["jayden"]
    assert out["balanced"]["dj"] > out["balanced"]["caleb"]
    assert out["balanced"]["dj"] > out["balanced"]["rb4"]
    assert out["balanced"]["jayden"] > out["balanced"]["caleb"]
    assert out["balanced"]["dj"] >= out["balanced"]["otherWr"]
    # A 1QB shelf cliff is real, but it must not leap two missing WR starters.
    assert out["urgentJayden"]["dj"] >= out["urgentJayden"]["jayden"]


def test_streamable_qb_te_need_does_not_leap_skill_depth():
    """Empty 1QB/1TE slots must not outrank remaining WR/RB depth on need alone.

    Reported shape: pick 8.04, 4 RB / 3 WR / 0 QB / 0 TE, a WR run on the board,
    and the rec list still led with QBs/TEs because an empty streamable starter
    scored a full 1.0 utility (~8 Decision Score points over WR4).
    """
    out = _run_need_cases("""(() => {
      const rc={QB:1,RB:2,WR:3,TE:1,FLEX:1,BN:7};
      const counts={QB:0,RB:4,WR:3,TE:0};
      const opts={draftType:'redraft'};
      const util=p=>C.positionNeedUtility(p,counts,rc,opts);
      const wls=(p,miss)=>C.waitLossScaleFor(p,miss,{sf:false,tep:0});
      const score=(base,p,waitLoss)=>C.decisionScore({
        base:base,utility:util(p),waitLoss:waitLoss,waitLossScale:wls(p,p==='WR'||p==='RB'?0:1)});
      const dyn=p=>C.positionNeedUtility(p,counts,rc,{draftType:'startup'});
      const sfQb=C.positionNeedUtility('QB',counts,Object.assign({},rc,{SF:1}),{draftType:'redraft',sf:true});
      const tepTe=C.positionNeedUtility('TE',counts,rc,{draftType:'redraft',tep:1});
      return {
        utility:{qb:util('QB'),wr:util('WR'),te:util('TE'),rb:util('RB'),
          dynQb:dyn('QB'),dynTe:dyn('TE'),sfQb:sfQb,tepTe:tepTe},
        scale:{qb:wls('QB',1),te:wls('TE',1),wrShort:wls('WR',1),wrDepth:wls('WR',0),sfQb:C.waitLossScaleFor('QB',1,{sf:true,tep:0})},
        recs:{wr:score(88,'WR',5),qb:score(88,'QB',5),te:score(88,'TE',5),
          qbReach:score(86,'QB',3),wrValue:score(88,'WR',4)},
        cliff:{wr:score(88,'WR',4),qb:score(92,'QB',18)}
      };
    })()""")

    # Streamable empty starters sit near WR4/RB4 depth, not at full 1.0.
    assert out["utility"]["qb"] < 0.85
    assert out["utility"]["te"] < 0.85
    assert abs(out["utility"]["qb"] - out["utility"]["wr"]) < 0.12
    # Superflex QB and TEP keep the full starter premium.
    assert out["utility"]["sfQb"] > out["utility"]["qb"]
    assert out["utility"]["tepTe"] > out["utility"]["te"]
    assert out["utility"]["dynQb"] >= out["utility"]["qb"]
    # 1QB/1TE scarcity is muted; a WR still missing a dedicated slot is not.
    assert out["scale"]["qb"] == out["scale"]["te"] == 0.4
    assert out["scale"]["wrShort"] == 0.6
    assert out["scale"]["sfQb"] == 0.6
    # At-value WR depth beats an ordinary empty-slot QB/TE; a true cliff can still win.
    assert out["recs"]["wr"] > out["recs"]["qb"]
    assert out["recs"]["wrValue"] > out["recs"]["qbReach"]
    assert out["recs"]["wr"] > out["recs"]["te"]
    assert out["cliff"]["qb"] > out["cliff"]["wr"]


def test_redraft_recs_prefer_starter_fills_over_luxury_bench():
    """Redraft Decision Score taxes luxury bench while RB/WR/FLEX holes remain.

    Pick Score weights stay untouched. Empty 1QB/1TE slots are not holes, so
    the streamable-QB-vs-WR-depth ranking is unchanged. Startup is unaffected.
    An extreme ADP fall can still buy the tax back.
    """
    out = _run_need_cases("""(() => {
      const rc={QB:1,RB:2,WR:3,TE:1,FLEX:1,BN:7};
      const holesOpen={QB:1,RB:2,WR:0,TE:1};   // 3 WR + FLEX still empty
      const holesFilled={QB:1,RB:3,WR:3,TE:1}; // lineup done except streamable 1QB
      const onlyQbTe={QB:0,RB:4,WR:3,TE:0};    // streamable empties only
      const ob=(counts,o={})=>C.remainingObligations(counts,rc,10,!!o.sf,o);
      const wrUtil=C.positionNeedUtility('WR',holesOpen,rc,{draftType:'redraft'});
      const rbUtil=C.positionNeedUtility('RB',holesOpen,rc,{draftType:'redraft',role:'bench1'});
      const starter=C.decisionScore({base:84,utility:wrUtil,bench:false,quality:.7,
        draftType:'redraft',lineupHoles:ob(holesOpen).lineupHoles});
      const luxury=C.decisionScore({base:94,utility:rbUtil,bench:true,quality:.75,
        draftType:'redraft',lineupHoles:ob(holesOpen).lineupHoles,required:4,freePicks:6});
      const luxuryNoHoles=C.decisionScore({base:94,utility:.82,bench:true,quality:.75,
        draftType:'redraft',lineupHoles:0,required:2,freePicks:6});
      const starterNoHoles=C.decisionScore({base:84,utility:1,bench:false,quality:.7,
        draftType:'redraft',lineupHoles:0});
      const startupLuxury=C.decisionScore({base:94,utility:rbUtil,bench:true,quality:.75,
        draftType:'startup',lineupHoles:ob(holesOpen).lineupHoles,required:4,freePicks:6});
      const startupStarter=C.decisionScore({base:84,utility:wrUtil,bench:false,quality:.7,
        draftType:'startup',lineupHoles:ob(holesOpen).lineupHoles});
      const fallen=C.decisionScore({base:99,utility:.82,bench:true,quality:1,exceptional:1,
        draftType:'redraft',lineupHoles:ob(holesOpen).lineupHoles,required:4,freePicks:6});
      const wrDepth=C.decisionScore({base:88,utility:C.positionNeedUtility('WR',onlyQbTe,rc,{draftType:'redraft'}),
        bench:true,draftType:'redraft',lineupHoles:ob(onlyQbTe).lineupHoles});
      const emptyQb=C.decisionScore({base:88,utility:C.positionNeedUtility('QB',onlyQbTe,rc,{draftType:'redraft'}),
        bench:false,draftType:'redraft',lineupHoles:ob(onlyQbTe).lineupHoles});
      const sfQbEmpty=ob({QB:0,RB:2,WR:3,TE:1},{sf:true,tep:0});
      const tepEmpty=ob({QB:1,RB:2,WR:3,TE:0},{tep:1});
      return {
        holes:{open:ob(holesOpen).lineupHoles, filled:ob(holesFilled).lineupHoles,
          onlyQbTe:ob(onlyQbTe).lineupHoles, sfQb:sfQbEmpty.lineupHoles, tep:tepEmpty.lineupHoles},
        recs:{starter,luxury,luxuryNoHoles,starterNoHoles,startupLuxury,startupStarter,fallen,wrDepth,emptyQb}
      };
    })()""")

    # 3 WR + 1 FLEX still empty. Extra RBs on a 2-RB roster do not fill those WR slots.
    assert out["holes"]["open"] == 4
    # Only K/DEF / streamable 1QB remain — not rec-steering holes.
    assert out["holes"]["filled"] == 0
    assert out["holes"]["onlyQbTe"] == 0
    # Superflex QB and premium TE empties DO count.
    assert out["holes"]["sfQb"] >= 1
    assert out["holes"]["tep"] >= 1
    # Mid-draft: an 84 starter beats a 94 luxury bench piece. Same scores without
    # holes (or in startup) keep the old BPA-can-win relationship.
    assert out["recs"]["starter"] > out["recs"]["luxury"]
    assert out["recs"]["luxuryNoHoles"] > out["recs"]["starterNoHoles"]
    assert out["recs"]["startupLuxury"] > out["recs"]["startupStarter"]
    # A true ADP fall can still overcome the tax.
    assert out["recs"]["fallen"] > out["recs"]["starter"]
    # Streamable empty 1QB still must not leap WR depth on need alone.
    assert out["recs"]["wrDepth"] > out["recs"]["emptyQb"]


def test_wait_loss_scale_damps_single_slot_scarcity():
    out = _run_need_cases("""(() => {
      const rc={QB:1,RB:2,WR:2,TE:1,FLEX:1};
      const counts={QB:0,RB:2,WR:0,TE:0};
      const util=p=>C.positionNeedUtility(p,counts,rc,{draftType:'redraft'});
      // Same cliff, only the scale differs: a single-slot need (0.6) must yield a
      // strictly lower urgency bonus than a full multi-slot need (1.0), even when
      // the raw bonus would otherwise saturate the headroom cap.
      const full=C.decisionScore({base:86,utility:1,waitLoss:30,waitLossScale:1});
      const damped=C.decisionScore({base:86,utility:1,waitLoss:30,waitLossScale:0.6});
      // Omitting waitLossScale must behave exactly like scale 1 (back-compat).
      const legacy=C.decisionScore({base:86,utility:1,waitLoss:30});
      // The pick-3.09 shape: an elite single-slot TE on a steep shelf cliff should
      // not leap a higher-need WR whose pool is deep (small cliff) on scarcity
      // alone. TE has one open dedicated slot (scale 0.6), WR has two (scale 1.0).
      const te=C.decisionScore({base:86,utility:util('TE'),waitLoss:30,waitLossScale:0.6});
      const wr=C.decisionScore({base:85,utility:util('WR'),waitLoss:4,waitLossScale:1});
      return {full:full, damped:damped, legacy:legacy, te:te, wr:wr};
    })()""")

    assert out["damped"] < out["full"]
    assert out["legacy"] == out["full"]
    assert out["wr"] > out["te"]


def test_autodraft_waits_on_single_slot_te_who_survives_to_the_turn():
    """1TE autodraft must not 1.35x-reach a TE whose ADP is still after the turn.

    The reported bug: autodraft took a TE 13 spots early in a 1TE league even
    though that player would still be there 8 picks later at the turn. CPU
    sampling is ADP-weighted so that reach almost never wins; autodraft is
    argmax of score, so the empty-starter 1.35x was enough to force it.
    """
    out = _run_need_cases("""(() => {
      const m = (o={}) => C.autoDraftNeedMultiplier(Object.assign({
        pos:'TE', have:0, target:1, starterSlots:1, adp:37, pickNo:24,
        teams:12, nextPick:32, surviveProb:58, tep:0, sf:false, qbStarters:1
      }, o));
      // Same decision scores the live layer would hand autodraft: TE slightly
      // ahead on need, WR is the at-value alternative. After the wait discount
      // the WR must win; with the old uncapped 1.35x the TE would win.
      const teWait = 90 * m();
      const wrNow = 84 * m({pos:'WR', starterSlots:2, have:1, target:4, adp:25, surviveProb:20});
      const teAtValue = m({adp:24, nextPick:32, surviveProb:30});
      const teSmallReach = m({adp:26, nextPick:32, surviveProb:35});
      const teRun = m({surviveProb:20});
      return {
        wait: m(),
        wrBoost: m({pos:'WR', starterSlots:2, have:1, target:4, adp:25, surviveProb:20}),
        teAtValue: teAtValue,
        teSmallReach: teSmallReach,
        teRun: teRun,
        backupTe: m({have:1, adp:40}),
        overfillWr: m({pos:'WR', have:5, target:4, starterSlots:2, adp:50}),
        tepNoWait: m({tep:1, surviveProb:58}),
        teWaitScore: teWait,
        wrNowScore: wrNow,
      };
    })()""")

    assert out["wait"] == 0.35
    assert out["wrBoost"] == 1.35
    assert out["teAtValue"] == 1.35
    assert out["teSmallReach"] == 1.35
    assert out["teRun"] == 1
    assert out["backupTe"] == 0
    assert out["overfillWr"] == 0.4
    # TEP is allowed to chase TE earlier, but a 13-spot reach still gets no 1.35x.
    assert out["tepNoWait"] == 1
    assert out["wrNowScore"] > out["teWaitScore"]


def test_special_teams_fill_order_follows_team_plan_not_kicker_first():
    out = _run_need_cases("""(() => {
      const f = C.specialTeamsFillPos;
      return {
        none: f(0, 0, {prefer:'K'}),
        onlyK: f(1, 0, {prefer:'DEF'}),
        onlyDef: f(0, 1, {prefer:'K'}),
        preferDef: f(1, 1, {prefer:'DEF', order:0.1, flip:false}),
        preferK: f(1, 1, {prefer:'K', order:0.9, flip:false}),
        mixHigh: f(1, 1, {prefer:'mix', order:0.7, flip:false}),
        mixLow: f(1, 1, {prefer:'mix', order:0.2, flip:false}),
        flippedDef: f(1, 1, {prefer:'DEF', order:0.1, flip:true}),
      };
    })()""")

    assert out["none"] is None
    assert out["onlyK"] == "K"
    assert out["onlyDef"] == "DEF"
    assert out["preferDef"] == "DEF"
    assert out["preferK"] == "K"
    assert out["mixHigh"] == "DEF"
    assert out["mixLow"] == "K"
    assert out["flippedDef"] == "K"


def test_source_adp_reads_rankings_column_not_sleeper_overlay():
    """Consensus ADP is the per-source map rankings shows, not avg_pick."""
    out = _run_need_cases("""(() => {
      const p = {
        avg_pick: 10,
        sf_avg_pick: 8,
        redraft_avg_pick: 20,
        adp_by_source: {
          consensus: { avg_pick: 12.4, sf_avg_pick: 9.7, redraft_avg_pick: 18.15 },
          sleeper: { avg_pick: 9.1, sf_avg_pick: 7.2, redraft_avg_pick: 21 },
        },
      };
      return {
        cons: C.consensusAdpOf(p, 'dynasty', false),
        consSf: C.consensusAdpOf(p, 'dynasty', true),
        consRedraft: C.sourceAdpOf(p, 'consensus', 'redraft', false),
        sleeper: C.sourceAdpOf(p, 'sleeper', 'dynasty', false),
        fallback: C.consensusAdpOf({ avg_pick: 7 }, 'dynasty', false),
        missing: C.sourceAdpOf(p, 'yahoo', 'dynasty', false),
      };
    })()""")

    assert out["cons"] == 12.4
    assert out["consSf"] == 9.7
    assert out["consRedraft"] == 18.15
    assert out["sleeper"] == 9.1
    assert out["fallback"] == 7
    assert out["missing"] is None


def test_scoring_proj_ppg_reflects_half_ppr_and_six_point_tds():
    """Draft-room scoring settings rescale the displayed / ranked proj PPG."""
    out = _run_need_cases("""(() => {
      const wr = { proj_ppg: 20, proj_pts: 340,
        proj_ppg_by: { ppr: 18, half_ppr: 15, std: 12, '6pt_ppr': 18, '6pt_half': 15 } };
      const qb = { proj_ppg: 22, proj_pts: 374,
        proj_ppg_by: { ppr: 20, half_ppr: 20, '6pt_ppr': 24, '6pt_half': 24 } };
      const bare = { proj_ppg: 16 };
      const ppr = { ppr: 1, tep: 0, passTd: 4 };
      const half = { ppr: 0.5, tep: 0, passTd: 4 };
      const six = { ppr: 1, tep: 0, passTd: 6 };
      const halfSix = { ppr: 0.5, tep: 0, passTd: 6 };
      return {
        keys: { ppr: C.pickProjVariant(ppr), half: C.pickProjVariant(half),
                six: C.pickProjVariant(six), halfSix: C.pickProjVariant(halfSix) },
        wrPpr: C.scoringProjPpg(wr, ppr),
        wrHalf: C.scoringProjPpg(wr, half),
        wrSix: C.scoringProjPpg(wr, six),
        wrPtsHalf: C.scoringProjPts(wr, half),
        qbPpr: C.scoringProjPpg(qb, ppr),
        qbSix: C.scoringProjPpg(qb, six),
        qbPtsSix: C.scoringProjPts(qb, six),
        bareHalf: C.scoringProjPpg(bare, half),
        ppgOfHalf: C.ppgOf(wr, half),
        ppgOfDefault: C.ppgOf(wr),
      };
    })()""")

    assert out["keys"] == {"ppr": "ppr", "half": "half_ppr", "six": "6pt_ppr", "halfSix": "6pt_half"}
    # Full PPR keeps the canonical FantasyPros/Sleeper PPR number.
    assert out["wrPpr"] == 20
    assert out["qbPpr"] == 22
    assert out["ppgOfDefault"] == 20
    # Half PPR scales the canonical number by Sleeper half/ppr (20 * 15/18 = 16.7).
    assert out["wrHalf"] == 16.7
    assert out["ppgOfHalf"] == 16.7
    assert out["wrPtsHalf"] == 283.9  # 340 * 16.7/20
    # 6-pt passing TDs lift QBs (22 * 24/20 = 26.4) and leave WRs unchanged.
    assert out["qbSix"] == 26.4
    assert out["qbPtsSix"] == 448.8  # 374 * 26.4/22
    assert out["wrSix"] == 20
    # No variant map -> keep the stored PPR projection.
    assert out["bareHalf"] == 16


def test_future_pick_decision_score_drops_zero_survival_elites():
    """Pick-9 recs: 1.01 talent at 0% must not outrank a likely survivor."""
    driver = (
        "global.self = global;\n"
        "const C = require(%s);\n"
        "process.stdout.write(JSON.stringify({"
        "floor:C.REC_FUTURE_SURVIVE_FLOOR,"
        "gibbs:C.futurePickDecisionScore(95,0),"
        "likely:C.futurePickDecisionScore(70,40),"
        "nullSurvive:C.futurePickDecisionScore(80,null)"
        "}));\n" % json.dumps(str(CORE_JS))
    )
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["floor"] == 0.08
    assert out["gibbs"] == pytest.approx(95 * 0.08)
    assert out["likely"] == pytest.approx(70 * (0.08 + 0.92 * 0.40))
    assert out["gibbs"] < out["likely"]
    assert out["nullSurvive"] == 80


def test_late_round_upside_is_smooth_and_requires_a_role_path():
    out = _run_need_cases("""(() => ({
      early:C.lateRoundUpsideBonus({round:2,totalRounds:16,path:1,aboveReplacement:.7,tierQuality:.7,ppgQuality:.7,functionalUtility:.8,rosterNeedPath:1,youngWithPath:true}),
      latePath:C.lateRoundUpsideBonus({round:15,totalRounds:16,path:.9,aboveReplacement:.6,tierQuality:.6,ppgQuality:.6,functionalUtility:.8,rosterNeedPath:1,youngWithPath:true}),
      lateNoPath:C.lateRoundUpsideBonus({round:15,totalRounds:16,path:0,aboveReplacement:.1,tierQuality:.1,ppgQuality:.1,functionalUtility:.8,rosterNeedPath:.3,youngWithPath:false})
    }))()""")
    assert out["early"] == 0
    assert out["latePath"] > 0
    assert out["lateNoPath"] < 0


def test_opportunity_cost_requires_market_survival_and_material_gap():
    out = _run_need_cases("""(() => ({
      tiny:C.opportunityCostVerdict({selectedScore:84,bestAlternativeScore:85,outsideMarketRange:true,isBpa:false,survivePct:70}),
      large:C.opportunityCostVerdict({selectedScore:71,bestAlternativeScore:91,outsideMarketRange:true,isBpa:false,survivePct:70}),
      adpOnly:C.opportunityCostVerdict({selectedScore:84,bestAlternativeScore:85,outsideMarketRange:true,isBpa:false,survivePct:70}),
      wontLast:C.opportunityCostVerdict({selectedScore:71,bestAlternativeScore:91,outsideMarketRange:true,isBpa:false,survivePct:10})
    }))()""")
    assert out["tiny"]["severity"] == "none"
    assert out["tiny"]["significantReach"] is False
    assert out["large"]["severity"] == "severe"
    assert out["large"]["significantReach"] is True
    assert out["adpOnly"]["significantReach"] is False
    assert out["wontLast"]["significantReach"] is False


def test_bye_severity_uses_starter_impact_not_raw_count():
    out = _run_need_cases("""(() => ({
      bench:C.byeWeekSeverity([
        {bye:7,pos:'WR',role:'fringe',quality:.3},{bye:7,pos:'RB',role:'fringe',quality:.3},{bye:7,pos:'WR',role:'fringe',quality:.3}],{}),
      starters:C.byeWeekSeverity([
        {bye:7,pos:'RB',role:'starter',quality:1},{bye:7,pos:'RB',role:'starter',quality:.9},{bye:7,pos:'WR',role:'starter',quality:.9}],{}),
      covered:C.byeWeekSeverity([
        {id:'s1',bye:7,pos:'RB',role:'starter',quality:1},
        {id:'c1',bye:10,pos:'RB',role:'primary',quality:.8}],{})
    }))()""")
    assert out["bench"][0]["level"] == "none"
    assert out["starters"][0]["level"] in {"meaningful", "severe"}
    assert out["covered"][0]["players"][0]["coverQuality"] == pytest.approx(0.8)


def test_late_round_path_ignores_ppg_and_requires_a_role():
    out = _run_need_cases("""(() => ({
      ppgOnly:C.lateRoundPathEvidence({}),
      breakout:C.lateRoundPathEvidence({breakoutScore:80}),
      handcuff:C.lateRoundPathEvidence({handcuff:true}),
      none:C.lateRoundPathEvidence({projectedRole:null})
    }))()""")
    assert out["ppgOnly"] == 0
    assert out["none"] == 0
    assert out["breakout"] == pytest.approx(0.8)
    assert out["handcuff"] == pytest.approx(0.75)


def test_historical_alternatives_rank_by_decision_score():
    out = _run_need_cases("""(() => {
      const rows=[
        {id:'a',decisionScore:71,absolutePickScore:80,player:{name:'A'}},
        {id:'b',decisionScore:91,absolutePickScore:82,player:{name:'B'}}
      ];
      return C.summarizeHistoricalAlternatives(rows,'a');
    })()""")
    assert out["selectedScore"] == 71
    assert out["bestAlternativeScore"] == 91
    assert out["bestAlternative"]["id"] == "b"
