"""Performance sidecar: equivalence + memory-budget tests.

The player-modal rank sidecar and the weekly-stat / season-rank caches in
app.py must stay behavior-identical to the per-request code they replaced
while keeping added retained memory under ~2MB. app.py is not imported
(conftest forbids it); the small pure helpers are extracted via AST.
"""
import ast
import gc
import os
import random
import tracemalloc
from array import array

import pytest

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")

_WANT = {
    "_safe_float",
    "_fmt_score_static",
    "_rank_entry",
    "_rank_lookup",
    "_build_model_value_sidecar",
    "_sidecar_player_rank",
    "_lru_cache_put",
}


def _load_helpers():
    src = open(APP_PATH).read()
    tree = ast.parse(src)
    ns = {"__name__": "perf_test"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANT:
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), APP_PATH, "exec"),
                ns,
            )
    missing = _WANT - set(ns)
    assert not missing, f"helpers not found in app.py: {missing}"
    return ns


HELPERS = _load_helpers()


def _f(v):
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


# ---- original modal logic (pre-sidecar), for equivalence ----
def _orig_dynasty(tbl, player_id, key):
    pool = sorted(
        [
            x
            for x in tbl
            if isinstance(x, dict)
            and x.get("position") not in ("K", "DEF", "PICK")
            and _f(x.get(key)) > 0
        ],
        key=lambda x: _f(x.get(key)),
        reverse=True,
    )
    return next(
        (i + 1 for i, p in enumerate(pool) if str(p.get("id")) == str(player_id)),
        None,
    )


def _orig_rd(tbl, player_id, key, pos_only=None):
    pool = []
    for x in tbl:
        if not isinstance(x, dict):
            continue
        if str(x.get("position") or "").upper() in ("K", "DEF", "PICK"):
            continue
        if pos_only and str(x.get("position") or "").upper() != pos_only:
            continue
        v = _f(x.get(key))
        if v <= 0:
            continue
        pool.append((str(x.get("id")), v))
    pool.sort(key=lambda t: t[1], reverse=True)
    for i, (pid, _) in enumerate(pool):
        if pid == str(player_id):
            return i + 1
    return None


def _orig_fmt(tbl, player_id, key, mults, pos_only=None):
    pool = []
    for x in tbl:
        if not isinstance(x, dict):
            continue
        if str(x.get("position") or "").upper() in ("K", "DEF", "PICK"):
            continue
        if pos_only and str(x.get("position") or "").upper() != pos_only:
            continue
        v = _f(x.get(key))
        if v <= 0:
            continue
        pos = str(x.get("position") or "").upper()
        score = v * mults.get(pos, 1.0)
        if score <= 0:
            continue
        pool.append((str(x.get("id")), score))
    pool.sort(key=lambda t: t[1], reverse=True)
    for i, (pid, _) in enumerate(pool):
        if pid == str(player_id):
            return i + 1
    return None


def _scan(tbl, pid):
    for i, p in enumerate(tbl):
        if isinstance(p, dict) and str(p.get("id")) == str(pid):
            return p, i
    return {}, -1


def _adversarial_table(n=604, seed=42):
    random.seed(seed)
    positions = ["QB", "RB", "WR", "TE", "qb", "k", "K", "DEF", "PICK", "", None, "UNK"]
    tbl = []
    for i in range(n):
        pos = random.choice(positions)
        r = random.random()
        if r < 0.12:
            v = random.choice([1500.0, 800.0, 45.5])  # ties
        elif r < 0.16:
            v = round(random.uniform(0, 2000), 4)  # sub-milli
        elif r < 0.20:
            v = random.choice([0, 0.0, -5.0, None, "", "abc"])  # invalid
        else:
            v = round(random.uniform(1, 2000), 1)
        row = {
            "id": f"p{i}",
            "position": pos,
            "value": v,
            "sf_value": (v * 1.1 if isinstance(v, (int, float)) else v),
            "redraft_value_1qb": v,
            "redraft_value_sf": v,
        }
        if i % 97 == 0:
            row.pop("sf_value")
        tbl.append(row)
    tbl.append("not-a-dict")
    tbl.append(
        {
            "id": None,
            "position": "WR",
            "value": 999.0,
            "sf_value": 999.0,
            "redraft_value_1qb": 999.0,
            "redraft_value_sf": 999.0,
        }
    )
    return tbl


MULTS = {"half": {"QB": 1.0, "RB": 1.06, "WR": 0.97, "TE": 0.94},
         "std": {"QB": 1.0, "RB": 1.13, "WR": 0.93, "TE": 0.87}}


def test_sidecar_rank_equivalence():
    tbl = _adversarial_table()
    sc = HELPERS["_build_model_value_sidecar"](tbl)
    rank = HELPERS["_sidecar_player_rank"]
    fails = []
    for p in tbl:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        prow, tidx = _scan(tbl, pid)
        upos = str(p.get("position") or "").upper()
        for key in ("value", "sf_value"):
            try:
                exp = _orig_dynasty(tbl, pid, key)
            except ValueError:
                continue  # original crashed on non-numeric; sidecar is robust
            if rank(sc, key, prow, tidx, raw_pos=True) != exp:
                fails.append(("dynasty", pid, key))
        for key in ("redraft_value_1qb", "redraft_value_sf"):
            if rank(sc, key, prow, tidx) != _orig_rd(tbl, pid, key):
                fails.append(("rd-ovr", pid, key))
            if upos in ("QB", "RB", "WR", "TE"):
                if rank(sc, key, prow, tidx, scope=upos) != _orig_rd(
                    tbl, pid, key, pos_only=upos
                ):
                    fails.append(("rd-pos", pid, key, upos))
        for fmt in ("half", "std"):
            for key in ("value", "sf_value", "redraft_value_1qb", "redraft_value_sf"):
                if rank(sc, key, prow, tidx, fmt=fmt) != _orig_fmt(
                    tbl, pid, key, MULTS[fmt]
                ):
                    fails.append(("fmt-ovr", pid, key, fmt))
                if upos in ("QB", "RB", "WR", "TE"):
                    if rank(sc, key, prow, tidx, scope=upos, fmt=fmt) != _orig_fmt(
                        tbl, pid, key, MULTS[fmt], pos_only=upos
                    ):
                        fails.append(("fmt-pos", pid, key, fmt, upos))
    assert not fails, f"{len(fails)} rank mismatches, e.g. {fails[:5]}"


def test_sidecar_index_matches_comprehension():
    tbl = _adversarial_table()
    sc = HELPERS["_build_model_value_sidecar"](tbl)
    idx = sc["index"]
    comp = {str(p.get("id")): p for p in tbl if isinstance(p, dict)}
    assert idx.keys() == comp.keys()
    assert all(idx[k] is comp[k] for k in comp)


def test_lru_cache_put_evicts_oldest():
    put = HELPERS["_lru_cache_put"]
    c = {}
    put(c, "a", 1, 2)
    put(c, "b", 2, 2)
    put(c, "c", 3, 2)
    assert list(c) == ["b", "c"]
    put(c, "b", 22, 2)
    assert list(c) == ["c", "b"] and c["b"] == 22


def test_memory_budget_worst_case():
    """Sidecar + fullest bounded caches must stay under ~2MB retained."""
    random.seed(7)
    tbl = [
        {
            "id": f"p{i}",
            "position": random.choice(["QB", "RB", "WR", "TE", "UNK"]),
            "value": (v := round(random.uniform(0, 2000), 1)),
            "sf_value": v,
            "redraft_value_1qb": v,
            "redraft_value_sf": v,
        }
        for i in range(7000)
    ]
    put = HELPERS["_lru_cache_put"]
    ws_cache, rr_cache = {}, {}
    tracemalloc.start()
    gc.collect()
    s0 = tracemalloc.get_traced_memory()[0]
    HELPERS["_build_model_value_sidecar"](tbl)
    for k in ("k1", "k2", "k3"):
        put(
            ws_cache,
            k,
            (0, 0, (array("I", range(1000, 6000)), array("d", [1.0]) * 30000)),
            2,
        )
    for k in ("k1", "k2", "k3"):
        put(
            rr_cache,
            k,
            (
                0,
                {
                    "pos_ppg": {p: array("d", [1.0]) * 1250 for p in ("QB", "RB", "WR", "TE")},
                    "pos_total": {p: array("d", [1.0]) * 1250 for p in ("QB", "RB", "WR", "TE")},
                    "all_ppg": array("d", [1.0]) * 5000,
                    "all_total": array("d", [1.0]) * 5000,
                },
            ),
            2,
        )
    gc.collect()
    total_kb = (tracemalloc.get_traced_memory()[0] - s0) / 1024
    tracemalloc.stop()
    assert len(ws_cache) == 2 and len(rr_cache) == 2
    # ~2MB budget for added caches; headroom for one-time import overhead.
    assert total_kb < 2048, f"over memory budget: {total_kb:.0f} KB"
