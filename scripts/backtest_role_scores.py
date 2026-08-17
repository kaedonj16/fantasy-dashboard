#!/usr/bin/env python3
"""
Backtest the snap-free role score (current formula) against the previous
snap-based one, on real historical usage — so you can see exactly how every
player's role score moves before recomputing the live metrics.

  OLD = the previous snap-based v2 index + old anchors. Because this data set
        has no snap share, OLD reproduces the *deflated* scores currently in the
        DB (elite WRs ~45, a QB1 ~31).
  NEW = the current snap-free index + recalibrated anchors (what the next
        advanced-metrics recompute will produce).

Reads cache/player_history/usage_rows_<season>.json (pure JSON — no DB, no
Flask), so it is safe to run anywhere the cache is present.

Usage:
    python scripts/backtest_role_scores.py                  # latest season, all positions
    python scripts/backtest_role_scores.py --season 2024
    python scripts/backtest_role_scores.py --all            # every cached season
    python scripts/backtest_role_scores.py --pos WR --limit 40
    python scripts/backtest_role_scores.py --movers         # biggest new-vs-old swings
    python scripts/backtest_role_scores.py --html out.html  # write a visual report
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_GLOB = os.path.join(ROOT, "cache", "player_history", "usage_rows_*.json")
MODULE_SRC = os.path.join(ROOT, "data_building", "advanced_metrics.py")
POSITIONS = ("QB", "RB", "WR", "TE")
MIN_GAMES = 8  # qualify a player for the report


# ── Load the *current* (snap-free) scoring symbols ────────────────────────────
def load_new_symbols() -> Dict[str, Any]:
    """Prefer the real module; fall back to extracting the pure functions from
    source so the backtest runs without Flask/DB installed."""
    try:
        from data_building import advanced_metrics as m  # type: ignore
        return {
            "role_opportunity_index": m.role_opportunity_index,
            "build_team_opportunity_context": m.build_team_opportunity_context,
            "anchor": m._ROLE_ELITE_ANCHOR,
            "full_games": m._ROLE_FULL_SAMPLE_GAMES,
            "clip": m._clip,
        }
    except Exception:
        pass
    # Fallback: exec just the pure helpers, the index, the team-context builder
    # and the anchor/const assignments out of the source file.
    src = open(MODULE_SRC).read()
    tree = ast.parse(src)
    ns: Dict[str, Any] = {"Optional": Optional, "Dict": Dict, "List": List, "Any": Any}
    want_fns = {"_clip", "_safe", "_norm", "_share",
                "role_opportunity_index", "build_team_opportunity_context"}
    want_consts = {"_ROLE_ELITE_ANCHOR", "_ROLE_FULL_SAMPLE_GAMES"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_fns:
            exec(ast.get_source_segment(src, node), ns)
        elif isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) in want_consts:
            exec(ast.get_source_segment(src, node), ns)
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in want_consts:
            exec(ast.get_source_segment(src, node), ns)
    return {
        "role_opportunity_index": ns["role_opportunity_index"],
        "build_team_opportunity_context": ns["build_team_opportunity_context"],
        "anchor": ns["_ROLE_ELITE_ANCHOR"],
        "full_games": ns["_ROLE_FULL_SAMPLE_GAMES"],
        "clip": ns["_clip"],
    }


# ── OLD (snap-based) index, replicated so we can compare against it ────────────
OLD_ANCHOR = {"WR": 0.46, "TE": 0.35, "RB": 0.74, "QB": 0.77}


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _safe(v):
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _norm(v, lo, hi):
    return _clip((v - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 0.0


def _share(part, whole):
    return _clip(part / whole, 0.0, 1.0) if whole > 0 else 0.0


def old_role_index(usage, position, team_ctx, rz_available=True):
    if _safe(usage.get("games")) <= 0:
        return None
    snap = _clip(_safe(usage.get("avg_off_snap_pct")), 0.0, 1.0)  # ~0 in this data
    tshare = _safe(usage.get("target_share"))
    if tshare <= 0:
        tshare = _share(_safe(usage.get("avg_targets")), _safe(team_ctx.get("targets")))
    tshare = _clip(tshare, 0.0, 1.0)
    rz_tgt = _share(_safe(usage.get("rec_rz_tgt_pg")), _safe(team_ctx.get("rz_tgt")))
    rz_rush = _share(_safe(usage.get("rush_rz_att_pg")), _safe(team_ctx.get("rz_rush")))
    if position == "WR":
        comps = [(tshare, 0.50, False), (snap, 0.28, False), (rz_tgt, 0.22, True)]
    elif position == "TE":
        comps = [(tshare, 0.55, False), (rz_tgt, 0.27, True), (snap, 0.18, False)]
    elif position == "RB":
        rshare = _share(_safe(usage.get("avg_carries")), _safe(team_ctx.get("carries")))
        core = _clip(rshare + 1.7 * tshare, 0.0, 1.0)
        comps = [(core, 0.46, False), (rz_rush, 0.20, True), (snap, 0.18, False), (rz_tgt, 0.16, True)]
    elif position == "QB":
        pass_vol = _norm(_safe(usage.get("avg_pass_att")), 18, 42)
        rush_vol = _norm(_safe(usage.get("avg_carries")), 0, 9)
        comps = [(pass_vol, 0.47, False), (snap, 0.33, False), (rush_vol, 0.20, False)]
    else:
        return None
    if rz_available:
        idx = sum(v * w for v, w, _ in comps)
    else:
        kept = [(v, w) for v, w, rz in comps if not rz]
        wsum = sum(w for _, w in kept) or 1.0
        idx = sum(v * (w / wsum) for v, w in kept)
    return _clip(idx, 0.0, 1.0)


# ── Scoring + season loading ──────────────────────────────────────────────────
def score_from_index(idx, position, games, anchor, full_games, clip):
    if idx is None or position not in anchor:
        return None
    conf = clip(games / full_games, 0.0, 1.0)
    return round(clip(idx / anchor[position], 0.0, 1.0) * 100.0 * conf, 1)


def season_path(season: int) -> str:
    return os.path.join(ROOT, "cache", "player_history", f"usage_rows_{season}.json")


def available_seasons() -> List[int]:
    out = []
    for p in glob.glob(CACHE_GLOB):
        base = os.path.basename(p)
        try:
            out.append(int(base.replace("usage_rows_", "").replace(".json", "")))
        except ValueError:
            continue
    return sorted(out)


def rz_available_for(usage_table) -> bool:
    return any(
        (_safe((p.get("usage") or {}).get("rec_rz_tgt_pg")) > 0
         or _safe((p.get("usage") or {}).get("rush_rz_att_pg")) > 0)
        for p in usage_table
    )


def compute_season(season: int, sym) -> List[Dict[str, Any]]:
    data = json.load(open(season_path(season)))
    usage_table = [{"id": d.get("id"), "team": d.get("team"),
                    "position": d.get("position"), "usage": d.get("usage") or {}}
                   for d in data]
    ctx_new = sym["build_team_opportunity_context"](usage_table)
    ctx_old = sym["build_team_opportunity_context"](usage_table)  # same shape
    rz_ok = rz_available_for(usage_table)
    rows = []
    for d in data:
        u = d.get("usage") or {}
        pos = d.get("position")
        g = _safe(u.get("games"))
        if pos not in POSITIONS or g < MIN_GAMES:
            continue
        tc = ctx_new.get(d.get("team"), {})
        new_idx = sym["role_opportunity_index"](u, pos, tc, rz_ok)
        old_idx = old_role_index(u, pos, ctx_old.get(d.get("team"), {}), rz_ok)
        new = score_from_index(new_idx, pos, g, sym["anchor"], sym["full_games"], sym["clip"])
        old = score_from_index(old_idx, pos, g, OLD_ANCHOR, sym["full_games"], sym["clip"])
        rows.append({"name": d.get("name") or d.get("id"), "pos": pos,
                     "games": int(g), "old": old, "new": new,
                     "delta": (None if old is None or new is None else round(new - old, 1))})
    return rows


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_report(season: int, rows, pos_filter, limit, movers):
    print(f"\n########## SEASON {season}  (old snap-based  →  new snap-free) ##########")
    if movers:
        r = [x for x in rows if x["delta"] is not None and (not pos_filter or x["pos"] == pos_filter)]
        r.sort(key=lambda x: x["delta"])
        print(f"\n=== BIGGEST FALLERS ===\n{'pos':4}{'player':24}{'old':>7}{'new':>7}{'Δ':>8}")
        for x in r[:limit]:
            print(f"{x['pos']:4}{x['name'][:24]:24}{x['old']:7.1f}{x['new']:7.1f}{x['delta']:+8.1f}")
        print(f"\n=== BIGGEST RISERS ===\n{'pos':4}{'player':24}{'old':>7}{'new':>7}{'Δ':>8}")
        for x in reversed(r[-limit:]):
            print(f"{x['pos']:4}{x['name'][:24]:24}{x['old']:7.1f}{x['new']:7.1f}{x['delta']:+8.1f}")
        return
    for pos in ([pos_filter] if pos_filter else POSITIONS):
        pr = [x for x in rows if x["pos"] == pos]
        pr.sort(key=lambda x: (x["new"] if x["new"] is not None else -1), reverse=True)
        n100_old = sum(1 for x in pr if (x["old"] or 0) >= 99.5)
        n100_new = sum(1 for x in pr if (x["new"] or 0) >= 99.5)
        print(f"\n=== {pos}  (top {limit} by new; {len(pr)} qualified; "
              f"@100 old={n100_old} new={n100_new}) ===")
        print(f"{'player':24}{'old':>7}{'new':>7}{'Δ':>8}")
        for x in pr[:limit]:
            o = "   -  " if x["old"] is None else f"{x['old']:6.1f}"
            nw = "   -  " if x["new"] is None else f"{x['new']:6.1f}"
            dd = "   -  " if x["delta"] is None else f"{x['delta']:+6.1f}"
            print(f"{x['name'][:24]:24}{o:>7}{nw:>7}{dd:>8}")


def write_html(path, season_rows_map, limit):
    def bar(v, color):
        v = max(0, min(100, v or 0))
        return (f'<div style="background:#e5e7eb;border-radius:99px;height:7px;overflow:hidden;min-width:70px;">'
                f'<div style="width:{v}%;height:100%;background:{color};border-radius:99px;"></div></div>')
    css = """<style>body{font-family:-apple-system,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;}
    h1{font-size:20px;} h2{font-size:15px;margin-top:28px;color:#93c5fd;text-transform:uppercase;letter-spacing:.05em;}
    table{border-collapse:collapse;width:100%;max-width:760px;margin-top:8px;}
    td,th{padding:6px 10px;text-align:left;font-size:13px;border-bottom:1px solid #1e293b;}
    th{color:#94a3b8;font-size:11px;text-transform:uppercase;} .num{text-align:right;font-variant-numeric:tabular-nums;}
    .up{color:#22c55e;} .dn{color:#ef4444;} .flat{color:#94a3b8;}</style>"""
    parts = [css, "<h1>Role Score Backtest — old (snap-based) → new (snap-free)</h1>"]
    for season, rows in season_rows_map.items():
        parts.append(f"<h1 style='font-size:16px;margin-top:32px;'>Season {season}</h1>")
        for pos in POSITIONS:
            pr = [x for x in rows if x["pos"] == pos]
            pr.sort(key=lambda x: (x["new"] if x["new"] is not None else -1), reverse=True)
            parts.append(f"<h2>{pos}</h2><table><tr><th>Player</th><th class='num'>Old</th>"
                         f"<th class='num'>New</th><th class='num'>Δ</th><th>New (bar)</th></tr>")
            for x in pr[:limit]:
                d = x["delta"]
                cls = "flat" if not d else ("up" if d > 0 else "dn")
                ds = "-" if d is None else f"{d:+.0f}"
                parts.append(
                    f"<tr><td>{x['name'][:26]}</td>"
                    f"<td class='num'>{'-' if x['old'] is None else round(x['old'])}</td>"
                    f"<td class='num'>{'-' if x['new'] is None else round(x['new'])}</td>"
                    f"<td class='num {cls}'>{ds}</td>"
                    f"<td>{bar(x['new'], '#38bdf8')}</td></tr>")
            parts.append("</table>")
    open(path, "w").write("\n".join(parts))
    print(f"\nWrote HTML report → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, help="single season (default: latest cached)")
    ap.add_argument("--all", action="store_true", help="every cached season")
    ap.add_argument("--pos", choices=POSITIONS)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--movers", action="store_true")
    ap.add_argument("--html", metavar="PATH", help="write a visual HTML report")
    args = ap.parse_args()

    seasons = available_seasons()
    if not seasons:
        print(f"No usage caches found at {CACHE_GLOB}")
        return
    if args.all:
        chosen = seasons
    elif args.season:
        chosen = [args.season]
    else:
        chosen = [seasons[-1]]

    sym = load_new_symbols()
    season_rows_map = {}
    for s in chosen:
        if not os.path.exists(season_path(s)):
            print(f"(skip {s}: no cache)")
            continue
        rows = compute_season(s, sym)
        season_rows_map[s] = rows
        if not args.html:
            print_report(s, rows, args.pos, args.limit, args.movers)
    if args.html and season_rows_map:
        write_html(args.html, season_rows_map, args.limit)


if __name__ == "__main__":
    main()
