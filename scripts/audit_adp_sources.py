#!/usr/bin/env python3
"""Diagnose every ADP source locally or from Render.

Safe, read-only. It fetches (or inspects the persisted snapshot for) each source,
reports how many players each representative format returns, the provider-id
mapping percentage, a sample of unresolved players, any source errors, and the
capabilities the source declares. Use it to verify — from an environment with
outbound network access — the endpoint/parameter/response assumptions the code
makes, exactly as Priority 5 requires (CI never hits live APIs; this does).

Usage:
    python scripts/audit_adp_sources.py [--season 2026] [--live] [--json]

Without --live it only inspects persisted snapshots and the DB crawler tables
(no external calls). With --live it also hits the public Yahoo/ESPN/MFL and
Sleeper endpoints. Never prints credentials; the global feeds require none.
"""

from __future__ import annotations

import argparse
import json as _json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pct(mapped: int, raw: int) -> str:
    return f"{(100.0 * mapped / raw):.1f}%" if raw else "n/a"


def _fmt_count(m) -> int:
    return len(m or {})


def audit(season: int, live: bool) -> dict:
    from dashboard_services import adp_service as A
    from dashboard_services.adp_formats import (
        AdpFormat, SOURCE_CAPABILITIES, classify_match,
    )

    report: dict = {"season": season, "live": live,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "sources": {}, "capabilities": {}, "errors": {}}

    # Declared capabilities for every source.
    for key, cap in SOURCE_CAPABILITIES.items():
        report["capabilities"][key] = {
            "scope": cap.scope, "axes": sorted(cap.axes()),
            "provides_tep": cap.provides_tep,
            "league_size_known": cap.league_size_known,
            "real_vs_mock_known": cap.real_vs_mock_known,
            "notes": cap.notes,
        }

    # ── Sleeper (explicit scoring fields) ────────────────────────────────────
    try:
        s = {
            "redraft PPR": _fmt_count(A.resolve_market_adp(season, False, "redraft", "sleeper")),
            "redraft 2QB": _fmt_count(A.resolve_market_adp(season, True, "redraft", "sleeper")),
            "dynasty PPR": _fmt_count(A.resolve_market_adp(season, False, "dynasty", "sleeper")),
            "dynasty 2QB": _fmt_count(A.resolve_market_adp(season, True, "dynasty", "sleeper")),
        }
        report["sources"]["sleeper"] = s
    except Exception as exc:  # noqa: BLE001
        report["errors"]["sleeper"] = f"{type(exc).__name__}: {exc}"

    # ── BR Fantasy (observed drafts, DB) ─────────────────────────────────────
    try:
        report["sources"]["brfantasy"] = {
            "startup 1QB": _fmt_count(A.fetch_crawler_adp(season, False, "dynasty")),
            "startup SF": _fmt_count(A.fetch_crawler_adp(season, True, "dynasty")),
            "rookie SF": _fmt_count(A.fetch_crawler_adp(season, True, "rookie")),
            "redraft PPR": _fmt_count(A.fetch_crawler_adp(season, False, "redraft")),
        }
    except Exception as exc:  # noqa: BLE001
        report["errors"]["brfantasy"] = f"{type(exc).__name__}: {exc}"

    # ── Global feeds: snapshot state (+ optional live fetch) ──────────────────
    for src in ("yahoo", "espn", "mfl"):
        entry: dict = {}
        snap = A.load_adp_snapshot(src, "redraft", season)
        ca = snap.get("collected_at")
        entry["snapshot_rows"] = _fmt_count(snap.get("adp"))
        entry["last_refresh"] = (datetime.utcfromtimestamp(ca).isoformat() + "Z") if ca else None
        report["sources"][src] = entry

    if live:
        try:
            from dashboard_services.providers import global_adp as G
        except Exception as exc:  # noqa: BLE001
            report["errors"]["global_adp_import"] = f"{type(exc).__name__}: {exc}"
            G = None
        if G is not None:
            _live_yahoo(G, season, report)
            _live_espn(G, season, report)
            _live_mfl(G, season, report)

    # ── Consensus source-count distribution for a common request ─────────────
    try:
        detailed = A.resolve_market_adp_detailed(
            season, AdpFormat("redraft", "1qb", 1.0))
        dist: dict = {}
        for rec in detailed.values():
            n = rec["source_count"]
            dist[n] = dist.get(n, 0) + 1
        report["consensus_redraft_ppr_1qb"] = {
            "players": len(detailed),
            "by_source_count": {str(k): dist[k] for k in sorted(dist)},
        }
    except Exception as exc:  # noqa: BLE001
        report["errors"]["consensus"] = f"{type(exc).__name__}: {exc}"

    # Which formats each source is exact/compatible/generic for (spot examples).
    report["match_matrix"] = _match_matrix()
    return report


def _live_yahoo(G, season, report):
    try:
        r = G.fetch_yahoo_global_adp(season)
        report["sources"].setdefault("yahoo", {}).update(
            live_global_redraft=r["mapped_count"], raw=r["raw_count"],
            mapping_pct=_pct(r["mapped_count"], r["raw_count"]),
            unmapped_sample=r["unmapped"][:10])
    except Exception as exc:  # noqa: BLE001
        report["errors"]["yahoo_live"] = f"{type(exc).__name__}: {exc}"


def _live_espn(G, season, report):
    try:
        r = G.fetch_espn_global_adp(season)
        report["sources"].setdefault("espn", {}).update(
            live_global_adp=r["mapped_count"], live_ppr_rank=len(r.get("ppr_rank") or {}),
            raw=r["raw_count"], mapping_pct=_pct(r["mapped_count"], r["raw_count"]),
            unmapped_sample=r["unmapped"][:10])
    except Exception as exc:  # noqa: BLE001
        report["errors"]["espn_live"] = f"{type(exc).__name__}: {exc}"


def _live_mfl(G, season, report):
    try:
        ppr = G.fetch_mfl_adp(season, is_ppr=1, fcount=12, is_mock=0)
        rookie_note = "MFL exposes no verified rookie ADP filter"
        report["sources"].setdefault("mfl", {}).update(
            live_redraft_ppr_12team_real=ppr["mapped_count"], raw=ppr["raw_count"],
            mapping_pct=_pct(ppr["mapped_count"], ppr["raw_count"]),
            unmapped_sample=ppr["unmapped"][:10], rookie=rookie_note)
    except Exception as exc:  # noqa: BLE001
        report["errors"]["mfl_live"] = f"{type(exc).__name__}: {exc}"


def _match_matrix() -> dict:
    from dashboard_services.adp_formats import AdpFormat, classify_match, SOURCE_CAPABILITIES
    examples = {
        "redraft/1qb/ppr": AdpFormat("redraft", "1qb", 1.0),
        "dynasty/superflex/ppr/+0.5TEP": AdpFormat("startup", "superflex", 1.0, 0.5),
        "rookie/superflex": AdpFormat("rookie", "superflex", "unknown"),
    }
    return {label: {s: classify_match(fmt, s) for s in SOURCE_CAPABILITIES}
            for label, fmt in examples.items()}


def main():
    ap = argparse.ArgumentParser(description="Audit ADP sources.")
    ap.add_argument("--season", type=int, default=datetime.utcnow().year)
    ap.add_argument("--live", action="store_true",
                    help="also hit the public Yahoo/ESPN/MFL endpoints")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    report = audit(args.season, args.live)
    if args.json:
        print(_json.dumps(report, indent=2, default=str))
        return
    print(_json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
