#!/usr/bin/env python3
"""Site-wide projected-PPG consistency diagnostic.

Puts the TWO projection pipelines side by side for one roster so we can see
exactly where they diverge (different fallback source, different week, missing
players, or a flat scale difference):

  * SIM path  — data_building.simulate_playoff_odds.build_ppg_map
                (what Playoff Impact / playoff-odds uses)
  * SITE path — app.build_projections_by_week
                (what Start/Sit, the player modal, and matchups show — the
                 number you compare against, e.g. "my lineup projects 134")

The canonical season-average value is printed with provenance beside the
weekly site and simulation contexts.  Weekly disagreement with season average
is allowed and explicitly labelled; disagreement between a season consumer and
CANONICAL is a failure.

Usage:
    python scripts/diagnose_proj_ppg.py --league sleeper:LEAGUE_ID:SEASON
    python scripts/diagnose_proj_ppg.py --league sleeper:LEAGUE_ID:SEASON --roster 3
    python scripts/diagnose_proj_ppg.py --league sleeper:LEAGUE_ID:SEASON --owner myusername

If neither --roster nor --owner is given, every roster's two totals are listed,
then the first roster gets the detailed per-player breakdown.
"""
from __future__ import annotations

import argparse
import os
import sys

# Repo root on path so `app` and `data_building` import when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_roster_id(ctx, roster_arg, owner_arg):
    rosters = ctx.get("rosters") or []
    if roster_arg is not None:
        return int(roster_arg)
    if owner_arg:
        users = {str(u.get("user_id")): u for u in (ctx.get("users") or [])}
        want = owner_arg.strip().lower()
        for r in rosters:
            u = users.get(str(r.get("owner_id"))) or {}
            names = {
                str(u.get("display_name") or "").lower(),
                str((u.get("metadata") or {}).get("team_name") or "").lower(),
            }
            if want in names:
                return int(r.get("roster_id"))
        print(f"[warn] owner '{owner_arg}' not found; using first roster", file=sys.stderr)
    return int(rosters[0].get("roster_id")) if rosters else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose Proj PPG gap between sim and site")
    ap.add_argument("--league", required=True, help="platform:league_id:season")
    ap.add_argument("--roster", type=int, default=None, help="roster_id to detail")
    ap.add_argument("--owner", default=None, help="owner display name / team name to detail")
    args = ap.parse_args()

    try:
        platform, league_id, season = args.league.split(":")
        season = int(season)
    except ValueError:
        print("[error] --league must be platform:league_id:season", file=sys.stderr)
        return 1

    from app import build_league_context, build_projections_by_week
    from data_building.simulate_playoff_odds import build_ppg_map, _position_aware_lineup
    from utils.utils import pick_proj_variant
    from utils.projection_resolver import resolve_projected_ppg_many

    try:
        from utils.utils import load_players_index
        _player_index = load_players_index() or {}
        names = {str(k): (v or {}).get("name", k) for k, v in _player_index.items()}
        positions = {str(k): str((v or {}).get("pos") or (v or {}).get("position") or "")
                     for k, v in _player_index.items()}
    except Exception:
        names = {}
        positions = {}

    ctx = build_league_context(platform, league_id, season)
    rosters          = ctx.get("rosters") or []
    roster_positions = ctx.get("roster_positions") or []
    current_week     = int(ctx.get("current_week") or 0)
    raw_ss           = ctx.get("raw_scoring_settings") or {}
    variant          = pick_proj_variant(raw_ss)

    all_pids = sorted({str(pid) for r in rosters for pid in (r.get("players") or [])})
    canonical = resolve_projected_ppg_many(all_pids, raw_ss, season,
        positions=positions)

    if not rosters:
        print("[error] no rosters in context", file=sys.stderr)
        return 1

    # ── SIM path map ─────────────────────────────────────────────────────────
    sim_ppg, sim_pos = build_ppg_map(ctx)

    # ── SITE path map (the current week's bundle) ────────────────────────────
    proj_by_week = build_projections_by_week(season, 18, raw_ss)
    site_available = bool(proj_by_week.get("_available"))
    detail_week = current_week if current_week > 0 else 1
    site_flat = ((proj_by_week.get(detail_week) or {}).get("projections")) or {}
    # Wrap into the {pid: {ppg, pos}} shape the lineup solver expects.
    site_ppg = {str(pid): {"ppg": float(v), "pos": sim_pos.get(str(pid), "")}
                for pid, v in site_flat.items() if v}

    print("=" * 64)
    from utils.projection_resolver import scoring_fingerprint
    print(f"League {platform}:{league_id}:{season}  current_week={current_week}  "
          f"variant={variant} scoring_fingerprint={scoring_fingerprint(raw_ss)}")
    print("Normalized scoring: " + " ".join(
        f"{key}={raw_ss.get(key)!r}" for key in
        ("rec", "pass_td", "bonus_rec_te", "pass_yd", "rush_yd", "rec_yd")
    ))
    print(f"SIM   map: {len(sim_ppg):>5} players (build_ppg_map, proj_week="
          f"{current_week + 1 if current_week > 0 else 1})")
    print(f"SITE  map: {len(site_ppg):>5} players (build_projections_by_week "
          f"week={detail_week}, available={site_available})")
    print("=" * 64)

    # ── Per-roster totals ────────────────────────────────────────────────────
    print(f"\n{'roster':>6} {'SIM total':>10} {'SITE total':>11} {'gap':>8}")
    print("-" * 40)
    for r in sorted(rosters, key=lambda x: x.get("roster_id") or 0):
        rid  = r.get("roster_id")
        pids = r.get("players") or []
        sim_tot,  _ = _position_aware_lineup(pids, sim_ppg,  sim_pos, roster_positions)
        site_tot, _ = _position_aware_lineup(pids, site_ppg, sim_pos, roster_positions)
        print(f"{rid:>6} {sim_tot:>10.1f} {site_tot:>11.1f} {site_tot - sim_tot:>+8.1f}")

    # ── Detailed per-player breakdown for one roster ─────────────────────────
    detail_rid = _resolve_roster_id(ctx, args.roster, args.owner)
    detail_roster = next((r for r in rosters if int(r.get("roster_id")) == detail_rid), None)
    if not detail_roster:
        return 0

    pids = detail_roster.get("players") or []
    print(f"\n{'='*64}\nRoster {detail_rid} — per-player projection (starters first)\n{'='*64}")
    print(f"{'player':<24} {'pos':>4} {'CANON':>7} {'SIM/wk':>7} {'SITE/wk':>8}  status/source")
    print("-" * 78)

    def _row(pid):
        s  = sim_ppg.get(str(pid))
        w  = site_ppg.get(str(pid))
        sv = float(s["ppg"]) if s else 0.0
        wv = float(w["ppg"]) if w else 0.0
        pos = (s or w or {}).get("pos") or sim_pos.get(str(pid), "")
        if s and w:
            src = "both"
        elif s:
            src = "SIM only"
        elif w:
            src = "SITE only"
        else:
            src = "neither"
        nm = names.get(str(pid), str(pid))[:24]
        cv = (canonical.get(str(pid)) or {}).get("ppg")
        result = canonical.get(str(pid)) or {}
        provenance = result.get("source") or "unavailable"
        source_type = result.get("source_projection_type") or "none"
        season_detail = ""
        if result.get("season_points") is not None:
            season_detail = f" {result['season_points']:.1f}pts/{result.get('projected_games') or '?'}g"
        return (pos, nm, cv, sv, wv, src, provenance + "/" + source_type + season_detail)

    rows = sorted((_row(p) for p in pids), key=lambda x: max(x[2] or 0, x[3], x[4]), reverse=True)
    for pos, nm, cv, sv, wv, src, provenance in rows:
        canon_text = f"{cv:.1f}" if cv is not None else "—"
        print(f"{nm:<24} {pos:>4} {canon_text:>7} {sv:>7.1f} {wv:>8.1f}  weekly:{src}; season:{provenance}")

    sim_tot,  _ = _position_aware_lineup(pids, sim_ppg,  sim_pos, roster_positions)
    site_tot, _ = _position_aware_lineup(pids, site_ppg, sim_pos, roster_positions)
    print("-" * 64)
    print(f"{'OPTIMAL LINEUP TOTAL':<29} {sim_tot:>7.1f} {site_tot:>7.1f} "
          f"{site_tot - sim_tot:>+7.1f}")
    print("\nReading it:")
    print("  - 'SITE only' rows  → sim's map is missing players the site projects")
    print("    (sim fell back to a sparser source for this week).")
    print("  - gaps roughly proportional across all players → a flat scale/variant")
    print("    difference between the two fallback sources.")
    print("  - big per-player gaps on a few names → stale/backup listings differ.")
    print("  - CANONICAL is season_average; SIM/wk and SITE/wk are explicitly weekly")
    print("    and may differ. Season consumers must equal CANONICAL exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
