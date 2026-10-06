#!/usr/bin/env python3
"""
Fix player_advanced_metrics IDs: remap PFF player_id -> Sleeper player_id.

The NFL PFF stat CSVs were imported into player_advanced_metrics keyed by the
PFF player_id (e.g. Matthew Stafford = 4924). The rest of the app — including
the breakout archetype cache (scripts/export_archetype_cache.py) and the
gsis->sleeper cross-reference in the breakout build — keys everything by
Sleeper player_id, so those PFF-keyed rows can never be matched and role-fit
labels stay empty.

This migration builds a PFF-id -> Sleeper-id map by matching player NAME (the
CSVs carry both the PFF id and the player name) against cache/players_index.json,
disambiguating collisions by position and team, then rewrites every matching
player_advanced_metrics.player_id to the Sleeper id.

The rewrite is two-phase (PFF id -> 'SLPR:<sleeper>' -> '<sleeper>') so that an
in-place id that happens to equal another row's target value can't collide
mid-update. Rows whose current player_id isn't a known PFF id are left untouched.

Run on a machine with DB access:
    python scripts/fix_advanced_metrics_ids.py            # apply
    python scripts/fix_advanced_metrics_ids.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CSV_DIR = Path("data/pff_nfl_2025")
CSV_FILES = ["passing_summary.csv", "receiving_summary.csv", "rushing_summary.csv"]
PLAYERS_INDEX = Path("cache/players_index.json")

# PFF team codes that differ from Sleeper's
TEAM_ALIAS = {
    "LA": "LAR", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "JAX": "JAC",
}

# PFF position -> set of acceptable Sleeper positions
POS_GROUP = {
    "QB": {"QB"}, "HB": {"RB"}, "RB": {"RB"}, "FB": {"RB", "FB"},
    "WR": {"WR"}, "TE": {"TE"}, "P": {"P"}, "K": {"K"},
}

# Manual name aliases (PFF spelling -> Sleeper spelling) for cases the
# normalizer can't bridge (nicknames that aren't simple prefixes).
NAME_ALIASES = {
    "marquise brown": "hollywood brown",
}


def _norm(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"[.'’,\-]", " ", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _build_index_maps(idx: dict):
    """Return (by_name, by_lastname) maps keyed by normalized name / last name.

    Each entry is (sid, pos, team, norm_name) so the fuzzy fallback can compare
    first-name tokens (e.g. Scott -> Scotty, Mitch -> Mitchell).
    """
    by_name: dict[str, list] = {}
    by_lastname: dict[str, list] = {}
    for sid, v in idx.items():
        nm = _norm(v.get("name"))
        if not nm:
            continue
        entry = (sid, (v.get("pos") or "").upper(), (v.get("team") or "").upper(), nm)
        by_name.setdefault(nm, []).append(entry)
        by_lastname.setdefault(nm.split()[-1], []).append(entry)
    return by_name, by_lastname


def _resolve(name, pff_pos, pff_team, by_name, by_lastname):
    """Resolve a CSV (name,pos,team) to a single sleeper_id, or None."""
    key = _norm(name)
    key = _norm(NAME_ALIASES.get(key, key))
    pg = POS_GROUP.get((pff_pos or "").upper(), {(pff_pos or "").upper()})
    team = TEAM_ALIAS.get((pff_team or "").upper(), (pff_team or "").upper())

    cands = by_name.get(key, [])
    # Fuzzy fallback: same last name + position group, preferring a first-name
    # prefix match so nicknames (Scott/Scotty, Mitch/Mitchell, Ben/Benjamin)
    # still resolve.
    if not cands:
        first = key.split()[0] if key else ""
        last = key.split()[-1] if key else ""
        pool = [c for c in by_lastname.get(last, []) if c[1] in pg]
        prefix = [c for c in pool
                  if first and (c[3].split()[0].startswith(first)
                                or first.startswith(c[3].split()[0]))]
        cands = prefix or pool

    if not cands:
        return None
    if len(cands) == 1:
        return cands[0][0]
    pos_match = [c for c in cands if c[1] in pg]
    if len(pos_match) == 1:
        return pos_match[0][0]
    pool = pos_match or cands
    team_match = [c for c in pool if c[2] == team]
    if len(team_match) == 1:
        return team_match[0][0]
    return pool[0][0]  # last resort: best positional candidate


def build_pff_to_sleeper() -> dict[str, str]:
    """Build {pff_id: sleeper_id} from the NFL CSVs + players_index."""
    idx = json.loads(PLAYERS_INDEX.read_text())
    by_name, by_lastname = _build_index_maps(idx)

    pff_to_sleeper: dict[str, str] = {}
    unmatched: list[str] = []
    for fname in CSV_FILES:
        path = CSV_DIR / fname
        if not path.exists():
            print(f"  [warn] missing {path}")
            continue
        for row in csv.DictReader(path.open()):
            pff_id = str(row.get("player_id") or "").strip()
            if not pff_id:
                continue
            if pff_id in pff_to_sleeper:
                continue
            sid = _resolve(row.get("player"), row.get("position"),
                           row.get("team_name"), by_name, by_lastname)
            if sid:
                pff_to_sleeper[pff_id] = sid
            else:
                unmatched.append(f"{row.get('player')} ({row.get('position')},{row.get('team_name')})")

    print(f"  mapped {len(pff_to_sleeper)} PFF ids -> sleeper ids; {len(unmatched)} unmatched")
    if unmatched:
        print("  unmatched (left untouched):", ", ".join(sorted(set(unmatched))))
    return pff_to_sleeper


def apply_migration(pff_to_sleeper: dict[str, str], dry_run: bool = False):
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT player_id FROM player_advanced_metrics")
        existing_ids = {str(r["player_id"]) for r in cur.fetchall()}

        remap = {pff: sid for pff, sid in pff_to_sleeper.items() if pff in existing_ids}
        print(f"  {len(remap)} of {len(existing_ids)} distinct ids in the table will be remapped")

        if dry_run:
            sample = list(remap.items())[:10]
            for pff, sid in sample:
                print(f"    {pff} -> {sid}")
            print("  --dry-run: no changes written")
            return

        # Phase 1: PFF id -> temp namespace (avoids collisions if a sleeper
        # target equals an as-yet-unmapped PFF id elsewhere in the table).
        phase1 = 0
        for pff, sid in remap.items():
            cur.execute(
                "UPDATE player_advanced_metrics SET player_id = %s WHERE player_id = %s",
                (f"SLPR:{sid}", pff),
            )
            phase1 += cur.rowcount

        # Phase 2: strip the temp prefix. Where a real sleeper row already
        # exists for the same (player_id, as_of_date), drop the duplicate temp
        # row instead of violating the unique constraint.
        cur.execute(
            "SELECT player_id, as_of_date FROM player_advanced_metrics "
            "WHERE player_id LIKE 'SLPR:%'"
        )
        temp_rows = [(str(r["player_id"]), r["as_of_date"]) for r in cur.fetchall()]
        phase2 = dropped = 0
        for temp_id, as_of in temp_rows:
            sid = temp_id.split("SLPR:", 1)[1]
            cur.execute(
                "SELECT 1 FROM player_advanced_metrics WHERE player_id = %s AND as_of_date = %s",
                (sid, as_of),
            )
            if cur.fetchone():
                cur.execute(
                    "DELETE FROM player_advanced_metrics WHERE player_id = %s AND as_of_date = %s",
                    (temp_id, as_of),
                )
                dropped += cur.rowcount
            else:
                cur.execute(
                    "UPDATE player_advanced_metrics SET player_id = %s "
                    "WHERE player_id = %s AND as_of_date = %s",
                    (sid, temp_id, as_of),
                )
                phase2 += cur.rowcount

        conn.commit()
        print(f"  remapped {phase2} rows ({phase1} matched in phase 1, "
              f"{dropped} duplicate rows dropped)")


def main():
    ap = argparse.ArgumentParser(description="Remap player_advanced_metrics PFF ids to Sleeper ids")
    ap.add_argument("--dry-run", action="store_true", help="report the mapping without writing")
    args = ap.parse_args()

    print("Building PFF -> Sleeper id map from NFL CSVs + players_index...")
    pff_to_sleeper = build_pff_to_sleeper()
    if not pff_to_sleeper:
        print("No mappings built — aborting.")
        sys.exit(1)

    print("Applying migration to player_advanced_metrics...")
    apply_migration(pff_to_sleeper, dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
