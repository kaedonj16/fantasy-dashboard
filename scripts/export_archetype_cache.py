#!/usr/bin/env python3
"""
Export WR/TE archetype profiles from nfl_data_py to a JSON cache
the (DB-free) breakout rebuild reads for role-fit context labels.

Run: python scripts/export_archetype_cache.py 2025
Writes cache/archetype_{season}.json:
    {"season": 2025, "players": {"<sleeper_id>": {"adot": .., "yac_per_rec": ..,
      "slot_rate": null, "wide_rate": null, "inline_rate": null}, ...}}

adot  = receiving_air_yards / targets   (nfl_data_py weekly)
yac   = receiving_yards_after_catch / receptions (nfl_data_py weekly)
slot_rate / wide_rate / inline_rate are set to null — not available from
nfl_data_py without PFF data; the archetype vector treats null alignment
as 0.5 (neutral), so role-fit still differentiates players by target depth
and YAC profile. Role fits degrade safely when absent from this cache.
"""
import json
import sys
from pathlib import Path

MIN_TARGETS = 30  # minimum targets to include a player


def run(season: int) -> int:
    """Export archetype profiles for `season` to cache/archetype_{season}.json.

    Returns the number of player profiles written.
    """
    import nfl_data_py as nfl

    print(f"  [archetype] loading nfl_data_py weekly data for {season}...")
    weekly = nfl.import_weekly_data([season])
    skill = weekly[weekly["position"].isin(["WR", "TE"])].copy()

    print(f"  [archetype] loading seasonal rosters for {season} (gsis→sleeper map)...")
    rosters = nfl.import_seasonal_rosters([season])
    gsis_to_sleeper: dict[str, str] = {}
    for _, row in rosters.iterrows():
        gsis = str(row.get("player_id", "") or "")
        sid = str(row.get("sleeper_id", "") or "")
        if gsis and sid and sid not in ("None", "nan", ""):
            gsis_to_sleeper[gsis] = sid

    # Aggregate season totals per player (regular season only)
    reg = skill[skill["season_type"] == "REG"] if "season_type" in skill.columns else skill
    totals: dict[str, dict] = {}
    for _, row in reg.iterrows():
        gsis = str(row.get("player_id", "") or "")
        if not gsis or gsis == "nan":
            continue
        t = float(row.get("targets") or 0)
        air = float(row.get("receiving_air_yards") or 0)
        yac = float(row.get("receiving_yards_after_catch") or 0)
        rec = float(row.get("receptions") or 0)

        if gsis not in totals:
            totals[gsis] = {"targets": 0.0, "air_yards": 0.0, "yac": 0.0, "receptions": 0.0}
        totals[gsis]["targets"] += t
        totals[gsis]["air_yards"] += air
        totals[gsis]["yac"] += yac
        totals[gsis]["receptions"] += rec

    players: dict[str, dict] = {}
    skipped_targets = skipped_no_sid = 0
    for gsis, stats in totals.items():
        t = stats["targets"]
        if t < MIN_TARGETS:
            skipped_targets += 1
            continue
        sid = gsis_to_sleeper.get(gsis)
        if not sid:
            skipped_no_sid += 1
            continue

        adot = round(stats["air_yards"] / t, 2) if t > 0 else None
        rec = stats["receptions"]
        yac_per_rec = round(stats["yac"] / rec, 2) if rec > 0 else None

        players[sid] = {
            "adot": adot,
            "slot_rate": None,
            "wide_rate": None,
            "inline_rate": None,
            "yac_per_rec": yac_per_rec,
        }

    out_dir = Path("cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"archetype_{season}.json"
    out_path.write_text(json.dumps({"season": season, "players": players}, indent=2))
    print(
        f"  [archetype] wrote {len(players)} WR/TE profiles → {out_path} "
        f"(skipped {skipped_targets} low-vol, {skipped_no_sid} no-sleeper-id)"
    )
    return len(players)


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/export_archetype_cache.py <season>")
        sys.exit(1)
    run(int(sys.argv[1]))


if __name__ == "__main__":
    main()
