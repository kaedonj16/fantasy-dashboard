#!/usr/bin/env python3
"""
Download NFL team logos from ESPN and save them to
static/images/team_logos/{TEAM}.png (32 teams).

Run from the project root:
    python -m data_building.updates.download_team_logos
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

# ESPN team abbreviation -> ESPN team ID mapping
# ESPN uses numeric IDs in their CDN URLs for team logos.
ESPN_TEAMS: dict[str, int] = {
    "ARI": 22, "ATL": 1,  "BAL": 33, "BUF": 2,
    "CAR": 29, "CHI": 3,  "CIN": 4,  "CLE": 5,
    "DAL": 6,  "DEN": 7,  "DET": 8,  "GB":  9,
    "HOU": 34, "IND": 11, "JAX": 30, "KC":  12,
    "LAC": 24, "LAR": 14, "LV":  13, "MIA": 15,
    "MIN": 16, "NE":  17, "NO":  18, "NYG": 19,
    "NYJ": 20, "PHI": 21, "PIT": 23, "SEA": 26,
    "SF":  25, "TB":  27, "TEN": 10, "WAS": 28,
}

ESPN_LOGO_URL = "https://a.espncdn.com/i/teamlogos/nfl/500/{team_lower}.png"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def download_team_logos() -> None:
    out_dir = _project_root() / "static" / "images" / "team_logos"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"

    ok = skipped = failed = 0
    for team in sorted(ESPN_TEAMS):
        dest = out_dir / f"{team}.png"
        if dest.exists():
            print(f"  skip  {team} (already exists)")
            skipped += 1
            continue

        url = ESPN_LOGO_URL.format(team_lower=team.lower())
        try:
            r = session.get(url, timeout=10)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"  ok    {team} ({len(r.content):,} bytes)")
            ok += 1
        except Exception as exc:
            print(f"  FAIL  {team}: {exc}")
            failed += 1

        time.sleep(0.05)

    print(f"\nDone. {ok} downloaded, {skipped} skipped, {failed} failed.")
    print(f"Logos saved to: {out_dir}")


if __name__ == "__main__":
    download_team_logos()
