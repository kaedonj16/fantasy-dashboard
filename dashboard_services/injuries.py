from __future__ import annotations

import pandas as pd
from datetime import datetime
from typing import Dict, List
from zoneinfo import ZoneInfo

from .api import get_nfl_players
from .players import build_roster_map, get_league_rostered_player_ids

INJURY_STATUSES = {"IR", "OUT", "DOUBTFUL", "QUESTIONABLE", "PUP", "NFI", "SUSP"}


def build_injury_report(league_id: str,
                        local_tz: str = "America/New_York",
                        include_free_agents: bool = False,
                        ) -> pd.DataFrame:
    """
    Build an injury table for the league:
      columns: Team, Player, Pos, NFL, Status, Injury, Body, Last Updated
    """
    players = get_nfl_players()
    roster_map = build_roster_map(league_id)  # roster_id -> Team Name
    rostered = get_league_rostered_player_ids(league_id)

    tz_local = ZoneInfo(local_tz)
    rows: list[dict] = []

    # --- De-dupe fix (1): build reverse index with SETS so a player isn't repeated per roster ---
    pid_to_rids: Dict[str, set[str]] = {}
    for rid, pids in rostered.items():
        for pid in pids:
            pid_to_rids.setdefault(str(pid), set()).add(str(rid))

    for pid, p in players.items():
        pid_s = str(pid)

        # core identity
        name = (p.get("full_name")
                or " ".join([x for x in (p.get("first_name"), p.get("last_name")) if x])
                or pid_s)
        pos = p.get("position") or ((p.get("fantasy_positions") or [None])[0]) or ""
        nfl_team = p.get("team") or "FA"
        status = p.get("status") or ""  # e.g., Active, IR, Out, Questionable
        inj = p.get("injury_status") or ""  # e.g., Out, Questionable
        body = p.get("injury_body_part") or ""

        lm = p.get("last_modified")
        last_mod_local = None
        if isinstance(lm, (int, float)) and lm > 0:
            last_mod_local = datetime.fromtimestamp(lm / 1000.0, tz=ZoneInfo("UTC")).astimezone(tz_local)

        # filter: only include clearly flagged injuries unless include_free_agents is True
        is_flagged = (status.upper() in INJURY_STATUSES) or (inj.upper() in INJURY_STATUSES)
        if not is_flagged and not include_free_agents:
            if pid_s not in pid_to_rids:
                continue
            if not (inj or body or status.upper() in {"Q", "D", "O", "IR"}):
                continue

        rids = pid_to_rids.get(pid_s, set())

        # Free agents (when requested)
        if include_free_agents and not rids:
            rows.append({
                "Team": "Free Agent",
                "Player": name, "Pos": pos, "NFL": nfl_team, "PlayerID": pid_s,
                "Status": status, "Injury": inj, "Body": body,
                "Last Updated": last_mod_local
            })
            continue

        # Normal rostered rows (sets already prevent duplicates)
        for rid in sorted(rids):
            rows.append({
                "Team": roster_map.get(str(rid), f"Roster {rid}"),
                "Player": name, "Pos": pos, "NFL": nfl_team,
                "Status": status, "Injury": inj, "Body": body,
                "Last Updated": last_mod_local
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["Team", "Player", "Pos", "NFL", "Status", "Injury", "Body", "Last Updated"]]

        # --- De-dupe fix (2): if a player appears multiple times for same team, keep most severe, latest ---
        sev_rank = {"IR": 4, "OUT": 3, "DOUBTFUL": 2, "QUESTIONABLE": 1}
        df["__sev"] = df["Injury"].str.upper().map(sev_rank).fillna(0).astype(int)
        df["__ts"] = pd.to_datetime(df["Last Updated"], errors="coerce")

        df = (
            df.sort_values(["__sev", "__ts"], ascending=[False, False])
            .drop_duplicates(subset=["Team", "Player"], keep="first")
            .drop(columns=["__sev", "__ts"])
        )

        # Final display sort (IR > OUT > DOUBTFUL > QUESTIONABLE > others)
        priority = {"IR": 0, "OUT": 1, "DOUBTFUL": 2, "QUESTIONABLE": 3}
        df["__pri"] = df["Injury"].str.upper().map(priority).fillna(9)
        df = (
            df.sort_values(["__pri", "Team", "Player"])
            .drop(columns="__pri")
            .reset_index(drop=True)
        )

    return df


def render_injury_accordion(df_inj: pd.DataFrame) -> str:
    if df_inj.empty:
        return ""
    parts = [
        "<div class='card injury-overview' data-section='activity'>"
        "<h2>Team Injury Overview</h2>"
        "<div class='scroll-box'>"
    ]
    for team, g in df_inj.groupby("Team"):
        injury_count = len(g)
        rows = []
        for _, r in g.iterrows():
            status = (str(r.get("Injury") or r.get("Status") or "")).strip()
            lw = "Bench"
            rows.append(
                "<div class='inj-row'>"
                f"  <div class='left'>"
                f"    <div class='pname'>{r['Player']}</div>"
                f"    <div class='sub'>{r['NFL']} • {r['Pos']} • {r.get('Body', '')}</div>"
                f"  </div>"
                f"  <div class='right'>"
                f"    <span class='chip'>{status or 'Note'}</span>"
                f"    <span class='chip'>{lw}</span>"
                f"  </div>"
                "</div>"
            )
        parts.append(
            f"<details class='inj-acc card'>"
            f"  <summary style='display:flex; justify-content:space-between; align-items:center;'>"
            f"    <span style='font-weight:600; color: #122d4b'>{team}</span>"
            f"    <span class='chip injury-count'>{injury_count}</span>"
            f"  </summary>"
            f"  <div class='inj-body'>{''.join(rows)}</div>"
            f"</details>"
        )
    parts.append("</div></div>")
    return "".join(parts)
