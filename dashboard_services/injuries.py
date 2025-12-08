from __future__ import annotations

import pandas as pd
from datetime import datetime
from typing import Dict, List
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

from .api import get_nfl_players
from .players import build_roster_map, get_league_rostered_player_ids

INJURY_STATUSES = {"IR", "OUT", "DOUBTFUL", "QUESTIONABLE", "PUP", "NFI", "SUSP"}
INJURY_SHORT_STATUSES = {"Q", "D", "O", "IR"}


def build_player_news_url(name: str, nfl_team: str | None = None) -> str:
    """
    Build a generic 'news / injury' search URL for a player.
    You can swap this out for a real news API later.
    """
    if nfl_team:
        query = f"{name} {nfl_team} injury"
    else:
        query = f"{name} injury"

    q = quote_plus(query)
    return f"https://www.google.com/search?q={q}"


def build_injury_report(
    league_id: str,
    local_tz: str = "America/New_York",
    include_free_agents: bool = False,
) -> pd.DataFrame:
    """
    Build an injury table for the league:
      columns: Team, Player, Pos, NFL, Status, Injury, Body, Last Updated, NewsUrl
    """
    players = get_nfl_players()
    roster_map = build_roster_map(league_id)  # roster_id -> Team Name
    rostered = get_league_rostered_player_ids(league_id)

    tz_local = ZoneInfo(local_tz)
    tz_utc = ZoneInfo("UTC")

    rows: list[dict] = []

    # --- De-dupe fix (1): build reverse index with SETS so a player isn't repeated per roster ---
    pid_to_rids: Dict[str, set[str]] = {}
    for rid, pids in rostered.items():
        rid_s = str(rid)
        for pid in pids:
            pid_to_rids.setdefault(str(pid), set()).add(rid_s)

    for pid, p in players.items():
        pid_s = str(pid)

        # core identity
        first_name = p.get("first_name")
        last_name = p.get("last_name")
        full_name = p.get("full_name")
        if full_name:
            name = full_name
        else:
            if first_name or last_name:
                name = " ".join(x for x in (first_name, last_name) if x)
            else:
                name = pid_s

        fantasy_positions = p.get("fantasy_positions") or [None]
        pos = p.get("position") or fantasy_positions[0] or ""
        nfl_team = p.get("team") or "FA"

        status = p.get("status") or ""           # e.g., Active, IR, Out, Questionable
        inj = p.get("injury_status") or ""       # e.g., Out, Questionable
        body = p.get("injury_body_part") or ""
        news_url = build_player_news_url(name, nfl_team)

        lm = p.get("last_modified")
        last_mod_local = None
        if isinstance(lm, (int, float)) and lm > 0:
            # timestamp is ms from epoch
            last_mod_local = datetime.fromtimestamp(lm / 1000.0, tz=tz_utc).astimezone(tz_local)

        status_upper = status.upper()
        inj_upper = inj.upper()

        # filter: only include clearly flagged injuries unless include_free_agents is True
        is_flagged = (status_upper in INJURY_STATUSES) or (inj_upper in INJURY_STATUSES)
        if not is_flagged and not include_free_agents:
            if pid_s not in pid_to_rids:
                continue
            if not (inj or body or status_upper in INJURY_SHORT_STATUSES):
                continue

        rids = pid_to_rids.get(pid_s, set())

        # Free agents (when requested)
        if include_free_agents and not rids:
            rows.append(
                {
                    "Team": "Free Agent",
                    "Player": name,
                    "Pos": pos,
                    "NFL": nfl_team,
                    "PlayerID": pid_s,
                    "Status": status,
                    "Injury": inj,
                    "Body": body,
                    "Last Updated": last_mod_local,
                    "NewsUrl": news_url,
                }
            )
            continue

        # Normal rostered rows (sets already prevent duplicates)
        for rid in sorted(rids):
            rows.append(
                {
                    "Team": roster_map.get(rid, f"Roster {rid}"),
                    "Player": name,
                    "Pos": pos,
                    "NFL": nfl_team,
                    "Status": status,
                    "Injury": inj,
                    "Body": body,
                    "Last Updated": last_mod_local,
                    "NewsUrl": news_url,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[
            [
                "Team",
                "Player",
                "Pos",
                "NFL",
                "Status",
                "Injury",
                "Body",
                "Last Updated",
                "NewsUrl",
            ]
        ]

        # --- De-dupe fix (2): if a player appears multiple times for same team, keep most severe, latest ---
        sev_rank = {"IR": 4, "OUT": 3, "DOUBTFUL": 2, "QUESTIONABLE": 1}
        inj_upper_series = df["Injury"].str.upper()
        df["__sev"] = inj_upper_series.map(sev_rank).fillna(0).astype(int)
        df["__ts"] = pd.to_datetime(df["Last Updated"], errors="coerce")

        df = (
            df.sort_values(["__sev", "__ts"], ascending=[False, False])
            .drop_duplicates(subset=["Team", "Player"], keep="first")
            .drop(columns=["__sev", "__ts"])
        )

        # Final display sort (IR > OUT > DOUBTFUL > QUESTIONABLE > others)
        priority = {"IR": 0, "OUT": 1, "DOUBTFUL": 2, "QUESTIONABLE": 3}
        inj_upper_series = df["Injury"].str.upper()
        df["__pri"] = inj_upper_series.map(priority).fillna(9)
        df = (
            df.sort_values(["__pri", "Team", "Player"])
            .drop(columns="__pri")
            .reset_index(drop=True)
        )

    return df


def render_injury_accordion(df_inj: pd.DataFrame) -> str:
    if df_inj.empty:
        return ""

    parts: List[str] = [
        "<div class='card injury-overview' data-section='activity'>"
        "<h2>Team Injury Overview</h2>"
        "<div class='scroll-box'>"
    ]

    # groupby is already efficient; just avoid extra work in the inner loop
    for team, g in df_inj.groupby("Team"):
        injury_count = len(g)
        rows: List[str] = []
        for _, r in g.iterrows():
            status_val = r.get("Injury") or r.get("Status") or ""
            status = str(status_val).strip()
            lw = "Bench"
            player = r["Player"]
            nfl = r["NFL"]
            pos = r["Pos"]
            body = r.get("Body", "")
            news_url = r["NewsUrl"]

            # NOTE: keep HTML and inline styles exactly as before
            rows.append(
                "<div class='inj-row'>"
                f"  <div class='left'>"
                f"    <div class='pname'><a href='{news_url}' target='_blank' rel='noopener noreferrer'style='color:#122d4b'>{player}</a></div>"
                f"    <div class='sub'>{nfl} • {pos} • {body}</div>"
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
