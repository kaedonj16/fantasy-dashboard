from __future__ import annotations

import html
import json
import logging

import pandas as pd

from dashboard_services.ai.cache import build_ai_cache_key, load_cached_ai_text, save_cached_ai_text
from dashboard_services.ai.client import (
    AIRateLimitError,
    AIUnavailableError,
    get_ai_client,
)
from dashboard_services.ai.renderer import ai_available

logger = logging.getLogger(__name__)


def _streak_for(results: list[str]) -> str:
    """Compute current streak like 'W3' or 'L2' from a list of 'W'/'L' results."""
    if not results:
        return "-"
    last = results[-1]
    n = 1
    for r in reversed(results[:-1]):
        if r == last:
            n += 1
        else:
            break
    return f"{last}{n}"


def _build_team_storylines(
        df_weekly: pd.DataFrame,
        selected_week: int,
        team_by_rid: dict,
) -> list[dict]:
    """For each team, compute record/rank/streak before and after the selected week."""
    fin_df = df_weekly[(df_weekly["finalized"] == True) & (df_weekly["week"] <= selected_week)].copy()
    if fin_df.empty:
        return []

    fin_df["win"] = fin_df["points"] > fin_df["points_against"]

    # Build per-team history sorted by week
    teams: dict[str, dict] = {}
    for rid, grp in fin_df.groupby("roster_id"):
        grp = grp.sort_values("week")
        results = ["W" if w else "L" for w in grp["win"].tolist()]
        # Snapshot through selected_week
        owner = str(grp["owner"].iloc[0])
        teams[str(rid)] = {
            "rid": str(rid),
            "team": team_by_rid.get(str(rid), owner),
            "owner": owner,
            "results": results,
            "weeks": [int(w) for w in grp["week"].tolist()],
            "pts_by_week": [float(p) for p in grp["points"].tolist()],
        }

    # Compute snapshots BEFORE selected_week
    snap_before: dict[str, dict] = {}
    snap_after: dict[str, dict] = {}
    for rid, t in teams.items():
        # Before this week: only weeks < selected_week
        before_idx = [i for i, w in enumerate(t["weeks"]) if w < selected_week]
        wins_b = sum(1 for i in before_idx if t["results"][i] == "W")
        losses_b = sum(1 for i in before_idx if t["results"][i] == "L")
        pf_b = sum(t["pts_by_week"][i] for i in before_idx)
        snap_before[rid] = {
            "wins": wins_b, "losses": losses_b, "pf": pf_b,
            "streak": _streak_for([t["results"][i] for i in before_idx]),
        }
        # After (through this week, inclusive)
        wins_a = sum(1 for r in t["results"] if r == "W")
        losses_a = sum(1 for r in t["results"] if r == "L")
        pf_a = sum(t["pts_by_week"])
        snap_after[rid] = {
            "wins": wins_a, "losses": losses_a, "pf": pf_a,
            "streak": _streak_for(t["results"]),
        }

    # Compute rank (1 = best) by wins then PF
    def _rank(snap_map):
        items = list(snap_map.items())
        items.sort(key=lambda x: (-x[1]["wins"], -x[1]["pf"]))
        return {rid: i + 1 for i, (rid, _s) in enumerate(items)}

    rank_before = _rank(snap_before)
    rank_after = _rank(snap_after)

    # This-week result per team
    week_df = df_weekly[(df_weekly["week"] == selected_week) & (df_weekly["finalized"] == True)]
    week_result_by_rid: dict[str, dict] = {}
    for _, row in week_df.iterrows():
        rid = str(row.get("roster_id"))
        week_result_by_rid[rid] = {
            "pts": float(row.get("points") or 0),
            "opp_pts": float(row.get("points_against") or 0),
            "won": bool(row.get("points", 0) > row.get("points_against", 0)),
        }

    storylines = []
    for rid, t in teams.items():
        wr = week_result_by_rid.get(rid, {})
        before = snap_before[rid]
        after = snap_after[rid]
        storylines.append({
            "team": t["team"],
            "owner": t["owner"],
            "this_week_pts": wr.get("pts"),
            "opp_pts": wr.get("opp_pts"),
            "won_this_week": wr.get("won"),
            "record_before": f"{before['wins']}-{before['losses']}",
            "record_after": f"{after['wins']}-{after['losses']}",
            "rank_before": rank_before.get(rid),
            "rank_after": rank_after.get(rid),
            "rank_change": (rank_before.get(rid) or 0) - (rank_after.get(rid) or 0),  # +ve = moved up
            "streak": after["streak"],
            "streak_before": before["streak"],
            "pf_after": round(after["pf"], 1),
        })

    storylines.sort(key=lambda x: x["rank_after"] or 99)
    return storylines


def build_weekly_recap_payload(
        df_weekly: pd.DataFrame,
        matchups_by_week: dict,
        selected_week: int,
        team_by_rid: dict,
        league: dict,
) -> dict:
    storylines = _build_team_storylines(df_weekly, selected_week, team_by_rid)

    settings = (league or {}).get("settings") or {}
    playoff_start = int(settings.get("playoff_week_start") or 14)
    playoff_teams = int(settings.get("playoff_teams") or 6)
    league_name = (league or {}).get("name") or "the league"

    weeks_until_playoffs = max(0, playoff_start - 1 - selected_week)

    # Identify upsets (lower-ranked-before beats higher-ranked-before)
    week_matchups_raw = matchups_by_week.get(selected_week) or []
    upsets = []
    biggest_blowout = None
    for m in week_matchups_raw:
        l = m.get("left") or {}
        r = m.get("right") or {}
        l_rid = str(l.get("roster_id") or "")
        r_rid = str(r.get("roster_id") or "")
        l_pts = float(l.get("pts_total") or 0)
        r_pts = float(r.get("pts_total") or 0)
        if l_pts == r_pts:
            continue
        winner_rid, loser_rid = (l_rid, r_rid) if l_pts > r_pts else (r_rid, l_rid)
        winner_pts, loser_pts = (l_pts, r_pts) if l_pts > r_pts else (r_pts, l_pts)
        winner = next((s for s in storylines if s["team"] == team_by_rid.get(winner_rid)
                       or s["owner"] == (l.get("username") if winner_rid == l_rid else r.get("username"))), None)
        loser = next((s for s in storylines if s["team"] == team_by_rid.get(loser_rid)
                      or s["owner"] == (l.get("username") if loser_rid == l_rid else r.get("username"))), None)
        margin = winner_pts - loser_pts
        if winner and loser and winner["rank_before"] and loser["rank_before"]:
            if winner["rank_before"] > loser["rank_before"] + 2:
                upsets.append({
                    "winner": winner["team"], "loser": loser["team"],
                    "winner_rank_before": winner["rank_before"],
                    "loser_rank_before": loser["rank_before"],
                    "margin": round(margin, 1),
                })
        if biggest_blowout is None or margin > biggest_blowout["margin"]:
            biggest_blowout = {
                "winner": team_by_rid.get(winner_rid, "Unknown"),
                "loser": team_by_rid.get(loser_rid, "Unknown"),
                "winner_pts": round(winner_pts, 1),
                "loser_pts": round(loser_pts, 1),
                "margin": round(margin, 1),
            }

    # Playoff race tightness: gap between cutline team and just-outside team
    playoff_race = None
    if storylines and playoff_teams and len(storylines) > playoff_teams:
        cutline = storylines[playoff_teams - 1]
        bubble = storylines[playoff_teams]
        wins_gap = (cutline["record_after"].split("-")[0] != bubble["record_after"].split("-")[0])
        playoff_race = {
            "cutline_team": cutline["team"],
            "cutline_record": cutline["record_after"],
            "bubble_team": bubble["team"],
            "bubble_record": bubble["record_after"],
            "tight": not wins_gap,
        }

    # Notable highlights
    high_scorer = max(storylines, key=lambda x: x["this_week_pts"] or 0) if storylines else None
    low_scorer = min(storylines, key=lambda x: x["this_week_pts"] if x["this_week_pts"] is not None else 1e9) if storylines else None
    hot_streaks = [s for s in storylines if s["streak"].startswith("W") and int(s["streak"][1:] or 0) >= 3]
    cold_streaks = [s for s in storylines if s["streak"].startswith("L") and int(s["streak"][1:] or 0) >= 2]
    big_movers = sorted([s for s in storylines if abs(s["rank_change"]) >= 2],
                        key=lambda x: -abs(x["rank_change"]))[:3]

    return {
        "league_name": league_name,
        "week": selected_week,
        "weeks_until_playoffs": weeks_until_playoffs,
        "playoff_teams": playoff_teams,
        "teams": [
            {
                "team": s["team"],
                "rank_before": s["rank_before"],
                "rank_after": s["rank_after"],
                "record_after": s["record_after"],
                "streak": s["streak"],
                "won": s["won_this_week"],
                "pts": round(s["this_week_pts"], 1) if s["this_week_pts"] is not None else None,
                "opp_pts": round(s["opp_pts"], 1) if s["opp_pts"] is not None else None,
                "rank_change": s["rank_change"],
            } for s in storylines
        ],
        "high_scorer": {"team": high_scorer["team"], "pts": round(high_scorer["this_week_pts"], 1)} if high_scorer and high_scorer["this_week_pts"] else None,
        "low_scorer":  {"team": low_scorer["team"],  "pts": round(low_scorer["this_week_pts"], 1)} if low_scorer and low_scorer["this_week_pts"] else None,
        "biggest_blowout": biggest_blowout,
        "upsets": upsets,
        "hot_streaks": [{"team": s["team"], "streak": s["streak"]} for s in hot_streaks],
        "cold_streaks": [{"team": s["team"], "streak": s["streak"]} for s in cold_streaks],
        "big_movers": [
            {"team": s["team"], "from": s["rank_before"], "to": s["rank_after"]}
            for s in big_movers
        ],
        "playoff_race": playoff_race,
    }


def _generate_ai_storyline(payload: dict) -> dict:
    """Call OpenAI to generate the weekly recap column."""
    client = get_ai_client()

    schema = {
        "type": "object",
        "properties": {
            "headline": {"type": "string", "description": "Punchy headline (max 80 chars) for the week."},
            "paragraphs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-4 paragraphs of league-wide storytelling. Each paragraph should be 2-4 sentences.",
            },
        },
        "required": ["headline", "paragraphs"],
        "additionalProperties": False,
    }

    system_prompt = """
You are a fantasy football beat writer for a small league. Write a 2-4 paragraph weekly recap column that captures the storylines from this week — like ESPN's weekend wrap.

Guidelines:
- Tell a STORY. Don't list facts. Connect events into a narrative.
- Lead with the most compelling storyline (a hot/cold streak, an upset, a #1 seed in trouble, etc.).
- Reference teams by name and use specific numbers (scores, records, streak lengths).
- Conversational but sharp. Use active voice and strong verbs.
- If late in the season, weave in playoff implications.
- 2-4 paragraphs, each 2-4 sentences. NEVER more than 4 paragraphs.

Do NOT:
- Invent facts not in the data.
- Use markdown formatting or bullet points.
- Generic openers like "What a week!"
- Repeat the same stat in multiple paragraphs.
""".strip()

    user_prompt = f"""
Write a weekly recap column for {payload['league_name']}, Week {payload['week']}.

Weeks until playoffs: {payload['weeks_until_playoffs']} (playoffs start week {payload['week'] + payload['weeks_until_playoffs'] + 1 if payload['weeks_until_playoffs'] else payload['week']})

Standings (sorted by rank after this week):
{json.dumps(payload['teams'], indent=2)}

High scorer: {json.dumps(payload['high_scorer'])}
Low scorer: {json.dumps(payload['low_scorer'])}
Biggest blowout: {json.dumps(payload['biggest_blowout'])}
Upsets (lower-seeded beat higher-seeded by 3+ ranks): {json.dumps(payload['upsets'])}
Hot streaks (3+ wins in a row): {json.dumps(payload['hot_streaks'])}
Cold streaks (2+ losses in a row): {json.dumps(payload['cold_streaks'])}
Big movers (rank moved 2+ spots): {json.dumps(payload['big_movers'])}
Playoff race: {json.dumps(payload['playoff_race'])}

Write the column now. Lead with whatever storyline is most compelling.
""".strip()

    resp = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "weekly_recap",
                "schema": schema,
            }
        },
    )
    return json.loads(resp.output_text.strip())


def _render_recap_html(result: dict) -> str:
    headline = html.escape(str(result.get("headline") or "Week Recap"))
    paragraphs = result.get("paragraphs") or []
    paragraphs_html = "\n".join(
        f"<p style='margin:0 0 10px 0;'>{html.escape(str(p))}</p>"
        for p in paragraphs[:4] if str(p).strip()
    )

    return f"""
<div class="card" style="padding:18px 20px;margin-bottom:20px;">
  <div style="font-size:10px;font-weight:800;letter-spacing:.1em;color:var(--accent);margin-bottom:6px;">THE COLUMN</div>
  <div style="font-size:18px;font-weight:800;margin-bottom:12px;line-height:1.25;">{headline}</div>
  <div style="font-size:13px;line-height:1.55;color:var(--text);">
    {paragraphs_html}
  </div>
</div>
"""


def get_weekly_ai_recap(
        df_weekly: pd.DataFrame,
        matchups_by_week: dict,
        selected_week: int,
        team_by_rid: dict,
        league: dict,
        league_id: str,
        season,
) -> str:
    """Return cached or freshly-generated HTML for the weekly AI recap column."""
    if df_weekly is None or df_weekly.empty:
        return ""

    cache_key = f"weekly_recap_{league_id}_{season}_w{selected_week}_v2"
    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    try:
        payload = build_weekly_recap_payload(
            df_weekly, matchups_by_week, selected_week, team_by_rid, league,
        )
    except Exception as exc:
        logger.warning("[weekly-recap] payload build failed: %s", exc)
        return ""

    if not ai_available():
        return ""  # silently skip when AI is off

    try:
        result = _generate_ai_storyline(payload)
        html_out = _render_recap_html(result)
        save_cached_ai_text(cache_key, html_out)
        return html_out
    except (AIRateLimitError, AIUnavailableError) as exc:
        logger.warning("[weekly-recap] AI unavailable: %s", exc)
        return ""
    except Exception as exc:
        logger.warning("[weekly-recap] AI error: %s", exc)
        return ""


def get_weekly_ai_recap_preview() -> str:
    """Static sample for preview mode."""
    return _render_recap_html({
        "headline": "Dynasty Kings cling to the top as Endzone Elite roar back",
        "paragraphs": [
            "Week 1 served notice that this league won't have a runaway leader. Dynasty Kings opened the season with a convincing 132.4–98.7 win to claim early bragging rights, but the real story was Endzone Elite, who responded to a sluggish preseason with a 124.2 point outburst and a statement victory over Pocket Protectors.",
            "Blitz Brigade are already looking like the league's hard-luck team — their 118-point effort would have beaten three other squads, but it ran into a buzzsaw and now they're 0-1 staring at a tough Week 2 slate. Redzone Rebels, by contrast, snuck out a single-score win and will take it.",
            "Watch the trade wire over the next two weeks: with bye weeks looming and playoff seeding lines this tight, the manager who flips Week 1's overreactions into real moves will be the one we're writing about in November.",
        ],
    })
