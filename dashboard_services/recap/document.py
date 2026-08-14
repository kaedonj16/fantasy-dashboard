"""Build the canonical, presentation-independent weekly recap document.

The existing weekly recap already computes trustworthy league facts.  This
module turns those facts into ranked story modules before AI is involved.  Page
HTML, share cards, notifications, and future delivery formats can therefore use
the same selected stories without asking a language model to decide what
happened.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


SCHEMA_VERSION = 1
STORY_SELECTOR_VERSION = "v1"


def _story(story_id: str, kind: str, score: float, title: str, body: str,
           facts: dict[str, Any], teams: list[str]) -> dict[str, Any]:
    return {
        "id": story_id,
        "type": kind,
        "score": round(float(score), 4),
        "title": title,
        "body": body,
        "facts": facts,
        "teams": [team for team in teams if team],
    }


def _closest_finish(payload: dict) -> dict | None:
    games = payload.get("matchups") or []
    if not games:
        return None
    game = min(
        games,
        key=lambda row: abs(float(row.get("team_a_pts") or 0) - float(row.get("team_b_pts") or 0)),
    )
    a, b = str(game.get("team_a") or ""), str(game.get("team_b") or "")
    a_pts, b_pts = float(game.get("team_a_pts") or 0), float(game.get("team_b_pts") or 0)
    margin = abs(a_pts - b_pts)
    winner = a if a_pts > b_pts else b
    loser = b if a_pts > b_pts else a
    # A sub-point finish should nearly always lead; ordinary margins remain a
    # useful module but lose to an upset, record score, or meaningful rank move.
    score = 0.58 + max(0.0, 0.42 * (1.0 - min(margin, 12.0) / 12.0))
    facts = {
        "winner": winner, "loser": loser, "winner_points": max(a_pts, b_pts),
        "loser_points": min(a_pts, b_pts), "margin": margin,
    }
    title = f"{winner} survived the closest finish"
    body = f"{winner} beat {loser} by {margin:.2f} points, {max(a_pts, b_pts):.2f}-{min(a_pts, b_pts):.2f}."
    return _story("closest_finish", "closest_matchup", score, title, body, facts, [winner, loser])


def _high_scorer(payload: dict) -> dict | None:
    high = payload.get("high_scorer") or {}
    if not high.get("team") or high.get("pts") is None:
        return None
    team, points = str(high["team"]), float(high["pts"])
    return _story(
        "high_scorer", "high_scorer", min(0.84, 0.48 + points / 500.0),
        f"{team} set the pace", f"{team} led the league with {points:.1f} points.",
        {"team": team, "points": points}, [team],
    )


def _biggest_blowout(payload: dict) -> dict | None:
    blowout = payload.get("biggest_blowout") or {}
    if not blowout.get("winner") or blowout.get("margin") is None:
        return None
    winner, loser = str(blowout["winner"]), str(blowout.get("loser") or "")
    margin = float(blowout["margin"])
    if margin < 15:
        return None
    return _story(
        "biggest_blowout", "blowout", min(0.86, 0.48 + margin / 120.0),
        f"{winner} ran away with it",
        f"{winner} beat {loser} by {margin:.1f} points.", deepcopy(blowout), [winner, loser],
    )


def _best_upset(payload: dict) -> dict | None:
    upsets = payload.get("upsets") or []
    if not upsets:
        return None
    upset = max(
        upsets,
        key=lambda row: (
            int(row.get("winner_rank_before") or 0) - int(row.get("loser_rank_before") or 0),
            float(row.get("margin") or 0),
        ),
    )
    winner, loser = str(upset.get("winner") or ""), str(upset.get("loser") or "")
    rank_gap = max(0, int(upset.get("winner_rank_before") or 0) - int(upset.get("loser_rank_before") or 0))
    score = min(0.96, 0.66 + rank_gap * 0.045 + min(float(upset.get("margin") or 0), 30) / 300)
    return _story(
        "biggest_upset", "upset", score, f"{winner} flipped the standings script",
        f"No. {upset.get('winner_rank_before')} {winner} beat No. {upset.get('loser_rank_before')} {loser}.",
        deepcopy(upset), [winner, loser],
    )


def _rank_mover(payload: dict) -> dict | None:
    movers = payload.get("big_movers") or []
    if not movers:
        return None
    mover = max(movers, key=lambda row: abs(int(row.get("from") or 0) - int(row.get("to") or 0)))
    team = str(mover.get("team") or "")
    old, new = int(mover.get("from") or 0), int(mover.get("to") or 0)
    delta = abs(old - new)
    direction = "climbed" if new < old else "fell"
    return _story(
        "biggest_rank_move", "rank_movement", min(0.82, 0.52 + delta * 0.07),
        f"{team} made the week's biggest move", f"{team} {direction} from No. {old} to No. {new}.",
        deepcopy(mover), [team],
    )


def _playoff_race(payload: dict) -> dict | None:
    race = payload.get("playoff_race") or {}
    if not race.get("tight"):
        return None
    cutline, bubble = str(race.get("cutline_team") or ""), str(race.get("bubble_team") or "")
    return _story(
        "playoff_bubble", "playoff_race", 0.74,
        "The playoff cutline is still crowded",
        f"{cutline} holds the last spot over {bubble}, with both teams at {race.get('cutline_record')}.",
        deepcopy(race), [cutline, bubble],
    )


def _select_stories(candidates: list[dict], limit: int = 4) -> list[dict]:
    """Choose a varied set and avoid turning one team's bad week into the page."""
    selected: list[dict] = []
    type_seen: set[str] = set()
    team_appearances: dict[str, int] = {}
    for candidate in sorted(candidates, key=lambda row: (-row["score"], row["id"])):
        if candidate["type"] in type_seen:
            continue
        # A team may headline twice, but not dominate every module.
        if selected and any(team_appearances.get(team, 0) >= 2 for team in candidate["teams"]):
            continue
        selected.append(candidate)
        type_seen.add(candidate["type"])
        for team in candidate["teams"]:
            team_appearances[team] = team_appearances.get(team, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def build_recap_document(payload: dict) -> dict:
    """Create the canonical recap document from an existing factual payload."""
    facts = deepcopy(payload)
    candidates = [
        candidate for candidate in (
            _closest_finish(facts), _high_scorer(facts), _biggest_blowout(facts),
            _best_upset(facts), _rank_mover(facts), _playoff_race(facts),
        ) if candidate is not None
    ]
    stories = _select_stories(candidates)
    fallback_headline = stories[0]["title"] if stories else f"Week {facts.get('week')} Recap"
    signature_source = json.dumps(facts, sort_keys=True, default=str, separators=(",", ":"))
    return {
        "schema_version": SCHEMA_VERSION,
        "selector_version": STORY_SELECTOR_VERSION,
        "league_name": str(facts.get("league_name") or "the league"),
        "week": facts.get("week"),
        "data_signature": hashlib.sha256(signature_source.encode("utf-8")).hexdigest(),
        "facts": facts,
        "stories": stories,
        "featured_story_id": stories[0]["id"] if stories else None,
        "narrative": {
            "source": "deterministic",
            "headline": fallback_headline,
            "paragraphs": [story["body"] for story in stories],
            "looking_ahead": "",
        },
    }


def apply_ai_narrative(document: dict, result: dict) -> dict:
    """Attach optional AI wording without allowing it to alter verified facts."""
    updated = deepcopy(document)
    fallback = updated.get("narrative") or {}
    paragraphs = [str(p).strip() for p in (result.get("paragraphs") or []) if str(p).strip()]
    updated["narrative"] = {
        "source": "ai",
        "headline": str(result.get("headline") or fallback.get("headline") or "Week Recap").strip(),
        "paragraphs": paragraphs or list(fallback.get("paragraphs") or []),
        "looking_ahead": str(result.get("looking_ahead") or "").strip(),
    }
    return updated


def recap_document_to_json(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


def recap_document_from_json(raw: str) -> dict | None:
    try:
        document = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        return None
    if not isinstance(document.get("facts"), dict) or not isinstance(document.get("narrative"), dict):
        return None
    return document
