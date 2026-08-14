"""Delivery-format presenters for a canonical weekly recap document."""

from __future__ import annotations

from copy import deepcopy


def _story_copy(story: dict) -> tuple[str, str]:
    narrative = story.get("narrative") or {}
    return (
        str(narrative.get("title") or story.get("title") or "").strip(),
        str(narrative.get("body") or story.get("body") or "").strip(),
    )


def build_recap_text(document: dict, canonical_url: str = "") -> str:
    """Build concise, copy-ready group-chat text from selected stories."""
    league = str(document.get("league_name") or "League")
    week = document.get("week")
    narrative = document.get("narrative") or {}
    headline = str(narrative.get("headline") or f"Week {week} Recap").strip()
    lines = [f"{league} · Week {week}", headline]
    for story in (document.get("stories") or [])[:4]:
        title, body = _story_copy(story)
        if title and body:
            lines.append(f"{title}: {body}")
        elif body:
            lines.append(body)
    if canonical_url:
        lines.append(canonical_url)
    return "\n\n".join(lines)


def augment_recap_share_payload(base_payload: dict, document: dict | None,
                                canonical_url: str = "") -> dict:
    """Add canonical story data without breaking the existing canvas payload."""
    payload = deepcopy(base_payload)
    if not document:
        return payload

    narrative = document.get("narrative") or {}
    stories = []
    for story in (document.get("stories") or [])[:4]:
        title, body = _story_copy(story)
        stories.append({
            "id": story.get("id"),
            "type": story.get("type"),
            "title": title,
            "body": body,
        })
    payload.update({
        "headline": str(narrative.get("headline") or "").strip(),
        "featured_story_id": document.get("featured_story_id"),
        "stories": stories,
        "text": build_recap_text(document, canonical_url=canonical_url),
        "url": canonical_url,
        "document_schema": document.get("schema_version"),
    })
    return payload
