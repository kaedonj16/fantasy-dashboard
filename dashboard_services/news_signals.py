"""
Structured free news signals for player modals and Start/Sit rows.

The headline blend in :mod:`dashboard_services.news` answers "what was
written about this player". This module answers the narrower question a
lineup decision actually turns on: "is there anything this week that
changes whether / how much this player plays?" It builds that answer only
from free, no-key sources:

  1. ESPN injuries (site.web.api, the same unblocked host news.py uses) --
     the live per-team injury list: status, body part, return date, and
     the wire comment (which often names the practice week in prose).
  2. The Sleeper players feed, via the app's existing cached accessor
     (``dashboard_services.api.get_nfl_players``, ttl-cached) -- never a
     second raw download. Sleeper carries the structured
     ``practice_participation`` field ESPN lacks.
  3. RotoBaller RSS -- fantasy-written blurbs (headline + take), labelled
     "Fantasy" in the UI. ProFootballTalk RSS -- general NFL reporting.

Signals are CHIPS, not score inputs (Kaedon's settled signal-not-score
rule, same as the QB-situation chip): nothing here touches
``compute_start_score``. The only number derived is a play probability for
questionable players, always shown labelled as a heuristic alongside the
raw projection, never folded into it silently.

Historical note: nflverse publishes the official weekly injury reports
(injuries_{season}.parquet: report_status + practice_status by gsis_id)
and the repo already reads them offline (data_building breakout
reconstruction, oline prior). There is no live sync of that file into
the web process, so v1 uses ESPN injuries as the live source for the
same official-report data and leaves nflverse as the backfill/history
source. Do not add a parquet download to the request path.

httpx is imported lazily inside the fetch functions (same pattern as
news.py) so the pure parsers stay importable in the httpx-free CI shard.
Every source fails soft: one dead feed degrades to the others and is
named in ``sources_failed``.
"""

from __future__ import annotations

import html as _html
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

SRC_ESPN_INJ = "espn_injuries"
SRC_SLEEPER = "sleeper"
SRC_ROTO = "rotoballer"
SRC_PFT = "pft"

_ESPN_INJURIES_URL = (
    "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
)
_ROTO_URL = "https://www.rotoballer.com/player-news?feed=rss"
_PFT_URL = "https://profootballtalk.nbcsports.com/feed/"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fantasy-dashboard/1.0)"}
_TIMEOUT = 6

_CACHE: dict = {}
_TTL = 900          # 15 min for bulk feeds (injuries / RSS)
_FAIL_TTL = 120     # failed bulk fetches retry sooner, but not per request

# ── Play-probability heuristic (labelled, never silent) ─────────────────────
# Probability a player with a game-status designation actually plays, by his
# latest practice participation. Labelled heuristic from the standard beat
# reading of the official report (a Friday full practice almost always
# plays; a Friday DNP tagged questionable is a true coin flip; doubtful
# almost never plays). Shown to the user as "Play probability ~NN%".
# Out / IR / PUP / suspended are not probabilities: the player is
# unavailable, probability 0.
PLAY_PROB_DOUBTFUL = 0.10
PLAY_PROB_QUESTIONABLE = {
    "full": 0.90,
    "limited": 0.75,
    "dnp": 0.50,
}

_UNAVAILABLE = {
    "OUT": "Out",
    "INJURED RESERVE": "Injured Reserve",
    "IR": "Injured Reserve",
    "PUP": "PUP",
    "PHYSICALLY UNABLE TO PERFORM": "PUP",
    "SUSPENDED": "Suspended",
    "SUSPENSION": "Suspended",
    "NFI": "NFI",
    "RESERVE/INJURED": "Injured Reserve",
}


def _age_label(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        s = int(delta.total_seconds())
        if s < 3600:
            return f"{max(s // 60, 0)}m ago"
        if s < 86400:
            return f"{s // 3600}h ago"
        return f"{delta.days}d ago"
    except Exception:
        return ""


def _norm_name(name: str) -> str:
    """Lowercase, punctuation-free name key for cross-source matching."""
    n = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower())
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def normalize_practice(raw: str) -> Optional[str]:
    """Map any source's practice wording to 'full' / 'limited' / 'dnp'."""
    t = (raw or "").strip().lower()
    if not t:
        return None
    if "did not" in t or t == "dnp" or "no practice" in t or "not participate" in t:
        return "dnp"
    if "limited" in t:
        return "limited"
    if "full" in t:
        return "full"
    return None


_PRACTICE_TEXT_RES = (
    (re.compile(r"did not (?:practice|participate)|a non[- ]participant|"
                r"no practice (?:wednesday|thursday|friday)", re.I), "dnp"),
    (re.compile(r"limited (?:in )?(?:practice|participant)|"
                r"limited participant|practiced in a limited", re.I), "limited"),
    (re.compile(r"full (?:practice|participant)|practiced fully|"
                r"fully participated", re.I), "full"),
)


def practice_from_text(text: str) -> Optional[str]:
    """Mine a practice designation out of wire-comment prose (ESPN's
    injuries feed carries no structured practice field; the comment
    usually states it: 'was limited in practice Friday')."""
    for rx, val in _PRACTICE_TEXT_RES:
        if rx.search(text or ""):
            return val
    return None


def play_probability(status: str, practice: Optional[str]) -> Optional[float]:
    """Labelled heuristic probability the player plays this week.

    Returns 0.0 for unavailable designations, PLAY_PROB_DOUBTFUL for
    doubtful, the practice-keyed value for questionable, and None when
    there is no basis for a number (healthy, or questionable with no
    practice information from any source - no number is shown then,
    rather than a fabricated one).
    """
    s = (status or "").strip().upper()
    if s in _UNAVAILABLE:
        return 0.0
    if s in ("DOUBTFUL", "D"):
        return PLAY_PROB_DOUBTFUL
    if s in ("QUESTIONABLE", "Q"):
        return PLAY_PROB_QUESTIONABLE.get(practice or "")
    return None


# ── ESPN injuries parsing (pure) ─────────────────────────────────────────────

def _espn_id_from_href(href: str) -> str:
    m = re.search(r"/(\d+)\.png", href or "")
    return m.group(1) if m else ""


def parse_espn_injuries(payload: dict) -> dict:
    """Parse the ESPN injuries payload into {norm_name: record}.

    Record: {name, team (abbr), team_name, espn_id, position, status,
    body_part, detail, return_date, comment, published, age}.
    Only non-Active entries are kept: Active rows are post-game stat
    notes, not availability news. Malformed entries are skipped.
    """
    out: dict = {}
    for team_block in (payload or {}).get("injuries") or []:
        if not isinstance(team_block, dict):
            continue
        for entry in team_block.get("injuries") or []:
            try:
                if not isinstance(entry, dict):
                    continue
                status = (entry.get("status") or "").strip()
                if not status or status.lower() == "active":
                    continue
                athlete = entry.get("athlete") or {}
                name = (athlete.get("displayName") or "").strip()
                if not name:
                    continue
                team = athlete.get("team") or {}
                details = entry.get("details") or {}
                comment = (entry.get("longComment") or "").strip()
                if not comment or comment.lower() == "inactive":
                    comment = (entry.get("shortComment") or "").strip()
                published = (entry.get("date") or "").strip()
                rec = {
                    "name": name,
                    "team": (team.get("abbreviation") or "").upper(),
                    "team_name": team.get("displayName") or team_block.get("displayName") or "",
                    "espn_id": _espn_id_from_href((athlete.get("headshot") or {}).get("href") or ""),
                    "position": ((athlete.get("position") or {}).get("abbreviation") or "").upper(),
                    "status": status,
                    "body_part": (details.get("type") or "").strip(),
                    "detail": (details.get("detail") or "").strip(),
                    "return_date": (details.get("returnDate") or "").strip(),
                    "comment": comment,
                    "published": published,
                    "age": _age_label(published) if published else "",
                }
                out[_norm_name(name)] = rec
            except Exception:
                continue
    return out


# ── Sleeper record (pure) ────────────────────────────────────────────────────

def sleeper_record(feed_rec: Optional[dict]) -> Optional[dict]:
    """Normalize one Sleeper players-feed entry to the fields signals use."""
    if not feed_rec or not isinstance(feed_rec, dict):
        return None
    status = (feed_rec.get("injury_status") or feed_rec.get("status") or "").strip()
    practice = normalize_practice(feed_rec.get("practice_participation") or "")
    if not status and not practice and not feed_rec.get("injury_body_part"):
        return None
    return {
        "status": status,
        "body_part": (feed_rec.get("injury_body_part") or "").strip(),
        "practice": practice,
        "depth_chart_order": feed_rec.get("depth_chart_order"),
        "news_updated": feed_rec.get("news_updated"),
    }


# ── Signal derivation (pure) ─────────────────────────────────────────────────

_PRACTICE_LABEL = {"full": "Full practice", "limited": "Limited in practice",
                   "dnp": "Did not practice"}


def derive_signals(espn_rec: Optional[dict], sleeper_rec: Optional[dict]) -> list:
    """Build the signal chips for one player from the merged sources.

    Practice priority: Sleeper's structured field, then the practice
    designation mined from the ESPN wire comment. The winning source is
    named in the chip detail. Healthy players (no designation anywhere)
    get no signals - an empty list is the honest state, not a green chip.
    """
    status = ""
    body = ""
    sources = []
    if espn_rec:
        status = (espn_rec.get("status") or "").strip()
        body = (espn_rec.get("body_part") or "").strip()
        sources.append("ESPN injuries")
    sl_status = (sleeper_rec or {}).get("status") or ""
    if sl_status and sl_status.lower() not in ("active", "act", ""):
        if not status:
            status = sl_status.strip()
        sources.append("Sleeper")
    if not body:
        body = ((sleeper_rec or {}).get("body_part") or "").strip()

    practice = None
    practice_source = ""
    if sleeper_rec and sleeper_rec.get("practice"):
        practice = sleeper_rec["practice"]
        practice_source = "Sleeper"
    elif espn_rec:
        practice = practice_from_text(espn_rec.get("comment") or "")
        if practice:
            practice_source = "ESPN injuries"

    if not status and not practice:
        return []

    published = (espn_rec or {}).get("published") or ""
    age = (espn_rec or {}).get("age") or ""
    src_label = " + ".join(dict.fromkeys(sources)) or "ESPN injuries"
    body_txt = f" ({body.lower()})" if body else ""
    prob = play_probability(status, practice)
    s_up = status.upper()

    if s_up in _UNAVAILABLE or prob == 0.0:
        label = _UNAVAILABLE.get(s_up, status.title())
        detail = f"{label}{body_txt}."
        if espn_rec and espn_rec.get("return_date"):
            detail += f" ESPN lists a return date of {espn_rec['return_date']}."
        if espn_rec and espn_rec.get("comment"):
            detail += f" {espn_rec['comment']}"
        return [{
            "kind": "unavailable", "label": label, "tone": "bad",
            "detail": detail.strip(), "source": src_label,
            "published": published, "age": age, "play_probability": 0.0,
        }]

    if s_up in ("QUESTIONABLE", "Q", "DOUBTFUL", "D"):
        is_doubt = s_up in ("DOUBTFUL", "D")
        label = f"{'Doubtful' if is_doubt else 'Questionable'}{body_txt}"
        bits = [f"{'Doubtful' if is_doubt else 'Questionable'}{body_txt} this week."]
        if practice:
            bits.append(f"{_PRACTICE_LABEL[practice]} (per {practice_source}).")
        if espn_rec and espn_rec.get("comment"):
            bits.append(espn_rec["comment"])
        return [{
            "kind": "doubtful" if is_doubt else "questionable",
            "label": label, "tone": "bad" if is_doubt else "warn",
            "detail": " ".join(bits), "source": src_label,
            "published": published, "age": age, "play_probability": prob,
        }]

    # No game-status designation, but a practice signal exists (midweek).
    if practice in ("dnp", "limited"):
        return [{
            "kind": "practice",
            "label": _PRACTICE_LABEL[practice] + body_txt,
            "tone": "warn" if practice == "dnp" else "info",
            "detail": f"{_PRACTICE_LABEL[practice]} this week (per {practice_source})."
                      f"{(' ' + espn_rec['comment']) if espn_rec and espn_rec.get('comment') else ''}",
            "source": practice_source or src_label,
            "published": published, "age": age, "play_probability": None,
        }]
    return []


def top_signal_chip(signals: list) -> Optional[dict]:
    """The single most important signal as a Start/Sit row chip:
    {label, kind, note, play_probability} - the qb_situation chip shape."""
    if not signals:
        return None
    s = signals[0]
    return {
        "label": s["label"], "kind": s["kind"], "note": s["detail"],
        "play_probability": s.get("play_probability"),
    }


# ── RSS blurbs (pure parsers) ────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", _html.unescape(raw or ""))
    txt = re.sub(r"\[\s*…\s*\]|\[\.\.\.\]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def _rfc822_to_iso(pub: str) -> str:
    if not pub:
        return ""
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(pub)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return ""


def parse_rss_blurbs(xml_text: str, source: str) -> list:
    """Parse a standard RSS 2.0 feed into news-item dicts (newest-first),
    same shape news.py items use: headline/description/published/age/url/source."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    items = []
    for item_el in root.iter("item"):
        title = _clean_text(item_el.findtext("title") or "")
        link = (item_el.findtext("link") or "").strip()
        if not title or not link:
            continue
        desc = _clean_text(item_el.findtext("description") or "")
        published = _rfc822_to_iso((item_el.findtext("pubDate") or "").strip())
        items.append({
            "headline": title,
            "description": desc[:260] + ("..." if len(desc) > 260 else ""),
            "published": published,
            "age": _age_label(published) if published else "",
            "url": link,
            "source": source,
        })
    items.sort(key=lambda it: it["published"], reverse=True)
    return items


def match_blurbs(items: list, player_name: str, limit: int = 4) -> list:
    """Keep blurbs that name the player (first + last name in the
    headline or description), newest first."""
    parts = _norm_name(player_name).split()
    if len(parts) < 2:
        return []
    first, last = parts[0], parts[-1]
    out = []
    for it in items or []:
        text = _norm_name((it.get("headline") or "") + " " + (it.get("description") or ""))
        if first in text and last in text:
            out.append(it)
        if len(out) >= limit:
            break
    return out


# ── Bulk fetches (lazy httpx, cached, fail-soft) ─────────────────────────────

def _bulk(key: str, url: str, parser, fail_code: str):
    """Fetch + parse a bulk feed once per TTL. Returns (parsed, error).
    Failures are cached briefly so a dead feed cannot stall every request."""
    now = time.time()
    cached = _CACHE.get(key)
    if cached:
        age = now - cached[0]
        if cached[2] is None and age < _TTL:
            return cached[1], None
        if cached[2] is not None and age < _FAIL_TTL:
            return cached[1], cached[2]
    try:
        import httpx
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.get(url, headers=_HEADERS, follow_redirects=True)
        if not r.is_success:
            logger.warning("[news_signals] %s failed: HTTP %s", url, r.status_code)
            _CACHE[key] = (now, parser(None), fail_code)
            return _CACHE[key][1], fail_code
        parsed = parser(r.text if "rss" in key or "feed" in key else r.json())
        _CACHE[key] = (now, parsed, None)
        return parsed, None
    except Exception as e:
        logger.warning("[news_signals] %s failed: %s: %s", url, type(e).__name__, e)
        _CACHE[key] = (now, parser(None), fail_code)
        return _CACHE[key][1], fail_code


def get_espn_injuries_by_name() -> tuple[dict, Optional[str]]:
    """Bulk ESPN injuries, parsed to {norm_name: record}. One fetch per
    TTL for the whole process - Start/Sit calls this, never per row."""
    return _bulk("espn_injuries", _ESPN_INJURIES_URL,
                 lambda payload: parse_espn_injuries(payload or {}),
                 SRC_ESPN_INJ)


def _get_rss_items() -> tuple[list, list]:
    roto, roto_err = _bulk("rss_rotoballer", _ROTO_URL,
                           lambda txt: parse_rss_blurbs(txt or "", "Fantasy · RotoBaller")
                           if txt else [], SRC_ROTO)
    pft, pft_err = _bulk("rss_pft", _PFT_URL,
                         lambda txt: parse_rss_blurbs(txt or "", "ProFootballTalk")
                         if txt else [], SRC_PFT)
    failed = [c for c, e in ((SRC_ROTO, roto_err), (SRC_PFT, pft_err)) if e]
    return roto + pft, failed


def _sleeper_feed_rec(sleeper_id: str) -> tuple[Optional[dict], Optional[str]]:
    """One player's raw Sleeper feed entry via the app's cached accessor."""
    if not sleeper_id:
        return None, None
    try:
        from dashboard_services.api import get_nfl_players
        feed = get_nfl_players() or {}
        rec = feed.get(str(sleeper_id))
        return (rec if isinstance(rec, dict) else None), None
    except Exception as e:
        logger.warning("[news_signals] Sleeper players fetch failed: %s", e)
        return None, SRC_SLEEPER


# ── Public API ───────────────────────────────────────────────────────────────

def get_player_news_signals(player_name: str, team: str = "",
                            sleeper_id: str = "",
                            sleeper_feed_rec: Optional[dict] = None) -> dict:
    """Structured signals + fantasy blurbs for one player.

    Returns {"signals": [...], "blurbs": [...], "sources_failed": [...]}.
    ``sleeper_feed_rec`` lets callers that already hold the Sleeper feed
    entry (Start/Sit rows) pass it in and skip the accessor lookup.
    """
    failed: list = []
    inj_map, inj_err = get_espn_injuries_by_name()
    if inj_err:
        failed.append(inj_err)
    espn_rec = inj_map.get(_norm_name(player_name)) if player_name else None
    if espn_rec and team and espn_rec.get("team") and \
            espn_rec["team"] != team.strip().upper():
        espn_rec = None  # same name, different team: not this player

    if sleeper_feed_rec is not None:
        sl = sleeper_record(sleeper_feed_rec)
    else:
        raw, sl_err = _sleeper_feed_rec(sleeper_id)
        if sl_err:
            failed.append(sl_err)
        sl = sleeper_record(raw)

    signals = derive_signals(espn_rec, sl)

    rss_items, rss_failed = _get_rss_items()
    failed.extend(rss_failed)
    blurbs = match_blurbs(rss_items, player_name, limit=4) if player_name else []

    return {"signals": signals, "blurbs": blurbs,
            "sources_failed": list(dict.fromkeys(failed))}


def top_chip_for_player(player_name: str, sleeper_feed_rec: Optional[dict] = None,
                        team: str = "") -> Optional[dict]:
    """Top signal chip for a Start/Sit row, or None. Uses the single
    cached bulk ESPN injuries fetch plus the row's own Sleeper feed entry.
    Signal-only: callers render the chip, nothing here alters any score."""
    try:
        inj_map, _ = get_espn_injuries_by_name()
        espn_rec = inj_map.get(_norm_name(player_name)) if player_name else None
        if espn_rec and team and espn_rec.get("team") and \
                espn_rec["team"] != team.strip().upper():
            espn_rec = None
        signals = derive_signals(espn_rec, sleeper_record(sleeper_feed_rec))
        return top_signal_chip(signals)
    except Exception:
        return None
