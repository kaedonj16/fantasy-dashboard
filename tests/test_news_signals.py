"""Unit tests for the free structured news layer (dashboard_services.news_signals).

Pure-Python (no network, no httpx) - news_signals imports httpx lazily so
the parsers, probability table and signal derivation can be tested in the
httpx-free CI shard, same convention as test_news_blend.py.
"""
import pytest

pytest.importorskip("flask")  # repo convention guard for the lint shard

from dashboard_services.news_signals import (
    PLAY_PROB_DOUBTFUL,
    PLAY_PROB_QUESTIONABLE,
    derive_signals,
    match_blurbs,
    normalize_practice,
    parse_espn_injuries,
    parse_rss_blurbs,
    play_probability,
    practice_from_text,
    sleeper_record,
    top_signal_chip,
)


# ── Fixtures (shaped like the real feeds, verified 2026-10-18) ──────────────

def _espn_entry(name, status, body="Hamstring", comment="", date="2026-10-16T17:00:00Z",
                team_abbr="BUF", espn_id="4361001"):
    first, last = name.split(" ", 1)
    return {
        "id": "1",
        "status": status,
        "date": date,
        "longComment": comment,
        "shortComment": comment,
        "details": {"type": body, "detail": "Not Specified",
                    "returnDate": "2026-10-25",
                    "fantasyStatus": {"description": status.upper()}},
        "athlete": {
            "firstName": first, "lastName": last, "displayName": name,
            "headshot": {"href": f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"},
            "position": {"abbreviation": "WR"},
            "team": {"abbreviation": team_abbr, "displayName": "Buffalo Bills"},
        },
    }


def _payload(*entries):
    return {"injuries": [{"id": "2", "displayName": "Buffalo Bills",
                          "injuries": list(entries)}]}


def _sleeper(status="", body="", practice=""):
    rec = {}
    if status:
        rec["injury_status"] = status
    if body:
        rec["injury_body_part"] = body
    if practice:
        rec["practice_participation"] = practice
    return sleeper_record(rec)


ROTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Player News - RotoBaller</title>
<item>
  <title>Dalton Kincaid Questionable To Return With Ribs Issue</title>
  <link>https://www.rotoballer.com/player-news/dalton-kincaid-questionable/123</link>
  <description>Buffalo Bills tight end Dalton Kincaid (ribs) is listed as questionable to return for Sunday's game. Dawson Knox will operate as the top tight end until Kincaid returns. [&amp;#8230;]</description>
  <pubDate>Sun, 18 Oct 2026 22:42:00 +0000</pubDate>
</item>
<item>
  <title>Tyler Shough Totals Four Touchdowns In Win</title>
  <link>https://www.rotoballer.com/player-news/tyler-shough/124</link>
  <description>Tyler Shough completed 16 of 30 passes for three touchdowns.</description>
  <pubDate>Sun, 18 Oct 2026 23:00:00 +0000</pubDate>
</item>
</channel></rss>"""


# ── ESPN injuries parser ─────────────────────────────────────────────────────

def test_parse_espn_injuries_keys_by_name_with_fields():
    payload = _payload(_espn_entry("Josh Allen", "Questionable", body="Shoulder",
                                   comment="Allen was limited in practice Friday."))
    out = parse_espn_injuries(payload)
    assert list(out) == ["josh allen"]
    rec = out["josh allen"]
    assert rec["status"] == "Questionable"
    assert rec["body_part"] == "Shoulder"
    assert rec["team"] == "BUF"
    assert rec["espn_id"] == "4361001"
    assert rec["return_date"] == "2026-10-25"
    assert "limited in practice" in rec["comment"]


def test_parse_espn_injuries_skips_active_and_malformed():
    payload = _payload(
        _espn_entry("Stefon Diggs", "Active"),
        {"bogus": True},
        _espn_entry("Gabe Davis", "Out"),
    )
    out = parse_espn_injuries(payload)
    assert list(out) == ["gabe davis"]


# ── Practice normalization / text mining ─────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Full Participation in Practice", "full"),
    ("Limited Participation in Practice", "limited"),
    ("Did Not Participate in Practice", "dnp"),
    ("DNP", "dnp"),
    ("", None),
    ("Rest day", None),
])
def test_normalize_practice(raw, expected):
    assert normalize_practice(raw) == expected


@pytest.mark.parametrize("text,expected", [
    ("He was limited in practice Friday and is questionable.", "limited"),
    ("He did not practice Wednesday or Thursday.", "dnp"),
    ("He was a full participant in practice Friday.", "full"),
    ("He caught seven passes for 90 yards.", None),
])
def test_practice_from_text(text, expected):
    assert practice_from_text(text) == expected


# ── Play probability table ───────────────────────────────────────────────────

@pytest.mark.parametrize("status,practice,expected", [
    ("Questionable", "full", PLAY_PROB_QUESTIONABLE["full"]),
    ("Questionable", "limited", PLAY_PROB_QUESTIONABLE["limited"]),
    ("Questionable", "dnp", PLAY_PROB_QUESTIONABLE["dnp"]),
    ("Questionable", None, None),          # no practice info, no number
    ("Doubtful", "full", PLAY_PROB_DOUBTFUL),
    ("Doubtful", None, PLAY_PROB_DOUBTFUL),
    ("Out", "full", 0.0),
    ("Injured Reserve", None, 0.0),
    ("PUP", None, 0.0),
    ("Active", "limited", None),
    ("", "dnp", None),
])
def test_play_probability_table(status, practice, expected):
    assert play_probability(status, practice) == expected


def test_play_probability_constants_match_spec():
    assert PLAY_PROB_QUESTIONABLE == {"full": 0.90, "limited": 0.75, "dnp": 0.50}
    assert PLAY_PROB_DOUBTFUL == 0.10


# ── Signal derivation ────────────────────────────────────────────────────────

def test_signal_questionable_limited_from_sleeper_practice():
    espn = parse_espn_injuries(_payload(
        _espn_entry("Josh Allen", "Questionable", body="Shoulder")))["josh allen"]
    signals = derive_signals(espn, _sleeper(practice="Limited Participation in Practice"))
    assert len(signals) == 1
    s = signals[0]
    assert s["kind"] == "questionable" and s["tone"] == "warn"
    assert s["label"] == "Questionable (shoulder)"
    assert s["play_probability"] == 0.75
    assert "Limited in practice" in s["detail"] and "Sleeper" in s["detail"]


def test_signal_questionable_dnp_from_espn_comment():
    espn = parse_espn_injuries(_payload(_espn_entry(
        "Josh Allen", "Questionable",
        comment="Allen did not practice Friday and is a game-time decision.")))["josh allen"]
    signals = derive_signals(espn, None)
    assert signals[0]["play_probability"] == 0.50
    assert "Did not practice" in signals[0]["detail"]
    assert "ESPN injuries" in signals[0]["detail"]


def test_signal_questionable_full_practice():
    espn = parse_espn_injuries(_payload(
        _espn_entry("Josh Allen", "Questionable")))["josh allen"]
    signals = derive_signals(espn, _sleeper(practice="Full Participation in Practice"))
    assert signals[0]["play_probability"] == 0.90


def test_signal_doubtful():
    espn = parse_espn_injuries(_payload(
        _espn_entry("Josh Allen", "Doubtful", body="Ankle")))["josh allen"]
    signals = derive_signals(espn, None)
    assert signals[0]["kind"] == "doubtful" and signals[0]["tone"] == "bad"
    assert signals[0]["play_probability"] == 0.10


def test_signal_out_is_unavailable_with_zero_probability():
    espn = parse_espn_injuries(_payload(
        _espn_entry("Josh Allen", "Out", body="Knee")))["josh allen"]
    signals = derive_signals(espn, None)
    assert signals[0]["kind"] == "unavailable"
    assert signals[0]["label"] == "Out"
    assert signals[0]["play_probability"] == 0.0


def test_signal_ir_from_sleeper_only():
    signals = derive_signals(None, _sleeper(status="Injured Reserve", body="ACL"))
    assert signals[0]["kind"] == "unavailable"
    assert signals[0]["play_probability"] == 0.0


def test_signal_healthy_player_has_no_signal():
    assert derive_signals(None, None) == []
    assert derive_signals(None, _sleeper(status="Active")) == []
    espn = parse_espn_injuries(_payload(_espn_entry("Josh Allen", "Active")))
    assert espn == {}  # Active rows never reach derivation


def test_signal_practice_only_midweek():
    signals = derive_signals(None, _sleeper(practice="Did Not Participate in Practice",
                                            body="Hamstring"))
    assert signals[0]["kind"] == "practice"
    assert signals[0]["label"].startswith("Did not practice")
    assert signals[0]["play_probability"] is None


def test_top_signal_chip_shape():
    espn = parse_espn_injuries(_payload(
        _espn_entry("Josh Allen", "Questionable")))["josh allen"]
    chip = top_signal_chip(derive_signals(espn, _sleeper(
        practice="Limited Participation in Practice")))
    assert chip["label"] == "Questionable (hamstring)"
    assert chip["kind"] == "questionable"
    assert chip["play_probability"] == 0.75
    assert top_signal_chip([]) is None


# ── RSS blurbs ───────────────────────────────────────────────────────────────

def test_parse_rotoballer_rss_items():
    items = parse_rss_blurbs(ROTO_XML, "Fantasy · RotoBaller")
    assert len(items) == 2
    first = items[0]  # newest first (Shough 23:00 after Kincaid 22:42)
    assert first["headline"].startswith("Tyler Shough")
    kincaid = [i for i in items if "Kincaid" in i["headline"]][0]
    assert kincaid["source"] == "Fantasy · RotoBaller"
    assert "questionable to return" in kincaid["description"]
    assert "[&" not in kincaid["description"] and "&#8230;" not in kincaid["description"]
    assert kincaid["published"].startswith("2026-10-18T22:42")


def test_match_blurbs_requires_full_name():
    items = parse_rss_blurbs(ROTO_XML, "Fantasy · RotoBaller")
    assert [i["headline"] for i in match_blurbs(items, "Dalton Kincaid")] == [
        "Dalton Kincaid Questionable To Return With Ribs Issue"]
    assert match_blurbs(items, "Josh Allen") == []
    assert match_blurbs(items, "Kincaid") == []  # single token never matches


def test_parse_rss_blurbs_bad_xml_is_empty():
    assert parse_rss_blurbs("<not-xml", "X") == []
    assert parse_rss_blurbs("", "X") == []
