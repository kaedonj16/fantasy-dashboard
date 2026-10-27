"""Division framing for the AI weekly story (the recap's "Weekly Story").

The story prose is LLM-narrated, but it narrates a deterministic payload
under a hard no-invented-facts contract, so the division behaviour is
pinned at that surface: in leagues with divisions the payload carries the
division races (leaders, division records, games back, who also leads the
league), this week's division games are flagged, and the prompt gains the
division framing rules. In leagues without divisions the payload (and
therefore the prompt built from it) is exactly what it was before.

Fixture league mirrors the shape that motivated this: two 6-1 teams, one
leading each division. Alpha leads the East AND the league (most PF);
Foxtrot leads the West but sits #2 overall. In Week 7 both leaders beat
the 5-1 team behind them in a division game to take sole possession.
"""
import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

from dashboard_services.ai import weekly_recap as recap  # noqa: E402
from dashboard_services.ai.weekly_recap import build_weekly_recap_payload  # noqa: E402

TEAMS = {
    1: ("Alpha", 1), 7: ("Echo", 1), 6: ("Delta", 1), 5: ("Charlie", 1), 4: ("Bravo", 1),
    2: ("Foxtrot", 2), 3: ("Golf", 2), 8: ("Hotel", 2), 9: ("India", 2), 10: ("Juliett", 2),
}
DIVISION_BY_RID = {rid: div for rid, (_name, div) in TEAMS.items()}
DIVISION_NAMES = {1: "East", 2: "West"}
TEAM_BY_RID = {str(rid): name for rid, (name, _div) in TEAMS.items()}
# (week, winner rid, winner pts, loser rid, loser pts); verified to land on
# Alpha 6-1 / Foxtrot 6-1 / Echo 5-2 / Golf 5-2 with the PF order below.
GAMES = [[1, 6, 119, 3, 99], [1, 4, 126, 10, 93], [1, 7, 136, 5, 104], [1, 9, 117, 2, 98],
         [1, 1, 133, 8, 92], [2, 6, 131, 10, 103], [2, 3, 117, 5, 97], [2, 2, 127, 4, 91],
         [2, 7, 132, 8, 102], [2, 1, 122, 9, 96], [3, 6, 120, 5, 90], [3, 2, 132, 10, 101],
         [3, 3, 136, 8, 95], [3, 1, 127, 4, 89], [3, 7, 128, 9, 100], [4, 2, 137, 6, 94],
         [4, 5, 116, 8, 88], [4, 1, 132, 10, 99], [4, 3, 132, 9, 93], [4, 7, 117, 4, 104],
         [5, 8, 121, 6, 98], [5, 2, 133, 1, 92], [5, 9, 112, 5, 103], [5, 7, 122, 10, 97],
         [5, 3, 128, 4, 91], [6, 1, 142, 6, 102], [6, 8, 117, 9, 96], [6, 2, 129, 7, 90],
         [6, 5, 131, 4, 101], [6, 3, 117, 10, 95], [7, 9, 122, 6, 89], [7, 1, 138, 7, 100],
         [7, 8, 113, 4, 94], [7, 2, 125, 3, 88], [7, 10, 127, 5, 99]]
LEAGUE = {"name": "Fixture League",
          "settings": {"playoff_week_start": 15, "playoff_teams": 6}}


def _df_weekly():
    rows = []
    for i, (wk, w, wp, l, lp) in enumerate(GAMES):
        mid = wk * 10 + (i % 5)
        for rid, pts, opp in ((w, wp, lp), (l, lp, wp)):
            rows.append({"week": wk, "matchup_id": mid, "roster_id": str(rid),
                         "points": float(pts), "points_against": float(opp),
                         "finalized": True, "owner": TEAM_BY_RID[str(rid)]})
    return pd.DataFrame(rows)


def _matchups_by_week():
    by_week = {}
    for i, (wk, w, wp, l, lp) in enumerate(GAMES):
        mid = wk * 10 + (i % 5)
        by_week.setdefault(wk, []).append({
            "matchup_id": mid,
            "left": {"roster_id": str(w), "pts_total": float(wp),
                     "username": TEAM_BY_RID[str(w)]},
            "right": {"roster_id": str(l), "pts_total": float(lp),
                      "username": TEAM_BY_RID[str(l)]},
        })
    return by_week


def _payload(**kw):
    return build_weekly_recap_payload(_df_weekly(), _matchups_by_week(), 7,
                                      TEAM_BY_RID, LEAGUE, **kw)


def _divisions_payload():
    return _payload(division_by_rid=DIVISION_BY_RID, division_names=DIVISION_NAMES)


# ── division races: facts ────────────────────────────────────────────────────

def test_division_races_name_leaders_records_and_games_back():
    races = {r["name"]: r for r in _divisions_payload()["divisions"]}
    assert set(races) == {"East", "West"}

    east = races["East"]
    assert east["leader"] == "Alpha"
    assert east["leader_record"] == "6-1"
    assert east["leader_division_record"] == "3-0"
    assert east["tight"] is True  # Echo one game back
    assert east["teams"][1]["team"] == "Echo"
    assert east["teams"][1]["games_back"] == 1.0

    west = races["West"]
    assert west["leader"] == "Foxtrot"
    assert west["leader_record"] == "6-1"
    assert west["leader_division_record"] == "2-1"
    assert west["tight"] is True  # Golf one game back
    assert west["teams"][1]["team"] == "Golf"
    assert west["teams"][1]["games_back"] == 1.0


def test_division_only_leader_is_never_the_overall_leader():
    payload = _divisions_payload()
    # Overall #1 is Alpha (PF tiebreak); Foxtrot is #2 overall...
    assert payload["teams"][0]["team"] == "Alpha"
    fox = next(t for t in payload["teams"] if t["team"] == "Foxtrot")
    assert fox["rank_after"] == 2
    # ...but Foxtrot leads the West, and the payload says exactly that.
    west = next(r for r in payload["divisions"] if r["name"] == "West")
    assert west["leader"] == "Foxtrot"
    assert west["leader_leads_league"] is False
    assert west["teams"][0]["overall_rank"] == 2


def test_leader_of_both_league_and_division_reads_correctly():
    east = next(r for r in _divisions_payload()["divisions"] if r["name"] == "East")
    assert east["leader"] == "Alpha"
    assert east["leader_leads_league"] is True
    assert east["teams"][0]["overall_rank"] == 1


def test_division_order_and_records_match_the_recap_standings_card():
    """Site parity: the payload's division order and records must be the
    ones the recap page's standings card renders for the same frame
    (overall wins, then division win%, then PF)."""
    from dashboard_services.pages.recap_page import _recap_standings_rows

    card_rows = _recap_standings_rows(_df_weekly(), DIVISION_BY_RID, True)
    races = {r["name"]: r for r in _divisions_payload()["divisions"]}
    for div, name in DIVISION_NAMES.items():
        card_order = [row["owner"] for row in card_rows if row["division"] == div]
        story_order = [row["team"] for row in races[name]["teams"]]
        assert story_order == card_order
    # The card prints overall + division record as one string, '6-1 (2-1)';
    # the payload's split fields must compose to the same facts.
    fox_card = next(r for r in card_rows if r["owner"] == "Foxtrot")
    west = races["West"]
    assert fox_card["record"] == f'{west["leader_record"]} ({west["leader_division_record"]})'


def test_payload_teams_carry_their_division_record():
    payload = _divisions_payload()
    by_team = {t["team"]: t for t in payload["teams"]}
    assert by_team["Foxtrot"]["division_record"] == "2-1"
    assert by_team["Alpha"]["division_record"] == "3-0"
    assert by_team["Golf"]["division_record"] == "3-1"


def test_this_weeks_division_games_are_flagged():
    payload = _divisions_payload()
    flagged = {(m["team_a"], m["team_b"]): m for m in payload["matchups"]
               if m.get("division_game")}
    # Week 7: both 5-1 division leaders met the team behind them.
    assert flagged[("Alpha", "Echo")]["division_name"] == "East"
    assert flagged[("Foxtrot", "Golf")]["division_name"] == "West"
    assert len(flagged) == 2  # the other three Week 7 games are cross-division


# ── prompt: division framing only in divisions leagues ──────────────────────

class _FakeResponses:
    seen = None

    def create(self, model=None, input=None, text=None):
        _FakeResponses.seen = input

        class _Resp:
            output_text = json.dumps(
                {"headline": "H", "paragraphs": ["P"], "looking_ahead": ""})
        return _Resp()


class _FakeClient:
    responses = _FakeResponses()


def _prompt_for(payload, monkeypatch):
    monkeypatch.setattr(recap, "get_ai_client", lambda: _FakeClient())
    recap._generate_ai_storyline(payload)
    return _FakeResponses.seen[1]["content"]


def test_divisions_prompt_carries_races_and_framing_rules(monkeypatch):
    prompt = _prompt_for(_divisions_payload(), monkeypatch)
    assert "Division races" in prompt
    assert '"leader": "Foxtrot"' in prompt
    assert '"leader_leads_league": false' in prompt
    assert "Never call a team the league leader" in prompt
    assert "division_game" in prompt
    # No em dashes anywhere in the added copy.
    block = prompt[prompt.index("Division races"):]
    assert "—" not in block.split("Use the h2h")[0]


# ── division-less leagues: nothing changes ──────────────────────────────────

def test_divisionless_payload_has_no_division_surface():
    payload = _payload()
    assert "divisions" not in payload
    assert "division" not in json.dumps(payload)
    for m in payload["matchups"]:
        assert "division_game" not in m


def test_divisionless_payload_identical_with_empty_or_single_division_maps():
    base = json.dumps(_payload(), sort_keys=True)
    empty = json.dumps(_payload(division_by_rid={}, division_names={}), sort_keys=True)
    single = json.dumps(
        _payload(division_by_rid={rid: 1 for rid in DIVISION_BY_RID},
                 division_names={1: "Only"}),
        sort_keys=True)
    assert base == empty == single


def test_divisionless_prompt_has_no_division_block(monkeypatch):
    prompt = _prompt_for(_payload(), monkeypatch)
    assert "Division races" not in prompt
    assert "leader_leads_league" not in prompt


# ── story cache: divisions leagues regenerate, others keep their key ───────

def _seed_story(monkeypatch, tmp_path, html_text):
    from dashboard_services.ai import cache
    monkeypatch.setattr(recap, "AI_CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "AI_CACHE_DIR", tmp_path)
    df = _df_weekly()
    scored = df[(df["finalized"] == True) & (df["week"] <= 7)]
    raw = scored[["week", "roster_id", "points"]].sort_values(
        ["week", "roster_id"]).to_json()
    import hashlib
    revision = hashlib.sha256(raw.encode()).hexdigest()[:12]
    key = recap._recap_cache_key("L1", 2026, 7, "sleeper", revision)
    cache.save_cached_ai_text(key, html_text)
    return key


def test_story_cache_key_only_changes_for_divisions_leagues(monkeypatch, tmp_path):
    _seed_story(monkeypatch, tmp_path, "<div>old overall-framed story</div>")
    monkeypatch.setattr(recap, "ai_available", lambda: True)
    calls = []

    def generate(_payload):
        calls.append(1)
        return {"headline": "New story", "paragraphs": ["Fresh."], "looking_ahead": ""}

    monkeypatch.setattr(recap, "_generate_ai_storyline", generate)

    # Division-less: the pre-seeded story is served as-is, no regeneration.
    html_text, _ = recap.get_weekly_ai_recap(
        _df_weekly(), _matchups_by_week(), 7, TEAM_BY_RID, LEAGUE, "L1", 2026)
    assert "old overall-framed story" in html_text
    assert calls == []

    # Divisions: the old overall-framed story is NOT served; the story
    # regenerates once under the division-marked key.
    html_text, _ = recap.get_weekly_ai_recap(
        _df_weekly(), _matchups_by_week(), 7, TEAM_BY_RID, LEAGUE, "L1", 2026,
        division_by_rid=DIVISION_BY_RID, division_names=DIVISION_NAMES)
    assert "old overall-framed story" not in html_text
    assert "New story" in html_text
    assert calls == [1]
