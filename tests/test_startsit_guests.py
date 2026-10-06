"""Tests for Start/Sit guests (advice + Lab) and the player-modal strip.

Guests are non-roster players evaluated through the same Start/Sit row
pipeline and the same Lineup Lab profile machinery as roster players.
They never take roster flags, never enter lineup advice, and are never
written into roster data. The modal strip distills the same options
payload into a one-line verdict, with explicit unavailable states instead
of fabricated verdicts.
"""

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


# ── Start/Sit options + strip fixtures ────────────────────────────────────

WEEK = 5
PROJ = {"rb1": 20.0, "rb2": 14.0, "rb3": 11.0, "rb4": 8.0, "qb1": 22.0,
        "rbR": 17.0, "rbF": 9.0, "rbBye": 25.0, "k1": 7.0}
NAMES = {"rb1": "Runner One", "rb2": "Runner Two", "rb3": "Runner Three",
         "rb4": "Runner Four", "qb1": "Quarter Back", "rbR": "Rival Runner",
         "rbF": "Free Runner", "rbBye": "Bye Runner",
         "rbNoProj": "Ghost Runner", "k1": "Kicker One"}
TEAMS = {"rb1": "KC", "rb2": "BUF", "rb3": "MIA", "rb4": "DEN", "qb1": "KC",
         "rbR": "DAL", "rbF": "NYJ", "rbBye": "SEA", "rbNoProj": "GB",
         "k1": "KC"}
POS = {p: ("QB" if p == "qb1" else "K" if p == "k1" else "RB") for p in NAMES}
SCHED = [{"home": "KC", "away": "BUF"}, {"home": "MIA", "away": "DEN"},
         {"home": "DAL", "away": "NYJ"}, {"home": "GB", "away": "CHI"}]

BASE = "/api/start-sit-options?platform=sleeper&league_id=L1&season=2026"
STRIP = "/api/player-startsit-strip?platform=sleeper&league_id=L1&season=2026&player_id="


def _ss_ctx(viewer_roster_id=1):
    idx = {p: {"pos": POS[p], "position": POS[p], "team": TEAMS[p],
               "full_name": NAMES[p], "name": NAMES[p], "injury_status": ""}
           for p in NAMES}
    return {
        "viewer": {"viewer_roster_id": viewer_roster_id},
        "current_week": WEEK,
        "rosters": [
            {"roster_id": 1, "owner_id": "me-user",
             "players": ["rb1", "rb2", "rb3", "rb4", "qb1"],
             "reserve": [], "taxi": [], "metadata": {"team_name": "My Team"}},
            {"roster_id": 2, "owner_id": "rival-user", "players": ["rbR"],
             "reserve": [], "taxi": [], "metadata": {"team_name": "Rival Team"}},
        ],
        "users": [{"user_id": "me-user", "display_name": "Kaedon"},
                  {"user_id": "rival-user", "display_name": "Rival Rick"}],
        "players_index": idx,
        "players": idx,
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                             "BN", "BN", "BN"],
        "scoring_settings": {},
        "raw_scoring_settings": {},
        "proj_by_week": {WEEK: {"projections": dict(PROJ)}},
    }


@pytest.fixture
def ss_client(offline_client, monkeypatch):
    import app as appmod
    import data_building.start_score_bundle as bundle_mod
    import data_building.weekly_metrics as wm_mod
    import utils.game_conditions as gc_mod
    import utils.utils as uu_mod

    appmod._SS_STRIP_CACHE.clear()
    monkeypatch.setattr(appmod, "_session_signed_in", lambda: True)
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: _ss_ctx())
    rows = [{"id": p, "name": NAMES[p], "position": POS[p], "team": TEAMS[p],
             "value": 5000, "sf_value": 5000, "pos_rank_label": "RB1"}
            for p in NAMES]
    monkeypatch.setattr(appmod, "get_model_value_table_cached",
                        lambda *a, **k: rows)
    monkeypatch.setattr(appmod, "_matchup_rank_table",
                        lambda *a, **k: ({}, 32, None, None))
    monkeypatch.setattr(appmod, "_fpts_against_effective", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_team_play_volume", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_season_weekstat_points",
                        lambda *a, **k: ((), ()))
    monkeypatch.setattr(appmod, "_load_season_weekly_points", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_season_snap_totals", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "build_projections_by_week", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "start_score_pos_anchors", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_ss_game_env", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_ss_qb_situation", lambda *a, **k: None)
    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda *a, **k: None)
    monkeypatch.setattr(bundle_mod, "load_start_score_bundles",
                        lambda *a, **k: {})
    monkeypatch.setattr(wm_mod, "get_usage_trends", lambda *a, **k: {})
    monkeypatch.setattr(gc_mod, "build_week_conditions", lambda *a, **k: {})
    monkeypatch.setattr(uu_mod, "load_week_projection", lambda *a, **k: {})
    monkeypatch.setattr(uu_mod, "load_week_sched", lambda *a, **k: list(SCHED))
    yield offline_client
    appmod._SS_STRIP_CACHE.clear()


def _rb_rows(payload):
    return {p["player_id"]: p for p in payload["positions"]["RB"]}


# ── Advice guests ─────────────────────────────────────────────────────────

def test_guest_joins_position_group_with_owner_tag(ss_client):
    r = ss_client.get(BASE + "&guests=rbR,rbF")
    assert r.status_code == 200
    d = r.get_json()
    rows = _rb_rows(d)
    rival = rows["rbR"]
    assert rival["guest"] is True
    assert rival["owner_label"] == "Rival Rick"
    assert rival["proj_pts"] == 17.0
    fa = rows["rbF"]
    assert fa["guest"] is True
    assert fa["owner_label"] == "Free Agent"
    assert {g["player_id"] for g in d["guests"]} == {"rbR", "rbF"}


def test_guest_would_start_displaces_marginal_starter(ss_client):
    d = ss_client.get(BASE + "&guests=rbR").get_json()
    g = _rb_rows(d)["rbR"]
    assert g["would_start"] is True
    assert g["guest_slot"] == "RB2"
    assert g["guest_note"] == "Starts over your RB2 · Runner Two"
    assert g["guest_vs_proj"] == 14.0


def test_guest_bench_note_names_players_ahead(ss_client):
    d = ss_client.get(BASE + "&guests=rbF").get_json()
    g = _rb_rows(d)["rbF"]
    assert g["would_start"] is False
    assert g["guest_rank"] == 4
    assert g["guest_note"] == "Behind your RB2 and RB3"


def test_guests_never_take_roster_flags_or_advice(ss_client):
    plain = ss_client.get(BASE).get_json()
    withg = ss_client.get(BASE + "&guests=rbR,rbF").get_json()
    for payload in (plain, withg):
        rows = _rb_rows(payload)
        assert rows["rb1"]["start"] is True
        assert rows["rb2"]["start"] is True
        assert rows["rb3"]["flex_start"] is True
        assert rows["rb4"]["start"] is False
    # Roster start flags and optimal-lineup advice are identical either way.
    def flags(payload):
        return {pid: (p.get("start"), p.get("flex_start"))
                for pid, p in _rb_rows(payload).items() if not p.get("guest")}
    assert flags(plain) == flags(withg)
    assert plain["lineup_advice"] == withg["lineup_advice"]
    for swap in withg["lineup_advice"] or []:
        assert "rbR" not in str(swap) and "rbF" not in str(swap)


def test_guest_on_bye_gets_bye_note(ss_client):
    d = ss_client.get(BASE + "&guests=rbBye").get_json()
    g = _rb_rows(d)["rbBye"]
    assert g["guest"] is True
    assert g["would_start"] is False
    assert g["guest_note"] == "On bye this week"


def test_guest_without_position_group_is_listed_no_slot(ss_client):
    # The league starts no kickers, so a K guest has no group to join; the
    # guest list still names him with an explicit no_slot marker.
    d = ss_client.get(BASE + "&guests=k1").get_json()
    assert "K" not in d["positions"]
    entry = {g["player_id"]: g for g in d["guests"]}["k1"]
    assert entry["no_slot"] is True
    assert entry["owner_label"] == "Free Agent"


def test_rostered_player_passed_as_guest_stays_a_roster_row(ss_client):
    d = ss_client.get(BASE + "&guests=rb1").get_json()
    assert not _rb_rows(d)["rb1"].get("guest")
    assert d["guests"] == []


def test_parse_guest_pids_dedupes_and_caps():
    import app as appmod
    assert appmod._parse_guest_pids("a,b, a,,c") == ["a", "b", "c"]
    assert appmod._parse_guest_pids("") == []
    assert appmod._parse_guest_pids(None) == []
    assert len(appmod._parse_guest_pids("1,2,3,4,5,6,7")) == appmod._SS_GUEST_CAP


def test_guests_force_live_path_over_bundles(ss_client, monkeypatch):
    # Precomputed bundles cover the roster only. Without guests the bundle
    # projection wins; with a guest the whole request takes the live path
    # so roster and guest rows stay comparable.
    import data_building.start_score_bundle as bundle_mod
    roster = ["rb1", "rb2", "rb3", "rb4", "qb1"]
    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda *a, **k: "ppr")
    monkeypatch.setattr(
        bundle_mod, "load_start_score_bundles",
        lambda *a, **k: {pid: {"proj_pts": 99.0} for pid in roster})
    plain = ss_client.get(BASE).get_json()
    assert _rb_rows(plain)["rb1"]["proj_pts"] == 99.0
    withg = ss_client.get(BASE + "&guests=rbF").get_json()
    assert _rb_rows(withg)["rb1"]["proj_pts"] == 20.0
    assert _rb_rows(withg)["rbF"]["proj_pts"] == 9.0


# ── Modal strip ───────────────────────────────────────────────────────────

def test_strip_guest_would_start(ss_client):
    d = ss_client.get(STRIP + "rbR").get_json()
    assert d["state"] == "ok"
    assert d["verdict"] == "Would start for you"
    assert d["tone"] == "start"
    assert d["slot"] == "RB2"
    assert "17.0" in d["reason"] and "14.0" in d["reason"]
    assert d["compare"] is True


def test_strip_guest_bench(ss_client):
    d = ss_client.get(STRIP + "rbF").get_json()
    assert d["state"] == "ok"
    assert d["verdict"] == "Bench for you"
    assert d["tone"] == "bench"
    assert d["slot"] == "RB4"
    assert "Behind your RB2 and RB3" in d["reason"]


def test_strip_roster_starter_and_bench(ss_client):
    d = ss_client.get(STRIP + "rb1").get_json()
    assert d["state"] == "ok" and d["verdict"] == "Start" and d["slot"] == "RB1"
    assert "top RB score in your lineup" in d["reason"]
    d = ss_client.get(STRIP + "rb4").get_json()
    assert d["state"] == "ok" and d["verdict"] == "Bench" and d["slot"] == "RB4"
    assert "Runner Three" in d["reason"]  # the flex starter he sits behind


def test_strip_unavailable_states_are_explicit(ss_client):
    d = ss_client.get(STRIP + "rbBye").get_json()
    assert d["state"] == "unavailable"
    assert d["text"] == "On bye in Week 5"
    assert "verdict" not in d
    d = ss_client.get(STRIP + "rbNoProj").get_json()
    assert d["state"] == "unavailable"
    assert d["text"] == "No projection for Week 5"
    assert "verdict" not in d


def test_strip_hidden_without_position_group(ss_client):
    d = ss_client.get(STRIP + "k1").get_json()
    assert d == {"state": "hidden"}


def test_strip_hidden_without_viewer_team(ss_client, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: _ss_ctx(viewer_roster_id=None))
    assert ss_client.get(STRIP + "rb1").get_json() == {"state": "hidden"}
    assert ss_client.get(BASE).status_code == 409


def _ss_ctx_with_injury(pid, status):
    ctx = _ss_ctx()
    ctx["players_index"][pid]["injury_status"] = status
    return ctx


def test_strip_guest_ruled_out_gets_injury_state(ss_client, monkeypatch):
    # A guest who is OUT must not get a start/sit "Bench for you" verdict.
    import app as appmod
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: _ss_ctx_with_injury("rbR", "Out"))
    appmod._SS_STRIP_CACHE.clear()
    d = ss_client.get(STRIP + "rbR").get_json()
    assert d["state"] == "unavailable"
    assert d["text"] == "Ruled out for Week 5"
    assert "verdict" not in d
    assert d["compare"] is True


def test_strip_rostered_ruled_out_gets_injury_state(ss_client, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: _ss_ctx_with_injury("rb4", "IR"))
    appmod._SS_STRIP_CACHE.clear()
    d = ss_client.get(STRIP + "rb4").get_json()
    assert d["state"] == "unavailable"
    assert d["text"] == "Ruled out for Week 5"
    assert "verdict" not in d


def test_strip_questionable_keeps_startsit_verdict(ss_client, monkeypatch):
    # Questionable/Doubtful may still play, so the verdict stays.
    import app as appmod
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: _ss_ctx_with_injury("rbF", "Questionable"))
    appmod._SS_STRIP_CACHE.clear()
    d = ss_client.get(STRIP + "rbF").get_json()
    assert d["state"] == "ok"
    assert d["verdict"] == "Bench for you"


def test_strip_hidden_when_signed_out(ss_client, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "_session_signed_in", lambda: False)
    assert ss_client.get(STRIP + "rb1").get_json() == {"state": "hidden"}
    assert ss_client.get(STRIP + "rb1").status_code == 200


# ── Lineup Lab guests ─────────────────────────────────────────────────────

import data_building.lineup_lab as lab_mod
import data_building.player_distributions as pd_mod
import utils.fantasy_scoring as fs_mod
import utils.utils as utils_mod
import dashboard_services.api as api_mod

_LAB_PROJ = {
    "1": 22.5, "2": 18.0, "3": 16.9, "4": 14.2, "5": 9.0, "6": 12.0,
    "7": 3.0, "8": 8.0, "9": 20.0, "10": 15.0,
    "11": 13.0,  # rival WR guest
    "12": 10.0,  # free-agent RB guest
    "15": 12.0,  # bye-week WR guest (NYJ plays nobody in the fake sched)
}
_LAB_POS = {"1": "QB", "2": "RB", "3": "WR", "4": "WR", "5": "TE",
            "6": "RB", "7": "K", "8": "DEF", "9": "QB", "10": "RB",
            "11": "WR", "12": "RB", "14": "RB", "15": "WR"}
_LAB_TEAM = {"1": "LAR", "2": "DET", "3": "LAR", "4": "CIN", "5": "KC",
             "6": "SF", "7": "DAL", "8": "BUF", "9": "BUF", "10": "MIA",
             "11": "CIN", "12": "DET", "14": "DET", "15": "NYJ"}


def _lab_ctx():
    idx = {pid: {"position": _LAB_POS[pid], "pos": _LAB_POS[pid],
                 "team": _LAB_TEAM[pid], "full_name": f"Player {pid}",
                 "injury_status": ""}
           for pid in _LAB_POS}
    return {
        "current_week": 4,
        "rosters": [
            {"roster_id": 7, "owner_id": "u1",
             "players": ["1", "2", "3", "4", "5", "6", "7", "8"],
             "reserve": [], "taxi": []},
            {"roster_id": 3, "owner_id": "u2",
             "players": ["9", "10", "11"], "reserve": [], "taxi": []},
        ],
        "users": [{"user_id": "u2", "display_name": "Pittsburgh Pilots"}],
        "players_index": idx,
        "players": {},
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF",
                             "BN", "BN"],
        "raw_scoring_settings": {},
        "scoring_settings": {},
    }


def _lab_matchups(league_id, week):
    return [
        {"roster_id": 7, "matchup_id": 1,
         "starters": ["1", "2", "3", "5", "6", "7", "8"],
         "players": ["1", "2", "3", "4", "5", "6", "7", "8"]},
        {"roster_id": 3, "matchup_id": 1,
         "starters": ["9", "10", "11"], "players": ["9", "10", "11"]},
    ]


def _lab_profiles(requests, season, week):
    out = {}
    for req in requests:
        pid = str(req["player_id"])
        mean = float(req["mean"])
        out[pid] = {"player_id": pid, "pos": req["pos"], "mean": mean,
                    "std": round(2.0 + 0.42 * mean, 2), "skew_alpha": 2.0,
                    "dud_risk": 0.0, "n_games": 3.0, "factors": {}}
    return out


@pytest.fixture
def lab_mocks(monkeypatch):
    monkeypatch.setattr(api_mod, "get_matchups", _lab_matchups)
    monkeypatch.setattr(pd_mod, "build_profiles", _lab_profiles)
    monkeypatch.setattr(pd_mod, "correlation_pairs",
                        lambda pids, season, ctx=None: {})
    monkeypatch.setattr(utils_mod, "load_week_projection",
                        lambda season, week: {})
    monkeypatch.setattr(utils_mod, "load_week_sched", lambda season, week: [
        {"home": "LAR", "away": "SF", "gameDate": "2026-09-27"},
        {"home": "DET", "away": "KC", "gameDate": "2026-09-27"},
        {"home": "CIN", "away": "MIA", "gameDate": "2026-09-27"},
        {"home": "DAL", "away": "BUF", "gameDate": "2026-09-28"},
    ])
    monkeypatch.setattr(fs_mod, "weekly_projection_points",
                        lambda raw, pid, scoring, pos="": _LAB_PROJ.get(str(pid)))


def _build_lab(guests):
    return lab_mod.build_lineup_lab_payload(
        ctx=_lab_ctx(), league_id="123", viewer_roster_id=7,
        season=2026, week=4, guest_pids=guests)


def test_lab_guest_entry_uses_shared_profile_machinery(lab_mocks):
    data = _build_lab(["11", "12"])
    guests = {g["player_id"]: g for g in data["guests"]}
    rival = guests["11"]
    assert rival["guest"] is True
    assert rival["owner_label"] == "Pittsburgh Pilots"
    assert rival["available"] is True
    assert rival["proj"] == 13.0
    assert rival["profile"]["mean"] == 13.0
    assert rival["floor"] < 13.0 < rival["ceiling"]
    fa = guests["12"]
    assert fa["owner_label"] == "Free Agent"
    assert fa["available"] is True


def test_lab_guests_are_never_written_into_lineup(lab_mocks):
    data = _build_lab(["11", "12"])
    starter_pids = {e["player_id"] for e in data["you"]["lineup"]}
    bench_pids = {b["player_id"] for e in data["you"]["lineup"]
                  for b in (e.get("bench") or [])}
    assert "11" not in starter_pids and "12" not in starter_pids
    assert "11" not in bench_pids and "12" not in bench_pids


def test_lab_rostered_pid_is_not_a_guest(lab_mocks):
    data = _build_lab(["4"])
    assert data["guests"] == []


def test_lab_guest_unavailable_states(lab_mocks):
    data = _build_lab(["13", "14", "15"])
    guests = {g["player_id"]: g for g in data["guests"]}
    assert guests["13"]["available"] is False
    assert "not found" in guests["13"]["unavailable_reason"]
    assert guests["14"]["available"] is False
    assert guests["14"]["unavailable_reason"] == "No projection for Week 4"
    assert guests["15"]["available"] is False
    assert guests["15"]["unavailable_reason"] == "On bye in Week 4"


def test_lab_route_cache_key_separates_guest_sets(offline_client, monkeypatch):
    import app as appmod

    appmod._LAB_PAYLOAD_CACHE.clear()
    monkeypatch.setattr(appmod, "_session_signed_in", lambda: True)
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: {"viewer": {"viewer_roster_id": 7},
                                         "current_week": 4})
    calls = []

    def fake_build(**kwargs):
        calls.append(list(kwargs.get("guest_pids") or []))
        return {"week": kwargs["week"],
                "you": {"roster_id": kwargs["viewer_roster_id"]},
                "n_sims": 2000}

    monkeypatch.setattr(lab_mod, "build_lineup_lab_payload", fake_build)
    url = "/api/lineup-lab?platform=sleeper&league_id=L1&season=2026&week=4"
    try:
        assert offline_client.get(url).status_code == 200
        assert offline_client.get(url + "&guests=g1").status_code == 200
        assert offline_client.get(url + "&guests=g1").status_code == 200
        assert offline_client.get(url + "&guests=g2").status_code == 200
        # No-guest build, one g1 build (repeat served from cache), one g2.
        assert calls == [[], ["g1"], ["g2"]]
    finally:
        appmod._LAB_PAYLOAD_CACHE.clear()


# ── Frontend wiring guards ────────────────────────────────────────────────

def test_waivers_page_guest_wiring():
    from dashboard_services.pages.waivers_page import build_waivers_body
    body = build_waivers_body("sleeper", 2026, "league1", {})
    assert 'id="wvGuestBar"' in body
    assert 'id="wvGuestInput"' in body
    assert 'id="wvGuestChips"' in body
    # Both surfaces request the guest set.
    assert "guests=' + encodeURIComponent(wvGuestParam())" in body
    assert "&guests=' + encodeURIComponent(wvGuestParam())" in body
    # Deep-link seeding + URL sync.
    assert "getAll('guest')" in body
    assert "wvInitGuestsFromUrl" in body
    assert "wvSyncGuestUrl" in body
    # Guest search reuses the canonical player search endpoint.
    assert "fetch('/api/players?q='" in body
    # Lab payload guests are rendered, available or not.
    assert "wvLabRenderGuests" in body
    assert "unavailable_reason" in body


def test_player_modal_strip_wiring():
    from pathlib import Path
    src = Path("static/player_modal.js").read_text()
    assert 'id="pmStartSitStrip"' in src
    assert "_pmLoadStartSitStrip" in src
    assert "/api/player-startsit-strip" in src
    # The Compare link deep-links into Start/Sit with the guest attached.
    assert "waivers?tab=startsit&guest=" in src
    css = Path("static/dashboard.css").read_text()
    assert ".pm-ss-strip" in css
    assert "border-radius: 999px" not in css.split(".pm-ss-strip")[1].split("}")[0]
