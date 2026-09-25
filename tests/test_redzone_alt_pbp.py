"""Tests for Sleeper / ESPN alternate Redzone play-by-play helpers."""
from utils.redzone_alt_pbp import (
    attach_cumulative,
    build_name_indexes,
    extract_espn_pbp_plays,
    extract_espn_scoreboard_lookup,
    extract_sleeper_pbp_plays,
    fetch_alt_pbp_plays,
    parse_pbp_play_stats,
    parse_tank_game_id,
    pids_mentioned_in_text,
)


def test_alternate_pbp_defaults_to_espn_first(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "utils.redzone_alt_pbp.fetch_espn_event_id",
        lambda **kwargs: calls.append("espn") or "event-1",
    )
    monkeypatch.setattr(
        "utils.redzone_alt_pbp.fetch_espn_pbp", lambda *args, **kwargs: {"ok": 1}
    )
    monkeypatch.setattr(
        "utils.redzone_alt_pbp.extract_espn_pbp_plays",
        lambda *args, **kwargs: [{"play_id": "espn-1"}],
    )
    monkeypatch.setattr(
        "utils.redzone_alt_pbp.sleeper_game_id_for_matchup",
        lambda **kwargs: calls.append("sleeper") or "sleeper-1",
    )

    plays = fetch_alt_pbp_plays(
        "20260909_NE@SEA", season=2026, week=1
    )

    assert plays == [{"play_id": "espn-1"}]
    assert calls == ["espn"]


def test_fetch_espn_pbp_prefers_summary_over_cdn(monkeypatch):
    """The web API summary is the freshest source; the CDN gamepackage (which
    can trail live play by minutes) is only touched when summary yields no
    plays."""
    import utils.redzone_alt_pbp as alt

    summary = {"drives": {"current": {"plays": [{"id": "s1", "text": "x"}]}}}
    calls = []
    monkeypatch.setattr(alt, "fetch_espn_pbp_summary",
                        lambda eid, **kw: calls.append("summary") or summary)
    monkeypatch.setattr(alt, "_fetch_espn_pbp_cdn",
                        lambda eid, **kw: calls.append("cdn") or {"cdn": 1})

    assert alt.fetch_espn_pbp("event-1") is summary
    assert calls == ["summary"]  # CDN never fetched when summary has plays


def test_fetch_espn_pbp_falls_back_to_cdn_when_summary_empty(monkeypatch):
    import utils.redzone_alt_pbp as alt

    cdn = {"gamepackageJSON": {"drives": {"current": {"plays": [{"id": "c1"}]}}}}
    calls = []
    # Summary returns no usable drives -> treated as a miss.
    monkeypatch.setattr(alt, "fetch_espn_pbp_summary",
                        lambda eid, **kw: calls.append("summary") or {})
    monkeypatch.setattr(alt, "_fetch_espn_pbp_cdn",
                        lambda eid, **kw: calls.append("cdn") or cdn)

    assert alt.fetch_espn_pbp("event-1") is cdn
    assert calls == ["summary", "cdn"]


def test_espn_payload_has_plays_and_latest_marker_read_both_shapes():
    from utils.redzone_alt_pbp import _espn_payload_has_plays, _espn_latest_play_marker

    # Summary shape: drives at the root.
    summary = {"drives": {
        "previous": [{"plays": [{"clock": {"displayValue": "9:00"},
                                 "period": {"number": 3}}]}],
        "current": {"plays": [{"clock": {"displayValue": "0:47"},
                               "period": {"number": 4}}]},
    }}
    assert _espn_payload_has_plays(summary) is True
    assert _espn_latest_play_marker(summary) == "Q4 0:47"

    # CDN shape: drives under gamepackageJSON.
    cdn = {"gamepackageJSON": {"drives": {"current": {
        "plays": [{"clock": {"displayValue": "2:00"}, "period": {"number": 2}}]}}}}
    assert _espn_payload_has_plays(cdn) is True
    assert _espn_latest_play_marker(cdn) == "Q2 2:00"

    assert _espn_payload_has_plays({}) is False
    assert _espn_payload_has_plays({"drives": {"current": {"plays": []}}}) is False
    assert _espn_latest_play_marker({}) == ""


def test_final_game_force_refreshes_espn_until_completed(monkeypatch):
    """A game our status calls final must keep pulling fresh ESPN PBP (ttl=0)
    until ESPN reports the game completed -- otherwise the last live snapshot,
    a few plays short, freezes under the long final TTL."""
    import utils.redzone_alt_pbp as alt
    alt._ESPN_PBP_FINAL_DONE.discard("event-1")

    ttls = []
    # ESPN still shows the game in progress on the first look, completed on the next.
    payloads = iter([
        {"gamepackageJSON": {"header": {"competitions": [
            {"status": {"type": {"completed": False}}}]}}},
        {"gamepackageJSON": {"header": {"competitions": [
            {"status": {"type": {"completed": True}}}]}}},
    ])
    monkeypatch.setattr(alt, "fetch_espn_event_id", lambda **kw: "event-1")
    monkeypatch.setattr(alt, "extract_espn_pbp_plays",
                        lambda *a, **k: [{"play_id": "p"}])

    def fake_pbp(eid, *, ttl=30.0):
        ttls.append(ttl)
        return next(payloads)
    monkeypatch.setattr(alt, "fetch_espn_pbp", fake_pbp)

    # First final poll: ESPN not yet complete -> forced fresh (ttl 0), not marked done.
    alt.fetch_alt_pbp_plays("20260909_NE@SEA", season=2026, week=1, final=True)
    assert ttls[-1] == 0.0
    assert "event-1" not in alt._ESPN_PBP_FINAL_DONE

    # Second poll: ESPN now complete -> still forced fresh, and marked done.
    alt.fetch_alt_pbp_plays("20260909_NE@SEA", season=2026, week=1, final=True)
    assert ttls[-1] == 0.0
    assert "event-1" in alt._ESPN_PBP_FINAL_DONE

    # Once done, subsequent final polls serve the immutable long-TTL cache.
    payloads = iter([{"gamepackageJSON": {"header": {"competitions": [
        {"status": {"type": {"completed": True}}}]}}}])
    monkeypatch.setattr(alt, "fetch_espn_pbp", fake_pbp)
    alt.fetch_alt_pbp_plays("20260909_NE@SEA", season=2026, week=1, final=True)
    assert ttls[-1] == 300.0
    alt._ESPN_PBP_FINAL_DONE.discard("event-1")


def test_attach_cumulative_builds_running_totals_in_order():
    plays = [
        {"pid": "qb", "stat_line": {"pass_yds": 12, "pass_cmp": 1, "pass_att": 1}},
        {"pid": "qb", "stat_line": {"pass_att": 1}},  # incompletion
        {"pid": "qb", "stat_line": {"pass_yds": 1, "pass_cmp": 1, "pass_att": 1}},
        {"pid": "wr", "stat_line": {"rec": 1, "rec_yds": 12, "targets": 1}},
    ]
    out = attach_cumulative(plays)
    assert out[0]["cume"] == {"pass_yds": 12, "pass_cmp": 1, "pass_att": 1}
    # After the incompletion: 1/2 CMP, still 12 yards.
    assert out[1]["cume"] == {"pass_yds": 12, "pass_cmp": 1, "pass_att": 2}
    # After the third pass: 2/3 CMP, 13 yards.
    assert out[2]["cume"] == {"pass_yds": 13, "pass_cmp": 2, "pass_att": 3}
    assert out[3]["cume"] == {"rec": 1, "rec_yds": 12, "targets": 1}


def test_espn_extract_attaches_cumulative_per_player():
    payload = {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {
                        "plays": [
                            {
                                "id": "1",
                                "text": "D.Maye pass short right to M.Hollins for 12 yards (E.Jones).",
                                "clock": {"displayValue": "2:00"},
                                "period": {"number": 4},
                                "start": {"down": 4, "distance": 9},
                                "type": {"text": "Pass"},
                            },
                            {
                                "id": "2",
                                "text": "D.Maye pass short left to M.Hollins to SEA 36 for 11 yards (E.Jones).",
                                "clock": {"displayValue": "1:40"},
                                "period": {"number": 4},
                                "start": {"down": 1, "distance": 10},
                                "type": {"text": "Pass"},
                            },
                        ]
                    }
                ]
            }
        }
    }
    plays = extract_espn_pbp_plays(
        payload,
        "20260909_NE@SEA",
        name_to_pid={"drake maye": "11564", "malik hollins": "200"},
    )
    maye = [p for p in plays if p["pid"] == "11564"]
    assert maye[-1]["cume"] == {"pass_yds": 23, "pass_cmp": 2, "pass_att": 2}
    holl = [p for p in plays if p["pid"] == "200"]
    assert holl[-1]["cume"] == {"rec": 2, "rec_yds": 23, "targets": 2}


def test_espn_extract_preserves_drive_offense_and_provider_sequence():
    payload = {"gamepackageJSON": {"drives": {"current": {
        "team": {"abbreviation": "WSH"},
        "plays": [{
            "id": "play-7", "sequenceNumber": "107",
            "text": "J.Daniels pass complete to T.McLaurin for 8 yards.",
            "period": {"number": 2}, "clock": {"displayValue": "4:20"},
            "start": {"down": 2, "distance": 4},
        }],
    }}}}
    plays = extract_espn_pbp_plays(
        payload, "20260909_WSH@NYG",
        name_to_pid={"jayden daniels": "1", "terry mclaurin": "2"},
    )
    assert plays
    assert {play["team"] for play in plays} == {"WAS"}
    assert {play["seq"] for play in plays} == {"107"}


def test_parse_pbp_completed_pass_credits_passer_and_receiver():
    sl = parse_pbp_play_stats(
        "(Shotgun) D.Maye pass short right to M.Hollins pushed ob at SEA 16 "
        "for 12 yards (J.Jobe)."
    )
    assert sl["d.maye"] == {"pass_yds": 12, "pass_cmp": 1, "pass_att": 1}
    assert sl["m.hollins"] == {"rec": 1, "rec_yds": 12, "targets": 1}


def test_espn_multi_letter_receiver_prefix_preserves_both_contributions():
    payload = {"gamepackageJSON": {"drives": {"current": {
        "team": {"abbreviation": "ARI"},
        "plays": [{
            "id": "401", "sequenceNumber": "9001",
            "text": "(Shotgun) J.Brissett pass short right to Mi.Wilson to LAC 22 for 10 yards (D.Jackson).",
            "type": {"text": "Pass"},
        }],
    }}}}
    metadata = {
        "qb": {"name": "Jacoby Brissett", "team": "ARI", "position": "QB"},
        "wr": {"name": "Michael Wilson", "team": "ARI", "position": "WR"},
        "tackler": {"name": "Derius Jackson", "team": "LAC", "position": "CB"},
    }
    rows = extract_espn_pbp_plays(
        payload, "20260909_ARI@LAC",
        name_to_pid={"jacoby brissett": "qb", "michael wilson": "wr"},
        player_meta_by_pid=metadata,
    )
    assert {(r["pid"], r["game_id"], r["play_id"]) for r in rows} == {
        ("qb", "20260909_ARI@LAC", "401"),
        ("wr", "20260909_ARI@LAC", "401"),
    }
    by_pid = {r["pid"]: r for r in rows}
    assert by_pid["wr"]["stat_line"] == {"rec": 1, "rec_yds": 10, "targets": 1}
    assert by_pid["qb"]["stat_line"] == {"pass_yds": 10, "pass_cmp": 1, "pass_att": 1}
    assert "tackler" not in by_pid


def test_team_scoped_multi_letter_prefix_resolution_and_ambiguity():
    from utils.redzone_alt_pbp import _resolve_abbrev_pid

    _, abbrev = build_name_indexes({"michael wilson": "ari-michael"})
    metadata = {
        "ari-michael": {"name": "Michael Wilson", "team": "ARI"},
        "nyj-mike": {"name": "Mike Wilson", "team": "NYJ"},
    }
    assert _resolve_abbrev_pid("M.Wilson", abbrev_index=abbrev,
                               team="ARI", player_meta_by_pid=metadata) == ("ari-michael", 1)
    assert _resolve_abbrev_pid("Mi.Wilson", abbrev_index=abbrev,
                               team="ARI", player_meta_by_pid=metadata) == ("ari-michael", 1)
    assert _resolve_abbrev_pid("Mic.Wilson", abbrev_index=abbrev,
                               team="ARI", player_meta_by_pid=metadata) == ("ari-michael", 1)
    assert _resolve_abbrev_pid("Mic.Wilson", abbrev_index=abbrev,
                               team="NYJ", player_meta_by_pid=metadata) == ("", 0)

    ambiguous = {**metadata, "ari-micah": {"name": "Micah Wilson", "team": "ARI"}}
    assert _resolve_abbrev_pid("Mi.Wilson", abbrev_index={},
                               team="ARI", player_meta_by_pid=ambiguous) == ("", 2)


def test_prefix_resolution_handles_suffixes_and_compound_surnames():
    from utils.redzone_alt_pbp import _resolve_abbrev_pid

    metadata = {
        "mhj": {"name": "Marvin Harrison Jr.", "team": "ARI"},
        "olave": {"name": "Chris Olave", "team": "NO"},
        "jsn": {"name": "Jaxon Smith-Njigba", "team": "SEA"},
        "arsb": {"name": "Amon-Ra St. Brown", "team": "DET"},
    }
    assert _resolve_abbrev_pid("Mar.Harrison", abbrev_index={}, team="ARI", player_meta_by_pid=metadata)[0] == "mhj"
    assert _resolve_abbrev_pid("Ch.Olave", abbrev_index={}, team="NO", player_meta_by_pid=metadata)[0] == "olave"
    # Existing single-initial compound resolution remains exact.
    _, abbrev = build_name_indexes({m["name"]: pid for pid, m in metadata.items()})
    assert _resolve_abbrev_pid("J.Smith-Njigba", abbrev_index=abbrev)[0] == "jsn"
    assert _resolve_abbrev_pid("A.St. Brown", abbrev_index=abbrev)[0] == "arsb"


def test_espn_no_play_emits_no_player_contributions_or_cumulative_stats():
    payload = {"gamepackageJSON": {"drives": {"current": {
        "team": {"abbreviation": "ARI"}, "plays": [{
            "id": "np1",
            "text": "J.Brissett pass incomplete deep right to Mi.Wilson PENALTY on LAC-D.Jackson, Defensive Pass Interference, 23 yards, enforced at ARZ 45 - No Play.",
        }],
    }}}}
    rows = extract_espn_pbp_plays(
        payload, "20260909_ARI@LAC",
        name_to_pid={"jacoby brissett": "qb", "michael wilson": "wr"},
        player_meta_by_pid={"qb": {"name": "Jacoby Brissett", "team": "ARI"},
                            "wr": {"name": "Michael Wilson", "team": "ARI"}},
    )
    assert len(rows) == 1
    assert rows[0]["pid"] == ""
    assert rows[0]["stat_line"] == {}
    assert rows[0]["cume"] == {}
    assert rows[0]["is_no_play"] is True


def test_parse_pbp_td_pass_and_extra_point():
    sl = parse_pbp_play_stats(
        "D.Lock pass short left to J.Smith-Njigba for 45 yards, TOUCHDOWN. "
        "J.Myers extra point is GOOD, Center-C.Stoll, Holder-M.Dickson."
    )
    assert sl["d.lock"] == {"pass_yds": 45, "pass_cmp": 1, "pass_att": 1, "pass_td": 1}
    assert sl["j.smith-njigba"] == {"rec": 1, "rec_yds": 45, "targets": 1, "rec_td": 1}
    assert sl["j.myers"] == {"xpm": 1}


def test_parse_pbp_td_credit_is_case_insensitive_and_accepts_td_token():
    # ESPN (the primary live source) writes "Touchdown"/"td", not only Tank01's
    # uppercase "TOUCHDOWN". The TD points must still be credited, else a live
    # total runs light versus the box score (a QB read ~20 pts low).
    pass_td = parse_pbp_play_stats(
        "(Shotgun) C.Williams pass short right to D.Moore for 15 yards, Touchdown."
    )
    assert pass_td["c.williams"] == {
        "pass_yds": 15, "pass_cmp": 1, "pass_att": 1, "pass_td": 1,
    }
    assert pass_td["d.moore"] == {"rec": 1, "rec_yds": 15, "targets": 1, "rec_td": 1}

    rush_td = parse_pbp_play_stats("C.Williams up the middle for 3 yards, TD.")
    assert rush_td["c.williams"] == {"rush_yds": 3, "carries": 1, "rush_td": 1}


def test_parse_pbp_lowercase_turnover_return_still_denies_offense_td():
    # A pick-six / fumble-return score names a touchdown but the offense is not
    # credited -- the exclusion must hold regardless of casing.
    assert parse_pbp_play_stats(
        "C.Williams pass INTERCEPTED at CAR 20, returned by J.Jobe for a touchdown."
    ) == {"c.williams": {"int": 1, "pass_att": 1}}


def test_parse_pbp_captures_compound_surnames_in_booth_text():
    # The name token must span multi-word surnames ("St. Brown", with or without
    # the internal space) so these players aren't silently dropped from plays.
    for text in (
        "J.Goff pass short right to A.St. Brown for 12 yards, TOUCHDOWN.",
        "J.Goff pass short right to A.St.Brown for 12 yards, TOUCHDOWN.",
    ):
        sl = parse_pbp_play_stats(text)
        key = next(k for k in sl if k.startswith("a.st"))
        assert sl[key] == {"rec": 1, "rec_yds": 12, "targets": 1, "rec_td": 1}
        assert sl["j.goff"]["pass_td"] == 1


def test_abbrev_index_resolves_suffixes_compounds_and_middle_initials():
    from utils.redzone_alt_pbp import build_name_indexes, _stat_lines_by_pid

    # Real full names (as the redzone endpoint feeds them, already lowercased).
    names = {
        "amon-ra st. brown": "stbrown",
        "a.j. brown": "ajbrown",
        "michael pittman jr.": "pittman",
        "kenneth walker iii": "kwalker",
        "marquez valdes-scantling": "mvs",
        "d.j. moore": "djmoore",
        "deebo samuel sr.": "deebo",
        "jared goff": "goff",
    }
    _full, abbrev = build_name_indexes(names)

    def who(text):
        return sorted(k for k in _stat_lines_by_pid(text, abbrev) if k != "goff")

    # Suffixes never become the surname; compound names resolve; and A.St. Brown
    # no longer collides A.J. Brown onto a shared "abrown" key.
    assert who("J.Goff pass to A.St. Brown for 5 yards.") == ["stbrown"]
    assert who("J.Goff pass to A.Brown for 5 yards.") == ["ajbrown"]
    assert who("J.Goff pass to M.Pittman for 5 yards.") == ["pittman"]
    assert who("K.Walker up the middle for 5 yards.") == ["kwalker"]
    assert who("J.Goff pass to M.Valdes-Scantling for 5 yards.") == ["mvs"]
    assert who("J.Goff pass to D.Moore for 5 yards.") == ["djmoore"]
    assert who("J.Goff pass to D.Samuel for 5 yards.") == ["deebo"]


def test_parse_pbp_interception_only_credits_passer_pick():
    sl = parse_pbp_play_stats(
        "(Shotgun) D.Maye pass deep right intended for M.Hollins INTERCEPTED "
        "by J.Jobe [D.Lawrence] at SEA -3. Touchback."
    )
    # An interception is a target for the intended receiver.
    assert sl == {"d.maye": {"int": 1, "pass_att": 1}, "m.hollins": {"targets": 1}}


def test_parse_pbp_sack_is_not_scored_as_a_rush():
    assert parse_pbp_play_stats(
        "(Shotgun) D.Maye sacked at SEA 28 for -7 yards (D.Hall)."
    ) == {}


def test_parse_pbp_rush_with_and_without_td():
    assert parse_pbp_play_stats(
        "R.Stevenson up the middle to NE 11 for 3 yards (D.Lawrence)."
    ) == {"r.stevenson": {"rush_yds": 3, "carries": 1}}
    assert parse_pbp_play_stats(
        "R.Stevenson up the middle for 2 yards, TOUCHDOWN."
    ) == {"r.stevenson": {"rush_yds": 2, "carries": 1, "rush_td": 1}}


def test_parse_pbp_rush_td_scores_regardless_of_touchdown_casing():
    # The uppercase-only check badged these as scores but dropped the 6 points,
    # so a two-rush-TD back showed his yards with none of the TD value.
    for text in (
        "D.Montgomery up the middle for 2 yards, Touchdown.",
        "D.Montgomery up the middle for 2 yards, touchdown.",
        "D.Montgomery right guard for 1 yard for a TD.",
    ):
        assert parse_pbp_play_stats(text) == {
            "d.montgomery": {"rush_yds": 2 if "2 yards" in text else 1,
                             "carries": 1, "rush_td": 1}
        }, text


def test_parse_pbp_pass_td_scores_regardless_of_touchdown_casing():
    sl = parse_pbp_play_stats(
        "C.Stroud pass short right to N.Collins for 12 yards, Touchdown."
    )
    assert sl["c.stroud"] == {"pass_yds": 12, "pass_cmp": 1, "pass_att": 1, "pass_td": 1}
    assert sl["n.collins"] == {"rec": 1, "rec_yds": 12, "targets": 1, "rec_td": 1}


def test_parse_pbp_td_credit_still_suppressed_on_turnover():
    # A ball turned over first is never an offensive TD, whatever the casing.
    assert "rush_td" not in parse_pbp_play_stats(
        "D.Montgomery up the middle for 2 yards, fumble, TOUCHDOWN Seattle."
    ).get("d.montgomery", {})
    assert "pass_td" not in parse_pbp_play_stats(
        "C.Stroud pass deep left INTERCEPTED by T.Bland, returned for a TD."
    ).get("c.stroud", {})


def test_parse_pbp_rush_credit_survives_leading_clause():
    # A pre-snap clause must not steal the carry from the actual ball carrier.
    assert parse_pbp_play_stats(
        "G.Van Roten reported in as eligible. D.Maye scrambles left end ran "
        "ob at SEA 21 for 6 yards (J.Jobe)."
    ) == {"d.maye": {"rush_yds": 6, "carries": 1}}


def test_parse_pbp_no_gain_reception_still_counts():
    sl = parse_pbp_play_stats("D.Maye pass complete to H.Henry for no gain.")
    assert sl["h.henry"] == {"rec": 1, "rec_yds": 0, "targets": 1}


def test_parse_pbp_field_goal_keeps_distance():
    # Distance is retained so the client can score fgm_40_49 / fgm_50p buckets.
    assert parse_pbp_play_stats(
        "J.Myers 45 yard field goal is GOOD, Center-C.Stoll."
    ) == {"j.myers": {"fgm": 1, "fg_yds": 45, "fgm_40_49": 1}}
    assert parse_pbp_play_stats(
        "C.Santos 52 yard field goal is GOOD."
    ) == {"c.santos": {"fgm": 1, "fg_yds": 52, "fgm_50_59": 1}}


def test_espn_plays_attach_real_stat_lines():
    payload = {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {
                        "plays": [
                            {
                                "id": "9",
                                "text": (
                                    "D.Lock pass short left to J.Smith-Njigba "
                                    "for 45 yards, TOUCHDOWN."
                                ),
                                "clock": {"displayValue": "11:28"},
                                "period": {"number": 4},
                                "start": {"down": 4, "distance": 1},
                                "type": {"text": "Passing Touchdown"},
                                "scoringPlay": True,
                            }
                        ]
                    }
                ]
            }
        }
    }
    plays = extract_espn_pbp_plays(
        payload,
        "20260909_NE@SEA",
        name_to_pid={"jaxon smith-njigba": "8155", "drew lock": "99"},
    )
    by_pid = {p["pid"]: p for p in plays}
    assert by_pid["8155"]["stat_line"] == {
        "rec": 1, "rec_yds": 45, "targets": 1, "rec_td": 1,
    }
    assert by_pid["99"]["stat_line"] == {
        "pass_yds": 45, "pass_cmp": 1, "pass_att": 1, "pass_td": 1,
    }


def test_kicking_unit_linemen_never_headline_a_scoring_play():
    # Reported bug: a rushing TD followed by a made PAT credited the extra
    # point's long snapper ("Center-R.Underwood") as the scorer, because the
    # inline kicking-unit credit resolved as a mention and the row inherited the
    # play's TD flag. The snapper/holder must never produce a card; only the
    # ball carrier and the kicker do.
    text = (
        "T.Bigsby up the middle for 2 yards, TOUCHDOWN. J.Elliott extra point "
        "is GOOD, Center-R.Underwood, Holder-B.Mann."
    )
    payload = {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {
                        "plays": [
                            {
                                "id": "42",
                                "text": text,
                                "clock": {"displayValue": "12:49"},
                                "period": {"number": 2},
                                "start": {"down": 2, "distance": 1},
                                "type": {"text": "Rushing Touchdown"},
                                "scoringPlay": True,
                            }
                        ]
                    }
                ]
            }
        }
    }
    plays = extract_espn_pbp_plays(
        payload,
        "20260920_PHI@TEN",
        name_to_pid={
            "tank bigsby": "bigsby",
            "jake elliott": "elliott",
            "rocco underwood": "underwood",
            "braden mann": "mann",
        },
    )
    by_pid = {p["pid"]: p for p in plays}
    # The snapper and holder are stripped as non-actor credits before mention
    # resolution, so they never surface at all.
    assert "underwood" not in by_pid
    assert "mann" not in by_pid
    # The ball carrier keeps the rushing TD; the kicker keeps the made PAT and
    # is not falsely flagged as the touchdown scorer.
    assert by_pid["bigsby"]["is_td"] is True
    assert by_pid["bigsby"]["stat_line"]["rush_td"] == 1
    assert by_pid["elliott"]["stat_line"] == {"xpm": 1}
    assert by_pid["elliott"]["is_td"] is False


def test_parse_tank_game_id():
    assert parse_tank_game_id("20260909_NE@SEA") == ("20260909", "NE", "SEA")
    assert parse_tank_game_id("bad") == ("", "", "")


def test_abbrev_name_matching():
    full, abbrev = build_name_indexes({
        "drake maye": "11564",
        "rhamondre stevenson": "8155",
        "george holani": "12048",
    })
    text = "D.Maye scrambles up the middle to NE 21 for 10 yards (E.Jones)."
    assert pids_mentioned_in_text(text, full_index=full, abbrev_index=abbrev) == ["11564"]
    text2 = "R.Stevenson up the middle to NE 11 for 3 yards (D.Lawrence)."
    assert "8155" in pids_mentioned_in_text(text2, full_index=full, abbrev_index=abbrev)


def test_extract_espn_pbp_maps_abbreviated_names():
    payload = {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {
                        "plays": [
                            {
                                "id": "1",
                                "text": "R.Stevenson up the middle to NE 11 for 3 yards (D.Lawrence).",
                                "clock": {"displayValue": "11:32"},
                                "period": {"number": 1},
                                "start": {"down": 1, "distance": 10, "possessionText": "NE 8"},
                                "type": {"text": "Rush"},
                                "scoringPlay": False,
                            },
                            {
                                "id": "2",
                                "text": "A.Borregales kicks 63 yards from NE 35 to SEA 2.",
                                "type": {"text": "Kickoff"},
                                "clock": {"displayValue": "15:00"},
                                "period": {"number": 1},
                                "start": {},
                            },
                        ]
                    }
                ]
            }
        }
    }
    plays = extract_espn_pbp_plays(
        payload,
        "20260909_NE@SEA",
        name_to_pid={"rhamondre stevenson": "8155"},
    )
    assert len(plays) == 1
    assert plays[0]["pid"] == "8155"
    assert plays[0]["play_text"].startswith("R.Stevenson")
    assert plays[0]["source"] == "espn"
    assert plays[0]["clock"] == "11:32"
    assert plays[0]["down"] == "1"


def test_espn_rostered_tackler_does_not_create_a_card():
    """A defender credited only in the parenthetical tackle group must not be
    emitted as a play row, even when he is a rostered/tracked player. The play
    belongs to the ball carrier -- here the rusher must be the only row."""
    payload = {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {
                        "plays": [
                            {
                                "id": "1",
                                "text": (
                                    "M.Washington left tackle to MIA 34 "
                                    "for 3 yards (J.Rodriguez)."
                                ),
                                "clock": {"displayValue": "11:23"},
                                "period": {"number": 2},
                                "start": {"down": 1, "distance": 10,
                                          "possessionText": "MIA 37"},
                                "type": {"text": "Rush"},
                                "scoringPlay": False,
                            }
                        ]
                    }
                ]
            }
        }
    }
    plays = extract_espn_pbp_plays(
        payload,
        "20260913_LV@MIA",
        # Both the rusher and the tackler are tracked players.
        name_to_pid={"malik washington": "100", "jacob rodriguez": "200"},
    )
    assert [p["pid"] for p in plays] == ["100"]
    assert plays[0]["stat_line"] == {"rush_yds": 3, "carries": 1}


def test_espn_rostered_sacker_in_parens_does_not_create_a_card():
    """The sacker (parenthetical credit) must not headline the sacked QB's play."""
    payload = {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {
                        "plays": [
                            {
                                "id": "2",
                                "text": (
                                    "T.Tagovailoa sacked at MIA 20 for -7 yards "
                                    "(M.Crosby)."
                                ),
                                "clock": {"displayValue": "9:00"},
                                "period": {"number": 2},
                                "start": {"down": 2, "distance": 10},
                                "type": {"text": "Sack"},
                                "scoringPlay": False,
                            }
                        ]
                    }
                ]
            }
        }
    }
    plays = extract_espn_pbp_plays(
        payload,
        "20260913_LV@MIA",
        name_to_pid={"maxx crosby": "300", "tua tagovailoa": "400"},
    )
    # The sacked QB is the subject of the play (outside parens) and still gets a
    # row. The sacker (Maxx Crosby) is a parenthetical credit and must not
    # appear -- a defender never headlines an offensive snap.
    assert [p["pid"] for p in plays] == ["400"]


def test_extract_sleeper_pbp_when_rows_present():
    raw = [
        {
            "id": "sl1",
            "text": "Drake Maye pass complete to Hunter Henry for 12 yards",
            "quarter": "2",
            "clock": "5:01",
            "down": "2",
            "distance": "8",
        }
    ]
    plays = extract_sleeper_pbp_plays(
        raw,
        "20260909_NE@SEA",
        name_to_pid={"drake maye": "11564", "hunter henry": "4037"},
    )
    assert len(plays) == 2
    assert {p["pid"] for p in plays} == {"11564", "4037"}
    assert plays[0]["source"] == "sleeper"


def test_extract_sleeper_empty_is_safe():
    assert extract_sleeper_pbp_plays([], "g") == []
    assert extract_sleeper_pbp_plays(None, "g") == []  # type: ignore[arg-type]


def _espn_scoreboard_payload():
    return {
        "content": {
            "sbData": {
                "events": [
                    {
                        "id": "401700000",
                        "date": "2026-09-11T00:20Z",
                        "status": {
                            "displayClock": "12:34",
                            "period": 2,
                            "type": {
                                "state": "in",
                                "completed": False,
                                "shortDetail": "12:34 - 2nd",
                            },
                        },
                        "competitions": [
                            {
                                "competitors": [
                                    {
                                        "homeAway": "home",
                                        "team": {"abbreviation": "SEA"},
                                        "score": "10",
                                    },
                                    {
                                        "homeAway": "away",
                                        "team": {"abbreviation": "NE"},
                                        "score": "7",
                                    },
                                ]
                            }
                        ],
                    },
                    {
                        "id": "401700001",
                        "date": "2026-09-14T17:00Z",
                        "status": {
                            "period": 4,
                            "type": {
                                "state": "post",
                                "completed": True,
                                "shortDetail": "Final",
                            },
                        },
                        "competitions": [
                            {
                                "competitors": [
                                    {
                                        "homeAway": "home",
                                        "team": {"abbreviation": "WSH"},
                                        "score": "24",
                                    },
                                    {
                                        "homeAway": "away",
                                        "team": {"abbreviation": "DAL"},
                                        "score": "21",
                                    },
                                ]
                            }
                        ],
                    },
                ]
            }
        }
    }


def test_extract_espn_scoreboard_lookup_shapes_tank01_style():
    lookup = extract_espn_scoreboard_lookup(_espn_scoreboard_payload())
    # Both teams of each game are keyed to the same game dict.
    assert lookup["SEA"] is lookup["NE"]
    sea = lookup["SEA"]
    assert sea["gameID"] == "20260911_NE@SEA"
    assert sea["gameStatusCode"] == "1"  # in-progress
    assert sea["gameClock"] == "12:34"
    assert sea["lineScore"]["period"] == "2"
    assert sea["homePts"] == "10" and sea["awayPts"] == "7"
    assert sea["source"] == "espn"


def test_extract_espn_scoreboard_normalizes_abbrev_and_final():
    lookup = extract_espn_scoreboard_lookup(_espn_scoreboard_payload())
    # ESPN's WSH is normalized to Sleeper/Tank01 WAS so rostered teams match.
    assert "WAS" in lookup and "WSH" not in lookup
    was = lookup["WAS"]
    assert was["gameID"] == "20260914_DAL@WAS"
    assert was["gameStatusCode"] == "2"  # final


def test_extract_espn_scoreboard_empty_is_safe():
    assert extract_espn_scoreboard_lookup({}) == {}
    assert extract_espn_scoreboard_lookup(None) == {}  # type: ignore[arg-type]
    assert extract_espn_scoreboard_lookup({"content": {"sbData": {"events": []}}}) == {}


def test_parse_pbp_incomplete_pass_credits_receiver_target():
    """Regression: the running line read "5/5 REC" on a 5-catch, 12-target
    game because only completions credited targets."""
    sl = parse_pbp_play_stats(
        "(Shotgun) J.Love pass incomplete deep right to M.Golden."
    )
    assert sl["j.love"] == {"pass_att": 1}
    assert sl["m.golden"] == {"targets": 1}


def test_parse_pbp_incomplete_without_target_credits_only_passer():
    sl = parse_pbp_play_stats(
        "M.Penix pass incomplete short middle [L.Van Ness]."
        "PENALTY on ATL-M.Penix, Intentional Grounding, 10 yards, "
        "enforced at ATL 35."
    )
    assert sl["m.penix"] == {"pass_att": 1}
    assert "m.golden" not in sl


def test_parse_pbp_interception_credits_intended_receiver_target():
    sl = parse_pbp_play_stats(
        "(Shotgun) M.Penix pass short middle intended for J.Dotson "
        "INTERCEPTED by X.McKinney at GB 40."
    )
    assert sl["m.penix"] == {"int": 1, "pass_att": 1}
    assert sl["j.dotson"] == {"targets": 1}


def test_parse_pbp_nullified_play_credits_nothing():
    """A penalty-wiped ("No Play") deep shot is not a 13th target."""
    sl = parse_pbp_play_stats(
        "(Shotgun) J.Love pass incomplete deep left to M.Golden."
        "PENALTY on ATL-M.Hughes, Defensive Pass Interference, 34 yards, "
        "enforced at GB 33 - No Play."
    )
    assert sl == {}


def test_parse_pbp_reversed_completion_scores_as_incompletion():
    sl = parse_pbp_play_stats(
        "(Shotgun) J.Love pass short right to T.Kraft to GB 45 for 4 yards "
        "(J.Bates). FUMBLES (J.Bates), ball out of bounds at GB 42."
        "The Replay Official reviewed the pass completion and the play was "
        "REVERSED - incomplete pass."
    )
    assert sl["j.love"] == {"pass_att": 1}
    assert sl["t.kraft"] == {"targets": 1}


def test_parse_pbp_five_catches_twelve_targets_cumulative():
    """Golden's real game: 5 completions + 7 targeted incompletions = 12."""
    plays = [
        "J.Love pass short right to M.Golden for 15 yards, TOUCHDOWN.",
        "(Shotgun) J.Love pass deep right to M.Golden to ATL 43 for 45 yards (M.Hughes).",
        "(Shotgun) J.Love pass short right to M.Golden pushed ob at GB 41 for 8 yards (M.Hughes).",
        "(Shotgun) J.Love pass short left to M.Golden pushed ob at 50 for 12 yards (D.Deablo).",
        "J.Love pass short left to M.Golden ran ob at ATL 10 for 20 yards (B.Bowman).",
        "(Shotgun) J.Love pass incomplete deep right to M.Golden.",
        "(Shotgun) J.Love pass incomplete deep left to M.Golden (B.Bowman).",
        "(Shotgun) J.Love pass incomplete deep right to M.Golden (C.Henderson).",
        "(Shotgun) J.Love pass incomplete short left to M.Golden (C.Henderson).",
        "(Shotgun) J.Love pass incomplete deep right to M.Golden [C.Thomas].",
        "(No Huddle, Shotgun) J.Love pass incomplete short right to M.Golden.",
        "(Shotgun) J.Love pass incomplete short right to M.Golden [C.Thomas].",
    ]
    rec = targets = 0
    for p in plays:
        sl = parse_pbp_play_stats(p).get("m.golden", {})
        rec += sl.get("rec", 0)
        targets += sl.get("targets", 0)
    assert (rec, targets) == (5, 12)
