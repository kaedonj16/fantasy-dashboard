"""ESPN live-draft sync: parsing, mapping, reconciliation, and fallback.

Mocks ESPN payloads. Does not contact ESPN, Flask, or the player index.
"""
from __future__ import annotations

from types import SimpleNamespace

from dashboard_services.draft_sync import (
    DraftSyncAuthError,
    DraftSyncUnsupportedError,
    NormalizedDraftPick,
    apply_viewer_team,
    espn_draft_sync_poll_ms,
    espn_status_from_flags,
    get_draft_sync_provider,
    live_picks_payload,
    make_espn_draft_id,
    map_espn_player_id,
    merge_picks_idempotent,
    new_picks_since,
    normalize_espn_picks,
    parse_espn_draft_detail,
    parse_espn_draft_id,
    snapshot_fingerprint,
    snapshot_to_live_payload,
)
from dashboard_services.providers.espn_draft import ESPNDraftSyncProvider


def _detail(picks=None, drafted=None, in_progress=None, **settings):
    payload = {
        "draftDetail": {
            "drafted": drafted,
            "inProgress": in_progress,
            "picks": picks if picks is not None else [],
        },
        "settings": {
            "draftSettings": {
                "type": settings.get("type", "SNAKE"),
                "date": settings.get("date", 1_700_000_000_000),
                "timePerSelection": settings.get("timePerSelection", 90),
                "pickOrder": settings.get("pickOrder", [1, 2, 3, 4]),
                "rounds": settings.get("rounds", 15),
            },
            "rosterSettings": {
                "lineupSlotCounts": {"0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "20": 6},
            },
        },
        "teams": [
            {"id": 1, "name": "Alpha", "owners": ["{AAA}"], "primaryOwner": "{AAA}"},
            {"id": 2, "name": "Bravo", "owners": [{"id": "{BBB}"}]},
            {"id": 3, "name": "Charlie", "primaryOwner": "{CCC}"},
            {"id": 4, "name": "Delta", "owners": ["{DDD}"]},
        ],
    }
    if "draftDetail" in settings and settings["draftDetail"] is None:
        del payload["draftDetail"]
    return payload


def _pick(overall, player, team, rnd=1, slot=1, keeper=False):
    return {
        "playerId": player,
        "teamId": team,
        "overallPickNumber": overall,
        "roundId": rnd,
        "roundPickNumber": slot,
        "keeper": keeper,
    }


CANON = {"4039057": "5938", "4241479": "4866"}
LOOKUP = {
    "5938": {"name": "Justin Jefferson", "position": "WR", "team": "MIN"},
    "4866": {"name": "Ja'Marr Chase", "position": "WR", "team": "CIN"},
    "BAL": {"name": "Ravens D/ST", "position": "DEF", "team": "BAL"},
}


def _norm(payload, canon=None):
    detail = parse_espn_draft_detail(payload)
    return normalize_espn_picks(
        detail,
        espn_to_canon=canon if canon is not None else CANON,
        player_lookup=LOOKUP.get,
        dst_mapper=lambda pid: "BAL" if str(pid) == "-16033" else None,
        team_owner_map={"1": "{AAA}", "2": "{BBB}", "3": "{CCC}", "4": "{DDD}"},
        team_slot_map={"1": 1, "2": 2, "3": 3, "4": 4},
        n_teams=4,
    )


# ── parse ─────────────────────────────────────────────────────────────────────

def test_parse_draft_not_started():
    detail = parse_espn_draft_detail(_detail(picks=[], drafted=False, in_progress=False))
    assert detail.drafted is False
    assert detail.in_progress is False
    assert detail.picks_observed is True
    assert detail.picks == ()
    assert espn_status_from_flags(detail.drafted, detail.in_progress, pick_count=0) == "pre_draft"


def test_normalize_skips_predraft_placeholder_grid():
    """ESPN predraft mDraftDetail often lists every seat with playerId 0."""
    from dashboard_services.draft_sync import espn_player_id_is_selected
    assert espn_player_id_is_selected("0") is False
    assert espn_player_id_is_selected("-1") is False
    assert espn_player_id_is_selected(None) is False
    assert espn_player_id_is_selected("4039057") is True
    assert espn_player_id_is_selected("-16033") is True
    placeholders = [
        _pick(i, 0, ((i - 1) % 4) + 1, rnd=(i - 1) // 4 + 1, slot=((i - 1) % 4) + 1)
        for i in range(1, 17)
    ]
    picks = _norm(_detail(picks=placeholders, drafted=False, in_progress=False))
    assert picks == []
    assert espn_status_from_flags(False, False, pick_count=len(picks)) == "pre_draft"


def test_normalize_skips_null_and_sentinel_player_placeholders():
    rows = [
        {"playerId": None, "teamId": 1, "overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1},
        {"playerId": -1, "teamId": 2, "overallPickNumber": 2, "roundId": 1, "roundPickNumber": 2},
        {"playerId": "", "teamId": 3, "overallPickNumber": 3, "roundId": 1, "roundPickNumber": 3},
    ]
    picks = _norm(_detail(picks=rows, drafted=False, in_progress=False))
    assert picks == []


def test_normalize_keeps_keepers_among_placeholders():
    rows = [_pick(i, 0, ((i - 1) % 4) + 1) for i in range(1, 9)]
    rows[0] = _pick(1, 4039057, 1, keeper=True)
    picks = _norm(_detail(picks=rows, drafted=False, in_progress=False))
    assert [p.overall_pick for p in picks] == [1]
    assert picks[0].canonical_player_id == "5938"
    assert picks[0].keeper is True


def test_placeholder_picks_do_not_count_as_drafting_when_flags_missing():
    placeholders = [_pick(i, 0, 1) for i in range(1, 5)]
    detail = parse_espn_draft_detail(_detail(picks=placeholders))
    real = _norm(_detail(picks=placeholders))
    assert espn_status_from_flags(detail.drafted, detail.in_progress, pick_count=len(real)) != "drafting"
    assert real == []


def test_parse_draft_in_progress():
    picks = [_pick(1, 4039057, 1)]
    detail = parse_espn_draft_detail(_detail(picks=picks, drafted=False, in_progress=True))
    assert detail.in_progress is True
    assert len(detail.picks) == 1
    assert espn_status_from_flags(detail.drafted, detail.in_progress, pick_count=1) == "drafting"


def test_parse_completed_draft():
    picks = [_pick(i, 4000000 + i, ((i - 1) % 4) + 1, rnd=(i - 1) // 4 + 1) for i in range(1, 9)]
    detail = parse_espn_draft_detail(_detail(picks=picks, drafted=True, in_progress=False))
    assert espn_status_from_flags(detail.drafted, detail.in_progress, pick_count=8) == "complete"


def test_parse_no_picks_list():
    payload = _detail(drafted=False, in_progress=True)
    payload["draftDetail"]["picks"] = None
    detail = parse_espn_draft_detail(payload)
    assert detail.picks_observed is False
    assert detail.picks == ()


def test_parse_missing_draft_detail():
    detail = parse_espn_draft_detail({"settings": {}})
    assert detail.detail_present is False
    assert detail.picks_observed is False
    assert espn_status_from_flags(detail.drafted, detail.in_progress) == "unknown"


def test_parse_malformed_payloads_do_not_raise():
    for payload in (None, [], "nope", 7, {"draftDetail": "x"}, {"draftDetail": {"picks": [None, 1, "x"]}}):
        detail = parse_espn_draft_detail(payload)
        assert detail.picks == () or all(p.overall_pick or p.player_id for p in detail.picks)


def test_parse_picks_from_objects_and_dicts():
    obj = SimpleNamespace(playerId=4039057, teamId=2, overallPickNumber=3, roundId=1, roundPickNumber=3, keeper=False)
    detail = parse_espn_draft_detail({"draftDetail": {"picks": [obj, _pick(4, 4241479, 4)]}})
    assert [p.player_id for p in detail.picks] == ["4039057", "4241479"]


# ── mapping ───────────────────────────────────────────────────────────────────

def test_player_mapping_resolves_canonical_id():
    cid, unresolved = map_espn_player_id("4039057", CANON)
    assert cid == "5938" and unresolved is False


def test_unresolved_player_does_not_fuzzy_match():
    cid, unresolved = map_espn_player_id("999999", CANON)
    assert cid is None and unresolved is True


def test_dst_fallback_mapping():
    cid, unresolved = map_espn_player_id("-16033", {}, dst_mapper=lambda pid: "BAL")
    assert cid == "BAL" and unresolved is False


def test_dst_display_synthesizes_when_index_misses():
    """players_index has no DEF rows; team-abbr ids must still become 'BAL D/ST'."""
    detail = parse_espn_draft_detail(_detail(
        picks=[_pick(12, -16033, 1, rnd=3, slot=4)],
        in_progress=True, drafted=False,
    ))
    picks = normalize_espn_picks(
        detail,
        espn_to_canon={},
        player_lookup=lambda _pid: {},
        dst_mapper=lambda pid: "BAL" if str(pid) == "-16033" else None,
        team_owner_map={"1": "{AAA}"},
        team_slot_map={"1": 1},
        n_teams=4,
    )
    assert len(picks) == 1
    assert picks[0].canonical_player_id == "BAL"
    assert picks[0].unresolved is False
    assert picks[0].name == "BAL D/ST"
    assert picks[0].position == "DEF"
    assert picks[0].team == "BAL"


def test_kicker_pos_pk_normalizes_to_k():
    """Tank01 stores kickers as PK; starter slots are labeled K."""
    detail = parse_espn_draft_detail(_detail(
        picks=[_pick(8, 4241457, 2, rnd=2, slot=4)],
        in_progress=True, drafted=False,
    ))
    picks = normalize_espn_picks(
        detail,
        espn_to_canon={"4241457": "421"},
        player_lookup=lambda pid: (
            {"name": "Justin Tucker", "pos": "PK", "team": "BAL"}
            if str(pid) == "421" else {}
        ),
        team_owner_map={"2": "{BBB}"},
        team_slot_map={"2": 2},
        n_teams=4,
    )
    assert len(picks) == 1
    assert picks[0].name == "Justin Tucker"
    assert picks[0].position == "K"


def test_normalize_unresolved_preserves_pick():
    picks = _norm(_detail(picks=[_pick(1, 999999, 1)], in_progress=True, drafted=False), canon={})
    assert len(picks) == 1
    assert picks[0].unresolved is True
    assert picks[0].canonical_player_id is None
    assert picks[0].player_id == ""
    assert picks[0].external_player_id == "999999"
    assert picks[0].overall_pick == 1


def test_user_team_detection_sets_picked_by():
    from dashboard_services.draft_sync import DraftSyncSnapshot
    picks = _norm(_detail(picks=[_pick(5, 4039057, 3, rnd=2, slot=1)], in_progress=True))
    assert picks[0].picked_by == "{CCC}"
    assert picks[0].roster_id == "3"
    assert picks[0].draft_slot == 3
    snap = apply_viewer_team(
        DraftSyncSnapshot(
            source="espn", draft_id="espn_1_2026", league_id="1", season=2026,
            status="drafting", picks=picks, user_roster_map={"{CCC}": "3"},
        ),
        viewer_user_id="{CCC}",
        viewer_roster_id="3",
    )
    assert snap.viewer_team_id == "3"


# ── reconciliation ────────────────────────────────────────────────────────────

def _np(n, pid="5938"):
    return NormalizedDraftPick(source="espn", overall_pick=n, canonical_player_id=pid, external_player_id=str(n))


def test_one_new_pick():
    remote = [_np(i) for i in range(1, 19)]
    added = new_picks_since(range(1, 18), remote)
    assert [p.overall_pick for p in added] == [18]


def test_multiple_new_picks_and_missed_gap():
    remote = [_np(i) for i in range(1, 21)]
    added = new_picks_since(range(1, 18), remote)
    assert [p.overall_pick for p in added] == [18, 19, 20]


def test_duplicate_response_is_idempotent():
    remote = [_np(i) for i in range(1, 18)]
    first = new_picks_since(range(1, 18), remote)
    assert first == []
    merged = merge_picks_idempotent({p.overall_pick: p for p in remote}, remote)
    assert len(merged) == 17


def test_duplicate_overall_in_espn_payload_kept_once():
    payload = _detail(picks=[_pick(1, 4039057, 1), _pick(1, 4241479, 2)], in_progress=True)
    picks = _norm(payload)
    assert len(picks) == 1
    assert picks[0].canonical_player_id == "5938"


def test_reconciliation_after_refresh_uses_full_remote_set():
    remote = [_np(i) for i in range(1, 13)]
    added = new_picks_since([], remote)  # browser refresh: local state empty
    assert [p.overall_pick for p in added] == list(range(1, 13))


def test_live_payload_shape_and_no_secrets():
    picks = _norm(_detail(picks=[_pick(1, 4039057, 1)], in_progress=True, drafted=False))
    body = live_picks_payload(picks)
    assert body[0]["pick_no"] == 1
    assert body[0]["player_id"] == "5938"
    assert body[0]["picked_by"] == "{AAA}"
    blob = str(body)
    assert "espn_s2" not in blob and "SWID" not in blob.lower()


# ── provider snapshot (mocked ESPN HTTP) ──────────────────────────────────────

class _FakeEspn:
    def _espn_to_canon_cached(self):
        return CANON

    def _dst_canonical_id(self, _bp, pid_int):
        return "BAL" if str(pid_int).startswith("-160") else None

    def _players_index_cached(self):
        return LOOKUP

    def _is_espn_access_denied(self, exc):
        return type(exc).__name__ == "ESPNAccessDenied"


def _provider(monkeypatch, payload):
    import dashboard_services.providers.espn_draft as ed
    monkeypatch.setattr(ed, "fetch_espn_draft_payload", lambda season, league_id: payload)
    monkeypatch.setattr(ed, "_espn_api", lambda: _FakeEspn())
    return ESPNDraftSyncProvider()


def test_provider_in_progress_snapshot(monkeypatch):
    payload = _detail(picks=[_pick(1, 4039057, 1), _pick(2, 4241479, 2)], drafted=False, in_progress=True)
    snap = _provider(monkeypatch, payload).get_snapshot("99", 2026, viewer_user_id="{AAA}", viewer_roster_id="1")
    assert snap.status == "drafting"
    assert snap.source == "espn"
    assert snap.draft_id == "espn_99_2026"
    assert snap.viewer_team_id == "1"
    assert [p.overall_pick for p in snap.picks] == [1, 2]
    body = snapshot_to_live_payload(snap)
    assert body["status"] == "drafting"
    assert "espn_s2" not in body and "swid" not in body
    assert body["picks"][0]["player_id"] == "5938"


def test_provider_not_started(monkeypatch):
    snap = _provider(monkeypatch, _detail(picks=[], drafted=False, in_progress=False)).get_snapshot("1", 2026)
    assert snap.status == "pre_draft"
    assert snap.picks == []


def test_provider_predraft_placeholder_grid_is_empty(monkeypatch):
    placeholders = [
        _pick(i, 0, ((i - 1) % 4) + 1, rnd=(i - 1) // 4 + 1, slot=((i - 1) % 4) + 1)
        for i in range(1, 17)
    ]
    snap = _provider(monkeypatch, _detail(picks=placeholders, drafted=False, in_progress=False)).get_snapshot("1", 2026)
    assert snap.status == "pre_draft"
    assert snap.picks == []
    body = snapshot_to_live_payload(snap)
    assert body["picks"] == []
    assert body["status"] == "pre_draft"


def test_provider_complete(monkeypatch):
    picks = [_pick(i, 4039057 if i == 1 else 4241479, ((i - 1) % 4) + 1) for i in range(1, 5)]
    snap = _provider(monkeypatch, _detail(picks=picks, drafted=True, in_progress=False)).get_snapshot("1", 2026)
    assert snap.status == "complete"


def test_temporary_espn_error(monkeypatch):
    import dashboard_services.providers.espn_draft as ed
    from dashboard_services.draft_sync import DraftSyncUnavailableError

    class Boom(Exception):
        pass

    monkeypatch.setattr(ed, "_espn_api", lambda: _FakeEspn())

    def fail(season, league_id):
        raise ed.DraftSyncUnavailableError("ESPN is temporarily unavailable.")

    monkeypatch.setattr(ed, "fetch_espn_draft_payload", fail)
    try:
        ESPNDraftSyncProvider().get_snapshot("1", 2026)
        assert False, "expected unavailable"
    except DraftSyncUnavailableError as exc:
        assert exc.retry is True
        assert "espn_s2" not in str(exc)


def test_auth_error_does_not_retry(monkeypatch):
    import dashboard_services.providers.espn_draft as ed

    def fail(season, league_id):
        raise ed.DraftSyncAuthError("ESPN denied access to this league.")

    monkeypatch.setattr(ed, "fetch_espn_draft_payload", fail)
    try:
        ESPNDraftSyncProvider().get_snapshot("1", 2026)
        assert False, "expected auth"
    except DraftSyncAuthError as exc:
        assert exc.retry is False
        assert "secret" not in str(exc).lower()


def test_debug_log_is_sanitized(monkeypatch, caplog):
    import logging
    import dashboard_services.providers.espn_draft as ed
    monkeypatch.setenv("ESPN_DRAFT_SYNC_DEBUG", "1")
    caplog.set_level(logging.INFO)
    payload = _detail(picks=[_pick(1, 4039057, 1)], drafted=False, in_progress=True)
    _provider(monkeypatch, payload).get_snapshot("77", 2026)
    text = caplog.text
    assert "league_id=77" in text
    assert "inProgress=True" in text or "inProgress=true" in text.lower() or "inProgress=" in text
    assert "espn_s2" not in text
    assert "SWID" not in text
    assert "cookie" not in text.lower()


def test_unresolved_player_is_logged_without_secrets(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    payload = _detail(picks=[_pick(1, 111, 1)], drafted=False, in_progress=True)
    _provider(monkeypatch, payload).get_snapshot("77", 2026)
    assert "unresolved ESPN player mapping" in caplog.text
    assert "espn_player_id=111" in caplog.text
    assert "espn_s2" not in caplog.text


def test_automatic_to_manual_fallback_when_picks_never_arrive():
    from dashboard_services.draft_sync import espn_live_should_fallback
    assert espn_live_should_fallback(
        in_progress=True, status="drafting", picks_observed=False,
        detail_present=False, ever_grew=False, stall_polls=3, stall_limit=8,
    )
    assert espn_live_should_fallback(
        in_progress=True, status="drafting", picks_observed=True,
        detail_present=True, ever_grew=False, stall_polls=8, stall_limit=8,
        pick_count=0,
    )
    assert not espn_live_should_fallback(
        in_progress=True, status="drafting", picks_observed=True,
        detail_present=True, ever_grew=True, stall_polls=20, stall_limit=8,
        pick_count=20,
    )
    assert not espn_live_should_fallback(
        in_progress=False, status="pre_draft", picks_observed=True,
        detail_present=True, ever_grew=False, stall_polls=20, stall_limit=8,
    )


def test_fingerprint_changes_when_picks_grow():
    a = _norm(_detail(picks=[_pick(1, 4039057, 1)], in_progress=True))
    b = _norm(_detail(picks=[_pick(1, 4039057, 1), _pick(2, 4241479, 2)], in_progress=True))
    from dashboard_services.draft_sync import DraftSyncSnapshot
    sa = DraftSyncSnapshot(source="espn", draft_id="x", league_id="1", season=2026, status="drafting", picks=a)
    sb = DraftSyncSnapshot(source="espn", draft_id="x", league_id="1", season=2026, status="drafting", picks=b)
    assert snapshot_fingerprint(sa) != snapshot_fingerprint(sb)


def test_parse_espn_draft_id():
    assert parse_espn_draft_id("espn_12345_2026") == ("12345", 2026)
    assert parse_espn_draft_id("espn_12_34_2026") == ("12_34", 2026)
    assert parse_espn_draft_id("sleeper-abc") is None
    assert make_espn_draft_id("99", 2026) == "espn_99_2026"


def test_poll_interval_clamped(monkeypatch):
    monkeypatch.setenv("ESPN_DRAFT_SYNC_POLL_SECONDS", "3")
    assert espn_draft_sync_poll_ms() == 5000
    monkeypatch.setenv("ESPN_DRAFT_SYNC_POLL_SECONDS", "30")
    assert espn_draft_sync_poll_ms() == 10000
    monkeypatch.setenv("ESPN_DRAFT_SYNC_POLL_SECONDS", "8")
    assert espn_draft_sync_poll_ms() == 8000


def test_registry_unknown_platform():
    try:
        get_draft_sync_provider("sleeper")
        assert False
    except DraftSyncUnsupportedError:
        pass


def test_registry_espn():
    provider = get_draft_sync_provider("espn")
    assert provider.source == "espn"


def test_http_200_without_usable_picks_is_not_success(monkeypatch):
    """A 200 with missing draftDetail is not treated as a live feed."""
    snap = _provider(monkeypatch, {"id": 1, "status": "ok"}).get_snapshot("1", 2026)
    assert snap.live_detail_present is False
    assert snap.picks_observed is False
    assert snap.picks == []
    assert snap.status == "unknown"
