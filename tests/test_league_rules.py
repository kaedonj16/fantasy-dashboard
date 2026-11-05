"""League rules record (Wave 1): precedence, overlay, confirmation.

The record is the league's ground truth for playoff format / waiver
system / keepers, resolved user_confirmed > feed > derived >
default_unverified and applied to ctxs as a serve-time overlay. Storage
is stubbed at the module's fetch/store seams so these tests pin the
resolution logic without a database; the DB paths fail soft by design.
"""
import pytest

from utils import league_rules as lr


@pytest.fixture()
def store(monkeypatch):
    holder = {"stored": None, "saved": []}

    def fake_fetch(platform, league_id, season):
        return holder["stored"]

    def fake_store(record):
        holder["saved"].append(dict(record))
        holder["stored"] = dict(record)
        return True

    monkeypatch.setattr(lr, "_fetch_record", fake_fetch)
    monkeypatch.setattr(lr, "_store_record", fake_store)
    lr._DERIVE_ATTEMPTED.clear()
    return holder


def _record(**kw):
    rec = lr._blank_record("fleaflicker", "92916", 2026)
    rec.update(kw)
    return rec


# ── Precedence ──────────────────────────────────────────────────────────

def test_user_confirmed_beats_feed(store):
    store["stored"] = _record(
        source=lr.SOURCE_USER_CONFIRMED, playoff_teams=8,
        playoff_week_start=15,
    )
    out = lr.resolve_league_rules(
        "sleeper", "92916", 2026,
        provider_settings={"playoff_teams": 6, "playoff_week_start": 14},
        total_rosters=12,
    )
    assert out["source"] == lr.SOURCE_USER_CONFIRMED
    assert out["playoff_teams"] == 8
    assert store["saved"] == []


def test_feed_beats_derived_and_default(store):
    deriver_calls = []

    def deriver(platform, league_id, season):
        deriver_calls.append(1)
        return {"playoff_teams": 4, "playoff_week_start": 15}

    out = lr.resolve_league_rules(
        "sleeper", "abc", 2026,
        provider_settings={"playoff_teams": 8, "playoff_week_start": 15},
        total_rosters=12, deriver=deriver,
    )
    assert out["source"] == lr.SOURCE_FEED
    assert out["playoff_teams"] == 8
    assert deriver_calls == []


def test_derived_beats_default(store):
    out = lr.resolve_league_rules(
        "fleaflicker", "92916", 2026,
        provider_settings={"playoff_teams": 6, "playoff_week_start": 15},
        total_rosters=12,
        deriver=lambda p, l, s: {"playoff_teams": 8,
                                 "playoff_week_start": 15},
    )
    assert out["source"] == lr.SOURCE_DERIVED
    assert out["playoff_teams"] == 8


def test_default_unverified_when_nothing_better(store):
    out = lr.resolve_league_rules(
        "fleaflicker", "92916", 2026,
        provider_settings={"playoff_teams": 6, "playoff_week_start": 15},
        total_rosters=12, deriver=lambda p, l, s: None,
    )
    assert out["source"] == lr.SOURCE_DEFAULT
    # The provider's silent defaults are carried, but labelled.
    assert out["playoff_teams"] == 6
    assert out["playoff_week_start"] == 15


def test_stored_feed_survives_a_settings_blip(store):
    store["stored"] = _record(
        platform="sleeper", league_id="abc",
        source=lr.SOURCE_FEED, playoff_teams=10, playoff_week_start=14,
    )
    out = lr.resolve_league_rules(
        "sleeper", "abc", 2026, provider_settings={}, total_rosters=12,
        deriver=lambda p, l, s: {"playoff_teams": 4},
    )
    assert out["source"] == lr.SOURCE_FEED
    assert out["playoff_teams"] == 10


def test_fleaflicker_settings_are_never_feed(store):
    out = lr.resolve_league_rules(
        "fleaflicker", "92916", 2026,
        provider_settings={"playoff_teams": 8, "playoff_week_start": 15},
        total_rosters=12, deriver=lambda p, l, s: None,
    )
    assert out["source"] == lr.SOURCE_DEFAULT


def test_mfl_counts_as_feed_only_with_payload_flag(store):
    no_flag = lr.resolve_league_rules(
        "mfl", "123", 2026,
        provider_settings={"playoff_teams": 8, "playoff_week_start": 15},
        total_rosters=12, deriver=lambda p, l, s: None,
    )
    assert no_flag["source"] == lr.SOURCE_DEFAULT
    store["stored"] = None
    lr._DERIVE_ATTEMPTED.clear()
    flagged = lr.resolve_league_rules(
        "mfl", "123", 2026,
        provider_settings={
            "playoff_teams": 8, "playoff_week_start": 15,
            "playoff_teams_from_feed": True,
        },
        total_rosters=12, deriver=lambda p, l, s: None,
    )
    assert flagged["source"] == lr.SOURCE_FEED
    assert flagged["playoff_teams"] == 8


def test_playoff_teams_clamped_to_league_size(store):
    out = lr.resolve_league_rules(
        "sleeper", "abc", 2026,
        provider_settings={"playoff_teams": 20, "playoff_week_start": 15},
        total_rosters=12,
    )
    assert out["playoff_teams"] == 12


def test_resolution_is_fail_soft_without_a_database(monkeypatch):
    monkeypatch.setattr(lr, "_ensure_table", lambda: False)
    lr._MEMO.clear()
    lr._DERIVE_ATTEMPTED.clear()
    out = lr.resolve_league_rules(
        "fleaflicker", "92916", 2026,
        provider_settings={"playoff_teams": 6, "playoff_week_start": 15},
        total_rosters=12,
    )
    assert out["source"] == lr.SOURCE_DEFAULT
    assert out["playoff_teams"] == 6


# ── Confirmation ────────────────────────────────────────────────────────

def test_confirm_validates_range(store):
    with pytest.raises(ValueError):
        lr.confirm_league_rules(
            "fleaflicker", "92916", 2026, playoff_teams=16,
            playoff_week_start=15, total_rosters=12,
        )
    with pytest.raises(ValueError):
        lr.confirm_league_rules(
            "fleaflicker", "92916", 2026, playoff_teams=8,
            playoff_week_start=15, waiver_type="sideways",
            total_rosters=12,
        )


def test_confirm_stores_user_confirmed_record(store):
    out = lr.confirm_league_rules(
        "fleaflicker", "92916", 2026, playoff_teams=8,
        playoff_week_start=15, waiver_type="rolling",
        waiver_process_time="03:00", waiver_process_days="daily",
        account_id=42, total_rosters=12,
    )
    assert out["source"] == lr.SOURCE_USER_CONFIRMED
    assert out["playoff_teams"] == 8
    assert out["waiver_type"] == "rolling"
    assert out["waiver_process_time"] == "03:00"
    assert out["confirmed_by"] == 42


# ── Overlay ─────────────────────────────────────────────────────────────

def _ctx(settings):
    return {
        "platform": "fleaflicker", "league_id": "92916", "season": 2026,
        "league_settings": dict(settings),
        "rosters": [{"roster_id": i} for i in range(1, 13)],
    }


def test_overlay_applies_derived_values_but_not_verification():
    # A Redis-hydrated ctx carries the provider's defaulted settings;
    # the overlay must correct them at serve time.
    ctx = _ctx({"playoff_teams": 6, "playoff_week_start": 15})
    record = _record(
        source=lr.SOURCE_DERIVED, playoff_teams=8, playoff_week_start=15,
        updated_at="2026-11-04T00:00:00+00:00",
    )
    lr.apply_rules_overlay(ctx, record=record)
    assert ctx["league_settings"]["playoff_teams"] == 8
    # Derived is enough for odds/picture/bracket, never for the
    # championship-odds verification flag.
    assert ctx["league_settings"]["playoff_format_verified"] is False
    assert ctx["_league_rules"]["source"] == lr.SOURCE_DERIVED


def test_overlay_user_confirmed_verifies_format():
    ctx = _ctx({"playoff_teams": 6, "playoff_week_start": 15})
    record = _record(
        source=lr.SOURCE_USER_CONFIRMED, playoff_teams=8,
        playoff_week_start=15, updated_at="2026-11-04T00:00:00+00:00",
    )
    lr.apply_rules_overlay(ctx, record=record)
    assert ctx["league_settings"]["playoff_teams"] == 8
    assert ctx["league_settings"]["playoff_format_verified"] is True


def test_overlay_keeps_provider_value_when_record_field_missing():
    ctx = _ctx({"playoff_teams": 6, "playoff_week_start": 15})
    record = _record(source=lr.SOURCE_FEED, playoff_teams=8,
                     playoff_week_start=None)
    lr.apply_rules_overlay(ctx, record=record)
    assert ctx["league_settings"]["playoff_teams"] == 8
    assert ctx["league_settings"]["playoff_week_start"] == 15
    assert ctx["league_settings"]["playoff_format_verified"] is True


# ── Fingerprint ─────────────────────────────────────────────────────────

# ── Keeper legality (PR 2) ──────────────────────────────────────────

def test_keeper_map_from_ctx_rosters():
    ctx = {"rosters": [
        {"roster_id": 1, "players": ["a", "b"], "keepers": ["a"]},
        {"roster_id": 2, "players": ["c"], "keepers": ["c", "d"]},
        {"roster_id": 3, "players": ["e"]},
    ]}
    by_roster = lr.keeper_locked_by_roster(ctx)
    assert by_roster == {"1": frozenset({"a"}), "2": frozenset({"c", "d"})}
    assert lr.keeper_locked_ids(ctx) == frozenset({"a", "c", "d"})
    assert lr.keeper_locked_ids(ctx, [2]) == frozenset({"c", "d"})
    assert lr.keeper_locked_ids(ctx, ["3"]) == frozenset()


def test_keeper_map_unknown_league_is_empty(store):
    ctx = {"platform": "fleaflicker", "league_id": "999", "season": 2026,
           "rosters": [{"roster_id": 1, "players": ["a"]}]}
    assert lr.keeper_locked_by_roster(ctx) == {}
    assert lr.keeper_locked_ids(ctx) == frozenset()


def test_keeper_map_falls_back_to_the_record(store):
    store["stored"] = _record(
        source=lr.SOURCE_FEED, designated_keepers={"7": ["x", "y"]},
        keepers_known=True)
    ctx = {"platform": "fleaflicker", "league_id": "92916", "season": 2026,
           "rosters": [{"roster_id": 7, "players": ["x", "y", "z"]}]}
    assert lr.keeper_locked_by_roster(ctx) == {"7": frozenset({"x", "y"})}


def test_keeper_map_record_without_known_flag_is_empty(store):
    store["stored"] = _record(
        source=lr.SOURCE_FEED, designated_keepers={"7": ["x"]},
        keepers_known=False)
    ctx = {"platform": "fleaflicker", "league_id": "92916", "season": 2026,
           "rosters": [{"roster_id": 7, "players": ["x"]}]}
    assert lr.keeper_locked_by_roster(ctx) == {}


def test_update_designated_keepers_writes_only_changes(store, monkeypatch):
    writes = []
    monkeypatch.setattr(
        lr, "_store_keepers",
        lambda p, l, s, m: writes.append(dict(m)) or True)
    rosters = [
        {"roster_id": 1, "keepers": ["b", "a"]},
        {"roster_id": 2, "players": ["c"]},  # no data: not in the map
    ]
    assert lr.update_designated_keepers("fleaflicker", "92916", 2026,
                                        rosters) is True
    assert writes == [{"1": ["a", "b"]}]
    # Same designations already stored: no second write.
    store["stored"] = _record(
        designated_keepers={"1": ["a", "b"]}, keepers_known=True)
    assert lr.update_designated_keepers("fleaflicker", "92916", 2026,
                                        rosters) is True
    assert len(writes) == 1
    # No roster carries designation data: nothing to persist.
    assert lr.update_designated_keepers(
        "fleaflicker", "92916", 2026,
        [{"roster_id": 9, "players": ["z"]}]) is False
    assert len(writes) == 1


def test_fingerprint_changes_with_the_record():
    a = _record(playoff_teams=6, playoff_week_start=15,
                updated_at="2026-11-01T00:00:00+00:00")
    b = _record(playoff_teams=8, playoff_week_start=15,
                updated_at="2026-11-01T00:00:00+00:00")
    c = _record(playoff_teams=8, playoff_week_start=15,
                updated_at="2026-11-04T00:00:00+00:00")
    assert lr.rules_fingerprint(a) != lr.rules_fingerprint(b)
    assert lr.rules_fingerprint(b) != lr.rules_fingerprint(c)
    assert lr.ctx_rules_fingerprint({}) == "none"
    assert lr.ctx_rules_fingerprint({"_league_rules": b}) \
        == lr.rules_fingerprint(b)
