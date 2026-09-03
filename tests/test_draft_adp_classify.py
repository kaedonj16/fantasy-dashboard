"""Unit tests for draft-type classification (draft_adp_crawler._classify_draft).

The Sleeper league type drives this: 0 = redraft, 1 = keeper, 2 = dynasty.
Keeper drafts must NOT count as redraft — kept rosters skew them toward rookies.
"""
from datetime import datetime

from data_building.trade_intel.draft_adp_crawler import (
    _classify_draft, _draft_started_at,
)


def _d(rounds):
    return {"settings": {"rounds": rounds}}


# ── dynasty (type 2) ─────────────────────────────────────────────────────────

def test_dynasty_full_draft_is_startup():
    assert _classify_draft(_d(15), league_type=2) == "startup"
    assert _classify_draft(_d(10), league_type=2) == "startup"


def test_dynasty_short_draft_is_rookie():
    assert _classify_draft(_d(4), league_type=2) == "rookie"
    assert _classify_draft(_d(1), league_type=2) == "rookie"


def test_dynasty_ambiguous_is_skipped():
    assert _classify_draft(_d(7), league_type=2) is None


# ── true redraft (type 0) ────────────────────────────────────────────────────

def test_redraft_full_draft_is_redraft():
    assert _classify_draft(_d(16), league_type=0) == "redraft"


def test_redraft_short_draft_is_skipped():
    assert _classify_draft(_d(5), league_type=0) is None


# ── keeper (type 1) — never redraft ──────────────────────────────────────────

def test_keeper_full_draft_is_not_redraft():
    # This is the bug fix: a keeper full draft used to be labeled 'redraft',
    # which pulled rookies (who go ~1.01 once vets are kept) into redraft ADP.
    assert _classify_draft(_d(15), league_type=1) is None


def test_keeper_short_draft_is_skipped():
    assert _classify_draft(_d(4), league_type=1) is None


# ── misc ─────────────────────────────────────────────────────────────────────

def test_missing_rounds_is_skipped():
    assert _classify_draft({"settings": {}}, league_type=0) is None
    assert _classify_draft({}, league_type=2) is None


def test_unknown_league_type_is_skipped():
    assert _classify_draft(_d(15), league_type=9) is None


# ── draft_started_at parsing (Live ADP window) ───────────────────────────────

def test_draft_started_at_prefers_start_time_ms():
    # 2024-01-01T00:00:00Z in ms
    got = _draft_started_at({"start_time": 1_704_067_200_000, "created": 1})
    assert isinstance(got, datetime)
    assert got.year == 2024 and got.month == 1 and got.day == 1


def test_draft_started_at_accepts_seconds():
    got = _draft_started_at({"start_time": 1_704_067_200})
    assert got is not None and got.year == 2024


def test_draft_started_at_falls_back_to_created():
    got = _draft_started_at({"created": 1_704_067_200_000})
    assert got is not None and got.year == 2024


def test_draft_started_at_missing_returns_none():
    assert _draft_started_at({}) is None
    assert _draft_started_at({"start_time": 0}) is None
