"""Backtest draft-class loader: draft_picks fill for missing picks.

The nflverse roster CSV leaves draft_number empty for a slice of
2016-2019 draftees (109 drafted skill players, incl. overall picks
8, 9, 12, 15, 19 and 21). Without a fill, _build_prospect_dicts
defaults those players to pick 300 and the gate prices their draft
capital at the floor (O.J. Howard, the 19th pick in 2017, scored
capital 2.0 in every gate run through 2026-11-20).

Pinned here, with _fetch_csv faked (no network):
- a missing roster pick is filled from draft_picks by gsis_id when
  the season matches;
- a present roster pick is never overwritten;
- a player with no draft_picks row stays pick 0 (the UDFA path,
  which downstream defaults to 300);
- a draft_picks row from another season does not fill.
"""

import os
import sys

import pytest

pytest.importorskip("pandas")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import backtest_prospect_model as bt  # noqa: E402


def _roster_row(gid, name, pick, pos="TE"):
    return {
        "rookie_year": "2017",
        "position": pos,
        "gsis_id": gid,
        "draft_number": pick,
        "full_name": name,
        "college": "Somewhere",
        "birth_date": "1995-01-01",
    }


def _fake_fetch(roster_rows, pick_rows):
    def fetch(url, **kwargs):
        if "draft_picks" in url:
            return pick_rows
        if "roster_" in url:
            return roster_rows
        return []
    return fetch


def test_missing_pick_filled_from_draft_picks(monkeypatch):
    roster = [_roster_row("00-0033879", "O.J. Howard", "")]
    picks = [{"season": "2017", "gsis_id": "00-0033879", "pick": "19"}]
    monkeypatch.setattr(bt, "_fetch_csv", _fake_fetch(roster, picks))
    rows = bt._load_draft_class(2017)
    assert len(rows) == 1
    assert rows[0]["draft_pick"] == 19


def test_present_pick_never_overwritten(monkeypatch):
    roster = [_roster_row("00-0033881", "Evan Engram", "23")]
    picks = [{"season": "2017", "gsis_id": "00-0033881", "pick": "99"}]
    monkeypatch.setattr(bt, "_fetch_csv", _fake_fetch(roster, picks))
    rows = bt._load_draft_class(2017)
    assert rows[0]["draft_pick"] == 23


def test_undrafted_player_stays_zero(monkeypatch):
    roster = [_roster_row("00-0033261", "Bug Howard", "", pos="WR")]
    monkeypatch.setattr(bt, "_fetch_csv", _fake_fetch(roster, []))
    rows = bt._load_draft_class(2017)
    assert rows[0]["draft_pick"] == 0


def test_other_season_pick_does_not_fill(monkeypatch):
    roster = [_roster_row("00-0033879", "O.J. Howard", "")]
    picks = [{"season": "2018", "gsis_id": "00-0033879", "pick": "19"}]
    monkeypatch.setattr(bt, "_fetch_csv", _fake_fetch(roster, picks))
    rows = bt._load_draft_class(2017)
    assert rows[0]["draft_pick"] == 0
