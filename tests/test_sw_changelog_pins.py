"""Designated home for the two genuine whole-list pin behaviours.

Everything else that used to pin the SW cache version or a changelog
position now derives its subject (see tests/pin_helpers.py). Only two
assertions in the suite are allowed to be positional/exact:

* the served SW cache name must be the content-derived one for the
  current tree (utils/sw_cache.py computes it from the cached asset
  set; there is no manual version pin left to sync, so this is the
  consolidated mechanism assertion, kept here so it runs exactly once
  as a named behaviour);
* CHANGELOG[0] must be the newest entry: app._notify_changelog_on_startup
  broadcasts CHANGELOG[0] as the latest announcement and the changelog
  API serves the list newest-first, so head-is-newest is real product
  behaviour, not an accident of authoring order.
"""
from __future__ import annotations

from pin_helpers import assert_sw_cache_current


def test_sw_cache_name_matches_derived_expectation():
    assert_sw_cache_current()


def test_changelog_head_is_the_newest_entry():
    from dashboard_services.changelog import CHANGELOG

    assert CHANGELOG, "changelog must not be empty"
    dates = [entry["date"] for entry in CHANGELOG]
    # Newest-first surfacing: the head is a newest-dated entry and dates
    # never increase down the list (relative order only; prepending a new
    # newest entry keeps this green with no edits here).
    assert dates[0] == max(dates)
    assert all(dates[i] >= dates[i + 1] for i in range(len(dates) - 1))
