"""Draft capital as display-only context under capital-free model 3.7.

Kaedon's ruling (2026-11-23): grades ignore draft capital entirely, but
the projected pick / mock consensus stays surfaced on the site as reader
context. Before 3.7 the grading model embedded a "Projected pick #N"
bullet in the stored key_reasons and the modal's scouting notes showed
it from there; 3.7's _build_reasons writes no capital bullet and stores
capital_delta NULL, so the line is now assembled at the display layer
(rookie_api) from the stored consensus metadata the rankings row
already carries (rookie_mock_draft_consensus). These tests pin:

(a) the modal payload carries the projected-pick display data sourced
    from that stored metadata, with a 3.7-graded row;
(b) display assembly never touches a grade field — rows that differ
    only in consensus data assemble to identical grade fields.

Synthetic data only; no DB needed.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import sys
import os
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services import rookie_api as ra
from dashboard_services.pages.rookies_page import build_prospects_body

GRADE_FIELDS = (
    "prospect_score", "talent_score", "capital_delta",
    "production_score", "efficiency_score", "age_score",
    "breakout_profile_score", "athleticism_score",
    "competition_score", "confidence_score",
    "projected_draft_capital_score",
)

NOTES_37 = (
    "• Elite production profile: dominant volume for his position\n"
    "• Team dominator rating 44%: commanded an outsized share of team production"
)


def _row37(**over):
    """A rankings row as get_rookie_rankings_from_db returns it after a
    3.7 grading run: key_reasons with NO capital bullet, capital_delta
    NULL, and the consensus fields present from the
    rookie_mock_draft_consensus join."""
    row = {
        "player_id": "ROOKIE_2027_JEREMIAH_SMITH",
        "name": "Jeremiah Smith",
        "position": "WR",
        "prospect_score": 92.48,
        "talent_score": 92.48,
        "capital_delta": None,
        "production_score": 96.0,
        "efficiency_score": 88.0,
        "age_score": 90.0,
        "breakout_profile_score": 91.0,
        "athleticism_score": None,
        "competition_score": 89.7,
        "confidence_score": 55.0,
        "projected_draft_capital_score": None,
        "key_reasons": NOTES_37,
        "projected_round": 1,
        "projected_pick": 2,
        "projected_pick_low": 1,
        "projected_pick_high": 5,
        "num_mocks_used": 12,
        "consensus_confidence": Decimal("78.0"),
    }
    row.update(over)
    return row


class TestCapitalDisplayNote:
    def test_full_consensus(self):
        assert ra._capital_display_note(_row37()) == (
            "Projected pick #2 (Round 1) across 12 mock drafts, 78% consensus"
        )

    def test_confidence_suffix_needs_three_mocks(self):
        note = ra._capital_display_note(_row37(num_mocks_used=2))
        assert note == "Projected pick #2 (Round 1) across 2 mock drafts"

    def test_single_mock_singular(self):
        note = ra._capital_display_note(_row37(num_mocks_used=1))
        assert note == "Projected pick #2 (Round 1) across 1 mock draft"

    def test_seed_consensus_no_mock_clause(self):
        # Seed-fallback consensus carries num_mocks_used = 0; never print
        # "across 0 mock drafts".
        note = ra._capital_display_note(
            _row37(num_mocks_used=0, consensus_confidence=Decimal("40.0")))
        assert note == "Projected pick #2 (Round 1)"

    def test_missing_round_derived_from_pick(self):
        note = ra._capital_display_note(
            _row37(projected_pick=40, projected_round=None, num_mocks_used=0))
        assert note == "Projected pick #40 (Round 2)"

    def test_decimal_pick_from_db(self):
        note = ra._capital_display_note(
            _row37(projected_pick=Decimal("2"), num_mocks_used=0))
        assert note == "Projected pick #2 (Round 1)"

    def test_no_pick_no_note(self):
        assert ra._capital_display_note(_row37(projected_pick=None)) is None
        assert ra._capital_display_note({}) is None

    def test_missing_confidence_no_suffix_no_crash(self):
        note = ra._capital_display_note(_row37(consensus_confidence=None))
        assert note == "Projected pick #2 (Round 1) across 12 mock drafts"


class TestWithCapitalNote:
    def test_appends_to_37_reasons(self):
        d = _row37()
        ra._with_capital_note(d)
        lines = d["key_reasons"].split("\n")
        assert len(lines) == 3
        assert lines[:2] == NOTES_37.split("\n")
        assert lines[2] == (
            "• Projected pick #2 (Round 1) across 12 mock drafts, 78% consensus"
        )

    def test_empty_and_none_reasons(self):
        for empty in ("", None):
            d = _row37(key_reasons=empty)
            ra._with_capital_note(d)
            assert d["key_reasons"] == (
                "• Projected pick #2 (Round 1) across 12 mock drafts, "
                "78% consensus"
            )

    def test_no_duplicate_for_pre37_rows(self):
        stored = NOTES_37 + "\n• Projected pick #2 (Round 1) across 9 mock drafts"
        d = _row37(key_reasons=stored)
        ra._with_capital_note(d)
        assert d["key_reasons"] == stored

    def test_no_duplicate_for_actual_pick_rows(self):
        stored = NOTES_37 + "\n• Drafted: Pick #2 (Round 1)"
        d = _row37(key_reasons=stored)
        ra._with_capital_note(d)
        assert d["key_reasons"] == stored

    def test_no_consensus_row_unchanged(self):
        d = _row37(projected_pick=None, projected_round=None,
                   num_mocks_used=None, consensus_confidence=None)
        before = dict(d)
        ra._with_capital_note(d)
        assert d == before

    def test_grade_fields_untouched(self):
        d = _row37()
        before = {k: d.get(k) for k in GRADE_FIELDS}
        keys_before = set(d)
        ra._with_capital_note(d)
        assert {k: d.get(k) for k in GRADE_FIELDS} == before
        assert set(d) == keys_before


class TestDisplayOnlySeparation:
    def test_consensus_changes_move_no_grade_field(self):
        a = ra._with_capital_note(ra._row_to_dict(_row37()))
        b = ra._with_capital_note(ra._row_to_dict(_row37(
            projected_pick=250, projected_round=7, projected_pick_low=210,
            projected_pick_high=260, num_mocks_used=4,
            consensus_confidence=Decimal("55.0"))))
        for k in GRADE_FIELDS:
            assert a[k] == b[k], k
        assert a["key_reasons"] != b["key_reasons"]
        assert "pick #250" in b["key_reasons"]

    def test_consensus_presence_moves_no_grade_field(self):
        a = ra._with_capital_note(ra._row_to_dict(_row37()))
        b = ra._with_capital_note(ra._row_to_dict(_row37(
            projected_pick=None, projected_round=None,
            projected_pick_low=None, projected_pick_high=None,
            num_mocks_used=None, consensus_confidence=None)))
        for k in GRADE_FIELDS:
            assert a[k] == b[k], k

    def test_cached_row_not_mutated(self):
        # Routes assemble on the _row_to_dict copy; the cached rankings
        # row must keep its stored key_reasons byte-identical.
        cached = _row37()
        payload = ra._with_capital_note(ra._row_to_dict(cached))
        assert "Projected pick #2" in payload["key_reasons"]
        assert cached["key_reasons"] == NOTES_37


class TestModalPayload:
    def test_player_payload_carries_capital_display_from_metadata(self):
        # The /player assembly: rankings row copy + capital note.
        payload = ra._with_capital_note(ra._row_to_dict(_row37()))
        # Consensus display fields ride the payload from the stored
        # rookie_mock_draft_consensus metadata, not the grading model.
        assert payload["projected_pick"] == 2
        assert payload["projected_round"] == 1
        assert payload["num_mocks_used"] == 12
        assert payload["consensus_confidence"] == Decimal("78.0")
        # The notes line the modal renders is restored.
        assert "Projected pick #2 (Round 1)" in payload["key_reasons"]
        # Grade purity is preserved in the same payload.
        assert payload["capital_delta"] is None
        assert payload["prospect_score"] == 92.48
        assert payload["talent_score"] == payload["prospect_score"]

    def test_page_modal_reads_pick_from_row(self):
        # The modal header meta and Mock Pick hero card read the row's
        # projected_pick / projected_round (the stored consensus fields).
        html = build_prospects_body(False)
        assert "Mock Pick" in html
        assert "r.projected_pick" in html
        assert "r.projected_round" in html
