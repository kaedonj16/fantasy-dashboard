"""
NFL calendar phase detection and phase-based weighting logic.

Detects the current phase of the NFL calendar (offseason, post-draft, in-season, etc.)
and provides appropriate component score weights for each phase.

Environment variables:
    NFL_DRAFT_DATE: Override the draft boundary used for post_free_agency / post_draft
                    phase detection. Format: MMDD (e.g. "0424" for April 24).
                    Defaults to "0425" (April 25) when not set.
"""

import os
from datetime import date
from typing import Dict

from .config import PHASE_WEIGHTS

# Read draft date override once at import time.
# The NFL draft shifts by a day or two each year; set NFL_DRAFT_DATE=MMDD in
# your environment to avoid hard-coding it here.
_draft_date_str = os.environ.get("NFL_DRAFT_DATE", "0425")
try:
    _DRAFT_MONTH = int(_draft_date_str[:2])
    _DRAFT_DAY = int(_draft_date_str[2:])
except (ValueError, IndexError):
    _DRAFT_MONTH, _DRAFT_DAY = 4, 25  # safe default


class PhaseDetector:
    """
    Detects the current NFL calendar phase based on date.

    Phases:
    - offseason: January - February (pre-free agency)
    - post_free_agency: March - late April (after FA, before draft)
    - post_draft: Late April - July (after NFL draft, before preseason)
    - preseason: August - early September (training camp and preseason games)
    - in_season: September - January (regular season and playoffs)
    """

    # Phase date ranges (month, day tuples)
    # These are approximate and can be adjusted based on actual NFL calendar
    PHASE_BOUNDARIES = {
        'offseason': {
            'start': (1, 1),  # January 1
            'end': (2, 28)  # February 28
        },
        'post_free_agency': {
            'start': (3, 1),  # March 1
            'end': (4, 24)  # April 24 (day before typical draft)
        },
        'post_draft': {
            'start': (4, 25),  # April 25 (day after draft typically ends)
            'end': (7, 31)  # July 31
        },
        'preseason': {
            'start': (8, 1),  # August 1
            'end': (9, 4)  # September 4 (before season starts)
        },
        'in_season': {
            'start': (9, 5),  # September 5
            'end': (12, 31)  # December 31 (wraps to January)
        }
    }

    @classmethod
    def detect_phase(cls, as_of_date: date = None) -> str:
        """
        Detect NFL calendar phase for a given date.

        Args:
            as_of_date: Date to check (defaults to today)

        Returns:
            Phase name: 'offseason', 'post_free_agency', 'post_draft',
                       'preseason', or 'in_season'
        """
        if as_of_date is None:
            as_of_date = date.today()

        month = as_of_date.month
        day = as_of_date.day

        # Check each phase's date range
        # Note: Order matters for overlapping dates

        # Offseason: January - February
        if month <= 2:
            return 'offseason'

        # Post-free agency: March - eve of NFL draft
        elif month == 3:
            return 'post_free_agency'
        elif month == _DRAFT_MONTH and day < _DRAFT_DAY:
            return 'post_free_agency'

        # Post-draft: NFL draft day through July
        elif month == _DRAFT_MONTH and day >= _DRAFT_DAY:
            return 'post_draft'
        elif month in [5, 6, 7]:
            return 'post_draft'

        # Preseason: August - early September
        elif month == 8:
            return 'preseason'
        elif month == 9 and day < 5:
            return 'preseason'

        # In-season: September - December/January
        else:
            return 'in_season'

    @classmethod
    def get_phase_weights(cls, phase: str) -> Dict[str, float]:
        """
        Get component score weights for a given phase.

        Args:
            phase: Phase name

        Returns:
            Dictionary mapping component names to weights (0.0-1.0)

        Raises:
            KeyError: If phase is not recognized
        """
        if phase not in PHASE_WEIGHTS:
            raise KeyError(f"Unknown phase: {phase}. Valid phases: {list(PHASE_WEIGHTS.keys())}")

        return PHASE_WEIGHTS[phase]

    @classmethod
    def calculate_aggregate_score(
            cls,
            component_scores: Dict[str, float],
            phase: str
    ) -> float:
        """
        Calculate weighted aggregate breakout score based on phase.

        When competition signals (opportunity_opened, competition_removed) are
        both 0 they are treated as *unavailable* rather than genuine zeros, and
        the remaining components are renormalized to fill the 0-100 range.
        This prevents scores being structurally capped at ~45 when the
        roster-changes DB table hasn't been populated.

        competition_added_penalty = 0 means no new competition was added — a
        neutral/positive outcome — so it IS kept in the denominator.

        Args:
            component_scores: Dictionary of component name -> score (0-100)
            phase: Current NFL calendar phase

        Returns:
            Aggregate score (0-100)
        """
        weights = cls.get_phase_weights(phase)

        # Detect DB-absent state: all three competition components return 0 when
        # the roster-changes table hasn't been populated. Treat them as absent
        # (exclude from both numerator and denominator) so they don't drag down
        # the renormalized score. Note that competition_added_penalty ranges from
        # -38 to 0 (it is a pure penalty, never positive), so a value of 0 is
        # ambiguous — it could mean "no data" or "no new competition added".
        # When combined with opportunity_opened=0 and competition_removed=0, all
        # three are almost certainly absent rather than genuinely zero.
        opp_opened = component_scores.get('opportunity_opened', 0.0)
        comp_removed = component_scores.get('competition_removed', 0.0)
        comp_added = component_scores.get('competition_added_penalty', 0.0)
        competition_data_absent = (opp_opened == 0.0 and comp_removed == 0.0 and comp_added == 0.0)
        absent = {'opportunity_opened', 'competition_removed', 'competition_added_penalty'} if competition_data_absent else set()

        total = 0.0
        active_weight = 0.0
        for component, score in component_scores.items():
            weight = weights.get(component, 0.0)
            if component in absent:
                continue
            total += score * weight
            active_weight += weight

        if active_weight <= 0:
            return 0.0

        # Renormalize so the active-weight components span the full 0-100 range
        return max(0.0, min(100.0, total / active_weight))

    @classmethod
    def get_phase_description(cls, phase: str) -> str:
        """
        Get human-readable description of a phase.

        Args:
            phase: Phase name

        Returns:
            Description string
        """
        descriptions = {
            'offseason': 'Offseason (Jan-Feb): Pre-free agency period',
            'post_free_agency': 'Post-Free Agency (Mar-Apr): After FA signings, before draft',
            'post_draft': 'Post-Draft (May-Jul): After NFL draft, before preseason',
            'preseason': 'Preseason (Aug-early Sep): Training camp and preseason games',
            'in_season': 'In-Season (Sep-Jan): Regular season and playoffs'
        }
        return descriptions.get(phase, f'Unknown phase: {phase}')


def detect_phase(as_of_date: date = None) -> str:
    """
    Convenience function to detect phase.

    Args:
        as_of_date: Date to check (defaults to today)

    Returns:
        Phase name
    """
    return PhaseDetector.detect_phase(as_of_date)


def get_phase_weights(phase: str) -> Dict[str, float]:
    """
    Convenience function to get phase weights.

    Args:
        phase: Phase name

    Returns:
        Dictionary of component weights
    """
    return PhaseDetector.get_phase_weights(phase)


def calculate_aggregate_score(
        component_scores: Dict[str, float],
        phase: str
) -> float:
    """
    Convenience function to calculate aggregate score.

    Args:
        component_scores: Dictionary of component scores
        phase: Current phase

    Returns:
        Weighted aggregate score (0-100)
    """
    return PhaseDetector.calculate_aggregate_score(component_scores, phase)
