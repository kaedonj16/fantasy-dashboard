"""
NFL calendar phase detection and phase-based weighting logic.

Detects the current phase of the NFL calendar (offseason, post-draft, in-season, etc.)
and provides appropriate component score weights for each phase.
"""

from datetime import date
from typing import Dict
from .config import PHASE_WEIGHTS


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
            'start': (1, 1),    # January 1
            'end': (2, 28)      # February 28
        },
        'post_free_agency': {
            'start': (3, 1),    # March 1
            'end': (4, 24)      # April 24 (day before typical draft)
        },
        'post_draft': {
            'start': (4, 25),   # April 25 (day after draft typically ends)
            'end': (7, 31)      # July 31
        },
        'preseason': {
            'start': (8, 1),    # August 1
            'end': (9, 4)       # September 4 (before season starts)
        },
        'in_season': {
            'start': (9, 5),    # September 5
            'end': (12, 31)     # December 31 (wraps to January)
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

        # Post-free agency: March - late April
        elif month == 3:
            return 'post_free_agency'
        elif month == 4 and day < 25:
            return 'post_free_agency'

        # Post-draft: Late April - July
        elif month == 4 and day >= 25:
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

        Args:
            component_scores: Dictionary of component name -> score
            phase: Current NFL calendar phase

        Returns:
            Aggregate score (0-100)
        """
        weights = cls.get_phase_weights(phase)

        total = 0.0
        for component, score in component_scores.items():
            weight = weights.get(component, 0.0)
            total += score * weight

        # Ensure result is in valid range
        return max(0.0, min(100.0, total))

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
