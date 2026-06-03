"""
NFL calendar phase detection and phase-based weighting logic.

Detects the current phase of the NFL calendar (offseason, post-draft, in-season, etc.)
and provides appropriate component score weights for each phase.
"""

from datetime import date
from typing import Dict

from .config import (
    PHASE_WEIGHTS,
    BREAKOUT_GATE_OPP_MIN,
    BREAKOUT_GATE_COMP_MIN,
    BREAKOUT_GATE_READY_MIN,
    BREAKOUT_GATE_TRAJ_MIN,
    BREAKOUT_ASCENSION_READY_MIN,
    BREAKOUT_ASCENSION_TRAJ_MIN,
    BREAKOUT_ASCENSION_SCORE_CAP,
    BREAKOUT_GATE_FAIL_CAP,
    BREAKOUT_CURVE_PIVOT,
    BREAKOUT_CURVE_SLOPE,
)


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

        When competition signals (opportunity_opened, competition_removed) are
        both 0 they are treated as *unavailable* rather than genuine zeros, and
        the remaining components are renormalized to fill the 0-100 range.
        This prevents scores being structurally capped at ~45 when the
        roster-changes DB table hasn't been populated.

        competition_added_penalty = 0 means no new competition was added - a
        neutral/positive outcome - so it IS kept in the denominator.

        Args:
            component_scores: Dictionary of component name -> score (0-100)
            phase: Current NFL calendar phase

        Returns:
            Aggregate score (0-100)
        """
        weights = cls.get_phase_weights(phase)

        opp_opened = component_scores.get('opportunity_opened', 0.0)
        comp_removed = component_scores.get('competition_removed', 0.0)
        comp_added = component_scores.get('competition_added_penalty', 0.0)

        absent: set = set()

        # competition_added_penalty ranges -38..0 and is a pure penalty. A value
        # of 0 means "no new competition added" — the *ideal* breakout setup, not
        # a low score on a 0-100 scale. Averaging a 0 in at full weight drags the
        # score down and buries legitimate opportunity breakouts, so exclude it
        # unless it is an actual (negative) penalty.
        if comp_added >= 0.0:
            absent.add('competition_added_penalty')

        # Detect DB-absent / stable-roster state: when opportunity AND
        # competition-removed are also both zero, there is no roster-change signal
        # for this player. Exclude those too so the score reflects the components
        # we do have (readiness, trajectory, environment) rather than being
        # dragged toward zero.
        competition_data_absent = (opp_opened == 0.0 and comp_removed == 0.0 and comp_added == 0.0)
        if competition_data_absent:
            absent.update({'opportunity_opened', 'competition_removed'})

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
        raw = max(0.0, min(100.0, total / active_weight))

        # ── Steepen the curve so the mediocre middle collapses ────────────────
        # The raw aggregate is an average and clusters near ~50; pivot+stretch
        # pushes average players below the candidate floor while keeping genuine
        # standouts high. See config BREAKOUT_CURVE_* for tuning.
        curved = BREAKOUT_CURVE_PIVOT + (raw - BREAKOUT_CURVE_PIVOT) * BREAKOUT_CURVE_SLOPE
        curved = max(0.0, min(100.0, curved))

        # ── Qualification gates: a candidate qualifies via EITHER of two paths.
        opp = component_scores.get('opportunity_opened', 0.0)
        comp_removed = component_scores.get('competition_removed', 0.0)
        readiness = component_scores.get('player_readiness', 0.0)
        trajectory = component_scores.get('role_trajectory', 0.0)

        # Path A — opportunity-driven: a real opening AND the ability to take it.
        # A player on a stable roster (no opening) has opportunity_ok=False and
        # must instead qualify via the ascension path below.
        opportunity_ok = (
            opp >= BREAKOUT_GATE_OPP_MIN
            or comp_removed >= BREAKOUT_GATE_COMP_MIN
        )
        readiness_ok = (
            readiness >= BREAKOUT_GATE_READY_MIN
            or trajectory >= BREAKOUT_GATE_TRAJ_MIN
        )
        opportunity_breakout = opportunity_ok and readiness_ok

        # Path B — ascension-driven: no new opening required, but the player must
        # be clearly ascending on their own (the Year-2 leap), which demands a
        # high readiness AND a strong upward trajectory.
        ascension_breakout = (
            readiness >= BREAKOUT_ASCENSION_READY_MIN
            and trajectory >= BREAKOUT_ASCENSION_TRAJ_MIN
        )

        if not (opportunity_breakout or ascension_breakout):
            curved = min(curved, BREAKOUT_GATE_FAIL_CAP)
        elif not opportunity_breakout:
            # Qualified via ascension but NOT via a real opening — cap below the
            # genuine opportunity-driven breakouts. This applies to any player
            # without a meaningful opportunity/competition opening (not only the
            # exactly-zero case), so an ascending young starter with no vacancy
            # can't outrank a true opportunity breakout.
            curved = min(curved, BREAKOUT_ASCENSION_SCORE_CAP)

        return curved

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
