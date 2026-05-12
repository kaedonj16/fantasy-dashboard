"""
Configuration constants for the unified breakout scoring engine.

Contains:
- Phase-based weights for component scores
- Scoring thresholds and scaling factors
- Position-specific constants
"""

from typing import Dict

# ==============================================================================
# PHASE-BASED WEIGHTING SYSTEM
# ==============================================================================
# Weights determine how much each component contributes to the aggregate score
# based on the current NFL calendar phase.

PHASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    'offseason': {
        'opportunity_opened': 0.20,  # High - offseason focus on vacated opportunity
        'competition_removed': 0.20,  # High - who left matters most
        'competition_added_penalty': 0.10,  # Low - signings still happening
        'team_environment': 0.15,
        'player_readiness': 0.25,  # High - historical data important
        'role_trajectory': 0.1,  # Low - no recent in-season data
        'confidence': 0.05
    },
    'post_free_agency': {
        'opportunity_opened': 0.22,
        'competition_removed': 0.18,
        'competition_added_penalty': 0.15,  # Higher - FA signings impact
        'team_environment': 0.15,
        'player_readiness': 0.18,
        'role_trajectory': 0.05,  # Still offseason
        'confidence': 0.07
    },
    'post_draft': {
        'opportunity_opened': 0.20,
        'competition_removed': 0.15,
        'competition_added_penalty': 0.20,  # Highest - draft picks compete heavily
        'team_environment': 0.15,
        'player_readiness': 0.20,  # Draft capital boosts rookies
        'role_trajectory': 0.05,
        'confidence': 0.05
    },
    'preseason': {
        'opportunity_opened': 0.15,  # Lower - opportunity mostly known
        'competition_removed': 0.12,
        'competition_added_penalty': 0.15,  # Roster cuts happening
        'team_environment': 0.13,
        'player_readiness': 0.18,
        'role_trajectory': 0.20,  # Higher - preseason usage trends
        'confidence': 0.07
    },
    'in_season': {
        'opportunity_opened': 0.10,  # Lowest - past data less relevant
        'competition_removed': 0.08,
        'competition_added_penalty': 0.05,  # Roster mostly set
        'team_environment': 0.12,
        'player_readiness': 0.15,
        'role_trajectory': 0.40,  # Highest - recent performance key
        'confidence': 0.10
    }
}

# ==============================================================================
# SCORING THRESHOLDS
# ==============================================================================

# Minimum breakout opportunity score to be considered a candidate
# Set to 40 to be selective while capturing top RB opportunities
MIN_BREAKOUT_SCORE = 40.0

# Component score ranges (most are 0-100)
COMPONENT_SCORE_MIN = 0.0
COMPONENT_SCORE_MAX = 100.0
COMPETITION_PENALTY_MAX = -50.0  # Can go negative

# ==============================================================================
# OPPORTUNITY OPENED SCORE - Thresholds
# ==============================================================================

# Vacated targets that equal max score (100 points)
MAX_VACATED_TARGETS_WR_TE = 150  # 150 targets = max opportunity for WR/TE

# Vacated carries that equal max score for RBs
MAX_VACATED_CARRIES_RB = 250  # 250 carries = 70 points (primary)
MAX_VACATED_TARGETS_RB = 100  # 100 targets = 30 points (secondary)

# Snap share bonus
MAX_SNAP_SHARE_BONUS = 20  # Up to 20 bonus points for high snap share vacated

# QB thresholds
QB_STARTER_SNAP_THRESHOLD = 0.70  # 70%+ snap share = starter left

# ==============================================================================
# COMPETITION REMOVED SCORE - Thresholds
# ==============================================================================

# How much more usage a departed player needed vs current player to be a threat
HIGH_THREAT_MULTIPLIER = 1.5  # 1.5x current player's usage
MEDIUM_THREAT_MULTIPLIER = 1.0  # 1.0x current player's usage

# Points awarded for threats at each level
HIGH_THREAT_MAX_POINTS = 40
MEDIUM_THREAT_MAX_POINTS = 25
LOW_THREAT_MAX_POINTS = 10

# ==============================================================================
# COMPETITION ADDED PENALTY - Thresholds
# ==============================================================================

# Draft pick penalties by round
DRAFT_PENALTY_ROUND_1 = -30
DRAFT_PENALTY_ROUND_2 = -20
DRAFT_PENALTY_ROUND_3 = -10
DRAFT_PENALTY_ROUND_4_PLUS = -5

# Free agent / trade threat levels (based on previous season usage)
FA_HIGH_THREAT_TARGETS = 80  # 80+ targets last season
FA_MEDIUM_THREAT_TARGETS = 50  # 50+ targets last season
FA_HIGH_THREAT_CARRIES = 150  # 150+ carries last season
FA_MEDIUM_THREAT_CARRIES = 80  # 80+ carries last season

FA_HIGH_THREAT_PENALTY = -25
FA_MEDIUM_THREAT_PENALTY = -15
FA_LOW_THREAT_PENALTY = -5

# ==============================================================================
# TEAM ENVIRONMENT SCORE - Thresholds
# ==============================================================================

# Pace scoring (plays per game)
ELITE_PLAYS_PER_GAME = 95  # 95+ plays = max pace score (30)
PACE_SCORE_MAX = 30
PACE_BASELINE = 50  # Below this = 0 points

# Pass rate thresholds
PASS_RATE_SCORE_MAX = 30
HIGH_PASS_RATE = 0.60  # 60%+ = max for WR/TE
BALANCED_PASS_RATE = 0.50  # 50% = balanced

# Offensive ranking (total yards per game)
ELITE_YARDS_PER_GAME = 380
GOOD_YARDS_PER_GAME = 350
AVERAGE_YARDS_PER_GAME = 320

ELITE_OFFENSE_SCORE = 25
GOOD_OFFENSE_SCORE = 18
AVERAGE_OFFENSE_SCORE = 12
BELOW_AVERAGE_OFFENSE_SCORE = 5

# QB quality (pass TD per game)
ELITE_PASS_TD_PER_GAME = 2.0
GOOD_PASS_TD_PER_GAME = 1.5

QB_ELITE_BONUS = 15
QB_GOOD_BONUS = 10
QB_AVERAGE_BONUS = 5

# ==============================================================================
# PLAYER READINESS SCORE - Thresholds
# ==============================================================================

# Age/Experience scoring
SECOND_YEAR_SCORE = 30  # Year 2 players (prime breakout)
THIRD_YEAR_SCORE = 25  # Year 3 players
YEAR_3_4_SCORE = 20  # Years 3-4
ROOKIE_SCORE = 15  # Year 0 (rookies)
VETERAN_SCORE = 5  # Year 5+

YOUNG_AGE_THRESHOLD = 26
VETERAN_AGE_THRESHOLD = 27

# Efficiency thresholds - WR/TE
WR_ELITE_YARDS_PER_TARGET = 9.0
WR_GOOD_YARDS_PER_TARGET = 7.5
WR_AVERAGE_YARDS_PER_TARGET = 6.0

WR_ELITE_CATCH_RATE = 0.70
WR_GOOD_CATCH_RATE = 0.60
WR_AVERAGE_CATCH_RATE = 0.50

# Efficiency thresholds - RB
RB_ELITE_YARDS_PER_CARRY = 5.0
RB_GOOD_YARDS_PER_CARRY = 4.2
RB_AVERAGE_YARDS_PER_CARRY = 3.8

RB_ELITE_YARDS_PER_TARGET = 7.0
RB_GOOD_YARDS_PER_TARGET = 5.5

# Efficiency score maximums
EFFICIENCY_YPT_MAX = 20
EFFICIENCY_CATCH_RATE_MAX = 15
EFFICIENCY_YPC_MAX = 20
EFFICIENCY_RECEIVING_RB_MAX = 15

# Draft capital boost (for rookies only)
DRAFT_CAPITAL_ROUND_1 = 35
DRAFT_CAPITAL_ROUND_2 = 25
DRAFT_CAPITAL_ROUND_3 = 15
DRAFT_CAPITAL_ROUND_4_5 = 5

# Usage baseline scoring (for non-rookies)
WR_ESTABLISHED_TARGETS = 60
WR_BACKUP_TARGETS = 40
WR_ROTATION_TARGETS = 20

RB_ESTABLISHED_CARRIES = 100
RB_BACKUP_CARRIES = 60
RB_ROTATION_CARRIES = 30

ESTABLISHED_USAGE_SCORE = 20
BACKUP_USAGE_SCORE = 15
ROTATION_USAGE_SCORE = 10
MINIMAL_USAGE_SCORE = 5

# ==============================================================================
# ROLE TRAJECTORY SCORE - Thresholds
# ==============================================================================

# Lookback window for trend calculation
DEFAULT_LOOKBACK_DAYS = 14

# Snap share trend thresholds (% increase)
SNAP_ELITE_INCREASE = 30
SNAP_GOOD_INCREASE = 20
SNAP_MODERATE_INCREASE = 10

SNAP_TREND_MAX = 30

# Opportunity share trend thresholds
OPP_ELITE_INCREASE = 25
OPP_GOOD_INCREASE = 15
OPP_MODERATE_INCREASE = 5

OPP_TREND_MAX = 35

# Red zone usage trend thresholds
RZ_ELITE_INCREASE = 30
RZ_GOOD_INCREASE = 15

RZ_TREND_MAX = 20

# Role score improvement thresholds
ROLE_ELITE_IMPROVEMENT = 15
ROLE_GOOD_IMPROVEMENT = 10
ROLE_MODERATE_IMPROVEMENT = 5

ROLE_IMPROVEMENT_MAX = 15

# Neutral score for offseason (no trend data available)
OFFSEASON_NEUTRAL_SCORE = 50

# ==============================================================================
# CONFIDENCE SCORE - Thresholds
# ==============================================================================

# Sample size thresholds
FULL_SEASON_GAMES = 12
FULL_SEASON_TOUCHES = 100

HALF_SEASON_GAMES = 8
HALF_SEASON_TOUCHES = 60

QUARTER_SEASON_GAMES = 4
QUARTER_SEASON_TOUCHES = 30

SAMPLE_FULL_SCORE = 40
SAMPLE_HALF_SCORE = 30
SAMPLE_QUARTER_SCORE = 20
SAMPLE_MINIMAL_SCORE = 10
SAMPLE_ROOKIE_SCORE = 5

# Data completeness scoring
HAS_EFFICIENCY_DATA_SCORE = 10
HAS_ADVANCED_METRICS_SCORE = 10
HAS_USAGE_HISTORY_SCORE = 5

# Usage consistency (variance thresholds)
VERY_CONSISTENT_VARIANCE = 0.2
CONSISTENT_VARIANCE = 0.4
MODERATE_VARIANCE = 0.6

CONSISTENCY_HIGH_SCORE = 20
CONSISTENCY_GOOD_SCORE = 15
CONSISTENCY_MODERATE_SCORE = 10
CONSISTENCY_LOW_SCORE = 5

# Phase certainty scoring
PHASE_CERTAINTY = {
    'in_season': 15,  # Highest - seeing it in real time
    'preseason': 10,  # Preseason games provide data
    'post_draft': 8,  # Roster locked
    'post_free_agency': 6,  # Still some moves possible
    'offseason': 5  # Most uncertain
}

# ==============================================================================
# INJURY STATUS MODIFIER - Thresholds
# ==============================================================================

# Current injury status penalties applied to player_readiness_score
INJURY_STATUS_PENALTIES = {
    'healthy': 0,
    'probable': -2,
    'questionable': -5,
    'doubtful': -12,
    'out': -18,
    'ir': -22,
    'pup': -22,
    'nfi': -20,
    'dnr': -10,  # Did Not Return (in-game)
}

# Injury history discount: games missed last season
INJURY_HISTORY_GAMES_MISSED_MODERATE = 4   # 4-7 games missed
INJURY_HISTORY_GAMES_MISSED_SEVERE = 8     # 8+ games missed
INJURY_HISTORY_MODERATE_PENALTY = -5
INJURY_HISTORY_SEVERE_PENALTY = -12

# ==============================================================================
# AIR YARDS / TARGET QUALITY - Thresholds
# ==============================================================================

# Average depth of target tiers for WR/TE opportunity quality bonus
AIR_YARDS_ELITE_ADOT = 12.0   # Deep threat (12+ yards per target avg)
AIR_YARDS_GOOD_ADOT = 9.0     # Intermediate
AIR_YARDS_AVERAGE_ADOT = 6.5  # Short / underneath

# Max bonus for elite air yards vacated (added to opportunity_opened_score)
AIR_YARDS_ELITE_BONUS = 15
AIR_YARDS_GOOD_BONUS = 8
AIR_YARDS_AVERAGE_BONUS = 3

# Vacated air yards maximum for normalization (elite WR1 seasons)
MAX_VACATED_AIR_YARDS = 1800  # ~1800 vacated air yards = max bonus

# ==============================================================================
# COACHING / OC CHANGE SIGNAL - Thresholds
# ==============================================================================

# Scheme tendencies for new OC (affects WR/TE vs RB)
OC_PASS_HEAVY_THRESHOLD = 0.58   # Pass rate >= 58% = pass-heavy system
OC_RUN_HEAVY_THRESHOLD = 0.44    # Pass rate <= 44% = run-heavy system

# WR/TE bonuses / penalties from OC change
OC_PASS_HEAVY_WR_BONUS = 8
OC_RUN_HEAVY_WR_PENALTY = -6
OC_UNKNOWN_WR_NEUTRAL = 0

# RB bonuses / penalties from OC change (opposite direction)
OC_RUN_HEAVY_RB_BONUS = 8
OC_PASS_HEAVY_RB_PENALTY = -4

# HC change (smaller effect, adds uncertainty)
HC_CHANGE_UNCERTAINTY_PENALTY = -3  # Uncertainty until scheme is known

# ==============================================================================
# QB DOWNGRADE / UPGRADE SIGNAL - Thresholds
# ==============================================================================

# Applied to WR/TE team_environment_score when QB situation changes
QB_UPGRADE_WR_BONUS = 10    # Elite QB replacing good QB, etc.
QB_DOWNGRADE_WR_PENALTY = -14  # Starting QB lost, replaced by inferior option
QB_LATERAL_CHANGE = 0

# QB tier definitions (passer rating proxies)
QB_TIER_ELITE_RATING = 100    # 100+ passer rating = elite
QB_TIER_GOOD_RATING = 92      # 92+ = good
QB_TIER_AVERAGE_RATING = 84   # 84+ = average
QB_TIER_POOR_RATING = 0       # below 84 = poor

# Passer rating penalty/bonus deltas for WR/TE team environment
QB_TIER_WR_SCORES = {
    'elite': 10,
    'good': 5,
    'average': 0,
    'poor': -8,
    'unknown': 0,
}

# ==============================================================================
# DIRECTIONAL TREND - Thresholds
# ==============================================================================

TREND_RISING_THRESHOLD = 70  # role_trajectory > 70 = rising
TREND_FALLING_THRESHOLD = 40  # role_trajectory < 40 = falling

SCORE_CHANGE_RISING = 10  # Offseason: +10 score change = rising
SCORE_CHANGE_FALLING = -10  # Offseason: -10 score change = falling

# ==============================================================================
# ROLE CLASSIFIER - Thresholds
# ==============================================================================

# WR/TE role tiers (based on projected targets)
WR1_TARGETS = 120
WR1_SNAP_SHARE = 0.75
WR2_TARGETS = 80
WR3_TARGETS = 50

# RB role tiers (based on projected carries)
RB_BELLCOW_CARRIES = 200
RB_BELLCOW_SNAP_SHARE = 0.70
RB1_CARRIES = 150
RB2_CARRIES = 100

# Role modifiers thresholds
RED_ZONE_USAGE_THRESHOLD = 0.20  # 20%+ RZ usage = "Red Zone" modifier
THREE_DOWN_SNAP_THRESHOLD = 0.85  # 85%+ snaps = "3-Down" modifier
PASSING_DOWN_TARGETS = 60  # 60+ targets for RB = "Passing Down" modifier
WORKHORSE_SNAP_THRESHOLD = 0.80  # 80%+ snaps = "Workhorse" for RB

# QB role tiers
QB_LOCKED_STARTER_SNAP = 0.95  # 95%+ snaps = "Locked Starter"
QB_STARTER_SNAP = 0.70  # 70%+ snaps = "QB1"

# ==============================================================================
# EXPLAINABILITY - Configuration
# ==============================================================================

# Maximum number of reasons to show in key_reasons
MAX_KEY_REASONS = 5

# Component score thresholds to include in explanation
# Lowered thresholds to provide more detailed explanations for most candidates
EXPLAIN_OPPORTUNITY_OPENED_THRESHOLD = 30  # Lowered from 40 - captures moderate opportunities
EXPLAIN_COMPETITION_REMOVED_THRESHOLD = 20  # Lowered from 30 - any meaningful departure
EXPLAIN_PLAYER_READINESS_THRESHOLD = 45  # Lowered from 50 - captures most viable candidates
EXPLAIN_TEAM_ENVIRONMENT_THRESHOLD = 55  # Slightly raised from 50 - only noteworthy teams
EXPLAIN_ROLE_TRAJECTORY_THRESHOLD = 55  # Unchanged
EXPLAIN_COMPETITION_PENALTY_THRESHOLD = -10  # Unchanged - any significant addition

# Enhanced readiness explanation thresholds
EXPLAIN_READINESS_EFFICIENCY_ELITE = 30  # Show "Elite efficiency metrics" message
EXPLAIN_READINESS_EFFICIENCY_STRONG = 20  # Show "Strong efficiency metrics" message
EXPLAIN_READINESS_USAGE_BASELINE = 15  # Show "Established backup opportunity" message
EXPLAIN_READINESS_DRAFT_CAPITAL = 25  # Minimum draft score to mention draft capital
EXPLAIN_READINESS_DRAFT_ROUND_MAX = 3  # Only mention draft capital for rounds 1-3

# Depth Chart Filtering
DEPTH_CHART_BLOCKING_PENALTY = 0.3  # Multiplier for candidates blocked by established starter
DEPTH_CHART_AGE_WINDOW = 2  # Years of age difference to consider "similar age"
DEPTH_CHART_EXCEPTION_OPP_THRESHOLD = 75  # Don't penalize if opportunity score this high

# Top-12 positional thresholds (established starter level)
TOP_12_PPG_THRESHOLDS = {
    'QB': 16.0,   # ~QB12 level in PPR
    'RB': 10.0,   # ~RB12 level in PPR
    'WR': 11.0,   # ~WR12 level in PPR
    'TE': 8.0     # ~TE12 level in PPR
}

# ==============================================================================
# WR FALSE-POSITIVE REDUCTION - Thresholds
# ==============================================================================

# Contested-catch profile penalty: low catch rate + low yards per target
# penalizes highly-drafted WRs who show poor separation proxies
WR_FP_CATCH_RATE_THRESHOLD = 0.52   # Below this is a concern
WR_FP_YPT_THRESHOLD = 7.0           # Below this compounds the concern
WR_FP_MIN_TARGETS = 30              # Only apply to players with meaningful sample

# Penalty applied to readiness score for contested-catch profiles
WR_FP_PENALTY_R1 = -12   # Round 1 draft capital + poor efficiency = overrated
WR_FP_PENALTY_R2 = -7    # Round 2
WR_FP_PENALTY_OTHER = -3  # Other rounds

# Day-2/3 breakout lift: skill-over-draft bonus for efficient players with modest draft capital
WR_SKILL_LIFT_YPT_THRESHOLD = 9.0    # Elite yards per target
WR_SKILL_LIFT_CATCH_THRESHOLD = 0.65  # Elite catch rate
WR_SKILL_LIFT_MIN_TARGETS = 40        # Must have meaningful sample
WR_SKILL_LIFT_R1 = 0                  # Round 1 picks already get full credit
WR_SKILL_LIFT_R2 = 6                  # Round 2 outperforming draft slot
WR_SKILL_LIFT_R3_R4 = 12             # Round 3-4 outperforming draft slot (Kupp archetype)
WR_SKILL_LIFT_UDFA = 8               # UDFAs with elite efficiency

RB_SKILL_LIFT_YPC_THRESHOLD = 4.8   # Elite yards per carry
RB_SKILL_LIFT_MIN_CARRIES = 60       # Must have meaningful sample
RB_SKILL_LIFT_R2 = 5
RB_SKILL_LIFT_R3_R4 = 10

# ==============================================================================
# TE STABILIZATION - Thresholds
# ==============================================================================

# Higher minimum confidence for TE (smaller sample, higher variance position)
TE_SAMPLE_MIN_CONFIDENCE = 0.50   # vs 0.35 default — more shrinkage toward mean
TE_USAGE_MIN_CONFIDENCE = 0.50

# TE-specific efficiency thresholds (routes run weight > raw totals)
TE_ELITE_YPT = 8.5
TE_GOOD_YPT = 7.0
TE_ELITE_CATCH_RATE = 0.68
TE_GOOD_CATCH_RATE = 0.60

# ==============================================================================
# POSITION-SPECIFIC CONFIGURATIONS
# ==============================================================================

POSITIONS = ['QB', 'RB', 'WR', 'TE']

PASS_CATCHING_POSITIONS = ['WR', 'TE', 'QB']
RUSHING_POSITIONS = ['RB']

# Position groupings for team environment scoring
BENEFITS_FROM_HIGH_PASS_RATE = ['WR', 'TE', 'QB']
BENEFITS_FROM_BALANCED_OFFENSE = ['RB']

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================

# Table names
BREAKOUT_SCORES_TABLE = 'breakout_opportunity_scores'
ROSTER_CHANGES_TABLE = 'roster_changes'
VACATED_OPPORTUNITY_TABLE = 'vacated_opportunity'
PLAYER_ADVANCED_METRICS_TABLE = 'player_advanced_metrics'
PLAYER_VALUES_TABLE = 'player_values'

# ==============================================================================
# CHANGE TYPE CATEGORIES
# ==============================================================================

DEPARTURE_TYPES = ['free_agent', 'trade', 'cut', 'retirement']
ARRIVAL_TYPES = ['free_agent', 'trade', 'draft']

# Transaction type that indicates drafted player
DRAFT_CHANGE_TYPE = 'draft'
