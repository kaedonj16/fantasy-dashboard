"""Random future-injury rates shared by the season-sim engines.

Single source of truth for the injury model: the playoff-odds season
simulator (``simulate_playoff_odds.py``) and the Lineup Lab payload builder
(``lineup_lab.py``) both draw from this table. This module is dependency-free
on purpose (no numpy) so the lint CI shard can import it.

Model: each simulated week, every starter may suffer a NEW injury (current-week
injury designations are already priced into the Sleeper projections both
engines draw from, so this models only injuries that happen later). Onset and
duration are position-based, calibrated from published NFL injury research
rather than invented:

  - Weekly missed-game rates: games missed per 17-game season by position
    (QB 2.06, RB 2.57, WR 2.28, TE 2.73) from a snap-count + roster-paperwork
    study of fantasy-relevant players, which found tight ends (not RBs) miss
    the most games and RB/WR/TE all within ~0.5 games of each other.
    Source: zinkelburger/fantasy-football-tool, "Running back is not the
    injury position" (site/posts/37-injury-by-position-age-week.md).
    https://github.com/zinkelburger/fantasy-football-tool/blob/HEAD/site/posts/37-injury-by-position-age-week.md
  - Cross-check: top-20 fantasy prospects play 83.6-85.0% of possible games
    (~15% of games missed across positions, RBs no worse than WR/TE/QB).
    Source: Fantasy Index, "Risky business - Are running backs more likely
    to get hurt?" https://fantasyindex.com/2026/08/24/factoid/risky-business
  - QB hazard is the lowest of the skill positions (injury rate per 1000
    athlete-exposures: QB 8.6 vs RB 20.7, WR 17.1, TE 16.9).
    Source: Hogshaven, "Understanding Injuries in the NFL Part 3"
    https://www.hogshaven.com/2019/6/22/18658887/understanding-injuries-in-the-nfl-part-3
  - Duration buckets: "once a player actually sits, the typical absence is
    two weeks, and a quarter never play again that season."
    Source: zinkelburger/fantasy-football-tool, "What a Friday injury tag is
    actually worth" (site/posts/21-injury-model.md).
    https://github.com/zinkelburger/fantasy-football-tool/blob/HEAD/site/posts/21-injury-model.md

The hazard below is the per-week probability that a given starter MISSES the
week's game (games_missed / 17). Multi-week absences are clustered into
realistic spells: the per-week ONSET rate is hazard / mean duration, and an
onset samples a duration from the buckets. Expected games missed per season
stays ~ hazard * 17 (e.g. RB 0.151 * 17 = 2.57, matching the study).
"""

# Per-week probability a given starter misses that week's game, by position.
_INJURY_HAZARD: dict = {
    "QB": 0.121, "RB": 0.151, "WR": 0.134, "TE": 0.161, "K": 0.01, "DEF": 0.0,
}
_INJURY_HAZARD_DEFAULT = 0.13

# When a starter goes down, the fantasy team starts someone else — never a
# zero. The loss is (starter_ppg - replacement_ppg) where the replacement is
# the best eligible bench player. If the roster has no eligible bench player,
# a waiver-wire pickup plays at this fraction of the starter's projection.
_INJURY_REPLACEMENT = 0.45

# Injury duration buckets (weeks missed): 1 / 2-3 / 4+ weeks, plus a long
# season-ending tail. Mean = 2.66 weeks, matching "the typical absence is two
# weeks, and a quarter never play again that season" (the 5- and 8-week
# buckets sum to 23%).
_INJURY_DURATION_CHOICES = (1, 2, 3, 5, 8)
_INJURY_DURATION_PROBS = (0.45, 0.20, 0.12, 0.13, 0.10)
_INJURY_MEAN_DURATION = sum(
    c * p for c, p in zip(_INJURY_DURATION_CHOICES, _INJURY_DURATION_PROBS)
)


def injury_onset_rate(pos: str) -> float:
    """Per-week probability a healthy starter at `pos` suffers a new injury.

    Hazard / mean duration, so multi-week absences cluster into realistic
    spells while the expected games missed stays ~ hazard.
    """
    haz = _INJURY_HAZARD.get(str(pos or "").upper(), _INJURY_HAZARD_DEFAULT)
    return haz / _INJURY_MEAN_DURATION if _INJURY_MEAN_DURATION else haz


def expected_injury_loss_per_week(mean_ppg: float, pos: str) -> float:
    """Expected weekly points lost to future injuries for one starter.

    onset * (starter_mean - replacement_mean), with the waiver-wire fallback
    as the replacement level. Used to haircut team-level opponent means in
    engines that simulate injuries on one side only (Lineup Lab).
    """
    mean = max(float(mean_ppg), 0.0)
    return injury_onset_rate(pos) * mean * (1.0 - _INJURY_REPLACEMENT)


def override_injury_rates(
    hazard: dict | None = None,
    duration_choices: tuple | None = None,
    duration_probs: tuple | None = None,
    replacement: float | None = None,
    default: float | None = None,
) -> None:
    """Override the injury-simulation constants (tests, experiments).

    The whole injury model is driven by the single table above, so one call
    retunes or disables it. Pass ``hazard={}`` to turn injuries off entirely
    (the per-position default is zeroed too). There is no thread-safety
    contract here — call it at startup or from a test, not mid-sim.
    """
    global _INJURY_DURATION_CHOICES, _INJURY_DURATION_PROBS
    global _INJURY_MEAN_DURATION, _INJURY_REPLACEMENT, _INJURY_HAZARD_DEFAULT
    if hazard is not None:
        _INJURY_HAZARD.clear()
        _INJURY_HAZARD.update({str(k).upper(): float(v) for k, v in hazard.items()})
        # An empty map is an explicit opt-out: no position should fall back
        # to a nonzero default.
        _INJURY_HAZARD_DEFAULT = (
            float(default) if default is not None
            else (0.0 if not _INJURY_HAZARD else _INJURY_HAZARD_DEFAULT)
        )
    elif default is not None:
        _INJURY_HAZARD_DEFAULT = float(default)
    if duration_choices is not None:
        _INJURY_DURATION_CHOICES = tuple(duration_choices)
    if duration_probs is not None:
        _INJURY_DURATION_PROBS = tuple(duration_probs)
    _INJURY_MEAN_DURATION = sum(
        c * p for c, p in zip(_INJURY_DURATION_CHOICES, _INJURY_DURATION_PROBS)
    )
    if replacement is not None:
        _INJURY_REPLACEMENT = float(replacement)
