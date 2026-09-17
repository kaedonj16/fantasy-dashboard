# Opponent-adjusted Schedule Rankings

## Pipeline and migration

The former pipeline summed league-scored player points by opponent and divided
by games. Weekly ranks, schedule averages, and Ease then followed partly
separate raw-FPA/rank paths. The authoritative pipeline is now
`utils.defensive_matchup_ratings.py`; `data_building.matchup_ratings` rebuilds
its atomic JSON snapshot, the Wednesday daily-cron step invokes that builder,
and both Schedule Assistant API views consume the same adjusted multiplier.
Raw FPA remains in every rating for diagnostics and tooltips.

Snapshots are keyed by season and the stable full scoring-profile hash. The
legacy un-suffixed file is the standard PPR build only. Custom profiles never
fall back to another league's file; an unavailable profile produces `N/A`.
Rebuilding replaces the snapshot atomically, so corrected source statistics are
safe to replay. Only regular-season, completed source rows through the requested
week are considered.

## Formulae

For each meaningfully active player, the pregame baseline is a configurable
blend of the most recent four qualifying games (50%), earlier current-season
games (30%), and previous-season/position-role replacement (20%). Previous data
fades over eight current games. Missing history uses a non-zero positional
median and lowers reliability. A row is appended to history only **after** its
expectation is computed, preventing same-game and future leakage.

Participation is an OR test: QB 8 attempts or 15 snaps; RB 3 touches or 8
snaps; WR 2 targets, 5 routes, or 8 snaps; TE 1 target, 5 routes, or 8 snaps.
When opportunity is absent, positive scored points are the conservative
fallback. RB/WR/TE player expectations and actuals are summed into game units.

For a game unit:

* `points over expected = actual - expected`
* `multiplier = actual / expected`
* `effect % = (multiplier - 1) × 100`

Game multipliers are winsorized to 0.50–1.50 for aggregation (original actual
and expected totals remain visible), weighted by expected volume, baseline
reliability, and a modest 0.90–1.00 recency factor, then shrunk toward 1.00.
The neutral prior has four equivalent weight units scaled by the continuous
week 0–6 prior schedule. Confidence is low below four games, medium from four,
and high at eight games with at least 80 reliable weight units.

The product uses current evaluation: the latest stabilized defense multiplier
is applied to every available opponent in the selected range. Byes and missing
ratings are omitted. `ADJ AVG` is the arithmetic mean of full-precision
multipliers, displayed as `(mean - 1) × 100`. Ease rank sorts that same mean
descending. Ease score is min-max normalization across comparable schedules:
`100 × (value - minimum) / (maximum - minimum)`; tied schedules receive 50.

## Backtest and support status

The rollout retains raw and adjusted fields side by side so a chronological
backtest can compare raw FPA, average-opponent rank, unshrunk multiplier, and
shrunk multiplier against next-game production. No representative historical
cache/data corpus is checked into this repository, so **no honest predictive
winner or correlation is claimed by this change**; production validation must
be run after historical profile snapshots are backfilled.

QB/RB/WR/TE and custom passing, rushing, receiving, reception/TE-premium, and
turnover categories supported by the shared fantasy scorer are handled. The
nflverse weekly source does not provide reliable K/DST fantasy production, so
K/DST adjusted ratings remain unsupported and display `N/A`. Garbage-time and
in-game injury labels are unavailable; opportunity thresholds, unit totals,
winsorization, reliability weighting, and shrinkage provide conservative
protection rather than pretending those events can be classified perfectly.
