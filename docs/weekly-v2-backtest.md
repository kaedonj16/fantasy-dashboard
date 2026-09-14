# Weekly-v2 breakout backtest

## Reproducible run

```bash
python -m data_building.breakout_engine.backtest_weekly_breakout \
  --start-season 2023 --end-season 2025 --eval-from 4 --eval-to 15 \
  --horizon 3 --source cache --output /tmp/weekly-v2-backtest.json
```

The cache-backed run evaluated 17,294 player-week observations. It uses the
committed Sleeper weekly stat caches and has no route or red-zone-opportunity
coverage; those optional signals therefore remain `None` and are not evaluated.

## Observed calibration

| Score | N | role persisted (next 3) | fantasy hit | future PPR PPG |
|---|---:|---:|---:|---:|
| 0-49 | 13,387 | 6.3% | 21.4% | 6.47 |
| 50-59 | 623 | 18.2% | 33.0% | 8.77 |
| 60-69 | 512 | 19.8% | 33.9% | 9.10 |
| 70-79 | 539 | 19.1% | 30.8% | 8.43 |
| 80-89 | 552 | 21.4% | 25.3% | 7.51 |
| 90+ | 1,394 | 34.5% | 34.2% | 9.56 |

The high and low bands separate, but the middle buckets are not monotonic.
Consequently this run does **not** support changing weekly-v2 weights or
thresholds. In particular, route and high-value-touch weights cannot be
calibrated from a dataset where those inputs are absent.

The existing top-15 comparison for 2025 produced 42.1% model role-persistence
precision versus 45.8% for raw usage growth and 10.0% for recent fantasy points.
Fantasy-usefulness precision was 42.1%, versus 32.4% for raw usage growth and
70.6% for recent points. The latter is expected to win a near-term fantasy-point
outcome but performed poorly at detecting new persistent roles. These metrics
should be treated as measurement, not proof of calibrated probability.
