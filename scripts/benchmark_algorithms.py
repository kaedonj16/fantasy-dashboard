"""Representative before/after microbenchmarks for algorithm roadmap work."""
from __future__ import annotations
import random
import time
from utils.all_play import all_play_analysis


def legacy_all_play(scores):
    wins = 0.0
    for week in scores.values():
        rows = list(week.values())
        for a in rows:
            for b in rows:
                wins += a > b
    return wins


def timed(fn, value, repeats=30):
    start = time.perf_counter()
    for _ in range(repeats):
        fn(value)
    return (time.perf_counter() - start) * 1000 / repeats


if __name__ == "__main__":
    rng = random.Random(20260814)
    weekly = {w: {str(t): round(rng.gauss(110, 25), 2) for t in range(100)}
              for w in range(1, 19)}
    before = timed(legacy_all_play, weekly)
    after = timed(lambda x: all_play_analysis(x, {}), weekly)
    print(f"all-play 18x100: pairwise={before:.3f}ms grouped-sort={after:.3f}ms "
          f"speedup={before / after:.2f}x")
