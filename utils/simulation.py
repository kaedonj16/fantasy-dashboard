"""Reusable deterministic random draws for comparable Monte Carlo scenarios."""
from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class ScenarioBank:
    independent: tuple
    game_environment: tuple

    @classmethod
    def create(cls, simulations: int, weeks: int, teams: int, seed: int = 0):
        rng = random.Random(seed)
        shape = lambda: tuple(tuple(tuple(rng.gauss(0, 1) for _ in range(teams))
                                   for _ in range(weeks)) for _ in range(simulations))
        return cls(shape(), shape())

    def weekly_draws(self, environment_loading: float = 0.22) -> tuple:
        """Hierarchical team/game-environment draws, variance-normalized."""
        loading = min(max(float(environment_loading), 0.0), 0.95)
        independent_weight = (1 - loading ** 2) ** 0.5
        return tuple(tuple(tuple(independent_weight * x + loading * y
                                 for x, y in zip(ix, ex))
                           for ix, ex in zip(isim, esim))
                     for isim, esim in zip(self.independent, self.game_environment))
