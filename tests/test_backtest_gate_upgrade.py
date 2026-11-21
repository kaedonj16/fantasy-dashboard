"""Gate upgrade (2026-11-20): Spearman beside Pearson, and the
no-capital slice.

Why: every historical gate number priced draft capital at the ACTUAL
pick, but the live board grades most players with no projected
capital at all (19 of 140 on the 2027 board). A model can clear the
headline bars on capital's back while ordering the no-capital regime
poorly, and no gate had ever isolated that regime. The harness now
re-grades each class with capital masked and prints the slice, and
reports Spearman (rank fidelity) alongside Pearson per position.
"""

import os
import sys

import pytest

pytest.importorskip("pandas")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import backtest_prospect_model as bt  # noqa: E402


class TestSpearman:
    def test_perfect_monotone_is_one(self):
        assert bt._spearman_r([1.0, 2.0, 3.0, 4.0],
                              [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)

    def test_reversed_is_minus_one(self):
        assert bt._spearman_r([1.0, 2.0, 3.0, 4.0],
                              [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)

    def test_rank_based_not_value_based(self):
        # Same ordering, wildly different spacing: Spearman stays 1
        # where Pearson would sag.
        assert bt._spearman_r([1.0, 2.0, 3.0, 100.0],
                              [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)

    def test_ties_use_average_ranks(self):
        rho = bt._spearman_r([1.0, 2.0, 2.0, 4.0],
                             [1.0, 2.0, 3.0, 4.0])
        assert 0.9 < rho < 1.0

    def test_too_few_points_is_nan(self):
        import math
        assert math.isnan(bt._spearman_r([1.0, 2.0], [1.0, 2.0]))


class TestMaskCapital:
    def test_strips_capital_fields(self):
        prospects = [{"player_id": "P1", "name": "A",
                      "draft_capital": {"score": 90},
                      "projected_pick": 12,
                      "projected_draft_capital_score": 88.0,
                      "seasons": [{"season": 2025}]}]
        masked = bt._mask_capital(prospects)
        assert "draft_capital" not in masked[0]
        assert "projected_pick" not in masked[0]
        assert "projected_draft_capital_score" not in masked[0]
        assert masked[0]["seasons"] == [{"season": 2025}]

    def test_originals_not_mutated(self):
        prospects = [{"player_id": "P1", "draft_capital": {"score": 90},
                      "seasons": [{"season": 2025, "targets": 50}]}]
        masked = bt._mask_capital(prospects)
        masked[0]["seasons"][0]["targets"] = 999
        assert prospects[0]["draft_capital"] == {"score": 90}
        assert prospects[0]["seasons"][0]["targets"] == 50


class TestNoCapitalSlicePrinter:
    def _rows(self, pos, scores, pprs):
        return [{"name": f"{pos}{i}", "position": pos, "draft_year": 2024,
                 "model_score": s, "model_rank": i + 1, "ppr_cum": p}
                for i, (s, p) in enumerate(zip(scores, pprs))]

    def test_prints_per_position_metrics(self, capsys):
        rows = (self._rows("WR", [90, 80, 70, 60, 50, 40],
                            [300, 250, 200, 150, 100, 50])
                + self._rows("QB", [90, 80, 70, 60, 50, 40],
                             [50, 100, 150, 200, 250, 300]))
        bt._print_no_capital_slice(rows)
        out = capsys.readouterr().out
        assert "NO-CAPITAL SLICE" in out
        assert "WR" in out and "QB" in out
        assert "+1.000" in out   # WR perfectly ordered
        assert "-1.000" in out   # QB perfectly reversed

    def test_small_positions_skipped(self, capsys):
        rows = self._rows("TE", [90, 80, 70], [300, 200, 100])
        bt._print_no_capital_slice(rows)
        out = capsys.readouterr().out
        assert "NO-CAPITAL SLICE" in out
        assert "TE" not in out.split("Spearman")[-1]
