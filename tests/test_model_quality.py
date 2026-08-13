from utils.evaluation_metrics import brier_score, decision_regret, log_loss, precision_at_k
from utils.model_confidence import confidence_from_inputs, rank_interval


def test_confidence_tracks_completeness_and_rank_range():
    high = confidence_from_inputs(8, 8, 300)
    low = confidence_from_inputs(2, 8, 5)
    assert high["score"] > low["score"]
    assert rank_interval(5, high["score"], 50)[1] - rank_interval(5, high["score"], 50)[0] < \
           rank_interval(5, low["score"], 50)[1] - rank_interval(5, low["score"], 50)[0]


def test_shared_decision_metrics():
    assert brier_score([0.9, 0.1], [1, 0]) < brier_score([0.6, 0.4], [1, 0])
    assert log_loss([0.9, 0.1], [1, 0]) < log_loss([0.6, 0.4], [1, 0])
    assert precision_at_k([.9, .8, .1], [1, 0, 0], 1) == 1.0
    assert decision_regret(75, 100) == 25


def test_rank_interval_is_bounded_at_top_and_bottom():
    assert rank_interval(1, 10, 20)[0] == 1
    assert rank_interval(20, 10, 20)[1] == 20
