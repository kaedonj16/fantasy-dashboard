"""Tests for utils.perf_monitor."""
from utils import perf_monitor


def setup_function(_):
    perf_monitor.reset()


def test_empty_snapshot():
    s = perf_monitor.snapshot()
    assert s["distinct_endpoints"] == 0
    assert s["endpoints"] == []


def test_record_and_aggregate():
    perf_monitor.record("page_x", "GET", 100.0, 200)
    perf_monitor.record("page_x", "GET", 300.0, 200)
    s = perf_monitor.snapshot()
    ep = s["endpoints"][0]
    assert ep["endpoint"] == "GET page_x"
    assert ep["count"] == 2
    assert ep["avg_ms"] == 200.0
    assert ep["max_ms"] == 300.0
    assert ep["total_ms"] == 400.0


def test_slow_and_error_counts():
    perf_monitor.record("api_slow", "GET", perf_monitor.SLOW_MS + 1, 200)
    perf_monitor.record("api_slow", "GET", 10.0, 500)
    ep = perf_monitor.snapshot()["endpoints"][0]
    assert ep["slow_count"] == 1
    assert ep["error_count"] == 1


def test_sort_by_total_puts_heaviest_first():
    perf_monitor.record("light", "GET", 50.0, 200)
    perf_monitor.record("heavy", "GET", 5000.0, 200)
    eps = perf_monitor.snapshot(sort="total")["endpoints"]
    assert eps[0]["endpoint"] == "GET heavy"


def test_reset():
    perf_monitor.record("page_x", "GET", 100.0, 200)
    perf_monitor.reset()
    assert perf_monitor.snapshot()["distinct_endpoints"] == 0
