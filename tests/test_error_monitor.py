"""Tests for utils.error_monitor (warning/error counting)."""
import logging

import pytest

from utils import error_monitor
from utils.error_monitor import ErrorCounterHandler


@pytest.fixture(autouse=True)
def _clean_counts():
    error_monitor.reset()
    yield
    error_monitor.reset()


@pytest.fixture()
def counted_logger():
    lg = logging.getLogger("test.error_monitor")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    handler = ErrorCounterHandler(level=logging.DEBUG)
    lg.addHandler(handler)
    yield lg
    lg.removeHandler(handler)


def test_warning_is_counted(counted_logger):
    counted_logger.warning("something failed: %s", "detail")
    snap = error_monitor.snapshot()
    assert snap["distinct_errors"] == 1
    assert snap["errors"][0]["count"] == 1
    assert snap["errors"][0]["level"] == "WARNING"


def test_info_without_exc_not_counted(counted_logger):
    counted_logger.info("routine message")
    assert error_monitor.snapshot()["distinct_errors"] == 0


def test_debug_with_exc_info_counted(counted_logger):
    try:
        raise ValueError("boom")
    except ValueError:
        counted_logger.debug("suppressed exception", exc_info=True)
    assert error_monitor.snapshot()["distinct_errors"] == 1


def test_same_template_groups_into_one_bucket(counted_logger):
    counted_logger.warning("failed for league %s", "111")
    counted_logger.warning("failed for league %s", "222")
    snap = error_monitor.snapshot()
    assert snap["distinct_errors"] == 1
    assert snap["errors"][0]["count"] == 2


def test_most_frequent_first(counted_logger):
    counted_logger.warning("rare failure")
    for _ in range(3):
        counted_logger.error("common failure")
    snap = error_monitor.snapshot()
    assert snap["errors"][0]["sample"] == "common failure"
    assert snap["errors"][0]["count"] == 3


def test_reset_clears(counted_logger):
    counted_logger.warning("x")
    error_monitor.reset()
    assert error_monitor.snapshot()["distinct_errors"] == 0


def test_key_cap_prevents_unbounded_growth(counted_logger):
    for i in range(error_monitor._MAX_KEYS + 50):
        # Unique msg template each time to force new buckets.
        counted_logger.warning(f"unique failure {i}")
    assert error_monitor.snapshot(limit=10_000)["distinct_errors"] <= error_monitor._MAX_KEYS


def test_install_idempotent():
    root = logging.getLogger()
    before = len(root.handlers)
    error_monitor.install()
    after_first = len(root.handlers)
    error_monitor.install()
    assert len(root.handlers) == after_first
    assert after_first <= before + 1
