"""Failure-path tests for dashboard_services.news: news failures must be loud.

- Every source failure is logged at warning level with context (source,
  HTTP status or exception) instead of vanishing into an empty list.
- get_nfl_news / get_player_news return {"news", "sources_failed"} so the
  routes and frontend can render a visible warning instead of a misleading
  empty feed ("No news available" when ESPN/Reddit/Google News are down).

httpx is not installed in this environment, so the tests stub sys.modules
httpx (mirroring news.py's lazy import) and stub the low-level fetchers.
"""
import logging
import sys
import types

import pytest

import dashboard_services.news as news


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    monkeypatch.setattr(news, "_CACHE", {})


class _FakeResp:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self._json = json_body if json_body is not None else {}
        self.text = text

    def json(self):
        return self._json


def _install_fake_httpx(monkeypatch, response=None, exc=None):
    """Satisfy news.py's lazy `import httpx` without the real dependency."""
    mod = types.ModuleType("httpx")

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def get(self, *args, **kwargs):
            if exc is not None:
                raise exc
            return response

    mod.AsyncClient = _Client
    monkeypatch.setitem(sys.modules, "httpx", mod)


def _stub_pair(items, err):
    async def _fn(*args, **kwargs):
        return items, err
    return _fn


def _stub_triple(eid, items, err):
    async def _fn(*args, **kwargs):
        return eid, items, err
    return _fn


def _item(headline, url="https://example.com/a"):
    return {"headline": headline, "description": "", "published": "2026-09-25T12:00:00Z",
            "age": "", "url": url, "source": "ESPN"}


# ── general feed aggregation ────────────────────────────────────────────────

def test_general_all_sources_fail_named_in_sources_failed(monkeypatch):
    _install_fake_httpx(monkeypatch)
    monkeypatch.setattr(news, "_async_fetch_general", _stub_pair([], news.SRC_ESPN))
    monkeypatch.setattr(news, "_async_fetch_gnews_general", _stub_pair([], news.SRC_GNEWS))
    monkeypatch.setattr(news, "_async_fetch_reddit_hot", _stub_pair([], news.SRC_REDDIT))

    result = news.get_nfl_news(limit=5)

    assert set(result) == {"news", "sources_failed"}
    assert result["news"] == []
    assert set(result["sources_failed"]) == {"espn", "gnews", "reddit"}


def test_general_partial_failure_serves_remaining_sources(monkeypatch):
    _install_fake_httpx(monkeypatch)
    gnews_items = [_item("Puka Nacua questionable, limited practice", "https://example.com/g1")]
    reddit_items = [_item("Christian McCaffrey returns from calf injury", "https://example.com/r1")]
    monkeypatch.setattr(news, "_async_fetch_general", _stub_pair([], news.SRC_ESPN))
    monkeypatch.setattr(news, "_async_fetch_gnews_general", _stub_pair(gnews_items, None))
    monkeypatch.setattr(news, "_async_fetch_reddit_hot", _stub_pair(reddit_items, None))

    result = news.get_nfl_news(limit=5)

    assert len(result["news"]) == 2
    assert result["sources_failed"] == ["espn"]


def test_general_happy_path_sources_failed_empty(monkeypatch):
    _install_fake_httpx(monkeypatch)
    espn_items = [_item("Christian McCaffrey returns from calf injury", "https://example.com/e1")]
    monkeypatch.setattr(news, "_async_fetch_general", _stub_pair(espn_items, None))
    monkeypatch.setattr(news, "_async_fetch_gnews_general", _stub_pair([], None))
    monkeypatch.setattr(news, "_async_fetch_reddit_hot", _stub_pair([], None))

    result = news.get_nfl_news(limit=5)

    assert result["news"] == espn_items
    assert result["sources_failed"] == []


# ── player feed aggregation ─────────────────────────────────────────────────

def test_player_news_athlete_feed_failure_reported(monkeypatch):
    _install_fake_httpx(monkeypatch)
    monkeypatch.setattr(news, "_async_fetch_athlete", _stub_triple("1", [], news.SRC_ESPN))
    monkeypatch.setattr(news, "_async_fetch_general", _stub_pair([], news.SRC_ESPN))
    monkeypatch.setattr(news, "_async_fetch_gnews", _stub_pair([], None))
    monkeypatch.setattr(news, "_async_fetch_reddit", _stub_pair([], None))

    result = news.get_player_news(
        "Some Player",
        espn_headshot="https://a.espncdn.com/i/headshots/nfl/players/full/1.png",
        limit=4,
    )

    assert set(result) == {"news", "sources_failed"}
    assert result["news"] == []
    # named once, not duplicated by the name-match fallback failing too
    assert result["sources_failed"] == ["espn"]


def test_player_news_partial_failure_serves_remaining_sources(monkeypatch):
    _install_fake_httpx(monkeypatch)
    reddit_items = [_item("Some Player breakout season continues", "https://example.com/r1")]
    monkeypatch.setattr(news, "_async_fetch_athlete", _stub_triple("1", [], news.SRC_ESPN))
    monkeypatch.setattr(news, "_async_fetch_general", _stub_pair([], news.SRC_ESPN))
    monkeypatch.setattr(news, "_async_fetch_gnews", _stub_pair([], None))
    monkeypatch.setattr(news, "_async_fetch_reddit", _stub_pair(reddit_items, None))

    result = news.get_player_news("Some Player", espn_headshot="", limit=4)

    assert result["news"] == reddit_items
    assert result["sources_failed"] == ["espn"]


# ── warning logs on the real fetchers ───────────────────────────────────────

def test_fetch_general_http_error_warns_and_reports_espn(monkeypatch, caplog):
    _install_fake_httpx(monkeypatch, response=_FakeResp(status_code=403))

    with caplog.at_level(logging.WARNING, logger="dashboard_services.news"):
        items, err = news._run(news._async_fetch_general())

    assert items == []
    assert err == news.SRC_ESPN
    assert any(r.levelno == logging.WARNING and "403" in r.getMessage()
               and "ESPN" in r.getMessage() for r in caplog.records)


def test_fetch_reddit_hot_exception_warns_and_reports_reddit(caplog):
    class _RaisingClient:
        async def get(self, *args, **kwargs):
            raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING, logger="dashboard_services.news"):
        items, err = news._run(news._async_fetch_reddit_hot(_RaisingClient()))

    assert items == []
    assert err == news.SRC_REDDIT
    assert any(r.levelno == logging.WARNING and "RuntimeError" in r.getMessage()
               and "Reddit" in r.getMessage() for r in caplog.records)


def test_fetch_gnews_general_http_error_warns_and_reports_gnews(monkeypatch, caplog):
    _install_fake_httpx(monkeypatch, response=_FakeResp(status_code=500))

    with caplog.at_level(logging.WARNING, logger="dashboard_services.news"):
        items, err = news._run(news._async_fetch_gnews_general())

    assert items == []
    assert err == news.SRC_GNEWS
    assert any(r.levelno == logging.WARNING and "500" in r.getMessage()
               and "Google News" in r.getMessage() for r in caplog.records)
