import os
import time

import pytest

from dashboard_services.market_intelligence.client import SportsGameOddsClient, SportsGameOddsError


class Response:
    def __init__(self, status=200, payload=None, malformed=False):
        self.status_code, self.payload, self.malformed = status, payload, malformed

    def json(self):
        if self.malformed:
            raise ValueError
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_missing_key_is_empty_without_http():
    session = Session([])
    assert list(SportsGameOddsClient("", session).iter_nfl_events(starts_after="a", starts_before="b")) == []
    assert session.calls == []


def test_valid_response_auth_and_pagination(tmp_path):
    session = Session([Response(payload={"data": [{"id": "1"}], "nextCursor": "c"}),
                       Response(payload={"data": [{"id": "2"}]})])
    rows = list(SportsGameOddsClient("secret", session, cache_dir=tmp_path).iter_nfl_events(starts_after="a", starts_before="b"))
    assert [x["id"] for x in rows] == ["1", "2"]
    assert session.calls[0][1]["headers"] == {"x-api-key": "secret"}
    assert session.calls[0][1]["params"] == {
        "leagueID": "NFL", "startsAfter": "a", "startsBefore": "b",
        "oddsAvailable": "true", "limit": 100,
    }
    assert session.calls[1][1]["params"]["cursor"] == "c"


def test_meta_cursor_pagination_and_repeated_cursor_guard(tmp_path):
    session = Session([Response(payload={"data": [{"id": "1"}], "meta": {"nextCursor": "c"}}),
                       Response(payload={"data": [{"id": "2"}], "meta": {"nextCursor": "c"}})])
    client = SportsGameOddsClient("secret", session, cache_dir=tmp_path)
    with pytest.raises(SportsGameOddsError, match="repeated pagination cursor"):
        list(client.iter_nfl_events(starts_after="a", starts_before="b"))
    assert session.calls[1][1]["params"]["cursor"] == "c"


def test_successful_api_page_is_cached_for_one_hour(tmp_path):
    first = Session([Response(payload={"data": [{"id": "1"}]})])
    assert list(SportsGameOddsClient("secret", first, cache_dir=tmp_path).iter_nfl_events(
        starts_after="a", starts_before="b")) == [{"id": "1"}]

    second = Session([])
    assert list(SportsGameOddsClient("secret", second, cache_dir=tmp_path).iter_nfl_events(
        starts_after="a", starts_before="b")) == [{"id": "1"}]
    assert second.calls == []

    for cache_file in tmp_path.glob("*.json"):
        old = time.time() - 3601
        os.utime(cache_file, (old, old))
    refreshed = Session([Response(payload={"data": [{"id": "2"}]})])
    assert list(SportsGameOddsClient("secret", refreshed, cache_dir=tmp_path).iter_nfl_events(
        starts_after="a", starts_before="b")) == [{"id": "2"}]
    assert len(refreshed.calls) == 1


@pytest.mark.parametrize("status,text", [(401, "authentication"), (429, "rate limit"), (500, "failed")])
def test_http_failures(status, text):
    client = SportsGameOddsClient("key", Session([Response(status=status, payload={})]))
    with pytest.raises(SportsGameOddsError, match=text):
        list(client.iter_nfl_events(starts_after="a", starts_before="b"))


def test_timeout_and_malformed_json():
    with pytest.raises(SportsGameOddsError, match="timed out"):
        list(SportsGameOddsClient("key", Session([TimeoutError()])).iter_nfl_events(starts_after="a", starts_before="b"))
    with pytest.raises(SportsGameOddsError, match="malformed"):
        list(SportsGameOddsClient("key", Session([Response(malformed=True)])).iter_nfl_events(starts_after="a", starts_before="b"))


def test_empty_events_and_invalid_shape(tmp_path):
    empty = SportsGameOddsClient(
        "key", Session([Response(payload={"data": []})]), cache_dir=tmp_path / "empty",
    )
    assert list(empty.iter_nfl_events(starts_after="a", starts_before="b")) == []
    with pytest.raises(SportsGameOddsError, match="event data"):
        invalid = SportsGameOddsClient(
            "key", Session([Response(payload={"data": {}})]), cache_dir=tmp_path / "invalid",
        )
        list(invalid.iter_nfl_events(starts_after="a", starts_before="b"))
