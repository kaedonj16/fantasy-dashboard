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


def test_valid_response_auth_and_pagination():
    session = Session([Response(payload={"data": [{"id": "1"}], "nextCursor": "c"}),
                       Response(payload={"data": [{"id": "2"}]})])
    rows = list(SportsGameOddsClient("secret", session).iter_nfl_events(starts_after="a", starts_before="b"))
    assert [x["id"] for x in rows] == ["1", "2"]
    assert session.calls[0][1]["headers"] == {"x-api-key": "secret"}
    assert session.calls[1][1]["params"]["cursor"] == "c"


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


def test_empty_events_and_invalid_shape():
    assert list(SportsGameOddsClient("key", Session([Response(payload={"data": []})])).iter_nfl_events(starts_after="a", starts_before="b")) == []
    with pytest.raises(SportsGameOddsError, match="event data"):
        list(SportsGameOddsClient("key", Session([Response(payload={"data": {}})])).iter_nfl_events(starts_after="a", starts_before="b"))
