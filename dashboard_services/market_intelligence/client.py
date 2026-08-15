from __future__ import annotations

import os
from typing import Iterator

class SportsGameOddsError(RuntimeError):
    pass


class SportsGameOddsClient:
    """Small client for the documented v2 events feed.

    SportsGameOdds returns player props in each event's ``odds`` mapping. The
    cursor is returned at the response root and is sent back as ``cursor``.
    """

    base_url = "https://api.sportsgameodds.com/v2"

    def __init__(self, api_key: str | None = None, session=None, timeout: float = 15.0):
        self.api_key = (api_key if api_key is not None else os.getenv("SPORTSGAMEODDS_API_KEY", "")).strip()
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict) -> dict:
        if not self.configured:
            return {"data": []}
        try:
            response = self.session.get(
                f"{self.base_url}/{path.lstrip('/')}", params=params,
                headers={"x-api-key": self.api_key}, timeout=self.timeout,
            )
        except Exception as exc:
            # requests and test transports expose different timeout base types.
            if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
                raise SportsGameOddsError("SportsGameOdds request timed out") from exc
            raise
        if response.status_code in (401, 403):
            raise SportsGameOddsError("SportsGameOdds authentication failed")
        if response.status_code == 429:
            raise SportsGameOddsError("SportsGameOdds rate limit reached")
        if response.status_code >= 400:
            raise SportsGameOddsError(f"SportsGameOdds request failed ({response.status_code})")
        try:
            payload = response.json()
        except ValueError as exc:
            raise SportsGameOddsError("SportsGameOdds returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise SportsGameOddsError("SportsGameOdds returned an invalid response")
        return payload

    def iter_nfl_events(self, *, starts_after: str, starts_before: str) -> Iterator[dict]:
        params = {"leagueID": "NFL", "startsAfter": starts_after,
                  "startsBefore": starts_before, "oddsAvailable": "true", "limit": 100}
        while True:
            payload = self._get("events", params)
            rows = payload.get("data", [])
            if rows is None:
                rows = []
            if not isinstance(rows, list):
                raise SportsGameOddsError("SportsGameOdds event data is invalid")
            yield from (row for row in rows if isinstance(row, dict))
            cursor = payload.get("nextCursor") or (payload.get("meta") or {}).get("nextCursor")
            if not cursor:
                break
            params["cursor"] = cursor
