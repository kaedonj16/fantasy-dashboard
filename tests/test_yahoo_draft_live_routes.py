"""Flask route tests for Yahoo live-draft detect/live/relay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

flask = pytest.importorskip("flask")

from dashboard_services.draft_sync import (
    DraftSyncAuthError,
    DraftSyncSnapshot,
    NormalizedDraftPick,
)
from routes.draft_api_bp import draft_api_bp


@dataclass
class _FakeProvider:
    source: str = "yahoo"
    snapshot: Optional[DraftSyncSnapshot] = None
    error: Optional[Exception] = None

    def get_snapshot(self, league_id, season, *, viewer_user_id=None, viewer_roster_id=None):
        if self.error:
            raise self.error
        snap = self.snapshot
        if viewer_roster_id:
            snap.viewer_team_id = str(viewer_roster_id)
        return snap


def _snap(status="drafting", picks=None):
    if picks is None:
        picks = [
            NormalizedDraftPick(
                source="yahoo", overall_pick=1, canonical_player_id="5938",
                external_player_id="30123", external_team_id="1",
                picked_by="guid-a", roster_id="1", name="Justin Jefferson",
                position="WR", team="MIN",
            )
        ]
    return DraftSyncSnapshot(
        source="yahoo", draft_id="yahoo_99_2026", league_id="99", season=2026,
        status=status, drafted=(status == "complete"),
        in_progress=(status == "drafting"), picks=picks, teams=4, rounds=15,
        order="snake", picks_observed=True, live_detail_present=True,
        poll_interval_ms=6000, viewer_team_id="1",
        user_roster_map={"guid-a": "1"}, draft_order={"guid-a": 1, "1": 1},
        slot_names={"1": "Alpha"},
    )


@pytest.fixture
def client(monkeypatch):
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(draft_api_bp)
    fake = _FakeProvider(snapshot=_snap())
    import dashboard_services.draft_sync as ds
    monkeypatch.setattr(ds, "get_draft_sync_provider", lambda platform: fake)
    with app.test_client() as test_client:
        with test_client.session_transaction() as sess:
            sess["viewer_user_id"] = "guid-a"
            sess["viewer_roster_id"] = "1"
        test_client._fake = fake  # type: ignore[attr-defined]
        yield test_client


def test_yahoo_live_returns_normalized_picks(client, monkeypatch):
    import dashboard_services.draft_sync as ds
    fake = client._fake
    monkeypatch.setattr(ds, "get_draft_sync_provider", lambda platform: fake)
    resp = client.get("/api/draft/live?platform=yahoo&draft_id=yahoo_99_2026")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["source"] == "yahoo"
    assert data["status"] == "drafting"
    assert data["picks"][0]["player_id"] == "5938"
    assert "access_token" not in data


def test_yahoo_detect_sync_lists_drafting(client, monkeypatch):
    import dashboard_services.draft_sync as ds
    fake = _FakeProvider(snapshot=_snap("drafting"))
    monkeypatch.setattr(ds, "get_draft_sync_provider", lambda platform: fake)
    resp = client.get("/api/draft/detect?platform=yahoo&league_id=99&season=2026&sync=1")
    assert resp.status_code == 200
    drafts = resp.get_json()["drafts"]
    assert drafts[0]["status"] == "drafting"
    assert drafts[0]["draft_id"] == "yahoo_99_2026"


def test_yahoo_detect_auth_denied(client, monkeypatch):
    import dashboard_services.draft_sync as ds
    fake = _FakeProvider(error=DraftSyncAuthError("Yahoo denied"))
    monkeypatch.setattr(ds, "get_draft_sync_provider", lambda platform: fake)
    resp = client.get("/api/draft/detect?platform=yahoo&league_id=99&season=2026&sync=1")
    assert resp.status_code == 200
    assert resp.get_json()["error"] == "auth_denied"


def test_yahoo_relay_normalizes_extension_picks(client, monkeypatch):
    import routes.draft_api_bp as bp

    def fake_normalize(body):
        assert body["leagueId"] == "99"
        assert len(body["picks"]) == 1
        return {
            "source": "yahoo-relay",
            "league_id": "99",
            "season": 2026,
            "status": "drafting",
            "picks": [{
                "pick_no": 1,
                "player_id": "5938",
                "external_player_id": "30123",
                "name": "Justin Jefferson",
                "position": "WR",
                "team": "MIN",
                "unresolved": False,
                "source": "yahoo-relay",
            }],
            "fingerprint": "drafting|1|0|1|1|30123",
        }

    monkeypatch.setattr(bp, "_yahoo_relay_normalize", fake_normalize)
    with client.session_transaction() as sess:
        sess["account_id"] = 1
    resp = client.post(
        "/api/draft/yahoo-relay",
        json={
            "leagueId": "99",
            "season": "2026",
            "inProgress": True,
            "picks": [{"overallPickNumber": 1, "playerId": "30123", "teamId": "1"}],
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["source"] == "yahoo-relay"
    assert data["picks"][0]["player_id"] == "5938"


def test_yahoo_relay_rejects_unauthenticated(client):
    with client.session_transaction() as sess:
        sess.clear()
    resp = client.post(
        "/api/draft/yahoo-relay",
        json={
            "leagueId": "99",
            "season": 2026,
            "picks": [{"overallPickNumber": 1, "playerId": "1", "teamId": "1"}],
        },
    )
    assert resp.status_code == 401


def test_yahoo_relay_requires_picks_list(client):
    with client.session_transaction() as sess:
        sess["account_id"] = 1
    resp = client.post("/api/draft/yahoo-relay", json={"leagueId": "99", "season": 2026})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "picks_required"
