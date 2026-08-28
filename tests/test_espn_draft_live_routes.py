"""Flask route tests for ESPN live-draft detect/live. ESPN is mocked."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

flask = pytest.importorskip("flask")

from dashboard_services.draft_sync import (
    DraftSyncAuthError,
    DraftSyncSnapshot,
    DraftSyncUnavailableError,
    NormalizedDraftPick,
    snapshot_to_live_payload,
)
from routes.draft_api_bp import draft_api_bp


@dataclass
class _FakeProvider:
    source: str = "espn"
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
                source="espn", overall_pick=1, canonical_player_id="5938",
                external_player_id="4039057", external_team_id="1",
                picked_by="{AAA}", roster_id="1", name="Justin Jefferson",
                position="WR", team="MIN",
            )
        ]
    return DraftSyncSnapshot(
        source="espn", draft_id="espn_99_2026", league_id="99", season=2026,
        status=status, drafted=(status == "complete"),
        in_progress=(status == "drafting"), picks=picks, teams=4, rounds=15,
        order="snake", picks_observed=True, live_detail_present=True,
        poll_interval_ms=8000, viewer_team_id="1",
        user_roster_map={"{AAA}": "1"}, draft_order={"{AAA}": 1, "1": 1},
        slot_names={"1": "Alpha"},
    )


@pytest.fixture
def client(monkeypatch):
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(draft_api_bp)
    fake = _FakeProvider(snapshot=_snap())
    monkeypatch.setattr(
        "dashboard_services.draft_sync.get_draft_sync_provider",
        lambda platform: fake,
    )
    # The detect helper imports get_draft_sync_provider from draft_sync inside
    # the function; patch the name used by the route module after import.
    import routes.draft_api_bp as bp
    monkeypatch.setattr(bp, "_espn_detect_sync", bp._espn_detect_sync)
    with app.test_client() as test_client:
        with test_client.session_transaction() as sess:
            sess["viewer_user_id"] = "{AAA}"
            sess["viewer_roster_id"] = "1"
        test_client._fake = fake  # type: ignore[attr-defined]
        yield test_client


def test_espn_live_returns_normalized_picks(client, monkeypatch):
    import routes.draft_api_bp as bp
    fake = client._fake

    def provider(_platform):
        return fake

    monkeypatch.setattr("dashboard_services.draft_sync.get_draft_sync_provider", provider)
    # _espn_live imports get_draft_sync_provider inside the function.
    import dashboard_services.draft_sync as ds
    monkeypatch.setattr(ds, "get_draft_sync_provider", provider)
    resp = client.get("/api/draft/live?platform=espn&draft_id=espn_99_2026")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["source"] == "espn"
    assert data["status"] == "drafting"
    assert data["picks"][0]["player_id"] == "5938"
    assert data["picks"][0]["pick_no"] == 1
    assert "espn_s2" not in data and "swid" not in data
    blob = resp.get_data(as_text=True)
    assert "espn_s2" not in blob and "SWID" not in blob


def test_espn_live_auth_error_does_not_retry(client, monkeypatch):
    import dashboard_services.draft_sync as ds
    fake = _FakeProvider(error=DraftSyncAuthError("ESPN denied access to this league."))
    monkeypatch.setattr(ds, "get_draft_sync_provider", lambda platform: fake)
    resp = client.get("/api/draft/live?platform=espn&draft_id=espn_99_2026")
    assert resp.status_code == 403
    assert resp.get_json()["retry"] is False
    assert "secret" not in resp.get_data(as_text=True)


def test_espn_live_temporary_error_retries(client, monkeypatch):
    import dashboard_services.draft_sync as ds
    fake = _FakeProvider(error=DraftSyncUnavailableError("ESPN is temporarily unavailable."))
    monkeypatch.setattr(ds, "get_draft_sync_provider", lambda platform: fake)
    resp = client.get("/api/draft/live?platform=espn&draft_id=espn_99_2026")
    assert resp.status_code == 502
    assert resp.get_json()["retry"] is True


def test_espn_detect_sync_lists_drafting(client, monkeypatch):
    import dashboard_services.draft_sync as ds
    fake = _FakeProvider(snapshot=_snap("drafting"))
    monkeypatch.setattr(ds, "get_draft_sync_provider", lambda platform: fake)
    resp = client.get("/api/draft/detect?platform=espn&league_id=99&season=2026&sync=1")
    assert resp.status_code == 200
    drafts = resp.get_json()["drafts"]
    assert drafts[0]["status"] == "drafting"
    assert drafts[0]["draft_id"] == "espn_99_2026"


def test_espn_detect_without_sync_does_not_call_provider(client, monkeypatch):
    called = []

    def boom(*a, **k):
        called.append(1)
        raise AssertionError("should not fetch mDraftDetail")

    import dashboard_services.draft_sync as ds
    monkeypatch.setattr(ds, "get_draft_sync_provider", boom)
    monkeypatch.setattr(
        "dashboard_services.platform_api.get_drafts",
        lambda *a, **k: [{"draft_id": "espn_99_2026", "status": "pre_draft", "start_time": 1}],
    )
    # draft_api_bp imports get_drafts at module level
    import routes.draft_api_bp as bp
    monkeypatch.setattr(bp, "get_drafts", lambda *a, **k: [{
        "draft_id": "espn_99_2026", "status": "pre_draft", "season": 2026, "start_time": 1, "type": "snake",
    }])
    resp = client.get("/api/draft/detect?platform=espn&league_id=99&season=2026")
    assert resp.status_code == 200
    assert called == []
    assert resp.get_json()["drafts"][0]["status"] == "pre_draft"


def test_espn_live_predraft_returns_empty_picks(client, monkeypatch):
    import dashboard_services.draft_sync as ds
    fake = _FakeProvider(snapshot=_snap("pre_draft", picks=[]))
    monkeypatch.setattr(ds, "get_draft_sync_provider", lambda platform: fake)
    resp = client.get("/api/draft/live?platform=espn&draft_id=espn_99_2026")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "pre_draft"
    assert data["picks"] == []


def test_sleeper_live_still_rejects_empty_draft_id(client):
    resp = client.get("/api/draft/live?platform=sleeper&draft_id=")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "unsupported"
