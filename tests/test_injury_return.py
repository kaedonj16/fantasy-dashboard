"""ESPN injury return cache behavior."""
import json
import time
from urllib.error import HTTPError

import pytest

from dashboard_services import injury_return as ir


@pytest.fixture(autouse=True)
def _reset_injury_return_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "espn_return_dates.json"
    monkeypatch.setattr(ir, "_CACHE_FILE", cache_file)
    monkeypatch.setattr(ir, "_CACHE", {"ts": 0.0, "by_pid": {}})
    monkeypatch.setattr(ir, "_FETCH_BLOCKED_UNTIL", 0.0)
    monkeypatch.setattr(ir, "_LAST_FETCH_WARN", 0.0)
    yield


def test_refresh_uses_stale_disk_when_fetch_blocked(monkeypatch, tmp_path):
    stale = {"p1": {"player_id": "p1", "return_date": "2026-09-01"}}
    ir._CACHE_FILE.write_text(json.dumps(stale), encoding="utf-8")
    # Backdate file so it is outside the fresh TTL.
    old = time.time() - ir._TTL - 60
    import os

    os.utime(ir._CACHE_FILE, (old, old))

    def _boom():
        raise HTTPError(ir._ESPN_INJURIES_URL, 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr(ir, "_fetch_espn_json", _boom)
    out = ir.refresh_espn_return_dates()
    assert out == stale
    assert ir._FETCH_BLOCKED_UNTIL > time.time()


def test_refresh_skips_network_while_blocked(monkeypatch):
    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        raise HTTPError(ir._ESPN_INJURIES_URL, 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr(ir, "_fetch_espn_json", _boom)
    ir._FETCH_BLOCKED_UNTIL = time.time() + 3600
    assert ir.refresh_espn_return_dates() == {}
    assert ir.refresh_espn_return_dates() == {}
    assert calls["n"] == 0


# ── Parser: REAL live payload shape (regression, 2026-10-18) ────────────────
# The live ESPN injuries payload's athlete object has NO "id" field in any
# entry; the athlete id lives only in headshot.href / links. The entry's own
# "id" is a per-report entry id. Fixtures that include athlete.id hid the bug
# where by_pid was keyed by entry ids and every consumer lookup missed.

def _real_shape_payload():
    return {"injuries": [{
        "id": "11",  # IND team group
        "injuries": [{
            "id": "643002",  # report-entry id, NOT the athlete id
            "status": "Out",
            "athlete": {
                "displayName": "Tyler Warren",
                "shortName": "T. Warren",
                "headshot": {"href": "https://a.espncdn.com/i/headshots/nfl/players/full/4431459.png"},
                "links": [{"href": "https://www.espn.com/nfl/player/_/id/4431459/tyler-warren"}],
            },
            "details": {"type": {"description": "Undisclosed"}},
        }],
    }]}


def test_parse_real_shape_headshot_id_keys_by_canonical():
    out = ir.parse_espn_injuries_payload(
        _real_shape_payload(), {"4431459": "12518"})
    assert list(out) == ["12518"]
    assert "643002" not in out
    row = out["12518"]
    assert row["espn_id"] == "4431459"
    assert row["status"] == "Out"
    assert row["name"] == "Tyler Warren"
    assert row["team"] == "IND"


def test_parse_real_shape_links_id_when_no_headshot():
    payload = _real_shape_payload()
    del payload["injuries"][0]["injuries"][0]["athlete"]["headshot"]
    out = ir.parse_espn_injuries_payload(payload, {"4431459": "12518"})
    assert list(out) == ["12518"]
    assert out["12518"]["espn_id"] == "4431459"


def test_parse_athlete_id_field_takes_priority_when_present():
    payload = _real_shape_payload()
    payload["injuries"][0]["injuries"][0]["athlete"]["id"] = "9999999"
    out = ir.parse_espn_injuries_payload(payload, {"9999999": "canon9"})
    assert list(out) == ["canon9"]
    assert out["canon9"]["espn_id"] == "9999999"


def test_parse_skips_entry_with_no_identifiable_athlete():
    payload = _real_shape_payload()
    athlete = payload["injuries"][0]["injuries"][0]["athlete"]
    del athlete["headshot"]
    del athlete["links"]
    out = ir.parse_espn_injuries_payload(payload, {"4431459": "12518"})
    assert out == {}


def test_refresh_force_bypasses_block(monkeypatch):
    payload = {"injuries": []}
    calls = {"n": 0}

    def _ok():
        calls["n"] += 1
        return payload

    monkeypatch.setattr(ir, "_fetch_espn_json", _ok)
    monkeypatch.setattr(
        "dashboard_services.providers.global_adp.espn_id_to_canonical",
        lambda: {},
    )
    ir._FETCH_BLOCKED_UNTIL = time.time() + 3600
    ir.refresh_espn_return_dates(force=True)
    assert calls["n"] == 1
