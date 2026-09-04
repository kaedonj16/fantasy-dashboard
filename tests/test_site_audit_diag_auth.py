"""Site-audit #24: diagnostic APIs require X-Admin-Secret."""


def test_proj_debug_requires_admin_secret(offline_client, monkeypatch):
    monkeypatch.setenv("ADMIN_SECRET", "diag-secret")
    r = offline_client.get("/api/proj-debug?name=stroud")
    assert r.status_code == 403
    assert r.get_json().get("error") == "Forbidden"


def test_market_intel_health_requires_admin_secret(offline_client, monkeypatch):
    monkeypatch.setenv("ADMIN_SECRET", "diag-secret")
    r = offline_client.get("/api/market-intel/health")
    assert r.status_code == 403
    assert r.get_json().get("error") == "Forbidden"


def test_diag_endpoints_reject_wrong_secret(offline_client, monkeypatch):
    monkeypatch.setenv("ADMIN_SECRET", "diag-secret")
    headers = {"X-Admin-Secret": "wrong"}
    for path in ("/api/proj-debug?name=stroud", "/api/market-intel/health"):
        r = offline_client.get(path, headers=headers)
        assert r.status_code == 403, path
        assert r.get_json().get("error") == "Forbidden"


def test_diag_endpoints_accept_correct_secret(offline_client, monkeypatch):
    monkeypatch.setenv("ADMIN_SECRET", "diag-secret")
    headers = {"X-Admin-Secret": "diag-secret"}

    # Missing/empty name → 400 once past the auth gate (or 200 if data present).
    r = offline_client.get("/api/proj-debug", headers=headers)
    assert r.status_code != 403
    assert r.status_code in (200, 400)

    # Health may hit DB; accept non-forbidden outcomes.
    r = offline_client.get("/api/market-intel/health", headers=headers)
    assert r.status_code != 403
    assert r.status_code in (200, 500)


def test_diag_endpoints_forbidden_when_admin_secret_unset(offline_client, monkeypatch):
    monkeypatch.delenv("ADMIN_SECRET", raising=False)
    for path in ("/api/proj-debug?name=stroud", "/api/market-intel/health"):
        r = offline_client.get(path, headers={"X-Admin-Secret": "anything"})
        assert r.status_code == 403, path
