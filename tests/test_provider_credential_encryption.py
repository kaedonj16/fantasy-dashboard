from pathlib import Path

import pytest

from dashboard_services import accounts


def test_render_generates_a_stable_provider_encryption_key():
    blueprint = Path("render.yaml").read_text()
    declaration = blueprint[blueprint.index("- key: PROVIDER_CREDENTIAL_ENCRYPTION_KEY") :]
    declaration = declaration[:declaration.index("- key:", 1)]
    assert "generateValue: true" in declaration


def test_provider_credential_secret_falls_back_to_flask_secret(monkeypatch, caplog):
    monkeypatch.delenv("PROVIDER_CREDENTIAL_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("FLASK_SECRET_KEY", "stable-deployment-secret")
    monkeypatch.setattr(accounts, "_ENCRYPTION_FALLBACK_LOGGED", False)

    assert accounts._provider_credential_secret() == "stable-deployment-secret"
    assert "using the stable FLASK_SECRET_KEY fallback" in caplog.text
    assert "stable-deployment-secret" not in caplog.text


def test_provider_credential_secret_prefers_dedicated_key(monkeypatch):
    monkeypatch.setenv("PROVIDER_CREDENTIAL_ENCRYPTION_KEY", "provider-key")
    monkeypatch.setenv("FLASK_SECRET_KEY", "flask-key")

    assert accounts._provider_credential_secret() == "provider-key"


def test_provider_credentials_require_a_stable_server_secret(monkeypatch):
    monkeypatch.delenv("PROVIDER_CREDENTIAL_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)

    with pytest.raises(accounts.ProviderCredentialConfigurationError):
        accounts._provider_credential_secret()
