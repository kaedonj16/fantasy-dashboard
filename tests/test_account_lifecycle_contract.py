from pathlib import Path


def test_google_oidc_uses_pkce_nonce_and_verified_id_token():
    source = Path("routes/google_auth_bp.py").read_text()
    assert '"code_challenge_method": "S256"' in source
    assert '"nonce": nonce' in source
    assert "verify_oauth2_token" in source
    assert 'info.get("nonce") != stored_nonce' in source
    assert "upsert_google_account(sub, email" in source


def test_login_and_onboarding_share_one_google_endpoint():
    markup = Path("app.py").read_text()
    assert '/auth/google?intent=login&amp;next=/' in markup
    assert '/auth/google?intent=onboarding&amp;next=/' in markup
    assert "account_auth_identities" in Path("dashboard_services/accounts.py").read_text()


def test_account_schema_tracks_lifecycle_and_connection_status():
    migration = Path("migrations/023_account_lifecycle.sql").read_text()
    assert "UNIQUE (auth_provider, auth_provider_subject)" in migration
    assert "last_active_league_id" in migration
    assert "last_successful_sync_at" in migration
    assert "reauth_required" in migration


def test_saved_leagues_remain_visible_with_connection_status():
    accounts = Path("dashboard_services/accounts.py").read_text()
    api = Path("routes/league_meta_bp.py").read_text()
    assert "LEFT JOIN fantasy_provider_connections" in accounts
    assert '"needs_reconnect"' in api


def test_last_active_destination_does_not_call_provider():
    source = Path("dashboard_services/accounts.py").read_text()
    body = source[source.index("def get_post_login_destination"):source.index("def mark_espn_connection_status")]
    assert "list_user_leagues" in body
    assert "get_league" not in body


def test_pending_private_credentials_are_encrypted_server_side():
    accounts = Path("dashboard_services/accounts.py").read_text()
    migration = Path("migrations/024_pending_provider_connections.sql").read_text()
    script = Path("static/app.js").read_text()
    assert "_encrypt_provider_credentials" in accounts
    assert "expires_at" in migration
    assert 'fetch("/api/link/espn/private/pending"' in script
    assert "pending_provider_connection_token" in Path("routes/google_auth_bp.py").read_text()
