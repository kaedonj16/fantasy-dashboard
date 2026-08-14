from pathlib import Path


RENDER_CONFIG = Path(__file__).resolve().parents[1] / "render.yaml"


def test_google_oauth_environment_is_provisioned_for_web_service():
    """Keep every variable required by ``_google_configured`` in Render."""
    config = RENDER_CONFIG.read_text(encoding="utf-8")

    assert "- key: GOOGLE_CLIENT_ID\n        sync: false" in config
    assert "- key: GOOGLE_CLIENT_SECRET\n        sync: false" in config
    assert (
        "- key: GOOGLE_REDIRECT_URI\n"
        "        value: https://brfantasyfootball.com/auth/google/callback"
    ) in config
