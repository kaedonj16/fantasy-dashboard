"""Yahoo privacy placeholders must never render as a team or owner name."""
from dashboard_services.display_names import (
    public_owner_label,
    team_label_from_user,
    username_from_user,
)


def test_public_owner_label_skips_yahoo_hidden():
    assert public_owner_label("--hidden--", "Sunday Funday B") == "Sunday Funday B"
    assert public_owner_label("  --hidden--  ", None, fallback="Team 1") == "Team 1"
    assert public_owner_label("hidden", "-hidden-", "Red Zone Zach") == "Red Zone Zach"
    assert public_owner_label("Sunday alex") == "Sunday alex"
    assert public_owner_label(None, "", fallback="Team 3") == "Team 3"


def test_team_label_prefers_metadata_team_name_over_hidden_display():
    user = {
        "display_name": "--hidden--",
        "username": "--hidden--",
        "metadata": {"team_name": "Sunday Funday B"},
    }
    roster = {"metadata": {"team_name": "Sunday Funday B"}}
    assert team_label_from_user(user, roster, fallback="Team 1") == "Sunday Funday B"
    # /api/teams used to read top-level team_name / display_name only.
    assert team_label_from_user(
        {"display_name": "--hidden--"},
        {"metadata": {"team_name": "Flea Flicker Nick"}},
        fallback="Team 2",
    ) == "Flea Flicker Nick"


def test_username_from_user_never_returns_hidden():
    assert username_from_user({
        "username": "--hidden--",
        "display_name": "--hidden--",
        "metadata": {"team_name": "HENDO Z World"},
    }) == "HENDO Z World"
    assert username_from_user({"display_name": "Red Zone Zach"}) == "Red Zone Zach"
