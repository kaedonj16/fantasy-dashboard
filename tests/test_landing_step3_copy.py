"""Pin the connect card's step-3 copy logic.

Sleeper's picker holds leagues - username search results, or the single
pasted league - and the username already chose the team, so step 3 reads
"Choose league" there and the subtitle names the team once it resolves.
Every other platform's third step is a genuine team pick."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "static" / "app.js").read_text()
APP_PY = (ROOT / "app.py").read_text()


def test_step3_label_switches_on_picker_mode():
    for needle in (
        "setStep3Label",
        '"Choose league"',
        '"Choose team"',
        "applyStep3ForLeague",
        'leagueSelect.dataset.mode = "username"',
        'leagueSelect.dataset.mode = "link"',
        "leagueSelect.dataset.username",
    ):
        assert needle in APP_JS, f"app.js missing step-3 logic: {needle}"


def test_step3_subtitle_names_the_team_for_username_picks():
    for needle in (
        'setStep3Sub("Your league in", name, ", so every tool is about your roster.")',
        '", with team "',
        "/api/teams?league_id=",
        "step3TeamCache",
    ):
        assert needle in APP_JS, f"app.js missing team-aware subtitle piece: {needle}"


def test_board_strip_markup_and_week_context():
    assert "hp-mb-strip" in APP_PY
    assert "This week&#x27;s matchup" in APP_PY
    assert "Week {{ current_week }}" in APP_PY
    # Every FORM_BODY render in the index route passes the week.
    assert APP_PY.count('current_week=int(nfl_state.get("week") or 0)') >= 3
