from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mfl_is_a_platform_option_on_the_home_page():
    source = (ROOT / "app.py").read_text()
    # Selectable platform button in the home connect widget.
    assert '<button type="button" class="platform-btn" data-platform="mfl">MFL <span class="platform-limit">Public</span></button>' in source
    # Dedicated flow with a League ID + season entry and a connect action.
    assert 'id="mflFlow"' in source
    assert 'id="mflLeagueIdInput"' in source
    assert 'id="mflSeasonInput"' in source
    assert 'id="mflSubmitBtn"' in source


def test_home_mfl_flow_is_wired_in_the_client():
    script = (ROOT / "static" / "app.js").read_text()
    # Switching to MFL reveals its flow.
    assert 'mflFlow.style.display     = platform === "mfl"' in script
    # Connect validates via the public MFL preview endpoint.
    assert "/api/link/mfl/preview?league_id=" in script
    # Success reveals the shared choose-league + continue choice and tags the form.
    assert 'formPlatform.value = "mfl"' in script
