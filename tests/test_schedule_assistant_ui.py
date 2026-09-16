from pathlib import Path


SOURCE = Path("dashboard_services/pages/schedule_page.py").read_text()


def test_range_presets_and_manual_sync_are_derived_from_ranges():
    assert 'id="schedRosPreset"' in SOURCE
    assert "applyPreset(CFG.startWeek, CFG.maxWeek)" in SOURCE
    assert "applyPreset(1, CFG.maxWeek)" in SOURCE
    assert "applyPreset(CFG.playoffStart, CFG.playoffEnd)" in SOURCE
    assert "wkStart === 1 && wkEnd === CFG.maxWeek" in SOURCE
    assert "syncPresetBtns();" in SOURCE
    assert "for (var w = 1; w <= CFG.maxWeek; w++)" in SOURCE


def test_tooltips_explain_metric_seasons_weights_and_not_projection():
    assert "SOS measures a player’s matchup difficulty" in SOURCE
    assert "This is not ' + p.name + '’s projected score" in SOURCE
    assert "Early-season ratings blend" in SOURCE
    assert "currentWeight" in SOURCE and "previousWeight" in SOURCE
    assert "#1 = easiest matchup · Points below = fantasy points allowed per game" in SOURCE


def test_unavailable_values_are_not_rendered_as_zero():
    assert "c.fpts != null ? (c.fpts + ' pts') : 'N/A'" in SOURCE

