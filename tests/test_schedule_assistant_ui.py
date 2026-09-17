from pathlib import Path


SOURCE = Path("dashboard_services/pages/schedule_page.py").read_text()


def test_range_presets_and_manual_sync_are_derived_from_ranges():
    assert 'class="sched-presets" role="group" aria-label="Schedule range presets"' in SOURCE
    assert 'id="schedRosPreset"' in SOURCE
    assert "applyPreset(CFG.startWeek, CFG.maxWeek)" in SOURCE
    assert "applyPreset(1, CFG.maxWeek)" in SOURCE
    assert "applyPreset(CFG.playoffStart, CFG.playoffEnd)" in SOURCE
    assert "wkStart === 1 && wkEnd === CFG.maxWeek" in SOURCE
    assert "syncPresetBtns();" in SOURCE
    assert "for (var w = 1; w <= CFG.maxWeek; w++)" in SOURCE


def test_mobile_range_presets_have_a_dedicated_wrapping_row():
    css = Path("static/dashboard.css").read_text()
    assert ".sched-presets" in css
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in css
    assert "flex: 0 0 100%;" in css
    assert ".sched-week-range .sched-select" in css


def test_tooltips_explain_metric_seasons_weights_and_not_projection():
    assert "Ease ranks the selected schedule" in SOURCE
    assert "This is not ' + p.name + '’s projected score" in SOURCE
    assert "Early-season ratings blend" in SOURCE
    assert "currentWeight" in SOURCE and "previousWeight" in SOURCE
    assert "Adjusted matchup ratings compare what a defense allowed" in SOURCE


def test_unavailable_values_are_not_rendered_as_zero():
    assert "c.adjusted_percent != null" in SOURCE
