"""Regression checks for the dashboard Front Office Report control."""

from pathlib import Path


APP_JS = Path(__file__).parents[1] / "static" / "app.js"
DASHBOARD_CSS = Path(__file__).parents[1] / "static" / "dashboard.css"


def test_generate_report_uses_delegated_click_handler():
    """Cold dashboard builds replace the body after DOMContentLoaded.

    The report control must therefore be discovered from the click event rather
    than captured once during initial page setup.
    """
    source = APP_JS.read_text(encoding="utf-8")
    report_code = source[source.index("// GM Memo generation functionality"):]

    assert "document.addEventListener('click', async function(event)" in report_code
    assert "event.target.closest('#generateGmMemoBtn')" in report_code
    assert "fetch('/api/gm-memo'" in report_code
    # Only the post-result "Refresh Report" label forces a regeneration;
    # the first Generate click may serve the server's cached report.
    assert "force: forceRefresh" in report_code
    assert "'Refresh Report'" in report_code


def test_generate_report_restores_button_after_request():
    source = APP_JS.read_text(encoding="utf-8")
    report_code = source[source.index("// GM Memo generation functionality"):]

    assert "generateGmMemoBtn.disabled = true" in report_code
    assert "} finally {" in report_code
    assert "generateGmMemoBtn.disabled = false" in report_code
    assert "Refresh Report" in report_code


def test_in_season_dashboard_does_not_prefill_report():
    """The report body must stay hidden until Generate Report is clicked."""
    from pathlib import Path

    dash = (Path(__file__).parents[1] / "dashboard_services" / "pages" / "dashboard_page.py").read_text(
        encoding="utf-8"
    )
    assert "get_team_gm_memo" not in dash
    assert "Generate Report" in dash
    assert 'Refresh Report" if gm_memo_html else "Generate Report' not in dash
    assert 'id="generateGmMemoBtn"' in dash
    assert 'id="gm-memo-result" style="display:none;"' in dash


def test_generate_report_surfaces_server_errors():
    """HTTP/API failures must not all collapse to a generic 'Network error'."""
    source = APP_JS.read_text(encoding="utf-8")
    report_code = source[source.index("// GM Memo generation functionality"):]
    assert "Failed to fetch" in report_code
    assert "Network error. Please try again." in report_code
    assert "response.text()" in report_code
    assert "JSON.parse(raw)" in report_code


def test_gm_memo_does_not_block_http_on_cold_playoff_sim():
    """Refresh Report used to stack a blocking Monte Carlo + OpenAI call and
    trip the edge proxy (UI: Network error). Odds warm in the background."""
    from pathlib import Path

    renderer = (Path(__file__).parents[1] / "dashboard_services" / "ai" / "renderer.py").read_text(
        encoding="utf-8"
    )
    fn = renderer[renderer.index("def get_team_gm_memo"):]
    fn = fn[: fn.index("\ndef get_front_office_briefing")]
    assert "block=False" in fn
    assert "block=True" not in fn

    source = DASHBOARD_CSS.read_text(encoding="utf-8")
    hero_rule = source[source.index(".os-hero-stats {"):]
    hero_rule = hero_rule[:hero_rule.index("}")]

    assert "grid-template-columns: repeat(4, minmax(0, 1fr));" in hero_rule

    responsive_source = source[source.index("@media (max-width: 1280px)", source.index(".os-hero-stats {")):]
    responsive_rule = responsive_source[responsive_source.index(".os-hero-stats {"):]
    responsive_rule = responsive_rule[:responsive_rule.index("}")]
    assert "grid-template-columns: 1fr 1fr;" in responsive_rule


def test_front_office_schema_required_covers_all_properties():
    """Strict structured output rejects the request when `required` misses any
    key in `properties` (400 invalid_json_schema, seen live 2026-09-25:
    "Missing 'trade_notes'")."""
    from dashboard_services.ai.prompts import _front_office_report_schema

    for scoring_type in ("redraft", "dynasty"):
        data = {
            "scoring_type": scoring_type,
            "trade_targets": [{"gets": [{"id": "t1"}]}],
            "waiver_targets": [{"id": "w1"}],
        }
        schema = _front_office_report_schema(data)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"].keys()), (
            f"required must list every property key ({scoring_type})"
        )


# -- Fingerprint-first cache probe --------------------------------------------

def _probe_ctx():
    return {
        "league_id": "L1",
        "current_week": 7,
        "current_season": 2026,
        "rosters": [
            {"roster_id": 1, "players": ["p1", "p2"]},
            {"roster_id": 2, "players": ["p3", "p4"]},
        ],
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX"],
        "roster_map": {},
        "standings_map": {1: {"record": "5-2", "PF": 900.0, "PA": 800.0}},
        "players_index": {
            "p1": {"full_name": "Alpha QB", "position": "QB", "team": "KC"},
            "p2": {"full_name": "Beta RB", "position": "RB", "team": "BUF"},
            "p3": {"full_name": "Gamma WR", "position": "WR", "team": "MIA"},
            "p4": {"full_name": "Delta TE", "position": "TE", "team": "BAL"},
        },
        "players_map": {},
        "players": {"p1": {"injury_status": "Active"}},
        "model_value_table": [
            {"id": "p1", "value": 500.0, "position": "QB"},
            {"id": "p2", "value": 400.0, "position": "RB"},
            {"id": "p3", "value": 300.0, "position": "WR"},
            {"id": "p4", "value": 200.0, "position": "TE"},
        ],
    }


def _patch_report_io(monkeypatch, fo, *, cached_payload=None):
    state = {"builds": 0, "build_kwargs": None, "saved_keys": [], "loads": []}

    def fake_build(ctx, rid, **kwargs):
        state["builds"] += 1
        state["build_kwargs"] = kwargs
        return {
            "week": 7, "season": 2026, "record": "5-2",
            "grades": kwargs.get("grades") or [],
            "roster_rows": kwargs.get("roster_rows") or [],
        }

    def fake_load(key):
        state["loads"].append(key)
        return cached_payload

    monkeypatch.setattr(fo, "build_front_office_data", fake_build)
    monkeypatch.setattr(fo, "load_cached_ai_text", fake_load)
    monkeypatch.setattr(fo, "save_cached_ai_text",
                        lambda key, text, ttl=None: state["saved_keys"].append(key))
    monkeypatch.setattr(fo, "ai_available", lambda: False)
    monkeypatch.setattr(fo, "_fallback_ai", lambda data, reason=None: {"verdict": "HOLD"})
    monkeypatch.setattr(fo, "render_front_office_card_html", lambda data, ai: "CARD")
    monkeypatch.setattr(fo, "render_front_office_report_html", lambda data, ai: "REPORT")
    return state


def test_fingerprint_hit_serves_without_building(monkeypatch):
    import json as _json

    import dashboard_services.ai.front_office_report as fo

    ctx = _probe_ctx()
    payload = _json.dumps({"card_html": "C", "report_html": "R", "verdict": "BUY"})
    state = _patch_report_io(monkeypatch, fo, cached_payload=payload)
    out = fo.get_front_office_report(ctx, "1")
    assert out["cached"] is True and out["verdict"] == "BUY"
    assert state["builds"] == 0  # the expensive build never ran
    probe_key, _ = fo._front_office_probe(ctx, "1")
    assert state["loads"] == [probe_key]


def test_fingerprint_miss_builds_once_with_probe_inputs(monkeypatch):
    import dashboard_services.ai.front_office_report as fo

    ctx = _probe_ctx()
    state = _patch_report_io(monkeypatch, fo, cached_payload=None)
    out = fo.get_front_office_report(ctx, "1")
    assert out["cached"] is False
    assert state["builds"] == 1
    # The build reused the probe's lookup/rows/grades instead of recomputing.
    assert state["build_kwargs"]["model_value_lookup"]
    assert state["build_kwargs"]["roster_rows"]
    assert state["build_kwargs"]["grades"]
    probe_key, _ = fo._front_office_probe(ctx, "1")
    assert state["saved_keys"] == [probe_key]  # probe key == build key

    # force_refresh skips the probe entirely and rebuilds.
    out = fo.get_front_office_report(ctx, "1", force_refresh=True)
    assert out["cached"] is False
    assert state["builds"] == 2
    assert state["build_kwargs"] == {}


def test_injury_change_changes_fingerprint(monkeypatch):
    import dashboard_services.ai.front_office_report as fo

    ctx = _probe_ctx()
    key_before, _ = fo._front_office_probe(ctx, "1")
    ctx["players"]["p1"] = {"injury_status": "Out"}
    key_after, _ = fo._front_office_probe(ctx, "1")
    assert key_before != key_after
