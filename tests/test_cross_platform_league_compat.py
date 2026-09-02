"""Cross-platform helpers and league-format switching coverage."""
from __future__ import annotations

import json
from pathlib import Path

from dashboard_services.ai.context_builders import ctx_scoring_type
from dashboard_services.providers.mfl_api import _mfl_sleeper_league_type
from utils.trade_value import player_trade_value


def test_tzset_guarded_in_app_source():
    src = Path("app.py").read_text(encoding="utf-8")
    assert 'hasattr(time, "tzset")' in src
    assert "time.tzset()" in src


def test_no_unix_only_strftime_dash_format():
    for path in (
        Path("app.py"),
        Path("dashboard_services/injuries.py"),
        Path("dashboard_services/player_league_trades.py"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "%-m" not in text, f"{path} still uses %-m"
        assert "%-d" not in text, f"{path} still uses %-d"


def test_page_html_tmp_uses_tempdir():
    src = Path("app.py").read_text(encoding="utf-8")
    assert "tempfile.gettempdir()" in src
    assert "/tmp/br_page_" not in src


def test_extension_declares_storage_permission():
    manifest = json.loads(Path("extension/manifest.json").read_text(encoding="utf-8"))
    assert "storage" in manifest["permissions"]


def test_ctx_scoring_type_honors_mfl_league_type_string():
    assert ctx_scoring_type({"league_settings": {"league_type": "redraft"}}) == "redraft"
    assert ctx_scoring_type({"league_settings": {"league_type": "keeper"}}) == "redraft"
    assert ctx_scoring_type({"league_settings": {"league_type": "dynasty"}}) == "dynasty"
    assert ctx_scoring_type(
        {"league_settings": {"type": 2, "league_type": "redraft"}}
    ) == "dynasty"


def test_mfl_sleeper_league_type_mapping():
    assert _mfl_sleeper_league_type("Redraft") == (0, "redraft")
    assert _mfl_sleeper_league_type("Keeper") == (1, "keeper")
    assert _mfl_sleeper_league_type("Dynasty") == (2, "dynasty")
    assert _mfl_sleeper_league_type("D") == (2, "dynasty")


def test_yahoo_sleeper_league_type_from_max_keepers():
    # Avoid importing yahoo_api (pulls requests/bs4). Eval the helper from source.
    import ast
    import types

    from utils.coerce import safe_int

    src = Path("dashboard_services/providers/yahoo_api.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_yahoo_sleeper_league_type"
    )
    mod = types.ModuleType("_yahoo_iso")
    mod._safe_int = safe_int
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<yahoo>", "exec"), mod.__dict__)
    fn_impl = mod._yahoo_sleeper_league_type

    assert fn_impl({}, {}, 12) == 0
    assert fn_impl({}, {"max_keepers": 3}, 12) == 1
    assert fn_impl({}, {"max_keepers": 30}, 12) == 2
    assert fn_impl({"cant_cut_list": "yahoo"}, {}, 12) == 1


def test_redraft_size_columns_in_player_trade_value():
    p = {
        "position": "WR",
        "redraft_value_1qb": 100,
        "redraft_value_12": 140,
        "redraft_value_sf": 110,
        "redraft_sf_value_12": 160,
    }
    assert player_trade_value(
        p, league_type="1qb", league_size=10, scoring_type="redraft"
    ) == 100.0
    assert player_trade_value(
        p, league_type="1qb", league_size=12, scoring_type="redraft"
    ) == 140.0
    assert player_trade_value(
        p, league_type="sf", league_size=12, scoring_type="redraft"
    ) == 160.0


def test_load_pick_value_table_selects_sf_axis(tmp_path, monkeypatch):
    from dashboard_services import picks as picks_mod

    payload = {
        "date": "2026-01-01",
        "1qb": {"2027_1_early": 50.0, "2027_1_01": 80.0},
        "sf": {"2027_1_early": 90.0, "2027_1_01": 120.0},
    }
    wls = tmp_path / "pick_values_wls_latest.json"
    wls.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(picks_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(picks_mod, "_load_fc_slot_pick_values", lambda *a, **k: {})

    oneqb = picks_mod.load_pick_value_table(current_year=2026, is_sf=False)
    sf = picks_mod.load_pick_value_table(current_year=2026, is_sf=True)
    assert oneqb["2027_1_early"] == 50.0
    assert sf["2027_1_early"] == 90.0
    assert sf["2027_1_01"] == 120.0


def test_league_is_redraft_source_honors_league_type_string():
    src = Path("app.py").read_text(encoding="utf-8")
    assert 'settings.get("league_type")' in src
    assert '"redraft", "keeper"' in src


def test_league_is_redraft_yahoo_defaults_to_redraft():
    src = Path("app.py").read_text(encoding="utf-8")
    assert 'platform") or "").strip().lower() == "yahoo"' in src
    assert "Yahoo has no dynasty product" in src
    yahoo_block = src.split("Yahoo has no dynasty product", 1)[1][:500]
    assert "return True" in yahoo_block
