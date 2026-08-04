"""save_usage_rows_for_season must not clobber a good usage file with
target_share-starved data (which happens when nfl_data_py play-by-play is
unavailable for the season and every target_share comes back 0). That silently
tanks all breakout scores, so the write is refused unless explicitly forced.
"""
import json

import pytest

ph = pytest.importorskip("data_building.external_data.player_history")


def _rows(n_with_ts, n_total):
    rows = []
    for i in range(n_total):
        ts = 0.15 if i < n_with_ts else 0.0
        rows.append({"id": str(i), "name": f"P{i}", "team": "IND",
                     "usage": {"target_share": ts, "avg_targets": 5.0}})
    return rows


def _write(monkeypatch, tmp_path, rows, force=False):
    monkeypatch.setattr(ph, "usage_rows_json_path_for_season",
                        lambda season: tmp_path / f"usage_rows_{season}.json")
    return ph.save_usage_rows_for_season(rows, 2025, force=force)


def test_refuses_to_overwrite_good_file_with_starved_data(monkeypatch, tmp_path):
    good = _rows(n_with_ts=400, n_total=800)
    _write(monkeypatch, tmp_path, good)
    # Rebuilt rows with target_share all zeroed (source down) -> keep the good file.
    starved = _rows(n_with_ts=0, n_total=800)
    _write(monkeypatch, tmp_path, starved)
    saved = json.loads((tmp_path / "usage_rows_2025.json").read_text())
    assert ph._target_share_coverage(saved) == 400  # unchanged — write refused


def test_force_overrides_guard(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _rows(400, 800))
    _write(monkeypatch, tmp_path, _rows(0, 800), force=True)
    saved = json.loads((tmp_path / "usage_rows_2025.json").read_text())
    assert ph._target_share_coverage(saved) == 0  # forced through


def test_env_var_overrides_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("FORCE_USAGE_OVERWRITE", "1")
    _write(monkeypatch, tmp_path, _rows(400, 800))
    _write(monkeypatch, tmp_path, _rows(0, 800))
    saved = json.loads((tmp_path / "usage_rows_2025.json").read_text())
    assert ph._target_share_coverage(saved) == 0


def test_healthy_rebuild_still_writes(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _rows(400, 800))
    # A healthy rebuild with comparable coverage overwrites normally.
    _write(monkeypatch, tmp_path, _rows(420, 820))
    saved = json.loads((tmp_path / "usage_rows_2025.json").read_text())
    assert ph._target_share_coverage(saved) == 420


def test_writes_when_no_existing_file(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _rows(0, 10))  # nothing to protect -> write
    assert (tmp_path / "usage_rows_2025.json").exists()
