import os

import utils.utils as utils


def test_parsed_json_cache_is_bounded_and_file_changes_invalidate(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "_JSON_CACHE_MAX", 2)
    with utils._JSON_CACHE_LOCK:
        utils._JSON_CACHE.clear()
    paths = []
    for number in range(3):
        path = tmp_path / f"{number}.json"
        path.write_text(f'{{"value": {number}}}', encoding="utf-8")
        paths.append(str(path))
        assert utils.read_json_cached(str(path))["value"] == number
    assert len(utils._JSON_CACHE) == 2
    assert paths[0] not in utils._JSON_CACHE

    path = tmp_path / "2.json"
    path.write_text('{"value": 99, "changed": true}', encoding="utf-8")
    os.utime(path, None)
    assert utils.read_json_cached(str(path))["value"] == 99
