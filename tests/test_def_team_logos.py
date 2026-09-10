"""NFL DEF/DST images use team logos, not Sleeper headshots."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_def_team_logo_urls_local_and_espn_was():
    pytest.importorskip("flask")
    from utils.utils import def_team_logo_urls, _espn_logo_url

    local, espn = def_team_logo_urls("SF")
    assert local == "/static/images/team_logos/SF.png"
    assert "espncdn.com" in espn and "/sf.png" in espn.lower()

    # Washington: local file is WAS.png; ESPN CDN slug is wsh.
    local_w, espn_w = def_team_logo_urls("WSH")
    assert local_w == "/static/images/team_logos/WAS.png"
    assert "wsh.png" in espn_w.lower()
    assert "was.png" not in _espn_logo_url("WAS").lower()


def test_pinfo_for_pid_includes_def_logo():
    pytest.importorskip("flask")
    from utils.utils import pinfo_for_pid, load_teams_index

    ti = load_teams_index() or {}
    assert "SF" in ti
    info = pinfo_for_pid("SF", {}, ti, {})
    assert info["pos"] == "DEF"
    assert info["nfl"] == "SF"
    assert info.get("logo")
    assert "espncdn.com" in info["logo"]
    assert info.get("logo_local") == "/static/images/team_logos/SF.png"


def test_redzone_and_app_js_use_team_logos_for_def():
    rz = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
    assert "brTeamLogoLocal" in rz or "/static/images/team_logos/" in rz
    assert "rz-team-logo" in rz
    assert "brDefImgOnError" in rz or "espncdn.com/i/teamlogos" in rz

    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    # Real end-of-core marker (not the earlier comment that mentions the same string).
    core = app_js.split("\n// @public-js:core-end")[0]
    assert "window.brPlayerImgUrl" in core
    assert "window.brDefImgOnError" in core
    assert "window.brTeamLogoLocal" in core

    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".rz-headshot.rz-team-logo" in css


def test_face_html_and_league_players_attach_def_logos():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    face = app.split("def _face_html")[1].split("\ndef _player_row")[0]
    assert "def_team_logo_urls" in face
    assert 'pos == "DEF"' in face

    assert '"espnHeadshot": _logo' in app
    assert "Synthesize a minimal meta so the modal can show the" in app


def test_draft_room_and_extension_cover_def_logos():
    dr = (ROOT / "static" / "draft_room.js").read_text(encoding="utf-8")
    assert "playerImgTag" in dr
    assert "brPlayerImgUrl" in dr
    assert "WAS" in dr and "wsh" in dr

    ext = (ROOT / "extension" / "overlay.js").read_text(encoding="utf-8")
    hs = ext.split("function hsUrl(p)")[1].split("function hsMark")[0]
    assert "DEF" in hs
    assert "teamlogos" in hs
