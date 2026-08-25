"""Ensure deploy startup can locate scripts/post_deploy.py (ADP refresh)."""

import os
import runpy


def test_resolve_post_deploy_script_points_at_repo_scripts():
    startup_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data_building",
        "updates",
        "startup.py",
    )
    ns = runpy.run_path(os.path.abspath(startup_path))
    resolved = ns["resolve_post_deploy_script"]()
    expected = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "post_deploy.py")
    )
    assert resolved == expected
    assert os.path.exists(resolved), (
        f"post_deploy.py missing at {resolved}; deploy-time ADP refresh would skip"
    )


def test_resolve_post_deploy_script_accepts_explicit_root(tmp_path):
    startup_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data_building",
        "updates",
        "startup.py",
    )
    ns = runpy.run_path(os.path.abspath(startup_path))
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    resolved = ns["resolve_post_deploy_script"](str(fake_root))
    assert resolved == str(fake_root / "scripts" / "post_deploy.py")
