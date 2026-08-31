"""Ensure deploy startup can locate scripts/post_deploy.py (ADP refresh)."""

import os
import runpy
import sys


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


def test_post_deploy_puts_repo_root_on_sys_path():
    """`python scripts/post_deploy.py` must resolve project packages.

    Render's startup spawns this as a file path, so Python puts scripts/ on
    sys.path instead of the repo root. Without an explicit insert, the
    post-deploy imports of scripts / dashboard_services / data_building all
    raise ModuleNotFoundError and migrations, ADP refresh, and breakout
    rebuilds silently no-op.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scripts_dir = os.path.join(repo_root, "scripts")
    post_deploy = os.path.join(scripts_dir, "post_deploy.py")

    original = list(sys.path)
    try:
        # Simulate `python scripts/post_deploy.py`: script dir is first.
        sys.path.insert(0, scripts_dir)
        ns = runpy.run_path(post_deploy, run_name="post_deploy_path_test")
        # Do not call ns["main"] — that sleeps and hits the DB.
        assert "main" in ns
        abs_paths = [os.path.abspath(p) for p in sys.path]
        assert repo_root in abs_paths, abs_paths[:8]
        assert abs_paths.index(repo_root) < abs_paths.index(scripts_dir)
        # Repo-root-first is what makes these production imports resolve:
        #   from scripts.run_migrations import run_migrations
        #   from dashboard_services.adp_service import refresh_global_adp_sources
        #   from data_building.breakout_engine.build_historical_scores import run
        assert os.path.isdir(os.path.join(repo_root, "dashboard_services"))
        assert os.path.isdir(os.path.join(repo_root, "data_building"))
        assert os.path.isfile(os.path.join(repo_root, "scripts", "run_migrations.py"))
    finally:
        sys.path[:] = original
