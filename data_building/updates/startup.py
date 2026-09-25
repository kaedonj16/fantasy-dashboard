#!/usr/bin/env python3
"""
Production Startup Script
Entry point for Render deployment that handles initialization
before starting the Flask application.
"""

import os
import sys
from datetime import datetime

# This file lives at data_building/updates/startup.py; post_deploy lives at
# scripts/post_deploy.py under the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_post_deploy_script(repo_root=None):
    """Return the absolute path to scripts/post_deploy.py."""
    root = repo_root if repo_root is not None else _REPO_ROOT
    return os.path.join(root, "scripts", "post_deploy.py")


def _db_already_initialized() -> bool:
    """True when the production database already has its core tables.

    The previous first-run flag lived in tempfile.gettempdir(), which is
    fresh on every Render deploy, so the heavy first-time init
    (migrations + daily data build + full-app health check) ran before
    gunicorn on EVERY deploy and blocked the port bind -- failed deploys
    with "No open ports detected". Probing the database itself is the
    correct already-initialized signal.
    """
    try:
        from dashboard_services.db import get_conn
    except Exception as exc:  # pragma: no cover - defensive
        print(f"init check: could not import db layer ({exc}); assuming fresh")
        return False
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM player_values LIMIT 1")
                cur.fetchone()
        return True
    except Exception as exc:
        # UndefinedTable on a genuinely fresh database, or a transient
        # connection problem: fall through to init, whose own try/except
        # keeps startup moving toward gunicorn on failure.
        print(f"init check: probe failed ({type(exc).__name__}: {exc}); "
              "assuming fresh database")
        return False


def main():
    # Line-buffer stdout/stderr so Render logs show startup progress live.
    # A hang before the port bind is otherwise invisible: block-buffered
    # prints never flush when the process never reaches gunicorn.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    from dashboard_services.memory_diagnostics import format_memory_snapshot
    print("Production Startup - Fantasy Dashboard")
    print(f"Started: {datetime.now().isoformat()}")
    print(format_memory_snapshot("web startup begin"))

    if _db_already_initialized():
        print("Database already initialized - skipping first-time init")
    else:
        print("Fresh database detected - running initialization...")
        print(format_memory_snapshot("before first-time initialization"))
        try:
            from scripts.initialize_production import main as init_main
            init_main()
            print("First-time initialization completed successfully")
        except Exception as e:
            print(f"First-time initialization failed: {e}")
            print("Continuing with app startup (manual initialization may be needed)")
        finally:
            print(format_memory_snapshot("after first-time initialization"))

    # Spawn post-deploy in the background so it doesn't delay gunicorn startup.
    # That process runs migrations and refreshes tokenless global ADP snapshots
    # (Yahoo/ESPN/MFL) onto THIS web container's disk (cron writes a different
    # disk). The subprocess outlives this process (execvp
    # replaces us with gunicorn) and writes to stdout/stderr for Render logs.
    import subprocess
    post_deploy_script = resolve_post_deploy_script()
    if os.path.exists(post_deploy_script):
        print(
            "Spawning background post-deploy "
            "(migrations + global ADP refresh)..."
        )
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            _REPO_ROOT if not existing_pp
            else _REPO_ROOT + os.pathsep + existing_pp
        )
        subprocess.Popen(
            [sys.executable, post_deploy_script],
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=_REPO_ROOT,
            env=env,
        )
    else:
        print(
            f"WARNING: post-deploy script not found at {post_deploy_script}; "
            "skipping deploy-time ADP refresh"
        )

    port = int(os.environ.get('PORT', 5000))
    workers = int(os.environ.get('WEB_WORKERS', 2))
    threads = int(os.environ.get('WEB_THREADS', 2))

    print(f"\nStarting gunicorn on port {port} ({workers} workers x {threads} threads)")
    print(format_memory_snapshot("before gunicorn exec"))

    cmd = [
        sys.executable, "-m", "gunicorn",
        "app:app",
        # Lifecycle hooks (post_fork cache warmup). CLI flags still override
        # anything in the config file. Path is absolute: the exec'd gunicorn
        # inherits this process's cwd, but absolute is robust either way.
        "--config", os.path.join(_REPO_ROOT, "gunicorn_conf.py"),
        "--bind", f"0.0.0.0:{port}",
        "--workers", str(workers),
        "--threads", str(threads),
        "--worker-class", "gthread",
        "--timeout", "120",
        "--keep-alive", "5",
        "--max-requests", "1000",
        "--max-requests-jitter", "100",
        "--preload",
        "--access-logfile", "-",
        "--error-logfile", "-",
    ]
    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
