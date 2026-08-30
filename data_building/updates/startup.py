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


def main():
    print("Production Startup - Fantasy Dashboard")
    print(f"Started: {datetime.now().isoformat()}")

    first_run_flag = os.path.join(
        __import__("tempfile").gettempdir(), "fantasy_dashboard_initialized"
    )

    if not os.path.exists(first_run_flag):
        print("First deployment detected - running initialization...")
        try:
            from scripts.initialize_production import main as init_main
            init_main()
            with open(first_run_flag, 'w', encoding='utf-8') as f:
                f.write(f"Initialized: {datetime.now().isoformat()}")
            print("First-time initialization completed successfully")
        except Exception as e:
            print(f"First-time initialization failed: {e}")
            print("Continuing with app startup (manual initialization may be needed)")
    else:
        print("Existing deployment detected - skipping initialization")
        with open(first_run_flag, 'r', encoding='utf-8') as f:
            print(f"Previously initialized: {f.read().strip()}")

    # Spawn post-deploy in the background so it doesn't delay gunicorn startup.
    # That process refreshes tokenless global ADP snapshots (Yahoo/ESPN/MFL) onto
    # THIS web container's disk (cron writes a different disk), then optionally
    # rebuilds breakout scores. The subprocess outlives this process (execvp
    # replaces us with gunicorn) and writes to stdout/stderr for Render logs.
    import subprocess
    post_deploy_script = resolve_post_deploy_script()
    if os.path.exists(post_deploy_script):
        print(
            "Spawning background post-deploy "
            "(global ADP refresh + breakout check)..."
        )
        subprocess.Popen(
            [sys.executable, post_deploy_script],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    else:
        print(
            f"WARNING: post-deploy script not found at {post_deploy_script}; "
            "skipping deploy-time ADP refresh"
        )

    port = int(os.environ.get('PORT', 5000))
    workers = int(os.environ.get('WEB_WORKERS', 3))
    threads = int(os.environ.get('WEB_THREADS', 2))

    print(f"\nStarting gunicorn on port {port} ({workers} workers x {threads} threads)")

    cmd = [
        sys.executable, "-m", "gunicorn",
        "app:app",
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
