#!/usr/bin/env python3
"""
Production Startup Script
Entry point for Render deployment that handles initialization
before starting the Flask application.
"""

import os
import sys
from datetime import datetime


def main():
    print("Production Startup - Fantasy Dashboard")
    print(f"Started: {datetime.now().isoformat()}")

    # Always run migrations on every startup — all SQL files use
    # CREATE TABLE/INDEX IF NOT EXISTS so re-running is safe.
    print("Running database migrations...")
    try:
        from scripts.run_migrations import run_migrations
        run_migrations()
        print("Migrations completed successfully")
    except Exception as e:
        print(f"Migration failed: {e}")
        print("Continuing with app startup (some features may be unavailable)")

    first_run_flag = "/tmp/fantasy_dashboard_initialized"

    if not os.path.exists(first_run_flag):
        print("First deployment detected - running full initialization...")
        try:
            from scripts.initialize_production import main as init_main
            init_main()
            with open(first_run_flag, 'w') as f:
                f.write(f"Initialized: {datetime.now().isoformat()}")
            print("First-time initialization completed successfully")
        except Exception as e:
            print(f"First-time initialization failed: {e}")
            print("Continuing with app startup (manual initialization may be needed)")
    else:
        print("Existing deployment detected - skipping full initialization")
        with open(first_run_flag, 'r') as f:
            print(f"Previously initialized: {f.read().strip()}")

    # Spawn the post-deploy breakout rebuild in the background so it doesn't
    # delay gunicorn startup.  The subprocess outlives this process (execvp
    # replaces us with gunicorn) and writes directly to stdout/stderr which
    # Render captures in the service logs.
    import subprocess
    post_deploy_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scripts", "post_deploy.py"
    )
    if os.path.exists(post_deploy_script):
        print("Spawning background post-deploy breakout refresh...")
        subprocess.Popen(
            [sys.executable, post_deploy_script],
            stdout=sys.stdout,
            stderr=sys.stderr,
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
