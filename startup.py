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

    first_run_flag = "/tmp/fantasy_dashboard_initialized"

    if not os.path.exists(first_run_flag):
        print("First deployment detected - running initialization...")
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
        print("Existing deployment detected - skipping initialization")
        with open(first_run_flag, 'r') as f:
            print(f"Previously initialized: {f.read().strip()}")

    port = int(os.environ.get('PORT', 5000))
    workers = int(os.environ.get('WEB_WORKERS', 3))
    threads = int(os.environ.get('WEB_THREADS', 2))

    print(f"\nStarting gunicorn on port {port} ({workers} workers x {threads} threads)")

    import subprocess
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
