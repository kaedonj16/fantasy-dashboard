#!/usr/bin/env python3
"""Root entry point shim for Render.

render.yaml's web service uses ``startCommand: python startup.py`` (run from
the repo root). The real production startup logic lives in
``data_building/updates/startup.py``; this shim executes it as ``__main__`` so
its ``__file__``-relative paths (e.g. scripts/post_deploy.py) still resolve.
"""
import os
import runpy

_REAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data_building", "updates", "startup.py",
)

if __name__ == "__main__":
    runpy.run_path(_REAL, run_name="__main__")
