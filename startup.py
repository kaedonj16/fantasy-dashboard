#!/usr/bin/env python3
"""Root entry point shim for Render.

render.yaml's web service uses ``startCommand: python startup.py`` (run from
the repo root). The real production startup logic lives in
``data_building/updates/startup.py``; this shim runs that file as ``__main__``.
Post-deploy path resolution is rooted at the repo (two levels above the real
startup module), not at this shim's ``__file__``.
"""
import os
import runpy

_REAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data_building", "updates", "startup.py",
)

if __name__ == "__main__":
    runpy.run_path(_REAL, run_name="__main__")
