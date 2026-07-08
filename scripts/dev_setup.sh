#!/usr/bin/env bash
#
# Create a local virtualenv the app can import and be tested from.
#
# Why a venv (and not the system interpreter): the full app needs the Flask +
# scientific stack (pandas, numpy, scipy, scikit-learn, plotly, psycopg, ...).
# Two pinned sdists in requirements.txt fail to build under some toolchains and
# are NOT needed to import or test the web app, so they're excluded here:
#   - pywebpush     (web push notifications)
#   - requests-html (a scraper used by offline data pipelines)
# Install the full requirements.txt directly if you need those features.
#
# Usage:
#   bash scripts/dev_setup.sh            # creates ./.venv
#   VENV=/path/to/venv bash scripts/dev_setup.sh
#
# Then:
#   .venv/bin/python -m pytest           # full suite incl. integration pages
#   FLASK_SECRET_KEY=dev .venv/bin/python -m pytest tests/test_integration_pages.py
#
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"
REQ_TMP="$(mktemp)"
trap 'rm -f "$REQ_TMP"' EXIT

echo "==> Creating venv at $VENV"
python3 -m venv "$VENV"

echo "==> Upgrading build tooling"
"$VENV/bin/pip" install -q -U pip setuptools wheel

echo "==> Installing requirements (excluding pywebpush, requests-html)"
grep -viE '^(pywebpush|requests-html)' requirements.txt > "$REQ_TMP"
"$VENV/bin/pip" install -q -r "$REQ_TMP"

echo "==> Installing test tooling"
"$VENV/bin/pip" install -q pytest

echo
echo "Done. The app imports from this venv; run the suite with:"
echo "  FLASK_SECRET_KEY=dev $VENV/bin/python -m pytest -q"
