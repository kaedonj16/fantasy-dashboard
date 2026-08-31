#!/usr/bin/env bash
#
# Cloud Agent install: durable, idempotent setup for the Fantasy Dashboard.
#
# Installs system packages (PostgreSQL), a pinned Python 3.11 toolchain, and the
# project's Python dependencies into a local virtualenv. Per-boot service
# startup and schema reconciliation live in start.sh, not here.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- System packages: PostgreSQL server (durable; baked into build snapshots) ---
if ! command -v pg_ctlcluster >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq postgresql postgresql-contrib
fi

# --- Python 3.11 via uv (matches .python-version / CI's astral setup-uv) ---
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
PY_VERSION="$(tr -d '[:space:]' < .python-version)"
uv python install "$PY_VERSION"

# --- Virtualenv + dependencies ---
# pywebpush and requests-html are excluded: two pinned sdists that fail to build
# on some toolchains and are not needed to import or test the web app (the one
# test that needs pywebpush importorskips itself). Mirrors scripts/dev_setup.sh
# and the CI integration job.
#
# Recreate the venv only when it is missing or built against a different Python;
# otherwise reuse it so repeated installs stay fast.
if [ -x .venv/bin/python ] && .venv/bin/python -c "import sys; raise SystemExit(0 if sys.version.split()[0]=='${PY_VERSION}' else 1)"; then
  echo "install.sh: reusing existing .venv (Python ${PY_VERSION})"
else
  uv venv --clear --python "$PY_VERSION" .venv
fi
grep -viE '^(pywebpush|requests-html)' requirements.txt > /tmp/req-dev.txt
uv pip install --python .venv/bin/python -r /tmp/req-dev.txt ruff pytest

echo "install.sh: complete"
