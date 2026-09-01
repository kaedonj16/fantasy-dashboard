#!/usr/bin/env python3
"""Preview or send a scoped weekly digest. Never intended as the production cron.

Examples:
    python scripts/preview_weekly_digest.py --dry-run --account-id 123 --out /tmp/digest.html
    python scripts/preview_weekly_digest.py --preview-platform sleeper --preview-league L1 \\
        --preview-season 2026 --preview-roster 1 --out /tmp/digest.html
    python scripts/preview_weekly_digest.py --account-id 123 --force
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.weekly_email import main

if __name__ == "__main__":
    raise SystemExit(main())
