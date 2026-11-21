"""Sunday rookie step budget + CFBD store (2026-11-21).

On 2026-11-08 and 2026-11-15 the weekly rookie step inside the
daily cron was killed at the default 3600s step timeout while still
fetching college stats, so no prospect grades were written for two
Sundays running (the job still exited 0, so Render showed success).
Two causes, both pinned here:

1. The step ran with the default 1h budget for an eval + full
   pipeline pass. It now gets 3h.
2. The daily-job ran WITHOUT the persistent CFBD response store, so
   the fetch stage re-pulled everything live from CFBD every week.
   The step now enables the store in its own subprocess env, and
   render.yaml carries the flag for the service as well.
"""

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CRON = (_REPO / "cron_daily.py").read_text(encoding="utf-8")
_YAML = (_REPO / "render.yaml").read_text(encoding="utf-8")


def _rookie_step_block() -> str:
    start = _CRON.index("Step 6: Weekly rookie data")
    end = _CRON.index("Step 6b", start)
    return _CRON[start:end]


class TestRookieStepBudget:
    def test_rookie_step_timeout_is_three_hours(self):
        block = _rookie_step_block()
        assert '"build_weekly_rookie_data", timeout=10800' in block

    def test_rookie_step_enables_cfbd_persistent_store(self):
        block = _rookie_step_block()
        assert 'os.environ.setdefault("CFBD_PERSIST_STORE", "1")' in block

    def test_other_steps_keep_the_default_timeout(self):
        # The raise is scoped to the rookie step: _run_step's default
        # stays 3600 for everything else.
        assert "def _run_step(code: str, step_name: str, timeout: int = 3600)" in _CRON


class TestRenderYamlDailyJob:
    def test_daily_job_env_enables_cfbd_persistent_store(self):
        start = _YAML.index("name: daily-job")
        nxt = _YAML.index("- type:", start + 1)
        block = _YAML[start:nxt]
        assert re.search(r"key: CFBD_PERSIST_STORE\s*\n\s*value: \"1\"", block)
