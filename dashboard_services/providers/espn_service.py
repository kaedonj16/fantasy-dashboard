from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

# ESPN lineupSlotId conventions (most leagues):
# 20 = Bench, 21 = IR
ESPN_BENCH_SLOT = 20
ESPN_IR_SLOT = 21


def _record_and_streak(team_raw: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns:
      record_str like 'LLWLLLLLWWWWWW' (chronological order not guaranteed by ESPN)
      streak_str like '6W'
    """
    rec = (team_raw.get("record") or {}).get("overall") or {}
    streak_len = int(rec.get("streakLength") or 0)
    streak_type = (rec.get("streakType") or "").upper()  # WIN / LOSS / TIE

    streak = "0"
    if streak_len > 0 and streak_type:
        streak = f"{streak_len}{'W' if streak_type == 'WIN' else ('L' if streak_type == 'LOSS' else 'T')}"

    # ESPN does not always provide a clean game-by-game string.
    # Some leagues expose "recordByPeriod" / "outcomes" in other views,
    # but in your raw snippet I only see aggregates.
    #
    # If you later pull a per-week outcomes list, plug it in here.
    record_str = ""  # default empty when we don't have per-week results

    return record_str, streak


