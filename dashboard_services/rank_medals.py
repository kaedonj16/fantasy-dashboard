"""Flat rank chips for leaderboards (standings, power rankings, awards).

A shared helper so every record/achievement board renders the same podium marks
as the All-Time Standings table: a gold "1", a muted "2", a bronze "3", and a
plain bordered chip for the rest. The marks are styled by shared classes in
dashboard.css (.rank / .rank-first / .rank-second / .rank-third / .rank-plain),
so callers just drop the returned string into a cell — no inline SVG.

Deliberately NOT used on value boards — value fluctuates, so there's no
meaningful "first place" to crown.
"""
from __future__ import annotations

# rank → shared chip class (matches _MEDALS in the All-Time Standings table).
_RANK_CLASS = {
    1: "rank rank-first",
    2: "rank rank-second",
    3: "rank rank-third",
}


def rank_mark(rank, size: int = 36, wrap: bool = True, ring_others: bool = True) -> str:
    """Return an All-Time-Standings rank chip for a leaderboard row.

    Ranks 1/2/3 render as the gold/grey/bronze numerals and 4+ as a bordered
    `.rank-plain` chip — the same flat marks the All-Time Standings table uses.
    `rank` may be an int or anything int-coercible; a non-numeric rank falls back
    to the original text in a plain chip. `wrap` centers the chip in a fixed-width
    flex box so table columns line up; `size` sets that box's min width. `ring_others`
    is accepted for backwards compatibility and no longer changes the 4+ mark.
    """
    try:
        r = int(rank)
        label = str(r)
    except (TypeError, ValueError):
        r, label = 0, str(rank)

    cls = _RANK_CLASS.get(r)
    inner = f'<span class="{cls}">{label}</span>' if cls else f'<span class="rank-plain">{label}</span>'

    if not wrap:
        return inner
    return (
        f'<span class="rank-mark" style="display:inline-flex;align-items:center;'
        f'justify-content:center;min-width:{size + 8}px">{inner}</span>'
    )
