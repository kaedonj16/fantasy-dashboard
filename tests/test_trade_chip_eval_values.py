"""Trade Calculator: chips repaint their values from the eval response.

Chips price client-side from the cached player pool (brGetLeaguePlayersData's
localStorage stale-while-revalidate copy), while the Trade Summary totals are
priced by /api/trade-eval from the DB. After a value update the two vintages
disagree (2026-11-01 screenshot: Javonte Williams chip 404.6, summary and
modal 397.4). The eval response already carries per-asset values in each
side's breakdown, so when an eval lands recomputeTrade repaints the chips
from that same response: a chip can then only ever show the number the
summary on screen was priced from.

Pins:
- recomputeTrade calls paintChipValuesFromEval(data) AFTER the generation
  staleness guard, so a superseded response can never repaint chips.
- The painter reads both sides' breakdowns, skips unvalued items (their
  dash treatment belongs to applyChipValueDashes), matches chips by
  data-chip-pid, and writes through formatValue.
- renderChips stamps dataset.chipPid on pick chips as well as player
  chips, or picks could never be repainted.
"""
import os

_HERE = os.path.dirname(__file__)
_JS = os.path.join(_HERE, "..", "static", "app.js")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _body(js, start_marker, end_marker):
    return js.split(start_marker, 1)[1].split(end_marker, 1)[0]


def test_recompute_paints_chips_from_the_landing_eval():
    js = _read(_JS)
    recompute = _body(js, "async function recomputeTrade()", "async function fetchSimilarTrades(gen, signal)")
    guard = "if (gen !== _tradeGeneration) return;"
    call = "paintChipValuesFromEval(data);"
    assert guard in recompute
    assert call in recompute
    # The paint must sit after the staleness guard AND after the totals
    # are read from the same response (the summary it has to match).
    assert recompute.index(guard) < recompute.index(call)
    assert recompute.index("Number(data.side_a.effective_total)") < recompute.index(call)


def test_painter_reads_both_breakdowns_and_skips_unvalued():
    js = _read(_JS)
    painter = _body(js, "function paintChipValuesFromEval(data)", "async function recomputeTrade()")
    assert "data.side_a" in painter and "data.side_b" in painter
    assert "#sideAChips" in painter and "#sideBChips" in painter
    assert "sideData.breakdown" in painter
    # Unvalued assets keep the dash applyChipValueDashes owns.
    assert "item.unvalued === true" in painter
    # Only finite numbers paint; a missing/garbled value leaves the chip.
    assert "Number.isFinite(val)" in painter
    # Chips are matched by the id renderChips stamps, values formatted
    # the same way renderChips formats them.
    assert 'data-chip-pid="${String(item.id)}"' in painter
    assert 'querySelector(".otc-chip-value")' in painter
    assert "formatValue(val)" in painter


def test_render_chips_stamps_chip_pid_on_picks_and_players():
    js = _read(_JS)
    chips = _body(js, "function renderChips(side)", "function formatPickId")
    assert chips.count("chip.dataset.chipPid = String(") == 2
    players = chips.split("players.forEach", 1)[1].split("picks.forEach", 1)[0]
    assert "chip.dataset.chipPid = String(p.id)" in players
    picks = chips.split("picks.forEach", 1)[1]
    assert "chip.dataset.chipPid = String(pk.id)" in picks
