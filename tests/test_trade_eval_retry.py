"""Regression tests for trade-calculator eval failure recovery (static/app.js).

Incident 2026-10-15: an assistant "Open in Trade Calculator" deep link staged
the trade correctly (both sides' chips rendered with values) but the Trade
Summary stayed at 0.0 / 0.0 / 0.0, the balance bar stayed centered, and
Playoff Impact sat in its "Build a trade" empty state. The URL loader was
NOT at fault: loadTradeFromURL writes the same state the summary reads and
calls the same renderChips -> recomputeTrade chain as the interactive
add-player path. The /api/trade-eval POSTs had failed (non-200), and the
failure handling stranded the page: the instant-restore pass paints the
empty state first, a failed recompute only wrote the error box, and nothing
ever retried, so the empty renders won permanently. A hand-built trade masks
this because every edit fires another eval; a URL-staged trade fires its
whole burst in the first second and never gets a second chance.

The fix: recomputeTrade retries a failed eval for the SAME trade a bounded
number of times (signature-scoped, generation-guarded), so a transient
failure recovers on its own and a persistent one stays bounded.
"""
from __future__ import annotations

import re
from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent / "static" / "app.js"


def _read() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _function_body(name: str, prefix: str = "") -> str:
    src = _read()
    m = re.search(
        rf"{prefix}function {name}\([^)]*\) \{{(.*?)\n  \}}\n",
        src,
        re.DOTALL,
    )
    assert m, f"{name}() not found in app.js"
    return m.group(1)


def _recompute() -> str:
    body = _function_body("recomputeTrade", prefix="async ")
    assert 'fetch("/api/trade-eval"' in body, "extraction sanity: eval fetch"
    return body


# -- The staged trade and the summary share one state -------------------------


def test_url_loader_stages_into_the_state_the_summary_reads():
    """loadTradeFromURL must write state.side{A,B}{Players,Picks} (the exact
    lists recomputeTrade serializes into the eval payload) and run the same
    render/recompute chain as the interactive path."""
    body = _function_body("loadTradeFromURL")
    for assignment in (
        "state.sideAPlayers = aIds.map(",
        "state.sideBPlayers = bIds.map(",
        "state.sideAPicks = apIds.map(",
        "state.sideBPicks = bpIds.map(",
    ):
        assert assignment in body, f"URL loader must stage via {assignment}"
    assert 'renderChips("A")' in body and 'renderChips("B")' in body
    assert "recomputeTrade()" in body


def test_recompute_serializes_payload_from_state():
    """The eval payload's four id lists come from the shared state, so a
    staged trade (players AND picks) is evaluated exactly like a built one."""
    body = _recompute()
    assert "const sideAIds = state.sideAPlayers.map(" in body
    assert "const sideBIds = state.sideBPlayers.map(" in body
    assert "const sideAPickIds = state.sideAPicks.map(" in body
    assert "const sideBPickIds = state.sideBPicks.map(" in body
    for key in ("side_a_players: sideAIds", "side_b_players: sideBIds",
                "side_a_picks: sideAPickIds", "side_b_picks: sideBPickIds"):
        assert key in body, f"eval payload must carry {key}"


def test_render_chips_funnels_edits_into_recompute():
    """Manual edits after staging (chip add/remove call renderChips) must
    re-evaluate: renderChips ends in syncEmptyState + recomputeTrade."""
    body = _function_body("renderChips")
    assert re.search(r"syncEmptyState\(side\);\s*\n\s*recomputeTrade\(\);", body), (
        "renderChips must finish by recomputing the trade"
    )


# -- Bounded, generation-guarded retry on eval failure -------------------------


def test_retry_state_is_signature_scoped():
    """Retry bookkeeping keys on the trade's signature (all four id lists),
    so an edit resets the budget instead of retrying a stale trade."""
    src = _read()
    assert "let _evalRetryTimer = null;" in src
    assert "let _evalRetrySig = \"\";" in src
    assert "let _evalRetryAttempts = 0;" in src

    body = _recompute()
    assert re.search(
        r"const _evalSig = sideAIds\.join\(\",\"\) \+ \"\|\" \+ sideBIds\.join\(\",\"\)"
        r" \+ \"\|\" \+\s*sideAPickIds\.join\(\",\"\) \+ \"\|\" \+ sideBPickIds\.join\(\",\"\)",
        body,
    ), "signature must cover players and picks on both sides"
    assert re.search(
        r"if \(_evalSig !== _evalRetrySig\) \{\s*"
        r"_evalRetrySig = _evalSig;\s*"
        r"_evalRetryAttempts = 0;\s*"
        r"if \(_evalRetryTimer\) \{ clearTimeout\(_evalRetryTimer\); _evalRetryTimer = null; \}",
        body,
    ), "a new signature must reset attempts and cancel a pending retry"


def test_failed_eval_retries_bounded_and_generation_guarded():
    """The catch path schedules at most 2 retries, only for a non-empty trade
    whose recompute is still the latest, and the timer re-checks the
    generation before firing so a superseded retry never runs."""
    body = _recompute()
    catch = body.split("} catch (err) {", 1)[1]
    assert "gen === _tradeGeneration" in catch
    assert "_evalRetryAttempts < 2" in catch, "retries must be bounded"
    assert "_evalRetryAttempts += 1" in catch
    assert re.search(
        r"\(sideAIds\.length \|\| sideBIds\.length \|\| "
        r"sideAPickIds\.length \|\| sideBPickIds\.length\)",
        catch,
    ), "retries are only for a non-empty trade"
    assert re.search(
        r"_evalRetryTimer = setTimeout\(\(\) => \{\s*"
        r"_evalRetryTimer = null;\s*"
        r"if \(_retryGen === _tradeGeneration\) recomputeTrade\(\);",
        catch,
    ), "the retry timer must re-check the generation before recomputing"
    assert "1500" in catch and "4000" in catch, "backoff delays (1.5s, 4s)"


def test_successful_eval_clears_retry_bookkeeping():
    """Once an eval lands, attempts reset and any pending retry timer is
    cancelled, so a recovered trade does not keep re-firing."""
    body = _recompute()
    after_gen_check = body.split("if (gen !== _tradeGeneration) return;", 1)[1]
    applied = after_gen_check.split("const diff = Number(data.diff)", 1)[0]
    assert "_evalRetryAttempts = 0;" in applied
    assert "clearTimeout(_evalRetryTimer)" in applied
