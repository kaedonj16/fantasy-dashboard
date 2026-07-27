"""Guard: the players SSR path must actually hide the loading skeleton.

The /players route server-renders the top players into #prList, then hides the
#prLoading skeleton with a string .replace() on the exact skeleton markup. When
the skeleton was redesigned (a centered spinner -> a .sk-list skeleton) the
replace target wasn't updated, so it silently no-op'd and the skeleton stayed
stranded above the SSR table forever ("the page never finishes loading").

This test doesn't need the DB: it checks the invariant on app.py's source — the
tag the SSR path replaces must match the tag actually in the template, and the
replacement must hide it. If the skeleton markup is edited again without updating
the SSR hide, this fails instead of shipping a stuck skeleton.
"""
import os
import re

_APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_ssr_path_hides_the_players_skeleton():
    src = open(_APP, encoding="utf-8").read()

    # The skeleton element as it appears in the players template (raw HTML: the
    # occurrence preceded by whitespace, not a single-quoted Python literal).
    m = re.search(r'\n[ \t]*(<div id="prLoading"[^>]*>)\n', src)
    assert m, "could not find the #prLoading skeleton element in the players template"
    tag = m.group(1)
    assert "display:none" not in tag, "template skeleton should start visible"

    # The SSR path must string-replace exactly this tag (as a Python literal),
    # otherwise the .replace() no-ops and the skeleton is never hidden.
    assert ("'" + tag + "'") in src, (
        "SSR skeleton-hide does not target the current #prLoading markup — the "
        ".replace() will silently no-op and strand the skeleton over the SSR table"
    )

    # …and the replacement it produces must actually hide the skeleton.
    hidden = re.findall(r'<div id="prLoading"[^>]*display:none[^>]*>', src)
    assert hidden, "the SSR path never sets display:none on #prLoading — skeleton stays visible"
