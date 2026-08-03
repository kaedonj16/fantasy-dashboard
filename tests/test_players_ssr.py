"""Guard: the players SSR path must actually hide the loading skeleton.

The /players route server-renders the top players into #prList, then hides the
#prLoading skeleton with a string .replace() on the exact skeleton markup. When
the skeleton was redesigned (a centered spinner -> a .sk-list skeleton) the
replace target wasn't updated, so it silently no-op'd and the skeleton stayed
stranded above the SSR table forever ("the page never finishes loading").

This test doesn't need the DB: it checks the invariant across the two source
files — the players shell (dashboard_services/pages/players_page.py) holds the
template markup, while app.py's page_players does the SSR string .replace(). The
tag the SSR path replaces must match the tag actually in the shell, and the
replacement must hide it. If the skeleton markup is edited in one place without
updating the other, this fails instead of shipping a stuck skeleton.
"""
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP = os.path.join(_ROOT, "app.py")
_SHELL = os.path.join(_ROOT, "dashboard_services", "pages", "players_page.py")


def test_ssr_path_hides_the_players_skeleton():
    app_src = open(_APP, encoding="utf-8").read()
    shell_src = open(_SHELL, encoding="utf-8").read()

    # The skeleton element as it appears in the players shell template (raw HTML:
    # the occurrence preceded by whitespace, not a single-quoted Python literal).
    m = re.search(r'\n[ \t]*(<div id="prLoading"[^>]*>)\n', shell_src)
    assert m, "could not find the #prLoading skeleton element in the players shell"
    tag = m.group(1)
    assert "display:none" not in tag, "shell skeleton should start visible"

    # The SSR path (in app.py) must string-replace exactly this tag (as a Python
    # literal), otherwise the .replace() no-ops and the skeleton is never hidden.
    assert ("'" + tag + "'") in app_src, (
        "SSR skeleton-hide does not target the current #prLoading markup — the "
        ".replace() will silently no-op and strand the skeleton over the SSR table"
    )

    # …and the replacement it produces must actually hide the skeleton.
    hidden = re.findall(r'<div id="prLoading"[^>]*display:none[^>]*>', app_src)
    assert hidden, "the SSR path never sets display:none on #prLoading — skeleton stays visible"
