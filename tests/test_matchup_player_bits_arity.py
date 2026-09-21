"""Empty starter slots must not 500 matchup slides on league switch."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "dashboard_services" / "matchups.py").read_text(encoding="utf-8")


def test_player_bits_empty_returns_nine_values():
    """zip_longest fillvalue=None hits the empty branch; the compact board rows
    unpack a 9-tuple (info, pos, actual, proj, bye, not_started, stats, nfl,
    pid), so the empty branch must match that arity or the slide 500s."""
    start = SRC.find("def player_bits(")
    end = SRC.find("rows_html: List[str] = []", start)
    body = SRC[start:end]
    assert 'if not p:' in body
    assert 'return "", "", 0.0, None, False, False, None, "", ""' in body
    # Success path returns the same 9 values, ending in the nfl / pid strings.
    assert 'str(nfl or "").upper()' in body
    assert 'str(pid or "")' in body
