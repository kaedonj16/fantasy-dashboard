"""Empty starter slots must not 500 matchup slides on league switch."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "dashboard_services" / "matchups.py").read_text(encoding="utf-8")


def test_player_bits_empty_returns_six_values():
    """zip_longest fillvalue=None hits the empty branch; unpack expects 6."""
    start = SRC.find("def player_bits(")
    end = SRC.find("rows_html: List[str] = []", start)
    body = SRC[start:end]
    assert 'if not p:' in body
    assert 'return "", 0.0, None, False, False, None' in body
    # Success path also returns 6.
    assert "is_not_started, (stats if stats else None)" in body
