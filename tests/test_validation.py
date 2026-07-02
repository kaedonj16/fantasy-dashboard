"""Unit tests for utils.validation.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.validation import safe_int, validate_league_id


# ---- safe_int -------------------------------------------------------------

def test_safe_int_valid():
    assert safe_int("42") == 42
    assert safe_int(42.9) == 42
    assert safe_int(0) == 0


def test_safe_int_invalid_returns_default():
    assert safe_int("abc") is None
    assert safe_int(None) is None
    assert safe_int("abc", -1) == -1
    assert safe_int(None, 0) == 0


# ---- validate_league_id ---------------------------------------------------

def test_missing_league_id():
    ok, err = validate_league_id("sleeper", "")
    assert ok is False
    assert "required" in err.lower()


def test_sleeper_numeric_ok():
    assert validate_league_id("sleeper", "123456789") == (True, None)


def test_sleeper_non_numeric_rejected():
    ok, err = validate_league_id("sleeper", "abc123")
    assert ok is False
    assert "Sleeper" in err


def test_espn_and_yahoo_numeric_ok():
    assert validate_league_id("espn", "555")[0] is True
    assert validate_league_id("yahoo", "777")[0] is True


def test_espn_non_numeric_rejected():
    ok, err = validate_league_id("espn", "x")
    assert ok is False
    assert "ESPN" in err


def test_platform_case_and_whitespace_normalized():
    assert validate_league_id("  Sleeper ", "123")[0] is True


def test_unsupported_platform():
    ok, err = validate_league_id("myfantasyleague", "123")
    assert ok is False
    assert "Unsupported platform" in err
