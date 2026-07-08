"""Tests for utils.format ordinal helpers."""
from utils.format import ordinal, ord_suffix


def test_basic_ordinals():
    assert ordinal(1) == "1st"
    assert ordinal(2) == "2nd"
    assert ordinal(3) == "3rd"
    assert ordinal(4) == "4th"
    assert ordinal(10) == "10th"


def test_teens_are_th():
    # The 11-13 special case is the whole reason to centralize this.
    assert ordinal(11) == "11th"
    assert ordinal(12) == "12th"
    assert ordinal(13) == "13th"
    assert ord_suffix(11) == "th"
    assert ord_suffix(13) == "th"


def test_twenties_and_beyond():
    assert ordinal(21) == "21st"
    assert ordinal(22) == "22nd"
    assert ordinal(23) == "23rd"
    assert ordinal(111) == "111th"
    assert ordinal(112) == "112th"
    assert ordinal(101) == "101st"


def test_handles_stringish_and_bad_input():
    assert ordinal("5") == "5th"
    assert ord_suffix(None) == "th"
    assert ordinal("x") == "x"
