"""Tests for the Linear A Fractional Arithmetic Engine (Ferrara et al. 2020)."""

from fractions import Fraction
from linear_a.accounting.fractions import FractionEngine


def test_fraction_engine_canonical_values():
    engine = FractionEngine()
    assert engine.parse_fraction_symbols("J") == Fraction(1, 2)
    assert engine.parse_fraction_symbols("E") == Fraction(1, 4)
    assert engine.parse_fraction_symbols("F") == Fraction(1, 8)
    assert engine.parse_fraction_symbols("K") == Fraction(1, 16)
    assert engine.parse_fraction_symbols("H") == Fraction(1, 12)
    assert engine.parse_fraction_symbols("A") == Fraction(1, 6)


def test_fraction_engine_compound_symbols():
    engine = FractionEngine()
    # JE = 1/2 + 1/4 = 3/4
    assert engine.parse_fraction_symbols("JE") == Fraction(3, 4)
    # EF = 1/4 + 1/8 = 3/8
    assert engine.parse_fraction_symbols("EF") == Fraction(3, 8)
    # JEF = 1/2 + 1/4 + 1/8 = 7/8
    assert engine.parse_fraction_symbols("JEF") == Fraction(7, 8)
    # List format ['J', 'E']
    assert engine.parse_fraction_symbols(["J", "E"]) == Fraction(3, 4)


def test_fraction_formatting():
    engine = FractionEngine()
    assert engine.format_fraction(Fraction(31, 1)) == "31"
    assert engine.format_fraction(Fraction(1, 2)) == "1/2"
    assert engine.format_fraction(Fraction(7, 4)) == "1 3/4"
    assert engine.format_fraction(Fraction(11, 8)) == "1 3/8"


def test_evaluate_entry():
    engine = FractionEngine()
    total_frac, formatted = engine.evaluate_entry(5, "J")
    assert total_frac == Fraction(11, 2)
    assert formatted == "5 1/2"

    total_frac2, formatted2 = engine.evaluate_entry(10, ["E"])
    assert total_frac2 == Fraction(41, 4)
    assert formatted2 == "10 1/4"
