"""Unit tests for Diophantine Multi-Fraction Tablet Solver."""

from fractions import Fraction
from linear_a.accounting.diophantine_solver import DiophantineTabletSolver


def test_diophantine_solver_single_item():
    solver = DiophantineTabletSolver()
    # E.g. Known: 3, 5, 10, 5. Total: 31. Unknown: 8.
    known = [Fraction(3, 1), Fraction(5, 1), Fraction(10, 1), Fraction(5, 1)]
    total = Fraction(31, 1)

    sol = solver.solve_missing_item(known, total, "HT_009", "item[4]")
    assert sol.is_valid_minoan is True
    assert sol.is_unique is True
    assert sol.integer_part == 8
    assert sol.fraction_part == Fraction(0, 1)
    assert sol.fraction_symbols == ""


def test_diophantine_solver_fraction_recovery():
    solver = DiophantineTabletSolver()
    # E.g. Known: 7 + 1/4 = 29/4. Total: 11 + 5/8 = 93/8.
    # Unknown should be 93/8 - 58/8 = 35/8 = 4 + 3/8 (4 EF)
    known = [Fraction(29, 4)]
    total = Fraction(93, 8)

    sol = solver.solve_missing_item(known, total, "TY_002", "item[1]")
    assert sol.is_valid_minoan is True
    assert sol.integer_part == 4
    assert sol.fraction_part == Fraction(3, 8)
    assert sol.fraction_symbols == "EF"
    assert "DIOPHANTINE EXACT PROOF" in sol.proof_certificate


def test_diophantine_benchmark_synthetic_masks():
    solver = DiophantineTabletSolver()
    res = solver.benchmark_synthetic_masks()

    assert res.tablets_tested >= 5
    assert res.total_masks >= 20
    assert res.accuracy_pct == 100.0
    assert res.exact_recoveries == res.total_masks
