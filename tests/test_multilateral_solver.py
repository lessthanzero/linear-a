"""Tests for the Multilateral Joint Bayesian Lacunae Solver."""

import pytest
from linear_a.predictive.multilateral_solver import MultilateralLacunaeSolver


def test_multilateral_solver_catalog_loading():
    solver = MultilateralLacunaeSolver()
    assert len(solver.catalog) >= 20
    assert any(e.genre == "votive" for e in solver.catalog)
    assert any(e.genre == "administrative" for e in solver.catalog)
    assert any(e.genre == "toponymic" for e in solver.catalog)
    assert any(e.genre == "arithmetic" for e in solver.catalog)


def test_solve_votive_liturgical_lacunae():
    solver = MultilateralLacunaeSolver()
    pk12 = next(e for e in solver.catalog if e.id == "LAC-VOT-01")
    res_pk12 = solver.solve_entry(pk12)

    assert res_pk12.is_exact_match is True
    assert res_pk12.predicted_sign == "U"
    assert res_pk12.epistemic_grade == "HIGH_CONFIDENCE_E4"
    assert "U-NA-KA-NA-SI" in res_pk12.entry.completed_word
    assert res_pk12.bayes_factor > 1000.0


def test_solve_prosopographical_lacunae():
    solver = MultilateralLacunaeSolver()
    ht85_kuro = next(e for e in solver.catalog if e.id == "LAC-PRO-01")
    res = solver.solve_entry(ht85_kuro)

    assert res.is_exact_match is True
    assert res.predicted_sign == "PA"
    assert "KU-PA-NU" in res.entry.completed_word
    assert res.epistemic_grade == "HIGH_CONFIDENCE_E4"


def test_solve_toponymic_lacunae():
    solver = MultilateralLacunaeSolver()
    paito = next(e for e in solver.catalog if e.id == "LAC-TOP-01")
    res_paito = solver.solve_entry(paito)

    assert res_paito.is_exact_match is True
    assert res_paito.predicted_sign == "I"
    assert res_paito.entry.completed_word == "PA-I-TO"
    assert res_paito.bayes_factor > 100.0


def test_solve_diophantine_arithmetic_lacunae():
    solver = MultilateralLacunaeSolver()
    ki15 = next(e for e in solver.catalog if e.id == "LAC-NUM-02")
    res_ki15 = solver.solve_entry(ki15)

    assert res_ki15.is_exact_match is True
    assert res_ki15.predicted_sign == "15"
    assert res_ki15.epistemic_grade == "DETERMINISTIC_E3"
    assert res_ki15.posterior_confidence == 1.0


def test_multilateral_solver_full_report():
    solver = MultilateralLacunaeSolver()
    report = solver.solve_all()

    assert report.total_lacunae_analyzed >= 20
    assert report.top1_accuracy_rate >= 90.0
    assert report.mean_confidence >= 0.90
    assert report.deterministic_arithmetic_count >= 4
    assert report.liturgical_votive_count >= 4
    assert report.prosopographical_count >= 8
    assert report.toponymic_count >= 4
    assert "Overall Top-1 Reconstruction Accuracy" in report.summary
