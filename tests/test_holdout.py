"""Tests for Holdout Generalization and Predictive Validation Engine (LADP v1.0 Step 16)."""

import pytest
from linear_a.predictive.holdout_engine import HoldoutEngine


def test_holdout_accounting_generalization():
    engine = HoldoutEngine()
    summary = engine.evaluate_holdout_accounting()

    assert summary.total_holdout_tablets >= 10
    assert summary.tablets_with_stated_kuro >= 10
    assert summary.exact_kuro_prediction_accuracy_pct == 100.0
    assert summary.masked_reconstruction_accuracy_pct == 100.0
    assert summary.total_masked_entries_tested >= 23

    # Verify specific tablet predictions
    pred_map = {p.tablet_id: p for p in summary.detailed_predictions}
    assert "KH_007" in pred_map
    assert pred_map["KH_007"].predicted_kuro == "30"
    assert pred_map["KH_007"].is_exact_match is True

    assert "KN_001" in pred_map
    assert pred_map["KN_001"].predicted_kuro == "30"

    assert "MA_004" in pred_map
    assert pred_map["MA_004"].predicted_kuro == "11 5/8"

    assert "TY_002" in pred_map
    assert pred_map["TY_002"].predicted_kuro == "11 5/8"
    assert pred_map["TY_002"].is_exact_match is True


def test_holdout_morphology_evaluation():
    engine = HoldoutEngine()
    morph = engine.evaluate_holdout_morphology()

    assert morph.holdout_tokens_tested > 0
    assert morph.training_vocabulary_size > 0
    assert morph.morphological_generalization_score > 0.0


def test_holdout_regional_profiles():
    engine = HoldoutEngine()
    profiles = engine.evaluate_regional_variation()

    assert len(profiles) == 7
    site_names = [p.site for p in profiles]
    assert "Hagia_Triada" in site_names
    assert "Knossos" in site_names
    assert "Malia" in site_names
    assert "Khania" in site_names
    assert "Zakros" in site_names
    assert "Tylissos" in site_names


def test_full_holdout_suite():
    engine = HoldoutEngine()
    report = engine.run_full_holdout_suite()

    assert "LADP v1.0 Step 16 HOLDOUT PASS" in report.epistemic_verdict
    assert report.accounting_summary.exact_kuro_prediction_accuracy_pct == 100.0
