"""Tests for Phaistos Disc Cross-Script Structural Bridge & Epigraphic Firewall."""

from linear_a.bridge.phaistos_matrix import PhaistosBridgeEngine


def test_phaistos_firewall_integrity():
    engine = PhaistosBridgeEngine()
    fw = engine.verify_firewall_integrity()

    assert fw.sound_leak_detected is False
    assert fw.evidence_layer_violation is False
    assert "FIREWALL POLICY" in fw.rationale


def test_cross_script_homology_matrix_is_exploratory():
    engine = PhaistosBridgeEngine()
    report = engine.evaluate_cross_script_homology()

    assert len(report.correspondences) == 4
    assert report.overall_structural_homology_score_pct == 0.0
    assert report.plumed_head_prefix_correlation == 0.0
    assert "EXPLORATORY" in report.epistemic_verdict
    assert "confirmation" in report.epistemic_verdict.lower() or "CONFIRMATION" in report.epistemic_verdict

    for c in report.correspondences:
        assert c.firewall_compliant is True
        assert c.concordance_level in {"EXPLORATORY", "EQUIVOCAL", "STRUCTURAL_ANALOGY"}
        assert c.concordance_level != "HIGH_HOMOLOGY"
