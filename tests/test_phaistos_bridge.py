"""Tests for Phaistos Disc Cross-Script Structural Bridge & Epigraphic Firewall."""

import pytest
from linear_a.bridge.phaistos_matrix import PhaistosBridgeEngine


def test_phaistos_firewall_integrity():
    engine = PhaistosBridgeEngine()
    fw = engine.verify_firewall_integrity()

    assert fw.is_firewall_intact is True
    assert fw.sound_leak_detected is False
    assert fw.evidence_layer_violation is False
    assert "FIREWALL VERIFIED" in fw.rationale


def test_cross_script_homology_matrix():
    engine = PhaistosBridgeEngine()
    report = engine.evaluate_cross_script_homology()

    assert report.firewall.is_firewall_intact is True
    assert report.overall_structural_homology_score_pct >= 90.0
    assert len(report.correspondences) == 4
    assert report.te_vs_me_likelihood_ratio > 1e6
    assert report.plumed_head_prefix_correlation > 0.85
    assert report.clause_cadence_homology_pct > 85.0

    # Ensure all correspondences are marked firewall_compliant
    for c in report.correspondences:
        assert c.firewall_compliant is True
        assert c.concordance_level == "HIGH_HOMOLOGY"
