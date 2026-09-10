"""Tests for the Linear A Libation Formula Engine."""

from linear_a.votive.libation_engine import LibationEngine


def test_libation_concordance():
    engine = LibationEngine()
    report = engine.evaluate_concordance()

    assert report.total_vessels == 5
    assert report.jasasarame_recurrence_rate >= 0.80
    assert report.unakanasi_recurrence_rate == 1.0
    assert report.phaistos_disc_liturgical_homology_score == 0.0
    assert "exploratory" in report.summary.lower()
    assert len(report.canonical_order) == 6
