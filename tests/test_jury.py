"""Tests for the Tripartite Blind Skeptic Jury."""

from linear_a.llm.jury import SkepticJury


def test_skeptic_jury_structure():
    jury = SkepticJury()
    dossier = jury.evaluate_claim(
        claim="Linear A is archaic Hebrew where KU-RO means 'all' and KI-RO means 'missing'",
        claim_domain="phonetic_translation",
        structural_fit=2.0,
        corpus_coverage=1.0,
        arbitrary_assumptions=4.0,
        ad_hoc_rules=3.0,
    )

    assert len(dossier.juror_critiques) == 3
    assert dossier.quantitative_score_s >= 0.0
    assert dossier.quantitative_score_s <= 100.0
    assert dossier.epistemic_grade in ["SPECULATIVE / FALSIFIED", "WEAK", "PLAUSIBLE", "STRONG", "VERY STRONG (Requires Independent Replication)"]
    assert "Tripartite jury evaluated claim" in dossier.synthesis_summary


def test_skeptic_jury_scoring_bounds():
    jury = SkepticJury()
    # Test heavily overfit claim with max penalties
    dossier_fail = jury.evaluate_claim(
        claim="Arbitrary reading of HT 13 as a Luwian royal decree",
        arbitrary_assumptions=8.0,
        exceptions=6.0,
        ad_hoc_rules=5.0,
    )
    assert dossier_fail.quantitative_score_s < 30.0
    assert "FALSIFIED" in dossier_fail.epistemic_grade or "WEAK" in dossier_fail.epistemic_grade
