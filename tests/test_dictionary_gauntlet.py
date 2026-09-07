"""Tests for the Cross-Linguistic Dictionary Gauntlet."""

from linear_a.skeptic.dictionary_gauntlet import (
    DictionaryGauntlet,
    load_candidate_lexicon,
)


def test_load_candidate_lexicons():
    semitic = load_candidate_lexicon("semitic_northwest")
    luwian = load_candidate_lexicon("anatolian_luwian")
    null_lex = load_candidate_lexicon("synthetic_null")

    assert len(semitic) >= 5
    assert len(luwian) >= 5
    assert len(null_lex) >= 10

    assert any(r["root"] == "kull" for r in semitic)
    assert any(r["root"] == "ashasara" for r in luwian)


def test_semitic_gauntlet_evaluation():
    gauntlet = DictionaryGauntlet()
    report = gauntlet.run_gauntlet("semitic_northwest", n_surrogates=200, seed=42)

    assert report.total_candidate_roots >= 5
    assert report.observed_matches_count >= 1
    # Check that KU-RO or KI-RO matches were detected
    matched_tokens = [m.target_token for m in report.top_matches]
    assert "KU-RO" in matched_tokens or "KI-RO" in matched_tokens

    # Verify unicity ratio and statistical bounds
    assert report.estimated_degrees_of_freedom_bits > 0.0
    assert report.unicity_ratio > 0.0
    assert report.empirical_p_value <= 1.0


def test_luwian_gauntlet_evaluation():
    gauntlet = DictionaryGauntlet()
    report = gauntlet.run_gauntlet("anatolian_luwian", n_surrogates=200, seed=42)

    assert report.total_candidate_roots >= 5
    assert report.observed_matches_count >= 1
    matched_roots = [m.root for m in report.top_matches]
    assert "ashasara" in matched_roots


def test_synthetic_null_gauntlet():
    gauntlet = DictionaryGauntlet()
    report = gauntlet.run_gauntlet("synthetic_null", n_surrogates=200, seed=42)

    # Synthetic random roots must not yield statistically significant decipherment
    assert report.is_statistically_significant is False
