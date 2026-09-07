"""Tests for the Linear A Morphological & Suffix Sieve."""

from linear_a.morphology.affix_sieve import AffixSieve


def test_affix_sieve_distributions():
    sieve = AffixSieve()
    report = sieve.evaluate_affixes()

    assert report.total_terminal_tokens_analyzed == 1427
    assert len(report.top_suffixes) >= 6
    assert len(report.top_prefixes) >= 4

    # Top terminal sign must be A (AB08) and top directive suffix must be TE (AB04)
    assert report.top_suffixes[0].glyph_id == "AB08"
    assert report.top_suffixes[1].glyph_id == "AB04"

    # Likelihood ratio for TE vs ME must strongly favor TE
    assert report.te_vs_me_likelihood_ratio > 1e7
    assert report.phaistos_disc_bridge_te_match is True
