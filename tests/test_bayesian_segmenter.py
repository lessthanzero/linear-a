"""Unit tests for Unsupervised Morphological Sieve & Stem Alternation Engine."""

from linear_a.morphology.bayesian_segmenter import BayesianMorphologicalSegmenter


def test_morphological_segmenter_ingests_and_runs():
    segmenter = BayesianMorphologicalSegmenter()
    report = segmenter.run_induction()

    assert report.total_tokens_evaluated > 30
    assert report.unique_types > 20
    assert report.compression_ratio > 1.0
    assert len(report.top_prefixes) >= 3
    assert len(report.top_suffixes) >= 3


def test_votive_stem_alternations_detected():
    segmenter = BayesianMorphologicalSegmenter()
    report = segmenter.run_induction()

    # Find SA-SA-RA root
    sasara = next((a for a in report.stem_alternations if "SA-SA-RA" in a.stem or a.stem == "SA-SA-RA"), None)
    assert sasara is not None
    assert sasara.is_votive is True
    assert len(sasara.variants) >= 2

    # Check discovered prefixes include JA- and A-
    prefix_forms = {p.form for p in report.top_prefixes}
    assert "JA" in prefix_forms or "A" in prefix_forms


def test_votive_token_segmentations():
    segmenter = BayesianMorphologicalSegmenter()
    report = segmenter.run_induction()

    votive_tokens = {s.token for s in report.votive_segmentations}
    assert any("SA-SA-RA" in t for t in votive_tokens)
    assert any("NA-KA-NA-SI" in t for t in votive_tokens)
