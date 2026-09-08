"""Tests for Minoan Moraic Prosodic Scansion and Sacred Metric Cadence Engine."""

import pytest
from linear_a.phonology.prosodic_meter import (
    InscriptionMeterReport,
    MoraicSyllable,
    ProsodicMeterEngine,
    WordScansion,
)


def test_word_scansion_simple():
    engine = ProsodicMeterEngine()

    # Scanning JA-SA-SA-RA-ME
    res = engine.scan_word("JA-SA-SA-RA-ME")
    assert res.word == "JA-SA-SA-RA-ME"
    assert len(res.syllables) == 5
    # Terminal -ME is heavy enclitic
    assert res.syllables[-1].weight == 2
    assert res.syllables[-1].symbol == "¯"
    assert res.total_morae == 6
    assert "˘ ˘ ˘ ˘ ¯" in res.scansion_str


def test_diphthong_and_complex_scansion():
    engine = ProsodicMeterEngine()

    # A-TA-I-*301-WA-JA: A + TA + (I) + *301 (heavy complex) + WA + JA (terminal heavy)
    res = engine.scan_word("A-TA-I-*301-WA-JA")
    assert res.total_morae >= 6
    # Contains heavy *301 symbol
    symbols = [s.symbol for s in res.syllables]
    assert "¯" in symbols


def test_inscription_scansion():
    engine = ProsodicMeterEngine()

    text = "A-TA-I-*301-WA-JA JA-SA-SA-RA-ME U-NA-KA-NA-SI"
    report = engine.scan_inscription(text, id="IO_Za_2", site="Mount Juktas", carrier="Stone Ladle")

    assert report.id == "IO_Za_2"
    assert len(report.words) == 3
    assert report.total_morae > 10
    assert report.regularity_score > 0.0
    assert report.dominant_foot in ("DACTYLIC", "TROCHAIC", "AMPHIBRACHIC", "ISOCHRONIC")


def test_corpus_wide_cadence_analysis():
    engine = ProsodicMeterEngine()
    analysis = engine.analyze_votive_corpus()

    assert analysis["vessel_count"] > 0
    assert analysis["mean_morae_per_vessel"] > 0
    assert "meter_distribution" in analysis
    assert len(analysis["vessel_reports"]) == analysis["vessel_count"]
