"""Tests for Minoan Inscription Reciter Engine."""

import pytest
from linear_a.reading.reciter_engine import (
    FLAGSHIP_INSCRIPTIONS,
    RecitationPackage,
    ReciterEngine,
)


def test_prepare_recitation_votive():
    engine = ReciterEngine()

    pkg = engine.prepare_recitation(
        id_str="IO_Za_2",
        raw_text="A-TA-I-*301-WA-JA JA-SA-SA-RA-ME U-NA-KA-NA-SI",
        site="Mount Juktas",
        carrier="Stone Ladle",
        genre="VOTIVE_LIBATION",
    )

    assert pkg.id == "IO_Za_2"
    assert pkg.genre == "VOTIVE_LIBATION"
    assert "ja" in pkg.ipa_text
    assert pkg.total_morae > 10
    assert len(pkg.syllables) > 10
    assert len(pkg.synthesis_schedule) == len(pkg.syllables)
    assert 0.0 < pkg.mean_confidence <= 1.0


def test_get_recitation_by_id_curated():
    engine = ReciterEngine()

    # Query with variations
    pkg1 = engine.get_recitation_by_id("IO_Za_2")
    assert pkg1 is not None
    assert pkg1.id == "IO_Za_2"

    pkg2 = engine.get_recitation_by_id("HT 85")
    assert pkg2 is not None
    assert "HT_085" in pkg2.id or pkg2.id == "HT_085"

    pkg3 = engine.get_recitation_by_id("PK_Za_11")
    assert pkg3 is not None
    assert "A-SA-SA-RA-ME" in pkg3.raw_text


def test_get_curated_recitations():
    engine = ReciterEngine()
    curated = engine.get_curated_recitations()

    assert len(curated) == len(FLAGSHIP_INSCRIPTIONS)
    assert any(c.genre == "VOTIVE_LIBATION" for c in curated)
    assert any(c.genre == "ADMINISTRATIVE_LEDGER" for c in curated)
    for c in curated:
        assert len(c.syllables) > 0
        assert len(c.synthesis_schedule) > 0
        assert c.gorila_ref != ""
        assert "GORILA" in c.gorila_ref
        assert c.dating != ""
        d = c.to_dict()
        assert "gorila_ref" in d
        assert "dating" in d

