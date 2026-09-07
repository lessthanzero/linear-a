"""Tests for Linear A Interlinear Epigraphic Reader (LADP v1.0)."""

import pytest
from linear_a.corpus.loader import get_tablet_by_id
from linear_a.palaeography.ligatures import LigatureEngine
from linear_a.reading.interlinear import InterlinearReader
from linear_a.votive.libation_engine import LibationEngine


def test_interlinear_reader_administrative_tablet():
    tablet_ht9 = get_tablet_by_id("HT_009")
    assert tablet_ht9 is not None

    reader = InterlinearReader()
    doc = reader.parse_tablet(tablet_ht9)

    assert doc.id == "HT_009"
    assert doc.site == "Hagia_Triada"
    assert doc.genre == "ADMINISTRATIVE_LEDGER"
    assert doc.is_mathematically_balanced is True
    assert doc.stated_total == 31.0
    assert doc.calculated_total == 31.0
    assert len(doc.lines) == 6

    # Check the KU-RO line
    kuro_line = doc.lines[-1]
    assert kuro_line.line_type == "TOTAL"
    assert kuro_line.tokens[0].transliteration == "KU-RO"
    assert kuro_line.tokens[0].category == "TRANSACTION"
    assert kuro_line.tokens[0].epistemic_tier == "E3"

    # Check recipient line
    recip_line = doc.lines[0]
    assert recip_line.tokens[0].category == "ANTHROPONYM"
    assert recip_line.tokens[1].category == "COMMODITY"
    assert recip_line.tokens[1].transliteration == "FIC"
    assert recip_line.tokens[2].category == "NUMBER"
    assert recip_line.tokens[2].numerical_val == 3.0

    # Verify markdown generation
    md = doc.to_markdown()
    assert "# Interlinear Inscription: HT_009" in md
    assert "✓ EXACT BALANCE" in md
    assert "KU-RO" in md


def test_interlinear_reader_votive_vessel():
    lib_engine = LibationEngine()
    assert len(lib_engine.vessels) > 0
    juktas_vessel = next(v for v in lib_engine.vessels if v.id == "IO_Za_002")

    reader = InterlinearReader()
    doc = reader.parse_vessel(juktas_vessel)

    assert doc.id == "IO_Za_002"
    assert doc.genre == "VOTIVE_LIBATION"
    assert len(doc.lines) == 5

    # Check Great Goddess invocation
    jasasarame_line = doc.lines[1]
    assert jasasarame_line.tokens[0].transliteration == "JA-SA-SA-RA-ME"
    assert jasasarame_line.tokens[0].category == "DIVINE_EPITHET"
    assert jasasarame_line.tokens[0].epistemic_tier == "E4"

    # Check dedicatory verb
    verb_line = doc.lines[2]
    assert verb_line.tokens[0].transliteration == "U-NA-KA-NA-SI"
    assert verb_line.tokens[0].category == "DEDICATORY_VERB"

    # Check allative locative
    loc_line = doc.lines[4]
    assert loc_line.tokens[0].transliteration == "SI-RU-TE"
    assert loc_line.tokens[0].category == "SANCTUARY_LOCATIVE"
    assert loc_line.tokens[0].suffix == "-TE"


def test_ligature_engine():
    engine = LigatureEngine()
    assert len(engine.ligatures) >= 10

    # Test lookup
    lig = engine.get_ligature("OLE+U")
    assert lig is not None
    assert lig.base_commodity == "OLE"
    assert lig.modifier_reading == "U"
    assert lig.modifier_type == "syllabogram"

    # Test fraction ligature
    pithos = engine.get_ligature("*304+E")
    assert pithos is not None
    assert pithos.modifier_reading == "1/4"
    assert pithos.modifier_type == "fraction"

    # Test decomposition
    decomp = engine.decompose("GRA+PA")
    assert decomp is not None
    assert decomp["base_name"] == "Grain / Wheat"
    assert decomp["modifier"] == "PA"

    rep = engine.analyze_corpus()
    assert rep.total_ligatures_cataloged >= 10
    assert "OLE" in rep.commodity_distribution


def test_interlinear_reader_syme_and_ht085():
    reader = InterlinearReader()

    # Test reading Syme Sanctuary libation table SY_Za_001
    lib_engine = LibationEngine()
    syme = next(v for v in lib_engine.vessels if v.id == "SY_Za_001")
    doc_syme = reader.parse_vessel(syme)
    assert doc_syme.id == "SY_Za_001"
    assert doc_syme.site == "Syme"
    assert any(t.transliteration == "JA-DI-KI-TU" for line in doc_syme.lines for t in line.tokens)

    # Test reading HT_085 with ligatures and fractions
    ht85 = get_tablet_by_id("HT_085")
    assert ht85 is not None
    doc_ht85 = reader.parse_tablet(ht85)
    assert doc_ht85.is_mathematically_balanced is True
    assert doc_ht85.stated_total == 25.75
    assert doc_ht85.calculated_total == 25.75
    # Verify ligature classification in line 1 and 2
    assert doc_ht85.lines[0].tokens[1].category == "COMPOSITE_LIGATURE"
    assert doc_ht85.lines[1].tokens[1].category == "COMPOSITE_LIGATURE"

