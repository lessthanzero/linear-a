"""Tests for Minoan Acoustic Formant Modeling and Ventris-Grid Phonetic Transfer Engine."""

import pytest
from linear_a.phonology.acoustic_reconstruction import (
    CANONICAL_SIGN_PROFILES,
    AcousticReconstructionEngine,
    PhoneticSignProfile,
)


def test_sign_lookup_and_profiles():
    engine = AcousticReconstructionEngine()

    # Core vowel lookup
    a_prof = engine.get_sign_profile("A")
    assert a_prof is not None
    assert a_prof.sign_code == "AB08"
    assert a_prof.confidence_tier == "E4"
    assert a_prof.formant_f1 == 750.0

    # Stop with voicing neutrality
    da_prof = engine.get_sign_profile("DA")
    assert da_prof is not None
    assert da_prof.sign_code == "AB01"
    assert da_prof.voicing_neutral is True
    assert da_prof.consonant_burst_freq == 3500.0

    # Lookup by sign code
    by_code = engine.get_sign_profile("AB31")
    assert by_code is not None
    assert by_code.transliteration == "SA"
    assert by_code.consonant_manner == "SIBILANT"

    # Asterisk sign
    a301 = engine.get_sign_profile("*301")
    assert a301 is not None
    assert a301.confidence_tier == "E0"
    assert a301.confidence_score < 0.30
    assert a301.consonant_place == "DENTAL"
    assert "Melena" in a301.notes



def test_ipa_transcription():
    engine = AcousticReconstructionEngine()

    # Sacred libation epithet JA-SA-SA-RA-ME
    ipa, profiles = engine.transcribe_to_ipa("JA-SA-SA-RA-ME")
    assert len(profiles) == 5
    assert "ja" in ipa
    assert "sa" in ipa
    assert "me" in ipa
    assert all(p.confidence_tier in ("E4", "E3") for p in profiles)

    # Votive opening formula with asterisk sign
    ipa_votive, profiles_votive = engine.transcribe_to_ipa("A-TA-I-*301-WA-JA")
    assert len(profiles_votive) == 6
    assert profiles_votive[3].sign_code == "A301"
    assert profiles_votive[3].confidence_tier == "E0"


def test_synthesis_schedule_generation():
    engine = AcousticReconstructionEngine()

    # Generate schedule for accounting term KU-RO
    schedule = engine.generate_synthesis_schedule("KU-RO", base_pitch_hz=140.0, tempo_scale=1.0)
    assert len(schedule) == 2

    first = schedule[0]
    second = schedule[1]

    assert first.syllable == "KU"
    assert second.syllable == "RO"
    assert first.duration_ms > 0
    assert second.start_time_ms > first.start_time_ms
    assert first.vowel_gain > 0.5
    assert second.duration_ms > first.duration_ms  # Pre-pausal lengthening on final


def test_formant_space_summary():
    engine = AcousticReconstructionEngine()
    summary = engine.get_formant_space_summary()

    assert summary["vowel_count"] >= 4
    vowel_signs = {v["sign"] for v in summary["vowels"]}
    assert {"A", "E", "I", "U"}.issubset(vowel_signs)
    assert summary["f1_range"] == [250.0, 850.0]
