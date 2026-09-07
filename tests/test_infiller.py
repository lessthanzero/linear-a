"""Tests for Linear A Masked Phonotactic Lacunae Infilling Engine."""

import pytest
from linear_a.predictive.lacunae_infiller import LacunaeInfiller


def test_infill_known_anthroponym():
    infiller = LacunaeInfiller()
    res = infiller.infill_token("KU-?-NU")

    assert res.masked_position == 1
    assert len(res.top_candidates) >= 1
    assert res.best_candidate.reading == "PA"
    assert res.is_exact_lexical_recovery is True
    assert res.best_candidate.lexical_match == "KU-PA-NU"
    assert res.best_candidate.confidence_tier == "E4"


def test_infill_libation_formula():
    infiller = LacunaeInfiller()
    res = infiller.infill_token("JA-SA-?-RA-ME")

    assert res.masked_position == 2
    assert res.best_candidate.reading == "SA"
    assert res.is_exact_lexical_recovery is True
    assert res.best_candidate.lexical_match == "JA-SA-SA-RA-ME"


def test_infill_ht_085_and_ht_117_lacunae():
    infiller = LacunaeInfiller()

    # HT 085 line 3: DA-?-RE -> DA-TA-RE
    res_datare = infiller.infill_token("DA-?-RE")
    assert res_datare.masked_position == 1
    assert res_datare.best_candidate.reading == "TA"
    assert res_datare.best_candidate.lexical_match == "DA-TA-RE"
    assert res_datare.best_candidate.confidence_tier == "E4"

    # HT 117 line 2: TE-? -> TE-TU
    res_tetu = infiller.infill_token("TE-?")
    assert res_tetu.masked_position == 1
    assert res_tetu.best_candidate.reading == "TU"
    assert res_tetu.best_candidate.lexical_match == "TE-TU"
    assert res_tetu.best_candidate.confidence_tier == "E4"



def test_benchmark_reconstruction():
    infiller = LacunaeInfiller()
    bench = infiller.benchmark_reconstruction_accuracy(sample_size=15)

    assert bench["tokens_tested"] > 0
    assert bench["top1_accuracy_pct"] >= 70.0
    assert bench["top3_accuracy_pct"] >= 90.0
