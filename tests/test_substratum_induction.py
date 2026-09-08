"""Tests for Minoan Phonological Substratum & Script Adaptation Engine."""

from linear_a.phonology.substratum_induction import (
    KNOSSOS_SUBSTRATE_LEXICON,
    MinoanSubstratumInducer,
    SubstratumInductionReport,
)


def test_substratum_inducer_report_generation():
    inducer = MinoanSubstratumInducer()
    report = inducer.generate_substratum_report()

    assert isinstance(report, SubstratumInductionReport)
    assert report.total_substrate_entries == 25
    assert report.direct_homologies_count >= 7
    assert report.phonetic_cognates_count >= 10
    assert report.homology_rate_pct > 65.0
    assert "pre-Greek Cretan substrate" in report.summary


def test_vowel_o_deficiency_statistical_significance():
    inducer = MinoanSubstratumInducer()
    o_rep = inducer.analyze_vowel_o_deficiency()

    assert o_rep.linear_a_o_percentage < 5.0
    assert o_rep.linear_b_o_percentage == 22.0
    assert o_rep.z_score_vs_linear_b < -5.0
    assert o_rep.empirical_p_value < 1e-6
    assert "FALSIFIES MYCENAEAN" in o_rep.verdict


def test_knossos_substrate_lexicon_direct_homologies():
    homologies = {e.linear_b_form: e.linear_a_form for e in KNOSSOS_SUBSTRATE_LEXICON if e.preservation_status == "DIRECT_HOMOLOGY"}

    assert "pa-i-to" in homologies
    assert homologies["pa-i-to"] == "PA-I-TO"

    assert "a-mi-ni-so" in homologies
    assert homologies["a-mi-ni-so"] == "A-MI-NI-SO"

    assert "ku-do-ni-ja" in homologies
    assert homologies["ku-do-ni-ja"] == "KU-DO-NI-JA"

    assert "se-to-i-ja" in homologies
    assert homologies["se-to-i-ja"] == "SE-TO-I-JA"

    assert "da-wo" in homologies
    assert homologies["da-wo"] == "DA-WO"


def test_voicing_neutrality_metrics():
    inducer = MinoanSubstratumInducer()
    metrics = inducer.analyze_voicing_neutrality()

    assert len(metrics) == 3
    series_names = [m.series_name for m in metrics]
    assert any("DENTAL" in s for s in series_names)
    assert any("VELAR" in s for s in series_names)
    assert any("LABIAL" in s for s in series_names)

    for m in metrics:
        assert m.neutrality_ratio >= 0.80
        assert m.linear_a_distinct_series == 1


def test_substratum_report_to_dict():
    inducer = MinoanSubstratumInducer()
    rep_dict = inducer.generate_substratum_report().to_dict()

    assert "total_substrate_entries" in rep_dict
    assert "direct_homologies_count" in rep_dict
    assert "o_deficiency" in rep_dict
    assert "voicing_metrics" in rep_dict
    assert "entries" in rep_dict
    assert len(rep_dict["entries"]) == 25
    assert rep_dict["o_deficiency"]["z_score"] < 0
