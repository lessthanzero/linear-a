"""Unit tests for Cross-Linguistic Typological Profiler."""

from linear_a.skeptic.typological_profiler import TypologicalProfiler


def test_typological_profiler_analysis():
    profiler = TypologicalProfiler()
    report = profiler.analyze_profile()

    assert report.total_words_analyzed > 20
    assert report.votive_mean_morae > report.admin_mean_morae
    assert report.vowel_distribution["A"] > report.vowel_distribution.get("O", 0.0)
    assert report.unigram_entropy_bits > 2.0
    assert len(report.distance_rankings) == 5
    assert report.best_matching_family in [
        "Hurro-Urartian / Hattic",
        "Etruscan / Tyrsenian",
    ]
    assert report.distance_rankings[0].compatibility_score > 50.0
