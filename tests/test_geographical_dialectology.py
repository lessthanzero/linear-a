"""Tests for Geographical Dialectology and Spatial Distance Matrix."""

from linear_a.dialect.geographical_dialectology import (
    GeographicalDialectologyEngine,
    haversine_distance_km,
)


def test_haversine_distance_known_points():
    # Knossos to Phaistos distance is approx 45-55 km
    kn_lat, kn_lon = 35.2981, 25.1633
    ph_lat, ph_lon = 35.0514, 24.8142
    dist = haversine_distance_km(kn_lat, kn_lon, ph_lat, ph_lon)
    assert 40.0 < dist < 60.0


def test_geographical_dialectology_engine_generation():
    engine = GeographicalDialectologyEngine()
    rep = engine.generate_dialectology_report(min_words_threshold=2, permutations=200)

    assert rep.total_sites_profiled >= 4
    assert rep.pairwise_comparisons_count > 0
    assert rep.mean_geographic_distance_km > 10.0
    assert 0.0 <= rep.mean_jaccard_dissimilarity <= 1.0

    # Mantel test output
    assert -1.0 <= rep.mantel_test.correlation_r <= 1.0
    assert 0.0 <= rep.mantel_test.p_value <= 1.0
    assert rep.mantel_test.permutations_count == 200
    assert rep.mantel_test.epistemic_verdict != ""

    # Check to_dict serialization
    d = rep.to_dict()
    assert "total_sites_profiled" in d
    assert "mantel_test" in d
    assert "site_profiles" in d
    assert "pairwise_comparisons" in d
    assert len(d["site_profiles"]) == rep.total_sites_profiled


def test_distance_matrix_symmetry():
    engine = GeographicalDialectologyEngine()
    rep = engine.generate_dialectology_report(min_words_threshold=2, permutations=50)

    # Check pairwise comparisons have symmetric distances
    dist_map = {}
    for pc in rep.pairwise_comparisons:
        dist_map[(pc.site_a, pc.site_b)] = pc.geographic_distance_km
        dist_map[(pc.site_b, pc.site_a)] = pc.geographic_distance_km

    for (s1, s2), d12 in dist_map.items():
        assert dist_map[(s2, s1)] == d12
        assert d12 >= 0.0


def test_mantel_test_seed_stability():
    engine = GeographicalDialectologyEngine()
    rep1 = engine.generate_dialectology_report(min_words_threshold=2, permutations=100, random_seed=99)
    rep2 = engine.generate_dialectology_report(min_words_threshold=2, permutations=100, random_seed=99)

    assert rep1.mantel_test.correlation_r == rep2.mantel_test.correlation_r
    assert rep1.mantel_test.p_value == rep2.mantel_test.p_value
