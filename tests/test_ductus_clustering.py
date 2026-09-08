"""Tests for Unsupervised Scribal Hand Ductus Clustering & LM IB Disaster Horizon Engine."""

from linear_a.palaeography.ductus_clustering import (
    DuctusClusteringEngine,
    ScribalDuctusReport,
    ScribalDuctusVector,
    ScribalHandCluster,
)


def test_extract_tablet_ductus_vectors():
    engine = DuctusClusteringEngine()
    vectors = engine.extract_tablet_ductus_vectors()

    assert len(vectors) >= 12
    for v in vectors:
        assert isinstance(v, ScribalDuctusVector)
        assert len(v.feature_vector) == 5
        assert 0.0 <= v.feature_vector[0] <= 2.0  # Token length scale
        assert 0.0 <= v.feature_vector[1] <= 1.0  # Ligature propensity
        assert 0.0 <= v.feature_vector[2] <= 1.5  # Sign complexity
        assert 0.0 <= v.feature_vector[3] <= 1.0  # Affix density
        assert 0.0 <= v.feature_vector[4] <= 1.0  # Layout density


def test_cluster_scribal_hands():
    engine = DuctusClusteringEngine()
    clusters, silhouette = engine.cluster_scribal_hands(k=4)

    assert len(clusters) == 4
    assert silhouette > 0.30
    for c in clusters:
        assert isinstance(c, ScribalHandCluster)
        assert c.total_tablets > 0
        assert len(c.centroid_vector) == 5
        assert len(c.defining_traits) > 0


def test_cross_site_mobility_matches():
    engine = DuctusClusteringEngine()
    clusters, _ = engine.cluster_scribal_hands(k=4)
    matches = engine.evaluate_cross_site_mobility(clusters)

    assert len(matches) > 0
    for m in matches:
        assert m.source_site != "Hagia Triada"
        assert 0.0 <= m.palaeographic_similarity_pct <= 100.0
        assert len(m.epigraphic_rationale) > 0


def test_ductus_report_generation_and_to_dict():
    engine = DuctusClusteringEngine()
    report = engine.generate_ductus_report()

    assert isinstance(report, ScribalDuctusReport)
    assert report.total_tablets_profiled >= 12
    assert report.optimal_k_clusters == 4
    assert "Discovered 4 distinct latent scribal hands" in report.summary
    assert "EVIDENCE FOR REGIONAL SCRIPTORIUM INTEGRATION" in report.mobility_verdict

    rep_dict = report.to_dict()
    assert "total_tablets_profiled" in rep_dict
    assert "optimal_k_clusters" in rep_dict
    assert "mean_silhouette_score" in rep_dict
    assert "hands" in rep_dict
    assert len(rep_dict["hands"]) == 4
    assert "cross_site_mobility" in rep_dict
