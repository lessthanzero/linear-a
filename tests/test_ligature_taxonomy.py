"""Unit tests for Ligature Taxonomy and Scriptorium Repertoire Engine."""

from linear_a.palaeography.ligature_taxonomy import LigatureTaxonomyEngine


def test_ligature_taxonomy_runs_and_decomposes():
    engine = LigatureTaxonomyEngine()
    report = engine.generate_taxonomy_report()

    assert report.total_ligatures_cataloged >= 5
    assert report.total_instances_attested > 20
    assert "OLE" in report.commodity_classes or "GRA" in report.commodity_classes
    assert len(report.linear_b_concordances) >= 3


def test_linear_b_parallels_identified():
    engine = LigatureTaxonomyEngine()
    report = engine.generate_taxonomy_report()

    ole_u = next((d for d in report.decomposed_records if d.notation == "OLE+U"), None)
    assert ole_u is not None
    assert ole_u.modifier_role == "QUALITY_GRADE"
    assert "Linear B" in str(ole_u.linear_b_parallel)

    gra_pa = next((d for d in report.decomposed_records if d.notation == "GRA+PA"), None)
    assert gra_pa is not None
    assert gra_pa.modifier_role == "SPECIES_VARIANT"
    assert "Linear B" in str(gra_pa.linear_b_parallel)


def test_scriptorium_profiles_aggregated():
    engine = LigatureTaxonomyEngine()
    report = engine.generate_taxonomy_report()

    assert len(report.scriptorium_profiles) >= 3
    site_names = {p.site_name for p in report.scriptorium_profiles}
    assert any("Hagia Triada" in s for s in site_names)
