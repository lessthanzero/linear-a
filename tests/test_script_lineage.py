"""Tests for Aegean Bronze Age Script Phylogeny Engine."""

from linear_a.phylogeny.script_lineage import AegeanScriptPhylogenyEngine


def test_script_phylogeny_engine():
    engine = AegeanScriptPhylogenyEngine()
    rep = engine.generate_phylogeny_report()

    assert rep.total_homologues_cataloged >= 15
    assert len(rep.scripts_profiled) == 4

    # Transmission retention
    assert rep.chic_to_linear_a_retention_pct > 70.0
    assert rep.linear_a_to_linear_b_retention_pct > 70.0
    assert 0.0 < rep.linear_a_to_cypro_minoan_retention_pct < 100.0

    # Stroke reduction from pictorial hieroglyphs to linear cursive
    assert rep.mean_stroke_reduction_chic_to_la_pct > 30.0

    # Entropy delta is bounded
    assert abs(rep.entropy_delta_chic_to_la) < 1.0

    # Serialization
    d = rep.to_dict()
    assert "total_homologues_cataloged" in d
    assert "scripts" in d
    assert "homologues" in d
    assert len(d["homologues"]) == rep.total_homologues_cataloged
    assert len(d["scripts"]) == 4


def test_script_lineage_chronology_and_parentage():
    engine = AegeanScriptPhylogenyEngine()
    rep = engine.generate_phylogeny_report()

    script_map = {s.script_id: s for s in rep.scripts_profiled}
    assert script_map["CHIC"].parent_script is None
    assert script_map["LINEAR_A"].parent_script == "Cretan Hieroglyphic"
    assert script_map["LINEAR_B"].parent_script == "Linear A"
    assert script_map["CYPRO_MINOAN"].parent_script == "Linear A"


def test_homologue_evidence_integrity():
    engine = AegeanScriptPhylogenyEngine()
    for h in engine.homologues:
        assert h.linear_a_id.startswith("A") or h.linear_a_id.startswith("AB")
        assert h.confidence_tier.startswith("E")
        assert h.chic_stroke_count >= h.linear_a_stroke_count
