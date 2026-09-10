"""Tests for Standalone Interactive Epigraphic Research Workbench Generator."""

from pathlib import Path

from linear_a.visualizer.workbench import generate_workbench_html, collect_workbench_dataset


def test_collect_workbench_dataset():
    data = collect_workbench_dataset(pages_safe=True)
    assert "tablets" in data
    assert len(data["tablets"]) >= 13
    assert "grid" in data
    assert data["grid"]["total_signs"] > 0
    assert "gauntlet" in data
    assert "semitic" in data["gauntlet"]
    assert "libation" in data
    assert "holdout" in data
    assert data["holdout"]["exact_kuro_acc"] == 100.0
    assert data["restoration_benchmark"]["scored_references"] == 0
    assert data["substratum_benchmark"]["qualified_entries"] == 0
    assert data["substratum_benchmark"]["empirical_p_value"] is None
    assert data["restoration_evaluation"]["phonotactic"]["precision_top1_pct"] < 20.0
    assert "phaistos" in data
    assert data["phaistos"]["firewall_intact"] is True
    assert "jury" in data
    assert len(data["jury"]) >= 2
    assert "lacunae" in data
    assert data["lacunae"]["total"] == 23
    assert data["lacunae"]["top1_accuracy"] < 90.0
    assert data["toponym_audit"]["status_counts"] == {"attested": 1, "disputed": 1}
    assert data["census_snapshot"]["metadata"]["availability"] == "omitted_for_pages"
    assert data["census_snapshot"]["entries"] == []
    first_syntax = data["interlinear"]["tablets"][0]["syntax"]
    assert first_syntax["genre_hypothesis"] == "ADMINISTRATIVE_LEDGER_PATTERN"
    assert "do not establish grammar" in first_syntax["limitations"]
    assert "morphology_induction" in data
    assert data["morphology_induction"]["compression_ratio"] > 1.0
    assert "diophantine_bench" in data
    assert data["diophantine_bench"]["accuracy_pct"] == 100.0
    assert "typological_profile" in data
    assert data["typological_profile"]["open_syllable_ratio"] > 0.9
    assert "votive_grammar" in data
    assert data["votive_grammar"]["total_vessels_parsed"] > 0
    assert "ligature_taxonomy" in data
    assert data["ligature_taxonomy"]["total_ligatures_cataloged"] > 0
    assert "substratum_induction" in data
    assert data["substratum_induction"]["total_substrate_entries"] >= 20
    assert data["substratum_induction"]["o_deficiency"]["linear_a_o_pct"] < 5.0
    assert "multi_commodity" in data
    assert data["multi_commodity"]["accuracy_pct"] == 100.0
    assert "ductus_clustering" in data
    assert data["ductus_clustering"]["optimal_k_clusters"] >= 2
    assert "adjudication_portal" in data
    assert data["adjudication_portal"]["total_candidate_packets"] > 0
    assert "geographical_dialectology" in data
    assert data["geographical_dialectology"]["total_sites_profiled"] >= 4
    assert "script_phylogeny" in data
    assert data["script_phylogeny"]["total_homologues_cataloged"] >= 15
    assert "unified_metrology" in data
    assert data["unified_metrology"]["base_weight_unit_grams"] == 61.0
    assert "stroke_vectors" in data
    assert data["stroke_vectors"]["total_glyphs_vectorized"] >= 8
    assert "recitations" in data
    assert len(data["recitations"]) >= 4
    assert all("gorila_ref" in r for r in data["recitations"])
    assert "phonetics_atlas" in data
    assert len(data["phonetics_atlas"]["profiles"]) >= 30


def test_generate_workbench_html(tmp_path):
    out_file = tmp_path / "linear_a_workbench.html"
    res_path = generate_workbench_html(str(out_file), pages_safe=True)

    assert res_path.exists()
    assert res_path.stat().st_size > 50_000

    content = res_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Linear A Epigraphic Inspection" in content
    assert "tab-tablets" in content
    assert "tab-grid" in content
    assert "tab-gauntlet" in content
    assert "tab-votive" in content
    assert "tab-holdout" in content
    assert "tab-phaistos" in content
    assert "tab-jury" in content
    assert "lacunaeTable" in content
    assert "Rule-Based Structural Pattern Hypothesis" in content
    assert "renderSyntaxPattern" in content
    assert "Leakage-Resistant Restoration Evaluation" in content
    assert "restorationEvaluationBody" in content
    assert "restorationBenchmarkStatus" in content
    assert "Published Linear A–Linear B Lexical Pair Benchmark" in content
    assert "Source-Linked Cretan Toponym Audit" in content
    assert "toponymAuditStatus" in content
    assert "tab-census" in content
    assert "censusSnapshotSummary" in content
    assert "Primary-edition locator unavailable" in content
    assert "PHAISTOS FIREWALL ACTIVE" in content
    assert "tab-morphology" in content
    assert "tab-diophantine" in content
    assert "tab-typology" in content
    assert "tab-votive-grammar" in content
    assert "renderMorphologyTab" in content
    assert "renderDiophantineTab" in content
    assert "renderTypologyTab" in content
    assert "renderVotiveGrammarTab" in content
    assert "tab-substratum" in content
    assert "tab-multi-commodity" in content
    assert "tab-ductus" in content
    assert "tab-adjudication-portal" in content
    assert "renderSubstratumInductionTab" in content
    assert "renderMultiCommodityTab" in content
    assert "renderDuctusClusteringTab" in content
    assert "renderAdjudicationPortalTab" in content
    assert "tab-dialectology" in content
    assert "tab-phylogeny" in content
    assert "tab-unified-metrology" in content
    assert "tab-stroke-vectors" in content
    assert "renderDialectologyTab" in content
    assert "renderPhylogenyTab" in content
    assert "renderUnifiedMetrologyTab" in content
    assert "renderStrokeVectorsTab" in content
    assert "tab-reciter" in content
    assert "tab-phonetics-atlas" in content
    assert "renderReciterTab" in content
    assert "renderPhoneticsAtlasTab" in content
    assert "synthesizeSyllableAudio" in content
    assert "reciter-word-block" in content
    assert "setPhoFilter" in content
    assert "Primary Critical Edition:" in content
