"""Unit tests for Corpus-Wide Epigraphic Lacunae Census Engine (Horizon 1)."""

import pytest
from linear_a.predictive.corpus_census import CorpusLacunaeCensusEngine, CorpusSourceUnavailable


def test_census_runs_and_scans_documents():
    engine = CorpusLacunaeCensusEngine()
    report = engine.run_census()

    assert report.total_inscriptions_scanned >= 1500
    assert report.total_tokens_scanned >= 5000
    assert report.total_damaged_tokens >= 2000
    assert report.total_recoverable_count >= 60
    assert report.information_recoverability_pct < 5.0
    assert report.accounting_context_count > 500
    assert report.source_snapshot["verified"] is True


def test_census_tiers_partition_correctly():
    engine = CorpusLacunaeCensusEngine()
    report = engine.run_census()

    calculated_sum = (
        report.accounting_context_count
        + report.deterministic_e3_count
        + report.sacred_liturgy_e4_count
        + report.attested_template_e4_count
        + report.phonotactic_e2_count
        + report.irrecoverable_e0_count
    )
    assert calculated_sum == report.total_damaged_tokens
    assert report.total_recoverable_count == (
        report.deterministic_e3_count
        + report.sacred_liturgy_e4_count
        + report.attested_template_e4_count
    )
    open_contexts = [entry for entry in report.census_entries if entry.recoverability_tier == "OPEN_PHONOTACTIC_E2"]
    assert open_contexts
    assert all(entry.suggested_infill is None and entry.completed_word is None for entry in open_contexts)
    assert all(entry.bayes_factor == 1.0 and entry.epistemic_confidence == 0.0 for entry in open_contexts)


def test_site_and_carrier_breakdowns():
    engine = CorpusLacunaeCensusEngine()
    report = engine.run_census()

    assert "Haghia Triada" in report.site_breakdown
    assert "Khania" in report.site_breakdown
    assert "Phaistos" in report.site_breakdown

    ht_stats = report.site_breakdown["Haghia Triada"]
    assert ht_stats["damaged_tokens"] > 500
    assert ht_stats["recoverable"] >= 20

    assert len(report.carrier_breakdown) >= 3


def test_census_yaml_export(tmp_path):
    engine = CorpusLacunaeCensusEngine()
    report = engine.run_census()

    out_file = tmp_path / "test_census.yaml"
    res_path = engine.save_census_yaml(report, out_file)

    assert res_path.exists()
    assert res_path.stat().st_size > 10_000
    text = res_path.read_text(encoding="utf-8")
    assert "suggested_infill_hypothesis" in text
    assert "Open E2 phonotactic contexts carry no sign proposal" in text
    assert "primary_locator_available" in text
    assert all(entry.primary_locator_available is False for entry in report.census_entries)


def test_census_requires_the_tracked_local_source_snapshot(tmp_path):
    engine = CorpusLacunaeCensusEngine(corpus_dir=tmp_path)
    with pytest.raises(CorpusSourceUnavailable, match="manifest"):
        engine.run_census()


def test_census_rejects_a_source_that_does_not_match_the_manifest(tmp_path):
    raw = tmp_path / "raw"
    paleo = tmp_path / "palaeography"
    raw.mkdir()
    paleo.mkdir()
    (raw / "annotations.js").write_text("var wordAnnotations = [];", encoding="utf-8")
    (raw / "LinearAInscriptions.js").write_text("var inscriptions = new Map([]);", encoding="utf-8")
    (paleo / "corpus_source_manifest.yaml").write_text(
        "files:\n  annotations.js:\n    bytes: 1\n    sha256: wrong\n  LinearAInscriptions.js:\n    bytes: 1\n    sha256: wrong\n",
        encoding="utf-8",
    )

    with pytest.raises(CorpusSourceUnavailable, match="does not match"):
        CorpusLacunaeCensusEngine(corpus_dir=tmp_path).run_census()
