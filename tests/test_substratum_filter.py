"""Tests for the publication-linked Linear A / Linear B benchmark."""

from pathlib import Path

import pytest
import yaml

from linear_a.skeptic.substratum_filter import (
    CorrespondenceBenchmarkError,
    PublishedPairBenchmark,
    normalize_syllabic_form,
)


def write_benchmark(path: Path, entries: list[dict]) -> Path:
    path.write_text(yaml.safe_dump({"metadata": {"id": "fixture", "target_entries": 2}, "entries": entries}), encoding="utf-8")
    return path


def valid_entry() -> dict:
    return {
        "id": "PAIR-001",
        "linear_a_form": "KU-PA-RO",
        "linear_b_form": "ku-pa-ro2",
        "lexical_category": "plant",
        "source_status": "uncertain",
        "pair_citation": {"author": "Example", "year": "2026", "title": "Example", "locator": "p. 1"},
        "linear_b_attestations": ["PY Un 249"],
        "limitation": "Published comparison only; no cognacy inference.",
    }


def test_default_benchmark_discloses_qualified_set_shortfall():
    report = PublishedPairBenchmark().run()
    assert report.qualified_entries == 0
    assert report.qualifying_shortfall == 12
    assert report.null_mean_matches is None
    assert report.empirical_p_value is None


def test_normalization_removes_transcription_indices():
    assert normalize_syllabic_form("ku-pa-ro2") == "KU-PA-RO"
    with pytest.raises(CorrespondenceBenchmarkError):
        normalize_syllabic_form("KU-? -RO")


def test_benchmark_rejects_nonlexical_and_uncited_entries(tmp_path: Path):
    entry = valid_entry()
    entry["lexical_category"] = "place_name"
    with pytest.raises(CorrespondenceBenchmarkError, match="non-proper lexical"):
        PublishedPairBenchmark(benchmark_path=write_benchmark(tmp_path / "bad.yaml", [entry]))

    entry = valid_entry()
    entry["pair_citation"] = {"author": "Example"}
    with pytest.raises(CorrespondenceBenchmarkError, match="pair_citation"):
        PublishedPairBenchmark(benchmark_path=write_benchmark(tmp_path / "uncited.yaml", [entry]))


def test_frequency_preserving_surrogates_are_seeded_and_length_preserving(tmp_path: Path):
    benchmark = PublishedPairBenchmark(benchmark_path=write_benchmark(tmp_path / "ok.yaml", [valid_entry()]))
    corpus = ["KU-PA-RO", "SA-RU", "KU-RO", "PA-I-TO"]
    first = benchmark._surrogate_matches(corpus, benchmark.entries, surrogates=20, seed=7)
    second = benchmark._surrogate_matches(corpus, benchmark.entries, surrogates=20, seed=7)
    assert first == second
    assert len(first) == 20
    assert all(value in (0, 1) for value in first)


def test_substratum_cli_reports_shortfall_and_exports(tmp_path: Path):
    from typer.testing import CliRunner
    from linear_a.cli.main import app

    path = tmp_path / "report.json"
    result = CliRunner().invoke(app, ["substratum", "--export", str(path)])
    assert result.exit_code == 0, result.stdout
    assert "No score was run" in result.stdout
    assert '"qualified_entries": 0' in path.read_text(encoding="utf-8")
