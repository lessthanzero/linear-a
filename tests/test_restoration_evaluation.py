"""Tests for leakage-resistant restoration evaluation."""

import pytest
from typer.testing import CliRunner

from linear_a.cli.main import app
from linear_a.predictive.restoration_evaluation import RestorationEvaluationEngine
from linear_a.predictive.source_linked_benchmark import SourceLinkedBenchmark


def test_source_linked_benchmark_excludes_all_unadjudicated_entries():
    report = RestorationEvaluationEngine().evaluate()

    assert report.benchmark_status == "citation-only bootstrap"
    assert report.total_entries == 23
    assert report.status_counts["unadjudicated"] == 23
    assert report.scored_references == 0
    assert report.template.references == 0
    assert "excluded from every accuracy denominator" in report.limitations


def test_adjudicated_entries_require_reviewer_decision_metadata():
    benchmark = SourceLinkedBenchmark()
    entry = benchmark.entries[0]
    entry.evidence_status = "accepted"
    with pytest.raises(ValueError, match="requires reviewers"):
        benchmark.validate()

    entry.adjudication = {
        "reviewers": ["reviewer-1", "reviewer-2"],
        "decision": "accepted",
        "decision_at": "2026-09-08",
    }
    benchmark.validate()


def test_review_handoff_has_no_preassigned_decisions():
    handoff = SourceLinkedBenchmark().review_handoff()

    assert len(handoff["records"]) == 23
    assert all(record["decision"] is None for record in handoff["records"])


def test_exploratory_evaluation_is_separate_from_source_linked_scoring():
    report = RestorationEvaluationEngine().evaluate_exploratory()

    assert report.reference_set_status == "project-curated reference set"
    assert report.template.references == 18
    assert report.template.abstained > 0
    assert report.phonotactic.attempted <= report.phonotactic.references
    assert "excluded from its template" in report.limitations


def test_evaluation_cli_exports_benchmark_and_review_handoff(tmp_path):
    report_path = tmp_path / "benchmark.json"
    handoff_path = tmp_path / "handoff.json"
    result = CliRunner().invoke(
        app,
        ["evaluate-restorations", "--export", str(report_path), "--review-handoff", str(handoff_path)],
    )

    assert result.exit_code == 0
    assert "Accepted scoreable references: 0/23" in result.stdout
    assert report_path.exists()
    assert handoff_path.exists()


def test_arithmetic_controls_are_only_counted_when_exactly_solved():
    metric = RestorationEvaluationEngine().evaluate_arithmetic_controls()

    assert metric.references >= 30
    assert metric.attempted == metric.references
    assert metric.correct_top1 == metric.references
    assert metric.coverage_pct == 100.0


def test_evaluate_with_external_adjudications(tmp_path):
    adj_file = tmp_path / "adjudications.json"
    adj_file.write_text(
        """{
          "records": [
            {
              "benchmark_id": "BENCH-VOT-01",
              "decision": "accepted",
              "decision_at": "2026-09-08",
              "reviewers": ["expert-reviewer-1"],
              "review_notes": "Facsimile inspection confirms initial upright trace."
            },
            {
              "benchmark_id": "BENCH-VOT-02",
              "decision": "rejected",
              "decision_at": "2026-09-08",
              "reviewers": ["expert-reviewer-1"],
              "review_notes": "Damaged area does not uniquely constrain -I-."
            }
          ]
        }""",
        encoding="utf-8",
    )
    engine = RestorationEvaluationEngine()
    adjudicated_bench = engine.source_linked_benchmark.load_external_adjudications(adj_file)
    assert adjudicated_bench.status_counts()["accepted"] == 1
    assert adjudicated_bench.status_counts()["rejected"] == 1
    assert adjudicated_bench.status_counts()["unadjudicated"] == 21

    report = engine.evaluate(benchmark=adjudicated_bench)
    assert report.scored_references == 1
    assert report.total_entries == 23


def test_evaluation_cli_with_adjudications(tmp_path):
    adj_file = tmp_path / "adjudications.json"
    adj_file.write_text(
        """{
          "records": [
            {
              "benchmark_id": "BENCH-VOT-01",
              "decision": "accepted",
              "decision_at": "2026-09-08",
              "reviewers": ["expert-reviewer-1"]
            }
          ]
        }""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["evaluate-restorations", "--adjudications", str(adj_file)],
    )
    assert result.exit_code == 0
    assert "Accepted scoreable references: 1/23" in result.stdout
