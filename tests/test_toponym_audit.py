"""Tests for the source-linked Cretan toponym audit."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from linear_a.cli.main import app
from linear_a.predictive.multilateral_solver import MultilateralLacunaeSolver
from linear_a.predictive.toponym_audit import CretanToponymAudit, ToponymAuditError


def test_audit_has_one_attested_correspondence_and_one_dispute():
    report = CretanToponymAudit().report()
    assert report["status_counts"] == {"attested": 1, "disputed": 1}
    paito = next(record for record in report["records"] if record["id"] == "TOPO-PAITO-001")
    assert paito["linear_a_form"] == "PA-I-TO"
    assert paito["linear_b_form"] == "pa-i-to"


def test_audit_rejects_an_attested_record_without_complete_forms(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"records": [{
        "id": "bad", "status": "attested", "linear_a_form": "PA-I-TO",
        "linear_b_form": None, "alphabetic_form": "Phaistos",
        "linear_a_attestations": ["HT 120"], "linear_b_attestations": ["KN"],
        "claim": "x", "limitations": "x",
        "citations": [{"author": "x", "year": "2026", "title": "x", "locator": "x", "url": "https://example.com"}],
    }]}), encoding="utf-8")
    with pytest.raises(ToponymAuditError, match="all three conventional forms"):
        CretanToponymAudit(audit_path=path)


def test_toponymic_solver_results_disclose_audit_status():
    solver = MultilateralLacunaeSolver()
    paito = next(entry for entry in solver.catalog if entry.id == "LAC-TOP-01")
    unreviewed = next(entry for entry in solver.catalog if entry.id == "LAC-TOP-02")
    audited = solver.solve_entry(paito)
    unaudited = solver.solve_entry(unreviewed)
    assert audited.source_evidence_status == "attested"
    assert audited.is_exact_match is False
    assert audited.epistemic_grade == "SOURCE_LINKED_TOPONYMIC_HYPOTHESIS"
    assert unaudited.source_evidence_status == "unaudited"
    assert unaudited.epistemic_grade == "UNAUDITED_TOPONYMIC_HYPOTHESIS"


def test_toponym_audit_cli_exports_json(tmp_path: Path):
    output = tmp_path / "toponyms.json"
    result = CliRunner().invoke(app, ["toponym-audit", "--export", str(output)])
    assert result.exit_code == 0, result.stdout
    assert "Attested correspondences: 1" in result.stdout
    assert '"TOPO-PAITO-001"' in output.read_text(encoding="utf-8")
