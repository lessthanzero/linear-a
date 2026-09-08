"""Tests for cautious structural-pattern analysis."""

from typer.testing import CliRunner

from linear_a.cli.main import app
from linear_a.grammar.pcfg_engine import StructuralPatternParser
from linear_a.grammar.tree_renderer import render_ascii_tree


def test_ledger_pattern_reports_coverage_without_validity_claims():
    result = StructuralPatternParser().parse_inscription("HT_009", ["SA-RU", "FIC", "3", "KU-RO", "3"])

    assert result.genre_hypothesis == "ADMINISTRATIVE_LEDGER_PATTERN"
    assert result.pattern_coverage == 1.0
    assert "do not establish grammar" in result.limitations
    assert result.parse_tree is not None
    assert "SUMMARY_PATTERN" in render_ascii_tree(result.parse_tree)


def test_votive_pattern_and_empty_input_are_explicit_hypotheses():
    parser = StructuralPatternParser()
    result = parser.parse_inscription("IO_Za_002", ["JA-SA-SA-RA-ME", "U-NA-KA-NA-SI", "SI-RU-TE"])
    empty = parser.parse_inscription("EMPTY", [])

    assert result.genre_hypothesis == "VOTIVE_FORMULA_PATTERN"
    assert result.pattern_coverage > 0
    assert empty.genre_hypothesis == "NO_PATTERN"
    assert empty.parse_tree is None


def test_syntax_cli_resolves_curated_documents_and_rejects_unknown_ids():
    runner = CliRunner()
    known = runner.invoke(app, ["syntax", "HT_009"])
    unknown = runner.invoke(app, ["syntax", "NOT_A_DOCUMENT"])

    assert known.exit_code == 0
    assert "Structural Pattern Hypothesis: HT_009" in known.stdout
    assert "does not establish grammar" in known.stdout
    assert unknown.exit_code == 1
    assert "was not found" in unknown.stdout
