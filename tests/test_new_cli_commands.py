"""Unit tests for new CLI commands: segment-morphology, solve-diophantine, typological-profile."""

from typer.testing import CliRunner
from linear_a.cli.main import app

runner = CliRunner()


def test_cli_segment_morphology():
    result = runner.invoke(app, ["segment-morphology", "--votive-only"])
    assert result.exit_code == 0
    assert "Linear A Unsupervised Morphological Sieve" in result.stdout
    assert "SA-SA-RA" in result.stdout or "VOTIVE" in result.stdout


def test_cli_solve_diophantine():
    result = runner.invoke(app, ["solve-diophantine"])
    assert result.exit_code == 0
    assert "Linear A Diophantine Multi-Fraction Solver" in result.stdout
    assert "100.0%" in result.stdout


def test_cli_typological_profile():
    result = runner.invoke(app, ["typological-profile"])
    assert result.exit_code == 0
    assert "Linear A Typological Linguistic Profiler" in result.stdout
    assert "Cross-Linguistic Structural Distance Rankings" in result.stdout


def test_cli_votive_grammar():
    result = runner.invoke(app, ["votive-grammar"])
    assert result.exit_code == 0
    assert "Linear A Votive Clausal Grammar" in result.stdout
    assert "Regional Liturgical Profiles" in result.stdout


def test_cli_analyze_ligatures():
    result = runner.invoke(app, ["analyze-ligatures"])
    assert result.exit_code == 0
    assert "Linear A Paleographic Ligature Taxonomy" in result.stdout
    assert "Composite Monograms & Adjunct Modifiers" in result.stdout


def test_cli_induce_substratum():
    result = runner.invoke(app, ["induce-substratum"])
    assert result.exit_code == 0
    assert "Minoan Phonological Substratum" in result.stdout
    assert "Voicing Neutrality Matrix" in result.stdout


def test_cli_solve_multi_commodity():
    result = runner.invoke(app, ["solve-multi-commodity"])
    assert result.exit_code == 0
    assert "Multi-Commodity Metrological Diophantine Solver" in result.stdout
    assert "100.0%" in result.stdout


def test_cli_cluster_ductus():
    result = runner.invoke(app, ["cluster-ductus"])
    assert result.exit_code == 0
    assert "Unsupervised Scribal Hand Ductus Clustering" in result.stdout
    assert "Discovered Latent Scribal Hands" in result.stdout


def test_cli_review_adjudications():
    result = runner.invoke(app, ["review-adjudications"])
    assert result.exit_code == 0
    assert "Epigrapher Peer-Review Adjudication Portal" in result.stdout
    assert "Candidate-Supported Damaged Token Dossiers" in result.stdout


def test_cli_analyze_dialectology():
    result = runner.invoke(app, ["analyze-dialectology", "--permutations", "50"])
    assert result.exit_code == 0
    assert "Geographical Dialectology" in result.stdout
    assert "Mantel Permutation Test" in result.stdout


def test_cli_analyze_script_phylogeny():
    result = runner.invoke(app, ["analyze-script-phylogeny"])
    assert result.exit_code == 0
    assert "Script Phylogeny & Lineage Engine" in result.stdout
    assert "Aegean Writing Systems Lineage" in result.stdout


def test_cli_solve_unified_metrology():
    result = runner.invoke(app, ["solve-unified-metrology"])
    assert result.exit_code == 0
    assert "Unified Metrological Weight & Commodity Tree" in result.stdout
    assert "Palatial Commodity Equivalence Coefficients" in result.stdout


def test_cli_render_glyph_vectors():
    result = runner.invoke(app, ["render-glyph-vectors"])
    assert result.exit_code == 0
    assert "Stroke Vector Engine" in result.stdout
    assert "Vectorized Linear A Syllabic Signs" in result.stdout

