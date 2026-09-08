"""Tests for reading, phonetics, and acoustic reciter CLI commands."""

from typer.testing import CliRunner
from linear_a.cli.main import app

runner = CliRunner()


def test_cli_phonetics_atlas():
    result = runner.invoke(app, ["phonetics-atlas", "--limit", "5"])
    assert result.exit_code == 0
    assert "Aegean Syllabary Phonetics & Acoustic Formant Atlas" in result.output
    assert "AB08" in result.output or "DA" in result.output


def test_cli_phonetics_atlas_filter():
    result = runner.invoke(app, ["phonetics-atlas", "--tier", "E4", "--limit", "5"])
    assert result.exit_code == 0
    assert "E4" in result.output


def test_cli_analyze_prosody():
    result = runner.invoke(app, ["analyze-prosody"])
    assert result.exit_code == 0
    assert "Minoan Moraic Prosodic Scansion & Sacred Metric Cadence" in result.output
    assert "Mean Morae per Inscription" in result.output


def test_cli_recite_text_votive():
    result = runner.invoke(app, ["recite-text", "IO_Za_2"])
    assert result.exit_code == 0
    assert "Minoan Epigraphic Reciter: IO_Za_2" in result.output
    assert "A-TA-I-*301-WA-JA" in result.output
    assert "Syllabic Ductus & Acoustic Formant Schedule" in result.output


def test_cli_recite_text_tablet():
    result = runner.invoke(app, ["recite-text", "HT_085"])
    assert result.exit_code == 0
    assert "Minoan Epigraphic Reciter: HT_085" in result.output
    assert "DA-MA-TE" in result.output


def test_cli_recite_text_unknown():
    result = runner.invoke(app, ["recite-text", "NON_EXISTENT_ID_999"])
    assert result.exit_code != 0
    assert "not found" in result.output
