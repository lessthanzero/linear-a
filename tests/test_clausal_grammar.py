"""Unit tests for Probabilistic Votive Clausal Grammar Engine."""

from linear_a.votive.clausal_grammar import VotiveClausalGrammarEngine


def test_votive_clausal_grammar_runs():
    engine = VotiveClausalGrammarEngine()
    report = engine.evaluate_votive_grammar()

    assert report.total_vessels_parsed >= 4
    assert report.canonical_syntax_conformance_pct >= 60.0
    assert len(report.regional_profiles) >= 3
    assert "PHASE_1_INVOCATION_HEADER" in report.markov_matrix.states
    assert report.markov_matrix.total_transitions > 10


def test_regional_profiles_capture_variants():
    engine = VotiveClausalGrammarEngine()
    report = engine.evaluate_votive_grammar()

    # Find Mount Juktas and Syme profiles
    profile_names = {r.region_name for r in report.regional_profiles}
    assert any("Central" in p or "Juktas" in p for p in profile_names)
    assert any("Syme" in p or "South" in p for p in profile_names)

    # Check Syme has JA-DI-KI-TU epiclesis
    syme = next((r for r in report.regional_profiles if "Syme" in r.region_name or "South" in r.region_name), None)
    assert syme is not None
    assert "DI-KI-TU" in syme.epithet_variant
