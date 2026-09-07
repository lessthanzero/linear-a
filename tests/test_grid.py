"""Tests for Unsupervised Kober-Ventris SVD Grid Factorization."""

import numpy as np
from linear_a.palaeography.grid_factorization import (
    KoberVentrisGridEngine,
    load_attested_lexicon,
)


def test_attested_lexicon_loading():
    words = load_attested_lexicon()
    assert len(words) >= 20
    assert any(w["token"] == "JA-SA-SA-RA-ME" for w in words)
    assert any(w["token"] == "KU-RO" for w in words)
    assert any(w["token"] == "DI-RA-DI-NA" for w in words)


def test_transition_matrix_and_ppmi():
    engine = KoberVentrisGridEngine()
    vocab = engine.extract_vocabulary()
    assert len(vocab) >= 15

    t_mat = engine.build_transition_matrix(vocab)
    assert t_mat.shape == (len(vocab), len(vocab))
    assert np.all(t_mat >= 0.0)

    ppmi = engine.compute_ppmi(t_mat)
    assert ppmi.shape == t_mat.shape
    assert not np.isnan(ppmi).any()
    assert not np.isinf(ppmi).any()
    assert np.all(ppmi >= 0.0)


def test_grid_svd_factorization():
    engine = KoberVentrisGridEngine()
    report = engine.factorize_grid(
        n_components=4,
        n_consonant_clusters=4,
        n_vowel_clusters=3,
        n_permutations=200,
        seed=42,
    )

    assert report.total_signs_analyzed >= 15
    assert report.n_components == 4
    assert len(report.singular_values) == 4
    # Singular values must be non-negative and decreasing
    assert report.singular_values[0] >= report.singular_values[1] >= report.singular_values[2]
    assert report.spectral_variance_explained_pct > 30.0

    assert len(report.consonant_clusters) == 4
    assert len(report.vowel_clusters) == 3

    # Statistical verification vs null permutation
    assert report.z_score is not None
    assert report.p_value <= 1.0
    assert "KOBER-VENTRIS UNSUPERVISED SVD CONCORDANCE" in report.epistemic_verdict
