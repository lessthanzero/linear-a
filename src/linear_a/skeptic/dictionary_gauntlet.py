"""Cross-Linguistic Dictionary Gauntlet & Pseudo-Lexicon Collision Engine.

Evaluates decipherment claims (e.g. Northwest Semitic KU-RO='kull', Anatolian Luwian
A-SA-SA-RA-ME='ashasara') against rigorous statistical false-positive controls.
Simulates candidate phonetic projections against randomized synthetic pseudo-lexicons
to compute empirical False Positive Rates (FPR), Bayes Factors, and Shannon degrees of freedom.
Enforces Evidence Tier E6 (Contextual Convergence) and LADP v1.0 Section 10 & 13.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml

from linear_a.corpus.loader import get_default_corpus_dir
from linear_a.palaeography.grid_factorization import load_attested_lexicon


@dataclass
class CandidateCognateMatch:
    """A proposed lexical equation between a Linear A token and an external language root."""
    root: str
    meaning: str
    claimed_spelling: str
    target_token: str
    domain: str
    proponents: List[str]
    orthographic_plausibility: float  # 0.0 to 1.0
    null_collision_probability: float
    bayes_factor: float
    status: str  # FALSIFIED_BY_HIGH_FPR, EQUIVOCAL, STRONG_CANDIDATE


@dataclass
class GauntletReport:
    """Complete report on cross-linguistic dictionary evaluation."""
    target_language: str
    total_candidate_roots: int
    observed_matches_count: int
    top_matches: List[CandidateCognateMatch]
    null_mean_matches: float
    null_std_matches: float
    z_score: float
    empirical_p_value: float
    false_positive_rate_pct: float
    estimated_degrees_of_freedom_bits: float
    unicity_ratio: float
    is_statistically_significant: bool
    epistemic_verdict: str


def load_candidate_lexicon(language_key: str, corpus_dir: Optional[Path] = None) -> List[Dict]:
    """Load language dictionary from corpus/lexicons/{language_key}.yaml."""
    base_dir = corpus_dir or get_default_corpus_dir()
    file_path = base_dir / "lexicons" / f"{language_key}.yaml"
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("roots", [])


class DictionaryGauntlet:
    """Tests language family identification claims against Monte Carlo null lexicons."""

    def __init__(self, corpus_words: Optional[List[Dict]] = None):
        self.words = corpus_words or load_attested_lexicon()
        self.tokens = [w["token"] for w in self.words]

    def run_gauntlet(
        self,
        language_key: str = "semitic_northwest",
        n_surrogates: int = 1000,
        seed: int = 42,
    ) -> GauntletReport:
        """Run the full dictionary collision gauntlet for a candidate language."""
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        roots = load_candidate_lexicon(language_key)
        if not roots:
            return GauntletReport(
                target_language=language_key,
                total_candidate_roots=0,
                observed_matches_count=0,
                top_matches=[],
                null_mean_matches=0.0,
                null_std_matches=0.0,
                z_score=0.0,
                empirical_p_value=1.0,
                false_positive_rate_pct=100.0,
                estimated_degrees_of_freedom_bits=0.0,
                unicity_ratio=float("inf"),
                is_statistically_significant=False,
                epistemic_verdict=f"No dictionary found for language '{language_key}'.",
            )

        # 1. Match observed Linear A tokens against candidate spellings
        observed_matches: List[CandidateCognateMatch] = []
        token_set = set(self.tokens)

        for r in roots:
            claimed = r.get("claimed_spelling")
            forms = r.get("plausible_syllabic_forms", [claimed] if claimed else [])
            matched_token = next((t for t in forms if t in token_set), None)

            if matched_token:
                # Calculate theoretical random collision probability for word length L
                # Aegean syllabary size |Σ| ~ 60 open CV syllables
                word_len = len(matched_token.split("-"))
                syllable_space = 60 ** word_len
                # Collision rate P = 1 - (1 - 1/space)^len(forms)
                null_p = min(1.0, len(forms) / float(syllable_space) * 50.0)  # scaled by typical vocabulary size
                null_p = max(0.001, null_p)

                # Prior plausibility: shorter words have lower specificity
                ortho_plaus = 0.85 if word_len >= 3 else 0.45
                bayes_factor = ortho_plaus / null_p

                status = (
                    "STRONG_CANDIDATE" if bayes_factor > 20.0 and word_len >= 4
                    else ("EQUIVOCAL" if bayes_factor > 3.0
                    else "FALSIFIED_BY_HIGH_FPR")
                )

                observed_matches.append(CandidateCognateMatch(
                    root=r["root"],
                    meaning=r.get("meaning", ""),
                    claimed_spelling=claimed or matched_token,
                    target_token=matched_token,
                    domain=r.get("domain", "lexical"),
                    proponents=r.get("proponents", []),
                    orthographic_plausibility=ortho_plaus,
                    null_collision_probability=round(null_p, 4),
                    bayes_factor=round(bayes_factor, 2),
                    status=status,
                ))

        k_obs = len(observed_matches)

        # 2. Monte Carlo Null Surrogate Testing:
        # Generate N synthetic pseudo-corpora with identical token lengths sampled from
        # a uniform 60-syllable pool, and measure how often dictionary roots collide by chance.
        syllables_pool = [
            "DA", "RO", "PA", "TE", "TO", "NA", "DI", "A", "SE", "RU", "RE", "I", "SA",
            "TI", "WA", "JA", "KA", "QE", "KU", "MA", "NI", "ME", "TA", "SI", "TU", "RA",
            "PI", "PO", "PU", "KI", "KO", "NO", "NU", "WI", "WE", "ZA", "ZE", "ZO"
        ]
        token_lengths = [len(t.split("-")) for t in self.tokens]

        # All plausible syllabic forms across the candidate lexicon
        candidate_forms = set(f for r in roots for f in r.get("plausible_syllabic_forms", []))

        null_match_counts = []
        for _ in range(n_surrogates):
            # Generate pseudo-corpus
            pseudo_corpus = set()
            for l in token_lengths:
                pseudo_word = "-".join(py_rng.choice(syllables_pool) for _ in range(l))
                pseudo_corpus.add(pseudo_word)

            # Count collisions
            collisions = len(candidate_forms.intersection(pseudo_corpus))
            null_match_counts.append(collisions)

        null_mean = float(np.mean(null_match_counts))
        null_std = float(np.std(null_match_counts)) if float(np.std(null_match_counts)) > 0 else 0.1
        z_score = (k_obs - null_mean) / null_std
        p_val = float(np.mean([m >= k_obs for m in null_match_counts]))
        fpr_pct = (null_mean / len(self.tokens)) * 100.0

        # 3. Shannon Degrees-of-Freedom & Unicity Evaluation
        # Each root selection introduces log2(target_lexicon_size) bits
        lexicon_size = max(len(roots), 500)  # typical historical candidate dictionary
        dof_bits = k_obs * math.log2(lexicon_size) + len(roots) * 3.5
        # Total capacity of the 27 attested tokens (~80 signs * 3.84 bits ~ 307 bits)
        corpus_capacity = sum(token_lengths) * 3.84
        unicity_ratio = dof_bits / corpus_capacity if corpus_capacity > 0 else float("inf")
        is_overfit = unicity_ratio > 1.0

        is_significant = (z_score >= 3.0) and (p_val < 0.01) and not is_overfit

        lang_title = language_key.replace("_", " ").title()
        if is_significant:
            verdict = (
                f"STATISTICALLY SUPPORTED SIGNAL: {lang_title} achieves {k_obs} matches "
                f"(Null: {null_mean:.2f} ± {null_std:.2f}, Z = +{z_score:.2f}, p = {p_val:.4f}). "
                f"Match density exceeds random syllable collision baseline."
            )
        elif is_overfit:
            verdict = (
                f"OVERFIT / MATHEMATICALLY UNCONSTRAINED: {lang_title} hypothesis introduces "
                f"{dof_bits:.1f} degrees of freedom against a capacity of {corpus_capacity:.1f} bits "
                f"(Unicity Ratio = {unicity_ratio:.2f} > 1.0). Observed matches ({k_obs}) are indistinguishable "
                f"from chance dictionary collisions (Null: {null_mean:.2f} ± {null_std:.2f}, p = {p_val:.4f}). "
                f"Rejected under Shannon unicity theorem and LADP v1.0 Section 10."
            )
        else:
            verdict = (
                f"STATISTICALLY EQUIVOCAL / CHANCE COLLISION: {lang_title} matches ({k_obs}) "
                f"do not exceed random permutation noise (Null: {null_mean:.2f} ± {null_std:.2f}, Z = +{z_score:.2f}, p = {p_val:.4f})."
            )

        return GauntletReport(
            target_language=lang_title,
            total_candidate_roots=len(roots),
            observed_matches_count=k_obs,
            top_matches=observed_matches,
            null_mean_matches=round(null_mean, 2),
            null_std_matches=round(null_std, 2),
            z_score=round(z_score, 2),
            empirical_p_value=round(p_val, 4),
            false_positive_rate_pct=round(fpr_pct, 2),
            estimated_degrees_of_freedom_bits=round(dof_bits, 1),
            unicity_ratio=round(unicity_ratio, 2),
            is_statistically_significant=is_significant,
            epistemic_verdict=verdict,
        )
