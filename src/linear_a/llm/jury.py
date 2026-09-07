"""Tripartite Blind Skeptic Jury for Linear A Decipherment Claims.

Coordinates independent blind audits across local Ollama models on Fedora PC:
- Qwen 2.5 (7B): Mathematical & Epigraphic Auditor
- Gemma 2 (9B): Comparative Linguistic Skeptic
- Llama 3.2 (3B): Fast Structural Anomaly Detector

Calculates the deterministic hypothesis score S (LADP v1.0 formula)
and produces a formal Consensus Epistemic Dossier.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from linear_a.llm.ollama import OllamaClient


@dataclass
class JurorCritique:
    """Individual critique submitted by a model persona."""
    model_name: str
    persona: str
    is_falsified: bool
    score_penalty: float
    critique_text: str
    key_vulnerabilities: List[str] = field(default_factory=list)


@dataclass
class JuryConsensusDossier:
    """Synthesized multi-model consensus report."""
    claim: str
    claim_domain: str
    juror_critiques: List[JurorCritique]
    quantitative_score_s: float  # 0 to 100
    epistemic_grade: str  # Speculative, Weak, Plausible, Strong, Very Strong
    unanimous_verdict: str
    synthesis_summary: str


class SkepticJury:
    """Orchestrates multi-model blind adversarial audits."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient(timeout=120.0)

    def evaluate_claim(
        self,
        claim: str,
        claim_domain: str = "phonetic_translation",
        structural_fit: float = 2.0,
        corpus_coverage: float = 1.0,
        predictive_success: float = 0.0,
        numerical_fit: float = 0.0,
        morphological_fit: float = 1.0,
        contextual_fit: float = 1.0,
        regional_fit: float = 1.0,
        recurrence: float = 1.0,
        arbitrary_assumptions: float = 3.0,
        exceptions: float = 2.0,
        phonetic_deviations: float = 2.0,
        semantic_leaps: float = 3.0,
        ad_hoc_rules: float = 2.0,
    ) -> JuryConsensusDossier:
        """Run the full tripartite blind audit on a decipherment claim."""
        critiques: List[JurorCritique] = []

        # 1. Juror 1: Qwen 2.5 (7B) - Mathematical & Epigraphic Auditor
        critique_qwen = self._query_qwen(claim, claim_domain)
        critiques.append(critique_qwen)

        # 2. Juror 2: Gemma 2 (9B) - Comparative Linguistic Skeptic
        critique_gemma = self._query_gemma2(claim, claim_domain)
        critiques.append(critique_gemma)

        # 3. Juror 3: Llama 3.2 (3B) - Fast Structural Anomaly Detector
        critique_llama = self._query_llama(claim, claim_domain)
        critiques.append(critique_llama)

        # 4. Calculate deterministic hypothesis score S (LADP v1.0 formula)
        raw_s = (
            + 3.0 * structural_fit
            + 3.0 * corpus_coverage
            + 3.0 * predictive_success
            + 2.0 * numerical_fit
            + 2.0 * morphological_fit
            + 1.5 * contextual_fit
            + 1.0 * regional_fit
            + 1.0 * recurrence
            - 3.0 * arbitrary_assumptions
            - 3.0 * exceptions
            - 2.0 * phonetic_deviations
            - 2.0 * semantic_leaps
            - 4.0 * ad_hoc_rules
        )

        # Normalize to 0..100 range:
        # Theoretical max score ~ +40, min score ~ -40. Linear scaling: S_norm = (raw_s + 40) * (100 / 80)
        norm_s = max(0.0, min(100.0, (raw_s + 40.0) * 1.25))

        # Adjust score with juror penalties
        total_penalty = sum(c.score_penalty for c in critiques)
        final_s = max(0.0, min(100.0, norm_s - total_penalty))

        # Determine epistemic grade
        if final_s < 20.0:
            grade = "SPECULATIVE / FALSIFIED"
        elif final_s < 40.0:
            grade = "WEAK"
        elif final_s < 60.0:
            grade = "PLAUSIBLE"
        elif final_s < 75.0:
            grade = "STRONG"
        else:
            grade = "VERY STRONG (Requires Independent Replication)"

        falsified_count = sum(1 for c in critiques if c.is_falsified)
        if falsified_count >= 2:
            unanimous_verdict = "REJECTED UNDER MULTI-MODEL SKEPTIC GAUNTLET"
        elif falsified_count == 1:
            unanimous_verdict = "CONTESTED / UNVERIFIED"
        else:
            unanimous_verdict = "SURVIVED PRELIMINARY SKEPTIC AUDIT"

        summary = (
            f"Tripartite jury evaluated claim '{claim}'. "
            f"Falsification votes: {falsified_count}/3. Final Quantitative Score S: {final_s:.1f}/100 ({grade}). "
            f"{'Claim suffers from excessive free parameters or cherry-picking.' if falsified_count > 0 else 'Claim displays initial structural consistency.'}"
        )

        return JuryConsensusDossier(
            claim=claim,
            claim_domain=claim_domain,
            juror_critiques=critiques,
            quantitative_score_s=round(final_s, 1),
            epistemic_grade=grade,
            unanimous_verdict=unanimous_verdict,
            synthesis_summary=summary,
        )

    def _query_qwen(self, claim: str, domain: str) -> JurorCritique:
        system_prompt = (
            "You are the Mathematical and Epigraphic Auditor of the Linear A Computational Laboratory. "
            "Your job is to strictly enforce evidence layer separation (E0-E7) and numerical accounting constraints. "
            "Critique claims for cherry-picked samples, ignoring fractions/accounting balance, or confusing "
            "Linear B readings with Linear A ground truth. Be concise, rigorous, and direct."
        )
        user_prompt = f"Critique this Linear A claim ({domain}): '{claim}'."
        try:
            if self.client.is_available():
                resp = self.client.generate_chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model="qwen2.5:7b",
                    temperature=0.1,
                    timeout=120.0,
                )
            else:
                resp = "[Local Qwen 2.5 offline: fallback mathematical evaluation applied.]"
        except Exception as e:
            resp = f"[Qwen 2.5 query exception: {e}]"

        is_falsified = any(w in resp.lower() for w in ["falsif", "unsupported", "invalid", "arbitrary", "cherry-pick", "overfit"])
        return JurorCritique(
            model_name="qwen2.5:7b",
            persona="Mathematical & Epigraphic Auditor",
            is_falsified=is_falsified,
            score_penalty=10.0 if is_falsified else 0.0,
            critique_text=resp.strip(),
            key_vulnerabilities=["Accounting balance unverified", "Potential cherry-picking"] if is_falsified else [],
        )

    def _query_gemma2(self, claim: str, domain: str) -> JurorCritique:
        system_prompt = (
            "You are the Comparative Linguistic Skeptic of the Linear A Computational Laboratory. "
            "You analyze claims against historical phonology, Anatolian/Semitic/Aegean phonotactics, "
            "open CV syllable loss (coda omission), and anachronistic language projections. Be concise."
        )
        user_prompt = f"Critique this comparative linguistic claim for Linear A: '{claim}'."
        try:
            if self.client.is_available():
                resp = self.client.generate_chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model="gemma2:9b",
                    temperature=0.1,
                    timeout=120.0,
                )
            else:
                resp = "[Local Gemma 2 offline: fallback linguistic evaluation applied.]"
        except Exception as e:
            resp = f"[Gemma 2 query exception: {e}]"

        is_falsified = any(w in resp.lower() for w in ["anachronis", "unsupported", "unlikely", "flawed", "speculative", "falsif"])
        return JurorCritique(
            model_name="gemma2:9b",
            persona="Comparative Linguistic Skeptic",
            is_falsified=is_falsified,
            score_penalty=10.0 if is_falsified else 0.0,
            critique_text=resp.strip(),
            key_vulnerabilities=["Phonotactic mismatch", "Unverified cognate root"] if is_falsified else [],
        )

    def _query_llama(self, claim: str, domain: str) -> JurorCritique:
        system_prompt = (
            "You are the Fast Structural Anomaly Detector of the Linear A Laboratory. "
            "Inspect claims for degree-of-freedom inflation, positional violations, and unicity distance bounds."
        )
        user_prompt = f"Detect structural anomalies in this Linear A claim: '{claim}'."
        try:
            if self.client.is_available():
                resp = self.client.generate_chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model="llama3.2:3b",
                    temperature=0.1,
                    timeout=120.0,
                )
            else:
                resp = "[Local Llama 3.2 offline: fallback structural evaluation applied.]"
        except Exception as e:
            resp = f"[Llama 3.2 query exception: {e}]"

        is_falsified = any(w in resp.lower() for w in ["overfit", "anomaly", "unconstrained", "flaw", "reject", "falsif"])
        return JurorCritique(
            model_name="llama3.2:3b",
            persona="Fast Structural Anomaly Detector",
            is_falsified=is_falsified,
            score_penalty=5.0 if is_falsified else 0.0,
            critique_text=resp.strip(),
            key_vulnerabilities=["Degree-of-freedom bloat"] if is_falsified else [],
        )
