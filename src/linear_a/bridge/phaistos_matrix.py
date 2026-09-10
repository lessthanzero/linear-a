"""Phaistos Disc Structural Bridge and Epigraphic Firewall.

Implements Section 14 of Linear A Decipherment Protocol (LADP):
Maintains quarantine between CORPUS_LINEAR_A and CORPUS_PHAISTOS_DISK.
Cross-script comparisons are exploratory hypotheses, not confirmations.
"""

from typing import List
from pydantic import BaseModel, Field

from linear_a.morphology.affix_sieve import AffixSieve
from linear_a.votive.libation_engine import LibationEngine


class PhaistosFirewallStatus(BaseModel):
    is_firewall_intact: bool
    isolated_corpora: List[str] = Field(default_factory=lambda: ["CORPUS_LINEAR_A", "CORPUS_PHAISTOS_DISK"])
    sound_leak_detected: bool = False
    evidence_layer_violation: bool = False
    rationale: str


class CrossScriptCorrespondence(BaseModel):
    feature_id: str
    feature_name: str
    disc_evidence: str
    linear_a_evidence: str
    statistical_metric: str
    metric_value: float
    concordance_level: str  # EXPLORATORY | STRUCTURAL_ANALOGY | EQUIVOCAL
    firewall_compliant: bool = True


class PhaistosHomologyReport(BaseModel):
    firewall: PhaistosFirewallStatus
    overall_structural_homology_score_pct: float
    correspondences: List[CrossScriptCorrespondence]
    te_vs_me_likelihood_ratio: float
    plumed_head_prefix_correlation: float
    clause_cadence_homology_pct: float
    epistemic_verdict: str


class PhaistosBridgeEngine:
    """Evaluates exploratory cross-script structural analogies under firewall constraints."""

    def __init__(self):
        self.affix_sieve = AffixSieve()
        self.libation_engine = LibationEngine()

    def verify_firewall_integrity(self) -> PhaistosFirewallStatus:
        """Report firewall *policy* status (stub check — not a formal quarantine proof)."""
        return PhaistosFirewallStatus(
            is_firewall_intact=True,
            isolated_corpora=["CORPUS_LINEAR_A", "CORPUS_PHAISTOS_DISK"],
            sound_leak_detected=False,
            evidence_layer_violation=False,
            rationale=(
                "FIREWALL POLICY: Linear A phonetic priors are intended to derive from "
                "Linear B correspondences only. Phaistos Disc signs are treated as an "
                "independent external corpus. This routine does not scan the codebase for "
                "leakage; it records the intended policy only."
            ),
        )

    def evaluate_cross_script_homology(self) -> PhaistosHomologyReport:
        """Return exploratory analogy features — not a confirmation of shared liturgy."""
        fw = self.verify_firewall_integrity()
        affix_rep = self.affix_sieve.evaluate_affixes()
        libation_rep = self.libation_engine.evaluate_concordance()

        te_lr = affix_rep.te_vs_me_likelihood_ratio
        c1 = CrossScriptCorrespondence(
            feature_id="HOM-001",
            feature_name="Word-Final Position Analogy (Disc Sign 35 vs Linear A AB04)",
            disc_evidence="Sign 35 often word-final in the Disc transcription.",
            linear_a_evidence="AB04 (conventionally TE) is frequent word-finally in Linear A priors.",
            statistical_metric="Likelihood Ratio (TE vs ME) under stated priors",
            metric_value=float(te_lr),
            concordance_level="EXPLORATORY",
            firewall_compliant=True,
        )

        # Prefix correlation is not computed live; leave as unset exploratory marker.
        c2 = CrossScriptCorrespondence(
            feature_id="HOM-002",
            feature_name="Word-Initial Prefixation Analogy (Disc Sign 02 vs Linear A JA-/A-)",
            disc_evidence="Sign 02 (Plumed Head) is frequent word-initially.",
            linear_a_evidence="JA- and A- are common openings in curated votive formulas.",
            statistical_metric="Positional analogy (not a fitted correlation)",
            metric_value=0.0,
            concordance_level="EXPLORATORY",
            firewall_compliant=True,
        )

        clause_homology = libation_rep.phaistos_disc_liturgical_homology_score
        c3 = CrossScriptCorrespondence(
            feature_id="HOM-003",
            feature_name="Clause / Formula Pacing Analogy",
            disc_evidence="Disc groups and strokes invite formulaic readings as a hypothesis.",
            linear_a_evidence="Stone libation formulas show recurring segment order in curated samples.",
            statistical_metric="Exploratory placeholder score (%)",
            metric_value=float(clause_homology),
            concordance_level="EXPLORATORY",
            firewall_compliant=True,
        )

        c4 = CrossScriptCorrespondence(
            feature_id="HOM-004",
            feature_name="Findspot Co-occurrence (Room 8 / PH 1)",
            disc_evidence="Disc found at Phaistos in association with Linear A materials (published archaeology).",
            linear_a_evidence="Tablet PH 1 is a separate Linear A document from the same broader complex.",
            statistical_metric="Contextual note (not a statistical score)",
            metric_value=0.0,
            concordance_level="EQUIVOCAL",
            firewall_compliant=True,
        )

        verdict = (
            "EXPLORATORY ANALOGY ONLY: Cross-script features are hypotheses under the "
            f"firewall policy. TE-vs-ME LR under stated priors is {te_lr:.1e}. "
            "Do not treat this report as confirmation of identical sacred formulaic syntax "
            "or shared decipherment."
        )

        return PhaistosHomologyReport(
            firewall=fw,
            overall_structural_homology_score_pct=0.0,
            correspondences=[c1, c2, c3, c4],
            te_vs_me_likelihood_ratio=float(te_lr),
            plumed_head_prefix_correlation=0.0,
            clause_cadence_homology_pct=float(clause_homology),
            epistemic_verdict=verdict,
        )
