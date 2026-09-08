"""Minoan Moraic Prosodic Scansion & Sacred Metric Cadence Engine (LADP v1.0).

Quantifies syllable weight (light ˘ = 1 mora, heavy ¯ = 2 morae), rhythmic feet,
caesura positions, and poetic meter across peak sanctuary libation vessels.

References:
- Duhoux (1989), "Le Linéaire A: problèmes de déchiffrement" (rhythmic cadence).
- Younger (2000), "Linear A Texts in Phonetic Transcription" (poetic cola).
- Davis (2014), "Minoan Stone Vessels with Linear A Inscriptions" (metrical structures).
"""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional

from linear_a.votive.libation_engine import LibationEngine, LibationVessel


@dataclass
class MoraicSyllable:
    """Individual syllable or offglide with moraic weight assignment."""
    syllable: str
    weight: int           # 1 = light (˘), 2 = heavy (¯), 0 = offglide (˘̯)
    symbol: str           # "˘", "¯", or "·"
    is_diphthong: bool    # True if closing diphthong nucleus or offglide
    is_terminal: bool     # True if word-final syllable

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WordScansion:
    """Moraic scansion for an individual word token."""
    word: str
    syllables: List[MoraicSyllable]
    scansion_str: str     # e.g., "˘ ˘ ¯ ¯ ˘ ˘"
    total_morae: int
    stress_prediction: str  # "PENULTIMATE", "ANTEPENULTIMATE", "FINAL", "MONOSYLLABIC"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "syllables": [s.to_dict() for s in self.syllables],
            "scansion_str": self.scansion_str,
            "total_morae": self.total_morae,
            "stress_prediction": self.stress_prediction,
        }


@dataclass
class InscriptionMeterReport:
    """Metric analysis for an entire inscription."""
    id: str
    site: str
    carrier: str
    words: List[WordScansion]
    total_morae: int
    scansion_line: str
    dominant_foot: str     # "DACTYLIC", "TROCHAIC", "AMPHIBRACHIC", "ISOCHRONIC"
    regularity_score: float  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "site": self.site,
            "carrier": self.carrier,
            "words": [w.to_dict() for w in self.words],
            "total_morae": self.total_morae,
            "scansion_line": self.scansion_line,
            "dominant_foot": self.dominant_foot,
            "regularity_score": self.regularity_score,
        }


class ProsodicMeterEngine:
    """Evaluates moraic weight and rhythmic meter in Linear A texts."""

    # Diphthong offglides in Aegean orthography (pure vowels following CV open syllables)
    DIPHTHONG_OFFGLIDES = {"I", "U"}
    DIPHTHONG_NUCLEI_VOWELS = {"A", "O", "E", "U"}
    # Formulaic heavy enclitics or word-final long positions
    HEAVY_ENCLITICS = {"ME", "TE", "SI", "JA", "NA"}

    def __init__(self, libation_engine: Optional[LibationEngine] = None):
        self.libation_engine = libation_engine or LibationEngine()

    def scan_word(self, word: str) -> WordScansion:
        """Parse a hyphenated Linear A word into its moraic scansion."""
        clean = word.strip().upper()
        raw_syllables = [s.strip() for s in clean.split("-") if s.strip()]

        if not raw_syllables:
            return WordScansion(word=word, syllables=[], scansion_str="", total_morae=0, stress_prediction="NONE")

        scanned_sylls: List[MoraicSyllable] = []
        n = len(raw_syllables)

        i = 0
        while i < n:
            s = raw_syllables[i]
            is_terminal = (i == n - 1)

            # Check if current syllable forms a diphthong with next pure vowel sign (e.g. PA + I -> PAI)
            is_diphthong_nucleus = False
            if i + 1 < n:
                next_s = raw_syllables[i + 1]
                # If s ends in A/O/E/U and next sign is I or U
                s_vowel = s[-1] if s else ""
                if s_vowel in self.DIPHTHONG_NUCLEI_VOWELS and next_s in self.DIPHTHONG_OFFGLIDES:
                    is_diphthong_nucleus = True

            if is_diphthong_nucleus:
                # Diphthong nucleus carries heavy weight (2 morae)
                scanned_sylls.append(
                    MoraicSyllable(
                        syllable=s,
                        weight=2,
                        symbol="¯",
                        is_diphthong=True,
                        is_terminal=False,
                    )
                )
                # Next sign is the offglide, absorbed into the diphthong (0 extra morae, marked ·)
                offglide_s = raw_syllables[i + 1]
                scanned_sylls.append(
                    MoraicSyllable(
                        syllable=offglide_s,
                        weight=0,
                        symbol="·",
                        is_diphthong=True,
                        is_terminal=(i + 1 == n - 1),
                    )
                )
                i += 2
                continue

            # Standard moraic rules for Bronze Age Aegean syllabic scripts:
            # Word-final syllables with closed or enclitic weight count as 2 morae
            if is_terminal and s in self.HEAVY_ENCLITICS and n > 2:
                weight = 2
                symbol = "¯"
            elif s in ("*301", "A301"):  # Complex labialized cluster counts heavy
                weight = 2
                symbol = "¯"
            else:
                weight = 1
                symbol = "˘"

            scanned_sylls.append(
                MoraicSyllable(
                    syllable=s,
                    weight=weight,
                    symbol=symbol,
                    is_diphthong=False,
                    is_terminal=is_terminal,
                )
            )
            i += 1

        total_m = sum(sy.weight for sy in scanned_sylls)
        # Produce clean scansion display string omitting zero-weight offglides
        scansion_tokens = [sy.symbol for sy in scanned_sylls if sy.symbol != "·"]
        scansion_str = " ".join(scansion_tokens) if scansion_tokens else "˘"

        # Stress rule in open CV languages: penultimate if heavy, antepenultimate otherwise
        audible_sylls = [sy for sy in scanned_sylls if sy.weight > 0]
        if len(audible_sylls) == 1:
            stress = "MONOSYLLABIC"
        elif len(audible_sylls) == 2:
            stress = "PENULTIMATE"
        else:
            penult = audible_sylls[-2]
            stress = "PENULTIMATE" if penult.weight == 2 else "ANTEPENULTIMATE"

        return WordScansion(
            word=word,
            syllables=scanned_sylls,
            scansion_str=scansion_str,
            total_morae=total_m,
            stress_prediction=stress,
        )

    def scan_inscription(
        self,
        text: str,
        id: str = "TEXT",
        site: str = "Unknown",
        carrier: str = "Stone Vessel",
    ) -> InscriptionMeterReport:
        """Scan an entire multi-word inscription and determine rhythmic meter."""
        words_raw = [w.strip() for w in text.strip().split() if w.strip()]
        scanned_words = [self.scan_word(w) for w in words_raw]

        total_morae = sum(w.total_morae for w in scanned_words)
        scansion_line = " | ".join(w.scansion_str for w in scanned_words)

        # Detect dominant foot pattern from mora distribution
        all_symbols = [sy.symbol for w in scanned_words for sy in w.syllables if sy.symbol != "·"]
        symbol_seq = "".join(all_symbols)

        dactylic_matches = len(re.findall(r"¯˘˘", symbol_seq))
        trochaic_matches = len(re.findall(r"¯˘", symbol_seq))
        amphibrachic_matches = len(re.findall(r"˘¯˘", symbol_seq))

        if dactylic_matches >= trochaic_matches and dactylic_matches > 0:
            dominant_foot = "DACTYLIC"
            regularity = min(1.0, 0.4 + 0.15 * dactylic_matches)
        elif trochaic_matches > 0:
            dominant_foot = "TROCHAIC"
            regularity = min(1.0, 0.4 + 0.12 * trochaic_matches)
        elif amphibrachic_matches > 0:
            dominant_foot = "AMPHIBRACHIC"
            regularity = min(1.0, 0.4 + 0.10 * amphibrachic_matches)
        else:
            dominant_foot = "ISOCHRONIC"
            regularity = 0.50

        return InscriptionMeterReport(
            id=id,
            site=site,
            carrier=carrier,
            words=scanned_words,
            total_morae=total_morae,
            scansion_line=scansion_line,
            dominant_foot=dominant_foot,
            regularity_score=round(regularity, 3),
        )

    def analyze_votive_corpus(self) -> Dict[str, Any]:
        """Perform corpus-wide prosodic and metric analysis across peak sanctuary vessels."""
        vessels = self.libation_engine.vessels
        reports: List[InscriptionMeterReport] = []

        for v in vessels:
            report = self.scan_inscription(
                text=v.transcription_raw,
                id=v.id,
                site=v.site,
                carrier=v.vessel_type,
            )
            reports.append(report)

        mora_counts = [r.total_morae for r in reports if r.total_morae > 0]
        mean_morae = sum(mora_counts) / len(mora_counts) if mora_counts else 0.0

        foot_counts: Dict[str, int] = {}
        for r in reports:
            foot_counts[r.dominant_foot] = foot_counts.get(r.dominant_foot, 0) + 1

        return {
            "vessel_count": len(reports),
            "mean_morae_per_vessel": round(mean_morae, 2),
            "dominant_corpus_meter": max(foot_counts, key=foot_counts.get) if foot_counts else "ISOCHRONIC",
            "meter_distribution": foot_counts,
            "vessel_reports": [r.to_dict() for r in reports],
        }
