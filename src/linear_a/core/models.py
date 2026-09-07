"""Core Pydantic data models for the Linear A Computational Laboratory.

Implements the data architecture defined in Linear A Decipherment Protocol (LADP v1.0),
strictly segregating Glyphs, Signs, Phonetic Values, Words, Morphemes, and Meanings.
"""

from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SignClass(str, Enum):
    SYLLABIC = "syllabic"
    LOGOGRAM = "logogram"
    NUMERAL = "numeral"
    FRACTION = "fraction"
    PUNCTUATION = "punctuation"
    UNKNOWN = "unknown"


class ObjectType(str, Enum):
    TABLET = "tablet"
    ROUNDEL = "roundel"
    NODULE = "nodule"
    LIBATION_VESSEL = "libation_vessel"
    METAL_AXE = "metal_axe"
    METAL_RING = "metal_ring"
    DIPINTO = "dipinto"
    GRAFFITO = "graffito"
    OTHER = "other"


class Site(str, Enum):
    HAGIA_TRIADA = "Hagia_Triada"
    PHAISTOS = "Phaistos"
    KNOSSOS = "Knossos"
    MALIA = "Malia"
    ZAKROS = "Zakros"
    KHANIA = "Khania"
    TYLISSOS = "Tylissos"
    MOUNT_JUKTAS = "Mount_Juktas"
    PSYCHRO = "Psychro"
    PALAIKASTRO = "Palaikastro"
    ARKALOCHORI = "Arkalochori"
    PRASSAS = "Prassas"
    SYME = "Syme"
    OTHER = "Other"


class PhoneticCandidate(BaseModel):
    """Candidate phonetic reading with Bayesian prior confidence."""
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "Linear_B_Bridge"
    inference_level: str = "E2"


class LinearBCorrespondence(BaseModel):
    """Linear B sign correspondence prior."""
    candidate: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: Optional[str] = None


class Sign(BaseModel):
    """Sign metadata catalog entry."""
    glyph_id: str  # e.g. "AB01", "AB04", "A301", "GRA", "FIC"
    canonical_name: Optional[str] = None
    variant_ids: List[str] = Field(default_factory=list)
    sign_class: SignClass = SignClass.SYLLABIC
    linearB_correspondence: Optional[LinearBCorrespondence] = None
    phonetic_candidates: List[PhoneticCandidate] = Field(default_factory=list)
    description: Optional[str] = None
    relative_frequency_rank: Optional[int] = None


class TextUnit(BaseModel):
    """Individual line or structural section within an inscription."""
    id: str
    inscription_id: str
    line_index: int
    raw_transcription: str
    sign_sequence: List[str] = Field(default_factory=list)  # List of glyph_ids
    word_boundaries: List[int] = Field(default_factory=list)
    damage_mask: List[bool] = Field(default_factory=list)
    numerals: List[int] = Field(default_factory=list)
    fractions: List[str] = Field(default_factory=list)
    logograms: List[str] = Field(default_factory=list)
    divider_positions: List[int] = Field(default_factory=list)
    notes: Optional[str] = None


class Inscription(BaseModel):
    """A documented Linear A artifact / inscription."""
    id: str  # e.g. "HT_009", "PH_001", "IO_Za_002"
    museum_id: Optional[str] = None
    site: Site
    findspot_details: Optional[str] = None
    date_range: Optional[str] = None  # e.g. "LM IB", "MM III"
    object_type: ObjectType
    source: str = "GORILA"
    text_units: List[TextUnit] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: Optional[str] = None

    @property
    def total_signs(self) -> int:
        return sum(len(tu.sign_sequence) for tu in self.text_units)


class TokenPosition(str, Enum):
    INITIAL = "initial"
    MEDIAL = "medial"
    FINAL = "final"
    STANDALONE = "standalone"


class SemanticContext(str, Enum):
    ADMINISTRATIVE = "administrative"
    RELIGIOUS = "religious"
    COMMODITY = "commodity"
    PERSON = "person"
    PLACE = "place"
    UNKNOWN = "unknown"


class Token(BaseModel):
    """Segmented word or sign-group."""
    token_id: str
    inscription_id: str
    glyph_sequence: List[str]
    transliteration_candidate: Optional[str] = None
    position: TokenPosition = TokenPosition.STANDALONE
    preceding_tokens: List[str] = Field(default_factory=list)
    following_tokens: List[str] = Field(default_factory=list)
    numerical_context: Optional[Dict[str, Any]] = None
    semantic_context: SemanticContext = SemanticContext.UNKNOWN


class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    WEAK = "weak"
    FALSIFIED = "falsified"
    SUPPORTED = "supported"


class HypothesisType(str, Enum):
    PHONETIC = "phonetic"
    MORPHOLOGICAL = "morphological"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    LANGUAGE_FAMILY = "language_family"
    CROSS_SCRIPT = "cross_script"


class Hypothesis(BaseModel):
    """Formally registered scientific claim subject to the Skeptic Gauntlet."""
    id: str  # e.g. "H-001"
    type: HypothesisType
    claim: str
    evidence_ids: List[str] = Field(default_factory=list)
    predictions: List[str] = Field(default_factory=list)
    counterexamples: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    score: Optional[float] = None  # Normalized 0..100


# Accounting & Fractional Ledger Models

class FractionSymbol(BaseModel):
    """Linear A fractional sign with exact mathematical value."""
    symbol: str  # e.g. "J", "E", "F", "K", "A", "H"
    fraction_numerator: int
    fraction_denominator: int
    decimal_value: float
    confidence: float = 0.95
    proponents: List[str] = Field(default_factory=lambda: ["Ferrara & Corazza 2020"])

    @property
    def value(self) -> Fraction:
        return Fraction(self.fraction_numerator, self.fraction_denominator)


class LedgerLineItem(BaseModel):
    """Single item entry in an accounting ledger."""
    entry_header: str  # e.g. "PA-DE", "KU-RO", person name or heading
    commodity: Optional[str] = None  # e.g. "FIC", "GRA", "VIN", "CYP", "VIR"
    integer_amount: int = 0
    fractional_symbols: List[str] = Field(default_factory=list)
    fractional_amount_num: int = 0
    fractional_amount_den: int = 1
    notes: Optional[str] = None

    @property
    def total_fraction(self) -> Fraction:
        frac = Fraction(self.fractional_amount_num, self.fractional_amount_den)
        return Fraction(self.integer_amount, 1) + frac

    @property
    def total_decimal(self) -> float:
        return float(self.total_fraction)


class LedgerVerificationResult(BaseModel):
    """Result of verifying tablet accounting balance sum(inputs) == KU-RO."""
    tablet_id: str
    commodity: Optional[str]
    input_items_count: int
    computed_sum_fraction: str  # e.g. "31" or "15 3/4"
    computed_sum_decimal: float
    stated_kuro_fraction: Optional[str] = None
    stated_kuro_decimal: Optional[float] = None
    is_balanced: bool
    discrepancy_decimal: float
    evidence_tier: str = "E3"
    rationale: str
