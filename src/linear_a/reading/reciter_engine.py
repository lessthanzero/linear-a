"""Minoan Inscription Reciter Engine (LADP v1.0).

Bridges acoustic formant modeling, prosodic meter scansion, and epigraphic texts
to produce comprehensive recitation packages and real-time Web Audio synthesis schedules.
"""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional

from linear_a.corpus.loader import get_tablet_by_id, load_all_tablets
from linear_a.phonology.acoustic_reconstruction import (
    AcousticReconstructionEngine,
    AcousticScheduleItem,
    PhoneticSignProfile,
)
from linear_a.phonology.prosodic_meter import (
    InscriptionMeterReport,
    MoraicSyllable,
    ProsodicMeterEngine,
)
from linear_a.votive.libation_engine import LibationEngine, LibationVessel


@dataclass
class RecitationSyllable:
    """Richly annotated syllable for interactive recitation and playback tracking."""
    syllable: str
    sign_code: str
    ipa: str
    confidence_tier: str    # "E4", "E3", "E2", "E0"
    confidence_score: float # 0.0 to 1.0
    weight: int             # 1 (light ˘), 2 (heavy ¯)
    symbol: str             # "˘" or "¯"
    f0_start_hz: float
    duration_ms: float
    formant_f1: float
    formant_f2: float
    formant_f3: float
    consonant_manner: str
    is_word_terminal: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecitationPackage:
    """Full multimodal recitation package for an inscription."""
    id: str
    genre: str              # "VOTIVE_LIBATION" or "ADMINISTRATIVE_LEDGER"
    site: str
    carrier: str
    raw_text: str
    ipa_text: str
    scansion_str: str
    dominant_foot: str
    total_morae: int
    mean_confidence: float
    gorila_ref: str = ""
    dating: str = "LM I (c. 1600–1450 BCE)"
    syllables: List[RecitationSyllable] = field(default_factory=list)
    synthesis_schedule: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "genre": self.genre,
            "site": self.site,
            "carrier": self.carrier,
            "raw_text": self.raw_text,
            "ipa_text": self.ipa_text,
            "scansion_str": self.scansion_str,
            "dominant_foot": self.dominant_foot,
            "total_morae": self.total_morae,
            "mean_confidence": round(self.mean_confidence, 3),
            "gorila_ref": self.gorila_ref,
            "dating": self.dating,
            "syllables": [s.to_dict() for s in self.syllables],
            "synthesis_schedule": self.synthesis_schedule,
        }


# Curated Flagship Inscriptions for Live Acoustic Reading
FLAGSHIP_INSCRIPTIONS: List[Dict[str, str]] = [
    {
        "id": "IO_Za_2",
        "genre": "VOTIVE_LIBATION",
        "site": "Mount Juktas Peak Sanctuary",
        "carrier": "Steatite Libation Ladle",
        "dating": "MM III – LM I (c. 1650–1450 BCE)",
        "gorila_ref": "GORILA 4, IO Za 2, pp. 16–17; Karetsou (1974)",
        "raw_text": "A-TA-I-*301-WA-JA JA-SA-SA-RA-ME U-NA-KA-NA-SI I-PI-NA-MA SI-RU-TE",
    },
    {
        "id": "PK_Za_11",
        "genre": "VOTIVE_LIBATION",
        "site": "Palaikastro Peak Sanctuary",
        "carrier": "Serpentine Blossom Bowl",
        "dating": "LM I (c. 1550–1450 BCE)",
        "gorila_ref": "GORILA 4, PK Za 11, pp. 38–39; Bosanquet & Dawkins (1923)",
        "raw_text": "A-TA-I-*301-WA-JA A-SA-SA-RA-ME PI-TE-RI A-KO-A-NE",
    },
    {
        "id": "PR_Za_1",
        "genre": "VOTIVE_LIBATION",
        "site": "Prassas Peak Sanctuary",
        "carrier": "Chlorite Libation Table",
        "dating": "LM I (c. 1550–1450 BCE)",
        "gorila_ref": "GORILA 4, PR Za 1, pp. 48–49; Platon (1954)",
        "raw_text": "JA-SA-SA-RA-MA-NA TA-NA-I-TE TA-NA-TI",
    },
    {
        "id": "SY_Za_1",
        "genre": "VOTIVE_LIBATION",
        "site": "Syme Mountain Sanctuary",
        "carrier": "Stone Libation Table",
        "dating": "MM III – LM I (c. 1650–1450 BCE)",
        "gorila_ref": "GORILA 4, SY Za 1, pp. 58–59; Lebessi et al. (1981)",
        "raw_text": "A-TA-I-*301-WA-JA JA-SA-SA-RA-ME WI-TE-MO-TI",
    },
    {
        "id": "HT_085",
        "genre": "ADMINISTRATIVE_LEDGER",
        "site": "Hagia Triada Royal Villa",
        "carrier": "Clay Tablet (LM IB)",
        "dating": "LM IB destruction horizon (c. 1450 BCE)",
        "gorila_ref": "GORILA 1, HT 85, pp. 138–139; Halbherr, Stefani & Banti (1977)",
        "raw_text": "DA-MA-TE GRA 20 E OLE 10 F KU-RO 30 EF",
    },
    {
        "id": "HT_013",
        "genre": "ADMINISTRATIVE_LEDGER",
        "site": "Hagia Triada Royal Villa",
        "carrier": "Clay Tablet (LM IB)",
        "dating": "LM IB destruction horizon (c. 1450 BCE)",
        "gorila_ref": "GORILA 1, HT 13, pp. 24–25; Halbherr, Stefani & Banti (1977)",
        "raw_text": "KA-PA GRA 15 VIN 5 KU-RO 20",
    },
    {
        "id": "PH_001",
        "genre": "ADMINISTRATIVE_LEDGER",
        "site": "Phaistos Palatial Archive",
        "carrier": "Clay Tablet (LM IB)",
        "dating": "LM IB (c. 1450 BCE)",
        "gorila_ref": "GORILA 1, PH 1, pp. 296–297; Pernier (1935)",
        "raw_text": "PA-I-TO GRA 100 TEL 20 KU-RO 120",
    },
    {
        "id": "ZA_001",
        "genre": "ADMINISTRATIVE_LEDGER",
        "site": "Zakros Palatial Archive",
        "carrier": "Clay Tablet (LM IB)",
        "dating": "LM IB destruction horizon (c. 1450 BCE)",
        "gorila_ref": "GORILA 3, ZA 1, pp. 2–3; Platon (1971)",
        "raw_text": "DI-KA-TA GRA 50 OLE 25 KU-RO 75",
    },
]


class ReciterEngine:
    """Coordinates phonetic transcription, prosodic scansion, and audio synthesis schedules."""

    def __init__(
        self,
        acoustic_engine: Optional[AcousticReconstructionEngine] = None,
        prosodic_engine: Optional[ProsodicMeterEngine] = None,
        libation_engine: Optional[LibationEngine] = None,
    ):
        self.acoustic = acoustic_engine or AcousticReconstructionEngine()
        self.prosody = prosodic_engine or ProsodicMeterEngine()
        self.libations = libation_engine or LibationEngine()

    def _normalize_id(self, identifier: str) -> str:
        """Normalize tablet or vessel ID strings across formats (e.g. HT 85 -> HT_085)."""
        clean = identifier.strip().replace(" ", "_").upper()
        # Handle HT 85 -> HT_085
        match = re.match(r"^([A-Z]+)[_\s]*0*([0-9]+)$", clean)
        if match:
            prefix, num = match.groups()
            return f"{prefix}_{int(num):03d}"
        return clean

    def prepare_recitation(
        self,
        id_str: str,
        raw_text: str,
        site: str = "Unknown",
        carrier: str = "Epigraphic Carrier",
        genre: str = "VOTIVE_LIBATION",
        gorila_ref: str = "",
        dating: str = "LM I (c. 1600–1450 BCE)",
        tempo_scale: float = 1.0,
        base_pitch_hz: float = 130.0,
    ) -> RecitationPackage:
        """Construct a complete RecitationPackage for a given text."""
        # 1. Phonetic transcription and synthesis schedule
        ipa_str, sign_profiles = self.acoustic.transcribe_to_ipa(raw_text)
        schedule = self.acoustic.generate_synthesis_schedule(
            raw_text, base_pitch_hz=base_pitch_hz, tempo_scale=tempo_scale
        )

        # 2. Prosodic scansion
        meter_report = self.prosody.scan_inscription(raw_text, id=id_str, site=site, carrier=carrier)

        # 3. Align syllables with meter and acoustic schedules
        # Flatten all scanned syllables
        all_scanned_sylls: List[MoraicSyllable] = [
            syll for w in meter_report.words for syll in w.syllables
        ]

        recitation_syllables: List[RecitationSyllable] = []
        n_items = min(len(schedule), len(sign_profiles))

        confidences = []
        for i in range(n_items):
            sched_item = schedule[i]
            prof = sign_profiles[i]
            confidences.append(prof.confidence_score)

            weight = 1
            symbol = "˘"
            if i < len(all_scanned_sylls):
                weight = all_scanned_sylls[i].weight
                symbol = all_scanned_sylls[i].symbol

            recitation_syllables.append(
                RecitationSyllable(
                    syllable=prof.transliteration,
                    sign_code=prof.sign_code,
                    ipa=prof.ipa_realization,
                    confidence_tier=prof.confidence_tier,
                    confidence_score=prof.confidence_score,
                    weight=weight,
                    symbol=symbol,
                    f0_start_hz=sched_item.f0_start_hz,
                    duration_ms=sched_item.duration_ms,
                    formant_f1=prof.formant_f1,
                    formant_f2=prof.formant_f2,
                    formant_f3=prof.formant_f3,
                    consonant_manner=prof.consonant_manner,
                    is_word_terminal=sched_item.is_word_terminal,
                    notes=prof.notes,
                )
            )

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.85

        return RecitationPackage(
            id=id_str,
            genre=genre,
            site=site,
            carrier=carrier,
            raw_text=raw_text,
            ipa_text=ipa_str,
            scansion_str=meter_report.scansion_line,
            dominant_foot=meter_report.dominant_foot,
            total_morae=meter_report.total_morae,
            mean_confidence=mean_conf,
            gorila_ref=gorila_ref,
            dating=dating,
            syllables=recitation_syllables,
            synthesis_schedule=[s.to_dict() for s in schedule],
        )

    def get_recitation_by_id(self, target_id: str) -> Optional[RecitationPackage]:
        """Look up and generate a recitation package for an inscription by ID."""
        norm_target = self._normalize_id(target_id)

        # 1. Check curated flagship inscriptions first
        for item in FLAGSHIP_INSCRIPTIONS:
            if self._normalize_id(item["id"]) == norm_target:
                return self.prepare_recitation(
                    id_str=item["id"],
                    raw_text=item["raw_text"],
                    site=item["site"],
                    carrier=item["carrier"],
                    genre=item["genre"],
                    gorila_ref=item.get("gorila_ref", ""),
                    dating=item.get("dating", "LM I (c. 1600–1450 BCE)"),
                )

        # 2. Check libation vessels corpus
        for v in self.libations.vessels:
            if self._normalize_id(v.id) == norm_target:
                return self.prepare_recitation(
                    id_str=v.id,
                    raw_text=v.transcription_raw,
                    site=v.site,
                    carrier=v.vessel_type or "Stone Libation Vessel",
                    genre="VOTIVE_LIBATION",
                    gorila_ref=f"GORILA 4, {v.id}",
                    dating="MM III – LM I (c. 1650–1450 BCE)",
                )

        # 3. Check tablet ledgers
        for t in load_all_tablets():
            if self._normalize_id(t["id"]) == norm_target:
                tokens = []
                for it in t.get("items", []):
                    tokens.append(it["entry_header"])
                    comm = it.get("commodity")
                    if comm:
                        tokens.append(comm)
                    amt = it.get("integer_amount")
                    if amt:
                        tokens.append(str(amt))
                kuro = t.get("stated_kuro")
                if kuro:
                    tokens.append("KU-RO")
                    if kuro.get("integer_amount"):
                        tokens.append(str(kuro["integer_amount"]))

                raw_text = " ".join(tokens)
                return self.prepare_recitation(
                    id_str=t["id"],
                    raw_text=raw_text,
                    site=t.get("site", "Crete"),
                    carrier=f"Clay Tablet ({t.get('date_range', 'LM IB')})",
                    genre="ADMINISTRATIVE_LEDGER",
                    gorila_ref=f"GORILA {t.get('gorila_vol', 1)}, {t['id']}",
                    dating=t.get("date_range", "LM IB (c. 1450 BCE)"),
                )

        return None

    def get_curated_recitations(self) -> List[RecitationPackage]:
        """Return all curated flagship recitation packages."""
        packages: List[RecitationPackage] = []
        for item in FLAGSHIP_INSCRIPTIONS:
            pkg = self.prepare_recitation(
                id_str=item["id"],
                raw_text=item["raw_text"],
                site=item["site"],
                carrier=item["carrier"],
                genre=item["genre"],
                gorila_ref=item.get("gorila_ref", ""),
                dating=item.get("dating", "LM I (c. 1600–1450 BCE)"),
            )
            packages.append(pkg)
        return packages
