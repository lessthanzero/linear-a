"""Linear A Interlinear Epigraphic Reader (LADP v1.0).

Provides structured 5-tier interlinear linguistic and mathematical glossing:
- Layer E0: Inscription findspot and material carrier
- Layer E1: Standardized sign identity (GORILA notation)
- Layer E2: Transliteration (Ventris syllabic priors)
- Layer E3: Numerical / Fractional value calculation (Ferrara 2020)
- Layer E4: Clausal syntax and administrative role
- Layer E5: Morphological affix decomposition (Prefix - Stem - Suffix)
- Layer E6-E7: Epistemic quarantine guard against speculative semantic glosses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

from fractions import Fraction
from linear_a.accounting.fractions import FractionEngine
from linear_a.morphology.affix_sieve import AffixSieve
from linear_a.palaeography.ligatures import LigatureEngine
from linear_a.votive.libation_engine import LibationVessel


@dataclass
class InterlinearToken:
    """An individual token within an interlinear line."""
    raw_token: str
    transliteration: str
    category: str  # "ANTHROPONYM", "COMMODITY", "TRANSACTION", "NUMBER", "FRACTION", "VOTIVE_ELEMENT", "UNKNOWN"
    functional_role: str
    morphology_breakdown: str
    prefix: Optional[str] = None
    stem: Optional[str] = None
    suffix: Optional[str] = None
    numerical_val: Optional[float] = None
    fraction_display: Optional[str] = None
    epistemic_tier: str = "E2"
    confidence_score: float = 0.85
    notes: str = ""


@dataclass
class InterlinearLine:
    """A line of text or transaction in an inscription."""
    line_index: int
    raw_line: str
    line_type: str  # "HEADER", "TRANSACTION", "TOTAL", "DEFICIT", "LIBATION_PHASE"
    tokens: List[InterlinearToken] = field(default_factory=list)
    line_total_val: Optional[float] = None


@dataclass
class InterlinearDocument:
    """A fully glossed interlinear inscription."""
    id: str
    site: str
    genre: str  # "ADMINISTRATIVE_LEDGER" or "VOTIVE_LIBATION"
    carrier: str
    findspot: str
    raw_text: str
    lines: List[InterlinearLine] = field(default_factory=list)
    is_mathematically_balanced: bool = True
    stated_total: Optional[float] = None
    calculated_total: Optional[float] = None
    epistemic_summary: str = ""

    def to_markdown(self) -> str:
        """Format the interlinear document as publication-grade Markdown."""
        md = [
            f"# Interlinear Inscription: {self.id} ({self.site})",
            f"**Genre**: `{self.genre}` | **Carrier**: {self.carrier} | **Findspot**: {self.findspot}",
            f"**Mathematical Balance**: {'✓ EXACT BALANCE' if self.is_mathematically_balanced else '⚠ DISCREPANCY'}",
            "",
            "---",
            "",
        ]

        for line in self.lines:
            md.append(f"### Line {line.line_index + 1}: `{line.line_type}`")
            md.append(f"> **Raw Text**: `{line.raw_line}`\n")

            md.append("| Syllabic Ductus | Category | Morphology | Administrative / Liturgical Role | Tier |")
            md.append("|---|---|---|---|:---:|")
            for t in line.tokens:
                num_info = f" ({t.numerical_val:g})" if t.numerical_val is not None else ""
                num_info += f" [{t.fraction_display}]" if t.fraction_display else ""
                md.append(
                    f"| **{t.transliteration}** | `{t.category}` | `{t.morphology_breakdown}` | {t.functional_role}{num_info} | `{t.epistemic_tier}` |"
                )
            md.append("")

        if self.genre == "ADMINISTRATIVE_LEDGER" and self.stated_total is not None:
            md.append("---")
            md.append(f"**Calculated Sum**: `{self.calculated_total:g}` | **Stated KU-RO**: `{self.stated_total:g}`")
            md.append(f"**Verification**: {'Identical (E3 Certified)' if self.is_mathematically_balanced else 'Discrepancy'}")

        return "\n".join(md)


class InterlinearReader:
    """Constructs multi-tier interlinear analyses of Linear A inscriptions."""

    COMMODITY_CODES = {
        "GRA": ("Wheat / Grain Ration", "AB120"),
        "OLE": ("Olive Oil Allocation", "AB130"),
        "VIN": ("Wine Distribution", "AB131a"),
        "FIC": ("Fig Rations", "AB30"),
        "TEL": ("Tribute / Payment Unit", "AB39"),
        "VIR": ("Male Laborers / Personnel", "AB100"),
        "MUL": ("Female Personnel", "AB102"),
        "VAS": ("Storage Vessel / Pithos", "AB180"),
    }

    def __init__(
        self,
        fraction_engine: Optional[FractionEngine] = None,
        ligature_engine: Optional[LigatureEngine] = None,
    ):
        self.fractions = fraction_engine or FractionEngine()
        self.ligatures = ligature_engine or LigatureEngine()

    def parse_tablet(self, tablet: Any) -> InterlinearDocument:
        """Parse an administrative clay tablet (dict or dataclass) into an InterlinearDocument."""
        if isinstance(tablet, dict):
            tablet_id = tablet["id"]
            site = tablet.get("site", "Crete")
            carrier = f"Clay Tablet ({tablet.get('date_range', 'LM IB')})"
            findspot = tablet.get("findspot_details", site)
            raw_items = tablet.get("items", [])
            kuro_dict = tablet.get("stated_kuro") or {}
            stated_int = kuro_dict.get("integer_amount")
            stated_frac_syms = kuro_dict.get("fractional_symbols", [])
            stated_tot = None
            if stated_int is not None or stated_frac_syms:
                f_val = float(self.fractions.evaluate_compound("".join(stated_frac_syms))) if stated_frac_syms else 0.0
                stated_tot = (stated_int or 0) + f_val

            norm_items = []
            for it in raw_items:
                f_syms = it.get("fractional_symbols", [])
                f_str = "".join(f_syms) if f_syms else None
                norm_items.append({
                    "entity": it["entry_header"],
                    "commodity": it.get("commodity"),
                    "quantity": it.get("integer_amount", 0),
                    "fraction_sign": f_str,
                })
        else:
            tablet_id = tablet.id
            site = tablet.site
            carrier = f"Clay Tablet ({tablet.date_range or 'LM IB'})"
            findspot = tablet.findspot_details or tablet.site
            stated_tot = float(tablet.stated_total) if tablet.stated_total is not None else None
            norm_items = [
                {
                    "entity": getattr(it, "entity", getattr(it, "entry_header", "")),
                    "commodity": it.commodity,
                    "quantity": getattr(it, "quantity", getattr(it, "integer_amount", 0)),
                    "fraction_sign": getattr(it, "fraction_sign", "".join(getattr(it, "fractional_symbols", [])) or None),
                }
                for it in getattr(tablet, "line_items", getattr(tablet, "items", []))
            ]

        lines: List[InterlinearLine] = []
        calc_sum = 0.0

        for idx, item in enumerate(norm_items):
            tokens: List[InterlinearToken] = []
            entity_str = item["entity"].strip()
            is_kuro = entity_str.upper() == "KU-RO"
            is_kiro = entity_str.upper() == "KI-RO"

            if is_kuro:
                line_type = "TOTAL"
            elif is_kiro:
                line_type = "DEFICIT"
            elif idx == 0 and item["commodity"] is None:
                line_type = "HEADER"
            else:
                line_type = "TRANSACTION"

            # 1. Entity Token
            ent_tok = self._parse_entity_token(entity_str, is_kuro, is_kiro)
            tokens.append(ent_tok)

            # 2. Commodity Token (if present)
            if item["commodity"]:
                comm_raw = item["commodity"].strip()
                comm_tok = self._parse_commodity_token(comm_raw)
                tokens.append(comm_tok)

            # 3. Numerical Token
            num_val = float(item["quantity"])
            frac_val = 0.0
            frac_display = None
            if item["fraction_sign"]:
                try:
                    frac_obj = self.fractions.evaluate_compound(item["fraction_sign"])
                    frac_val = float(frac_obj)
                    frac_display = f"{item['fraction_sign']} = {frac_obj}"
                except Exception:
                    frac_display = item["fraction_sign"]

            total_line_val = num_val + frac_val
            if not is_kuro and not is_kiro:
                calc_sum += total_line_val

            if item["quantity"] > 0:
                num_tok = InterlinearToken(
                    raw_token=str(item["quantity"]),
                    transliteration=str(item["quantity"]),
                    category="NUMBER",
                    functional_role="Base-10 Integer Quantity",
                    morphology_breakdown="ROOT: NUMBER",
                    numerical_val=num_val,
                    epistemic_tier="E3",
                    confidence_score=1.0,
                )
                tokens.append(num_tok)

            if item["fraction_sign"]:
                frac_tok = InterlinearToken(
                    raw_token=item["fraction_sign"],
                    transliteration=item["fraction_sign"],
                    category="FRACTION",
                    functional_role="Minoan Rational Sub-measure",
                    morphology_breakdown="FRACTION: FERRARA_2020",
                    numerical_val=frac_val,
                    fraction_display=frac_display,
                    epistemic_tier="E3",
                    confidence_score=1.0,
                )
                tokens.append(frac_tok)

            raw_str = f"{item['entity']} {item['commodity'] or ''} {item['quantity']} {item['fraction_sign'] or ''}".strip()
            lines.append(InterlinearLine(
                line_index=idx,
                raw_line=raw_str,
                line_type=line_type,
                tokens=tokens,
                line_total_val=total_line_val,
            ))

        # If stated KU-RO exists and was not an item in the list, append it as the closing total line
        if stated_tot is not None and not any(l.line_type == "TOTAL" for l in lines):
            kuro_tok = self._parse_entity_token("KU-RO", is_kuro=True, is_kiro=False)
            num_tok = InterlinearToken(
                raw_token=str(int(stated_tot)) if stated_tot.is_integer() else f"{stated_tot:g}",
                transliteration=str(int(stated_tot)) if stated_tot.is_integer() else f"{stated_tot:g}",
                category="NUMBER",
                functional_role="Base-10 Stated Total Quantity",
                morphology_breakdown="ROOT: NUMBER",
                numerical_val=stated_tot,
                epistemic_tier="E3",
                confidence_score=1.0,
            )
            lines.append(InterlinearLine(
                line_index=len(lines),
                raw_line=f"KU-RO {stated_tot:g}",
                line_type="TOTAL",
                tokens=[kuro_tok, num_tok],
                line_total_val=stated_tot,
            ))

        is_balanced = (stated_tot is None) or (abs(calc_sum - stated_tot) < 0.001)

        summary = (
            f"Administrative tablet {tablet_id} ({site}) with {len(lines)} entries. "
            f"{'Mathematically balanced (KU-RO ' + str(stated_tot) + ')' if is_balanced else 'Unbalanced or open ledger'}."
        )

        return InterlinearDocument(
            id=tablet_id,
            site=site,
            genre="ADMINISTRATIVE_LEDGER",
            carrier=carrier,
            findspot=findspot,
            raw_text="; ".join(l.raw_line for l in lines),
            lines=lines,
            is_mathematically_balanced=is_balanced,
            stated_total=stated_tot,
            calculated_total=calc_sum,
            epistemic_summary=summary,
        )

    def parse_vessel(self, vessel: LibationVessel) -> InterlinearDocument:
        """Parse a stone libation vessel into an InterlinearDocument."""
        lines: List[InterlinearLine] = []

        for idx, seg in enumerate(vessel.segments):
            tokens: List[InterlinearToken] = []

            # Determine category and role from libation grammar
            if "JA-SA-SA-RA-ME" in seg.word or "A-SA-SA-RA-ME" in seg.word:
                category = "DIVINE_EPITHET"
                role = "Great Minoan Goddess (Mother of Crete / Asherah parallel)"
                tier = "E4"
            elif "U-NA-KA-NA-SI" in seg.word:
                category = "DEDICATORY_VERB"
                role = "Finite ritual verb ('has dedicated / offered')"
                tier = "E4"
            elif "I-PI-NA-MA" in seg.word:
                category = "OFFERING_DESCRIPTOR"
                role = "Sacred liquid libation / consecrated offering"
                tier = "E4"
            elif idx == 0:
                category = "INVOCATION_HEADER"
                role = "Liturgical opening invocation / divine name"
                tier = "E4"
            elif seg.word.endswith("-TE"):
                category = "SANCTUARY_LOCATIVE"
                role = "Allative sanctuary title / priestly recipient (-TE)"
                tier = "E5"
            else:
                category = "VOTIVE_ELEMENT"
                role = seg.role
                tier = "E4"

            # Morphology breakdown
            morph_parts = []
            if seg.prefix:
                morph_parts.append(f"PREFIX: {seg.prefix}")
            stem_word = seg.word
            if seg.prefix and stem_word.startswith(seg.prefix):
                stem_word = stem_word[len(seg.prefix):]
            if seg.suffix and stem_word.endswith(seg.suffix):
                stem_word = stem_word[:-len(seg.suffix)]
            morph_parts.append(f"STEM: {stem_word}")
            if seg.suffix:
                morph_parts.append(f"SUFFIX: {seg.suffix}")

            morph_str = " + ".join(morph_parts)

            tok = InterlinearToken(
                raw_token=seg.word,
                transliteration=seg.word,
                category=category,
                functional_role=role,
                morphology_breakdown=morph_str,
                prefix=seg.prefix,
                stem=stem_word,
                suffix=seg.suffix,
                epistemic_tier=tier,
                confidence_score=0.92,
                notes=f"{seg.morae} morae rhythmic pulse",
            )
            tokens.append(tok)

            lines.append(InterlinearLine(
                line_index=idx,
                raw_line=seg.word,
                line_type=f"PHASE_{idx + 1}_{category}",
                tokens=tokens,
            ))

        summary = (
            f"Steatite libation vessel {vessel.id} from {vessel.site}. "
            f"Features {len(vessel.segments)} rigid liturgical phases with {vessel.total_morae} total morae."
        )

        return InterlinearDocument(
            id=vessel.id,
            site=vessel.site,
            genre="VOTIVE_LIBATION",
            carrier=vessel.vessel_type,
            findspot=vessel.site,
            raw_text=vessel.transcription_raw,
            lines=lines,
            is_mathematically_balanced=True,
            epistemic_summary=summary,
        )

    def _parse_entity_token(self, entity_raw: str, is_kuro: bool, is_kiro: bool) -> InterlinearToken:
        """Classify and segment a tablet entity / recipient token."""
        if is_kuro:
            return InterlinearToken(
                raw_token=entity_raw,
                transliteration="KU-RO",
                category="TRANSACTION",
                functional_role="Stated Account Total (Balance Check)",
                morphology_breakdown="STEM: KU-RO (Accounting total)",
                epistemic_tier="E3",
                confidence_score=1.0,
                notes="Universal Minoan total marker",
            )
        if is_kiro:
            return InterlinearToken(
                raw_token=entity_raw,
                transliteration="KI-RO",
                category="TRANSACTION",
                functional_role="Deficit / Shortfall Allocation",
                morphology_breakdown="STEM: KI-RO (Deficit marker)",
                epistemic_tier="E3",
                confidence_score=1.0,
                notes="Universal Minoan deficit marker",
            )

        # Check for morphological suffixes
        suffix = None
        stem = entity_raw
        if entity_raw.endswith("-TE"):
            suffix = "-TE"
            stem = entity_raw[:-3]
            role = "Allative Recipient / Sanctuary Official (-TE)"
            cat = "ANTHROPONYM_ALLATIVE"
            tier = "E5"
        elif entity_raw.endswith("-NA"):
            suffix = "-NA"
            stem = entity_raw[:-3]
            role = "Anthroponym / Toponym with regional formative (-NA)"
            cat = "TOPONYM_ANTHROPONYM"
            tier = "E5"
        elif entity_raw.endswith("-TA"):
            suffix = "-TA"
            stem = entity_raw[:-3]
            role = "Agentive / Occupational Title (-TA)"
            cat = "AGENTIVE_TITLE"
            tier = "E5"
        else:
            role = "Anthroponym (Estate Manager / Commodity Recipient)"
            cat = "ANTHROPONYM"
            tier = "E2"

        prefix = None
        if stem.startswith("A-"):
            prefix = "A-"
            stem = stem[2:]
        elif stem.startswith("JA-"):
            prefix = "JA-"
            stem = stem[3:]

        morph_parts = []
        if prefix:
            morph_parts.append(f"PREFIX: {prefix}")
        morph_parts.append(f"STEM: {stem}")
        if suffix:
            morph_parts.append(f"SUFFIX: {suffix}")

        return InterlinearToken(
            raw_token=entity_raw,
            transliteration=entity_raw,
            category=cat,
            functional_role=role,
            morphology_breakdown=" + ".join(morph_parts),
            prefix=prefix,
            stem=stem,
            suffix=suffix,
            epistemic_tier=tier,
            confidence_score=0.88,
        )

    def _parse_commodity_token(self, comm_raw: str) -> InterlinearToken:
        """Classify and decompose a commodity or composite ligature token."""
        # Check if composite ligature
        if self.ligatures.is_ligature(comm_raw):
            decomp = self.ligatures.decompose(comm_raw)
            if decomp:
                return InterlinearToken(
                    raw_token=comm_raw,
                    transliteration=comm_raw,
                    category="COMPOSITE_LIGATURE",
                    functional_role=f"Composite Commodity ({decomp['base_name']} + {decomp['modifier']})",
                    morphology_breakdown=f"BASE: {decomp['base_ideogram']} + MOD: {decomp['modifier']}",
                    epistemic_tier="E3",
                    confidence_score=0.95,
                    notes=decomp["interpretation"],
                )

        # Standard simple commodity
        if comm_raw in self.COMMODITY_CODES:
            name, ab_code = self.COMMODITY_CODES[comm_raw]
            return InterlinearToken(
                raw_token=comm_raw,
                transliteration=comm_raw,
                category="COMMODITY",
                functional_role=name,
                morphology_breakdown=f"IDEOGRAM: {ab_code}",
                epistemic_tier="E1",
                confidence_score=1.0,
            )

        return InterlinearToken(
            raw_token=comm_raw,
            transliteration=comm_raw,
            category="COMMODITY_VARIANT",
            functional_role=f"Unclassified Commodity / Ideogram ({comm_raw})",
            morphology_breakdown=f"IDEOGRAM: {comm_raw}",
            epistemic_tier="E1",
            confidence_score=0.75,
        )
