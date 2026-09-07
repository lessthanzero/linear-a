"""Linear A Ideographic Ligature and Compound Sign Decomposition Engine.

Models composite signs where base ideograms (grain, oil, wine, vessels)
fuse with phonetic syllabograms or fractional modifiers in GORILA inscriptions.
Strictly adheres to Evidence Tier E3 (Numerical and Commoditized Accounting).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from linear_a.corpus.loader import get_default_corpus_dir


@dataclass
class MinoanLigature:
    """A composite Linear A sign with base ideogram and modifier."""
    id: str
    notation: str
    base_ideogram: str
    base_commodity: str
    base_name: str
    modifier_sign: str
    modifier_reading: str
    modifier_type: str  # "syllabogram" or "fraction"
    frequency_gorila: int
    findspots: List[str]
    epigraphic_class: str
    interpretation_hypothesis: str
    confidence_tier: str


@dataclass
class LigatureDecompositionReport:
    """Statistical summary of ideographic ligatures across Minoan archives."""
    total_ligatures_cataloged: int
    commodity_distribution: Dict[str, int]
    modifier_type_breakdown: Dict[str, int]
    regional_distribution: Dict[str, int]
    most_frequent_ligatures: List[Dict[str, str]]
    summary: str


class LigatureEngine:
    """Analyzes and decomposes Minoan composite ideograms and ligatures."""

    def __init__(self, catalog_path: Optional[Path] = None):
        base_path = catalog_path or (get_default_corpus_dir() / "palaeography" / "ligatures.yaml")
        self.ligatures: List[MinoanLigature] = []
        self._lookup: Dict[str, MinoanLigature] = {}

        if base_path.exists():
            with open(base_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            for item in raw.get("ligatures", []):
                lig = MinoanLigature(
                    id=item["id"],
                    notation=item["notation"],
                    base_ideogram=item["base_ideogram"],
                    base_commodity=item["base_commodity"],
                    base_name=item["base_name"],
                    modifier_sign=item["modifier_sign"],
                    modifier_reading=item["modifier_reading"],
                    modifier_type=item["modifier_type"],
                    frequency_gorila=item.get("frequency_gorila", 1),
                    findspots=item.get("findspots", []),
                    epigraphic_class=item.get("epigraphic_class", ""),
                    interpretation_hypothesis=item.get("interpretation_hypothesis", ""),
                    confidence_tier=item.get("confidence_tier", "E3"),
                )
                self.ligatures.append(lig)
                self._lookup[lig.notation] = lig
                self._lookup[lig.id] = lig

    def get_ligature(self, notation_or_id: str) -> Optional[MinoanLigature]:
        """Look up a ligature by its standard notation (e.g. 'OLE+U') or ID."""
        return self._lookup.get(notation_or_id)

    def is_ligature(self, token: str) -> bool:
        """Check if a token notation represents a composite ligature."""
        return "+" in token or token in self._lookup

    def decompose(self, notation: str) -> Optional[Dict[str, str]]:
        """Decompose a notation string into its base and modifier components."""
        if notation in self._lookup:
            lig = self._lookup[notation]
            return {
                "notation": lig.notation,
                "base_ideogram": lig.base_ideogram,
                "base_name": lig.base_name,
                "modifier": lig.modifier_reading,
                "modifier_type": lig.modifier_type,
                "interpretation": lig.interpretation_hypothesis,
            }
        if "+" in notation:
            parts = notation.split("+")
            return {
                "notation": notation,
                "base_ideogram": parts[0],
                "base_name": f"Base {parts[0]}",
                "modifier": parts[1] if len(parts) > 1 else "",
                "modifier_type": "unknown_modifier",
                "interpretation": "Ad-hoc composite ligature",
            }
        return None

    def analyze_corpus(self) -> LigatureDecompositionReport:
        """Compute structural breakdown of the ligature catalog."""
        comm_dist: Dict[str, int] = {}
        mod_dist: Dict[str, int] = {}
        reg_dist: Dict[str, int] = {}

        for lig in self.ligatures:
            comm_dist[lig.base_commodity] = comm_dist.get(lig.base_commodity, 0) + lig.frequency_gorila
            mod_dist[lig.modifier_type] = mod_dist.get(lig.modifier_type, 0) + 1
            for site in lig.findspots:
                reg_dist[site] = reg_dist.get(site, 0) + 1

        sorted_freq = sorted(self.ligatures, key=lambda l: l.frequency_gorila, reverse=True)
        top_ligs = [
            {
                "notation": l.notation,
                "base": l.base_name,
                "modifier": l.modifier_reading,
                "frequency": str(l.frequency_gorila),
                "interpretation": l.interpretation_hypothesis,
            }
            for l in sorted_freq[:5]
        ]

        summary = (
            f"Cataloged {len(self.ligatures)} composite ligatures across {len(reg_dist)} regional sites. "
            f"Dominant base commodities: {', '.join(f'{k} ({v})' for k, v in sorted(comm_dist.items(), key=lambda x: x[1], reverse=True))}. "
            f"Modifiers: {mod_dist.get('syllabogram', 0)} syllabic, {mod_dist.get('fraction', 0)} fractional."
        )

        return LigatureDecompositionReport(
            total_ligatures_cataloged=len(self.ligatures),
            commodity_distribution=comm_dist,
            modifier_type_breakdown=mod_dist,
            regional_distribution=reg_dist,
            most_frequent_ligatures=top_ligs,
            summary=summary,
        )
