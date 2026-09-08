"""Paleographic Ligature Taxonomy & Scribal Distribution Engine (LADP v1.0).

Analyzes Linear A composite monograms, adjunct syllabic modifiers, and commodity ideogram
taxonomies across palatial scriptoria (Hagia Triada, Knossos, Khania, Zakros, Malia).
Correlates Linear A agricultural modifiers (GRA+X, OLE+X, VIN+X) with Linear B parallels.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from linear_a.corpus.loader import get_default_corpus_dir, load_all_tablets


@dataclass
class DecomposedLigature:
    """A decomposed Linear A composite sign or monogram."""
    ligature_id: str
    notation: str
    base_sign: str
    base_commodity: str
    base_name: str
    modifier_sign: str
    modifier_reading: str
    modifier_role: str  # "QUALITY_GRADE", "PROCESSING_UNGUENT", "SPECIES_VARIANT", "ACCOUNTING_DEFICIT"
    findspots: List[str]
    observed_frequency: int
    linear_b_parallel: Optional[str]
    interpretation: str


@dataclass
class ScriptoriumLigatureProfile:
    """Ligature and commodity repertoire for a specific palatial scriptorium."""
    site_name: str
    total_ligatures_attested: int
    dominant_commodities: List[Tuple[str, int]]
    unique_monograms: List[str]
    scribal_specialization: str


@dataclass
class LigatureTaxonomyReport:
    """Comprehensive taxonomy report across Linear A composite signs."""
    total_ligatures_cataloged: int
    total_instances_attested: int
    commodity_classes: Dict[str, int]
    decomposed_records: List[DecomposedLigature]
    scriptorium_profiles: List[ScriptoriumLigatureProfile]
    linear_b_concordances: List[Dict[str, str]]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ligatures_cataloged": self.total_ligatures_cataloged,
            "total_instances_attested": self.total_instances_attested,
            "commodity_classes": self.commodity_classes,
            "decomposed_records": [
                {
                    "notation": d.notation,
                    "base": d.base_name,
                    "modifier": d.modifier_reading,
                    "role": d.modifier_role,
                    "findspots": d.findspots,
                    "freq": d.observed_frequency,
                    "linear_b": d.linear_b_parallel,
                    "interpretation": d.interpretation,
                }
                for d in self.decomposed_records
            ],
            "scriptorium_profiles": [
                {
                    "site": p.site_name,
                    "total": p.total_ligatures_attested,
                    "commodities": [c[0] for c in p.dominant_commodities[:3]],
                    "specialization": p.scribal_specialization,
                }
                for p in self.scriptorium_profiles
            ],
            "summary": self.summary,
        }


class LigatureTaxonomyEngine:
    """Analyzes composite monograms, paleographic adjuncts, and scribal scriptorium repertoires."""

    def __init__(self, corpus_dir: Optional[Path] = None):
        self.corpus_dir = corpus_dir or get_default_corpus_dir()
        self.raw_ligatures = self._load_raw_ligatures()

    def _load_raw_ligatures(self) -> List[Dict[str, Any]]:
        path = self.corpus_dir / "palaeography" / "ligatures.yaml"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("ligatures", [])

    @staticmethod
    def _classify_modifier_role(base: str, mod: str) -> str:
        """Classify the functional role of the adjunct modifier."""
        if mod == "KI":
            return "ACCOUNTING_DEFICIT"  # Linked to KI-RO deficit records
        if base == "OLE":
            if mod in ("A", "DI", "MI"):
                return "PROCESSING_UNGUENT"  # Perfumed oil / unguent (Linear B a-re-pa)
            return "QUALITY_GRADE"  # First pressing, virgin oil (OLE+U)
        if base == "GRA":
            if mod in ("PA", "TE"):
                return "SPECIES_VARIANT"  # Barley / coarse grain vs wheat
            return "QUALITY_GRADE"  # Cleaned seed grain (GRA+QE)
        if base == "VIN":
            return "SPECIES_VARIANT"  # Wine variety / vintage year
        return "SYLLABIC_MONOGRAM"

    def decompose_all(self) -> List[DecomposedLigature]:
        """Decompose every cataloged ligature into base ideogram and adjunct modifier."""
        results: List[DecomposedLigature] = []

        # Known Linear B parallels
        lb_parallels = {
            "OLE+A": "Linear B OLE+A / a-re-pa (ἀλείφαρ, 'unguent oil')",
            "GRA+PA": "Linear B GRA+PA / pa-ka-na ('barley ration')",
            "OLE+U": "Linear B OLE+U / u-wo-we ('poured / unperfumed virgin oil')",
            "VIN+RA": "Linear B VIN+RA ('resinated / stored wine')",
            "RO+KE": "Linear B monogram RO+KE (textile / cloth allocation)",
        }

        for item in self.raw_ligatures:
            notation = item.get("notation", "")
            base_comm = item.get("base_commodity", "UNKNOWN")
            mod_read = item.get("modifier_reading", "")
            role = self._classify_modifier_role(base_comm, mod_read)

            parallel = lb_parallels.get(notation)

            results.append(DecomposedLigature(
                ligature_id=item.get("id", "LIG"),
                notation=notation,
                base_sign=item.get("base_ideogram", ""),
                base_commodity=base_comm,
                base_name=item.get("base_name", ""),
                modifier_sign=item.get("modifier_sign", ""),
                modifier_reading=mod_read,
                modifier_role=role,
                findspots=item.get("findspots", []),
                observed_frequency=item.get("frequency_gorila", 1),
                linear_b_parallel=parallel,
                interpretation=item.get("interpretation_hypothesis", ""),
            ))

        return results

    def build_scriptorium_profiles(self, records: List[DecomposedLigature]) -> List[ScriptoriumLigatureProfile]:
        """Map ligature distributions across archaeological scriptoria."""
        site_counts: Dict[str, Counter] = defaultdict(Counter)
        site_monograms: Dict[str, Set[str]] = defaultdict(set)

        for r in records:
            for site in r.findspots:
                site_counts[site][r.base_commodity] += r.observed_frequency
                site_monograms[site].add(r.notation)

        # Ingest tablet usages as well
        tablets = load_all_tablets(self.corpus_dir)
        for t in tablets:
            site = t.get("site", "Crete").replace(" ", "_")
            for item in t.get("items", []):
                comm = item.get("commodity", "")
                if "+" in comm:
                    base = comm.split("+")[0]
                    site_counts[site][base] += 1
                    site_monograms[site].add(comm)

        profiles: List[ScriptoriumLigatureProfile] = []
        for site, counts in site_counts.items():
            tot = sum(counts.values())
            top_comms = counts.most_common(3)
            monos = sorted(site_monograms[site])

            spec = "Multi-commodity central administrative archive"
            if top_comms and top_comms[0][0] == "OLE":
                spec = "Specialized liquid oil unguent scriptorium"
            elif top_comms and top_comms[0][0] == "GRA":
                spec = "Grain rationing and labor allocation archive"
            elif top_comms and top_comms[0][0] == "VIN":
                spec = "Viticulture and wine distribution archive"

            profiles.append(ScriptoriumLigatureProfile(
                site_name=site.replace("_", " "),
                total_ligatures_attested=tot,
                dominant_commodities=top_comms,
                unique_monograms=monos,
                scribal_specialization=spec,
            ))

        profiles.sort(key=lambda p: p.total_ligatures_attested, reverse=True)
        return profiles

    def generate_taxonomy_report(self) -> LigatureTaxonomyReport:
        """Compile complete taxonomy and scriptorium report."""
        decomposed = self.decompose_all()
        tot_inst = sum(d.observed_frequency for d in decomposed)

        comm_classes: Counter = Counter(d.base_commodity for d in decomposed)
        profiles = self.build_scriptorium_profiles(decomposed)

        parallels = [
            {"linear_a": d.notation, "parallel": d.linear_b_parallel, "role": d.modifier_role}
            for d in decomposed
            if d.linear_b_parallel
        ]

        summary = (
            f"Cataloged {len(decomposed)} unique composite monograms ({tot_inst} total epigraphic instances) "
            f"across {len(profiles)} regional scriptoria. Dominant commodity classes: "
            f"Oil (OLE: {comm_classes.get('OLE', 0)}), Grain (GRA: {comm_classes.get('GRA', 0)}), "
            f"Wine (VIN: {comm_classes.get('VIN', 0)}). Identified {len(parallels)} exact Mycenaean Linear B "
            f"ideographic parallels demonstrating administrative and technological continuity (e.g. OLE+A unguent)."
        )

        return LigatureTaxonomyReport(
            total_ligatures_cataloged=len(decomposed),
            total_instances_attested=tot_inst,
            commodity_classes=dict(comm_classes),
            decomposed_records=decomposed,
            scriptorium_profiles=profiles,
            linear_b_concordances=parallels,
            summary=summary,
        )
