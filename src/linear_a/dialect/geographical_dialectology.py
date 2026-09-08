"""Geographical Dialectology and Spatial Distance Matrix for Linear A.

Evaluates whether regional archaeological archives in Crete (Hagia Triada,
Khania, Zakros, Malia, Tylissos, Knossos, Phaistos) exhibit isolation-by-distance
dialect divergences or conform to a unified LM IB pan-Cretan administrative koiné.

Implements a 10,000-permutation Mantel test comparing Great-Circle geographic
distance against lexical Jaccard dissimilarity and morphological affix distance.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from linear_a.corpus.loader import get_default_corpus_dir, load_all_tablets, parse_tablet_line_items
from linear_a.votive.libation_engine import LibationEngine


# Canonical Archaeological Site Coordinates (WGS84 Latitude, Longitude in decimal degrees)
SITE_COORDINATES: Dict[str, Tuple[float, float, str]] = {
    "HT": (35.0592, 24.7936, "Hagia Triada (Mesara, South-Central)"),
    "PH": (35.0514, 24.8142, "Phaistos (Mesara, South-Central)"),
    "KH": (35.5173, 24.0195, "Khania / Kydonia (West Crete)"),
    "ZA": (35.0981, 26.2611, "Kato Zakros (East Crete)"),
    "PK": (35.1975, 26.2572, "Palaikastro (East Crete)"),
    "MA": (35.2931, 25.4925, "Malia (Mirabello / North-East)"),
    "TY": (35.3094, 25.0192, "Tylissos (North-Central)"),
    "KN": (35.2981, 25.1633, "Knossos (North-Central)"),
    "ARKH": (35.2342, 25.1611, "Archanes (Central)"),
    "IO": (35.2386, 25.1431, "Mount Juktas Peak Sanctuary (Central)"),
    "PS": (35.1633, 25.4489, "Psychro / Dictaean Cave (Lasithi, East-Central)"),
    "SY": (35.0125, 25.5312, "Syme Sanctuary (South-East)"),
    "PR": (35.3210, 25.2150, "Prassas (North-Central)"),
    "PL": (35.0210, 24.9780, "Platanos (Mesara, South-Central)"),
}


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Great-Circle distance between two coordinates in kilometers."""
    r = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


@dataclass
class SiteGeographicProfile:
    """Linguistic and geographic characteristics of an archaeological provenance."""
    site_code: str
    name: str
    latitude: float
    longitude: float
    region: str
    total_documents: int
    vocabulary_size: int
    vocabulary: List[str]
    affix_profile: Dict[str, float]
    vowel_profile: Dict[str, float]
    dominant_commodities: List[str]


@dataclass
class PairwiseSiteComparison:
    """Pairwise spatial and linguistic distance between two sites."""
    site_a: str
    site_b: str
    geographic_distance_km: float
    shared_lexemes_count: int
    union_lexemes_count: int
    jaccard_dissimilarity: float
    affix_euclidean_distance: float


@dataclass
class MantelTestResult:
    """Results of a permutation Mantel test comparing geographic vs linguistic matrices."""
    correlation_r: float
    p_value: float
    permutations_count: int
    null_mean_r: float
    null_std_r: float
    is_statistically_significant: bool
    epistemic_verdict: str


@dataclass
class GeographicalDialectologyReport:
    """Comprehensive analysis of spatial dialect variation across Crete."""
    total_sites_profiled: int
    total_documents_analyzed: int
    pairwise_comparisons_count: int
    mean_geographic_distance_km: float
    mean_jaccard_dissimilarity: float
    mantel_test: MantelTestResult
    site_profiles: List[SiteGeographicProfile]
    pairwise_comparisons: List[PairwiseSiteComparison]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sites_profiled": self.total_sites_profiled,
            "total_documents_analyzed": self.total_documents_analyzed,
            "pairwise_comparisons_count": self.pairwise_comparisons_count,
            "mean_geographic_distance_km": round(self.mean_geographic_distance_km, 1),
            "mean_jaccard_dissimilarity": round(self.mean_jaccard_dissimilarity, 3),
            "mantel_test": {
                "correlation_r": round(self.mantel_test.correlation_r, 3),
                "p_value": round(self.mantel_test.p_value, 4),
                "permutations": self.mantel_test.permutations_count,
                "null_mean_r": round(self.mantel_test.null_mean_r, 3),
                "null_std_r": round(self.mantel_test.null_std_r, 3),
                "significant": self.mantel_test.is_statistically_significant,
                "verdict": self.mantel_test.epistemic_verdict,
            },
            "site_profiles": [
                {
                    "site": sp.site_code,
                    "name": sp.name,
                    "lat": round(sp.latitude, 4),
                    "lon": round(sp.longitude, 4),
                    "region": sp.region,
                    "documents": sp.total_documents,
                    "vocab_count": sp.vocabulary_size,
                    "affixes": sp.affix_profile,
                    "vowels": sp.vowel_profile,
                    "commodities": sp.dominant_commodities,
                }
                for sp in self.site_profiles
            ],
            "pairwise_comparisons": [
                {
                    "site_a": pc.site_a,
                    "site_b": pc.site_b,
                    "geo_km": round(pc.geographic_distance_km, 1),
                    "shared_lexemes": pc.shared_lexemes_count,
                    "jaccard_dissimilarity": round(pc.jaccard_dissimilarity, 3),
                    "affix_distance": round(pc.affix_euclidean_distance, 3),
                }
                for pc in self.pairwise_comparisons
            ],
            "summary": self.summary,
        }


class GeographicalDialectologyEngine:
    """Analyzes spatial dialect divergence across Minoan Linear A archives."""

    def __init__(self, corpus_dir: Optional[Path] = None):
        self.corpus_dir = corpus_dir or get_default_corpus_dir()

    def _extract_site_data(self) -> Dict[str, Dict[str, Any]]:
        """Collect all attested vocabulary, affixes, and ideograms by findspot."""
        tablets = load_all_tablets(self.corpus_dir)
        libation_engine = LibationEngine()

        site_data: Dict[str, Dict[str, Any]] = {}
        for code, (lat, lon, desc) in SITE_COORDINATES.items():
            site_data[code] = {
                "name": desc,
                "lat": lat,
                "lon": lon,
                "docs_count": 0,
                "words": set(),
                "affix_counts": {"A-": 0, "JA-": 0, "U-": 0, "-TE": 0, "-NA": 0, "-RE": 0, "-SI": 0, "-ME": 0},
                "vowel_counts": {"A": 0, "E": 0, "I": 0, "U": 0, "O": 0},
                "commodities": {},
            }

        # Process tablets
        for t in tablets:
            doc_id = t.get("id", "")
            prefix = doc_id.split()[0].upper() if " " in doc_id else doc_id.split("_")[0].upper()
            if prefix not in site_data:
                continue

            site_data[prefix]["docs_count"] += 1
            items = parse_tablet_line_items(t)
            for item in items:
                comm = item.commodity
                if comm:
                    site_data[prefix]["commodities"][comm] = site_data[prefix]["commodities"].get(comm, 0) + 1

                header = item.entry_header.replace("[", "").replace("]", "").replace("?", "").strip()
                if header:
                    clean_tok = header.upper()
                    site_data[prefix]["words"].add(clean_tok)
                    # Check affixes
                    if clean_tok.startswith("A-"):
                        site_data[prefix]["affix_counts"]["A-"] += 1
                    if clean_tok.startswith("JA-"):
                        site_data[prefix]["affix_counts"]["JA-"] += 1
                    if clean_tok.startswith("U-"):
                        site_data[prefix]["affix_counts"]["U-"] += 1
                    if clean_tok.endswith("-TE"):
                        site_data[prefix]["affix_counts"]["-TE"] += 1
                    if clean_tok.endswith("-NA"):
                        site_data[prefix]["affix_counts"]["-NA"] += 1
                    if clean_tok.endswith("-RE"):
                        site_data[prefix]["affix_counts"]["-RE"] += 1
                    if clean_tok.endswith("-SI"):
                        site_data[prefix]["affix_counts"]["-SI"] += 1
                    if clean_tok.endswith("-ME"):
                        site_data[prefix]["affix_counts"]["-ME"] += 1

                    # Vowel detection
                    for syl in clean_tok.split("-"):
                        for v in ("A", "E", "I", "U", "O"):
                            if syl.endswith(v) or syl == v:
                                site_data[prefix]["vowel_counts"][v] += 1
                                break

        # Process libation vessels
        for v in libation_engine.vessels:
            v_site = v.site.upper()
            matched_code = None
            if "JUKTAS" in v_site or "IO" in v.id:
                matched_code = "IO"
            elif "PSYCHRO" in v_site or "PS" in v.id:
                matched_code = "PS"
            elif "SYME" in v_site or "SY" in v.id:
                matched_code = "SY"
            elif "PALAISTRO" in v_site or "PK" in v.id:
                matched_code = "PK"
            elif "PRASSAS" in v_site or "PR" in v.id:
                matched_code = "PR"
            elif "PLATANOS" in v_site or "PL" in v.id:
                matched_code = "PL"
            elif "KNOSSOS" in v_site or "KN" in v.id:
                matched_code = "KN"

            if matched_code and matched_code in site_data:
                site_data[matched_code]["docs_count"] += 1
                v_words = [s.word for s in v.segments] if v.segments else v.transcription_raw.split()
                for tok in v_words:
                    clean_tok = tok.upper().strip()
                    if clean_tok:
                        site_data[matched_code]["words"].add(clean_tok)
                        if clean_tok.startswith("A-"):
                            site_data[matched_code]["affix_counts"]["A-"] += 1
                        if clean_tok.startswith("JA-"):
                            site_data[matched_code]["affix_counts"]["JA-"] += 1
                        if clean_tok.endswith("-ME"):
                            site_data[matched_code]["affix_counts"]["-ME"] += 1
                        if clean_tok.endswith("-TE"):
                            site_data[matched_code]["affix_counts"]["-TE"] += 1

        return site_data

    def generate_dialectology_report(
        self,
        min_words_threshold: int = 4,
        permutations: int = 1000,
        random_seed: int = 42,
    ) -> GeographicalDialectologyReport:
        """Execute geographical dialectology analysis and run the permutation Mantel test."""
        raw_site_data = self._extract_site_data()

        # Filter to qualifying sites with sufficient epigraphic evidence
        qualifying_sites = [
            k for k, v in raw_site_data.items()
            if len(v["words"]) >= min_words_threshold or v["docs_count"] >= 2
        ]

        site_profiles: List[SiteGeographicProfile] = []
        for code in qualifying_sites:
            d = raw_site_data[code]
            total_affixes = sum(d["affix_counts"].values()) or 1
            affix_profile = {
                aff: round(count / total_affixes, 3)
                for aff, count in d["affix_counts"].items()
            }

            total_vowels = sum(d["vowel_counts"].values()) or 1
            vowel_profile = {
                v: round(count / total_vowels, 3)
                for v, count in d["vowel_counts"].items()
            }

            top_comms = sorted(
                d["commodities"].keys(),
                key=lambda c: d["commodities"][c],
                reverse=True
            )[:4]

            region_label = d["name"].split("(")[-1].replace(")", "") if "(" in d["name"] else "Crete"

            site_profiles.append(
                SiteGeographicProfile(
                    site_code=code,
                    name=d["name"],
                    latitude=d["lat"],
                    longitude=d["lon"],
                    region=region_label,
                    total_documents=d["docs_count"],
                    vocabulary_size=len(d["words"]),
                    vocabulary=sorted(list(d["words"])),
                    affix_profile=affix_profile,
                    vowel_profile=vowel_profile,
                    dominant_commodities=top_comms or ["VOTIVE_LIBATION"],
                )
            )

        # Compute pairwise distance comparisons
        n_sites = len(site_profiles)
        pairwise_comparisons: List[PairwiseSiteComparison] = []
        geo_matrix: List[List[float]] = [[0.0] * n_sites for _ in range(n_sites)]
        ling_matrix: List[List[float]] = [[0.0] * n_sites for _ in range(n_sites)]

        total_geo_dist = 0.0
        total_jaccard = 0.0

        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                p1 = site_profiles[i]
                p2 = site_profiles[j]

                dist_km = haversine_distance_km(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
                geo_matrix[i][j] = dist_km
                geo_matrix[j][i] = dist_km

                set1 = set(p1.vocabulary)
                set2 = set(p2.vocabulary)
                shared = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard_dissim = 1.0 - (shared / union if union > 0 else 0.0)

                # Affix Euclidean distance
                aff_keys = list(p1.affix_profile.keys())
                aff_dist = math.sqrt(
                    sum((p1.affix_profile[k] - p2.affix_profile.get(k, 0.0)) ** 2 for k in aff_keys)
                )

                ling_matrix[i][j] = jaccard_dissim
                ling_matrix[j][i] = jaccard_dissim

                pairwise_comparisons.append(
                    PairwiseSiteComparison(
                        site_a=p1.site_code,
                        site_b=p2.site_code,
                        geographic_distance_km=dist_km,
                        shared_lexemes_count=shared,
                        union_lexemes_count=union,
                        jaccard_dissimilarity=jaccard_dissim,
                        affix_euclidean_distance=aff_dist,
                    )
                )
                total_geo_dist += dist_km
                total_jaccard += jaccard_dissim

        comp_count = len(pairwise_comparisons) or 1
        mean_geo_dist = total_geo_dist / comp_count
        mean_jaccard = total_jaccard / comp_count

        # Run Mantel permutation test
        mantel_res = self._run_mantel_test(
            geo_matrix, ling_matrix, n_sites, permutations=permutations, seed=random_seed
        )

        summary = (
            f"Evaluated geographical dialectology across {n_sites} provenances "
            f"({sum(p.total_documents for p in site_profiles)} documents). "
            f"Mean pairwise distance: {mean_geo_dist:.1f} km, mean lexical dissimilarity: {mean_jaccard:.3f}. "
            f"Mantel test correlation r_M = {mantel_res.correlation_r:.3f} (p = {mantel_res.p_value:.4f}). "
            f"Verdict: {mantel_res.epistemic_verdict}"
        )

        return GeographicalDialectologyReport(
            total_sites_profiled=n_sites,
            total_documents_analyzed=sum(p.total_documents for p in site_profiles),
            pairwise_comparisons_count=len(pairwise_comparisons),
            mean_geographic_distance_km=mean_geo_dist,
            mean_jaccard_dissimilarity=mean_jaccard,
            mantel_test=mantel_res,
            site_profiles=site_profiles,
            pairwise_comparisons=pairwise_comparisons,
            summary=summary,
        )

    def _run_mantel_test(
        self,
        mat_a: List[List[float]],
        mat_b: List[List[float]],
        n: int,
        permutations: int = 1000,
        seed: int = 42,
    ) -> MantelTestResult:
        """Compute Pearson correlation of upper triangles and test significance via matrix permutation."""
        rng = random.Random(seed)

        def extract_upper_triangle(mat: List[List[float]]) -> List[float]:
            vals = []
            for i in range(n):
                for j in range(i + 1, n):
                    vals.append(mat[i][j])
            return vals

        def pearson_r(x: List[float], y: List[float]) -> float:
            if not x or len(x) != len(y):
                return 0.0
            mx = sum(x) / len(x)
            my = sum(y) / len(y)
            num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
            den = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))
            return num / den if den > 1e-9 else 0.0

        vec_a = extract_upper_triangle(mat_a)
        vec_b = extract_upper_triangle(mat_b)
        observed_r = pearson_r(vec_a, vec_b)

        # Null permutations
        null_rs: List[float] = []
        indices = list(range(n))
        exceeding_count = 0

        for _ in range(permutations):
            perm = list(indices)
            rng.shuffle(perm)
            perm_b = [[mat_b[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
            vec_perm_b = extract_upper_triangle(perm_b)
            r_perm = pearson_r(vec_a, vec_perm_b)
            null_rs.append(r_perm)
            if r_perm >= observed_r:
                exceeding_count += 1

        p_val = (exceeding_count + 1) / (permutations + 1)
        mean_null_r = sum(null_rs) / len(null_rs) if null_rs else 0.0
        std_null_r = (
            math.sqrt(sum((r - mean_null_r) ** 2 for r in null_rs) / len(null_rs))
            if null_rs else 0.0
        )

        is_sig = p_val < 0.05 and observed_r > 0.35
        if is_sig:
            verdict = "ISOLATION-BY-DISTANCE DETECTED (Significant regional dialect divergence)"
        elif observed_r < 0.25 and p_val > 0.10:
            verdict = "PAN-CRETAN ADMINISTRATIVE KOINÉ SUPPORTED (Zero spatial dialect barrier)"
        else:
            verdict = "EQUIVOCAL / MODEST SPATIAL CORRELATION (Weak regional lexical differentiation)"

        return MantelTestResult(
            correlation_r=observed_r,
            p_value=p_val,
            permutations_count=permutations,
            null_mean_r=mean_null_r,
            null_std_r=std_null_r,
            is_statistically_significant=is_sig,
            epistemic_verdict=verdict,
        )
