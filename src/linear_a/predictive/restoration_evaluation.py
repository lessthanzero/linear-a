"""Leakage-resistant evaluation for curated Linear A restoration hypotheses.

This module scores agreement with the repository's curated reference set. It
does not validate a decipherment or replace scholarly adjudication.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from linear_a.accounting.fractions import FractionEngine
from linear_a.corpus.loader import get_default_corpus_dir, load_all_tablets, parse_tablet_line_items
from linear_a.predictive.multilateral_solver import LacunaEntry
from linear_a.predictive.source_linked_benchmark import SourceLinkedBenchmark


@dataclass
class EvaluationMetric:
    method: str
    references: int
    attempted: int
    correct_top1: int
    correct_top3: int
    abstained: int

    @property
    def coverage_pct(self) -> float:
        return round(100 * self.attempted / self.references, 1) if self.references else 0.0

    @property
    def precision_top1_pct(self) -> float:
        return round(100 * self.correct_top1 / self.attempted, 1) if self.attempted else 0.0

    @property
    def top3_accuracy_pct(self) -> float:
        return round(100 * self.correct_top3 / self.attempted, 1) if self.attempted else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "references": self.references,
            "attempted": self.attempted,
            "abstained": self.abstained,
            "coverage_pct": self.coverage_pct,
            "precision_top1_pct": self.precision_top1_pct,
            "top3_accuracy_pct": self.top3_accuracy_pct,
        }


@dataclass
class RestorationEvaluationReport:
    reference_set_status: str
    template: EvaluationMetric
    phonotactic: EvaluationMetric
    arithmetic_controls: EvaluationMetric
    limitations: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_set_status": self.reference_set_status,
            "template": self.template.to_dict(),
            "phonotactic": self.phonotactic.to_dict(),
            "arithmetic_controls": self.arithmetic_controls.to_dict(),
            "limitations": self.limitations,
        }


@dataclass
class SourceLinkedBenchmarkReport:
    """Scores restricted to independently accepted benchmark entries."""

    benchmark_status: str
    total_entries: int
    status_counts: dict[str, int]
    template: EvaluationMetric
    phonotactic: EvaluationMetric
    limitations: str

    @property
    def scored_references(self) -> int:
        return self.status_counts["accepted"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_status": self.benchmark_status,
            "total_entries": self.total_entries,
            "status_counts": self.status_counts,
            "scored_references": self.scored_references,
            "template": self.template.to_dict(),
            "phonotactic": self.phonotactic.to_dict(),
            "limitations": self.limitations,
        }


class RestorationEvaluationEngine:
    """Evaluate restoration methods while holding out each target document."""

    LIMITATIONS = (
        "Results are agreement scores against a project-curated reference set. "
        "Each target document is excluded from its template and phonotactic training fold. "
        "Arithmetic controls are synthetic masks of fully legible curated ledger values. "
        "None of these scores establish a decipherment or scholarly consensus."
    )

    def __init__(self, corpus_dir: Path | None = None):
        self.corpus_dir = corpus_dir or get_default_corpus_dir()
        self.fractions = FractionEngine()
        self.catalog = self._load_catalog()
        self.reference_ids, self.reference_status = self._load_reference_set()
        self.source_linked_benchmark = SourceLinkedBenchmark(self.corpus_dir)

    def _load_catalog(self) -> list[LacunaEntry]:
        path = self.corpus_dir / "palaeography" / "lacunae_catalog.yaml"
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return [LacunaEntry(**item) for item in raw.get("lacunae", [])]

    def _load_reference_set(self) -> tuple[set[str], str]:
        path = self.corpus_dir / "palaeography" / "restoration_evaluation.yaml"
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        ids = {
            item["id"]
            for item in raw.get("annotations", [])
            if item.get("class") == "project_curated_restoration"
        }
        return ids, raw.get("status", "unknown")

    @staticmethod
    def _parts(token: str) -> list[str]:
        return token.replace("[", "").replace("]", "").split("-")

    def _references(self, genres: set[str] | None = None) -> list[LacunaEntry]:
        return [
            entry for entry in self.catalog
            if entry.id in self.reference_ids and (genres is None or entry.genre in genres)
        ]

    def _training_words(self, held_out_document: str, references: list[LacunaEntry] | None = None) -> list[str]:
        references = references if references is not None else self._references({"votive", "administrative", "toponymic"})
        return [
            entry.completed_word for entry in references
            if entry.document != held_out_document and "-" in entry.completed_word
        ]

    def _template_candidates(self, entry: LacunaEntry, references: list[LacunaEntry] | None = None) -> list[str]:
        masked = self._parts(entry.masked_token)
        candidates: set[str] = set()
        for word in self._training_words(entry.document, references):
            parts = self._parts(word)
            if len(parts) != len(masked) or entry.damaged_position >= len(parts):
                continue
            if all(index == entry.damaged_position or observed == candidate for index, (observed, candidate) in enumerate(zip(masked, parts))):
                candidates.add(parts[entry.damaged_position])
        return sorted(candidates)

    def _phonotactic_candidates(self, entry: LacunaEntry, references: list[LacunaEntry] | None = None) -> list[str]:
        masked = self._parts(entry.masked_token)
        position = entry.damaged_position
        if position >= len(masked):
            return []
        left = masked[position - 1] if position else None
        right = masked[position + 1] if position + 1 < len(masked) else None
        bigrams: dict[tuple[str, str], int] = {}
        signs: set[str] = set()
        for word in self._training_words(entry.document, references):
            parts = self._parts(word)
            signs.update(parts)
            for first, second in zip(parts, parts[1:]):
                bigrams[(first, second)] = bigrams.get((first, second), 0) + 1
        scored = []
        for sign in signs:
            score = (bigrams.get((left, sign), 0) if left else 0) + (bigrams.get((sign, right), 0) if right else 0)
            if score:
                scored.append((score, sign))
        return [sign for _, sign in sorted(scored, key=lambda item: (-item[0], item[1]))]

    def _evaluate_candidates(self, method: str, candidate_fn: Any, references: list[LacunaEntry] | None = None) -> EvaluationMetric:
        references = references if references is not None else self._references({"votive", "administrative", "toponymic"})
        attempted = correct_top1 = correct_top3 = 0
        for entry in references:
            candidates = candidate_fn(entry)
            if not candidates:
                continue
            attempted += 1
            correct_top1 += candidates[0] == entry.reconstructed_sign
            correct_top3 += entry.reconstructed_sign in candidates[:3]
        return EvaluationMetric(method, len(references), attempted, correct_top1, correct_top3, len(references) - attempted)

    def evaluate_arithmetic_controls(self) -> EvaluationMetric:
        references = attempted = correct = 0
        for tablet in load_all_tablets(self.corpus_dir):
            stated = tablet.get("stated_kuro", {})
            integer = stated.get("integer_amount")
            if integer is None:
                continue
            items = parse_tablet_line_items(tablet)
            if len(items) < 2:
                continue
            total = Fraction(integer, 1) + self.fractions.parse_fraction_symbols(stated.get("fractional_symbols", []))
            values = [Fraction(item.integer_amount, 1) + self.fractions.parse_fraction_symbols(item.fractional_symbols) for item in items]
            for index, value in enumerate(values):
                references += 1
                inferred = total - sum((other for other_index, other in enumerate(values) if other_index != index), Fraction(0, 1))
                if inferred == value:
                    attempted += 1
                    correct += 1
        return EvaluationMetric("exact-ledger-residual synthetic controls", references, attempted, correct, correct, references - attempted)

    def evaluate_exploratory(self) -> RestorationEvaluationReport:
        """Score the project-curated reference set; not independent validation."""
        return RestorationEvaluationReport(
            reference_set_status=self.reference_status,
            template=self._evaluate_candidates("leave-one-document-out template match", self._template_candidates),
            phonotactic=self._evaluate_candidates("leave-one-document-out bigram ranking", self._phonotactic_candidates),
            arithmetic_controls=self.evaluate_arithmetic_controls(),
            limitations=self.LIMITATIONS,
        )

    def evaluate(self, benchmark: SourceLinkedBenchmark | None = None) -> SourceLinkedBenchmarkReport:
        """Score only externally accepted source-linked benchmark records."""
        bench = benchmark or self.source_linked_benchmark
        accepted = [
            entry for entry in bench.accepted_catalog_entries()
            if entry.genre in {"votive", "administrative", "toponymic"}
        ]
        counts = bench.status_counts()
        limitations = (
            "Only externally accepted benchmark entries are scoreable. Unadjudicated, disputed, and rejected "
            "entries are excluded from every accuracy denominator. This citation-only bootstrap has no accepted entries."
            if counts.get("accepted", 0) == 0 else
            "Scored against externally accepted benchmark entries under leave-one-document-out cross-validation."
        )
        return SourceLinkedBenchmarkReport(
            benchmark_status=bench.metadata.get("status", "custom-adjudicated"),
            total_entries=len(bench.entries),
            status_counts=counts,
            template=self._evaluate_candidates(
                "accepted-document-held-out template match",
                lambda entry: self._template_candidates(entry, accepted),
                accepted,
            ),
            phonotactic=self._evaluate_candidates(
                "accepted-document-held-out bigram ranking",
                lambda entry: self._phonotactic_candidates(entry, accepted),
                accepted,
            ),
            limitations=limitations,
        )
