"""Published-pair benchmark for Linear A / Linear B lexical comparisons.

This module tests the rarity of a pre-registered set of *published* exact
syllabic-form comparisons. It does not infer cognacy, translation, language
relationship, or a source language for any form.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any, Iterable

import yaml

from linear_a.corpus.loader import get_default_corpus_dir
from linear_a.predictive.corpus_census import CorpusLacunaeCensusEngine


class CorrespondenceBenchmarkError(ValueError):
    """Raised when the publication-linked benchmark is malformed."""


@dataclass(frozen=True)
class PublishedPair:
    """A pre-registered lexical comparison attributed to a publication."""

    id: str
    linear_a_form: str
    linear_b_form: str
    lexical_category: str
    source_status: str
    pair_citation: dict[str, str]
    linear_b_attestations: tuple[str, ...]
    limitation: str


@dataclass(frozen=True)
class SubstratumBenchmarkReport:
    """Result of an exact-form published-pair permutation test."""

    benchmark_id: str
    target_entries: int
    qualified_entries: int
    qualifying_shortfall: int
    observed_exact_matches: int
    null_mean_matches: float | None
    empirical_p_value: float | None
    surrogates: int
    source_snapshot: dict[str, Any] | None
    limitations: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "target_entries": self.target_entries,
            "qualified_entries": self.qualified_entries,
            "qualifying_shortfall": self.qualifying_shortfall,
            "observed_exact_matches": self.observed_exact_matches,
            "null_mean_matches": self.null_mean_matches,
            "empirical_p_value": self.empirical_p_value,
            "surrogates": self.surrogates,
            "source_snapshot": self.source_snapshot,
            "limitations": self.limitations,
        }


def normalize_syllabic_form(value: str) -> str:
    """Normalize a conventional transcription for pre-registered exact matching."""
    pieces = []
    for raw_piece in value.upper().split("-"):
        piece = re.sub(r"[₀₁₂₃₄₅₆₇₈₉0-9]", "", raw_piece)
        if not piece or not piece.isalpha():
            raise CorrespondenceBenchmarkError(
                f"Form {value!r} is not a complete alphabetic syllabic transcription."
            )
        pieces.append(piece)
    if not pieces:
        raise CorrespondenceBenchmarkError("An empty form cannot be benchmarked.")
    return "-".join(pieces)


class PublishedPairBenchmark:
    """Validate and evaluate a citation-bound exact-form comparison set."""

    VALID_SOURCE_STATUSES = {"non_greek", "uncertain"}

    def __init__(self, corpus_dir: Path | None = None, benchmark_path: Path | None = None):
        self.corpus_dir = corpus_dir or get_default_corpus_dir()
        self.benchmark_path = benchmark_path or (
            self.corpus_dir / "lexicons" / "linear_b_correspondence_benchmark.yaml"
        )
        with self.benchmark_path.open(encoding="utf-8") as handle:
            self.raw = yaml.safe_load(handle) or {}
        self.metadata = self.raw.get("metadata", {})
        self.entries = tuple(self._parse_entry(item) for item in self.raw.get("entries", []))
        self._validate_dataset()

    def _parse_entry(self, item: dict[str, Any]) -> PublishedPair:
        citation = item.get("pair_citation") or {}
        return PublishedPair(
            id=str(item.get("id", "")),
            linear_a_form=str(item.get("linear_a_form", "")),
            linear_b_form=str(item.get("linear_b_form", "")),
            lexical_category=str(item.get("lexical_category", "")),
            source_status=str(item.get("source_status", "")),
            pair_citation={str(key): str(value) for key, value in citation.items()},
            linear_b_attestations=tuple(str(value) for value in item.get("linear_b_attestations", [])),
            limitation=str(item.get("limitation", "")),
        )

    def _validate_dataset(self) -> None:
        target = self.metadata.get("target_entries")
        if not isinstance(target, int) or target <= 0:
            raise CorrespondenceBenchmarkError("metadata.target_entries must be a positive integer.")
        seen: set[str] = set()
        for entry in self.entries:
            if not entry.id or entry.id in seen:
                raise CorrespondenceBenchmarkError("Every entry needs a unique non-empty id.")
            seen.add(entry.id)
            if entry.lexical_category in {"personal_name", "place_name", ""}:
                raise CorrespondenceBenchmarkError(f"{entry.id}: only non-proper lexical entries are allowed.")
            if entry.source_status not in self.VALID_SOURCE_STATUSES:
                raise CorrespondenceBenchmarkError(f"{entry.id}: source_status must be non_greek or uncertain.")
            normalize_syllabic_form(entry.linear_a_form)
            normalize_syllabic_form(entry.linear_b_form)
            if not entry.linear_b_attestations:
                raise CorrespondenceBenchmarkError(f"{entry.id}: at least one Linear B tablet attestation is required.")
            if not entry.limitation:
                raise CorrespondenceBenchmarkError(f"{entry.id}: an explicit limitation is required.")
            if not all(entry.pair_citation.get(key) for key in ("author", "year", "title", "locator")):
                raise CorrespondenceBenchmarkError(f"{entry.id}: pair_citation needs author, year, title, and locator.")

    @staticmethod
    def _usable_tokens(values: Iterable[str]) -> list[str]:
        tokens = []
        for value in values:
            try:
                tokens.append(normalize_syllabic_form(value))
            except CorrespondenceBenchmarkError:
                continue
        return tokens

    def _verified_linear_a_tokens(self) -> tuple[list[str], dict[str, Any]]:
        """Read all complete alphabetic token forms from the manifest-verified snapshot."""
        census = CorpusLacunaeCensusEngine(corpus_dir=self.corpus_dir)
        snapshot = census._source_snapshot()
        raw = census.annotations_path.read_text(encoding="utf-8", errors="ignore")
        values = re.findall(r'"transliteratedWord":\s*"([^"]+)"', raw)
        tokens = self._usable_tokens(values)
        if not tokens:
            raise CorrespondenceBenchmarkError("The verified Linear A source contains no complete syllabic tokens.")
        return tokens, snapshot

    @staticmethod
    def _surrogate_matches(
        tokens: list[str], pairs: tuple[PublishedPair, ...], surrogates: int, seed: int
    ) -> list[int]:
        syllables = Counter(piece for token in tokens for piece in token.split("-"))
        population = list(syllables)
        weights = [syllables[item] for item in population]
        lengths = [len(token.split("-")) for token in tokens]
        target_forms = {normalize_syllabic_form(pair.linear_b_form) for pair in pairs}
        rng = random.Random(seed)
        counts = []
        for _ in range(surrogates):
            pseudo = {
                "-".join(rng.choices(population, weights=weights, k=length))
                for length in lengths
            }
            counts.append(sum(form in pseudo for form in target_forms))
        return counts

    def run(self, surrogates: int = 10_000, seed: int = 42) -> SubstratumBenchmarkReport:
        """Run the pre-registered exact-form test, or disclose a qualified-set shortfall."""
        target = self.metadata["target_entries"]
        limitations = str(self.metadata.get("limitations", ""))
        if not self.entries:
            return SubstratumBenchmarkReport(
                benchmark_id=str(self.metadata.get("id", "linear-a-linear-b-published-pairs")),
                target_entries=target,
                qualified_entries=0,
                qualifying_shortfall=target,
                observed_exact_matches=0,
                null_mean_matches=None,
                empirical_p_value=None,
                surrogates=0,
                source_snapshot=None,
                limitations=limitations,
            )
        if surrogates <= 0:
            raise CorrespondenceBenchmarkError("surrogates must be positive.")
        tokens, snapshot = self._verified_linear_a_tokens()
        token_set = set(tokens)
        observed = sum(normalize_syllabic_form(pair.linear_b_form) in token_set for pair in self.entries)
        null = self._surrogate_matches(tokens, self.entries, surrogates, seed)
        return SubstratumBenchmarkReport(
            benchmark_id=str(self.metadata.get("id", "linear-a-linear-b-published-pairs")),
            target_entries=target,
            qualified_entries=len(self.entries),
            qualifying_shortfall=max(0, target - len(self.entries)),
            observed_exact_matches=observed,
            null_mean_matches=round(sum(null) / len(null), 6),
            empirical_p_value=round((sum(value >= observed for value in null) + 1) / (len(null) + 1), 6),
            surrogates=surrogates,
            source_snapshot=snapshot,
            limitations=limitations,
        )
