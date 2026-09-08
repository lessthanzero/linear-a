"""Source-linked restoration review queue and adjudication validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from linear_a.corpus.loader import get_default_corpus_dir
from linear_a.predictive.multilateral_solver import LacunaEntry


VALID_EVIDENCE_STATUSES = {"unadjudicated", "accepted", "disputed", "rejected"}


@dataclass
class BenchmarkEntry:
    id: str
    catalog_id: str
    evidence_status: str
    citation: dict[str, Any]
    adjudication: dict[str, Any] | None = None


class SourceLinkedBenchmark:
    """Load source-linked entries and expose only accepted items for scoring."""

    def __init__(self, corpus_dir: Path | None = None):
        self.corpus_dir = corpus_dir or get_default_corpus_dir()
        self.catalog = self._load_catalog()
        self.metadata, self.entries = self._load_entries()
        self.validate()

    def _load_catalog(self) -> dict[str, LacunaEntry]:
        path = self.corpus_dir / "palaeography" / "lacunae_catalog.yaml"
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return {item["id"]: LacunaEntry(**item) for item in raw.get("lacunae", [])}

    def _load_entries(self) -> tuple[dict[str, Any], list[BenchmarkEntry]]:
        path = self.corpus_dir / "palaeography" / "source_linked_benchmark.yaml"
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        metadata = {key: raw.get(key) for key in ("title", "status", "limitations")}
        entries = [BenchmarkEntry(**item) for item in raw.get("entries", [])]
        return metadata, entries

    def validate(self) -> None:
        ids = [entry.id for entry in self.entries]
        catalog_ids = [entry.catalog_id for entry in self.entries]
        if len(ids) != len(set(ids)) or len(catalog_ids) != len(set(catalog_ids)):
            raise ValueError("Benchmark IDs and catalog references must be unique.")
        if set(catalog_ids) != set(self.catalog):
            raise ValueError("Source-linked benchmark must cover every catalog entry exactly once.")
        for entry in self.entries:
            if entry.evidence_status not in VALID_EVIDENCE_STATUSES:
                raise ValueError(f"Invalid evidence status for {entry.id}: {entry.evidence_status}")
            citation = entry.citation
            if not all(citation.get(key) for key in ("series", "volume", "pages")):
                raise ValueError(f"Missing normalized citation fields for {entry.id}")
            if entry.evidence_status == "unadjudicated":
                if entry.adjudication:
                    raise ValueError(f"Unadjudicated entry {entry.id} cannot include a decision.")
                continue
            adjudication = entry.adjudication or {}
            if not adjudication.get("reviewers") or not adjudication.get("decision_at"):
                raise ValueError(f"Adjudicated entry {entry.id} requires reviewers and decision_at.")
            if adjudication.get("decision") != entry.evidence_status:
                raise ValueError(f"Adjudication decision must match evidence status for {entry.id}.")

    def entries_with_status(self, status: str) -> list[BenchmarkEntry]:
        return [entry for entry in self.entries if entry.evidence_status == status]

    def accepted_catalog_entries(self) -> list[LacunaEntry]:
        return [self.catalog[entry.catalog_id] for entry in self.entries_with_status("accepted")]

    def status_counts(self) -> dict[str, int]:
        return {status: len(self.entries_with_status(status)) for status in VALID_EVIDENCE_STATUSES}

    def review_handoff(self) -> dict[str, Any]:
        """Return reviewable records without assigning a reviewer decision."""
        records = []
        for entry in self.entries:
            catalog = self.catalog[entry.catalog_id]
            records.append({
                "benchmark_id": entry.id,
                "catalog_id": entry.catalog_id,
                "document": catalog.document,
                "masked_token": catalog.masked_token,
                "proposed_sign": catalog.reconstructed_sign,
                "proposed_completion": catalog.completed_word,
                "catalog_rationale": catalog.epigraphic_rationale,
                "citation": entry.citation,
                "evidence_status": entry.evidence_status,
                "reviewers": [],
                "decision": None,
                "decision_at": None,
                "review_notes": None,
            })
        return {
            "title": "Linear A source-linked restoration review handoff",
            "benchmark_status": self.metadata["status"],
            "limitations": self.metadata["limitations"],
            "records": records,
        }

    def with_adjudications(self, adjudication_records: list[dict[str, Any]]) -> "SourceLinkedBenchmark":
        """Return a new SourceLinkedBenchmark instance with external adjudications applied."""
        new_benchmark = SourceLinkedBenchmark.__new__(SourceLinkedBenchmark)
        new_benchmark.corpus_dir = self.corpus_dir
        new_benchmark.catalog = self.catalog
        new_benchmark.metadata = dict(self.metadata)

        adjudication_map = {
            item.get("benchmark_id") or item.get("id"): item
            for item in adjudication_records
            if item.get("decision")
        }

        new_entries = []
        for entry in self.entries:
            adj = adjudication_map.get(entry.id)
            if adj and adj.get("decision"):
                new_entries.append(BenchmarkEntry(
                    id=entry.id,
                    catalog_id=entry.catalog_id,
                    evidence_status=adj["decision"],
                    citation=entry.citation,
                    adjudication={
                        "reviewers": adj.get("reviewers", []),
                        "decision": adj["decision"],
                        "decision_at": adj.get("decision_at"),
                        "notes": adj.get("review_notes") or adj.get("notes"),
                    },
                ))
            else:
                new_entries.append(BenchmarkEntry(
                    id=entry.id,
                    catalog_id=entry.catalog_id,
                    evidence_status=entry.evidence_status,
                    citation=entry.citation,
                    adjudication=entry.adjudication,
                ))
        new_benchmark.entries = new_entries
        new_benchmark.validate()
        return new_benchmark

    def load_external_adjudications(self, path: Path | str) -> "SourceLinkedBenchmark":
        """Load external reviewer adjudications from a file without mutating baseline files."""
        import json
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Adjudication file not found: {p}")
        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        records = data.get("records", data) if isinstance(data, dict) else data
        return self.with_adjudications(records)
