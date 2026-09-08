"""Source-linked audit of Linear A / Linear B Cretan toponym claims."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from linear_a.corpus.loader import get_default_corpus_dir
from linear_a.skeptic.substratum_filter import normalize_syllabic_form


class ToponymAuditError(ValueError):
    """Raised when a source-linked toponym record is incomplete or inconsistent."""


@dataclass(frozen=True)
class ToponymAuditRecord:
    id: str
    status: str
    linear_a_form: str | None
    linear_b_form: str | None
    alphabetic_form: str | None
    linear_a_attestations: tuple[str, ...]
    linear_b_attestations: tuple[str, ...]
    claim: str
    citations: tuple[dict[str, str], ...]
    limitations: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "linear_a_form": self.linear_a_form,
            "linear_b_form": self.linear_b_form,
            "alphabetic_form": self.alphabetic_form,
            "linear_a_attestations": list(self.linear_a_attestations),
            "linear_b_attestations": list(self.linear_b_attestations),
            "claim": self.claim,
            "citations": list(self.citations),
            "limitations": self.limitations,
        }


class CretanToponymAudit:
    """Load published correspondences and disputed geographic interpretations."""

    VALID_STATUSES = {"attested", "disputed"}

    def __init__(self, corpus_dir: Path | None = None, audit_path: Path | None = None):
        base = corpus_dir or get_default_corpus_dir()
        self.audit_path = audit_path or (base / "palaeography" / "cretan_toponym_audit.yaml")
        with self.audit_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        self.metadata = raw.get("metadata", {})
        self.records = tuple(self._record(item) for item in raw.get("records", []))
        self._validate()

    @staticmethod
    def _record(item: dict[str, Any]) -> ToponymAuditRecord:
        return ToponymAuditRecord(
            id=str(item.get("id", "")),
            status=str(item.get("status", "")),
            linear_a_form=item.get("linear_a_form"),
            linear_b_form=item.get("linear_b_form"),
            alphabetic_form=item.get("alphabetic_form"),
            linear_a_attestations=tuple(item.get("linear_a_attestations", [])),
            linear_b_attestations=tuple(item.get("linear_b_attestations", [])),
            claim=str(item.get("claim", "")),
            citations=tuple(item.get("citations", [])),
            limitations=str(item.get("limitations", "")),
        )

    def _validate(self) -> None:
        ids: set[str] = set()
        for record in self.records:
            if not record.id or record.id in ids:
                raise ToponymAuditError("Every record needs a unique non-empty id.")
            ids.add(record.id)
            if record.status not in self.VALID_STATUSES:
                raise ToponymAuditError(f"{record.id}: status must be attested or disputed.")
            if not record.claim or not record.limitations:
                raise ToponymAuditError(f"{record.id}: claim and limitations are required.")
            if not record.citations:
                raise ToponymAuditError(f"{record.id}: at least one source citation is required.")
            for citation in record.citations:
                if not all(citation.get(key) for key in ("author", "year", "title", "locator", "url")):
                    raise ToponymAuditError(f"{record.id}: citations need author, year, title, locator, and url.")
            if record.status == "attested":
                if not all((record.linear_a_form, record.linear_b_form, record.alphabetic_form)):
                    raise ToponymAuditError(f"{record.id}: attested records need all three conventional forms.")
                if not record.linear_a_attestations or not record.linear_b_attestations:
                    raise ToponymAuditError(f"{record.id}: attested records need Linear A and Linear B attestations.")
                normalize_syllabic_form(record.linear_a_form)
                normalize_syllabic_form(record.linear_b_form)

    def record_for_linear_a_form(self, form: str) -> ToponymAuditRecord | None:
        normalized = normalize_syllabic_form(form)
        return next(
            (
                record
                for record in self.records
                if record.linear_a_form and normalize_syllabic_form(record.linear_a_form) == normalized
            ),
            None,
        )

    def report(self) -> dict[str, Any]:
        counts = {status: sum(record.status == status for record in self.records) for status in self.VALID_STATUSES}
        return {
            "title": self.metadata.get("title", "Cretan Toponym Audit"),
            "limitations": self.metadata.get("limitations", ""),
            "status_counts": counts,
            "records": [record.to_dict() for record in self.records],
        }
