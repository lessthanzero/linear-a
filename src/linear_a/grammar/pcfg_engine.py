"""Rule-based structural-pattern parser for curated Linear A readings.

The script remains undeciphered. This module recognizes recurring, explicitly
encoded administrative and votive patterns; it makes no linguistic or
decipherment claim.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParseNode:
    """A constituent in a structural-pattern tree."""

    symbol: str
    text: str = ""
    children: List["ParseNode"] = field(default_factory=list)
    evidence_level: str = "documented-pattern"

    def to_dict(self) -> Dict[str, Any]:
        return {"symbol": self.symbol, "text": self.text, "evidence_level": self.evidence_level,
                "children": [child.to_dict() for child in self.children]}


@dataclass
class ParseResult:
    """A transparent structural hypothesis for one curated reading."""

    document_id: str
    genre_hypothesis: str
    parse_tree: Optional[ParseNode]
    pattern_coverage: float
    terminals: List[Tuple[str, str]]
    clausal_breakdown: List[Dict[str, Any]]
    limitations: str

    @property
    def genre(self) -> str:
        return self.genre_hypothesis


class StructuralPatternParser:
    """Recognize documented ledger and libation-formula patterns."""

    COMMODITIES = {"GRA", "OLE", "VIN", "FIC", "VIR", "TEL", "VAS", "CYP", "LIV", "GRA+PA", "OLE+U", "OLE+DI", "OLE+A", "*304", "*304+E", "*118", "VIR+KA", "VIR+*310"}
    ADMINISTRATORS = {"SA-RU", "TE-TU", "KU-PA-NU", "DA-TA-RE", "PA-DE", "DA-RE", "KA-DU-MA-NE", "A-KA-RU", "KA-DI", "DI-DI-ZA-KE", "ZU-SU", "A-RA-NA-RE", "KI-DA", "MA-KA-RI-TE"}
    CURATED_TOPONYMIC_FORMS = {"PA-I-TO", "PA-DA-NI", "DI-RA-DI-NA", "TY-LI-SO", "SE-TO-I-JA", "KU-DO-NI-JA", "A-DU-NI-TA", "A-DU-NI-TA-NA", "QE-RA₂-U"}
    INVOCATIONS = {"A-TA-I-*301-WA-JA", "TA-NA-I-*301-TI", "A-TA-I-*301-DE"}
    THEONYMS = {"JA-SA-SA-RA-ME", "A-SA-SA-RA-ME", "JA-SA-SA-RA-MA-NA", "JA-DI-KI-TU"}
    DEDICATORY_VERBS = {"U-NA-KA-NA-SI"}
    OFFERINGS = {"I-PI-NA-MA"}
    SUMMARIES = {"KU-RO", "PO-TO-KU-RO"}
    DEFICITS = {"KI-RO", "KI"}
    LIMITATIONS = "Rule-based structural hypothesis only. Roles derive from curated transliterations and do not establish grammar, language, meaning, or decipherment."

    def tokenize_and_tag(self, words: List[str]) -> List[Tuple[str, str]]:
        """Assign transparent pattern labels without interpreting meaning."""
        tagged: List[Tuple[str, str]] = []
        for word in words:
            clean = word.upper().strip("[]*? ")
            if not clean:
                continue
            if clean.isdigit(): label = "INTEGER"
            elif clean in {"J", "E", "F", "K", "H", "EF", "BB", "DD", "L2", "1/2", "1/4", "1/8", "3/8"}: label = "FRACTION"
            elif clean in self.COMMODITIES or clean.startswith(("GRA", "OLE", "VIN", "VIR")): label = "COMMODITY"
            elif clean in self.SUMMARIES: label = "SUMMARY_MARKER"
            elif clean in self.DEFICITS: label = "DEFICIT_MARKER"
            elif clean in self.INVOCATIONS: label = "INVOCATION_PATTERN"
            elif clean in self.THEONYMS: label = "THEONYM_PATTERN"
            elif clean in self.DEDICATORY_VERBS: label = "DEDICATORY_PATTERN"
            elif clean in self.OFFERINGS: label = "OFFERING_PATTERN"
            elif clean in self.CURATED_TOPONYMIC_FORMS: label = "CURATED_TOPONYMIC_FORM_PATTERN"
            elif clean in self.ADMINISTRATORS: label = "ADMINISTRATOR_PATTERN"
            else: label = "UNCLASSIFIED_TOKEN"
            tagged.append((word, label))
        return tagged

    def parse_inscription(self, document_id: str, words: List[str]) -> ParseResult:
        tagged = self.tokenize_and_tag(words)
        if not tagged:
            return ParseResult(document_id, "NO_PATTERN", None, 0.0, [], [], self.LIMITATIONS)
        votive = {"INVOCATION_PATTERN", "THEONYM_PATTERN", "DEDICATORY_PATTERN", "OFFERING_PATTERN"}
        return self._parse_votive(document_id, tagged) if any(label in votive for _, label in tagged) else self._parse_ledger(document_id, tagged)

    def _parse_ledger(self, document_id: str, tagged: List[Tuple[str, str]]) -> ParseResult:
        root, clauses, entries = ParseNode("ADMINISTRATIVE_LEDGER_PATTERN"), [], ParseNode("TRANSACTION_SEQUENCE")
        index = 0
        while index < len(tagged):
            word, label = tagged[index]
            if label in {"SUMMARY_MARKER", "DEFICIT_MARKER"}:
                marker = ParseNode("SUMMARY_PATTERN" if label == "SUMMARY_MARKER" else "DEFICIT_PATTERN", text=word)
                index += 1
                if index < len(tagged) and tagged[index][1] in {"INTEGER", "FRACTION", "COMMODITY"}:
                    marker.children.append(ParseNode(tagged[index][1], text=tagged[index][0])); index += 1
                root.children.append(marker)
                clauses.append({"pattern": marker.symbol, "tokens": [node.text for node in [marker, *marker.children]]})
                continue
            entry = ParseNode("TRANSACTION_PATTERN")
            while index < len(tagged) and tagged[index][1] not in {"SUMMARY_MARKER", "DEFICIT_MARKER"}:
                token, token_label = tagged[index]; entry.children.append(ParseNode(token_label, text=token)); index += 1
                if token_label in {"INTEGER", "FRACTION"}: break
            if entry.children:
                entries.children.append(entry); clauses.append({"pattern": "TRANSACTION_PATTERN", "tokens": [node.text for node in entry.children]})
        if entries.children: root.children.insert(0, entries)
        covered = sum(label != "UNCLASSIFIED_TOKEN" for _, label in tagged)
        return ParseResult(document_id, "ADMINISTRATIVE_LEDGER_PATTERN", root, round(covered / len(tagged), 2), tagged, clauses, self.LIMITATIONS)

    def _parse_votive(self, document_id: str, tagged: List[Tuple[str, str]]) -> ParseResult:
        root = ParseNode("VOTIVE_FORMULA_PATTERN")
        phases = {"INVOCATION_PATTERN": "PHASE_1_INVOCATION_PATTERN", "THEONYM_PATTERN": "PHASE_2_THEONYM_PATTERN", "DEDICATORY_PATTERN": "PHASE_3_DEDICATORY_PATTERN", "OFFERING_PATTERN": "PHASE_4_OFFERING_PATTERN", "CURATED_TOPONYMIC_FORM_PATTERN": "CURATED_TOPONYMIC_FORM_PATTERN"}
        clauses, covered = [], 0
        for word, label in tagged:
            symbol = phases.get(label, "UNCLASSIFIED_TOKEN"); root.children.append(ParseNode(symbol, text=word)); clauses.append({"pattern": symbol, "tokens": [word]}); covered += symbol != "UNCLASSIFIED_TOKEN"
        return ParseResult(document_id, "VOTIVE_FORMULA_PATTERN", root, round(covered / len(tagged), 2), tagged, clauses, self.LIMITATIONS)


# Compatibility alias for callers of the interrupted implementation.
MinoanPCFGParser = StructuralPatternParser
