"""Corpus loader and parser for Linear A inscriptions, tablets, and signs."""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from linear_a.core.models import (
    Inscription,
    LedgerLineItem,
    ObjectType,
    Sign,
    Site,
)


def get_default_corpus_dir() -> Path:
    """Return the absolute path to the local corpus directory."""
    return Path(__file__).resolve().parent.parent.parent.parent / "corpus"


def load_signs_catalogue(corpus_dir: Optional[Path] = None) -> Dict[str, Sign]:
    """Load the full Linear A sign catalogue from corpus/signs.yaml."""
    base_dir = corpus_dir or get_default_corpus_dir()
    file_path = base_dir / "signs.yaml"
    if not file_path.exists():
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    catalogue: Dict[str, Sign] = {}
    for s_raw in data.get("signs", []):
        sign = Sign(**s_raw)
        catalogue[sign.glyph_id] = sign
    return catalogue


def load_tablet_ledgers(site_name: str = "hagia_triada", corpus_dir: Optional[Path] = None) -> List[Dict]:
    """Load raw tablet dictionaries from corpus/tablets/{site_name}.yaml."""
    base_dir = corpus_dir or get_default_corpus_dir()
    file_path = base_dir / "tablets" / f"{site_name}.yaml"
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("tablets", [])


def parse_tablet_line_items(tablet_raw: Dict) -> List[LedgerLineItem]:
    """Parse raw tablet item dictionary into LedgerLineItem objects."""
    items = []
    for it in tablet_raw.get("items", []):
        items.append(LedgerLineItem(
            entry_header=it["entry_header"],
            commodity=it.get("commodity"),
            integer_amount=it.get("integer_amount", 0),
            fractional_symbols=it.get("fractional_symbols", []),
            notes=it.get("notes"),
        ))
    return items
