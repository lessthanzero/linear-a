"""Tests for the Linear A Accounting Ledger Validator."""

from linear_a.accounting.ledger import LedgerValidator
from linear_a.corpus.loader import load_tablet_ledgers, parse_tablet_line_items


def test_ht_009_exact_mathematical_balance():
    """Verify that HT 9 FIC entries sum exactly to stated KU-RO 31."""
    tablets = load_tablet_ledgers("hagia_triada")
    ht9_raw = next(t for t in tablets if t["id"] == "HT_009")
    items = parse_tablet_line_items(ht9_raw)

    validator = LedgerValidator()
    result = validator.verify_ledger(
        tablet_id="HT_009",
        items=items,
        stated_kuro_integer=ht9_raw["stated_kuro"]["integer_amount"],
        commodity="FIC",
    )

    assert result.is_balanced is True
    assert result.computed_sum_decimal == 31.0
    assert result.stated_kuro_decimal == 31.0
    assert result.discrepancy_decimal == 0.0
    assert result.input_items_count == 5


def test_ht_013_fractional_allocation_totals():
    """Verify HT 13 multi-commodity fractional entries (VIN 5 J, GRA 10 E, CYP 2 F)."""
    tablets = load_tablet_ledgers("hagia_triada")
    ht13_raw = next(t for t in tablets if t["id"] == "HT_013")
    items = parse_tablet_line_items(ht13_raw)

    validator = LedgerValidator()

    # Total across all agricultural commodities: 5 1/2 + 10 1/4 + 2 1/8 = 17 7/8
    result = validator.verify_ledger(tablet_id="HT_013", items=items)
    assert result.computed_sum_fraction == "17 7/8"
    assert result.computed_sum_decimal == 17.875


def test_ht_122_commodity_distribution_balance():
    """Verify HT 122 entries (15 + 16 = 31) match stated KU-RO 31."""
    tablets = load_tablet_ledgers("hagia_triada")
    ht122_raw = next(t for t in tablets if t["id"] == "HT_122")
    items = parse_tablet_line_items(ht122_raw)

    validator = LedgerValidator()
    result = validator.verify_ledger(
        tablet_id="HT_122",
        items=items,
        stated_kuro_integer=ht122_raw["stated_kuro"]["integer_amount"],
    )
    assert result.is_balanced is True
    assert result.computed_sum_decimal == 31.0


def test_ph_001_room_8_excavation_items():
    """Verify Tablet PH 1 line items found in Room 8 with Phaistos Disc."""
    tablets = load_tablet_ledgers("phaistos")
    ph1_raw = next(t for t in tablets if t["id"] == "PH_001")
    items = parse_tablet_line_items(ph1_raw)

    assert len(items) == 3
    # Check Cyperus entry: 1 H = 1 1/12
    cyp_item = next(it for it in items if it.commodity == "CYP")
    assert cyp_item.integer_amount == 1
    assert cyp_item.fractional_symbols == ["H"]
    assert cyp_item.fractional_symbols[0] == "H"
