"""Linear A Accounting Ledger Validator.

Verifies mathematical accounting consistency on Minoan clay tablets:
sum(inputs) == KU-RO (Total), and tracks KI-RO (deficit/shortfall).
Applies Evidence Tier E3 (Numerical & Accounting Constraints).
"""

from fractions import Fraction
from typing import Dict, List, Optional, Union
from linear_a.accounting.fractions import FractionEngine
from linear_a.core.models import LedgerLineItem, LedgerVerificationResult


class LedgerValidator:
    """Validates accounting equations and fractional balances on Linear A tablets."""

    def __init__(self, fraction_engine: Optional[FractionEngine] = None):
        self.fraction_engine = fraction_engine or FractionEngine()

    def verify_ledger(
        self,
        tablet_id: str,
        items: List[LedgerLineItem],
        stated_kuro_fraction: Optional[Union[str, List[str]]] = None,
        stated_kuro_integer: Optional[int] = None,
        commodity: Optional[str] = None,
    ) -> LedgerVerificationResult:
        """Verify that line items sum exactly to the stated KU-RO total.

        Parameters
        ----------
        tablet_id: e.g. "HT_009", "HT_013"
        items: List of line items representing inputs
        stated_kuro_fraction: String fraction (e.g. "31", "15 3/4", "J") or list of symbols
        stated_kuro_integer: Optional integer if KU-RO has an integer amount
        commodity: Target commodity filter if ledger handles multiple commodities
        """
        relevant_items = [
            item for item in items
            if commodity is None or item.commodity == commodity or item.commodity is None
        ]

        computed_sum = Fraction(0, 1)
        for item in relevant_items:
            if item.fractional_symbols:
                frac = self.fraction_engine.parse_fraction_symbols(item.fractional_symbols)
                item_total = Fraction(item.integer_amount, 1) + frac
            else:
                item_total = item.total_fraction
            computed_sum += item_total

        # Determine stated KU-RO Fraction
        stated_frac: Optional[Fraction] = None
        frac_component = Fraction(0, 1)
        has_frac = False

        if stated_kuro_fraction is not None:
            has_frac = True
            if isinstance(stated_kuro_fraction, (list, tuple)):
                frac_component = self.fraction_engine.parse_fraction_symbols(list(stated_kuro_fraction))
            elif isinstance(stated_kuro_fraction, str):
                s = stated_kuro_fraction.strip()
                if "/" in s or " " in s:
                    parts = s.split()
                    if len(parts) == 2:
                        whole = int(parts[0])
                        num, den = map(int, parts[1].split("/"))
                        frac_component = Fraction(whole, 1) + Fraction(num, den)
                    elif len(parts) == 1 and "/" in parts[0]:
                        num, den = map(int, parts[0].split("/"))
                        frac_component = Fraction(num, den)
                    else:
                        frac_component = Fraction(int(parts[0]), 1)
                elif s.isdigit():
                    frac_component = Fraction(int(s), 1)
                else:
                    frac_component = self.fraction_engine.parse_fraction_symbols(s)

        if stated_kuro_integer is not None:
            stated_frac = Fraction(stated_kuro_integer, 1) + frac_component
        elif has_frac:
            stated_frac = frac_component

        formatted_computed = self.fraction_engine.format_fraction(computed_sum)
        computed_dec = float(computed_sum)

        if stated_frac is not None:
            formatted_kuro = self.fraction_engine.format_fraction(stated_frac)
            stated_dec = float(stated_frac)
            discrepancy = abs(computed_dec - stated_dec)
            is_balanced = discrepancy < 1e-6

            if is_balanced:
                rationale = (
                    f"E3 NUMERICAL PROOF: Tablet {tablet_id} achieves exact mathematical balance. "
                    f"The sum of {len(relevant_items)} input entries equals {formatted_computed} "
                    f"(decimal {computed_dec:.4f}), matching the stated KU-RO total of {formatted_kuro}."
                )
            else:
                rationale = (
                    f"E3 NUMERICAL DISCREPANCY: Tablet {tablet_id} inputs sum to {formatted_computed} "
                    f"(decimal {computed_dec:.4f}), but stated KU-RO is {formatted_kuro} "
                    f"(discrepancy = {discrepancy:+.4f}). Possible unrecorded entry, damaged sign, or alternate fractional reading."
                )
        else:
            formatted_kuro = None
            stated_dec = None
            is_balanced = True
            discrepancy = 0.0
            rationale = (
                f"E3 TALLY CALCULATED: Tablet {tablet_id} contains {len(relevant_items)} entries "
                f"yielding a calculated total of {formatted_computed} ({computed_dec:.4f}). "
                f"No explicit KU-RO token preserved on tablet."
            )

        return LedgerVerificationResult(
            tablet_id=tablet_id,
            commodity=commodity,
            input_items_count=len(relevant_items),
            computed_sum_fraction=formatted_computed,
            computed_sum_decimal=round(computed_dec, 4),
            stated_kuro_fraction=formatted_kuro,
            stated_kuro_decimal=round(stated_dec, 4) if stated_dec is not None else None,
            is_balanced=is_balanced,
            discrepancy_decimal=round(discrepancy, 4),
            evidence_tier="E3",
            rationale=rationale,
        )
