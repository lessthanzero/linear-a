"""Linear A Fractional Arithmetic Engine.

Implements the modern mathematical consensus for Linear A fractions established by
Silvia Ferrara, Michele Corazza, Barbara Montecchi, and Miguel Valério (2020):
'The mathematical values of Linear A fractions', Journal of Archaeological Science.
"""

from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
from linear_a.core.models import FractionSymbol


# Canonical Ferrara et al. (2020) values
CANONICAL_FRACTIONS: Dict[str, Tuple[int, int, str]] = {
    # symbol: (numerator, denominator, citation_note)
    "J": (1, 2, "Half unit (1/2). Secure anchor across all Aegean accounting archives."),
    "E": (1, 4, "Quarter unit (1/4). Derived from subdivision of J."),
    "F": (1, 8, "Eighth unit (1/8). Primary liquid/dry measure sub-fraction."),
    "K": (1, 16, "Sixteenth unit (1/16). Base unit for fine commodity tallies."),
    "A": (1, 6, "Sixth unit (1/6) in primary dry measure series; alternating with 1/3 in older literature."),
    "H": (1, 12, "Twelfth unit (1/12). Attested on PH 1 and HT ledgers."),
    "B": (1, 5, "Fifth unit (1/5) / decimal subdivision candidate."),
    "D": (1, 10, "Tenth unit (1/10)."),
    "L": (1, 24, "Twenty-fourth unit (1/24). Small capacity liquid fraction."),
    "L2": (1, 48, "Forty-eighth unit (1/48). Secondary small fraction (attested on PH 1)."),
    "L3": (1, 96, "Ninety-sixth unit (1/96)."),
}


class FractionEngine:
    """Evaluates Linear A fractional signs and compounds using exact rational arithmetic."""

    def __init__(self, custom_mapping: Optional[Dict[str, Tuple[int, int, str]]] = None):
        mapping = custom_mapping or CANONICAL_FRACTIONS
        self.symbols: Dict[str, FractionSymbol] = {}
        for sym, (num, den, note) in mapping.items():
            self.symbols[sym] = FractionSymbol(
                symbol=sym,
                fraction_numerator=num,
                fraction_denominator=den,
                decimal_value=round(num / den, 6),
                confidence=0.95 if sym in ["J", "E", "F", "K"] else 0.85,
                proponents=["Ferrara & Corazza 2020"],
            )

    def parse_fraction_symbols(self, symbols: Union[str, List[str]]) -> Fraction:
        """Parse one or more fractional symbols into an exact Fraction sum.

        Supports single symbols (e.g. 'J'), compounds (e.g. 'JE', 'JEF'), or lists (e.g. ['J', 'E']).
        """
        if isinstance(symbols, str):
            symbols = symbols.strip()
            if not symbols:
                return Fraction(0, 1)
            # Check if direct match
            if symbols in self.symbols:
                return self.symbols[symbols].value
            # Handle compound string like 'JE', 'JEF', 'EF'
            total = Fraction(0, 1)
            # Try parsing multi-char symbols like 'L2', 'L3' first
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i : i + 2] in self.symbols:
                    total += self.symbols[symbols[i : i + 2]].value
                    i += 2
                elif symbols[i] in self.symbols:
                    total += self.symbols[symbols[i]].value
                    i += 1
                elif symbols[i] in [" ", "+", ","]:
                    i += 1
                else:
                    # Unrecognized fraction token
                    i += 1
            return total

        total = Fraction(0, 1)
        for s in symbols:
            total += self.parse_fraction_symbols(s)
        return total

    def format_fraction(self, frac: Fraction) -> str:
        """Format Fraction as mixed number (e.g. '31', '15 3/4', '1/2')."""
        if frac.denominator == 1:
            return str(frac.numerator)
        integer_part = frac.numerator // frac.denominator
        remainder = frac.numerator % frac.denominator
        if integer_part == 0:
            return f"{remainder}/{frac.denominator}"
        return f"{integer_part} {remainder}/{frac.denominator}"

    def evaluate_entry(self, integer_amount: int, fraction_symbols: Union[str, List[str]]) -> Tuple[Fraction, str]:
        """Compute exact Fraction and formatted string for an entry."""
        frac_val = self.parse_fraction_symbols(fraction_symbols)
        total = Fraction(integer_amount, 1) + frac_val
        return total, self.format_fraction(total)
