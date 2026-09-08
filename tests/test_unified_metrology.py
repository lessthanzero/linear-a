"""Tests for Minoan Unified Metrological Weight & Commodity Equivalence Tree."""

from fractions import Fraction
from linear_a.accounting.unified_metrology import MinoanUnifiedMetrologyEngine


def test_unified_metrology_engine():
    engine = MinoanUnifiedMetrologyEngine()
    rep = engine.generate_metrology_report()

    assert rep.base_weight_unit_grams == 61.0
    assert rep.base_volume_unit_liters == 28.8
    assert len(rep.balance_weights) >= 8
    assert len(rep.volume_units) >= 8
    assert len(rep.equivalence_ratios) >= 4

    # Talent scaling
    assert rep.talent_subdivisions["talent_kg"] > 28.0
    assert rep.talent_subdivisions["light_minas_M"] == 480.0

    # Check fractional volume relationships
    vol_map = {vu.symbol: vu.volume_liters for vu in rep.volume_units}
    assert vol_map["J"] == vol_map["1"] / 2.0
    assert vol_map["E"] == vol_map["1"] / 4.0
    assert vol_map["F"] == vol_map["1"] / 8.0

    # Serialization
    d = rep.to_dict()
    assert "base_weight_unit_grams" in d
    assert "balance_weights" in d
    assert "volume_units" in d
    assert "equivalence_ratios" in d
    assert "talent_subdivisions" in d


def test_balance_weight_hierarchy():
    engine = MinoanUnifiedMetrologyEngine()
    rep = engine.generate_metrology_report()

    # Verify weights are strictly monotonic
    for i in range(len(rep.balance_weights) - 1):
        w1 = rep.balance_weights[i]
        w2 = rep.balance_weights[i + 1]
        assert w1.mass_grams < w2.mass_grams
        assert w1.base_unit_ratio < w2.base_unit_ratio


def test_commodity_equivalence_ratios():
    engine = MinoanUnifiedMetrologyEngine()
    rep = engine.generate_metrology_report()

    ole_gra = next(er for er in rep.equivalence_ratios if "OLE" in er.commodity_a)
    assert ole_gra.canonical_ratio == 2.0
    assert "HT 13" in ole_gra.tablets_attested
