"""Tests for Multi-Commodity Metrological Diophantine Solver."""

from linear_a.accounting.multi_commodity_solver import (
    MultiCommodityDiophantineSolver,
    MultiCommodityReport,
    MultiCommodityTabletLedger,
    SimultaneousMaskSolution,
)


def test_parse_multi_commodity_ledgers():
    solver = MultiCommodityDiophantineSolver()
    ledgers = solver.parse_multi_commodity_ledgers()

    assert len(ledgers) >= 3
    tablet_ids = {l.tablet_id for l in ledgers}
    assert "HT_085" in tablet_ids
    assert "HT_013" in tablet_ids
    assert "PH_001" in tablet_ids

    ht85 = next(l for l in ledgers if l.tablet_id == "HT_085")
    assert "GRA+PA" in ht85.commodities_present
    assert "OLE+U" in ht85.commodities_present
    assert "VIN" in ht85.commodities_present
    assert ht85.is_grand_balanced is True


def test_solve_simultaneous_masks_on_ht085():
    solver = MultiCommodityDiophantineSolver()
    ledgers = solver.parse_multi_commodity_ledgers()
    ht85 = next(l for l in ledgers if l.tablet_id == "HT_085")

    # Single mask on item 0 (GRA+PA, true value = 12 + 1/2)
    sol1 = solver.solve_simultaneous_masks(ht85, [0])
    assert sol1.is_exact_recovery is True
    assert sol1.residual_delta == 0.0
    assert sol1.minoan_symbols[0] == "12 J"

    # Single mask on item 2 (VIN, true value = 8 + 1/4)
    sol2 = solver.solve_simultaneous_masks(ht85, [2])
    assert sol2.is_exact_recovery is True
    assert sol2.residual_delta == 0.0
    assert sol2.minoan_symbols[0] == "8 E"

    # Simultaneous double mask on items [0, 2] across GRA+PA and VIN
    sol_double = solver.solve_simultaneous_masks(ht85, [0, 2])
    assert sol_double.is_exact_recovery is True
    assert sol_double.residual_delta == 0.0
    assert len(sol_double.solved_values) == 2


def test_commodity_fraction_associations():
    solver = MultiCommodityDiophantineSolver()
    associations = solver.analyze_commodity_fraction_associations()

    assert len(associations) > 0
    comms = {a.commodity for a in associations}
    assert any("GRA" in c for c in comms)
    assert any("VIN" in c for c in comms)
    assert any("FIC" in c for c in comms)

    for a in associations:
        assert a.total_attestations > 0
        assert a.metrology_class in ("DRY_MEASURE", "LIQUID_MEASURE", "COUNT_DISCRETE", "UNKNOWN")


def test_multi_commodity_benchmark_report():
    solver = MultiCommodityDiophantineSolver()
    report = solver.benchmark_multi_commodity_solver()

    assert isinstance(report, MultiCommodityReport)
    assert report.total_multi_commodity_tablets >= 3
    assert report.simultaneous_benchmark_masks >= 9
    assert report.accuracy_pct == 100.0
    assert report.exact_recoveries_count == report.simultaneous_benchmark_masks
    assert "Multi-Commodity Diophantine Solver evaluated" in report.summary

    rep_dict = report.to_dict()
    assert "total_multi_commodity_tablets" in rep_dict
    assert "accuracy_pct" in rep_dict
    assert "solutions" in rep_dict
    assert len(rep_dict["solutions"]) == report.simultaneous_benchmark_masks
