"""Tests for Palaeographic Stroke Vector Engine."""

from linear_a.palaeography.stroke_engine import PalaeographicStrokeEngine


def test_stroke_engine_report():
    engine = PalaeographicStrokeEngine()
    rep = engine.generate_stroke_report()

    assert rep.total_glyphs_vectorized >= 8
    assert len(rep.carriers_profiled) == 3
    assert rep.mean_stroke_count > 2.0
    assert rep.mean_lapidary_angularity > rep.mean_clay_angularity

    # Test glyph properties
    for g in rep.glyphs:
        assert g.sign_id.startswith("AB") or g.sign_id.startswith("A")
        assert len(g.strokes) > 0
        assert "CLAY_TABLET" in g.primary_carrier_styles
        assert "STONE_VESSEL" in g.primary_carrier_styles
        assert "GOLD_METAL" in g.primary_carrier_styles
        assert "<svg" in g.primary_carrier_styles["CLAY_TABLET"]
        assert "</svg>" in g.primary_carrier_styles["CLAY_TABLET"]

    # Serialization
    d = rep.to_dict()
    assert "total_glyphs_vectorized" in d
    assert "carriers_profiled" in d
    assert "glyphs" in d
    assert len(d["glyphs"]) == rep.total_glyphs_vectorized


def test_carrier_style_differentiation():
    engine = PalaeographicStrokeEngine()
    rep = engine.generate_stroke_report()

    # Find AB08 (Double Axe)
    a_glyph = next(g for g in rep.glyphs if g.sign_id == "AB08")
    assert "fill=\"none\"" in a_glyph.primary_carrier_styles["CLAY_TABLET"]
    assert "stroke=\"#3b82f6\"" in a_glyph.primary_carrier_styles["CLAY_TABLET"]
    assert "stroke=\"#10b981\"" in a_glyph.primary_carrier_styles["STONE_VESSEL"]
    assert "stroke=\"#f59e0b\"" in a_glyph.primary_carrier_styles["GOLD_METAL"]


def test_stroke_direction_descriptions():
    engine = PalaeographicStrokeEngine()
    rep = engine.generate_stroke_report()

    for g in rep.glyphs:
        for s in g.strokes:
            assert s.stroke_order >= 1
            assert s.stroke_direction != ""
            assert s.path_data.startswith("M")
