"""Tests for Standalone Interactive Epigraphic Research Workbench Generator."""

import pytest
from pathlib import Path
from linear_a.visualizer.workbench import generate_workbench_html, collect_workbench_dataset


def test_collect_workbench_dataset():
    data = collect_workbench_dataset()
    assert "tablets" in data
    assert len(data["tablets"]) >= 13
    assert "grid" in data
    assert data["grid"]["total_signs"] > 0
    assert "gauntlet" in data
    assert "semitic" in data["gauntlet"]
    assert "libation" in data
    assert "holdout" in data
    assert data["holdout"]["exact_kuro_acc"] == 100.0
    assert "phaistos" in data
    assert data["phaistos"]["firewall_intact"] is True
    assert "jury" in data
    assert len(data["jury"]) >= 2
    assert "lacunae" in data
    assert data["lacunae"]["total"] == 23
    assert data["lacunae"]["top1_accuracy"] >= 95.0


def test_generate_workbench_html(tmp_path):
    out_file = tmp_path / "linear_a_workbench.html"
    res_path = generate_workbench_html(str(out_file))

    assert res_path.exists()
    assert res_path.stat().st_size > 50_000

    content = res_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Linear A Decipherment Laboratory" in content
    assert "tab-tablets" in content
    assert "tab-grid" in content
    assert "tab-gauntlet" in content
    assert "tab-votive" in content
    assert "tab-holdout" in content
    assert "tab-phaistos" in content
    assert "tab-jury" in content
    assert "lacunaeTable" in content
    assert "PHAISTOS FIREWALL ACTIVE" in content
