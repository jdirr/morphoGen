"""Alloys designed by thermodynamic and kinetic tools as microstructures: the description
read and checked, its scales (precipitates in the matrix, grains with their phases and
particles) as configurations, their effective elasticity by FFT within the
Hashin–Shtrikman bounds of their phases, and the command line."""

from __future__ import annotations

import json
import math

import pytest
import yaml

from morphogen.config import load_config
from morphogen.interop import alloys as al

ALUMINIUM = {
    "alloy": "AA7075-T6 (illustrative)",
    "composition": {"Al": 90.0, "Zn": 5.6, "Mg": 2.5, "Cu": 1.6},
    "basis": "wt%",
    "grain_size": 50.0,
    "phases": [
        {"name": "FCC_A1", "role": "matrix", "young": 70.0, "poisson": 0.33},
        {"name": "MgZn2", "role": "precipitates", "fraction": 0.05,
         "radius": {"mean": 0.003, "std": 0.001}, "young": 77.0, "poisson": 0.3},
        {"name": "Al7Cu2Fe", "role": "particles", "fraction": 0.02, "radius": 5.0,
         "young": 150.0, "poisson": 0.3},
    ],
    "properties": {"young": 71.7},
}  # fmt: skip


def test_the_description_of_an_alloy(tmp_path):
    """The composition normalized, the matrix the rest of the volume, the description
    written and read back; its errors explicit."""
    alloy = al.read_alloy(ALUMINIUM)
    assert sum(alloy.composition.values()) == pytest.approx(100.0)
    assert alloy.composition["Zn"] == pytest.approx(5.6 / 99.7 * 100)
    assert alloy.matrix.fraction == pytest.approx(0.93) and alloy.matrix.name == "FCC_A1"
    back = al.read_alloy(al.write_alloy(tmp_path / "a.json", alloy))
    assert back == alloy
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(ALUMINIUM))
    assert al.read_alloy(tmp_path / "a.yaml") == alloy

    def broken(**change):
        data = json.loads(json.dumps(ALUMINIUM))
        data.update(change)
        return data

    phases = ALUMINIUM["phases"]
    with pytest.raises(ValueError, match="one matrix"):
        al.read_alloy(broken(phases=[phases[0], phases[0]]))
    with pytest.raises(ValueError, match="role one of"):
        al.read_alloy(broken(phases=[phases[0], {**phases[2], "role": "dispersoids"}]))
    with pytest.raises(ValueError, match="radius"):
        al.read_alloy(broken(phases=[phases[0], {**phases[1], "radius": None}]))
    with pytest.raises(ValueError, match="leave no matrix"):
        al.read_alloy(broken(phases=[phases[0], {**phases[1], "fraction": 0.6},
                                     {**phases[2], "fraction": 0.5}]))  # fmt: skip
    with pytest.raises(ValueError, match="young and poisson"):
        al.read_alloy(broken(phases=[{"name": "x", "role": "matrix"}]))
    with pytest.raises(ValueError, match="unknown keys"):
        al.read_alloy(broken(hardness=200))
    with pytest.raises(ValueError, match="basis"):
        al.read_alloy(broken(basis="ppm"))


def test_the_scales_of_an_alloy():
    """The precipitates in a box of about sixty of them, at their fraction in the matrix;
    the grains, about sixty-four in theirs, the particles their inclusions; each scale a
    configuration of morphoGen."""
    alloy = al.read_alloy(ALUMINIUM)
    fine, coarse = al.alloy_scales(alloy)
    assert fine["name"] == "precipitates" and coarse["name"] == "grains"
    local = 0.05 / 0.98
    assert fine["model"]["volume_fraction"] == pytest.approx(local)
    from scipy import stats

    law = stats.truncnorm(-3, 3, loc=0.003, scale=0.001)  # the radii within 3 std
    assert fine["model"]["radius"]["high"] == pytest.approx(0.006)
    count = local * fine["domain"]["size"] ** 3 / (4.0 / 3.0 * math.pi * law.moment(3))
    assert count == pytest.approx(60, rel=1e-6) and fine["domain"]["size"] > 4 * 0.006
    grains = coarse["domain"]["size"] ** 3 / (math.pi / 6 * 50.0**3)
    assert grains == pytest.approx(64, rel=1e-9)
    assert coarse["model"]["operation"] == "inclusions"
    assert coarse["model"]["inclusion_phases"] == [1] and coarse["materials"][1]["young"] == 150
    for scale in ("precipitates", "grains"):
        cfg = load_config(al.alloy_configuration(alloy, scale, shape=16))
        assert cfg.homogenization.properties == ["stiffness"]
    with pytest.raises(ValueError, match="scale: one of"):
        al.alloy_configuration(alloy, "atoms")


def test_the_elasticity_of_an_alloy(tmp_path, capsys):
    """Each scale within the Hashin–Shtrikman bounds of its phases at their realized
    fractions, the grains on the matrix of the precipitates; a dual-phase steel of cubic
    ferrite crystals turned at random; the command line."""
    out = al.homogenize_alloy(ALUMINIUM, shape=24)
    for entry in out["scales"].values():
        for modulus in ("bulk_modulus", "shear_modulus"):
            lo, hi = entry["bounds"][modulus]["hs_lower"], entry["bounds"][modulus]["hs_upper"]
            assert lo * (1 - 2e-3) <= entry[modulus] <= hi * (1 + 2e-3), modulus
        k, g = entry["bulk_modulus"], entry["shear_modulus"]
        assert entry["young"] == pytest.approx(9 * k * g / (3 * k + g))
    assert out["departure"] == pytest.approx((out["young"] - 71.7) / 71.7)
    steel = {"alloy": "DP600 (illustrative)", "grain_size": 10.0, "phases": [
        {"name": "ferrite", "role": "matrix", "cubic": [231.4, 134.7, 116.4]},
        {"name": "martensite", "role": "grains", "fraction": 0.2, "young": 200.0,
         "poisson": 0.3}]}  # fmt: skip
    dual = al.homogenize_alloy(steel, shape=16)["scales"]["grains"]
    assert "bounds" not in dual and 190.0 < dual["young"] < 230.0  # Hill's iron: 210
    path = tmp_path / "alloy.json"
    path.write_text(json.dumps(ALUMINIUM))
    from morphogen.cli import main

    assert main(["alloy", str(path), "--config", str(tmp_path / "cfg"), "--homogenize",
                 "--shape", "16", "--json", str(tmp_path / "out.json")]) == 0  # fmt: skip
    printed = capsys.readouterr().out
    assert "precipitates: E = " in printed and "departure" in printed
    assert (tmp_path / "cfg" / "alloy_grains.yaml").is_file()
    assert json.loads((tmp_path / "out.json").read_text())["alloy"] == ALUMINIUM["alloy"]
