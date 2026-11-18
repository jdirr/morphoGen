"""Viscous flow through the pores of voxel images (the lattice Boltzmann method with two
relaxation times): Darcy's permeability.

Exact references: the flow between two planes (the parabola at the nodes, whatever the
viscosity of the lattice), the flow along a square duct, square arrays of cylinders
across and along the flow (Sangani & Acrivos 1982; Drummond & Tahir 1984), simple cubic
arrays of spheres (Zick & Homsy 1982), closed pores.
"""

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import VOID, Material, homogenize, permeability_tensor, stokes_flow
from morphogen.homogenization.flow import ENTRIES, MAGIC, pore_mask

SOLID = Material(young=1.0, poisson=0.3)


def slit(n=8, width=6):
    """Solid planes normal to y, ``width`` voxels of fluid (phase 0) between them."""
    data = np.ones((1, n, 1), np.uint8)
    data[0, :width, 0] = 0
    return mg.VoxelImage(data, 1.0, periodic=True)


def cylinders(n, fraction):
    """A square array of solid cylinders (phase 1) along z, one voxel thick."""
    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    data = (X**2 + Y**2 < fraction / np.pi).astype(np.uint8)[:, :, None]
    return mg.VoxelImage(data, 1.0 / n, periodic=True)


def spheres(n, fraction):
    """A simple cubic array of solid spheres (phase 1)."""
    radius = (3 * fraction / (4 * np.pi)) ** (1 / 3)
    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return mg.VoxelImage((X**2 + Y**2 + Z**2 < radius**2).astype(np.uint8), 1.0 / n, periodic=True)


@pytest.mark.parametrize("relaxation", [1.0, 4.0])
def test_the_flow_between_two_planes_is_exact(relaxation):
    """A parabola at the nodes, the walls midway between the fluid and the solid nodes:
    the permeability ``H^2 / 12 + 1 / 24`` of a channel of H voxels (the mean of the
    parabola at its nodes), whatever the viscosity of the lattice."""
    n, width = 8, 6
    flow = stokes_flow(slit(n, width), [1, 0, 0], pores=0, tol=1e-10, relaxation=relaxation,
                       fields=True)  # fmt: skip
    y = np.arange(width) + 0.5
    assert np.allclose(flow.field[0, :width, 0, 0], 0.5 * y * (width - y), atol=1e-7)
    assert np.all(flow.field[0, width:, 0] == 0.0)
    exact = (width / n) * (width**2 / 12 + 1 / 24)
    assert flow.velocity[0] == pytest.approx(exact, rel=1e-8)
    assert np.abs(flow.velocity[1:]).max() < 1e-9 and flow.porosity == width / n
    assert flow.error < 1e-9 and flow.steps > 0
    # the same in the units of the image, and without a flow across the planes
    image = mg.VoxelImage(slit(n, width).data, 0.5, periodic=True)
    out = permeability_tensor(image, {0: VOID, 1: SOLID}, tol=1e-9)
    assert out["k11"] == pytest.approx(0.25 * exact, rel=1e-7)
    assert out["k33"] == pytest.approx(out["k11"], rel=1e-7) and out["k22"] == 0.0
    assert out["mean"] == pytest.approx(2 * out["k11"] / 3)
    assert np.allclose(out["tensor"], out["tensor"].T) and set(ENTRIES) <= set(out)


def test_the_flow_along_a_square_duct():
    """``k = 0.035144 a^2`` for a duct of side ``a`` (the series of the Poisson problem),
    by the lattice Boltzmann flow and by the finite differences of the thermal
    permeability, which is the same problem."""
    from morphogen.homogenization.acoustics import thermal_permeability

    n, side = 20, 16
    data = np.ones((n, n, 1), np.uint8)
    data[:side, :side, 0] = 0
    image = mg.VoxelImage(data, 1.0, periodic=True)
    exact = (side / n) ** 2 * 0.035144 * side**2
    flow = stokes_flow(image, [0, 0, 1], pores=0, tol=1e-7)
    assert flow.velocity[2] == pytest.approx(exact, rel=5e-3)
    assert thermal_permeability(image, pores=0) == pytest.approx(exact, rel=5e-3)
    assert thermal_permeability(image, pores=0) == pytest.approx(flow.velocity[2], rel=5e-3)


def test_square_arrays_of_cylinders():
    """Across the cylinders: the drag ``F / (mu U)`` of Sangani & Acrivos (1982), 51.53 and
    102.90 at the fractions 0.2 and 0.3, ``k = pi a^2 / (c F)``; along them: ``k = a^2
    (-ln c - 1.476 + 2 c - c^2 / 2) / (4 c)`` (Drummond & Tahir 1984). The cylinders are
    drawn in voxels: a few percent."""
    for fraction, drag in ((0.2, 51.53), (0.3, 102.90)):
        image = cylinders(48, fraction)
        c = float(image.data.mean())
        radius2 = c / np.pi
        out = permeability_tensor(image, pores=0, tol=1e-5)
        assert out["porosity"] == pytest.approx(1 - c)
        assert out["k11"] == pytest.approx(np.pi * radius2 / (fraction * drag), rel=0.05)
        assert out["k22"] == pytest.approx(out["k11"], rel=1e-3)
        along = radius2 * (-np.log(c) - 1.476 + 2 * c - 0.5 * c**2) / (4 * c)
        assert out["k33"] == pytest.approx(along, rel=0.02)
        assert out["k33"] > 2 * out["k11"]
        assert abs(out["k12"]) < 1e-6 * out["k11"]


def test_simple_cubic_arrays_of_spheres():
    """The drag of Zick & Homsy (1982), ``K = F / (6 pi mu a U)``: 4.292 and 7.442 at the
    fractions 0.125 and 0.216; ``k = 2 a^2 / (9 c K)``."""
    for fraction, drag in ((0.125, 4.292), (0.216, 7.442)):
        image = spheres(24, fraction)
        c = float(image.data.mean())
        radius2 = (3 * c / (4 * np.pi)) ** (2 / 3)
        flow = stokes_flow(image, [1, 0, 0], pores=0, tol=1e-5)
        assert flow.velocity[0] == pytest.approx(2 * radius2 / (9 * c * drag), rel=0.05)
        assert np.abs(flow.velocity[1:]).max() < 1e-6 * flow.velocity[0]


def test_closed_pores_and_oblique_forces():
    """A cavity holds a fluid at rest, found at once; a force oblique to the axes of a
    slit drives its components along the planes only."""
    data = np.ones((8, 8, 8), np.uint8)
    data[2:6, 2:6, 2:6] = 0
    closed = permeability_tensor(mg.VoxelImage(data, 1.0, periodic=True), pores=0)
    assert np.all(closed["tensor"] == 0.0) and closed["steps"] <= 3 * 1000
    # a slab closed across: no flow there, whatever the momentum of its nodes
    data = np.ones((6, 6, 1), np.uint8)
    data[:, 1:4, 0] = 0
    slab = permeability_tensor(mg.VoxelImage(data, 1.0, periodic=True), pores=0, tol=1e-9)
    assert slab["k22"] == 0.0 and slab["k11"] == pytest.approx(0.5 * (9 / 12 + 1 / 24), rel=1e-7)
    assert closed["porosity"] == pytest.approx(0.125)
    flow = stokes_flow(slit(), np.array([1.0, 1.0, 1.0]), pores=0, tol=1e-9)
    k = (6 / 8) * (6**2 / 12 + 1 / 24)
    assert flow.velocity == pytest.approx(k * np.array([1, 0, 1]) / np.sqrt(3), rel=1e-7, abs=1e-9)
    partial = permeability_tensor(slit(), pores=0, axes=(0,), tol=1e-9)
    assert partial["k11"] == pytest.approx(k, rel=1e-7) and partial["k33"] == 0.0
    assert partial["mean"] == pytest.approx(k, rel=1e-7)


def test_flows_check_their_images():
    assert MAGIC == 3 / 16
    image = slit()
    assert pore_mask(image, {0: VOID, 1: SOLID}).sum() == 6
    assert pore_mask(image, pores=[1]).sum() == 2
    with pytest.raises(ValueError, match="no pore"):
        pore_mask(image, {0: SOLID, 1: SOLID})
    with pytest.raises(ValueError, match="materials"):
        pore_mask(image)
    with pytest.raises(TypeError, match="VoxelImage"):
        pore_mask(image.data, pores=0)
    with pytest.raises(ValueError, match="cubic voxels"):
        stokes_flow(mg.VoxelImage(image.data, (1.0, 2.0, 1.0), periodic=True), [1, 0, 0], pores=0)
    # an image that is not flagged periodic is the cell of a periodic medium all the same
    cell = mg.VoxelImage(image.data, 1.0, periodic=False)
    assert stokes_flow(cell, [1, 0, 0], pores=0, tol=1e-9).velocity[0] == pytest.approx(
        (6 / 8) * (6**2 / 12 + 1 / 24), rel=1e-7
    )
    with pytest.raises(ValueError, match="direction"):
        stokes_flow(image, [0, 0, 0], pores=0)
    with pytest.raises(ValueError, match="relaxation"):
        stokes_flow(image, [1, 0, 0], pores=0, relaxation=0.5)
    with pytest.raises(ValueError, match="axes"):
        permeability_tensor(image, pores=0, axes=(0, 3))
    solid = mg.VoxelImage(np.ones((2, 2, 2), np.uint8), 1.0, periodic=True)
    assert stokes_flow(solid, [1, 0, 0], pores=0).steps == 0


def test_permeability_in_homogenize_and_configurations(tmp_path, capsys):
    import json

    import yaml

    from helpers import box_mesh
    from morphogen import cli, pipeline
    from morphogen.config import ConfigError, load_config

    image = cylinders(24, 0.2)
    out = homogenize(image, {0: VOID, 1: SOLID}, properties=["hydraulic_permeability"])
    res = out["hydraulic_permeability.periodic"]
    assert res["k33"] > res["k11"] > 0.0 and res["tensor"].shape == (3, 3)
    with pytest.raises(ValueError, match="voxel images"):
        homogenize(box_mesh(), {0: SOLID}, properties=["hydraulic_permeability"])
    data = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
        "domain": {"periodic": True},
        "seed": 4,
        "voxels": {"voxel_size": 0.25},
        "homogenization": {"method": "fft",
                           "materials": {0: "void", 1: {"young": 1, "poisson": 0.3}},
                           "properties": ["hydraulic_permeability"], "tolerance": 1e-4},
        "rve": {"sizes": [4.0, 5.0], "realizations": [3, 2], "bootstrap": 0,
                "properties": ["hydraulic_permeability.mean", "hydraulic_permeability.k11.periodic",
                               "volume_fraction"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "bed"},
    }  # fmt: skip
    outdir = pipeline.run(load_config(data))
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    rows = pipeline.read_rows(outdir / "results.csv")
    stats = report["properties"]["hydraulic_permeability.mean.periodic"]["statistics"]
    assert [s["n"] for s in stats] == [3, 2]
    first = [r["hydraulic_permeability.periodic.mean"] for r in rows
             if r["volume"] == pytest.approx(64.0)]  # fmt: skip
    assert stats[0]["mean"] == pytest.approx(np.mean(first), rel=1e-12) and first[0] > 0.0
    assert all(0.0 < r["hydraulic_permeability.periodic.porosity"] < 1.0 for r in rows)
    homog = data["homogenization"]
    for change, match in (
        ({"materials": {0: {"young": 1, "poisson": 0.3}, 1: {"young": 2, "poisson": 0.3}}},
         "need pores"),
        ({"method": "xfem"}, "voxel images"),
    ):  # fmt: skip
        with pytest.raises(ConfigError, match=match):
            load_config({**data, "homogenization": {**homog, **change}})
    with pytest.raises(ConfigError, match="unknown result 'k44'"):
        load_config({**data, "rve": {**data["rve"], "properties": ["hydraulic_permeability.k44"]}})
    # the command line
    path = tmp_path / "rods.vti"
    mg.io.save_image(image, path)
    (tmp_path / "materials.yaml").write_text(
        yaml.safe_dump({0: "void", 1: {"young": 1.0, "poisson": 0.3}}), encoding="utf-8"
    )
    args = ["homogenize", str(path), "--materials", str(tmp_path / "materials.yaml"),
            "--property", "hydraulic_permeability"]  # fmt: skip
    assert cli.main(args) == 0
    assert "permeability tensor" in capsys.readouterr().out
    assert cli.main([*args, "--json"]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["hydraulic_permeability.periodic"]["k33"] == pytest.approx(res["k33"], rel=1e-3)
