"""The flow through the pores of a structure of the plane (homogenization.plane_flow): the
channel of Poiseuille exactly, the square arrays of cylinders across them against the drags
of Sangani & Acrivos (1982) and along them against Drummond & Tahir (1984), closed pores at
rest, the lattice Boltzmann method on the voxels of the same fibres, and the flow in
homogenize, the configurations and the runs."""

from __future__ import annotations

import numpy as np
import pytest

import morphogen as mg
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import Material, homogenize
from morphogen.homogenization.plane_flow import plane_permeability, pore_phases
from morphogen.mesh.conforming import mmg_executable
from morphogen.mesh.triangles import TriMesh, background_triangulation, plane_mesh
from morphogen.models.long_fibers import LongFibers


def _has_mmg2d() -> bool:
    try:
        mmg_executable(None, "mmg2d")
    except (ImportError, FileNotFoundError):
        return False
    return True


needs_mmg2d = pytest.mark.skipif(not _has_mmg2d(), reason="mmg2d (pymmg) is not installed")

#: The pores (phase 0) about rigid fibres (phase 1)
BED = {0: Material(void=True), 1: Material(young=230e3, poisson=0.2)}


def _channel(n: int = 10) -> TriMesh:
    """A channel between y = 0.3 and 0.7 (phase 0, the pores) in a periodic unit cell, its
    walls (phase 1) along the lines of the grid: six-node triangles."""
    nodes, tri, _, _ = background_triangulation((n, n), (0.0, 0.0), (1.0, 1.0))
    centres = nodes[tri].mean(axis=1)
    phase = np.where(np.abs(centres[:, 1] - 0.5) < 0.2, 0, 1)
    return TriMesh(nodes, tri, phase, (0.0, 0.0), (1.0, 1.0), periodic=True).quadratic()


def _square_array(fraction: float, thickness: float = 0.5):
    """Long straight fibres in a square array, four in a periodic box of edge 2: a square
    array of cylinders, one per unit cell."""
    radius = float(np.sqrt(fraction / np.pi))
    return LongFibers(radius=radius, volume_fraction=fraction, arrangement="square").generate(
        mg.Domain(size=(2.0, 2.0, thickness), periodic=True), seed=0
    )


def test_channel_of_poiseuille():
    """A channel of height 0.4 in a cell of height 1: the parabola of Poiseuille in the
    quadratic elements exactly, ``k = H^3 / 12 L`` along the channel and along z, nothing
    across; its velocities at the nodes those of the parabola."""
    mesh = _channel()
    out = plane_permeability(mesh, BED, fields=True)
    exact = 0.4**3 / 12.0
    assert out["k11"] == pytest.approx(exact, rel=1e-12)
    assert out["k33"] == pytest.approx(exact, rel=1e-12)
    assert abs(out["k22"]) < 1e-15 and abs(out["k12"]) < 1e-15
    assert out["porosity"] == pytest.approx(0.4) and out["parts"] == 1
    assert out["mean"] == pytest.approx(2.0 * exact / 3.0, rel=1e-12)
    y = out["pores"].nodes[:, 1]
    parabola = 0.5 * (y - 0.3) * (0.7 - y)  # the velocity under a unit force, viscosity 1
    np.testing.assert_allclose(out["velocity"][0][:, 0], parabola, atol=1e-13)
    np.testing.assert_allclose(out["velocity"][2][:, 2], parabola, atol=1e-13)
    assert np.abs(out["velocity"][1]).max() < 1e-13
    # the viscosity does not change the permeability; the pores named by their phase
    again = plane_permeability(mesh, pores=[0], viscosity=3.0)
    np.testing.assert_allclose(again["tensor"], out["tensor"], rtol=1e-12, atol=1e-16)
    assert pore_phases(mesh, BED) == [0]


@needs_mmg2d
@pytest.mark.parametrize(("fraction", "drag"), [(0.2, 51.53), (0.3, 102.90)])
def test_square_arrays_of_cylinders(fraction, drag):
    """Across the cylinders: the drag ``F / (mu U)`` of Sangani & Acrivos (1982), 51.53 and
    102.90 at the fractions 0.2 and 0.3, ``k = pi a^2 / (c F) = 1 / F`` for a unit cell,
    within 2e-4 (the digits of the drag) on curved six-node triangles of size 0.05, the
    straight ones 1e-3 to 1e-2 off; along them: ``k = a^2 (-ln c - 1.476 + 2 c - c^2 / 2) /
    (4 c)`` (Drummond & Tahir 1984), within the 3e-3 of the series."""
    micro = _square_array(fraction)
    radius2 = fraction / np.pi
    along = (
        radius2 * (-np.log(fraction) - 1.476 + 2 * fraction - 0.5 * fraction**2) / (4 * fraction)
    )
    errors = []
    for curved in (True, False):
        mesh = plane_mesh(micro, size=0.05, order=2, curved=curved)
        out = plane_permeability(mesh, BED)
        errors.append(abs(out["k11"] * drag - 1.0))
        if curved:
            assert out["k22"] == pytest.approx(out["k11"], rel=2e-4)
            assert abs(out["k12"]) < 1e-5 * out["k11"]
            assert out["k33"] == pytest.approx(along, rel=3e-3)
            assert out["k33"] > 2.0 * out["k11"]
            assert out["porosity"] == pytest.approx(1.0 - fraction, rel=1e-6)
    assert errors[0] < 2e-4 and errors[1] > 1e-3


@needs_mmg2d
def test_closed_pores_and_pipes():
    """Pores that do not meet in the plane (the fibres of an array are the pores, the matrix
    solid): across the plane a fluid at rest in each, its pressure fixed in each of them
    (the flow 1e-7 of that along z, the pressures linear in the curved elements to their
    bending); along z the pipes of Hagen and Poiseuille, ``k33 = c a^2 / 8``, within 1e-5
    on curved six-node triangles of size 0.05 (7.5e-5 at 0.1, 3.7e-7 at 0.025)."""
    fraction = 0.3
    mesh = plane_mesh(_square_array(fraction), size=0.05, order=2)
    tubes = {0: Material(young=1.0, poisson=0.3), 1: Material(void=True)}
    out = plane_permeability(mesh, tubes, fields=True)
    assert out["parts"] == 4 and out["porosity"] == pytest.approx(fraction, rel=1e-5)
    pipes = fraction * (fraction / np.pi) / 8.0
    assert out["k33"] == pytest.approx(pipes, rel=1e-5)
    assert np.abs(out["tensor"][:2, :2]).max() < 1e-7 * pipes
    with pytest.raises(ValueError, match="without walls"):
        plane_permeability(mesh, pores=[0, 1])


@needs_mmg2d
def test_homogenize_the_flow_of_the_plane():
    """``homogenize`` of a mesh of the plane: the permeability under periodic conditions
    beside the stiffness under every condition (the pores left out of its elastic problems);
    the pores left out of the mesh, and the flow without periodic conditions, refused."""
    micro = _square_array(0.3)
    mesh = plane_mesh(micro, size=0.1, order=2)
    out = homogenize(mesh, BED, bcs=["periodic", "kubc"],
                     properties=["hydraulic_permeability", "stiffness"])  # fmt: skip
    assert "hydraulic_permeability.periodic" in out and "hydraulic_permeability.kubc" not in out
    flow = out["hydraulic_permeability.periodic"]
    assert flow["k11"] == pytest.approx(plane_permeability(mesh, BED)["k11"], rel=1e-12)
    assert {"stiffness.periodic", "stiffness.kubc"} <= set(out)
    with pytest.raises(ValueError, match="periodic conditions"):
        homogenize(mesh, BED, bcs=["kubc"], properties=["hydraulic_permeability"])
    solid = plane_mesh(micro, size=0.1, order=2, phases="solid")
    with pytest.raises(ValueError, match="left out of the mesh"):
        homogenize(solid, BED, properties=["hydraulic_permeability"])
    with pytest.raises(ValueError, match="not by the finite elements of the plane"):
        homogenize(mesh, BED, properties=["acoustic"])


def test_configurations_of_the_flow_of_the_plane(tmp_path):
    """The permeability on the meshes of the plane (method fe, mesh.plane true) and not on
    the tetrahedra; the template of the fibre bed."""
    base = {
        "model": {"type": "long_fibers", "radius": 0.15, "volume_fraction": 0.4,
                  "arrangement": "square"},
        "domain": {"size": [1.0, 1.0, 0.25], "periodic": True},
        "homogenization": {"method": "fe",
                           "materials": {0: "void", 1: {"young": 1.0, "poisson": 0.2}},
                           "boundary_conditions": ["periodic"],
                           "properties": ["hydraulic_permeability"]},
        "outputs": {"directory": str(tmp_path / "out")},
    }  # fmt: skip
    cfg = load_config({**base, "mesh": {"size": 0.1, "plane": True}}, base_dir=tmp_path)
    assert cfg.homogenization.method == "fe" and cfg.mesh.order == 2
    with pytest.raises(ConfigError, match="voxel images"):
        load_config({**base, "mesh": {"size": 0.1}}, base_dir=tmp_path)
    auto = {**base, "homogenization": {**base["homogenization"], "method": "auto"}}
    assert load_config({**auto, "mesh": {"size": 0.1, "plane": True}},
                       base_dir=tmp_path).homogenization.method == "fe"  # fmt: skip
    voxels = {**auto, "voxels": {"voxel_size": 0.05}}
    assert load_config(voxels, base_dir=tmp_path).homogenization.method == "fft"
    with pytest.raises(ConfigError, match="without periodic conditions"):
        kubc = {**base["homogenization"], "boundary_conditions": ["kubc"]}
        load_config({**base, "homogenization": kubc, "mesh": {"size": 0.1, "plane": True}},
                    base_dir=tmp_path)  # fmt: skip


@needs_mmg2d
def test_run_of_a_fibre_bed(tmp_path):
    """A run of fibres in a square array, the permeability in its table of results, its
    pores meshed with the fibres."""
    from morphogen import pipeline

    spec = {
        "model": {"type": "long_fibers", "radius": 0.15, "volume_fraction": 0.3,
                  "arrangement": "square"},
        "domain": {"size": [1.0, 1.0, 0.25], "periodic": True},
        "seed": 1,
        "mesh": {"size": 0.1, "plane": True},
        "homogenization": {"method": "fe",
                           "materials": {0: "void", 1: {"young": 1.0, "poisson": 0.2}},
                           "boundary_conditions": ["periodic"],
                           "properties": ["hydraulic_permeability"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "bed",
                    "conforming_mesh": ["vtu"]},
    }  # fmt: skip
    outdir = pipeline.run(load_config(spec, base_dir=tmp_path))
    text = (outdir / "results.csv").read_text().splitlines()
    header, row = text[0].split(","), text[1].split(",")
    values = dict(zip(header, row, strict=True))
    k = float(values["hydraulic_permeability.periodic.k11"])
    assert k > 0.0 and float(values["hydraulic_permeability.periodic.k22"]) == pytest.approx(
        k, rel=3e-3
    )
    mesh = mg.io.read_tri_mesh(next(outdir.glob("bed_*_tri.vtu")))
    assert set(np.unique(mesh.phase)) == {0, 1}


@pytest.mark.slow
@needs_mmg2d
def test_lattice_boltzmann_on_the_voxels_of_the_same_fibres():
    """A random bed of twelve fibres (0.38 of the area): the lattice Boltzmann method on
    96 x 96 voxels of its sections, their walls stairs, within 8 % across the fibres and 1 %
    along them of the finite elements of the plane, converged to 1e-4."""
    from morphogen.homogenization.flow import permeability_tensor

    micro = LongFibers(radius=0.1, number=12, arrangement="random", min_gap=0.02).generate(
        mg.Domain(size=(1.0, 1.0, 1.0 / 96.0), periodic=True), seed=2
    )
    fe = plane_permeability(plane_mesh(micro, size=0.0125, order=2), BED)
    lbm = permeability_tensor(micro.voxelize((96, 96, 1)), BED, tol=1e-5)
    for key, tol in (("k11", 0.08), ("k22", 0.08), ("k33", 0.01)):
        assert lbm[key] == pytest.approx(fe[key], rel=tol)
    assert lbm["porosity"] == pytest.approx(fe["porosity"], abs=2e-3)
