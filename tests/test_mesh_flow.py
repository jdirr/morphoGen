"""The flow through the pores of a conforming mesh of space (homogenization.mesh_flow): the
channel of Poiseuille on the layer of a mesh of the plane exactly, by the factorization and
by MINRES, a square duct as on the plane, the simple cubic arrays of spheres against Zick &
Homsy (1982), the lattice Boltzmann method on the voxels of the same spheres, and the flow
in homogenize and the configurations."""

from __future__ import annotations

import numpy as np
import pytest

import morphogen as mg
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import Material, homogenize
from morphogen.homogenization.mesh_flow import mesh_permeability
from morphogen.homogenization.plane_flow import plane_permeability
from morphogen.mesh.conforming import has_mmg
from morphogen.mesh.triangles import TriMesh, background_triangulation
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Spheres

needs_mmg = pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")

#: The pores (phase 0) about a rigid solid (phase 1)
PORES = {0: Material(void=True), 1: Material(young=1.0, poisson=0.3)}


def _plane(n: int, inside) -> TriMesh:
    """A six-node mesh of the periodic unit square, its pores (phase 0) where ``inside``
    holds at the centroids of the triangles of a grid of ``n`` squares."""
    nodes, tri, _, _ = background_triangulation((n, n), (0.0, 0.0), (1.0, 1.0))
    phase = np.where(inside(nodes[tri].mean(axis=1)), 0, 1)
    return TriMesh(nodes, tri, phase, (0.0, 0.0), (1.0, 1.0), periodic=True).quadratic()


def _sphere(fraction: float):
    """One sphere in a periodic unit cube: a simple cubic array."""
    radius = (3.0 * fraction / (4.0 * np.pi)) ** (1.0 / 3.0)
    domain = mg.Domain.cube(1.0, periodic=True)
    return PrimitiveUnion(domain, Spheres(np.array([[0.47, 0.52, 0.49]]), np.array([radius])),
                          None), radius  # fmt: skip


def test_channel_of_poiseuille_in_space():
    """The channel of the plane between y = 0.3 and 0.7 on its layer of ten-node
    tetrahedra: the parabola of Poiseuille, ``k = H^3 / 12 L`` along x and along z, nothing
    across, to round-off, by the factorization and by MINRES (preconditioned)."""
    layer = _plane(10, lambda c: np.abs(c[:, 1] - 0.5) < 0.2).layer()
    exact = 0.4**3 / 12.0
    for solver in ("direct", "minres"):
        out = mesh_permeability(layer, PORES, solver=solver, tol=1e-12)
        assert out["solver"] == solver and out["parts"] == 1
        assert out["k11"] == pytest.approx(exact, rel=1e-10)
        assert out["k33"] == pytest.approx(exact, rel=1e-10)
        assert abs(out["k22"]) < 1e-8 * exact and abs(out["k12"]) < 1e-8 * exact
    assert out["iterations"] > 0 and out["porosity"] == pytest.approx(0.4)


def test_square_duct_as_on_the_plane():
    """A square duct of side 0.5 along z: the permeability of its flow along the duct,
    ``0.0351443 a^2`` times its share of the cell, on the layer of the mesh of the plane as
    by the flow along z of the plane, within 1e-5 of each other and 4e-4 of the series."""
    duct = _plane(16, lambda c: np.all(np.abs(c - 0.5) < 0.25, axis=1))
    reference = 0.25 * 0.0351443 * 0.25
    plane = plane_permeability(duct, PORES)
    space = mesh_permeability(duct.layer(), PORES)
    assert space["k33"] == pytest.approx(plane["k33"], rel=1e-5)
    assert space["k33"] == pytest.approx(reference, rel=4e-4)
    assert abs(space["k11"]) < 1e-12


@needs_mmg
def test_simple_cubic_arrays_of_spheres():
    """The drag of Zick & Homsy (1982), ``K = F / (6 pi mu a U)`` = 4.292 at the fraction
    0.125, ``k = 2 a^2 / (9 c K)``: within 1e-3 on curved ten-node tetrahedra of size 0.2
    (7.5e-4; 2.2e-4 at 0.12, 5e-5 at 0.08), along the three axes; MINRES about a hundred
    iterations per force."""
    micro, radius = _sphere(0.125)
    mesh = mg.mesh.conforming_mesh(micro, size=0.2, order=2, phases="all", periodic=True)
    out = mesh_permeability(mesh, PORES)
    assert out["solver"] == "minres" and out["iterations"] < 3 * 200
    reference = 2.0 * radius**2 / (9.0 * 0.125 * 4.292)
    for key in ("k11", "k22", "k33"):
        assert out[key] == pytest.approx(reference, rel=1e-3)
    assert abs(out["k12"]) < 1e-3 * out["k11"]
    assert out["porosity"] == pytest.approx(0.875, rel=1e-5)


@needs_mmg
def test_homogenize_and_configurations_of_the_flow_in_space(tmp_path):
    """``homogenize`` of a conforming mesh with its pores: the permeability under periodic
    conditions; the configurations with ``method: fe`` and a mesh, the pores meshed with the
    solid, the velocities of its flows written with ``fields``; ``auto`` keeps the lattice
    Boltzmann method on voxels."""
    micro, _ = _sphere(0.125)
    mesh = mg.mesh.conforming_mesh(micro, size=0.2, order=2, phases="all", periodic=True)
    out = homogenize(mesh, PORES, bcs=["periodic"], properties=["hydraulic_permeability"])
    flow = out["hydraulic_permeability.periodic"]
    assert flow["k11"] == pytest.approx(mesh_permeability(mesh, PORES)["k11"], rel=1e-6)
    with pytest.raises(ValueError, match="periodic conditions"):
        homogenize(mesh, PORES, bcs=["kubc"], properties=["hydraulic_permeability"])
    with pytest.raises(ValueError, match="voxel images"):
        homogenize(mesh, PORES, properties=["acoustic"])
    base = {
        "model": {"type": "hardcore_spheres", "radius": 0.2, "volume_fraction": 0.1},
        "domain": {"size": 1.0, "periodic": True},
        "seed": 1,
        "homogenization": {"method": "fe", "materials": {0: "void", 1: {"young": 1.0,
                                                                         "poisson": 0.3}},
                           "boundary_conditions": ["periodic"],
                           "properties": ["hydraulic_permeability"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "spheres"},
    }  # fmt: skip
    cfg = load_config({**base, "mesh": {"size": 0.25}}, base_dir=tmp_path)
    assert cfg.homogenization.method == "fe"
    from morphogen import pipeline

    outdir = pipeline.run(cfg)
    header, row = (outdir / "results.csv").read_text().splitlines()[:2]
    values = dict(zip(header.split(","), row.split(","), strict=True))
    assert float(values["hydraulic_permeability.periodic.k11"]) > 0.0
    fields = {**base["homogenization"], "fields": True}
    outdir = pipeline.run(load_config({**base, "homogenization": fields, "mesh": {"size": 0.25},
                                       "outputs": {**base["outputs"], "directory":
                                                   str(tmp_path / "fields")}},
                                      base_dir=tmp_path))  # fmt: skip
    point = mg.io.read_vtu(next(outdir.glob("spheres_*_hydraulic_permeability_periodic.vtu")))
    assert np.asarray(point["point_data"]["velocity_z"]).shape[1] == 3
    auto = {**base["homogenization"], "method": "auto"}
    voxels = {**base, "homogenization": auto, "voxels": {"shape": 24}}
    assert load_config(voxels, base_dir=tmp_path).homogenization.method == "fft"
    kubc = {**base["homogenization"], "boundary_conditions": ["kubc"]}
    with pytest.raises(ConfigError, match="under periodic"):
        load_config({**base, "homogenization": kubc, "mesh": {"size": 0.25}}, base_dir=tmp_path)


@pytest.mark.slow
@needs_mmg
def test_lattice_boltzmann_on_the_voxels_of_the_same_spheres():
    """The simple cubic array of spheres at 0.216 (Zick & Homsy: 7.442): the finite elements
    within 1e-3 of the drag, the lattice Boltzmann method on 40 voxels a side within 3 % of
    them, its spheres stairs."""
    from morphogen.homogenization.flow import permeability_tensor

    micro, radius = _sphere(0.216)
    mesh = mg.mesh.conforming_mesh(micro, size=0.1, order=2, phases="all", periodic=True)
    fe = mesh_permeability(mesh, PORES)
    reference = 2.0 * radius**2 / (9.0 * 0.216 * 7.442)
    assert fe["k11"] == pytest.approx(reference, rel=1e-3)
    lbm = permeability_tensor(micro.voxelize(shape=40), PORES, tol=1e-5)
    assert lbm["k11"] == pytest.approx(fe["k11"], rel=0.03)
