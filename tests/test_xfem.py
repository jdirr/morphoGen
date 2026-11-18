"""Extended finite elements of level sets (:mod:`morphogen.homogenization.xfem`).

The interfaces cut the elements of a regular grid: plane interfaces of any orientation
are represented exactly, so that laminates oblique to the grid (or between its planes)
are solved to round-off, in elasticity, in conduction, under thermal strains and along
loading paths; curved interfaces converge faster than the stairs of voxels; voids are
integrated on their solid side.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

import morphogen as mg
from helpers import backus
from morphogen.homogenization import (
    VOID,
    LevelSetSample,
    Loading,
    Material,
    NonlinearXFEMProblem,
    XFEMElasticProblem,
    XFEMThermalProblem,
    conductivity_tensor,
    homogenize,
    make_problem,
    simulate_point,
    stiffness_tensor,
    thermal_expansion,
)
from morphogen.homogenization.coupled import laminate
from morphogen.homogenization.laws import analytic_model
from morphogen.homogenization.materials import rotate_stiffness
from morphogen.homogenization.xfem import _gradients, subtetrahedra
from morphogen.mesh.conforming import ImplicitBody
from morphogen.orientations import matrix_to_euler

SOFT = Material(young=1.0, poisson=0.2, conductivity=1.0, expansion=1e-5)
STIFF = Material(young=20.0, poisson=0.35, conductivity=30.0, expansion=3e-5)
MATS = {0: SOFT, 1: STIFF}
#: The axes of layers normal to (1, 1, 0): two directions in their plane, their normal
NORMAL = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
AXES = np.stack([np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0), np.array([0.0, 0.0, -1.0]), NORMAL])


def oblique(fraction=0.5, shape=(9, 9, 2)):
    """Layers of the phase 1 normal to (1, 1, 0), of period 1 along x + y: their level
    set is linear in the elements that their interfaces cut."""

    def level_set(points):
        s = (points[:, 0] + points[:, 1]) % 1.0
        return (np.abs(s - 0.5) - fraction / 2.0) / np.sqrt(2.0)

    return LevelSetSample.of(ImplicitBody(mg.Domain.cube(1.0, periodic=True), level_set), shape)


def sphere(radius=0.35, shape=8):
    def level_set(points):
        return np.linalg.norm(points % 1.0 - 0.5, axis=1) - radius

    return LevelSetSample.of(ImplicitBody(mg.Domain.cube(1.0, periodic=True), level_set), shape)


def voxels(radius, n):
    grid = mg.Grid(mg.Domain.cube(1.0, periodic=True), (n, n, n))
    inside = np.linalg.norm(grid.center_points() % 1.0 - 0.5, axis=1) < radius
    return mg.VoxelImage(inside.reshape(n, n, n).astype(np.uint8), 1.0 / n, periodic=True)


def test_sub_tetrahedra_fill_the_cut_elements():
    rng = np.random.default_rng(0)
    x = rng.random((200, 4, 3))
    phi = rng.standard_normal((200, 4))
    mixed = ((phi < 0).sum(axis=1) % 4) != 0
    x, phi = x[mixed], phi[mixed]
    verts, side = subtetrahedra(x, phi)
    vol = np.abs(np.linalg.det(verts[:, :, 1:] - verts[:, :, :1])) / 6.0
    grads, total = _gradients(x)
    assert np.allclose(vol.sum(axis=1), total, rtol=1e-12)
    assert np.all(vol[side == 0] == 0.0)
    # the level set at the centroid of each piece has the sign of its side
    centroids = verts.mean(axis=2)
    lam = np.einsum("mad,msd->msa", grads[:, 1:], centroids - x[:, None, 0])
    shape = np.concatenate([1.0 - lam.sum(axis=2, keepdims=True), lam], axis=2)
    values = np.einsum("msa,ma->ms", shape, phi)
    used = side != 0
    assert np.all(np.sign(values[used]) == side[used])
    with pytest.raises(ValueError, match="both signs"):
        subtetrahedra(x[:1], np.abs(phi[:1]))


def test_layers_oblique_to_the_grid_are_exact():
    """The stiffness of Backus and the conductivity of the layers, turned to their normal
    (1, 1, 0): to round-off on 9 x 9 x 2 cells, whose planes are none of the interfaces."""
    f1 = 0.5
    sample = oblique(f1)
    fractions = sample.phase_fractions()
    assert fractions[1] == pytest.approx(f1, rel=1e-12)
    problem = XFEMElasticProblem(sample, MATS, tol=1e-12)
    assert problem.enriched and problem.summary()["enriched"] > 0
    c, sols = stiffness_tensor(problem)
    layered = backus([SOFT.matrix("elastic"), STIFF.matrix("elastic")], [1 - f1, f1])
    exact = rotate_stiffness(layered, matrix_to_euler(AXES)[0])
    assert np.allclose(c, exact, rtol=1e-10, atol=1e-11)
    assert {p.phase: p.fraction for p in sols[0].phases} == pytest.approx({0: 0.5, 1: 0.5})
    k, _ = conductivity_tensor(XFEMThermalProblem(sample, MATS, tol=1e-12))
    across, along = 1.0 / (0.5 / 1.0 + 0.5 / 30.0), 0.5 * (1.0 + 30.0)
    assert np.allclose(k, along * np.eye(3) + (across - along) * np.outer(NORMAL, NORMAL),
                       rtol=1e-10)  # fmt: skip
    # the same layers on voxels: stairs, far from the exact stiffness at this resolution
    n = 16
    grid = mg.Grid(mg.Domain.cube(1.0, periodic=True), (n, n, 2))
    s = (grid.center_points()[:, 0] + grid.center_points()[:, 1]) % 1.0
    image = mg.VoxelImage((np.abs(s - 0.5) < f1 / 2).reshape(n, n, 2).astype(np.uint8),
                          (1 / n, 1 / n, 0.5), periodic=True)  # fmt: skip
    stairs, _ = stiffness_tensor(make_problem(image, MATS, "elastic"))
    assert np.abs(stairs - exact).max() > 0.02 * exact.max()


def test_interfaces_between_the_planes_of_the_grid():
    """Layers normal to z whose interfaces lie between the nodes: the thermal expansion
    of the laminate (the exact thermal stress of the layers), a uniaxial tension by the
    condensation of the macroscopic unknowns, and the fields."""

    def level_set(points):
        return np.abs(points[:, 2] % 1.0 - 0.5) - 0.23

    body = ImplicitBody(mg.Domain.cube(1.0, periodic=True), level_set)
    sample = LevelSetSample.of(body, (2, 2, 7))
    problem = XFEMElasticProblem(sample, MATS, tol=1e-12)
    laws = [m.matrix("elastic") for m in (SOFT, STIFF)]
    eigen = [law @ (m.expansion * np.array([1.0, 1, 1, 0, 0, 0]))
             for law, m in zip(laws, (SOFT, STIFF), strict=True)]  # fmt: skip
    c_exact, stress = laminate(laws, [0.54, 0.46], ("displacement",), eigen)
    c, _ = stiffness_tensor(problem)
    assert np.allclose(c, c_exact, rtol=1e-10, atol=1e-11)
    alpha, _ = thermal_expansion(problem)
    exact = np.linalg.solve(c_exact, stress)
    assert np.allclose(np.diag(alpha), exact[:3], rtol=1e-9) and abs(alpha[0, 1]) < 1e-15
    out = homogenize(sample, MATS, properties=["bulk_modulus"], tol=1e-12, fields=True,
                     loadings={"tension": {"stress": {"33": 2.0}}})  # fmt: skip
    pulled = out["tension.periodic"].components()
    compliance = np.linalg.inv(c_exact)
    assert pulled["E33"] == pytest.approx(2.0 * compliance[2, 2], rel=1e-9)
    assert pulled["E11"] == pytest.approx(2.0 * compliance[0, 2], rel=1e-9)
    assert abs(pulled["S11"]) < 1e-10
    sol = out["bulk_modulus.periodic"].solution
    assert sol.info.method.startswith("xfem")
    fields = sol.fields
    mesh = sol.support
    assert fields["stress"].shape == (mesh.n_elements, 6)
    assert fields["displacement"].shape == (mesh.n_nodes, 3)
    # the mean of the element stresses is the macroscopic stress
    volume = np.abs(mesh.volumes())
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
    mean = volume @ fields["stress"] / volume.sum()
    assert np.allclose(mean, [sol.stress[i, j] for i, j in pairs], rtol=1e-9, atol=1e-10)


def test_spheres_converge_faster_than_voxels():
    """A cubic array of spheres ten times stiffer and more conducting: 8 cells of
    extended finite elements are closer to the converged values than 32 voxels."""
    mats = {0: Material(young=1.0, poisson=0.3, conductivity=1.0),
            1: Material(young=10.0, poisson=0.2, conductivity=10.0)}  # fmt: skip
    reference = conductivity_tensor(XFEMThermalProblem(sphere(shape=20), mats))[0][0, 0]
    coarse = conductivity_tensor(XFEMThermalProblem(sphere(shape=8), mats))[0]
    assert np.allclose(coarse, coarse[0, 0] * np.eye(3), atol=1e-9)  # cubic: isotropic
    stairs = conductivity_tensor(make_problem(voxels(0.35, 32), mats, "thermal"))[0][0, 0]
    assert abs(coarse[0, 0] - reference) < 0.5 * abs(stairs - reference)
    assert abs(coarse[0, 0] - reference) < 5e-3 * reference
    c, _ = stiffness_tensor(XFEMElasticProblem(sphere(shape=8), mats))
    fine, _ = stiffness_tensor(make_problem(voxels(0.35, 32), mats, "elastic"))
    # the stiffness of the voxels comes down to that of the cut elements as they refine
    assert 0.98 * fine[0, 0] < c[0, 0] < fine[0, 0] and 0.98 * fine[3, 3] < c[3, 3] < fine[3, 3]
    assert np.allclose(c[[0, 1, 2], [0, 1, 2]], c[0, 0], rtol=1e-9)


def test_voids_and_insulators_are_integrated_on_their_solid_side():
    """A spherical pore: the cut elements are not enriched, their solid part alone is
    integrated. Linear tetrahedra are stiffer than the trilinear voxels: the stiffness
    comes down to that of fine voxels as the square of the size of the cells."""
    porous = {0: Material(young=1.0, poisson=0.3, conductivity=1.0), 1: VOID}
    sample = sphere(0.3, shape=8)
    problem = XFEMElasticProblem(sample, porous)
    assert not problem.enriched and problem.summary()["enriched"] == 0
    exact = 1.0 - 4.0 / 3.0 * np.pi * 0.3**3
    assert problem.material_fraction == pytest.approx(exact, rel=2e-2)
    assert problem.material_fraction == pytest.approx(sample.phase_fractions()[0], rel=1e-12)
    c, sols = stiffness_tensor(problem)
    assert sols[0].phases[0].phase == 0 and len(sols[0].phases) == 1
    fine, _ = stiffness_tensor(make_problem(voxels(0.3, 40), porous, "elastic"))
    finer, _ = stiffness_tensor(XFEMElasticProblem(sphere(0.3, shape=12), porous))
    for i in (0, 3):
        assert fine[i, i] < finer[i, i] < c[i, i]
        assert finer[i, i] == pytest.approx(fine[i, i], rel=3e-2)
        assert (finer[i, i] - fine[i, i]) < 0.6 * (c[i, i] - fine[i, i])
    # an insulating phase of a transport is such a pore, for that physics only
    mats = {0: Material(diffusivity=2.0, conductivity=1.0),
            1: Material(diffusivity=0.0, conductivity=5.0)}  # fmt: skip
    d, _ = conductivity_tensor(make_problem(sample, mats, "diffusion"))
    ref, _ = conductivity_tensor(make_problem(sample, {0: Material(conductivity=2.0), 1: VOID},
                                              "thermal"))  # fmt: skip
    assert np.array_equal(d, ref)
    # a dilute cubic array of insulating spheres: Maxwell's estimate (Hashin and
    # Shtrikman's upper bound), from above as the linear tetrahedra refine
    f = 1.0 - 4.0 / 3.0 * np.pi * 0.3**3
    maxwell = 2.0 * 2.0 * f / (3.0 - f)
    assert maxwell < d[0, 0] < 1.025 * maxwell  # 2.4 cells in the radius of the pore
    k, _ = conductivity_tensor(make_problem(sample, mats, "thermal"))
    assert k[0, 0] > 1.0  # both phases conduct heat: the problem is enriched
    # all void: nothing
    empty = XFEMElasticProblem(sample, {0: VOID, 1: VOID})
    assert empty.empty and not empty.solve(Loading.strain(np.eye(3))).stress.any()


def test_loading_paths_on_cut_elements():
    """Viscoelastic layers oblique to the grid, sheared in their plane: in parallel with
    the elastic ones, the mean of their relaxations (to round-off); a homogeneous sample
    is its material point."""
    visco = Material(young=3.0, poisson=0.3, viscoelasticity={"shear": [[0.5, 1.0]]})
    elastic = Material(young=30.0, poisson=0.2)
    g = 0.01 / np.sqrt(2.0)  # a shear of 0.01 between (1, -1, 0) and z
    strain = {c: [0.0, 0.0, 0.0] for c in ("11", "22", "33", "12")}
    strain.update({"31": [0, g, g], "23": [0, -g, -g]})
    path = {"times": [0, 1e-9, 4.0], "increments": [1, 8], "strain": strain}
    problem = NonlinearXFEMProblem(oblique(), {0: visco, 1: elastic})
    r = problem.run(path, fields=True)
    t = r.times[1:] - 1e-9
    _, relaxation = visco.viscoelasticity.relaxation(t, 2.5, 3.0 / 2.6)
    shear = (r.stress[1:, 4] - r.stress[1:, 3]) / np.sqrt(2.0)
    assert np.allclose(shear, (0.5 * relaxation + 0.5 * 12.5) * 0.01, rtol=1e-8)
    assert set(r.phases) == {0, 1}
    fields = r.fields[4.0]
    assert fields["stress"].shape == (problem.mesh.n_elements, 6)
    # a homogeneous plastic sample (no interface at all) under a uniaxial stress
    metal = Material(young=200.0, poisson=0.3, plasticity={"yield": 1.0, "hardening": 20.0})
    whole = LevelSetSample(np.ones((3, 3, 3)))
    tension = {"times": [0, 1.0], "increments": 6, "strain": {"11": [0, 0.02]}}
    ref = simulate_point(metal, tension)
    r = homogenize(whole, {0: metal, 1: elastic}, properties=[], paths={"t": tension})["t.periodic"]
    assert np.allclose(r.stress, ref.stress, rtol=1e-7, atol=1e-9)
    assert np.allclose(r.plastic, ref.plastic, rtol=1e-7, atol=1e-12)
    with pytest.raises(ValueError, match="kinematics"):
        NonlinearXFEMProblem(whole, {0: metal, 1: elastic}, kinematics="large")


def test_finite_strains_on_cut_elements():
    """Total Lagrangian paths: one rubber on both sides of interfaces oblique to the grid
    is its material point, in tension and in simple shear; two rubbers in layers
    stretched across them, their sides held: the stretches of the layers that give the
    same nominal stress across them (to the tolerance of the iterations)."""
    soft = Material(hyperelasticity={"model": "neo_hookean", "mu": 1.0, "bulk": 5.0})
    stiff = Material(hyperelasticity={"model": "mooney_rivlin", "c10": 2.0, "c01": 0.5,
                                      "bulk": 30.0})  # fmt: skip
    tension = {"times": [0, 1.0], "increments": 5, "strain": {"11": [0, 0.5]}}
    held = {c: [0.0, 0.0] for c in ("11", "22", "33", "23", "31", "12")}
    shear = {"times": [0, 1.0], "increments": 4, "strain": {**held, "12": [0, 0.6]}}
    same = NonlinearXFEMProblem(oblique(), {0: soft, 1: soft}, kinematics="finite")
    assert same.finite and same.enriched
    for path in (tension, shear):
        ref = simulate_point(soft, path, kinematics="finite")
        r = same.run(path)
        assert np.allclose(r.stress, ref.stress, rtol=1e-6, atol=1e-7)
        assert np.allclose(r.strain, ref.strain, rtol=1e-6, atol=1e-8)
        assert np.allclose(r.cauchy, ref.cauchy, rtol=1e-6, atol=1e-7)

    def level_set(points):  # layers normal to x, between the planes of the grid
        return np.abs(points[:, 0] % 1.0 - 0.5) - 0.2

    f1 = 0.4
    sample = LevelSetSample.of(
        ImplicitBody(mg.Domain.cube(1.0, periodic=True), level_set), (7, 2, 2)
    )
    assert sample.phase_fractions()[1] == pytest.approx(f1, rel=1e-12)
    stretch = {"times": [0, 1.0], "increments": 5, "strain": {**held, "11": [0, 0.5]}}
    r = NonlinearXFEMProblem(sample, {0: soft, 1: stiff}, kinematics="finite").run(
        stretch, fields=True
    )
    laws = {0: analytic_model(soft, finite=True), 1: analytic_model(stiff, finite=True)}

    def nominal(law, lam):  # P11 and P22 of a layer stretched by lam, its sides held
        e = np.array([[0.5 * (lam * lam - 1.0), 0, 0, 0, 0, 0.0]])
        s = law.step(e, e, np.zeros((1, 0)), 1.0)[0][0]
        return lam * s[0], s[1]

    for k in range(1, len(r.times)):
        mean = 1.0 + r.strain[k, 0]

        def gap(lam, mean=mean):  # the nominal stresses across the two layers
            return nominal(laws[1], lam)[0] - nominal(laws[0], (mean - f1 * lam) / (1 - f1))[0]

        lam1 = brentq(gap, 1.0, mean)
        lam0 = (mean - f1 * lam1) / (1 - f1)
        across = nominal(laws[1], lam1)[0]
        along = f1 * nominal(laws[1], lam1)[1] + (1 - f1) * nominal(laws[0], lam0)[1]
        assert r.stress[k, 0] == pytest.approx(across, rel=1e-6)
        assert r.stress[k, 1] == pytest.approx(along, rel=1e-6)
        assert r.stress[k, 2] == pytest.approx(along, rel=1e-6)
    # the fields: the logarithmic strains of the two layers
    strain = r.fields[1.0]["strain"]
    assert strain[:, 0].max() == pytest.approx(np.log(lam0), rel=1e-6)
    assert strain[:, 0].min() == pytest.approx(np.log(lam1), rel=1e-6)


def test_a_damaging_inclusion_softens_the_cell():
    """A stiff brittle sphere in an elastic matrix, pulled: its damage (the variable p of
    the phase 1) grows and the cell softens; the path converges on the cut elements, the
    damage delayed by its viscosity (without it the sphere softens at once and the cell
    snaps back, which a prescribed strain cannot follow)."""
    brittle = Material(young=20.0, poisson=0.2, damage={
        "threshold": 2e-3, "ultimate": 4e-2, "softening": "linear", "equivalent": "energy",
        "viscosity": 0.05})  # fmt: skip
    matrix = Material(young=1.0, poisson=0.3)
    tension = {"times": [0, 1.0], "increments": 8, "strain": {"11": [0, 0.02]}}
    r = NonlinearXFEMProblem(sphere(0.3, shape=6), {0: matrix, 1: brittle}).run(tension)
    damage = r.phases[1]["plastic"]
    assert damage[0] == 0.0 and damage[-1] > 0.05 and np.all(np.diff(damage) >= -1e-12)
    secant = r.stress[1:, 0] / r.strain[1:, 0]
    assert secant[-1] < secant[0] * (1 - 1e-3)


def test_samples_and_problems_are_checked():
    model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
    micro = model.generate(mg.Domain.cube(6.0, periodic=True), seed=1)
    sample = LevelSetSample.of(micro, shape=10)
    assert sample.shape == (10, 10, 10) and np.allclose(sample.spacing, 0.6)
    assert LevelSetSample.of(micro, spacing=0.75).shape == (8, 8, 8)
    fractions = sample.phase_fractions()
    assert fractions[1] == pytest.approx(micro.voxelize(shape=40).data.mean(), rel=0.1)
    with pytest.raises(ValueError, match="periodic"):
        LevelSetSample.of(model.generate(mg.Domain.cube(6.0, periodic=False), seed=1), shape=8)
    with pytest.raises(ValueError, match="shape of the grid or its spacing"):
        LevelSetSample.of(micro)
    mosaic = mg.models.Voronoi(n_cells=8).generate(mg.Domain.cube(1.0, periodic=True), seed=1)
    with pytest.raises(ValueError, match="no level-set"):
        LevelSetSample.of(mosaic, shape=6)
    with pytest.raises(ValueError, match="finite"):
        LevelSetSample(np.full((2, 2, 2), np.nan))
    with pytest.raises(ValueError, match="periodic"):
        XFEMElasticProblem(sample, MATS, "kubc")
    with pytest.raises(ValueError, match="periodic"):
        homogenize(sample, MATS, bcs=["kubc"])
    with pytest.raises(ValueError, match="no material"):
        XFEMElasticProblem(sample, {1: STIFF})
    with pytest.raises(ValueError, match="snap"):
        XFEMElasticProblem(sample, MATS, snap=0.6)
    with pytest.raises(ValueError, match="physics"):
        XFEMThermalProblem(sample, MATS, physics="elastic")
    with pytest.raises(ValueError, match="no parameters"):
        make_problem(sample, MATS, "magnetic")
    with pytest.raises(ValueError, match="CPU"):
        make_problem(sample, MATS, "elastic", device="gpu")
    with pytest.raises(TypeError, match="LevelSetSample"):
        XFEMElasticProblem(micro, MATS)


def test_xfem_in_a_configuration(tmp_path):
    from morphogen import pipeline
    from morphogen.config import ConfigError, load_config

    data = {
        "model": {"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2},
        "domain": {"size": 6.0, "periodic": True},
        "seed": 5,
        "realizations": 2,
        "voxels": {"shape": 10},
        "homogenization": {
            "method": "xfem",
            "materials": {0: {"young": 1.0, "poisson": 0.3, "conductivity": 1.0},
                          1: {"young": 10.0, "poisson": 0.2, "conductivity": 10.0}},
            "properties": ["bulk_modulus", "conductivity", "stiffness"],
            "fields": True,
        },
        "outputs": {"directory": str(tmp_path / "out"), "name": "spheres"},
    }  # fmt: skip
    outdir = pipeline.run(load_config(data))
    rows = pipeline.read_rows(outdir / "results.csv")
    assert len(rows) == 2
    for row in rows:
        assert 1.0 < row["conductivity.periodic"] < 1.0 + 9.0 * row["fraction_1"]
        assert row["stiffness.periodic.bulk_modulus"] == pytest.approx(
            row["bulk_modulus.periodic"], rel=1e-5
        )
        assert 0.1 < row["fraction_1"] < 0.25
    long = pipeline.read_rows(outdir / "homogenization.csv")
    assert {r["solver"] for r in long} <= {"xfem-direct", "xfem-amg"}
    assert (outdir / "spheres_0001_bulk_modulus_periodic.vtu").exists()
    for change, match in (
        ({"boundary_conditions": ["kubc"]}, "periodic"),
        ({"device": "gpu"}, "GPU"),
        ({"properties": ["piezoelectric"]}, "voxel images"),
    ):  # fmt: skip
        bad = {**data, "homogenization": {**data["homogenization"], **change}}
        with pytest.raises(ConfigError, match=match):
            load_config(bad)
    with pytest.raises(ConfigError, match="voxels"):
        load_config({k: v for k, v in data.items() if k != "voxels"})
    with pytest.raises(ConfigError, match="level set"):
        load_config({**data, "model": {"type": "voronoi", "n_cells": 10}})
    with pytest.raises(ConfigError, match="periodic"):
        load_config({**data, "domain": {"size": 6.0, "periodic": False}})
