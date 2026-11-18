"""Topology optimization of periodic cells (inverse homogenization with the FFT solver)."""

import json

import numpy as np
import pytest
import yaml
from scipy.optimize import minimize

import morphogen as mg
from helpers import backus
from morphogen import cli
from morphogen.config import ConfigError
from morphogen.homogenization import Material
from morphogen.homogenization.elements import isotropic_stiffness
from morphogen.homogenization.materials import lame
from morphogen.optimization import (
    MMA,
    CellDesign,
    CellOptimization,
    DensityFilter,
    PeriodicCell,
    Symmetry,
    count_hinges,
    initial_density,
    load_optimization_config,
    tensor_properties,
)
from morphogen.optimization.filters import heaviside, heaviside_derivative

STEEL = Material(young=1.0, poisson=0.3, conductivity=1.0)


def _hs_bulk_2d(f, material=STEEL):
    """Hashin-Shtrikman upper bound of the plane-strain bulk modulus, solid and voids."""
    lam, mu = material.lame
    k = lam + mu
    return k * mu * f / ((1 - f) * k + mu)


# -- graded cells --------------------------------------------------------------------------


def test_cell_of_a_laminate():
    n = 8
    s = np.ones((3, 2, n))
    s[:, :, n // 2 :] = 0.1
    cell = PeriodicCell(s.shape, material=STEEL, tol=1e-12)
    c, _ = cell.solve(s)
    lam, mu = lame(1.0, 0.3)
    d = isotropic_stiffness(lam, mu)
    np.testing.assert_allclose(c, backus([d, 0.1 * d], [0.5, 0.5]), atol=1e-10)
    th = PeriodicCell(s.shape, physics="thermal", material=STEEL, tol=1e-12)
    k, _ = th.solve(s)
    np.testing.assert_allclose(np.diag(k), [0.55, 0.55, 1.0 / (0.5 / 1.0 + 0.5 / 0.1)])


@pytest.mark.parametrize("physics", ["elastic", "thermal"])
def test_sensitivities_match_finite_differences(physics):
    rng = np.random.default_rng(0)
    shape = (5, 4, 3)
    s = 0.05 + rng.random(shape).ravel()
    cell = PeriodicCell(shape, physics=physics, material=STEEL, tol=1e-13)
    c, energies = cell.solve(s)
    pairs = [(0, 0), (0, 1), (3, 3), (2, 5)] if physics == "elastic" else [(0, 0), (1, 2)]
    for i, j in pairs:
        d = cell.derivative(energies, i, j)
        for e in (0, 17, 41):
            step = np.zeros_like(s)
            step[e] = 1e-6
            hi, _ = PeriodicCell(shape, physics=physics, material=STEEL, tol=1e-13).solve(s + step)
            lo, _ = PeriodicCell(shape, physics=physics, material=STEEL, tol=1e-13).solve(s - step)
            assert d[e] == pytest.approx((hi[i, j] - lo[i, j]) / 2e-6, rel=1e-6, abs=1e-12)
    # the tensor is the scale-weighted mean of the energies
    assert c[0, 0] == pytest.approx(s @ cell.derivative(energies, 0, 0))


def test_cell_options():
    cell = PeriodicCell((4, 4, 1), material=STEEL, cases=[0, 1])
    c, _ = cell.solve(np.ones(16))
    assert np.isnan(c[5, 5]) and c[0, 0] == pytest.approx(STEEL.lame[0] + 2 * STEEL.lame[1])
    with pytest.raises(ValueError, match="scale"):
        cell.solve(np.ones(3))
    with pytest.raises(ValueError, match="conductivity"):
        PeriodicCell((2, 2, 2), physics="thermal", material=Material(young=1.0, poisson=0.3))


# -- filters, projection, symmetry ---------------------------------------------------------


def test_filters_and_symmetry():
    rng = np.random.default_rng(1)
    for shape in ((12, 10, 8), (16, 16, 1)):
        f = DensityFilter(shape, 2.5)
        a, b = rng.random(shape), rng.random(shape)
        assert f(a).mean() == pytest.approx(a.mean())
        assert np.sum(f(a) * b) == pytest.approx(np.sum(a * f.transpose(b)))
        assert np.ptp(f(a)) < np.ptp(a)
    assert np.array_equal(
        DensityFilter((4, 4, 4), 0.5)(np.eye(4)[:, :, None] * np.ones(4)),
        np.eye(4)[:, :, None] * np.ones(4),
    )
    x = np.linspace(0, 1, 11)
    assert heaviside(0.0, 8.0) == pytest.approx(0.0) and heaviside(1.0, 8.0) == pytest.approx(1.0)
    fd = (heaviside(x + 1e-7, 8.0) - heaviside(x - 1e-7, 8.0)) / 2e-7
    np.testing.assert_allclose(heaviside_derivative(x, 8.0), fd, rtol=1e-6)
    for kind, shape in (("orthotropic", (6, 5, 4)), ("cubic", (6, 6, 6)), ("cubic", (8, 8, 1))):
        sym = Symmetry(shape, kind)
        a, b = rng.random(shape), rng.random(shape)
        pa = sym(a)
        np.testing.assert_allclose(sym(pa), pa)  # idempotent
        assert np.sum(pa * b) == pytest.approx(np.sum(a * sym(b)))  # self-adjoint
        np.testing.assert_allclose(pa, np.flip(pa, axis=0))
        if kind == "cubic":
            np.testing.assert_allclose(pa, np.swapaxes(pa, 0, 1))
    with pytest.raises(ValueError, match="cubic"):
        Symmetry((6, 4, 4), "cubic")


def test_mma_on_svanberg_toy_problem():
    centres = np.array([[5.0, 2.0, 1.0], [3.0, 4.0, 3.0]])

    def constraints(x):
        return ((x - centres) ** 2).sum(axis=1) - 9.0, 2.0 * (x - centres)

    x = np.array([4.0, 3.0, 2.0])
    opt = MMA(3, 2, 0.0, 5.0)
    for it in range(40):
        g, dg = constraints(x)
        x = opt.update(x, x @ x, 2 * x, g, dg)
        if it == 5:
            opt.reset()  # fresh asymptotes, as after a continuation step
    ref = minimize(
        lambda y: y @ y,
        [4, 3, 2],
        jac=lambda y: 2 * y,
        method="SLSQP",
        bounds=[(0, 5)] * 3,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda y: -constraints(y)[0],
                "jac": lambda y: -constraints(y)[1],
            }
        ],
    )
    np.testing.assert_allclose(x, ref.x, atol=1e-5)
    np.testing.assert_allclose(x, [2.0175, 1.7800, 1.2375], atol=1e-4)  # Svanberg 2007
    with pytest.raises(ValueError, match="constraints"):
        MMA(2, 2, 0.0, 1.0)


# -- optimizations -------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mma", "beso"])
def test_bulk_modulus_designs_reach_hashin_shtrikman(method):
    opt = CellOptimization(
        (32, 32, 1),
        0.5,
        method=method,
        filter_radius=1.5,
        projection=32.0 if method == "mma" else 0.0,
        iterations=300,
    )
    design = opt.run(np.random.default_rng(0))
    assert design.converged
    f = design.binary_fraction
    assert abs(f - 0.5) < 0.02 and abs(design.volume_fraction - 0.5) < 0.005
    kappa = design.properties()["bulk_modulus"]
    assert 0.9 * _hs_bulk_2d(f) < kappa <= _hs_bulk_2d(f)
    # black and white, without hinges: the moduli hold on a finer grid
    assert design.hinges == 0
    assert 0.85 * _hs_bulk_2d(f) < design.refined_properties(2)["bulk_modulus"] < 1.02 * kappa
    hist = design.history
    if method == "mma":  # BESO removes material as it goes
        assert hist["objective"][-1] < hist["objective"][0]
    # square symmetry of the design
    np.testing.assert_allclose(design.density, np.swapaxes(design.density, 0, 1), atol=1e-9)


def test_conductivity_by_hybrid_cellular_automaton_and_oc():
    hs = 0.4 / (2 - 0.4)  # 2D Hashin-Shtrikman upper bound, conductor and insulator
    for method in ("hca", "oc"):
        opt = CellOptimization(
            (32, 32, 1),
            0.4,
            objective="conductivity",
            method=method,
            iterations=100,
            filter_radius=1.5,
        )
        design = opt.run(np.random.default_rng(0))
        k = design.properties()["conductivity"]
        assert 0.92 * hs < k <= hs * (1 + 1e-6)
        assert abs(design.volume_fraction - 0.4) < 0.01


def test_auxetic_design():
    opt = CellOptimization(
        (32, 32, 1),
        0.35,
        objective="poisson_ratio",
        filter_radius=1.5,
        projection=8.0,
        iterations=150,
    )
    design = opt.run(np.random.default_rng(0))
    props = design.properties()
    assert props["poisson_12"] < -0.3
    assert design.binary_tensor[0, 0] > 0.5 * opt.min_stiffness
    with pytest.raises(ValueError, match="mma"):
        CellOptimization((8, 8, 1), 0.3, objective="poisson_ratio", method="oc")


def test_target_tensor():
    target = np.full((6, 6), np.nan)
    target[np.ix_([0, 1], [0, 1])] = [[0.30, 0.05], [0.05, 0.30]]  # an anisotropic goal
    opt = CellOptimization(
        (24, 24, 1),
        0.5,
        objective="target",
        target=target,
        symmetry="orthotropic",
        filter_radius=1.5,
        iterations=60,
    )
    design = opt.run(np.random.default_rng(0))
    assert design.history["objective"][-1] < 0.2 * design.history["objective"][0]
    assert opt.cases == [0, 1]
    with pytest.raises(ValueError, match="target"):
        CellOptimization((8, 8, 1), 0.3, objective="target")


def _fixed_design(density):
    """A black-and-white design given voxel by voxel."""
    c, _ = PeriodicCell(density.shape, material=STEEL, tol=1e-10).solve(density)
    return CellDesign(
        density=density,
        design=density,
        tensor=c,
        binary_tensor=c,
        objective=0.0,
        volume_fraction=float(density.mean()),
        binary_fraction=float(density.mean()),
        iterations=0,
        converged=True,
        settings={"material": STEEL.to_config(), "voxel_size": 1.0},
    )


def test_optimization_from_a_microstructure():
    """An optimization starts from the densities of a microstructure: the fraction of its
    solid phase in each voxel (its middle section for a planar cell), which the symmetry
    then averages; densities of another shape or out of [0, 1] are refused."""
    micro = mg.model_from_config(
        {"type": "tpms", "surface": "gyroid", "cell_size": 1.0, "volume_fraction": 0.3}
    ).generate(mg.Domain(size=1.0, periodic=True), 0)
    cube = initial_density(micro, (16, 16, 16))
    assert cube.shape == (16, 16, 16) and cube.mean() == pytest.approx(0.3, abs=0.02)
    np.testing.assert_allclose(initial_density(micro, (16, 16, 16), phase=0), 1.0 - cube)
    plane = initial_density(micro, (20, 20, 1))
    assert plane.shape == (20, 20, 1) and 0.0 <= plane.min() and plane.max() <= 1.0
    assert np.any((plane > 0.0) & (plane < 1.0))  # partial volumes along the interfaces
    run = CellOptimization((20, 20, 1), 0.3, initial=plane, symmetry="none", iterations=1)
    assert np.array_equal(run.initial_design(np.random.default_rng(0)), plane)
    assert run.settings()["initial"] == "given"
    design = CellOptimization(
        (20, 20, 1), 0.3, initial=plane, method="oc", iterations=8, symmetry="none"
    ).run(np.random.default_rng(0))
    assert design.volume_fraction == pytest.approx(0.3, abs=0.01)
    with pytest.raises(ValueError, match="shape of the cell"):
        CellOptimization((20, 20, 1), 0.3, initial=np.ones((8, 8, 1)))
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        CellOptimization((20, 20, 1), 0.3, initial=2.0 * plane)


def test_hinges_and_refined_evaluation():
    """Voxels touching only at a corner are stiff hinges for trilinear elements: they are
    counted, and the moduli of a hinged design drop on a finer grid."""
    diagonal = np.zeros((4, 4, 1))
    diagonal[0, 0] = diagonal[1, 1] = 1
    assert count_hinges(diagonal) == 1
    diagonal[1, 0] = 1
    assert count_hinges(diagonal) == 0
    vertex = np.zeros((4, 4, 4))
    vertex[0, 0, 0] = vertex[1, 1, 1] = 1
    assert count_hinges(vertex) == 1
    edge = np.zeros((4, 4, 4))
    edge[0, 0, 0] = edge[1, 1, 0] = 1
    assert count_hinges(edge) == 2  # the edge lies in two blocks
    assert count_hinges(np.ones((3, 3, 3))) == 0
    # a square hole: the same geometry on a finer grid, slightly softer
    hole = np.ones((8, 8, 1))
    hole[2:6, 2:6] = 0.0
    square = _fixed_design(hole)
    k = square.properties()["bulk_modulus"]
    assert square.hinges == 0
    assert square.refined_properties(1, tol=1e-10)["bulk_modulus"] == pytest.approx(k)
    assert 0.95 * k < square.refined_properties(2, tol=1e-10)["bulk_modulus"] < k
    # a checkerboard of 2 x 2 squares touching at their corners
    checker = np.kron(np.indices((4, 4)).sum(axis=0) % 2, np.ones((2, 2)))[:, :, None]
    hinged = _fixed_design(checker.astype(float))
    assert hinged.hinges == 16
    assert hinged.refined_properties(2)["bulk_modulus"] < 0.8 * hinged.properties()["bulk_modulus"]


def test_design_outputs_and_command_line(tmp_path, capsys):
    config = {
        "optimization": {
            "objective": "bulk_modulus",
            "volume_fraction": 0.4,
            "grid": [20, 20, 1],
            "cell_size": 2.0,
            "method": "beso",
            "iterations": 60,
        },
        "seed": 3,
        "outputs": {
            "directory": str(tmp_path / "opt"),
            "name": "cell",
            "image": ["vti"],
            "density": ["npz"],
            "geometry": ["json"],
        },
    }
    path = tmp_path / "opt.yaml"
    path.write_text(yaml.safe_dump(config))
    assert cli.main(["optimize", str(path)]) == 0
    text = capsys.readouterr().out
    assert "bulk_modulus" in text and "thresholded design: stiffness C (Voigt" in text
    out = tmp_path / "opt"
    data = json.loads((out / "optimization.json").read_text(encoding="utf-8"))
    assert data["settings"]["method"] == "beso" and data["iterations"] > 1
    assert data["properties"]["bulk_modulus"] > 0
    c, s = np.array(data["binary_tensor"]), np.array(data["binary_compliance"])
    assert np.allclose(s @ c, np.eye(6), atol=1e-6) and "engineering shears" in data["notation"]
    assert data["compliance"] is None  # the densities of a planar cell: its in-plane entries
    assert data["hinges"] >= 0 and data["check"]["refinement"] == 2
    fine = data["check"]["properties"]["bulk_modulus"]
    assert fine == pytest.approx(data["properties"]["bulk_modulus"], rel=0.25)
    micro = mg.io.load_microstructure(out / "cell.json")
    assert micro.shape == (20, 20, 1) and micro.domain.lengths[0] == pytest.approx(2.0)
    image = mg.io.load_image(out / "cell.vti")
    assert np.array_equal(image.data, micro.phases)
    assert np.array_equal(micro.level_set().data < 0, micro.phases == 1)
    density = mg.io.load_image(out / "cell_density.npz")
    assert density.data.dtype == np.float32
    assert (out / "history.csv").exists()
    bad = dict(config, optimization={**config["optimization"], "objective": "toughness"})
    with pytest.raises(ConfigError, match="objective"):
        load_optimization_config(bad)
    with pytest.raises(ConfigError, match="grid"):
        load_optimization_config({"optimization": {"volume_fraction": 0.3, "grid": [4, 4]}})
    with pytest.raises(ConfigError, match="check_refinement"):
        load_optimization_config(dict(config, outputs={"check_refinement": 1.5}))
    unchecked = load_optimization_config(dict(config, outputs={"check_refinement": 0}))
    assert unchecked.check_refinement == 0
    assert cli.main(["run", str(path)]) == 2
    assert "morphogen optimize" in capsys.readouterr().err


def test_tensor_properties():
    lam, mu = 0.3, 0.35
    c = isotropic_stiffness(lam, mu)
    p = tensor_properties(c)
    assert p["poisson_12"] == pytest.approx(lam / (2 * (lam + mu)))
    assert p["young_1"] == pytest.approx(mu * (3 * lam + 2 * mu) / (lam + mu))
    assert p["bulk_modulus"] == pytest.approx(lam + 2 * mu / 3)
    planar = tensor_properties(c, planar=True)
    assert planar["poisson_12"] == pytest.approx(lam / (lam + 2 * mu))  # plane strain
    assert tensor_properties(np.eye(3) * 2.0)["conductivity"] == 2.0
