"""Inverse design of active structures (morphogen.optimization.morphing): the adjoint
sensitivities against finite differences, at equilibrium and through the steps of a
diffusion; the states against the coupled solver; strips designed to bend to a target
curvature, and to bend one way and then the other as two materials take up a species at
different rates."""

from __future__ import annotations

import numpy as np
import pytest

from morphogen.coupling import ThermoChemoElastic, ThermoChemoMaterial, timoshenko_curvature
from morphogen.image import VoxelImage
from morphogen.optimization import MorphingOptimization, MorphingTarget, hat_filter

M0 = ThermoChemoMaterial(
    young=1.0, poisson=0.3, conductivity=1.0, heat_capacity=1.0, diffusivity=1.0, solubility=1.0
)
M1 = ThermoChemoMaterial(
    young=2.0, poisson=0.25, expansion=1e-2, conductivity=0.2, heat_capacity=3.0,
    diffusivity=0.3, solubility=2.0, swelling=0.02,
)  # fmt: skip
TIP = [[6.0, 1.0, 1.5], [6.0, 0.0, 3.0]]


def test_hat_filter():
    """Rows of weights ``r - d`` that sum to one, cut at the faces of the box; the
    identity below a voxel."""
    f = hat_filter((5, 4, 3), 1.5)
    np.testing.assert_allclose(np.asarray(f.sum(axis=1)).ravel(), 1.0)
    w = f.toarray().reshape(5, 4, 3, 5, 4, 3)[2, 2, 1]
    assert w[2, 2, 1] == pytest.approx(3.0 * w[3, 2, 1])  # weights r - d: 1.5 and 0.5
    assert w[4, 2, 1] == 0.0
    assert (hat_filter((3, 3, 3), 0.5) != hat_filter((3, 3, 3), 0.9)).nnz == 0
    planar = hat_filter((6, 1, 4), 2.0).toarray().reshape(6, 4, 6, 4)
    np.testing.assert_allclose(planar.sum(axis=(2, 3)), 1.0)


def _gradient_error(problem, n, rng, beta=2.0):
    x = rng.uniform(0.2, 0.9, n)
    _, g = problem.evaluate(x, beta)
    worst = 0.0
    for k in rng.choice(n, 6, replace=False):
        step = np.zeros(n)
        step[k] = 1e-6
        fd = (problem.evaluate(x + step, beta, gradient=False)[0]
              - problem.evaluate(x - step, beta, gradient=False)[0]) / 2e-6  # fmt: skip
        worst = max(worst, abs(fd - g[k]) / np.abs(g).max())
    return worst


def test_adjoint_sensitivities_against_finite_differences():
    """The derivatives of the objective by the adjoints against central differences: at
    equilibrium (the thermal and the chemical stimuli, a design extruded along y and a
    penalized eigenstrain, voids), and in time (a face held at a value and another at a
    function of time; an output at the start)."""
    rng = np.random.default_rng(1)
    shape = (6, 2, 3)
    tip = [MorphingTarget(TIP, 2, [-0.05, 0.02])]
    cases = [
        (MorphingOptimization(shape, 1.0, (M0, M1), tip, filter_radius=1.5), 36),
        (MorphingOptimization(shape, 1.0, (M0, M1), tip, stimulus="activity", change=0.7,
                              filter_radius=1.2, extrude=1, eigen_penalty=2.0), 18),
        (MorphingOptimization(shape, 1.0, (M0, M1), tip, void=True, volume_fraction=0.4,
                              solid_fraction=0.7, filter_radius=1.5), 72),
    ]  # fmt: skip
    timed = [MorphingTarget(TIP, 2, [-0.05, 0.02], time=0),
             MorphingTarget(TIP, 2, [0.03, 0.01], time=2, weight=2.0)]  # fmt: skip
    cases += [
        (MorphingOptimization(shape, 1.0, (M0, M1), timed, times=[0.2, 0.5, 2.0],
                              faces={"z+": 1.0, "x+": lambda t: 0.5 * t}, steps=4,
                              filter_radius=1.5), 36),
        (MorphingOptimization(shape, 1.0, (M0, M1), timed, times=[0.0, 0.5, 2.0],
                              stimulus="activity", faces={"z-": 1.0}, steps=5,
                              filter_radius=1.0, initial=0.1), 36),
    ]  # fmt: skip
    for problem, n in cases:
        assert _gradient_error(problem, n, rng) < 1e-6


def test_states_are_those_of_the_coupled_solver():
    """The displacements of a layout of the two materials (densities 0 and 1) are those
    that :class:`~morphogen.coupling.ThermoChemoElastic` computes for its phases, at
    equilibrium and while the species diffuses."""
    rng = np.random.default_rng(3)
    shape = (5, 2, 4)
    layout = (rng.random(shape) < 0.5).astype(np.uint8)
    image = VoxelImage(layout, spacing=(1.0, 1.0, 1.0))
    target = [MorphingTarget([[5.0, 1.0, 2.0]], 2, 1.0)]
    steady = MorphingOptimization(shape, 1.0, (M0, M1), target, change=2.0)
    _, _, u = steady.objective(layout.ravel().astype(float), gradient=False)
    ref = ThermoChemoElastic(
        image,
        {0: M0, 1: M1},
        supports={"x-": "clamped"},
        reference_temperature=300.0,
        initial_temperature=302.0,
    ).run([0.0])
    np.testing.assert_allclose(u[0], ref.displacement[0].reshape(-1, 3), atol=1e-12)
    timed = MorphingOptimization(shape, 1.0, (M0, M1), target, stimulus="activity",
                                 times=[0.3, 1.0], faces={"z+": 1.0}, steps=6)  # fmt: skip
    _, _, u = timed.objective(layout.ravel().astype(float), gradient=False)
    ref = ThermoChemoElastic(image, {0: M0, 1: M1}, supports={"x-": "clamped"},
                             activity={"z+": 1.0}).run([0.3, 1.0], steps=6)  # fmt: skip
    for k in range(2):
        np.testing.assert_allclose(u[k], ref.displacement[k].reshape(-1, 3), atol=1e-12)


def test_strip_designed_to_bend():
    """A strip clamped at one end, an active material that expands in a passive one: the
    layout found from uniform densities for the deflections of the strip whose upper sixth
    is active (55 % of the curvature of the bilayer of equal halves, Timoshenko's 0.556)
    puts the active material in its upper part and reaches them within 5 %, as its
    black-and-white layout solved by the coupled solver (the inverse problem has many
    solutions on six voxels through the thickness: thinner layers near the clamp, columns
    near the free end)."""
    passive = ThermoChemoMaterial(young=1.0, poisson=0.3)
    active = ThermoChemoMaterial(young=1.0, poisson=0.3, expansion=1e-3)
    shape = (24, 1, 6)
    probe = MorphingOptimization(shape, 1.0, (passive, active), MorphingTarget(TIP, 2, 1.0))
    line = probe.bending_target(1.0)
    layout = np.zeros(shape)
    layout[:, :, 5] = 1.0
    nodes = np.array(line.points)
    index = ((nodes[:, 0] * 2 + nodes[:, 1]) * 7 + nodes[:, 2]).astype(int)
    wanted = probe.objective(layout.ravel(), gradient=False)[2][0][index, 2]
    ratio = timoshenko_curvature(1e-3, 5.0, 1.0, 1.0, 1.0) / timoshenko_curvature(
        1e-3, 3.0, 3.0, 1.0, 1.0
    )
    assert ratio == pytest.approx(0.556, abs=1e-3)
    target = MorphingTarget(nodes, 2, wanted)
    problem = MorphingOptimization(shape, 1.0, (passive, active), target, extrude=1,
                                   volume_fraction=0.5, filter_radius=1.5, iterations=150,
                                   continuation=20)  # fmt: skip
    design = problem.run()
    assert design.history["objective"][-1] < 0.01 * design.history["objective"][0]
    assert np.abs(design.reached[0] - wanted).max() < 0.05 * np.abs(wanted).max()
    density = design.density[:, 0, :]
    assert density[:, 3:].mean() > 2.0 * density[:, :3].mean()  # the active part above
    assert density.mean() <= 0.5 + 1e-3
    result, reached = problem.verify(design)
    assert np.abs(reached[0] - wanted).max() < 0.05 * np.abs(wanted).max()
    assert result.curvature(0, span=(4, 20)) > 0


@pytest.mark.slow
def test_strip_designed_to_bend_one_way_then_the_other():
    """Two materials that swell with a species, one that it enters fast and that swells
    little, one slow that swells more, the species entering from both faces of the strip:
    the layout found from uniform densities makes the tip go down at first and up at
    equilibrium, the targets of the strip whose upper half is the fast material, within 3 %
    for the densities and 10 % for their black-and-white layout."""
    slow = ThermoChemoMaterial(young=1.0, poisson=0.3, diffusivity=0.02, swelling=1.5e-3)
    fast = ThermoChemoMaterial(young=1.0, poisson=0.3, diffusivity=1.0, swelling=0.5e-3)
    shape = (24, 1, 8)
    tip = [[24.0, 0.0, 4.0]]
    common = {"stimulus": "activity", "times": [2.0, 100.0], "faces": {"z-": 1.0, "z+": 1.0},
              "steps": 8}  # fmt: skip
    probe = MorphingOptimization(shape, 1.0, (slow, fast), MorphingTarget(tip, 2, 1.0), **common)
    layout = np.zeros(shape)
    layout[:, :, 4:] = 1.0
    _, _, u = probe.objective(layout.ravel(), gradient=False)
    early, late = u[0][24 * 2 * 9 + 4, 2], u[1][24 * 2 * 9 + 4, 2]
    assert early < 0 < late
    targets = [MorphingTarget(tip, 2, early, time=0), MorphingTarget(tip, 2, late, time=1)]
    problem = MorphingOptimization(
        shape,
        1.0,
        (slow, fast),
        targets,
        extrude=1,
        volume_fraction=0.5,
        filter_radius=3.0,
        move=0.1,
        **common,
    )
    design = problem.run()
    for reached, wanted in zip(design.reached, (early, late), strict=True):
        assert reached[0] == pytest.approx(wanted, rel=0.03)
    _, reached = problem.verify(design)
    assert reached[0][0] == pytest.approx(early, rel=0.1)
    assert reached[1][0] == pytest.approx(late, rel=0.1)


def test_errors():
    target = MorphingTarget(TIP, 2, 0.1)
    with pytest.raises(ValueError, match="faces"):
        MorphingOptimization((4, 1, 2), 1.0, (M0, M1), target, times=[1.0])
    with pytest.raises(ValueError, match="void"):
        MorphingOptimization((4, 1, 2), 1.0, (M0, M1), target, times=[1.0],
                             faces={"z+": 1.0}, void=True, solid_fraction=0.6)  # fmt: skip
    with pytest.raises(ValueError, match="solid_fraction"):
        MorphingOptimization((4, 1, 2), 1.0, (M0, M1), target, void=True)
    with pytest.raises(ValueError, match="stimulus"):
        MorphingOptimization((4, 1, 2), 1.0, (M0, M1), target, stimulus="light")
    with pytest.raises(ValueError, match="time"):
        MorphingOptimization((4, 1, 2), 1.0, (M0, M1), MorphingTarget(TIP, 2, 0.1, time=3))
    with pytest.raises(ValueError, match="not all zero"):
        MorphingOptimization((4, 1, 2), 1.0, (M0, M1), MorphingTarget(TIP, 2, 0.0))
    with pytest.raises(ValueError, match="component"):
        MorphingTarget(TIP, 3, 0.1)
