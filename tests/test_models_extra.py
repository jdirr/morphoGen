"""Random laminates, dead leaves, Kelvin cells and Laguerre tessellations of packings."""

import itertools
import json
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.models.base import Microstructure
from morphogen.models.lattice import lattice_struts

D = mg.distributions
CUBE = mg.Domain.cube(1.0, periodic=True)


def test_random_laminate():
    base = {
        "thickness": D.LogNormal(0.03, 0.01),
        "spacing": D.LogNormal(0.05, 0.02),
        "hole_fraction": 0.25,
        "hole_radius": D.Uniform(0.02, 0.05),
    }
    t = mg.models.RandomLaminate(**base).theory()
    assert t["volume_fraction"] == pytest.approx(0.375 * 0.75)
    assert t["layer_period"] == pytest.approx(0.08) and math.isinf(t["integral_range"])
    for periodic, axis in ((True, 2), (False, 0)):
        model = mg.models.RandomLaminate(**base, axis=axis)
        micro = model.generate(mg.Domain.cube(1.0, periodic=periodic), seed=3)
        img = micro.voxelize((48, 48, 48))
        assert np.array_equal(micro.level_set(grid=img.grid()).data < 0, img.data == 1)
        inside = micro.contains(img.grid().center_points()).reshape(img.shape)
        assert np.array_equal(inside, img.data == 1)
        # along the axis, phase 1 only in the layers; in each layer, holes on a fraction q
        profile = np.moveaxis(img.data, axis, -1).reshape(-1, 48).max(axis=0)
        z = (np.arange(48) + 0.5) / 48
        k = np.searchsorted(micro.starts, z, side="right") - 1
        in_layer = (k >= 0) & (z < micro.ends[np.maximum(k, 0)])
        assert np.array_equal(profile == 1, in_layer)
        holes = 1.0 - np.moveaxis(img.data, axis, -1)[..., in_layer].mean(axis=(0, 1))
        assert holes.mean() == pytest.approx(0.25, abs=0.06)
        back = Microstructure.from_dict(json.loads(json.dumps(micro.to_dict())))
        assert np.array_equal(back.voxelize((48, 48, 48)).data, img.data)
    # periodic stacks: whole periods scaled to the box, continued across the faces
    plain = mg.models.RandomLaminate(thickness=D.LogNormal(0.03, 0.01), spacing=0.05)
    micro = plain.generate(CUBE, seed=1)
    assert micro.metadata["thickness_scale"] == pytest.approx(1.0, abs=0.1)
    starts, ends, ids = micro.starts, micro.ends, micro.layer_ids
    if ends[-1] == 1.0:  # a layer cut by the faces: the same layer on both sides
        assert starts[0] == 0.0 and ids[0] == ids[-1] and micro.n_layers == len(ids) - 1
    # variance of the fraction along a column: p (1 - p) A / L for long columns
    renewal = mg.models.RandomLaminate(thickness=D.LogNormal(0.02, 0.015), spacing=0.03)
    t = renewal.theory()
    assert t["integral_range_1d"] == pytest.approx(0.03**2 * 0.015**2 / (0.05 * 0.02 * 0.03))
    column = mg.Domain(size=(0.05, 0.05, 1.0))
    fractions = []
    for seed in range(200):
        m = renewal.generate(column, seed=seed)
        fractions.append(np.sum(m.ends - m.starts))
    p = t["volume_fraction"]
    assert np.mean(fractions) == pytest.approx(p, abs=0.01)
    assert np.var(fractions, ddof=1) == pytest.approx(p * (1 - p) * t["integral_range_1d"], rel=0.3)
    with pytest.raises(ValueError, match="hole_radius"):
        mg.models.RandomLaminate(thickness=0.1, spacing=0.1, hole_fraction=0.2)


def test_dead_leaves():
    model = mg.models.DeadLeavesModel(radius=0.08, phase_fractions=(0.6, 0.4))
    assert model.same_leaf_probability(0.0) == pytest.approx(1.0)
    assert model.same_leaf_probability(0.16) == pytest.approx(0.0)
    t = model.theory()
    assert t["volume_fraction"] == 0.4 and t["integral_range"] > 0
    measured, n = np.zeros(9), 96
    for seed in range(2):
        micro = model.generate(CUBE, seed=seed)
        lab = micro.labels((n, n, n)).data
        measured += [np.mean(lab == np.roll(lab, -k, axis=0)) for k in range(9)]
    expected = model.same_leaf_probability(np.arange(9) / n)
    np.testing.assert_allclose(measured / 2, expected, atol=0.03)
    img = micro.voxelize((n, n, n))
    assert np.array_equal(img.data, micro.cell_phases[lab])
    window = model.generate(mg.Domain.cube(1.0), seed=4)
    small = window.voxelize((20, 20, 20))
    inside = window.contains(small.grid().center_points()).reshape(small.shape)
    assert np.array_equal(inside, small.data == 1)
    assert 0.3 < small.data.mean() < 0.5
    with pytest.raises(ValueError, match="phase_fractions"):
        mg.models.DeadLeavesModel(radius=0.1, phase_fractions=(0.5, 0.6))


def _same_leaf(model, seeds, n=96, lags=10):
    """The probability that two voxels at the lags 0..lags-1 along each axis see the same
    leaf, the mean fraction of each phase and the realizations."""
    same, fractions, micros = np.zeros((3, lags)), {}, []
    for seed in seeds:
        micro = model.generate(CUBE, seed=seed)
        lab = micro.labels((n, n, n)).data
        cells = micro.cell_phases[lab]
        for p in np.unique(cells):
            fractions[int(p)] = fractions.get(int(p), 0.0) + np.mean(cells == p) / len(seeds)
        for a in range(3):
            same[a] += [np.mean(lab == np.roll(lab, -k, axis=a)) for k in range(lags)]
        micros.append(micro)
    return same / len(seeds), fractions, micros


def test_dead_leaves_families_spheroids_and_coverage():
    """Families of leaves of their own sizes (the proportions of their leaves, ``q_i /
    E[V_i]``), spheroids (isotropic, and aligned: the probability of the same leaf along
    each axis), a finite coverage over a background (its fraction ``e^{-c}``, the
    background as the last cell): the realizations against the theory, the limits of the
    bivariate laws, the round trips of the configurations."""
    h = np.arange(10) / 96
    families = mg.models.DeadLeavesModel(phase_radius=(0.03, 0.07), phase_fractions=(0.4, 0.6))
    (_, _, q0, _), (_, _, q1, p1) = families.families()
    assert (q0, q1) == (0.4, 0.6) and p1 == pytest.approx(
        (0.6 / 0.07**3) / (0.4 / 0.03**3 + 0.6 / 0.07**3))  # fmt: skip
    same, fractions, micros = _same_leaf(families, seeds=(0, 1, 2))
    assert np.mean(micros[0].leaf_phases == 1) == pytest.approx(p1, abs=0.01)
    assert fractions[1] == pytest.approx(0.6, abs=0.04)
    np.testing.assert_allclose(same.mean(axis=0), families.same_leaf_probability(h), atol=0.02)
    assert families.covariance(0.0, 1) == pytest.approx(0.6)
    assert families.covariance(1.0, 1) == pytest.approx(0.36)  # independent beyond 2 R
    # spheroids, isotropic and aligned along z
    prolate = mg.models.DeadLeavesModel(radius=0.04, shape="spheroid", aspect_ratio=2.5)
    same, _, _ = _same_leaf(prolate, seeds=(0, 1))
    np.testing.assert_allclose(same.mean(axis=0), prolate.same_leaf_probability(h), atol=0.02)
    aligned = mg.models.DeadLeavesModel(
        radius=0.04,
        shape="spheroid",
        aspect_ratio=2.5,
        orientation={"type": "aligned", "direction": [0, 0, 1]},
    )
    same, _, (micro, _) = _same_leaf(aligned, seeds=(0, 1))
    for a in range(3):
        expected = aligned.same_leaf_probability(h, direction=np.eye(3)[a])
        np.testing.assert_allclose(same[a], expected, atol=0.02)
    assert same[2, 5] > same[0, 5] + 0.1  # longer along the axes of the leaves
    grid = mg.Domain.cube(1.0, periodic=True)
    img = micro.voxelize((24, 24, 24))
    inside = micro.contains(img.grid().center_points()).reshape(img.shape)
    assert np.array_equal(inside, img.data == 1)
    # a finite coverage over a background: the uncovered points, the last cell
    covered = mg.models.DeadLeavesModel(radius=0.06, coverage=0.8, background=2)
    assert covered.active_phases() == {0: "leaves of phase 0", 1: "leaves of phase 1",
                                       2: "background (phase 2)"}  # fmt: skip
    t = covered.theory()
    assert t["volume_fraction_2"] == t["background_fraction"] == pytest.approx(math.exp(-0.8))
    assert t["volume_fraction_1"] == pytest.approx(0.5 * (1 - math.exp(-0.8)))
    fraction, centred = 0.0, np.zeros(10)  # four realizations (sigma 0.016 each)
    for seed in range(4):
        micro = covered.generate(grid, seed=seed)
        lab = micro.labels((64, 64, 64)).data
        background = lab == micro.n_leaves
        assert micro.cell_phases[micro.n_leaves] == 2
        fraction += background.mean() / 4
        centred += np.array([np.mean(background & np.roll(background, -k, axis=0))
                             for k in range(10)]) / 4 - background.mean() ** 2 / 4  # fmt: skip
    p = math.exp(-0.8)
    assert fraction == pytest.approx(p, abs=0.025)
    np.testing.assert_allclose(centred, covered.covariance(np.arange(10) / 64, 2) - p * p,
                               atol=0.01)  # fmt: skip
    img = micro.voxelize((24, 24, 24))
    assert np.array_equal(micro.leaf_of(img.grid().center_points()).reshape(img.shape),
                          micro.labels((24, 24, 24)).data)  # fmt: skip
    # the limits of the theory: a large coverage, the independence beyond the leaves
    full = mg.models.DeadLeavesModel(radius=0.06, coverage=60.0, background=2)
    complete = mg.models.DeadLeavesModel(radius=0.06)
    np.testing.assert_allclose(full.covariance(h, 1), complete.covariance(h, 1), atol=1e-12)
    assert covered.covariance(0.5, 2) == pytest.approx(math.exp(-1.6))
    assert np.allclose(covered.bivariate(h).sum(axis=(-2, -1)), 1.0)
    assert complete.theory()["integral_range"] == pytest.approx(
        0.6064 * 4 / 3 * math.pi * 0.06**3, rel=1e-3)  # fmt: skip
    for model in (families, aligned, covered):
        again = mg.models.model_from_config(json.loads(json.dumps(model.to_config())))
        assert again == model
        restored = Microstructure.from_dict(json.loads(json.dumps(
            model.generate(grid, seed=1).to_dict())))  # fmt: skip
        assert np.array_equal(
            restored.labels((16, 16, 16)).data,
            model.generate(grid, seed=1).labels((16, 16, 16)).data,
        )
    for bad, match in [({"radius": 0.1, "phase_radius": [0.1, 0.1]}, "radius"),
                       ({"phase_radius": [0.1]}, "phase_radius"),
                       ({"radius": 0.1, "aspect_ratio": 2.0}, "spheroid"),
                       ({"radius": 0.1, "shape": "cube"}, "shape"),
                       ({"radius": 0.1, "coverage": -1.0}, "coverage")]:  # fmt: skip
        with pytest.raises(ValueError, match=match):
            mg.models.DeadLeavesModel(**bad)


def test_kelvin_cells():
    """Struts of the Kelvin foam: 24 per cubic cell, 12 nodes of four struts each."""
    start, end = lattice_struts("kelvin", 1.0, np.zeros(3), np.full(3, 2.0))

    def key(x):
        return tuple(np.round(np.mod(x, 2.0), 6) % 2.0)

    struts = {tuple(sorted((key(a), key(b)))) for a, b in zip(start, end, strict=True)}
    degree = {}
    for a, b in struts:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    assert len(struts) == 24 * 8 and len(degree) == 12 * 8 and set(degree.values()) == {4}
    assert np.allclose(np.linalg.norm(end - start, axis=1), math.sqrt(2.0) / 4.0)
    model = mg.models.StrutLattice(cell="kelvin", cell_size=0.5, volume_fraction=0.1)
    img = model.generate(CUBE, seed=0).voxelize((64, 64, 64))
    assert img.data.mean() == pytest.approx(0.1, abs=0.01)


def test_diamond_and_pyrochlore_cells():
    """Struts of the diamond lattice (16 per cell, 8 nodes of four struts along the
    diagonals of the cell) and of the pyrochlore lattice (48 per cell, 16 nodes of six
    struts along the diagonals of its faces), and their models."""
    for cell, struts, nodes, degree, length in (
        ("diamond", 16, 8, 4, math.sqrt(3.0) / 4.0),
        ("pyrochlore", 48, 16, 6, math.sqrt(2.0) / 4.0),
    ):
        start, end = lattice_struts(cell, 1.0, np.zeros(3), np.full(3, 2.0))

        def key(x):
            return tuple(np.round(np.mod(x, 2.0), 6) % 2.0)

        found = {tuple(sorted((key(a), key(b)))) for a, b in zip(start, end, strict=True)}
        count = {}
        for a, b in found:
            count[a] = count.get(a, 0) + 1
            count[b] = count.get(b, 0) + 1
        assert len(found) == struts * 8 and len(count) == nodes * 8
        assert set(count.values()) == {degree}
        assert np.allclose(np.linalg.norm(end - start, axis=1), length)
        model = mg.models.StrutLattice(cell=cell, cell_size=1.0, volume_fraction=0.1)
        img = model.generate(CUBE, seed=0).voxelize((64, 64, 64))
        assert img.data.mean() == pytest.approx(0.1, abs=0.01)


def test_laguerre_of_packings():
    """Germs of a hard-sphere packing: the cells follow the sizes of the spheres."""
    model = mg.models.Laguerre(
        n_cells=150, radius=D.LogNormal(1.0, 0.5), germs="packing", packing_fraction=0.35
    )
    tess = model.generate(CUBE, seed=2)
    radii = np.sqrt(tess.weights)
    assert 4 / 3 * math.pi * np.sum(radii**3) == pytest.approx(0.35)
    gaps = [
        np.linalg.norm((tess.germs[i] - tess.germs[j] + 0.5) % 1.0 - 0.5) - radii[i] - radii[j]
        for i, j in itertools.combinations(range(tess.n_cells), 2)
    ]
    assert min(gaps) >= -1e-12  # no overlap
    volumes = tess.cell_complex().cell_volumes(tess.n_cells)
    assert np.corrcoef(volumes, radii**3)[0, 1] > 0.9
    poisson = mg.models.Laguerre(n_cells=150, radius=D.LogNormal(0.05, 0.025))
    other = poisson.generate(CUBE, seed=2)
    loose = other.cell_complex().cell_volumes(other.n_cells)
    assert np.corrcoef(loose, other.weights**1.5)[0, 1] < 0.9
    for bad, message in (({"germs": "lattice"}, "germs"), ({"packing_fraction": 0.7}, "packing")):
        with pytest.raises(ValueError, match=message):
            mg.models.Laguerre(n_cells=10, **bad)


def test_transparent_dead_leaves():
    """The random function of transparent leaves: its mean, variance and covariance
    against ``σ^2 α r / (1 + r (1 - α))``, the fraction of its excursion over
    realizations; opaque leaves are the visible leaves of the dead leaves; the values at
    points and on the voxels; the threshold of a fraction from the law of the series."""
    from morphogen.models.deadleaves import DeadLeaves

    model = mg.models.TransparentDeadLeavesModel(radius=0.05, transparency=0.4,
                                                 volume_fraction=0.4)  # fmt: skip
    t = model.theory()
    assert t["field_mean"] == 0.5 and t["field_variance"] == pytest.approx(0.4 / 1.6 / 12)
    assert model.field_covariance(0.0) == pytest.approx(t["field_variance"])
    assert model.field_covariance(0.2) == pytest.approx(0.0)  # beyond the diameters
    fractions, cov, var = [], np.zeros(8), 0.0
    for seed in range(4):
        micro = model.generate(CUBE, seed=seed)
        z = micro.field((64, 64, 64)).data.astype(float)
        fractions.append(np.mean(z > micro.threshold))
        var += z.var() / 4
        cov += np.array([np.mean((z - z.mean()) * (np.roll(z, -k, axis=1) - z.mean()))
                         for k in range(8)]) / 4  # fmt: skip
    assert np.mean(fractions) == pytest.approx(0.4, abs=0.01)
    assert var == pytest.approx(t["field_variance"], rel=0.05)
    np.testing.assert_allclose(cov, model.field_covariance(np.arange(8) / 64),
                               atol=0.05 * t["field_variance"])  # fmt: skip
    img = micro.field((12, 12, 12))
    np.testing.assert_allclose(micro.values_at(img.grid().center_points()), img.data.ravel(),
                               atol=1e-6)  # fmt: skip
    phase = micro.voxelize((12, 12, 12))
    inside = micro.contains(phase.grid().center_points()).reshape(phase.shape)
    assert np.array_equal(inside, phase.data == 1)
    # opaque leaves: the value of the visible leaf of the dead leaves of the same leaves
    opaque = mg.models.TransparentDeadLeavesModel(
        radius=0.06, transparency=1.0, threshold=0.5, values="bernoulli", value_probability=0.3
    )
    micro = opaque.generate(CUBE, seed=5)
    leaves = DeadLeaves(CUBE, micro.centers, micro.radii, np.zeros(micro.n_leaves))
    labels = leaves.labels((32, 32, 32)).data
    assert np.array_equal(micro.field((32, 32, 32)).data, micro.values[labels].astype(np.float32))
    assert opaque.theory()["volume_fraction"] == pytest.approx(0.3, abs=2e-3)
    uniform = mg.models.TransparentDeadLeavesModel(radius=0.06, transparency=1.0,
                                                   volume_fraction=0.4)  # fmt: skip
    assert uniform.field_threshold() == pytest.approx(0.6, abs=2e-3)  # a quantile of U(0, 1)
    again = mg.models.model_from_config(json.loads(json.dumps(model.to_config())))
    assert again == model
    back = Microstructure.from_dict(json.loads(json.dumps(micro.to_dict())))
    assert np.array_equal(back.voxelize((16, 16, 16)).data, micro.voxelize((16, 16, 16)).data)
    for bad, match in [({"transparency": 0.0, "threshold": 0.5}, "transparency"),
                       ({"threshold": 0.5, "volume_fraction": 0.3}, "exactly one"),
                       ({"threshold": 0.5, "values": "gauss"}, "values")]:  # fmt: skip
        with pytest.raises(ValueError, match=match):
            mg.models.TransparentDeadLeavesModel(radius=0.05, **bad)
