"""Bending fibres packed by forces (Altendorf & Jeulin 2011): the beta law of their main
directions, the multivariate von Mises-Fisher law of the steps of their random walks,
their bending, then the packing: no overlap, the volume fraction, seeds, periodicity."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import spatial, stats

import morphogen as mg
from morphogen.distributions import Constant
from morphogen.kernels.rsa import segment_distance2
from morphogen.models import model_from_config
from morphogen.models.fibers import (
    RandomWalkFibers,
    multivariate_von_mises_fisher_walk,
    von_mises_fisher,
)
from morphogen.orientations import Beta, Isotropic, orientation_from_config


def _clearance(micro, m: int, related: int) -> float:
    """Smallest ``d - r_a - r_b`` over the pairs of segments (``m`` per fibre, in order),
    nearest periodic images, but those of a fibre at most ``related`` steps apart: brute
    force over the pairs of midpoints within reach (scipy's periodic k-d tree)."""
    seg = micro.segments
    box = micro.domain.lengths
    mid = np.mod(seg.midpoints - micro.domain.lower, box)
    reach = 2.0 * seg.radii.max() + seg.lengths.max()
    pairs = spatial.cKDTree(mid, boxsize=box).query_pairs(reach, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    keep = ~((i // m == j // m) & (np.abs(i % m - j % m) <= related))
    out = np.inf
    for a, b in zip(i[keep], j[keep], strict=True):
        shift = mid[b] - mid[a]
        shift = shift - box * np.round(shift / box) - (seg.midpoints[b] - seg.midpoints[a])
        d2 = segment_distance2(
            seg.start[a], seg.end[a] - seg.start[a], seg.start[b] + shift, seg.end[b] - seg.start[b]
        )
        out = min(out, math.sqrt(d2) - seg.radii[a] - seg.radii[b])
    return out


def test_beta_orientation_law():
    rng = np.random.default_rng(1)
    for beta in (0.2, 1.0, 4.0):
        law = orientation_from_config({"type": "beta", "beta": beta, "axis": [1, 1, 0]})
        assert law == Beta(beta, (1.0, 1.0, 0.0))
        assert orientation_from_config(law.to_config()) == law
        c = np.abs(law.sample(rng, 20_000) @ np.array(law.axis))
        cdf = beta * c / np.sqrt(1.0 + (beta**2 - 1.0) * c**2)  # P(|cos| <= c)
        assert stats.kstest(cdf, "uniform").pvalue > 1e-3, beta
    with pytest.raises(ValueError, match="beta"):
        Beta(0.0)
    with pytest.raises(ValueError, match="beta"):
        orientation_from_config({"type": "beta", "kappa": 1.0})


def test_steps_follow_the_multivariate_von_mises_fisher_law():
    """Each step: the von Mises-Fisher law about ``k1 mu + k2 t'`` of concentration its
    length, i.e. the density proportional to ``exp(k1 mu.t + k2 t'.t)`` of the paper."""
    rng = np.random.default_rng(2)
    n, m, k1, k2 = 500, 20, 4.0, 12.0
    main = Isotropic().sample(rng, n)
    t = multivariate_von_mises_fisher_walk(rng, main, k1, k2, m)
    assert np.allclose(np.linalg.norm(t, axis=2), 1.0)
    prev = np.concatenate([main[:, None], t[:, :-1]], axis=1)
    v = k1 * main[:, None] + k2 * prev
    kappa = np.linalg.norm(v, axis=2)
    mean = v / kappa[..., None]
    w = np.einsum("fki,fki->fk", t, mean)  # density kappa e^(kappa w) / (2 sinh kappa)
    u = np.expm1(kappa * (w + 1.0)) / np.expm1(2.0 * kappa)  # its distribution function
    assert stats.kstest(u.ravel(), "uniform").pvalue > 1e-3
    e1 = np.cross(mean, [0.3, 0.5, 0.8])
    e1 /= np.linalg.norm(e1, axis=2, keepdims=True)
    phi = np.arctan2(
        np.einsum("fki,fki->fk", t, np.cross(mean, e1)), np.einsum("fki,fki->fk", t, e1)
    )
    assert stats.kstest((phi.ravel() + math.pi) / (2.0 * math.pi), "uniform").pvalue > 1e-3
    # kappa = 0: uniform; a large one: the mean cosine coth(kappa) - 1/kappa
    w0 = von_mises_fisher(rng, np.tile([0.0, 0.0, 1.0], (20_000, 1)), 0.0)[:, 2]
    assert stats.kstest((w0 + 1.0) / 2.0, "uniform").pvalue > 1e-3
    w1 = von_mises_fisher(rng, np.tile([0.0, 0.6, 0.8], (20_000, 1)), 500.0) @ [0.0, 0.6, 0.8]
    assert abs(w1.mean() - (1.0 / math.tanh(500.0) - 1.0 / 500.0)) < 4 * w1.std() / math.sqrt(2e4)


def _tortuosity_small_angles(k1: float, k2: float, m: int) -> float:
    """``E[T] - 1`` of the walks for small angles: the deviations ``xi`` of the steps from
    the main direction follow ``xi_k = a xi_(k-1) + eta_k``, ``a = k2 / (k1 + k2)``, ``eta``
    of second moment ``2 / (k1 + k2)`` and ``xi_0 = 0``, so that ``T - 1 = sum_k |xi_k -
    mean xi|^2 / 2m``."""
    a = k2 / (k1 + k2)
    k = np.arange(1, m + 1)
    lo = np.minimum.outer(k, k)
    cov = 2.0 / (k1 + k2) * a ** np.abs(np.subtract.outer(k, k)) * (1 - a ** (2 * lo)) / (1 - a * a)
    return (np.trace(cov) - cov.sum() / m) / (2 * m)


def test_bending_of_the_walks():
    """The reliability to the previous step recovered by the paper's estimator 2 / Var2
    (``k2 / k1 > 2``), that to the main direction overestimated by 2 / Var1 for chains
    shorter than the memory of the walk (as the paper notes before packing), the
    end-to-end vector along the main direction, and the tortuosity of the small angles."""
    rng = np.random.default_rng(3)
    n, m = 3000, 40
    for k1, k2 in ((10.0, 100.0), (10.0, 300.0), (100.0, 100.0)):
        model = RandomWalkFibers(radius=Constant(1.0), length=Constant(0.5 * m), kappa1=k1,
                                 kappa2=k2, number=n, orientation=Isotropic())  # fmt: skip
        nodes, first = model.walks(rng, np.ones(n), np.full(n, 0.5 * m))
        assert np.array_equal(first, np.arange(n + 1) * (m + 1))
        x = nodes.reshape(n, m + 1, 3)
        t = np.diff(x, axis=1)
        assert np.allclose(np.linalg.norm(t, axis=2), 0.5)
        t /= 0.5
        chord = x[:, -1] - x[:, 0]
        mean = chord / np.linalg.norm(chord, axis=1, keepdims=True)
        var1 = np.mean(np.sum((t - mean[:, None]) ** 2, axis=2))
        var2 = np.mean(np.sum(np.diff(t, axis=1) ** 2, axis=2))
        if k2 / k1 > 2.0:
            assert abs(2.0 / var2 / k2 - 1.0) < 0.08, (k1, k2, 2.0 / var2)
            assert 2.0 / var1 > 2.0 * k1
        tortuosity = 0.5 * m / np.linalg.norm(chord, axis=1) - 1.0
        expected = _tortuosity_small_angles(k1, k2, m)
        assert abs(tortuosity.mean() / expected - 1.0) < 0.06, (k1, k2, tortuosity.mean())
    # straight fibres along the main directions
    model = RandomWalkFibers(kappa1=0.0, kappa2=1e9, number=1, orientation=Beta(0.1))
    nodes, first = model.walks(rng, np.full(50, 0.5), np.full(50, 5.0))
    x = nodes.reshape(50, 21, 3)
    assert np.allclose(np.linalg.norm(x[:, -1] - x[:, 0], axis=1), 5.0)


@pytest.fixture(scope="module")
def packed():
    model = model_from_config(
        {
            "type": "random_walk_fibers",
            "radius": 0.3,
            "length": 3.0,
            "orientation": {"type": "beta", "beta": 0.5},
            "volume_fraction": 0.35,
        }
    )
    return model, model.generate(mg.Domain.cube(6.0, periodic=True), seed=4)


def test_packing_removes_the_overlaps(packed):
    model, micro = packed
    summary = micro.metadata["summary"]
    assert summary["packing_steps"] > 10 and summary["radius_reduction"] < 1e-6
    m = 20  # 3 / (0.5 x 0.3) segments per fibre, five steps apart at least
    assert len(micro.segments) == m * summary["fibres"]
    assert _clearance(micro, m, 5) > -1e-6 * 0.3
    # the chains kept: lengths of the segments and angles of the joints near their rest
    d = (micro.segments.end - micro.segments.start).reshape(-1, m, 3)
    ratio = np.linalg.norm(d, axis=2) / 0.15
    assert 0.98 < ratio.mean() < 1.06 and np.quantile(np.abs(ratio - 1.0), 0.9) < 0.12
    u = d / np.linalg.norm(d, axis=2, keepdims=True)
    turn = np.arccos(np.clip(np.einsum("fki,fki->fk", u[:, 1:], u[:, :-1]), -1.0, 1.0))
    assert np.quantile(turn, 0.99) < 0.6
    # the soft-core walks overlap
    soft = model_from_config({**model.to_config(), "packing": False})
    loose = soft.generate(mg.Domain.cube(6.0, periodic=True), seed=4)
    assert _clearance(loose, m, 5) < -0.3


def test_volume_fraction_voxels_and_periodicity(packed):
    model, micro = packed
    summary = micro.metadata["summary"]
    # the target, within the change of the contour lengths (the springs' 5-10 %)
    assert abs(summary["volume_fraction"] / 0.35 - 1.0) < 0.05
    assert model.theory(micro.domain)["volume_fraction"] == 0.35
    image = micro.voxelize(shape=120)
    assert abs(image.data.mean() / summary["volume_fraction"] - 1.0) < 0.01
    # periodic: each fibre continues across the faces (its next segment one period away)
    seg, box = micro.segments, micro.domain.lengths
    gap = (seg.start[1:] - seg.end[:-1]).reshape(-1, 3)
    joined = np.arange(1, len(seg)) % 20 != 0
    periods = gap[joined] / box
    assert np.allclose(periods, np.round(periods), atol=1e-9)
    assert np.any(np.round(periods) != 0)  # some fibres cross the faces
    lo, hi = micro.domain.lower - 1e-9, micro.domain.upper + 1e-9
    assert np.all((seg.midpoints >= lo) & (seg.midpoints <= hi))


def test_initial_placement_of_least_overlap(packed):
    """The initial placement of Altendorf, Jeulin & Willot (2014): each fibre at the best of
    its trial positions, the one of least overlap with those placed before it, which halves
    the packing steps at least (here from 92 to 14), the fibres still without overlap."""
    from morphogen.kernels.force_biased import place

    model, micro = packed
    placed = model_from_config({**model.to_config(), "placement_trials": 20})
    out = placed.generate(mg.Domain.cube(6.0, periodic=True), seed=4)
    s, s0 = out.metadata["summary"], micro.metadata["summary"]
    assert s["packing_steps"] < 0.5 * s0["packing_steps"] and s["placement_overlap"] > 0.0
    assert _clearance(out, 20, 5) > -1e-6 * 0.3
    assert abs(s["volume_fraction"] / 0.35 - 1.0) < 0.05
    # two straight chains along x, the second tried across the first, then away from it
    x = np.zeros((6, 3))
    x[:3, 0] = x[3:, 0] = [0.0, 1.0, 2.0]
    first = np.array([0, 3, 6])
    shifts = np.zeros((2, 3, 3))
    shifts[0, 0] = [1.0, 1.0, 1.0]
    shifts[1] = [[1.0, 1.2, 1.0], [1.0, 1.5, 1.0], [1.0, 3.0, 4.0]]  # depths 0.8, 0.5, 0
    radius = np.full(2, 0.5)
    box = np.full(3, 8.0)
    y = x.copy()
    assert place(y, first, radius, box, 0.0, shifts) == pytest.approx(0.0)
    assert np.allclose(y[3:] - x[3:], [1.0, 3.0, 4.0])
    y = x.copy()  # the shallower of two overlaps: 0.5 for each of the four pairs of segments
    assert place(y, first, radius, box, 0.0, shifts[:, :2]) == pytest.approx(4 * 0.5)
    assert np.allclose(y[3:] - x[3:], [1.0, 1.5, 1.0])
    with pytest.raises(ValueError, match="placement_trials"):
        RandomWalkFibers(volume_fraction=0.2, placement_trials=0)
    with pytest.raises(ValueError, match="needs packing"):  # a Boolean model, its overlaps kept
        RandomWalkFibers(volume_fraction=0.2, placement_trials=5, packing=False)
    empty = RandomWalkFibers(radius=0.25, length=2.0, number=0, placement_trials=5)
    assert empty.generate(mg.Domain.cube(5.0), seed=1).metadata["summary"] == {"fibres": 0}


def test_fibres_longer_than_the_box_avoid_their_images():
    cfg = {"type": "random_walk_fibers", "radius": 0.3, "length": 9.0,
           "orientation": {"type": "beta", "beta": 0.05}, "volume_fraction": 0.25}  # fmt: skip
    micro = model_from_config(cfg).generate(mg.Domain.cube(5.0, periodic=True), seed=2)
    assert _clearance(micro, 60, 5) > -1e-6 * 0.3
    assert abs(micro.metadata["summary"]["volume_fraction"] / 0.25 - 1.0) < 0.06


def test_seeds_end_steps_and_non_periodic_domains():
    cfg = {"type": "random_walk_fibers", "radius": 0.25, "length": 2.0, "kappa1": 5.0,
           "kappa2": 50.0, "volume_fraction": 0.3}  # fmt: skip
    model = model_from_config(cfg)
    box = mg.Domain.cube(5.0, periodic=True)
    a, b = model.generate(box, seed=7), model.generate(box, seed=7)
    assert np.array_equal(a.segments.start, b.segments.start)
    assert np.array_equal(a.segments.radii, b.segments.radii)
    assert not np.array_equal(model.generate(box, seed=8).segments.start, a.segments.start)
    # the radii reduced by half the largest overlap of their fibres, exactly hard core
    shrink = model_from_config({**cfg, "end_step": "shrink"}).generate(box, seed=7)
    assert shrink.metadata["summary"]["end_steps"] == 0
    assert 0.0 <= shrink.metadata["summary"]["radius_reduction"] < 0.05
    assert _clearance(shrink, 16, 5) > -1e-12
    assert np.all(shrink.segments.radii <= 0.25)
    # a non-periodic sample cut from a larger periodic box
    cut = model.generate(mg.Domain.cube(5.0, periodic=False), seed=7)
    assert abs(cut.voxelize(shape=80).data.mean() - 0.3) < 0.05
    with pytest.raises(RuntimeError, match="lower the volume fraction"):
        model_from_config({**cfg, "volume_fraction": 0.9, "max_steps": 50}).generate(box, seed=1)
    with pytest.raises(ValueError, match="end_step"):
        model_from_config({**cfg, "end_step": "none"})
