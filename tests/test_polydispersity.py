"""Monodisperse or polydisperse objects: a size given as a number is one size, a
distribution one size per object (fibre, strut, fibre of a layer), drawn once; with a volume
fraction, the law of the sizes keeps its shape and its scale is solved. The closed forms
with the moments of the laws against realizations and independent computations."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import spatial, stats

import morphogen as mg
from morphogen.distributions import (
    Constant,
    Discrete,
    LogNormal,
    TruncatedNormal,
    Uniform,
    distribution_from_config,
    ks_statistic,
)
from morphogen.homogenization.networks import network_frame, network_stiffness
from morphogen.kernels.rsa import segment_distance2
from morphogen.models import _sizes as sizes
from morphogen.models.base import model_from_config
from morphogen.models.fibers import excluded_volume, mean_sine
from morphogen.models.lattice import lattice_struts, periodic_struts

#: Two radii, 0.5 and 1, three fibres in five of the first
TWO = {"type": "discrete", "values": [0.5, 1.0], "weights": [0.6, 0.4]}


@pytest.mark.parametrize(
    "law",
    [
        Constant(2.0),
        Uniform(0.5, 1.5),
        LogNormal(1.0, 0.3),
        TruncatedNormal(1.0, 0.4, 0.2, 2.0),
        TruncatedNormal(1.0, 0.4),
        Discrete((1.0, 2.0), (0.3, 0.7)),
    ],
    ids=lambda law: f"{type(law).__name__}",
)
def test_laws_scaled(law):
    """The law of s X: its moments s^k E[X^k], its largest value s times, its draws those of
    the law times s, its configuration read back."""
    s = 2.5
    scaled = law.scaled(s)
    assert type(scaled) is type(law)
    for k in (1, 2, 3):
        assert scaled.moment(k) == pytest.approx(s**k * law.moment(k), rel=1e-9)
    assert scaled.upper() == pytest.approx(s * law.upper(), rel=1e-6)
    x = scaled.sample(np.random.default_rng(1), 2000)
    y = law.sample(np.random.default_rng(1), 2000)
    np.testing.assert_allclose(x, s * y, rtol=1e-9)
    assert distribution_from_config(scaled.to_config()) == scaled
    with pytest.raises(ValueError, match="scale factor"):
        law.scaled(0.0)


@pytest.mark.parametrize(
    "law",
    [
        Constant(2.0),
        Uniform(0.5, 1.5),
        LogNormal(1.0, 0.3),
        TruncatedNormal(1.0, 0.4, 0.2, 2.0),
        Discrete((1.0, 2.0), (0.3, 0.7)),
    ],
    ids=lambda law: f"{type(law).__name__}",
)
def test_distribution_functions(law):
    """The distribution function of each law against the empirical one of 40 000 draws
    (within 0.01) and against that of its quadrature; the Kolmogorov–Smirnov distance of the
    draws, that of SciPy for a continuous law, NaN for a constant or a discrete one."""
    x = law.sample(np.random.default_rng(2), 40000)
    points = np.quantile(x, [0.05, 0.3, 0.5, 0.7, 0.95])
    empirical = np.array([(x <= q).mean() for q in points])
    np.testing.assert_allclose(law.cdf(points), empirical, atol=0.01)
    base = mg.distributions.Distribution.cdf(law, points)
    np.testing.assert_allclose(base, law.cdf(points), atol=0.01)
    if isinstance(law, (Constant, Discrete)):
        assert math.isnan(ks_statistic(x, law))
    else:
        assert ks_statistic(x[:500], law) == pytest.approx(
            stats.kstest(x[:500], law.cdf).statistic, rel=1e-12
        )
        assert ks_statistic(x, law) < 0.01


def test_sizes_of_two_fibres_in_contact():
    """The moments of D = R1 + R2 + gap of two independent radii against Monte Carlo; the
    sizes drawn and summarized only for a law that is not a constant."""
    law = LogNormal(1.0, 0.4)
    first, second = (law.sample(np.random.default_rng(k), 400_000) for k in (3, 4))
    d = first + second + 0.25
    for k, m in enumerate(sizes.sum_moments(law, 0.25), start=1):
        assert m == pytest.approx(np.mean(d**k), rel=3e-3)
    assert sizes.sum_moments(Constant(1.0), 0.5) == pytest.approx((2.5, 6.25, 15.625))
    assert sizes.size_summary("radius", [1.0, 2.0], Constant(1.0)) == {}
    assert sizes.size_summary("radius", [1.0, 3.0], law) == {"radius_mean": 2.0, "radius_std": 1.0}
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    assert np.all(sizes.draw(Constant(0.3), rng, 5) == 0.3)
    assert rng.bit_generator.state == state  # a number draws nothing: the same realizations


def test_polydisperse_poisson_fibres():
    """Lines along z of radii 0.2 and 1.2: in every plane normal to them, a Boolean model of
    discs of random radii; P(dist(x, A) <= r) = 1 - exp(-L_V pi E[(R + r)^2]) (the Choquet
    capacity of the balls, E[(R + r)^2] = E[R^2] + 2 r E[R] + r^2), measured on the exact
    distance at random points of twelve windows within four standard errors of their
    scatter, where the mean radius alone would be wrong; the fraction and specific surface
    with E[R^2] and E[R]."""
    law = {"type": "discrete", "values": [0.2, 1.2], "weights": [0.5, 0.5]}
    model = model_from_config({"type": "poisson_fibers", "radius": law, "volume_fraction": 0.3,
                               "orientation": "aligned"})  # fmt: skip
    r1, r2 = 0.7, 0.74
    lv = -math.log(0.7) / (math.pi * r2)
    theory = model.theory()
    assert model.lv == pytest.approx(lv) and theory["volume_fraction"] == pytest.approx(0.3)
    assert theory["specific_surface"] == pytest.approx(2.0 * math.pi * r1 * lv * 0.7)
    rho = np.array([0.0, 0.5, 1.0])
    expected = model.choquet_capacity("ball", rho)
    dilated = 0.5 * (0.2 + rho) ** 2 + 0.5 * (1.2 + rho) ** 2
    assert expected == pytest.approx(1.0 - np.exp(-lv * math.pi * dilated))
    mean_radius = 1.0 - np.exp(-lv * math.pi * (r1 + rho) ** 2)  # without the variance
    domain = mg.Domain(size=(60.0, 60.0, 1.0), periodic=False)
    rng = np.random.default_rng(7)
    measured, radii = [], []
    for seed in range(12):
        micro = model.generate(domain, seed=seed)
        radii.append(micro.segments.radii)
        points = np.column_stack([rng.uniform(5.0, 55.0, (20_000, 2)), np.full(20_000, 0.5)])
        phi = micro.level_set_at(points, reach=2.0)
        measured.append(np.mean(phi[:, None] <= rho[None, :], axis=0))
    measured = np.array(measured)
    error = measured.std(axis=0, ddof=1) / math.sqrt(len(measured))
    assert np.all(np.abs(measured.mean(axis=0) - expected) < 4.0 * error + 1e-3)
    assert abs(measured.mean(axis=0)[2] - mean_radius[2]) > 4.0 * error[2]
    radii = np.concatenate(radii)
    assert set(np.unique(radii)) == {0.2, 1.2}
    assert np.mean(radii == 0.2) == pytest.approx(0.5, abs=0.05)
    summary = micro.summary()
    assert summary["radius_mean"] == pytest.approx(np.mean(micro.segments.radii))
    assert "radius_mean" not in _poisson_fibres_of(0.5).generate(domain, seed=1).summary()


def _poisson_fibres_of(r):  # a model of a fixed radius
    return model_from_config({"type": "poisson_fibers", "radius": r, "volume_fraction": 0.3})


def _min_gaps(segments, period, per_fibre: int) -> np.ndarray:
    """The least distance between the surfaces of every two fibres of a periodic box
    (chains of ``per_fibre`` segments; nearest images), by every pair of segments within
    reach."""
    start, end, radii = segments.start, segments.end, segments.radii
    fibre = np.arange(len(start)) // per_fibre
    mid = 0.5 * (start + end)
    reach = float(np.max(segments.lengths) + 2.0 * radii.max())
    tree = spatial.cKDTree(np.mod(mid, period), boxsize=period)
    gaps = {}
    for a, b in tree.query_pairs(reach, output_type="ndarray"):
        if fibre[a] == fibre[b]:
            continue
        shift = -period * np.round((mid[b] - mid[a]) / period)
        d2 = segment_distance2(start[a], end[a] - start[a], start[b] + shift, end[b] - start[b])
        key = (min(fibre[a], fibre[b]), max(fibre[a], fibre[b]))
        gap = math.sqrt(d2) - radii[a] - radii[b]
        gaps[key] = min(gaps.get(key, math.inf), gap)
    return np.array(list(gaps.values()))


def test_polydisperse_curved_fibres():
    """Curved fibres of radii of a law: one radius along each fibre; placed without overlap,
    no two closer than the sum of their radii (every pair of segments); the Boolean model of
    straight ones of volume fraction 1 - exp(-theta (pi E[R^2] L + 4/3 pi E[R^3]))."""
    law = {"type": "uniform", "low": 0.1, "high": 0.3}
    packed = model_from_config({"type": "curved_fibers", "radius": law, "length": 2.0,
                                "volume_fraction": 0.06, "overlap": False, "segments": 4,
                                "persistence": 3.0})  # fmt: skip
    micro = packed.generate(mg.Domain.cube(6.0, periodic=True), seed=3)
    radii = micro.segments.radii.reshape(-1, 4)
    assert np.all(radii == radii[:, :1]) and len(radii) > 30
    assert radii.min() >= 0.1 and radii.max() <= 0.3 and np.ptp(radii[:, 0]) > 0.1
    assert _min_gaps(micro.segments, np.full(3, 6.0), 4).min() >= -1e-9
    e2, e3 = (0.3**3 - 0.1**3) / 0.6, (0.3**4 - 0.1**4) / 0.8
    assert packed.mean_grain_volume == pytest.approx(math.pi * e2 * 2.0 + 4.0 / 3.0 * math.pi * e3)
    assert micro.summary()["radius_mean"] == pytest.approx(radii[:, 0].mean())
    hollow = model_from_config({**packed.to_config(), "lumen": 0.5})  # the same fibres
    walls = hollow.generate(mg.Domain.cube(6.0, periodic=True), seed=3)
    assert walls.summary()["radius_mean"] == micro.summary()["radius_mean"]
    straight = model_from_config({"type": "curved_fibers", "radius": law, "length": 2.0,
                                  "volume_fraction": 0.2, "persistence": 1e9})  # fmt: skip
    box = mg.Domain.cube(8.0, periodic=True)
    fractions = [straight.generate(box, seed=s).voxelize(64).data.mean() for s in range(4)]
    assert np.mean(fractions) == pytest.approx(0.2, abs=0.012)
    assert straight.theory()["volume_fraction"] == pytest.approx(0.2)


def test_polydisperse_cross_linked_fibres():
    """Fibres of radii of a law in contact where their axes come closer than r_a + r_b +
    gap: exactly the pairs of an independent computation; the excluded volume over the
    moments of D against Monte Carlo; contacts per unit length 2 E[D] L_V E|sin g|."""
    law = {"type": "uniform", "low": 0.2, "high": 0.6}
    model = model_from_config({"type": "cross_linked_fibers", "radius": law, "length": 4.0,
                               "volume_fraction": 0.1, "contact_gap": 0.05})  # fmt: skip
    domain = mg.Domain.cube(12.0, periodic=True)
    micro = model.generate(domain, seed=2)
    seg = micro.parts[0].segments
    pairs, _ = model.contacts(seg.start, seg.end, domain, seg.radii)
    found = {(int(a), int(b)) for a, b in pairs}
    true = set()
    mid = 0.5 * (seg.start + seg.end)
    for a in range(len(seg)):
        for b in range(a + 1, len(seg)):
            shift = -12.0 * np.round((mid[b] - mid[a]) / 12.0)
            d2 = segment_distance2(seg.start[a], seg.end[a] - seg.start[a],
                                   seg.start[b] + shift, seg.end[b] - seg.start[b])  # fmt: skip
            if d2 < (seg.radii[a] + seg.radii[b] + 0.05) ** 2:
                true.add((a, b))
    assert found == true and len(true) > 20
    theory = model.theory()
    theta, sine = model.lv / 4.0, mean_sine(model.orientation)
    rng = np.random.default_rng(5)
    d = rng.uniform(0.2, 0.6, 200_000) + rng.uniform(0.2, 0.6, 200_000) + 0.05
    monte_carlo = np.mean(excluded_volume(d, 4.0, 16.0, sine))
    assert theory["contacts_per_fibre"] == pytest.approx(theta * monte_carlo, rel=3e-3)
    assert theory["contact_spacing"] == pytest.approx(1.0 / (2.0 * (0.8 + 0.05) * model.lv * sine))
    e2, e3 = (0.6**3 - 0.2**3) / 1.2, (0.6**4 - 0.2**4) / 1.6
    assert model.grain_volume == pytest.approx(math.pi * e2 * 4.0 + 4.0 / 3.0 * math.pi * e3)
    assert {"radius_mean", "radius_std"} <= set(micro.summary())


def test_polydisperse_bouligand():
    """Fibres of radii of their own in the layers of a plywood: along x and y in a periodic
    box, a layer holds L / s fibres, each of its own radius, its fraction pi mean(r^2) /
    (s h) on the voxels; in a window, the radii drawn follow their law; the theory with
    E[R^2] and E[R]; no law wider than the spacing."""
    law = {"type": "uniform", "low": 0.03, "high": 0.06}
    model = model_from_config({"type": "bouligand", "fiber_radius": law, "fiber_spacing": 0.15,
                               "layer_thickness": 0.125, "pitch_angle": 90.0,
                               "start_angle": 0.0})  # fmt: skip
    micro = model.generate(mg.Domain(size=[1.2, 1.2, 0.5], periodic=True), seed=1)
    assert [len(r) for r in micro.radii] == [8, 8, 8, 8]  # 1.2 / 0.15 fibres in each layer
    exact = np.mean([math.pi * np.mean(r * r) / (0.15 * 0.125) for r in micro.radii])
    image = micro.voxelize(240)
    assert image.data.mean() == pytest.approx(exact, rel=0.01)
    assert micro.summary()["fiber_fraction"] == pytest.approx(exact)
    e1, e2 = 0.045, (0.06**3 - 0.03**3) / 0.09
    theory = model.theory()
    assert theory["volume_fraction"] == pytest.approx(math.pi * e2 / (0.15 * 0.125))
    assert theory["specific_surface"] == pytest.approx(2.0 * math.pi * e1 / (0.15 * 0.125))
    window = model_from_config({**model.to_config(), "pitch_angle": 30.0})
    drawn = np.concatenate(window.generate(mg.Domain.cube(2.0, periodic=False), seed=3).radii)
    assert len(drawn) > 200
    assert stats.kstest(drawn, "uniform", args=(0.03, 0.03)).pvalue > 0.01
    with pytest.raises(ValueError, match="fiber_spacing"):
        model_from_config({"type": "bouligand", "fiber_radius": law, "fiber_spacing": 0.1})


def test_struts_of_periodic_lattices():
    """Each strut of a periodic box once: 3, 8 and 24 per cubic, bcc and octet cell, the
    length per unit volume 3, 4 sqrt 3 and 12 sqrt 2 over a^2."""
    box = mg.Domain.cube(2.0, periodic=True)
    for cell, count, lv in (("cubic", 3, 3.0), ("bcc", 8, 4.0 * math.sqrt(3.0)),
                            ("octet", 24, 12.0 * math.sqrt(2.0))):  # fmt: skip
        start, end = lattice_struts(cell, 1.0, box.lower, box.upper)
        once, _ = periodic_struts(start, end, box)
        assert len(once) == 8 * count, cell
        law = {"type": "uniform", "low": 0.02, "high": 0.04}
        model = model_from_config({"type": "strut_lattice", "cell": cell, "cell_size": 0.5,
                                   "strut_radius": law})  # fmt: skip
        assert model.length_density == pytest.approx(lv / 0.25)


def test_polydisperse_strut_lattices():
    """A radius per strut: thin struts take pi E[R^2] L_V to first order (the nodes, a few
    percent, left out); with a volume fraction, the scale of the law reaches it; hollow
    struts of several radii take their radius."""
    law = {"type": "uniform", "low": 0.02, "high": 0.05}
    domain = mg.Domain.cube(2.0, periodic=True)
    model = model_from_config({"type": "strut_lattice", "cell": "cubic", "strut_radius": law})
    micro = model.generate(domain, seed=4)
    radii = micro.segments.radii
    assert len(radii) == 24 and 0.02 <= radii.min() and radii.max() <= 0.05
    theory = model.theory(domain)
    e2 = (0.05**3 - 0.02**3) / 0.09
    assert theory["volume_fraction_first_order"] == pytest.approx(math.pi * e2 * 3.0)
    exact = math.pi * np.sum(radii**2) * 1.0 / domain.volume  # 24 struts of unit length
    measured = micro.voxelize(160, supersampling=2).data.mean()
    assert measured == pytest.approx(exact, rel=0.04) and measured < exact
    lognormal = {"type": "lognormal", "mean": 0.05, "std": 0.015}
    fitted = model_from_config({"type": "strut_lattice", "cell": "octet", "volume_fraction": 0.15,
                                "strut_radius": lognormal})  # fmt: skip
    solved = fitted.generate(domain, seed=1)
    assert solved.voxelize(128).data.mean() == pytest.approx(0.15, abs=0.004)
    assert solved.summary()["radius_scale"] > 0
    with pytest.raises(ValueError, match="hollow struts of several radii"):
        model_from_config({"type": "strut_lattice", "volume_fraction": 0.1, "strut_radius": law,
                           "wall_thickness": 0.01})  # fmt: skip
    with pytest.raises(ValueError, match="exactly one of 'strut_radius'"):
        model_from_config({"type": "strut_lattice", "volume_fraction": 0.1, "strut_radius": 0.05})


def test_polydisperse_networks():
    """Foams and fibre networks of struts of radii of a law: the scale of the law solved for
    the volume fraction, the spheres of the nodes node_ratio times the largest strut that
    meets there, the frame of beams of each radius (its stiffness that of the radii kept)."""
    domain = mg.Domain.cube(1.0, periodic=True)
    law = {"type": "lognormal", "mean": 0.02, "std": 0.006}
    foam = model_from_config({"type": "open_foam", "n_cells": 12, "volume_fraction": 0.08,
                              "node_ratio": 1.3, "strut_radius": law})  # fmt: skip
    micro = foam.generate(domain, seed=3)
    assert micro.voxelize(96).data.mean() == pytest.approx(0.08, abs=0.002)
    radii = np.asarray(micro.metadata["radii"])
    bars = np.asarray(micro.metadata["framework"]["bars"])
    largest = np.zeros(len(micro.spheres))
    for k in (0, 1):
        np.maximum.at(largest, bars[:, k], radii)
    np.testing.assert_allclose(micro.spheres.radii, 1.3 * largest)
    np.testing.assert_allclose(micro.segments.radii, radii)
    summary = micro.summary()
    assert summary["strut_radius"] == pytest.approx(radii.mean())
    # the radii over their scale follow the law given (mean 0.02, 158 struts)
    scale = summary["strut_radius_scale"]
    assert np.mean(radii / scale) == pytest.approx(0.02, abs=4.0 * 0.006 / math.sqrt(len(radii)))
    two = {"type": "discrete", "values": [0.01, 0.02], "weights": [0.5, 0.5]}
    network = model_from_config({"type": "delaunay_network", "n_points": 40, "radius": two,
                                 "frame_stiffness": True})  # fmt: skip
    net = network.generate(domain, seed=2)
    radii = np.asarray(net.metadata["radii"])
    assert set(np.unique(radii)) == {0.01, 0.02}
    frame = network_frame(net)
    assert np.allclose(np.unique(np.round(frame.area / math.pi, 12)), [1e-4, 4e-4])
    stiffness = network_stiffness(net.metadata["framework"], radii).summary()
    one = network_stiffness(net.metadata["framework"], radii.mean()).summary()
    assert net.summary()["frame_young"] == pytest.approx(stiffness["frame_young"], rel=1e-6)
    assert abs(stiffness["frame_young"] / one["frame_young"] - 1.0) > 0.01  # each radius
    lattice = model_from_config({"type": "random_lattice", "strut_radius": {"type": "uniform",
                                 "low": 0.04, "high": 0.08}, "removal_fraction": 0.1})  # fmt: skip
    struts = lattice.generate(mg.Domain.cube(2.0, periodic=True), seed=1)
    assert np.allclose(struts.segments.radii, struts.metadata["radii"])
    assert {"strut_radius_mean", "strut_radius_std"} <= set(struts.summary())
    with pytest.raises(ValueError, match="exactly one of 'radius'"):
        model_from_config({"type": "delaunay_network", "n_points": 20, "radius": 0.02,
                           "volume_fraction": 0.05})  # fmt: skip
    with pytest.raises(ValueError, match="exactly one of 'strut_radius'"):
        model_from_config({"type": "open_foam", "n_cells": 20})
