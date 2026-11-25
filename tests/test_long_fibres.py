"""Long fibres along an axis (unidirectional composites): their sections packed at random or
in arrays, straight or wavy, without overlap (an independent check of every pair of
segments), closed across the period, their volume fraction against its exact formula and
the voxels, their misalignment, the covariances of their waviness against the Gaussian
covariance, polydisperse radii, windows, configurations and seeds."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, optimize, spatial, stats

import morphogen as mg
from morphogen.kernels.rsa import segment_distance2
from morphogen.models import model_from_config
from morphogen.models.base import Microstructure
from morphogen.models.long_fibers import LongFibers, secant_mean

BOX = mg.Domain(size=(8.0, 8.0, 12.0), periodic=True)


def _chains(segments) -> np.ndarray:
    """The fibre of each segment: the chains of the segments joined end to start."""
    label = np.arange(len(segments))

    def root(i):
        while label[i] != i:
            label[i] = label[label[i]]
            i = label[i]
        return i

    starts = {tuple(np.round(p, 9)): i for i, p in enumerate(segments.start)}
    for i, p in enumerate(segments.end):
        j = starts.get(tuple(np.round(p, 9)))
        if j is not None:
            label[root(i)] = root(j)
    return np.array([root(i) for i in range(len(segments))])


def _clearance(micro, m: int | None) -> float:
    """Smallest ``d - r_a - r_b`` over the pairs of segments of different fibres (``m`` per
    fibre, in order; None: the chains of a window), nearest periodic images of the box (all
    three axes), by brute force over the pairs of midpoints within reach (scipy's periodic
    k-d tree) and Ericson's closest points (:func:`morphogen.kernels.rsa.segment_distance2`)."""
    seg = micro.segments
    fibre = np.arange(len(seg)) // m if m else _chains(seg)
    box = micro.domain.lengths
    mid = np.mod(seg.midpoints - micro.domain.lower, box)
    reach = 2.0 * seg.radii.max() + seg.lengths.max()
    tree = spatial.cKDTree(mid, boxsize=box) if micro.domain.periodic else spatial.cKDTree(mid)
    pairs = tree.query_pairs(reach, output_type="ndarray")
    out = np.inf
    for a, b in pairs:
        if fibre[a] == fibre[b]:
            continue
        shift = np.zeros(3)
        if micro.domain.periodic:
            d = mid[b] - mid[a]
            shift = d - box * np.round(d / box) - (seg.midpoints[b] - seg.midpoints[a])
        d2 = segment_distance2(
            seg.start[a], seg.end[a] - seg.start[a], seg.start[b] + shift, seg.end[b] - seg.start[b]
        )
        out = min(out, math.sqrt(d2) - seg.radii[a] - seg.radii[b])
    return out


def _per_fibre(micro):
    """The segments of the fibres (fibres x segments x 3: starts and vectors) and radii."""
    s = micro.metadata["summary"]
    n = s["fibres"]
    m = len(micro.segments) // n
    start = micro.segments.start.reshape(n, m, 3)
    vec = (micro.segments.end - micro.segments.start).reshape(n, m, 3)
    radii = micro.segments.radii.reshape(n, m)
    assert np.all(radii == radii[:, :1])  # one radius per fibre
    return start, vec, radii[:, 0]


def _exact_volume(vec, radii) -> float:
    """``pi r^2`` times the contour length of each closed chain, less ``2/3 r^3 (2 tan(d/2)
    - d)`` at each joint turning by ``d``."""
    lengths = np.linalg.norm(vec, axis=2)
    u = vec / lengths[..., None]
    d = np.arccos(np.clip(np.einsum("fki,fki->fk", np.roll(u, 1, axis=1), u), -1.0, 1.0))
    joints = np.sum(2.0 * np.tan(0.5 * d) - d, axis=1)
    return float(np.sum(math.pi * radii**2 * lengths.sum(axis=1) - 2.0 / 3.0 * radii**3 * joints))


def test_straight_parallel_fibres():
    """No misalignment: every fibre two segments of half the period exactly along the axis,
    closed across it, apart from the others, as many as bring the area of their sections
    closest to the fraction of the section of the box, which the voxels measure."""
    model = LongFibers(radius=0.5, volume_fraction=0.55)
    micro = model.generate(BOX, seed=3)
    start, vec, radii = _per_fibre(micro)
    n = len(radii)
    assert vec.shape == (n, 2, 3)
    assert np.allclose(vec, [0.0, 0.0, 6.0], atol=1e-12)
    assert np.allclose(start[:, 0, :2], start[:, 1, :2], atol=1e-12)
    area = math.pi * 0.25
    assert abs(n * area - 0.55 * 64.0) <= 0.5 * area  # the closest number of fibres
    s = micro.metadata["summary"]
    assert s["volume_fraction"] == pytest.approx(n * area / 64.0, rel=1e-12)
    assert s["misalignment_rms"] == s["misalignment_max"] == 0.0
    assert 0.0 < _clearance(micro, 2) == pytest.approx(s["smallest_gap"], abs=1e-9)
    image = micro.voxelize(voxel_size=0.04)
    assert np.mean(image.data) == pytest.approx(s["volume_fraction"], abs=2e-3)
    # the fibres cross the periodic faces normal to the axis unchanged
    assert np.array_equal(image.data[:, :, 0], image.data[:, :, -1])


@pytest.mark.parametrize(
    ("correlation", "gap"), [(None, 0.0), (0.0, 0.0), (None, 0.05)], ids=["wave", "own", "gap"]
)
def test_wavy_fibres_do_not_overlap(correlation, gap):
    """Misaligned fibres, waving together (by default) or each on its own, with or without a
    gap: no two segments of different fibres closer than ``r_a + r_b + min_gap``, the chains
    closed across the period, their volume the exact volume of the chains."""
    model = LongFibers(radius=0.5, volume_fraction=0.5, misalignment=4.0, min_gap=gap,
                       transverse_correlation=correlation)  # fmt: skip
    micro = model.generate(BOX, seed=4)
    start, vec, radii = _per_fibre(micro)
    m = vec.shape[1]
    assert m == 24  # 8 per wavelength of the three modes of a quarter of the period
    assert np.allclose(vec[..., 2], 12.0 / m)
    assert np.allclose(vec[..., :2].sum(axis=1), 0.0, atol=1e-9)  # closed across the period
    joined = np.mod(start[:, 1:] - (start[:, :-1] + vec[:, :-1]) + 4.0, 8.0) - 4.0
    assert np.allclose(joined[..., :2], 0.0, atol=1e-9)
    clearance = _clearance(micro, m)
    assert clearance >= gap
    s = micro.metadata["summary"]
    assert s["smallest_gap"] == pytest.approx(clearance, abs=1e-9)
    assert s["volume_fraction"] == pytest.approx(_exact_volume(vec, radii) / BOX.volume, rel=1e-9)
    image = micro.voxelize(voxel_size=0.04)
    assert np.mean(image.data) == pytest.approx(s["volume_fraction"], abs=2e-3)


def test_misalignment():
    """The RMS angle of the segments to the axis: exactly the misalignment before the
    corrections, near it after (the corrections bend the fibres a little), the summary
    that of the segments; the largest angle above it."""
    model = LongFibers(radius=0.5, volume_fraction=0.3, misalignment=5.0)
    rng = np.random.default_rng(5)
    x0 = rng.random((200, 2)) * 8.0
    offsets = model.waviness(rng, x0, (8.0, 8.0), 12.0, 24, 3.0)
    step = np.linalg.norm(np.roll(offsets, -1, axis=1) - offsets, axis=2)
    rms = math.degrees(math.sqrt(np.mean(np.arctan2(step, 0.5) ** 2)))
    assert rms == pytest.approx(5.0, rel=1e-9)
    for p, tol in ((0.3, 0.03), (0.6, 0.12)):
        micro = LongFibers(radius=0.5, volume_fraction=p, misalignment=5.0).generate(BOX, seed=6)
        _, vec, _ = _per_fibre(micro)
        angles = np.degrees(np.arccos(np.abs(vec[..., 2]) / np.linalg.norm(vec, axis=2)))
        s = micro.metadata["summary"]
        assert s["misalignment_rms"] == pytest.approx(math.sqrt(np.mean(angles**2)), rel=1e-9)
        assert s["misalignment_max"] == pytest.approx(angles.max(), rel=1e-9)
        assert s["misalignment_rms"] == pytest.approx(5.0, rel=tol)
        assert s["misalignment_max"] > 5.0


def test_covariances_of_the_waviness():
    """The offsets of the waviness: along a fibre the covariance of its Fourier modes,
    ``sum_j exp(-(k_j l)^2 / 2) cos(k_j h)`` over ``j >= 1``, normalized; across the fibres
    of the field the Gaussian covariance ``exp(-d^2 / 2 l_t^2)`` of their distance; the
    fibres of ``transverse_correlation: 0`` uncorrelated."""
    period, m, ell = 12.0, 24, 2.0
    k = 2.0 * math.pi * np.arange(1, 6) / period
    lags = np.arange(m // 2)
    weights = np.exp(-0.5 * (k * ell) ** 2)
    expected = (weights[:, None] * np.cos(np.outer(k, lags * period / m))).sum(0) / weights.sum()
    rng = np.random.default_rng(7)
    own = LongFibers(radius=0.5, volume_fraction=0.3, misalignment=3.0, transverse_correlation=0.0)
    x0 = rng.random((4000, 2)) * 8.0
    w = own.waviness(rng, x0, (8.0, 8.0), period, m, ell)
    w /= math.sqrt(np.mean(w**2))
    along = [np.mean(w * np.roll(w, -lag, axis=1)) for lag in lags]
    assert np.allclose(along, expected, atol=0.03)
    pair = np.mean(np.sum(w[0::2] * w[1::2], axis=2)) / 2.0  # two fibres apart
    assert abs(pair) < 0.03
    # across: the field at pairs of fibres at given distances, over many realizations
    wave = LongFibers(radius=0.5, volume_fraction=0.3, misalignment=3.0, transverse_correlation=1.5)
    distances = np.array([0.5, 1.0, 1.5, 2.5, 4.0])
    got = np.zeros(len(distances))
    for _ in range(300):
        base = rng.random((40, 2)) * 40.0
        phi = rng.uniform(0.0, 2.0 * math.pi, (40, len(distances)))
        other = base[:, None, :] + distances[None, :, None] * np.stack(
            [np.cos(phi), np.sin(phi)], 2
        )
        points = np.concatenate([base, other.reshape(-1, 2)])
        f = wave.waviness(rng, points, (40.0, 40.0), period, m, ell)
        f /= math.sqrt(np.mean(f**2))
        near = f[40:].reshape(40, len(distances), m, 2)
        got += np.mean(np.sum(f[:40, None] * near, axis=3), axis=(0, 2)) / 2.0 / 300.0
    assert np.allclose(got, np.exp(-0.5 * (distances / 1.5) ** 2), atol=0.04)


def test_secant_mean():
    """``E[sec a]`` of angles whose tangents follow the Rayleigh law of RMS angle ``theta``:
    against quadratures of its definition and its series ``1 + theta^2/2 + 5 theta^4/12``."""

    def density(r, sigma):  # the Rayleigh law of scale sigma
        return r / sigma**2 * math.exp(-0.5 * (r / sigma) ** 2)

    def mean(f, sigma):
        return integrate.quad(lambda r: f(r) * density(r, sigma), 0.0, np.inf, epsabs=1e-14)[0]

    for theta_deg in (1.0, 5.0, 15.0):
        theta = math.radians(theta_deg)
        sigma = optimize.brentq(
            lambda s, t=theta: mean(lambda r: math.atan(r) ** 2, s) - t * t, 1e-6, 10.0, xtol=1e-15
        )
        secant = mean(lambda r: math.sqrt(1.0 + r * r), sigma)
        assert secant_mean(theta) == pytest.approx(secant, rel=1e-9)
    theta = math.radians(3.0)
    assert secant_mean(theta) == pytest.approx(1.0 + theta**2 / 2 + 5 * theta**4 / 12, abs=1e-7)
    assert secant_mean(0.0) == 1.0


def test_polydisperse_radii():
    """One radius per fibre from its law (lognormal, a Kolmogorov-Smirnov test), the total
    area of the sections within half a section of the fraction of the box."""
    law = {"type": "lognormal", "mean": 0.5, "std": 0.08}
    domain = mg.Domain(size=(24.0, 24.0, 6.0), periodic=True)
    model = LongFibers(radius=law, volume_fraction=0.5)
    micro = model.generate(domain, seed=8)
    _, vec, radii = _per_fibre(micro)
    frozen = stats.lognorm(s=model.radius.sigma, scale=math.exp(model.radius.mu))
    assert stats.kstest(radii, frozen.cdf).pvalue > 1e-3
    area = math.pi * radii**2
    assert abs(area.sum() - 0.5 * 24.0**2) <= 0.5 * area.max()
    assert _clearance(micro, vec.shape[1]) > 0.0
    assert len(radii) == pytest.approx(model.expected_count(domain), rel=0.05)


@pytest.mark.parametrize("arrangement", ["hexagonal", "square"])
def test_arrays(arrangement):
    """The arrays of a box that holds a whole number of their cells: the exact spacing, the
    nearest distances of the sites, the volume fraction of the theory; their disordered
    sites apart; their largest fraction, with a gap, enforced."""
    p, r = 0.6, 0.5
    model = LongFibers(radius=r, volume_fraction=p, arrangement=arrangement)
    a = model.spacing()
    if arrangement == "hexagonal":
        assert a == pytest.approx(math.sqrt(2.0 * math.pi * r * r / (math.sqrt(3.0) * p)))
        size, sites, nearest = (10 * a, 6 * math.sqrt(3.0) * a, 5.0), 120, 6
    else:
        assert a == pytest.approx(math.sqrt(math.pi * r * r / p))
        size, sites, nearest = (8 * a, 8 * a, 5.0), 64, 4
    domain = mg.Domain(size=size, periodic=True)
    micro = model.generate(domain, seed=9)
    start, _, radii = _per_fibre(micro)
    assert len(radii) == sites == model.expected_count(domain)
    centres = np.mod(start[:, 0, :2], size[:2])
    d, _ = spatial.cKDTree(centres, boxsize=size[:2]).query(centres, k=nearest + 2)
    assert np.allclose(d[:, 1 : nearest + 1], a, rtol=1e-9)  # the first neighbours
    assert np.all(d[:, nearest + 1] > 1.4 * a)
    theory = model.theory(domain)
    assert theory["volume_fraction"] == pytest.approx(p, rel=1e-12)
    assert micro.metadata["summary"]["volume_fraction"] == pytest.approx(p, rel=1e-12)
    largest = math.pi / (2.0 * math.sqrt(3.0)) if arrangement == "hexagonal" else math.pi / 4.0
    assert theory["largest_fraction"] == pytest.approx(largest)
    disordered = LongFibers(radius=r, volume_fraction=p, arrangement=arrangement, disorder=0.2)
    micro = disordered.generate(domain, seed=10)
    assert len(micro.segments) == 2 * sites
    assert _clearance(micro, 2) > 0.0
    with pytest.raises(ValueError, match="at most"):
        LongFibers(radius=r, volume_fraction=largest * 1.01, arrangement=arrangement)
    with pytest.raises(ValueError, match="at most"):  # the gap lowers it
        LongFibers(radius=r, volume_fraction=0.98 * largest, arrangement=arrangement, min_gap=0.1)


def test_window_and_axes():
    """A non-periodic window of the medium: its fibres apart, its voxel fraction near the
    target; fibres along x and y."""
    window = mg.Domain(size=(10.0, 10.0, 8.0), periodic=False)
    micro = LongFibers(radius=0.5, volume_fraction=0.5, misalignment=3.0).generate(window, seed=11)
    lo, hi = micro.segments.bounds()
    assert np.all((hi >= window.lower) & (lo <= window.upper))
    assert _clearance(micro, None) > 0.0  # the segments of different chains
    assert len(np.unique(_chains(micro.segments))) == pytest.approx(64, rel=0.25)
    fraction = float(np.mean(micro.voxelize(voxel_size=0.05).data))
    assert fraction == pytest.approx(0.5, abs=0.05)
    for axis, size in ((0, (12.0, 8.0, 8.0)), (1, (8.0, 12.0, 8.0))):
        domain = mg.Domain(size=size, periodic=True)
        model = LongFibers(radius=0.5, volume_fraction=0.5, misalignment=3.0, axis=axis)
        micro = model.generate(domain, seed=12)
        _, vec, _ = _per_fibre(micro)
        assert np.allclose(vec[..., axis], 12.0 / vec.shape[1])
        assert np.allclose(vec[..., [a for a in range(3) if a != axis]].sum(axis=1), 0.0,
                           atol=1e-9)  # fmt: skip
        image = micro.voxelize(voxel_size=0.05)
        assert np.mean(image.data) == pytest.approx(
            micro.metadata["summary"]["volume_fraction"], abs=3e-3
        )


def test_theory():
    """The fraction, number of fibres per unit area of a section, length per unit volume
    and specific surface of the theory against a realization."""
    model = LongFibers(radius=0.5, volume_fraction=0.6, misalignment=3.0)
    micro = model.generate(BOX, seed=13)
    _, vec, radii = _per_fibre(micro)
    t = model.theory(BOX)
    s = micro.metadata["summary"]
    contour = np.linalg.norm(vec, axis=2).sum(axis=1)
    assert t["volume_fraction"] == 0.6
    assert s["volume_fraction"] == pytest.approx(0.6, abs=0.5 * math.pi * 0.25 / 64.0 * 1.01)
    assert t["intensity"] == pytest.approx(len(radii) / 64.0, rel=0.02)
    assert t["length_density"] == pytest.approx(contour.sum() / BOX.volume, rel=0.02)
    surface = np.sum(2.0 * math.pi * radii * contour) / BOX.volume
    assert t["specific_surface"] == pytest.approx(surface, rel=0.02)
    assert t["contour_factor"] == pytest.approx(contour.mean() / 12.0, rel=1e-3)
    assert model.expected_count(BOX) == pytest.approx(t["intensity"] * 64.0)
    fixed = LongFibers(radius=0.5, number=50)
    assert fixed.theory(BOX)["volume_fraction"] == pytest.approx(50 * math.pi * 0.25 / 64.0)
    assert fixed.theory(None) == {}


def test_configuration_json_and_seeds():
    """The configuration and the JSON of a realization read back; the seeds."""
    cfg = {
        "type": "long_fibers",
        "radius": {"type": "uniform", "low": 0.4, "high": 0.6},
        "volume_fraction": 0.55,
        "misalignment": 2.0,
        "waviness_length": 4.0,
        "transverse_correlation": 2.0,
        "min_gap": 0.02,
    }
    model = model_from_config(cfg)
    assert isinstance(model, LongFibers)
    assert model_from_config(model.to_config()) == model
    micro = model.generate(BOX, seed=14)
    back = Microstructure.from_dict(micro.to_dict())
    assert np.array_equal(back.segments.start, micro.segments.start)
    assert np.array_equal(back.segments.radii, micro.segments.radii)
    assert back.model == model
    again = model.generate(BOX, seed=14)
    assert np.array_equal(again.segments.start, micro.segments.start)
    other = model.generate(BOX, seed=15)
    assert not np.array_equal(other.segments.start[:10], micro.segments.start[:10])


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({}, "exactly one"),
        ({"volume_fraction": 0.85}, "at most 0.8"),
        ({"volume_fraction": 0.5, "number": 10}, "exactly one"),
        ({"number": 10, "arrangement": "hexagonal"}, "take a volume_fraction"),
        ({"volume_fraction": 0.5, "arrangement": "triangular"}, "arrangement"),
        ({"volume_fraction": 0.5, "axis": 3}, "axis"),
        ({"volume_fraction": 0.5, "disorder": 0.1}, "disorder"),
        ({"volume_fraction": 0.5, "arrangement": "square", "disorder": 1.0}, "disorder"),
        ({"volume_fraction": 0.5, "misalignment": 40.0}, "misalignment"),
        ({"volume_fraction": 0.5, "misalignment": -1.0}, "misalignment"),
        ({"volume_fraction": 0.5, "waviness_length": 0.0}, "waviness_length"),
        ({"volume_fraction": 0.5, "transverse_correlation": -1.0}, "transverse_correlation"),
        ({"volume_fraction": 0.5, "segments": 1}, "segments"),
        ({"volume_fraction": 0.5, "min_gap": -0.1}, "min_gap"),
        ({"volume_fraction": 0.5, "max_steps": 0}, "max_steps"),
    ],
)
def test_invalid_parameters(params, match):
    with pytest.raises(ValueError, match=match):
        LongFibers(radius=0.5, **params)


def test_boxes_too_small_and_fractions_out_of_reach():
    with pytest.raises(ValueError, match="twice the reach"):
        LongFibers(radius=0.5, volume_fraction=0.3).generate(mg.Domain(size=(1.9, 1.9, 5.0)))
    with pytest.raises(ValueError, match="number: 200 fibres"):
        LongFibers(radius=0.5, number=200).generate(BOX, seed=1)
    with pytest.raises(RuntimeError, match="raise max_steps"):
        LongFibers(radius=0.5, volume_fraction=0.79, max_steps=50).generate(BOX, seed=1)
