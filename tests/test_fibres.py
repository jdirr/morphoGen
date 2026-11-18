"""Hard-core fibres (random sequential addition of spherocylinders) and curved fibres
(Boolean model of worm-like chains): exclusion, fractions, orientations, tangent
correlations and end-to-end distances against their theory."""

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.kernels.rsa import segment_distance2
from morphogen.models import CurvedFibers, HardcoreFibers


def _min_distance(segments, box) -> float:
    """Smallest distance between the axes of distinct segments, nearest periodic images
    (brute force)."""
    a, b = segments.start, segments.end
    centres, d = 0.5 * (a + b), b - a
    best = math.inf
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            shift = centres[j] - centres[i]
            shift -= box * np.round(shift / box)
            p2 = centres[i] + shift - 0.5 * d[j]
            best = min(best, segment_distance2(a[i], d[i], p2, d[j]))
    return math.sqrt(best)


def test_segment_distance():
    """Closest points of segments: crossing, parallel, collinear and degenerate."""
    zero = np.zeros(3)
    ex, ey = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert segment_distance2(zero, ex, np.array([0.5, -0.5, 2.0]), ey) == pytest.approx(4.0)
    assert segment_distance2(zero, ex, np.array([0.2, 3.0, 0.0]), ex) == pytest.approx(9.0)
    assert segment_distance2(zero, ex, np.array([3.0, 0, 0]), ex) == pytest.approx(4.0)
    assert segment_distance2(zero, zero, np.array([0, 0, 2.0]), zero) == pytest.approx(4.0)
    rng = np.random.default_rng(0)
    for _ in range(200):  # against a fine sampling of both segments
        p1, d1, p2, d2 = rng.normal(size=(4, 3))
        s = np.linspace(0, 1, 401)
        x = p1 + s[:, None] * d1
        y = p2 + s[:, None] * d2
        sampled = ((x[:, None] - y[None]) ** 2).sum(-1).min()
        exact = segment_distance2(p1, d1, p2, d2)
        assert exact <= sampled + 1e-12 and sampled - exact < 1e-4 * (1 + sampled)


def test_hardcore_fibres():
    """No two fibres closer than the gap (periodic images included), the target
    fraction in the box, aligned axes when asked."""
    domain = mg.Domain.cube(10.0, periodic=True)
    model = HardcoreFibers(radius=0.3, length=3.0, volume_fraction=0.12, min_gap=0.05)
    micro = model.generate(domain, seed=4)
    seg = micro.segments
    assert _min_distance(seg, domain.lengths) >= 2 * 0.3 + 0.05 - 1e-9
    one = model.mean_volume / domain.volume
    assert abs(seg.volumes().sum() / domain.volume - 0.12) <= one
    measured = micro.voxelize(shape=100, supersampling=2).data.mean()
    assert measured == pytest.approx(0.12, rel=0.03)
    assert model.theory()["volume_fraction"] == 0.12
    aligned = HardcoreFibers(radius=0.3, length=3.0, orientation="aligned", volume_fraction=0.3)
    seg = aligned.generate(domain, seed=1).segments
    axes = (seg.end - seg.start) / seg.lengths[:, None]
    assert np.allclose(np.abs(axes[:, 2]), 1.0)  # along the default axis, beyond isotropic RSA
    window = HardcoreFibers(radius=0.3, length=3.0, volume_fraction=0.1)
    micro = window.generate(mg.Domain.cube(8.0, periodic=False), seed=2)
    assert micro.voxelize(shape=64, supersampling=2).data.mean() == pytest.approx(0.1, abs=0.02)
    with pytest.raises(ValueError, match="shorter than half"):
        HardcoreFibers(radius=0.3, length=6.0, volume_fraction=0.1).generate(domain, seed=0)
    with pytest.raises(RuntimeError, match="jammed"):
        HardcoreFibers(radius=0.5, length=3.5, volume_fraction=0.5, max_attempts=2000).generate(
            domain, seed=0
        )


def test_curved_fibres():
    """Worm-like chains: tangent correlations exp(-s / persistence), the mean square
    end-to-end distance of the discrete chain, and the fraction of the Boolean model
    (exact for straight fibres, a little lower for curved ones)."""
    model = CurvedFibers(radius=0.2, length=6.0, persistence=3.0, segments=12, intensity=0.05)
    domain = mg.Domain.cube(30.0, periodic=True)
    seg = model.generate(domain, seed=7).segments
    m = model.segments
    n = len(seg) // m
    assert n > 1000 and len(seg) == n * m
    t = (seg.end - seg.start).reshape(n, m, 3)
    t /= np.linalg.norm(t, axis=2)[..., None]
    c = model.correlation
    for lag in (1, 3, 6):
        dots = np.sum(t[:, :-lag] * t[:, lag:], axis=2).ravel()
        assert dots.mean() == pytest.approx(c**lag, abs=4 * dots.std() / math.sqrt(n))
    ends = seg.end.reshape(n, m, 3)[:, -1] - seg.start.reshape(n, m, 3)[:, 0]
    r2 = np.sum(ends**2, axis=1)
    expected = model.mean_square_end_to_end()
    assert r2.mean() == pytest.approx(expected, abs=4 * r2.std() / math.sqrt(n))
    assert expected < model.length**2  # tortuous
    # the fraction of a 12^3 box fluctuates by about 0.011 (integral range): six of them
    small = mg.Domain.cube(12.0, periodic=True)
    for persistence, segments in ((1e6, 4), (1.0, 8)):  # straight, then tortuous fibres
        fibres = CurvedFibers(radius=0.3, length=4.0, persistence=persistence,
                              segments=segments, volume_fraction=0.15)  # fmt: skip
        fractions = [
            fibres.generate(small, seed=s).voxelize(shape=72, supersampling=2).data.mean()
            for s in range(6)
        ]
        assert np.mean(fractions) == pytest.approx(0.15, abs=0.014)
    window = fibres.generate(mg.Domain.cube(8.0, periodic=False), seed=1)
    assert len(window.segments) > 0
