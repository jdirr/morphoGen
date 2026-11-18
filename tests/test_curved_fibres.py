"""Curved fibres: circular arcs (PuMA's curved fibres), helices (crimp), and chains that
do not overlap."""

from __future__ import annotations

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.distributions import Constant, Uniform
from morphogen.kernels.rsa import segment_distance2
from morphogen.orientations import Aligned

BOX = mg.Domain.cube(1.0, periodic=True)


def _axes(structure, m: int) -> np.ndarray:
    """Points (fibres x (m + 1) x 3) of the axes of a periodic structure."""
    seg = structure.segments
    start, end = seg.start.reshape(-1, m, 3), seg.end.reshape(-1, m, 3)
    return np.concatenate([start, end[:, -1:]], axis=1)


def _circumradius(p0, p1, p2) -> np.ndarray:
    a, b, c = p1 - p0, p2 - p1, p2 - p0
    area2 = np.linalg.norm(np.cross(a, b), axis=-1)
    return (
        np.linalg.norm(a, axis=-1)
        * np.linalg.norm(b, axis=-1)
        * np.linalg.norm(c, axis=-1)
        / (2 * area2)
    )


def test_arcs_have_their_radius_of_curvature():
    m, rho, length = 12, 0.15, 0.3
    model = mg.models.CurvedFibers(
        radius=0.01,
        length=length,
        segments=m,
        shape="arc",
        curvature_radius=Constant(rho),
        intensity=300,
    )
    pts = _axes(model.generate(BOX, seed=1), m)
    np.testing.assert_allclose(_circumradius(pts[:, :-2], pts[:, 1:-1], pts[:, 2:]), rho, rtol=1e-9)
    chords = np.linalg.norm(np.diff(pts, axis=1), axis=-1)
    np.testing.assert_allclose(chords, 2 * rho * math.sin(length / (2 * m * rho)), rtol=1e-9)
    ends = np.sum((pts[:, -1] - pts[:, 0]) ** 2, axis=-1)
    assert ends.mean() == pytest.approx(model.mean_square_end_to_end(), rel=1e-9)
    # distributed radii; aligned arcs: the middle tangents along z
    model = mg.models.CurvedFibers(
        radius=0.01,
        length=length,
        segments=m,
        shape="arc",
        curvature_radius=Uniform(0.1, 0.3),
        orientation=Aligned((0.0, 0.0, 1.0)),
        intensity=300,
    )
    pts = _axes(model.generate(BOX, seed=2), m)
    radii = _circumradius(pts[:, 0], pts[:, m // 2], pts[:, -1])
    assert radii.min() > 0.1 - 1e-9 and radii.max() < 0.3 + 1e-9
    middle = pts[:, m // 2 + 1] - pts[:, m // 2 - 1]
    np.testing.assert_allclose(middle[:, :2], 0.0, atol=1e-12)
    ends = np.sum((pts[:, -1] - pts[:, 0]) ** 2, axis=-1)
    assert ends.mean() == pytest.approx(model.mean_square_end_to_end(), rel=0.05)


def test_helices():
    m, a, pitch, length = 40, 0.02, 0.1, 0.4
    model = mg.models.CurvedFibers(
        radius=0.005,
        length=length,
        segments=m,
        shape="helix",
        helix_radius=a,
        helix_pitch=pitch,
        orientation=Aligned((0.0, 0.0, 1.0)),
        intensity=100,
    )
    pts = _axes(model.generate(BOX, seed=3), m)
    flat = pts.copy()
    flat[..., 2] = 0.0
    np.testing.assert_allclose(
        _circumradius(flat[:, :-2], flat[:, 1:-1], flat[:, 2:]), a, rtol=1e-9
    )
    c = pitch / (2 * math.pi)
    np.testing.assert_allclose(
        np.diff(pts[..., 2], axis=1), c * length / m / math.hypot(a, c), rtol=1e-9
    )
    ends = np.sum((pts[:, -1] - pts[:, 0]) ** 2, axis=-1)
    np.testing.assert_allclose(ends, model.mean_square_end_to_end(), rtol=1e-9)


@pytest.mark.parametrize("shape", ["arc", "worm_like"])
def test_curved_fibres_without_overlap(shape):
    m, r, gap = 8, 0.02, 0.005
    model = mg.models.CurvedFibers(
        radius=r,
        length=0.3,
        persistence=0.2,
        segments=m,
        shape=shape,
        curvature_radius=Constant(0.2),
        overlap=False,
        min_gap=gap,
        volume_fraction=0.08,
    )
    structure = model.generate(BOX, seed=4)
    pts = _axes(structure, m)
    n = len(pts)
    assert n == round(model.theory()["intensity"])
    centres = pts[:, m // 2]
    closest = np.inf
    for i in range(n):
        shift = centres - centres[i]
        shift -= np.round(shift)  # the nearest periodic images
        near = np.flatnonzero(np.linalg.norm(shift, axis=1) < 0.3 + 2 * r + gap)
        for j in near[near > i]:
            other = pts[j] - centres[j] + centres[i] + shift[j]
            for k in range(m):
                for kk in range(m):
                    d2 = segment_distance2(
                        pts[i, k], pts[i, k + 1] - pts[i, k], other[kk], other[kk + 1] - other[kk]
                    )
                    closest = min(closest, d2)
    assert math.sqrt(closest) >= 2 * r + gap - 1e-9
    img = structure.voxelize((128, 128, 128))
    assert mg.analysis.volume_fraction(img) == pytest.approx(0.08, rel=0.08)


def test_curved_fibres_without_overlap_in_a_box():
    model = mg.models.CurvedFibers(
        radius=0.02,
        length=0.2,
        segments=6,
        shape="arc",
        curvature_radius=Constant(0.15),
        overlap=False,
        volume_fraction=0.05,
    )
    img = model.generate(mg.Domain.cube(1.0, periodic=False), seed=5).voxelize((64, 64, 64))
    assert mg.analysis.volume_fraction(img) == pytest.approx(0.05, rel=0.25)
    with pytest.raises(ValueError, match="shape"):
        mg.models.CurvedFibers(shape="spiral", volume_fraction=0.1)


def test_hollow_fibres():
    from morphogen.models.base import Microstructure

    lam, r = 0.6, 0.04
    model = mg.models.HardcoreFibers(
        radius=Constant(r), length=Constant(0.3), volume_fraction=0.15, lumen=lam
    )
    walls = model.generate(BOX, seed=1)
    img = walls.voxelize((128, 128, 128))
    assert mg.analysis.volume_fraction(img) == pytest.approx(
        model.theory()["solid_fraction"], rel=0.03
    )
    # on the axis of a fibre: its lumen; halfway through its wall: solid
    seg = walls.parts[0].segments
    middle = 0.5 * (seg.start[0] + seg.end[0])
    axis = (seg.end[0] - seg.start[0]) / np.linalg.norm(seg.end[0] - seg.start[0])
    side = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    assert not walls.contains(middle[None])[0]
    assert walls.contains((middle + 0.5 * (1.0 + lam) * r * side)[None])[0]
    # stored and read back
    again = Microstructure.from_dict(walls.to_dict())
    np.testing.assert_array_equal(
        again.voxelize((32, 32, 32)).data, walls.voxelize((32, 32, 32)).data
    )
    curved = mg.models.CurvedFibers(
        radius=0.03,
        length=0.4,
        segments=8,
        shape="arc",
        curvature_radius=Constant(0.3),
        overlap=False,
        volume_fraction=0.08,
        lumen=0.5,
    )
    img = curved.generate(BOX, seed=2).voxelize((128, 128, 128))
    assert mg.analysis.volume_fraction(img) == pytest.approx(
        curved.theory()["solid_fraction"], rel=0.03
    )
    with pytest.raises(ValueError, match="lumen"):
        mg.models.CurvedFibers(volume_fraction=0.1, lumen=1.0)
