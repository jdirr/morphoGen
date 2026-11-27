"""The exact Minkowski functionals of unions of balls and of discs (morphology.balls): closed
forms of one and two balls, the topology of rings and shells, the derivative of the volume
of a union as its surface, Monte Carlo volumes, and the densities of Boolean models of
spheres against Miles' formulae, on the torus of periodic boxes."""

from __future__ import annotations

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.morphology import (
    minkowski_functionals,
    section_discs,
    section_functionals,
    union_of_balls,
    union_of_discs,
)


def _lens(r1, r2, d):
    """Volume of the intersection of two balls whose centres are ``d`` apart."""
    return (
        math.pi
        * (r1 + r2 - d) ** 2
        * (d * d + 2 * d * r2 - 3 * r2 * r2 + 2 * d * r1 + 6 * r1 * r2 - 3 * r1 * r1)
        / (12 * d)
    )


def test_one_and_two_balls_in_closed_form():
    one = union_of_balls([[0.1, 0.2, 0.3]], [1.5])
    assert one.volume == pytest.approx(4 / 3 * math.pi * 1.5**3, rel=1e-12)
    assert one.surface == pytest.approx(4 * math.pi * 1.5**2, rel=1e-12)
    assert one.mean_curvature == pytest.approx(4 * math.pi * 1.5, rel=1e-12)
    assert one.euler == 1
    r1, r2, d = 1.0, 0.7, 1.2
    two = union_of_balls([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], [r1, r2])
    x = (d * d + r1 * r1 - r2 * r2) / (2 * d)  # the plane of the circle from the first centre
    cap1 = 2 * math.pi * r1 * (r1 - x)
    cap2 = 2 * math.pi * r2 * (r2 - (d - x))
    a = math.sqrt(r1 * r1 - x * x)
    gamma = math.acos((r1 * r1 + r2 * r2 - d * d) / (2 * r1 * r2))
    assert two.volume == pytest.approx(4 / 3 * math.pi * (r1**3 + r2**3) - _lens(r1, r2, d))
    assert two.surface == pytest.approx(4 * math.pi * (r1**2 + r2**2) - cap1 - cap2)
    expected_m = (4 * math.pi * r1**2 - cap1) / r1 + (4 * math.pi * r2**2 - cap2) / r2
    assert two.mean_curvature == pytest.approx(expected_m - 0.5 * gamma * 2 * math.pi * a)
    assert two.euler == 1
    # apart: the sum of the two
    far = union_of_balls([[0.0, 0.0, 0.0], [5.0, 0.1, 0.0]], [r1, r2])
    assert far.euler == 2
    assert far.volume == pytest.approx(4 / 3 * math.pi * (r1**3 + r2**3))


def test_topology_of_rings_and_shells():
    ring = [[0, 0, 0], [1.9, 0.01, 0.02], [0.95, 1.65, -0.03]]
    assert union_of_balls(ring, [1.0] * 3).euler == 0  # a tunnel
    triple = [[0, 0, 0], [1.5, 0.01, 0.02], [0.75, 1.3, -0.03]]
    assert union_of_balls(triple, [1.0] * 3).euler == 1  # the triple overlap fills it
    # a shell of balls around a cavity: one component, one cavity
    rng = np.random.default_rng(5)
    k = np.arange(200) + 0.5
    phi, theta = np.arccos(1 - 2 * k / 200), math.pi * (1 + 5**0.5) * k
    pts = 4.0 * np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )
    pts += 0.01 * rng.standard_normal(pts.shape)
    assert union_of_balls(pts, 0.8).euler == 2


def test_volume_and_surface_of_random_unions():
    rng = np.random.default_rng(3)
    c = rng.random((40, 3)) * 6.0
    r = 0.5 + 0.3 * rng.random(40)
    u = union_of_balls(c, r)
    dt = 1e-5  # the union of the balls of radii R + t: dV/dt = S
    rate = (union_of_balls(c, r + dt).volume - union_of_balls(c, r - dt).volume) / (2 * dt)
    assert rate == pytest.approx(u.surface, rel=1e-6)
    pts = rng.random((400_000, 3)) * 8.0 - 1.0
    inside = np.zeros(len(pts), dtype=bool)
    for ci, ri in zip(c, r, strict=True):
        inside |= np.einsum("ij,ij->i", pts - ci, pts - ci) < ri * ri
    volume = inside.mean() * 8.0**3
    assert u.volume == pytest.approx(volume, rel=4 * math.sqrt(1 / inside.sum()))


def test_periodic_union_is_that_of_its_images():
    """The functionals on the torus: a ball across a face and corner counts once."""
    box = ((0.0, 0.0, 0.0), (5.0, 5.0, 5.0))
    u = union_of_balls([[0.2, 4.9, 2.5], [2.5, 2.5, 2.5]], [1.0, 0.7], box, periodic=True)
    assert u.volume == pytest.approx(4 / 3 * math.pi * (1.0 + 0.7**3), rel=1e-10)
    assert u.euler == 2
    assert u.volume_fraction == pytest.approx(u.volume / 125.0)
    # a slab across the box is a solid torus band: no component boundary, χ = 0
    xs = np.linspace(0.0, 5.0, 10, endpoint=False)
    slab = np.column_stack([xs, np.full(10, 2.5), np.full(10, 2.5)])
    assert union_of_balls(slab, 0.6, box, periodic=True).euler == 0


def test_discs_in_closed_form_and_holes():
    one = union_of_discs([[0.3, 0.1]], [1.2])
    assert one.area == pytest.approx(math.pi * 1.44) and one.perimeter == pytest.approx(
        2 * math.pi * 1.2
    )
    r1, r2, d = 1.0, 0.8, 1.1
    two = union_of_discs([[0.0, 0.0], [d, 0.0]], [r1, r2])
    x = (d * d + r1 * r1 - r2 * r2) / (2 * d)
    a1, a2 = math.acos(x / r1), math.acos((d - x) / r2)
    lens = r1 * r1 * (a1 - math.sin(2 * a1) / 2) + r2 * r2 * (a2 - math.sin(2 * a2) / 2)
    assert two.area == pytest.approx(math.pi * (r1**2 + r2**2) - lens)
    assert two.perimeter == pytest.approx(2 * math.pi * (r1 + r2) - 2 * a1 * r1 - 2 * a2 * r2)
    t = np.linspace(0.0, 2 * math.pi, 24, endpoint=False)
    rng = np.random.default_rng(1)
    ring = np.column_stack([5 * np.cos(t), 5 * np.sin(t)]) + 0.01 * rng.random((24, 2))
    assert union_of_discs(ring, 0.8).euler == 0  # one component around one hole
    # random discs: dA/dt = L and a Monte Carlo area
    c = rng.random((60, 2)) * 10.0
    r = 0.4 + 0.4 * rng.random(60)
    u = union_of_discs(c, r)
    dt = 1e-5
    rate = (union_of_discs(c, r + dt).area - union_of_discs(c, r - dt).area) / (2 * dt)
    assert rate == pytest.approx(u.perimeter, rel=1e-6)
    pts = rng.random((1_000_000, 2)) * 12.0 - 1.0
    inside = np.zeros(len(pts), dtype=bool)
    for ci, ri in zip(c, r, strict=True):
        inside |= np.einsum("ij,ij->i", pts - ci, pts - ci) < ri * ri
    assert u.area == pytest.approx(inside.mean() * 144.0, rel=4 / math.sqrt(inside.sum()))


@pytest.mark.slow
def test_boolean_spheres_against_miles():
    """The four densities of eight realizations on the torus within four standard errors of
    their mean from Miles' formulae (their own fluctuation alone: no bias of a grid); the
    sections give L_A = π S_V / 4 and χ_A = M_V / 2π."""
    model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.5)
    th = model.theory()
    values, sections = [], []
    for seed in range(8):
        micro = model.generate(mg.Domain.cube(16.0), seed=seed)
        values.append(minkowski_functionals(micro).densities())
        for axis in range(3):
            for position in (3.0, 11.0):
                sections.append(section_functionals(micro, axis, position).densities())
    for name in ("volume_fraction", "specific_surface", "mean_curvature_density", "euler_density"):
        v = np.array([d[name] for d in values])
        err = v.std(ddof=1) / math.sqrt(len(v))
        assert abs(v.mean() - th[name]) < 4 * err, name
    la = np.mean([d["boundary_density"] for d in sections])
    chi = np.array([d["euler_density"] for d in sections])
    surface = np.mean([d["specific_surface"] for d in values])
    assert la == pytest.approx(math.pi / 4 * surface, rel=0.03)
    assert abs(chi.mean() - th["mean_curvature_density"] / (2 * math.pi)) < 4 * chi.std(
        ddof=1
    ) / math.sqrt(len(chi))


def test_sections_of_spheres_are_discs():
    model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
    micro = model.generate(mg.Domain.cube(8.0), seed=2)
    centres, radii, _ = section_discs(micro, axis=2, position=4.0)
    # the discs agree with the membership of the points of the plane
    rng = np.random.default_rng(0)
    xy = rng.random((5000, 2)) * 8.0
    pts = np.column_stack([xy, np.full(len(xy), 4.0)])
    inside = np.zeros(len(xy), dtype=bool)
    for c, r in zip(centres, radii, strict=True):
        d = xy - c
        d -= 8.0 * np.round(d / 8.0)
        inside |= np.einsum("ij,ij->i", d, d) < r * r
    assert np.array_equal(inside, micro.contains(pts))
    assert section_functionals(micro, 2, 4.0).area_fraction == pytest.approx(
        np.mean(inside), abs=0.02
    )


def test_unions_of_other_objects_are_refused():
    fibres = mg.models.BooleanCylinders(radius=0.2, length=2.0, volume_fraction=0.1)
    micro = fibres.generate(mg.Domain.cube(6.0), seed=0)
    with pytest.raises(ValueError, match="union of spheres"):
        minkowski_functionals(micro)
