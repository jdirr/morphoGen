"""The exact contours of the sections of oblique and wavy fibres (``mesh.contours``): the
capsules that reach a plane clipped to its slab, grouped into bodies, the contours of each
traced on its own distance function, and their meshes against the areas of ellipses, the
gaps that the level sets close and the reciprocal theorem of Keller."""

from __future__ import annotations

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import Material, homogenize
from morphogen.mesh.conforming import mmg_executable
from morphogen.mesh.contours import (
    TRACE_CELLS,
    _crossing,
    _pair_corners,
    _segment_distances,
    _simple,
    _traced,
    _traced_at,
    contour_mesh,
    section_bodies,
    section_capsules,
    section_contours,
    section_layers,
)
from morphogen.mesh.triangles import disc_mesh, plane_mesh, section_mesh
from morphogen.models.long_fibers import LongFibers
from morphogen.models.nested import CoatedStructure, Coating
from morphogen.models.tessellation import Tessellation
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments, Spheres


def _has_mmg2d() -> bool:
    try:
        mmg_executable(None, "mmg2d")
    except (FileNotFoundError, RuntimeError, ImportError):
        return False
    return True


needs_mmg2d = pytest.mark.skipif(not _has_mmg2d(), reason="mmg2d (pymmg) is not installed")
BOX = mg.Domain.cube(1.0, periodic=True)


def _oblique(degrees: float, radius: float = 0.2, y: float = 0.1) -> PrimitiveUnion:
    """A straight fibre through the middle plane of the periodic unit cube, turned by
    ``degrees`` about y, its ends far from the plane: its section an ellipse of semi-axes
    ``radius`` (along y, across the side y = 0) and ``radius / cos``."""
    run = 0.4 * np.tan(np.radians(degrees))
    axes = Segments(np.array([[0.5 - run, y, 0.1]]), np.array([[0.5 + run, y, 0.9]]),
                    np.array([radius]))  # fmt: skip
    return PrimitiveUnion(BOX, None, axes)


def _pieces(mesh) -> int:
    """The connected parts of the fibres (phase 1) of a mesh, triangles that share a node
    connected."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    fibres = np.flatnonzero(mesh.phase == 1)
    rows = np.repeat(fibres, 3)
    incidence = coo_matrix((np.ones(len(rows)), (rows, mesh.elements[fibres, :3].ravel())),
                           shape=(len(mesh.elements), len(mesh.nodes))).tocsr()  # fmt: skip
    return connected_components((incidence @ incidence.T)[fibres][:, fibres])[0]


def _counted(micro, n: int) -> float:
    """The fraction of the box in the bodies of the section of ``micro``, counted at the
    centres of the cells of an ``n`` x ``n`` grid, body by body."""
    inside = np.zeros((n, n), dtype=bool)
    for body in section_bodies(micro):
        lo, hi = body.bounds
        i = np.arange(max(int(lo[0] * n), 0), min(int(np.ceil(hi[0] * n)) + 1, n))
        j = np.arange(max(int(lo[1] * n), 0), min(int(np.ceil(hi[1] * n)) + 1, n))
        if len(i) and len(j):
            gx, gy = np.meshgrid((i + 0.5) / n, (j + 0.5) / n, indexing="ij")
            d = body.distance(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            inside[np.ix_(i, j)] |= d < 0.0
    return float(inside.mean())


def _straight(xy: list, radii: list) -> PrimitiveUnion:
    """Straight fibres along z through a thin periodic box: their sections discs."""
    xy = np.asarray(xy, dtype=float)
    ends = [np.column_stack([xy, np.full(len(xy), z)]) for z in (0.0, 0.2)]
    domain = mg.Domain(size=(1.0, 1.0, 0.2), periodic=True)
    return PrimitiveUnion(domain, None, Segments(ends[0], ends[1], np.asarray(radii, dtype=float)))


def _lens(a: float, b: float, d: float) -> float:
    """The area of the lens of two discs of radii ``a`` and ``b`` whose centres are ``d``
    apart."""
    return (a * a * np.arccos((d * d + a * a - b * b) / (2.0 * d * a))
            + b * b * np.arccos((d * d + b * b - a * a) / (2.0 * d * b))
            - 0.5 * np.sqrt((a + b - d) * (d + a - b) * (d - a + b) * (d + a + b)))  # fmt: skip


def _phase_at(mesh, point) -> int:
    """The phase of the triangle of a mesh that holds a point."""
    p = mesh.nodes[mesh.elements[:, :3]]
    u, v, w = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], np.asarray(point) - p[:, 0]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    s = (w[:, 0] * v[:, 1] - w[:, 1] * v[:, 0]) / det
    t = (u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]) / det
    return int(mesh.phase[np.flatnonzero((s >= 0.0) & (t >= 0.0) & (s + t <= 1.0))[0]])


def test_distances_of_segments():
    """The distances between segments against the least distance between their points
    (convex in the parameters along them: minimized from the best of a sampling), points
    among them, parallel, collinear and skew."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(3)
    p1, q1, p2, q2 = (rng.normal(size=(100, 3)) for _ in range(4))
    q1[:10] = p1[:10]  # points
    q2[10:20] = p2[10:20] + (q1[10:20] - p1[10:20])  # parallel
    p2[20:25], q2[20:25] = 3.0 * q1[20:25] - 2.0 * p1[20:25], 4.0 * q1[20:25] - 3.0 * p1[20:25]
    exact = _segment_distances(p1, q1, p2, q2)
    s = np.linspace(0.0, 1.0, 41)
    a = p1[:, None] + s[None, :, None] * (q1 - p1)[:, None]
    b = p2[:, None] + s[None, :, None] * (q2 - p2)[:, None]
    sampled = np.linalg.norm(a[:, :, None] - b[:, None, :], axis=-1).reshape(100, -1)
    for k, flat in enumerate(sampled.argmin(axis=1)):

        def gap(st, k=k):
            return np.linalg.norm(p1[k] + st[0] * (q1[k] - p1[k]) - p2[k] - st[1] * (q2[k] - p2[k]))

        first = np.array([s[flat // 41], s[flat % 41]])
        least = minimize(gap, first, bounds=[(0.0, 1.0), (0.0, 1.0)], tol=1e-14).fun
        assert exact[k] <= least + 1e-12 and exact[k] == pytest.approx(least, abs=1e-7)


def test_capsules_and_bodies_of_a_section():
    """The axes that reach the plane clipped to its slab (a sphere an axis of no length),
    the periodic images along z taken; the capsules that meet one body, the images across
    the sides apart; flat caps in the slab, capsules that touch and other structures
    refused; the contours on the surfaces."""
    joint = [0.32, 0.3, 0.5]
    chain = Segments(np.array([[0.3, 0.3, 0.0], joint]), np.array([joint, [0.3, 0.3, 1.0]]),
                     np.array([0.1, 0.1]))  # fmt: skip
    ball = Spheres(np.array([[0.7, 0.7, 1.45]]), np.array([0.1]))  # its image reaches 0.5
    union = PrimitiveUnion(BOX, ball, chain)
    start, end, radii = section_capsules(union)
    assert len(radii) == 3 and np.all(np.abs(np.r_[start[:, 2], end[:, 2]] - 0.5) <= 0.1 + 1e-12)
    bodies = section_bodies(union, margin=0.0)
    assert sorted(len(b.radii) for b in bodies) == [1, 2]  # the chain one body
    assert len(section_bodies(union, margin=0.3)) == 8  # four images of each within 0.3
    for body in bodies:
        (loop,) = section_contours(body)
        assert np.abs(body.distance(loop)).max() < 1e-12
    flat = Segments(chain.start, chain.end, chain.radii, round_caps=False)
    assert section_capsules(PrimitiveUnion(BOX, None, flat)) is None  # (the joint at 0.5)
    feet = np.array([[0.3, 0.3, 0.0], [0.5, 0.3, 0.0]])
    touching = Segments(feet, feet + np.array([0.0, 0.0, 1.0]), np.array([0.1, 0.1]))
    assert section_bodies(PrimitiveUnion(BOX, None, touching)) is None
    grains = Tessellation(BOX, np.random.default_rng(1).random((8, 3)), np.zeros(8),
                          np.ones(8, dtype=np.uint8))  # fmt: skip
    assert section_capsules(grains) is None


@needs_mmg2d
@pytest.mark.parametrize("degrees", [0.0, 30.0, 50.0])
def test_contours_of_oblique_fibres(degrees):
    """The section of a straight fibre turned by 0, 30 and 50 degrees across a side of the
    periodic box, an ellipse of area ``pi r^2 / cos``: within 2e-5 at the size 0.05 and
    1e-6 at 0.025 on curved six-node triangles (1e-3 off at least with the chords), the
    sides matching."""
    micro = _oblique(degrees)
    exact = np.pi * 0.04 / np.cos(np.radians(degrees))
    for size, within in ((0.05, 2e-5), (0.025, 1e-6)):
        mesh = contour_mesh(micro, size=size, order=2)
        assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.metadata["bodies"] == 2
        assert mesh.matching_sides() and mesh.min_jacobian() > 0.3
        assert mesh.phase_fractions()[1] == pytest.approx(exact, rel=within)
    chords = contour_mesh(micro, size=0.025, order=2, curved=False).phase_fractions()[1]
    assert abs(chords / exact - 1.0) > 1e-3


@needs_mmg2d
def test_contours_keep_the_gaps_of_misaligned_fibres():
    """A ply of fibres misaligned by 3 degrees (chains of spherocylinders filling half of a
    periodic cube): the level sets of its section at the size 0.05 close the narrow gaps
    between its fibres — 7 parts of fibres instead of 74, 2.7e-2 too much fibre —, its exact
    contours keep every fibre apart, their area within 3e-5 of a count on a grid of 2000 x
    2000 points (1.2e-5), and the reciprocal theorem of Keller and Mendelson within 1e-3."""
    micro = LongFibers(radius=0.05, volume_fraction=0.5, misalignment=3.0).generate(BOX, seed=1)
    mesh = plane_mesh(micro, size=0.05, order=2)
    assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.matching_sides()
    assert _pieces(mesh) == mesh.metadata["bodies"] > 60
    counted = _counted(micro, 2000)
    assert mesh.phase_fractions()[1] == pytest.approx(counted, abs=3e-5)
    levels = section_mesh(micro, size=0.05, order=2)
    assert _pieces(levels) < 20 and levels.phase_fractions()[1] - counted > 2e-2
    quarter = np.array([[0.0, -1.0], [1.0, 0.0]])
    k = []
    for t in (10.0, 0.1):
        materials = {0: Material(young=1.0, poisson=0.3, conductivity=1.0),
                     1: Material(young=1.0, poisson=0.3, conductivity=t)}  # fmt: skip
        out = homogenize(mesh, materials, properties=["conductivity_tensor"])
        k.append(out["conductivity_tensor.periodic"]["tensor"][:2, :2])
    assert np.abs(k[0] @ quarter @ k[1] @ quarter.T - np.eye(2)).max() < 1e-3


@needs_mmg2d
def test_what_the_contours_leave_to_the_level_sets():
    """Flat caps in the plane (Boolean cylinders turned at random) are left to the level
    sets; a box not periodic, its sides free, keeps the contours, the part of the ellipse
    in the box within 2e-5."""
    flat = mg.models.BooleanCylinders(radius=0.05, length=0.3, volume_fraction=0.1)
    micro = flat.generate(BOX, seed=2)
    assert contour_mesh(micro, size=0.1) is None
    assert plane_mesh(micro, size=0.1).metadata["mesher"].startswith("section")
    free = plane_mesh(_oblique(30.0), size=0.05, order=2, periodic=False)
    assert free.metadata["mesher"] == "exact contours (mmg2d)" and not free.periodic
    a, b = 0.2 / np.cos(np.radians(30.0)), 0.2  # cut at half its semi-axis b
    exact = a * b * (np.pi / 2.0 + np.arcsin(0.5) + 0.5 * np.sqrt(0.75))
    assert free.phase_fractions()[1] == pytest.approx(exact, rel=2e-5)


@needs_mmg2d
def test_corners_of_capsules_that_cross():
    """Two fibres of radius 0.2 whose axes are 0.3 apart: their contour has its corners where
    the circles cross, by Newton's method on the two distances, to round-off, the angle of
    their normals arccos(-1/8); the area of their union within 1e-5, 1e-6 and 1e-7 at the
    sizes 0.05, 0.025 and 0.0125, the fourth power of the size, as the union of discs."""
    pair = _straight([[0.3, 0.5], [0.6, 0.5]], [0.2, 0.2])
    (body,) = section_bodies(pair)
    ((loop, corners, angles),) = _traced(body)
    crossing = np.array([[0.45, 0.5 - np.sqrt(0.0175)], [0.45, 0.5 + np.sqrt(0.0175)]])
    found = loop[corners][np.argsort(loop[corners][:, 1])]
    assert len(corners) == 2 and np.abs(found - crossing).max() < 1e-12
    assert np.allclose(angles, np.degrees(np.arccos(-0.125)))
    exact = 2.0 * np.pi * 0.04 - _lens(0.2, 0.2, 0.3)
    for size, within in ((0.05, 1e-5), (0.025, 1e-6), (0.0125, 1e-7)):
        mesh = contour_mesh(pair, size=size, order=2)
        assert mesh.phase_fractions()[1] == pytest.approx(exact, rel=within)


@needs_mmg2d
def test_a_narrow_gap_within_a_body():
    """Two discs of radius 0.1 that a third joins, their circles 0.002 apart: on the grid of
    an eighth of the least radius the contours of their body bridge the gap, on one four
    times as fine they follow it, and the mesh takes them — matrix in the middle of the
    gap, the area of the union within 1e-5 at the size 0.05 and 1e-6 at 0.0125, as the
    union of discs."""
    micro = _straight([[0.4, 0.5], [0.602, 0.5], [0.5, 0.42]], [0.1, 0.1, 0.06])
    (body,) = section_bodies(micro)
    h = 0.06 / TRACE_CELLS
    assert _traced_at(body, h) is None and _traced_at(body, h / 4.0) is not None
    exact = (np.pi * 0.0236 - _lens(0.1, 0.06, np.hypot(0.1, 0.08))
             - _lens(0.1, 0.06, np.hypot(0.102, 0.08)))  # fmt: skip
    for size, within in ((0.05, 1e-5), (0.0125, 1e-6)):
        mesh = contour_mesh(micro, size=size, order=2)
        assert _phase_at(mesh, (0.501, 0.5)) == 0 and mesh.min_jacobian() > 0.3
        assert mesh.phase_fractions()[1] == pytest.approx(exact, rel=within)


def test_contours_checked_before_meshing():
    """Two edges that cross are found, two that share a node do not cross; a bow tie is
    not a simple contour, a square is."""
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    assert _crossing(points, np.array([[0, 1], [2, 3]]))
    assert not _crossing(points, np.array([[0, 1], [1, 2], [0, 2]]))
    assert _simple(points[[0, 3, 1, 2]]) and not _simple(points)


@needs_mmg2d
def test_boolean_networks_of_spherocylinders():
    """Spherocylinders turned at random (a Boolean model, radius 0.03, length 0.4), their
    sections bodies of several capsules that cross: filling a tenth of the box, meshed by
    their contours within 2e-5 of a count on a grid of 3000 x 3000 points; filling a
    fourth, their crossings (holes and clusters of corners finer than the grids) leave this
    realization to the level sets."""
    sparse = mg.models.BooleanCylinders(radius=0.03, length=0.4, volume_fraction=0.1,
                                        caps="round").generate(BOX, seed=1)  # fmt: skip
    mesh = plane_mesh(sparse, size=0.05, order=2)
    assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.matching_sides()
    assert mesh.min_jacobian() > 0.3
    assert mesh.phase_fractions()[1] == pytest.approx(_counted(sparse, 3000), abs=2e-5)
    dense = mg.models.BooleanCylinders(radius=0.03, length=0.4, volume_fraction=0.25,
                                       caps="round").generate(BOX, seed=1)  # fmt: skip
    assert contour_mesh(dense, size=0.05) is None
    assert plane_mesh(dense, size=0.05).metadata["mesher"].startswith("section")


@needs_mmg2d
def test_contours_of_coated_and_hollow_fibres():
    """The layers of a section: a coated oblique fibre, its core and the ring of its coating
    within 1e-6 of their ellipses at the size 0.05; hollow misaligned fibres, their shells
    about bores of a third phase, within 3e-5 of counts on a grid; two coated discs whose
    coatings cross, which the rings leave, their cores and the union of their coatings
    within 2e-6."""
    coated = CoatedStructure(BOX, _oblique(30.0, radius=0.15), 0.05, "outside", 2, 0)
    unions, bands = section_layers(coated)
    assert len(unions) == 2 and bands == [1, 2, 0] and section_layers(object()) is None
    mesh = plane_mesh(coated, size=0.05, order=2)
    assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.metadata["layers"] == 2
    cos = np.cos(np.radians(30.0))
    fractions = mesh.phase_fractions()
    assert fractions[1] == pytest.approx(np.pi * 0.0225 / cos, rel=1e-6)
    assert fractions[2] == pytest.approx(np.pi * 0.0175 / cos, rel=1e-6)
    fibres = LongFibers(radius=0.05, volume_fraction=0.4, misalignment=3.0)
    hollow = Coating(model=fibres, thickness=0.015, side="inside", core_phase=3).generate(
        BOX, seed=1
    )
    (bores, members), bands = section_layers(hollow)
    assert bands == [3, 1, 0]
    mesh = plane_mesh(hollow, size=0.05, order=2)
    assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.matching_sides()
    fractions, inner = mesh.phase_fractions(), _counted(bores, 2000)
    assert fractions[3] == pytest.approx(inner, abs=3e-5)
    assert fractions[1] == pytest.approx(_counted(members, 2000) - inner, abs=3e-5)
    pair = _straight([[0.35, 0.5], [0.6, 0.5]], [0.1, 0.1])
    meeting = CoatedStructure(pair.domain, pair, 0.04, "outside", 2, 0)
    assert disc_mesh(meeting, size=0.05) is None
    mesh = plane_mesh(meeting, size=0.05, order=2)
    assert mesh.metadata["mesher"] == "exact contours (mmg2d)"
    fractions = mesh.phase_fractions()
    outer = 2.0 * np.pi * 0.14**2 - _lens(0.14, 0.14, 0.25)
    assert fractions[1] == pytest.approx(2.0 * np.pi * 0.01, rel=2e-6)
    assert fractions[2] == pytest.approx(outer - 2.0 * np.pi * 0.01, rel=2e-6)


@needs_mmg2d
def test_every_corner_of_a_body_traced():
    """Every corner of a body — two of its capsules crossing, no other covering the point —,
    found on its own along polygons of the sections of its capsules (Newton's method on the
    two distances), lies on its contours, or the body is traced again: two crossing discs
    have their two; coated fibres whose coatings meet across the box (a cluster that
    percolates), with corners at the tips of narrow wedges of matrix that the grid cuts,
    their cores and coatings within 3e-5 of counts on a grid at the size 0.05, and within
    2e-6 of themselves at 0.05 and 0.0125."""
    pair = _straight([[0.3, 0.5], [0.6, 0.5]], [0.2, 0.2])
    (body,) = section_bodies(pair)
    m = len(body.radii)
    found = [x for a in range(m) for b in range(a + 1, m) for x in _pair_corners(body, a, b, 1e-10)]
    crossing = np.unique(np.round(np.array(found), 9), axis=0)
    assert (
        np.abs(crossing - [[0.45, 0.5 - np.sqrt(0.0175)], [0.45, 0.5 + np.sqrt(0.0175)]]).max()
        < 1e-9
    )
    domain = mg.Domain(size=(1.0, 1.0, 0.25), periodic=True)
    close = LongFibers(radius=0.1, number=10, arrangement="random", min_gap=0.02)
    meeting = Coating(model=close, thickness=0.03).generate(domain, seed=2)
    (cores, coats), _ = section_layers(meeting)
    fractions = []
    for size in (0.05, 0.0125):
        mesh = plane_mesh(meeting, size=size, order=2)
        assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.matching_sides()
        fractions.append(mesh.phase_fractions())
    core, coat = _counted(cores, 2000), _counted(coats, 2000)
    assert fractions[0][1] == pytest.approx(core, abs=3e-5)
    assert fractions[0][2] == pytest.approx(coat - core, abs=3e-5)
    assert abs(fractions[1][1] - fractions[0][1]) < 2e-6
    assert abs(fractions[1][2] - fractions[0][2]) < 2e-6
