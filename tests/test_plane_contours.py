"""The exact contours of the sections of oblique and wavy fibres (``mesh.contours``): the
capsules that reach a plane clipped to its slab, grouped into bodies, the contours of each
traced on its own distance function, and their meshes against the areas of ellipses, the
gaps that the level sets close and the reciprocal theorem of Keller."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import Material, homogenize
from morphogen.mesh.conforming import mmg_executable
from morphogen.mesh.contours import (
    TRACE_CELLS,
    _corners,
    _crossing,
    _grid_distances,
    _joined,
    _layer_offsets,
    _line_crossings,
    _line_root,
    _marching_loops,
    _pair_corners,
    _segment_distances,
    _shared,
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


def _counted_bodies(bodies, n: int) -> float:
    """The fraction of the box in bodies (:func:`_counted`), given."""
    inside = np.zeros((n, n), dtype=bool)
    for body in bodies:
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
    """A structure that is no union of capsules (a gyroid) is left to the level sets; a
    box not periodic, its sides free, keeps the contours, the part of the ellipse in the
    box within 2e-5."""
    gyroid = mg.models.TPMS(surface="gyroid", volume_fraction=0.3).generate(BOX, seed=1)
    assert contour_mesh(gyroid, size=0.1) is None
    assert plane_mesh(gyroid, size=0.1).metadata["mesher"].startswith("section")
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
    fourth too, their holes, clusters of corners and wedges finer than the grids traced."""
    sparse = mg.models.BooleanCylinders(radius=0.03, length=0.4, volume_fraction=0.1,
                                        caps="round").generate(BOX, seed=1)  # fmt: skip
    mesh = plane_mesh(sparse, size=0.05, order=2)
    assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.matching_sides()
    assert mesh.min_jacobian() > 0.3
    assert mesh.phase_fractions()[1] == pytest.approx(_counted(sparse, 3000), abs=2e-5)
    dense = mg.models.BooleanCylinders(radius=0.03, length=0.4, volume_fraction=0.25,
                                       caps="round").generate(BOX, seed=1)  # fmt: skip
    mesh = plane_mesh(dense, size=0.05, order=2)
    assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.matching_sides()
    assert mesh.min_jacobian() > 0.3
    assert mesh.phase_fractions()[1] == pytest.approx(_counted(dense, 3000), abs=2e-5)


def test_what_the_grid_cannot_see_of_dense_networks():
    """The features of the sections of dense Boolean networks narrower than the cells of
    the grids: a wedge of matrix about a node of the grid, which the marching squares close
    into a loop of its own inside the contour that passes around the wedge to its corner —
    the loop left out, no two contours of a body crossing (a realization filling a
    fourth); three surfaces that meet nearly at a point, the contour from the first to the
    second along the third for less than a cell — the two corners with the third (three
    discs); and the finer grids of the bridges computed in bands about the surface, their
    contours those of the whole grids."""
    dense = mg.models.BooleanCylinders(radius=0.03, length=0.4, volume_fraction=0.25,
                                       caps="round").generate(BOX, seed=1)  # fmt: skip
    for body in section_bodies(dense):
        loops = section_contours(body)
        ends = np.cumsum([0] + [len(loop) for loop in loops])
        edges = np.concatenate([np.column_stack([np.arange(a, b), np.roll(np.arange(a, b), -1)])
                                for a, b in itertools.pairwise(ends)])  # fmt: skip
        assert not _crossing(np.concatenate(loops), edges)
    # the corner of the first two discs inside the third by 1e-8, the contour along its arc
    r = 0.1
    corner = np.array([0.475, 0.5 + np.sqrt(r**2 - 0.075**2)])
    centres = [[0.4, 0.5], [0.55, 0.5], [corner[0], corner[1] - r + 1e-8]]
    (body,) = section_bodies(_straight(centres, [r, r, r]))
    found = _corners(body, 0, 1, corner[None], 1e-12)
    assert len(found) == 2 and all(np.linalg.norm(x - corner) < 2e-8 for x, _ in found)
    assert found[0][0][0] < corner[0] < found[1][0][0]  # (from the first disc to the third)
    (loop,) = section_contours(body)
    assert all(np.linalg.norm(loop - x, axis=1).min() < 1e-12 for x, _ in found)
    lo, hi = body.bounds
    grids = []
    for h in (r / TRACE_CELLS, r / TRACE_CELLS / 4, r / TRACE_CELLS / 16):
        shape = tuple(int(np.ceil((hi - lo)[k] / h + 4)) + 1 for k in (0, 1))
        grids.append(_grid_distances(body, lo - 2.0 * h, h, shape, grids[-1] if grids else None))
    whole = _grid_distances(body, grids[-1].lower, h, grids[-1].shape)
    assert len(grids[-1].keys) < 0.25 * len(whole.keys)  # (the nodes of a band about the surface)
    traced, full = _marching_loops(grids[-1]), _marching_loops(whole)
    assert len(traced) == len(full)
    assert all(np.array_equal(a, b) for a, b in zip(traced, full, strict=True))


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
def test_contours_of_hollow_hardcore_fibres():
    """Hollow fibres, the difference of their union and of that of their lumens: the lumens
    a layer within the fibres, of the phase of the matrix; the walls of a dense packing (a
    fourth of the box, the lumens six tenths of the radii) within 1e-5 of counts, though a
    fibre of it grazes the side y = 0 (its dip across the line shallower than the bulge of
    the chords of its trace: :func:`test_contours_that_graze_a_side`)."""
    from morphogen.models.base import model_from_config

    model = model_from_config({"type": "hardcore_fibers", "radius": 0.04, "length": 0.3,
                               "volume_fraction": 0.25, "lumen": 0.6})  # fmt: skip
    micro = model.generate(BOX, seed=1)
    unions, bands = section_layers(micro)
    assert bands == [0, 1, 0] and unions == [micro.parts[1], micro.parts[0]]
    assert _layer_offsets(micro) == [0.0, 0.0]
    mesh = plane_mesh(micro, size=0.05, order=2)
    assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.matching_sides()
    assert mesh.metadata["layers"] == 2
    walls = _counted(micro.parts[0], 4000) - _counted(micro.parts[1], 4000)
    assert mesh.phase_fractions()[1] == pytest.approx(walls, abs=1e-5)


@needs_mmg2d
def test_contours_of_curved_fibres():
    """Curved fibres, worm-like chains of short segments (a tenth of their length, a radius
    and a third) that bend by some 16 degrees at each joint: the round end of a capsule and
    the side of the next tangent where a chain bends, no corner (the corners of the
    contours checked against those of the pairs of capsules made it one, and left the
    chains to the level sets); the angles of the corners those of the normals of the
    contours in the plane. Solid and hollow, apart or overlapping, and random walks of
    fibres in contact, within 1e-5 of counts."""
    from morphogen.models.base import model_from_config

    walks = {"type": "random_walk_fibers", "radius": 0.03, "volume_fraction": 0.15}
    for lumen, overlap in ((0.0, False), (0.5, False), (0.5, True), (0.5, None)):
        if overlap is None:  # (random walks, whose fibres touch: away from the plane)
            model = model_from_config({**walks, "lumen": lumen})
        else:
            model = model_from_config({"type": "curved_fibers", "radius": 0.03, "length": 0.4,
                                       "persistence": 0.5, "volume_fraction": 0.1,
                                       "lumen": lumen, "overlap": overlap})  # fmt: skip
        micro = model.generate(BOX, seed=1)
        mesh = plane_mesh(micro, size=0.05, order=2)
        assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.matching_sides()
        if lumen:
            fibres, lumens = micro.parts
            counted = _counted(fibres, 4000) - _counted(lumens, 4000)
        else:
            counted = _counted(micro, 4000)
        assert mesh.phase_fractions()[1] == pytest.approx(counted, abs=1e-5)
    for body in section_bodies(micro.parts[0]):  # (the fibres)
        for loop, corners, angles in _traced(body):
            for k, angle in zip(corners, angles, strict=True):
                a, b = np.argsort([abs(body.capsule(c, loop[k][None])[0][0])
                                   for c in range(len(body.radii))])[:2]  # fmt: skip
                ga, gb = body.capsule(a, loop[k][None])[1][0], body.capsule(b, loop[k][None])[1][0]
                cosine = ga @ gb / (np.linalg.norm(ga) * np.linalg.norm(gb))
                assert angle == pytest.approx(np.degrees(np.arccos(cosine)), abs=1e-6)


@needs_mmg2d
def test_capsules_that_touch_away_from_the_plane():
    """Two oblique capsules that touch at a point 0.02 below the plane, within its slab:
    their sections apart (a gap of 0.0029), two bodies meshed by their contours, two
    ellipses (the capsules that touched within the slab left the section to the level
    sets); in the plane of the point, their sections touch, left to the level sets."""
    radius, c = 0.05, 0.8  # (the cosine of the capsules to z)
    centres = np.array([[0.5, 0.5, 0.48], [0.5, 0.6, 0.48]])  # (2 radii apart at z = 0.48)
    directions = np.array([[0.6, 0.0, c], [-0.6, 0.0, c]])
    axes = Segments(centres - 0.45 * directions, centres + 0.5 * directions,
                    np.full(2, radius))  # fmt: skip
    touching = PrimitiveUnion(BOX, None, axes)
    assert len(section_bodies(touching)) == 2
    mesh = contour_mesh(touching, size=0.05, order=2)
    assert mesh is not None and mesh.metadata["bodies"] == 2
    assert mesh.phase_fractions()[1] == pytest.approx(2.0 * np.pi * radius**2 / c, rel=2e-5)
    assert section_bodies(touching, 0.48) is None
    assert contour_mesh(touching, size=0.05, z=0.48) is None


@needs_mmg2d
def test_contours_that_graze_a_side():
    """A contour that dips across the line of a side by less than the bulge of the chords
    of its trace: the chords miss the line; the point of the surface farthest across it
    between them gives its two crossings, and the piece of the contour in the box and the
    lens of its image across the side are meshed (the nodes of the pieces fell out of the
    box, and the mesh to the level sets); the edges graded at the crossings by their angle,
    so that the curved elements of the thin wedges keep their bulge (straightened, they
    lost ten times as much of the ellipse): within 2e-5 of it."""
    depth = 1e-5
    micro = _oblique(10.0, radius=0.2, y=0.2 - depth)
    for body in section_bodies(micro, margin=0.15):
        for loop, *_ in _traced(body):
            crossings = _line_crossings(body, loop, np.zeros(2), np.ones(2), 1e-9)
            assert not [c for c in crossings if c[2] == 1]  # (the lines y = 0 and 1)
    mesh = contour_mesh(micro, size=0.05, order=2)
    assert mesh is not None and mesh.metadata["mesher"] == "exact contours (mmg2d)"
    assert mesh.matching_sides() and np.all((mesh.nodes >= 0.0) & (mesh.nodes <= 1.0))
    exact = np.pi * 0.04 / np.cos(np.radians(10.0))
    assert mesh.phase_fractions()[1] == pytest.approx(exact, rel=2e-5)
    assert _phase_at(mesh, [0.5, 1.0 - 0.5 * depth]) == 1  # (the lens of the image)
    assert _phase_at(mesh, [0.5, 1.0 - 2.0 * depth]) == 0
    assert _phase_at(mesh, [0.5, 0.5 * depth]) == 1
    # a dip that the chords cross, one vertex in it: each crossing the root between the ends
    # of its own edge, where the root nearest to the crossing of its chord was the other's
    micro = _oblique(20.0, radius=0.2, y=0.2 - 2.0 * depth)
    body = section_bodies(micro, margin=0.15)[0]
    ((loop, *_),) = _traced(body)
    found = _line_crossings(body, loop, np.zeros(2), np.ones(2), 1e-9)
    centre = (0.2 - 2.0 * depth) / 0.2  # (of the ellipse, over its semi-axis across y = 0)
    half = 0.2 / np.cos(np.radians(20.0)) * np.sqrt(1.0 - centre**2)
    assert sorted(c[4] for c in found if c[2] == 1) == pytest.approx([0.5 - half, 0.5 + half])
    mesh = contour_mesh(micro, size=0.05, order=2)
    exact = np.pi * 0.04 / np.cos(np.radians(20.0))
    assert mesh.phase_fractions()[1] == pytest.approx(exact, rel=2e-5)


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


@needs_mmg2d
@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2])
def test_sections_of_many_realizations(seed):
    """The sections of misaligned plies, sparse and dense, coated, hollow, of Boolean
    networks of spherocylinders and of cylinders with flat caps (dense: the rims of the
    caps between the corners of two capsules and their vertices, the contours their corners
    knot traced again) and of Boolean spheres, at two sizes: every one meshed —
    by its exact contours or discs, or else by the level sets —, its six-node elements
    valid and its periodic sides matching."""
    structures = [
        LongFibers(radius=0.05, volume_fraction=0.3, misalignment=2.0).generate(BOX, seed=seed),
        LongFibers(radius=0.05, volume_fraction=0.55, misalignment=4.0).generate(BOX, seed=seed),
        Coating(model=LongFibers(radius=0.05, volume_fraction=0.35, misalignment=3.0,
                                 min_gap=0.02), thickness=0.01).generate(BOX, seed=seed),
        Coating(model=LongFibers(radius=0.05, volume_fraction=0.4, misalignment=3.0),
                thickness=0.015, side="inside", core_phase=2).generate(BOX, seed=seed),
        mg.models.BooleanCylinders(radius=0.04, length=0.3, volume_fraction=0.15,
                                   caps="round").generate(BOX, seed=seed),
        mg.models.BooleanCylinders(radius=0.03, length=0.4,
                                   volume_fraction=0.25).generate(BOX, seed=seed),
        mg.models.BooleanSpheres(radius=0.08, volume_fraction=0.3).generate(BOX, seed=seed),
    ]  # fmt: skip
    meshers = set()
    for micro in structures:
        for size in (0.05, 0.025):
            mesh = plane_mesh(micro, size=size, order=2)
            assert mesh.min_jacobian() > 0.25 and mesh.matching_sides()
            meshers.add(mesh.metadata["mesher"])
    assert "exact contours (mmg2d)" in meshers and "exact discs (mmg2d)" in meshers


def test_owners_that_change_without_a_corner():
    """Along a contour, the surface may pass from a capsule to another without a corner only
    where the two share an end (a joint of a chain) or coincide (both vertices on both
    surfaces): elsewhere a corner that the grid did not resolve."""
    joint = [0.32, 0.3, 0.5]
    chain = Segments(np.array([[0.3, 0.3, 0.0], joint]), np.array([joint, [0.3, 0.3, 1.0]]),
                     np.array([0.1, 0.1]))  # fmt: skip
    (body,) = section_bodies(PrimitiveUnion(BOX, None, chain))
    assert _joined(body, 0, 1)
    pair = _straight([[0.3, 0.5], [0.6, 0.5]], [0.2, 0.2])
    (body,) = section_bodies(pair)
    low = np.minimum(body.start[:, 2], body.end[:, 2])
    high = np.maximum(body.start[:, 2], body.end[:, 2])
    across = np.flatnonzero((low <= 0.1) & (high >= 0.1)).tolist()  # (axes through the plane)
    a = next(k for k in across if np.allclose(body.start[k, :2], [0.3, 0.5]))
    b = next(k for k in across if np.allclose(body.start[k, :2], [0.6, 0.5]))
    on_both = np.array([[0.45, 0.5 + np.sqrt(0.0175)]])  # a corner of the two circles
    on_one = np.array([[0.1, 0.5]])  # on the first circle only
    assert not _joined(body, a, b) and _shared(body, a, b, on_both, 0.01)
    assert not _shared(body, a, b, on_one, 0.01)


def test_contours_of_flat_caps():
    """Cylinders whose flat caps cut the plane: an oblique one (40 degrees) whose flat end
    cuts its ellipse along a line, within 2e-6 of the area of the segment of the ellipse at
    the size 0.025, the two corners of its rim nodes of the mesh; Boolean cylinders turned
    at random (flat caps, filling a tenth of the box), meshed by their contours within
    1e-5 of a count on a grid of 3000 x 3000 points (the straight edges of the caps)."""
    theta, r = np.radians(40.0), 0.15
    axis = np.array([np.sin(theta), 0.0, np.cos(theta)])
    cross = np.array([0.45, 0.5, 0.5])  # where the axis crosses the plane
    end = cross + 0.1 * axis  # the flat end
    segments = Segments((cross - 0.6 * axis)[None], end[None], np.array([r]), round_caps=False)
    micro = PrimitiveUnion(mg.Domain(size=(1.0, 1.0, 1.0)), None, segments)
    (body,) = section_bodies(micro)
    assert body.flat.tolist() == [[False, True]]
    a, b = r / np.cos(theta), r  # the ellipse of the infinite cylinder, cut at x = cut
    cut = end[0] - (0.5 - end[2]) / np.tan(theta)
    u = (cut - cross[0]) / a
    exact = np.pi * a * b - a * b * (np.arccos(u) - u * np.sqrt(1.0 - u * u))
    mesh = contour_mesh(micro, size=0.025, order=2)
    assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.min_jacobian() > 0.3
    assert mesh.phase_fractions()[1] == pytest.approx(exact, rel=2e-6)
    for sign in (1.0, -1.0):
        rim = np.array([cut, cross[1] + sign * b * np.sqrt(1.0 - u * u)])
        assert np.linalg.norm(mesh.nodes[:, :2] - rim, axis=1).min() < 1e-12
    flat = mg.models.BooleanCylinders(radius=0.03, length=0.4, volume_fraction=0.1)
    network = flat.generate(BOX, seed=1)
    assert any(body.capped for body in section_bodies(network))
    mesh = plane_mesh(network, size=0.05, order=2)
    assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.matching_sides()
    assert mesh.min_jacobian() > 0.3
    assert mesh.phase_fractions()[1] == pytest.approx(_counted(network, 3000), abs=1e-5)


def test_contours_of_coated_and_hollow_flat_caps():
    """The coatings and the shells of cylinders with flat caps: a coating is the union grown
    by its thickness, its caps moved out and its rims round (no corner: the middle nodes
    on its distance, not on round caps); the cores of shells the union shrunk, the caps
    moved in and the rims sharp. Boolean cylinders coated and hollowed, the fractions
    inside each union within 1e-5 of counts on grids of 3000 x 3000 points."""
    flat = mg.models.BooleanCylinders(radius=0.04, length=0.4, volume_fraction=0.1)
    for side, phases in (("outside", (1, 2)), ("inside", (2, 1))):
        coating = Coating(model=flat, thickness=0.012, side=side, core_phase=2)
        micro = coating.generate(BOX, seed=1)
        unions, _ = section_layers(micro)
        offsets = _layer_offsets(micro)
        bodies = [section_bodies(u, offset=o) for u, o in zip(unions, offsets, strict=True)]
        assert all(b.offset == o for found, o in zip(bodies, offsets, strict=True) for b in found)
        assert any(b.flat is not None for found in bodies for b in found)
        mesh = plane_mesh(micro, size=0.05, order=2)
        assert mesh.metadata["mesher"] == "exact contours (mmg2d)" and mesh.matching_sides()
        assert mesh.min_jacobian() > 0.5
        f = mesh.phase_fractions()
        inner, outer = f[phases[0]], f[phases[0]] + f[phases[1]]
        assert inner == pytest.approx(_counted_bodies(bodies[0], 3000), abs=1e-5)
        assert outer == pytest.approx(_counted_bodies(bodies[1], 3000), abs=1e-5)


def test_contours_at_the_corners_and_along_the_sides():
    """Where a contour leaves the box near a corner of it, its crossings of the two lines
    of the sides in turn along it as their points on the surface (those of its chord came
    in the other order: a piece ended off its side, and the mesh failed); where it crosses
    the line of a side at a grazing angle, the point of the surface on the line bracketed
    (Newton's steps overshoot): a dense ply misaligned by 6 degrees, three realizations,
    meshed by their contours; a circle that a line nearly touches, its two points on it."""
    box = mg.Domain(size=(50.0, 50.0, 200.0), periodic=True)
    for seed in (1, 2, 3):
        ply = LongFibers(radius=3.5, volume_fraction=0.65, min_gap=0.2,
                         misalignment=6.0).generate(box, seed=seed)  # fmt: skip
        mesh = contour_mesh(ply, size=1.0, order=1)
        assert mesh is not None and mesh.matching_sides()
    r, gap = 0.1, 1e-4  # the line x = 0.4 + gap, the circle about (0.5, 0.5)
    (body,) = section_bodies(_straight([[0.5, 0.5]], [r]))
    half = np.sqrt(r**2 - (r - gap) ** 2)
    for start, root in ((0.6, 0.5 + half), (0.45, 0.5 - half)):
        t = _line_root(body, 0, 0.4 + gap, start, 0.01, 1e-12)
        assert t == pytest.approx(root, abs=1e-12)
    assert _line_root(body, 0, 0.3, 0.5, 0.01, 1e-12) is None  # (a line that misses it)
