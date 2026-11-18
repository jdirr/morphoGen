"""Geometries for isogeometric analyses: patches and assemblies with exact arcs.

A circular arc of half-angle ``a`` is a rational curve of degree 2: its ends, the point
where their tangents meet, and the weights ``(1, cos a, 1)`` (Piegl & Tiller 1997, ch.
7); an ellipse is the image of a circle by a linear map of its control points. The
patches here are *ruled* between two such curves — linear across, of the degree of the
curves along — with positive Jacobians:

* :func:`rectangle`, :func:`box`: a rectangle or a box, of any degree;
* :func:`annulus`: a sector of a ring (a thick-walled cylinder in plane strain);
* :func:`plate_with_hole`: a quarter of a square plate with a central hole, two patches
  (the benchmark of Kirsch's solution, and of shape optimization);
* :func:`cell_with_hole`: a rectangular cell with a hole, four patches around it, whose
  opposite sides match: a periodic cell (porous and architectured materials), the hole
  a circle, an ellipse, or any closed curve through its control points; the cell may
  be a parallelogram (the rhombus of a hexagonal array of holes);
* :func:`alternating_holes`: a sheet with elongated holes at the nodes of a square
  lattice, turned in turn along its two axes — the pattern of the *rotating squares*,
  auxetic when its ligaments are thin (Grima & Evans 2000; Bertoldi et al. 2010;
  Taylor et al. 2014);
* :func:`tiling`: cells side by side as one assembly (several holes per period);
  :func:`repeated`: a cell and its neighbours along its periods;
* :func:`extruded`: a surface swept along the third axis into a solid.
"""

from __future__ import annotations

import math

import numpy as np

from morphogen.iga.basis import open_knots
from morphogen.iga.geometry import Multipatch, Patch


def rectangle(lower=(0.0, 0.0), upper=(1.0, 1.0), degree: int = 1, phase: int = 0) -> Patch:
    """The rectangle of corners ``lower`` and ``upper``: one element of ``degree``."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    n = int(degree) + 1
    axes = [np.linspace(lower[d], upper[d], n) for d in range(2)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    knots = open_knots(n, degree)
    return Patch((degree, degree), (knots, knots), points, None, phase)


def box(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0), degree: int = 1, phase: int = 0) -> Patch:
    """The box of corners ``lower`` and ``upper``: one element of ``degree``."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    n = int(degree) + 1
    axes = [np.linspace(lower[d], upper[d], n) for d in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    knots = open_knots(n, degree)
    return Patch((degree,) * 3, (knots,) * 3, points, None, phase)


def arc(
    radius: float, start: float, end: float, centre=(0.0, 0.0)
) -> tuple[np.ndarray, np.ndarray]:
    """The three control points and weights of the circular arc from the angle ``start``
    to ``end`` (less than a half-turn apart): exact, of degree 2."""
    half = 0.5 * (end - start)
    if not 0.0 < abs(half) < 0.5 * math.pi:
        raise ValueError("arc: an angle between 0 and a half-turn")
    middle = 0.5 * (start + end)
    centre = np.asarray(centre, dtype=float)
    points = np.array(
        [
            [radius * math.cos(start), radius * math.sin(start)],
            [
                radius / math.cos(half) * math.cos(middle),
                radius / math.cos(half) * math.sin(middle),
            ],
            [radius * math.cos(end), radius * math.sin(end)],
        ]
    )
    return points + centre, np.array([1.0, math.cos(half), 1.0])


def ruled(first, second, weights_first=None, weights_second=None, degree: int = 2,
          phase: int = 0) -> Patch:  # fmt: skip
    """The patch between two curves of the same degree given by their control points
    (``n x 2`` each, without inner knots): direction 0 along the curves, direction 1
    from the first to the second (linear)."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    n = len(first)
    if first.shape != second.shape or n != int(degree) + 1:
        raise ValueError("ruled: two curves of degree + 1 control points")
    w1 = np.ones(n) if weights_first is None else np.asarray(weights_first, dtype=float)
    w2 = np.ones(n) if weights_second is None else np.asarray(weights_second, dtype=float)
    points = np.stack([first, second], axis=1)
    weights = np.stack([w1, w2], axis=1)
    return Patch((degree, 1), (open_knots(n, degree), open_knots(2, 1)), points, weights, phase)


def annulus(inner: float, outer: float, angle: float = 0.5 * math.pi, start: float = 0.0,
            centre=(0.0, 0.0), phase: int = 0) -> Patch:  # fmt: skip
    """A sector of the ring between the radii ``inner`` and ``outer``, from the angle
    ``start`` over ``angle`` (less than a half-turn): direction 0 along the arcs
    (clockwise, so that the Jacobian is positive), direction 1 from the inner to the
    outer radius."""
    if not 0.0 < inner < outer:
        raise ValueError("annulus: 0 < inner < outer")
    p1, w1 = arc(inner, start + angle, start, centre)
    p2, w2 = arc(outer, start + angle, start, centre)
    return ruled(p1, p2, w1, w2, phase=phase)


def plate_with_hole(radius: float = 1.0, half_width: float = 4.0, phase: int = 0) -> Multipatch:
    """A quarter (``x, y >= 0``) of a square plate of half-width ``half_width`` with a
    circular hole of ``radius`` at the origin: two patches, from the right and upper
    sides (direction 1 = 0) to the arcs of the hole (direction 1 = 1), along which
    direction 0 turns anticlockwise."""
    if not 0.0 < radius < half_width:
        raise ValueError("plate_with_hole: 0 < radius < half_width")
    size = float(half_width)
    quarter = 0.25 * math.pi
    sides = [
        np.array([[size, 0.0], [size, 0.5 * size], [size, size]]),
        np.array([[size, size], [0.5 * size, size], [0.0, size]]),
    ]
    patches = []
    for k, side in enumerate(sides):
        points, weights = arc(radius, k * quarter, (k + 1) * quarter)
        patches.append(ruled(side, points, None, weights, phase=phase))
    return Multipatch(patches)


def _ellipse(hole) -> np.ndarray:
    """The linear map of the unit circle onto an ellipse: ``hole`` a radius, or ``(a,
    b)`` or ``(a, b, angle)`` its half-axes and the inclination of the first."""
    values = (float(hole),) * 2 if np.isscalar(hole) else tuple(float(v) for v in hole)
    if len(values) not in (2, 3) or values[0] <= 0 or values[1] <= 0:
        raise ValueError("hole: a radius, or the positive half-axes of an ellipse (and its angle)")
    tilt = values[2] if len(values) > 2 else 0.0
    rotation = np.array([[math.cos(tilt), -math.sin(tilt)], [math.sin(tilt), math.cos(tilt)]])
    return rotation @ np.diag(values[:2]) @ rotation.T


def _ellipse_arc(shape: np.ndarray, centre, start: float, end: float):
    """The exact arc of the ellipse that ``shape`` makes of the unit circle, between two
    angles of the circle: control points and weights."""
    points, weights = arc(1.0, start, end)
    return points @ shape.T + np.asarray(centre, dtype=float), weights


def _hole_arcs(hole, centre: np.ndarray, corners=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """The four arcs (control points, weights) of a hole around ``centre``: from the
    angles -45 to 45 degrees, then 45 to 135... (between the directions of the
    ``corners`` of the cell, when given). ``hole``: a radius, ``(a, b)`` or ``(a, b,
    angle)`` the half-axes of an ellipse and its inclination, or the 8 control points of
    a closed curve of four quadratic arcs (corner points included) with their weights."""
    if np.isscalar(hole) or np.ndim(hole) == 1:
        shape = _ellipse(hole)
        if corners is None:
            angles = (2 * np.arange(5) - 1) * 0.25 * math.pi
        else:  # towards the corners: the angles of the circle that the ellipse is made of
            directions = (np.asarray(corners) - centre) @ np.linalg.inv(shape).T
            angles = np.unwrap(np.arctan2(directions[:, 1], directions[:, 0]))
            angles = np.append(angles, angles[0] + 2 * math.pi)
        return [_ellipse_arc(shape, centre, angles[k], angles[k + 1]) for k in range(4)]
    hole = np.asarray(hole, dtype=float)
    if hole.shape != (8, 3):
        raise ValueError(
            "hole: a radius, the half-axes (and inclination) of an ellipse, or 8 control "
            "points with their weights (x, y, w), anticlockwise from the angle -45 degrees"
        )
    arcs = []
    for k in range(4):
        rows = [2 * k, 2 * k + 1, (2 * k + 2) % 8]
        arcs.append((hole[rows, :2], hole[rows, 2]))
    return arcs


def cell_with_hole(lengths=(1.0, 1.0), hole=0.25, elements: int = 1, origin=(0.0, 0.0),
                   phase: int = 0) -> Multipatch:  # fmt: skip
    """The rectangular cell of ``lengths`` (its lower corner at ``origin``) with a hole
    at its centre (see :func:`_hole_arcs`: a radius, an ellipse, or a closed curve): four
    patches, one per side of the cell, from the side (direction 1 = 0) to the arc of the
    hole that faces it (direction 1 = 1), direction 0 anticlockwise; ``elements`` along
    each arc. Opposite sides have matching control points: a periodic cell.

    ``lengths`` may be the two vectors of a parallelogram instead (2 x 2, one per row,
    anticlockwise): the cell of an oblique lattice, which the assembly then carries
    (``lattice``) — ``[[1, 0], [0.5, sqrt(3) / 2]]`` for a hexagonal array of holes."""
    lengths = np.asarray(lengths, dtype=float)
    origin = np.asarray(origin, dtype=float)
    lattice = None
    if lengths.ndim == 2:
        lattice = lengths
        if lattice.shape != (2, 2) or np.linalg.det(lattice) <= 0:
            raise ValueError("cell_with_hole: two vectors of the plane, anticlockwise")
        first, second = lattice
    else:
        first, second = np.array([lengths[0], 0.0]), np.array([0.0, lengths[1]])
    centre = origin + 0.5 * (first + second)
    corners = origin + np.array([first, first + second, second, 0.0 * first])
    patches = []
    arcs = _hole_arcs(hole, centre, None if lattice is None else corners)
    for k, (points, weights) in enumerate(arcs):
        a, b = corners[k], corners[(k + 1) % 4]
        side = np.array([a, 0.5 * (a + b), b])
        patch = ruled(side, points, None, weights, phase=phase)
        patches.append(patch.refined((int(elements), 1)) if elements > 1 else patch)
    return Multipatch(patches, lattice=lattice)


def alternating_holes(spacing: float = 1.0, hole=(0.35, 0.15), elements: int = 1,
                      phase: int = 0) -> Multipatch:  # fmt: skip
    """A sheet with elliptic holes at the nodes of a square lattice of ``spacing``
    along x and y, elongated in turn along x and along y: ``hole = (a, b)``, the
    half-axes along and across a hole (``a + b < spacing``; a radius for circles), the
    hole at the origin along x. Between four holes the sheet is a square grain, held to
    its four neighbours by the ligaments left between the tip of a hole and the flank of
    the next, of width ``spacing - a - b``: thin ligaments let the grains turn, and the
    sheet widens when pulled (the *rotating squares*).

    The assembly is two grains side by side, cut nowhere — ``[0, 2 spacing] x [0,
    spacing]`` without its quarters of holes — whose periods are ``(spacing, spacing)``
    and ``(spacing, -spacing)`` (its ``lattice``): two holes a period. Each grain is
    eight patches around its centre, two along each of its quarters of holes, from the
    arc (direction 1 = 0) to a curve between the centre of the grain and the middle of
    a ligament, which it crosses squarely (direction 1 = 1), with ``elements`` along each
    arc. The patches 0 and 1 are those of the hole at the origin: their arcs run from
    its flank ``(0, b)`` to a point ``p``, then from ``p`` to its tip ``(a, 0)``, facing
    the curves from the middle of the left ligament to the centre and from the centre
    to the middle of the lower ligament; the other quarters are their images by the
    symmetries of the pattern (the mirrors x = 0 and y = 0, the quarter-turn about the
    centre of a grain)."""
    d = float(spacing)
    shape = _ellipse(hole)
    a, b = float(shape[0, 0]), float(shape[1, 1])
    if np.ndim(hole) == 1 and len(hole) > 2:
        raise ValueError("alternating_holes: hole = (a, b), the half-axes along and across")
    if not a + b < d:
        raise ValueError("alternating_holes: a + b < spacing (holes apart)")
    centre = np.array([0.5 * d, 0.5 * d])
    middle = np.array([0.5 * (a + d - b), 0.0])  # of the ligament of the lower side
    left = np.array([0.0, 0.5 * (b + d - a)])  # of that of the left side
    # the point of the arc of the hole seen from the centre half-way between them
    angle = math.atan2(middle[1] - centre[1], middle[0] - centre[0]) - 0.25 * math.pi
    ray = np.array([math.cos(angle), math.sin(angle)])
    roots = np.roots(
        [
            (ray[0] / a) ** 2 + (ray[1] / b) ** 2,
            2.0 * (centre[0] * ray[0] / a**2 + centre[1] * ray[1] / b**2),
            (centre[0] / a) ** 2 + (centre[1] / b) ** 2 - 1.0,
        ]
    )
    hit = centre + float(np.min(roots.real)) * ray
    split = math.atan2(hit[1] / b, hit[0] / a)
    flank, flank_weights = _ellipse_arc(shape, (0.0, 0.0), 0.5 * math.pi, split)
    tip, tip_weights = _ellipse_arc(shape, (0.0, 0.0), split, 0.0)
    # from the centre to the middle of a ligament, which the curve crosses squarely
    pieces = [
        (flank, flank_weights, np.array([left, [0.25 * d, left[1]], centre])),
        (tip, tip_weights, np.array([centre, [middle[0], 0.25 * d], middle])),
    ]
    turn = np.array([[0.0, -1.0], [1.0, 0.0]])
    patches = []
    for mirrored in (False, True):
        for k in range(4):
            rotation = np.linalg.matrix_power(turn, k)
            for curve, weights, line in pieces:
                first = centre + (curve - centre) @ rotation.T
                second = centre + (line - centre) @ rotation.T
                if mirrored:  # the second grain: the mirror image in the line x = spacing
                    first = (first * [-1.0, 1.0] + [2.0 * d, 0.0])[::-1]
                    second = (second * [-1.0, 1.0] + [2.0 * d, 0.0])[::-1]
                    weights = weights[::-1]
                patch = ruled(first, second, weights, None, phase=phase)
                patches.append(patch.refined((int(elements), 1)) if elements > 1 else patch)
    return Multipatch(patches, lattice=[[d, d], [d, -d]])


def tiling(cells) -> Multipatch:
    """The assembly of several assemblies (or patches) placed side by side: their
    patches together, the control points of their common sides merged."""
    patches = []
    for cell in cells:
        patches.extend([cell] if isinstance(cell, Patch) else cell.patches)
    return Multipatch(patches)


def repeated(cell: Multipatch, counts=(2, 2)) -> Multipatch:
    """A cell and its neighbours: ``counts`` copies along each of its periods (those of
    its ``lattice``, or the sides of its box), as one assembly — for figures, or a
    larger period."""
    lattice = np.diag(cell.upper - cell.lower) if cell.lattice is None else cell.lattice
    counts = (int(counts),) * cell.dim if np.isscalar(counts) else tuple(int(c) for c in counts)
    if len(counts) != cell.dim or min(counts) < 1:
        raise ValueError("repeated: a count of 1 or more per period")
    patches = []
    for index in np.ndindex(*counts):
        shift = np.asarray(index, dtype=float) @ lattice
        patches.extend(p.with_points(p.points + shift) for p in cell.patches)
    return Multipatch(patches, cell.tolerance, lattice=np.asarray(counts)[:, None] * lattice)


def extruded(surface, height: float, degree: int = 1) -> Multipatch:
    """The solid swept by a surface (a patch or an assembly in the plane) along the
    third axis over ``height``: direction 2 of each patch, of ``degree``."""
    surfaces = [surface] if isinstance(surface, Patch) else surface.patches
    n = int(degree) + 1
    levels = np.linspace(0.0, float(height), n)
    patches = []
    for p in surfaces:
        if p.ndim != 2 or p.dim != 2:
            raise ValueError("extruded: surfaces of the plane")
        points = np.empty((*p.shape, n, 3))
        points[..., :2] = p.points[:, :, None, :]
        points[..., 2] = levels
        weights = np.repeat(p.weights[:, :, None], n, axis=2)
        patches.append(
            Patch((*p.degrees, degree), (*p.knots, open_knots(n, degree)), points, weights, p.phase)
        )
    return Multipatch(patches)


__all__ = [
    "alternating_holes",
    "annulus",
    "arc",
    "box",
    "cell_with_hole",
    "extruded",
    "plate_with_hole",
    "rectangle",
    "repeated",
    "ruled",
    "tiling",
]
