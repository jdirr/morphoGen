"""Scale-rich networks: ligaments of power-law thicknesses grown one after another until
they meet (Both, Chen, Gao, Mooij, Charara, Wadhwa, Portela & Barabási 2026).

Mechanical metamaterials are usually ruled by one length, that of their cell. Scale-rich
networks give up that length: their ligaments span orders of magnitude in thickness,
length and connectivity within one sample, as the veins of leaves and dragonfly wings, the
threads of spider webs or the roads of a city. Both et al. (Nature 659, 345, 2026) grow
them from two parameters:

* the ligament ``t = 1, 2, ..., T`` is ``lambda_t = lambda_0 t^-alpha`` thick
  (``initial_thickness`` and ``decay``);
* it nucleates at a point drawn uniformly in the voids left by the ligaments before it,
  along an angle ``theta_t``, and grows both ways until it meets the face of another
  ligament or the boundary of the box.

The voids are convex cells (at first the rectangle of the box), each bounded by faces of
ligaments and of the box. A ligament lies in the cell of its nucleation point: it is the
strip of width ``lambda_t`` along its line clipped by that cell, which it splits into two
convex cells (the scheme of the reference implementation of the authors). The point is
drawn uniformly in the box until it falls in a void, the same law as a void drawn in
proportion to its area then a point uniform in it. Ligaments and voids partition the box
exactly, so that the relative density ``1 - sum(void areas) / A`` is exact; the ligaments
never overlap. A ligament fits when its cell holds the two points at
``(lambda_t + epsilon) / 2`` on either side of its nucleation point (``epsilon``:
``min_spacing``, the least width of the voids beside a new ligament); when
``max_attempts`` draws in a row do not fit, the network is jammed and stops growing. The
length of a ligament is that of its centre line in its cell; its neighbours are the
ligaments (or the box) on which its two ends lie. As roads in the dual graphs of road
networks (Rosvall et al. 2005), the ligaments are the nodes of the network and their
junctions its links: the degree ``k`` of a ligament counts the ligaments on which it ends
and those that end on it.

Laws (Both et al. 2026): the thicknesses follow ``P(lambda) ~ lambda^-(1 + 1/alpha)``
exactly; a ligament splits a cell of area about ``A_t / t``, so that its length decays
as ``t^-1/2`` (``A_t``: the area of the voids) and the lengths have the tail ``P(l) ~
l^-3``; a long ligament is met by the ligaments that end on it in proportion to its
length (a spatial preferential attachment), and the degrees have the tail ``P(k) ~ k^-3``.
The relative density is ``sum_t lambda_t l_t / A`` (``l_t`` the length of ``t``), see
:meth:`ScaleRichNetwork.theory`.

Orientations: uniform (the scale-rich networks of the paper), drawn among ``angles``
(aligned networks, e.g. ``[0, 90]``), or polarized by a ``pole`` (the ligaments along the
direction of the pole turned by ``pole_angle``: radial, tangential or spiral, for the
``polarized_fraction`` first of them), each spread by ``angle_spread`` (a normal law).
The nucleation density may follow a gradient along an axis of the plane or away from the
pole (``nucleation_gradient``). Their combinations draw the designs of Fig. 6 of the
paper, as the radial gradient of the pomelo peel protecting a payload.

The network lies in the plane of the two axes other than ``axis`` and is extruded along
it through the box: the prismatic specimens of the paper (plane strain), one voxel thick
for planar images. Its ligaments end on the faces of the box, which is not periodic; a
``frame`` of material around the network, on which they end, makes the sample the cell of
a periodic medium (FFT solvers, Bloch waves), stiffened by the frame.

With ``dimension: 3``, the network grows in space: plates that split convex polyhedral
voids (closed cells), or beams (open cells), see :mod:`morphogen.models.scale_rich_3d`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import ClassVar

import numpy as np
from numba import njit, prange

from morphogen._utils import check_fraction, check_positive, exactly_one
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.models.planar import pixel_centres, plane_axes
from morphogen.models.scale_rich_3d import (
    AREA_CONSTANT,
    BOX,
    ScaleRichBeams,
    ScaleRichPlates,
    area_sum,
    beam_density_estimate,
    beam_thickness_estimate,
    first_area,
    grow_beams,
    grow_plates,
)
from morphogen.orientations import OrientationDistribution, orthonormal_basis, uniform_unit_vectors
from morphogen.seeding import SeedLike, make_rng

#: Tolerance on the relative density of a realization solved for a ``volume_fraction``
DENSITY_TOL = 2e-4
#: ``lim sqrt(t / A) E[l_t]``: the mean length of the ligament ``t`` of a thin network
#: (``lambda_0 -> 0``) in a box of area ``A``, measured with uniform orientations: 1.1663
#: and 1.1658 (standard errors 0.0004) over the ligaments ``t >= 2000`` of 120
#: realizations of 20,000 in a square and in a 2 x 1 rectangle. The first ligament is
#: ``l_0`` (:func:`first_length`, 0.9464 in a square), the next ones approach the limit
#: (1.05, 1.09, 1.12 at ``t`` = 2, 3, 5)
LENGTH_CONSTANT = 1.166
#: Uniform draws of a nucleation point: its two coordinates, its rejection by the density
#: of nucleation, and two for its angle (the angle, and the normal of its spread)
_DRAWS = 5

# -- convex polygons (counter-clockwise vertices, the label of the face of each edge) -------


@njit(cache=True)
def polygon_area(xy) -> float:
    """Area of a polygon (counter-clockwise: positive) by the shoelace formula."""
    m = xy.shape[0]
    s = 0.0
    for j in range(m):
        k = (j + 1) % m
        s += xy[j, 0] * xy[k, 1] - xy[k, 0] * xy[j, 1]
    return 0.5 * s


@njit(cache=True)
def _merged(xy, faces, tol):
    """The polygon without its edges shorter than ``tol`` (a vertex that coincides with
    the next one goes, the next one keeping the face of its outgoing edge)."""
    m = xy.shape[0]
    keep = np.ones(m, dtype=np.bool_)
    for j in range(m):
        k = (j + 1) % m
        if abs(xy[j, 0] - xy[k, 0]) <= tol and abs(xy[j, 1] - xy[k, 1]) <= tol:
            keep[j] = False
    if keep.all():
        return xy, faces
    if not keep.any():
        return xy[:1].copy(), faces[:1].copy()
    return xy[keep].copy(), faces[keep].copy()


@njit(cache=True)
def clip_polygon(xy, faces, nx, ny, c, label, tol):
    """The part of a convex polygon where ``nx x + ny y <= c``: its vertices and the faces
    of its edges, the edge along the line labelled ``label`` (Sutherland-Hodgman)."""
    m = xy.shape[0]
    out = np.empty((m + 2, 2))
    out_faces = np.empty(m + 2, dtype=np.int64)
    k = 0
    for j in range(m):
        jn = (j + 1) % m
        dj = nx * xy[j, 0] + ny * xy[j, 1] - c
        dn = nx * xy[jn, 0] + ny * xy[jn, 1] - c
        if dj <= 0.0:
            out[k, 0], out[k, 1] = xy[j, 0], xy[j, 1]
            out_faces[k] = faces[j]
            k += 1
            if dn > 0.0:  # leaves the half-plane: the rest of the edge, then the line
                t = dj / (dj - dn)
                out[k, 0] = xy[j, 0] + t * (xy[jn, 0] - xy[j, 0])
                out[k, 1] = xy[j, 1] + t * (xy[jn, 1] - xy[j, 1])
                out_faces[k] = label
                k += 1
        elif dn <= 0.0:  # enters it: the rest of the edge
            t = dj / (dj - dn)
            out[k, 0] = xy[j, 0] + t * (xy[jn, 0] - xy[j, 0])
            out[k, 1] = xy[j, 1] + t * (xy[jn, 1] - xy[j, 1])
            out_faces[k] = faces[j]
            k += 1
    return _merged(out[:k], out_faces[:k], tol)


@njit(cache=True)
def chord(xy, faces, px, py, dx, dy):
    """Where the line ``p + s d`` leaves a convex polygon holding ``p``: ``s0 <= 0 <=
    s1``, and the faces of the edges it leaves by."""
    m = xy.shape[0]
    s0, s1 = -np.inf, np.inf
    f0, f1 = BOX, BOX
    for j in range(m):
        jn = (j + 1) % m
        mx = -(xy[jn, 1] - xy[j, 1])  # the inward normal of the edge (not unit)
        my = xy[jn, 0] - xy[j, 0]
        h = mx * (px - xy[j, 0]) + my * (py - xy[j, 1])
        md = mx * dx + my * dy
        if md < 0.0:
            s = -h / md
            if s < s1:
                s1, f1 = s, faces[j]
        elif md > 0.0:
            s = -h / md
            if s > s0:
                s0, f0 = s, faces[j]
    return s0, s1, f0, f1


@njit(cache=True)
def fits(xy, px, py, nx, ny, half) -> bool:
    """Whether a convex polygon holds the points ``p +- half n``."""
    m = xy.shape[0]
    for j in range(m):
        jn = (j + 1) % m
        ex, ey = xy[jn, 0] - xy[j, 0], xy[jn, 1] - xy[j, 1]
        mx, my = -ey, ex
        h = mx * (px - xy[j, 0]) + my * (py - xy[j, 1])
        if h - half * abs(mx * nx + my * ny) < 0.0:
            return False
    return True


@njit(cache=True)
def inside_polygon(xy, px, py) -> bool:
    """Whether a convex polygon (counter-clockwise) holds the point ``p``."""
    m = xy.shape[0]
    for j in range(m):
        jn = (j + 1) % m
        cross = (xy[jn, 0] - xy[j, 0]) * (py - xy[j, 1]) - (xy[jn, 1] - xy[j, 1]) * (px - xy[j, 0])
        if cross < 0.0:
            return False
    return True


class _Voids:
    """The voids of a growing network registered in a grid of buckets of the rectangle, to
    find the void that holds a point (or none: the point lies in a ligament)."""

    def __init__(self, lower, upper, n_buckets: int) -> None:
        self.lo = np.asarray(lower, dtype=float)
        size = np.asarray(upper, dtype=float) - self.lo
        g = np.clip(np.ceil(math.sqrt(max(n_buckets, 1)) * size / math.sqrt(np.prod(size))),
                    1, 256)  # fmt: skip
        self.g0, self.g1 = int(g[0]), int(g[1])
        self.h0, self.h1 = float(size[0] / g[0]), float(size[1] / g[1])
        self.buckets: list[set[int]] = [set() for _ in range(self.g0 * self.g1)]
        self.boxes: dict[int, tuple] = {}
        self.cells: list[np.ndarray] = []

    def _range(self, v, lo, h, n) -> int:
        return min(max(math.floor((v - lo) / h), 0), n - 1)

    def put(self, cid: int, xy: np.ndarray) -> None:
        """Registers the void ``cid`` (a new one, or in place of the old ``cid``)."""
        if cid < len(self.cells):
            self.drop(cid)
            self.cells[cid] = xy
        else:
            self.cells.append(xy)
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        i0, i1 = (
            self._range(x0, self.lo[0], self.h0, self.g0),
            self._range(x1, self.lo[0], self.h0, self.g0),
        )
        j0, j1 = (
            self._range(y0, self.lo[1], self.h1, self.g1),
            self._range(y1, self.lo[1], self.h1, self.g1),
        )
        self.boxes[cid] = (x0, y0, x1, y1, i0, i1, j0, j1)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self.buckets[i * self.g1 + j].add(cid)

    def drop(self, cid: int) -> None:
        *_, i0, i1, j0, j1 = self.boxes.pop(cid)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self.buckets[i * self.g1 + j].discard(cid)

    def locate(self, px: float, py: float) -> int:
        i = self._range(px, self.lo[0], self.h0, self.g0)
        j = self._range(py, self.lo[1], self.h1, self.g1)
        for cid in self.buckets[i * self.g1 + j]:
            x0, y0, x1, y1 = self.boxes[cid][:4]
            if x0 <= px <= x1 and y0 <= py <= y1 and inside_polygon(self.cells[cid], px, py):
                return cid
        return -1


# -- the growth --------------------------------------------------------------------------


@dataclass
class Growth:
    """The ligaments and voids of a scale-rich network (coordinates of the plane)."""

    nucleation: np.ndarray  # (N, 2) the nucleation points
    angle: np.ndarray  # (N,) radians, of the centre lines
    thickness: np.ndarray  # (N,)
    ends: np.ndarray  # (N, 2, 2) the ends of the centre lines
    neighbours: np.ndarray  # (N, 2) the ligaments of the ends (BOX: the boundary)
    body_offsets: np.ndarray  # (N + 1,) CSR of the vertices of the ligaments
    body_vertices: np.ndarray  # (M, 2)
    cell_offsets: np.ndarray  # (C + 1,) CSR of the vertices of the voids
    cell_vertices: np.ndarray  # (K, 2)
    cell_faces: np.ndarray  # (K,) the face of each edge (ligament index or BOX)
    jammed: bool = False
    attempts: int = 0

    @property
    def n(self) -> int:
        return len(self.thickness)

    @property
    def n_cells(self) -> int:
        return len(self.cell_offsets) - 1

    def bodies(self) -> list[np.ndarray]:
        """The vertices of each ligament (convex polygons, counter-clockwise)."""
        o = self.body_offsets
        return [self.body_vertices[o[i] : o[i + 1]] for i in range(self.n)]

    def cells(self) -> list[np.ndarray]:
        """The vertices of each void (convex polygons, counter-clockwise)."""
        o = self.cell_offsets
        return [self.cell_vertices[o[i] : o[i + 1]] for i in range(self.n_cells)]

    def body_areas(self) -> np.ndarray:
        return np.array([polygon_area(b) for b in self.bodies()])

    def cell_areas(self) -> np.ndarray:
        return np.array([polygon_area(c) for c in self.cells()])

    @property
    def lengths(self) -> np.ndarray:
        """The lengths of the centre lines of the ligaments."""
        return np.linalg.norm(self.ends[:, 1] - self.ends[:, 0], axis=1)

    def links(self) -> np.ndarray:
        """The pairs of ligaments joined: each ligament with those on which its ends lie
        (the edges of the dual graph, each once)."""
        i = np.repeat(np.arange(self.n), 2)
        j = self.neighbours.ravel()
        keep = j >= 0
        return np.column_stack([i[keep], j[keep]])

    def degrees(self) -> np.ndarray:
        """The number of ligaments each ligament meets (its degree in the dual graph)."""
        e = self.links()
        return np.bincount(e.ravel(), minlength=self.n).astype(np.int64)

    def orientation_tensor(self) -> np.ndarray:
        """``M = sum A_t d_t d_t / sum A_t``: the directions ``d_t`` of the ligaments
        weighted by their areas ``A_t`` (Both et al. 2026, eq. 1, normalized)."""
        a = self.body_areas()
        d = np.column_stack([np.cos(self.angle), np.sin(self.angle)])
        if not a.sum() > 0:
            return np.full((2, 2), np.nan)
        return np.einsum("t,ti,tj->ij", a, d, d) / a.sum()


def _csr(items: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(items) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(a) for a in items])
    data = np.concatenate(items) if items else np.empty((0, 2))
    return offsets, data


def grow(
    lower,
    upper,
    n_ligaments: int,
    thickness,
    entropy: int,
    *,
    orientation=None,
    density=None,
    min_spacing: float = 0.0,
    max_attempts: int = 1000,
) -> Growth:
    """Grow a scale-rich network in the rectangle ``[lower, upper]`` of the plane.

    ``thickness(t)`` is the thickness of the ligament ``t = 1, 2, ...``; ``orientation(t,
    x, y, u, v)`` its angle (radians) at its nucleation point from two uniform numbers
    (default: ``pi u``, uniform); ``density(x, y)`` the relative density of nucleation in
    ``[0, 1]`` (default: uniform), by rejection. A nucleation point is drawn uniformly in
    the rectangle until it falls in a void (the law of a void drawn in proportion to its
    area, then a point uniform in it); each ligament draws from its own generator
    (``SeedSequence(entropy, spawn_key=(t,))``): its draws do not depend on the ligaments
    before it, so that a small change of the thicknesses changes the network only where a
    point or an end falls close to a moved side (the relative density of
    :class:`ScaleRichNetwork` is solved on realizations of the same seed). The growth stops,
    jammed, after ``max_attempts`` nucleation points in a row where the ligament does not
    fit, or when the voids vanish.
    """
    lo = np.asarray(lower, dtype=float).reshape(2)
    hi = np.asarray(upper, dtype=float).reshape(2)
    size = hi - lo
    area = float(np.prod(size))
    tol = 1e-12 * float(np.max(size))
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    voids = _Voids(lo, hi, n_buckets=max(1, int(n_ligaments) // 2))
    voids.put(0, box)
    cells_faces = [np.full(4, BOX, dtype=np.int64)]
    void_area = area

    nucleation, angle, lam, ends, neighbours, bodies = [], [], [], [], [], []
    attempts = 0
    jammed = False
    for t in range(1, int(n_ligaments) + 1):
        width = float(thickness(t))
        half = 0.5 * (width + min_spacing)
        gen = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=(t,))))
        failures = k = 0
        block = gen.random((16, _DRAWS))
        while True:
            if k == len(block):
                if void_area <= 1e-12 * area:
                    jammed = True
                    break
                block, k = gen.random((256, _DRAWS)), 0
            draw = block[k]
            k += 1
            px, py = lo[0] + draw[0] * size[0], lo[1] + draw[1] * size[1]
            if density is not None and draw[2] >= density(px, py):
                continue
            i = voids.locate(px, py)
            if i < 0:
                continue  # in a ligament
            attempts += 1
            if orientation is None:
                theta = math.pi * draw[3]
            else:
                theta = orientation(t, px, py, draw[3], draw[4])
            dx, dy = math.cos(theta), math.sin(theta)
            nx, ny = -dy, dx
            if fits(voids.cells[i], px, py, nx, ny, half):
                break
            failures += 1
            if failures >= max_attempts:
                jammed = True
                break
        if jammed:
            break
        xy, faces = voids.cells[i], cells_faces[i]
        s0, s1, f0, f1 = chord(xy, faces, px, py, dx, dy)
        c = nx * px + ny * py
        label = t - 1
        minus_xy, minus_f = clip_polygon(xy, faces, nx, ny, c - 0.5 * width, label, tol)
        plus_xy, plus_f = clip_polygon(xy, faces, -nx, -ny, -(c + 0.5 * width), label, tol)
        body, body_f = clip_polygon(xy, faces, nx, ny, c + 0.5 * width, label, tol)
        body, _ = clip_polygon(body, body_f, -nx, -ny, -(c - 0.5 * width), label, tol)
        void_area += polygon_area(minus_xy) + polygon_area(plus_xy) - polygon_area(xy)
        voids.put(i, minus_xy)
        cells_faces[i] = minus_f
        voids.put(len(voids.cells), plus_xy)
        cells_faces.append(plus_f)
        nucleation.append((px, py))
        angle.append(math.atan2(dy, dx) % math.pi)
        lam.append(width)
        ends.append(((px + s0 * dx, py + s0 * dy), (px + s1 * dx, py + s1 * dy)))
        neighbours.append((f0, f1))
        bodies.append(body)

    body_offsets, body_vertices = _csr(bodies)
    cell_offsets, cell_vertices = _csr(voids.cells)
    n = len(lam)
    return Growth(
        nucleation=np.asarray(nucleation, dtype=float).reshape(n, 2),
        angle=np.asarray(angle, dtype=float),
        thickness=np.asarray(lam, dtype=float),
        ends=np.asarray(ends, dtype=float).reshape(n, 2, 2),
        neighbours=np.asarray(neighbours, dtype=np.int64).reshape(n, 2),
        body_offsets=body_offsets,
        body_vertices=body_vertices,
        cell_offsets=cell_offsets,
        cell_vertices=cell_vertices,
        cell_faces=np.concatenate(cells_faces).astype(np.int64),
        jammed=jammed,
        attempts=attempts,
    )


# -- theory ------------------------------------------------------------------------------


def first_length(a: float, b: float) -> float:
    """The mean length of the first ligament in an ``a x b`` rectangle: the chord through
    a uniform point along a uniform direction, ``int l^2 dG / (pi a b)`` over the lines
    ``G``. With ``d = sqrt(a^2 + b^2)``::

        l_0 = (2 / pi) (a asinh(b / a) + b asinh(a / b) - (a^2 (d - a) + b^2 (d - b)) / (3 a b))

    0.946402 for the unit square (Kellerer 1971)."""
    a, b = float(a), float(b)
    d = math.hypot(a, b)
    corners = (a * a * (d - a) + b * b * (d - b)) / (3.0 * a * b)
    return (2.0 / math.pi) * (a * math.asinh(b / a) + b * math.asinh(a / b) - corners)


def length_sum(decay: float, n_ligaments: int, a: float, b: float) -> float:
    """``sum_t l_t t^-alpha / sqrt(A)`` of a thin network of ``T`` ligaments in an ``a x
    b`` rectangle: ``l_0 / sqrt(A) + c sum_{t=2..T} t^-(alpha + 1/2)``
    (:data:`LENGTH_CONSTANT`), the relative density per unit of ``lambda_0 / sqrt(A)``."""
    area = a * b
    tail = 0.0
    if n_ligaments >= 2:
        t = np.arange(2, int(n_ligaments) + 1, dtype=float)
        tail = float(np.sum(t ** (-(decay + 0.5))))
    return first_length(a, b) / math.sqrt(area) + LENGTH_CONSTANT * tail


def density_estimate(first_order: float, dimension: int = 2) -> float:
    """The relative density of a network from its first order ``rho_1 = sum_t lambda_t
    E[l_t] / A`` (thin ligaments): a ligament splits a void of area ``A_v / t`` (``A_v``
    the area of the voids), so that ``dA_v/dt = -lambda_t c sqrt(A_v / t)`` and
    ``sqrt(A_v / A) = 1 - rho_1 / 2``: ``rho = rho_1 (1 - rho_1 / 4)``, jammed (1) from
    ``rho_1 = 2`` on. In ``d`` dimensions (plates in 3D: ``rho_1 = sum_t lambda_t E[a_t] /
    V``, the area ``a_t`` of a plate about ``c_3 (V_v / t)^(2/3)``), ``(V_v / V)^(1/d) = 1 -
    rho_1 / d``: ``rho = 1 - (1 - rho_1 / d)^d``, jammed from ``rho_1 = d`` on."""
    r, d = float(first_order), int(dimension)
    return 1.0 if r >= d else 1.0 - (1.0 - r / d) ** d


# -- the field of the ligaments on points of the plane: a grid of buckets ------------------


@njit(cache=True)
def _half_planes(vertices, offsets):
    """Outward unit normals ``n`` and offsets ``c`` of the edges of convex polygons
    (counter-clockwise): ``phi(x) = max_e (n_e . x - c_e)``, negative inside."""
    m = vertices.shape[0]
    nx = np.zeros(m)
    ny = np.zeros(m)
    c = np.full(m, np.inf)  # degenerate edges: no constraint
    for p in range(offsets.shape[0] - 1):
        a, b = offsets[p], offsets[p + 1]
        k = b - a
        for j in range(k):
            i0, i1 = a + j, a + (j + 1) % k
            ex, ey = vertices[i1, 0] - vertices[i0, 0], vertices[i1, 1] - vertices[i0, 1]
            length = math.hypot(ex, ey)
            if length > 0.0:
                nx[i0], ny[i0] = ey / length, -ex / length
                c[i0] = nx[i0] * vertices[i0, 0] + ny[i0] * vertices[i0, 1]
    return nx, ny, c


@njit(cache=True)
def _register(boxes, reach, lo0, lo1, h0, h1, g0, g1):
    """The polygons of each bucket of a ``g0 x g1`` grid: those whose box, widened by
    ``reach``, meets it (compressed rows)."""
    n = boxes.shape[0]
    counts = np.zeros(g0 * g1, dtype=np.int64)
    for pass_ in range(2):
        if pass_ == 1:
            offsets = np.zeros(g0 * g1 + 1, dtype=np.int64)
            for b in range(g0 * g1):
                offsets[b + 1] = offsets[b] + counts[b]
            items = np.empty(offsets[-1], dtype=np.int64)
            fill = offsets[:-1].copy()
        for p in range(n):
            i0 = min(max(math.floor((boxes[p, 0] - reach - lo0) / h0), 0), g0 - 1)
            i1 = min(max(math.floor((boxes[p, 2] + reach - lo0) / h0), 0), g0 - 1)
            j0 = min(max(math.floor((boxes[p, 1] - reach - lo1) / h1), 0), g1 - 1)
            j1 = min(max(math.floor((boxes[p, 3] + reach - lo1) / h1), 0), g1 - 1)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    if pass_ == 0:
                        counts[i * g1 + j] += 1
                    else:
                        items[fill[i * g1 + j]] = p
                        fill[i * g1 + j] += 1
    return offsets, items


@njit(parallel=True, cache=True)
def _field(px, py, reach, boxes, offsets, nx, ny, c, b_off, b_items, lo0, lo1, h0, h1, g0, g1):
    """``min(reach, min_p phi_p(x))`` at points of the plane: negative in the ligaments, the
    distance to the nearest one near them (``reach``: far from every one)."""
    out = np.empty(px.shape[0])
    for q in prange(px.shape[0]):
        x, y = px[q], py[q]
        i = min(max(math.floor((x - lo0) / h0), 0), g0 - 1)
        j = min(max(math.floor((y - lo1) / h1), 0), g1 - 1)
        b = i * g1 + j
        best = reach
        for k in range(b_off[b], b_off[b + 1]):
            p = b_items[k]
            if (x < boxes[p, 0] - reach or x > boxes[p, 2] + reach
                    or y < boxes[p, 1] - reach or y > boxes[p, 3] + reach):  # fmt: skip
                continue
            phi = -np.inf
            for e in range(offsets[p], offsets[p + 1]):
                v = nx[e] * x + ny[e] * y - c[e]
                if v > phi:
                    phi = v
                    if phi >= best:
                        break
            if phi < best:
                best = phi
        out[q] = best
    return out


class _Index:
    """The ligaments of a network registered in a grid of buckets of the plane, to
    evaluate their field (:func:`_field`) within ``reach`` of them."""

    def __init__(self, growth: Growth, lower, upper, reach: float) -> None:
        self.reach = float(reach)
        v, o = growth.body_vertices, growth.body_offsets
        self.offsets = o
        self.nx, self.ny, self.c = _half_planes(v, o)
        n = growth.n
        boxes = np.empty((n, 4))
        for p in range(n):
            pts = v[o[p] : o[p + 1]]
            boxes[p, :2], boxes[p, 2:] = pts.min(axis=0), pts.max(axis=0)
        self.boxes = boxes
        lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        size = hi - lo
        cells = max(1.0, math.sqrt(max(n, 1)))
        g = np.clip(np.ceil(cells * size / math.sqrt(float(np.prod(size)))), 1, 1024)
        self.g = (int(g[0]), int(g[1]))
        self.lo = lo
        self.h = size / g
        self.b_off, self.b_items = _register(
            boxes, self.reach, lo[0], lo[1], self.h[0], self.h[1], *self.g
        )

    def field(self, points) -> np.ndarray:
        pts = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 2))
        if not len(self.boxes):
            return np.full(len(pts), self.reach)
        return _field(pts[:, 0].copy(), pts[:, 1].copy(), self.reach, self.boxes, self.offsets,
                      self.nx, self.ny, self.c, self.b_off, self.b_items, self.lo[0],
                      self.lo[1], self.h[0], self.h[1], *self.g)  # fmt: skip


# -- the microstructure ------------------------------------------------------------------


def _array(value, shape_tail, dtype=float) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1, *shape_tail)


@register_kind
class ScaleRichStructure(Microstructure):
    """A scale-rich network extruded along ``axis``: its ligaments (phase 1, convex
    polygons of the plane of the two other axes, which never overlap) and its voids
    (phase 0, convex cells), which partition the face of the box exactly
    (:func:`grow`); or, with a ``frame``, the face within a frame of that thickness
    (phase 1 too), on which the ligaments end."""

    kind = "scale_rich_structure"

    def __init__(self, domain: Domain, axis: int, growth: Growth, frame: float = 0.0,
                 **kwargs) -> None:  # fmt: skip
        super().__init__(domain, **kwargs)
        self.axis = int(axis)
        self.growth = growth
        self.frame = float(frame)
        self._indices: dict[float, _Index] = {}

    # -- the network -------------------------------------------------------------------

    @property
    def plane(self) -> tuple[np.ndarray, np.ndarray]:
        """The lower and upper corners of the face of the box in the plane."""
        b, c = plane_axes(self.axis)
        d = self.domain
        return d.lower[[b, c]], d.upper[[b, c]]

    @property
    def area(self) -> float:
        lo, hi = self.plane
        return float(np.prod(hi - lo))

    @property
    def n_ligaments(self) -> int:
        return self.growth.n

    @property
    def thicknesses(self) -> np.ndarray:
        return self.growth.thickness

    @property
    def lengths(self) -> np.ndarray:
        """The lengths of the centre lines of the ligaments in their voids."""
        return self.growth.lengths

    def degrees(self) -> np.ndarray:
        """The degree of each ligament in the dual graph: the ligaments it meets."""
        return self.growth.degrees()

    def links(self) -> np.ndarray:
        """The pairs of ligaments that meet (an end of the first on the second)."""
        return self.growth.links()

    @property
    def volume_fraction(self) -> float:
        """The fraction of the face in the ligaments and the frame, exact (``1 - sum(void
        areas) / A``)."""
        return float(1.0 - self.growth.cell_areas().sum() / self.area)

    def orientation_tensor(self) -> np.ndarray:
        """The orientation tensor of the ligaments (2 x 2, trace 1, axes of the plane)."""
        return self.growth.orientation_tensor()

    @property
    def geometric_anisotropy(self) -> float:
        """``mu_1 / mu_2``, the ratio of the eigenvalues of the orientation tensor: 1 for
        ligaments of isotropic directions, 0 when they are all parallel (Both et al.
        2026)."""
        mu = np.linalg.eigvalsh(self.orientation_tensor())
        return float(mu[0] / mu[1]) if mu[1] > 0 else math.nan

    def statistics(self, min_tail: int = 50) -> dict:
        """The power-law tails of the thicknesses, lengths and degrees of the ligaments
        (:func:`morphogen.analysis.power_laws.power_law_tail`, degrees as integers): Both
        et al. find ``1 + 1/alpha``, 3 and 3 for their exponents. No thicknesses with a
        decay of 0 (one thickness)."""
        from morphogen.analysis.power_laws import power_law_tail

        tail = min(int(min_tail), max(2, self.n_ligaments // 2))
        out = {}
        if np.ptp(self.thicknesses) > 0:  # a decay of 0: one thickness, no tail
            out["thickness"] = power_law_tail(self.thicknesses, min_tail=tail)
        out["length"] = power_law_tail(self.lengths, min_tail=tail)
        out["degree"] = power_law_tail(self.degrees(), discrete=True, min_tail=tail)
        return out

    def convex_pieces(self) -> tuple[list[tuple], np.ndarray, np.ndarray]:
        """The convex pieces that partition the box: the ligaments (phase 1, cells 1 to N)
        and the voids (phase 0, cell 0) extruded along the axis into prisms, and the frame
        (phase 1, cell N + 1), for the exact meshes of
        :func:`morphogen.mesh.exact.polyhedral_mesh`."""
        from morphogen.models.scale_rich_3d import prism

        g = self.growth
        axes = plane_axes(self.axis)
        z0, z1 = float(self.domain.lower[self.axis]), float(self.domain.upper[self.axis])
        pieces, phase, cells = [], [], []
        for t, body in enumerate(g.bodies()):
            if len(body) >= 3 and polygon_area(body) > 0:
                pieces.append(prism(body, axes, self.axis, z0, z1))
                phase.append(1)
                cells.append(t + 1)
        for cell in g.cells():
            if len(cell) >= 3 and polygon_area(cell) > 0:
                pieces.append(prism(cell, axes, self.axis, z0, z1))
                phase.append(0)
                cells.append(0)
        if self.frame > 0:
            (u0, v0), (u1, v1) = self.plane
            f = self.frame
            for (a0, b0), (a1, b1) in (((u0, v0), (u0 + f, v1)), ((u1 - f, v0), (u1, v1)),
                                       ((u0 + f, v0), (u1 - f, v0 + f)),
                                       ((u0 + f, v1 - f), (u1 - f, v1))):  # fmt: skip
                rect = [(a0, b0), (a1, b0), (a1, b1), (a0, b1)]
                pieces.append(prism(rect, axes, self.axis, z0, z1))
                phase.append(1)
                cells.append(g.n + 1)
        return pieces, np.asarray(phase, dtype=np.int64), np.asarray(cells, dtype=np.int64)

    # -- the phases on points and voxels -------------------------------------------------

    def _index(self, reach: float) -> _Index:
        key = round(float(reach), 12)
        if key not in self._indices:
            lo, hi = self.plane
            self._indices.clear()
            self._indices[key] = _Index(self.growth, lo, hi, reach)
        return self._indices[key]

    def plane_field(self, points, reach: float = 0.0) -> np.ndarray:
        """At points of the plane: ``max_e (n_e . x - c_e)`` of the ligament holding them
        (negative: the distance to its nearest side, or less), or the least such value of
        the ligaments within ``reach`` (the distance to the nearest one, or less), capped at
        ``reach``; with a frame, the least of this and the signed distance to the inner
        side of the frame."""
        out = self._index(reach).field(points)
        if self.frame > 0:
            pts = np.asarray(points, dtype=float).reshape(-1, 2)
            lo, hi = self.plane
            inner = np.minimum(pts - lo, hi - pts).min(axis=1)
            out = np.minimum(out, inner - self.frame)
        return out

    def contains(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        b, c = plane_axes(self.axis)
        out = self.plane_field(pts[:, [b, c]]) < 0
        return out & np.all((pts >= self.domain.lower) & (pts <= self.domain.upper), axis=1)

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """:meth:`plane_field` of the points within ``reach`` (default: the diagonal of the
        box), the network extruded along the axis."""
        if not np.isfinite(reach):
            reach = float(self.domain.diagonal)
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        b, c = plane_axes(self.axis)
        return self.plane_field(pts[:, [b, c]], float(reach))

    def _image(self, grid: Grid, reach: float) -> np.ndarray:
        plane, shape = pixel_centres(grid, plane_axes(self.axis))
        return self.plane_field(plane, reach).astype(np.float32).reshape(shape)

    def _extruded(self, image2d: np.ndarray, grid: Grid) -> np.ndarray:
        out = np.broadcast_to(image2d[:, :, None], (*image2d.shape, grid.shape[self.axis]))
        return np.ascontiguousarray(np.moveaxis(out, 2, self.axis))

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = self._extruded((self._image(grid, 0.0) < 0).astype(np.uint8), grid)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        b, c = plane_axes(self.axis)
        fine_shape = list(grid.shape)
        fine_shape[b] *= ss
        fine_shape[c] *= ss
        fine = Grid(grid.domain, tuple(fine_shape))
        inside = (self._image(fine, 0.0) < 0).astype(np.float32)
        nb, nc = grid.shape[b], grid.shape[c]
        coarse = inside.reshape(nb, ss, nc, ss).mean(axis=(1, 3), dtype=np.float32)
        data = self._extruded(coarse, grid)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Negative in the ligaments, positive in the voids: the distance to the sides of
        the ligaments near them (a lower bound near their corners), within the largest
        half thickness and four voxels of them."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        thick = float(self.thicknesses.max()) if self.n_ligaments else 0.0
        reach = 0.5 * thick + 4.0 * float(np.max(grid.spacing))
        data = self._extruded(self._image(grid, reach), grid)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    # -- description --------------------------------------------------------------------

    def summary(self) -> dict:
        """The counts, thicknesses, lengths, degrees and anisotropy of the network (the
        columns of the tables of results)."""
        g = self.growth
        out = {"ligaments": g.n, "jammed": int(g.jammed), "volume_fraction": self.volume_fraction}
        if g.n:
            deg = g.degrees()
            out.update(
                {
                    "initial_thickness": float(g.thickness[0]),
                    "final_thickness": float(g.thickness[-1]),
                    "mean_length": float(g.lengths.mean()),
                    "mean_degree": float(deg.mean()),
                    "max_degree": int(deg.max()),
                    "geometric_anisotropy": self.geometric_anisotropy,
                }
            )
        return out

    def _geometry(self) -> dict:
        g = self.growth
        return {
            "axis": self.axis,
            "nucleation": g.nucleation.tolist(),
            "angle": g.angle.tolist(),
            "thickness": g.thickness.tolist(),
            "ends": g.ends.tolist(),
            "neighbours": g.neighbours.tolist(),
            "bodies": {"offsets": g.body_offsets.tolist(), "vertices": g.body_vertices.tolist()},
            "cells": {
                "offsets": g.cell_offsets.tolist(),
                "vertices": g.cell_vertices.tolist(),
                "faces": g.cell_faces.tolist(),
            },
            "jammed": bool(g.jammed),
            "attempts": int(g.attempts),
            "frame": self.frame,
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        bodies, cells = geometry["bodies"], geometry["cells"]
        growth = Growth(
            nucleation=_array(geometry["nucleation"], (2,)),
            angle=np.asarray(geometry["angle"], dtype=float).reshape(-1),
            thickness=np.asarray(geometry["thickness"], dtype=float).reshape(-1),
            ends=_array(geometry["ends"], (2, 2)),
            neighbours=_array(geometry["neighbours"], (2,), np.int64),
            body_offsets=np.asarray(bodies["offsets"], dtype=np.int64),
            body_vertices=_array(bodies["vertices"], (2,)),
            cell_offsets=np.asarray(cells["offsets"], dtype=np.int64),
            cell_vertices=_array(cells["vertices"], (2,)),
            cell_faces=np.asarray(cells["faces"], dtype=np.int64),
            jammed=bool(geometry.get("jammed", False)),
            attempts=int(geometry.get("attempts", 0)),
        )
        return cls(domain, geometry["axis"], growth, geometry.get("frame", 0.0), **kwargs)


# -- the model ---------------------------------------------------------------------------


@register
@dataclass(frozen=True)
class ScaleRichNetwork(Model):
    """Scale-rich network (Both et al. 2026): ligaments ``lambda_0 t^-alpha`` thick grown
    one after another from points of the voids until they meet: by default (``dimension:
    2``) in the plane of the axes other than ``axis`` and extruded along it (see
    :mod:`morphogen.models.scale_rich`); with ``dimension: 3``, plates or beams in the box
    (see :mod:`morphogen.models.scale_rich_3d`). The initial thickness is given, or solved on
    each realization for a relative density (``volume_fraction``). Orientations uniform,
    among ``angles`` (2D) or drawn from a law (3D), or polarized by a ``pole``; a gradient
    of the nucleation density. Non-periodic: the ligaments end on the boundary of the box
    (or on a frame)."""

    name = "scale_rich_network"
    phases: ClassVar[dict[int, str]] = {0: "voids", 1: "ligaments"}
    supports_periodic: ClassVar[bool] = False
    count_label = "ligaments"

    n_ligaments: int = param(500, doc="Number of ligaments T, plates or beams in 3D (fewer if "
                             "the network jams)", kind="int")  # fmt: skip
    dimension: int = param(
        2, doc="2: a network of the plane extruded along axis (prisms); 3: plates or beams "
        "in the box", kind="int",
    )  # fmt: skip
    element: str = param(
        "plates", doc="The ligaments in 3D: plates (closed cells) or beams (open cells, "
        "struts)", kind="str",
    )  # fmt: skip
    decay: float = param(0.5, doc="Exponent alpha of the thicknesses lambda_0 t^-alpha")
    initial_thickness: float | None = param(
        None, doc="Thickness lambda_0 of the first ligament (the diameter of a beam)",
        unit="length",
    )  # fmt: skip
    volume_fraction: float | None = param(
        None, doc="Relative density, reached on each realization (solves lambda_0)",
        kind="fraction",
    )  # fmt: skip
    min_spacing: float = param(
        0.0, doc="Least width epsilon of the voids beside a new ligament, at its nucleation "
        "point", unit="length",
    )  # fmt: skip
    angles: tuple[float, ...] | None = param(
        None, doc="In 2D: angles of the ligaments in the plane, drawn among these (degrees; "
        "default: uniform)", kind="floats", unit="deg",
    )  # fmt: skip
    orientation: OrientationDistribution | None = param(
        None, doc="In 3D: law of the normals of the plates or of the axes of the beams "
        "(default: isotropic)", kind="orientation",
    )  # fmt: skip
    angle_spread: float = param(
        0.0, doc="Standard deviation of the angles (3D: of the tilt of the directions) about "
        "those given or polarized", unit="deg",
    )  # fmt: skip
    pole: tuple[float, ...] | None = param(
        None, doc="Point polarizing the ligaments (two coordinates of the plane in 2D, three "
        "in 3D): they make pole_angle with the direction of the pole", kind="floats",
        unit="length",
    )  # fmt: skip
    pole_angle: float = param(
        0.0, doc="Angle of the polarized ligaments to the direction of the pole (0: radial, "
        "90: tangential; 3D: plates through it or on spheres about it, beams along it or "
        "around it, at a uniform azimuth)", unit="deg",
    )  # fmt: skip
    polarized_fraction: float = param(
        1.0, doc="Fraction of the ligaments, the first ones, polarized by the pole",
        kind="fraction",
    )  # fmt: skip
    nucleation_gradient: tuple[float, ...] | None = param(
        None, doc="Relative nucleation densities at the lower and upper faces along "
        "gradient_axis (at the pole and the farthest corner with radial_gradient)",
        kind="floats",
    )  # fmt: skip
    gradient_axis: int | None = param(
        None, doc="Axis of the gradient of nucleation (in 2D, of the plane; default: the "
        "first axis of the plane, x in 3D)", kind="int",
    )  # fmt: skip
    radial_gradient: bool = param(
        False, doc="The gradient of nucleation away from the pole", kind="bool"
    )
    frame: float = param(
        0.0, doc="Thickness of a frame of material around the network, on which its "
        "ligaments end (0: none): the sample as the cell of a periodic medium", unit="length",
    )  # fmt: skip
    axis: int = param(2, doc="In 2D: axis of the prisms (0: x, 1: y, 2: z)", kind="int")
    max_attempts: int = param(
        1000, doc="Draws in a row that do not fit, after which the network is jammed",
        kind="int",
    )  # fmt: skip

    def _validate(self) -> None:
        if self.n_ligaments < 1:
            raise ValueError("n_ligaments: at least one ligament")
        if self.dimension not in (2, 3):
            raise ValueError("dimension: 2 (a planar network extruded) or 3")
        if self.element not in ("plates", "beams"):
            raise ValueError("element: plates or beams")
        if self.dimension == 2 and self.element != "plates":
            raise ValueError("element: beams in 3D only (dimension: 3); the ligaments of a "
                             "planar network are extruded into plates")  # fmt: skip
        check_positive(self.decay, "decay", allow_zero=True)
        check_positive(self.frame, "frame", allow_zero=True)
        exactly_one("initial_thickness", self.initial_thickness,
                    "volume_fraction", self.volume_fraction)  # fmt: skip
        if self.initial_thickness is not None:
            check_positive(self.initial_thickness, "initial_thickness")
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        check_positive(self.min_spacing, "min_spacing", allow_zero=True)
        if self.angles is not None:
            if self.dimension == 3:
                raise ValueError("angles: in 2D only; in 3D, an orientation law")
            if not self.angles or not all(math.isfinite(a) for a in self.angles):
                raise ValueError("angles: a list of finite angles in degrees")
        if self.orientation is not None and self.dimension == 2:
            raise ValueError("orientation: in 3D only; in 2D, angles")
        check_positive(self.angle_spread, "angle_spread", allow_zero=True)
        if self.pole is not None:
            d = self.dimension
            if len(self.pole) != d or not all(math.isfinite(v) for v in self.pole):
                where = "of the plane" if d == 2 else "of space"
                raise ValueError(f"pole: the {d} coordinates of a point {where}")
        if not math.isfinite(self.pole_angle):
            raise ValueError("pole_angle: a finite angle in degrees")
        if not 0.0 <= self.polarized_fraction <= 1.0:
            raise ValueError("polarized_fraction: from 0 to 1")
        if self.polarized_fraction < 1.0 and self.pole is None:
            raise ValueError("polarized_fraction: the fraction polarized by a pole")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis: 0, 1 or 2")
        if self.gradient_axis is not None:
            axes = plane_axes(self.axis) if self.dimension == 2 else (0, 1, 2)
            if self.gradient_axis not in axes:
                where = "of the plane" if self.dimension == 2 else "of space"
                raise ValueError(f"gradient_axis: an axis {where}, {tuple(axes)}")
        if self.nucleation_gradient is not None:
            w = self.nucleation_gradient
            if len(w) != 2 or not all(math.isfinite(v) and v >= 0 for v in w) or max(w) <= 0:
                raise ValueError("nucleation_gradient: two non-negative densities, not both 0")
        if self.radial_gradient and (self.pole is None or self.nucleation_gradient is None):
            raise ValueError("radial_gradient: a pole and a nucleation_gradient")
        if self.max_attempts < 1:
            raise ValueError("max_attempts: at least 1")

    @property
    def isotropic(self) -> bool:
        """Uniform orientations and nucleation: the laws of :meth:`theory` hold."""
        return (self.angles is None and self.orientation is None and self.pole is None
                and self.nucleation_gradient is None)  # fmt: skip

    @property
    def plates(self) -> bool:
        """Plates in 3D (closed cells), or the extruded ligaments of a planar network."""
        return self.element == "plates"

    # -- the growth -----------------------------------------------------------------------

    def _orientation(self):
        """The angles of the ligaments of a planar network (:func:`grow`)."""
        if self.angles is None and self.pole is None:
            return None
        spread = math.radians(self.angle_spread)
        angles = None if self.angles is None else [math.radians(a) for a in self.angles]
        polar = 0 if self.pole is None else math.ceil(self.polarized_fraction * self.n_ligaments)
        turn = math.radians(self.pole_angle)
        pole = self.pole
        normal = NormalDist()

        def orientation(t, x, y, u, v):
            if t <= polar:
                base = math.atan2(pole[1] - y, pole[0] - x) + turn
            elif angles is not None:
                base = angles[min(int(u * len(angles)), len(angles) - 1)]
            else:
                return math.pi * u
            if spread > 0.0:
                base += spread * normal.inv_cdf(min(max(v, 1e-16), 1.0 - 1e-16))
            return base

        return orientation

    def _directions(self):
        """The normals of the plates or the axes of the beams in 3D (:func:`grow_plates`):
        polarized by the pole, at the angle ``beta`` (``pole_angle``) to its direction
        ``r`` and a uniform azimuth (plates: ``n . r = sin beta``, beams: ``d . r = cos
        beta``), drawn from the orientation law or uniform; tilted by a normal law of
        ``angle_spread``."""
        if self.orientation is None and self.pole is None:
            return None
        spread = math.radians(self.angle_spread)
        law = self.orientation
        polar = 0 if self.pole is None else math.ceil(self.polarized_fraction * self.n_ligaments)
        beta = math.radians(self.pole_angle)
        # the share of the direction of the pole in the normal of a plate, the axis of a beam
        along, side = math.cos(beta), math.sin(beta)
        if self.plates:
            along, side = side, along
        pole = None if self.pole is None else np.asarray(self.pole, dtype=float)

        def directions(t, points, gen):
            k = len(points)
            if t <= polar:
                r = pole - points
                norm = np.linalg.norm(r, axis=1, keepdims=True)
                r = np.where(norm > 0, r / np.where(norm > 0, norm, 1.0), (0.0, 0.0, 1.0))
                e1, e2 = orthonormal_basis(r)
                psi = 2.0 * np.pi * gen.random(k)[:, None]
                d = along * r + side * (np.cos(psi) * e1 + np.sin(psi) * e2)
            elif law is not None:
                d = np.asarray(law.sample(gen, k), dtype=float).reshape(k, 3)
            else:
                d = uniform_unit_vectors(gen, k)
            if spread > 0.0:
                e1, e2 = orthonormal_basis(d)
                g = spread * gen.standard_normal((k, 2))
                tilt = np.hypot(g[:, 0], g[:, 1])[:, None]
                way = (g[:, :1] * e1 + g[:, 1:] * e2) / np.where(tilt > 0, tilt, 1.0)
                d = np.cos(tilt) * d + np.sin(tilt) * way
            return d / np.linalg.norm(d, axis=1, keepdims=True)

        return directions

    def _density(self, lower, upper):
        """The relative density of nucleation at a point (2 or 3 coordinates)."""
        if self.nucleation_gradient is None:
            return None
        w0, w1 = (float(w) for w in self.nucleation_gradient)
        top = max(w0, w1)
        lower = [float(v) for v in lower]
        upper = [float(v) for v in upper]
        if self.radial_gradient:
            pole = [float(v) for v in self.pole]
            corners = itertools.product(*zip(lower, upper, strict=True))
            far = max(math.dist(c, pole) for c in corners)

            def density(*x):
                s = min(math.dist(x, pole) / far, 1.0) if far > 0 else 0.0
                return (w0 + (w1 - w0) * s) / top

            return density
        if self.dimension == 2:
            axes = plane_axes(self.axis)
            k = axes.index(axes[0] if self.gradient_axis is None else self.gradient_axis)
        else:
            k = 0 if self.gradient_axis is None else int(self.gradient_axis)
        lo, span = lower[k], upper[k] - lower[k]

        def density(*x):
            s = (x[k] - lo) / span
            return (w0 + (w1 - w0) * s) / top

        return density

    def _grow(self, lower, upper, thickness: float, entropy: int):
        decay = float(self.decay)

        def law(t):
            return thickness * t ** (-decay)

        common = {
            "density": self._density(lower, upper),
            "min_spacing": float(self.min_spacing),
            "max_attempts": int(self.max_attempts),
        }
        if self.dimension == 2:
            return grow(lower, upper, self.n_ligaments, law, entropy,
                        orientation=self._orientation(), **common)  # fmt: skip
        grow3 = grow_plates if self.plates else grow_beams
        return grow3(lower, upper, self.n_ligaments, law, entropy,
                     directions=self._directions(), **common)  # fmt: skip

    def _region(self, domain: Domain) -> tuple[np.ndarray, np.ndarray]:
        """The corners of the region of the network within the frame: the face of the box
        in the plane (2D), or the box."""
        if self.dimension == 2:
            b, c = plane_axes(self.axis)
            lower, upper = domain.lower[[b, c]], domain.upper[[b, c]]
        else:
            lower, upper = domain.lower, domain.upper
        lower, upper = lower + self.frame, upper - self.frame
        if not np.all(upper > lower):
            raise ValueError("frame: thinner than half the sides of the box")
        return lower, upper

    def _sides(self, domain: Domain) -> tuple[float, ...]:
        """The sides of the region of the network (within the frame)."""
        lower, upper = self._region(domain)
        return tuple(float(v) for v in upper - lower)

    def _total(self, domain: Domain) -> float:
        """The area of the face of the box (2D), or its volume."""
        if self.dimension == 2:
            b, c = plane_axes(self.axis)
            return float(domain.lengths[b] * domain.lengths[c])
        return float(domain.volume)

    def _relative_density(self, growth, domain: Domain) -> float:
        total = self._total(domain)
        if self.dimension == 2:
            return float(1.0 - growth.cell_areas().sum() / total)
        if self.plates:
            return float(1.0 - growth.cell_volumes().sum() / total)
        return ScaleRichBeams(domain, growth, frame=self.frame).volume_fraction

    def thickness_estimate(self, domain: Domain) -> float:
        """The initial thickness of the ``volume_fraction`` by the theory (uniform
        orientations and nucleation), ``rho`` the density of the network within the frame
        (of area ``A``, volume ``V``): in 2D, ``lambda_0 = 2 sqrt(A) (1 - sqrt(1 - rho)) /
        S`` (:func:`density_estimate`, :func:`length_sum`); with plates, ``lambda_0 = 3
        V^(1/3) (1 - (1 - rho)^(1/3)) / S`` (:func:`area_sum`); with beams, the mean field
        of :func:`beam_density_estimate`."""
        if self.volume_fraction is None:
            return float(self.initial_thickness)
        sides = self._sides(domain)
        inner = float(np.prod(sides))
        rho = 1.0 - (1.0 - self.volume_fraction) * self._total(domain) / inner
        if rho <= 0:
            raise ValueError("volume_fraction: below the fraction of the frame alone")
        if self.dimension == 2:
            s = length_sum(self.decay, self.n_ligaments, *sides)
            return 2.0 * math.sqrt(inner) * (1.0 - math.sqrt(1.0 - rho)) / s
        if self.plates:
            s = area_sum(self.decay, self.n_ligaments, *sides)
            return 3.0 * inner ** (1.0 / 3.0) * (1.0 - (1.0 - rho) ** (1.0 / 3.0)) / s
        return beam_thickness_estimate(rho, self.decay, self.n_ligaments, sides)

    def _solve(self, entropy: int, domain: Domain):
        """The initial thickness of the realization of ``entropy`` at the relative density:
        regula falsi (Illinois) from the estimate of the theory, to :data:`DENSITY_TOL`. The
        draws of each ligament do not depend on the thicknesses (:func:`grow`), but a
        nucleation point or an end close to a moved side may change the network from it on:
        the density may jump where the tolerance is not met, and the side of the jump
        closer to the target is kept."""
        target = float(self.volume_fraction)
        lower, upper = self._region(domain)
        size = float(np.max(domain.lengths))
        runs: dict[float, tuple] = {}

        def excess(lam: float) -> float:
            if lam not in runs:
                g = self._grow(lower, upper, lam, entropy)
                runs[lam] = (self._relative_density(g, domain) - target, g)
            return runs[lam][0]

        guess = self.thickness_estimate(domain)
        a, b = 0.9 * guess, 1.1 * guess
        while excess(a) > 0:
            a *= 0.7
        while excess(b) < 0:
            if runs[b][1].jammed or b > size:
                raise ValueError(
                    f"volume_fraction: {target} is beyond the network, which jams at about "
                    f"{excess(b) + target:.3f} (fewer ligaments, a smaller decay or no "
                    "min_spacing)"
                )
            a, b = b, 1.4 * b
        fa, fb = excess(a), excess(b)
        side = 0
        for _ in range(60):
            if min(abs(excess(a)), abs(excess(b))) <= DENSITY_TOL or b - a <= 1e-9 * guess:
                break
            c = (a * fb - b * fa) / (fb - fa)
            if not a < c < b:
                c = 0.5 * (a + b)
            fc = excess(c)
            if fc > 0:
                b, fb = c, fc
                if side == 1:
                    fa *= 0.5  # Illinois: the stale end weighs less
                side = 1
            else:
                a, fa = c, fc
                if side == -1:
                    fb *= 0.5
                side = -1
        lam = min((a, b), key=lambda x: abs(excess(x)))
        return float(lam), runs[lam][1]

    def generate(self, domain: Domain, seed: SeedLike = None) -> Microstructure:
        self._check_domain(domain)
        rng, seed_info = make_rng(seed)
        entropy = int(rng.integers(0, 2**63))
        if self.initial_thickness is not None:
            lam0 = float(self.initial_thickness)
            growth = self._grow(*self._region(domain), lam0, entropy)
        else:
            lam0, growth = self._solve(entropy, domain)
        if self.dimension == 2:
            micro = ScaleRichStructure(domain, self.axis, growth, frame=self.frame, model=self,
                                       seed=seed_info)  # fmt: skip
            micro.metadata["summary"] = self._summary(micro, lam0)
            return micro
        kind = ScaleRichPlates if self.plates else ScaleRichBeams
        micro = kind(domain, growth, frame=self.frame, model=self, seed=seed_info)
        micro.metadata["summary"] = self._summary_3d(micro, lam0)
        return micro

    @staticmethod
    def _summary(micro: ScaleRichStructure, lam0: float) -> dict:
        g = micro.growth
        out = {
            "ligaments": g.n,
            "jammed": int(g.jammed),
            "attempts": g.attempts,
            "initial_thickness": lam0,
            "volume_fraction": micro.volume_fraction,
            "voids": g.n_cells,
            "boundary_ends": int(np.count_nonzero(g.neighbours == BOX)),
        }
        if not g.n:
            return out
        lengths, degrees = g.lengths, g.degrees()
        mu, vectors = np.linalg.eigh(g.orientation_tensor())
        main = vectors[:, 1]  # the direction of the larger eigenvalue
        out.update(
            {
                "final_thickness": float(g.thickness[-1]),
                "mean_length": float(lengths.mean()),
                "max_length": float(lengths.max()),
                "mean_degree": float(degrees.mean()),
                "max_degree": int(degrees.max()),
                "geometric_anisotropy": float(mu[0] / mu[1]),
                "principal_angle": float(math.degrees(math.atan2(main[1], main[0])) % 180.0),
            }
        )
        return out

    @staticmethod
    def _summary_3d(micro, lam0: float) -> dict:
        g = micro.growth
        plates = isinstance(micro, ScaleRichPlates)
        out = {
            "ligaments": g.n,
            "jammed": int(g.jammed),
            "attempts": g.attempts,
            "initial_thickness": lam0,
            "volume_fraction": micro.volume_fraction,
        }
        if plates:
            out["voids"] = g.n_cells
            out["boundary_plates"] = int(np.count_nonzero(g.boundary_contacts()))
        else:
            out["boundary_ends"] = int(np.count_nonzero(g.neighbours == BOX))
        if not g.n:
            return out
        sizes = g.areas if plates else g.lengths
        degrees = g.degrees()
        mu = np.linalg.eigvalsh(g.orientation_tensor())
        name = "area" if plates else "length"
        out.update(
            {
                "final_thickness": float(g.thickness[-1]),
                f"mean_{name}": float(sizes.mean()),
                f"max_{name}": float(sizes.max()),
                "mean_degree": float(degrees.mean()),
                "max_degree": int(degrees.max()),
                "geometric_anisotropy": float(mu[0] / mu[2]),
            }
        )
        return out

    # -- theory ------------------------------------------------------------------------

    def expected_count(self, domain: Domain) -> float:
        return float(self.n_ligaments)

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The exponent ``1 + 1/alpha`` of the law of the thicknesses (exact, ``alpha >
        0``), and the asymptotic exponents of the tails of the sizes and degrees: 3 for
        the lengths and degrees of a planar network (Both et al. 2026), 5/2 for the areas
        and degrees of plates (``(2d - 1) / (d - 1)`` in ``d`` dimensions). With the
        domain, for uniform orientations and nucleation: in 2D, the mean length of the
        first ligament (:func:`first_length`, exact), the first order ``rho_1 = lambda_0
        S / sqrt(A)`` of the relative density (:func:`length_sum`), its mean-field
        estimate ``rho_1 (1 - rho_1 / 4)`` (:func:`density_estimate`), the initial
        thickness at which ``T`` ligaments jam (``rho_1 = 2``) and, for ``alpha > 1/2``,
        at which an endless growth jams (``S`` with ``zeta(alpha + 1/2)``: the boundary
        of the phase diagram of the paper); with plates, the mean area of the first plate
        (:func:`first_area`, exact), ``rho_1 = lambda_0 S / V^(1/3)`` (:func:`area_sum`),
        ``1 - (1 - rho_1 / 3)^3``, jammed at ``rho_1 = 3``, endless for ``alpha > 1/3``
        (``zeta(alpha + 2/3)``); with beams, the mean-field estimate of
        :func:`beam_density_estimate`. With a ``volume_fraction``, the initial thickness
        that the theory gives it. With a frame, the network fills the region within it,
        and the relative densities count the frame."""
        out: dict[str, float] = {}
        if self.decay > 0:
            out["thickness_exponent"] = 1.0 + 1.0 / self.decay
        if self.dimension == 2:
            out.update({"length_exponent": 3.0, "degree_exponent": 3.0})
        elif self.plates:
            out.update({"area_exponent": 2.5, "degree_exponent": 2.5})
        if domain is None or not self.isotropic:
            return out
        sides = self._sides(domain)
        inner = float(np.prod(sides)) / self._total(domain)  # the share within the frame
        lam0 = self.thickness_estimate(domain)
        if self.volume_fraction is not None:
            out["initial_thickness"] = lam0
        if self.dimension == 3 and not self.plates:
            rho = beam_density_estimate(lam0, self.decay, self.n_ligaments, sides)
            out["volume_fraction"] = 1.0 - inner + inner * rho
            return out
        d = self.dimension
        root = float(np.prod(sides)) ** (1.0 / d)  # sqrt(A) or V^(1/3)
        if d == 2:
            s = length_sum(self.decay, self.n_ligaments, *sides)
            out["first_length"] = first_length(*sides)
            first, constant, shift = out["first_length"] / root, LENGTH_CONSTANT, 0.5
        else:
            s = area_sum(self.decay, self.n_ligaments, *sides)
            out["first_area"] = first_area(*sides)
            first, constant, shift = out["first_area"] / root**2, AREA_CONSTANT, 1.0 - 1.0 / d
        out["jamming_thickness"] = d * root / s
        if self.decay > 1.0 - shift:
            from scipy.special import zeta

            endless = first + constant * (zeta(self.decay + shift) - 1.0)
            out["endless_jamming_thickness"] = float(d * root / endless)
        rho1 = lam0 * s / root
        out["volume_fraction_first_order"] = 1.0 - inner + inner * rho1
        if rho1 < d:
            out["volume_fraction"] = 1.0 - inner + inner * density_estimate(rho1, d)
        return out
