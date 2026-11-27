"""Unions of balls and of discs measured exactly: the four Minkowski functionals of a union
of balls in space and the three of a union of discs in a plane, without a grid and without
sampling.

The union of the balls ``B_i = B(c_i, R_i)`` is cut by the power diagram of their centres
weighted by ``w_i = R_i^2``: the cell ``P_i`` holds the points ``x`` whose power ``|x -
c_i|^2 - w_i`` to the ball ``i`` is the smallest. The part of the union in ``P_i`` is ``B_i
∩ P_i`` and the part of its boundary there is ``∂B_i ∩ P_i`` (a point of the sphere ``i``
in its cell is inside no other ball), so that (Edelsbrunner 1995)

* the volume is ``sum_i |B_i ∩ P_i|``: the cell is cut into the pyramids of apex ``c_i``
  over its faces (signed: a face seen from the outside of the cell counts negatively), and
  the ball cuts the pyramid over a face ``F`` at the height ``h`` in the cone over ``F ∩ D``
  (``D`` the disc of radius ``sqrt(R^2 - h^2)`` where the sphere meets the plane of the face)
  and the sector of the ball over ``F \\ D``: ``|h| |F ∩ D| / 3 + R^3 Ω(F \\ D) / 3``, with the
  solid angles ``Ω`` of the plane figures seen from ``c_i`` (triangles by the formula of
  Van Oosterom & Strackee 1983, sectors of the disc ``Δθ (1 - |h| / R)``);
* the surface is ``sum_i R_i^2 Ω_i``, the solid angle of the sphere ``i`` in its cell, ``sum_F
  ± Ω(F \\ D)``;
* the integral of mean curvature ``M = ∫ H dS`` (``H`` the mean of the principal
  curvatures) is that of the spherical parts, ``sum_i S_i / R_i``, less half the length of
  the circles where two spheres meet on the boundary times the angle ``γ`` between their
  normals there, ``cos γ = (R_i^2 + R_j^2 - d^2) / (2 R_i R_j)``: such a circle lies in the
  common face of the two cells, and its part on the boundary of the union is its part in
  that face;
* the Euler characteristic is that of the dual complex ``K`` of the union, the simplices of
  the regular (weighted Delaunay) triangulation whose balls have a common point in the
  intersection of their cells: by the nerve theorem the union has the homotopy type of
  ``K``, and ``χ = #vertices - #edges + #triangles - #tetrahedra``. A simplex of the
  triangulation is in ``K`` if its orthocentre ``z`` (the point of its affine hull at the
  same power ``ρ^2`` from its vertices) has ``ρ^2 <= 0`` and lies in the dual face of the
  simplex (it is *unattached*: no vertex of a coface is closer in power), or if it is a
  face of a simplex in ``K`` (Edelsbrunner & Mücke 1994, the alpha complex at ``α = 0``).

The regular triangulation is the lower convex hull of the centres lifted to ``(c_i, |c_i|^2 -
w_i)`` (Qhull, through SciPy). In a periodic box the balls are repeated across the faces
within a margin, the balls of the box are measured in their cells and the simplices of
``K`` whose barycentre lies in the box are counted once: the functionals of the union on
the torus, whose densities are those of the periodic realization. The margin is checked:
no ball beyond it could come closer in power to the orthocentre of a tetrahedron near the
box than its vertices (it is doubled until none can).

In the plane, the same with the discs: the area of ``D_i ∩ P_i`` (triangles from ``c_i`` to
the edges of the cell, cut by the disc into triangles and circular sectors), the length of
the circle ``i`` in its cell, and ``χ = #vertices - #edges + #triangles`` of the dual
complex. The sections of a union of balls by a plane are unions of discs: their
functionals are those of the sections, exactly.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.spatial import ConvexHull, QhullError

# ----------------------------------------------------------------------------------
# Plane figures: fans of triangles from a point, cut by a disc centred on it


@njit(cache=True)
def _solid_angle(bx, by, cx, cy, h):
    """Signed solid angle of the triangle ``(o, b, c)`` of the plane at the height ``h``
    above the apex, ``o`` the foot of the apex on the plane (Van Oosterom & Strackee
    1983)."""
    if h <= 0.0:
        return 0.0
    nb = math.sqrt(bx * bx + by * by + h * h)
    nc = math.sqrt(cx * cx + cy * cy + h * h)
    det = h * (bx * cy - by * cx)
    den = h * nb * nc + h * h * nc + h * h * nb + (bx * cx + by * cy + h * h) * h
    return 2.0 * math.atan2(det, den)


@njit(cache=True)
def _angle(ax, ay, bx, by):
    """Signed angle from the vector ``a`` to the vector ``b``."""
    return math.atan2(ax * by - ay * bx, ax * bx + ay * by)


@njit(cache=True)
def _fan(xs, ys, h, a):
    """The fan of triangles from the origin to the edges of a polygon (its vertices in
    order, closed implicitly), signed by their orientation: ``(area, solid angle, area in the
    disc, solid angle in the disc)`` of the polygon, the solid angles seen from the height
    ``h`` above the origin and the disc of radius ``a`` centred on the origin (``a = 0``: no
    disc)."""
    n = xs.shape[0]
    area = 0.0
    omega = 0.0
    area_in = 0.0
    omega_in = 0.0
    r = math.sqrt(h * h + a * a)
    sector = 1.0 - h / r if r > 0.0 else 0.0
    for k in range(n):
        px, py = xs[k], ys[k]
        qx, qy = xs[(k + 1) % n], ys[(k + 1) % n]
        area += 0.5 * (px * qy - py * qx)
        omega += _solid_angle(px, py, qx, qy, h)
        if a <= 0.0:
            continue
        # the segment p -> q cut by the circle |x| = a: inside between the roots t1 < t2
        dx, dy = qx - px, qy - py
        qa = dx * dx + dy * dy
        qb = 2.0 * (px * dx + py * dy)
        qc = px * px + py * py - a * a
        disc = qb * qb - 4.0 * qa * qc
        t1, t2 = 2.0, 2.0  # no root in [0, 1]: the segment outside
        if qa > 0.0 and disc > 0.0:
            s = math.sqrt(disc)
            t1 = (-qb - s) / (2.0 * qa)
            t2 = (-qb + s) / (2.0 * qa)
        lo = min(max(t1, 0.0), 1.0)
        hi = min(max(t2, 0.0), 1.0)
        if hi <= lo:  # wholly outside the disc: a sector of the disc
            d = _angle(px, py, qx, qy)
            area_in += 0.5 * a * a * d
            omega_in += d * sector
            continue
        ux, uy = px + lo * dx, py + lo * dy
        vx, vy = px + hi * dx, py + hi * dy
        if lo > 0.0:  # p -> u outside
            d = _angle(px, py, ux, uy)
            area_in += 0.5 * a * a * d
            omega_in += d * sector
        area_in += 0.5 * (ux * vy - uy * vx)  # u -> v inside
        omega_in += _solid_angle(ux, uy, vx, vy, h)
        if hi < 1.0:  # v -> q outside
            d = _angle(vx, vy, qx, qy)
            area_in += 0.5 * a * a * d
            omega_in += d * sector
    return area, omega, area_in, omega_in


@njit(cache=True)
def _arc_inside(xs, ys, a, orientation):
    """The angle of the circle of radius ``a`` about the origin that lies in the convex
    polygon (vertices in order, ``orientation`` +1 counterclockwise, -1 clockwise): ``2 pi``
    less the union of the arcs that the half-planes of its edges leave out."""
    n = xs.shape[0]
    starts = np.empty(2 * n)
    ends = np.empty(2 * n)
    m = 0
    two_pi = 2.0 * math.pi
    for k in range(n):
        px, py = xs[k], ys[k]
        qx, qy = xs[(k + 1) % n], ys[(k + 1) % n]
        ex, ey = qx - px, qy - py
        norm = math.sqrt(ex * ex + ey * ey)
        if norm == 0.0:
            continue
        nx, ny = -ey / norm * orientation, ex / norm * orientation  # inward normal
        c = nx * px + ny * py  # the edge: n . x >= c inside
        if c <= -a:
            continue
        if c >= a:
            return 0.0
        beta = math.acos(c / a)
        phi = math.atan2(ny, nx)
        s = (phi + beta) % two_pi  # left out: from phi + beta to phi + 2 pi - beta
        length = two_pi - 2.0 * beta
        e = s + length
        if e > two_pi:
            starts[m], ends[m] = s, two_pi
            m += 1
            starts[m], ends[m] = 0.0, e - two_pi
        else:
            starts[m], ends[m] = s, e
        m += 1
    if m == 0:
        return two_pi
    order = np.argsort(starts[:m])
    covered = 0.0
    cur_s, cur_e = starts[order[0]], ends[order[0]]
    for i in range(1, m):
        s, e = starts[order[i]], ends[order[i]]
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
    covered += cur_e - cur_s
    return max(two_pi - covered, 0.0)


@njit(cache=True)
def _faces(ptr, xs, ys, heights, radii, out):
    """For each face (its polygon ``xs[ptr[f]:ptr[f + 1]]`` in the coordinates of its plane
    about the foot of the centre of the ball, at the signed ``heights`` from it): the signed
    volume and solid angle of the ball in the pyramid over the face, and the angle of the
    circle of the sphere in the face (``out[f] = (volume, solid angle, arc)``)."""
    for f in range(ptr.shape[0] - 1):
        h = heights[f]
        r = radii[f]
        out[f, 0] = 0.0
        out[f, 1] = 0.0
        out[f, 2] = 0.0
        if h == 0.0:
            continue
        x = xs[ptr[f] : ptr[f + 1]]
        y = ys[ptr[f] : ptr[f + 1]]
        hh = abs(h)
        a = math.sqrt(r * r - hh * hh) if hh < r else 0.0
        area, omega, area_in, omega_in = _fan(x, y, hh, a)
        o = 1.0 if area >= 0.0 else -1.0
        sign = 1.0 if h > 0.0 else -1.0
        outer = o * (omega - omega_in)  # the solid angle of the face beyond the disc
        out[f, 0] = sign * (hh * o * area_in / 3.0 + r**3 * outer / 3.0)
        out[f, 1] = sign * outer
        if a > 0.0:
            out[f, 2] = _arc_inside(x, y, a, o)


@njit(cache=True)
def _cells_2d(ptr, xs, ys, radii, out):
    """For each cell of a disc (its polygon about the centre of the disc): the area of the
    disc in the cell and the angle of its circle there (``out[i] = (area, arc)``)."""
    for i in range(ptr.shape[0] - 1):
        out[i, 0] = 0.0
        out[i, 1] = 0.0
        if ptr[i + 1] - ptr[i] < 3:  # a disc covered by the others: no cell
            continue
        x = xs[ptr[i] : ptr[i + 1]]
        y = ys[ptr[i] : ptr[i + 1]]
        r = radii[i]
        area, _, area_in, _ = _fan(x, y, 0.0, r)
        o = 1.0 if area >= 0.0 else -1.0
        out[i, 0] = o * area_in
        out[i, 1] = _arc_inside(x, y, r, o)


# ----------------------------------------------------------------------------------
# Regular triangulations and their dual complexes


def _lower_hull(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The simplices of the regular triangulation of the weighted points: the facets of the
    lower convex hull of the lifted points ``(p, |p|^2 - w)``."""
    centre = points.mean(axis=0)
    p = points - centre
    lifted = np.column_stack([p, np.einsum("ij,ij->i", p, p) - weights])
    try:
        hull = ConvexHull(lifted)
    except QhullError as exc:  # flat or too few points
        raise ValueError(f"no regular triangulation of these balls ({exc})") from None
    # the normals of the lower facets point down along the lifted coordinate (the offset of
    # each facet equation is its last column)
    tops = hull.simplices[hull.equations[:, -2] < 0.0].astype(np.int64)
    # flat simplices, which Qhull makes of the coplanar points on the rim of the hull (the
    # outer images of a periodic box): left out, they are far from what is measured
    e = p[tops[:, 1:]] - p[tops[:, :1]]
    size = float(np.ptp(p, axis=0).max())
    flat = np.abs(np.linalg.det(e)) <= 1e-12 * size ** p.shape[1]
    return tops[~flat]


def _orthocentres(c: np.ndarray, w: np.ndarray, simplices: np.ndarray):
    """The orthocentres of simplices (rows of vertex indices) in their affine hulls and their
    powers ``ρ^2``: ``z = c_a + sum_k λ_k e_k``, ``2 e_k . (z - c_a) = |e_k|^2 + w_a - w_k``."""
    a = simplices[:, 0]
    if simplices.shape[1] == 1:
        return c[a].copy(), -w[a]
    e = c[simplices[:, 1:]] - c[a][:, None, :]  # (n, m, d)
    gram = 2.0 * np.einsum("nkd,nld->nkl", e, e)
    rhs = np.einsum("nkd,nkd->nk", e, e) + w[a][:, None] - w[simplices[:, 1:]]
    lam = np.linalg.solve(gram, rhs[..., None])[..., 0]
    z = c[a] + np.einsum("nk,nkd->nd", lam, e)
    return z, np.einsum("nd,nd->n", z - c[a], z - c[a]) - w[a]


def _power(c, w, index, z) -> np.ndarray:
    """The power of the points ``z`` to the balls ``index``."""
    d = z - c[index]
    return np.einsum("nd,nd->n", d, d) - w[index]


def _faces_of(simplices: np.ndarray, k: int):
    """The faces of ``k`` vertices of simplices: the unique faces (sorted rows), and for each
    incidence (simplex, face) the face, the simplex and the vertices left out."""
    m = simplices.shape[1]
    combos = list(itertools.combinations(range(m), k))
    faces = np.sort(simplices[:, combos], axis=2)  # (n, f, k)
    rest = [tuple(i for i in range(m) if i not in cmb) for cmb in combos]
    others = simplices[:, rest]  # (n, f, m - k)
    flat = faces.reshape(-1, k)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    owner = np.repeat(np.arange(len(simplices)), len(combos))
    return unique, inverse.reshape(-1), owner, others.reshape(len(flat), m - k)


def dual_complex(c: np.ndarray, w: np.ndarray, tops: np.ndarray, tol: float = 0.0):
    """The dual complex at ``α = 0`` of weighted points from the top simplices of their
    regular triangulation (tetrahedra in space, triangles in the plane): ``levels[k] =
    (simplices of k + 1 vertices, in K)`` for ``k = 0 .. dim``."""
    dim = tops.shape[1] - 1
    levels: list = [None] * (dim + 1)
    _, rho = _orthocentres(c, w, tops)
    inside = rho <= tol
    levels[dim] = (tops, inside)
    upper, upper_in = tops, inside
    for k in range(dim, 0, -1):  # the faces of k vertices
        faces, inverse, owner, others = _faces_of(upper, k)
        zf, rf = _orthocentres(c, w, faces)
        # attached: a vertex of a coface closer in power to the orthocentre
        attached = np.zeros(len(faces), dtype=bool)
        for col in range(others.shape[1]):
            closer = _power(c, w, others[:, col], zf[inverse]) < rf[inverse] - tol
            attached[inverse[closer]] = True
        member = ~attached & (rf <= tol)
        member[inverse[upper_in[owner]]] = True
        levels[k - 1] = (faces, member)
        upper, upper_in = faces, member
    return levels


# ----------------------------------------------------------------------------------
# Periodic images


def _extended(centres, radii, lower, lengths, periodic, margin):
    """The balls and, in a periodic box, their images within ``margin`` of it: the centres
    (the balls of the box first, brought into it), the radii and the number of balls of the
    box."""
    n = len(radii)
    if not periodic:
        return centres, radii, n
    centres = lower + np.mod(centres - lower, lengths)
    dims = len(lengths)
    parts_c, parts_r = [centres], [radii]
    upper = lower + lengths
    k = max(1, math.ceil(margin / float(np.min(lengths))))  # periods within the margin
    for shift in itertools.product(range(-k, k + 1), repeat=dims):
        if not any(shift):
            continue
        moved = centres + np.asarray(shift, dtype=float) * lengths
        keep = np.all((moved >= lower - margin) & (moved < upper + margin), axis=1)
        parts_c.append(moved[keep])
        parts_r.append(radii[keep])
    return np.concatenate(parts_c), np.concatenate(parts_r), n


def _ghosts(centres, radii):
    """Points far away and of a negative weight around a free union, which bound the cells of
    its balls without touching them."""
    lo = centres.min(axis=0) - radii.max()
    hi = centres.max(axis=0) + radii.max()
    mid = 0.5 * (lo + hi)
    far = 10.0 * float(np.max(hi - lo)) + 10.0 * float(radii.max()) + 1.0
    dims = centres.shape[1]
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=dims)))
    # moved off the cube (a fixed pattern), lest their lifts be cospherical
    jitter = np.sin(np.arange(corners.size, dtype=float).reshape(corners.shape) * 2.39996 + 1.0)
    return mid + far * (corners + 0.1 * jitter)


def _margin_ok(c, w, tops, z, rho, lower, upper, margin, n) -> bool:
    """Whether no ball beyond the margin could be closer in power to the orthocentre of a top
    simplex that the measures use (one of a ball of the box, or one whose barycentre is in
    the box) than its vertices: the reach of the orthocentre to the outside of the margin,
    less the largest radius, is at least its ``ρ``."""
    bary = c[tops].mean(axis=1)
    sel = np.any(tops < n, axis=1) | np.all((bary >= lower) & (bary < upper), axis=1)
    zs = z[sel]
    reach = np.minimum(zs - (lower - margin), (upper + margin) - zs).min(axis=1)
    wmax = float(w.max())
    return bool(np.all((reach > 0.0) & (reach**2 - wmax >= rho[sel])))


# ----------------------------------------------------------------------------------
# Results


@dataclass
class BallUnion:
    """The Minkowski functionals of a union of balls: its ``volume``, the area of its
    boundary ``surface``, its integral of mean curvature ``mean_curvature`` (``∫ H dS``, ``4 pi
    R`` for one ball) and its Euler characteristic ``euler`` (components - tunnels +
    cavities), on the torus of a periodic box of volume ``box_volume`` (None: the union in
    space)."""

    volume: float
    surface: float
    mean_curvature: float
    euler: int
    n_balls: int
    box_volume: float | None = None

    def _density(self, value: float) -> float:
        if self.box_volume is None:
            raise ValueError("densities need the box of a periodic union")
        return value / self.box_volume

    @property
    def volume_fraction(self) -> float:
        return self._density(self.volume)

    @property
    def specific_surface(self) -> float:
        return self._density(self.surface)

    @property
    def mean_curvature_density(self) -> float:
        """``M_V``, as ``theory()['mean_curvature_density']`` of the models."""
        return self._density(self.mean_curvature)

    @property
    def euler_density(self) -> float:
        return self._density(self.euler)

    def densities(self) -> dict[str, float]:
        """The four densities, named as in the theory of the models."""
        return {
            "volume_fraction": self.volume_fraction,
            "specific_surface": self.specific_surface,
            "mean_curvature_density": self.mean_curvature_density,
            "euler_density": self.euler_density,
        }


@dataclass
class DiscUnion:
    """The Minkowski functionals of a union of discs: its ``area``, the length of its
    boundary ``perimeter`` and its Euler characteristic ``euler`` (components - holes), on
    the torus of a periodic rectangle of area ``box_area`` (None: the union in the plane)."""

    area: float
    perimeter: float
    euler: int
    n_discs: int
    box_area: float | None = None

    def _density(self, value: float) -> float:
        if self.box_area is None:
            raise ValueError("densities need the rectangle of a periodic union")
        return value / self.box_area

    @property
    def area_fraction(self) -> float:
        return self._density(self.area)

    @property
    def boundary_density(self) -> float:
        """``L_A``: the length of the boundary per unit area."""
        return self._density(self.perimeter)

    @property
    def euler_density(self) -> float:
        """``χ_A``: the Euler characteristic per unit area."""
        return self._density(self.euler)

    def densities(self) -> dict[str, float]:
        return {
            "area_fraction": self.area_fraction,
            "boundary_density": self.boundary_density,
            "euler_density": self.euler_density,
        }


# ----------------------------------------------------------------------------------
# The union of balls


def _box(box, dims):
    if box is None:
        return None, None
    lower, upper = (np.asarray(v, dtype=float).reshape(dims) for v in box)
    if np.any(upper <= lower):
        raise ValueError("box: the upper corner above the lower one")
    return lower, upper


def _prepare(centres, radii, dims, name):
    c = np.ascontiguousarray(centres, dtype=float).reshape(-1, dims)
    r = np.ascontiguousarray(radii, dtype=float).reshape(-1)
    if np.ndim(radii) == 0:
        r = np.full(len(c), float(radii))
    if len(r) != len(c):
        raise ValueError(f"{name}: one radius per centre")
    if np.any(r <= 0.0) or not np.all(np.isfinite(r)):
        raise ValueError(f"{name}: positive radii")
    return c, r


def _triangulated(c0, r0, box, periodic, dims, margin):
    """The weighted points (with their images or ghosts), their regular triangulation, the
    orthocentres of its top simplices and the number of balls of the box."""
    lower, upper = _box(box, dims)
    if periodic:
        lengths = upper - lower
        spacing = (np.prod(lengths) / max(len(r0), 1)) ** (1.0 / dims)
        margin = float(margin) if margin is not None else 2.0 * max(spacing, 2.0 * r0.max())
        # few balls in a box have large cells: images of several periods, if need be
        limit = 3.0 * float(lengths.max())
        while True:
            reach = min(margin, limit)
            c, r, n = _extended(c0, r0, lower, lengths, True, reach)
            w = r * r
            tops = _lower_hull(c, w)
            z, rho = _orthocentres(c, w, tops)
            if _margin_ok(c, w, tops, z, rho, lower, upper, reach, n):
                return c, w, tops, z, rho, n, lower, upper
            if reach >= limit:
                raise ValueError(
                    "the periodic box is too small for these balls: their power cells reach "
                    "beyond the images of three periods"
                )
            margin *= 2.0
    n = len(r0)
    ghosts = _ghosts(c0, r0)
    c = np.concatenate([c0, ghosts])
    w = np.concatenate([r0 * r0, -np.ones(len(ghosts))])
    tops = _lower_hull(c, w)
    z, rho = _orthocentres(c, w, tops)
    return c, w, tops, z, rho, n, lower, upper


def _euler(c, levels, n, periodic, lower, upper) -> int:
    """``sum_k (-1)^k #(simplices of k + 1 vertices in K)``, counting in a periodic box the
    simplices whose barycentre is in it, and leaving out the ghosts of a free union."""
    total = 0
    for k, (simplices, member) in enumerate(levels):
        chosen = simplices[member]
        if periodic:
            bary = c[chosen].mean(axis=1)
            keep = np.all((bary >= lower) & (bary < upper), axis=1)
        else:
            keep = np.all(chosen < n, axis=1)
        total += (-1) ** k * int(keep.sum())
    return total


def _plane_basis(n: np.ndarray):
    """Two unit vectors completing the unit normals ``n`` (rows) into direct frames."""
    helper = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(n, u)
    return u, v


def _cell_faces(c, w, tops, z, balls):
    """The faces of the power cells of the ``balls``: for each edge ``(i, j)`` of the
    triangulation with ``i`` among them, the orthocentres of the tetrahedra around it in
    order about the edge, as polygons in the coordinates of the plane of the face about the
    foot of ``c_i``, with the signed height of that plane from ``c_i``. Returns the pointers
    and coordinates of the polygons, the heights, the radii and the pairs ``(i, j)``."""
    pairs = list(itertools.combinations(range(4), 2))
    edges = tops[:, pairs].reshape(-1, 2)
    tet = np.repeat(np.arange(len(tops)), len(pairs))
    both = np.concatenate([edges, edges[:, ::-1]])  # each face seen from both of its balls
    tet = np.concatenate([tet, tet])
    keep = np.isin(both[:, 0], balls)
    both, tet = both[keep], tet[keep]
    order = np.lexsort((both[:, 1], both[:, 0]))
    both, tet = both[order], tet[order]
    new = np.ones(len(both), dtype=bool)
    new[1:] = np.any(both[1:] != both[:-1], axis=1)
    starts = np.flatnonzero(new)
    ptr = np.append(starts, len(both))
    pair = both[starts]
    i, j = pair[:, 0], pair[:, 1]
    d = c[j] - c[i]
    dist = np.linalg.norm(d, axis=1)
    normal = d / dist[:, None]
    h = (dist * dist + w[i] - w[j]) / (2.0 * dist)
    foot = c[i] + h[:, None] * normal
    u, v = _plane_basis(normal)
    face = np.repeat(np.arange(len(pair)), np.diff(ptr))
    rel = z[tet] - foot[face]
    xs = np.einsum("nd,nd->n", rel, u[face])
    ys = np.einsum("nd,nd->n", rel, v[face])
    # in order about the centroid of each (convex) polygon
    count = np.diff(ptr)
    cx = np.bincount(face, xs, len(pair)) / count
    cy = np.bincount(face, ys, len(pair)) / count
    theta = np.arctan2(ys - cy[face], xs - cx[face])
    order = np.lexsort((theta, face))
    return ptr, xs[order], ys[order], h, pair, dist


def union_of_balls(centres, radii, box=None, *, periodic: bool = False, margin=None) -> BallUnion:
    """The four Minkowski functionals of the union of the balls of ``centres`` and ``radii``
    (see the module): in space, or on the torus of the periodic ``box`` (``(lower, upper)``
    corners; ``periodic=True``), the balls then repeated across its faces within
    ``margin`` (by default three mean spacings of the centres or four radii, doubled while
    too small)."""
    c0, r0 = _prepare(centres, radii, 3, "union_of_balls")
    if periodic and box is None:
        raise ValueError("a periodic union needs its box")
    if len(r0) == 0:
        lower, upper = _box(box, 3)
        vol = float(np.prod(upper - lower)) if periodic else None
        return BallUnion(0.0, 0.0, 0.0, 0, 0, vol)
    c, w, tops, z, _, n, lower, upper = _triangulated(c0, r0, box, periodic, 3, margin)
    levels = dual_complex(c, w, tops, tol=1e-12 * float(w.max()))
    euler = _euler(c, levels, n, periodic, lower, upper)
    balls = np.arange(n)
    ptr, xs, ys, h, pair, dist = _cell_faces(c, w, tops, z, balls)
    r = np.sqrt(np.maximum(w, 0.0))  # the ghosts of a free union: no ball
    out = np.empty((len(pair), 3))
    _faces(ptr, xs, ys, h, r[pair[:, 0]], out)
    i, j = pair[:, 0], pair[:, 1]
    volume = float(out[:, 0].sum())
    solid = np.bincount(i, out[:, 1], len(c))
    surface_i = r * r * solid
    surface = float(surface_i[:n].sum())
    # the circles where two spheres meet on the boundary, each seen from both of its balls:
    # half the angle of their normals times their length, half from each side
    real = w[j] > 0.0
    rj = np.where(real, r[j], 1.0)
    cos_gamma = np.clip((w[i] + w[j] - dist * dist) / (2.0 * r[i] * rj), -1.0, 1.0)
    a = np.sqrt(np.maximum(w[i] - h * h, 0.0))
    edges = 0.25 * float(np.sum(np.where(real, np.arccos(cos_gamma) * a * out[:, 2], 0.0)))
    mean_curvature = float(np.sum(surface_i[:n] / r[:n]) - edges)
    box_volume = float(np.prod(upper - lower)) if periodic else None
    return BallUnion(volume, surface, mean_curvature, euler, len(r0), box_volume)


# ----------------------------------------------------------------------------------
# The union of discs


def union_of_discs(centres, radii, box=None, *, periodic: bool = False, margin=None) -> DiscUnion:
    """The three Minkowski functionals of the union of the discs of ``centres`` (two
    coordinates) and ``radii``: in the plane, or on the torus of the periodic rectangle
    ``box`` (``(lower, upper)`` corners; ``periodic=True``)."""
    c0, r0 = _prepare(centres, radii, 2, "union_of_discs")
    if periodic and box is None:
        raise ValueError("a periodic union needs its box")
    if len(r0) == 0:
        lower, upper = _box(box, 2)
        return DiscUnion(0.0, 0.0, 0, 0, float(np.prod(upper - lower)) if periodic else None)
    c, w, tops, z, _, n, lower, upper = _triangulated(c0, r0, box, periodic, 2, margin)
    levels = dual_complex(c, w, tops, tol=1e-12 * float(w.max()))
    euler = _euler(c, levels, n, periodic, lower, upper)
    # the cell of each disc: the orthocentres of the triangles around it, in order
    vert = tops.reshape(-1)
    tri = np.repeat(np.arange(len(tops)), 3)
    keep = vert < n
    vert, tri = vert[keep], tri[keep]
    rel = z[tri] - c[vert]
    count = np.bincount(vert, minlength=n)
    cx = np.bincount(vert, rel[:, 0], n) / np.maximum(count, 1)
    cy = np.bincount(vert, rel[:, 1], n) / np.maximum(count, 1)
    theta = np.arctan2(rel[:, 1] - cy[vert], rel[:, 0] - cx[vert])
    order = np.lexsort((theta, vert))
    ptr = np.concatenate([[0], np.cumsum(count)])
    r = np.sqrt(w[:n])
    out = np.empty((n, 2))
    _cells_2d(ptr, rel[order, 0].copy(), rel[order, 1].copy(), r, out)
    box_area = float(np.prod(upper - lower)) if periodic else None
    return DiscUnion(float(out[:, 0].sum()), float(np.sum(r * out[:, 1])), euler, len(r0), box_area)


# ----------------------------------------------------------------------------------
# Realizations


def _spheres_of(micro):
    """The balls of a union of spheres (a realization of the Boolean models, packings...)."""
    spheres = getattr(micro, "spheres", None)
    segments = getattr(micro, "segments", None)
    if spheres is None or (segments is not None and len(segments)):
        raise ValueError(
            "exact functionals: a union of spheres only (Boolean spheres, packings...); the "
            "other structures are measured by sampling (morphogen.morphology.measures)"
        )
    if getattr(micro, "implicit_phases", lambda: [0, 1])() != [0, 1]:
        raise ValueError("exact functionals: a union of spheres of one phase")
    return spheres.centers, spheres.radii


def minkowski_functionals(micro) -> BallUnion:
    """The exact Minkowski functionals of a realization that is a union of spheres, on the
    torus of its periodic box (densities: :meth:`BallUnion.densities`) or, in a box that is
    not periodic, of the union of its spheres in space (whole, the box not cutting them)."""
    centres, radii = _spheres_of(micro)
    domain = micro.domain
    box = (domain.lower, domain.upper)
    return union_of_balls(centres, radii, box, periodic=domain.periodic)


def section_discs(micro, axis: int = 2, position: float | None = None):
    """The discs of the section of a union of spheres by the plane normal to ``axis`` at
    ``position`` (default: the middle of the box): their centres in the coordinates of the
    two other axes, their radii, and the rectangle of the section."""
    centres, radii = _spheres_of(micro)
    domain = micro.domain
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis: 0, 1 or 2")
    lo, hi = domain.lower, domain.upper
    pos = 0.5 * (lo[axis] + hi[axis]) if position is None else float(position)
    others = [k for k in range(3) if k != axis]
    d = centres[:, axis] - pos
    if domain.periodic:  # the nearest image along the axis
        length = float(domain.lengths[axis])
        d = d - length * np.round(d / length)
    cut = np.abs(d) < radii
    return (
        centres[cut][:, others],
        np.sqrt(radii[cut] ** 2 - d[cut] ** 2),
        (lo[others], hi[others]),
    )


def section_functionals(micro, axis: int = 2, position: float | None = None) -> DiscUnion:
    """The exact Minkowski functionals of the section of a union of spheres by a plane of the
    box (:func:`section_discs`), on the torus of the rectangle of the section for a periodic
    box."""
    centres, radii, box = section_discs(micro, axis, position)
    return union_of_discs(centres, radii, box, periodic=micro.domain.periodic)
