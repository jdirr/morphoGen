"""Scale-rich networks in three dimensions: plates or beams of power-law thicknesses grown
one after another until they meet (Both et al. 2026, Fig. 6a and Supplementary S2.5-S2.7).

The sequential growth of :mod:`morphogen.models.scale_rich` extends to space by adding
non-crossing plates or beams, ``lambda_t = lambda_0 t^-alpha`` thick:

* **plates**: the plate ``t`` nucleates at a point drawn uniformly in the voids, with a
  normal ``n_t`` (uniform on the sphere, or drawn from a law), and grows in its plane until
  it meets other plates or the box. The voids are convex polyhedra (at first the box),
  bounded by faces of plates and of the box; the plate is the slab ``|n_t . x - c_t| <=
  lambda_t / 2`` clipped by the void of its nucleation point, which it splits into two
  convex voids. Plates and voids partition the box exactly (the relative density is
  exact), the plates never overlap, and every void is a closed cell. The *rim* of a plate
  is the section of its void by its mid-plane: a convex polygon whose edges lie on the
  faces of the plates (or of the box) on which the plate ends, its neighbours. The area
  of a plate is that of its rim.
* **beams**: the beam ``t`` (a cylinder of diameter ``lambda_t`` with round ends)
  nucleates at a point of the voids where it clears the other beams and the box, along an
  axis ``d_t``, and grows both ways until its axis reaches the surface of another beam
  (a T-junction), or passes within contact of one (the axis ends at its closest approach,
  a lateral junction), or reaches the box. Beams do not split space: their voids form one
  connected, open cell.

A plate (beam) fits when its void holds the points ``p +- (lambda_t + epsilon) / 2 n_t``
(when the ball of radius ``(lambda_t + epsilon) / 2`` about ``p`` clears the beams and the
box); ``max_attempts`` draws in a row that do not fit jam the network. Each element draws
its nucleation points and directions from its own generator, so that the network moves
little when ``lambda_0`` changes (see :func:`morphogen.models.scale_rich.grow`).

Laws: the thicknesses as in the plane, ``P(lambda) ~ lambda^-(1 + 1/alpha)``; a plate
splits a void of volume about ``V_v / t``, so that its area decays as ``t^-2/3`` and the
areas have the tail ``P(a) ~ a^-5/2``, as the degrees (a plate is met by the later ones in
proportion to its area): ``(2d - 1) / (d - 1)`` in ``d`` dimensions, 3 in the plane. The
relative density of plates is exact, and in closed form to a mean field (:func:`first_area`,
:func:`area_sum`, :meth:`morphogen.models.ScaleRichNetwork.theory`); that of beams, the
volume of the cylinders on their axes, follows the mean field of their free paths
(:func:`beam_density_estimate`). Orientation laws, poles, gradients and frames: see
:class:`morphogen.models.ScaleRichNetwork`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit, prange

from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, register_kind
from morphogen.orientations import uniform_unit_vectors

#: Label of the faces of the voids on the boundary of the box (plates: their index)
BOX = -1
#: ``lim (t / V)^(2/3) E[a_t]``: the mean area of the plate ``t`` of a thin network
#: (``lambda_0 -> 0``) in a box of volume ``V``, measured with uniform normals: 1.2363
#: (standard error 0.0005) over the plates ``t >= 2000`` of 120 realizations of 20,000 in
#: a cube. The first plate is ``a_0`` (:func:`first_area`, 0.9412 in a cube), the next
#: ones approach the limit faster than in the plane (1.08, 1.13 at ``t`` = 2, 3; 1.24 from
#: ``t`` = 10 on)
AREA_CONSTANT = 1.236
#: Uniform draws of a nucleation point: its three coordinates and its rejection by the
#: density of nucleation
_DRAWS = 4
#: Points (``2^k``) of the quasi-Monte Carlo estimate of the relative density of beams, the
#: volume of their union in the box (a scrambled Sobol set of a fixed seed, the same for
#: every network): within 1.5e-4 of 2^22 points over 8 networks (5.8e-4 with 2^18)
UNION_POINTS_LOG2 = 20

# -- convex polyhedra: faces (convex polygons counter-clockwise about their outward unit
# normal), each with the plane n . x = c that bears it and the label of what lies beyond --


@njit(cache=True)
def _put(out, k, start, x, y, z, tol):
    """Appends a vertex to the face that starts at ``start``, unless it repeats the last."""
    if (k > start and abs(out[k - 1, 0] - x) <= tol and abs(out[k - 1, 1] - y) <= tol
            and abs(out[k - 1, 2] - z) <= tol):  # fmt: skip
        return k
    out[k, 0], out[k, 1], out[k, 2] = x, y, z
    return k + 1


@njit(cache=True)
def _basis(nx, ny, nz):
    """Two unit vectors ``e1, e2`` of the plane normal to ``n`` with ``e1 x e2 = n``."""
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if ax <= ay and ax <= az:
        hx, hy, hz = 1.0, 0.0, 0.0
    elif ay <= az:
        hx, hy, hz = 0.0, 1.0, 0.0
    else:
        hx, hy, hz = 0.0, 0.0, 1.0
    e1x, e1y, e1z = ny * hz - nz * hy, nz * hx - nx * hz, nx * hy - ny * hx  # n x h
    s = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    e1x, e1y, e1z = e1x / s, e1y / s, e1z / s
    e2x, e2y, e2z = ny * e1z - nz * e1y, nz * e1x - nx * e1z, nx * e1y - ny * e1x  # n x e1
    return e1x, e1y, e1z, e2x, e2y, e2z


@njit(cache=True)
def _cut(verts, fofs, labels, normals, planes, nx, ny, nz, c, tol):
    """The faces of a convex polyhedron clipped to the half-space ``n . x <= c``
    (Sutherland-Hodgman on each face; the faces left with fewer than three vertices go),
    and the section of the polyhedron by the plane ``n . x = c``: its vertices,
    counter-clockwise about ``n``, and the label of the face on which each edge (from a
    vertex to the next) lies."""
    nf = fofs.shape[0] - 1
    out = np.empty((verts.shape[0] + nf, 3))
    ofs = np.zeros(nf + 1, dtype=np.int64)
    lab = np.empty(nf, dtype=np.int64)
    nrm = np.empty((nf, 3))
    pln = np.empty(nf)
    seg = np.empty((nf, 6))  # in each cut face: where its boundary enters the half-space,
    seg_lab = np.empty(nf, dtype=np.int64)  # then where it leaves it (the section's way)
    k = g = s = 0
    for f in range(nf):
        a = fofs[f]
        m = fofs[f + 1] - a
        start = k
        got_in = got_out = False
        ix = iy = iz = ox = oy = oz = 0.0
        for j in range(m):
            i0 = a + j
            i1 = a + (j + 1) % m
            d0 = nx * verts[i0, 0] + ny * verts[i0, 1] + nz * verts[i0, 2] - c
            d1 = nx * verts[i1, 0] + ny * verts[i1, 1] + nz * verts[i1, 2] - c
            if d0 <= 0.0:
                k = _put(out, k, start, verts[i0, 0], verts[i0, 1], verts[i0, 2], tol)
                if d1 > 0.0:  # leaves the half-space
                    t = d0 / (d0 - d1)
                    ox = verts[i0, 0] + t * (verts[i1, 0] - verts[i0, 0])
                    oy = verts[i0, 1] + t * (verts[i1, 1] - verts[i0, 1])
                    oz = verts[i0, 2] + t * (verts[i1, 2] - verts[i0, 2])
                    got_out = True
                    k = _put(out, k, start, ox, oy, oz, tol)
            elif d1 <= 0.0:  # enters it
                t = d0 / (d0 - d1)
                ix = verts[i0, 0] + t * (verts[i1, 0] - verts[i0, 0])
                iy = verts[i0, 1] + t * (verts[i1, 1] - verts[i0, 1])
                iz = verts[i0, 2] + t * (verts[i1, 2] - verts[i0, 2])
                got_in = True
                k = _put(out, k, start, ix, iy, iz, tol)
        if (k - start >= 2 and abs(out[k - 1, 0] - out[start, 0]) <= tol
                and abs(out[k - 1, 1] - out[start, 1]) <= tol
                and abs(out[k - 1, 2] - out[start, 2]) <= tol):  # fmt: skip
            k -= 1  # the last vertex repeats the first
        if k - start >= 3:
            ofs[g + 1] = k
            lab[g] = labels[f]
            nrm[g, 0], nrm[g, 1], nrm[g, 2] = normals[f, 0], normals[f, 1], normals[f, 2]
            pln[g] = planes[f]
            g += 1
        else:
            k = start
        if got_in and got_out and (abs(ix - ox) > tol or abs(iy - oy) > tol
                                   or abs(iz - oz) > tol):  # fmt: skip
            seg[s, 0], seg[s, 1], seg[s, 2] = ix, iy, iz
            seg[s, 3], seg[s, 4], seg[s, 5] = ox, oy, oz
            seg_lab[s] = labels[f]
            s += 1
    # the section: the segments of the cut faces, in the order of their angles about n
    cap = np.empty((0, 3))
    cap_lab = np.empty(0, dtype=np.int64)
    if s >= 3:
        e1x, e1y, e1z, e2x, e2y, e2z = _basis(nx, ny, nz)
        gx = gy = gz = 0.0
        for q in range(s):
            gx += seg[q, 0] + seg[q, 3]
            gy += seg[q, 1] + seg[q, 4]
            gz += seg[q, 2] + seg[q, 5]
        gx, gy, gz = gx / (2 * s), gy / (2 * s), gz / (2 * s)
        angle = np.empty(s)
        for q in range(s):
            mx = 0.5 * (seg[q, 0] + seg[q, 3]) - gx
            my = 0.5 * (seg[q, 1] + seg[q, 4]) - gy
            mz = 0.5 * (seg[q, 2] + seg[q, 5]) - gz
            angle[q] = math.atan2(mx * e2x + my * e2y + mz * e2z, mx * e1x + my * e1y + mz * e1z)
        order = np.argsort(angle)
        cap = np.empty((s, 3))
        cap_lab = np.empty(s, dtype=np.int64)
        for q in range(s):
            cap[q, 0], cap[q, 1], cap[q, 2] = seg[order[q], 0], seg[order[q], 1], seg[order[q], 2]
            cap_lab[q] = seg_lab[order[q]]
    return out[:k], ofs[: g + 1], lab[:g], nrm[:g], pln[:g], cap, cap_lab


@njit(cache=True)
def clip_polyhedron(verts, fofs, labels, normals, planes, nx, ny, nz, c, label, tol):
    """The part of a convex polyhedron where ``n . x <= c``: its faces, the section by the
    plane (when it is a polygon) labelled ``label`` (``n`` a unit vector)."""
    v, o, lab, nrm, pln, cap, _ = _cut(verts, fofs, labels, normals, planes, nx, ny, nz, c, tol)
    if cap.shape[0] < 3:
        return v, o, lab, nrm, pln
    nf = o.shape[0] - 1
    v2 = np.empty((v.shape[0] + cap.shape[0], 3))
    v2[: v.shape[0]] = v
    v2[v.shape[0] :] = cap
    o2 = np.empty(nf + 2, dtype=np.int64)
    o2[: nf + 1] = o
    o2[nf + 1] = v2.shape[0]
    lab2 = np.empty(nf + 1, dtype=np.int64)
    lab2[:nf] = lab
    lab2[nf] = label
    nrm2 = np.empty((nf + 1, 3))
    nrm2[:nf] = nrm
    nrm2[nf, 0], nrm2[nf, 1], nrm2[nf, 2] = nx, ny, nz
    pln2 = np.empty(nf + 1)
    pln2[:nf] = pln
    pln2[nf] = c
    return v2, o2, lab2, nrm2, pln2


@njit(cache=True)
def section(verts, fofs, labels, normals, planes, nx, ny, nz, c, tol):
    """The section of a convex polyhedron by the plane ``n . x = c``: its vertices,
    counter-clockwise about ``n``, and the label of the face of each edge."""
    out = _cut(verts, fofs, labels, normals, planes, nx, ny, nz, c, tol)
    return out[5], out[6]


@njit(cache=True)
def polyhedron_volume(verts, fofs, normals, planes) -> float:
    """Volume of a convex polyhedron: ``sum_f (c_f - n_f . x_0) A_f / 3`` (the cones of
    its faces from a vertex ``x_0``)."""
    nf = fofs.shape[0] - 1
    if nf < 4:
        return 0.0
    x0, y0, z0 = verts[0, 0], verts[0, 1], verts[0, 2]
    v = 0.0
    for f in range(nf):
        a, b = fofs[f], fofs[f + 1]
        sx = sy = sz = 0.0
        for j in range(a + 1, b - 1):
            ux = verts[j, 0] - verts[a, 0]
            uy = verts[j, 1] - verts[a, 1]
            uz = verts[j, 2] - verts[a, 2]
            wx = verts[j + 1, 0] - verts[a, 0]
            wy = verts[j + 1, 1] - verts[a, 1]
            wz = verts[j + 1, 2] - verts[a, 2]
            sx += uy * wz - uz * wy
            sy += uz * wx - ux * wz
            sz += ux * wy - uy * wx
        area = 0.5 * abs(sx * normals[f, 0] + sy * normals[f, 1] + sz * normals[f, 2])
        h = planes[f] - (normals[f, 0] * x0 + normals[f, 1] * y0 + normals[f, 2] * z0)
        v += h * area
    return v / 3.0


@njit(cache=True)
def polygon_area_3d(verts) -> float:
    """Area of a planar polygon of space (vertices in order)."""
    m = verts.shape[0]
    sx = sy = sz = 0.0
    for j in range(1, m - 1):
        ux, uy, uz = verts[j, 0] - verts[0, 0], verts[j, 1] - verts[0, 1], verts[j, 2] - verts[0, 2]
        wx = verts[j + 1, 0] - verts[0, 0]
        wy = verts[j + 1, 1] - verts[0, 1]
        wz = verts[j + 1, 2] - verts[0, 2]
        sx += uy * wz - uz * wy
        sy += uz * wx - ux * wz
        sz += ux * wy - uy * wx
    return 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)


@njit(cache=True)
def inside_polyhedron(normals, planes, px, py, pz) -> bool:
    """Whether a convex polyhedron (its face planes) holds the point ``p``."""
    for f in range(planes.shape[0]):
        if normals[f, 0] * px + normals[f, 1] * py + normals[f, 2] * pz > planes[f]:
            return False
    return True


@njit(cache=True)
def fits_polyhedron(normals, planes, px, py, pz, nx, ny, nz, half) -> bool:
    """Whether a convex polyhedron holds the points ``p +- half n``."""
    for f in range(planes.shape[0]):
        m = normals[f, 0] * nx + normals[f, 1] * ny + normals[f, 2] * nz
        if normals[f, 0] * px + normals[f, 1] * py + normals[f, 2] * pz + half * abs(m) > planes[f]:
            return False
    return True


def box_polyhedron(lower, upper):
    """The box ``[lower, upper]`` as a convex polyhedron (faces labelled :data:`BOX`)."""
    (x0, y0, z0), (x1, y1, z1) = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    faces = [
        ((-1, 0, 0), -x0, [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]),
        ((1, 0, 0), x1, [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
        ((0, -1, 0), -y0, [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        ((0, 1, 0), y1, [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]),
        ((0, 0, -1), -z0, [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]),
        ((0, 0, 1), z1, [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
    ]
    verts = np.array([v for *_, vs in faces for v in vs], dtype=float)
    fofs = np.arange(0, 25, 4, dtype=np.int64)
    labels = np.full(6, BOX, dtype=np.int64)
    normals = np.array([n for n, *_ in faces], dtype=float)
    planes = np.array([c for _, c, _ in faces], dtype=float)
    return verts, fofs, labels, normals, planes


def frame_boxes(lower, upper, frame: float) -> list[tuple]:
    """The six boxes that make the frame ``[lower, upper]`` minus its inner box (``frame``
    thick), as convex polyhedra."""
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    f = float(frame)
    a, b = lo + f, hi - f
    spans = [
        ((lo[0], lo[1], lo[2]), (a[0], hi[1], hi[2])),
        ((b[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])),
        ((a[0], lo[1], lo[2]), (b[0], a[1], hi[2])),
        ((a[0], b[1], lo[2]), (b[0], hi[1], hi[2])),
        ((a[0], a[1], lo[2]), (b[0], b[1], a[2])),
        ((a[0], a[1], b[2]), (b[0], b[1], hi[2])),
    ]
    return [box_polyhedron(x, y) for x, y in spans]


def prism(polygon, axes, axis: int, z0: float, z1: float) -> tuple:
    """The convex polygon of the plane of ``axes`` (counter-clockwise vertices) extruded
    along ``axis`` from ``z0`` to ``z1``, as a convex polyhedron (faces labelled
    :data:`BOX`)."""
    poly = np.asarray(polygon, dtype=float)
    m = len(poly)
    b, c = axes

    def lift(uv, z):
        x = np.empty(3)
        x[b], x[c], x[axis] = uv[0], uv[1], z
        return x

    faces, normals, planes = [], [], []
    e = np.zeros(3)
    e[axis] = 1.0
    faces.append([lift(q, z0) for q in poly[::-1]])
    normals.append(-e)
    planes.append(-z0)
    faces.append([lift(q, z1) for q in poly])
    normals.append(e)
    planes.append(z1)
    for i in range(m):
        p, q = poly[i], poly[(i + 1) % m]
        du, dv = q - p
        length = float(np.hypot(du, dv))
        if length == 0.0:
            continue
        n = np.zeros(3)
        n[b], n[c] = dv / length, -du / length
        loop = [lift(p, z0), lift(q, z0), lift(q, z1), lift(p, z1)]
        faces.append(loop)
        normals.append(n)
        planes.append(float(n @ loop[0]))
    verts = np.array([v for f in faces for v in f], dtype=float)
    fofs = np.concatenate([[0], np.cumsum([len(f) for f in faces])]).astype(np.int64)
    labels = np.full(len(faces), BOX, dtype=np.int64)
    return verts, fofs, labels, np.array(normals, dtype=float), np.array(planes, dtype=float)


@dataclass
class Polyhedra:
    """Convex polyhedra in compressed rows: polyhedron ``i`` has the faces ``offsets[i]``
    to ``offsets[i + 1]``; face ``f`` the vertices ``face_offsets[f]`` to
    ``face_offsets[f + 1]`` (counter-clockwise about its outward unit normal
    ``normals[f]``), the plane ``normals[f] . x = planes[f]`` and the label of what lies
    beyond (a plate, or :data:`BOX`)."""

    offsets: np.ndarray
    face_offsets: np.ndarray
    vertices: np.ndarray
    normals: np.ndarray
    planes: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @classmethod
    def from_list(cls, items: list[tuple]) -> Polyhedra:
        """From ``(vertices, face offsets, labels, normals, planes)`` tuples."""
        offsets = np.zeros(len(items) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(p[2]) for p in items])
        shifts = np.concatenate([[0], np.cumsum([len(p[0]) for p in items])[:-1]]) if items else []
        fo = [p[1][:-1] + s for p, s in zip(items, shifts, strict=True)]
        face_offsets = np.concatenate([*fo, [sum(len(p[0]) for p in items)]]).astype(np.int64)
        return cls(
            offsets=offsets,
            face_offsets=face_offsets,
            vertices=np.concatenate([p[0] for p in items]) if items else np.empty((0, 3)),
            normals=np.concatenate([p[3] for p in items]) if items else np.empty((0, 3)),
            planes=np.concatenate([p[4] for p in items]) if items else np.empty(0),
            labels=(np.concatenate([p[2] for p in items]) if items
                    else np.empty(0, dtype=np.int64)).astype(np.int64),
        )  # fmt: skip

    def polyhedron(self, i: int) -> tuple:
        """``(vertices, face offsets, labels, normals, planes)`` of polyhedron ``i``."""
        f0, f1 = self.offsets[i], self.offsets[i + 1]
        v0, v1 = self.face_offsets[f0], self.face_offsets[f1]
        return (self.vertices[v0:v1], self.face_offsets[f0 : f1 + 1] - v0, self.labels[f0:f1],
                self.normals[f0:f1], self.planes[f0:f1])  # fmt: skip

    def faces(self, i: int) -> list[np.ndarray]:
        """The faces of polyhedron ``i`` (vertices of each, counter-clockwise from
        outside)."""
        verts, fofs, *_ = self.polyhedron(i)
        return [verts[fofs[f] : fofs[f + 1]] for f in range(len(fofs) - 1)]

    def volumes(self) -> np.ndarray:
        out = np.empty(len(self))
        for i in range(len(self)):
            verts, fofs, _, normals, planes = self.polyhedron(i)
            out[i] = polyhedron_volume(verts, fofs, normals, planes)
        return out

    def bounds(self) -> np.ndarray:
        """``(n, 6)``: the lower and upper corners of the box of each polyhedron."""
        out = np.full((len(self), 6), np.nan)
        for i in range(len(self)):
            v = self.polyhedron(i)[0]
            if len(v):
                out[i, :3], out[i, 3:] = v.min(axis=0), v.max(axis=0)
        return out

    def to_dict(self) -> dict:
        return {
            "offsets": self.offsets.tolist(),
            "face_offsets": self.face_offsets.tolist(),
            "vertices": self.vertices.tolist(),
            "normals": self.normals.tolist(),
            "planes": self.planes.tolist(),
            "labels": self.labels.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Polyhedra:
        return cls(
            offsets=np.asarray(data["offsets"], dtype=np.int64),
            face_offsets=np.asarray(data["face_offsets"], dtype=np.int64),
            vertices=np.asarray(data["vertices"], dtype=float).reshape(-1, 3),
            normals=np.asarray(data["normals"], dtype=float).reshape(-1, 3),
            planes=np.asarray(data["planes"], dtype=float).reshape(-1),
            labels=np.asarray(data["labels"], dtype=np.int64).reshape(-1),
        )


class _Cells:
    """The voids of a growing network of plates registered in a grid of buckets of the
    box, to find the void that holds a point (or none: the point lies in a plate)."""

    def __init__(self, lower, upper, n_buckets: int) -> None:
        self.lo = np.asarray(lower, dtype=float)
        size = np.asarray(upper, dtype=float) - self.lo
        cube = float(np.prod(size)) ** (1.0 / 3.0)
        g = np.clip(np.ceil(max(n_buckets, 1) ** (1.0 / 3.0) * size / cube), 1, 64).astype(int)
        self.g = (int(g[0]), int(g[1]), int(g[2]))
        self.h = size / g
        self.buckets: list[set[int]] = [set() for _ in range(int(np.prod(g)))]
        self.ranges: dict[int, tuple] = {}
        self.cells: list[tuple] = []

    def _index(self, v: float, axis: int) -> int:
        return min(max(math.floor((v - self.lo[axis]) / self.h[axis]), 0), self.g[axis] - 1)

    def put(self, cid: int, poly: tuple) -> None:
        """Registers the void ``cid`` (a new one, or in place of the old ``cid``); a
        void with fewer than four faces is empty and holds no point."""
        if cid < len(self.cells):
            self.drop(cid)
            self.cells[cid] = poly
        else:
            self.cells.append(poly)
        if len(poly[2]) < 4:
            return
        lo, hi = poly[0].min(axis=0), poly[0].max(axis=0)
        r = [(self._index(lo[a], a), self._index(hi[a], a)) for a in range(3)]
        self.ranges[cid] = (lo, hi, r)
        g1, g2 = self.g[1], self.g[2]
        for i in range(r[0][0], r[0][1] + 1):
            for j in range(r[1][0], r[1][1] + 1):
                for k in range(r[2][0], r[2][1] + 1):
                    self.buckets[(i * g1 + j) * g2 + k].add(cid)

    def drop(self, cid: int) -> None:
        if cid not in self.ranges:
            return
        *_, r = self.ranges.pop(cid)
        g1, g2 = self.g[1], self.g[2]
        for i in range(r[0][0], r[0][1] + 1):
            for j in range(r[1][0], r[1][1] + 1):
                for k in range(r[2][0], r[2][1] + 1):
                    self.buckets[(i * g1 + j) * g2 + k].discard(cid)

    def locate(self, p) -> int:
        i, j, k = (self._index(p[a], a) for a in range(3))
        for cid in self.buckets[(i * self.g[1] + j) * self.g[2] + k]:
            lo, hi, _ = self.ranges[cid]
            if np.all(lo <= p) and np.all(p <= hi):
                poly = self.cells[cid]
                if inside_polyhedron(poly[3], poly[4], p[0], p[1], p[2]):
                    return cid
        return -1


# -- the growth of plates ------------------------------------------------------------------


@dataclass
class PlateGrowth:
    """The plates and voids of a scale-rich network of plates."""

    nucleation: np.ndarray  # (N, 3) the nucleation points
    normal: np.ndarray  # (N, 3) the unit normals of the plates
    thickness: np.ndarray  # (N,)
    rim_offsets: np.ndarray  # (N + 1,) CSR of the rims: the sections of the voids by the
    rim_vertices: np.ndarray  # (R, 3)     mid-planes of the plates
    rim_faces: np.ndarray  # (R,) the label of the face of each edge (plate or BOX)
    bodies: Polyhedra  # the plates
    cells: Polyhedra  # the voids
    jammed: bool = False
    attempts: int = 0

    @property
    def n(self) -> int:
        return len(self.thickness)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def rims(self) -> list[np.ndarray]:
        """The rim of each plate: the polygon of its mid-plane in its void."""
        o = self.rim_offsets
        return [self.rim_vertices[o[i] : o[i + 1]] for i in range(self.n)]

    @property
    def areas(self) -> np.ndarray:
        """The areas of the plates (of their rims)."""
        return np.array([polygon_area_3d(r) for r in self.rims()])

    def body_volumes(self) -> np.ndarray:
        return self.bodies.volumes()

    def cell_volumes(self) -> np.ndarray:
        return self.cells.volumes()

    def links(self) -> np.ndarray:
        """The pairs of plates joined: each plate with the plates on which its rim lies
        (the edges of the dual graph, each once)."""
        counts = np.diff(self.rim_offsets)
        i = np.repeat(np.arange(self.n), counts)
        j = self.rim_faces
        keep = j >= 0
        if not np.any(keep):
            return np.empty((0, 2), dtype=np.int64)
        return np.unique(np.column_stack([i[keep], j[keep]]), axis=0)

    def degrees(self) -> np.ndarray:
        """The number of plates each plate meets (its degree in the dual graph)."""
        e = self.links()
        return np.bincount(e.ravel(), minlength=self.n).astype(np.int64)

    def boundary_contacts(self) -> np.ndarray:
        """Whether each plate ends on the box."""
        counts = np.diff(self.rim_offsets)
        i = np.repeat(np.arange(self.n), counts)
        out = np.zeros(self.n, dtype=bool)
        out[i[self.rim_faces == BOX]] = True
        return out

    def orientation_tensor(self) -> np.ndarray:
        """``N = sum V_t n_t n_t / sum V_t``: the normals ``n_t`` of the plates weighted
        by their volumes ``V_t`` (the fabric tensor of the walls)."""
        v = self.body_volumes()
        if not v.sum() > 0:
            return np.full((3, 3), np.nan)
        return np.einsum("t,ti,tj->ij", v, self.normal, self.normal) / v.sum()


def _csr3(items: list[np.ndarray], dtype=float, width: int = 3):
    offsets = np.zeros(len(items) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(a) for a in items])
    shape = (0, width) if width else (0,)
    data = np.concatenate(items).astype(dtype) if items else np.empty(shape, dtype=dtype)
    return offsets, data


def isotropic_directions(t: int, points: np.ndarray, gen: np.random.Generator) -> np.ndarray:
    """Directions uniform on the sphere, one per point."""
    return uniform_unit_vectors(gen, len(points))


def grow_plates(
    lower,
    upper,
    n_plates: int,
    thickness,
    entropy: int,
    *,
    directions=None,
    density=None,
    min_spacing: float = 0.0,
    max_attempts: int = 1000,
) -> PlateGrowth:
    """Grow a scale-rich network of plates in the box ``[lower, upper]``.

    ``thickness(t)`` is the thickness of the plate ``t = 1, 2, ...``; ``directions(t,
    points, generator)`` the unit normals of the plate at candidate nucleation points
    (``(k, 3)`` arrays; default: uniform on the sphere); ``density(x, y, z)`` the relative
    density of nucleation in ``[0, 1]`` (default: uniform), by rejection. As in
    :func:`morphogen.models.scale_rich.grow`, a nucleation point is drawn uniformly in the
    box until it falls in a void, each plate draws from its own generators (points:
    ``SeedSequence(entropy, spawn_key=(t,))``, normals: ``spawn_key=(t, 1)``), and the
    growth jams after ``max_attempts`` points in a row where the plate does not fit."""
    lo = np.asarray(lower, dtype=float).reshape(3)
    hi = np.asarray(upper, dtype=float).reshape(3)
    size = hi - lo
    volume = float(np.prod(size))
    tol = 1e-12 * float(np.max(size))
    draw_normals = isotropic_directions if directions is None else directions
    voids = _Cells(lo, hi, n_buckets=max(1, int(n_plates) // 2))
    voids.put(0, box_polyhedron(lo, hi))
    void_volume = volume

    nucleation, normal, lam, rims, rim_faces, bodies = [], [], [], [], [], []
    attempts = 0
    jammed = False
    for t in range(1, int(n_plates) + 1):
        width = float(thickness(t))
        half = 0.5 * (width + min_spacing)
        gen = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=(t,))))
        gdir = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=(t, 1)))
        )
        failures = k = 0
        block = gen.random((16, _DRAWS))
        pts = lo + block[:, :3] * size
        dirs = draw_normals(t, pts, gdir)
        while True:
            if k == len(block):
                if void_volume <= 1e-12 * volume:
                    jammed = True
                    break
                block, k = gen.random((256, _DRAWS)), 0
                pts = lo + block[:, :3] * size
                dirs = draw_normals(t, pts, gdir)
            p, u = pts[k], block[k, 3]
            n = dirs[k]
            k += 1
            if density is not None and u >= density(p[0], p[1], p[2]):
                continue
            i = voids.locate(p)
            if i < 0:
                continue  # in a plate
            attempts += 1
            _, _, _, fn, fc = voids.cells[i]
            if fits_polyhedron(fn, fc, p[0], p[1], p[2], n[0], n[1], n[2], half):
                break
            failures += 1
            if failures >= max_attempts:
                jammed = True
                break
        if jammed:
            break
        cell = voids.cells[i]
        nx, ny, nz = (float(v) for v in n)
        c = nx * p[0] + ny * p[1] + nz * p[2]
        label = t - 1
        w2 = 0.5 * width
        minus = clip_polyhedron(*cell, nx, ny, nz, c - w2, label, tol)
        plus = clip_polyhedron(*cell, -nx, -ny, -nz, -(c + w2), label, tol)
        body = clip_polyhedron(*cell, nx, ny, nz, c + w2, label, tol)
        body = clip_polyhedron(*body, -nx, -ny, -nz, -(c - w2), label, tol)
        rim, rim_lab = section(*cell, nx, ny, nz, c, tol)
        void_volume += (polyhedron_volume(minus[0], minus[1], minus[3], minus[4])
                        + polyhedron_volume(plus[0], plus[1], plus[3], plus[4])
                        - polyhedron_volume(cell[0], cell[1], cell[3], cell[4]))  # fmt: skip
        voids.put(i, minus)
        voids.put(len(voids.cells), plus)
        nucleation.append(p.copy())
        normal.append((nx, ny, nz))
        lam.append(width)
        rims.append(rim)
        rim_faces.append(rim_lab)
        bodies.append(body)

    n = len(lam)
    rim_offsets, rim_vertices = _csr3(rims)
    _, rim_labels = _csr3(rim_faces, dtype=np.int64, width=0)
    return PlateGrowth(
        nucleation=np.asarray(nucleation, dtype=float).reshape(n, 3),
        normal=np.asarray(normal, dtype=float).reshape(n, 3),
        thickness=np.asarray(lam, dtype=float),
        rim_offsets=rim_offsets,
        rim_vertices=rim_vertices,
        rim_faces=rim_labels,
        bodies=Polyhedra.from_list(bodies),
        cells=Polyhedra.from_list(voids.cells),
        jammed=jammed,
        attempts=attempts,
    )


# -- the growth of beams -------------------------------------------------------------------


@njit(cache=True)
def _point_segment(px, py, pz, ax, ay, az, bx, by, bz) -> float:
    """Distance from the point ``p`` to the segment ``[a, b]``."""
    ex, ey, ez = bx - ax, by - ay, bz - az
    ee = ex * ex + ey * ey + ez * ez
    u = 0.0
    if ee > 0.0:
        u = min(max(((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / ee, 0.0), 1.0)
    qx, qy, qz = ax + u * ex - px, ay + u * ey - py, az + u * ez - pz
    return math.sqrt(qx * qx + qy * qy + qz * qz)


@njit(cache=True)
def _ray_segment(px, py, pz, dx, dy, dz, smax, ax, ay, az, bx, by, bz):
    """The closest approach of the ray ``p + s d`` (``0 <= s <= smax``, ``|d| = 1``) and
    the segment ``[a, b]``: ``s`` and the distance (Ericson 2005, §5.1.9)."""
    ex, ey, ez = bx - ax, by - ay, bz - az
    rx, ry, rz = px - ax, py - ay, pz - az
    ee = ex * ex + ey * ey + ez * ez
    f = ex * rx + ey * ry + ez * rz
    c = dx * rx + dy * ry + dz * rz
    if ee <= 1e-300:
        s = min(max(-c, 0.0), smax)
        u = 0.0
    else:
        b = dx * ex + dy * ey + dz * ez
        denom = ee - b * b
        s = min(max((b * f - c * ee) / denom, 0.0), smax) if denom > 1e-14 * ee else 0.0
        u = (b * s + f) / ee
        if u < 0.0:
            u = 0.0
            s = min(max(-c, 0.0), smax)
        elif u > 1.0:
            u = 1.0
            s = min(max(b - c, 0.0), smax)
    qx = ax + u * ex - (px + s * dx)
    qy = ay + u * ey - (py + s * dy)
    qz = az + u * ez - (pz + s * dz)
    return s, math.sqrt(qx * qx + qy * qy + qz * qz)


@njit(cache=True)
def beam_clearance(px, py, pz, ends, radii, n) -> float:
    """``min_j (|p - axis_j| - r_j)`` over the ``n`` first beams: negative in a beam, the
    distance to the nearest one outside (``inf``: none)."""
    best = np.inf
    for j in range(n):
        lo0 = min(ends[j, 0, 0], ends[j, 1, 0]) - radii[j]
        hi0 = max(ends[j, 0, 0], ends[j, 1, 0]) + radii[j]
        lo1 = min(ends[j, 0, 1], ends[j, 1, 1]) - radii[j]
        hi1 = max(ends[j, 0, 1], ends[j, 1, 1]) + radii[j]
        lo2 = min(ends[j, 0, 2], ends[j, 1, 2]) - radii[j]
        hi2 = max(ends[j, 0, 2], ends[j, 1, 2]) + radii[j]
        gx = max(lo0 - px, 0.0, px - hi0)
        gy = max(lo1 - py, 0.0, py - hi1)
        gz = max(lo2 - pz, 0.0, pz - hi2)
        if gx * gx + gy * gy + gz * gz >= best * best and best > 0.0:
            continue  # its box is farther than the nearest beam yet
        d = _point_segment(px, py, pz, ends[j, 0, 0], ends[j, 0, 1], ends[j, 0, 2],
                           ends[j, 1, 0], ends[j, 1, 1], ends[j, 1, 2]) - radii[j]  # fmt: skip
        if d < best:
            best = d
            if best < 0.0:
                return best  # in a beam
    return best


@njit(cache=True)
def beam_stop(px, py, pz, dx, dy, dz, smax, ends, radii, n, radius):
    """How far a beam of ``radius`` grows from ``p`` along ``d`` (``|d| = 1``) among the
    ``n`` first beams, up to ``smax`` (the box): its axis stops where it reaches the
    surface of a beam, or at its closest approach to a beam it passes within contact
    (``r_j + radius``); the distance and the beam (:data:`BOX`: none)."""
    best = smax
    who = BOX
    for j in range(n):
        reach = radii[j] + radius
        # the box of the axis of j widened by the contact against the box of the ray
        qx, qy, qz = px + best * dx, py + best * dy, pz + best * dz
        if (max(ends[j, 0, 0], ends[j, 1, 0]) + reach < min(px, qx)
                or min(ends[j, 0, 0], ends[j, 1, 0]) - reach > max(px, qx)
                or max(ends[j, 0, 1], ends[j, 1, 1]) + reach < min(py, qy)
                or min(ends[j, 0, 1], ends[j, 1, 1]) - reach > max(py, qy)
                or max(ends[j, 0, 2], ends[j, 1, 2]) + reach < min(pz, qz)
                or min(ends[j, 0, 2], ends[j, 1, 2]) - reach > max(pz, qz)):  # fmt: skip
            continue
        ax, ay, az = ends[j, 0, 0], ends[j, 0, 1], ends[j, 0, 2]
        bx, by, bz = ends[j, 1, 0], ends[j, 1, 1], ends[j, 1, 2]
        s, dmin = _ray_segment(px, py, pz, dx, dy, dz, best, ax, ay, az, bx, by, bz)
        if dmin > reach:
            continue
        if dmin <= radii[j]:  # the axis meets the surface of j first where it crosses it
            lo, hi = 0.0, s
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                if _point_segment(px + mid * dx, py + mid * dy, pz + mid * dz,
                                  ax, ay, az, bx, by, bz) > radii[j]:  # fmt: skip
                    lo = mid
                else:
                    hi = mid
            s = hi
        if s < best:
            best, who = s, j
    return best, who


@njit(cache=True)
def beam_contacts(ax, ay, az, dx, dy, dz, length, radius, ends, radii, n, tol):
    """The beams among the ``n`` first ones that the beam of axis ``a + s d`` (``0 <= s <=
    length``) and ``radius`` touches or overlaps (within ``tol``)."""
    out = np.empty(n, dtype=np.int64)
    m = 0
    bx, by, bz = ax + length * dx, ay + length * dy, az + length * dz
    for j in range(n):
        reach = radii[j] + radius + tol
        if (max(ends[j, 0, 0], ends[j, 1, 0]) + reach < min(ax, bx)
                or min(ends[j, 0, 0], ends[j, 1, 0]) - reach > max(ax, bx)
                or max(ends[j, 0, 1], ends[j, 1, 1]) + reach < min(ay, by)
                or min(ends[j, 0, 1], ends[j, 1, 1]) - reach > max(ay, by)
                or max(ends[j, 0, 2], ends[j, 1, 2]) + reach < min(az, bz)
                or min(ends[j, 0, 2], ends[j, 1, 2]) - reach > max(az, bz)):  # fmt: skip
            continue
        e = ends[j]
        _, dmin = _ray_segment(ax, ay, az, dx, dy, dz, length, e[0, 0], e[0, 1], e[0, 2],
                               e[1, 0], e[1, 1], e[1, 2])  # fmt: skip
        if dmin <= reach:
            out[m] = j
            m += 1
    return out[:m]


def _exit(p, d, lo, hi) -> float:
    """How far the ray ``p + s d`` runs in the box ``[lo, hi]`` holding ``p``."""
    s = math.inf
    for a in range(3):
        if d[a] > 0:
            s = min(s, (hi[a] - p[a]) / d[a])
        elif d[a] < 0:
            s = min(s, (lo[a] - p[a]) / d[a])
    return max(s, 0.0)


@dataclass
class BeamGrowth:
    """The beams of a scale-rich network of beams."""

    nucleation: np.ndarray  # (N, 3) the nucleation points
    direction: np.ndarray  # (N, 3) the unit vectors of the axes
    thickness: np.ndarray  # (N,) the diameters
    ends: np.ndarray  # (N, 2, 3) the ends of the axes
    neighbours: np.ndarray  # (N, 2) the beams that stop the ends (BOX: the boundary)
    contact_offsets: np.ndarray  # (N + 1,) CSR of the earlier beams that each beam touches
    contacts: np.ndarray  # (M,)
    jammed: bool = False
    attempts: int = 0

    @property
    def n(self) -> int:
        return len(self.thickness)

    @property
    def radii(self) -> np.ndarray:
        return 0.5 * self.thickness

    @property
    def lengths(self) -> np.ndarray:
        """The lengths of the axes of the beams."""
        return np.linalg.norm(self.ends[:, 1] - self.ends[:, 0], axis=1)

    def volumes(self) -> np.ndarray:
        """The volumes of the cylinders on the axes, ``pi r^2 l`` (their round ends lie in
        the beams they meet or beyond the box)."""
        return np.pi * self.radii**2 * self.lengths

    def links(self) -> np.ndarray:
        """The pairs of beams joined: each beam with the earlier beams it touches, those
        that stop its ends and those it meets beside them (the edges of the dual graph,
        each once)."""
        i = np.repeat(np.arange(self.n), np.diff(self.contact_offsets))
        return np.column_stack([i, self.contacts]).astype(np.int64).reshape(-1, 2)

    def degrees(self) -> np.ndarray:
        """The number of beams each beam touches (its degree in the dual graph)."""
        e = self.links()
        return np.bincount(e.ravel(), minlength=self.n).astype(np.int64)

    def orientation_tensor(self) -> np.ndarray:
        """``M = sum V_t d_t d_t / sum V_t``: the axes ``d_t`` of the beams weighted by
        their volumes ``V_t``."""
        v = self.volumes()
        if not v.sum() > 0:
            return np.full((3, 3), np.nan)
        return np.einsum("t,ti,tj->ij", v, self.direction, self.direction) / v.sum()


def grow_beams(
    lower,
    upper,
    n_beams: int,
    thickness,
    entropy: int,
    *,
    directions=None,
    density=None,
    min_spacing: float = 0.0,
    max_attempts: int = 1000,
) -> BeamGrowth:
    """Grow a scale-rich network of beams in the box ``[lower, upper]``: as
    :func:`grow_plates`, ``thickness(t)`` the diameter of the beam ``t`` and
    ``directions`` its axes. A nucleation point inside a beam is drawn again; the beam fits
    where the ball of radius ``(lambda_t + epsilon) / 2`` about the point clears the other
    beams and the faces of the box (:func:`beam_clearance`), and grows both ways
    (:func:`beam_stop`)."""
    lo = np.asarray(lower, dtype=float).reshape(3)
    hi = np.asarray(upper, dtype=float).reshape(3)
    size = hi - lo
    draw_axes = isotropic_directions if directions is None else directions
    cap = int(n_beams)
    ends = np.zeros((cap, 2, 3))
    radii = np.zeros(cap)
    tol = 1e-9 * float(np.max(size))
    nucleation, axes, neighbours, contacts = [], [], [], []
    attempts = 0
    jammed = False
    n = 0
    for t in range(1, cap + 1):
        radius = 0.5 * float(thickness(t))
        clear = radius + 0.5 * min_spacing
        gen = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=(t,))))
        gdir = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=(t, 1)))
        )
        failures = k = 0
        block = gen.random((16, _DRAWS))
        pts = lo + block[:, :3] * size
        dirs = draw_axes(t, pts, gdir)
        while True:
            if k == len(block):
                block, k = gen.random((256, _DRAWS)), 0
                pts = lo + block[:, :3] * size
                dirs = draw_axes(t, pts, gdir)
            p, u = pts[k], block[k, 3]
            d = dirs[k]
            k += 1
            if density is not None and u >= density(p[0], p[1], p[2]):
                continue
            gap = beam_clearance(p[0], p[1], p[2], ends, radii, n)
            if gap < 0:
                continue  # in a beam
            attempts += 1
            if gap >= clear and np.all(p - lo >= clear) and np.all(hi - p >= clear):
                break
            failures += 1
            if failures >= max_attempts:
                jammed = True
                break
        if jammed:
            break
        d = np.asarray(d, dtype=float)
        s1, j1 = beam_stop(p[0], p[1], p[2], d[0], d[1], d[2], _exit(p, d, lo, hi), ends, radii,
                           n, radius)  # fmt: skip
        s0, j0 = beam_stop(p[0], p[1], p[2], -d[0], -d[1], -d[2], _exit(p, -d, lo, hi), ends,
                           radii, n, radius)  # fmt: skip
        ends[n, 0] = p - s0 * d
        ends[n, 1] = p + s1 * d
        start = ends[n, 0]
        contacts.append(beam_contacts(start[0], start[1], start[2], d[0], d[1], d[2], s0 + s1,
                                      radius, ends, radii, n, tol))  # fmt: skip
        radii[n] = radius
        n += 1
        nucleation.append(p.copy())
        axes.append(d)
        neighbours.append((j0, j1))
    contact_offsets, contact_items = _csr3(contacts, dtype=np.int64, width=0)
    return BeamGrowth(
        nucleation=np.asarray(nucleation, dtype=float).reshape(n, 3),
        direction=np.asarray(axes, dtype=float).reshape(n, 3),
        thickness=2.0 * radii[:n].copy(),
        ends=ends[:n].copy(),
        neighbours=np.asarray(neighbours, dtype=np.int64).reshape(n, 2),
        contact_offsets=contact_offsets,
        contacts=contact_items,
        jammed=jammed,
        attempts=attempts,
    )


# -- theory --------------------------------------------------------------------------------


def _radial_moments(radius: float, c: float) -> tuple[float, float, float]:
    """``int_0^R rho^k g(rho) d rho`` for ``k = 1, 2, 3``, with ``g(rho) = int_0^c (c - z) /
    sqrt(rho^2 + z^2) dz = c asinh(c / rho) - sqrt(rho^2 + c^2) + rho`` (closed forms)."""
    r = radius
    s = math.sqrt(r * r + c * c)
    ash = math.asinh(r / c)
    inv = math.asinh(c / r) if r > 0 else 0.0
    q1 = s - c  # int rho^k / sqrt(rho^2 + c^2)
    q2 = 0.5 * (r * s - c * c * ash)
    q3 = (s**3 - c**3) / 3.0 - c * c * (s - c)
    a1 = r**2 / 2 * inv + c / 2 * q1  # int rho^k asinh(c / rho), by parts
    a2 = r**3 / 3 * inv + c / 3 * q2
    a3 = r**4 / 4 * inv + c / 4 * q3
    r1 = (s**3 - c**3) / 3.0  # int rho^k sqrt(rho^2 + c^2)
    r2 = r * (2 * r * r + c * c) * s / 8.0 - c**4 / 8.0 * ash
    r3 = (s**5 - c**5) / 5.0 - c * c * (s**3 - c**3) / 3.0
    return c * a1 - r1 + r**3 / 3, c * a2 - r2 + r**4 / 4, c * a3 - r3 + r**5 / 5


@lru_cache(maxsize=64)
def _box_integral(a: float, b: float, c: float) -> float:
    """``int_[0,a]x[0,b]x[0,c] (a - x)(b - y)(c - z) / |r| dr``: the integral over ``z`` and
    the radius of polar coordinates of the plane in closed form, the angle by Gauss-Legendre
    on either side of the diagonal (smooth there)."""
    x, w = np.polynomial.legendre.leggauss(48)
    diagonal = math.atan2(b, a)
    total = 0.0
    for lo, hi, side in ((0.0, diagonal, 0), (diagonal, 0.5 * math.pi, 1)):
        for xi, wi in zip(lo + (x + 1) * (hi - lo) / 2, w * (hi - lo) / 2, strict=True):
            sin, cos = math.sin(xi), math.cos(xi)
            m1, m2, m3 = _radial_moments(a / cos if side == 0 else b / sin, c)
            total += wi * (a * b * m1 - (a * sin + b * cos) * m2 + sin * cos * m3)
    return float(total)


def mean_inverse_distance(a: float, b: float, c: float) -> float:
    """``E[1 / |X - Y|]`` for two points uniform in an ``a x b x c`` box (1.882313 for the
    unit cube, Bailey, Borwein & Crandall 2007)."""
    a, b, c = float(a), float(b), float(c)
    return 8.0 * _box_integral(a, b, c) / (a * b * c) ** 2


def first_area(a: float, b: float, c: float) -> float:
    """The mean area of the first plate in an ``a x b x c`` box: the section by the plane
    through a uniform point with a uniform normal, ``int A^2 dE / (4 pi V)`` over the
    planes ``E``. As ``int_S2 delta(n . r) dn = 2 pi / |r|``, it is ``V E[1/|X - Y|] / 2``
    (:func:`mean_inverse_distance`): 0.941156 for the unit cube."""
    a, b, c = float(a), float(b), float(c)
    return 0.5 * a * b * c * mean_inverse_distance(a, b, c)


def area_sum(decay: float, n_plates: int, a: float, b: float, c: float) -> float:
    """``sum_t a_t t^-alpha / V^(2/3)`` of a thin network of ``T`` plates in an ``a x b x
    c`` box: ``a_0 / V^(2/3) + c_3 sum_{t=2..T} t^-(alpha + 2/3)`` (:func:`first_area`,
    :data:`AREA_CONSTANT`), the relative density per unit of ``lambda_0 / V^(1/3)``."""
    volume = a * b * c
    tail = 0.0
    if n_plates >= 2:
        t = np.arange(2, int(n_plates) + 1, dtype=float)
        tail = float(np.sum(t ** (-(decay + 2.0 / 3.0))))
    return float(first_area(a, b, c) / volume ** (2.0 / 3.0) + AREA_CONSTANT * tail)


@lru_cache(maxsize=16)
def _exits(a: float, b: float, c: float, n: int = 4096) -> np.ndarray:
    """How far rays from uniform points of an ``a x b x c`` box along uniform directions
    run in it: ``n`` draws of a fixed seed."""
    rng = np.random.default_rng(2026)
    size = np.array([a, b, c])
    p = rng.random((n, 3)) * size
    d = uniform_unit_vectors(rng, n)
    with np.errstate(divide="ignore"):
        s = np.where(d > 0, (size - p) / d, np.where(d < 0, -p / d, np.inf))
    return s.min(axis=1)


def beam_density_estimate(initial_thickness: float, decay: float, n_beams: int, sides) -> float:
    """The mean-field relative density of ``T`` beams in a box (isotropic axes and
    nucleation): a beam of radius ``r_t`` is stopped by a beam ``j`` it passes within ``r_j
    + r_t`` of, whose capsule of that radius has the mean cross-section ``sigma_j = (pi / 2)
    (r_j + r_t) l_j + pi (r_j + r_t)^2`` (Cauchy: a quarter of its surface), so that each
    half of its axis runs ``min(X, D)``: ``X`` exponential of rate ``h_t = sum_j sigma_j /
    V``, ``D`` the distance to the box. Then ``l_t = 2 E[(1 - exp(-h_t D)) / h_t]`` and
    ``rho = sum_t pi r_t^2 l_t / V``; jamming is ignored."""
    a, b, c = (float(v) for v in sides)
    volume = a * b * c
    exits = _exits(a, b, c)
    a1 = a0 = b2 = b1 = rho = 0.0
    for t in range(1, int(n_beams) + 1):
        r = 0.5 * float(initial_thickness) * t ** (-float(decay))
        n = t - 1
        h = (0.5 * np.pi * (a1 + r * a0) + np.pi * (b2 + 2.0 * r * b1 + n * r * r)) / volume
        if h * exits.max() < 1e-12:
            half = float(exits.mean())
        else:
            half = float(np.mean(-np.expm1(-h * exits)) / h)
        length = 2.0 * half
        rho += np.pi * r * r * length / volume
        a1 += r * length
        a0 += length
        b2 += r * r
        b1 += r
    return float(rho)


def beam_thickness_estimate(volume_fraction: float, decay: float, n_beams: int, sides) -> float:
    """The initial thickness at which the mean field of :func:`beam_density_estimate` gives
    the relative density (by bisection of its logarithm)."""
    target = float(volume_fraction)
    lo, hi = 1e-6 * min(sides), float(max(sides))
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if beam_density_estimate(mid, decay, n_beams, sides) < target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + 1e-6:
            break
    return math.sqrt(lo * hi)


@lru_cache(maxsize=1)
def union_points() -> np.ndarray:
    """The ``2^UNION_POINTS_LOG2`` points of the unit cube (a scrambled Sobol set of a fixed
    seed) on which the relative density of beams is estimated."""
    from scipy.stats import qmc

    pts = qmc.Sobol(3, scramble=True, seed=2026).random_base2(UNION_POINTS_LOG2)
    pts.setflags(write=False)
    return pts


# -- the field of the plates or beams at points of the box: a grid of buckets --------------


@njit(cache=True)
def _register(boxes, reach, lo, h, g):
    """The elements of each bucket of a ``g[0] x g[1] x g[2]`` grid: those whose box (lower
    and upper corners), widened by ``reach``, meets it (compressed rows)."""
    n = boxes.shape[0]
    nb = g[0] * g[1] * g[2]
    counts = np.zeros(nb, dtype=np.int64)
    offsets = np.zeros(nb + 1, dtype=np.int64)
    items = np.empty(0, dtype=np.int64)
    fill = np.zeros(nb, dtype=np.int64)
    for pass_ in range(2):
        if pass_ == 1:
            for b in range(nb):
                offsets[b + 1] = offsets[b] + counts[b]
            items = np.empty(offsets[-1], dtype=np.int64)
            fill = offsets[:-1].copy()
        for p in range(n):
            r0 = np.empty(3, dtype=np.int64)
            r1 = np.empty(3, dtype=np.int64)
            for a in range(3):
                r0[a] = min(max(math.floor((boxes[p, a] - reach - lo[a]) / h[a]), 0), g[a] - 1)
                r1[a] = min(max(math.floor((boxes[p, 3 + a] + reach - lo[a]) / h[a]), 0), g[a] - 1)
            for i in range(r0[0], r1[0] + 1):
                for j in range(r0[1], r1[1] + 1):
                    for k in range(r0[2], r1[2] + 1):
                        b = (i * g[1] + j) * g[2] + k
                        if pass_ == 0:
                            counts[b] += 1
                        else:
                            items[fill[b]] = p
                            fill[b] += 1
    return offsets, items


@njit(inline="always")
def _bucket(x, y, z, lo, h, g):
    i = min(max(math.floor((x - lo[0]) / h[0]), 0), g[0] - 1)
    j = min(max(math.floor((y - lo[1]) / h[1]), 0), g[1] - 1)
    k = min(max(math.floor((z - lo[2]) / h[2]), 0), g[2] - 1)
    return (i * g[1] + j) * g[2] + k


@njit(parallel=True, cache=True)
def _plates_field(pts, reach, boxes, offsets, normals, planes, b_off, b_items, lo, h, g):
    """``min(reach, min_p max_f (n_f . x - c_f))`` at points: negative in the plates (convex
    polyhedra ``p`` of faces ``f``), the distance to the nearest one near them, or less."""
    out = np.empty(pts.shape[0])
    for q in prange(pts.shape[0]):
        x, y, z = pts[q, 0], pts[q, 1], pts[q, 2]
        b = _bucket(x, y, z, lo, h, g)
        best = reach
        for kk in range(b_off[b], b_off[b + 1]):
            p = b_items[kk]
            if (x < boxes[p, 0] - reach or x > boxes[p, 3] + reach or y < boxes[p, 1] - reach
                    or y > boxes[p, 4] + reach or z < boxes[p, 2] - reach
                    or z > boxes[p, 5] + reach):  # fmt: skip
                continue
            phi = -np.inf
            for f in range(offsets[p], offsets[p + 1]):
                v = normals[f, 0] * x + normals[f, 1] * y + normals[f, 2] * z - planes[f]
                if v > phi:
                    phi = v
                    if phi >= best:
                        break
            if phi < best:
                best = phi
        out[q] = best
    return out


@njit(parallel=True, cache=True)
def _beams_field(pts, reach, boxes, ends, radii, b_off, b_items, lo, h, g):
    """``min(reach, min_j (|x - axis_j| - r_j))`` at points: the signed distance to the
    beams (round ends) near them, negative inside."""
    out = np.empty(pts.shape[0])
    for q in prange(pts.shape[0]):
        x, y, z = pts[q, 0], pts[q, 1], pts[q, 2]
        b = _bucket(x, y, z, lo, h, g)
        best = reach
        for kk in range(b_off[b], b_off[b + 1]):
            j = b_items[kk]
            if (x < boxes[j, 0] - reach or x > boxes[j, 3] + reach or y < boxes[j, 1] - reach
                    or y > boxes[j, 4] + reach or z < boxes[j, 2] - reach
                    or z > boxes[j, 5] + reach):  # fmt: skip
                continue
            d = _point_segment(x, y, z, ends[j, 0, 0], ends[j, 0, 1], ends[j, 0, 2],
                               ends[j, 1, 0], ends[j, 1, 1], ends[j, 1, 2]) - radii[j]  # fmt: skip
            if d < best:
                best = d
        out[q] = best
    return out


class _Index:
    """Elements of a network (their boxes) registered in a grid of buckets of the box, to
    evaluate their field within ``reach`` of them."""

    def __init__(self, boxes: np.ndarray, lower, upper, reach: float) -> None:
        self.reach = float(reach)
        self.boxes = np.ascontiguousarray(boxes, dtype=float).reshape(-1, 6)
        lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        size = hi - lo
        n = len(self.boxes)
        cube = float(np.prod(size)) ** (1.0 / 3.0)
        g = np.clip(np.ceil(max(n, 1) ** (1.0 / 3.0) * size / cube), 1, 128).astype(np.int64)
        self.g = g
        self.lo = lo
        self.h = size / g
        self.b_off, self.b_items = _register(self.boxes, self.reach, lo, self.h, g)


def _slabs(grid: Grid, ss: int, values_of, reduce_dtype) -> np.ndarray:
    """``values_of(points)`` on the voxel centres of ``grid`` (``ss`` = 1) or the mean of
    its ``ss^3`` sub-voxels (sub-grid of ``ss`` times the resolution), slab by slab."""
    nx, ny, nz = grid.shape
    out = np.empty(grid.shape, dtype=reduce_dtype)
    fine = grid if ss == 1 else Grid(grid.domain, (nx * ss, ny * ss, nz * ss))
    step = max(1, (1 << 21) // max(1, nx * ny * ss**3))
    for k0 in range(0, nz, step):
        k1 = min(nz, k0 + step)
        vals = values_of(fine.center_points(k0 * ss, k1 * ss))
        if ss == 1:
            out[:, :, k0:k1] = vals.reshape(nx, ny, k1 - k0)
        else:
            v = vals.reshape(nx, ss, ny, ss, k1 - k0, ss).astype(np.float32)
            out[:, :, k0:k1] = v.mean(axis=(1, 3, 5), dtype=np.float32)
    return out


# -- the microstructures -------------------------------------------------------------------


class _Network3D(Microstructure):
    """A scale-rich network of plates or beams in the box, within a ``frame`` of material
    on which they end (0: none)."""

    def __init__(self, domain: Domain, growth, frame: float = 0.0, **kwargs) -> None:
        super().__init__(domain, **kwargs)
        self.growth = growth
        self.frame = float(frame)
        self._indices: dict[float, _Index] = {}

    @property
    def n_ligaments(self) -> int:
        return self.growth.n

    @property
    def thicknesses(self) -> np.ndarray:
        return self.growth.thickness

    def degrees(self) -> np.ndarray:
        """The degree of each element in the dual graph: the elements it meets."""
        return self.growth.degrees()

    def links(self) -> np.ndarray:
        """The pairs of elements that meet."""
        return self.growth.links()

    def orientation_tensor(self) -> np.ndarray:
        """The orientation tensor of the elements (3 x 3, trace 1)."""
        return self.growth.orientation_tensor()

    @property
    def geometric_anisotropy(self) -> float:
        """``mu_min / mu_max``, the ratio of the extreme eigenvalues of the orientation
        tensor: 1 for isotropic directions, 0 when they all lie in a plane (or along an
        axis)."""
        mu = np.linalg.eigvalsh(self.orientation_tensor())
        return float(mu[0] / mu[2]) if mu[2] > 0 else math.nan

    # -- the field -------------------------------------------------------------------

    def _boxes(self) -> np.ndarray:
        raise NotImplementedError

    def _elements_field(self, pts: np.ndarray, index: _Index) -> np.ndarray:
        raise NotImplementedError

    def _index(self, reach: float) -> _Index:
        key = round(float(reach), 12)
        if key not in self._indices:
            self._indices.clear()
            self._indices[key] = _Index(self._boxes(), self.domain.lower, self.domain.upper,
                                        reach)  # fmt: skip
        return self._indices[key]

    def field(self, points, reach: float = 0.0) -> np.ndarray:
        """Negative in the elements (and the frame), positive in the voids: their signed
        distance near them (plates: a lower bound near their edges), capped at
        ``reach``."""
        pts = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 3))
        index = self._index(reach)
        if len(index.boxes):
            out = self._elements_field(pts, index)
        else:
            out = np.full(len(pts), float(reach))
        if self.frame > 0:
            inner = np.minimum(pts - self.domain.lower, self.domain.upper - pts).min(axis=1)
            out = np.minimum(out, inner - self.frame)
        return out

    def contains(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        inside = np.all((pts >= self.domain.lower) & (pts <= self.domain.upper), axis=1)
        return (self.field(pts) < 0) & inside

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """:meth:`field` within ``reach`` (default: the diagonal of the box): the signed
        distance to the beams; for plates, the largest of the signed distances to the planes
        of the faces (the distance outside them near their faces, a lower bound near their
        edges)."""
        if not np.isfinite(reach):
            reach = float(self.domain.diagonal)
        return self.field(points, float(reach))

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        meta = self.image_metadata()
        if ss == 1:
            data = _slabs(grid, 1, lambda p: (self.field(p) < 0).astype(np.uint8), np.uint8)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=meta)
        data = _slabs(grid, ss, lambda p: self.field(p) < 0, np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=meta)

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Negative in the elements, positive in the voids: the distance to them near
        them (plates: a lower bound near their edges), within the largest half thickness
        and four voxels of them."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        thick = float(self.thicknesses.max()) if self.n_ligaments else 0.0
        reach = 0.5 * thick + 4.0 * float(np.max(grid.spacing))
        data = _slabs(grid, 1, lambda p: self.field(p, reach).astype(np.float32), np.float32)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def _frame_volume(self) -> float:
        inner = np.clip(self.domain.lengths - 2.0 * self.frame, 0.0, None)
        return float(self.domain.volume - np.prod(inner))


@register_kind
class ScaleRichPlates(_Network3D):
    """A scale-rich network of plates: its plates (phase 1, convex polyhedra that never
    overlap) and its voids (phase 0, closed convex cells), which partition the box exactly
    (:func:`grow_plates`); or, with a ``frame``, the box within a frame of that thickness
    (phase 1 too), on which the plates end."""

    kind = "scale_rich_plates"

    @property
    def areas(self) -> np.ndarray:
        """The areas of the plates (of their rims, the sections of their voids by their
        mid-planes)."""
        return self.growth.areas

    @property
    def normals(self) -> np.ndarray:
        return self.growth.normal

    @property
    def volume_fraction(self) -> float:
        """The fraction of the box in the plates and the frame, exact (``1 - sum(void
        volumes) / V``)."""
        return float(1.0 - self.growth.cell_volumes().sum() / self.domain.volume)

    def statistics(self, min_tail: int = 50) -> dict:
        """The power-law tails of the thicknesses, areas and degrees of the plates
        (:func:`morphogen.analysis.power_laws.power_law_tail`, degrees as integers):
        ``1 + 1/alpha``, 5/2 and 5/2 for their exponents in theory. No thicknesses with a
        decay of 0 (one thickness)."""
        from morphogen.analysis.power_laws import power_law_tail

        tail = min(int(min_tail), max(2, self.n_ligaments // 2))
        out = {}
        if np.ptp(self.thicknesses) > 0:  # a decay of 0: one thickness, no tail
            out["thickness"] = power_law_tail(self.thicknesses, min_tail=tail)
        out["area"] = power_law_tail(self.areas, min_tail=tail)
        out["degree"] = power_law_tail(self.degrees(), discrete=True, min_tail=tail)
        return out

    def convex_pieces(self) -> tuple[list[tuple], np.ndarray, np.ndarray]:
        """The convex pieces that partition the box: the plates (phase 1, cells 1 to N), the
        voids (phase 0, cell 0) and the boxes of the frame (phase 1, cell N + 1), for the
        exact meshes of :func:`morphogen.mesh.exact.polyhedral_mesh`."""
        g = self.growth
        pieces = [g.bodies.polyhedron(i) for i in range(len(g.bodies))]
        phase, cells = [1] * len(pieces), list(range(1, len(pieces) + 1))
        for i in range(len(g.cells)):
            poly = g.cells.polyhedron(i)
            if len(poly[2]) >= 4 and polyhedron_volume(poly[0], poly[1], poly[3], poly[4]) > 0:
                pieces.append(poly)
                phase.append(0)
                cells.append(0)
        if self.frame > 0:
            boxes = frame_boxes(self.domain.lower, self.domain.upper, self.frame)
            pieces += boxes
            phase += [1] * len(boxes)
            cells += [g.n + 1] * len(boxes)
        return pieces, np.asarray(phase, dtype=np.int64), np.asarray(cells, dtype=np.int64)

    def _boxes(self) -> np.ndarray:
        return self.growth.bodies.bounds()

    def _elements_field(self, pts, index) -> np.ndarray:
        b = self.growth.bodies
        return _plates_field(pts, index.reach, index.boxes, b.offsets, b.normals, b.planes,
                             index.b_off, index.b_items, index.lo, index.h, index.g)  # fmt: skip

    def summary(self) -> dict:
        """The counts, thicknesses, areas, degrees and anisotropy of the network."""
        g = self.growth
        out = {"ligaments": g.n, "jammed": int(g.jammed), "volume_fraction": self.volume_fraction}
        if g.n:
            deg = g.degrees()
            out.update(
                {
                    "initial_thickness": float(g.thickness[0]),
                    "final_thickness": float(g.thickness[-1]),
                    "mean_area": float(g.areas.mean()),
                    "mean_degree": float(deg.mean()),
                    "max_degree": int(deg.max()),
                    "geometric_anisotropy": self.geometric_anisotropy,
                }
            )
        return out

    def _geometry(self) -> dict:
        g = self.growth
        return {
            "nucleation": g.nucleation.tolist(),
            "normal": g.normal.tolist(),
            "thickness": g.thickness.tolist(),
            "rims": {
                "offsets": g.rim_offsets.tolist(),
                "vertices": g.rim_vertices.tolist(),
                "faces": g.rim_faces.tolist(),
            },
            "bodies": g.bodies.to_dict(),
            "cells": g.cells.to_dict(),
            "jammed": bool(g.jammed),
            "attempts": int(g.attempts),
            "frame": self.frame,
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        rims = geometry["rims"]
        growth = PlateGrowth(
            nucleation=np.asarray(geometry["nucleation"], dtype=float).reshape(-1, 3),
            normal=np.asarray(geometry["normal"], dtype=float).reshape(-1, 3),
            thickness=np.asarray(geometry["thickness"], dtype=float).reshape(-1),
            rim_offsets=np.asarray(rims["offsets"], dtype=np.int64),
            rim_vertices=np.asarray(rims["vertices"], dtype=float).reshape(-1, 3),
            rim_faces=np.asarray(rims["faces"], dtype=np.int64).reshape(-1),
            bodies=Polyhedra.from_dict(geometry["bodies"]),
            cells=Polyhedra.from_dict(geometry["cells"]),
            jammed=bool(geometry.get("jammed", False)),
            attempts=int(geometry.get("attempts", 0)),
        )
        return cls(domain, growth, geometry.get("frame", 0.0), **kwargs)


@register_kind
class ScaleRichBeams(_Network3D):
    """A scale-rich network of beams: cylinders with round ends (phase 1) grown until they
    meet (:func:`grow_beams`), in one open void (phase 0); or, with a ``frame``, within a
    frame of that thickness (phase 1 too), on which they end."""

    kind = "scale_rich_beams"

    def __init__(self, domain: Domain, growth, frame: float = 0.0, **kwargs) -> None:
        super().__init__(domain, growth, frame, **kwargs)
        self._union: float | None = None

    @property
    def lengths(self) -> np.ndarray:
        """The lengths of the axes of the beams."""
        return self.growth.lengths

    @property
    def directions(self) -> np.ndarray:
        return self.growth.direction

    @property
    def volume_fraction(self) -> float:
        """The fraction of the box in the beams (their union, round ends and junctions
        included, cut by the box) and the frame, by quasi-Monte Carlo on
        :func:`union_points` (within about ``1.5 10^-4``)."""
        if self._union is None:
            pts = self.domain.lower + union_points() * self.domain.lengths
            self._union = float(np.count_nonzero(self.field(pts) < 0) / len(pts))
        return self._union

    @property
    def cylinder_fraction(self) -> float:
        """The fraction of the box in the cylinders on the axes of the beams and the
        frame, ``(sum pi r^2 l + V_frame) / V`` (the beams of the mean field of
        :func:`beam_density_estimate`): within a few percent of :attr:`volume_fraction`,
        the round ends and the overlaps at the junctions counted out, the parts of fat beams
        beyond the box counted in."""
        return float((self.growth.volumes().sum() + self._frame_volume()) / self.domain.volume)

    def segments(self):
        """The beams as :class:`morphogen.primitives.Segments` (round caps)."""
        from morphogen.primitives import Segments

        g = self.growth
        return Segments(g.ends[:, 0], g.ends[:, 1], g.radii, round_caps=True)

    def statistics(self, min_tail: int = 50) -> dict:
        """The power-law tails of the thicknesses, lengths and degrees of the beams (the
        exponents of the lengths and degrees depend on ``alpha`` and on the box: see
        docs/theory.md, §93.1). No thicknesses with a decay of 0 (one thickness)."""
        from morphogen.analysis.power_laws import power_law_tail

        tail = min(int(min_tail), max(2, self.n_ligaments // 2))
        out = {}
        if np.ptp(self.thicknesses) > 0:  # a decay of 0: one thickness, no tail
            out["thickness"] = power_law_tail(self.thicknesses, min_tail=tail)
        out["length"] = power_law_tail(self.lengths, min_tail=tail)
        out["degree"] = power_law_tail(self.degrees(), discrete=True, min_tail=tail)
        return out

    def _boxes(self) -> np.ndarray:
        e = self.growth.ends
        r = self.growth.radii[:, None]
        return np.hstack([np.minimum(e[:, 0], e[:, 1]) - r, np.maximum(e[:, 0], e[:, 1]) + r])

    def _elements_field(self, pts, index) -> np.ndarray:
        g = self.growth
        return _beams_field(pts, index.reach, index.boxes, np.ascontiguousarray(g.ends),
                            np.ascontiguousarray(g.radii), index.b_off, index.b_items, index.lo,
                            index.h, index.g)  # fmt: skip

    def summary(self) -> dict:
        """The counts, thicknesses, lengths, degrees and anisotropy of the network."""
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
            "nucleation": g.nucleation.tolist(),
            "direction": g.direction.tolist(),
            "thickness": g.thickness.tolist(),
            "ends": g.ends.tolist(),
            "neighbours": g.neighbours.tolist(),
            "contacts": {"offsets": g.contact_offsets.tolist(), "items": g.contacts.tolist()},
            "jammed": bool(g.jammed),
            "attempts": int(g.attempts),
            "frame": self.frame,
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        growth = BeamGrowth(
            nucleation=np.asarray(geometry["nucleation"], dtype=float).reshape(-1, 3),
            direction=np.asarray(geometry["direction"], dtype=float).reshape(-1, 3),
            thickness=np.asarray(geometry["thickness"], dtype=float).reshape(-1),
            ends=np.asarray(geometry["ends"], dtype=float).reshape(-1, 2, 3),
            neighbours=np.asarray(geometry["neighbours"], dtype=np.int64).reshape(-1, 2),
            contact_offsets=np.asarray(geometry["contacts"]["offsets"], dtype=np.int64),
            contacts=np.asarray(geometry["contacts"]["items"], dtype=np.int64).reshape(-1),
            jammed=bool(geometry.get("jammed", False)),
            attempts=int(geometry.get("attempts", 0)),
        )
        return cls(domain, growth, geometry.get("frame", 0.0), **kwargs)
