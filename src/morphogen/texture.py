"""Crystallographic textures: the orientations of the crystals of a polycrystal — measured
(the maps of electron backscatter diffraction, :mod:`morphogen.io.ebsd`) or computed
(the lattices that slip and twins turn along a loading path) —, their symmetries, their
disorientations, and the figures that show them.

Orientations are Bunge Euler angles (phi1, Phi, phi2) in degrees, an array ``(n, 3)``, or
the matrices ``g`` of :func:`morphogen.orientations.euler_to_matrix`, an array ``(n, 3,
3)``: ``crystal = g @ sample``, the rows of ``g`` are the axes of the crystal in those of
the sample. The axes of the crystal are those of
:mod:`morphogen.homogenization.crystal`: the edges of the cube; for a hexagonal lattice,
``a_1 = [2 -1 -1 0]`` along the first axis and ``c`` along the third.

**Symmetries.** A crystal turned by a rotation ``S`` of its point group is the same
crystal: ``g`` and ``S g`` are one orientation (:func:`symmetries`: 24 rotations of the
cube, 12 of the hexagonal prism, and those of the nine other Laue classes).

**Disorientation.** The rotation that brings the axes of a crystal ``a`` onto those of a
crystal ``b`` is ``g_b g_a^T``; their disorientation is the smallest angle of the
rotations ``S g_b g_a^T``, ``S`` in the group (:func:`disorientation`): at most 62.8
degrees for cubic crystals, 93.8 for hexagonal ones. For two cubic crystals of random
orientations its density is that of Mackenzie (1958) and Handscomb (1958)
(:func:`mackenzie`): mean 40.7 degrees.

**Pole figures.** The normals of the planes ``{hkl}`` of a crystal, ``g^T n`` in the axes
of the sample (:func:`poles`), are shown on a disc (:func:`project`: stereographic or
equal-area projection of the upper hemisphere, the first axis of the sample to the
right, the second upwards, the third towards the reader). Their density
(:func:`pole_figure`) is in multiples of that of a random texture: each pole is spread
by an axial von Mises-Fisher kernel of a given half-width, whose mean over the sphere is
one, so that the mean of the density over the sphere is one for any texture.

**Inverse pole figures.** The direction of the crystal that lies along a direction of
the sample (the axis of a tension), ``g d``, is brought into the fundamental sector of
the lattice by its symmetries (:func:`inverse_poles`: the triangle [001], [011], [111]
of the cube; the sector between [0001], [2 -1 -1 0] and [1 0 -1 0] of the hexagonal
lattice), and gives each crystal a colour (:func:`ipf_colours`: red, green and blue at
these corners).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from morphogen.homogenization.crystal import _rotations, _vector
from morphogen.orientations import euler_to_matrix, matrix_to_euler, uniform_unit_vectors

#: The lattices whose planes and directions are known (pole figures, inverse pole figures)
LATTICES = ("cubic", "hexagonal")
#: The eleven Laue classes (:func:`symmetries`): the class of highest symmetry of each
#: system by the name of the system (m-3m, 6/mmm, 4/mmm, -3m, mmm, 2/m and -1), the four
#: others by their symbol
CLASSES = ("cubic", "hexagonal", "tetragonal", "trigonal", "orthorhombic", "monoclinic",
           "triclinic", "m-3", "6/m", "4/m", "-3")  # fmt: skip
#: Projections of the hemisphere on the unit disc
PROJECTIONS = ("stereographic", "equal_area")
#: The ratio c/a of a hexagonal lattice of close-packed spheres (the default of the planes
#: and directions that depend on it)
IDEAL_RATIO = math.sqrt(8.0 / 3.0)


def _matrices(orientations) -> np.ndarray:
    """The matrices ``(n, 3, 3)`` of orientations given as Euler angles ``(n, 3)`` (or
    ``(3,)``: one crystal), in degrees, or as matrices ``(n, 3, 3)``."""
    a = np.asarray(orientations, dtype=float)
    if a.ndim == 3 and a.shape[1:] == (3, 3):
        return np.ascontiguousarray(a)
    if a.ndim in (1, 2) and a.shape[-1] == 3:
        return euler_to_matrix(a.reshape(-1, 3))
    raise ValueError(
        f"orientations: Euler angles (n, 3) or matrices (n, 3, 3), got an array {a.shape}"
    )


def _turns(axes, angles) -> np.ndarray:
    """The rotations ``(n, 3, 3)`` of ``angles`` (radians) about ``axes`` ``(n, 3)``."""
    u = np.asarray(axes, dtype=float).reshape(-1, 3)
    u = u / np.linalg.norm(u, axis=1)[:, None]
    t = np.asarray(angles, dtype=float).reshape(-1)
    k = np.zeros((len(u), 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -u[:, 2], u[:, 1]
    k[:, 1, 0], k[:, 1, 2] = u[:, 2], -u[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -u[:, 1], u[:, 0]
    return np.eye(3) + np.sin(t)[:, None, None] * k + (1.0 - np.cos(t))[:, None, None] * (k @ k)


def _turn(axis, angle: float) -> np.ndarray:
    """The rotation of ``angle`` (radians) about ``axis``."""
    return _turns(axis, angle)[0]


def symmetries(lattice: str = "cubic") -> np.ndarray:
    """The proper rotations of a Laue class, ``(m, 3, 3)``, in the axes of the crystal:
    24 for ``"cubic"`` (m-3m), 12 for ``"hexagonal"`` (6/mmm), 8 for ``"tetragonal"``
    (4/mmm), 6 for ``"trigonal"`` (-3m, the three-fold axis along the third axis, a
    two-fold one along the first), 4 for ``"orthorhombic"`` (mmm), 2 for ``"monoclinic"``
    (2/m, the two-fold axis along the second axis) and the identity alone for
    ``"triclinic"`` (-1); 12 for ``"m-3"`` (the rotations of the tetrahedron), and the
    6, 4 and 3 rotations about the third axis for ``"6/m"``, ``"4/m"`` and ``"-3"``."""
    if lattice in ("cubic", "hexagonal"):
        return np.array(_rotations(lattice))
    x, y, z = np.eye(3)
    if lattice == "m-3":  # (of the rotations of the cube, the even permutations of its axes)
        return np.array([r for r in _rotations("cubic") if np.linalg.det(np.abs(r)) > 0])
    about_z = {"6/m": 6, "4/m": 4, "-3": 3}
    if lattice in about_z:
        n = about_z[lattice]
        return np.array([_turn(z, k * 2.0 * math.pi / n) for k in range(n)])
    if lattice == "tetragonal":
        quarter = [_turn(z, k * math.pi / 2.0) for k in range(4)]
        return np.array(quarter + [_turn(x, math.pi) @ q for q in quarter])
    if lattice == "trigonal":
        third = [_turn(z, k * 2.0 * math.pi / 3.0) for k in range(3)]
        return np.array(third + [_turn(x, math.pi) @ q for q in third])
    if lattice == "orthorhombic":
        return np.array([np.eye(3)] + [_turn(a, math.pi) for a in (x, y, z)])
    if lattice == "monoclinic":
        return np.array([np.eye(3), _turn(y, math.pi)])
    if lattice == "triclinic":
        return np.eye(3)[None]
    raise ValueError(f"lattice: one of {CLASSES}, got {lattice!r}")


@njit(parallel=True, cache=True)
def _disorientations(ga, gb, ops, out):
    """``out[p]``: the disorientation (radians) of the crystals ``ga[p]`` and ``gb[p]``
    (one of the two arrays may hold a single crystal): the angle of the rotation ``S g_b
    g_a^T`` of largest trace, from its sine (the length of its axial vector) and its
    cosine — exact to rounding for crystals that are close, where the cosine alone gives
    the square root of the rounding."""
    na, nb = ga.shape[0], gb.shape[0]
    for p in prange(out.size):
        a = ga[p % na]
        b = gb[p % nb]
        # d = g_b g_a^T
        d00 = b[0, 0] * a[0, 0] + b[0, 1] * a[0, 1] + b[0, 2] * a[0, 2]
        d01 = b[0, 0] * a[1, 0] + b[0, 1] * a[1, 1] + b[0, 2] * a[1, 2]
        d02 = b[0, 0] * a[2, 0] + b[0, 1] * a[2, 1] + b[0, 2] * a[2, 2]
        d10 = b[1, 0] * a[0, 0] + b[1, 1] * a[0, 1] + b[1, 2] * a[0, 2]
        d11 = b[1, 0] * a[1, 0] + b[1, 1] * a[1, 1] + b[1, 2] * a[1, 2]
        d12 = b[1, 0] * a[2, 0] + b[1, 1] * a[2, 1] + b[1, 2] * a[2, 2]
        d20 = b[2, 0] * a[0, 0] + b[2, 1] * a[0, 1] + b[2, 2] * a[0, 2]
        d21 = b[2, 0] * a[1, 0] + b[2, 1] * a[1, 1] + b[2, 2] * a[1, 2]
        d22 = b[2, 0] * a[2, 0] + b[2, 1] * a[2, 1] + b[2, 2] * a[2, 2]
        best = -4.0
        which = 0
        for s in range(ops.shape[0]):
            o = ops[s]
            trace = (o[0, 0] * d00 + o[0, 1] * d10 + o[0, 2] * d20
                     + o[1, 0] * d01 + o[1, 1] * d11 + o[1, 2] * d21
                     + o[2, 0] * d02 + o[2, 1] * d12 + o[2, 2] * d22)  # fmt: skip
            if trace > best:
                best = trace
                which = s
        o = ops[which]
        # the axial vector of m = o d
        x = (o[2, 0] * d01 + o[2, 1] * d11 + o[2, 2] * d21
             - o[1, 0] * d02 - o[1, 1] * d12 - o[1, 2] * d22)  # fmt: skip
        y = (o[0, 0] * d02 + o[0, 1] * d12 + o[0, 2] * d22
             - o[2, 0] * d00 - o[2, 1] * d10 - o[2, 2] * d20)  # fmt: skip
        z = (o[1, 0] * d00 + o[1, 1] * d10 + o[1, 2] * d20
             - o[0, 0] * d01 - o[0, 1] * d11 - o[0, 2] * d21)  # fmt: skip
        out[p] = math.atan2(0.5 * math.sqrt(x * x + y * y + z * z), 0.5 * (best - 1.0))


def disorientation(a, b, lattice: str = "cubic") -> np.ndarray:
    """The disorientations (degrees) of the crystals ``a`` and ``b`` of a lattice, pair
    by pair (as many of each, or one of the two alone: of every crystal from it): the
    smallest angle of the rotations that bring the one onto the other, over the
    symmetries of the lattice."""
    ga, gb = _matrices(a), _matrices(b)
    if len(ga) != len(gb) and 1 not in (len(ga), len(gb)):
        raise ValueError(f"disorientation: {len(ga)} crystals and {len(gb)}")
    out = np.empty(max(len(ga), len(gb)))
    _disorientations(ga, gb, symmetries(lattice), out)
    return np.degrees(out)


@njit(cache=True)
def _sum_nearest(g, weights, reference, ops, total):
    """``total``: the weighted sum of the crystals ``g``, each taken as that of its
    symmetric equivalents which is nearest to ``reference``."""
    total[:, :] = 0.0
    for k in range(g.shape[0]):
        best = -4.0
        which = 0
        for s in range(ops.shape[0]):
            trace = 0.0
            for i in range(3):
                for j in range(3):
                    d = 0.0
                    for m in range(3):
                        d += g[k, j, m] * reference[i, m]
                    trace += ops[s, i, j] * d
            if trace > best:
                best = trace
                which = s
        for i in range(3):
            for m in range(3):
                e = 0.0
                for j in range(3):
                    e += ops[which, i, j] * g[k, j, m]
                total[i, m] += weights[k] * e


def _nearest_rotation(m: np.ndarray) -> np.ndarray:
    """The rotation nearest to a matrix (its polar factor, a determinant of one)."""
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def mean_orientation(orientations, lattice: str = "cubic", weights=None):
    """The mean orientation of crystals that are close to one another (the points of a
    grain) and their spread: the Euler angles (degrees) of the rotation nearest to the
    weighted sum of their matrices, each crystal taken as that of its symmetric
    equivalents which is nearest to the mean (three passes from the crystal of largest
    weight), and the weighted mean of their disorientations from it (degrees)."""
    g = _matrices(orientations)
    w = np.ones(len(g)) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != (len(g),) or len(g) == 0 or not w.sum() > 0:
        raise ValueError("mean_orientation: one positive weight per crystal")
    w = w / w.sum()
    ops = symmetries(lattice)
    mean = g[int(np.argmax(w))].copy()
    total = np.empty((3, 3))
    for _ in range(3):
        _sum_nearest(g, w, mean, ops, total)
        mean = _nearest_rotation(total)
    spread = float(np.sum(w * disorientation(g, mean[None], lattice)))
    return matrix_to_euler(mean)[0], spread


def crystal_axes(indices, lattice: str = "cubic", *, plane: bool = True,
                 c_over_a: float | None = None) -> np.ndarray:  # fmt: skip
    """The distinct axes of a family of planes ``{hkl}`` (their normals) or of directions
    ``<uvw>`` of a lattice, unit vectors ``(m, 3)`` in the axes of the crystal, one for
    each pair of opposite ones: Miller indices of the cube; Miller-Bravais indices
    ``(hkil)`` of the hexagonal lattice (or three: ``i = -h - k``), of ratio ``c_over_a``
    (:data:`IDEAL_RATIO` by default)."""
    if lattice not in LATTICES:
        raise ValueError(f"lattice: one of {LATTICES} for planes and directions, got {lattice!r}")
    idx = [int(v) for v in indices]
    if lattice == "hexagonal" and len(idx) == 3:
        idx = [idx[0], idx[1], -idx[0] - idx[1], idx[2]]
    if len(idx) != (3 if lattice == "cubic" else 4) or not any(idx):
        raise ValueError(f"indices: three (cubic) or four (hexagonal) integers, got {indices!r}")
    ratio = IDEAL_RATIO if c_over_a is None else float(c_over_a)
    v = _vector(idx, lattice, ratio if lattice == "hexagonal" else None, plane)
    v = v / np.linalg.norm(v)
    found: list[np.ndarray] = []
    for s in symmetries(lattice):
        w = s @ v
        if not any(abs(abs(float(w @ f)) - 1.0) < 1e-9 for f in found):
            first = w[np.flatnonzero(np.abs(w) > 1e-9)[0]]
            found.append(w if first > 0 else -w)
    return np.array(found)


def poles(orientations, indices, lattice: str = "cubic", *, plane: bool = True,
          c_over_a: float | None = None) -> np.ndarray:  # fmt: skip
    """The axes of a family of planes or directions of each crystal in the axes of the
    sample, unit vectors ``(n, m, 3)``: ``g^T v`` for each axis ``v`` of
    :func:`crystal_axes`."""
    axes = crystal_axes(indices, lattice, plane=plane, c_over_a=c_over_a)
    return np.einsum("nji,mj->nmi", _matrices(orientations), axes)


def project(directions, projection: str = "stereographic") -> np.ndarray:
    """The points of the unit disc of directions ``(..., 3)`` of the sample, folded onto
    the upper hemisphere (a direction and its opposite are one axis): the first axis of
    the sample to the right, the second upwards, the third at the centre. At the angle
    ``t`` from the third axis, the radius is ``tan(t / 2)`` (``"stereographic"``:
    angles are kept) or ``sqrt(2) sin(t / 2)`` (``"equal_area"``: areas are kept)."""
    if projection not in PROJECTIONS:
        raise ValueError(f"projection: one of {PROJECTIONS}, got {projection!r}")
    d = np.asarray(directions, dtype=float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    d = np.where(d[..., 2:3] < 0.0, -d, d)
    z = d[..., 2:3]
    scale = 1.0 / (1.0 + z) if projection == "stereographic" else 1.0 / np.sqrt(1.0 + z)
    return d[..., :2] * scale


def unproject(points, projection: str = "stereographic") -> np.ndarray:
    """The directions of the upper hemisphere of points ``(..., 2)`` of the unit disc
    (the inverse of :func:`project`)."""
    if projection not in PROJECTIONS:
        raise ValueError(f"projection: one of {PROJECTIONS}, got {projection!r}")
    p = np.asarray(points, dtype=float)
    r2 = np.sum(p * p, axis=-1, keepdims=True)
    if projection == "stereographic":
        z = (1.0 - r2) / (1.0 + r2)
        return np.concatenate([p * (1.0 + z), z], axis=-1)
    z = 1.0 - r2
    return np.concatenate([p * np.sqrt(np.maximum(2.0 - r2, 0.0)), z], axis=-1)


@njit(parallel=True, cache=True)
def _axial_density(points, axes, weights, kappa, out):
    """``out[q] = sum_p weights[p] K(points[q] . axes[p])`` with the axial von Mises-Fisher
    kernel ``K(t) = kappa cosh(kappa t) / sinh(kappa)``, of mean one over the sphere
    (terms below 1e-12 of the largest are left out)."""
    scale = kappa / (1.0 - math.exp(-2.0 * kappa))
    cut = 1.0 + math.log(1e-12) / kappa
    for q in prange(points.shape[0]):
        x, y, z = points[q, 0], points[q, 1], points[q, 2]
        s = 0.0
        for p in range(axes.shape[0]):
            t = abs(x * axes[p, 0] + y * axes[p, 1] + z * axes[p, 2])
            if t > cut:
                s += weights[p] * (math.exp(kappa * (t - 1.0)) + math.exp(-kappa * (t + 1.0)))
        out[q] = scale * s


def _concentration(halfwidth: float) -> float:
    """The concentration of the kernel whose value is halved at ``halfwidth`` degrees
    from its axis."""
    if not 0.0 < halfwidth < 90.0:
        raise ValueError(f"halfwidth: an angle between 0 and 90 degrees, got {halfwidth!r}")
    return math.log(2.0) / (1.0 - math.cos(math.radians(halfwidth)))


def axis_density(directions, axes, weights=None, halfwidth: float = 10.0) -> np.ndarray:
    """The density of axes on the sphere (unit vectors ``(p, 3)``, a vector and its
    opposite being one axis, of ``weights`` that sum to one by default) at the directions
    ``(q, 3)``, in multiples of the uniform density: each axis spread by a kernel of
    half-width ``halfwidth`` degrees."""
    d = np.ascontiguousarray(np.asarray(directions, dtype=float).reshape(-1, 3))
    a = np.ascontiguousarray(np.asarray(axes, dtype=float).reshape(-1, 3))
    w = np.full(len(a), 1.0 / max(len(a), 1)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(a),):
        raise ValueError("axis_density: one weight per axis")
    out = np.empty(len(d))
    _axial_density(d, a, np.ascontiguousarray(w), _concentration(halfwidth), out)
    return out


@dataclass(frozen=True)
class PoleFigure:
    """The density of the poles of a family of planes (or of directions) on the disc of
    a projection, in multiples of that of a random texture."""

    #: The indices of the family, its lattice and whether they are those of planes
    indices: tuple
    lattice: str
    plane: bool
    #: The projection of the hemisphere, and the half-width (degrees) of the kernel
    projection: str
    halfwidth: float
    #: The coordinates of the columns and of the rows of the grid, from -1 to 1
    x: np.ndarray
    #: ``density[i, j]`` at the point ``(x[i], x[j])`` of the disc; NaN outside it
    density: np.ndarray

    @property
    def maximum(self) -> float:
        """The largest density of the figure."""
        return float(np.nanmax(self.density))

    def at(self, direction) -> float:
        """The density at the grid point of the disc nearest to a direction of the
        sample."""
        p = project(np.asarray(direction, dtype=float), self.projection)
        gap = (self.x[:, None] - p[0]) ** 2 + (self.x[None, :] - p[1]) ** 2
        gap[np.isnan(self.density)] = np.inf
        i, j = np.unravel_index(int(np.argmin(gap)), gap.shape)
        return float(self.density[i, j])


def pole_figure(orientations, indices, lattice: str = "cubic", *, weights=None,
                halfwidth: float = 10.0, resolution: int = 101,
                projection: str = "equal_area", plane: bool = True,
                c_over_a: float | None = None) -> PoleFigure:  # fmt: skip
    """The pole figure of a family of planes ``indices`` of crystals of a lattice
    (``weights``: the volume of each, equal by default): the density of their poles on a
    grid of ``resolution`` points across the disc of a projection, in multiples of that of
    a random texture, each pole spread by a kernel of half-width ``halfwidth`` degrees."""
    if resolution < 3:
        raise ValueError("resolution: three points at least across the disc")
    directions = poles(orientations, indices, lattice, plane=plane, c_over_a=c_over_a)
    n, m, _ = directions.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != (n,) or n == 0 or not w.sum() > 0:
        raise ValueError("pole_figure: one positive weight per crystal")
    each = np.repeat(w / (w.sum() * m), m)
    x = np.linspace(-1.0, 1.0, int(resolution))
    grid = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1)
    inside = np.sum(grid * grid, axis=-1) <= 1.0 + 1e-12
    density = np.full(grid.shape[:2], np.nan)
    density[inside] = axis_density(unproject(grid[inside], projection),
                                   directions.reshape(-1, 3), each, halfwidth)  # fmt: skip
    return PoleFigure(tuple(int(v) for v in indices), lattice, bool(plane), projection,
                      float(halfwidth), x, density)  # fmt: skip


def fundamental_directions(directions, lattice: str = "cubic") -> np.ndarray:
    """Directions of a crystal ``(n, 3)`` (in its axes) brought into the fundamental
    sector of the lattice by its symmetries and the inversion: unit vectors with ``0 <=
    x <= y <= z`` for the cube (the triangle of [001], [011] and [111]); with ``z >= 0``
    and an azimuth between 0 (``[2 -1 -1 0]``) and 30 degrees (``[1 0 -1 0]``) for the
    hexagonal lattice."""
    if lattice not in LATTICES:
        raise ValueError(f"lattice: one of {LATTICES} for inverse pole figures, got {lattice!r}")
    c = np.asarray(directions, dtype=float).reshape(-1, 3)
    c = c / np.linalg.norm(c, axis=1)[:, None]
    if lattice == "cubic":
        return np.sort(np.abs(c), axis=1)
    radius = np.hypot(c[:, 0], c[:, 1])
    azimuth = np.mod(np.arctan2(c[:, 1], c[:, 0]), math.pi / 3.0)
    azimuth = np.where(azimuth > math.pi / 6.0, math.pi / 3.0 - azimuth, azimuth)
    return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), np.abs(c[:, 2])], axis=1)


def inverse_poles(orientations, direction=(0.0, 0.0, 1.0), lattice: str = "cubic") -> np.ndarray:
    """The direction of each crystal that lies along a direction of the sample, ``g d``,
    in the fundamental sector of the lattice (:func:`fundamental_directions`): unit
    vectors ``(n, 3)`` in the axes of the crystal."""
    if lattice not in LATTICES:
        raise ValueError(f"lattice: one of {LATTICES} for inverse pole figures, got {lattice!r}")
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return fundamental_directions(np.einsum("nij,j->ni", _matrices(orientations), d), lattice)


def direction_colours(directions, lattice: str = "cubic") -> np.ndarray:
    """The colours ``(n, 3)`` (red, green, blue, from 0 to 1) of directions of a crystal:
    red, green and blue at the corners of the fundamental sector — [001], [011] and
    [111] of the cube; [0001], [2 -1 -1 0] and [1 0 -1 0] of the hexagonal lattice — and
    their mixtures in between, the largest of the three brought to one."""
    c = fundamental_directions(directions, lattice)
    if lattice == "cubic":
        rgb = np.stack([c[:, 2] - c[:, 1], c[:, 1] - c[:, 0], math.sqrt(3.0) * c[:, 0]], axis=1)
    else:
        share = np.arctan2(c[:, 1], c[:, 0]) / (math.pi / 6.0)
        basal = np.hypot(c[:, 0], c[:, 1])
        rgb = np.stack([c[:, 2], basal * (1.0 - share), basal * share], axis=1)
    rgb = np.maximum(rgb, 0.0)
    return rgb / np.maximum(rgb.max(axis=1, keepdims=True), 1e-300)


def ipf_colours(orientations, direction=(0.0, 0.0, 1.0), lattice: str = "cubic") -> np.ndarray:
    """The colours ``(n, 3)`` of crystals by the direction that they have along a
    direction of the sample (:func:`direction_colours` of ``g d``): the colours of an
    inverse pole figure map."""
    return direction_colours(inverse_poles(orientations, direction, lattice), lattice)


#: The corners of the fundamental sector of a lattice (crystal directions), with their
#: names: the colours red, green and blue of :func:`direction_colours`
SECTOR_CORNERS = {
    "cubic": (((0.0, 0.0, 1.0), "001"), ((0.0, 1.0, 1.0), "011"), ((1.0, 1.0, 1.0), "111")),
    "hexagonal": (((0.0, 0.0, 1.0), "0001"), ((1.0, 0.0, 0.0), "2-1-10"),
                  ((math.sqrt(3.0), 1.0, 0.0), "10-10")),
}  # fmt: skip


def sector_points(directions, lattice: str = "cubic") -> np.ndarray:
    """The points ``(n, 2)`` of the fundamental sector, as an inverse pole figure shows
    it (the stereographic projection about the third axis of the crystal), of directions
    of a crystal: [001] at the origin, [011] on the first axis and [111] above it for the
    cube; [0001] at the origin, ``[2 -1 -1 0]`` on the first axis and ``[1 0 -1 0]``
    above it for the hexagonal lattice."""
    c = fundamental_directions(directions, lattice)
    p = c[:, :2] / (1.0 + c[:, 2:3])
    return p[:, ::-1] if lattice == "cubic" else p


def _sector_directions(points, lattice: str):
    """The directions of the points of the plane of an inverse pole figure, and whether
    they are in the fundamental sector."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    d = unproject(p[:, ::-1] if lattice == "cubic" else p, "stereographic")
    if lattice == "cubic":
        inside = (d[:, 0] >= -1e-12) & (d[:, 0] <= d[:, 1] + 1e-12) & (d[:, 1] <= d[:, 2] + 1e-12)
    else:
        azimuth = np.arctan2(d[:, 1], d[:, 0])
        inside = (azimuth >= -1e-12) & (azimuth <= math.pi / 6.0 + 1e-12)
        inside &= np.sum(p * p, axis=1) <= 1.0 + 1e-12
    return d, inside


@dataclass(frozen=True)
class InversePoleFigure:
    """The density of the directions of the crystals that lie along a direction of the
    sample, on the fundamental sector of the lattice (the plane of
    :func:`sector_points`), in multiples of that of a random texture."""

    #: The direction of the sample, the lattice and the half-width (degrees) of the kernel
    direction: tuple
    lattice: str
    halfwidth: float
    #: The coordinates of the columns and of the rows of the grid
    x: np.ndarray
    y: np.ndarray
    #: ``density[i, j]`` at the point ``(x[i], y[j])``; NaN outside the sector
    density: np.ndarray

    @property
    def maximum(self) -> float:
        """The largest density of the figure."""
        return float(np.nanmax(self.density))

    def at(self, direction) -> float:
        """The density at the grid point of the sector nearest to a direction of the
        crystal."""
        p = sector_points(np.asarray(direction, dtype=float), self.lattice)[0]
        gap = (self.x[:, None] - p[0]) ** 2 + (self.y[None, :] - p[1]) ** 2
        gap[np.isnan(self.density)] = np.inf
        i, j = np.unravel_index(int(np.argmin(gap)), gap.shape)
        return float(self.density[i, j])


def inverse_pole_figure(orientations, direction=(0.0, 0.0, 1.0), lattice: str = "cubic", *,
                        weights=None, halfwidth: float = 10.0,
                        resolution: int = 101) -> InversePoleFigure:  # fmt: skip
    """The inverse pole figure of a direction of the sample (the axis of a tension):
    the density of the directions ``g d`` of the crystals (``weights``: the volume of
    each, equal by default) with all their equivalents by the symmetries of the lattice,
    on a grid of ``resolution`` points across the fundamental sector, in multiples of
    that of a random texture, each direction spread by a kernel of half-width
    ``halfwidth`` degrees."""
    if lattice not in LATTICES:
        raise ValueError(f"lattice: one of {LATTICES} for inverse pole figures, got {lattice!r}")
    if resolution < 3:
        raise ValueError("resolution: three points at least across the sector")
    g = _matrices(orientations)
    w = np.ones(len(g)) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != (len(g),) or len(g) == 0 or not w.sum() > 0:
        raise ValueError("inverse_pole_figure: one positive weight per crystal")
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    ops = symmetries(lattice)
    along = np.einsum("sij,nj->nsi", ops, np.einsum("nij,j->ni", g, d))
    each = np.repeat(w / (w.sum() * len(ops)), len(ops))
    width, height = (math.sqrt(2.0) - 1.0, 0.3661) if lattice == "cubic" else (1.0, 0.5)
    x = np.linspace(0.0, width, int(resolution))
    y = np.linspace(0.0, height, max(3, round(resolution * height / width)))
    grid = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    points, inside = _sector_directions(grid.reshape(-1, 2), lattice)
    density = np.full(len(points), np.nan)
    density[inside] = axis_density(points[inside], along.reshape(-1, 3), each, halfwidth)
    return InversePoleFigure(tuple(float(v) for v in d), lattice, float(halfwidth), x, y,
                             density.reshape(len(x), len(y)))  # fmt: skip


#: The angles (degrees) at which the density of Mackenzie changes its expression, and the
#: largest disorientation of two cubic crystals
MACKENZIE_ANGLES = (
    45.0,
    60.0,
    math.degrees(2.0 * math.atan(math.sqrt(2.0) * (math.sqrt(2.0) - 1.0))),
    math.degrees(2.0 * math.atan(math.sqrt(2.0 * (math.sqrt(2.0) - 1.0) ** 2
                                           + (3.0 - 2.0 * math.sqrt(2.0)) ** 2))),
)  # fmt: skip


def mackenzie(angles) -> np.ndarray:
    """The density, per degree, of the disorientation of two cubic crystals of random
    orientations (Mackenzie 1958; Handscomb 1958) at ``angles`` (degrees): ``(2/15) (1 -
    cos t)`` up to 45 degrees — 24 times the density of the angle of a random rotation —,
    then less, as the rotations leave the fundamental zone through its faces {100} (45
    degrees) and {111} (60 degrees), its edges (60.72) and its corners (62.80, the
    largest disorientation). Its mean is 40.74 degrees."""
    t = np.radians(np.asarray(angles, dtype=float))
    r1, r2, r3, r4 = np.radians(MACKENZIE_ANGLES)
    root = math.sqrt(2.0)
    a = 3.0 * (root - 1.0)
    b = a + 4.0 / math.sqrt(3.0)
    rise = 1.0 - np.cos(t)
    out = np.zeros_like(t)
    m = (t >= 0.0) & (t <= r1)
    out[m] = rise[m]
    m = (t > r1) & (t <= r2)
    out[m] = a * np.sin(t[m]) - 2.0 * rise[m]
    m = (t > r2) & (t <= r4)
    out[m] = b * np.sin(t[m]) - 6.0 * rise[m]
    m = (t > r3) & (t < r4)
    cot = 1.0 / np.tan(0.5 * t[m])
    x = (root - 1.0) / np.sqrt(1.0 - (root - 1.0) ** 2 * cot**2)
    y = (root - 1.0) ** 2 / np.sqrt(3.0 - cot**2)
    first = 2.0 * (root - 1.0) * np.arccos(np.clip(x * cot, -1.0, 1.0))
    first += np.arccos(np.clip(y * cot, -1.0, 1.0)) / math.sqrt(3.0)
    second = 2.0 * np.arccos(np.clip((root + 1.0) * x / root, -1.0, 1.0))
    second += np.arccos(np.clip((root + 1.0) * y / root, -1.0, 1.0))
    out[m] += (12.0 / math.pi) * (second * rise[m] - first * np.sin(t[m]))
    out[t >= r4] = 0.0
    return (2.0 / 15.0) * out


def boundary_disorientations(labels, orientations, lattice: str = "cubic"):
    """The disorientations across the boundaries of the grains of an image of labels (2D
    or 3D; negative labels: no grain) whose grains have the given orientations: the pairs
    of neighbouring grains ``(b, 2)``, the number of faces of pixels (voxels) that each
    pair shares, and its disorientation (degrees)."""
    lab = np.asarray(labels)
    pairs = []
    for axis in range(lab.ndim):
        a = np.moveaxis(lab, axis, 0)
        first, second = a[:-1].ravel(), a[1:].ravel()
        keep = (first != second) & (first >= 0) & (second >= 0)
        pairs.append(np.stack([np.minimum(first[keep], second[keep]),
                               np.maximum(first[keep], second[keep])], axis=1))  # fmt: skip
    both = np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=np.int64)
    if not len(both):
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0)
    unique, counts = np.unique(both, axis=0, return_counts=True)
    g = _matrices(orientations)
    angles = disorientation(g[unique[:, 0]], g[unique[:, 1]], lattice)
    return unique.astype(np.int64), counts.astype(np.int64), angles


def sample_texture(rng: np.random.Generator, orientations, n: int, *, weights=None,
                   spread: float = 0.0) -> np.ndarray:  # fmt: skip
    """``n`` orientations (Euler angles, degrees) drawn from a measured texture: each is
    one of the given crystals, drawn with its weight (the area of a grain of a map;
    equal by default), then turned about a random axis by an angle of the half-normal
    law of standard deviation ``spread`` degrees (0: the measured orientations
    themselves). The crystals of a synthetic polycrystal then have the texture of the
    map."""
    g = _matrices(orientations)
    w = np.ones(len(g)) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != (len(g),) or len(g) == 0 or not w.sum() > 0 or np.any(w < 0):
        raise ValueError("sample_texture: one weight per crystal, none negative")
    if spread < 0:
        raise ValueError("spread: an angle, not negative")
    drawn = g[rng.choice(len(g), size=int(n), p=w / w.sum())]
    if spread > 0:
        axes = uniform_unit_vectors(rng, int(n))
        angles = np.radians(np.abs(rng.normal(0.0, float(spread), int(n))))
        drawn = drawn @ _turns(axes, angles)
    return matrix_to_euler(drawn)


__all__ = [
    "CLASSES",
    "IDEAL_RATIO",
    "LATTICES",
    "MACKENZIE_ANGLES",
    "PROJECTIONS",
    "SECTOR_CORNERS",
    "InversePoleFigure",
    "PoleFigure",
    "axis_density",
    "boundary_disorientations",
    "crystal_axes",
    "direction_colours",
    "disorientation",
    "fundamental_directions",
    "inverse_pole_figure",
    "inverse_poles",
    "ipf_colours",
    "mackenzie",
    "mean_orientation",
    "pole_figure",
    "poles",
    "project",
    "sample_texture",
    "sector_points",
    "symmetries",
    "unproject",
]
