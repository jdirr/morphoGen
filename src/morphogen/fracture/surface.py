"""Cracks of any surface and front in 3D: the surface swept by the successive positions of
a front, which advances by its own amount and kinks by its own angle at each of its
points.

The planar cracks of :mod:`morphogen.fracture.lefm` have a straight or circular front that
advances uniformly, and the kinked cracks of :mod:`morphogen.fracture.mixed` a path in a
plane, the same all along a straight front. A :class:`SurfaceCrack` is a grid of points,
``rows x stations``: each row a position of the front, from the centre of the crack (a
closed front: a crack inside the body, such as an ellipse) or from its mouth (an open
front, which leaves the body by its two ends: a through, edge or surface crack) to the
front, the last row; each station follows its own path across the rows. Between the rows
the surface is that of the triangles of the grid.

*Front.* At each station the tangent ``τ`` of the front (along the stations), the
direction of propagation ``d`` (that of the last move of the station, made normal to
``τ``) and the normal ``n = d × τ`` of the surface: the axes ``(d, n, t = d × n)`` of the
asymptotic fields. They are interpolated along the front, whose curvature the auxiliary
fields of the interaction integrals then see (Gosz & Moran 2002). A point has the
coordinates ``x1 = (x - f) · d`` and ``x2 = (x - f) · n`` from its nearest point ``f`` of the
front: ``ψ = x1`` (negative behind the front), and ``φ`` the signed distance to the
triangles behind the front and ``x2`` ahead of it, the surface continued along ``d``: the
two level sets that the conforming meshes and the opening of the lips take
(:func:`~morphogen.fracture.lefm.crack_mesh`, :func:`~morphogen.fracture.lefm.crack_faces`).
An open front is continued beyond its ends, and its surface behind its mouth.

*Growth* (:meth:`SurfaceCrack.grown`): a new row, each station moved by its advance at its
angle from ``d`` towards ``n``. :func:`propagate_front` solves the body on a mesh rebuilt
for each crack, takes the energy release rate and the stress intensity factors at
stations along the front (:meth:`~morphogen.fracture.mixed.CrackSolution.modes`), and
moves each point of the front in the direction of the largest hoop stress of its own
``K_I`` and ``K_II`` (Erdogan & Sih 1963), by an advance that grows with its own energy
release rate, ``da = Δa (G / G_max)^(m / 2)`` (Paris' law ``da/dN = C ΔK^m`` for the
cycles that advance the fastest point by ``Δa``): an elliptical crack in tension grows
towards a circle, a crack inclined to the load turns its surface towards the plane normal
to it, by a different angle at each point of its front.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

#: How far the surface of an open crack is continued beyond the ends of its front and
#: behind its mouth, in units of its size
FAR = 1e3

_COUNT = 400  # points of the front


@njit(cache=True)
def _closest_on_triangle(p, a, b, c):
    """The barycentric coordinates of the point of the triangle ``abc`` nearest ``p``
    (Ericson 2005)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return 1.0, 0.0, 0.0
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return 0.0, 1.0, 0.0
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return 1.0 - v, v, 0.0
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0.0 and d5 <= d6:
        return 0.0, 0.0, 1.0
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return 1.0 - w, 0.0, w
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return 0.0, 1.0 - w, w
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return 1.0 - v - w, v, w


@njit(parallel=True, cache=True)
def _signed_distance(points, vertices, triangles, normals):
    """The distance of each point to the triangles (rows of three vertices), signed by
    the normal of the surface interpolated at its nearest point (``normals``: those of
    the vertices)."""
    out = np.empty(len(points))
    for i in prange(len(points)):
        p = points[i]
        best = np.inf
        side = 1.0
        for k in range(len(triangles)):
            a = vertices[triangles[k, 0]]
            b = vertices[triangles[k, 1]]
            c = vertices[triangles[k, 2]]
            u, v, w = _closest_on_triangle(p, a, b, c)
            q = p - (u * a + v * b + w * c)
            d2 = q @ q
            if d2 < best:
                best = d2
                normal = (
                    u * normals[triangles[k, 0]]
                    + v * normals[triangles[k, 1]]
                    + w * normals[triangles[k, 2]]
                )
                side = 1.0 if q @ normal >= 0.0 else -1.0
        out[i] = side * math.sqrt(best)
    return out


@njit(parallel=True, cache=True)
def _nearest_on_front(points, front, closed):
    """The segment of the front nearest each point and the position along it (0 at its
    first station, 1 at its second; beyond them on the end segments of an open front,
    which are continued)."""
    m = len(front)
    count = m if closed else m - 1
    segment = np.empty(len(points), dtype=np.int64)
    position = np.empty(len(points))
    for i in prange(len(points)):
        p = points[i]
        best = np.inf
        for k in range(count):
            a = front[k]
            e = front[(k + 1) % m] - a
            u = ((p - a) @ e) / (e @ e)
            low = -np.inf if (not closed and k == 0) else 0.0
            high = np.inf if (not closed and k == count - 1) else 1.0
            u = min(max(u, low), high)
            q = p - a - u * e
            d2 = q @ q
            if d2 < best:
                best = d2
                segment[i] = k
                position[i] = u
    return segment, position


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


class SurfaceCrack:
    """A crack of any surface (see the module): ``points`` (rows x stations x 3), the
    successive positions of its front from its centre (``closed``: the stations go round
    a front inside the body; the first row may be one point repeated) or from its mouth
    (an open front, whose first and last stations lie outside the body or on its
    boundary); the front is the last row. With the normal ``n`` of the surface towards
    the reader, the stations of a closed front go anticlockwise."""

    curved = True  # the auxiliary fields of the interaction integrals turn along the front

    def __init__(self, points, closed: bool = True) -> None:
        p = np.array(points, dtype=float)
        if p.ndim != 3 or p.shape[2] != 3 or p.shape[0] < 2 or p.shape[1] < (3 if closed else 2):
            raise ValueError(
                "SurfaceCrack: points (rows x stations x 3), two rows at least, three "
                "stations around a closed front (two along an open one)"
            )
        if not np.all(np.isfinite(p)):
            raise ValueError("SurfaceCrack: finite points")
        p.setflags(write=False)
        self.points = p
        self.closed = bool(closed)
        front = p[-1]
        step = np.roll(front, -1, axis=0) - front if self.closed else np.diff(front, axis=0)
        lengths = np.linalg.norm(step, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError("SurfaceCrack: two stations of the front at the same point")
        self._lengths = lengths
        self._length = float(lengths.sum())
        if self.closed:  # the tangent at a station: across its two neighbours
            tangent = _unit(np.roll(front, -1, axis=0) - np.roll(front, 1, axis=0))
        else:
            tangent = np.empty_like(front)
            tangent[1:-1] = front[2:] - front[:-2]
            tangent[0], tangent[-1] = step[0], step[-1]
            tangent = _unit(tangent)
        back = front - p[-2]
        back = back - np.einsum("ij,ij->i", back, tangent)[:, None] * tangent
        if np.any(np.linalg.norm(back, axis=1) <= 1e-12 * self._length):
            raise ValueError("SurfaceCrack: the last two rows meet, or the front runs along them")
        self._d = _unit(back)
        self._n = _unit(np.cross(self._d, tangent))
        self._t = np.cross(self._d, self._n)
        # the coordinate of the stations: the angle of a closed front (the length along it
        # over its whole length), the length along an open one
        arc = np.concatenate([[0.0], np.cumsum(lengths)])
        self._arc = arc
        self._s = -np.pi + 2 * np.pi * arc[:-1] / self._length if self.closed else arc
        self._surface()

    # -- the surface behind the front ---------------------------------------------------

    def _surface(self) -> None:
        """The triangles of the grid of the rows, with those that continue the surface
        of an open crack beyond the ends of its front and behind its mouth, and the
        normals of their vertices."""
        grid = self.points
        scale = float(np.ptp(grid.reshape(-1, 3), axis=0).max())
        if not self.closed:
            far = FAR * scale
            first = _unit(grid[:, 1] - grid[:, 0])
            last = _unit(grid[:, -1] - grid[:, -2])
            grid = np.concatenate(
                [(grid[:, 0] - far * first)[:, None], grid, (grid[:, -1] + far * last)[:, None]],
                axis=1,
            )
            behind = grid[0] - far * _unit(grid[1] - grid[0])
            grid = np.concatenate([behind[None], grid], axis=0)
        rows, cols = grid.shape[:2]
        index = np.arange(rows * cols).reshape(rows, cols)
        nxt = np.roll(index, -1, axis=1)
        if self.closed:
            a, b, c, e = index[:-1], index[1:], nxt[:-1], nxt[1:]
        else:
            a, b, c, e = index[:-1, :-1], index[1:, :-1], index[:-1, 1:], index[1:, 1:]
        triangles = np.concatenate(
            [
                np.stack([a, b, c], axis=-1).reshape(-1, 3),
                np.stack([b, e, c], axis=-1).reshape(-1, 3),
            ]
        )
        vertices = grid.reshape(-1, 3)
        corners = vertices[triangles]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        area = np.linalg.norm(cross, axis=1)
        keep = area > 1e-14 * scale * scale  # (the fan around a centre repeated)
        triangles, cross = triangles[keep], cross[keep]
        normals = np.zeros_like(vertices)
        for k in range(3):
            np.add.at(normals, triangles[:, k], cross)
        self._vertices = np.ascontiguousarray(vertices)
        self._triangles = np.ascontiguousarray(triangles)
        self._normals = np.ascontiguousarray(_unit(normals))
        self._scale = scale

    # -- constructors ---------------------------------------------------------------------

    @classmethod
    def ellipse(cls, centre, normal, a: float, b: float | None = None, *, major=None,
                stations: int = 48, rows: int = 4) -> SurfaceCrack:  # fmt: skip
        """A planar elliptical crack of semi-axes ``a`` (along ``major``, by default the
        axis of the frame least aligned with the normal, made normal to it) and ``b`` (by
        default ``a``: a penny), its plane through ``centre`` normal to ``normal``: rows
        from its centre to its front."""
        n = _unit(np.asarray(normal, dtype=float))
        b = float(a) if b is None else float(b)
        if not (a > 0 and b > 0) or stations < 8 or rows < 2:
            raise ValueError("SurfaceCrack.ellipse: positive semi-axes, 8 stations, 2 rows")
        if major is None:
            major = np.eye(3)[int(np.argmin(np.abs(n)))]
        e1 = np.asarray(major, dtype=float)
        e1 = _unit(e1 - (e1 @ n) * n)
        e2 = np.cross(n, e1)
        angle = 2 * np.pi * np.arange(stations) / stations
        front = float(a) * np.outer(np.cos(angle), e1) + b * np.outer(np.sin(angle), e2)
        radii = np.linspace(0.0, 1.0, rows)
        return cls(np.asarray(centre, dtype=float) + radii[:, None, None] * front[None], True)

    @classmethod
    def of(cls, crack, lower=None, upper=None, *, stations: int = 48, rows: int = 4):
        """The surface crack of a planar :class:`~morphogen.fracture.lefm.Crack` (a penny:
        a closed front) or of a straight one or a
        :class:`~morphogen.fracture.mixed.KinkedCrack` in the box ``[lower, upper]`` (an
        open front, continued beyond the box)."""
        if getattr(crack, "kind", None) == "penny":
            return cls.ellipse(crack.point, crack.normal, crack.size, stations=stations, rows=rows)
        from morphogen.fracture.mixed import KinkedCrack

        path = crack if isinstance(crack, KinkedCrack) else KinkedCrack.of(crack)
        if lower is None or upper is None:
            raise ValueError("SurfaceCrack.of: the box of a crack with a straight front")
        axis = np.asarray(path.axis)
        corners = np.array(
            [[(lower, upper)[i][k] for k, i in enumerate(bits)] for bits in np.ndindex(2, 2, 2)]
        )
        along = (corners - np.asarray(path.points[0])) @ axis
        margin = 0.05 * (along.max() - along.min())
        # the stations against the axis: the normal d × τ is then that of the path
        s = np.linspace(along.max() + margin, along.min() - margin, stations)
        base = np.asarray(path.points)
        return cls(base[:, None, :] + s[None, :, None] * axis[None, None, :], False)

    # -- the front -------------------------------------------------------------------------

    @property
    def stations(self) -> int:
        return self.points.shape[1]

    @property
    def s(self) -> np.ndarray:
        """The coordinate of the stations along the front: an angle from -π for a closed
        front (the length along it over its whole length), a length for an open one."""
        return self._s

    @property
    def length(self) -> float:
        """The length of the front."""
        return self._length

    @property
    def size(self) -> float:
        """The size of the crack: the mean distance of the front to its centre (a closed
        front), or the mean length of the paths of its stations."""
        if self.closed:
            front = self.points[-1]
            return float(np.linalg.norm(front - front.mean(axis=0), axis=1).mean())
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=2).sum(axis=0).mean())

    @property
    def reach(self) -> float:
        """How far behind the front the surface stays in the plane of the axes ``d`` and
        ``t`` of the front, at the station where it does least: the domain of a virtual
        advance tangent to the lips (the size of a planar crack, the last advance of a
        kinked one)."""
        back = self.points[::-1] - self.points[-1][None]  # the rows, from the front
        off = np.abs(np.einsum("rij,ij->ri", back, self._n))
        behind = -np.einsum("rij,ij->ri", back, self._d)
        planar = np.logical_and.accumulate(off <= 1e-6 * self._scale, axis=0)
        return float(np.where(planar, behind, 0.0).max(axis=0).min())

    def area(self) -> float:
        """The area of the surface (without its continuations)."""
        grid = self.points
        nxt = np.roll(grid, -1, axis=1) if self.closed else grid[:, 1:]
        cur = grid if self.closed else grid[:, :-1]
        a, b, c, e = cur[:-1], cur[1:], nxt[:-1], nxt[1:]
        first = np.linalg.norm(np.cross(b - a, c - a), axis=-1)
        second = np.linalg.norm(np.cross(e - b, c - b), axis=-1)
        return float(0.5 * (first.sum() + second.sum()))

    def _nearest(self, x):
        """Of points: the nearest point of the front, its axes ``d``, ``n``, ``t``
        (interpolated between the stations) and its coordinate along the front."""
        x = np.ascontiguousarray(np.atleast_2d(np.asarray(x, dtype=float)))
        front = np.ascontiguousarray(self.points[-1])
        segment, u = _nearest_on_front(x, front, self.closed)
        after = (segment + 1) % self.stations
        f = front[segment] + u[:, None] * (front[after] - front[segment])
        w = np.clip(u, 0.0, 1.0)[:, None]
        n = _unit((1.0 - w) * self._n[segment] + w * self._n[after])
        d = (1.0 - w) * self._d[segment] + w * self._d[after]
        d = _unit(d - np.einsum("ij,ij->i", d, n)[:, None] * n)
        s = self._arc[segment] + u * self._lengths[segment]
        if self.closed:
            s = -np.pi + 2 * np.pi * s / self._length
        return x, f, d, n, np.cross(d, n), s

    def local(self, x):
        """The coordinates of points along the axes ``d`` and ``n`` of the front."""
        x, f, d, n, _, _ = self._nearest(x)
        y = x - f
        return np.einsum("ij,ij->i", y, d), np.einsum("ij,ij->i", y, n)

    def psi(self, x) -> np.ndarray:
        """Distance beyond the front along its direction of propagation (negative behind
        it)."""
        return self.local(x)[0]

    def phi(self, x) -> np.ndarray:
        """Signed distance to the surface behind the front; ahead of it, to the surface
        continued along the direction of propagation."""
        x1, x2 = self.local(x)
        x = np.ascontiguousarray(np.atleast_2d(np.asarray(x, dtype=float)))
        behind = x1 < 0.0
        out = x2.copy()
        if np.any(behind):
            out[behind] = _signed_distance(
                x[behind], self._vertices, self._triangles, self._normals
            )
        return out

    def frame(self, x):
        """As :meth:`~morphogen.fracture.lefm.Crack.frame`: the distance ``r`` to the front,
        the axes ``n``, ``d``, ``t`` of its nearest point and the coordinate ``s`` along
        it."""
        x, f, d, n, t, s = self._nearest(x)
        y = x - f
        r = np.hypot(np.einsum("ij,ij->i", y, d), np.einsum("ij,ij->i", y, n))
        return r, n, d, t, s

    def front(self, lower, upper, count: int = _COUNT):
        """Points of the front inside the box ``[lower, upper]`` and the length of front
        that each represents."""
        front = self.points[-1]
        ends = np.vstack([front, front[:1]]) if self.closed else front
        at = (np.arange(count) + 0.5) * self._length / count
        points = np.stack([np.interp(at, self._arc, ends[:, k]) for k in range(3)], axis=1)
        inside = np.all(
            (points >= np.asarray(lower) - 1e-9) & (points <= np.asarray(upper) + 1e-9), axis=1
        )
        return points[inside], np.full(int(inside.sum()), self._length / count)

    # -- growth ----------------------------------------------------------------------------

    def grown(self, da, angle=0.0) -> SurfaceCrack:
        """The crack whose stations have advanced by ``da`` at ``angle`` (radians) from
        their direction of propagation towards the normal of the surface (numbers, or one
        value per station): a new row. A negative ``da``, the same at all the stations
        and shorter than the last row, moves the front back along it."""
        da = np.broadcast_to(np.asarray(da, dtype=float), (self.stations,))
        angle = np.broadcast_to(np.asarray(angle, dtype=float), (self.stations,))
        if np.all(da < 0.0):
            if np.any(angle != 0.0) or np.any(-da >= (1 - 1e-9) * self.reach):
                raise ValueError("SurfaceCrack.grown: shortened within its last row, straight")
            front = self.points[-1] + da[:, None] * self._d
            return SurfaceCrack(np.concatenate([self.points[:-1], front[None]]), self.closed)
        if np.any(da <= 0.0):
            raise ValueError("SurfaceCrack.grown: positive advances (or the same negative one)")
        step = np.cos(angle)[:, None] * self._d + np.sin(angle)[:, None] * self._n
        front = np.round(self.points[-1] + da[:, None] * step, 10)
        return SurfaceCrack(np.concatenate([self.points, front[None]]), self.closed)


@dataclass
class FrontGrowth:
    """The steps of the growth of a :class:`SurfaceCrack`: the ``cracks``, the coordinates
    ``s`` of the stations of the interaction integrals along the front and, at them (steps
    x stations), the energy release ``rates``, the stress intensity ``factors`` (a last
    axis: the three modes) and the ``angles`` of kinking; the largest advance of each
    step and the ``cycles`` (Paris), with ``keep`` the solutions."""

    cracks: list
    s: np.ndarray
    rates: np.ndarray
    factors: np.ndarray
    angles: np.ndarray
    advances: np.ndarray
    cycles: np.ndarray
    solutions: list = field(default_factory=list, repr=False)

    def rows(self) -> list[dict]:
        """A row per step: the size and the area of the crack, the extremes of the energy
        release rate and of the factors along its front, the largest angle of kinking."""
        out = []
        for k, crack in enumerate(self.cracks):
            f = self.factors[k]
            out.append(
                {
                    "step": k,
                    "size": crack.size,
                    "area": crack.area(),
                    "front_length": crack.length,
                    "G_min": float(self.rates[k].min()),
                    "G_max": float(self.rates[k].max()),
                    "K_I_min": float(f[:, 0].min()),
                    "K_I_max": float(f[:, 0].max()),
                    "K_II_max": float(np.abs(f[:, 1]).max()),
                    "K_III_max": float(np.abs(f[:, 2]).max()),
                    "angle_max_deg": float(np.degrees(np.abs(self.angles[k]).max())),
                    "N": float(self.cycles[k]),
                }
            )
        return out


def at_stations(crack: SurfaceCrack, s, values, harmonics: int | None = None) -> np.ndarray:
    """Values given at the coordinates ``s`` along the front (increasing), at the stations
    of the crack: interpolated linearly, or, around a closed front whose ``s`` are evenly
    spaced, by their Fourier series up to ``harmonics`` (a smooth distribution, less the
    noise of the values from station to station)."""
    s, values = np.asarray(s, dtype=float), np.asarray(values, dtype=float)
    if not crack.closed:
        return np.interp(crack.s, s, values)
    if harmonics is None:
        return np.interp(crack.s, s, values, period=2 * np.pi)
    count = len(s)
    if not np.allclose(np.diff(s), 2 * np.pi / count):
        raise ValueError("at_stations: evenly spaced stations around the front")
    series = np.fft.rfft(values) / count
    out = np.full(crack.stations, series[0].real)
    for k in range(1, min(int(harmonics), (count - 1) // 2) + 1):
        out += 2.0 * (series[k] * np.exp(1j * k * (crack.s - s[0]))).real
    return out


def propagate_front(
    mesh_of,
    crack: SurfaceCrack,
    materials,
    prescribed,
    load: float = 1.0,
    *,
    increment: float,
    steps: int,
    stations: int = 12,
    exponent: float = 2.0,
    harmonics: int | None = None,
    paris: tuple[float, float] | None = None,
    kinematics: str = "small",
    contact: float | bool | None = None,
    domain: dict | None = None,
    keep: bool = False,
    **options,
) -> FrontGrowth:
    """Grow a crack of any surface, each point of its front by its own advance and at its
    own angle (see the module): ``mesh_of(crack)`` meshes the body conforming to it
    (:func:`~morphogen.fracture.lefm.crack_mesh`), ``prescribed`` (entries, or a function
    of the opened mesh) and ``load`` load it (``options``: those of
    :func:`~morphogen.fracture.mixed.solve_crack`). At each of the ``steps`` the energy
    release rate and the factors are taken at ``stations`` along the front (at least two
    elements apart), and the point where ``G`` is largest advances by ``increment``, the
    others by ``increment (G / G_max)^(exponent / 2)`` (``exponent``: that of Paris' law;
    with ``paris = (C, m)`` it is ``m``, and the cycles are counted), each in the
    direction of the largest hoop stress of its ``K_I`` and ``K_II``; ``G`` is that of
    the factors, ``(K_I² + K_II²) / E' + K_III² / 2μ``. ``domain``: the radii ``r_in``
    and ``r_out`` of the domain of the interaction integrals (by default one and four
    elements, within the last advance). Around a closed front the advances and the
    angles are those of the Fourier series of their values at the stations up to
    ``harmonics`` (by default a third of the stations): a front that stays smooth, where
    the noise of the factors, raised to the power of Paris' law, would scallop it."""
    from morphogen.fracture.mixed import _isotropic_moduli, kink_angle, solve_crack

    if not increment > 0:
        raise ValueError("propagate_front: a positive increment")
    if paris is not None:
        exponent = float(paris[1])
    if harmonics is None and crack.closed:
        harmonics = max(1, int(stations) // 3)
    cracks, rates, factors, angles, advances, cycles, solutions = [], [], [], [], [], [], []
    n, s = 0.0, None
    for step in range(int(steps) + 1):
        solution = solve_crack(mesh_of(crack), crack, materials, prescribed, load,
                               kinematics=kinematics, contact=contact, **options)  # fmt: skip
        modes = solution.modes(stations=stations, **(domain or {}))
        s = modes["s"]
        k = np.stack([modes["K_I"], modes["K_II"], modes["K_III"]], axis=1)
        # the energy release rate of the factors: along a front that turns sharply it is
        # more even than that of the G-θ method, whose domain the curvature narrows
        young, nu = _isotropic_moduli(solution.materials[solution.front_phase()])
        e_prime, mu = young / (1 - nu**2), young / (2 * (1 + nu))
        g = (k[:, 0] ** 2 + k[:, 1] ** 2) / e_prime + k[:, 2] ** 2 / (2 * mu)
        turn = np.array([kink_angle(a, b) for a, b in k[:, :2]])
        cracks.append(crack)
        rates.append(g)
        factors.append(k)
        angles.append(turn)
        cycles.append(n)
        if keep:
            solutions.append(solution)
        if step == steps:
            advances.append(0.0)
            break
        top = float(g.max())
        if not top > 0:
            raise ValueError("propagate_front: no energy is released along the front")
        share = at_stations(crack, s, (g / top) ** (0.5 * exponent), harmonics)
        share = share / share.max()  # the fastest station advances by the increment
        advance = increment * np.clip(share, 0.02, 1.0)  # (none left behind for good)
        advances.append(float(advance.max()))
        if paris is not None:
            c, m = paris
            n += increment / (c * math.sqrt(e_prime * top) ** m)
        # the angles through their sines and cosines: no jump from π to -π between stations
        sine = at_stations(crack, s, np.sin(turn), harmonics)
        cosine = at_stations(crack, s, np.cos(turn), harmonics)
        crack = crack.grown(advance, np.arctan2(sine, cosine))
    return FrontGrowth(
        cracks,
        np.asarray(s),
        np.array(rates),
        np.array(factors),
        np.array(angles),
        np.array(advances),
        np.array(cycles),
        solutions,
    )
