"""Laser paths: the patterns of tracks that architecture a sheet.

A :class:`LaserPath` is a sequence of :class:`Track` s, polylines (mm) that the beams
follow in order, each with its beams (the upper one, and the lower one of a double-sided
treatment) and the pause that follows it (the cooling before the next track, at least
30 s between neighbouring tracks in Wang's experiments). Coordinates are those of the
sheet: ``x`` along the tensile axis (the rolling direction), ``y`` across.

Generators (:data:`PATTERNS`, :func:`make_pattern`), for a band ``[x0, x1] x [y0, y1]``
(the tracks overhang the band along x by ``margin``, the specimen being cut afterwards,
so that no track starts or stops in it):

``straight``
    ``count`` parallel tracks at ``angle`` (degrees from x), at ``spacing`` (default:
    the band's width over ``count``), centred;
``corrugated``
    ``count`` tracks ``y = y_i + A sin(2π x/P + φ_i)`` (``amplitude`` ``A``, the height
    of Wang et al., ``period`` ``P``), ``phase`` the shift between neighbours (0: in
    phase, 180: mirrored), ``shape`` ``sine`` or ``triangle`` (zigzag);
``fourier``
    the same with the waveform ``Σ a_k sin(2πkx/P) + b_k cos(2πkx/P)`` of
    ``coefficients`` ``[[a1, b1], [a2, b2], ...]``: the free shapes of an optimization;
``crossed``
    two families of straight tracks at ``±angle`` (a rhombic or square grid), the
    second family after the first, at ``spacing`` or at its own ``spacing2`` (a
    rectangular grid: more tracks one way than the other);
``honeycomb``
    the edges of hexagons of side ``cell``, drawn as zigzag rows and the rungs between
    them;
``spiral``
    an Archimedean spiral ``r = radius + pitch θ/2π`` about ``center`` (the centre of
    the band by default), ``turns`` turns from the direction ``angle``, counter-clockwise
    or ``clockwise``: one track (the path of a crack to guide);
``polylines``
    given ``tracks`` (lists of points).

Measures: length, time and energy of the path, the smallest radius of curvature
(:meth:`LaserPath.min_radius`) and the smallest gap between distinct tracks
(:meth:`LaserPath.min_gap`); :meth:`LaserPath.to_gcode` writes it for a machine.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np
from numba import njit, prange

from morphogen.laser.heat import Beam


@dataclass
class Track:
    """A polyline (n x 2, mm) followed by ``beam`` (upper face) and ``bottom`` (lower
    face, or None), then ``pause`` seconds lasers off."""

    points: np.ndarray
    beam: Beam
    bottom: Beam | None = None
    pause: float = 30.0

    def __post_init__(self):
        self.points = np.ascontiguousarray(np.asarray(self.points, dtype=float).reshape(-1, 2))
        if len(self.points) < 2:
            raise ValueError("a track needs at least two points")

    @property
    def length(self) -> float:
        d = np.diff(self.points, axis=0)
        return float(np.hypot(d[:, 0], d[:, 1]).sum())

    @property
    def time(self) -> float:
        """Time with the lasers on (s)."""
        return self.length / self.beam.speed

    @property
    def energy(self) -> float:
        """Incident energy of both beams (J)."""
        power = self.beam.power + (self.bottom.power if self.bottom is not None else 0.0)
        return power * self.time

    def shifted(self, dx: float = 0.0, dy: float = 0.0) -> Track:
        return replace(self, points=self.points + np.array([dx, dy]))

    def reversed(self) -> Track:
        return replace(self, points=self.points[::-1].copy())


@dataclass
class LaserPath:
    """Tracks in the order of the treatment."""

    tracks: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self):
        return iter(self.tracks)

    @property
    def length(self) -> float:
        return sum(t.length for t in self.tracks)

    @property
    def time(self) -> float:
        """Treatment time (s): lasers on, and the pauses between tracks."""
        return sum(t.time for t in self.tracks) + sum(t.pause for t in self.tracks[:-1])

    @property
    def energy(self) -> float:
        return sum(t.energy for t in self.tracks)

    def segments(self):
        """``(points, beam, bottom, pause)`` of each track, for
        :meth:`~morphogen.laser.heat.TransientHeat.run`."""
        return [(t.points, t.beam, t.bottom, t.pause) for t in self.tracks]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.vstack([t.points for t in self.tracks])
        return pts.min(axis=0), pts.max(axis=0)

    def clipped_length(self, lo, hi) -> float:
        """Length of the tracks inside the box ``[lo, hi]`` (the specimen)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        total = 0.0
        for t in self.tracks:
            p = t.points
            for a, b in itertools.pairwise(p):
                total += _clip_segment(a, b, lo, hi)
        return total

    def min_radius(self) -> float:
        """Smallest radius of curvature of the tracks (the circumradius of consecutive
        triplets of points; inf for straight tracks)."""
        best = math.inf
        for t in self.tracks:
            p = t.points
            if len(p) < 3:
                continue
            a, b, c = p[:-2], p[1:-1], p[2:]
            ab = np.hypot(*(b - a).T)
            bc = np.hypot(*(c - b).T)
            ca = np.hypot(*(a - c).T)
            cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
            with np.errstate(divide="ignore", invalid="ignore"):
                r = ab * bc * ca / (2.0 * cross)
            r = r[np.isfinite(r)]
            if r.size:
                best = min(best, float(r.min()))
        return best

    def min_gap(self, lo=None, hi=None) -> float:
        """Smallest distance between two distinct tracks (within the box ``[lo, hi]``
        if given): the tracks closer than their width overlap and temper each other."""
        best = math.inf
        pts = [t.points for t in self.tracks]
        for i in range(len(pts)):
            sample = _densify(pts[i], 0.1)
            if lo is not None:
                inside = np.all((sample >= np.asarray(lo)) & (sample <= np.asarray(hi)), axis=1)
                sample = sample[inside]
            if not len(sample):
                continue
            for j in range(len(pts)):
                if j == i:
                    continue
                d = distance_to_polyline(sample, pts[j])
                best = min(best, float(d.min()))
        return best

    def to_gcode(self, *, laser_on: str = "M3", laser_off: str = "M5") -> str:
        """The path as G-code (mm, mm/min): rapid moves to the starts of the tracks, the
        lasers on (``S``: the power of the upper beam, W; a comment gives the lower one),
        linear moves at the speed of the track, the lasers off, a dwell for the pause."""
        lines = ["; morphoGen laser path", "G21 ; mm", "G90 ; absolute"]
        for k, t in enumerate(self.tracks):
            p = t.points
            lines.append(
                f"; track {k}: {t.length:.3f} mm"
                + (f", lower beam {t.bottom.power:g} W" if t.bottom is not None else "")
            )
            lines.append(f"G0 X{p[0, 0]:.4f} Y{p[0, 1]:.4f}")
            lines.append(f"{laser_on} S{t.beam.power:g}")
            feed = t.beam.speed * 60.0
            for x, y in p[1:]:
                lines.append(f"G1 X{x:.4f} Y{y:.4f} F{feed:.1f}")
            lines.append(laser_off)
            if t.pause > 0 and k < len(self.tracks) - 1:
                lines.append(f"G4 P{t.pause:g}")
        lines.append("M2")
        return "\n".join(lines) + "\n"


def _clip_segment(a, b, lo, hi) -> float:
    """Length of the segment ``ab`` inside the box (Liang–Barsky)."""
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-d[0], a[0] - lo[0]),
        (d[0], hi[0] - a[0]),
        (-d[1], a[1] - lo[1]),
        (d[1], hi[1] - a[1]),
    ):
        if p == 0:
            if q < 0:
                return 0.0
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return 0.0
    return float((t1 - t0) * math.hypot(d[0], d[1]))


def _densify(points, step: float) -> np.ndarray:
    out = [points[:1]]
    for a, b in itertools.pairwise(points):
        n = max(1, math.ceil(math.hypot(*(b - a)) / step))
        t = (np.arange(1, n + 1) / n)[:, None]
        out.append(a + t * (b - a))
    return np.vstack(out)


@njit(parallel=True, cache=True)
def _distances(points, poly, out):  # pragma: no cover - compiled
    n, m = points.shape[0], poly.shape[0]
    for i in prange(n):
        px, py = points[i, 0], points[i, 1]
        best = 1e300
        for k in range(m - 1):
            ax, ay = poly[k, 0], poly[k, 1]
            bx, by = poly[k + 1, 0], poly[k + 1, 1]
            dx, dy = bx - ax, by - ay
            ll = dx * dx + dy * dy
            if ll > 0:
                t = ((px - ax) * dx + (py - ay) * dy) / ll
                t = min(1.0, max(0.0, t))
            else:
                t = 0.0
            ex, ey = ax + t * dx - px, ay + t * dy - py
            d2 = ex * ex + ey * ey
            if d2 < best:
                best = d2
        out[i] = math.sqrt(best)


def distance_to_polyline(points, polyline) -> np.ndarray:
    """Distance (mm) of points (n x 2) to a polyline (m x 2)."""
    p = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 2))
    poly = np.ascontiguousarray(np.asarray(polyline, dtype=float).reshape(-1, 2))
    out = np.empty(len(p))
    if len(poly) == 1:
        return np.hypot(p[:, 0] - poly[0, 0], p[:, 1] - poly[0, 1])
    _distances(p, poly, out)
    return out


# -- generators ----------------------------------------------------------------------------


def _positions(count: int, y0: float, y1: float, spacing: float | None, offset: float):
    if count < 1:
        raise ValueError("count: at least one track")
    centre = 0.5 * (y0 + y1) + offset
    s = (y1 - y0) / count if spacing is None else float(spacing)
    return centre + (np.arange(count) - 0.5 * (count - 1)) * s


def _waveform(x, period: float, amplitude: float, phase: float, shape: str, coefficients):
    u = 2.0 * math.pi * x / period + phase
    if coefficients is not None:
        out = np.zeros_like(x)
        for k, (a, b) in enumerate(np.asarray(coefficients, dtype=float).reshape(-1, 2), start=1):
            out += a * np.sin(k * u) + b * np.cos(k * u)
        return out
    if shape == "sine":
        return amplitude * np.sin(u)
    if shape == "triangle":
        return amplitude * (2.0 / math.pi) * np.arcsin(np.sin(u))
    raise ValueError("shape: sine or triangle")


def _wavy(
    x0,
    x1,
    y0,
    y1,
    *,
    count,
    period,
    amplitude=0.0,
    phase=0.0,
    shape="sine",
    coefficients=None,
    spacing=None,
    offset=0.0,
    margin=5.0,
    resolution=24,
    alternate=True,
    beam,
    bottom,
    pause,
):
    if not period > 0:
        raise ValueError("period must be positive")
    xs = np.linspace(
        x0 - margin,
        x1 + margin,
        max(2, math.ceil((x1 - x0 + 2 * margin) / period * resolution) + 1),
    )
    if shape == "triangle" and coefficients is None:  # exact corners
        u0 = (x0 - margin) / period
        corners = (
            np.arange(
                math.floor(4 * u0 - 4 * phase / (2 * math.pi)) - 1,
                math.ceil(4 * (x1 + margin) / period) + 2,
            )
            + 1
        )
        xc = (corners / 4.0 - phase / (2 * math.pi)) * period
        xs = np.unique(np.concatenate([xs, xc[(xc > xs[0]) & (xc < xs[-1])]]))
    tracks = []
    for i, yc in enumerate(_positions(count, y0, y1, spacing, offset)):
        y = yc + _waveform(xs, period, amplitude, math.radians(phase) * i, shape, coefficients)
        t = Track(np.column_stack([xs, y]), beam, bottom, pause)
        tracks.append(t.reversed() if alternate and i % 2 else t)
    return LaserPath(tracks)


def _lines(
    x0,
    x1,
    y0,
    y1,
    *,
    angle,
    spacing,
    count=None,
    offset=0.0,
    margin=5.0,
    alternate=True,
    beam,
    bottom,
    pause,
):
    """Parallel straight tracks at ``angle`` covering the band (plus margin)."""
    th = math.radians(angle)
    u = np.array([math.cos(th), math.sin(th)])  # along the tracks
    nrm = np.array([-u[1], u[0]])
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    corners = np.array([[x0 - margin, y0], [x1 + margin, y0], [x0 - margin, y1], [x1 + margin, y1]])
    proj_n = (corners - [cx, cy]) @ nrm
    proj_u = (corners - [cx, cy]) @ u
    if count is not None and spacing is None:
        width = proj_n.max() - proj_n.min() if abs(math.sin(th)) > 1e-12 else (y1 - y0)
        spacing = width / count
    if spacing is None or not spacing > 0:
        raise ValueError("spacing (or count) of the tracks")
    if count is None:
        k0 = math.ceil((proj_n.min() - offset) / spacing - 0.5)
        k1 = math.floor((proj_n.max() - offset) / spacing - 0.5)
        offsets = offset + (np.arange(k0, k1 + 1) + 0.5) * spacing
    else:
        offsets = offset + (np.arange(count) - 0.5 * (count - 1)) * spacing
    half = 0.5 * (proj_u.max() - proj_u.min()) + margin
    lo = np.array([x0 - margin, y0 - margin])
    hi = np.array([x1 + margin, y1 + margin])
    tracks = []
    for o in offsets:
        c = np.array([cx, cy]) + o * nrm
        a, b = c - half * u, c + half * u
        seg = _clip_points(a, b, lo, hi)
        if seg is not None:
            tracks.append(Track(np.array(seg), beam, bottom, pause))
    if alternate:
        tracks = [t.reversed() if i % 2 else t for i, t in enumerate(tracks)]
    return LaserPath(tracks)


def _clip_points(a, b, lo, hi):
    """The part of the segment ``ab`` inside the box, or None (Liang–Barsky)."""
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-d[0], a[0] - lo[0]),
        (d[0], hi[0] - a[0]),
        (-d[1], a[1] - lo[1]),
        (d[1], hi[1] - a[1]),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 >= t1:
            return None
    return a + t0 * d, a + t1 * d


def straight(
    x0,
    x1,
    y0,
    y1,
    *,
    count=1,
    angle=0.0,
    spacing=None,
    offset=0.0,
    margin=5.0,
    alternate=True,
    beam,
    bottom=None,
    pause=30.0,
) -> LaserPath:
    """Parallel straight tracks (see the module)."""
    if abs(math.sin(math.radians(angle))) < 1e-12:
        if count is None:  # a family at its spacing over the band
            if not spacing:
                raise ValueError("straight: a count or a spacing")
            count = max(1, math.floor((y1 - y0) / float(spacing) + 1e-9))
        return _wavy(
            x0,
            x1,
            y0,
            y1,
            count=count,
            period=x1 - x0 + 2 * margin,
            spacing=spacing,
            offset=offset,
            margin=margin,
            resolution=1,
            alternate=alternate,
            beam=beam,
            bottom=bottom,
            pause=pause,
        )
    return _lines(
        x0,
        x1,
        y0,
        y1,
        angle=angle,
        spacing=spacing,
        count=count,
        offset=offset,
        margin=margin,
        alternate=alternate,
        beam=beam,
        bottom=bottom,
        pause=pause,
    )


def corrugated(
    x0,
    x1,
    y0,
    y1,
    *,
    count=1,
    amplitude=1.0,
    period=10.0,
    phase=0.0,
    shape="sine",
    spacing=None,
    offset=0.0,
    margin=5.0,
    resolution=24,
    alternate=True,
    beam,
    bottom=None,
    pause=30.0,
) -> LaserPath:
    """Corrugated tracks (see the module)."""
    return _wavy(
        x0,
        x1,
        y0,
        y1,
        count=count,
        period=period,
        amplitude=amplitude,
        phase=phase,
        shape=shape,
        spacing=spacing,
        offset=offset,
        margin=margin,
        resolution=resolution,
        alternate=alternate,
        beam=beam,
        bottom=bottom,
        pause=pause,
    )


def fourier(
    x0,
    x1,
    y0,
    y1,
    *,
    count=1,
    period=10.0,
    coefficients=((1.0, 0.0),),
    phase=0.0,
    spacing=None,
    offset=0.0,
    margin=5.0,
    resolution=32,
    alternate=True,
    beam,
    bottom=None,
    pause=30.0,
) -> LaserPath:
    """Tracks of a Fourier waveform (see the module)."""
    return _wavy(
        x0,
        x1,
        y0,
        y1,
        count=count,
        period=period,
        phase=phase,
        coefficients=coefficients,
        spacing=spacing,
        offset=offset,
        margin=margin,
        resolution=resolution,
        alternate=alternate,
        beam=beam,
        bottom=bottom,
        pause=pause,
    )


def crossed(
    x0,
    x1,
    y0,
    y1,
    *,
    angle=45.0,
    spacing=3.0,
    spacing2=None,
    offset=0.0,
    margin=5.0,
    alternate=True,
    beam,
    bottom=None,
    pause=30.0,
) -> LaserPath:
    """Two families of straight tracks at ``±angle``, the second at ``spacing2`` if
    given (see the module)."""
    a = _lines(
        x0,
        x1,
        y0,
        y1,
        angle=angle,
        spacing=spacing,
        offset=offset,
        margin=margin,
        alternate=alternate,
        beam=beam,
        bottom=bottom,
        pause=pause,
    )
    b = _lines(
        x0,
        x1,
        y0,
        y1,
        angle=-angle,
        spacing=spacing if spacing2 is None else spacing2,
        offset=offset,
        margin=margin,
        alternate=alternate,
        beam=beam,
        bottom=bottom,
        pause=pause,
    )
    return LaserPath(a.tracks + b.tracks)


def chain_edges(vertices, edges) -> list[np.ndarray]:
    """Polylines that cover the edges of a network once each: from an unused edge, a
    walk continues along the unused edge that turns the least, both ways, until none
    is left at its ends (few laser starts and stops)."""
    v = np.asarray(vertices, dtype=float).reshape(-1, 2)
    e = [tuple(sorted(map(int, ed))) for ed in edges]
    e = sorted(set(ed for ed in e if ed[0] != ed[1]))
    adj: dict[int, list[int]] = {}
    for k, (a, b) in enumerate(e):
        adj.setdefault(a, []).append(k)
        adj.setdefault(b, []).append(k)
    used = np.zeros(len(e), dtype=bool)

    def other(k, a):
        return e[k][1] if e[k][0] == a else e[k][0]

    def walk(prev, cur):
        out = []
        while True:
            d0 = v[cur] - v[prev]
            best, best_cos = None, -2.0
            for k in adj[cur]:
                if used[k]:
                    continue
                d1 = v[other(k, cur)] - v[cur]
                c = float(d0 @ d1) / max(np.linalg.norm(d0) * np.linalg.norm(d1), 1e-300)
                if c > best_cos:
                    best, best_cos = k, c
            if best is None:
                return out
            used[best] = True
            prev, cur = cur, other(best, cur)
            out.append(cur)

    lines = []
    for k in range(len(e)):
        if used[k]:
            continue
        used[k] = True
        a, b = e[k]
        forward = walk(a, b)
        backward = walk(b, a)
        idx = [*backward[::-1], a, b, *forward]
        lines.append(v[idx])
    return lines


def honeycomb(
    x0,
    x1,
    y0,
    y1,
    *,
    cell=3.0,
    angle=0.0,
    offset=0.0,
    margin=5.0,
    alternate=True,
    beam,
    bottom=None,
    pause=30.0,
) -> LaserPath:
    """The edges of regular hexagons of side ``cell``, two of their sides along x (turned
    by ``angle`` degrees), chained into continuous tracks (:func:`chain_edges`)."""
    a = float(cell)
    if not a > 0:
        raise ValueError("cell must be positive")
    h = math.sqrt(3.0) * a
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1) + offset
    th = math.radians(angle)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    reach = math.hypot(x1 - x0 + 2 * margin, y1 - y0 + 2 * margin) / 2 + 2 * a
    ni, nj = math.ceil(reach / (1.5 * a)) + 1, math.ceil(reach / h) + 1
    corners = a * np.array(
        [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
    )
    keys: dict[tuple, int] = {}
    verts, edges = [], []
    for i in range(-ni, ni + 1):
        for j in range(-nj, nj + 1):
            c = np.array([1.5 * a * i, h * (j + 0.5 * (i % 2))])
            ids = []
            for q in corners + c:
                key = (round(q[0] / a * 1e6), round(q[1] / a * 1e6))
                if key not in keys:
                    keys[key] = len(verts)
                    verts.append(q)
                ids.append(keys[key])
            edges.extend((ids[k], ids[(k + 1) % 6]) for k in range(6))
    verts = np.array(verts) @ rot.T + [cx, cy]
    lo = np.array([x0 - margin, y0 - margin])
    hi = np.array([x1 + margin, y1 + margin])
    inside = np.all((verts >= lo - a) & (verts <= hi + a), axis=1)
    edges = [ed for ed in edges if inside[ed[0]] and inside[ed[1]]]
    lines = chain_edges(verts, edges)
    tracks = []
    for i, pts in enumerate(lines):
        t = Track(pts, beam, bottom, pause)
        tracks.append(t.reversed() if alternate and i % 2 else t)
    return LaserPath(tracks)


def spiral(
    x0,
    x1,
    y0,
    y1,
    *,
    center=None,
    radius=0.0,
    pitch=3.0,
    turns=2.0,
    angle=0.0,
    clockwise=False,
    resolution=0.2,
    margin=5.0,
    beam,
    bottom=None,
    pause=30.0,
) -> LaserPath:
    """An Archimedean spiral track (see the module), its points about ``resolution``
    mm apart along it."""
    if not pitch > 0 or not turns > 0:
        raise ValueError("spiral: positive pitch and turns")
    cx, cy = (0.5 * (x0 + x1), 0.5 * (y0 + y1)) if center is None else map(float, center)
    th_end = 2.0 * math.pi * float(turns)
    ths = [0.0]
    while ths[-1] < th_end:
        r = float(radius) + float(pitch) * ths[-1] / (2.0 * math.pi)
        ths.append(min(th_end, ths[-1] + float(resolution) / max(r, float(resolution))))
    th = np.array(ths)
    r = float(radius) + float(pitch) * th / (2.0 * math.pi)
    phi = math.radians(angle) + (-th if clockwise else th)
    pts = np.column_stack([cx + r * np.cos(phi), cy + r * np.sin(phi)])
    return LaserPath([Track(pts, beam, bottom, pause)])


def polylines(tracks, *, beam, bottom=None, pause=30.0) -> LaserPath:
    """Tracks given as lists of points (see the module)."""
    return LaserPath([Track(np.asarray(t, dtype=float), beam, bottom, pause) for t in tracks])


#: Pattern generators of :func:`make_pattern`.
PATTERNS = {
    "straight": straight,
    "corrugated": corrugated,
    "fourier": fourier,
    "crossed": crossed,
    "honeycomb": honeycomb,
    "spiral": spiral,
}


def clip_path(path: LaserPath, lo, hi) -> LaserPath:
    """The parts of the tracks inside the box ``[lo, hi]`` (a track that leaves and
    comes back splits into several)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    out = []
    for t in path.tracks:
        piece: list = []
        for a, b in itertools.pairwise(t.points):
            seg = _clip_points(a, b, lo, hi)
            if seg is None:
                if len(piece) > 1:
                    out.append(replace(t, points=np.array(piece)))
                piece = []
                continue
            p, q = seg
            if piece and np.allclose(piece[-1], p):
                piece.append(q)
            else:
                if len(piece) > 1:
                    out.append(replace(t, points=np.array(piece)))
                piece = [p, q]
            if not np.allclose(q, b):  # it leaves the box here
                out.append(replace(t, points=np.array(piece)))
                piece = []
        if len(piece) > 1:
            out.append(replace(t, points=np.array(piece)))
    return LaserPath(out)


def make_pattern(
    kind: str,
    band,
    *,
    beam,
    bottom=None,
    pause=30.0,
    rotation: float = 0.0,
    cover: bool = False,
    **params,
) -> LaserPath:
    """A pattern of :data:`PATTERNS` (or ``polylines``) over the band ``(x0, x1, y0, y1)``
    (mm), its tracks followed by ``beam`` and ``bottom``. ``rotation`` (degrees) turns
    the pattern about the centre of the band: the specimen is then cut at ``-rotation``
    from the axes of the pattern (the anisotropy of an architectured sheet, tested at
    0°, 45° and 90°); the pattern is drawn on a square that covers the turned band, its
    families of tracks repeated at their spacing over it (the treated sheet), then
    clipped to the band and its margin — so is it unturned with ``cover`` (the periodic
    sheet around a cell, :mod:`~morphogen.laser.periodic`)."""
    if kind == "polylines":
        path = polylines(params.pop("tracks"), beam=beam, bottom=bottom, pause=pause)
        if rotation:
            raise ValueError("rotation: give the polylines turned")
        return path
    if kind not in PATTERNS:
        raise ValueError(f"pattern: one of {[*sorted(PATTERNS), 'polylines']}, got {kind!r}")
    x0, x1, y0, y1 = (float(v) for v in band)
    rotation = float(rotation) % 360.0
    if rotation == 0.0 and not cover:
        return PATTERNS[kind](x0, x1, y0, y1, beam=beam, bottom=bottom, pause=pause, **params)
    margin = float(params.get("margin", 5.0))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * math.hypot(x1 - x0, y1 - y0) + margin
    if kind in ("straight", "corrugated", "fourier") and params.get("count", 1) is not None:
        # the tracks of the sheet, at their spacing, over the whole square: a parallel
        # family (same parity of the count, so that the tracks keep their positions)
        count = int(params.get("count", 1))
        spacing = params.get("spacing")
        spacing = (y1 - y0) / count if spacing is None else float(spacing)
        params["spacing"] = spacing
        params["count"] = count + 2 * max(0, math.ceil((2.0 * half / spacing - count) / 2.0))
    path = PATTERNS[kind](
        cx - half, cx + half, cy - half, cy + half, beam=beam, bottom=bottom, pause=pause, **params
    )
    th = math.radians(rotation)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    turned = LaserPath(
        [replace(t, points=(t.points - [cx, cy]) @ rot.T + [cx, cy]) for t in path.tracks]
    )
    return clip_path(turned, (x0 - margin, y0 - margin), (x1 + margin, y1 + margin))
