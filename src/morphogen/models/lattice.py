"""Strut (beam) lattices: periodic arrangements of cylindrical struts."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

import numpy as np
from scipy import optimize

from morphogen._utils import check_fraction, check_positive
from morphogen.distributions import Distribution
from morphogen.domain import Domain, Grid
from morphogen.kernels.union import UnionAccelerator
from morphogen.models import _sizes as sizes
from morphogen.models.affine import ANISOTROPY, Anisotropy, anisotropic
from morphogen.models.base import Model, param, register
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments, with_periodic_images
from morphogen.seeding import SeedLike, make_rng

_CORNERS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
_FACE_CENTERS = np.array(
    [
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 1.0],
        [0.5, 0.0, 0.5],
        [0.5, 1.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0, 0.5, 0.5],
    ]
)
_BODY_CENTER = np.array([[0.5, 0.5, 0.5]])


def _pairs_at(nodes_a, nodes_b, distance, same: bool):
    """Pairs of nodes (one from each set) at the given distance, in the unit cell."""
    out = []
    for i, p in enumerate(nodes_a):
        for j, q in enumerate(nodes_b):
            if same and j <= i:
                continue
            if abs(np.linalg.norm(p - q) - distance) < 1e-9:
                out.append((p, q))
    return out


def unit_cell_struts(cell: str) -> np.ndarray:
    """Struts of a unit cell in fractional coordinates, shape (n, 2, 3)."""
    h = math.sqrt(0.5)
    if cell == "cubic":
        pairs = _pairs_at(_CORNERS, _CORNERS, 1.0, True)
    elif cell == "bcc":
        pairs = _pairs_at(_BODY_CENTER, _CORNERS, math.sqrt(3.0) / 2.0, False)
    elif cell == "fcc":
        pairs = _pairs_at(_CORNERS, _FACE_CENTERS, h, False)
    elif cell == "octet":
        pairs = _pairs_at(_CORNERS, _FACE_CENTERS, h, False)
        pairs += _pairs_at(_FACE_CENTERS, _FACE_CENTERS, h, True)
    elif cell == "kelvin":
        # edges of the truncated octahedron around the body centre (Voronoi cell of the
        # bcc lattice): its vertices are the permutations of (0, +-1/4, +-1/2), its
        # square faces lie on the faces of the cell
        signs = itertools.product((-1.0, 1.0), repeat=2)
        base = {p for a, b in signs for p in itertools.permutations((0.0, 0.25 * a, 0.5 * b))}
        vertices = 0.5 + np.array(sorted(base))
        pairs = _pairs_at(vertices, vertices, 0.25 * math.sqrt(2.0), True)
    elif cell == "diamond":
        # two face-centred lattices a quarter of the diagonal apart: each node of one
        # joined to its four neighbours of the other, at the corners of a tetrahedron
        # (coordination 4, bending-dominated: the honeycomb of space)
        sites = np.vstack([_CORNERS, _FACE_CENTERS])
        inner = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75],
                          [0.25, 0.75, 0.75]])  # fmt: skip
        pairs = _pairs_at(inner, sites, 0.25 * math.sqrt(3.0), False)
    elif cell == "pyrochlore":
        # tetrahedra joined by their corners (the middles of the bonds of the diamond
        # lattice): six struts per node along three straight lines, coordination 6,
        # stretching-dominated with half the struts of the octet truss: the kagome of
        # space. The origin is at an eighth of the diagonal: every tetrahedron in the cell
        pairs = []
        centres = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
        even = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / 8.0
        for centre in centres:
            for origin, corners in ((0.125, even), (0.375, -even)):
                vertices = centre + origin + corners
                pairs += _pairs_at(vertices, vertices, 0.25 * math.sqrt(2.0), True)
    else:
        raise ValueError(f"cell: expected one of {sorted(CELLS)}, got {cell!r}")
    return np.array([[p, q] for p, q in pairs], dtype=float)


CELLS = ("cubic", "bcc", "fcc", "octet", "kelvin", "diamond", "pyrochlore")


def _unique_segments(start: np.ndarray, end: np.ndarray, scale: float):
    """Remove duplicated struts (shared by neighbouring cells)."""
    a = np.round(start / scale, 9)
    b = np.round(end / scale, 9)
    # orient each strut so that its first endpoint is the lexicographically smaller one
    diff = b - a
    first = np.argmax(diff != 0, axis=1)
    swap = diff[np.arange(len(diff)), first] < 0
    lo = np.where(swap[:, None], b, a)
    hi = np.where(swap[:, None], a, b)
    _, idx = np.unique(np.hstack([lo, hi]), axis=0, return_index=True)
    idx.sort()
    return start[idx], end[idx]


def periodic_struts(start, end, domain: Domain) -> tuple[np.ndarray, np.ndarray]:
    """The struts of a periodic box, each once: those that are periodic images of one
    another (the struts of opposite faces of the box) kept once, the middle of each moved
    into the box."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    period = np.asarray(domain.lengths, dtype=float)
    tol = 1e-9 * float(period.max())
    middle = 0.5 * (start + end)
    wrapped = np.mod(middle - domain.lower, period)
    wrapped = np.where(np.abs(wrapped - period) <= tol, 0.0, wrapped)
    shift = domain.lower + wrapped - middle
    a, b = start + shift, end + shift
    d = b - a
    first = np.argmax(np.abs(d) > tol, axis=1)
    flip = d[np.arange(len(d)), first] < 0  # each strut with its direction one way
    d = np.where(flip[:, None], -d, d)
    keys = np.round(np.hstack([wrapped, d]) / tol)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep.sort()
    return a[keep], b[keep]


def lattice_struts(cell: str, cell_size: float, lower, upper) -> tuple[np.ndarray, np.ndarray]:
    """All struts of the lattice anchored at the origin whose cells meet [lower, upper]."""
    unit = unit_cell_struts(cell) * cell_size
    lo_idx = np.floor(np.asarray(lower) / cell_size).astype(int)
    hi_idx = np.ceil(np.asarray(upper) / cell_size).astype(int) - 1
    ranges = [range(lo_idx[a], max(hi_idx[a], lo_idx[a]) + 1) for a in range(3)]
    shifts = np.array(list(itertools.product(*ranges)), dtype=float) * cell_size
    start = (shifts[:, None, :] + unit[None, :, 0, :]).reshape(-1, 3)
    end = (shifts[:, None, :] + unit[None, :, 1, :]).reshape(-1, 3)
    return _unique_segments(start, end, cell_size)


#: Offset of the lattice from the voxels of the fraction estimates, in voxels: off the
#: symmetries of the cells, so that the voxels sample the struts as if at random.
_JITTER = np.array([math.sqrt(2.0) - 1.0, math.sqrt(3.0) - 1.0, math.sqrt(5.0) - 2.0]) / 2.0


def lattice_fraction(cell: str, radius: float, n: int | None = None) -> float:
    """Solid volume fraction of a lattice of unit cell size 1 (voxel estimate)."""
    if n is None:
        n = int(np.clip(math.ceil(10.0 / radius), 48, 256))
    unit = Domain.cube(1.0, periodic=True)
    start, end = lattice_struts(cell, 1.0, unit.lower, unit.upper)
    shift = _JITTER / n  # voxels aligned with the struts would count them in batches
    seg = Segments(start - shift, end - shift, np.full(len(start), radius), round_caps=True)
    seg = with_periodic_images(seg, unit)
    acc = UnionAccelerator(None, seg, unit.lower, unit.upper, min_cell=2.0 / n)
    counts = acc.count(unit.lower, np.full(3, 1.0 / n), (n, n, n), 1)
    return float(counts.mean())


@lru_cache(maxsize=512)
def strut_fraction(cell: str, r: float, t: float = 0.0, n: int | None = None) -> float:
    """Solid fraction of a lattice of unit cell size 1, struts of radius ``r`` hollow
    with walls ``t`` (0: solid): the union of the struts minus that of their bores.
    Exact for the cubic cell, a voxel estimate on ``n^3`` voxels otherwise (default: 8
    voxels across the walls of hollow struts, 10 across the radius of solid ones)."""
    hollow = 0 < t < r
    if cell == "cubic":  # three orthogonal cylinders per node, minus their overlaps

        def union(x):
            return 3.0 * math.pi * x * x - 8.0 * math.sqrt(2.0) * x**3

    else:
        if n is None:
            n = int(np.clip(math.ceil(8.0 / t if hollow else 10.0 / r), 48, 256))

        def union(x):
            return lattice_fraction(cell, x, n)

    return union(r) - (union(r - t) if hollow else 0.0)


@lru_cache(maxsize=128)
def strut_radius(cell: str, target: float, t: float = 0.0) -> float:
    """Radius of the struts (unit cell size) of a lattice of solid fraction ``target``,
    hollow with walls ``t``. The fraction of solid struts grows with their radius; that
    of hollow struts peaks once their bores start to fill the cell: the radius is taken
    below the peak."""
    lo, hi = max(1e-4, 1.0001 * t), 0.499
    if t > 0:
        radii = np.linspace(lo, hi, 25)
        k = int(np.argmax([strut_fraction(cell, r, t, 64) for r in radii]))
        hi = float(radii[max(k, 1)])
    top = strut_fraction(cell, hi, t)
    if top <= target:
        reach = f"hollow struts with walls of {t:.3g} cell" if t > 0 else "this lattice"
        raise ValueError(f"volume_fraction: at most {top:.3g} for {reach}")
    return float(optimize.brentq(lambda r: strut_fraction(cell, r, t) - target, lo, hi, xtol=1e-5))


@register
@anisotropic
@dataclass(frozen=True)
class StrutLattice(Model):
    """Lattice of cylindrical struts with spherical nodes (spherocylinders).

    Cells: ``cubic`` (cube edges), ``bcc`` (body diagonals), ``fcc`` (face diagonals),
    ``octet`` (face diagonals and inner octahedron), ``kelvin`` (edges of the Kelvin
    foam: truncated octahedra on a bcc lattice, 24 struts per cell). Give the strut
    radius or the target relative density (solid volume fraction). With
    ``wall_thickness``, the struts are hollow tubes: their union minus the union of their
    bores (which join at the nodes).

    In non-periodic domains, the lattice may be turned (``rotation``: Bunge angles of its
    axes in those of the sample; it is then anchored at the centre of the domain) and
    graded: with ``radius_gradient`` (the radii at the lower and upper faces of the
    domain along ``gradient_axis``), each strut takes the radius interpolated linearly at
    its midpoint, a functionally graded lattice.

    A distribution of the strut radius draws a radius for each strut (the scatter of the
    struts of printed lattices; in a periodic box, each strut once, its periodic images
    alike); with a ``volume_fraction``, its law keeps its shape and its scale is solved on
    each realization (voxels a quarter of the mean radius wide). The theory then gives
    the mean radius and the fraction to first order, ``pi E[R^2] L_V`` (``L_V`` the length
    of the struts per unit volume; the nodes, where they overlap, left out).
    """

    name = "strut_lattice"
    phases: ClassVar[dict[int, str]] = {0: "void", 1: "struts"}
    count_label = "unit cells"

    cell: str = param(
        "octet",
        doc="Unit cell: cubic, bcc, fcc, octet, kelvin, diamond or pyrochlore",
        kind="str",
    )
    cell_size: float = param(1.0, doc="Edge of the cubic unit cell", unit="length")
    strut_radius: Distribution | None = param(
        None,
        doc="Strut radius: a number (monodisperse) or a distribution (polydisperse, one "
        "radius per strut; with volume_fraction, the shape of the law whose scale is solved)",
        kind="distribution",
        unit="length",
    )
    volume_fraction: float | None = param(None, doc="Target relative density", kind="fraction")
    wall_thickness: float | None = param(
        None, doc="Wall thickness of hollow struts (default: solid)", unit="length"
    )
    rotation: tuple[float, ...] | None = param(
        None,
        doc="Bunge angles (degrees) of the axes of the lattice in those of the sample "
        "(non-periodic domains)",
        kind="floats",
    )
    radius_gradient: tuple[float, ...] | None = param(
        None,
        doc="Strut radii at the lower and upper faces of the domain along gradient_axis: a "
        "graded lattice (non-periodic domains)",
        kind="floats",
        unit="length",
    )
    gradient_axis: int = param(
        0, doc="Axis of the gradient of the strut radius: 0, 1 or 2", kind="int"
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        if self.cell not in CELLS:
            raise ValueError(f"cell: expected one of {list(CELLS)}, got {self.cell!r}")
        check_positive(self.cell_size, "cell_size")
        if self.rotation is not None and len(self.rotation) != 3:
            raise ValueError("rotation: three Bunge angles (degrees)")
        if self.gradient_axis not in (0, 1, 2):
            raise ValueError("gradient_axis: 0, 1 or 2")
        if self.radius_gradient is not None:
            if len(self.radius_gradient) != 2:
                raise ValueError("radius_gradient: the radii at the lower and upper faces")
            if self.strut_radius is not None or self.volume_fraction is not None:
                raise ValueError("give the radius_gradient, or the strut_radius or volume_fraction")
            if self.wall_thickness is not None:
                raise ValueError("radius_gradient: solid struts only (no wall_thickness)")
            for r in self.radius_gradient:
                check_positive(r, "radius_gradient")
                if r >= 0.5 * self.cell_size:
                    raise ValueError("radius_gradient: radii smaller than half the cell size")
            return
        sizes.check_scale(self.strut_radius, self.volume_fraction, "strut_radius")
        law = self.strut_radius
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        elif law.upper() >= 0.5 * self.cell_size:
            raise ValueError("strut_radius: must be smaller than half the cell size")
        if self.wall_thickness is not None:
            check_positive(self.wall_thickness, "wall_thickness")
            if law is not None and self.volume_fraction is not None:
                raise ValueError(
                    "wall_thickness: hollow struts of several radii take their strut_radius, "
                    "not a volume_fraction"
                )
            if law is not None and self.wall_thickness >= law.moment(1):
                raise ValueError("wall_thickness: smaller than the strut radius")

    @property
    def radius(self) -> float:
        """Strut radius (solved numerically from the volume fraction if needed; the mean
        of the gradient of a graded lattice; the mean of the law of polydisperse struts,
        that of uniform struts of the volume fraction when the fraction sets its scale)."""
        if self.radius_gradient is not None:
            return 0.5 * (self.radius_gradient[0] + self.radius_gradient[1])
        if self.strut_radius is not None and self.volume_fraction is None:
            return self.strut_radius.moment(1)
        t = 0.0 if self.wall_thickness is None else self.wall_thickness / self.cell_size
        return strut_radius(self.cell, float(self.volume_fraction), t) * self.cell_size

    @property
    def length_density(self) -> float:
        """Length of the struts per unit volume, each strut of the periodic lattice once."""
        unit = Domain.cube(1.0, periodic=True)
        start, end = lattice_struts(self.cell, 1.0, unit.lower, unit.upper)
        start, end = periodic_struts(start, end, unit)
        return float(np.linalg.norm(end - start, axis=1).sum()) / self.cell_size**2

    def _struts(self, domain: Domain) -> tuple[np.ndarray, np.ndarray]:
        """Ends of the struts that may meet the domain (turned by ``rotation``)."""
        a = self.cell_size
        margin = 0.0 if domain.periodic else a
        if self.rotation is None:
            return lattice_struts(self.cell, a, domain.lower - margin, domain.upper + margin)
        from morphogen.orientations import euler_to_matrix

        g = euler_to_matrix(np.asarray(self.rotation, dtype=float))[0]  # lattice = g sample
        box = domain.dilated(margin)
        corners = np.array([[box.lower[i] if c[i] == 0 else box.upper[i] for i in range(3)]
                            for c in itertools.product((0, 1), repeat=3)])  # fmt: skip
        local = (corners - domain.center) @ g.T
        start, end = lattice_struts(self.cell, a, local.min(axis=0), local.max(axis=0))
        return domain.center + start @ g, domain.center + end @ g

    def _radii(self, domain: Domain, start, end) -> np.ndarray:
        if self.radius_gradient is None:
            return np.full(len(start), self.radius)
        axis = self.gradient_axis
        s = (0.5 * (start + end)[:, axis] - domain.lower[axis]) / domain.lengths[axis]
        r0, r1 = self.radius_gradient
        return r0 + (r1 - r0) * np.clip(s, 0.0, 1.0)

    def _scaled(self, domain: Domain, start, end, unit) -> float:
        """The scale ``s`` of the radii ``s unit`` (one per strut) of a realization at its
        volume fraction, from that of uniform struts of the same fraction."""
        guess = self.radius / float(np.mean(unit))
        h = self.radius / 4.0
        shape = np.clip(np.ceil(domain.lengths / h), 16, 256).astype(int)
        grid = Grid(domain, tuple(int(n) for n in shape))

        def fraction(s):
            union = PrimitiveUnion(domain, segments=Segments(start, end, s * unit, True))
            return float(union.voxelize(grid=grid).data.mean())

        limit = 0.5 * self.cell_size / float(np.max(unit))
        return sizes.solve_scale(fraction, float(self.volume_fraction), guess, limit,
                                 1e-3 * guess)  # fmt: skip

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        a = self.cell_size
        if domain.periodic:
            if self.rotation is not None or self.radius_gradient is not None:
                raise ValueError(
                    "strut_lattice: rotated and graded lattices need a non-periodic domain"
                )
            ratio = domain.lengths / a
            if np.any(np.abs(ratio - np.rint(ratio)) > 1e-6 * np.maximum(ratio, 1.0)):
                raise ValueError(
                    "strut_lattice: a periodic domain must contain a whole number of cells"
                )
        start, end = self._struts(domain)
        law, metadata = self.strut_radius, {}
        if sizes.polydisperse(law):  # a radius per strut, each strut of the box once
            if domain.periodic:
                start, end = periodic_struts(start, end, domain)
            radii = sizes.draw(law, rng, len(start))
            if self.volume_fraction is not None:
                s = self._scaled(domain, start, end, radii)
                radii, metadata = s * radii, {"radius_scale": s}
            metadata.update(sizes.size_summary("strut_radius", radii, law))
            if self.wall_thickness is not None and np.any(radii <= self.wall_thickness):
                raise ValueError(
                    f"wall_thickness: smaller than every strut radius (the smallest drawn: "
                    f"{float(radii.min()):.4g})"
                )
        else:
            radii = self._radii(domain, start, end)
        segments = Segments(start, end, radii, round_caps=True)
        if not domain.periodic:
            lo, hi = segments.bounds()
            keep = np.all((hi >= domain.lower) & (lo <= domain.upper), axis=1)
            segments = segments.subset(keep)
        struts = PrimitiveUnion(domain, segments=segments, model=self, seed=seed_info)
        if metadata:
            struts.metadata["summary"] = metadata
        if self.wall_thickness is None:
            return struts
        from morphogen.models.combination import Combined
        from morphogen.models.nested import bores_of

        bores = bores_of(struts, self.wall_thickness)
        return Combined(domain, "difference", [struts, bores], model=self, seed=seed_info)

    def expected_count(self, domain: Domain) -> float:
        """Number of unit cells in the domain."""
        return domain.volume / self.cell_size**3

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        if self.radius_gradient is not None:
            # the local density along the gradient, averaged (Gauss-Legendre)
            x, w = np.polynomial.legendre.leggauss(6)
            r0, r1 = (r / self.cell_size for r in self.radius_gradient)
            local = [strut_fraction(self.cell, float(r0 + (r1 - r0) * (1 + xi) / 2)) for xi in x]
            return {
                "volume_fraction": float(np.dot(w, local) / 2.0),
                "volume_fraction_lower": strut_fraction(self.cell, float(r0)),
                "volume_fraction_upper": strut_fraction(self.cell, float(r1)),
            }
        if sizes.polydisperse(self.strut_radius):  # the mean radius; thin struts
            if self.volume_fraction is not None:
                return {"volume_fraction": float(self.volume_fraction)}
            law, lv = self.strut_radius, self.length_density
            return {
                "strut_radius": law.moment(1),
                "length_density": lv,
                "volume_fraction_first_order": sizes.first_order_fraction(law, lv),
            }
        r = self.radius / self.cell_size
        t = 0.0 if self.wall_thickness is None else self.wall_thickness / self.cell_size
        # solid struts on fine voxels, hollow ones on the voxels of the radius solve
        fraction = strut_fraction(self.cell, r, t, None if t else 256)
        return {"strut_radius": self.radius, "volume_fraction": fraction}
