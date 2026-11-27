"""Triangle surfaces of morphologies and of the fields of simulations, for the 3D view of
the interface (:mod:`morphogen.gui`), with the volumes that colour the faces of the box
and the sections of that view.

A :class:`Surface` is made of triangles in *groups*, each lying between two *regions*:
the one behind its triangles (``inside``) and the one in front of them (``outside``, -1
beyond the box). Its triangles turn counter-clockwise around their normal, which points
from inside to outside, so that a view of opaque regions draws, on each side of a
triangle, the region that a ray entering through it meets: the inside seen from the
front, the outside seen from the back (and the inside again behind the faces of the box,
whose shell is then seen from within, through a section). Between two regions, the
inside is the larger of the two. The regions are the phases of a morphology, or its
cells (grains) with their phases (:attr:`Surface.region_phase`,
:attr:`Surface.region_label`).

What the surfaces are:

* morphologies with a level set (two phases): its zero level set by the marching cubes of
  scikit-image (the ``surface`` extra), sampled at the centres of the voxels of
  ``resolution`` cells along the longest edge of the domain (as
  :func:`morphogen.viz.sections.level_set_measures`), padded by one cell on every side
  (periodically, or by its last values) and cut by the faces of the box; without
  scikit-image, the faces of its voxels of phase 1. The level set is the volume of the
  view (:attr:`Surface.volume`), which draws the phases on the faces of the box and on a
  section;
* tessellations without grain boundaries (Voronoi and Laguerre polycrystals): the exact
  facets of their cells (:meth:`~morphogen.models.tessellation.Tessellation.cell_complex`),
  between cells and on the faces of the box; their voxels give the sections;
* the other morphologies (cells or grains, several phases, no level set): the faces
  between voxels of different regions, those regions being the volume;
* simulations on tetrahedral meshes (:func:`vtu_surface`): the faces of the elements on
  the boundary of the mesh and between regions (phases, grains), with the field at their
  vertices (the nodal values, or on each side the mean of the values of the elements of
  that region around the node) and the displacement of their nodes;
* simulations on voxels (FFT, :func:`vti_surface`): the faces between voxels of
  different regions; the regions and the field of the voxels are volumes.

The number of triangles is bounded (``max_triangles``, :data:`MAX_TRIANGLES` by default):
the grids of morphologies are made coarser, voxel images subsampled, meshes simplified by
clustering their vertices. :func:`encode` turns a surface into the JSON data of the page,
its arrays in base64 (little-endian).
"""

from __future__ import annotations

import base64
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from morphogen.domain import Grid
from morphogen.viz.png import PALETTE, label_colors

__all__ = [
    "DIRECTIONS",
    "EXTERIOR",
    "MAX_TRIANGLES",
    "MAX_VOLUME",
    "MIN_RESOLUTION",
    "Surface",
    "Volume",
    "encode",
    "magnification",
    "mesh_surface",
    "morphology_surface",
    "voxel_surface",
    "vti_surface",
    "vtu_surface",
]

#: Largest number of triangles of a surface (the 3D view stays fluid with it)
MAX_TRIANGLES = 300_000
#: Largest number of cells along an edge of a volume (and of the voxel faces drawn)
MAX_VOLUME = 128
#: Coarsest grid (cells along the longest edge) to which a budget lowers a resolution
MIN_RESOLUTION = 8
#: Tessellations of more cells are drawn by their voxels (their exact cells take long)
MAX_EXACT_CELLS = 500
#: Region beyond the faces of the box
EXTERIOR = -1
#: Arrays of the files of fields that are not fields (phases, grains, orientations...)
NOT_FIELDS = ("phase", "grain", "cell", "euler", "outer", "fluctuation")


@dataclass
class Volume:
    """Values on the cells of a regular grid of the box (arrays ``[i, j, k]`` along x, y
    and z, encoded x fastest), which colour the faces of the box (when ``box``) and the
    sections of the 3D view.

    ``kind``: ``"level_set"`` (negative in phase 1, quantized on 8 bits, ``phi = step (q -
    128)``, interpolated linearly between the cell centres, across the faces of the box
    when ``periodic``), ``"regions"`` (the region of each cell) or ``"field"`` (values
    quantized on 16 bits over ``range``: ``value = lo + q (hi - lo) / 65535``).
    """

    kind: str
    data: np.ndarray
    periodic: bool = False
    step: float = 1.0
    range: tuple[float, float] | None = None
    box: bool = True


@dataclass
class Surface:
    """Triangles in groups between regions (see the module).

    Attributes
    ----------
    positions, normals
        Vertices (n, 3) and their unit normals, from inside to outside.
    triangles
        Vertex indices (m, 3), counter-clockwise around the normals.
    groups
        One row per group (g, 6): inside and outside regions, first vertex and number of
        vertices, first triangle and number of triangles (vertices and triangles are
        stored group after group; vertices are not shared between groups).
    region_phase, region_label
        Phase of each region, and its cell or grain (None without labels; -1: none).
    lower, upper
        The box.
    method
        How the triangles were made: ``"marching cubes"``, ``"cells"`` (exact facets),
        ``"voxels"`` (voxel faces) or ``"mesh"`` (faces of a tetrahedral mesh).
    resolution
        Cells along the longest edge of the grid used (None for meshes and cells).
    reduced
        Whether the grid was made coarser, or the mesh simplified, to fit the budget.
    volume
        Level set or regions of the cells of the box (:class:`Volume`), or None.
    field, fields
        Field drawn and the fields available (simulations).
    values
        Field at the vertices (n, 2): seen from the inside, from the outside.
    field_volume
        Field of the cells (voxel images), quantized over ``field_range``.
    field_range
        Smallest and largest finite values of the field (all its elements or voxels).
    point_field
        Whether the field is nodal (otherwise of the elements or voxels).
    displacement
        Displacement of the vertices (n, 3), or None.
    lattice
        Voxel faces: the nodes of their grid along each edge, and the node (C order) and
        the direction of the normal (:data:`DIRECTIONS`) of each vertex, which encode
        them compactly; None otherwise.
    """

    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    groups: np.ndarray
    region_phase: np.ndarray
    region_label: np.ndarray | None
    lower: np.ndarray
    upper: np.ndarray
    method: str
    resolution: int | None = None
    reduced: bool = False
    volume: Volume | None = None
    field: str | None = None
    fields: list[str] = dataclasses.field(default_factory=list)
    values: np.ndarray | None = None
    field_volume: Volume | None = None
    field_range: tuple[float, float] | None = None
    point_field: bool = False
    displacement: np.ndarray | None = None
    lattice: tuple | None = None

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    @property
    def n_vertices(self) -> int:
        return len(self.positions)

    def group_triangles(self, a: int, b: int) -> np.ndarray:
        """Triangles of the groups between regions ``a`` and ``b`` (either side)."""
        rows = [g for g in self.groups if {int(g[0]), int(g[1])} == {int(a), int(b)}]
        if not rows:
            return np.zeros((0, 3), dtype=np.int64)
        return np.concatenate([self.triangles[g[4] : g[4] + g[5]] for g in rows])

    def area(self, a: int | None = None, b: int | None = None) -> float:
        """Area of the triangles between regions ``a`` and ``b`` (all of them by
        default)."""
        tri = self.triangles if a is None else self.group_triangles(a, b)
        p = self.positions[tri].astype(np.float64)
        return float(
            0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum()
        )


# -- helpers ------------------------------------------------------------------------------------


def _skimage() -> bool:
    try:
        import skimage.measure  # noqa: F401
    except ImportError:
        return False
    return True


def _vertex_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Unit normals of the vertices: the sum of the normals of their triangles weighted by
    their areas (z for vertices of no triangle of positive area)."""
    n = len(positions)
    out = np.zeros((n, 3))
    if len(triangles):
        p = positions[triangles].astype(np.float64)
        normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        for c in range(3):
            for k in range(3):
                out[:, c] += np.bincount(triangles[:, k], weights=normal[:, c], minlength=n)
    norm = np.linalg.norm(out, axis=1)
    flat = norm <= 1e-30
    out[flat] = (0.0, 0.0, 1.0)
    norm[flat] = 1.0
    return out / norm[:, None]


def _compact(positions: np.ndarray, triangles: np.ndarray):
    """The vertices used by the triangles, renumbered."""
    used, inverse = np.unique(triangles, return_inverse=True)
    return positions[used], inverse.reshape(-1, 3), used


def _assemble(pair: np.ndarray, local: np.ndarray, size: int):
    """Vertices shared within each group of triangles: ``pair`` (m, 2) the regions of each
    triangle, ``local`` (m, 3) the identities of their corners (below ``size``), merged in
    a group when equal. Returns the triangles (group after group), the identity and the
    group of each vertex, and the table of groups (see :attr:`Surface.groups`)."""
    if not len(pair):
        empty = np.zeros((0, 3), dtype=np.int64)
        return (
            empty,
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 6), dtype=np.int64),
        )
    pairs, group = np.unique(pair, axis=0, return_inverse=True)
    group = group.ravel()
    order = np.argsort(group, kind="stable")
    group, local = group[order], local[order]
    keys = group[:, None].astype(np.int64) * int(size) + local
    unique, inverse = np.unique(keys.ravel(), return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    vgroup = unique // int(size)
    ids, first_v, count_v = np.unique(vgroup, return_index=True, return_counts=True)
    _, first_t, count_t = np.unique(group, return_index=True, return_counts=True)
    table = np.column_stack([pairs[ids, 0], pairs[ids, 1], first_v, count_v, first_t, count_t])
    return triangles, unique % int(size), vgroup, table.astype(np.int64)


def _canonical(inside: np.ndarray, outside: np.ndarray):
    """Whether each face must be turned so that its inside is the larger region (the
    exterior always outside)."""
    return (outside > inside) & (outside != EXTERIOR) | (inside == EXTERIOR)


def _cell_step(phi: np.ndarray) -> float:
    """Typical change of a level set over a cell at its zero level set: the median of its
    differences between neighbours of opposite signs (1 without any)."""
    neg = phi < 0
    steps = []
    for a in range(phi.ndim):
        if phi.shape[a] < 2:
            continue
        lo = [slice(None)] * phi.ndim
        hi = [slice(None)] * phi.ndim
        lo[a], hi[a] = slice(0, -1), slice(1, None)
        cross = neg[tuple(lo)] != neg[tuple(hi)]
        steps.append(np.abs(phi[tuple(hi)] - phi[tuple(lo)])[cross])
    steps = np.concatenate(steps) if steps else np.zeros(0)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    return float(np.median(steps)) if steps.size else 1.0


def _crossed_cubes(phi: np.ndarray) -> int:
    """Cubes of 2 x 2 x 2 samples whose values change sign (marching cubes puts two
    triangles in each, on average)."""
    neg = phi < 0
    some, every = neg, neg
    for a in range(3):
        if neg.shape[a] < 2:
            return 0
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[a], hi[a] = slice(0, -1), slice(1, None)
        some = some[tuple(lo)] | some[tuple(hi)]
        every = every[tuple(lo)] & every[tuple(hi)]
    return int(np.count_nonzero(some & ~every))


def _finest(sample, requested: int, budget: int):
    """The finest grid, of at most ``requested`` cells along the longest edge, whose
    triangles fit ``budget``: ``sample(n)`` gives the data of a grid and its count of
    triangles. From the requested grid, coarser by the square root of the ratio of the
    counts (those of a surface grow as the square of the resolution), then finer again
    while the count leaves room (thin features vanish on coarse grids, which lowers the
    count more than the resolution); six samplings at most. Returns the resolution, its
    data and count, and whether it is coarser than requested."""
    fits = last = None  # (resolution, data, count): the finest grid within, the last tried
    above = None  # the coarsest grid beyond
    n = int(requested)
    for _ in range(6):
        data, count = sample(n)
        last = (n, data, count)
        if count <= budget or n <= MIN_RESOLUTION:
            if fits is None or n > fits[0]:
                fits = last
            if n == requested or count > 0.75 * budget:
                break
        elif above is None or n < above:
            above = n
        low = fits[0] if fits is not None else MIN_RESOLUTION - 1
        top = int(requested) if above is None else above - 1
        guess = int(n * math.sqrt(budget / max(count, 1)) * 0.97)
        n = min(max(guess, low + 1, MIN_RESOLUTION), top)
        if n <= low:
            break
    n, data, count = fits if fits is not None else last
    return n, data, count, n < int(requested)


def _shape(lengths, n: int) -> tuple[int, int, int]:
    """Cells along each edge for ``n`` along the longest (square cells, as the grids of
    :meth:`morphogen.domain.Grid.for_domain`)."""
    lengths = np.asarray(lengths, dtype=float)
    h = float(lengths.max()) / int(n)
    return tuple(int(k) for k in np.maximum(1, np.rint(lengths / h)))


def _resampled(data: np.ndarray, shape, *, linear: bool, periodic: bool = False) -> np.ndarray:
    """Values of cells at the centres of the cells of a coarser grid of the same box:
    those of the cells containing them, or interpolated linearly."""
    if tuple(shape) == data.shape:
        return data
    coords = np.meshgrid(
        *[(np.arange(m) + 0.5) * (n / m) - 0.5 for n, m in zip(data.shape, shape, strict=True)],
        indexing="ij",
    )
    if not linear:
        idx = [
            np.clip(np.rint(c), 0, n - 1).astype(np.int64)
            for c, n in zip(coords, data.shape, strict=True)
        ]
        return data[tuple(idx)]
    from scipy.ndimage import map_coordinates

    mode = "grid-wrap" if periodic else "nearest"
    out = map_coordinates(data.astype(np.float32), np.stack(coords), order=1, mode=mode)
    return out.astype(data.dtype, copy=False)


def _level_volume(phi: np.ndarray, periodic: bool) -> Volume:
    """A level set at the cell centres as a :class:`Volume` (8 bits, a sixteenth of the
    change over a cell at the interface per step, at most :data:`MAX_VOLUME` cells along
    an edge)."""
    m = min(max(phi.shape), MAX_VOLUME)
    coarse = _resampled(phi, _shape(phi.shape, m), linear=True, periodic=periodic)
    step = _cell_step(coarse) / 16.0
    q = np.clip(np.floor(128.0 + coarse / step), 0, 255).astype(np.uint8)
    return Volume("level_set", q, periodic=periodic, step=step)


def _region_volume(regions: np.ndarray, box: bool = True) -> Volume:
    """Regions of cells as a :class:`Volume` (at most :data:`MAX_VOLUME` cells along an
    edge, the region of the cell holding each centre)."""
    m = min(max(regions.shape), MAX_VOLUME)
    coarse = _resampled(regions, _shape(regions.shape, m), linear=False)
    top = int(coarse.max()) if coarse.size else 0
    return Volume("regions", np.ascontiguousarray(coarse, dtype=_index_type(top)), box=box)


def _field_volume(values: np.ndarray, shape, value_range) -> Volume:
    """Values of cells as a :class:`Volume` (16 bits over their range; 0 for the values
    that are not finite)."""
    coarse = _resampled(values, shape, linear=False).astype(np.float64)
    lo, hi = value_range
    span = hi - lo if hi > lo else 1.0
    q = np.clip(np.rint((coarse - lo) / span * 65535.0), 0, 65535)
    q[~np.isfinite(coarse)] = 0
    return Volume("field", q.astype(np.uint16), range=(float(lo), float(hi)))


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)


def _scalars(values: np.ndarray) -> np.ndarray:
    """One value per row: the values, or the norms of vectors."""
    v = np.asarray(values, dtype=np.float64)
    return np.linalg.norm(v.reshape(len(v), -1), axis=1) if v.ndim > 1 else v


# -- morphologies: marching cubes of the level set ---------------------------------------------


def _level_set(micro, grid: Grid) -> np.ndarray:
    """The level set at the cell centres of ``grid``, made finite (its placeholders of no
    interface at the size of the domain, as for its measures)."""
    phi = np.asarray(micro.level_set(grid=grid).data, dtype=np.float32)
    far = np.float32(micro.domain.diagonal)
    return np.clip(np.nan_to_num(phi, nan=far, posinf=far, neginf=-far), -far, far)


def _clip(positions: np.ndarray, triangles: np.ndarray, axis: int, value: float, above: bool):
    """The triangles cut by the plane ``x_axis = value``: their parts on the side kept
    (``above`` or below), with a new vertex on each edge crossing it (the vertex of the
    edge lying on the plane, if any). The triangles left with a repeated corner are
    dropped: the surface stays closed wherever it was."""
    d = positions[:, axis] - value
    if not above:
        d = -d
    inside = d >= 0.0
    if inside.all() or not len(triangles):
        return positions, triangles
    flags = inside[triangles]
    count = flags.sum(axis=1)
    one, two = triangles[count == 1], triangles[count == 2]
    turn = np.arange(3)
    one = np.take_along_axis(one, (np.argmax(flags[count == 1], axis=1)[:, None] + turn) % 3, 1)
    two = np.take_along_axis(two, (np.argmin(flags[count == 2], axis=1)[:, None] + turn) % 3, 1)
    edges = np.concatenate([one[:, [0, 1]], one[:, [0, 2]], two[:, [0, 1]], two[:, [0, 2]]])
    edges.sort(axis=1)
    n = len(positions)
    unique, inverse = np.unique(edges[:, 0] * n + edges[:, 1], return_inverse=True)
    i, j = unique // n, unique % n
    t = d[i] / (d[i] - d[j])
    fresh = (t > 0.0) & (t < 1.0)  # otherwise an end of the edge lies on the plane
    points = positions[i[fresh]] + t[fresh, None] * (positions[j[fresh]] - positions[i[fresh]])
    points[:, axis] = value
    vertex = np.where(t <= 0.0, i, j)
    vertex[fresh] = n + np.arange(int(fresh.sum()))
    ids = vertex[inverse.ravel()]
    a, b = len(one), len(two)
    ab1, ac1, ab2, ac2 = np.split(ids, [a, 2 * a, 2 * a + b])
    out = np.concatenate(
        [
            triangles[count == 3],
            np.stack([one[:, 0], ab1, ac1], axis=1),
            np.stack([ab2, two[:, 1], two[:, 2]], axis=1),
            np.stack([ab2, two[:, 2], ac2], axis=1),
        ]
    )
    repeated = (out[:, 0] == out[:, 1]) | (out[:, 1] == out[:, 2]) | (out[:, 0] == out[:, 2])
    return np.concatenate([positions, points]), out[~repeated]


def _outward(positions, triangles, phi, origin, spacing) -> np.ndarray:
    """The triangles turned, if needed, so that their normals point to the increasing
    values of ``phi`` (samples at ``origin + index * spacing``): out of phase 1."""
    from scipy.ndimage import map_coordinates

    if not len(triangles):
        return triangles
    sample = triangles[:: max(1, len(triangles) // 4000)]
    p = positions[sample]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    norm = np.linalg.norm(normal, axis=1)
    ok = norm > 0
    normal = normal[ok] / norm[ok, None]
    centre = p[ok].mean(axis=1)
    delta = 0.25 * float(np.min(spacing))

    def at(x):
        return map_coordinates(phi, ((x - origin) / spacing).T, order=1, mode="nearest")

    rise = at(centre + delta * normal) - at(centre - delta * normal)
    return triangles if np.sum(rise > 0) >= np.sum(rise < 0) else triangles[:, ::-1].copy()


def _marching_cubes_surface(micro, resolution: int, max_triangles: int) -> Surface:
    """The zero level set of a two-phase morphology by marching cubes (see the module)."""
    from skimage.measure import marching_cubes

    domain = micro.domain
    lower, upper = domain.lower.astype(float), domain.upper.astype(float)
    per_cube = 2.0  # triangles per cube crossed by the surface, on average

    def sample(n):
        grid = Grid.for_domain(domain, shape=n)
        phi = _level_set(micro, grid)
        padded = np.pad(phi, 1, mode="wrap" if domain.periodic else "edge")
        zero = padded == 0  # values at the level: vertices on samples, pushed outside
        if zero.any():
            padded = np.where(zero, np.float32(1e-3 * _cell_step(padded)), padded)
        crossed = _crossed_cubes(padded)
        return (grid, phi, padded, crossed), int(per_cube * crossed)

    requested, reduced = int(resolution), False
    for _ in range(3):  # beyond the budget, coarser again with the count of this structure
        n, (grid, phi, padded, crossed), _, coarser = _finest(sample, requested, max_triangles)
        reduced = reduced or coarser
        spacing = grid.spacing.astype(float)
        origin = lower - 0.5 * spacing  # the sample 0 of the padded grid, a cell outside
        positions, triangles = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        if not crossed:
            break
        verts, faces, _, _ = marching_cubes(
            padded, 0.0, spacing=tuple(spacing), allow_degenerate=False
        )
        positions = verts.astype(np.float64) + origin
        triangles = _outward(positions, faces.astype(np.int64), padded, origin, spacing)
        for axis in range(3):
            positions, triangles = _clip(positions, triangles, axis, lower[axis], True)
            positions, triangles = _clip(positions, triangles, axis, upper[axis], False)
        positions, triangles, _ = _compact(positions, triangles)
        if len(triangles) <= max_triangles or n <= MIN_RESOLUTION:
            break
        per_cube = max(per_cube, 1.02 * len(triangles) / crossed)  # thin struts make more
        requested, reduced = n - 1, True
    m, k = len(triangles), len(positions)
    groups = np.array([[1, 0, 0, k, 0, m]], dtype=np.int64) if m else np.zeros((0, 6), np.int64)
    surface = Surface(
        positions=positions,
        normals=_vertex_normals(positions, triangles),
        triangles=triangles,
        groups=groups,
        region_phase=np.array([0, 1]),
        region_label=None,
        lower=lower,
        upper=upper,
        method="marching cubes",
        resolution=n,
        reduced=reduced,
        volume=_level_volume(phi, domain.periodic),
    )
    return _within(surface, max_triangles)


# -- morphologies: exact cells of tessellations -----------------------------------------------


def _exact_cells(micro) -> bool:
    from morphogen.models.tessellation import Tessellation

    return (
        isinstance(micro, Tessellation)
        and micro.boundary_thickness == 0
        and 0 < micro.n_cells <= MAX_EXACT_CELLS
    )


def _cells_surface(micro, resolution: int) -> Surface:
    """The facets of the exact cells of a tessellation, between cells and on the faces of
    the box (flat, their vertices their own); its labels give the sections."""
    cx = micro.cell_complex()
    vectors, _ = cx.facet_vectors()
    pieces = cx.facet_pieces
    grains = np.asarray(micro.cell_grains(), dtype=np.int64)  # the regions: the grains
    cells = grains[np.asarray(cx.piece_cell)]
    region_phase = np.zeros(int(grains.max()) + 1 if len(grains) else 0, dtype=np.int64)
    region_phase[grains] = micro.cell_phases
    rows = []
    for f, loop in enumerate(cx.facets):
        if len(loop) < 3:
            continue
        a, b = int(pieces[f, 0]), int(pieces[f, 1])
        inside = int(cells[a])
        outside = int(cells[b]) if b >= 0 else EXTERIOR
        if inside == outside:
            continue
        normal = vectors[f]
        norm = float(np.linalg.norm(normal))
        if norm <= 0:
            continue
        loop = np.asarray(loop, dtype=np.int64)
        if _canonical(np.array([inside]), np.array([outside]))[0]:
            inside, outside, loop, normal = outside, inside, loop[::-1], -normal
        rows.append((inside, outside, loop, normal / norm))
    rows.sort(key=lambda r: (r[0], r[1]))
    positions, normals, triangles, groups = [], [], [], []
    nv = nt = 0
    for inside, outside, loop, normal in rows:
        k = len(loop)
        fan = np.stack([np.zeros(k - 2, np.int64), np.arange(1, k - 1), np.arange(2, k)], axis=1)
        if groups and groups[-1][0] == inside and groups[-1][1] == outside:
            groups[-1][3] += k
            groups[-1][5] += k - 2
        else:
            groups.append([inside, outside, nv, k, nt, k - 2])
        positions.append(cx.vertices[loop])
        normals.append(np.repeat(normal[None], k, axis=0))
        triangles.append(fan + nv)
        nv += k
        nt += k - 2
    domain = micro.domain
    n = min(int(resolution), MAX_VOLUME)
    labels = micro.labels(grid=Grid.for_domain(domain, shape=n)).data
    return Surface(
        positions=np.concatenate(positions) if positions else np.zeros((0, 3)),
        normals=np.concatenate(normals) if normals else np.zeros((0, 3)),
        triangles=np.concatenate(triangles) if triangles else np.zeros((0, 3), np.int64),
        groups=np.array(groups, dtype=np.int64).reshape(-1, 6),
        region_phase=region_phase,
        region_label=np.arange(len(region_phase)),
        lower=domain.lower.astype(float),
        upper=domain.upper.astype(float),
        method="cells",
        resolution=None,
        volume=_region_volume(np.asarray(labels, dtype=np.int64), box=False),
    )


# -- voxel faces ---------------------------------------------------------------------------------


#: Directions of the normals of voxel faces: +x, -x, +y, -y, +z, -z
DIRECTIONS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
)


def _voxel_faces(regions: np.ndarray):
    """The faces between voxels of different regions, two triangles each: their regions
    (m, 2, the larger inside) and the identities of their corners (m, 3), a node of the
    grid (C order) and the direction of the normal (:data:`DIRECTIONS`), ``node * 6 +
    direction``."""
    shape = np.array(regions.shape)
    nodes = shape + 1
    strides = np.array([nodes[1] * nodes[2], nodes[2], 1], dtype=np.int64)
    pairs, corners = [], []
    for a in range(3):
        if shape[a] < 2:
            continue
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[a], hi[a] = slice(0, -1), slice(1, None)
        first, second = regions[tuple(lo)], regions[tuple(hi)]
        where = np.nonzero(first != second)
        inside, outside = first[where].astype(np.int64), second[where].astype(np.int64)
        base = np.stack(where, axis=1).astype(np.int64)
        base[:, a] += 1  # the plane of nodes between the two voxels
        u, v = np.eye(3, dtype=np.int64)[(a + 1) % 3], np.eye(3, dtype=np.int64)[(a + 2) % 3]
        quad = np.stack([base, base + u, base + u + v, base + v], axis=1) @ strides  # around +a
        turn = _canonical(inside, outside)
        inside, outside = np.where(turn, outside, inside), np.where(turn, inside, outside)
        quad[turn] = quad[turn][:, ::-1]
        direction = np.where(turn, 2 * a + 1, 2 * a)
        ids = quad * 6 + direction[:, None]
        corners += [ids[:, [0, 1, 2]], ids[:, [0, 2, 3]]]
        pair = np.stack([inside, outside], axis=1)
        pairs += [pair, pair]
    if not pairs:
        return np.zeros((0, 2), np.int64), np.zeros((0, 3), np.int64)
    return np.concatenate(pairs), np.concatenate(corners)


def _count_faces(regions: np.ndarray) -> int:
    """Faces between voxels of different regions."""
    total = 0
    for a in range(3):
        if regions.shape[a] >= 2:
            total += int(np.count_nonzero(np.diff(regions, axis=a) != 0))
    return total


def voxel_surface(
    regions: np.ndarray,
    region_phase,
    region_label=None,
    lower=(0.0, 0.0, 0.0),
    upper=(1.0, 1.0, 1.0),
    *,
    values: np.ndarray | None = None,
    field: str | None = None,
    fields=(),
    max_triangles: int = MAX_TRIANGLES,
    method: str = "voxels",
) -> Surface:
    """The faces between voxels of different regions of an image of regions (cells of
    the box ``[lower, upper]``), subsampled when its faces exceed ``max_triangles`` or its
    edges :data:`MAX_VOLUME` cells; the regions and the ``values`` of a field of the
    voxels are its volumes (the field: :attr:`Surface.field_volume`)."""
    regions = np.asarray(regions, dtype=np.int64)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    full = regions.shape

    def sample(m):
        coarse = _resampled(regions, _shape(full, m), linear=False)
        return coarse, 2 * _count_faces(coarse)

    requested = min(max(full), MAX_VOLUME)
    _, coarse, _, reduced = _finest(sample, requested, max_triangles)
    reduced = reduced or requested < max(full)
    nodes = np.array(coarse.shape) + 1
    pair, corners = _voxel_faces(coarse)
    triangles, local, _, groups = _assemble(pair, corners, int(np.prod(nodes)) * 6)
    node, direction = local // 6, local % 6
    index = np.stack(np.unravel_index(node, tuple(nodes)), axis=1)
    surface = Surface(
        positions=lower + index * ((upper - lower) / (nodes - 1)),
        normals=DIRECTIONS[direction],
        triangles=triangles,
        groups=groups,
        region_phase=np.asarray(region_phase, dtype=np.int64),
        region_label=None if region_label is None else np.asarray(region_label, dtype=np.int64),
        lower=lower,
        upper=upper,
        method=method,
        resolution=int(max(coarse.shape)),
        reduced=reduced,
        volume=_region_volume(coarse),
        field=field,
        fields=list(fields),
        lattice=(nodes, node, direction),
    )
    if values is not None:
        value_range = _finite_range(values)
        surface.field_range = value_range
        surface.field_volume = _field_volume(np.asarray(values), coarse.shape, value_range)
    return surface


def _regions_of(micro, grid: Grid):
    """Regions of the voxels of a morphology: its phases, or its (cell, phase) pairs
    when it has cells or grains; with the phase and the label of each region."""
    phase = np.asarray(micro.voxelize(grid=grid).data, dtype=np.int64)
    if not micro.has_labels:
        return phase, np.arange(int(phase.max()) + 1 if phase.size else 1), None
    labels = np.asarray(micro.labels(grid=grid).data, dtype=np.int64)
    span = int(phase.max()) + 1 if phase.size else 1
    unique, inverse = np.unique((labels + 1) * span + phase, return_inverse=True)
    return inverse.reshape(phase.shape), unique % span, unique // span - 1


def _voxels_of_morphology(micro, resolution: int, max_triangles: int) -> Surface:
    """The voxel faces of a morphology between its regions (see the module)."""
    domain = micro.domain

    def sample(n):
        regions, phase, label = _regions_of(micro, Grid.for_domain(domain, shape=n))
        return (regions, phase, label), 2 * _count_faces(regions)

    requested = min(int(resolution), MAX_VOLUME)
    _, (regions, phase, label), _, reduced = _finest(sample, requested, max_triangles)
    reduced = reduced or requested < int(resolution)
    surface = voxel_surface(
        regions, phase, label, domain.lower, domain.upper, max_triangles=max_triangles
    )
    surface.reduced = surface.reduced or reduced
    return surface


def morphology_surface(
    micro, resolution: int = 96, *, max_triangles: int = MAX_TRIANGLES, method: str = "auto"
) -> Surface:
    """The surface of a morphology between its regions, on a grid of ``resolution`` cells
    along the longest edge of its domain (coarser if it exceeds ``max_triangles``).

    ``method``: ``"marching cubes"`` of the level set (two phases), ``"cells"`` (exact
    facets of tessellations), ``"voxels"`` (faces between voxels of different regions),
    or ``"auto"``: the cells of tessellations without grain boundaries, the voxels of
    morphologies with cells or grains, several phases or no level set, the marching cubes
    of the others (their voxels without scikit-image). See the module.
    """
    resolution = int(resolution)
    if resolution < MIN_RESOLUTION:
        raise ValueError(f"resolution: at least {MIN_RESOLUTION} cells per edge")
    if method == "auto":
        from morphogen.viz.sections import has_level_set

        if _exact_cells(micro):
            method = "cells"
        elif has_level_set(micro) and not micro.has_labels and _skimage():
            method = "marching cubes"
        else:
            method = "voxels"
    if method == "cells":
        if not _exact_cells(micro):
            raise ValueError("method cells: tessellations without grain boundaries")
        surface = _cells_surface(micro, resolution)
        if surface.n_triangles <= max_triangles:
            return surface
        method = "voxels"
    if method == "marching cubes":
        try:
            return _marching_cubes_surface(micro, resolution, max_triangles)
        except NotImplementedError:  # no level set after all: its voxels
            method = "voxels"
    if method != "voxels":
        raise ValueError("method: auto, marching cubes, cells or voxels")
    return _voxels_of_morphology(micro, resolution, max_triangles)


# -- simulations on tetrahedral meshes -------------------------------------------------------------


def _tet_faces(nodes: np.ndarray, tets: np.ndarray, regions: np.ndarray):
    """The faces of the elements on the boundary of a mesh and between regions, turned
    outwards from their inside (the larger region; the exterior outside): corners (m, 3),
    regions (m, 2)."""
    from morphogen.mesh.tetra import TET_FACES, tet_volumes

    tets = tets.copy()
    flip = tet_volumes(nodes, tets) < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    faces = tets[:, TET_FACES].reshape(-1, 3)
    element = np.repeat(np.arange(len(tets)), 4)
    s = np.sort(faces, axis=1)
    n = len(nodes)
    if n < 2**21:
        key = (s[:, 0] * n + s[:, 1]) * n + s[:, 2]
        _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    else:  # pragma: no cover - meshes of millions of nodes
        _, inverse, counts = np.unique(s, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    times = counts[inverse]
    single = np.flatnonzero(times == 1)
    shared = np.flatnonzero(times == 2)
    shared = shared[np.argsort(inverse[shared], kind="stable")]
    f1, f2 = shared[0::2], shared[1::2]
    r1, r2 = regions[element[f1]], regions[element[f2]]
    differ = r1 != r2
    f1, f2, r1, r2 = f1[differ], f2[differ], r1[differ], r2[differ]
    keep = np.where(r1 > r2, f1, f2)  # the face of the element of the larger region
    corners = np.concatenate([faces[single], faces[keep]])
    pair = np.concatenate(
        [
            np.stack([regions[element[single]], np.full(len(single), EXTERIOR)], axis=1),
            np.stack([np.maximum(r1, r2), np.minimum(r1, r2)], axis=1),
        ]
    )
    return corners, pair.astype(np.int64)


def _node_region_means(tets: np.ndarray, regions: np.ndarray, values: np.ndarray):
    """Mean of the finite values of the elements of each region around each node: the
    sorted keys ``node * R + region`` and their means."""
    span = int(regions.max()) + 1 if regions.size else 1
    node = tets.ravel().astype(np.int64)
    region = np.repeat(regions, tets.shape[1])
    value = np.repeat(np.asarray(values, dtype=np.float64), tets.shape[1])
    finite = np.isfinite(value)
    keys, inverse = np.unique(node * span + region, return_inverse=True)
    sums = np.bincount(inverse, weights=np.where(finite, value, 0.0), minlength=len(keys))
    counts = np.bincount(inverse, weights=finite.astype(float), minlength=len(keys))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return keys, means, span


def _lookup(keys, means, span, node, region) -> np.ndarray:
    want = node * span + region
    i = np.minimum(np.searchsorted(keys, want), len(keys) - 1)
    return np.where(keys[i] == want, means[i], np.nan)


@njit(cache=True)
def _raster_tets(nodes, tets, lower, spacing, out):  # pragma: no cover - compiled
    """The element containing the centre of each cell of a grid (``out``, -1 where none,
    the last one found on shared faces)."""
    nx, ny, nz = out.shape
    size = (nx, ny, nz)
    first = np.empty(3, np.int64)
    last = np.empty(3, np.int64)
    m = np.empty((3, 3))
    for e in range(tets.shape[0]):
        p0 = nodes[tets[e, 0]]
        lo = p0.copy()
        hi = p0.copy()
        for c in range(1, 4):
            q = nodes[tets[e, c]]
            for a in range(3):
                lo[a] = min(lo[a], q[a])
                hi[a] = max(hi[a], q[a])
        for a in range(3):
            first[a] = max(0, math.ceil((lo[a] - lower[a]) / spacing[a] - 0.5))
            last[a] = min(size[a] - 1, math.floor((hi[a] - lower[a]) / spacing[a] - 0.5))
        if first[0] > last[0] or first[1] > last[1] or first[2] > last[2]:
            continue
        for c in range(3):  # columns: the edges from the first corner
            q = nodes[tets[e, c + 1]]
            for a in range(3):
                m[a, c] = q[a] - p0[a]
        det = (
            m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
            - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
        )
        if det == 0.0:
            continue
        inv = np.linalg.inv(m)
        tol = 1e-9
        for i in range(first[0], last[0] + 1):
            x = lower[0] + (i + 0.5) * spacing[0] - p0[0]
            for j in range(first[1], last[1] + 1):
                y = lower[1] + (j + 0.5) * spacing[1] - p0[1]
                for k in range(first[2], last[2] + 1):
                    z = lower[2] + (k + 0.5) * spacing[2] - p0[2]
                    l1 = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2] * z
                    l2 = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2] * z
                    l3 = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2] * z
                    if l1 >= -tol and l2 >= -tol and l3 >= -tol and l1 + l2 + l3 <= 1.0 + tol:
                        out[i, j, k] = e


def _index_type(top: int):
    """The smallest unsigned integers holding ``top``."""
    return np.uint8 if top < 256 else (np.uint16 if top < 65536 else np.uint32)


def _mesh_volumes(nodes, tets, regions, n_regions, lower, upper, resolution, values=None):
    """The regions of the cells of a grid of the box (``resolution`` cells along its
    longest edge) from the elements containing their centres (``n_regions`` where none:
    the pores left out of the mesh), and the field there: the value of the element, or
    the nodal values interpolated linearly."""
    shape = _shape(upper - lower, min(int(resolution), MAX_VOLUME))
    spacing = (upper - lower) / np.array(shape)
    element = np.full(shape, -1, dtype=np.int64)
    _raster_tets(nodes, tets, lower, spacing, element)
    inside = element >= 0
    cells = np.where(inside, regions[np.maximum(element, 0)], n_regions)
    volume = Volume("regions", np.ascontiguousarray(cells, dtype=_index_type(n_regions)), box=False)
    if values is None:
        return volume, None
    field = np.full(shape, np.nan)
    if len(values) == len(tets):
        field[inside] = values[element[inside]]
    else:  # nodal values, by the barycentric coordinates of the centres
        index = np.argwhere(inside)
        for start in range(0, len(index), 1 << 18):
            chunk = index[start : start + (1 << 18)]
            e = element[tuple(chunk.T)]
            x = nodes[tets[e]]
            local = np.linalg.solve(
                np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2),
                (lower + (chunk + 0.5) * spacing - x[:, 0])[..., None],
            )[..., 0]
            weights = np.concatenate([1.0 - local.sum(axis=1, keepdims=True), local], axis=1)
            field[tuple(chunk.T)] = np.einsum("ij,ij->i", weights, values[tets[e]])
    return volume, _field_volume(field, shape, _finite_range(values))


def mesh_surface(
    nodes,
    elements,
    regions,
    region_phase,
    region_label=None,
    lower=None,
    upper=None,
    *,
    values=None,
    field: str | None = None,
    fields=(),
    displacement=None,
    resolution: int | None = 96,
    max_triangles: int = MAX_TRIANGLES,
) -> Surface:
    """The faces of a tetrahedral mesh (corners of ``elements``) on its boundary and
    between ``regions`` of elements, with a field (``values``: one per node, or one per
    element, averaged at the nodes in each region) and the ``displacement`` of the nodes.

    Vertices are shared by the faces of a group on the same face of the box (or inside
    it), whose normals they average. The volumes of the sections are the regions and the
    field of the cells of a grid of ``resolution`` cells along the longest edge (None:
    none), from the elements containing their centres. A surface beyond
    ``max_triangles`` is simplified by clustering its vertices (:func:`_clustered`).
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(elements, dtype=np.int64)[:, :4]
    regions = np.asarray(regions, dtype=np.int64)
    lower = nodes.min(axis=0) if lower is None else np.asarray(lower, dtype=float)
    upper = nodes.max(axis=0) if upper is None else np.asarray(upper, dtype=float)
    corners, pair = _tet_faces(nodes, tets, regions)
    # the face of the box of each boundary face (1-6; 7 inside: walls of pores), 0 between
    # regions: vertices are shared only within a group on the same plane
    tol = 1e-7 * float(np.max(upper - lower))
    plane = np.zeros(len(corners), dtype=np.int64)
    boundary = pair[:, 1] == EXTERIOR
    plane[boundary] = 7
    for a in range(3):
        for side, value in enumerate((lower[a], upper[a])):
            on = np.abs(nodes[:, a] - value) <= tol
            plane[boundary & np.all(on[corners], axis=1)] = 1 + 2 * a + side
    local = corners * 8 + plane[:, None]
    triangles, ids, vgroup, groups = _assemble(pair, local, len(nodes) * 8)
    node = ids // 8
    positions = nodes[node]
    surface = Surface(
        positions=positions,
        normals=_vertex_normals(positions, triangles),
        triangles=triangles,
        groups=groups,
        region_phase=np.asarray(region_phase, dtype=np.int64),
        region_label=None if region_label is None else np.asarray(region_label, dtype=np.int64),
        lower=lower,
        upper=upper,
        method="mesh",
        field=field,
        fields=list(fields),
    )
    if values is not None:
        values = _scalars(values)
    if resolution is not None:
        surface.volume, surface.field_volume = _mesh_volumes(
            nodes, tets, regions, len(surface.region_phase), lower, upper, resolution, values
        )
    if values is not None:
        surface.field_range = _finite_range(values)
        inside, outside = groups[vgroup, 0], groups[vgroup, 1]  # every group has vertices
        if len(values) == len(nodes):
            front = back = values[node]
            surface.point_field = True
        elif len(values) == len(tets):
            keys, means, span = _node_region_means(tets, regions, values)
            front = _lookup(keys, means, span, node, inside)
            back = _lookup(keys, means, span, node, np.maximum(outside, 0))
            back = np.where(outside >= 0, back, front)
        else:
            raise ValueError("values: one per node or one per element")
        surface.values = np.stack([front, back], axis=1)
    if displacement is not None:
        u = np.asarray(displacement, dtype=np.float64).reshape(len(nodes), -1)[:, :3]
        surface.displacement = u[node]
    return _within(surface, max_triangles)


def vtu_surface(
    path,
    field: str | None = None,
    *,
    resolution: int | None = 96,
    max_triangles: int = MAX_TRIANGLES,
) -> Surface:
    """The surface of the fields of a simulation on a tetrahedral mesh (a ``.vtu`` file of
    :func:`morphogen.io.write_vtu_tet`): its faces between its phases (and grains, the
    ``cell`` array of polycrystals), ``field`` at their vertices (by default the von Mises
    stress, else its first field; ``"displacement"``: its norm) and the displacement of
    their nodes; its volumes for the sections (:func:`mesh_surface`)."""
    from morphogen.io.vtk import read_vtu

    data = read_vtu(path)
    nodes, cells = data["points"], data["cells"]
    if cells.ndim != 2 or cells.shape[1] not in (4, 10):
        raise ValueError(f"{Path(path).name}: not a mesh of tetrahedra")
    cell_data, point_data = data["cell_data"], data["point_data"]
    phase = np.asarray(cell_data.get("phase", np.zeros(len(cells))), dtype=np.int64)
    grain = cell_data.get("cell", cell_data.get("grain"))
    if grain is None:
        regions, region_phase, region_label = phase, np.arange(int(phase.max(initial=0)) + 1), None
    else:
        grain = np.asarray(grain, dtype=np.int64)
        span = int(phase.max(initial=0)) + 1
        unique, inverse = np.unique((grain + 1) * span + phase, return_inverse=True)
        regions, region_phase, region_label = inverse.ravel(), unique % span, unique // span - 1
    names = [n for n in dict.fromkeys([*point_data, *cell_data]) if n not in NOT_FIELDS]
    field = field or _default_field(names)
    values = None
    if field is not None:
        if field not in names:
            raise ValueError(f"field: one of {', '.join(names) or 'none here'}")
        values = point_data[field] if field in point_data else cell_data[field]
    boxes = data.get("field_data", {})
    lower = boxes.get("box_lower")
    upper = boxes.get("box_upper")
    return mesh_surface(
        nodes,
        cells,
        regions,
        region_phase,
        region_label,
        None if lower is None else np.asarray(lower, dtype=float).reshape(3),
        None if upper is None else np.asarray(upper, dtype=float).reshape(3),
        values=values,
        field=field,
        fields=names,
        displacement=point_data.get("displacement"),
        resolution=resolution,
        max_triangles=max_triangles,
    )


def _default_field(names: list[str]) -> str | None:
    """The field drawn first: the von Mises stress, else the first field."""
    return "von_mises" if "von_mises" in names else (names[0] if names else None)


def vti_surface(
    path,
    field: str | None = None,
    *,
    phases=None,
    morphology=None,
    max_triangles: int = MAX_TRIANGLES,
) -> Surface:
    """The surface of the fields of a simulation on voxels (a ``.vti`` file of fields):
    the faces between its phases and grains (``grain``) and ``field`` of the voxels as a
    volume (by default the von Mises stress, else its first field). The phases are the
    ``phase`` array of the file, else ``phases`` (an image of the same shape), else those
    of ``morphology`` voxelized on the grid of the file, else a single phase. Only the
    arrays needed are read."""
    from morphogen.domain import Domain
    from morphogen.io.vtk import read_vti, vti_names

    names = [n for n in vti_names(path) if n not in NOT_FIELDS]
    field = field or _default_field(names)
    if field is not None and field not in names:
        raise ValueError(f"field: one of {', '.join(names) or 'none here'}")
    wanted = {"phase", "grain"} | ({field} if field else set())
    images = {img.name: img for img in read_vti(path, names=wanted)}
    if not images:
        raise ValueError(f"{Path(path).name}: no fields of voxels")
    first = next(iter(images.values()))
    shape = first.data.shape
    lower = np.asarray(first.origin, dtype=float)
    upper = lower + np.asarray(first.spacing, dtype=float) * np.array(shape)
    if "phase" in images:
        phase = images["phase"].data
    elif phases is not None and np.shape(phases) == shape:
        phase = phases
    elif morphology is not None:
        grid = Grid(Domain(size=tuple(upper - lower), origin=tuple(lower)), shape)
        phase = morphology.voxelize(grid=grid).data
    else:
        phase = np.zeros(shape, dtype=np.int64)
    phase = np.asarray(phase, dtype=np.int64)
    if "grain" in images:
        grain = np.asarray(images["grain"].data, dtype=np.int64)
        span = int(phase.max(initial=0)) + 1
        unique, inverse = np.unique((grain + 1) * span + phase, return_inverse=True)
        regions, region_phase = inverse.reshape(shape), unique % span
        region_label = unique // span - 1
    else:
        regions, region_phase, region_label = phase, np.arange(int(phase.max(initial=0)) + 1), None
    values = None if field is None else np.asarray(images[field].data, dtype=np.float64)
    return voxel_surface(
        regions,
        region_phase,
        region_label,
        lower,
        upper,
        values=values,
        field=field,
        fields=names,
        max_triangles=max_triangles,
    )


# -- budget ------------------------------------------------------------------------------------


def _within(surface: Surface, max_triangles: int) -> Surface:
    """The surface, simplified by :func:`_clustered` when it exceeds ``max_triangles``."""
    if surface.n_triangles <= max_triangles:
        return surface
    return _clustered(surface, max_triangles)


def _clustered(surface: Surface, max_triangles: int) -> Surface:
    """A coarser surface by clustering of vertices: those of a group in the same cell of
    a grid merge (their attributes averaged), and the triangles that lose a corner, or
    repeat another, are dropped; the cells grow until the triangles fit the budget."""
    area = max(surface.area(), 1e-300)
    size = math.sqrt(2.0 * area / max(int(max_triangles), 1))
    span = surface.upper - surface.lower
    group_of = np.repeat(np.arange(len(surface.groups)), surface.groups[:, 3])
    for _ in range(40):
        cells = np.floor((surface.positions - surface.lower) / size).astype(np.int64)
        dims = np.maximum(np.ceil(span / size).astype(np.int64) + 2, 1)
        cell = ((cells[:, 0] + 1) * dims[1] + cells[:, 1] + 1) * dims[2] + cells[:, 2] + 1
        key = group_of * int(np.prod(dims)) + cell
        unique, inverse = np.unique(key, return_inverse=True)
        tri = inverse.ravel()[surface.triangles]
        ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
        tri = tri[ok]
        _, first = np.unique(np.sort(tri, axis=1), axis=0, return_index=True)
        tri = tri[np.sort(first)]
        if len(tri) <= max_triangles:
            break
        size *= 1.25
    counts = np.bincount(inverse.ravel(), minlength=len(unique)).astype(float)

    def mean(a):
        out = np.zeros((len(unique), a.shape[1]))
        for c in range(a.shape[1]):
            out[:, c] = np.bincount(inverse.ravel(), weights=a[:, c], minlength=len(unique))
        return out / counts[:, None]

    positions = mean(surface.positions)
    vgroup = unique // int(np.prod(dims))
    # triangles in the order of their groups (vertices are, by their keys)
    tri = tri[np.argsort(vgroup[tri[:, 0]], kind="stable")]
    ids, first_v, count_v = np.unique(vgroup, return_index=True, return_counts=True)
    tgroup = vgroup[tri[:, 0]]
    t_ids, first_t, count_t = np.unique(tgroup, return_index=True, return_counts=True)
    start = np.zeros(len(ids), dtype=np.int64)
    number = np.zeros(len(ids), dtype=np.int64)
    where = np.searchsorted(ids, t_ids)
    start[where], number[where] = first_t, count_t
    groups = np.column_stack(
        [surface.groups[ids, 0], surface.groups[ids, 1], first_v, count_v, start, number]
    )
    out = Surface(
        positions=positions,
        normals=_vertex_normals(positions, tri),
        triangles=tri,
        groups=groups.astype(np.int64),
        region_phase=surface.region_phase,
        region_label=surface.region_label,
        lower=surface.lower,
        upper=surface.upper,
        method=surface.method,
        resolution=surface.resolution,
        reduced=True,
        volume=surface.volume,
        field=surface.field,
        fields=surface.fields,
        field_volume=surface.field_volume,
        field_range=surface.field_range,
        point_field=surface.point_field,
    )
    if surface.values is not None:
        out.values = mean(np.nan_to_num(surface.values, nan=0.0))
    if surface.displacement is not None:
        out.displacement = mean(surface.displacement)
    return out


# -- encoding ----------------------------------------------------------------------------------


def _b64(array: np.ndarray, dtype: str, order: str = "C") -> str:
    data = np.asarray(array).astype(dtype, copy=False)
    return base64.b64encode(data.tobytes(order=order)).decode("ascii")


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


def _volume(volume: Volume | None) -> dict | None:
    if volume is None:
        return None
    dtype = {"uint8": "<u1", "uint16": "<u2", "uint32": "<u4"}[volume.data.dtype.name]
    out = {
        "kind": volume.kind,
        "shape": [int(k) for k in volume.data.shape],
        "dtype": volume.data.dtype.name,
        "data": _b64(volume.data, dtype, order="F"),  # x fastest, as 3D textures
        "box": bool(volume.box),
    }
    if volume.kind == "level_set":
        out.update(periodic=bool(volume.periodic), step=float(volume.step))
    if volume.range is not None:
        out["range"] = [float(volume.range[0]), float(volume.range[1])]
    return out


#: The share of the values of a field left beyond each end of its default colour scale
SHOWN_TAIL = 0.01


def _shown_range(values, lo: float, hi: float) -> list[float]:
    """The default colour scale of a field on the vertices: from the first to the last
    percentile of its values, so that the peak of a singular field (the tip of a crack,
    a poor element) does not darken all the others; the whole range when that would leave
    too little (the phases of a field of few values)."""
    if values is None:
        return [float(lo), float(hi)]
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [float(lo), float(hi)]
    a, b = np.quantile(finite, [SHOWN_TAIL, 1.0 - SHOWN_TAIL])
    if not b > a or (b - a) < 1e-6 * max(abs(hi - lo), 1e-300):
        return [float(lo), float(hi)]
    return [float(a), float(b)]


def magnification(surface: Surface) -> float | None:
    """The magnification of the displacement drawing the largest one as a tenth of the
    box (as the videos of :mod:`morphogen.viz.video`), None without displacement."""
    if surface.displacement is None or not len(surface.displacement):
        return None
    peak = float(np.max(np.linalg.norm(surface.displacement, axis=1)))
    size = float(np.max(surface.upper - surface.lower))
    return 0.1 * size / peak if peak > 0 else 0.0


def encode(surface: Surface) -> dict:
    """JSON data of a surface for the page: its arrays in base64, little-endian
    (``positions``, ``normals``, field ``values`` and ``displacement`` in float32,
    ``indices`` in uint32, the volumes in unsigned integers, x fastest; voxel faces give
    the node and the direction of each vertex instead of its position and normal,
    ``lattice``), its groups, the phase, label and colours of its regions (those of the
    sections, :data:`morphogen.viz.png.PALETTE`, and of their cells)."""
    phase = np.asarray(surface.region_phase, dtype=np.int64)
    colors = [_hex(PALETTE[min(max(int(p), 0), len(PALETTE) - 1)]) for p in phase]
    regions: dict = {"phase": phase.tolist(), "color": colors, "label": None, "label_color": None}
    if surface.region_label is not None:
        label = np.asarray(surface.region_label, dtype=np.int64)
        regions["label"] = label.tolist()
        regions["label_color"] = [
            _hex(c) if k >= 0 else colors[i]
            for i, (k, c) in enumerate(zip(label, label_colors(label), strict=True))
        ]
    out = {
        "method": surface.method,
        "resolution": surface.resolution,
        "reduced": bool(surface.reduced),
        "lower": [float(v) for v in surface.lower],
        "upper": [float(v) for v in surface.upper],
        "vertices": surface.n_vertices,
        "triangles": surface.n_triangles,
        "positions": None,
        "normals": None,
        "lattice": None,
        "indices": _b64(surface.triangles, "<u4"),
        "groups": surface.groups.astype(np.int64).tolist(),
        "regions": regions,
        "volume": _volume(surface.volume),
        "field": None,
        "displacement": None,
        "magnification": None,
    }
    if surface.lattice is None:
        out["positions"] = _b64(surface.positions, "<f4")
        out["normals"] = _b64(surface.normals, "<f4")
    else:  # voxel faces: the nodes and directions of the vertices
        nodes, node, direction = surface.lattice
        out["lattice"] = {
            "nodes": [int(k) for k in nodes],
            "node": _b64(node, "<u4"),
            "direction": _b64(direction, "<u1"),
        }
    if surface.field is not None:
        lo, hi = surface.field_range or (0.0, 1.0)
        values = surface.values
        out["field"] = {
            "name": surface.field,
            "names": list(surface.fields),
            "range": [float(lo), float(hi)],
            "shown": _shown_range(values, lo, hi),
            "point": bool(surface.point_field),
            "values": None if values is None else _b64(np.nan_to_num(values, nan=lo), "<f4"),
            "volume": _volume(surface.field_volume),
        }
    if surface.displacement is not None:
        out["displacement"] = _b64(surface.displacement, "<f4")
        out["magnification"] = magnification(surface)
    return out
