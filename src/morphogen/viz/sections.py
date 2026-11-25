"""Sections of microstructures computed in their planes, without their 3D images.

A section normal to ``axis`` is sampled on the pixel centres of a plane: those of the
voxel grid of the domain whose cells have the edge ``h = max(L_u, L_v) / resolution``
(``L_u``, ``L_v``: the edges of the plane), ``rint(L_a / h)`` cells along each edge
(:func:`volume_grid`), at the cells of index ``min(floor(fraction n), n - 1)`` along the
axis (:func:`plane_index`: the cells containing ``lower + fraction L``, the middle ones for
``fraction = 1/2`` as in :func:`morphogen.viz.png.slice_of`; ``fraction`` 0 and 1 give the
first and the last cells, inside the domain). The structure is evaluated on a grid one
cell thick (:func:`plane_grid`) whose cell centres are those of that slice, bit for bit:
a section is the slice of the voxel image at the same resolution, at the cost of ``n^2``
values instead of ``n^3`` (up to 2048 x 2048 pixels and more).

Every 2D array is displayed like :func:`morphogen.viz.png.slice_of`: the first remaining
axis horizontal, the second vertical, rows from the top (``np.flipud(s.T)`` of the slice
``s[u, v]``).

How the structures are sampled (:attr:`Section.sampling`):

* ``plane``: unions of spheres and segments (Boolean and hard-core spheres, fibres, strut
  and random lattices, foams, trees), TPMS, tessellations, Gaussian fields, lattice
  fields (automata, growth, reconstructions, tile assemblies; the cells themselves when
  the grid is their lattice), laminates, honeycombs, nacre, Bouligand plywoods, level-set
  foams, osteons, fractal sponges, scale-rich networks, grains with subgrains, woven
  textiles and the shapes of Microgen: they only evaluate the points of the grid they are
  given. So are the combinations of such structures, Cox spheres in such aggregates, and
  the coatings and shells of unions of spheres and segments (exact distances);
* ``plane`` too for dead leaves and transparent dead leaves, on a cell as thick as the box
  along the axis (centred on the plane): their kernels wrap the leaves over the grid, a
  periodic grid being the whole box;
* ``volume``: the slice of the voxel image at the resolution, for the coatings and shells
  computed through the distance transform of the voxels of their structure (any
  structure but unions of spheres and segments), and the combinations, Cox spheres and
  coatings of such structures or of periodic dead leaves; ``coarse`` beyond
  :data:`MAX_VOLUME` voxels: the slice of a coarser voxel image, enlarged (phases and
  labels of the coarse voxels containing the pixel centres, level set interpolated
  linearly).

The level set ``phi`` (negative in phase 1) gives the anti-aliased fraction of phase 1 in
each pixel (:func:`pixel_coverage`). A straight interface passing at the signed distance
``d = phi / |grad phi|`` from the centre of a pixel of edges ``(h_u, h_v)`` leaves the
whole pixel in phase 1 for ``d <= -w/2`` and none of it for ``d >= w/2``, ``w = h_u |n_u| +
h_v |n_v|`` being the width of the pixel across the interface (``n``: its normal); in
between, linearly (exact for interfaces along the axes),

::

    coverage = clip(1/2 - d / w, 0, 1) = clip(1/2 - phi / (|D_u phi| + |D_v phi|), 0, 1)

with ``D_a phi = h_a dphi/da`` estimated by the larger of the one-sided differences of the
level set with the two neighbours along ``a``: independent of the scale of the level set
(Gaussian fields, concentrations), and finite at its extrema (features thinner than a
pixel keep a partial coverage).

The structures of several phases (dead leaves of several families, osteons, coatings,
inclusions, tessellations whose cells have phases) have one implicit function per phase
(:meth:`~morphogen.models.base.Microstructure.phase_level_set`), each negative in its
phase: the sections sampled on their plane keep the fraction of each phase in each pixel
(:func:`phase_coverages`: the coverage of each function, scaled so that they sum to one),
and their images blend the colours of the phases by them.

:func:`level_set_measures` measures the volume fraction and the specific surface of
phase 1 from the 3D level set at a modest resolution (the smoothed Heaviside and the
marching cubes of its zero level set).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from morphogen.domain import Domain, Grid
from morphogen.models.bouligand import BouligandStructure
from morphogen.models.combination import Combined
from morphogen.models.deadleaves import DeadLeaves
from morphogen.models.gaussian import GaussianField
from morphogen.models.honeycomb import PrismaticCells
from morphogen.models.laminate import Laminate
from morphogen.models.lattice_field import LatticeField
from morphogen.models.levelset_foam import DistanceFieldFoam
from morphogen.models.microgen import MicrogenStructure
from morphogen.models.nacre import NacreStructure
from morphogen.models.nested import ClusteredSpheres, CoatedStructure, FractalStructure
from morphogen.models.nested_grains import NestedGrainStructure
from morphogen.models.osteons import OsteonStructure
from morphogen.models.planar import plane_axes
from morphogen.models.scale_rich import ScaleRichStructure
from morphogen.models.scale_rich_3d import ScaleRichBeams, ScaleRichPlates
from morphogen.models.tessellation import Tessellation
from morphogen.models.textile import TextileStructure
from morphogen.models.tpms import TPMSStructure
from morphogen.models.transparent import TransparentDeadLeaves
from morphogen.models.union import PrimitiveUnion
from morphogen.viz.png import PALETTE, label_colors

__all__ = [
    "CONTOUR",
    "MAX_PHASES",
    "MAX_VOLUME",
    "Section",
    "has_level_set",
    "is_planar",
    "level_set_measures",
    "phase_coverages",
    "pixel_coverage",
    "plane_grid",
    "plane_index",
    "render",
    "section",
    "sections",
    "volume_grid",
]

#: Largest voxel image (in voxels) of the structures that cannot be sampled on a plane;
#: beyond, their sections are those of a coarser image, enlarged.
MAX_VOLUME = 256**3
#: Most phases whose implicit functions the sections evaluate (one function per phase);
#: beyond, the pixels take the plain colour of the phase of their centre.
MAX_PHASES = 8
#: Colour of the contours.
CONTOUR = np.array([20.0, 20.0, 20.0])
_AXES = "xyz"

#: Structures whose methods only evaluate the points of the grid they are given (point
#: queries, or kernels of the cell centres of any grid).
_POINTWISE = (
    PrimitiveUnion,
    TPMSStructure,
    Tessellation,
    GaussianField,
    LatticeField,
    Laminate,
    PrismaticCells,
    BouligandStructure,
    NacreStructure,
    DistanceFieldFoam,
    OsteonStructure,
    FractalStructure,
    ScaleRichStructure,
    ScaleRichPlates,
    ScaleRichBeams,
    TextileStructure,
    MicrogenStructure,
)
#: Structures whose kernels wrap over the grid (sampled on a cell as thick as the box).
_LEAVES = (DeadLeaves, TransparentDeadLeaves)


# -- grids ------------------------------------------------------------------------------------


def _axis(axis) -> int:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)) or not 0 <= axis <= 2:
        raise ValueError(f"axis: 0, 1 or 2 (the normal x, y or z), got {axis!r}")
    return int(axis)


def _fraction(fraction) -> float:
    try:
        f = float(fraction)
    except (TypeError, ValueError):
        f = math.nan
    if isinstance(fraction, (bool, str)) or not 0.0 <= f <= 1.0:
        raise ValueError(f"fraction: between 0 and 1, got {fraction!r}")
    return f


def _resolution(resolution) -> int:
    try:
        n = int(resolution)
    except (TypeError, ValueError):
        n = 0
    if isinstance(resolution, (bool, str)) or n != resolution or n < 1:
        raise ValueError(f"resolution: a positive integer, got {resolution!r}")
    return n


def volume_grid(domain: Domain, axis: int = 2, resolution: int = 512) -> Grid:
    """The voxel grid of ``domain`` whose slices normal to ``axis`` are its sections at
    ``resolution``: cells of edge about ``h = max(L_u, L_v) / resolution`` (``L_u``,
    ``L_v``: the edges of the plane), ``max(1, rint(L_a / h))`` along each axis, square in
    the plane up to the rounding of its edges to whole pixels."""
    axis, resolution = _axis(axis), _resolution(resolution)
    u, v = plane_axes(axis)
    lengths = domain.lengths
    h = max(lengths[u], lengths[v]) / resolution
    n = np.maximum(1, np.rint(lengths / h)).astype(int)
    return Grid(domain, tuple(int(k) for k in n))


def plane_index(domain: Domain, axis: int = 2, fraction: float = 0.5, resolution: int = 512) -> int:
    """Index along ``axis`` of the cells of :func:`volume_grid` holding the section: those
    containing ``lower + fraction L``, ``min(floor(fraction n), n - 1)`` (``n // 2`` for a
    half)."""
    n = volume_grid(domain, axis, resolution).shape[_axis(axis)]
    return min(math.floor(_fraction(fraction) * n), n - 1)


def _slab(grid: Grid, axis: int, index: int, thickness: float | None = None) -> Grid:
    """The cells ``index`` of ``grid`` along ``axis`` as a grid one cell thick: the same
    cells in the plane, of the thickness of the cells (or ``thickness``), centred on the
    coordinate of their centres in ``grid`` bit for bit (``lower + 0.5 t`` computes to it)."""
    centre = float(grid.centers(axis)[index])
    t = float(grid.spacing[axis]) if thickness is None else float(thickness)
    half = 0.5 * t
    lower = centre - half
    for _ in range(8):
        got = lower + half
        if got == centre:
            break
        lower = math.nextafter(lower, math.inf if got < centre else -math.inf)
    d = grid.domain
    origin, size, shape = list(d.origin), list(d.size), list(grid.shape)
    origin[axis], size[axis], shape[axis] = lower, t, 1
    return Grid(Domain(size=size, origin=origin, periodic=d.periodic, unit=d.unit), tuple(shape))


def plane_grid(domain: Domain, axis: int = 2, fraction: float = 0.5, resolution: int = 512) -> Grid:
    """A grid one cell thick of the plane normal to ``axis`` at ``fraction`` of the domain
    (0 to 1 from ``domain.lower``): the cells of :func:`volume_grid` (``resolution``
    pixels along the longest edge of the plane, square pixels) of index
    :func:`plane_index` along the axis, whose centres lie exactly on the plane."""
    grid = volume_grid(domain, axis, resolution)
    return _slab(grid, _axis(axis), plane_index(domain, axis, fraction, resolution))


# -- sections ---------------------------------------------------------------------------------


@dataclass
class Section:
    """A section of a microstructure normal to ``axis``, its 2D arrays displayed like
    :func:`morphogen.viz.png.slice_of` (first in-plane axis horizontal, rows from the top).

    Attributes
    ----------
    axis, fraction
        Normal of the plane and the position asked for (0 to 1 from the lower face).
    phase
        Phase at each pixel centre (uint8).
    level_set
        Level set at the pixel centres, negative in phase 1 (float32; None for the
        structures without one).
    coverage
        Anti-aliased fraction of phase 1 in each pixel, from the level set (float32 in
        [0, 1], :func:`pixel_coverage`; None without a level set).
    labels
        Cell or grain labels (int32) of the structures that have them, else None.
    pixel
        Pixel size (the larger of its two edges).
    extent
        In-plane bounds ``(h0, h1, v0, v1)``: horizontal, then vertical.
    names
        Names of the horizontal and vertical axes, e.g. ``("x", "y")``.
    position
        Coordinate of the plane along the axis (the centre of its cells).
    index
        Index of the plane in the voxel grid of the same pixels (:func:`volume_grid`).
    sampling
        ``plane`` (evaluated on the plane), ``volume`` (slice of the voxel image) or
        ``coarse`` (slice of a coarser voxel image, enlarged).
    coverages
        For the structures of several phases sampled on the plane: the anti-aliased
        fraction of each phase in each pixel, from the implicit function of each phase
        (float32 in [0, 1], by phase, :func:`phase_coverages`); None otherwise.
    """

    axis: int
    fraction: float
    phase: np.ndarray
    level_set: np.ndarray | None
    coverage: np.ndarray | None
    labels: np.ndarray | None
    pixel: float
    extent: tuple[float, float, float, float]
    names: tuple[str, str]
    position: float = math.nan
    index: int = 0
    sampling: str = "plane"
    coverages: dict[int, np.ndarray] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Rows and columns of the arrays."""
        return tuple(int(n) for n in self.phase.shape)

    @property
    def title(self) -> str:
        """Name of the plane, e.g. ``"x-y"``."""
        return f"{self.names[0]}-{self.names[1]}"


def has_level_set(micro) -> bool:
    """Whether the phase 1 of ``micro`` is described by a level set (two phases, the model
    saying so; structures without a model are tried)."""
    model = getattr(micro, "model", None)
    return True if model is None else bool(getattr(model, "has_level_set", True))


def is_planar(micro, resolution: int = 512) -> bool:
    """Whether the sections of ``micro`` reduce to the x-y one: one cell along z at this
    resolution, or a lattice of one cell along z (2D structures, extruded)."""
    if isinstance(micro, LatticeField) and micro.shape[2] == 1:
        return True
    return volume_grid(micro.domain, 2, resolution).shape[2] == 1


def _pointwise(micro) -> bool:
    """Whether the methods of ``micro`` evaluate exactly the points of any grid (a slab
    of the voxel grid) they are given."""
    if isinstance(micro, _LEAVES):
        return not micro.domain.periodic  # their kernels wrap the leaves over the grid
    if isinstance(micro, ClusteredSpheres):
        return micro.aggregate_phase is None or _pointwise(micro.aggregates)
    if isinstance(micro, CoatedStructure):
        # exact distances to unions of spheres and segments; else the EDT of the voxels
        return (
            isinstance(micro.inner, PrimitiveUnion)
            and _pointwise(micro.inner)
            and (micro.bores is None or _pointwise(micro.bores))
        )
    if isinstance(micro, Combined):
        return all(_pointwise(p) for p in micro.parts)
    if isinstance(micro, NestedGrainStructure):  # the k-d trees of the grains and subgrains
        return True  # and the point queries of the inner structure
    return isinstance(micro, _POINTWISE)


def _display(a: np.ndarray) -> np.ndarray:
    """A slice ``s[u, v]`` as displayed by ``slice_of``."""
    return np.ascontiguousarray(np.flipud(a.T))


def _evaluate(
    micro, grid: Grid, axis: int, index: int = 0, cells: int | None = None, implicit: bool = False
):
    """Phase, level set and labels of ``micro`` on ``grid``, their slices ``index`` along
    ``axis`` as arrays ``[u, v]``; ``cells``: the slice of the lattice of a lattice field
    (the grid being a slab of it), whose cells give the phases and labels as they are.
    With ``implicit``, the implicit functions of the phases of a structure of several
    phases as well (by phase; None for the others)."""

    def take(image):
        return np.take(image.data, index, axis=axis)

    if cells is not None:
        phase = np.take(micro.phases, cells, axis=axis)
    else:
        phase = take(micro.voxelize(grid=grid))
    level = levels = None
    if has_level_set(micro):
        try:
            level = take(micro.level_set(grid=grid)).astype(np.float32)
        except NotImplementedError:
            level = None
    elif implicit and cells is None:
        levels = _phase_functions(micro, grid, take)
    labels = None
    if micro.has_labels:
        if cells is not None:
            labels = np.take(micro.cell_labels, cells, axis=axis)
        else:
            labels = take(micro.labels(grid=grid))
        labels = labels.astype(np.int32, copy=False)
    return phase.astype(np.uint8, copy=False), level, labels, levels


def _phase_functions(micro, grid: Grid, take) -> dict[int, np.ndarray] | None:
    """The implicit functions of the phases of a structure of several phases on a grid
    (None for those of two phases, or without them)."""
    phases = getattr(micro, "implicit_phases", lambda: [0, 1])()
    if phases == [0, 1] or len(phases) > MAX_PHASES:
        return None
    try:
        return {
            int(k): take(micro.phase_level_set(k, grid=grid)).astype(np.float32) for k in phases
        }
    except NotImplementedError:
        return None


def _resampled(values: np.ndarray, coarse: Grid, fine: Grid, axes, linear: bool):
    """A coarse slice ``[u, v]`` at the pixel centres of a finer grid: the value of the
    coarse cell containing each centre, or linearly interpolated between their centres."""
    out = values
    for k, a in enumerate(axes):
        t = (fine.centers(a) - coarse.origin[a]) / coarse.spacing[a]
        n = coarse.shape[a]
        if not linear:
            idx = np.clip(np.floor(t).astype(np.int64), 0, n - 1)
            out = np.take(out, idx, axis=k)
            continue
        t = np.clip(t - 0.5, 0.0, n - 1.0)
        i0 = np.minimum(np.floor(t).astype(np.int64), max(n - 2, 0))
        i1 = np.minimum(i0 + 1, n - 1)
        w = (t - i0).astype(np.float32)
        shape = [1, 1]
        shape[k] = len(w)
        w = w.reshape(shape)
        out = np.take(out, i0, axis=k) * (1 - w) + np.take(out, i1, axis=k) * w
    return out.astype(values.dtype, copy=False)


def _from_volume(micro, full: Grid, axis: int, index: int):
    """The slice of the voxel image (of a coarser one beyond :data:`MAX_VOLUME` voxels,
    enlarged to the pixels of ``full``)."""
    if full.n_voxels <= MAX_VOLUME:
        return (*_evaluate(micro, full, axis, index)[:3], "volume")
    scale = (MAX_VOLUME / full.n_voxels) ** (1.0 / 3.0)
    coarse = Grid(full.domain, tuple(max(1, math.floor(n * scale + 1e-9)) for n in full.shape))
    position = float(full.centers(axis)[index])
    k = int((position - coarse.origin[axis]) // coarse.spacing[axis])
    k = min(max(k, 0), coarse.shape[axis] - 1)
    phase, level, labels, _ = _evaluate(micro, coarse, axis, k)
    axes = plane_axes(axis)
    phase = _resampled(phase, coarse, full, axes, False)
    if level is not None:
        level = _resampled(level, coarse, full, axes, True)
    if labels is not None:
        labels = _resampled(labels, coarse, full, axes, False)
    return phase, level, labels, "coarse"


def section(micro, axis: int = 2, fraction: float = 0.5, resolution: int = 512) -> Section:
    """The section of ``micro`` normal to ``axis`` at ``fraction`` of the domain, with
    ``resolution`` pixels along the longest edge of the plane: the slice of index
    :func:`plane_index` of ``micro.voxelize(grid=volume_grid(...))`` (and of its level set
    and labels), computed on the plane alone when the structure allows it (see the
    module)."""
    axis, fraction = _axis(axis), _fraction(fraction)
    full = volume_grid(micro.domain, axis, resolution)
    index = plane_index(micro.domain, axis, fraction, resolution)
    levels = None
    if isinstance(micro, _LEAVES):
        grid = _slab(full, axis, index, thickness=float(micro.domain.lengths[axis]))
        phase, level, labels, levels = _evaluate(micro, grid, axis, implicit=True)
        sampling = "plane"
    elif _pointwise(micro):
        on_lattice = isinstance(micro, LatticeField) and tuple(full.shape) == micro.shape
        cells = index if on_lattice else None
        grid = _slab(full, axis, index)
        phase, level, labels, levels = _evaluate(micro, grid, axis, cells=cells, implicit=True)
        sampling = "plane"
    else:
        phase, level, labels, sampling = _from_volume(micro, full, axis, index)
    u, v = plane_axes(axis)
    lo, hi = micro.domain.lower, micro.domain.upper
    level = None if level is None else _display(level)
    coverages = None
    if levels:
        coverages = phase_coverages({k: _display(f) for k, f in levels.items()}, _display(phase))
    return Section(
        axis=axis,
        fraction=fraction,
        phase=_display(phase),
        level_set=level,
        coverage=None if level is None else pixel_coverage(level),
        labels=None if labels is None else _display(labels),
        pixel=float(max(full.spacing[u], full.spacing[v])),
        extent=(float(lo[u]), float(hi[u]), float(lo[v]), float(hi[v])),
        names=(_AXES[u], _AXES[v]),
        position=float(full.centers(axis)[index]),
        index=int(index),
        sampling=sampling,
        coverages=coverages,
    )


def sections(micro, resolution: int = 512) -> list[Section]:
    """The middle sections normal to x, y and z (y-z, x-z, x-y), or the x-y one alone
    for planar structures (:func:`is_planar`)."""
    axes = (2,) if is_planar(micro, resolution) else (0, 1, 2)
    return [section(micro, a, 0.5, resolution) for a in axes]


# -- images -----------------------------------------------------------------------------------


def _variation(phi: np.ndarray) -> np.ndarray:
    """``|D_u phi| + |D_v phi|``: the larger one-sided differences with the neighbours
    along each axis, summed (``h_u |dphi/du| + h_v |dphi/dv|``)."""
    out = np.zeros(phi.shape, dtype=phi.dtype)
    m = np.empty_like(out)
    for a in range(2):
        if phi.shape[a] < 2:
            continue

        def part(s, a=a):
            return (s, slice(None)) if a == 0 else (slice(None), s)

        d = np.abs(np.diff(phi, axis=a))
        m[part(slice(None, -1))] = d  # with the next pixel
        m[part(slice(-1, None))] = 0.0
        later = m[part(slice(1, None))]
        np.maximum(later, d, out=later)  # and with the previous one
        out += m
    return out


def pixel_coverage(phi) -> np.ndarray:
    """Fraction of phase 1 in each pixel from the level set at the pixel centres:
    ``clip(1/2 - phi / (|D_u phi| + |D_v phi|), 0, 1)`` (see the module), 1 or 0 by the
    sign of ``phi`` where it does not vary."""
    p = np.asarray(phi, dtype=np.float32)
    if p.ndim != 2:
        raise ValueError("pixel_coverage: a 2D level set")
    with np.errstate(divide="ignore", invalid="ignore"):  # infinite placeholders: by sign
        width = _variation(p)
        out = 0.5 - p / width
    flat = ~(width > 0)
    out[flat] = np.where(p[flat] < 0, 1.0, np.where(p[flat] > 0, 0.0, 0.5))
    return np.clip(out, 0.0, 1.0, out=out)


def phase_coverages(levels: dict[int, np.ndarray], phase: np.ndarray) -> dict[int, np.ndarray]:
    """Fraction of each phase in each pixel from the implicit functions ``levels`` of the
    phases at the pixel centres (by phase, each negative in its phase): the
    :func:`pixel_coverage` of each, scaled so that they sum to one; where none of them
    covers the pixel, the ``phase`` of its centre takes it."""
    cover = {int(k): pixel_coverage(f) for k, f in levels.items()}
    total = np.zeros(np.shape(phase), dtype=np.float32)
    for c in cover.values():
        total += c
    covered = total > 0
    out = {}
    for k, c in cover.items():
        share = np.divide(c, total, out=np.zeros_like(c), where=covered)
        share[~covered & (np.asarray(phase) == k)] = 1.0
        out[k] = share
    return out


def _changes(values: np.ndarray) -> np.ndarray:
    """Pixels whose value differs from that of the next pixel along a row or a column."""
    out = np.zeros(values.shape, dtype=bool)
    out[:, :-1] |= values[:, :-1] != values[:, 1:]
    out[:-1, :] |= values[:-1, :] != values[1:, :]
    return out


def render(sec: Section, kind: str = "phase", *, contour: bool = False) -> np.ndarray:
    """RGB image (uint8, rows x columns x 3) of a section.

    ``phase``: the colours of :data:`morphogen.viz.png.PALETTE`, those of phases 0 and 1
    blended by the coverage when the section has a level set (anti-aliased boundaries),
    those of all the phases by their coverages when it has the implicit functions of
    several phases, the plain colours of the phases otherwise; ``labels``: a colour per
    cell or grain (:func:`morphogen.viz.png.label_colors`). ``contour`` draws the zero
    level set as a thin dark line (``1 - |2 coverage - 1|`` of the dark colour; between
    several phases, twice the share of the pixel that is not of its main phase), or
    without a level set the boundaries between the phases or labels shown.
    """
    several = sec.coverages if kind == "phase" else None
    if kind == "phase":
        index = np.minimum(sec.phase, len(PALETTE) - 1)
        if several:
            colours = PALETTE.astype(np.float32)
            rgb = np.zeros((*sec.phase.shape, 3), dtype=np.float32)
            total = np.zeros(sec.phase.shape, dtype=np.float32)
            for k, c in several.items():
                rgb += c[..., None] * colours[min(k, len(PALETTE) - 1)]
                total += c
            np.copyto(rgb, colours[index], where=(total <= 0)[..., None])
        elif sec.coverage is None:
            rgb = PALETTE[index]
        else:
            colours = PALETTE.astype(np.float32)
            rgb = colours[index]
            c = sec.coverage[..., None]
            blend = colours[0] + c * (colours[1] - colours[0])
            np.copyto(rgb, blend, where=(sec.phase <= 1)[..., None])
        shown = sec.phase
    elif kind == "labels":
        if sec.labels is None:
            raise ValueError("render: this section has no labels")
        rgb = label_colors(sec.labels)
        shown = sec.labels
    else:
        raise ValueError(f"render: kind 'phase' or 'labels', got {kind!r}")
    if contour:
        if several:
            main = np.max(np.stack(list(several.values())), axis=0)
            alpha = np.clip(2.0 * (1.0 - main), 0.0, 1.0)
        elif sec.coverage is not None:
            alpha = 1.0 - np.abs(2.0 * sec.coverage - 1.0)
        else:
            alpha = _changes(shown).astype(np.float32)
        rgb = rgb + alpha[..., None] * (CONTOUR.astype(np.float32) - rgb)
    if rgb.dtype == np.uint8:
        return rgb
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


# -- measures ---------------------------------------------------------------------------------


def _gradient_norm(phi: np.ndarray, spacing, periodic: bool) -> np.ndarray:
    """``|grad phi|`` by centred differences (one-sided at the faces of a non-periodic
    grid)."""
    out = np.zeros(phi.shape)
    for a in range(3):
        h = float(spacing[a])
        if phi.shape[a] < 2:
            continue
        if periodic:
            g = (np.roll(phi, -1, axis=a) - np.roll(phi, 1, axis=a)) / (2.0 * h)
        else:
            g = np.gradient(phi, h, axis=a)
        out += g * g
    return np.sqrt(out)


def _marching_cubes_area(phi: np.ndarray, spacing, periodic: bool) -> float | None:
    """Area of the zero level set by the marching cubes of scikit-image (None without
    it); a periodic grid is closed by a copy of its first cells, so that the cells of the
    surface cover exactly one period."""
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
    except ImportError:
        return None
    volume = np.pad(phi, [(0, 1)] * 3, mode="wrap") if periodic else phi
    if not volume.min() < 0.0 < volume.max():
        return 0.0
    verts, faces, _, _ = marching_cubes(volume, level=0.0, spacing=tuple(float(h) for h in spacing))
    return float(mesh_surface_area(verts, faces))


def level_set_measures(micro, resolution: int = 128) -> dict[str, float]:
    """Volume fraction and specific surface of phase 1 from its level set ``phi``.

    ``phi`` is sampled on the voxel grid of the domain with ``resolution`` cells along its
    longest edge (at the cell centres of a periodic domain, at the nodes from face to face
    otherwise, weighted by the trapezoidal rule), and turned into the distance
    ``d = phi / |grad phi|`` to its zero level set (centred differences). With
    ``eps = h`` (the largest cell edge):

    * ``volume_fraction``: the mean of the smoothed Heaviside ``H(-d)``, ``H(x) =
      (1 + x / eps + sin(pi x / eps) / pi) / 2`` on ``[-eps, eps]``. Its samples sum
      exactly to the volume for plane interfaces (the spectrum of ``H'`` vanishes at the
      multiples of ``1 / h``); curvature adds about ``0.13 eps^2`` times the integral of
      the mean curvature (``0.39 (eps / R)^2`` of the volume of spheres of radius ``R``),
      more for struts a few cells thick;
    * ``specific_surface``: the area of the zero level set in the domain per unit volume,
      by marching cubes (scikit-image, the ``surface`` extra), else by the coarea formula
      ``mean(delta(phi') |grad phi'|)``, ``delta(x) = (1 + cos(pi x / eps)) / (2 eps)`` on
      ``[-eps, eps]``, ``phi'`` being ``phi`` over its median gradient at the interface (an
      estimate, low for gaps and struts thinner than ``2 eps``).

    Only for structures with a level set (ValueError otherwise).
    """
    name = getattr(getattr(micro, "model", None), "name", None) or type(micro).__name__
    if not has_level_set(micro):
        raise ValueError(
            f"{name}: no level-set function (its phase 1 is not described by one, e.g. "
            "tessellations, dead leaves, several phases); measure its voxel image instead"
        )
    resolution = _resolution(resolution)
    domain = micro.domain
    base = Grid.for_domain(domain, shape=resolution)
    h = base.spacing
    periodic = domain.periodic
    if periodic:
        grid = base
    else:  # the nodes of the cells, from face to face
        n = np.array(base.shape) + 1
        nodes = Domain(size=n * h, origin=domain.lower - 0.5 * h, periodic=False, unit=domain.unit)
        grid = Grid(nodes, tuple(int(k) for k in n))
    try:
        phi = micro.level_set(grid=grid).data.astype(np.float64)
    except NotImplementedError as exc:
        raise ValueError(f"{name}: no level-set function ({exc})") from None
    far = float(domain.diagonal)  # placeholders of 'no interface' (inf) made finite
    phi = np.clip(np.nan_to_num(phi, nan=far, posinf=far, neginf=-far), -far, far)
    eps = float(h.max())
    norm = _gradient_norm(phi, grid.spacing, periodic)
    d = np.where(phi < 0, -np.inf, np.where(phi > 0, np.inf, 0.0))
    np.divide(phi, norm, out=d, where=norm > 0)
    t = np.clip(-d / eps, -1.0, 1.0)
    heaviside = 0.5 * (1.0 + t + np.sin(np.pi * t) / np.pi)
    if periodic:
        weights = np.ones(grid.shape)
    else:  # trapezoidal rule over the nodes
        w = [np.ones(k) for k in grid.shape]
        for a, k in enumerate(grid.shape):
            if k > 1:
                w[a][[0, -1]] = 0.5
        weights = w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]
    total = float(np.sum(weights))
    fraction = float(np.sum(weights * heaviside) / total)
    area = _marching_cubes_area(phi, grid.spacing, periodic)
    if area is None:  # coarea formula, phi scaled to a distance by its gradient at the surface
        near = np.abs(d) < eps
        scale = float(np.median(norm[near])) if near.any() else 1.0
        x = phi / scale
        band = np.abs(x) < eps
        delta = np.zeros(phi.shape)
        delta[band] = (1.0 + np.cos(np.pi * x[band] / eps)) / (2.0 * eps) * norm[band] / scale
        surface = float(np.sum(weights * delta) / total)
    else:
        surface = area / domain.volume
    return {"volume_fraction": fraction, "specific_surface": surface}
