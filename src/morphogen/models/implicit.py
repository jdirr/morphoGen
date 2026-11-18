"""Implicit functions of the phases of the microstructures: helpers.

Every realization describes each of its phases by an implicit function
(:meth:`~morphogen.models.base.Microstructure.phase_level_set`): negative inside the
phase, positive outside and zero on its boundary, so that its interfaces are those of the
model itself, at any resolution, and not those of a voxel image: conforming meshes
(:func:`morphogen.mesh.conforming.conforming_mesh`), surfaces and sections are cut from
them. The functions are exact where it matters, on the interfaces and around them: signed
distances to unions of spheres and segments, distances to the faces of cells, minima and
maxima of the functions of the components of a combination (the union, intersection and
difference of sets); away from the interfaces they are clipped to a band or only keep
their sign.

One interface gives these functions on grids and at any point:

* :meth:`~morphogen.models.base.Microstructure.implicit_phases`: the phases;
* :meth:`~morphogen.models.base.Microstructure.phase_level_set`: the function of a phase
  at the centres of the voxels of a grid, clipped to a band of four voxels;
* :meth:`~morphogen.models.base.Microstructure.phase_level_sets_at`: the functions of all
  the phases at any points, ``{k: phi_k}``, exact within a ``reach`` of the interfaces
  (the same values at the centres of the voxels): those that the exact meshes cut and
  that the curved elements follow; NotImplementedError for the structures whose
  functions are those of grids only (:func:`point_functions`);
* ``level_set`` and ``level_set_at``: the function of phase 1, on a grid and at points;
  ``implicit_at(phase, points, band)``: one phase of ``phase_level_sets_at``, clipped
  (None without point-wise functions);
* ``lipschitz``: a bound of the slopes of the point-wise functions;
* :meth:`~morphogen.models.base.Microstructure.implicit_pieces`: the smooth pieces of the
  phases of the structures whose functions have folds (:mod:`morphogen.models.pieces`).

The helpers of this module evaluate on the points of a regular grid (the centres of the
voxels of a :class:`~morphogen.domain.Grid`, which may extend beyond the box of a periodic
microstructure: the nodes of a background mesh) the functions of objects that cover one
another in an order: the leaves of a dead-leaves model, the osteons of a cross-section.
"""

from __future__ import annotations

import math

import numpy as np

from morphogen.domain import Grid


def implicit_phases_of(micro) -> list[int]:
    """The phases of a structure (:meth:`~morphogen.models.base.Microstructure.
    implicit_phases`; 0 and 1 for an object that has a level set only)."""
    phases = getattr(micro, "implicit_phases", None)
    return list(phases()) if callable(phases) else [0, 1]


def functions_at(micro, points, reach: float) -> dict[int, np.ndarray]:
    """The functions of all the phases of a structure at points
    (:meth:`~morphogen.models.base.Microstructure.phase_level_sets_at`; ``{1: phi, 0:
    -phi}`` for an object with a ``level_set_at`` only, such as a level set written as a
    function of the points)."""
    sets_at = getattr(micro, "phase_level_sets_at", None)
    if callable(sets_at):
        return sets_at(points, reach)
    phi = np.asarray(micro.level_set_at(points, reach), dtype=float)
    return {0: -phi, 1: phi}


def point_functions(micro) -> bool:
    """Whether a structure gives the functions of its phases at any point
    (:func:`functions_at`), not only at the points of grids."""
    if not callable(getattr(micro, "phase_level_sets_at", None)) and not callable(
        getattr(micro, "level_set_at", None)
    ):
        return False
    try:
        functions_at(micro, np.reshape(np.asarray(micro.domain.lower, dtype=float), (1, 3)), 1.0)
    except NotImplementedError:
        return False
    return True


#: Voxels of the band around the interfaces to which the functions are clipped
BAND = 4.0


def band_of(grid: Grid) -> float:
    """The band of a grid: :data:`BAND` times its largest spacing, along the axes that it
    samples with more than one voxel (a slab one voxel thick, the plane of a section, has
    the band of the volume it is cut from)."""
    spacing = np.asarray(grid.spacing, dtype=float)
    sampled = np.array(grid.shape) > 1
    return BAND * float(spacing[sampled].max() if sampled.any() else spacing.max())


def grid_key(grid: Grid) -> tuple:
    """A key of the points of a grid (for the caches of the functions of all the phases,
    computed together)."""
    return (
        tuple(int(n) for n in grid.shape),
        tuple(float(v) for v in grid.origin),
        tuple(float(v) for v in grid.spacing),
    )


def implicit_functions(
    micro, shape=None, *, voxel_size=None, grid=None
) -> list[tuple[str, object]]:
    """The implicit functions that describe a microstructure, on the voxel centres of a
    grid (as :meth:`~morphogen.models.base.Microstructure.level_set`), with the suffix of
    their files: its level set alone (suffix ``""``) for two phases; for several, the
    function of each phase (``"_phase<k>"``), in the order of the phases."""
    where = {"shape": shape, "voxel_size": voxel_size, "grid": grid}
    phases = micro.implicit_phases()
    if phases == [0, 1]:
        return [("", micro.level_set(**where))]
    return [(f"_phase{k}", micro.phase_level_set(k, **where)) for k in phases]


def lattice_boxes(centre, reach: float, grid: Grid, lengths=None, axes=(0, 1, 2)):
    """The points of a grid within ``reach`` of ``centre`` along each of ``axes`` (a box
    around it), for the centre and, with the ``lengths`` of a periodic box, for each of
    its periodic images that comes that close to the grid: yields the slices of the points
    along the axes and their coordinates from the image of the centre (one array per axis,
    to broadcast)."""
    ranges = []
    for a, c in zip(axes, centre, strict=True):
        h = float(grid.spacing[a])
        x0 = float(grid.origin[a]) + 0.5 * h
        n = int(grid.shape[a])
        images = [0.0]
        if lengths is not None:
            period = float(lengths[a])
            first = math.floor((x0 - reach - c) / period)
            last = math.ceil((x0 + (n - 1) * h + reach - c) / period)
            images = [m * period for m in range(first, last + 1)]
        along = []
        for shift in images:
            lo = max(0, math.ceil((c + shift - reach - x0) / h - 1e-9))
            hi = min(n - 1, math.floor((c + shift + reach - x0) / h + 1e-9))
            if hi >= lo:
                along.append((slice(lo, hi + 1), x0 + np.arange(lo, hi + 1) * h - (c + shift)))
        if not along:
            return
        ranges.append(along)
    if len(ranges) == 2:
        for sa, da in ranges[0]:
            for sb, db in ranges[1]:
                yield (sa, sb), (da[:, None], db[None, :])
        return
    for sa, da in ranges[0]:
        for sb, db in ranges[1]:
            for sc, dc in ranges[2]:
                yield (sa, sb, sc), (da[:, None, None], db[None, :, None], dc[None, None, :])


def ordered_cover(shape, band: float, objects, n_parts: int):
    """The implicit functions of the visible parts of objects that cover one another, the
    first on top. ``objects`` yields, in that order and for each object and each box of
    points near it (:func:`lattice_boxes`), ``(slices, cover, parts)``: the implicit
    function ``cover`` of the object at these points (negative inside it) and, for each
    of the parts of the objects (its phases, or the rings of an osteon), its function
    there, or None where the object has no such part. The visible part of an object is
    what lies in it and in none of those on top: the maximum of its function and of
    minus the smallest function of those on top. Returns the functions of the ``n_parts``
    parts (the minimum over the objects) and the smallest function of all the objects
    (positive where nothing covers), clipped to ``band``."""
    parts = np.full((n_parts, *shape), band, dtype=np.float32)
    top = np.full(shape, band, dtype=np.float32)  # the smallest function of those on top
    for slices, cover, functions in objects:
        above = -top[slices]
        for k, f in enumerate(functions):
            if f is None:
                continue
            view = parts[(k, *slices)]
            np.minimum(view, np.maximum(f, above), out=view)
        view = top[slices]
        np.minimum(view, cover, out=view)
    np.clip(parts, -band, band, out=parts)
    np.clip(top, -band, band, out=top)
    return parts, top
