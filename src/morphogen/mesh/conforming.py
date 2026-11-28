"""Conforming tetrahedral meshes of implicit morphologies.

The level-set function of a microstructure (negative inside phase 1) is sampled at the
nodes of a background mesh of the box: a Kuhn triangulation of a regular grid (six
tetrahedra per cube, so that the faces of the box are exactly represented). MMG
(``mmg3d -ls``, LGPL; installed by the ``pymmg`` package or found on the ``PATH``)
discretizes the zero level set into a conforming interface between the two phases and
remeshes the result: elements are adapted to the curvature of the interface within a
Hausdorff distance, graded, and of good quality, while the faces of the box stay plane.

For periodic microstructures, opposite faces get matching nodes, so that periodic
boundary conditions are exact: the level set takes the same values on opposite faces of
the background mesh, MMG first discretizes it without remeshing (the cut faces match),
then remeshes the volume and the interface while keeping the faces of the box
unchanged (along all the axes, or some of them). Elsewhere, the faces are remeshed
freely.

Mirrored meshes (:func:`mirrored_mesh`): a microstructure symmetric about the mid-planes
of its box is meshed in the irreducible part of the box and reflected across them, so
that the mesh has the symmetries exactly (an unstructured mesh of the whole box has
none), which the analyses by group theory need (:mod:`morphogen.homogenization.
bifurcation`); a periodic microstructure symmetric about its mid-planes is symmetric
about the faces of its box too, so that the reflected faces match: periodic conditions
apply. With the diagonal planes of a square or cubic box among its mirrors (the groups
of the square, 4/mmm, and of the cube, m-3m), the irreducible part is the chamber of the
group (a triangular prism, a tetrahedron), a union of tetrahedra of the background mesh,
meshed and mapped by every operation of the group. :class:`ImplicitBody` makes a
microstructure of a level-set function.

Sharp edges. The function of a union of spheres or segments has a fold where two of
them meet, and its surface a sharp edge there (a groove), which a function sampled on a
grid rounds. The structures made of such primitives give the smooth *pieces* of their
phases (:mod:`morphogen.models.pieces`: primitives that do not overlap); the box is then
cut by the surface of each piece in turn, the function of the piece taken at the
vertices themselves, and the lines where two cuts meet — the sharp edges — are edges of
the mesh, given to MMG as ridges and kept through the remeshing (``sharp_edges``;
:attr:`~morphogen.mesh.tetra.TetMesh.sharp`).

Element sizes: ``size`` bounds the edge length (``-hmax``); the interface is refined to
follow its curvature within the Hausdorff distance ``hausdorff``. Thin features must be
resolved by the background grid (``background``, default ``size / 2``) and by a few
elements: a sheet of thickness ``t`` needs ``size`` of the order of ``t`` with 10-node
elements, smaller with 4-node elements.
"""

from __future__ import annotations

import importlib.util
import itertools
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from morphogen.domain import Domain, Grid
from morphogen.io import medit
from morphogen.mesh.tetra import TET_FACES, TetMesh, tet_volumes

log = logging.getLogger("morphogen")

#: MMG references of the subdomains created by the level-set discretization
REF_INSIDE, REF_OUTSIDE, REF_INTERFACE = 3, 2, 10
#: The references of the tetrahedra of a mesh of several phases: this plus their phase
REF_PHASE = 20
#: Level-set values closer to zero than this fraction of an edge of the background mesh
#: (times the gradient) are moved away from zero: the interface then never cuts an edge
#: close to a node, which avoids slivers and halves the remeshing time; the interface
#: moves by at most this fraction of the background spacing.
SNAP = 0.2
#: The references of the tetrahedra of the pieces of a mesh cut piece by piece (the
#: meshes that follow sharp edges): this plus the number of their piece
REF_PIECE = 1000
#: The references of the edges of MMG's mesh along the sharp edges of an interface:
#: this plus a number of the two pieces that meet there
REF_SHARP = 1000
#: :data:`SNAP` for the cuts of the pieces: the lines where two cuts meet are the sharp
#: edges, which a smaller share keeps within the Hausdorff distance of the remeshing
SNAP_SHARP = 0.05


def mmg_executable(path: str | None = None, program: str = "mmg3d") -> str:
    """Path of an MMG executable (``mmg3d``, ``mmg2d`` or ``mmgs``, the surfaces): explicit,
    ``$MORPHOGEN_MMG3D`` (``$MORPHOGEN_MMG2D``, ``$MORPHOGEN_MMGS``), the ``pymmg``
    package, or ``mmg3d_O3`` / ``mmg3d`` on the ``PATH``."""
    if program not in ("mmg3d", "mmg2d", "mmgs"):
        raise ValueError("program: 'mmg3d', 'mmg2d' or 'mmgs'")
    explicit = path or os.environ.get(f"MORPHOGEN_{program.upper()}")
    if explicit:
        if Path(explicit).is_file() and os.access(explicit, os.X_OK):
            return str(explicit)
        raise FileNotFoundError(f"{program} executable not found: {explicit}")
    candidates = []
    spec = importlib.util.find_spec(program)  # the pymmg package
    if spec is not None and spec.submodule_search_locations:
        folder = Path(next(iter(spec.submodule_search_locations)))
        names = (
            [f"{program}.exe", f"{program}_O3.exe"]
            if platform.system() == "Windows"
            else [f"{program}_O3", program]
        )
        for name in names:
            exe = folder / name
            if exe.is_file() and not os.access(exe, os.X_OK):
                try:  # a wheel unpacked without its permission bits
                    exe.chmod(exe.stat().st_mode | 0o111)
                except OSError:
                    pass
            candidates.append(str(exe))
    candidates += [shutil.which(f"{program}_O3"), shutil.which(program)]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise ImportError(
        f"conforming meshes need the MMG mesher ({program}): pip install pymmg "
        f"(or install {program} and set MORPHOGEN_{program.upper()})"
    )


def has_mmg() -> bool:
    """Whether the MMG mesher is available."""
    try:
        mmg_executable()
    except (ImportError, FileNotFoundError):
        return False
    return True


def background_mesh(shape, lower, upper):
    """Triangulation of the box in six tetrahedra per cube: nodes, positive tetrahedra,
    boundary triangles and their references (1-6 for the faces x0, x1, y0, y1, z0, z1).

    Each cube is split along a main diagonal (Kuhn triangulation), reflected along the
    axes where the cube index is odd: the triangulation stays conforming, its opposite
    faces match, and it is symmetric with respect to the planes of the grid (a single
    diagonal direction would bias the discretized interfaces and make the apparent
    stiffness of cubic structures slightly anisotropic).
    """
    n = np.asarray(shape, dtype=np.int64)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    g = n + 1
    axes = [np.linspace(lower[d], upper[d], g[d]) for d in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    nodes = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    for d in range(3):  # exact faces
        nodes[:, d] = np.where(np.isclose(nodes[:, d], lower[d]), lower[d], nodes[:, d])
        nodes[:, d] = np.where(np.isclose(nodes[:, d], upper[d]), upper[d], nodes[:, d])
    cubes = np.stack(np.meshgrid(*(np.arange(k) for k in n), indexing="ij"), axis=-1).reshape(-1, 3)
    odd = cubes % 2 == 1
    strides = np.array([g[1] * g[2], g[2], 1])
    tets = []
    for perm in itertools.permutations(range(3)):
        corner = np.zeros(3, dtype=np.int64)
        path = [corner.copy()]
        for axis in perm:
            corner[axis] += 1
            path.append(corner.copy())
        tets.append(np.stack([(cubes + np.where(odd, 1 - p, p)) @ strides for p in path], axis=1))
    tets = np.concatenate(tets)
    flip = tet_volumes(nodes, tets) < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    faces = np.concatenate(
        [tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]]
    )
    tri, refs = [], []
    for axis in range(3):
        for side, value in ((0, lower[axis]), (1, upper[axis])):
            on = np.all(nodes[faces][:, :, axis] == value, axis=1)
            tri.append(faces[on])
            refs.append(np.full(int(on.sum()), 1 + 2 * axis + side))
    return nodes, tets, np.concatenate(tri), np.concatenate(refs)


def node_level_set(
    micro, spacing: float, phase: int | None = None
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Level-set values at the nodes of a regular grid covering the box exactly: those of
    the phase 1, or of the implicit function of ``phase``
    (:meth:`~morphogen.models.base.Microstructure.phase_level_set`)."""
    domain: Domain = micro.domain
    n = np.maximum(1, np.ceil(domain.lengths / spacing - 1e-9)).astype(int)
    h = domain.lengths / n
    sampling = Domain(
        size=tuple(domain.lengths + h), origin=tuple(domain.lower - h / 2), periodic=False
    )
    grid = Grid(sampling, tuple(int(k) + 1 for k in n))
    try:
        if phase is None:
            field = micro.level_set(grid=grid)
        else:
            field = micro.phase_level_set(phase, grid=grid)
    except NotImplementedError:
        raise ValueError(
            f"{type(micro).__name__}: no level-set function, so no conforming mesh; "
            "use the FFT solver on voxel images instead"
        ) from None
    return np.asarray(field.data, dtype=np.float64), tuple(int(k) for k in n)


def background_interpolation(values: np.ndarray, lower, upper, points) -> np.ndarray:
    """The values at ``points`` of the function that is linear in each tetrahedron of the
    background mesh (:func:`background_mesh`) and takes ``values`` at its nodes (an array
    over the nodes of the grid): in the cube of a point, mirrored along the axes where its
    index is odd, the tetrahedron is that of the order of its coordinates."""
    values = np.asarray(values, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    n = np.array(values.shape) - 1
    u = (np.asarray(points, dtype=float).reshape(-1, 3) - lower) / (upper - lower) * n
    cube = np.clip(np.floor(u).astype(np.int64), 0, n - 1)
    odd = cube % 2 == 1
    xi = np.clip(u - cube, 0.0, 1.0)
    xi = np.where(odd, 1.0 - xi, xi)
    order = np.argsort(-xi, axis=1, kind="stable")  # the largest coordinate first
    rows = np.arange(len(u))
    corner = np.zeros((len(u), 3), dtype=np.int64)

    def at(c):
        node = cube + np.where(odd, 1 - c, c)
        return values[node[:, 0], node[:, 1], node[:, 2]]

    out = at(corner)
    previous = out.copy()
    for k in range(3):
        axis = order[:, k]
        corner[rows, axis] = 1
        current = at(corner)
        out += xi[rows, axis] * (current - previous)
        previous = current
    return out


def _snap_level_set(values: np.ndarray, fraction: float) -> np.ndarray:
    """Move values close to zero away from it, so that the zero level set never cuts
    an edge of the background mesh very close to a node (which would create slivers).

    The threshold at a node is ``fraction`` times the largest variation of the level set
    along the grid edges at that node (``|grad phi| h`` for a smooth function, and still
    positive at the bottom of thin valleys, where the gradient vanishes).
    """
    variation = np.zeros_like(values)
    for axis in range(values.ndim):
        d = np.abs(np.diff(values, axis=axis))
        lo = [slice(None)] * values.ndim
        hi = [slice(None)] * values.ndim
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        variation[tuple(lo)] = np.maximum(variation[tuple(lo)], d)
        variation[tuple(hi)] = np.maximum(variation[tuple(hi)], d)
    threshold = fraction * variation
    near = np.abs(values) < threshold
    out = values.copy()
    out[near] = np.where(values[near] < 0, -threshold[near], threshold[near])
    return out


def _snap_on_mesh(values: np.ndarray, tets: np.ndarray, fraction) -> np.ndarray:
    """:func:`_snap_level_set` at the vertices of a mesh: the values closer to zero than
    ``fraction`` (one value, or one per vertex) times their largest variation along the
    edges at the vertex are moved away from it, so that the zero level set never cuts an
    edge close to a vertex (those of a previous cut among them: the lines where three
    phases meet)."""
    values = np.asarray(values, dtype=float)
    variation = np.zeros_like(values)
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        d = np.abs(values[tets[:, a]] - values[tets[:, b]])
        np.maximum.at(variation, tets[:, a], d)
        np.maximum.at(variation, tets[:, b], d)
    threshold = fraction * variation
    near = np.abs(values) < threshold
    out = values.copy()
    out[near] = np.where(values[near] < 0, -threshold[near], threshold[near])
    return out


def _matched_vertices(values: np.ndarray, vertices: np.ndarray, box, axes) -> np.ndarray:
    """The values of the vertices of the upper faces of the matching axes taken from the
    vertices that face them on the lower ones (a mesh whose faces match)."""
    from scipy.spatial import cKDTree

    lower, upper = np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)
    tol = 1e-9 * float((upper - lower).max())
    out = np.asarray(values, dtype=float).copy()
    for d in range(3):
        if not axes[d]:
            continue
        low = np.flatnonzero(np.abs(vertices[:, d] - lower[d]) <= tol)
        high = np.flatnonzero(np.abs(vertices[:, d] - upper[d]) <= tol)
        if not (len(low) and len(high)):
            continue
        others = [a for a in range(3) if a != d]
        dist, index = cKDTree(vertices[low][:, others]).query(vertices[high][:, others])
        found = dist <= 1e-6 * float((upper - lower).max())
        out[high[found]] = out[low[index[found]]]
    return out


def _run_mmg(exe, mesh_in, sol_in, mesh_out, args, timeout):
    cmd = [exe, str(mesh_in)]
    if sol_in is not None:
        cmd += ["-sol", str(sol_in)]
    cmd += ["-out", str(mesh_out), *args]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0 or not Path(mesh_out).exists():
        tail = (res.stdout + res.stderr)[-2000:]
        raise RuntimeError(f"{Path(exe).name} failed (exit code {res.returncode}):\n{tail}")
    return medit.read_mesh(mesh_out)


def _second_order(mesh: TetMesh, micro, curved: bool | None) -> TetMesh:
    """The 10-node elements of a 4-node mesh: curved onto the implicit surfaces of the
    phases (:func:`morphogen.mesh.curved.curved_mesh`), or with their mid-edge nodes at
    the middles of the straight edges (``curved`` false)."""
    if curved or curved is None:
        from morphogen.mesh.curved import curved_mesh

        return curved_mesh(mesh, micro)
    return mesh.quadratic()


def _columnar_mesh(micro, *, size, min_size, gradation, phases, order, periodic, min_feature,
                   executable, workdir, timeout) -> TetMesh:  # fmt: skip
    """The columnar grains of a tessellation whose germs lie in one plane: the mesh of its
    exact polygons in the plane (:func:`morphogen.mesh.triangles.tessellation_mesh`)
    extruded through the box in layers of about the size of the elements, each prism cut
    into three tetrahedra; its opposite faces match, those along ``z`` too."""
    from morphogen.mesh.triangles import tessellation_mesh

    axes = _periodic_axes(micro.domain.periodic if periodic is None else periodic)
    if axes[0] != axes[1]:
        raise ValueError("periodic: the columnar grains of the plane are periodic along x and "
                         "y together, or not")  # fmt: skip
    plane = tessellation_mesh(micro, size=size, min_size=min_size, gradation=gradation,
                              phases=phases, periodic=bool(axes[0]), min_feature=min_feature,
                              executable=executable, workdir=workdir, timeout=timeout)  # fmt: skip
    layers = max(1, round(float(micro.domain.lengths[2]) / size))
    mesh = plane.extruded(layers=layers)
    mesh.metadata.update({"mesher": "columnar grains (exact polygons of the plane, mmg2d), "
                                    "extruded", "size": size})  # fmt: skip
    return mesh.quadratic() if order == 2 else mesh


def _map_mesh(micro, *, size, order, phases, periodic) -> TetMesh:
    """A map of grains (an orientation map): its mesh of the plane by its pixels
    (:func:`morphogen.mesh.triangles.lattice_mesh`) extruded through the box in layers of
    about the size of the elements, each element the grain of its pixel."""
    from morphogen.mesh.triangles import lattice_mesh

    axes = _periodic_axes(micro.domain.periodic if periodic is None else periodic)
    plane = lattice_mesh(micro, phases=phases, periodic=bool(axes[0] and axes[1]))
    layers = max(1, round(float(micro.domain.lengths[2]) / size))
    mesh = plane.extruded(layers=layers)
    mesh.metadata.update({"mesher": "map of grains (two triangles per pixel), extruded"})
    return mesh.quadratic() if order == 2 else mesh


def conforming_mesh(
    micro,
    *,
    size: float,
    min_size: float | None = None,
    hausdorff: float | None = None,
    gradation: float = 2.0,
    background: float | None = None,
    phases: str = "all",
    order: int = 1,
    curved: bool | None = None,
    periodic: bool | None = None,
    remove_islands: float | None = 1e-5,
    sharp_edges: bool | None = None,
    grains: bool | None = None,
    min_feature: float | None = None,
    cells: str = "whole",
    method: str = "auto",
    executable: str | None = None,
    workdir=None,
    timeout: float | None = None,
) -> TetMesh:
    """Tetrahedral mesh of a microstructure, conforming to the interface of its phases.

    By default from its exact geometry, without a background grid, when the
    microstructure knows it (``method``; :mod:`morphogen.mesh.exact`). Voronoi and Laguerre
    tessellations are meshed grain by grain from their cells
    (:func:`morphogen.mesh.polycrystal.polycrystal_mesh`): every grain boundary is
    conforming, and each element knows its grain (``mesh.cell``) and its crystal
    orientation (``mesh.orientations``). The columnar grains of a tessellation whose germs
    lie in one plane (the seeds of the plane) are the exact polygons of its mesh of the
    plane (:func:`morphogen.mesh.triangles.tessellation_mesh`) extruded through the box,
    in layers of about ``size``; a map of grains (an orientation map) its pixels likewise,
    two triangles each, each element the grain of its pixel.

    Parameters
    ----------
    size
        Largest element size (``-hmax``).
    min_size, hausdorff
        Smallest element size (default ``size / 10``) and Hausdorff distance between the
        meshed and the discrete interfaces (default ``size / 20``).
    gradation
        Maximal ratio of the sizes of neighbouring elements (``-hgrad``).
    background
        Spacing of the grid sampling the level set (default ``size / 2``): it bounds the
        geometric accuracy of the interface. Give at least three samples across the
        thinnest feature (sheet, strut, fibre): an under-resolved level set biases the
        volume fraction and the apparent properties (by a few percent for two samples).
    phases
        ``"all"`` meshes both phases (composites); ``"solid"`` meshes phase 1 only
        (porous media: the pores carry no stress), ``"matrix"`` phase 0 only (the pores
        are phase 1, e.g. spherical pores in a matrix). Structures of more than two
        phases, or of other phases than 0 and 1 (coated inclusions, inclusions in a
        polycrystal, cells of several phases, osteons...), are meshed conforming to the
        interfaces of all their phases, each cut by its implicit function
        (:meth:`~morphogen.models.base.Microstructure.phase_level_sets_at` on the exact
        route, :meth:`~morphogen.models.base.Microstructure.phase_level_set` on the grid) in
        the part of the box that the previous ones left: ``"all"``, or the phases to keep
        (a list, of any structure).
    order
        1 (4-node tetrahedra) or 2 (10-node tetrahedra).
    curved
        10-node elements: put the nodes of the interfaces, corners and mid-edge nodes, on
        the implicit surfaces of the phases, so that the elements along the interfaces
        have curved edges (isoparametric elements,
        :func:`morphogen.mesh.curved.curved_mesh`) and the phases their volumes: by
        default; false keeps the flat faces that MMG makes, with the mid-edge nodes at the
        middles of their edges. The grain boundaries of tessellations are plane.
    periodic
        Opposite faces with matching nodes, for periodic boundary conditions (default:
        when the microstructure is periodic), or three booleans: along those axes only.
        The level set is cut on the background mesh, whose faces match, then MMG remeshes
        the inside and the interface while the faces of the box along those axes stay
        unchanged.
    remove_islands
        Relative volume below which MMG removes isolated components (``-rmc``;
        non-periodic meshes only).
    sharp_edges
        Follow the sharp edges of the phases with edges of the mesh: the grooves where
        the spheres or the segments of a union overlap, those of their coatings and of
        their bores. The structure gives the smooth pieces of its phases
        (:meth:`~morphogen.models.base.Microstructure.implicit_pieces`), the box is cut
        by the surface of each in turn, and the lines where two cuts meet are kept as
        ridges through the remeshing (``mesh.sharp``: these edges; the curved elements
        then put their nodes on both surfaces). By default, when the structure gives
        pieces that meet; false cuts the box by the function of each phase, which
        rounds its sharp edges over an element. The overlaps shallower than the
        ``hausdorff`` distance (exact route) or half the ``background`` spacing (grid) are
        not followed. The cells and convex pieces of the exact route keep the edges of
        their facets.
    grains
        Mesh the grains of a tessellation (default: for Voronoi and Laguerre
        tessellations without a grain-boundary phase); ``False`` meshes only the
        interface between its phases (the grains and their boundary phase).
    min_feature
        Grains: features of the tessellation smaller than this length are removed
        before meshing (default ``size / 4``).
    cells
        Grains of a periodic tessellation: ``"whole"`` (default) meshes the whole grains,
        periodized at the design (their union fills the volume of the box, their boundary
        is made of translated pairs of faces); ``"box"`` cuts them by the faces of the box.
    method
        ``"exact"``: the exact geometry, without a grid
        (:func:`morphogen.mesh.exact.exact_mesh`: the cells of tessellations, exact
        convex pieces, or the adaptive mesh of the exact level set, refined to
        ``hausdorff`` and ``min_size``; ``background`` unused); ``"grid"``: the level set
        sampled on a background grid of spacing ``background``; ``"auto"`` (default):
        ``"exact"`` when the microstructure has an exact route
        (:func:`morphogen.mesh.exact.exact_route`: every structure that gives the
        functions of its phases at any point, of two phases or more, the phases kept by a
        list and the sharp edges of smooth pieces included; not transparent dead leaves,
        the coatings of structures that are not unions of spheres and segments, nor cells
        carrying crystal orientations but tessellations), ``"grid"`` otherwise, and should
        the exact mesh fail (MMG) or its periodic faces not match. The 10-node elements of
        a level set are those of its 4-node mesh on either route, curved by
        :func:`morphogen.mesh.curved.curved_mesh`.
    """
    if method not in ("auto", "exact", "grid"):
        raise ValueError(
            "method: 'auto' (exact when possible), 'exact' (no background grid) or 'grid' "
            "(level set on a background grid)"
        )
    listed = not isinstance(phases, str)
    if listed:
        phases = tuple(int(k) for k in phases)
    elif phases not in ("all", "solid", "matrix"):
        raise ValueError(
            "phases: 'all', 'solid' (phase 1), 'matrix' (phase 0) or the phases to keep"
        )
    from morphogen.mesh.triangles import lattice_map
    from morphogen.models.tessellation import Tessellation

    if lattice_map(micro) and grains is not False:  # its grains: the map by its pixels
        return _map_mesh(micro, size=float(size), order=order, phases=phases, periodic=periodic)
    tessellation = isinstance(micro, Tessellation)
    if grains is None:
        grains = tessellation and not micro.boundary_thickness > 0
    if grains and tessellation and micro.boundary_thickness > 0:
        raise ValueError("grains: the grain-boundary phase is meshed through its level set")
    if grains and phases == "matrix":
        raise ValueError("phases: 'matrix' meshes phase 0 of a two-phase structure, not grains")
    if cells not in ("whole", "box"):
        raise ValueError("cells: 'whole' or 'box'")
    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    if curved and order == 1:
        raise ValueError("curved: the edges of 10-node elements (order 2)")
    size = float(size)
    if not size > 0:
        raise ValueError("size: must be positive")
    from morphogen.mesh.triangles import planar_tessellation

    if grains and planar_tessellation(micro):
        return _columnar_mesh(micro, size=size, min_size=min_size, gradation=gradation,
                              phases=phases, order=order, periodic=periodic,
                              min_feature=min_feature, executable=executable, workdir=workdir,
                              timeout=timeout)  # fmt: skip
    if method != "grid":
        from morphogen.mesh.exact import exact_mesh, exact_route

        route = exact_route(micro, grains)
        if grains and route != "cells":  # the grains of a structure that is no tessellation
            if method == "exact":
                raise ValueError(
                    "grains: the exact meshes by grains are those of tessellations without "
                    "a boundary phase"
                )
            route = ""
        if route or method == "exact":
            options = {}
            if route == "cells":
                options = {"cells": cells, "min_feature": min_feature}
            # the 10-node elements of a level set: those of its 4-node mesh, curved as on
            # the grid (the same mesh at both orders)
            curve = route == "level_set" and order == 2
            try:
                mesh = exact_mesh(micro, size=size, min_size=min_size, hausdorff=hausdorff,
                                  gradation=gradation, phases=phases,
                                  order=1 if curve else order, periodic=periodic,
                                  grains=grains, sharp_edges=sharp_edges,
                                  executable=executable, workdir=workdir, timeout=timeout,
                                  **options)  # fmt: skip
            except (RuntimeError, subprocess.TimeoutExpired) as error:  # MMG failed
                if method == "exact":
                    raise
                log.warning("conforming mesh: the exact mesh failed (%s); on a grid instead",
                            str(error).splitlines()[0])  # fmt: skip
            else:
                axes = _periodic_axes(micro.domain.periodic if periodic is None else periodic)
                if method == "exact" or not all(axes) or mesh.matching_faces():
                    return _second_order(mesh, micro, curved) if curve else mesh
                log.warning("conforming mesh: the faces of the exact mesh do not match; on a "
                            "grid instead")  # fmt: skip
    if grains:
        from morphogen.mesh.polycrystal import polycrystal_mesh

        return polycrystal_mesh(
            micro,
            size=size,
            min_size=min_size,
            hausdorff=hausdorff,
            gradation=gradation,
            min_feature=min_feature,
            phases=phases,
            order=order,
            periodic=periodic,
            cells=cells,
            executable=executable,
            workdir=workdir,
            timeout=timeout,
        )
    domain: Domain = micro.domain
    axes = _periodic_axes(domain.periodic if periodic is None else periodic)
    periodic = all(axes)
    exe = mmg_executable(executable)
    t0 = time.perf_counter()
    spacing = float(background) if background is not None else size / 2.0
    present = list(micro.implicit_phases()) if hasattr(micro, "implicit_phases") else [0, 1]
    several = present != [0, 1]
    pieces = None
    if sharp_edges is None or sharp_edges:
        found = getattr(micro, "implicit_pieces", lambda depth: None)(0.5 * spacing)
        if found is None and sharp_edges:
            raise ValueError(
                f"sharp_edges: {type(micro).__name__} does not give the smooth pieces of its "
                "phases (the unions of spheres and segments and their coatings do)"
            )
        if found is not None:
            counts: dict[int, int] = {}
            for piece in found[0]:
                counts[piece.phase] = counts.get(piece.phase, 0) + 1
            # by default, only where two pieces of a phase can meet
            if sharp_edges or max(counts.values(), default=0) > 1:
                pieces, rest = found
    if listed and not several:
        if phases not in ((1,), (0,), (0, 1)):
            raise ValueError("phases: this structure has the phases 0 and 1")
        phases = {(1,): "solid", (0,): "matrix"}.get(phases, "all")
    if several:
        if phases in ("solid", "matrix"):
            raise ValueError(
                f"phases: 'all' or the phases to keep among {present} (a structure of "
                "several phases)"
            )
        unknown = sorted(set(phases) - set(present)) if listed else []
        if unknown:
            raise ValueError(f"phases: {unknown} are not phases of this structure ({present})")
    data = None
    sizes = {"size": size, "min_size": min_size, "hausdorff": hausdorff, "gradation": gradation}
    shape = tuple(int(k) for k in np.maximum(1, np.ceil(domain.lengths / spacing - 1e-9)))
    h = domain.lengths / np.array(shape)
    nodes, tets, tri, tri_refs = background_mesh(shape, domain.lower, domain.upper)
    if pieces is not None:
        # the box cut by the surface of each piece in turn: the sharp edges are followed
        t_bg = time.perf_counter() - t0
        t1 = time.perf_counter()
        try:
            data, sharp, sharp_pieces = _mmg_pieces(
                exe,
                nodes,
                tets,
                tri,
                tri_refs,
                pieces,
                rest,
                (domain.lower, domain.upper),
                axes,
                _on_faces(axes, domain.lower, domain.upper),
                4.0 * float(h.max()),
                **sizes,
                workdir=workdir,
                timeout=timeout,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            if sharp_edges:
                raise
            log.warning(
                "conforming mesh: the sharp edges could not be followed (%s); they are "
                "rounded instead",
                str(error).splitlines()[0],
            )
            pieces = None
    if data is None and several:
        # each phase but one is cut in what the previous ones left: the largest index last.
        # A phase is where its function is the smallest of all: the same set where the
        # functions are resolved (one of them negative), and still a partition where they
        # are not (thin layers, the lines where three phases meet)
        sampled = {k: node_level_set(micro, spacing, k)[0] for k in present}
        fields = []
        for k in present[:-1]:
            others = np.minimum.reduce([sampled[q] for q in present if q != k])
            fields.append((k, _matched(_snap_level_set(sampled[k] - others, SNAP), axes)))
        t_bg = time.perf_counter() - t0
        t1 = time.perf_counter()
        data = _mmg_phases(
            exe,
            nodes,
            tets,
            tri,
            tri_refs,
            fields,
            present[-1],
            (domain.lower, domain.upper),
            axes,
            _on_faces(axes, domain.lower, domain.upper),
            **sizes,
            workdir=workdir,
            timeout=timeout,
        )
    elif data is None:
        values = _matched(_snap_level_set(node_level_set(micro, spacing)[0], SNAP), axes)
        t_bg = time.perf_counter() - t0
        t1 = time.perf_counter()
        data = _mmg_level_set(
            exe,
            nodes,
            tets,
            tri,
            tri_refs,
            values.ravel(),
            _matching_refs(axes),
            _on_faces(axes, domain.lower, domain.upper),
            **sizes,
            phases=phases,
            remove_islands=remove_islands,
            workdir=workdir,
            timeout=timeout,
        )
    t_mmg = time.perf_counter() - t1
    xyz = data["Vertices"]
    tol = 1e-9 * float(domain.lengths.max())
    for d in range(3):  # nodes of the faces exactly on the faces
        xyz[:, d] = np.where(np.abs(xyz[:, d] - domain.lower[d]) <= tol, domain.lower[d], xyz[:, d])
        xyz[:, d] = np.where(np.abs(xyz[:, d] - domain.upper[d]) <= tol, domain.upper[d], xyz[:, d])
    elements = data["Tetrahedra"]
    refs = data["Tetrahedra_refs"]
    if several or pieces is not None:
        phase = refs - REF_PHASE
        named = isinstance(phases, str)  # 'all', or 'solid' and 'matrix' of two phases
        wanted = {"solid": (1,), "matrix": (0,)}.get(phases) if named else phases
        if wanted is not None:  # the phases kept
            kept = np.isin(phase, wanted)
            elements, phase = elements[kept], phase[kept]
    else:
        phase = np.where(refs == REF_INSIDE, 1, 0)
    flip = tet_volumes(xyz, elements) < 0
    elements[flip] = elements[flip][:, [0, 2, 1, 3]]
    used, inverse = np.unique(elements, return_inverse=True)
    metadata = {
        "mesher": "mmg3d -ls",
        "size": size,
        "background_spacing": [float(x) for x in h],
        "background_elements": len(tets),
        "phases": phases if isinstance(phases, str) else list(phases),
        "matching_faces": [d for d in range(3) if axes[d]],
        "time_background_s": round(t_bg, 3),
        "time_mmg_s": round(t_mmg, 3),
    }
    followed = {}
    if pieces is not None:  # the sharp edges, between the nodes of the mesh
        numbers = np.full(len(xyz), -1, dtype=np.int64)
        numbers[used] = np.arange(len(used))
        there = (numbers[sharp] >= 0).all(axis=1)
        followed = {"sharp": numbers[sharp[there]], "sharp_pieces": sharp_pieces[there]}
        metadata["sharp_edges"] = {
            "pieces": len(pieces),
            "edges": int(there.sum()),
            "depth": 0.5 * spacing,
        }
    mesh = TetMesh(
        nodes=xyz[used],
        elements=inverse.reshape(elements.shape),
        phase=phase,
        lower=domain.lower,
        upper=domain.upper,
        periodic=periodic,
        metadata=metadata,
        **followed,
    )
    q = mesh.quality()
    if q.size and q.min() < 1e-6:
        log.warning(
            "conforming mesh: %d degenerate elements; the background grid (spacing %.3g) "
            "may not resolve thin features of the microstructure: decrease 'background'",
            int((q < 1e-6).sum()),
            float(h.max()),
        )
    from morphogen.mesh.exact import labelled

    mesh = labelled(micro, mesh)  # the labels and orientations of the elements, if any
    if order == 2:
        mesh = _second_order(mesh, micro, curved)
    log.info(
        "conforming mesh: %d elements, %d nodes (MMG %.1f s)",
        mesh.n_elements,
        mesh.n_nodes,
        t_mmg,
    )
    return mesh


def _matching_refs(axes) -> list[int]:
    """The references of the faces of the background mesh along the matching axes."""
    return [1 + 2 * d + side for d in range(3) if axes[d] for side in (0, 1)]


def _on_faces(axes, lower, upper):
    """A mask of the vertices on the faces of the box along the matching axes."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    tol = 1e-9 * float((upper - lower).max())

    def mask(vtx):
        on = np.zeros(len(vtx), dtype=bool)
        for d in range(3):
            if axes[d]:
                on |= np.abs(vtx[:, d] - lower[d]) <= tol
                on |= np.abs(vtx[:, d] - upper[d]) <= tol
        return on

    return mask


def _mmg_level_set(
    exe,
    nodes,
    tets,
    tri,
    tri_refs,
    values,
    required,
    required_vertices,
    *,
    size,
    min_size=None,
    hausdorff=None,
    gradation=2.0,
    phases="all",
    remove_islands=1e-5,
    workdir=None,
    timeout=None,
):
    """MMG's discretization of the zero level set of ``values`` (at the nodes) on a
    background mesh, and its remeshing: in one pass, or, when the boundary triangles of
    the references ``required`` must stay (matching faces), a cut without remeshing, then a
    remeshing that keeps them and the vertices of ``required_vertices`` (a mask of the
    vertices of the cut mesh, a function of their positions). Returns MMG's mesh."""
    sizing = [
        "-hmax",
        f"{size:.10g}",
        "-hmin",
        f"{(min_size if min_size is not None else size / 10):.10g}",
        "-hausd",
        f"{(hausdorff if hausdorff is not None else size / 20):.10g}",
        "-hgrad",
        f"{float(gradation):.10g}",
        "-v",
        "-1",
    ]
    keep = {"solid": ["-nsd", str(REF_INSIDE)], "matrix": ["-nsd", str(REF_OUTSIDE)]}.get(
        phases, []
    )
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        medit.write_mesh(folder / "bg.mesh", nodes, tets, tri, tri_refs=tri_refs)
        medit.write_sol(folder / "bg.sol", np.asarray(values, dtype=float).ravel())
        if not required:
            args = ["-ls", "0", *sizing, *keep]
            if remove_islands:
                args += ["-rmc", f"{float(remove_islands):.10g}"]
            return _run_mmg(
                exe, folder / "bg.mesh", folder / "bg.sol", folder / "out.mesh", args, timeout
            )
        cut = _run_mmg(
            exe,
            folder / "bg.mesh",
            folder / "bg.sol",
            folder / "cut.mesh",
            ["-ls", "0", "-nosurf", "-noinsert", "-noswap", "-nomove", "-v", "-1"],
            timeout,
        )
        refs = cut["Triangles_refs"]
        vtx = cut["Vertices"]
        medit.write_mesh(
            folder / "cut2.mesh",
            vtx,
            cut["Tetrahedra"],
            cut["Triangles"],
            tet_refs=cut["Tetrahedra_refs"],
            tri_refs=refs,
            required_triangles=np.flatnonzero(np.isin(refs, required)),
            required_vertices=np.flatnonzero(required_vertices(vtx)),
        )
        return _run_mmg(
            exe, folder / "cut2.mesh", None, folder / "out.mesh", [*sizing, *keep], timeout
        )


def _matched(values: np.ndarray, axes) -> np.ndarray:
    """Identical values on the opposite faces of the matching axes: the cut faces match."""
    if axes[0]:
        values[-1, :, :] = values[0, :, :]
    if axes[1]:
        values[:, -1, :] = values[:, 0, :]
    if axes[2]:
        values[:, :, -1] = values[:, :, 0]
    return values


def _mmg_phases(
    exe,
    nodes,
    tets,
    tri,
    tri_refs,
    fields,
    last: int,
    box,
    matching,
    required_vertices,
    *,
    size,
    min_size=None,
    hausdorff=None,
    gradation=2.0,
    workdir=None,
    timeout=None,
):
    """MMG's mesh of a box of several phases: the tetrahedra of the background mesh, all
    of the phase ``last`` at first, are cut by the zero level set of each of ``fields``
    (a phase and its values at the nodes of the grid) where they are not of a phase cut
    before (MMG's level sets of several materials: the references that a level set splits
    and those it leaves), without remeshing; the level set of a cut is the function linear
    in the tetrahedra of the background mesh, taken at the vertices that the previous cuts
    made (:func:`background_interpolation`). The mesh is then remeshed, its interfaces
    kept, with the triangles of the faces along the ``matching`` axes and the vertices of
    ``required_vertices`` unchanged (matching faces). The references of the tetrahedra of
    MMG's mesh are :data:`REF_PHASE` plus their phase."""
    required = _matching_refs(matching)
    sizing = [
        "-hmax",
        f"{size:.10g}",
        "-hmin",
        f"{(min_size if min_size is not None else size / 10):.10g}",
        "-hausd",
        f"{(hausdorff if hausdorff is not None else size / 20):.10g}",
        "-hgrad",
        f"{float(gradation):.10g}",
        "-v",
        "-1",
    ]
    rest = REF_PHASE + int(last)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        mesh = {
            "Vertices": nodes,
            "Tetrahedra": tets,
            "Tetrahedra_refs": np.full(len(tets), rest),
            "Triangles": tri,
            "Triangles_refs": tri_refs,
        }
        done: list[int] = []
        for step, (phase, values) in enumerate(fields):
            name = folder / f"cut{step}.mesh"
            if step:  # the vertices of the previous cut: numbered alike on opposite faces
                mesh = _in_order(mesh, box)
            medit.write_mesh(
                name,
                mesh["Vertices"],
                mesh["Tetrahedra"],
                mesh["Triangles"],
                tet_refs=mesh["Tetrahedra_refs"],
                tri_refs=mesh["Triangles_refs"],
            )
            if step == 0:
                at = values.ravel()
            else:
                at = background_interpolation(values, box[0], box[1], mesh["Vertices"])
                at = _snap_on_mesh(at, mesh["Tetrahedra"], SNAP)
                at = _matched_vertices(at, mesh["Vertices"], box, matching)
            medit.write_sol(folder / f"cut{step}.sol", at)
            lines = ["LSReferences", str(1 + len(done)), f"{rest} {REF_PHASE + phase} {rest}"]
            lines += [f"{REF_PHASE + k} nosplit" for k in done]
            (folder / f"cut{step}.mmg3d").write_text("\n".join(lines) + "\n", encoding="ascii")
            mesh = _run_mmg(
                exe,
                name,
                folder / f"cut{step}.sol",
                folder / f"cut{step + 1}_in.mesh",
                ["-ls", "0", "-nosurf", "-noinsert", "-noswap", "-nomove", "-v", "-1"],
                timeout,
            )
            done.append(int(phase))
        refs = mesh["Triangles_refs"]
        vtx = mesh["Vertices"]
        keep = {}
        if required:
            keep = {
                "required_triangles": np.flatnonzero(np.isin(refs, required)),
                "required_vertices": np.flatnonzero(required_vertices(vtx)),
            }
        medit.write_mesh(
            folder / "cut.mesh",
            vtx,
            mesh["Tetrahedra"],
            mesh["Triangles"],
            tet_refs=mesh["Tetrahedra_refs"],
            tri_refs=refs,
            **keep,
        )
        return _run_mmg(exe, folder / "cut.mesh", None, folder / "out.mesh", sizing, timeout)


def _mmg_pieces(
    exe,
    nodes,
    tets,
    tri,
    tri_refs,
    pieces,
    last: int,
    box,
    matching,
    required_vertices,
    band: float,
    *,
    size,
    min_size=None,
    hausdorff=None,
    gradation=2.0,
    workdir=None,
    timeout=None,
):
    """MMG's mesh of a box whose phases are given by their smooth pieces
    (:mod:`morphogen.models.pieces`), which follows the sharp edges of their surfaces.

    The tetrahedra of the background mesh, all of the phase ``last`` at first, are cut
    by the surface of each piece in turn where no piece cut before holds them, without
    remeshing; the function of a piece is taken at the vertices that the previous cuts
    made, exactly. The line where the surfaces of two pieces of the same phase meet — a
    sharp edge of the phase — is then made of edges of the mesh. These edges are given
    to MMG as ridges, with the edges of the box and the lines where an interface meets a
    face of the box, and the detection of ridges by their angle is turned off (the
    facets that the cuts leave on the smooth surfaces are no edges); the pieces of a
    phase are merged, and the mesh is remeshed as in :func:`_mmg_phases`.

    Returns MMG's mesh (the references of its tetrahedra: :data:`REF_PHASE` plus their
    phase), the sharp edges among its edges (pairs of vertices) and the two pieces
    that meet along each."""
    required = _matching_refs(matching)
    sizing = [
        "-hmax",
        f"{size:.10g}",
        "-hmin",
        f"{(min_size if min_size is not None else size / 10):.10g}",
        "-hausd",
        f"{(hausdorff if hausdorff is not None else size / 20):.10g}",
        "-hgrad",
        f"{float(gradation):.10g}",
        "-nr",
        "-v",
        "-1",
    ]
    rest = REF_PHASE + int(last)
    count = len(pieces)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        mesh = {
            "Vertices": nodes,
            "Tetrahedra": tets,
            "Tetrahedra_refs": np.full(len(tets), rest),
            "Triangles": tri,
            "Triangles_refs": tri_refs,
        }
        done: list[int] = []
        for step, piece in enumerate(pieces):
            if done:  # the vertices of the previous cuts: numbered alike on opposite faces
                mesh = _in_order(mesh, box)
            at = piece.at(mesh["Vertices"], band)
            # on the faces of the box, whose triangles the remeshing may have to keep, the
            # share of the other meshes
            on_box = _on_faces((True,) * 3, box[0], box[1])(mesh["Vertices"])
            at = _snap_on_mesh(at, mesh["Tetrahedra"], np.where(on_box, SNAP, SNAP_SHARP))
            at = _matched_vertices(at, mesh["Vertices"], box, matching)
            if at.min() >= 0.0:  # a piece that is not in the box
                continue
            name = folder / f"cut{step}.mesh"
            medit.write_mesh(
                name,
                mesh["Vertices"],
                mesh["Tetrahedra"],
                mesh["Triangles"],
                tet_refs=mesh["Tetrahedra_refs"],
                tri_refs=mesh["Triangles_refs"],
            )
            medit.write_sol(folder / f"cut{step}.sol", at)
            lines = ["LSReferences", str(1 + len(done)), f"{rest} {REF_PIECE + step} {rest}"]
            lines += [f"{REF_PIECE + k} nosplit" for k in done]
            (folder / f"cut{step}.mmg3d").write_text("\n".join(lines) + "\n", encoding="ascii")
            mesh = _run_mmg(
                exe,
                name,
                folder / f"cut{step}.sol",
                folder / f"cut{step + 1}_in.mesh",
                ["-ls", "0", "-nosurf", "-noinsert", "-noswap", "-nomove", "-v", "-1"],
                timeout,
            )
            done.append(step)
        vtx, elements, refs = mesh["Vertices"], mesh["Tetrahedra"], mesh["Tetrahedra_refs"]
        # the piece and the phase of each tetrahedron (what is left: after all the pieces)
        cut = np.where(refs >= REF_PIECE, refs - REF_PIECE, count)
        phase_of = np.array([int(piece.phase) for piece in pieces] + [int(last)])
        phase = phase_of[cut]
        # the faces between two phases: their piece is the first of the two that was cut
        faces = elements[:, TET_FACES].reshape(-1, 3)
        owner = np.repeat(np.arange(len(elements)), 4)
        key = np.sort(faces, axis=1)
        order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
        key, faces, owner = key[order], faces[order], owner[order]
        twin = np.flatnonzero(np.all(key[1:] == key[:-1], axis=1))
        a, b = owner[twin], owner[twin + 1]
        between = phase[a] != phase[b]
        a, b = a[between], b[between]
        inner = faces[twin[between]]
        low, high = np.minimum(phase[a], phase[b]), np.maximum(phase[a], phase[b])
        label = np.minimum(cut[a], cut[b])
        surface = 7 + (low * (int(phase_of.max()) + 1) + high) * (count + 1) + label
        # the faces of the box, as MMG kept them through the cuts
        on_box = np.isin(mesh["Triangles_refs"], np.arange(1, 7))
        triangles = np.concatenate([mesh["Triangles"][on_box], inner])
        tri_surface = np.concatenate([mesh["Triangles_refs"][on_box], surface])
        tri_out = np.concatenate(
            [mesh["Triangles_refs"][on_box], np.full(len(inner), REF_INTERFACE)]
        )
        # the edges between two surfaces: ridges; between two pieces of an interface: sharp
        edges = np.sort(
            np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]]),
            axis=1,
        )
        of = np.tile(tri_surface, 3)
        order = np.lexsort((of, edges[:, 1], edges[:, 0]))
        edges, of = edges[order], of[order]
        same = np.all(edges[1:] == edges[:-1], axis=1)
        change = np.flatnonzero(same & (of[1:] != of[:-1]))
        ridge, inverse = np.unique(edges[change], axis=0, return_inverse=True)
        ridge_refs = np.zeros(len(ridge), dtype=np.int64)
        s1, s2 = of[change], of[change + 1]  # two surfaces around the edge, s1 < s2
        both = (s1 >= 7) & ((s1 - 7) // (count + 1) == (s2 - 7) // (count + 1))
        code = ((s1 - 7) % (count + 1)) * (count + 1) + (s2 - 7) % (count + 1)
        ridge_refs[inverse.ravel()[both]] = REF_SHARP + code[both]
        keep = {}
        if required:
            keep = {
                "required_triangles": np.flatnonzero(np.isin(tri_out, required)),
                "required_vertices": np.flatnonzero(required_vertices(vtx)),
            }
        medit.write_mesh(
            folder / "cut.mesh",
            vtx,
            elements,
            triangles,
            tet_refs=REF_PHASE + phase,
            tri_refs=tri_out,
            edges=ridge,
            edge_refs=ridge_refs,
            ridges=np.arange(len(ridge)),
            **keep,
        )
        out = _run_mmg(exe, folder / "cut.mesh", None, folder / "out.mesh", sizing, timeout)
    edge_refs = out.get("Edges_refs", np.zeros(0, dtype=np.int64))
    along = edge_refs >= REF_SHARP
    codes = edge_refs[along] - REF_SHARP
    pairs = np.stack([codes // (count + 1), codes % (count + 1)], axis=1)
    return out, out.get("Edges", np.zeros((0, 2), dtype=np.int64))[along], pairs


def _in_order(mesh: dict, box) -> dict:
    """MMG's mesh with its vertices numbered in the order of their coordinates. MMG
    chooses the diagonal along which it splits a face that a level set cuts from the
    numbers of its vertices: the vertices that a first cut made are numbered as they were
    made, not alike on opposite faces, and a second cut could then split two matching
    faces along different diagonals (the same nodes, another edge: mid-edge nodes of
    10-node elements without an image). In the order of their coordinates, the vertices
    of a face and their images on the opposite face follow each other alike."""
    vertices = mesh["Vertices"]
    scale = 1e-9 * float(np.max(np.asarray(box[1]) - np.asarray(box[0])))
    keys = np.round((vertices - np.asarray(box[0])) / scale).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    out = dict(mesh)
    out["Vertices"] = vertices[order]
    out["Tetrahedra"] = rank[mesh["Tetrahedra"]]
    out["Triangles"] = rank[mesh["Triangles"]]
    return out


def _periodic_axes(periodic) -> tuple[bool, bool, bool]:
    """The axes along which the faces match: all of them, none, or three booleans."""
    if np.ndim(periodic) == 0:
        return (bool(periodic),) * 3
    axes = tuple(bool(p) for p in periodic)
    if len(axes) != 3:
        raise ValueError("periodic: a boolean or three booleans")
    return axes


class ImplicitBody:
    """A microstructure given by its level-set function in ``domain``: ``function(points)``
    of points (rows) negative inside phase 1, positive outside (for instance the distance
    to the nearest hole less its radius, for the solid of a porous plate)."""

    def __init__(self, domain: Domain, function):
        self.domain, self.function = domain, function

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None):
        from morphogen.image import VoxelImage

        if grid is None:
            grid = Grid.for_domain(self.domain, shape=shape, voxel_size=voxel_size)
        values = np.asarray(self.function(grid.center_points()), dtype=float)
        return VoxelImage(values.reshape(grid.shape), spacing=tuple(grid.spacing),
                          origin=tuple(grid.origin))  # fmt: skip


class _Part:
    """A microstructure seen in a part of its box (its own level set)."""

    def __init__(self, micro, domain: Domain):
        self.micro, self.domain = micro, domain

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None):
        return self.micro.level_set(grid=grid)


def _merge(nodes: np.ndarray, elements: np.ndarray, tol: float):
    """The nodes closer than ``tol`` merged (the first of each group kept), the elements
    renumbered."""
    from scipy.spatial import cKDTree

    pairs = cKDTree(nodes).query_pairs(tol, output_type="ndarray")
    parent = np.arange(len(nodes))
    for a, b in pairs[np.argsort(pairs.max(axis=1))]:
        ra, rb = a, b
        while parent[ra] != ra:
            ra = parent[ra]
        while parent[rb] != rb:
            rb = parent[rb]
        parent[max(ra, rb)] = min(ra, rb)
    for i in range(len(parent)):  # flatten
        parent[i] = parent[parent[i]]
    used, inverse = np.unique(parent, return_inverse=True)
    return nodes[used], inverse[elements]


def _components(pairs) -> list[list[int]]:
    """The axes joined by diagonal mirrors (sorted groups, singletons included)."""
    parent = list(range(3))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for a in range(3):
        groups.setdefault(find(a), []).append(a)
    return [sorted(g) for g in groups.values()]


def point_group(mirrors, diagonals=()) -> list[np.ndarray]:
    """The operations (3 x 3 matrices about the centre of the box) of the group generated by
    the reflections across the mid-planes of the axes ``mirrors`` and across the diagonal
    planes ``x_i - c_i = x_j - c_j`` of the pairs ``diagonals`` (axes among the mirrors):
    signed permutations, 16 for the square of 4/mmm (mirrors x, y, z and the diagonal of x
    and y), 48 for the cube of m-3m (two diagonals)."""
    mirrors = sorted({int(k) for k in np.atleast_1d(mirrors)})
    components = _components(diagonals)
    out = []
    for parts in itertools.product(*(itertools.permutations(c) for c in components)):
        perm = np.arange(3)
        for c, image in zip(components, parts, strict=True):
            perm[c] = image
        for signs in itertools.product((1.0, -1.0), repeat=len(mirrors)):
            m = np.zeros((3, 3))
            m[np.arange(3), perm] = 1.0
            m[mirrors] *= np.asarray(signs)[:, None]
            out.append(m)
    return out


def _chamber_background(shape, lower, upper, pairs):
    """The background mesh of the box restricted to the chamber ``u_a >= u_b`` of each pair
    (``u = x - lower``; the diagonal planes through the corner are unions of faces of the
    Kuhn triangulation, reflected or not, when the grid has equal steps along the axes of
    a pair): its nodes (and their indices in the box's), tetrahedra, boundary triangles and
    their references (1-6 the faces of the box, 7 + k the plane of the pair k)."""
    nodes, tets, _, _ = background_mesh(shape, lower, upper)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    u = nodes - lower
    centroids = u[tets].mean(axis=1)
    inside = np.ones(len(tets), dtype=bool)
    for a, b in pairs:
        inside &= centroids[:, a] > centroids[:, b]
    tets = tets[inside]
    faces = np.concatenate(
        [tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]]
    )
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True,
                                 return_counts=True)  # fmt: skip
    boundary = faces[first[counts == 1]]
    points = u[boundary]
    tol = 1e-9 * float((upper - lower).max())
    refs = np.zeros(len(boundary), dtype=np.int64)
    for d in range(3):
        refs[np.all(np.abs(points[:, :, d]) <= tol, axis=1)] = 1 + 2 * d
        refs[np.all(np.abs(points[:, :, d] - (upper[d] - lower[d])) <= tol, axis=1)] = 2 + 2 * d
    for k, (a, b) in enumerate(pairs):
        refs[np.all(np.abs(points[:, :, a] - points[:, :, b]) <= tol, axis=1)] = 7 + k
    if np.any(refs == 0):
        raise RuntimeError("chamber: boundary faces off its planes (unequal grid steps?)")
    used = np.unique(tets)
    index = np.full(len(nodes), -1, dtype=np.int64)
    index[used] = np.arange(len(used))
    return nodes[used], used, index[tets], index[boundary], refs


def _project_planes(xyz, triangles, refs, lower, upper, pairs, rounds: int = 60):
    """The vertices of the boundary triangles moved exactly onto the planes of their
    references (alternating projections where planes meet)."""
    xyz = xyz.copy()
    groups = []
    for d in range(3):
        for side, value in ((0, lower[d]), (1, upper[d])):
            v = np.unique(triangles[refs == 1 + 2 * d + side])
            if v.size:
                groups.append(("box", d, value, v))
    for k, (a, b) in enumerate(pairs):
        v = np.unique(triangles[refs == 7 + k])
        if v.size:
            groups.append(("diagonal", a, b, v))
    for _ in range(rounds):
        for kind, a, b, v in groups:
            if kind == "box":
                xyz[v, a] = b
            else:
                mean = 0.5 * ((xyz[v, a] - lower[a]) + (xyz[v, b] - lower[b]))
                xyz[v, a] = lower[a] + mean
                xyz[v, b] = lower[b] + mean
    return xyz


def _chamber_mesh(micro, lower, upper, pairs, matching, *, size, phases, background=None,
                  executable=None, **mmg) -> TetMesh:  # fmt: skip
    """A conforming mesh of the chamber of the box ``[lower, upper]`` (see
    :func:`_chamber_background`), its faces along the ``matching`` axes matching."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    part = _Part(micro, Domain(size=tuple(upper - lower), origin=tuple(lower), periodic=False))
    spacing = float(background) if background is not None else size / 2.0
    values, shape = node_level_set(part, spacing)
    values = _snap_level_set(values, SNAP)
    if matching[0]:
        values[-1, :, :] = values[0, :, :]
    if matching[1]:
        values[:, -1, :] = values[:, 0, :]
    if matching[2]:
        values[:, :, -1] = values[:, :, 0]
    nodes, used, tets, tri, tri_refs = _chamber_background(shape, lower, upper, pairs)
    data = _mmg_level_set(mmg_executable(executable), nodes, tets, tri, tri_refs,
                          values.ravel()[used], _matching_refs(matching),
                          _on_faces(matching, lower, upper), size=size, phases=phases,
                          **mmg)  # fmt: skip
    xyz = _project_planes(data["Vertices"], data["Triangles"], data["Triangles_refs"], lower,
                          upper, pairs)  # fmt: skip
    elements = data["Tetrahedra"]
    flip = tet_volumes(xyz, elements) < 0
    elements[flip] = elements[flip][:, [0, 2, 1, 3]]
    used_nodes, inverse = np.unique(elements, return_inverse=True)
    phase = np.where(data["Tetrahedra_refs"] == REF_INSIDE, 1, 0)
    return TetMesh(xyz[used_nodes], inverse.reshape(elements.shape), phase, lower, upper,
                   periodic=False, metadata={"mesher": "mmg3d -ls, chamber", "size": size,
                                             "phases": phases})  # fmt: skip


def mirrored_mesh(
    micro,
    *,
    size: float,
    mirrors=(0, 1, 2),
    diagonals=(),
    cells=(1, 1, 1),
    order: int = 1,
    phases: str = "all",
    check: bool = True,
    **options,
) -> TetMesh:
    """A conforming tetrahedral mesh of a microstructure symmetric about the mid-planes of
    its box along the axes ``mirrors`` (see the module): the irreducible part of the box,
    halved along each of them, meshed by MMG (:func:`conforming_mesh`: ``size``,
    ``phases`` and the other ``options``; matching faces along the other axes), then
    reflected across the mid-planes, the nodes of each plane shared by its two halves. The
    mesh has these mirrors exactly, and its opposite faces match along every axis (the
    reflected ones, a periodic microstructure being symmetric about the faces of its box
    as well). ``diagonals``: pairs of mirrored axes of equal edges whose diagonal planes
    through the centre, ``x_i - c_i = x_j - c_j``, are mirrors too (``((0, 1),)`` for the
    square of 4/mmm, ``((0, 1), (1, 2))`` for the cube of m-3m): the chamber of the group
    (:func:`point_group`) is meshed and mapped by each of its operations. ``cells``: copies
    of the box side by side (a cell of several primitive cells, whose translations are
    exact symmetries of the mesh). ``check``: the level set is sampled to verify its
    symmetry. Elements of ``order`` 1 or 2."""
    domain: Domain = micro.domain
    lower, upper = np.asarray(domain.lower, float), np.asarray(domain.upper, float)
    mirrors = sorted({int(k) for k in np.atleast_1d(mirrors)})
    if any(k not in (0, 1, 2) for k in mirrors):
        raise ValueError("mirrors: axes 0, 1 or 2")
    counts = np.asarray(cells, dtype=int)
    if counts.shape != (3,) or np.any(counts < 1):
        raise ValueError("cells: three positive counts")
    mid = 0.5 * (lower + upper)
    span = float((upper - lower).max())
    diagonals = [tuple(sorted((int(a), int(b)))) for a, b in diagonals]
    for a, b in diagonals:
        if a == b or a not in mirrors or b not in mirrors:
            raise ValueError("diagonals: pairs of two mirrored axes")
        if abs((upper - lower)[a] - (upper - lower)[b]) > 1e-9 * span:
            raise ValueError("diagonals: axes of equal edges")
    if check and mirrors:
        n = 16
        sample = Grid(
            Domain(size=tuple(upper - lower), origin=tuple(lower), periodic=False), (n,) * 3
        )
        values = np.asarray(micro.level_set(grid=sample).data, dtype=float)
        scale = float(np.abs(values).max()) or 1.0
        for k in mirrors:
            if np.abs(values - np.flip(values, axis=k)).max() > 1e-6 * scale:
                raise ValueError(
                    f"mirrored_mesh: the microstructure is not symmetric along axis {k}"
                )
        for a, b in diagonals:
            if np.abs(values - np.swapaxes(values, a, b)).max() > 1e-6 * scale:
                raise ValueError(
                    f"mirrored_mesh: the microstructure is not symmetric about the diagonal "
                    f"of axes {a} and {b}"
                )
    if diagonals:
        return _mirrored_chambers(micro, lower, upper, mirrors, diagonals, counts, size=size,
                                  order=order, phases=phases, **options)  # fmt: skip
    top = upper.copy()
    top[mirrors] = mid[mirrors]
    part = _Part(micro, Domain(size=tuple(top - lower), origin=tuple(lower), periodic=False))
    matching = tuple(k not in mirrors for k in range(3))
    mesh = conforming_mesh(part, size=size, phases=phases, order=1, periodic=matching, **options)
    nodes, tets, phase = mesh.nodes.copy(), mesh.elements.copy(), mesh.phase.copy()
    for k in mirrors:
        on = np.abs(nodes[:, k] - mid[k]) <= 1e-9 * span
        nodes[on, k] = mid[k]
        index = np.empty(len(nodes), dtype=np.int64)
        index[on] = np.flatnonzero(on)
        index[~on] = len(nodes) + np.arange(int((~on).sum()))
        image = nodes[~on].copy()
        image[:, k] = 2.0 * mid[k] - image[:, k]
        nodes = np.vstack([nodes, image])
        tets = np.vstack([tets, index[tets][:, [0, 2, 1, 3]]])  # reflected, turned back
        phase = np.concatenate([phase, phase])
    length = upper - lower
    if np.any(counts > 1):
        blocks, shifted = [], []
        for offset, shift in enumerate(itertools.product(*(range(c) for c in counts))):
            blocks.append(nodes + np.asarray(shift) * length)
            shifted.append(tets + offset * len(nodes))
        nodes, tets = _merge(np.vstack(blocks), np.vstack(shifted), 1e-9 * span)
        phase = np.tile(phase, int(np.prod(counts)))
    out = TetMesh(nodes, tets, phase, lower, lower + counts * length, periodic=True,
                  metadata={**mesh.metadata, "mesher": "mmg3d -ls, mirrored",
                            "mirrors": mirrors, "cells": counts.tolist(),
                            "matching_faces": [0, 1, 2]})  # fmt: skip
    return out.quadratic() if order == 2 else out


def _mirrored_chambers(micro, lower, upper, mirrors, diagonals, counts, *, size, order,
                       phases, **options) -> TetMesh:  # fmt: skip
    """The mirrored mesh of a group with diagonal mirrors (see :func:`mirrored_mesh`)."""
    mid = 0.5 * (lower + upper)
    span = float((upper - lower).max())
    top = upper.copy()
    top[mirrors] = mid[mirrors]
    chamber_pairs = [(c[i], c[i + 1]) for c in _components(diagonals) for i in range(len(c) - 1)]
    matching = tuple(k not in mirrors for k in range(3))
    chamber = _chamber_mesh(micro, lower, top, chamber_pairs, matching, size=size,
                            phases=phases, **options)  # fmt: skip
    group = point_group(mirrors, diagonals)
    blocks, shifted, phases_out = [], [], []
    for k, m in enumerate(group):
        blocks.append(mid + (chamber.nodes - mid) @ m.T)
        tets = chamber.elements if np.linalg.det(m) > 0 else chamber.elements[:, [0, 2, 1, 3]]
        shifted.append(tets + k * chamber.n_nodes)
        phases_out.append(chamber.phase)
    nodes, tets = _merge(np.vstack(blocks), np.vstack(shifted), 1e-9 * span)
    phase = np.concatenate(phases_out)
    length = upper - lower
    if np.any(counts > 1):
        blocks, shifted = [], []
        for offset, shift in enumerate(itertools.product(*(range(c) for c in counts))):
            blocks.append(nodes + np.asarray(shift) * length)
            shifted.append(tets + offset * len(nodes))
        nodes, tets = _merge(np.vstack(blocks), np.vstack(shifted), 1e-9 * span)
        phase = np.tile(phase, int(np.prod(counts)))
    out = TetMesh(nodes, tets, phase, lower, lower + counts * length, periodic=True,
                  metadata={**chamber.metadata, "mesher": "mmg3d -ls, mirrored chambers",
                            "mirrors": mirrors, "diagonals": [list(d) for d in diagonals],
                            "group_order": len(group), "cells": counts.tolist(),
                            "matching_faces": [0, 1, 2]})  # fmt: skip
    return out.quadratic() if order == 2 else out
