"""morphoGen and Microgen (3MAH, Mechanics of microstructures; Microgen 2, GPL-3.0): the
shapes and meshes of each as the other's.

Microgen describes its shapes (TPMS, spinodoids, primitives) by implicit fields negative
inside, as morphoGen describes its phases (:meth:`Microstructure.phase_level_sets_at`):

* :func:`to_microgen`: a phase of a realization of morphoGen as a shape of Microgen
  (``microgen.from_field``), its field the point-wise implicit function of the phase
  (a signed distance where the structure knows it), for the tools of Microgen: its
  surface and volume meshes (``generate_surface_mesh``, ``generate_volume_mesh`` with
  MMG), its infills and smooth booleans with its own shapes;
* :func:`from_microgen`: a shape of Microgen as a microstructure of morphoGen (voxels,
  level sets, conforming meshes, homogenization, RVE studies); the model ``microgen``
  (:mod:`morphogen.models.microgen`) does it from a configuration, which can be saved;
* :func:`to_pyvista` and :func:`from_pyvista`: tetrahedral meshes between morphoGen
  (:class:`~morphogen.mesh.tetra.TetMesh`) and PyVista (``UnstructuredGrid``, the meshes
  of Microgen), with the phase of each element.
"""

from __future__ import annotations

import numpy as np

from morphogen.domain import Domain
from morphogen.models.microgen import MicrogenStructure, _microgen


def _bounds(domain: Domain) -> tuple:
    lo, hi = domain.lower, domain.upper
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))


def to_microgen(micro, phase: int = 1, *, reach: float | None = None, shape: int = 64):
    """A phase of a realization as a shape of Microgen: its field the implicit function of
    the phase at any point (negative inside, exact within ``reach`` of the interface, a
    tenth of the edge of the domain by default), its bounds those of the domain; a
    periodic structure gives the field of the points brought back into its box. The
    structures without point-wise functions give that of a grid of ``shape`` voxels
    along the longest edge, interpolated (trilinear, periodic when they are)."""
    mg = _microgen()
    domain = micro.domain
    phase = int(phase)
    if phase not in micro.implicit_phases():
        raise ValueError(
            f"phase {phase}: the phases of this structure are {micro.implicit_phases()}"
        )
    reach = 0.1 * float(np.max(domain.lengths)) if reach is None else float(reach)
    probe = micro.implicit_at(phase, domain.center[None], reach)
    if probe is not None:

        def field(x, y, z):
            p = np.stack(np.broadcast_arrays(x, y, z), axis=-1).astype(float)
            flat = p.reshape(-1, 3)
            if domain.periodic:
                flat = domain.wrap(flat)
            return np.asarray(micro.implicit_at(phase, flat, reach)).reshape(p.shape[:-1])

    else:
        field = _interpolated(micro.phase_level_set(phase, shape))
    return mg.from_field(field, bounds=_bounds(domain))


def _interpolated(image):
    """A function of the points interpolating an image trilinearly (its voxels at their
    centres; periodic images wrap around their faces)."""
    from scipy.ndimage import map_coordinates

    data = np.asarray(image.data, dtype=float)
    origin = np.asarray(image.origin, dtype=float)
    spacing = np.asarray(image.spacing, dtype=float)
    mode = "grid-wrap" if image.periodic else "nearest"

    def field(x, y, z):
        p = np.stack(np.broadcast_arrays(x, y, z), axis=-1).astype(float)
        index = ((p.reshape(-1, 3) - origin) / spacing - 0.5).T
        values = map_coordinates(data, index, order=1, mode=mode)
        return values.reshape(p.shape[:-1])

    return field


def from_microgen(shape, domain: Domain, *, description: dict | None = None):
    """A shape of Microgen as a microstructure of a domain: phase 1 the inside of its
    field (negative), voxelized, meshed and homogenized as any other. ``description``
    (``{"shape", "arguments", "part"}``, as the model ``microgen`` takes them) lets it be
    saved and read back; without one, it is not."""
    return MicrogenStructure(domain, shape, description)


def to_pyvista(mesh):
    """A tetrahedral mesh of morphoGen as a PyVista ``UnstructuredGrid`` (4- or 10-node
    tetrahedra, VTK's node order, morphoGen's), with the phase of each element (and its
    grain) as cell data: the meshes that Microgen's tools take (``BoxMesh``, MMG)."""
    import pyvista as pv

    quadratic = mesh.elements.shape[1] == 10
    kind = pv.CellType.QUADRATIC_TETRA if quadratic else pv.CellType.TETRA
    n = mesh.elements.shape[1]
    cells = np.hstack([np.full((len(mesh.elements), 1), n), mesh.elements]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(mesh.elements), kind, dtype=np.uint8),
                               np.asarray(mesh.nodes, dtype=float))  # fmt: skip
    grid.cell_data["phase"] = np.asarray(mesh.phase)
    if mesh.cell is not None:
        grid.cell_data["grain"] = np.asarray(mesh.cell)
    return grid


def from_pyvista(grid, *, phase: str = "phase", periodic: bool = False):
    """A volume mesh of PyVista (an ``UnstructuredGrid``: Microgen's volume meshes, any
    mesh read by PyVista) as a tetrahedral mesh of morphoGen, for its solvers: its
    tetrahedra (4 or 10 nodes; its hexahedra, wedges and pyramids split into tetrahedra
    first, the other cells left out), the phase of each element from the cell array
    ``phase`` (0 for all without one), the box of its points; ``periodic`` when it is the
    mesh of a periodic cell."""
    import pyvista as pv

    from morphogen.mesh.tetra import TetMesh

    if not isinstance(grid, pv.UnstructuredGrid):
        grid = grid.cast_to_unstructured_grid()
    linear = {pv.CellType.HEXAHEDRON, pv.CellType.WEDGE, pv.CellType.PYRAMID, pv.CellType.VOXEL}
    if linear & set(np.unique(np.asarray(grid.celltypes)).tolist()):
        grid = grid.triangulate()  # the hexahedra, wedges and pyramids split into tetrahedra
    types = np.asarray(grid.celltypes)
    blocks = []
    for kind, n in ((pv.CellType.TETRA, 4), (pv.CellType.QUADRATIC_TETRA, 10)):
        keep = np.flatnonzero(types == kind)
        if len(keep):
            blocks.append((keep, n))
    if len(blocks) != 1:
        raise ValueError("from_pyvista: tetrahedra of one kind (4 or 10 nodes) in the grid")
    keep, n = blocks[0]
    connectivity = np.asarray(grid.cell_connectivity)
    offsets = np.asarray(grid.cell_offsets if hasattr(grid, "cell_offsets") else grid.offset)
    elements = np.stack([connectivity[offsets[k] : offsets[k] + n] for k in keep])
    values = grid.cell_data.get(phase)
    phases = np.zeros(len(keep), np.int64) if values is None else np.asarray(values)[keep]
    nodes = np.asarray(grid.points, dtype=float)
    return TetMesh(nodes, elements, phases, nodes.min(axis=0), nodes.max(axis=0),
                   periodic=bool(periodic))  # fmt: skip
