"""Hexahedral (voxel) finite-element meshes of voxel images."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from morphogen.image import VoxelImage

#: Corner offsets of a hexahedron in VTK / Abaqus / Gmsh order.
HEX_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
)
#: Local nodes of the hexahedron faces lying on each face of the box (outward normals).
HEX_FACES = {
    "x0": (0, 4, 7, 3),
    "x1": (1, 2, 6, 5),
    "y0": (0, 1, 5, 4),
    "y1": (3, 7, 6, 2),
    "z0": (0, 3, 2, 1),
    "z1": (4, 5, 6, 7),
}
FACE_NAMES = tuple(HEX_FACES)


@dataclass
class VoxelMesh:
    """One 8-node hexahedron per selected voxel.

    ``nodes`` are numbered from 0 here; writers shift to 1-based numbering when the
    format requires it. ``node_sets`` holds the nodes on the six faces of the box.
    """

    nodes: np.ndarray
    elements: np.ndarray
    phase: np.ndarray
    label: np.ndarray | None
    node_ijk: np.ndarray
    element_ijk: np.ndarray
    grid_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def volume(self) -> float:
        return float(self.n_elements * np.prod(self.spacing))

    def element_sets(self) -> dict[str, np.ndarray]:
        """Elements of each phase, named ``phase_<p>``."""
        return {f"phase_{p}": np.flatnonzero(self.phase == p) for p in np.unique(self.phase)}

    def periodic_pairs(self, axis: int) -> np.ndarray:
        """Pairs (node on the lower face, node on the upper face) facing each other."""
        n = self.grid_shape[axis]
        lo = np.flatnonzero(self.node_ijk[:, axis] == 0)
        hi = np.flatnonzero(self.node_ijk[:, axis] == n)
        others = [a for a in range(3) if a != axis]
        size = self.grid_shape[others[1]] + 1

        def key(idx):
            ijk = self.node_ijk[idx]
            return ijk[:, others[0]].astype(np.int64) * size + ijk[:, others[1]]

        _, ia, ib = np.intersect1d(key(lo), key(hi), return_indices=True)
        return np.stack([lo[ia], hi[ib]], axis=1)

    def boundary_faces(self, face: str) -> np.ndarray:
        """Quadrilaterals (node indices) of the element faces lying on a box face."""
        axis = "xyz".index(face[0])
        at = 0 if face[1] == "0" else self.grid_shape[axis] - 1
        on = self.element_ijk[:, axis] == at
        return self.elements[on][:, list(HEX_FACES[face])]


def voxel_mesh(image: VoxelImage, phases=None, *, labels: VoxelImage | None = None) -> VoxelMesh:
    """Mesh the voxels of ``image`` whose value is in ``phases`` (default: all voxels).

    ``labels`` (e.g. the cell labels of a tessellation) are attached to the elements.
    """
    data = image.data
    if data.dtype.kind == "f":
        raise ValueError("voxel meshes need a phase or label image, not partial volumes")
    nx, ny, nz = image.shape
    mask = np.ones(data.shape, dtype=bool) if phases is None else np.isin(data, list(phases))
    ijk = np.argwhere(mask)
    if len(ijk) == 0:
        raise ValueError("no voxel to mesh: the selected phases are absent")
    strides = np.array([(ny + 1) * (nz + 1), nz + 1, 1], dtype=np.int64)
    corners = ijk[:, None, :] + HEX_CORNERS[None, :, :]
    full_ids = corners @ strides
    used, elements = np.unique(full_ids, return_inverse=True)
    elements = elements.reshape(full_ids.shape).astype(np.int64)
    node_ijk = np.stack(
        [used // strides[0], (used // strides[1]) % (ny + 1), used % (nz + 1)], axis=1
    )
    nodes = np.asarray(image.origin) + node_ijk * np.asarray(image.spacing)
    node_sets = {}
    for name in FACE_NAMES:
        axis = "xyz".index(name[0])
        target = 0 if name[1] == "0" else image.shape[axis]
        node_sets[name] = np.flatnonzero(node_ijk[:, axis] == target)
    phase = data[mask].astype(np.int64)
    label = labels.data[mask].astype(np.int64) if labels is not None else None
    return VoxelMesh(
        nodes=nodes,
        elements=elements,
        phase=phase,
        label=label,
        node_ijk=node_ijk,
        element_ijk=ijk,
        grid_shape=(nx, ny, nz),
        spacing=tuple(image.spacing),
        node_sets=node_sets,
        metadata=dict(image.metadata),
    )
