"""Triangulated surfaces extracted from level-set or phase images (marching cubes)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morphogen.image import VoxelImage


@dataclass
class SurfaceMesh:
    vertices: np.ndarray  # (n, 3)
    faces: np.ndarray  # (m, 3), counter-clockwise seen from outside

    def face_normals(self) -> np.ndarray:
        v = self.vertices[self.faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(norm > 0, norm, 1.0)

    def area(self) -> float:
        v = self.vertices[self.faces]
        return float(
            0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum()
        )

    def volume(self) -> float:
        """Enclosed volume (divergence theorem; positive for outward-oriented surfaces)."""
        v = self.vertices[self.faces]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)


def _marching_cubes():
    try:
        from skimage.measure import marching_cubes
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError(
            "surface meshes need the optional 'scikit-image' package: "
            "pip install 'morphogen[surface]'"
        ) from None
    return marching_cubes


def surface_mesh(
    image: VoxelImage, *, level: float = 0.0, phase: int = 1, closed: bool = True
) -> SurfaceMesh:
    """Surface of phase 1 from a level-set image (negative inside) or a phase image.

    Phase images (integers) and partial-volume images (floats named ``fraction``) are
    converted to ``0.5 - indicator``. With ``closed=True`` the solid is closed by caps on
    the faces of the box (watertight surface, e.g. for additive manufacturing).
    """
    marching_cubes = _marching_cubes()
    data = image.data
    if data.dtype.kind in "iub":
        field = 0.5 - (data == phase).astype(np.float32)
        level = 0.0
    elif image.name == "fraction":
        field = 0.5 - data.astype(np.float32)
        level = 0.0
    else:
        field = data.astype(np.float32)
    # values exactly at the level put vertices on grid nodes, where several edges then
    # share a position and the surface is no longer manifold: push them slightly outside
    # (far enough for the vertices to stay distinct in single precision)
    scale = float(np.abs(field).max()) if field.size else 1.0
    field = np.where(field == level, np.float32(level + 1e-3 * max(scale, 1e-30)), field)
    spacing = np.asarray(image.spacing, dtype=float)
    origin = np.asarray(image.origin, dtype=float) + 0.5 * spacing
    if closed:
        outside = np.float32(max(float(np.abs(field).max()), 1.0) + abs(level) + 1.0)
        field = np.pad(field, 1, mode="constant", constant_values=outside)
        origin = origin - spacing
    verts, faces, _, _ = marching_cubes(
        field,
        level=level,
        spacing=tuple(spacing),
        gradient_direction="ascent",
        allow_degenerate=False,
    )
    verts = verts + origin
    if closed:
        # cap vertices lie on edges joining the outer voxel centres to the padding:
        # move them onto the faces of the box
        lower = np.asarray(image.origin, dtype=float)
        upper = lower + spacing * np.array(image.shape)
        first = lower + 0.5 * spacing
        last = upper - 0.5 * spacing
        eps = 1e-9 * spacing
        verts = np.where(verts < first - eps, lower, verts)
        verts = np.where(verts > last + eps, upper, verts)
    mesh = SurfaceMesh(verts, faces.astype(np.int64))
    if mesh.volume() < 0:
        mesh = SurfaceMesh(verts, faces[:, ::-1].astype(np.int64))
    return mesh
