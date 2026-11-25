"""Particles given by their closed triangle surfaces: the particles of a phase written for
the discrete element method (:mod:`morphogen.io.dem`), settled by the rigid bodies of
Blender (:mod:`morphogen.interop.blender`), or read from the surfaces of another code, as
a microstructure (phase 1: the particles)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.kernels.meshes import surface_parity
from morphogen.mesh.surface import SurfaceMesh
from morphogen.models.base import Microstructure, register_kind


@register_kind
class MeshParticles(Microstructure):
    """Phase 1 is the union of particles given by closed triangle surfaces (outward or
    inward, the parity of the crossings does not tell), phase 0 the rest; in a periodic
    domain a particle that crosses a face comes back through the opposite one.

    The voxels are those whose centres lie inside a surface (the parity of the crossings
    of their columns); the implicit function of phase 1 is the signed distance of those
    voxels (exact Euclidean distance transforms, the boundary half a voxel from the centres
    on either side), within half a voxel of the distance to the surfaces."""

    kind = "mesh_particles"

    def __init__(self, domain: Domain, surfaces=(), **kwargs) -> None:
        super().__init__(domain, **kwargs)
        self.surfaces = [
            s if isinstance(s, SurfaceMesh) else SurfaceMesh(np.asarray(s[0], float),
                                                              np.asarray(s[1], np.int64))
            for s in surfaces
        ]  # fmt: skip

    def _inside(self, grid: Grid) -> np.ndarray:
        shape = np.array(grid.shape)
        out = np.zeros(grid.shape, dtype=np.uint8)
        origin, spacing = np.asarray(grid.origin), np.asarray(grid.spacing)
        for s in self.surfaces:
            index = (s.vertices - origin) / spacing - 0.5  # the centre of voxel i at i
            lo = np.floor(index.min(axis=0)).astype(np.int64) - 1
            hi = np.ceil(index.max(axis=0)).astype(np.int64) + 1
            if not self.domain.periodic:
                if np.any(hi < 0) or np.any(lo >= shape):
                    continue
            local = surface_parity(index - lo, s.faces, tuple(int(v) for v in hi - lo + 1))
            cells = [np.arange(lo[a], hi[a] + 1) for a in range(3)]
            if self.domain.periodic:
                target = np.ix_(*[c % shape[a] for a, c in enumerate(cells)])
                out[target] |= local
            else:
                keep = [(c >= 0) & (c < shape[a]) for a, c in enumerate(cells)]
                if not all(k.any() for k in keep):
                    continue
                out[np.ix_(*[c[k] for c, k in zip(cells, keep, strict=True)])] |= local[
                    np.ix_(*keep)
                ]
        return out

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        meta = self.image_metadata()
        if supersampling == 1:
            return VoxelImage.from_grid(self._inside(grid), grid, name="phase", metadata=meta)
        s = int(supersampling)
        fine = self._inside(Grid(grid.domain, tuple(n * s for n in grid.shape)))
        nx, ny, nz = grid.shape
        data = fine.reshape(nx, s, ny, s, nz, s).mean(axis=(1, 3, 5)).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=meta)

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        inside = self._inside(grid).astype(bool)
        spacing = np.asarray(grid.spacing)
        pad = [min(n, 16) for n in grid.shape] if self.domain.periodic else [0, 0, 0]
        big = np.pad(inside, [(p, p) for p in pad], mode="wrap") if any(pad) else inside
        outer = ndimage.distance_transform_edt(~big, sampling=spacing)
        inner = ndimage.distance_transform_edt(big, sampling=spacing)
        crop = tuple(slice(p, p + n) for p, n in zip(pad, grid.shape, strict=True))
        half = 0.5 * float(spacing.min())
        phi = np.where(inside, -(inner[crop] - half), outer[crop] - half).astype(np.float32)
        return VoxelImage.from_grid(phi, grid, name="level_set", metadata=self.image_metadata())

    def summary(self) -> dict:
        return {"particles": len(self.surfaces)}

    def _geometry(self) -> dict:
        return {
            "particles": [
                {"vertices": s.vertices.tolist(), "faces": s.faces.tolist()} for s in self.surfaces
            ]
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        surfaces = [(p["vertices"], p["faces"]) for p in geometry["particles"]]
        return cls(domain, surfaces, **kwargs)
