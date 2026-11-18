"""Voxel images: a 3D array on a regular grid with its geometry and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from morphogen._utils import as_triplet
from morphogen.domain import Domain, Grid


@dataclass
class VoxelImage:
    """3D image of a microstructure.

    Attributes
    ----------
    data
        Array of shape ``(nx, ny, nz)`` indexed ``[i, j, k]`` along ``(x, y, z)``.
        Typical contents: phase indices (``uint8``), cell labels (``int32``), partial
        volume fractions or level-set values (``float32``).
    spacing
        Voxel edge lengths.
    origin
        Lower corner of voxel ``(0, 0, 0)`` (not its centre).
    periodic
        Whether the image is periodic (measurements then wrap around the faces).
    name
        Name of the array in exported files (e.g. ``"phase"``, ``"material"``).
    metadata
        JSON-able information (model, seed, version...).
    """

    data: np.ndarray
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    periodic: bool = False
    name: str = "phase"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 3:
            raise ValueError(f"VoxelImage: expected a 3D array, got shape {self.data.shape}")
        self.spacing = as_triplet(self.spacing, "spacing", positive=True)
        self.origin = as_triplet(self.origin, "origin")
        self.periodic = bool(self.periodic)

    @classmethod
    def from_grid(cls, data, grid: Grid, *, name: str = "phase", metadata=None) -> VoxelImage:
        return cls(
            data=data,
            spacing=tuple(grid.spacing),
            origin=tuple(grid.origin),
            periodic=grid.domain.periodic,
            name=name,
            metadata=dict(metadata or {}),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.data.shape)

    @property
    def size(self) -> np.ndarray:
        """Physical extent of the image."""
        return np.array(self.shape) * np.array(self.spacing)

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    @property
    def n_voxels(self) -> int:
        return int(self.data.size)

    def domain(self, unit: str = "") -> Domain:
        return Domain(size=tuple(self.size), origin=self.origin, periodic=self.periodic, unit=unit)

    def grid(self) -> Grid:
        return Grid(self.domain(), self.shape)

    def with_data(self, data, *, name: str | None = None) -> VoxelImage:
        """New image on the same grid."""
        return VoxelImage(
            data=data,
            spacing=self.spacing,
            origin=self.origin,
            periodic=self.periodic,
            name=self.name if name is None else name,
            metadata=dict(self.metadata),
        )

    def layer(self, axis: int = 2, index: int | None = None) -> VoxelImage:
        """One layer of voxels normal to ``axis`` (the middle one by default), as an image
        one voxel thick of the same voxels: a planar section of a volume, for a study in
        its plane (a plane strain for the mechanics) on far fewer voxels. Periodic as the
        image (the fields then do not vary across the layer)."""
        if axis not in (0, 1, 2):
            raise ValueError(f"layer: axis 0, 1 or 2, got {axis!r}")
        n = self.shape[axis]
        k = n // 2 if index is None else int(index)
        if not 0 <= k < n:
            raise ValueError(f"layer: index from 0 to {n - 1}, got {index!r}")
        data = np.take(self.data, [k], axis=axis)
        origin = list(self.origin)
        origin[axis] += k * self.spacing[axis]
        return VoxelImage(np.ascontiguousarray(data), self.spacing, tuple(origin), self.periodic,
                          self.name, dict(self.metadata))  # fmt: skip

    def crop(self, start, stop) -> VoxelImage:
        """Sub-image ``data[start[0]:stop[0], ...]`` (never periodic)."""
        s = as_triplet(start, "start", integer=True)
        e = as_triplet(stop, "stop", integer=True)
        sub = self.data[s[0] : e[0], s[1] : e[1], s[2] : e[2]]
        origin = tuple(o + i * h for o, i, h in zip(self.origin, s, self.spacing, strict=True))
        return VoxelImage(sub, self.spacing, origin, False, self.name, dict(self.metadata))

    def __repr__(self) -> str:
        return (
            f"VoxelImage(name={self.name!r}, shape={self.shape}, dtype={self.data.dtype}, "
            f"spacing={self.spacing}, periodic={self.periodic})"
        )
