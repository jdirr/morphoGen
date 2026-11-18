"""NumPy-based files: .npz (array + geometry + metadata, round-trips) and raw .npy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from morphogen._version import __version__
from morphogen.image import VoxelImage


def write_npz(path, image: VoxelImage, *, compress: bool = True) -> Path:
    """A voxel image written as a NumPy archive (.npz, compressed if ``compress``) with its spacing,
    origin, periodicity, name, metadata and the version of morphoGen."""
    path = Path(path)
    save = np.savez_compressed if compress else np.savez
    save(
        path,
        data=image.data,
        spacing=np.array(image.spacing),
        origin=np.array(image.origin),
        periodic=np.array(image.periodic),
        name=np.array(image.name),
        metadata=np.array(json.dumps(image.metadata)),
        morphogen=np.array(__version__),
    )
    return path


def read_npz(path) -> VoxelImage:
    """A voxel image read back from the .npz archive of :func:`write_npz`, with its spacing, origin,
    periodicity, name and metadata."""
    with np.load(Path(path), allow_pickle=False) as f:
        return VoxelImage(
            data=f["data"],
            spacing=tuple(f["spacing"]),
            origin=tuple(f["origin"]),
            periodic=bool(f["periodic"]),
            name=str(f["name"]),
            metadata=json.loads(str(f["metadata"])),
        )


def write_npy(path, image: VoxelImage) -> Path:
    """The phases of a voxel image written as a NumPy array (.npy), without its spacing or
    metadata."""
    path = Path(path)
    np.save(path, image.data)
    return path


def read_npy(path) -> VoxelImage:
    """A voxel image of the phases of a NumPy array (.npy), of unit spacing."""
    data = np.load(Path(path), allow_pickle=False)
    return VoxelImage(data=data)
