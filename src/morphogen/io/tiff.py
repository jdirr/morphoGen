"""TIFF stacks (one page per z slice), via the optional ``tifffile`` package."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from morphogen.image import VoxelImage


def _tifffile():
    try:
        import tifffile
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError(
            "TIFF files need the optional 'tifffile' package: pip install 'morphogen[io]'"
        ) from None
    return tifffile


def write_tiff(path, image: VoxelImage) -> Path:
    """A voxel image written as a TIFF stack (z, y, x) for ImageJ and Fiji, its spacing as
    resolution (tifffile)."""
    tifffile = _tifffile()
    path = Path(path)
    data = image.data.astype(np.uint8) if image.data.dtype == np.bool_ else image.data
    stack = np.ascontiguousarray(data.transpose(2, 1, 0))  # (z, y, x)
    dx, dy, dz = image.spacing
    if stack.dtype in (np.uint8, np.uint16, np.float32):
        tifffile.imwrite(
            path,
            stack,
            imagej=True,
            resolution=(1.0 / dx, 1.0 / dy),
            metadata={"spacing": dz, "axes": "ZYX"},
        )
    else:
        tifffile.imwrite(
            path, stack, resolution=(1.0 / dx, 1.0 / dy), metadata={"spacing": dz, "axes": "ZYX"}
        )
    return path


def read_tiff(path) -> VoxelImage:
    """A voxel image read from a TIFF stack, with the spacing of its resolution tags (tifffile)."""
    tifffile = _tifffile()
    with tifffile.TiffFile(Path(path)) as tif:
        stack = tif.asarray()
        spacing = [1.0, 1.0, 1.0]
        page = tif.pages[0]
        for axis, tag in ((0, "XResolution"), (1, "YResolution")):
            res = page.tags.get(tag)
            if res is not None:
                num, den = res.value
                if num:
                    spacing[axis] = den / num
        meta = tif.imagej_metadata
        if meta is None:  # tifffile's own JSON description (non-ImageJ data types)
            shaped = tif.shaped_metadata or ({},)
            meta = shaped[0] or {}
        if "spacing" in meta:
            spacing[2] = float(meta["spacing"])
    if stack.ndim == 2:
        stack = stack[None]
    return VoxelImage(data=np.ascontiguousarray(stack.transpose(2, 1, 0)), spacing=tuple(spacing))
