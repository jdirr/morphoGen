"""XDMF images and voxel images written slab by slab (volumes larger than memory).

An XDMF image is an XML header (``.xdmf``, read by ParaView and VisIt) referring to raw
little-endian binary files, one per array, with the x index varying fastest. The
MetaImage format (``.mhd`` + ``.raw``) has the same layout. :func:`voxelize_to_file`
samples a microstructure slab by slab along z into a memory-mapped raw file, so that the
volume never needs to fit in memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage

_TYPES = {
    "uint8": ("UChar", 1),
    "int8": ("Char", 1),
    "uint16": ("UInt", 2),
    "int16": ("Int", 2),
    "uint32": ("UInt", 4),
    "int32": ("Int", 4),
    "uint64": ("UInt", 8),
    "int64": ("Int", 8),
    "float32": ("Float", 4),
    "float64": ("Float", 8),
}
_MET = {
    "uint8": "MET_UCHAR",
    "int8": "MET_CHAR",
    "uint16": "MET_USHORT",
    "int16": "MET_SHORT",
    "uint32": "MET_UINT",
    "int32": "MET_INT",
    "uint64": "MET_ULONG_LONG",
    "int64": "MET_LONG_LONG",
    "float32": "MET_FLOAT",
    "float64": "MET_DOUBLE",
}


def _dtype(dtype) -> np.dtype:
    dtype = np.dtype(np.uint8 if np.dtype(dtype) == np.bool_ else dtype)
    if dtype.name not in _TYPES:
        raise ValueError(f"unsupported data type for raw images: {dtype}")
    return dtype


def _xdmf_header(shape, spacing, origin, arrays) -> str:
    """``arrays``: (name, raw file name, dtype)."""
    nx, ny, nz = shape
    attrs = []
    for name, raw, dtype in arrays:
        number, precision = _TYPES[dtype.name]
        attrs.append(
            f'      <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">\n'
            f'        <DataItem Dimensions="{nz} {ny} {nx}" NumberType="{number}" '
            f'Precision="{precision}" Format="Binary" Endian="Little">{raw}</DataItem>\n'
            "      </Attribute>"
        )
    z = " ".join(repr(float(v)) for v in origin[::-1])
    h = " ".join(repr(float(v)) for v in spacing[::-1])
    return (
        '<?xml version="1.0" ?>\n'
        '<Xdmf Version="3.0">\n'
        "  <Domain>\n"
        '    <Grid Name="image" GridType="Uniform">\n'
        f'      <Topology TopologyType="3DCoRectMesh" Dimensions="{nz + 1} {ny + 1} {nx + 1}"/>\n'
        '      <Geometry GeometryType="ORIGIN_DXDYDZ">\n'
        f'        <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">{z}'
        "</DataItem>\n"
        f'        <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">{h}'
        "</DataItem>\n"
        "      </Geometry>\n" + "\n".join(attrs) + "\n"
        "    </Grid>\n"
        "  </Domain>\n"
        "</Xdmf>\n"
    )


def write_xdmf(path, images) -> Path:
    """XDMF header ``path`` and one raw file per image (same grid): ``<stem>_<name>.raw``."""
    path = Path(path)
    images = [images] if isinstance(images, VoxelImage) else list(images)
    if not images:
        raise ValueError("no image to write")
    first = images[0]
    arrays = []
    for img in images:
        if img.shape != first.shape:
            raise ValueError("the images of an XDMF file share their grid")
        dtype = _dtype(img.data.dtype)
        raw = path.parent / f"{path.stem}_{img.name}.raw"
        data = np.ascontiguousarray(img.data.astype(dtype).transpose(2, 1, 0))
        raw.write_bytes(data.astype(dtype.newbyteorder("<"), copy=False).tobytes())
        arrays.append((img.name, raw.name, dtype))
    path.write_text(_xdmf_header(first.shape, first.spacing, first.origin, arrays), "ascii")
    return path


def read_xdmf(path, name: str | None = None) -> VoxelImage:
    """Image (array ``name``, or the first one) of an XDMF file written by morphoGen."""
    import re

    path = Path(path)
    text = path.read_text(encoding="ascii")
    dims = [int(v) for v in re.search(r'Dimensions="(\d+) (\d+) (\d+)"/>', text).groups()]
    nz, ny, nx = (d - 1 for d in dims)
    geo = re.findall(r'Format="XML">([^<]+)</DataItem>', text)
    origin = tuple(float(v) for v in geo[0].split()[::-1])
    spacing = tuple(float(v) for v in geo[1].split()[::-1])
    for match in re.finditer(
        r'<Attribute Name="([^"]+)".*?NumberType="(\w+)" Precision="(\d)".*?>([^<]+)</DataItem>',
        text,
        re.S,
    ):
        array, number, precision, raw = match.groups()
        if name is not None and array != name:
            continue
        dtype = next(np.dtype(k) for k, v in _TYPES.items() if v == (number, int(precision)))
        data = np.fromfile(path.parent / raw, dtype=dtype.newbyteorder("<")).astype(dtype)
        data = data.reshape(nz, ny, nx).transpose(2, 1, 0)
        return VoxelImage(np.ascontiguousarray(data), spacing=spacing, origin=origin, name=array)
    raise ValueError(f"{path}: no array {name!r}")


def _slab_grid(grid: Grid, k0: int, k1: int) -> Grid:
    """Voxels ``[:, :, k0:k1]`` of a grid, as a grid of their own."""
    h = np.asarray(grid.spacing, dtype=float)
    lower = np.asarray(grid.domain.lower, dtype=float)
    size = (grid.domain.lengths[0], grid.domain.lengths[1], h[2] * (k1 - k0))
    origin = (lower[0], lower[1], lower[2] + h[2] * k0)
    return Grid(
        Domain(size=size, origin=origin, periodic=False), (grid.shape[0], grid.shape[1], k1 - k0)
    )


def voxelize_to_file(
    micro,
    path,
    shape=None,
    *,
    voxel_size=None,
    labels: bool = False,
    slab: int | None = None,
    max_voxels: int = 2**24,
) -> Path:
    """Phase (or with ``labels``, cell label) image of a microstructure written to an
    ``.xdmf`` or ``.mhd`` file slab by slab along z: memory holds one slab (about
    ``max_voxels`` voxels, or ``slab`` layers), whatever the size of the image. The
    models whose images are computed pointwise give the same voxels as at once."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in (".xdmf", ".mhd"):
        raise ValueError("chunked images are written to .xdmf or .mhd files")
    grid = micro.grid(shape, voxel_size=voxel_size)
    nx, ny, nz = grid.shape
    step = int(slab) if slab else max(1, int(max_voxels) // (nx * ny))
    name = "cell" if labels else "phase"
    raw = path.parent / (f"{path.stem}_{name}.raw" if ext == ".xdmf" else f"{path.stem}.raw")
    out = None
    for k0 in range(0, nz, step):
        k1 = min(nz, k0 + step)
        sub = _slab_grid(grid, k0, k1)
        img = micro.labels(grid=sub) if labels else micro.voxelize(grid=sub)
        dtype = _dtype(img.data.dtype)
        if out is None:
            out = np.memmap(raw, dtype=dtype.newbyteorder("<"), mode="w+", shape=(nz, ny, nx))
            first = dtype
        if dtype != first:
            raise ValueError("the slabs of the image have different data types")
        out[k0:k1] = img.data.astype(dtype).transpose(2, 1, 0)
    out.flush()
    del out
    if ext == ".xdmf":
        header = _xdmf_header(
            grid.shape, grid.spacing, grid.domain.lower, [(name, raw.name, first)]
        )
        path.write_text(header, encoding="ascii")
    else:
        centre = np.asarray(grid.domain.lower, dtype=float) + 0.5 * np.asarray(grid.spacing)
        lines = [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            f"Offset = {' '.join(repr(float(v)) for v in centre)}",
            "CenterOfRotation = 0 0 0",
            f"ElementSpacing = {' '.join(repr(float(v)) for v in grid.spacing)}",
            f"DimSize = {nx} {ny} {nz}",
            f"ElementType = {_MET[first.name]}",
            f"ElementDataFile = {raw.name}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path
