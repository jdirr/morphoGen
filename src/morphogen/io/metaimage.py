"""MetaImage (.mhd header + .raw data), read by ITK, ParaView, Fiji and 3D Slicer.

MetaImage gives the position of the *centre* of the first voxel (``Offset``).
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from morphogen.image import VoxelImage

_TO_MET = {
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
_FROM_MET = {v: np.dtype(k) for k, v in _TO_MET.items()}


def write_mhd(path, image: VoxelImage, *, compress: bool = False) -> Path:
    """A voxel image written as a MetaImage header (.mhd) and its raw data (.raw, or .zraw
    compressed with zlib if ``compress``), read by ITK, ParaView and 3D Slicer."""
    path = Path(path)
    if path.suffix.lower() != ".mhd":
        raise ValueError("MetaImage header files must end with .mhd")
    data = image.data.astype(np.uint8) if image.data.dtype == np.bool_ else image.data
    if data.dtype.name not in _TO_MET:
        raise ValueError(f"unsupported data type for MetaImage: {data.dtype}")
    raw_path = path.with_suffix(".zraw" if compress else ".raw")
    raw = (
        np.ascontiguousarray(data.transpose(2, 1, 0)).astype(data.dtype.newbyteorder("<")).tobytes()
    )
    if compress:
        raw = zlib.compress(raw, 6)
    raw_path.write_bytes(raw)
    centre = np.array(image.origin) + 0.5 * np.array(image.spacing)
    lines = [
        "ObjectType = Image",
        "NDims = 3",
        "BinaryData = True",
        "BinaryDataByteOrderMSB = False",
        f"CompressedData = {compress}",
        *([f"CompressedDataSize = {len(raw)}"] if compress else []),
        "TransformMatrix = 1 0 0 0 1 0 0 0 1",
        f"Offset = {' '.join(repr(float(v)) for v in centre)}",
        "CenterOfRotation = 0 0 0",
        f"ElementSpacing = {' '.join(repr(float(v)) for v in image.spacing)}",
        f"DimSize = {' '.join(str(n) for n in image.shape)}",
        f"ElementType = {_TO_MET[data.dtype.name]}",
        f"ElementDataFile = {raw_path.name}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def read_mhd(path) -> VoxelImage:
    """A voxel image read from a MetaImage header (.mhd) and its raw data (.raw, or .zraw compressed
    with zlib)."""
    path = Path(path)
    header = {}
    for text in path.read_text(encoding="ascii", errors="replace").splitlines():
        if "=" in text:
            key, value = text.split("=", 1)
            header[key.strip()] = value.strip()
    if int(header.get("NDims", "3")) != 3:
        raise ValueError(f"{path}: only 3D MetaImages are supported")
    shape = tuple(int(v) for v in header["DimSize"].split())
    spacing = np.array([float(v) for v in header.get("ElementSpacing", "1 1 1").split()])
    offset = header.get("Offset", header.get("Origin", header.get("Position", "0 0 0")))
    centre = np.array([float(v) for v in offset.split()])
    dtype = _FROM_MET[header["ElementType"]]
    big = header.get("BinaryDataByteOrderMSB", header.get("ElementByteOrderMSB", "False"))
    dtype = dtype.newbyteorder(">" if big.lower() == "true" else "<")
    data_file = header["ElementDataFile"]
    if data_file == "LOCAL":
        content = path.read_bytes()
        nbytes = int(np.prod(shape)) * dtype.itemsize
        raw = content[-nbytes:]
    else:
        raw = (path.parent / data_file).read_bytes()
    if header.get("CompressedData", "False").lower() == "true":
        raw = zlib.decompress(raw)
    values = np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape))).astype(
        dtype.newbyteorder("=")
    )
    data = values.reshape(shape[::-1]).transpose(2, 1, 0).copy()
    return VoxelImage(data=data, spacing=tuple(spacing), origin=tuple(centre - 0.5 * spacing))
