"""VTK files: XML image data (.vti), legacy structured points (.vtk), unstructured grids (.vtu).

Voxel values are stored as cell data, x varying fastest. XML files use appended raw
binary data, optionally zlib-compressed; the reader also accepts base64 and ASCII data
written by other VTK-based tools.
"""

from __future__ import annotations

import base64
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

from morphogen.image import VoxelImage

_TO_VTK = {
    "uint8": "UInt8",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "uint64": "UInt64",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
}
_FROM_VTK = {v: np.dtype(k) for k, v in _TO_VTK.items()}
_BLOCK = 1 << 20


def _as_vtk_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if arr.dtype.name not in _TO_VTK:
        raise ValueError(f"unsupported data type for VTK files: {arr.dtype}")
    return arr


def _fortran_bytes(arr: np.ndarray) -> bytes:
    """Bytes of a 3D array with x varying fastest, little-endian."""
    a = np.ascontiguousarray(arr.transpose(2, 1, 0))
    return a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()


def _encode_block(raw: bytes, compress: bool) -> bytes:
    if not compress:
        return struct.pack("<Q", len(raw)) + raw
    chunks = [zlib.compress(raw[i : i + _BLOCK], 6) for i in range(0, len(raw), _BLOCK)] or [b""]
    n = len(chunks)
    last = len(raw) - (n - 1) * _BLOCK if raw else 0
    partial = 0 if last == _BLOCK else last
    header = struct.pack(f"<{3 + n}Q", n, _BLOCK, partial, *(len(c) for c in chunks))
    return header + b"".join(chunks)


def _check_same_grid(images: list[VoxelImage]) -> VoxelImage:
    ref = images[0]
    for img in images[1:]:
        if (
            img.shape != ref.shape
            or not np.allclose(img.spacing, ref.spacing)
            or not np.allclose(img.origin, ref.origin)
        ):
            raise ValueError("all arrays of a file must share the same grid")
    names = [img.name for img in images]
    if len(set(names)) != len(names):
        raise ValueError(f"array names must be unique, got {names}")
    return ref


def write_vti(path, images, *, compress: bool = True) -> Path:
    """Write one or several images on the same grid to a VTK XML ImageData file."""
    images = [images] if isinstance(images, VoxelImage) else list(images)
    ref = _check_same_grid(images)
    nx, ny, nz = ref.shape
    extent = f"0 {nx} 0 {ny} 0 {nz}"
    origin = " ".join(repr(float(v)) for v in ref.origin)
    spacing = " ".join(repr(float(v)) for v in ref.spacing)
    blocks, arrays, offset = [], [], 0
    for img in images:
        arr = _as_vtk_array(img.data)
        block = _encode_block(_fortran_bytes(arr), compress)
        arrays.append(
            f'        <DataArray type="{_TO_VTK[arr.dtype.name]}" Name="{img.name}" '
            f'format="appended" offset="{offset}"/>'
        )
        blocks.append(block)
        offset += len(block)
    compressor = ' compressor="vtkZLibDataCompressor"' if compress else ""
    header = (
        '<?xml version="1.0"?>\n'
        f'<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
        f'header_type="UInt64"{compressor}>\n'
        f'  <ImageData WholeExtent="{extent}" Origin="{origin}" Spacing="{spacing}">\n'
        f'    <Piece Extent="{extent}">\n'
        f'      <CellData Scalars="{images[0].name}">\n' + "\n".join(arrays) + "\n"
        "      </CellData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        '  <AppendedData encoding="raw">\n   _'
    )
    path = Path(path)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for block in blocks:
            f.write(block)
        f.write(b"\n  </AppendedData>\n</VTKFile>\n")
    return path


# -- reading XML ----------------------------------------------------------------------


def _decode_binary(raw: bytes, dtype: np.dtype, count: int, header: str, compressed: bool):
    """Decode a raw (already base64-decoded if needed) data block."""
    hfmt, hsize = ("<Q", 8) if header == "UInt64" else ("<I", 4)
    if not compressed:
        (nbytes,) = struct.unpack(hfmt, raw[:hsize])
        data = raw[hsize : hsize + nbytes]
    else:
        (nblocks,) = struct.unpack(hfmt, raw[:hsize])
        values = struct.unpack(f"<{3 + nblocks}{hfmt[1]}", raw[: (3 + nblocks) * hsize])
        sizes = values[3:]
        pos = (3 + nblocks) * hsize
        parts = []
        for size in sizes:
            parts.append(zlib.decompress(raw[pos : pos + size]))
            pos += size
        data = b"".join(parts)
    return np.frombuffer(data, dtype=dtype.newbyteorder("<"), count=count)


def _b64len(nbytes: int) -> int:
    return 4 * ((nbytes + 2) // 3)


def _decode_base64(text: bytes, header: str, compressed: bool) -> bytes:
    """Decode a base64 data block (header and data encoded separately or together)."""
    hsize = 8 if header == "UInt64" else 4
    hfmt = "<Q" if header == "UInt64" else "<I"
    # the first characters decode to the first value whatever the layout
    first = base64.b64decode(text[: _b64len(hsize)])
    if not compressed:
        (nbytes,) = struct.unpack(hfmt, first[:hsize])
        hbytes, dbytes = hsize, nbytes
    else:
        (nblocks,) = struct.unpack(hfmt, first[:hsize])
        hbytes = (3 + nblocks) * hsize
        head = base64.b64decode(text[: _b64len(hbytes)])[:hbytes]
        dbytes = sum(struct.unpack(f"<{3 + nblocks}{hfmt[1]}", head)[3:])
    # a separately encoded header ends with padding unless its size is a multiple of 3
    separate = hbytes % 3 == 0 or text[_b64len(hbytes) - 1 : _b64len(hbytes)] == b"="
    if separate:
        head = base64.b64decode(text[: _b64len(hbytes)])[:hbytes]
        start = _b64len(hbytes)
        return head + base64.b64decode(text[start : start + _b64len(dbytes)])
    return base64.b64decode(text[: _b64len(hbytes + dbytes)])[: hbytes + dbytes]


def _read_xml_arrays(path: Path):
    content = Path(path).read_bytes()
    split = content.find(b"<AppendedData")
    if split >= 0:
        xml_part = content[:split] + b"</VTKFile>"
        tail = content[split:]
        encoding = re.search(rb'encoding="(\w+)"', tail[:200])
        encoding = encoding.group(1).decode() if encoding else "raw"
        start = tail.find(b"_") + 1
        appended = tail[start:]
    else:
        xml_part, appended, encoding = content, b"", "raw"
    root = ET.fromstring(xml_part)
    if root.get("byte_order", "LittleEndian") != "LittleEndian":
        raise ValueError("only little-endian VTK XML files are supported")
    header = root.get("header_type", "UInt32")
    compressed = root.get("compressor") is not None
    return root, appended, encoding, header, compressed


def _array_values(el, count, appended, encoding, header, compressed) -> np.ndarray:
    dtype = _FROM_VTK[el.get("type")]
    count = count * int(el.get("NumberOfComponents", "1"))
    fmt = el.get("format", "ascii")
    if fmt == "ascii":
        return np.array((el.text or "").split(), dtype=dtype)[:count]
    if fmt == "binary":
        raw = _decode_base64("".join((el.text or "").split()).encode(), header, compressed)
        return _decode_binary(raw, dtype, count, header, compressed)
    offset = int(el.get("offset"))
    if encoding == "base64":
        raw = _decode_base64(appended[offset:], header, compressed)
    else:
        raw = appended[offset:]
    return _decode_binary(raw, dtype, count, header, compressed)


def vti_names(path) -> list[str]:
    """Names of the scalar arrays of a VTK XML ImageData file, from its header alone."""
    head = b""
    with open(path, "rb") as f:
        while b"<AppendedData" not in head:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            head += chunk
    split = head.find(b"<AppendedData")
    root = ET.fromstring(head[:split] + b"</VTKFile>" if split >= 0 else head)
    piece = root.find("ImageData").find("Piece")
    return [
        el.get("Name", "data")
        for section in ("CellData", "PointData")
        if piece.find(section) is not None
        for el in piece.find(section).findall("DataArray")
        if int(el.get("NumberOfComponents", "1")) == 1
    ]


def read_vti(path, names=None) -> list[VoxelImage]:
    """Read the cell (or point) arrays of a VTK XML ImageData file (only those called
    ``names``, if given: :func:`vti_names` lists them)."""
    root, appended, encoding, header, compressed = _read_xml_arrays(Path(path))
    image = root.find("ImageData")
    ext = [int(v) for v in image.get("WholeExtent").split()]
    npts = (ext[1] - ext[0] + 1, ext[3] - ext[2] + 1, ext[5] - ext[4] + 1)
    origin = np.array([float(v) for v in image.get("Origin", "0 0 0").split()])
    spacing = np.array([float(v) for v in image.get("Spacing", "1 1 1").split()])
    origin = origin + spacing * np.array(ext[0::2])
    piece = image.find("Piece")
    out = []
    for section, shape, shift in (
        ("CellData", tuple(n - 1 for n in npts), 0.0),
        ("PointData", npts, -0.5),
    ):
        node = piece.find(section)
        if node is None:
            continue
        count = int(np.prod(shape))
        for el in node.findall("DataArray"):
            if int(el.get("NumberOfComponents", "1")) != 1:
                continue
            if names is not None and el.get("Name", "data") not in names:
                continue
            values = _array_values(el, count, appended, encoding, header, compressed)
            data = values.reshape(shape[::-1]).transpose(2, 1, 0).copy()
            out.append(
                VoxelImage(
                    data=data,
                    spacing=tuple(spacing),
                    origin=tuple(origin + shift * spacing),
                    name=el.get("Name", "data"),
                )
            )
    if not out and names is None:
        raise ValueError(f"{path}: no scalar arrays found")
    return out


# -- legacy VTK -------------------------------------------------------------------------

_TO_LEGACY = {
    "uint8": "unsigned_char",
    "int8": "char",
    "uint16": "unsigned_short",
    "int16": "short",
    "uint32": "unsigned_int",
    "int32": "int",
    "uint64": "unsigned_long",
    "int64": "long",
    "float32": "float",
    "float64": "double",
}
_FROM_LEGACY = {v: np.dtype(k) for k, v in _TO_LEGACY.items()}


def write_vtk(path, images, *, binary: bool = True) -> Path:
    """Legacy VTK STRUCTURED_POINTS with cell data (big-endian binary or ASCII)."""
    images = [images] if isinstance(images, VoxelImage) else list(images)
    ref = _check_same_grid(images)
    nx, ny, nz = ref.shape
    path = Path(path)
    with open(path, "wb") as f:
        head = (
            "# vtk DataFile Version 3.0\n"
            f"morphoGen {ref.name}\n"
            f"{'BINARY' if binary else 'ASCII'}\n"
            "DATASET STRUCTURED_POINTS\n"
            f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n"
            f"ORIGIN {' '.join(repr(float(v)) for v in ref.origin)}\n"
            f"SPACING {' '.join(repr(float(v)) for v in ref.spacing)}\n"
            f"CELL_DATA {nx * ny * nz}\n"
        )
        f.write(head.encode("ascii"))
        for img in images:
            arr = _as_vtk_array(img.data)
            f.write(f"SCALARS {img.name} {_TO_LEGACY[arr.dtype.name]} 1\n".encode("ascii"))
            f.write(b"LOOKUP_TABLE default\n")
            flat = np.ascontiguousarray(arr.transpose(2, 1, 0)).ravel()
            if binary:
                f.write(flat.astype(flat.dtype.newbyteorder(">")).tobytes())
                f.write(b"\n")
            else:
                np.savetxt(f, flat.reshape(-1, 1), fmt="%.9g" if arr.dtype.kind == "f" else "%d")
    return path


def read_vtk(path) -> list[VoxelImage]:
    """Read scalar arrays of a legacy STRUCTURED_POINTS file (cell or point data)."""
    content = Path(path).read_bytes()
    pos = 0

    def line():
        nonlocal pos
        end = content.index(b"\n", pos)
        text = content[pos:end].decode("ascii", "replace").strip()
        pos = end + 1
        return text

    line()
    line()
    mode = line().upper()
    if line().upper() != "DATASET STRUCTURED_POINTS":
        raise ValueError(f"{path}: only STRUCTURED_POINTS datasets are supported")
    dims = origin = None
    spacing = np.ones(3)
    out, section, count = [], None, 0
    while pos < len(content):
        text = line()
        if not text:
            continue
        key, *vals = text.split()
        key = key.upper()
        if key == "DIMENSIONS":
            dims = np.array([int(v) for v in vals])
        elif key == "ORIGIN":
            origin = np.array([float(v) for v in vals])
        elif key in ("SPACING", "ASPECT_RATIO"):
            spacing = np.array([float(v) for v in vals])
        elif key in ("CELL_DATA", "POINT_DATA"):
            section, count = key, int(vals[0])
        elif key == "METADATA":
            while pos < len(content) and line():
                pass
        elif key == "FIELD":
            for _ in range(int(vals[1])):
                text = line()
                while not text:
                    text = line()
                name, ncomp, ntuples, tname = text.split()[:4]
                dtype = _FROM_LEGACY[tname]
                n = int(ncomp) * int(ntuples)
                if mode == "BINARY":
                    be = dtype.newbyteorder(">")
                    values = np.frombuffer(content, dtype=be, count=n, offset=pos).astype(dtype)
                    pos += n * dtype.itemsize
                else:
                    tokens = []
                    while len(tokens) < n:
                        tokens.extend(line().split())
                    values = np.array(tokens[:n], dtype=dtype)
                if pos < len(content) and content[pos : pos + 1] == b"\n":
                    pos += 1
                if content[pos : pos + 8] == b"METADATA":
                    line()
                    while pos < len(content) and line():
                        pass
                if int(ncomp) != 1 or int(ntuples) != count:
                    continue
                shape = tuple(dims - 1) if section == "CELL_DATA" else tuple(dims)
                shift = 0.0 if section == "CELL_DATA" else -0.5
                data = values.reshape(shape[::-1]).transpose(2, 1, 0).copy()
                o = (origin if origin is not None else np.zeros(3)) + shift * spacing
                out.append(
                    VoxelImage(data=data, spacing=tuple(spacing), origin=tuple(o), name=name)
                )
        elif key == "SCALARS":
            name, dtype = vals[0], _FROM_LEGACY[vals[1]]
            ncomp = int(vals[2]) if len(vals) > 2 else 1
            if content[pos : pos + 12].upper() == b"LOOKUP_TABLE":
                line()
            n = count * ncomp
            if mode == "BINARY":
                be = dtype.newbyteorder(">")
                values = np.frombuffer(content, dtype=be, count=n, offset=pos).astype(dtype)
                pos += n * dtype.itemsize
            else:
                tokens = []
                while len(tokens) < n:
                    tokens.extend(line().split())
                values = np.array(tokens[:n], dtype=dtype)
            if ncomp != 1:
                continue
            shape = tuple(dims - 1) if section == "CELL_DATA" else tuple(dims)
            shift = 0.0 if section == "CELL_DATA" else -0.5
            data = values.reshape(shape[::-1]).transpose(2, 1, 0).copy()
            o = (origin if origin is not None else np.zeros(3)) + shift * spacing
            out.append(VoxelImage(data=data, spacing=tuple(spacing), origin=tuple(o), name=name))
    if not out:
        raise ValueError(f"{path}: no scalar arrays found")
    return out


# -- unstructured grids -----------------------------------------------------------------


def write_vtu(
    path,
    points,
    cells,
    cell_type: int,
    cell_data: dict | None = None,
    *,
    point_data: dict | None = None,
    field_data: dict | None = None,
    compress: bool = True,
) -> Path:
    """VTK XML UnstructuredGrid with a single cell type.

    Cell types: 12 hexahedron, 10 tetrahedron, 24 quadratic tetrahedron, 5 triangle.
    ``cell_data`` and ``point_data`` map names to arrays of one value (shape ``(n,)``)
    or several components (shape ``(n, k)``) per cell or point; ``field_data`` to small
    arrays describing the whole grid.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    cells = np.ascontiguousarray(cells, dtype=np.int64)
    n_cells, per_cell = cells.shape
    arrays = [
        ("Points", "Float64", 3, points.ravel(), "Points"),
        ("connectivity", "Int64", 1, cells.ravel(), "Cells"),
        ("offsets", "Int64", 1, np.arange(1, n_cells + 1, dtype=np.int64) * per_cell, "Cells"),
        ("types", "UInt8", 1, np.full(n_cells, cell_type, dtype=np.uint8), "Cells"),
    ]
    sections = (("CellData", cell_data), ("PointData", point_data), ("FieldData", field_data))
    for section, data in sections:
        for name, values in (data or {}).items():
            values = np.ascontiguousarray(values)
            if section == "FieldData":
                values = values.reshape(-1)
            if values.dtype == np.bool_ or values.dtype.name not in _TO_VTK:
                values = values.astype(np.int64)
            ncomp = 1 if values.ndim == 1 else int(values.shape[1])
            arrays.append((name, _TO_VTK[values.dtype.name], ncomp, values.ravel(), section))
    xml = {"Points": [], "Cells": [], "CellData": [], "PointData": [], "FieldData": []}
    blocks, offset = [], 0
    for name, vtype, ncomp, values, section in arrays:
        raw = values.astype(values.dtype.newbyteorder("<"), copy=False).tobytes()
        block = _encode_block(raw, compress)
        tuples = f' NumberOfTuples="{values.size}"' if section == "FieldData" else ""
        indent = "      " if section == "FieldData" else "        "
        xml[section].append(
            f'{indent}<DataArray type="{vtype}" Name="{name}" NumberOfComponents="{ncomp}"'
            f'{tuples} format="appended" offset="{offset}"/>'
        )
        blocks.append(block)
        offset += len(block)
    field = (
        "    <FieldData>\n" + "\n".join(xml["FieldData"]) + "\n    </FieldData>\n"
        if xml["FieldData"]
        else ""
    )
    compressor = ' compressor="vtkZLibDataCompressor"' if compress else ""
    header = (
        '<?xml version="1.0"?>\n'
        f'<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
        f'header_type="UInt64"{compressor}>\n'
        "  <UnstructuredGrid>\n"
        + field
        + f'    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{n_cells}">\n'
        "      <Points>\n" + "\n".join(xml["Points"]) + "\n      </Points>\n"
        "      <Cells>\n" + "\n".join(xml["Cells"]) + "\n      </Cells>\n"
        "      <CellData>\n" + "\n".join(xml["CellData"]) + "\n      </CellData>\n"
        "      <PointData>\n" + "\n".join(xml["PointData"]) + "\n      </PointData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        '  <AppendedData encoding="raw">\n   _'
    )
    path = Path(path)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for block in blocks:
            f.write(block)
        f.write(b"\n  </AppendedData>\n</VTKFile>\n")
    return path


def read_vtu(path) -> dict:
    """Minimal reader of the files written by :func:`write_vtu` (points, cells, data)."""
    root, appended, encoding, header, compressed = _read_xml_arrays(Path(path))
    piece = root.find("UnstructuredGrid").find("Piece")
    n_points = int(piece.get("NumberOfPoints"))
    n_cells = int(piece.get("NumberOfCells"))

    def values(el, count):
        return _array_values(el, count, appended, encoding, header, compressed)

    pts = values(piece.find("Points").find("DataArray"), n_points).reshape(n_points, 3)
    cell_arrays = {el.get("Name"): el for el in piece.find("Cells").findall("DataArray")}
    offsets = values(cell_arrays["offsets"], n_cells)
    conn = values(cell_arrays["connectivity"], int(offsets[-1]) if n_cells else 0)
    types = values(cell_arrays["types"], n_cells)
    out = {}
    for section, count in (("CellData", n_cells), ("PointData", n_points)):
        data = {}
        node = piece.find(section)
        if node is not None:
            for el in node.findall("DataArray"):
                v = values(el, count)
                ncomp = int(el.get("NumberOfComponents", "1"))
                data[el.get("Name")] = v.reshape(count, ncomp) if ncomp > 1 else v
        out[section] = data
    fields = {}
    node = root.find("UnstructuredGrid").find("FieldData")
    if node is not None:
        for el in node.findall("DataArray"):
            fields[el.get("Name")] = values(el, int(el.get("NumberOfTuples", "1")))
    per = int(offsets[0]) if n_cells else 0
    return {
        "points": pts,
        "cells": conn.reshape(n_cells, per),
        "types": types,
        "cell_data": out["CellData"],
        "point_data": out["PointData"],
        "field_data": fields,
    }


#: The number of nodes of the cell types of the legacy files (VTK's numbers)
_CELL_NODES = {1: 1, 3: 2, 5: 3, 9: 4, 10: 4, 12: 8, 13: 6, 14: 5, 22: 6, 23: 8, 24: 10, 25: 20}


def read_vtk_grid(path) -> dict:
    """A legacy VTK file of an unstructured grid (ASCII or binary; the cells as counts
    and nodes, or as the ``OFFSETS`` and ``CONNECTIVITY`` of VTK 5.1), e.g. an
    animation of OpenRadioss converted by its ``anim_to_vtk``: ``points``, ``offsets``
    (n + 1), ``connectivity``, ``types``, ``cell_data`` and ``point_data`` (name ->
    values: ``SCALARS`` of one or more components, ``VECTORS``, ``TENSORS`` (n, 9),
    ``NORMALS`` and the arrays of ``FIELD`` blocks) and ``field_data`` (the ``FIELD``
    arrays given before the data of the points and cells). :func:`cells_of_type` picks
    the cells of one type."""
    content = Path(path).read_bytes()
    pos = 0
    size = len(content)

    def line() -> str:
        nonlocal pos
        end = content.find(b"\n", pos)
        end = size if end < 0 else end
        text = content[pos:end].decode("latin-1").strip()
        pos = end + 1
        return text

    def tokens(n: int, dtype) -> np.ndarray:
        """``n`` values: ASCII tokens over as many lines as needed, or big-endian binary."""
        nonlocal pos
        dtype = np.dtype(dtype)
        if binary:
            values = np.frombuffer(content, dtype=dtype.newbyteorder(">"), count=n, offset=pos)
            pos += n * dtype.itemsize
            if pos < size and content[pos : pos + 1] == b"\n":
                pos += 1
            return values.astype(dtype)
        found: list[bytes] = []
        while len(found) < n:
            if pos >= size:
                raise ValueError(f"{path}: the file ends inside an array")
            end = content.find(b"\n", pos)
            end = size if end < 0 else end
            found.extend(content[pos:end].split())
            pos = end + 1
        if len(found) > n:
            raise ValueError(f"{path}: more values on a line than an array holds")
        if dtype.kind in "iu":
            return np.array([int(float(v)) for v in found], dtype=dtype)
        return np.array([float(v) for v in found], dtype=dtype)

    if not line().lower().startswith("# vtk datafile"):
        raise ValueError(f"{path}: not a legacy VTK file")
    line()  # the title
    mode = line().upper()
    if mode not in ("ASCII", "BINARY"):
        raise ValueError(f"{path}: ASCII or BINARY, got {mode!r}")
    binary = mode == "BINARY"
    text = line()
    while not text and pos < size:
        text = line()
    if text.upper().split() != ["DATASET", "UNSTRUCTURED_GRID"]:
        raise ValueError(f"{path}: not an UNSTRUCTURED_GRID dataset ({text!r})")

    def dtype_of(name: str):
        name = name.lower()
        legacy = {**_FROM_LEGACY, "vtktypeint64": np.dtype("int64"),
                  "vtktypeint32": np.dtype("int32"), "vtkidtype": np.dtype("int64")}  # fmt: skip
        if name not in legacy:
            raise ValueError(f"{path}: unknown data type {name!r}")
        return legacy[name]

    points = np.zeros((0, 3))
    offsets = np.zeros(1, dtype=np.int64)
    connectivity = np.zeros(0, dtype=np.int64)
    types = np.zeros(0, dtype=np.int64)
    data = {"FIELD": {}, "CELL_DATA": {}, "POINT_DATA": {}}
    section, count = "FIELD", 0

    def read_field(target: dict, n_arrays: int) -> None:
        for _ in range(n_arrays):
            head = line()
            while not head and pos < size:
                head = line()
            name, ncomp, ntuples, kind = head.split()[:4]
            values = tokens(int(ncomp) * int(ntuples), dtype_of(kind))
            target[name] = values.reshape(int(ntuples), int(ncomp)) if int(ncomp) > 1 else values

    while pos < size:
        text = line()
        if not text:
            continue
        key, *vals = text.split()
        key = key.upper()
        if key == "POINTS":
            n = int(vals[0])
            points = tokens(3 * n, dtype_of(vals[1])).astype(float).reshape(n, 3)
        elif key == "CELLS":
            n, total = int(vals[0]), int(vals[1])
            nxt = content[pos : pos + 7].upper()
            if nxt.startswith(b"OFFSETS"):  # VTK 5.1: offsets then connectivity
                kind = line().split()[1]
                offsets = tokens(n, dtype_of(kind)).astype(np.int64)
                kind = line().split()[1]
                connectivity = tokens(total, dtype_of(kind)).astype(np.int64)
            else:  # the counts and nodes of each cell (32-bit integers in binary files)
                flat = tokens(total, np.int32).astype(np.int64)
                counts, k = [], 0
                cells = []
                while k < len(flat):
                    m = int(flat[k])
                    counts.append(m)
                    cells.append(flat[k + 1 : k + 1 + m])
                    k += m + 1
                if len(counts) != n:
                    raise ValueError(f"{path}: {len(counts)} cells, the header says {n}")
                offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
                connectivity = np.concatenate(cells).astype(np.int64) if cells else connectivity
        elif key == "CELL_TYPES":
            types = tokens(int(vals[0]), np.int32).astype(np.int64)
        elif key in ("CELL_DATA", "POINT_DATA"):
            section, count = key, int(vals[0])
        elif key == "FIELD":
            read_field(data[section], int(vals[1]))
        elif key in ("SCALARS", "VECTORS", "NORMALS", "TENSORS"):
            name, kind = vals[0], vals[1]
            ncomp = {"VECTORS": 3, "NORMALS": 3, "TENSORS": 9}.get(key)
            if key == "SCALARS":
                ncomp = int(vals[2]) if len(vals) > 2 else 1
                if content[pos : pos + 12].upper().startswith(b"LOOKUP_TABLE"):
                    line()
            values = tokens(count * ncomp, dtype_of(kind))
            data[section][name] = values.reshape(count, ncomp) if ncomp > 1 else values
        elif key == "LOOKUP_TABLE" and len(vals) > 1:  # a colour table: skipped
            tokens(4 * int(vals[1]), np.float32 if not binary else np.uint8)
        elif key == "METADATA":
            while pos < size and line():
                pass
        else:
            raise ValueError(f"{path}: unknown keyword {key!r}")
    if len(offsets) != len(types) + 1:
        if len(offsets) == len(types) and len(types):  # offsets given without the start
            offsets = np.concatenate([[0], offsets])
        elif len(types) or len(offsets) > 1:
            raise ValueError(f"{path}: {len(types)} cell types for {len(offsets) - 1} cells")
    return {
        "points": points,
        "offsets": offsets,
        "connectivity": connectivity,
        "types": types,
        "cell_data": data["CELL_DATA"],
        "point_data": data["POINT_DATA"],
        "field_data": data["FIELD"],
    }


def cells_of_type(grid: dict, cell_type: int) -> tuple[np.ndarray, np.ndarray]:
    """The cells of one VTK type of a grid of :func:`read_vtk_grid`: their nodes (n, k)
    and their indices among the cells (for their data)."""
    types = np.asarray(grid["types"])
    picked = np.flatnonzero(types == int(cell_type))
    offsets = np.asarray(grid["offsets"])
    if not len(picked):
        k = _CELL_NODES.get(int(cell_type), 0)
        return np.zeros((0, k), dtype=np.int64), picked
    starts, ends = offsets[picked], offsets[picked + 1]
    widths = np.unique(ends - starts)
    if len(widths) != 1:
        raise ValueError(f"cells of type {cell_type} with {widths.tolist()} nodes")
    k = int(widths[0])
    index = starts[:, None] + np.arange(k)[None, :]
    return np.asarray(grid["connectivity"])[index], picked
