"""PNG images and colour maps without plotting libraries (used by the GUI and the CLI).

``to_rgb`` turns a 2D array into colours: phase indices with a categorical palette,
real fields with a perceptual colour map (viridis-like); ``encode_png`` writes 8-bit RGB
PNG data (zlib, standard library only).
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

#: Colours of the phases 0, 1, 2... (0: light background, 1: dark blue...).
PALETTE = np.array(
    [
        [236, 236, 230],
        [31, 78, 121],
        [214, 96, 77],
        [91, 155, 94],
        [237, 177, 32],
        [126, 87, 160],
        [77, 175, 214],
        [140, 86, 75],
    ],
    dtype=np.uint8,
)

# viridis at nine stops (interpolated linearly in between)
_VIRIDIS = np.array(
    [
        [68, 1, 84],
        [71, 44, 122],
        [59, 81, 139],
        [44, 113, 142],
        [33, 144, 141],
        [39, 173, 129],
        [92, 200, 99],
        [170, 220, 50],
        [253, 231, 37],
    ],
    dtype=float,
)


def colormap(values, vmin=None, vmax=None) -> np.ndarray:
    """RGB (uint8) of real values with a viridis-like map."""
    v = np.asarray(values, dtype=float)
    lo = np.nanmin(v) if vmin is None else float(vmin)
    hi = np.nanmax(v) if vmax is None else float(vmax)
    t = np.zeros_like(v) if hi <= lo else np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    t = np.nan_to_num(t)
    pos = t * (len(_VIRIDIS) - 1)
    i = np.minimum(pos.astype(int), len(_VIRIDIS) - 2)
    f = (pos - i)[..., None]
    return np.rint((1 - f) * _VIRIDIS[i] + f * _VIRIDIS[i + 1]).astype(np.uint8)


def label_colors(labels) -> np.ndarray:
    """RGB of integer labels (grains): a pseudo-random colour per label, black for -1."""
    lab = np.asarray(labels, dtype=np.int64)
    h = (lab * 2654435761) % (2**32)
    rgb = np.stack([(h >> s) & 255 for s in (0, 8, 16)], axis=-1).astype(np.uint8)
    rgb = (rgb // 2 + 96).astype(np.uint8)
    rgb[lab < 0] = 0
    return rgb


def to_rgb(image2d, kind: str = "auto", vmin=None, vmax=None) -> np.ndarray:
    """Colours of a 2D array: ``phase`` (palette), ``labels`` or ``field`` (colour map);
    ``auto`` picks phases for small non-negative integers and a colour map otherwise."""
    a = np.asarray(image2d)
    if a.ndim != 2:
        raise ValueError("to_rgb: a 2D array")
    if kind == "auto":
        small = a.dtype.kind in "biu" and a.size and a.min() >= 0 and a.max() < len(PALETTE)
        kind = "phase" if small else "field"
    if kind == "phase":
        return PALETTE[np.clip(a.astype(np.int64), 0, len(PALETTE) - 1)]
    if kind == "labels":
        return label_colors(a)
    return colormap(a, vmin, vmax)


def upscale(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour enlargement (crisp voxels in small images)."""
    f = max(1, int(factor))
    return np.repeat(np.repeat(rgb, f, axis=0), f, axis=1)


def encode_png(rgb) -> bytes:
    """PNG bytes of an RGB (h, w, 3) or grey (h, w) uint8 array."""
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("encode_png: an RGB image (h, w, 3)")
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def decode_png(data: bytes) -> np.ndarray:
    """RGB array of PNG bytes written by :func:`encode_png` (for tests)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    idat = b""
    w = h = 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    rows = [raw[y * (3 * w + 1) + 1 : (y + 1) * (3 * w + 1)] for y in range(h)]
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(h, w, 3)


def slice_of(volume, axis: int = 2, index: int | None = None) -> np.ndarray:
    """A 2D section of a 3D array (middle section by default), displayed with the first
    remaining axis horizontal and the second vertical (rows from the top)."""
    v = np.asarray(volume)
    if v.ndim != 3:
        raise ValueError("slice_of: a 3D array")
    i = v.shape[axis] // 2 if index is None else int(index)
    s = np.take(v, i, axis=axis)
    return np.flipud(s.T)
