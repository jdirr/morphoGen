"""Tile sets for combinatorial design: cubic building blocks with labelled faces.

A tile is a cubic block of ``k^3`` voxels (its geometry) whose six faces carry integer
labels. Two tiles can be neighbours along an axis when the labels of the faces in
contact match: equal labels (``match="equal"``, Wang tiles), or opposite labels
(``match="opposite"``: a face that moves out against a face that moves in). An
assembly is an array of tile indices; it is compatible when every pair of neighbouring
tiles matches. Faces are numbered ``-x, +x, -y, +y, -z, +z`` (face ``2 a`` is the lower
face along axis ``a``, ``2 a + 1`` the upper one). Planar tile sets (``dims=2``) ignore
the z faces: their assemblies have one tile through the thickness.

Built-in sets:

``polarized_blocks``
    The building blocks of combinatorial mechanical metamaterials (after Coulais et al.
    2016): each block has a single deformation mode in which it elongates along its axis
    and shrinks along the two others, or the reverse. Its faces move out or in
    accordingly; neighbouring blocks are compatible when their common face moves the
    same way (out of one block, into the other). Six tiles: three axes, two signs.
``pipe_tiles``
    Nodes with channels (or struts) towards some of their faces; compatible
    assemblies are networks without dead ends at the tile boundaries, with controlled
    node degrees.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

FACES = ("-x", "+x", "-y", "+y", "-z", "+z")
MATCHES = ("equal", "opposite")


def _signed_permutations(dims: int) -> list[np.ndarray]:
    """Proper rotations of the cube (24), or the 4 rotations about z (dims=2), as
    signed permutation matrices."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=int)
            for j in range(3):
                m[j, perm[j]] = signs[j]
            if round(np.linalg.det(m)) != 1:
                continue
            if dims == 2 and m[2, 2] != 1:
                continue
            out.append(m)
    return out


def _rotate_faces(faces: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Face labels after the rotation m: the face of direction m e carries the label of
    the face of direction e."""
    out = np.empty_like(faces)
    for a in range(3):
        j = int(np.flatnonzero(m[:, a])[0])
        s = int(m[j, a])
        for side in (0, 1):  # 0: lower (-), 1: upper (+)
            sigma = -1 if side == 0 else 1
            new_side = 0 if sigma * s < 0 else 1
            out[..., 2 * j + new_side] = faces[..., 2 * a + side]
    return out


def _rotate_block(block: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Voxels of a block after the rotation m (about the centre of the block)."""
    perm = [int(np.flatnonzero(m[j])[0]) for j in range(3)]
    out = np.transpose(block, perm)
    for j in range(3):
        if m[j, perm[j]] < 0:
            out = np.flip(out, axis=j)
    return np.ascontiguousarray(out)


@dataclass
class TileSet:
    """Tiles with labelled faces, their geometry and their weights.

    Parameters
    ----------
    faces
        Integer labels ``(n, 6)`` of the faces ``-x, +x, -y, +y, -z, +z``.
    match
        ``equal`` or ``opposite``: rule for the labels of faces in contact.
    voxels
        Optional geometry ``(n, k, k, k)`` (``(n, k, k, 1)`` for planar sets): phase of
        each voxel of each tile.
    weights
        Relative frequencies of the tiles in random choices (default: equal).
    names
        Tile names.
    dims
        3, or 2 for planar tile sets (z faces ignored).
    """

    faces: np.ndarray
    match: str = "equal"
    voxels: np.ndarray | None = None
    weights: np.ndarray | None = None
    names: tuple[str, ...] | None = None
    dims: int = 3
    _compat: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 6)
        n = len(self.faces)
        if n < 1:
            raise ValueError("a tile set needs at least one tile")
        if self.match not in MATCHES:
            raise ValueError(f"match: one of {MATCHES}, got {self.match!r}")
        if self.dims not in (2, 3):
            raise ValueError("dims: 2 or 3")
        if self.voxels is not None:
            self.voxels = np.ascontiguousarray(self.voxels, dtype=np.uint8)
            if self.voxels.ndim != 4 or len(self.voxels) != n:
                raise ValueError("voxels: an array (n_tiles, k, k, k)")
            k = self.voxels.shape[1]
            expected = (k, k, 1) if self.dims == 2 else (k, k, k)
            if self.voxels.shape[1:] != expected:
                raise ValueError(f"voxels: blocks of shape {expected}")
        w = np.ones(n) if self.weights is None else np.asarray(self.weights, dtype=float)
        if w.shape != (n,) or np.any(w < 0) or not np.any(w > 0):
            raise ValueError("weights: one non-negative weight per tile, not all zero")
        self.weights = w
        if self.names is None:
            self.names = tuple(f"t{i}" for i in range(n))
        self.names = tuple(str(s) for s in self.names)
        if len(self.names) != n:
            raise ValueError("names: one per tile")

    @property
    def n_tiles(self) -> int:
        return len(self.faces)

    @property
    def tile_size(self) -> int | None:
        """Voxels per tile edge (None without geometry)."""
        return None if self.voxels is None else int(self.voxels.shape[1])

    def label_matches(self, upper, lower):
        """Whether an upper face label (of the lower tile) matches a lower face label."""
        upper = np.asarray(upper)
        lower = np.asarray(lower)
        return upper == lower if self.match == "equal" else upper == -lower

    def compatibility(self) -> np.ndarray:
        """Boolean array ``(3, n, n)``: ``[a, A, B]`` when tile B may follow tile A along
        axis a (planar sets: every pair along z)."""
        if self._compat is None:
            n = self.n_tiles
            c = np.ones((3, n, n), dtype=bool)
            for a in range(self.dims):
                c[a] = self.label_matches(
                    self.faces[:, None, 2 * a + 1], self.faces[None, :, 2 * a]
                )
            self._compat = c
        return self._compat

    def boundary_tiles(self, axis: int, side: int, label: int) -> np.ndarray:
        """Tiles whose face (axis, side) matches a boundary of the given label."""
        face = self.faces[:, 2 * axis + side]
        if side == 1:
            return self.label_matches(face, label)
        return self.label_matches(label, face)

    def mismatches(self, assembly, periodic: bool = True) -> int:
        """Number of incompatible pairs of neighbouring tiles."""
        t = np.asarray(assembly, dtype=np.int64)
        if t.ndim != 3:
            raise ValueError("assembly: a 3D array of tile indices ((n0, n1, 1) in 2D)")
        compat = self.compatibility()
        bad = 0
        for a in range(self.dims):
            if t.shape[a] < 2 and not periodic:
                continue
            if periodic:
                nxt = np.roll(t, -1, axis=a)
                bad += int(np.count_nonzero(~compat[a][t, nxt]))
            else:
                lo = np.take(t, range(t.shape[a] - 1), axis=a)
                hi = np.take(t, range(1, t.shape[a]), axis=a)
                bad += int(np.count_nonzero(~compat[a][lo, hi]))
        return bad

    def is_compatible(self, assembly, periodic: bool = True) -> bool:
        return self.mismatches(assembly, periodic) == 0

    def assemble(self, assembly) -> np.ndarray:
        """Voxels of an assembly: the blocks of its tiles side by side."""
        if self.voxels is None:
            raise ValueError("this tile set has no geometry (voxels)")
        t = np.asarray(assembly, dtype=np.int64)
        n0, n1, n2 = t.shape
        blocks = self.voxels[t]  # (n0, n1, n2, k0, k1, k2)
        k0, k1, k2 = self.voxels.shape[1:]
        return np.ascontiguousarray(
            blocks.transpose(0, 3, 1, 4, 2, 5).reshape(n0 * k0, n1 * k1, n2 * k2)
        )

    def subset(self, keep) -> TileSet:
        """The tiles selected by a boolean mask or an index list."""
        idx = np.arange(self.n_tiles)[np.asarray(keep)]
        if idx.size == 0:
            raise ValueError("subset: no tile left")
        return TileSet(
            self.faces[idx],
            self.match,
            None if self.voxels is None else self.voxels[idx],
            self.weights[idx],
            tuple(self.names[i] for i in idx),
            self.dims,
        )

    def with_rotations(self) -> TileSet:
        """The set closed under the rotations of the cube (about z for planar sets),
        without duplicates; rotated tiles keep the weight of their original."""
        faces, voxels, weights, names = [], [], [], []
        seen = set()
        for i in range(self.n_tiles):
            for r, m in enumerate(_signed_permutations(self.dims)):
                f = _rotate_faces(self.faces[i], m)
                v = None if self.voxels is None else _rotate_block(self.voxels[i], m)
                key = (tuple(f), None if v is None else v.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                faces.append(f)
                voxels.append(v)
                weights.append(self.weights[i])
                names.append(self.names[i] if r == 0 else f"{self.names[i]}/r{r}")
        return TileSet(
            np.array(faces),
            self.match,
            None if self.voxels is None else np.array(voxels),
            np.array(weights),
            tuple(names),
            self.dims,
        )


# -- built-in sets -------------------------------------------------------------------------


def _centred_coordinates(k: int, dims: int) -> np.ndarray:
    """Coordinates of the voxel centres of a block, in units of the block edge, centred."""
    c = (np.arange(k) + 0.5) / k - 0.5
    kz = k if dims == 3 else 1
    cz = c if dims == 3 else np.zeros(1)
    return np.stack(np.meshgrid(c, c, cz, indexing="ij"), axis=-1).reshape(k, k, kz, 3)


def polarized_blocks(size: int = 8, channel: float = 0.25, dims: int = 3) -> TileSet:
    """Blocks with one deformation mode: elongation along their axis and contraction
    along the others (sign +), or the reverse (sign -).

    The faces normal to the axis carry the sign of the mode, the others its opposite
    (+1: the face moves out); faces in contact must move the same way, hence opposite
    labels (``match="opposite"``). Along every line of blocks, the strain along the line
    alternates in sign. The geometry is a solid block pierced along its axis by a
    channel of radius ``channel`` (in block edges), which draws the orientation of the
    block; replace it with the geometry of an actual building block if needed.
    """
    if not 0.0 <= channel < 0.5:
        raise ValueError("channel: a radius in [0, 0.5) block edges")
    axes = range(dims)
    faces, voxels, names = [], [], []
    x = _centred_coordinates(int(size), dims)
    for a in axes:
        r2 = sum(x[..., b] ** 2 for b in range(dims) if b != a)
        block = (r2 > channel**2).astype(np.uint8)
        for s in (1, -1):
            f = np.empty(6, dtype=np.int64)
            for b in range(3):
                f[2 * b] = f[2 * b + 1] = s if b == a else -s
            if dims == 2:
                f[4] = f[5] = 0
            faces.append(f)
            voxels.append(block)
            names.append(f"{'xyz'[a]}{'+' if s > 0 else '-'}")
    return TileSet(np.array(faces), "opposite", np.array(voxels), None, tuple(names), dims)


def pipe_tiles(
    size: int = 8,
    radius: float = 0.2,
    dims: int = 3,
    *,
    open_probability: float = 0.5,
    degrees=None,
) -> TileSet:
    """Nodes joined to some of their faces by straight pipes (phase 1).

    One tile per subset of open faces (64 in 3D, 16 in 2D); open faces carry the label
    1 and must face open faces (``match="equal"``). Each open face independently has
    probability ``open_probability`` (tile weights ``p^m (1 - p)^(2 d - m)`` for ``m``
    open faces); ``degrees`` restricts the numbers of open faces allowed (e.g. no 1 to
    forbid dead ends).
    """
    if not 0.0 < radius < 0.5:
        raise ValueError("radius: in (0, 0.5) block edges")
    if not 0.0 < open_probability < 1.0:
        raise ValueError("open_probability: in (0, 1)")
    n_faces = 2 * dims
    x = _centred_coordinates(int(size), dims)
    faces, voxels, weights, names = [], [], [], []
    for mask in range(2**n_faces):
        open_faces = [f for f in range(n_faces) if mask >> f & 1]
        m = len(open_faces)
        if degrees is not None and m not in set(degrees):
            continue
        f = np.zeros(6, dtype=np.int64)
        block = np.zeros(x.shape[:3], dtype=bool)
        for face in open_faces:
            f[face] = 1
            a, side = divmod(face, 2)
            sign = 1.0 if side else -1.0
            along = x[..., a] * sign
            r2 = sum(x[..., b] ** 2 for b in range(dims) if b != a)
            block |= (along >= -radius) & (r2 <= radius**2)
        if m:
            block |= (x[..., :dims] ** 2).sum(axis=-1) <= radius**2
        faces.append(f)
        voxels.append(block.astype(np.uint8))
        weights.append(open_probability**m * (1.0 - open_probability) ** (n_faces - m))
        names.append("".join(FACES[i] for i in open_faces) or "empty")
    if not faces:
        raise ValueError("degrees: no tile left")
    return TileSet(
        np.array(faces), "equal", np.array(voxels), np.array(weights), tuple(names), dims
    )


TILE_SETS = {"polarized": polarized_blocks, "pipes": pipe_tiles}
