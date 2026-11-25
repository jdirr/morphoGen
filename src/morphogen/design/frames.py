"""Periodic pin-jointed frameworks: strut lattices with defects, and their rigidity.

A framework is a set of nodes in a periodic box and of bars joining pairs of nodes, the
second end possibly in a neighbouring copy of the box (``shift``). Lattices are built
from the unit cells of :mod:`morphogen.models.lattice`; defects are struts removed at
random (dilution) and nodes displaced at random (perturbation).

Rigidity (Maxwell 1864, Calladine 1978, Pellegrino and Calladine 1986; periodic
frameworks: Guest and Hutchinson 2003). The compatibility matrix ``A`` maps periodic
node displacements to bar extensions, and ``B`` maps a macroscopic strain ``E`` to the
extensions ``n . E L`` it imposes on the bars. Then

* mechanisms (periodic displacement modes extending no bar, rigid translations
  excluded): ``m = 3 j - 3 - rank(A)``;
* states of self-stress (bar tensions in equilibrium without load):
  ``s = b - rank(A)``, so that ``m - s = 3 j - 3 - b`` (Maxwell-Calladine);
* floppy macroscopic strains (accommodated by some displacement without extending any
  bar): ``6 - rank([A B]) + rank(A)``.

A lattice whose macroscopic strains all stretch bars is stretching-dominated: its
stiffness scales with the relative density, against its square for bending-dominated
lattices (Deshpande, Ashby and Fleck 2001). Maxwell's count requires a mean coordination
of at least 6 in 3D for large frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from morphogen.primitives import Segments


@dataclass
class Framework:
    """Nodes in a periodic box and bars from ``nodes[p]`` to ``nodes[q] + shift * box``."""

    nodes: np.ndarray
    bars: np.ndarray
    shifts: np.ndarray
    box: np.ndarray

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float).reshape(-1, 3)
        self.bars = np.asarray(self.bars, dtype=np.int64).reshape(-1, 2)
        self.shifts = np.asarray(self.shifts, dtype=np.int64).reshape(-1, 3)
        self.box = np.asarray(self.box, dtype=float).reshape(3)
        if len(self.bars) != len(self.shifts):
            raise ValueError("bars and shifts: same length")
        if len(self.bars) and (self.bars.min() < 0 or self.bars.max() >= len(self.nodes)):
            raise ValueError("bars: node indices out of range")

    # -- construction --------------------------------------------------------------------

    @classmethod
    def lattice(cls, cell: str, n_cells=1, cell_size: float = 1.0) -> Framework:
        """The periodic lattice of ``n_cells`` unit cells (cubic, bcc, fcc, octet, kelvin,
        diamond or pyrochlore: :data:`morphogen.models.lattice.CELLS`)."""
        from morphogen.models.lattice import CELLS, unit_cell_struts  # (models import us)

        if cell not in CELLS:
            raise ValueError(f"cell: one of {CELLS}")
        n = np.broadcast_to(np.asarray(n_cells, dtype=int), (3,))
        if np.any(n < 1):
            raise ValueError("n_cells: at least one cell along each axis")
        unit = unit_cell_struts(cell)  # fractional coordinates, (s, 2, 3)
        offsets = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1)
        ends = (offsets.reshape(-1, 1, 1, 3) + unit[None]).reshape(-1, 2, 3)
        # the nodes of the cells are multiples of 1/8 of their edge (1/2 for the cubic
        # lattices, 1/4 for the Kelvin and diamond cells, 1/8 for the pyrochlore)
        keys = np.mod(np.rint(ends * 8).astype(np.int64), 8 * n)
        wrapped = keys / 8.0
        flat = keys.reshape(-1, 3)
        uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
        ids = inverse.reshape(-1, 2)
        shifts = np.rint((ends[:, 1] - wrapped[:, 1]) / n).astype(np.int64) - np.rint(
            (ends[:, 0] - wrapped[:, 0]) / n
        ).astype(np.int64)
        frame = cls(uniq / 8.0 * cell_size, ids, shifts, n * float(cell_size))
        return frame._deduplicated()

    def _deduplicated(self) -> Framework:
        """Remove repeated bars (a bar p-q with shift s is the bar q-p with shift -s)."""
        p, q = self.bars[:, 0], self.bars[:, 1]
        s = self.shifts
        # orient each bar canonically: p < q, or p == q with a positive first non-zero shift
        first = np.argmax(s != 0, axis=1)
        neg = s[np.arange(len(s)), first] < 0
        flip = (p > q) | ((p == q) & neg)
        pp = np.where(flip, q, p)
        qq = np.where(flip, p, q)
        ss = np.where(flip[:, None], -s, s)
        key = np.column_stack([pp, qq, ss])
        _, idx = np.unique(key, axis=0, return_index=True)
        idx.sort()
        keep = idx[~((pp[idx] == qq[idx]) & np.all(ss[idx] == 0, axis=1))]  # no zero bars
        return Framework(self.nodes, np.column_stack([pp, qq])[keep], ss[keep], self.box)

    # -- description ---------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_bars(self) -> int:
        return len(self.bars)

    def degrees(self) -> np.ndarray:
        return np.bincount(self.bars.ravel(), minlength=self.n_nodes)

    def coordination(self) -> float:
        """Mean number of bars per node, ``Z = 2 b / j``."""
        return 2.0 * self.n_bars / self.n_nodes if self.n_nodes else 0.0

    def vectors(self) -> np.ndarray:
        """Bar vectors ``(b, 3)``."""
        p, q = self.bars[:, 0], self.bars[:, 1]
        return self.nodes[q] + self.shifts * self.box - self.nodes[p]

    def components(self) -> np.ndarray:
        """Connected component of every node (through the periodic boundaries)."""
        p, q = self.bars[:, 0], self.bars[:, 1]
        graph = coo_matrix((np.ones(len(p)), (p, q)), shape=(self.n_nodes, self.n_nodes))
        return connected_components(graph, directed=False)[1]

    # -- transformations -----------------------------------------------------------------

    def with_bars(self, keep) -> Framework:
        """The framework with the selected bars only (same nodes)."""
        keep = np.asarray(keep)
        return Framework(self.nodes, self.bars[keep], self.shifts[keep], self.box)

    def without_isolated_nodes(self) -> Framework:
        used = np.zeros(self.n_nodes, dtype=bool)
        used[self.bars.ravel()] = True
        index = np.cumsum(used) - 1
        return Framework(self.nodes[used], index[self.bars], self.shifts, self.box)

    def pruned(self, min_degree: int = 2) -> Framework:
        """Remove, repeatedly, the bars at nodes of fewer than ``min_degree`` bars."""
        frame = self
        while frame.n_bars:
            deg = frame.degrees()
            weak = deg < min_degree
            bad = weak[frame.bars[:, 0]] | weak[frame.bars[:, 1]]
            if not bad.any():
                break
            frame = frame.with_bars(~bad)
        return frame.without_isolated_nodes()

    def largest_component(self) -> Framework:
        if self.n_bars == 0:
            return self.without_isolated_nodes()
        comp = self.components()
        sizes = np.bincount(comp[self.bars[:, 0]], minlength=comp.max() + 1)
        keep = comp[self.bars[:, 0]] == np.argmax(sizes)
        return self.with_bars(keep).without_isolated_nodes()

    def diluted(
        self,
        rng: np.random.Generator,
        *,
        coordination: float | None = None,
        fraction: float | None = None,
        min_degree: int = 3,
    ) -> Framework:
        """Struts removed at random.

        With ``coordination``: until the mean coordination reaches the target, never
        below ``min_degree`` bars at a node (the target may not be reached). With
        ``fraction``: that fraction of the bars, then dangling bars are pruned. The
        largest connected component is kept.
        """
        if (coordination is None) == (fraction is None):
            raise ValueError("give exactly one of 'coordination' and 'fraction'")
        if fraction is not None:
            if not 0.0 <= fraction < 1.0:
                raise ValueError("fraction: in [0, 1)")
            keep = np.ones(self.n_bars, dtype=bool)
            keep[rng.permutation(self.n_bars)[: round(fraction * self.n_bars)]] = False
            return self.with_bars(keep).pruned(2).largest_component()
        target = round(float(coordination) * self.n_nodes / 2.0)
        deg = self.degrees().copy()
        keep = np.ones(self.n_bars, dtype=bool)
        n_bars = self.n_bars
        for k in rng.permutation(self.n_bars):
            if n_bars <= target:
                break
            p, q = self.bars[k]
            need = 2 if p == q else 1  # a self-bar counts twice at its node
            if deg[p] - need >= min_degree and deg[q] - need >= min_degree:
                keep[k] = False
                deg[p] -= 1
                deg[q] -= 1
                n_bars -= 1
        return self.with_bars(keep).largest_component()

    def perturbed(self, rng: np.random.Generator, amplitude: float) -> Framework:
        """Nodes displaced uniformly in balls of radius ``amplitude``."""
        v = rng.standard_normal((self.n_nodes, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        r = amplitude * rng.random(self.n_nodes) ** (1.0 / 3.0)
        return Framework(self.nodes + v * r[:, None], self.bars, self.shifts, self.box)

    def segments(self, radius, origin=(0.0, 0.0, 0.0)) -> Segments:
        """Struts as round-capped cylinders of ``radius`` (one value, or one per bar;
        periodic images are added when needed)."""
        start = self.nodes[self.bars[:, 0]] + np.asarray(origin, dtype=float)
        if np.ndim(radius) == 0:
            radii = np.full(self.n_bars, float(radius))
        else:
            radii = np.asarray(radius, dtype=float).reshape(-1)
            if radii.shape != (self.n_bars,):
                raise ValueError(f"segments: one radius or {self.n_bars} (one per bar)")
        return Segments(start, start + self.vectors(), radii, round_caps=True)

    # -- rigidity ------------------------------------------------------------------------

    def compatibility(self) -> tuple[np.ndarray, np.ndarray]:
        """Dense matrices ``A (b, 3 j)`` and ``B (b, 6)`` (strains 11, 22, 33, 23, 13, 12,
        shear components counted twice)."""
        v = self.vectors()
        length = np.linalg.norm(v, axis=1)
        if np.any(length <= 0):
            raise ValueError("bars of zero length")
        n = v / length[:, None]
        a = np.zeros((self.n_bars, 3 * self.n_nodes))
        rows = np.arange(self.n_bars)
        for d in range(3):
            np.add.at(a, (rows, 3 * self.bars[:, 1] + d), n[:, d])
            np.add.at(a, (rows, 3 * self.bars[:, 0] + d), -n[:, d])
        b = np.column_stack(
            [
                n[:, 0] * v[:, 0],
                n[:, 1] * v[:, 1],
                n[:, 2] * v[:, 2],
                n[:, 1] * v[:, 2] + n[:, 2] * v[:, 1],
                n[:, 0] * v[:, 2] + n[:, 2] * v[:, 0],
                n[:, 0] * v[:, 1] + n[:, 1] * v[:, 0],
            ]
        )
        return a, b / np.mean(length)

    def rigidity(self, tol: float = 1e-9) -> dict:
        """Mechanisms, states of self-stress and floppy macroscopic strains (SVD ranks)."""
        a, b = self.compatibility()

        def rank(m):
            if m.size == 0:
                return 0
            s = np.linalg.svd(m, compute_uv=False)
            return int(np.sum(s > tol * max(1.0, s[0])))

        ra = rank(a)
        rab = rank(np.hstack([a, b]))
        mechanisms = 3 * self.n_nodes - 3 - ra
        floppy = 6 - (rab - ra)
        return {
            "n_nodes": self.n_nodes,
            "n_bars": self.n_bars,
            "coordination": self.coordination(),
            "mechanisms": int(mechanisms),
            "self_stresses": int(self.n_bars - ra),
            "floppy_strains": int(floppy),
            "stretching_dominated": bool(floppy == 0),
        }

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes.tolist(),
            "bars": self.bars.tolist(),
            "shifts": self.shifts.tolist(),
            "box": self.box.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Framework:
        return cls(data["nodes"], data["bars"], data["shifts"], data["box"])
