"""Transition rules of binary cellular automata (states 0 and 1).

A rule maps the state of the lattice to the next one. Most rules depend on the number
``n`` of neighbours in state 1 (outer-totalistic rules):

``life`` (``B.../S...``)
    birth of a cell in state 0 when ``n`` is in the birth set, survival of a cell in
    state 1 when ``n`` is in the survival set (Conway's Game of Life is ``B3/S23`` in 2D).
``majority``
    the cell takes the state of the majority of its neighbourhood, itself included
    (voting rule): sharp interfaces of positive curvature retreat.
``anneal``
    twisted majority (Vichniac 1984): the outcome is inverted when the vote is closest
    to a tie. Interfaces then move with their mean curvature, as in the annealing of
    a binary alloy: domains coarsen into bicontinuous patterns.
``conserved``
    majority with a conserved volume fraction: the cells with the highest local
    density of phase 1 (random tie-breaking) take state 1, exactly as many as before.
    Domains coarsen at constant volume fraction (spinodal-like patterns).
``probabilistic``
    birth and survival probabilities depending on ``n``.
``activator_inhibitor``
    Young's (1984) model of Turing patterns: a cell weighs the active cells of a near
    disc or ball (activation) against those of a surrounding annulus or shell
    (inhibition); spots, stripes or labyrinths of a characteristic wavelength emerge.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from morphogen.automata.lattice import kernel_sum, neighbour_counts, offsets
from morphogen.automata.tilings import TilingContext


@dataclass(frozen=True)
class LatticeContext:
    """Neighbourhood and boundary of the lattice on which a rule acts."""

    offsets: np.ndarray
    periodic: bool = True
    boundary: int = 0
    dims: int = 3

    @property
    def n_neighbours(self) -> int:
        return len(self.offsets)

    def counts(self, state: np.ndarray) -> np.ndarray:
        return neighbour_counts(state, self.offsets, periodic=self.periodic, boundary=self.boundary)


class Rule(ABC):
    """Transition rule of a binary automaton."""

    #: whether the volume fraction is conserved (synchronous updates only)
    conserves_fraction = False

    @abstractmethod
    def next_state(
        self, state: np.ndarray, lattice: LatticeContext, rng: np.random.Generator
    ) -> np.ndarray:
        """State after one synchronous update of every cell."""

    def describe(self) -> str:
        return type(self).__name__


def _count_set(text: str) -> set[int]:
    """'5678' (digits) or '13-14,17-19' (numbers and ranges) -> set of counts."""
    text = text.strip()
    if not text:
        return set()
    if re.fullmatch(r"\d+", text) and "," not in text:
        return {int(c) for c in text}  # Life-like notation: one digit per count
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"rule: cannot read the counts {text!r}")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a:
            raise ValueError(f"rule: empty range {part!r}")
        out.update(range(a, b + 1))
    return out


def _format_counts(values) -> str:
    """Counts as Life-like digits ('5678') when all are below 10, otherwise as numbers and
    ranges ('13,15-26'), so that :func:`_count_set` reads them back."""
    v = sorted(values)
    if not v or v[-1] <= 9:
        return "".join(str(c) for c in v)
    runs = [[v[0], v[0]]]
    for c in v[1:]:
        if c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    text = ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
    return f"{text}-{text}" if text.isdigit() else text  # '13' would read as {1, 3}


class LifeRule(Rule):
    """Outer-totalistic rule ``B<birth>/S<survival>``.

    Counts are written as digits (``B5678/S45678``) or as comma-separated numbers and
    ranges for 3D neighbourhoods (``B13-14,17-19/S13-26``).
    """

    def __init__(self, birth, survival):
        self.birth = frozenset(int(b) for b in birth)
        self.survival = frozenset(int(s) for s in survival)

    @classmethod
    def parse(cls, text: str) -> LifeRule:
        m = re.fullmatch(r"\s*B([\d,\- ]*)\s*/\s*S([\d,\- ]*)\s*", text, re.I)
        if not m:
            m2 = re.fullmatch(r"\s*S([\d,\- ]*)\s*/\s*B([\d,\- ]*)\s*", text, re.I)
            if not m2:
                raise ValueError(
                    f"rule: expected 'B<counts>/S<counts>' (e.g. B5678/S45678 or "
                    f"B13-14,17-19/S13-26), got {text!r}"
                )
            return cls(_count_set(m2.group(2)), _count_set(m2.group(1)))
        return cls(_count_set(m.group(1)), _count_set(m.group(2)))

    def _lut(self, values, n: int) -> np.ndarray:
        lut = np.zeros(n + 1, dtype=np.uint8)
        for v in values:
            if 0 <= v <= n:
                lut[v] = 1
        return lut

    def next_state(self, state, lattice, rng):
        n = int(np.max(lattice.n_neighbours))
        c = lattice.counts(state)
        born = self._lut(self.birth, n)[c]
        kept = self._lut(self.survival, n)[c]
        return np.where(state == 1, kept, born).astype(np.uint8)

    def describe(self) -> str:
        return f"B{_format_counts(self.birth)}/S{_format_counts(self.survival)}"


class MajorityRule(Rule):
    """Majority vote over the neighbourhood and the cell itself; ``twist`` inverts the
    outcome of the two votes closest to a tie (Vichniac's annealing rule).

    With an odd number of neighbours (triangles across their edges, some Penrose rhombi
    across their vertices), the cell's own vote counts twice, so that the number of votes
    stays odd: ties keep the state of the cell.
    """

    def __init__(self, twist: bool = False):
        self.twist = bool(twist)

    def next_state(self, state, lattice, rng):
        n = lattice.n_neighbours
        own = 1 + np.asarray(n) % 2  # 1 on the grid (symmetric neighbourhoods)
        total = lattice.counts(state) + own * state
        m = n + own  # odd
        half = m // 2
        out = (total > half).astype(np.uint8)
        if self.twist:
            out[total == half] = 1
            out[total == half + 1] = 0
        return out

    def describe(self) -> str:
        return "anneal" if self.twist else "majority"


def _top_fraction(score: np.ndarray, n_ones: int) -> np.ndarray:
    """Binary array with ones at the ``n_ones`` largest scores."""
    flat = score.ravel()
    out = np.zeros(flat.size, dtype=np.uint8)
    if n_ones >= flat.size:
        out[:] = 1
    elif n_ones > 0:
        out[np.argpartition(-flat, n_ones - 1)[:n_ones]] = 1
    return out.reshape(score.shape)


class ConservedMajorityRule(Rule):
    """Majority with a conserved number of cells in state 1 (``fraction`` of the cells):
    the cells of highest local density of phase 1 win, ties broken at random (the density
    is the fraction of phase 1 in the neighbourhood and the cell when neighbourhoods
    differ in size)."""

    conserves_fraction = True

    def __init__(self, fraction: float):
        if not 0.0 < float(fraction) < 1.0:
            raise ValueError(f"fraction: must lie in (0, 1), got {fraction!r}")
        self.fraction = float(fraction)

    def n_ones(self, n_cells: int) -> int:
        return round(self.fraction * n_cells)

    def next_state(self, state, lattice, rng):
        n = lattice.n_neighbours
        if np.ndim(n) == 0:
            score = lattice.counts(state) + state + rng.random(state.shape)
        else:  # fractions p / m differ by at least 1 / m_max^2: the noise only breaks ties
            m = n + 1
            score = (lattice.counts(state) + state) / m + rng.random(state.shape) / m.max() ** 2
        return _top_fraction(score, self.n_ones(state.size))

    def describe(self) -> str:
        return f"conserved({self.fraction:g})"


class ProbabilisticRule(Rule):
    """Birth with probability ``p_birth[n]`` and survival with ``p_survival[n]``
    (arrays indexed by the number ``n`` of neighbours in state 1)."""

    def __init__(self, p_birth, p_survival):
        self.p_birth = np.asarray(p_birth, dtype=float)
        self.p_survival = np.asarray(p_survival, dtype=float)
        for name, p in (("p_birth", self.p_birth), ("p_survival", self.p_survival)):
            if p.ndim != 1 or np.any((p < 0) | (p > 1)):
                raise ValueError(f"{name}: probabilities in [0, 1] indexed by counts")

    def next_state(self, state, lattice, rng):
        n = int(np.max(lattice.n_neighbours))
        c = lattice.counts(state)
        pb = np.zeros(n + 1)
        ps = np.zeros(n + 1)
        pb[: min(n + 1, len(self.p_birth))] = self.p_birth[: n + 1]
        ps[: min(n + 1, len(self.p_survival))] = self.p_survival[: n + 1]
        p = np.where(state == 1, ps[c], pb[c])
        return (rng.random(state.shape) < p).astype(np.uint8)

    def describe(self) -> str:
        return "probabilistic"


class ActivatorInhibitorRule(Rule):
    """Young's (1984) activator-inhibitor automaton.

    The score of a cell is the number of active cells within ``activation_radius``
    (itself included) minus ``weight`` times the number of active cells at distances in
    (``activation_radius``, ``inhibition_radius``] (Euclidean distances in cells). The
    cell becomes active if the score is positive, inactive if negative. With a
    ``fraction``, the cells of highest score are active instead (conserved volume
    fraction). The default weight balances the two regions (a uniform state scores 0); on
    tilings whose discs differ from cell to cell (Penrose), it is balanced cell by cell.
    """

    def __init__(
        self,
        activation_radius: float,
        inhibition_radius: float,
        weight: float | None = None,
        fraction: float | None = None,
        dims: int = 3,
    ):
        r1, r2 = float(activation_radius), float(inhibition_radius)
        if not (1.0 <= r1 < r2):
            raise ValueError("radii: 1 <= activation_radius < inhibition_radius (in cells)")
        self.r1, self.r2 = r1, r2
        inner = offsets("euclidean", r1, dims=dims, include_center=True)
        outer = offsets("euclidean", r2, dims=dims, include_center=True)
        ring = outer[(outer * outer).sum(axis=1) > r1 * r1 + 1e-9]
        self.balanced = weight is None
        self.weight = len(inner) / len(ring) if weight is None else float(weight)
        if not self.weight > 0:
            raise ValueError("weight: must be positive")
        self.offsets = np.concatenate([inner, ring])
        self.weights = np.concatenate([np.ones(len(inner)), -self.weight * np.ones(len(ring))])
        self.fraction = None if fraction is None else float(fraction)
        self.conserves_fraction = self.fraction is not None

    def tiling_kernels(self, lattice: TilingContext):
        """Discs (CSR, the cell included), rings (CSR) and weights on a tiling."""
        key = ("activator_inhibitor", self.r1, self.r2, self.balanced, self.weight)
        if key not in lattice.cache:
            t = lattice.tiling
            n = t.n_cells

            def matrix(ball):
                ip, ix = ball
                return sparse.csr_matrix((np.ones(len(ix), dtype=np.int8), ix, ip), shape=(n, n))

            disc = matrix(t.ball(self.r1))
            ring = matrix(t.ball(self.r2)) - disc
            ring.eliminate_zeros()
            ring.sort_indices()
            inner = (disc.indptr.astype(np.int64), disc.indices.astype(np.int64))
            outer = (ring.indptr.astype(np.int64), ring.indices.astype(np.int64))
            w = self.weight
            if self.balanced:
                w = np.diff(inner[0]) / np.maximum(1, np.diff(outer[0]))
                w = float(w[0]) if np.ptp(w) == 0 else w
            lattice.cache[key] = (inner, outer, w)
        return lattice.cache[key]

    def score(self, state, lattice) -> np.ndarray:
        if isinstance(lattice, TilingContext):
            inner, ring, w = self.tiling_kernels(lattice)
            t = lattice.tiling
            return t.counts(state, *inner) - w * t.counts(state, *ring)
        return kernel_sum(
            state, self.offsets, self.weights, periodic=lattice.periodic, boundary=lattice.boundary
        )

    def next_state(self, state, lattice, rng):
        s = self.score(state, lattice)
        if self.fraction is not None:
            return _top_fraction(
                s + 1e-6 * rng.random(state.shape), round(self.fraction * state.size)
            )
        tol = 1e-9 * (1.0 + self.weight) * len(self.offsets)
        out = state.copy()
        out[s > tol] = 1
        out[s < -tol] = 0
        return out

    def describe(self) -> str:
        return f"activator_inhibitor({self.r1:g}, {self.r2:g}, w={self.weight:.4g})"


#: Named rules. ``clouds`` is a 3D Moore rule known to turn noise into cave-like
#: structures (B13-14,17-19/S13-26).
NAMED_RULES = {
    "majority": lambda: MajorityRule(twist=False),
    "anneal": lambda: MajorityRule(twist=True),
    "life": lambda: LifeRule.parse("B3/S23"),
    "clouds": lambda: LifeRule.parse("B13-14,17-19/S13-26"),
}


def make_rule(rule, *, fraction: float | None = None) -> Rule:
    """A rule from a name (``majority``, ``anneal``, ``conserved``, ``life``,
    ``clouds``), a ``B.../S...`` string, or a :class:`Rule`."""
    if isinstance(rule, Rule):
        return rule
    text = str(rule).strip()
    if text.lower() == "conserved":
        if fraction is None:
            raise ValueError("the 'conserved' rule needs the volume fraction")
        return ConservedMajorityRule(fraction)
    if text.lower() in NAMED_RULES:
        return NAMED_RULES[text.lower()]()
    return LifeRule.parse(text)
