"""Evolution of binary cellular automata on 2D or 3D lattices.

Updates are

``synchronous``
    every cell applies the rule to the same previous state (classical automata);
``asynchronous``
    at each step, each cell is updated with probability ``async_fraction`` (the
    alpha-asynchronous automata of Fates): it avoids the period-2 oscillations of
    synchronous majority rules;
``sequential``
    the sublattices of cells that are not neighbours are updated one after the other,
    in random order, each seeing the latest state (radius-1 neighbourhoods; on tilings,
    the colour classes of a greedy colouring of the neighbourhood graph, any radius).

On a :class:`~morphogen.automata.tilings.Tiling` (triangles, hexagons, Penrose rhombi,
Ammann-Beenker tiles, prisms, truncated octahedra, golden rhombohedra...), states hold one
value per cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from morphogen.automata.lattice import offsets, sublattices
from morphogen.automata.rules import LatticeContext, Rule, make_rule
from morphogen.automata.tilings import Tiling, TilingContext

UPDATES = ("synchronous", "asynchronous", "sequential")


def interface_density(state: np.ndarray, periodic: bool = True) -> float:
    """Mean fraction, over the three axes, of pairs of adjacent cells in different
    states (twice its value divided by the cell size estimates the specific surface)."""
    values = []
    for axis in range(3):
        n = state.shape[axis]
        if n < 2:
            continue
        if periodic:
            diff = state != np.roll(state, -1, axis=axis)
            values.append(diff.mean())
        else:
            a = np.take(state, range(n - 1), axis=axis)
            b = np.take(state, range(1, n), axis=axis)
            values.append((a != b).mean())
    return float(np.mean(values)) if values else 0.0


@dataclass
class Evolution:
    """Final state and history (one entry per recorded step) of a run."""

    state: np.ndarray
    steps: int
    history: dict[str, list] = field(default_factory=dict)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.history.items()}


class CellularAutomaton:
    """Binary automaton: a rule acting on a neighbourhood of a periodic or fixed lattice,
    the square grid or a ``tiling`` (whose ``dims`` and ``periodic`` then apply; finite
    patches have free boundaries, ``boundary`` is ignored).

    Example::

        ca = CellularAutomaton("anneal", neighbourhood="moore")
        state = (rng.random((64, 64, 64)) < 0.5).astype(np.uint8)
        result = ca.run(state, steps=30, rng=rng)
    """

    def __init__(
        self,
        rule: Rule | str,
        *,
        neighbourhood: str = "moore",
        radius: float = 1,
        dims: int = 3,
        periodic: bool = True,
        boundary: int = 0,
        fraction: float | None = None,
        tiling: Tiling | None = None,
    ):
        self.rule = make_rule(rule, fraction=fraction)
        self.neighbourhood = neighbourhood
        self.radius = radius
        self.tiling = tiling
        if tiling is not None:
            self.dims = tiling.dims
            self.lattice = TilingContext(tiling, neighbourhood, radius)
        else:
            self.dims = dims
            self.lattice = LatticeContext(
                offsets(neighbourhood, radius, dims=dims), bool(periodic), int(boundary), dims
            )

    def _check_state(self, state: np.ndarray) -> np.ndarray:
        s = np.ascontiguousarray(state, dtype=np.uint8)
        if self.tiling is not None:
            if s.shape != (self.tiling.n_cells,):
                raise ValueError(f"tiling states: one value per cell ({self.tiling.n_cells})")
        elif s.ndim != 3:
            raise ValueError("the state must be a 3D array (use shape (nx, ny, 1) in 2D)")
        elif self.dims == 2 and s.shape[2] != 1:
            raise ValueError("2D automata act on states of shape (nx, ny, 1)")
        if np.any(s > 1):
            raise ValueError("binary automata: states must be 0 or 1")
        return s

    def step(
        self,
        state,
        rng: np.random.Generator,
        *,
        update: str = "synchronous",
        async_fraction: float = 0.5,
    ) -> np.ndarray:
        """State after one update."""
        if update not in UPDATES:
            raise ValueError(f"update: one of {UPDATES}, got {update!r}")
        state = self._check_state(state)
        if update != "synchronous" and self.rule.conserves_fraction:
            raise ValueError("rules conserving the volume fraction need synchronous updates")
        if update == "synchronous":
            return self.rule.next_state(state, self.lattice, rng)
        if update == "asynchronous":
            if not 0.0 < async_fraction <= 1.0:
                raise ValueError("async_fraction: must lie in (0, 1]")
            new = self.rule.next_state(state, self.lattice, rng)
            mask = rng.random(state.shape) < async_fraction
            return np.where(mask, new, state).astype(np.uint8)
        if self.tiling is not None:
            groups = self.lattice.cache.get("sublattices")
            if groups is None:
                groups = self.lattice.cache["sublattices"] = self.lattice.sublattices()
        elif float(self.radius) >= 2:
            raise ValueError("sequential updates need radius-1 neighbourhoods")
        else:
            groups = sublattices(self.neighbourhood, state.shape, self.dims)
        current = state.copy()
        for g in rng.permutation(len(groups)):
            new = self.rule.next_state(current, self.lattice, rng)
            current = np.where(groups[g], new, current).astype(np.uint8)
        return current

    def run(
        self,
        state,
        steps: int,
        rng: np.random.Generator | None = None,
        *,
        update: str = "synchronous",
        async_fraction: float = 0.5,
        record: bool = True,
        stop_when_stable: bool = True,
        callback=None,
    ) -> Evolution:
        """Apply ``steps`` updates (fewer if the state stops changing).

        The history records, after each step, the fraction of cells in state 1, the
        number of cells that changed and the interface density. ``callback(step,
        state)`` is called after every step (e.g. to save frames of an animation).
        """
        rng = np.random.default_rng() if rng is None else rng
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps: must be non-negative")
        state = self._check_state(state)
        hist: dict[str, list] = {"step": [], "fraction": [], "changed": [], "interface": []}

        def log(k, changed):
            if record:
                hist["step"].append(k)
                hist["fraction"].append(float(state.mean()))
                hist["changed"].append(int(changed))
                if self.tiling is not None:
                    hist["interface"].append(self.lattice.interface(state))
                else:
                    hist["interface"].append(interface_density(state, self.lattice.periodic))

        log(0, 0)
        done = 0
        for k in range(1, steps + 1):
            new = self.step(state, rng, update=update, async_fraction=async_fraction)
            changed = int(np.count_nonzero(new != state))
            state = new
            done = k
            log(k, changed)
            if callback is not None:
                callback(k, state)
            if stop_when_stable and changed == 0 and update == "synchronous":
                break
        return Evolution(state=state, steps=done, history=hist)


def random_state(
    shape, fraction: float, rng: np.random.Generator, *, exact: bool = False
) -> np.ndarray:
    """Random binary state: each cell is 1 with probability ``fraction`` (or exactly
    ``round(fraction * n)`` cells at random positions with ``exact``)."""
    shape = tuple(int(n) for n in shape)
    if exact:
        n = int(np.prod(shape))
        out = np.zeros(n, dtype=np.uint8)
        out[rng.permutation(n)[: round(fraction * n)]] = 1
        return out.reshape(shape)
    return (rng.random(shape) < fraction).astype(np.uint8)
