"""Search algorithms for discrete design problems, and the inverse design of automata.

``simulated_annealing`` and ``genetic_algorithm`` minimize a cost over any
representation (bit strings, tile weights, rule tables...), given the moves.
``fit_rule`` searches the outer-totalistic rules (birth and survival sets) of a cellular
automaton whose structures, grown from noise, match target morphological descriptors:
volume fraction, interface density and two-point correlation function, measured on an
image (e.g. a real microstructure, at the resolution of the automaton's cells) or given.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from morphogen.automata.engine import CellularAutomaton, interface_density
from morphogen.automata.lattice import offsets
from morphogen.automata.rules import LifeRule
from morphogen.automata.tilings import QUASIPERIODIC, TilingContext, make_tiling


@dataclass
class SearchResult:
    """Best solution found, its cost and the history of the search."""

    best: Any
    cost: float
    evaluations: int
    history: dict[str, list] = field(default_factory=dict)


def simulated_annealing(
    x0,
    cost: Callable[[Any], float],
    neighbour: Callable[[Any, np.random.Generator], Any],
    rng: np.random.Generator,
    *,
    steps: int = 1000,
    t_start: float | None = None,
    t_end: float | None = None,
    callback=None,
) -> SearchResult:
    """Minimize ``cost`` by Metropolis moves ``x -> neighbour(x, rng)`` with a geometric
    cooling from ``t_start`` to ``t_end`` (default: from the spread of the costs of 20
    random moves, down to a thousandth of it)."""
    x, c = x0, float(cost(x0))
    evaluations = 1
    if t_start is None:
        diffs = []
        for _ in range(20):
            diffs.append(abs(float(cost(neighbour(x0, rng))) - c))
            evaluations += 1
        t_start = max(float(np.mean(diffs)), 1e-12)
    if t_end is None:
        t_end = 1e-3 * t_start
    if not 0 < t_end <= t_start:
        raise ValueError("temperatures: 0 < t_end <= t_start")
    best, best_c = x, c
    hist: dict[str, list] = {"step": [], "cost": [], "best": [], "temperature": []}
    ratio = (t_end / t_start) ** (1.0 / max(1, steps - 1))
    t = t_start
    for k in range(int(steps)):
        y = neighbour(x, rng)
        cy = float(cost(y))
        evaluations += 1
        if cy <= c or rng.random() < math.exp(-(cy - c) / t):
            x, c = y, cy
            if c < best_c:
                best, best_c = x, c
        hist["step"].append(k)
        hist["cost"].append(c)
        hist["best"].append(best_c)
        hist["temperature"].append(t)
        if callback is not None:
            callback(k, x, c)
        t *= ratio
    return SearchResult(best, best_c, evaluations, hist)


def genetic_algorithm(
    population: list,
    cost: Callable[[Any], float],
    rng: np.random.Generator,
    *,
    mutate: Callable[[Any, np.random.Generator], Any],
    crossover: Callable[[Any, Any, np.random.Generator], Any] | None = None,
    generations: int = 50,
    elite: int = 2,
    tournament: int = 3,
    callback=None,
) -> SearchResult:
    """Minimize ``cost`` with a generational genetic algorithm: the ``elite`` best
    individuals are kept, the others are children of parents chosen by tournaments of
    ``tournament`` individuals, by ``crossover`` (if given) then ``mutate``."""
    pop = list(population)
    if len(pop) < 2:
        raise ValueError("population: at least two individuals")
    if not 0 <= elite < len(pop):
        raise ValueError("elite: fewer than the population")
    costs = [float(cost(x)) for x in pop]
    evaluations = len(pop)
    hist: dict[str, list] = {"generation": [], "best": [], "mean": []}

    def pick():
        idx = rng.choice(len(pop), size=min(tournament, len(pop)), replace=False)
        return pop[min(idx, key=lambda i: costs[i])]

    for g in range(int(generations) + 1):
        order = np.argsort(costs)
        hist["generation"].append(g)
        hist["best"].append(costs[order[0]])
        hist["mean"].append(float(np.mean(costs)))
        if callback is not None:
            callback(g, pop[order[0]], costs[order[0]])
        if g == generations:
            break
        children = [pop[i] for i in order[:elite]]
        child_costs = [costs[i] for i in order[:elite]]
        while len(children) < len(pop):
            a = pick()
            child = crossover(a, pick(), rng) if crossover is not None else a
            child = mutate(child, rng)
            children.append(child)
            child_costs.append(float(cost(child)))
            evaluations += 1
        pop, costs = children, child_costs
    best = int(np.argmin(costs))
    return SearchResult(pop[best], costs[best], evaluations, hist)


# -- inverse design of cellular automata ---------------------------------------------------


def structure_descriptors(state, max_lag: int = 4, periodic: bool = True) -> np.ndarray:
    """Volume fraction, interface density (fraction of unlike neighbour pairs) and
    two-point correlation ``rho(h) = (C(h) - f^2) / (f - f^2)`` at lags 1..max_lag
    (mean over the axes of more than one cell)."""
    s = np.asarray(state.data if hasattr(state, "data") else state) == 1
    if s.ndim != 3:
        raise ValueError("descriptors of 3D arrays ((nx, ny, 1) in 2D)")
    f = float(s.mean())
    out = [f, interface_density(s.astype(np.uint8), periodic)]
    axes = [a for a in range(3) if s.shape[a] > max_lag]
    for h in range(1, max_lag + 1):
        if f <= 0.0 or f >= 1.0 or not axes:
            out.append(0.0)
            continue
        c = np.mean([np.mean(s & np.roll(s, h, axis=a)) for a in axes])
        out.append((c - f * f) / (f - f * f))
    return np.array(out)


def descriptor_cost(d: np.ndarray, target: np.ndarray) -> float:
    """Squared errors: relative for the volume fraction and the interface density,
    absolute for the correlation function."""
    e_f = (d[0] - target[0]) / max(target[0], 0.05)
    e_s = (d[1] - target[1]) / max(target[1], 0.01)
    return float(e_f**2 + e_s**2 + np.sum((d[2:] - target[2:]) ** 2))


@dataclass
class RuleFit:
    """Result of :func:`fit_rule`."""

    rule: str
    initial_fraction: float
    cost: float
    descriptors: np.ndarray
    target: np.ndarray
    evaluations: int
    history: dict[str, list]


def fit_rule(
    target,
    *,
    neighbourhood: str = "moore",
    radius: float = 1,
    dims: int = 3,
    lattice: str = "square",
    shape=(24, 24, 24),
    steps: int = 15,
    initial_fraction: float | None = None,
    population: int = 24,
    generations: int = 15,
    max_lag: int = 4,
    rng: np.random.Generator | None = None,
    callback=None,
) -> RuleFit:
    """Rule ``B.../S...`` (and initial noise density) whose structures best match
    ``target``.

    ``target`` is a phase image (phase 1 is the automaton's state 1) or an array of
    descriptors (see :func:`structure_descriptors`). Each candidate rule is run for
    ``steps`` synchronous updates from noise of density ``initial_fraction`` (searched
    too when not given) on a periodic lattice of ``shape`` cells, and scored by
    :func:`descriptor_cost`; a genetic algorithm searches the birth and survival sets.
    All candidates start from the same noise (common random numbers). Other ``lattice``
    values than ``square`` run the candidates on a tiling of the box of ``shape`` cell
    sizes (:mod:`morphogen.automata.tilings`; a free patch for quasi-periodic tilings),
    rasterized at one voxel per cell size before their descriptors are measured.
    """
    rng = np.random.default_rng() if rng is None else rng
    if isinstance(target, np.ndarray) and target.ndim == 1:
        goal = np.asarray(target, dtype=float)
        max_lag = len(goal) - 2
    else:
        goal = structure_descriptors(target, max_lag)
    if initial_fraction is not None and not 0.0 < initial_fraction < 1.0:
        raise ValueError("initial_fraction: must lie in (0, 1)")
    shape = tuple(int(n) for n in shape)
    tiling = pixels = None
    if lattice == "square":
        k = len(offsets(neighbourhood, radius, dims=dims))
        noise = rng.random(shape)
    else:
        from morphogen.domain import Domain, Grid

        periodic = lattice not in QUASIPERIODIC
        tiling = make_tiling(lattice, shape, 1.0, dims=dims, periodic=periodic, rng=rng)
        pixels = tiling.cell_index(Grid(Domain(size=shape, periodic=periodic), shape))
        k = int(np.max(TilingContext(tiling, neighbourhood, radius).n_neighbours))
        noise = rng.random(tiling.n_cells)

    def rule_of(genome) -> LifeRule:
        bits = genome[0]
        return LifeRule(np.flatnonzero(bits[: k + 1]), np.flatnonzero(bits[k + 1 :]))

    def describe(genome):
        ca = CellularAutomaton(
            rule_of(genome), neighbourhood=neighbourhood, radius=radius, dims=dims, tiling=tiling
        )
        state = (noise < genome[1]).astype(np.uint8)
        evo = ca.run(state, steps, rng, record=False)
        return structure_descriptors(evo.state if tiling is None else evo.state[pixels], max_lag)

    def cost(genome) -> float:
        return descriptor_cost(describe(genome), goal)

    def random_genome():
        # threshold rules: birth above b neighbours, survival from s neighbours
        b, s = rng.integers(k // 4, 3 * k // 4 + 1, size=2)
        bits = np.zeros(2 * (k + 1), dtype=bool)
        bits[b + 1 : k + 1] = True
        bits[k + 1 + s :] = True
        p0 = initial_fraction if initial_fraction is not None else rng.uniform(0.2, 0.8)
        return bits, float(p0)

    def mutate(genome, rng):
        bits, p0 = genome
        out = bits.copy()
        flips = rng.random(out.size) < 1.5 / out.size
        if not flips.any() and (initial_fraction is not None or rng.random() < 0.7):
            flips[rng.integers(out.size)] = True
        out[flips] = ~out[flips]
        if initial_fraction is None and rng.random() < 0.3:
            p0 = float(np.clip(p0 + 0.05 * rng.standard_normal(), 0.02, 0.98))
        return out, p0

    def crossover(a, b, rng):
        mask = rng.random(a[0].size) < 0.5
        return np.where(mask, a[0], b[0]), (a[1] if rng.random() < 0.5 else b[1])

    init = [mutate(random_genome(), rng) for _ in range(int(population))]
    res = genetic_algorithm(
        init,
        cost,
        rng,
        mutate=mutate,
        crossover=crossover,
        generations=generations,
        callback=callback,
    )
    best = res.best
    return RuleFit(
        rule_of(best).describe(),
        best[1],
        res.cost,
        describe(best),
        goal,
        res.evaluations,
        res.history,
    )
