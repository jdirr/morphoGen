"""Branching networks grown by space colonization: vascular trees, roots, corals.

Blood vessels, the airways of lungs, the roots and crowns of plants grow toward the
tissue or the space they serve. The space colonization algorithm (Runions, Lane and
Prusinkiewicz 2007) grows a tree toward attraction points scattered in the volume (the
cells that call for vessels, the light the crown seeks): at each step, every point
pulls the node of the tree nearest to it within the influence distance ``d_i``; each
pulled node grows a branch of length ``D`` toward the mean direction of its points, and
the points closer to the tree than the kill distance ``d_k`` are reached and removed.
The growth stops when no point is left within reach of the tree.

The radii follow Murray's law (1926): the radius of a vessel to the power ``n`` is the
sum of those of its branches, from tips of radius ``r_t``: ``n = 3`` minimizes the work
of a laminar flow and the volume of blood (Murray), ``n = 2`` keeps the cross-section
area (the pipe model of trees, da Vinci's rule). A root then has the radius ``r_t
N^(1/n)`` for the ``N`` tips it feeds. In a periodic box the tree grows on the torus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.spatial import cKDTree

from morphogen._utils import check_positive
from morphogen.domain import Domain
from morphogen.models.base import Model, param, register
from morphogen.models.union import PrimitiveUnion
from morphogen.orientations import uniform_unit_vectors
from morphogen.primitives import Segments
from morphogen.seeding import SeedLike, make_rng

ROOT_POSITIONS = ("face", "center", "random")


def murray_radii(parents: np.ndarray, tip: float, exponent: float) -> np.ndarray:
    """Radius of the segment ending at each node: ``tip`` for the tips, and Murray's law
    ``r^n = sum of r_child^n`` above (the parents come before their children)."""
    n = len(parents)
    flow = np.zeros(n)
    has_child = np.zeros(n, dtype=bool)
    has_child[parents[parents >= 0]] = True
    flow[~has_child] = tip**exponent
    for i in range(n - 1, -1, -1):
        if parents[i] >= 0:
            flow[parents[i]] += flow[i]
    return flow ** (1.0 / exponent)


@dataclass
class _Tree:
    positions: np.ndarray  # nodes (unwrapped), in order of growth
    parents: np.ndarray  # -1 for the roots
    reached: int  # attraction points reached
    steps: int


@register
@dataclass(frozen=True)
class BranchingNetwork(Model):
    """Branching network grown toward ``n_attractors`` random points by space
    colonization (segments of ``step``), with Murray's law for the radii."""

    name = "branching"
    phases: ClassVar[dict[int, str]] = {0: "tissue", 1: "vessels"}
    count_label = "segments"

    n_attractors: int = param(2000, doc="Attraction points in the domain", kind="int")
    step: float = param(0.02, doc="Length of the branch segments", unit="length")
    influence: float | None = param(
        None, doc="Distance within which a point pulls the tree (default: 8 steps)", unit="length"
    )
    kill: float | None = param(
        None, doc="Distance at which the tree reaches a point (default: 2 steps)", unit="length"
    )
    roots: int = param(1, doc="Number of trees", kind="int")
    root_position: str = param(
        "face", doc="face (on the lower face along the axis), center (one tree) or random",
        kind="str",
    )  # fmt: skip
    tip_radius: float = param(0.004, doc="Radius of the tips", unit="length")
    murray_exponent: float = param(3.0, doc="Exponent n of Murray's law (3: vessels, 2: trees)")
    max_steps: int = param(2000, doc="Largest number of growth steps", kind="int")
    axis: int = param(2, doc="Axis of the face of the roots (0: x, 1: y, 2: z)", kind="int")

    def _validate(self) -> None:
        if self.n_attractors < 1:
            raise ValueError("n_attractors: at least one point")
        d = check_positive(self.step, "step")
        if not self.kill_distance < self.influence_distance:
            raise ValueError("kill: closer than the influence distance")
        if self.kill_distance < 0.5 * d:
            raise ValueError("kill: at least half a step (or the tree misses points)")
        if self.roots < 1:
            raise ValueError("roots: at least one tree")
        if self.root_position not in ROOT_POSITIONS:
            raise ValueError(f"root_position: one of {list(ROOT_POSITIONS)}")
        if self.root_position == "center" and self.roots != 1:
            raise ValueError("root_position: center grows one tree")
        check_positive(self.tip_radius, "tip_radius")
        if not 1.0 <= self.murray_exponent <= 4.0:
            raise ValueError("murray_exponent: from 1 to 4")
        if self.max_steps < 1:
            raise ValueError("max_steps: at least one step")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis: 0, 1 or 2")

    @property
    def influence_distance(self) -> float:
        return 8.0 * self.step if self.influence is None else float(self.influence)

    @property
    def kill_distance(self) -> float:
        return 2.0 * self.step if self.kill is None else float(self.kill)

    # -- growth --------------------------------------------------------------------------

    def _roots(self, rng, size: np.ndarray) -> np.ndarray:
        if self.root_position == "center":
            return 0.5 * size[None, :]
        roots = rng.random((self.roots, 3)) * size
        if self.root_position == "face":
            roots[:, self.axis] = 0.0
        return roots

    def grow(self, rng, size: np.ndarray, periodic: bool) -> _Tree:
        """Grow the trees in the box ``[0, size)`` (on the torus if ``periodic``)."""
        box = size if periodic else None

        def wrap(p):
            if not periodic:
                return p
            w = np.mod(p, size)
            w[w >= size] = 0.0
            return w

        def towards(points, nodes):
            v = points - nodes
            if periodic:
                v -= size * np.round(v / size)
            return v

        points = rng.random((self.n_attractors, 3)) * size
        positions = [self._roots(rng, size)]
        parents = [np.full(len(positions[0]), -1, dtype=np.int64)]
        n_nodes = len(positions[0])
        alive = np.ones(len(points), dtype=bool)
        d, di, dk = self.step, self.influence_distance, self.kill_distance

        def reach(new):  # the points close to new nodes are reached
            idx = np.flatnonzero(alive)
            if idx.size:
                dist, _ = cKDTree(wrap(new), boxsize=box).query(
                    points[idx], k=1, distance_upper_bound=dk
                )
                alive[idx[np.isfinite(dist)]] = False

        reach(positions[0])
        steps = 0
        for steps in range(1, self.max_steps + 1):  # noqa: B007 (the count is returned)
            idx = np.flatnonzero(alive)
            if not idx.size:
                break
            pos = np.concatenate(positions)
            tree = cKDTree(wrap(pos), boxsize=box)
            dist, near = tree.query(points[idx], k=1, distance_upper_bound=di)
            pulled = np.isfinite(dist)
            if not pulled.any():
                break
            near = near[pulled]
            v = towards(points[idx[pulled]], wrap(pos)[near])
            v /= np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]
            nodes = np.unique(near)
            pull = np.zeros((len(pos), 3))
            np.add.at(pull, near, v)
            pull = pull[nodes]
            norm = np.linalg.norm(pull, axis=1)
            balanced = norm < 1e-9  # opposite pulls: a random direction
            if balanced.any():
                pull[balanced] = uniform_unit_vectors(rng, int(balanced.sum()))
                norm[balanced] = 1.0
            new = pos[nodes] + d * pull / norm[:, None]
            # no node on top of another (a node pulled the same way twice)
            gap, _ = tree.query(wrap(new), k=1)
            fresh = gap > 0.25 * d
            if not fresh.any():
                break
            new, nodes = new[fresh], nodes[fresh]
            positions.append(new)
            parents.append(nodes)
            n_nodes += len(new)
            reach(new)
        return _Tree(
            np.concatenate(positions),
            np.concatenate(parents),
            int((~alive).sum()),
            steps,
        )

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        size = domain.lengths
        tree = self.grow(rng, size, domain.periodic)
        radii = murray_radii(tree.parents, self.tip_radius, self.murray_exponent)
        child = np.flatnonzero(tree.parents >= 0)
        start = tree.positions[tree.parents[child]]
        end = tree.positions[child]
        if domain.periodic:  # each segment from its end inside the box
            offset = np.floor(start / size) * size
            start, end = start - offset, end - offset
        segments = Segments(start + domain.lower, end + domain.lower, radii[child], round_caps=True)
        has_child = np.zeros(len(tree.parents), dtype=bool)
        has_child[tree.parents[tree.parents >= 0]] = True
        roots = np.flatnonzero(tree.parents < 0)
        counts = np.bincount(tree.parents[child], minlength=len(tree.parents))
        summary = {
            "n_tips": int((~has_child).sum()),
            "n_branchings": int((counts >= 2).sum()),
            "total_length": float(np.linalg.norm(end - start, axis=1).sum()),
            "root_radius": float(radii[roots].max()),
            "reached_fraction": tree.reached / self.n_attractors,
            "growth_steps": tree.steps,
        }
        return PrimitiveUnion(
            domain, segments=segments, model=self, seed=seed_info, metadata={"summary": summary}
        )

    def expected_count(self, domain: Domain) -> float | None:
        return None

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The radius of a root feeding ``N`` tips is ``r_t N^(1/n)``: the exponent of
        Murray's law and the area ratio of a symmetric branching, ``2^(1 - 2/n)``."""
        n = self.murray_exponent
        return {
            "murray_exponent": n,
            "symmetric_area_ratio": 2.0 ** (1.0 - 2.0 / n),
            "symmetric_radius_ratio": 2.0 ** (-1.0 / n),
        }
