"""Grain structures and aggregates grown on lattices.

``nucleation_growth``
    Johnson-Mehl-Avrami-Kolmogorov nucleation and growth: Voronoi (site saturation) or
    Johnson-Mehl (continuous nucleation) polycrystals, partially transformed structures
    (grains in an untransformed matrix), faceted grains, and normal grain growth
    (Potts model) after the transformation.
``aggregation``
    Diffusion-limited (fractal), Eden (compact) or ballistic (porous) aggregates:
    clusters grown from seeds, or deposits grown from a face of the domain.

See :mod:`morphogen.automata.growth` for the algorithms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen._utils import check_fraction, check_positive
from morphogen.automata.growth import (
    AGGREGATION_GEOMETRIES,
    AGGREGATION_PROCESSES,
    GROWTH_SHAPES,
    aggregate,
    grow_grains,
    potts_coarsening,
    shape_volume_coefficient,
)
from morphogen.domain import Domain
from morphogen.models.affine import ANISOTROPY, Anisotropy, anisotropic
from morphogen.models.automata import MAX_MARGIN, _LatticeModel
from morphogen.models.base import param, register
from morphogen.models.lattice_field import LatticeField, lattice_shape
from morphogen.orientations import euler_to_matrix, random_orientations
from morphogen.seeding import SeedLike, make_rng

NUCLEATION = ("site_saturation", "continuous")

#: Fractal dimensions of on-lattice aggregates (Meakin 1983; compact for Eden growth).
DLA_DIMENSION = {2: 1.71, 3: 2.50}


def _check_phase_fractions(p) -> None:
    if p is None:
        return
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or len(p) < 1 or np.any(p < 0) or abs(p.sum() - 1.0) > 1e-9:
        raise ValueError("phase_fractions: non-negative probabilities summing to 1")
    if len(p) > 255:
        raise ValueError("phase_fractions: at most 255 phases")


@register
@anisotropic
@dataclass(frozen=True)
class NucleationGrowthModel(_LatticeModel):
    """Nucleation and growth of grains until impingement (JMAK kinetics).

    Grains nucleate at random and grow at constant velocity (Johnson-Mehl-Avrami-
    Kolmogorov): Voronoi or Johnson-Mehl polycrystals, or grains in an untransformed
    matrix when the transformation is stopped.
    """

    name = "nucleation_growth"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "grains"}
    count_label = "grains"

    nucleation: str = param(
        "site_saturation",
        doc="site_saturation (every nucleus at t = 0: Voronoi grains) or continuous "
        "(constant rate in the untransformed volume: Johnson-Mehl grains)",
        kind="str",
    )
    nucleus_density: float | None = param(
        None, doc="Nuclei per unit volume (site saturation)", unit="1/length^3"
    )
    nucleation_rate: float | None = param(
        None,
        doc="Nuclei per unit volume and unit time (continuous nucleation)",
        unit="1/(length^3 time)",
    )
    growth_rate: float = param(
        1.0, doc="Radius gained by a grain per unit time", unit="length/time"
    )
    time: float | None = param(
        None,
        doc="Duration of the transformation (default: until it is complete)",
        unit="time",
    )
    growth_shape: str = param(
        "sphere",
        doc="sphere (isotropic growth), cube or octahedron (faceted grains, oriented by "
        "their crystal orientation when crystal_orientations is set)",
        kind="str",
    )
    crystal_orientations: bool = param(
        False, doc="Draw a uniformly random crystal orientation per grain", kind="bool"
    )
    phase_fractions: tuple[float, ...] | None = param(
        None,
        doc="Probabilities of phases 0, 1, ... for each grain (default: every grain is "
        "phase 1; the untransformed matrix is phase 0)",
        kind="floats",
    )
    coarsening_steps: int = param(
        0,
        doc="Monte Carlo sweeps of normal grain growth (Potts model) after a complete "
        "transformation",
        kind="int",
    )
    coarsening_temperature: float = param(
        0.5, doc="Temperature of the Potts model (energy of a pair of unlike neighbours)"
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        self._validate_lattice()
        if self.nucleation not in NUCLEATION:
            raise ValueError(f"nucleation: one of {NUCLEATION}")
        site = self.nucleation == "site_saturation"
        if site and (self.nucleus_density is None or self.nucleation_rate is not None):
            raise ValueError("site saturation needs 'nucleus_density' (not nucleation_rate)")
        if not site and (self.nucleation_rate is None or self.nucleus_density is not None):
            raise ValueError("continuous nucleation needs 'nucleation_rate' (not nucleus_density)")
        check_positive(
            self.nucleus_density if site else self.nucleation_rate,
            "nucleus_density" if site else "nucleation_rate",
        )
        check_positive(self.growth_rate, "growth_rate")
        if self.time is not None:
            check_positive(self.time, "time")
        if self.growth_shape not in GROWTH_SHAPES:
            raise ValueError(f"growth_shape: one of {GROWTH_SHAPES}")
        _check_phase_fractions(self.phase_fractions)
        if self.coarsening_steps < 0:
            raise ValueError("coarsening_steps: must be non-negative")
        if self.coarsening_steps and self.time is not None:
            raise ValueError(
                "grain growth (coarsening_steps) follows a complete transformation: remove 'time'"
            )
        check_positive(self.coarsening_temperature, "coarsening_temperature", allow_zero=True)

    # -- kinetics ------------------------------------------------------------------------

    def _densities(self, domain: Domain | None) -> tuple[float, int]:
        """Nucleus density (or rate) per unit volume in 3D, per unit area in 2D."""
        value = (
            self.nucleus_density if self.nucleation == "site_saturation" else self.nucleation_rate
        )
        if self.dimension == 3:
            return float(value), 3
        if domain is None:
            raise ValueError("planar kinetics depend on the thickness of the domain")
        return float(value) * float(domain.lengths[2]), 2

    def extended_fraction(self, t, domain: Domain | None = None):
        """Extended volume fraction X_ext(t): volume of the grains, overlaps counted."""
        rho, d = self._densities(domain)
        c = shape_volume_coefficient(self.growth_shape, d)
        t = np.asarray(t, dtype=float)
        if self.nucleation == "site_saturation":
            return rho * c * (self.growth_rate * t) ** d
        return rho * c * self.growth_rate**d * t ** (d + 1) / (d + 1)

    def transformed_fraction(self, t, domain: Domain | None = None):
        """Avrami's law X(t) = 1 - exp(-X_ext(t))."""
        return 1.0 - np.exp(-self.extended_fraction(t, domain))

    def _time_to_extended(self, value: float, domain: Domain | None) -> float:
        """Time at which X_ext reaches ``value``."""
        rho, d = self._densities(domain)
        c = shape_volume_coefficient(self.growth_shape, d)
        if self.nucleation == "site_saturation":
            return (value / (rho * c)) ** (1.0 / d) / self.growth_rate
        return (value * (d + 1) / (rho * c * self.growth_rate**d)) ** (1.0 / (d + 1))

    def grain_density(self, domain: Domain | None = None) -> float:
        """Grains per unit volume after a complete transformation."""
        rho, d = self._densities(domain)
        if self.nucleation == "site_saturation":
            n = rho
        else:
            c = shape_volume_coefficient(self.growth_shape, d)
            a = rho * c * self.growth_rate**d / (d + 1)
            n = rho * math.gamma(1.0 + 1.0 / (d + 1)) * a ** (-1.0 / (d + 1))
        return n if d == 3 else n / float(domain.lengths[2])

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        if self.dimension == 2 and domain is None:
            return {}
        d = self.dimension
        site = self.nucleation == "site_saturation"
        out = {
            "avrami_exponent": float(d if site else d + 1),
            "half_transformation_time": self._time_to_extended(math.log(2.0), domain),
        }
        if self.time is not None:
            x = float(self.transformed_fraction(self.time, domain))
            out["transformed_fraction"] = x
            if self.phase_fractions is None:
                out["volume_fraction"] = x
        elif self.coarsening_steps == 0:
            n = self.grain_density(domain)
            out["grain_density"] = n
            out["mean_grain_volume"] = 1.0 / n
        return out

    def expected_count(self, domain: Domain) -> float | None:
        if self.time is not None or self.coarsening_steps:
            return None
        return self.grain_density(domain) * domain.volume

    # -- generation ----------------------------------------------------------------------

    def generate(self, domain: Domain, seed: SeedLike = None) -> LatticeField:
        rng, seed_info = make_rng(seed)
        fields, _ = self._simulate(domain, rng)
        return LatticeField(domain, smoothing=self.smoothing, model=self, seed=seed_info, **fields)

    def _simulate(self, domain: Domain, rng, frames: int = 0):
        """Fields of the realization, and label images at evenly spaced times (frames)."""
        d = self.dimension
        shape = lattice_shape(domain, self.cell_size, d)
        h = domain.lengths / np.array(shape)
        # duration of the simulation: the given time, or completion (an expected
        # untransformed volume of 1e-4 cells)
        complete = self._time_to_extended(math.log(max(np.prod(shape), 2.0)) + 9.2, domain)
        horizon = complete if self.time is None else min(float(self.time), complete)
        _, sim, crop = self._window(domain, self.growth_rate * horizon / self.cell_size)
        volume = float(np.prod(h)) * float(np.prod(sim))
        if self.nucleation == "site_saturation":
            n = int(rng.poisson(self.nucleus_density * volume))
            times = np.zeros(n)
        else:
            n = int(rng.poisson(self.nucleation_rate * volume * horizon))
            times = np.sort(rng.random(n) * horizon)
        cells = np.stack([rng.integers(0, s, n) for s in sim], axis=1)
        if self.crystal_orientations:
            euler = random_orientations(rng, n)
            if d == 2:  # in-plane rotations
                euler[:, 1:] = 0.0
        else:
            euler = None
        grain_phase = (
            np.ones(n, dtype=np.uint8)
            if self.phase_fractions is None
            else rng.choice(
                len(self.phase_fractions),
                size=n,
                p=np.asarray(self.phase_fractions) / np.sum(self.phase_fractions),
            ).astype(np.uint8)
        )
        faceted = self.growth_shape != "sphere" and euler is not None
        labels, arrival, real = grow_grains(
            sim,
            h,
            cells,
            times,
            growth_rate=self.growth_rate,
            growth_shape=self.growth_shape,
            rotations=euler_to_matrix(euler) if faceted else None,
            periodic=True,
            dims=d,
        )
        attributes: dict = {"n_nuclei": n, "n_phantom_nuclei": int(n - real.sum())}
        if self.time is not None:
            labels[arrival >= self.time] = -1  # transformed before the end (level set < 0)
        elif self.coarsening_steps and real.any():
            hist: dict[str, list] = {"step": [], "n_grains": []}

            def record(step, lab):
                if step % max(1, self.coarsening_steps // 20) == 0 or step == self.coarsening_steps:
                    hist["step"].append(step)
                    hist["n_grains"].append(int(np.unique(lab).size))

            labels = potts_coarsening(
                labels,
                self.coarsening_steps,
                temperature=self.coarsening_temperature,
                seed=int(rng.integers(2**31 - 1)),
                dims=d,
                callback=record,
            )
            attributes["coarsening"] = hist
        labels = labels[crop]
        arrival = arrival[crop]
        # grains of the window, numbered in order of nucleation
        present = np.unique(labels[labels >= 0])
        present = present[np.argsort(times[present], kind="stable")]
        relabel = np.full(max(n, 1), -1, dtype=np.int32)
        relabel[present] = np.arange(len(present), dtype=np.int32)
        cell_labels = np.where(labels >= 0, relabel[np.maximum(labels, 0)], -1).astype(np.int32)
        phases = np.where(labels >= 0, grain_phase[np.maximum(labels, 0)], 0).astype(np.uint8)
        field = threshold = None
        if (
            self.time is not None
            and self.phase_fractions is None
            and len(present)
            and np.all(np.isfinite(arrival))
        ):
            # smooth level set: growth distance minus the radius reached at the end
            field = -self.growth_rate * arrival
            threshold = -self.growth_rate * float(self.time)
        finite = arrival[np.isfinite(arrival)]
        t_max = (
            float(self.time)
            if self.time is not None
            else (float(finite.max()) if finite.size else 0.0)
        )
        grid = np.linspace(0.0, t_max, 41)
        attributes["kinetics"] = {
            "time": grid.tolist(),
            "transformed": [float(np.mean(arrival < t)) for t in grid],
        }
        attributes["nucleation_times"] = times[present].tolist()
        record = [
            np.where(arrival < tk, cell_labels, -1).astype(np.int32)
            for tk in np.linspace(0.0, t_max, frames)
        ]
        if record:
            record[-1] = cell_labels.copy()  # the realization (after grain growth)
        fields = {
            "phases": phases,
            "labels": cell_labels,
            "orientations": None if euler is None else euler[present],
            "field": field,
            "threshold": 0.0 if threshold is None else threshold,
            "attributes": attributes,
        }
        return fields, record


@register
@anisotropic
@dataclass(frozen=True)
class AggregationModel(_LatticeModel):
    """Aggregates of particles: diffusion-limited, Eden or ballistic growth.

    Fractal (diffusion-limited), compact (Eden) or porous (ballistic) clusters grown
    from seeds, or deposits grown from a face of the domain.
    """

    name = "aggregation"
    phases: ClassVar[dict[int, str]] = {0: "pores", 1: "aggregate"}

    process: str = param(
        "dla",
        doc="dla (diffusion-limited: random walkers), eden (random perimeter cells) or "
        "ballistic (straight trajectories)",
        kind="str",
    )
    geometry: str = param(
        "cluster",
        doc="cluster (growth from seeds) or deposit (growth from the face z = 0, y = 0 in "
        "2D, of a non-periodic domain)",
        kind="str",
    )
    n_seeds: int = param(
        1, doc="Seeds of clusters in the domain (one: a centred cluster)", kind="int"
    )
    volume_fraction: float | None = param(
        None,
        doc="Fraction of the cells in the aggregate (default: 0.1 for clusters; deposits "
        "grow until they reach the opposite face)",
        kind="fraction",
    )
    sticking: float = param(
        1.0, doc="Probability of sticking at each contact (dla, ballistic)", kind="fraction"
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        self._validate_lattice()
        if self.process not in AGGREGATION_PROCESSES:
            raise ValueError(f"process: one of {AGGREGATION_PROCESSES}")
        if self.geometry not in AGGREGATION_GEOMETRIES:
            raise ValueError(f"geometry: one of {AGGREGATION_GEOMETRIES}")
        if self.n_seeds < 1:
            raise ValueError("n_seeds: at least 1")
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        if not 0.0 < self.sticking <= 1.0:
            raise ValueError("sticking: must lie in (0, 1]")

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        d = self.dimension
        dim = DLA_DIMENSION[d] if self.process == "dla" else float(d)
        out = {"fractal_dimension": dim}
        if self.geometry == "cluster":
            out["volume_fraction"] = (
                self.volume_fraction if self.volume_fraction is not None else 0.1
            )
        return out

    def generate(self, domain: Domain, seed: SeedLike = None) -> LatticeField:
        rng, seed_info = make_rng(seed)
        fields, _ = self._simulate(domain, rng)
        return LatticeField(domain, smoothing=self.smoothing, model=self, seed=seed_info, **fields)

    def _simulate(self, domain: Domain, rng, frames: int = 0):
        """Fields of the realization, and the aggregate after evenly spaced numbers of
        particles (frames)."""
        deposit = self.geometry == "deposit"
        if deposit and domain.periodic:
            raise ValueError(
                "deposits grow on a face of the domain: use a non-periodic domain (periodic: false)"
            )
        d = self.dimension
        if deposit or self.n_seeds == 1:
            shape, sim, crop = self._window(domain, 0)
            if not domain.periodic:
                sim, crop = shape, (slice(None),) * 3
        else:
            spacing = (domain.volume / self.n_seeds) ** (1.0 / 3.0) / self.cell_size
            shape, sim, crop = self._window(domain, min(MAX_MARGIN, 2.0 * spacing))
        cells = float(np.prod(shape))
        sim_cells = float(np.prod(sim))
        seeds = 1 if self.n_seeds == 1 else max(1, round(self.n_seeds * sim_cells / cells))
        fraction = self.volume_fraction
        if fraction is None and not deposit:
            fraction = 0.1
        if fraction is None:
            n_particles = None
        else:
            n_particles = max(0, round(fraction * sim_cells) - (0 if deposit else seeds))
        occ, owner, order = aggregate(
            sim,
            rng,
            process=self.process,
            geometry=self.geometry,
            n_seeds=seeds,
            n_particles=n_particles,
            sticking=self.sticking,
            dims=d,
        )
        occ, owner, order = occ[crop], owner[crop], order[crop]
        present = np.unique(owner[owner >= 0])
        relabel = np.full(int(owner.max(initial=0)) + 1, -1, dtype=np.int32)
        relabel[present] = np.arange(len(present), dtype=np.int32)
        labels = np.where(owner >= 0, relabel[np.maximum(owner, 0)], -1).astype(np.int32)
        last = int(order.max(initial=0))
        record = [
            ((order >= 0) & (order <= q)).astype(np.uint8) for q in np.linspace(0, last, frames)
        ]
        if record:
            record[-1] = occ.copy()
        fields = {
            "phases": occ,
            "labels": labels,
            "attributes": {
                "n_particles": int(np.count_nonzero(order > 0)),
                "label_kind": "trees" if deposit else "clusters",
            },
        }
        return fields, record
