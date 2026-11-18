"""Hard-core (non-overlapping) spheres: random sequential addition, dense packings grown
by the Lubachevsky–Stillinger algorithm, and penetrable spheres with impenetrable cores
(the cherry-pit model of Torquato)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen import theory
from morphogen._utils import check_fraction, check_positive, exactly_one
from morphogen.distributions import Constant, Distribution
from morphogen.domain import Domain
from morphogen.kernels.packing import grow_packing
from morphogen.kernels.rsa import random_sequential_addition
from morphogen.models.affine import ANISOTROPY, Anisotropy, anisotropic
from morphogen.models.base import Model, param, register
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Spheres
from morphogen.seeding import SeedLike, make_rng

#: Methods of the hard-core spheres.
METHODS = ("rsa", "lubachevsky_stillinger")


def largest_first(radii, box, gap: float, rng, max_attempts: int) -> np.ndarray:
    """Positions (in ``[0, box)``, in the order of ``radii``) of non-overlapping spheres
    placed by random sequential addition, the largest first: drawn late among
    polydisperse spheres, a large one may find no room left (radii uniform between 0.8
    and 1.2 and 0.05 apart jam from a fraction of 0.31 in the order of their draw; the
    largest first, they still fill 0.41)."""
    radii = np.asarray(radii, dtype=float)
    order = np.argsort(-radii, kind="stable")
    pos = np.empty((len(radii), 3))
    pos[order] = random_sequential_addition(radii[order], box, gap, rng, max_attempts)
    return pos


def grown_packing(radii, box, rng, *, growth_rate: float = 0.01, max_events=None):
    """Positions (in ``[0, box)``) of non-overlapping spheres of the given radii in a
    periodic box, grown by the Lubachevsky–Stillinger algorithm
    (:mod:`morphogen.kernels.packing`): from the positions of a random sequential
    addition of the spheres shrunk to a packing fraction of at most 0.25, the largest
    first (drawn late, a large sphere may find no room left), with Gaussian velocities of
    the mean radius per unit time, the radii growing at ``growth_rate`` times themselves
    per unit time. Raises ``RuntimeError`` when the packing jams before the spheres
    reach their radii (after ``max_events``, 20000 per sphere by default)."""
    rho = np.ascontiguousarray(radii, dtype=float)
    box = np.asarray(box, dtype=float)
    n = len(rho)
    if n == 0:
        return np.zeros((0, 3))
    if np.any(box < 6.0 * rho.max() * (1.0 + 1e-9)):
        raise ValueError("lubachevsky_stillinger: the box spans three diameters at least")
    target = 4.0 / 3.0 * math.pi * float(np.sum(rho**3)) / float(np.prod(box))
    start = min(0.25, 0.5 * target)
    s0 = (start / target) ** (1.0 / 3.0)
    pos = largest_first(rho * s0, box, 0.0, rng, 10_000 * n)
    vel = rng.normal(0.0, float(rho.mean()), (n, 3))
    vel -= vel.mean(axis=0)
    growth = float(growth_rate)
    t_end = (1.0 - s0) / growth
    events = 20_000 * n if max_events is None else int(max_events)
    t, _, _ = grow_packing(pos, vel, rho, box, s0, growth, t_end, events)
    if t < t_end:
        reached = target * (s0 + growth * t) ** 3
        raise RuntimeError(
            f"lubachevsky_stillinger: the packing jammed at a fraction of {reached:.4f} "
            f"before {target:.4f} ({events} events); lower the volume fraction (equal "
            "spheres jam near 0.64) or the growth rate, or raise max_attempts"
        )
    return pos


@register
@anisotropic
@dataclass(frozen=True)
class HardcoreSpheres(Model):
    """Non-overlapping spheres, by random sequential addition (RSA) or grown into dense
    packings by the Lubachevsky–Stillinger algorithm.

    RSA adds the spheres one at a time at uniformly random positions, the largest first,
    a trial being rejected when the sphere would come closer than ``min_gap`` to an
    existing one: the process jams near a volume fraction of 0.38 for equal spheres. The
    Lubachevsky–Stillinger algorithm grows the spheres of a hard-sphere fluid (an
    event-driven molecular dynamics) until they reach their radii (plus half the gap):
    fast growths jam equal spheres near 0.64 (the maximally random jammed state), slow
    ones let them order.

    Generation always takes place in a periodic box; for non-periodic domains the box
    is enlarged by two grain diameters and cropped, which avoids wall effects.
    """

    name = "hardcore_spheres"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "spheres"}
    count_label = "spheres"

    radius: Distribution = param(
        Constant(1.0),
        doc="Sphere radius (number or distribution)",
        kind="distribution",
        unit="length",
    )
    volume_fraction: float | None = param(None, doc="Target volume fraction", kind="fraction")
    number: int | None = param(None, doc="Number of spheres in the generation box", kind="int")
    min_gap: float = param(0.0, doc="Minimal distance between sphere surfaces", unit="length")
    max_attempts: int | None = param(
        None,
        doc="Maximal number of trials (RSA, default 10000 per sphere) or of events "
        "(Lubachevsky-Stillinger, default 20000 per sphere)",
        kind="int",
    )
    method: str = param(
        "rsa",
        doc="rsa (random sequential addition, up to about 0.38) or lubachevsky_stillinger "
        "(growing spheres: dense packings, up to about 0.64 for equal spheres)",
        kind="str",
    )
    growth_rate: float = param(
        0.01,
        doc="Lubachevsky-Stillinger: growth of the radii per unit time over the thermal "
        "speed (fast: jammed, random packings; slow: ordered ones)",
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        exactly_one("volume_fraction", self.volume_fraction, "number", self.number)
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        elif self.number < 0:
            raise ValueError("number: must be non-negative")
        check_positive(self.min_gap, "min_gap", allow_zero=True)
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts: must be positive")
        if self.method not in METHODS:
            raise ValueError(f"method: {' or '.join(METHODS)}, got {self.method!r}")
        check_positive(self.growth_rate, "growth_rate")

    def _draw_radii(self, rng, box_volume: float) -> np.ndarray:
        if self.number is not None:
            return self.radius.sample(rng, self.number)
        target = self.volume_fraction * box_volume
        mean_v = 4.0 / 3.0 * math.pi * self.radius.moment(3)
        radii = np.empty(0)
        while True:
            more = self.radius.sample(rng, int(1.1 * target / mean_v) + 16)
            radii = np.concatenate([radii, more])
            cum = np.cumsum(4.0 / 3.0 * math.pi * radii**3)
            if cum[-1] >= target:
                break
        # number of spheres whose total volume is closest to the target
        k = int(np.searchsorted(cum, target))
        n = k + 1 if k == 0 or cum[k] - target <= target - cum[k - 1] else k
        return radii[:n]

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        rmax = self.radius.upper()
        if domain.periodic:
            if 2.0 * rmax + self.min_gap >= min(domain.size) / 2.0:
                raise ValueError("hardcore_spheres: spheres are too large for the periodic box")
            box = domain
        else:
            box = domain.dilated(2.0 * rmax + self.min_gap)
        radii = self._draw_radii(rng, box.volume)
        if self.method == "lubachevsky_stillinger":
            local = grown_packing(
                radii + 0.5 * self.min_gap,
                box.lengths,
                rng,
                growth_rate=self.growth_rate,
                max_events=self.max_attempts,
            )
        else:
            max_attempts = self.max_attempts or 10_000 * max(len(radii), 1)
            local = largest_first(radii, box.lengths, self.min_gap, rng, max_attempts)
        spheres = Spheres(box.lower + local, radii)
        if not domain.periodic:
            spheres = spheres.subset(domain.distance_to_box(spheres.centers) <= spheres.radii)
        return PrimitiveUnion(domain, spheres=spheres, model=self, seed=seed_info)

    def expected_count(self, domain: Domain) -> float | None:
        """Number of sphere centres in the domain (given a volume fraction)."""
        if self.volume_fraction is None:
            return None
        return self.volume_fraction * domain.volume / (4.0 / 3.0 * math.pi * self.radius.moment(3))

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        jammed = theory.RSA_JAMMING_FRACTION if self.method == "rsa" else theory.MRJ_FRACTION
        out = {"jamming_fraction_equal_spheres": jammed}
        if self.volume_fraction is not None:
            out["volume_fraction"] = self.volume_fraction
            out["specific_surface"] = (
                self.volume_fraction * 3.0 * self.radius.moment(2) / self.radius.moment(3)
            )
        return out


@register
@anisotropic
@dataclass(frozen=True)
class PenetrableSpheres(Model):
    """Penetrable spheres with impenetrable cores: the cherry-pit model of Torquato (1984),
    or penetrable-concentric-shell model.

    Each sphere of radius ``R`` has a core of radius ``λR`` that no other core overlaps:
    the cores are placed by random sequential addition, the largest first (or grown into
    dense packings by the Lubachevsky–Stillinger algorithm), and the shells
    interpenetrate. ``λ = 0`` gives fully penetrable spheres (independent uniform
    centres), ``λ = 1`` hard spheres; phase 1 is the union of the spheres, whose fraction
    lies between ``1 - e^{-η}`` and ``η`` for the reduced density ``η`` (the number of
    spheres per unit volume times their mean volume).
    """

    name = "penetrable_spheres"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "spheres"}
    count_label = "spheres"

    radius: Distribution = param(
        Constant(1.0),
        doc="Sphere radius (number or distribution)",
        kind="distribution",
        unit="length",
    )
    impenetrability: float = param(
        0.5,
        doc="Radius of the impenetrable core over that of the sphere (lambda: 0 fully "
        "penetrable, 1 hard spheres)",
        kind="fraction",
    )
    reduced_density: float | None = param(
        None, doc="Number of spheres per unit volume times their mean volume (eta)"
    )
    number: int | None = param(None, doc="Number of spheres in the generation box", kind="int")
    method: str = param(
        "rsa", doc="Placement of the cores: rsa or lubachevsky_stillinger", kind="str"
    )
    growth_rate: float = param(0.01, doc="Lubachevsky-Stillinger: growth rate of the cores")
    max_attempts: int | None = param(
        None,
        doc="Maximal number of trials (RSA) or of events (Lubachevsky-Stillinger)",
        kind="int",
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        exactly_one("reduced_density", self.reduced_density, "number", self.number)
        if self.reduced_density is not None:
            check_positive(self.reduced_density, "reduced_density")
        elif self.number < 0:
            raise ValueError("number: must be non-negative")
        check_fraction(self.impenetrability, "impenetrability", open_interval=False)
        if self.method not in METHODS:
            raise ValueError(f"method: {' or '.join(METHODS)}, got {self.method!r}")
        check_positive(self.growth_rate, "growth_rate")
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts: must be positive")

    def _mean_volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius.moment(3)

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        rmax = self.radius.upper()
        if domain.periodic:
            if 2.0 * rmax >= min(domain.size) / 2.0:
                raise ValueError("penetrable_spheres: spheres are too large for the periodic box")
            box = domain
        else:
            box = domain.dilated(2.0 * rmax)
        if self.number is not None:
            n = int(self.number)
        else:
            n = round(self.reduced_density * box.volume / self._mean_volume())
        radii = self.radius.sample(rng, n)
        cores = self.impenetrability * radii
        if n == 0 or self.impenetrability == 0.0:
            local = rng.random((n, 3)) * box.lengths
        elif self.method == "lubachevsky_stillinger":
            local = grown_packing(cores, box.lengths, rng, growth_rate=self.growth_rate,
                                  max_events=self.max_attempts)  # fmt: skip
        else:
            local = largest_first(cores, box.lengths, 0.0, rng, self.max_attempts or 10_000 * n)
        spheres = Spheres(box.lower + local, radii)
        if not domain.periodic:
            spheres = spheres.subset(domain.distance_to_box(spheres.centers) <= spheres.radii)
        return PrimitiveUnion(domain, spheres=spheres, model=self, seed=seed_info)

    def expected_count(self, domain: Domain) -> float | None:
        if self.reduced_density is None:
            return None
        return self.reduced_density * domain.volume / self._mean_volume()

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The reduced density, the fraction of the cores (``λ^3 η``), and the fraction and
        specific surface of the union in the two limits: ``1 - e^{-η}`` and its surface
        ``3 η E[R^2] / E[R^3] e^{-η}`` for fully penetrable spheres (large boxes), ``η`` and
        ``3 η E[R^2] / E[R^3]`` for hard spheres."""
        if self.reduced_density is None:
            return {}
        eta, lam = float(self.reduced_density), float(self.impenetrability)
        out = {
            "reduced_density": eta,
            "core_fraction": eta * lam**3,
            "volume_fraction_lower": 1.0 - math.exp(-eta),
            "volume_fraction_upper": eta,
        }
        surface = 3.0 * eta * self.radius.moment(2) / self.radius.moment(3)
        if lam == 0.0:
            bare = math.exp(-eta)
            out.update(volume_fraction=1.0 - bare, specific_surface=surface * bare)
        elif lam == 1.0:
            out.update(volume_fraction=eta, specific_surface=surface)
        return out
