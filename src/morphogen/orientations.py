"""Orientation distributions of fibre axes, and random crystallographic orientations.

Fibre axes are given in configuration files as::

    orientation: isotropic                              # uniform on the unit sphere
    orientation: {type: aligned, direction: [0, 0, 1]}  # all fibres along a direction
    orientation: {type: planar, normal: [0, 0, 1]}      # uniform in the plane normal to `normal`
    orientation: {type: beta, beta: 0.1, axis: [0, 0, 1]}  # Schladitz's beta law about `axis`
    orientation: {type: girdle, sigma: 0.4, axis: [0, 0, 1]}  # Gaussian girdle (Chatti et al.)
    orientation: {type: von_mises_fisher, kappa: 10, axis: [0, 0, 1]}  # about `axis`
    orientation: {type: mixture, laws: [...], weights: [0.4, 0.6]}  # Altendorf et al. 2014

Every law has a quadrature (:meth:`OrientationDistribution.quadrature`): axes and weights
that average functions of the axis (the covariances of the fibre models), and
:func:`angle_weights` gives the law of the angle between the axes and a direction.

Crystal orientations are Bunge Euler angles (phi1, Phi, phi2), convention ZXZ, of
rotations drawn uniformly on SO(3), of a fibre texture about the axes of one of these
laws (:func:`fibre_orientations`), or of the texture of an orientation map::

    texture: {type: measured, file: map.ctf, spread: 2.0}  # the grains of an EBSD map
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from morphogen._utils import unit_vector

#: Angles of the uniform grid of [0, pi/2] on which functions of the angle between an axis
#: and a direction are tabulated when a law has more axes than that (:func:`angle_weights`)
ANGLE_GRID = 193


def uniform_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` directions uniformly distributed on the unit sphere."""
    cos_t = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


def orthonormal_basis(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing each direction into a right-handed orthonormal basis."""
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    # pick the coordinate axis least aligned with each direction
    helper = np.zeros_like(d)
    helper[np.arange(len(d)), np.argmin(np.abs(d), axis=1)] = 1.0
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)
    return e1, e2


class OrientationDistribution(ABC):
    """Distribution of undirected axes."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` unit vectors, shape (n, 3)."""

    @abstractmethod
    def to_config(self):
        """Plain configuration value."""

    def quadrature(self, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
        """Axes (m x 3) and weights (m, summing to 1) such that ``sum(w f(u))``
        approximates ``E[f(u)]`` for smooth functions of the axis. The laws below have
        rules of their own; this default takes ``8 n^2`` axes drawn from a fixed seed (a
        deterministic Monte Carlo rule)."""
        u = self.sample(np.random.default_rng(2026), 8 * n * n).reshape(-1, 3)
        return u, np.full(len(u), 1.0 / len(u))


def _about(axis, cosines, weights, azimuths: int) -> tuple[np.ndarray, np.ndarray]:
    """The product rule of the axes at the polar ``cosines`` (of the given weights)
    about ``axis``, each at ``azimuths`` equally spaced azimuths."""
    a = np.asarray(axis, dtype=float)
    e1, e2 = orthonormal_basis(a[None])
    phi = (np.arange(azimuths) + 0.5) * (2.0 * np.pi / azimuths)
    c = np.clip(np.asarray(cosines, dtype=float), -1.0, 1.0)[:, None, None]
    side = np.cos(phi)[None, :, None] * e1 + np.sin(phi)[None, :, None] * e2
    u = c * a + np.sqrt(1.0 - c * c) * side
    w = np.repeat(np.asarray(weights, dtype=float) / azimuths, azimuths)
    return u.reshape(-1, 3), w


def _legendre(n: int, lo: float = 0.0, hi: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights of ``[lo, hi]`` (weights summing to its length)."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return lo + 0.5 * (x + 1.0) * (hi - lo), 0.5 * (hi - lo) * w


@dataclass(frozen=True)
class Isotropic(OrientationDistribution):
    def sample(self, rng, n):
        return uniform_unit_vectors(rng, int(n))

    def to_config(self):
        return "isotropic"

    def quadrature(self, n=24):
        """Gauss-Legendre in the cosine to z (``n`` nodes) times ``2n`` azimuths."""
        c, w = _legendre(n, -1.0, 1.0)
        return _about((0.0, 0.0, 1.0), c, 0.5 * w, 2 * n)


@dataclass(frozen=True)
class Aligned(OrientationDistribution):
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        u = unit_vector(self.direction, "aligned.direction")
        object.__setattr__(self, "direction", tuple(float(c) for c in u))

    def sample(self, rng, n):
        return np.tile(np.array(self.direction), (int(n), 1))

    def to_config(self):
        return {"type": "aligned", "direction": list(self.direction)}

    def quadrature(self, n=24):
        return np.array([self.direction]), np.ones(1)


@dataclass(frozen=True)
class Planar(OrientationDistribution):
    """Directions uniformly distributed in the plane orthogonal to ``normal``."""

    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        u = unit_vector(self.normal, "planar.normal")
        if np.max(np.abs(u - np.asarray(self.normal, dtype=float))) <= 1e-15:
            u = np.asarray(self.normal, dtype=float)  # a unit normal kept (round trips)
        object.__setattr__(self, "normal", tuple(float(c) for c in u))

    def sample(self, rng, n):
        e1, e2 = orthonormal_basis(np.array([self.normal]))
        phi = rng.uniform(0.0, 2.0 * np.pi, int(n))[:, None]
        return np.cos(phi) * e1 + np.sin(phi) * e2

    def to_config(self):
        return {"type": "planar", "normal": list(self.normal)}

    def quadrature(self, n=24):
        """``8n`` equally spaced azimuths in the plane (the trapezoidal rule)."""
        return _about(self.normal, np.zeros(1), np.ones(1), 8 * n)


@dataclass(frozen=True)
class Beta(OrientationDistribution):
    """The beta law of Schladitz et al. (2006) about ``axis`` (Ohser & Schladitz 2009),
    of density ``beta sin(t) / (4 pi (1 + (beta^2 - 1) cos^2 t)^(3/2))`` in the angle
    ``t`` to the axis: uniform for ``beta = 1``, concentrated on the axis as ``beta -> 0``
    and on the plane normal to it as ``beta -> infinity``. Drawn as isotropic directions
    whose component along the axis is divided by ``beta`` (the angular central Gaussian
    law of ``diag(1, 1, 1 / beta^2)``), so that ``P(|cos t| <= c) = beta c / sqrt(1 +
    (beta^2 - 1) c^2)``."""

    beta: float = 1.0
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.beta) and self.beta > 0.0):
            raise ValueError(f"beta.beta: a positive number, got {self.beta!r}")
        u = unit_vector(self.axis, "beta.axis")
        if np.max(np.abs(u - np.asarray(self.axis, dtype=float))) <= 1e-15:
            u = np.asarray(self.axis, dtype=float)  # a unit axis kept (round trips)
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "axis", tuple(float(c) for c in u))

    def sample(self, rng, n):
        v = uniform_unit_vectors(rng, int(n))
        a = np.array(self.axis)
        v += (1.0 / self.beta - 1.0) * (v @ a)[:, None] * a
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def to_config(self):
        return {"type": "beta", "beta": self.beta, "axis": list(self.axis)}

    def quadrature(self, n=24):
        """The quantiles ``c = u / sqrt(beta^2 (1 - u^2) + u^2)`` of ``|cos t|`` at ``n``
        Gauss-Legendre nodes ``u`` of [0, 1], both signs, times ``2n`` azimuths."""
        u, w = _legendre(n)
        c = u / np.sqrt(self.beta**2 * (1.0 - u * u) + u * u)
        return _about(self.axis, np.concatenate([-c, c]), np.concatenate([w, w]) / 2.0, 2 * n)


@dataclass(frozen=True)
class Girdle(OrientationDistribution):
    """The Gaussian girdle of Chatti et al. (2018, eq. 3) about ``axis``: the angle ``t`` to
    the axis of density proportional to ``sin(t) exp(-(t - pi/2)^2 / 2 sigma^2)`` on [0, pi]
    (``sigma`` in radians), the azimuth uniform: axes gathered about the plane normal to the
    axis as ``sigma -> 0``, isotropic as ``sigma`` grows (their compressed entangled fibres:
    0.4; at most 20, where the density is within 0.4 % of the isotropic one: the rejection
    below keeps fewer draws as ``sigma`` grows). Drawn by rejection: normal angles truncated
    to [0, pi], each kept with the probability ``sin(t)``."""

    sigma: float = 0.4
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.sigma) and self.sigma > 0.0):
            raise ValueError(f"girdle.sigma: a positive angle in radians, got {self.sigma!r}")
        if self.sigma > 20.0:
            raise ValueError(
                f"girdle.sigma: at most 20 radians, got {self.sigma!r} (the girdle is then "
                "isotropic within 0.4 %: use the isotropic law)"
            )
        u = unit_vector(self.axis, "girdle.axis")
        if np.max(np.abs(u - np.asarray(self.axis, dtype=float))) <= 1e-15:
            u = np.asarray(self.axis, dtype=float)  # a unit axis kept (round trips)
        object.__setattr__(self, "sigma", float(self.sigma))
        object.__setattr__(self, "axis", tuple(float(c) for c in u))

    def sample(self, rng, n):
        n = int(n)
        angles = np.empty(0)
        while len(angles) < n:
            t = rng.normal(0.5 * np.pi, self.sigma, max(2 * (n - len(angles)), 64))
            t = t[(t >= 0.0) & (t <= np.pi)]
            angles = np.concatenate([angles, t[rng.random(len(t)) < np.sin(t)]])
        t = angles[:n]
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        a = np.array(self.axis)
        e1, e2 = orthonormal_basis(a)
        side = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
        return np.cos(t)[:, None] * a + np.sin(t)[:, None] * side

    def to_config(self):
        return {"type": "girdle", "sigma": self.sigma, "axis": list(self.axis)}

    def quadrature(self, n=24):
        """``2n`` Gauss-Legendre nodes of the angle ``t`` over ``pi/2 +- 8 sigma`` (within
        [0, pi]) weighted by its density, times ``2n`` azimuths."""
        lo, hi = (
            max(0.0, 0.5 * np.pi - 8.0 * self.sigma),
            min(np.pi, 0.5 * np.pi + 8.0 * self.sigma),
        )
        t, w = _legendre(2 * n, lo, hi)
        w = w * np.sin(t) * np.exp(-((t - 0.5 * np.pi) ** 2) / (2.0 * self.sigma**2))
        return _about(self.axis, np.cos(t), w / w.sum(), 2 * n)


@dataclass(frozen=True)
class VonMisesFisher(OrientationDistribution):
    """The von Mises-Fisher law about ``axis``, of density ``kappa exp(kappa cos t) / (4 pi
    sinh kappa)`` in the angle ``t`` to the axis: uniform for ``kappa = 0``, concentrated
    on the axis as ``kappa`` grows (``E[1 - cos t] ~ 1 / kappa``). Drawn by inverting the
    law of ``w = cos t``, ``w = 1 + ln(u + (1 - u) e^(-2 kappa)) / kappa``, the azimuth
    uniform (as the steps of :func:`morphogen.models.fibers.von_mises_fisher`)."""

    kappa: float = 10.0
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.kappa) and self.kappa >= 0.0):
            raise ValueError(f"von_mises_fisher.kappa: a non-negative number, got {self.kappa!r}")
        u = unit_vector(self.axis, "von_mises_fisher.axis")
        if np.max(np.abs(u - np.asarray(self.axis, dtype=float))) <= 1e-15:
            u = np.asarray(self.axis, dtype=float)  # a unit axis kept (round trips)
        object.__setattr__(self, "kappa", float(self.kappa))
        object.__setattr__(self, "axis", tuple(float(c) for c in u))

    def _cosines(self, u) -> np.ndarray:
        """The quantiles of ``cos t`` at the probabilities ``u`` in (0, 1]."""
        k = self.kappa
        if k < 1e-8:
            return 2.0 * u - 1.0
        return 1.0 + np.log(u + (1.0 - u) * math.exp(-2.0 * k)) / k

    def sample(self, rng, n):
        n = int(n)
        w = self._cosines(1.0 - rng.random(n))
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        a = np.array(self.axis)
        e1, e2 = orthonormal_basis(a[None])
        s = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
        return w[:, None] * a + s[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)

    def to_config(self):
        return {"type": "von_mises_fisher", "kappa": self.kappa, "axis": list(self.axis)}

    def quadrature(self, n=24):
        """The quantiles of ``cos t`` at ``2n`` Gauss-Legendre nodes of [0, 1], times
        ``2n`` azimuths."""
        u, w = _legendre(2 * n)
        return _about(self.axis, self._cosines(u), w, 2 * n)


@dataclass(frozen=True)
class Mixture(OrientationDistribution):
    """A mixture of laws: an axis follows ``laws[i]`` with the probability ``weights[i]``
    (normalized), as the mixed beta laws of Altendorf, Jeulin & Willot (2014, eq. 2), an
    orthotropic law of glass fibres (0.393 of a beta law of 0.114 about z, 0.607 of one of
    10.1 about a normal to it)."""

    laws: tuple[OrientationDistribution, ...] = (Isotropic(),)
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        laws = tuple(self.laws)
        if not laws or not all(isinstance(law, OrientationDistribution) for law in laws):
            raise ValueError("mixture.laws: one or more orientation laws")
        w = np.ones(len(laws)) if self.weights is None else np.asarray(self.weights, dtype=float)
        if w.shape != (len(laws),) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("mixture.weights: one non-negative weight per law, not all zero")
        if abs(w.sum() - 1.0) > 1e-12:  # normalized weights kept (round trips)
            w = w / w.sum()
        object.__setattr__(self, "laws", laws)
        object.__setattr__(self, "weights", tuple(float(x) for x in w))

    def sample(self, rng, n):
        n = int(n)
        which = rng.choice(len(self.laws), size=n, p=np.asarray(self.weights))
        out = np.empty((n, 3))
        for i, law in enumerate(self.laws):
            idx = np.flatnonzero(which == i)
            if len(idx):
                out[idx] = law.sample(rng, len(idx)).reshape(-1, 3)
        return out

    def to_config(self):
        return {"type": "mixture", "laws": [law.to_config() for law in self.laws],
                "weights": list(self.weights)}  # fmt: skip

    def quadrature(self, n=24):
        """The rules of the laws, their weights times those of the mixture."""
        rules = [law.quadrature(n) for law in self.laws]
        u = np.concatenate([r[0].reshape(-1, 3) for r in rules])
        w = np.concatenate([p * r[1] for p, r in zip(self.weights, rules, strict=True)])
        return u, w


@dataclass(frozen=True)
class Measured(OrientationDistribution):
    """The texture of an orientation map (EBSD, :mod:`morphogen.io.ebsd`): the crystals
    have the orientations of the grains of the map (``file``: an ``.ang`` or a ``.ctf``
    file, or the name of a map that comes with morphoGen; ``threshold`` and
    ``min_size``: those of its grains; ``axes`` and ``setting``: those of its
    orientations), each drawn with the area of its grain, then turned about a random
    axis by an angle of the half-normal law of standard deviation ``spread`` degrees (0:
    the orientations of the grains themselves). ``phase``: the phase of the map whose
    grains are taken (the one of largest area by default). As a law of axes, it is that
    of the third axes of these crystals."""

    file: str = ""
    spread: float = 0.0
    threshold: float = 5.0
    min_size: int = 5
    phase: int | None = None
    axes: str = "map"
    setting: int = 1

    def __post_init__(self) -> None:
        if not self.file:
            raise ValueError("measured.file: the orientation map (.ang or .ctf)")
        if not (np.isfinite(self.spread) and self.spread >= 0.0):
            raise ValueError(f"measured.spread: an angle, not negative, got {self.spread!r}")
        object.__setattr__(self, "file", str(self.file))
        object.__setattr__(self, "spread", float(self.spread))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "min_size", int(self.min_size))
        self._model()  # (the checks of its other parameters)

    def _model(self):
        from morphogen.models.ebsd import EBSD

        try:
            return EBSD(file=self.file, threshold=self.threshold, min_size=self.min_size,
                        axes=self.axes, setting=self.setting)  # fmt: skip
        except ValueError as exc:
            raise ValueError(f"measured.{exc}") from None

    def grains(self):
        """The orientations (Bunge angles, degrees) and the areas of the grains of the
        phase of the map, and the Laue class of that phase."""
        grains = self._model().grains()
        phase = self.phase
        if phase is None:
            areas = np.bincount(grains.phase, weights=grains.areas)
            phase = int(np.argmax(areas))
        of = grains.phase == phase
        if not of.any():
            raise ValueError(f"measured.phase: no grain of the phase {phase} in the map")
        return grains.orientations[of], grains.areas[of], grains.lattice(phase)

    def orientations(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Bunge angles (degrees) of ``n`` crystals of the texture of the map."""
        from morphogen.texture import sample_texture

        euler, areas, _ = self.grains()
        return sample_texture(rng, euler, int(n), weights=areas, spread=self.spread)

    def sample(self, rng, n):
        return euler_to_matrix(self.orientations(rng, n))[:, 2, :]

    def to_config(self):
        out = {"type": "measured", "file": self.file}
        defaults = {"spread": 0.0, "threshold": 5.0, "min_size": 5, "phase": None, "axes": "map",
                    "setting": 1}  # fmt: skip
        for name, default in defaults.items():
            if getattr(self, name) != default:
                out[name] = getattr(self, name)
        return out


def half_sphere(n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Directions of the half sphere ``z >= 0`` and their weights (summing to 1): ``n``
    Gauss-Legendre cosines times ``2n`` azimuths, the mean over the directions of an even
    function (``f(-u) = f(u)``)."""
    c, w = _legendre(n)
    return _about((0.0, 0.0, 1.0), c, w, 2 * n)


def angle_weights(orientation: OrientationDistribution, direction=None, n: int = 24, *,
                  grid: bool = False):  # fmt: skip
    """The angles ``b`` in [0, pi/2] between the axes of ``orientation`` and ``direction``
    (between undirected lines) and their weights, so that ``sum(w f(b))`` approximates
    ``E[f(b)]``: 128 Gauss-Legendre nodes of ``cos b`` in [0, 1] for isotropic axes or
    without a direction (a direction uniform on the sphere); otherwise the angles of the
    axes of the quadrature of the law, gathered on the :data:`ANGLE_GRID` nodes of a
    uniform grid of [0, pi/2] when there are more, or always with ``grid`` (the weight of
    an angle shared by the two nodes about it, in proportion to their nearness: linear
    interpolation of ``f``)."""
    if (direction is None or isinstance(orientation, Isotropic)) and not grid:
        c, w = _legendre(128)
        return np.arccos(c), w
    if direction is None:
        orientation, direction = Isotropic(), (0.0, 0.0, 1.0)
    d = unit_vector(direction, "direction")
    u, w = orientation.quadrature(n)
    angle = np.arccos(np.clip(np.abs(u.reshape(-1, 3) @ d), 0.0, 1.0))
    if len(angle) <= ANGLE_GRID and not grid:
        return angle, w
    grid = np.linspace(0.0, 0.5 * np.pi, ANGLE_GRID)
    pos = angle / grid[1]
    i = np.minimum(pos.astype(np.int64), ANGLE_GRID - 2)
    frac = pos - i
    out = np.zeros(ANGLE_GRID)
    np.add.at(out, i, w * (1.0 - frac))
    np.add.at(out, i + 1, w * frac)
    return grid, out


_TYPES = "isotropic, aligned, planar, beta, girdle, von_mises_fisher, mixture, measured"


def orientation_from_config(value, name: str = "orientation") -> OrientationDistribution:
    """An orientation distribution from a configuration value: its name (``isotropic``) or
    a mapping with its ``type`` (isotropic, aligned, planar, beta, girdle,
    von_mises_fisher, mixture; measured: the texture of an orientation map) and
    parameters."""
    if isinstance(value, OrientationDistribution):
        return value
    if isinstance(value, str):
        value = {"type": value}
    if not isinstance(value, dict) or "type" not in value:
        raise ValueError(
            f"{name}: expected 'isotropic' or a mapping with a 'type' key ({_TYPES}), got {value!r}"
        )
    cfg = dict(value)
    kind = str(cfg.pop("type")).lower()
    if kind == "mixture" and isinstance(cfg.get("laws"), (list, tuple)):  # the laws first
        laws = cfg["laws"]
        cfg["laws"] = tuple(
            orientation_from_config(v, f"{name}.laws[{i}]") for i, v in enumerate(laws)
        )
    try:
        if kind == "isotropic" and not cfg:
            return Isotropic()
        if kind == "aligned" and set(cfg) <= {"direction"}:
            return Aligned(tuple(cfg.get("direction", (0.0, 0.0, 1.0))))
        if kind == "planar" and set(cfg) <= {"normal"}:
            return Planar(tuple(cfg.get("normal", (0.0, 0.0, 1.0))))
        if kind == "beta" and set(cfg) <= {"beta", "axis"}:
            return Beta(float(cfg.get("beta", 1.0)), tuple(cfg.get("axis", (0.0, 0.0, 1.0))))
        if kind == "girdle" and set(cfg) <= {"sigma", "axis"}:
            axis = tuple(cfg.get("axis", (0.0, 0.0, 1.0)))
            return Girdle(float(cfg.get("sigma", 0.4)), axis)
        if kind in ("von_mises_fisher", "vmf") and set(cfg) <= {"kappa", "axis"}:
            axis = tuple(cfg.get("axis", (0.0, 0.0, 1.0)))
            return VonMisesFisher(float(cfg.get("kappa", 10.0)), axis)
        if kind == "measured" and set(cfg) <= {"file", "spread", "threshold", "min_size",
                                               "phase", "axes", "setting"}:  # fmt: skip
            return Measured(**cfg)
        if kind == "mixture" and "laws" in cfg and set(cfg) <= {"laws", "weights"}:
            if not isinstance(cfg["laws"], tuple):
                raise ValueError("mixture.laws: a list of orientation laws")
            weights = cfg.get("weights")
            return Mixture(cfg["laws"], None if weights is None else tuple(map(float, weights)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: {exc}") from None
    raise ValueError(f"{name}: invalid orientation {value!r}")


def random_orientations(rng: np.random.Generator, n: int, *, degrees: bool = True) -> np.ndarray:
    """Bunge Euler angles (phi1, Phi, phi2) of ``n`` rotations uniform on SO(3).

    Uniformity on SO(3) means phi1 and phi2 uniform on [0, 2 pi) and cos(Phi) uniform on
    [-1, 1] (the Haar measure is sin(Phi) dphi1 dPhi dphi2 / 8 pi^2).
    """
    phi1 = rng.uniform(0.0, 2.0 * np.pi, n)
    Phi = np.arccos(rng.uniform(-1.0, 1.0, n))
    phi2 = rng.uniform(0.0, 2.0 * np.pi, n)
    euler = np.stack([phi1, Phi, phi2], axis=1)
    return np.degrees(euler) if degrees else euler


def fibre_orientations(
    rng: np.random.Generator, n: int, law: OrientationDistribution, *, degrees: bool = True
) -> np.ndarray:
    """Bunge Euler angles of ``n`` crystals of a fibre texture: the third axis of each
    crystal (the axis c of a hexagonal lattice) is a direction of the sample drawn from
    ``law`` (:class:`VonMisesFisher` about an axis: a basal texture; :class:`Planar`: axes
    c in a plane...), and the crystal is turned about it by a uniform angle. A measured
    texture (:class:`Measured`) gives the orientations of the grains of its map."""
    if isinstance(law, Measured):
        euler = law.orientations(rng, int(n))
        return euler if degrees else np.radians(euler)
    axes = np.asarray(law.sample(rng, int(n)), dtype=float).reshape(-1, 3)
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    e1, e2 = orthonormal_basis(axes)
    spin = rng.uniform(0.0, 2.0 * np.pi, len(axes))
    c, s = np.cos(spin)[:, None], np.sin(spin)[:, None]
    # the rows of g: the axes of the crystal in those of the sample (crystal = g @ sample)
    g = np.stack([c * e1 + s * e2, -s * e1 + c * e2, axes], axis=1)
    return matrix_to_euler(g, degrees=degrees)


def euler_to_matrix(euler, *, degrees: bool = True) -> np.ndarray:
    """Bunge (ZXZ, passive) rotation matrices g, crystal = g @ sample, shape (n, 3, 3)."""
    e = np.asarray(euler, dtype=float).reshape(-1, 3)
    if degrees:
        e = np.radians(e)
    c1, s1 = np.cos(e[:, 0]), np.sin(e[:, 0])
    c, s = np.cos(e[:, 1]), np.sin(e[:, 1])
    c2, s2 = np.cos(e[:, 2]), np.sin(e[:, 2])
    g = np.empty((len(e), 3, 3))
    g[:, 0, 0] = c1 * c2 - s1 * s2 * c
    g[:, 0, 1] = s1 * c2 + c1 * s2 * c
    g[:, 0, 2] = s2 * s
    g[:, 1, 0] = -c1 * s2 - s1 * c2 * c
    g[:, 1, 1] = -s1 * s2 + c1 * c2 * c
    g[:, 1, 2] = c2 * s
    g[:, 2, 0] = s1 * s
    g[:, 2, 1] = -c1 * s
    g[:, 2, 2] = c
    return g


def matrix_to_euler(g, *, degrees: bool = True) -> np.ndarray:
    """Inverse of :func:`euler_to_matrix`; angles in [0, 360) x [0, 180] x [0, 360)."""
    g = np.asarray(g, dtype=float).reshape(-1, 3, 3)
    Phi = np.arccos(np.clip(g[:, 2, 2], -1.0, 1.0))
    sin_phi = np.sin(Phi)
    regular = sin_phi > 1e-9
    phi1 = np.where(
        regular, np.arctan2(g[:, 2, 0], -g[:, 2, 1]), np.arctan2(g[:, 0, 1], g[:, 0, 0])
    )
    phi2 = np.where(regular, np.arctan2(g[:, 0, 2], g[:, 1, 2]), 0.0)
    euler = np.stack([np.mod(phi1, 2 * np.pi), Phi, np.mod(phi2, 2 * np.pi)], axis=1)
    return np.degrees(euler) if degrees else euler
