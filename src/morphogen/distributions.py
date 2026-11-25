"""Probability distributions of positive scalars (radii, lengths, weights...).

A distribution is given in configuration files either as a number (constant) or as a
mapping with a ``type`` key::

    radius: 1.0
    radius: {type: uniform, low: 0.5, high: 1.5}
    radius: {type: lognormal, mean: 1.0, std: 0.3}
    radius: {type: normal, mean: 20.0, std: 13.5}      # truncated to positive values
    radius: {type: discrete, values: [1.0, 2.0], weights: [0.7, 0.3]}

Unbounded distributions are truncated at their ``1 - 1e-9`` quantile so that every
model can bound the extent of its grains (needed to generate unbiased samples).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import stats

from morphogen._utils import check_positive

#: Quantile at which unbounded distributions are truncated.
TAIL_QUANTILE = 1.0 - 1e-9


class Distribution(ABC):
    """Distribution of a positive random variable."""

    kind: str = ""

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` values."""

    @abstractmethod
    def moment(self, k: int) -> float:
        """Raw moment ``E[X**k]``."""

    @abstractmethod
    def upper(self) -> float:
        """Largest value that can be drawn."""

    @abstractmethod
    def quadrature(self, n: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Nodes and weights approximating ``E[f(X)]`` by ``sum(w * f(x))``."""

    @abstractmethod
    def to_config(self):
        """Plain configuration value (number or mapping)."""

    def mean(self) -> float:
        return self.moment(1)

    def std(self) -> float:
        return math.sqrt(max(self.moment(2) - self.moment(1) ** 2, 0.0))

    def median(self) -> float:
        x, w = self.quadrature(257)
        c = np.cumsum(w)
        return float(x[np.searchsorted(c, 0.5 * c[-1])])

    @property
    def is_constant(self) -> bool:
        return False

    def expect(self, f, n: int = 128) -> float:
        """``E[f(X)]`` for a vectorized function ``f``."""
        x, w = self.quadrature(n)
        return float(np.sum(w * f(x)))

    def scaled(self, factor: float) -> Distribution:
        """The law of ``factor X`` (``factor > 0``): the same shape at another scale, its
        moments ``factor^k E[X^k]`` (the law of the sizes of a model whose scale is solved
        for a volume fraction)."""
        factor = check_positive(factor, "scale factor")
        return self._scaled(factor)

    @abstractmethod
    def _scaled(self, factor: float) -> Distribution:
        """:meth:`scaled` for a positive factor."""


@dataclass(frozen=True)
class Constant(Distribution):
    value: float
    kind = "constant"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", check_positive(self.value, "constant value"))

    def sample(self, rng, n):
        return np.full(int(n), self.value)

    def moment(self, k):
        return self.value**k

    def upper(self):
        return self.value

    def quadrature(self, n=128):
        return np.array([self.value]), np.array([1.0])

    @property
    def is_constant(self):
        return True

    def median(self):
        return self.value

    def _scaled(self, factor):
        return Constant(factor * self.value)

    def to_config(self):
        return self.value


@dataclass(frozen=True)
class Uniform(Distribution):
    low: float
    high: float
    kind = "uniform"

    def __post_init__(self) -> None:
        lo = check_positive(self.low, "uniform.low", allow_zero=True)
        hi = check_positive(self.high, "uniform.high")
        if not hi > lo:
            raise ValueError(f"uniform: high must exceed low, got low={lo}, high={hi}")
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "high", hi)

    def sample(self, rng, n):
        return rng.uniform(self.low, self.high, int(n))

    def moment(self, k):
        a, b = self.low, self.high
        return (b ** (k + 1) - a ** (k + 1)) / ((k + 1) * (b - a))

    def upper(self):
        return self.high

    def quadrature(self, n=128):
        x, w = np.polynomial.legendre.leggauss(n)
        return self.low + (x + 1) * 0.5 * (self.high - self.low), 0.5 * w

    def _scaled(self, factor):
        return Uniform(factor * self.low, factor * self.high)

    def to_config(self):
        return {"type": "uniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Log-normal distribution given by the mean and standard deviation of X itself."""

    mean_value: float
    std_value: float
    kind = "lognormal"

    def __post_init__(self) -> None:
        m = check_positive(self.mean_value, "lognormal.mean")
        s = check_positive(self.std_value, "lognormal.std")
        object.__setattr__(self, "mean_value", m)
        object.__setattr__(self, "std_value", s)

    @property
    def sigma(self) -> float:
        return math.sqrt(math.log1p((self.std_value / self.mean_value) ** 2))

    @property
    def mu(self) -> float:
        return math.log(self.mean_value) - 0.5 * self.sigma**2

    @property
    def _frozen(self):
        return stats.lognorm(s=self.sigma, scale=math.exp(self.mu))

    def sample(self, rng, n):
        x = rng.lognormal(self.mu, self.sigma, int(n))
        return np.minimum(x, self.upper())

    def moment(self, k):
        return math.exp(k * self.mu + 0.5 * k * k * self.sigma**2)

    def upper(self):
        return float(self._frozen.ppf(TAIL_QUANTILE))

    def quadrature(self, n=128):
        u = (np.arange(n) + 0.5) / n
        return self._frozen.ppf(u), np.full(n, 1.0 / n)

    def _scaled(self, factor):
        return LogNormal(factor * self.mean_value, factor * self.std_value)

    def to_config(self):
        return {"type": "lognormal", "mean": self.mean_value, "std": self.std_value}


@dataclass(frozen=True)
class TruncatedNormal(Distribution):
    """Normal distribution N(mean, std) truncated to ``[low, high]`` (default: positive)."""

    loc: float
    scale: float
    low: float = 0.0
    high: float = math.inf
    kind = "normal"

    def __post_init__(self) -> None:
        object.__setattr__(self, "loc", float(self.loc))
        object.__setattr__(self, "scale", check_positive(self.scale, "normal.std"))
        object.__setattr__(self, "low", check_positive(self.low, "normal.low", allow_zero=True))
        object.__setattr__(self, "high", float(self.high))
        if not self.high > self.low:
            raise ValueError("normal: high must exceed low")
        if not (self.low < self.loc + 8 * self.scale and self.high > self.loc - 8 * self.scale):
            raise ValueError("normal: the truncation interval has negligible probability")

    @property
    def _frozen(self):
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        return stats.truncnorm(a, b, loc=self.loc, scale=self.scale)

    def sample(self, rng, n):
        x = self._frozen.rvs(size=int(n), random_state=rng)
        return np.minimum(np.atleast_1d(x).astype(float), self.upper())

    def moment(self, k):
        return float(self._frozen.moment(k))

    def upper(self):
        if math.isfinite(self.high):
            return float(self.high)
        return float(self._frozen.ppf(TAIL_QUANTILE))

    def quadrature(self, n=128):
        u = (np.arange(n) + 0.5) / n
        return self._frozen.ppf(u), np.full(n, 1.0 / n)

    def _scaled(self, factor):
        return TruncatedNormal(
            factor * self.loc, factor * self.scale, factor * self.low, factor * self.high
        )

    def to_config(self):
        cfg = {"type": "normal", "mean": self.loc, "std": self.scale}
        if self.low != 0.0:
            cfg["low"] = self.low
        if math.isfinite(self.high):
            cfg["high"] = self.high
        return cfg


@dataclass(frozen=True)
class Discrete(Distribution):
    values: tuple[float, ...]
    weights: tuple[float, ...]
    kind = "discrete"

    def __post_init__(self) -> None:
        v = tuple(check_positive(x, "discrete.values") for x in self.values)
        w = np.asarray(self.weights, dtype=float)
        if len(v) == 0 or w.shape != (len(v),):
            raise ValueError("discrete: 'values' and 'weights' must have the same non-zero length")
        if np.any(w < 0) or not w.sum() > 0:
            raise ValueError("discrete: weights must be non-negative and not all zero")
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "weights", tuple(float(x) for x in w / w.sum()))

    def sample(self, rng, n):
        return rng.choice(np.array(self.values), size=int(n), p=np.array(self.weights))

    def moment(self, k):
        return float(np.dot(self.weights, np.array(self.values) ** k))

    def upper(self):
        return max(self.values)

    def quadrature(self, n=128):
        return np.array(self.values), np.array(self.weights)

    def _scaled(self, factor):
        return Discrete(tuple(factor * v for v in self.values), self.weights)

    def to_config(self):
        return {"type": "discrete", "values": list(self.values), "weights": list(self.weights)}


def distribution_from_config(value, name: str = "distribution") -> Distribution:
    """Build a distribution from a number or a mapping (see the module docstring)."""
    if isinstance(value, Distribution):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected a number or a distribution mapping, got {value!r}")
    if isinstance(value, (int, float)):
        return Constant(float(value))
    if not isinstance(value, dict) or "type" not in value:
        raise ValueError(
            f"{name}: expected a number or a mapping with a 'type' key "
            f"(constant, uniform, lognormal, normal, discrete), got {value!r}"
        )
    cfg = dict(value)
    kind = str(cfg.pop("type")).lower()
    allowed = {
        "constant": {"value"},
        "uniform": {"low", "high"},
        "lognormal": {"mean", "std"},
        "normal": {"mean", "std", "low", "high"},
        "discrete": {"values", "weights"},
    }
    if kind not in allowed:
        raise ValueError(
            f"{name}: unknown distribution type {kind!r}; use one of {sorted(allowed)}"
        )
    unknown = set(cfg) - allowed[kind]
    missing = allowed[kind] - set(cfg) - ({"low", "high"} if kind == "normal" else set())
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)} for a {kind} distribution")
    if missing:
        raise ValueError(f"{name}: missing keys {sorted(missing)} for a {kind} distribution")
    try:
        if kind == "constant":
            return Constant(cfg["value"])
        if kind == "uniform":
            return Uniform(cfg["low"], cfg["high"])
        if kind == "lognormal":
            return LogNormal(cfg["mean"], cfg["std"])
        if kind == "normal":
            return TruncatedNormal(
                cfg["mean"], cfg["std"], cfg.get("low", 0.0), cfg.get("high", math.inf)
            )
        return Discrete(tuple(cfg["values"]), tuple(cfg["weights"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: {exc}") from None
