"""Power-law tails of samples: maximum likelihood with the lower bound of the tail chosen by
the Kolmogorov-Smirnov distance (Clauset, Shalizi & Newman 2009).

A sample follows ``p(x) ~ x^-a`` above ``x_min``. For continuous values the estimate is
``a = 1 + n / sum ln(x_i / x_min)``, of standard error ``(a - 1) / sqrt(n)``; for
integers (degrees, counts), ``p(x) = x^-a / zeta(a, x_min)`` (Hurwitz's zeta), the
likelihood maximized numerically, of standard error ``1 / sqrt(n (zeta''/zeta -
(zeta'/zeta)^2))``. Unless given, ``x_min`` is the value of the sample that minimizes the
largest distance between the empirical distribution of the tail and the fitted law, among
those that leave at least ``min_tail`` values in the tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special


@dataclass(frozen=True)
class PowerLawTail:
    """The tail ``p(x) ~ x^-exponent`` of a sample above ``x_min``: its ``n_tail`` values,
    the standard error of the exponent and the Kolmogorov-Smirnov distance of the fit."""

    exponent: float
    x_min: float
    n_tail: int
    error: float
    ks: float
    discrete: bool

    def to_dict(self) -> dict:
        return {
            "exponent": self.exponent,
            "x_min": self.x_min,
            "n_tail": self.n_tail,
            "error": self.error,
            "ks": self.ks,
            "discrete": self.discrete,
        }


def _continuous(tail: np.ndarray, x_min: float) -> tuple[float, float]:
    n = len(tail)
    s = float(np.sum(np.log(tail / x_min)))
    if not s > 0:
        return math.inf, math.inf
    a = 1.0 + n / s
    cdf = 1.0 - (tail / x_min) ** (1.0 - a)  # the fitted law at the sorted values
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(n) / n
    return a, float(max(np.max(np.abs(emp_hi - cdf)), np.max(np.abs(cdf - emp_lo))))


def _discrete_exponent(tail: np.ndarray, x_min: float) -> float:
    logs = float(np.mean(np.log(tail)))

    def nll(a):
        return math.log(special.zeta(a, x_min)) + a * logs

    res = optimize.minimize_scalar(nll, bounds=(1.0 + 1e-6, 20.0), method="bounded",
                                   options={"xatol": 1e-9})  # fmt: skip
    return float(res.x)


def _discrete(tail: np.ndarray, x_min: float) -> tuple[float, float]:
    if np.all(tail == x_min):
        return math.inf, math.inf
    a = _discrete_exponent(tail, x_min)
    values, counts = np.unique(tail, return_counts=True)
    emp = np.cumsum(counts) / len(tail)
    norm = special.zeta(a, x_min)
    cdf = 1.0 - special.zeta(a, values + 1.0) / norm  # P(X <= v)
    below = 1.0 - special.zeta(a, values) / norm  # P(X < v): the integers below v too
    emp_below = np.concatenate([[0.0], emp[:-1]])
    return a, float(max(np.max(np.abs(emp - cdf)), np.max(np.abs(emp_below - below))))


def _discrete_error(a: float, x_min: float, n: int) -> float:
    """The standard error of the discrete estimate, from the curvature of the likelihood
    (central differences of ``ln zeta(a, x_min)``)."""
    h = 1e-4 * max(a, 1.0)

    def f(x):
        return math.log(special.zeta(x, x_min))

    curvature = (f(a + h) - 2.0 * f(a) + f(a - h)) / (h * h)
    return 1.0 / math.sqrt(n * curvature) if curvature > 0 else math.inf


#: Largest number of lower bounds of the tail tried (quantiles of the sample beyond it)
CANDIDATES = 400


def power_law_tail(
    values,
    *,
    x_min: float | None = None,
    discrete: bool = False,
    min_tail: int = 10,
) -> PowerLawTail:
    """The power-law tail of a sample (see the module): its exponent by maximum
    likelihood above ``x_min``, given or chosen by the Kolmogorov-Smirnov distance among
    the values that leave at least ``min_tail`` of them in the tail (at most
    :data:`CANDIDATES` of them, quantiles of the sample). ``discrete``: integer values (at
    least 1), the law ``x^-a / zeta(a, x_min)``."""
    x = np.sort(np.asarray(values, dtype=float).ravel())
    x = x[np.isfinite(x)]
    if discrete:
        if np.any(x != np.round(x)):
            raise ValueError("power_law_tail: discrete values must be integers")
        x = x[x >= 1]
    else:
        x = x[x > 0]
    if min_tail < 2:
        raise ValueError("power_law_tail: min_tail of at least 2 values")
    if len(x) < min_tail:
        raise ValueError(f"power_law_tail: at least {min_tail} positive values, got {len(x)}")
    fit = _discrete if discrete else _continuous
    if x_min is not None:
        x_min = float(x_min)
        tail = x[x >= x_min]
        if len(tail) < 2:
            raise ValueError("power_law_tail: fewer than two values above x_min")
        best = (x_min, *fit(tail, x_min))
    else:
        candidates = np.unique(x[: len(x) - min_tail + 1])
        if len(candidates) > CANDIDATES:
            ranks = np.linspace(0, len(x) - min_tail, CANDIDATES).astype(np.int64)
            candidates = np.unique(x[ranks])
        best = (math.nan, math.nan, math.inf)
        for xm in candidates:
            a, ks = fit(x[np.searchsorted(x, xm) :], float(xm))
            if ks < best[2]:
                best = (float(xm), a, ks)
        if not math.isfinite(best[1]):
            raise ValueError("power_law_tail: no tail with distinct values")
    xm, a, ks = best
    n = int(np.count_nonzero(x >= xm))
    error = _discrete_error(a, xm, n) if discrete else (a - 1.0) / math.sqrt(n)
    return PowerLawTail(float(a), float(xm), n, float(error), float(ks), bool(discrete))
