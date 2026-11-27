"""The sizes of the objects of the models (the radii of fibres and struts...): monodisperse
when the parameter is a number (a constant law, which draws nothing from the random
generator, so that the realizations stay those of a fixed size), polydisperse when it is a
distribution, one size per object; and the scale of a law solved on a realization for a
volume fraction (the shape of the law kept)."""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize

from morphogen.distributions import Distribution


def polydisperse(law) -> bool:
    """Whether a size parameter draws a size per object (a law that is not a constant)."""
    return isinstance(law, Distribution) and not law.is_constant


def draw(law: Distribution, rng, n: int) -> np.ndarray:
    """One size per object (``n`` of them); a constant law uses no random number."""
    return np.asarray(law.sample(rng, int(n)), dtype=float).reshape(-1)


def size_summary(name: str, sizes, law) -> dict:
    """The mean and standard deviation of the sizes drawn on a realization
    (``<name>_mean``, ``<name>_std``), for a polydisperse law; nothing for a number."""
    sizes = np.asarray(sizes, dtype=float).reshape(-1)
    if not polydisperse(law) or not sizes.size:
        return {}
    return {f"{name}_mean": float(sizes.mean()), f"{name}_std": float(sizes.std())}


def solve_scale(fraction_of, target: float, guess: float, limit: float, xtol: float) -> float:
    """The scale ``s`` of the sizes at which ``fraction_of(s)`` (a volume fraction growing
    with ``s``) reaches ``target``: bracketed from ``[guess / 4, 2 guess]``, doubled until
    above it (at most ``limit``), then Brent's method."""

    def excess(s):
        return fraction_of(s) - target

    lo, hi = 0.25 * guess, 2.0 * guess
    while excess(hi) < 0 and hi < limit:
        lo, hi = hi, 2.0 * hi
    return float(optimize.brentq(excess, lo, hi, xtol=xtol))


def sum_moments(law: Distribution, gap: float = 0.0) -> tuple[float, float, float]:
    """The first three moments of ``D = R_1 + R_2 + gap``, two independent sizes of a law
    and a constant: the distance between the axes of two fibres in contact."""
    m1, m2, m3 = (law.moment(k) for k in (1, 2, 3))
    s1 = 2.0 * m1  # S = R_1 + R_2
    s2 = 2.0 * m2 + 2.0 * m1 * m1
    s3 = 2.0 * m3 + 6.0 * m2 * m1
    g = float(gap)
    return s1 + g, s2 + 2.0 * g * s1 + g * g, s3 + 3.0 * g * s2 + 3.0 * g * g * s1 + g**3


def node_radii(bars: np.ndarray, radii: np.ndarray, n_nodes: int, ratio: float) -> np.ndarray:
    """The radii of the spheres at the nodes of a frame of struts: ``ratio`` times the
    largest radius of the struts that meet there (0 at a node without struts)."""
    out = np.zeros(int(n_nodes))
    bars = np.asarray(bars, dtype=np.int64).reshape(-1, 2)
    for k in (0, 1):
        np.maximum.at(out, bars[:, k], np.asarray(radii, dtype=float))
    return float(ratio) * out


def check_scale(law, volume_fraction, name: str) -> None:
    """A size parameter and a volume fraction: one of them, or the law of a polydisperse
    size whose scale the fraction sets (the shape of its law kept)."""
    if volume_fraction is None:
        if law is None:
            raise ValueError(
                f"give exactly one of '{name}' and 'volume_fraction' (neither is given)"
            )
        return
    if law is not None and not polydisperse(law):
        raise ValueError(
            f"give exactly one of '{name}' and 'volume_fraction', not both (got {name}="
            f"{law!r} and volume_fraction={volume_fraction!r}), or a distribution of "
            f"'{name}' whose scale the volume fraction sets"
        )


def first_order_fraction(law: Distribution, length_density: float) -> float:
    """``pi E[R^2] L_V``: the volume fraction of thin struts of radii of a law, ``L_V`` of
    axis per unit volume, to first order (their joints left out)."""
    return math.pi * law.moment(2) * float(length_density)
