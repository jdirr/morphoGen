"""Measures of a realization on its implicit geometry: points, pairs, segments, balls and
lines probe the phases through their functions at points of space, without a voxel image.

A realization describes each phase by an implicit function at any point
(:meth:`~morphogen.models.base.Microstructure.phase_level_sets_at`): its sign says in which
phase a point lies, exactly, and a bound ``L`` of its slope (``lipschitz``) says how far
the interface is at least, ``|φ(x)| / L``. The measures of mathematical morphology are then
estimated from the exact answers of the probes, so that their only errors are those of
the sampling (which the estimates give, and which more probes reduce) and the fluctuation
of the realization itself — not the bias of a discretization:

* *points*: the fraction of a phase, ``P(x ∈ A)``;
* *pairs of points*: the covariance ``C(h) = P(x ∈ A, x + h ∈ A)`` along a direction, or
  averaged over the directions (a random one per pair);
* *segments*: ``P(K_x ∩ A ≠ ∅)`` for the segment ``K`` of length ``l`` from ``x``, by
  marching along it from the point by steps ``φ / L`` (sphere tracing, Hart 1996: a step
  never crosses the interface), all the lengths at once from the first hit: the Choquet
  capacity of segments and the linear contact distribution;
* *balls*: ``P(B(x, r) ∩ A ≠ ∅) = P(d(x, A) <= r)`` from the distance to the phase, exact
  outside it for the unions of spheres and segments (grains, fibres, struts: their
  function is the signed distance): the capacity of balls and the spherical contact
  distribution for all radii at once;
* *lines*: the crossings of the interface along lines, located by sphere tracing and
  bisection: the chords of the phases (their lengths within the lines) and, by the formula
  of Crofton, the specific surface ``S_V = 2 P_L``, ``P_L`` the crossings per unit length of
  isotropic lines.

The points are uniform in the box (scrambled Sobol points, or pseudo-random ones); in a
periodic box the probes wrap around its faces, in another the pairs and segments lie in
the box (minus sampling) and the lines end on its faces (the chords they cut are left out).
The errors are the standard deviations of the estimators over the probes of one
realization, as for independent probes; they do not include the fluctuation from one
realization to another.

Unions of spheres have their four Minkowski functionals exactly
(:mod:`morphogen.morphology.balls`); :func:`compare_with_theory` takes them, with the
probes above, beside the closed forms of the model of a realization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

from morphogen.models.implicit import functions_at

# ----------------------------------------------------------------------------------
# Results


@dataclass
class Estimate:
    """A measured value and the standard deviation of its estimator over the probes."""

    value: float
    error: float
    probes: int

    def __float__(self) -> float:
        return float(self.value)


@dataclass
class Curve:
    """A measured function of a size (a lag, a radius, a length): its values and their
    standard deviations over the probes."""

    name: str
    sizes: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    probes: int
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------
# Probes


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def sample_points(domain, n: int, *, seed=None, method: str = "sobol", margin=None) -> np.ndarray:
    """``n`` points uniform in the box of a domain (``margin``: in the box shrunk by these
    lengths from its upper faces, for the pairs and segments of minus sampling): scrambled
    Sobol points (``method="sobol"``, a low-discrepancy sequence) or pseudo-random ones
    (``"random"``)."""
    n = int(n)
    if n < 1:
        raise ValueError("n: at least one point")
    lower = np.asarray(domain.lower, dtype=float)
    lengths = np.asarray(domain.lengths, dtype=float)
    if margin is not None:
        lengths = lengths - np.asarray(margin, dtype=float)
        if np.any(lengths <= 0.0):
            raise ValueError("the probes do not fit in the box")
    rng = _rng(seed)
    if method == "sobol":  # the first n points of a power of two
        unit = qmc.Sobol(3, scramble=True, seed=rng).random_base2(math.ceil(math.log2(n)))[:n]
    elif method == "random":
        unit = rng.random((n, 3))
    else:
        raise ValueError('method: "sobol" or "random"')
    return lower + unit * lengths


def isotropic_directions(n: int, *, seed=None) -> np.ndarray:
    """``n`` unit vectors uniform on the sphere."""
    v = _rng(seed).standard_normal((int(n), 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def _direction(direction) -> np.ndarray:
    d = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise ValueError("direction: a nonzero vector")
    return d / norm


def _wrap(micro, points):
    domain = micro.domain
    return domain.wrap(points) if domain.periodic else points


def _default_reach(micro) -> float:
    """The clipping of the functions for the probes: a few sizes of the objects of a union of
    spheres and segments, else a twentieth of the box."""
    sizes = []
    for name in ("spheres", "segments"):
        prims = getattr(micro, name, None)
        if prims is not None and len(prims):
            sizes.append(float(np.median(prims.radii)))
    if sizes:
        return 2.0 * max(sizes)
    return 0.05 * float(np.min(micro.domain.lengths))


def phase_function(micro, phase: int, points, reach: float | None = None) -> np.ndarray:
    """The implicit function of a phase at points (negative inside), clipped to ``reach``."""
    reach = _default_reach(micro) if reach is None else float(reach)
    values = functions_at(micro, _wrap(micro, np.asarray(points, dtype=float)), reach)
    if int(phase) not in values:
        raise ValueError(f"phase {phase}: the phases are {sorted(values)}")
    return np.clip(np.asarray(values[int(phase)], dtype=float), -reach, reach)


def phases_at(micro, points) -> np.ndarray:
    """The phase of each point (the phase whose function is the smallest, negative)."""
    reach = _default_reach(micro)
    values = functions_at(micro, _wrap(micro, np.asarray(points, dtype=float)), reach)
    keys = sorted(values)
    stack = np.stack([np.asarray(values[k], dtype=float) for k in keys])
    return np.asarray(keys)[np.argmin(stack, axis=0)]


def _inside(micro, phase, points) -> np.ndarray:
    """Whether points lie in ``phase``: its function is negative there (the functions of the
    phases partition space)."""
    return phase_function(micro, phase, points) < 0.0


def _exact_distance(micro) -> bool:
    """Whether the function of phase 1 is the distance to it outside: the unions of spheres
    and segments of one phase."""
    return (
        hasattr(micro, "spheres")
        and hasattr(micro, "segments")
        and callable(getattr(micro, "signed_distance", None))
        and micro.implicit_phases() == [0, 1]
    )


def _lipschitz(micro) -> float:
    return float(getattr(micro, "lipschitz", 1.0) or 1.0)


def first_hits(micro, starts, directions, length: float, *, phase: int = 1, tol=None):
    """The distance along each ray (``starts``, unit ``directions``) to the first point of
    ``phase`` within ``length`` (0 when the start is in it, inf when none): sphere tracing
    by steps ``|φ| / L``, a hit where the function falls below ``tol`` (default ``1e-7`` of
    the box)."""
    starts = np.asarray(starts, dtype=float).reshape(-1, 3)
    dirs = np.asarray(directions, dtype=float).reshape(-1, 3)
    if len(dirs) == 1:
        dirs = np.repeat(dirs, len(starts), axis=0)
    tol = 1e-7 * float(np.max(micro.domain.lengths)) if tol is None else float(tol)
    lip = _lipschitz(micro)
    reach = max(_default_reach(micro), 4.0 * tol)
    t = np.zeros(len(starts))
    hit = np.full(len(starts), np.inf)
    active = np.ones(len(starts), dtype=bool)
    while active.any():
        idx = np.flatnonzero(active)
        f = phase_function(micro, phase, starts[idx] + t[idx, None] * dirs[idx], reach)
        landed = f <= tol
        hit[idx[landed]] = t[idx[landed]]
        step = np.maximum(f / lip, tol)
        t[idx] += step
        over = t[idx] > length
        active[idx[landed | over]] = False
    return hit


def crossings(micro, starts, directions, lengths, *, phase: int = 1, tol=None):
    """The crossings of the boundary of ``phase`` along segments (``starts``, unit
    ``directions``, ``lengths``): for each segment, whether its start is in the phase and the
    sorted distances of its crossings, located by sphere tracing and then by bisection to
    ``tol`` (default ``1e-9`` of the box)."""
    starts = np.asarray(starts, dtype=float).reshape(-1, 3)
    dirs = np.asarray(directions, dtype=float).reshape(-1, 3)
    lengths = np.broadcast_to(np.asarray(lengths, dtype=float), (len(starts),)).copy()
    tol = 1e-9 * float(np.max(micro.domain.lengths)) if tol is None else float(tol)
    lip = _lipschitz(micro)
    reach = _default_reach(micro)
    floor = max(tol, 1e-6 * reach)  # the smallest step of the march
    t = np.zeros(len(starts))
    f = phase_function(micro, phase, starts, reach)
    first_inside = f < 0.0
    found: list[list[float]] = [[] for _ in range(len(starts))]
    active = lengths > 0.0
    while active.any():
        idx = np.flatnonzero(active)
        step = np.maximum(np.abs(f[idx]) / lip, floor)
        t_new = np.minimum(t[idx] + step, lengths[idx])
        f_new = phase_function(micro, phase, starts[idx] + t_new[:, None] * dirs[idx], reach)
        changed = (f_new < 0.0) != (f[idx] < 0.0)
        if changed.any():
            sel = idx[changed]
            lo, hi = t[sel].copy(), t_new[changed].copy()
            f_lo = f[sel].copy()
            while np.any(hi - lo > tol):
                mid = 0.5 * (lo + hi)
                f_mid = phase_function(micro, phase, starts[sel] + mid[:, None] * dirs[sel], reach)
                same = (f_mid < 0.0) == (f_lo < 0.0)
                lo = np.where(same, mid, lo)
                f_lo = np.where(same, f_mid, f_lo)
                hi = np.where(same, hi, mid)
            for k, where in zip(sel, 0.5 * (lo + hi), strict=True):
                found[k].append(float(where))
        t[idx] = t_new
        f[idx] = f_new
        active[idx[t_new >= lengths[idx]]] = False
    return first_inside, found


# ----------------------------------------------------------------------------------
# Measures


def volume_fraction(micro, phase: int = 1, *, points: int = 2**16, seed=0) -> Estimate:
    """The fraction of ``phase`` at uniform points of the box."""
    x = sample_points(micro.domain, points, seed=seed)
    p = float(np.mean(_inside(micro, phase, x)))
    return Estimate(p, math.sqrt(p * (1.0 - p) / len(x)), len(x))


def _pair_starts(micro, lag_max, direction, n, rng):
    """The first points of pairs (or of segments) of extent ``lag_max`` and their directions
    (one random direction each if ``direction`` is None)."""
    domain = micro.domain
    if direction is None:
        dirs = isotropic_directions(n, seed=rng)
    else:
        dirs = np.repeat(_direction(direction)[None, :], n, axis=0)
    if domain.periodic:
        return sample_points(domain, n, seed=rng, method="random"), dirs
    # minus sampling: the whole probe in the box
    reach = np.abs(dirs) * lag_max
    lengths = np.asarray(domain.lengths, dtype=float)
    if np.any(lengths <= reach.max(axis=0)):
        raise ValueError("the probes do not fit in the box: shorter lags or sizes")
    u = rng.random((n, 3))
    lower = np.asarray(domain.lower, dtype=float) + np.where(dirs < 0.0, reach, 0.0)
    return lower + u * (lengths - reach), dirs


def covariance(micro, lags, *, phase: int = 1, direction=None, pairs: int = 2**15, seed=0) -> Curve:
    """``C(h) = P(x ∈ A, x + h u ∈ A)`` of ``phase`` (``A``) at the ``lags`` along the
    ``direction`` ``u`` (None: a random direction for each pair, the covariance averaged
    over the directions), from pairs of points."""
    lags = np.atleast_1d(np.asarray(lags, dtype=float))
    rng = _rng(seed)
    x, u = _pair_starts(micro, float(lags.max()), direction, int(pairs), rng)
    first = _inside(micro, phase, x)
    values, errors = [], []
    for h in lags:
        both = first & _inside(micro, phase, x + h * u)
        c = float(np.mean(both))
        values.append(c)
        errors.append(math.sqrt(c * (1.0 - c) / len(x)))
    return Curve("covariance", lags, np.array(values), np.array(errors), len(x))


def capacity(
    micro, element: str, sizes, *, phase: int = 1, direction=None, points: int = 2**15, seed=0
) -> Curve:
    """The Choquet capacity ``T(K) = P(K_x ∩ A ≠ ∅)`` of ``phase`` for structuring elements
    of growing ``sizes``: ``"ball"`` (radius; the unions of spheres and segments, whose
    function is the distance outside them), ``"segment"`` (length, along ``direction``, or a
    random direction per probe), ``"pair"`` (two points at the distance, ``2 p - C(h)``)."""
    sizes = np.atleast_1d(np.asarray(sizes, dtype=float))
    if np.any(sizes < 0.0):
        raise ValueError("sizes: non-negative sizes")
    rng = _rng(seed)
    n = int(points)
    if element == "ball":
        if not _exact_distance(micro) or int(phase) != 1:
            raise ValueError(
                "balls: the phase 1 of a union of spheres and segments (an exact distance); "
                "segments and pairs probe any structure"
            )
        x = sample_points(micro.domain, n, seed=rng)
        d = phase_function(micro, 1, x, float(sizes.max()) + 1e-9)
        values = np.array([np.mean(d <= r) for r in sizes])
        info = {"distances": d}
    elif element == "segment":
        x, u = _pair_starts(micro, float(sizes.max()), direction, n, rng)
        hit = first_hits(micro, x, u, float(sizes.max()), phase=phase)
        values = np.array([np.mean(hit <= s) for s in sizes])
        info = {"first_hits": hit}
    elif element == "pair":
        c = covariance(micro, sizes, phase=phase, direction=direction, pairs=n, seed=rng)
        p = volume_fraction(micro, phase, points=n, seed=rng).value
        values = 2.0 * p - c.values
        info = {}
    else:
        raise ValueError('element: "ball", "segment" or "pair"')
    errors = np.sqrt(np.clip(values * (1.0 - values), 0.0, None) / n)
    return Curve(f"capacity_{element}", sizes, values, errors, n, info)


def contact_distribution(
    micro, sizes, kind: str = "spherical", *, direction=None, points: int = 2**15, seed=0
) -> Curve:
    """The contact distribution of the phase 1 from its complement: the distribution of the
    distance from a point outside the phase to it, ``H(r) = 1 - Q(K_r) / q``, through balls
    (``"spherical"``) or along a direction (``"linear"``: segments)."""
    element = {"spherical": "ball", "linear": "segment"}.get(kind)
    if element is None:
        raise ValueError('kind: "spherical" or "linear"')
    t = capacity(micro, element, sizes, direction=direction, points=points, seed=seed)
    key = "distances" if element == "ball" else "first_hits"
    d = t.info[key]
    outside = d > 0.0
    q = int(outside.sum())
    if q == 0:
        values = np.full(len(t.sizes), np.nan)
    else:
        values = np.array([np.mean(d[outside] <= s) for s in t.sizes])
    errors = np.sqrt(np.clip(values * (1.0 - values), 0.0, None) / max(q, 1))
    return Curve(f"contact_{kind}", t.sizes, values, errors, q)


def _lines(micro, n, length, direction, rng):
    """Lines of the measures along lines: in a periodic box, of ``length`` from uniform
    points (around the torus); in another, through uniform points and across the box."""
    domain = micro.domain
    if direction is None:
        dirs = isotropic_directions(n, seed=rng)
    else:
        dirs = np.repeat(_direction(direction)[None, :], n, axis=0)
    x = sample_points(domain, n, seed=rng, method="random")
    if domain.periodic:
        span = float(np.max(domain.lengths)) if length is None else float(length)
        return x, dirs, np.full(n, span)
    lo, hi = domain.lower, domain.upper
    with np.errstate(divide="ignore", invalid="ignore"):
        back = np.where(dirs > 0, (x - lo) / dirs, np.where(dirs < 0, (x - hi) / dirs, np.inf))
        ahead = np.where(dirs > 0, (hi - x) / dirs, np.where(dirs < 0, (lo - x) / dirs, np.inf))
    t0 = np.min(back, axis=1)
    t1 = np.min(ahead, axis=1)
    return x - t0[:, None] * dirs, dirs, t0 + t1


@dataclass
class Chords:
    """The chords of a phase along lines: their ``lengths`` and ``weights``, ``ℓ / (ℓ - c)``
    for a chord of length ``c`` on a line of length ``ℓ`` (the chords cut by the ends of the
    lines are left out, which a chord escapes with the probability ``(ℓ - c) / ℓ``: the
    weights correct it, Miles 1978)."""

    lengths: np.ndarray
    weights: np.ndarray

    @property
    def mean(self) -> float:
        """The number mean of the chord lengths."""
        if not len(self.lengths):
            return float("nan")
        return float(np.average(self.lengths, weights=self.weights))

    def distribution(self, sizes) -> np.ndarray:
        """``P(c <= l)`` of the chord lengths at the ``sizes``."""
        sizes = np.atleast_1d(np.asarray(sizes, dtype=float))
        total = float(self.weights.sum())
        return np.array([self.weights[self.lengths <= s].sum() / total for s in sizes])


def chords(micro, *, phase: int = 1, direction=None, lines: int = 512, length=None, seed=0):
    """The chords of ``phase`` along lines (isotropic, or along ``direction``): the intervals
    of the lines in the phase between two crossings of its boundary (:class:`Chords`). In a
    periodic box the lines run around the torus over ``length`` (default: twice its longest
    side), in another they cross the box."""
    rng = _rng(seed)
    if length is None and micro.domain.periodic:
        length = 2.0 * float(np.max(micro.domain.lengths))
    x, u, ell = _lines(micro, int(lines), length, direction, rng)
    inside, found = crossings(micro, x, u, ell, phase=phase)
    out, weights = [], []
    for start_in, ts, span in zip(inside, found, ell, strict=True):
        # the crossings alternate: entries and exits of the phase
        ts = list(ts)
        if start_in:
            ts = ts[1:]  # the first exit closes a chord cut by the start
        for a, b in zip(ts[0::2], ts[1::2], strict=False):
            out.append(b - a)
            weights.append(span / max(span - (b - a), 1e-300))
    return Chords(np.asarray(out, dtype=float), np.asarray(weights, dtype=float))


def specific_surface(micro, *, phase: int = 1, lines: int = 512, length=None, seed=0) -> Estimate:
    """``S_V = 2 P_L`` (Crofton): twice the crossings of the boundary of ``phase`` per unit
    length of isotropic lines."""
    rng = _rng(seed)
    x, u, ell = _lines(micro, int(lines), length, None, rng)
    _, found = crossings(micro, x, u, ell, phase=phase)
    counts = np.array([len(f) for f in found], dtype=float)
    total = float(ell.sum())
    value = 2.0 * counts.sum() / total
    # the ratio estimator over independent lines
    mean_len = total / len(ell)
    resid = counts - value / 2.0 * ell
    error = 2.0 * float(np.std(resid, ddof=1)) / (mean_len * math.sqrt(len(ell)))
    return Estimate(value, error, len(ell))


# ----------------------------------------------------------------------------------
# Morphological operations on the implicit geometry


def dilation(micro, radius: float):
    """The dilation of the phase 1 of a union of spheres and segments by the ball of
    ``radius``: the union of the same objects, ``radius`` thicker (exactly ``A ⊕ B_r``)."""
    from morphogen.models.union import PrimitiveUnion
    from morphogen.primitives import Segments, Spheres

    r = float(radius)
    if r < 0.0:
        raise ValueError("radius: non-negative")
    if not _exact_distance(micro):
        raise ValueError("dilation: a union of spheres and segments (morphogen.models.union)")
    s, g = micro.spheres, micro.segments
    spheres = Spheres(s.centers.copy(), s.radii + r)
    segments = Segments(g.start.copy(), g.end.copy(), g.radii + r, g.round_caps)
    return PrimitiveUnion(
        micro.domain,
        spheres=spheres,
        segments=segments,
        model=None,
        seed=micro.seed,
        metadata={**micro.metadata, "dilation": r},
    )


# ----------------------------------------------------------------------------------
# Measures beside the theory of the model


@dataclass
class Comparison:
    """Measures of a realization on its implicit geometry beside the closed forms of its
    model: rows ``(name, measured, error, theory)``; the curves (covariance, capacities)
    with their theory in ``curves``."""

    rows: list[tuple[str, float, float, float]]
    curves: dict[str, tuple[Curve, np.ndarray]] = field(default_factory=dict)
    exact: bool = False

    def table(self) -> str:
        """The rows as a plain text table."""
        lines = [f"{'measure':28s} {'measured':>12s} {'error':>10s} {'theory':>12s}"]
        for name, value, err, ref in self.rows:
            lines.append(f"{name:28s} {value:12.6g} {err:10.3g} {ref:12.6g}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {"measured": value, "error": err, "theory": ref}
            for name, value, err, ref in self.rows
        }


def compare_with_theory(
    micro,
    model=None,
    *,
    lags=None,
    sizes=None,
    points: int = 2**15,
    lines: int = 256,
    seed=0,
) -> Comparison:
    """The measures of a realization on its implicit geometry beside the closed forms of its
    model (``model``, by default that of the realization): the four Minkowski densities
    exactly for a union of spheres (:func:`~morphogen.morphology.balls.minkowski_functionals`,
    on the torus of a periodic box), else the fraction and the specific surface by probes;
    the covariance at ``lags`` and the capacities of balls and segments of ``sizes`` where
    the model has their closed forms."""
    from morphogen.morphology.balls import minkowski_functionals

    model = micro.model if model is None else model
    if model is None:
        raise ValueError("compare_with_theory: the realization has no model")
    theory = model.theory(micro.domain)
    rows = []
    exact = False
    spheres = getattr(micro, "spheres", None)
    segments = getattr(micro, "segments", None)
    only_spheres = (
        spheres is not None
        and len(spheres) > 0
        and (segments is None or not len(segments))
        and micro.domain.periodic
        and micro.implicit_phases() == [0, 1]
    )
    if only_spheres:
        exact = True
        for name, value in minkowski_functionals(micro).densities().items():
            if name in theory:
                rows.append((name, float(value), 0.0, float(theory[name])))
    else:
        if "volume_fraction" in theory:
            vf = volume_fraction(micro, points=points, seed=seed)
            rows.append(("volume_fraction", vf.value, vf.error, float(theory["volume_fraction"])))
        if "specific_surface" in theory:
            sv = specific_surface(micro, lines=lines, seed=seed)
            rows.append(("specific_surface", sv.value, sv.error, float(theory["specific_surface"])))
    curves = {}
    scale = 0.25 * float(np.min(micro.domain.lengths))
    lags = np.linspace(0.0, scale, 9) if lags is None else np.asarray(lags, dtype=float)
    sizes = np.linspace(0.0, 0.5 * scale, 6) if sizes is None else np.asarray(sizes, dtype=float)
    if callable(getattr(model, "covariance", None)):
        c = covariance(micro, lags, pairs=points, seed=seed)
        curves["covariance"] = (c, np.asarray(model.covariance(lags), dtype=float))
    if callable(getattr(model, "choquet_capacity", None)):
        for element in ("ball", "segment"):
            if element == "ball" and not _exact_distance(micro):
                continue
            t = capacity(micro, element, sizes, points=points, seed=seed)
            curves[f"capacity_{element}"] = (
                t,
                np.asarray(model.choquet_capacity(element, sizes), dtype=float),
            )
    return Comparison(rows, curves, exact)


# ----------------------------------------------------------------------------------
# Named measures (configurations, the command line, the interface)


def _exact_fraction(micro, seed) -> dict[str, float]:
    e = volume_fraction(micro, 1, seed=seed)
    return {"exact_fraction": float(e.value), "exact_fraction_std": float(e.error)}


def _exact_surface(micro, seed) -> dict[str, float]:
    e = specific_surface(micro, phase=1, seed=seed)
    return {"exact_specific_surface": float(e.value), "exact_specific_surface_std": float(e.error)}


def _exact_chords(micro, seed) -> dict[str, float]:
    return {f"exact_mean_chord_{p}": chords(micro, phase=p, seed=seed).mean for p in (0, 1)}


def _exact_minkowski(micro, seed) -> dict[str, float]:
    from morphogen.morphology.balls import minkowski_functionals

    d = minkowski_functionals(micro).densities()
    return {
        "exact_V_V": float(d["volume_fraction"]),
        "exact_S_V": float(d["specific_surface"]),
        "exact_M_V": float(d["mean_curvature_density"]),
        "exact_chi_V": float(d["euler_density"]),
    }


def _exact_grains(micro, seed) -> dict[str, float]:
    """The grains of a tessellation from its exact cells: their number, the mean and the
    coefficient of variation of their equivalent diameters, the medians of the ratios a/b
    and b/c of their ellipsoids (over the grains that the faces of the box do not cut), the
    fractions of the volume of the phases; for the grains of seeds, the gaps to their seeds
    (the relative RMS error of the diameters, the RMS errors of the ratios, the mean angle
    between their axes, the largest Kolmogorov–Smirnov distance to the laws of the
    seeds)."""
    from morphogen.models.seeded import SeededTessellation
    from morphogen.models.tessellation import Tessellation

    if not isinstance(micro, Tessellation):
        raise ValueError("exact_grains: the grains of a tessellation (voronoi, laguerre, "
                         "seeded_laguerre)")  # fmt: skip
    shapes = micro.grain_shapes()
    use = shapes["inside"].copy()
    if isinstance(micro, SeededTessellation):
        use &= ~micro.amorphous
    d = shapes["diameter"][use]
    semi = shapes["semi_axes"][use]
    out = {"grains": float(use.sum())}
    if d.size:
        out["grain_diameter_mean"] = float(d.mean())
        out["grain_diameter_cv"] = float(d.std() / d.mean())
        out["grain_aspect_ratio_median"] = float(np.median(semi[:, 0] / semi[:, 1]))
        out["grain_flatness_median"] = float(np.median(semi[:, 1] / semi[:, 2]))
    phases = np.asarray(getattr(micro, "grain_phases", micro.cell_phases), dtype=np.int64)
    volume = np.bincount(phases, shapes["volume"])
    for k in np.flatnonzero(volume):
        out[f"grain_fraction_{k}"] = float(volume[k] / volume.sum())
    if isinstance(micro, SeededTessellation):
        cmp = micro.seed_comparison(shapes)
        errors = cmp["errors"]
        fit = cmp["inside"]
        rel = (cmp["grain_diameter"] - cmp["seed_diameter"]) / cmp["seed_diameter"]
        if fit.any():
            out["seed_diameter_error"] = float(np.sqrt(np.mean(rel[fit] ** 2)))
        out["seed_aspect_ratio_error"] = errors["aspect_ratio"]["rmse"]
        out["seed_flatness_error"] = errors["flatness"]["rmse"]
        out["seed_axis_angle"] = errors["axis_angle"]["mean"]
        for name in ("diameter", "aspect_ratio", "flatness"):
            ks = [v for v in errors[name]["ks"].values() if np.isfinite(v)]
            if ks:
                out[f"seed_{name}_ks"] = float(max(ks))
    return {k: v for k, v in out.items() if np.isfinite(v)}


#: The named measures of the implicit geometry of a realization, each a function of the
#: realization and a seed to its columns: the fraction of the phase 1 by points and its
#: specific surface by lines (Crofton), with the standard deviations of their estimators,
#: the mean chords of the two phases, the four Minkowski densities of a union of spheres
#: exactly (:mod:`morphogen.morphology.balls`), and the grains of a tessellation from its
#: exact cells, against their seeds for the grains of seeds
EXACT_MEASURES = {
    "exact_fraction": _exact_fraction,
    "exact_surface": _exact_surface,
    "exact_chords": _exact_chords,
    "exact_minkowski": _exact_minkowski,
    "exact_grains": _exact_grains,
}


def measure_exact(micro, names, *, seed=0) -> dict[str, float]:
    """The named measures ``names`` (:data:`EXACT_MEASURES`) of a realization, on its
    implicit geometry, merged into one dictionary of columns."""
    out: dict[str, float] = {}
    for name in names:
        if name not in EXACT_MEASURES:
            raise ValueError(f"exact measure {name!r}: one of {sorted(EXACT_MEASURES)}")
        out.update(EXACT_MEASURES[name](micro, seed))
    return out
