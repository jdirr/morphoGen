"""Multiaxial high-cycle fatigue: criteria on the stress histories of a cycle.

In the high-cycle regime a material stays elastic at the scale of the cell, and whether
a point lasts is read from the cycle of its stress: an amplitude of shear, which
initiates the cracks, and a normal or hydrostatic stress, which opens them. The criteria
below judge the stresses of the points of a cell along a cycle of a loading path (its
fields at each step: :func:`path_fatigue`), or any history of stresses (``steps x ... x
6``, Voigt components 11, 22, 33, 23, 31, 12).

*The amplitude of a multiaxial cycle.* The deviator of the stress describes a closed
path in the five-dimensional space of deviators, with the norm ``sqrt(J2) = sqrt(s : s /
2)``. Its amplitude is the radius of the smallest hypersphere that holds the path, and
its mean the centre of that hypersphere (Papadopoulos et al. 1997; the longest chord, which
does not see the phase of the components, is not used): half the range of a
proportional cycle, the larger of ``σ_a / sqrt(3)`` and ``τ_a`` for a tension and a
torsion a quarter of a cycle apart. The hypersphere is found by Frank and Wolfe's
iteration with away steps on the dual problem (Yıldırım 2008), from the two points
farthest apart: at once for a proportional cycle.

*Criteria* (``f``: the fatigue limit in fully reversed tension or bending, ``t``: in
fully reversed torsion, ``f0``: in repeated tension, the largest stress of a cycle from
zero; each criterion gives an equivalent stress, to be compared with its limit):

* :class:`Crossland` (1956): ``sqrt(J2)_a + α σ_H,max <= β``, ``σ_H`` the hydrostatic
  stress; ``α = 3 t / f - sqrt(3)``, ``β = t``.
* :class:`Sines` (1959): ``sqrt(J2)_a + α σ_H,mean <= β``; ``α = 6 t / f0 - sqrt(3)``,
  ``β = t``.
* :class:`DangVan` (1973; Dang Van, Griveau & Message 1989): at each instant, Tresca's
  shear of the deviator less its mean (the stress of a grain that has shaken down in
  its elastic matrix) and the hydrostatic stress, ``max_t [τ(t) + α σ_H(t)] <= β``;
  ``α = 3 t / f - 3 / 2``, ``β = t``.
* :class:`Findley` (1959), a critical plane: ``max_n [τ_a(n) + k σ_n,max(n)] <= λ`` over
  the planes of normal ``n``, ``τ_a`` the amplitude of the shear vector of the plane
  (the radius of the smallest circle that holds its path) and ``σ_n,max`` its largest
  normal stress; with ``r = f / t``, ``k = (2 - r) / (2 sqrt(r - 1))`` and ``λ = t
  sqrt(1 + k²)``.
* :class:`Matake` (1977): ``τ_a + k σ_n,max <= λ`` on the plane of the largest shear
  amplitude; ``k = 2 t / f - 1``, ``λ = t``.

The planes are searched on a spiral of normals over a hemisphere, then around the best
of them by grids three times finer at each of four levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

#: The criteria by their names in configurations
CRITERIA = ("crossland", "sines", "dang_van", "findley", "matake")
#: Normals of the first search of a critical plane (a spiral over a hemisphere)
PLANES = 400
_SQRT2 = math.sqrt(2.0)


# -- the smallest ball of a path ------------------------------------------------------------


@njit(cache=True)
def _ball(points, tol, weights, centre):
    """The smallest ball that holds ``points`` (rows): its radius, to the relative
    accuracy ``tol``, and its ``centre`` (written); ``weights``: a work array, one value
    per point."""
    n, d = points.shape
    first, best = 0, -1.0
    for i in range(n):
        dist = 0.0
        for c in range(d):
            dist += (points[i, c] - points[0, c]) ** 2
        if dist > best:
            best, first = dist, i
    second, best = first, -1.0
    for i in range(n):
        dist = 0.0
        for c in range(d):
            dist += (points[i, c] - points[first, c]) ** 2
        if dist > best:
            best, second = dist, i
    for i in range(n):
        weights[i] = 0.0
    weights[first] += 0.5
    weights[second] += 0.5
    for c in range(d):
        centre[c] = 0.5 * (points[first, c] + points[second, c])
    if best <= 0.0:
        return 0.0
    bound = (1.0 + tol) ** 2 - 1.0
    far_dist = 0.25 * best
    for _ in range(100 * n + 1000):
        gamma, far, near = 0.0, 0, -1
        far_dist, near_dist = -1.0, math.inf
        for i in range(n):
            dist = 0.0
            for c in range(d):
                dist += (points[i, c] - centre[c]) ** 2
            if dist > far_dist:
                far_dist, far = dist, i
            if weights[i] > 0.0:
                gamma += weights[i] * dist
                if dist < near_dist:
                    near_dist, near = dist, i
        if gamma <= 0.0:
            break
        plus = far_dist / gamma - 1.0
        minus = 1.0 - near_dist / gamma
        if plus <= bound and minus <= bound:
            break
        if plus > minus:  # towards the farthest point
            step = plus / (2.0 * (1.0 + plus))
            for i in range(n):
                weights[i] *= 1.0 - step
            weights[far] += step
            for c in range(d):
                centre[c] = (1.0 - step) * centre[c] + step * points[far, c]
        else:  # away from the nearest point of the support
            limit = weights[near] / (1.0 - weights[near])
            step = limit if near_dist <= 0.0 else min(limit, minus / (2.0 * (1.0 - minus)))
            for i in range(n):
                weights[i] *= 1.0 + step
            weights[near] = 0.0 if step >= limit else weights[near] - step
            for c in range(d):
                centre[c] = (1.0 + step) * centre[c] - step * points[near, c]
    far_dist = 0.0
    for i in range(n):
        dist = 0.0
        for c in range(d):
            dist += (points[i, c] - centre[c]) ** 2
        if dist > far_dist:
            far_dist = dist
    return math.sqrt(far_dist)


@njit(parallel=True, cache=True)
def _amplitudes(vectors, tol, centres, radii):
    """The smallest balls of the paths ``vectors`` (steps x points x d) of each point."""
    steps, n, d = vectors.shape
    for p in prange(n):
        path = np.empty((steps, d))
        for t in range(steps):
            for c in range(d):
                path[t, c] = vectors[t, p, c]
        centre = np.empty(d)
        radii[p] = _ball(path, tol, np.empty(steps), centre)
        for c in range(d):
            centres[p, c] = centre[c]


@njit(cache=True)
def _tresca(s):
    """Tresca's shear (half the difference of the extreme principal values) of a
    deviator in Voigt components."""
    j2 = 0.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) + s[3] * s[3] + s[4] * s[4] + s[5] * s[5]
    if j2 <= 0.0:
        return 0.0
    j3 = (
        s[0] * s[1] * s[2]
        + 2.0 * s[3] * s[4] * s[5]
        - s[0] * s[3] * s[3]
        - s[1] * s[4] * s[4]
        - s[2] * s[5] * s[5]
    )
    cos3 = 1.5 * math.sqrt(3.0) * j3 / j2**1.5
    cos3 = min(1.0, max(-1.0, cos3))
    theta = math.acos(cos3) / 3.0  # Lode's angle, in [0, π/3]
    return math.sqrt(j2) * math.sin(theta + math.pi / 3.0)


@njit(parallel=True, cache=True)
def _dang_van(deviators, hydro, means, alpha, out):
    """``max_t [τ(t) + α σ_H(t)]`` of each point, ``τ`` Tresca's shear of its deviator
    less its mean."""
    steps, n, _ = deviators.shape
    for p in prange(n):
        s = np.empty(6)
        best = -math.inf
        for t in range(steps):
            for c in range(6):
                s[c] = deviators[t, p, c] - means[p, c]
            value = _tresca(s) + alpha * hydro[t, p]
            if value > best:
                best = value
        out[p] = best


# -- critical planes ---------------------------------------------------------------------------


def _spiral(count: int) -> np.ndarray:
    """Normals spread over a hemisphere (a Fibonacci spiral)."""
    k = np.arange(count) + 0.5
    z = 1.0 - k / count
    angle = math.pi * (1.0 + math.sqrt(5.0)) * k
    r = np.sqrt(1.0 - z * z)
    return np.ascontiguousarray(np.stack([r * np.cos(angle), r * np.sin(angle), z], axis=1))


@njit(cache=True)
def _plane(stress, normal, tol, shear, weights, centre):
    """The amplitude of the shear vector and the largest normal stress of the plane of
    ``normal`` along the stresses ``stress`` (steps x 6); ``shear`` (steps x 3),
    ``weights`` and ``centre``: work arrays."""
    steps = stress.shape[0]
    nx, ny, nz = normal[0], normal[1], normal[2]
    largest = -math.inf
    for t in range(steps):
        s = stress[t]
        tx = s[0] * nx + s[5] * ny + s[4] * nz
        ty = s[5] * nx + s[1] * ny + s[3] * nz
        tz = s[4] * nx + s[3] * ny + s[2] * nz
        sn = tx * nx + ty * ny + tz * nz
        shear[t, 0] = tx - sn * nx
        shear[t, 1] = ty - sn * ny
        shear[t, 2] = tz - sn * nz
        if sn > largest:
            largest = sn
    return _ball(shear, tol, weights, centre), largest


@njit(parallel=True, cache=True)
def _critical_planes(stress, normals, k, matake, tol, values, planes, shears, pressures):
    """The critical plane of each point (``stress``: steps x points x 6): the value of
    its criterion, its normal, its shear amplitude and its largest normal stress.
    ``matake``: the plane of the largest shear amplitude (among the planes within a
    thousandth of it, that of the largest value); otherwise that of the largest value
    ``τ_a + k σ_n,max``."""
    steps, n, _ = stress.shape
    m = normals.shape[0]
    spacing = math.sqrt(2.0 * math.pi / m)
    starts = 6
    for p in prange(n):
        history = np.empty((steps, 6))
        for t in range(steps):
            for c in range(6):
                history[t, c] = stress[t, p, c]
        shear = np.empty((steps, 3))
        weights = np.empty(steps)
        centre = np.empty(3)
        # the best normals of the spiral, for the objective of the search
        top = np.full(starts, -math.inf)
        index = np.zeros(starts, dtype=np.int64)
        for j in range(m):
            amplitude, largest = _plane(history, normals[j], tol, shear, weights, centre)
            objective = amplitude if matake else amplitude + k * largest
            worst = 0
            for q in range(1, starts):
                if top[q] < top[worst]:
                    worst = q
            if objective > top[worst]:
                top[worst], index[worst] = objective, j
        found = np.empty((starts, 5))  # objective, amplitude, largest, and the normal below
        found_normals = np.empty((starts, 3))
        trial = np.empty(3)
        e1 = np.empty(3)
        e2 = np.empty(3)
        for q in range(starts):
            best = normals[index[q]].copy()
            amplitude, largest = _plane(history, best, tol, shear, weights, centre)
            objective = amplitude if matake else amplitude + k * largest
            step = 0.5 * spacing
            for _ in range(4):
                # two tangents of the sphere at the best normal
                a = 0 if abs(best[0]) < 0.9 else 1
                e1[:] = 0.0
                e1[a] = 1.0
                dot = e1[0] * best[0] + e1[1] * best[1] + e1[2] * best[2]
                for c in range(3):
                    e1[c] -= dot * best[c]
                norm = math.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
                for c in range(3):
                    e1[c] /= norm
                e2[0] = best[1] * e1[2] - best[2] * e1[1]
                e2[1] = best[2] * e1[0] - best[0] * e1[2]
                e2[2] = best[0] * e1[1] - best[1] * e1[0]
                origin = best.copy()
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        if i == 0 and j == 0:
                            continue
                        for c in range(3):
                            trial[c] = origin[c] + step * (i * e1[c] + j * e2[c])
                        norm = math.sqrt(trial[0] ** 2 + trial[1] ** 2 + trial[2] ** 2)
                        for c in range(3):
                            trial[c] /= norm
                        amp, big = _plane(history, trial, tol, shear, weights, centre)
                        value = amp if matake else amp + k * big
                        if value > objective:
                            objective, amplitude, largest = value, amp, big
                            best[:] = trial
                step /= 3.0
            found[q, 0], found[q, 1], found[q, 2] = objective, amplitude, largest
            found_normals[q] = best
        choice, chosen = 0, -math.inf
        peak = found[:, 0].max()
        for q in range(starts):
            if matake and found[q, 0] < 0.999 * peak:
                continue
            value = found[q, 1] + k * found[q, 2]
            if value > chosen:
                chosen, choice = value, q
        values[p] = chosen
        shears[p] = found[choice, 1]
        pressures[p] = found[choice, 2]
        for c in range(3):
            planes[p, c] = found_normals[choice, c]


# -- measures of a history ---------------------------------------------------------------------


def _history(history) -> np.ndarray:
    stress = np.asarray(history, dtype=float)
    if stress.ndim < 2 or stress.shape[-1] != 6 or stress.shape[0] < 2:
        raise ValueError(
            "fatigue: the stresses of a cycle, steps x ... x 6 (Voigt components 11, 22, "
            "33, 23, 31, 12), at two steps at least"
        )
    if not np.all(np.isfinite(stress)):
        raise ValueError("fatigue: stresses that are not finite")
    return stress


def hydrostatic(history) -> np.ndarray:
    """The hydrostatic stress ``tr σ / 3`` of the stresses of a history (steps x ...)."""
    return _history(history)[..., :3].mean(axis=-1)


def stress_amplitude(history, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """The amplitude and the mean of the deviator of a cycle of stresses (``history``:
    steps x ... x 6): ``sqrt(J2)_a``, the radius of the smallest hypersphere that holds
    the path of the deviator (norm ``sqrt(s : s / 2)``; to the relative accuracy
    ``tol``), and its centre, the mean deviator (... x 6)."""
    stress = _history(history)
    shape = stress.shape[1:-1]
    flat = stress.reshape(stress.shape[0], -1, 6)
    deviator = flat.copy()
    deviator[..., :3] -= flat[..., :3].mean(axis=-1, keepdims=True)
    # components whose Euclidean norm is sqrt(J2)
    scale = np.array([1.0, 1.0, 1.0, _SQRT2, _SQRT2, _SQRT2]) / _SQRT2
    vectors = np.ascontiguousarray(deviator * scale)
    centres = np.zeros((flat.shape[1], 6))
    radii = np.zeros(flat.shape[1])
    _amplitudes(vectors, float(tol), centres, radii)
    return radii.reshape(shape), (centres / scale).reshape(*shape, 6)


# -- criteria -------------------------------------------------------------------------------------


def _positive(name: str, **values) -> None:
    for key, value in values.items():
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name}: a positive {key}, got {value!r}")


def _ratio(name: str, bending: float, torsion: float, low: float, high: float) -> float:
    """The ratio of the torsion limit to the bending limit, within the bounds for which
    the criterion has a meaning (a normal stress that lowers the limit)."""
    _positive(name, bending=bending, torsion=torsion)
    ratio = torsion / bending
    if not low <= ratio <= high:
        raise ValueError(
            f"{name}: a torsion limit between {low:.3g} and {high:.3g} times the bending "
            f"limit, got {ratio:.3g} times"
        )
    return ratio


@dataclass(frozen=True)
class Crossland:
    """Crossland's criterion (see the module): ``sqrt(J2)_a + alpha σ_H,max <= beta``."""

    alpha: float
    beta: float
    name = "crossland"

    def __post_init__(self) -> None:
        _positive("Crossland", beta=self.beta)
        if not (math.isfinite(self.alpha) and self.alpha >= 0):
            raise ValueError("Crossland: alpha of 0 or more")

    @classmethod
    def from_limits(cls, bending: float, torsion: float) -> Crossland:
        """From the fatigue limits in fully reversed bending (or tension) and torsion."""
        ratio = _ratio("Crossland", bending, torsion, 1.0 / math.sqrt(3.0), math.inf)
        return cls(3.0 * ratio - math.sqrt(3.0), float(torsion))

    @property
    def limit(self) -> float:
        return self.beta

    def equivalent(self, history) -> np.ndarray:
        """The equivalent stress of each point, to be compared with :attr:`limit`."""
        amplitude, _ = stress_amplitude(history)
        return amplitude + self.alpha * hydrostatic(history).max(axis=0)


@dataclass(frozen=True)
class Sines:
    """Sines' criterion (see the module): ``sqrt(J2)_a + alpha σ_H,mean <= beta``, the
    mean of the hydrostatic stress being that of its extremes over the cycle."""

    alpha: float
    beta: float
    name = "sines"

    def __post_init__(self) -> None:
        _positive("Sines", beta=self.beta)
        if not (math.isfinite(self.alpha) and self.alpha >= 0):
            raise ValueError("Sines: alpha of 0 or more")

    @classmethod
    def from_limits(cls, torsion: float, repeated: float) -> Sines:
        """From the fatigue limits in fully reversed torsion and in repeated tension (the
        largest stress of a cycle from zero)."""
        _positive("Sines", torsion=torsion, repeated=repeated)
        alpha = 6.0 * torsion / repeated - math.sqrt(3.0)
        if alpha < 0:
            raise ValueError(
                "Sines: a limit in repeated tension below 2 sqrt(3) times the torsion limit"
            )
        return cls(alpha, float(torsion))

    @property
    def limit(self) -> float:
        return self.beta

    def equivalent(self, history) -> np.ndarray:
        amplitude, _ = stress_amplitude(history)
        hydro = hydrostatic(history)
        return amplitude + self.alpha * 0.5 * (hydro.max(axis=0) + hydro.min(axis=0))


@dataclass(frozen=True)
class DangVan:
    """Dang Van's criterion (see the module): ``max_t [τ(t) + alpha σ_H(t)] <= beta``."""

    alpha: float
    beta: float
    name = "dang_van"

    def __post_init__(self) -> None:
        _positive("DangVan", beta=self.beta)
        if not (math.isfinite(self.alpha) and self.alpha >= 0):
            raise ValueError("DangVan: alpha of 0 or more")

    @classmethod
    def from_limits(cls, bending: float, torsion: float) -> DangVan:
        """From the fatigue limits in fully reversed bending (or tension) and torsion."""
        ratio = _ratio("DangVan", bending, torsion, 0.5, math.inf)
        return cls(3.0 * ratio - 1.5, float(torsion))

    @property
    def limit(self) -> float:
        return self.beta

    def equivalent(self, history) -> np.ndarray:
        stress = _history(history)
        shape = stress.shape[1:-1]
        _, mean = stress_amplitude(stress)
        flat = stress.reshape(stress.shape[0], -1, 6)
        hydro = flat[..., :3].mean(axis=-1)
        deviator = flat.copy()
        deviator[..., :3] -= hydro[..., None]
        out = np.zeros(flat.shape[1])
        _dang_van(np.ascontiguousarray(deviator), np.ascontiguousarray(hydro),
                  np.ascontiguousarray(mean.reshape(-1, 6)), float(self.alpha), out)  # fmt: skip
        return out.reshape(shape)


class _CriticalPlane:
    """The criteria of a critical plane: ``τ_a + k σ_n,max <= limit`` on a plane."""

    k: float
    limit: float
    planes: int
    _matake = False

    def critical_plane(self, history) -> dict:
        """The critical plane of each point: the ``equivalent`` stress of the criterion,
        the ``normal`` of the plane (... x 3), its ``shear`` amplitude and its largest
        ``normal_stress``."""
        stress = _history(history)
        shape = stress.shape[1:-1]
        flat = np.ascontiguousarray(stress.reshape(stress.shape[0], -1, 6))
        n = flat.shape[1]
        values, shears, pressures = np.zeros(n), np.zeros(n), np.zeros(n)
        normals = np.zeros((n, 3))
        _critical_planes(flat, _spiral(self.planes), float(self.k), self._matake, 1e-7,
                         values, normals, shears, pressures)  # fmt: skip
        return {
            "equivalent": values.reshape(shape),
            "normal": normals.reshape(*shape, 3),
            "shear": shears.reshape(shape),
            "normal_stress": pressures.reshape(shape),
        }

    def equivalent(self, history) -> np.ndarray:
        return self.critical_plane(history)["equivalent"]

    def _check(self, name: str) -> None:
        _positive(name, limit=self.limit)
        if not (math.isfinite(self.k) and self.k >= 0):
            raise ValueError(f"{name}: k of 0 or more")
        if self.planes < 20:
            raise ValueError(f"{name}: 20 planes at least for the first search")


@dataclass(frozen=True)
class Findley(_CriticalPlane):
    """Findley's criterion (see the module): ``max_n [τ_a(n) + k σ_n,max(n)] <= limit``;
    ``planes``: the normals of the first search."""

    k: float
    limit: float
    planes: int = PLANES
    name = "findley"

    def __post_init__(self) -> None:
        self._check("Findley")

    @classmethod
    def from_limits(cls, bending: float, torsion: float, planes: int = PLANES) -> Findley:
        """From the fatigue limits in fully reversed bending (or tension) and torsion."""
        ratio = 1.0 / _ratio("Findley", bending, torsion, 0.5, 1.0 - 1e-12)
        k = (2.0 - ratio) / (2.0 * math.sqrt(ratio - 1.0))
        return cls(k, float(torsion) * math.sqrt(1.0 + k * k), planes)


@dataclass(frozen=True)
class Matake(_CriticalPlane):
    """Matake's criterion (see the module): ``τ_a + k σ_n,max <= limit`` on the plane of
    the largest shear amplitude; ``planes``: the normals of the first search."""

    k: float
    limit: float
    planes: int = PLANES
    name = "matake"
    _matake = True

    def __post_init__(self) -> None:
        self._check("Matake")

    @classmethod
    def from_limits(cls, bending: float, torsion: float, planes: int = PLANES) -> Matake:
        """From the fatigue limits in fully reversed bending (or tension) and torsion."""
        ratio = _ratio("Matake", bending, torsion, 0.5, math.inf)
        return cls(2.0 * ratio - 1.0, float(torsion), planes)


def criterion_from_config(spec):
    """A criterion from a mapping: its ``criterion`` (one of :data:`CRITERIA`) and either
    its fatigue limits (``bending`` and ``torsion``; ``torsion`` and ``repeated`` for
    Sines) or its coefficients (``alpha`` and ``beta``; ``k`` and ``limit`` for the
    critical planes, with their ``planes``)."""
    if not isinstance(spec, dict):
        raise ValueError("fatigue: a mapping of a criterion and its limits")
    spec = dict(spec)
    name = spec.pop("criterion", None)
    classes = {"crossland": Crossland, "sines": Sines, "dang_van": DangVan,
               "findley": Findley, "matake": Matake}  # fmt: skip
    if name not in classes:
        raise ValueError(f"fatigue: the criterion is one of {list(CRITERIA)}, got {name!r}")
    cls = classes[name]
    plane = name in ("findley", "matake")
    coefficients = ("k", "limit") if plane else ("alpha", "beta")
    limits = ("torsion", "repeated") if name == "sines" else ("bending", "torsion")
    allowed = set(coefficients) | set(limits) | ({"planes"} if plane else set())
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(
            f"fatigue: unknown parameters {unknown} of the criterion {name!r}; use "
            f"{list(limits)} or {list(coefficients)}" + (" and 'planes'" if plane else "")
        )
    try:
        values = {key: float(value) for key, value in spec.items() if key != "planes"}
        extra = {"planes": int(spec["planes"])} if "planes" in spec else {}
    except (TypeError, ValueError):
        raise ValueError(f"fatigue: the parameters of the criterion {name!r} are numbers") from None
    if all(key in values for key in coefficients) and not any(key in values for key in limits):
        return cls(*(values[key] for key in coefficients), **extra)
    if all(key in values for key in limits) and not any(key in values for key in coefficients):
        return cls.from_limits(*(values[key] for key in limits), **extra)
    raise ValueError(
        f"fatigue: the criterion {name!r} takes its limits {list(limits)} or its "
        f"coefficients {list(coefficients)}"
    )


# -- the fields of a loading path -----------------------------------------------------------------


@dataclass
class FatigueResult:
    """The fatigue of the points of a cell along a cycle: the ``equivalent`` stress of
    the criterion of each point and its ``utilization``, the equivalent stress over the
    limit (1: at the fatigue limit; NaN where no criterion applies), the ``amplitude``
    ``sqrt(J2)_a`` of the deviator, the largest and the mean hydrostatic stress
    (``hydrostatic_max``, ``hydrostatic_mean``), the ``times`` of the cycle, the
    ``weights`` of the points (volumes, when their support gives them) and their
    ``support`` (mesh or image)."""

    equivalent: np.ndarray
    utilization: np.ndarray
    amplitude: np.ndarray
    hydrostatic_max: np.ndarray
    hydrostatic_mean: np.ndarray
    times: np.ndarray
    weights: np.ndarray | None = field(default=None, repr=False)
    support: object | None = field(default=None, repr=False)

    @property
    def largest(self) -> float:
        """The largest utilization of the cell (NaN when no point is judged)."""
        judged = self.utilization[np.isfinite(self.utilization)]
        return float(judged.max()) if judged.size else float("nan")

    @property
    def safety(self) -> float:
        """The factor by which the stresses of the cycle may be multiplied before a
        point reaches its limit, for a loading that grows in proportion: the inverse of
        the largest utilization."""
        largest = self.largest
        return 1.0 / largest if largest > 0 else float("inf")

    def fraction(self, level: float = 1.0) -> float:
        """The share (of the volume, with ``weights``; of the points otherwise) of the
        points judged whose utilization is above ``level``."""
        judged = np.isfinite(self.utilization)
        if not judged.any():
            return float("nan")
        weights = np.ones(self.utilization.shape) if self.weights is None else self.weights
        above = judged & (self.utilization > level)
        return float(weights[above].sum() / weights[judged].sum())

    def fields(self) -> dict:
        """The fields of the points, by name (for the writers of fields)."""
        return {
            "fatigue_utilization": self.utilization,
            "fatigue_equivalent": self.equivalent,
            "fatigue_amplitude": self.amplitude,
            "fatigue_hydrostatic_max": self.hydrostatic_max,
        }

    def components(self) -> dict:
        """The scalar results: the largest utilization, the safety factor and the share
        of the points beyond their limit."""
        return {"utilization": self.largest, "safety": self.safety, "beyond": self.fraction()}


def fatigue(history, criterion, *, phases=None, weights=None, times=None, support=None):
    """The fatigue of points along a cycle of stresses (``history``: steps x ... x 6):
    a :class:`FatigueResult`. ``criterion``: one criterion for all the points, or a
    mapping of the phases to their criteria (or to mappings of them:
    :func:`criterion_from_config`), with ``phases`` the phase of each point; the points
    of the other phases are not judged."""
    stress = np.asarray(history, dtype=float)
    if stress.ndim >= 2 and stress.shape[-1] == 6 and np.isnan(stress).any():
        return _fatigue_of_the_solid(stress, criterion, phases, weights, times, support)
    stress = _history(stress)
    shape = stress.shape[1:-1]
    amplitude, _ = stress_amplitude(stress)
    hydro = hydrostatic(stress)
    equivalent = np.full(shape, np.nan)
    utilization = np.full(shape, np.nan)
    if isinstance(criterion, dict) and "criterion" not in criterion:
        if phases is None:
            raise ValueError("fatigue: criteria by phase need the phase of each point (phases)")
        labels = np.asarray(phases)
        if labels.shape != shape:
            raise ValueError(f"fatigue: one phase per point {shape}, got {labels.shape}")
        groups = [(labels == phase, law) for phase, law in criterion.items()]
    else:
        groups = [(np.ones(shape, dtype=bool), criterion)]
    for mask, law in groups:
        law = criterion_from_config(law) if isinstance(law, dict) else law
        if not hasattr(law, "equivalent") or not hasattr(law, "limit"):
            raise TypeError("fatigue: a criterion (Crossland, Sines, DangVan, Findley, Matake)")
        if not mask.any():
            continue
        equivalent[mask] = law.equivalent(stress[:, mask])
        utilization[mask] = equivalent[mask] / law.limit
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != shape:
            raise ValueError(f"fatigue: one weight per point {shape}, got {weights.shape}")
    count = stress.shape[0]
    return FatigueResult(
        equivalent=equivalent,
        utilization=utilization,
        amplitude=amplitude,
        hydrostatic_max=hydro.max(axis=0),
        hydrostatic_mean=0.5 * (hydro.max(axis=0) + hydro.min(axis=0)),
        times=np.arange(count, dtype=float) if times is None else np.asarray(times, dtype=float),
        weights=weights,
        support=support,
    )


def _fatigue_of_the_solid(stress, criterion, phases, weights, times, support) -> FatigueResult:
    """:func:`fatigue` for points some of which have no stress (NaN: the voxels of the
    pores of an image): the others are judged, and these are left out of every field."""
    shape = stress.shape[1:-1]
    solid = ~np.isnan(stress).any(axis=(0, -1))
    if not solid.any():
        raise ValueError("fatigue: stresses that are not finite")
    labels = None if phases is None else np.asarray(phases)
    if labels is not None and labels.shape != shape:
        raise ValueError(f"fatigue: one phase per point {shape}, got {labels.shape}")
    if weights is not None and np.shape(weights) != shape:
        raise ValueError(f"fatigue: one weight per point {shape}, got {np.shape(weights)}")
    part = fatigue(
        stress[:, solid],
        criterion,
        phases=None if labels is None else labels[solid],
        times=times,
    )

    def whole(values):
        out = np.full(shape, np.nan)
        out[solid] = values
        return out

    return FatigueResult(
        equivalent=whole(part.equivalent),
        utilization=whole(part.utilization),
        amplitude=whole(part.amplitude),
        hydrostatic_max=whole(part.hydrostatic_max),
        hydrostatic_mean=whole(part.hydrostatic_mean),
        times=part.times,
        weights=None if weights is None else np.asarray(weights, dtype=float),
        support=support,
    )


def path_fatigue(result, criterion, *, cycle=None) -> FatigueResult:
    """The fatigue of the points of a cell along a cycle of a loading path: ``result``, a
    :class:`~morphogen.homogenization.paths.PathResult` that kept its fields at the
    steps of the cycle (``fields="steps"``); ``criterion``: as for :func:`fatigue`, the
    phases being those of the support of the fields; ``cycle``: the times of its start
    and of its end (by default the whole path). The stresses are those of the fields
    (Cauchy's in finite strains)."""
    times = np.array(sorted(result.fields))
    if cycle is not None:
        start, end = (float(v) for v in cycle)
        if not start < end:
            raise ValueError("fatigue: a cycle from a time to a later one")
        slack = 1e-9 * max(abs(start), abs(end), 1.0)
        times = times[(times >= start - slack) & (times <= end + slack)]
    times = np.array([t for t in times if "stress" in result.fields[t]])
    if times.size < 2:
        raise ValueError(
            "fatigue: the stress fields of two steps of the cycle at least (run the path "
            "with fields='steps')"
        )
    history = np.stack([np.asarray(result.fields[t]["stress"], dtype=float) for t in times])
    support = result.support
    phases = weights = None
    if support is not None and hasattr(support, "volumes"):  # a mesh
        phases, weights = np.asarray(support.phase), np.abs(support.volumes())
    elif support is not None and hasattr(support, "data"):  # an image
        phases = np.asarray(support.data)
    if phases is not None and phases.shape != history.shape[1:-1]:
        phases = weights = None
    return fatigue(history, criterion, phases=phases, weights=weights, times=times,
                   support=support)  # fmt: skip
