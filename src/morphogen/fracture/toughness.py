"""The toughness of the phases of a morphology and of the interfaces between them, and the
direction of each advance of a crack through them: the kink of the bulk against the
deflection along an interface ahead of the front (He & Hutchinson 1989), which turns cracks
along the weak interfaces of architectured materials (the brick-and-mortar of nacre, the
plies of Bouligand structures) and holds them back at tough phases.

*Toughness* (:class:`Toughness`): a critical energy release rate ``Γ`` (in the units of
``G``) for each phase, that of its material, and for the interfaces between two phases (a
value per pair, or one for all; none: no interface criterion).

*Kinked factors* (Cotterell & Rice 1980): a kink of infinitesimal length at the angle ``θ``
(from the direction of propagation towards the normal of the crack) has to first order the
factors ``k_I(θ) = cos³(θ/2) K_I - 3 sin(θ/2) cos²(θ/2) K_II`` and ``k_II(θ) = sin(θ/2)
cos²(θ/2) K_I + cos(θ/2) (1 - 3 sin²(θ/2)) K_II``, the hoop and shear stresses of the
asymptotic field on the ray ``θ`` (:func:`kinked_factors`), and the energy release rate
``G(θ) = (k_I² + k_II²) / E' + K_III² / 2μ`` (:func:`kinked_release_rate`; a kink whose
``k_I`` would be negative is closed, its mode I releasing nothing; the tearing mode is
kept). ``k_II`` vanishes in the direction of the largest hoop stress (Erdogan & Sih), where
``k_I`` is largest. A crack normal to an interface in pure mode I has ``G(±90°) = G(0) / 4``.

*Candidates* (:func:`advance_candidates`), in the plane of growth through the middle of the
front (straight fronts of :class:`~morphogen.fracture.mixed.KinkedCrack`):

* the bulk: the kink of the largest hoop stress (straight ahead with ``kink=False``), by one
  increment; its toughness is that of the toughest solid phase met along it, the phase at the
  front or a tougher one that the advance enters;
* the interfaces (:func:`interface_paths`), when the toughness has interfaces: where the front
  lies on an interface (two of its phases within a quarter of an element of the front, ahead
  of it) or meets one straight ahead within an increment, the chords of one increment from
  that point along the interfaces running from it (to the transitions of the phases on the
  circle of one increment around it, the phases on the two sides of the chord those of the
  interface along it), in the senses within 90° of the direction of the crack: the crack
  goes on to the interface and turns along it, and follows a curved interface by chords. A
  chord along which the pair of phases changes, at a junction of interfaces (a head joint
  meeting the interface between two layers of a brick-and-mortar), stops there, or, within
  half an increment, the branches of the junction are the advances. Their toughness is that
  of the interface.

The phases at points are those of the ``morphology`` of the toughness (exact for a
morphology of point queries), or those of the elements of the crack mesh containing them
(the interfaces of conforming meshes, or the staircases of the others). The bulk takes the
driving force of the crack (``G`` of the G-θ method, ``J_tip`` of an inelastic bulk), the
interfaces that driving force in the proportion of their first-order ``G(θ)`` to that of
the kink of the bulk, so that the choice between them is that of the first-order kinked
factors and the bulk alone is the growth without interfaces. The advance is the candidate of
the largest ratio ``G(θ) / Γ(θ)`` (:func:`choose_advance`; the bulk on a tie): an elastic
body grows at the load factor ``sqrt(Γ / G)`` of that candidate, an inelastic one once its
ratio reaches 1 (the driving force at the tip against the toughness of the phase at the
front, unless an interface ahead or a tougher phase within the increment rules the
advance).

*Deflection or penetration* (He & Hutchinson 1989): a crack normal to the interface of two
identical materials deflects into it, at ±90°, when ``Γ_interface / Γ_bulk`` is below
``G(90°) / G(0) = 1/4`` (0.26 for the kinks of He & Hutchinson, α = β = 0), and goes through
it above. Once along an interface the crack follows it while that stays the most favourable:
along an interface parallel to the remote tension, in pure mode II, rather than kink back
into the bulk while ``Γ_interface / Γ_bulk < G(0) / G(-70.5°) = 3/4``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The senses of an interface kept: within 90° of the direction of the crack (and 5°)
_AHEAD = -math.sin(math.radians(5.0))
#: Points of the circles of the transitions of the phases (1.5°)
_CIRCLE = 240


@dataclass(frozen=True)
class Toughness:
    """The toughness of the phases and of their interfaces (see the module): ``phases`` one
    value for every phase or a dict of the phases (``default``: that of the phases not in
    it); ``interfaces`` None (no interface criterion), one value for every interface between
    two solid phases, or a dict of pairs of phases (in either order; the pairs not in it: no
    criterion); ``morphology``: the phases at points whose interfaces the advances follow (a
    function of points, or a microstructure: :func:`~morphogen.fracture.lefm.phases_at`; by
    default the phases of the elements of the crack meshes of a morphology)."""

    phases: Any = 1.0
    interfaces: Any = None
    default: float | None = None
    morphology: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        def positive(value, what: str) -> float:
            value = float(value)
            if not value > 0:
                raise ValueError(f"Toughness: a positive toughness of {what}")
            return value

        if isinstance(self.phases, dict):
            phases = {int(p): positive(v, f"the phase {p}") for p, v in self.phases.items()}
        else:
            phases = positive(self.phases, "the phases")
        object.__setattr__(self, "phases", phases)
        if self.default is not None:
            object.__setattr__(self, "default", positive(self.default, "the other phases"))
        interfaces = self.interfaces
        if isinstance(interfaces, dict):
            pairs = {}
            for pair, value in interfaces.items():
                a, b = (int(p) for p in pair)
                if a == b:
                    raise ValueError("Toughness: an interface between two phases")
                pairs[(min(a, b), max(a, b))] = positive(value, f"the interface {a}/{b}")
            interfaces = pairs
        elif interfaces is not None:
            interfaces = positive(interfaces, "the interfaces")
        object.__setattr__(self, "interfaces", interfaces)

    @classmethod
    def of(cls, value) -> Toughness | None:
        """A toughness from None, a :class:`Toughness`, one value for all the phases or a
        dict of the phases."""
        if value is None or isinstance(value, Toughness):
            return value
        if isinstance(value, dict):
            return cls(dict(value))
        return cls(float(value))

    @property
    def uniform(self) -> bool:
        """Whether it is one value for all the phases without interfaces: the toughness of
        a homogeneous body, the growth without interfaces."""
        if self.interfaces is not None:
            return False
        if not isinstance(self.phases, dict):
            return True
        values = set(self.phases.values()) | ({self.default} - {None})
        return len(values) == 1

    @property
    def value(self) -> float:
        """The toughness of a uniform toughness (:attr:`uniform`)."""
        if not self.uniform:
            raise ValueError("Toughness: not one value for all the phases")
        if isinstance(self.phases, dict):
            return next(iter(set(self.phases.values()) | ({self.default} - {None})))
        return self.phases

    def of_phase(self, phase: int) -> float:
        """The toughness of a phase."""
        if not isinstance(self.phases, dict):
            return self.phases
        if int(phase) in self.phases:
            return self.phases[int(phase)]
        if self.default is not None:
            return self.default
        raise KeyError(f"Toughness: none for the phase {phase} (a default)")

    def of_interface(self, a: int, b: int) -> float | None:
        """The toughness of the interface between two phases (None: no criterion)."""
        if self.interfaces is None or int(a) == int(b):
            return None
        if not isinstance(self.interfaces, dict):
            return self.interfaces
        return self.interfaces.get((min(int(a), int(b)), max(int(a), int(b))))

    def phases_at(self, points) -> np.ndarray:
        """The phases of the ``morphology`` at points (rows): its point queries, or a voxel
        image of it made once (:func:`~morphogen.fracture.lefm.phases_at`)."""
        if self.morphology is None:
            raise ValueError("Toughness: no morphology")
        lookup = self.__dict__.get("_lookup")
        if lookup is None:
            lookup = _phase_function(self.morphology)
            object.__setattr__(self, "_lookup", lookup)
        return np.asarray(lookup(np.atleast_2d(np.asarray(points, dtype=float))), dtype=np.int64)


def _phase_function(morphology):
    """The phases of a morphology at points: a function of points, the point queries of a
    two-phase microstructure, or a voxel image of another made once (as
    :func:`~morphogen.fracture.lefm.phases_at`)."""
    from morphogen.fracture.lefm import phases_at

    if callable(morphology) and not hasattr(morphology, "domain"):
        return morphology
    names = getattr(morphology, "phase_names", None) or {0: "0", 1: "1"}
    if set(names) <= {0, 1}:
        try:
            morphology.contains(np.asarray(morphology.domain.lower, dtype=float)[None])
            return lambda points: phases_at(morphology, points)
        except NotImplementedError:
            pass
    lengths = np.asarray(morphology.domain.lengths, dtype=float)
    image = morphology.voxelize(voxel_size=float(lengths.max()) / 128.0)
    data = np.asarray(image.data)
    if data.dtype.kind not in "iub":
        raise ValueError("Toughness: the morphology gives no phase image")
    origin = np.asarray(image.origin, dtype=float)
    steps = np.asarray(image.spacing, dtype=float)
    shape = np.asarray(data.shape)

    def lookup(points):
        index = np.clip(np.floor((points - origin) / steps).astype(np.int64), 0, shape - 1)
        return data[index[:, 0], index[:, 1], index[:, 2]]

    return lookup


def kinked_factors(k1, k2, angle):
    """The stress intensity factors ``(k_I, k_II)`` of a kink of infinitesimal length at
    ``angle`` (radians, from the direction of propagation towards the normal of the crack)
    from a front of factors ``k1``, ``k2`` (Cotterell & Rice 1980, to first order)."""
    half = 0.5 * np.asarray(angle, dtype=float)
    c, s = np.cos(half), np.sin(half)
    return c**3 * k1 - 3.0 * s * c**2 * k2, s * c**2 * k1 + c * (1.0 - 3.0 * s**2) * k2


def kinked_release_rate(k1, k2, k3, angle, e_prime: float, mu: float):
    """The energy release rate ``G(θ) = (k_I² + k_II²) / E' + K_III² / 2μ`` of the kink at
    ``angle`` (:func:`kinked_factors`; a negative ``k_I``, a closed kink, counts nothing)."""
    a, b = kinked_factors(k1, k2, angle)
    return (np.maximum(a, 0.0) ** 2 + b**2) / e_prime + k3**2 / (2.0 * mu)


@dataclass(frozen=True)
class Advance:
    """A candidate advance of a crack (see the module): ``kind`` ``"bulk"`` or
    ``"interface"``, the ``angle`` of its last segment from the last segment of the crack
    (radians, towards its normal), the ``crack`` after it, the energy release rate ``G(θ)``
    of its direction (``release_rate``), the toughness ``Γ(θ)`` it meets and what it meets
    (``met``: the phase whose toughness it is, or the pair of phases of the interface)."""

    kind: str
    angle: float
    crack: Any = field(repr=False)
    release_rate: float
    toughness: float
    met: Any

    @property
    def ratio(self) -> float:
        """``G(θ) / Γ(θ)``: the advance happens once it reaches 1."""
        return self.release_rate / self.toughness

    @property
    def label(self) -> str:
        """What the advance meets: a phase, or the interface ``p/q``."""
        if self.kind == "interface":
            return "/".join(str(p) for p in self.met)
        return str(self.met)


# -- the phases in the plane of growth ------------------------------------------------------


class _Phases:
    """The solid phase at points of a solved body (``-1``: a void phase, or outside the
    body): that of the morphology of a ``toughness``, or of the element of the crack mesh
    containing each point."""

    def __init__(self, solution, toughness: Toughness):
        mesh = solution.mesh
        self.mesh = mesh
        self.toughness = toughness if toughness.morphology is not None else None
        self.voids = [int(p) for p, m in solution.materials.items() if getattr(m, "void", False)]
        self.lower, self.upper = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
        self._tree = None

    def __call__(self, points) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if self.toughness is not None:
            labels = self.toughness.phases_at(points)
            tol = 1e-9 * float(np.max(self.upper - self.lower))
            inside = np.all((points >= self.lower - tol) & (points <= self.upper + tol), axis=1)
            labels = np.where(inside, labels, -1)
        else:
            element = self._containing(points)
            labels = np.where(element >= 0, self.mesh.phase[np.maximum(element, 0)], -1)
        if self.voids:
            labels = np.where(np.isin(labels, self.voids), -1, labels)
        return labels

    def _containing(self, points, tol: float = 1e-9) -> np.ndarray:
        """The element containing each point (-1: none), among the elements of the nearest
        centroids, 24 of them, then more for the points not found (the flat elements of a
        crack along an interface have far centroids)."""
        from scipy.spatial import cKDTree

        x = self.mesh.nodes[self.mesh.elements[:, :4]]
        if self._tree is None:
            self._tree = cKDTree(x.mean(axis=1))
        out = np.full(len(points), -1, dtype=np.int64)
        todo = np.arange(len(points))
        for k in (24, 128, 512):
            count = min(k, self.mesh.n_elements)
            _, near = self._tree.query(points[todo], k=count)
            near = near.reshape(len(todo), count)
            origin = x[near, 0]
            edges = np.stack([x[near, 1] - origin, x[near, 2] - origin, x[near, 3] - origin],
                             axis=-1)  # fmt: skip
            local = np.linalg.solve(edges, (points[todo, None, :] - origin)[..., None])[..., 0]
            smallest = np.minimum(local.min(axis=2), 1.0 - local.sum(axis=2))
            best = np.argmax(smallest, axis=1)
            rows = np.arange(len(todo))
            found = smallest[rows, best] >= -tol
            out[todo[found]] = near[rows, best][found]
            todo = todo[~found]
            if not len(todo) or count == self.mesh.n_elements:
                break
        return out


def _plane(crack, mesh):
    """The point of the front in the middle of the body, the direction ``e`` of the last
    segment and its normal ``n``: the plane of growth."""
    lower, upper = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    points, _ = crack.front(lower, upper, 64)
    origin = points.mean(axis=0) if len(points) else np.asarray(crack.tip, dtype=float)
    _, n, d, _, _ = crack.frame(origin[None])
    return origin, d[0], n[0]


def _toughest_along(phases, toughness: Toughness, origin, direction, length: float, step: float):
    """The toughest solid phase met along the segment from ``origin`` (left out) over
    ``length`` in ``direction`` and its toughness (None, None: no solid phase)."""
    s = np.arange(step, length + 0.5 * step, step)
    labels = phases(origin + s[:, None] * direction)
    met = np.unique(labels[labels >= 0])
    if not len(met):
        return None, None
    values = [toughness.of_phase(int(p)) for p in met]
    k = int(np.argmax(values))
    return int(met[k]), float(values[k])


def _interface(toughness: Toughness, a, b) -> np.ndarray:
    """Whether the phases ``a`` and ``b`` (arrays) are the two sides of an interface with a
    toughness."""
    out = (a != b) & (a >= 0) & (b >= 0)
    for i in np.flatnonzero(out):
        out[i] = toughness.of_interface(int(a[i]), int(b[i])) is not None
    return out


def _bisect(phases, start, end, label, iterations: int = 14):
    """The points between ``start`` (the phase ``label``) and ``end`` (another) where the
    phase changes (rows), by bisection."""
    lo, hi = np.array(start, dtype=float), np.array(end, dtype=float)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        same = phases(mid) == label
        lo = np.where(same[:, None], mid, lo)
        hi = np.where(same[:, None], hi, mid)
    return 0.5 * (lo + hi)


def _near(phases, toughness: Toughness, origin, e, n, radius: float) -> bool:
    """Whether an interface with a toughness passes within ``radius`` of ``origin`` ahead
    (within 90° of ``e``): two of its phases on a small circle around it."""
    a = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    a = a[np.cos(a) >= _AHEAD]
    points = origin + radius * (np.outer(np.cos(a), e) + np.outer(np.sin(a), n))
    labels = np.unique(np.concatenate([phases(origin[None]), phases(points)]))
    labels = [int(p) for p in labels if p >= 0]
    return any(toughness.of_interface(p, q) is not None
               for i, p in enumerate(labels) for q in labels[i + 1 :])  # fmt: skip


def _crossing(phases, toughness: Toughness, origin, e, n, reach: float, step: float, offsets):
    """The distance along ``e`` (from ``origin`` displaced by each of ``offsets`` along
    ``n``) of the first change of phase within ``reach`` between the two sides of an
    interface with a toughness (None: none, or another change first)."""
    s = np.arange(step, reach + 0.5 * step, step)
    best = None
    for b in offsets:
        start = origin + b * n
        labels = phases(start + s[:, None] * e)
        change = np.flatnonzero(labels != labels[0])
        if not len(change) or labels[0] < 0:
            continue
        k = int(change[0])
        if not _interface(toughness, labels[k - 1 : k], labels[k : k + 1])[0]:
            continue
        point = _bisect(phases, (start + s[k - 1] * e)[None], (start + s[k] * e)[None],
                        labels[k - 1])[0]  # fmt: skip
        distance = float((point - origin) @ e)
        best = distance if best is None else min(best, distance)
    return best


def _sides(phases, toughness: Toughness, centre, u, across, distances, width: float):
    """Along the chord from ``centre`` in the direction ``u``, at ``distances``: whether the
    phases ``width`` apart on its two sides are those of an interface with a toughness, and
    their pair (the smaller phase, the larger)."""
    points = centre + np.outer(distances, u)
    left, right = phases(points + width * across), phases(points - width * across)
    return _interface(toughness, left, right), np.minimum(left, right), np.maximum(left, right)


def _branches(phases, toughness: Toughness, centre, e, n, radius: float, width: float):
    """The interfaces running from ``centre`` (see the module): for each transition of the
    phases on the circle of ``radius`` around it between the two sides of an interface with
    a toughness, the unit direction ``u`` of the chord to it, the pair of phases of the
    interface along the chord from ``centre`` (the phases ``width`` apart on its two sides),
    the length of the advance along it and whether it ends at a junction: the chord, or its
    part up to where that pair changes (another interface meets it, or it curves away)."""
    a = np.linspace(0.0, 2.0 * np.pi, _CIRCLE, endpoint=False)
    points = centre + radius * (np.outer(np.cos(a), e) + np.outer(np.sin(a), n))
    labels = phases(points)
    k = np.flatnonzero(_interface(toughness, labels, np.roll(labels, -1)))
    if not len(k):
        return []
    ends = _bisect(phases, points[k], np.roll(points, -1, axis=0)[k], labels[k])
    axis = np.cross(e, n)
    t = np.arange(1, 17) / 16.0
    out = []
    for y in ends:
        u = (y - centre) / np.linalg.norm(y - centre)
        across = np.cross(axis, u)  # the normal of the chord in the plane
        valid, lo, hi = _sides(phases, toughness, centre, u, across, t * radius, width)
        good = np.flatnonzero(valid & (t >= 0.1))
        if not len(good) or good[0] > 2:  # the chord does not start along an interface
            continue
        first = int(good[0])
        pair = (int(lo[first]), int(hi[first]))
        same = valid & (lo == pair[0]) & (hi == pair[1])
        same[:first] = True
        if same.all():
            out.append((u, pair, radius, False))
            continue
        stop = int(np.argmin(same))
        s0, s1 = t[stop - 1] * radius, t[stop] * radius
        for _ in range(12):  # where the pair changes
            mid = 0.5 * (s0 + s1)
            v, low, high = _sides(phases, toughness, centre, u, across, [mid], width)
            if v[0] and (int(low[0]), int(high[0])) == pair:
                s0 = mid
            else:
                s1 = mid
        length = 0.5 * (s0 + s1)
        if length >= 0.25 * radius:
            out.append((u, pair, length, True))
    return out


def _extended(crack, points):
    """The crack grown through the points (in the plane of growth), a segment each."""
    for p in points:
        _, n, d, _, _ = crack.frame(crack.tip[None])
        v = np.asarray(p, dtype=float) - crack.tip
        a, b = float(v @ d[0]), float(v @ n[0])
        length = math.hypot(a, b)
        if length > 1e-9:
            angle = math.atan2(b, a)
            crack = crack.grown(length, angle if abs(angle) > 1e-12 else 0.0)
    return crack


def interface_paths(solution, toughness, increment: float) -> list:
    """The advances of the crack of a solved body along the interfaces within an
    ``increment`` ahead of its front (see the module): ``(angle, crack, pair)`` for each, the
    angle of its last segment from the last segment of the crack, the crack after it and the
    pair of phases of the interface (none without interfaces in the toughness)."""
    rule = Toughness.of(toughness)
    if rule is None or rule.interfaces is None:
        return []
    crack, mesh = solution.crack, solution.mesh
    if not hasattr(crack, "points"):
        raise ValueError("interfaces: a straight front (a KinkedCrack)")
    if rule.morphology is None and mesh.side is None:
        raise ValueError("interfaces: a crack mesh of a morphology, or the morphology")
    phases = _Phases(solution, rule)
    origin, e, n = _plane(crack, mesh)
    h = solution.element_size()
    increment = float(increment)
    on = 0.25 * min(h, increment)
    step = min(h, increment) / 8.0
    exact = rule.morphology is not None or "multi-material" in str(mesh.metadata.get("mesher"))
    width = 0.15 * increment if exact else max(0.15 * increment, 0.5 * h)  # staircases
    # where the crack meets an interface: on it at the front, or straight ahead (a path to it)
    starts = []
    near = _near(phases, rule, origin, e, n, on)
    if near:
        starts.append((origin, (), e))
    meet = _crossing(phases, rule, origin, e, n, increment, step, (on, -on) if near else (0.0,))
    if meet is not None and meet > on:
        starts.append((origin + meet * e, (origin + meet * e,), e))
    out: list = []
    while starts:
        centre, approach, arrival = starts.pop(0)
        for u, pair, length, junction in _branches(phases, rule, centre, e, n, increment, width):
            if u @ arrival < _AHEAD:  # within 90° of the direction of the crack
                continue
            end = centre + length * u
            if junction and length < 0.5 * increment and len(approach) < 2:
                starts.append((end, (*approach, end), u))  # a junction close ahead: its branches
                continue
            angle = math.atan2(float(u @ n), float(u @ e))
            if any(p == pair and abs(math.remainder(angle - a, 2 * math.pi)) < math.radians(5.0)
                   for a, _, p in out):  # fmt: skip
                continue
            grown = _extended(crack, (*approach, end))
            # the lips are the faces behind the plane of the front: no path turned back on itself
            if np.any(grown.psi(np.asarray(grown.points[:-1])) > 0.05 * step):
                continue
            out.append((angle, grown, pair))
    return out


def advance_candidates(solution, toughness, increment: float, *, modes=None, driving=None,
                       kink: bool = True, paths=None) -> list[Advance]:  # fmt: skip
    """The candidate advances of the crack of a solved body by ``increment`` (see the
    module): the bulk, and along the interfaces ahead (``paths``: those of
    :func:`interface_paths`, computed by default). ``modes``: those of
    :meth:`~morphogen.fracture.mixed.CrackSolution.modes` (computed by default); ``driving``:
    the driving force of the bulk (by default ``G`` of the modes; ``J_tip`` of an inelastic
    bulk); ``kink``: the bulk in the direction of the largest hoop stress, else straight."""
    from morphogen.fracture.mixed import _isotropic_moduli

    rule = Toughness.of(toughness)
    if rule is None:
        raise ValueError("advance_candidates: a toughness")
    crack = solution.crack
    if modes is None:
        modes = solution.modes()
    k = np.array([modes["K_I"], modes["K_II"], modes["K_III"]], dtype=float)
    finite = bool(np.all(np.isfinite(k)))
    angle = float(modes["angle"]) if kink and finite else 0.0
    driving = float(modes["G"] if driving is None else driving)
    grown = crack.grown(increment, angle)
    if rule.uniform:
        return [Advance("bulk", angle, grown, driving, rule.value, solution.front_phase())]
    if rule.morphology is None and solution.mesh.side is None:
        raise ValueError("a toughness per phase or of interfaces: a crack mesh of a morphology, "
                         "or the morphology")  # fmt: skip
    phases = _Phases(solution, rule)
    origin, e, n = _plane(crack, solution.mesh)
    h = solution.element_size()
    step = min(h, float(increment)) / 8.0
    direction = math.cos(angle) * e + math.sin(angle) * n
    met, gamma = _toughest_along(phases, rule, origin, direction, float(increment), step)
    if met is None:  # nothing solid ahead (a pore): the phase at the front
        met = solution.front_phase()
        gamma = rule.of_phase(met)
    out = [Advance("bulk", angle, grown, driving, gamma, met)]
    if rule.interfaces is None or not finite:
        return out
    young, nu = _isotropic_moduli(solution.materials[solution.front_phase()])
    e_prime, mu = young / (1.0 - nu**2), young / (2.0 * (1.0 + nu))
    reference = float(kinked_release_rate(*k, angle, e_prime, mu))
    if paths is None:
        paths = interface_paths(solution, rule, increment)
    for theta, path, pair in paths:
        rate = float(kinked_release_rate(*k, theta, e_prime, mu))
        rate = driving * rate / reference if reference > 0 else 0.0
        out.append(Advance("interface", theta, path, rate, rule.of_interface(*pair), pair))
    return out


def choose_advance(candidates) -> Advance:
    """The candidate of the largest ratio ``G(θ) / Γ(θ)`` (the first, the bulk, on a tie)."""
    best = None
    for c in candidates:
        if best is None or c.ratio > best.ratio * (1.0 + 1e-12):
            best = c
    if best is None:
        raise ValueError("choose_advance: no candidate")
    return best
