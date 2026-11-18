"""Crystal plasticity of Méric & Cailletaud (1991).

Plastic flow is the sum of the slips ``gamma^s`` on the slip systems ``s`` of the lattice
(plane normal ``n^s``, slip direction ``l^s``), driven by the resolved shear stress
``tau^s = sigma : m^s``, ``m^s = sym(l^s (x) n^s)``:

* viscoplastic slip ``dgamma^s/dt = <(|tau^s - x^s| - r^s) / K>^n sign(tau^s - x^s)``;
* kinematic hardening of each system ``x^s = C alpha^s``, ``dalpha^s = dgamma^s - D
  alpha^s |dgamma^s|``;
* isotropic hardening ``r^s = tau_0 + Q sum_r h_sr (1 - exp(-b v^r))`` with the cumulated
  slips ``v^r`` (``dv^r = |dgamma^r|``) and the interaction matrix ``h``;
* ``deps_p = sum_s dgamma^s m^s`` in small strains; ``dF_p F_p^-1 = sum_s dgamma^s l^s (x)
  n^s`` in finite strains (multiplicative decomposition ``F = F_e F_p``, the slip systems
  fixed in the intermediate configuration, the lattice rotating with ``F_e``).

The parameters ``tau_0, Q, b, C, D, K, n`` belong to the families of slip systems. Slip
systems are given in the axes of the crystal; each grain turns them by its orientation,
as its elasticity (:mod:`morphogen.homogenization.materials`).

Families: ``octahedral`` {111}<110> and ``cube`` {100}<110> (face-centred cubic crystals,
and the cube slip of nickel-base superalloys), ``bcc110`` {110}<111> and ``bcc112``
{112}<111> (body-centred cubic crystals); for hexagonal crystals of ratio ``c_over_a``
(axis c along the third axis of the crystal, a_1 along the first), ``basal`` (0001)<11-20>,
``prismatic`` {10-10}<11-20>, ``pyramidal_a`` {10-11}<11-20>, ``pyramidal_ca`` {10-11}<11-23>
and ``pyramidal2_ca`` {11-22}<11-23>.

**Twinning** (``twins``) is a slip in one sense (the pseudo-slip of Kalidindi 1998):
a twin system of plane ``K_1`` and direction ``eta_1`` shears by ``gamma`` when the
fraction ``f = gamma / gamma_tw`` of the crystal has twinned, ``gamma_tw`` the shear of the
twin (the one that leaves the second plane ``K_2`` undistorted). Its flow rule is that of
a slip system for a positive resolved shear stress, with a resistance that grows without
bound as the twinned fraction ``F = sum_t gamma^t / gamma_tw^t`` comes to its saturation
``F_sat`` (a barrier: 1 % of ``tau_0`` at 90 % of the saturation, ``tau_0`` at 99.9 %)::

    dgamma^t/dt = <(tau^t - x^t - r^t - 10^-3 tau_0 F / (F_sat - F)) / K>^n,

so that the twins stop there, whatever the stress. Under a stress that is reversed the
twins shear back, down to none (detwinning, at the resistance of the twinning without
its barrier, or at a share of it: ``twin_detwinning``, a number; ``false``: they stay).
Families: ``twin111`` {111}<11-2>
(face-centred cubic, shear 1 / sqrt 2), ``twin112`` {112}<11-1> (body-centred cubic, 1 /
sqrt 2), and for hexagonal crystals ``twin1012`` {10-12}<10-1-1> (which extends the axis c
below ``c / a = sqrt 3``: magnesium, titanium, and shortens it above: zinc), ``twin1011``
{10-11}<10-1-2>, ``twin1122`` {11-22}<11-2-3> and ``twin1121`` {11-21}<-1-126>.

**The lattice of a twin** (``twin_reorientation``, a twinned fraction): when the twinned
fraction of a point comes to it at the end of a step, the lattice of the point becomes
that of its predominant twin (Van Houtte 1978; Tomé, Lebensohn & Kocks 1991, at each
integration point): turned by half a turn about the normal of the twin plane, ``Q = 2 n
(x) n - I``, with its systems, its elasticity and its thermal strain. Its twins start
again from nothing; in the new lattice the system of that twin shears the other way and
brings the first lattice back, the other twins of its family being held. With ``{threshold,
growth}`` the lattices are turned grain by grain after each step (:func:`turn_twins`): the
most twinned points first, while the turned volume of the grain is below its twinned
volume. Without it the lattice of the twinned fraction is not turned: the slip of the
twins is not followed.

**Cleavage** (``cleavage``) opens crystallographic planes (the damage systems of Marchal,
Forest, Rémy & Duvinage 2006, see Aslan & Forest 2009): a plane of normal ``n`` has an
opening system, whose strain is ``delta n (x) n`` and which flows under a positive normal
stress ``n . sigma . n``, and accommodation systems ``sym(n (x) l)`` along directions
``l`` of the plane, which release its shear stresses once it has opened. Their flow rule
is that of the slip systems, with a strength that softens with the opening and the
slips of the plane::

    Y = Y_0 exp(-w d),    d = delta + sum_i |delta_i|,

(``shear`` times ``Y_0`` for the accommodation systems): the energy dissipated by a plane
that breaks is ``Y_0 / w`` per unit volume, a fracture energy ``G_c`` per unit area for
elements of size ``h`` when ``w = Y_0 h / G_c`` (the crack band of Bažant & Oh 1983),
which requires ``w Y_0`` below the stiffness of the opening (``h < E G_c / Y_0^2``).
Planes: ``cube`` {100}, ``dodecahedral`` {110} and ``octahedral`` {111} of cubic crystals,
``basal`` (0001) and ``prismatic`` {10-10} of hexagonal ones. Under pressure a plane
closes: its opening follows the sign of the normal stress, back to nothing (Aslan &
Forest 2009), and the closed plane carries the pressure as the sound crystal (the
unilateral condition; ``closure: false``: the opening stays, and the lattice alone is
pressed; a number: the share of its strength at which the plane closes). With a
``coupling`` the slip of the crystal nucleates its cleavage (Aslan, Cordero, Gaubert &
Forest 2011): the strength of the planes falls with the cumulated slip of the slip
systems, ``Y = Y_0 exp(-w d) exp(-Theta gamma_cum)``, and their damage takes ``Theta
exp(-Theta gamma_cum) sum_p (Y_0 / w) (1 - exp(-w d_p))`` off the resistance of the slip
systems (the two derive from one energy). With a ``regularization`` the planes are those
of the microdamage theory of Aslan & Forest (2009): the planes of a point have one
threshold, which softens linearly with the damage ``d`` of them all and hardens with
this damage less the microdamage of the point, ``Y = Y_0 (1 - w d) + chi_H (d - chi_d)``,
the microdamage ``chi_d`` following the damage over a length, ``chi_d - ell^2 lap chi_d
= d`` (the solvers of the loading paths solve it, and break through the planes whose
strength has come to nothing): a crack is then a band of a width that the elements do
not set (:func:`microdamage_band`). The lips of a plane under pressure rub: the strength
of its accommodation systems has Coulomb's ``friction`` times the pressure (0 by
default).

The interaction matrix is uniform (Taylor hardening) by default; ``[h_self, h_latent]``;
the six coefficients of Franciosi for the octahedral systems, in the order self,
coplanar, Hirth lock, collinear, glissile junction, Lomer lock; or a full matrix.

Integration is implicit (backward Euler, the Armstrong-Frederick variables in closed
form): Newton's method on the magnitudes of the slips of an active set of systems, with
the flow rule written as stresses (see the kernels below), and substeps when it fails;
its linearization gives the consistent tangent, symmetric in small strains without
isotropic hardening (``Q b = 0``) and not otherwise (the latent hardening of a system by
the slips of the others; the multiplicative decomposition of finite strains), symmetrized
for the conjugate gradient solvers or kept whole for GMRES (``sym``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import cache

import numpy as np
from numba import njit

from morphogen.homogenization.tensors import (
    elastic_stress,
    gauss_solve,
    inverse3,
    matmul3,
    matmult3,
    project,
    stress_tensor,
    symmetrize,
    tmatmul3,
)

#: Parameters of a family of slip systems, in the order of the kernels
PARAMS = ("tau0", "Q", "b", "C", "D", "K", "n")
_DEFAULTS = {"tau0": None, "Q": 0.0, "b": 0.0, "C": 0.0, "D": 0.0, "K": None, "n": None}
_ALIASES = {"r0": "tau0", "R0": "tau0", "q": "Q", "c": "C", "d": "D", "k": "K", "N": "n"}
#: Order of the six interaction coefficients of the octahedral systems
INTERACTIONS = ("self", "coplanar", "hirth", "collinear", "glissile", "lomer")


def _canonical(v) -> tuple[int, ...]:
    """The direction ``v`` or ``-v``: its first nonzero component positive."""
    v = tuple(int(x) for x in v)
    first = next(x for x in v if x != 0)
    return v if first > 0 else tuple(-x for x in v)


def _family_vectors(normals_type, directions_type):
    """Slip systems (normal, direction) of the cubic vectors of the given types: all the
    pairs of a plane normal and a perpendicular slip direction, normals and directions
    taken once up to their sign."""

    def vectors(kind):
        out = set()
        for perm in set(itertools.permutations(kind)):
            for signs in itertools.product((1, -1), repeat=3):
                v = tuple(s * p for s, p in zip(signs, perm, strict=True))
                if any(v):
                    out.add(_canonical(v))
        return sorted(out, reverse=True)

    systems = []
    for n in vectors(normals_type):
        for d in vectors(directions_type):
            if sum(a * b for a, b in zip(n, d, strict=True)) == 0:
                systems.append((n, d))
    return systems


#: Slip families: (normal type, direction type) of the cubic lattice
FAMILIES = {
    "octahedral": ((1, 1, 1), (1, 1, 0)),
    "cube": ((1, 0, 0), (1, 1, 0)),
    "bcc110": ((1, 1, 0), (1, 1, 1)),
    "bcc112": ((1, 1, 2), (1, 1, 1)),
}


#: Slip families of the hexagonal lattice: the plane (hkil) and the direction [uvtw] of
#: one of their systems
HEXAGONAL = {
    "basal": ((0, 0, 0, 1), (1, 1, -2, 0)),
    "prismatic": ((1, 0, -1, 0), (1, -2, 1, 0)),
    "pyramidal_a": ((1, 0, -1, 1), (1, -2, 1, 0)),
    "pyramidal_ca": ((1, 0, -1, 1), (-1, -1, 2, 3)),
    "pyramidal2_ca": ((1, 1, -2, 2), (-1, -1, 2, 3)),
}
#: Twinning modes: the lattice, the plane K_1 and the direction eta_1 of one of their
#: systems, and the direction eta_2 of its second undistorted plane (which sets the shear)
TWINS = {
    "twin111": ("cubic", (1, 1, 1), (1, 1, -2), (1, 1, 2)),
    "twin112": ("cubic", (1, 1, 2), (-1, -1, 1), (1, 1, 1)),
    "twin1012": ("hexagonal", (1, 0, -1, 2), (-1, 0, 1, 1), (1, 0, -1, 1)),
    "twin1011": ("hexagonal", (1, 0, -1, 1), (1, 0, -1, -2), (3, 0, -3, 2)),
    "twin1122": ("hexagonal", (1, 1, -2, 2), (1, 1, -2, -3), (2, 2, -4, 3)),
    "twin1121": ("hexagonal", (1, 1, -2, 1), (-1, -1, 2, 6), (1, 1, -2, 0)),
}
#: Cleavage planes: the lattice, one plane and the directions of its accommodation systems
CLEAVAGE = {
    "cube": ("cubic", (1, 0, 0), ((0, 1, 0), (0, 0, 1))),
    "dodecahedral": ("cubic", (1, 1, 0), ((0, 0, 1), (1, -1, 0))),
    "octahedral": ("cubic", (1, 1, 1), ((1, -1, 0), (1, 0, -1), (0, 1, -1))),
    "basal": ("hexagonal", (0, 0, 0, 1), ((1, 1, -2, 0), (1, -2, 1, 0), (-2, 1, 1, 0))),
    "prismatic": ("hexagonal", (1, 0, -1, 0), ((1, -2, 1, 0), (0, 0, 0, 1))),
}
#: Kinds of the systems of a crystal: slip, twin (in one sense), opening of a cleavage
#: plane (in one sense), accommodation of a cleavage plane
SLIP, TWIN, OPENING, ACCOMMODATION = 0, 1, 2, 3
#: The fields of the crystals that twin or cleave, and the curves of their means along a
#: loading path (:meth:`CrystalPlasticity.point_fields`)
FIELDS = (
    "twin_fraction",
    "twin_reoriented",
    "twin_misorientation",
    "cleavage_opening",
    "cleavage_damage",
    "cleavage_microdamage",
)


def lattice_of(family: str) -> str:
    """The lattice (``"cubic"`` or ``"hexagonal"``) of a family of slip systems, of twins
    or of cleavage planes."""
    if family in FAMILIES:
        return "cubic"
    if family in HEXAGONAL:
        return "hexagonal"
    for table in (TWINS, CLEAVAGE):
        if family in table:
            return table[family][0]
    raise ValueError(f"crystal: unknown family {family!r}")


@cache
def _rotations(lattice: str) -> tuple:
    """The proper rotations of the point group of a lattice (24 of the cube, 12 of the
    hexagonal prism), in the axes of the crystal."""
    out = []
    if lattice == "cubic":
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1.0, -1.0), repeat=3):
                m = np.zeros((3, 3))
                for i, (p, s) in enumerate(zip(perm, signs, strict=True)):
                    m[i, p] = s
                if np.linalg.det(m) > 0:
                    out.append(m)
        return tuple(out)
    for k in range(6):
        a = k * math.pi / 3.0
        c, s = math.cos(a), math.sin(a)
        out.append(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
        b = k * math.pi / 6.0  # a two-fold axis of the basal plane
        u = np.array([math.cos(b), math.sin(b), 0.0])
        out.append(2.0 * np.outer(u, u) - np.eye(3))
    return tuple(out)


def disorientation(rotations, lattice: str) -> np.ndarray:
    """The disorientation (degrees) of lattices turned by ``rotations`` (..., 3, 3; in the
    axes of the crystal) from the one that they were: the smallest angle of the rotations
    that bring the one onto the other, over the symmetries of the lattice (60 degrees for
    the twins of cubic crystals, 86.3 for the {10-12} twin of magnesium)."""
    ops = np.array(_rotations(lattice))
    traces = np.einsum("...ij,kji->...k", np.asarray(rotations, dtype=float), ops)
    cosine = np.clip(0.5 * (traces.max(axis=-1) - 1.0), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _ratio(c_over_a, family: str) -> float:
    if c_over_a is None:
        raise ValueError(f"crystal: the hexagonal family {family!r} needs the ratio c_over_a")
    ratio = float(c_over_a)
    if not (math.isfinite(ratio) and ratio > 0):
        raise ValueError(f"crystal: c_over_a is a positive number, got {c_over_a!r}")
    return ratio


def _vector(indices, lattice: str, ratio: float | None, plane: bool) -> np.ndarray:
    """The normal of a plane or a direction of a lattice in the Cartesian axes of the
    crystal (not normalized): Miller indices of the cube; Miller-Bravais indices of the
    hexagonal lattice, ``a_1`` along the first axis and ``c`` along the third."""
    if lattice == "cubic":
        return np.array(indices, dtype=float)
    h, k, i, last = indices
    if h + k + i != 0:
        raise ValueError(f"crystal: Miller-Bravais indices {indices}: the first three sum to 0")
    if plane:
        return np.array([h, (h + 2.0 * k) / math.sqrt(3.0), last / ratio])
    u, v = h - i, k - i  # the three-index direction
    return np.array([u - 0.5 * v, 0.5 * math.sqrt(3.0) * v, last * ratio])


def _key(*vectors) -> tuple:
    return tuple(np.round(np.concatenate(vectors), 9) + 0.0)


def _variants(normal, direction, lattice: str, polar: bool) -> list[tuple]:
    """The systems (unit normal, unit direction) that the rotations of the lattice make of
    one: once each up to the signs of the normal and of the direction (slip), or up to
    the sign of both together (``polar``: a twin, sheared in one sense)."""
    n0 = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    l0 = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    if abs(n0 @ l0) > 1e-9:
        raise ValueError("crystal: the direction of a system lies in its plane")
    signs = [(1, 1), (-1, -1)] if polar else [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    seen: set = set()
    out = []
    for r in _rotations(lattice):
        n, d = r @ n0, r @ l0
        keys = [_key(a * n, b * d) for a, b in signs]
        if not any(k in seen for k in keys):
            seen.add(keys[0])
            out.append((n, d))
    return out


def slip_systems(family: str, c_over_a: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Unit plane normals and slip directions (N x 3 each, axes of the crystal) of a
    family of slip systems; ``c_over_a``: the ratio of the hexagonal lattice."""
    if family in HEXAGONAL:
        ratio = _ratio(c_over_a, family)
        plane, direction = HEXAGONAL[family]
        pairs = _variants(
            _vector(plane, "hexagonal", ratio, True),
            _vector(direction, "hexagonal", ratio, False),
            "hexagonal",
            False,
        )
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    if family not in FAMILIES:
        raise ValueError(
            f"slip family: one of {sorted(FAMILIES) + sorted(HEXAGONAL)}, got {family!r}"
        )
    pairs = _family_vectors(*FAMILIES[family])
    n = np.array([p[0] for p in pairs], dtype=float)
    d = np.array([p[1] for p in pairs], dtype=float)
    return n / np.linalg.norm(n, axis=1)[:, None], d / np.linalg.norm(d, axis=1)[:, None]


def twin_systems(family: str, c_over_a: float | None = None):
    """Unit normals of the twin planes ``K_1`` and directions of their shear (N x 3 each,
    axes of the crystal) of a family of twins, and the shear of the twin: the one that
    leaves the direction ``eta_2`` of the second undistorted plane with its length,
    ``gamma_tw = 2 |eta_2 . l| / (eta_2 . n)``; the directions are those of the shear
    (the twin does not form in the other sense)."""
    if family not in TWINS:
        raise ValueError(f"twin family: one of {sorted(TWINS)}, got {family!r}")
    lattice, plane, direction, second = TWINS[family]
    ratio = _ratio(c_over_a, family) if lattice == "hexagonal" else None
    n = _vector(plane, lattice, ratio, True)
    d = _vector(direction, lattice, ratio, False)
    eta2 = _vector(second, lattice, ratio, False)
    shear = -2.0 * (eta2 @ d) / np.linalg.norm(d) / ((eta2 @ n) / np.linalg.norm(n))
    if abs(shear) < 1e-9:
        raise ValueError(f"twin family {family!r}: no shear at c_over_a = {c_over_a}")
    pairs = _variants(n, d if shear > 0 else -d, lattice, True)
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]), abs(float(shear))


def cleavage_systems(planes: str, c_over_a: float | None = None):
    """The systems of a family of cleavage planes: unit normals and directions (N x 3
    each, axes of the crystal), their kinds (:data:`OPENING`, whose direction is the
    normal, then the :data:`ACCOMMODATION` systems of the plane) and the plane of each
    (its index in the family)."""
    if planes not in CLEAVAGE:
        raise ValueError(f"cleavage planes: one of {sorted(CLEAVAGE)}, got {planes!r}")
    lattice, plane, directions = CLEAVAGE[planes]
    ratio = _ratio(c_over_a, planes) if lattice == "hexagonal" else None
    n0 = _vector(plane, lattice, ratio, True)
    n0 /= np.linalg.norm(n0)
    in_plane = [_vector(d, lattice, ratio, False) for d in directions]
    in_plane = [d / np.linalg.norm(d) for d in in_plane]
    seen: set = set()
    normals, slips, kinds, groups = [], [], [], []
    for r in _rotations(lattice):
        n = r @ n0
        if _key(n) in seen or _key(-n) in seen:
            continue
        seen.add(_key(n))
        group = len(seen) - 1
        normals.append(n)
        slips.append(n)
        kinds.append(OPENING)
        groups.append(group)
        for d in in_plane:
            normals.append(n)
            slips.append(r @ d)
            kinds.append(ACCOMMODATION)
            groups.append(group)
    return np.array(normals), np.array(slips), np.array(kinds), np.array(groups)


def interaction_types(normals, directions) -> np.ndarray:
    """Type of the interaction of each pair of octahedral systems (index in
    :data:`INTERACTIONS`): self, coplanar, Hirth lock (orthogonal Burgers vectors),
    collinear, and for the other pairs a glissile junction when the junction ``b_s +-
    b_r`` lies in one of the two planes, a Lomer lock otherwise."""
    n, b = np.asarray(normals, float), np.asarray(directions, float)
    count = len(n)
    out = np.zeros((count, count), dtype=np.int64)
    for s in range(count):
        for r in range(count):
            if s == r:
                continue
            if abs(abs(n[s] @ n[r]) - 1.0) < 1e-9:
                out[s, r] = 1
            elif abs(abs(b[s] @ b[r]) - 1.0) < 1e-9:
                out[s, r] = 3
            elif abs(b[s] @ b[r]) < 1e-9:
                out[s, r] = 2
            else:
                junction = b[s] - b[r] if b[s] @ b[r] > 0 else b[s] + b[r]
                glissile = abs(junction @ n[s]) < 1e-9 or abs(junction @ n[r]) < 1e-9
                out[s, r] = 4 if glissile else 5
    return out


def _family_params(name: str, spec: dict, defaults: dict) -> tuple[float, ...]:
    values = dict(defaults)
    for key, value in spec.items():
        key = _ALIASES.get(key, key)
        if key not in PARAMS:
            raise ValueError(f"crystal_plasticity.{name}: unknown parameter {key!r} ({PARAMS})")
        values[key] = float(value)
    missing = [k for k in PARAMS if values.get(k) is None]
    if missing:
        raise ValueError(f"crystal_plasticity.{name}: give {missing}")
    tau0, q, b, c, d, k, n = (values[p] for p in PARAMS)
    checks = [
        (tau0 >= 0, "tau0 >= 0"),
        (q >= 0, "Q >= 0"),
        (b > 0 or q == 0, "b > 0 with Q"),
        (c >= 0 and d >= 0, "C >= 0 and D >= 0"),
        (k > 0, "K > 0"),
        (n >= 1, "n >= 1"),
    ]
    for ok, text in checks:
        if not (ok and all(math.isfinite(v) for v in (tau0, q, b, c, d, k, n))):
            raise ValueError(f"crystal_plasticity.{name}: {text}")
    return (tau0, q, b, c, d, k, n)


#: Parameters of a family of cleavage planes: the normal stress that opens them, the rate
#: of their softening (per unit of opening strain), the ratio of the strength of their
#: accommodation systems to it, and the viscosity (``K``, ``n``) of their flow
CLEAVAGE_PARAMS = ("strength", "softening", "shear", "K", "n")


def _cleavage_params(name: str, spec: dict, defaults: dict) -> tuple[float, ...]:
    """The parameters of a family of cleavage planes (:data:`CLEAVAGE_PARAMS`) from a
    mapping: the ``softening``, or the fracture ``energy`` of the planes and the
    ``length`` of the elements that dissipate it (``softening = strength length /
    energy``); ``K`` and ``n`` are those of the crystal unless given."""
    where = f"crystal_plasticity.cleavage.{name}"
    values = {"shear": 1.0, "K": defaults.get("K"), "n": defaults.get("n")}
    allowed = (*CLEAVAGE_PARAMS, "energy", "length")
    for key, value in spec.items():
        key = _ALIASES.get(key, key)
        if key not in allowed:
            raise ValueError(f"{where}: unknown parameter {key!r} {allowed}")
        try:
            values[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{where}.{key}: a number, got {value!r}") from None
    if values.get("strength") is None:
        raise ValueError(f"{where}: give the strength (the normal stress that opens the planes)")
    banded = "energy" in values or "length" in values
    if banded == ("softening" in values) or (banded and not {"energy", "length"} <= set(values)):
        raise ValueError(
            f"{where}: give the softening, or the fracture energy of the planes and the "
            "length of the elements (softening = strength length / energy)"
        )
    if banded:
        if not (values["energy"] > 0 and values["length"] > 0):
            raise ValueError(f"{where}: energy > 0 and length > 0")
        values["softening"] = values["strength"] * values["length"] / values["energy"]
    missing = [k for k in ("K", "n") if values.get(k) is None]
    if missing:
        raise ValueError(f"{where}: give {missing} (those of the crystal, or of the planes)")
    out = tuple(values[k] for k in CLEAVAGE_PARAMS)
    strength, softening, shear, k, n = out
    checks = [
        (strength > 0, "strength > 0"),
        (softening > 0, "softening > 0"),
        (shear > 0, "shear > 0"),
        (k > 0, "K > 0"),
        (n >= 1, "n >= 1"),
    ]
    for ok, text in checks:
        if not (ok and all(math.isfinite(v) for v in out)):
            raise ValueError(f"{where}: {text}")
    return out


def _band_angle(ratio: float) -> float:
    """The phase ``omega x`` at the edge of the damage band of regularized cleavage
    planes, ``pi - atan sqrt(r / (1 - r))`` with ``r = |H| / chi_H`` the ratio of their
    softening modulus to their micromorphic one (:func:`microdamage_band`)."""
    return math.pi - math.atan(math.sqrt(ratio / (1.0 - ratio)))


def microdamage_band(strength: float, softening: float, modulus: float, length: float):
    """The damage band of cleavage planes regularized by their microdamage (Aslan &
    Forest 2009), when a plane breaks across a bar normal to it: its width and the
    fracture energy of the planes (per unit area).

    ``strength``: ``Y_0``; ``softening``: ``w``, their linear softening ``Y_0 (1 - w
    d)``, of modulus ``H = -w Y_0``; ``modulus``: the micromorphic modulus ``chi_H``, above
    ``w Y_0``; ``length``: ``ell = sqrt(A / chi_H)``, that of the equation of the microdamage,
    ``chi_d - ell^2 lap chi_d = d``. With ``r = w Y_0 / chi_H`` and ``theta = pi - atan sqrt(r /
    (1 - r))``, the damage is ``(Y_0 - sigma) / |H| + beta cos(omega x)`` over the width
    ``b = 2 theta / omega``, ``omega = sqrt(r / (1 - r)) / ell``; the bar softens linearly
    from ``Y_0`` to nothing as the band opens by ``Y_0 (b + 2 ell) / |H|``, and the work
    that breaks it is ``G_c = Y_0^2 (b + 2 ell) / (2 |H|)``."""
    softening_modulus = softening * strength
    if not 0.0 < softening_modulus < modulus:
        raise ValueError(
            "microdamage: a micromorphic modulus above softening x strength = "
            f"{softening_modulus:.4g} (the planes harden at a fixed microdamage)"
        )
    ratio = softening_modulus / modulus
    width = 2.0 * _band_angle(ratio) * length * math.sqrt((1.0 - ratio) / ratio)
    energy = strength * strength * (width + 2.0 * length) / (2.0 * softening_modulus)
    return width, energy


def microdamage_parameters(strength: float, energy: float, width: float, ratio: float = 1 / 3):
    """The parameters of cleavage planes regularized by their microdamage from the
    fracture ``energy`` of the planes (per unit area) and the ``width`` of their damage
    band (:func:`microdamage_band`, inverted): their ``softening`` ``w`` (linear: ``Y_0 (1
    - w d)``), their micromorphic ``modulus`` ``chi_H = w Y_0 / ratio`` and the ``length``
    of the equation of the microdamage (a mapping of the three). ``ratio``: that of the
    softening modulus to the micromorphic one, in ]0, 1[ (a third by default, as in the
    reference): the band is ``2 theta sqrt((1 - r) / r)`` lengths wide."""
    if not (strength > 0 and energy > 0 and width > 0 and 0.0 < ratio < 1.0):
        raise ValueError(
            "microdamage: strength > 0, energy > 0, width > 0 and a ratio of the softening "
            "modulus to the micromorphic one in ]0, 1["
        )
    length = width / (2.0 * _band_angle(ratio) * math.sqrt((1.0 - ratio) / ratio))
    softening_modulus = strength * strength * (width + 2.0 * length) / (2.0 * energy)
    return {
        "softening": softening_modulus / strength,
        "modulus": softening_modulus / ratio,
        "length": length,
    }


def _regularization(name: str, spec: dict, band) -> tuple[dict, tuple[float, float]]:
    """The parameters of a family of cleavage planes regularized by their microdamage
    (``band``, the mapping ``regularization`` of the planes): ``{modulus, length}``, with
    the linear softening of the family; or ``{width, ratio}``, the width of its damage
    band, with its fracture energy (:func:`microdamage_parameters`). Returns the
    parameters of the family, with its softening, and the modulus and the length."""
    where = f"crystal_plasticity.cleavage.{name}"
    if not isinstance(band, dict) or not (
        set(band) == {"modulus", "length"} or {"width"} <= set(band) <= {"width", "ratio"}
    ):
        raise ValueError(
            "crystal_plasticity.cleavage.regularization: {modulus, length} (with the "
            "softening of the planes), or {width, ratio} of their damage band (with their "
            "fracture energy)"
        )
    try:
        values = {key: float(value) for key, value in band.items()}
        if "width" not in values:
            return spec, (values["modulus"], values["length"])
        if "softening" in spec or "length" in spec or "energy" not in spec:
            raise ValueError(
                f"{where}: with the width of the damage band, the fracture energy of the "
                "planes (no softening, no length of elements)"
            )
        found = microdamage_parameters(
            float(spec.get("strength", math.nan)),
            float(spec["energy"]),
            values["width"],
            values.get("ratio", 1 / 3),
        )
    except (TypeError, ValueError) as exc:
        if str(exc).startswith(("crystal_plasticity", "microdamage")):
            raise ValueError(str(exc).replace("microdamage:", f"{where}.regularization:")) from None
        raise ValueError(f"{where}.regularization: numbers") from None
    spec = {k: v for k, v in spec.items() if k != "energy"}
    spec["softening"] = found["softening"]
    return spec, (found["modulus"], found["length"])


def _share(value, what: str) -> float:
    """The share of its resistance at which a system goes back to nothing: all of it
    (true, 1), none of the way (false, 0: it does not go back), or a number."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        share = float(value)
    except (TypeError, ValueError):
        share = math.nan
    if not (math.isfinite(share) and share >= 0.0):
        raise ValueError(
            f"crystal_plasticity.{what}: true, false, or the share of the resistance at "
            f"which the system goes back (a number of 0 or more), got {value!r}"
        )
    return share


def _named(value, what: str) -> list[tuple[str, dict]]:
    """The families of a configuration: a name, a list of names, or a mapping of the
    names to their own parameters."""
    if isinstance(value, str):
        return [(value, {})]
    if isinstance(value, dict):
        return [(str(name), dict(own or {})) for name, own in value.items()]
    try:
        return [(str(name), {}) for name in value]
    except TypeError:
        raise ValueError(f"crystal_plasticity.{what}: a name, a list or a mapping") from None


@dataclass(frozen=True)
class CrystalPlasticity:
    """Méric-Cailletaud crystal plasticity: families of slip systems with their
    parameters (``tau0, Q, b, C, D, K, n``, see :data:`PARAMS`) and the interaction
    matrix (``()``: uniform; ``(h_self, h_latent)``; the six coefficients of
    :data:`INTERACTIONS` for the octahedral family; or a full matrix). ``twins``:
    families of twins (:data:`TWINS`) with the same parameters, which shear in one sense
    until the fraction ``twin_saturation`` of the crystal has twinned; ``cleavage``:
    families of cleavage planes (:data:`CLEAVAGE`) with their parameters
    (:data:`CLEAVAGE_PARAMS`), which open under a normal stress; ``c_over_a``: the ratio
    of a hexagonal lattice (its families are those of :data:`HEXAGONAL`). The systems of
    a crystal are its slip systems, then its twins, then those of its cleavage planes.

    ``twin_detwinning`` and ``cleavage_closure``: whether the twins shear back and the
    planes close when the stress is reversed, or the share of their resistance at which
    they do (true: 1). ``twin_reorientation``: the twinned fraction at which the lattice
    of a point becomes that of its predominant twin; with ``twin_reorientation_growth``,
    the lattices are turned grain by grain (:func:`turn_twins`), the threshold of a grain
    growing by it times the share of its twinned volume that is turned.
    ``cleavage_coupling``: the rate at which the cumulated slip of the crystal lowers the
    strength of its cleavage planes (Aslan et al. 2011), the damage of the planes
    easing its slip in return. ``cleavage_regularization``: the micromorphic modulus and
    the length of the microdamage that regularizes the planes (Aslan & Forest 2009:
    :func:`microdamage_band`), whose softening is then linear."""

    families: tuple[tuple[str, tuple[float, ...]], ...]
    interaction: tuple = ()
    twins: tuple[tuple[str, tuple[float, ...]], ...] = ()
    cleavage: tuple[tuple[str, tuple[float, ...]], ...] = ()
    c_over_a: float | None = None
    twin_saturation: float = 1.0
    twin_reorientation: float | None = None
    twin_detwinning: bool | float = True
    cleavage_closure: bool | float = True
    cleavage_friction: float = 0.0
    twin_reorientation_growth: float | None = None
    cleavage_coupling: float = 0.0
    cleavage_regularization: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        fams = []
        for name, values in self.families:
            if name not in FAMILIES and name not in HEXAGONAL:
                raise ValueError(
                    f"crystal_plasticity: slip family {name!r}; "
                    f"{sorted(FAMILIES) + sorted(HEXAGONAL)}"
                )
            if len(values) != len(PARAMS):
                raise ValueError(f"crystal_plasticity.{name}: the parameters {PARAMS}")
            fams.append((name, _family_params(name, dict(zip(PARAMS, values, strict=True)), {})))
        twins = []
        for name, values in self.twins:
            if name not in TWINS:
                raise ValueError(f"crystal_plasticity: twin family {name!r}; {sorted(TWINS)}")
            if len(values) != len(PARAMS):
                raise ValueError(f"crystal_plasticity.twins.{name}: the parameters {PARAMS}")
            spec = dict(zip(PARAMS, values, strict=True))
            twins.append((name, _family_params(f"twins.{name}", spec, {})))
        planes = []
        for name, values in self.cleavage:
            if name not in CLEAVAGE:
                raise ValueError(
                    f"crystal_plasticity: cleavage planes {name!r}; {sorted(CLEAVAGE)}"
                )
            if len(values) != len(CLEAVAGE_PARAMS):
                raise ValueError(
                    f"crystal_plasticity.cleavage.{name}: the parameters {CLEAVAGE_PARAMS}"
                )
            spec = dict(zip(CLEAVAGE_PARAMS, values, strict=True))
            planes.append((name, _cleavage_params(name, spec, {})))
        if not fams and not twins and not planes:
            raise ValueError("crystal_plasticity: give at least one slip family")
        for what, group in (("slip family", fams), ("twin family", twins), ("cleavage", planes)):
            if len({f[0] for f in group}) != len(group):
                raise ValueError(f"crystal_plasticity: a {what} given twice")
        lattices = {lattice_of(name) for name, _ in (*fams, *twins, *planes)}
        if len(lattices) > 1:
            raise ValueError(
                "crystal_plasticity: the families of one lattice, cubic "
                f"({sorted(FAMILIES)}, twin111, twin112; cleavage: cube, dodecahedral, "
                f"octahedral) or hexagonal ({sorted(HEXAGONAL)}, twin10..; cleavage: basal, "
                "prismatic)"
            )
        ratio = self.c_over_a
        if lattices == {"hexagonal"}:
            ratio = _ratio(ratio, next(iter((*fams, *twins, *planes)))[0])
        elif ratio is not None:
            raise ValueError("crystal_plasticity: c_over_a is the ratio of a hexagonal lattice")
        object.__setattr__(self, "c_over_a", ratio)
        saturation = float(self.twin_saturation)
        if not 0.0 < saturation <= 1.0:
            raise ValueError("crystal_plasticity: twin_saturation, a fraction in ]0, 1]")
        object.__setattr__(self, "twin_saturation", saturation)
        turn = self.twin_reorientation
        growth = self.twin_reorientation_growth
        if turn is not None:
            turn = float(turn)
            if not twins:
                raise ValueError("crystal_plasticity: twin_reorientation is that of twins")
            if not 0.0 < turn < saturation:
                raise ValueError(
                    "crystal_plasticity: twin_reorientation, the twinned fraction at which the "
                    f"lattice is turned, in ]0, twin_saturation = {saturation:g}["
                )
            if growth is not None:
                growth = float(growth)
                if not (math.isfinite(growth) and growth >= 0.0):
                    raise ValueError(
                        "crystal_plasticity.twin_reorientation: a growth of 0 or more (the "
                        "threshold of a grain grows by it times the share of its twinned "
                        "volume that is turned)"
                    )
        elif growth is not None:
            raise ValueError(
                "crystal_plasticity: twin_reorientation_growth is that of a "
                "twin_reorientation threshold"
            )
        object.__setattr__(self, "twin_reorientation", turn)
        object.__setattr__(self, "twin_reorientation_growth", growth)
        untwins = _share(self.twin_detwinning, "twin_detwinning")
        object.__setattr__(self, "twin_detwinning", untwins)
        closes = _share(self.cleavage_closure, "cleavage.closure")
        object.__setattr__(self, "cleavage_closure", closes)
        friction = float(self.cleavage_friction)
        if not (math.isfinite(friction) and friction >= 0.0):
            raise ValueError("crystal_plasticity.cleavage: a friction of 0 or more")
        if friction > 0.0 and not self.cleavage_closure:
            raise ValueError(
                "crystal_plasticity.cleavage: the friction is that of planes that close "
                "(closure: true)"
            )
        object.__setattr__(self, "cleavage_friction", friction)
        coupling = float(self.cleavage_coupling)
        if not (math.isfinite(coupling) and coupling >= 0.0):
            raise ValueError("crystal_plasticity.cleavage: a coupling of 0 or more")
        if coupling > 0.0:
            if not planes:
                raise ValueError("crystal_plasticity: cleavage_coupling is that of cleavage planes")
            # what the planes take off the resistance of the slip once they are broken
            # through: coupling x sum over the planes of strength / softening
            energy = sum(
                (int(cleavage_systems(name, ratio)[3].max()) + 1) * values[0] / values[1]
                for name, values in planes
            )
            weakest = min((values[0] for _, values in fams), default=math.inf)
            if coupling * energy >= weakest:
                raise ValueError(
                    "crystal_plasticity.cleavage: the coupling of the planes with the slip "
                    f"would leave no resistance to the slip systems of {weakest:.4g} once the "
                    f"planes are broken (it falls by coupling x sum of strength / softening = "
                    f"{coupling * energy:.4g}): a coupling below {weakest / energy:.4g}"
                )
        object.__setattr__(self, "cleavage_coupling", coupling)
        regular = self.cleavage_regularization
        if regular is not None:
            if not planes:
                raise ValueError(
                    "crystal_plasticity: cleavage_regularization is that of cleavage planes"
                )
            try:
                modulus, length = (float(v) for v in regular)
            except (TypeError, ValueError):
                raise ValueError(
                    "crystal_plasticity.cleavage.regularization: the micromorphic modulus "
                    "and the length of the microdamage"
                ) from None
            if not (math.isfinite(modulus) and math.isfinite(length) and length > 0.0):
                raise ValueError("crystal_plasticity.cleavage.regularization: a length > 0")
            for name, values in planes:
                if not modulus > values[1] * values[0]:
                    raise ValueError(
                        f"crystal_plasticity.cleavage.{name}: a micromorphic modulus above "
                        f"softening x strength = {values[1] * values[0]:.4g} (the planes "
                        "harden at a fixed microdamage; they soften as it follows their damage)"
                    )
            regular = (modulus, length)
        object.__setattr__(self, "cleavage_regularization", regular)
        object.__setattr__(self, "families", tuple(fams))
        object.__setattr__(self, "twins", tuple(twins))
        object.__setattr__(self, "cleavage", tuple(planes))
        h = np.asarray(self.interaction, dtype=float)
        count = self.n_slip + self.n_twins
        if h.ndim == 2:
            if h.shape != (count, count):
                raise ValueError(f"crystal_plasticity.interaction: a {count} x {count} matrix")
            value = tuple(tuple(float(x) for x in row) for row in h)
        else:
            value = tuple(float(x) for x in h.ravel())
            if len(value) not in (0, 2, len(INTERACTIONS)):
                raise ValueError(
                    "crystal_plasticity.interaction: [h_self, h_latent], six coefficients "
                    f"{INTERACTIONS} (octahedral systems) or a full matrix"
                )
            alone = [f[0] for f in fams] == ["octahedral"] and not twins
            if len(value) == len(INTERACTIONS) and not alone:
                raise ValueError(
                    "crystal_plasticity.interaction: the six coefficients classify the "
                    "octahedral systems only; give [h_self, h_latent] or a full matrix"
                )
        if not np.all(np.isfinite(h)) or np.any(h < 0):
            raise ValueError("crystal_plasticity.interaction: non-negative coefficients")
        object.__setattr__(self, "interaction", value)

    @property
    def n_slip(self) -> int:
        """The slip systems."""
        return sum(len(slip_systems(name, self.c_over_a)[0]) for name, _ in self.families)

    @property
    def n_twins(self) -> int:
        """The twin systems."""
        return sum(len(twin_systems(name, self.c_over_a)[0]) for name, _ in self.twins)

    @property
    def n_cleavage(self) -> int:
        """The systems of the cleavage planes (openings and accommodations)."""
        return sum(len(cleavage_systems(name, self.c_over_a)[0]) for name, _ in self.cleavage)

    @property
    def n_systems(self) -> int:
        return self.n_slip + self.n_twins + self.n_cleavage

    @property
    def extended(self) -> bool:
        """Whether the crystal twins or cleaves (systems that are not slip systems)."""
        return bool(self.twins or self.cleavage)

    @property
    def reorients(self) -> bool:
        """Whether the twins of the crystal turn its lattice (``twin_reorientation``):
        its points then keep their rotation (:data:`CP_TURNED` internal variables more)."""
        return self.twin_reorientation is not None

    @property
    def regularized(self) -> bool:
        """Whether the cleavage planes are regularized by their microdamage
        (``cleavage_regularization``): their points then keep it
        (:attr:`micro_width` internal variables more), and the solvers of the loading
        paths solve its equation."""
        return self.cleavage_regularization is not None

    @property
    def n_planes(self) -> int:
        """The cleavage planes of the crystal."""
        ratio = self.c_over_a
        return sum(int(cleavage_systems(name, ratio)[3].max()) + 1 for name, _ in self.cleavage)

    @property
    def micro_width(self) -> int:
        """The internal variables of the microdamage of a point (:func:`state_width`):
        the microdamage, and the mark of each plane (0, or 1 + the damage that it had
        when it was broken through); none for planes that are not regularized."""
        return 1 + self.n_planes if self.regularized else 0

    @property
    def by_grain(self) -> bool:
        """Whether the lattices are turned grain by grain, after each step
        (:func:`turn_twins`: ``twin_reorientation_growth``), not point by point."""
        return self.twin_reorientation_growth is not None

    def systems(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit normals and slip directions of all the systems (axes of the crystal): the
        direction of the opening of a cleavage plane is its normal."""
        ratio = self.c_over_a
        parts = [slip_systems(name, ratio) for name, _ in self.families]
        parts += [twin_systems(name, ratio)[:2] for name, _ in self.twins]
        parts += [cleavage_systems(name, ratio)[:2] for name, _ in self.cleavage]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def params_array(self) -> np.ndarray:
        """Parameters of each system (N x 7, see :data:`PARAMS`): of a cleavage system,
        its strength (``tau0``) and its viscosity (``K``, ``n``)."""
        ratio = self.c_over_a
        rows = []
        for name, values in self.families:
            rows.extend([values] * len(slip_systems(name, ratio)[0]))
        for name, values in self.twins:
            rows.extend([values] * len(twin_systems(name, ratio)[0]))
        for name, (strength, _, shear, k, n) in self.cleavage:
            for kind in cleavage_systems(name, ratio)[2]:
                scale = 1.0 if kind == OPENING else shear
                rows.append((scale * strength, 0.0, 0.0, 0.0, 0.0, k, n))
        return np.array(rows, dtype=float).reshape(-1, len(PARAMS))

    def kinds_array(self) -> np.ndarray:
        """What each system is, beside its parameters (N x 9): its kind (:data:`SLIP`,
        :data:`TWIN`, :data:`OPENING`, :data:`ACCOMMODATION`), the rate of its softening
        (cleavage), its share of the saturation per unit of shear, ``1 / (gamma_tw
        F_sat)`` (twins), its cleavage plane (-1 for the others), the share of its
        resistance at which it goes back to nothing (a twin that untwins, the opening of
        a plane that closes; 0: it does not) or the friction of its plane under pressure
        (an accommodation system), its share per unit of shear of the fraction at which
        the lattice is turned, ``1 / (gamma_tw F_turn)`` (twins that turn it; -1 where the
        lattices are turned grain by grain), the opening system of its plane (an
        accommodation system; -1 for the others), its family (a twin; -1 for the others),
        its twinned fraction per unit of shear, ``1 / gamma_tw`` (a twin), the rate at
        which the slip of the crystal lowers its strength and the micromorphic modulus of
        its regularization (the systems of the cleavage planes: ``cleavage_coupling``,
        ``cleavage_regularization``)."""
        ratio = self.c_over_a
        out = np.zeros((self.n_systems, 11))
        out[:, 3] = -1.0
        out[:, 6:8] = -1.0
        row = self.n_slip
        for family, (name, _) in enumerate(self.twins):
            count = len(twin_systems(name, ratio)[0])
            shear = twin_systems(name, ratio)[2]
            out[row : row + count, 0] = TWIN
            out[row : row + count, 2] = 1.0 / (shear * self.twin_saturation)
            out[row : row + count, 4] = self.twin_detwinning
            out[row : row + count, 7] = family
            out[row : row + count, 8] = 1.0 / shear
            if self.by_grain:
                out[row : row + count, 5] = -1.0
            elif self.reorients:
                out[row : row + count, 5] = 1.0 / (shear * self.twin_reorientation)
            row += count
        plane = 0
        for name, (_, softening, *_rest) in self.cleavage:
            _, _, kinds, groups = cleavage_systems(name, ratio)
            out[row : row + len(kinds), 0] = kinds
            out[row : row + len(kinds), 1] = softening
            out[row : row + len(kinds), 3] = plane + groups
            opening = kinds == OPENING
            out[row : row + len(kinds), 4] = np.where(
                opening, self.cleavage_closure, self.cleavage_friction
            )
            # the opening system of the plane of each accommodation system
            first = row + np.flatnonzero(opening)
            out[row : row + len(kinds), 6] = np.where(opening, -1.0, first[groups])
            out[row : row + len(kinds), 9] = self.cleavage_coupling
            if self.regularized:
                out[row : row + len(kinds), 10] = self.cleavage_regularization[0]
            row += len(kinds)
            plane += int(groups.max()) + 1
        return out

    def interaction_matrix(self) -> np.ndarray:
        """The interaction of the slip systems and of the twins (none with the systems of
        the cleavage planes)."""
        count = self.n_slip + self.n_twins
        h = self.interaction
        if not h:
            block = np.ones((count, count))
        elif isinstance(h[0], tuple):
            block = np.array(h, dtype=float)
        elif len(h) == 2:
            block = np.full((count, count), h[1])
            np.fill_diagonal(block, h[0])
        else:
            normals, directions = self.systems()
            block = np.asarray(h)[interaction_types(normals[:count], directions[:count])]
        out = np.zeros((self.n_systems, self.n_systems))
        out[:count, :count] = block
        return out

    @property
    def lattice(self) -> str:
        """The lattice of the families of the crystal, ``"cubic"`` or ``"hexagonal"``."""
        return lattice_of(next(iter((*self.families, *self.twins, *self.cleavage)))[0])

    @property
    def field_names(self) -> tuple[str, ...]:
        """The fields of the crystal beside those of every behaviour (among
        :data:`FIELDS`): those of its twins (the lattices that they turn with
        ``twin_reorientation``, and their disorientation) and of its cleavage planes."""
        twins = ("twin_fraction",) if self.twins else ()
        if self.reorients:
            twins += ("twin_reoriented", "twin_misorientation")
        planes = ("cleavage_opening", "cleavage_damage") if self.cleavage else ()
        if self.regularized:
            planes += ("cleavage_microdamage",)
        return twins + planes

    def check_softening(self, stiffness, force: bool = False) -> None:
        """Refuse cleavage planes that soften faster than the crystal unloads
        (``stiffness``: its 6 x 6 matrix in its axes): the opening ``delta`` of a point
        whose strain is held releases the stress ``(n (x) n) : C : (n (x) n) delta``,
        and its strength must fall more slowly, ``w Y_0 < (n (x) n) : C : (n (x) n)``
        (likewise for the accommodation systems), for the state of the point to follow
        its strain. With the fracture energy ``G_c`` of the planes dissipated in elements
        of size ``h`` (``w = Y_0 h / G_c``): ``h < C_nn G_c / Y_0^2``. Not for planes
        regularized by their microdamage, which harden at a fixed microdamage (``force``:
        for them too, as a point alone softens with its damage)."""
        if not self.cleavage or (self.regularized and not force):
            return
        c = np.asarray(stiffness, dtype=float)
        normals, directions = self.systems()
        par, kinds = self.params_array(), self.kinds_array()
        names = []
        for name, _ in self.cleavage:
            names += [name] * len(cleavage_systems(name, self.c_over_a)[0])
        first = self.n_slip + self.n_twins
        for s in range(first, self.n_systems):
            m = 0.5 * (np.outer(directions[s], normals[s]) + np.outer(normals[s], directions[s]))
            voigt = np.array([m[0, 0], m[1, 1], m[2, 2], 2 * m[1, 2], 2 * m[2, 0], 2 * m[0, 1]])
            modulus = float(voigt @ c @ voigt)
            if kinds[s, 1] * par[s, 0] >= modulus:
                what = "opening" if kinds[s, 0] == OPENING else "accommodation"
                raise ValueError(
                    f"crystal_plasticity.cleavage.{names[s - first]}: the {what} of the planes "
                    f"softens faster (softening x strength = {kinds[s, 1] * par[s, 0]:.4g}) than "
                    f"the crystal unloads (stiffness {modulus:.4g}): a softening below "
                    f"{modulus / par[s, 0]:.4g} (smaller elements for this fracture energy: "
                    "length < stiffness x energy / strength^2)"
                )

    def point_fields(
        self, slips, *, openings=None, turned=None, orientations=None, micro=None
    ) -> dict:
        """The fields of points from the cumulated slips of their systems (points x N):
        the ``twin_fraction`` (with twins), the ``cleavage_opening`` (the opening strain
        of the planes) and the ``cleavage_damage`` of the most damaged plane, the share of
        its strength that its damage has taken (with cleavage planes: ``1 - exp(-w d)``;
        for planes regularized by their microdamage, what the damage of the planes of
        the point and its microdamage have taken, ``w d - chi_H (d - chi_d) / Y_0``
        between 0 and 1, and 1 for a plane broken through); for these, the
        ``cleavage_microdamage`` of the points (``micro``, points x :attr:`micro_width`:
        the microdamage — of a point alone, a negative value: the damage of its planes —
        and the marks of the planes broken through).

        ``openings``: the slips of the systems now (points x N, the variables ``alpha``):
        the opening is then that of the planes now, less than what they have opened once
        they close. ``turned``: the variables of the lattices that their twins turn
        (points x :data:`CP_TURNED`): the twinned fraction of a turned lattice is the one
        that it had when it was turned, less the fraction of the twin that brings the
        first lattice back (detwinning) — the fraction of the shears of the twins, which
        a turn does not change; ``twin_reoriented`` is 1 where the lattice is that of a
        twin (its mean: the turned volume) and ``twin_misorientation`` its
        :func:`disorientation` from the lattice of the grain (degrees; with the
        ``orientations`` of the grains of the points, points x 3 x 3, their axes in those
        of the sample: the rotations are kept in the latter)."""
        slips = np.asarray(slips, dtype=float)
        kinds = self.kinds_array()
        out = {}
        if self.twins:
            shares = kinds[:, 8]  # the twinned fraction per unit of shear
            fraction = slips @ shares
            if turned is not None and self.reorients:
                turned = np.asarray(turned, dtype=float)
                moved = np.any(turned[:, :9] != 0.0, axis=1)
                back = np.zeros(len(slips))
                known = np.flatnonzero(moved & (turned[:, 9] > 0))
                undo = turned[known, 9].astype(np.int64) - 1
                back[known] = shares[undo] * slips[known, undo]
                # what it had, less what is undone, and the twins of the new lattice
                kept = np.maximum(turned[:, 10] + fraction - 2.0 * back, 0.0)
                fraction = np.where(moved, kept, fraction)
                angle = np.zeros(len(slips))
                if moved.any():
                    rotation = turned[moved, :9].reshape(-1, 3, 3) + np.eye(3)
                    if orientations is not None:  # into the axes of the crystal: g R g^T
                        g = np.asarray(orientations, dtype=float)[moved]
                        rotation = np.einsum("pij,pjk,plk->pil", g, rotation, g)
                    angle[moved] = disorientation(rotation, self.lattice)
                out["twin_fraction"] = fraction
                out["twin_reoriented"] = moved.astype(float)
                out["twin_misorientation"] = angle
            else:
                out["twin_fraction"] = fraction
        if self.cleavage:
            opened = slips if openings is None else np.asarray(openings, dtype=float)
            out["cleavage_opening"] = opened[:, kinds[:, 0] == OPENING].sum(axis=1)
            damage = np.zeros(slips.shape[0])
            planes = np.unique(kinds[kinds[:, 3] >= 0, 3]).astype(np.int64)
            own = np.stack([slips[:, kinds[:, 3] == p].sum(axis=1) for p in planes], axis=1)
            if self.regularized:
                # the damage of the planes of the point (that which a plane had when it was
                # broken through: what it opens since is a crack) and its microdamage
                marks = np.zeros_like(own)
                if micro is not None:
                    micro = np.asarray(micro, dtype=float).reshape(len(slips), -1)
                    marks = micro[:, 1 : 1 + planes.size]
                whole = np.where(marks > 0.5, marks - 1.0, own).sum(axis=1)
                follows = (
                    whole if micro is None else np.where(micro[:, 0] < 0.0, whole, micro[:, 0])
                )
                out["cleavage_microdamage"] = follows
                modulus = self.cleavage_regularization[0]
                strength = self.params_array()[:, 0]
            for k, plane in enumerate(planes):
                members = np.flatnonzero(kinds[:, 3] == plane)
                rate = kinds[members[0], 1]
                if self.regularized:
                    first = members[kinds[members, 0] == OPENING][0]
                    lost = rate * whole - modulus / strength[first] * (whole - follows)
                    lost = np.where(marks[:, k] > 0.5, 1.0, np.clip(lost, 0.0, 1.0))
                else:
                    lost = -np.expm1(-rate * own[:, k])
                damage = np.maximum(damage, lost)
            out["cleavage_damage"] = damage
        return out

    def state_fields(self, state, finite: bool = False, orientations=None) -> dict:
        """The fields of points (:meth:`point_fields`) from their internal variables
        (points x width, those of the kernels in small or ``finite`` strains)."""
        state = np.asarray(state, dtype=float)
        n = self.n_systems
        turned = None
        if self.reorients:
            first = state_width(n, finite)
            turned = state[:, first : first + CP_TURNED]
        micro = None
        if self.regularized:
            first = state_width(n, finite, self.reorients)
            micro = state[:, first : first + self.micro_width]
        return self.point_fields(
            state[:, CP_ALPHA + n : CP_ALPHA + 2 * n],
            openings=state[:, CP_ALPHA : CP_ALPHA + n],
            turned=turned,
            orientations=orientations,
            micro=micro,
        )

    def to_config(self) -> dict:
        families = {name: dict(zip(PARAMS, v, strict=True)) for name, v in self.families}
        out: dict = {"families": families}
        if self.c_over_a is not None:
            out["c_over_a"] = self.c_over_a
        if self.twins:
            out["twins"] = {name: dict(zip(PARAMS, v, strict=True)) for name, v in self.twins}
            if self.twin_saturation != 1.0:
                out["twin_saturation"] = self.twin_saturation
            if self.by_grain:
                out["twin_reorientation"] = {
                    "threshold": self.twin_reorientation,
                    "growth": self.twin_reorientation_growth,
                }
            elif self.reorients:
                out["twin_reorientation"] = self.twin_reorientation
            if self.twin_detwinning != 1.0:
                out["twin_detwinning"] = self.twin_detwinning or False
        if self.cleavage:
            planes = {n: dict(zip(CLEAVAGE_PARAMS, v, strict=True)) for n, v in self.cleavage}
            out["cleavage"] = {"planes": planes}
            if self.cleavage_closure != 1.0:
                out["cleavage"]["closure"] = self.cleavage_closure or False
            if self.cleavage_friction:
                out["cleavage"]["friction"] = self.cleavage_friction
            if self.cleavage_coupling:
                out["cleavage"]["coupling"] = self.cleavage_coupling
            if self.regularized:
                modulus, length = self.cleavage_regularization
                out["cleavage"]["regularization"] = {"modulus": modulus, "length": length}
        if self.interaction:
            h = self.interaction
            out["interaction"] = [list(r) for r in h] if isinstance(h[0], tuple) else list(h)
        return out

    @classmethod
    def from_config(cls, spec) -> CrystalPlasticity:
        """``{families: [octahedral, ...] or {octahedral: {tau0: ..}, ...}, tau0, Q, b, C, D,
        K, n, interaction}``: the parameters given at the top apply to every family, and
        those of a family override them. ``twins``: families of twins, given as the slip
        families (``twin_saturation``: the fraction at which they stop;
        ``twin_reorientation``: the fraction at which the lattice of a point becomes that
        of its predominant twin, none by default, or ``{threshold, growth}``: the
        lattices are turned grain by grain, the most twinned points first while the
        turned volume of the grain is below its twinned volume, past a threshold that
        grows by ``growth`` times their ratio; ``twin_detwinning: false``: the twins do
        not shear back, or the share of their resistance at which they do); ``cleavage``:
        ``{planes: basal, or a list, or {basal: {strength: ..}}, strength, softening (or
        energy and length), shear, K, n, closure, friction, coupling}`` (``closure:
        false``: the planes do not close, or the share of their strength at which they
        do; ``friction``: Coulomb's, of the planes under pressure; ``coupling``: the rate
        at which the cumulated slip lowers their strength; ``regularization``: the
        microdamage of the planes, ``{modulus, length}`` with their linear ``softening``,
        or ``{width, ratio}`` of their damage band with their fracture ``energy``:
        :func:`microdamage_parameters`); ``c_over_a``: the ratio
        of a hexagonal lattice. The slip families default to the octahedral one, to none
        with twins or cleavage planes."""
        if isinstance(spec, CrystalPlasticity):
            return spec
        if not isinstance(spec, dict):
            raise ValueError("crystal_plasticity: a mapping (families, parameters, interaction)")
        spec = dict(spec)
        twins = spec.pop("twins", None)
        cleavage = spec.pop("cleavage", None)
        none = twins is not None or cleavage is not None
        families = spec.pop("families", [] if none else ["octahedral"])
        interaction = spec.pop("interaction", ())
        ratio = spec.pop("c_over_a", None)
        saturation = spec.pop("twin_saturation", 1.0)
        turn = spec.pop("twin_reorientation", None)
        growth = None
        if isinstance(turn, dict):
            if "threshold" not in turn or set(turn) - {"threshold", "growth"}:
                raise ValueError(
                    "crystal_plasticity.twin_reorientation: a twinned fraction, or "
                    "{threshold, growth} (the lattices turned grain by grain: the threshold "
                    "of a grain grows with the share of its twinned volume that is turned)"
                )
            growth = turn.get("growth", 0.0)
            turn = turn["threshold"]
        untwins = _share(spec.pop("twin_detwinning", True), "twin_detwinning")
        defaults = dict(_DEFAULTS)
        for key, value in spec.items():
            key = _ALIASES.get(key, key)
            if key not in PARAMS:
                raise ValueError(
                    f"crystal_plasticity: unknown key {key!r}; parameters {PARAMS}, families, "
                    "interaction, twins, twin_saturation, twin_reorientation, "
                    "twin_detwinning, cleavage, c_over_a"
                )
            defaults[key] = float(value)
        fams = []
        for name, own in _named(families if families is not None else [], "families"):
            if name not in FAMILIES and name not in HEXAGONAL:
                raise ValueError(
                    f"crystal_plasticity: slip family {name!r}; "
                    f"{sorted(FAMILIES) + sorted(HEXAGONAL)}"
                )
            fams.append((name, _family_params(name, own, defaults)))
        twinned = []
        for name, own in _named(twins if twins is not None else [], "twins"):
            if name not in TWINS:
                raise ValueError(f"crystal_plasticity: twin family {name!r}; {sorted(TWINS)}")
            twinned.append((name, _family_params(f"twins.{name}", own, defaults)))
        planes = []
        closure, friction, coupling, regular = True, 0.0, 0.0, None
        if cleavage is not None:
            if not isinstance(cleavage, dict) or cleavage.get("planes") is None:
                raise ValueError(
                    "crystal_plasticity.cleavage: a mapping of its planes "
                    f"({sorted(CLEAVAGE)}) and of their parameters {CLEAVAGE_PARAMS}"
                )
            common = dict(cleavage)
            closure = _share(common.pop("closure", True), "cleavage.closure")
            try:
                friction = float(common.pop("friction", 0.0))
            except (TypeError, ValueError):
                raise ValueError("crystal_plasticity.cleavage.friction: a number") from None
            try:
                coupling = float(common.pop("coupling", 0.0))
            except (TypeError, ValueError):
                raise ValueError("crystal_plasticity.cleavage.coupling: a number") from None
            band = common.pop("regularization", None)
            for name, own in _named(common.pop("planes"), "cleavage.planes"):
                if name not in CLEAVAGE:
                    raise ValueError(
                        f"crystal_plasticity: cleavage planes {name!r}; {sorted(CLEAVAGE)}"
                    )
                merged = {**common, **own}
                if band is not None:
                    merged, found = _regularization(name, merged, band)
                    if regular is not None and not np.allclose(found, regular, rtol=1e-12):
                        raise ValueError(
                            "crystal_plasticity.cleavage.regularization: one microdamage for "
                            "the planes of a crystal; give its {modulus, length}, and the "
                            "softening of each family of planes"
                        )
                    regular = found
                planes.append((name, _cleavage_params(name, merged, defaults)))
        try:
            saturation = float(saturation)
            ratio = None if ratio is None else float(ratio)
            turn = None if turn is None else float(turn)
            growth = None if growth is None else float(growth)
        except (TypeError, ValueError):
            raise ValueError(
                "crystal_plasticity: c_over_a, twin_saturation and twin_reorientation are numbers"
            ) from None
        return cls(
            tuple(fams),
            interaction if interaction is not None else (),
            twins=tuple(twinned),
            cleavage=tuple(planes),
            c_over_a=ratio,
            twin_saturation=saturation,
            twin_reorientation=turn,
            twin_detwinning=untwins,
            cleavage_closure=closure,
            cleavage_friction=friction,
            twin_reorientation_growth=growth,
            cleavage_coupling=coupling,
            cleavage_regularization=regular,
        )


# -- integration kernels ---------------------------------------------------------------------
#
# Internal variables of a crystal point: the plastic strain (6, tensor components; the
# Green-Lagrange plastic strain in finite strains), the cumulated plastic strain p, then
# alpha (N) and v (N) and, in small strains, the mechanical strain (6, engineering), in
# finite strains F_p^-1 - I and the displacement gradient (9 each, row by row), at the end
# of the step: the start of the substeps of the next one; then, for a lattice that its
# twins turn, its rotation less the identity (9), the twin that brings it back (1) and
# the twinned fraction that the point had when it was turned (1); then, for cleavage
# planes regularized by their microdamage, the microdamage of the point (1) and the mark
# of each plane (1 each): 0, or once it is broken through 1 + the damage that it had.
#
# The slips of a step solve, for the active systems (those whose slip over the step
# exceeds _SLIP_FLOOR), the flow rule written as stresses,
#
#     R_s = sign_s (tau_s - x_s) - r_s - K (|dgamma_s| / dt)^(1/n) = 0,
#
# with ``dgamma_s = sign_s delta_s``, ``delta_s > 0``: Newton's method on the magnitudes,
# from below (each R_s is convex and decreasing in delta_s, so that the iteration does not
# overshoot, however large the elastic trial overstress), the other systems inactive;
# then the inactive systems whose overstress exceeds the floor join the active set, until
# it is stable. (The flow rule written as rates, ``dgamma = dt <f/K>^n``, stalls under
# large overstresses: its Jacobian spans tens of orders of magnitude.) When the iteration
# fails (many redundant systems under a large overstress), the step is divided into 4,
# then 16 and 64 substeps along which the strain varies linearly; the tangent is then
# that of the last substep.
#
# Twins and cleavage planes are systems of the same iteration (their kinds and the
# numbers of ``CP_KIND ... CP_FAMILY`` after the parameters of each system): a twin and
# the opening of a cleavage plane flow in one sense (their overstress is ``tau - x - r``,
# not ``|tau - x| - r``) and, unless told not to, back to nothing (at a share of their
# resistance, ``CP_CLOSE``): the second variable of a twin is then its net shear, and a
# system that goes back by all that it has over a step is held there (its slip fixed, out
# of the active set; released if the others leave its flow rule short of it). The
# resistance of a twin has the barrier ``_BARRIER tau0 (1 / phi - 1)`` with ``phi = 1 -
# sum_t sat_t v_t`` the share of the crystal left to twin (a twin
# does not flow below ``_PHI_FLOOR``, and slips that would exhaust it are refused: the line
# search shortens them); the strength of the systems of a cleavage plane is ``tau0 exp(-w d)``,
# ``d`` the sum of the cumulated slips of the systems of the plane. The cumulated plastic
# strain counts the slips and the twins, not the systems of the cleavage planes.
#
# Work arrays of a kernel (``work``): vectors (26, N): residual, resolved shear stresses,
# overstresses, signs, trial resolved shear stresses (small strains), trial slips, trial
# residual, slips, active flags, trial overstresses, the exponentials of the isotropic
# hardening, the softening of the cleavage planes, the systems held at nothing, the
# friction of the planes, the derivatives of the strength of the planes (with their
# damage, with that of all the planes, with the slip) and of what they take off the slip,
# the active systems on their way out, the systems of the planes broken through, the
# damage of the plane of each opening system, the slips below which a system does not
# start with the viscosity that they were computed for (3), the exponential of the
# damage of the plane of each opening system, and the systems whose resolved stresses
# are derived (finite strains);
# matrices (4, N, N): Jacobian, its
# factorization, d tau / d dgamma, m_s : C : m_r; right-hand sides (N, 9); 3 x 3 tensors
# (24, 3, 3); their derivatives with respect to each slip (3, N, 3, 3); Schmid tensors
# and C : m (2, N, 6); vectors (6, 9); indices of the active systems (N); the internal
# variables of two substeps (2, width); the share of the crystal left to twin, the
# microdamage of the point, the twin that brings a turned lattice back, the damage that
# the planes broken through had, the derivatives that the last evaluation of the slips
# had (0: none, 1: those of the active systems, 2: of all), the time step of the
# starting slips kept, and the material whose Schmid tensors are kept (its key + 1; 0:
# none) (7); the systems, the elasticity and the thermal strain of a turned lattice
# ((N, 6) and (3, 6, 6)); the indices of the systems that slip (small strains) and of
# those that have slipped, which harden the others (N each).
#
# What a point does not need is not computed, with the same results to the bit: a point
# whose systems all stay below their thresholds is evaluated once, without derivatives;
# the derivatives of the resolved stresses are those of the active systems, and of the
# others only where one of them starts; the correction that the line search accepts is
# evaluated with its derivatives, and is the next iterate.

#: Index of alpha among the internal variables
CP_ALPHA = 7
#: Columns of the parameters of a system after those of :data:`PARAMS`, which the kernels
#: read when the array has them (:meth:`CrystalPlasticity.kinds_array`): its kind, the
#: rate of its softening, its share of the saturation of the twins, its cleavage plane,
#: the share of its resistance at which it goes back to nothing (a twin that untwins, the
#: opening of a plane that closes; 0: it does not) or the friction of its plane under
#: pressure (an accommodation system), its share of the twinned fraction at which the
#: lattice is turned into that of the twin (negative: the lattices are turned grain by
#: grain, :func:`turn_twins`), the opening system of its plane (an accommodation system;
#: -1 for the others), its family (a twin; -1 for the others), the twinned fraction per
#: unit of its shear, ``1 / gamma_tw`` (a twin), the rate at which the slip of the crystal
#: lowers its strength, and the micromorphic modulus of the planes that their microdamage
#: regularizes (the systems of the cleavage planes; 0: a local softening)
CP_KIND, CP_SOFT, CP_SAT, CP_GROUP, CP_CLOSE, CP_TURN, CP_OPEN, CP_FAMILY, CP_SHEAR = range(7, 16)
CP_COUPLE, CP_MICRO = 16, 17
#: Width of the parameters of a system for the kernels
CP_WIDTH = 18
#: The internal variables of a lattice that its twins turn, after the others: its rotation
#: from that of the grain, less the identity (9), the twin that would bring it back (its
#: index + 1; 0 without one, -1 for a lattice turned more than once) and the twinned
#: fraction of the point when it was turned
CP_TURNED = 11
#: Share of the crystal left to twin below which its twins do not flow
_PHI_FLOOR = 1e-12
#: The barrier of the saturation of the twins: their resistance grows by this fraction of
#: ``tau0`` times ``F / (F_sat - F)``
_BARRIER = 1e-3
#: Slips below this magnitude (over a step) are not resolved
_SLIP_FLOOR = 1e-15
#: Residual of the local iteration, relative to the scale of the stresses
_LOCAL_TOL = 1e-10
_MAX_LOCAL = 100
_MAX_ROUNDS = 20
#: Largest relative decrease of a slip in one iteration
_SHRINK = 1e-3
#: Divisions of a step into 4**k substeps, k = 1 ... _SUBSTEP_LEVELS
_SUBSTEP_LEVELS = 3
#: The share of its strength that a regularized cleavage plane keeps once it is broken
#: through (its linear softening stops there: no threshold below nothing)
_BROKEN = 1e-6
#: Twins whose fractions are within this (relative) of the largest are as predominant: the
#: first of them turns the lattice (variants that the symmetry of a loading makes equal
#: differ by their round-off, from a point to the next)
_TIE = 1e-6


def state_width(n_systems: int, finite: bool = False, turns: bool = False, micro: int = 0) -> int:
    """Internal variables of a crystal point with ``n_systems`` slip systems; ``turns``:
    its twins turn its lattice (:data:`CP_TURNED` more); ``micro``: those of the
    microdamage that regularizes its cleavage planes, the last ones (the microdamage,
    then the mark of each plane, 1 + its damage when it was broken through:
    :attr:`CrystalPlasticity.micro_width`)."""
    width = CP_ALPHA + 2 * int(n_systems) + (18 if finite else 6)
    return width + (CP_TURNED if turns else 0) + int(micro)


@njit(cache=True)
def work_blocks(count, n_max):
    """``count`` sets of work arrays of the kernels for at most ``n_max`` slip systems:
    those of :func:`work_arrays` with one more axis, the first one (:func:`work_block`
    takes a set)."""
    n = max(n_max, 1)
    return (
        np.zeros((count, 26, n)),
        np.zeros((count, 4, n, n)),
        np.zeros((count, n, 9)),
        np.zeros((count, 24, 3, 3)),
        np.zeros((count, 3, n, 3, 3)),
        np.zeros((count, 2, n, 6)),
        np.zeros((count, 6, 9)),
        np.zeros((count, n), dtype=np.int64),
        np.zeros((count, 2, CP_ALPHA + 2 * n + 18 + CP_TURNED + 1 + n)),
        np.zeros((count, 7)),
        np.zeros((count, n, 6)),
        np.zeros((count, 3, 6, 6)),
        np.zeros((count, n), dtype=np.int64),
        np.zeros((count, n), dtype=np.int64),
    )


@njit(cache=True)
def work_block(blocks, k):
    """The set ``k`` of the work arrays of :func:`work_blocks` (views).

    The kernels take rows of their work arrays at each call. Rows of arrays that the
    compiled code manages count their references, atomically: a third to a half of the
    time of a point where the arrays are allocated in the loop over the points. The
    loops over the points (:func:`morphogen.homogenization.behaviours.update_points`,
    :func:`morphogen.homogenization.finite.update_points_finite`) allocate instead the
    sets of all their chunks of points before their parallel loop, which then holds them
    as plain memory, and take the set of each chunk."""
    return (
        blocks[0][k],
        blocks[1][k],
        blocks[2][k],
        blocks[3][k],
        blocks[4][k],
        blocks[5][k],
        blocks[6][k],
        blocks[7][k],
        blocks[8][k],
        blocks[9][k],
        blocks[10][k],
        blocks[11][k],
        blocks[12][k],
        blocks[13][k],
    )


@njit(cache=True)
def work_arrays(n_max):
    """Work arrays of the kernels for at most ``n_max`` slip systems."""
    return work_block(work_blocks(1, n_max), 0)


@njit(cache=True, inline="always")
def _ddot6(a, b):
    """a : b of two symmetric tensors given by their Voigt components."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + 2.0 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5])


@njit(cache=True)
def _tau_small(ns, dg, tau_tr, amat, tau, slipping):
    """Resolved shear stresses in small strains: ``tau = tau_tr - A dgamma``, summed over
    the systems that slip (``slipping``: work array of their indices). Their derivatives
    with respect to the slips, ``-A``, are kept with the Schmid tensors
    (:func:`_schmid_small`)."""
    count = 0
    for r in range(ns):
        if dg[r] != 0.0:
            slipping[count] = r
            count += 1
    for s in range(ns):
        t = tau_tr[s]
        for j in range(count):
            r = slipping[j]
            t -= amat[s, r] * dg[r]
        tau[s] = t


@njit(cache=True)
def _tau_finite(ns, dg, sys, fe_tr, dm, eth, tau, dtau, with_jac, w3, wg, s6, act, every,
                rows):  # fmt: skip
    """Resolved shear stresses in finite strains: ``tau_s = l_s . M . n_s``, the Mandel
    stress ``M = C_e S_e`` of ``F_e = F_e^tr (I - G + G^2 / 2)``, ``G = sum dgamma_r
    l_r (x) n_r``, ``S_e = C : (E_e - E_th)``. Leaves in ``w3`` G (0), I - G + G^2/2 (1),
    F_e (2), C_e (3), S_e (4) and, with the Jacobian, the derivatives of the second one,
    of C_e and of S_e with respect to the slips in ``wg``: those of the active systems
    (``act``), or of ``every`` system. The derivatives of the resolved stresses: of the
    systems of ``rows`` with respect to the slips of the active ones, and of each of the
    others with respect to its own slip (``every``: what a system that starts needs)."""
    gam = w3[0]
    g = w3[1]
    fe = w3[2]
    ce = w3[3]
    se = w3[4]
    mm = w3[5]
    t1 = w3[6]
    t2 = w3[7]
    ee = w3[8]
    for i in range(3):
        for j in range(3):
            gam[i, j] = 0.0
    for r in range(ns):
        d = dg[r]
        if d != 0.0:
            for i in range(3):
                for j in range(3):
                    gam[i, j] += d * sys[r, i] * sys[r, 3 + j]
    matmul3(gam, gam, t1)
    for i in range(3):
        for j in range(3):
            g[i, j] = (1.0 if i == j else 0.0) - gam[i, j] + 0.5 * t1[i, j]
    matmul3(fe_tr, g, fe)
    tmatmul3(fe, fe, ce)
    for i in range(3):
        for j in range(3):
            ee[i, j] = 0.5 * (ce[i, j] - (1.0 if i == j else 0.0)) - eth[i, j]
    elastic_stress(dm, ee, se, s6[0], s6[1])
    matmul3(ce, se, mm)
    for s in range(ns):
        tau[s] = project(sys[s, 0:3], sys[s, 3:6], mm)
    if not with_jac:
        return
    av = s6[2]
    for r in range(ns):
        if not every and act[r] < 0.5:
            continue
        dgr = wg[0, r]
        dce = wg[1, r]
        dsr = wg[2, r]
        lr = sys[r, 0:3]
        nr = sys[r, 3:6]
        for i in range(3):
            av[i] = gam[0, i] * nr[0] + gam[1, i] * nr[1] + gam[2, i] * nr[2]
            av[3 + i] = gam[i, 0] * lr[0] + gam[i, 1] * lr[1] + gam[i, 2] * lr[2]
        for i in range(3):
            for j in range(3):
                dgr[i, j] = -lr[i] * nr[j] + 0.5 * (lr[i] * av[j] + av[3 + i] * nr[j])
        matmul3(fe_tr, dgr, t1)
        tmatmul3(t1, fe, t2)
        for i in range(3):
            for j in range(3):
                dce[i, j] = t2[i, j] + t2[j, i]
                ee[i, j] = 0.5 * dce[i, j]
        elastic_stress(dm, ee, dsr, s6[0], s6[1])
        matmul3(dce, se, t1)
        matmul3(ce, dsr, t2)
        for i in range(3):
            for j in range(3):
                t1[i, j] += t2[i, j]
        if act[r] > 0.5:
            for s in range(ns):
                if rows[s] > 0.5:
                    dtau[s, r] = project(sys[s, 0:3], sys[s, 3:6], t1)
        else:
            dtau[r, r] = project(sys[r, 0:3], sys[r, 3:6], t1)


@njit(cache=True, inline="always")
def _one_sense(par, s):
    """Whether a system flows in one sense only: a twin that does not untwin, the
    opening of a cleavage plane that does not close."""
    kind = par[s, CP_KIND]
    return (kind == 1.0 or kind == 2.0) and par[s, CP_CLOSE] == 0.0


@njit(cache=True, inline="always")
def _returns(par, s):
    """Whether a system flows in one sense and back to nothing, no further: a twin that
    untwins, the opening of a cleavage plane that closes."""
    kind = par[s, CP_KIND]
    return (kind == 1.0 or kind == 2.0) and par[s, CP_CLOSE] > 0.0


@njit(cache=True, inline="always")
def _net(par, s):
    """Whether the second variable of a system is its net slip, not the cumulated one:
    a twin that untwins (its shear is the fraction that has twinned)."""
    return par[s, CP_KIND] == 1.0 and par[s, CP_CLOSE] > 0.0


@njit(cache=True, inline="always")
def _room(par, s, alpha0, v0):
    """How far a system that returns may go back over a step: the shear of a twin, the
    opening of a plane, at its start."""
    return v0[s] if par[s, CP_KIND] == 1.0 else alpha0[s]


@njit(cache=True)
def _cp_eval(finite, par, h, ns, dt, alpha0, v0, dg, act, sgn, sys, dm, fe_tr, eth, work, res, fv,
             with_jac):  # fmt: skip
    """Resolved shear stresses at the slips ``dg``, the overstresses ``fv = |tau - x| - r`` of
    all the systems (``tau - x - r`` for those that flow in one sense; the larger of the
    two senses for a twin that untwins; what would bring it back further for a system
    held at nothing), the residual ``res`` of the active ones (``act``, signs ``sgn``)
    and, if ``with_jac``, its Jacobian with respect to the magnitudes of their slips
    (1, or true; 2: with the derivatives of the resolved stresses with respect to the
    slips of the inactive systems too, which one that starts needs).
    Returns the largest |R_s|: infinite when the slips of the active twins exhaust the
    crystal."""
    wv = work[0]
    wm = work[1]
    tau = wv[1]
    dtau = wm[2]
    ex = wv[10]  # exp(-b v_r) of the systems r for the b of the last family
    soft = wv[11]  # exp(-w d) of the cleavage plane of each system
    bound = wv[12]  # the systems held at nothing (a crack closed, a twin undone)
    press = wv[13]  # the friction of the plane of an accommodation system, if pressed
    b_ex = 0.0
    have = False
    hardening = work[13]  # the systems that have slipped (for the ``b`` of ``ex``): those
    count = 0  # that harden the others
    extended = par.shape[1] > CP_KIND
    if finite:
        rows = wv[25]  # the systems whose resolved stresses are derived: the active ones,
        if with_jac > 0:  # and the opening of the planes whose pressure holds one of them
            for s in range(ns):
                rows[s] = act[s]
            if extended:
                for s in range(ns):
                    if act[s] > 0.5 and par[s, CP_KIND] == 3.0 and par[s, CP_CLOSE] > 0.0:
                        rows[int(par[s, CP_OPEN])] = 1.0
        _tau_finite(ns, dg, sys, fe_tr, dm, eth, tau, dtau, with_jac > 0, work[3], work[4],
                    work[6], act, with_jac > 1, rows)  # fmt: skip
    else:
        _tau_small(ns, dg, wv[4], wm[3], tau, work[12])
    phi = 1.0  # the share of the crystal left to twin
    back = int(work[9][2]) - 1  # the twin that brings a turned lattice back (-1: none)
    coupled = 0.0  # the rate at which the slip lowers the strength of the cleavage planes
    released = 0.0  # what the damage of the planes takes off the resistance of the slip
    dreleased = 0.0  # its derivative with respect to the cumulated slip
    # the derivatives of the share ``soft`` of its strength that a system of a plane has:
    # with the damage of its plane (less), with that of all the planes, with the cumulated
    # slip (less); and of ``released`` with the damage of the plane of each system
    slope = wv[14]  # (the planes that their microdamage regularizes: none of their own)
    lift = wv[15]
    drop = wv[16]
    gain = wv[17]
    gone = wv[19]  # of the systems of the regularized planes broken through: 1 + their damage
    if extended:
        fall = 1.0  # of the strength of the planes with the cumulated slip
        stiff = 0.0  # the micromorphic modulus of the planes that their microdamage regularizes
        for s in range(ns):
            coupled = max(coupled, par[s, CP_COUPLE])
            stiff = max(stiff, par[s, CP_MICRO])
        if coupled > 0.0:
            slipped = 0.0
            for r in range(ns):
                if par[r, CP_KIND] == 0.0:
                    slipped += v0[r] + abs(dg[r])
            fall = math.exp(-coupled * slipped)
        # the microdamage of the point (negative: a point alone, whose microdamage is its
        # damage) and the damage of all its planes
        micro = work[9][1]
        total = 0.0
        if stiff > 0.0:
            total = work[9][3]  # of the planes broken through: the damage that they had
            for r in range(ns):
                if par[r, CP_SOFT] > 0.0 and gone[r] < 0.5:
                    total += v0[r] + abs(dg[r])
        capped = 0.0  # strength / softening of the regularized planes broken through
        planes = wv[20]  # the damage of the plane of each opening system
        keeps = wv[24]  # exp(-w d) of this plane, where it softens so
        for s in range(ns):
            if par[s, CP_SOFT] > 0.0 and par[s, CP_KIND] == 2.0:
                acc = 0.0
                for r in range(ns):
                    if par[r, CP_GROUP] == par[s, CP_GROUP]:
                        acc += v0[r] + abs(dg[r])
                planes[s] = acc
                if par[s, CP_MICRO] <= 0.0:  # (a plane that has not opened: no exponential)
                    keeps[s] = 1.0 if acc == 0.0 else math.exp(-par[s, CP_SOFT] * acc)
        for s in range(ns):
            soft[s] = 1.0
            slope[s] = 0.0
            lift[s] = 0.0
            drop[s] = 0.0
            gain[s] = 0.0
            w = par[s, CP_SOFT]
            if w > 0.0:
                opening = par[s, CP_KIND] == 2.0
                o = s if opening else int(par[s, CP_OPEN])
                acc = planes[o]
                y0 = par[o, 0]  # the strength of the plane
                if par[s, CP_MICRO] > 0.0:
                    # regularized: a linear softening with the damage of all the planes of
                    # the point (one threshold for them, as in Aslan & Forest 2009: the
                    # planes of a point harden alike at a fixed microdamage, none against
                    # the others), down to nothing, and the micromorphic term of this
                    # damage less the microdamage
                    value = fall - w * total
                    rise = -w
                    if micro >= 0.0:
                        value += stiff * (total - micro) / y0
                        rise += stiff / y0
                    if gone[s] > 0.5:
                        # broken through: no strength left, whatever it opens (a crack),
                        # and its damage is that which it had
                        soft[s] = _BROKEN
                        acc = gone[s] - 1.0
                    elif value > _BROKEN:
                        soft[s] = value
                        lift[s] = rise
                        drop[s] = coupled * fall
                    else:
                        soft[s] = _BROKEN
                    if coupled > 0.0:
                        # the energy of the plane that the slip releases: Y_0 d, up to the
                        # damage that breaks it
                        if acc < fall / w:
                            if gone[s] < 0.5:
                                gain[s] = coupled * fall * y0
                            if opening:
                                released += y0 * acc
                        elif opening:
                            released += y0 * fall / w
                            capped += y0 / w
                else:
                    # (the exponential of its plane, for the systems that soften alike)
                    if w == par[o, CP_SOFT] and par[o, CP_MICRO] <= 0.0:
                        kept = keeps[o]
                    else:
                        kept = math.exp(-w * acc)
                    soft[s] = kept * fall
                    slope[s] = w * soft[s]
                    drop[s] = coupled * soft[s]
                    if coupled > 0.0:
                        gain[s] = coupled * y0 * soft[s]
                        if opening:
                            # the energy that the plane has dissipated, per unit of its strength
                            released -= y0 * math.expm1(-w * acc) / w
            if par[s, CP_SAT] > 0.0:
                phi -= par[s, CP_SAT] * (v0[s] + (dg[s] if _net(par, s) else abs(dg[s])))
        released *= coupled * fall
        dreleased = -coupled * released - (coupled * fall) ** 2 * capped
        work[9][0] = phi
    worst = 0.0
    for s in range(ns):
        tau0 = par[s, 0]
        q = par[s, 1]
        b = par[s, 2]
        c = par[s, 3]
        d = par[s, 4]
        k = par[s, 5]
        n = par[s, 6]
        mag = abs(dg[s])
        over = tau[s] - c * (alpha0[s] + dg[s]) / (1.0 + d * mag)
        iso = tau0
        wall = 0.0  # the barrier of the saturation of the twins, in the sense that twins
        one_sense = False
        returns = False
        eased = 1.0  # the share of its resistance at which a system goes back to nothing
        if extended:
            iso = tau0 * soft[s]
            if par[s, CP_KIND] == 0.0:
                iso -= released  # the slip of a crystal whose planes are damaged
            one_sense = _one_sense(par, s)
            returns = _returns(par, s)
            if returns:
                eased = par[s, CP_CLOSE]
            press[s] = 0.0
            if par[s, CP_KIND] == 3.0 and par[s, CP_CLOSE] > 0.0:
                # Coulomb friction: the pressure on the plane holds its lips together
                normal = tau[int(par[s, CP_OPEN])]
                if normal < 0.0:
                    press[s] = par[s, CP_CLOSE]
                    iso -= press[s] * normal
            if par[s, CP_SAT] > 0.0:
                # the lattice of a twin: it twins back, and not again on the family of its twin
                if back >= 0 and s != back and par[s, CP_FAMILY] == par[back, CP_FAMILY]:
                    if act[s] > 0.5:
                        return math.inf
                    fv[s] = -1.0
                    res[s] = 0.0
                    continue
                if phi <= _PHI_FLOOR:  # the crystal has twinned as far as it may
                    if act[s] > 0.5 and sgn[s] > 0.0:
                        return math.inf
                    wall = math.inf
                else:
                    wall = _BARRIER * tau0 * (1.0 / phi - 1.0)
        if q != 0.0:
            if not have or b != b_ex:  # the systems of a family share their b
                count = 0
                for r in range(ns):
                    turned = extended and _net(par, r)
                    slipped = v0[r] + (dg[r] if turned else abs(dg[r]))
                    if slipped == 0.0:  # (a system that has not slipped: no exponential)
                        ex[r] = 1.0
                    else:
                        ex[r] = math.exp(-b * slipped)
                        hardening[count] = r
                        count += 1
                b_ex = b
                have = True
            if count > 0:
                acc = 0.0
                for j in range(count):
                    r = hardening[j]
                    acc += h[s, r] * (1.0 - ex[r])
                iso += q * acc
        if extended and bound[s] > 0.5:
            fv[s] = -over - eased * iso
        elif one_sense:
            fv[s] = over - iso - wall
        elif returns:
            fv[s] = max(over - iso - wall, -over - eased * iso)
        else:
            fv[s] = abs(over) - iso
        if act[s] > 0.5:
            if sgn[s] > 0.0:
                res[s] = over - iso - wall - k * (mag / dt) ** (1.0 / n)
            else:
                res[s] = -over - eased * iso - k * (mag / dt) ** (1.0 / n)
            worst = max(worst, abs(res[s]))
        else:
            res[s] = 0.0
    if not math.isfinite(worst):
        return math.inf
    work[9][4] = 1.0 * with_jac  # the derivatives that this evaluation has
    if with_jac > 0:
        jac = wm[0]
        for s in range(ns):
            if act[s] < 0.5:
                continue
            tau0 = par[s, 0]
            q = par[s, 1]
            b = par[s, 2]
            c = par[s, 3]
            d = par[s, 4]
            k = par[s, 5]
            n = par[s, 6]
            mag = abs(dg[s])
            w = 0.0
            twin = False
            rub = 0.0
            o = 0
            eased = 1.0  # the share of the resistance of a system on its way back
            if extended:
                w = par[s, CP_SOFT]
                twin = par[s, CP_SAT] > 0.0 and sgn[s] > 0.0
                rub = press[s]
                o = int(par[s, CP_OPEN]) if rub > 0.0 else 0
                if sgn[s] < 0.0 and _returns(par, s):
                    eased = par[s, CP_CLOSE]
            if q != 0.0 and (not have or b != b_ex):
                for r in range(ns):
                    turned = extended and _net(par, r)
                    slipped = v0[r] + (dg[r] if turned else abs(dg[r]))
                    ex[r] = 1.0 if slipped == 0.0 else math.exp(-b * slipped)
                b_ex = b
                have = True
            for r in range(ns):
                if act[r] < 0.5:
                    continue
                v = sgn[s] * sgn[r] * dtau[s, r]
                # the slip that hardens and saturates: cumulated, or the net one of a twin
                way = sgn[r] if extended and _net(par, r) else 1.0
                if q != 0.0:
                    v -= eased * q * h[s, r] * b * ex[r] * way
                if extended:
                    if w > 0.0:  # a system of a cleavage plane: its strength
                        if par[r, CP_GROUP] == par[s, CP_GROUP]:
                            v += eased * tau0 * slope[s]  # softens with the damage of its plane
                        if par[r, CP_SOFT] > 0.0 and gone[r] < 0.5:
                            v -= eased * tau0 * lift[s]  # the micromorphic term: all the planes
                        elif par[r, CP_KIND] == 0.0:
                            v += eased * tau0 * drop[s]  # the slip lowers it
                    elif coupled > 0.0 and par[s, CP_KIND] == 0.0:
                        # a slip system: what the damage of the planes takes off its resistance
                        if par[r, CP_KIND] == 0.0:
                            v += dreleased
                        elif par[r, CP_SOFT] > 0.0:
                            v += gain[r]
                    if twin and par[r, CP_SAT] > 0.0:
                        v -= _BARRIER * tau0 * par[r, CP_SAT] / (phi * phi) * way
                    if rub > 0.0:  # the pressure of the plane changes with the slips
                        v += rub * sgn[r] * dtau[o, r]
                if r == s:
                    v -= c * (1.0 - d * sgn[s] * alpha0[s]) / (1.0 + d * mag) ** 2
                    v -= k / (n * dt) * (mag / dt) ** (1.0 / n - 1.0)
                jac[s, r] = v
    return worst


@njit(cache=True)
def _active_indices(act, ns, idx):
    na = 0
    for s in range(ns):
        if act[s] > 0.5:
            idx[na] = s
            na += 1
    return na


@njit(cache=True)
def _active_newton(finite, par, h, ns, dt, alpha0, v0, sys, dm, fe_tr, eth, work, tol, first):
    """Newton's method on the magnitudes of the slips of the active systems, from those in
    ``dg``, with a backtracking line search; a system brought down to the floor by a
    negative residual leaves the active set. ``first``: the largest residual of these
    slips where the caller has just evaluated them (with the Jacobian if any system is
    active), negative otherwise. Returns 0 (the state of the slips evaluated, with the
    Jacobian of its active systems), or 1 on failure."""
    wv = work[0]
    wm = work[1]
    wr = work[2]
    idx = work[7]
    res = wv[0]
    fv = wv[2]
    sgn = wv[3]
    trial = wv[5]
    res_t = wv[6]
    dg = wv[7]
    act = wv[8]
    fv_t = wv[9]
    bound = wv[12]
    leave = wv[18]  # the active systems on their way out
    jac = wm[0]
    lu = wm[1]
    extended = par.shape[1] > CP_KIND
    for _ in range(_MAX_LOCAL):
        if first >= 0.0:
            worst = first
            first = -1.0
        else:
            worst = _cp_eval(finite, par, h, ns, dt, alpha0, v0, dg, act, sgn, sys, dm, fe_tr,
                             eth, work, res, fv, 1)  # fmt: skip
        if not math.isfinite(worst):
            return 1
        if worst <= tol:
            return 0
        if extended:
            # a crack that closes (a twin that untwins) as far as it may, and would go
            # further: held at nothing
            held = False
            for s in range(ns):
                if act[s] > 0.5 and sgn[s] < 0.0 and res[s] > 0.0 and _returns(par, s):
                    room = _room(par, s, alpha0, v0)
                    if abs(dg[s]) >= room * (1.0 - 1e-12):
                        act[s] = 0.0
                        bound[s] = 1.0
                        dg[s] = -room
                        held = True
            if held:
                continue
        dropped = False
        for s in range(ns):
            if act[s] > 0.5 and abs(dg[s]) <= _SLIP_FLOOR and res[s] < 0.0:
                act[s] = 0.0
                dg[s] = 0.0
                dropped = True
        if dropped:
            continue
        na = _active_indices(act, ns, idx)
        for a in range(na):
            for c in range(na):
                lu[a, c] = jac[idx[a], idx[c]]
            wr[a, 0] = -res[idx[a]]
        if gauss_solve(lu, wr, na, 1) != 0:
            return 1
        if extended:
            # a step that would bring a crack (a twin) back beyond nothing: held there, and
            # the others solved again (it is released later if they leave it short of it)
            held = False
            for a in range(na):
                s = idx[a]
                if sgn[s] < 0.0 and _returns(par, s):
                    room = _room(par, s, alpha0, v0)
                    if abs(dg[s]) + wr[a, 0] >= room:
                        act[s] = 0.0
                        bound[s] = 1.0
                        dg[s] = -room
                        held = True
            if held:
                continue
        # a system that the others have relaxed below its threshold — with no slip of
        # its own and no viscous stress it would still be below it — and that the
        # correction would bring to nothing is on its way out of the active set: its slip
        # shrinks (by _SHRINK at a time) and no step brings its residual down. The
        # correction of the others is solved again without it (it cannot go where the
        # whole correction sends it), it does not count in the decrease that the step
        # must bring as long as it stays below its flow rule, and it is out once the
        # others are at theirs (it comes back if they leave it an overstress). A system
        # above its threshold that its viscous stress alone holds back is not one of
        # them: its slip is smaller, not nil.
        some = False
        nf = 0
        free_worst = 0.0
        norm0 = 0.0
        for a in range(na):
            s = idx[a]
            leave[s] = 0.0
            mag = abs(dg[s])
            if res[s] < 0.0 and mag + wr[a, 0] <= 0.0:
                k = par[s, 5]
                n = par[s, 6]
                viscous = k * (mag / dt) ** (1.0 / n)
                # the residual without the viscous stress, brought back to no slip
                still = res[s] + viscous - (jac[s, s] + viscous / (n * mag)) * mag
                if still < 0.0:
                    leave[s] = 1.0
                    some = True
            if leave[s] < 0.5:
                nf += 1
                free_worst = max(free_worst, abs(res[s]))
                norm0 += res[s] ** 2
        if some:
            if free_worst <= tol:
                for a in range(na):
                    s = idx[a]
                    if leave[s] > 0.5:
                        act[s] = 0.0
                        dg[s] = 0.0
                continue
            # the correction of the others, for the slips of those that leave shrunk
            b = 0
            for a in range(na):
                if leave[idx[a]] > 0.5:
                    continue
                c = 0
                rhs = -res[idx[a]]
                for a2 in range(na):
                    s2 = idx[a2]
                    if leave[s2] < 0.5:
                        lu[b, c] = jac[idx[a], s2]
                        c += 1
                    else:
                        rhs += jac[idx[a], s2] * (1.0 - _SHRINK) * abs(dg[s2])
                wr[b, 0] = rhs
                b += 1
            if gauss_solve(lu, wr, nf, 1) != 0:
                return 1
            b = nf - 1
            for a in range(na - 1, -1, -1):
                s = idx[a]
                if leave[s] > 0.5:
                    wr[a, 0] = -(1.0 - _SHRINK) * abs(dg[s])
                else:
                    wr[a, 0] = wr[b, 0]
                    b -= 1
        t = 1.0
        accepted = False
        for _ls in range(30):
            for s in range(ns):
                trial[s] = dg[s]
            for a in range(na):
                s = idx[a]
                mag = max(abs(dg[s]) + t * wr[a, 0], _SHRINK * abs(dg[s]), _SLIP_FLOOR)
                if extended and sgn[s] < 0.0 and _returns(par, s):
                    # a crack closes by its opening at most, a twin untwins by its shear
                    mag = min(mag, _room(par, s, alpha0, v0))
                trial[s] = sgn[s] * mag
            # (the whole correction: with its derivatives — if it is accepted it is the
            # next iterate, which is then not evaluated again)
            lead = _ls == 0
            w = _cp_eval(finite, par, h, ns, dt, alpha0, v0, trial, act, sgn, sys, dm, fe_tr, eth,
                         work, res_t, fv_t, 1 if lead else 0)  # fmt: skip
            norm1 = 0.0
            for a in range(na):
                if leave[idx[a]] < 0.5 or res_t[idx[a]] > 0.0:
                    norm1 += res_t[idx[a]] ** 2
            if math.isfinite(w) and norm1 <= (1.0 - 1e-4 * t) * norm0:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            # stagnation at the level of rounding errors
            _cp_eval(finite, par, h, ns, dt, alpha0, v0, dg, act, sgn, sys, dm, fe_tr, eth, work,
                     res, fv, 1)  # fmt: skip
            if not some:
                return 0 if worst <= 1e3 * tol else 1
            if free_worst > 1e3 * tol:
                return 1
            for a in range(na):  # the others stagnate at their flow rules
                s = idx[a]
                if leave[s] > 0.5:
                    act[s] = 0.0
                    dg[s] = 0.0
            continue
        for s in range(ns):
            dg[s] = trial[s]
        if lead:  # evaluated with its derivatives: the residuals of the next iterate
            for s in range(ns):
                res[s] = res_t[s]
                fv[s] = fv_t[s]
            first = w
    return 1


@njit(cache=True)
def _released(par, ns, dt, alpha0, v0, fv, act, sgn, bound, tol):
    """The systems held at nothing (``bound``: a crack whose opening, a twin whose shear
    has gone back to zero over the step) after an iteration on the active set: one whose
    flow rule would bring it back less is released, and flows again. Returns whether the
    active set changed."""
    changed = False
    for s in range(ns):
        if bound[s] < 0.5:
            continue
        k = par[s, 5]
        n = par[s, 6]
        # its flow rule where it is held: -(tau - x) - r - K (what it went back / dt)^(1/n)
        rest = fv[s] - k * (_room(par, s, alpha0, v0) / dt) ** (1.0 / n)
        if rest < -tol:
            bound[s] = 0.0
            act[s] = 1.0
            sgn[s] = -1.0
            changed = True
    return changed


@njit(cache=True)
def _cp_solve(finite, par, h, ns, dt, alpha0, v0, sys, dm, fe_tr, eth, work):
    """Slips of the step (``work[0][7]``, holding the initial guess) by the active set
    strategy. Returns 0, or 1 on failure."""
    wv = work[0]
    wm = work[1]
    res = wv[0]
    tau = wv[1]
    fv = wv[2]
    sgn = wv[3]
    dg = wv[7]
    act = wv[8]
    bound = wv[12]  # the openings of the cracks held closed: their slips are fixed
    extended = par.shape[1] > CP_KIND
    twins = 0
    active = False
    for s in range(ns):
        bound[s] = 0.0
        one_sense = extended and _one_sense(par, s)
        if extended and par[s, CP_SAT] > 0.0:
            twins += 1
        if extended and dg[s] < 0.0 and _returns(par, s) and -dg[s] >= _room(par, s, alpha0, v0):
            act[s] = 0.0  # a crack closes by its opening at most, a twin untwins by its shear
            dg[s] = 0.0
            room = _room(par, s, alpha0, v0)
            if room > _SLIP_FLOOR:
                dg[s] = -room
                bound[s] = 1.0
        elif abs(dg[s]) > _SLIP_FLOOR and not (one_sense and dg[s] < 0.0):
            act[s] = 1.0
            sgn[s] = 1.0 if dg[s] > 0.0 else -1.0
            active = True
        else:
            act[s] = 0.0
            dg[s] = 0.0
    # (the derivatives of the resolved stresses: only with active systems, or where one
    # starts — a point that stays elastic is evaluated once, without them)
    worst = _cp_eval(finite, par, h, ns, dt, alpha0, v0, dg, act, sgn, sys, dm, fe_tr, eth, work,
                     res, fv, 1 if active else 0)  # fmt: skip
    if not math.isfinite(worst):
        return 1
    scale = 0.0
    for s in range(ns):
        scale = max(scale, abs(tau[s]) + par[s, 0] + par[s, 5])
    tol = _LOCAL_TOL * scale
    first = worst  # the slips are evaluated: the iteration starts from there
    # the slips below which a system does not start: kept from a point to the next, for
    # the viscosity of each system and the time step
    floors = wv[21]
    floor_k = wv[22]
    floor_n = wv[23]
    if work[9][5] != dt:
        for s in range(ns):
            floor_k[s] = -1.0
        work[9][5] = dt
    for _ in range(_MAX_ROUNDS):
        if _active_newton(finite, par, h, ns, dt, alpha0, v0, sys, dm, fe_tr, eth, work, tol,
                          first):  # fmt: skip
            return 1
        first = -1.0
        changed = False
        if extended:
            changed = _released(par, ns, dt, alpha0, v0, fv, act, sgn, bound, tol)
        for s in range(ns):
            if act[s] > 0.5 or bound[s] > 0.5:
                continue
            c = par[s, 3]
            k = par[s, 5]
            n = par[s, 6]
            if floor_k[s] != k or floor_n[s] != n:
                floors[s] = k * (_SLIP_FLOOR / dt) ** (1.0 / n)
                floor_k[s] = k
                floor_n[s] = n
            if fv[s] <= floors[s] + tol:
                continue
            sign = 1.0 if tau[s] - c * alpha0[s] >= 0.0 else -1.0
            room = math.inf
            if extended and sign < 0.0 and _returns(par, s):
                room = _room(par, s, alpha0, v0)
                if room <= _SLIP_FLOOR:
                    continue  # a closed crack carries the pressure; no twin to undo
            if work[9][4] < 1.5:  # the derivatives with respect to its slip, now needed
                _cp_eval(finite, par, h, ns, dt, alpha0, v0, dg, act, sgn, sys, dm, fe_tr, eth,
                         work, res, fv, 2)  # fmt: skip
            # a new active system, from below: a slip that relaxes a tenth of its
            # overstress at most, with a viscous stress half of it at most
            act[s] = 1.0
            sgn[s] = sign
            relax = abs(wm[2][s, s]) + c
            d0 = min(dt * (0.5 * fv[s] / k) ** n, 0.1 * fv[s] / max(relax, 1e-300))
            if extended and par[s, CP_SAT] > 0.0 and sign > 0.0:
                # a twin: half of the share of the crystal left to those that may start
                d0 = min(d0, 0.5 * work[9][0] / (par[s, CP_SAT] * twins))
            dg[s] = sgn[s] * min(max(d0, _SLIP_FLOOR), 0.5 * room)
            changed = True
        if not changed:
            return 0
    return 1


@njit(cache=True)
def _initial_slips(old, new, par, ns, dg, guess, ratio, rot):
    """Initial slips of the local iteration: zero (``guess`` 0); those of the current
    iterate of the step (1: ``new`` holds it); those of the previous step times ``ratio``
    (2: ``new`` holds the state before ``old``). ``|dgamma| = dv`` and ``dgamma = alpha
    (1 + D |dgamma|) - alpha_n`` invert the update of the internal variables. None across
    a turn of the lattice (``rot``: where its rotation is kept, -1 without one): the slips
    of the one are not those of the other."""
    for s in range(ns):
        dg[s] = 0.0
    if guess == 0:
        return
    if rot >= 0:
        for r in range(CP_TURNED):
            if old[rot + r] != new[rot + r]:
                return
    extended = par.shape[1] > CP_KIND
    a = CP_ALPHA
    for s in range(ns):
        d = par[s, 4]
        if guess == 1:
            mag = new[a + ns + s] - old[a + ns + s]
            val = new[a + s] * (1.0 + d * mag) - old[a + s]
        else:
            mag = old[a + ns + s] - new[a + ns + s]
            val = ratio * (old[a + s] * (1.0 + d * mag) - new[a + s])
        if extended and _net(par, s):  # a twin that untwins: its net shear
            if guess == 1:
                val = new[a + ns + s] - old[a + ns + s]
            else:
                val = ratio * (old[a + ns + s] - new[a + ns + s])
        if math.isfinite(val):
            dg[s] = val


@njit(cache=True)
def _solve_slips(finite, par, h, ns, dt, old, new, sys, dm, fe_tr, eth, guess, ratio, work):
    """Slips of the step from the initial guess, again from zero if that fails."""
    dg = work[0][7]
    alpha0 = old[CP_ALPHA : CP_ALPHA + ns]
    v0 = old[CP_ALPHA + ns : CP_ALPHA + 2 * ns]
    rot = CP_ALPHA + 2 * ns + (18 if finite else 6) if _turns(par, ns) else -1
    _initial_slips(old, new, par, ns, dg, guess, ratio, rot)
    if _cp_solve(finite, par, h, ns, dt, alpha0, v0, sys, dm, fe_tr, eth, work) == 0:
        return 0
    if guess == 0:
        return 1
    _initial_slips(old, new, par, ns, dg, 0, 1.0, rot)
    return _cp_solve(finite, par, h, ns, dt, alpha0, v0, sys, dm, fe_tr, eth, work)


@njit(cache=True)
def _update_hardening(par, ns, dg, old, new):
    extended = par.shape[1] > CP_KIND
    for s in range(ns):
        d = par[s, 4]
        new[CP_ALPHA + s] = (old[CP_ALPHA + s] + dg[s]) / (1.0 + d * abs(dg[s]))
        new[CP_ALPHA + ns + s] = old[CP_ALPHA + ns + s] + abs(dg[s])
        if extended and _returns(par, s):
            if par[s, CP_KIND] == 1.0:  # the shear of a twin that untwins: the net one
                new[CP_ALPHA + ns + s] = old[CP_ALPHA + ns + s] + dg[s]
                if new[CP_ALPHA + ns + s] < _SLIP_FLOOR:
                    new[CP_ALPHA + ns + s] = 0.0
            elif new[CP_ALPHA + s] < _SLIP_FLOOR:
                new[CP_ALPHA + s] = 0.0  # the opening of a crack: nil when it is closed


@njit(cache=True)
def _regularized(par, ns):
    """Whether the cleavage planes of a crystal are regularized by their microdamage."""
    if par.shape[1] <= CP_MICRO:
        return False
    for s in range(ns):
        if par[s, CP_MICRO] > 0.0:
            return True
    return False


@njit(cache=True)
def _broken_planes(par, ns, old, slot, gone):
    """The systems of the regularized planes broken through (``gone``: the mark of
    their plane, 1 + the damage that it had then; 0 for the others; the marks at
    ``old[slot + 1:]``, after the microdamage, are those of the solver of the path,
    which breaks the planes as their microdamage settles). Returns the damage of these
    planes: what they open since is the opening of a crack, which would otherwise run
    ahead of the microdamage and give the planes their strength back."""
    kept = 0.0
    if par.shape[1] <= CP_MICRO:
        for s in range(ns):
            gone[s] = 0.0
        return kept
    for s in range(ns):
        gone[s] = 0.0
        if par[s, CP_MICRO] > 0.0:
            mark = old[slot + 1 + int(par[s, CP_GROUP])]
            if mark > 0.5:
                gone[s] = mark
                if par[s, CP_KIND] == 2.0:  # once for the plane: its opening system
                    kept += mark - 1.0
    return kept


@njit(cache=True)
def _turns(par, ns):
    """Whether the twins of a crystal turn its lattice."""
    if par.shape[1] <= CP_TURN:
        return False
    for s in range(ns):
        if par[s, CP_TURN] != 0.0:
            return True
    return False


@njit(cache=True)
def rotate_stiffness(dm, rot, out, bond, tmp):
    """The stiffness ``dm`` (6 x 6, stresses for engineering strains) of a lattice turned
    by the rotation ``rot`` (its vectors ``v`` become ``rot v``): ``out = K dm K^T``, with
    ``K`` (left in ``bond``) the matrix of the tensor components of a turned symmetric
    tensor, ``(rot a rot^T) = K a``."""
    for p in range(6):
        i = p if p < 3 else (p - 2) % 3
        j = p if p < 3 else (p - 1) % 3
        for q in range(6):
            k = q if q < 3 else (q - 2) % 3
            ll = q if q < 3 else (q - 1) % 3
            v = rot[i, k] * rot[j, ll]
            if q >= 3:
                v += rot[i, ll] * rot[j, k]
            bond[p, q] = v
    for p in range(6):
        for q in range(6):
            acc = 0.0
            for r in range(6):
                acc += bond[p, r] * dm[r, q]
            tmp[p, q] = acc
    for p in range(6):
        for q in range(6):
            acc = 0.0
            for r in range(6):
                acc += tmp[p, r] * bond[q, r]
            out[p, q] = acc


@njit(cache=True)
def _turned_lattice(old, rot, ns, sys, dm, eth, work):
    """The lattice of a point that its twins turned (the rotation at ``old[rot:]``, less
    the identity): fills ``work[10]`` with its systems, ``work[11][0]`` with its
    elasticity and ``work[11][2][0]`` with its thermal strain (tensor components), those
    of the grain turned. Returns False, and fills nothing, for a lattice that was not
    turned."""
    turned = False
    for r in range(9):
        if old[rot + r] != 0.0:
            turned = True
    if not turned:
        return False
    q = work[3][20]
    for i in range(3):
        for j in range(3):
            q[i, j] = old[rot + 3 * i + j] + (1.0 if i == j else 0.0)
    out = work[10]
    for s in range(ns):
        for i in range(3):
            out[s, i] = q[i, 0] * sys[s, 0] + q[i, 1] * sys[s, 1] + q[i, 2] * sys[s, 2]
            out[s, 3 + i] = q[i, 0] * sys[s, 3] + q[i, 1] * sys[s, 4] + q[i, 2] * sys[s, 5]
    mats = work[11]
    rotate_stiffness(dm, q, mats[0], mats[1], mats[2])
    turned_eth = mats[2][0]  # free once the stiffness is turned
    for p in range(6):
        acc = 0.0
        for r in range(6):
            acc += mats[1][p, r] * eth[r]
        turned_eth[p] = acc
    return True


@njit(cache=True)
def turn_lattice(par, ns, normal, new, rot, best):
    """Turn the lattice of a point (its internal variables ``new``, the rotation at
    ``new[rot:]``) into that of its twin ``best``, of plane ``normal`` (axes of the
    sample, in the lattice as it is): by half a turn about the normal, ``Q = 2 n (x) n -
    I``, a rotation that composes with those of the earlier twins. Its twins start again
    from nothing: in the new lattice, the system of that twin shears the other way and
    brings the first lattice back (detwinning), the other systems of its family being
    held (the conjugate twin of a twin of small shear deforms the lattice as the
    detwinning does, under the same resolved stress, and would otherwise share it). The
    point keeps the twinned fraction that it had (by the shears of its twins); when the
    lattice of the grain comes back, what is left of it is the shear of that twin, which
    goes on shearing back."""
    a = CP_ALPHA
    total = 0.0  # the fractions of the twins of the lattice as it is
    for s in range(ns):
        total += par[s, CP_SHEAR] * new[a + ns + s]
    before = new[rot + 10]  # the twinned fraction of the point when it was turned
    undo = int(new[rot + 9]) - 1  # the twin that brings the earlier lattice back
    undone = par[undo, CP_SHEAR] * new[a + ns + undo] if undo >= 0 else 0.0
    twinned = max(before + total - 2.0 * undone, 0.0)  # of the point, as it is turned
    n0, n1, n2 = normal[0], normal[1], normal[2]
    r0 = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            r0[i, j] = new[rot + 3 * i + j] + (1.0 if i == j else 0.0)
    off = 0.0
    for j in range(3):
        along = n0 * r0[0, j] + n1 * r0[1, j] + n2 * r0[2, j]
        for i in range(3):
            ni = n0 if i == 0 else (n1 if i == 1 else n2)
            value = 2.0 * ni * along - r0[i, j] - (1.0 if i == j else 0.0)
            new[rot + 3 * i + j] = value
            off = max(off, abs(value))
    for s in range(ns):
        if par[s, CP_SAT] > 0.0:
            new[a + s] = 0.0
            new[a + ns + s] = 0.0
    if off < 1e-9:  # the lattice of the grain again, exactly
        for r in range(CP_TURNED):
            new[rot + r] = 0.0
        if undo == best and before > undone:  # what is left of the twin that turned it
            new[a + ns + best] = (before - undone) / par[best, CP_SHEAR]
    elif undo == best:
        new[rot + 9] = -1.0  # a lattice turned more than once: no single twin undoes it
        new[rot + 10] = twinned
    else:
        new[rot + 9] = best + 1.0
        new[rot + 10] = twinned


@njit(cache=True)
def _turn_lattice(par, ns, sys, new, rot):
    """The twins of a point turn its lattice at the end of a step (``new``): when its
    twinned fraction has come to the threshold (``sum_t CP_TURN_t v_t >= 1``), the lattice
    becomes that of its predominant twin, the one of the largest fraction (the first of
    those within :data:`_TIE` of it; :func:`turn_lattice`). ``sys``: the systems of the
    lattice of the step (axes of the sample). Not where the lattices are turned grain by
    grain (a negative ``CP_TURN``: :func:`turn_twins`, after the step)."""
    a = CP_ALPHA
    total = 0.0
    most = 0.0
    for s in range(ns):
        if par[s, CP_TURN] < 0.0:
            return
        share = par[s, CP_TURN] * new[a + ns + s]
        total += share
        most = max(most, share)
    if total < 1.0 or most <= 0.0:
        return
    for s in range(ns):
        if par[s, CP_TURN] * new[a + ns + s] >= (1.0 - _TIE) * most:
            turn_lattice(par, ns, sys[s, 3:6], new, rot, s)
            return


def turn_twins(
    crystal, state, before, grains, systems, *, points=None, finite: bool = False, weights=None
) -> int:
    """The lattices that the twins of grains turn at the end of a step, grain by grain:
    the predominant twin reorientation of Tomé, Lebensohn & Kocks (1991), with the points
    of a grain in place of the grains of an aggregate. In each grain, of twinned volume
    ``V_acc`` (the mean of the twinned fractions of its points, by the shears of their
    twins) and turned volume ``V_eff`` (that of its points whose lattice is that of a
    twin), the points whose twinned fraction has grown over the step past the threshold

        F_T = min(1, A_1 + A_2 V_eff / V_acc)

    take the lattice of their predominant twin (:func:`turn_lattice`; the first of the
    twins within :data:`_TIE` of the largest fraction), the most twinned first, while the
    turned volume is below the twinned one. When the twins shear back, the lattices come
    back — the points whose twin has been undone the most first, by ``A_1`` at least,
    while the turned volume is above the twinned one — and those whose twin is undone
    altogether in any case.

    ``crystal``: a :class:`CrystalPlasticity` with ``twin_reorientation`` (``A_1``) and
    ``twin_reorientation_growth`` (``A_2``); ``state`` and ``before``: the internal
    variables of points at the end and at the start of the step (rows x width; ``state``
    is changed in place); ``points``: the rows of the points of the crystal (all by
    default), ``grains`` the grain of each of them, ``systems`` the systems of the
    lattices of the grains (grains x N x 6: direction and normal, axes of the sample) and
    ``weights`` their volumes (equal by default). Returns the number of lattices turned."""
    if not crystal.by_grain:
        return 0
    rows = np.arange(len(state)) if points is None else np.asarray(points, dtype=np.int64)
    if rows.size == 0:
        return 0
    n = crystal.n_systems
    par = np.zeros((n, CP_WIDTH))
    par[:, :CP_KIND] = crystal.params_array()
    par[:, CP_KIND:] = crystal.kinds_array()
    shares = par[:, CP_SHEAR]
    rot = state_width(n, finite)
    a = CP_ALPHA + n
    grains = np.asarray(grains)
    now = state[rows, a : a + n] * shares  # the fractions of the twins of the lattices
    was = before[rows, a : a + n] * shares
    total = now.sum(axis=1)
    turned = np.any(state[rows, rot : rot + 9] != 0.0, axis=1)
    undo = state[rows, rot + 9].astype(np.int64) - 1
    known = turned & (undo >= 0)
    index = np.arange(rows.size)
    back = np.where(known, now[index, np.maximum(undo, 0)], 0.0)
    kept = state[rows, rot + 10]
    twinned = np.where(turned, np.maximum(kept + total - 2.0 * back, 0.0), total)
    # the twins that grow: those of the grain, or the one that brings its lattice back
    grows = np.where(known, back > was[index, np.maximum(undo, 0)], total > was.sum(axis=1))
    w = np.ones(rows.size) if weights is None else np.abs(np.asarray(weights, dtype=float))
    first, growth = crystal.twin_reorientation, crystal.twin_reorientation_growth
    count = 0
    for g in np.unique(grains):
        pts = np.flatnonzero(grains == g)
        share = w[pts] / w[pts].sum()
        acc = float(share @ twinned[pts])
        eff = float(share[turned[pts]].sum())
        if acc > 0.0 and eff < acc:
            threshold = min(1.0, first + growth * eff / acc)
            ready = np.flatnonzero(~turned[pts] & grows[pts] & (total[pts] >= threshold))
            for k in ready[np.argsort(-total[pts[ready]], kind="stable")]:
                if eff >= acc:
                    break
                p = pts[k]
                best = int(np.flatnonzero(now[p] >= (1.0 - _TIE) * now[p].max())[0])
                normal = np.ascontiguousarray(systems[g, best, 3:6], dtype=float)
                turn_lattice(par, n, normal, state[rows[p]], rot, best)
                eff += share[k]
                count += 1
        ready = np.flatnonzero(known[pts] & grows[pts])
        if ready.size == 0:
            continue
        for k in ready[np.argsort(-back[pts[ready]], kind="stable")]:
            p = pts[k]
            gone = back[p] >= kept[p]  # the twin of the point is undone altogether
            if not gone and not (eff > acc and back[p] >= first):
                continue
            row = state[rows[p]]
            lattice = row[rot : rot + 9].reshape(3, 3) + np.eye(3)
            normal = np.ascontiguousarray(lattice @ systems[g, undo[p], 3:6])
            turn_lattice(par, n, normal, row, rot, int(undo[p]))
            eff -= share[k]
            count += 1
    return count


@njit(cache=True)
def _schmid_small(sys, ns, dm, m, cm, me, amat, dtau):
    """Schmid tensors ``m = sym(l (x) n)`` (tensor components), ``C : m``, the
    interaction of the systems through the elasticity, ``A_sr = m_s : C : m_r``, and the
    derivatives of the resolved stresses with respect to the slips, ``dtau = -A``."""
    for s in range(ns):
        l0, l1, l2 = sys[s, 0], sys[s, 1], sys[s, 2]
        n0, n1, n2 = sys[s, 3], sys[s, 4], sys[s, 5]
        m[s, 0] = l0 * n0
        m[s, 1] = l1 * n1
        m[s, 2] = l2 * n2
        m[s, 3] = 0.5 * (l1 * n2 + l2 * n1)
        m[s, 4] = 0.5 * (l2 * n0 + l0 * n2)
        m[s, 5] = 0.5 * (l0 * n1 + l1 * n0)
        for r in range(6):
            me[r] = m[s, r] if r < 3 else 2.0 * m[s, r]
        for r in range(6):
            acc = 0.0
            for c in range(6):
                acc += dm[r, c] * me[c]
            cm[s, r] = acc
    for s in range(ns):
        for r in range(ns):
            amat[s, r] = _ddot6(m[s], cm[r])
            dtau[s, r] = -amat[s, r]


@njit(cache=True)
def _small_step(sys, par, h, ns, dm, emech, dt, old, new, sig, guess, ratio, work):
    """Backward Euler step of a small-strain point from the internal variables ``old`` to
    the mechanical strain ``emech`` (engineering, without the thermal strain): the slips,
    the stress and ``new`` (but the strain). Returns 0, or 1 on failure."""
    wv = work[0]
    w3 = work[3]
    w6 = work[5]
    s6 = work[6]
    m = w6[0]
    cm = w6[1]
    tau_tr = wv[4]
    ee = s6[0]
    for r in range(6):
        ee[r] = emech[r] - old[r] * (1.0 if r < 3 else 2.0)
    for r in range(6):
        acc = 0.0
        for c in range(6):
            acc += dm[r, c] * ee[c]
        sig[r] = acc
    for s in range(ns):
        tau_tr[s] = _ddot6(m[s], sig)
    if _solve_slips(False, par, h, ns, dt, old, new, sys, dm, w3[0], w3[0], guess, ratio, work):
        return 1
    for r in range(old.shape[0]):
        new[r] = old[r]
    dg = wv[7]
    dep = s6[1]
    flow = s6[2]  # of the slips and of the twins: the plastic strain of p
    extended = par.shape[1] > CP_KIND
    for r in range(6):
        dep[r] = 0.0
        flow[r] = 0.0
    for s in range(ns):
        if dg[s] == 0.0:
            continue
        plastic = not extended or par[s, CP_KIND] < 1.5
        for r in range(6):
            dep[r] += dg[s] * m[s, r]
            sig[r] -= dg[s] * cm[s, r]
            if plastic:
                flow[r] += dg[s] * m[s, r]
    for r in range(6):
        new[r] = old[r] + dep[r]
    new[6] = old[6] + math.sqrt(max(2.0 / 3.0 * _ddot6(flow, flow), 0.0))
    _update_hardening(par, ns, dg, old, new)
    return 0


@njit(cache=True)
def cp_small(sys, par, h, ns, dm, eps, eth, dt, old, new, sig, tan, guess, ratio, work,
             sym=True, key=-1):  # fmt: skip
    """Implicit update of a Méric-Cailletaud point in small strains over a step ``dt``:
    strain ``eps`` (engineering), thermal strain ``eth`` (tensor components), slip
    systems ``sys`` (slip direction, plane normal) and elasticity ``dm`` in the axes of the
    sample. Fills the stress, the internal variables and the consistent tangent
    (symmetrized with ``sym``); returns 1 if the local iteration fails, even in
    substeps. ``key``: a number of the material of the point (its systems and its
    elasticity), 0 or more: the Schmid tensors of the last point integrated with these
    ``work`` arrays are kept for a point of the same key (-1: computed for each point)."""
    wv = work[0]
    wm = work[1]
    wr = work[2]
    w6 = work[5]
    s6 = work[6]
    m = w6[0]
    cm = w6[1]
    slot = CP_ALPHA + 2 * ns
    turns = _turns(par, ns)
    work[9][2] = 0.0
    work[9][1] = -1.0  # the microdamage of the point, after its other internal variables
    mslot = slot + 6 + (CP_TURNED if turns else 0)
    regular = _regularized(par, ns)
    if regular:
        work[9][1] = old[mslot]
    work[9][3] = _broken_planes(par, ns, old, mslot, wv[19])
    kept = key >= 0 and work[9][6] == key + 1.0
    if turns and _turned_lattice(old, slot + 6, ns, sys, dm, eth, work):
        sys = work[10]  # the lattice of a twin: its systems, elasticity, thermal strain
        dm = work[11][0]
        eth = work[11][2][0]
        work[9][2] = max(old[slot + 15], 0.0)  # the twin that brings it back
        kept = False
        key = -1
    if not kept:  # (those of the last point otherwise: the same material)
        _schmid_small(sys, ns, dm, m, cm, s6[3], wm[3], wm[2])
        work[9][6] = key + 1.0
    emech = s6[4]
    for r in range(6):
        emech[r] = eps[r] - eth[r] * (1.0 if r < 3 else 2.0)
    if _small_step(sys, par, h, ns, dm, emech, dt, old, new, sig, guess, ratio, work) != 0:
        bufs = work[8]
        width = min(old.shape[0], bufs.shape[1])
        ek = s6[5]
        done = False
        for level in range(_SUBSTEP_LEVELS):
            count = 4 ** (level + 1)
            cur = bufs[0, :width]
            nxt = bufs[1, :width]
            for r in range(width):
                cur[r] = old[r]
            ok = True
            for k in range(1, count + 1):
                for r in range(6):
                    ek[r] = old[slot + r] + k / count * (emech[r] - old[slot + r])
                if _small_step(sys, par, h, ns, dm, ek, dt / count, cur, nxt, sig,
                               2 if k > 1 else 0, 1.0, work):  # fmt: skip
                    ok = False
                    break
                cur, nxt = nxt, cur
            if ok:
                for r in range(width):
                    new[r] = cur[r]
                done = True
                break
        if not done:
            return 1
    for r in range(6):
        new[slot + r] = emech[r]
    if turns:  # the lattice of the next step: that of the predominant twin, past the threshold
        _turn_lattice(par, ns, sys, new, slot + 6)
    # consistent tangent (of the last substep): d delta_a / d eps = -J^-1 sign_a d tau_a /
    # d eps on the active systems, d tau / d eps = C : m
    sgn = wv[3]
    idx = work[7]
    for r in range(6):
        for c in range(6):
            tan[r, c] = dm[r, c]
    na = _active_indices(wv[8], ns, idx)
    if na == 0:
        return 0
    jac = wm[0]
    lu = wm[1]
    extended = par.shape[1] > CP_KIND
    press = wv[13]
    for a in range(na):
        s = idx[a]
        for b in range(na):
            lu[a, b] = jac[s, idx[b]]
        for j in range(6):
            wr[a, j] = -sgn[s] * cm[s, j]
        if extended and press[s] > 0.0:  # the friction follows the pressure of its plane
            o = int(par[s, CP_OPEN])
            for j in range(6):
                wr[a, j] -= press[s] * cm[o, j]
    if gauss_solve(lu, wr, na, 6) != 0:
        return 1
    for j in range(6):
        for r in range(6):
            acc = 0.0
            for a in range(na):
                s = idx[a]
                acc += sgn[s] * cm[s, r] * wr[a, j]
            tan[r, j] -= acc
    if sym:
        symmetrize(tan)
    return 0


@njit(cache=True)
def _finite_step(sys, par, h, ns, dm, hg, eth, dt, old, new, guess, ratio, work):
    """Backward Euler step of a finite-strain point from the internal variables ``old`` to
    the displacement gradient ``hg``: the slips and ``new`` (but the gradient). Leaves in
    the 3 x 3 work tensors F (9), F_p^-1 at the start (10) and the end (13) of the step and
    those of ``_tau_finite``. Returns 0, or 1 on failure."""
    w3 = work[3]
    f = w3[9]
    fpn = w3[10]
    fe_tr = w3[11]
    fpi = w3[13]
    t1 = w3[15]
    t2 = w3[16]
    off = CP_ALPHA + 2 * ns
    for i in range(3):
        for j in range(3):
            f[i, j] = (1.0 if i == j else 0.0) + hg[3 * i + j]
            fpn[i, j] = (1.0 if i == j else 0.0) + old[off + 3 * i + j]
    matmul3(f, fpn, fe_tr)
    if _solve_slips(True, par, h, ns, dt, old, new, sys, dm, fe_tr, eth, guess, ratio, work):
        return 1
    for r in range(old.shape[0]):
        new[r] = old[r]
    gam = w3[0]
    matmul3(fpn, w3[1], fpi)
    for i in range(3):
        for j in range(3):
            new[off + 3 * i + j] = fpi[i, j] - (1.0 if i == j else 0.0)
    inverse3(fpi, t1)
    tmatmul3(t1, t1, t2)
    new[0] = 0.5 * (t2[0, 0] - 1.0)
    new[1] = 0.5 * (t2[1, 1] - 1.0)
    new[2] = 0.5 * (t2[2, 2] - 1.0)
    new[3] = 0.5 * t2[1, 2]
    new[4] = 0.5 * t2[2, 0]
    new[5] = 0.5 * t2[0, 1]
    if par.shape[1] > CP_KIND:  # the slips and the twins: the plastic strain of p
        dg = work[0][7]
        for i in range(3):
            for j in range(3):
                t1[i, j] = 0.0
        for r in range(ns):
            if par[r, CP_KIND] < 1.5 and dg[r] != 0.0:
                for i in range(3):
                    for j in range(3):
                        t1[i, j] += dg[r] * sys[r, i] * sys[r, 3 + j]
        gam = t1
    acc = 0.0
    for i in range(3):
        for j in range(3):
            sym = 0.5 * (gam[i, j] + gam[j, i])
            acc += sym * sym
    new[6] = old[6] + math.sqrt(2.0 / 3.0 * acc)
    _update_hardening(par, ns, work[0][7], old, new)
    return 0


@njit(cache=True)
def cp_finite(sys, par, h, ns, dm, hg, eth6, dt, old, new, pk1, tan, guess, ratio, work):
    """Implicit update of a Méric-Cailletaud point in finite strains: displacement gradient
    ``hg`` (``F - I``, 9 components row by row), thermal strain ``eth6`` (tensor
    components, Green-Lagrange, of the lattice), slip systems ``sys`` and elasticity
    ``dm`` of the lattice in the intermediate configuration, axes of the sample. Fills
    the first Piola-Kirchhoff stress ``pk1``, the internal variables and the tangent
    ``tan = d P / d F`` (9 x 9); returns 1 if the local iteration fails, even in
    substeps."""
    wv = work[0]
    wm = work[1]
    wr = work[2]
    w3 = work[3]
    wg = work[4]
    s6 = work[6]
    eth = w3[12]
    hslot = CP_ALPHA + 2 * ns + 9
    turns = _turns(par, ns)
    work[9][2] = 0.0
    work[9][1] = -1.0  # the microdamage of the point, after its other internal variables
    mslot = hslot + 9 + (CP_TURNED if turns else 0)
    regular = _regularized(par, ns)
    if regular:
        work[9][1] = old[mslot]
    work[9][3] = _broken_planes(par, ns, old, mslot, wv[19])
    if turns and _turned_lattice(old, hslot + 9, ns, sys, dm, eth6, work):
        sys = work[10]  # the lattice of a twin: its systems, elasticity, thermal strain
        dm = work[11][0]
        eth6 = work[11][2][0]
        work[9][2] = max(old[hslot + 18], 0.0)  # the twin that brings it back
    stress_tensor(eth6, eth)
    if _finite_step(sys, par, h, ns, dm, hg, eth, dt, old, new, guess, ratio, work) != 0:
        bufs = work[8]
        width = min(old.shape[0], bufs.shape[1])
        hk = s6[5]
        done = False
        for level in range(_SUBSTEP_LEVELS):
            count = 4 ** (level + 1)
            cur = bufs[0, :width]
            nxt = bufs[1, :width]
            for r in range(width):
                cur[r] = old[r]
            ok = True
            for k in range(1, count + 1):
                for r in range(9):
                    hk[r] = old[hslot + r] + k / count * (hg[r] - old[hslot + r])
                if _finite_step(sys, par, h, ns, dm, hk, eth, dt / count, cur, nxt,
                                2 if k > 1 else 0, 1.0, work):  # fmt: skip
                    ok = False
                    break
                cur, nxt = nxt, cur
            if ok:
                for r in range(width):
                    new[r] = cur[r]
                done = True
                break
        if not done:
            return 1
    for r in range(9):
        new[hslot + r] = hg[r]
    if turns:  # the lattice of the next step: that of the predominant twin, past the threshold
        _turn_lattice(par, ns, sys, new, hslot + 9)
    # stress and tangent of the (last sub)step
    f = w3[9]
    fpn = w3[10]
    fpi = w3[13]
    s2 = w3[14]
    t1 = w3[15]
    t2 = w3[16]
    t3 = w3[17]
    t4 = w3[18]
    dpr = w3[19]
    fe = w3[2]
    ce = w3[3]
    se = w3[4]
    ee = w3[8]
    sgn = wv[3]
    idx = work[7]
    extended = par.shape[1] > CP_KIND
    na = _active_indices(wv[8], ns, idx)
    matmul3(fpi, se, t1)
    matmult3(t1, fpi, s2)
    matmul3(f, s2, t1)
    for i in range(3):
        for j in range(3):
            pk1[3 * i + j] = t1[i, j]
    # tangent at fixed slips, and d tau / d F
    for kl in range(9):
        k = kl // 3
        ll = kl % 3
        for i in range(3):
            for j in range(3):
                t1[i, j] = fpi[ll, i] * fe[k, j]
        for i in range(3):
            for j in range(3):
                t2[i, j] = t1[i, j] + t1[j, i]
                ee[i, j] = 0.5 * t2[i, j]
        elastic_stress(dm, ee, t3, s6[0], s6[1])
        matmul3(t2, se, t1)
        matmul3(ce, t3, t4)
        for i in range(3):
            for j in range(3):
                t1[i, j] += t4[i, j]
        for a in range(na):
            s = idx[a]
            wr[a, kl] = -sgn[s] * project(sys[s, 0:3], sys[s, 3:6], t1)
            if extended and wv[13][s] > 0.0:  # the friction follows the pressure of its plane
                o = int(par[s, CP_OPEN])
                wr[a, kl] -= wv[13][s] * project(sys[o, 0:3], sys[o, 3:6], t1)
        matmul3(fe, t3, t1)
        matmult3(t1, fpi, t2)
        for i in range(3):
            for j in range(3):
                v = t2[i, j]
                if i == k:
                    v += s2[ll, j]
                tan[3 * i + j, kl] = v
    if na > 0:
        jac = wm[0]
        lu = wm[1]
        for a in range(na):
            for b in range(na):
                lu[a, b] = jac[idx[a], idx[b]]
        if gauss_solve(lu, wr, na, 9) != 0:
            return 1
        matmul3(fe, se, t4)
        for a in range(na):
            r = idx[a]
            # d P / d dgamma_r, with d F_p^-1 = F_p^-1(n) dG_r
            matmul3(fpn, wg[0, r], t1)
            matmul3(f, t1, t2)
            matmul3(t2, se, t3)
            matmult3(t3, fpi, dpr)
            matmul3(fe, wg[2, r], t2)
            matmult3(t2, fpi, t3)
            for i in range(3):
                for j in range(3):
                    dpr[i, j] += t3[i, j]
            matmult3(t4, t1, t3)
            for i in range(3):
                for j in range(3):
                    dpr[i, j] += t3[i, j]
            for kl in range(9):
                x = sgn[r] * wr[a, kl]
                if x != 0.0:
                    for i in range(3):
                        for j in range(3):
                            tan[3 * i + j, kl] += dpr[i, j] * x
    return 0
