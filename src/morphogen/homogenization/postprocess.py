"""Post-processing of apparent properties: compliance, engineering constants and
anisotropy of stiffness tensors, macroscopic values of porous media, normalized and
specific values.

Stiffness and compliance. Every output of an elastic stiffness ``C`` gives its compliance
``S = C^-1`` beside it, in the same notation (:data:`NOTATION`): morphoGen's Voigt order
``11, 22, 33, 23, 31, 12`` with engineering shears (``11, 22, 12`` for plane problems,
:data:`PLANE_NOTATION`), so that ``S_11 = 1/E_1``, ``S_12 = -nu_12/E_1 = -nu_21/E_2`` and
``S_44 = 1/G_23``, ``S_55 = 1/G_31``, ``S_66 = 1/G_12`` for an orthotropic material in its
axes (:func:`engineering_constants`). TFEL (MFront) and Abaqus order their shears ``12,
13, 23``: a component ``C_44`` of morphoGen is their ``C_66``. The Mandel form of a
stiffness is ``W C W``, ``W = diag(1, 1, 1, √2, √2, √2)``, and of a compliance ``W^-1 S
W^-1``.

Macroscopic values of porous media. Meshes (or images) without their pores give the
macroscopic stress as the average over the whole volume element, the pores carrying
none: ``Sigma = phi <sigma>_s``, the average over the solid times the solid fraction
``phi`` (as in the homogenization notice of Z-set). The macroscopic strain is not the
average over the solid: it is the prescribed strain, or the macroscopic unknowns solved
for under stress or mixed loadings (:func:`morphogen.homogenization.core.superpose`), as
the degrees of freedom of a periodic element. Solutions hold both (``Solution.strain``,
``Solution.stress``) and the averages over each phase (``Solution.phases``).

Normalized values. For a property ``P`` (bulk, shear or Young's modulus, conductivity)
of a microstructure of solid fraction ``phi``, whose solid phases have the properties
``P_i`` and the fractions ``f_i``:

* relative: ``P / P_s``, with ``P_s = sum f_i P_i / phi`` the property of the solid (of
  its single phase for a porous medium);
* normalized: ``P / (phi P_s) = P / sum f_i P_i``, the property per unit of solid
  fraction, relative to the Voigt bound (rule of mixtures), which bounds it by 1;
* specific: ``P / rho`` with the effective density ``rho = sum f_i rho_i``.

Relative properties against the relative density ``phi`` are the coordinates of the
normalized Ashby charts (:mod:`morphogen.results`); Gibson & Ashby (1997) scalings read
``E / E_s ~ phi^n`` (``n = 1`` stretching-dominated, 2 bending-dominated foams).
"""

from __future__ import annotations

import numpy as np

from morphogen.homogenization.materials import TRANSPORT, Material

#: Mandel weights of the Voigt order (sqrt(2) on the shears)
_W = np.array([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])

#: morphoGen's Voigt order of the rows and columns of stiffnesses and compliances, in space
#: and in the plane
VOIGT_ORDER = ("11", "22", "33", "23", "31", "12")
PLANE_ORDER = ("11", "22", "12")
#: The notation of a stiffness and of its compliance, stated by the outputs that give them
NOTATION = (
    "Voigt, order 11 22 33 23 31 12 (TFEL and Abaqus: 11 22 33 12 13 23), engineering "
    "shears; S = C^-1 in the same notation"
)
PLANE_NOTATION = "Voigt, order 11 22 12, engineering shear; S = C^-1 in the same notation"


#: The smallest eigenvalue of a stiffness, relative to its largest, at or below which the
#: stiffness is singular to round-off (a direction without stiffness, as across layers of
#: void, at 1e-17 of the rest), or not positive definite: it has no compliance
_ROUND_OFF = 10.0 * np.finfo(float).eps


def compliance(stiffness) -> np.ndarray:
    """The compliance ``S = C^-1`` of a Voigt stiffness (6 x 6, order 11 22 33 23 31 12; 3
    x 3, order 11 22 12, for plane problems; engineering shears), in the same notation:
    ``S_11 = 1/E_1``, ``S_12 = -nu_12/E_1``, ``S_44 = 1/G_23`` (see the module). A
    stiffness singular to round-off (its smallest eigenvalue within 10 machine epsilons of
    its largest, or negative: a direction without stiffness, as across layers of void) has
    none: NaN, not the inverse of its round-off; an exactly singular one raises
    :class:`numpy.linalg.LinAlgError`; one only badly scaled keeps its inverse (a design
    stiff along its fibres, 3, whose plane is as stiff as its void, 1e-12: 1e-14)."""
    c = np.asarray(stiffness, dtype=float)
    if c.shape not in ((6, 6), (3, 3)):
        raise ValueError(f"stiffness: a 6 x 6 matrix (3 x 3 in the plane), got shape {c.shape}")
    s = np.linalg.inv(c)
    if np.all(np.isfinite(c)):
        w = np.linalg.eigvalsh(0.5 * (c + c.T))
        if not w[0] > _ROUND_OFF * w[-1]:
            return np.full_like(c, np.nan)
    return s


def notation(stiffness) -> str:
    """The notation of a stiffness and of its compliance (:data:`NOTATION`;
    :data:`PLANE_NOTATION` for a 3 x 3 one)."""
    return PLANE_NOTATION if np.shape(stiffness) == (3, 3) else NOTATION


def matrix_entries(matrix, prefix: str) -> dict[str, float]:
    """The entries of the upper triangle of a symmetric matrix as ``<prefix><i><j>``, the
    indices from 1 in the Voigt order (``C11`` ... ``C66``, ``S11`` ... ``S66``): the
    columns of the tables of results."""
    m = np.asarray(matrix, dtype=float)
    n = len(m)
    return {f"{prefix}{i + 1}{j + 1}": float(m[i, j]) for i in range(n) for j in range(i, n)}


def engineering_constants(stiffness) -> dict[str, float]:
    """Young's moduli ``E1, E2, E3``, Poisson's ratios ``nu_ij`` (contraction along
    ``j`` under a stress along ``i``: ``-S_ij / S_ii``) and shear moduli ``G23, G31,
    G12`` of a Voigt stiffness (order 11 22 33 23 31 12, engineering shears)."""
    s = np.linalg.inv(np.asarray(stiffness, dtype=float))
    out = {f"E{i + 1}": float(1.0 / s[i, i]) for i in range(3)}
    for i in range(3):
        for j in range(3):
            if i != j:
                out[f"nu{i + 1}{j + 1}"] = float(-s[i, j] / s[i, i])
    for k, name in zip((3, 4, 5), ("G23", "G31", "G12"), strict=True):
        out[name] = float(1.0 / s[k, k])
    return out


#: The Voigt index of the shear of each plane of the axes
_SHEAR = {(0, 1): 5, (1, 2): 3, (0, 2): 4}


def directional_modulus(stiffness, angles, *, plane=(0, 1), plane_strain: bool = True):
    """The Young's modulus ``1 / (n n : S : n n)`` along the directions ``n = cos(theta)
    e_a + sin(theta) e_b`` of ``angles`` (radians) in the plane of the axes ``(a, b)`` of a
    Voigt stiffness (6 x 6), under a uniaxial stress along ``n``: in plane strain (``S``
    the inverse of the in-plane block ``a, b, ab`` of the stiffness: prisms across the
    plane, as planar images one voxel thick) or in plane stress (``S`` the in-plane block
    of the compliance, the other stresses free). A plane stiffness (3 x 3, Voigt 11, 22,
    12) is taken in plane strain as it is. The ratio of its least to its largest value
    over the angles is the elastic anisotropy ``eta_E = E_min / E_max`` of Both et al.
    (2026)."""
    c = np.asarray(stiffness, dtype=float)
    a, b = sorted(int(v) for v in plane)
    if c.shape == (3, 3):
        s = np.linalg.inv(c)
    elif c.shape == (6, 6):
        if (a, b) not in _SHEAR:
            raise ValueError("plane: two distinct axes among 0, 1 and 2")
        k = [a, b, _SHEAR[(a, b)]]
        s = np.linalg.inv(c[np.ix_(k, k)]) if plane_strain else np.linalg.inv(c)[np.ix_(k, k)]
    else:
        raise ValueError("stiffness: a Voigt matrix of 6 x 6, or 3 x 3 in the plane")
    if tuple(int(v) for v in plane) == (b, a):  # the angles from the second axis given
        s = s[np.ix_([1, 0, 2], [1, 0, 2])]
    t = np.asarray(angles, dtype=float)
    v = np.stack([np.cos(t) ** 2, np.sin(t) ** 2, np.cos(t) * np.sin(t)], axis=-1)
    return 1.0 / np.einsum("...i,ij,...j->...", v, s, v)


def sphere_directions(count: int = 2000) -> np.ndarray:
    """``count`` unit vectors spread evenly over the half-sphere ``z > 0`` (a Fibonacci
    lattice): the directions of space up to their sign, as axes of moduli."""
    k = np.arange(int(count)) + 0.5
    z = k / int(count)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * k
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def spatial_modulus(stiffness, directions) -> np.ndarray:
    """The Young's modulus ``1 / (n n : S : n n)`` along unit vectors ``n`` of space
    (``(m, 3)``, normalized here) of a Voigt stiffness (6 x 6, engineering shears), under a
    uniaxial stress along ``n``: the plane-stress :func:`directional_modulus` in every
    plane. Its least and largest values over :func:`sphere_directions` give the elastic
    anisotropy ``eta_E = E_min / E_max`` of a 3D sample (Both et al. 2026)."""
    c = np.asarray(stiffness, dtype=float)
    if c.shape != (6, 6):
        raise ValueError("stiffness: a Voigt matrix of 6 x 6")
    s = np.linalg.inv(c)
    n = np.asarray(directions, dtype=float).reshape(-1, 3)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    v = np.column_stack([n[:, 0] ** 2, n[:, 1] ** 2, n[:, 2] ** 2,
                         n[:, 1] * n[:, 2], n[:, 2] * n[:, 0], n[:, 0] * n[:, 1]])  # fmt: skip
    return 1.0 / np.einsum("mi,ij,mj->m", v, s, v)


def anisotropy(stiffness) -> dict[str, float]:
    """Voigt and Reuss bulk and shear moduli of a Voigt stiffness, the universal
    anisotropy index ``A_U = 5 G_V / G_R + K_V / K_R - 6`` (Ranganathan & Ostoja-
    Starzewski 2008, zero for isotropy), the Zener ratio ``2 C44 / (C11 - C12)`` of its
    mean cubic part (1 for isotropy) and the relative distance to the nearest isotropic
    stiffness (its Voigt average, in the Mandel norm)."""
    c = np.asarray(stiffness, dtype=float)
    s = np.linalg.inv(c)
    a, b, d = np.trace(c[:3, :3]), c[0, 1] + c[0, 2] + c[1, 2], np.trace(c[3:, 3:])
    kv, gv = (a + 2.0 * b) / 9.0, (a - b + 3.0 * d) / 15.0
    a, b, d = np.trace(s[:3, :3]), s[0, 1] + s[0, 2] + s[1, 2], np.trace(s[3:, 3:])
    kr, gr = 1.0 / (a + 2.0 * b), 15.0 / (4.0 * a - 4.0 * b + 3.0 * d)
    c11 = np.trace(c[:3, :3]) / 3.0
    c12 = (c[0, 1] + c[0, 2] + c[1, 2]) / 3.0
    c44 = np.trace(c[3:, 3:]) / 3.0
    iso = np.zeros((6, 6))
    iso[:3, :3] = kv - 2.0 * gv / 3.0
    iso[[0, 1, 2], [0, 1, 2]] += 2.0 * gv
    iso[[3, 4, 5], [3, 4, 5]] = gv
    mandel = _W[:, None] * _W[None, :]
    return {
        "bulk_voigt": float(kv),
        "bulk_reuss": float(kr),
        "shear_voigt": float(gv),
        "shear_reuss": float(gr),
        "universal_anisotropy": float(5.0 * gv / gr + kv / kr - 6.0),
        "zener_ratio": float(2.0 * c44 / (c11 - c12)) if c11 != c12 else float("nan"),
        "isotropy_distance": float(np.linalg.norm((c - iso) * mandel) / np.linalg.norm(c * mandel)),
    }


def solid_fraction(fractions: dict[int, float], materials: dict[int, Material]) -> float:
    """Volume fraction of the phases that are not voids."""
    return float(sum(f for p, f in fractions.items() if not materials[p].void))


def effective_density(fractions: dict[int, float], materials: dict[int, Material]):
    """``sum f_i rho_i`` (exact), or None when a solid phase has no density."""
    solid = [p for p in fractions if not materials[p].void]
    if not solid or any(materials[p].density is None for p in solid):
        return None
    return float(sum(fractions[p] * materials[p].density for p in solid))


def _property(material: Material, name: str) -> float | None:
    """Property of a phase: exact for isotropic materials, the Hill average of a crystal
    (uniform orientations)."""
    if name in _TRANSPORT:  # the coefficient of a transport phenomenon
        physics = _TRANSPORT[name]
        if not material.has(physics):
            return None
        voigt = material.mean_conductivity("voigt", physics)
        if voigt == 0.0:  # an insulator
            return 0.0
        return 0.5 * (voigt + material.mean_conductivity("reuss", physics))
    if not material.elastic:
        return None
    k, g = material.moduli("hill")
    return {"bulk_modulus": k, "shear_modulus": g, "young": 9.0 * k * g / (3.0 * k + g)}[name]


#: the physics of the scalar property of each transport phenomenon
_TRANSPORT = {name: physics for physics, name in TRANSPORT.items()}
#: base property of the scalar results: apparent properties, and entries of tensors
_BASE = {
    "bulk_modulus": "bulk_modulus",
    "shear_modulus": "shear_modulus",
    "conductivity": "conductivity",
    "E1": "young",
    "E2": "young",
    "E3": "young",
    "G23": "shear_modulus",
    "G31": "shear_modulus",
    "G12": "shear_modulus",
    "mean": "conductivity",
}


def _base(key: str) -> str | None:
    parts = key.split(".")
    if parts[0] in ("bulk_modulus", "shear_modulus", *_TRANSPORT) and len(parts) == 2:
        return parts[0]
    if parts[0] == "stiffness" and len(parts) == 3 and parts[2] != "mean":
        return _BASE.get(parts[2])
    scalar = parts[0].removesuffix("_tensor")  # the mean of the tensor of a transport
    if scalar != parts[0] and scalar in _TRANSPORT and len(parts) == 3 and parts[2] == "mean":
        return scalar
    return None


def normalized_values(
    values: dict, fractions: dict[int, float], materials: dict[int, Material]
) -> dict[str, float]:
    """Solid fraction, effective density, and the relative, normalized and specific
    values (see the module) of the apparent properties in ``values`` (keys
    ``<property>.<bc>``, and ``stiffness.<bc>.E1``... for the entries of tensors)."""
    phi = solid_fraction(fractions, materials)
    rho = effective_density(fractions, materials)
    out: dict[str, float] = {"solid_fraction": phi}
    if rho is not None:
        out["density"] = rho
    solid = [p for p in fractions if not materials[p].void and fractions[p] > 0]
    for key, value in values.items():
        base = _base(key)
        if base is None or not isinstance(value, float):
            continue
        props = [_property(materials[p], base) for p in solid]
        if not props or any(v is None for v in props):
            continue
        voigt = sum(fractions[p] * v for p, v in zip(solid, props, strict=True))
        if voigt > 0 and phi > 0:
            out[f"{key}.relative"] = value * phi / voigt
            out[f"{key}.normalized"] = value / voigt
        if rho:
            out[f"{key}.specific"] = value / rho
    return out
