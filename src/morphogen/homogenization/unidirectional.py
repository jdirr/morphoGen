"""Unidirectional fibre composites in closed form: the tows of woven textiles, the plies of
laminates.

Aligned fibres in a matrix make a material transversely isotropic about their axis (the
axis 3). In Hill's notation (Hill 1964) its stiffness has five moduli: ``k`` the
plane-strain bulk modulus of the transverse plane, ``l`` the coupling of the transverse and
axial strains, ``n`` the axial modulus in uniaxial strain, ``m`` the transverse shear modulus
and ``p`` the axial shear modulus,

    sigma_11 + sigma_22 = 2 k (eps_11 + eps_22) + 2 l eps_33,
    sigma_33 = l (eps_11 + eps_22) + n eps_33,
    sigma_11 - sigma_22 = 2 m (eps_11 - eps_22),   sigma_12 = 2 m eps_12,
    sigma_13 = 2 p eps_13,   sigma_23 = 2 p eps_23

(Voigt: ``C11 = k + m``, ``C12 = k - m``, ``C13 = l``, ``C33 = n``, ``C44 = C55 = p``, ``C66 =
m``). An isotropic phase has ``k = lambda + mu``, ``l = lambda``, ``n = lambda + 2 mu``, ``m = p
= mu``; a transversely isotropic fibre (carbon) its own five, its axis along the axis 3.

**The composite cylinder assemblage** of Hashin & Rosen (1964): cylinders of every size,
each a fibre in a coaxial shell of matrix in the proportions ``v_f``, ``v_m = 1 - v_f`` of
the composite, filling space. Its moduli ``k``, ``l``, ``n`` and ``p`` are exact (Hashin
1979 for transversely isotropic phases), with ``D = (k_f + m_m) v_m + (k_m + m_m) v_f``:

    k = [k_m (k_f + m_m) v_m + k_f (k_m + m_m) v_f] / D,
    l = [l_m (k_f + m_m) v_m + l_f (k_m + m_m) v_f] / D,
    n = v_m n_m + v_f n_f - v_m v_f (l_f - l_m)^2 / D,
    p = p_m [p_m v_m + p_f (1 + v_f)] / [p_m (1 + v_f) + p_f v_m];

``k`` and ``p`` are the Hashin–Shtrikman lower bounds of fibre composites of any transverse
geometry (Hashin 1965) when the fibres are the stiffer phase. The transverse shear
modulus ``m`` of the assemblage is not unique: :func:`composite_cylinders` takes that of
the generalized self-consistent scheme (Christensen & Lo 1979: a fibre in its shell of
matrix embedded in the effective medium; the root of Christensen's quadratic, with the
plane-strain constants ``eta = 1 + 2 m / k``, ``3 - 4 nu`` for an isotropic phase), or
Hashin's lower or upper bound (``transverse_shear``).

**Universal relations.** Hill (1964) showed that a uniform strain field is admissible in
any two-phase fibre composite under suitable loads, so that ``(l - l_m) / (k - k_m) = (l_f
- l_m) / (k_f - k_m)`` and ``n - v_m n_m - v_f n_f = (l - v_m l_m - v_f l_f)(l_f - l_m) /
(k_f - k_m)`` whatever the transverse geometry: with isotropic phases, the axial modulus
and Poisson's ratio follow from ``k`` alone,

    E_L = v_f E_f + v_m E_m + 4 (nu_f - nu_m)^2 / (1/k_f - 1/k_m)^2 (v_f/k_f + v_m/k_m - 1/k),
    nu_LT = v_f nu_f + v_m nu_m + (nu_f - nu_m) / (1/k_m - 1/k_f) (v_f/k_f + v_m/k_m - 1/k)

(:func:`hill_relations`). By the same argument the thermal expansion of any two-phase
composite follows from its compliance (Levin 1967; Rosen & Hashin 1970): ``alpha = <alpha> +
(S - <S>) (S_f - S_m)^-1 (alpha_f - alpha_m)``, ``<.>`` the means over the phases.

**Conduction.** Along the fibres the conductivities add, ``K_L = v_f K_f,L + v_m K_m``
(exact); across them the assemblage gives Hashin's ``K_T = K_m + v_f / (1 / (K_f,T - K_m) +
v_m / (2 K_m))``, the lower Hashin–Shtrikman bound of two dimensions when the fibres conduct
better.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from morphogen.homogenization.materials import Material, lame

TRANSVERSE_SHEAR = ("gscs", "lower", "upper")


@dataclass(frozen=True)
class Hill:
    """The five moduli of a material transversely isotropic about the axis 3 (Hill 1964)."""

    k: float
    l: float
    n: float
    m: float
    p: float

    @property
    def stiffness(self) -> np.ndarray:
        """The 6 x 6 stiffness (Voigt order 11, 22, 33, 23, 31, 12; engineering shears)."""
        c = np.zeros((6, 6))
        c[0, 0] = c[1, 1] = self.k + self.m
        c[0, 1] = c[1, 0] = self.k - self.m
        c[0, 2] = c[2, 0] = c[1, 2] = c[2, 1] = self.l
        c[2, 2] = self.n
        c[3, 3] = c[4, 4] = self.p
        c[5, 5] = self.m
        return c


def hill_moduli(material: Material) -> Hill:
    """Hill's moduli of an isotropic material, or of a crystal transversely isotropic about
    its axis 3 (the axis of the fibres: ``hexagonal``); ValueError otherwise."""
    if material.void or not material.elastic:
        raise ValueError("hill_moduli: an elastic material")
    if material.stiffness is None:
        lam, mu = lame(material.young, material.poisson)
        return Hill(lam + mu, lam, lam + 2.0 * mu, mu, mu)
    c = np.asarray(material.stiffness, dtype=float)
    scale = float(np.abs(c).max())
    expected = Hill(0.5 * (c[0, 0] + c[0, 1]), c[0, 2], c[2, 2], c[5, 5], c[3, 3]).stiffness
    if not np.allclose(c, expected, rtol=0.0, atol=1e-9 * scale):
        raise ValueError(
            "hill_moduli: the stiffness is not transversely isotropic about the axis 3 "
            "(the axis of the fibres)"
        )
    return Hill(0.5 * (c[0, 0] + c[0, 1]), float(c[0, 2]), float(c[2, 2]), float(c[5, 5]),
                float(c[3, 3]))  # fmt: skip


def _gscs_transverse_shear(f: Hill, m: Hill, c: float) -> float:
    """The transverse shear modulus of the generalized self-consistent scheme (Christensen
    & Lo 1979; Christensen 1990): the positive root of ``A x^2 + 2 B x + C =
    0``, ``x = m / m_m``."""
    g = f.m / m.m
    eta_f = 1.0 + 2.0 * f.m / f.k
    eta_m = 1.0 + 2.0 * m.m / m.k
    d = 3.0 * c * (1.0 - c) ** 2 * (g - 1.0) * (g + eta_f)
    w = g * eta_m - eta_f
    a = d + (g * eta_m + eta_f * eta_m - w * c**3) * (c * eta_m * (g - 1.0) - (g * eta_m + 1.0))
    b = (-d + 0.5 * (g * eta_m + (g - 1.0) * c + 1.0)
         * ((eta_m - 1.0) * (g + eta_f) - 2.0 * w * c**3)
         + 0.5 * c * (eta_m + 1.0) * (g - 1.0) * (g + eta_f + w * c**3))  # fmt: skip
    cc = d + (g * eta_m + (g - 1.0) * c + 1.0) * (g + eta_f + w * c**3)
    if abs(a) < 1e-300:
        return m.m * (-cc / (2.0 * b))
    roots = np.roots([a, 2.0 * b, cc])
    roots = roots[np.isreal(roots)].real
    positive = roots[roots > 0.0]
    if not len(positive):  # pragma: no cover (a negative discriminant: no such phases)
        raise ValueError("composite_cylinders: no positive transverse shear modulus")
    return float(m.m * positive.min() if len(positive) > 1 else m.m * positive[0])


def transverse_shear_bounds(fibre: Material, matrix: Material, fibre_fraction: float):
    """Hashin's bounds (Hashin 1965) of the transverse shear modulus of aligned fibres of any
    transverse arrangement: ``m_m + v_f / (1 / (m_f - m_m) + v_m (k_m + 2 m_m) / (2 m_m (k_m
    + m_m)))`` and the same with the phases exchanged (lower, upper when the fibres are the
    stiffer phase)."""
    f, m = hill_moduli(fibre), hill_moduli(matrix)
    vf = float(fibre_fraction)

    def bound(a: Hill, b: Hill, va: float) -> float:  # b the reference phase
        if a.m == b.m:
            return b.m
        return b.m + va / (1.0 / (a.m - b.m) + (1.0 - va) * (b.k + 2.0 * b.m) / (
            2.0 * b.m * (b.k + b.m)))  # fmt: skip

    lo, hi = bound(f, m, vf), bound(m, f, 1.0 - vf)
    return min(lo, hi), max(lo, hi)


def _voigt_strain(t: np.ndarray) -> np.ndarray:
    """A symmetric 3 x 3 strain as a Voigt vector (engineering shears)."""
    t = np.asarray(t, dtype=float)
    return np.array([t[0, 0], t[1, 1], t[2, 2], 2 * t[1, 2], 2 * t[2, 0], 2 * t[0, 1]])


def _stiffness_of(material: Material) -> np.ndarray:
    if material.stiffness is not None:
        return np.asarray(material.stiffness, dtype=float)
    return hill_moduli(material).stiffness


def _conductivities(material: Material) -> tuple[float, float] | None:
    """(transverse, axial) conductivity, or None without one."""
    if material.conductivities is not None:
        k = np.asarray(material.conductivities, dtype=float)
        if not np.allclose(k, np.diag([k[0, 0], k[0, 0], k[2, 2]]), atol=1e-12 * abs(k).max()):
            raise ValueError("composite_cylinders: a conductivity transversely isotropic about 3")
        return float(k[0, 0]), float(k[2, 2])
    if material.conductivity is not None:
        return float(material.conductivity), float(material.conductivity)
    return None


@dataclass(frozen=True)
class Unidirectional:
    """Aligned fibres in a matrix at the fibre fraction ``fibre_fraction``: Hill's moduli
    (:attr:`moduli`), the conductivities across and along the fibres and the thermal
    expansions across and along them (None when the phases have none), and the scheme of
    the transverse shear modulus. The axis 3 is along the fibres."""

    fibre_fraction: float
    moduli: Hill
    conductivity: tuple[float, float] | None = None
    expansion: tuple[float, float] | None = None
    transverse_shear: str = "gscs"

    @property
    def stiffness(self) -> np.ndarray:
        """The 6 x 6 stiffness, Voigt order 11, 22, 33, 23, 31, 12."""
        return self.moduli.stiffness

    def constants(self) -> dict[str, float]:
        """The engineering constants: the axial and transverse Young's moduli, the axial
        (``nu_LT``, the transverse contraction under an axial stress) and transverse Poisson's
        ratios, the axial and transverse shear moduli, the plane-strain bulk modulus."""
        s = np.linalg.inv(self.stiffness)
        return {
            "young_axial": float(1.0 / s[2, 2]),
            "young_transverse": float(1.0 / s[0, 0]),
            "poisson_axial": float(-s[0, 2] / s[2, 2]),
            "poisson_transverse": float(-s[0, 1] / s[0, 0]),
            "shear_axial": float(self.moduli.p),
            "shear_transverse": float(self.moduli.m),
            "plane_strain_bulk": float(self.moduli.k),
        }

    def material(self) -> Material:
        """The tow as a :class:`~morphogen.homogenization.materials.Material` (``hexagonal``,
        the axis 3 along the fibres), with its conductivities and expansions."""
        c = self.stiffness
        return Material.hexagonal(c[0, 0], c[0, 1], c[0, 2], c[2, 2], c[3, 3],
                                  conductivity=self.conductivity,
                                  expansion=self.expansion)  # fmt: skip


def composite_cylinders(
    fibre: Material, matrix: Material, fibre_fraction: float, *, transverse_shear: str = "gscs"
) -> Unidirectional:
    """The aligned fibres ``fibre`` (isotropic, or transversely isotropic about the axis 3)
    in ``matrix`` at ``fibre_fraction``, by the composite cylinder assemblage of Hashin &
    Rosen (see the module): ``k``, ``l``, ``n``, ``p`` exact for the assemblage, ``m`` by
    the generalized self-consistent scheme (``transverse_shear="gscs"``, Christensen & Lo
    1979) or Hashin's ``"lower"`` or ``"upper"`` bound; the conductivities (rule of mixtures
    along the fibres, Hashin's across) and the thermal expansions (Levin's relation) when
    both phases have them."""
    vf = float(fibre_fraction)
    if not 0.0 <= vf <= 1.0:
        raise ValueError("fibre_fraction: in [0, 1]")
    if transverse_shear not in TRANSVERSE_SHEAR:
        raise ValueError(f"transverse_shear: one of {', '.join(TRANSVERSE_SHEAR)}")
    f, m = hill_moduli(fibre), hill_moduli(matrix)
    vm = 1.0 - vf
    d = (f.k + m.m) * vm + (m.k + m.m) * vf
    k = (m.k * (f.k + m.m) * vm + f.k * (m.k + m.m) * vf) / d
    l = (m.l * (f.k + m.m) * vm + f.l * (m.k + m.m) * vf) / d
    n = vm * m.n + vf * f.n - vm * vf * (f.l - m.l) ** 2 / d
    p = m.p * (m.p * vm + f.p * (1.0 + vf)) / (m.p * (1.0 + vf) + f.p * vm)
    if vf in (0.0, 1.0):
        shear = m.m if vf == 0.0 else f.m
    elif transverse_shear == "gscs":
        shear = _gscs_transverse_shear(f, m, vf)
    else:
        lo, hi = transverse_shear_bounds(fibre, matrix, vf)
        shear = lo if transverse_shear == "lower" else hi
    moduli = Hill(k, l, n, shear, p)
    conductivity = None
    kf, km = _conductivities(fibre), _conductivities(matrix)
    if kf is not None and km is not None:
        if km[0] != km[1]:
            raise ValueError("composite_cylinders: a matrix of isotropic conductivity")
        kt_f, kl_f, k_m = kf[0], kf[1], km[0]
        across = k_m if kt_f == k_m else k_m + vf / (1.0 / (kt_f - k_m) + vm / (2.0 * k_m))
        conductivity = (float(across), float(vf * kl_f + vm * k_m))
    expansion = None
    if fibre.expansion is not None and matrix.expansion is not None:
        sf, sm = (np.linalg.inv(_stiffness_of(x)) for x in (fibre, matrix))
        af, am = (_voigt_strain(x.expansion_tensor()) for x in (fibre, matrix))
        mean_a = vf * af + vm * am
        mean_s = vf * sf + vm * sm
        s = np.linalg.inv(moduli.stiffness)
        a = mean_a + (s - mean_s) @ np.linalg.pinv(sf - sm, rcond=1e-12) @ (af - am)
        expansion = (float(a[0]), float(a[2]))
    return Unidirectional(vf, moduli, conductivity, expansion, transverse_shear)


def hill_relations(fibre: Material, matrix: Material, fibre_fraction: float, k: float):
    """The axial Young's modulus and Poisson's ratio that Hill's universal relations give a
    composite of isotropic fibres and matrix whose plane-strain bulk modulus is ``k``,
    whatever its transverse geometry (see the module)."""
    if fibre.stiffness is not None or matrix.stiffness is not None:
        raise ValueError("hill_relations: isotropic phases")
    vf = float(fibre_fraction)
    vm = 1.0 - vf
    kf, km = hill_moduli(fibre).k, hill_moduli(matrix).k
    ef, em, nf, nm = fibre.young, matrix.young, fibre.poisson, matrix.poisson
    if math.isclose(kf, km):
        return vf * ef + vm * em, vf * nf + vm * nm
    gap = vf / kf + vm / km - 1.0 / float(k)
    inv = 1.0 / kf - 1.0 / km
    young = vf * ef + vm * em + 4.0 * (nf - nm) ** 2 / inv**2 * gap
    poisson = vf * nf + vm * nm - (nf - nm) / inv * gap
    return float(young), float(poisson)
