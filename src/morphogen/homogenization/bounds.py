"""Bounds on the effective properties of multiphase media and polycrystals.

Isotropic phases: Voigt and Reuss bounds, and the Hashin-Shtrikman bounds in Walpole's
form (valid for any number of isotropic phases); for conduction, the Wiener and
Hashin-Shtrikman bounds, which hold alike for every transport phenomenon of a scalar
potential (:data:`~morphogen.homogenization.materials.TRANSPORT`: the electrical
conductivity, the permittivity, the permeability, the diffusivity). Voids have zero moduli
and conductivity, as the insulators of a transport: every lower bound is then zero.

Thermal expansion: the expansion of phases expanding alike is exact; for two isotropic
phases, Levin's exact relation (Levin 1967) maps the bulk modulus onto the expansion of a
statistically isotropic composite, and the Hashin-Shtrikman bounds of the former give
those of the latter (Schapery 1968).

Crystals (statistically isotropic aggregates, uniform orientations): the Voigt and
Reuss averages of each phase bound the moduli; for a polycrystal of one cubic phase, the
Hashin-Shtrikman bounds of the shear modulus (Hashin & Shtrikman 1962, J. Mech. Phys.
Solids 10, 343-352), its bulk modulus being exactly ``(C11 + 2 C12) / 3``; for
conduction, the Hashin-Shtrikman bounds of a polycrystal are those of a mixture of its
principal conductivities in equal fractions.
"""

from __future__ import annotations

import numpy as np

from morphogen.homogenization.materials import TRANSPORT, Material


def _check(fractions, values):
    f = np.asarray(fractions, dtype=float)
    v = np.asarray(values, dtype=float)
    if f.shape != v.shape or f.ndim != 1 or f.size == 0:
        raise ValueError("give one fraction per phase")
    if np.any(f < 0) or not np.isclose(f.sum(), 1.0, atol=1e-9):
        raise ValueError(f"phase fractions must be non-negative and sum to 1, got {f.sum()}")
    if np.any(v < 0):
        raise ValueError("moduli and conductivities must be non-negative")
    return f, v


def voigt(fractions, values) -> float:
    """The Voigt bound (arithmetic mean) of the properties of phases in their volume fractions."""
    f, v = _check(fractions, values)
    return float(np.sum(f * v))


def reuss(fractions, values) -> float:
    """The Reuss bound (harmonic mean) of the properties of phases in their volume fractions; zero
    when a phase of nonzero fraction has a zero property."""
    f, v = _check(fractions, values)
    if np.any((v == 0) & (f > 0)):
        return 0.0
    return float(1.0 / np.sum(f / np.where(v > 0, v, np.inf)))


def _shifted_harmonic(f, v, shift) -> float:
    """[sum f_i / (v_i + s)]^-1 - s (zero when s = 0 and a phase has v_i = 0)."""
    den = v + shift
    if np.any((den == 0) & (f > 0)):
        return 0.0
    return float(1.0 / np.sum(f / np.where(den > 0, den, np.inf)) - shift)


def hashin_shtrikman_bulk(fractions, bulk, shear) -> tuple[float, float]:
    """(lower, upper) Hashin-Shtrikman bounds of the bulk modulus."""
    f, k = _check(fractions, bulk)
    _, mu = _check(fractions, shear)
    return (
        _shifted_harmonic(f, k, 4.0 * mu.min() / 3.0),
        _shifted_harmonic(f, k, 4.0 * mu.max() / 3.0),
    )


def _zeta(k: float, mu: float) -> float:
    return mu / 6.0 * (9.0 * k + 8.0 * mu) / (k + 2.0 * mu) if (k + 2.0 * mu) > 0 else 0.0


def hashin_shtrikman_shear(fractions, bulk, shear) -> tuple[float, float]:
    """(lower, upper) Hashin-Shtrikman bounds of the shear modulus (Walpole 1966)."""
    f, k = _check(fractions, bulk)
    _, mu = _check(fractions, shear)
    return (
        _shifted_harmonic(f, mu, _zeta(k.min(), mu.min())),
        _shifted_harmonic(f, mu, _zeta(k.max(), mu.max())),
    )


def hashin_shtrikman_conductivity(fractions, conductivity) -> tuple[float, float]:
    """(lower, upper) Hashin-Shtrikman bounds of the conductivity (3D)."""
    f, c = _check(fractions, conductivity)
    return _shifted_harmonic(f, c, 2.0 * c.min()), _shifted_harmonic(f, c, 2.0 * c.max())


def cubic_shear_bounds(c11: float, c12: float, c44: float) -> tuple[float, float]:
    """(lower, upper) Hashin-Shtrikman bounds of the shear modulus of a statistically
    isotropic polycrystal of a cubic crystal (Hashin & Shtrikman 1962)."""
    k = (c11 + 2.0 * c12) / 3.0
    mu1, mu2 = 0.5 * (c11 - c12), c44
    if np.isclose(mu1, mu2):
        return mu1, mu1
    beta1 = -3.0 * (k + 2.0 * mu1) / (5.0 * mu1 * (3.0 * k + 4.0 * mu1))
    beta2 = -3.0 * (k + 2.0 * mu2) / (5.0 * mu2 * (3.0 * k + 4.0 * mu2))
    g1 = mu1 + 3.0 / (5.0 / (mu2 - mu1) - 4.0 * beta1)
    g2 = mu2 + 2.0 / (5.0 / (mu1 - mu2) - 6.0 * beta2)
    return float(min(g1, g2)), float(max(g1, g2))


def levin_expansion(fractions, materials, bulk: float) -> float:
    """Thermal expansion of a statistically isotropic composite of two isotropic phases
    of effective bulk modulus ``bulk`` (Levin 1967): ``alpha = <alpha> + (alpha_1 -
    alpha_2) / (1/k_1 - 1/k_2) (1/k - <1/k>)``."""
    f = np.asarray(fractions, dtype=float)
    (m1, m2) = materials
    a = np.array([float(m1.expansion or 0.0), float(m2.expansion or 0.0)])
    k = np.array([m1.bulk, m2.bulk])
    if np.isclose(k[0], k[1]):
        return float(f @ a)
    return float(f @ a + (a[0] - a[1]) / (1.0 / k[0] - 1.0 / k[1]) * (1.0 / bulk - f @ (1.0 / k)))


def _expansion_bounds(f, mats) -> dict | None:
    solid = [(fi, m) for fi, m in zip(f, mats, strict=True) if not m.void and fi > 0]
    if not solid or not all(m.elastic for _, m in solid):
        return None
    if any(isinstance(m.expansion, tuple) for _, m in solid):
        return None
    alphas = {float(m.expansion or 0.0) for _, m in solid}
    if len(alphas) == 1:  # a uniform thermal strain is free of stress
        a = alphas.pop()
        return {"hs_lower": a, "hs_upper": a}
    if len(solid) != 2 or len(mats) != 2 or not all(m.isotropic for _, m in solid):
        return None
    fr = np.array([fi for fi, _ in solid])
    k = np.array([m.bulk for _, m in solid])
    mu = np.array([m.shear for _, m in solid])
    two = [m for _, m in solid]
    ends = [levin_expansion(fr, two, kb) for kb in hashin_shtrikman_bulk(fr, k, mu)]
    return {"hs_lower": min(ends), "hs_upper": max(ends)}


def _crystal_bounds(f, mats) -> dict[str, dict]:
    """Voigt and Reuss bounds with the orientation averages of each phase, and the
    Hashin-Shtrikman bounds of one-phase polycrystals."""
    out: dict[str, dict] = {}
    solid = [m for m in mats if not m.void]
    single = len(mats) == 1 and not mats[0].void
    if all(m.void or m.elastic for m in mats):
        kv = np.array([0.0 if m.void else m.moduli("voigt")[0] for m in mats])
        gv = np.array([0.0 if m.void else m.moduli("voigt")[1] for m in mats])
        kr = np.array([0.0 if m.void else m.moduli("reuss")[0] for m in mats])
        gr = np.array([0.0 if m.void else m.moduli("reuss")[1] for m in mats])
        out["bulk_modulus"] = {"voigt": voigt(f, kv), "reuss": reuss(f, kr)}
        out["shear_modulus"] = {"voigt": voigt(f, gv), "reuss": reuss(f, gr)}
        if single and solid[0].symmetry == "cubic":
            d = np.array(solid[0].stiffness)
            k = float((d[0, 0] + 2.0 * d[0, 1]) / 3.0)
            out["bulk_modulus"].update(hs_lower=k, hs_upper=k)
            lo, hi = cubic_shear_bounds(d[0, 0], d[0, 1], d[3, 3])
            out["shear_modulus"].update(hs_lower=lo, hs_upper=hi)
    for physics, name in TRANSPORT.items():
        if not all(m.void or m.has(physics) for m in mats):
            continue
        inert = [m.inert(physics) for m in mats]
        cv = np.array([0.0 if i else m.mean_conductivity("voigt", physics)
                       for m, i in zip(mats, inert, strict=True)])  # fmt: skip
        cr = np.array([0.0 if i else m.mean_conductivity("reuss", physics)
                       for m, i in zip(mats, inert, strict=True)])  # fmt: skip
        out[name] = {"voigt": voigt(f, cv), "reuss": reuss(f, cr)}
        if single and not inert[0]:
            k = np.linalg.eigvalsh(solid[0].matrix(physics))
            lo, hi = hashin_shtrikman_conductivity(np.full(3, 1.0 / 3.0), k)
            out[name].update(hs_lower=lo, hs_upper=hi)
    return out


def bounds(fractions: dict[int, float], materials: dict[int, Material]) -> dict[str, dict]:
    """All bounds for the phases present, keyed by property then bound name.

    ``fractions`` maps phase -> volume fraction (e.g. from
    :func:`morphogen.analysis.phase_fractions`). With crystals (anisotropic materials),
    the bounds are those of statistically isotropic aggregates (uniform orientations).
    """
    phases = [p for p, v in fractions.items() if v > 0]
    f = np.array([fractions[p] for p in phases], dtype=float)
    f = f / f.sum()
    mats = [materials[p] for p in phases]
    expansion = _expansion_bounds(f, mats)
    if not all(m.void or m.isotropic for m in mats):
        out = _crystal_bounds(f, mats)
        if expansion is not None and any(m.expansion is not None for m in mats):
            out["thermal_expansion"] = expansion
        return out
    out: dict[str, dict] = {}
    if expansion is not None and any(m.expansion is not None for m in mats):
        out["thermal_expansion"] = expansion
    if all(m.void or m.elastic for m in mats):
        k = np.array([0.0 if m.void else m.bulk for m in mats])
        mu = np.array([0.0 if m.void else m.shear for m in mats])
        k_hs = hashin_shtrikman_bulk(f, k, mu)
        mu_hs = hashin_shtrikman_shear(f, k, mu)
        out["bulk_modulus"] = {
            "voigt": voigt(f, k),
            "reuss": reuss(f, k),
            "hs_lower": k_hs[0],
            "hs_upper": k_hs[1],
        }
        out["shear_modulus"] = {
            "voigt": voigt(f, mu),
            "reuss": reuss(f, mu),
            "hs_lower": mu_hs[0],
            "hs_upper": mu_hs[1],
        }
    for physics, name in TRANSPORT.items():
        if not all(m.void or m.has(physics) for m in mats):
            continue
        c = np.array([0.0 if m.inert(physics) else m.mean_conductivity("voigt", physics)
                      for m in mats])  # fmt: skip
        c_hs = hashin_shtrikman_conductivity(f, c)
        out[name] = {
            "voigt": voigt(f, c),
            "reuss": reuss(f, c),
            "hs_lower": c_hs[0],
            "hs_upper": c_hs[1],
        }
    return out
