"""MFront sources of morphoGen's behaviours (TFEL, https://thelfer.github.io/tfel).

:func:`mfront_source` writes a nonlinear material of morphoGen as an MFront behaviour, for
the codes that integrate MFront behaviours through its interfaces (AMITEX_FFTP and Cast3M
with ``umat``, Abaqus with ``abaqus``, Code_Aster, MGIS: see
:mod:`morphogen.homogenization.mfront`); the values of the material are constants of the
source:

* von Mises elasto-(visco)plasticity (:class:`~morphogen.homogenization.Plasticity`): the
  brick ``StandardElastoViscoPlasticity`` of MFront: Hooke's law, the Norton flow of the
  von Mises criterion (``Plastic``, rate-independent, without viscosity), linear and Voce
  isotropic hardening (``R0 + H p + sum Q (1 - exp(-b p))``), Armstrong-Frederick and
  Prager kinematic hardening (``X = 2/3 C a``, ``da = deps_p - D a dp``); in finite
  strains, in the space of the logarithmic strain (``@StrainMeasure Hencky``), as
  morphoGen's finite strains. Hill's criterion makes it an orthotropic behaviour (the
  criterion ``Hill`` of the brick, with TFEL's ``F`` ... ``N`` from Z-set's coefficients:
  :mod:`morphogen.homogenization.zmat`), in the axes of the sample;
* porous plasticity (:class:`~morphogen.homogenization.porous.PorousPlasticity`): where
  MFront's brick is exactly morphoGen's porous plasticity (:func:`_porous_brick`), the
  criterion of Gurson-Tvergaard-Needleman: the same brick with its porous criterion
  ``GursonTvergaardNeedleman1982`` (``q_1``, ``q_2``, ``q_3``; the effective porosity
  ``f`` or Tvergaard & Needleman's coalescence, MFront's ``f_c`` and ``f_r``, ``f_r``
  the porosity of morphoGen's slope of ``f*`` when ``q3`` is not ``q1²``) and the
  porosity in its implicit system (``porosity_evolution``, ``standard implicit
  scheme``: morphoGen's backward Euler, ``ε̇p = (1 - f) ṗ ∂σ*/∂σ``, ``ḟ = (1 - f) tr
  ε̇p + A ṗ``), Plastic or Norton flows with the isotropic hardening of von Mises,
  morphoGen's constant nucleation from ``p = 0`` (MFront's power law of exponent 0,
  ``"PowerLaw (strain)"``); its porosity is ``f0`` until the matrix flows (the source
  sets it while ``p = 0``: the natural state of its internal variables is zero, as
  those of morphoGen's other exports). Small and finite strains (Hencky), thermal
  expansion as von Mises. The bounds of the porosity differ: morphoGen keeps ``f*``
  below ``0.995 / q1``, the brick its porosity below 0.985 of its upper bound (1, or
  ``f_r``) and breaks the material (zero stress) beyond 0.984 of it. The others
  (Rousselier's criterion, the elliptic one, the effective porosities of other tables
  and the coalescence with ``q3 > q1²``, Chu & Needleman's Gaussian and the exponential
  nucleation, the thresholds of the nucleation, kinematic hardening: :func:`_porous_brick`
  says why the brick differs) hold morphoGen's own integration of a point
  (``porous_update``, the C port of its kernel, with its row of parameters and its
  internal variables: :func:`_porous_kernel`), in small strains or in the space of the
  logarithmic strain (``@StrainMeasure Hencky``), its consistent tangent the operator;
* Méric-Cailletaud crystal plasticity
  (:class:`~morphogen.homogenization.CrystalPlasticity`), in small strains: an implicit
  scheme of morphoGen's equations (its Jacobian by finite differences), the elasticity of
  the crystal as orthotropic constants in its axes (``@ComputeStiffnessTensor``: E1, E2,
  E3, nu12, nu23, nu13, G12, G23, G13, MFront's order), morphoGen's slip systems as their
  Schmid tensors in TFEL's components (11, 22, 33, then sqrt(2) times 12, 13, 23: not
  morphoGen's order, 23, 31, 12), the parameters of each system and the interaction
  matrix. The behaviour is orthotropic: the solver gives the axes of each grain (MGIS
  through morphoGen turns the strains to them; the ``umat`` interface takes them from
  the coefficients V1, V2; Abaqus from ``*ORIENTATION``); in finite strains, morphoGen's
  own integration of a point (its kernel in the source);
* in finite strains, the elastic materials too: Hencky's elasticity (the stress conjugate
  to the logarithmic strain linear in it, ``@StrainMeasure Hencky``), isotropic or in
  the axes of the crystal;
* the learned laws (a material whose ``learned`` names a trained model:
  :mod:`morphogen.learning.mfront`): the trained parameters as constants and morphoGen's
  C++ port of the model (the perceptron, the potential, the recurrent network and its
  state, the standard model and its internal variables), in small strains or in finite
  strains (``@StrainMeasure Hencky`` for the laws of the logarithmic strain,
  ``GreenLagrange`` for the others), their thermal strain that of morphoGen's learned
  laws, orthotropic in the axes of their material.

The interfaces of the other codes compute the logarithmic strain of the Hencky laws from
the deformation gradients (the function ``umat<behaviour>_malls`` of the ``umat``
interface: :func:`umat_function`). Abaqus gives the axes of the orthotropic laws in
finite strains through their first six state variables (``@AbaqusOrthotropyManagementPolicy``
``MFront``, which the other interfaces ignore). :func:`mfront_state_size` gives the
number of internal state variables of a behaviour (those of the ``umat`` and ``abaqus``
interfaces in three dimensions, without these axes).

Thermal expansion is exported in small strains: ``thermal_expansion`` of the Hooke
potential of the brick (von Mises; Hill's criterion also takes three principal values in
the axes of the sample), ``@ComputeThermalExpansion`` of the crystals (one coefficient,
or three principal values in the axes of the crystal), morphoGen's thermal strain ``alpha
(T + dT - 293.15)`` in the sources of its porous kernel. Its reference temperatures are
MFront's, 293.15 K, the natural temperature of morphoGen's MFront behaviours (MGIS gets
``293.15 + dT``): MFront's ``alpha (T - 293.15)`` is morphoGen's ``alpha dT``. Finite
strains: MFront's coefficient is a mean one (its Hencky brick stretches by ``1 + alpha
dT``), morphoGen's that of the logarithmic strain ``alpha dT``: the brick gets the mean
coefficient ``(exp(alpha theta) - 1) / theta`` (:func:`_mean_expansion`); the crystals,
Hencky's elasticity, the porous kernels and the learned laws of finite strains,
morphoGen's own sources, compute its logarithmic thermal strain themselves. The Abaqus
inputs start their nodes at 293.15 K when a material expands (their steps give absolute
temperatures); MFront's
abaqus interface has no stress-free expansion in finite strains, so they refuse the
brick's expanding laws of finite strains (:func:`mfront_brick`; morphoGen's own sources
compute their thermal strain and pass). The AMITEX_FFTP exports refuse every expanding
phase written as an MFront law: their loadings carry no change of temperature.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from morphogen._version import __version__

_SQRT2 = math.sqrt(2.0)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: The axes of the orthotropic laws of finite strains in their first six state variables,
#: in Abaqus (the only orthotropy of its interface in finite strains; others ignore it)
ABAQUS_AXES = "@AbaqusOrthotropyManagementPolicy[abaqus] MFront;"
#: The function of the umat interface (Cast3M, AMITEX) of each strain measure in finite
#: strains: its strategies (Miehe, Apel & Lambrecht's logarithmic strain for Hencky,
#: finite rotations and small strains for Green-Lagrange)
_UMAT_FINITE = {"Hencky": "_malls", "GreenLagrange": "_frst"}
_STRAIN_MEASURE = re.compile(r"^\s*@StrainMeasure\s*(?:\[[^\]]*\]\s*)?(\w+)", re.M)
_ABAQUS_POLICY = re.compile(
    r"^\s*@AbaqusOrthotropyManagementPolicy\s*(?:\[[^\]]*\]\s*)?MFront\b", re.M
)


def _real(value: float) -> str:
    return repr(float(value))


def identifier(name: str) -> str:
    """An identifier for the names of MFront behaviours, from the name of a file or of an
    export: its letters and digits only (the dispatcher of MFront's laws in Abaqus splits
    the names of the materials, ``ABAQUSBEHAVIOUR_<behaviour>_3D``, at their
    underscores)."""
    out = re.sub(r"[^A-Za-z0-9]", "", name)
    return out if out[:1].isalpha() else f"M{out}"


def _check(material, kinematics: str) -> str:
    """The kind of behaviour of a material to export: ``vonmises``, ``porous`` (the brick),
    ``porous_kernel`` (morphoGen's kernel in the source: :func:`_porous_brick`),
    ``crystal``, ``learned`` (:mod:`morphogen.learning.mfront`), and in finite strains
    ``crystal_finite`` and ``elastic_finite``."""
    if kinematics not in ("small", "finite"):
        raise ValueError("kinematics: 'small' or 'finite'")
    if material.void:
        raise ValueError("mfront: a void has no behaviour")
    if not material.nonlinear and kinematics == "small":
        raise ValueError("mfront: a nonlinear material (plasticity or crystal_plasticity)")
    if material.mfront is not None:
        raise ValueError("mfront: the material is already an MFront behaviour (its source)")
    if material.analytic:
        raise ValueError(
            "mfront: no source for the viscoelasticity, hyperelasticity and damage of "
            "morphogen.homogenization.laws (integrated by morphoGen's solvers only)"
        )
    if material.learned is not None:  # its model and expansion: morphogen.learning.mfront
        from morphogen.learning.mfront import learned_model

        learned_model(material, kinematics)
        return "learned"
    if material.expands:
        _check_expansion(material, kinematics)
    if material.porous:
        return "porous" if _porous_brick(material.porous_plasticity) else "porous_kernel"
    if material.crystal_plastic:
        if material.crystal_plasticity.extended:
            raise ValueError(
                "mfront: no source for the twins and the cleavage planes of a crystal "
                "(integrated by morphoGen's solvers only)"
            )
        return "crystal_finite" if kinematics == "finite" else "crystal"
    if not material.nonlinear:
        if not material.elastic:
            raise ValueError("mfront: the material has no elastic parameters")
        return "elastic_finite"
    return "vonmises"


def _check_expansion(material, kinematics: str) -> None:
    """The thermal expansions that the sources take: in small strains, one coefficient, or
    principal values in the axes of the behaviour (those of a crystal, or of the sample for
    Hill's criterion)."""
    if not isinstance(material.expansion, tuple):
        return
    a = np.asarray(material.expansion, dtype=float)
    if np.abs(a - np.diag(np.diag(a))).max() > 1e-12 * np.abs(a).max():
        raise ValueError("mfront: an expansion tensor with shear components in its axes")
    if material.plastic and material.plasticity.hill is None:
        raise ValueError("mfront: an anisotropic expansion of a von Mises material")
    if material.porous:
        raise ValueError("mfront: an anisotropic expansion of a porous material")


def _mean_expansion(alpha: float) -> str:
    """MFront's mean coefficient of thermal expansion (``dl / l0 = alpha_m (T - T0)``, the
    stretch of its Hencky brick) whose logarithmic strain is morphoGen's ``alpha (T - T0)``
    in finite strains: ``(exp(alpha theta) - 1) / theta``, ``theta = T - 293.15``, by its
    series (five terms: below 1e-11 of it up to ``alpha theta = 0.02``)."""
    u = f"({_real(alpha)}*(T-293.15))"
    return (f'"{_real(alpha)}*(1+{u}*(0.5+{u}*(0.16666666666666666+{u}*'
            f'(0.041666666666666664+{u}*0.008333333333333333))))"')  # fmt: skip


def _thermal_strain(material, name: str) -> list[str]:
    """The C lines of morphoGen's thermal strain at the end of the step (logarithmic in
    finite strains), its tensor components ``name[6]`` in the axes of the behaviour:
    ``alpha (T + dT - 293.15)`` (the sources that hold morphoGen's own kernels: crystals and
    Hencky's elasticity of finite strains, the porous kernels)."""
    values = _expansion_values(material) if material.expands else [0.0]
    if len(values) == 1:
        values = values * 3
    theta = "(T + dT - 293.15)"
    rows = ", ".join(f"{_real(v)} * {theta}" for v in values)
    return [f"  const double {name}[6] = {{{rows}, 0, 0, 0}};"]


def _hencky_strain(material) -> list[str]:
    """The C lines of the elastic Hencky strain ``e``: the logarithmic strain at the end of
    the step without morphoGen's thermal strain."""
    if not material.expands:
        return ["  const auto e = eto + deto;"]
    thermal = "StrainStensor{eth[0], eth[1], eth[2], 0, 0, 0}"
    return [*_thermal_strain(material, "eth"), f"  const auto e = eto + deto - {thermal};"]


def _expansion_values(material) -> list[float]:
    """The coefficient of thermal expansion, or its three principal values."""
    if isinstance(material.expansion, tuple):
        return [float(v) for v in np.diag(np.asarray(material.expansion, dtype=float))]
    return [float(material.expansion)]


def orthotropic_constants(stiffness) -> list[float]:
    """The engineering constants of an orthotropic stiffness (6 x 6, morphoGen's Voigt
    order 11, 22, 33, 23, 31, 12 with engineering strains) in MFront's order: E1, E2, E3,
    nu12, nu23, nu13, G12, G23, G13 (``nu_ij = -eps_j / eps_i`` under ``sigma_i``)."""
    d = np.asarray(stiffness, dtype=float)
    coupling = np.concatenate([d[:3, 3:].ravel(), d[3:, :3].ravel(),
                               d[3:, 3:][~np.eye(3, dtype=bool)]])  # fmt: skip
    if np.abs(coupling).max() > 1e-12 * np.abs(d).max():
        raise ValueError(
            "mfront: the stiffness of the crystal couples normal and shear components "
            "(MFront's orthotropic elasticity in its axes)"
        )
    s = np.linalg.inv(d)
    e1, e2, e3 = 1 / s[0, 0], 1 / s[1, 1], 1 / s[2, 2]
    return [e1, e2, e3, -s[1, 0] * e1, -s[2, 1] * e2, -s[2, 0] * e1,
            1 / s[5, 5], 1 / s[3, 3], 1 / s[4, 4]]  # fmt: skip


def _stiffness(material) -> np.ndarray:
    from morphogen.homogenization.materials import isotropic_matrix

    if material.stiffness is not None:
        return np.asarray(material.stiffness, dtype=float)
    return isotropic_matrix(*material.lame)


def _young_poisson(material) -> tuple[float, float]:
    lam, mu = material.lame
    return mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))


def mfront_state_size(material, kinematics: str = "small") -> int:
    """The number of internal state variables of the MFront behaviour of a material (see
    :func:`mfront_source`) in three dimensions: the elastic strain (6), then the back
    strains of the kinematic terms (6 each) and the cumulated plastic strain (von Mises),
    the porosity, the cumulated plastic strain of the matrix, the rupture of the brick and,
    with a nucleation, the nucleated porosity (porous plasticity of the brick), or the
    slips, cumulated slips and kinematic variables of the systems and the cumulated plastic
    strain (crystals); in finite strains, the internal variables of morphoGen's kernel and
    the cumulated plastic strain (crystals), none (elasticity). The porous kernels: the
    elastic strain (6, small strains only), the internal variables of morphoGen's kernel
    (10, and 6 per backstress), the cumulated plastic strain and the porosity. The learned
    laws: the state of the recurrent network, the internal variables of the standard model
    (none for the perceptron and the potential)."""
    from morphogen.homogenization import porous
    from morphogen.homogenization.crystal import state_width

    kind = _check(material, kinematics)
    if kind == "learned":
        from morphogen.learning.mfront import learned_model

        return learned_model(material, kinematics).n_state
    n = material.crystal_plasticity.n_systems if material.crystal_plastic else 0
    if kind == "crystal_finite":
        return state_width(n, True) + 1
    if kind == "crystal":
        return 7 + 3 * n
    if kind == "elastic_finite":
        return 0
    if kind == "porous":
        return 9 + (material.porous_plasticity.nucleation is not None)
    if kind == "porous_kernel":
        width = porous.state_width(len(material.porous_plasticity.kinematic))
        return width + 2 + (6 if kinematics == "small" else 0)
    return 7 + 6 * len(material.plasticity.kinematic)


def mfront_brick(material, kinematics: str = "small") -> bool:
    """Whether the source of a material is MFront's brick ``StandardElastoViscoPlasticity``
    (von Mises, Hill's criterion, the porous plasticity that it matches), whose thermal
    expansion is MFront's stress-free one (``thermal_expansion`` of its Hooke potential),
    or one of morphoGen's own sources (crystals, Hencky's elasticity, the porous kernels,
    the learned laws), which compute their thermal strain."""
    return _check(material, kinematics) in ("vonmises", "porous")


def mfront_symmetry(material, kinematics: str = "small") -> str:
    """``isotropic`` (von Mises, isotropic elasticity) or ``orthotropic`` (crystals,
    anisotropic elasticity and Hill's criterion: the solver gives their axes, those of the
    sample for Hill's criterion) behaviour of a material; a learned law is orthotropic in
    the axes of its material (``oriented``), isotropic in those of the sample."""
    kind = _check(material, kinematics)
    if kind == "learned":
        return "orthotropic" if material.learned.oriented else "isotropic"
    crystal = kind in ("crystal", "crystal_finite")
    hill = kind == "vonmises" and material.plasticity.hill is not None
    orthotropic = crystal or hill or material.stiffness is not None
    return "orthotropic" if orthotropic else "isotropic"


def mfront_strain_measure(material, kinematics: str = "small") -> str | None:
    """The strain measure of the source of a material (:func:`mfront_source`): Hencky's
    for elasticity, von Mises and porous plasticity in finite strains, none for the others (small
    strains, and crystals in finite strains, whose kernel takes the deformation
    gradient); a learned law in finite strains, Hencky's (a law of the logarithmic strain)
    or Green-Lagrange's."""
    kind = _check(material, kinematics)
    if kind == "learned":
        from morphogen.learning.mfront import learned_strain_measure

        return learned_strain_measure(material, kinematics)
    hencky = ("vonmises", "porous", "porous_kernel", "elastic_finite")
    return "Hencky" if kinematics == "finite" and kind in hencky else None


def umat_function(behaviour: str, strain_measure: str | None = None) -> str:
    """The function of a behaviour in the library of the umat interface (Cast3M, AMITEX):
    ``umat<behaviour>`` (in lower case), and for the strain measures of finite strains
    that of the strategy of the interface (``_malls``: Hencky's logarithmic strain,
    ``_frst``: Green-Lagrange's)."""
    if strain_measure is not None and strain_measure not in _UMAT_FINITE:
        raise ValueError(f"umat: the strain measure {strain_measure} (Hencky or GreenLagrange)")
    return f"umat{behaviour.lower()}{_UMAT_FINITE.get(strain_measure, '')}"


def _kernels(ns: int, width: int) -> str:
    """morphoGen's kernels of the points for a host compiler: the C port of the GPU
    (``behaviours.cu``) without its kernels of launches, its functions inline (the block
    is in several translation units) and its constants ``constexpr`` (of the namespace of
    the behaviour: macros would reach TFEL's headers)."""
    from importlib import resources

    from morphogen.homogenization.behaviours import kernel_constants

    path = resources.files("morphogen.homogenization") / "behaviours.cu"
    text = path.read_text(encoding="utf-8").split('extern "C" __global__')[0]
    text = text.replace("#define MG_INF __longlong_as_double(0x7ff0000000000000LL)\n", "")
    for qualifier in ("__device__ __forceinline__ ", "__device__ __noinline__ ", "__device__ "):
        text = text.replace(qualifier, "inline ")
    head = [
        f"constexpr {'double' if isinstance(value, float) else 'int'} {name} = {value!r};"
        for name, value in kernel_constants(ns, width).items()
    ]
    return "\n".join([*head, "constexpr double MG_INF = HUGE_VAL;", text])


def _crystal_finite(material, name: str) -> str:
    """Méric-Cailletaud crystal plasticity in finite strains: morphoGen's own integration
    of a point (``cp_finite``, the C port of its kernel), the first Piola-Kirchhoff stress
    and its consistent derivative, wrapped in a behaviour of the ``DefaultFiniteStrain``
    DSL; the internal variables of the kernel (in the axes of the crystal) are an array of
    auxiliary state variables."""
    from morphogen.homogenization.crystal import state_width

    cp = material.crystal_plasticity
    normals, directions = cp.systems()
    ns = len(normals)
    width = state_width(ns, True)
    systems = np.concatenate([directions, normals], axis=1)  # l, n of each system
    families = "; ".join(
        f"{fam} (" + ", ".join(f"{k} = {v:g}" for k, v in
                               zip(("tau0", "Q", "b", "C", "D", "K", "n"), values,
                                   strict=True)) + ")"
        for fam, values in cp.families
    )  # fmt: skip
    space = f"morphogen_{name}"
    lines = [
        "@DSL DefaultFiniteStrain;",
        f"@Behaviour {name};",
        "@Author morphoGen;",
        "@Description {",
        f'  "morphoGen {__version__}: Meric-Cailletaud crystal plasticity in finite strains '
        f"(F = Fe Fp, Saint Venant-Kirchhoff elasticity of Fe, the slip systems in the "
        f"intermediate configuration), {ns} slip systems: {families}; integrated by "
        f"morphoGen's kernel (cp_finite), in the axes of the crystal.\"",
        "}",
        "@ModellingHypothesis Tridimensional;",
        "@OrthotropicBehaviour;",
        ABAQUS_AXES,
        f"@AuxiliaryStateVariable real mg[{width}];",
        'mg.setEntryName("MorphogenState");',
        "@AuxiliaryStateVariable strain p;",
        'p.setGlossaryName("EquivalentPlasticStrain");',
        "@LocalVariable real K[81];",
        "",
        "@Includes{",
        "// morphoGen's kernels of the points and the constants of the material (MFront",
        "// includes this block in several headers of one translation unit)",
        f"#ifndef {space.upper()}_KERNELS",
        f"#define {space.upper()}_KERNELS",
        "#include <cmath>",
        "#include <math.h>",
        f"namespace {space} {{",
        _kernels(ns, width),
        "// the slip systems (directions l, normals n), the parameters of each system, the",
        "// interaction matrix and the stiffness (Voigt 11 22 33 23 31 12, engineering",
        "// shears), in the axes of the crystal",
        f"constexpr double systems[{6 * ns}] = {_array(systems.ravel())};",
        f"constexpr double params[{7 * ns}] = {_array(cp.params_array().ravel())};",
        f"constexpr double h[{ns * ns}] = {_array(cp.interaction_matrix().ravel())};",
        f"constexpr double stiffness[36] = {_array(_stiffness(material).ravel())};",
        "// the components of TFEL's tensors (11 22 33 12 21 13 31 23 32) row by row",
        "constexpr int ROWS[9] = {0, 4, 8, 1, 3, 2, 6, 5, 7};",
        f"}}  // end of namespace {space}",
        "#endif",
        "}",
        "",
        "@Integrator{",
        f"  namespace mk = {space};",
        "  double hg[9], f[9], old[mk::WIDTH], nw[mk::WIDTH], pk1[9];",
        "  for (int k = 0; k != 9; ++k) {",
        "    f[mk::ROWS[k]] = F1[k];",
        "    hg[mk::ROWS[k]] = F1[k] - (k < 3 ? 1 : 0);",
        "  }",
        "  for (int r = 0; r != mk::WIDTH; ++r) {",
        "    old[r] = mg[r];",
        "  }",
        *_thermal_strain(material, "eth"),
        "  mk::Work work;",
        "  if (mk::cp_finite(mk::systems, mk::params, mk::h, mk::NS, mk::stiffness, hg, eth, dt,",
        "                    old, nw, pk1, &K[0], 0, 1.0, &work) != 0) {",
        "    return false;",
        "  }",
        "  for (int r = 0; r != mk::WIDTH; ++r) {",
        "    mg[r] = nw[r];",
        "  }",
        "  p = nw[mk::STATE_P];",
        "  // the Cauchy stress, P F^T / J, in TFEL's components (sqrt(2) on the shears)",
        "  const auto J = det(F1);",
        "  double t[3][3];",
        "  for (int i = 0; i != 3; ++i) {",
        "    for (int j = 0; j != 3; ++j) {",
        "      t[i][j] = 0;",
        "      for (int a = 0; a != 3; ++a) {",
        "        t[i][j] += pk1[3 * i + a] * f[3 * j + a] / J;",
        "      }",
        "    }",
        "  }",
        "  const auto r2 = std::sqrt(2.);",
        "  sig = StressStensor{t[0][0], t[1][1], t[2][2], r2 * t[0][1], r2 * t[0][2],",
        "                      r2 * t[1][2]};",
        "}",
        "",
        "@TangentOperator<DPK1_DF>{",
        f"  namespace mk = {space};",
        "  tfel::math::t2tot2<N, stress> k;",
        "  for (int I = 0; I != 9; ++I) {",
        "    for (int L = 0; L != 9; ++L) {",
        "      k(I, L) = K[9 * mk::ROWS[I] + mk::ROWS[L]];",
        "    }",
        "  }",
        "  Dt = k;",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _elastic_finite(material, name: str) -> str:
    """Hencky's elasticity in finite strains: the stress conjugate to the logarithmic
    strain linear in it (``@StrainMeasure Hencky``), isotropic (the Lamé coefficients) or
    anisotropic in the axes of the crystal (its orthotropic constants, MFront's order)."""
    lines = ["@DSL Default;", f"@Behaviour {name};", "@Author morphoGen;", "@Description {"]
    if material.stiffness is None:
        lam, mu = material.lame
        e, nu = _young_poisson(material)
        lines += [
            f'  "morphoGen {__version__}: isotropic Hencky elasticity (the space of the '
            f'logarithmic strain): E = {e:g}, nu = {nu:g}."',
            "}",
            "@StrainMeasure Hencky;",
            "@ProvidesSymmetricTangentOperator;",
            "",
            "@Integrator{",
            f"  constexpr auto lambda = {_real(lam)};",
            f"  constexpr auto mu = {_real(mu)};",
            *_hencky_strain(material),
            "  sig = lambda * trace(e) * StrainStensor::Id() + 2 * mu * e;",
            "  if (computeTangentOperator_) {",
            "    Dt = lambda * Stensor4::IxI() + 2 * mu * Stensor4::Id();",
            "  }",
            "}",
        ]
    else:
        constants = orthotropic_constants(material.stiffness)
        values = ", ".join(f"{k} = {v:g}" for k, v in zip(
            ("E1", "E2", "E3", "nu12", "nu23", "nu13", "G12", "G23", "G13"), constants,
            strict=True))  # fmt: skip
        lines += [
            f'  "morphoGen {__version__}: Hencky elasticity of a crystal (the space of the '
            f'logarithmic strain), in its axes: {values}."',
            "}",
            "@ModellingHypothesis Tridimensional;",
            "@OrthotropicBehaviour;",
            ABAQUS_AXES,
            "@StrainMeasure Hencky;",
            "@ProvidesSymmetricTangentOperator;",
            "@ComputeStiffnessTensor<UnAltered>{" + ", ".join(_real(c) for c in constants) + "};",
            "",
            "@Integrator{",
            *_hencky_strain(material),
            "  sig = D * e;",
            "  if (computeTangentOperator_) {",
            "    Dt = D;",
            "  }",
            "}",
        ]
    return "\n".join(lines) + "\n"


def _isotropic_hardening(r0: float, h: float, isotropic) -> list[str]:
    """The isotropic hardening ``R0 + H p + sum Q (1 - exp(-b p))`` of the brick: linear,
    and Voce's ``Rinf + (R0 - Rinf) exp(-b p)`` from ``R0 = 0`` to ``Rinf = Q``."""
    hardening = [f'    isotropic_hardening : "Linear" {{R0 : {_real(r0)}, H : {_real(h)}}},']
    for q, b in isotropic:
        hardening.append(f'    isotropic_hardening : "Voce" {{R0 : 0, Rinf : {_real(q)}, '
                         f"b : {_real(b)}}},")  # fmt: skip
    return hardening


def _vonmises(material, name: str, kinematics: str) -> str:
    pl = material.plasticity
    e, nu = _young_poisson(material)
    hardening = _isotropic_hardening(pl.yield_stress, pl.hardening, pl.isotropic)
    for c, d in pl.kinematic:
        if d > 0:
            hardening.append(f'    kinematic_hardening : "Armstrong-Frederick" {{C : {_real(c)}, '
                             f"D : {_real(d)}}},")  # fmt: skip
        else:
            hardening.append(f'    kinematic_hardening : "Prager" {{C : {_real(c)}}},')
    if pl.viscous:
        flow = "Norton"
        hardening.append(f"    K : {_real(pl.viscosity)},")
        hardening.append(f"    n : {_real(pl.exponent)},")
    else:
        flow = "Plastic"
    hardening[-1] = hardening[-1].rstrip(",")
    terms = [f"R0 = {pl.yield_stress:g}", f"H = {pl.hardening:g}"]
    terms += [f"Voce (Q, b) = ({q:g}, {b:g})" for q, b in pl.isotropic]
    terms += [f"kinematic (C, D) = ({c:g}, {d:g})" for c, d in pl.kinematic]
    terms += [f"Norton K = {pl.viscosity:g}, n = {pl.exponent:g}"] if pl.viscous else []
    if pl.hill is not None:
        from morphogen.homogenization.behaviours import HILL_COMPONENTS

        terms += [", ".join(f"h{c} = {h:g}" for c, h in zip(HILL_COMPONENTS, pl.hill,
                                                              strict=True))]  # fmt: skip
    space = (" in the space of the logarithmic strain (Hencky)" if kinematics == "finite"
             else " in small strains")  # fmt: skip
    criterion = '"Mises"'
    if pl.hill is not None:
        # Z-set's hill1 ... hill6 (components 11, 22, 33, 12, 23, 31) as TFEL's F ... N
        from morphogen.homogenization.zmat import _hill

        coefficients = _hill({f"hill{i}": h for i, h in enumerate(pl.hill, start=1)})
        criterion = (
            '"Hill" {' + ", ".join(f"{k} : {_real(v)}" for k, v in coefficients.items()) + "}"
        )
    law = "Hill" if pl.hill is not None else "von Mises"
    hooke = f"young_modulus : {_real(e)}, poisson_ratio : {_real(nu)}"
    if material.expands:
        values = _expansion_values(material)
        terms.append("thermal expansion " + ", ".join(f"{v:g}" for v in values))
        # finite strains: the mean coefficient whose stretch is morphoGen's exp(alpha dT)
        text = [_mean_expansion(v) if kinematics == "finite" else _real(v) for v in values]
        if len(values) == 1:
            hooke += f", thermal_expansion : {text[0]}"
        else:
            hooke += "".join(f", thermal_expansion{i} : {v}"
                             for i, v in enumerate(text, start=1))  # fmt: skip
    lines = [
        "@DSL Implicit;",
        f"@Behaviour {name};",
        "@Author morphoGen;",
        "@Description {",
        f'  "morphoGen {__version__}: {law} elasto-'
        f"{'visco' if pl.viscous else ''}plasticity{space}: E = {e:g}, nu = {nu:g}, "
        f'{", ".join(terms)}."',
        "}",
        "@Algorithm NewtonRaphson;",
        "@Epsilon 1.e-14;",
        "@Theta 1;",
    ]
    if kinematics == "finite":
        lines.append("@StrainMeasure Hencky;")
    if pl.hill is not None:
        # Hill's criterion is written in the axes of the material, those of the sample
        if isinstance(material.expansion, tuple):  # MFront asks it of orthotropic expansions
            lines.append("@ModellingHypothesis Tridimensional;")
        lines.append("@OrthotropicBehaviour;")
    lines += [
        "@Brick StandardElastoViscoPlasticity{",
        f'  stress_potential : "Hooke" {{{hooke}}},',
        f'  inelastic_flow : "{flow}" {{',
        f"    criterion : {criterion},",
        *hardening,
        "  }",
        "};",
    ]
    return "\n".join(lines) + "\n"


#: The threshold ``en`` of MFront's power law of the nucleation of exponent 0 that is
#: morphoGen's constant nucleation from ``p = 0`` (MFront's rate ``fn (p / en - 1)^m``
#: needs ``en > 0``; it integrates it from ``max(p, en)``: ``fn en`` short of ``A dp``)
_NUCLEATION_EN = 1e-200


def _porous_brick(pp) -> tuple[str, str | None] | None:
    """The porous criterion and the nucleation model (None: none) of MFront's brick that
    are exactly those of a porous plasticity, or None: its source then holds morphoGen's
    kernel (:func:`_porous_kernel`). The brick's ``GursonTvergaardNeedleman1982`` is GTN's
    criterion with its effective porosity ``f`` (MFront's ``f_c`` beyond any porosity) or
    Tvergaard & Needleman's coalescence (MFront's ``f* = f_c + delta (f - f_c)`` beyond
    ``f_c``, ``delta = (f_u - f_c) / (f_r - f_c)``, ``f_u = (q1 - sqrt(q1² - q3)) / q3``
    the ultimate porosity, ``1 / q1`` when ``q3 = q1²``: ``f_r`` the porosity of
    morphoGen's slope, ``fF`` when ``q3 = q1²``); its nucleation, morphoGen's constant
    nucleation ``A`` from ``p = 0`` (MFront's power law of exponent 0, ``"PowerLaw
    (strain)"``, its threshold :data:`_NUCLEATION_EN`, its maximum nucleated porosity 1).
    Not the brick's: Rousselier's criterion (MFront's ``RousselierTanguyBesson2002`` puts
    the matrix stress in place of morphoGen's constant ``σ1``: homogeneous of degree 1 in
    the stress), the elliptic one (none in the brick), the effective porosities of other
    tables, the coalescence with ``q3 > q1²`` (MFront's ``f*`` reaches ``f_u`` at
    ``f_r``, not real then), Chu & Needleman's Gaussian nucleation (MFront integrates the
    Gaussian exactly over a step, ``fN / 2 (erf((p + dp - eN) / (sqrt(2) sN)) - erf((p -
    eN) / (sqrt(2) sN)))``, morphoGen by backward Euler, ``A(p + dp) dp``), the exponential
    one (none), the thresholds of the nucleation (MFront's power law integrates its rate
    from its threshold ``en``, ``fn (p + dp - en)`` in the step that crosses it, morphoGen's
    backward Euler ``A dp``) and kinematic hardening (MFront 5.1's brick: "kinematic
    hardening rules are not supported when coupled with a porosity evolution")."""
    nuc = pp.nucleation
    if pp.criterion != "gurson" or pp.kinematic:
        return None
    if nuc is not None and (nuc["type"] != "constant" or nuc["yield"] > 0):
        return None
    q1, q2, q3 = pp.q1, pp.q2, pp.q3
    table = pp.effective
    if not table:
        fc, fr = 1.0, 2.0  # the porosity never reaches f_c: f* = f
    else:
        on_identity = len(table) == 3 and all(abs(x - y) <= 1e-12 for x, y in table[:2])
        if not (on_identity and table[0][0] == 0.0 and table[2][1] > table[1][1]):
            return None
        (_, _), (fc, _), (ff, fstar) = table
        if abs(q3 - q1 * q1) <= 1e-12 * q1 * q1 and abs(fstar - 1 / q1) <= 1e-12 / q1:
            fr = ff
        elif q3 > q1 * q1:
            return None
        else:
            fu = (q1 - math.sqrt(q1 * q1 - q3)) / q3
            fr = fc + (fu - fc) * (ff - fc) / (fstar - fc)
    values = {"f_c": fc, "f_r": fr, "q_1": q1, "q_2": q2, "q_3": q3}
    criterion = ('"GursonTvergaardNeedleman1982" {'
                 + ", ".join(f"{k} : {_real(v)}" for k, v in values.items()) + "}")  # fmt: skip
    if nuc is None:
        return criterion, None
    return criterion, (f'"PowerLaw (strain)" {{fn : {_real(nuc["A"])}, '
                       f"en : {_real(_NUCLEATION_EN)}, m : 0, fmax : 1}}")  # fmt: skip


_CRITERION_NAMES = {"gurson": "Gurson-Tvergaard-Needleman", "rousselier": "Rousselier",
                    "elliptic": "elliptic"}  # fmt: skip


def _porous_terms(pp) -> list[str]:
    """The parameters of a porous plasticity, for the descriptions of its sources."""

    def pairs(table):
        return ", ".join(f"({x:g}, {y:g})" for x, y in table)

    terms = [f"f0 = {pp.porosity:g}"]
    if pp.criterion == "gurson":
        terms += [f"q1 = {pp.q1:g}", f"q2 = {pp.q2:g}", f"q3 = {pp.q3:g}"]
        terms += [f"effective porosity {pairs(pp.effective)}"] if pp.effective else []
    elif pp.criterion == "rousselier":
        terms += [f"sigma1 = {pp.sigma1:g}", f"D = {pp.D:g}", f"q2 = {pp.q2:g}"]
    else:
        terms += [f"C {pairs(pp.C)}", f"F {pairs(pp.F)}"]
    terms += [f"R0 = {pp.yield_stress:g}", f"H = {pp.hardening:g}"]
    terms += [f"Voce (Q, b) = ({q:g}, {b:g})" for q, b in pp.isotropic]
    terms += [f"kinematic (C, D) = ({c:g}, {d:g})" for c, d in pp.kinematic]
    if pp.viscous:
        terms.append(f"Norton K = {pp.viscosity:g}, n = {pp.exponent:g}")
    nuc = pp.nucleation
    if nuc is not None:
        values = ", ".join(f"{k} = {v:g}" for k, v in nuc.items() if k not in ("type", "yield"))
        start = f" from p = {nuc['yield']:g}" if nuc["yield"] > 0 else ""
        terms.append(f"{nuc['type']} nucleation {values}{start}")
    return terms


def _porous(material, name: str, kinematics: str) -> str:
    """The source of a porous plasticity of the brick (:func:`_porous_brick`): the porosity
    in its implicit system (``standard implicit scheme``), its initial value ``f0`` until
    the matrix flows."""
    pp = material.porous_plasticity
    e, nu = _young_poisson(material)
    criterion, nucleation = _porous_brick(pp)
    hardening = _isotropic_hardening(pp.yield_stress, pp.hardening, pp.isotropic)
    terms = _porous_terms(pp)
    if pp.viscous:
        flow = "Norton"
        hardening += [f"    K : {_real(pp.viscosity)},", f"    n : {_real(pp.exponent)},"]
    else:
        flow = "Plastic"
    hardening[-1] = hardening[-1].rstrip(",")
    hooke = f"young_modulus : {_real(e)}, poisson_ratio : {_real(nu)}"
    if material.expands:
        alpha = _expansion_values(material)[0]
        terms.append(f"thermal expansion {alpha:g}")
        text = _mean_expansion(alpha) if kinematics == "finite" else _real(alpha)
        hooke += f", thermal_expansion : {text}"
    space = (" in the space of the logarithmic strain (Hencky)" if kinematics == "finite"
             else " in small strains")  # fmt: skip
    evolution = ['    algorithm : "standard implicit scheme"']
    if nucleation is not None:
        evolution[-1] += ","
        evolution.append(f"    nucleation_model : {nucleation}")
    lines = [
        "@DSL Implicit;",
        f"@Behaviour {name};",
        "@Author morphoGen;",
        "@Description {",
        f'  "morphoGen {__version__}: porous elasto-{"visco" if pp.viscous else ""}plasticity '
        f"(Gurson-Tvergaard-Needleman){space}: E = {e:g}, nu = {nu:g}, "
        f'{", ".join(terms)}."',
        "}",
        "@Algorithm NewtonRaphson;",
        "@Epsilon 1.e-14;",
        "@Theta 1;",
    ]
    if kinematics == "finite":
        lines.append("@StrainMeasure Hencky;")
    lines += [
        "@Brick StandardElastoViscoPlasticity{",
        f'  stress_potential : "Hooke" {{{hooke}}},',
        f'  inelastic_flow : "{flow}" {{',
        f"    criterion : {criterion},",
        *hardening,
        "  },",
        "  porosity_evolution : {",
        *evolution,
        "  }",
        "};",
        "",
        "// the porosity is f0 until the matrix flows (it changes with the plastic strain",
        "// only): the natural state of the internal variables is zero, as in the other",
        "// exports of morphoGen",
        "@InitLocalVariables<AtBeginning>{",
        "  if (this->p <= 0) {",
        f"    this->f = {_real(pp.porosity)};",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _porous_kernel(material, name: str, kinematics: str) -> str:
    """The source of a porous plasticity outside the brick (:func:`_porous_brick`):
    morphoGen's own integration of a point (``porous_update``, the C port of its kernel,
    with its row of parameters), in small strains or in the space of the logarithmic
    strain (``@StrainMeasure Hencky``: the stress conjugate to it), the consistent tangent
    of the kernel the operator, morphoGen's thermal strain ``alpha (T + dT - 293.15)``
    (:func:`_thermal_strain`). Its auxiliary state variables: the elastic strain (small
    strains), the internal variables of the kernel (``MorphogenState``: the plastic strain,
    ``p``, the growth of the porosity, the nucleated porosity, the matrix stress, the
    backstresses), the cumulated plastic strain and the porosity; their natural state,
    zero, is the porosity ``f0``."""
    from morphogen.homogenization.porous import state_width

    pp = material.porous_plasticity
    width = state_width(len(pp.kinematic))
    row = pp.params(*material.lame)
    e, nu = _young_poisson(material)
    terms = _porous_terms(pp)
    if material.expands:
        terms.append(f"thermal expansion {_expansion_values(material)[0]:g}")
    small = kinematics == "small"
    where = " in small strains" if small else " in the space of the logarithmic strain (Hencky)"
    space = f"morphogen_{name}"
    lines = [
        "@DSL Default;",
        f"@Behaviour {name};",
        "@Author morphoGen;",
        "@Description {",
        f'  "morphoGen {__version__}: porous elasto-{"visco" if pp.viscous else ""}plasticity '
        f"({_CRITERION_NAMES[pp.criterion]}){where}: E = {e:g}, nu = {nu:g}, "
        f"{', '.join(terms)}; integrated by morphoGen's kernel (porous_update).\"",
        "}",
        "@ModellingHypothesis Tridimensional;",
        *([] if small else ["@StrainMeasure Hencky;"]),
        "@ProvidesSymmetricTangentOperator;",
    ]
    if small:
        lines += ["@AuxiliaryStateVariable StrainStensor eel;",
                  'eel.setGlossaryName("ElasticStrain");']  # fmt: skip
    lines += [
        f"@AuxiliaryStateVariable real mg[{width}];",
        'mg.setEntryName("MorphogenState");',
        "@AuxiliaryStateVariable strain p;",
        'p.setGlossaryName("EquivalentPlasticStrain");',
        "@AuxiliaryStateVariable real f;",
        'f.setGlossaryName("Porosity");',
        "",
        "@Includes{",
        "// morphoGen's kernels of the points and the parameters of the material (MFront",
        "// includes this block in several headers of one translation unit)",
        f"#ifndef {space.upper()}_KERNELS",
        f"#define {space.upper()}_KERNELS",
        "#include <cmath>",
        "#include <math.h>",
        f"namespace {space} {{",
        _kernels(1, width),
        "// the row of parameters of the porous point (morphogen.homogenization.porous)",
        f"constexpr double params[{row.size}] = {_array(row)};",
        "// morphoGen's components (11 22 33 23 31 12) of TFEL's (11 22 33 12 13 23)",
        "constexpr int VOIGT[6] = {0, 1, 2, 5, 4, 3};",
        f"}}  // end of namespace {space}",
        "#endif",
        "}",
        "",
        "@Integrator{",
        f"  namespace mk = {space};",
        "  // TFEL's components of tensors: sqrt(2) times the shears; morphoGen's strain: the",
        "  // engineering shears",
        "  const auto r2 = std::sqrt(2.);",
        "  const StrainStensor e = eto + deto;",
        "  double e6[6], old[mk::WIDTH], nw[mk::WIDTH], s6[6], k6[36];",
        "  for (int i = 0; i != 6; ++i) {",
        "    e6[mk::VOIGT[i]] = (i < 3 ? 1 : r2) * e[i];",
        "  }",
        "  for (int r = 0; r != mk::WIDTH; ++r) {",
        "    old[r] = mg[r];",
        "  }",
        *_thermal_strain(material, "eth"),
        "  if (mk::porous_update(mk::params, e6, eth, dt, old, nw, s6, k6) != 0) {",
        "    return false;",
        "  }",
        "  for (int r = 0; r != mk::WIDTH; ++r) {",
        "    mg[r] = nw[r];",
        "  }",
        "  p = nw[mk::STATE_P];",
        "  f = mk::params[mk::P_F0] + nw[mk::STATE_FG] + nw[mk::STATE_FN];",
        "  for (int i = 0; i != 6; ++i) {",
        "    const auto c = i < 3 ? 1 : r2;",
        "    sig[i] = c * s6[mk::VOIGT[i]];",
    ]
    if small:  # the strain without the thermal and the plastic ones (the kernel's 0 ... 5)
        lines.append("    eel[i] = e[i] - c * (eth[mk::VOIGT[i]] + nw[mk::VOIGT[i]]);")
    lines += [
        "  }",
        "  if (computeTangentOperator_) {",
        "    for (int i = 0; i != 6; ++i) {",
        "      for (int j = 0; j != 6; ++j) {",
        "        const auto c = (i < 3 ? 1 : r2) * (j < 3 ? 1 : r2);",
        "        Dt(i, j) = c * k6[6 * mk::VOIGT[i] + mk::VOIGT[j]];",
        "      }",
        "    }",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _array(values) -> str:
    return "{" + ", ".join(_real(v) for v in values) + "}"


def _crystal(material, name: str) -> str:
    cp = material.crystal_plasticity
    normals, directions = cp.systems()
    ns = len(normals)
    par = cp.params_array()  # tau0, Q, b, C, D, K, n of each system
    h = cp.interaction_matrix()
    m = 0.5 * (directions[:, :, None] * normals[:, None, :]
               + normals[:, :, None] * directions[:, None, :])  # fmt: skip
    # TFEL's components of the Schmid tensors: 11, 22, 33, then sqrt(2) times 12, 13, 23
    tfel = np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], _SQRT2 * m[:, 0, 1],
                     _SQRT2 * m[:, 0, 2], _SQRT2 * m[:, 1, 2]], axis=1)  # fmt: skip
    constants = orthotropic_constants(_stiffness(material))
    families = "; ".join(
        f"{fam} (" + ", ".join(f"{k} = {v:g}" for k, v in
                               zip(("tau0", "Q", "b", "C", "D", "K", "n"), values,
                                   strict=True)) + ")"
        for fam, values in cp.families
    )  # fmt: skip
    space = f"morphogen_{name}"
    expansion = []  # one coefficient, or the principal values in the axes of the crystal
    if material.expands:
        values = _expansion_values(material)
        text = _array(values) if len(values) > 1 else _real(values[0])
        expansion = [f"@ComputeThermalExpansion {text};"]
    rows = ",\n".join("    " + _array(row) for row in tfel)
    hrows = ",\n".join("    " + _array(row) for row in h)
    lines = [
        "@DSL Implicit;",
        f"@Behaviour {name};",
        "@Author morphoGen;",
        "@Description {",
        f'  "morphoGen {__version__}: Meric-Cailletaud crystal plasticity in small strains, '
        f"{ns} slip systems: {families}; the elasticity and the slip systems in the axes "
        'of the crystal."',
        "}",
        "@ModellingHypothesis Tridimensional;",
        "@OrthotropicBehaviour;",
        "@Algorithm NewtonRaphson_NumericalJacobian;",
        "@PerturbationValueForNumericalJacobianComputation 1.e-10;",
        "@Epsilon 1.e-13;",
        "@Theta 1;",
        "@Brick StandardElasticity;",
        "@ComputeStiffnessTensor<UnAltered>{" + ", ".join(_real(c) for c in constants) + "};",
        *expansion,
        f"@IntegerConstant Nss = {ns};",
        "",
        "@Includes{",
        "// morphoGen's slip systems, the parameters of each system and the interaction matrix",
        "// (MFront includes this block in several headers of one translation unit)",
        f"#ifndef {space.upper()}_SYSTEMS",
        f"#define {space.upper()}_SYSTEMS",
        f"namespace {space} {{",
        "  // Schmid tensors, TFEL's components: 11, 22, 33, then sqrt(2) times 12, 13, 23",
        f"  constexpr double m[{ns}][6] = {{",
        rows,
        "  };",
    ]
    for k, key in enumerate(("tau0", "Q", "b", "C", "D", "K", "n")):
        lines.append(f"  constexpr double {key}[{ns}] = {_array(par[:, k])};")
    lines += [
        f"  constexpr double h[{ns}][{ns}] = {{",
        hrows,
        "  };",
        f"}}  // end of namespace {space}",
        "#endif",
        "}",
        "",
        "@StateVariable strain g[Nss];",
        'g.setEntryName("Slip");',
        "@AuxiliaryStateVariable strain v[Nss];",
        'v.setEntryName("CumulatedSlip");',
        "@AuxiliaryStateVariable strain a[Nss];",
        'a.setEntryName("KinematicVariable");',
        "@AuxiliaryStateVariable strain p;",
        'p.setGlossaryName("EquivalentPlasticStrain");',
        "",
        "@Integrator{",
        f"  namespace cs = {space};",
        "  for (unsigned short i = 0; i != Nss; ++i) {",
        "    auto tau = real(0);",
        "    for (unsigned short k = 0; k != 6; ++k) {",
        "      feel[k] += dg[i] * cs::m[i][k];",
        "      tau += sig[k] * cs::m[i][k];",
        "    }",
        "    auto r = real(cs::tau0[i]);",
        "    if (cs::Q[i] != 0) {",
        "      for (unsigned short j = 0; j != Nss; ++j) {",
        "        r += cs::Q[i] * cs::h[i][j] * (1 - exp(-cs::b[i] * (v[j] + abs(dg[j]))));",
        "      }",
        "    }",
        "    const auto x = cs::C[i] * (a[i] + dg[i]) / (1 + cs::D[i] * abs(dg[i]));",
        "    const auto f = abs(tau - x) - r;",
        "    const auto vp = (f > 0) ? pow(f / cs::K[i], cs::n[i]) : real(0);",
        "    fg[i] -= dt * vp * ((tau - x >= 0) ? 1 : -1);",
        "  }",
        "}",
        "",
        "@UpdateAuxiliaryStateVariables{",
        f"  namespace cs = {space};",
        "  auto dep = Stensor(real(0));",
        "  for (unsigned short i = 0; i != Nss; ++i) {",
        "    a[i] = (a[i] + dg[i]) / (1 + cs::D[i] * abs(dg[i]));",
        "    v[i] += abs(dg[i]);",
        "    for (unsigned short k = 0; k != 6; ++k) {",
        "      dep[k] += dg[i] * cs::m[i][k];",
        "    }",
        "  }",
        "  p += sqrt(2 * (dep | dep) / 3);",
        "}",
    ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class UserBehaviour:
    """The MFront behaviour of a material (its ``mfront`` key) as the exports to other
    codes need it: its source, name and symmetry, the names of its material properties
    in the order of their declaration, the number of its internal state variables, its
    strain measure (finite strains: Hencky, GreenLagrange; None: the deformation
    gradient) and whether Abaqus gives its axes through its first six state variables
    (``@AbaqusOrthotropyManagementPolicy MFront``)."""

    source: Path
    name: str
    orthotropic: bool
    properties: tuple[str, ...]
    internal_variables: int
    strain_measure: str | None = None
    abaqus_axes: bool = False


def _source_text(path: Path) -> str:
    """The text of an MFront source without its comments."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def user_behaviour(material, kinematics: str = "small") -> UserBehaviour:
    """The behaviour of an MFront material for the exports (:class:`UserBehaviour`), as
    MGIS describes it (its source compiled with the generic interface, which needs MGIS
    and MFront): a behaviour of the kinematics of the export (finite strains: written in
    the deformation gradient, or with Hencky's or Green-Lagrange's strain measure)
    without external state variables but the temperature, given by its source (the
    exports compile it for their interface)."""
    from morphogen.homogenization.mfront import load_behaviour

    if kinematics not in ("small", "finite"):
        raise ValueError("kinematics: 'small' or 'finite'")
    spec = material.mfront
    if spec.source is None:
        raise ValueError(
            "an MFront material given by its library: the exports compile its source with "
            "the interface of their code"
        )
    others = [name for name, _ in spec.external_state_variables if name != "Temperature"]
    if others:
        raise ValueError(f"mfront: the external state variables {others} are not exported")
    try:
        law = load_behaviour(spec, finite=kinematics == "finite")
    except ValueError as exc:
        raise ValueError(
            f"mfront: the exports in {kinematics} strains take {kinematics} strain behaviours "
            f"({exc})"
        ) from None
    text = _source_text(Path(spec.source))
    found = _STRAIN_MEASURE.search(text)
    measure = found.group(1) if found and kinematics == "finite" else None
    if measure is not None and measure not in _UMAT_FINITE:
        raise ValueError(
            f"mfront: {spec.behaviour}: the strain measure {measure} in finite strains (the "
            f"exports take {', '.join(_UMAT_FINITE)}, or the deformation gradient)"
        )
    names = tuple(v.name for v in law.behaviour.mps)
    axes = law.orthotropic and _ABAQUS_POLICY.search(text) is not None
    return UserBehaviour(Path(spec.source), spec.behaviour, law.orthotropic, names, law.n_isv,
                         measure, axes)  # fmt: skip


def write_parameters(directory, behaviour: str, parameters) -> Path | None:
    """``<behaviour>-parameters.txt``: the values of the parameters of an MFront behaviour,
    which its compiled code reads from the working directory of the run (none: no
    file)."""
    if not parameters:
        return None
    path = Path(directory) / f"{behaviour}-parameters.txt"
    path.write_text("".join(f"{name} {_real(value)}\n" for name, value in parameters), "utf-8")
    return path


def mfront_source(material, name: str, *, kinematics: str = "small") -> str:
    """The MFront source of the behaviour ``name`` of a material (see the module): von
    Mises plasticity, porous plasticity (the brick, or morphoGen's kernel), crystal
    plasticity and the learned laws (:mod:`morphogen.learning.mfront`) in small or finite
    strains, and in finite strains Hencky's elasticity of the elastic materials."""
    if not _NAME.fullmatch(name):
        raise ValueError(f"mfront: the name of a behaviour is an identifier, got {name!r}")
    kind = _check(material, kinematics)
    if kind == "learned":
        from morphogen.learning.mfront import learned_source

        return learned_source(material, name, kinematics=kinematics)
    if kind == "crystal_finite":
        return _crystal_finite(material, name)
    if kind == "crystal":
        return _crystal(material, name)
    if kind == "elastic_finite":
        return _elastic_finite(material, name)
    if kind == "porous":
        return _porous(material, name, kinematics)
    if kind == "porous_kernel":
        return _porous_kernel(material, name, kinematics)
    return _vonmises(material, name, kinematics)


def write_mfront(material, path, *, name: str | None = None, kinematics: str = "small") -> Path:
    """Write the MFront source of a material (:func:`mfront_source`); the name of the
    behaviour is the stem of ``path`` by default."""
    path = Path(path)
    if path.suffix != ".mfront":
        raise ValueError(f"mfront: a .mfront file, got {path.name!r}")
    text = mfront_source(material, name or path.stem, kinematics=kinematics)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
