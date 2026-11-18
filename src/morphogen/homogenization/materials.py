"""Materials of the phases: linear (thermo-)elasticity and the transport phenomena of one
scalar potential (heat conduction, electrical conduction, dielectric and magnetic
polarization, diffusion), isotropic or anisotropic (crystals), and voids.

Every transport phenomenon is a flux proportional to the gradient of a potential,
``flux = -K grad(potential)`` without sources (:data:`TRANSPORT`): Fourier's law (``q = -k
grad T``, the ``conductivity``), Ohm's law (``j = -sigma grad V``, the
``electrical_conductivity``), electrostatics (``D = -eps grad V``, the ``permittivity``),
magnetostatics without currents (``B = -mu grad psi``, the ``permeability``) and Fick's law
(``J = -D grad c``, the ``diffusivity``). They share their solvers, their boundary
conditions and their bounds; a phase whose coefficient is zero is inert for that physics
(an insulator), as a void is for all.

Thermal expansion makes the stress ``sigma = D (eps - alpha dT)`` for a change of
temperature ``dT`` from the stress-free state; a crystal's expansion tensor turns with it.

An isotropic elastic material may also yield: von Mises elasto-(visco)plasticity with
isotropic and kinematic hardening (:class:`~morphogen.homogenization.behaviours.Plasticity`,
for the loading paths of :mod:`morphogen.homogenization.nonlinear`), or porous
elasto-(visco)plasticity (Gurson-Tvergaard-Needleman, Rousselier, the elliptic criterion
of compaction: :class:`~morphogen.homogenization.porous.PorousPlasticity`); the linear
properties use its elasticity. A crystal may slip on its slip systems: the crystal
plasticity of Méric and Cailletaud (:class:`~morphogen.homogenization.crystal.
CrystalPlasticity`), which turns with the orientation of each grain.

Elasticity matrices use the Voigt order ``(11, 22, 33, 23, 31, 12)`` with engineering
shear strains, so that ``sigma = D eps`` (:mod:`morphogen.homogenization.elements`). The
matrices of a crystal are given in its own axes; a crystal orientation (Bunge Euler
angles ``(phi1, Phi, phi2)`` in degrees, :mod:`morphogen.orientations`) turns them into
the axes of the sample: ``C_ijkl = g_pi g_qj g_rk g_sl C'_pqrs`` with ``crystal = g @
sample``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from morphogen.homogenization.behaviours import Plasticity
from morphogen.homogenization.crystal import CrystalPlasticity
from morphogen.homogenization.porous import PorousPlasticity

log = logging.getLogger("morphogen")

#: The transport phenomena of one scalar potential, ``flux = -K grad(potential)``: the
#: name of each physics and the parameter of :class:`Material` that holds its ``K``
#: (``thermal``: the ``conductivity``, or the tensor ``conductivities`` of a crystal)
TRANSPORT = {
    "thermal": "conductivity",
    "electrical": "electrical_conductivity",
    "dielectric": "permittivity",
    "magnetic": "permeability",
    "diffusion": "diffusivity",
}
#: The potential, the flux and the law of each transport phenomenon (for the messages and
#: the documentation)
TRANSPORT_LAWS = {
    "thermal": ("temperature", "heat flux", "Fourier's law q = -k grad T"),
    "electrical": ("electric potential", "current density", "Ohm's law j = -sigma grad V"),
    "dielectric": ("electric potential", "electric displacement", "D = -eps grad V"),
    "magnetic": ("magnetic scalar potential", "magnetic induction", "B = -mu grad psi"),
    "diffusion": ("concentration", "mass flux", "Fick's law J = -D grad c"),
}
#: The physics of the linear problems: elasticity and the transport phenomena
PHYSICS = ("elastic", *TRANSPORT)
#: The behaviours of :mod:`morphogen.homogenization.laws` of a material
_ANALYTIC = ("viscoelasticity", "hyperelasticity", "damage")


def transport_physics(physics: str) -> bool:
    """Whether ``physics`` is a transport phenomenon of :data:`TRANSPORT` (a scalar
    potential), not elasticity; an unknown name is an error."""
    if physics not in PHYSICS:
        raise ValueError(f"physics: one of {PHYSICS}, got {physics!r}")
    return physics != "elastic"


#: Voigt index of each pair of tensor indices
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
_VOIGT_INDEX = np.empty((3, 3), dtype=np.int64)
for _k, (_i, _j) in enumerate(_VOIGT_PAIRS):
    _VOIGT_INDEX[_i, _j] = _VOIGT_INDEX[_j, _i] = _k
_I0 = np.array([i for i, _ in _VOIGT_PAIRS])
_I1 = np.array([j for _, j in _VOIGT_PAIRS])


def voigt_to_tensor4(d) -> np.ndarray:
    """Fourth-order tensor(s) ``C_ijkl`` of Voigt matrices (..., 6, 6)."""
    d = np.asarray(d, dtype=float)
    return d[..., _VOIGT_INDEX[:, :, None, None], _VOIGT_INDEX[None, None, :, :]]


def tensor4_to_voigt(c) -> np.ndarray:
    """Voigt matrices (..., 6, 6) of fourth-order tensors with the minor symmetries."""
    c = np.asarray(c, dtype=float)
    return c[..., _I0[:, None], _I1[:, None], _I0[None, :], _I1[None, :]]


def rotate_stiffness(d, euler) -> np.ndarray:
    """Elasticity matrix of a crystal (crystal axes) in the sample axes, for one
    orientation (shape (6, 6)) or several (shape (n, 6, 6))."""
    from morphogen.orientations import euler_to_matrix

    g = euler_to_matrix(euler)
    c = np.einsum("npi,nqj,nrk,nsl,pqrs->nijkl", g, g, g, g, voigt_to_tensor4(d), optimize=True)
    out = tensor4_to_voigt(c)
    out = 0.5 * (out + np.swapaxes(out, -1, -2))
    return out[0] if np.ndim(euler) == 1 else out


def rotate_conductivity(k, euler) -> np.ndarray:
    """Conductivity tensor of a crystal in the sample axes (``g^T K g``)."""
    from morphogen.orientations import euler_to_matrix

    g = euler_to_matrix(euler)
    out = np.einsum("npi,pq,nqj->nij", g, np.asarray(k, dtype=float), g)
    out = 0.5 * (out + np.swapaxes(out, -1, -2))
    return out[0] if np.ndim(euler) == 1 else out


def isotropic_matrix(lam: float, mu: float) -> np.ndarray:
    """Elasticity matrix (6 x 6) of an isotropic material."""
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[[0, 1, 2], [0, 1, 2]] += 2.0 * mu
    d[[3, 4, 5], [3, 4, 5]] = mu
    return d


def voigt_average(d) -> tuple[float, float]:
    """Bulk and shear moduli of the Voigt average over uniform orientations."""
    d = np.asarray(d, dtype=float)
    a = d[0, 0] + d[1, 1] + d[2, 2]
    b = d[1, 2] + d[2, 0] + d[0, 1]
    c = d[3, 3] + d[4, 4] + d[5, 5]
    return float((a + 2.0 * b) / 9.0), float((a - b + 3.0 * c) / 15.0)


def reuss_average(d) -> tuple[float, float]:
    """Bulk and shear moduli of the Reuss average over uniform orientations."""
    s = np.linalg.inv(np.asarray(d, dtype=float))
    a = s[0, 0] + s[1, 1] + s[2, 2]
    b = s[1, 2] + s[2, 0] + s[0, 1]
    c = s[3, 3] + s[4, 4] + s[5, 5]
    return float(1.0 / (a + 2.0 * b)), float(15.0 / (4.0 * a - 4.0 * b + 3.0 * c))


def _expansion(value, name: str = "expansion"):
    """Thermal expansion (or the swelling ``name``) as one coefficient (isotropic) or a
    3 x 3 tensor (tuple)."""
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite coefficients")
    if a.ndim == 0:
        return float(a)
    if a.shape == (3,):
        a = np.diag(a)
    if a.shape != (3, 3) or not np.allclose(a, a.T, atol=1e-12 * max(np.abs(a).max(), 1e-300)):
        raise ValueError(f"{name}: one coefficient, three principal values or a 3 x 3 tensor")
    if np.allclose(a, a[0, 0] * np.eye(3), rtol=0, atol=1e-15 * abs(a[0, 0])):
        return float(a[0, 0])
    return tuple(tuple(float(v) for v in row) for row in 0.5 * (a + a.T))


def _matrix(value, shape, name) -> tuple:
    m = np.asarray(value, dtype=float)
    if m.shape != shape or not np.all(np.isfinite(m)):
        raise ValueError(f"{name}: expected a finite {shape[0]} x {shape[1]} matrix")
    if not np.allclose(m, m.T, rtol=1e-10, atol=1e-12 * float(np.abs(m).max())):
        raise ValueError(f"{name}: the matrix must be symmetric")
    if np.linalg.eigvalsh(0.5 * (m + m.T)).min() <= 0:
        raise ValueError(f"{name}: the matrix must be positive definite")
    return tuple(tuple(float(v) for v in row) for row in 0.5 * (m + m.T))


def _transport(value, name: str):
    """A transport coefficient: one value (isotropic; zero: a phase inert for that
    physics, an insulator), or a symmetric positive definite 3 x 3 tensor in the axes of
    the crystal (three principal values, or the tensor), kept as a tuple."""
    k = np.asarray(value, dtype=float)
    if k.shape == (3,):
        k = k[0] if np.allclose(k, k[0], rtol=1e-12, atol=0.0) else np.diag(k)
    if np.ndim(k) == 0:
        if not (math.isfinite(k) and k >= 0):
            raise ValueError(f"{name}: must be non-negative, got {value!r}")
        return float(k)
    return _matrix(k, (3, 3), name)


def _index(value):
    """A complex refractive index ``n + i k``: a number, a pair ``(n, k)`` (the form of
    the configurations), or one of them per wavelength; kept as a complex number or a
    tuple of them."""
    a = np.asarray(value)
    if a.dtype.kind not in "iufc":
        raise ValueError("refractive_index: n, (n, k), or a list of them per wavelength")
    if not np.iscomplexobj(a):
        if a.ndim == 2 and a.shape[1] == 2:  # pairs (n, k) per wavelength
            a = a[:, 0] + 1j * a[:, 1]
        elif a.shape == (2,):  # one pair
            a = a[0] + 1j * a[1]
    a = np.asarray(a, dtype=complex)
    transparent = np.all(a.imag == 0)
    if a.ndim > 1 or not np.all(np.isfinite(a)) or (transparent and np.any(a.real <= 0)):
        raise ValueError(
            "refractive_index: n, (n, k), or a list of pairs (n, k) per wavelength (finite, "
            "n positive for a transparent phase)"
        )
    return complex(a) if a.ndim == 0 else tuple(complex(v) for v in a)


def lame(young: float, poisson: float) -> tuple[float, float]:
    """Lame coefficients (lambda, mu) from Young's modulus and Poisson's ratio."""
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return lam, mu


def bulk_shear(young: float, poisson: float) -> tuple[float, float]:
    """Bulk and shear moduli (k, mu) from Young's modulus and Poisson's ratio."""
    return young / (3.0 * (1.0 - 2.0 * poisson)), young / (2.0 * (1.0 + poisson))


def young_poisson(bulk: float, shear: float) -> tuple[float, float]:
    """Young's modulus and Poisson's ratio from the bulk and shear moduli."""
    young = 9.0 * bulk * shear / (3.0 * bulk + shear)
    poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))
    return young, poisson


@dataclass(frozen=True)
class Material:
    """Linear material of a phase, or a void.

    Isotropic materials: ``young`` and ``poisson`` define the elastic behaviour,
    ``conductivity`` the thermal one; a material may define either or both. Crystals:
    ``stiffness``, the 6 x 6 elasticity matrix in the axes of the crystal (see
    :meth:`cubic`, :meth:`hexagonal`, :meth:`anisotropic`), and ``conductivities``, the
    3 x 3 conductivity tensor in the same axes (or ``conductivity`` when it is isotropic).
    ``expansion`` is the thermal expansion: one coefficient, or (crystals) principal
    values or a 3 x 3 tensor in the axes of the crystal. In a polycrystal, each grain
    turns them by its orientation. ``swelling`` is likewise the strain of a unit content
    of a species the phase absorbs (moisture, a solvent, lithium: the hygroscopic or
    chemical expansion), for the effective swelling of
    :mod:`morphogen.homogenization.poroelastic`. ``density`` (optional) gives the
    effective density and specific properties. The other transport phenomena
    (:data:`TRANSPORT`) each have their coefficient: ``electrical_conductivity`` (Ohm),
    ``permittivity`` (dielectrics), ``permeability`` (magnetostatics) and ``diffusivity``
    (Fick), one value or, in the
    axes of the crystal, three principal values or a 3 x 3 tensor; zero makes the phase
    inert for that physics (an insulator in a conducting composite), as a void is for
    all. ``refractive_index``: the complex refractive index ``n + i k`` of the phase at
    optical frequencies (a number, a pair ``(n, k)``, or one pair per wavelength), for the
    effective index of :mod:`morphogen.homogenization.harmonic`. ``coupling`` holds the
    coefficients of the strong couplings between these physics
    (:class:`~morphogen.homogenization.coupled.Coupling`: piezoelectric, piezomagnetic,
    magnetoelectric, Seebeck and pyroelectric coefficients; a mapping of them), for the
    coupled problems of :mod:`morphogen.homogenization.coupled`. ``viscoelasticity``
    (a generalized Maxwell model), ``hyperelasticity`` (a strain energy, in finite strains)
    and ``damage`` (the isotropic damage of quasi-brittle solids, Mazars' law of
    concrete) give an isotropic
    material one of the behaviours of :mod:`morphogen.homogenization.laws` along loading
    paths (a viscoelasticity and a hyperelasticity together: finite viscoelasticity; a
    ``damage`` of Lemaitre or of Kachanov goes with a ``plasticity``, whose effective
    stress it degrades).
    ``plasticity`` (isotropic materials) makes it yield along
    loading paths, ``porous_plasticity`` (isotropic materials) makes it a porous plastic
    material, ``crystal_plasticity`` makes it slip on the systems of its lattice,
    ``mfront`` gives it an MFront behaviour along them
    (:class:`~morphogen.homogenization.mfront.MFrontBehaviour`; its elastic parameters
    remain those of the linear problems and of the reference media) and ``learned`` a law
    learned from data (:class:`~morphogen.learning.behaviour.LearnedBehaviour`; likewise).
    A void carries neither stress nor flux.
    """

    young: float | None = None
    poisson: float | None = None
    conductivity: float | None = None
    void: bool = False
    stiffness: tuple | None = None
    conductivities: tuple | None = None
    symmetry: str = "isotropic"
    expansion: float | tuple | None = None
    density: float | None = None
    plasticity: Plasticity | None = None
    crystal_plasticity: CrystalPlasticity | None = None
    mfront: object | None = None
    porous_plasticity: PorousPlasticity | None = None
    learned: object | None = None
    electrical_conductivity: float | tuple | None = None
    permittivity: float | tuple | None = None
    permeability: float | tuple | None = None
    diffusivity: float | tuple | None = None
    refractive_index: complex | tuple | None = None
    coupling: object | None = None
    viscoelasticity: object | None = None
    hyperelasticity: object | None = None
    damage: object | None = None
    swelling: float | tuple | None = None

    def __post_init__(self) -> None:
        if self.hyperelasticity is not None and self.young is None and not self.void:
            self._elasticity_of_the_law()
        others = [name for name in TRANSPORT.values() if name != "conductivity"]
        params = (self.young, self.poisson, self.conductivity, self.stiffness,
                  self.conductivities, self.refractive_index,
                  *(getattr(self, name) for name in others))  # fmt: skip
        if self.density is not None:
            if not (math.isfinite(self.density) and self.density >= 0):
                raise ValueError(f"density: must be non-negative, got {self.density!r}")
            object.__setattr__(self, "density", float(self.density))
        if self.void:
            if any(v is not None for v in params) or self.expansion is not None:
                raise ValueError("a void has no elastic or thermal parameters")
            if self.swelling is not None:
                raise ValueError("a void does not swell")
            if (
                self.plasticity is not None
                or self.crystal_plasticity is not None
                or self.porous_plasticity is not None
            ):
                raise ValueError("a void has no plasticity")
            if self.mfront is not None:
                raise ValueError("a void has no MFront behaviour")
            if self.learned is not None:
                raise ValueError("a void has no learned law")
            if self.coupling is not None:
                raise ValueError("a void has no coupling coefficients")
            if any(getattr(self, name) is not None for name in _ANALYTIC):
                raise ValueError("a void has no behaviour")
            if self.density:
                raise ValueError("a void has no density")
            return
        if self.expansion is not None:
            if not (self.young is not None or self.stiffness is not None):
                raise ValueError("expansion: a material with elastic parameters")
            object.__setattr__(self, "expansion", _expansion(self.expansion))
        if self.swelling is not None:
            if not (self.young is not None or self.stiffness is not None):
                raise ValueError("swelling: a material with elastic parameters")
            object.__setattr__(self, "swelling", _expansion(self.swelling, "swelling"))
        if (self.young is None) != (self.poisson is None):
            raise ValueError("give both 'young' and 'poisson' (or neither)")
        if self.young is not None and self.stiffness is not None:
            raise ValueError("give young/poisson (isotropic) or a stiffness matrix, not both")
        if self.conductivity is not None and self.conductivities is not None:
            raise ValueError("give a conductivity or a conductivity tensor, not both")
        if all(v is None for v in params):
            raise ValueError(
                "a material needs elastic parameters, a transport coefficient (conductivity, "
                f"{', '.join(others)}), or several of them"
            )
        for name in others:
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _transport(getattr(self, name), name))
        if self.refractive_index is not None:
            object.__setattr__(self, "refractive_index", _index(self.refractive_index))
        if self.young is not None:
            if not (math.isfinite(self.young) and self.young > 0):
                raise ValueError(f"young: must be positive, got {self.young!r}")
            if not -1.0 < self.poisson < 0.5:
                raise ValueError(f"poisson: must lie in (-1, 0.5), got {self.poisson!r}")
        if self.conductivity is not None and not (
            math.isfinite(self.conductivity) and self.conductivity > 0
        ):
            raise ValueError(f"conductivity: must be positive, got {self.conductivity!r}")
        if self.stiffness is not None:
            object.__setattr__(self, "stiffness", _matrix(self.stiffness, (6, 6), "stiffness"))
        if self.conductivities is not None:
            object.__setattr__(
                self, "conductivities", _matrix(self.conductivities, (3, 3), "conductivities")
            )
        if self.plasticity is not None:
            object.__setattr__(self, "plasticity", Plasticity.from_config(self.plasticity))
            if self.young is None:
                raise ValueError(
                    "plasticity: an isotropic elastic material (young and poisson, or bulk and "
                    "shear)"
                )
        if self.crystal_plasticity is not None:
            if self.plasticity is not None:
                raise ValueError("give plasticity (von Mises) or crystal_plasticity, not both")
            cp = CrystalPlasticity.from_config(self.crystal_plasticity)
            object.__setattr__(self, "crystal_plasticity", cp)
            if self.young is None and self.stiffness is None:
                raise ValueError("crystal_plasticity: a material with elastic parameters")
            cp.check_softening(self.matrix("elastic"))  # of its cleavage planes
        if self.mfront is not None:
            from morphogen.homogenization.mfront import MFrontBehaviour

            if self.plasticity is not None or self.crystal_plasticity is not None:
                raise ValueError("give plasticity, crystal_plasticity or mfront, one of them")
            if self.young is None and self.stiffness is None:
                raise ValueError(
                    "mfront: a material with elastic parameters (the linear problems and "
                    "the reference media)"
                )
            object.__setattr__(self, "mfront", MFrontBehaviour.from_config(self.mfront))
        if self.porous_plasticity is not None:
            if (
                self.plasticity is not None
                or self.crystal_plasticity is not None
                or (self.mfront is not None)
            ):
                raise ValueError(
                    "give plasticity, porous_plasticity, crystal_plasticity or mfront, one of them"
                )
            pp = PorousPlasticity.from_config(self.porous_plasticity)
            object.__setattr__(self, "porous_plasticity", pp)
            if self.young is None:
                raise ValueError(
                    "porous_plasticity: an isotropic elastic material (young and poisson)"
                )
        if self.learned is not None:
            from morphogen.learning.behaviour import LearnedBehaviour

            behaviours = (self.plasticity, self.crystal_plasticity, self.mfront,
                          self.porous_plasticity)  # fmt: skip
            if any(b is not None for b in behaviours):
                raise ValueError(
                    "give plasticity, porous_plasticity, crystal_plasticity, mfront or learned, "
                    "one of them"
                )
            if self.young is None and self.stiffness is None:
                raise ValueError(
                    "learned: a material with elastic parameters (the linear problems and the "
                    "reference media; morphogen.learning.learned_material takes those of the "
                    "model)"
                )
            object.__setattr__(self, "learned", LearnedBehaviour.from_config(self.learned))
        if self.coupling is not None:
            from morphogen.homogenization.coupled import Coupling

            object.__setattr__(self, "coupling", Coupling.from_config(self.coupling))
        self._analytic_laws()
        tensors = [self.stiffness, self.conductivities]
        tensors += [v for v in (getattr(self, name) for name in others) if isinstance(v, tuple)]
        if all(t is None for t in tensors):
            object.__setattr__(self, "symmetry", "isotropic")
        elif self.symmetry == "isotropic":
            object.__setattr__(self, "symmetry", "anisotropic")

    def _elasticity_of_the_law(self) -> None:
        """Young's modulus and Poisson's ratio of a hyperelastic material given by its
        law alone: those of its bulk and shear moduli at the natural state."""
        from morphogen.homogenization.laws import Hyperelasticity

        law = Hyperelasticity.from_config(self.hyperelasticity)
        bulk, shear = law.initial_moduli()
        if bulk is None or shear is None:
            raise ValueError(
                "hyperelasticity: give young and poisson, or the bulk modulus and the "
                "parameters of the law that set its shear modulus"
            )
        young, poisson = young_poisson(bulk, shear)
        object.__setattr__(self, "young", young)
        object.__setattr__(self, "poisson", poisson)

    def _analytic_laws(self) -> None:
        """The behaviours of :mod:`morphogen.homogenization.laws` of the material,
        checked against its elasticity and its other behaviours."""
        given = [name for name in _ANALYTIC if getattr(self, name) is not None]
        if not given:
            return
        from morphogen.homogenization import laws

        read = {"viscoelasticity": laws.Viscoelasticity.from_config,
                "hyperelasticity": laws.Hyperelasticity.from_config,
                "damage": laws.damage_from_config}  # fmt: skip
        for name in given:
            object.__setattr__(self, name, read[name](getattr(self, name)))
        # a damage of Lemaitre or of Kachanov is that of the plasticity of the material
        coupled = isinstance(self.damage, laws.DuctileDamage)
        others = (None if coupled else self.plasticity, self.crystal_plasticity, self.mfront,
                  self.porous_plasticity, self.learned)  # fmt: skip
        if any(b is not None for b in others) or ("damage" in given and len(given) > 1):
            raise ValueError(
                "give one behaviour: plasticity, porous_plasticity, crystal_plasticity, "
                "mfront, learned, damage, or viscoelasticity and hyperelasticity (alone or "
                "together); a damage of Lemaitre or of Kachanov (damage.law) goes with a "
                "plasticity"
            )
        if self.young is None:
            raise ValueError(f"{given[0]}: an isotropic elastic material (young and poisson)")
        if self.hyperelasticity is not None:
            bulk, shear = self.hyperelasticity.initial_moduli()
            k0, g0 = bulk_shear(self.young, self.poisson)
            for name, own, mine in (("bulk", bulk, k0), ("shear", shear, g0)):
                if own is not None and not math.isclose(own, mine, rel_tol=1e-6):
                    raise ValueError(
                        f"hyperelasticity: its {name} modulus at the natural state, {own:.6g}, "
                        f"is not that of young and poisson, {mine:.6g} (the linear problems); "
                        "leave young and poisson out, or the parameter of the law"
                    )
        if coupled:  # its plasticity: von Mises' (and Norton's flow for Kachanov's law)
            laws._PlasticDamage(self.plasticity, self.damage, self.young, self.poisson)
        elif self.damage is not None:
            self.damage.strains(self.young)

    # -- constructors ----------------------------------------------------------------

    @classmethod
    def from_moduli(
        cls, bulk: float, shear: float, conductivity: float | None = None, *, expansion=None
    ):
        young, poisson = young_poisson(bulk, shear)
        return cls(young=young, poisson=poisson, conductivity=conductivity, expansion=expansion)

    @classmethod
    def cubic(
        cls,
        c11: float,
        c12: float,
        c44: float,
        *,
        conductivity: float | None = None,
        expansion: float | None = None,
    ):
        """Cubic crystal (stiffness constants in its axes); its conductivity and thermal
        expansion are isotropic."""
        d = np.zeros((6, 6))
        d[:3, :3] = c12
        d[[0, 1, 2], [0, 1, 2]] = c11
        d[[3, 4, 5], [3, 4, 5]] = c44
        return cls(stiffness=d, conductivity=conductivity, symmetry="cubic", expansion=expansion)

    @classmethod
    def hexagonal(
        cls,
        c11: float,
        c12: float,
        c13: float,
        c33: float,
        c44: float,
        *,
        conductivity=None,
        expansion=None,
    ):
        """Hexagonal (transversely isotropic) crystal, axis c along the third crystal
        axis; ``conductivity`` and ``expansion``: one value, or (basal plane, axis c)."""
        d = np.zeros((6, 6))
        d[0, 0] = d[1, 1] = c11
        d[2, 2] = c33
        d[0, 1] = d[1, 0] = c12
        d[0, 2] = d[2, 0] = d[1, 2] = d[2, 1] = c13
        d[3, 3] = d[4, 4] = c44
        d[5, 5] = 0.5 * (c11 - c12)
        k, kt = _thermal(conductivity, uniaxial=True)
        if expansion is not None and np.shape(expansion) == (2,):
            expansion = (expansion[0], expansion[0], expansion[1])
        return cls(
            stiffness=d,
            conductivity=k,
            conductivities=kt,
            symmetry="hexagonal",
            expansion=expansion,
        )

    @classmethod
    def anisotropic(cls, stiffness=None, *, conductivity=None, expansion=None):
        """Crystal of any symmetry: 6 x 6 elasticity matrix (Voigt order 11, 22, 33, 23,
        31, 12, engineering shear strains), conductivity and thermal expansion (one value,
        three principal values along the crystal axes, or a 3 x 3 tensor), in the axes of
        the crystal."""
        k, kt = _thermal(conductivity, uniaxial=False)
        return cls(stiffness=stiffness, conductivity=k, conductivities=kt, expansion=expansion)

    # -- properties ------------------------------------------------------------------

    @property
    def elastic(self) -> bool:
        return self.young is not None or self.stiffness is not None

    @property
    def thermal(self) -> bool:
        return self.conductivity is not None or self.conductivities is not None

    def has(self, physics: str) -> bool:
        """Whether the material has the parameters of ``physics`` (:data:`PHYSICS`):
        elastic ones, or the coefficient of a transport phenomenon."""
        if not transport_physics(physics):
            return self.elastic
        if physics == "thermal":
            return self.thermal
        return getattr(self, TRANSPORT[physics]) is not None

    def inert(self, physics: str) -> bool:
        """Whether the material carries nothing for ``physics``: a void, or a phase whose
        transport coefficient is zero (an insulator), left out of that problem as a pore."""
        if self.void:
            return True
        if physics in ("elastic", "thermal"):
            return False
        value = getattr(self, TRANSPORT[physics])
        return value is not None and not isinstance(value, tuple) and value == 0.0

    @property
    def isotropic(self) -> bool:
        return self.symmetry == "isotropic"

    @property
    def plastic(self) -> bool:
        """Whether the material yields (von Mises; nonlinear loading paths)."""
        return self.plasticity is not None

    @property
    def crystal_plastic(self) -> bool:
        """Whether the material slips on its slip systems (crystal plasticity)."""
        return self.crystal_plasticity is not None

    @property
    def porous(self) -> bool:
        """Whether the material is porous plastic (Gurson, Rousselier, elliptic)."""
        return self.porous_plasticity is not None

    @property
    def analytic(self) -> bool:
        """Whether the material has a behaviour of :mod:`morphogen.homogenization.laws`
        (viscoelasticity, hyperelasticity, damage)."""
        return any(getattr(self, name) is not None for name in _ANALYTIC)

    @property
    def nonlinear(self) -> bool:
        """Whether the behaviour of the material is nonlinear (loading paths)."""
        external = self.mfront is not None or self.learned is not None or self.analytic
        return self.plastic or self.crystal_plastic or self.porous or external

    @property
    def expands(self) -> bool:
        """Whether the material has a (nonzero) thermal expansion."""
        return self.expansion is not None and bool(np.any(np.asarray(self.expansion) != 0))

    def oriented(self, physics: str) -> bool:
        """Whether the behaviour for ``physics`` depends on the orientation (a learned law
        runs in the axes of its material, unless it is not oriented)."""
        if not transport_physics(physics):
            return (
                self.stiffness is not None
                or isinstance(self.expansion, tuple)
                or isinstance(self.swelling, tuple)
                or self.crystal_plasticity is not None
                or (self.learned is not None and self.learned.oriented)
            )
        if physics == "thermal":
            return self.conductivities is not None
        return isinstance(getattr(self, TRANSPORT[physics]), tuple)

    def expansion_tensor(self, euler=None) -> np.ndarray:
        """Thermal expansion tensor (3 x 3, zero without expansion) in the axes of the
        sample: for the orientation ``euler`` (shape (3,) or (n, 3)), or of the crystal."""
        a = self.expansion if self.expansion is not None else 0.0
        if not isinstance(a, tuple):
            base = float(a) * np.eye(3)
            if euler is not None and np.ndim(euler) == 2:
                return np.repeat(base[None], len(euler), axis=0)
            return base
        if euler is None:
            return np.array(a)
        return rotate_conductivity(np.array(a), euler)  # a symmetric second-order tensor

    @property
    def lame(self) -> tuple[float, float]:
        self._need_elastic()
        if self.stiffness is not None:
            raise ValueError("Lame coefficients: isotropic materials only")
        return lame(self.young, self.poisson)

    @property
    def bulk(self) -> float:
        """Bulk modulus (isotropic), or of the Voigt average of a crystal."""
        return self.moduli("voigt")[0]

    @property
    def shear(self) -> float:
        """Shear modulus (isotropic), or of the Voigt average of a crystal."""
        return self.moduli("voigt")[1]

    def moduli(self, average: str = "voigt") -> tuple[float, float]:
        """Bulk and shear moduli: exact for isotropic materials; for crystals, the
        ``"voigt"``, ``"reuss"`` or ``"hill"`` average over uniform orientations."""
        self._need_elastic()
        if self.stiffness is None:
            return bulk_shear(self.young, self.poisson)
        if average == "voigt":
            return voigt_average(self.stiffness)
        if average == "reuss":
            return reuss_average(self.stiffness)
        if average == "hill":
            (kv, gv), (kr, gr) = voigt_average(self.stiffness), reuss_average(self.stiffness)
            return 0.5 * (kv + kr), 0.5 * (gv + gr)
        raise ValueError("average: 'voigt', 'reuss' or 'hill'")

    def mean_conductivity(self, average: str = "voigt", physics: str = "thermal") -> float:
        """Conductivity (isotropic), or the arithmetic (``"voigt"``) or harmonic
        (``"reuss"``) mean of the principal conductivities of a crystal; likewise the
        coefficient of another transport ``physics`` (:data:`TRANSPORT`)."""
        k = np.linalg.eigvalsh(self.matrix(physics))
        if np.all(k == k[0]):
            return float(k[0])
        return float(k.mean() if average == "voigt" else 1.0 / np.mean(1.0 / k))

    def matrix(self, physics: str, euler=None) -> np.ndarray:
        """Elasticity matrix (6 x 6), or conductivity tensor (3 x 3; that of any transport
        ``physics`` of :data:`TRANSPORT`) in the axes of the sample: for the orientation
        ``euler`` (Bunge angles in degrees; shape (3,) or (n, 3) for n orientations), or in
        the axes of the crystal."""
        if physics == "elastic":
            self._need_elastic()
            base = (
                isotropic_matrix(*lame(self.young, self.poisson))
                if self.stiffness is None
                else np.array(self.stiffness)
            )
            rotate = rotate_stiffness
        elif physics == "thermal":
            self._need_thermal()
            base = (
                self.conductivity * np.eye(3)
                if self.conductivities is None
                else np.array(self.conductivities)
            )
            rotate = rotate_conductivity
        elif physics in TRANSPORT:
            value = getattr(self, TRANSPORT[physics])
            if value is None:
                raise ValueError(f"this material has no {TRANSPORT[physics]}")
            base = np.array(value) if isinstance(value, tuple) else value * np.eye(3)
            rotate = rotate_conductivity
        else:
            raise ValueError(f"physics: one of {PHYSICS}, got {physics!r}")
        if euler is None or not self.oriented(physics):
            if euler is not None and np.ndim(euler) == 2:
                return np.repeat(base[None], len(euler), axis=0)
            return base
        return rotate(base, euler)

    def _need_elastic(self) -> None:
        if not self.elastic:
            raise ValueError("this material has no elastic parameters")

    def _need_thermal(self) -> None:
        if not self.thermal:
            raise ValueError("this material has no conductivity")

    # -- configuration ---------------------------------------------------------------

    def to_config(self):
        if self.void:
            return "void"
        out = {}
        if self.stiffness is not None:
            d = np.array(self.stiffness)
            if self.symmetry == "cubic":
                out["cubic"] = [float(d[0, 0]), float(d[0, 1]), float(d[3, 3])]
            elif self.symmetry == "hexagonal":
                out["hexagonal"] = [
                    float(d[i, j]) for i, j in ((0, 0), (0, 1), (0, 2), (2, 2), (3, 3))
                ]
            else:
                out["stiffness"] = d.tolist()
        for key in ("young", "poisson", "conductivity"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.conductivities is not None:
            out["conductivity"] = np.array(self.conductivities).tolist()
        if self.expansion is not None:
            out["expansion"] = np.array(self.expansion).tolist()
        if self.swelling is not None:
            out["swelling"] = np.array(self.swelling).tolist()
        if self.density is not None:
            out["density"] = self.density
        if self.plasticity is not None:
            out["plasticity"] = self.plasticity.to_config()
        if self.crystal_plasticity is not None:
            out["crystal_plasticity"] = self.crystal_plasticity.to_config()
        if self.mfront is not None:
            out["mfront"] = self.mfront.to_config()
        if self.porous_plasticity is not None:
            out["porous_plasticity"] = self.porous_plasticity.to_config()
        if self.learned is not None:
            out["learned"] = self.learned.to_config()
        for name in TRANSPORT.values():
            value = getattr(self, name)
            if name != "conductivity" and value is not None:
                out[name] = np.array(value).tolist()
        if self.refractive_index is not None:
            n = np.atleast_1d(self.refractive_index)
            pairs = [[float(v.real), float(v.imag)] for v in n]
            out["refractive_index"] = pairs[0] if np.ndim(self.refractive_index) == 0 else pairs
        if self.coupling is not None:
            out["coupling"] = self.coupling.to_config()
        for name in _ANALYTIC:
            if getattr(self, name) is not None:
                out[name] = getattr(self, name).to_config()
        return out

    @classmethod
    def from_config(cls, spec) -> Material:
        """``"void"``, or a mapping with young/poisson or bulk/shear (isotropic), or
        ``cubic: [c11, c12, c44]``, ``hexagonal: [c11, c12, c13, c33, c44]`` or
        ``stiffness: 6 x 6`` (crystals); and/or ``conductivity`` (one value, three
        principal values or a 3 x 3 tensor, in the axes of the crystal); and
        ``expansion``, the thermal expansion (likewise), ``swelling``, the strain of a unit
        content of an absorbed species (likewise), ``density``, ``plasticity``
        (:meth:`~morphogen.homogenization.behaviours.Plasticity.from_config`),
        ``porous_plasticity`` (:meth:`~morphogen.homogenization.porous.PorousPlasticity.
        from_config`), ``crystal_plasticity`` (:meth:`~morphogen.homogenization.crystal.
        CrystalPlasticity.from_config`), ``mfront`` (:meth:`~morphogen.homogenization.mfront.
        MFrontBehaviour.from_config`) and ``learned`` (the file of a learned model,
        :meth:`~morphogen.learning.behaviour.LearnedBehaviour.from_config`); or ``zmat``, a
        material file of Z-mat (its path, or a
        mapping ``file``, ``behaviour``, ``name``, ``native``, ``porosity``:
        :func:`~morphogen.homogenization.zmat.zmat_material`), with a ``conductivity`` and a
        ``density``."""
        if isinstance(spec, Material):
            return spec
        if isinstance(spec, dict) and "zmat" in spec:
            import dataclasses

            from morphogen.homogenization.zmat import zmat_material

            extra = sorted(set(spec) - {"zmat", "conductivity", "density"})
            if extra:
                raise ValueError(
                    f"zmat: the other keys of the material are {extra}; allowed "
                    "beside it: conductivity, density"
                )
            options = spec["zmat"]
            options = {"file": options} if isinstance(options, str) else dict(options)
            known = {"file", "behaviour", "name", "native", "directory", "porosity"}
            unknown = sorted(set(options) - known)
            if unknown or "file" not in options:
                raise ValueError("zmat: a path, or {file, behaviour, name, native, porosity}")
            base = zmat_material(options.pop("file"), **options)
            values = {k: float(spec[k]) for k in ("conductivity", "density") if k in spec}
            return dataclasses.replace(base, **values) if values else base
        late = (
            "expansion",
            "swelling",
            "density",
            "plasticity",
            "crystal_plasticity",
            "mfront",
            "learned",
            "porous_plasticity",
            "coupling",
            "refractive_index",
            *_ANALYTIC,
            *(name for name in TRANSPORT.values() if name != "conductivity"),
        )
        if isinstance(spec, dict) and (set(late) & set(spec)):
            import dataclasses

            extra = {k: spec[k] for k in late if k in spec}
            rest = {k: v for k, v in spec.items() if k not in extra}
            if not rest:  # a material of transport coefficients only
                return cls(**extra)
            base = cls.from_config(rest)
            return dataclasses.replace(base, **extra)
        if isinstance(spec, str):
            if spec.strip().lower() in ("void", "pore", "pores", "empty"):
                return VOID
            raise ValueError(f"unknown material {spec!r}; use 'void' or a mapping of parameters")
        if not isinstance(spec, dict):
            raise ValueError(f"a material is 'void' or a mapping, got {spec!r}")
        allowed = {
            "young",
            "poisson",
            "bulk",
            "shear",
            "conductivity",
            "expansion",
            "swelling",
            "density",
            "plasticity",
            "crystal_plasticity",
            "mfront",
            "learned",
            "porous_plasticity",
            "coupling",
            "refractive_index",
            *_ANALYTIC,
            *TRANSPORT.values(),
        }
        crystal = {"cubic", "hexagonal", "stiffness"}
        unknown = sorted(set(spec) - allowed - crystal)
        if unknown:
            raise ValueError(
                f"material: unknown keys {unknown}; allowed: {sorted(allowed | crystal)}"
            )
        conductivity = spec.get("conductivity")
        kinds = [k for k in crystal if k in spec]
        if kinds:
            if len(kinds) > 1 or set(spec) & {"young", "poisson", "bulk", "shear"}:
                raise ValueError(
                    "material: give one of cubic, hexagonal, stiffness, young/poisson, bulk/shear"
                )
            values = spec[kinds[0]]
            if kinds[0] == "cubic":
                if len(values) != 3:
                    raise ValueError("cubic: [c11, c12, c44]")
                return cls.cubic(*map(float, values), conductivity=conductivity)
            if kinds[0] == "hexagonal":
                if len(values) != 5:
                    raise ValueError("hexagonal: [c11, c12, c13, c33, c44]")
                return cls.hexagonal(*map(float, values), conductivity=conductivity)
            return cls.anisotropic(values, conductivity=conductivity)
        if conductivity is not None and np.ndim(conductivity) > 0:
            k, kt = _thermal(conductivity, uniaxial=False)
            iso = {k2: float(v) for k2, v in spec.items() if k2 != "conductivity"}
            if iso:
                raise ValueError("material: an anisotropic conductivity needs a crystal stiffness")
            return cls(conductivity=k, conductivities=kt)
        values = {k: float(v) for k, v in spec.items()}
        if "bulk" in values or "shear" in values:
            if "young" in values or "poisson" in values:
                raise ValueError("material: give young/poisson or bulk/shear, not both")
            if not ("bulk" in values and "shear" in values):
                raise ValueError("material: give both 'bulk' and 'shear'")
            return cls.from_moduli(values["bulk"], values["shear"], values.get("conductivity"))
        return cls(
            young=values.get("young"),
            poisson=values.get("poisson"),
            conductivity=values.get("conductivity"),
        )


def _thermal(conductivity, *, uniaxial: bool):
    """(isotropic value, tensor) of a conductivity given as one value, principal values
    (two for a uniaxial crystal: basal plane and axis), or a 3 x 3 tensor."""
    if conductivity is None:
        return None, None
    c = np.asarray(conductivity, dtype=float)
    if c.ndim == 0:
        return float(c), None
    if uniaxial and c.shape == (2,):
        c = np.array([c[0], c[0], c[1]])
    if c.shape == (3,):
        if np.allclose(c, c[0]):
            return float(c[0]), None
        return None, np.diag(c)
    if c.shape == (3, 3):
        return None, c
    raise ValueError("conductivity: one value, principal values or a 3 x 3 tensor")


VOID = Material(void=True)


def any_oriented(materials: dict[int, Material]) -> bool:
    """Whether some material depends on the orientation of its grain (anisotropic
    elasticity, conductivity or expansion, or slip systems): the grains and their
    orientations are then needed."""
    return any(not m.void and any(m.oriented(p) for p in PHYSICS) for m in materials.values())


def materials_from_config(spec) -> dict[int, Material]:
    """Phase -> material mapping from a configuration mapping (keys may be strings)."""
    if not isinstance(spec, dict) or not spec:
        raise ValueError("materials: expected a mapping phase -> material")
    out = {}
    for key, value in spec.items():
        try:
            phase = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"materials: phase keys must be integers, got {key!r}") from None
        try:
            out[phase] = Material.from_config(value)
        except ValueError as exc:
            raise ValueError(f"materials[{phase}]: {exc}") from None
    return out


@dataclass(frozen=True)
class MaterialTable:
    """Material of every element (voxel or tetrahedron) of a sample, for one physics.

    ``mat[e]`` indexes ``matrices`` (-1 for voids): elasticity matrices (n, 6, 6) or
    conductivity tensors (n, 3, 3) in the axes of the sample; one per phase, or one per
    grain for the oriented phases of a polycrystal. ``stat[e]`` indexes ``phases``.
    ``eigen`` (elasticity) holds the thermal strain of each material for a unit change of
    temperature (Voigt order, engineering shears), ``rotations`` the orientation of each
    material (``crystal = g @ sample``; the identity for the materials of the phases that
    are not oriented).
    """

    mat: np.ndarray
    stat: np.ndarray
    matrices: np.ndarray
    phases: tuple[int, ...]
    isotropic: bool
    eigen: np.ndarray | None = None
    rotations: np.ndarray | None = None

    @property
    def active(self) -> np.ndarray:
        return self.mat >= 0


def material_table(
    phase, materials: dict[int, Material], physics: str, *, grains=None, orientations=None
) -> MaterialTable:
    """Materials of the elements of phases ``phase`` for ``physics`` (:data:`PHYSICS`:
    'elastic', or a transport phenomenon such as 'thermal' or 'electrical'). Crystals are
    turned by the orientation of their grain (``grains``: grain of each element,
    ``orientations``: Bunge angles of each grain); without them, their axes are those of
    the sample. The phases inert for the physics (voids, insulators:
    :meth:`Material.inert`) have no material (``mat`` -1)."""
    from morphogen.orientations import euler_to_matrix

    transport = transport_physics(physics)
    phase = np.asarray(phase)
    if phase.dtype.kind not in "iub":
        raise ValueError("homogenization needs an integer phase image, not partial volumes")
    phase = phase.ravel().astype(np.int64)
    present = [int(v) for v in np.unique(phase)]
    missing = [p for p in present if p not in materials]
    if missing:
        raise ValueError(f"no material for phase(s) {missing}; phases present: {present}")
    phases = tuple(present)
    lookup = np.full(max(phases, default=0) + 1, -1, dtype=np.int64)
    lookup[list(phases)] = np.arange(len(phases))
    stat = lookup[phase]
    mat = np.full(phase.size, -1, dtype=np.int64)
    blocks, eigen, rotations, count, isotropic = [], [], [], 0, True
    oriented = grains is not None and orientations is not None
    for p in phases:
        m = materials[p]
        if m.inert(physics):
            continue
        if not transport and not m.elastic:
            raise ValueError(f"phase {p}: the material has no elastic parameters")
        if transport and not m.has(physics):
            raise ValueError(f"phase {p}: the material has no {TRANSPORT[physics]}")
        sel = np.flatnonzero(phase == p)
        # an MFront behaviour may be orthotropic (a crystal) whatever the elastic parameters
        # of its material: its grains keep their rotations (unused by isotropic behaviours)
        mfront = physics == "elastic" and m.mfront is not None
        if m.oriented(physics) or mfront:
            if m.oriented(physics):
                isotropic = False
            if oriented:
                ids, inverse = np.unique(np.asarray(grains).ravel()[sel], return_inverse=True)
                euler = np.asarray(orientations, dtype=float)[ids]
                blocks.append(m.matrix(physics, euler))
                eigen.append(m.expansion_tensor(euler))
                rotations.append(euler_to_matrix(euler).reshape(-1, 3, 3))
                mat[sel] = count + inverse
                count += len(ids)
                continue
            if m.oriented(physics):
                log.info("phase %d: crystal axes along the axes of the sample (no orientations)",
                         p)  # fmt: skip
        blocks.append(m.matrix(physics)[None])
        eigen.append(m.expansion_tensor()[None])
        rotations.append(np.eye(3)[None])
        mat[sel] = count
        count += 1
    d = 3 if transport else 6
    matrices = np.concatenate(blocks) if blocks else np.zeros((0, d, d))
    strains = None
    if not transport:
        from morphogen.homogenization.elements import tensor_to_voigt

        a = np.concatenate(eigen) if eigen else np.zeros((0, 3, 3))
        strains = np.array([tensor_to_voigt(t, strain=True) for t in a]).reshape(-1, 6)
    return MaterialTable(
        mat=mat,
        stat=stat,
        matrices=np.ascontiguousarray(matrices),
        phases=phases,
        isotropic=isotropic,
        eigen=None if strains is None else np.ascontiguousarray(strains),
        rotations=np.ascontiguousarray(
            np.concatenate(rotations) if rotations else np.zeros((0, 3, 3))
        ),
    )
