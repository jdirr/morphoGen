"""Weak couplings of elasticity with a scalar field that loads the phases: the pressure of
the fluid of the pores (the poroelasticity of Biot) and the content of a species that the
phases absorb (hygroscopic and chemical swelling).

Both are eigenstrain problems of the elastic solvers (FFT on voxel images, finite elements
on conforming meshes, extended finite elements on level sets), as the thermal expansion is.

**Swelling.** A phase that absorbs a species (moisture in a polymer or in wood, a solvent
in a gel, lithium in an electrode) strains by ``beta c`` for a content ``c``:
``sigma = C : (eps - beta c)``. With the ``swelling`` of the materials, the strain of
each phase for a unit of the quantity that drives the sorption (a content, a relative
humidity, an activity: the coefficient of a phase is then its strain at equilibrium with a
unit of that quantity, the product of its coefficient of expansion and of its
solubility), :func:`swelling_tensor` gives the macroscopic strain of the stress-free
volume element, ``E = S : Sigma + B c``; Levin's relation holds, ``B = <beta : C : A> :
C*^-1``.

**Poroelasticity.** In a porous solid whose pores hold a fluid at the pressure ``p``,

    ``Sigma = C_d : E - b p``,    ``phi - phi_0 = b : E + p / N``

with the drained stiffness ``C_d`` (that of the solid with empty pores), the tensor of
Biot ``b`` and the modulus of Biot ``N`` of the solid (Biot 1941; Coussy 2004). The
pressure on the walls of the pores is, in the solid, the uniform stress ``-p 1`` and the
eigenstrain ``-p S_s : 1`` of each solid phase: one more solution of the elastic
problem, at zero macroscopic strain, gives ``b = 1 - Sigma'`` (``Sigma'``: the mean stress
of that problem for a unit pressure) and ``1 / N = -(1 - phi_0) <tr eps>_s``, the volume
the pores gain. A homogeneous solid recovers ``b = 1 - C_d : S_s : 1`` and ``1 / N = (b -
phi_0 1) : S_s : 1`` (``b = 1 - K_d / K_s``, ``1 / N = (b - phi_0) / K_s`` when isotropic);
the solid may as well be heterogeneous (several phases, crystals), where these formulae do
not hold. A fluid of bulk modulus ``K_f`` that cannot drain stiffens the medium (Gassmann
1951): ``1 / M = 1 / N + phi_0 / K_f``, ``C_u = C_d + M b (x) b``, and a stress raises its
pressure by ``p = -B : Sigma / 3`` with the tensor of Skempton ``B = 3 M S_u : b``.

References: M. A. Biot, J. Appl. Phys. 12 (1941) 155; F. Gassmann, Vierteljahrsschr.
Naturforsch. Ges. Zürich 96 (1951) 1; O. Coussy, Poromechanics, Wiley (2004); L. Dormieux,
D. Kondo, F.-J. Ulm, Microporomechanics, Wiley (2006); V. M. Levin, Mekh. Tverd. Tela 2
(1967) 88.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from morphogen.homogenization.core import VOIGT_NAMES, Loading
from morphogen.homogenization.elements import VOIGT, voigt_to_tensor
from morphogen.homogenization.materials import Material

__all__ = [
    "POROELASTIC_ENTRIES",
    "UNDRAINED_ENTRIES",
    "poroelastic_tensors",
    "pressure_materials",
    "swelling_materials",
    "swelling_tensor",
    "undrained",
]

#: The scalar results of :func:`poroelastic_tensors` (columns of the tables of results and
#: quantities of the statistics of RVE studies)
POROELASTIC_ENTRIES = (
    "porosity",
    "biot_coefficient",
    "biot_modulus",
    "bulk_modulus",
    *(f"b{name}" for name in VOIGT_NAMES),
)
#: Those of a fluid that cannot drain (with its bulk modulus)
UNDRAINED_ENTRIES = (
    "storage_modulus",
    "undrained_bulk_modulus",
    "skempton_coefficient",
    *(f"B{name}" for name in VOIGT_NAMES),
)


def _hydrostatic_strain(material: Material):
    """The strain of a material under a unit hydrostatic tension, ``S : 1``: one
    coefficient (isotropic) or a 3 x 3 tensor in the axes of the crystal."""
    if material.stiffness is None:
        return (1.0 - 2.0 * material.poisson) / material.young  # 1 / (3 K)
    compliance = np.linalg.inv(np.array(material.stiffness))
    return voigt_to_tensor(compliance[:, :3].sum(axis=1), strain=True)


def pressure_materials(materials: dict[int, Material]) -> dict[int, Material]:
    """The materials of the problem of a unit pressure in the pores: each solid phase
    with the eigenstrain ``-S : 1`` (as a thermal expansion: the temperature of the
    elastic problems is then the pressure), the voids as they are."""
    out = {}
    for phase, m in materials.items():
        if m.void:
            out[phase] = m
            continue
        if m.young is None and m.stiffness is None:
            raise ValueError(f"poroelasticity: the material of phase {phase} is not elastic")
        out[phase] = dataclasses.replace(m, expansion=-_hydrostatic_strain(m))
    return out


def swelling_materials(materials: dict[int, Material]) -> dict[int, Material]:
    """The materials of the problem of a unit content of the absorbed species: each phase
    with its ``swelling`` as a thermal expansion (none: a phase that does not swell)."""
    if not any(m.swelling is not None for m in materials.values()):
        raise ValueError("swelling: no material swells (Material(swelling=...))")
    return {
        phase: m if m.void else dataclasses.replace(m, expansion=m.swelling)
        for phase, m in materials.items()
    }


def _problem(sample, materials, bc, options):
    from morphogen.homogenization.apparent import make_problem

    return make_problem(sample, materials, "elastic", bc, **options)


def swelling_tensor(sample, materials: dict[int, Material], bc: str = "periodic", **options):
    """Apparent swelling tensor (3 x 3) of a sample and the solution that gives it: the
    macroscopic strain of a unit content of the absorbed species under zero macroscopic
    stress, with the ``swelling`` of the materials (``E = S : Sigma + B c``). ``options``:
    those of :func:`~morphogen.homogenization.apparent.make_problem` (``solver``, ``tol``,
    ``maxiter``, ``grains``, ``orientations``, ``device``)."""
    from morphogen.homogenization.apparent import thermal_expansion

    return thermal_expansion(_problem(sample, swelling_materials(materials), bc, options))


def undrained(stiffness, biot, biot_modulus: float, porosity: float, fluid_bulk: float) -> dict:
    """The response of a saturated porous medium whose fluid, of bulk modulus
    ``fluid_bulk``, cannot drain (Gassmann): ``storage_modulus`` ``M`` (``1 / M = 1 / N +
    phi_0 / K_f``), ``undrained_stiffness`` ``C_u = C_d + M b (x) b`` (Voigt),
    ``undrained_bulk_modulus`` (``1 / (1 : S_u : 1)``), ``skempton`` (the 3 x 3 tensor ``B
    = 3 M S_u : b``: a macroscopic stress raises the pressure by ``-B : Sigma / 3``) and
    ``skempton_coefficient`` (its mean), from the drained ``stiffness`` (6 x 6), the tensor
    (3 x 3) and the modulus of Biot and the porosity."""
    if not fluid_bulk > 0.0:
        raise ValueError("fluid_bulk: a positive bulk modulus")
    c = np.asarray(stiffness, dtype=float)
    b = np.asarray(biot, dtype=float)
    vec = np.array([b[i, j] for i, j in VOIGT])  # as a stress: no factor on the shears
    inverse = (1.0 / biot_modulus if np.isfinite(biot_modulus) else 0.0) + porosity / fluid_bulk
    storage = 1.0 / inverse if inverse > 0.0 else float("inf")
    if not np.isfinite(storage):  # no pore, a rigid solid: nothing to store
        cu = c.copy()
        skempton = np.zeros((3, 3))
    else:
        cu = c + storage * np.outer(vec, vec)
        skempton = 3.0 * storage * voigt_to_tensor(np.linalg.solve(cu, vec), strain=True)
    out = {
        "storage_modulus": float(storage),
        "undrained_stiffness": cu,
        "undrained_bulk_modulus": float(1.0 / np.linalg.inv(cu)[:3, :3].sum()),
        "skempton": skempton,
        "skempton_coefficient": float(np.trace(skempton) / 3.0),
    }
    out.update(
        {f"B{n}": float(skempton[i, j]) for (i, j), n in zip(VOIGT, VOIGT_NAMES, strict=True)}
    )
    return out


def poroelastic_tensors(sample, materials: dict[int, Material], bc: str = "periodic", *,
                        fluid_bulk: float | None = None, **options) -> dict:  # fmt: skip
    """The poroelastic law of a porous sample (voxel image, conforming mesh or level set;
    its pores: the void phases), under periodic or kinematic uniform (``kubc``) conditions.

    Returns the drained stiffness (``"tensor"``, Voigt 6 x 6, with its ``"compliance"``),
    the tensor of Biot (``"biot"``, 3 x 3; its entries ``b11`` ... ``b12`` and its mean
    ``"biot_coefficient"``), the modulus of Biot of the solid (``"biot_modulus"``; infinite
    for a rigid solid), the ``"porosity"`` and the drained ``"bulk_modulus"`` (``1 / (1 :
    S_d : 1)``); with the bulk modulus of the fluid ``fluid_bulk``, also the undrained
    response of :func:`undrained`. The solid may hold several phases and crystals
    (``grains``, ``orientations``). ``options``: those of :func:`~morphogen.homogenization.
    apparent.make_problem` (``solver``, ``tol``, ``maxiter``, ``grains``, ``orientations``,
    ``device``)."""
    from morphogen.homogenization.apparent import stiffness_tensor

    if bc not in ("periodic", "kubc"):
        raise ValueError(
            "poroelasticity: periodic or kinematic uniform (kubc) conditions (the macroscopic "
            f"strain is prescribed), not {bc!r}"
        )
    if not any(m.void for m in materials.values()):
        raise ValueError("poroelasticity: no pore (a void phase) in the materials")
    problem = _problem(sample, pressure_materials(materials), bc, options)
    drained, _ = stiffness_tensor(problem)
    # a unit pressure at zero macroscopic strain: the stress -p 1 everywhere, and that of
    # the eigenstrains -p S_s : 1 of the solid phases
    sol = problem.solve(Loading.strain(np.zeros(6), temperature=1.0))
    biot = np.eye(3) - 0.5 * (sol.stress + sol.stress.T)
    porosity = 1.0 - float(sol.material_fraction)
    if not porosity > 0.0:
        raise ValueError("poroelasticity: the sample has no pore")
    # the volume gained by the pores is that lost by the solid (no macroscopic strain)
    gained = -sum(p.fraction * float(np.sum(p.mean["strain"][:3])) for p in sol.phases)
    compliance = np.linalg.inv(drained)
    out = {
        "tensor": drained,
        "compliance": compliance,
        "biot": biot,
        "porosity": porosity,
        "biot_coefficient": float(np.trace(biot) / 3.0),
        "biot_modulus": float(1.0 / gained) if gained > 0.0 else float("inf"),
        "bulk_modulus": float(1.0 / compliance[:3, :3].sum()),
    }
    out.update({f"b{n}": float(biot[i, j]) for (i, j), n in zip(VOIGT, VOIGT_NAMES, strict=True)})
    if fluid_bulk is not None:
        out.update(undrained(drained, biot, out["biot_modulus"], porosity, float(fluid_bulk)))
    return out
