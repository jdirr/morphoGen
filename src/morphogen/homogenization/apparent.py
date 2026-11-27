"""Apparent properties of a volume element (Kanit et al. 2003; Dirrenberger et al. 2014).

Scalar properties follow Dirrenberger et al. (2014), sections 3.5 and 4:

``bulk_modulus``
    hydrostatic strain ``E = 1`` (eq. 63): ``k = tr(Sigma) / 9`` (eq. 70); the energy
    estimate is ``<sigma : eps> / 9`` (eq. 64). Under SUBC, ``Sigma = 1`` is imposed and
    ``k = 1 / tr(E)``.
``shear_modulus``
    simple shear ``E_12 = E_21 = 1/2`` (eq. 65): ``mu = Sigma_12`` (eq. 71); the energy
    estimate is ``<sigma : eps>`` (eq. 66). Under SUBC, ``Sigma_12 = 1`` and
    ``mu = 1 / (2 E_12)``.
``conductivity``
    gradient ``G = e_1`` (eq. 60): ``k = -Q_1`` with ``q = -k grad T`` (eq. 68); the
    energy estimate is ``<k grad T . grad T>`` (eq. 61). Under UHF (``subc``), ``Q = -e_1``
    and ``k = 1 / G_1``.
``electrical_conductivity``, ``permittivity``, ``permeability``, ``diffusivity``
    the same problem for the other transport phenomena of a scalar potential
    (:data:`TRANSPORT_PROPERTIES`): Ohm's law, electrostatics, magnetostatics and Fick's
    law, each with the coefficient of the materials for it
    (:data:`~morphogen.homogenization.materials.TRANSPORT`).

With voids, averages are taken over the whole box with zero stress and flux in the
voids (eqs. 24 and 29). Each result also carries the isotropy criterion of the
realization (eqs. 72-74), computed on its macroscopic response.

Tensors: the stiffness (six loadings), the conductivity tensor (three; that of each
transport phenomenon, ``<property>_tensor``), and the thermal
expansion ``A`` of ``E = S : Sigma + A dT``: the macroscopic strain of a unit change of
temperature without macroscopic stress (a stress-controlled loading, see
:func:`morphogen.homogenization.core.superpose`).

Loading paths (:class:`~morphogen.homogenization.paths.LoadingPath`) follow the
nonlinear behaviours of the phases in time: by finite elements with periodic or
kinematic conditions (:mod:`morphogen.homogenization.nonlinear`), or by the FFT solver of
voxel images with periodic conditions (:mod:`morphogen.homogenization.fft_nonlinear`).

A triangular mesh of the plane (:class:`~morphogen.mesh.triangles.TriMesh`) is solved by
the finite elements of the plane (:mod:`morphogen.homogenization.plane`): the apparent
properties and tensors of its structure invariant across the plane, with periodic,
kinematic or static conditions on its sides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from morphogen.homogenization.core import VOIGT_NAMES, Loading, Solution, boundary_condition
from morphogen.homogenization.coupled import COUPLINGS, coupled_tensors
from morphogen.homogenization.elements import VOIGT, tensor_to_voigt
from morphogen.homogenization.fe import FEElasticProblem, FEThermalProblem, integration_of
from morphogen.homogenization.fft import FFTElasticProblem, FFTThermalProblem
from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem
from morphogen.homogenization.materials import TRANSPORT, Material, transport_physics
from morphogen.homogenization.nonlinear import NonlinearFEProblem
from morphogen.homogenization.paths import LoadingPath, PathResult
from morphogen.homogenization.plane import PLANES, PlaneElasticProblem, PlaneThermalProblem
from morphogen.homogenization.postprocess import anisotropy, compliance, engineering_constants
from morphogen.homogenization.xfem import (
    LevelSetSample,
    NonlinearXFEMProblem,
    XFEMElasticProblem,
    XFEMThermalProblem,
)
from morphogen.image import VoxelImage
from morphogen.mesh.tetra import TetMesh
from morphogen.mesh.triangles import TriMesh
from morphogen.stopping import checkpoint

log = logging.getLogger("morphogen")

#: The apparent scalar property and tensor of each transport phenomenon (the coefficient
#: of the materials for it: :data:`~morphogen.homogenization.materials.TRANSPORT`)
TRANSPORT_PROPERTIES = {physics: (name, f"{name}_tensor") for physics, name in TRANSPORT.items()}
_OTHERS = [names for physics, names in TRANSPORT_PROPERTIES.items() if physics != "thermal"]
#: The scalar properties: one value per realization and boundary condition
PROPERTIES = ("bulk_modulus", "shear_modulus", "conductivity", *(s for s, _ in _OTHERS))
#: The tensors (``swelling``: that of the strains of an absorbed species; ``poroelastic``:
#: the drained stiffness, the tensor and the modulus of Biot of a porous sample:
#: :mod:`morphogen.homogenization.poroelastic`)
TENSORS = (
    "stiffness",
    "conductivity_tensor",
    "thermal_expansion",
    *(t for _, t in _OTHERS),
    "swelling",
    "poroelastic",
)
#: The strong couplings (:mod:`morphogen.homogenization.coupled`): each gives the tensors of
#: its coupled law (``piezoelectric``: the stiffness, the piezoelectric matrix and the
#: permittivity, with the constants derived from them)
COUPLED = tuple(COUPLINGS)
#: The properties of a harmonic regime (:mod:`morphogen.homogenization.harmonic`): the
#: complex stiffness of viscoelastic phases and the complex conductivity of leaky
#: dielectrics at the ``frequencies`` of :func:`homogenize`, the effective refractive
#: index at its ``wavelengths``
HARMONIC = ("dynamic_stiffness", "complex_conductivity", "refractive_index")
#: The properties of the flow of a fluid through the pores (the void phases) of a voxel
#: image: ``hydraulic_permeability``, the tensor of Darcy's law
#: (:mod:`morphogen.homogenization.flow`), and ``acoustic``, the parameters of the
#: equivalent fluid of a rigid porous frame (:mod:`morphogen.homogenization.acoustics`)
FLOW = ("hydraulic_permeability", "acoustic")
#: The elastic properties of the finite elements of the plane (meshes of the plane), with
#: those of the transport phenomena
PLANE_PROPERTIES = ("bulk_modulus", "shear_modulus", "stiffness", "thermal_expansion")
#: The physics of the problem that gives each property (``"coupled"``: the fields of a
#: strong coupling solved together; ``"harmonic"``: a complex law at each frequency)
PHYSICS = {
    "bulk_modulus": "elastic",
    "shear_modulus": "elastic",
    "stiffness": "elastic",
    "thermal_expansion": "elastic",
    "swelling": "elastic",
    "poroelastic": "elastic",
    **{name: physics for physics, names in TRANSPORT_PROPERTIES.items() for name in names},
    **dict.fromkeys(COUPLED, "coupled"),
    **dict.fromkeys(HARMONIC, "harmonic"),
    **dict.fromkeys(FLOW, "flow"),
}


def bulk_isotropy(tensor) -> float:
    """sqrt(T:K:T / T:J:T) (eq. 73): zero for a purely spherical response."""
    t = np.asarray(tensor, dtype=float)
    tr = np.trace(t)
    dev = t - tr / 3.0 * np.eye(3)
    sph = tr * tr / 3.0
    return float(np.sqrt(np.sum(dev * dev) / sph)) if sph > 0 else float("nan")


def shear_isotropy(tensor) -> float:
    """sqrt(T:J:T / T:K:T) (eq. 74): zero for a purely deviatoric response."""
    t = np.asarray(tensor, dtype=float)
    tr = np.trace(t)
    dev = t - tr / 3.0 * np.eye(3)
    dd = float(np.sum(dev * dev))
    return float(np.sqrt(tr * tr / 3.0 / dd)) if dd > 0 else float("nan")


def thermal_isotropy(vector, axis: int = 0) -> float:
    """``(|Q_2| + |Q_3|) / |Q_1|`` for a loading along ``axis`` (eq. 72)."""
    v = np.abs(np.asarray(vector, dtype=float))
    return float((v.sum() - v[axis]) / v[axis]) if v[axis] > 0 else float("nan")


@dataclass
class Apparent:
    """Apparent property of one volume element for one boundary condition."""

    property: str
    bc: str
    value: float
    energy: float
    isotropy: float
    response: np.ndarray
    solution: Solution

    def row(self) -> dict:
        """Flat record (long format) with the response components and solver data."""
        out = {
            "property": self.property,
            "bc": self.bc,
            "value": self.value,
            "energy_value": self.energy,
            "isotropy": self.isotropy,
            "material_fraction": self.solution.material_fraction,
        }
        r = np.asarray(self.response)
        if r.shape == (3, 3):
            for (a, b), name in zip(VOIGT, ("11", "22", "33", "23", "31", "12"), strict=True):
                out[f"R{name}"] = float(r[a, b])
        else:
            for d in range(3):
                out[f"R{d + 1}"] = float(r[d])
        out.update(self.solution.info.to_dict())
        return out


@dataclass
class Response:
    """Macroscopic response of a volume element to a named loading for one boundary
    condition: the macroscopic strain and stress (gradient and flux), prescribed or
    solved for."""

    name: str
    bc: str
    solution: Solution

    def components(self) -> dict[str, float]:
        """``E11 ... E12`` (engineering shears) and ``S11 ... S12``, or ``G1 ... G3``
        and ``Q1 ... Q3``, and the energy."""
        sol = self.solution
        if sol.physics == "elastic":
            e = tensor_to_voigt(sol.strain, strain=True)
            s = tensor_to_voigt(sol.stress, strain=False)
            out = {f"E{n}": float(v) for n, v in zip(VOIGT_NAMES, e, strict=True)}
            out.update({f"S{n}": float(v) for n, v in zip(VOIGT_NAMES, s, strict=True)})
        else:
            out = {f"G{d + 1}": float(sol.strain[d]) for d in range(3)}
            out.update({f"Q{d + 1}": float(sol.stress[d]) for d in range(3)})
        out["energy"] = float(sol.energy)
        return out

    def row(self) -> dict:
        """Flat record (long format) with the components and solver data."""
        out = {"property": self.name, "bc": self.bc, **self.components()}
        out["material_fraction"] = self.solution.material_fraction
        out.update(self.solution.info.to_dict())
        return out


def _shear_planes(plane) -> tuple[int, int]:
    a, b = sorted(int(x) for x in plane)
    if a == b or not (0 <= a < 3 and 0 <= b < 3):
        raise ValueError(f"shear plane: two different axes in 0..2, got {plane!r}")
    return a, b


def apparent_bulk_modulus(problem, *, fields: bool = False) -> Apparent:
    """The apparent bulk modulus of a problem under its boundary conditions (:class:`Apparent`):
    under a hydrostatic strain, or a hydrostatic stress with static uniform conditions
    (``subc``); the fields of the solution with ``fields``."""
    if problem.bc == "subc":
        sol = problem.solve(Loading.stress(np.eye(3)), fields=fields)
        tr = float(np.trace(sol.strain))
        return Apparent(
            "bulk_modulus",
            problem.bc,
            1.0 / tr,
            1.0 / sol.energy,
            bulk_isotropy(sol.strain),
            sol.strain,
            sol,
        )
    sol = problem.solve(Loading.strain(np.eye(3)), fields=fields)
    return Apparent(
        "bulk_modulus",
        problem.bc,
        float(np.trace(sol.stress) / 9.0),
        sol.energy / 9.0,
        bulk_isotropy(sol.stress),
        sol.stress,
        sol,
    )


def apparent_shear_modulus(problem, plane=(0, 1), *, fields: bool = False) -> Apparent:
    """The apparent shear modulus of a problem in a plane (its two axes; :class:`Apparent`): under a
    shear strain in it, or a shear stress with static uniform conditions (``subc``); the fields
    of the solution with ``fields``."""
    a, b = _shear_planes(plane)
    t = np.zeros((3, 3))
    if problem.bc == "subc":
        t[a, b] = t[b, a] = 1.0
        sol = problem.solve(Loading.stress(t), fields=fields)
        return Apparent(
            "shear_modulus",
            problem.bc,
            float(1.0 / (2.0 * sol.strain[a, b])),
            1.0 / sol.energy,
            shear_isotropy(sol.strain),
            sol.strain,
            sol,
        )
    t[a, b] = t[b, a] = 0.5
    sol = problem.solve(Loading.strain(t), fields=fields)
    return Apparent(
        "shear_modulus",
        problem.bc,
        float(sol.stress[a, b]),
        sol.energy,
        shear_isotropy(sol.stress),
        sol.stress,
        sol,
    )


def apparent_conductivity(problem, axis: int = 0, *, fields: bool = False) -> Apparent:
    """The apparent conductivity of a problem along an axis (:class:`Apparent`): under a gradient
    along it, or a flux with static uniform conditions (``subc``); the fields of the solution
    with ``fields``. The problem of another transport phenomenon gives its property
    (:data:`TRANSPORT_PROPERTIES`: the apparent electrical conductivity, permittivity,
    permeability or diffusivity)."""
    e = np.zeros(3)
    e[axis] = 1.0
    name = TRANSPORT_PROPERTIES[problem.physics][0]
    if problem.bc == "subc":
        sol = problem.solve(Loading.flux(-e, physics=problem.physics), fields=fields)
        return Apparent(
            name,
            problem.bc,
            float(1.0 / sol.strain[axis]),
            1.0 / sol.energy,
            thermal_isotropy(sol.strain, axis),
            sol.strain,
            sol,
        )
    sol = problem.solve(Loading.gradient(e, physics=problem.physics), fields=fields)
    return Apparent(
        name,
        problem.bc,
        float(-sol.stress[axis]) + 0.0,
        sol.energy,
        thermal_isotropy(sol.stress, axis),
        sol.stress,
        sol,
    )


def stiffness_tensor(problem) -> tuple[np.ndarray, list[Solution]]:
    """Apparent stiffness (6 x 6, Voigt order 11 22 33 23 31 12) from six loadings.

    KUBC and periodic conditions impose the six unit strains (Dirrenberger et al. 2014,
    appendix A); SUBC impose the six unit stresses and the compliance is inverted.
    """
    if problem.bc == "mixed":
        raise ValueError("the full stiffness tensor is defined for kubc, periodic or subc")
    c = np.zeros((6, 6))
    sols = []
    for j in range(6):
        unit = np.zeros(6)
        unit[j] = 1.0
        if problem.bc == "subc":
            sol = problem.solve(Loading.stress(unit))
            c[:, j] = tensor_to_voigt(sol.strain, strain=True)
        else:
            sol = problem.solve(Loading.strain(unit))
            c[:, j] = tensor_to_voigt(sol.stress, strain=False)
        sols.append(sol)
    if problem.bc == "subc":
        c = np.linalg.inv(c)
    return 0.5 * (c + c.T), sols


def thermal_expansion(problem) -> tuple[np.ndarray, Solution]:
    """Apparent thermal expansion tensor (3 x 3): the macroscopic strain of a unit change
    of temperature under zero macroscopic stress (``E = S : Sigma + A dT``)."""
    if problem.bc == "mixed":
        raise ValueError("the thermal expansion is defined for kubc, periodic or subc")
    sol = problem.solve(Loading.stress(np.zeros(6), temperature=1.0))
    return 0.5 * (sol.strain + sol.strain.T), sol


def conductivity_tensor(problem) -> tuple[np.ndarray, list[Solution]]:
    """Apparent conductivity tensor (3 x 3) from three gradient (or flux) loadings; that
    of the transport phenomenon of the problem (:data:`TRANSPORT_PROPERTIES`)."""
    if problem.bc == "mixed":
        raise ValueError("the conductivity tensor is defined for kubc, periodic or subc")
    k = np.zeros((3, 3))
    sols = []
    for d in range(3):
        e = np.zeros(3)
        e[d] = 1.0
        if problem.bc == "subc":
            sol = problem.solve(Loading.flux(-e, physics=problem.physics))
            k[:, d] = sol.strain
        else:
            sol = problem.solve(Loading.gradient(e, physics=problem.physics))
            k[:, d] = -sol.stress
        sols.append(sol)
    if problem.bc == "subc":
        k = np.linalg.inv(k)
    return 0.5 * (k + k.T), sols


def isotropic_projection(stiffness) -> tuple[float, float]:
    """Bulk and shear moduli of the isotropic part of a Voigt stiffness matrix."""
    c = np.asarray(stiffness, dtype=float)
    d = c[0, 0] + c[1, 1] + c[2, 2]
    o = c[0, 1] + c[0, 2] + c[1, 2]
    s = c[3, 3] + c[4, 4] + c[5, 5]
    return (d + 2.0 * o) / 9.0, (d - o + 3.0 * s) / 15.0


def isotropy_errors(stiffness) -> dict[str, float]:
    """Relative deviations from isotropy of a Voigt stiffness (Dirrenberger 2014, 5.1.4).

    ``normal``: spread of C11, C22, C33; ``coupling``: of C12, C13, C23; ``shear``: of
    C44, C55, C66; ``cauchy``: ``|C44 - (C11 - C12)/2| / C44`` with mean values; ``other``:
    largest remaining component relative to the mean of C11, C22, C33.
    """
    c = np.asarray(stiffness, dtype=float)

    def spread(values):
        v = np.asarray(values)
        m = v.mean()
        return float(np.max(np.abs(v - m)) / abs(m)) if m != 0 else float("nan")

    diag = [c[0, 0], c[1, 1], c[2, 2]]
    coup = [c[0, 1], c[0, 2], c[1, 2]]
    shear = [c[3, 3], c[4, 4], c[5, 5]]
    mask = np.ones((6, 6), dtype=bool)
    mask[:3, :3] = False
    mask[[3, 4, 5], [3, 4, 5]] = False
    return {
        "normal": spread(diag),
        "coupling": spread(coup),
        "shear": spread(shear),
        "cauchy": float(abs(np.mean(shear) - (np.mean(diag) - np.mean(coup)) / 2) / np.mean(shear)),
        "other": float(np.max(np.abs(c[mask])) / np.mean(diag)),
    }


def _cpu_only(device: str) -> None:
    """Finite elements run on the CPU: ``device`` may be ``"cpu"`` or ``"auto"``."""
    if device not in ("cpu", "auto"):
        raise ValueError(
            f"device {device!r}: the GPU runs the FFT solvers of voxel images; finite "
            "elements run on the CPU"
        )


def make_problem(
    sample,
    materials: dict[int, Material],
    physics: str,
    bc: str = "periodic",
    *,
    solver: str = "auto",
    tol: float = 1e-8,
    maxiter: int | None = None,
    grains=None,
    orientations=None,
    device: str = "cpu",
    integration: str | None = None,
    plane: str | None = None,
):
    """Boundary-value problem of a sample for ``physics`` 'elastic', 'thermal' or another
    transport phenomenon of a scalar potential (:data:`~morphogen.homogenization.materials.
    TRANSPORT`: 'electrical', 'dielectric', 'magnetic', 'diffusion'), which the thermal
    problems solve with the coefficient of the materials for it.

    A :class:`~morphogen.mesh.tetra.TetMesh` (conforming mesh) is solved by finite
    elements, with any boundary condition; a :class:`~morphogen.image.VoxelImage` by
    the FFT solver, with periodic conditions; a :class:`~morphogen.homogenization.xfem.
    LevelSetSample` (the level set of a morphology on a regular grid) by extended finite
    elements, with periodic conditions. ``solver`` is the linear solver of the
    finite element problems (see :mod:`morphogen.homogenization.solvers`), or ``"fenicsx"``
    to assemble and solve them by FEniCSx (:mod:`morphogen.homogenization.fenicsx`); ``device``
    that of the FFT solver (``"cpu"``, ``"gpu"`` or ``"auto"``, see :mod:`morphogen.gpu`).
    ``integration`` (elastic problems on a mesh of 10-node tetrahedra): ``"full"``, or
    ``"bbar"``, the volume change of each element as its mean, for nearly incompressible
    phases (:mod:`morphogen.homogenization.dilatation`).

    A :class:`~morphogen.mesh.triangles.TriMesh` (a mesh of the plane) is solved by the
    finite elements of the plane (:mod:`morphogen.homogenization.plane`), with periodic,
    kinematic or static conditions; ``plane``: ``"generalized"`` (default: the anti-plane
    fluctuation free, the structure in space) or ``"strain"`` (plane strain) for its
    elastic problems.

    Crystals (anisotropic materials) are turned by the orientation of their grain: the
    grains and orientations of a polycrystal mesh (``mesh.cell``, ``mesh.orientations``),
    or, for a voxel image, ``grains`` (image or array of grain labels, e.g.
    ``tessellation.labels(...)``) and ``orientations`` (Bunge angles of each grain).
    """
    elastic = not transport_physics(physics)
    if plane is not None and not isinstance(sample, TriMesh):
        raise ValueError("plane: the hypothesis of the elastic problems of a mesh of the plane")
    if isinstance(sample, TriMesh):
        if grains is not None or orientations is not None:
            raise ValueError("a mesh carries its own grains and orientations (mesh.cell)")
        _cpu_only(device)
        if solver == "fenicsx":
            raise ValueError("solver 'fenicsx' solves tetrahedral meshes, not those of the plane")
        if integration is not None and integration_of(integration) != "full":
            raise ValueError("integration 'bbar': the 10-node tetrahedra, not the triangles")
        options = {"solver": solver, "tol": tol, "maxiter": maxiter, "physics": physics}
        if elastic:
            return PlaneElasticProblem(sample, materials, bc, plane=plane or "generalized",
                                       **options)  # fmt: skip
        return PlaneThermalProblem(sample, materials, bc, **options)
    if isinstance(sample, LevelSetSample):
        if grains is not None or orientations is not None:
            raise ValueError("XFEM: two phases without grains (no orientations)")
        _cpu_only(device)
        if solver == "fenicsx":
            raise ValueError("solver 'fenicsx' solves conforming tetrahedral meshes")
        cls = XFEMElasticProblem if elastic else XFEMThermalProblem
        return cls(sample, materials, bc, solver=solver, tol=tol, maxiter=maxiter, physics=physics)
    if isinstance(sample, TetMesh):
        if grains is not None or orientations is not None:
            raise ValueError("a mesh carries its own grains and orientations (mesh.cell)")
        _cpu_only(device)
        if solver == "fenicsx":
            from morphogen.homogenization.fenicsx import (
                FenicsxElasticProblem,
                FenicsxThermalProblem,
            )

            cls = FenicsxElasticProblem if elastic else FenicsxThermalProblem
            if integration_of(integration) != "full":
                raise ValueError("integration 'bbar': the solvers of morphoGen, not 'fenicsx'")
        else:
            cls = FEElasticProblem if elastic else FEThermalProblem
        options = {"integration": integration_of(integration)} if elastic else {}
        if solver == "fenicsx":
            options = {}
        return cls(
            sample, materials, bc, solver=solver, tol=tol, maxiter=maxiter, physics=physics,
            **options,
        )  # fmt: skip
    if isinstance(sample, VoxelImage):
        if solver == "fenicsx":
            raise ValueError(
                "solver 'fenicsx' solves tetrahedral meshes: mesh the image "
                "(morphogen.mesh.conforming) or solve it by FFT"
            )
        cls = FFTElasticProblem if elastic else FFTThermalProblem
        return cls(
            sample,
            materials,
            bc,
            tol=tol,
            maxiter=maxiter,
            grains=grains,
            orientations=orientations,
            device=device,
            physics=physics,
        )
    raise TypeError(
        "homogenization needs a TetMesh (finite elements, see morphogen.mesh.conforming), a "
        "VoxelImage (FFT solver) or a LevelSetSample (extended finite elements), got "
        f"{type(sample).__name__}"
    )


def homogenize(
    sample,
    materials: dict[int, Material],
    *,
    bcs=("periodic",),
    properties=("bulk_modulus", "shear_modulus"),
    solver: str = "auto",
    tol: float = 1e-8,
    maxiter: int | None = None,
    fields: bool = False,
    grains=None,
    orientations=None,
    loadings: dict | None = None,
    paths: dict | None = None,
    nonlinear: dict | None = None,
    device: str = "cpu",
    ntfa: dict | None = None,
    mean_temperature: float = 300.0,
    frequencies=None,
    wavelengths=None,
    fracture=None,
    fluid_bulk_modulus: float | None = None,
    integration: str | None = None,
    plane: str | None = None,
) -> dict[str, Apparent | Response | PathResult | dict]:
    """Apparent properties of ``sample`` for every pair (property, boundary condition).

    ``sample`` is a conforming tetrahedral mesh (finite elements: ``kubc``, ``mixed``,
    ``periodic`` and ``subc`` conditions), a voxel image (FFT solver: ``periodic``
    only) or the level set of a morphology on a regular grid (:class:`~morphogen.
    homogenization.xfem.LevelSetSample`: extended finite elements, ``periodic`` only, whose
    interfaces cut the elements). Returns a mapping ``"<property>.<bc>"`` ->
    :class:`Apparent` (:data:`PROPERTIES`:
    the bulk and shear moduli, and the property of each transport phenomenon:
    ``conductivity``, ``electrical_conductivity``, ``permittivity``, ``permeability``,
    ``diffusivity``); the tensors (:data:`TENSORS`: ``stiffness``, ``thermal_expansion``,
    ``conductivity_tensor`` and the ``<property>_tensor`` of the other transports) map to
    ``{"tensor": array, ...}``, the stiffness with its ``compliance`` (``S = C^-1``, in the same
    notation: Voigt order 11 22 33 23 31 12 with engineering shears,
    :data:`morphogen.homogenization.postprocess.NOTATION`), its engineering constants and
    anisotropy indices. ``loadings`` (name -> :class:`Loading`, or a mapping for
    :meth:`Loading.from_spec`) are solved too: ``"<name>.<bc>"`` -> :class:`Response`.
    Problems sharing a boundary condition reuse the assembled system and the unit
    solutions. With ``fields``, the solutions keep the element fields (strain/stress or
    gradient/flux). ``grains`` and ``orientations``: crystal orientations of a voxel
    image (see :func:`make_problem`); ``device``: that of the FFT solver of an image
    (``"cpu"``, ``"gpu"`` or ``"auto"``, see :mod:`morphogen.gpu`).

    ``paths`` (name -> :class:`LoadingPath`, or a mapping for
    :meth:`LoadingPath.from_spec`) are followed in time with the nonlinear behaviours of
    the materials, under periodic and kinematic conditions (finite elements) or periodic
    ones (FFT): ``"<name>.<bc>"`` -> :class:`PathResult`; ``nonlinear``: options of
    :class:`NonlinearFEProblem` and :class:`NonlinearFFTProblem` (``newton_tol``,
    ``max_newton``, ``max_cuts``, ``kinematics``: ``"small"`` or ``"finite"`` strains;
    ``integration``: ``"bbar"`` or ``"full"``; ``tangent``: ``"auto"``,
    ``"symmetric"`` or ``"full"``, the consistent tangents of the points symmetrized for
    the conjugate gradients or kept whole for GMRES; ``contact``: True, or a mapping of
    the options ``friction``, ``stiffness``, ``tangential``, ``depth``, ``augmentations``
    and ``tolerance`` of :class:`~morphogen.contact.periodic.ContactCell`: on the mesh
    of the solid of a porous sample, the walls of the pores that close along a path
    press on each other, and on their images across the faces of a periodic cell, with
    Coulomb's friction — in finite strains unless ``kinematics`` says otherwise, the
    tangent factorized; the curves of the paths get the columns ``contact_area``,
    ``contact_pressure``, ``contact_penetration`` and ``contact_dissipation``;
    ``interfaces``: a mapping of the law of the interfaces of the phases of a mesh
    (:func:`~morphogen.fracture.periodic.interface_law`: ``law``, ``normal_strength``,
    ``mode_I``, ``friction``...), of the ``faces`` that are interfaces (``"phases"``,
    ``"grains"`` or pairs of phases), of the ``viscosity`` of their regularization and
    of the elastic ``slip`` of their friction; ``laws`` in the place of a law: one for
    each pair of phases, ``{"0-1": {...}, "0-2": {...}}``:
    :class:`~morphogen.fracture.periodic.CohesiveCell`, whose interfaces open, break,
    close again and rub along a path, the tangent factorized; the curves get the columns
    ``debonded_area``, ``interface_damage``, ``interface_opening`` and
    ``interface_dissipation``; ``microdamage``: a mapping of the options of the
    alternate scheme of the microdamage of regularized cleavage planes, its
    ``tolerance``, the most ``passes`` of a step and its ``acceleration``
    (:meth:`~morphogen.homogenization.nonlinear._PathSolver.microdamage_options`)).

    ``integration``: ``"full"`` integrates the strain energy at every point of the
    elements; ``"bbar"`` (also ``"reduced"``) takes the volume change of each element as
    its mean — selective reduced integration, ``B``-bar in small strains and ``F``-bar in
    finite strains (:mod:`morphogen.homogenization.dilatation`) —, which keeps the
    elements from locking when a phase is nearly incompressible (elastomers: Poisson's
    ratio near 1/2) or flows plastically. On a mesh of 10-node tetrahedra it applies to
    the elastic problems and to the loading paths (default ``"full"``); on a voxel image,
    to the loading paths (default ``"bbar"``: the linear FFT solver integrates its voxels
    at every point).
    With ``fields``, their element (voxel) fields are kept at the end of each segment
    (``fields="steps"``: of each step, the other solutions keeping their fields).

    The strong couplings (:data:`COUPLED`: ``piezoelectric``, ``piezomagnetic``,
    ``magnetoelectroelastic``, ``magnetoelectric``, ``thermoelectric``) are properties too:
    the fields of each are solved together on a voxel image, under periodic conditions
    (:mod:`morphogen.homogenization.coupled`), with the coupling coefficients of the
    materials (``Material(coupling=...)``), and ``"<coupling>.periodic"`` maps to the
    tensors of the effective coupled law and the constants derived from them
    (:func:`~morphogen.homogenization.coupled.split_law`); ``mean_temperature``: the mean
    absolute temperature of the thermoelectric law (K).

    The properties of a harmonic regime (:data:`HARMONIC`, voxel images;
    :mod:`morphogen.homogenization.harmonic`): ``dynamic_stiffness``, the complex stiffness
    of viscoelastic phases, and ``complex_conductivity``, that of leaky dielectrics
    (``sigma + i omega eps``), at the angular ``frequencies``; ``refractive_index``, the
    effective index of the ``refractive_index`` of the materials at the ``wavelengths``
    (the long-wavelength limit of optics). ``"<property>.periodic"`` maps to the storage
    and loss parts of the tensors at each frequency and their scalar entries
    (:meth:`~morphogen.homogenization.harmonic.HarmonicResponse.to_dict`).

    ``fracture`` (voxel images; a :class:`~morphogen.fracture.phasefield.PhaseField`, or
    a mapping of its parameters with the options ``staggered_tol`` and ``max_staggered``
    of its solver): the ``paths`` are followed by the phase-field method of brittle
    fracture (:mod:`morphogen.fracture.phasefield`), the phases elastic, and
    ``"<name>.periodic"`` maps to a :class:`~morphogen.fracture.phasefield.
    PhaseFieldResult`, whose quantities ``strength`` and ``work`` are those of the
    statistics of representative volumes.

    The flow of a fluid through the pores (:data:`FLOW`; voxel images of cubic voxels,
    the pores their void phases): ``hydraulic_permeability``, the permeability tensor of
    Darcy's law by the lattice Boltzmann method (:mod:`morphogen.homogenization.flow`),
    and ``acoustic``, the parameters of the equivalent fluid of the rigid porous frame
    (porosity, permeability, tortuosity, viscous and thermal characteristic lengths,
    thermal permeability: :mod:`morphogen.homogenization.acoustics`, whose
    ``equivalent_fluid`` and ``absorption`` give the sound absorption of a layer).

    The weak couplings of :mod:`morphogen.homogenization.poroelastic` are properties of
    the elastic solvers: ``swelling``, the tensor of the strains of a unit content of an
    absorbed species (``Material(swelling=...)``; as ``thermal_expansion``), and
    ``poroelastic``, the law of a porous sample whose pores (the voids) hold a fluid under
    pressure, under periodic or kinematic uniform conditions: its drained stiffness, the
    tensor and the modulus of Biot and, with the bulk modulus of the fluid
    ``fluid_bulk_modulus``, its undrained response (Gassmann, Skempton).

    A triangular mesh of the plane (:class:`~morphogen.mesh.triangles.TriMesh`) is solved
    by the finite elements of the plane (:mod:`morphogen.homogenization.plane`; ``kubc``,
    ``periodic`` and ``subc`` conditions on its sides): the elastic properties and tensors
    (:data:`PLANE_PROPERTIES`), those of the transport phenomena and the named loadings, of
    its structure invariant across the plane; ``plane``: ``"generalized"`` (default, the
    anti-plane fluctuation free) or ``"strain"`` (plane strain).

    ``ntfa`` (voxel images): the ``paths`` are followed by a reduced model instead
    (:mod:`morphogen.homogenization.ntfa`), trained on the paths of ``ntfa["training"]``
    with ``ntfa["modes"]`` modes per plastic phase (default 6): ``"<name>.ntfa"`` ->
    :class:`PathResult`; ``ntfa["save"]``: a file for the model; ``ntfa["full_field"]``:
    the full-field paths too, for comparison.
    """
    props = list(properties)
    path_fields, fields = fields, bool(fields)
    named = {str(k): Loading.from_spec(v) for k, v in (loadings or {}).items()}
    named_paths = {str(k): LoadingPath.from_spec(v) for k, v in (paths or {}).items()}
    unknown = [p for p in props if p not in PHYSICS]
    if unknown:
        raise ValueError(f"unknown properties {unknown}; available: {list(PHYSICS)}")
    bc_names = [boundary_condition(b) for b in bcs]
    if plane is not None and plane not in PLANES:
        raise ValueError(f"plane: one of {PLANES}")
    if isinstance(sample, TriMesh):
        allowed = set(PLANE_PROPERTIES) | {n for names in TRANSPORT_PROPERTIES.values()
                                           for n in names}  # fmt: skip
        other = [p for p in props if p not in allowed]
        if other:
            raise ValueError(f"{other}: not by the finite elements of the plane, which give "
                             f"{sorted(allowed)}")  # fmt: skip
        if paths or ntfa or fracture is not None or nonlinear:
            raise ValueError("the loading paths and the nonlinear behaviours: meshes in "
                             "space (TriMesh.extruded: tetrahedra) or images")  # fmt: skip
        if "mixed" in bc_names:
            raise ValueError("the problems of the plane take the conditions kubc, periodic "
                             "and subc")  # fmt: skip
    elif plane is not None:
        raise ValueError("plane: the hypothesis of the elastic problems of a mesh of the plane")
    if isinstance(sample, (VoxelImage, LevelSetSample)):
        other = [b for b in bc_names if b != "periodic"]
        if other:
            raise ValueError(
                f"boundary conditions {other} need a conforming mesh (finite elements): the "
                "FFT solver of voxel images and the extended finite elements of level sets "
                "are periodic"
            )
    out: dict = {}
    options = {"solver": solver, "tol": tol, "maxiter": maxiter}
    if grains is not None or orientations is not None:
        options.update(grains=grains, orientations=orientations)
    if isinstance(sample, VoxelImage):
        options["device"] = device
    else:
        _cpu_only(device)
    nonlinear = dict(nonlinear or {})
    # the options of the microdamage of regularized cleavage planes (the loading paths)
    microdamage = nonlinear.pop("microdamage", None)
    contact = _contact_options(nonlinear.pop("contact", None), sample, solver)
    interfaces = _interface_options(nonlinear.pop("interfaces", None), sample, solver)
    if contact is not None and interfaces is not None:
        raise ValueError(
            "nonlinear: 'contact' (the walls of the pores) or 'interfaces' (those of the "
            "phases, whose lips touch and rub once broken), not both"
        )
    if integration is not None:
        if isinstance(sample, LevelSetSample):
            raise ValueError(
                "integration: an option of the finite elements of a mesh and of the voxels "
                "of the loading paths, not of the extended finite elements"
            )
        if fracture is None:  # the phase field has its own elements
            nonlinear.setdefault("integration", integration_of(integration))
    elif "integration" in nonlinear and isinstance(sample, TetMesh):
        integration = nonlinear["integration"] = integration_of(nonlinear["integration"])
    for bc in bc_names:
        elastic = [p for p in props if PHYSICS[p] == "elastic"]
        # the properties of each physics, and the problems of those in use
        wanted = {"elastic": elastic}
        wanted.update({ph: [p for p in props if PHYSICS[p] == ph] for ph in TRANSPORT_PROPERTIES})
        problems = {}
        for physics, mine in wanted.items():
            if physics == "elastic":  # the weak couplings solve problems of their own
                mine = [p for p in mine if p not in ("swelling", "poroelastic")]
            if mine or any(lo.physics == physics for lo in named.values()):
                extra = {}
                if physics == "elastic" and isinstance(sample, TetMesh):
                    extra["integration"] = integration
                if physics == "elastic" and isinstance(sample, TriMesh):
                    extra["plane"] = plane
                problems[physics] = make_problem(sample, materials, physics, bc, **options, **extra)
        for prop in (p for p in props if PHYSICS[p] == "coupled"):
            checkpoint()
            if not isinstance(sample, VoxelImage):
                raise ValueError(
                    f"{prop}: the coupled problems are solved on voxel images (the FFT "
                    "solver, periodic conditions), not on meshes"
                )
            out[f"{prop}.{bc}"] = coupled_tensors(
                sample, materials, prop, grains=grains, orientations=orientations,
                temperature=mean_temperature, tol=tol, maxiter=maxiter,
            )  # fmt: skip
        for prop in (p for p in props if PHYSICS[p] == "harmonic"):
            checkpoint()
            out[f"{prop}.{bc}"] = _harmonic(
                sample,
                materials,
                prop,
                frequencies,
                wavelengths,
                grains,
                orientations,
                tol,
                maxiter,
            )
        flows = [p for p in props if PHYSICS[p] == "flow"]
        if flows:
            checkpoint()
            for prop, res in _flow(sample, materials, flows, tol).items():
                out[f"{prop}.{bc}"] = res
        for name, loading in named.items():
            checkpoint()
            if bc == "mixed" and not loading.strain_controlled.all():
                log.info("loading %s skipped for mixed boundary conditions", name)
                continue
            sol = problems[loading.physics].solve(loading, fields=fields)
            out[f"{name}.{bc}"] = Response(name, bc, sol)
        if elastic:
            problem = problems.get("elastic")
            for prop in elastic:
                checkpoint()  # each property a stop of morphogen.stopping
                if prop in ("swelling", "poroelastic"):
                    if bc == "mixed" or (prop == "poroelastic" and bc == "subc"):
                        log.info("%s skipped for %s boundary conditions", prop, bc)
                        continue
                    out[f"{prop}.{bc}"] = _weak_coupling(
                        sample, materials, prop, bc, fluid_bulk_modulus, options
                    )
                elif prop == "bulk_modulus":
                    out[f"{prop}.{bc}"] = apparent_bulk_modulus(problem, fields=fields)
                elif prop == "shear_modulus":
                    out[f"{prop}.{bc}"] = apparent_shear_modulus(problem, fields=fields)
                elif bc == "mixed":
                    log.info("%s skipped for mixed boundary conditions", prop)
                elif prop == "stiffness":
                    tensor, _ = stiffness_tensor(problem)
                    k, mu = isotropic_projection(tensor)
                    out[f"{prop}.{bc}"] = {
                        "tensor": tensor,
                        "compliance": compliance(tensor),
                        "bulk_modulus": k,
                        "shear_modulus": mu,
                        "isotropy_errors": isotropy_errors(tensor),
                        **engineering_constants(tensor),
                        **anisotropy(tensor),
                    }
                else:
                    tensor, _ = thermal_expansion(problem)
                    out[f"{prop}.{bc}"] = {
                        "tensor": tensor,
                        **{
                            f"A{name}": float(tensor[a, b])
                            for (a, b), name in zip(VOIGT, VOIGT_NAMES, strict=True)
                        },
                        "mean": float(np.trace(tensor) / 3.0),
                    }
        for physics, (scalar, _) in TRANSPORT_PROPERTIES.items():
            for prop in wanted[physics]:
                problem = problems[physics]
                checkpoint()
                if prop == scalar:
                    out[f"{prop}.{bc}"] = apparent_conductivity(problem, fields=fields)
                else:
                    if bc == "mixed":
                        log.info("%s skipped for mixed boundary conditions", prop)
                        continue
                    tensor, _ = conductivity_tensor(problem)
                    out[f"{prop}.{bc}"] = {"tensor": tensor, "mean": float(np.trace(tensor) / 3)}
        if named_paths and (not ntfa or ntfa.get("full_field")):
            if bc not in ("periodic", "kubc"):
                log.info("loading paths skipped for %s conditions", bc)
                continue
            if fracture is not None:
                problem = _fracture_problem(
                    sample, materials, fracture, grains, orientations, nonlinear
                )
            elif isinstance(sample, VoxelImage):
                problem = NonlinearFFTProblem(
                    sample, materials, bc, tol=tol, maxiter=maxiter, grains=grains,
                    orientations=orientations, device=device, **(nonlinear or {}),
                )  # fmt: skip
            elif isinstance(sample, LevelSetSample):
                if "integration" in (nonlinear or {}):
                    raise ValueError(
                        "nonlinear: 'integration' is an option of the voxels of the FFT solver"
                    )
                problem = NonlinearXFEMProblem(
                    sample, materials, bc, solver=solver, tol=tol, maxiter=maxiter,
                    **(nonlinear or {}),
                )  # fmt: skip
            elif contact is not None:
                from morphogen.contact.periodic import ContactCell

                problem = ContactCell(
                    sample, materials, bc, tol=tol, maxiter=maxiter, **contact, **nonlinear
                )
            elif interfaces is not None:
                from morphogen.fracture.periodic import CohesiveCell

                law, cell = interfaces
                problem = CohesiveCell(
                    sample, materials, law, bc, tol=tol, maxiter=maxiter, **cell, **nonlinear
                )
            else:
                problem = NonlinearFEProblem(
                    sample, materials, bc, solver=solver, tol=tol, maxiter=maxiter,
                    **(nonlinear or {}),
                )  # fmt: skip
            if microdamage and hasattr(problem, "microdamage_options"):
                problem.microdamage_options(**microdamage)
            for name, path in named_paths.items():
                out[f"{name}.{bc}"] = problem.run(path, name=name, fields=path_fields)
    if ntfa:
        out.update(_ntfa_paths(sample, materials, named_paths, ntfa, tol, nonlinear))
    return out


#: The options of the contact of the walls of the pores along the loading paths
#: (``nonlinear["contact"]``): those of morphogen.contact.periodic.ContactCell
CONTACT_OPTIONS = ("friction", "stiffness", "tangential", "depth", "points", "augmentations",
                   "tolerance")  # fmt: skip


def _contact_options(contact, sample, solver: str) -> dict | None:
    """The options of the cell whose walls may touch (``nonlinear["contact"]``: True or a
    mapping of :data:`CONTACT_OPTIONS`), or None without contact."""
    if contact is None or contact is False:
        return None
    if not isinstance(sample, TetMesh):
        raise ValueError(
            "nonlinear: 'contact' is that of the walls of the pores of a mesh (finite "
            "elements on the conforming mesh of the solid), not of a voxel image or a level set"
        )
    if solver not in ("auto", "direct"):
        raise ValueError(
            f"nonlinear: with 'contact' the tangent is factorized (solver 'direct', not {solver!r})"
        )
    if contact is True:
        return {}
    if not isinstance(contact, dict):
        raise ValueError("nonlinear: 'contact' is True or a mapping of its options")
    unknown = sorted(set(contact) - set(CONTACT_OPTIONS))
    if unknown:
        raise ValueError(
            f"nonlinear: unknown options {unknown} of 'contact'; use {list(CONTACT_OPTIONS)}"
        )
    return dict(contact)


#: The options of the interfaces of the phases along the loading paths
#: (``nonlinear["interfaces"]``): their law and its parameters
#: (morphogen.fracture.periodic.interface_law), and those of
#: morphogen.fracture.periodic.CohesiveCell
INTERFACE_OPTIONS = (
    "law",
    "stiffness",
    "normal_strength",
    "shear_strength",
    "mode_I",
    "mode_II",
    "eta",
    "strength",
    "energy",
    "beta",
    "penalty",
    "tangential",
    "friction",
    "faces",
    "viscosity",
    "slip",
    "laws",
)


def _interface_options(interfaces, sample, solver: str) -> tuple[dict, dict] | None:
    """The law of the interfaces and the options of their cell (``nonlinear["interfaces"]``:
    a mapping of :data:`INTERFACE_OPTIONS`), or None without interfaces."""
    if interfaces is None or interfaces is False:
        return None
    if not isinstance(sample, TetMesh):
        raise ValueError(
            "nonlinear: 'interfaces' are those of the phases of a mesh (finite elements on "
            "a conforming mesh of the phases), not of a voxel image or a level set"
        )
    if solver not in ("auto", "direct"):
        raise ValueError(
            "nonlinear: with 'interfaces' the tangent is factorized (solver 'direct', not "
            f"{solver!r})"
        )
    if not isinstance(interfaces, dict):
        raise ValueError("nonlinear: 'interfaces' is a mapping of their law and options")
    unknown = sorted(set(interfaces) - set(INTERFACE_OPTIONS))
    if unknown:
        raise ValueError(
            f"nonlinear: unknown options {unknown} of 'interfaces'; use {list(INTERFACE_OPTIONS)}"
        )
    cell = {k: interfaces[k] for k in ("faces", "viscosity", "slip") if k in interfaces}
    law = {k: v for k, v in interfaces.items() if k not in cell}
    if "laws" in law:  # a law per pair of phases: "0-1", or the two phases
        laws = law.pop("laws")
        if law or not isinstance(laws, dict) or not laws:
            raise ValueError(
                "nonlinear: 'interfaces' with 'laws' (a mapping of pairs of phases to "
                f"their laws) takes no law of its own: {sorted(law)}"
            )
        try:
            law = {
                tuple(sorted(int(p) for p in (key.split("-") if isinstance(key, str) else key))):
                spec
                for key, spec in laws.items()
            }  # fmt: skip
        except (TypeError, ValueError):
            raise ValueError(
                "nonlinear: the 'laws' of the interfaces are those of pairs of phases "
                f"('0-1'), got {list(laws)}"
            ) from None
    return law, cell


def _flow(sample, materials, props, tol) -> dict:
    """The permeability tensor and the acoustic parameters (those of ``props``) of the
    pores of a voxel image (:mod:`morphogen.homogenization.flow`,
    :mod:`morphogen.homogenization.acoustics`); the flows are solved once for both."""
    from morphogen.homogenization.acoustics import acoustic_parameters
    from morphogen.homogenization.flow import ENTRIES, permeability_tensor

    if not isinstance(sample, VoxelImage):
        raise ValueError(
            f"{props[0]}: the flow through the pores is solved on voxel images (the lattice "
            "Boltzmann method, periodic conditions), not on meshes"
        )
    lattice = min(max(tol, 1e-6), 1e-3)  # of the mean velocity of the flows
    out = {}
    if "acoustic" in props:
        out["acoustic"] = acoustic_parameters(sample, materials, tol=lattice)
    if "hydraulic_permeability" in props:
        if "acoustic" in out:
            res = out["acoustic"]
            flow = {"tensor": res["permeability_tensor"], "mean": res["permeability"]}
            flow.update({k: res[k] for k in (*ENTRIES[1:], "steps", "error")})
            out["hydraulic_permeability"] = flow
        else:
            out["hydraulic_permeability"] = permeability_tensor(sample, materials, tol=lattice)
    return {p: out[p] for p in props}


def _weak_coupling(sample, materials, prop, bc, fluid_bulk, options) -> dict:
    """The swelling tensor or the poroelastic law of a sample
    (:mod:`morphogen.homogenization.poroelastic`)."""
    from morphogen.homogenization.poroelastic import poroelastic_tensors, swelling_tensor

    if prop == "poroelastic":
        return poroelastic_tensors(sample, materials, bc, fluid_bulk=fluid_bulk, **options)
    tensor, _ = swelling_tensor(sample, materials, bc, **options)
    names = zip(VOIGT, VOIGT_NAMES, strict=True)
    return {
        "tensor": tensor,
        **{f"B{name}": float(tensor[a, b]) for (a, b), name in names},
        "mean": float(np.trace(tensor) / 3.0),
    }


def _fracture_problem(sample, materials, fracture, grains, orientations, nonlinear=None):
    """The phase-field problem of the loading paths of a voxel image that breaks: in small
    strains, or in finite strains with ``nonlinear: {kinematics: finite}`` (and the other
    options of the nonlinear FFT solver)."""
    from morphogen.fracture.phasefield import PhaseField, PhaseFieldProblem

    if not isinstance(sample, VoxelImage):
        raise ValueError(
            "fracture: the phase-field method runs on voxel images (the FFT solver, periodic "
            "conditions), not on meshes"
        )
    options = {}
    if isinstance(fracture, dict):
        fracture = dict(fracture)
        options = {k: fracture.pop(k) for k in ("staggered_tol", "max_staggered", "tol")
                   if k in fracture}  # fmt: skip
    nonlinear = dict(nonlinear or {})
    if nonlinear.pop("kinematics", "small") == "finite":
        from morphogen.fracture.phasefield_finite import FinitePhaseFieldProblem

        if grains is not None or orientations is not None:
            raise ValueError("fracture in finite strains: isotropic phases (no crystals)")
        return FinitePhaseFieldProblem(
            sample, materials, PhaseField.from_config(fracture), **options, **nonlinear
        )
    return PhaseFieldProblem(sample, materials, PhaseField.from_config(fracture), grains=grains,
                             orientations=orientations, **options)  # fmt: skip


def _harmonic(sample, materials, prop, frequencies, wavelengths, grains, orientations, tol,
              maxiter) -> dict:  # fmt: skip
    """A property of a harmonic regime (:data:`HARMONIC`) of a voxel image."""
    from morphogen.homogenization import harmonic

    if not isinstance(sample, VoxelImage):
        raise ValueError(
            f"{prop}: the harmonic problems are solved on voxel images (the FFT solver, "
            "periodic conditions), not on meshes"
        )
    if prop == "refractive_index":
        if wavelengths is None:
            raise ValueError("refractive_index: give the wavelengths")
        return harmonic.effective_index(
            sample, materials, wavelengths, tol=tol, maxiter=maxiter
        ).to_dict()
    if frequencies is None:
        raise ValueError(f"{prop}: give the angular frequencies")
    solve = (
        harmonic.dynamic_stiffness if prop == "dynamic_stiffness" else harmonic.complex_conductivity
    )
    return solve(sample, materials, frequencies, grains=grains, orientations=orientations,
                 tol=tol, maxiter=maxiter).to_dict()  # fmt: skip


def _ntfa_paths(sample, materials, named_paths, spec: dict, tol: float, nonlinear) -> dict:
    """The loading paths followed by an NTFA model trained on the sample."""
    from morphogen.homogenization.ntfa import train_ntfa

    if not isinstance(sample, VoxelImage):
        raise ValueError("ntfa: a periodic voxel image (the reduced model is trained by FFT)")
    unknown = set(spec) - {"training", "modes", "save", "full_field"}
    if unknown:
        raise ValueError(f"ntfa: unknown options {sorted(unknown)}")
    training = [LoadingPath.from_spec(p) for p in dict(spec.get("training") or {}).values()]
    if not training:
        raise ValueError("ntfa: give the training paths ('training')")
    newton = (nonlinear or {}).get("newton_tol", 1e-6)
    model = train_ntfa(
        sample, materials, training, modes=spec.get("modes", 6), tol=tol, newton_tol=newton
    )
    if spec.get("save"):
        model.save(spec["save"])
    return {f"{name}.ntfa": model.run(path, name=name) for name, path in named_paths.items()}
