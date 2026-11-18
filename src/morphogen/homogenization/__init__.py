"""Computational homogenization: apparent properties of volume elements.

Two solvers, chosen by the representation of the sample:

* finite elements on conforming tetrahedral meshes of the implicit morphologies
  (:func:`morphogen.mesh.conforming.conforming_mesh`), with kinematic (``kubc``),
  mixed, periodic or static (``subc``) boundary conditions;
* an FFT-preconditioned solver on voxel images, with periodic conditions;
* extended finite elements on a regular grid cut by the level set of the morphology
  (:mod:`morphogen.homogenization.xfem`), with periodic conditions: no mesher, no stairs.

All solve linear elasticity (with thermal expansion) and the transport phenomena of a
scalar potential: heat and electrical conduction, dielectrics, magnetostatics and
diffusion (:data:`~morphogen.homogenization.materials.TRANSPORT`). The strong couplings
between these physics (piezoelectricity, piezomagnetism, magneto-electro-elasticity,
thermoelectricity) are solved field by field together on voxel images
(:mod:`morphogen.homogenization.coupled`).

Materials may yield (von Mises elasto-(visco)plasticity, :class:`Plasticity`), slip
on the systems of their lattice (crystal plasticity of Méric and Cailletaud,
:class:`CrystalPlasticity`), relax (:class:`Viscoelasticity`), stretch as rubbers
(:class:`Hyperelasticity`) or soften (:class:`Damage`): loading paths
(:class:`LoadingPath`) are then followed in
time, in small or finite strains, with the macroscopic strain components as degrees of
freedom (the periodic element): by finite elements (:class:`NonlinearFEProblem`) or by
the FFT solver of voxel images (:class:`NonlinearFFTProblem`); :func:`simulate_point`
integrates a material point. The stresses of the points along a cycle of a path are
judged by criteria of multiaxial fatigue (:mod:`morphogen.homogenization.fatigue`:
Crossland, Sines, Dang Van, Findley, Matake), and those of a path by the criterion of
cleavage of Beremin (:mod:`morphogen.homogenization.beremin`: the Weibull stress).

Typical use::

    from morphogen.homogenization import Material, VOID, homogenize
    from morphogen.mesh import conforming_mesh

    mesh = conforming_mesh(micro, size=0.4, phases="solid")
    materials = {0: VOID, 1: Material(young=210e3, poisson=0.3, conductivity=100)}
    results = homogenize(mesh, materials, bcs=["kubc", "mixed"],
                         properties=["bulk_modulus", "shear_modulus", "conductivity"])
    results["bulk_modulus.kubc"].value
"""

from morphogen.homogenization.acoustics import (
    absorption,
    acoustic_parameters,
    equivalent_fluid,
    surface_absorption,
)
from morphogen.homogenization.apparent import (
    PROPERTIES,
    TENSORS,
    Apparent,
    Response,
    apparent_bulk_modulus,
    apparent_conductivity,
    apparent_shear_modulus,
    conductivity_tensor,
    homogenize,
    isotropic_projection,
    isotropy_errors,
    make_problem,
    stiffness_tensor,
    thermal_expansion,
)
from morphogen.homogenization.behaviours import Plasticity
from morphogen.homogenization.beremin import Beremin, path_beremin
from morphogen.homogenization.bifurcation import (
    Bifurcation,
    PathBifurcation,
    path_bifurcation,
    post_bifurcation,
)
from morphogen.homogenization.bloch import (
    BandStructure,
    band_structure,
    brillouin_path,
    long_wavelength_speeds,
    scalar_band_structure,
)
from morphogen.homogenization.bounds import bounds
from morphogen.homogenization.core import (
    BOUNDARY_CONDITIONS,
    Loading,
    PhaseStatistics,
    Solution,
    boundary_condition,
)
from morphogen.homogenization.coupled import (
    CoupledProblem,
    Coupling,
    coupled_problem,
    coupled_tensors,
)
from morphogen.homogenization.crystal import CrystalPlasticity
from morphogen.homogenization.fatigue import (
    Crossland,
    DangVan,
    Findley,
    Matake,
    Sines,
    fatigue,
    path_fatigue,
    stress_amplitude,
)
from morphogen.homogenization.fe import FEElasticProblem, FEThermalProblem
from morphogen.homogenization.fft import FFTElasticProblem, FFTThermalProblem
from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem
from morphogen.homogenization.flow import permeability_tensor, stokes_flow
from morphogen.homogenization.frames import Frame, SpaceFrame
from morphogen.homogenization.laws import (
    Damage,
    Hyperelasticity,
    Viscoelasticity,
    fit_hyperelasticity,
)
from morphogen.homogenization.materials import VOID, Material, materials_from_config
from morphogen.homogenization.nonlinear import KINEMATICS, NonlinearFEProblem, simulate_point
from morphogen.homogenization.ntfa import NTFAModel, train_ntfa
from morphogen.homogenization.paths import LoadingPath, PathResult
from morphogen.homogenization.photonic import electromagnetic_band_structure
from morphogen.homogenization.poroelastic import poroelastic_tensors, swelling_tensor
from morphogen.homogenization.xfem import (
    LevelSetSample,
    NonlinearXFEMProblem,
    XFEMElasticProblem,
    XFEMThermalProblem,
)

__all__ = [
    "BOUNDARY_CONDITIONS",
    "KINEMATICS",
    "PROPERTIES",
    "TENSORS",
    "VOID",
    "Apparent",
    "BandStructure",
    "Beremin",
    "Bifurcation",
    "CoupledProblem",
    "Coupling",
    "Crossland",
    "CrystalPlasticity",
    "Damage",
    "DangVan",
    "FEElasticProblem",
    "FEThermalProblem",
    "FFTElasticProblem",
    "FFTThermalProblem",
    "Findley",
    "Frame",
    "Hyperelasticity",
    "LevelSetSample",
    "Loading",
    "LoadingPath",
    "Matake",
    "Material",
    "NTFAModel",
    "NonlinearFEProblem",
    "NonlinearFFTProblem",
    "NonlinearXFEMProblem",
    "PathBifurcation",
    "PathResult",
    "PhaseStatistics",
    "Plasticity",
    "Response",
    "Sines",
    "Solution",
    "SpaceFrame",
    "Viscoelasticity",
    "XFEMElasticProblem",
    "XFEMThermalProblem",
    "absorption",
    "acoustic_parameters",
    "apparent_bulk_modulus",
    "apparent_conductivity",
    "apparent_shear_modulus",
    "band_structure",
    "boundary_condition",
    "bounds",
    "brillouin_path",
    "conductivity_tensor",
    "coupled_problem",
    "coupled_tensors",
    "electromagnetic_band_structure",
    "equivalent_fluid",
    "fatigue",
    "fit_hyperelasticity",
    "homogenize",
    "isotropic_projection",
    "isotropy_errors",
    "long_wavelength_speeds",
    "make_problem",
    "materials_from_config",
    "path_beremin",
    "path_bifurcation",
    "path_fatigue",
    "permeability_tensor",
    "poroelastic_tensors",
    "post_bifurcation",
    "scalar_band_structure",
    "simulate_point",
    "stiffness_tensor",
    "stokes_flow",
    "stress_amplitude",
    "surface_absorption",
    "swelling_tensor",
    "thermal_expansion",
    "train_ntfa",
]
