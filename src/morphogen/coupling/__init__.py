"""Couplings of transport, chemistry and mechanics: transient diffusion in voxel images,
the thermal oxidation of polymers, properties that evolve with ageing, hygromorph
plates, and the thermo-chemo-elastic fields of a microstructure in time."""

from morphogen.coupling.ageing import Hydrolysis, PropertyLaw, graded_materials
from morphogen.coupling.diffusion import (
    DiffusionResult,
    slab_profile,
    slab_uptake,
    transient_diffusion,
)
from morphogen.coupling.fenicsx import (
    OxidationResult,
    SwellingResult,
    arc_curvature,
    oxidation_stresses,
    strip_curvature,
    swelling,
)
from morphogen.coupling.oxidation import ClosedLoopOxidation, OxidationField, oxidation_field
from morphogen.coupling.plates import Layer, Multilayer, timoshenko_curvature
from morphogen.coupling.thermochemoelastic import (
    ThermoChemoElastic,
    ThermoChemoElasticResult,
    ThermoChemoMaterial,
    arrhenius,
    beam_curvature,
    thermochemoelastic,
)

__all__ = [
    "ClosedLoopOxidation",
    "DiffusionResult",
    "Hydrolysis",
    "Layer",
    "Multilayer",
    "OxidationField",
    "OxidationResult",
    "PropertyLaw",
    "SwellingResult",
    "ThermoChemoElastic",
    "ThermoChemoElasticResult",
    "ThermoChemoMaterial",
    "arc_curvature",
    "arrhenius",
    "beam_curvature",
    "graded_materials",
    "oxidation_field",
    "oxidation_stresses",
    "slab_profile",
    "slab_uptake",
    "strip_curvature",
    "swelling",
    "thermochemoelastic",
    "timoshenko_curvature",
    "transient_diffusion",
]
