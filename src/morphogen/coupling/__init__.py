"""Couplings of transport, chemistry and mechanics: transient diffusion in voxel images,
the thermal oxidation of polymers, properties that evolve with ageing, and hygromorph
plates."""

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
    "arc_curvature",
    "graded_materials",
    "oxidation_field",
    "oxidation_stresses",
    "slab_profile",
    "slab_uptake",
    "strip_curvature",
    "swelling",
    "timoshenko_curvature",
    "transient_diffusion",
]
