"""Architectured sheets by localized laser treatment: the simulation and the optimization
of the laser path that hardens (or softens, carburizes, nitrides, alloys) a pattern of
tracks in a metal sheet, for a mechanical load case.

The chain of a study (after Wang's thesis, PIMM 2022, and the ANR project SCOLASTIC):

* :mod:`morphogen.laser.heat`: the moving beams on a plate (one face, or both faces at
  once), quasi-steady cycles of a track (Rosenthal's limits) and transient finite volumes
  of a whole blank;
* :mod:`morphogen.laser.metallurgy`: austenite, martensite and tempering of C–Mn steels
  (lever rule, Koistinen–Marburger, Maynier, Hollomon–Jaffe), the peak-temperature
  hardening of ferritic stainless steels, melt-pool carburizing, nitriding and alloying;
* :mod:`morphogen.laser.materials`: flow laws from tensile properties (Considère), the
  strength of each hardness, the steels of the PIMM studies (AISI 430, annealed
  MS1500, DP1180);
* :mod:`morphogen.laser.patterns`: laser paths (straight, corrugated, Fourier, crossed,
  honeycomb tracks; G-code);
* :mod:`morphogen.laser.process`: the hardness a path leaves at the points of a sheet;
* :mod:`morphogen.laser.specimen` and :mod:`morphogen.laser.mechanics`: tensile
  specimens and their finite elements (a layered membrane in plane stress, large
  rotations, Considère's necking, the fracture of brittle points, local or nonlocal;
  the replay of the history of a test with other flow laws);
* :mod:`morphogen.laser.bending`: three-point bending of a strip (fibre sections);
* :mod:`morphogen.laser.design`: the measures, objectives and constraints of a case
  and the optimization of its path (grid, Nelder–Mead, differential evolution, genetic,
  Bayesian);
* :mod:`morphogen.laser.cells`: cellular designs (phases on a grid of cells, the
  ideal two-phase sheets of Wang et al. 2023) and their optimization by cellular
  automata (BESO, neighbour exchanges, HCA) and by densities (MMA), the tracks that
  draw them;
* :mod:`morphogen.laser.structures`: load cases beyond the tensile test, a compact
  tension specimen (the initiation of a crack) and an L-shaped plate (its strength);
* :mod:`morphogen.laser.periodic`: the periodic cell of a sheet treated all over, its
  homogenized tensile tests in any direction (the anisotropy of the sheet itself: a
  rolling texture compensated by the pattern);
* :mod:`morphogen.laser.architectures`: any morphology treated at the scale of the spot,
  its treatment and the sheet it makes (the periodic cell of a design of pixels), smooth
  tracks over a periodic cell, phases of different stiffness;
* :mod:`morphogen.laser.solid`: the treated sheet in its volume, voxels of graded
  elastoplastic phases with free faces for the solvers of three-dimensional fields
  (its tensile test by FFT, its fields across the thickness);
* :mod:`morphogen.laser.run`: configuration files and outputs (``morphogen laser``).
"""

from morphogen.laser.architectures import (
    design_membrane,
    periodic_lines,
    pixelate,
    sheet_tests,
    spot_opening,
    track_membrane,
    treatment_measures,
)
from morphogen.laser.bending import BendingResult, three_point_bending
from morphogen.laser.cells import (
    CellEvaluation,
    CellGrid,
    CellOptimization,
    CellProblem,
    Phase,
    band_fractions,
    beso,
    contrast_phases,
    design_tracks,
    failure_measures,
    hca,
    topology,
    treatable,
    wang_phases,
)
from morphogen.laser.design import (
    LOADINGS,
    MEASURES,
    METHODS,
    OBJECTIVES,
    GaussianProcess,
    OptimizationResult,
    Parameter,
    SheetCase,
    SheetResult,
    hill48,
    optimize,
)
from morphogen.laser.heat import (
    Beam,
    ThermalProperties,
    TrackKernel,
    TransientHeat,
    moving_source,
    moving_source_grid,
    rosenthal_line,
    rosenthal_point,
    track_kernel,
)
from morphogen.laser.materials import (
    PRESETS,
    FlowLaw,
    SheetMaterial,
    StrengthTable,
    flow_parameters,
    sheet_material,
)
from morphogen.laser.mechanics import (
    Fracture,
    Membrane,
    Replay,
    TensileResult,
    curve_measures,
    hill_from_r,
    replay,
    tensile_test,
    yld2000_from_data,
    yld2000_profile,
)
from morphogen.laser.metallurgy import (
    Alloying,
    PeakHardening,
    PeakSoftening,
    Tempering,
    TransformationHardening,
    andrews_ac,
    andrews_ms,
    maynier_hardness,
    sieverts_nitrogen,
)
from morphogen.laser.patterns import (
    PATTERNS,
    LaserPath,
    Track,
    clip_path,
    distance_to_polyline,
    make_pattern,
)
from morphogen.laser.periodic import PeriodicCell, cell_test, lattice, reduce_basis
from morphogen.laser.process import LaserTreatment, TreatmentResult
from morphogen.laser.run import load_laser_config, run_laser
from morphogen.laser.solid import case_volume, flow_plasticity, sheet_volume, volume_tension
from morphogen.laser.specimen import QuadMesh, Specimen
from morphogen.laser.structures import CompactTension, LPlate, TensionCase, load_test

__all__ = [
    "LOADINGS",
    "MEASURES",
    "METHODS",
    "OBJECTIVES",
    "PATTERNS",
    "PRESETS",
    "Alloying",
    "Beam",
    "BendingResult",
    "CellEvaluation",
    "CellGrid",
    "CellOptimization",
    "CellProblem",
    "CompactTension",
    "FlowLaw",
    "Fracture",
    "GaussianProcess",
    "LPlate",
    "LaserPath",
    "LaserTreatment",
    "Membrane",
    "OptimizationResult",
    "Parameter",
    "PeakHardening",
    "PeakSoftening",
    "PeriodicCell",
    "Phase",
    "QuadMesh",
    "Replay",
    "SheetCase",
    "SheetMaterial",
    "SheetResult",
    "Specimen",
    "StrengthTable",
    "Tempering",
    "TensileResult",
    "TensionCase",
    "ThermalProperties",
    "Track",
    "TrackKernel",
    "TransformationHardening",
    "TransientHeat",
    "TreatmentResult",
    "andrews_ac",
    "andrews_ms",
    "band_fractions",
    "beso",
    "case_volume",
    "cell_test",
    "clip_path",
    "contrast_phases",
    "curve_measures",
    "design_membrane",
    "design_tracks",
    "distance_to_polyline",
    "failure_measures",
    "flow_parameters",
    "flow_plasticity",
    "hca",
    "hill48",
    "hill_from_r",
    "lattice",
    "load_laser_config",
    "load_test",
    "make_pattern",
    "maynier_hardness",
    "moving_source",
    "moving_source_grid",
    "optimize",
    "periodic_lines",
    "pixelate",
    "reduce_basis",
    "replay",
    "rosenthal_line",
    "rosenthal_point",
    "run_laser",
    "sheet_material",
    "sheet_tests",
    "sheet_volume",
    "sieverts_nitrogen",
    "spot_opening",
    "tensile_test",
    "three_point_bending",
    "topology",
    "track_kernel",
    "track_membrane",
    "treatable",
    "treatment_measures",
    "volume_tension",
    "wang_phases",
    "yld2000_from_data",
    "yld2000_profile",
]
