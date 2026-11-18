"""Fracture: cohesive zone elements along interfaces and crack paths, in small and finite
strains (:mod:`morphogen.fracture.cohesive`), the interfaces of the phases of periodic
cells that debond along the loading paths of the homogenization
(:mod:`morphogen.fracture.periodic`), the linear elastic fracture mechanics of 3D
cracks, their energy release rate and propagation on conforming (remeshed) meshes or by
XFEM (:mod:`morphogen.fracture.lefm`), and mixed-mode cracks (I, II, III) in small and
finite strains: the G-θ method, the interaction integrals, kinked cracks and their growth
(:mod:`morphogen.fracture.mixed`), with the behaviours of morphoGen's points as their bulk
(elasto-viscoplasticity, porous and crystal plasticity, MFront; :mod:`morphogen.fracture.bulk`),
and their ductile growth on remeshed meshes, the state of the points transferred from mesh to
mesh, grain by grain in polycrystals (:mod:`morphogen.fracture.ductile`); cracks through a
morphology, its phases and pores on the crack meshes (:func:`crack_mesh` with ``morphology``),
the sides of the crack apart, which feel the toughness of each phase and of the interfaces
between them: deflected along weak interfaces, held back by tough phases
(:mod:`morphogen.fracture.toughness`); cracks of any surface and front, each point of
which advances and kinks on its own (:mod:`morphogen.fracture.surface`); and the
brittle fracture of voxel images by the
phase-field method, whose cracks start, branch and merge without being tracked
(:mod:`morphogen.fracture.phasefield`; in finite strains, the rubbers and soft solids of
:mod:`morphogen.fracture.phasefield_finite`)."""

from morphogen.fracture.bulk import PointBulk
from morphogen.fracture.cohesive import (
    Bilinear,
    CohesiveProblem,
    CohesiveResult,
    Contact,
    Exponential,
    Interfaces,
    cohesive_zone_length,
    dcb_beam_theory,
    dcb_specimen,
    insert_interfaces,
    traction,
    turon_strength,
)
from morphogen.fracture.ductile import (
    DuctileGrowth,
    dissipation,
    propagate_ductile,
    transfer_displacement,
    transfer_state,
)
from morphogen.fracture.lefm import (
    XFEM,
    Crack,
    Propagation,
    conforming_work_of,
    crack_faces,
    crack_mesh,
    cut_mesh,
    face_forces,
    interface_nodes,
    open_crack,
    phase_mesh,
    phases_at,
    propagate,
    quarter_points,
    refine_front,
    release_rate,
    sent_factor,
    snap_front,
    with_grains,
    with_phases,
    without_voids,
    xfem_work,
)
from morphogen.fracture.mixed import (
    CrackSolution,
    KinkedCrack,
    KinkedGrowth,
    energy_release_rate,
    inside_elements,
    kink_angle,
    material_groups,
    material_nodes,
    propagate_kinked,
    solve_crack,
    through_voids,
)
from morphogen.fracture.periodic import CohesiveCell, interface_law, open_cell
from morphogen.fracture.phasefield import (
    PhaseField,
    PhaseFieldProblem,
    PhaseFieldResult,
    phase_field_fracture,
)
from morphogen.fracture.phasefield_finite import FinitePhaseFieldProblem
from morphogen.fracture.surface import FrontGrowth, SurfaceCrack, propagate_front
from morphogen.fracture.toughness import (
    Advance,
    Toughness,
    advance_candidates,
    choose_advance,
    interface_paths,
    kinked_factors,
    kinked_release_rate,
)

__all__ = [
    "XFEM",
    "Advance",
    "Bilinear",
    "CohesiveCell",
    "CohesiveProblem",
    "CohesiveResult",
    "Contact",
    "Crack",
    "CrackSolution",
    "DuctileGrowth",
    "Exponential",
    "FinitePhaseFieldProblem",
    "FrontGrowth",
    "Interfaces",
    "KinkedCrack",
    "KinkedGrowth",
    "PhaseField",
    "PhaseFieldProblem",
    "PhaseFieldResult",
    "PointBulk",
    "Propagation",
    "SurfaceCrack",
    "Toughness",
    "advance_candidates",
    "choose_advance",
    "cohesive_zone_length",
    "conforming_work_of",
    "crack_faces",
    "crack_mesh",
    "cut_mesh",
    "dcb_beam_theory",
    "dcb_specimen",
    "dissipation",
    "energy_release_rate",
    "face_forces",
    "insert_interfaces",
    "inside_elements",
    "interface_law",
    "interface_nodes",
    "interface_paths",
    "kink_angle",
    "kinked_factors",
    "kinked_release_rate",
    "material_groups",
    "material_nodes",
    "open_cell",
    "open_crack",
    "phase_field_fracture",
    "phase_mesh",
    "phases_at",
    "propagate",
    "propagate_ductile",
    "propagate_front",
    "propagate_kinked",
    "quarter_points",
    "refine_front",
    "release_rate",
    "sent_factor",
    "snap_front",
    "solve_crack",
    "through_voids",
    "traction",
    "transfer_displacement",
    "transfer_state",
    "turon_strength",
    "with_grains",
    "with_phases",
    "without_voids",
    "xfem_work",
]
