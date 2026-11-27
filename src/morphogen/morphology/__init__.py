"""Mathematical morphology of the realizations on their implicit geometry.

The realizations of the models are random closed sets described exactly, at any point, by
the implicit functions of their phases (:mod:`morphogen.models.implicit`); voxel images
and meshes are cut from these functions afterwards. This package measures the
realizations on that geometry itself, so that the measures compare with the theory of
random closed sets without the bias of a discretization:

* :mod:`morphogen.morphology.balls`: the Minkowski functionals of unions of balls (volume,
  surface, integral of mean curvature, Euler characteristic) and of their plane sections,
  unions of discs (area, perimeter, Euler characteristic), exactly, from the power diagram
  and the dual complex of the balls;
* :mod:`morphogen.morphology.measures`: probes of any structure — points, pairs, segments,
  balls, lines — answered exactly by the functions of its phases (the fraction, the
  covariance, the Choquet capacities, the contact distributions, the chords, the specific
  surface by Crofton's formula), with the standard deviations of the estimators, and the
  measures of a realization beside the closed forms of its model;
* :mod:`morphogen.morphology.identification`: models fitted to images and volumes, from
  their measures and the closed forms of the models (or their simulations), ranked by how
  well they reproduce what was not fitted;
* :mod:`morphogen.morphology.segmentation`: grey-level images and stacks turned into phases
  and particles (thresholds, the separation of touching particles), and the statistics of
  the particles that models of their own take.
"""

from morphogen.morphology.balls import (
    BallUnion,
    DiscUnion,
    minkowski_functionals,
    section_discs,
    section_functionals,
    union_of_balls,
    union_of_discs,
)
from morphogen.morphology.measures import (
    Chords,
    Comparison,
    Curve,
    Estimate,
    capacity,
    chords,
    compare_with_theory,
    contact_distribution,
    covariance,
    crossings,
    dilation,
    first_hits,
    isotropic_directions,
    phase_function,
    phases_at,
    sample_points,
    specific_surface,
    volume_fraction,
)

__all__ = [
    "BallUnion",
    "Chords",
    "Comparison",
    "Curve",
    "DiscUnion",
    "Estimate",
    "capacity",
    "chords",
    "compare_with_theory",
    "contact_distribution",
    "covariance",
    "crossings",
    "dilation",
    "first_hits",
    "isotropic_directions",
    "minkowski_functionals",
    "phase_function",
    "phases_at",
    "sample_points",
    "section_discs",
    "section_functionals",
    "specific_surface",
    "union_of_balls",
    "union_of_discs",
    "volume_fraction",
]
