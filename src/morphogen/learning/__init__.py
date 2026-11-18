"""Behaviours learned from data: networks trained on the responses of morphoGen's
behaviours (or of a microstructure) along random loading paths, then integrated in place
of a behaviour.

* Data (:mod:`.data`): random strain paths, the responses of a material point
  (:func:`~morphogen.homogenization.nonlinear.simulate_point`) or of the path solvers of a
  microstructure along them, splits and normalization.
* Models (:mod:`.networks`), their gradients written out: a perceptron of the stress
  (:class:`FeedForward`), the stress derived from a learned convex potential
  (:class:`Potential`: hyperelastic, a symmetric positive tangent) and a recurrent network
  over the strain increments with learned internal variables (:class:`Recurrent`:
  path-dependent responses, plasticity) and a learned generalized standard material
  (:class:`Standard`, :mod:`.standard`: a convex free energy and a convex dissipation
  potential of learned internal variables, the state of a step the minimizer of the
  incremental potential: a nonnegative dissipation and a symmetric tangent by
  construction). In finite strains every model is of the logarithmic strain ``1/2 ln C``
  (by default; its stress ``T`` conjugate to it, ``S = J^T T``, :mod:`.logarithmic`) or of
  the Green-Lagrange strain, objective.
* Training (:mod:`.training`): Adam on mini-batches, early stopping on a validation
  set, the paths turned by the symmetries of the behaviour (uniform rotations, those of
  the cube), seeded; the files of the models (``.npz`` with JSON metadata).
* Learned laws (:mod:`.behaviour`): ``learned`` in the configuration of a material
  integrates its points by a model, at a material point, by finite elements and on voxels,
  in the axes of its material (turned by the orientation of each grain, as crystals), the
  stress-free thermal strain of its expansion taken out.
* :func:`learn`: data, training, evaluation and file of a configuration (``morphogen
  learn``).
* MFront sources (:mod:`.mfront`): a learned law as an MFront behaviour (its trained
  parameters as constants, morphoGen's C++ port of the model, ``learned.cpp``), for
  AMITEX_FFTP, Cast3M, Abaqus and Code_Aster (:func:`morphogen.io.mfront.mfront_source`,
  ``morphogen mfront``, the AMITEX and Abaqus exports).
"""

from morphogen.learning.behaviour import (
    LearnedBehaviour,
    LearnedLaw,
    LearnedPoints,
    isotropic_constants,
    learned_law,
    learned_material,
)
from morphogen.learning.data import Dataset, Scales, generate, random_paths, responder
from morphogen.learning.networks import (
    KINDS,
    FeedForward,
    Model,
    Potential,
    Recurrent,
    load_model,
    make_model,
)
from morphogen.learning.run import learn, path_errors
from morphogen.learning.standard import Standard
from morphogen.learning.training import Training, errors, train

__all__ = [
    "KINDS",
    "Dataset",
    "FeedForward",
    "LearnedBehaviour",
    "LearnedLaw",
    "LearnedPoints",
    "Model",
    "Potential",
    "Recurrent",
    "Scales",
    "Standard",
    "Training",
    "errors",
    "generate",
    "isotropic_constants",
    "learn",
    "learned_law",
    "learned_material",
    "load_model",
    "make_model",
    "path_errors",
    "random_paths",
    "responder",
    "train",
]
