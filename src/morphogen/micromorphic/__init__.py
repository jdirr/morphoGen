"""Micromorphic continua (Eringen, Mindlin; Forest's micromorphic approach) on
tetrahedral meshes, in small and finite strains (:mod:`morphogen.micromorphic.finite`):
the mixed finite elements of a displacement and generalized degrees of freedom
(:mod:`morphogen.micromorphic.space`), the microfoam model of metallic foams
(:mod:`morphogen.micromorphic.microfoam`), a micromorphic damage model to crack with
(:mod:`morphogen.micromorphic.damage`) and a micromorphic ductile damage of
elasto-viscoplastic solids (:mod:`morphogen.micromorphic.ductile`)."""

from morphogen.micromorphic.damage import DamageProblem, MicromorphicDamage
from morphogen.micromorphic.ductile import DuctileProblem, MicromorphicDuctile
from morphogen.micromorphic.microfoam import (
    CHI,
    Microfoam,
    MicrofoamProblem,
    characteristic_length,
)
from morphogen.micromorphic.space import MixedProblem, MixedSpace, PathResult

__all__ = [
    "CHI",
    "DamageProblem",
    "DuctileProblem",
    "Microfoam",
    "MicrofoamProblem",
    "MicromorphicDamage",
    "MicromorphicDuctile",
    "MixedProblem",
    "MixedSpace",
    "PathResult",
    "characteristic_length",
]
