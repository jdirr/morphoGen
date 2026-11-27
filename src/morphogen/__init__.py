"""morphoGen: morphological modelling and generation of virtual microstructures.

Typical use::

    import morphogen as mg

    model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
    micro = model.generate(mg.Domain.cube(20.0), seed=1)   # explicit ("vector") geometry
    image = micro.voxelize(shape=128)                        # phase image (uint8)
    print(mg.analysis.volume_fraction(image), model.theory()["volume_fraction"])
    mg.io.save_image(image, "spheres.vti")

    mesh = mg.mesh.conforming_mesh(micro, size=0.5)         # conforming tetrahedra (MMG)
    materials = {0: mg.homogenization.Material(young=1.0, poisson=0.3),
                 1: mg.homogenization.Material(young=10.0, poisson=0.3)}
    results = mg.homogenization.homogenize(mesh, materials, bcs=["kubc", "periodic"])
"""

from morphogen import (
    analysis,
    automata,
    design,
    distributions,
    homogenization,
    io,
    mesh,
    models,
    morphology,
    orientations,
    rve,
    texture,
)
from morphogen._version import __version__
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.io.export import export
from morphogen.models import get_model, list_models, model_from_config
from morphogen.settings import set_num_threads

__all__ = [
    "Domain",
    "Grid",
    "VoxelImage",
    "__version__",
    "analysis",
    "automata",
    "design",
    "distributions",
    "export",
    "get_model",
    "homogenization",
    "io",
    "list_models",
    "mesh",
    "model_from_config",
    "models",
    "morphology",
    "orientations",
    "rve",
    "set_num_threads",
    "texture",
]
