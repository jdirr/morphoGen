"""Gurson's law of a porous morphology: q1 and q2 identified on its responses to two stress
triaxialities, then written back as a porous_plastic material file of Z-mat for Z-set.

Hard-core spherical pores (10 % of the volume) in a von Mises matrix (the FFT solver of
voxel images, periodic), loaded in tension (triaxiality 1/3) and in a tension whose
lateral strains are held (its triaxiality grows as the matrix flows): Gurson's model of the
matrix, its porosity the fraction of the voids, needs q1 and q2 to follow both, as on the
unit cells of Tvergaard (1981) and Faleskog, Gao and Shih (1998). A minute or two on a
laptop (Numba compiles the kernels the first time).
"""

import numpy as np

from morphogen.domain import Domain
from morphogen.homogenization import VOID, Material, NonlinearFFTProblem
from morphogen.homogenization.identification import macroscopic_law
from morphogen.homogenization.zmat import write_zmat
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed

model = model_from_config({"type": "hardcore_spheres", "radius": 0.8, "volume_fraction": 0.1})
image = model.generate(Domain(size=4.0, periodic=True), derive_seed(3, 0)).voxelize(shape=12)
porosity = float(np.mean(image.data == 1))
matrix = {"young": 1000.0, "poisson": 0.3, "plasticity": {"yield": 1.0, "hardening": 10.0}}
materials = {0: Material.from_config(matrix), 1: VOID}


def respond(path):
    """The macroscopic response of the porous voxels along a loading path."""
    return NonlinearFFTProblem(image, materials).run(path)


law = macroscopic_law(respond, law="gurson", loadings=["11", "confined11"], matrix=matrix,
                      porosity=porosity, strain=0.01, increments=10)  # fmt: skip
fit = law.identification
print(f"porosity {porosity:.3f}")
print({name: round(value, 3) for name, value in fit.values.items()})
gaps, initial = [round(g, 3) for g in fit.gaps], [round(g, 3) for g in fit.initial_gaps]
print("relative gaps:", gaps, "from Tvergaard's q1 = 1.5, q2 = 1:", initial)
print(write_zmat(fit.material, digits=4, title="Gurson's law of pores (morphoGen)"))
