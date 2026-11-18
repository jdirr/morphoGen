"""The macroscopic law of an anisotropic morphology: Hill's criterion identified on the
responses of a laminate, then written back as a material file of Z-mat for Z-set.

Two von Mises phases of yield stresses 1 and 2 in layers normal to z (the FFT solver of
voxel images, periodic): in tension along x or z the laminate yields alike (the soft
layer, held laterally by the hard one, carries more than its yield stress), but in shear
it is weaker across its layers (the shear 31 loads them in series) than in their plane
(the shear 12 loads them in parallel). A von Mises law cannot follow both shears; Hill's
criterion, its coefficients fitted from von Mises's (all 1), takes h31 about 2: the
yield stress in the shear 31 is R0 / sqrt(3 h31). A few seconds on a laptop (a minute
the first time, Numba compiling the kernels).
"""

import numpy as np

from morphogen.homogenization import Material, NonlinearFFTProblem
from morphogen.homogenization.identification import Reference, identify_references
from morphogen.homogenization.zmat import write_zmat
from morphogen.image import VoxelImage

phases = np.zeros((2, 2, 8), dtype=np.uint8)
phases[:, :, 4:] = 1  # two layers of equal thickness, normal to z
image = VoxelImage(phases, periodic=True)
elastic = {"young": 1000.0, "poisson": 0.3}
materials = {
    0: Material.from_config({**elastic, "plasticity": {"yield": 1.0, "hardening": 10.0}}),
    1: Material.from_config({**elastic, "plasticity": {"yield": 2.0, "hardening": 10.0}}),
}

# the responses of the laminate: tensions along x and z, shears in its plane and across it
references = []
for component, column in (("11", 0), ("33", 2), ("12", 5), ("31", 4)):
    strain = 0.006 if component[0] == component[1] else 0.01
    path = {"times": [0, 1], "increments": 20, "strain": {component: [0, strain]}}
    response = NonlinearFFTProblem(image, materials).run(path)
    references.append(Reference(path, response.times, response.stress))
    print(f"stress {component} at the end: {response.stress[-1, column]:.3f}")

# Hill's law of the laminate, from von Mises's; x and y alike (h22 = h11 = 1, which fixes
# the scale of the coefficients against R0), and the shear 23 like the shear 31
start = {**elastic, "plasticity": {"yield": 1.5, "hardening": 10.0, "hill": {}}}
fit = identify_references(start, references, ["yield", "hardening", "hill33", "hill12", "hill31"])
fit.material["plasticity"]["hill"]["23"] = fit.values["hill31"]
print({name: round(value, 3) for name, value in fit.values.items()})
gaps = [round(g, 3) for g in fit.gaps]
print("relative gaps:", gaps, "from", [round(g, 3) for g in fit.initial_gaps])
print(write_zmat(fit.material, digits=4, title="Hill's law of a laminate (morphoGen)"))
