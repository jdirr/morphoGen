"""RVE study of a two-phase Voronoi mosaic with the Python API (ensemble method).

The integral range of the volume fraction of a Poisson-Voronoi mosaic is
A3 = 1.179 / theta (Kanit et al. 2003): the fit with gamma = 1 recovers it.
"""

import numpy as np

import morphogen as mg
from morphogen import rve
from morphogen.seeding import derive_seed

model = mg.models.Voronoi(intensity=1.0, phase_fractions=(0.5, 0.5))
sizes = [4.0, 6.0, 8.0, 12.0]
n_realizations = [64, 32, 16, 8]

volumes, fractions = [], []
for s, (size, n) in enumerate(zip(sizes, n_realizations, strict=True)):
    for r in range(n):
        micro = model.generate(mg.Domain.cube(size), seed=derive_seed(2003, s, r))
        image = micro.voxelize(voxel_size=0.2)
        volumes.append(size**3)
        fractions.append(mg.analysis.volume_fraction(image))

stats = rve.ensemble_statistics(volumes, fractions)
for row in stats.rows():
    print(row)

law = rve.fit_statistics(stats, gamma=1.0)
mean = float(np.mean(fractions))
print(f"A3 = {law.integral_range(rve.point_variance(mean)):.3f} (theory 1.179)")
for row in rve.rve_table(law, mean, eps_rel=(0.01, 0.05), n=(1, 10)):
    print(row)
