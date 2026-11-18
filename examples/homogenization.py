"""Apparent properties of virtual microstructures: finite elements on conforming meshes,
and the FFT solver on voxel images.

Usage: python examples/homogenization.py   (needs MMG: pip install pymmg)
"""

import time

import numpy as np

import morphogen as mg
from morphogen.homogenization import VOID, Material, bounds, homogenize


def report(results):
    for key, res in results.items():
        if isinstance(res, dict):  # tensors
            print(f"  {key}:")
            print(np.array2string(res["tensor"], precision=1, suppress_small=True))
            continue
        info = res.solution.info
        print(
            f"  {key:24s} {res.value:12.5g}  ({info.method}, {info.n_dofs} unknowns, "
            f"{info.iterations} iterations, {info.solve_s:.1f} s)"
        )


# 1. Stiff spheres in a soft matrix: conforming mesh of both phases, four conditions
model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
micro = model.generate(mg.Domain.cube(6.0, periodic=True), seed=1)
t = time.perf_counter()
mesh = mg.mesh.conforming_mesh(micro, size=0.5)
print(
    f"spheres: {mesh.n_elements} tetrahedra ({time.perf_counter() - t:.1f} s), fractions "
    f"{mesh.phase_fractions()}"
)
materials = {
    0: Material(young=1.0, poisson=0.3, conductivity=1.0),
    1: Material(young=20.0, poisson=0.3, conductivity=10.0),
}
results = homogenize(
    mesh,
    materials,
    bcs=["kubc", "mixed", "periodic", "subc"],
    properties=["bulk_modulus", "shear_modulus", "conductivity"],
)
report(results)  # SUBC <= PBC <= KUBC, MBC between SUBC and KUBC
b = bounds(mesh.phase_fractions(), materials)["bulk_modulus"]
print(f"  bulk modulus bounds: Hashin-Shtrikman [{b['hs_lower']:.4g}, {b['hs_upper']:.4g}]")

# 2. Porous sheet gyroid (titanium), 10-node tetrahedra of the solid, periodic cell
gyroid = mg.models.TPMS(surface="gyroid", variant="sheet", cell_size=1.0, volume_fraction=0.2)
cell = gyroid.generate(mg.Domain.cube(1.0, periodic=True))
t = time.perf_counter()
# the sheet is about 0.065 thick: sample the level set finely (background spacing)
mesh = mg.mesh.conforming_mesh(cell, size=0.08, background=0.025, phases="solid", order=2)
print(
    f"gyroid: {mesh.n_elements} tetrahedra ({time.perf_counter() - t:.1f} s), solid fraction "
    f"{mesh.phase_fractions()[1]:.4f}"
)
titanium = {0: VOID, 1: Material(young=110e3, poisson=0.34, conductivity=7.0)}
report(homogenize(mesh, titanium, properties=["stiffness", "conductivity"]))

# 3. Voronoi mosaic: no level set, so the FFT solver on voxels (periodic)
mosaic = mg.models.Voronoi(n_cells=200, phase_fractions=(0.5, 0.5))
image = mosaic.generate(mg.Domain.cube(1.0, periodic=True), seed=3).voxelize(shape=64)
print(f"Voronoi mosaic: {image.shape} voxels")
report(homogenize(image, materials, properties=["bulk_modulus", "shear_modulus"]))
