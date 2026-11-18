"""The pattern transformation of a square array of circular holes (specimen 1 of Bertoldi et
al. 2008, d/l = 8.67/9.97) by the post-bifurcation analysis along a nonlinear path, on a
conforming mesh of its level set, mirrored so that its symmetries are exact: the first
bifurcation in uniaxial strain (the infinite layer of Sperling et al. 2023: 2.25 %), its
irrep, which doubles the period, and the alternating ellipses of its pattern; or, with
the diagonal mirrors of the square on the mesh (the group 4/mmm, which equibiaxial
compression keeps), the first bifurcation in equibiaxial compression.

Usage: python examples/pattern_transformation.py [element size, a fraction of the cell]
[uniaxial | equibiaxial]
(0.065 by default: about 90 000 dofs and 3 minutes; 0.04: 181 000 dofs, 8 minutes, 2.99 %)
"""

import math
import sys
import time

import numpy as np

from morphogen.domain import Domain
from morphogen.homogenization import path_bifurcation
from morphogen.homogenization.materials import Material
from morphogen.mesh import ImplicitBody, mirrored_mesh

size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.065
loading = sys.argv[2] if len(sys.argv) > 2 else "uniaxial"
radius = 0.5 * 8.67 / 9.97  # the holes, in units of the cell


def solid(points):
    """Negative in the solid: the distance to the nearest hole less its radius."""
    d = points[:, :2] % 1.0 - 0.5
    return radius - np.hypot(d[:, 0], d[:, 1])


# a slab of two elements, mirrored along x, y and z (and the diagonal of x and y for the
# equibiaxial loading): a 2 x 2 cell with exact symmetries
body = ImplicitBody(Domain(size=(1.0, 1.0, 2 * size), periodic=True), solid)
diagonals = ((0, 1),) if loading == "equibiaxial" else ()
t0 = time.perf_counter()
mesh = mirrored_mesh(body, size=size, diagonals=diagonals, cells=(2, 2, 1), order=2,
                     phases="solid")  # fmt: skip
porosity = 1.0 - mesh.volumes().sum() / mesh.box_volume
elapsed = time.perf_counter() - t0
print(f"mesh: {mesh.n_elements} 10-node tetrahedra, {3 * mesh.n_nodes} dofs, porosity "
      f"{porosity:.4f} (exact {math.pi * radius**2:.4f}), {elapsed:.1f} s")  # fmt: skip

# a nearly incompressible Hencky solid (K / mu = 50, the Mooney-Rivlin fit of the
# reference); compression along y with the lateral strain held (uniaxial strain), or
# along x and y alike (equibiaxial)
material = {1: Material(young=1.0, poisson=0.49)}
lateral = -0.05 if loading == "equibiaxial" else 0.0
strain = {"11": lateral, "22": -0.05, "33": 0.0, "23": 0.0, "31": 0.0, "12": 0.0}
path = {"times": [0, 1], "increments": 25, "strain": strain}
t0 = time.perf_counter()
b = path_bifurcation(mesh, path, materials=material, cells=(2, 2, 1))
_, irrep, count = b.first
elapsed = time.perf_counter() - t0
reference = " (Sperling et al.: 2.25 %)" if loading == "uniaxial" else ""
print(f"group of {b.group.order} operations; first bifurcation at a nominal strain of "
      f"{-100 * b.critical_strain[1]:.2f} %{reference}, irrep {irrep} of dimension "
      f"{b.irreps[irrep].dim}, multiplicity {count} ({elapsed:.0f} s)")  # fmt: skip
pattern = min((p for p in b.patterns if p.irrep == irrep), key=lambda p: p.energy)
u = b.displacement(pattern.mode)
for hx in (0, 1):  # the ellipticity of each hole: its radial displacement along x less y
    for hy in (0, 1):
        d = b.mesh.nodes[:, :2] - np.array([hx + 0.5, hy + 0.5])
        rad = np.hypot(d[:, 0], d[:, 1])
        ring = np.abs(rad - radius) < 0.5 * size
        radial = np.einsum("ij,ij->i", u[:, :2], d) / np.maximum(rad, 1e-12)
        along_x = radial[ring & (np.abs(d[:, 0]) > 0.9 * rad)].mean()
        along_y = radial[ring & (np.abs(d[:, 1]) > 0.9 * rad)].mean()
        print(f"hole ({hx}, {hy}): ellipticity {along_x - along_y:+.3e}")
