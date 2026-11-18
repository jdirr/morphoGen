"""Topology optimization of periodic cells by inverse homogenization.

1. The planar cell of maximal bulk modulus at 40 % solid (method of moving asymptotes),
   against the Hashin-Shtrikman upper bound.
2. The same problem with the hybrid cellular automaton (Tovar et al. 2006): every voxel
   moves its density to bring the mean stimulus of its neighbourhood to a set point.
3. An auxetic cell: Poisson's ratio minimized, the stiffness kept above 5 % of the solid's.
4. The design as a microstructure: phase image, smooth level set for conforming meshes,
   exports, and an independent check of its stiffness with the FFT solver.

Planar cells have one voxel through the thickness: prismatic microstructures, whose
in-plane properties are those of plane strain. Each design is also re-evaluated on a
grid twice as fine: voxels touching only at a corner ("hinges") are stiff for voxel
elements, and moduli that drop on refinement would reveal them. Run time: about a
minute.
"""

import numpy as np

import morphogen as mg
from morphogen.homogenization import VOID, Material, homogenize
from morphogen.optimization import CellOptimization, tensor_properties

solid = Material(young=1.0, poisson=0.3, conductivity=1.0)
rng = np.random.default_rng(1994)


def hashin_shtrikman_bulk(f, material):
    """Upper bound of the plane-strain bulk modulus of a solid with voids."""
    lam, mu = material.lame
    k = lam + mu
    return k * mu * f / ((1 - f) * k + mu)


def report(name, design, bound=None):
    print(f"{name}: {design.iterations} iterations, {design.time_s:.0f} s")
    p, fine = design.properties(), design.refined_properties(2)
    if bound is not None:
        b = bound(design.binary_fraction, solid)
        print(
            f"  solid fraction {design.binary_fraction:.3f}, k = {p['bulk_modulus']:.4f}"
            f" = {100 * p['bulk_modulus'] / b:.1f} % of the Hashin-Shtrikman bound"
            f" ({100 * fine['bulk_modulus'] / b:.1f} % on the finer grid)"
        )
    else:
        print(
            f"  Poisson's ratio {p['poisson_12']:.3f} ({fine['poisson_12']:.3f} on the finer"
            f" grid), Young's modulus {p['young_1']:.4f}"
        )
    print(f"  hinges: {design.hinges}")


# 1. Stiffest cell in bulk: square symmetry, filter radius 1.5 voxels, projection
#    sharpened up to 32 (black-and-white design at the requested volume)
bulk = CellOptimization(
    (48, 48, 1),
    0.4,
    objective="bulk_modulus",
    material=solid,
    method="mma",
    projection=32.0,
    iterations=300,
).run(rng)
report("bulk modulus (MMA)", bulk, hashin_shtrikman_bulk)

# 2. Hybrid cellular automaton: local rules, no global gradient step
hca = CellOptimization(
    (48, 48, 1), 0.4, objective="bulk_modulus", material=solid, method="hca", iterations=200
).run(rng)
report("bulk modulus (HCA)", hca, hashin_shtrikman_bulk)

# 3. Auxetic cell: minimize C12 / C11 with C11 above 0.05 E (re-entrant cell); a softer
#    projection (8) keeps the members connected through faces
auxetic = CellOptimization(
    (48, 48, 1),
    0.35,
    objective="poisson_ratio",
    material=solid,
    method="mma",
    filter_radius=2.0,
    projection=8.0,
    iterations=300,
).run(rng)
report("auxetic cell (MMA)", auxetic)

# 4. The design as a microstructure (periodic cell of edge 48 voxels): exact phases on
#    its lattice, a smooth level set from the densities elsewhere
micro = auxetic.to_microstructure()
image = micro.voxelize(shape=(96, 96, 1))  # finer image, interfaces from the level set
mg.io.save_microstructure(micro, "auxetic.json")  # morphogen view auxetic.json --tile 3
mg.io.save_image(image, "auxetic.vti")

# independent check: FFT homogenization of the finer image, voids without stiffness
check = homogenize(image, {0: VOID, 1: solid}, properties=["stiffness"])
c = check["stiffness.periodic"]["tensor"]
nu = tensor_properties(c, planar=True)["poisson_12"]
print(f"  homogenization of the 96 x 96 image: Poisson's ratio {nu:.3f}")

try:
    from morphogen.viz import figures

    figures.tiled_figure(bulk.to_microstructure(), tiles=3).savefig("bulk_cell.png")
    figures.tiled_figure(micro, tiles=3).savefig("auxetic_cell.png")
    figures.history_figure(auxetic.history).savefig("auxetic_history.png")
    print("figures: bulk_cell.png, auxetic_cell.png, auxetic_history.png")
except ImportError as exc:
    print(f"no figures: {exc}")
