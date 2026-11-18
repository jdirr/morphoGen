"""Exact meshes, without voxels nor a background grid (docs/theory.md §94): a morphology of
each route meshed from its exact geometry and written as VTK (.vtu), with what makes the
mesh exact.

1. Spheres of a periodic Boolean model: the adaptive mesh of their exact level set; the
   nodes of the interface on the spheres, the opposite faces matching.
2. A sheet gyroid: a sheet a tenth of a cell thick, resolved where the grid of the same
   size would lose it.
3. Dead leaves of three phases: their level sets cut in turn.
4. Scale-rich plates: the exact convex pieces, whose volumes are those of the plates.
5. A Menger sponge, periodic: its cubes merged into boxes, the faces matching.

Usage: python examples/exact_meshes.py [output directory]   (about three minutes)
"""

import sys
import time
from pathlib import Path

import numpy as np

import morphogen as mg
from morphogen import io
from morphogen.mesh import exact_mesh, exact_route

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/exact_meshes")
out.mkdir(parents=True, exist_ok=True)
periodic = mg.Domain.cube(1.0, periodic=True)
box = mg.Domain.cube(1.0, periodic=False)
cases = {
    "spheres": (
        mg.models.BooleanSpheres(radius=0.15, volume_fraction=0.2).generate(periodic, 1),
        dict(size=0.2, hausdorff=0.01),
    ),
    "sheet_gyroid": (
        mg.models.TPMS(
            surface="gyroid", variant="sheet", cell_size=1.0, volume_fraction=0.1
        ).generate(periodic),
        dict(size=0.2, hausdorff=0.01),
    ),
    "dead_leaves": (
        mg.models.DeadLeavesModel(radius=0.25, phase_fractions=[0.3, 0.3, 0.4]).generate(box, 2),
        dict(size=0.25, hausdorff=0.02),
    ),
    "plates": (
        mg.models.ScaleRichNetwork(
            dimension=3, n_ligaments=60, decay=0.3, volume_fraction=0.3
        ).generate(box, 3),
        dict(size=0.2),
    ),
    "menger": (
        mg.models.FractalSponge(pattern="menger", level=2, cell_size=1.0).generate(periodic),
        dict(size=0.2),
    ),
}
for name, (micro, options) in cases.items():
    t0 = time.perf_counter()
    mesh = exact_mesh(micro, **options)
    seconds = time.perf_counter() - t0
    path = io.save_mesh(mesh, out / f"{name}.vtu")
    volumes = mesh.volumes()
    fractions = {k: volumes[mesh.phase == k].sum() / volumes.sum() for k in np.unique(mesh.phase)}
    shares = ", ".join(f"{k}: {v:.4f}" for k, v in fractions.items())
    line = f"{name:12s} ({exact_route(micro)}): {mesh.n_elements} elements in {seconds:.1f} s"
    line += f", phases {shares}"
    if mesh.periodic:
        line += f", faces matching: {mesh.matching_faces()}"
    if exact_route(micro) == "level_set":
        sets = micro.phase_level_sets_at(mesh.nodes, options["size"])
        between = np.zeros(len(mesh.nodes), dtype=int)
        for k in np.unique(mesh.phase):
            between[np.unique(mesh.elements[mesh.phase == k])] += 1
        nodes = between == 2
        error = np.min([np.abs(v[nodes]) for v in sets.values()], axis=0)
        line += f", interface nodes off the exact surface: median {np.median(error):.1e}"
        line += f", 99 % below {np.quantile(error, 0.99):.1e}"
    print(line)
    print(f"  {path}")
