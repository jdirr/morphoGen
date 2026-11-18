"""Micromorphic ductile damage with and without the stress triaxiality: a notched plate in
plane strain, pulled. Driven by the plastic strain alone, the damage spreads over the
ligament, most at the root of the notch and on the far side that the grips stretch, and the
plate keeps two thirds of its peak force; weighted by the triaxiality (Rice & Tracey's
growth of voids), it concentrates ahead of the notch, where plane strain makes the stress
triaxial, and a band of full damage runs across the ligament (mode I), as wide as the
length ℓ = 2h makes it.

Usage: python examples/ductile_triaxiality.py [output folder] [element size]
"""

import sys
import time
from pathlib import Path

import numpy as np

from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.image import VoxelImage
from morphogen.io.tables import write_table
from morphogen.micromorphic import DuctileProblem, MicromorphicDuctile

out = Path(sys.argv[1] if len(sys.argv) > 1 else "ductile_output")
out.mkdir(parents=True, exist_ok=True)
h = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0625
PULL = 0.016  # of the top; the crack across most of the ligament with the triaxiality


def notched_plate(h):
    """The plate [0, 1]² of one layer of voxels, a notch of depth 0.3 and width 0.1 at
    mid-height."""
    n = round(1.0 / h)
    centre = (np.arange(n) + 0.5) * h
    data = np.zeros((n, n, 1), np.uint8)
    data[np.ix_(centre < 0.3, np.abs(centre - 0.5) < 0.05, [0])] = 1  # the notch, removed
    return voxel_tet_mesh(VoxelImage(data, spacing=(h, h, h), periodic=False), phases=[0])


def plane_strain_pull(problem, pull):
    """The bottom held vertically (a node pinned), the top pulled, no displacement along
    the thickness (plane strain)."""
    x = problem.mesh.nodes
    bottom = np.flatnonzero(np.abs(x[:, 1]) < 1e-9)
    top = np.flatnonzero(np.abs(x[:, 1] - 1.0) < 1e-9)
    pin = bottom[np.argmin(np.linalg.norm(x[bottom], axis=1))]
    return [(bottom, 1, 0.0), (pin, 0, 0.0), (np.arange(len(x)), 2, 0.0), (top, 1, pull)]


rows, fields = [], {}
for law in ("none", "rice_tracey"):
    t0 = time.time()
    material = MicromorphicDuctile(young=200.0, poisson=0.3, yield_stress=0.4, hardening=10.0,
                                   length=2 * h, onset=0.002, critical=0.02,
                                   triaxiality=law)  # fmt: skip
    problem = DuctileProblem(notched_plate(h), material)
    result = problem.run(plane_strain_pull(problem, PULL), steps=32)
    force = np.array([r[3][0] for r in result.reactions]) / h  # per unit thickness
    damage = problem.damage(result.state)
    centre = problem.mesh.nodes[problem.mesh.elements[:, :4]].mean(axis=1)
    cracked = centre[(damage > 0.9) & (centre[:, 0] > 0.3)]
    offset = np.abs(cracked[:, 1] - 0.5).max() if len(cracked) else float("nan")
    reach = cracked[:, 0].max() if len(cracked) else 0.3
    sections = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # the path: the most damaged of each section
    path = [np.abs(centre[c, 1] - 0.5)[np.argmax(damage[c])]
            for c in (np.abs(centre[:, 0] - x) < h / 2 for x in sections)]  # fmt: skip
    elapsed = time.time() - t0
    print(f"{law}: peak force {force.max():.4f}, final {force[-1]:.4f}; crack (d > 0.9) "
          f"to x = {reach:.2f}, at most {offset:.2f} from mid-height, {elapsed:.0f} s")  # fmt: skip
    print("   the most damaged element of the sections x = 0.4 ... 0.9 at |y - 0.5| =",
          np.round(path, 3), f"(d up to {damage.max():.2f})")  # fmt: skip
    rows += [{"law": law, "pull": float(u), "force": float(f)}
             for u, f in zip(PULL * result.loads, force, strict=True)]  # fmt: skip
    fields[law] = (centre, damage)
write_table(out / "ductile_triaxiality.csv", rows)
try:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 4), layout="constrained")
    for ax, (law, (centre, damage)) in zip(fig.subplots(1, 2), fields.items(), strict=True):
        ax.scatter(centre[:, 0], centre[:, 1], c=damage, s=6, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"damage, triaxiality: {law}")
        ax.set_aspect("equal")
    fig.savefig(out / "ductile_triaxiality.png", dpi=150)
except ImportError:
    pass
