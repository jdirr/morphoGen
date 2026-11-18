"""Scale-rich networks in three dimensions (Both, Chen, ..., Portela & Barabási, Nature
659, 345, 2026, Fig. 6a and Supplementary S2.5-S2.7): plates that split convex voids into
closed cells, and beams grown until they meet (open cells, struts).

1. The tails of 10,000 plates (alpha = 0.5): thicknesses of exponent 1 + 1/alpha exactly,
   areas and degrees of exponent 5/2 (the 3 of the plane becomes (2d - 1) / (d - 1)).
2. The relative density of 500 plates (lambda_0 = 0.03) and of 500 beams (lambda_0 = 0.06,
   the cylinders on their axes) against alpha: the mean of realizations against the mean
   fields of the theory, and the initial thickness at which the plates jam.
3. Plates and beams at 30 % and 15 % relative density, alpha = 0 and 0.9: their sections
   by the mid-plane of the cube (exact, 400 x 400 points).

Usage: python examples/scale_rich_networks_3d.py [output directory]   (about 20 seconds)
"""

import sys
from pathlib import Path

import numpy as np

import morphogen as mg
from morphogen.models import ScaleRichNetwork

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/scale_rich_3d")
out.mkdir(parents=True, exist_ok=True)
cube = mg.Domain(size=(1.0, 1.0, 1.0), periodic=False)

# 1. the laws of the plates
big = ScaleRichNetwork(dimension=3, n_ligaments=10000, decay=0.5, initial_thickness=0.01)
plates = big.generate(cube, seed=1)
tails = plates.statistics(min_tail=100)
print("tails of 10,000 plates (alpha 0.5):")
for name, expected in (("thickness", 3.0), ("area", 2.5), ("degree", 2.5)):
    fit = tails[name]
    print(f"  {name:9s} {fit.exponent:.3f} +- {fit.error:.3f}  (above {fit.x_min:.3g}; {expected})")
summary = plates.metadata["summary"]
print(f"  {summary['voids']} closed voids, mean degree {summary['mean_degree']:.2f}")

# 2. the relative density against alpha
alphas = np.round(np.arange(0.0, 1.0, 0.15), 2)
curves = {}
for element, lam0 in (("plates", 0.03), ("beams", 0.06)):
    print(f"\nrelative density of 500 {element}, lambda_0 = {lam0}:")
    measured, estimated = [], []
    for alpha in alphas:
        model = ScaleRichNetwork(dimension=3, element=element, n_ligaments=500, decay=alpha,
                                 initial_thickness=lam0)  # fmt: skip
        realizations = [model.generate(cube, seed=s) for s in range(6)]
        # the mean field of beams counts the cylinders on their axes, not their union
        name = "volume_fraction" if element == "plates" else "cylinder_fraction"
        rho = [getattr(micro, name) for micro in realizations]
        theory = model.theory(cube)
        measured.append(np.mean(rho))
        estimated.append(theory.get("volume_fraction", np.nan))  # none: jammed
        line = (f"  alpha {alpha:.2f}: {np.mean(rho):.4f} +- {np.std(rho) / np.sqrt(6):.4f}, "
                f"mean field {estimated[-1]:.4f}")  # fmt: skip
        if element == "plates":
            line += f", jams above lambda_0 = {theory['jamming_thickness']:.3f}"
        print(line)
    curves[element] = (measured, estimated)

# 3. the networks drawn
networks = {}
for element, rho in (("plates", 0.3), ("beams", 0.15)):
    for alpha in (0.0, 0.9):
        model = ScaleRichNetwork(dimension=3, element=element, n_ligaments=300, decay=alpha,
                                 volume_fraction=rho)  # fmt: skip
        micro = model.generate(cube, seed=2)
        networks[element, alpha] = micro
        s = micro.metadata["summary"]
        print(f"\n{element}, alpha {alpha}: lambda_0 = {s['initial_thickness']:.4f}, relative "
              f"density {s['volume_fraction']:.4f}, geometric anisotropy "
              f"{s['geometric_anisotropy']:.3f}")  # fmt: skip

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # the extra [plot]
    sys.exit(0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
ax = axes[0]
samples = {"thickness": plates.thicknesses, "area": plates.areas, "degree": plates.degrees()}
for name, marker in (("thickness", "o"), ("area", "s"), ("degree", "^")):
    values = samples[name]
    values = values[values > 0]
    bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 30)
    density, edges = np.histogram(values, bins=bins, density=True)
    centres = np.sqrt(edges[1:] * edges[:-1])
    ax.loglog(centres[density > 0], density[density > 0], marker, ms=4, label=name)
ax.set_xlabel("thickness, area or degree")
ax.set_ylabel("probability density")
ax.set_title("10,000 plates, alpha = 0.5")
ax.legend(frameon=False)
ax = axes[1]
for element, colour in (("plates", "C0"), ("beams", "C1")):
    measured, estimated = curves[element]
    ax.plot(alphas, measured, "o", color=colour, label=f"{element}, realizations")
    ax.plot(alphas, estimated, "-", color=colour, label=f"{element}, mean field")
ax.set_xlabel("decay alpha")
ax.set_ylabel("relative density")
ax.set_yscale("log")
ax.set_title("500 plates (lambda_0 = 0.03), 500 beams (0.06)")
ax.legend(frameon=False, fontsize=8)
fig.savefig(out / "scale_rich_3d_laws.png", dpi=150)

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), layout="constrained")
u = (np.arange(400) + 0.5) / 400
x, y = np.meshgrid(u, u, indexing="ij")
plane = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.5)])
for ax, ((element, alpha), micro) in zip(axes, networks.items(), strict=True):
    inside = micro.contains(plane).reshape(400, 400)
    ax.imshow(inside.T, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=1.4)
    ax.set(xticks=[], yticks=[])
    ax.set_title(f"{element}, alpha = {alpha}")
fig.savefig(out / "scale_rich_3d_sections.png", dpi=150)
print(f"\nfigures in {out}")
