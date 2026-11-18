"""Scale-rich networks (Both, Chen, ..., Portela & Barabási, Nature 659, 345, 2026): their
laws, their relative density and the anisotropy that their decay exponent tunes.

1. The tails of 30,000 ligaments (alpha = 0.5): thicknesses of exponent 1 + 1/alpha
   exactly, lengths and degrees of exponent 3 (the paper's Fig. 1e-g).
2. The relative density of 500 ligaments against alpha at lambda_0 = 0.06: the mean of
   realizations against the mean-field estimate of the theory, and the jamming threshold
   (Fig. 1c, h).
3. The anisotropy at 40 % relative density (Fig. 2b, c): the geometric anisotropy of the
   orientation tensor of the ligaments against the elastic anisotropy E_min / E_max of
   their stiffness (plane strain), samples in a thin frame as the cells of a periodic
   medium (voxel finite elements). The frame gives the anisotropy of kinematic conditions
   on the edges of the plane (the paper's), not their stiffness (docs/theory.md, §93).

A figure of the three, and the networks of the anisotropy study.

Usage: python examples/scale_rich_networks.py [output directory]   (about a minute)
"""

import sys
from pathlib import Path

import numpy as np

import morphogen as mg
from morphogen.homogenization import VOID, Material
from morphogen.homogenization.apparent import stiffness_tensor
from morphogen.homogenization.postprocess import directional_modulus
from morphogen.homogenization.voxel_fe import VoxelFEElasticProblem
from morphogen.models import ScaleRichNetwork

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/scale_rich")
out.mkdir(parents=True, exist_ok=True)
PIXELS = 320
plane = mg.Domain(size=(1.0, 1.0, 1.0 / PIXELS), periodic=False)

# 1. the laws of the network
big = ScaleRichNetwork(n_ligaments=30000, decay=0.5, initial_thickness=0.01)
net = big.generate(plane, seed=1)
tails = net.statistics(min_tail=100)
print("tails of 30,000 ligaments (alpha 0.5):")
for name, expected in (("thickness", 3.0), ("length", 3.0), ("degree", 3.0)):
    fit = tails[name]
    print(f"  {name:9s} {fit.exponent:.3f} +- {fit.error:.3f}  (above {fit.x_min:.3g}; {expected})")

# 2. the relative density against alpha
print("\nrelative density of 500 ligaments, lambda_0 = 0.06:")
alphas = np.round(np.arange(0.0, 1.0, 0.1), 1)
measured, estimated = [], []
for alpha in alphas:
    model = ScaleRichNetwork(n_ligaments=500, decay=alpha, initial_thickness=0.06)
    rho = [model.generate(plane, seed=s).volume_fraction for s in range(8)]
    theory = model.theory(plane)
    measured.append(np.mean(rho))
    estimated.append(theory.get("volume_fraction", np.nan))  # none: jammed
    estimate = "jammed" if np.isnan(estimated[-1]) else f"{estimated[-1]:.3f}"
    jams = theory["jamming_thickness"]
    print(f"  alpha {alpha:.1f}: {np.mean(rho):.3f} +- {np.std(rho) / np.sqrt(8):.3f}, "
          f"theory {estimate}, jams above lambda_0 = {jams:.3f}")  # fmt: skip

# 3. the anisotropy at 40 % relative density
print("\nanisotropy at 40 % relative density (500 ligaments, a frame of 0.006):")
materials = {0: VOID, 1: Material(young=1.0, poisson=0.3)}
angles = np.linspace(0.0, np.pi, 361)
study = []
for alpha in (0.0, 0.3, 0.6, 0.9):
    for seed in range(3):
        model = ScaleRichNetwork(n_ligaments=500, decay=alpha, volume_fraction=0.4, frame=0.006)
        micro = model.generate(plane, seed=seed)
        image = micro.voxelize(shape=(PIXELS, PIXELS, 1))
        stiffness, _ = stiffness_tensor(VoxelFEElasticProblem(image, materials, tol=1e-8))
        e = directional_modulus(stiffness, angles)
        summary = micro.metadata["summary"]
        study.append((alpha, summary["geometric_anisotropy"], e.min() / e.max(), e.max() / 0.4,
                      e.min() / 0.4, micro))  # fmt: skip
        print(f"  alpha {alpha:.1f} seed {seed}: eta_G {study[-1][1]:.3f}, eta_E "
              f"{study[-1][2]:.3f}, E_max / rho E_s {study[-1][3]:.3f}, E_min / rho E_s "
              f"{study[-1][4]:.3f}")  # fmt: skip
eta = np.array([s[1:3] for s in study])
print(f"  correlation of eta_G and eta_E: {np.corrcoef(eta[:, 0], eta[:, 1])[0, 1]:.2f}")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
except ImportError:  # the extra [plot]
    sys.exit(0)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
ax = axes[0]
for name, marker in (("thickness", "o"), ("length", "s"), ("degree", "^")):
    values = {"thickness": net.thicknesses, "length": net.lengths, "degree": net.degrees()}[name]
    values = values[values > 0]
    bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 30)
    density, edges = np.histogram(values, bins=bins, density=True)
    centres = np.sqrt(edges[1:] * edges[:-1])
    ax.loglog(centres[density > 0], density[density > 0], marker, ms=4, label=name)
ax.set_xlabel("thickness, length or degree")
ax.set_ylabel("probability density")
ax.set_title("30,000 ligaments, alpha = 0.5")
ax.legend(frameon=False)
ax = axes[1]
ax.plot(alphas, measured, "o", label="realizations")
ax.plot(alphas, estimated, "-", label="mean field")
ax.set_xlabel("decay alpha")
ax.set_ylabel("relative density")
ax.set_title("500 ligaments, lambda_0 = 0.06")
ax.legend(frameon=False)
ax = axes[2]
for alpha in (0.0, 0.3, 0.6, 0.9):
    rows = [s for s in study if s[0] == alpha]
    ax.plot([r[1] for r in rows], [r[2] for r in rows], "o", label=f"alpha = {alpha}")
ax.plot([0, 1], [0, 1], "k:", lw=0.8)
ax.set_xlabel("geometric anisotropy eta_G")
ax.set_ylabel("elastic anisotropy E_min / E_max")
ax.set_title("40 % relative density")
ax.legend(frameon=False, fontsize=8)
fig.savefig(out / "scale_rich_laws.png", dpi=150)

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), layout="constrained")
for ax, alpha in zip(axes, (0.0, 0.3, 0.6, 0.9), strict=True):
    micro = next(s[5] for s in study if s[0] == alpha)
    f = micro.frame
    frame = [
        [(0, 0), (1, 0), (1, f), (0, f)],
        [(0, 1 - f), (1, 1 - f), (1, 1), (0, 1)],
        [(0, 0), (f, 0), (f, 1), (0, 1)],
        [(1 - f, 0), (1, 0), (1, 1), (1 - f, 1)],
    ]
    ax.add_collection(PolyCollection(micro.growth.bodies() + frame, facecolors="#1f4e79"))
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
    ax.set_title(f"alpha = {alpha}")
fig.savefig(out / "scale_rich_networks.png", dpi=150)
print(f"\nfigures in {out}")
