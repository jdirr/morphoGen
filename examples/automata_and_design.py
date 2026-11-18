"""Cellular automata, growth processes and combinatorial design.

1. A bicontinuous two-phase structure coarsened by Vichniac's annealing rule.
2. A partially crystallized polycrystal (nucleation and growth): the automaton's
   kinetics against the Avrami law.
3. A diffusion-limited aggregate and its fractal dimension.
4. Combinatorial metamaterials: exact count of the polarized-block assemblies, and a
   compatible assembly by wave function collapse.
5. Strut lattices with missing struts: Maxwell's rigidity transition.
6. Inverse design: the automaton rule whose structures look like a target image.

Figures are written when matplotlib is installed (``pip install 'morphogen[plot]'``).
Run time: about one minute (the first call of each kernel compiles it).
"""

import numpy as np

import morphogen as mg
from morphogen import automata, design
from morphogen.models import (
    AggregationModel,
    CellularAutomatonModel,
    NucleationGrowthModel,
    RandomLattice,
    TileAssembly,
)

rng = np.random.default_rng(2026)

# 1. Vichniac's annealing rule (twisted majority: B4678/S35678 on the Moore neighbourhood
#    in 2D, its analogue in 3D): random noise coarsens into a bicontinuous structure. The
#    rule treats both phases alike, so the volume fraction stays near 1/2 on average (the
#    'conserved' rule keeps any fraction exactly)
ca = automata.CellularAutomaton("anneal", neighbourhood="moore", dims=3)
evolution = ca.run(automata.random_state((48, 48, 48), 0.5, rng), steps=20, rng=rng)
h = evolution.history
print("annealing rule, 48^3 cells")
print(f"  volume fraction   {h['fraction'][0]:.3f} -> {h['fraction'][-1]:.3f}")
print(f"  interface density {h['interface'][0]:.3f} -> {h['interface'][-1]:.3f}")

# the same structure as a model: periodic, any domain, smooth interfaces off the lattice
model = CellularAutomatonModel(rule="anneal", steps=20, cell_size=0.5, smoothing=1.0)
micro = model.generate(mg.Domain.cube(24.0, periodic=True), seed=1)
image = micro.voxelize(shape=96)
print(f"  model: {micro.summary()}, measured volume fraction {image.data.mean():.3f}")

# 2. Continuous nucleation and isotropic growth (Johnson-Mehl grains), stopped at t = 12
nucleation = NucleationGrowthModel(
    nucleation="continuous", nucleation_rate=4e-4, growth_rate=1.0, time=12.0, dimension=2
)
domain2d = mg.Domain(size=(512.0, 512.0, 1.0), periodic=True)
grains = nucleation.generate(domain2d, seed=3)
theory = nucleation.theory(domain2d)
print("\nnucleation and growth (2D, continuous nucleation)")
print(f"  Avrami exponent {theory['avrami_exponent']:.0f}, grains {grains.summary()['n_grains']}")
print(
    f"  transformed fraction {grains.voxelize().data.mean():.3f}"
    f"  (Avrami {theory['transformed_fraction']:.3f})"
)

# 3. Diffusion-limited aggregation from one seed
dla = AggregationModel(process="dla", dimension=2, volume_fraction=0.05)
cluster = dla.generate(mg.Domain(size=(384.0, 384.0, 1.0)), seed=4)
# box counting between the width of the branches and the size of the cluster (boxes of a
# few cells see the branches as lines: local dimension near 1)
dimension = mg.analysis.box_dimension(cluster.voxelize(), sizes=[8, 16, 32])
print(f"\nDLA cluster: {cluster.summary()['n_particles']} particles")
print(f"  box-counting dimension {dimension:.2f} (theory {dla.theory()['fractal_dimension']})")

# 4. Polarized blocks: each cubic block carries a channel along one of 6 directions;
#    neighbours must deform compatibly (Coulais et al. 2016). Exact counts:
counts = [design.count_polarized_blocks((n, n, n)) for n in (1, 2, 3, 4)]
print(f"\ncompatible assemblies of n^3 polarized blocks, n = 1..4: {counts}")
tiles = design.polarized_blocks(size=8)
assembly = design.wave_function_collapse(tiles, (6, 6, 6), rng)
print(f"  wave function collapse on 6^3 blocks: compatible = {tiles.is_compatible(assembly)}")
metamaterial = TileAssembly(tiles="polarized", tile_cells=8).generate(mg.Domain.cube(6.0), seed=5)
print(f"  model: {metamaterial.summary()}")

# 5. Octet lattice with struts removed at random down to a mean coordination z:
#    stretching-dominated (no floppy macroscopic strain) above Maxwell's z = 6
print("\nrandom octet lattices (4^3 cells)")
for z in (12.0, 8.0, 6.5, 5.0):
    lattice = RandomLattice(cell="octet", coordination=z if z < 12 else None)
    s = lattice.generate(mg.Domain.cube(4.0), seed=6).summary()
    kind = "stretching" if s["floppy_strains"] == 0 else "bending"
    print(
        f"  z = {s['coordination']:5.2f}: {s['mechanisms']:4d} mechanisms,"
        f" {s['floppy_strains']} floppy strains -> {kind}-dominated"
    )

# 6. Find the automaton rule (and initial density) whose structures match the volume
#    fraction, interface density and correlation function of a target
target = image.data[::2, ::2, ::2]  # 48^3 cells of the structure of step 1
fit = design.fit_rule(target, rng=rng, shape=(20, 20, 20), steps=15, generations=8)
print(f"\nfitted rule {fit.rule} from noise of density {fit.initial_fraction:.2f}")
print(f"  descriptor misfit {fit.cost:.3f}")

try:
    from morphogen.viz import figures
except ImportError:
    figures = None
if figures is not None:
    try:
        figures.slices_figure(micro, shape=96, title="annealing rule").savefig("anneal.png")
        figures.history_figure(evolution.history).savefig("anneal_history.png")
        figures.slices_figure(grains, labels=True, title="Johnson-Mehl grains").savefig(
            "grains.png"
        )
        figures.kinetics_figure(grains).savefig("kinetics.png")
        figures.slices_figure(cluster, title="DLA").savefig("dla.png")
        figures.surface_figure(metamaterial, shape=48).savefig("polarized_blocks.png")
        frames = [f[:, :, 0] for f in nucleation.evolution(domain2d, seed=3, frames=24)]
        figures.save_gif(frames, "nucleation.gif", kind="labels", scale=2)
        print("\nfigures: anneal.png, grains.png, kinetics.png, dla.png, ...")
    except ImportError as exc:
        print(f"\nno figures: {exc}")
