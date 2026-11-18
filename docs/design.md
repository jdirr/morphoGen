# Design and optimization

Architected materials are designed as much as they are drawn at random: assemblies of
compatible building blocks, strut lattices with defects and their rigidity, periodic
cells optimized for their homogenized properties, holes shaped on their exact geometry,
and the patterns a periodic material may take after buckling. The designs are
microstructures like the others: they are meshed, homogenized and exported.

## Combinatorial design

`morphogen.design` builds assemblies and frameworks and searches design spaces:

* Tile sets with labelled faces; compatible assemblies by wave function collapse
  with backtracking; exact counts and exact uniform sampling by transfer matrices;
  polarized blocks after Coulais et al. (2016), with their exact counts.
* Periodic pin-jointed frameworks: random dilution and perturbation, rigidity
  (mechanisms, states of self-stress, floppy macroscopic strains).
* Simulated annealing, genetic algorithm, and the inverse design of automaton rules
  that reproduce the morphology of an image (`morphogen fit-rule`, below).

```python
import numpy as np
import morphogen as mg
from morphogen import design
from morphogen.models import RandomLattice, TileAssembly

rng = np.random.default_rng(3)
design.count_polarized_blocks((4, 4, 4))  # 145 456 074 compatible assemblies
blocks = design.polarized_blocks()
assembly = design.wave_function_collapse(blocks, (8, 8, 8), rng)
metamaterial = TileAssembly(tiles="polarized").generate(mg.Domain.cube(8.0), seed=3)

# an octet lattice diluted to a mean coordination of 5: below Maxwell's 6, it is floppy
lattice = RandomLattice(cell="octet", coordination=5.0).generate(mg.Domain.cube(4.0), seed=4)
print(lattice.summary()["floppy_strains"])
```

## Topology optimization

`morphogen.optimization` designs periodic cells of maximal bulk or shear modulus or
conductivity, of negative Poisson's ratio, or of a target stiffness tensor, by inverse
homogenization with the FFT solver and exact sensitivities: SIMP, density filter,
Heaviside projection, cubic or orthotropic symmetry; optimality criteria, method of
moving asymptotes, BESO and the hybrid cellular automaton. Designs are microstructures
like any other (meshing, finite element validation, exports).

```python
from morphogen.optimization import CellOptimization

# planar cell (one voxel through the thickness) with a negative Poisson's ratio
opt = CellOptimization(
    (64, 64, 1),
    0.35,
    objective="poisson_ratio",
    method="mma",
    filter_radius=3.0,
    projection=8.0,
    iterations=300,
)
cell = opt.run()
print(cell.properties())  # moduli of the black-and-white design
print(cell.refined_properties(2), cell.hinges)  # checks: finer grid, hinges
micro = cell.to_microstructure()  # smooth level set: mesh it, homogenize it
```

The same from a configuration file: `morphogen template optimize_auxetic -o
auxetic.yaml`, then `morphogen optimize auxetic.yaml`; in three dimensions, the template
`optimize_auxetic_3d` (a cell auxetic along its three axes, on 24³ voxels).

## Isogeometric analysis and shape optimization

`morphogen.iga` analyses NURBS patches and their assemblies (exact circles and
ellipses), refined without changing the geometry: linear elasticity (plane strain, plane
stress, solids) and conduction with the functions of the geometry; the effective tensors
of periodic cells, boxes or cells of any lattice (hexagonal arrays, patterns cut along no
straight line). Its shape optimization takes control points as variables, linked by the
symmetries of the pattern, with analytical sensitivities (the energy-momentum tensor)
and SLSQP or the moving asymptotes: the hole of largest bulk modulus of a cell reaches
the bound of Hashin and Shtrikman, and a sheet with alternating holes is brought to a
Poisson's ratio of -0.8 at a given stiffness. The [tutorial](tutorial_iga.md) goes from a
patch to that auxetic sheet.

## Patterns after buckling

`homogenization.post_bifurcation` and `morphogen.symmetry` give the patterns that a
periodic architectured material (a frame of beams, a tetrahedral mesh or a voxel image)
may adopt after buckling, from the symmetry of its cell (group theory and the
equivariant branching lemma, after Azulay & Combescure), with the first bifurcation of a
loading by linear buckling, and the cell they need from a Bloch-wave stability analysis
(cells of several primitive cells split by their translations and the little groups of
their wave vectors, the patterns that double the period). The frames are periodic frames
of Euler-Bernoulli beams in the plane (honeycomb, triangular, square, kagome) and in
space (`SpaceFrame`: cubic, body-centred cubic, octet, Kelvin, diamond and pyrochlore
lattices of struts, their apparent stiffness and relative density); the patterns are
drawn and animated in 3D. Along nonlinear loading paths in finite strains with nonlinear
behaviours (Hencky elasticity, von Mises elasto-(visco)plasticity, crystals, MFront
laws), `homogenization.path_bifurcation` follows the tangent of each step block by
block, the first bifurcation, its patterns and the bifurcated branch.

## Inverse design of automaton rules

`morphogen fit-rule image.vti -o rule.yaml` finds the automaton rule, and the density
of its initial noise, whose structures match the volume fraction, interface density and
correlation function of a two-phase image, and writes the configuration that generates
them; `--tiling hexagonal` (or any other lattice: `triangular`, `penrose`,
`ammann_beenker`, `bcc`, `fcc`, `icosahedral`) searches the rules of the automaton on
the cells of that tiling.
