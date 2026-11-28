# Models of microstructures

A model holds the parameters of a random set or of a deterministic architecture, with the
closed-form characteristics of the set when they exist (`model.theory()`: volume
fraction, specific surface, covariance, integral range...). `model.generate(domain,
seed)` draws one realization of it in a box, periodic or not; the same seed gives the
same realization. morphoGen has 46 models, listed below by family, and takes the shapes of
[Microgen](https://github.com/3MAH/microgen) (3MAH) through the model `microgen`
([below](#shapes-of-microgen)): `morphogen models`
lists them, `morphogen models <type>` gives the parameters of one, and the
[gallery](gallery.md) shows each with a configuration that generates it. In a
configuration file, a model is the mapping of its `type` and its parameters
([configuration files](running.md#configuration-files)).

```python
import morphogen as mg

model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
micro = model.generate(mg.Domain.cube(20.0, periodic=True), seed=42)  # explicit geometry
print(model.theory()["volume_fraction"], model.theory()["specific_surface"])
```

What a realization is made of, and how it becomes images and meshes, is the subject of
[Geometry, voxels and meshes](geometry.md).

## The models by family

### Point processes and Boolean models

| Model | Description |
|---|---|
| `boolean_spheres` | Boolean model of spheres, constant or random radii (porous and granular media) |
| `penetrable_spheres` | Penetrable spheres with impenetrable cores: Torquato's cherry-pit model, between the Boolean model and hard spheres |
| `hardcore_spheres` | Non-overlapping spheres by random sequential addition, or dense random packings grown by the Lubachevsky–Stillinger algorithm (up to about 0.64) |

### Fibres and layers

| Model | Description |
|---|---|
| `boolean_cylinders` | Boolean model of finite fibres (cylinders or spherocylinders) of any law of the axes, with its exact covariance along any direction (Peyrega et al. 2009) and integral range; their sections meshed in the plane by their exact contours, flat caps, coatings and shells included |
| `poisson_fibers` | Infinite fibres on Poisson lines (Dirrenberger et al. 2014), isotropic or of any law of directions, with their exact covariance along any direction |
| `curved_fibers` | Curved fibres: worm-like chains whose tangents decorrelate along the contour (nonwovens, entangled networks), circular arcs of distributed radii of curvature (as in PuMA) or helices (crimp); a Boolean model or fibres without overlap; hollow (lumens); their sections meshed in the plane by their exact contours |
| `hardcore_fibers` | Non-overlapping fibres (spherocylinders) by random sequential addition, isotropic, aligned or planar, with a minimal gap; hollow (lumens); their sections meshed in the plane by their exact contours |
| `random_walk_fibers` | Bending fibres without overlap at high volume fractions (Altendorf & Jeulin 2011): random walks whose steps follow a von Mises–Fisher law between the main direction of their fibre (isotropic, aligned, planar, Schladitz's beta law, or mixtures of laws, as the orthotropic glass fibres of the 2014 paper) and the previous step, packed by repulsions against springs on the segments and the joints after an initial placement of least overlap (Altendorf, Jeulin & Willot 2014), then drawn into contact by the contact forces of Keilmann, Redenbach & Willot (2026); their sections meshed in the plane by their exact contours |
| `long_fibers` | Long fibres along an axis, the unidirectional composites: their sections a random packing (to 0.8) or a hexagonal or square array with a disorder, the fibres parallel or misaligned by an RMS angle (waving together over correlation lengths along and across them, or each on its own), monodisperse or polydisperse; their sections meshed by their exact contours, the ply's stiffness, conductivities and expansion estimated from them (`homogenization.misaligned`) |
| `cross_linked_fibers` | Entangled cross-linked fibres (Mezeix et al. 2009; Chatti et al. 2018, 2020): straight fibres, infinite on Poisson lines or finite, isotropic or in the Gaussian girdle of moulded fibres, bonded by cross-links (phase 2) at a fraction of their contacts, with the exact mean spacing of the contacts and cross-links |
| `delaunay_network` | Fibre networks on the edges of the periodic Delaunay triangulation of Poisson or hard-core points (Ban et al. 2016), in space or in a plane, cross-linked at every node (15.5 fibres per node in space, 6 in a plane), with the stiffness of their frame of beams |
| `random_laminate` | Layers of random thicknesses with holes through the thin layers (nanolayered films, Bironeau et al. 2016) |
| `woven_textile` | Woven composites: warp and weft tows of elliptic, lenticular or power-ellipse sections along crimped paths, 2D weaves (plain, twill, satin, basket) in plies and 3D weaves (orthogonal, layer-to-layer and through-thickness angle interlocks) of any binding depth, with or without stuffer warps; the variability of the tows along their paths (their centrelines, widths and thicknesses, as measured by tomography), periodic, correlated between neighbours; the tows a partition of their union, each segment with the orientation of its material for anisotropic tows, the tows from their fibres and matrix (composite cylinders), meshed from their exact geometry for the finite elements, each element turned with its tow; nominal fractions, crimps and thickness in closed form |

### Tessellations and polycrystals

| Model | Description |
|---|---|
| `voronoi`, `laguerre` | Poisson–Voronoi and Laguerre tessellations (Poisson germs, germs of sphere packings, or stealthy hyperuniform germs: disordered but regular cells): polycrystals with random orientations, multiphase mosaics, grain-boundary phases |
| `seeded_laguerre` | Grains grown from ellipsoidal seeds (MicroStructPy: Hart & Rimoli 2020): seeds of laws of equivalent diameters and of ratios a/b and b/c, their axes at random or aligned with the box, broken down into the spheres of their medial surfaces and placed the largest first within an overlap tolerance; the grain of a seed is the union of the Laguerre cells of its spheres, elongated and turned as the seed, of two or more phases and of crystal orientations; the seeds of each phase of their own laws, amorphous phases merged into one region (a matrix, a glass; a void left out of the meshes); boxes and cubes, or rectangles and squares in the plane (`shape: box`); ellipses in the plane (`dimension: 2`: columnar grains); meshed grain by grain, and verified against their seeds on their exact cells (`exact_grains`, the figures and the tool *Grains and their seeds*; [theory notes](theory.md) §115–§118) |
| `nested_grains` | Grains with subgrains: the germs of a finer tessellation in each grain, the subgrains turned by small angles from the crystal of their grain (low-angle boundaries), laths along an axis of the crystal; or grains filled with another model turned with their crystals (lamellar colonies, precipitates) |
| `ebsd` | Measured polycrystals: windows of an orientation map (EBSD: `.ang` and `.ctf` files), its grains found from the disorientations of its points, each with its mean orientation |

### Random functions and reconstructions

| Model | Description |
|---|---|
| `gaussian_excursion` | Excursions of Gaussian random fields (bicontinuous porous media) |
| `dead_leaves` | Dead leaves of spheres or spheroids: mosaics of non-convex grains, families of leaves of their own sizes, finite coverage over a background, with their exact covariances (Jeulin 1997) |
| `transparent_dead_leaves` | Excursions of the random function of transparent leaves (Jeulin 1997), with its exact covariance |
| `reconstruction` | Stochastic reconstruction (Yeong & Torquato 1998): a lattice annealed to the two-point probability and lineal-path function of a Debye medium, a model or an image; or (`method: gradient`, as MCRpy) its cells relaxed and descended to the two-point probability at every lag, from the closed forms of a model or from an image, a volume from a planar section (§114) |
| `image` | The microstructure of an image — a tomography, a stack of sections, a section — segmented into its phases (Otsu's thresholds, or its own phases) and described by implicit functions at any point (the iso-surfaces of its interpolated grey levels), in windows at its origin or drawn at random for each realization; a tomography of a particulate composite comes with morphoGen ([theory notes](theory.md) §107) |

### Lattices, foams and TPMS

| Model | Description |
|---|---|
| `tpms` | Gyroid, Schwarz P, Schwarz D and Neovius, sheet or network |
| `strut_lattice` | Cubic, BCC, FCC, octet, Kelvin, diamond and pyrochlore lattices, with solid or hollow struts; rotated and functionally graded lattices (non-periodic samples); their sections meshed in the plane by their exact contours |
| `random_lattice` | Strut lattices with missing struts and displaced nodes, with their rigidity (Maxwell–Calladine); their sections meshed in the plane by their exact contours |
| `hierarchical_lattice` | Hierarchical lattices of (hollow) tubes: frames of generation G of regular octahedra (Rayneau-Kirkhope, Mao & Farr 2012); their sections meshed in the plane by their exact contours |
| `fractal_sponge` | Menger sponge, Vicsek fractal, Cantor dust and any iterated function system on a cubic grid |
| `open_foam` | Open-cell foams and trabecular bone: struts on the edges of relaxed, elongated Voronoi cells, at a given relative density, with their framework and the stiffness of its frame of beams |
| `level_set_foam` | Open- or closed-cell foams as level sets of the distance fields of packed aligned ellipsoids: Plateau-border struts of controlled curvature thickening towards the nodes, at a given thickness or solid fraction (Degdeg & Dirrenberger); `calibrate_seeds` fits the seeds to measured section radii |
| `scale_rich_network` | Scale-rich networks (Both et al. 2026): ligaments of thicknesses `lambda_0 t^-alpha` grown one after another from points of the voids until they meet, extruded prisms; their thicknesses, lengths and degrees span orders of magnitude (power-law tails `1 + 1/alpha`, 3 and 3); the relative density in closed form (mean field) or solved on each realization, the jamming threshold; uniform, aligned or polarized ligaments, gradients of nucleation, a frame for periodic cells; in 3D (`dimension: 3`), plates that split convex voids into closed cells (areas and degrees of tails 5/2, the density and the jamming threshold in closed form) or beams grown until they touch (open cells, a mean-field density) |

### Automata and growth

| Model | Description |
|---|---|
| `cellular_automaton` | Two-phase structures grown by a cellular automaton from noise: annealing (bicontinuous), majority, conserved volume fraction, any `B/S` rule; on squares or cubes, triangles, hexagons, Penrose rhombi, Ammann–Beenker tiles, prisms, BCC or FCC cells, golden rhombohedra (quasi-periodic tilings in periodic boxes: their periodic approximants) |
| `activator_inhibitor` | Turing patterns of Young's activator–inhibitor automaton (spots, stripes, labyrinths) |
| `reaction_diffusion` | Gray–Scott reaction–diffusion patterns (spots, worms, labyrinths, holes), on the grid or the cells of any tiling |
| `nucleation_growth` | Nucleation and growth of grains (JMAK kinetics): Voronoi or Johnson–Mehl polycrystals, partially crystallized structures, faceted grains, Potts coarsening |
| `aggregation` | Diffusion-limited, Eden and ballistic aggregates: fractal clusters and deposits |
| `tile_assembly` | Compatible assemblies of building blocks: polarized blocks (combinatorial metamaterials), pipe networks |
| `branching` | Vascular trees, roots and corals grown by space colonization, radii by Murray's law |
| `spinodal` | Spinodal decomposition by the Cahn–Hilliard equation (spectral, semi-implicit): bicontinuous or droplet patterns of two phases at a fraction of the lever rule and an age of their coarsening (as `t^(1/3)`), the wavelength of the early decomposition as the unit of length (§113) |

### Bio-inspired structures

| Model | Description |
|---|---|
| `nacre` | Brick-and-mortar stacks of mineral tablets, polygonal or staggered bricks |
| `bouligand` | Twisted plywoods of fibres (arthropod cuticles, fish scales), periodic fibre families |
| `honeycomb` | Prismatic cells: hexagonal, square or random honeycombs; softwood with radial files, growth rings and rays; the honeycombs meshed exactly in their plane, their lumens polygons |
| `osteons` | Haversian bone: osteons with canals and cement lines, remodelled one after another (dead leaves) |

### Combinations and nested models

| Model | Description |
|---|---|
| `combination` | Union, intersection, difference and complement of models; inclusions in a matrix, e.g. a polycrystal; nested models (each fills the phase 1 of the previous one); the unions of unions of spheres and segments, and their differences where those removed lie within the first, meshed in the plane by their exact contours |
| `cox_spheres` | Spheres clustered in the phase 1 of another model: Cox models of nanocomposites (particles in aggregates), hard-core or not |
| `coating` | A coating (interphase) around the phase 1 of any model, or a shell inside it: hollow spheres, fibres and struts, core–shell particles |

### Shapes of Microgen

[Microgen](https://github.com/3MAH/microgen), developed by the 3MAH team, generates and
meshes microstructures from implicit fields. morphoGen calls it, installed apart (`pip install --pre microgen`), and
copies none of it: its shapes are built and evaluated by Microgen, then voxelized, meshed,
measured and homogenized by morphoGen like its own models
([Other codes](interop.md#libraries-and-programs)). Cite Microgen when you use them
([Citing](citing.md)).

| Model | Description |
|---|---|
| `microgen` | A shape of Microgen that has a field: the TPMS sheets and skeletons of its many surfaces, its spinodoids and primitives |

## Anisotropy

Point processes, tessellations, Gaussian fields, dead leaves, growths and lattices take the
same field `anisotropy: {stretch: [1, 1, 2], euler: [0, 0, 0]}` (or `[1, 1, 2]`): the
realization of the isotropic model mapped by stretches along turned principal axes,
preserving volumes (aligned ellipsoids, elongated cells and grains, anisotropic
covariances); periodic boxes take principal axes along their axes, the Gaussian field any.

## Monodisperse or polydisperse

Every size of the objects of a model is a number (one size) or a distribution (a size per
fibre, strut, sphere, cell or layer: uniform, lognormal, truncated normal, discrete), the
closed forms of the models taking the moments of the laws; where a volume fraction sets
the size, a law keeps its shape and its scale is solved on each realization.

## Automata and growth

The models `cellular_automaton`, `activator_inhibitor`, `reaction_diffusion`,
`nucleation_growth` and `aggregation` grow their structures on a lattice with the
automata of `morphogen.automata`, which also run on their own:

* Von Neumann, Moore, edge and Euclidean neighbourhoods in 2D and 3D; Life-like
  `B/S` rules, majority, Vichniac's annealing rule, majority at constant volume
  fraction, probabilistic rules, activator–inhibitor rules; synchronous,
  asynchronous and sequential updates.
* Gray–Scott reaction–diffusion, on the grid or on any tiling (finite-volume
  Laplacians).
* Automata on other lattices than the square grid: triangles, hexagons, Penrose
  rhombi (de Bruijn's pentagrid; quasi-periodic), Ammann–Beenker squares and rhombi
  (octagonal, quasi-periodic), the prisms stacking them in 3D, truncated octahedra
  (BCC), rhombic dodecahedra (FCC) and the golden rhombohedra of the icosahedral
  quasi-periodic tiling, rasterized to images cell by cell (`lattice: hexagonal`);
  the periodic approximants of the quasi-periodic tilings for periodic boxes (the same
  tiles in a rectangle or a cube: `approximant: 4`; the cubic approximants 1/1, 2/1,
  3/2... of the icosahedral tiling).
* Nucleation and growth by an event-driven automaton with exact (or faceted,
  oriented) growth fronts, checked against the Avrami law; normal grain growth by
  the Monte Carlo Potts model; diffusion-limited, Eden and ballistic aggregation.
* Lattice structures are exact on their lattice and smooth elsewhere (a level set
  for conforming meshes); non-periodic domains are cut from larger periodic
  simulations, so that their structures do not wrap around. Their formation can be
  watched step by step.

```python
import numpy as np
import morphogen as mg
from morphogen import automata
from morphogen.models import CellularAutomatonModel, NucleationGrowthModel

# an automaton on its own: Vichniac's annealing rule coarsens noise into a
# bicontinuous structure
rng = np.random.default_rng(1)
run = automata.CellularAutomaton("anneal", neighbourhood="moore").run(
    automata.random_state((64, 64, 64), 0.5, rng), steps=30, rng=rng
)
print(run.history["interface"][-1])  # interface density

# as a model: any domain, periodic or not, smooth interfaces off the lattice
micro = CellularAutomatonModel(rule="anneal", steps=30, cell_size=0.5).generate(
    mg.Domain.cube(20.0), seed=1
)

# a partially crystallized structure, and its kinetics against the Avrami law
model = NucleationGrowthModel(nucleation="continuous", nucleation_rate=1e-3, time=6.0)
grains = model.generate(mg.Domain.cube(40.0), seed=2)
print(grains.voxelize().data.mean(), model.theory()["transformed_fraction"])
frames = model.evolution(mg.Domain.cube(40.0), seed=2, frames=20)  # formation
```

## Combinations, nested and bio-inspired models

Models take models as parameters: particles clustered in the aggregates of another model,
coatings and shells around its phase, frames of tubes, fractal sponges. The structures of
biology are models like the others:

```python
# carbon black in rubber: particles in aggregates (a Cox model), with a bound-rubber phase
nano = mg.models.CoxSpheres(
    aggregates={"type": "boolean_spheres", "radius": 0.1, "volume_fraction": 0.25},
    radius=0.012,
    local_fraction=0.35,
    hardcore=True,
    aggregate_phase=2,
)
# hollow spheres, and a hierarchical lattice of hollow tubes (Rayneau-Kirkhope et al.)
shells = mg.models.Coating(
    model={"type": "hardcore_spheres", "radius": 0.1, "volume_fraction": 0.2},
    thickness=0.02,
    side="inside",
)
frames = mg.models.HierarchicalLattice(
    generations=1, cells=3, tube_radius=0.012, wall_thickness=0.004
)
menger = mg.models.FractalSponge(pattern="menger", level=3)
# from biology
nacre = mg.models.Nacre(
    arrangement="brick", tablet_size=1.0, tablet_thickness=0.2, mortar_thickness=0.05
)
plywood = mg.models.Bouligand(fiber_radius=0.03, fiber_spacing=0.08, pitch_angle=15.0)
wood = mg.models.Honeycomb(
    arrangement="wood",
    cell_size=0.4,
    wall_thickness=0.04,
    ring_width=4.0,
    ray_fraction=0.08,  # rays: 3D
)
bone = mg.models.OpenFoam(n_cells=64, volume_fraction=0.15, relaxation=10, anisotropy=1.6)
cortical = mg.models.Osteons(radius=100.0, coverage=0.65, cement_thickness=4.0)
tree = mg.models.BranchingNetwork(n_attractors=3000, step=0.02, murray_exponent=3.0)
print(cortical.theory())  # exact fractions of the four phases (dead leaves)
```

## Woven textiles

`woven_textile` sweeps the sections of the tows along the crimped paths of their weave
([theory notes](theory.md) §96, §110): 2D weaves (`plain`, `twill`, `satin`, `basket`) in
plies, and 3D weaves of straight warp layers between straight weft layers tied by binders.
The binding depth of a 3D weave, `binder_depth`, is the number of weft layers that a binder
ties together (Boussu, Cristian & Nauman 2015): through the thickness by default for
`orthogonal` (`layers + 1`), 2 for `layer_to_layer`; an orthogonal weave of shallower
binders sends the binders of the columns in turn between the layers, an angle interlock of
depth `D` has `layers + 2 - D` binders per column. `stuffers: false` leaves out the
straight warps of the angle interlocks: the binders, of the sizes of the warps by default,
are then the only warp yarns.

The tows of real fabrics wander about their idealized paths, and their sections change
along them (Vanaerschot et al. 2013, from the tomography of a twill laminate):

```python
import morphogen as mg

model = mg.models.WovenTextile(
    weave="twill",
    warp_spacing=2.0,
    yarn_width=1.6,
    yarn_thickness=0.3,
    deviation_in_plane=0.05,  # mm, the standard deviation of the centrelines across the tows
    deviation_through=0.02,  # through the thickness
    width_variation=0.08,  # coefficients of variation of the widths and thicknesses
    thickness_variation=0.1,
    size_correlation=-0.5,  # a tow pressed flatter is wider
    correlation_length=[3.0, 2.0],  # mm, along the tows: the centrelines, the sections
    neighbour_correlation=0.6,  # neighbouring tows of a ply wander together
)
micro = model.generate(mg.Domain(size=(16.0, 16.0, 1.0), periodic=True), seed=1)
print(micro.summary()["deviation_in_plane"], micro.summary()["interpenetration"])
```

Each quantity is a stationary Gaussian process along the direction of the tows (the
position of the slices of a tomography), of a gaussian covariance (smooth paths) or an
exponential one (`variation_covariance`), periodic over the box so that the tows close on
themselves across its faces; the sizes are lognormal, of the mean of their tow. The
deviations move the nodes of the centrelines, the frames of the sections turn with them,
and the half sizes vary linearly between the nodes, in the point-wise functions of the
phases from which the textile is meshed. The summary gives the deviations, the
coefficients of variation and the correlation realized; the theory, the fractions of the
mean varying sections along the lengthened paths. A tow, aligned fibres in the matrix,
can be given by its constituents and the fibre fraction in the tows, the transversely
isotropic material of the composite cylinder assemblage, turned along the paths like any
tow (`morphogen.homogenization.unidirectional`):

```yaml
homogenization:
  materials:
    0: {young: 3.5, poisson: 0.35}            # epoxy (GPa)
    1:
      unidirectional:
        fibre: {hexagonal: [18.08, 6.54, 4.97, 232.05, 15.0]}  # carbon: E_L 230, E_T 16 GPa
        matrix: {young: 3.5, poisson: 0.35}
        fibre_fraction: 0.6
```

## Scale-rich networks

The networks of Both et al. (2026) grow ligaments one after another, each thinner than
the previous one, until they meet; in the plane:

```python
from morphogen.homogenization.postprocess import directional_modulus

# Both et al. (2026): 500 ligaments of thicknesses lambda_0 t^-0.6 at 40 % relative density
model = mg.models.ScaleRichNetwork(n_ligaments=500, decay=0.6, volume_fraction=0.4)
plane = mg.Domain(size=(1.0, 1.0, 1 / 512), periodic=False)  # one voxel thick
net = model.generate(plane, seed=1)
print(model.theory(plane))  # lambda_0 of the density, the jamming threshold...
print(net.metadata["summary"])  # ligaments, density, degrees, geometric anisotropy
print({k: v.exponent for k, v in net.statistics().items()})  # power-law tails
# aligned, polarized (tangential for the first 20 %), graded nucleation; a frame for a
# periodic cell (FFT, Bloch waves)
aligned = mg.models.ScaleRichNetwork(volume_fraction=0.4, angles=[0, 90])
swirl = mg.models.ScaleRichNetwork(
    volume_fraction=0.4, pole=[0.5, 0.5], pole_angle=90, polarized_fraction=0.2
)
cell = mg.models.ScaleRichNetwork(volume_fraction=0.4, frame=0.006).generate(plane, seed=1)
# the elastic anisotropy E_min / E_max of a stiffness tensor C, in plane strain
# eta = directional_modulus(C, np.linspace(0, np.pi, 361)).min() / ....max()
```

In 3D, the plates or the beams of Fig. 6a of Both et al. (2026) fill the box:

```python
cube = mg.Domain(size=(1.0, 1.0, 1.0), periodic=False)
# plates: closed cells
plates = mg.models.ScaleRichNetwork(dimension=3, n_ligaments=2000, decay=0.5, volume_fraction=0.3)
cells = plates.generate(cube, seed=1)
print(plates.theory(cube))  # first plate area, lambda_0, jamming, tails 5/2...
print(cells.volume_fraction, cells.areas[:5], cells.degrees().mean())
# beams: open cells, struts
beams = mg.models.ScaleRichNetwork(
    dimension=3, element="beams", n_ligaments=500, decay=0.3, volume_fraction=0.15
)
struts = beams.generate(cube, seed=1)
segments = struts.segments()  # the beams as capsules (meshing, STL)
# layers, vertical walls, shells about a pole, a gradient along z
layers = mg.models.ScaleRichNetwork(
    dimension=3, volume_fraction=0.3, orientation={"type": "aligned", "direction": [0, 0, 1]}
)
shells = mg.models.ScaleRichNetwork(
    dimension=3,
    volume_fraction=0.3,
    pole=[0.5, 0.5, 0.5],
    pole_angle=90,
    nucleation_gradient=[1, 4],
    gradient_axis=2,
)
image = cells.voxelize(shape=128)
# the elastic anisotropy E_min / E_max of a 3D stiffness C (homogenization.postprocess):
# e = spatial_modulus(C, sphere_directions()); eta = e.min() / e.max()
```

`examples/scale_rich_networks.py` reproduces the laws of the paper and the anisotropy of
its networks, `examples/scale_rich_networks_3d.py` those of plates and beams.

## Templates

`morphogen template` lists the example configurations, and the [gallery](gallery.md)
shows each with its image. Among them, nested and bio-inspired structures:
`nested_grains` (27 grains of about twenty subgrains turned by 1 to 3°),
`cox_nanocomposite`, `coated_spheres`, `hierarchical_lattice`, `menger_sponge`, `nacre`,
`bouligand`, `wood`, `osteons`, `trabecular_bone`, `vascular_tree`. Fibres and lattices:
`hardcore_fibers`, `curved_fibers`, `random_walk_fibers` (bending fibres packed by forces
to 0.35), `contact_fibers` (the same drawn into contact, Keilmann et al. 2026),
`long_fibers` (a unidirectional composite at 0.6 misaligned by 3°), `woven_textile` (a 3D
orthogonal carbon/epoxy composite, its tows turned along their paths, its stiffness by
finite elements on the conforming mesh of its exact geometry), `cross_linked_fibers` (entangled carbon fibres bonded by epoxy, Chatti et al.),
`delaunay_network` (a fibre network on the edges of a Delaunay triangulation, Ban et
al.), `graded_lattice` (an octet lattice graded and turned, with its STL surface for
additive manufacturing), `microgen` (a sheet of Schoen's I-WP surface built by
Microgen, 3MAH). Dead leaves and Torquato's models: `sequential_dead_leaves`,
`transparent_dead_leaves`, `dense_packing`, `penetrable_spheres`, `reconstruction`,
`hyperuniform_cells`.
