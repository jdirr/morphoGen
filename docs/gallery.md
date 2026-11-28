# Gallery of examples

Each card below is one of the example configurations of morphoGen, its *templates*: the
image of its first realization, a line of description, the commands that reproduce it and
the configuration itself, to change at will. `morphogen template` lists them, `morphogen
template NAME -o NAME.yaml` writes one, `morphogen run NAME.yaml` runs it (`morphogen
optimize` for the topology optimizations, `morphogen laser` for the laser-architectured
sheets), and the graphical interface (`morphogen gui`)
opens the form of each model filled with its template. The [tutorial](tutorial.md) walks
through a whole study; the [configuration reference](configuration.md) describes the configurations.

The images show the realization that `morphogen run` makes first (the realization 0 of the
seed of the template), in its domain: three faces of its box, each the section of the
microstructure computed in its plane, as the interface draws them (anti-aliased
interfaces, the colours of the phases, a colour per grain of a polycrystal). A planar
structure shows its x-y section, a template that studies a planar section of its volume
(`voxels.section`) that section, an optimization its design, and the thin structures that
the faces barely cut (the vascular tree, the aggregates, the networks of struts or fibres
and the open foams) the surface of their phase 1 in 3D. Some cards show a second
figure, what the run of the template computes: the curve of its loading path, or the
damage of a cell that broke, drawn from the files of the run. They are written by
`python docs/gallery/make_gallery.py`, which writes this page too.

## Point processes and Boolean models

Grains placed at random points: overlapping, penetrable or hard, clustered in aggregates, coated.

::::{container} gallery

:::{container} gallery-card
:name: gallery-boolean-spheres

```{image} _static/gallery/boolean_spheres.png
:alt: boolean_spheres
:loading: lazy
```

**`boolean_spheres`** — Boolean model of spheres: overlapping grains of a porous or granular medium, with its exact covariance and integral range.

```bash
morphogen template boolean_spheres -o boolean_spheres.yaml
morphogen run boolean_spheres.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/boolean_spheres.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-penetrable-spheres

```{image} _static/gallery/penetrable_spheres.png
:alt: penetrable_spheres
:loading: lazy
```

**`penetrable_spheres`** — Penetrable spheres with impenetrable cores (Torquato's cherry-pit model), between the Boolean model and hard spheres.

```bash
morphogen template penetrable_spheres -o penetrable_spheres.yaml
morphogen run penetrable_spheres.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/penetrable_spheres.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-hardcore-spheres

```{image} _static/gallery/hardcore_spheres.png
:alt: hardcore_spheres
:loading: lazy
```

**`hardcore_spheres`** — Non-overlapping spheres by random sequential addition (jamming near 0.38).

```bash
morphogen template hardcore_spheres -o hardcore_spheres.yaml
morphogen run hardcore_spheres.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/hardcore_spheres.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-dense-packing

```{image} _static/gallery/dense_packing.png
:alt: dense_packing
:loading: lazy
```

**`dense_packing`** — A dense random packing of hard spheres grown to 0.6 by the Lubachevsky–Stillinger algorithm.

```bash
morphogen template dense_packing -o dense_packing.yaml
morphogen run dense_packing.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/dense_packing.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-cox-nanocomposite

```{image} _static/gallery/cox_nanocomposite.png
:alt: cox_nanocomposite
:loading: lazy
```

**`cox_nanocomposite`** — Particles clustered in the aggregates of a Boolean model (a Cox model), with a bound-rubber phase: carbon black in an elastomer.

```bash
morphogen template cox_nanocomposite -o cox_nanocomposite.yaml
morphogen run cox_nanocomposite.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/cox_nanocomposite.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-coated-spheres

```{image} _static/gallery/coated_spheres.png
:alt: coated_spheres
:loading: lazy
```

**`coated_spheres`** — Hard spheres with a coating of constant thickness: three phases, homogenized by FFT.

```bash
morphogen template coated_spheres -o coated_spheres.yaml
morphogen run coated_spheres.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/coated_spheres.yaml
:language: yaml
```

</details>
:::

::::

## Dead leaves, Gaussian fields and reconstructions

Random sets defined through random functions, and a random set rebuilt from its correlation functions.

::::{container} gallery

:::{container} gallery-card
:name: gallery-dead-leaves

```{image} _static/gallery/dead_leaves.png
:alt: dead_leaves
:loading: lazy
```

**`dead_leaves`** — Dead leaves of spheres: a two-phase mosaic of non-convex grains with random crystal orientations (Jeulin 1997).

```bash
morphogen template dead_leaves -o dead_leaves.yaml
morphogen run dead_leaves.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/dead_leaves.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-sequential-dead-leaves

```{image} _static/gallery/sequential_dead_leaves.png
:alt: sequential_dead_leaves
:loading: lazy
```

**`sequential_dead_leaves`** — Two families of spheroidal leaves, prolate and flat, falling on a background until it is covered 1.5 times.

```bash
morphogen template sequential_dead_leaves -o sequential_dead_leaves.yaml
morphogen run sequential_dead_leaves.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/sequential_dead_leaves.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-transparent-dead-leaves

```{image} _static/gallery/transparent_dead_leaves.png
:alt: transparent_dead_leaves
:loading: lazy
```

**`transparent_dead_leaves`** — The random function of transparent leaves of random grey levels, thresholded to a volume fraction of 0.35.

```bash
morphogen template transparent_dead_leaves -o transparent_dead_leaves.yaml
morphogen run transparent_dead_leaves.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/transparent_dead_leaves.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-gaussian-excursion

```{image} _static/gallery/gaussian_excursion.png
:alt: gaussian_excursion
:loading: lazy
```

**`gaussian_excursion`** — Excursion set of a Gaussian random field: a bicontinuous porous medium.

*Needs scikit-image (extra `surface`).*

```bash
morphogen template gaussian_excursion -o gaussian_excursion.yaml
morphogen run gaussian_excursion.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/gaussian_excursion.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-reconstruction

```{image} _static/gallery/reconstruction.png
:alt: reconstruction
:loading: lazy
```

**`reconstruction`** — A Boolean model of spheres rebuilt from its two-point and lineal-path functions by simulated annealing (Yeong & Torquato 1998).

```bash
morphogen template reconstruction -o reconstruction.yaml
morphogen run reconstruction.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/reconstruction.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-image

```{image} _static/gallery/image.png
:alt: image
:loading: lazy
```

**`image`** — The microstructure of an image: a tomography segmented by Otsu's threshold, its phases the iso-surfaces of its interpolated grey levels, in windows drawn at random.

```bash
morphogen template image -o image.yaml
morphogen run image.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/image.yaml
:language: yaml
```

</details>
:::

::::

## Fibres and layers

Straight, curved and bending fibres, with or without overlap, networks of fibres, laminates and woven textiles.

::::{container} gallery

:::{container} gallery-card
:name: gallery-boolean-cylinders

```{image} _static/gallery/boolean_cylinders.png
:alt: boolean_cylinders
:loading: lazy
```

**`boolean_cylinders`** — Boolean model of finite fibres (cylinders or spherocylinders) of any law of orientations.

```bash
morphogen template boolean_cylinders -o boolean_cylinders.yaml
morphogen run boolean_cylinders.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/boolean_cylinders.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-poisson-fibers

```{image} _static/gallery/poisson_fibers.png
:alt: poisson_fibers
:loading: lazy
```

**`poisson_fibers`** — Infinite fibres on isotropic Poisson lines (Dirrenberger et al. 2014), in a non-periodic domain.

```bash
morphogen template poisson_fibers -o poisson_fibers.yaml
morphogen run poisson_fibers.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/poisson_fibers.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-hardcore-fibers

```{image} _static/gallery/hardcore_fibers.png
:alt: hardcore_fibers
:loading: lazy
```

**`hardcore_fibers`** — Non-overlapping straight fibres by random sequential addition, at a minimal gap: a fibre-reinforced composite.

```bash
morphogen template hardcore_fibers -o hardcore_fibers.yaml
morphogen run hardcore_fibers.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/hardcore_fibers.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-curved-fibers

```{image} _static/gallery/curved_fibers.png
:alt: curved_fibers
:loading: lazy
```

**`curved_fibers`** — Boolean model of curved fibres: worm-like chains whose tangents decorrelate along the contour.

```bash
morphogen template curved_fibers -o curved_fibers.yaml
morphogen run curved_fibers.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/curved_fibers.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-curved-arcs

```{image} _static/gallery/curved_arcs.png
:alt: curved_arcs
:loading: lazy
```

**`curved_arcs`** — Circular arcs of uniform radii of curvature without overlap, lying close to a plane, as the fibrous preforms of PuMA.

```bash
morphogen template curved_arcs -o curved_arcs.yaml
morphogen run curved_arcs.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/curved_arcs.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-random-walk-fibers

```{image} _static/gallery/random_walk_fibers.png
:alt: random_walk_fibers
:loading: lazy
```

**`random_walk_fibers`** — Bending fibres packed without overlap by forces to 0.35 (Altendorf & Jeulin 2011).

```bash
morphogen template random_walk_fibers -o random_walk_fibers.yaml
morphogen run random_walk_fibers.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/random_walk_fibers.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-contact-fibers

```{image} _static/gallery/contact_fibers.png
:alt: contact_fibers
:loading: lazy
```

**`contact_fibers`** — The same bending fibres drawn into contact by contact forces (Keilmann, Redenbach & Willot 2026).

```bash
morphogen template contact_fibers -o contact_fibers.yaml
morphogen run contact_fibers.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/contact_fibers.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-long-fibers

```{image} _static/gallery/long_fibers.png
:alt: long_fibers
:loading: lazy
```

**`long_fibers`** — A unidirectional composite: long fibres of slightly scattered radii at 0.6, their sections a random packing, misaligned by 3 degrees.

```bash
morphogen template long_fibers -o long_fibers.yaml
morphogen run long_fibers.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/long_fibers.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-fibre-bed-plane

```{image} _static/gallery/fibre_bed_plane.png
:alt: fibre_bed_plane
:loading: lazy
```

```{image} _static/gallery/fibre_bed_plane_result.png
:alt: the result of fibre_bed_plane
:loading: lazy
```

**`fibre_bed_plane`** — The permeability of a bed of parallel fibres by the finite elements of the plane: the Stokes flows across and along the fibres through the pores of their sections, meshed by triangles curved along the fibres.

*Needs the mesher MMG (extra `mesh`), VTK and ffmpeg (extra `video`).*

```bash
morphogen template fibre_bed_plane -o fibre_bed_plane.yaml
morphogen run fibre_bed_plane.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/fibre_bed_plane.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-misaligned-ply-plane

```{image} _static/gallery/misaligned_ply_plane.png
:alt: misaligned_ply_plane
:loading: lazy
```

```{image} _static/gallery/misaligned_ply_plane_result.png
:alt: the result of misaligned_ply_plane
:loading: lazy
```

**`misaligned_ply_plane`** — The transverse stiffness and conductivity of a ply of fibres misaligned by 3 degrees in its plane: the exact contours of their sections, each fibre traced on its own distance function, meshed by curved triangles; the bounds of its axial stiffness and conductivity and the estimates of its thermal expansion, from the section turned along the segments of the fibres.

*Needs the mesher MMG (extra `mesh`), VTK and ffmpeg (extra `video`).*

```bash
morphogen template misaligned_ply_plane -o misaligned_ply_plane.yaml
morphogen run misaligned_ply_plane.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/misaligned_ply_plane.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-cross-linked-fibers

```{image} _static/gallery/cross_linked_fibers.png
:alt: cross_linked_fibers
:loading: lazy
```

**`cross_linked_fibers`** — Entangled carbon fibres moulded about a plane, half their contacts bonded by epoxy cross-links (Chatti et al.).

```bash
morphogen template cross_linked_fibers -o cross_linked_fibers.yaml
morphogen run cross_linked_fibers.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/cross_linked_fibers.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-delaunay-network

```{image} _static/gallery/delaunay_network.png
:alt: delaunay_network
:loading: lazy
```

**`delaunay_network`** — A fibre network on the edges of the Delaunay triangulation of random points, cross-linked at every node (Ban et al. 2016).

```bash
morphogen template delaunay_network -o delaunay_network.yaml
morphogen run delaunay_network.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/delaunay_network.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-random-laminate

```{image} _static/gallery/random_laminate.png
:alt: random_laminate
:loading: lazy
```

**`random_laminate`** — A nanolayered film: layers of lognormal thicknesses, the thinner ones broken up by holes (Bironeau et al. 2016).

```bash
morphogen template random_laminate -o random_laminate.yaml
morphogen run random_laminate.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/random_laminate.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-woven-textile

```{image} _static/gallery/woven_textile.png
:alt: woven_textile
:loading: lazy
```

**`woven_textile`** — A 3D orthogonal woven composite: straight warp and weft tows tied through the thickness by binders, each tow turned along its path, its stiffness by finite elements on the conforming mesh of its exact geometry.

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template woven_textile -o woven_textile.yaml
morphogen run woven_textile.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/woven_textile.yaml
:language: yaml
```

</details>
:::

::::

## Tessellations

Mosaics of cells: polycrystals, computed or measured, multiphase mosaics and closed cells.

::::{container} gallery

:::{container} gallery-card
:name: gallery-voronoi

```{image} _static/gallery/voronoi.png
:alt: voronoi
:loading: lazy
```

**`voronoi`** — A Poisson–Voronoi polycrystal of 343 grains with random crystal orientations.

```bash
morphogen template voronoi -o voronoi.yaml
morphogen run voronoi.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/voronoi.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-laguerre

```{image} _static/gallery/laguerre.png
:alt: laguerre
:loading: lazy
```

**`laguerre`** — A Laguerre (power) tessellation as a two-phase mosaic (Kanit et al. 2003).

```bash
morphogen template laguerre -o laguerre.yaml
morphogen run laguerre.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/laguerre.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-seeded-laguerre

```{image} _static/gallery/seeded_laguerre.png
:alt: seeded_laguerre
:loading: lazy
```

**`seeded_laguerre`** — Grains grown from ellipsoidal seeds (MicroStructPy; Hart & Rimoli 2020): the Laguerre cells of the spheres that fill each seed gathered into its grain, elongated along x as the seeds.

```bash
morphogen template seeded_laguerre -o seeded_laguerre.yaml
morphogen run seeded_laguerre.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/seeded_laguerre.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-seeded-matrix

```{image} _static/gallery/seeded_matrix.png
:alt: seeded_matrix
:loading: lazy
```

**`seeded_matrix`** — Crystals grown from seeds in an amorphous matrix: laths and equiaxed crystals of their own sizes, the seeds of the glass merged into one region; the grains checked against their seeds on their exact cells.

```bash
morphogen template seeded_matrix -o seeded_matrix.yaml
morphogen run seeded_matrix.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/seeded_matrix.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-seeded-plane

```{image} _static/gallery/seeded_plane.png
:alt: seeded_plane
:loading: lazy
```

**`seeded_plane`** — Grains of the plane grown from elliptical seeds: the section of a rolled sheet, grains elongated along x among the smaller equiaxed grains of a second phase, columnar through a thin slice.

```bash
morphogen template seeded_plane -o seeded_plane.yaml
morphogen run seeded_plane.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/seeded_plane.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-seeded-plane-stiffness

```{image} _static/gallery/seeded_plane_stiffness.png
:alt: seeded_plane_stiffness
:loading: lazy
```

```{image} _static/gallery/seeded_plane_stiffness_result.png
:alt: the result of seeded_plane_stiffness
:loading: lazy
```

**`seeded_plane_stiffness`** — The stiffness of the section of a rolled sheet by the finite elements of the plane: the columnar grains of its seeds meshed by triangles in their plane, crystals of copper, periodic, kinematic and static conditions.

*Needs the mesher MMG (extra `mesh`), VTK and ffmpeg (extra `video`).*

```bash
morphogen template seeded_plane_stiffness -o seeded_plane_stiffness.yaml
morphogen run seeded_plane_stiffness.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/seeded_plane_stiffness.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-seeded-precipitates

```{image} _static/gallery/seeded_precipitates.png
:alt: seeded_precipitates
:loading: lazy
```

**`seeded_precipitates`** — Cuboidal precipitates in a matrix (the cubes of gamma prime of a nickel superalloy): seeds of cubes aligned with the box, of lognormal sizes, in a matrix whose seeds merge into one region.

```bash
morphogen template seeded_precipitates -o seeded_precipitates.yaml
morphogen run seeded_precipitates.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/seeded_precipitates.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-nested-grains

```{image} _static/gallery/nested_grains.png
:alt: nested_grains
:loading: lazy
```

**`nested_grains`** — Grains with subgrains: each grain of a polycrystal divided by the germs of a finer tessellation, its subgrains turned by 1 to 3° from its crystal (low-angle boundaries).

```bash
morphogen template nested_grains -o nested_grains.yaml
morphogen run nested_grains.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/nested_grains.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-hyperuniform-cells

```{image} _static/gallery/hyperuniform_cells.png
:alt: hyperuniform_cells
:loading: lazy
```

**`hyperuniform_cells`** — The walls of the Voronoi cells of stealthy hyperuniform germs: disordered but regular closed cells.

```bash
morphogen template hyperuniform_cells -o hyperuniform_cells.yaml
morphogen run hyperuniform_cells.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/hyperuniform_cells.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-ebsd-plane

```{image} _static/gallery/ebsd_plane.png
:alt: ebsd_plane
:loading: lazy
```

```{image} _static/gallery/ebsd_plane_result.png
:alt: the result of ebsd_plane
:loading: lazy
```

**`ebsd_plane`** — A measured polycrystal by the finite elements of the plane: windows of an orientation map meshed by their pixels, each with the grain and the orientation of its pixel, under kinematic and static conditions.

*Needs the mesher MMG (extra `mesh`), VTK and ffmpeg (extra `video`).*

```bash
morphogen template ebsd_plane -o ebsd_plane.yaml
morphogen run ebsd_plane.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/ebsd_plane.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-ebsd

```{image} _static/gallery/ebsd.png
:alt: ebsd
:loading: lazy
```

**`ebsd`** — A measured polycrystal: windows of an orientation map (EBSD, an .ang or a .ctf file), its grains found from the disorientations of its points, each with its mean orientation; their stiffness by FFT.

```bash
morphogen template ebsd -o ebsd.yaml
morphogen run ebsd.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/ebsd.yaml
:language: yaml
```

</details>
:::

::::

## Foams and lattices

Architected materials: strut lattices, regular or not, hierarchical lattices, fractal sponges and foams.

::::{container} gallery

:::{container} gallery-card
:name: gallery-strut-lattice

```{image} _static/gallery/strut_lattice.png
:alt: strut_lattice
:loading: lazy
```

**`strut_lattice`** — An octet-truss strut lattice at 15 % relative density.

*Needs scikit-image (extra `surface`).*

```bash
morphogen template strut_lattice -o strut_lattice.yaml
morphogen run strut_lattice.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/strut_lattice.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-graded-lattice

```{image} _static/gallery/graded_lattice.png
:alt: graded_lattice
:loading: lazy
```

**`graded_lattice`** — An octet lattice graded along x and turned by 45° about z, with its STL surface for additive manufacturing.

*Needs scikit-image (extra `surface`).*

```bash
morphogen template graded_lattice -o graded_lattice.yaml
morphogen run graded_lattice.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/graded_lattice.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-random-lattice

```{image} _static/gallery/random_lattice.png
:alt: random_lattice
:loading: lazy
```

**`random_lattice`** — An octet lattice with missing struts and displaced nodes, and the rigidity of its framework (Maxwell–Calladine).

```bash
morphogen template random_lattice -o random_lattice.yaml
morphogen run random_lattice.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/random_lattice.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-hierarchical-lattice

```{image} _static/gallery/hierarchical_lattice.png
:alt: hierarchical_lattice
:loading: lazy
```

**`hierarchical_lattice`** — A hierarchical lattice of hollow tubes: frames of octahedra of generation 1 (Rayneau-Kirkhope, Mao & Farr 2012).

```bash
morphogen template hierarchical_lattice -o hierarchical_lattice.yaml
morphogen run hierarchical_lattice.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/hierarchical_lattice.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-honeycomb-plane

```{image} _static/gallery/honeycomb_plane.png
:alt: honeycomb_plane
:loading: lazy
```

```{image} _static/gallery/honeycomb_plane_result.png
:alt: the result of honeycomb_plane
:loading: lazy
```

**`honeycomb_plane`** — A hexagonal honeycomb of aluminium meshed exactly in its plane, its lumens the cells offset inward by half their walls: its stiffness and conductivity in space by the finite elements of the plane, the axial ones the fraction of the walls of aluminium's, the transverse ones near the beams of Gibson and Ashby.

*Needs the mesher MMG (extra `mesh`), VTK and ffmpeg (extra `video`).*

```bash
morphogen template honeycomb_plane -o honeycomb_plane.yaml
morphogen run honeycomb_plane.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/honeycomb_plane.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-scale-rich-network

```{image} _static/gallery/scale_rich_network.png
:alt: scale_rich_network
:loading: lazy
```

**`scale_rich_network`** — A scale-rich network: 500 ligaments of power-law thicknesses grown until they meet, at 40 % relative density (Both et al. 2026).

```bash
morphogen template scale_rich_network -o scale_rich_network.yaml
morphogen run scale_rich_network.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/scale_rich_network.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-scale-rich-plates

```{image} _static/gallery/scale_rich_plates.png
:alt: scale_rich_plates
:loading: lazy
```

**`scale_rich_plates`** — A scale-rich network in space: 400 plates of power-law thicknesses, each splitting a closed cell in two, at 30 % relative density.

```bash
morphogen template scale_rich_plates -o scale_rich_plates.yaml
morphogen run scale_rich_plates.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/scale_rich_plates.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-menger-sponge

```{image} _static/gallery/menger_sponge.png
:alt: menger_sponge
:loading: lazy
```

**`menger_sponge`** — The Menger sponge of level 3, of fractal dimension ln 20 / ln 3.

```bash
morphogen template menger_sponge -o menger_sponge.yaml
morphogen run menger_sponge.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/menger_sponge.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-level-set-foam

```{image} _static/gallery/level_set_foam.png
:alt: level_set_foam
:loading: lazy
```

**`level_set_foam`** — An open-cell copper foam as a level set of distance fields of packed ellipsoids (Degdeg & Dirrenberger 2026).

```bash
morphogen template level_set_foam -o level_set_foam.yaml
morphogen run level_set_foam.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/level_set_foam.yaml
:language: yaml
```

</details>
:::

::::

## TPMS

Triply periodic minimal surfaces, as sheets or networks (the stiffness of the gyroid is under Linear homogenization).

::::{container} gallery

:::{container} gallery-card
:name: gallery-tpms

```{image} _static/gallery/tpms.png
:alt: tpms
:loading: lazy
```

**`tpms`** — A sheet gyroid at 20 % relative density, with its watertight STL surface.

*Needs scikit-image (extra `surface`).*

```bash
morphogen template tpms -o tpms.yaml
morphogen run tpms.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/tpms.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-microgen

```{image} _static/gallery/microgen.png
:alt: microgen
:loading: lazy
```

**`microgen`** — A sheet of Schoen's I-WP surface built by [Microgen](https://github.com/3MAH/microgen) (3MAH), which morphoGen calls: any shape of Microgen with a field (TPMS, spinodoids, primitives) is taken as a model (needs the package microgen; cite Microgen when you use it).

```bash
morphogen template microgen -o microgen.yaml
morphogen run microgen.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/microgen.yaml
:language: yaml
```

</details>
:::

::::

## Automata and tilings

Structures grown on lattices by local rules, and assemblies of building blocks.

::::{container} gallery

:::{container} gallery-card
:name: gallery-cellular-automaton

```{image} _static/gallery/cellular_automaton.png
:alt: cellular_automaton
:loading: lazy
```

**`cellular_automaton`** — A bicontinuous structure coarsened from noise by Vichniac's annealing rule.

```bash
morphogen template cellular_automaton -o cellular_automaton.yaml
morphogen run cellular_automaton.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/cellular_automaton.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-activator-inhibitor

```{image} _static/gallery/activator_inhibitor.png
:alt: activator_inhibitor
:loading: lazy
```

**`activator_inhibitor`** — Turing patterns of Young's activator–inhibitor automaton.

```bash
morphogen template activator_inhibitor -o activator_inhibitor.yaml
morphogen run activator_inhibitor.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/activator_inhibitor.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-reaction-diffusion

```{image} _static/gallery/reaction_diffusion.png
:alt: reaction_diffusion
:loading: lazy
```

**`reaction_diffusion`** — Turing patterns of the Gray–Scott reaction–diffusion system (Pearson 1993).

*Needs scikit-image (extra `surface`).*

```bash
morphogen template reaction_diffusion -o reaction_diffusion.yaml
morphogen run reaction_diffusion.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/reaction_diffusion.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-tile-assembly

```{image} _static/gallery/tile_assembly.png
:alt: tile_assembly
:loading: lazy
```

**`tile_assembly`** — A compatible assembly of pipe tiles drawn by wave function collapse: a network without dead ends.

```bash
morphogen template tile_assembly -o tile_assembly.yaml
morphogen run tile_assembly.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/tile_assembly.yaml
:language: yaml
```

</details>
:::

::::

## Growth

Structures grown by a process in time: grains, aggregates and trees.

::::{container} gallery

:::{container} gallery-card
:name: gallery-nucleation-growth

```{image} _static/gallery/nucleation_growth.png
:alt: nucleation_growth
:loading: lazy
```

**`nucleation_growth`** — A Johnson–Mehl polycrystal: grains nucleated at a constant rate grow until they impinge (JMAK kinetics).

```bash
morphogen template nucleation_growth -o nucleation_growth.yaml
morphogen run nucleation_growth.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/nucleation_growth.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-aggregation

```{image} _static/gallery/aggregation.png
:alt: aggregation
:loading: lazy
```

**`aggregation`** — Diffusion-limited aggregation: fractal clusters grown by random walkers (Witten & Sander 1981).

```bash
morphogen template aggregation -o aggregation.yaml
morphogen run aggregation.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/aggregation.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-vascular-tree

```{image} _static/gallery/vascular_tree.png
:alt: vascular_tree
:loading: lazy
```

**`vascular_tree`** — A vascular tree grown by space colonization, its radii by Murray's law.

*Needs scikit-image (extra `surface`).*

```bash
morphogen template vascular_tree -o vascular_tree.yaml
morphogen run vascular_tree.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/vascular_tree.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-spinodal

```{image} _static/gallery/spinodal.png
:alt: spinodal
:loading: lazy
```

**`spinodal`** — Spinodal decomposition: two interwoven phases of a symmetric mixture by the Cahn–Hilliard equation, at an age of their coarsening.

```bash
morphogen template spinodal -o spinodal.yaml
morphogen run spinodal.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/spinodal.yaml
:language: yaml
```

</details>
:::

::::

## Bio-inspired

Microstructures drawn from biology.

::::{container} gallery

:::{container} gallery-card
:name: gallery-nacre

```{image} _static/gallery/nacre.png
:alt: nacre
:loading: lazy
```

**`nacre`** — Nacre: staggered aragonite tablets bonded by a soft organic mortar (brick and mortar).

```bash
morphogen template nacre -o nacre.yaml
morphogen run nacre.yaml --seed 1
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/nacre.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-bouligand

```{image} _static/gallery/bouligand.png
:alt: bouligand
:loading: lazy
```

**`bouligand`** — A Bouligand twisted plywood: layers of fibres turned by 15° from one to the next (the dactyl club of the mantis shrimp).

```bash
morphogen template bouligand -o bouligand.yaml
morphogen run bouligand.yaml --seed 1
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/bouligand.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-wood

```{image} _static/gallery/wood.png
:alt: wood
:loading: lazy
```

**`wood`** — Softwood: radial files of tracheids through a growth ring, with rays.

```bash
morphogen template wood -o wood.yaml
morphogen run wood.yaml --seed 1
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/wood.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-osteons

```{image} _static/gallery/osteons.png
:alt: osteons
:loading: lazy
```

**`osteons`** — Cortical bone: osteons with their Haversian canals and cement lines, remodelled one after another.

```bash
morphogen template osteons -o osteons.yaml
morphogen run osteons.yaml --seed 1
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/osteons.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-trabecular-bone

```{image} _static/gallery/trabecular_bone.png
:alt: trabecular_bone
:loading: lazy
```

**`trabecular_bone`** — Trabecular bone or an open-cell foam: struts on the edges of relaxed, elongated Voronoi cells.

```bash
morphogen template trabecular_bone -o trabecular_bone.yaml
morphogen run trabecular_bone.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/trabecular_bone.yaml
:language: yaml
```

</details>
:::

::::

## Linear homogenization

Apparent elastic and thermal properties, by finite elements on conforming meshes or by the FFT solver on voxels.

::::{container} gallery

:::{container} gallery-card
:name: gallery-homogenization-spheres

```{image} _static/gallery/homogenization_spheres.png
:alt: homogenization_spheres
:loading: lazy
```

**`homogenization_spheres`** — The apparent moduli of stiff spheres in a soft matrix under the four boundary conditions (finite elements: 10-node tetrahedra whose edges follow the spheres).

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_spheres -o homogenization_spheres.yaml
morphogen run homogenization_spheres.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_spheres.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-gyroid

```{image} _static/gallery/homogenization_gyroid.png
:alt: homogenization_gyroid
:loading: lazy
```

**`homogenization_gyroid`** — The cubic stiffness tensor and the conductivity of a sheet gyroid cell (finite elements, quadratic tetrahedra).

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_gyroid -o homogenization_gyroid.yaml
morphogen run homogenization_gyroid.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_gyroid.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-voronoi

```{image} _static/gallery/homogenization_voronoi.png
:alt: homogenization_voronoi
:loading: lazy
```

**`homogenization_voronoi`** — The apparent moduli and stiffness tensor of a two-phase Voronoi mosaic (FFT).

```bash
morphogen template homogenization_voronoi -o homogenization_voronoi.yaml
morphogen run homogenization_voronoi.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_voronoi.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-polycrystal

```{image} _static/gallery/homogenization_polycrystal.png
:alt: homogenization_polycrystal
:loading: lazy
```

**`homogenization_polycrystal`** — The apparent moduli of a copper polycrystal meshed grain by grain (finite elements).

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_polycrystal -o homogenization_polycrystal.yaml
morphogen run homogenization_polycrystal.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_polycrystal.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-polycrystal-inclusions

```{image} _static/gallery/homogenization_polycrystal_inclusions.png
:alt: homogenization_polycrystal_inclusions
:loading: lazy
```

**`homogenization_polycrystal_inclusions`** — A copper polycrystal with an amorphous grain-boundary phase and alumina inclusions (FFT).

```bash
morphogen template homogenization_polycrystal_inclusions -o homogenization_polycrystal_inclusions.yaml
morphogen run homogenization_polycrystal_inclusions.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_polycrystal_inclusions.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-mixed-loading

```{image} _static/gallery/homogenization_mixed_loading.png
:alt: homogenization_mixed_loading
:loading: lazy
```

**`homogenization_mixed_loading`** — Thermo-elastic loadings mixing strains and stresses on a unidirectional glass-epoxy composite.

```bash
morphogen template homogenization_mixed_loading -o homogenization_mixed_loading.yaml
morphogen run homogenization_mixed_loading.yaml --seed 1
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_mixed_loading.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-crystallization

```{image} _static/gallery/homogenization_crystallization.png
:alt: homogenization_crystallization
:loading: lazy
```

**`homogenization_crystallization`** — A partially crystallized material: grains of the Avrami law meshed conformingly (finite elements).

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_crystallization -o homogenization_crystallization.yaml
morphogen run homogenization_crystallization.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_crystallization.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-interphase

```{image} _static/gallery/homogenization_interphase.png
:alt: homogenization_interphase
:loading: lazy
```

**`homogenization_interphase`** — Spheres, their interphase and the matrix meshed conformingly from the implicit function of each phase: three phases by finite elements whose nodes lie on the interfaces, with the level sets and the surfaces of the phases.

*Needs the mesher MMG (extra `mesh`), scikit-image (extra `surface`).*

```bash
morphogen template homogenization_interphase -o homogenization_interphase.yaml
morphogen run homogenization_interphase.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_interphase.yaml
:language: yaml
```

</details>
:::

::::

## Nonlinear behaviours

Loading paths of elasto-viscoplastic composites and polycrystals, a rubber in finite strains, a reduced model, a matrix that plasticity damages, and a porous cell whose walls come into contact.

::::{container} gallery

:::{container} gallery-card
:name: gallery-homogenization-viscoplastic

```{image} _static/gallery/homogenization_viscoplastic.png
:alt: homogenization_viscoplastic
:loading: lazy
```

```{image} _static/gallery/homogenization_viscoplastic_result.png
:alt: the result of homogenization_viscoplastic
:loading: lazy
```

**`homogenization_viscoplastic`** — SiC particles in a viscoplastic aluminium alloy: tension, relaxation and unloading (finite elements: 10-node tetrahedra, which do not lock when the matrix flows).

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_viscoplastic -o homogenization_viscoplastic.yaml
morphogen run homogenization_viscoplastic.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_viscoplastic.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-crystal-plasticity

```{image} _static/gallery/homogenization_crystal_plasticity.png
:alt: homogenization_crystal_plasticity
:loading: lazy
```

**`homogenization_crystal_plasticity`** — A nickel polycrystal pulled to 10 % in finite strains, Méric–Cailletaud crystal plasticity (finite elements).

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_crystal_plasticity -o homogenization_crystal_plasticity.yaml
morphogen run homogenization_crystal_plasticity.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_crystal_plasticity.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-crystal-plasticity-fft

```{image} _static/gallery/homogenization_crystal_plasticity_fft.png
:alt: homogenization_crystal_plasticity_fft
:loading: lazy
```

```{image} _static/gallery/homogenization_crystal_plasticity_fft_result.png
:alt: the result of homogenization_crystal_plasticity_fft
:loading: lazy
```

**`homogenization_crystal_plasticity_fft`** — The same on 24³ voxels of 20 grains, to 5 % (the nonlinear FFT solver).

```bash
morphogen template homogenization_crystal_plasticity_fft -o homogenization_crystal_plasticity_fft.yaml
morphogen run homogenization_crystal_plasticity_fft.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_crystal_plasticity_fft.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-rubber

```{image} _static/gallery/homogenization_rubber.png
:alt: homogenization_rubber
:loading: lazy
```

```{image} _static/gallery/homogenization_rubber_result.png
:alt: the result of homogenization_rubber
:loading: lazy
```

**`homogenization_rubber`** — A filled rubber stretched by half its length, unloaded and stretched again: the eight chains of Arruda and Boyce and the softening of Mullins, in finite strains (the nonlinear FFT solver).

```bash
morphogen template homogenization_rubber -o homogenization_rubber.yaml
morphogen run homogenization_rubber.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_rubber.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-ntfa

```{image} _static/gallery/homogenization_ntfa.png
:alt: homogenization_ntfa
:loading: lazy
```

**`homogenization_ntfa`** — A reduced model (NTFA) of an elasto-plastic composite trained on full-field paths.

```bash
morphogen template homogenization_ntfa -o homogenization_ntfa.yaml
morphogen run homogenization_ntfa.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_ntfa.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-ductile-damage

```{image} _static/gallery/homogenization_ductile_damage.png
:alt: homogenization_ductile_damage
:loading: lazy
```

```{image} _static/gallery/homogenization_ductile_damage_result.png
:alt: the result of homogenization_ductile_damage
:loading: lazy
```

**`homogenization_ductile_damage`** — SiC particles in an aluminium matrix whose plasticity is coupled to Lemaitre's damage: the composite reaches its strength, softens as the matrix is damaged and unloads along its damaged modulus (the nonlinear FFT solver, a delayed damage).

```bash
morphogen template homogenization_ductile_damage -o homogenization_ductile_damage.yaml
morphogen run homogenization_ductile_damage.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_ductile_damage.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-contact

```{image} _static/gallery/homogenization_contact.png
:alt: homogenization_contact
:loading: lazy
```

```{image} _static/gallery/homogenization_contact_result.png
:alt: the result of homogenization_contact
:loading: lazy
```

**`homogenization_contact`** — Plates parted by gaps and held by pillars, compressed until the gaps close: the walls of the pores press on each other (self-contact with Coulomb's friction, in finite strains) and the cell stiffens fivefold.

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_contact -o homogenization_contact.yaml
morphogen run homogenization_contact.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_contact.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-intergranular

```{image} _static/gallery/homogenization_intergranular.png
:alt: homogenization_intergranular
:loading: lazy
```

```{image} _static/gallery/homogenization_intergranular_result.png
:alt: the result of homogenization_intergranular
:loading: lazy
```

**`homogenization_intergranular`** — A polycrystal that breaks along its grain boundaries: cohesive interfaces between the grains of a Voronoi tessellation of cubic crystals. The stress passes a peak and falls to zero as a crack crosses the cell from boundary to boundary; the work that broke it is the fracture energy times the area of the crack.

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_intergranular -o homogenization_intergranular.yaml
morphogen run homogenization_intergranular.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_intergranular.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-grain-fracture

```{image} _static/gallery/homogenization_grain_fracture.png
:alt: homogenization_grain_fracture
:loading: lazy
```

```{image} _static/gallery/homogenization_grain_fracture_result.png
:alt: the result of homogenization_grain_fracture
:loading: lazy
```

**`homogenization_grain_fracture`** — Cracks between the grains and through them: a polycrystal in crystal plasticity whose grain boundaries are cohesive and whose {100} planes cleave. With boundaries weaker than the planes the crack follows them (intergranular fracture); with strong ones it cuts through the grains along their cleavage planes (transgranular).

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_grain_fracture -o homogenization_grain_fracture.yaml
morphogen run homogenization_grain_fracture.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_grain_fracture.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-twinning

```{image} _static/gallery/homogenization_twinning.png
:alt: homogenization_twinning
:loading: lazy
```

```{image} _static/gallery/homogenization_twinning_result.png
:alt: the result of homogenization_twinning
:loading: lazy
```

**`homogenization_twinning`** — The twins of a textured magnesium: a polycrystal whose axes c are near the load. A twin shears in one sense only: pulled, the grains twin and flow at a low stress; compressed, they do not twin and are twice as hard (crystal plasticity with twins, the nonlinear FFT solver).

```bash
morphogen template homogenization_twinning -o homogenization_twinning.yaml
morphogen run homogenization_twinning.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_twinning.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-detwinning

```{image} _static/gallery/homogenization_detwinning.png
:alt: homogenization_detwinning
:loading: lazy
```

```{image} _static/gallery/homogenization_detwinning_result.png
:alt: the result of homogenization_detwinning
:loading: lazy
```

**`homogenization_detwinning`** — The twins of a textured magnesium, there and back: pulled along the axes c of its grains, the polycrystal twins, and the lattices of its points become those of their predominant twins, grain by grain (86 degrees off: the polycrystal hardens), the turned volume of each grain following its twinned volume; brought back, the twins shear back at about the stress of the tension (detwinning), and the lattices turn back to those of their grains.

```bash
morphogen template homogenization_detwinning -o homogenization_detwinning.yaml
morphogen run homogenization_detwinning.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_detwinning.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-cleavage

```{image} _static/gallery/homogenization_cleavage.png
:alt: homogenization_cleavage
:loading: lazy
```

```{image} _static/gallery/homogenization_cleavage_result.png
:alt: the result of homogenization_cleavage
:loading: lazy
```

**`homogenization_cleavage`** — The cleavage of a polycrystal of iron at a low temperature: the {100} planes of its grains open at their strength and soften (a crack band of one voxel), and a crack crosses the grains from plane to plane. With a softer slip the grains flow first: slip shields cleavage.

```bash
morphogen template homogenization_cleavage -o homogenization_cleavage.yaml
morphogen run homogenization_cleavage.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_cleavage.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-slip-cleavage

```{image} _static/gallery/homogenization_slip_cleavage.png
:alt: homogenization_slip_cleavage
:loading: lazy
```

```{image} _static/gallery/homogenization_slip_cleavage_result.png
:alt: the result of homogenization_slip_cleavage
:loading: lazy
```

**`homogenization_slip_cleavage`** — Cleavage that the slip nucleates: a polycrystal of iron whose grains slip far below the strength of their {100} planes. The slip of a crystal lowers that strength, and the damage of the planes eases the slip in return: the polycrystal yields, hardens, and breaks across a section once its grains have slipped enough, under a lower stress and at a smaller strain than with planes of a constant strength.

```bash
morphogen template homogenization_slip_cleavage -o homogenization_slip_cleavage.yaml
morphogen run homogenization_slip_cleavage.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_slip_cleavage.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-cleavage-band

```{image} _static/gallery/homogenization_cleavage_band.png
:alt: homogenization_cleavage_band
:loading: lazy
```

```{image} _static/gallery/homogenization_cleavage_band_result.png
:alt: the result of homogenization_cleavage_band
:loading: lazy
```

**`homogenization_cleavage_band`** — Cleavage across a band of a given width: the {100} planes of a columnar polycrystal of iron regularized by their microdamage, a field that follows their damage over a length. A plane breaks across a band five voxels wide, whatever the grid, where the fracture energy of the planes is dissipated; the crack crosses the grains in a band that winds a little from one to the next, and the planes of its middle are broken through.

```bash
morphogen template homogenization_cleavage_band -o homogenization_cleavage_band.yaml
morphogen run homogenization_cleavage_band.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_cleavage_band.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-crack-closure

```{image} _static/gallery/homogenization_crack_closure.png
:alt: homogenization_crack_closure
:loading: lazy
```

```{image} _static/gallery/homogenization_crack_closure_result.png
:alt: the result of homogenization_crack_closure
:loading: lazy
```

**`homogenization_crack_closure`** — A cleavage crack that closes: a polycrystal of iron pulled until a crack has crossed its grains, then pushed. The opening of the cleavage planes goes back under a low pressure, and once the crack is closed the polycrystal carries the pressure as the sound one, its lips rubbing on each other (the unilateral condition of the planes, Coulomb's friction).

```bash
morphogen template homogenization_crack_closure -o homogenization_crack_closure.yaml
morphogen run homogenization_crack_closure.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_crack_closure.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-beremin

```{image} _static/gallery/homogenization_beremin.png
:alt: homogenization_beremin
:loading: lazy
```

```{image} _static/gallery/homogenization_beremin_result.png
:alt: the result of homogenization_beremin
:loading: lazy
```

**`homogenization_beremin`** — The probability of cleavage of a ferritic steel (the local approach of Beremin): the Weibull stress of the voxels of a polycrystal that have yielded, each driven by the normal stress of the {100} planes of its grain, gives the probability that the weakest link of a tensile bar has failed, from nothing at the yield stress to a certainty as the steel hardens.

```bash
morphogen template homogenization_beremin -o homogenization_beremin.yaml
morphogen run homogenization_beremin.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_beremin.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-fatigue

```{image} _static/gallery/homogenization_fatigue.png
:alt: homogenization_fatigue
:loading: lazy
```

```{image} _static/gallery/homogenization_fatigue_result.png
:alt: the result of homogenization_fatigue
:loading: lazy
```

**`homogenization_fatigue`** — Pores in a cast alloy under a tension that alternates: Dang Van's criterion of multiaxial fatigue on the stresses of a cycle maps the utilization of each point, the largest at the equators of the pores, and gives the factor of safety of the cell.

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_fatigue -o homogenization_fatigue.yaml
morphogen run homogenization_fatigue.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_fatigue.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-debonding

```{image} _static/gallery/homogenization_debonding.png
:alt: homogenization_debonding
:loading: lazy
```

```{image} _static/gallery/homogenization_debonding_result.png
:alt: the result of homogenization_debonding
:loading: lazy
```

**`homogenization_debonding`** — Glass beads that debond from their matrix: cohesive interfaces between the phases (a strength, a fracture energy, Coulomb's friction once broken). The composite softens to a matrix with holes in tension and finds most of its stiffness again in compression.

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template homogenization_debonding -o homogenization_debonding.yaml
morphogen run homogenization_debonding.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_debonding.yaml
:language: yaml
```

</details>
:::

::::

## Fracture

Cells that break along a loading path, by the phase-field method: cracks start, run and merge in the phases without being meshed or tracked; a planar section of the volume (`voxels.section`) makes them quick.

::::{container} gallery

:::{container} gallery-card
:name: gallery-fracture-composite

```{image} _static/gallery/fracture_composite.png
:alt: fracture_composite
:loading: lazy
```

```{image} _static/gallery/fracture_composite_result.png
:alt: the result of fracture_composite
:loading: lazy
```

**`fracture_composite`** — Glass beads in a brittle epoxy pulled until a crack crosses the cell: the strength and the work of fracture of the composite.

```bash
morphogen template fracture_composite -o fracture_composite.yaml
morphogen run fracture_composite.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/fracture_composite.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-fracture-rubber

```{image} _static/gallery/fracture_rubber.png
:alt: fracture_rubber
:loading: lazy
```

```{image} _static/gallery/fracture_rubber_result.png
:alt: the result of fracture_rubber
:loading: lazy
```

**`fracture_rubber`** — A perforated rubber sheet that tears at a stretch of 1.45, in finite strains: the ligaments between the holes break one after another.

```bash
morphogen template fracture_rubber -o fracture_rubber.yaml
morphogen run fracture_rubber.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/fracture_rubber.yaml
:language: yaml
```

</details>
:::

::::

## Multiphysics: transport, couplings, harmonic regimes, flow and sound

The other physics of the homogenization module on the same microstructures: transport of a scalar potential, weak and strong couplings, harmonic regimes, the flow of a fluid through the pores and the acoustics of porous media, and the extended finite elements that cut a regular grid by the interfaces ([guide to the physics](physics.md)).

::::{container} gallery

:::{container} gallery-card
:name: gallery-homogenization-transport

```{image} _static/gallery/homogenization_transport.png
:alt: homogenization_transport
:loading: lazy
```

**`homogenization_transport`** — Thermal and electrical conductivities, permittivity and diffusivity of conducting, impermeable spheres in a polymer (FFT), with their bounds.

```bash
morphogen template homogenization_transport -o homogenization_transport.yaml
morphogen run homogenization_transport.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_transport.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-piezoelectric

```{image} _static/gallery/homogenization_piezoelectric.png
:alt: homogenization_piezoelectric
:loading: lazy
```

**`homogenization_piezoelectric`** — A piezoelectric composite of barium titanate particles in a polymer: the displacement and the electric potential solved together, the constants of the effective law.

```bash
morphogen template homogenization_piezoelectric -o homogenization_piezoelectric.yaml
morphogen run homogenization_piezoelectric.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_piezoelectric.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-viscoelastic

```{image} _static/gallery/homogenization_viscoelastic.png
:alt: homogenization_viscoelastic
:loading: lazy
```

```{image} _static/gallery/homogenization_viscoelastic_result.png
:alt: the result of homogenization_viscoelastic
:loading: lazy
```

**`homogenization_viscoelastic`** — Glass beads in a viscoelastic resin: storage and loss moduli over five decades of frequency, and the relaxation in time.

```bash
morphogen template homogenization_viscoelastic -o homogenization_viscoelastic.yaml
morphogen run homogenization_viscoelastic.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_viscoelastic.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-poroelastic

```{image} _static/gallery/homogenization_poroelastic.png
:alt: homogenization_poroelastic
:loading: lazy
```

**`homogenization_poroelastic`** — The poroelastic law of a porous solid: drained stiffness, tensor and modulus of Biot, undrained response with water in the pores.

```bash
morphogen template homogenization_poroelastic -o homogenization_poroelastic.yaml
morphogen run homogenization_poroelastic.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_poroelastic.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-xfem

```{image} _static/gallery/homogenization_xfem.png
:alt: homogenization_xfem
:loading: lazy
```

**`homogenization_xfem`** — Extended finite elements: a regular grid cut by the level set of the spheres, elastic and thermal properties without a mesher.

```bash
morphogen template homogenization_xfem -o homogenization_xfem.yaml
morphogen run homogenization_xfem.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_xfem.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-homogenization-acoustic-foam

```{image} _static/gallery/homogenization_acoustic_foam.png
:alt: homogenization_acoustic_foam
:loading: lazy
```

**`homogenization_acoustic_foam`** — The flow and the sound in an open-cell foam: Darcy's permeability by a Stokes flow through the pores, and the parameters of its equivalent fluid (tortuosity, characteristic lengths).

```bash
morphogen template homogenization_acoustic_foam -o homogenization_acoustic_foam.yaml
morphogen run homogenization_acoustic_foam.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/homogenization_acoustic_foam.yaml
:language: yaml
```

</details>
:::

::::

## RVE studies

The size of the representative volume element by the ensemble and window methods.

::::{container} gallery

:::{container} gallery-card
:name: gallery-rve-poisson-fibers

```{image} _static/gallery/rve_poisson_fibers.png
:alt: rve_poisson_fibers
:loading: lazy
```

**`rve_poisson_fibers`** — The variance of the volume fraction and specific surface of Poisson fibres at five sizes (ensemble method).

```bash
morphogen template rve_poisson_fibers -o rve_poisson_fibers.yaml
morphogen run rve_poisson_fibers.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/rve_poisson_fibers.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-rve-poisson-fibers-fe

```{image} _static/gallery/rve_poisson_fibers_fe.png
:alt: rve_poisson_fibers_fe
:loading: lazy
```

**`rve_poisson_fibers_fe`** — The RVE sizes of the elastic and thermal properties of Poisson fibres, KUBC and MBC (finite elements).

*Needs the mesher MMG (extra `mesh`).*

```bash
morphogen template rve_poisson_fibers_fe -o rve_poisson_fibers_fe.yaml
morphogen run rve_poisson_fibers_fe.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/rve_poisson_fibers_fe.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-rve-voronoi-windows

```{image} _static/gallery/rve_voronoi_windows.png
:alt: rve_voronoi_windows
:loading: lazy
```

**`rve_voronoi_windows`** — The window method on one large two-phase Voronoi mosaic.

```bash
morphogen template rve_voronoi_windows -o rve_voronoi_windows.yaml
morphogen run rve_voronoi_windows.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/rve_voronoi_windows.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-rve-cellular-automaton

```{image} _static/gallery/rve_cellular_automaton.png
:alt: rve_cellular_automaton
:loading: lazy
```

**`rve_cellular_automaton`** — The window method on a structure grown by a cellular automaton.

```bash
morphogen template rve_cellular_automaton -o rve_cellular_automaton.yaml
morphogen run rve_cellular_automaton.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/rve_cellular_automaton.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-rve-crystal-plasticity-fft

```{image} _static/gallery/rve_crystal_plasticity_fft.png
:alt: rve_crystal_plasticity_fft
:loading: lazy
```

**`rve_crystal_plasticity_fft`** — The RVE sizes of the elastic and viscoplastic responses of a polycrystal (Yang et al. 2019; FFT).

```bash
morphogen template rve_crystal_plasticity_fft -o rve_crystal_plasticity_fft.yaml
morphogen run rve_crystal_plasticity_fft.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/rve_crystal_plasticity_fft.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-rve-multiphysics

```{image} _static/gallery/rve_multiphysics.png
:alt: rve_multiphysics
:loading: lazy
```

**`rve_multiphysics`** — The RVE sizes of several physics at once: bulk modulus, conductivity, swelling, loss modulus and the stress of a loading path of the same realizations (FFT).

```bash
morphogen template rve_multiphysics -o rve_multiphysics.yaml
morphogen run rve_multiphysics.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/rve_multiphysics.yaml
:language: yaml
```

</details>
:::

::::

## Topology optimization

Periodic cells optimized for their homogenized properties; the images show their designs.

::::{container} gallery

:::{container} gallery-card
:name: gallery-optimize-bulk

```{image} _static/gallery/optimize_bulk.png
:alt: optimize_bulk
:loading: lazy
```

**`optimize_bulk`** — A 3D cell of maximal bulk modulus at 30 % of solid (method of moving asymptotes).

```bash
morphogen template optimize_bulk -o optimize_bulk.yaml
morphogen optimize optimize_bulk.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/optimize_bulk.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-optimize-auxetic

```{image} _static/gallery/optimize_auxetic.png
:alt: optimize_auxetic
:loading: lazy
```

**`optimize_auxetic`** — A planar cell of negative Poisson's ratio, near -0.87 (method of moving asymptotes).

```bash
morphogen template optimize_auxetic -o optimize_auxetic.yaml
morphogen optimize optimize_auxetic.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/optimize_auxetic.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-optimize-auxetic-3d

```{image} _static/gallery/optimize_auxetic_3d.png
:alt: optimize_auxetic_3d
:loading: lazy
```

**`optimize_auxetic_3d`** — A 3D cell of negative Poisson's ratio along its three axes, near -0.14 on 24^3 voxels (method of moving asymptotes).

```bash
morphogen template optimize_auxetic_3d -o optimize_auxetic_3d.yaml
morphogen optimize optimize_auxetic_3d.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/optimize_auxetic_3d.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-optimize-conductivity

```{image} _static/gallery/optimize_conductivity.png
:alt: optimize_conductivity
:loading: lazy
```

**`optimize_conductivity`** — A 3D cell of maximal conductivity at 30 % of solid (hybrid cellular automaton).

```bash
morphogen template optimize_conductivity -o optimize_conductivity.yaml
morphogen optimize optimize_conductivity.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/optimize_conductivity.yaml
:language: yaml
```

</details>
:::

::::

## Laser-architectured sheets

Steel sheets architectured by localized laser treatment, their tensile test and the optimization of the laser path (`morphogen laser`); the images show the hardness of the parallel part of the specimen through its thickness (the mean, the upper and the lower layers) and the axes of the tracks.

::::{container} gallery

:::{container} gallery-card
:name: gallery-laser-corrugated

```{image} _static/gallery/laser_corrugated.png
:alt: laser_corrugated
:loading: lazy
```

**`laser_corrugated`** — Corrugated tracks of martensite in an annealed 0.23 C steel by a double-sided laser (Wang 2022), and their amplitude and period optimized for the strength-ductility balance (Bayesian optimization).

```bash
morphogen template laser_corrugated -o laser_corrugated.yaml
morphogen laser laser_corrugated.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/laser_corrugated.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-laser-bending

```{image} _static/gallery/laser_bending.png
:alt: laser_bending
:loading: lazy
```

**`laser_bending`** — A strip hardened on one face by longitudinal tracks in three-point bending: the number of tracks and the power optimized for the load at a deflection of 1 mm within a budget of laser energy.

```bash
morphogen template laser_bending -o laser_bending.yaml
morphogen laser laser_bending.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/laser_bending.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-laser-cells

```{image} _static/gallery/laser_cells.png
:alt: laser_cells
:loading: lazy
```

**`laser_cells`** — The corrugated track of Wang et al. (2023) drawn by a 1 mm spot on cells of 0.5 mm and changed by a cellular automaton (neighbour exchanges tested by finite elements) for the failure strain at a tensile strength of 700 MPa, the martensite brittle; its tracks treated by the laser.

```bash
morphogen template laser_cells -o laser_cells.yaml
morphogen laser laser_cells.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/laser_cells.yaml
:language: yaml
```

</details>
:::

::::

## Couplings and the whole chain

Transient diffusion in a microstructure, and every step from the model to the figures and videos.

::::{container} gallery

:::{container} gallery-card
:name: gallery-diffusion

```{image} _static/gallery/diffusion.png
:alt: diffusion
:loading: lazy
```

**`diffusion`** — Moisture uptake of a composite of hollow flax-like fibres exposed on two faces.

```bash
morphogen template diffusion -o diffusion.yaml
morphogen run diffusion.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/diffusion.yaml
:language: yaml
```

</details>
:::

:::{container} gallery-card
:name: gallery-workflow

```{image} _static/gallery/workflow.png
:alt: workflow
:loading: lazy
```

**`workflow`** — The whole chain: model, voxels, conforming mesh, homogenization, a loading path, tables, figures and videos.

*Needs the mesher MMG (extra `mesh`), VTK and ffmpeg (extra `video`).*

```bash
morphogen template workflow -o workflow.yaml
morphogen run workflow.yaml
```

<details><summary>Configuration</summary>

```{literalinclude} ../src/morphogen/templates/workflow.yaml
:language: yaml
```

</details>
:::

::::
