# Measures and theory

The measures of `morphogen.analysis` work on voxel images, of a model or of any other
origin (a tomography, a section). Each model gives its closed forms beside them, so that a
measure is checked against theory; the [tutorial](tutorial.md#3-characterize-it) measures
a composite and its percolation.

```python
import morphogen as mg

model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
image = model.generate(mg.Domain.cube(20.0, periodic=True), seed=42).voxelize(shape=128)
print(mg.analysis.volume_fraction(image), model.theory()["volume_fraction"])
print(mg.analysis.specific_surface(image), model.theory()["specific_surface"])
```

## The theory of the models

Each model knows its closed-form characteristics when they exist: volume fraction,
specific surface, covariance, integral range, variance exponent, cell statistics. The
formulas and their sources are in the [theory notes](theory.md).

## Measures of the implicit geometry

A realization is described exactly, and so are its phases at any point of space: the
measures of `morphogen.morphology` ask the realization itself, not a voxel image of it, so
that their only errors are those of their sampling (which they give) and the fluctuation
of the realization — never the bias of a grid ([theory notes](theory.md) §104, §105):

* **Probes**: points (the fraction), pairs of points (the covariance, along a direction or
  over all directions), segments (the Choquet capacity and the linear contact
  distribution, by marching along them without ever crossing an interface), balls (the
  capacity and the spherical contact distribution, from the distance to the phase), and
  lines (the chords of each phase and the specific surface by Crofton's formula), with
  the standard deviations of their estimators.
* **Exact Minkowski functionals** of the unions of spheres (the Boolean models, the
  packings): the volume, the surface, the integral of mean curvature and the Euler
  characteristic of the union, on the torus of a periodic box, from the power diagram of
  the spheres and its dual complex; and those of the sections of the union by a plane,
  unions of discs (area, perimeter, Euler characteristic) — the densities that Miles'
  formulae and the stereological identities `L_A = (π/4) S_V`, `χ_A = M_V / 2π` give.
* **Operations**: the dilation of a union of spheres and fibres by a ball, exactly; the
  measures of a realization beside the closed forms of its model (`compare_with_theory`).

```python
from morphogen import morphology as mm

micro = model.generate(mg.Domain.cube(16.0, periodic=True), seed=1)
print(mm.minkowski_functionals(micro).densities())  # V_V, S_V, M_V, χ_V of the union
print(model.theory()["euler_density"])  # Miles' formula
print(mm.capacity(micro, "segment", [0.0, 1.0, 2.0]).values)  # T(l), exact probes
print(mm.compare_with_theory(micro).table())
```

The [tutorial of mathematical morphology](tutorial_morphology.md) follows the theory step
by step with these measures: the Choquet capacity and Matheron's tests of a Boolean model,
the morphological operations, the Minkowski functionals, stereology.

The same measures run by name from a configuration (`analysis: [exact_fraction,
exact_surface, exact_chords, exact_minkowski, exact_grains]`, which need no voxels: their
columns join `results.csv`), from the command line on a geometry (`morphogen analyze
spheres_0000.json --measure exact_minkowski`, beside the theory of its model), and in the
tool *Morphological measures* of the interface, whose table of the theory then gives the
exact densities beside the closed forms. `exact_grains` measures the grains of a
tessellation on its exact cells — their number, equivalent diameters, the ratios of the
ellipsoids of their second moments, the fractions of the phases — and the grains of seeds
against their seeds ([theory notes](theory.md) §116).

## The model of an image

An image — a volume of voxels, a stack of sections, or a single section — is turned into
its phases by Otsu's thresholds (one or several, after a smoothing of its noise), its
particles split and measured with the ellipsoids of their moments, and the scores of a
segmentation against a reference given (`morphology.segmentation`). Its morphological
model is then identified among candidates — Boolean spheres (of one radius or of a
lognormal law of radii), hard spheres, a Gaussian excursion, Poisson fibres, a mosaic of
Voronoi cells — each fitted to the fraction and the specific surface of the image by its
closed forms, simulated on voxels of the size of the image, calibrated, and ranked by
how well it reproduces the shape of the covariance and the contact distributions of the
image (`morphology.identification`, §106): from a volume, or from one section
(stereology). The image itself is also a microstructure of morphoGen, the model `image`:
its phases described by implicit functions at any point (the iso-surfaces of its
interpolated grey levels), in windows at the origin or drawn at random for the size of its
representative volume, ready for the probes, the meshes and the solvers (§107).

```python
from morphogen.models.image import ImageModel
from morphogen.morphology import identification, segmentation

grey_image = ImageModel().image()  # a tomography that comes with morphoGen
phases = segmentation.segment(grey_image, sigma=1.0)  # Otsu, after a smoothing
found = identification.identify(phases)  # candidates fitted and ranked
print(found.table())
larger = found.best.model.generate(mg.Domain.cube(50.0), seed=0)
```

## Structures rebuilt from their descriptors

A structure can also be rebuilt from descriptors alone, without a model of its own: the
cells of a periodic grid, relaxed to values between 0 and 1, descend the gradient of the
gap between their two-point probability at every lag (and their lineal paths along the
axes) and the targets, by L-BFGS-B, then are thresholded at the fraction
(`morphology.reconstruction`, as MCRpy; §114). The targets are the closed forms of a model
— the exact covariance of a Boolean model of spheres or of a Gaussian excursion, the lineal
path of the pores from the Choquet capacity — or the measures of an image; a single
section gives a volume, its two-point probability imposed on the three coordinate planes of
the grid. The covariance alone gives the fraction and the specific surface, not the
connectivity, which the lineal paths constrain: the pores of a Boolean model rebuilt from
their covariance miss their lineal path by 7 %, within 0.5 % with it.

```python
import morphogen as mg
from morphogen.morphology import reconstruction as rc

spheres = mg.models.BooleanSpheres(radius=4.0, volume_fraction=0.3)
target = rc.descriptors_of_model(
    spheres, (96, 96, 1), (1.0, 1.0, 1.0), phase=0, max_lag=16.0, lineal=True
)  # exact, from the closed forms
done = rc.reconstruct(target, lineal_weight=1.0, seed=0)
print(done.errors)  # the gaps of the two-point probability and of the lineal paths
# as a model: a volume from the exact covariance of a reference
model = mg.models.Reconstruction(reference=spheres, cell_size=1.0, method="gradient")
```

## Measures of images

* **Geometry**: volume fraction, specific surface (Cauchy–Crofton, 13 directions),
  covariance, cell volumes and neighbours, fractal dimension (box counting), chord
  lengths, granulometry by openings, local thickness and pore sizes, Euler
  characteristic and Minkowski densities, geometric tortuosity; the mean intercept
  length and its fabric tensor (degree of anisotropy), the first minima of the
  covariance (hole effect).
* **Connectivity and percolation**: connected components and percolation (labels, or a
  forest fire and its chemical distance), percolation thresholds of families of
  microstructures and their finite-size scaling to the infinite medium, in 2D and 3D;
  the anisotropy of percolation (along any direction across a core of the box, the
  thresholds of a family along directions, their ratio and threshold tensor); the
  rigidity threshold where the elastic moduli of the voxels vanish, and the critical
  exponents of the conductivity and of the elastic moduli at the threshold (t and f, by
  the finite-size scaling of the FFT solutions, or of the voxels assembled and factorized
  near the rigidity threshold, with their errors).
* **Torquato's descriptors**: lineal-path, two-point cluster and pore-size functions,
  radial distribution function and structure factor (hyperuniformity) of point
  patterns.
* **Mathematical morphology**: Choquet capacities of balls, segments, pairs and cubes,
  dilations, erosions, openings, closings and contact distributions, the morphological
  covariance of grey-level images, the four Minkowski functionals (with the integral of
  mean curvature) and their curves for excursion sets, Betti numbers.
* **Stereology**: section sizes and Saltykov's unfolding of the size distribution of
  spheres or oriented ellipsoids.
* **Particles**: the particles of a phase measured one by one in the volume
  (`analysis.particle_sizes`: touching particles split by a watershed of their distance
  map, their sizes and number per unit volume, with the frame correction of images that
  are not periodic).

## Crystallographic textures and orientation maps

`morphogen.texture` and `morphogen.io.ebsd` handle the orientations of polycrystals: the
symmetries of the eleven Laue classes, disorientations (against Mackenzie's distribution
for cubic crystals), mean orientations, pole figures in multiples of a random texture,
inverse pole figures and their colours; EBSD maps read and written (`.ang`, `.ctf`, with
the axes of each acquisition system), their grains, and the polycrystal of a map for
every solver (the model `ebsd`); the texture of a map given to the cells of a
tessellation of space (`texture: {type: measured, file: map.ctf}`); `morphogen texture`
for the numbers, the tables and the figures of a map. Along a loading path in finite
strains, the lattices that slip turns (`PathResult.texture`, the `orientation` of a
material point), checked against Schmid and Boas in single slip and against the fibres of
Taylor's polycrystal. The [tutorial on textures](tutorial_texture.md) goes from an
orientation map to a textured polycrystal.

## From the command line

```bash
morphogen analyze out/tpms/gyroid_0000.vti --measure volume_fraction specific_surface
morphogen identify scan.tif --sigma 1 -o model.yaml   # the model of a tomography, ranked
morphogen texture map.ctf -o texture/      # grains, pole figures, disorientations of an EBSD map
morphogen texture out/polycrystal_0000_tension_periodic_4.vti -o texture/  # lattices of a path
morphogen percolation spheres.yaml --lower 0.15 --upper 0.45  # percolation threshold
morphogen percolation fibres.yaml --lower 0.02 --upper 0.6 --directions lattice  # anisotropy
morphogen percolation spheres.yaml --lower 0.1 --upper 0.4 --sizes 24,32,48,64 \
    --exponent thermal --realizations 49   # the exponent t of the conductivity at p_c
morphogen percolation discs.yaml --lower 0.4 --upper 0.8 --sizes 32,48,64 \
    --exponent elastic --contrast 0        # f at the threshold of the moduli (--rigidity)
```
