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
morphogen texture map.ctf -o texture/      # grains, pole figures, disorientations of an EBSD map
morphogen texture out/polycrystal_0000_tension_periodic_4.vti -o texture/  # lattices of a path
morphogen percolation spheres.yaml --lower 0.15 --upper 0.45  # percolation threshold
morphogen percolation fibres.yaml --lower 0.02 --upper 0.6 --directions lattice  # anisotropy
morphogen percolation spheres.yaml --lower 0.1 --upper 0.4 --sizes 24,32,48,64 \
    --exponent thermal --realizations 49   # the exponent t of the conductivity at p_c
morphogen percolation discs.yaml --lower 0.4 --upper 0.8 --sizes 32,48,64 \
    --exponent elastic --contrast 0        # f at the threshold of the moduli (--rigidity)
```
