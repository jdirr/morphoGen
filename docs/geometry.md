# Geometry, voxels and meshes

A realization is not an image: it is the exact description of one draw of its model, from
which images, surfaces and meshes are cut at any resolution. This page goes from that
description to what the solvers take: voxel images for the FFT solver and the measures,
conforming meshes for finite elements. The
[tutorial on exact geometry](tutorial_geometry.md) follows a composite of three phases
along the same way.

## Representations

* Each realization is first an explicit description: centres and radii, fibre axes,
  germs and weights, Fourier modes, surface equations. It is saved to JSON and
  regenerated exactly.
* It is voxelized at any resolution, with optional partial-volume voxels; volumes
  larger than memory are written slab by slab (`io.voxelize_to_file`).
* Every phase of every model has an implicit function (`phase_level_set`: negative
  in the phase, zero on its boundary): signed distances for unions of grains, minima
  and maxima for combinations, coatings and inclusions, distances to the spheres of
  the leaves, to the faces of the cells of tessellations and tilings, to the cubes of
  a sponge. The exports of the geometry (conforming meshes, surfaces, sections) are
  cut from them, not from voxels ([theory notes](theory.md) §75).

```python
import morphogen as mg

model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
micro = model.generate(mg.Domain.cube(20.0, periodic=True), seed=42)  # explicit geometry
image = micro.voxelize(shape=128)  # phase image

mg.io.save_image(image, "spheres.vti")  # ParaView, FFT codes
mg.io.save_mesh(mg.mesh.voxel_mesh(image, phases=[1]), "spheres.inp")
mg.io.save_microstructure(micro, "spheres.json")  # regenerates it exactly
```

## Meshes

A realization becomes vector, implicit or voxel geometry, conforming tetrahedral meshes or
voxel meshes for finite elements, and volumes for FFT solvers. The conforming meshes of
two phases need no voxels nor background grid (by default, `mesh: {method: auto}`):
tessellations from their exact cells, faceted morphologies from their exact convex
pieces, the others from their exact level set, refined where the interface needs it, its
nodes on the exact surface.

* Conforming tetrahedral meshes of every model, made with MMG from the implicit
  functions of its phases: the interfaces are followed within a Hausdorff distance,
  and opposite faces match for periodic media. Porous media are meshed without their
  pores; structures of several phases (coated inclusions, inclusions in a polycrystal,
  osteons, leaves of several phases) are cut phase by phase, conforming to all their
  interfaces.
* Polycrystals (Voronoi and Laguerre tessellations) are meshed grain by grain from
  their exact cells, after removing features smaller than a chosen length: conforming
  grain boundaries, one element set and orientation per grain, matching periodic
  faces.
* Periodic polycrystals meshed from their whole grains (periodized at the design):
  node-to-node periodic links, no cell cut by the faces of the box.

The meshes and images go to files in the formats of other codes
([Other codes and file formats](interop.md)).

## Symmetric cells

A cell symmetric about its mid-planes can be meshed in its irreducible part and
mirrored: the mesh then has its symmetries exactly (and matching opposite faces), as the
analyses by group theory need; any level-set function makes a microstructure:

```python
import numpy as np
import morphogen as mg
from morphogen.mesh import ImplicitBody, mirrored_mesh


def holes(points):  # negative in the solid: a square array of holes
    d = points[:, :2] % 1.0 - 0.5
    return 0.4348 - np.hypot(d[:, 0], d[:, 1])


body = ImplicitBody(mg.Domain(size=(1.0, 1.0, 0.13), periodic=True), holes)
mesh = mirrored_mesh(body, size=0.065, cells=(2, 2, 1), order=2, phases="solid")
# with the diagonal mirrors of the square (4/mmm), or of the cube: ((0, 1), (1, 2))
square = mirrored_mesh(body, size=0.065, diagonals=((0, 1),), cells=(2, 2, 1), phases="solid")
```
