# Tutorials

Eleven studies to follow from start to end. The first ten are written in Python snippets
that follow each other in one session and run in seconds to a minute each, at sizes
chosen for that; the tests of morphoGen run them at each change, so that they stay
correct. The eleventh is a tour of the graphical interface, which runs the same functions
from forms. Each page shows the figures of its own snippets.

::::{container} gallery tutorials

:::{container} gallery-card
```{image} _static/tutorial/study_sections.png
:alt: sections of a Boolean model of spheres
```

**[A whole study](tutorial.md)** — SiC particles in an aluminium alloy, from the
installation to a behaviour learned from its responses: generation (command line,
Python, interface), measures against theory, percolation, apparent moduli and the size
of the representative volume, a plastic loading path, exports for MFront, Abaqus and
AMITEX_FFTP.

*Start here. About a minute in all.*
:::

:::{container} gallery-card
```{image} _static/tutorial/rve_variance.png
:alt: the variance of the volume fraction of spheres and of fibres against the volume
```

**[How large is a representative volume?](tutorial_rve.md)** — the statistics of
Kanit et al.: the variance of a property against the volume for spheres, whose
correlations are short, and for fibres on Poisson lines, whose correlations never stop;
the volumes that a precision asks for; the windows of one large image; the conductivity
and the bulk modulus of a composite from a configuration, with its report.

*About half a minute.*
:::

:::{container} gallery-card
```{image} _static/tutorial/rubber_tests.png
:alt: the tests of a rubber and the laws fitted to them
```

**[A rubber, from its tests to its tearing](tutorial_rubber.md)** — the finite strains:
a hyperelastic law fitted to three tests, the softening of Mullins at a material point,
a filled rubber stretched by half its length, and a cracked strip that tears at the
stretch of Rivlin and Thomas (the phase field in finite strains).

*About two minutes.*
:::

:::{container} gallery-card
```{image} _static/tutorial/geometry_mesh.png
:alt: the cut of a conforming mesh of three phases
```

**[Exact geometry, from implicit functions to meshes](tutorial_geometry.md)** — a
composite of three phases from the implicit function of each phase to its sections,
its surfaces (STL), its conforming mesh, and three solvers on the same geometry:
voxels, finite elements, extended finite elements.

*About forty seconds; needs the mesher MMG (extra `mesh`).*
:::

:::{container} gallery-card
```{image} _static/tutorial/iga_auxetic.png
:alt: a sheet with alternating holes before and after its optimization, and its Poisson's ratio
```

**[Isogeometric analysis, from a patch to an auxetic sheet](tutorial_iga.md)** — the
geometry exact on NURBS patches: Lamé's cylinder at the third order, periodic cells
(square, hexagonal, cut along no straight line), the hole of largest bulk modulus on
the bound of Hashin and Shtrikman, and a perforated sheet whose Poisson's ratio the
shape optimization brings to -0.8 at a given stiffness.

*About twenty seconds.*
:::

:::{container} gallery-card
```{image} _static/tutorial/contact_crush.png
:alt: the stress of a block with a slot against its compression, and the areas in contact
```

**[Contact and damage](tutorial_contact.md)** — a block with a flat pore crushed between
two platens until the pore closes on itself, an indenter that reads its modulus, the
walls of a periodic cell that touch along a loading path; then the damage laws at a
material point (Lemaitre, Kachanov and Rabotnov, Mazars) against their closed forms, and
a composite whose matrix fails.

*About twenty seconds.*
:::

:::{container} gallery-card
```{image} _static/tutorial/fracture_cleavage.png
:alt: the damage of the cleavage planes of a polycrystal, its stress, and the probability of cleavage
```

**[How a microstructure breaks](tutorial_fracture.md)** — interfaces that debond at
their strength and with their fracture energy, a pore that doubles the utilization of a
cyclic load, a crystal of magnesium that twins one way and not the other and a textured
polycrystal that keeps the asymmetry, a crystal and a polycrystal of iron that cleave,
and the probability of cleavage by the local approach of Beremin.

*About half a minute.*
:::

:::{container} gallery-card
```{image} _static/tutorial/texture_poles.png
:alt: the pole figures of an orientation map
```

**[From an orientation map to a textured polycrystal](tutorial_texture.md)** — an EBSD
map read with its axes, its grains and its texture (pole figures, the disorientations of
the boundaries), the stiffness of its polycrystal, a tension in which its lattices turn,
the fibres that a large extension forms, and a polycrystal of space with the texture of
the map.

*About twenty seconds.*
:::

:::{container} gallery-card
```{image} _static/tutorial/morphology_capacities.png
:alt: the capacities of balls and segments of a Boolean model, measured and in closed form
```

**[Mathematical morphology, from random sets to the model of an image](tutorial_morphology.md)**
— the Choquet capacity probed on the exact geometry and Matheron's tests of a Boolean
model, dilation, erosion, opening, closing and granulometry, the four Minkowski
functionals exactly against Miles' formulae, the stereology of sections, and the model of
a tomography found among the candidates.

*About fifteen seconds.*
:::

:::{container} gallery-card
```{image} _static/tutorial/plane_grains.png
:alt: the triangles of the grains of a sheet grown from seeds of the plane
```

**[Polycrystals of the plane, meshed and solved in their plane](tutorial_plane.md)** — the
grains of a rolled sheet grown from elliptical seeds, their exact polygons meshed by
triangles, the stiffness of the sheet under the three uniform conditions, the section of a
polycrystal of space, an orientation map meshed by its pixels, a plastic path and the files
of Abaqus.

*About a minute.*
:::

:::{container} gallery-card
```{image} _static/tutorial/interface_fracture.png
:alt: the tab Fracture of the graphical interface
```

**[A study in the graphical interface](tutorial_interface.md)** — the menus, the
workspace and the search of the page along a small study: a morphology, its measures, a
loading path, a crack, its files.

*No code. A few seconds per step.*
:::

::::

What the tutorials do not cover is in the [guide to the physics](physics.md) (every
problem of the homogenization module, in snippets that the tests run too), the
[recipes](howto.md) (short answers to "how do I...") and the [gallery](gallery.md) (a
configuration to start from for each model and solver).

```{toctree}
:hidden:
:maxdepth: 1

tutorial
tutorial_rve
tutorial_rubber
tutorial_geometry
tutorial_iga
tutorial_contact
tutorial_fracture
tutorial_texture
tutorial_morphology
tutorial_plane
tutorial_interface
```
