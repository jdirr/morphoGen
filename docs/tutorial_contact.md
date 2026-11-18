# Tutorial: contact and damage, a porous block crushed and a matrix that fails

Two things end the elastic life of a microstructure long before it breaks in two: its
surfaces meet — the walls of its pores close on each other, its struts touch, a platen
or an indenter presses on it — and its phases are damaged. This tutorial follows both: a
block with flat pores crushed between two platens until the pores close, the same solid
under an indenter whose force reads its modulus, a periodic cell whose walls touch along
a loading path, then the damage laws at a material point against their closed forms
(Lemaitre's ductile damage, the creep damage of Kachanov and Rabotnov, Mazars' law of
concrete) and a composite whose matrix fails.

The snippets follow each other in one session and take about twenty seconds in all.
Contact
needs surfaces: it runs on tetrahedral meshes, here those of voxel images (six
tetrahedra per voxel), which need no mesher.

## 1. A block with flat pores between two platens

Any segmented image is a sample. Here a block of 4 x 2 x 2 with a flat pore, a slot of 2
x 2 x 0.125 through it, drawn in an array; its solid becomes a mesh.

```python
import numpy as np

from morphogen.contact import compression
from morphogen.homogenization import Material
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.image import VoxelImage

solid = Material(young=1000.0, poisson=0.3)
data = np.ones((8, 4, 16), np.uint8)  # 1: the solid
data[2:6, :, 8] = 0  # the slot: one layer of voxels at mid-height
image = VoxelImage(data, spacing=(0.5, 0.5, 0.125), periodic=False)
block = voxel_tet_mesh(image).subset([1])  # the tetrahedra of the solid alone
print(block.n_elements, block.n_nodes)
```

`compression` puts the block between two rigid platens, the lower one fixed, and brings
the upper one down. The whole surface of the block is free to press on the platens,
with Coulomb's friction, and on itself:

```python
crush = compression(block, {1: solid}, -0.12, friction=0.2, steps=12)
closed = np.flatnonzero(crush.self_area > 0)[0]
print(crush.modulus)  # 673: the slot makes the block soft
print(crush.strain[closed])  # -0.07: the slot closes
print(crush.tangent()[-1])  # 1194: its walls carry the load
print(crush.self_area[-1], crush.area[-1])  # areas in contact: with itself, with the platens
```

Before the slot closes the block bends around it: its apparent modulus is two thirds of
that of its solid. Then its walls meet, the area in contact grows, and the tangent
modulus climbs past that of the solid (1000: the friction of the platens holds its
faces).

```python
from matplotlib.figure import Figure

fig = Figure(figsize=(9.0, 3.6), layout="constrained")
left, right = fig.subplots(1, 2)
left.plot(-crush.strain, -crush.stress, "o-", ms=3)
left.set_xlabel("nominal compression")
left.set_ylabel("nominal stress")
right.plot(-crush.strain, crush.self_area, "o-", ms=3, label="the block with itself")
right.plot(-crush.strain, crush.area[:, 1], "s-", ms=3, label="with the moving platen")
right.set_xlabel("nominal compression")
right.set_ylabel("area in contact")
right.legend()
fig.savefig("contact_crush.png", dpi=110)
crush.problem.write("crushed", crush.result)  # the deformed block and its contact pressure
```

```{image} _static/tutorial/contact_crush.png
:alt: the stress of the block against its compression, and the areas in contact
:class: tutorial-figure
```

Without `self_contact` the walls go through each other and the block stays soft:

```python
through = compression(block, {1: solid}, -0.12, friction=0.2, steps=6, self_contact=False)
print(through.stress[-1] / crush.stress[-1])  # 0.82
```

The same test from the command line, on a mesh file or on a geometry that the mesher
meshes: `morphogen compress block.vtu --material 1=young:1000,poisson:0.3 --strain -0.12
--friction 0.2 --fields crushed`.

The contact is enforced at points inside the faces of the sample, which suits walls that
meet face to face. A sample that meets the platens, or itself, by corners and edges — a
lattice cut through its struts — takes `points="nodes"` (`--points nodes`): the points
are then at the nodes of the faces, each weighted by what its face shows to the other
surface.

## 2. An indenter reads the modulus

A rigid sphere pressed into the upper face of a block of the same solid. Hertz's law
gives the reduced modulus `E / (1 - ν²)` that the force reads on a half-space; a block
is a layer on its base, which stiffens it, and `layer_modulus` takes that into account
(Dimitriadis et al.).

```python
from morphogen.contact import effective_modulus, indentation

whole = voxel_tet_mesh(VoxelImage(np.ones((8, 8, 4), np.uint8), periodic=False)).quadratic()
dent = indentation(whole, {1: solid}, 0.08, indenter="sphere", radius=20.0, steps=3, unload=True)
peak = int(np.argmax(dent.depth))
print(effective_modulus(1000.0, 0.3))  # 1099: E / (1 - ν²)
print(dent.modulus[peak])  # 1426: read as a half-space, too stiff
print(dent.layer_modulus(0.3)[peak])  # 1048: with the base of the block
print(dent.radius[peak], dent.hardness[peak])  # the radius of the contact, the mean pressure
```

The contact (a radius of 1.3) is a third of the thickness of the block: the half-space
is 30 % off, the layer 5 %. A cone (`indenter="cone"`, Sneddon's law) and the slope of
the unloading (`dent.unloading_modulus`, Oliver and Pharr) are there too, and `morphogen
indent` runs the test from the command line.

## 3. The walls of the pores of a periodic cell

Along the loading paths of the homogenization, the option `contact` makes the walls of
the pores of a cell press on each other — and on their images through the faces of the
cell, where a pore crosses them. A periodic cell with a slot, compressed across it:

```python
from morphogen.homogenization import homogenize

cell = np.ones((16, 1, 10), np.uint8)
cell[4:12, :, 4] = 0  # a slot of 2 x 0.25 across the cell
periodic = voxel_tet_mesh(VoxelImage(cell, spacing=(0.25,) * 3, periodic=True)).subset([1])
path = {
    "times": [0, 1],
    "increments": 12,
    "strain": {"33": -0.2, "11": 0.0, "22": 0.0, "23": 0.0, "13": 0.0, "12": 0.0},
}
out = homogenize(
    periodic,
    {1: solid},
    properties=[],
    paths={"press": path},
    nonlinear={"contact": {"friction": 0.3}},
)
press = out["press.periodic"]
print(press.series["contact_area"][[3, -1]])  # 0, then 0.375: three quarters of the slot
print(press.stress[-1, 2])
```

The curves of the path have the columns `contact_area`, `contact_pressure`,
`contact_penetration` and `contact_dissipation`. In a configuration this is
`homogenization: {nonlinear: {contact: {friction: 0.3}}}`: the template
`homogenization_contact` of the [gallery](gallery.md) crushes plates held apart by
pillars that way.

## 4. Damage at a material point

A damage `D` takes away the share `D` of the section that carries the stress. Three
laws, each against its closed form at a material point.

*Lemaitre's ductile damage* goes with a plasticity: the plasticity is that of the
effective stress `σ / (1 - D)`, and the damage grows with the plastic strain, `dD = (Y /
S)^s dp`, `Y` the elastic energy that a crack would release. For a perfectly plastic bar
`Y = R0² / (2 E)` stays constant:

```python
from morphogen.homogenization import simulate_point

steel = Material(
    young=200e3,
    poisson=0.3,
    plasticity={"yield": 400.0},
    damage={"law": "lemaitre", "S": 2.0, "threshold": 0.01},
)
bar = simulate_point(steel, {"times": [0, 1], "increments": 60, "strain": {"11": [0, 0.3]}})
plastic = np.maximum(bar.strain[:, 0] - 400.0 / 200e3, 0.0)
exact = 400.0**2 / (2 * 200e3 * 2.0) * np.maximum(plastic - 0.01, 0.0)
print(np.abs(bar.plastic - exact).max())  # 1e-13: the variable p of a damaged phase is D
print(bar.stress[-1, 0], (1 - exact[-1]) * 400.0)  # the stress: (1 - D) R0
```

*The creep damage of Kachanov and Rabotnov* grows with the time under stress, `dD/dt =
(σ / A)^r (1 - D)^-k`; a viscoplastic bar held at a stress breaks at the time `1 / ((k +
1) (σ / A)^r)`, its creep speeding up as the section is lost:

```python
alloy = Material(
    young=150e3,
    poisson=0.3,
    plasticity={"yield": 0.0, "norton": {"K": 800.0, "n": 5.0}},
    damage={"law": "kachanov", "A": 500.0, "r": 4.0, "k": 6.0},
)
t_r = alloy.damage.rupture_time(200.0)
held = {
    "times": [0, 1e-6 * t_r, 0.9 * t_r],
    "increments": [1, 100],
    "stress": {"11": [0, 200.0, 200.0]},
}
creep = simulate_point(alloy, held)
print(t_r, creep.plastic[-1], 1 - 0.1 ** (1 / 7))  # the damage at 0.9 t_R: 0.29 for 0.28
```

*Mazars' law of concrete* has a damage of tension and one of compression: a strength of
`E κ0` in tension, ten times more in compression, which only extends the point sideways.

```python
concrete = Material(
    young=30e3,
    poisson=0.2,
    damage={
        "law": "mazars",
        "threshold": 1e-4,
        "tension": [1.0, 1.5e4],
        "compression": [1.2, 1500.0],
    },
)
pull = simulate_point(concrete, {"times": [0, 1], "increments": 100, "strain": {"11": [0, 6e-4]}})
push = simulate_point(concrete, {"times": [0, 1], "increments": 100, "strain": {"11": [0, -6e-3]}})
print(pull.stress[:, 0].max(), -push.stress[:, 0].min())  # 3.0 and 34

fig = Figure(figsize=(11.0, 3.4), layout="constrained")
a, b, c = fig.subplots(1, 3)
a.plot(bar.strain[:, 0], bar.stress[:, 0])
a.set_title("Lemaitre: a plastic bar pulled")
a.set_xlabel("strain")
a.set_ylabel("stress")
b.plot(creep.times / t_r, creep.strain[:, 0])
b.set_title("Kachanov: creep to rupture")
b.set_xlabel("time / time to rupture")
b.set_ylabel("strain")
c.plot(pull.strain[:, 0] * 1e3, pull.stress[:, 0], label="tension")
c.plot(-push.strain[:, 0] * 1e3, -push.stress[:, 0], label="compression")
c.set_title("Mazars: concrete")
c.set_xlabel("strain (per thousand)")
c.set_ylabel("stress")
c.legend()
fig.savefig("damage_laws.png", dpi=110)
```

```{image} _static/tutorial/damage_laws.png
:alt: a plastic bar that Lemaitre's law damages, a creep to rupture, concrete in tension and in compression
:class: tutorial-figure
```

The quasi-brittle `damage` of a strength and a fracture energy keeps, with `split:
spectral`, the stiffness of cracks that close; the table of the
[physics guide](physics.md) says which damage suits which material.

## 5. A composite whose matrix fails

In a cell the damage concentrates where the matrix flows most, between the particles.
A law that softens has no width of its own: past the peak the steps of a cell lose
their solution, unless the damage is delayed (`viscosity`, a time).

```python
import morphogen as mg

micro = mg.models.HardcoreSpheres(radius=1.0, volume_fraction=0.2).generate(
    mg.Domain.cube(5.0, periodic=True), seed=7
)
image = micro.voxelize(shape=10)
matrix = Material(
    young=70e3,
    poisson=0.33,
    plasticity={"yield": 100.0, "isotropic": [{"Q": 80.0, "b": 20.0}]},
    damage={"law": "lemaitre", "S": 0.35, "threshold": 0.01, "viscosity": 0.05},
)
particles = Material(young=400e3, poisson=0.2)
pulled = homogenize(
    image,
    {0: matrix, 1: particles},
    properties=[],
    paths={"tension": {"times": [0, 1], "increments": 15, "strain": {"11": [0, 0.06]}}},
)
tension = pulled["tension.periodic"]
print(tension.strength())  # 223: the largest stress of the composite
print(tension.phases[0]["plastic"][-1])  # 0.14: the mean damage of the matrix at the end
```

The stress passes its largest value and falls as the matrix is damaged: the template
`homogenization_ductile_damage` of the gallery is this study at a finer resolution,
with the unloading along the damaged modulus. For a band of damage whose width does not
depend on the voxels, see the micromorphic damage and the phase field of the
[physics guide](physics.md).
