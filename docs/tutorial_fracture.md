# Tutorial: how a microstructure breaks — interfaces, fatigue, twins and cleavage

A microstructure gives way where it is weakest: an interface that debonds, a pore that
concentrates a cyclic stress, a grain that twins or cleaves, a grain boundary. This
tutorial follows each on a small cell, and each time against a number known in advance:
two layers whose interfaces debond (their strength, their fracture energy), a pore under
a cyclic load (the stress it doubles), a crystal of magnesium that twins one way and not
the other, then a textured polycrystal; a crystal of iron that cleaves (the energy that
breaks it), a polycrystal that a crack crosses from cleavage plane to cleavage plane,
and the probability that this happens, by the local approach of Beremin.

The snippets follow each other in one session and take about half a minute in all. They
need no mesher: the cells are voxel images, and the mesh of the layers is that of its
voxels (six tetrahedra each).

## 1. Two layers that debond

Interfaces that come apart are cohesive zones: a traction that grows with the opening
up to a strength, then falls to zero, the area under the curve being the fracture energy
of the interface. `CohesiveCell` puts them between the phases of a periodic mesh and
follows a loading path. Two layers, a soft one and a stiff one, pulled across:

```python
import numpy as np

from morphogen.fracture import Bilinear, CohesiveCell
from morphogen.homogenization import Material
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.image import VoxelImage

data = np.zeros((2, 2, 8), np.uint8)
data[:, :, 4:] = 1  # two layers along the axis 3
layers = voxel_tet_mesh(VoxelImage(data, spacing=(0.25, 0.25, 0.25)))
soft, stiff = Material(young=3000.0, poisson=0.0), Material(young=70000.0, poisson=0.0)
law = Bilinear(
    stiffness=1e6, normal_strength=30.0, shear_strength=30.0, mode_I=0.4, mode_II=0.4
)  # a strength (MPa) and a fracture energy (N/mm)
cell = CohesiveCell(layers, {0: soft, 1: stiff}, law)
pull = cell.run({"times": [0, 1], "increments": 40, "strain": {"33": 0.03}})
stress, strain = pull.stress[:, 2], pull.strain[:, 2]
print(stress.max())  # 30: the strength of the interfaces
print(np.trapezoid(stress, strain) * 2.0)  # 0.8: the work, per unit area of the cell
print(pull.series["debonded_area"][-1], stress[-1])  # 0.5: both interfaces; no stress left
```

The cell of length 2 has two interfaces (the layers alternate): the stress peaks at
their strength, and the work that parts them is twice their fracture energy, 0.8 N/mm.
An interface less tough would break at once — the layers store more elastic energy than
it takes — and the path would stop there, asking for a `viscosity`.

## 2. A pore under a cyclic load

Below the yield stress a cell lasts or not according to the cycle of stress of its worst
point. A criterion of multiaxial fatigue, calibrated on two fatigue limits of the solid,
judges every point of a cell along a cycle of a path:

```python
from morphogen.homogenization import DangVan, homogenize, path_fatigue

n = 16
x = (np.indices((n, n, n)) + 0.5) / n - 0.5
pore = (x**2).sum(axis=0) < 0.2**2  # a spherical pore: 3 % of the cell
block = VoxelImage((~pore).astype(np.uint8), spacing=1.0 / n)
alloy = Material(young=70000.0, poisson=0.33)
cycle = {
    "times": [0, 1, 2, 3],
    "increments": [1, 2, 2],
    "stress": {"33": [0, 40.0, -40.0, 40.0]},
}  # a tension that alternates (MPa)
out = homogenize(
    block, {0: Material(void=True), 1: alloy}, properties=[], paths={"cycle": cycle}, fields="steps"
)
criterion = DangVan.from_limits(bending=110.0, torsion=80.0)
judged = path_fatigue(out["cycle.periodic"], {1: criterion}, cycle=(1, 3))
print(judged.largest, judged.safety)  # 0.70 of the limit: a factor of safety of 1.43
print(judged.fraction())  # 0: no point beyond its limit
```

Without the pore the alloy would be at 0.36 of its limit (40 MPa for 110 MPa in
bending); the pore brings its equator to 0.70, nearly twice (Goodier's sphere: 2.05,
which these voxels round off). `judged.utilization` is the map, `NaN` in the pore.

## 3. Twins: a crystal that flows one way

A twin shears a crystal in one sense only. The {10-12} twin of magnesium lengthens its
axis c by 6.4 % at most: a crystal pulled along c twins at a low stress, and the same
crystal compressed cannot, and must slip on its hard pyramidal systems.

```python
from morphogen.homogenization import simulate_point
from morphogen.homogenization.crystal import twin_systems

magnesium = Material.from_config(
    {
        "hexagonal": [59400.0, 25600.0, 21400.0, 61600.0, 16400.0],  # MPa
        "crystal_plasticity": {
            "c_over_a": 1.624,
            "K": 5.0,
            "n": 10.0,
            "families": {
                "basal": {"tau0": 10.0},
                "prismatic": {"tau0": 60.0},
                "pyramidal2_ca": {"tau0": 100.0},
            },
            "twins": {"twin1012": {"tau0": 25.0, "Q": 15.0, "b": 20.0}},
        },
    }
)
normals, directions, shear = twin_systems("twin1012", 1.624)
print(shear, normals[0, 2] * directions[0, 2])  # 0.129: the shear; 0.499: Schmid, along c
path = {"times": [0, 1], "increments": 30}
pulled = simulate_point(magnesium, {**path, "strain": {"33": 0.03}})
pushed = simulate_point(magnesium, {**path, "strain": {"33": -0.03}})
print(pulled.stress[-1, 2], pulled.series["twin_fraction"][-1])  # 88 MPa, 0.44 twinned
print(pushed.stress[-1, 2], pushed.series["twin_fraction"][-1])  # -231 MPa, no twin
```

The twins start at 25 MPa over their Schmid factor, 50 MPa, and by 3 % have consumed
0.44 of the crystal: their strain is the fraction times the shear times the Schmid
factor. Pushed back, the crystal untwins before anything else: its twins shear back at
about the stress at which they formed, and when none is left it is the crystal that it
was. And a twinned crystal is another crystal: with `twin_reorientation`, a point whose
twinned fraction comes to that threshold takes the lattice of its twin — 86° off, for
this twin — which slips on other systems, and twins back when the load is reversed:

```python
cycle = {"times": [0, 1, 2], "increments": 30, "strain": {"33": [0.03, -0.005]}}
back = simulate_point(magnesium, cycle)
print(back.series["twin_fraction"][[30, 45, -1]])  # 0.44, 0.22, 0: the twins shear back
print(back.stress[[45, -1], 2])  # at -73 MPa; then -231, the slip of the crystal as it was
spec = magnesium.to_config()
spec["crystal_plasticity"]["twin_reorientation"] = 0.3
turned = simulate_point(Material.from_config(spec), cycle)
print(turned.series["twin_misorientation"][[30, -1]])  # 86.3 degrees, then 0: there and back
print(turned.stress[[30, 45], 2])  # 147 MPa: the twin is harder to pull; -79: it twins back
```

In a polycrystal, `twin_reorientation: {threshold: 0.2}` turns the lattices grain by
grain rather than point by point — the most twinned points first, while the turned volume
of the grain (the field `twin_reoriented`) is below its twinned volume (`twin_fraction`)
— and `twin_detwinning: 0.5` lets the twins shear back at half of the resistance of the
twinning: they have no twin to nucleate on the way back.

A rolled sheet or an extruded bar has its axes c near one direction; a polycrystal with
that texture keeps the asymmetry of its grains:

```python
from morphogen.domain import Domain
from morphogen.homogenization import NonlinearFFTProblem
from morphogen.models.base import model_from_config

model = model_from_config(
    {
        "type": "voronoi",
        "n_cells": 8,
        "crystal_orientations": True,
        "texture": {"type": "von_mises_fisher", "kappa": 30.0, "axis": [0, 0, 1]},
    }
)
grains = model.generate(Domain(size=1.0, periodic=True), 5)
image = grains.voxelize(shape=8)
sheet = {}
for name, strain in (("tension", 0.02), ("compression", -0.02)):
    problem = NonlinearFFTProblem(
        image,
        {1: magnesium},
        grains=grains.labels(grid=image.grid()),
        orientations=grains.orientations,
    )
    sheet[name] = problem.run({"times": [0, 1], "increments": 10, "strain": {"33": strain}})
print(sheet["tension"].stress[-1, 2], sheet["tension"].series["twin_fraction"][-1])  # 77, 0.20
print(sheet["compression"].stress[-1, 2])  # -153 MPa: twice as hard
```

```python
from matplotlib.figure import Figure

fig = Figure(figsize=(9.0, 3.6), layout="constrained")
left, right = fig.subplots(1, 2)
for name, colour in (("tension", "#2a78d6"), ("compression", "#9aa1ab")):
    curve = sheet[name]
    left.plot(
        np.abs(curve.strain[:, 2]), np.abs(curve.stress[:, 2]), "-o", ms=3, color=colour, label=name
    )
    right.plot(np.abs(curve.strain[:, 2]), curve.series["twin_fraction"], "-o", ms=3, color=colour)
left.set(xlabel="|strain 33|", ylabel="|stress 33| (MPa)")
right.set(xlabel="|strain 33|", ylabel="twinned fraction")
left.locator_params(axis="x", nbins=5)
right.locator_params(axis="x", nbins=5)
left.legend(frameon=False)
fig.savefig("fracture_twins.png", dpi=110)
```

```{image} _static/tutorial/fracture_twins.png
:alt: the stress and the twinned fraction of a textured magnesium, pulled and compressed
:class: tutorial-figure
```

## 4. Cleavage: a plane that opens

A crystal cleaves on a crystallographic plane when its normal stress reaches a
strength; the plane then opens and its strength falls. In `crystal_plasticity`, the
`cleavage` planes are systems of the crystal beside its slip systems: an opening, and
slips of the plane that release its shear. A crystal of iron pulled along [001], with
{100} planes of 500 MPa that soften as `exp(-100 opening)`:

```python
iron = {"cubic": [230000.0, 135000.0, 117000.0]}  # MPa
planes = {"planes": "cube", "strength": 500.0, "softening": 100.0, "K": 0.5, "n": 1.0}
crystal = Material.from_config({**iron, "crystal_plasticity": {"cleavage": planes}})
cleaved = simulate_point(crystal, {"times": [0, 1], "increments": 150, "strain": {"33": 0.05}})
opening = cleaved.series["cleavage_opening"]
exact = 500.0 * np.exp(-100.0 * opening) + 0.5 * np.gradient(opening, cleaved.times)
print(np.abs(cleaved.stress[20:, 2] - exact[20:]).max())  # 3e-4 MPa: the law of the plane
print(np.trapezoid(cleaved.stress[:, 2], cleaved.strain[:, 2]))  # 4.97: strength / softening
```

The energy that breaks the plane is its strength over its softening, 5 MPa: an energy
per unit volume. Pushed back, the plane closes: its opening goes back to nothing, and
the crystal carries the pressure as if it had never opened (the unilateral condition of
a crack: `closure`, the default; `friction` makes its lips rub):

```python
there = {"times": [0, 1, 2], "increments": [150, 60], "strain": {"33": [0.05, -0.002]}}
closed = simulate_point(crystal, there)
print(closed.series["cleavage_opening"][[150, -1]])  # 0.05, then 0: the plane has closed
print(closed.stress[-1, 2] / closed.strain[-1, 2])  # 130137 MPa: the modulus of the crystal
```

In a cell the opening localizes in one layer of voxels, and that
energy times the size of the voxels is the fracture energy of the plane: give it
(`energy`) with the size of the voxels (`length`), and the softening follows. A
polycrystal of iron at a low temperature, whose slip is harder than its planes:

```python
size, shape, energy = 0.1, 8, 0.05  # mm, voxels, N/mm
steel = Material.from_config(
    {
        **iron,
        "crystal_plasticity": {
            "K": 10.0,
            "n": 10.0,
            "families": {"bcc110": {"tau0": 300.0, "Q": 100.0, "b": 20.0}},
            "cleavage": {
                "planes": "cube",
                "strength": 500.0,
                "energy": energy,
                "length": size / shape,
                "K": 50.0,
                "n": 1.0,
            },
        },
    }
)
model = model_from_config({"type": "voronoi", "n_cells": 8, "crystal_orientations": True})
poly = model.generate(Domain(size=size, periodic=True), 2)
voxels = poly.voxelize(shape=shape)
labels = poly.labels(grid=voxels.grid())
problem = NonlinearFFTProblem(voxels, {1: steel}, grains=labels, orientations=poly.orientations)
broken = problem.run({"times": [0, 1], "increments": 30, "strain": {"33": 0.008}}, fields=True)
damage = broken.fields[1.0]["cleavage_damage"]
print(broken.stress[:, 2].max(), broken.stress[-1, 2])  # 592 MPa, then 44: a crack
print(int((damage > 0.9).sum()), shape**2)  # 64 voxels broken: one section of the cell
print(np.trapezoid(broken.stress[:, 2], broken.strain[:, 2]) * size / energy)  # 2.6
```

The polycrystal is elastic up to the strength of its best oriented planes over the
`cos²` of their angle to the load, then a crack crosses its grains: a section of the
cell is broken, and the work is 2.6 times the fracture energy times that section — the
planes of the grains are inclined and do not meet, and what bridges them still carries
44 MPa. With a softer slip (`tau0: 150`) the grains would flow before any plane opens:
slip shields cleavage — unless it nucleates it: with `coupling: 5` among the parameters
of the planes, their strength falls as `exp(−5 γ)` of the cumulated slip `γ` of the
crystal, and the polycrystal breaks once its grains have slipped enough. The crack is
one voxel wide, and its energy that of voxels of this size; with `regularization:
{width: ...}` in place of `length`, the planes are regularized by their microdamage and
break across a band of that width, whatever the grid
(`homogenization_cleavage_band`). And with
cohesive grain boundaries weaker than the planes
(`CohesiveCell(mesh, materials, law, faces="grains")` on a mesh of the grains), the
crack would follow the boundaries instead.

## 5. How likely is cleavage? The local approach

Before any crack, the question is whether one starts: at a microcrack that the plastic
strain has nucleated, the weakest of the cell. Beremin's criterion reads its
probability from the fields of a path: the Weibull stress of the voxels that have
yielded, each driven here by the normal stress of the {100} planes of its grain.

```python
from morphogen.homogenization import Beremin, path_beremin

ferrite = Material.from_config(
    {
        **iron,
        "crystal_plasticity": {
            "families": ["bcc110", "bcc112"],
            "tau0": 250.0,
            "Q": 80.0,
            "b": 15.0,
            "K": 10.0,
            "n": 10.0,
        },
    }
)
problem = NonlinearFFTProblem(voxels, {1: ferrite}, grains=labels, orientations=poly.orientations)
curve = problem.run({"times": [0, 1], "increments": 15, "strain": {"33": 0.03}}, fields="steps")
law = Beremin(
    modulus=22.0,
    scale=1550.0,
    volume=1.25e-10,  # m, sigma_u (MPa), V0 (mm³)
    planes="cube",
    plastic=1e-4,
)
euler = poly.orientations[np.asarray(labels.data)]  # the Bunge angles of each voxel
found = path_beremin(curve, law, orientations=euler)
print(found.weibull[[2, 7, 14]])  # 1185, 1459, 1683 MPa at 0.6, 1.6 and 3 %
print(found.probability[[2, 7, 14]])  # 0.003, 0.23, 0.998
```

The reference volume is that of a few grains, (50 µm)³, divided here by the 10⁶ cells
of the gauge of a tensile bar that the cell stands for. The Weibull stress is twice
the tensile stress (805 MPa at 3 %): the eight million reference volumes of the gauge,
and the stresses of the grains above their mean. The probability rises from nothing at
the yield stress to a certainty at 3 %.

```python
fig = Figure(figsize=(11.0, 3.4), layout="constrained")
section, middle, right = fig.subplots(1, 3, width_ratios=(0.8, 1.0, 1.0))
section.imshow(damage[:, shape // 2, :].T, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
section.set(title="cleavage damage", xticks=[], yticks=[])
middle.plot(
    broken.strain[:, 2],
    broken.stress[:, 2],
    "-o",
    ms=3,
    color="#eb6834",
    label="cleaves (slip of 300 MPa)",
)
middle.plot(
    curve.strain[:, 2],
    curve.stress[:, 2],
    "-o",
    ms=3,
    color="#2a78d6",
    label="flows (slip of 250 MPa)",
)
middle.set(xlabel="strain 33", ylabel="stress 33 (MPa)")
middle.legend(frameon=False)
right.plot(curve.strain[1:, 2], found.probability, "-o", ms=3, color="#2a78d6")
right.set(xlabel="strain 33", ylabel="probability of cleavage (Beremin)")
fig.savefig("fracture_cleavage.png", dpi=110)
```

```{image} _static/tutorial/fracture_cleavage.png
:alt: the damage of the cleavage planes on a section of a polycrystal, its stress, and the probability of cleavage of a ferritic polycrystal
:class: tutorial-figure
```

## Where to go from here

Each part has a configuration to start from in the [gallery](gallery.md):
`homogenization_debonding` (beads that debond, with friction), `homogenization_fatigue`
(pores in a cast alloy), `homogenization_twinning`, `homogenization_detwinning` (twins
there and back, lattices that turn), `homogenization_cleavage`,
`homogenization_slip_cleavage` (cleavage that the slip nucleates),
`homogenization_cleavage_band` (cleavage across a band of a given width),
`homogenization_crack_closure` (a crack that closes),
`homogenization_grain_fracture` (cracks between the grains or through them) and
`homogenization_beremin`. The formulas and what each model leaves out are in the
[theory](theory.md), §84 to §89; the graphical interface has the interfaces, the fatigue,
the twins and the cleavage planes in its tool *Loading path of a morphology*.
