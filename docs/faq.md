# Questions and troubles

Answers to what comes up first. The [recipes](howto.md) answer "how do I..." with a
snippet; this page is about installing, choosing and understanding what a run says, and
the [glossary](glossary.md) says what the words mean here.

## Installing and running

**Which Python, which systems?** Python 3.12 or newer on Windows, macOS (Apple Silicon)
and Linux. Every dependency is a binary wheel: no compiler is needed.
[Installation](installation.md) gives the commands for uv, pip and conda.

**What do the extras add?** The core (NumPy, SciPy, Numba, PyYAML) generates, measures,
homogenizes by FFT and runs the graphical interface. `surface` adds the STL surfaces
(scikit-image), `io` TIFF and other mesh formats, `plot` the figures (matplotlib),
`mesh` the conforming meshes and so the finite elements (the mesher MMG), `video` the
videos and 3D renderings (VTK, ffmpeg), `gpu` the FFT solvers on NVIDIA GPUs. `[all]` is
`surface`, `io`, `plot` and `mesh`.

**The first call of a function takes seconds, even a minute.** Its kernels are compiled
at their first call (Numba) and cached: the second call, and every later session, starts
at once. A new version of morphoGen or of Numba compiles them again.

**`morphogen info` says no mesher is found.** The extra `mesh` installs MMG from the
`pymmg` wheels (Windows, macOS, Linux x86-64). Elsewhere (Linux on ARM), build `mmg3d`
from <https://github.com/MmgTools/mmg> and set the environment variable
`MORPHOGEN_MMG3D` to its path. Without a mesher everything that runs on voxels still
works: the FFT solver, the extended finite elements, the phase field.

**How many threads does it use?** All the cores by default; `--threads N` on the command
line, or the environment variable `NUMBA_NUM_THREADS`, sets their number. A run with `-j
8` shares them between its 8 processes. `morphogen info` shows the number in use.

**Does the graphical interface send my data anywhere?** No. `morphogen gui` starts a
server of the Python standard library on `127.0.0.1` (this machine only); the page loads
no outside resource and works offline.

**Videos or 3D figures fail on a server without a screen.** They are rendered off-screen
by VTK, which needs a software or hardware OpenGL; where there is none, the figures of
sections and the curves are still written, and the interface draws its 3D views in the
browser instead. `MORPHOGEN_FFMPEG` names an ffmpeg to use when none is on the path.

## Models and geometry

**Which units?** None are imposed: lengths are in the unit of the domain (`unit` is a
label for the files), stresses in the unit of the moduli you give, and so on. A
toughness is then in units of stress times length: with micrometres and MPa, J/m².

**Which phase is which?** Phase 0 is what surrounds (the matrix, the pores of a lattice),
phase 1 what the model places (grains, fibres, the solid of a lattice or a foam); models
of more phases number them further (a coating, inclusions, the rings of an osteon).
`model.active_phases()` names them, and the cards of the [gallery](gallery.md) show
them in their colours.

**Periodic or not?** A periodic domain (`periodic: true`) wraps the grains across its
faces: the realization is a cell of an infinite medium, the right sample for the
periodic conditions of the FFT solver and for meshes with matching faces. The FFT solver
treats any image as such a cell; a non-periodic realization then has faces that do not
match, and is better solved on a conforming mesh under kinematic (`kubc`) or static
(`subc`) conditions. For the same volume, periodic conditions come closest to the
effective property.

**Is a realization the same on another machine?** It is a function of the model, the
domain and the seed, and the geometry saved (`outputs: {geometry: [json]}`) is read back
exactly anywhere. `run.json` keeps the configuration, the seed and the versions.

**How fine must the voxels be?** Fine enough for the smallest feature that matters: a
radius of 2 voxels gives the fractions, not the stresses between particles. The
[tutorial](tutorial.md) shows a plastic composite whose stress falls from 2 to 1.4 times
that of its matrix between 1.6 and 5 voxels per radius. Halve the voxels and compare, or
use a conforming mesh or the extended finite elements, which do not draw the interfaces
as stairs ([guide to the physics](physics.md), §8).

## Solvers

**FFT, finite elements or extended finite elements?** See the
[recipe](howto.md#choose-between-voxels-a-conforming-mesh-and-extended-finite-elements)
and the table of [what runs where](physics.md#11-what-runs-where).

**Four-node or ten-node elements?** Ten nodes for an analysis. The 4-node tetrahedra are
too stiff, and lock when a phase flows plastically or is nearly incompressible; the
10-node ones bend along the interfaces (their nodes lie on the implicit surfaces of the
phases), and give a better answer with elements twice as large. A configuration takes
them by default for the meshes it analyses; `mesh: {order: 1}` gives the 4-node ones,
for a quick look. A mesh of 10-node elements has about eight times the nodes of the
same mesh of 4-node ones: double `mesh.size`, and keep `mesh.background` fine where
there are thin gaps or necks, which the grid of the mesher must resolve
([theory notes](theory.md), §76). The cohesive zones and the cracks of conforming meshes
take them as well (`insert_interfaces` on a 10-node mesh, `conforming_work(...,
order=2)`); the Bloch waves, the bifurcations and the micromorphic media take them with
straight edges; the XFEM, the mixed-mode cracks and the inelastic bulk of the crack
problems keep 4-node elements.

**Does the mesh follow the sharp edges of my phases?** For the structures made of
spheres and segments — Boolean models, fibres, lattices of struts, their coatings and
hollow members —, yes: where two primitives overlap, their surfaces meet along a sharp
edge (a groove), and the mesh has edges along it (`mesh.sharp`), with the nodes of its
10-node elements on both surfaces ([theory notes](theory.md), §80). It costs a cut of
the box per group of primitives that do not overlap: such meshes take 40 to 80 % longer;
`mesh: {sharp_edges: false}` rounds the edges over an element as before. The meshes of
4-node elements, polyhedra inscribed in the phases, lose more volume than before at the
same sizes, the cuts being smoother: give them a smaller `mesh.hausdorff`. The other
structures (their functions are those of a grid), the rims of cylinders with flat caps
and the overlaps shallower than half of `mesh.background` are rounded; the grains of
tessellations have their exact edges, as they always had.

**A rubber is too stiff on a mesh, or a run warns of a nearly incompressible phase.**
With a Poisson's ratio near 1/2, elements whose strain is integrated at every point
must keep their volume at each of them, and lock. Take `integration: bbar` (in the
`homogenization` section; `homogenize(..., integration="bbar")`, `--integration bbar`):
the volume change of each 10-node tetrahedron is then its mean, and the rest of its
strain is integrated as before ([theory notes](theory.md), §77). The voxels of the
loading paths do the same by default. Keep the default, `full`, for compressible
phases.

**When is the isogeometric analysis the one to take?** When the geometry is a few
smooth curves and their exact shape matters: a cell with holes or inclusions, an
architectured sheet, a part whose boundary is to be optimized. NURBS patches hold
circles and ellipses exactly, a few hundred unknowns give what a conforming mesh gives
with a hundred times more, and the control points are the variables of a shape
optimization with exact derivatives ([tutorial](tutorial_iga.md); `morphogen.iga`). It
takes patches that the user assembles (or the shapes of `iga.shapes`), linear elasticity
and conduction, in the plane and in space. For a random morphology, or a nonlinear
behaviour, the voxels and the meshes of the other solvers are the tools; the image of
an optimized shape (`Multipatch.voxelize`) goes to them like any other — the tutorial
compares the two on an auxetic sheet.

**"FFT solver stopped at relative residual ..."** The conjugate gradients reached their
number of iterations before the tolerance: usual with voids or a very high contrast, or
with broken voxels. The result is that of the residual printed. Raise
`homogenization.max_iterations`, give the voids a `void` material rather than a tiny
modulus, or loosen `homogenization.tolerance` if that residual is enough.

**"nonlinear solution: no convergence at time t after dividing the step N times"** A
step of a loading path that Newton's method cannot solve even in small pieces. Often a
behaviour without a solution there (a softening without a length, a law that locks at
the stretches reached near a crack tip: prefer a polynomial energy), or a load beyond
what the cell can carry in stress control. More increments, `nonlinear: {max_cuts: 14}`,
or the strain component prescribed instead of the stress, help otherwise. A damage law
that softens (`damage`, alone or beside a `plasticity`) takes a `viscosity`, the time of
a delayed damage: a tenth to a twentieth of the duration of the loading keeps the steps
well posed past the peak.

**Which damage for which material?** Quasi-brittle phases that crack diffusely: `damage`
with a strength and a fracture energy (`split: spectral` if they are also compressed:
closed cracks carry the compression); concrete in tension and in compression: `damage:
{law: mazars}`. Metals that fail after plastic flow, or after a few cycles: a
`plasticity` with `damage: {law: lemaitre}` (the damage grows with the plastic strain,
faster under a mean tension), or `porous_plasticity` when the growth of voids is the
mechanism (Gurson, Rousselier). Creep to rupture: a viscoplastic `plasticity` with
`damage: {law: kachanov}`. A crack with a width of its own, independent of the elements:
the micromorphic damage, or the phase field (`fracture`). An interface that comes
apart: cohesive zones — along the loading paths of a cell, `nonlinear: {interfaces:
{normal_strength: ..., mode_I: ...}}` on the conforming mesh of its phases (the template
`homogenization_debonding`). A polycrystal that breaks: between its grains with
`interfaces: {faces: grains}`, through them with the `cleavage` planes of its
`crystal_plasticity` (a strength, a fracture energy and the size of the elements), or
both, the crack taking the weaker path (the templates `homogenization_cleavage` and
`homogenization_grain_fracture`). How likely a brittle steel is to cleave at a strain,
before any crack: the Weibull stress of Beremin on the fields of its path,
`homogenization: {beremin: {modulus: ..., scale: ..., volume: ...}}` (the template
`homogenization_beremin`).

**Why does my hexagonal polycrystal not flow at the same stress in tension and in
compression?** Its twins: a twin shears in one sense only ({10-12} lengthens the axis c
of magnesium and titanium, shortens that of zinc), so that a textured polycrystal twins
under one sign of the load and must slip on its hard systems under the other (the
template `homogenization_twinning`; `twins` and `texture`).

**My crystal yields below its cleavage strength: will it never cleave?** Not with a
constant strength: the slip shields the cleavage. In body-centred cubic and hexagonal
metals the slip nucleates it: `cleavage: {coupling: 8}` lowers the strength of the
planes as `exp(−8 γ)` of the cumulated slip `γ` of the crystal (Aslan et al. 2011), so
that a plane under the normal stress `σ` opens once the crystal has slipped by `ln(Y₀ /
σ) / 8`; the damage of the planes eases the slip in return.

**My cleavage cracks are one voxel wide and follow the grid: can they have a width of
their own?** Yes: give the planes the width of their damage band in place of the size of
the elements, `cleavage: {energy: 0.05, regularization: {width: 0.015}}`. Their strength
then falls with a field that follows their damage over a length, the microdamage (Aslan
& Forest 2009), and a plane breaks across a band of that width — some elements: eight
across it for a work within 1 % of the fracture energy on voxels —, the same on a finer
grid. The path solves the equilibrium of each step several times, as the microdamage
settles; a crack that runs across the cell takes some equilibria per voxel crossed
(template `homogenization_cleavage_band`). The scheme is accelerated by default;
`nonlinear: {microdamage: {tolerance: 1e-4}}` settles the microdamage ten times closer,
`passes` bounds the equilibria of a step, and `acceleration: false` gives the plain
scheme.

**What happens to twins and cleavage cracks when the load is reversed?** The twins shear
back, down to none, at about the stress at which they formed, and the cleavage planes
close: a closed crack carries the pressure as the sound crystal does, and its lips rub
(`cleavage: {friction: 0.3}`; they slide freely by default). Both are the defaults
(`twin_detwinning: false` and `cleavage: {closure: false}` keep the twins and the
openings; a number is the share of their resistance at which they go back: twins have
no twin to nucleate on the way back). A twinned crystal is also another crystal:
`twin_reorientation: 0.5` turns the lattice of a point into that of its predominant twin
once half of it has twinned — 86° off for the {10-12} twin of magnesium, the field
`twin_misorientation` — so that it slips and twins as the twin does; `twin_reorientation:
{threshold: 0.2}` turns the lattices grain by grain, the most twinned points first, so
that the turned volume of each grain (`twin_reoriented`) is its twinned volume
(`twin_fraction`) (the templates `homogenization_detwinning` and
`homogenization_crack_closure`).

**Will my microstructure last under cyclic loading?** Below the yield stress, ask a
criterion of multiaxial fatigue: `homogenization: {fatigue: {criterion: dang_van,
bending: ..., torsion: ...}}` judges the stresses of each point along a cycle of a path
(the two fatigue limits are those of the phase, fully reversed) and gives the map of the
utilization and the factor of safety of the cell (the template
`homogenization_fatigue`). Crossland and Dang Van take a fraction of a second; the
critical planes of Findley and Matake search some hundreds of planes per point. A cell
that yields at each cycle is another matter: follow the cycles with a `plasticity` and
its damage (`law: lemaitre`).

**My interfaces break and the path stops ("give the cell a viscosity").** An interface
that is brittle against the elastic energy around it breaks at once: the cell has no
equilibrium just after, under prescribed strains (a snap-back). `interfaces: {viscosity:
T}` slows the damage over a time `T` of the path (a twentieth to a fiftieth of the
duration of the loading): the steps pass, and the response tends to that of the law as
`T` vanishes.

**The walls of my pores go through each other.** A loading path does not know that two
surfaces met unless told: on the conforming mesh of the solid, `nonlinear: {contact:
{friction: 0.3}}` makes the walls of the pores press on each other and rub (the template
`homogenization_contact`); for a sample that is not periodic, `morphogen compress`
crushes it between two platens. The FFT solver has no surfaces: its voids just close.

**A corner goes through the face that it lands on.** The contact is enforced at points
inside the faces of the slave surface: its corners and its edges carry none, and a
corner enters a face until the points next to it press. `points: nodes` (of `contact` in
a configuration; `points="nodes"` of a `Pair`, an `Obstacle`, `ContactCell`,
`compression`; `morphogen compress --points nodes`) puts the points at the nodes of the
faces: the struts of a lattice cut by the faces of its sample, the corners of grains
that meet. The rule is of lower order (the pressures of a flat contact between meshes
that do not match are less even), which is why it is not the default.

**The phase field spends minutes on one step.** The step where a crack runs across the
cell: each alternate minimization moves the crack by about a voxel, so that a crack of
`n` voxels needs about `n` solves. It is the price of an unstable propagation; the steps
before and after are quick. A planar section (`voxels.section`), a coarser image, or a
path that stops soon after the peak, shorten it.

**My apparent modulus depends on the boundary conditions.** As it must on a volume
smaller than the representative one: kinematic conditions bound it from above, static
ones from below, periodic ones lie between. Their gap closes as the volume grows; the
tutorial [how large is a representative volume?](tutorial_rve.md) and the
[RVE studies](gallery.md#rve-studies) of the gallery measure the size needed.

**How large a problem fits in memory?** The linear FFT solver holds a few fields of 3 or
6 components per voxel: 256³ voxels fit in a few gigabytes. The nonlinear paths keep
eight integration points per voxel with their strains, stresses, tangents and internal
variables, about 8 kB per voxel: 64³ voxels take 2 GB, 128³ sixteen. The interface
refuses a run that would need more than half of the free memory.

## Files and results

**Where are the results of a run?** In `outputs.directory`: `results.csv` (a row per
realization: the measures and the properties), `homogenization.csv` (a row per solve:
iterations, residual, time), `run.json` (the configuration, the seed, the theory of the
model, the versions), the files of each realization (`<name>_<index>...`), and the
report of an RVE study (`rve_report.md`, with its figures).

**How do I look at a `.vti` or a `.vtu`?** In ParaView, or `morphogen view FILE` for a
PNG of its sections; in the interface, **View in 3D** on any item.

**How do I run a result again?** `morphogen run` on the configuration kept in the folder
of the result gives the same files; the interface writes the configuration of each of
its simulations (Export, *configuration*).

**How do I cite morphoGen?** See [Citing](citing.md).
