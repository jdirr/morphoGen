# Fracture, contact and fatigue

Microstructures break along loading paths, up to fracture. Cracks nucleate and merge in
voxel images (the phase field, in small or finite strains), or grow in meshes (stress
intensity factors, cohesive zones), through polycrystals and any morphology. Surfaces
touch and rub (contact, self-contact and Coulomb's friction): the walls of the pores of a
cell that densifies, a sample between platens or under an indenter. The interfaces of
the phases of a cell debond along a path (cohesive zones on its torus). The stresses of
a cycle are judged by criteria of multiaxial fatigue (Crossland, Sines, Dang Van,
Findley, Matake): where a cell would start to crack, and its factor of safety. The
[tutorial on fracture](tutorial_fracture.md) and the
[tutorial on contact and damage](tutorial_contact.md) follow these steps on small
problems; the damage of the phases themselves (crack bands, Lemaitre, Mazars) and the
micromorphic damage are behaviours ([Behaviours along loading paths](behaviours.md)).

## Phase field of fracture

The phase field of brittle fracture runs on voxel images (`homogenize(...,
fracture=...)`: AT1 and AT2, a toughness per phase; in finite strains with
`nonlinear: {kinematics: finite}`, for hyperelastic phases), beside the cracks, cohesive
zones and ductile damage of `morphogen.fracture`. In a configuration,
`homogenization.fracture` (`{toughness: ..., length: ...}`) makes the loading paths
break the voxels by the phase-field method; the
[tutorial on a rubber](tutorial_rubber.md) tears a cracked strip at the stretch of Rivlin
and Thomas, and the [guide to the physics](physics.md#7-fracture) compares the routes.

## Cohesive zones and cracks

`morphogen.fracture` opens the interfaces of a tetrahedral mesh (between phases, or
along chosen faces) and gives them cohesive zone elements, in small or finite strains:

```python
from morphogen.fracture import Bilinear, CohesiveProblem, insert_interfaces

opened, interfaces = insert_interfaces(mesh, "phases")  # debonding of the particles
law = Bilinear(
    stiffness=1e5,
    normal_strength=30.0,
    shear_strength=40.0,
    mode_I=0.3,
    mode_II=0.6,
    eta=2.0,  # Benzeggagh-Kenane mixity
)
problem = CohesiveProblem(opened, interfaces, materials, law, kinematics="finite")
result = problem.run(
    [(bottom, [0, 1, 2], 0.0), (top, 2, 1.0)], load=0.05, steps=50, viscosity=1e-4, fields=True
)
problem.write("debonding", result)  # the bulk and the interfaces (damage), VTU
```

Laws: bilinear mixed-mode and the exponential law of Ortiz & Pandolfi; helpers: the
length of the cohesive zone, the strength of coarse meshes (Turon et al.), the double
cantilever beam (`dcb_specimen`) and its corrected beam theory (`dcb_beam_theory`).

Cracks (`morphogen.fracture.lefm`): planar cracks by level sets, their energy release
rate on conforming meshes remeshed at each step (MMG, as Z-cracks) or by XFEM (jump and
branch enrichments of linear tetrahedra), and their growth by Paris' law or Griffith:

```python
from morphogen.fracture import Crack, propagate, xfem_work

crack = Crack("penny", centre, normal=(0, 0, 1), size=2.0)
growth = propagate(
    xfem_work(mesh, material, forces, supports),
    crack,
    young=E,
    poisson=nu,
    increment=0.5,
    steps=10,
    paris=(C, m),
)
```

Mixed modes, in small or finite strains (`morphogen.fracture.mixed`): the energy release
rate along the front by the G-θ method (exact for the neo-Hookean body of finite strains),
the factors of modes I, II and III by interaction integrals, the frictionless contact of
the lips, and kinked cracks remeshed by MMG at each step, growing in the direction of
Erdogan & Sih:

```python
from morphogen.fracture import KinkedCrack, crack_mesh, propagate_kinked, solve_crack

solution = solve_crack(
    mesh, crack, material, supports, forces=loads, kinematics="finite", contact=True
)
modes = solution.modes(stations=5)  # K_I, K_II, K_III, G, G_I, G_II, G_III along the front
crack = KinkedCrack(((0.0, 16.0, 0.0), (4.5, 16.0, 0.0)))  # mouth to front, front along z
growth = propagate_kinked(
    lambda c: crack_mesh(lower, upper, c, size=0.5),
    crack,
    material,
    supports,
    forces=loads,
    increment=1.5,
    steps=10,
    paris=(C, m),
)
```

Cracks of any surface and front (`morphogen.fracture.surface`): the surface swept by the
successive positions of a front, closed (a crack inside the body) or open (through, edge
and surface cracks), each point of which advances by its own amount and kinks by its own
angle; the mesh is refined around the front alone:

```python
from morphogen.fracture import SurfaceCrack, crack_mesh, propagate_front

crack = SurfaceCrack.ellipse(centre, normal=(0.5, 0, 0.87), a=1.0, b=0.6)  # inclined
mesh_of = lambda c: crack_mesh(lower, upper, c, size=0.6, front_size=0.08)
growth = propagate_front(
    mesh_of,
    crack,
    material,
    supports,
    forces=loads,
    increment=0.3,
    steps=5,
    stations=12,
    paris=(C, m),
)
growth.rows()  # per step: area, K_I, K_II, K_III along the front, kink angles, cycles
growth.cracks[-1].points  # rows x stations x 3: the surface, its last row the front
```

Inelastic solids (`morphogen.fracture.bulk`): the crack solvers take the behaviours of
morphoGen's points as their bulk (von Mises elasto-viscoplasticity, with Hencky's
logarithmic strains in finite strains; crystal plasticity with the orientations of the
grains; MFront laws) and give the driving forces of dissipative solids:

```python
steel = Material(
    young=210e3,
    poisson=0.3,
    plasticity={"yield": 300.0, "hardening": 1e3, "norton": {"K": 50.0, "n": 5.0}},
)
solution = solve_crack(mesh, crack, steel, grips, kinematics="finite", steps=20)
solution.release_rate(energy="work")  # the J of the deformation theory (Begley-Landes)
solution.release_rate(energy="tip")  # the driving force at the tip (configurational forces)
# the grain boundaries
opened, boundaries = insert_interfaces(
    polycrystal, faces=lambda c: c[:, 0] > -1e9, sides=polycrystal.cell
)
problem = CohesiveProblem(opened, boundaries, {0: crystal}, law, kinematics="finite")
result = problem.run(grips, 0.02, steps=40, duration=1.0)  # result.work: loads, bulk, interfaces
```

Ductile growth on meshes rebuilt at each advance (`morphogen.fracture.ductile`): the state
of the points (internal variables, dissipated energy, stresses) and the displacement move
from mesh to mesh, the lips apart, the body is brought back to equilibrium, and the crack
advances when its driving force at the tip reaches a toughness, or the porosity of its GTN
points a critical value; the far J at each advance draws the resistance curve:

```python
from morphogen.fracture import KinkedCrack, crack_mesh, propagate_ductile, solve_crack

growth = propagate_ductile(
    lambda c: crack_mesh(lower, upper, c, size=1.0),
    crack,
    steel,
    grips,
    load_step=0.25,
    increment=1.5,
    steps=5,
    toughness=30.0,
    kinematics="finite",
    transfer="nearest",
)
advance, resistance = growth.resistance()  # the R-curve (far J against the advance)
moved = solve_crack(new_mesh, longer_crack, steel, grips, 1.2, initial=solution)  # one transfer
```

A polycrystal grows its crack through its grains: the crack meshes carry the grains of a
tessellation and their orientations, and the state of each grain moves within the grain
from one mesh to the next:

```python
mesh_of = lambda c: crack_mesh(lower, upper, c, size=0.4, grains=tessellation)
growth = propagate_ductile(
    mesh_of,
    KinkedCrack(((0, 4, 0), (2.5, 4, 0))),
    crystal,
    grips,
    load_step=1e-3,
    increment=1.0,
    steps=3,
    toughness=0.6,
)
```

Any morphology can be cracked: the crack meshes carry its phases (the materials, in
`mesh.phase`) and the sides of the crack apart (`mesh.side`), as staircases or conforming
to the level set of a two-phase morphology (MMG cuts the grid along it once, then each
crack cuts that mesh and remeshes it), its void phases left out. Stiff inclusions deflect
the crack, soft ones and pores attract it, and a front reaching a pore runs through it;
the state of inelastic points moves material by material:

```python
from morphogen.fracture import phase_mesh

phases = phase_mesh(micro, lower, upper, 0.5)  # cut along its level set, once
mesh_of = lambda c: crack_mesh(lower, upper, c, size=0.5, morphology=phases, voids=[1])
growth = propagate_kinked(
    mesh_of,
    crack,
    {0: matrix, 1: Material(void=True)},
    held,
    forces=tension,
    increment=0.75,
    steps=4,
)
```

The phases and their interfaces have toughnesses of their own: each advance is the
candidate of the largest `G(θ)/Γ(θ)`, the kink of the bulk or a path along an interface
met within an increment (the first-order kinked factors of Cotterell & Rice), so that a
crack turns along interfaces weaker than about a quarter of the phases (He & Hutchinson),
climbs a staircase through the weak boundaries of a brick-and-mortar structure, and
needs a higher load to enter a tough phase:

```python
from morphogen.fracture import Toughness

growth = propagate_kinked(
    mesh_of,
    crack,
    {0: mortar, 1: tablets},
    held,
    forces=tension,
    increment=0.5,
    steps=6,
    toughness=Toughness({0: 1.0, 1: 4.0}, interfaces=0.2),
)
growth.rows()  # each advance: path (bulk or interface), G / toughness, critical load
```

## Interfaces that debond

The interfaces of the phases of a periodic cell (or the boundaries of its grains) open,
break, close again and rub along a loading path ([theory notes](theory.md) §84):
cohesive interface elements inserted on the torus of the cell, so that an interface that
crosses a face goes on in the next cell.

```python
from morphogen.fracture import Bilinear, CohesiveCell

law = Bilinear(
    stiffness=1e7,
    normal_strength=30.0,
    shear_strength=30.0,
    mode_I=0.05,
    mode_II=0.05,
    friction=0.3,
)
cell = CohesiveCell(mesh, materials, law, viscosity=0.02)  # faces="phases", "grains", pairs
res = cell.run({"times": [0, 1, 2], "increments": [24, 8], "strain": {"11": [0, 0.03, 0]}})
res.stress, res.series["debonded_area"], res.series["interface_opening"]
```

In a configuration, `homogenization: {nonlinear: {interfaces: {normal_strength: 30,
mode_I: 0.05, friction: 0.3, viscosity: 0.02}}}` (the template
`homogenization_debonding`: glass beads that debond from their matrix, soft in tension
and nearly whole again in compression); on the command line, `morphogen homogenize ...
--interfaces normal_strength:30,mode_I:0.05,friction:0.3,viscosity:0.02`. A bead lets go
at once: the `viscosity` (a time of the path) slows the damage and lets the steps
through. Each pair of phases may have its law (`{(0, 1): law, (0, 2): other}`; `laws: {0-1:
{...}}`), and `faces: grains` makes the grain boundaries of a polycrystal cohesive (the
template `homogenization_intergranular`: a crack crosses the cell from boundary to
boundary, and the work that broke it is the fracture energy times the area of the
crack). The fields of the path map the damage and the opening of the interfaces. In the
graphical interface, the box *Interfaces* of the tool *Loading path of a morphology*.

## Contact, self-contact and friction

`morphogen.contact` keeps the surfaces of a mesh out of each other, out of themselves and
out of rigid obstacles, with Coulomb's friction, in small or finite strains
([theory notes](theory.md) §82): the integration points of the faces against the
nearest point of the surface that they face, a penalty or the augmented Lagrangian;
with `points="nodes"`, the nodes of the faces instead, for bodies that meet by their
corners and edges.

```python
from morphogen.contact import ContactProblem, Obstacle, Pair, Surface

skin = Surface.of(mesh)  # the free faces of the mesh
walls = Pair(skin, friction=0.3)  # a surface against itself
punch = Obstacle.sphere(skin, centre, radius, motion=(0, 0, -0.1))
problem = ContactProblem(mesh, materials, [walls], [punch])
result = problem.run(prescribed, steps=20)  # forces, areas, pressures, works
```

The walls of the pores of a periodic cell close on themselves, and on their images
through the faces of the cell, along a loading path (`ContactCell`; in a configuration,
`homogenization: {nonlinear: {contact: {friction: 0.3}}}`; the template
`homogenization_contact`). Two tests come ready-made: a sample between two platens, and
under an indenter, whose force reads its modulus:

```python
from morphogen.contact import compression, indentation

crush = compression(mesh, materials, -0.2, friction=0.1)  # stress against strain
crush.modulus, crush.tangent(), crush.self_area  # stiffer as the walls close
dent = indentation(mesh, materials, 0.05, indenter="sphere", radius=10.0, unload=True)
dent.modulus, dent.layer_modulus(poisson=0.3), dent.hardness  # Hertz; with the base
```

```bash
morphogen compress foam.json --size 0.2 --material 0=void 1=young:1000,poisson:0.3 \
    --strain -0.3 --friction 0.1 --fields crushed
morphogen indent block.vtu --material 0=young:1000,poisson:0.3 --depth 0.05 --radius 10 \
    --unload --poisson 0.3
```

## High-cycle fatigue

In the high-cycle regime a cell stays elastic, and whether a point lasts is read from
the cycle of its stress ([theory notes](theory.md) §85): the amplitude of its
deviator (the radius of the smallest hypersphere that holds its path) and a normal or
hydrostatic stress, in a criterion calibrated on two fatigue limits of the phase.

```python
from morphogen.homogenization import DangVan, homogenize, path_fatigue

cycle = {"times": [0, 1, 2, 3], "increments": [1, 2, 2], "stress": {"11": [0, 40, -40, 40]}}
out = homogenize(mesh, materials, properties=[], paths={"cycle": cycle}, fields="steps")
alloy = DangVan.from_limits(bending=110.0, torsion=80.0)  # or Crossland, Sines, Findley, Matake
res = path_fatigue(out["cycle.periodic"], {0: alloy}, cycle=(1, 3))  # a criterion per phase
res.utilization  # of each point: its equivalent stress over its limit
res.largest, res.safety, res.fraction()  # of the cell; the share of its volume beyond the limit
```

In a configuration, `homogenization: {fatigue: {criterion: dang_van, bending: 110,
torsion: 80, cycle: [1, 3]}}` writes the map of the utilization beside the curve of the
path and adds `fatigue_utilization`, `fatigue_safety` and `fatigue_beyond` to the
results (the template `homogenization_fatigue`: pores in a cast alloy; a spherical pore
doubles the stress at its equator, where the utilization is the largest). In the
graphical interface, the box *Fatigue* of the tool *Loading path of a morphology*.
