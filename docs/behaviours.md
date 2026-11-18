# Behaviours along loading paths

A loading path prescribes each macroscopic component as a strain or a stress in time,
with a change of temperature, and the phases follow nonlinear behaviours along it, in
small or finite strains. The same paths run on conforming meshes (finite elements), on
voxel images (the FFT solver) and at a material point, so that a behaviour is tried
alone before it goes into a microstructure; the curves of a morphology then give it a
macroscopic law, identified and written for other codes. The
[tutorial](tutorial.md) follows a plastic loading path of a composite, and the
[tutorial on a rubber](tutorial_rubber.md) the finite strains of a hyperelastic one.

## Behaviours

* Nonlinear behaviours along loading paths: von Mises elasto-viscoplasticity (Norton
  flow, linear and Voce isotropic hardening, Prager and Armstrong–Frederick
  kinematic hardening) or rate-independent plasticity, integrated implicitly with
  the consistent tangent, or with Hill's anisotropic criterion (Z-set's coefficients
  `h11` … `h31` by component; the closest point projection by Newton's method); paths that prescribe each component as a strain or a
  stress in time, with a change of temperature; Newton iterations whose unknowns
  include the macroscopic strain components (periodic and kinematic conditions on
  meshes, periodic on voxels), with line search and step division; macroscopic curves,
  phase averages and fields of plastic strain; a material point simulator.
* Crystal plasticity of Méric and Cailletaud for polycrystals: octahedral, cube and
  body-centred cubic slip families, Norton flow, isotropic hardening coupled by an
  interaction matrix (Franciosi's six interactions), kinematic hardening per system,
  slip systems turned by the orientation of each grain; robust implicit integration
  with its consistent tangent.
* Finite strains for every nonlinear behaviour: total Lagrangian finite elements,
  Hencky elasticity and von Mises plasticity in the logarithmic strain space,
  multiplicative crystal plasticity with lattice rotations; nominal and true (Cauchy)
  stresses, logarithmic strains.
* Behaviours in closed form for every solver: viscoelasticity (Prony series, WLF),
  hyperelasticity (neo-Hookean, the polynomial of Rivlin with Mooney–Rivlin and Yeoh,
  Arruda–Boyce and the eight chains, Gent, van der Waals, Fung, Ogden, the fibres of
  Holzapfel–Gasser–Ogden, the foams of Blatz–Ko and Ogden–Hill; the softening of
  Mullins, finite viscoelasticity; `fit_hyperelasticity` fits them to the uniaxial,
  equibiaxial, planar and shear tests of a rubber) and isotropic damage
  (crack band, viscous regularization, the unilateral effect of cracks that close),
  Mazars' damage of concrete, and the damage coupled to plasticity (Lemaitre's ductile
  damage, the creep damage of Kachanov and Rabotnov: `damage: {law: lemaitre, S: ...}`
  beside `plasticity`), beside plasticity, crystal plasticity and porous plasticity.
* Behaviours written in MFront (TFEL), integrated by MGIS (optional, from conda-forge):
  standard behaviours in small or finite strains in place of morphoGen's own, by
  finite elements, on voxels (CPU or GPU) and at a material point, crystals turned by
  the orientation of each grain; `.mfront` sources compiled on first use and cached;
  validated against MFront twins of morphoGen's behaviours.

## Loading paths

```python
from morphogen.homogenization import LoadingPath, Material, homogenize, simulate_point

# mesh: a conforming mesh of two phases (Apparent properties, Elasticity and conduction)
matrix = Material(
    young=70e3,
    poisson=0.33,
    plasticity={  # aluminium alloy (MPa, s)
        "yield": 100,
        "isotropic": [{"Q": 80, "b": 20}],
        "kinematic": [{"C": 1e4, "D": 100}],
        "norton": {"K": 50, "n": 5},
    },
)
particles = Material(young=400e3, poisson=0.2)  # SiC
# tension at 1e-3 /s to 2 %, relaxation for 100 s, unloading; lateral stresses free
path = LoadingPath.parse("t=0,20,120,130 n=10,10,5 E11=0,0.02,0.02,0.017")
out = homogenize(mesh, {0: matrix, 1: particles}, properties=[], paths={"tension": path})
curve = out["tension.periodic"]  # times, strain, stress, plastic, phases, fields
print(curve.stress[:, 0], curve.phases[0]["plastic"])
print(simulate_point(matrix, path).stress[-1])  # the matrix alone
```

## Crystal plasticity

Crystal plasticity (Méric–Cailletaud) in the grains of a polycrystal mesh, in finite
strains:

```python
from morphogen.homogenization import CrystalPlasticity, Material, NonlinearFEProblem

nickel = Material.from_config(
    {
        "cubic": [250e3, 160e3, 120e3],  # MPa
        "crystal_plasticity": {
            "families": ["octahedral"],  # also cube, bcc110, bcc112
            "tau0": 80,  # r = tau0 + Q sum h (1 - e^-bv)
            "Q": 20,
            "b": 10,
            "C": 2e4,  # x = C alpha
            "D": 300,
            "K": 25,  # <(|tau - x| - r) / K>^n
            "n": 10,
            "interaction": [1, 1, 0.6, 12.3, 1.6, 1.3],  # Franciosi
        },
    }
)
# poly_mesh: a polycrystal meshed grain by grain (Apparent properties, Polycrystals)
problem = NonlinearFEProblem(poly_mesh, {1: nickel}, "periodic", kinematics="finite")
curve = problem.run({"times": [0, 100], "increments": 40, "strain": {"11": 0.1}}, fields=True)
print(curve.stress[:, 0], curve.cauchy[:, 0], curve.log_strain[:, 0])  # S, TS, LE
print(curve.fields[100.0]["orientation"][:5])  # current lattice orientations (Bunge)
```

## Twins and cleavage

Crystals also twin and cleave ([theory notes](theory.md) §86–87). Hexagonal
crystals have their slip families (`basal`, `prismatic`, `pyramidal_a`, `pyramidal_ca`,
`pyramidal2_ca`, with the ratio `c_over_a`); twins shear in one sense until the crystal
has twinned; cleavage planes open at a normal stress and soften, with the fracture
energy of a crack band:

```python
magnesium = Material.from_config(
    {
        "hexagonal": [59400, 25600, 21400, 61600, 16400],  # c11, c12, c13, c33, c44
        "crystal_plasticity": {
            "c_over_a": 1.624,
            "K": 5,
            "n": 10,
            "families": {
                "basal": {"tau0": 10},
                "prismatic": {"tau0": 60},
                "pyramidal2_ca": {"tau0": 100},
            },
            "twins": {"twin1012": {"tau0": 25}},  # {10-12}: lengthens c
        },
    }
)
iron = Material.from_config(
    {
        "cubic": [230e3, 135e3, 117e3],
        "crystal_plasticity": {
            "K": 10,
            "n": 10,
            "families": {"bcc110": {"tau0": 300}},
            "cleavage": {
                "planes": "cube",
                "strength": 500,  # {100}: opens at 500 MPa
                "energy": 0.05,  # G_c
                "length": 0.008,  # in elements of 8 µm
                "K": 50,
                "n": 1,
            },
        },
    }
)
curve = simulate_point(magnesium, {"times": [0, 1], "increments": 20, "strain": {"33": 0.03}})
curve.series["twin_fraction"]  # of a crystal pulled along c; cleavage_damage for iron
```

A textured magnesium flows at half the stress in the sense that twins it (the template
`homogenization_twinning`); a polycrystal of iron breaks through its grains from
cleavage plane to cleavage plane (`homogenization_cleavage`), or between them when its
boundaries are cohesive and weaker (`interfaces: {faces: grains}`, the template
`homogenization_grain_fracture`). The fields map the `twin_fraction`, the
`cleavage_opening` and the `cleavage_damage`. Under a stress that is reversed the twins
shear back and the cleavage planes close ([theory notes](theory.md) §89): a
closed crack carries the pressure as the sound crystal, its lips rubbing with
`friction` (the template `homogenization_crack_closure`); and with `twin_reorientation`
the lattice of a point becomes that of its predominant twin at a threshold of its
twinned fraction, slips and twins as the twin does, and twins back (the field
`twin_misorientation`; the template `homogenization_detwinning`), point by point, or
grain by grain (`{threshold, growth}`: the turned volume of a grain, `twin_reoriented`,
then follows its twinned volume). `twin_detwinning` and `closure` also take the share of
their resistance at which the twins shear back and the planes close. And the slip can
nucleate the cleavage ([theory notes](theory.md) §90): with `cleavage: {coupling:
Θ}` the strength of the planes falls as `exp(−Θ γ)` of the cumulated slip of the
crystal, and their damage eases its slip.

How likely is cleavage to start? The local approach of Beremin
([theory notes](theory.md) §88) reads it from the fields of a path: the Weibull
stress of the points that have yielded, and the probability that the weakest link has
failed.

```python
from morphogen.homogenization import Beremin, path_beremin

law = Beremin(
    modulus=22,  # m
    scale=2600.0,  # sigma_u (MPa)
    volume=1.25e-4,  # V0 (mm³)
    planes="cube",  # the {100} planes of the grains
    plastic=1e-4,
)
found = path_beremin(curve, law, orientations=euler_of_each_voxel)  # fields="steps"
found.weibull, found.probability  # at each step; found.driving: the stress of each point
```

In a configuration, `homogenization: {beremin: {modulus: 22, scale: 1550, volume:
1.25e-10, planes: cube, plastic: 1e-4}}` adds the curves `beremin_weibull` and
`beremin_probability` to those of the path (the template `homogenization_beremin`); in
the graphical interface, the box *Cleavage* of the tool *Loading path of a morphology*.

## Finite strains

In finite strains the macroscopic gradient is upper triangular: `E11 ... E12` are its
components (nominal strains, and simple shears for `E23, E31, E12`), `S11 ... S12` the
same components of the nominal (first Piola–Kirchhoff) stress; free components leave
the faces free of tractions, so that a tension is a uniaxial Cauchy stress.

## On voxel images

The same paths are followed on voxel images by the FFT solver, with the grains and
orientations of a tessellation:

```python
from morphogen.homogenization import NonlinearFFTProblem

image = poly.voxelize(shape=32)  # a Voronoi realization, 32³ voxels
problem = NonlinearFFTProblem(
    image,
    {1: nickel},
    grains=poly.labels(grid=image.grid()),
    orientations=poly.orientations,
    kinematics="finite",
)
curve = problem.run({"times": [0, 50], "increments": 25, "strain": {"11": 0.05}})
```

Each voxel is a trilinear element with 8 Gauss points; the volumetric part of the strain
is averaged over the voxel (`integration="bbar"`: B-bar, and F-bar in finite strains),
without which the voxels lock under plastic flow (the flow stress of elastic cubes in a
plastic matrix is twice too high on 10³ voxels with `integration="full"`). The memory
is that of the integration points: about 8 kB per voxel for a crystal with 12 slip
systems in small strains (2 GB for 64³ voxels).

## MFront behaviours

Behaviours written in MFront take the place of morphoGen's, with MGIS and MFront
installed (the source is compiled on first use and cached):

```python
from morphogen.homogenization.mfront import MFrontBehaviour

alloy = Material(
    young=70e3,  # with poisson: the linear problems, the FFT reference
    poisson=0.33,
    mfront=MFrontBehaviour(source="tests/mfront/VonMisesTwin.mfront"),
)
print(simulate_point(alloy, path).stress[-1])  # its own elasticity and viscoplasticity
out = homogenize(mesh, {0: alloy, 1: particles}, properties=[], paths={"tension": path})
```

In configurations and materials files, `mfront` is the path of the source (relative to
the file) or a mapping: `source`, or `library` and `behaviour`; `material_properties`,
`parameters` and `external_state_variables` (constants); `temperature` (the reference, in
K, to which the paths add their change of temperature); `plastic_strain` (the internal
variable of the cumulated plastic strain of the outputs).

## Identification of a macroscopic law

The parameters of a behaviour are identified on reference responses, as Zopt does for
Zsim: the curves of `homogenize` (`PathResult`: `curve.path`, `curve.times`,
`curve.stress`) of a morphology under several paths give it a macroscopic law, material
points or the points of a test calibrate a law; each reference counts by its weight
whatever its number of points. Gurson's q1 and q2 under two triaxialities, written back
for Z-set:

```python
from morphogen.homogenization import Material, simulate_point
from morphogen.homogenization.identification import Reference, identify_references
from morphogen.homogenization.zmat import write_zmat

tension = {"times": [0, 1], "increments": 12, "strain": {"11": [0, 0.05]}}  # triaxiality 1/3
confined = {**tension, "strain": {"11": [0, 0.05], "22": [0, 0], "33": [0, 0]}}  # held laterally
voids = {"criterion": "gurson", "yield": 100, "hardening": 1000, "porosity": 0.07}
law = Material.from_config(
    {"young": 70e3, "poisson": 0.33, "porous_plasticity": {**voids, "q1": 1.5, "q2": 1.0}}
)
refs = [
    Reference(r.path, r.times, r.stress, components)
    for r, components in (
        (simulate_point(law, tension), [0]),
        (simulate_point(law, confined), [0, 1, 2]),
    )
]
start = {"young": 70e3, "poisson": 0.33, "porous_plasticity": {**voids, "q1": 1.0, "q2": 1.3}}
fit = identify_references(start, refs, ["q1", "q2"])
print(fit.values, fit.gaps)  # q1 1.5, q2 1.0; the gap to each response
write_zmat(fit.material, "gurson.mat", digits=6)  # porous_plastic, for Z-set
```

In the interface, "Identification of a behaviour" (tab Mechanics) takes the runs of the
workspace, each under its own path, and "Z-set file of a behaviour" (Solvers & formats)
writes the behaviour found.

An anisotropic morphology calls for an anisotropic law: Hill's criterion
(`hill: {11: …, 22: …, 33: …, 12: …, 23: …, 31: …}`, Z-set's form, 1 where not given),
whose coefficients `hill11` … `hill31` the identification fits from von Mises's. Two
plastic phases in layers normal to z (`examples/hill_of_a_laminate.py`, a few seconds)
yield alike in tension along x and z, but their shear across the layers (31, the layers in
series) saturates at 0.63 where the shear in their plane (12, in parallel) reaches 0.89:
von Mises misses the first by 38 %, Hill's law found (`h31 ≈ 2`, `h12 ≈ h33 ≈ 1`) follows
all four responses within 4 %, and `write_zmat` writes it as a `gen_evp` with
`*criterion hill` for Z-set. `macroscopic_law(respond)` does it in one go: it runs the
tensions and shears chosen through `respond(path)` (the response of the morphology),
estimates a starting law on them (elasticity of the first increment, yield stress,
hardening; with `cycle=True` the loadings reverse and the law takes a kinematic
hardening, the Bauschinger effect of the morphology) and identifies the von Mises or Hill
law on them together; in the interface,
"Macroscopic law of a morphology" (tab Mechanics) does the same on its voxels and writes
the law for Z-set. `law="gurson"` gives a porous morphology Gurson's law: the matrix
of its phase, its voids as the porosity, q1 and q2 identified on loadings of several
triaxialities (`loadings=["11", "confined11"]`: a tension, then one whose lateral
strains are held), as on unit cells.

## Z-mat material files

The material files of Z-mat (Z-set) run as they are: a `gen_evp` behaviour of von Mises
or Hill elasto-(visco)plasticity on morphoGen's own kernels, the others (hyperbolic or
double Norton flows, power-law hardening, several potentials...) as MFront behaviours
(MFront's brick StandardElastoViscoPlasticity):

```yaml
homogenization:
  materials:
    0: {zmat: alloy.mat}            # ***behavior gen_evp ... ***return
    1: {young: 400000, poisson: 0.2}
```

`morphogen zmat alloy.mat` shows the translation, `morphogen zmat --coverage` what of
Z-mat's library morphoGen covers ([Z-mat material files](zmat.md)). The other way, `write_zmat` writes an
elastic, von Mises, Hill or porous material of morphoGen (a behaviour identified on the response
of a morphology: its Gurson model) as a material file of Z-mat, read back as the same:

```python
from morphogen.homogenization.zmat import write_zmat

write_zmat(
    {
        "young": 200000,
        "poisson": 0.3,
        "plasticity": {
            "yield": 130,
            "kinematic": [{"C": 15000, "D": 300}],
            "norton": {"K": 400, "n": 7},
        },
    },
    "alloy.mat",
)
```

Porous plasticity (`porous_plastic` files of Z-mat, or a material of its own): the ductile
damage of Gurson-Tvergaard-Needleman (coalescence, Chu and Needleman's nucleation) and of
Rousselier, and the compaction of powders and foams by the elliptic criterion of Abouaf,
with the Armstrong-Frederick backstresses of the matrix for GTN and the elliptic criterion
(`kinematic: [{C: 30000, D: 200}]`: the potential of σ - X), in small and finite strains,
on morphoGen's kernels (CPU); the curves give the porosity `f`:

```yaml
homogenization:
  materials:
    0:
      young: 200000
      poisson: 0.3
      porous_plasticity: {criterion: gurson, yield: 300, porosity: 0.01, q1: 1.5, q2: 1.0,
                          fc: 0.05, fF: 0.2, hardening: 1000}
    1: {zmat: {file: powder.mat, porosity: 0.4}}   # ***behavior porous_plastic ...
```

## Reduced and learned models

Reduced models of nonlinear voxel microstructures (nonuniform transformation
field analysis, Michel & Suquet): modes from full-field snapshots, then loading paths
in milliseconds, within a few percent of the full fields; also from configurations
(`homogenization: ntfa`).

Behaviours are also learned from data (`morphogen.learning`, `morphogen learn`; NumPy and
Numba only): random strain paths at a material point or on a microstructure, a
perceptron, a learned convex potential (hyperelastic: a symmetric positive tangent),
a recurrent network over the increments (plasticity) or a learned generalized
standard material (a convex free energy and dissipation potential of learned
internal variables, the state of a step the minimizer of the incremental potential:
a nonnegative dissipation and a symmetric positive tangent by construction, trained
through the minimizations by implicit differentiation), every model in finite
strains of the logarithmic strain (by default, as the Hencky points: its stress
conjugate to `½ ln C`, objective) or of the Green-Lagrange strain, trained with their
gradients written out, then integrated in place of a behaviour (`learned` in a
material) at a material point, by finite elements and on voxels; a law learned in
the axes of a crystal (its paths turned by the rotations of the cube) turned grain
by grain in polycrystals, the stress-free thermal strain of its material taken out;
a tool of the interface that learns a behaviour in the background and keeps its law;
every learned law exported as an MFront behaviour (its trained parameters and a C++
port of its model, in small or finite strains, its state as state variables), for
Abaqus, Cast3M, Code_Aster and AMITEX_FFTP, checked against morphoGen's own
integration to round-off.

## Micromorphic continua

`morphogen.micromorphic` solves micromorphic media on tetrahedral meshes (quadratic
displacements, linear generalized degrees of freedom), in small or finite strains: the
microfoam model of nickel foams (Dillard, Forest & Ienny 2006: anisotropic compressible
plasticity in a micromorphic continuum, whose length removes the strain concentration of
holes as small as the cells; in finite strains the plasticity of the logarithmic strains
and the objective measures `χ⁻¹ F` and `χ⁻¹ Grad χ` of Forest & Sievert), a micromorphic
damage model (the implicit gradient model of Peerlings et al.) and micromorphic ductile
damage (von Mises elasto-(visco)plasticity whose damage follows the micromorphic
cumulated plastic strain), whose cracks are bands of damage independent of the mesh:

```python
from morphogen.micromorphic import (
    CHI,
    DamageProblem,
    DuctileProblem,
    Microfoam,
    MicrofoamProblem,
    MicromorphicDamage,
    MicromorphicDuctile,
)

# mesh: a tetrahedral mesh; bottom, held, top, supports: its nodes held and driven
foam = Microfoam.nickel_foam(length=0.1)  # Tables 2 and 3 of Dillard et al. (MPa, mm)
problem = MicrofoamProblem(mesh, foam)  # u (P2) and the microdeformation (P1)
result = problem.run([(bottom, 1, 0.0), (held, CHI["chi12"], 0.0), (top, 1, 0.1)], steps=10)
fields = problem.fields(result.values, result.state)  # strain, stress, p, chi
crushed = MicrofoamProblem(mesh, foam, kinematics="finite")  # fields: log strain, Cauchy, J

damage = MicromorphicDamage(young=1000.0, poisson=0.2, threshold=1e-3, length=0.08)
cracked = DamageProblem(mesh, damage, kinematics="finite").run(supports, steps=40)

steel = MicromorphicDuctile(
    young=200e3,
    poisson=0.3,
    yield_stress=400.0,
    hardening=1e3,
    voce=((200.0, 20.0),),
    length=0.1,
    onset=0.02,
    critical=0.1,
    triaxiality="rice_tracey",  # or johnson_cook, lemaitre, none
)
ductile = DuctileProblem(mesh, steel, kinematics="finite")
result = ductile.run(supports, steps=40)  # a predictor by the last tangent
ductile.damage(result.state), ductile.plastic_strain(result.state)
```

## From the command line

`morphogen homogenize` follows the paths of a sample (a geometry `.json`, a mesh
`.vtu`/`.mesh`, or an image for the FFT solver), with the behaviours of a materials file:

```bash
# a loading path with the plasticity of the materials (YAML), curves in curves/
morphogen homogenize composite.vtu --materials alloy.yaml --property stiffness \
    --path "tension: t=0,20,120 n=10,10 E11=0,0.02,0.02" --curves curves/
# the same in finite strains: the curves add the true strains LE and stresses TS
morphogen homogenize polycrystal.vtu --materials nickel.yaml --periodic --finite \
    --path "tension: t=0,100 n=40 E11=0.1" --curves curves/ --fields fields/
# on a voxel image: the FFT solver (fields as .vti)
morphogen homogenize composite.vti --materials alloy.yaml \
    --path "tension: t=0,20 n=10 E11=0,0.02" --curves curves/ --fields fields/
# the same on an NVIDIA GPU (CuPy, the extra [gpu])
morphogen homogenize composite.vti --materials alloy.yaml --device gpu \
    --path "tension: t=0,20 n=10 E11=0,0.02" --curves curves/
```

## Templates

`homogenization_viscoplastic` (tension, relaxation and unloading of a
particle-reinforced viscoplastic alloy), `homogenization_crystal_plasticity` (tension of
a polycrystal to 10 % in finite strains, Méric–Cailletaud crystal plasticity),
`homogenization_crystal_plasticity_fft` (the same on 24³ voxels of 20 grains, to 5 %,
FFT), `homogenization_twinning` (the twins of a textured magnesium, pulled and
compressed), `homogenization_detwinning` (the same pulled and brought back: twins that
shear back, lattices that turn), `homogenization_cleavage` (a polycrystal of iron that
cleaves through its grains), `homogenization_crack_closure` (the same pushed back: its
crack closes), `homogenization_slip_cleavage` (a polycrystal that yields first: the slip
of its grains nucleates their cleavage), `homogenization_cleavage_band` (planes
regularized by their microdamage: a crack across a band of a given width, whatever the
grid), `homogenization_grain_fracture` (cohesive grain boundaries and cleavage planes:
cracks between the grains or through them), `homogenization_beremin` (the probability of
cleavage of a ferritic steel, by the Weibull stress of Beremin).
