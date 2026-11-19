# Architectured sheets by localized laser treatment

`morphogen.laser` simulates and optimizes metal sheets *architectured* by a laser: a beam
(or two aligned beams, one on each face) follows a path that hardens — or softens,
carburizes, nitrides, alloys — narrow tracks of the sheet; the pattern of the tracks
gives the sheet a behaviour that neither phase has alone (Bouaziz, Bréchet & Embury
2008; Brechet & Embury 2013). The module follows the PhD thesis of Zhige Wang
(*Architecturation de tôles métalliques par traitement laser localisé*, HESAM
Université, PIMM, 2022) and the ANR project SCOLASTIC (ANR-16-CE08-0009: "systematic
computational optimisation and local laser processing for steel-based architectured
materials"), whose loop it closes: a process model gives the hardness that a path leaves
in the sheet, a finite element model gives the response of the specimen, an optimizer
changes the path.

The formulas are in [theory](theory.md#58-architectured-sheets-by-localized-laser-treatment-laser),
the classes and functions in the [API reference](api.md). This page gives the choices of
a study, the chain of models and what checks them (§1 to §3), then how to run it
(§4, [Use](#4-use)).

## 1. The choices of a study

The thesis left six choices open; the module takes each as an input, with defaults that
the thesis and the literature support.

| Choice | Default | Alternatives in the module | Why |
|---|---|---|---|
| Load case and objective | uniaxial tension of a flat specimen along the rolling direction; maximize the strength–ductility product `R_m A_g` at a minimum yield strength | the energy absorbed up to necking, `A_g`, `R_m`, `R_p`, targets; tension in several directions (specimens cut at 0°, 45°, 90° from the treated sheet: the anisotropy of its yield, an `isotropy` objective); three-point bending of a strip (the load at a deflection, the limit load, the energy); constraints on any measure | hardening leaves Young's modulus unchanged, so stiffness and elastic buckling cannot improve; the pattern acts on yielding, strain hardening and necking, which tension measures cheaply (Considère's maximum of the force); the case of the thesis and of the literature (Fraser et al. 2018a, 2020; Wang et al. 2023; Breukelman et al. 2022). The product alone favours the untreated sheet: the yield constraint fixes the strength level at which ductility is sought |
| Base material | MS1500 annealed at 850 °C (ferrite–pearlite, 0.23 C, 1.7 Mn, 1 mm; 167 HV, `R_p` 300 MPa) | AISI 430 (0.5 mm), DP1180 (soft tracks), any material given as a mapping | the largest contrast of the hard tracks (laser martensite of about 500 HV, `R_p` about 1300 MPa: a ratio near 4) — corrugated reinforcements only pay off when the reinforcement stays nearly elastic while the matrix flows (Fraser et al. 2020); AISI 430 hardens by 50 % at most (ratio 1.56 in argon) and its corrugations gave nothing (thesis ch. 3); soft tracks in a hard sheet localize strain and fail early (Lapouge et al. 2019, Wang 2020 report) |
| Treatment | transformation hardening (martensite), peak between `Ac3` and the solidus | carburizing from a graphite coating, nitriding under nitrogen, alloying from a pre-placed coating (melt pool); tempering of martensitic sheets | no coating, no melting needed in a C–Mn steel; carburizing raises the contrast of low-carbon steels (430: +130 % HV, thesis ch. 4) but needs melting (columnar grains, oxide: argon); nitrogen dissolves in iron only with chromium (0.04 wt% in Fe, 0.3 in Fe–18Cr at 1 atm) |
| Specimen | flat dog-bone, parallel part 30 × 5 × 1 mm, fillets of radius 6 mm, grips 10 mm wide; a DIC-like extensometer over 25 mm | strip (clamped or free ends), any dimensions | the specimens of Wang (2022, ch. 5): one track 1.2 mm wide is 24 % of the section |
| Laser configuration | symmetric: a Gaussian fibre beam (150 W, 1 mm) above and a top-hat diode (100 W, 1.2 mm) below, together, 10 mm/s | single (the upper face), the same beam on both faces | the platform of Wang et al. (2022): through 0.5 mm the gap of peak temperature is about 10 K with two beams and 260 K with one; the tracks are then through-thickness reinforcements, which unbend together |
| Pattern (laser path) | one corrugated track `y = A sin(2π x/P)` along the load, its amplitude and period optimized | straight, oblique, Fourier waveforms (free shapes), crossed families, honeycombs, any polylines | the geometric work hardening of corrugations that unbend (Bouaziz 2013; Fraser et al. 2014, 2018a): `A/P` trades yield strength for uniform elongation (Wang et al. 2023) |

Process constraints enter as constraints of the optimization: the smallest gap between
tracks (closer tracks temper each other; Wang waited at least 30 s between neighbouring
tracks), the smallest radius of curvature of a track (the cycles of the model are those
of a straight track: radii above the heated width; and the kinematics of the robot), the
length of the path in the specimen (time and energy).

## 2. The chain of models

1. **Heat** (`laser.heat`). A beam of absorbed power `P` moving at `v` on a plate of
   thickness `h`: the plane profile of the beam (Gaussian or top hat) spread by the heat
   kernel, images through the thickness for adiabatic faces, a second beam on the lower
   face, surface losses; the quasi-steady field `T(ξ, y, z)` is one integral over the
   delay, a product of matrices on grids. Its limits are Rosenthal's point source and
   Rykalin's line source with losses. A straight track then imposes a thermal cycle
   `T(t; d, z)` on each point at the distance `d` of its axis and the depth `z`
   (`TrackKernel`). Transient finite volumes of the whole blank (implicit, split
   along x, y and z) check the quasi-steady cycles and give the heat that the
   previous tracks leave.
2. **Metallurgy** (`laser.metallurgy`). C–Mn steels: austenite at the peak of each cycle
   (lever rule with the `A3` line, `Ac1` and `Ac3` of Andrews plus a superheat of the
   laser heating rates), martensite on cooling (Koistinen–Marburger below Andrews's `Ms`
   of the austenite's carbon; bainite beyond a critical cooling time `t8/5`),
   hardnesses of Maynier et al. (1978), tempering of martensite by the Hollomon–Jaffe
   parameter accumulated by additivity. AISI 430: the hardness gained as a function of
   the peak temperature (calibrated on the thesis), tempered the same way. Carburizing,
   nitriding and alloying: the coating (or nitrogen, Sieverts) mixed into the melt pool
   of each track by mass balance.
3. **Strength** (`laser.materials`). The tensile properties of each state against its
   hardness (measured: the base and the fully treated 430 at 15 and 20 kJ/m, the
   carburized 430, the annealed MS1500 and its martensite), interpolated, extrapolated by
   Pavlina & Van Tyne's correlations; each point gets the Ludwik (or Voce) law fitted to
   `R_p`, `R_m` and `A_g` by Considère's condition.
4. **Treatment of a path** (`laser.process`). The tracks in their order: the distance
   of each Gauss point to a track gives its cycle at each layer; untouched points take
   the tabulated response of the track, the others (the neighbours of earlier tracks)
   integrate their cycle — the tempered rings, the re-hardened crossings.
5. **Mechanics** (`laser.mechanics`, `laser.bending`). The specimen as a membrane of bilinear
   quadrilaterals in plane stress, each Gauss point a stack of layers with their own flow
   laws; large rotations by a corotational update of the logarithmic strain increments
   (the unbending of the corrugations), von Mises plasticity by a radial return with the
   thickness strain from `σ33 = 0`, the consistent tangent of the Kirchhoff stress;
   displacement-controlled tension to past the maximum force. In several directions,
   the pattern is turned by the opposite angle and repeated at its spacing over the
   sheet before the specimen is cut. Three-point bending: the strip as a statically
   determinate beam whose sections are grids of fibres (across the width, through the
   layers) with their own flow laws, the neutral axis from the axial equilibrium (off
   the mid-plane under a one-sided treatment), the moment–curvature of each section
   inverted for the moment of the load, the deflection by virtual work.
6. **Optimization** (`laser.design`). Parameters of the pattern and of the beams between
   bounds; grid, Nelder–Mead, differential evolution, genetic algorithm, or Bayesian
   optimization (a Gaussian process of the penalized costs, the expected improvement):
   a few tens of evaluations of a few seconds each.

## 3. Checks

The tests (`tests/test_laser.py`) compare with closed forms and independent
computations: the moving source with Rosenthal's thick-plate and Rykalin's thin-plate
solutions (2e-4); the energy `P/v` of a cross-section far behind the beam (2e-3); the
transient finite volumes with the quasi-steady cycles (4 %); Andrews's and Maynier's
values (the annealed MS1500: 173 HV computed, 167 ± 6 measured); the lever rule at `A1`
(the pearlite at once) and `A3`; Koistinen–Marburger after a quench; the additivity of
the Hollomon–Jaffe parameter (an isothermal hold in 50 steps equals the formula to
1e-6); Sieverts's solubility; the mass balance of a carburized melt; the flow laws
against the tensile properties they are fitted to (1e-6); the arc length and curvature
of corrugated tracks; the membrane in uniaxial tension against the exact response of a
homogeneous strip (2e-4 on the curve, `R_m` to 1e-4, `A_g` to 2e-3); the tangent
against finite differences (2e-5); a rigid rotation that turns the stresses and leaves
the plastic strain; two bands along the load at equal strain (5e-3).

Against the experiments of the thesis (measured values in brackets):

* AISI 430, 0.5 mm, double laser: at 50 + 50 W the mid-thickness stays below `Ac1`
  [below Ac1]; at 100 + 100 W the track reaches the melting point through the thickness
  [peak near Tm]; at 150 + 100 W it melts [full fusion at 25 kJ/m].
* Annealed MS1500, 1 mm, 150 + 100 W, absorptivity 0.55: austenitized through the
  thickness over 1.14 mm [tracks of 1.2 mm, nearly fully martensitic].
* AISI 430 fully treated by ten tracks at 1 mm pitch, double laser under argon, after
  the hardness gained at 1300 °C and the tempering of the gain were fitted to the mean
  hardness of these sheets (212, 240 and 235 HV computed [211.5, 241.8, 233.6]):

  | Energy | `R_p` (MPa) | `R_m` (MPa) | `A_g` (%) |
  |---|---|---|---|
  | 15 kJ/m (75 + 75 W) | 468 [488.7] | 626 [623.5] | 15.6 [15.52] |
  | 20 kJ/m (100 + 100 W) | 525 [539.7] | 704 [702.5] | 13.7 [13.65] |
  | 25 kJ/m (150 + 100 W) | 507 [513.0] | 682 [680.6] | 14.1 [13.91] |

  The tempering of earlier tracks by the next ones, which the thesis held responsible
  for the drop from 20 to 25 kJ/m, gives it here too.

## 4. Use

```python
from morphogen.laser import Beam, SheetCase, Specimen, optimize

case = SheetCase(
    material="annealed_ms1500",
    specimen=Specimen(kind="dogbone", length=30.0, width=5.0, thickness=1.0),
    beam=Beam(power=150.0, speed=10.0, radius=0.5, absorptivity=0.55),
    bottom=Beam(power=100.0, speed=10.0, radius=0.6, absorptivity=0.55, profile="tophat"),
    configuration="symmetric",
    pattern={"kind": "corrugated", "count": 1, "amplitude": 1.0, "period": 15.0},
)
result = case.evaluate()
print(result.measures["tensile_strength"], result.measures["uniform_elongation"])

best = optimize(
    case,
    {"pattern.amplitude": [0.0, 2.5], "pattern.period": [6.0, 30.0]},
    objective="strength_ductility",
    constraints={"yield_strength": [400.0, None], "min_radius": [1.5, None]},
    method="bayesian",
    evaluations=30,
)
print(best.values, best.objective)
print(best.result.path.to_gcode())  # the laser path of the optimum
```

Bending and directions are options of the case: `loading={"kind": "bending", "span":
30.0, "deflection": 1.0}` with a `Specimen(kind="strip", ...)`, and `directions=(0, 45,
90)` (the measures are then the means, `yield_strength_45`... each direction's, and
`yield_anisotropy`, `strength_anisotropy`, `elongation_anisotropy`).

Cellular designs:

```python
from morphogen.laser import CellProblem, SheetCase, beso, design_tracks, wang_phases
from morphogen.laser.cells import corrugation

problem = CellProblem(
    phases=wang_phases(),  # ferrite and brittle martensite (Wang et al. 2023)
    size=0.5,  # cells of 0.5 mm
    objective="ductility",  # the failure strain...
    constraints={"tensile_strength": [700.0, None]},  # ...at R_m >= 700 MPa
)
start = problem.grid.stamp(corrugation(problem.specimen, 1.5, 10.0))  # a 1 mm spot
run = beso(
    problem,
    start,
    budget=None,
    moves="exchange",
    symmetry=("point",),
    period=10.0,
    min_width=2,
    candidates=6,
)
print(run.summary())
tracks = design_tracks(problem.grid, run.design)  # the laser path that draws it
case = SheetCase(pattern={"kind": "polylines", "tracks": [t.tolist() for t in tracks]})
print(case.evaluate().measures["tensile_strength"])
```

Structures and directions:

```python
from morphogen.laser import CellProblem, FlowLaw, SheetCase
from morphogen.laser.cells import Phase
from morphogen.laser.structures import CompactTension, LPlate

phases = (
    Phase("ferrite", FlowLaw.hollomon(280, 986, 0.24), critical=0.5, cost=0),
    Phase("martensite", FlowLaw.hollomon(1300, 1966, 0.06), critical=0.075),
)
ct = CellProblem(
    phases=phases,
    case=CompactTension(max_disp=5.0),
    size=0.25,
    fracture_radius=0.5,
    objective="energy",
)
band = ...  # an array of the cells of ct.grid (1: martensite)
print(ct.evaluate(band).measures)  # failure_strain: the initiation (load-line / W)
sheet = SheetCase(
    pattern={"kind": "crossed", "angle": 45.0, "spacing": 5.0}, directions=(0, 45, 90)
)
m = sheet.evaluate().measures
print(m["r_value_0"], m["r_bar"], m["delta_r"], m["hill_n"])
```

A rolled sheet treated all over, on the periodic cell of its pattern:

```python
from dataclasses import replace
from morphogen.laser import SheetCase, sheet_material

rolled = replace(
    sheet_material("annealed_ms1500"), anisotropy={"r0": 1.6, "r45": 1.2, "r90": 2.0}
)  # + "memory": 0..1
sheet = SheetCase(
    material=rolled,
    element=0.25,
    pattern={"kind": "honeycomb", "cell": 3.0},
    loading={"kind": "cell"},  # the cell of the lattice of the pattern
    directions=(0, 45, 90),
)
m = sheet.evaluate().measures
print(m["r_value_0"], m["r_value_45"], m["r_value_90"], m["r_anisotropy"], m["r_bar"])
```

Yld2000-2d (yield stresses and Lankford coefficients of a DC04) and the growth of a
crack:

```python
from dataclasses import replace
import numpy as np
from morphogen.laser import CellProblem, FlowLaw, SheetCase, sheet_material
from morphogen.laser.cells import Phase
from morphogen.laser.structures import CompactTension

dc04 = {
    "criterion": "yld2000",
    "s0": 151,
    "s45": 166,
    "s90": 163,
    "sb": 192,
    "r0": 1.83,
    "r45": 1.39,
    "r90": 2.11,
    "rb": 0.87,
    "a": 6,
}
rolled = replace(sheet_material("annealed_ms1500"), anisotropy=dc04)
sheet = SheetCase(
    material=rolled,
    element=0.25,
    loading={"kind": "cell"},
    pattern={"kind": "straight", "angle": 0.0, "count": None, "spacing": 60.0},
    directions=tuple(range(0, 91, 15)),
)
print(sheet.evaluate().measures["yield_anisotropy"])

ct = CompactTension(
    max_disp=15.0, fine=0.25, refine=(4.0, 20.0, 9.0), propagate=True, region=(4.0, 20.0, -9.0, 9.0)
)
phases = (
    Phase("ferrite", FlowLaw.hollomon(280, 986, 0.24), critical=0.5, cost=0),
    Phase("hard wall", FlowLaw.hollomon(1300, 1966, 0.06), critical=float("inf")),
)
problem = CellProblem(phases=phases, case=ct, size=0.25, fracture_radius=0.5)
walls = ...  # an array of the cells of problem.grid (1: hard), e.g. along a spiral
result = problem.evaluate(walls).result
broken, at = result.fields["broken"]  # the elements of the crack, in their order
print(np.trapezoid(result.force, result.displacement) / 1000)  # the work (J)
```

The `cellular` section of a configuration takes a structure too (`case: {kind: ct,
max_disp: 5}` or `{kind: l_plate, region: [0, 34, 0, 36]}`, phases with their critical
strains, a start of `tracks`).

From the command line, the template `laser_corrugated` optimizes a corrugated track and
`laser_cells` runs a cellular design and treats its tracks (the `cellular` section of
the configuration, about a minute):

```bash
morphogen template laser_corrugated -o laser.yaml
morphogen laser laser.yaml            # results/ laser.json, curves, history, G-code, maps
morphogen laser laser.yaml --no-optimization
```

## References

* Wang Z. (2022) *Architecturation de tôles métalliques par traitement laser localisé*.
  PhD thesis, HESAM Université (ENSAM, PIMM), supervised by J. Dirrenberger.
  [tel-04041478](https://pastel.hal.science/tel-04041478).
* Wang Z., Dirrenberger J., Lapouge P., Dubent S. (2022) Laser treatment of 430 ferritic
  stainless steel for enhanced mechanical properties. *Mater. Sci. Eng. A* 831, 142205.
  [10.1016/j.msea.2021.142205](https://doi.org/10.1016/j.msea.2021.142205)
* Wang Z., Dirrenberger J., Lapouge P., Dubent S., Jabir H., Michel V. (2022)
  Microstructure evolution and mechanical properties of AISI 430 ferritic stainless
  steel strengthened through laser carburization. *J. Eng. Mater. Technol.* 144, 041005.
  [10.1115/1.4055025](https://doi.org/10.1115/1.4055025)
* Wang Z., Bouaziz O., Dirrenberger J., Lapouge P. (2023) Corrugation reinforced
  architectured materials by direct laser hardening: a study of geometrically induced
  work hardening in steel. *steel res. int.* 94, 2200695.
  [10.1002/srin.202200695](https://doi.org/10.1002/srin.202200695)
* Lapouge P., Dirrenberger J., Coste F., Schneider M. (2019) Laser heat treatment of
  martensitic steel and dual-phase steel with high martensite content. *Mater. Sci.
  Eng. A* 752, 128–135. [10.1016/j.msea.2019.03.016](https://doi.org/10.1016/j.msea.2019.03.016)
* Bouaziz O., Bréchet Y., Embury J.D. (2008) Heterogeneous and architectured materials:
  a possible strategy for design of structural materials. *Adv. Eng. Mater.* 10, 24–36.
  [10.1002/adem.200700289](https://doi.org/10.1002/adem.200700289)
* Brechet Y., Embury J.D. (2013) Architectured materials: expanding materials space.
  *Scripta Mater.* 68, 1–3. [10.1016/j.scriptamat.2012.07.038](https://doi.org/10.1016/j.scriptamat.2012.07.038)
* Bouaziz O. (2013) Geometrically induced strain hardening. *Scripta Mater.* 68, 28–30.
  [10.1016/j.scriptamat.2012.08.008](https://doi.org/10.1016/j.scriptamat.2012.08.008)
* Fraser M., Zurob H., Wu P., Bouaziz O. (2014) Analytical model of the unbending
  behavior of corrugated reinforcements. *Adv. Eng. Mater.* 16, 872–877.
  [10.1002/adem.201300525](https://doi.org/10.1002/adem.201300525)
* Fraser M., Zurob H.S., Wu P. (2018a) Corrugation reinforced composites: a method for
  filling holes in material-property space. *Adv. Eng. Mater.* 20, 1700834.
  [10.1002/adem.201700834](https://doi.org/10.1002/adem.201700834)
* Fraser M., Zurob H., Wu P., Bouaziz O. (2020) Increasing necking strain through
  corrugation: identifying composite systems that can benefit from corrugated geometry.
  *Materials* 13, 5175. [10.3390/ma13225175](https://doi.org/10.3390/ma13225175)
* Breukelman H.J., Santofimia M.J., Hidalgo J. (2022) Hierarchically patterned
  multiphase steels created by localised laser treatments. *Mater. Des.* 221, 110984.
  [10.1016/j.matdes.2022.110984](https://doi.org/10.1016/j.matdes.2022.110984)
* Rosenthal D. (1946) The theory of moving sources of heat and its application to metal
  treatments. *Trans. ASME* 68, 849–865. [10.1115/1.4018624](https://doi.org/10.1115/1.4018624)
* Andrews K.W. (1965) Empirical formulae for the calculation of some transformation
  temperatures. *J. Iron Steel Inst.* 203, 721–727.
* Koistinen D.P., Marburger R.E. (1959) A general equation prescribing the extent of the
  austenite-martensite transformation in pure iron-carbon alloys and plain carbon
  steels. *Acta Metall.* 7, 59–60. [10.1016/0001-6160(59)90170-1](https://doi.org/10.1016/0001-6160(59)90170-1)
* Maynier P., Jungmann B., Dollet J. (1978) Creusot-Loire system for the prediction of
  the mechanical properties of low alloy steel products. In *Hardenability Concepts
  with Applications to Steel*, TMS-AIME, 518–545.
* Hollomon J.H., Jaffe L.D. (1945) Time-temperature relations in tempering steel.
  *Trans. AIME* 162, 223–249.
* Pavlina E.J., Van Tyne C.J. (2008) Correlation of yield strength and tensile strength
  with hardness for steels. *J. Mater. Eng. Perform.* 17, 888–893.
  [10.1007/s11665-008-9225-5](https://doi.org/10.1007/s11665-008-9225-5)
* Lankford W.T., Snyder S.C., Bauscher J.A. (1950) New criteria for predicting the press
  performance of deep drawing sheets. *Trans. ASM* 42, 1197–1232.
