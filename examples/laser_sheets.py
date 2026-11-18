"""Architectured sheets by localized laser treatment: the studies of docs/laser.md.

1. The thermal cycles of a track: one beam against two (a Gaussian fibre beam above, a
   top-hat diode below), the bands that reach Ac1, Ac3 and the solidus.
2. Corrugated martensite in an annealed 0.23 C steel (Wang 2022, thesis ch. 5; Wang et
   al. 2023): the uniform elongation that the height over the period of the corrugation
   buys, against the yield and tensile strength it costs.
3. The other choices: AISI 430 hardened, carburized or nitrided; one beam against two;
   tracks across the load; soft tracks in DP1180.
4. Three-point bending of a strip: the treatment of one face against both.
5. The anisotropy of a corrugated sheet, tested at 0, 45 and 90 degrees.
6. The Bayesian optimization of the amplitude and period of the corrugation for the
   strength-ductility product at a minimum yield strength.

Elements of 0.5 mm (``--element 0.25`` for converged values, four times longer). Run
time: about three minutes.
"""

import argparse

from morphogen.laser import Beam, SheetCase, Specimen, optimize, sheet_material, track_kernel

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--element", type=float, default=0.5, help="element size (mm)")
parser.add_argument("--evaluations", type=int, default=20, help="of the optimization")
args = parser.parse_args()

FIBRE = Beam(150.0, speed=10.0, radius=0.5, absorptivity=0.55)  # Gaussian, 1 mm
DIODE = Beam(100.0, speed=10.0, radius=0.6, absorptivity=0.55, profile="tophat")  # 1.2 mm


def show(name, case):
    m = case.evaluate().measures
    if case.loading.get("kind") == "bending":
        print(
            f"{name:44s} F(1 mm) {m['bending_load_at_deflection']:6.1f} N, largest "
            f"{m['bending_load']:6.1f} N, energy {m['bending_energy']:6.1f} mJ, "
            f"treated {m['volume_fraction']:.2f}"
        )
    else:
        print(
            f"{name:44s} Rp {m['yield_strength']:5.0f}  Rm {m['tensile_strength']:5.0f} "
            f" Ag {100 * m['uniform_elongation']:5.1f} %  energy {m['energy']:5.1f}  "
            f"treated {m['volume_fraction']:.2f}"
        )
    return m


# 1. thermal cycles --------------------------------------------------------------------
steel = sheet_material("annealed_ms1500")
a1, a3 = steel.metallurgy.a1, steel.metallurgy.a3
print("1. Tracks on 1 mm of annealed MS1500 at 10 mm/s")
for name, beam, bottom in (
    ("fibre 250 W alone", FIBRE.with_(power=250.0), None),
    ("fibre 150 W + diode 100 W", FIBRE, DIODE),
):
    k = track_kernel(beam, 1.0, steel.thermal, bottom=bottom, layers=4)
    print(
        f"  {name:28s} peaks through the thickness {k.peak[0].round()} °C; widths through "
        f"the thickness: A1 {k.through_width(a1):.2f} mm, A3 {k.through_width(a3):.2f} mm; "
        f"melted {k.width(steel.metallurgy.solidus):.2f} mm"
    )

# 2. corrugations ----------------------------------------------------------------------
print("\n2. Corrugated martensite in annealed MS1500 (dog-bone 30 x 5 x 1 mm)")
base = SheetCase(element=args.element)
show("no track", SheetCase(element=args.element, pattern={"kind": "none"}))
show("straight track", base)
for period, amplitude in ((15.0, 0.5), (15.0, 1.0), (15.0, 1.5), (30.0, 1.5), (10.0, 1.5)):
    pattern = {"kind": "corrugated", "count": 1, "amplitude": amplitude, "period": period}
    show(
        f"corrugated, P {period:g} mm, H {amplitude:g} mm",
        SheetCase(element=args.element, pattern=pattern),
    )

# 3. other choices ---------------------------------------------------------------------
print("\n3. Materials, treatments and configurations")
s430 = Specimen(kind="dogbone", length=30.0, width=10.0, thickness=0.5, grip_width=16.0)
c430 = dict(
    material="aisi430",
    specimen=s430,
    element=args.element,
    beam=FIBRE.with_(power=100.0, absorptivity=0.42),
    bottom=DIODE.with_(absorptivity=0.42),
)
wavy = {"kind": "corrugated", "count": 2, "amplitude": 1.0, "period": 6.0}
show("AISI 430, no track", SheetCase(**c430, pattern={"kind": "none"}))
show("AISI 430, two straight tracks", SheetCase(**c430, pattern={"kind": "straight", "count": 2}))
show("AISI 430, two corrugated tracks", SheetCase(**c430, pattern=wavy))
carb = {"treatment": "carburizing", "alloying": {"coating": 20.0, "efficiency": 0.5}}
show(
    "AISI 430 carburized, two straight tracks",
    SheetCase(**c430, **carb, pattern={"kind": "straight", "count": 2}),
)
show("AISI 430 carburized, two corrugated tracks", SheetCase(**c430, **carb, pattern=wavy))
show(
    "AISI 430 nitrided, two straight tracks",
    SheetCase(
        **c430,
        treatment="nitriding",
        alloying={"efficiency": 0.5},
        pattern={"kind": "straight", "count": 2},
    ),
)
one = {"configuration": "single", "beam": FIBRE.with_(power=250.0), "bottom": None}
show("annealed MS1500, one beam 250 W, straight", SheetCase(element=args.element, **one))
show(
    "annealed MS1500, one beam 250 W, corrugated",
    SheetCase(
        element=args.element,
        **one,
        pattern={"kind": "corrugated", "count": 1, "amplitude": 1.0, "period": 15.0},
    ),
)
across = {"kind": "straight", "angle": 90.0, "count": None, "spacing": 8.0}
show("annealed MS1500, three tracks across", SheetCase(element=args.element, pattern=across))
dp = dict(
    material="dp1180",
    element=args.element,
    beam=FIBRE.with_(power=100.0),
    bottom=DIODE.with_(power=70.0),
)
show("DP1180, no track", SheetCase(**dp, pattern={"kind": "none"}))
show("DP1180, three tempered tracks across", SheetCase(**dp, pattern=across))

# 4. bending ---------------------------------------------------------------------------
print("\n4. Three-point bending of a strip 40 x 10 x 1 mm, span 30 mm, four tracks along it")
strip = Specimen(kind="strip", length=40.0, width=10.0, thickness=1.0)
bend = {"kind": "bending", "span": 30.0, "deflection": 1.0}
four = {"kind": "straight", "count": 4}
show("no track", SheetCase(specimen=strip, loading=bend, pattern={"kind": "none"}))
show("both faces, 150 + 100 W", SheetCase(specimen=strip, loading=bend, pattern=four))
show("upper face, 250 W", SheetCase(specimen=strip, loading=bend, pattern=four, **one))
show(
    "upper face, 150 W",
    SheetCase(
        specimen=strip, loading=bend, pattern=four, configuration="single", beam=FIBRE, bottom=None
    ),
)

# 5. anisotropy ------------------------------------------------------------------------
print("\n5. A corrugated sheet (P 15 mm, H 1 mm) tested at 0, 45 and 90 degrees")
m = (
    SheetCase(
        element=args.element,
        directions=(0.0, 45.0, 90.0),
        pattern={"kind": "corrugated", "count": 1, "amplitude": 1.0, "period": 15.0},
    )
    .evaluate()
    .measures
)
for d in (0, 45, 90):
    print(
        f"  {d:2d}°: Rp {m[f'yield_strength_{d}']:5.0f}  Rm {m[f'tensile_strength_{d}']:5.0f}"
        f"  Ag {100 * m[f'uniform_elongation_{d}']:5.1f} %"
    )
print(f"  anisotropy of Rp {m['yield_anisotropy']:.2f}, of Rm {m['strength_anisotropy']:.2f}")

# 6. optimization ----------------------------------------------------------------------
print(f"\n6. Bayesian optimization ({args.evaluations} evaluations): Rm Ag at Rp >= 400 MPa")
case = SheetCase(
    element=args.element,
    pattern={"kind": "corrugated", "count": 1, "amplitude": 1.0, "period": 15.0},
)
best = optimize(
    case,
    {"pattern.amplitude": [0.0, 2.5], "pattern.period": [6.0, 30.0]},
    constraints={"yield_strength": [400.0, None], "min_radius": [1.5, None]},
    evaluations=args.evaluations,
    seed=1,
)
r = best.result.measures
print(
    f"  A = {best.values['pattern.amplitude']:.2f} mm, P = {best.values['pattern.period']:.2f} "
    f"mm: Rp {r['yield_strength']:.0f} MPa, Rm {r['tensile_strength']:.0f} MPa, Ag "
    f"{100 * r['uniform_elongation']:.1f} %, Rm Ag {r['strength_ductility']:.1f} MPa "
    f"({best.time_s:.0f} s)"
)
