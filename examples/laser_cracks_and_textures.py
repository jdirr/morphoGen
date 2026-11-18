"""Yld2000-2d textures, the growth of a crack and the anisotropy of an architecture: the
studies of docs/laser.md §10.

1. The texture of a DC04 (Yld2000-2d fitted to its yield stresses and Lankford
   coefficients) on the periodic cell of the sheet: untreated, a track along the
   rolling direction every 24 mm, a sparse grid along the rolling and transverse
   directions; the spreads of r and of the yield strength over 0°-90°.
2. The crack of a compact tension specimen grown through the ligament: untreated, a
   spiral of unbreakable hard walls about its tip, straight hard bands across the
   ligament, brittle (martensite) or unbreakable; the work to a load-line displacement
   of 10 mm and the length of the crack.
3. Sine tracks on an isotropic sheet (the annealed MS1500), pulled in every direction:
   the yield strength and Lankford's coefficient for heights of 0, 1.5 and 3 mm.

Run time: about fifty minutes (the grid of §1 and the cracks of §2).
"""

from dataclasses import replace

import numpy as np

from morphogen.laser import Beam, SheetCase, sheet_material
from morphogen.laser.cells import CellProblem, Phase
from morphogen.laser.materials import FlowLaw
from morphogen.laser.patterns import distance_to_polyline, make_pattern
from morphogen.laser.structures import CompactTension

DIRS = tuple(float(d) for d in range(0, 91, 15))

# 1. a DC04 texture -----------------------------------------------------------------------
print("1. A DC04 texture (Yld2000-2d) on the cell of the sheet")
dc04 = {
    "criterion": "yld2000",
    "s0": 151.0,
    "s45": 166.0,
    "s90": 163.0,
    "sb": 192.0,
    "r0": 1.83,
    "r45": 1.39,
    "r90": 2.11,
    "rb": 0.87,
    "a": 6.0,
}
rolled = replace(sheet_material("annealed_ms1500"), anisotropy=dc04)
for name, pattern, element in (
    ("untreated", {"kind": "none"}, 0.25),
    ("a track along RD every 24 mm", {"kind": "straight", "count": None, "spacing": 24.0}, 0.5),
    (
        "a grid: RD every 35 mm, TD every 52 mm",
        {"kind": "crossed", "angle": 45.0, "rotation": -45.0, "spacing": 35.0, "spacing2": 52.0},
        0.5,
    ),
):
    res = SheetCase(
        material=rolled,
        element=element,
        pattern=pattern,
        loading={"kind": "cell"},
        directions=DIRS,
    ).evaluate()
    m = res.measures
    r = [res.directions[d].measures["r_value"] for d in DIRS]
    rp = [res.directions[d].measures["yield_strength"] for d in DIRS]
    print(
        f"  {name:40s} {m['volume_fraction']:4.2f}  r "
        + "/".join(f"{v:.2f}" for v in r)
        + f" (spread {m['r_anisotropy']:.2f})  R_p "
        + "/".join(f"{v:.0f}" for v in rp)
        + f" (spread {m['yield_anisotropy']:.3f})"
    )

# 2. the growth of a crack ----------------------------------------------------------------
print("\n2. The crack of a CT specimen grown through the ligament (annealed steel)")
ferrite = Phase("ferrite", FlowLaw.hollomon(280.0, 986.0, 0.24), critical=0.5, cost=0.0)
box = (4.0, 20.0, 9.0)
ct = CompactTension(
    max_disp=15.0,
    fine=0.25,
    refine=box,
    propagate=True,
    region=(box[0], box[1], -box[2], box[2]),
)
spiral = (
    make_pattern(
        "spiral",
        (0.0, 20.0, -10.0, 10.0),
        beam=Beam(300.0, 10.0, 0.5),
        center=(10.0, 0.0),
        radius=1.5,
        pitch=3.0,
        turns=2.0,
        angle=20.0,
    )
    .tracks[0]
    .points
)


def walls(C):
    return distance_to_polyline(C.reshape(-1, 2), spiral).reshape(C.shape[:2]) < 0.5


def bands(C):
    near = np.min(np.abs(C[..., :1] - np.array([12.0, 14.5, 17.0])), axis=-1) < 0.5
    return near & (np.abs(C[..., 1]) < 6.0)


for name, critical, design in (
    ("untreated: a straight crack", None, None),
    ("a spiral of unbreakable hard walls", float("inf"), walls),
    ("martensite bands across the ligament", 0.075, bands),
    ("unbreakable hard bands across", float("inf"), bands),
):
    hard = Phase("hard", FlowLaw.hollomon(1300.0, 1966.0, 0.06), critical=critical or 0.075)
    problem = CellProblem(phases=(ferrite, hard), case=ct, size=0.25, fracture_radius=0.5)
    C = problem.grid.centres()
    cells = np.zeros(C.shape[:2], int) if design is None else design(C).astype(int)
    result = problem.evaluate(cells).result
    broken, at = result.fields["broken"]
    keep = result.displacement <= 10.0  # the work up to 10 mm exactly
    force = np.append(result.force[keep], np.interp(10.0, result.displacement, result.force))
    work = np.trapezoid(force, np.append(result.displacement[keep], 10.0)) / 1000
    length = 0.25 * np.sum(at * ct.width <= 10.0)
    print(
        f"  {name:40s} treated {cells.mean():.2f}  F_max {result.force.max() / 1000:.2f} kN"
        f"  work to 10 mm {work:.2f} J  crack {length:.1f} mm"
    )

# 3. sine tracks in every direction ---------------------------------------------------------
print("\n3. Sine tracks (period 10 mm, 5 mm apart) on an isotropic sheet, every 15°")
for height in (0.0, 1.5, 3.0):
    res = SheetCase(
        element=0.25,
        pattern={
            "kind": "corrugated",
            "count": 1,
            "amplitude": height,
            "period": 10.0,
            "spacing": 5.0,
        },
        loading={"kind": "cell", "max_strain": 0.2},
        directions=DIRS,
    ).evaluate()
    rp = [res.directions[d].measures["yield_strength"] for d in DIRS]
    r = [res.directions[d].measures["r_value"] for d in DIRS]
    ag = [res.directions[d].measures["uniform_elongation"] for d in DIRS]
    print(
        f"  H {height:3.1f} mm  R_p "
        + "/".join(f"{v:.0f}" for v in rp)
        + "  r "
        + "/".join(f"{v:.2f}" for v in r)
        + "  A_g "
        + "/".join(f"{100 * v:.1f}" for v in ag)
    )
