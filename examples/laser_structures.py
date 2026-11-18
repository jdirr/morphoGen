"""Load cases beyond the tensile test and alternative materials: the studies of
docs/laser.md, §10 and §11.

1. A compact tension specimen (W 20 mm, a/W 0.5, 1 mm): hard martensitic bands in the
   annealed steel, ahead of the crack or along it — the initiation of the crack at its
   tip (a critical plastic strain of each phase, averaged over 0.5 mm).
2. The same specimen in hard sheets with laser-softened bands: DP1180 tempered, and
   AA6082-T6 softened (the crack tip in a soft band, soft bands ahead or along).
3. An L-shaped plate (40 mm, arms 16 mm) held along its top edge and pulled down at the
   end of its arm: the limit load with hard tracks, and the BESO automaton for it.
4. The plastic anisotropy of treated sheets: yield strength, tensile strength and
   Lankford's coefficient at 0°, 45° and 90°, Hill's 1948 coefficients; the angle and
   spacing of crossed tracks optimized for an isotropic yield strength.
5. A rolled sheet (the Lankford coefficients of a deep-drawing steel given to the
   annealed MS1500, Hill's 1948 criterion) compensated by the treatment, on the
   periodic cell of the sheet treated all over: a narrow specimen against the cell,
   families of tracks along the rolling direction, honeycombs, the texture kept by the
   tracks or not.
6. Alternative materials through the whole chain: 22MnB5 hardened, AA6082-T6 softened.

Run time: about half an hour (the profiles of §5 on their cells).
"""

from dataclasses import replace

import numpy as np

from morphogen.laser import Beam, SheetCase, Specimen, optimize, sheet_material
from morphogen.laser.cells import CellProblem, Phase, beso
from morphogen.laser.materials import LAWS, FlowLaw
from morphogen.laser.structures import CompactTension, LPlate

FERRITE = Phase("ferrite", FlowLaw.hollomon(280.0, 986.0, 0.24), critical=0.5, cost=0.0)
MARTENSITE = Phase("martensite", FlowLaw.hollomon(1300.0, 1966.0, 0.06), critical=0.075)


def initiation(prob, design, ct=None):
    """Force (kN), load-line displacement (mm), energy (J) at initiation and where."""
    ev = prob.evaluate(design.astype(int))
    m, r = ev.measures, ev.result
    f = np.interp(m["failure_strain"], r.strain, r.force) / 1000
    case = prob.case
    lld = m["failure_strain"] * case.length
    energy = m["energy"] * case.section * case.length / 1000
    mem = prob.membrane(ev.fractions)
    pe = r.fields["plastic_strain"]
    ratio = mem.fracture.ratios(np.repeat(pe[:, :, None], mem.layers, axis=2))
    k = int(np.argmax(ratio))
    phase = min(prob.phases, key=lambda p: abs(p.critical - mem.fracture.element_critical[k]))
    return f, lld, energy, r.force.max() / 1000, mem.centroids[k], phase.name, m["budget"]


# 1. compact tension, hard bands -------------------------------------------------------
print("1. Compact tension of the annealed steel, hard bands (treated: share of the 12 x 10 mm")
print("   box around the tip); initiation force, load-line displacement, energy")
ct = CompactTension(max_disp=5.0)
prob = CellProblem(phases=(FERRITE, MARTENSITE), case=ct, size=0.25, fracture_radius=0.5)
a = ct.crack * ct.width
C = prob.grid.centres()
X, Y = C[..., 0] - a, C[..., 1]
for name, mask in (
    ("none", np.zeros_like(X, bool)),
    ("band 1 mm, 1 mm ahead", (X >= 1) & (X < 2)),
    ("band 2 mm, 1.5 mm ahead", (X >= 1.5) & (X < 3.5)),
    ("band 1 mm, 2 mm ahead", (X >= 2) & (X < 3)),
    ("band 1 mm, 4 mm ahead", (X >= 4) & (X < 5)),
    ("barrier 1 mm, 2 mm ahead, |y| < 2.5", (X >= 2) & (X < 3) & (np.abs(Y) < 2.5)),
    ("bands along the crack, y = ±1 mm", np.abs(np.abs(Y) - 1) < 0.5),
):
    f, lld, en, fmax, c, ph, b = initiation(prob, mask)
    print(
        f"  {name:38s} {b:4.2f}  {f:5.2f} kN {lld:5.2f} mm {en:5.2f} J  "
        f"at ({c[0] - a:+.1f}, {c[1]:+.1f}) mm of the tip, {ph}"
    )

# 2. hard sheets, soft bands -----------------------------------------------------------
print("\n2. Compact tension of hard sheets with softened bands")
al = sheet_material("aa6082_t6")


def table_law(material, hv):
    kind, s0, k, n = material.strength.flow(hv)
    return FlowLaw(LAWS[int(kind)], float(s0), float(k), float(n))


cases = {
    "DP1180, tempered bands": (
        Phase("DP1180", FlowLaw.from_tensile(850.0, 1200.0, 0.05), critical=0.25, cost=0.0),
        Phase("tempered", FlowLaw.from_tensile(600.0, 816.0, 0.08), critical=0.40),
        {},
    ),
    "AA6082-T6, softened bands": (
        Phase("T6", table_law(al, 105.0), critical=0.20, cost=0.0),
        Phase("softened", table_law(al, 60.0), critical=0.35),
        {"young": 70000.0, "poisson": 0.33},
    ),
}
for title, (hard, soft, elastic) in cases.items():
    print(f"  {title}")
    ct = CompactTension(max_disp=3.0)
    prob = CellProblem(phases=(hard, soft), case=ct, size=0.25, fracture_radius=0.5, **elastic)
    for name, mask in (
        ("none", np.zeros_like(X, bool)),
        ("soft band 1.5 mm across the tip", (X >= -0.75) & (X < 0.75)),
        ("soft band 1.5 mm, 2 mm ahead", (X >= 2) & (X < 3.5)),
        ("soft bands along the crack, y = ±2 mm", np.abs(np.abs(Y) - 2) < 0.75),
    ):
        f, lld, en, fmax, c, ph, b = initiation(prob, mask)
        print(f"    {name:38s} {b:4.2f}  {f:5.2f} kN {lld:5.2f} mm {en:5.3f} J  ({ph})")

# 3. the L-shaped plate ----------------------------------------------------------------
print("\n3. L-shaped plate: limit load (tip pulled down 6 mm)")
lp = LPlate(max_disp=6.0, region=(0.0, 34.0, 0.0, 36.0))  # without the grips' zones
prob = CellProblem(
    phases=(FERRITE, MARTENSITE),
    case=lp,
    size=1.0,
    fracture_radius=0.5,
    objective="tensile_strength",
)
C = prob.grid.centres()
X, Y = C[..., 0], C[..., 1]
b = lp.arm
inside = ~((X > b) & (Y > b))
designs = {
    "none": np.zeros_like(X, bool),
    "outer edges": ((Y < 1) | (X < 1)) & inside,
    "inner edges (the corner)": (
        ((Y > b - 1) & (Y < b) & (X > b - 1)) | ((X > b - 1) & (X < b) & (Y > b - 1))
    ),
    "corner block 4 mm": (np.abs(X - b) < 4) & (np.abs(Y - b) < 4) & inside,
}
for name, mask in designs.items():
    f, d, en, fmax, c, ph, bud = initiation(prob, mask)
    broken = d < 0.999 * lp.max_disp
    print(
        f"  {name:28s} {bud:4.2f}  limit {fmax:5.3f} kN"
        + (f", {ph} breaks at {d:.1f} mm near ({c[0]:.0f}, {c[1]:.0f})" if broken else "")
    )
run = beso(
    prob, designs["outer edges"].astype(int), budget=None, moves="swap", iterations=12, patience=6
)
print(
    f"  BESO from the outer edges ({run.evaluations} tests): limit "
    f"{run.evaluation.result.force.max() / 1000:.3f} kN at "
    f"{run.evaluation.measures['budget']:.2f}"
)
for row in run.design[::-1][::2]:
    print("    " + "".join("#" if v else "." for v in row[::1]))

# 4. anisotropy ------------------------------------------------------------------------
print("\n4. Anisotropy of treated sheets (annealed MS1500, tests at 0/45/90°): specimens")
print("   5 mm wide against the sheet itself (its periodic cell)")
for name, pat in (
    ("one family along 0°, spacing 2.5", {"kind": "straight", "count": None, "spacing": 2.5}),
    ("crossed ±30°, spacing 5", {"kind": "crossed", "angle": 30.0, "spacing": 5.0}),
    ("crossed ±45°, spacing 5", {"kind": "crossed", "angle": 45.0, "spacing": 5.0}),
    ("honeycomb", {"kind": "honeycomb"}),
):
    spec = SheetCase(element=0.5, pattern=pat, directions=(0.0, 45.0, 90.0)).evaluate().measures
    m = (
        SheetCase(element=0.25, pattern=pat, loading={"kind": "cell"}, directions=(0.0, 45.0, 90.0))
        .evaluate()
        .measures
    )
    print(
        f"  {name:32s} r {spec['r_value_0']:.2f}/{spec['r_value_45']:.2f}/"
        f"{spec['r_value_90']:.2f} (specimens), {m['r_value_0']:.2f}/{m['r_value_45']:.2f}/"
        f"{m['r_value_90']:.2f} (sheet)  r̄ {m['r_bar']:.2f}  Δr {m['delta_r']:+.2f}  Rp "
        f"{m['yield_strength_0']:.0f}/{m['yield_strength_45']:.0f}/"
        f"{m['yield_strength_90']:.0f}  Hill F G H N {m['hill_f']:.2f} {m['hill_g']:.2f} "
        f"{m['hill_h']:.2f} {m['hill_n']:.2f}"
    )
best = optimize(
    SheetCase(
        element=0.5,
        pattern={"kind": "crossed", "angle": 45.0, "spacing": 5.0},
        loading={"kind": "cell"},
        directions=(0.0, 45.0, 90.0),
    ),
    {"pattern.angle": [15.0, 75.0], "pattern.spacing": [3.0, 9.0]},
    objective="isotropy",
    constraints={"yield_strength": [380.0, None]},
    evaluations=12,
    seed=1,
)
m = best.result.measures
print(
    f"  isotropic yield of the sheet (Bayesian, 12 evaluations): ±"
    f"{best.values['pattern.angle']:.0f}°, spacing {best.values['pattern.spacing']:.1f} mm: "
    f"Rp {m['yield_strength_0']:.0f}/{m['yield_strength_45']:.0f}/"
    f"{m['yield_strength_90']:.0f} (spread {m['yield_anisotropy']:.2f}), treated "
    f"{m['volume_fraction']:.2f}"
)

# 5. a rolled sheet compensated -----------------------------------------------------------
print("\n5. A rolled sheet (r 1.6 / 1.2 / 2.0) compensated by the tracks")


def rolled(memory=0.0):
    texture = {"r0": 1.6, "r45": 1.2, "r90": 2.0, "memory": memory}
    return replace(sheet_material("annealed_ms1500"), anisotropy=texture)


def anisotropy(name, pattern, cell=True, memory=0.0, element=0.25, profile=False):
    """The anisotropy of the rolled sheet treated by a pattern: at 0°, 45° and 90°, or over
    the whole profile (every 15°)."""
    dirs = tuple(float(d) for d in range(0, 91, 15)) if profile else (0.0, 45.0, 90.0)
    case = SheetCase(
        material=rolled(memory),
        element=element if cell else 0.5,
        pattern=pattern,
        loading={"kind": "cell"} if cell else {},
        directions=dirs,
    )
    m = case.evaluate().measures
    if profile:
        r = [m[f"r_value_{d:g}"] for d in dirs]
        print(
            f"  {name:44s} {m['volume_fraction']:4.2f}  r(0..90°) "
            + "/".join(f"{v:.2f}" for v in r)
            + f"  spread {m['r_anisotropy']:.2f}  r̄ {m['r_bar']:.2f}"
        )
        return
    print(
        f"  {name:44s} {m['volume_fraction']:4.2f}  r {m['r_value_0']:.2f}/"
        f"{m['r_value_45']:.2f}/{m['r_value_90']:.2f}  r̄ {m['r_bar']:.2f}  Δr "
        f"{m['delta_r']:+.2f}  spread {m['r_anisotropy']:.2f}  R_p {m['yield_strength_0']:.0f}/"
        f"{m['yield_strength_45']:.0f}/{m['yield_strength_90']:.0f}  A_g "
        f"{100 * min(m[f'uniform_elongation_{d}'] for d in (0, 45, 90)):.1f} %"
    )


family = {"kind": "straight", "angle": 0.0, "count": None, "spacing": 5.57}
anisotropy("untreated (the cell)", {"kind": "none"})
anisotropy("tracks along RD, 5.6 mm: specimen 30 x 5 mm", family, cell=False)
anisotropy("the same: the sheet (periodic cell)", family)
for spacing in (10.0, 14.0, 20.0):
    anisotropy(f"tracks along RD, {spacing:g} mm", {**family, "spacing": spacing})
anisotropy(
    "tracks along RD, 14 mm, the tracks keep the texture", {**family, "spacing": 14.0}, memory=1.0
)
for side in (3.0, 5.0):
    anisotropy(f"honeycomb of side {side:g} mm", {"kind": "honeycomb", "cell": side}, element=0.125)
anisotropy("square grid turned 45°, 10 mm", {"kind": "crossed", "angle": 45.0, "spacing": 10.0})
print("  the whole profile, every 15°:")
grid = {"kind": "crossed", "angle": 45.0, "rotation": -45.0}  # families along RD and TD
anisotropy("untreated", {"kind": "none"}, profile=True)
anisotropy("tracks along RD, 20 mm", {**family, "spacing": 20.0}, profile=True)
anisotropy(
    "honeycomb of side 4 mm", {"kind": "honeycomb", "cell": 4.0}, element=0.125, profile=True
)
anisotropy(
    "tracks along RD every 24 mm, along TD every 39 mm (optimized)",
    {**grid, "spacing": 24.0, "spacing2": 39.0},
    element=0.5,  # a cell of 24 x 39 mm (0.25 mm: spread 0.20 instead of 0.21)
    profile=True,
)

# 6. alternative materials -------------------------------------------------------------
print("\n6. Alternative materials, the whole chain")
spec = Specimen(thickness=1.5)
fibre = Beam(200.0, speed=10.0, radius=0.5, absorptivity=0.4)
diode = Beam(150.0, speed=10.0, radius=0.6, absorptivity=0.4, profile="tophat")
for name, pat in (
    ("none", {"kind": "none"}),
    ("straight", {"kind": "straight", "count": 1}),
    ("corrugated P 10 H 1.5", {"kind": "corrugated", "count": 1, "amplitude": 1.5, "period": 10.0}),
):
    m = (
        SheetCase(
            material="22mnb5", specimen=spec, beam=fibre, bottom=diode, element=0.5, pattern=pat
        )
        .evaluate()
        .measures
    )
    print(
        f"  22MnB5 1.5 mm, {name:22s} Rp {m['yield_strength']:.0f} Rm "
        f"{m['tensile_strength']:.0f} Ag {100 * m['uniform_elongation']:.1f} % treated "
        f"{m['volume_fraction']:.2f}"
    )
beam = Beam(260.0, speed=20.0, radius=0.75, absorptivity=0.3)
for name, pat in (
    ("none", {"kind": "none"}),
    ("straight", {"kind": "straight", "count": 1}),
    ("corrugated P 10 H 1.5", {"kind": "corrugated", "count": 1, "amplitude": 1.5, "period": 10.0}),
):
    m = (
        SheetCase(material="aa6082_t6", beam=beam, bottom=beam, element=0.5, pattern=pat)
        .evaluate()
        .measures
    )
    print(
        f"  AA6082-T6 1 mm, {name:20s} Rp {m['yield_strength']:.0f} Rm "
        f"{m['tensile_strength']:.0f} Ag {100 * m['uniform_elongation']:.1f} % softened "
        f"{m['volume_fraction']:.2f}"
        + (f" (peak {m['peak_temperature']:.0f} °C)" if m["volume_fraction"] > 0 else "")
    )
