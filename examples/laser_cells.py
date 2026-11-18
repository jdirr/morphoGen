"""Cellular design of laser-architectured sheets: the studies of docs/laser.md, §5 to §8.

1. The six samples of Wang et al. (2023, steel res. int. 94, 2200695): annealed
   MS1500 with one martensitic track (24 %), straight or corrugated, the phases of the
   paper (Hollomon laws) and the fracture of the martensite (critical plastic strain
   averaged over 1.5 mm, fitted to sample 1): yield, strength and failure strain against
   their finite elements and their tests.
2. Design maps (their Fig. 10): the treated fraction (12 to 48 %) against the height
   over the period of the corrugation.
3. The cellular automaton (SCOLASTIC): cells of 0.5 mm, the track drawn by a spot of
   1 mm; neighbour exchanges, periodic and symmetric, tested by finite elements, for
   the failure strain at a tensile strength of at least 700 MPa; then the same from
   the straight track (a stationary design) and with the BESO rule.
4. The continuous automata: the hybrid cellular automaton (Tovar) and density-based
   topology optimization (MMA), with the derivatives of the replays.
5. From cells to laser paths: the centre lines of the designs, laser-treated by the
   process model (fibre and diode beams, metallurgy of the annealed MS1500).

Elements of 0.5 mm. Run time: about five minutes.
"""

import argparse
import time

import numpy as np

from morphogen.laser import SheetCase
from morphogen.laser.cells import (
    CellProblem,
    band_fractions,
    beso,
    constant_area_width,
    corrugation,
    design_tracks,
    hca,
    topology,
    treatable,
    wang_phases,
)

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--element", type=float, default=0.5, help="element size (mm)")
parser.add_argument("--iterations", type=int, default=40, help="of the automaton")
args = parser.parse_args()


def show(design):
    for row in np.asarray(design)[::-1]:
        print("    " + "".join("#" if v else "." for v in row))


# 1. the six samples ---------------------------------------------------------------------
print("1. Wang et al. 2023: 24 % of martensite, yield / strength / failure strain")
problem = CellProblem(phases=wang_phases(), element=args.element, size=1.0)
spec = problem.specimen
tests = {  # sample: P, H, t (mm); their FE (Fig. 5) and tests (Table 4)
    1: (None, 0.0, 1.20, (434, 757, 0.096), (458, 878, 0.083)),
    2: (15, 0.5, 1.19, (385, 755, 0.099), (456, 868, 0.085)),
    3: (15, 1.0, 1.15, (349, 748, 0.101), (424, 860, 0.086)),
    4: (15, 1.5, 1.10, (316, 741, 0.113), (405, 843, 0.103)),
    5: (30, 1.5, 1.16, (327, 757, 0.097), (418, 870, 0.089)),
    6: (10, 1.5, 1.01, (308, 686, 0.132), (393, 778, 0.139)),
}
print("  sample    H/P   here               their FE           tests")
for k, (P, H, t, fe, ex) in tests.items():
    share = band_fractions(problem.mesh, corrugation(spec, H, P), t)
    m = problem.evaluate(fractions=np.stack([1 - share, share], axis=-1)).measures
    hp = 0.0 if P is None else H / P
    print(
        f"  {k}        {hp:5.3f}  {m['yield_strength']:4.0f} {m['tensile_strength']:4.0f} "
        f"{m['failure_strain']:.3f}     {fe[0]} {fe[1]} {fe[2]:.3f}     "
        f"{ex[0]} {ex[1]} {ex[2]:.3f}"
    )

# 2. design maps -------------------------------------------------------------------------
print("\n2. Design maps (P = 10 mm): R_m (MPa) / failure strain")
ratios = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
print("  fraction " + "".join(f"   H/P {r:<5.2f}" for r in ratios))
for vf in (0.12, 0.24, 0.36, 0.48):
    line = f"  {vf:4.2f}     "
    for r in ratios:
        H = r * 10.0
        P = 10.0 if H > 0 else None
        t = constant_area_width(vf * spec.width, H, P)
        share = band_fractions(problem.mesh, corrugation(spec, H, P), t)
        m = problem.evaluate(fractions=np.stack([1 - share, share], axis=-1)).measures
        line += f"  {m['tensile_strength']:4.0f}/{m['failure_strain']:.3f}"
    print(line)

# 3. the cellular automaton --------------------------------------------------------------
print("\n3. Cellular automaton: failure strain at R_m >= 700 MPa, cells of 0.5 mm")
cells = CellProblem(
    phases=wang_phases(),
    element=args.element,
    size=0.5,
    objective="ductility",
    constraints={"tensile_strength": [700.0, None]},
)
grid = cells.grid
start = grid.stamp(corrugation(spec, 1.5, 10.0))  # a spot of 1 mm along P 10, H 1.5
t0 = time.perf_counter()
run = beso(
    cells,
    start,
    budget=None,
    moves="exchange",
    symmetry=("point",),
    period=10.0,
    min_width=2,
    candidates=6,
    iterations=args.iterations,
    patience=8,
)
first, best = run.history[0], run.evaluation.measures
print(
    f"  start: R_m {first['tensile_strength']:.0f} MPa, failure {first['failure_strain']:.3f}"
    f", treated {first['budget']:.2f}"
)
print(
    f"  after {run.evaluations} tests ({time.perf_counter() - t0:.0f} s): R_m "
    f"{best['tensile_strength']:.0f} MPa, failure {best['failure_strain']:.3f}, treatable "
    f"{treatable(run.design)}"
)
show(run.design)
straight = beso(
    cells,
    grid.stamp(corrugation(spec, 0.0, None)),
    budget=None,
    moves="exchange",
    min_width=2,
    candidates=6,
    iterations=5,
)
print(
    f"  from the straight track: {straight.evaluations} tests, failure "
    f"{straight.history[0]['failure_strain']:.3f} -> "
    f"{straight.evaluation.measures['failure_strain']:.3f} (a stationary design)"
)
swap = beso(
    cells, start, budget=None, moves="swap", symmetry=("point",), period=10.0, iterations=10
)
print(
    f"  BESO rule (swaps): {swap.evaluations} tests, failure "
    f"{swap.history[0]['failure_strain']:.3f} -> {swap.evaluation.measures['failure_strain']:.3f}"
)

# 4. continuous automata -----------------------------------------------------------------
print("\n4. Continuous designs with the derivatives of the replays")
grey = start.astype(float)
budget = float(start.mean())
for name, fun in (("HCA", hca), ("topology (MMA)", topology)):
    r = fun(cells, grey, budget=budget, iterations=8)
    m = r.evaluation.measures
    print(
        f"  {name:15s} R_m {m['tensile_strength']:.0f} MPa, failure {m['failure_strain']:.3f}"
        f", treatable {treatable(r.design)}"
    )

# 5. laser paths -------------------------------------------------------------------------
print("\n5. The designs drawn by the laser (process model of the annealed MS1500)")
for name, design in (
    ("straight track", grid.stamp(corrugation(spec, 0.0, None))),
    ("corrugated P 10 H 1.5", start),
    ("automaton", run.design),
):
    tracks = design_tracks(grid, design)
    case = SheetCase(
        element=args.element, pattern={"kind": "polylines", "tracks": [p.tolist() for p in tracks]}
    )
    m = case.evaluate().measures
    print(
        f"  {name:22s} {len(tracks)} track(s): R_p {m['yield_strength']:.0f}, R_m "
        f"{m['tensile_strength']:.0f} MPa, A_g {100 * m['uniform_elongation']:.1f} %, treated "
        f"{m['volume_fraction']:.2f}"
    )
