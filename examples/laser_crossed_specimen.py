"""Two families of straight tracks crossed at ±θ about the load in a dog-bone
(docs/laser.md, §17): the periodic sheet against a specimen whose grips restrain the
shear and whose edges cut the tracks.

A dog-bone cut from the sheet (the gauge 44 mm wide, 90 or 180 mm long, as in §16.4)
with two families of straight tracks 1 mm wide at ±θ from the load, treating 24 %
together (``laser_drawing.crossed_spacing``); the as-quenched martensite under the rules
``ductile`` and ``graded``, the tempered state of contrast 3. Writes
``results/laser_crossed_specimen.csv``.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import laser_drawing as ld
import laser_geometric_hardening as g

FIELDS = (
    "spread",
    "state",
    "length",
    "treated",
    "yield_strength",
    "tensile_strength",
    "uniform_elongation",
    "failure_strain",
    "energy",
    "seconds",
)


def evaluate(job: dict) -> dict:
    from morphogen.laser.architectures import track_membrane
    from morphogen.laser.cells import failure_measures
    from morphogen.laser.mechanics import tensile_test
    from morphogen.laser.specimen import Specimen

    t0 = time.perf_counter()
    width = 44.0
    spec = Specimen(
        kind="dogbone",
        length=job["length"],
        width=width,
        radius=20.0,
        grip_width=1.5 * width,
        grip_length=15.0,
    )
    mesh = spec.mesh(0.5)
    lo, hi = mesh.nodes.min(axis=0) - 5.0, mesh.nodes.max(axis=0) + 5.0
    lines = ld.crossed_lines(job["spread"], ld.crossed_spacing(), (lo[0], hi[0], lo[1], hi[1]))
    contrast, rule = ld.STATES[job["state"]]
    phases = g.phases_of({"law": "contrast", "contrast": contrast, "rule": rule})
    mem = track_membrane(mesh, lines, 1.0, phases=phases)
    res = tensile_test(spec, mem, max_strain=0.6, failure_margin=0.03)
    fail = res.failure_strain if math.isfinite(res.failure_strain) else math.inf
    m = failure_measures(res.strain, res.stress, fail, young=mem.young)
    share = mem.law[..., 0, 8]
    gx = mem.points[..., 0]
    gauge = ((gx >= 0) & (gx <= spec.length)).astype(float)
    row = dict(job)
    for k in ("yield_strength", "tensile_strength", "uniform_elongation", "failure_strain"):
        row[k] = m[k]
    row["energy"] = m["energy"]
    row["treated"] = float((share * mem.w0 * gauge).sum() / (mem.w0 * gauge).sum())
    row["seconds"] = time.perf_counter() - t0
    return row


def main() -> None:
    jobs = [
        {"spread": a, "state": st, "length": L}
        for a in (60.0,)
        for st in ("ductile", "as-quenched", "tempered")
        for L in (90.0, 180.0)
    ]
    with ProcessPoolExecutor(len(jobs), initializer=g._init) as pool:
        rows = list(pool.map(evaluate, jobs))
    out = Path("results/laser_crossed_specimen.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"±{r['spread']:g}° {r['state']} {r['length']:g} mm, treated {r['treated']:.3f}: "
            f"Rm {r['tensile_strength']:.0f}, Ag {100 * r['uniform_elongation']:.1f}, "
            f"fail {100 * r['failure_strain']:.1f}, E {r['energy']:.0f} ({r['seconds']:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
