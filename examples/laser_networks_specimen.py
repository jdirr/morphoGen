"""Straight tracks at 45° to the load in a dog-bone: the periodic sheet shears freely
(docs/laser.md, §16), the grips of a specimen do not.

A dog-bone cut from the sheet (the gauge 44 mm wide and 90 mm long, as the eight-track
specimens of §15.7) with straight tracks 1 mm wide treating 24 % of it, along the load
(0°) and at 45°; the as-quenched martensite under the rules ``ductile`` and ``graded``.
Writes ``results/laser_networks_specimen.csv``.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import laser_geometric_hardening as g

FRACTION = 0.24
FIELDS = (
    "angle",
    "rule",
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
    th = math.radians(job["angle"])
    e, n = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    s = 1.0 / FRACTION  # the spacing of straight tracks 1 mm wide
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]]])
    off, along = corners @ n, corners @ e
    lines = []
    for k in range(math.floor(off.min() / s) - 1, math.ceil(off.max() / s) + 2):
        base = (k + 0.5) * s * n
        lines.append(np.array([base + along.min() * e, base + along.max() * e]))
    phases = g.phases_of({"law": "contrast", "contrast": g.MARTENSITE, "rule": job["rule"]})
    mem = track_membrane(mesh, lines, 1.0, phases=phases)
    res = tensile_test(spec, mem, max_strain=0.5, failure_margin=0.03)
    fail = res.failure_strain if math.isfinite(res.failure_strain) else math.inf
    m = failure_measures(res.strain, res.stress, fail, young=mem.young)
    share = mem.law[..., 0, 8]
    gx = mem.points[..., 0]
    gauge = ((gx >= 0) & (gx <= spec.length)).astype(float)
    row = dict(job)
    row.update(
        {
            k: m[k]
            for k in (
                "yield_strength",
                "tensile_strength",
                "uniform_elongation",
                "failure_strain",
                "energy",
            )
        }
    )
    row["treated"] = float((share * mem.w0 * gauge).sum() / (mem.w0 * gauge).sum())
    row["seconds"] = time.perf_counter() - t0
    return row


def main() -> None:
    jobs = [
        {"angle": a, "rule": r, "length": L}
        for a in (0.0, 45.0)
        for r in ("ductile", "graded")
        for L in (90.0, 180.0)
    ]
    with ProcessPoolExecutor(len(jobs), initializer=g._init) as pool:
        rows = list(pool.map(evaluate, jobs))
    out = Path("results/laser_networks_specimen.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            r["angle"],
            r["rule"],
            r["length"],
            f"treated {r['treated']:.3f}",
            f"Ag {100 * r['uniform_elongation']:.1f} fail {100 * r['failure_strain']:.1f}",
            f"E {r['energy']:.0f} Rm {r['tensile_strength']:.0f} ({r['seconds']:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
