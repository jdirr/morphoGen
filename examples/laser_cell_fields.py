"""Snapshots of the periodic cell (docs/laser.md, §17.5): the equivalent plastic strain
and the deformed mesh of corrugated tracks (``H/P`` = 0.25, ``P`` = 20 mm) at 10, 20 and
30 % of strain and of crossed families (±60°) at 10, 25 and 40 %, 24 % treated, ductile
martensite, and of the phases switched (``inverse``: annealed crossed tracks in a
martensite matrix) at 3 and 10 %, pulled along the load with the shear free, on elements
of 0.25 mm (a test stops when the load has fallen by 5 % after its maximum). Writes
``results/laser_cell_fields.npz`` (keys ``<design>_<strain>__<field>``)."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laser_drawing as ld
import laser_geometric_hardening as g
import laser_networks as nw

OUT = Path("results/laser_cell_fields.npz")
SNAPSHOTS = (
    ("corrugated", (0.10, 0.20, 0.30)),
    ("crossed", (0.10, 0.25, 0.40)),
    ("inverse", (0.03, 0.10)),
)


def run(job: dict) -> tuple[dict, dict]:
    from morphogen.laser.architectures import track_membrane
    from morphogen.laser.periodic import PeriodicCell, cell_test, reduce_basis

    t0 = time.perf_counter()
    design, strain = job["design"], job["strain"]
    if design in ("crossed", "inverse"):
        s = ld.crossed_spacing()
        a1, a2 = reduce_basis(*ld.crossed_lattice(60.0, s))

        def lines(box):
            return ld.crossed_lines(60.0, s, box)

    else:
        lx, ly = nw.pattern("family", 0.25)["cell"]
        a1, a2 = np.array([lx, 0.0]), np.array([0.0, ly])

        def lines(box):
            return nw.lines_over("family", 0.25, box)

    cell = PeriodicCell.build(a1, a2, element=0.25, centre=0.5 * (a1 + a2))
    phases = g.phases_of({"law": "contrast", "contrast": g.MARTENSITE, "rule": "ductile"})
    if design == "inverse":
        phases = phases[::-1]
    mem = track_membrane(cell.mesh, lines(cell.box(nw.TRACK)), nw.TRACK, phases=phases)
    res = cell_test(
        mem, cell, 0.0, max_strain=strain, failure_margin=0.03, keep_fields=True, free_shear=True
    )
    f = res.fields
    nodes, quads = f["mesh"]
    out = {
        "nodes": np.asarray(nodes),
        "quads": np.asarray(quads),
        "u": np.asarray(f["displacement"]),
        "p": np.asarray(f["plastic_strain"]).mean(axis=1),
        "share": mem.law[..., 0, 8].mean(axis=1),
        "a1": cell.a1,
        "a2": cell.a2,
        "macro": np.array([f["macroscopic"][k][-1] for k in ("a", "b", "c")]),
        "strain": float(res.strain[-1]),
    }
    print(
        f"{design} {strain:g}: reached {res.strain[-1]:.3f} ({time.perf_counter() - t0:.0f} s)",
        flush=True,
    )
    return job, out


def main() -> None:
    jobs = [{"design": d, "strain": e} for d, es in SNAPSHOTS for e in es]
    data = {}
    with ProcessPoolExecutor(len(jobs), initializer=g._init) as pool:
        for job, out in pool.map(run, jobs):
            tag = f"{job['design']}_{round(100 * job['strain'])}"
            data.update({f"{tag}__{k}": v for k, v in out.items()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **data)
    print("written:", OUT)


if __name__ == "__main__":
    main()
