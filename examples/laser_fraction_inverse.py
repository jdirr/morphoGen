"""Two checks of the crossed families (docs/laser.md, §17.7).

``--study fraction``: the treated fraction from 0 to 100 %. Crossed families of
martensite (ductile and graded rules) treating 6 to 60 % of the sheet, each over
half-angles of 45 to 65°, and the homogeneous sheets (100 % treated) in every laser
state, ``c`` from 1 to 4.64: the trivial designs, against which an architecture is worth
its complexity only if it does better at the same strength.

``--study inverse``: the phases switched, annealed tracks (24 %) in a martensite
matrix (a hardened sheet softened along tracks): straight tracks along the load and at
45°, corrugated tracks (``H/P`` = 0.25, ``P`` = 20 mm) and crossed families at ±45 to
±65°, ductile and graded rules.

The periodic sheet is pulled along the load with the shear free (elements of 0.5 mm,
0.25 mm from 48 % treated). Writes ``results/laser_fraction`` and
``results/laser_inverse`` (``.csv``, and the curves in ``.npz``).

Usage: python examples/laser_fraction_inverse.py [--study fraction|inverse|all]
[--workers N]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laser_drawing as ld
import laser_geometric_hardening as g
import laser_networks as nw

FRACTIONS = (0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.48, 0.60)
SPREADS = (45.0, 50.0, 55.0, 57.5, 60.0, 62.5, 65.0)
CONTRASTS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, g.MARTENSITE)
RULES = ("ductile", "graded")
FIELDS = (
    "study", "design", "swap", "rule", "contrast", "fraction", "spread", "ratio", "theta",
    "element", "treated", "yield_strength", "tensile_strength", "uniform_elongation",
    "failure_strain", "energy", "seconds",
)  # fmt: skip


def _job(study, design, rule, *, contrast=g.MARTENSITE, fraction=nw.FRACTION, spread=0.0,
         ratio=0.0, theta=0.0, swap=False) -> dict:  # fmt: skip
    element = 0.25 if fraction >= 0.48 and design == "crossed" else 0.5
    return {
        "study": study, "design": design, "swap": swap, "rule": rule,
        "contrast": float(contrast), "fraction": float(fraction), "spread": float(spread),
        "ratio": float(ratio), "theta": float(theta), "element": element,
    }  # fmt: skip


def jobs(study: str) -> list:
    out = []
    if study in ("fraction", "all"):
        for rule in RULES:
            out += [_job("fraction", "uniform", rule, contrast=c, fraction=1.0) for c in CONTRASTS]
            out += [_job("fraction", "crossed", rule, fraction=f, spread=s)
                    for f in FRACTIONS for s in SPREADS]  # fmt: skip
    if study in ("inverse", "all"):
        for rule in RULES:
            out.append(_job("inverse", "family", rule, swap=True))
            out.append(_job("inverse", "family", rule, theta=45.0, swap=True))
            out.append(_job("inverse", "family", rule, ratio=0.25, swap=True))
            out += [_job("inverse", "crossed", rule, spread=s, swap=True)
                    for s in (45.0, 55.0, 60.0, 65.0)]  # fmt: skip
    return out


def evaluate(job: dict) -> dict:
    """The tensile test of one design (see the module)."""
    from morphogen.laser.architectures import track_membrane
    from morphogen.laser.cells import failure_measures
    from morphogen.laser.periodic import PeriodicCell, cell_test, reduce_basis

    t0 = time.perf_counter()
    design = job["design"]
    if design == "uniform":  # the homogeneous sheet: one state everywhere
        a1, a2 = np.array([10.0, 0.0]), np.array([0.0, 5.0])

        def lines(box):
            return [np.array([[box[0] - 5.0, 2.5], [box[1] + 5.0, 2.5]])]

        width = 100.0
    elif design == "crossed":
        s = ld.crossed_spacing(job["fraction"])
        a1, a2 = reduce_basis(*ld.crossed_lattice(job["spread"], s))

        def lines(box):
            return ld.crossed_lines(job["spread"], s, box)

        width = nw.TRACK
    else:
        lx, ly = nw.pattern("family", job["ratio"])["cell"]
        if job["ratio"] == 0.0:
            lx = 2.0  # straight tracks: invariant along x
        a1, a2 = np.array([lx, 0.0]), np.array([0.0, ly])

        def lines(box):
            return nw.lines_over("family", job["ratio"], box)

        width = nw.TRACK
    cell = PeriodicCell.build(a1, a2, element=job["element"], centre=0.5 * (a1 + a2))
    phases = g.phases_of({"law": "contrast", "contrast": job["contrast"], "rule": job["rule"]})
    if job["swap"]:
        phases = phases[::-1]
    mem = track_membrane(cell.mesh, lines(cell.box(width)), width, phases=phases)
    res = cell_test(mem, cell, -job["theta"], max_strain=0.6, failure_margin=0.03,
                    free_shear=True)  # fmt: skip
    fail = res.failure_strain if math.isfinite(res.failure_strain) else math.inf
    m = failure_measures(res.strain, res.stress, fail, young=mem.young)
    row = dict(job)
    share = mem.law[..., 0, 8]
    row["treated"] = float((share * mem.w0).sum() / mem.w0.sum())
    for k in ("yield_strength", "tensile_strength", "uniform_elongation", "failure_strain",
              "energy"):  # fmt: skip
        row[k] = m[k]
    row["curve"] = np.array([res.strain, res.stress])
    row["seconds"] = time.perf_counter() - t0
    return row


def save(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    curves = {f"c{i}": r["curve"] for i, r in enumerate(rows)}
    np.savez_compressed(path.with_suffix(".npz"), **curves)


def load(path: Path) -> list:
    """The rows of a study, numbers parsed and curves attached."""
    rows = []
    csv_path = path.with_suffix(".csv")
    if not csv_path.exists():
        return rows
    text = {"study", "design", "rule"}
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in FIELDS:
                if k == "swap":
                    r[k] = r[k] == "True"
                elif k not in text:
                    r[k] = float(r[k]) if r[k] not in ("", None) else math.nan
            rows.append(r)
    npz = path.with_suffix(".npz")
    if npz.exists():
        data = np.load(npz)
        for i, r in enumerate(rows):
            r["curve"] = data[f"c{i}"]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="all", choices=("fraction", "inverse", "all"))
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()
    for study in ("fraction", "inverse") if args.study == "all" else (args.study,):
        todo = jobs(study)
        t0 = time.perf_counter()
        rows = []
        with ProcessPoolExecutor(args.workers, initializer=g._init) as pool:
            for k, r in enumerate(pool.map(evaluate, todo), start=1):
                rows.append(r)
                print(
                    f"{k}/{len(todo)} {study} {r['design']} swap={r['swap']} {r['rule']} "
                    f"c {r['contrast']:.2f} f {r['fraction']:.2f} ±{r['spread']:g}° "
                    f"H/P {r['ratio']:g} θ {r['theta']:g}: Rp {r['yield_strength']:.0f}, "
                    f"Rm {r['tensile_strength']:.0f}, Ag {100 * r['uniform_elongation']:.1f}, "
                    f"fail {100 * r['failure_strain']:.1f}, E {r['energy']:.0f} "
                    f"({r['seconds']:.0f} s; {time.perf_counter() - t0:.0f} s)",
                    flush=True,
                )
        save(rows, Path(f"results/laser_{study}"))
        print("written:", Path(f"results/laser_{study}.csv"), flush=True)


if __name__ == "__main__":
    main()
