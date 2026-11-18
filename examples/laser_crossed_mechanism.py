"""How two crossed families of straight tracks harden a sheet late (docs/laser.md, §17):
the mean stress along the tracks of one family, and the stretch of their direction,
against the strain of the sheet pulled along the bisector of the families.

The periodic sheet of ``laser_drawing._tension`` (24 % of ductile martensite, two
families at ±θ about the load); each Gauss point within half a track of a line of the
family at +θ counts for that family (the crossings with the other family excluded).
Writes ``results/laser_crossed_mechanism.npz``: for each half-angle, the strain, the
nominal stress, the mean stress along the family (MPa), that of the matrix along the
load, and the stretch of the direction of the family (from the mean gradient).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import laser_drawing as ld
import laser_geometric_hardening as g


def run(spread: float, state: str = "ductile") -> dict:
    from morphogen.laser.architectures import track_membrane
    from morphogen.laser.periodic import PeriodicCell, cell_test, reduce_basis

    s = ld.crossed_spacing()
    a1, a2 = reduce_basis(*ld.crossed_lattice(spread, s))
    cell = PeriodicCell.build(a1, a2, element=0.5, centre=0.5 * (a1 + a2))
    contrast, rule = ld.STATES[state]
    phases = g.phases_of({"law": "contrast", "contrast": contrast, "rule": rule})
    lines = ld.crossed_lines(spread, s, cell.box(1.0))
    mem = track_membrane(cell.mesh, lines, 1.0, phases=phases)
    th = math.radians(spread)
    e_p = np.array([math.cos(th), math.sin(th)])
    n_p = np.array([-math.sin(th), math.cos(th)])
    n_m = np.array([math.sin(th), math.cos(th)])
    pts = mem.points  # (elements, gauss, 2)

    def inside(normal):
        u = (pts @ normal) / s - 0.5
        return np.abs(u - np.round(u)) * s < 0.5  # within half a track of a line

    plus = inside(n_p) & ~inside(n_m)
    matrix = mem.law[..., 0, 8] < 1e-6
    w = mem.w0

    def along(mask, d):
        def f(_q):
            sig = mem.sig_n.mean(axis=2)  # over the layers: xx, yy, xy
            sdd = sig[..., 0] * d[0] ** 2 + sig[..., 1] * d[1] ** 2 + 2 * sig[..., 2] * d[0] * d[1]
            return float((sdd * w * mask).sum() / (w * mask).sum())

        return f

    res = cell_test(
        mem,
        cell,
        0.0,
        max_strain=0.6,
        failure_margin=0.03,
        monitors={
            "track": along(plus, e_p),
            "matrix": along(matrix, np.array([1.0, 0.0])),
        },
    )
    mon = res.fields["monitors"]
    mac = res.fields["macroscopic"]
    a, b, c = (np.asarray(mac[k]) for k in ("a", "b", "c"))
    # the stretch of the direction of the family under the mean gradient I + H
    F11, F22, F12 = 1.0 + a, 1.0 + b, c
    lam = np.sqrt((F11 * e_p[0] + F12 * e_p[1]) ** 2 + (F12 * e_p[0] + F22 * e_p[1]) ** 2)
    return {
        "strain": np.asarray(res.strain),
        "stress": np.asarray(res.stress),
        "track": np.asarray(mon["track"]),
        "matrix": np.asarray(mon["matrix"]),
        "stretch": lam,
    }


def main() -> None:
    g._init()
    out = {}
    for spread in (45.0, 60.0, 70.0):
        r = run(spread)
        for k, v in r.items():
            out[f"{k}_{spread:g}"] = v
        k = int(np.argmax(r["stress"]))
        print(
            f"±{spread:g}°: Rm {r['stress'][k]:.0f} at {100 * r['strain'][k]:.1f} %, "
            f"track stress from {r['track'].min():.0f} to {r['track'].max():.0f} MPa, "
            f"least stretch {r['stretch'].min():.4f} at "
            f"{100 * r['strain'][int(np.argmin(r['stretch']))]:.1f} %",
            flush=True,
        )
    np.savez_compressed("results/laser_crossed_mechanism.npz", **out)
    print("written: results/laser_crossed_mechanism.npz")


if __name__ == "__main__":
    main()
