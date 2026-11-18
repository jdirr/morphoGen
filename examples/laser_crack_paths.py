"""Architectured sheets by localized laser treatment: steering a ductile crack with a
design treated pixel by pixel (docs/laser.md, §14).

A compact tension specimen (W = 30 mm, a/W = 0.5) of the annealed 0.23 C steel whose
crack grows by the breaking of the elements (the averaged plastic strain of a phase
at its critical strain: 0.5 for the matrix at a crack tip, as in §10), a design of
pixels of 1 mm over the ligament; the hard phase tough (a tempered martensite,
``contrast_phases(3)``: critical strain 0.24), brittle (the as-quenched martensite,
0.072) or unbreakable. Designs: soft corridors in a hard sheet and corridors between
hard walls in the untreated sheet, inclined to the crack, straight or zigzag; a hard
band across the path at 45°; staggered hard bricks (nacre); a network of hard walls
(a Voronoi tessellation). Measures: the work to separation, the largest force, the
length and the deviation of the crack, and the share of it that runs in the corridor.

Elements of 0.5 mm over the ligament, the designs in parallel: about two minutes.
"""

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

W, A = 30.0, 15.0  # the width and the crack length (mm)
BOX = (13.0, 30.0, 12.0)  # the fine box: x0, x1, |y| (mm)
PIXEL = 1.0


def corridor_distance(X, Y, angle: float, start=(A, 0.0)):
    """The distance to the ray from the crack tip at ``angle`` degrees, and the
    abscissa along it."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    dx, dy = X - start[0], Y - start[1]
    along = dx * c + dy * s
    across = np.abs(-dx * s + dy * c)
    return across, along


def zigzag_distance(X, Y, amplitude=4.0, period=8.0):
    """The distance to a zigzag line from the crack tip (a triangle wave along x)."""
    x = X - A
    tri = 2.0 / math.pi * np.arcsin(np.sin(2.0 * math.pi * x / period))  # from 0, ±1
    return np.abs(Y - np.where(x < 0, 0.0, amplitude * tri))


def voronoi_walls(X, Y, cell=4.0, seed=3):
    """A network of walls one pixel wide between the cells of a Poisson–Voronoi
    tessellation of mean size ``cell`` (mm): the pixels whose nearest seed differs from
    a neighbour's."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    x0, x1, y1 = BOX
    n = int((x1 - x0 + 8) * (2 * y1 + 8) / cell**2)
    seeds = np.column_stack([rng.uniform(x0 - 4, x1 + 4, n), rng.uniform(-y1 - 4, y1 + 4, n)])
    lab = cKDTree(seeds).query(np.column_stack([X.ravel(), Y.ravel()]))[1].reshape(X.shape)
    wall = np.zeros(X.shape, bool)
    wall[:-1] |= lab[:-1] != lab[1:]
    wall[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return wall


def bricks(X, Y, length=4, height=2, mortar=1):
    """Staggered hard bricks of ``length`` x ``height`` pixels in a mortar one pixel
    wide (nacre): row ``k`` shifted by half a brick."""
    i = np.floor((Y + BOX[2]) / PIXEL).astype(int)
    j = np.floor((X - BOX[0]) / PIXEL).astype(int)
    row = i // (height + mortar)
    in_row = (i % (height + mortar)) < height
    shift = (row % 2) * ((length + mortar) // 2)
    in_brick = ((j + shift) % (length + mortar)) < length
    return in_row & in_brick


def designs(X, Y):
    """The designs: ``name -> (cells, hard phase, corridor mask or None)``."""
    out = {"untreated": (np.zeros(X.shape, int), "tough", None)}
    out["fully treated, tough"] = (np.ones(X.shape, int), "tough", None)
    for angle in (30.0, 45.0):
        across, along = corridor_distance(X, Y, angle)
        soft = (across < 1.5) & (along > -2.0)
        out[f"soft corridor at {angle:.0f} deg in a tough hard sheet"] = (
            (~soft).astype(int),
            "tough",
            soft,
        )
        out[f"soft corridor at {angle:.0f} deg in a brittle hard sheet"] = (
            (~soft).astype(int),
            "brittle",
            soft,
        )
        wall = (across >= 1.5) & (across < 3.5) & (along > -1.0)
        out[f"tough hard walls along a corridor at {angle:.0f} deg"] = (
            wall.astype(int),
            "tough",
            (across < 1.5) & (along > -2.0),
        )
        out[f"brittle hard walls along a corridor at {angle:.0f} deg"] = (
            wall.astype(int),
            "brittle",
            (across < 1.5) & (along > -2.0),
        )
    zz = zigzag_distance(X, Y)
    soft = (zz < 1.5) & (X > A - 2.0)
    out["zigzag soft corridor in a tough hard sheet"] = ((~soft).astype(int), "tough", soft)
    across, along = corridor_distance(X, Y, -45.0, start=(A + 4.0, 4.0))
    band = (across < 1.0) & (np.abs(along) < 7.0)
    out["tough hard band at 45 deg ahead of the tip"] = (band.astype(int), "tough", None)
    out["unbreakable hard band at 45 deg ahead of the tip"] = (
        band.astype(int),
        "unbreakable",
        None,
    )
    br = bricks(X, Y) & (X > A - 1.0)
    out["staggered tough bricks (nacre)"] = (br.astype(int), "tough", None)
    out["staggered brittle bricks (nacre)"] = (br.astype(int), "brittle", None)
    vw = voronoi_walls(X, Y) & (X > A - 1.0)
    out["tough Voronoi walls"] = (vw.astype(int), "tough", None)
    return out


def steering(X, Y, angles=(15.0, 30.0, 45.0, 60.0), widths=(1, 2, 3), contrasts=(2.0, 3.0, 4.0)):
    """The map of steering: a soft corridor of ``width`` pixels from the crack tip at
    ``angle`` in a hard sheet of ``contrast`` (graded rule)."""
    out = {"untreated": (np.zeros(X.shape, int), "c3", None)}
    for c in contrasts:
        out[f"fully treated, c {c:g}"] = (np.ones(X.shape, int), f"c{c:g}", None)
        # an arc of radius 8 mm from the tip, turning the crack by 90 degrees
        r, th = np.hypot(X - A, Y - 8.0), np.degrees(np.arctan2(Y - 8.0, X - A))
        arc = (np.abs(r - 8.0) < 1.0) & (th >= -95.0) & (th <= 0.0)
        out[f"arc corridor turning by 90 deg, c {c:g}"] = ((~arc).astype(int), f"c{c:g}", arc)
        # a corridor at 60 deg that ends 6 mm from the tip, in the hard sheet
        across, along = corridor_distance(X, Y, 60.0)
        short = (across < 1.0) & (along > -2.0) & (along < 6.0)
        out[f"corridor at 60 deg ending after 6 mm, c {c:g}"] = (
            (~short).astype(int),
            f"c{c:g}",
            short,
        )
        for w in widths:
            for angle in angles:
                across, along = corridor_distance(X, Y, angle)
                soft = (across < 0.5 * w * PIXEL + 1e-9) & (along > -2.0)
                out[f"corridor {angle:.0f} deg, {w} px, c {c:g}"] = (
                    (~soft).astype(int),
                    f"c{c:g}",
                    soft,
                )
    return out


def phases(kind: str):
    from morphogen.laser import FlowLaw
    from morphogen.laser.cells import Phase, contrast_phases

    matrix = Phase("matrix", FlowLaw.hollomon(280.0, 986.0, 0.24), critical=0.5, cost=0.0)
    if kind == "brittle":
        hard = contrast_phases(1300.0 / 280.0, "graded")[1]
    elif kind == "unbreakable":
        hard = contrast_phases(3.0, "ductile")[1]
    elif kind.startswith("c"):
        hard = contrast_phases(float(kind[1:]), "graded")[1]
    else:
        hard = contrast_phases(3.0, "graded")[1]
    return matrix, hard


def _init() -> None:
    import numba

    from morphogen._platform import full_speed

    full_speed()
    numba.set_num_threads(1)  # one thread per process: the processes share the cores


def run(job) -> dict:
    """One crack test: ``(name, cells, kind, corridor)``."""
    from morphogen.laser.cells import CellProblem
    from morphogen.laser.structures import CompactTension

    name, cells, kind, corridor, fine = job
    ct = CompactTension(
        width=W,
        crack=A / W,
        element=2.0,
        fine=fine,
        refine=BOX,
        propagate=True,
        max_disp=40.0,
        region=(BOX[0], BOX[1], -BOX[2], BOX[2]),
    )
    problem = CellProblem(phases=phases(kind), case=ct, size=PIXEL, fracture_radius=0.5)
    t0 = time.perf_counter()
    res = problem.evaluate(np.asarray(cells, dtype=int)).result
    broken, at = res.fields["broken"]
    cen = problem.mesh.nodes[problem.mesh.quads].mean(axis=1)
    order = np.argsort(at, kind="stable")
    path = cen[np.asarray(broken)[order]]
    work = float(np.trapezoid(res.force, res.displacement)) / 1000.0  # J
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
    ahead = path[path[:, 0] > A]
    slope = np.polyfit(ahead[:, 0], ahead[:, 1], 1)[0] if len(ahead) > 2 else math.nan
    share = math.nan
    if corridor is not None and len(path):
        x0, y0 = BOX[0], -BOX[2]
        j = np.clip(((path[:, 0] - x0) / PIXEL).astype(int), 0, corridor.shape[1] - 1)
        i = np.clip(((path[:, 1] - y0) / PIXEL).astype(int), 0, corridor.shape[0] - 1)
        share = float(corridor[i, j].mean())
    separated = bool(res.force[-1] < 0.05 * res.force.max())
    return {
        "name": name,
        "kind": kind,
        "treated": float(np.mean(cells)),
        "work": work,
        "force": float(res.force.max()) / 1000.0,
        "length": length,
        "broken": len(broken),
        "slope_deg": math.degrees(math.atan(slope)) if math.isfinite(slope) else math.nan,
        "in_corridor": share,
        "separated": separated,
        "displacement": float(res.displacement[-1]),
        "path": path,
        "cells": cells,
        "curve": (res.displacement.copy(), res.force.copy()),
        "time": time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--output", type=Path, default=Path("results/laser_crack_paths.npz"))
    parser.add_argument("--map", action="store_true", help="the map of steering instead")
    parser.add_argument("--fine", type=float, default=0.5, help="elements over the ligament")
    parser.add_argument("--only", nargs="+", help="the designs whose names contain these")
    args = parser.parse_args()
    x0, x1, y1 = BOX
    xs = np.arange(x0 + 0.5 * PIXEL, x1, PIXEL)
    ys = np.arange(-y1 + 0.5 * PIXEL, y1, PIXEL)
    X, Y = np.meshgrid(xs, ys)
    cases = steering(X, Y) if args.map else designs(X, Y)
    if args.only:
        cases = {k: v for k, v in cases.items() if any(s in k for s in args.only)}
    todo = [(name, c, kind, corr, args.fine) for name, (c, kind, corr) in cases.items()]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        rows = list(pool.map(run, todo))
    print(f"{len(rows)} crack tests in {time.perf_counter() - t0:.0f} s\n")
    ref = rows[0]["work"]
    print(f"  {'design':55s} treated  work (J)  F_max (kN)  broken  slope  in corridor")
    for r in rows:
        print(
            f"  {r['name']:55s} {r['treated']:5.2f}  {r['work']:6.2f} ({r['work'] / ref:4.2f})"
            f"  {r['force']:5.2f}      {r['broken']:5d}  {r['slope_deg']:+6.1f}  "
            f"{r['in_corridor']:5.2f}" + ("" if r["separated"] else "  (not separated)")
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        names=np.array([r["name"] for r in rows]),
        **{f"path_{k}": r["path"] for k, r in enumerate(rows)},
        **{f"cells_{k}": r["cells"] for k, r in enumerate(rows)},
        **{f"disp_{k}": r["curve"][0] for k, r in enumerate(rows)},
        **{f"force_{k}": r["curve"][1] for k, r in enumerate(rows)},
    )


if __name__ == "__main__":
    main()
