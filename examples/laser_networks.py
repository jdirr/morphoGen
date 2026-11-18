"""Geometric work hardening over the plane of the sheet and along the paths of forming:
networks of corrugated laser tracks (docs/laser.md, §16).

Corrugated tracks harden a sheet geometrically along their direction (§15). A sheet is
loaded in other directions too, and formed along biaxial paths: the tracks are here
joined into networks, a crossed (square) and a triangular network of corrugated tracks,
and compared with one family at the same treated fraction (24 %), the same width (1 mm)
and the same rules of fracture.

Studies (``--study``):

``directions``
    the periodic sheets pulled in the directions of the plane: one family of tracks
    (``y = (k + ½) s + H sin(2π x/P)``, ``P`` = 20 mm) from 0 to 90°, the square network
    (two families along x and y, two tracks of each per period ``P``, the second a
    quarter of a period behind) from 0 to 90°, the triangular network (three families at 0, 60
    and 120° through the nodes of a triangular lattice of side ``L``, one wave between
    two nodes) from 0 to 45° (it is chiral: its directions repeat every 60° without
    mirroring); heights ``H/P`` from 0 to 0.3, the period or the side setting the
    fraction; the as-quenched martensite under the rules ``ductile`` (the necking
    alone) and ``graded`` (it breaks at 0.072), and the tempered state of contrast 3
    (``graded``, it breaks at 0.24).
``forming``
    the forming limits along linear paths of the true strains, from plane strain to
    equibiaxial stretching (``ratio`` of the minor to the major strain 0 to 1, the
    major strain along x or y): the Marciniak–Kuczyński groove, a band 1 mm wide (the
    thickness of the sheet) and 0.5 % weaker, across a periodic cell of several
    periods (60 mm at least) at three positions over the pattern; the limit is the
    strain outside the band of 1 mm that strains ten times faster than its images in
    the other periods (the groove or a neck elsewhere: a neck breaks the periodicity of
    the pattern), the strain at which the path can no longer be followed (``stall``),
    or the fracture of the reinforcement, whichever comes first.
``mk``
    the groove in the untreated sheet against the classical Marciniak–Kuczyński
    analysis (rigid plastic, von Mises, the Swift law of the matrix, the same strength
    imperfection), cells of 20 to 200 mm.

Writes ``results/laser_networks_<study>.csv`` and ``.npz`` (the curves and, for
``forming`` and ``mk``, the strains outside the band that localizes and the ratio of its
strain rate to that of its images); ``all`` runs the three studies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import laser_geometric_hardening as g

TRACK = 1.0  # mm: the width of a track, one spot
FRACTION = 0.24
PERIOD = 20.0  # mm: one family (§15)
RATIOS = (0.0, 0.15, 0.2, 0.25, 0.3)
#: the rules of fracture: (contrast, rule) of the reinforcement
STATES = {
    "ductile": (g.MARTENSITE, "ductile"),
    "as-quenched": (g.MARTENSITE, "graded"),
    "tempered": (3.0, "graded"),
}
DIRECTIONS = {
    "family": (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
    "square": (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
    "triangular": (0.0, 15.0, 30.0, 45.0),
}
PATHS = (0.0, 0.25, 0.5, 0.75, 1.0)
GROOVE = 1.0  # mm
IMPERFECTION = 0.995  # the flow stress of the groove, a share of the sheet's
LOCALIZED = 10.0  # the ratio of the strain rates of the groove and of the rest
MIN_LENGTH = 60.0  # mm: the cell of the groove, along the major strain
FIELDS = (
    "key",
    "study",
    "design",
    "state",
    "ratio",
    "period",
    "spacing",
    "radius",
    "treated",
    "angle",
    "path",
    "groove",
    "length",
    "element",
    "yield_strength",
    "tensile_strength",
    "uniform_elongation",
    "failure_strain",
    "energy",
    "r_value",
    "broke",
    "major",
    "minor",
    "limit",
    "seconds",
)


_SCALES: dict = {}


def _geometry(design: str, ratio: float, scale: float) -> dict:
    arc = g.arc_ratio(ratio)
    if design == "none":
        return {"period": 10.0, "spacing": math.nan, "cell": (10.0, 2.0)}
    if design == "family":
        s = TRACK * arc / FRACTION
        return {"period": PERIOD, "spacing": s, "cell": (PERIOD, s)}
    if design == "square":  # two families, two tracks of each per period
        s = scale * 2.0 * TRACK * arc / FRACTION
        return {"period": 2.0 * s, "spacing": s, "cell": (2.0 * s, 2.0 * s)}
    if design == "triangular":  # three families through the nodes, one wave between two
        side = scale * 3.0 * TRACK * arc / (FRACTION * math.sqrt(3.0) / 2.0)
        return {
            "period": side,
            "spacing": side * math.sqrt(3.0) / 2.0,
            "cell": (side, side * math.sqrt(3.0)),
        }
    raise ValueError(design)


def _treated(design: str, ratio: float, scale: float) -> float:
    from morphogen.laser.cells import band_fractions
    from morphogen.laser.periodic import PeriodicCell

    lx, ly = _geometry(design, ratio, scale)["cell"]
    cell = PeriodicCell.build((lx, 0.0), (0.0, ly), element=0.5, centre=(0.5 * lx, 0.5 * ly))
    share = band_fractions(cell.mesh, lines_over(design, ratio, cell.box(TRACK), scale), TRACK)
    w0 = cell.mesh.gauss()[2]
    return float((share * w0).sum() / w0.sum())


def pattern(design: str, ratio: float) -> dict:
    """The geometry of a design: its period (along a track, mm), the spacing of its
    tracks, the rectangular periodic cell ``(Lx, Ly)``; the networks scaled so that
    their tracks, which cross, treat 24 % of the sheet (one family: the spacing of
    §15, ``t L/(P f)``)."""
    if design not in ("square", "triangular"):
        return _geometry(design, ratio, 1.0)
    key = (design, round(ratio, 6))
    if key not in _SCALES:
        scale = 1.0
        for _ in range(4):  # the fraction goes as the inverse of the scale
            scale *= _treated(design, ratio, scale) / FRACTION
        _SCALES[key] = scale
    out = _geometry(design, ratio, _SCALES[key])
    out["scale"] = _SCALES[key]
    return out


def _line(base, theta: float, period: float, height: float, u0: float, u1: float):
    """A corrugated centre line from ``base`` along the direction ``theta`` (rad):
    ``base + u e + H sin(2π u/P) n``, ``u`` from ``u0`` to ``u1``."""
    e = np.array([math.cos(theta), math.sin(theta)])
    n = np.array([-math.sin(theta), math.cos(theta)])
    count = max(8, int(64 * (u1 - u0) / period) + 1)
    u = np.linspace(u0, u1, count)
    return (
        np.asarray(base)
        + u[:, None] * e
        + (height * np.sin(2.0 * math.pi * u / period))[:, None] * n
    )


def lines_over(design: str, ratio: float, box, scale: float | None = None) -> list:
    """The centre lines of the tracks of a design that cross the box ``(x0, x1, y0,
    y1)`` (see the module)."""
    p = pattern(design, ratio) if scale is None else _geometry(design, ratio, scale)
    if design == "none":
        return []
    P, s = p["period"], p["spacing"]
    H = ratio * P
    x0, x1, y0, y1 = box
    m = P + H + 2 * TRACK
    out = []
    if design in ("family", "square"):
        for k in range(math.floor((y0 - m) / s), math.ceil((y1 + m) / s) + 1):
            out.append(_line((0.0, (k + 0.5) * s), 0.0, P, H, x0 - m, x1 + m))
        if design == "square":
            for k in range(math.floor((x0 - m) / s), math.ceil((x1 + m) / s) + 1):
                # along y, a quarter of a period behind: x = (k + ½) s + H cos(2π y/P),
                # the families crossing at large angles (in phase, they run along each
                # other where the crests of one meet the flanks of the other)
                out.append(_line(((k + 0.5) * s, 0.25 * P), 0.5 * math.pi, P, H, y0 - m, y1 + m))
        return out
    side = P
    a1 = np.array([side, 0.0])
    a2 = np.array([0.5 * side, 0.5 * math.sqrt(3.0) * side])
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    for theta, step in ((0.0, a2), (math.pi / 3, a1), (2 * math.pi / 3, a1)):
        e = np.array([math.cos(theta), math.sin(theta)])
        n = np.array([-math.sin(theta), math.cos(theta)])
        off = corners @ n
        sn = float(step @ n)  # the offset of line k along the normal: k sn
        lo, hi = sorted(((off.min() - m) / sn, (off.max() + m) / sn))
        for k in range(math.floor(lo) - 1, math.ceil(hi) + 2):
            base = k * step
            along = (corners - base) @ e
            # from a node: whole periods, the nodes at the zeros of the sine
            u0 = math.floor((along.min() - m) / side) * side
            u1 = math.ceil((along.max() + m) / side) * side
            out.append(_line(base, theta, side, H, u0, u1))
    return out


def build(job: dict):
    """The periodic cell of a job (repeated along the major strain for a groove), its
    membrane, the elements of the groove (none for a tensile test) and the bands across
    the major strain, two elements wide (about 1 mm), the groove the first."""
    from morphogen.laser.architectures import track_membrane
    from morphogen.laser.periodic import PeriodicCell

    design, ratio, element = job["design"], job["ratio"], job["element"]
    p = pattern(design, ratio)
    lx, ly = p["cell"]
    sizes = [element, element]
    reps, per = 1, 0
    if job["study"] in ("forming", "mk"):
        axis = 0 if job["angle"] == 0.0 else 1
        unit = p["cell"][axis]
        # at least two periods and MIN_LENGTH; an even number of elements per period, so
        # that the bands of two elements repeat with the pattern
        reps = max(2, math.ceil(job.get("length", MIN_LENGTH) / unit - 1e-9))
        per = 2 * math.ceil(unit / (2.0 * element) - 1e-9)
        sizes[axis] = unit / per
        if axis == 0:
            lx *= reps
        else:
            ly *= reps
    cell = PeriodicCell.build((lx, 0.0), (0.0, ly), element=min(sizes), centre=(0.5 * lx, 0.5 * ly))
    if sizes[0] != sizes[1]:  # one size along each axis
        a1, a2 = np.array([lx, 0.0]), np.array([0.0, ly])
        cell = _cell(a1, a2, sizes)
    lines = lines_over(design, ratio, cell.box(TRACK))
    phases = g.phases_of(
        {"law": "contrast", "contrast": STATES[job["state"]][0], "rule": STATES[job["state"]][1]}
    )
    mem = track_membrane(cell.mesh, lines, TRACK, phases=phases)
    groove = np.zeros(len(mem.quads), dtype=bool)
    bands = np.zeros((len(mem.quads), 2), dtype=np.int64)
    if job["study"] in ("forming", "mk"):
        n = cell.mesh.shape[axis]  # the elements along the major strain
        h = (lx if axis == 0 else ly) / n
        i = np.floor((mem.centroids[:, axis] - cell.origin[axis]) / h + 1e-6).astype(np.int64)
        # the groove: the two elements about the edge nearest its position (a share of
        # the pattern's cell) in the first period
        edge = round(job["groove"] * per)
        k = (i - edge + 1) % n
        # each band of two elements: its period and its place in the period
        bands = np.stack([k // per, (k % per) // 2], axis=1)
        groove = (bands[:, 0] == 0) & (bands[:, 1] == 0)
        for j in (1, 2, 5, 6):  # the flow stresses of both laws of the points
            mem.law[groove, ..., j] *= IMPERFECTION
    return cell, mem, groove, bands


def _cell(a1, a2, sizes):
    """A rectangular periodic cell meshed by elements of ``sizes`` (along x and y)."""
    from morphogen.laser.periodic import PeriodicCell

    n1 = math.ceil(a1[0] / sizes[0] - 1e-6)
    n2 = math.ceil(a2[1] / sizes[1] - 1e-6)
    # PeriodicCell.build meshes by one size: build with that of x, then scale y
    ref = PeriodicCell.build(
        (n1 * 1.0, 0.0), (0.0, n2 * 1.0), element=1.0, centre=(0.5 * n1, 0.5 * n2)
    )
    scale = np.array([a1[0] / n1, a2[1] / n2])
    mesh = ref.mesh
    mesh.nodes = mesh.nodes * scale
    return PeriodicCell(
        np.asarray(a1, float),
        np.asarray(a2, float),
        ref.origin * scale,
        mesh,
        ref.master,
        ref.shift,
    )


def _major(mem, angle: float):
    """The true strains along and across the direction ``angle`` of each element."""
    th = math.radians(angle)
    n = np.array([math.cos(th), math.sin(th)])
    t = np.array([-math.sin(th), math.cos(th)])
    F = mem.Fn
    e1 = np.log(np.linalg.norm(F @ n, axis=-1)).mean(axis=1)
    e2 = np.log(np.linalg.norm(F @ t, axis=-1)).mean(axis=1)
    return e1, e2


def evaluate(job: dict) -> dict:
    from morphogen.laser.cells import failure_measures
    from morphogen.laser.periodic import cell_test

    t0 = time.perf_counter()
    cell, mem, _, bands = build(job)
    p = pattern(job["design"], job["ratio"])
    share = mem.law[..., 0, 8]
    treated = float((share * mem.w0).sum() / mem.w0.sum())
    row = {k: job[k] for k in FIELDS if k in job}
    row.update(
        {
            "period": p["period"],
            "spacing": p["spacing"],
            "radius": g.radius(p["period"], job["ratio"]),
            "treated": treated,
        }
    )
    if job["study"] == "directions":
        res = cell_test(mem, cell, job["angle"], max_strain=0.6, failure_margin=0.03)
        fail = res.failure_strain if math.isfinite(res.failure_strain) else math.inf
        m = failure_measures(res.strain, res.stress, fail, young=mem.young)
        r = res.r_value()
        row.update(
            {
                "yield_strength": m["yield_strength"],
                "tensile_strength": m["tensile_strength"],
                "uniform_elongation": m["uniform_elongation"],
                "failure_strain": m["failure_strain"],
                "energy": m["energy"],
                "r_value": float(r),
                "broke": bool(fail < m["uniform_elongation"] - 1e-9),
            }
        )
        row["curve"] = [np.asarray(res.strain).tolist(), np.asarray(res.stress).tolist()]
    else:
        w = mem.w0.sum(axis=1)
        nper, nrep = int(bands[:, 1].max()) + 1, int(bands[:, 0].max()) + 1
        flat = bands[:, 0] * nper + bands[:, 1]
        wb = np.bincount(flat, weights=w, minlength=nper * nrep).reshape(nrep, nper)
        state = {"e1": np.zeros(len(w)), "after": 0, "band": 0, "groove": 0.0}
        angle = job["angle"]

        def rate(_q):
            """The band that strains fastest against its images in the other periods
            (a neck breaks the periodicity of the pattern; the groove is band 0)."""
            e1, _ = _major(mem, angle)
            d = e1 - state["e1"]
            state["e1"] = e1
            db = np.bincount(flat, weights=d * w, minlength=nper * nrep).reshape(nrep, nper)
            db = db / wb
            tot, wt = (db * wb).sum(axis=0), wb.sum(axis=0)
            images = (tot[None, :] - db * wb) / (wt[None, :] - wb)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(images > 1e-12, db / np.maximum(images, 1e-300), np.inf)
            r[(images <= 1e-12) & (db <= 1e-12)] = 0.0
            r = r.ravel()
            k = int(np.argmax(r))
            state["band"], state["groove"] = k, float(r[0])
            state["after"] += int(r[k] >= LOCALIZED)
            return float(r[k])

        def strains(j):  # the mean strains outside the fastest band
            def f(_q):
                v = _major(mem, angle)[j]
                out = flat != state["band"]
                return float((v[out] * w[out]).sum() / w[out].sum())

            return f

        res = cell_test(
            mem,
            cell,
            angle,
            ratio=job["path"],
            max_strain=1.2,
            drop=1.0,
            step=0.001,
            max_step=0.002,
            max_steps=3000,
            failure_margin=0.002,
            monitors={
                "rate": rate,
                "band": lambda _q: float(state["band"]),
                "groove": lambda _q: state["groove"],
                "major": strains(0),
                "minor": strains(1),
            },
            until=lambda _q: state["after"] >= 5,  # localized: five steps beyond
        )
        mon = res.fields["monitors"]
        e1, e2 = np.asarray(mon["major"]), np.asarray(mon["minor"])
        ratio_ = np.nan_to_num(np.asarray(mon["rate"]), nan=0.0)
        band = np.asarray(mon["band"])
        limit = "none"
        major = minor = math.nan
        hit = np.flatnonzero(ratio_ >= LOCALIZED)
        top = float(np.asarray(res.strain)[-1])
        stalled = top < 1.2 * (1 - 1e-6) and not math.isfinite(res.failure_strain)
        if hit.size:
            k = int(hit[0])
            f = 1.0
            if k > 0 and math.isfinite(ratio_[k]) and ratio_[k - 1] > 0:
                # the crossing of the threshold, the ratio interpolated in its logarithm
                lo, hi = math.log(max(ratio_[k - 1], 1e-9)), math.log(ratio_[k])
                f = min(max((math.log(LOCALIZED) - lo) / max(hi - lo, 1e-12), 0.0), 1.0)
            j = max(k - 1, 0)
            major = float(e1[j] + f * (e1[k] - e1[j]))
            minor = float(e2[j] + f * (e2[k] - e2[j]))
            limit = "groove" if band[k] == 0 else "band"
        elif stalled:
            # the path could not be followed further: the cell snapped back as a neck
            # formed (the rest unloading elastically while the band softens), or the
            # strain gathered in every period at once; the strain at which it stopped
            k = int(np.argmax(e1))
            major, minor = float(e1[k]), float(e2[k])
            limit = "stall"
        if math.isfinite(res.failure_strain):
            ef = math.log1p(res.failure_strain)  # along the major direction of the cell
            if not math.isfinite(major) or ef < major:
                k = int(np.searchsorted(np.log1p(np.asarray(res.strain)), ef))
                k = min(max(k, 1), len(e1) - 1)
                major, minor, limit = float(e1[k]), float(e2[k]), "fracture"
        row.update({"major": major, "minor": minor, "limit": limit})
        row["curve"] = [
            np.asarray(res.strain).tolist(),
            np.asarray(res.stress).tolist(),
            e1.tolist(),
            e2.tolist(),
            np.nan_to_num(ratio_, posinf=1e9).tolist(),
            band.tolist(),
            np.nan_to_num(np.asarray(mon["groove"]), nan=0.0, posinf=1e9).tolist(),
        ]
    row["seconds"] = time.perf_counter() - t0
    return row


def _job(study, design, state, ratio, element, **extra) -> dict:
    job = {
        "study": study,
        "design": design,
        "state": state,
        "ratio": float(ratio),
        "element": float(element),
    }
    job.update(extra)
    job["key"] = json.dumps({k: job[k] for k in sorted(job)}, sort_keys=True)
    return job


def jobs(study: str, element: float) -> list:
    out = []
    if study in ("directions", "all"):
        out.append(_job("directions", "none", "ductile", 0.0, element, angle=0.0))
        for state in STATES:
            for design, angles in DIRECTIONS.items():
                for r in RATIOS:
                    out += [_job("directions", design, state, r, element, angle=a) for a in angles]
    if study in ("forming", "all"):
        designs = [("none", 0.0, (0.0,)), ("family", 0.0, (0.0, 90.0))]
        designs += [("family", 0.25, (0.0, 90.0)), ("square", 0.25, (0.0,))]
        designs += [("triangular", 0.25, (0.0, 90.0))]
        for state in ("ductile", "tempered"):
            for design, r, angles in designs:
                if design == "none" and state != "ductile":
                    continue
                grooves = (0.5,) if design == "none" else (0.25,)
                for a in angles:
                    for path in PATHS:
                        for gr in grooves:
                            out.append(
                                _job(
                                    "forming",
                                    design,
                                    state,
                                    r,
                                    element,
                                    angle=a,
                                    path=path,
                                    groove=gr,
                                    length=MIN_LENGTH,
                                )
                            )
    if study == "mk":
        for path in PATHS:
            for length in (20.0, 50.0, 100.0, 200.0):
                out.append(
                    _job(
                        "mk",
                        "none",
                        "ductile",
                        0.0,
                        element,
                        angle=0.0,
                        path=path,
                        groove=0.5,
                        length=length,
                    )
                )
    unique = {}
    for j in out:
        unique.setdefault(j["key"], j)
    return list(unique.values())


def run(todo: list, workers: int, done: dict) -> dict:
    todo = [j for j in todo if j["key"] not in done]
    # the longest first: the networks, the cells of the groove
    todo.sort(key=lambda j: (j["study"] != "forming", j["design"] == "none", j["design"]))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=g._init) as pool:
        futures = {pool.submit(evaluate, j): j for j in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            done[row["key"]] = row
            if n % 20 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} jobs, {time.perf_counter() - t0:.0f} s", flush=True)
    return done


def save(rows: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows.values())
    curves = {r["key"]: r["curve"] for r in rows.values() if "curve" in r}
    np.savez_compressed(
        path.with_suffix(".npz"),
        keys=np.array(list(curves)),
        **{f"c{i}": np.array(v) for i, v in enumerate(curves.values())},
    )


def load(path: Path) -> dict:
    rows = {}
    csv_path = path.with_suffix(".csv")
    if not csv_path.exists():
        return rows
    text = {"key", "study", "design", "state", "limit", "broke"}
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in set(FIELDS) - text:
                r[k] = float(r[k]) if r.get(k) not in (None, "") else math.nan
            r["broke"] = r.get("broke") == "True"
            rows[r["key"]] = r
    npz = path.with_suffix(".npz")
    if npz.exists():
        data = np.load(npz)
        for i, key in enumerate(data["keys"]):
            if str(key) in rows:
                rows[str(key)]["curve"] = data[f"c{i}"].tolist()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", default="all", choices=("directions", "forming", "mk", "all"))
    ap.add_argument("--element", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--output", type=Path, default=None, help="default: results/laser_networks_<study>"
    )
    args = ap.parse_args()
    for study in ("directions", "forming", "mk") if args.study == "all" else (args.study,):
        out = args.output or Path(f"results/laser_networks_{study}")
        done = load(out)
        todo = jobs(study, args.element)
        print(f"{study}: {len(todo)} jobs ({sum(j['key'] in done for j in todo)} done)", flush=True)
        rows = run(todo, args.workers, done)
        save(rows, out)
        print(f"written: {out}.csv")


if __name__ == "__main__":
    main()
