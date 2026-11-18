"""Geometric work hardening in one laser-treated steel: corrugated laser tracks over a sheet
(docs/laser.md, §15).

A corrugated reinforcement that unbends while the matrix around it flows turns towards
the load and hardens the composite after its matrix would have necked (Bouaziz 2013;
Fraser et al. 2018, 2020). Fraser's composites join two metals (copper and a 4130
steel); here the reinforcement is the same steel hardened by the laser, tracks one spot
wide (1 mm) treated over a sheet of tens of centimetres, so that the height of the
corrugation is no longer bounded by the gauge of a specimen (§5: ``H/P`` at most 0.15 in
the 5 mm gauge of Wang et al., just below Fraser's threshold).

Studies (``--study``):

``sheet``
    the periodic sheet of corrugated tracks in phase, ``y = (k + ½) s + H sin(2π x/P)``,
    1 mm wide: rules of fracture (``graded``, ``ductile``), contrasts (3, 4, 4.64),
    treated fractions (12, 24, 36 %), periods (10, 20, 30 mm), heights over period (0 to
    0.3); the spacing ``s`` gives the fraction. Feasible: a radius of curvature of the
    track of at least 1.5 mm (§5).
``moduli``
    the stiffness contrast against the strength contrast: one corrugated design and its
    straight counterpart with the moduli of the steel (one modulus), of Fraser's copper
    and steel (117 and 210 GPa), and with the stiffness contrast alone (the matrix's flow
    law in both phases; moduli in the ratios 1.8 and 3, the range of metal pairs).
``stiffness``
    the modulus of the reinforcement at a fixed contrast of strength (70 to 630 GPa
    against a matrix of 210 GPa, and a matrix of 70 GPa), and the stiffness contrast
    alone in a ratio of 3 both ways: the stresses and the strains of the treated and of
    the untreated material recorded at each step from a strain of 0.01 %, how each
    contrast shares them out (``results/laser_geometric_hardening_stiffness``).
``recruit``
    tracks of several heights in one cell (they straighten one after the other) against
    one height, at the same fraction.
``fraser``
    Fraser's copper and 4130 steel (Fraser et al. 2018, Table 2) in the periodic sheet:
    a period of 8 mm, heights (peak to trough) of 0 to 4 mm, 20 % of reinforcement.
``crests``
    the martensite breaks at the crests of the corrugation, where the track bends: the
    crests (a share of the period about each) treated softer, tempered (contrasts 2 and
    3) or left untreated, the flanks as-quenched.
``mesh``
    the key designs (martensite, 24 %, ``P`` = 20 mm) for a check of the mesh
    (``--study mesh --element 0.25``).
``specimen``
    dog-bones cut from the sheet, four and eight tracks wide, against the periodic
    sheet; their edges, where the necking starts, bordered by a tempered band.

Measures: the yield and tensile strengths, the uniform elongation (the maximum of the
nominal stress, Considère's), the failure strain (the first of the necking and of the
fracture of the reinforcement) and the energy absorbed up to it; the curves are kept for
the figure (``--figure``). Writes ``results/laser_geometric_hardening.csv`` and ``.npz``.
About 20 minutes on 16 processes for every study (elements of 0.5 mm).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TRACK = 1.0  # mm: the width of a track, one spot
MIN_RADIUS = 1.5  # mm: the tightest curve of a track (§5)
STEEL = 210000.0  # MPa
COPPER = 117000.0  # MPa: Fraser's copper in the ratio of his steel (115 / 207)
MARTENSITE = 1300.0 / 280.0
CONTRASTS = (3.0, 4.0, MARTENSITE)
FRACTIONS = (0.12, 0.24, 0.36)
PERIODS = (10.0, 20.0, 30.0)
RATIOS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
RULES = ("graded", "ductile")
#: (matrix, reinforcement) moduli of the stiffness study, MPa
STIFFNESS = (
    (STEEL, STEEL / 3),
    (STEEL, COPPER),
    (STEEL, STEEL),
    (STEEL, 2 * STEEL),
    (STEEL, 3 * STEEL),
    (STEEL / 3, STEEL),
)
MAX_STRAIN = 0.5
EXTRA = ("crest", "extent", "count", "border")  # the options of some studies
RECRUITS = {
    "ramp 0.05 to 0.25": (0.05, 0.15, 0.25, 0.15),
    "ramp 0.10 to 0.30": (0.1, 0.2, 0.3, 0.2),
    "pair 0.10 and 0.25": (0.1, 0.25),
    "pair 0.15 and 0.30": (0.15, 0.3),
}
FIELDS = (
    "key",
    "study",
    "rule",
    "contrast",
    "fraction",
    "period",
    "ratios",
    "spacing",
    "radius",
    "feasible",
    "treated",
    "young_soft",
    "young_hard",
    "law",
    "element",
    "crest",
    "extent",
    "count",
    "border",
    "slope",
    "yield_strength",
    "tensile_strength",
    "uniform_elongation",
    "failure_strain",
    "energy",
    "broke",
    "seconds",
)


def arc_ratio(ratio: float) -> float:
    """The length of a period of ``y = H sin(2π x/P)`` over ``P``, at ``H/P = ratio``."""
    if ratio == 0:
        return 1.0
    u = np.linspace(0.0, 2.0 * math.pi, 4001)
    k = 2.0 * math.pi * ratio
    return float(np.trapezoid(np.sqrt(1.0 + (k * np.cos(u)) ** 2), u) / (2.0 * math.pi))


def spacing(ratios, fraction: float, width: float = TRACK) -> float:
    """The spacing of the tracks (mm) that treats ``fraction`` of the sheet."""
    return width * sum(arc_ratio(r) for r in ratios) / (len(ratios) * fraction)


def radius(period: float, ratio: float) -> float:
    """The radius of curvature of the track at its crests (mm)."""
    return math.inf if ratio == 0 else period / (4.0 * math.pi**2 * ratio)


def crosses(ratios, period: float, gap: float, width: float = TRACK) -> bool:
    """Whether neighbouring tracks of a cell (in phase, ``gap`` apart, the last next to
    the first of the next cell) come closer than their width at their crests."""
    amp = [r * period for r in ratios]
    pairs = list(zip(amp, amp[1:] + amp[:1], strict=True))
    return any(gap - abs(a - b) < width - 1e-9 for a, b in pairs)


def phases_of(job):
    from morphogen.laser.cells import Phase, contrast_phases, wang_phases

    if job["law"] == "fraser":  # Fraser et al. 2018, Table 2: copper and 4130 steel
        from morphogen.laser.materials import FlowLaw

        return (
            Phase("copper", FlowLaw.hollomon(25.0, 540.0, 0.53)),
            Phase("steel", FlowLaw.hollomon(350.0, 1200.0, 0.12)),
        )
    if job["law"] == "matrix":  # the stiffness contrast alone
        soft, _ = wang_phases()
        return soft, Phase("stiff", soft.law)
    phases = contrast_phases(job["contrast"], job["rule"])
    crest, border = job.get("crest", 0.0), job.get("border", 0.0)
    extra = crest if crest > 1.0 else border
    if extra > 1.0:  # a third phase: softer crests, or the border of a specimen
        phases = (*phases, contrast_phases(extra, job["rule"])[1])
    return phases


def split_crests(line, period: float, extent: float):
    """The pieces of a corrugated centre line about its crests (within ``extent``
    periods of ``x = P/4`` and ``3P/4``) and between them: ``(crests, flanks)``."""
    u = (line[:, 0] / period) % 1.0
    near = (np.abs(u - 0.25) <= extent) | (np.abs(u - 0.75) <= extent)
    pieces = ([], [])
    start = 0
    for i in range(1, len(line) + 1):
        if i == len(line) or near[i] != near[start]:
            j = min(i + 1, len(line))  # the pieces share their ends
            pieces[0 if near[start] else 1].append(line[max(start - 1, 0) : j])
            start = i
    return pieces


def cell_lines(
    period: float, ratios, gap: float, element: float, extent: float = 0.0, crests: int = 2
):
    """The periodic cell of a sheet of tracks (one period along x, one gap per track
    along y), the centre lines of the tracks crossing it and the phase of each: 1, or
    with ``extent`` > 0 the crests ``crests`` (2; 0: left untreated) and the flanks 1."""
    from morphogen.laser.architectures import periodic_lines
    from morphogen.laser.periodic import PeriodicCell

    n = len(ratios)
    a1, a2 = (period, 0.0), (0.0, n * gap)
    cell = PeriodicCell.build(a1, a2, element=element, centre=(0.5 * period, 0.5 * n * gap))
    x = np.linspace(-period, 2.0 * period, 385)
    tracks = [
        np.column_stack([x, (k + 0.5) * gap + r * period * np.sin(2.0 * math.pi * x / period)])
        for k, r in enumerate(ratios)
    ]
    pieces = []  # (line, phase)
    for t in tracks:
        if extent > 0:
            top, flanks = split_crests(t, period, extent)
            pieces += [(ln, 1) for ln in flanks]
            if crests:
                pieces += [(ln, crests) for ln in top]
        else:
            pieces.append((t, 1))
    lines, kinds = [], []
    for ln, k in pieces:
        images = periodic_lines([ln], a1, a2, cell.box(), reach=TRACK)
        lines += images
        kinds += [k] * len(images)
    return cell, lines, kinds


def specimen_of(gap: float, period: float, count: int = 4):
    """A dog-bone ``count`` tracks wide, three periods long (four and a half beyond
    four tracks)."""
    from morphogen.laser.specimen import Specimen

    width = count * gap
    return Specimen(
        kind="dogbone",
        length=(3.0 if count <= 4 else 4.5) * period,
        width=width,
        radius=20.0,
        grip_width=1.5 * width,
        grip_length=15.0,
    )


def partition(mem) -> dict:
    """Monitors of a membrane under test: the mean axial stress (Cauchy, along x) and
    the mean equivalent strain (logarithmic, the thickness strain with it) of the treated
    material (``_r``) and of the untreated (``_m``), each point weighted by its shares."""
    share = mem.law[..., 0, 8]
    weights = {"r": share * mem.w0, "m": (1.0 - share) * mem.w0}

    def equivalent():
        F = mem.Fn
        C = np.einsum("...ki,...kj->...ij", F, F)
        half = 0.5 * (C[..., 0, 0] + C[..., 1, 1])
        det = C[..., 0, 0] * C[..., 1, 1] - C[..., 0, 1] * C[..., 1, 0]
        disc = np.sqrt(np.maximum(half**2 - det, 0.0))
        e1 = 0.5 * np.log(half + disc)
        e2 = 0.5 * np.log(np.maximum(half - disc, 1e-300))
        e3 = np.log(mem.lam_n).mean(axis=2)
        return np.sqrt(2.0 / 3.0 * (e1**2 + e2**2 + e3**2))

    def mean(field, w):
        return float((field * w).sum() / w.sum())

    out = {}
    for k, w in weights.items():
        out[f"stress_{k}"] = lambda _q, w=w: mean(mem.sig_n[..., 0].mean(axis=2), w)
        out[f"strain_{k}"] = lambda _q, w=w: mean(equivalent(), w)
    return out


def evaluate(job: dict) -> dict:
    """The tensile test of one design along x (see the module)."""
    from morphogen.laser.architectures import track_membrane
    from morphogen.laser.cells import failure_measures
    from morphogen.laser.mechanics import tensile_test
    from morphogen.laser.periodic import PeriodicCell, cell_test

    t0 = time.perf_counter()
    phases = phases_of(job)
    ratios = tuple(job["ratios"])
    period, fraction, element = job["period"], job["fraction"], job["element"]
    width = job.get("width", TRACK)
    young = (job["young_soft"], job["young_hard"])
    if fraction in (0.0, 1.0):  # one phase over the whole sheet
        cell = PeriodicCell.build((10.0, 0.0), (0.0, 5.0), element=element, centre=(5.0, 2.5))
        gap = math.nan
        lines = [] if fraction == 0.0 else [np.array([[-20.0, 2.5], [30.0, 2.5]])]
        mem = track_membrane(cell.mesh, lines, 100.0, phases=phases, young=young)
    else:
        gap = job.get("gap") or spacing(ratios, fraction, width)
        if job["study"] == "specimen":
            count = int(job.get("count", 4))
            spec = specimen_of(gap, period, count)
            mesh = spec.mesh(element)
            x0, x1 = spec.extent()
            x = np.linspace(x0 - period, x1 + period, int(64 * (x1 - x0 + 2 * period) / period))
            r = ratios[0]
            lines = [
                np.column_stack(
                    [x, (k + 0.5) * gap + r * period * np.sin(2 * math.pi * x / period)]
                )
                for k in range(-3, count + 3)
            ]
            kinds, widths = [1] * len(lines), [width] * len(lines)
            if job.get("border", 0.0) > 1.0:  # tempered bands along the edges of the gauge
                b = job.get("border_width", 2.0)
                for yb in (0.5 * b, spec.width - 0.5 * b):
                    lines.append(np.array([[x0 - period, yb], [x1 + period, yb]]))
                    kinds.append(2)
                    widths.append(b)
                young = (*young, young[1])
            mem = track_membrane(mesh, lines, widths, phases=phases, kinds=kinds, young=young)
        else:
            crest = job.get("crest", 0.0)
            cell, lines, kinds = cell_lines(
                period,
                ratios,
                gap,
                element,
                extent=job.get("extent", 0.0) if crest else 0.0,
                crests=2 if crest > 1.0 else 0,
            )
            if len(phases) == 3:
                young = (*young, young[1])
            mem = track_membrane(cell.mesh, lines, width, phases=phases, kinds=kinds, young=young)
    top = job.get("max_strain", MAX_STRAIN)
    shared = job["study"] == "stiffness"
    if job["study"] == "specimen" and fraction not in (0.0, 1.0):
        res = tensile_test(spec, mem, max_strain=top, failure_margin=0.03)
    else:
        res = cell_test(
            mem,
            cell,
            0.0,
            max_strain=top,
            failure_margin=0.03,
            step=job.get("step", 0.004),
            monitors=partition(mem) if shared else None,
        )
    fail = res.failure_strain if math.isfinite(res.failure_strain) else math.inf
    # the stiffness study: the proof stress on the initial slope of the curve, the
    # modulus of the sheet whose phases differ (its first step, 0.01 %, is elastic)
    slope = float(res.stress[1] / res.strain[1]) if shared else math.nan
    m = failure_measures(res.strain, res.stress, fail, young=slope if shared else mem.young)
    share = mem.law[..., 0, 8]
    gauge = np.ones_like(share)
    if job["study"] == "specimen" and fraction not in (0.0, 1.0):
        gx = mem.points[..., 0]
        gauge = ((gx >= 0) & (gx <= spec.length)).astype(float)
    treated = float((share * mem.w0 * gauge).sum() / (mem.w0 * gauge).sum())
    rmin = min((radius(period, r) for r in ratios), default=math.inf)
    row = {k: job.get(k, 0.0) for k in FIELDS if k in job or k in EXTRA}
    row.update(
        {
            "ratios": " ".join(f"{r:g}" for r in ratios),
            "spacing": gap,
            "radius": rmin,
            "feasible": rmin >= MIN_RADIUS,
            "treated": treated,
            "slope": slope,
            "yield_strength": m["yield_strength"],
            "tensile_strength": m["tensile_strength"],
            "uniform_elongation": m["uniform_elongation"],
            "failure_strain": m["failure_strain"],
            "energy": m["energy"],
            "broke": bool(fail < m["uniform_elongation"] - 1e-9),
            "seconds": time.perf_counter() - t0,
        }
    )
    row["curve"] = [np.asarray(res.strain).tolist(), np.asarray(res.stress).tolist()]
    if shared:  # the partition between the phases, after the curve
        mon = res.fields["monitors"]
        row["curve"] += [np.asarray(mon[k]).tolist() for k in partition(mem)]
    return row


def _job(study, rule, contrast, fraction, period, ratios, element, **extra) -> dict:
    job = {
        "study": study,
        "rule": rule,
        "contrast": float(contrast),
        "fraction": float(fraction),
        "period": float(period),
        "ratios": tuple(float(r) for r in ratios),
        "element": float(element),
        "young_soft": STEEL,
        "young_hard": STEEL,
        "law": "contrast",
    }
    job.update(extra)
    job["key"] = json.dumps({k: job[k] for k in sorted(job)}, sort_keys=True)
    return job


def jobs(study: str, element: float) -> list:
    out = []
    if study in ("sheet", "all"):
        out.append(_job("sheet", "ductile", 1.0, 0.0, 10.0, (0.0,), element))  # the matrix
        for rule in RULES:
            for c in CONTRASTS:
                out.append(_job("sheet", rule, c, 1.0, 10.0, (0.0,), element))
                for f in FRACTIONS:
                    out.append(_job("sheet", rule, c, f, 10.0, (0.0,), element))
                    for P in PERIODS:
                        out += [_job("sheet", rule, c, f, P, (r,), element) for r in RATIOS]
    if study in ("moduli", "all"):
        for law, soft in (("contrast", STEEL), ("contrast", COPPER), ("matrix", COPPER)):
            for ratios in ((0.0,), (0.2,)):
                out.append(
                    _job(
                        "moduli",
                        "ductile",
                        MARTENSITE,
                        0.24,
                        20.0,
                        ratios,
                        element,
                        law=law,
                        young_soft=soft,
                    )
                )
        for ratios in ((0.0,), (0.2,)):
            out.append(
                _job(
                    "moduli",
                    "ductile",
                    MARTENSITE,
                    0.24,
                    20.0,
                    ratios,
                    element,
                    law="matrix",
                    young_soft=STEEL / 3,
                )
            )
    if study == "stiffness":  # a first step of 0.01 %: the elastic range resolved
        cases = [("contrast", soft, hard) for soft, hard in STIFFNESS]
        cases += [("matrix", STEEL, 3 * STEEL), ("matrix", STEEL / 3, STEEL)]
        for law, soft, hard in cases:
            for ratios in ((0.0,), (0.2,), (0.25,)):
                out.append(
                    _job(
                        "stiffness",
                        "ductile",
                        MARTENSITE,
                        0.24,
                        20.0,
                        ratios,
                        element,
                        law=law,
                        young_soft=soft,
                        young_hard=hard,
                        step=0.0002,
                    )
                )
    if study in ("recruit", "all"):
        for rule in RULES:
            for c in (4.0, MARTENSITE):
                for ratios in RECRUITS.values():
                    gap = spacing(ratios, 0.24)
                    if not crosses(ratios, 20.0, gap):
                        out.append(_job("recruit", rule, c, 0.24, 20.0, ratios, element))
                for r in (0.0, *RATIOS):
                    out.append(_job("sheet", rule, c, 0.24, 20.0 if r else 10.0, (r,), element))
    if study in ("fraser", "all"):
        hc = 5.0  # the height of Fraser's composite: one reinforcement in it
        for h in (0.0, 1.0, 2.0, 3.0, 4.0):  # peak to trough
            ratio = 0.5 * h / 8.0
            t = 0.2 * hc / arc_ratio(ratio)
            out.append(
                _job(
                    "fraser",
                    "ductile",
                    1.0,
                    0.2,
                    8.0,
                    (ratio,),
                    min(element, 0.2),
                    law="fraser",
                    young_soft=115000.0,
                    young_hard=207000.0,
                    width=t,
                    gap=hc,
                    max_strain=0.9,
                )
            )
    if study == "mesh":
        out.append(_job("sheet", "ductile", 1.0, 0.0, 10.0, (0.0,), element))
        for rule in RULES:
            out.append(_job("sheet", rule, MARTENSITE, 0.24, 10.0, (0.0,), element))
            out += [_job("sheet", rule, MARTENSITE, 0.24, 20.0, (r,), element) for r in RATIOS]
    if study in ("crests", "all"):
        for P in (10.0, 20.0):
            for r in (0.15, 0.2, 0.25, 0.3):
                for extent in (0.0625, 0.125):
                    for crest in (1.0, 2.0, 3.0):
                        out.append(
                            _job(
                                "crests",
                                "graded",
                                MARTENSITE,
                                0.24,
                                P,
                                (r,),
                                element,
                                crest=crest,
                                extent=extent,
                            )
                        )
    if study in ("specimen", "all"):
        for rule in RULES:
            for r in (0.0, 0.1, 0.2, 0.3):
                out.append(_job("specimen", rule, MARTENSITE, 0.24, 20.0, (r,), element))
                out.append(
                    _job("sheet", rule, MARTENSITE, 0.24, 20.0 if r else 10.0, (r,), element)
                )
            for r in (0.0, 0.2, 0.3):
                out.append(_job("specimen", rule, MARTENSITE, 0.24, 20.0, (r,), element, count=8))
            for r in (0.2, 0.3):
                out.append(
                    _job(
                        "specimen",
                        rule,
                        MARTENSITE,
                        0.24,
                        20.0,
                        (r,),
                        element,
                        count=8,
                        border=3.0,
                    )
                )
    unique = {}
    for j in out:
        unique.setdefault(j["key"], j)
    return list(unique.values())


def _init() -> None:
    import numba

    from morphogen._platform import full_speed

    full_speed()
    numba.set_num_threads(1)  # one thread per process: the processes share the cores


def run(todo: list, workers: int, done: dict) -> dict:
    todo = [j for j in todo if j["key"] not in done]
    # the specimens and the largest cells first: they take longest
    todo.sort(key=lambda j: (j["study"] != "specimen", j["fraction"]))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        futures = {pool.submit(evaluate, j): j for j in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            done[row["key"]] = row
            if n % 20 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} designs, {time.perf_counter() - t0:.0f} s", flush=True)
    return done


def save(rows: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows.values())
    curves = {r["key"]: np.array(r["curve"]) for r in rows.values()}
    np.savez_compressed(
        path.with_suffix(".npz"),
        keys=np.array(list(curves)),
        **{f"c{i}": v for i, v in enumerate(curves.values())},
    )


def load(path: Path) -> dict:
    rows = {}
    csv_path = path.with_suffix(".csv")
    if not csv_path.exists():
        return rows
    num = set(FIELDS) - {"key", "study", "rule", "ratios", "feasible", "broke", "law"}
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in num:
                r[k] = float(r[k]) if r.get(k) not in (None, "") else math.nan
            for k in ("feasible", "broke"):
                r[k] = r[k] == "True"
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
    ap.add_argument(
        "--study",
        default="all",
        choices=(
            "sheet",
            "moduli",
            "stiffness",
            "recruit",
            "fraser",
            "crests",
            "specimen",
            "mesh",
            "all",
        ),
    )
    ap.add_argument("--element", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.output is None:  # the stiffness study keeps its partitions apart
        name = "_stiffness" if args.study == "stiffness" else ""
        args.output = Path(f"results/laser_geometric_hardening{name}")
    done = load(args.output)
    todo = jobs(args.study, args.element)
    print(f"{len(todo)} designs ({sum(j['key'] in done for j in todo)} done)", flush=True)
    rows = run(todo, args.workers, done)
    save(rows, args.output)
    print(f"written: {args.output}.csv")


if __name__ == "__main__":
    main()
