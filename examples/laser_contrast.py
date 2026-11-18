"""Architectured sheets by localized laser treatment: the strength contrast of the
reinforcement, the third axis of the design space (docs/laser.md, §5).

The design maps of §5 (the height over the period of a corrugated track against the
treated fraction; Wang et al. 2023, Fig. 10) for reinforcements between the annealed
0.23 C steel and its laser martensite (``contrast_phases``: a yield stress 1.5 to 4.64
times the matrix's), under three rules of fracture of the reinforcement (``graded``,
and its bounds ``brittle`` and ``ductile``): the dog-bone of Wang et al., one track
along it, of constant area, periods of 10 and 15 mm. Measures: the yield and tensile
strengths, the failure strain (the first of the necking and of the fracture of the
reinforcement) and the energy absorbed up to it; each phase alone for reference.
Feasible designs: a radius of curvature of the track of at least 1.5 mm (the cycles of
a straight track; the constraint of the template ``laser_corrugated``) and the track
within the width of the gauge (beyond it the specimen holds pieces of oblique tracks,
and less of the treatment).

Writes every design to a CSV file; ``--figure`` draws the figure of docs/laser.md. The
designs run in parallel over the cores (about ten minutes on 16 processes, elements of
0.5 mm); ``--element 0.25 --fractions 0.24 --periods 10 --max-ratio 0.15`` checks the
mesh.
"""

import argparse
import csv
import itertools
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CONTRASTS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 1300.0 / 280.0)  # the last: the martensite
FRACTIONS = (0.12, 0.24, 0.36, 0.48)
#: the heights over period of the track for each period (mm), the longer period up to
#: where its track leaves the gauge
RATIOS = {
    10.0: (0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3),
    15.0: (0.025, 0.05, 0.075, 0.1, 0.125, 0.15),
}
RULES = ("graded", "brittle", "ductile")
MIN_RADIUS = 1.5  # mm
EDGE = 0.05  # mm: how far the edge of a track may pass the edge of the gauge
GAUGE = 5.0  # mm: the width of the gauge of the dog-bone
FIELDS = (
    "rule",
    "contrast",
    "fraction",
    "ratio",
    "period",
    "amplitude",
    "width",
    "radius",
    "margin",
    "feasible",
    "treated",
    "yield_strength",
    "tensile_strength",
    "uniform_elongation",
    "failure_strain",
    "energy",
    "broke",
)

_PROBLEMS: dict = {}


def _full_speed() -> None:
    """Opt the process out of the power throttling of Windows (EcoQoS), which runs the
    processes of a session in the background on the efficiency cores at a low clock: a
    test then takes five to seven times longer."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [(k, wintypes.ULONG) for k in ("Version", "ControlMask", "StateMask")]

    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    state = State(1, 1, 0)  # the execution speed never throttled
    k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))


def _threads(n: int) -> None:
    import numba

    _full_speed()
    numba.set_num_threads(n)


def evaluate(job) -> dict:
    """The tensile test of one design: ``(rule, contrast, fraction, ratio, period,
    element)``; ``fraction`` 0: the matrix alone, 1: the reinforcement alone."""
    from morphogen.laser.cells import (
        CellProblem,
        band_fractions,
        constant_area_width,
        contrast_phases,
        corrugation,
    )

    rule, contrast, fraction, ratio, period, element = job
    key = (rule, contrast, element)
    if key not in _PROBLEMS:
        _PROBLEMS[key] = CellProblem(
            phases=contrast_phases(contrast, rule), element=element, size=1.0
        )
    problem = _PROBLEMS[key]
    spec = problem.specimen
    amplitude = ratio * period if period else 0.0
    corrugated = amplitude > 0
    width = constant_area_width(fraction * spec.width, amplitude, period if corrugated else None)
    if fraction in (0.0, 1.0):
        share = np.full((len(problem.mesh.quads), 4), fraction)
    else:
        lines = corrugation(spec, amplitude, period if corrugated else None)
        share = band_fractions(problem.mesh, lines, width)
    m = problem.evaluate(fractions=np.stack([1 - share, share], axis=-1)).measures
    radius = period**2 / (4 * math.pi**2 * amplitude) if corrugated else math.inf
    margin = 0.5 * spec.width - (amplitude + 0.5 * width)
    return {
        "rule": rule,
        "contrast": contrast,
        "fraction": fraction,
        "ratio": ratio,
        "period": period if corrugated else 0.0,
        "amplitude": amplitude,
        "width": width,
        "radius": radius,
        "margin": margin,
        "feasible": radius >= MIN_RADIUS and margin >= -EDGE,
        "treated": m["budget"],
        "yield_strength": m["yield_strength"],
        "tensile_strength": m["tensile_strength"],
        "uniform_elongation": m["uniform_elongation"],
        "failure_strain": m["failure_strain"],
        "energy": m["energy"],
        "broke": m["failure_strain"] < m["uniform_elongation"] - 1e-9,
    }


def jobs(element, contrasts=CONTRASTS, fractions=FRACTIONS, periods=tuple(RATIOS), top=1.0):
    out = [("ductile", 1.0, 0.0, 0.0, 0.0, element)]  # the matrix alone
    for rule in RULES:
        for c in contrasts:
            out.append((rule, c, 1.0, 0.0, 0.0, element))  # the reinforcement alone
            for f in fractions:
                out.append((rule, c, f, 0.0, 0.0, element))  # straight
                for P in periods:
                    out += [(rule, c, f, r, P, element) for r in RATIOS[P] if r <= top]
    return out


def run(todo: list, workers: int) -> list:
    with ProcessPoolExecutor(workers, initializer=_threads, initargs=(1,)) as pool:
        return list(pool.map(evaluate, todo, chunksize=4))


def write(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)


def read(path: Path) -> list:
    num = set(FIELDS) - {"rule", "feasible", "broke"}
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in num:
            r[k] = float(r[k])
        for k in ("feasible", "broke"):
            r[k] = r[k] == "True"
    return rows


# -- reading the results -----------------------------------------------------------------


def select(rows, **match) -> list:
    return [
        r
        for r in rows
        if all(
            (abs(r[k] - v) < 1e-9 if isinstance(v, float) else r[k] == v) for k, v in match.items()
        )
    ]


def designs(rows, rule=None) -> list:
    """The designs with a track (neither phase alone), of one rule."""
    return [r for r in rows if 0 < r["fraction"] < 1 and rule in (None, r["rule"])]


def values(rows, key) -> list:
    return sorted({r[key] for r in designs(rows)})


def straight(rows, rule, c, f) -> dict:
    return select(rows, rule=rule, contrast=c, fraction=f, ratio=0.0)[0]


def best_corrugation(rows, rule, c, f, measure="failure_strain") -> dict:
    """The feasible corrugated design (either period) of the largest ``measure``."""
    cand = [r for r in select(rows, rule=rule, contrast=c, fraction=f) if r["ratio"] > 0]
    return max((r for r in cand if r["feasible"]), key=lambda r: r[measure])


def pareto(rows, a: str, b: str) -> list:
    """The designs that no other beats in both ``a`` and ``b``, by increasing ``a``."""
    out, top = [], -math.inf
    for r in sorted(rows, key=lambda r: (-r[a], -r[b])):
        if r[b] > top:
            out.append(r)
            top = r[b]
    return out[::-1]


def ratio_limit(fraction: float, period: float, width: float = GAUGE) -> float:
    """The largest feasible height over period of a track of ``fraction`` and
    ``period``: its radius of curvature and its edges in the gauge."""
    from morphogen.laser.cells import constant_area_width

    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        t = constant_area_width(fraction * width, mid * period, period)
        lo, hi = (mid, hi) if mid * period + 0.5 * t <= 0.5 * width + EDGE else (lo, mid)
    return min(period / (4 * math.pi**2 * MIN_RADIUS), lo)


def tables(rows) -> None:
    matrix = select(rows, fraction=0.0)[0]
    print(
        f"matrix alone: R_p {matrix['yield_strength']:.0f}, R_m {matrix['tensile_strength']:.0f}"
        f" MPa, failure {matrix['failure_strain']:.3f}, energy {matrix['energy']:.0f} MJ/m3"
    )
    contrasts, fractions = values(rows, "contrast"), values(rows, "fraction")
    print("reinforcement alone, R_m (MPa) / failure strain / energy (MJ/m3)")
    print("  c       " + "".join(f"{rule:24s}" for rule in RULES))
    for c in contrasts:
        line = f"  {c:4.2f}    "
        for rule in RULES:
            r = select(rows, rule=rule, contrast=c, fraction=1.0)[0]
            line += (
                f"{r['tensile_strength']:5.0f} {r['failure_strain']:.3f} {r['energy']:5.0f}     "
            )
        print(line)

    f = 0.24 if 0.24 in fractions else fractions[0]
    ratios = sorted({r["ratio"] for r in designs(rows) if r["period"] in (0.0, 10.0)})
    limit = ratio_limit(f, 10.0)
    for rule in RULES:
        print(f"\n{rule}, {100 * f:.0f} % treated, P 10 mm (feasible up to H/P {limit:.3f})")
        for measure, name, fmt, scale in (
            ("failure_strain", "failure strain (%)", "{:6.1f}", 100.0),
            ("tensile_strength", "R_m (MPa)", "{:6.0f}", 1.0),
            ("energy", "energy (MJ/m3)", "{:6.0f}", 1.0),
        ):
            print(f"  {name:20s} H/P " + "".join(f"{r:6.3f}" for r in ratios))
            for c in contrasts:
                cells = []
                for r in ratios:
                    d = select(rows, rule=rule, contrast=c, fraction=f, ratio=r)
                    d = next(x for x in d if r == 0 or x["period"] == 10.0)
                    cells.append(fmt.format(scale * d[measure]))
                print(f"  c {c:4.2f}                  " + "".join(cells))

    print("\nThe best feasible corrugation against the straight track (either period)")
    print("  rule      f     c     straight R_m/fail/energy   best: P, H/P, R_m/fail/energy")
    for rule in RULES:
        for f in fractions:
            for c in contrasts:
                s = straight(rows, rule, c, f)
                b = best_corrugation(rows, rule, c, f)
                gain = 100 * (b["failure_strain"] / s["failure_strain"] - 1)
                cost = 100 * (b["tensile_strength"] / s["tensile_strength"] - 1)
                print(
                    f"  {rule:8s} {f:4.2f}  {c:4.2f}  {s['tensile_strength']:5.0f} "
                    f"{s['failure_strain']:.3f} {s['energy']:5.1f}       P {b['period']:2.0f} "
                    f"H/P {b['ratio']:.3f}  {b['tensile_strength']:5.0f} "
                    f"{b['failure_strain']:.3f} {b['energy']:5.1f}  ({gain:+.0f} % failure, "
                    f"{cost:+.0f} % R_m)"
                )

    print("\nThe best feasible designs (graded rule), each objective")
    feas = [r for r in designs(rows, "graded") if r["feasible"]]
    for measure in ("tensile_strength", "failure_strain", "energy"):
        b = max(feas, key=lambda r: r[measure])
        print(
            f"  {measure:18s} c {b['contrast']:4.2f}, {100 * b['fraction']:.0f} %, "
            f"P {b['period']:.0f}, H/P {b['ratio']:.3f}: R_p {b['yield_strength']:.0f}, R_m "
            f"{b['tensile_strength']:.0f}, failure {b['failure_strain']:.3f}, energy "
            f"{b['energy']:.0f}"
        )
    print("  the front of R_m and failure strain:")
    for r in pareto(feas, "tensile_strength", "failure_strain"):
        print(
            f"    R_m {r['tensile_strength']:5.0f}  failure {r['failure_strain']:.3f}  c "
            f"{r['contrast']:4.2f}  {100 * r['fraction']:.0f} %  P {r['period']:2.0f}  H/P "
            f"{r['ratio']:.3f}  energy {r['energy']:.0f}"
        )


# -- the figure ----------------------------------------------------------------------------

INK, MUTED, GRID, AXIS, SHADE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#f0efec"
#: an ordinal blue ramp, light to dark (the contrast)
BLUES = ("#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b")
SHOWN = (1.5, 2.5, 3.5, 1300.0 / 280.0)  # the contrasts drawn as curves
STYLES = {"graded": ("-", "o"), "brittle": ("--", "s"), "ductile": (":", "^")}


def _axes(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=8.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _ramp(c: float) -> str:
    return BLUES[int(np.argmin([abs(c - x) for x in CONTRASTS]))]


def figure(rows, path: Path, fraction: float = 0.24, period: float = 10.0) -> None:
    """(a) the failure strain and (b) the energy against the height over the period
    (graded rule, ``fraction``, ``period``), one curve per contrast, the infeasible
    designs shaded; (c) the gain of failure strain of the best feasible corrugation over
    the straight track against the contrast, each rule; (d) the tensile strength against
    the failure strain of every feasible design (graded), and their front."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors as mcolors
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = [n for n in ("Inter", "Segoe UI", "DejaVu Sans") if n in names]
    plt.rcParams["font.family"] = sans[:1] or ["sans-serif"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), layout="constrained")
    matrix = select(rows, fraction=0.0)[0]
    (a, b), (c_ax, d) = axes
    limit = ratio_limit(fraction, period)
    top = max(r["ratio"] for r in designs(rows) if r["period"] == period)
    for ax, measure, label, scale in (
        (a, "failure_strain", "failure strain (%)", 100.0),
        (b, "energy", "energy absorbed to failure (MJ/m³)", 1.0),
    ):
        for c in SHOWN:
            pts = [
                r
                for r in select(rows, rule="graded", contrast=c, fraction=fraction)
                if r["ratio"] == 0 or r["period"] == period
            ]
            pts.sort(key=lambda r: r["ratio"])
            ax.plot(
                [r["ratio"] for r in pts],
                [scale * r[measure] for r in pts],
                color=_ramp(c),
                lw=2,
                marker="o",
                ms=4.5,
                label=f"c = {c:g}" if c < 4.6 else f"c = {c:.2f} (martensite)",
            )
        ax.axvspan(limit, top + 0.01, color=SHADE, zorder=0)
        ax.text(
            limit + 0.006,
            0.03,
            "infeasible: radius of\ncurvature below 1.5 mm",
            transform=ax.get_xaxis_transform(),
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
        ref = scale * matrix[measure]
        ax.axhline(ref, color=MUTED, lw=1, ls=":", zorder=1)
        ax.annotate(
            "matrix alone",
            (top + 0.008, ref),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
        ax.set_xlim(-0.008, top + 0.01)
        ax.set_xlabel("height over period of the track H/P")
        ax.set_ylabel(label)
    _axes(a, f"(a) failure strain, graded rule ({100 * fraction:.0f} % treated, P {period:g} mm)")
    _axes(
        b, f"(b) energy to failure, graded rule ({100 * fraction:.0f} % treated, P {period:g} mm)"
    )
    a.legend(
        frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.28), labelcolor=INK
    )

    contrasts = values(rows, "contrast")
    for rule, (ls, mk) in STYLES.items():
        gains = []
        for c in contrasts:
            s = straight(rows, rule, c, fraction)
            g = best_corrugation(rows, rule, c, fraction)
            gains.append(100 * (g["failure_strain"] / s["failure_strain"] - 1))
        c_ax.plot(contrasts, gains, ls=ls, marker=mk, color=INK, lw=1.6, ms=6, label=rule)
    c_ax.axhline(0.0, color=AXIS, lw=1)
    c_ax.set_xlabel("strength contrast c (yield stress of the reinforcement over the matrix's)")
    c_ax.set_ylabel("gain of failure strain (%)")
    _axes(
        c_ax,
        f"(c) the best feasible corrugation against the straight track ({100 * fraction:.0f} %)",
    )
    c_ax.legend(
        frameon=False,
        fontsize=8.5,
        title="fracture of the reinforcement",
        title_fontsize=8.5,
        labelcolor=INK,
    )

    feas = [r for r in designs(rows, "graded") if r["feasible"]]
    cmap = mcolors.ListedColormap(BLUES)
    mids = (0.5 * (x + y) for x, y in itertools.pairwise(CONTRASTS))
    norm = mcolors.BoundaryNorm([CONTRASTS[0] - 0.25, *mids, CONTRASTS[-1] + 0.25], cmap.N)
    for corrugated, marker in ((True, "o"), (False, "s")):
        pts = [r for r in feas if (r["ratio"] > 0) == corrugated]
        d.scatter(
            [100 * r["failure_strain"] for r in pts],
            [r["tensile_strength"] for r in pts],
            c=[r["contrast"] for r in pts],
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=20 if corrugated else 34,
            edgecolors="white" if corrugated else INK,
            linewidths=0.5 if corrugated else 0.9,
            label="corrugated tracks" if corrugated else "straight tracks",
            zorder=3 if corrugated else 4,
        )
    front = pareto(feas, "tensile_strength", "failure_strain")
    d.plot(
        [100 * r["failure_strain"] for r in front],
        [r["tensile_strength"] for r in front],
        color=INK,
        lw=1.2,
        zorder=2,
        label="front",
    )
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=d, ticks=CONTRASTS, pad=0.01)
    bar.ax.set_yticklabels([f"{c:.2f}" if c > 4.6 else f"{c:g}" for c in CONTRASTS])
    bar.set_label("contrast c", color=MUTED)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.outline.set_visible(False)
    d.set_xlabel("failure strain (%)")
    d.set_ylabel("tensile strength R_m (MPa)")
    _axes(d, "(d) every feasible design, graded rule: strength against failure strain")
    d.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--element", type=float, default=0.5, help="element size (mm)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--contrasts", type=float, nargs="+", default=CONTRASTS)
    parser.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    parser.add_argument("--periods", type=float, nargs="+", default=tuple(RATIOS))
    parser.add_argument("--max-ratio", type=float, default=1.0, help="of height over period")
    parser.add_argument("--output", type=Path, default=Path("results/laser_contrast.csv"))
    parser.add_argument("--reuse", action="store_true", help="read the CSV instead of running")
    parser.add_argument("--figure", type=Path, help="draw the figure to this file")
    args = parser.parse_args()
    if args.reuse:
        rows = read(args.output)
    else:
        todo = jobs(args.element, args.contrasts, args.fractions, args.periods, args.max_ratio)
        t0 = time.perf_counter()
        rows = run(todo, args.workers)
        write(rows, args.output)
        print(f"{len(rows)} designs in {time.perf_counter() - t0:.0f} s -> {args.output}\n")
    tables(rows)
    if args.figure:
        figure(rows, args.figure)
        print(f"\nfigure: {args.figure}")


if __name__ == "__main__":
    main()
