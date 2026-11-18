"""Reports of RVE studies: Markdown, LaTeX (booktabs), JSON and figures.

The tables follow Dirrenberger et al. (2014): realizations per volume (Table 3), mean
values with ``[Zbar - 2 D_Z, Zbar + 2 D_Z]`` and bounds (Table 4), statistical isotropy
(Tables 5-7), relative errors (Table 8), fitted exponents (Table 9) and RVE sizes
(Table 10), plus the discrepancy between boundary conditions (eq. 69) and the number of
realizations needed per volume. Volumes are printed as ``L^3``, the areas of the square
windows of a planar image as ``L^2``, and the edges of the RVE sizes are ``V^(1/3)`` (the
side ``A^(1/2)`` of a square of area ``A``).

When the realizations have their apparent stiffness
(:class:`~morphogen.rve.study.TensorStatistics`), each boundary condition adds the tables
of the mean stiffness ``Cbar_ij +/- 2 D`` and of the compliance ``S_ij = Cbar^-1 +/- 2 D``
per volume (every component of the upper triangle, Voigt order 11 22 33 23 31 12 with
engineering shears), the RVE sizes of their components, the figure
``rve_stiffness_<bc>.png`` (the components against the volume, their relative scatter and
its fits) and the table ``rve_stiffness.csv`` (one row per condition and volume: ``C11``
... ``C66``, ``S11`` ... ``S66`` and their standard deviations ``C11_std`` ...). The
conductivity tensor likewise, with its inverse the resistivity
(``rve_conductivity_tensor.csv``, the axes 1 2 3); by the window method
(:mod:`morphogen.rve.windows`) the tables count the windows of each size, whose tensors
are their mean mutual energies, as ``n x T`` with ``T`` tilings shifted around the
periodic realization (``n`` the windows of one, the count of the uncertainties).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from morphogen._utils import to_jsonable
from morphogen.rve.study import (
    TENSOR_WORDS,
    TRANSPORT_TENSORS,
    StudyOptions,
    StudyReport,
    TensorStatistics,
)

BC_LABELS = {
    "kubc": "KUBC",
    "subc": "SUBC",
    "mixed": "MBC",
    "periodic": "PBC",
}


def _num(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 1e6 or abs(x) < 1e-3:
        return f"{x:.{digits - 1}e}"
    v = float(f"{x:.{digits}g}")
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(v))))
    return f"{v:.{decimals}f}"


def _count(x) -> str:
    """A number of realizations (an integer; '-' when undefined)."""
    if x is None or not math.isfinite(float(x)):
        return "-"
    return str(int(x))


def _edge(volume: float, dimension: int = 3) -> str:
    """A volume as the power of its edge, ``L^3``, or an area as ``L^2`` (``dimension``
    2: the square windows of a planar image)."""
    edge = volume ** (1.0 / dimension)
    if abs(edge - round(edge)) < 1e-6 * max(1.0, edge):
        return f"{round(edge)}^{dimension}"
    return f"{edge:.4g}^{dimension}"


def _samples(stats, k: int) -> str:
    """The samples of the size ``k`` of statistics: its realizations (windows), ``n x T``
    with ``T`` shifted tilings of ``n`` windows each."""
    n = str(int(stats.n[k]))
    return n if stats.tilings is None else f"{n} x {int(stats.tilings[k])}"


#: The note of the tables that count the windows of shifted tilings
_TILINGS = (
    "n x T: the n windows of each of T tilings shifted around the periodic realization, the "
    "means over all the windows, the variances pooled within the tilings, the uncertainties "
    "counting the n windows of one tiling (the independent ones)."
)


def series_label(analysis) -> str:
    """The label of an analysed series: its property, with its boundary conditions if any."""
    s = analysis.series
    return f"{s.property} ({BC_LABELS.get(s.bc, s.bc)})" if s.bc else s.property


def _volumes(report: StudyReport) -> list[float]:
    vols = set()
    for a in report.analyses:
        vols.update(a.stats.volume.tolist())
    return sorted(vols)


def _per_volume(a, volumes, getter) -> list[str]:
    cells = []
    for v in volumes:
        idx = np.flatnonzero(a.stats.volume == v)
        cells.append(getter(int(idx[0])) if idx.size else "")
    return cells


def tables(report: StudyReport) -> list[tuple[str, list[str], list[list[str]], str]]:
    """The report tables as (title, header, rows, note): those of the series, then those
    of the mean stiffness and compliance of each boundary condition (:func:`tensor_tables`)."""
    out = _series_tables(report) if report.analyses else []
    for t in report.tensors:
        out += tensor_tables(t, report.options)
    return out


#: The notation of the tables of the mean stiffnesses, in space and in the plane
_SHORT_NOTATION = {
    6: "Voigt 11 22 33 23 31 12, engineering shears",
    3: "Voigt 11 22 12, engineering shear",
}


def _short_notation(t: TensorStatistics) -> str:
    """The notation of the tables of a mean tensor: Voigt for a stiffness, the axes for a
    conductivity."""
    if t.property in TRANSPORT_TENSORS:
        return "axes 1 2 3"
    return _SHORT_NOTATION[len(t.mean[0])]


def _words(t: TensorStatistics) -> tuple[str, str, str, str]:
    """The names of a tensor and of its inverse, singular and plural."""
    return TENSOR_WORDS.get(t.property, TENSOR_WORDS["stiffness"])


def _mutual(t: TensorStatistics) -> tuple[str, str]:
    """What the window tensors of ``t`` average, and the unit macroscopic loads."""
    if t.property in TRANSPORT_TENSORS:
        return "grad T_k . K grad T_l, the mutual dissipation", "gradients G_k"
    return "eps_k : C : eps_l, the mutual energy density", "strains E_k"


def tensor_label(t: TensorStatistics) -> str:
    """The label of the mean tensor of a boundary condition: ``stiffness (PBC)``."""
    return f"{t.property} ({BC_LABELS.get(t.bc, t.bc)})" if t.bc else t.property


def tensor_tables(
    t: TensorStatistics, options: StudyOptions | None = None
) -> list[tuple[str, list[str], list[list[str]], str]]:
    """The tables of the mean stiffness and compliance of a boundary condition, as
    (title, header, rows, note): each component of ``Cbar`` and of ``S = Cbar^-1`` per
    volume (mean +/- 2 D), then the RVE sizes of the components that have one (of a
    conductivity and its resistivity likewise; per window size by the window method)."""
    options = options or StudyOptions()
    d = len(t.mean[0])
    where = f"{BC_LABELS.get(t.bc, t.bc)}; " if t.bc else ""
    heads = ["component", *[_edge(v, t.dimension) for v in t.volume]]
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    count = [t.samples, *[_samples(t, k) for k in range(len(t.volume))]]
    one, many, inverse, inverses = _words(t)
    what = (
        f"Cbar: the mean of the apparent {many} of the realizations; D: the standard "
        "deviation of each component."
    )
    if t.method == "windows":
        mutual, loads = _mutual(t)
        what = (
            "Cbar: the mean of the tensors C of the windows of each size, C_kl the mean over "
            f"a window of {mutual} of the solutions of the whole realization under its unit "
            f"macroscopic {loads}; D: the standard deviation of each component over the "
            "windows."
        )
        if t.tilings is not None:
            what += f" {_TILINGS}"

    def cells(mean, std, i, j):
        out = []
        for k in range(len(t.volume)):
            m, dev = mean[k, i, j], std[k, i, j]
            out.append(f"{_num(m)} +/- {_num(2 * dev, 2)}" if math.isfinite(dev) else _num(m))
        return out

    out = []
    for letter, mean, std, title, note in (
        (
            "C",
            t.mean,
            t.std,
            f"Mean {one} Cbar_ij +/- 2 D",
            f"{what} Notation: {t.notation}.",
        ),
        (
            "S",
            t.compliance,
            t.compliance_std,
            f"{inverse.capitalize()} S_ij = Cbar^-1 +/- 2 D",
            f"S: the inverse of the mean {one} (S Cbar = I), not the mean of the "
            f"{inverses} of the {t.samples} (mean_compliance in rve_report.json); D: the "
            f"standard deviation of each component of the {inverses} C^-1 of the "
            f"{t.samples}.",
        ),
    ):
        rows = [count] + [[f"{letter}{i + 1}{j + 1}", *cells(mean, std, i, j)] for i, j in pairs]
        out.append((f"{title} ({where}{_short_notation(t)})", heads, rows, note))
    eps, ns = options.relative_errors, options.n_realizations
    rows = []
    for name, a in t.components.items():
        if not a.rve:
            continue
        edges = {(r["eps_rel"], r["n"]): r["edge"] for r in a.rve}
        size = [_num(edges.get((float(e), int(k))), 4) for e in eps for k in ns]
        rows.append([name, _num(a.reference_mean), _num(a.fit.gamma, 3), _num(a.fit.K, 3), *size])
    if rows:
        gamma = (
            f"gamma = {options.gamma:.4g} imposed" if options.gamma is not None else "fitted gamma"
        )
        heads = ["component", "Zbar", "gamma", "K"]
        heads += [f"edge, eps = {100 * e:g} %, n = {k}" for e in eps for k in ns]
        note = (
            f"D^2(V) = K V^-gamma fitted to each component of the {many} of the "
            f"{t.samples} and of their {inverses} C^-1 (Zbar: their mean at the largest "
            "volume); the edge of V_RVE = (4 K / (eps^2 Zbar^2 n))^(1/gamma). The components "
            f"off the diagonal whose mean the {t.samples} of the largest volume do not "
            f"distinguish from zero (Student's t test at {100 * t.level:g} %), as the "
            "couplings that vanish on average, have none."
        )
        out.append((f"RVE sizes per component ({where}{gamma})", heads, rows, note))
    return out


def _series_tables(report: StudyReport) -> list[tuple[str, list[str], list[list[str]], str]]:
    """The tables of the series of a study (see the module)."""
    vols = _volumes(report)
    dimension = report.dimension
    heads = ["series", *[_edge(v, dimension) for v in vols]]
    out = []
    rows = [
        [series_label(a), *_per_volume(a, vols, lambda i, a=a: _samples(a.stats, i))]
        for a in report.analyses
    ]
    title = "Windows per volume" if report.method == "windows" else "Realizations per volume"
    shifted = any(a.stats.tilings is not None for a in report.analyses)
    out.append((title, heads, rows, _TILINGS if shifted else ""))

    def interval(a, i):
        m, var = a.stats.mean[i], a.stats.variance[i]
        if not math.isfinite(var):
            return _num(m)
        return f"{_num(m)} +/- {_num(2 * math.sqrt(var), 2)}"

    rows = [
        [series_label(a), *_per_volume(a, vols, lambda i, a=a: interval(a, i))]
        for a in report.analyses
    ]
    out.append(("Mean values Zbar +/- 2 D_Z", heads, rows, ""))
    if report.bounds:
        bheads = ["property", "Voigt", "Reuss", "HS lower", "HS upper"]
        brows = [
            [p, *(_num(b.get(k)) for k in ("voigt", "reuss", "hs_lower", "hs_upper"))]
            for p, b in report.bounds.items()
        ]
        out.append(("Bounds", bheads, brows, "Voigt/Reuss are the Wiener bounds in conduction."))
    if report.discrepancy:
        rows = []
        for prop, drows in report.discrepancy.items():
            by_v = {r["volume"]: r for r in drows}
            rows.append(
                [prop, *[_num(by_v[v]["discrepancy"], 3) if v in by_v else "" for v in vols]]
            )
        out.append(
            (
                "Discrepancy between boundary conditions D(Z) = (Zmax - Zmin) / Zmin",
                ["property", *heads[1:]],
                rows,
                "Dirrenberger et al. (2014), eq. 69.",
            )
        )
    iso = [a for a in report.analyses if a.isotropy is not None]
    if iso:
        rows = [
            [series_label(a), *_per_volume(a, vols, lambda i, a=a: _num(a.isotropy[i], 2))]
            for a in iso
        ]
        out.append(
            (
                "Statistical isotropy of the mean response",
                heads,
                rows,
                "Criteria d_k, d_mu (eqs. 73-74) and (|Q2| + |Q3|) / |Q1| (eq. 72); "
                "isotropy when << 1.",
            )
        )

    def rel(a, i):
        e = a.stats.relative_error()[i]
        return f"{100 * e:.3g} %" if math.isfinite(e) else "-"

    rows = [
        [series_label(a), *_per_volume(a, vols, lambda i, a=a: rel(a, i))] for a in report.analyses
    ]
    out.append(
        ("Relative error of the samples eps_rel = 2 D_Z(V) / (Zbar sqrt(n))", heads, rows, "")
    )
    fheads = [
        "series",
        "gamma",
        f"{int(100 * report.options.level)} % interval",
        "K",
        "R^2",
        "sizes",
    ]
    rows = []
    for a in report.analyses:
        f = a.free_fit
        if f is None:
            rows.append([series_label(a), "-", "-", "-", "-", "-"])
            continue
        lo, hi = f.gamma_interval(report.options.level)
        boot = a.bootstrap if report.options.gamma is None else None
        ci = f"[{_num(lo, 3)}, {_num(hi, 3)}]" if math.isfinite(lo) else "-"
        if boot:
            ci += f" (bootstrap [{_num(boot['gamma'][0], 3)}, {_num(boot['gamma'][1], 3)}])"
        rows.append(
            [series_label(a), _num(f.gamma, 3), ci, _num(f.K, 3), _num(f.r2, 3), str(f.n_points)]
        )
    rng = ""
    if report.options.min_volume or report.options.max_volume:
        lo = _edge(report.options.min_volume, dimension) if report.options.min_volume else "-"
        hi = _edge(report.options.max_volume, dimension) if report.options.max_volume else "-"
        rng = f"Fits on volumes in [{lo}, {hi}]."
    out.append(("Scaling of the variance D_Z^2(V) = K V^-gamma (gamma fitted)", fheads, rows, rng))
    gamma_txt = (
        f"gamma = {report.options.gamma:.4g} imposed"
        if report.options.gamma is not None
        else "fitted gamma"
    )
    rheads = ["series", "K", "eps_rel", "n", "V_RVE", "edge"]
    has_count = any("count" in r for a in report.analyses for r in a.rve)
    has_ci = any("edge_ci" in r for a in report.analyses for r in a.rve)
    if has_ci:
        rheads.append(f"edge {int(100 * report.options.level)} % interval")
    if has_count:
        rheads.append("objects")
    rows = []
    for a in report.analyses:
        for r in a.rve:
            row = [
                series_label(a),
                _num(a.fit.K, 3),
                f"{100 * r['eps_rel']:g} %",
                str(r["n"]),
                _num(r["volume"], 3),
                _num(r["edge"], 4),
            ]
            if has_ci:
                ci = r.get("edge_ci")
                row.append(f"[{_num(ci[0], 3)}, {_num(ci[1], 3)}]" if ci else "-")
            if has_count:
                row.append(_num(r.get("count"), 3))
            rows.append(row)
    note = "Zbar is the mean at the largest volume (Dirrenberger et al. 2014, eq. 87)."
    if dimension == 2:
        note += " V_RVE is an area, its edge V_RVE^(1/2) the side of the square window."
    out.append(
        (f"RVE sizes V_RVE = (4 K / (eps^2 Zbar^2 n))^(1/gamma), {gamma_txt}", rheads, rows, note)
    )
    eps = report.options.relative_errors
    nheads = ["series", "volume", *[f"n for eps = {100 * e:g} %" for e in eps]]
    rows = []
    for a in report.analyses:
        for r in a.needed:
            needed = [_count(r[f"eps={e:g}"]) for e in eps]
            rows.append([series_label(a), _edge(r["volume"], dimension), *needed])
    if rows:
        out.append(("Realizations needed per volume", nheads, rows, ""))
    return out


def markdown(report: StudyReport, title: str | None = None) -> str:
    """The Markdown report of an RVE study: the point variance and integral range of each property,
    then the tables of the study."""
    lines = [title or f"# RVE report ({report.method} method)", ""]
    for a in report.analyses:
        if a.point_variance is not None and a.integral_range is not None:
            lines.append(
                f"{series_label(a)}: point variance D^2 = {_num(a.point_variance)}, "
                f"A3* = {_num(a.integral_range)}"
            )
    if lines[-1] != "":
        lines.append("")
    for name, head, rows, note in tables(report):
        lines += [
            f"## {name}",
            "",
            "| " + " | ".join(head) + " |",
            "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|",
        ]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        lines.append("")
        if note:
            lines += [note, ""]
    return "\n".join(lines)


#: An exponent of the text of the tables: ^3, ^-1, ^-gamma, ^(1/gamma)
_EXPONENT = re.compile(r"\^(\([^()]*\)|-?\w+)")


def _tex(s: str) -> str:
    s = (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("+/-", r"$\pm$")
        .replace("&", r"\&")
    )
    return _EXPONENT.sub(lambda m: f"$^{{{m[1]}}}$", s)


def latex(report: StudyReport) -> str:
    """The tables in LaTeX (booktabs package)."""
    parts = ["% RVE report written by morphoGen (needs \\usepackage{booktabs})", ""]
    for name, head, rows, note in tables(report):
        parts += [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{_tex(name)}}}",
            r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}",
            r"\toprule",
            " & ".join(_tex(h) for h in head) + r" \\",
            r"\midrule",
        ]
        parts += [" & ".join(_tex(c) for c in r) + r" \\" for r in rows]
        parts += [r"\bottomrule", r"\end{tabular}"]
        if note:
            parts.append(f"\\par\\smallskip{{\\footnotesize {_tex(note)}}}")
        parts += [r"\end{table}", ""]
    return "\n".join(parts)


def _measure_axis(dimension: int, windows: bool = False) -> str:
    """The axis of the volumes V of the figures: of cubes, or the areas of the squares of a
    planar image (``dimension`` 2); ``windows``, of the windows of the study."""
    return ("window " if windows else "") + ("area V" if dimension == 2 else "volume V")


def plot(report: StudyReport, outdir: Path) -> list[Path]:
    """Figures per property: variance, mean +/- 2 D_Z and RVE size (needs matplotlib),
    against the volumes, or the areas of the squares of a planar image."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    outdir = Path(outdir)
    files = []
    by_prop: dict[str, list] = {}
    for a in report.analyses:
        by_prop.setdefault(a.series.property, []).append(a)
    for prop, group in by_prop.items():
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for a in group:
            st = a.stats
            ok = np.isfinite(st.variance) & (st.variance > 0)
            if not ok.any():
                continue
            label = BC_LABELS.get(a.series.bc, a.series.bc) or "data"
            (line,) = ax.loglog(st.volume[ok], st.variance[ok], "o", label=label)
            for fit, style in ((a.free_fit, "--"), (a.fit if report.options.gamma else None, "-")):
                if fit is None:
                    continue
                vv = np.geomspace(st.volume[ok].min(), st.volume[ok].max(), 50)
                ax.loglog(
                    vv,
                    fit.variance(vv),
                    style,
                    color=line.get_color(),
                    label=f"{label}: gamma = {fit.gamma:.3g}",
                )
        ax.set_xlabel(_measure_axis(report.dimension))
        ax.set_ylabel(f"D^2(V) of {prop}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = outdir / f"rve_{prop}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        files.append(path)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for a in group:
            st = a.stats
            err = 2.0 * np.sqrt(np.where(np.isfinite(st.variance), st.variance, 0.0))
            ax.errorbar(
                st.volume,
                st.mean,
                yerr=err,
                fmt="o-",
                capsize=3,
                label=BC_LABELS.get(a.series.bc, a.series.bc) or prop,
            )
        for name, value in report.bounds.get(prop, {}).items():
            if name in ("hs_upper", "voigt") and value > 0:
                ax.axhline(value, ls=":", color="gray")
                ax.annotate(
                    name.replace("_", " "),
                    (ax.get_xlim()[0], value),
                    fontsize=7,
                    va="bottom",
                    color="gray",
                )
        ax.set_xscale("log")
        ax.set_xlabel(_measure_axis(report.dimension))
        ax.set_ylabel(f"{prop}: mean +/- 2 D")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = outdir / f"rve_{prop}_mean.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        files.append(path)
        fits = [a for a in group if a.fit is not None and a.reference_mean != 0]
        if fits:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            eps = np.geomspace(0.005, 0.2, 40)
            dimension = report.dimension  # the edge of a cube, or of a square (its area V)
            for a in fits:
                for k in report.options.n_realizations:
                    edge = a.fit.rve_volume(eps, a.reference_mean, k) ** (1.0 / dimension)
                    lab = BC_LABELS.get(a.series.bc, a.series.bc) or prop
                    ax.loglog(100 * eps, edge, label=f"{lab}, n = {k}")
            ax.set_xlabel("relative error eps_rel (%)")
            ax.set_ylabel(f"RVE edge V_RVE^(1/{dimension})")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            fig.tight_layout()
            path = outdir / f"rve_{prop}_size.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            files.append(path)
    for t in report.tensors:
        files.append(_tensor_figure(plt, t, outdir))
    return files


def _tensor_figure(plt, t: TensorStatistics, outdir: Path) -> Path:
    """The convergence of the mean stiffness and compliance of a boundary condition with
    the volume (the area of the squares of a planar image): the components of ``Cbar`` and
    of ``S = Cbar^-1`` (+/- 2 D) that the realizations (windows) distinguish from zero, and
    their relative scatter ``D / |Zbar|`` with the fitted laws (dashed); of a conductivity
    and its resistivity likewise."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    one, _, inverse, _ = _words(t)
    tensors = (("C", t.mean, t.std, f"{one} Cbar"), ("S", t.compliance, t.compliance_std,
               f"{inverse} S = Cbar^-1"))  # fmt: skip
    for col, (letter, mean, std, what) in enumerate(tensors):
        top, bottom = axes[0, col], axes[1, col]
        names = [n for n in t.components if n[0] == letter and t.significant(n)]
        for name in names:
            i, j = t.index(name)
            bars = top.errorbar(t.volume, mean[:, i, j], yerr=2.0 * np.nan_to_num(std[:, i, j]),
                                fmt="o-", ms=3, capsize=2, label=name)  # fmt: skip
            a = t.components[name]
            st, ref = a.stats, abs(a.reference_mean)
            ok = np.isfinite(st.variance) & (st.variance > 0)
            if not ok.any() or ref == 0:
                continue
            color = bars[0].get_color()
            bottom.loglog(st.volume[ok], np.sqrt(st.variance[ok]) / ref, "o", ms=3, color=color)
            if a.fit is not None:
                vv = np.geomspace(st.volume[ok].min(), st.volume[ok].max(), 50)
                bottom.loglog(vv, np.sqrt(a.fit.variance(vv)) / ref, "--", lw=1, color=color)
        top.set_xscale("log")
        top.set_title(f"{what}: mean +/- 2 D", fontsize=9)
        bottom.set_xlabel(_measure_axis(t.dimension, t.method == "windows"))
        bottom.set_ylabel(f"D / |Zbar| of the components of {letter}")
        if names:
            top.legend(fontsize=7, ncol=3)
    fig.suptitle(f"{tensor_label(t)}: {_short_notation(t)}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"rve_{t.key.replace('.', '_')}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write(
    report: StudyReport, outdir, *, stem: str = "rve_report", figures: bool = True
) -> list[Path]:
    """Write ``<stem>.json``, ``.md`` and ``.tex`` (and the figures) in ``outdir``, and
    ``rve_<tensor>.csv`` (the mean stiffness and compliance per boundary condition and
    volume, :meth:`~morphogen.rve.study.TensorStatistics.volume_rows`) when the study has
    them."""
    from morphogen.io.tables import write_table

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    path = outdir / f"{stem}.json"
    path.write_text(json.dumps(to_jsonable(report.to_dict()), indent=2) + "\n", encoding="utf-8")
    files.append(path)
    by_tensor: dict[str, list[dict]] = {}
    for t in report.tensors:
        by_tensor.setdefault(t.property, []).extend(t.volume_rows())
    for name, rows in by_tensor.items():
        files.append(write_table(outdir / f"rve_{name}.csv", rows))
    path = outdir / f"{stem}.md"
    path.write_text(markdown(report), encoding="utf-8")
    files.append(path)
    path = outdir / f"{stem}.tex"
    path.write_text(latex(report), encoding="utf-8")
    files.append(path)
    if figures:
        files += plot(report, outdir)
    return files
