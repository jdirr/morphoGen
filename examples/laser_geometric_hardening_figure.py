"""The figures of docs/laser.md, §15: geometric work hardening in one laser-treated steel.

Reads ``results/laser_geometric_hardening.csv`` and ``.npz``
(``examples/laser_geometric_hardening.py``) and draws three figures. The first: the
Considère plot of the matrix, straight and corrugated tracks; the failure strain against
the height over the period; the tensile strength against the failure strain (the
property chart of Fraser et al.); the stiffness contrast against the strength contrast.
The second: the copper and steel of Fraser et al. (2018) against their necking strains;
softer crests; the specimens against the sheet. The third
(``results/laser_geometric_hardening_stiffness``): how contrasts of stiffness and of
strength share the stresses and the strains between tracks and matrix, and the uniform
elongation against the moduli.

Usage: python examples/laser_geometric_hardening_figure.py [directory]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import laser_geometric_hardening as g

INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e2dc"
#: categorical hues in a fixed order (the palette of the other figures of the study)
HUES = ("#2a6fdb", "#e3803a", "#1f9e89", "#c2457a", "#7c5cc4", "#8a8a86")


def select(rows, **match) -> list:
    out = []
    for r in rows.values():
        ok = True
        for k, v in match.items():
            x = r[k]
            if isinstance(v, float):
                ok &= isinstance(x, float) and abs(x - v) < 1e-6
            else:
                ok &= x == v
        if ok:
            out.append(r)
    return out


def one(rows, **match) -> dict:
    found = select(rows, **match)
    if not found:
        raise KeyError(match)
    return found[0]


def considere(curve):
    """True strain, true stress and the work-hardening rate of a nominal curve."""
    e, s = (np.asarray(v, dtype=float) for v in curve)
    keep = e > 0
    et, st = np.log1p(e[keep]), s[keep] * (1.0 + e[keep])
    # the rate on a smoothed curve: a central difference over about 0.01 of strain
    grid = np.linspace(et[0], et[-1], max(int((et[-1] - et[0]) / 0.0025), 8))
    sg = np.interp(grid, et, st)
    k = 2
    theta = np.full_like(sg, np.nan)
    theta[k:-k] = (sg[2 * k :] - sg[: -2 * k]) / (grid[2 * k :] - grid[: -2 * k])
    return grid, sg, theta


def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = [n for n in ("Inter", "Segoe UI", "DejaVu Sans") if n in names]
    plt.rcParams["font.family"] = sans[:1] or ["sans-serif"]
    rows = g.load(Path("results/laser_geometric_hardening"))
    sheet = {k: r for k, r in rows.items() if r["study"] == "sheet" and r["element"] == 0.5}
    M = g.MARTENSITE
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.2), layout="constrained")

    # (a) the Considère plot: geometric hardening of corrugated tracks (ductile rule)
    ax = axes[0, 0]
    matrix = one(sheet, fraction=0.0)
    cases = [("untreated sheet", matrix)]
    cases.append(
        ("straight tracks", one(sheet, rule="ductile", contrast=M, fraction=0.24, ratios="0"))
    )
    for r in (0.1, 0.2, 0.3):
        row = one(sheet, rule="ductile", contrast=M, fraction=0.24, period=20.0, ratios=f"{r:g}")
        cases.append((f"H/P = {r:g}", row))
    for k, (label, row) in enumerate(cases):
        et, st, th = considere(row["curve"])
        c = MUTED if k == 0 else HUES[k - 1]
        ax.plot(et, st, color=c, lw=2.0, label=label)
        ax.plot(et, th, color=c, lw=1.2, ls=(0, (4, 2)))
        j = int(np.nanargmin(np.abs(th - st) + np.where(et < 0.03, 1e9, 0.0)))
        ax.plot(et[j], st[j], "o", ms=5, color=c, mec="white", mew=1.0)
    ax.set_xlim(0, 0.42)
    ax.set_ylim(0, 2000)
    ax.set_xlabel("true strain", color=INK, fontsize=9)
    ax.set_ylabel("true stress, solid; work-hardening rate, dashed (MPa)", color=INK, fontsize=9)
    ax.set_title(
        "a. Considère's plot: tracks of martensite, 24 %, P = 20 mm (ductile)",
        loc="left",
        fontsize=9.5,
        color=INK,
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    style(ax)

    # (b) the failure strain against the height over the period (martensite, 24 %)
    ax = axes[0, 1]
    for k, P in enumerate(g.PERIODS):
        for rule, ls in (("ductile", (0, (4, 2))), ("graded", "-")):
            pts = [one(sheet, rule=rule, contrast=M, fraction=0.24, ratios="0")]
            pts += sorted(
                select(sheet, rule=rule, contrast=M, fraction=0.24, period=P),
                key=lambda r: float(r["ratios"]),
            )
            x = np.array([float(r["ratios"]) for r in pts])
            y = np.array([100 * r["failure_strain"] for r in pts])
            ok = np.array([g.radius(P, v) >= g.MIN_RADIUS for v in x])
            ax.plot(x, y, color=HUES[k], lw=2.0 if rule == "graded" else 1.2, ls=ls, alpha=0.35)
            ax.plot(
                np.where(ok, x, np.nan),
                np.where(ok, y, np.nan),
                color=HUES[k],
                lw=2.0 if rule == "graded" else 1.2,
                ls=ls,
                label=f"P = {P:g} mm" if rule == "graded" else None,
            )
    ax.axhline(100 * matrix["failure_strain"], color=MUTED, lw=1.0, ls=":")
    ax.text(
        0.302,
        100 * matrix["failure_strain"] + 0.6,
        "untreated",
        fontsize=7,
        color=MUTED,
        ha="right",
    )
    ax.axvline(0.15, color=MUTED, lw=0.8)
    ax.text(0.152, 4.0, "5 mm gauge\n(Wang et al.)", fontsize=7, color=MUTED)
    ax.axvline(0.175, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.text(0.177, 1.0, "threshold of\nFraser et al.", fontsize=7, color=MUTED)
    ax.set_xlabel("height over period H/P", color=INK, fontsize=9)
    ax.set_ylabel("failure strain (%)", color=INK, fontsize=9)
    ax.set_title(
        "b. Martensite, 24 %: graded rule solid, ductile dashed (faint: R < 1.5 mm)",
        loc="left",
        fontsize=9.5,
        color=INK,
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_xlim(0, 0.31)
    ax.set_ylim(0, None)
    style(ax)

    # (c) the property chart: tensile strength against failure strain
    ax = axes[1, 0]
    ax.plot(100 * matrix["failure_strain"], matrix["tensile_strength"], "s", color=MUTED, ms=7)
    ax.annotate(
        "untreated",
        (100 * matrix["failure_strain"], matrix["tensile_strength"]),
        xytext=(4, -12),
        textcoords="offset points",
        fontsize=7.5,
        color=MUTED,
    )
    for k, c in enumerate(g.CONTRASTS):
        straight = [
            one(sheet, rule="graded", contrast=c, fraction=f, ratios="0") for f in g.FRACTIONS
        ]
        x = [100 * matrix["failure_strain"]] + [100 * r["failure_strain"] for r in straight]
        y = [matrix["tensile_strength"]] + [r["tensile_strength"] for r in straight]
        ax.plot(x, y, color=HUES[k], lw=1.0, ls=(0, (4, 2)), marker="o", ms=3.5)
        best = []
        for f in g.FRACTIONS:
            cand = [
                r
                for r in select(sheet, rule="graded", contrast=c, fraction=f)
                if r["feasible"] and float(r["ratios"]) > 0
            ]
            if cand:
                best.append(max(cand, key=lambda r: r["energy"]))
        bx = [100 * matrix["failure_strain"]] + [100 * r["failure_strain"] for r in best]
        by = [matrix["tensile_strength"]] + [r["tensile_strength"] for r in best]
        label = "martensite" if abs(c - M) < 1e-9 else f"contrast {c:g}"
        ax.plot(bx, by, color=HUES[k], lw=2.0, marker="o", ms=4.5, label=label)
    duct = []
    for f in g.FRACTIONS:
        cand = [
            r
            for r in select(sheet, rule="ductile", contrast=M, fraction=f)
            if r["feasible"] and float(r["ratios"]) > 0
        ]
        duct.append(max(cand, key=lambda r: r["energy"]))
    ax.plot(
        [100 * matrix["failure_strain"]] + [100 * r["failure_strain"] for r in duct],
        [matrix["tensile_strength"]] + [r["tensile_strength"] for r in duct],
        color=HUES[2],
        lw=1.2,
        ls=(0, (1, 1.5)),
        marker="o",
        ms=5,
        mfc="white",
        label="martensite, ductile bound",
    )
    ax.set_xlabel("failure strain (%)", color=INK, fontsize=9)
    ax.set_ylabel("tensile strength (MPa)", color=INK, fontsize=9)
    ax.set_title(
        "c. 12, 24, 36 % treated: straight dashed, best corrugation solid",
        loc="left",
        fontsize=9.5,
        color=INK,
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    style(ax)

    # (d) the stiffness contrast against the strength contrast
    ax = axes[1, 1]
    mod = {k: r for k, r in rows.items() if r["study"] == "moduli"}
    groups = [
        ("one steel,\nstrength contrast", "contrast", g.STEEL),
        ("strength contrast\n+ moduli 117/210", "contrast", g.COPPER),
        ("moduli 117/210\nalone", "matrix", g.COPPER),
        ("moduli 70/210\nalone", "matrix", g.STEEL / 3),
    ]
    xs = np.arange(len(groups))
    for j, (_label, law, soft) in enumerate(groups):
        for i, ratios in enumerate(("0", "0.2")):
            r = one(mod, law=law, young_soft=soft, ratios=ratios)
            ax.bar(
                xs[j] + (i - 0.5) * 0.36,
                100 * r["uniform_elongation"],
                width=0.34,
                color=HUES[1] if i else HUES[0],
                label=("straight tracks" if i == 0 else "corrugated, H/P = 0.2")
                if j == 0
                else None,
            )
    ax.axhline(100 * matrix["uniform_elongation"], color=MUTED, lw=1.0, ls=":")
    ax.text(
        len(groups) - 0.5,
        100 * matrix["uniform_elongation"] + 0.5,
        "untreated",
        fontsize=7,
        color=MUTED,
        ha="right",
    )
    ax.set_xticks(xs, [gr[0] for gr in groups], fontsize=7.5)
    ax.set_ylim(0, 36)
    ax.set_ylabel("uniform elongation (%)", color=INK, fontsize=9)
    ax.set_title(
        "d. Which contrast unbends the tracks (24 %, P = 20 mm, ductile)",
        loc="left",
        fontsize=9.5,
        color=INK,
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    style(ax)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"figure: {path}")


#: the necking strains of Fraser et al. (2018, Fig. 3a), their composites of copper and
#: 20 % of steel, against the height (peak to trough, mm) of a corrugation of 8 mm: read
#: from the figure
FRASER_2018 = {0.0: 0.40, 1.0: 0.375, 2.0: 0.39, 3.0: 0.45, 4.0: 0.545}


def checks(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = g.load(Path("results/laser_geometric_hardening"))
    M = g.MARTENSITE
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), layout="constrained")

    # (a) Fraser's copper and steel in the periodic sheet
    ax = axes[0]
    fr = sorted(select(rows, study="fraser"), key=lambda r: float(r["ratios"]))
    h = np.array([2 * 8.0 * float(r["ratios"]) for r in fr])
    ax.plot(
        h / 8.0,
        [100 * r["uniform_elongation"] for r in fr],
        "o-",
        color=HUES[0],
        lw=2.0,
        label="this model (plane stress)",
    )
    ax.plot(
        [k / 8.0 for k in FRASER_2018],
        [100 * v for v in FRASER_2018.values()],
        "s",
        color=INK,
        mfc="white",
        ms=6,
        label="Fraser et al. 2018 (plane strain)",
    )
    ax.axvline(0.35, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.text(0.36, 36.5, "their" + chr(10) + "threshold", fontsize=7, color=MUTED)
    ax.set_xlabel("height (peak to trough) over period h/P", color=INK, fontsize=9)
    ax.set_ylabel("necking strain (%)", color=INK, fontsize=9)
    ax.set_title("a. Copper and 20 % of 4130 steel, P = 8 mm", loc="left", fontsize=9.5, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_ylim(35, 60)
    style(ax)

    # (b) softer crests (martensite flanks, graded rule, 24 %, P = 20 mm)
    ax = axes[1]
    sheet = {k: r for k, r in rows.items() if r["study"] == "sheet" and r["element"] == 0.5}
    ratios = (0.15, 0.2, 0.25, 0.3)
    series = [
        (
            "as-quenched martensite",
            [
                one(sheet, rule="graded", contrast=M, fraction=0.24, period=20.0, ratios=f"{r:g}")
                for r in ratios
            ],
            HUES[0],
            "-",
        ),
        (
            "tempered (contrast 3)",
            [
                one(sheet, rule="graded", contrast=3.0, fraction=0.24, period=20.0, ratios=f"{r:g}")
                for r in ratios
            ],
            HUES[2],
            "-",
        ),
    ]
    cr = {k: r for k, r in rows.items() if r["study"] == "crests"}
    for crest, label, hue in (
        (3.0, "crests tempered", HUES[1]),
        (1.0, "crests untreated", HUES[3]),
    ):
        for extent, ls in ((0.125, "-"), (0.0625, (0, (4, 2)))):
            pts = [
                one(cr, period=20.0, ratios=f"{r:g}", crest=crest, extent=extent) for r in ratios
            ]
            share = "a quarter" if extent > 0.1 else "an eighth"
            series.append((f"{label}, {share} of P each", pts, hue, ls))
    for label, pts, hue, ls in series:
        ax.plot(
            ratios,
            [100 * r["failure_strain"] for r in pts],
            marker="o",
            ms=3.5,
            color=hue,
            ls=ls,
            lw=1.6,
            label=label,
        )
    matrix = one(sheet, fraction=0.0)
    ax.axhline(100 * matrix["failure_strain"], color=MUTED, lw=1.0, ls=":")
    ax.set_xlabel("height over period H/P", color=INK, fontsize=9)
    ax.set_ylabel("failure strain (%)", color=INK, fontsize=9)
    ax.set_title("b. Softer crests: 24 %, P = 20 mm (graded)", loc="left", fontsize=9.5, color=INK)
    ax.text(
        0.302,
        100 * matrix["failure_strain"] - 0.9,
        "untreated",
        fontsize=7,
        color=MUTED,
        ha="right",
    )
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    style(ax)

    # (c) the specimens against the sheet (martensite, 24 %, P = 20 mm)
    ax = axes[2]
    sp = {k: r for k, r in rows.items() if r["study"] == "specimen"}
    cases = [
        ("periodic sheet", None),
        ("dog-bone, 8 tracks", (8.0, 0.0)),
        ("dog-bone, 4 tracks", (4.0, 0.0)),
        ("8 tracks, tempered border", (8.0, 3.0)),
    ]
    xs = np.arange(3)
    for k, (label, spec) in enumerate(cases):
        vals = []
        for r in (0.0, 0.2, 0.3):
            ratios_s = f"{r:g}"
            if spec is None:
                row = one(
                    sheet,
                    rule="ductile",
                    contrast=M,
                    fraction=0.24,
                    ratios=ratios_s,
                    **({"period": 20.0} if r else {}),
                )
            else:
                cands = [
                    x
                    for x in select(sp, rule="ductile", ratios=ratios_s)
                    if (x["count"] if math.isfinite(x["count"]) and x["count"] else 4.0) == spec[0]
                    and (x["border"] if math.isfinite(x["border"]) else 0.0) == spec[1]
                ]
                row = cands[0] if cands else None
            vals.append(100 * row["failure_strain"] if row else np.nan)
        ax.bar(xs + (k - 1.5) * 0.2, vals, width=0.19, color=HUES[k] if k else MUTED, label=label)
    ax.axhline(100 * matrix["failure_strain"], color=MUTED, lw=1.0, ls=":")
    ax.set_xticks(xs, ["straight", "H/P = 0.2", "H/P = 0.3"], fontsize=8)
    ax.set_ylim(0, 36)
    ax.set_ylabel("uniform elongation (%)", color=INK, fontsize=9)
    ax.set_title("c. Dog-bones against the sheet (ductile)", loc="left", fontsize=9.5, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    style(ax)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"figure: {path}")


def stiffness(path: Path) -> None:
    """The third figure: how a contrast of stiffness and one of strength share the
    stresses and the strains between the tracks and the matrix, and the uniform
    elongation against the moduli (``--study stiffness``; the curves hold the strain,
    the stress, then the stress and the strain of the tracks and of the matrix)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    rows = g.load(Path("results/laser_geometric_hardening_stiffness"))
    S = g.STEEL
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), layout="constrained")
    cases = [
        ("stiffness alone, E_r = 3 E_m", "matrix", S, 3 * S, HUES[0]),
        ("strength alone (martensite)", "contrast", S, S, HUES[1]),
        ("strength and E_r = 3 E_m", "contrast", S, 3 * S, HUES[2]),
    ]
    yields = ((280.0 / S, "matrix\nyields"), (1300.0 / S, "martensite\nyields\n(210 GPa)"))
    panels = (
        (axes[0], 2, 4, "a. Mean stress along x, tracks over matrix", "stress ratio"),
        (axes[1], 5, 3, "b. Mean equivalent strain, matrix over tracks", "strain ratio"),
    )
    for ax, top, bottom, title, ylabel in panels:
        for label, law, soft, hard, hue in cases:
            for ratios, ls in (("0.25", "-"), ("0", (0, (4, 2)))):
                r = one(rows, law=law, young_soft=soft, young_hard=hard, ratios=ratios)
                c = np.asarray(r["curve"], dtype=float)[:, 1:]
                ax.plot(100 * c[0], c[top] / c[bottom], color=hue, lw=1.8, ls=ls, label=label)
        for e, text in yields:
            ax.axvline(100 * e, color=MUTED, lw=0.8, ls=(0, (1, 2)))
            if ax is axes[0]:  # below the curves, which stay above a ratio of 1
                ax.text(
                    100 * e * 1.1,
                    0.03,
                    text,
                    transform=ax.get_xaxis_transform(),
                    fontsize=7,
                    color=MUTED,
                    va="bottom",
                )
        ax.set_xscale("log")
        ax.set_xlim(0.01, 40)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("strain of the sheet (%)", color=INK, fontsize=9)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        style(ax)
    handles = [Line2D([], [], color=hue, lw=1.8, label=label) for label, *_, hue in cases]
    handles += [
        Line2D([], [], color=MUTED, lw=1.8, label="corrugated, H/P = 0.25"),
        Line2D([], [], color=MUTED, lw=1.8, ls=(0, (4, 2)), label="straight tracks"),
    ]
    axes[1].legend(handles=handles, fontsize=7, frameon=False, loc="upper left")

    # (c) the uniform elongation against the ratio of the moduli
    ax = axes[2]
    for ratios, ls, label in (
        ("0.25", "-", "corrugated, H/P = 0.25"),
        ("0", (0, (4, 2)), "straight"),
    ):
        pts = sorted(
            select(rows, law="contrast", young_soft=S, ratios=ratios), key=lambda r: r["young_hard"]
        )
        ax.plot(
            [r["young_hard"] / S for r in pts],
            [100 * r["uniform_elongation"] for r in pts],
            color=HUES[1],
            ls=ls,
            lw=1.8,
            marker="o",
            ms=4,
            label=f"strength contrast, {label}",
        )
        soft = one(rows, law="contrast", young_soft=S / 3, ratios=ratios)
        ax.plot(
            3.0,
            100 * soft["uniform_elongation"],
            "o",
            ms=6,
            color=HUES[1],
            mfc="white",
            label="the same, E_m = 70 GPa" if ratios == "0" else None,
        )
    alone = [r for r in select(rows, law="matrix") if r["ratios"] in ("0", "0.25")]
    ax.plot(
        [3.0] * len(alone),
        [100 * r["uniform_elongation"] for r in alone],
        "D",
        ms=5,
        color=HUES[0],
        label="stiffness alone (both ways)",
    )
    main_rows = g.load(Path("results/laser_geometric_hardening"))
    untreated = one(main_rows, study="sheet", fraction=0.0, element=0.5)["uniform_elongation"]
    ax.axhline(100 * untreated, color=MUTED, lw=1.0, ls=":")
    ax.text(0.35, 100 * untreated + 0.6, "untreated", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks([1 / 3, 0.56, 1, 2, 3], ["1/3", "0.56", "1", "2", "3"])
    ax.minorticks_off()
    ax.set_ylim(0, 36)
    ax.set_xlabel("modulus of the tracks over that of the matrix", color=INK, fontsize=9)
    ax.set_ylabel("uniform elongation (%)", color=INK, fontsize=9)
    ax.set_title("c. Uniform elongation against the moduli", loc="left", fontsize=9.5, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    style(ax)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"figure: {path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/_static")
    main(out / "laser_geometric_hardening.png")
    checks(out / "laser_geometric_hardening_checks.png")
    stiffness(out / "laser_geometric_hardening_stiffness.png")
