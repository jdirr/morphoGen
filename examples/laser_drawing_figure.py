"""The figure of docs/laser.md, §17: the drawing side of forming and crossed families.

Reads ``results/laser_drawing_mk``, ``_grooves``, ``_limits``, ``_tension`` and
``results/laser_crossed_specimens.csv``, ``results/laser_crossed_dogbone_map.npz``
(``examples/laser_drawing.py``, ``laser_crossed_specimen.py``,
``laser_crossed_specimens_table.py``) and draws four panels: the forming limits of the
drawing side for single families and crossed families against the untreated sheet and
the classical analysis of inclined grooves; the limit against the angle of the groove in
uniaxial tension; the uniform elongation of crossed families against their half-angle,
in the periodic sheet and in dog-bones; the plastic strain of a dog-bone of crossed
tracks.

Usage: python examples/laser_drawing_figure.py [output.png]
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import laser_drawing as ld
import laser_geometric_hardening_figure as gf

INK, MUTED, HUES, style = gf.INK, gf.MUTED, gf.HUES, gf.style
CLASSICAL = Path("results/laser_drawing_classical.csv")
#: the designs of the panels: (label, hue, design, H/P, angle, half-angle)
DESIGNS = (
    ("straight tracks at 60°", HUES[5], "family", 0.0, 60.0, 0.0),
    ("corrugated, H/P = 0.25", HUES[0], "family", 0.25, 0.0, 0.0),
    ("crossed at ±55°", HUES[2], "crossed", 0.0, 0.0, 55.0),
    ("crossed at ±60°", HUES[1], "crossed", 0.0, 0.0, 60.0),
    ("crossed at ±65°", HUES[3], "crossed", 0.0, 0.0, 65.0),
)


def classical(paths) -> np.ndarray:
    """The least classical limits over the angle of the groove, ``(ρ, limit, angle)``,
    cached in ``results/laser_drawing_classical.csv``."""
    have = {}
    if CLASSICAL.exists():
        for line in CLASSICAL.read_text(encoding="utf-8").splitlines()[1:]:
            r, v, p = (float(x) for x in line.split(","))
            have[round(r, 6)] = (v, p)
    for r in paths:
        if round(r, 6) not in have:
            have[round(r, 6)] = ld.mk_critical(r)
    CLASSICAL.write_text(
        "path,limit,angle\n" + "".join(f"{r},{v},{p}\n" for r, (v, p) in sorted(have.items())),
        encoding="utf-8",
    )
    return np.array([(r, *have[round(r, 6)]) for r in paths])


def scan(rows, design, ratio, theta, spread, path, shear="fixed"):
    """The grooves of a design along a path: (angle, major, minor), by angle."""
    out = {}
    for r in rows.values():
        if (
            r["design"] != design
            or abs(r["ratio"] - ratio) > 1e-9
            or abs(r["theta"] - theta) > 1e-9
            or abs(ld._spread(r) - spread) > 1e-9
            or abs(r["path"] - path) > 1e-9
            or r["shear"] != shear
            or r["study"] in ("tension", "uniaxial")
            or (design == "none" and r["length"] != 60.0)
        ):
            continue
        out.setdefault(round(r["psi"], 2), (r["psi"], r["major"], r["minor"]))
    return [out[k] for k in sorted(out)]


def least(rows, *args, **kw):
    fin = [s for s in scan(rows, *args, **kw) if math.isfinite(s[1])]
    return min(fin, key=lambda s: s[1]) if fin else None


def main(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = [n for n in ("Inter", "Segoe UI", "DejaVu Sans") if n in names]
    plt.rcParams["font.family"] = sans[:1] or ["sans-serif"]
    rows = {}
    for study in ("mk", "grooves", "limits"):
        rows.update(ld.load(Path(f"results/laser_drawing_{study}")))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.2), layout="constrained")

    # (a) the forming limits of the drawing side
    ax = axes[0, 0]
    mk = classical(np.linspace(-0.5, 0.0, 9))
    ax.plot(mk[:, 0] * mk[:, 1], mk[:, 1], color=INK, lw=1.0, ls=":", label="untreated, classical")
    pts = [least(rows, "none", 0.0, 0.0, 0.0, p) for p in ld.PATHS]
    pts = [(b[2], b[1]) for b in pts if b]
    ax.plot(
        *zip(*pts, strict=True),
        ls="none",
        marker="s",
        ms=4.5,
        mfc="white",
        mec=INK,
        label="untreated, groove in the cell",
    )
    for label, hue, design, ratio, theta, spread in DESIGNS:
        pts = [least(rows, design, ratio, theta, spread, p) for p in ld.PATHS]
        pts = sorted((b[2], b[1]) for b in pts if b)
        if pts:
            ax.plot(*zip(*pts, strict=True), color=hue, lw=1.5, marker="o", ms=3.5, label=label)
    ax.set_xlabel("minor true strain at the limit", color=INK, fontsize=9)
    ax.set_ylabel("major true strain at the limit", color=INK, fontsize=9)
    ax.set_xlim(-0.3, 0.02)
    ax.set_ylim(0.0, 0.7)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("(a) forming limits, drawing side (24 %)", fontsize=9, color=INK, loc="left")
    style(ax)

    # (b) the limit against the angle of the groove in uniaxial tension
    ax = axes[0, 1]
    for label, hue, design, ratio, theta, spread in (
        ("untreated", INK, "none", 0.0, 0.0, 0.0),
        *DESIGNS[:2],
        DESIGNS[3],
    ):
        s = scan(rows, design, ratio, theta, spread, -0.5)
        if not s:
            continue
        x = [v[0] for v in s]
        y = [v[1] if math.isfinite(v[1]) else 0.9 for v in s]
        ax.plot(x, y, color=hue, lw=1.4, marker="o", ms=3.5, label=label)
    ax.axhline(0.9, color=MUTED, lw=0.7, ls=":")
    ax.text(-40.0, 0.865, "no localization", fontsize=7, color=MUTED)
    ax.set_xlabel(
        "initial angle of the groove from the normal of the major strain (°)", color=INK, fontsize=9
    )
    ax.set_ylabel("major true strain at the limit", color=INK, fontsize=9)
    ax.set_xlim(-42, 42)
    ax.set_ylim(0.0, 0.95)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("(b) uniaxial tension: the angle of the neck", fontsize=9, color=INK, loc="left")
    style(ax)

    # (c) crossed families: the periodic sheet and the dog-bones
    ax = axes[1, 0]
    tension = ld.load(Path("results/laser_drawing_tension"))

    def fraction(r):
        v = r.get("fraction", math.nan)
        return 0.24 if not math.isfinite(v) else v

    for frac, hue in ((0.12, HUES[2]), (0.24, HUES[1]), (0.36, HUES[3])):
        pts = sorted(
            (r["spread"], 100 * r["uniform_elongation"])
            for r in tension.values()
            if r["design"] == "crossed"
            and r["state"] == "ductile"
            and r["element"] == 0.5
            and abs(fraction(r) - frac) < 1e-9
        )
        if pts:
            ax.plot(
                *zip(*pts, strict=True),
                color=hue,
                lw=1.4,
                marker="o",
                ms=3.5,
                label=f"periodic sheet, {100 * frac:.0f} %",
            )
    table = Path("results/laser_crossed_specimens.csv")
    if table.exists():
        with table.open(encoding="utf-8") as fh:
            spec = list(csv.DictReader(fh))
        pts = sorted(
            (float(r["spread"]), float(r["ag"]))
            for r in spec
            if r["state"] == "ductile"
            and r["length"] == "90"
            and r["edge"] == "free"
            and r.get("width", "44") == "44"
        )
        if pts:
            ax.plot(
                *zip(*pts, strict=True),
                color=HUES[0],
                lw=1.4,
                marker="s",
                ms=4,
                label="dog-bone 44 × 90 mm, 24 %",
            )
        wide = sorted(
            (float(r["spread"]), float(r["ag"]))
            for r in spec
            if r["state"] == "ductile" and r.get("width") == "88"
        )
        if wide:
            ax.plot(
                *zip(*wide, strict=True),
                ls="none",
                marker="s",
                ms=4.5,
                mfc="white",
                mec=HUES[0],
                label="dog-bone 88 × 90 mm",
            )
    u = next(r for r in tension.values() if r["design"] == "none")
    ax.axhline(100 * u["uniform_elongation"], color=MUTED, lw=1.0, ls=":")
    ax.text(44.5, 100 * u["uniform_elongation"] + 0.6, "untreated", fontsize=7, color=MUTED)
    lam = math.exp(0.24)
    th = math.degrees(math.acos(1.0 / math.sqrt(1.0 + lam + lam * lam)))
    ax.axvline(th, color=INK, lw=0.8, ls=(0, (4, 2)))
    ax.text(th + 0.4, 45.5, "θ* (n = 0.24)", fontsize=7, color=INK)
    ax.set_xlabel("half-angle of the families about the load (°)", color=INK, fontsize=9)
    ax.set_ylabel("uniform elongation (%)", color=INK, fontsize=9)
    ax.set_xlim(44, 71)
    ax.set_ylim(10, 48)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title(
        "(c) crossed families of straight tracks, ductile", fontsize=9, color=INK, loc="left"
    )
    style(ax)

    # (d) where a dog-bone of crossed tracks necks
    ax = axes[1, 1]
    path = Path("results/laser_crossed_dogbone_map.npz")
    if path.exists():
        d = np.load(path)
        pts = d["points"]
        pl = d["f_plastic_strain"].mean(axis=1)
        keep = (pts[:, 0] > -25) & (pts[:, 0] < 115)
        sc = ax.scatter(
            pts[keep, 0],
            pts[keep, 1],
            c=pl[keep],
            s=0.3,
            cmap="viridis",
            vmin=0.0,
            vmax=0.4,
            rasterized=True,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
        cb.set_label("equivalent plastic strain", fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "(d) dog-bone, crossed at ±60°: necking at the free edges",
        fontsize=9,
        color=INK,
        loc="left",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"written: {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/_static/laser_drawing.png"))
