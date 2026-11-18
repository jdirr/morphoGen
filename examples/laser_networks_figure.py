"""The figure of docs/laser.md, §16: geometric work hardening over the plane of the
sheet and along the paths of forming.

Reads ``results/laser_networks_directions``, ``results/laser_networks_forming`` and
``results/laser_networks_mk`` (``examples/laser_networks.py``) and draws four panels: the
uniform elongation of one family of corrugated tracks and of the networks against the
direction of the load; their energy to fracture with the tempered reinforcement; their
forming limits from plane strain to equibiaxial stretching against the untreated sheet
and the classical Marciniak–Kuczyński analysis; the stretch of a material line of the
matrix against its initial angle along the paths, which sets where a corrugation unbends
before it stretches.

Usage: python examples/laser_networks_figure.py [output.png]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
import laser_geometric_hardening_figure as gf
import laser_networks as nw

INK, MUTED, HUES, style = gf.INK, gf.MUTED, gf.HUES, gf.style
#: the designs of the panels: (label, design, H/P, hue)
DESIGNS = (
    ("straight tracks", "family", 0.0, gf.MUTED),
    ("corrugated tracks", "family", 0.25, HUES[0]),
    ("square network", "square", 0.25, HUES[1]),
    ("triangular network", "triangular", 0.25, HUES[2]),
)


def mk_curve(paths, f=nw.IMPERFECTION, n=0.24, sigma0=280.0, k=986.0, d=2e-4, crit=10.0):
    """The classical Marciniak–Kuczyński limits (rigid plastic, von Mises, Swift's law
    ``K (ε0 + p)^n``, a groove normal to the major strain whose flow stress is ``f`` times
    the sheet's) along linear paths of the true strains: ``(major, minor)`` for each."""
    e0 = (sigma0 / k) ** (1.0 / n)
    out = []
    for rho in paths:
        al = (2 * rho + 1) / (rho + 2)
        ga = math.sqrt(1 - al + al * al)
        e1a = eba = e1b = e2b = ebb = 0.0
        res = (math.nan, math.nan)
        while e1a < 1.5:
            de1a, de2 = d, rho * d
            e1a += de1a
            eba += (2 / math.sqrt(3)) * math.sqrt(de1a**2 + de1a * de2 + de2**2)
            rhs = (e0 + eba) ** n / ga * math.exp(-(1 + rho) * e1a)

            def g(x, e1b=e1b, e2b=e2b, ebb=ebb, rhs=rhs, de2=de2):
                rb = de2 / x
                ab = (2 * rb + 1) / (rb + 2)
                deb = (2 / math.sqrt(3)) * math.sqrt(x * x + x * de2 + de2 * de2)
                return (
                    f
                    * (e0 + ebb + deb) ** n
                    / math.sqrt(1 - ab + ab * ab)
                    * math.exp(-(e1b + x + e2b + de2))
                    - rhs
                )

            hi = de1a
            while g(hi) < 0 and hi < 1e3 * de1a:
                hi *= 1.5
            if g(hi) < 0:  # no equilibrium left in the groove
                res = (e1a, rho * e1a)
                break
            lo = hi
            while g(lo) > 0 and lo > 1e-14:
                lo /= 1.5
            x = brentq(g, lo, hi, xtol=1e-15)
            e1b += x
            e2b += de2
            ebb += (2 / math.sqrt(3)) * math.sqrt(x * x + x * de2 + de2 * de2)
            if x / de1a >= crit:
                res = (e1a, rho * e1a)
                break
        out.append(res)
    return out


def line_stretch(phi, e1, rho):
    """The stretch of a material line at ``phi`` (rad) from the major strain under the
    homogeneous strains ``e1`` and ``rho e1``."""
    return np.sqrt(np.exp(2 * e1) * np.cos(phi) ** 2 + np.exp(2 * rho * e1) * np.sin(phi) ** 2)


def main(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = [n for n in ("Inter", "Segoe UI", "DejaVu Sans") if n in names]
    plt.rcParams["font.family"] = sans[:1] or ["sans-serif"]
    dirs = nw.load(Path("results/laser_networks_directions"))
    forming = nw.load(Path("results/laser_networks_forming"))
    untreated = gf.one(dirs, design="none")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.2), layout="constrained")

    # (a, b) the directions: uniform elongation (ductile) and energy (tempered)
    for ax, state, key, scale, ylabel in (
        (axes[0, 0], "ductile", "uniform_elongation", 100.0, "uniform elongation (%)"),
        (axes[0, 1], "tempered", "energy", 1.0, "energy to fracture (MJ/m³)"),
    ):
        ax.axhline(untreated[key] * scale, color=INK, lw=1.0, ls=":", label="untreated sheet")
        for label, design, r, hue in DESIGNS:
            rows = sorted(
                gf.select(dirs, design=design, ratio=r, state=state), key=lambda x: x["angle"]
            )
            a = np.array([x["angle"] for x in rows])
            v = np.array([x[key] for x in rows]) * scale
            if design == "triangular":  # chiral: its directions repeat every 60°
                a, v = np.append(a, [60.0, 75.0, 90.0]), np.append(v, v[[0, 1, 2]])
            ax.plot(a, v, color=hue, lw=1.8, marker="o", ms=3.5, label=label)
        ax.set_xlabel("direction of the load from the tracks along x (°)", color=INK, fontsize=9)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
        style(ax)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper right")
    axes[0, 0].set_title(
        "(a) ductile reinforcement, 24 % treated", fontsize=9, color=INK, loc="left"
    )
    axes[0, 1].set_title(
        "(b) tempered reinforcement (breaks at 0.24)", fontsize=9, color=INK, loc="left"
    )

    # (c) the forming limits, ductile
    ax = axes[1, 0]
    paths = np.linspace(0.0, 1.0, 21)
    mk = np.array(mk_curve(paths))
    ax.plot(mk[:, 1], mk[:, 0], color=INK, lw=1.0, ls=":", label="untreated, classical M-K")
    rows = sorted(gf.select(forming, design="none"), key=lambda x: x["path"])
    ax.plot(
        [x["minor"] for x in rows],
        [x["major"] for x in rows],
        ls="none",
        marker="s",
        ms=4.5,
        mfc="white",
        mec=INK,
        label="untreated, groove in the cell",
    )
    for label, design, r, hue in DESIGNS:
        for angle, ls, mk_ in ((0.0, "-", "o"), (90.0, (0, (4, 2)), "^")):
            rows = sorted(
                gf.select(forming, design=design, ratio=r, state="ductile", angle=angle),
                key=lambda x: x["path"],
            )
            if not rows:
                continue
            x = [q["minor"] for q in rows]
            y = [q["major"] for q in rows]
            tag = label + (" (major strain across)" if angle == 90.0 else "")
            ax.plot(x, y, color=hue, lw=1.5, ls=ls, marker=mk_, ms=3.5, label=tag)
    ax.set_xlabel("minor true strain at the limit", color=INK, fontsize=9)
    ax.set_ylabel("major true strain at the limit", color=INK, fontsize=9)
    ax.set_xlim(-0.02, 0.7)
    ax.set_ylim(0.0, 0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("(c) forming limits, ductile reinforcement", fontsize=9, color=INK, loc="left")
    style(ax)

    # (d) the kinematics: the stretch of a material line of the matrix
    ax = axes[1, 1]
    phi = np.radians(np.linspace(0.0, 90.0, 361))
    for k, (rho, name) in enumerate(
        (
            (-1.0, "−1 (Fraser's plane-strain sections)"),
            (-0.5, "−0.5 (uniaxial tension)"),
            (0.0, "0 (plane strain)"),
            (0.5, "0.5"),
            (1.0, "1 (equibiaxial)"),
        )
    ):
        ax.plot(
            np.degrees(phi),
            line_stretch(phi, 0.15, rho) - 1.0,
            color=HUES[k],
            lw=1.6,
            label=f"ρ = {name}",
        )
    ax.axhline(0.0, color=MUTED, lw=0.8)
    for hp in (0.175, 0.25, 0.3):
        f = math.degrees(math.atan(2 * math.pi * hp))
        ax.axvline(f, color=MUTED, lw=0.8, ls=":")
        ax.text(f + 0.6, 0.155, f"H/P {hp:g}", rotation=90, fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("initial angle of the line from the major strain (°)", color=INK, fontsize=9)
    ax.set_ylabel("stretch − 1 at a major strain of 0.15", color=INK, fontsize=9)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("(d) where a flank is first shortened", fontsize=9, color=INK, loc="left")
    style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"written: {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/_static/laser_networks.png"))
