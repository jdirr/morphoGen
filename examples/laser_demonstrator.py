"""A demonstrator sheet of 300 × 300 mm treated at the scale of the spot (docs/laser.md,
§14.6): four zones of 144 mm, the designs of §14 that the results single out, and the
specimens to cut from them.

* straight tracks of tempered martensite (directional strength and ductility: dog-bones
  at 0°, 45° and 90°);
* a hexagonal honeycomb of tempered tracks (isotropic strengthening);
* Wang tiles of pipes (a network drawn by a small set of tiles, not periodic);
* a field of tempered martensite with soft corridors 2 mm wide from the notches of
  compact tension specimens (W = 30 mm) at 0°, 30° and 60°, and an untreated blank.

Phases: 0 untreated, 1 tempered martensite (2, the as-quenched martensite, left for
the variants). Prints the measures of each zone (the beam time at 10 mm/s with a spot
of 1 mm) and draws the sheet.

Usage: python examples/laser_demonstrator.py [figure]
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import laser_morphologies as lm  # the designs of §14

CELL = 0.5  # mm
SHEET = 300.0  # mm
ZONE = 144.0  # mm
GAP = 4.0  # mm
N = round(SHEET / CELL)


def tile(design: np.ndarray, size: float) -> np.ndarray:
    """A periodic design repeated over a square of ``size`` mm."""
    n = round(size / CELL)
    reps = math.ceil(n / design.shape[0])
    return np.tile(design, (reps, reps))[:n, :n]


def wang_tiles(size: float, seed: int = 4) -> np.ndarray:
    from morphogen.domain import Domain
    from morphogen.laser.architectures import spot_opening
    from morphogen.models import model_from_config

    n = round(size / CELL)
    model = model_from_config(
        {
            "type": "tile_assembly",
            "tiles": "pipes",
            "tile_length": 6.0,
            "tile_cells": 12,
            "dimension": 2,
            "pipe_radius": 0.12,
            "open_probability": 0.6,
        }
    )
    micro = model.generate(Domain(size=(size, size, 1.0), periodic=True), seed=seed)
    img = np.asarray(micro.voxelize(shape=(n, n, 1)).data[:, :, 0]).T > 0
    return spot_opening(img.astype(np.int64), 2)


def crack_zone(size: float):
    """The tempered field, the CT blanks and their corridors: the design and the
    outlines of the blanks ``(x0, y0, width, height, label)`` (mm, from the zone)."""
    n = round(size / CELL)
    d = np.ones((n, n), np.int64)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) * CELL
    W = 30.0
    blanks = []
    for k, (cx, cy, angle) in enumerate(
        ((40, 108, 0.0), (104, 108, 30.0), (40, 36, 60.0), (104, 36, None))
    ):
        x0, y0 = cx - 0.5 * 1.25 * W, cy - 0.6 * W
        tip = (x0 + 0.25 * W + 0.5 * W, cy)  # the load line at x0 + W/4, a/W = 0.5
        inside = (x >= x0) & (x <= x0 + 1.25 * W) & (np.abs(y - cy) <= 0.6 * W)
        if angle is None:  # an untreated blank for reference
            d[inside] = 0
            label = "untreated"
        else:
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            dx, dy = x - tip[0], y - tip[1]
            along, across = dx * c + dy * s, np.abs(-dx * s + dy * c)
            corridor = inside & (across < 1.0) & (along > -2.0)
            d[corridor] = 0
            label = f"corridor at {angle:.0f}°"
        blanks.append((x0, y0, 1.25 * W, 1.2 * W, label))
        del k
    return d, blanks


def sheet():
    from morphogen.laser.architectures import treatment_measures

    cat = lm.catalogue()
    zones = {
        "straight tracks, tempered": tile(cat["straight tracks"][1], ZONE),
        "hexagonal honeycomb, tempered": tile(cat["hexagonal honeycomb"][1], ZONE),
        "Wang tiles (pipes), tempered": wang_tiles(ZONE),
    }
    crack, blanks = crack_zone(ZONE)
    zones["crack steering: tempered field, soft corridors"] = crack
    d = np.zeros((N, N), np.int64)
    origins = {}
    for k, (name, z) in enumerate(zones.items()):
        i, j = divmod(k, 2)
        x0 = GAP + j * (ZONE + GAP)
        y0 = GAP + (1 - i) * (ZONE + GAP)
        r0, c0 = round(y0 / CELL), round(x0 / CELL)
        d[r0 : r0 + z.shape[0], c0 : c0 + z.shape[1]] = z
        origins[name] = (x0, y0)
        p = treatment_measures(z, pixel=CELL, spot=2 * CELL, periodic=False)
        print(
            f"{name:50s} treated {p['treated']:.2f}, beam {p['time'] / 60:.1f} h, "
            f"heat-affected {p['heat_affected']:.2f}"
        )
    p = treatment_measures(d, pixel=CELL, spot=2 * CELL, periodic=False)
    print(f"{'the sheet':50s} treated {p['treated']:.2f}, beam {p['time'] / 60:.1f} h")
    return d, origins, blanks


def figure(d, origins, blanks, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = [n for n in ("Inter", "Segoe UI", "DejaVu Sans") if n in names]
    plt.rcParams["font.family"] = sans[:1] or ["sans-serif"]
    fig, ax = plt.subplots(figsize=(8.0, 8.4), layout="constrained")
    ax.imshow(
        d,
        cmap=ListedColormap(["#f0efec", "#6b6a66", "#2a2a28"]),
        origin="lower",
        extent=(0, SHEET, 0, SHEET),
        interpolation="nearest",
        vmin=0,
        vmax=2,
    )
    ink = "#e34948"
    for name, (x0, y0) in origins.items():
        ax.text(
            x0 + 2,
            y0 + ZONE - 2,
            name,
            va="top",
            fontsize=8,
            color="#0b0b0b",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
    # dog-bones (gauge 30 x 5 mm, overall 80 x 20 mm) at 0, 45 and 90 degrees
    for name in (
        "straight tracks, tempered",
        "hexagonal honeycomb, tempered",
        "Wang tiles (pipes), tempered",
    ):
        x0, y0 = origins[name]
        # their centres, apart: 0° low, 45° in the middle, 90° on the right
        for angle, (dx, dy) in ((0.0, (45, 18)), (45.0, (55, 85)), (90.0, (125, 50))):
            cx, cy = x0 + dx, y0 + dy
            t = matplotlib.transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
            ax.add_patch(
                Rectangle((cx - 40, cy - 10), 80, 20, fill=False, ec=ink, lw=1.0, transform=t)
            )
            ax.add_patch(
                Rectangle((cx - 15, cy - 2.5), 30, 5, fill=False, ec=ink, lw=0.8, transform=t)
            )
    x0, y0 = origins["crack steering: tempered field, soft corridors"]
    for bx, by, w, h, label in blanks:
        ax.add_patch(Rectangle((x0 + bx, y0 + by), w, h, fill=False, ec=ink, lw=1.0))
        ax.text(x0 + bx + 1, y0 + by + 1, label, fontsize=7, color=ink)
    ax.set_xlabel("x, rolling direction (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(
        "A demonstrator sheet, 300 × 300 × 1 mm (light: untreated; grey: tempered martensite)",
        loc="left",
        fontsize=9.5,
    )
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    d, origins, blanks = sheet()
    if len(sys.argv) > 1:
        figure(d, origins, blanks, Path(sys.argv[1]))
        print(f"figure: {sys.argv[1]}")


if __name__ == "__main__":
    main()
