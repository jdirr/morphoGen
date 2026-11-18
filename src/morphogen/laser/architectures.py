"""Architectures treated pixel by pixel: a morphology on the laser grid of a sheet.

A laser treats a sheet through its thickness by squares about the size of its spot,
the *pixel* (1 mm by default: the smallest feature a spot of 1 mm treats), over sheets
much larger than any specimen (tens of centimetres: hundreds of pixels a side). A
*design* is an integer array ``(ny, nx)`` of phases on that grid, row ``i`` at ``y =
(i + ½) pixel`` and column ``j`` at ``x = (j + ½) pixel``; phase 0 is the untreated sheet
(:class:`~morphogen.laser.cells.Phase`, the phases of :func:`~morphogen.laser.cells.wang_phases`
or :func:`~morphogen.laser.cells.contrast_phases`). Any morphology of morphoGen (a
Boolean model, a tessellation, a random field, a lattice...) becomes a design by
:func:`pixelate`, the majority phase of each pixel.

**Treatment** (:func:`treatment_measures`). Each pixel can be treated, but heat diffuses: an
untreated pixel next to a treated one sits in its heat-affected zone (tempered,
partly transformed), and an untreated pixel enclosed by treated ones may not stay
untreated at all. The measures: the treated share; the treated length drawn by raster
lines one spot wide, its time at the speed of the beam, the switches of the beam along
the lines; the connectivity of each phase (whether it spans the design along x and y);
the untreated pixels next to the treatment and those enclosed by it.

**The sheet** (:func:`sheet_tests`). A design repeated over the sheet is periodic: the
cell of ``nx x ny`` pixels (:class:`~morphogen.laser.periodic.PeriodicCell`, elements
of half a pixel by default, so that the pixels fall on the elements), each Gauss point
with the phase of its pixel (:func:`design_membrane`), pulled in several directions
(:func:`~morphogen.laser.periodic.cell_test`): the yield and tensile strengths, the
failure strain (the first of the maximum of the nominal stress and of the fracture of
a brittle phase, :func:`~morphogen.laser.cells.failure_measures`), the energy absorbed
up to it and Lankford's coefficient of the sheet in each direction, and their spreads
over the directions.

**Tracks** (:func:`track_membrane`). Smooth tracks of constant width along polylines
(corrugations, oblique families), and their periodic images over a cell
(:func:`periodic_lines`): each Gauss point mixes the flow stresses and Young's moduli of
the untreated and treated phases by the share of its region inside the tracks
(:func:`~morphogen.laser.cells.band_fractions`). Phases may differ in stiffness as well
as in strength (``young``: one modulus per phase), which separates the two contrasts.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from morphogen.laser.cells import (
    FRACTURE_RADIUS,
    Phase,
    band_fractions,
    failure_measures,
    wang_phases,
)
from morphogen.laser.mechanics import Membrane
from morphogen.laser.periodic import PeriodicCell, cell_test

__all__ = [
    "design_membrane",
    "periodic_lines",
    "pixelate",
    "sheet_tests",
    "spot_opening",
    "track_membrane",
    "treatment_measures",
]


def _moduli(young, count: int) -> np.ndarray:
    """Young's modulus of each of ``count`` phases: one value for all, or one each."""
    e = np.atleast_1d(np.asarray(young, dtype=float))
    if e.size == 1:
        e = np.full(count, float(e[0]))
    if e.shape != (count,) or not np.all(e > 0):
        raise ValueError(f"young: one positive modulus, or one for each of the {count} phases")
    return e


def pixelate(image, shape) -> np.ndarray:
    """The phases (integer labels) of an ``image`` on a grid of ``shape = (ny, nx)``
    pixels covering the same extent: the most frequent label of the image points whose
    centres fall in each pixel (the lowest label on a tie). An image coarser than the
    grid is sampled at the pixel centres."""
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("pixelate: a 2D image")
    if not np.issubdtype(img.dtype, np.integer):
        if not np.all(img == np.round(img)):
            raise ValueError("pixelate: integer labels")
        img = img.astype(np.int64)
    ny, nx = (int(v) for v in shape)
    if ny < 1 or nx < 1:
        raise ValueError("pixelate: a positive shape")
    if img.min() < 0:
        raise ValueError("pixelate: labels from 0")
    iy = np.minimum((np.arange(img.shape[0]) + 0.5) * ny // img.shape[0], ny - 1).astype(int)
    ix = np.minimum((np.arange(img.shape[1]) + 0.5) * nx // img.shape[1], nx - 1).astype(int)
    if img.shape[0] < ny or img.shape[1] < nx:  # coarser: sample at the pixel centres
        sy = np.minimum(((np.arange(ny) + 0.5) * img.shape[0] / ny).astype(int), img.shape[0] - 1)
        sx = np.minimum(((np.arange(nx) + 0.5) * img.shape[1] / nx).astype(int), img.shape[1] - 1)
        return img[np.ix_(sy, sx)].astype(np.int64)
    labels = int(img.max()) + 1
    pix = (iy[:, None] * nx + ix[None, :]).ravel()
    counts = np.bincount(pix * labels + img.ravel(), minlength=ny * nx * labels)
    return counts.reshape(ny, nx, labels).argmax(axis=2).astype(np.int64)


def spot_opening(design, width: int = 2, *, periodic: bool = True) -> np.ndarray:
    """The part of a design (treated: phase > 0) that a square spot of ``width`` cells
    draws, its morphological opening (``periodic``: through the edges of the cell;
    otherwise the spot may overhang them), the treated cells keeping their phase."""
    d = np.asarray(design)
    treated = d > 0
    if width <= 1 or not treated.any():
        return d.copy()
    st = np.ones((width, width), dtype=bool)
    pad = np.pad(treated, width, mode="wrap" if periodic else "edge")
    kept = ndimage.binary_opening(pad, structure=st)[width:-width, width:-width]
    return np.where(kept, d, 0)


def _spans(mask, axis: int, periodic: bool) -> bool:
    """Whether a cluster of ``mask`` (4-connected) joins the two edges normal to
    ``axis`` (``periodic``: through the cell to its own image)."""
    if not mask.any():
        return False
    if periodic:  # 3 x 3 cells: a cluster that holds a pixel and its image one period on
        lab, _ = ndimage.label(np.tile(mask, (3, 3)))
        ny, nx = mask.shape
        centre = lab[ny : 2 * ny, nx : 2 * nx]
        image = lab[2 * ny :, nx : 2 * nx] if axis == 0 else lab[ny : 2 * ny, 2 * nx :]
        return bool(np.any((centre > 0) & (centre == image)))
    lab, _ = ndimage.label(mask)
    lo = set(np.unique(lab.take(0, axis=axis))) - {0}
    hi = set(np.unique(lab.take(-1, axis=axis))) - {0}
    return bool(lo & hi)


def treatment_measures(
    design, *, pixel: float = 1.0, spot: float = 1.0, speed: float = 10.0, periodic: bool = True
) -> dict:
    """The laser treatment of a design (see the module): ``treated`` share; ``length`` (m) of
    the raster lines one ``spot`` wide that treat it, ``time`` (min) at ``speed`` (mm/s),
    ``switches`` of the beam along the rows (per m² of sheet); whether the treated and
    the untreated phases span the design along x and along y (``hard_spans``,
    ``soft_spans``: pairs of booleans); the shares of the untreated pixels with a
    treated neighbour (``heat_affected``) and with treated pixels on all four sides
    (``enclosed``), among the untreated ones."""
    d = np.asarray(design)
    treated = d > 0
    t = treated.astype(np.int8)
    area = d.size * pixel**2  # mm²
    length = treated.sum() * pixel**2 / spot  # mm
    if periodic:
        switches = int(np.count_nonzero(t != np.roll(t, 1, axis=1)))
        neighbours = sum(np.roll(t, s, axis=a) for a in (0, 1) for s in (1, -1))
    else:
        switches = int(np.count_nonzero(np.diff(np.pad(t, ((0, 0), (1, 1))), axis=1)))
        p = np.pad(t, 1)
        neighbours = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    soft = ~treated
    near = soft & (neighbours > 0)
    enclosed = soft & (neighbours == 4)
    nsoft = max(int(soft.sum()), 1)
    return {
        "treated": float(treated.mean()),
        "length": float(length / 1000.0),
        "time": float(length / speed / 60.0),
        "switches": float(switches / (area * 1e-6)),
        "hard_spans": (_spans(treated, 1, periodic), _spans(treated, 0, periodic)),
        "soft_spans": (_spans(soft, 1, periodic), _spans(soft, 0, periodic)),
        "heat_affected": float(near.sum() / nsoft),
        "enclosed": float(enclosed.sum() / nsoft),
    }


def design_membrane(
    mesh,
    design,
    *,
    pixel: float = 1.0,
    phases=None,
    origin=(0.0, 0.0),
    young=210000.0,
    poisson: float = 0.3,
    thickness: float = 1.0,
    fracture_radius: float | None = FRACTURE_RADIUS,
) -> Membrane:
    """The membrane of a mesh whose Gauss points take the phase of their pixel of the
    design (from ``origin``, repeated beyond it), with the flow law, the critical
    strain and Young's modulus of the phase (``phases``:
    :func:`~morphogen.laser.cells.wang_phases` by default; ``young``: one modulus, or
    one per phase)."""
    phases = tuple(Phase.from_config(p) for p in (phases or wang_phases()))
    d = np.asarray(design, dtype=np.int64)
    if d.min() < 0 or d.max() >= len(phases):
        raise ValueError("design: phases from 0 to len(phases) - 1")
    pts = mesh.gauss()[0]
    ny, nx = d.shape
    j = np.floor((pts[..., 0] - origin[0]) / pixel).astype(np.int64) % nx
    i = np.floor((pts[..., 1] - origin[1]) / pixel).astype(np.int64) % ny
    ph = d[i, j]
    laws = np.array([[p.law.code, p.law.sigma0, p.law.a, p.law.b] for p in phases], dtype=float)
    crit = np.array([p.critical for p in phases], dtype=float)
    flow = [np.ascontiguousarray(laws[ph][..., k][..., None]) for k in range(4)]
    critical = np.ascontiguousarray(crit[ph][..., None])
    brittle = bool(np.isfinite(critical).any())
    return Membrane(
        mesh,
        flow,
        young=_moduli(young, len(phases))[ph][..., None],
        poisson=poisson,
        thickness=thickness,
        critical=critical if brittle else None,
        fracture_radius=fracture_radius if brittle else None,
    )


def sheet_tests(
    design,
    *,
    pixel: float = 1.0,
    phases=None,
    angles=(0.0, 45.0, 90.0),
    element: float | None = None,
    max_strain: float = 0.3,
    fracture_radius: float | None = FRACTURE_RADIUS,
    young: float = 210000.0,
    thickness: float = 1.0,
) -> dict:
    """The sheet of a periodic design pulled in each direction of ``angles`` (degrees
    from x; see the module): ``{"directions": {angle: measures}, ...}`` with, for each
    direction, the yield and tensile strengths (MPa), the failure strain, the energy up
    to it (MJ/m³), Lankford's coefficient, whether a brittle phase broke first, and the
    result of the test; and over the directions the means and spreads (max − min over
    the mean) of the yield strength, the failure strain and the energy."""
    d = np.asarray(design, dtype=np.int64)
    ny, nx = d.shape
    size = (nx * pixel, ny * pixel)
    cell = PeriodicCell.build(
        (size[0], 0.0),
        (0.0, size[1]),
        element=element or 0.5 * pixel,
        centre=(0.5 * size[0], 0.5 * size[1]),
    )
    out = {}
    for angle in angles:
        mem = design_membrane(
            cell.mesh,
            d,
            pixel=pixel,
            phases=phases,
            origin=cell.origin,
            young=young,
            thickness=thickness,
            fracture_radius=fracture_radius,
        )
        res = cell_test(mem, cell, float(angle), max_strain=max_strain, failure_margin=0.03)
        fail = res.failure_strain if math.isfinite(res.failure_strain) else math.inf
        m = failure_measures(res.strain, res.stress, fail, young=young)
        m["r_value"] = res.r_value()
        m["broke"] = bool(fail < m["uniform_elongation"] - 1e-9)
        m["result"] = res
        out[float(angle)] = m

    def spread(key):
        v = np.array([out[a][key] for a in out], dtype=float)
        return float(v.mean()), float((v.max() - v.min()) / v.mean()) if v.mean() else math.nan

    summary = {"directions": out}
    for key in ("yield_strength", "tensile_strength", "failure_strain", "energy", "r_value"):
        summary[f"{key}_mean"], summary[f"{key}_spread"] = spread(key)
    return summary


def periodic_lines(lines, a1, a2, box, reach: float = 0.0) -> list:
    """The polylines ``lines`` (mm) and their images by the lattice of ``a1`` and
    ``a2`` that come within ``reach`` of the box ``(x0, x1, y0, y1)``: the tracks of a
    pattern repeated over the sheet, as they cross a periodic cell."""
    a1, a2 = np.asarray(a1, dtype=float), np.asarray(a2, dtype=float)
    area = abs(a1[0] * a2[1] - a1[1] * a2[0])
    if area <= 0:
        raise ValueError("periodic lines: a1 and a2 parallel")
    h = area / max(np.linalg.norm(a1), np.linalg.norm(a2))  # the smaller height of the cell
    x0, x1, y0, y1 = box
    out = []
    for line in lines:
        pts = np.asarray(line, dtype=float)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre = 0.5 * np.array([x0 + x1, y0 + y1])
        # enough images to reach the box from wherever the line lies
        span = (
            max(x1 - x0, y1 - y0)
            + float(np.max(hi - lo))
            + float(np.linalg.norm(0.5 * (lo + hi) - centre))
            + 2.0 * reach
        )
        n = math.ceil(span / h) + 1
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                q = pts + i * a1 + j * a2
                qlo, qhi = q.min(axis=0), q.max(axis=0)
                if (
                    qhi[0] >= x0 - reach
                    and qlo[0] <= x1 + reach
                    and qhi[1] >= y0 - reach
                    and qlo[1] <= y1 + reach
                ):
                    out.append(q)
    return out


def track_membrane(
    mesh,
    lines,
    width,
    *,
    phases=None,
    kinds=None,
    young=210000.0,
    poisson: float = 0.3,
    thickness: float = 1.0,
    fracture_radius: float | None = FRACTURE_RADIUS,
    subdivisions: int = 4,
) -> Membrane:
    """The membrane of a mesh crossed by tracks along the polylines ``lines`` (their
    centre lines; see the module), ``width`` mm wide (one width, or one per line): the
    share ``w_k`` of the region of each Gauss point inside the tracks of each treated
    phase ``k`` (``kinds``: the phase of each line, 1 by default; ``phases``: the
    untreated first, :func:`~morphogen.laser.cells.wang_phases` by default) mixes the
    flow stress of the untreated phase with that of the treated phase of the largest
    share as ``(1 − w) σ_0 + w σ_k`` (``w`` the sum of the shares), and Young's moduli
    (``young``: one modulus, or one per phase) by the shares; a point mostly inside a
    track takes the critical strain of that phase. Several kinds: a track whose
    treatment changes along it, tracks of several treatments."""
    phases = tuple(Phase.from_config(p) for p in (phases or wang_phases()))
    n = len(phases)
    if n < 2:
        raise ValueError("track membrane: at least two phases, the untreated first")
    lines = [np.asarray(line, dtype=float) for line in lines]
    kinds = [1] * len(lines) if kinds is None else [int(k) for k in kinds]
    widths = np.broadcast_to(np.asarray(width, dtype=float), (len(lines),))
    if len(kinds) != len(lines) or any(not 1 <= k < n for k in kinds):
        raise ValueError(f"kinds: one phase from 1 to {n - 1} for each line")
    if np.any(widths <= 0):
        raise ValueError("track membrane: positive widths")
    ne = len(mesh.quads)
    shares = np.zeros((ne, 4, n))
    for k, w in {(k, float(w)) for k, w in zip(kinds, widths, strict=True)}:
        sel = [ln for ln, kk, ww in zip(lines, kinds, widths, strict=True) if kk == k and ww == w]
        part = band_fractions(mesh, sel, w, n=subdivisions)
        shares[..., k] = np.maximum(shares[..., k], part)
    total = shares[..., 1:].sum(axis=-1)
    over = total > 1.0  # crossing tracks: their shares scaled to the region
    shares[over, 1:] /= total[over, None]
    total = np.minimum(total, 1.0)
    shares[..., 0] = 1.0 - total
    top = 1 + np.argmax(shares[..., 1:], axis=-1)
    laws = np.array([[p.law.code, p.law.sigma0, p.law.a, p.law.b] for p in phases], dtype=float)
    shape = (ne, 4, 1)
    flow = [np.full(shape, v) for v in laws[0]]
    flow += [np.ascontiguousarray(laws[top][..., k][..., None]) for k in range(4)]
    flow.append(np.ascontiguousarray(total[..., None]))
    e = _moduli(young, n)
    moduli = np.ascontiguousarray((shares @ e)[..., None])
    crit = np.array([p.critical for p in phases], dtype=float)
    critical = np.where(total > 0.5, crit[top], crit[0])[..., None]
    brittle = bool(np.isfinite(critical).any())
    return Membrane(
        mesh,
        flow,
        young=moduli,
        poisson=poisson,
        thickness=thickness,
        critical=np.ascontiguousarray(critical) if brittle else None,
        fracture_radius=fracture_radius if brittle else None,
    )
