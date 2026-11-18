"""The drawing side of forming: inclined necks in laser-architectured sheets
(docs/laser.md, §17).

On the stretching side of forming every architecture lowers the forming limit
(§16.2), and the rule of §16.3 says why: no line of the sheet is shortened there. On the
drawing side, ``ρ < 0``, the lines beyond ``φ0 = atan (−1/ρ)^½`` from the major strain are
first shortened: the rule predicts that tracks laid there harden the sheet late. The
necks of that side are inclined (Hill 1952), and the Marciniak–Kuczyński analysis takes
the least limit over the angle of the groove (Hutchinson & Neale 1978).

**The cell of an inclined groove** (:func:`groove_geometry`): the groove runs along a
vector ``g`` of the lattice of the pattern (any direction for straight tracks, which are
invariant along themselves; the vectors ``i P x + j s y`` for corrugated ones, the angle
the nearest to the one asked), the cell is the parallelogram that ``g`` spans with a
vector of the lattice nearly normal to it, 60 mm across at least, meshed in rows along
``g``; the groove is two rows, about 1 mm wide, 0.5 % weaker. The path of the strains
is prescribed in full (``free_shear=False``: the principal axes of the mean stretch held,
as by tools and in the classical analysis) or with the shear free; the limit is the major
strain outside the band of two rows that thins ten times faster than its images in the
other periods, the stall of the path or the fracture of the reinforcement, as in §16.2.

Studies (``--study``):

``mk``
    the untreated sheet, grooves from 0 to 30°, against the classical analysis
    (:func:`mk_limit`: rigid plastic, von Mises, the Swift law of the matrix, the same
    imperfection).
``grooves``
    the designs at 24 %: straight tracks at 0 to 90° from the major strain and
    corrugated tracks along it, along two paths (``ρ`` = −½, uniaxial tension, and −¼),
    grooves from −40 to 40° and along the tracks.
``limits``
    the forming limits of the designs from uniaxial tension to plane strain, each over
    the grooves near its critical ones; with the shear free for the straight tracks.

Writes ``results/laser_drawing_<study>.csv`` and ``.npz``.
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
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
import laser_geometric_hardening as g
import laser_networks as nw

STATES = nw.STATES
GROOVE = nw.GROOVE  # mm: the width of the groove
IMPERFECTION = nw.IMPERFECTION
LOCALIZED = nw.LOCALIZED
MIN_LENGTH = nw.MIN_LENGTH  # mm: the cell across the groove
LONGEST = 80.0  # mm: the longest vector of a corrugated lattice along a groove
PATHS = (-0.5, -0.375, -0.25, -0.125, 0.0)
#: the critical angles of the classical analysis (degrees, ``mk_critical``)
CRITICAL = {-0.5: 21.25, -0.375: 20.75, -0.25: 18.25, -0.125: 13.25, 0.0: 0.0}
#: the designs: (name, design, H/P, angle of the tracks from the major strain, the
#: half-angle between two crossed families)
DESIGNS = (
    ("untreated", "none", 0.0, 0.0, 0.0),
    ("straight 0°", "family", 0.0, 0.0, 0.0),
    ("straight 30°", "family", 0.0, 30.0, 0.0),
    ("straight 45°", "family", 0.0, 45.0, 0.0),
    ("straight 60°", "family", 0.0, 60.0, 0.0),
    ("straight 75°", "family", 0.0, 75.0, 0.0),
    ("straight 90°", "family", 0.0, 90.0, 0.0),
    ("corrugated 0.2", "family", 0.2, 0.0, 0.0),
    ("corrugated 0.25", "family", 0.25, 0.0, 0.0),
    ("corrugated 0.3", "family", 0.3, 0.0, 0.0),
    ("corrugated 0.35", "family", 0.35, 0.0, 0.0),
    ("crossed ±45°", "crossed", 0.0, 0.0, 45.0),
    ("crossed ±55°", "crossed", 0.0, 0.0, 55.0),
    ("crossed ±60°", "crossed", 0.0, 0.0, 60.0),
    ("crossed ±65°", "crossed", 0.0, 0.0, 65.0),
    ("crossed ±70°", "crossed", 0.0, 0.0, 70.0),
)
#: the designs whose tensile properties are computed along the major strain
TENSION = (
    "straight 0°",
    "straight 45°",
    "straight 60°",
    "corrugated 0.25",
    "crossed ±45°",
    "crossed ±55°",
    "crossed ±60°",
    "crossed ±65°",
    "crossed ±70°",
)
#: the designs whose forming limits are computed from uniaxial tension to plane strain
LIMITS = ("straight 60°", "corrugated 0.25", "crossed ±55°", "crossed ±60°", "crossed ±65°")
FIELDS = (
    "key",
    "study",
    "name",
    "design",
    "state",
    "ratio",
    "theta",
    "spread",
    "psi_target",
    "psi",
    "path",
    "shear",
    "groove",
    "length",
    "element",
    "elements",
    "groove_width",
    "treated",
    "major",
    "minor",
    "limit",
    "fraction",
    "yield_strength",
    "tensile_strength",
    "uniform_elongation",
    "failure_strain",
    "energy",
    "seconds",
)


# -- the classical analysis -----------------------------------------------------------


def mk_limit(
    rho: float,
    psi0: float,
    *,
    f: float = IMPERFECTION,
    n: float = 0.24,
    sigma0: float = 280.0,
    k: float = 986.0,
    d: float = 2e-4,
    crit: float = LOCALIZED,
    emax: float = 1.5,
) -> float:
    """The Marciniak–Kuczyński limit (the major true strain of the sheet) of a groove at
    ``psi0`` degrees (its normal from the major strain, at the start) along the linear
    path ``ε2 = ρ ε1`` (Hutchinson & Neale 1978): rigid plastic, von Mises, plane
    stress, Swift's law ``K (ε0 + p)^n`` of the matrix, the flow stress of the groove
    ``f`` times the sheet's; the groove keeps the strain of the sheet along itself, the
    forces across it balance (normal and shear), its normal turns with the sheet, ``tan
    ψ = e^((1 − ρ) ε1) tan ψ0``; the limit when the groove thins ``crit`` times faster
    than the sheet (nan: the groove turns beyond the line of zero extension first)."""
    e0 = (sigma0 / k) ** (1.0 / n)
    t0 = math.tan(math.radians(psi0))
    q = math.sqrt(1.0 + rho + rho * rho)
    s1, s2 = (2.0 + rho) / (math.sqrt(3.0) * q), (1.0 + 2.0 * rho) / (math.sqrt(3.0) * q)
    dea = 2.0 / math.sqrt(3.0) * q * d  # the equivalent strain of the sheet per step
    e1a = eba = e3a = ebb = e3b = 0.0
    while e1a < emax:
        psi = math.atan(math.exp((1.0 - rho) * (e1a + 0.5 * d)) * t0)
        c, s = math.cos(psi), math.sin(psi)
        dtt = d * (s * s + rho * c * c)  # along the groove: the sheet's
        eba_n, e3a_n = eba + dea, e3a - (1.0 + rho) * d
        sa = k * (e0 + eba_n) ** n
        snn, snt = sa * (s1 * c * c + s2 * s * s), sa * (s2 - s1) * s * c
        force = snn * math.exp(e3a_n)  # across the groove, per unit of initial thickness
        beta = snt / snn  # the shear of the groove follows from the balance of shear

        def balance(x, dtt=dtt, beta=beta, force=force, ebb=ebb, e3b=e3b):
            y = beta * (2.0 * x + dtt)
            deb = 2.0 / math.sqrt(3.0) * math.sqrt(x * x + dtt * dtt + x * dtt + y * y)
            sb = f * k * (e0 + ebb + deb) ** n
            return sb * (2.0 / 3.0) * (2.0 * x + dtt) / deb * math.exp(e3b - x - dtt) - force

        # from where the groove carries nothing across, the first balance
        lo = -0.5 * dtt
        prev, hi = lo, None
        for j in range(400):
            x = lo + d * 1e-4 * 1.08**j
            if balance(x) > 0.0:
                hi = x
                break
            if x > lo + 1e3 * d:
                break
            prev = x
        if hi is None:  # no balance left: the groove has localized
            return e1a
        x = brentq(balance, prev if prev > lo else lo + 1e-18, hi, xtol=1e-16)
        y = beta * (2.0 * x + dtt)
        ebb += 2.0 / math.sqrt(3.0) * math.sqrt(x * x + dtt * dtt + x * dtt + y * y)
        e3b -= x + dtt
        e1a += d
        eba, e3a = eba_n, e3a_n
        if (x + dtt) / ((1.0 + rho) * d) >= crit:
            return e1a
    return math.nan


def mk_critical(rho: float, **kw) -> tuple[float, float]:
    """The least classical limit over the angle of the groove and that angle (degrees):
    a scan by 2.5°, refined by 0.25° about its least value."""
    scan = [(mk_limit(rho, p, **kw), p) for p in np.arange(0.0, 45.1, 2.5)]
    scan = [(v, p) for v, p in scan if math.isfinite(v)]
    _, p0 = min(scan)
    fine = [(mk_limit(rho, p, **kw), p) for p in np.arange(max(p0 - 2.5, 0.0), p0 + 2.6, 0.25)]
    return min((v, p) for v, p in fine if math.isfinite(v))


# -- the cell of an inclined groove ---------------------------------------------------


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    """``(r, x, y)`` with ``a x + b y = r``, ``r`` the greatest common divisor (≥ 0)."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if a < 0:
        a, x0, y0 = -a, -x0, -y0
    return a, x0, y0


def crossed_spacing(fraction: float = nw.FRACTION, width: float = nw.TRACK) -> float:
    """The spacing of each of two crossed families of straight tracks that treat
    ``fraction`` together, the crossings counted once: ``2 w/s − w²/s² = f``."""
    return width * (1.0 + math.sqrt(1.0 - fraction)) / fraction


def crossed_lattice(spread: float, s: float) -> tuple[np.ndarray, np.ndarray]:
    """The lattice of two families of straight tracks at ``±spread`` degrees from x, each
    at the spacing ``s``: the translations that carry each family onto itself."""
    th = math.radians(spread)
    a, b = s / (2.0 * math.sin(th)), s / (2.0 * math.cos(th))
    return np.array([a, b]), np.array([-a, b])


def crossed_lines(spread: float, s: float, box) -> list:
    """The centre lines of two families of straight tracks at ``±spread`` degrees from x
    that cross the box ``(x0, x1, y0, y1)``, the lines of each at ``(k + ½) s`` from the
    origin."""
    x0, x1, y0, y1 = box
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    th = math.radians(spread)
    out = []
    for sign in (1.0, -1.0):
        e = np.array([math.cos(th), sign * math.sin(th)])
        n = np.array([-sign * math.sin(th), math.cos(th)])
        off, along = corners @ n, corners @ e
        m = 2.0 * nw.TRACK
        for k in range(math.floor(off.min() / s) - 1, math.ceil(off.max() / s) + 1):
            base = (k + 0.5) * s * n
            out.append(np.array([base + (along.min() - m) * e, base + (along.max() + m) * e]))
    return out


def _lattice_groove(p1, p2, gh, length: float):
    """A groove along the vector ``g = i p1 + j p2`` of a lattice, the shortest within 2°
    of the direction ``gh`` (else the nearest), and the vector of the lattice across it
    that makes a cell of ``length`` mm at least: ``(a1, a2, reps, period)``."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    det = float(p1[0] * p2[1] - p1[1] * p2[0])
    if det < 0:
        p1, p2, det = p2, p1, -det
    best = None
    for i in range(-40, 41):
        for j in range(0, 41):
            if (i, j) == (0, 0) or math.gcd(i, j) != 1 or (j == 0 and i < 0):
                continue
            v = i * p1 + j * p2
            norm = float(np.hypot(*v))
            if norm > LONGEST:
                continue
            err = math.degrees(math.acos(min(1.0, abs(float(v @ gh)) / norm)))
            rank = (0.0, norm) if err <= 2.0 else (err, norm)
            if best is None or rank < best[:2]:
                best = (*rank, i, j)
    _, _, i, j = best
    a2 = i * p1 + j * p2
    norm = float(np.hypot(*a2))
    _, x, y = _egcd(i, j)  # i x + j y = 1: b = -y p1 + x p2, det(g, b) = det(p1, p2)
    b = -y * p1 + x * p2
    reps = max(2, round(length * norm / det))
    m = round(reps * float(b @ a2) / norm**2)
    return -reps * b + m * a2, a2, reps, det / norm


def groove_geometry(
    design: str,
    ratio: float,
    theta: float,
    psi: float,
    length: float = MIN_LENGTH,
    spread: float = 0.0,
) -> dict:
    """The periodic cell of a groove whose normal lies at ``psi`` degrees from the major
    strain, the tracks (along x; two families at ``±spread`` from x for ``crossed``) at
    ``theta`` degrees from it (the major strain at ``−theta`` from x): ``a1`` (across
    the groove, ``length`` mm at least), ``a2 = g`` (along it), the number of periods of
    the pattern across the groove, the width of a period across it (``period``), the
    angle of the groove (``psi``, the nearest that the lattice offers)."""
    nu = math.radians(psi - theta)  # the normal of the groove from x
    gh = np.array([-math.sin(nu), math.cos(nu)])
    if gh[1] < -1e-12 or (abs(gh[1]) <= 1e-12 and gh[0] < 0):
        gh = -gh  # the same groove, along y upwards
    p = nw.pattern(design, ratio) if design != "crossed" else None
    if design == "crossed":
        a1, a2, reps, period = _lattice_groove(
            *crossed_lattice(spread, crossed_spacing()), gh, length
        )
    elif design == "none":
        a2 = 2.0 * gh
        reps = round(length / GROOVE)
        period = length / reps
        nh = np.array([gh[1], -gh[0]])
        a1 = length * nh
    elif design == "family" and ratio == 0.0:  # straight tracks: invariant along x
        s = p["spacing"]
        if abs(gh[1]) <= 1e-9:  # along the tracks
            a2 = np.array([2.0, 0.0])
            reps = max(2, round(length / s))
            a1 = np.array([0.0, -reps * s])
            period = s
        else:  # all the positions of the groove are alike: one band a period
            a2 = s / gh[1] * gh
            nh = np.array([gh[1], -gh[0]])
            k = round(length * nh[1] / s)
            tau = (k * s - length * nh[1]) / gh[1]
            a1 = length * nh + tau * gh
            reps = round(length / GROOVE)
            period = length / reps
    else:  # corrugated: the lattice i P x + j s y
        P, s = p["period"], p["spacing"]
        a1, a2, reps, period = _lattice_groove((P, 0.0), (0.0, s), gh, length)
    if a1[0] * a2[1] - a1[1] * a2[0] < 0:
        a1 = -a1
    gh = a2 / float(np.hypot(*a2))
    nu = math.degrees(math.atan2(-gh[0], gh[1]))  # the normal of the groove, from x
    return {
        "a1": a1,
        "a2": a2,
        "reps": int(reps),
        "period": float(period),
        "psi": (nu + theta + 90.0) % 180.0 - 90.0,
    }


def build(job: dict):
    """The cell of a job, its membrane, the bands of two rows along the groove (their
    period and their place in it, the groove band 0 of period 0) and the geometry."""
    from morphogen.laser.architectures import track_membrane
    from morphogen.laser.periodic import PeriodicCell

    spread = job.get("spread", 0.0)
    geo = groove_geometry(
        job["design"], job["ratio"], job["theta"], job["psi_target"], job["length"], spread
    )
    a1, a2 = geo["a1"], geo["a2"]
    per = 2 * max(1, round(geo["period"] / GROOVE))
    n1 = geo["reps"] * per
    n2 = max(2, math.ceil(float(np.hypot(*a2)) / job["element"] - 1e-9))
    cell = PeriodicCell.build(a1, a2, shape=(n1, n2), centre=0.5 * (a1 + a2))
    if job["design"] == "crossed":
        lines = crossed_lines(spread, crossed_spacing(), cell.box(nw.TRACK))
    else:
        lines = nw.lines_over(job["design"], job["ratio"], cell.box(nw.TRACK))
    contrast, rule = STATES[job["state"]]
    phases = g.phases_of({"law": "contrast", "contrast": contrast, "rule": rule})
    mem = track_membrane(cell.mesh, lines, nw.TRACK, phases=phases)
    row = np.arange(len(mem.quads)) // n2  # the row along a1 of each element
    edge = round(job["groove"] * per)
    k = (row - edge + 1) % n1
    bands = np.stack([k // per, (k % per) // 2], axis=1)
    groove = (bands[:, 0] == 0) & (bands[:, 1] == 0)
    for j in (1, 2, 5, 6):  # the flow stresses of both laws of the points
        mem.law[groove, ..., j] *= IMPERFECTION
    geo["groove_width"] = 2.0 * geo["period"] / per
    return cell, mem, bands, geo


def _thinning(mem) -> np.ndarray:
    """The thickness strain of each element (the mean over its points and layers)."""
    w = mem.w0
    return (np.log(mem.lam_n).mean(axis=2) * w).sum(axis=1) / w.sum(axis=1)


def _tension(job: dict) -> dict:
    """The tensile test of a design along the major strain: the periodic sheet pulled
    in uniaxial stress (its own tensile properties, the shear free or held)."""
    from morphogen.laser.architectures import track_membrane
    from morphogen.laser.cells import failure_measures
    from morphogen.laser.periodic import PeriodicCell, cell_test, reduce_basis

    t0 = time.perf_counter()
    design, ratio = job["design"], job["ratio"]
    if design == "crossed":
        s = crossed_spacing(job.get("fraction", nw.FRACTION))
        a1, a2 = reduce_basis(*crossed_lattice(job["spread"], s))

        def lines(box):
            return crossed_lines(job["spread"], s, box)

    else:
        p = nw.pattern(design, ratio)
        lx, ly = p["cell"]
        if design == "family" and ratio == 0.0:
            lx = 2.0  # straight tracks: invariant along x
        a1, a2 = np.array([lx, 0.0]), np.array([0.0, ly])

        def lines(box):
            return nw.lines_over(design, ratio, box)

    cell = PeriodicCell.build(a1, a2, element=job["element"], centre=0.5 * (a1 + a2))
    contrast, rule = STATES[job["state"]]
    phases = g.phases_of({"law": "contrast", "contrast": contrast, "rule": rule})
    mem = track_membrane(cell.mesh, lines(cell.box(nw.TRACK)), nw.TRACK, phases=phases)
    res = cell_test(
        mem,
        cell,
        -job["theta"],
        max_strain=0.6,
        failure_margin=0.03,
        free_shear=job["shear"] == "free",
    )
    fail = res.failure_strain if math.isfinite(res.failure_strain) else math.inf
    m = failure_measures(res.strain, res.stress, fail, young=mem.young)
    row = {k: job[k] for k in FIELDS if k in job}
    share = mem.law[..., 0, 8]
    row["treated"] = float((share * mem.w0).sum() / mem.w0.sum())
    row["elements"] = len(mem.quads)
    for k in ("yield_strength", "tensile_strength", "uniform_elongation", "failure_strain"):
        row[k] = m[k]
    row["energy"] = m["energy"]
    row["major"], row["minor"], row["limit"], row["psi"] = math.nan, math.nan, "tension", 0.0
    row["curve"] = [np.asarray(res.strain).tolist(), np.asarray(res.stress).tolist()]
    row["seconds"] = time.perf_counter() - t0
    return row


def evaluate(job: dict) -> dict:
    from morphogen.laser.periodic import cell_test

    if job["study"] == "tension":
        return _tension(job)
    t0 = time.perf_counter()
    cell, mem, bands, geo = build(job)
    share = mem.law[..., 0, 8]
    row = {k: job[k] for k in FIELDS if k in job}
    row.update(
        {
            "treated": float((share * mem.w0).sum() / mem.w0.sum()),
            "psi": geo["psi"],
            "elements": len(mem.quads),
            "groove_width": geo["groove_width"],
        }
    )
    angle = -job["theta"]  # the major strain from x
    w = mem.w0.sum(axis=1)
    nper, nrep = int(bands[:, 1].max()) + 1, int(bands[:, 0].max()) + 1
    flat = bands[:, 0] * nper + bands[:, 1]
    wb = np.bincount(flat, weights=w, minlength=nper * nrep).reshape(nrep, nper)
    state = {"t": np.zeros(len(w)), "after": 0, "band": 0, "groove": 0.0}

    def rate(_q):
        """The band that thins fastest against its images in the other periods."""
        t = _thinning(mem)
        d = state["t"] - t
        state["t"] = t
        db = np.bincount(flat, weights=d * w, minlength=nper * nrep).reshape(nrep, nper)
        db = db / wb
        tot, wt = (db * wb).sum(axis=0), wb.sum(axis=0)
        images = (tot[None, :] - db * wb) / (wt[None, :] - wb)
        # against its images, but not less than a tenth of the mean thinning of the cell:
        # the images of a band inside a hard track that is shortened thicken
        mean = float((d * w).sum() / w.sum())
        ref = np.maximum(images, 0.1 * mean)
        r = db / ref if mean > 1e-12 else np.zeros_like(db)
        r = r.ravel()
        k = int(np.argmax(r))
        state["band"], state["groove"] = k, float(r[0])
        state["after"] += int(r[k] >= LOCALIZED)
        return float(r[k])

    def strains(j):  # the mean strains outside the fastest band
        def f(_q):
            v = nw._major(mem, angle)[j]
            out = flat != state["band"]
            return float((v[out] * w[out]).sum() / w[out].sum())

        return f

    res = cell_test(
        mem,
        cell,
        angle,
        ratio=None if job.get("loading") == "uniaxial" else job["path"],
        free_shear=job["shear"] == "free",
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
        until=lambda _q: state["after"] >= 5,
    )
    mon = res.fields["monitors"]
    e1, e2 = np.asarray(mon["major"]), np.asarray(mon["minor"])
    ratio_ = np.nan_to_num(np.asarray(mon["rate"]), nan=0.0)
    band = np.asarray(mon["band"])
    limit, major, minor = "none", math.nan, math.nan
    hit = np.flatnonzero(ratio_ >= LOCALIZED)
    top = float(np.asarray(res.strain)[-1])
    stalled = top < 1.2 * (1 - 1e-6) and not math.isfinite(res.failure_strain)
    if hit.size:
        k = int(hit[0])
        f = 1.0
        if k > 0 and math.isfinite(ratio_[k]) and ratio_[k - 1] > 0:
            lo, hi = math.log(max(ratio_[k - 1], 1e-9)), math.log(ratio_[k])
            f = min(max((math.log(LOCALIZED) - lo) / max(hi - lo, 1e-12), 0.0), 1.0)
        j = max(k - 1, 0)
        major = float(e1[j] + f * (e1[k] - e1[j]))
        minor = float(e2[j] + f * (e2[k] - e2[j]))
        limit = "groove" if band[k] == 0 else "band"
    elif stalled:
        k = int(np.argmax(e1))
        major, minor, limit = float(e1[k]), float(e2[k]), "stall"
    if math.isfinite(res.failure_strain):
        ef = math.log1p(res.failure_strain)
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
        np.asarray(res.fields["macroscopic"]["c"]).tolist(),
    ]
    row["seconds"] = time.perf_counter() - t0
    return row


# -- the studies ----------------------------------------------------------------------


def _job(
    study,
    name,
    design,
    ratio,
    theta,
    psi,
    path,
    *,
    spread=0.0,
    state="ductile",
    shear="fixed",
    groove=None,
    length=MIN_LENGTH,
    element=0.5,
) -> dict:
    if groove is None:  # straight tracks along the groove: in the middle of the matrix
        groove = 0.0 if design == "family" and ratio == 0.0 else 0.25
    job = {
        "study": study,
        "name": name,
        "design": design,
        "state": state,
        "ratio": float(ratio),
        "theta": float(theta),
        "psi_target": float(psi),
        "path": float(path),
        "shear": shear,
        "groove": float(groove),
        "length": float(length),
        "element": float(element),
    }
    if design == "crossed":
        job["spread"] = float(spread)
    job["key"] = json.dumps({k: job[k] for k in sorted(job) if k != "name"}, sort_keys=True)
    return job


def _grooves(design: str, ratio: float, theta: float, spread: float = 0.0) -> list:
    """The angles of the grooves scanned for a design: from −40 to 40° (0 to 40° for
    the designs symmetric about the major strain), and along straight tracks."""
    symmetric = design in ("none", "crossed") or theta in (0.0, 90.0)
    out = [0.0, 10.0, 20.0, 30.0, 40.0]
    if not symmetric:
        out += [-10.0, -20.0, -30.0, -40.0]
    along = None
    if design == "family" and ratio == 0.0:
        along = theta % 180.0 - 90.0  # the groove along the tracks: ψ + 90° = θ
    elif design == "crossed":
        along = 90.0 - spread  # along one of the families
    if along is not None and abs(along) < 89.0 and all(abs(along - p) > 1.0 for p in out):
        out.append(along)
    return out


def _spread(row) -> float:
    """The half-angle between the crossed families of a row (0 for the others)."""
    v = row.get("spread")
    return 0.0 if v is None or not math.isfinite(v) else float(v)


def _refine(design, ratio, theta, spread, psi, have) -> list:
    """The grooves about the critical one ``psi``: 5° on either side, or, for the
    patterns whose grooves follow their lattice, the angles that it offers within 7°;
    those not computed yet (``have``: the angles computed)."""
    lattice = design == "crossed" or (design == "family" and ratio != 0.0)
    if not lattice:
        cand = [psi - 5.0, psi + 5.0]
    else:
        cand, seen = [], set()
        for t in np.arange(psi - 7.0, psi + 7.01, 0.5):
            a = groove_geometry(design, ratio, theta, float(t), spread=spread)["psi"]
            if abs(a - psi) <= 7.0 and round(a, 2) not in seen:
                seen.add(round(a, 2))
                cand.append(float(t))
    out = []
    for t in cand:
        a = groove_geometry(design, ratio, theta, t, spread=spread)["psi"] if lattice else t
        if abs(a) < 89.0 and all(abs(a - h) > 0.05 for h in have):
            out.append(t)
            have = [*have, a]
    return out


def jobs(study: str) -> list:
    out = []
    if study == "mk":
        for path in PATHS:
            for psi in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
                out.append(_job("mk", "untreated", "none", 0.0, 0.0, psi, path))
            # by the degree about the critical angle of the classical analysis
            centre = CRITICAL[path]
            for psi in range(max(0, round(centre) - 4), round(centre) + 5):
                out.append(_job("mk", "untreated", "none", 0.0, 0.0, float(psi), path))
        for psi in (0.0, 20.0):  # the length of the cell, the shear
            out.append(_job("mk", "untreated", "none", 0.0, 0.0, psi, -0.5, length=120.0))
            out.append(_job("mk", "untreated", "none", 0.0, 0.0, psi, -0.5, shear="free"))
    if study in ("grooves", "limits", "shear"):
        paths = (-0.375, -0.125, 0.0) if study == "limits" else (-0.5, -0.25)
        shear = "free" if study == "shear" else "fixed"
        done = load(Path(f"results/laser_drawing_{study}"))
        for name, design, ratio, theta, spread in DESIGNS:
            if design == "none" or (study == "limits" and name not in LIMITS):
                continue  # the untreated sheet: --study mk
            if study == "shear" and (design != "family" or ratio != 0.0 or theta in (0.0, 90.0)):
                continue  # the shear free: the straight tracks oblique to the load
            for path in paths:
                for psi in _grooves(design, ratio, theta, spread):
                    out.append(
                        _job(
                            study, name, design, ratio, theta, psi, path, spread=spread, shear=shear
                        )
                    )
                # about the critical groove of the results so far: a second pass
                mine = [
                    r
                    for r in done.values()
                    if r["design"] == design
                    and abs(r["ratio"] - ratio) < 1e-9
                    and abs(r["theta"] - theta) < 1e-9
                    and abs(_spread(r) - spread) < 1e-9
                    and abs(r["path"] - path) < 1e-9
                ]
                fin = [r for r in mine if math.isfinite(r["major"])]
                if fin:
                    best = min(fin, key=lambda r: r["major"])
                    have = [r["psi"] for r in mine]
                    for psi in _refine(design, ratio, theta, spread, best["psi"], have):
                        out.append(
                            _job(
                                study,
                                name,
                                design,
                                ratio,
                                theta,
                                psi,
                                path,
                                spread=spread,
                                shear=shear,
                            )
                        )
    if study == "uniaxial":  # the groove under uniaxial stress (the shear free)
        for name, design, ratio, theta, spread in (
            ("untreated", "none", 0.0, 0.0, 0.0),
            ("crossed ±60°", "crossed", 0.0, 0.0, 60.0),
            ("crossed ±55°", "crossed", 0.0, 0.0, 55.0),
            ("corrugated 0.25", "family", 0.25, 0.0, 0.0),
            ("straight 45°", "family", 0.0, 45.0, 0.0),
        ):
            grooves = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
            if design == "family" and ratio == 0.0:
                grooves = (*grooves, -10.0, -20.0, -30.0, -40.0, -45.0)
            for psi in grooves:
                job = _job(
                    "uniaxial", name, design, ratio, theta, psi, -0.5, spread=spread, shear="free"
                )
                job["loading"] = "uniaxial"
                job["key"] = json.dumps(
                    {k: job[k] for k in sorted(job) if k != "name"}, sort_keys=True
                )
                out.append(job)
    if study == "tension":  # the tensile properties of the designs along the major strain
        # the half-angle of crossed families about ±60°, their fraction, the mesh
        for spread in (56.0, 58.0, 61.0, 62.0, 63.0, 64.0):
            for state in ("ductile", "as-quenched"):
                out.append(
                    _job(
                        "tension",
                        f"crossed ±{spread:g}°",
                        "crossed",
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        spread=spread,
                        state=state,
                        shear="free",
                    )
                )
        for fraction in (0.12, 0.36):
            for spread in (55.0, 60.0, 65.0):
                for state in ("ductile", "as-quenched"):
                    job = _job(
                        "tension",
                        f"crossed ±{spread:g}°",
                        "crossed",
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        spread=spread,
                        state=state,
                        shear="free",
                    )
                    job["fraction"] = fraction
                    job["key"] = json.dumps(
                        {k: job[k] for k in sorted(job) if k != "name"}, sort_keys=True
                    )
                    out.append(job)
        for spread in (55.0, 60.0):
            for state in ("ductile", "as-quenched"):
                out.append(
                    _job(
                        "tension",
                        f"crossed ±{spread:g}°",
                        "crossed",
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        spread=spread,
                        state=state,
                        shear="free",
                        element=0.25,
                    )
                )
        designs = [("untreated", "none", 0.0, 0.0, 0.0)]
        designs += [d for d in DESIGNS if d[0] in TENSION]
        for name, design, ratio, theta, spread in designs:
            for state in STATES if design != "none" else ("ductile",):
                for shear in ("free", "fixed"):
                    if shear == "fixed" and (design == "none" or theta in (0.0, 90.0)):
                        continue  # symmetric about the load: the shear stays nil
                    out.append(
                        _job(
                            "tension",
                            name,
                            design,
                            ratio,
                            theta,
                            0.0,
                            0.0,
                            spread=spread,
                            state=state,
                            shear=shear,
                        )
                    )
    unique = {}
    for j in out:
        unique.setdefault(j["key"], j)
    return list(unique.values())


def run(todo: list, workers: int, done: dict, out: Path) -> dict:
    todo = [j for j in todo if j["key"] not in done]
    todo.sort(key=lambda j: (j["design"] == "none", -j["ratio"]))  # the longest first
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=g._init) as pool:
        futures = {pool.submit(evaluate, j): j for j in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            done[row["key"]] = row
            print(
                f"{n}/{len(todo)} {row['name']} ρ {row['path']:+.3f} ψ {row['psi']:+.1f} "
                f"{row['shear']} {row['state']}: {row['major']:.4f} {row['limit']} "
                f"({row['elements']} el., {row['seconds']:.0f} s; "
                f"{time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
            if n % 10 == 0:
                save(done, out)
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
        **{f"c{i}": np.array(v, dtype=object) for i, v in enumerate(curves.values())},
    )


def load(path: Path) -> dict:
    rows = {}
    csv_path = path.with_suffix(".csv")
    if not csv_path.exists():
        return rows
    text = {"key", "study", "name", "design", "state", "shear", "limit"}
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in set(FIELDS) - text:
                r[k] = float(r[k]) if r.get(k) not in (None, "") else math.nan
            rows[r["key"]] = r
    npz = path.with_suffix(".npz")
    if npz.exists():
        data = np.load(npz, allow_pickle=True)
        for i, key in enumerate(data["keys"]):
            if str(key) in rows:
                rows[str(key)]["curve"] = [np.asarray(c, dtype=float) for c in data[f"c{i}"]]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--study",
        default="mk",
        choices=("mk", "grooves", "limits", "shear", "tension", "uniaxial"),
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    out = args.output or Path(f"results/laser_drawing_{args.study}")
    done = load(out)
    todo = jobs(args.study)
    have = sum(j["key"] in done for j in todo)
    print(f"{args.study}: {len(todo)} jobs ({have} done)", flush=True)
    rows = run(todo, args.workers, done, out)
    save(rows, out)
    print(f"written: {out}.csv")


if __name__ == "__main__":
    main()
