"""Three-point bending of a laser-architectured strip: fibre sections of a beam.

A strip of width ``W`` and thickness ``h`` rests on two supports ``span`` apart and is
loaded at mid-span by ``F``: the bending moment is ``M(x) = F/2 (S/2 − |x − x_c|)``
(statically determinate, small deflections, no membrane force). Each cross-section is a
grid of fibres across the width and through the thickness (the layers of the
treatment), each with the flow law of its hardness, loaded monotonically in uniaxial
stress: ``|σ| = E|ε|`` up to yield, then ``σ = σ_y(p)`` with ``|ε| = p + σ/E``. Plane
sections: ``ε = −κ (z − z_n)``, the neutral axis ``z_n`` from ``Σ σ dA = 0`` (it moves
off the mid-plane of a strip treated on one face), ``M = −Σ σ (z − z_n) dA``. The
curvature of each section follows from the moment by inverting ``M(κ)``, the deflection
under the load from the principle of virtual work, ``δ = ∫ κ(x) m(x) dx``,
``m = (S/2 − |x − x_c|)/2`` (exact for the statically determinate beam). The limit load
``4 M_max / S`` is reached when the weakest section near the load cannot carry more.
Checked: the elastic stiffness ``48 E I / S³`` and the limit load ``σ0 W h² / S`` of a
perfectly plastic strip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BendingResult:
    """The load (N) against the deflection (mm) under it, the largest load of the tables
    (the limit load of a perfectly plastic strip; of a hardening one, the load at an
    outer strain of about 100 times the strain of first yield), the initial stiffness
    (N/mm), the load and the energy (mJ) at the reference deflection, and the
    moment-curvature tables of the sections."""

    deflection: np.ndarray
    load: np.ndarray
    max_load: float
    stiffness: float
    load_at_deflection: float
    energy: float
    reference_deflection: float
    sections: dict = field(default_factory=dict, repr=False)

    def summary(self) -> dict:
        return {
            "bending_load": self.max_load,
            "bending_stiffness": self.stiffness,
            "bending_load_at_deflection": self.load_at_deflection,
            "bending_energy": self.energy,
        }


def _fibre_curves(kind, s0, a, b, young, strains, offset):
    """Stresses of fibres (n) at the strains (m) of a common grid, monotonic uniaxial
    loading: ``(n, m)``."""
    kind = np.asarray(kind)[:, None]
    s0, a, b = (np.asarray(v, dtype=float)[:, None] for v in (s0, a, b))
    e = np.asarray(strains, dtype=float)[None, :]

    def flow(p):
        ludwik = s0 + a * ((p + offset) ** b - offset**b)
        voce = s0 + a * (1.0 - np.exp(-b * p))
        with np.errstate(invalid="ignore", divide="ignore"):
            swift = a * ((s0 / a) ** (1.0 / b) + p) ** b
        return np.where(kind == 0, ludwik, np.where(kind == 2, swift, voce))

    elastic = young * e
    plastic = elastic > s0
    lo = np.zeros(np.broadcast(kind, e).shape)
    hi = np.broadcast_to(e, lo.shape).copy()
    for _ in range(48):  # bisection on p: e = p + sigma(p)/E
        mid = 0.5 * (lo + hi)
        up = mid + flow(mid) / young < e
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    p = 0.5 * (lo + hi)
    return np.where(plastic, flow(p), elastic)


def section_tables(
    kind,
    s0,
    a,
    b,
    *,
    width: float,
    thickness: float,
    young: float,
    curvatures,
    offset: float = 1e-4,
    strains: int = 600,
) -> dict:
    """Moment-curvature tables of sections: flow laws ``(sections, ny, layers)``
    (fibres across the width and through the thickness, the layer 0 at the upper
    face); returns the curvatures, moments ``(sections, curvatures)`` and neutral axes."""
    kind = np.asarray(kind)
    ns, ny, nl = kind.shape
    h = float(thickness)
    z = (np.arange(nl) + 0.5) * h / nl - 0.5 * h  # from the mid-plane, upwards = upper face
    z = -z  # layer 0 (upper face) at z = +h/2
    da = width / ny * h / nl
    kap = np.asarray(curvatures, dtype=float)
    emax = float(np.max(kap) * h * 1.05) + 1e-12
    grid = np.linspace(0.0, emax, strains)
    laws = np.column_stack([kind.ravel(), np.ravel(s0), np.ravel(a), np.ravel(b)])
    unique, inverse = np.unique(laws, axis=0, return_inverse=True)  # the distinct laws
    curves = _fibre_curves(
        unique[:, 0].astype(np.int64),
        unique[:, 1],
        unique[:, 2],
        unique[:, 3],
        young,
        grid,
        offset,
    )[inverse.ravel()].reshape(ns, ny, nl, strains)
    # a section: its fibres share z; sum the stresses across the width per layer
    layer_curves = curves.sum(axis=1) * da  # ns x nl x strains (force per strain)
    de = grid[1] - grid[0]

    def force(eps):  # ns x nl strains -> ns x nl forces (odd in the strain)
        idx = np.clip(np.abs(eps) / de, 0.0, strains - 1.000001)
        i0 = np.floor(idx).astype(np.int64)
        f = idx - i0
        ii = np.arange(ns)[:, None]
        ll = np.arange(nl)[None, :]
        v = (1.0 - f) * layer_curves[ii, ll, i0] + f * layer_curves[ii, ll, i0 + 1]
        return np.sign(eps) * v

    moments = np.zeros((ns, len(kap)))
    axes = np.zeros((ns, len(kap)))
    for j, k in enumerate(kap):
        if k == 0.0:
            continue
        lo, hi = np.full(ns, -0.5 * h), np.full(ns, 0.5 * h)
        for _ in range(50):  # the neutral axis: the axial force vanishes
            zn = 0.5 * (lo + hi)
            n_ax = force(-k * (z[None, :] - zn[:, None])).sum(axis=1)
            tension = n_ax > 0  # the force grows as the axis rises: lower it
            hi = np.where(tension, zn, hi)
            lo = np.where(tension, lo, zn)
        zn = 0.5 * (lo + hi)
        f = force(-k * (z[None, :] - zn[:, None]))
        moments[:, j] = -(f * (z[None, :] - zn[:, None])).sum(axis=1)
        axes[:, j] = zn
    return {"curvatures": kap, "moments": moments, "axes": axes, "z": z}


def _deflection(integrand, xs) -> float:
    """``∫ κ m dx`` by Simpson's rule on each half of the span (the moment has a kink
    under the load, at the middle section)."""
    from scipy.integrate import simpson

    mid = len(xs) // 2
    if len(xs) % 2 == 0 or mid < 2:
        return float(np.trapezoid(integrand, xs))
    left = simpson(integrand[: mid + 1], x=xs[: mid + 1])
    right = simpson(integrand[mid:], x=xs[mid:])
    return float(left + right)


def three_point_bending(
    kind,
    s0,
    a,
    b,
    *,
    span: float,
    width: float,
    thickness: float,
    young: float = 210000.0,
    deflection: float | None = None,
    loads: int = 120,
) -> BendingResult:
    """Three-point bending (see the module) of a strip whose sections, equally spaced
    over the span (``(sections, ny, layers)`` flow laws, the first section at the left
    support, the last at the right one; an odd number puts one under the load), are
    given; ``deflection``: the reference
    deflection of the load and the energy (default: a fiftieth of the span)."""
    kind = np.asarray(kind)
    ns = kind.shape[0]
    s = float(span)
    xs = np.linspace(0.0, s, ns)
    arm = 0.5 * s - np.abs(xs - 0.5 * s)  # M = F/2 arm, m = arm/2
    h, w = float(thickness), float(width)
    inertia = w * h**3 / 12.0
    s0_min = float(np.nanmin(s0))
    k_el = 2.0 * s0_min / (young * h)
    kap = np.concatenate(
        [np.linspace(0.0, k_el, 20, endpoint=False), np.geomspace(k_el, 100.0 * k_el, 160)]
    )
    tab = section_tables(
        kind, s0, a, b, width=w, thickness=h, young=young, curvatures=kap, strains=2000
    )
    mom = np.maximum.accumulate(tab["moments"], axis=1)  # monotone tables
    m_max = mom[:, -1]
    with np.errstate(divide="ignore"):
        f_lim = np.where(arm > 0, 2.0 * m_max / np.maximum(arm, 1e-300), np.inf)
    f_max = float(f_lim.min())
    forces = f_max * (1.0 - np.geomspace(1.0, 1e-3, loads))  # dense near the limit
    forces = np.concatenate([[0.0], forces[forces > 0]])
    defl = np.zeros(len(forces))
    for i, f in enumerate(forces):
        mx = 0.5 * f * arm
        kx = np.array([np.interp(mx[j], mom[j], kap) for j in range(ns)])
        defl[i] = _deflection(kx * 0.5 * arm, xs)
    ref = float(deflection) if deflection is not None else s / 50.0
    stiff = 48.0 * young * inertia / s**3
    if len(forces) > 2 and defl[1] > 0:
        stiff = float(forces[1] / defl[1])
    reached = defl[-1] >= ref
    f_ref = float(np.interp(ref, defl, forces)) if reached else math.nan
    upto = defl <= ref
    dd = np.append(defl[upto], ref) if reached else defl
    ff = np.append(forces[upto], f_ref) if reached else forces
    energy = float(np.trapezoid(ff, dd))  # N mm = mJ
    return BendingResult(defl, forces, f_max, stiff, f_ref, energy, ref, sections={"x": xs, **tab})
