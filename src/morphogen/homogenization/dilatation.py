"""Selective reduced integration of the 10-node tetrahedra: the volume change of each
element taken as its mean (``integration="bbar"`` of the finite element problems).

A displacement element whose strain is integrated at every point enforces, as a phase
becomes incompressible (Poisson's ratio near 1/2: elastomers; plastic flow), ``div u =
0`` at each of its points: four constraints per 10-node tetrahedron, as many as the mesh
has degrees of freedom to spare, and the element *locks* — its stresses oscillate and
its displacements fall short. Integrating the volumetric part of the energy at one point
(the mean over the element) and the rest fully leaves one constraint per element
(Nagtegaal, Parks & Rice 1974; the mean dilatation, or ``B``-bar, of Hughes 1980;
equivalent to a mixed element with a pressure and a dilatation uniform in each element,
Simo, Taylor & Pister 1985).

Small strains: the trace of the strain of each point is replaced by its mean over the
element, ``eps_bar = eps + (theta_bar - tr eps) 1 / 3``, ``theta_bar = sum_g w_g tr
eps_g / V``. Finite strains (``F``-bar, de Souza Neto, Peric, Dutko & Owen 1996): the
gradient of each point is scaled to the geometric mean of the Jacobians of the element,
``F_bar = alpha F``, ``alpha = (J_bar / J)^(1/3)``, ``ln J_bar = sum_g w_g ln J_g / V``.
The behaviours are integrated with these measures, and the forces are the derivatives
of their energy, so that the tangent is symmetric whenever those of the points are: the
stresses of the points are replaced by their conjugates, ``s + (p_bar - p) 1`` (the
deviator of the point and the mean pressure of the element) and ``alpha P + (tau_bar -
tau) F^-T`` (``tau = alpha P : F / 3``), as the voxels of the FFT solver do
(:mod:`morphogen.homogenization.kernels`).

The tangent of an element is ``sum_g w_g G_g^T M_g G_g + q_L c^T + c q_R^T + k c c^T``:
the usual sum over the points with modified moduli ``M_g``, and a term of rank two
through the mean operator ``c = sum_g w_g G_g^T f_g / V`` of the volume change (``f = 1``
in small strains, ``F^-T`` in finite strains; ``G`` the strain or gradient operator of a
point). Small strains, ``P = I - 1 (x) 1 / 3``: ``M = P C P``, ``q_L = sum w G^T (C 1 - k
1 / 3) / 3``, ``k = 1 . C 1``. Finite strains, with ``a = A F``, ``b = A^T F``, ``s = F .
a`` (``A = dP/dF`` at ``F_bar``, ``9 x 9``): ``M = alpha^2 (A - a (x) f / 3 - f (x) b / 3
+ s f (x) f / 9) - alpha (f (x) P + P (x) f) / 3 + tau f (x) f / 3 + (tau - tau_bar) B``,
``B_(ci)(ab) = f_ai f_cb`` (the second derivative of ``-ln J``), ``q_L = sum w G^T
(alpha^2 (a - s f / 3) / 3 + alpha P / 3 - tau f / 3)`` and ``k = sum w (alpha^2 s / 9 +
tau / 3)``.

A 4-node tetrahedron has one point: its strain is its mean, and these measures change
nothing.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from morphogen.homogenization.fe import (
    CHUNKS,
    INTEGRATIONS,
    _geometry,
    _point_gradients,
    integration_of,
    nearly_incompressible,
)
from morphogen.homogenization.kernels import _det3, _inverse_transpose


@njit(parallel=True, cache=True)
def mean_strains(eps, wts, ng):
    """``B``-bar: the trace of the strains of the points (points x 6) replaced, in place,
    by its mean over each element (``ng`` points, weights ``wts``)."""
    for e in prange(eps.shape[0] // ng):
        vol = 0.0
        mean = 0.0
        for g in range(ng):
            i = e * ng + g
            vol += wts[i]
            mean += wts[i] * (eps[i, 0] + eps[i, 1] + eps[i, 2])
        mean /= vol
        for g in range(ng):
            i = e * ng + g
            shift = (mean - (eps[i, 0] + eps[i, 1] + eps[i, 2])) / 3.0
            for r in range(3):
                eps[i, r] += shift


@njit(parallel=True, cache=True)
def mean_gradients(hg, wts, ng, base):
    """``F``-bar: the displacement gradients of the points (points x 9) kept in ``base``
    and replaced, in place, by ``alpha F - 1``, the gradients scaled to the geometric
    mean of the Jacobians of each element. An element with an inverted point keeps its
    gradients (the behaviours fail there, and the step is divided)."""
    for e in prange(hg.shape[0] // ng):
        f = np.empty((3, 3))
        logs = np.empty(ng)
        vol = 0.0
        mean = 0.0
        valid = True
        for g in range(ng):
            i = e * ng + g
            for k in range(9):
                base[i, k] = hg[i, k]
                f[k // 3, k % 3] = hg[i, k] + (1.0 if k % 4 == 0 else 0.0)
            det = _det3(f)
            if det <= 0.0:
                valid = False
                break
            logs[g] = math.log(det)
            vol += wts[i]
            mean += wts[i] * logs[g]
        if not valid:
            for g in range(ng):
                i = e * ng + g
                for k in range(9):
                    base[i, k] = hg[i, k]
            continue
        mean /= vol
        for g in range(ng):
            i = e * ng + g
            alpha = math.exp((mean - logs[g]) / 3.0)
            for k in range(9):
                one = 1.0 if k % 4 == 0 else 0.0
                hg[i, k] = alpha * (base[i, k] + one) - one


@njit(parallel=True, cache=True)
def conjugate_stresses(sigma, wts, ng):
    """The stresses of the points (points x 6) replaced, in place, by those conjugate to
    the strains of ``B``-bar: their deviator and the mean pressure of the element."""
    for e in prange(sigma.shape[0] // ng):
        vol = 0.0
        mean = 0.0
        for g in range(ng):
            i = e * ng + g
            vol += wts[i]
            mean += wts[i] * (sigma[i, 0] + sigma[i, 1] + sigma[i, 2])
        mean /= 3.0 * vol
        for g in range(ng):
            i = e * ng + g
            shift = mean - (sigma[i, 0] + sigma[i, 1] + sigma[i, 2]) / 3.0
            for r in range(3):
                sigma[i, r] += shift


@njit(cache=True, inline="always")
def _element_scales(base, wts, ng, e, f, fit, alpha, finv):
    """The factors ``alpha`` of the points of element ``e`` and their ``F^-T`` (``finv``,
    points x 9), from the gradients ``base``; false for an inverted element."""
    vol = 0.0
    mean = 0.0
    for g in range(ng):
        i = e * ng + g
        for k in range(9):
            f[k // 3, k % 3] = base[i, k] + (1.0 if k % 4 == 0 else 0.0)
        det = _det3(f)
        if det <= 0.0:
            return False
        _inverse_transpose(f, det, fit)
        for k in range(9):
            finv[g, k] = fit[k // 3, k % 3]
        alpha[g] = math.log(det)
        vol += wts[i]
        mean += wts[i] * alpha[g]
    mean /= vol
    for g in range(ng):
        alpha[g] = math.exp((mean - alpha[g]) / 3.0)
    return True


@njit(parallel=True, cache=True)
def conjugate_piola(pk1, wts, ng, base, kept):
    """The first Piola-Kirchhoff stresses of the points (points x 9, those of the
    behaviours at ``F_bar``) kept in ``kept`` and replaced, in place, by those conjugate
    to the gradients ``base`` of the points: ``alpha P + (tau_bar - tau) F^-T``."""
    for e in prange(pk1.shape[0] // ng):
        f = np.empty((3, 3))
        fit = np.empty((3, 3))
        alpha = np.empty(ng)
        tau = np.empty(ng)
        finv = np.empty((ng, 9))
        for g in range(ng):
            i = e * ng + g
            for k in range(9):
                kept[i, k] = pk1[i, k]
        if not _element_scales(base, wts, ng, e, f, fit, alpha, finv):
            continue
        vol = 0.0
        mean = 0.0
        for g in range(ng):
            i = e * ng + g
            pf = 0.0
            for k in range(9):
                pf += pk1[i, k] * (base[i, k] + (1.0 if k % 4 == 0 else 0.0))
            tau[g] = alpha[g] * pf / 3.0
            vol += wts[i]
            mean += wts[i] * tau[g]
        mean /= vol
        for g in range(ng):
            i = e * ng + g
            for k in range(9):
                pk1[i, k] = alpha[g] * pk1[i, k] + (mean - tau[g]) * finv[g, k]


@njit(parallel=True, cache=True)
def bar_moduli(tangent, moduli, vl, vr, kappa):
    """Small strains: the moduli ``P C P`` of the points, the vectors ``(C 1 - k 1 / 3) /
    3`` (``vl``) and ``(C^T 1 - k 1 / 3) / 3`` (``vr``) and ``k / 9`` (see the module)."""
    for i in prange(tangent.shape[0]):
        d = tangent[i]
        cm = np.empty(6)
        mc = np.empty(6)
        k = 0.0
        for r in range(6):
            cm[r] = d[r, 0] + d[r, 1] + d[r, 2]
            mc[r] = d[0, r] + d[1, r] + d[2, r]
        k = cm[0] + cm[1] + cm[2]
        for r in range(6):
            mr = 1.0 if r < 3 else 0.0
            vl[i, r] = (cm[r] - k * mr / 3.0) / 3.0
            vr[i, r] = (mc[r] - k * mr / 3.0) / 3.0
            for s in range(6):
                ms = 1.0 if s < 3 else 0.0
                moduli[i, r, s] = d[r, s] - (cm[r] * ms + mr * mc[s]) / 3.0 + k * mr * ms / 9.0
        kappa[i] = k / 9.0


@njit(parallel=True, cache=True)
def bar_moduli_finite(tangent, wts, ng, base, kept, moduli, vl, vr, kappa, fvec):
    """Finite strains: the moduli ``M`` of the points, the vectors of the terms of rank
    two, their scalar and ``F^-T`` (``fvec``), from the tangents ``dP/dF`` at ``F_bar``,
    the gradients ``base`` and the stresses ``kept`` of the behaviours (see the module).
    An inverted element keeps the moduli of its points (and no term of rank two)."""
    for e in prange(tangent.shape[0] // ng):
        f = np.empty((3, 3))
        fit = np.empty((3, 3))
        alpha = np.empty(ng)
        tau = np.empty(ng)
        finv = np.empty((ng, 9))
        fv = np.empty(9)
        a = np.empty(9)
        b = np.empty(9)
        if not _element_scales(base, wts, ng, e, f, fit, alpha, finv):
            for g in range(ng):
                i = e * ng + g
                kappa[i] = 0.0
                for r in range(9):
                    vl[i, r] = 0.0
                    vr[i, r] = 0.0
                    fvec[i, r] = 0.0
                    for s in range(9):
                        moduli[i, r, s] = tangent[i, r, s]
            continue
        vol = 0.0
        mean = 0.0
        for g in range(ng):
            i = e * ng + g
            pf = 0.0
            for k in range(9):
                pf += kept[i, k] * (base[i, k] + (1.0 if k % 4 == 0 else 0.0))
            tau[g] = alpha[g] * pf / 3.0
            vol += wts[i]
            mean += wts[i] * tau[g]
        mean /= vol
        for g in range(ng):
            i = e * ng + g
            al = alpha[g]
            t = tau[g]
            d = tangent[i]
            fi = finv[g]
            for k in range(9):
                fv[k] = base[i, k] + (1.0 if k % 4 == 0 else 0.0)
            s = 0.0
            for r in range(9):
                acc_a = 0.0
                acc_b = 0.0
                for c in range(9):
                    acc_a += d[r, c] * fv[c]
                    acc_b += d[c, r] * fv[c]
                a[r] = acc_a
                b[r] = acc_b
            for r in range(9):
                s += fv[r] * a[r]
            for r in range(9):
                fvec[i, r] = fi[r]
                vl[i, r] = al * al * (a[r] - s * fi[r] / 3.0) / 3.0 + al * kept[i, r] / 3.0
                vl[i, r] -= t * fi[r] / 3.0
                vr[i, r] = al * al * (b[r] - s * fi[r] / 3.0) / 3.0 + al * kept[i, r] / 3.0
                vr[i, r] -= t * fi[r] / 3.0
                for c in range(9):
                    value = (
                        al
                        * al
                        * (
                            d[r, c]
                            - a[r] * fi[c] / 3.0
                            - fi[r] * b[c] / 3.0
                            + s * fi[r] * fi[c] / 9.0
                        )
                    )
                    value -= al * (fi[r] * kept[i, c] + kept[i, r] * fi[c]) / 3.0
                    value += t * fi[r] * fi[c] / 3.0
                    # B_(ci)(ab) = f_ai f_cb, with r = (c, i) and the column (a, b)
                    value += (t - mean) * fi[3 * (c // 3) + r % 3] * fi[3 * (r // 3) + c % 3]
                    moduli[i, r, c] = value
            kappa[i] = al * al * s / 9.0 + t / 3.0


@njit(cache=True, inline="always")
def _operator(finite, grad, nn, gop):
    """The strain operator (6 x 3 nn: Voigt, engineering shears) or the gradient
    operator (9 x 3 nn: rows ``3 i + j``) of a point, by columns: ``gop[b, k, :]``."""
    for b in range(nn):
        gx, gy, gz = grad[b, 0], grad[b, 1], grad[b, 2]
        for k in range(3):
            for r in range(gop.shape[2]):
                gop[b, k, r] = 0.0
        if finite:
            for k in range(3):
                gop[b, k, 3 * k] = gx
                gop[b, k, 3 * k + 1] = gy
                gop[b, k, 3 * k + 2] = gz
        else:
            gop[b, 0, 0] = gx
            gop[b, 0, 4] = gz
            gop[b, 0, 5] = gy
            gop[b, 1, 1] = gy
            gop[b, 1, 3] = gz
            gop[b, 1, 5] = gx
            gop[b, 2, 2] = gz
            gop[b, 2, 3] = gy
            gop[b, 2, 4] = gx


@njit(parallel=True, cache=True)
def fill_bar(nodes, elements, order, gp, wts, finite, moduli, vl, vr, kappa, fvec, indptr_n,
             indices_n, by_colour, colours, data):  # fmt: skip
    """CSR values of the tangent of ``B``-bar or ``F``-bar (node-level pattern): for
    each element, ``sum_g w_g G_g^T M_g G_g + q_L c^T + c q_R^T + k c c^T`` (see the
    module; ``fvec``: the vectors ``f`` of the points in finite strains, unused in small
    strains, where ``f = 1``), the elements of a colour in parallel (``by_colour``,
    ``colours``: :func:`~morphogen.homogenization.fe.element_colours`)."""
    nn = elements.shape[1]
    ng = gp.shape[0]
    nc = 9 if finite else 6
    nd = 3 * nn
    for i in prange(data.size):
        data[i] = 0.0
    xs = np.empty((CHUNKS, 4, 3))
    gls = np.empty((CHUNKS, 4, 3))
    devs = np.empty((CHUNKS, 6, 3))
    grads = np.empty((CHUNKS, nn, 3))
    gops = np.empty((CHUNKS, nn, 3, nc))
    mgs = np.empty((CHUNKS, nn, 3, nc))
    kes = np.empty((CHUNKS, nd, nd))
    cs = np.empty((CHUNKS, nd))
    qls = np.empty((CHUNKS, nd))
    qrs = np.empty((CHUNKS, nd))
    places = np.empty((CHUNKS, nn, nn), dtype=np.int64)
    for colour in range(colours.size - 1):
        first = colours[colour]
        count = colours[colour + 1] - first
        chunks = min(count, CHUNKS)
        for chunk in prange(chunks):
            x = xs[chunk]
            gl = gls[chunk]
            dev = devs[chunk]
            grad = grads[chunk]
            gop = gops[chunk]
            mg = mgs[chunk]
            ke = kes[chunk]
            c = cs[chunk]
            ql = qls[chunk]
            qr = qrs[chunk]
            pos = places[chunk]
            for j in range(first + chunk * count // chunks, first + (chunk + 1) * count // chunks):
                e = by_colour[j]
                _, curved = _geometry(nodes, elements, e, x, gl, dev)
                ke[:, :] = 0.0
                c[:] = 0.0
                ql[:] = 0.0
                qr[:] = 0.0
                kap = 0.0
                vol = 0.0
                for g in range(ng):
                    _point_gradients(order, curved, gl, dev, gp[g], grad)
                    _operator(finite, grad, nn, gop)
                    i = e * ng + g
                    w = wts[i]
                    m = moduli[i]
                    vol += w
                    kap += w * kappa[i]
                    for b in range(nn):
                        for k in range(3):
                            col = gop[b, k]
                            fc = 0.0
                            fl = 0.0
                            fr = 0.0
                            for r in range(nc):
                                if finite:
                                    fc += col[r] * fvec[i, r]
                                elif r < 3:
                                    fc += col[r]
                                fl += col[r] * vl[i, r]
                                fr += col[r] * vr[i, r]
                                acc = 0.0
                                for s in range(nc):
                                    acc += m[r, s] * col[s]
                                mg[b, k, r] = acc
                            c[3 * b + k] += w * fc
                            ql[3 * b + k] += w * fl
                            qr[3 * b + k] += w * fr
                    for a in range(nn):
                        for k in range(3):
                            row = gop[a, k]
                            for b in range(nn):
                                for k2 in range(3):
                                    acc = 0.0
                                    for r in range(nc):
                                        acc += row[r] * mg[b, k2, r]
                                    ke[3 * a + k, 3 * b + k2] += w * acc
                for a in range(nd):
                    c[a] /= vol
                for a in range(nd):
                    for b in range(nd):
                        ke[a, b] += ql[a] * c[b] + c[a] * qr[b] + kap * c[a] * c[b]
                for a in range(nn):
                    na = elements[e, a]
                    lo = indptr_n[na]
                    hi = indptr_n[na + 1]
                    for b in range(nn):
                        nb = elements[e, b]
                        left, right = lo, hi
                        while left < right:
                            mid = (left + right) // 2
                            if indices_n[mid] < nb:
                                left = mid + 1
                            else:
                                right = mid
                        pos[a, b] = left - lo
                for a in range(nn):
                    na = elements[e, a]
                    length = indptr_n[na + 1] - indptr_n[na]
                    for k in range(3):
                        start = 9 * indptr_n[na] + k * 3 * length
                        for b in range(nn):
                            p = pos[a, b]
                            for k2 in range(3):
                                data[start + p * 3 + k2] += ke[3 * a + k, 3 * b + k2]


__all__ = [
    "INTEGRATIONS",
    "bar_moduli",
    "bar_moduli_finite",
    "conjugate_piola",
    "conjugate_stresses",
    "fill_bar",
    "integration_of",
    "mean_gradients",
    "mean_strains",
    "nearly_incompressible",
]
