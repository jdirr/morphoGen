"""Finite strains: the behaviours of the points and the total Lagrangian finite elements.

The displacement gradient ``H = F - I`` of the reference configuration (the mesh) is the
measure of strain; the first Piola-Kirchhoff stress ``P`` (force per unit reference area)
its conjugate, and ``A = dP/dF`` the tangent. Tensors of 9 components are stored row by
row (``iJ -> 3 i + J``).

* Elastic, von Mises and porous points work in the space of the logarithmic strain
  (Miehe, Apel & Lambrecht 2002): ``eps = 1/2 ln C`` (``C = F^T F``) replaces the small strain
  in the behaviour of :mod:`morphogen.homogenization.behaviours`, whose stress ``T`` is
  conjugate to it; ``S = T : 2 d eps/dC`` and ``P = F S``. Elasticity is then Hencky's
  (the quadratic energy of the logarithmic strain), the plastic flow additive in ``eps``
  and isochoric, and the formulation reduces to the small strain one for small strains.
  The derivatives of the logarithm of ``C`` are those of a function of a symmetric
  tensor, ``(Df(C)[H])_ab = f[c_a, c_b] H_ab`` in the eigenbasis of ``C`` and likewise
  for the second derivative with the second divided differences (Daleckii-Krein), exact
  for equal eigenvalues.
* Crystal plasticity is multiplicative, ``F = F_e F_p``
  (:func:`morphogen.homogenization.crystal.cp_finite`).

The finite element equations are those of the small strains with ``P`` and ``A``: the
nodal forces ``int P . grad N dV_0``, the stiffness ``int grad N . A . grad N dV_0``. The
macroscopic gradient is upper triangular (``H_21 = H_31 = H_32 = 0``): every gradient is
one of them up to a rotation (its QR decomposition), which the behaviours ignore. Its six
components ``H_11, H_22, H_33, H_23, H_13, H_12`` are the macroscopic degrees of freedom,
in the Voigt order of the small strains (the shears ``H_ij``, ``i < j``, are simple
shears, the engineering shears of small strains), conjugate to the same components of
the mean of ``P`` (nominal stresses: forces per unit reference area). Free components
(zero stress) leave the faces of the box free of tractions but the reactions ``P_21,
P_31, P_32``, which keep the material lines along the first axis along it and the
planes normal to the third axis normal to it: a tension along an axis is then a uniaxial
Cauchy stress, whatever the anisotropy of the volume element.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from morphogen.homogenization.behaviours import (
    CRYSTAL,
    HILL,
    MFRONT,
    PLASTIC,
    POROUS,
    _hill_set,
    _hill_update,
    _hill_works,
    _j2_update,
    point_chunks,
)
from morphogen.homogenization.crystal import cp_finite, work_block, work_blocks
from morphogen.homogenization.fe import CHUNKS, _geometry, _point_gradients, chunk_slots
from morphogen.homogenization.porous import _porous_set, _porous_update, _porous_works
from morphogen.homogenization.tensors import (
    eigh3,
    half_log,
    log_divided,
    log_divided2,
    matmul3,
    matmult3,
    strain_voigt,
    stress_tensor,
    symmetrize,
    tmatmul3,
)

#: Revision of the kernels of the other modules compiled into
#: :func:`update_points_finite` (crystal, behaviours, porous): as
#: ``morphogen.homogenization.behaviours._KERNELS``, a change of those kernels comes
#: with a change of this number, which makes the caches of Numba of the earlier
#: versions stale (2: twins and cleavage planes among the systems of the crystals).
_KERNELS = 22

#: The six symmetric directions of dC in the tangent of the logarithmic points
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
#: Components (11, 22, 33, 23, 13, 12) of a tensor of 9 components of the macroscopic
#: strains and stresses: the upper triangle
MACRO = (0, 4, 8, 5, 2, 1)


# -- behaviours of the points ----------------------------------------------------------------


@njit(cache=True)
def _log_point(kind, dm, par, nk, hg, eth, dt, old, new, pk1, tan, buf, lw, v6, tan6, f2,
               hw, pw, sym=True):  # fmt: skip
    """Elastic, von Mises, Hill or porous point (``kind``) in the space of the logarithmic
    strain: displacement gradient ``hg``, thermal strain ``eth`` (tensor components,
    logarithmic), work arrays ``hw`` of Hill's points and ``pw`` of the porous ones. Fills
    the first Piola-Kirchhoff stress and the tangent ``dP/dF`` (of the tangent of the
    logarithmic strain symmetrized with ``sym``); returns 1 on failure."""
    f = lw[0]
    e2 = lw[1]
    q = lw[2]
    tq = lw[3]
    s = lw[4]
    t1 = lw[5]
    t2 = lw[6]
    dcq = lw[7]
    dtq = lw[8]
    f1 = lw[9]
    mu = v6[0]
    eps6 = v6[1]
    sig6 = v6[2]
    d6 = v6[3]
    ds6 = v6[4]
    for i in range(3):
        for j in range(3):
            f[i, j] = (1.0 if i == j else 0.0) + hg[3 * i + j]
    # C - I = H + H^T + H^T H, without cancellation for small strains
    for i in range(3):
        for j in range(3):
            acc = hg[3 * i + j] + hg[3 * j + i]
            for k in range(3):
                acc += hg[3 * k + i] * hg[3 * k + j]
            e2[i, j] = acc
    for i in range(3):
        for j in range(3):
            t1[i, j] = e2[i, j]
    eigh3(t1, mu, q)
    for a in range(3):
        if not (mu[a] > -1.0 and math.isfinite(mu[a])):
            return 1
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for a in range(3):
                acc += q[i, a] * q[j, a] * half_log(mu[a])
            t1[i, j] = acc
    strain_voigt(t1, eps6)
    if kind == PLASTIC:
        if _j2_update(par, nk, eps6, eth, dt, old, new, sig6, tan6, buf, sym) != 0:
            return 1
    elif kind == HILL:
        if _hill_update(par, nk, eps6, eth, dt, old, new, sig6, tan6, hw, sym) != 0:
            return 1
    elif kind == POROUS:
        if _porous_update(par, eps6, eth, dt, old, new, sig6, tan6, pw, sym) != 0:
            return 1
    else:
        for r in range(6):
            d6[r] = eps6[r] - eth[r] * (1.0 if r < 3 else 2.0)
        for r in range(6):
            acc = 0.0
            for c in range(6):
                acc += dm[r, c] * d6[c]
                tan6[r, c] = dm[r, c]
            sig6[r] = acc
    stress_tensor(sig6, t1)
    tmatmul3(q, t1, t2)
    matmul3(t2, q, tq)
    # the divided differences are symmetric in their arguments: their 6 and 10 values
    for a in range(3):
        for b in range(a, 3):
            v = log_divided(mu[a], mu[b])
            f1[a, b] = v
            f1[b, a] = v
            for c in range(b, 3):
                v = log_divided2(mu[a], mu[b], mu[c])
                f2[a, b, c] = v
                f2[a, c, b] = v
                f2[b, a, c] = v
                f2[b, c, a] = v
                f2[c, a, b] = v
                f2[c, b, a] = v
    for a in range(3):
        for b in range(3):
            t1[a, b] = 2.0 * f1[a, b] * tq[a, b]
    matmul3(q, t1, t2)
    matmult3(t2, q, s)
    matmul3(f, s, t1)
    for i in range(3):
        for j in range(3):
            pk1[3 * i + j] = t1[i, j]
    # dS is linear in dC: F dS for the six directions of dC, B = e_m (x) e_n + e_n (x) e_m
    # (e_m (x) e_m when m = n), then for each dF = e_k (x) e_l the sum of the dC_mn F dS(B)
    for bb in range(6):
        m, n = _PAIRS[bb]
        for a in range(3):  # Q^T B Q
            for b in range(3):
                v = q[m, a] * q[n, b]
                if m != n:
                    v += q[n, a] * q[m, b]
                dcq[a, b] = v
        for a in range(3):
            for b in range(3):
                t1[a, b] = f1[a, b] * dcq[a, b]
        matmul3(q, t1, t2)
        matmult3(t2, q, t1)
        strain_voigt(t1, d6)
        for r in range(6):
            acc = 0.0
            for c in range(6):
                acc += tan6[r, c] * d6[c]
            ds6[r] = acc
        stress_tensor(ds6, t1)
        tmatmul3(q, t1, t2)
        matmul3(t2, q, dtq)
        for a in range(3):
            for b in range(3):
                acc = 2.0 * f1[a, b] * dtq[a, b]
                for c in range(3):
                    acc += 2.0 * f2[a, b, c] * (dcq[a, c] * tq[c, b] + tq[a, c] * dcq[c, b])
                t1[a, b] = acc
        matmul3(q, t1, t2)
        matmult3(t2, q, t1)
        matmul3(f, t1, lw[10 + bb])
    for kl in range(9):
        k = kl // 3
        ll = kl % 3
        # dC = dF^T F + F^T dF for dF = e_k (x) e_l: dC_mn = d_ml F_kn + d_nl F_km
        for i in range(3):
            for j in range(3):
                t2[i, j] = 0.0
        for bb in range(6):
            m, n = _PAIRS[bb]
            c = (f[k, n] if m == ll else 0.0) + (f[k, m] if n == ll else 0.0)
            if c != 0.0:
                g = lw[10 + bb]
                for i in range(3):
                    for j in range(3):
                        t2[i, j] += c * g[i, j]
        for i in range(3):
            for j in range(3):
                v = t2[i, j]
                if i == k:
                    v += s[ll, j]
                tan[3 * i + j, kl] = v
    return 0


@njit(parallel=True, cache=True)
def update_points_finite(kind, mat, pmat, data, hg, dtemp, dt, old, new, pk1, tangent, guess,
                         ratio, symmetric=True):  # fmt: skip
    """First Piola-Kirchhoff stress, internal variables and tangent ``dP/dF`` (symmetrized
    with ``symmetric``) of every integration point for the displacement gradients ``hg`` at
    the end of a step: the finite strain counterpart of :func:`morphogen.homogenization.
    behaviours.update_points` (same arguments). Returns the number of points whose
    integration failed."""
    matrices, eigen, params, nk, systems, cp_par, cp_h, cp_n = data
    n = hg.shape[0]
    n_chunks = point_chunks(n)
    failed = np.zeros(n_chunks, dtype=np.int64)
    # the work arrays of the laws, a set per chunk of points: allocated before the
    # parallel loop, which holds them as plain memory (crystal.work_block)
    blocks = work_blocks(n_chunks, cp_par.shape[1])
    bufs = np.empty((n_chunks, 7, 6))
    vectors = np.empty((n_chunks, 6))
    lws = np.zeros((n_chunks, 16, 3, 3))
    v6s = np.zeros((n_chunks, 5, 6))
    tan6s = np.zeros((n_chunks, 6, 6))
    f2s = np.zeros((n_chunks, 3, 3, 3))
    hws = _hill_works(n_chunks)
    pws = _porous_works(n_chunks)
    for ch in prange(n_chunks):
        buf = bufs[ch]
        eth = vectors[ch]
        lw = lws[ch]
        v6 = v6s[ch]
        tan6 = tan6s[ch]
        f2 = f2s[ch]
        work = work_block(blocks, ch)
        hw = _hill_set(hws, ch)
        pw = _porous_set(pws, ch)
        for i in range(ch * n // n_chunks, (ch + 1) * n // n_chunks):
            m = mat[i]
            k = kind[i]
            if k == MFRONT:
                continue
            for r in range(6):
                eth[r] = dtemp * eigen[m, r] * (1.0 if r < 3 else 0.5)
            if k == CRYSTAL:
                c = pmat[i]
                fail = cp_finite(systems[m], cp_par[c], cp_h[c], cp_n[c], matrices[m], hg[i],
                                 eth, dt, old[i], new[i], pk1[i], tangent[i], guess, ratio,
                                 work)  # fmt: skip
            else:
                fail = _log_point(k, matrices[m], params[max(pmat[i], 0)], nk, hg[i], eth, dt,
                                  old[i], new[i], pk1[i], tangent[i], buf, lw, v6, tan6,
                                  f2, hw, pw, symmetric)  # fmt: skip
            if fail == 0 and symmetric:
                symmetrize(tangent[i])
            failed[ch] += fail
    return int(failed.sum())


# -- finite elements -------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def point_gradients(nodes, elements, order, u, gp, gw, hg, wts):
    """Displacement gradients (9 components) and weights of the integration points."""
    n_el, nn = elements.shape
    ng = gp.shape[0]
    for e in prange(n_el):
        x = np.empty((4, 3))
        gl = np.empty((4, 3))
        dev = np.empty((6, 3))
        grad = np.empty((nn, 3))
        vol, curved = _geometry(nodes, elements, e, x, gl, dev)
        for g in range(ng):
            ratio = _point_gradients(order, curved, gl, dev, gp[g], grad)
            p = e * ng + g
            for c in range(9):
                hg[p, c] = 0.0
            for a in range(nn):
                k = elements[e, a]
                for i in range(3):
                    ui = u[3 * k + i]
                    for j in range(3):
                        hg[p, 3 * i + j] += ui * grad[a, j]
            wts[p] = gw[g] * vol * ratio


@njit(parallel=True, cache=True)
def internal_forces(nodes, elements, order, gp, pk1, wts, by_colour, colours, forces):
    """Nodal forces ``int P . grad N`` (into ``forces``), the integral of ``P`` (9) and the
    norm of the contributions of the integration points; the elements of a colour in
    parallel (``by_colour``, ``colours``:
    :func:`~morphogen.homogenization.fe.element_colours`), the sums over all the elements
    chunk by chunk, in the same order for any number of threads."""
    nn = elements.shape[1]
    ng = gp.shape[0]
    for j in prange(forces.size):
        forces[j] = 0.0
    n_colours = colours.size - 1
    # the sums of each chunk (rows wide enough for two chunks not to share a cache line)
    slots = chunk_slots(colours)
    sums = np.zeros((slots[n_colours], 24))
    xs = np.empty((CHUNKS, 4, 3))
    gls = np.empty((CHUNKS, 4, 3))
    devs = np.empty((CHUNKS, 6, 3))
    grads = np.empty((CHUNKS, nn, 3))
    for colour in range(n_colours):
        first = colours[colour]
        count = colours[colour + 1] - first
        chunks = min(count, CHUNKS)
        for chunk in prange(chunks):
            x = xs[chunk]
            gl = gls[chunk]
            dev = devs[chunk]
            grad = grads[chunk]
            part = sums[slots[colour] + chunk]
            for q in range(first + chunk * count // chunks, first + (chunk + 1) * count // chunks):
                e = by_colour[q]
                _, curved = _geometry(nodes, elements, e, x, gl, dev)
                for g in range(ng):
                    _point_gradients(order, curved, gl, dev, gp[g], grad)
                    p = e * ng + g
                    w = wts[p]
                    s = pk1[p]
                    for r in range(9):
                        part[r] += w * s[r]
                    for b in range(nn):
                        k = elements[e, b]
                        gx, gy, gz = grad[b, 0], grad[b, 1], grad[b, 2]
                        for i in range(3):
                            fi = w * (s[3 * i] * gx + s[3 * i + 1] * gy + s[3 * i + 2] * gz)
                            forces[3 * k + i] += fi
                            part[9] += fi * fi
    total = np.zeros(9)
    norm2 = 0.0
    for k in range(sums.shape[0]):
        for r in range(9):
            total[r] += sums[k, r]
        norm2 += sums[k, 9]
    return total, math.sqrt(norm2)


@njit(parallel=True, cache=True)
def fill_tangent(nodes, elements, order, gp, tangent, wts, indptr_n, indices_n, by_colour,
                 colours, data):  # fmt: skip
    """CSR values of ``sum_e int grad N . A . grad N`` (node-level pattern), the elements
    of a colour in parallel (``by_colour``, ``colours``:
    :func:`~morphogen.homogenization.fe.element_colours`)."""
    nn = elements.shape[1]
    ng = gp.shape[0]
    for i in prange(data.size):
        data[i] = 0.0
    xs = np.empty((CHUNKS, 4, 3))
    gls = np.empty((CHUNKS, 4, 3))
    devs = np.empty((CHUNKS, 6, 3))
    grads = np.empty((CHUNKS, nn, 3))
    dbs = np.empty((CHUNKS, nn, 9, 3))
    kes = np.empty((CHUNKS, nn * 3, nn * 3))
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
            db = dbs[chunk]
            ke = kes[chunk]
            pos = places[chunk]
            for m in range(first + chunk * count // chunks, first + (chunk + 1) * count // chunks):
                e = by_colour[m]
                _, curved = _geometry(nodes, elements, e, x, gl, dev)
                ke[:, :] = 0.0
                for g in range(ng):
                    _point_gradients(order, curved, gl, dev, gp[g], grad)
                    p = e * ng + g
                    w = wts[p]
                    a9 = tangent[p]
                    for b in range(nn):
                        gx, gy, gz = grad[b, 0], grad[b, 1], grad[b, 2]
                        for r in range(9):
                            for k in range(3):
                                db[b, r, k] = (
                                    a9[r, 3 * k] * gx
                                    + a9[r, 3 * k + 1] * gy
                                    + a9[r, 3 * k + 2] * gz
                                )
                    for a in range(nn):
                        gx, gy, gz = grad[a, 0], grad[a, 1], grad[a, 2]
                        for b in range(nn):
                            for i in range(3):
                                for k in range(3):
                                    ke[3 * a + i, 3 * b + k] += w * (
                                        gx * db[b, 3 * i, k]
                                        + gy * db[b, 3 * i + 1, k]
                                        + gz * db[b, 3 * i + 2, k]
                                    )
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
                    for c in range(3):
                        start = 9 * indptr_n[na] + c * 3 * length
                        for b in range(nn):
                            q = pos[a, b]
                            for c2 in range(3):
                                data[start + q * 3 + c2] += ke[3 * a + c, 3 * b + c2]


@njit(cache=True, inline="always")
def _macro_columns(a9, am):
    """``A M``: the columns of ``A`` of the six macroscopic components."""
    for r in range(9):
        for c in range(6):
            am[r, c] = a9[r, MACRO[c]]


@njit(parallel=True, cache=True)
def coupling(nodes, elements, order, gp, tangent, wts, by_colour, colours, out, kee):
    """``int grad N . A M`` (nodal forces of the six unit macroscopic strains, into
    ``out``, n_dofs x 6) and ``M^T (int A) M`` (into ``kee``); the elements of a colour in
    parallel, as :func:`internal_forces`."""
    nn = elements.shape[1]
    ng = gp.shape[0]
    for j in prange(out.shape[0]):
        for c in range(6):
            out[j, c] = 0.0
    n_colours = colours.size - 1
    slots = chunk_slots(colours)
    sums = np.zeros((slots[n_colours], 64))  # (54 values: the rows of two chunks apart)
    ams = np.empty((CHUNKS, 9, 6))
    xs = np.empty((CHUNKS, 4, 3))
    gls = np.empty((CHUNKS, 4, 3))
    devs = np.empty((CHUNKS, 6, 3))
    grads = np.empty((CHUNKS, nn, 3))
    for colour in range(n_colours):
        first = colours[colour]
        count = colours[colour + 1] - first
        chunks = min(count, CHUNKS)
        for chunk in prange(chunks):
            am = ams[chunk]
            x = xs[chunk]
            gl = gls[chunk]
            dev = devs[chunk]
            grad = grads[chunk]
            part = sums[slots[colour] + chunk]
            for q in range(first + chunk * count // chunks, first + (chunk + 1) * count // chunks):
                e = by_colour[q]
                _, curved = _geometry(nodes, elements, e, x, gl, dev)
                for g in range(ng):
                    _point_gradients(order, curved, gl, dev, gp[g], grad)
                    p = e * ng + g
                    w = wts[p]
                    _macro_columns(tangent[p], am)
                    for r in range(9):
                        for c in range(6):
                            part[6 * r + c] += w * am[r, c]
                    for b in range(nn):
                        k = elements[e, b]
                        gx, gy, gz = grad[b, 0], grad[b, 1], grad[b, 2]
                        for i in range(3):
                            for c in range(6):
                                out[3 * k + i, c] += w * (
                                    gx * am[3 * i, c]
                                    + gy * am[3 * i + 1, c]
                                    + gz * am[3 * i + 2, c]
                                )
    total = np.zeros((9, 6))
    for k in range(sums.shape[0]):
        for r in range(9):
            for c in range(6):
                total[r, c] += sums[k, 6 * r + c]
    for r in range(6):
        for c in range(6):
            kee[r, c] = total[MACRO[r], c]


@njit(parallel=True, cache=True)
def group_sums(group, n_groups, wts, pk1, hg, p):
    """Integrals over each group of the macroscopic components of ``P`` (6) and of the
    displacement gradient (6), of the cumulated plastic strain ``p``, the reference volume,
    the Kirchhoff stress ``P F^T`` (6) and the volume change ``det F``."""
    n = wts.shape[0]
    n_chunks = min(max(n, 1), 256)
    partial = np.zeros((n_chunks, n_groups, 21))
    for ch in prange(n_chunks):
        f = np.empty((3, 3))
        for i in range(ch * n // n_chunks, (ch + 1) * n // n_chunks):
            acc = partial[ch, group[i]]
            w = wts[i]
            s = pk1[i]
            h = hg[i]
            for r in range(6):
                acc[r] += w * s[MACRO[r]]
                acc[6 + r] += w * h[MACRO[r]]
            acc[12] += w * p[i]
            acc[13] += w
            for a in range(3):
                for b in range(3):
                    f[a, b] = (1.0 if a == b else 0.0) + h[3 * a + b]
            for r in range(6):
                a = (0, 1, 2, 1, 2, 0)[r]
                b = (0, 1, 2, 2, 0, 1)[r]
                acc[14 + r] += w * (
                    s[3 * a] * f[b, 0] + s[3 * a + 1] * f[b, 1] + s[3 * a + 2] * f[b, 2]
                )
            det = (
                f[0, 0] * (f[1, 1] * f[2, 2] - f[1, 2] * f[2, 1])
                - f[0, 1] * (f[1, 0] * f[2, 2] - f[1, 2] * f[2, 0])
                + f[0, 2] * (f[1, 0] * f[2, 1] - f[1, 1] * f[2, 0])
            )
            acc[20] += w * det
    return partial.sum(axis=0)


# -- outputs -------------------------------------------------------------------------------


def macro_part(t9) -> np.ndarray:
    """The macroscopic components (11, 22, 33, 23, 13, 12) of tensors of 9 components."""
    return np.asarray(t9, dtype=float)[..., list(MACRO)]


def gradient_of(e) -> np.ndarray:
    """Displacement gradient (9) of the macroscopic components ``e`` (..., 6): upper
    triangular."""
    e = np.asarray(e, dtype=float)
    out = np.zeros((*e.shape[:-1], 9))
    out[..., list(MACRO)] = e
    return out


def _voigt(t) -> np.ndarray:
    """Voigt components (tensor shears) of symmetric 3 x 3 tensors (..., 3, 3)."""
    return np.stack(
        [t[..., 0, 0], t[..., 1, 1], t[..., 2, 2], t[..., 1, 2], t[..., 2, 0], t[..., 0, 1]],
        axis=-1,
    )


def cauchy(pk1, hg) -> np.ndarray:
    """Cauchy stress (Voigt) of first Piola-Kirchhoff stresses and displacement gradients
    (..., 9): ``P F^T / det F``."""
    p = np.asarray(pk1, dtype=float).reshape(*np.shape(pk1)[:-1], 3, 3)
    f = np.eye(3) + np.asarray(hg, dtype=float).reshape(*np.shape(hg)[:-1], 3, 3)
    tau = p @ np.swapaxes(f, -1, -2)
    return _voigt(0.5 * (tau + np.swapaxes(tau, -1, -2))) / np.linalg.det(f)[..., None]


def log_strain(hg) -> np.ndarray:
    """Logarithmic (Hencky) strain ``1/2 ln(F^T F)`` (Voigt, engineering shears) of
    displacement gradients (..., 9)."""
    h = np.asarray(hg, dtype=float).reshape(*np.shape(hg)[:-1], 3, 3)
    c_minus = h + np.swapaxes(h, -1, -2) + np.swapaxes(h, -1, -2) @ h
    w, v = np.linalg.eigh(c_minus)
    eps = (v * (0.5 * np.log1p(w))[..., None, :]) @ np.swapaxes(v, -1, -2)
    out = _voigt(eps)
    out[..., 3:] *= 2.0
    return out


def lattice_rotations(hg, fp_inv) -> np.ndarray:
    """Rotations ``R_e`` of the lattice (polar decomposition of ``F_e = F F_p^-1``) of
    displacement gradients and ``F_p^-1`` (..., 9)."""
    f = np.eye(3) + np.asarray(hg, dtype=float).reshape(*np.shape(hg)[:-1], 3, 3)
    fe = f @ np.asarray(fp_inv, dtype=float).reshape(*np.shape(fp_inv)[:-1], 3, 3)
    u, _, vt = np.linalg.svd(fe)
    return u @ vt
