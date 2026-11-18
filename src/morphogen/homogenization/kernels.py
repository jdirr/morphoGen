"""Compiled kernels of the trilinear voxel elements of the FFT solver.

The grid has ``(nx, ny, nz)`` voxels (elements), numbered ``e = (i * ny + j) * nz + k``.
Nodes are numbered the same way on the ``(nx + 1, ny + 1, nz + 1)`` node grid, or on
the ``(nx, ny, nz)`` grid when the mesh is periodic (opposite faces identified).
``dof[node * nd + c]`` is the index of a degree of freedom, or -1 for an unused one.

Right-hand sides loop over nodes (each thread writes only the rows of its nodes, so no
atomics are needed); element sums loop over chunks of elements.

The kernels of the loading paths (:mod:`morphogen.homogenization.fft_nonlinear`) work on
the 8 Gauss points of the voxels that carry material: point ``8 t + g`` is Gauss point
``g`` of voxel ``elem[t]``, and ``slot[e]`` is the position ``t`` of voxel ``e`` (-1 for a
void). Their measures of strain are the engineering strains (6, Voigt order) or, in
finite strains, the displacement gradients (9, row by row).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from morphogen.mesh.voxel import HEX_CORNERS

CORNERS = np.ascontiguousarray(HEX_CORNERS, dtype=np.int64)
# CORNER_INDEX[a, b, c] = index of the corner with offsets (a, b, c)
CORNER_INDEX = np.zeros((2, 2, 2), dtype=np.int64)
for _q, (_a, _b, _c) in enumerate(HEX_CORNERS):
    CORNER_INDEX[_a, _b, _c] = _q


@njit(cache=True, inline="always")
def _node_grid(nx, ny, nz, periodic):
    if periodic:
        return nx, ny, nz
    return nx + 1, ny + 1, nz + 1


@njit(cache=True)
def _elements_around(i, j, k, nx, ny, nz, periodic, active, out_elem, out_local):
    """Active elements containing node (i, j, k) and the local corner of the node."""
    count = 0
    for a in range(2):
        ei = i - a
        if periodic:
            ei %= nx
        elif ei < 0 or ei >= nx:
            continue
        for b in range(2):
            ej = j - b
            if periodic:
                ej %= ny
            elif ej < 0 or ej >= ny:
                continue
            for c in range(2):
                ek = k - c
                if periodic:
                    ek %= nz
                elif ek < 0 or ek >= nz:
                    continue
                e = (ei * ny + ej) * nz + ek
                if active[e]:
                    out_elem[count] = e
                    out_local[count] = CORNER_INDEX[a, b, c]
                    count += 1
    return count


@njit(parallel=True, cache=True)
def element_load(nx, ny, nz, periodic, mat, dof, nd, loads, n_free):
    """Right-hand side -sum_e loads[mat_e] restricted to the free dofs (``loads[m]``: the
    element load vector of material ``m``; ``mat`` -1: no stiffness)."""
    gx, gy, gz = _node_grid(nx, ny, nz, periodic)
    active = mat >= 0
    f = np.zeros(n_free, dtype=np.float64)
    for n in prange(gx * gy * gz):
        has_free = False
        for c in range(nd):
            if dof[n * nd + c] >= 0:
                has_free = True
        if not has_free:
            continue
        k = n % gz
        j = (n // gz) % gy
        i = n // (gy * gz)
        elem = np.empty(8, dtype=np.int64)
        local = np.empty(8, dtype=np.int64)
        n_el = _elements_around(i, j, k, nx, ny, nz, periodic, active, elem, local)
        for t in range(n_el):
            e = elem[t]
            ln = local[t]
            for c in range(nd):
                r = dof[n * nd + c]
                if r >= 0:
                    f[r] -= loads[mat[e], ln * nd + c]
    return f


@njit(cache=True, inline="always")
def _element_nodes(e, nx, ny, nz, periodic, out):
    k = e % nz
    j = (e // nz) % ny
    i = e // (ny * nz)
    gx, gy, gz = _node_grid(nx, ny, nz, periodic)
    for q in range(8):
        ii = i + CORNERS[q, 0]
        jj = j + CORNERS[q, 1]
        kk = k + CORNERS[q, 2]
        if periodic:
            ii %= gx
            jj %= gy
            kk %= gz
        out[q] = (ii * gy + jj) * gz + kk


N_ELASTIC = 27  # eps(6) sig(6) energy(1) eps^2(6) sig^2(6) energy^2(1) volume(1)
N_THERMAL = 15  # grad(3) flux(3) dissipation(1) grad^2(3) flux^2(3) dissipation^2(1) volume(1)


@njit(cache=True, inline="always")
def _stress(d, eps, sig):
    for r in range(6):
        s = 0.0
        for c in range(6):
            s += d[r, c] * eps[c]
        sig[r] = s


@njit(parallel=True, cache=True)
def elastic_sums(nx, ny, nz, periodic, mat, stat, n_stat, dmat, eig, dt, w, u_loc, b_gauss, wt):
    """Integrals over each phase (``stat``) of strains, stresses, energy and squares.

    ``w`` holds the nodal fluctuation (3 per node), ``u_loc`` the affine part at the
    corners of an element (identical for all elements), ``dmat[mat_e]`` the elasticity
    matrix of element ``e`` (-1: void) and ``eig[mat_e]`` its thermal strain for a unit
    change of temperature (``dt``: the change). Strains are tensor components, stresses
    ``D (eps - dt eig)``.
    """
    n_el = nx * ny * nz
    n_chunks = min(n_el, 256)
    partial = np.zeros((n_chunks, n_stat, N_ELASTIC))
    for ch in prange(n_chunks):
        nodes = np.empty(8, dtype=np.int64)
        ue = np.empty(24)
        eps = np.empty(6)
        mech = np.empty(6)
        sig = np.empty(6)
        start = ch * n_el // n_chunks
        stop = (ch + 1) * n_el // n_chunks
        for e in range(start, stop):
            if mat[e] < 0:
                continue
            _element_nodes(e, nx, ny, nz, periodic, nodes)
            for q in range(8):
                for c in range(3):
                    ue[3 * q + c] = u_loc[3 * q + c] + w[3 * nodes[q] + c]
            d = dmat[mat[e]]
            acc = partial[ch, stat[e]]
            for gp in range(8):
                for r in range(6):
                    s = 0.0
                    for col in range(24):
                        s += b_gauss[gp, r, col] * ue[col]
                    eps[r] = s
                    mech[r] = s - dt * eig[mat[e], r]
                _stress(d, mech, sig)
                dens = 0.0
                for r in range(6):
                    dens += sig[r] * mech[r]
                for r in range(6):
                    et = eps[r] if r < 3 else 0.5 * eps[r]
                    acc[r] += wt * et
                    acc[6 + r] += wt * sig[r]
                    acc[13 + r] += wt * et * et
                    acc[19 + r] += wt * sig[r] * sig[r]
                acc[12] += wt * dens
                acc[25] += wt * dens * dens
                acc[26] += wt
    return partial.sum(axis=0)


@njit(parallel=True, cache=True)
def thermal_sums(nx, ny, nz, periodic, mat, stat, n_stat, kmat, t, t_loc, g_gauss, wt):
    """Integrals over each phase of gradients, fluxes, dissipation and squares
    (``kmat[mat_e]``: conductivity tensor of element ``e``)."""
    n_el = nx * ny * nz
    n_chunks = min(n_el, 256)
    partial = np.zeros((n_chunks, n_stat, N_THERMAL))
    for ch in prange(n_chunks):
        nodes = np.empty(8, dtype=np.int64)
        te = np.empty(8)
        grad = np.empty(3)
        start = ch * n_el // n_chunks
        stop = (ch + 1) * n_el // n_chunks
        for e in range(start, stop):
            if mat[e] < 0:
                continue
            _element_nodes(e, nx, ny, nz, periodic, nodes)
            for q in range(8):
                te[q] = t_loc[q] + t[nodes[q]]
            acc = partial[ch, stat[e]]
            kk = kmat[mat[e]]
            for gp in range(8):
                for d in range(3):
                    s = 0.0
                    for q in range(8):
                        s += g_gauss[gp, d, q] * te[q]
                    grad[d] = s
                dens = 0.0
                for d in range(3):
                    flux = -(kk[d, 0] * grad[0] + kk[d, 1] * grad[1] + kk[d, 2] * grad[2])
                    dens -= flux * grad[d]
                    acc[d] += wt * grad[d]
                    acc[3 + d] += wt * flux
                    acc[7 + d] += wt * grad[d] * grad[d]
                    acc[10 + d] += wt * flux * flux
                acc[6] += wt * dens
                acc[13] += wt * dens * dens
                acc[14] += wt
    return partial.sum(axis=0)


@njit(parallel=True, cache=True)
def elastic_fields(nx, ny, nz, periodic, mat, dmat, eig, dt, w, u_loc, b_mean, b_gauss):
    """Element means of the strain (tensor components), stress and energy density
    ``sigma : (eps - dt eig)`` (NaN where inactive)."""
    n_el = nx * ny * nz
    strain = np.full((n_el, 6), np.nan)
    stress = np.full((n_el, 6), np.nan)
    energy = np.full(n_el, np.nan)
    for e in prange(n_el):
        if mat[e] < 0:
            continue
        nodes = np.empty(8, dtype=np.int64)
        _element_nodes(e, nx, ny, nz, periodic, nodes)
        ue = np.empty(24)
        for q in range(8):
            for c in range(3):
                ue[3 * q + c] = u_loc[3 * q + c] + w[3 * nodes[q] + c]
        d = dmat[mat[e]]
        eps = np.zeros(6)
        mech = np.empty(6)
        sig = np.empty(6)
        for r in range(6):
            s = 0.0
            for col in range(24):
                s += b_mean[r, col] * ue[col]
            eps[r] = s
            mech[r] = s - dt * eig[mat[e], r]
        _stress(d, mech, sig)
        for r in range(6):
            stress[e, r] = sig[r]
            strain[e, r] = eps[r] if r < 3 else 0.5 * eps[r]
        dens = 0.0
        for gp in range(8):
            for r in range(6):
                s = 0.0
                for col in range(24):
                    s += b_gauss[gp, r, col] * ue[col]
                mech[r] = s - dt * eig[mat[e], r]
            _stress(d, mech, sig)
            for r in range(6):
                dens += sig[r] * mech[r]
        energy[e] = dens / 8.0
    return strain, stress, energy


@njit(parallel=True, cache=True)
def thermal_fields(nx, ny, nz, periodic, mat, kmat, t, t_loc, g_mean, g_gauss):
    """Element means of the temperature gradient, flux and dissipation ``grad T . K grad
    T`` (NaN where inactive)."""
    n_el = nx * ny * nz
    grad = np.full((n_el, 3), np.nan)
    flux = np.full((n_el, 3), np.nan)
    dissipation = np.full(n_el, np.nan)
    for e in prange(n_el):
        if mat[e] < 0:
            continue
        nodes = np.empty(8, dtype=np.int64)
        _element_nodes(e, nx, ny, nz, periodic, nodes)
        te = np.empty(8)
        for q in range(8):
            te[q] = t_loc[q] + t[nodes[q]]
        kk = kmat[mat[e]]
        g = np.empty(3)
        for d in range(3):
            s = 0.0
            for q in range(8):
                s += g_mean[d, q] * te[q]
            g[d] = s
            grad[e, d] = s
        for d in range(3):
            flux[e, d] = -(kk[d, 0] * g[0] + kk[d, 1] * g[1] + kk[d, 2] * g[2])
        dens = 0.0
        for gp in range(8):
            for d in range(3):
                s = 0.0
                for q in range(8):
                    s += g_gauss[gp, d, q] * te[q]
                g[d] = s
            for d in range(3):
                dens += g[d] * (kk[d, 0] * g[0] + kk[d, 1] * g[1] + kk[d, 2] * g[2])
        dissipation[e] = dens / 8.0
    return grad, flux, dissipation


# -- loading paths: the Gauss points of the voxels -----------------------------------------
#
# ``bbar``: the volumetric part of the measures is averaged over each voxel, which keeps
# the trilinear elements from locking under isochoric (plastic) flow. In small strains
# (B-bar, Hughes 1980) the trace of the strain at each point is replaced by its mean over
# the voxel; in finite strains (F-bar, de Souza Neto et al. 1996) the gradient of each
# point is scaled, ``F_bar = (J_bar / J)^(1/3) F``, to the geometric mean ``J_bar`` of the
# Jacobians of the voxel. The forces are the derivatives of the energy of these measures,
# so that the tangent stays symmetric: the stresses of the points are replaced by their
# conjugates (:func:`voxel_effective`), ``s + p_bar 1`` in small strains (the deviator of
# the point, the mean pressure of the voxel) and ``alpha P + (tau_bar - tau) F^-T`` in
# finite strains (``alpha = (J_bar / J)^(1/3)``, ``tau = alpha P : F / 3``).


@njit(cache=True, inline="always")
def _gradient(ue, g_gauss, g, h):
    """Displacement gradient (3 x 3) at Gauss point ``g`` of a voxel whose corners move
    by ``ue`` (8 x 3), into ``h``."""
    for i in range(3):
        for j in range(3):
            h[i, j] = 0.0
    for a in range(8):
        gx = g_gauss[g, 0, a]
        gy = g_gauss[g, 1, a]
        gz = g_gauss[g, 2, a]
        for i in range(3):
            u = ue[a, i]
            h[i, 0] += u * gx
            h[i, 1] += u * gy
            h[i, 2] += u * gz


@njit(cache=True, inline="always")
def _det3(f):
    return (
        f[0, 0] * (f[1, 1] * f[2, 2] - f[1, 2] * f[2, 1])
        - f[0, 1] * (f[1, 0] * f[2, 2] - f[1, 2] * f[2, 0])
        + f[0, 2] * (f[1, 0] * f[2, 1] - f[1, 1] * f[2, 0])
    )


@njit(cache=True, inline="always")
def _inverse_transpose(f, det, out):
    """``F^-T`` (the cofactor matrix over the determinant)."""
    inv = 1.0 / det
    out[0, 0] = (f[1, 1] * f[2, 2] - f[1, 2] * f[2, 1]) * inv
    out[0, 1] = (f[1, 2] * f[2, 0] - f[1, 0] * f[2, 2]) * inv
    out[0, 2] = (f[1, 0] * f[2, 1] - f[1, 1] * f[2, 0]) * inv
    out[1, 0] = (f[0, 2] * f[2, 1] - f[0, 1] * f[2, 2]) * inv
    out[1, 1] = (f[0, 0] * f[2, 2] - f[0, 2] * f[2, 0]) * inv
    out[1, 2] = (f[0, 1] * f[2, 0] - f[0, 0] * f[2, 1]) * inv
    out[2, 0] = (f[0, 1] * f[1, 2] - f[0, 2] * f[1, 1]) * inv
    out[2, 1] = (f[0, 2] * f[1, 0] - f[0, 0] * f[1, 2]) * inv
    out[2, 2] = (f[0, 0] * f[1, 1] - f[0, 1] * f[1, 0]) * inv


@njit(cache=True, inline="always")
def _engineering(h, out):
    """Engineering strains (Voigt order) of the symmetric part of a gradient ``h``."""
    out[0] = h[0, 0]
    out[1] = h[1, 1]
    out[2] = h[2, 2]
    out[3] = h[1, 2] + h[2, 1]
    out[4] = h[2, 0] + h[0, 2]
    out[5] = h[0, 1] + h[1, 0]


@njit(parallel=True, cache=True)
def voxel_measures(nx, ny, nz, elem, w, macro, g_gauss, finite, bbar, out, base):
    """Measures of strain at the Gauss points of the voxels ``elem`` of the periodic grid
    (into ``out``, points x 6 or 9): the gradient of the nodal fluctuation ``w`` (3 per
    node) plus the macroscopic gradient ``macro`` (9, row by row; symmetric in small
    strains). With ``bbar``, the measures of B-bar or F-bar (see above), and in finite
    strains the gradients themselves into ``base``. A voxel turned inside out gets NaN
    measures (its behaviours then fail)."""
    n_el = elem.size
    n_chunks = min(max(n_el, 1), 512)
    for ch in prange(n_chunks):
        nodes = np.empty(8, dtype=np.int64)
        ue = np.empty((8, 3))
        hs = np.empty((8, 3, 3))
        m = np.empty(6)
        log_j = np.empty(8)
        for t in range(ch * n_el // n_chunks, (ch + 1) * n_el // n_chunks):
            _element_nodes(elem[t], nx, ny, nz, True, nodes)
            for a in range(8):
                for c in range(3):
                    ue[a, c] = w[3 * nodes[a] + c]
            for g in range(8):
                _gradient(ue, g_gauss, g, hs[g])
                for i in range(3):
                    for j in range(3):
                        hs[g, i, j] += macro[3 * i + j]
            if not finite:
                mean = 0.0
                for g in range(8):
                    mean += hs[g, 0, 0] + hs[g, 1, 1] + hs[g, 2, 2]
                mean /= 8.0
                for g in range(8):
                    _engineering(hs[g], m)
                    if bbar:
                        shift = (mean - m[0] - m[1] - m[2]) / 3.0
                        for r in range(3):
                            m[r] += shift
                    p = 8 * t + g
                    for r in range(6):
                        out[p, r] = m[r]
                continue
            valid = True
            mean = 0.0
            for g in range(8):
                for i in range(3):
                    hs[g, i, i] += 1.0  # F
                det = _det3(hs[g])
                if not det > 0.0:
                    valid = False
                    break
                log_j[g] = math.log(det)
                mean += log_j[g] / 8.0
            for g in range(8):
                p = 8 * t + g
                if not valid:
                    for r in range(9):
                        out[p, r] = np.nan
                        if bbar:
                            base[p, r] = np.nan
                    continue
                alpha = math.exp((mean - log_j[g]) / 3.0) if bbar else 1.0
                for i in range(3):
                    for j in range(3):
                        f = hs[g, i, j]
                        out[p, 3 * i + j] = alpha * f - (1.0 if i == j else 0.0)
                        if bbar:
                            base[p, 3 * i + j] = f - (1.0 if i == j else 0.0)


@njit(parallel=True, cache=True)
def voxel_effective(n_el, finite, s, base):
    """The stresses of the points of ``n_el`` voxels replaced, in place, by those
    conjugate to the measures of B-bar or F-bar (see above): ``s`` holds the stresses of
    the behaviours (points x 6, or x 9 first Piola-Kirchhoff stresses), ``base`` the
    gradients of the points in finite strains."""
    n_chunks = min(max(n_el, 1), 512)
    for ch in prange(n_chunks):
        f = np.empty((3, 3))
        fit = np.empty((3, 3))
        alpha = np.empty(8)
        tau = np.empty(8)
        for t in range(ch * n_el // n_chunks, (ch + 1) * n_el // n_chunks):
            if not finite:
                mean = 0.0
                for g in range(8):
                    p = 8 * t + g
                    mean += (s[p, 0] + s[p, 1] + s[p, 2]) / 24.0
                for g in range(8):
                    p = 8 * t + g
                    shift = mean - (s[p, 0] + s[p, 1] + s[p, 2]) / 3.0
                    for r in range(3):
                        s[p, r] += shift
                continue
            mean_log = 0.0
            for g in range(8):
                p = 8 * t + g
                for i in range(3):
                    for j in range(3):
                        f[i, j] = base[p, 3 * i + j] + (1.0 if i == j else 0.0)
                alpha[g] = math.log(_det3(f))
                mean_log += alpha[g] / 8.0
            mean_tau = 0.0
            for g in range(8):
                p = 8 * t + g
                alpha[g] = math.exp((mean_log - alpha[g]) / 3.0)
                pf = 0.0
                for i in range(3):
                    for j in range(3):
                        pf += s[p, 3 * i + j] * (base[p, 3 * i + j] + (1.0 if i == j else 0.0))
                tau[g] = alpha[g] * pf / 3.0
                mean_tau += tau[g] / 8.0
            for g in range(8):
                p = 8 * t + g
                for i in range(3):
                    for j in range(3):
                        f[i, j] = base[p, 3 * i + j] + (1.0 if i == j else 0.0)
                _inverse_transpose(f, _det3(f), fit)
                for i in range(3):
                    for j in range(3):
                        k = 3 * i + j
                        s[p, k] = alpha[g] * s[p, k] + (mean_tau - tau[g]) * fit[i, j]


@njit(parallel=True, cache=True)
def voxel_tangent_stress(nx, ny, nz, elem, v, direction, tangent, s, base, g_gauss, finite,
                         bbar, out):  # fmt: skip
    """Increments of the stresses of the points (of their conjugates with ``bbar``) for
    the nodal displacements ``v`` (3 per node) plus the uniform gradient ``direction`` (9,
    row by row; symmetric in small strains), with the tangents ``tangent`` of the points
    (points x n x n), into ``out`` (points x n). In finite strains with ``bbar``, ``s``
    holds the first Piola-Kirchhoff stresses of the behaviours and ``base`` the gradients
    of the points (the derivatives of the F-bar measures depend on them)."""
    n_el = elem.size
    n_chunks = min(max(n_el, 1), 512)
    for ch in prange(n_chunks):
        nodes = np.empty(8, dtype=np.int64)
        ue = np.empty((8, 3))
        dh = np.empty((8, 3, 3))
        de = np.empty(9)
        ds = np.empty((8, 9))
        f = np.empty((3, 3))
        fit = np.empty((3, 3))
        alpha = np.empty(8)
        dalpha = np.empty(8)
        tau = np.empty(8)
        dtau = np.empty(8)
        for t in range(ch * n_el // n_chunks, (ch + 1) * n_el // n_chunks):
            _element_nodes(elem[t], nx, ny, nz, True, nodes)
            for a in range(8):
                for c in range(3):
                    ue[a, c] = v[3 * nodes[a] + c]
            for g in range(8):
                _gradient(ue, g_gauss, g, dh[g])
                for i in range(3):
                    for j in range(3):
                        dh[g, i, j] += direction[3 * i + j]
            if not finite:
                mean = 0.0
                for g in range(8):
                    mean += (dh[g, 0, 0] + dh[g, 1, 1] + dh[g, 2, 2]) / 8.0
                mean_p = 0.0
                for g in range(8):
                    _engineering(dh[g], de)
                    if bbar:
                        shift = (mean - de[0] - de[1] - de[2]) / 3.0
                        for r in range(3):
                            de[r] += shift
                    p = 8 * t + g
                    for r in range(6):
                        acc = 0.0
                        for c in range(6):
                            acc += tangent[p, r, c] * de[c]
                        ds[g, r] = acc
                    mean_p += (ds[g, 0] + ds[g, 1] + ds[g, 2]) / 24.0
                for g in range(8):
                    p = 8 * t + g
                    shift = mean_p - (ds[g, 0] + ds[g, 1] + ds[g, 2]) / 3.0 if bbar else 0.0
                    for r in range(6):
                        out[p, r] = ds[g, r] + (shift if r < 3 else 0.0)
                continue
            if not bbar:
                for g in range(8):
                    p = 8 * t + g
                    for r in range(9):
                        acc = 0.0
                        for c in range(9):
                            acc += tangent[p, r, c] * dh[g, c // 3, c % 3]
                        out[p, r] = acc
                continue
            # F-bar: F_bar = alpha F, alpha = (J_bar / J)^(1/3)
            mean_log = 0.0
            mean_c = 0.0
            for g in range(8):
                p = 8 * t + g
                for i in range(3):
                    for j in range(3):
                        f[i, j] = base[p, 3 * i + j] + (1.0 if i == j else 0.0)
                det = _det3(f)
                _inverse_transpose(f, det, fit)
                alpha[g] = math.log(det)
                mean_log += alpha[g] / 8.0
                c = 0.0  # d ln J = F^-T : dF
                for i in range(3):
                    for j in range(3):
                        c += fit[i, j] * dh[g, i, j]
                dalpha[g] = c
                mean_c += c / 8.0
            mean_tau = 0.0
            mean_dtau = 0.0
            for g in range(8):
                p = 8 * t + g
                a = math.exp((mean_log - alpha[g]) / 3.0)
                alpha[g] = a
                da = a * (mean_c - dalpha[g]) / 3.0
                dalpha[g] = da
                # dF_bar = alpha dF + dalpha F, dP = A : dF_bar
                for i in range(3):
                    for j in range(3):
                        fij = base[p, 3 * i + j] + (1.0 if i == j else 0.0)
                        de[3 * i + j] = a * dh[g, i, j] + da * fij
                pf = 0.0
                dpf = 0.0
                pdf = 0.0
                for r in range(9):
                    acc = 0.0
                    for c in range(9):
                        acc += tangent[p, r, c] * de[c]
                    ds[g, r] = acc
                    fr = base[p, r] + (1.0 if r % 4 == 0 else 0.0)
                    pf += s[p, r] * fr
                    dpf += acc * fr
                    pdf += s[p, r] * dh[g, r // 3, r % 3]
                tau[g] = a * pf / 3.0
                dtau[g] = (da * pf + a * (dpf + pdf)) / 3.0
                mean_tau += tau[g] / 8.0
                mean_dtau += dtau[g] / 8.0
            for g in range(8):
                p = 8 * t + g
                for i in range(3):
                    for j in range(3):
                        f[i, j] = base[p, 3 * i + j] + (1.0 if i == j else 0.0)
                _inverse_transpose(f, _det3(f), fit)
                a = alpha[g]
                da = dalpha[g]
                ct = mean_tau - tau[g]
                cdt = mean_dtau - dtau[g]
                for i in range(3):
                    for j in range(3):
                        # d(F^-T)_ij = -(F^-T dF^T F^-T)_ij
                        dfit = 0.0
                        for k in range(3):
                            for m in range(3):
                                dfit -= fit[i, k] * dh[g, m, k] * fit[m, j]
                        r = 3 * i + j
                        out[p, r] = da * s[p, r] + a * ds[g, r] + cdt * fit[i, j] + ct * dfit


@njit(parallel=True, cache=True)
def voxel_forces(nx, ny, nz, slot, s, weight, g_gauss, finite, out):
    """Nodal forces of the stresses ``s`` at the Gauss points (``int B^T s``; in finite
    strains ``int P . grad N`` of the first Piola-Kirchhoff stresses), into ``out`` (3 per
    node of the periodic grid; ``weight``: the weight of a Gauss point). Loops over nodes,
    so no atomics are needed. Returns the norm of the contributions of the points before
    they cancel."""
    n = nx * ny * nz
    norm2 = 0.0
    for node in prange(n):
        k = node % nz
        j = (node // nz) % ny
        i = node // (ny * nz)
        fx = 0.0
        fy = 0.0
        fz = 0.0
        sq = 0.0
        for a in range(2):
            ei = i - a if i - a >= 0 else nx - 1
            for b in range(2):
                ej = j - b if j - b >= 0 else ny - 1
                for c in range(2):
                    ek = k - c if k - c >= 0 else nz - 1
                    t = slot[(ei * ny + ej) * nz + ek]
                    if t < 0:
                        continue
                    ln = CORNER_INDEX[a, b, c]
                    for g in range(8):
                        p = 8 * t + g
                        gx = weight * g_gauss[g, 0, ln]
                        gy = weight * g_gauss[g, 1, ln]
                        gz = weight * g_gauss[g, 2, ln]
                        if finite:
                            cx = s[p, 0] * gx + s[p, 1] * gy + s[p, 2] * gz
                            cy = s[p, 3] * gx + s[p, 4] * gy + s[p, 5] * gz
                            cz = s[p, 6] * gx + s[p, 7] * gy + s[p, 8] * gz
                        else:
                            cx = gx * s[p, 0] + gz * s[p, 4] + gy * s[p, 5]
                            cy = gy * s[p, 1] + gz * s[p, 3] + gx * s[p, 5]
                            cz = gz * s[p, 2] + gy * s[p, 3] + gx * s[p, 4]
                        fx += cx
                        fy += cy
                        fz += cz
                        sq += cx * cx + cy * cy + cz * cz
        out[3 * node] = fx
        out[3 * node + 1] = fy
        out[3 * node + 2] = fz
        norm2 += sq
    return math.sqrt(norm2)


@njit(parallel=True, cache=True)
def voxel_centres(nx, ny, nz, w, out):
    """Mean of the nodal fluctuation ``w`` (3 per node) over the corners of every voxel:
    its value at the centre (voxels x 3)."""
    for e in prange(nx * ny * nz):
        nodes = np.empty(8, dtype=np.int64)
        _element_nodes(e, nx, ny, nz, True, nodes)
        for c in range(3):
            acc = 0.0
            for a in range(8):
                acc += w[3 * nodes[a] + c]
            out[e, c] = acc / 8.0


# -- the Fourier preconditioner ------------------------------------------------------------


@njit(parallel=True, cache=True)
def fourier_inverse(nx, ny, nz, blocks, out):
    """Inverse of the symbol of a homogeneous operator at the frequencies of the real FFT
    of the periodic grid, into ``out`` (nx x ny x (nz // 2 + 1) x nd x nd, complex): the
    stencil ``blocks`` (3 x 3 x 3 x nd x nd, node offsets -1, 0, 1) summed with its
    phases, then inverted (nd = 1 or 3, by cofactors); zero at the zero frequency (the
    fluctuation has a zero mean)."""
    nzh = nz // 2 + 1
    nd = blocks.shape[3]
    n_freq = nx * ny * nzh
    n_chunks = min(max(n_freq, 1), 512)
    for ch in prange(n_chunks):
        s = np.empty((3, 3), dtype=np.complex128)
        px = np.empty(3, dtype=np.complex128)
        py = np.empty(3, dtype=np.complex128)
        pz = np.empty(3, dtype=np.complex128)
        for f in range(ch * n_freq // n_chunks, (ch + 1) * n_freq // n_chunks):
            fz = f % nzh
            fy = (f // nzh) % ny
            fx = f // (ny * nzh)
            if fx == 0 and fy == 0 and fz == 0:
                for i in range(nd):
                    for j in range(nd):
                        out[fx, fy, fz, i, j] = 0.0
                continue
            tx = 2.0 * math.pi * fx / nx
            ty = 2.0 * math.pi * fy / ny
            tz = 2.0 * math.pi * fz / nz
            for a in range(3):
                px[a] = complex(math.cos((a - 1) * tx), math.sin((a - 1) * tx))
                py[a] = complex(math.cos((a - 1) * ty), math.sin((a - 1) * ty))
                pz[a] = complex(math.cos((a - 1) * tz), math.sin((a - 1) * tz))
            for i in range(nd):
                for j in range(nd):
                    s[i, j] = 0.0
            for a in range(3):
                for b in range(3):
                    pab = px[a] * py[b]
                    for c in range(3):
                        phase = pab * pz[c]
                        for i in range(nd):
                            for j in range(nd):
                                s[i, j] += phase * blocks[a, b, c, i, j]
            if nd == 1:
                out[fx, fy, fz, 0, 0] = 1.0 / s[0, 0]
                continue
            c00 = s[1, 1] * s[2, 2] - s[1, 2] * s[2, 1]
            c01 = s[1, 2] * s[2, 0] - s[1, 0] * s[2, 2]
            c02 = s[1, 0] * s[2, 1] - s[1, 1] * s[2, 0]
            inv = 1.0 / (s[0, 0] * c00 + s[0, 1] * c01 + s[0, 2] * c02)
            o = out[fx, fy, fz]
            o[0, 0] = c00 * inv
            o[0, 1] = (s[0, 2] * s[2, 1] - s[0, 1] * s[2, 2]) * inv
            o[0, 2] = (s[0, 1] * s[1, 2] - s[0, 2] * s[1, 1]) * inv
            o[1, 0] = c01 * inv
            o[1, 1] = (s[0, 0] * s[2, 2] - s[0, 2] * s[2, 0]) * inv
            o[1, 2] = (s[0, 2] * s[1, 0] - s[0, 0] * s[1, 2]) * inv
            o[2, 0] = c02 * inv
            o[2, 1] = (s[0, 1] * s[2, 0] - s[0, 0] * s[2, 1]) * inv
            o[2, 2] = (s[0, 0] * s[1, 1] - s[0, 1] * s[1, 0]) * inv
