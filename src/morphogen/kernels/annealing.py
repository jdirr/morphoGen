"""Stochastic reconstruction of two-phase media by simulated annealing (Yeong & Torquato
1998): swaps of a cell of phase 1 and a cell of phase 0, which keep the volume fraction,
accepted by Metropolis' rule for the energy ``E = Σ_a Σ_r (S_a(r) - S*_a(r))^2 + w Σ_a
Σ_r (L_a(r) - L*_a(r))^2``: the two-point probability ``S`` and the lineal-path function
``L`` of phase 1 along the axes of a periodic lattice, against their targets.

A swap changes the counts of the pairs and segments of the lines through its two cells
only: removing phase 1 at the position ``i`` of a line removes the pairs ``(i, i ± r)``
of phase 1, ``I(i + r) + I(i - r)`` at the lag ``r``; along a line, a run of ``l`` cells
of phase 1 holds ``(l - r)_+`` segments of ``r + 1`` cells, so that splitting or merging
runs changes them by the runs on both sides of the cell (a line full of phase 1 holds a
segment at each of its cells). The energy of a swap is thus evaluated in ``O(R)`` per
axis.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _at(img, idx, axis, d, n):
    """The cell at the offset ``d`` along ``axis`` from ``idx`` (periodic)."""
    if axis == 0:
        return img[(idx[0] + d) % n, idx[1], idx[2]]
    if axis == 1:
        return img[idx[0], (idx[1] + d) % n, idx[2]]
    return img[idx[0], idx[1], (idx[2] + d) % n]


@njit(cache=True)
def _pairs(img, idx, axis, n, sign, rmax, out):
    """Add ``sign * (I(i + r) + I(i - r))`` to ``out[r]`` along ``axis``, r = 1..rmax."""
    for r in range(1, rmax + 1):
        out[r] += sign * (_at(img, idx, axis, r, n) + _at(img, idx, axis, -r, n))


@njit(cache=True)
def _segments(img, idx, axis, n, sign, rmax, out):
    """The change (``sign`` 1: the cell ``idx`` turns to phase 1; -1: to phase 0) of the
    segments of phase 1 along ``axis``, the other cells as in ``img``, into ``out[r]``."""
    left = 0
    while left < n - 1 and _at(img, idx, axis, -(left + 1), n) == 1:
        left += 1
    right = 0
    while right < n - 1 - left and _at(img, idx, axis, right + 1, n) == 1:
        right += 1
    whole = left + right == n - 1  # the other cells of the line are all of phase 1
    for r in range(1, rmax + 1):
        if whole:  # a full line (n segments) against one run of n - 1 cells
            change = n - max(n - 1 - r, 0)
        else:
            change = max(left + right + 1 - r, 0) - max(left - r, 0) - max(right - r, 0)
        out[r] += sign * change


@njit(cache=True)
def _change(img, p, q, sizes, n_axes, rmax, weight, cs, cl, target_s, target_l, total, ds, dl):
    """The energy change of swapping the cells ``p`` (phase 1) and ``q`` (phase 0), with
    the changes of the counts in ``ds`` and ``dl``; on return ``p`` is of phase 0 and
    ``q`` still of phase 0."""
    ds[:, :] = 0.0
    dl[:, :] = 0.0
    for a in range(n_axes):
        _pairs(img, p, a, sizes[a], -1.0, rmax, ds[a])
        if weight > 0.0:
            _segments(img, p, a, sizes[a], -1.0, rmax, dl[a])
    img[p[0], p[1], p[2]] = 0
    for a in range(n_axes):
        _pairs(img, q, a, sizes[a], 1.0, rmax, ds[a])
        if weight > 0.0:
            _segments(img, q, a, sizes[a], 1.0, rmax, dl[a])
    change = 0.0
    for a in range(n_axes):
        for r in range(1, rmax + 1):
            if ds[a, r] != 0.0:
                old = cs[a, r] / total - target_s[a, r]
                new = (cs[a, r] + ds[a, r]) / total - target_s[a, r]
                change += new * new - old * old
            if weight > 0.0 and dl[a, r] != 0.0:
                old = cl[a, r] / total - target_l[a, r]
                new = (cl[a, r] + dl[a, r]) / total - target_l[a, r]
                change += weight * (new * new - old * old)
    return change


@njit(cache=True, inline="always")
def _unflat(c, ny, nz, out):
    out[0] = c // (ny * nz)
    out[1] = (c // nz) % ny
    out[2] = c % nz


@njit(cache=True)
def anneal(img, target_s, target_l, weight, n_axes, rmax, trials, cooling, stage, tolerance,
           seed):  # fmt: skip
    """Anneal the periodic phase image ``img`` (uint8, 0 and 1) in place towards the
    targets (``target_s[a, r]``, ``target_l[a, r]``, r = 0..rmax, for the first ``n_axes``
    axes; the lineal path with ``weight > 0`` only): at most ``trials`` swaps, the
    temperature multiplied by ``cooling`` after every ``stage`` trials, from a temperature
    at which half the swaps that raise the energy are accepted (the mean of their
    changes over 2000 trials, divided by ln 2); stops once the energy is below
    ``tolerance``, or once ten stages (a sweep) pass without an accepted swap. Returns
    the initial and final energies, the number of trials and of accepted swaps."""
    np.random.seed(seed)  # noqa: NPY002 (the generator of Numba)
    nx, ny, nz = img.shape
    sizes = np.array([nx, ny, nz], dtype=np.int64)
    total = nx * ny * nz
    ones = np.empty(total, dtype=np.int64)
    zeros = np.empty(total, dtype=np.int64)
    n1, n0 = 0, 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if img[i, j, k] == 1:
                    ones[n1] = (i * ny + j) * nz + k
                    n1 += 1
                else:
                    zeros[n0] = (i * ny + j) * nz + k
                    n0 += 1
    cs = np.zeros((3, rmax + 1))
    cl = np.zeros((3, rmax + 1))
    if n1 == 0 or n0 == 0:
        return 0.0, 0.0, 0, 0
    idx = np.empty(3, dtype=np.int64)
    for a in range(n_axes):  # the pairs and the segments along the axes
        n = sizes[a]
        for f in range(n1):
            _unflat(ones[f], ny, nz, idx)
            run = 1
            for r in range(rmax + 1):
                cs[a, r] += _at(img, idx, a, r, n)
                if weight > 0.0 and run == 1:
                    if r > 0 and _at(img, idx, a, r, n) == 0:
                        run = 0
                    else:
                        cl[a, r] += 1
    energy = 0.0
    for a in range(n_axes):
        for r in range(1, rmax + 1):
            energy += (cs[a, r] / total - target_s[a, r]) ** 2
            if weight > 0.0:
                energy += weight * (cl[a, r] / total - target_l[a, r]) ** 2
    initial = energy
    ds = np.zeros((3, rmax + 1))
    dl = np.zeros((3, rmax + 1))
    p = np.empty(3, dtype=np.int64)
    q = np.empty(3, dtype=np.int64)
    up, n_up = 0.0, 0
    for _ in range(2000):  # the scale of the energy changes
        _unflat(ones[np.random.randint(n1)], ny, nz, p)  # noqa: NPY002
        _unflat(zeros[np.random.randint(n0)], ny, nz, q)  # noqa: NPY002
        change = _change(img, p, q, sizes, n_axes, rmax, weight, cs, cl, target_s, target_l,
                         total, ds, dl)  # fmt: skip
        img[p[0], p[1], p[2]] = 1
        if change > 0.0:
            up += change
            n_up += 1
    temperature = up / n_up / math.log(2.0) if n_up else 1e-300
    accepted, done = 0, 0
    quiet, before = 0, 0  # stages without an accepted swap
    while done < trials and energy > tolerance:
        done += 1
        a1 = np.random.randint(n1)  # noqa: NPY002
        a0 = np.random.randint(n0)  # noqa: NPY002
        _unflat(ones[a1], ny, nz, p)
        _unflat(zeros[a0], ny, nz, q)
        change = _change(img, p, q, sizes, n_axes, rmax, weight, cs, cl, target_s, target_l,
                         total, ds, dl)  # fmt: skip
        draw = np.random.random()  # noqa: NPY002
        accept = change <= 0.0 or draw < math.exp(-change / temperature)
        if accept:
            img[q[0], q[1], q[2]] = 1
            ones[a1], zeros[a0] = zeros[a0], ones[a1]
            for a in range(n_axes):
                for r in range(1, rmax + 1):
                    cs[a, r] += ds[a, r]
                    cl[a, r] += dl[a, r]
            energy += change
            accepted += 1
        else:
            img[p[0], p[1], p[2]] = 1
        if done % stage == 0:
            temperature *= cooling
            quiet = quiet + 1 if accepted == before else 0
            before = accepted
            if quiet >= 10:  # frozen: a sweep without an accepted swap
                break
    return initial, energy, done, accepted
