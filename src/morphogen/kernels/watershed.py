"""Marker-controlled watershed of an image (Meyer's flooding): from the markers, the
voxels of the mask are conquered in increasing order of the largest cost on their path,
each by the label of the neighbour that reached it, ties in their order of arrival (a
plateau is shared evenly between the markers that reach it together). 26-connected
voxels (8-connected pixels in a plane), through the faces of a periodic image; a binary
heap on (cost, arrival). The markers are the maxima of a height of prominence at least
``h`` (the descent from them to the saddle towards a higher one), by a union-find over
the voxels in decreasing height, and the highest point of each connected part of the
mask. ``flood`` and ``prominent_maxima`` take the planes of an image (sections),
``flood_volume`` and ``prominent_maxima_volume`` its volume.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _before(keys, ticks, a, b):
    return keys[a] < keys[b] or (keys[a] == keys[b] and ticks[a] < ticks[b])


@njit(cache=True, inline="always")
def _swap(keys, ticks, items, a, b):
    keys[a], keys[b] = keys[b], keys[a]
    ticks[a], ticks[b] = ticks[b], ticks[a]
    items[a], items[b] = items[b], items[a]


@njit(cache=True)
def _push(keys, ticks, items, size, key, tick, item):
    i = size
    keys[i], ticks[i], items[i] = key, tick, item
    while i > 0:
        parent = (i - 1) // 2
        if not _before(keys, ticks, i, parent):
            break
        _swap(keys, ticks, items, i, parent)
        i = parent
    return size + 1


@njit(cache=True)
def _pop(keys, ticks, items, size):
    key, item = keys[0], items[0]
    size -= 1
    keys[0], ticks[0], items[0] = keys[size], ticks[size], items[size]
    i = 0
    while True:
        left, right, low = 2 * i + 1, 2 * i + 2, i
        if left < size and _before(keys, ticks, left, low):
            low = left
        if right < size and _before(keys, ticks, right, low):
            low = right
        if low == i:
            break
        _swap(keys, ticks, items, i, low)
        i = low
    return key, item, size


@njit(cache=True)
def flood_volume(cost, labels, periodic):
    """In place: ``labels`` (3D, int32) holds the markers (> 0), the voxels to conquer (0)
    and those outside the mask (< 0, never conquered); ``cost`` has the same shape.
    ``periodic``: the neighbours of a voxel are also those through the faces."""
    n0, n1, n2 = labels.shape
    capacity = n0 * n1 * n2
    keys = np.empty(capacity)
    ticks = np.empty(capacity, dtype=np.int64)
    items = np.empty(capacity, dtype=np.int64)
    size, tick = 0, 0
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                if labels[i, j, k] > 0:
                    item = (i * n1 + j) * n2 + k
                    size = _push(keys, ticks, items, size, cost[i, j, k], tick, item)
                    tick += 1
    while size > 0:
        key, item, size = _pop(keys, ticks, items, size)
        i, j, k = item // (n1 * n2), (item // n2) % n1, item % n2
        label = labels[i, j, k]
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    a, b, c = i + di, j + dj, k + dk
                    if periodic:
                        if (di != 0 and n0 == 1) or (dj != 0 and n1 == 1) or (dk != 0 and n2 == 1):
                            continue
                        a, b, c = a % n0, b % n1, c % n2
                    elif a < 0 or a >= n0 or b < 0 or b >= n1 or c < 0 or c >= n2:
                        continue
                    if labels[a, b, c] == 0:
                        labels[a, b, c] = label
                        new = max(cost[a, b, c], key)
                        size = _push(keys, ticks, items, size, new, tick, (a * n1 + b) * n2 + c)
                        tick += 1


@njit(cache=True)
def flood(cost, labels):
    """In place: ``labels`` (2D, int32) holds the markers (> 0), the pixels to conquer (0)
    and those outside the mask (< 0, never conquered); ``cost`` has the same shape."""
    n1, n2 = labels.shape
    flood_volume(cost.reshape(1, n1, n2), labels.reshape(1, n1, n2), False)


@njit(cache=True)
def _root(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@njit(cache=True)
def prominent_maxima_volume(height, mask, prominence, periodic):
    """Boolean image of the maxima of ``height`` in ``mask`` (3D) of prominence at least
    ``prominence``, and of the highest voxel of each 26-connected part of the mask
    (through the faces of a ``periodic`` image)."""
    n0, n1, n2 = height.shape
    n = n0 * n1 * n2
    flat = height.ravel()
    inside = mask.ravel()
    order = np.argsort(-flat)
    parent = np.arange(n)
    peak = np.arange(n)
    done = np.zeros(n, dtype=np.bool_)
    out = np.zeros(n, dtype=np.bool_)
    roots = np.empty(26, dtype=np.int64)
    for p in order:
        if not inside[p]:
            continue
        i, j, k = p // (n1 * n2), (p // n2) % n1, p % n2
        count = 0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    a, b, c = i + di, j + dj, k + dk
                    if periodic:
                        if (di != 0 and n0 == 1) or (dj != 0 and n1 == 1) or (dk != 0 and n2 == 1):
                            continue
                        a, b, c = a % n0, b % n1, c % n2
                    elif a < 0 or a >= n0 or b < 0 or b >= n1 or c < 0 or c >= n2:
                        continue
                    q = (a * n1 + b) * n2 + c
                    if done[q]:
                        r = _root(parent, q)
                        new = True
                        for m in range(count):
                            if roots[m] == r:
                                new = False
                        if new:
                            roots[count] = r
                            count += 1
        done[p] = True
        if count == 0:
            continue  # a new component, its own peak
        best = roots[0]
        for m in range(1, count):
            if flat[peak[roots[m]]] > flat[peak[best]]:
                best = roots[m]
        for m in range(count):
            r = roots[m]
            if r != best:
                if flat[peak[r]] - flat[p] >= prominence:
                    out[peak[r]] = True
                parent[r] = best
        parent[p] = best
    for p in range(n):
        if inside[p] and _root(parent, p) == p:
            out[peak[p]] = True
    return out.reshape(n0, n1, n2)


@njit(cache=True)
def prominent_maxima(height, mask, prominence):
    """Boolean image of the maxima of ``height`` in ``mask`` (2D) of prominence at least
    ``prominence``, and of the highest pixel of each 8-connected part of the mask."""
    n1, n2 = height.shape
    found = prominent_maxima_volume(
        height.reshape(1, n1, n2), mask.reshape(1, n1, n2), prominence, False
    )
    return found.reshape(n1, n2)
