"""Topology of the phases of voxel images: Betti numbers, connectivity density, and the
Euler characteristic of plane sections, which gives the integral of mean curvature.

The phase is 6-connected and its complement 26-connected, as in
:func:`morphogen.analysis.morphology.euler_characteristic`, so that the numbers agree:
``chi = b0 - b1 + b2`` with ``b0`` the connected components, ``b1`` the independent
tunnels (handles, the "connectivity" of trabecular bone) and ``b2`` the cavities. The box
is seen as a part of space (outside the box: the complement), even for periodic images,
whose components cut by the faces are counted once per piece.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from morphogen.analysis.basic import indicator
from morphogen.image import VoxelImage


@dataclass
class Betti:
    """Betti numbers and Euler characteristic of a phase."""

    b0: int  # connected components
    b1: int  # independent tunnels
    b2: int  # cavities (components of the complement inside the phase)
    euler: int


def _spacing(image) -> np.ndarray:
    spacing = image.spacing if isinstance(image, VoxelImage) else (1.0, 1.0, 1.0)
    return np.asarray(spacing, dtype=float)


def _periodic(image, periodic) -> bool:
    if periodic is None:
        return bool(image.periodic) if isinstance(image, VoxelImage) else False
    return bool(periodic)


def betti_numbers(image, phase: int = 1) -> Betti:
    """Betti numbers of ``phase`` in the box: components (6-connected), cavities
    (26-connected components of the complement that do not reach the faces) and, by the
    Euler-Poincaré formula, tunnels."""
    from morphogen.analysis.morphology import euler_characteristic

    x = indicator(image, phase)
    _, b0 = ndimage.label(x, structure=ndimage.generate_binary_structure(3, 1))
    outside = np.pad(~x, 1, constant_values=True)
    _, n = ndimage.label(outside, structure=ndimage.generate_binary_structure(3, 3))
    b2 = n - 1  # all but the component around the box
    chi = euler_characteristic(image, phase, periodic=False)
    return Betti(b0=int(b0), b1=int(b0 + b2 - chi), b2=int(b2), euler=int(chi))


def connectivity_density(image, phase: int = 1) -> float:
    """Number of independent tunnels per unit volume, ``b1 / V``."""
    volume = float(np.prod(np.asarray(indicator(image, phase).shape) * _spacing(image)))
    return betti_numbers(image, phase).b1 / volume


def section_euler_density(image, phase: int = 1, *, periodic=None) -> float:
    """Euler characteristic per unit area of the plane sections of ``phase`` normal to the
    three axes (4-connected in the planes: pixels - pairs of neighbours + 2 x 2 blocks),
    averaged over the axes. Periodic images are counted on the tori of the sections."""
    x = indicator(image, phase)
    shape = np.asarray(x.shape)
    h = _spacing(image)
    if not _periodic(image, periodic):
        x = np.pad(x, 1)
    densities = []
    for a in range(3):
        b, c = (ax for ax in range(3) if ax != a)
        xb = np.roll(x, -1, axis=b)
        xc = np.roll(x, -1, axis=c)
        edges_b, edges_c = x & xb, x & xc
        squares = edges_b & np.roll(edges_b, -1, axis=c)
        chi = int(x.sum()) - int(edges_b.sum()) - int(edges_c.sum()) + int(squares.sum())
        area = shape[a] * shape[b] * h[b] * shape[c] * h[c]
        densities.append(chi / area)
    return float(np.mean(densities))


def mean_curvature_density(image, phase: int = 1, *, periodic=None) -> float:
    """Integral of mean curvature per unit volume, ``M_V = 2 pi chi_A`` (Crofton: the
    Euler characteristic per unit area of plane sections; exact on average for
    isotropic structures, the mean of three orthogonal planes otherwise)."""
    return 2.0 * math.pi * section_euler_density(image, phase, periodic=periodic)
