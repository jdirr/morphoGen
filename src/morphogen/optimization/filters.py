"""Regularization of density designs: filter, projection and symmetry.

* Density filter (Bruns and Tortorelli 2001, Bourdin 2001): weighted mean of the design
  over a ball of radius ``r`` with weights ``r - distance`` (linear hat), periodic. It
  sets a minimal length scale and removes checkerboards.
* Heaviside projection (Wang, Lazarov and Sigmund 2011) of the filtered densities:
  ``(tanh(beta eta) + tanh(beta (x - eta))) / (tanh(beta eta) + tanh(beta (1 - eta)))``,
  sharper as ``beta`` grows (continuation).
* Symmetry: projection on the designs invariant by the mirror planes of the cell
  (orthotropic), and by the permutations of the axes (cubic; square in 2D).

The three operators are linear (filter, symmetry) or diagonal (projection), with
their transposes applied to sensitivities.
"""

from __future__ import annotations

import itertools

import numpy as np

from morphogen.automata.lattice import kernel_sum, offsets

SYMMETRIES = ("none", "orthotropic", "cubic")


class DensityFilter:
    """Periodic linear-hat filter of radius ``radius`` voxels (planar for nz = 1)."""

    def __init__(self, shape, radius: float):
        self.shape = tuple(int(n) for n in shape)
        self.radius = float(radius)
        dims = 2 if self.shape[2] == 1 else 3
        if self.radius < 1.0:
            self.offsets = np.zeros((1, 3), dtype=np.int64)
            self.weights = np.ones(1)
        else:
            self.offsets = offsets("euclidean", self.radius, dims=dims, include_center=True)
            dist = np.linalg.norm(self.offsets, axis=1)
            self.weights = np.maximum(0.0, self.radius - dist)
        self.weights = self.weights / self.weights.sum()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if len(self.weights) == 1:
            return np.asarray(x, dtype=float).copy()
        return kernel_sum(
            np.asarray(x, dtype=float).reshape(self.shape), self.offsets, self.weights
        )

    def transpose(self, g: np.ndarray) -> np.ndarray:
        """Transpose of the filter (the kernel mirrored: the same kernel)."""
        if len(self.weights) == 1:
            return np.asarray(g, dtype=float).copy()
        return kernel_sum(
            np.asarray(g, dtype=float).reshape(self.shape), -self.offsets, self.weights
        )


def heaviside(x, beta: float, eta: float = 0.5):
    """Smoothed Heaviside projection (identity-like for small beta)."""
    x = np.asarray(x, dtype=float)
    if beta <= 0:
        return x.copy()
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / den


def heaviside_derivative(x, beta: float, eta: float = 0.5):
    """The derivative of the smoothed Heaviside projection (:func:`heaviside`) with respect to the
    density, of sharpness ``beta`` and threshold ``eta``; 1 without projection (``beta`` ≤ 0)."""
    x = np.asarray(x, dtype=float)
    if beta <= 0:
        return np.ones_like(x)
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (x - eta)) ** 2) / den


class Symmetry:
    """Projection on the designs invariant by a symmetry group of the cell."""

    def __init__(self, shape, kind: str = "none"):
        if kind not in SYMMETRIES:
            raise ValueError(f"symmetry: one of {SYMMETRIES}")
        self.shape = tuple(int(n) for n in shape)
        self.kind = kind
        planar = self.shape[2] == 1
        axes = (0, 1) if planar else (0, 1, 2)
        if kind == "cubic":
            n = {self.shape[a] for a in axes}
            if len(n) != 1:
                raise ValueError("cubic symmetry needs a cubic (square in 2D) grid")
        self.ops = []
        if kind == "none":
            return
        perms = list(itertools.permutations(axes)) if kind == "cubic" else [axes]
        for perm in perms:
            for flips in itertools.product((False, True), repeat=len(axes)):
                self.ops.append((perm, flips))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.shape)
        if not self.ops:
            return x.copy()
        acc = np.zeros(self.shape)
        planar = self.shape[2] == 1
        for perm, flips in self.ops:
            full = tuple(perm) + ((2,) if planar else ())
            y = np.transpose(x, full)
            for a, f in enumerate(flips):
                if f:
                    y = np.flip(y, axis=a)
            acc += y
        return acc / len(self.ops)

    transpose = __call__  # an orthogonal projection is self-adjoint
