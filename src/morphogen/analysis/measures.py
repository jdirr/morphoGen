"""Named measurements used by configuration files and the command line.

Each measure maps a phase image (and the label image of tessellations and grain
structures) to a flat dictionary of numbers, so that results can be written as table
columns.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from morphogen.analysis.anisotropy import covariance_minima, fabric_tensor
from morphogen.analysis.basic import phase_fractions, volume_fraction
from morphogen.analysis.cells import cell_neighbors, cell_volumes
from morphogen.analysis.connectivity import connected_components, wrapping
from morphogen.analysis.fractal import box_dimension
from morphogen.analysis.morphology import (
    euler_characteristic,
    geometric_tortuosity,
    granulometry,
    mean_chord_length,
    minkowski,
)
from morphogen.analysis.particles import particle_sizes
from morphogen.analysis.percolation import _LATTICE_2D, directional_percolation
from morphogen.analysis.surface import DIRECTIONS_13, boundary_density, specific_surface
from morphogen.analysis.topology import betti_numbers
from morphogen.image import VoxelImage


def _volume_fraction(image: VoxelImage, labels=None) -> dict[str, float]:
    return {"volume_fraction": volume_fraction(image, 1)}


def _phase_fractions(image: VoxelImage, labels=None) -> dict[str, float]:
    return {f"fraction_{k}": v for k, v in phase_fractions(image).items()}


def _specific_surface(image: VoxelImage, labels=None) -> dict[str, float]:
    return {"specific_surface": specific_surface(image, 1)}


def _connectivity(image: VoxelImage, labels=None) -> dict[str, float]:
    comp = connected_components(image, 1)
    out = {"n_components": float(comp.n_components), "spanning_fraction": comp.spanning_fraction}
    axes = "xy" if image.shape[2] == 1 else "xyz"  # a planar image: not along z, trivially
    for axis in axes:
        out[f"percolates_{axis}"] = float(comp.percolating[axis])
    if isinstance(image, VoxelImage) and image.periodic:  # around the box, not face to face
        flags = wrapping(image, 1)
        for axis in axes:
            out[f"wraps_{axis}"] = float(flags[axis])
    return out


def _directional_percolation(image: VoxelImage, labels=None) -> dict[str, float]:
    """Along the axes and the diagonals of the grid, across the core of the image along each
    (4 directions in 2-D, 13 in 3-D, named by their indices: ``percolates_1-10``)."""
    planar = image.shape[2] == 1
    lattice = [d[:2] for d in _LATTICE_2D] if planar else DIRECTIONS_13
    result = directional_percolation(image, "lattice")
    out = {}
    for d, crossed, tortuosity in zip(lattice, result.percolates, result.tortuosity, strict=True):
        name = "".join(str(int(c)) for c in d)
        out[f"percolates_{name}"] = float(crossed)
        out[f"tortuosity_{name}"] = float(tortuosity)  # inf, then dropped: no crossing
    return out


def _fractal_dimension(image: VoxelImage, labels=None) -> dict[str, float]:
    return {"box_dimension": box_dimension(image, 1)}


def _cells(image: VoxelImage, labels=None) -> dict[str, float]:
    if labels is None:
        raise ValueError("the 'cells' measure needs cell labels (tessellations, grains)")
    vol = cell_volumes(labels)
    present = vol[vol > 0]
    nb = cell_neighbors(labels)[vol > 0]
    return {
        "n_cells_present": float(len(present)),
        "mean_cell_volume": float(present.mean()),
        "cell_volume_cv2": float(present.var() / present.mean() ** 2),
        "mean_neighbors": float(nb.mean()),
        "boundary_density": boundary_density(labels),
    }


def _chords(image: VoxelImage, labels=None) -> dict[str, float]:
    return {f"mean_chord_{p}": mean_chord_length(image, p) for p in (0, 1)}


def _thickness(image: VoxelImage, labels=None) -> dict[str, float]:
    out = {}
    for p, name in ((1, "thickness"), (0, "pore_size")):
        g = granulometry(image, p)
        t = g.thickness[g.thickness > 0]
        out[f"mean_{name}"] = float(t.mean()) if t.size else float("nan")
        out[f"granulometry_radius_{p}"] = g.mean_radius
    return out


def _euler(image: VoxelImage, labels=None) -> dict[str, float]:
    volume = float(np.prod(np.asarray(image.shape) * np.asarray(image.spacing)))
    return {f"euler_density_{p}": euler_characteristic(image, p) / volume for p in (0, 1)}


def _minkowski(image: VoxelImage, labels=None) -> dict[str, float]:
    m = minkowski(image, 1)
    return {
        "V_V": m.volume_fraction,
        "S_V": m.specific_surface,
        "M_V": m.mean_curvature,
        "chi_V": m.euler_density,
    }


def _betti(image: VoxelImage, labels=None) -> dict[str, float]:
    b = betti_numbers(image, 1)
    volume = float(np.prod(np.asarray(image.shape) * np.asarray(image.spacing)))
    return {
        "betti_0": b.b0,
        "betti_1": b.b1,
        "betti_2": b.b2,
        "connectivity_density": b.b1 / volume,
    }


def _anisotropy(image: VoxelImage, labels=None) -> dict[str, float]:
    f = fabric_tensor(image, 1)
    out = {f"mil_{k + 1}": float(v) for k, v in enumerate(f.principal)}
    out["degree_of_anisotropy"] = f.degree
    return out


def _covariance_minima(image: VoxelImage, labels=None) -> dict[str, float]:
    hole = covariance_minima(image, 1)
    return {f"covariance_minimum_{'xyz'[a]}": float(hole.lags[a]) for a in range(3)}


def _particles(image: VoxelImage, labels=None) -> dict[str, float]:
    """The particles of phase 1 in the volume (those that touch split by the watershed
    of their distance map): their number per unit volume and the mean and the coefficient
    of variation of their equivalent radii."""
    p = particle_sizes(image, 1)
    if not p.weights.sum() > 0:
        return {"particle_density": 0.0}
    return {
        "particle_density": p.density(),
        "particle_mean_radius": p.mean_radius(),
        "particle_radius_cv": p.radius_cv(),
    }


def _tortuosity(image: VoxelImage, labels=None) -> dict[str, float]:
    return {f"tortuosity_{'xyz'[a]}": geometric_tortuosity(image, 1, a) for a in range(3)}


MEASURES: dict[str, Callable[..., dict[str, float]]] = {
    "volume_fraction": _volume_fraction,
    "phase_fractions": _phase_fractions,
    "specific_surface": _specific_surface,
    "connectivity": _connectivity,
    "directional_percolation": _directional_percolation,
    "fractal_dimension": _fractal_dimension,
    "cells": _cells,
    "chords": _chords,
    "thickness": _thickness,
    "euler": _euler,
    "tortuosity": _tortuosity,
    "particles": _particles,
    "minkowski": _minkowski,
    "betti": _betti,
    "anisotropy": _anisotropy,
    "covariance_minima": _covariance_minima,
}


def measure(image: VoxelImage, names, labels: VoxelImage | None = None) -> dict[str, float]:
    """Evaluate the named measures and merge their results."""
    out: dict[str, float] = {}
    for name in names:
        if name not in MEASURES:
            raise ValueError(f"unknown measure {name!r}; available: {', '.join(MEASURES)}")
        out.update(MEASURES[name](image, labels))
    return {k: float(v) for k, v in out.items() if np.isfinite(v)}
