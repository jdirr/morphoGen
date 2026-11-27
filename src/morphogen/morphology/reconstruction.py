"""Reconstruction of a two-phase structure from its statistical descriptors by gradient
descent on a relaxed indicator (Seibert, Ambati, Raßloff & Kästner 2021; MCRpy, Seibert et
al. 2022): the alternative to the annealing of Yeong & Torquato (1998), which swaps the
cells one pair at a time (:class:`morphogen.models.Reconstruction`, ``method="annealing"``).

The phase is a field ``x`` on a periodic grid of ``N`` cells, relaxed from 0 and 1 to
``[0, 1]``, and

* its two-point probability at every lag of the grid, ``S2[x](r) = (1/N) Σ_y x(y) x(y + r)``,
  ``IFFT(|X̂|²) / N`` (``X̂`` the discrete Fourier transform of ``x``); the gradient of an
  energy ``E`` of it is ``∂E/∂x = (2/N) IFFT(Ĝ X̂)``, ``G = ∂E/∂S2`` (even in ``r``);
* its lineal paths along the axes, ``L_a[x](r) = (1/N) Σ_y P_r(y)``,
  ``P_r(y) = Π_{k=0..r} x(y + k e_a)``: for an indicator, the probability that ``r + 1``
  cells in a row along the axis ``a`` lie in the phase, the lineal-path function of
  :func:`morphogen.analysis.correlation.lineal_path` at the lag of ``r`` cells; its gradient
  ``∂L_a(r)/∂x(z) = (1/N) Σ_{k=0..r} P_r(z - k e_a) / x(z)``, the products of the other
  cells of the segments through ``z`` (a cell at 0 taken at :data:`TINY`);

are brought to their targets ``S2*`` and ``L*`` by L-BFGS-B within the bounds 0 and 1 (Byrd,
Lu, Nocedal & Zhu 1995; as MCRpy: the gradient does not vanish at the bounds, the cells
can turn back) on

    E = mean_M (S2 - S2*)² / v² + λ_L mean_{a,r} (L_a - L_a*)² / v²
        + λ_p (mean x - p)² / v² + μ mean x (1 - x),

``v = p (1 - p)`` the variance of the indicator of the target fraction ``p``, ``M`` the lags
within the largest lag (a ball of the grid), and ``μ`` raised from 0 in stages
(continuation), which draws ``x`` to 0 and 1; the lineal paths enter after the first stage,
the field near its structure. The field is then thresholded at the quantile of the target
fraction, which keeps the fraction exactly.

**Targets** (:class:`Descriptors`): those of a model of random sets in closed form
(:func:`descriptors_of_model`): its covariance ``C(|r|)`` at every lag (the isotropic models
with ``covariance(h)``: Boolean models of spheres, excursions of Gaussian fields...; the
others measured on a realization of the grid), and the lineal paths of the pores of the
models with a Choquet capacity (Boolean models of spheres, of fibres along each axis),
``1 - T(segment)``; those of an image, measured on
its voxels and interpolated at the lags of the grid (:func:`descriptors_of_image`). A planar
image gives a 3D structure through the two-point probability of the three coordinate
planes of the grid, which its sections share (``sections``: the dimensionality expansion of
MCRpy; the lineal paths of the image's two axes for the three of the grid).

The covariance does not determine a random set: a structure with the covariance of a model
at every lag has its fraction and its specific surface (``-4 S2'(0)`` in 3D, ``-π S2'(0)``
in 2D), not its connectivity — its lineal paths, the shape of its chords and its Euler
characteristic can differ, which the lineal paths among the targets constrain in part.

References: C. L. Y. Yeong, S. Torquato, Phys. Rev. E 57 (1998) 495; P. Seibert, M. Ambati,
A. Raßloff, M. Kästner, Comput. Mater. Sci. 196 (2021) 110455 (differentiable descriptors);
P. Seibert, A. Raßloff, K. A. Kalina, M. Ambati, M. Kästner, Integr. Mater. Manuf. Innov. 11
(2022) 450 (MCRpy); B. Lu, S. Torquato, Phys. Rev. A 45 (1992) 922 (lineal path); R. H.
Byrd, P. Lu, J. Nocedal, C. Zhu, SIAM J. Sci. Comput. 16 (1995) 1190 (L-BFGS-B).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator

from morphogen import settings

#: The default weights of the continuation: ``μ`` of the stages
STAGES = (0.0, 0.03, 0.3)
#: The value at which a cell at 0 enters the products of the lineal paths
TINY = 1e-12


def two_point(x) -> np.ndarray:
    """The two-point probability of a field on a periodic grid at every lag (the lag 0 at the
    index 0, the lag ``-r`` at ``n - r``): ``(1/N) Σ_y x(y) x(y + r)``."""
    x = np.asarray(x, dtype=np.float64)
    hat = sfft.rfftn(x, workers=settings.workers())
    return sfft.irfftn(hat * np.conj(hat), s=x.shape, workers=settings.workers()) / x.size


def lineal_paths(x, max_lag: int, axes=(0, 1, 2)) -> np.ndarray:
    """The lineal paths of a field in ``[0, 1]`` on a periodic grid along ``axes`` (rows), at
    the lags of 0 to ``max_lag`` cells (columns): ``(1/N) Σ_y Π_{k=0..r} x(y + k e_a)``."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty((len(axes), int(max_lag) + 1))
    for i, axis in enumerate(axes):
        p = x.copy()
        out[i, 0] = p.mean()
        for r in range(1, int(max_lag) + 1):
            p *= np.roll(x, -r, axis=axis)
            out[i, r] = p.mean()
    return out


def lag_vectors(shape, spacing) -> np.ndarray:
    """The lags of the cells of a periodic grid (the nearest images: the index ``i`` is the
    lag ``i`` for ``i <= n / 2``, ``i - n`` beyond), in units of length, of the shape of the
    grid plus 3."""
    parts = []
    for n, h in zip(shape, spacing, strict=True):
        i = np.arange(n)
        parts.append(np.where(i <= n // 2, i, i - n) * float(h))
    return np.stack(np.meshgrid(*parts, indexing="ij"), axis=-1)


@dataclass
class Descriptors:
    """Target descriptors on a periodic grid: the fraction of the phase, its two-point
    probability at the lags of the grid (lag 0 at the index 0), the lags fitted (``mask``),
    the lineal paths along ``axes`` at the lags of 0 to ``max_lag`` cells (or None)."""

    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    fraction: float
    two_point: np.ndarray
    mask: np.ndarray
    lineal: np.ndarray | None = None
    axes: tuple[int, ...] = (0, 1, 2)
    source: str = ""

    @property
    def max_lag(self) -> int:
        """The largest lag of the lineal paths, in cells (0 without them)."""
        return 0 if self.lineal is None else self.lineal.shape[1] - 1


def _grid_axes(shape) -> tuple[int, ...]:
    """The axes of a grid along which the structure extends (more than one cell)."""
    return tuple(a for a, n in enumerate(shape) if n > 1)


def _ball(shape, spacing, max_lag: float, sections: bool = False) -> np.ndarray:
    """The lags within ``max_lag`` (a length); with ``sections``, those of the three
    coordinate planes only."""
    lags = lag_vectors(shape, spacing)
    mask = np.linalg.norm(lags, axis=-1) <= max_lag * (1.0 + 1e-9)
    if sections:
        mask &= np.any(np.isclose(lags, 0.0), axis=-1)
    return mask


def _isotropic_covariance(model):
    """``C(h)`` of the phase 1 of a model whose covariance depends on the distance alone,
    or None."""
    function = getattr(model, "covariance", None)
    if function is None:
        return None
    anisotropy = getattr(model, "anisotropy", None)
    if anisotropy is not None and not getattr(anisotropy, "is_identity", False):
        return None
    try:
        names = list(inspect.signature(function).parameters)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return None
    return function if names == ["h"] else None


def descriptors_of_model(
    model,
    shape,
    spacing,
    *,
    phase: int = 1,
    max_lag: float,
    lineal: bool = False,
    domain=None,
    seed=None,
) -> Descriptors:
    """The descriptors of a phase of a model on a periodic grid of ``shape`` and ``spacing``:
    the covariance in closed form at every lag when it depends on the distance alone (see the
    module), else measured on a realization of the grid (``domain``: the box of the grid,
    ``seed``); the lags within ``max_lag`` (a length); with ``lineal``, the lineal paths along
    the axes of the grid up to ``max_lag``: those of the pores of a Boolean model from its
    capacity of segments, else measured on the realization."""
    from morphogen.domain import Domain, Grid

    shape = tuple(int(n) for n in shape)
    spacing = tuple(float(h) for h in spacing)
    axes = _grid_axes(shape)
    covariance = _isotropic_covariance(model)
    lengths = np.asarray(shape, dtype=float) * np.asarray(spacing)
    if domain is None:
        domain = Domain(size=tuple(lengths), periodic=True)
    realization = None

    def realized():
        nonlocal realization
        if realization is None:
            micro = model.generate(domain, seed=seed)
            image = micro.voxelize(grid=Grid(domain, shape))
            realization = (np.asarray(image.data) == phase).astype(np.float64)
        return realization

    if covariance is not None:
        distance = np.linalg.norm(lag_vectors(shape, spacing), axis=-1)
        c1 = np.asarray(covariance(distance), dtype=np.float64)
        p1 = float(np.asarray(covariance(np.zeros(1)))[0])
        s2 = c1 if phase == 1 else 1.0 - 2.0 * p1 + c1
        p = p1 if phase == 1 else 1.0 - p1
        source = f"the covariance of {model.name} in closed form"
    else:
        x = realized()
        s2, p = two_point(x), float(x.mean())
        source = f"a realization of {model.name}"
    steps = round(max_lag / min(spacing[a] for a in axes))
    lines = None
    if lineal:
        capacity = getattr(model, "choquet_capacity", None)
        if phase == 0 and capacity is not None:
            # the pores along a segment: 1 - T(segment), the Choquet capacity of the grains
            # (along the axis for oriented grains)
            oriented = "direction" in inspect.signature(capacity).parameters
            lines = np.empty((len(axes), steps + 1))
            for i, a in enumerate(axes):
                towards = {"direction": np.eye(3)[a]} if oriented else {}
                lines[i, 0] = p
                for r in range(1, steps + 1):
                    lines[i, r] = 1.0 - float(capacity("segment", r * spacing[a], **towards))
            source += ", the lineal paths from its Choquet capacity"
        else:
            lines = lineal_paths(realized(), steps, axes)
            source += ", the lineal paths measured on a realization"
    return Descriptors(shape, spacing, p, s2, _ball(shape, spacing, max_lag), lines, axes, source)


def descriptors_of_image(
    image,
    shape,
    spacing,
    *,
    phase: int = 1,
    max_lag: float,
    lineal: bool = False,
    sections: bool | None = None,
) -> Descriptors:
    """The descriptors of a phase of an image (a :class:`~morphogen.image.VoxelImage`, taken
    as periodic) on a periodic grid of ``shape`` and ``spacing``: its two-point probability on
    its own voxels, interpolated at the lags of the grid within ``max_lag`` (a length; the
    square of the fraction beyond its half size); with ``lineal``, its lineal paths along the
    axes, interpolated at the lags of the grid. ``sections`` (by default for a planar image and
    a 3D grid): the two-point probability of the image for the three coordinate planes of
    the grid, and the mean of its lineal paths along its two axes for each of the three."""
    from morphogen.analysis.basic import indicator
    from morphogen.analysis.correlation import lineal_path

    shape = tuple(int(n) for n in shape)
    spacing = tuple(float(h) for h in spacing)
    axes = _grid_axes(shape)
    data = indicator(image, phase).astype(np.float64)
    img_spacing = np.asarray(image.spacing, dtype=float)
    img_axes = _grid_axes(data.shape)
    p = float(data.mean())
    planar = len(img_axes) == 2
    if sections is None:
        sections = planar and len(axes) == 3
    if sections and not planar:
        raise ValueError("sections: the descriptors of a planar image for a 3D structure")
    s_img = np.fft.fftshift(two_point(data))
    # the lags of the image's own map, its axes of more than one voxel
    grids = []
    for a in img_axes:
        n = data.shape[a]
        grids.append((np.arange(n) - n // 2) * img_spacing[a])
    values = s_img.reshape(tuple(data.shape[a] for a in img_axes))
    at = RegularGridInterpolator(grids, values, bounds_error=False, fill_value=p * p)
    lags = lag_vectors(shape, spacing)
    s2 = np.full(shape, p * p)
    if sections:
        for keep in ((0, 1), (0, 2), (1, 2)):
            plane = np.isclose(lags[..., 3 - sum(keep)], 0.0)
            s2[plane] = at(lags[plane][:, list(keep)])
    else:
        if len(img_axes) != len(axes):
            raise ValueError("an image of the dimension of the grid (or sections of a planar one)")
        s2 = at(lags[..., list(axes)].reshape(-1, len(axes))).reshape(shape)
    s2[(0,) * len(shape)] = p  # the lag 0 exactly
    lines = None
    if lineal:
        steps = round(max_lag / min(spacing[a] for a in axes))
        measured = lineal_path(image, phase)
        names = "xyz"
        lines = np.empty((len(axes), steps + 1))
        for i, a in enumerate(axes):
            at_lags = np.arange(steps + 1) * spacing[a]
            # the axes of the image whose paths the axis takes: its own, or both of a section
            sources = img_axes if sections else (a,)
            lines[i] = np.mean(
                [np.interp(at_lags, measured.lags[names[b]], measured.values[names[b]],
                           right=0.0) for b in sources],
                axis=0,
            )  # fmt: skip
    mask = _ball(shape, spacing, max_lag, sections=bool(sections))
    source = "the sections of an image" if sections else "an image"
    return Descriptors(shape, spacing, p, s2, mask, lines, axes, source)


def debye_descriptors(shape, spacing, *, fraction: float, length: float, max_lag: float):
    """The descriptors of a Debye random medium (Debye, Anderson & Brumberger 1957):
    ``S2(r) = p² + p (1 - p) e^{-|r|/a}``."""
    shape = tuple(int(n) for n in shape)
    spacing = tuple(float(h) for h in spacing)
    p = float(fraction)
    distance = np.linalg.norm(lag_vectors(shape, spacing), axis=-1)
    s2 = p * p + p * (1.0 - p) * np.exp(-distance / float(length))
    axes = _grid_axes(shape)
    return Descriptors(shape, spacing, p, s2, _ball(shape, spacing, max_lag), None, axes,
                       "a Debye medium")  # fmt: skip


@dataclass
class Reconstructed:
    """A reconstruction (:func:`reconstruct`): the structure (0 and 1), the relaxed field,
    the energies after each stage, the iterations and evaluations of L-BFGS, and the errors
    of the structure's descriptors: the root mean square over the lags fitted, relative to
    the variance ``p (1 - p)``."""

    structure: np.ndarray
    relaxed: np.ndarray
    energies: list[float] = field(default_factory=list)
    iterations: int = 0
    evaluations: int = 0
    errors: dict[str, float] = field(default_factory=dict)


def _backward_sums(p: np.ndarray, r: int, axis: int) -> np.ndarray:
    """``Σ_{k=0..r} p(z - k e)`` along a periodic axis (``r`` below its length)."""
    n = p.shape[axis]
    total = np.cumsum(np.concatenate([p, p], axis=axis), axis=axis)
    upper = np.take(total, np.arange(n, 2 * n), axis=axis)
    lower = np.take(total, np.arange(n - r - 1, 2 * n - r - 1), axis=axis)
    return upper - lower


def energy(x, target: Descriptors, mu: float = 0.0, lineal_weight: float = 1.0,
           fraction_weight: float = 10.0) -> tuple[float, np.ndarray]:  # fmt: skip
    """The energy of the field ``x`` in ``[0, 1]`` (of the shape of the grid) and its
    gradient (see the module)."""
    x = np.asarray(x, dtype=np.float64).reshape(target.shape)
    n = x.size
    p = target.fraction
    scale = 1.0 / (p * (1.0 - p)) ** 2
    hat = sfft.rfftn(x, workers=settings.workers())
    s2 = sfft.irfftn(hat * np.conj(hat), s=x.shape, workers=settings.workers()) / n
    fitted = int(target.mask.sum())
    d = np.where(target.mask, s2 - target.two_point, 0.0)
    e = scale * float(np.sum(d * d)) / fitted
    g = (2.0 * scale / fitted) * d
    grad = (2.0 / n) * sfft.irfftn(sfft.rfftn(g, workers=settings.workers()) * hat,
                                   s=x.shape, workers=settings.workers())  # fmt: skip
    mean = float(x.mean())
    e += fraction_weight * scale * (mean - p) ** 2
    grad += fraction_weight * scale * 2.0 * (mean - p) / n
    if mu:
        e += mu * float(np.mean(x * (1.0 - x)))
        grad += mu * (1.0 - 2.0 * x) / n
    if target.lineal is not None and lineal_weight:
        # the products of a segment through a cell over its value: those of the other cells
        # (a cell at 0 held at TINY, its quotient finite)
        xc = np.maximum(x, TINY)
        terms = target.lineal.size
        for i, axis in enumerate(target.axes):
            prod = xc.copy()
            for r in range(target.lineal.shape[1]):
                if r:
                    prod *= np.roll(xc, -r, axis=axis)
                diff = float(prod.mean()) - target.lineal[i, r]
                e += lineal_weight * scale * diff * diff / terms
                coef = lineal_weight * scale * 2.0 * diff / terms / n
                grad += coef * _backward_sums(prod, r, axis) / xc
    return e, grad.ravel()


def descriptor_errors(structure, target: Descriptors) -> dict[str, float]:
    """The errors of the descriptors of a structure (0 and 1, or relaxed) against the
    targets: the root mean square of ``S2 - S2*`` over the lags fitted and of ``L - L*``,
    relative to ``p (1 - p)``, and the error of the fraction."""
    x = np.asarray(structure, dtype=np.float64)
    v = target.fraction * (1.0 - target.fraction)
    d = (two_point(x) - target.two_point)[target.mask]
    out = {"two_point": float(np.sqrt(np.mean(d * d))) / v,
           "fraction": float(x.mean() - target.fraction)}  # fmt: skip
    if target.lineal is not None:
        lines = lineal_paths(x, target.max_lag, target.axes)
        out["lineal_path"] = float(np.sqrt(np.mean((lines - target.lineal) ** 2))) / v
    return out


def reconstruct(
    target: Descriptors,
    *,
    seed=None,
    iterations: int = 2000,
    lineal_weight: float = 1.0,
    fraction_weight: float = 10.0,
    stages=STAGES,
    tolerance: float = 1e-12,
    initial=None,
) -> Reconstructed:
    """A structure of the descriptors ``target`` (see the module): L-BFGS from random logits
    about that of the fraction (or ``initial``, a field in ``(0, 1)``), ``iterations`` in all
    over the ``stages`` of the weight ``μ`` of the binarization, then the threshold at the
    quantile of the fraction."""
    from morphogen.seeding import make_rng

    rng, _ = make_rng(seed)
    p = float(target.fraction)
    if not 0.0 < p < 1.0:
        raise ValueError("the fraction of the target: strictly between 0 and 1")
    if initial is None:
        x = np.clip(p + 0.25 * rng.standard_normal(target.shape), 0.0, 1.0)
    else:
        x = np.clip(np.asarray(initial, dtype=np.float64), 0.0, 1.0).reshape(target.shape)
    out = Reconstructed(np.zeros(target.shape, np.uint8), np.zeros(target.shape))
    per_stage = max(1, int(iterations) // max(1, len(stages)))
    bounds = optimize.Bounds(np.zeros(x.size), np.ones(x.size))
    for stage, mu in enumerate(stages):
        # the lineal paths after the first stage, on a field already near its structure
        weight = lineal_weight if stage else 0.0
        result = optimize.minimize(
            energy, x.ravel(), args=(target, mu, weight, fraction_weight), jac=True,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": per_stage, "ftol": tolerance, "gtol": 1e-14},
        )  # fmt: skip
        x = result.x.reshape(target.shape)
        out.energies.append(float(result.fun))
        out.iterations += int(result.nit)
        out.evaluations += int(result.nfev)
    ones = round(p * x.size)
    order = np.argsort(x, axis=None, kind="stable")[::-1]
    structure = np.zeros(x.size, dtype=np.uint8)
    structure[order[:ones]] = 1
    out.structure = structure.reshape(target.shape)
    out.relaxed = x
    out.errors = descriptor_errors(out.structure, target)
    return out


__all__ = ["STAGES", "Descriptors", "Reconstructed", "debye_descriptors", "descriptor_errors",
           "descriptors_of_image", "descriptors_of_model", "energy", "lag_vectors",
           "lineal_paths", "reconstruct", "two_point"]  # fmt: skip
