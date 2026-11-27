"""A morphological model for an image or a volume: candidate models fitted to its measures,
then ranked by how well their realizations reproduce the measures of the image.

The measures of a phase of an image (a volume of voxels, a stack of sections, or one
section) are those of mathematical morphology: its fraction ``p``; its specific surface
``S_V`` (in a section, the length of the boundary per unit area ``L_A``, and ``S_V = 4 L_A /
π`` for an isotropic medium, Saltykov); its covariance ``C(h)``; the Choquet capacities of
balls (discs in a section) and of segments, ``T(K) = P(K ∩ A ≠ ∅)``; the density of its
Euler characteristic. Each candidate is fitted by the closed forms of its model to the
fraction and the specific surface (a method of moments, Serra 1982 ch. XIII; Jeulin 2021):

* Boolean spheres of radius ``R``: ``p = 1 - q``, ``S_V = θ q 4 π R^2``, ``-log q = θ 4 π R^3 /
  3``, so ``R = 3 q (-log q) / S_V``; with random radii, the capacity of balls ``-log Q(B_r)
  = θ (4 π / 3) E[(R + r)^3]`` is a cubic in ``r`` whose coefficients give ``θ`` and the first
  three moments of the radii (a lognormal law of the first two);
* hard spheres (no overlap): ``S_V = 3 p / R``;
* the excursion of a Gaussian field of covariance ``exp(-h^2 / 2 l^2)`` above ``u =
  Φ^{-1}(1 - p)``: ``S_V = (2 / π) e^{-u^2 / 2} / l``;
* Poisson fibres of radius ``R`` on isotropic lines: ``S_V = 2 q (-log q) / R``;
* a mosaic of Poisson–Voronoi cells, each of the phase with the probability ``p``: its faces
  between cells of the two phases, ``S_V = 2 p q c θ^{1/3}`` (``c`` of Meijering 1953).

The fitted models are then simulated in the box of the image and measured as the image
is, on voxels of its size (a section of a realization for a section): the bias of the
discretization is that of the image itself. Their length parameter is first calibrated
so that the specific surface of their images is that of the image; the score of a model
sums the differences of the covariances (over ``p q``), of the capacities and of the
specific surface and Euler density. The model of the smallest score reproduces best the
measures it was not fitted to — the shape of the covariance, the capacities — and is
proposed with its parameters, ready to generate realizations of any size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from morphogen import theory
from morphogen.analysis.basic import indicator
from morphogen.analysis.covariance import covariance as axis_covariance
from morphogen.analysis.random_sets import choquet_capacity
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage

#: The candidate models, in the order of the tables
CANDIDATES = (
    "boolean_spheres",
    "boolean_spheres_lognormal",
    "hardcore_spheres",
    "gaussian_excursion",
    "poisson_fibers",
    "voronoi",
)
#: The weights of the parts of the score
WEIGHTS = {"covariance": 1.0, "ball": 1.0, "segment": 1.0, "surface": 1.0, "euler": 0.5}


# ----------------------------------------------------------------------------------
# Measures of an image


@dataclass
class Descriptors:
    """The measures of a phase of an image: its fraction, its specific surface ``S_V`` (from
    ``L_A`` in a section), the density of its Euler characteristic (``χ_V``, or ``χ_A`` in a
    section), its covariance at ``lags`` (the mean over the axes of the image), the
    capacities of balls (discs in a section) of ``radii`` and of segments of ``lengths``
    (the mean over the axes)."""

    planar: bool
    periodic: bool
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    volume_fraction: float
    specific_surface: float
    euler_density: float
    lags: np.ndarray
    covariance: np.ndarray
    radii: np.ndarray
    ball: np.ndarray
    lengths: np.ndarray
    segment: np.ndarray

    @property
    def mean_chord(self) -> float:
        """``4 p / S_V``: the mean chord of the phase."""
        return 4.0 * self.volume_fraction / self.specific_surface

    def as_dict(self) -> dict:
        return {
            "planar": self.planar,
            "volume_fraction": self.volume_fraction,
            "specific_surface": self.specific_surface,
            "euler_density": self.euler_density,
            "lags": self.lags.tolist(),
            "covariance": self.covariance.tolist(),
            "radii": self.radii.tolist(),
            "ball": self.ball.tolist(),
            "lengths": self.lengths.tolist(),
            "segment": self.segment.tolist(),
        }


def _transitions_2d(mask: np.ndarray, d, periodic: bool) -> int:
    """The changes of phase between the pixels ``x`` and ``x + d`` of a section."""
    if periodic:
        return int(np.count_nonzero(mask != np.roll(mask, (-d[0], -d[1]), axis=(0, 1))))
    nx, ny = mask.shape
    a = mask[max(0, -d[0]) : nx - max(0, d[0]), max(0, -d[1]) : ny - max(0, d[1])]
    b = mask[max(0, d[0]) : nx - max(0, -d[0]), max(0, d[1]) : ny - max(0, -d[1])]
    return int(np.count_nonzero(a != b))


def boundary_length_density(image, phase: int = 1, *, periodic=None) -> float:
    """``L_A`` of a section (an image one voxel thick along z): ``(π / 2) P_L`` (Crofton in
    the plane), the crossings per unit length averaged over the axes and the diagonals of
    the pixels."""
    mask = indicator(image, phase)[:, :, 0]
    h = np.asarray(getattr(image, "spacing", (1.0, 1.0, 1.0)), dtype=float)[:2]
    per = bool(getattr(image, "periodic", False)) if periodic is None else bool(periodic)
    nx, ny = mask.shape
    densities = []
    for d in ((1, 0), (0, 1), (1, 1), (1, -1)):
        step = math.hypot(d[0] * h[0], d[1] * h[1])
        if per:
            pairs = nx * ny
        else:
            pairs = max(nx - abs(d[0]), 0) * max(ny - abs(d[1]), 0)
        if pairs == 0:
            continue
        densities.append(_transitions_2d(mask, d, per) / (pairs * step))
    return 0.5 * math.pi * float(np.mean(densities))


def euler_density_2d(image, phase: int = 1, *, periodic=None) -> float:
    """``χ_A`` of a section: pixels - pairs of 4-neighbours + 2 x 2 blocks, per unit area."""
    x = indicator(image, phase)[:, :, 0]
    h = np.asarray(getattr(image, "spacing", (1.0, 1.0, 1.0)), dtype=float)
    per = bool(getattr(image, "periodic", False)) if periodic is None else bool(periodic)
    area = x.shape[0] * h[0] * x.shape[1] * h[1]
    if not per:
        x = np.pad(x, 1)
    xb = np.roll(x, -1, axis=0)
    xc = np.roll(x, -1, axis=1)
    eb, ec = x & xb, x & xc
    squares = eb & np.roll(eb, -1, axis=1)
    chi = int(x.sum()) - int(eb.sum()) - int(ec.sum()) + int(squares.sum())
    return chi / area


def describe(image, phase: int = 1, *, lags: int | None = None, sizes: int = 12) -> Descriptors:
    """The measures of ``phase`` in an image of phases (a volume, or a section one voxel thick
    along z): see :class:`Descriptors`. ``lags`` voxels of covariance (default a quarter of the
    smallest side) and ``sizes`` radii and lengths of the capacities (up to a quarter and a
    half of the smallest side)."""
    from morphogen.analysis.morphology import euler_characteristic
    from morphogen.analysis.surface import specific_surface

    if not isinstance(image, VoxelImage):
        image = VoxelImage(np.asarray(image) if np.ndim(image) == 3 else
                           np.asarray(image)[:, :, None])  # fmt: skip
    data = indicator(image, phase)
    planar = data.shape[2] == 1
    h = np.asarray(image.spacing, dtype=float)
    axes = (0, 1) if planar else (0, 1, 2)
    n_min = min(data.shape[a] for a in axes)
    step = float(min(h[a] for a in axes))
    m = max(2, n_min // 4) if lags is None else int(lags)
    cov = axis_covariance(image, phase, max_lag=m)
    values = [cov.values[k][: m + 1] for k, a in zip("xyz", range(3), strict=True) if a in axes]
    covariance = np.mean(values, axis=0)
    lag_values = np.arange(m + 1) * step
    p = float(data.mean())
    if planar:
        surface = 4.0 / math.pi * boundary_length_density(image, phase)
        euler = euler_density_2d(image, phase)
    else:
        surface = specific_surface(image, phase)
        volume = float(np.prod(np.asarray(data.shape) * h))
        euler = euler_characteristic(image, phase) / volume
    extent = n_min * step
    radii = np.linspace(0.0, 0.25 * extent, int(sizes))
    ball = choquet_capacity(image, "ball", radii, phase=phase).values
    lengths = np.unique(np.rint(np.linspace(0, 0.5 * n_min, int(sizes)))).astype(int) * step
    seg = []
    for a in axes:
        d = [0, 0, 0]
        d[a] = 1
        seg.append(choquet_capacity(image, "segment", lengths, phase=phase, direction=d).values)
    return Descriptors(
        planar=planar,
        periodic=bool(image.periodic),
        shape=tuple(int(n) for n in data.shape),
        spacing=tuple(float(v) for v in h),
        volume_fraction=p,
        specific_surface=float(surface),
        euler_density=float(euler),
        lags=lag_values,
        covariance=covariance,
        radii=radii,
        ball=np.asarray(ball, dtype=float),
        lengths=lengths,
        segment=np.mean(seg, axis=0),
    )


# ----------------------------------------------------------------------------------
# Candidates fitted by their closed forms


def _box(desc: Descriptors) -> tuple[np.ndarray, float]:
    """The lengths of the box of the simulations (a slab as thick as the section is wide for
    a section) and its volume."""
    lengths = np.asarray(desc.shape, dtype=float) * np.asarray(desc.spacing)
    if desc.planar:
        lengths[2] = float(min(lengths[0], lengths[1]))
    return lengths, float(np.prod(lengths))


def fit(name: str, desc: Descriptors):
    """The model ``name`` (one of :data:`CANDIDATES`) fitted to the fraction and the specific
    surface of the descriptors (see the module); None when it cannot be (a fraction out of
    its range)."""
    from morphogen.distributions import LogNormal
    from morphogen.models.boolean import BooleanSpheres
    from morphogen.models.fibers import PoissonFibers
    from morphogen.models.gaussian import GaussianExcursion
    from morphogen.models.hardcore import HardcoreSpheres
    from morphogen.models.tessellation import Voronoi

    p = desc.volume_fraction
    s = desc.specific_surface
    if not 0.0 < p < 1.0 or s <= 0.0:
        return None
    q = 1.0 - p
    if name == "boolean_spheres":
        return BooleanSpheres(radius=3.0 * q * -math.log(q) / s, volume_fraction=p)
    if name == "boolean_spheres_lognormal":
        # -log Q(B_r) + log q = a1 r + a2 r^2 + a3 r^3 over the capacity of balls
        r, t = desc.radii, desc.ball
        ok = (t < 1.0 - 1e-9) & (r > 0.0)
        if ok.sum() < 3:
            return None
        y = -np.log(1.0 - t[ok]) + math.log(q)
        x = np.column_stack([r[ok], r[ok] ** 2, r[ok] ** 3])
        a1, a2, a3 = np.linalg.lstsq(x, y, rcond=None)[0]
        if desc.planar:  # discs in a section: θ (π² r E[R²] + 2 π r² E[R]), no cubic
            x2 = x[:, :2]
            b1, b2 = np.linalg.lstsq(x2, y, rcond=None)[0]
            if b1 <= 0.0 or b2 <= 0.0:
                return None
            m1 = 2.0 * b1 / (math.pi * b2)  # E[R^2] / E[R] for one radius: R
            theta = b2 / (2.0 * math.pi * m1)
            m2 = b1 / (math.pi**2 * theta)
        else:
            if a3 <= 0.0 or a2 <= 0.0 or a1 <= 0.0:
                return None
            theta = 3.0 * a3 / (4.0 * math.pi)
            m1 = a2 / (3.0 * a3)
            m2 = a1 / (3.0 * a3)
        cv2 = m2 / (m1 * m1) - 1.0
        if not np.isfinite(cv2) or not 1e-3 < cv2 < 4.0 or m1 * 4.0 > min(_box(desc)[0]):
            return None  # one radius (the model boolean_spheres), or no law of radii
        return BooleanSpheres(radius=LogNormal(m1, math.sqrt(cv2) * m1), volume_fraction=p)
    if name == "hardcore_spheres":
        method = "rsa" if p < 0.35 else "lubachevsky_stillinger"
        if p >= theory.MRJ_FRACTION:
            return None
        return HardcoreSpheres(radius=3.0 * p / s, volume_fraction=p, method=method)
    if name == "gaussian_excursion":
        u = float(stats.norm.isf(p))
        return GaussianExcursion(
            correlation_length=2.0 / math.pi * math.exp(-0.5 * u * u) / s, volume_fraction=p
        )
    if name == "poisson_fibers":
        return PoissonFibers(radius=2.0 * q * -math.log(q) / s, volume_fraction=p)
    if name == "voronoi":
        theta = (s / (2.0 * p * q * theory.VORONOI_SURFACE_COEFF)) ** 3
        _, volume = _box(desc)
        n = round(theta * volume)
        if n < 2:
            return None
        return Voronoi(n_cells=n, phase_fractions=(q, p))
    raise ValueError(f"candidate: one of {', '.join(CANDIDATES)}")


def _rescaled(model, factor: float):
    """The model with its lengths multiplied by ``factor`` (its fraction kept)."""
    import dataclasses

    from morphogen.distributions import Distribution

    name = model.name
    if name == "voronoi":
        return dataclasses.replace(model, n_cells=max(2, round(model.n_cells / factor**3)))
    key = {"gaussian_excursion": "correlation_length"}.get(name, "radius")
    value = getattr(model, key)
    if isinstance(value, Distribution):
        new = value._scaled(factor)
    else:
        new = float(value) * factor
    return dataclasses.replace(model, **{key: new})


# ----------------------------------------------------------------------------------
# Simulation, scores and the ranking


def _simulated(model, desc: Descriptors, seed: int, phase_of_model: int = 1):
    """The image of a realization of a model in the box of the descriptors, on voxels of
    their size (a section of it at mid-thickness for a section)."""
    lengths, _ = _box(desc)
    periodic = desc.periodic and model.supports_periodic
    domain = Domain(size=tuple(lengths), periodic=periodic)
    micro = model.generate(domain, seed=seed)
    shape = desc.shape if not desc.planar else (desc.shape[0], desc.shape[1], 1)
    img = micro.voxelize(grid=Grid(domain, shape))
    if desc.planar:
        img = VoxelImage(img.data, spacing=desc.spacing, periodic=periodic)
    elif not periodic:
        img = VoxelImage(img.data, spacing=img.spacing, periodic=False)
    return img


def _mean_descriptors(images, desc: Descriptors) -> Descriptors:
    """The descriptors of the images of realizations, averaged."""
    lags = len(desc.lags) - 1
    sizes = len(desc.radii)
    parts = [describe(img, 1, lags=lags, sizes=sizes) for img in images]
    first = parts[0]

    def mean(attr):
        return np.mean([getattr(d, attr) for d in parts], axis=0)

    return Descriptors(
        planar=first.planar,
        periodic=first.periodic,
        shape=first.shape,
        spacing=first.spacing,
        volume_fraction=float(mean("volume_fraction")),
        specific_surface=float(mean("specific_surface")),
        euler_density=float(mean("euler_density")),
        lags=first.lags,
        covariance=mean("covariance"),
        radii=first.radii,
        ball=mean("ball"),
        lengths=first.lengths,
        segment=mean("segment"),
    )


def _correlation(d: Descriptors) -> np.ndarray:
    """``ρ(h) = (C(h) - p^2) / (p q)``: the shape of the covariance, of the fraction of the
    image itself."""
    p = d.volume_fraction
    return (d.covariance - p * p) / max(p * (1.0 - p), 1e-12)


def _contact(t: np.ndarray, p: float) -> np.ndarray:
    """``H = 1 - (1 - T) / q``: a contact distribution from a capacity, of the fraction of the
    image itself."""
    return 1.0 - (1.0 - t) / max(1.0 - p, 1e-12)


def score(measured: Descriptors, predicted: Descriptors) -> dict[str, float]:
    """The parts of the score of a prediction (see the module) and their weighted sum
    ``total``: the root mean squares of the differences of the correlation functions ``(C(h)
    - p^2) / p q`` and of the contact distributions of balls and segments ``1 - (1 - T) /
    q`` (each of the fraction of its own image, so that the fluctuation of the fraction of
    a realization does not count), the relative difference of the specific surfaces and
    the difference of the Euler densities times the cube (the square in a section) of the
    mean chord ``4 p / S_V``, at most 1."""
    n = min(len(measured.covariance), len(predicted.covariance))
    pm, pp = measured.volume_fraction, predicted.volume_fraction

    def rms(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    parts = {
        "covariance": rms(_correlation(measured)[:n], _correlation(predicted)[:n]),
        "ball": rms(_contact(measured.ball, pm), _contact(predicted.ball, pp)),
        "segment": rms(_contact(measured.segment, pm), _contact(predicted.segment, pp)),
        "surface": abs(predicted.specific_surface / measured.specific_surface - 1.0),
    }
    chord = measured.mean_chord
    scale = chord**2 if measured.planar else chord**3
    parts["euler"] = float(min(abs(measured.euler_density - predicted.euler_density) * scale, 1.0))
    parts["total"] = float(sum(WEIGHTS[k] * v for k, v in parts.items()))
    return parts


@dataclass
class Candidate:
    """A model fitted to an image: the ``model`` (calibrated), the descriptors of its
    simulated images (``predicted``) and the parts of its ``score``."""

    name: str
    model: object
    predicted: Descriptors | None
    score: dict[str, float]
    note: str = ""

    @property
    def total(self) -> float:
        return self.score.get("total", math.inf)


@dataclass
class Identification:
    """The candidates fitted to an image, ranked by their scores (the best first)."""

    descriptors: Descriptors
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    def table(self) -> str:
        """The ranking as a plain text table."""
        head = f"{'model':28s} {'score':>8s} {'cov':>7s} {'ball':>7s} {'seg':>7s} {'S_V':>7s}"
        lines = [head]
        for c in self.candidates:
            s = c.score
            if not s:
                lines.append(f"{c.name:28s} {'-':>8s}  {c.note}")
                continue
            lines.append(
                f"{c.name:28s} {s['total']:8.4f} {s['covariance']:7.4f} {s['ball']:7.4f} "
                f"{s['segment']:7.4f} {s['surface']:7.4f}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "descriptors": self.descriptors.as_dict(),
            "candidates": [
                {"name": c.name, "model": c.model.to_config() if c.model is not None else None,
                 "score": c.score, "note": c.note}
                for c in self.candidates
            ],
        }  # fmt: skip


def identify(
    image,
    phase: int = 1,
    *,
    candidates=CANDIDATES,
    realizations: int = 2,
    calibration: int = 2,
    seed: int = 0,
) -> Identification:
    """The candidate models of a phase of an image of phases (a volume, a stack, or a section
    one voxel thick along z), fitted, calibrated and ranked (see the module): each is fitted
    to the fraction and the specific surface, its length scaled ``calibration`` times so that
    the specific surface of its simulated images is that of the image, and scored on the
    means of the measures of ``realizations`` images of its realizations (the same seeds for
    every model)."""
    desc = describe(image, phase)
    out = Identification(desc)
    for name in candidates:
        model = fit(name, desc)
        if model is None:
            out.candidates.append(Candidate(name, None, None, {}, "cannot be fitted"))
            continue
        try:
            for _ in range(int(calibration)):
                images = [_simulated(model, desc, seed + k) for k in range(int(realizations))]
                sim = _mean_descriptors(images, desc)
                if sim.specific_surface <= 0.0:
                    break
                model = _rescaled(model, sim.specific_surface / desc.specific_surface)
            images = [_simulated(model, desc, seed + k) for k in range(int(realizations))]
            sim = _mean_descriptors(images, desc)
        except (ValueError, RuntimeError) as exc:
            out.candidates.append(Candidate(name, model, None, {}, f"not simulated: {exc}"))
            continue
        out.candidates.append(Candidate(name, model, sim, score(desc, sim)))
    out.candidates.sort(key=lambda c: c.total)
    return out
