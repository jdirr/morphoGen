"""Percolation: the forest-fire algorithm, the thresholds of families of microstructures and
their finite-size scaling, the anisotropy of percolation: along directions of one image
and of a family, in 2-D and 3-D, the rigidity threshold, where the elastic moduli by the
FFT solvers vanish, and the critical exponents of the conductivity and of the elastic
moduli at the threshold.

A planar image (one voxel along z, the 2-D images of morphoGen) percolates in its plane,
along x or y, through 4 (``connectivity`` 4, or 6: through edges) or 8 (8, 18 or 26: and
corners) neighbours."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace

import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar

from morphogen.analysis.basic import as_array
from morphogen.analysis.connectivity import _structure, connected_components, wrapping
from morphogen.analysis.surface import DIRECTIONS_13
from morphogen.domain import Grid
from morphogen.image import VoxelImage
from morphogen.seeding import SeedLike

_CRITERIA = ("x", "y", "z", "any", "all")
_METHODS = ("labels", "fire")
_CONNECTIVITIES = (4, 6, 8, 18, 26)
#: The directions of the square lattice: its axes and the diagonals of its squares
_LATTICE_2D = ((1, 0, 0), (0, 1, 0), (1, 1, 0), (1, -1, 0))
#: The exponents of the correlation length of random percolation in 2-D (exact) and 3-D
NU_2D, NU_3D = 4.0 / 3.0, 0.8765


@dataclass(frozen=True)
class Fire:
    """Forest fire lit on a face of an image (:func:`burn`).

    ``percolates`` tells whether the fire reaches the opposite face, at the step ``time``
    (the chemical distance between the faces; ``None`` when it does not); ``front[t]`` is
    the number of voxels that catch fire at step ``t``, from the face (``front[0]``) to the
    step at which the opposite face is reached, or to the extinction of the fire when the
    whole clusters were burnt (``full``); ``times``, then, the step at which each voxel
    caught fire (-1: unburnt, outside the phase or in a cluster not lit), else ``None``.
    """

    percolates: bool
    time: int | None
    front: np.ndarray
    times: np.ndarray | None = field(default=None, repr=False)


@njit(cache=True)
def _burn(times, nx, ny, nz, offsets, axis, wx, wy, wz, full, queue):
    """Breadth-first burning of a C-ordered ``(nx, ny, nz)`` image, raveled in ``times`` (-1
    in the phase, -2 outside), from its face at 0 along ``axis``, through the neighbours
    ``offsets`` (in the plane when ``nz == 1``), wrapping along the directions ``wx``,
    ``wy``, ``wz``: ``times`` receives
    the step at which each voxel catches fire and ``queue`` the voxels in the order they do,
    step by step; the burning stops at the end of the step at which the face at the far end
    is reached, unless ``full``. Returns the number of voxels burnt and that step (-1: not
    reached)."""
    nyz = ny * nz
    last = nx - 1 if axis == 0 else (ny - 1 if axis == 1 else nz - 1)
    shifts = offsets[:, 0] * nyz + offsets[:, 1] * nz + offsets[:, 2]  # flat, inside the box
    tail = 0
    for i in range(1 if axis == 0 else nx):
        for j in range(1 if axis == 1 else ny):
            for k in range(1 if axis == 2 else nz):
                q = i * nyz + j * nz + k
                if times[q] == -1:
                    times[q] = 0
                    queue[tail] = q
                    tail += 1
    reached = 0 if tail > 0 and last == 0 else -1
    head, step = 0, 0
    while head < tail and (full or reached < 0):
        end = tail  # the front of this step: queue[head:end]
        while head < end:
            q = queue[head]
            head += 1
            i = q // nyz
            j = (q - i * nyz) // nz
            k = q - i * nyz - j * nz
            if 0 < i < nx - 1 and 0 < j < ny - 1 and (nz == 1 or 0 < k < nz - 1):  # no face
                c = i if axis == 0 else (j if axis == 1 else k)
                for m in range(shifts.shape[0]):
                    p = q + shifts[m]
                    if times[p] == -1:
                        times[p] = step + 1
                        queue[tail] = p
                        tail += 1
                        if reached < 0 and c + offsets[m, axis] == last:
                            reached = step + 1
                continue
            for m in range(offsets.shape[0]):
                ii, jj, kk = i + offsets[m, 0], j + offsets[m, 1], k + offsets[m, 2]
                if ii < 0 or ii >= nx:
                    if not wx:
                        continue
                    ii = ii + nx if ii < 0 else ii - nx
                if jj < 0 or jj >= ny:
                    if not wy:
                        continue
                    jj = jj + ny if jj < 0 else jj - ny
                if kk < 0 or kk >= nz:
                    if not wz:
                        continue
                    kk = kk + nz if kk < 0 else kk - nz
                p = ii * nyz + jj * nz + kk
                if times[p] == -1:
                    times[p] = step + 1
                    queue[tail] = p
                    tail += 1
                    if reached < 0 and (ii if axis == 0 else (jj if axis == 1 else kk)) == last:
                        reached = step + 1
        step += 1
    return tail, reached


def _check_connectivity(connectivity) -> None:
    if connectivity not in _CONNECTIVITIES:
        raise ValueError("connectivity: 6, 18 or 26 (4 or 8 for a planar image)")


def _in_space(connectivity: int, planar: bool) -> int:
    """The neighbours of the 3-D lattice that ``connectivity`` stands for: on a planar image
    those of 6 (4) and 18 (8) in its plane; 4 and 8 refused on 3-D images."""
    _check_connectivity(connectivity)
    if connectivity in (4, 8):
        if not planar:
            raise ValueError(
                f"connectivity {connectivity}: a planar image (one voxel along z); 6, 18 or 26 "
                "in 3-D"
            )
        return 6 if connectivity == 4 else 18
    return connectivity


def _offsets(connectivity: int, planar: bool = False) -> np.ndarray:
    """The offsets of the ``connectivity`` neighbours of a voxel (those in the plane of a
    planar image: 4 for 4 and 6, 8 for 8, 18 and 26)."""
    offsets = np.argwhere(_structure(_in_space(connectivity, planar))) - 1
    keep = np.any(offsets != 0, axis=1) & ((offsets[:, 2] == 0) | (not planar))
    return np.ascontiguousarray(offsets[keep], dtype=np.int64)


def _mask(image, phase) -> np.ndarray:
    mask = np.ascontiguousarray(as_array(image) == phase)
    if mask.ndim != 3:
        raise ValueError(
            f"image: expected a 3-D image ((nx, ny, 1) for a 2-D one), got the shape {mask.shape}"
        )
    return mask


def _axes(mask: np.ndarray) -> str:
    """The axes along which an image percolates: x and y in the plane of a planar image."""
    return "xy" if mask.shape[2] == 1 else "xyz"


def _spacing(image) -> np.ndarray:
    return np.asarray(image.spacing if isinstance(image, VoxelImage) else (1.0, 1.0, 1.0), float)


def _fire(mask: np.ndarray, axis: int, offsets: np.ndarray, periodic, full) -> Fire:
    """The forest fire of the C-ordered boolean ``mask`` along the axis number ``axis``."""
    dtype = np.int32 if mask.size < 2**31 else np.int64
    times = np.subtract(mask.ravel(), 2, dtype=dtype)  # -1 in the phase, -2 outside
    queue = np.empty(int(np.count_nonzero(mask)), dtype=dtype)
    wrap = [bool(periodic) and a != axis for a in range(3)]
    burnt, reached = _burn(times, *mask.shape, offsets, axis, *wrap, bool(full), queue)
    return Fire(
        percolates=bool(reached >= 0),
        time=int(reached) if reached >= 0 else None,
        front=np.bincount(times[queue[:burnt]]),  # the steps of the voxels, in burning order
        times=np.maximum(times, -1).reshape(mask.shape) if full else None,
    )


def burn(
    image,
    phase: int = 1,
    axis: str = "x",
    *,
    connectivity: int = 6,
    periodic: bool | None = None,
    full: bool = False,
) -> Fire:
    """The forest fire of the phase ``phase`` of a 3-D image along ``axis`` (the burning
    algorithm of Herrmann, Hong & Stanley 1984): the phase voxels of the face at 0 normal
    to the axis catch fire at time 0, and at each step the fire spreads to the unburnt
    phase voxels among the ``connectivity`` neighbours (6: faces; 18: and edges; 26: and
    corners) of those burning, the transverse directions wrapping around a periodic image
    (``periodic``, by default that of the image), never the axis. The phase percolates
    along the axis when the fire reaches the opposite face, at the step ``time``: the
    chemical distance between the faces, the number of steps of the shortest path through
    the phase (``n - 1`` for a straight channel ``n`` voxels long). The breadth-first
    burning, compiled, queues each voxel once and stops at the end of that step, unless
    ``full`` asks for the whole clusters lit on the face and their burning times; it costs
    the voxels burnt, where :func:`~morphogen.analysis.connectivity.connected_components`
    labels the whole image, and decides as its flags ``percolating`` on open boundaries.
    A planar image (one voxel along z) burns in its plane, along x or y, through 4
    (``connectivity`` 4 or 6) or 8 (8, 18 or 26) neighbours; :func:`burn_along` burns along
    any direction.
    """
    mask = _mask(image, phase)
    if axis not in _axes(mask):
        raise ValueError(f"axis: x, y or z (x or y for a planar image), got {axis!r}")
    offsets = _offsets(connectivity, mask.shape[2] == 1)
    if periodic is None:
        periodic = image.periodic if isinstance(image, VoxelImage) else False
    return _fire(mask, "xyz".index(axis), offsets, periodic, full)


@dataclass(frozen=True)
class PercolationThreshold:
    """Estimate of a percolation threshold.

    ``threshold`` is the middle of ``bracket``, the last interval of fractions at whose ends
    the percolation probability lies on either side of 1/2; ``fractions`` are the fractions
    tried, in increasing order, and ``probabilities`` the fractions of the ``realizations``
    images that percolate at each.
    """

    threshold: float
    bracket: tuple[float, float]
    fractions: np.ndarray
    probabilities: np.ndarray
    realizations: int


def _percolates(image, phase, axis, connectivity, periodic, method="labels") -> bool:
    if periodic is None:
        periodic = image.periodic if isinstance(image, VoxelImage) else False
    mask = _mask(image, phase)
    axes, planar = _axes(mask), mask.shape[2] == 1
    if axis not in (*axes, "any", "all"):
        raise ValueError(f"axis: x, y, any or all for a planar image, got {axis!r}")
    names = axes if axis in ("any", "all") else axis
    if method == "fire":  # face to face along each axis asked, the others wrapping if periodic
        offsets = _offsets(connectivity, planar)
        burns = (_fire(mask, "xyz".index(a), offsets, periodic, False).percolates for a in names)
        return all(burns) if axis == "all" else any(burns)
    connectivity = _in_space(connectivity, planar)  # one voxel thick: the same neighbours
    if periodic:
        flags = wrapping(mask, 1, connectivity=connectivity)
    else:
        flags = connected_components(mask, 1, connectivity=connectivity).percolating
    burns = [flags[a] for a in names]  # not z in a plane, joined and wrapped trivially
    return all(burns) if axis == "all" else any(burns)


def _bisection(lower, upper, tol, realizations, seed) -> list[np.random.SeedSequence]:
    """The bounds, the tolerance and the number of realizations of a bisection checked, and
    the seeds of the realizations: ``SeedSequence(entropy, spawn_key=(..., i))``."""
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"bounds: expected 0 <= lower < upper <= 1, got {lower!r}, {upper!r}")
    if not tol > 0.0:
        raise ValueError(f"tol: expected a positive tolerance, got {tol!r}")
    if isinstance(realizations, bool) or int(realizations) != realizations or realizations < 1:
        raise ValueError(f"realizations: expected a positive integer, got {realizations!r}")
    if isinstance(seed, np.random.Generator):
        seed = int(seed.integers(2**63))
    base = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return [
        np.random.SeedSequence(base.entropy, spawn_key=(*base.spawn_key, i))
        for i in range(int(realizations))
    ]


def percolation_threshold(
    generate: Callable[[float, np.random.Generator], object],
    lower: float = 0.0,
    upper: float = 1.0,
    *,
    phase: int = 1,
    axis: str = "x",
    connectivity: int = 6,
    periodic: bool | None = None,
    method: str = "labels",
    realizations: int = 9,
    tol: float = 0.005,
    seed: SeedLike = None,
) -> PercolationThreshold:
    """Percolation threshold of a family of microstructures: the fraction at which the
    probability that the phase ``phase`` of ``generate(fraction, rng)`` percolates crosses 1/2.

    ``generate`` returns a 3-D image (a :class:`~morphogen.image.VoxelImage`, or a boolean
    or label array; one voxel along z for a 2-D one, percolating along x or y). The
    probability is the fraction of ``realizations`` images that percolate; realization ``i``
    gets a generator built from the same seed at every fraction (common random numbers),
    independent of the other realizations and reproducible from
    ``seed`` (an integer, a seed sequence, a generator, or ``None`` for fresh entropy), so
    that a family growing with the fraction for a given stream, such as
    ``rng.random(shape) < fraction``, percolates from a threshold of its own in each
    realization and the estimate is the median of these thresholds (with an even number of
    realizations a probability of exactly 1/2 counts as percolating). The probabilities at
    ``lower`` and ``upper`` must lie on either side of 1/2, in either order (a phase that
    stops percolating as the fraction grows); the bracket is halved until it is narrower
    than ``tol`` and the threshold is its middle. The estimate carries the finite-size shift
    of the criterion and the spread of the realizations' thresholds, both of order
    ``L**(-1/nu)`` (``nu`` about 0.88 in 3-D, 4/3 in 2-D) for images ``L`` voxels across,
    the spread divided by the square root of ``realizations``.

    The phase percolates along ``axis`` (``"x"``, ``"y"``, ``"z"``, ``"any"`` or ``"all"``
    of them; x and y in 2-D) through its ``connectivity`` neighbours (6: faces; 18: and
    edges; 26: and corners; 4 or 8 in 2-D) when a component joins the two faces normal to
    the axis inside the box, as in
    :func:`~morphogen.analysis.connectivity.connected_components`, or, on periodic images
    (``periodic``, by default that of each image), when a component wraps around the box
    along the axis: a cycle of neighbours through the faces winds around it, so that the
    tiled component is an infinite cluster. For instance, for the spheres of a Boolean
    model, whose continuum threshold is 0.2895 (0.27 on these images, 64 voxels across a
    periodic box 16 radii across, where spheres closer than about a voxel touch)::

        from morphogen.domain import Domain
        from morphogen.models import BooleanSpheres

        domain = Domain(size=16.0, periodic=True)

        def spheres(fraction, rng):
            model = BooleanSpheres(radius=1.0, volume_fraction=fraction)
            return model.generate(domain, seed=rng).voxelize(64)

        result = percolation_threshold(spheres, 0.15, 0.45, seed=0)

    With ``method="fire"`` the phase percolates along the axis when the forest fire of
    :func:`burn`, lit on the face at 0, reaches the opposite face (along each axis for
    ``"any"`` and ``"all"``): face to face, through the transverse faces of a periodic image
    but not those normal to the axis, a cluster crossing the slab that the image tiles
    transversally. The labels (``method="labels"``, the default) decide alike on open
    boundaries; on periodic images their wrapping implies it (the lift of a cycle winding
    around the box along the axis crosses it from face to face) but not conversely (a
    staircase joins the faces without wrapping), so that the fire percolates in at least as
    many images and its estimate lies at or below theirs. The fire stops at the opposite
    face and burns only the clusters lit on the first, where the labels label the whole
    image.
    """
    if axis not in _CRITERIA:
        raise ValueError(f"axis: one of {', '.join(_CRITERIA)}, got {axis!r}")
    _check_connectivity(connectivity)
    if method not in _METHODS:
        raise ValueError(f"method: one of {', '.join(_METHODS)}, got {method!r}")
    seeds = _bisection(lower, upper, tol, realizations, seed)

    def probability(fraction: float) -> float:
        hits = sum(
            _percolates(
                generate(fraction, np.random.default_rng(s)),
                phase,
                axis,
                connectivity,
                periodic,
                method,
            )
            for s in seeds
        )
        return hits / len(seeds)

    return _bisect(probability, lower, upper, tol, len(seeds))


def _bisect(probability, lower, upper, tol, realizations, what="percolation", rising=None):
    """The bisection of :func:`percolation_threshold`: the bracket in which ``probability``
    of the fraction crosses 1/2 between ``lower`` and ``upper`` (rising or falling; 1/2
    counted above), halved down to ``tol`` or to the resolution of the fractions. With the
    direction known (``rising``), the bounds are tried last, and only those that the
    bracket reaches (the same bracket, for a monotone probability)."""
    tried: dict[float, float] = {}

    def above(fraction: float) -> bool:
        tried[fraction] = probability(fraction)
        return tried[fraction] >= 0.5

    lo, hi = float(lower), float(upper)
    if rising is None:
        rising = above(hi)
        if above(lo) == rising:
            raise ValueError(
                f"the {what} probability does not cross 1/2 between {lo:g} and {hi:g} "
                f"({tried[lo]:g} and {tried[hi]:g})"
            )
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:  # the floating-point resolution of the fractions
            break
        if above(mid) == rising:
            hi = mid
        else:
            lo = mid
    for end, side in ((lo, not rising), (hi, rising)):
        if end not in tried and above(end) != side:
            raise ValueError(
                f"the {what} probability does not cross 1/2 between {lower:g} and "
                f"{upper:g} ({tried[end]:g} at {end:g})"
            )
    fractions = np.array(sorted(tried))
    return PercolationThreshold(
        threshold=0.5 * (lo + hi),
        bracket=(lo, hi),
        fractions=fractions,
        probabilities=np.array([tried[f] for f in fractions]),
        realizations=realizations,
    )


@dataclass(frozen=True)
class PercolationScaling:
    """Finite-size scaling of a percolation threshold (:func:`percolation_scaling`).

    ``thresholds[k]`` is the threshold of the boxes ``sizes[k]`` voxels across, the middle
    of ``brackets[k]`` (``results[k]``: its :class:`PercolationThreshold`); ``threshold``
    is that of the infinite medium, extrapolated by the fit ``thresholds = threshold +
    amplitude * sizes**(-1/nu)``, with the standard error ``stderr``; ``nu`` is the exponent
    given or, with its standard error ``nu_stderr`` (else ``None``), fitted; ``residuals``
    are the thresholds minus the fit.
    """

    sizes: np.ndarray
    thresholds: np.ndarray
    brackets: np.ndarray
    threshold: float
    stderr: float
    amplitude: float
    nu: float
    nu_stderr: float | None
    residuals: np.ndarray
    results: tuple[PercolationThreshold, ...] = field(default=(), repr=False)


def _scaling_fit(sizes, thresholds, widths, nu, fit_nu):
    """Least squares of ``thresholds = p_c + a * sizes**(-theta)`` weighted by
    ``sizes**(2/nu)``, ``theta`` being ``1/nu`` or (``fit_nu``) fitted too: ``p_c``, ``a``,
    ``theta``, the residuals and the standard errors of the three, from the residuals and
    from the widths of the brackets (a threshold uniform in its own: variance width**2/12),
    linearized about the fit."""
    logs = np.log(sizes)
    root = sizes ** (1.0 / nu)  # the square roots of the weights

    def linear(theta):
        design = np.column_stack([np.ones_like(logs), np.exp(-theta * logs)])
        return np.linalg.lstsq(root[:, None] * design, root * thresholds, rcond=None)[0], design

    def cost(theta):
        beta, design = linear(theta)
        return float(np.sum((root * (thresholds - design @ beta)) ** 2))

    theta = 1.0 / nu
    if fit_nu:  # p_c and a eliminated: 1/nu scanned over [0.2, 5], refined about the best
        grid = np.geomspace(0.2, 5.0, 97)
        k = int(np.argmin([cost(t) for t in grid]))
        bounds = (grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)])
        best = minimize_scalar(cost, bounds=bounds, method="bounded", options={"xatol": 1e-12})
        theta = float(best.x)
    (pc, a), design = linear(theta)
    residuals = thresholds - design @ (pc, a)
    jacobian = np.column_stack([design, -a * design[:, 1] * logs]) if fit_nu else design
    weighted = root[:, None] * jacobian
    inverse = np.linalg.pinv(weighted.T @ weighted)
    gain = inverse @ (weighted.T * root)  # the derivatives of the parameters by the thresholds
    scatter = np.sum((root * residuals) ** 2) / (len(sizes) - jacobian.shape[1])
    errors = np.sqrt(scatter * np.diag(inverse) + gain**2 @ (widths**2 / 12.0))
    return float(pc), float(a), theta, residuals, errors


def _check_sizes(sizes, need: int, fitted: str) -> np.ndarray:
    """The distinct box sizes of ``sizes`` (positive integers, at least ``need``), sorted."""
    values = np.atleast_1d(np.asarray(sizes))
    if (
        values.ndim != 1
        or values.dtype.kind not in "iuf"
        or not np.all(np.isfinite(values))
        or np.any(values != np.round(values))
        or np.any(values < 1)
    ):
        raise ValueError(f"sizes: expected positive integers, got {sizes!r}")
    values = np.unique(values.astype(np.int64))
    if len(values) < need:
        raise ValueError(
            f"sizes: at least {need} distinct sizes to fit {fitted} with an error, got "
            f"{len(values)}"
        )
    return values


def _base_seed(seed: SeedLike) -> np.random.SeedSequence:
    """The seed sequence of ``seed`` (an integer, a seed sequence, a generator or None)."""
    if isinstance(seed, np.random.Generator):
        seed = int(seed.integers(2**63))
    return seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)


def percolation_scaling(
    generate: Callable[[float, np.random.Generator, int], object],
    sizes,
    lower: float = 0.0,
    upper: float = 1.0,
    *,
    nu: float | None = None,
    fit_nu: bool = False,
    phase: int = 1,
    axis: str = "x",
    connectivity: int = 6,
    periodic: bool | None = None,
    method: str = "labels",
    realizations: int = 9,
    tol: float = 0.005,
    seed: SeedLike = None,
) -> PercolationScaling:
    """Finite-size scaling of the percolation threshold of a family of microstructures: the
    thresholds p½(L) of :func:`percolation_threshold` in boxes of each of the ``sizes`` L,
    ``generate(fraction, rng, size)`` returning an image ``size`` voxels per edge,
    extrapolated to the infinite medium.

    Near the threshold p_c of the infinite medium, the correlation length diverging as
    ``|p - p_c|**(-nu)``, the probability that the phase of a box L voxels across
    percolates depends on the fraction p through ``(p - p_c) * L**(1/nu)`` only (Stauffer &
    Aharony 1994, ch. 4): it crosses 1/2 at ``p½(L) = p_c + a * L**(-1/nu)``, the amplitude
    ``a`` set by the criterion, the shape of the box and its boundaries (positive for the
    faces normal to one axis of an open cube, joined at p_c with a probability below 1/2),
    and the thresholds of the realizations spread over a width of the same order. With
    common random numbers a family growing with the fraction percolates from a threshold of
    its own in each realization, and p½(L) is the median of these thresholds (the lower
    median for an even number of realizations): it scatters from seed to seed by about
    ``sqrt(pi/2) = 1.25`` times their spread over ``sqrt(realizations)``, so that the shift
    and the scatter both shrink as ``L**(-1/nu)``, the scatter also as
    ``1/sqrt(realizations)``. ``p_c`` and ``a`` (and ``nu``, from the value given, with
    ``fit_nu``) are fitted by least squares weighted by ``L**(2/nu)``, the inverse variance
    of p½; the variance of ``p_c`` adds that of the fit, from its residuals, and that of the
    bisection, each p½ uniform in its bracket (variance ``width**2 / 12``). The
    extrapolation removes the shift but amplifies the scatter (the intercept of a line
    through points scattered as their abscissae ``L**(-1/nu)``): it improves on the largest
    box where the shift there exceeds the scatter. For the sites of the cubic lattice (by
    the fire, through faces, from 16 to 48 voxels across, ``tol=0.002``), the faces normal
    to the three axes of an open cube are joined at p_c + 0.008 on 48 voxels, and 9, 25
    and 99 realizations extrapolate to p_c within 0.0034, 0.0022 and 0.0015 (the scatter
    over seeds, matched by ``stderr``); those normal to one axis are joined at
    p_c + 0.0036, extrapolated within 0.0020 by 49 realizations. Four such sizes leave
    ``nu`` undetermined (1.0 +/- 1 with ``fit_nu``) and ``p_c`` then within 0.01 to 0.03
    only.

    ``nu`` is by default the exponent of random percolation, ``0.8765`` in 3-D (0.8765(16),
    Ballesteros et al. 1999, with p_c = 0.3116081(7) for the sites of the cubic lattice;
    0.88 in Stauffer & Aharony 1994) and ``4/3`` in 2-D (exact: den Nijs 1979, Nienhuis
    1982) for planar images (one voxel along z), universal: that of the lattices and of the
    continuum percolation of overlapping objects of a finite range. In 2-D a square is
    crossed at p_c with the probability 1/2 (by duality; Ziff 1992), and its 1/2-crossing
    lies off p_c by much less than its scatter, the fit finding ``a`` near 0: for the sites
    of the square lattice (by the fire, 32 to 256 across, 49 realizations, six seeds),
    squares extrapolate within 0.002 of p_c = 0.59274621(13) through 4 neighbours (Newman &
    Ziff 2000) and 0.003 of 0.407254 through 8 (1 - p_c: the matching lattice; Sykes &
    Essam 1964), and rectangles twice as long as wide, joined the long way (``a`` about
    0.36, the largest 0.006 above p_c), within 0.002. The ``sizes`` are at least 3 distinct positive
    integers (4 with ``fit_nu``), each bisected between ``lower`` and ``upper`` with the
    criterion (``phase``, ``axis``, ``connectivity``, ``periodic``, ``method``), the
    ``realizations`` and the ``tol`` of :func:`percolation_threshold`, from a seed of its
    own derived from ``seed`` and the size (independent estimates, each reproducible
    whatever the other sizes). For instance, the site percolation of the cubic lattice::

        def site(fraction, rng, size):
            return rng.random((size, size, size)) < fraction

        result = percolation_scaling(site, [16, 24, 32, 48], 0.2, 0.45, method="fire")

    and for the model of a configuration, the family of :func:`family_from_config` with
    ``sized=True``.
    """
    need, fitted = (4, "p_c, a and nu") if fit_nu else (3, "p_c and a")
    values = _check_sizes(sizes, need, fitted)
    if nu is not None and not nu > 0.0:
        raise ValueError(f"nu: expected a positive exponent, got {nu!r}")
    base = _base_seed(seed)
    planar = []  # whether each image is planar, for the default exponent

    def sized(fraction, rng, size):
        image = generate(fraction, rng, size)
        planar.append(np.ndim(as_array(image)) == 3 and np.shape(as_array(image))[2] == 1)
        return image

    results = tuple(
        percolation_threshold(
            lambda fraction, rng, size=size: sized(fraction, rng, size),
            lower,
            upper,
            phase=phase,
            axis=axis,
            connectivity=connectivity,
            periodic=periodic,
            method=method,
            realizations=realizations,
            tol=tol,
            seed=np.random.SeedSequence(base.entropy, spawn_key=(*base.spawn_key, size)),
        )
        for size in values.tolist()
    )
    if nu is None:
        nu = NU_2D if all(planar) else NU_3D
    thresholds = np.array([r.threshold for r in results])
    brackets = np.array([r.bracket for r in results])
    pc, a, theta, residuals, errors = _scaling_fit(
        values.astype(float), thresholds, brackets[:, 1] - brackets[:, 0], float(nu), fit_nu
    )
    return PercolationScaling(
        sizes=values,
        thresholds=thresholds,
        brackets=brackets,
        threshold=pc,
        stderr=float(errors[0]),
        amplitude=a,
        nu=1.0 / theta if fit_nu else float(nu),
        nu_stderr=float(errors[2]) / theta**2 if fit_nu else None,
        residuals=residuals,
        results=results,
    )


def family_from_config(
    config, parameter: str = "volume_fraction", *, base_dir=None, sized: bool = False
) -> Callable[..., VoxelImage]:
    """The family of microstructures of a run configuration, for :func:`percolation_threshold`.

    ``config`` is a file, a mapping or a :class:`~morphogen.config.RunConfig` (validated by
    :func:`~morphogen.config.load_config`, with ``base_dir``) whose ``model``, ``domain`` and
    ``voxels`` sections are used: ``generate(fraction, rng)`` draws from ``rng`` a
    realization of the model whose parameter ``parameter`` is ``fraction``, in the domain,
    and voxelizes it on the grid of the configuration, as :func:`morphogen.pipeline.run`
    does. A model without the parameter, and a configuration without a domain size or a
    voxel grid, are refused (:class:`~morphogen.config.ConfigError`). For instance::

        config = {
            "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
            "domain": {"size": 16.0, "periodic": True},
            "voxels": {"shape": 64},
        }
        result = percolation_threshold(family_from_config(config), 0.15, 0.45, seed=0)

    With ``sized``, ``generate(fraction, rng, size)``, for :func:`percolation_scaling`,
    voxelizes the realization in a box ``size`` voxels per edge at the voxel size of the
    configuration, the domain scaled to it (its origin and periodicity kept): the box grows
    in units of the microstructure, as finite-size scaling wants, where a finer grid of the
    same domain would converge to the threshold of that domain. The parameters of the model
    must then be intensive (sizes, fractions, densities), a number of objects being diluted
    in the growing box. A grid of one voxel along z (a 2-D configuration) keeps it: the box
    grows in the plane, ``size`` voxels along x and y.
    """
    from morphogen.config import ConfigError, RunConfig, load_config

    cfg = config if isinstance(config, RunConfig) else load_config(config, base_dir=base_dir)
    model, domain, voxels = cfg.model, cfg.domain, cfg.voxels
    names = [f.name for f in fields(model)]
    if parameter not in names:
        raise ConfigError(
            f"model {model.name}: no parameter {parameter!r} to vary (its parameters: "
            f"{', '.join(names)})"
        )
    if domain is None:
        raise ConfigError("percolation: give the size of the domain (domain.size)")
    if voxels is None:
        raise ConfigError("percolation: give the voxel grid (voxels section)")
    grid = voxels.kwargs()

    def varied(fraction: float):
        try:
            return replace(model, **{parameter: float(fraction)})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"model {model.name}: {parameter} = {fraction:g}: {exc}") from None

    def generate(fraction: float, rng: np.random.Generator) -> VoxelImage:
        return varied(fraction).generate(domain, seed=rng).voxelize(**grid)

    if not sized:
        return generate
    configured = Grid.for_domain(domain, **grid)
    spacing, flat = configured.spacing, configured.shape[2] == 1

    def generate_sized(fraction: float, rng: np.random.Generator, size: int) -> VoxelImage:
        if isinstance(size, bool) or int(size) != size or size < 1:
            raise ValueError(f"size: expected a positive integer, got {size!r}")
        shape = (int(size), int(size), 1 if flat else int(size))
        box = replace(domain, size=tuple(float(h) * n for h, n in zip(spacing, shape, strict=True)))
        return varied(fraction).generate(box, seed=rng).voxelize(shape=shape)

    return generate_sized


# -- along directions: the cores of an image, and the anisotropy of the thresholds -----------


@njit(cache=True)
def _spread(times, nx, ny, nz, offsets, sources, target, full, queue):
    """Breadth-first burning of a C-ordered ``(nx, ny, nz)`` image raveled in ``times`` (-1
    where it may burn, -2 elsewhere), lit on those of the flat indices ``sources``, through
    the neighbours ``offsets`` (in the plane when ``nz == 1``) inside the box: ``times``
    receives the step at which each voxel catches fire and ``queue`` the voxels in the order
    they do; the burning stops at the end of the step at which a voxel of ``target``
    (nonzero, flat) catches fire, unless ``full``. Returns the number of voxels burnt and
    that step (-1: not reached)."""
    nyz = ny * nz
    shifts = offsets[:, 0] * nyz + offsets[:, 1] * nz + offsets[:, 2]
    tail, reached = 0, -1
    for q in sources:
        if times[q] == -1:
            times[q] = 0
            queue[tail] = q
            tail += 1
            if target[q]:
                reached = 0
    head, step = 0, 0
    while head < tail and (full or reached < 0):
        end = tail
        while head < end:
            q = queue[head]
            head += 1
            i = q // nyz
            j = (q - i * nyz) // nz
            k = q - i * nyz - j * nz
            inner = 0 < i < nx - 1 and 0 < j < ny - 1 and (nz == 1 or 0 < k < nz - 1)
            for m in range(offsets.shape[0]):
                if not inner:
                    ii, jj, kk = i + offsets[m, 0], j + offsets[m, 1], k + offsets[m, 2]
                    if ii < 0 or ii >= nx or jj < 0 or jj >= ny or kk < 0 or kk >= nz:
                        continue
                p = q + shifts[m]
                if times[p] == -1:
                    times[p] = step + 1
                    queue[tail] = p
                    tail += 1
                    if reached < 0 and target[p]:
                        reached = step + 1
        step += 1
    return tail, reached


@dataclass(frozen=True)
class _Core:
    """The core of a box along a direction: its voxels (flat mask), those of its lower cap
    (flat indices) and of its upper one (flat mask), the chemical distance across it full."""

    inside: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    straight: int


def _core(shape, spacing, direction, offsets) -> _Core:
    """The core of a box of ``shape`` voxels of ``spacing`` along the unit ``direction``."""
    h = np.asarray(spacing, dtype=float)
    extents = np.asarray(shape, dtype=float) * h
    planar = shape[2] == 1
    half = float((extents[:2] if planar else extents).min()) / (2.0 * np.sqrt(2.0))  # R
    x, y, z = (
        (np.arange(n) + 0.5) * d - 0.5 * e for n, d, e in zip(shape, h, extents, strict=True)
    )  # the centres of the voxels, from the centre of the box
    x, y, z = x[:, None, None], y[None, :, None], z[None, None, :]
    t = direction[0] * x + direction[1] * y + direction[2] * z  # along the axis of the core
    slack = 1e-9 * half  # ties (systematic along diagonals) decided alike despite rounding
    inside = (np.abs(t) <= half + slack) & (x**2 + y**2 + z**2 - t**2 <= (half + slack) ** 2)
    step = float(np.abs(offsets @ (np.asarray(direction) * h)).max())  # longest, along it
    lower = inside & (t < step - half - slack)  # a step from crossing the cap, not onto it
    upper = inside & (t > half - step + slack)
    if not lower.any():
        raise ValueError(f"image: {tuple(shape)} voxels too few for a core along {direction}")
    times = np.where(inside.ravel(), -1, -2).astype(np.int32)
    queue = np.empty(int(np.count_nonzero(inside)), dtype=np.int32)
    sources = np.flatnonzero(lower)
    _, reached = _spread(times, *shape, offsets, sources, upper.ravel(), False, queue)
    if reached < 0:
        raise ValueError(f"image: {tuple(shape)} voxels too few for a core along {direction}")
    return _Core(inside.ravel(), sources, upper.ravel(), int(reached))


def _fire_core(mask: np.ndarray, core: _Core, offsets: np.ndarray, full: bool) -> Fire:
    """The forest fire of the C-ordered boolean ``mask`` across ``core``."""
    dtype = np.int32 if mask.size < 2**31 else np.int64
    times = np.where(core.inside & mask.ravel(), -1, -2).astype(dtype)
    queue = np.empty(int(np.count_nonzero(times == -1)), dtype=dtype)
    burnt, reached = _spread(times, *mask.shape, offsets, core.lower, core.upper, full, queue)
    return Fire(
        percolates=bool(reached >= 0),
        time=int(reached) if reached >= 0 else None,
        front=np.bincount(times[queue[:burnt]]),
        times=np.maximum(times, -1).reshape(mask.shape) if full else None,
    )


def _directions(directions, planar: bool, spacing) -> np.ndarray:
    """The unit vectors (rows) of ``directions`` (see :func:`directional_percolation`)."""
    if isinstance(directions, str) or directions is None:
        if directions == "axes":
            d = np.eye(3)[: 2 if planar else 3]
        elif directions == "lattice":
            d = np.array(_LATTICE_2D if planar else DIRECTIONS_13, dtype=float) * spacing
        else:
            raise ValueError(f"directions: axes, lattice, a number or vectors, got {directions!r}")
    elif np.ndim(directions) == 0:
        count = directions
        if isinstance(count, bool) or int(count) != count or count < 1:
            raise ValueError(f"directions: expected a positive number, got {count!r}")
        k = np.arange(int(count))
        if planar:  # equal angles over half a turn, from x
            angle = np.pi * k / int(count)
            d = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(len(k))])
        else:  # a Fibonacci spiral over the half sphere z > 0
            z, phi = 1.0 - (k + 0.5) / int(count), np.pi * (3.0 - np.sqrt(5.0)) * k
            r = np.sqrt(1.0 - z**2)
            d = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    else:
        d = np.atleast_2d(np.asarray(directions, dtype=float))
        if d.ndim != 2 or d.shape[1] not in (2, 3) or not np.all(np.isfinite(d)):
            raise ValueError(f"directions: vectors of 2 or 3 components, got {directions!r}")
        if d.shape[1] == 2:
            d = np.column_stack([d, np.zeros(len(d))])
    norms = np.linalg.norm(d, axis=1)
    if not np.all(norms > 0):
        raise ValueError("directions: a vector of zero length")
    d = d / norms[:, None]
    if planar and np.any(np.abs(d[:, 2]) > 1e-12):
        raise ValueError("directions: in the plane of a planar image (no z component)")
    return d


def burn_along(
    image, direction, phase: int = 1, *, connectivity: int = 6, full: bool = False
) -> Fire:
    """The forest fire of the phase ``phase`` of an image along ``direction`` (a vector, in
    physical units for anisotropic voxels; in the plane of a planar image), across the core
    of the image along it: the cylinder (a square in 2-D) of the voxels whose centres lie
    within ``R`` of the axis through the centre of the box along the direction and within
    ``R`` of the centre along it, ``2 R`` long and across, ``R = D / (2 sqrt 2)`` for the
    diameter ``D`` of the ball (disc) inscribed in the box, its smallest edge. The phase
    voxels of the core within a step of a neighbour of crossing its lower cap catch fire at
    time 0 (one layer along an axis, as the face of :func:`burn`; the cap is the plane
    normal to the direction at ``-R``), the fire spreads through the ``connectivity``
    neighbours inside the core, and the phase percolates along the direction when it reaches
    those of the upper cap, at the step ``time``, the chemical distance across the core
    (``Fire`` as :func:`burn`; ``times`` -1 outside the core too).

    Any direction crosses a congruent cylinder of the inscribed ball, so that a medium
    isotropic in law crosses alike along all (up to the voxels of the core) and the
    differences between directions are those of the medium, not of the box: the two planes
    of the box normal to a diagonal touch it only at corners. The core lies inside the box
    (the periodicity of the image unused) and spans ``1/sqrt 2`` of its smallest edge (half
    of a square, 0.28 of a cube); it has the aspect of a cube (a square), whose faces
    :func:`burn` joins, and a square of the plane is crossed at p_c with the probability 1/2
    (by duality). Its chemical distance when full, that of an image of the phase alone,
    makes ``time`` a tortuosity (:func:`directional_percolation`).
    """
    if isinstance(direction, str) or np.ndim(direction) != 1:
        raise ValueError(f"direction: a vector of 2 or 3 components, got {direction!r}")
    mask = _mask(image, phase)
    planar = mask.shape[2] == 1
    offsets = _offsets(connectivity, planar)
    spacing = _spacing(image)
    core = _core(mask.shape, spacing, _directions(direction, planar, spacing)[0], offsets)
    return _fire_core(mask, core, offsets, bool(full))


@dataclass(frozen=True)
class DirectionalPercolation:
    """Percolation of a phase of one image along directions (:func:`directional_percolation`).

    ``directions`` are unit vectors (rows; no z component in 2-D), ``percolates`` whether
    the fire crosses the core of the image along each, ``distances`` its chemical distance
    across (-1: it does not) and ``straight`` that across the full core, ``tortuosity`` the
    geometric tortuosity across (inf: no crossing).
    """

    directions: np.ndarray
    percolates: np.ndarray
    distances: np.ndarray
    straight: np.ndarray
    tortuosity: np.ndarray


def _path_lengths(inside: np.ndarray, spacing, lower: np.ndarray, upper: np.ndarray):
    """The Euclidean lengths of the shortest paths through the voxels ``inside`` (boolean;
    26 neighbours, 8 in the plane of a planar image) from those among ``lower`` (flat
    indices) to each of those of ``upper`` (a flat mask) that they reach."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    planar = inside.shape[2] == 1
    index = np.full(inside.shape, -1, dtype=np.int64)
    index[inside] = np.arange(int(np.count_nonzero(inside)))
    pad = [(1, 1), (1, 1), (0, 0) if planar else (1, 1)]
    padded, low = np.pad(index, pad, constant_values=-1), np.array([p[0] for p in pad])
    rows, cols, weights = [], [], []
    for d in np.argwhere(np.ones((3, 3, 1 if planar else 3), dtype=bool)) - low:
        if tuple(d) <= (0, 0, 0):  # each pair of neighbours once
            continue
        start = low + d
        other = padded[tuple(slice(s, s + n) for s, n in zip(start, inside.shape, strict=True))]
        keep = (index >= 0) & (other >= 0)
        rows.append(index[keep])
        cols.append(other[keep])
        weights.append(np.full(int(keep.sum()), float(np.linalg.norm(d * spacing))))
    count = int(np.count_nonzero(inside))
    graph = coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), (count, count)
    ).tocsr()
    flat = index.ravel()
    sources, targets = flat[lower], flat[upper]
    sources, targets = sources[sources >= 0], targets[targets >= 0]
    if not len(sources) or not len(targets):
        return np.empty(0)
    lengths = dijkstra(graph, directed=False, indices=sources, min_only=True)[targets]
    return lengths[np.isfinite(lengths)]


def directional_percolation(
    image, directions="lattice", phase: int = 1, *, connectivity: int = 6
) -> DirectionalPercolation:
    """Percolation of the phase ``phase`` of one image along each of ``directions``: the fire
    of :func:`burn_along` across the core of the image along each, a direction and its
    opposite alike. ``directions`` are ``"lattice"`` (the axes and the diagonals of the
    grid: 4 directions in 2-D, 13 in 3-D, in physical units for anisotropic voxels, as for
    the mean intercept length of :func:`~morphogen.analysis.anisotropy.fabric_tensor`),
    ``"axes"``, a number of directions sampled (in 2-D at equal angles over half a turn from
    x; in 3-D on a Fibonacci spiral over the half sphere z > 0, quasi-uniform), or vectors
    (rows of 2 or 3 components).

    The chemical distance across the core, the number of steps of the shortest path through
    the ``connectivity`` neighbours, follows the metric of the lattice, which is not
    isotropic: through 4 neighbours a diagonal of the square lattice is crossed by many
    staircases of the same length, which the phase seldom blocks all, where an axis has
    one straight path (on sites of the square lattice at p = 0.65, 1.03 and 1.33 times the
    distance across the full core). The geometric tortuosity compares the directions: as
    :func:`~morphogen.analysis.morphology.geometric_tortuosity` across the faces normal to
    an axis, the mean Euclidean length of the shortest paths through the phase (26
    neighbours, 8 in 2-D, steps between voxel centres) from the lower cap to the voxels of
    the upper one they reach, here over that across the full core (1 for a straight
    channel along the direction, the bias of the metric of the steps divided out), and only
    where the phase percolates along the direction.
    """
    mask = _mask(image, phase)
    planar = mask.shape[2] == 1
    offsets = _offsets(connectivity, planar)
    spacing = _spacing(image)
    dirs = _directions(directions, planar, spacing)
    cores = [_core(mask.shape, spacing, n, offsets) for n in dirs]
    fires = [_fire_core(mask, core, offsets, False) for core in cores]
    distances = np.array([-1 if f.time is None else f.time for f in fires], dtype=np.int64)
    tortuosity = np.full(len(dirs), np.inf)
    shape = mask.shape
    for k, core in enumerate(cores):
        if distances[k] >= 0:
            phase_in = (core.inside & mask.ravel()).reshape(shape)
            lengths = _path_lengths(phase_in, spacing, core.lower, core.upper)
            full = _path_lengths(core.inside.reshape(shape), spacing, core.lower, core.upper)
            tortuosity[k] = lengths.mean() / full.mean()
    return DirectionalPercolation(
        directions=dirs,
        percolates=distances >= 0,
        distances=distances,
        straight=np.array([core.straight for core in cores], dtype=np.int64),
        tortuosity=tortuosity,
    )


@dataclass(frozen=True)
class PercolationAnisotropy:
    """Percolation thresholds of a family along directions (:func:`percolation_anisotropy`).

    ``samples[i, k]`` is the threshold of realization ``i`` along ``directions[k]`` (unit
    vectors, rows; no z component in 2-D), the middle of a bracket narrower than the
    tolerance (``inf`` or ``-inf``: beyond the bounds); ``thresholds`` their medians, with
    the standard errors ``stderr``; ``deviations`` the medians of the differences of the
    realizations' thresholds from their means over the directions, with the standard errors
    ``deviation_stderr``; ``ratio`` the median of the ratios of their thresholds along the
    directions of the largest and the smallest deviation, with ``ratio_stderr`` (``inf``
    when a threshold lies beyond the bounds); ``tensor`` the threshold tensor (3 x 3,
    nothing along z in 2-D), its ``eigenvalues`` (increasing; 2 in 2-D) with
    ``eigenvalue_stderr`` and its ``principal`` directions (rows), ``residuals`` the paired
    thresholds (the median of the realizations' means plus the deviations) less its
    quadratic form. The standard errors leave out the samples beyond the bounds (NaN for
    fewer than two left); the tensor is NaN where a threshold lies beyond them.
    """

    directions: np.ndarray
    thresholds: np.ndarray
    stderr: np.ndarray
    deviations: np.ndarray
    deviation_stderr: np.ndarray
    ratio: float
    ratio_stderr: float
    tensor: np.ndarray
    eigenvalues: np.ndarray
    eigenvalue_stderr: np.ndarray
    principal: np.ndarray
    residuals: np.ndarray
    samples: np.ndarray = field(repr=False)
    realizations: int


def _median_stderr(values) -> np.ndarray:
    """The standard error of the median of the finite values of each column of ``values``
    (rows: realizations), ``sqrt(pi/2)`` times their standard deviation over the square root
    of their number (a normal law); NaN for fewer than two."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[1], np.nan)
    for k in range(values.shape[1]):
        finite = values[np.isfinite(values[:, k]), k]
        if len(finite) > 1:
            out[k] = np.std(finite, ddof=1) * np.sqrt(np.pi / 2 / len(finite))
    return out


def _medians(values) -> np.ndarray:
    """The medians of the columns of ``values`` over their rows that are not NaN (NaN for
    none): infinite values beyond the bounds count, on their side."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[1], np.nan)
    for k in range(values.shape[1]):
        kept = values[~np.isnan(values[:, k]), k]
        if len(kept):
            out[k] = np.median(kept)
    return out


def _quadratic(directions: np.ndarray, planar: bool) -> np.ndarray:
    """The rows of the least squares of ``n . T . n``: the components xx, yy, xy (2-D) or
    xx, yy, zz, yz, zx, xy (3-D) of ``T``."""
    x, y, z = directions.T
    if planar:
        return np.column_stack([x * x, y * y, 2 * x * y])
    return np.column_stack([x * x, y * y, z * z, 2 * y * z, 2 * z * x, 2 * x * y])


def _tensors(components: np.ndarray, planar: bool) -> np.ndarray:
    """The symmetric 3 x 3 tensors of the components of :func:`_quadratic` (last axis)."""
    pairs = ((0, 0), (1, 1), (0, 1)) if planar else ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
    out = np.zeros((*components.shape[:-1], 3, 3))
    for m, (a, b) in enumerate(pairs):
        out[..., a, b] = out[..., b, a] = components[..., m]
    return out


def percolation_anisotropy(
    generate: Callable[[float, np.random.Generator], object],
    lower: float = 0.0,
    upper: float = 1.0,
    *,
    directions="lattice",
    phase: int = 1,
    connectivity: int = 6,
    realizations: int = 9,
    tol: float = 0.005,
    seed: SeedLike = None,
) -> PercolationAnisotropy:
    """Percolation thresholds of a family of microstructures along directions, and their
    anisotropy: the fractions at which the phase ``phase`` of ``generate(fraction, rng)``
    (as for :func:`percolation_threshold`) percolates along each of ``directions`` (as for
    :func:`directional_percolation`: across the core of the image along each).

    Each realization, seeded as by :func:`percolation_threshold` and alike at every
    fraction, is bisected along every direction to a threshold of its own, the middle of a
    bracket narrower than ``tol`` (for a family growing with the fraction, the fraction from
    which it crosses; ``inf`` or ``-inf`` beyond ``lower`` and ``upper``: a direction along
    which it does not cross between them, or always does); the directions bracketed alike
    share the images, each image burnt along all those whose bracket it halves. The
    thresholds are the medians of the realizations' (the estimate of
    :func:`percolation_threshold`: the lower median for an even number, of a family growing
    with the fraction), with the standard errors ``sqrt(pi/2) s / sqrt(N)`` from the
    standard deviation ``s`` of the ``N`` realizations' (a normal law). The realizations are
    common to the directions, which are compared within each: the deviations are the
    medians of the differences of a realization's thresholds from their mean over the
    directions, the ratio that of its thresholds along the directions of the extreme
    deviations, with the standard errors of these paired differences, free of the scatter
    that the directions share and the thresholds carry (a difference of the medians along
    two directions would keep it: the same realization tends to be their median). An
    isotropic family has deviations within a few standard errors, and a ratio above 1 by
    about as much. The threshold tensor ``T`` is the symmetric tensor whose quadratic form
    ``p(n) = n . T . n`` fits the paired thresholds (the median of the realizations' means
    plus the deviations) by least squares (diagonal and exact on the axes alone); its
    eigenvalues bound the fitted thresholds, reached along its principal directions, their
    standard errors (the shared scatter included) from the tensors fitted to each
    realization, projected on these directions; the residuals tell how far a quadratic
    form describes the thresholds.

    In the infinite medium, a statistically homogeneous medium of a finite range of
    correlation percolates from one threshold along every direction: its infinite cluster,
    when there is one, is unbounded along all. A box ``L`` across crosses at fractions that
    depend on the direction through the shape of the clusters: elongated ones (aligned
    fibres, stretched correlations) cross it sooner along their elongation, along which the
    box is shorter in units of their correlation length, and the thresholds differ by an
    amount of the order of their shift from p_c, ``L**(-1/nu)``
    (:func:`percolation_scaling`), however large the realizations. This is the anisotropy of
    percolation of a volume of that size, a specimen or an RVE; it vanishes with the size
    unless the medium is disconnected along some directions (layers separated by sheets of
    the other phase: thresholds ``inf`` across them). For instance, for fibres aligned
    along x::

        from morphogen.config import model_from_config
        from morphogen.domain import Domain

        domain = Domain(size=12.0)
        config = {"type": "boolean_cylinders", "radius": 0.5, "length": 6.0,
                  "orientation": {"type": "aligned", "direction": [1, 0, 0]}}

        def fibres(fraction, rng):
            model = model_from_config({**config, "volume_fraction": fraction})
            return model.generate(domain, seed=rng).voxelize(48)

        result = percolation_anisotropy(fibres, 0.005, 0.6, realizations=25, tol=0.002, seed=0)

    cross along x from a volume fraction of 0.099, across from 0.40 to 0.42 (a ratio of 5.0
    +/- 0.8), the smallest eigenvalue of the tensor, 0.087, along x; the sections of the
    fibres by a plane (strips, ``Domain(size=(12, 12, 0.0625), periodic=False)`` on 192 x 192
    x 1 voxels) from 0.25 along x and 0.90 across.
    """
    _check_connectivity(connectivity)
    seeds = _bisection(lower, upper, tol, realizations, seed)
    geometry: dict = {}

    def crossings(fraction: float, s, ks=None) -> np.ndarray:
        image = generate(fraction, np.random.default_rng(s))
        mask = _mask(image, phase)
        key = (mask.shape, tuple(_spacing(image).tolist()))
        if not geometry:
            planar = mask.shape[2] == 1
            offsets = _offsets(connectivity, planar)
            dirs = _directions(directions, planar, np.asarray(key[1]))
            cores = [_core(mask.shape, key[1], n, offsets) for n in dirs]
            geometry.update(key=key, planar=planar, offsets=offsets, dirs=dirs, cores=cores)
        elif key != geometry["key"]:
            raise ValueError(f"generate: images of one grid, got {key} and {geometry['key']}")
        cores, offsets = geometry["cores"], geometry["offsets"]
        ks = range(len(cores)) if ks is None else ks
        return np.array([_fire_core(mask, cores[k], offsets, False).percolates for k in ks])

    at_lower = np.array([crossings(float(lower), s) for s in seeds])
    at_upper = np.array([crossings(float(upper), s) for s in seeds])
    switching = at_lower != at_upper
    if not switching.any():
        raise ValueError(
            f"the percolation does not change between {lower:g} and {upper:g} along any "
            f"direction ({int(at_lower.sum())} and {int(at_upper.sum())} crossings of "
            f"{at_lower.size})"
        )
    rising = bool(at_upper.sum() >= at_lower.sum())
    samples = np.where(at_lower == rising, -np.inf, np.inf)  # beyond the bounds
    for i, s in enumerate(seeds):
        stack = [(float(lower), float(upper), np.flatnonzero(switching[i]))]
        while stack:
            a, b, ks = stack.pop()
            mid = 0.5 * (a + b)
            if b - a <= tol or not a < mid < b:
                samples[i, ks] = mid
                continue
            moved = crossings(mid, s, ks) != at_lower[i, ks]  # switched within [a, mid]
            for u, v, part in ((a, mid, ks[moved]), (mid, b, ks[~moved])):
                if len(part):
                    stack.append((u, v, part))
    return _anisotropy(geometry["dirs"], samples, rising, geometry["planar"])


def _anisotropy(dirs, samples, rising: bool, planar: bool) -> PercolationAnisotropy:
    """The statistics of the thresholds ``samples`` (realizations x directions)."""
    count = len(samples)
    thresholds = np.sort(samples, axis=0)[(count - 1) // 2 if rising else count // 2]
    finite = np.isfinite(thresholds)
    with np.errstate(invalid="ignore", divide="ignore"):  # samples beyond the bounds: NaN
        means = np.full((count, 1), np.nan)  # of each realization over the finite thresholds
        if finite.any():
            means = samples[:, finite].mean(axis=1)[:, None]
        paired = samples - means  # within each realization: their common scatter cancels
        deviations = _medians(paired)
        ratio, logs = np.inf, np.full((count, 1), np.nan)
        if finite.all():  # between the directions of the extreme deviations, paired too
            big, small = int(np.argmax(deviations)), int(np.argmin(deviations))
            logs = np.log(samples[:, [big]]) - np.log(samples[:, [small]])
            ratio = float(np.exp(_medians(logs)[0]))
    rows, d = _quadratic(dirs, planar), 2 if planar else 3
    inverse = np.linalg.pinv(rows)
    nan = np.full(d, np.nan)
    tensor, values, errors, axes = np.full((3, 3), np.nan), nan, nan, np.full((d, 3), np.nan)
    residuals = np.full(len(dirs), np.nan)
    if finite.all():
        paired_thresholds = _medians(means)[0] + deviations
        components = inverse @ paired_thresholds
        tensor = _tensors(components, planar)
        values, vectors = np.linalg.eigh(tensor[:d, :d])
        largest = vectors[np.argmax(np.abs(vectors), axis=0), np.arange(d)]
        vectors *= np.where(largest < 0, -1.0, 1.0)  # the largest component positive
        axes = np.zeros((d, 3))
        axes[:, :d] = vectors.T
        with np.errstate(invalid="ignore"):  # realizations beyond the bounds: NaN errors
            each = _tensors(samples @ inverse.T, planar)[:, :d, :d]  # one per realization
            errors = _median_stderr(np.einsum("ja,iab,jb->ij", vectors.T, each, vectors.T))
        residuals = paired_thresholds - rows @ components
    return PercolationAnisotropy(
        directions=dirs,
        thresholds=thresholds,
        stderr=_median_stderr(samples),
        deviations=deviations,
        deviation_stderr=_median_stderr(paired),
        ratio=ratio,
        ratio_stderr=float(ratio * _median_stderr(logs)[0]),
        tensor=tensor,
        eigenvalues=values,
        eigenvalue_stderr=errors,
        principal=axes,
        residuals=residuals,
        samples=samples,
        realizations=count,
    )


# -- the critical exponents of the effective properties at the threshold ---------------------

#: The physics of :func:`percolation_exponent`: the conductivity (t) and the elasticity (f)
_PHYSICS = ("thermal", "elastic")
#: Poisson's ratio of the phases in elasticity
_POISSON = 0.3
#: The solvers of the trilinear voxels: the FFT solver, or their matrix assembled on the
#: voxels of the materials and factorized (:mod:`morphogen.homogenization.voxel_fe`)
_SOLVERS = ("fft", "fe")
#: The default cut of :func:`rigidity_threshold` for each solver: above the stress averages
#: of the floppy images of the FFT solver, its iterations stopping short of the round-off,
#: and above the energies of those of the voxels factorized, which resolve the moduli
#: of the realizations that bear down to the round-off
CUTS = {"fft": 1e-8, "fe": 1e-12}
#: The default tolerance of the conjugate gradients of each solver for the moduli: 1e-8 for
#: the FFT solver, 1e-10 for the voxels factorized, whose solutions at 1e-8 leave the soft
#: modes of floppy images unresolved, with energies of 1e-13 to 2e-8 and stress averages that
#: agree with them (sites of the cubic lattice near their rigidity threshold: 23 of 32
#: realizations of 32³ to 80³ voxels bear from fractions lower by 0.0003 to 0.003)
SOLVER_TOLS = {"fft": 1e-8, "fe": 1e-10}


def _check_solver(solver) -> str:
    if solver not in _SOLVERS:
        raise ValueError(f"solver: one of {', '.join(_SOLVERS)}, got {solver!r}")
    return solver


@dataclass(frozen=True)
class PercolationExponent:
    """Critical exponent of an effective property at a percolation threshold
    (:func:`percolation_exponent`).

    ``exponent`` is that of the property (``physics``: ``"thermal"``, the conductivity along
    the axis, t; ``"elastic"``, the modulus of a uniaxial strain along it, f), ``nu`` times
    ``ratio``, the slope of ``-ln <P>`` against ``ln L`` fitted to the ``means`` (standard
    errors ``mean_stderr``) of the realizations of the ``sizes`` at the ``threshold``:
    ``means = amplitude * sizes**(-ratio)``, ``residuals`` the logarithms of the means less
    the fit; ``stderr`` is its statistical standard error and ``threshold_error`` half its
    change between the threshold less and plus ``threshold_stderr`` (0 without).
    ``fractions`` are those computed (increasing: the threshold, the two shifted by its
    standard error, those of the ``sweep``), ``values[k, j]`` the mean property of the boxes
    ``sizes[k]`` at ``fractions[j]`` with the standard errors ``value_stderr``, and
    ``samples[k, i, j]`` that of realization ``i``; ``sweep_exponents[k]`` is the exponent
    fitted to the means of size ``k`` against ``fraction - threshold`` over the ``sweep``,
    with ``sweep_stderr`` and ``sweep_threshold_error`` (empty without a sweep).
    ``iterations`` and ``residual`` are the largest number of iterations and relative
    residual of the ``solver``.
    """

    physics: str
    threshold: float
    exponent: float
    stderr: float
    threshold_error: float
    ratio: float
    amplitude: float
    nu: float
    sizes: np.ndarray
    means: np.ndarray
    mean_stderr: np.ndarray
    residuals: np.ndarray
    fractions: np.ndarray
    values: np.ndarray
    value_stderr: np.ndarray
    sweep: np.ndarray
    sweep_exponents: np.ndarray
    sweep_stderr: np.ndarray
    sweep_threshold_error: np.ndarray
    contrast: float
    realizations: int
    iterations: int
    residual: float
    samples: np.ndarray = field(repr=False)
    solver: str = "fft"


def _extrapolated_threshold(scaling: PercolationScaling, lower: float, upper: float):
    """The threshold of a finite-size scaling and its standard error, for
    :func:`percolation_exponent`: refused when the threshold less or plus its error leaves
    the bounds of the bisection (boxes too small or realizations too few for an
    exponent)."""
    threshold, stderr = scaling.threshold, scaling.stderr
    if not (min(lower, upper) < threshold - stderr and threshold + stderr < max(lower, upper)):
        raise ValueError(
            f"the threshold extrapolated, {threshold:.4g} +/- {stderr:.2g}, leaves the bounds "
            f"{lower:g} and {upper:g}: too uncertain for an exponent (larger boxes or more "
            "realizations, or give the threshold)"
        )
    return threshold, stderr


def _power_fit(x, means, errors):
    """Least squares of ``ln means = c + b ln x`` weighted by ``(means / errors)**2``, the
    inverse variances of the logarithms: ``b``, ``exp(c)``, the residuals and the standard
    error of ``b``, from the errors scaled by the scatter of the residuals where it exceeds
    them (the root of their reduced chi-square, which corrections to the power law raise),
    or from that scatter alone, unweighted, when the realizations of a mean are all alike
    (a relative error below 10⁻⁶: a deterministic family, or boxes where none carries and
    the property is that of the other phase); NaN unless the means are positive."""
    x, means, errors = (np.asarray(v, dtype=float) for v in (x, means, errors))
    n = len(x)
    if n < 2 or not (np.all(means > 0) and np.all(x > 0)):
        return np.nan, np.nan, np.full(n, np.nan), np.nan
    weighted = bool(np.all(errors > 1e-6 * means))
    w = means / errors if weighted else np.ones(n)
    design = np.column_stack([np.ones(n), np.log(x)])
    (c, b), *_ = np.linalg.lstsq(design * w[:, None], np.log(means) * w, rcond=None)
    residuals = np.log(means) - design @ (c, b)
    inverse = np.linalg.pinv((design * w[:, None]).T @ (design * w[:, None]))
    scatter = float(np.sum((residuals * w) ** 2)) / (n - 2) if n > 2 else 0.0
    variance = inverse[1, 1] * (max(1.0, scatter) if weighted else scatter)
    return float(b), float(np.exp(c)), residuals, float(np.sqrt(variance))


#: The round-off of the element matrices perturbed to check an energy (:func:`_bears`): their
#: entries times ``1 + ROUNDING * r``, ``r`` symmetric and standard normal
_ROUNDING = 2.2e-16


def _effective(mask, spacing, axis, physics, contrast, tol, maxiter, device, solver="fft",
               rounding=None):  # fmt: skip
    """The effective property of a periodic image whose voxels ``mask`` conduct
    (conductivity 1; Young's modulus 1) and the others ``contrast`` times as much (void at
    0), by the FFT solver on ``device`` or (``solver="fe"``) by the matrix of the trilinear
    voxels assembled and factorized, on the CPU, on the voxels that bear along the axis
    alone when the others are void (:func:`~morphogen.homogenization.voxel_fe.load_bearing`,
    exact): the conductivity along the axis number ``axis``, or the modulus of a uniaxial
    strain along it from the energy of the solution, the mean of ``sigma : epsilon``
    (stationary, its error of second order, and an upper bound); with the iterations, the
    relative residual and the stress average ``sigma_aa`` (the conductivity again), which
    equals the energy at the exact solution. ``rounding`` seeds a perturbation of the
    element matrices at their round-off (:data:`_ROUNDING`)."""
    from morphogen.homogenization import VOID, Loading, Material, make_problem
    from morphogen.homogenization.apparent import apparent_conductivity

    if solver == "fe" and contrast == 0.0:
        from morphogen.homogenization.voxel_fe import load_bearing

        mask = load_bearing(mask, axis)
        if not mask.any():  # nothing winds around the box along the axis
            return 0.0, 0, 0.0, 0.0
    image = VoxelImage(mask.astype(np.uint8), spacing=tuple(spacing), periodic=True)
    solid = Material(young=1.0, poisson=_POISSON, conductivity=1.0)
    other = VOID
    if contrast > 0.0:
        other = Material(young=contrast, poisson=_POISSON, conductivity=contrast)
    if solver == "fe":
        from morphogen.homogenization.voxel_fe import VoxelFEElasticProblem, VoxelFEThermalProblem

        cls = VoxelFEElasticProblem if physics == "elastic" else VoxelFEThermalProblem
        problem = cls(image, {0: other, 1: solid}, tol=tol, maxiter=maxiter)
    else:
        problem = make_problem(image, {0: other, 1: solid}, physics, tol=tol, maxiter=maxiter,
                               device=device)  # fmt: skip
    if rounding is not None:
        k = problem.element_matrices
        r = np.random.default_rng(rounding).standard_normal(k.shape)
        problem.element_matrices = k * (1.0 + 0.5 * _ROUNDING * (r + np.swapaxes(r, 1, 2)))
    if physics == "thermal":
        result = apparent_conductivity(problem, axis)
        info = result.solution.info
        return float(result.value), info.iterations, info.residual, float(result.value)
    strain = np.zeros(6)
    strain[axis] = 1.0
    solution = problem.solve(Loading.strain(strain))
    info = solution.info
    return (float(solution.energy), info.iterations, info.residual,
            float(solution.stress[axis, axis]))  # fmt: skip


def _bound(maxiter, physics: str, shape) -> int | None:
    """The bound of the iterations of the conjugate gradients: ``maxiter``, else that of the
    solver (20 L + 1000, L the largest dimension) for the conductivity and 50 L + 1000 for
    the moduli, whose solutions near the rigidity threshold take up to 45 L."""
    if maxiter:
        return int(maxiter)
    return 50 * int(max(shape)) + 1000 if physics == "elastic" else None


def _samples(generate, size, indices, grid, base, phase, axis, physics, contrast, tol, maxiter,
             device, solver="fft"):  # fmt: skip
    """The property of the realizations ``indices`` of the boxes ``size`` voxels across at
    the fractions ``grid`` (:func:`percolation_exponent`; realization ``i`` drawn from the
    seed ``(base, size, i)``, the same at every fraction), with the largest number of
    iterations and relative residual of the solver and whether each image is planar."""
    samples = np.empty((len(indices), len(grid)))
    iterations, residual, planar = 0, 0.0, []
    for row, i in enumerate(indices):
        s = np.random.SeedSequence(base.entropy, spawn_key=(*base.spawn_key, size, i))
        for j, fraction in enumerate(np.asarray(grid).tolist()):
            image = generate(fraction, np.random.default_rng(s), size)
            mask = _mask(image, phase)
            if axis not in _axes(mask):
                raise ValueError(f"axis: x or y for a planar image, got {axis!r}")
            planar.append(mask.shape[2] == 1)
            value, it, res, _ = _effective(mask, _spacing(image), "xyz".index(axis), physics,
                                           contrast, tol, _bound(maxiter, physics, mask.shape),
                                           device, solver)  # fmt: skip
            samples[row, j] = value
            iterations, residual = max(iterations, int(it)), max(residual, float(res))
    return samples, iterations, residual, planar


def percolation_exponent(
    generate: Callable[[float, np.random.Generator, int], object],
    sizes,
    threshold: float,
    *,
    threshold_stderr: float = 0.0,
    fractions=(),
    physics: str = "thermal",
    phase: int = 1,
    axis: str = "x",
    contrast: float = 1e-8,
    nu: float | None = None,
    realizations: int = 16,
    tol: float | None = None,
    maxiter: int | None = None,
    device: str = "auto",
    solver: str = "fft",
    seed: SeedLike = None,
) -> PercolationExponent:
    """Critical exponent of the conductivity (t) or of the elastic moduli (f) of the
    percolating phase of a family of microstructures, by the finite-size scaling of the
    property at its percolation threshold, and optionally by a sweep above it.

    Near the threshold p_c of the infinite medium the conductivity of the phase ``phase``
    vanishes as ``(p - p_c)**t``, its elastic moduli as ``(p - p_c)**f``, and the
    correlation length diverges as ``|p - p_c|**(-nu)``; at p_c, in boxes L across, the
    mean property scales as ``L**(-t/nu)`` (``L**(-f/nu)``; Derrida & Vannimenus 1982,
    Stauffer & Aharony 1994). ``generate(fraction, rng, size)`` returns an image ``size``
    voxels per edge (one voxel along z for a planar one), as for :func:`percolation_scaling`
    (:func:`family_from_config` with ``sized=True`` for the model of a configuration); for
    each of the ``sizes`` (at least 3), the ``realizations`` are drawn at the ``threshold``,
    each from a seed of its own derived from ``seed``, the size and its number, and their
    phase conducts (conductivity 1) or bears (Young's modulus 1, Poisson's ratio 0.3) while
    the other phases have ``contrast`` times its properties (void at 0). The periodic FFT
    solver (:mod:`morphogen.homogenization.fft`; ``device``: ``"cpu"``, ``"gpu"`` or
    ``"auto"``, the GPU when there is one; ``tol`` and ``maxiter`` of its conjugate
    gradients, ``tol`` by default 1e-8, and 1e-10 for the voxels factorized
    (:data:`SOLVER_TOLS`), ``maxiter`` by default that of the solver, 20 L + 1000 for L the largest
    dimension, for the conductivity and 50 L + 1000 for the moduli, whose solutions near
    the rigidity threshold take up to 45 L), or (``solver="fe"``) the same trilinear voxels
    assembled and solved by conjugate gradients preconditioned by the sparse factorization
    of their matrix, on the CPU (:mod:`morphogen.homogenization.voxel_fe`: the same
    solutions in a few iterations near the rigidity threshold, void around; with a contrast
    every voxel is assembled; void around, on the voxels that bear alone), gives the
    conductivity along ``axis`` (``physics="thermal"``) or the modulus of a uniaxial strain
    along it, ``C_xxxx`` along x (``"elastic"``; all the moduli share f, and Young's modulus
    would take six solutions), from the energy of the solution, the mean of σ : ε
    (stationary: its error is of second order, where the stress average strays with the
    round-off of the element matrices on floppy images, see :func:`rigidity_threshold`).
    The logarithms of the means are fitted
    against ``ln L`` by least squares weighted by their inverse variances (the standard
    errors of the means over their values), the slope times ``nu`` being the exponent:
    ``nu`` is by default that of random percolation, 0.8765 in 3-D and 4/3 in 2-D
    (:func:`percolation_scaling`). Its standard error comes from those of the means, scaled
    by the scatter of the residuals when it exceeds them (corrections to scaling); with a
    ``threshold_stderr``, the realizations are also computed at the threshold less and plus
    it (the same images: common random numbers) and ``threshold_error`` is half the change
    of the exponent between them, the error that the threshold's own leaves.

    The finite-size scaling at p_c is the method of the literature (Derrida & Vannimenus
    1982; Gingold & Lobb 1990: t = 2.003 +/- 0.047 in 3-D; Grassberger 1999: t/nu = 0.9745
    +/- 0.0015 in 2-D): every box contributes at its threshold, where the property is
    critical, and the corrections are those of L alone. A sweep in ``p - p_c`` needs boxes
    much larger than the correlation length at every fraction of its window, which ends
    where the critical region does: on images of a few hundred voxels across the window is
    narrow and its exponent drifts with it (Kirkpatrick's 1.6 +/- 0.1 in 3-D, 1973, came
    from such sweeps). ``fractions`` (at least 3,
    above the threshold plus its standard error) add one on every size: ``sweep_exponents``
    are the slopes of ``ln <P>`` against ``ln(p - p_c)`` of each size, with their standard
    errors and the change that the threshold's error makes (refitted at p_c +/- its
    standard error), for comparison.

    The threshold must be that of the solver. Its trilinear voxels join two voxels through
    the nodes they share, a face, an edge or a corner: the conductivity percolates through
    26 neighbours (8 in 2-D), so that its threshold is that of :func:`percolation_scaling`
    with ``connectivity=26`` (wrapping around the periodic box, as the solver, or by the
    fire, faster on large planar images: the criteria extrapolate to the same p_c), below
    that of the faces (for sites of the cubic lattice 0.0976 against 0.3116; for Boolean
    spheres of 3 voxels of radius, 0.186 against 0.245). In elasticity a shared corner is a
    ball joint and an edge a hinge, about which the voxels turn freely: the moduli vanish
    where the joined voxels become rigid, between the thresholds of 26 and 6 neighbours (8
    and 4 in 2-D), at the threshold that :func:`rigidity_threshold` finds from the moduli
    themselves. The values of the realizations that do not percolate, of the order of
    the contrast, are known to the tolerance of the solver only. The contrast shifts the property
    by about ``contrast * (L / l)**((t + s)/nu)`` of itself at p_c (Efros & Shklovskii
    1976, Straley 1976; ``l`` the size of the grains, s the exponent of a superconducting
    phase, about 0.73 in 3-D and 1.30 in 2-D), and the sweep within ``contrast**(1/(t +
    s))`` of p_c: at the default 10⁻⁸, below 10⁻³ of the property and 0.002 in fraction for
    boxes up to 40 grains across in 3-D and 300 in 2-D. The threshold weighs most: periodic
    Boolean discs of 4 pixels of radius (64 to 512 pixels, 48 realizations; p_c = 0.5997
    +/- 0.0054 through 8 neighbours) give t = 1.13 +/- 0.16 +/- 0.21 (the second error that
    of the threshold, dt/dp_c about -40), spheres of 3 voxels of radius (24 to 96 voxels,
    32 realizations; 0.1861 +/- 0.0040 through 26) t = 2.37 +/- 0.36 +/- 0.32 (dt/dp_c
    about -80), against 1.30 and 2.0. At their rigidity thresholds (by
    :func:`rigidity_threshold`, void around, ``solver="fe"``, the energies at the tolerance
    1e-10, those that bear under 1e-9 confirmed at 1e-11), the sites of the square lattice
    (0.4617 +/- 0.0004; 16 to 256 pixels, 200 to 400 realizations) give f/nu = 2.26 +/-
    0.07 +/- 0.05 (f = 3.01 with nu = 4/3), the slope rising with the box (1.55 between 16²
    and 24² pixels, 2.93 +/- 0.68 between 192² and 256²; 2.46 +/- 0.06 +/- 0.07 from 32²,
    2.63 +/- 0.07 +/- 0.08 with a correction to scaling L**-1), and those of the cubic
    lattice (0.1304 +/- 0.0002; 8 to 80 voxels, 400 to 20 realizations) f/nu = 4.48 +/-
    0.09 +/- 0.09 (f = 3.93 with nu = 0.8765; 4.73 +/- 0.17 +/- 0.16 from 16³, 4.73 to 4.85
    with a correction), 1.4 to 2.0 combined errors above the bond-bending f/nu of about 4.3
    and far from the 2.1 of central forces on the bcc lattice (Arbabi & Sahimi 1993), and
    3 to 8 below the 2-D bond-bending 2.97 (f = 3.96), still rising there. At the
    thresholds of the first measurements (0.4627; 0.12904 by the energies at 1e-8, where
    floppy images bore) the same moduli give 2.15 and 5.04, the latter steepening to 6.15
    from 24³ as that threshold lay low (f = 2.88 +/- 0.12 +/- 0.23 and 4.58 +/- 0.19 +/-
    0.07 then, with 64 and 128 realizations). Boolean discs of 4 pixels of radius (0.608
    +/- 0.006, 32² to 256² pixels) give f = 3.20 +/- 0.19 +/- 0.24 and spheres of 3 voxels
    (0.202 +/- 0.004, 24³ to 48³ voxels) 4.65 +/- 0.59 +/- 0.05 (at 1e-8), those of the
    sites within the errors: the voxels do not resolve the necks narrower than a voxel. By
    their thresholds, distinct from those of the connectivity, and the exponents of the
    widths of the realizations' thresholds (nu = 1.14 +/- 0.02 in 2-D, 0.76 +/- 0.03 in
    3-D), the voxels' rigidity is a percolation of its own, as for central forces, whose
    f/nu is in 3-D that of bond bending. At the threshold the
    contrast 10⁻⁸ raises the mean moduli by 4·10⁻⁵ of those of void (10⁻⁶: 0.4 %, 10⁻⁴:
    30 % on 24² pixels, by the FFT solver), the realizations
    that do not bear taking it times 14 to 58. For instance, the conductivity
    of Boolean spheres (the inverted Swiss cheese), 3 voxels per radius::

        from morphogen.domain import Domain
        from morphogen.models import BooleanSpheres

        def spheres(fraction, rng, size):
            model = BooleanSpheres(radius=1.0, volume_fraction=fraction)
            domain = Domain(size=size / 3.0, periodic=True)
            return model.generate(domain, seed=rng).voxelize(size)

        sizes = [24, 32, 48, 64]
        pc = percolation_scaling(spheres, sizes, 0.1, 0.4, connectivity=26, periodic=True,
                                 realizations=49, tol=0.002, seed=0)
        result = percolation_exponent(spheres, sizes, pc.threshold,
                                      threshold_stderr=pc.stderr, seed=1)
    """
    if physics not in _PHYSICS:
        raise ValueError(f"physics: one of {', '.join(_PHYSICS)}, got {physics!r}")
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis: x, y or z (x or y for a planar image), got {axis!r}")
    _check_solver(solver)
    tol = SOLVER_TOLS[solver] if tol is None else tol
    values = _check_sizes(sizes, 3, "the exponent")
    threshold, sigma = float(threshold), float(threshold_stderr)
    if not np.isfinite(threshold):
        raise ValueError(f"threshold: expected a finite fraction, got {threshold!r}")
    if not (np.isfinite(sigma) and sigma >= 0.0):
        raise ValueError(f"threshold_stderr: expected a nonnegative error, got {sigma!r}")
    sweep = np.unique(np.asarray(fractions, dtype=float).ravel())
    if sweep.size and (
        sweep.size < 3 or not np.all(np.isfinite(sweep)) or not np.all(sweep > threshold + sigma)
    ):
        raise ValueError(
            f"fractions: at least 3 above the threshold plus its standard error "
            f"({threshold + sigma:g}), got {fractions!r}"
        )
    if not (np.isfinite(contrast) and 0.0 <= contrast < 1.0):
        raise ValueError(f"contrast: expected 0 <= contrast < 1, got {contrast!r}")
    if isinstance(realizations, bool) or int(realizations) != realizations or realizations < 2:
        raise ValueError(f"realizations: expected an integer of at least 2, got {realizations!r}")
    if nu is not None and not nu > 0.0:
        raise ValueError(f"nu: expected a positive exponent, got {nu!r}")
    if not tol > 0.0:
        raise ValueError(f"tol: expected a positive tolerance, got {tol!r}")
    base, count = _base_seed(seed), int(realizations)
    shifted = (threshold - sigma, threshold + sigma) if sigma > 0.0 else ()
    grid = np.unique([threshold, *shifted, *sweep.tolist()])
    samples = np.empty((len(values), count, len(grid)))
    iterations, residual, planar = 0, 0.0, []
    for k, size in enumerate(values.tolist()):
        samples[k], it, res, flat = _samples(generate, size, range(count), grid, base, phase,
                                             axis, physics, float(contrast), tol, maxiter,
                                             device, solver)  # fmt: skip
        iterations, residual = max(iterations, it), max(residual, res)
        planar.extend(flat)
    if nu is None:
        nu = NU_2D if all(planar) else NU_3D
    nu = float(nu)
    means = samples.mean(axis=1)
    errors = samples.std(axis=1, ddof=1) / np.sqrt(count)

    def column(fraction):
        j = int(np.searchsorted(grid, fraction))
        return means[:, j], errors[:, j]

    slope, amplitude, residuals, slope_stderr = _power_fit(values, *column(threshold))
    threshold_error = 0.0
    if shifted:
        lo, hi = (_power_fit(values, *column(f))[0] for f in shifted)
        threshold_error = 0.5 * nu * abs(hi - lo)
    sweeps = np.empty((len(values) if sweep.size else 0, 3))
    for k, row in enumerate(sweeps):
        y, e = means[k, np.searchsorted(grid, sweep)], errors[k, np.searchsorted(grid, sweep)]
        row[0], _, _, row[1] = _power_fit(sweep - threshold, y, e)
        row[2] = 0.0
        if shifted:
            lo, hi = (_power_fit(sweep - f, y, e)[0] for f in shifted)
            row[2] = 0.5 * abs(hi - lo)
    at = column(threshold)
    return PercolationExponent(
        physics=physics,
        threshold=threshold,
        exponent=-nu * slope,
        stderr=nu * slope_stderr,
        threshold_error=threshold_error,
        ratio=-slope,
        amplitude=amplitude,
        nu=nu,
        sizes=values,
        means=at[0],
        mean_stderr=at[1],
        residuals=residuals,
        fractions=grid,
        values=means,
        value_stderr=errors,
        sweep=sweep,
        sweep_exponents=sweeps[:, 0],
        sweep_stderr=sweeps[:, 1],
        sweep_threshold_error=sweeps[:, 2],
        contrast=float(contrast),
        realizations=count,
        iterations=iterations,
        residual=residual,
        samples=samples,
        solver=solver,
    )


# -- the rigidity threshold: where the elastic moduli of the phase vanish ---------------------

#: The neighbours through which the voxels of the FFT solvers join, at the nodes they share
#: (26: a face, an edge or a corner; 8 in 2-D), and through which they are welded, rigidly
#: (6: a face; 4 in 2-D: an edge), for 3-D and for planar images
_JOINED, _WELDED = {False: 26, True: 8}, {False: 6, True: 4}


@dataclass(frozen=True)
class RigidityThreshold(PercolationScaling):
    """Finite-size scaling of the rigidity threshold of a phase, the fraction from which its
    elastic moduli do not vanish (:func:`rigidity_threshold`).

    The fields of :class:`PercolationScaling`, the probabilities of ``results[k]`` being the
    shares of the realizations of the boxes ``sizes[k]`` that bear at its fractions;
    ``bearing[k][j, i]`` tells whether realization ``i`` bears at the fraction ``j`` and
    ``moduli[k][j, i]`` is its modulus over the Young's modulus of the phase, from the energy
    of its solution (NaN where the image was decided without the solver); ``cut`` is the
    modulus above which a realization bears, ``contrast`` the properties of the other
    phases over those of the phase (void at 0), ``floppy`` the largest modulus solved at or
    under the cut and ``rigid`` the smallest of those that bear (NaN without such
    solutions; the cut may move between them without changing a decision); ``solutions``
    is the number of solutions of the ``solver`` (``"fft"`` or ``"fe"``) and ``decided``
    that of the images decided without one, ``unresolved`` that of the solutions above the
    cut whose stress average strays from their energy by more than half of it (decided by
    the round-off of their energies for ``"fe"``, by their stress averages for ``"fft"``),
    ``confirmed`` that of the energies that bore under 1000 times the cut, decided again by
    a solution at the floor of the residual (``"fe"``), ``iterations`` and ``residual`` the
    largest number of iterations and relative residual of the solver.
    """

    contrast: float = 0.0
    cut: float = 1e-8
    floppy: float = float("nan")
    rigid: float = float("nan")
    solutions: int = 0
    decided: int = 0
    iterations: int = 0
    residual: float = 0.0
    bearing: tuple[np.ndarray, ...] = field(default=(), repr=False)
    moduli: tuple[np.ndarray, ...] = field(default=(), repr=False)
    solver: str = "fft"
    unresolved: int = 0
    confirmed: int = 0


#: A solution resolves its modulus when its stress average lies within this fraction of its
#: energy (:func:`_bears`)
_RESOLVED = 0.5
#: An energy is resolved when those of the element matrices perturbed at their round-off lie
#: within this fraction of it (:func:`_bears`)
_STABLE = 0.01
#: An energy that bears under this many times the cut is confirmed by a solution of the
#: voxels factorized at the tolerance :data:`_FLOOR` (:func:`_bears`): at 1e-10 the floppy
#: images whose soft modes the iterations leave unresolved keep energies of 1e-12 to 3e-11
#: whose stress averages agree with them, and even their perturbations
_CONFIRM = 1e3
#: The tolerance of the confirmations, near the floor of the residual (3e-12 to 1.2e-11)
_FLOOR = 1e-11


def _decision(args, value, average, cut, solver, tally) -> bool:
    """Whether a solution of :func:`_effective` (``args``) of energy ``value`` and stress
    average ``average`` bears (:func:`_bears`); ``tally`` counts."""
    if not value > cut:
        tally["floppy"] = max(tally["floppy"], value)
        return False
    if abs(average - value) <= _RESOLVED * value:
        return True
    tally["unresolved"] += 1
    if solver == "fe":
        energies = [value, *(_effective(*args, rounding=k)[0] for k in (1, 2))]
        return max(energies) - min(energies) <= _STABLE * max(energies)
    return bool(average > cut)


def _bears(mask, spacing, axis, known, contrast, cut, tol, maxiter, device, tally,
           solver="fft"):  # fmt: skip
    """Whether the phase ``mask`` bears a uniaxial strain along ``axis`` with a modulus
    above ``cut`` (:func:`rigidity_threshold`), and that modulus (NaN when decided without
    the solver): not when none of its components wraps around the box along the axis
    through the nodes of its voxels (they translate freely), when one of its components
    joined by faces does (a rigid cluster), else as an image of ``known`` (the packed
    masks of the other images of the realization and their decisions) that bears and that
    it contains, or that does not and that contains it (the moduli grow with the stiffness
    of the voxels), else by the ``solver``: the modulus is the energy of the solution, an
    upper bound, so that it does not bear at or under the cut; above it, it bears when the
    stress average, linear in the error of the solution, agrees with the energy within
    :data:`_RESOLVED` of it, and otherwise (a floppy image whose mechanisms take large
    displacements, which the round-off of the element matrices strains) when two more
    solutions, the element matrices perturbed at their round-off, leave the energy within
    :data:`_STABLE` of itself (``"fe"``), or when the stress average exceeds the cut
    (``"fft"``, whose iterations stop short of the round-off). An energy of the voxels
    factorized that bears under :data:`_CONFIRM` times the cut is decided again, so, by a
    solution at the tolerance :data:`_FLOOR` (when ``tol`` is looser), whose energy is the
    modulus: the soft modes of a floppy image that the iterations have not resolved keep
    energies near the cut whose stress averages and perturbations agree with them;
    ``tally`` counts."""
    flat = mask.shape[2] == 1
    decided = None
    if not _percolates(mask, 1, axis, _JOINED[flat], True):
        decided = False
    elif _percolates(mask, 1, axis, _WELDED[flat], True):
        decided = True
    packed = np.packbits(mask)
    for other, bears in known if decided is None else ():
        if (not np.any(other & ~packed)) if bears else (not np.any(packed & ~other)):
            decided = bears
            break
    if decided is not None:
        tally["decided"] += 1
        return decided, np.nan
    args = (mask, spacing, "xyz".index(axis), "elastic", contrast, tol,
            _bound(maxiter, "elastic", mask.shape), device, solver)  # fmt: skip
    value, it, res, average = _effective(*args)
    tally["solutions"] += 1
    tally["iterations"] = max(tally["iterations"], int(it))
    tally["residual"] = max(tally["residual"], float(res))
    bears = _decision(args, value, average, cut, solver, tally)
    if bears and solver == "fe" and value < _CONFIRM * cut and tol > _FLOOR:
        tally["confirmed"] += 1
        args = (*args[:5], _FLOOR, *args[6:])
        value, it, res, average = _effective(*args)
        tally["iterations"] = max(tally["iterations"], int(it))
        bears = _decision(args, value, average, cut, solver, tally)
    if bears:
        tally["rigid"] = min(tally["rigid"], value)
    known.append((packed, bears))
    return bears, value


def _tally() -> dict:
    """The counts of :func:`_bears`."""
    return {"solutions": 0, "decided": 0, "iterations": 0, "residual": 0.0, "floppy": -np.inf,
            "rigid": np.inf, "unresolved": 0, "confirmed": 0, "planar": []}  # fmt: skip


def _rigidity_size(generate, size, lower, upper, tol, realizations, base, phase, axis,
                   contrast, cut, solver_tol, maxiter, device, tally, solver="fft"):  # fmt: skip
    """The bisection of :func:`rigidity_threshold` on the boxes ``size`` voxels across, the
    realizations seeded from ``(base, size)``: its :class:`PercolationThreshold`, whether
    each realization bears at its fractions (rows) and their moduli; ``tally`` counts."""
    sub = np.random.SeedSequence(base.entropy, spawn_key=(*base.spawn_key, size))
    seeds = _bisection(lower, upper, tol, realizations, sub)
    known = [[] for _ in seeds]  # the images solved of each realization, packed
    computed: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def probability(fraction):
        bearing, moduli = np.zeros(len(seeds), dtype=bool), np.full(len(seeds), np.nan)
        for i, s in enumerate(seeds):
            image = generate(fraction, np.random.default_rng(s), size)
            mask = _mask(image, phase)
            if axis not in _axes(mask):
                raise ValueError(f"axis: x or y for a planar image, got {axis!r}")
            tally["planar"].append(mask.shape[2] == 1)
            bearing[i], moduli[i] = _bears(mask, _spacing(image), axis, known[i], contrast, cut,
                                           solver_tol, maxiter, device, tally,
                                           solver)  # fmt: skip
        computed[fraction] = bearing, moduli
        return float(bearing.mean())

    # the phase grows with the fraction or shrinks, and so does the share that bears: the
    # bounds are tried last, the images far above the threshold costing most to factorize
    shares = [_mask(generate(f, np.random.default_rng(seeds[0]), size), phase).mean()
              for f in (lower, upper)]  # fmt: skip
    result = _bisect(probability, lower, upper, tol, len(seeds), "bearing",
                     rising=bool(shares[1] >= shares[0]))  # fmt: skip
    rows = [computed[f] for f in result.fractions.tolist()]
    return result, np.array([b for b, _ in rows]), np.array([m for _, m in rows])


def _rigidity_fit(sizes, results, bearing, moduli, nu, fit_nu, contrast, cut, tally,
                  solver="fft"):  # fmt: skip
    """The :class:`RigidityThreshold` of the bisections ``results`` of the ``sizes``."""
    if nu is None:
        nu = NU_2D if all(tally["planar"]) else NU_3D
    thresholds = np.array([r.threshold for r in results])
    brackets = np.array([r.bracket for r in results])
    widths = brackets[:, 1] - brackets[:, 0]
    pc, a, theta, residuals, errors = _scaling_fit(
        np.asarray(sizes, dtype=float), thresholds, widths, float(nu), fit_nu
    )
    floppy, rigid = tally["floppy"], tally["rigid"]
    return RigidityThreshold(
        sizes=np.asarray(sizes),
        thresholds=thresholds,
        brackets=brackets,
        threshold=pc,
        stderr=float(errors[0]),
        amplitude=a,
        nu=1.0 / theta if fit_nu else float(nu),
        nu_stderr=float(errors[2]) / theta**2 if fit_nu else None,
        residuals=residuals,
        results=tuple(results),
        contrast=float(contrast),
        cut=float(cut),
        floppy=float(floppy) if np.isfinite(floppy) else np.nan,
        rigid=float(rigid) if np.isfinite(rigid) else np.nan,
        solutions=tally["solutions"],
        decided=tally["decided"],
        iterations=tally["iterations"],
        residual=tally["residual"],
        bearing=tuple(bearing),
        moduli=tuple(moduli),
        solver=solver,
        unresolved=tally["unresolved"],
        confirmed=tally["confirmed"],
    )


def rigidity_threshold(
    generate: Callable[[float, np.random.Generator, int], object],
    sizes,
    lower: float = 0.0,
    upper: float = 1.0,
    *,
    nu: float | None = None,
    fit_nu: bool = False,
    phase: int = 1,
    axis: str = "x",
    cut: float | None = None,
    contrast: float = 0.0,
    realizations: int = 9,
    tol: float = 0.005,
    solver_tol: float | None = None,
    maxiter: int | None = None,
    device: str = "auto",
    solver: str = "fft",
    seed: SeedLike = None,
) -> RigidityThreshold:
    """Rigidity threshold of a phase of a family of microstructures: the fraction from which
    its elastic moduli by the FFT solver do not vanish, the other phases void, by the
    finite-size scaling of the fractions at which half the realizations of boxes of each of
    the ``sizes`` bear.

    The trilinear voxels of the FFT solvers (:mod:`morphogen.homogenization.fft`) join
    through the nodes they share, but a corner is a ball joint and an edge a hinge (in 2-D a
    corner is a hinge), about which they turn freely; only a face (an edge in 2-D) welds
    them. The clusters welded by their faces are rigid bodies pinned together at the
    corners and edges between them, and the moduli of the phase are not zero when these
    bodies and joints make a rigid network wrapping around the periodic box: from a
    fraction between the thresholds of percolation through 26 and 6 neighbours (8 and 4 in
    2-D), which no connectivity gives (the rigidity percolation of central-force networks,
    of bodies and joints, whereas the bond-bending forces of lattice models make every
    connected cluster rigid). A realization bears when its modulus of a uniaxial strain
    along ``axis`` (C_xxxx along x; Young's modulus 1 and Poisson's ratio 0.3), the other
    phases at ``contrast`` times the properties of the phase (void at the default 0),
    exceeds ``cut``; the fraction p½(L) at which half the realizations of the boxes L
    voxels across bear is bisected as in :func:`percolation_threshold` (``realizations``,
    ``tol``; common random numbers: for a family growing with the fraction, the median of
    the fractions from which the realizations bear), each size from a seed of its own
    derived from ``seed`` (the realizations of :func:`percolation_scaling` with the same
    seed, which bear from fractions between those where they join through 26 and 6
    neighbours), and the p½(L) are extrapolated as ``p½(L) = p_c + a * L**(-1/nu)`` by the
    fit of :func:`percolation_scaling` (``nu``, ``fit_nu``; by default the exponent of
    random percolation, 4/3 in 2-D, 0.8765 in 3-D; that of the rigidity percolation of
    central-force networks is 1.21 +/- 0.06 in 2-D, Jacobs & Thorpe 1996).

    The criterion is that of the moduli themselves, from the energy of each solution, the
    mean of σ : ε, which equals the stress average σ_xx at the exact solution but is
    stationary: an upper bound whose error is of second order in that of the solution, that
    of the stress average being of first order. A realization does not bear when its energy
    is at or under ``cut``; above it, it bears when its stress average agrees with its
    energy within half of it (the modulus resolved), and otherwise (``unresolved``) by two
    more solutions with the element matrices perturbed at their round-off (2.2e-16 of their
    entries), bearing when the three energies agree within 1 % (``solver="fe"``), or by its
    stress average (``"fft"``, whose iterations stop short of the round-off). The floppy
    images of large boxes need it: their mechanisms take displacements of 1e3 to 7e5 for a
    unit strain, which the round-off of the element matrices, the same in every voxel,
    strains, so that their stress averages stray to -1e-10 to -6e-7 at any tolerance and
    their energies, 1e-20 to 2e-7, scatter by 17 to 50 % when the element matrices are
    perturbed, while those of the images that bear move by 1.4e-3 at most and agree with
    their stress averages within 30 %. The criterion wants solutions that have resolved the
    soft modes of the floppy images: the voxels factorized iterate to ``solver_tol``, by
    default 1e-10, and an energy that bears under 1000 times the cut is decided again by a
    solution at 1e-11, near the floor of the residual (``confirmed``). At 1e-8 one to three
    iterations leave floppy images at energies of 1e-13 to 2e-8 whose stress averages
    agree with them: 23 of 32 realizations of 32³ to 80³ voxels of the sites of the cubic
    lattice bore from fractions lower by 0.0003 to 0.003 than at 1e-10, where such energies
    fall to 1e-21 to 1e-14; at 1e-10 a few keep 1e-12 to 3e-11, their stress averages and
    perturbations agreeing (the iterates, not the mechanisms, carry them), and fall under the
    cut at 1e-11 or take stress averages of the opposite sign, while a modulus of 2.0e-12 on
    64³ voxels stays to three digits. Void around, the moduli of the realizations that bear
    then spread down to 1.2e-12 without a gap (77 of the 5,308 solutions that bear on 8³ to
    80³ voxels of the sites near their thresholds lie between 1e-12 and 1e-10). The cut is
    the round-off of the moduli of the solver, by default (:data:`CUTS`) 1e-8 for the FFT
    solver and 1e-12 for the voxels factorized, whose energies resolve the moduli down to
    it; p½(L) lies where the moduli of half the realizations exceed it, and a cut too high
    biases it up as the moduli come down with the box. With a contrast δ the realizations
    that do not bear take δ times an enhancement that rises
    steeply as they approach their own rigidity (up to 650 on 32² pixels and 4,400 on 64²
    for the sites of the square lattice), while those that have just become rigid bear as
    little as 2e-6 (32²): a cut that is a multiple of δ must exceed the one and stay below
    the other, which δ = 1e-8 (the default of :func:`percolation_exponent`) does not allow;
    ``cut`` must exceed ``contrast``. The median of the realizations' fractions is robust
    where the moduli spread over decades and needs no exponent, where the crossing of the
    curves L^(f/nu) <C(p)> presupposes f, the exponent sought, from means that scatter by
    tens of percent.

    The images are decided without the solver when they can be: the phase does not bear
    when none of its components wraps around the box along the axis through 26 neighbours
    (8 in 2-D: they translate freely), it bears when a component joined by faces (6; 4 in
    2-D) wraps (a rigid body), and, the moduli growing with the stiffness of the voxels, an
    image that contains an image of the same realization that bears bears and one contained
    in an image that does not does not (the images of a family growing with the fraction).
    The others are solved by conjugate gradients of tolerance ``solver_tol`` (by default
    :data:`SOLVER_TOLS`: 1e-8 for the FFT solver, 1e-10 for the voxels factorized) and at most
    ``maxiter`` iterations (by default 50 L + 1000, L the largest dimension: near the
    threshold the solutions of the FFT solver take up to 45 L, more than its 20 L + 1000),
    by the FFT solver on ``device`` (``solver="fft"``; ``"cpu"``, ``"gpu"`` or ``"auto"``)
    or (``solver="fe"``) by the same trilinear voxels assembled on those of the phase and
    preconditioned by the sparse factorization of their matrix, on the CPU
    (:mod:`morphogen.homogenization.voxel_fe`): the same moduli, to the tolerance, in one to
    three iterations, where the Fourier preconditioner of a homogeneous medium spreads the
    displacements of the voxels over the void around them and the clusters pinned at their
    corners turn about them with a vanishing energy, on the voxels that bear alone
    (:func:`~morphogen.homogenization.voxel_fe.load_bearing`: the same moduli, 15 to 25 %
    fewer voxels). For a family whose phase grows with the fraction (shrinks), the bounds
    are tried last and only when the bracket reaches them: the images far above the
    threshold cost most to factorize (on 48³ voxels of sites 122 s at 0.2096 against 1.5 s
    at 0.1307). The result counts the ``solutions``, the images ``decided`` and the energies
    ``unresolved``, gives the largest modulus solved under the cut (``floppy``) and the
    smallest of those that bear (``rigid``), and the shares, decisions and moduli of the
    fractions tried.

    For the sites of the square lattice (400 realizations of 16² to 128² pixels and 300 of
    192² and 256², each bisected to its own threshold, its images nested, ``solver="fe"``,
    the cut 1e-12 and the tolerance 1e-10), p½(L) = 0.4397, 0.4564, 0.4547, 0.4559,
    0.4595, 0.4611, 0.4614, 0.4617 and 0.4612, flat from 96² on, extrapolate to p_c =
    0.4617 +/- 0.0004 with the exponent of the fit free (1.5 +/- 0.2: ``nu`` 4/3 does not
    fit them, a chi-square of 34 for 7 degrees of freedom), and the widths of the
    realizations' thresholds fall as L**(-1/nu) with nu = 1.14 +/- 0.02 (central-force
    rigidity percolation in 2-D: 1.21 +/- 0.06, Jacobs & Thorpe 1996), the share that bears
    at p_c staying at about one half from 96² on; a quarter of the way from the threshold
    through 8 neighbours, 0.4073, to that through 4, 0.5927. The first measurement, 96
    realizations bisected to 0.0035 by the stress averages at the cut 1e-8, gave 0.4627 +/-
    0.0016 (32 realizations by the FFT solver: 0.4647 +/- 0.0056, the decisions the same).
    For the sites of the cubic lattice (1,725 realizations of 8³ to 80³ voxels, each
    bisected to its own threshold, at the tolerance 1e-10 with the confirmations), p½(L) =
    0.1356, 0.1338, 0.1332, 0.1316, 0.1321, 0.1314, 0.1313, 0.1309 and 0.1308 extrapolate to
    p_c = 0.1304 +/- 0.0002 (``nu`` 0.8765; 0.1300 +/- 0.0003 from 32³, the means 0.1303
    +/- 0.0001; the shares of 48³ to 80³ voxels cross at 0.1302, a third of the
    realizations bearing at p_c whatever the box; the widths give nu = 0.76 +/- 0.03),
    between the thresholds through 26 and 18 neighbours, 0.0975 and 0.1383, far below that
    through the faces, 0.3116: the corners alone hold the voxels. At the tolerance 1e-8,
    where floppy images bear, the first measurement (192 realizations of 8³ to 48³ voxels)
    gave 0.12904 +/- 0.00029, and 0.1307 +/- 0.0009 by the stress averages of 96
    realizations of 8³ to 32³. Boolean discs of 4
    pixels of radius and spheres of 3 voxels (64 realizations, 32² to 256² pixels and 24³ to
    48³ voxels) bear from 0.608 +/- 0.006 and 0.202 +/- 0.004, between their thresholds
    through 8 and 4 neighbours (0.596 and 0.624) and through 26 and 6 (0.182 and 0.236),
    their p½(L) near those through the corners: their grains meet at several nodes, which
    brace them, where the sites meet at one. With :func:`percolation_exponent`::

        def squares(fraction, rng, size):
            return rng.random((size, size, 1)) < fraction

        sizes = [16, 24, 32, 48, 64, 96, 128, 192, 256]
        rigid = rigidity_threshold(squares, sizes, 0.38, 0.58, realizations=96, tol=0.0035,
                                   solver="fe", seed=2026)
        result = percolation_exponent(squares, sizes, rigid.threshold,
                                      threshold_stderr=rigid.stderr, physics="elastic",
                                      contrast=0.0, realizations=64, solver="fe", seed=2027)

    (the measurements above bisected each realization to its own threshold instead, the
    images nested, by these decisions).
    """
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis: x, y or z (x or y for a planar image), got {axis!r}")
    _check_solver(solver)
    cut = CUTS[solver] if cut is None else cut
    solver_tol = SOLVER_TOLS[solver] if solver_tol is None else solver_tol
    need, fitted = (4, "p_c, a and nu") if fit_nu else (3, "p_c and a")
    values = _check_sizes(sizes, need, fitted)
    if nu is not None and not nu > 0.0:
        raise ValueError(f"nu: expected a positive exponent, got {nu!r}")
    if not (np.isfinite(contrast) and 0.0 <= contrast < 1.0):
        raise ValueError(f"contrast: expected 0 <= contrast < 1, got {contrast!r}")
    if not (np.isfinite(cut) and contrast < cut < 1.0):
        raise ValueError(f"cut: expected a modulus between the contrast ({contrast:g}) and 1, "
                         f"got {cut!r}")  # fmt: skip
    if not solver_tol > 0.0:
        raise ValueError(f"solver_tol: expected a positive tolerance, got {solver_tol!r}")
    if maxiter is not None and (isinstance(maxiter, bool) or int(maxiter) != maxiter
                                or maxiter < 1):  # fmt: skip
        raise ValueError(f"maxiter: expected a positive integer, got {maxiter!r}")
    base, tally = _base_seed(seed), _tally()
    runs = [_rigidity_size(generate, size, lower, upper, tol, realizations, base, phase, axis,
                           float(contrast), float(cut), solver_tol, maxiter, device, tally,
                           solver) for size in values.tolist()]  # fmt: skip
    results, bearing, moduli = zip(*runs, strict=True)
    return _rigidity_fit(values, results, bearing, moduli, nu, fit_nu, contrast, cut, tally,
                         solver)  # fmt: skip
