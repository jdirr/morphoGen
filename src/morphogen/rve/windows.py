"""The window method for tensors: the stiffness and the conductivity of the windows of one
large periodic realization.

The realization is solved once per unit macroscopic load by the FFT solver
(:mod:`morphogen.homogenization.fft`): the six unit strains ``E^(k)`` (Voigt order 11 22 33
23 31 12, engineering shears); the three in-plane ones ``E_11, E_22, E_12`` of a planar
image (``nz = 1``: the plane stiffness, Voigt order 11 22 12); the three unit gradients of
the conductivity. The tensor of a window ``w`` is the matrix of the window means of the
mutual energy densities of these solutions,

    C^w_kl = <eps^(k) : C : eps^(l)>_w        (K^w_kl = <grad T^(k) . K grad T^(l)>_w),

integrated over the Gauss points of its voxels as the solver integrates the energy. It is
symmetric, to the bit (its upper triangle is computed, then mirrored), and positive
semi-definite: ``a . C^w a`` is the mean energy over the window of the solution
``sum_k a_k u^(k)``. Over windows tiling the box, its mean is the box's mean energy to
round-off, which Hill's lemma ``<sigma : eps> = Sigma : E`` makes the box's apparent
stiffness ``C_kl = E^(k) . <sigma^(l)>`` (that of
:func:`~morphogen.homogenization.stiffness_tensor`; the conductivity of
:func:`~morphogen.homogenization.conductivity_tensor`), to the tolerance of the solver: the
energy is stationary at the solution, its error of the second order in that of the fields,
that of the mean stresses of the first. Its diagonal is the energy estimate of the apparent
moduli (Dirrenberger et al. 2014, eqs. 64 and 66).

This definition rather than the window means of the stresses under the unit strains,
``<sigma^(l)>_w`` (Kanit et al. 2003): those tile the box as exactly, but are neither
symmetric (``C A`` at a point, ``A`` the concentration of the strains: 16 % off on random
voxels of a contrast of 10) nor positive (their symmetric part indefinite at voxels of that
contrast, in windows of 2 voxels at a contrast of 1000), and in a laminate their components
across the layers are those of the box in every window, the tractions being continuous: no
scatter, whatever the window. ``<sigma>_w <eps>_w^-1``, the
window's own apparent tensor, is neither symmetric nor tiling, and is undefined where the
mean strain of a window is singular (a rigid inclusion). In a laminate, whose strains are
uniform in each layer (``eps = A_p E``), the window tensor is ``sum_p f_p^w A_p^T C_p A_p``,
``f_p^w`` the fraction of the layers ``p`` in the window.

The windows are the cubes of ``w`` voxels that tile the image from its origin, the
remainder left out, as in :func:`morphogen.rve.scaling.window_statistics` (squares of ``w``
pixels across a planar image, whose measure is then their area). Their statistics per size
are those of the ensemble method, the windows as realizations
(:class:`~morphogen.rve.study.TensorStatistics`, ``method="windows"``): the mean ``Cbar``
and the standard deviation of each component, ``S = Cbar^-1`` and the mean of the
compliances of the windows, the RVE size of each component. A window without stiffness in
some direction (within a pore) has no compliance (NaN, as a singular realization).

Shifted tilings (``shifts > 1``, as in :func:`~morphogen.rve.scaling.window_statistics`):
the tilings shifted by the offsets ``k w // shifts`` voxels along each axis, all their
combinations, wrapping around the periodic image. They take the fields already solved,
each tiling one more pass of the kernel over the image (a few percent of a solve). Each is
a translate of the first on the periodic image, its windows of the law of the unshifted
ones: the statistics of a size are the means over all the windows of all its tilings (over
each tiling of the box, the box's tensor), the variances pooled within the tilings and the
counts of one tiling, the independent windows (:mod:`morphogen.rve.scaling`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from morphogen.homogenization.kernels import CORNERS
from morphogen.rve.scaling import _offsets, _shift_count
from morphogen.rve.study import (
    TRANSPORT_TENSORS,
    StudyOptions,
    TensorStatistics,
    _tensor_statistics,
    tensor_notation,
)

#: The tensors of the window method and the physics of their problems: the stiffness, and
#: the tensor of each transport phenomenon (``conductivity_tensor``,
#: ``electrical_conductivity_tensor``, ``permittivity_tensor``, ``permeability_tensor``,
#: ``diffusivity_tensor``)
WINDOW_TENSORS = {"stiffness": "elastic", **TRANSPORT_TENSORS}
#: The tasks of one call of the kernel (windows times their planes along x): its memory
_TASKS = 1 << 16


@njit(parallel=True, cache=True)
def _window_sums(
    nx, ny, nz, nd, mat, dmat, w, u_loc, b_gauss, weight, ext, counts, offset, first, out
):
    """``out[t, p]``: the integral over the plane ``i`` (along x) of the window ``m`` (``t =
    (m - first) ext[0] + i``) of the mutual energy density ``eps_s . D eps_u`` of the loads
    ``(s, u)``, ``s <= u``, row by row (``p``). The windows of ``ext`` voxels are numbered
    ``m = (a counts[1] + b) counts[2] + c`` on the grid of ``counts`` shifted by ``offset``
    voxels, wrapping around the periodic image; ``w[s]`` holds the periodic nodal
    fluctuation of the load ``s`` (``nd`` per node), ``u_loc[s]`` its affine part at the
    corners of a voxel, ``b_gauss`` the measures (strains, gradients) of the Gauss points,
    ``dmat[mat_e]`` the matrix of voxel ``e`` (-1: a void, no energy)."""
    n_load = w.shape[0]
    n_rows = b_gauss.shape[1]
    n_cols = b_gauss.shape[2]
    ex, ey, ez = ext[0], ext[1], ext[2]
    cy, cz = counts[1], counts[2]
    for t in prange(out.shape[0]):
        m = first + t // ex
        i = (offset[0] + (m // (cy * cz)) * ex + t % ex) % nx
        j0 = offset[1] + ((m // cz) % cy) * ey
        k0 = offset[2] + (m % cz) * ez
        nodes = np.empty(8, dtype=np.int64)
        ue = np.empty((n_load, n_cols))
        eps = np.empty((n_load, n_rows))
        sig = np.empty((n_load, n_rows))
        acc = np.zeros(out.shape[1])
        for jj in range(j0, j0 + ey):
            j = jj % ny
            for kk in range(k0, k0 + ez):
                k = kk % nz
                e = (i * ny + j) * nz + k
                g = mat[e]
                if g < 0:
                    continue
                for q in range(8):
                    a = (i + CORNERS[q, 0]) % nx
                    b = (j + CORNERS[q, 1]) % ny
                    c = (k + CORNERS[q, 2]) % nz
                    nodes[q] = (a * ny + b) * nz + c
                for s in range(n_load):
                    for q in range(8):
                        for r in range(nd):
                            ue[s, nd * q + r] = u_loc[s, nd * q + r] + w[s, nd * nodes[q] + r]
                d = dmat[g]
                for gp in range(8):
                    for s in range(n_load):
                        for r in range(n_rows):
                            v = 0.0
                            for col in range(n_cols):
                                v += b_gauss[gp, r, col] * ue[s, col]
                            eps[s, r] = v
                        for r in range(n_rows):
                            v = 0.0
                            for col in range(n_rows):
                                v += d[r, col] * eps[s, col]
                            sig[s, r] = v
                    p = 0
                    for s in range(n_load):
                        for u in range(s, n_load):
                            v = 0.0
                            for r in range(n_rows):
                                v += eps[s, r] * sig[u, r]
                            acc[p] += v
                            p += 1
        for p in range(out.shape[1]):
            out[t, p] = weight * acc[p]


@dataclass
class WindowTensors:
    """The tensors of the windows of one realization (see the module documentation).

    ``property``: ``stiffness`` or ``conductivity_tensor``; ``edges``: the window edges
    (voxels), ``volumes`` their measures (their areas across a planar image, ``dimension``
    2) and ``tensors[k]`` the tensors of the windows of edge ``edges[k]``, ``(n_k, d, d)``:
    in morphoGen's Voigt order 11 22 33 23 31 12 with engineering shears (11 22 12 in the
    plane), the axes 1 2 3 for the conductivity (:attr:`notation`). ``box``: the tensor of
    the whole realization by the same definition, the mean of the windows that tile it;
    ``apparent``: its apparent tensor from its mean stresses (fluxes), symmetrized, as
    :func:`~morphogen.homogenization.homogenize` gives it: ``box`` to the tolerance of the
    solver. ``offsets[k]``: with shifted tilings, the offsets (voxels, ``(t_k, 3)``) of the
    ``t_k`` tilings of the size ``k``, the unshifted one first, ``tensors[k]`` then the
    windows of all of them, tiling by tiling; None: one tiling of each size.
    """

    property: str
    edges: np.ndarray
    volumes: np.ndarray
    tensors: list[np.ndarray]
    box: np.ndarray
    apparent: np.ndarray
    bc: str = "periodic"
    dimension: int = 3
    offsets: list[np.ndarray] | None = None

    @property
    def key(self) -> str:
        return f"{self.property}.{self.bc}"

    @property
    def notation(self) -> str:
        """The notation of the tensors (:func:`~morphogen.rve.study.tensor_notation`)."""
        return tensor_notation(self.property, self.box)

    @property
    def tilings(self) -> np.ndarray:
        """The number of tilings of each size (1 without shifts)."""
        if self.offsets is None:
            return np.ones(len(self.edges), dtype=int)
        return np.array([len(o) for o in self.offsets])

    def statistics(self, options: StudyOptions | None = None) -> TensorStatistics:
        """Their statistics per size, the windows as realizations: the mean tensor and the
        standard deviation of each component, its inverse and the mean of the inverses of
        the windows, the RVE size of each component (``options``: those of the study,
        without bootstrap; the windows of non-finite tensors left out, as the realizations
        of :func:`~morphogen.rve.study.tensor_statistics`); with shifted tilings, the means
        over all their windows, the variances pooled within the tilings, the counts of one
        (see the module)."""
        groups = []
        for k, (v, t) in enumerate(zip(self.volumes, self.tensors, strict=True)):
            finite = np.isfinite(t).all(axis=(1, 2))
            if self.offsets is None:
                groups.append((v, t[finite]))
            else:  # the tiling of each window, the windows tiling by tiling
                count = len(self.offsets[k])
                tiles = np.repeat(np.arange(count), len(t) // count)
                groups.append((v, t[finite], tiles[finite]))
        groups = [g for g in groups if len(g[1])]
        if not groups:
            raise ValueError(f"{self.key}: no finite tensor of a window")
        return _tensor_statistics(self.property, self.bc, groups, options, method="windows",
                                  dimension=self.dimension)  # fmt: skip


def _loads(prop: str, planar: bool) -> tuple[int, ...]:
    """The unit macroscopic loads solved: the components of the strain (Voigt) or of the
    gradient."""
    if prop in TRANSPORT_TENSORS:
        return (0, 1, 2)
    return (0, 1, 5) if planar else tuple(range(6))


def _apparent(problem, prop: str, loads) -> np.ndarray:
    """The apparent tensor of the sample from its mean stresses (fluxes) under the loads,
    symmetrized (the unit solutions cached by the problem)."""
    from morphogen.homogenization import Loading, conductivity_tensor
    from morphogen.homogenization.elements import tensor_to_voigt

    if prop in TRANSPORT_TENSORS:
        return conductivity_tensor(problem)[0]
    c = np.zeros((len(loads), len(loads)))
    for col, j in enumerate(loads):
        unit = np.zeros(6)
        unit[j] = 1.0
        stress = tensor_to_voigt(problem.solve(Loading.strain(unit)).stress, strain=False)
        c[:, col] = stress[list(loads)]
    return 0.5 * (c + c.T)


def _means(problem, fields, u_loc, b_gauss, ext, offset=(0, 0, 0)) -> np.ndarray:
    """The window means of the mutual energies of the loads, ``(n_windows, n_pairs)``, over
    the windows of ``ext`` voxels that tile the image from its origin, or from the voxel
    ``offset`` around the periodic image (by chunks of windows: the memory of ``_TASKS``
    rows)."""
    nx, ny, nz = problem.shape
    counts = np.array([nx // ext[0], ny // ext[1], nz // ext[2]], dtype=np.int64)
    n_windows = int(np.prod(counts))
    n_pairs = len(fields) * (len(fields) + 1) // 2
    dmat = np.ascontiguousarray(problem.table.matrices, dtype=np.float64)
    weight = float(problem.element.weight)
    sums = np.empty((n_windows, n_pairs))
    step = max(1, _TASKS // ext[0])
    shift = np.asarray(offset, dtype=np.int64)
    for first in range(0, n_windows, step):
        stop = min(n_windows, first + step)
        out = np.zeros(((stop - first) * ext[0], n_pairs))
        _window_sums(nx, ny, nz, problem.nd, problem.mat, dmat, fields, u_loc, b_gauss, weight,
                     np.asarray(ext, dtype=np.int64), counts, shift, first, out)  # fmt: skip
        sums[first:stop] = out.reshape(stop - first, ext[0], n_pairs).sum(axis=1)
    return sums / (float(np.prod(ext)) * problem.element.volume)


def _symmetric(means: np.ndarray, d: int) -> np.ndarray:
    """The symmetric matrices ``(n, d, d)`` of their upper triangles, row by row."""
    out = np.empty((len(means), d, d))
    rows, cols = np.triu_indices(d)
    out[:, rows, cols] = means
    out[:, cols, rows] = means
    return out


def window_tensors(
    image,
    materials,
    window_sizes,
    *,
    property: str = "stiffness",
    tol: float = 1e-8,
    maxiter: int | None = None,
    grains=None,
    orientations=None,
    device: str = "cpu",
    shifts: int = 1,
) -> WindowTensors:
    """The tensors of the windows of a periodic voxel image (see the module documentation):
    ``property`` ``stiffness`` (six unit strains; three in the plane of a planar image) or
    ``conductivity_tensor`` (three unit gradients), solved once each by the FFT solver
    (``tol``, ``maxiter``, ``device``; crystals turned by the orientations of their
    ``grains``, see :func:`~morphogen.homogenization.make_problem`), then averaged over the
    windows of each edge of ``window_sizes`` (voxels): of the tiling from the origin, or
    with ``shifts > 1`` of the tilings shifted by ``k w // shifts`` voxels along each axis
    (``k < shifts``, all their combinations, wrapping around the image), from the same
    solutions. :meth:`WindowTensors.statistics` gives their statistics per size."""
    from morphogen.homogenization import make_problem
    from morphogen.image import VoxelImage

    if property not in WINDOW_TENSORS:
        raise ValueError(f"property: one of {list(WINDOW_TENSORS)}, got {property!r}")
    if not isinstance(image, VoxelImage):
        raise TypeError(f"window_tensors: a VoxelImage (FFT solver), got {type(image).__name__}")
    shifts = _shift_count(shifts, image.periodic)
    problem = make_problem(image, materials, WINDOW_TENSORS[property], "periodic", tol=tol,
                           maxiter=maxiter, grains=grains, orientations=orientations,
                           device=device)  # fmt: skip
    shape = problem.shape
    planar = shape[2] == 1
    edges = sorted({int(w) for w in window_sizes})
    for w in edges:
        if w < 1 or w > min(shape[:2] if planar else shape):
            raise ValueError(f"window size {w} does not fit in the image {shape}")
    loads = _loads(property, planar)
    element = problem.element
    if WINDOW_TENSORS[property] == "elastic":
        units = np.eye(6)[list(loads)]
        u_loc = np.array([element.affine_displacement(e) for e in units])
        b_gauss = element.b_gauss
    else:
        u_loc = np.array([element.affine_temperature(e) for e in np.eye(3)])
        b_gauss = element.g_gauss
    fields = np.ascontiguousarray(np.stack([problem._unit(j)[0] for j in loads]))
    u_loc = np.ascontiguousarray(u_loc)
    d = len(loads)
    tensors, volumes, offsets = [], [], []
    h = element.spacing
    for w in edges:
        ext = (w, w, 1 if planar else w)
        tiles = _offsets(w, shifts, planar)  # (0, 0, 0) first
        means = [_means(problem, fields, u_loc, b_gauss, ext, o) for o in tiles]
        tensors.append(_symmetric(np.concatenate(means), d))
        volumes.append(w * w * h[0] * h[1] * (1.0 if planar else w * h[2]))
        offsets.append(tiles)
    (box,) = _symmetric(_means(problem, fields, u_loc, b_gauss, shape), d)
    return WindowTensors(
        property=property,
        edges=np.array(edges),
        volumes=np.array(volumes),
        tensors=tensors,
        box=box,
        apparent=_apparent(problem, property, loads),
        dimension=2 if planar else 3,
        offsets=offsets if shifts > 1 else None,
    )
