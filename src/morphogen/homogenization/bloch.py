"""Bloch waves in periodic cells: the dispersion of elastic, acoustic and electromagnetic
waves and the band gaps of architected materials (phononic, sonic and photonic crystals).

A Bloch wave ``u(x) = û(x) exp(i k·x)``, ``û`` periodic, is a free vibration of the
periodic medium at the wave vector ``k``. On a cell meshed by tetrahedra (the finite
elements of :mod:`morphogen.homogenization.fe`, 4 or 10 nodes),
``u(x + n∘L) = u(x) exp(i k·(n∘L))`` ties each node of the upper faces to its image on
the lower ones: ``w = T(k) w_masters``, the periodic map of the fluctuations, each of
its weights times the phase of the period between the node and that master. The angular
frequencies ``ω`` are those of the Hermitian problem ``T^H K T φ = ω² T^H M T φ`` (the
stiffness ``K`` and the consistent mass ``M`` of the cell), the lowest found by
shift-invert Lanczos on its real form of twice the size (each frequency twice). Along
the edges of the irreducible Brillouin zone (Γ–X–M–Γ–R for a cubic cell) the bands give
the dispersion; a range between two bands free of waves along the path is a band gap
(complete if the path holds the extrema of the bands, as for most cells of cubic
symmetry).

**Scalar waves.** A field ``u`` that obeys ``div(a grad u) + ω² b u = 0`` has the same
Bloch problem with one unknown per node, the "stiffness" of the coefficient ``a`` and the
"mass" of ``b`` (:func:`scalar_band_structure`). :data:`SCALAR_WAVES` names those of the
materials (:func:`wave_coefficients`):

==============  =======================  ================  ================
waves           field ``u``              ``a``             ``b``
==============  =======================  ================  ================
``acoustic``    pressure of a fluid      ``1 / rho``       ``1 / K``
``shear``       antiplane displacement   ``mu``            ``rho``
``tm``          electric field ``E_z``   ``1 / mu_m``      ``epsilon``
``te``          magnetic field ``H_z``   ``1 / epsilon``   ``mu_m``
==============  =======================  ================  ================

(``rho``: density, ``K``: bulk modulus, ``mu``: shear modulus, ``epsilon``: permittivity,
``mu_m``: magnetic permeability). The antiplane shear waves and the two polarizations of
the electromagnetic waves are those of a medium invariant along the third axis (a cell
one voxel thick) for wave vectors in its plane: the photonic crystals of rods and of
holes. In three dimensions the shear waves are those of the elastic solid (``"elastic"``)
and the electromagnetic waves are vector waves, ``curl(curl H / epsilon) = ω² mu_m H``,
solved by plane waves on the voxels of an image (``"electromagnetic"``,
:mod:`morphogen.homogenization.photonic`): the photonic crystals of spheres, of rods
stacked across each other, of any morphology, and the waves that leave the plane of a
crystal of rods. A phase whose coefficient ``a`` is zero is left out, its boundary free of
flux: the rigid scatterers of a sonic crystal (``1 / rho = 0``), the perfect conductors of
``te`` waves. At long wavelengths the first band is that of the homogenized medium, ``ω² =
k . a* . k / <b>`` (:func:`long_wavelength_speeds`: Wood's law for the sound of mixtures,
the effective index of a photonic crystal).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from morphogen.homogenization.fe import FEElasticProblem, FEThermalProblem, periodic_map
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage
from morphogen.mesh.tetra import TET_EDGES, TetMesh

#: Corners of the Brillouin zone of an orthorhombic cell, in units of ``pi / L`` per axis.
POINTS = {
    "G": (0.0, 0.0, 0.0),
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
    "M": (1.0, 1.0, 0.0),
    "R": (1.0, 1.0, 1.0),
}


#: The scalar waves of the materials: the field, and the coefficients ``a`` and ``b`` of
#: ``div(a grad u) + ω² b u = 0`` (see :func:`wave_coefficients`)
SCALAR_WAVES = {
    "acoustic": ("pressure", "1 / density", "1 / bulk modulus"),
    "shear": ("antiplane displacement", "shear modulus", "density"),
    "tm": ("electric field along the axis", "1 / permeability", "permittivity"),
    "te": ("magnetic field along the axis", "1 / permittivity", "permeability"),
}

#: The scalar waves of a medium invariant along the third axis, for wave vectors in its
#: plane; in three dimensions: ``"elastic"`` and ``"electromagnetic"``
PLANE_WAVES = ("shear", "tm", "te")

#: The waves of :func:`band_structure`
WAVES = ("elastic", *SCALAR_WAVES, "electromagnetic")


def _monomial_integral(exponents) -> float:
    """``∫ λ0^a λ1^b λ2^c λ3^d dV / V`` over a tetrahedron (barycentric coordinates)."""
    num = 6.0 * math.prod(math.factorial(int(a)) for a in exponents)
    return num / math.factorial(int(sum(exponents)) + 3)


def _p2_mass() -> np.ndarray:
    """Consistent mass matrix of the 10-node tetrahedron over its volume (corners, then
    the mid-edge nodes in ``TET_EDGES`` order)."""
    shapes = []
    for i in range(4):  # λi (2 λi - 1)
        two = [0, 0, 0, 0]
        two[i] = 2
        one = [0, 0, 0, 0]
        one[i] = 1
        shapes.append({tuple(two): 2.0, tuple(one): -1.0})
    for a, b in TET_EDGES:  # 4 λa λb
        e = [0, 0, 0, 0]
        e[a] += 1
        e[b] += 1
        shapes.append({tuple(e): 4.0})
    m = np.zeros((10, 10))
    for i, j in product(range(10), range(10)):
        m[i, j] = sum(
            ci * cj * _monomial_integral(np.add(ei, ej))
            for ei, ci in shapes[i].items()
            for ej, cj in shapes[j].items()
        )
    return m


#: Consistent mass matrices of the 4- and 10-node tetrahedra over their volume.
MASS = {1: (np.ones((4, 4)) + np.eye(4)) / 20.0, 2: _p2_mass()}


def mass_matrix(mesh: TetMesh, density) -> sp.csr_matrix:
    """Consistent mass matrix (nodes x nodes) of a tetrahedral mesh, ``density`` the
    density of each element."""
    ref = MASS[mesh.order]
    e = mesh.elements
    m = e.shape[1]
    rows = np.repeat(e, m, axis=1).ravel()
    cols = np.tile(e, (1, m)).ravel()
    vals = (np.asarray(density, dtype=float) * np.abs(mesh.volumes()))[:, None] * ref.ravel()
    return sp.csr_matrix((vals.ravel(), (rows, cols)), shape=(mesh.n_nodes, mesh.n_nodes))


def voxel_tet_mesh(image: VoxelImage, phases=None) -> TetMesh:
    """Six tetrahedra per voxel of an image (the Kuhn triangulation of its grid, whose
    opposite faces match), each with the phase of its voxel; ``phases``: those kept (the
    solid of a lattice, say)."""
    from morphogen.mesh.conforming import background_mesh

    lower = np.asarray(image.origin, dtype=float)
    upper = lower + image.size
    nodes, tets, _, _ = background_mesh(image.shape, lower, upper)
    phase = np.tile(np.asarray(image.data).ravel(), 6)  # six blocks of all the cubes
    mesh = TetMesh(nodes, tets, phase, lower, upper, periodic=bool(image.periodic))
    return mesh.subset(phases) if phases is not None else mesh


def brillouin_path(size, path=("G", "X", "M", "G", "R"), points: int = 12):
    """Wave vectors along the segments between the corners ``path`` (names of
    :data:`POINTS` or points in units of ``pi / L``) of the Brillouin zone of a cell of
    edges ``size``, ``points`` per segment: the wave vectors (n x 3), their distance along
    the path and the positions and names of the corners."""
    size = np.asarray(size, dtype=float)
    names = [("Γ" if p == "G" else p) if isinstance(p, str) else str(tuple(p)) for p in path]
    corners = [
        np.asarray(POINTS[p] if isinstance(p, str) else p, dtype=float) * np.pi / size for p in path
    ]
    ks, dist = [corners[0]], [0.0]
    labels = [(0.0, names[0])]
    for a, b, name in zip(corners[:-1], corners[1:], names[1:], strict=True):
        start, length = dist[-1], float(np.linalg.norm(b - a))
        for t in np.linspace(0.0, 1.0, int(points) + 1)[1:]:
            ks.append(a + t * (b - a))
            dist.append(start + t * length)
        labels.append((dist[-1], name))
    return np.array(ks), np.array(dist), labels


@dataclass
class BandStructure:
    """The angular frequencies (``frequencies``: wave vectors x bands, increasing) of
    the Bloch waves at the ``wavevectors``, their ``distance`` along the path and the
    positions and names of its corners (``labels``)."""

    wavevectors: np.ndarray
    distance: np.ndarray
    frequencies: np.ndarray
    labels: list

    def gaps(self, relative: float = 1e-3) -> list[tuple[int, float, float]]:
        """The band gaps along the path: ``(n, low, high)`` between the bands ``n`` and
        ``n + 1`` (counted from 1), wider than ``relative`` times their mid-frequency."""
        out = []
        for n in range(self.frequencies.shape[1] - 1):
            low, high = float(self.frequencies[:, n].max()), float(self.frequencies[:, n + 1].min())
            if high - low > relative * 0.5 * (high + low):
                out.append((n + 1, low, high))
        return out

    def normalized(self, length: float, speed: float) -> np.ndarray:
        """The frequencies in units of ``2 pi speed / length`` (``ω L / 2π c``)."""
        return self.frequencies * length / (2.0 * np.pi * speed)

    def plot(self, ax=None, *, scale: float = 1.0, ylabel: str = "ω"):
        """Draw the bands (times ``scale``) along the path with its corners and the band
        gaps shaded (matplotlib, without pyplot); returns the axes (``ax.figure.savefig``)."""
        if ax is None:
            from morphogen.viz.figures import _figure

            ax = _figure(5.0, 4.0).subplots()
        for _, low, high in self.gaps():
            ax.axhspan(low * scale, high * scale, color="0.88", lw=0)
        for position, _ in self.labels:
            ax.axvline(position, color="0.6", lw=0.6)
        ax.plot(self.distance, self.frequencies * scale, color="C0", lw=1.2)
        if self.labels:
            ax.set_xticks([d for d, _ in self.labels], [n for _, n in self.labels])
        ax.set_xlim(self.distance[0], self.distance[-1])
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel(ylabel)
        return ax


def _scalar(value, name: str, phase) -> float:
    """An isotropic coefficient of a material for the scalar waves."""
    if value is None:
        raise ValueError(f"waves: give the {name} of phase {phase}")
    if isinstance(value, tuple):
        raise ValueError(f"waves: an isotropic {name} (phase {phase})")
    return float(value)


def wave_coefficients(waves: str, materials: dict[int, Material]) -> tuple[dict, dict]:
    """The coefficients ``a`` and ``b`` of each phase for the scalar ``waves`` of
    :data:`SCALAR_WAVES`, from the materials: their ``density`` and elastic moduli
    (``acoustic``: the pressure waves of fluids, of bulk modulus ``K``; ``shear``:
    antiplane shear waves), or their ``permittivity`` and ``permeability`` (``tm``,
    ``te``; a permeability left out is that of the units, 1). Voids are left out; give
    the coefficients themselves to :func:`scalar_band_structure` for the phases that the
    materials do not describe (a fluid has no shear modulus)."""
    if waves not in SCALAR_WAVES:
        raise ValueError(f"waves: one of {WAVES}, got {waves!r}")
    a, b = {}, {}
    for phase, m in materials.items():
        if m.void:
            a[phase] = b[phase] = 0.0
            continue
        if waves in ("acoustic", "shear"):
            density = _scalar(m.density, "density", phase)
            if m.young is None:
                raise ValueError(f"waves: isotropic elastic moduli (phase {phase})")
            lam, mu = m.lame
            if waves == "acoustic":
                a[phase], b[phase] = 1.0 / density, 1.0 / (lam + 2.0 * mu / 3.0)
            else:
                a[phase], b[phase] = mu, density
        else:
            eps = _scalar(m.permittivity, "permittivity", phase)
            mu = 1.0 if m.permeability is None else _scalar(m.permeability, "permeability", phase)
            if not (eps > 0.0 and mu > 0.0):
                raise ValueError(f"waves: positive permittivity and permeability (phase {phase})")
            a[phase], b[phase] = (1.0 / mu, eps) if waves == "tm" else (1.0 / eps, mu)
    return a, b


def electromagnetic_coefficients(materials: dict[int, Material]) -> tuple[dict, dict]:
    """The permittivity and the permeability of each phase for the electromagnetic waves
    (numbers, or 3 x 3 tensors in the axes of the cell), from the materials: their
    ``permittivity`` and ``permeability`` (left out: that of the units, 1). A void holds
    no wave in the scalar problems; here every phase carries the field: give the vacuum
    its permittivity."""
    eps, mu = {}, {}
    for phase, m in materials.items():
        if m.void or m.permittivity is None:
            raise ValueError(
                f"waves: give the permittivity of phase {phase} (1 for the vacuum: every phase "
                "carries the electromagnetic field)"
            )
        eps[phase] = (
            np.array(m.permittivity) if isinstance(m.permittivity, tuple) else m.permittivity
        )
        value = 1.0 if m.permeability is None else m.permeability
        mu[phase] = np.array(value) if isinstance(value, tuple) else value
    return eps, mu


def _default_path(sample, path):
    """The path of a cell when none is given: Γ–X–M–Γ in the plane of a cell one voxel
    thick, Γ–X–M–Γ–R otherwise."""
    if path is not None:
        return path
    planar = isinstance(sample, VoxelImage) and sample.shape[2] == 1
    return ("G", "X", "M", "G") if planar else ("G", "X", "M", "G", "R")


def _check_plane(sample, path, points: int, waves: str) -> None:
    """The waves of a medium invariant along the third axis: a cell that does not vary
    along it, and wave vectors in its plane."""
    other = "'elastic'" if waves == "shear" else "'electromagnetic'"
    if isinstance(sample, VoxelImage) and sample.shape[2] > 1:
        data = np.asarray(sample.data)
        if np.any(data != data[:, :, :1]):
            raise ValueError(
                f"waves {waves!r}: those of a cell invariant along z (one voxel thick); the "
                f"waves of a three-dimensional cell: {other}"
            )
    if isinstance(path, np.ndarray) and path.ndim == 2:
        ks = np.asarray(path, dtype=float)
    else:
        ks, _, _ = brillouin_path(sample.size, path, points)
    if np.any(np.abs(ks[:, 2]) > 1e-12 * max(float(np.abs(ks).max()), 1e-300)):
        raise ValueError(
            f"waves {waves!r}: wave vectors in the plane x-y of the cell (a path such as G X M "
            f"G); the waves that leave it: {other}"
        )


def _wave_mesh(sample, order: int, phases) -> TetMesh:
    mesh = voxel_tet_mesh(sample, phases) if isinstance(sample, VoxelImage) else sample
    if int(order) == 2:
        return mesh.quadratic().straight()  # the mass matrices are those of straight edges
    if int(order) == 1:
        return mesh.linear()
    raise ValueError("order: 1 or 2")


def _path(mesh: TetMesh, path, points: int):
    if isinstance(path, np.ndarray) and path.ndim == 2:
        ks = np.asarray(path, dtype=float)
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ks, axis=0), axis=1))])
        return ks, dist, []
    return brillouin_path(mesh.size, path, points)


def _bands(stiffness, mass, mesh: TetMesh, nd: int, ks, bands: int) -> np.ndarray:
    """The lowest Bloch frequencies of the matrices of a cell (``nd`` unknowns per node)
    at the wave vectors ``ks``."""
    t, masters, _ = periodic_map(mesh)
    t = t.tocsr()
    # the periods between each node and each of its masters (interpolated ones included)
    rows = np.repeat(np.arange(t.shape[0]), np.diff(t.indptr))
    delta = mesh.nodes[rows] - mesh.nodes[masters[t.indices]]
    periods = np.rint(delta / mesh.size) * mesh.size
    scale = float(np.median(stiffness.diagonal()) / np.median(mass.diagonal()))
    sigma = -1e-6 * scale
    n = int(bands)
    if not 0 < n < t.shape[1] * nd:
        raise ValueError(
            f"band_structure: between 1 and {t.shape[1] * nd - 1} bands (the unknowns of the "
            f"cell, less one), got {n}"
        )
    freqs = np.zeros((len(ks), n))
    for i, k in enumerate(ks):
        tk = sp.csr_matrix(
            (t.data * np.exp(1j * (periods @ k)), t.indices, t.indptr), shape=t.shape
        )
        tn = sp.kron(tk, sp.identity(nd), format="csr") if nd > 1 else tk
        kr = _real(tn.getH() @ stiffness @ tn)
        mr = _real(tn.getH() @ mass @ tn)
        # minimum degree on A + A^T: a quarter of the fill of SuperLU's default ordering
        lu = splu((kr - sigma * mr).tocsc(), permc_spec="MMD_AT_PLUS_A")
        inverse = LinearOperator(kr.shape, matvec=lu.solve, dtype=float)
        values = eigsh(kr, 2 * n, mr, sigma=sigma, OPinv=inverse, return_eigenvectors=False)
        values = np.sort(values)
        freqs[i] = np.sqrt(np.clip(0.5 * (values[0::2] + values[1::2]), 0.0, None))
    return freqs


def scalar_band_structure(
    sample,
    stiffness: dict[int, float],
    mass: dict[int, float],
    path=("G", "X", "M", "G", "R"),
    *,
    bands: int = 8,
    points: int = 12,
    order: int = 2,
    phases=None,
) -> BandStructure:
    """The lowest ``bands`` Bloch frequencies of the scalar waves ``div(a grad u) + ω² b
    u = 0`` of a periodic cell (mesh or voxel image, as :func:`band_structure`): ``a``
    the ``stiffness`` coefficient of each phase and ``b`` its ``mass`` coefficient (the
    table of the module: acoustic, antiplane shear and electromagnetic waves). The phases
    whose ``a`` is zero are left out (rigid scatterers), their boundaries free of flux."""
    mesh = _wave_mesh(sample, order, phases)
    present = sorted({int(p) for p in mesh.phase})
    missing = [p for p in present if p not in stiffness or p not in mass]
    if missing:
        raise ValueError(f"scalar_band_structure: no coefficients for phase(s) {missing}")
    bad = [p for p in present if stiffness[p] < 0.0 or (stiffness[p] > 0.0 and not mass[p] > 0.0)]
    if bad:
        raise ValueError(
            f"scalar_band_structure: positive coefficients (a >= 0, b > 0) for phase(s) {bad}"
        )
    mats = {
        p: Material(conductivity=float(stiffness[p])) if stiffness[p] > 0.0 else Material(void=True)
        for p in present
    }
    problem = FEThermalProblem(mesh, mats, bc="periodic")
    if problem.empty:
        raise ValueError("scalar_band_structure: no phase carries the waves")
    mesh = problem.mesh
    weight = np.array([mass[int(p)] for p in mesh.phase], dtype=float)
    ks, dist, labels = _path(mesh, path, points)
    matrix = problem._stiffness().tocsr()
    freqs = _bands(matrix, mass_matrix(mesh, weight), mesh, 1, ks, bands)
    return BandStructure(ks, dist, freqs, labels)


def long_wavelength_speeds(
    sample, stiffness: dict[int, float], mass: dict[int, float], **options
) -> dict:
    """The speeds of the scalar waves of a periodic cell at long wavelengths, those of the
    homogenized medium: ``c_i = sqrt(a*_ii / <b>)`` along each axis, with the effective
    tensor ``a*`` of the coefficient ``a`` (a conduction problem: the FFT solver of a
    voxel image, finite elements on a mesh) and the mean of ``b`` (over the whole cell,
    the phases left out included: they hold no wave). Returns ``"tensor"`` (``a*``, 3 x
    3), ``"mass"`` (``<b>``) and ``"speeds"`` (3,). For the sound of a mixture of fluids
    this is Wood's law (``1 / K* = <1 / K>``) with the effective density ``(a*)^-1``; for
    light, ``c / n*``. ``options``: those of :func:`~morphogen.homogenization.apparent.
    make_problem`."""
    from morphogen.analysis import phase_fractions
    from morphogen.homogenization.apparent import conductivity_tensor, make_problem

    mats = {
        p: Material(conductivity=float(a)) if a > 0.0 else Material(void=True)
        for p, a in stiffness.items()
    }
    problem = make_problem(sample, mats, "thermal", "periodic", **options)
    tensor, _ = conductivity_tensor(problem)
    if isinstance(sample, VoxelImage):
        fractions = phase_fractions(sample)
    else:
        volumes = np.abs(sample.volumes())
        fractions = {
            int(p): float(volumes[sample.phase == p].sum() / np.prod(sample.size))
            for p in np.unique(sample.phase)
        }
    mean = float(sum(f * mass[p] for p, f in fractions.items() if stiffness.get(p, 0.0) > 0.0))
    return {"tensor": tensor, "mass": mean, "speeds": np.sqrt(np.diag(tensor) / mean)}


def band_structure(
    sample,
    materials: dict[int, Material],
    path=None,
    *,
    bands: int = 12,
    points: int = 12,
    order: int = 2,
    phases=None,
    waves: str = "elastic",
    coarsen: int = 1,
) -> BandStructure:
    """The lowest ``bands`` Bloch frequencies of a periodic cell (a
    :class:`~morphogen.mesh.tetra.TetMesh`, or a :class:`~morphogen.image.VoxelImage`
    cut into six tetrahedra per voxel, keeping ``phases``) along ``path`` (corners of the
    Brillouin zone, ``points`` per segment; or an array of wave vectors; by default
    Γ–X–M–Γ–R, Γ–X–M–Γ in the plane of an image one voxel thick), with elements of
    ``order`` 1 or 2. The materials need a ``density``; void phases are left out, as the
    clusters of elements that touch no face of the box. ``waves``: ``"elastic"`` (the
    vector waves of the solid); the scalar waves of :data:`SCALAR_WAVES` (``"acoustic"``,
    and ``"shear"``, ``"tm"``, ``"te"`` in the plane of a cell invariant along z) with
    the coefficients of the materials (:func:`wave_coefficients`,
    :func:`scalar_band_structure`); or ``"electromagnetic"``, the vector waves of light in
    a voxel image of any morphology, by plane waves on its voxels or on a grid
    ``coarsen`` times coarser (:func:`~morphogen.homogenization.photonic.
    electromagnetic_band_structure`; ``order`` and ``phases`` do not apply)."""
    if waves not in WAVES:
        raise ValueError(f"waves: one of {WAVES}, got {waves!r}")
    path = _default_path(sample, path)
    if waves == "electromagnetic":
        from morphogen.homogenization.photonic import electromagnetic_band_structure

        eps, mu = electromagnetic_coefficients(materials)
        return electromagnetic_band_structure(
            sample, eps, path, permeability=mu, bands=bands, points=points, coarsen=coarsen
        )
    if waves != "elastic":
        a, b = wave_coefficients(waves, materials)
        if waves in PLANE_WAVES:
            _check_plane(sample, path, points, waves)
        return scalar_band_structure(
            sample, a, b, path, bands=bands, points=points, order=order, phases=phases
        )
    mesh = voxel_tet_mesh(sample, phases) if isinstance(sample, VoxelImage) else sample
    if int(order) == 2:
        mesh = mesh.quadratic().straight()  # the mass matrices are those of straight edges
    elif int(order) == 1:
        mesh = mesh.linear()
    else:
        raise ValueError("order: 1 or 2")
    problem = FEElasticProblem(mesh, materials, bc="periodic")
    if problem.empty:
        raise ValueError("band_structure: no material in the cell")
    mesh = problem.mesh
    missing = sorted({int(p) for p in mesh.phase if materials[int(p)].density is None})
    if missing:
        raise ValueError(f"band_structure: give the density of phase(s) {missing}")
    density = np.array([materials[int(p)].density for p in mesh.phase], dtype=float)
    stiffness = problem._stiffness().tocsr()
    mass = sp.kron(mass_matrix(mesh, density), sp.identity(3), format="csr")
    ks, dist, labels = _path(mesh, path, points)
    freqs = _bands(stiffness, mass, mesh, 3, ks, bands)
    return BandStructure(ks, dist, freqs, labels)


def _real(h: sp.spmatrix) -> sp.csc_matrix:
    """The real symmetric matrix ``[[A, -B], [B, A]]`` of a Hermitian ``A + iB``: each
    eigenvalue twice, so that the symmetric Lanczos of ARPACK applies (the complex
    Hermitian one is that of nonsymmetric matrices)."""
    h = 0.5 * (h + h.getH())
    return sp.bmat([[h.real, -h.imag], [h.imag, h.real]], format="csc")
