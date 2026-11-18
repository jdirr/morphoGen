"""Strongly coupled linear homogenization on periodic voxel images: several fields
solved together, each a displacement or a scalar potential, their fluxes tied by one
symmetric constitutive law.

A *layout* lists the fields. A displacement ``u`` has the strain ``eps`` (6 components,
Voigt order 11 22 33 23 31 12, engineering shears) for its generalized gradient and the
stress ``sigma`` for its flux; a scalar potential ``phi`` has its gradient ``grad phi``
(3 components) and a flux vector. The generalized gradient ``Z`` stacks those of the
fields, the generalized flux ``S`` their fluxes, and the law of each phase is the
symmetric matrix ``L`` of ``S = L Z``. The couplings of :data:`COUPLINGS`:

``piezoelectric`` (``u``, the electric potential ``phi``; ``E = -grad phi``)
    ``sigma = C eps - e^T E``, ``D = e eps + kappa E``: ``Z = (eps, grad phi)``, ``S =
    (sigma, D)``, ``L = [[C, e^T], [e, -kappa]]`` with the stiffness ``C`` at constant
    electric field, the piezoelectric matrix ``e`` (3 x 6, stress-charge form, C/m^2) and
    the permittivity ``kappa`` at constant strain.
``piezomagnetic`` (``u``, the magnetic potential ``psi``; ``H = -grad psi``)
    ``sigma = C eps - q^T H``, ``B = q eps + mu H``: the same with the piezomagnetic
    matrix ``q`` (3 x 6, N/(A m)) and the permeability ``mu``.
``magnetoelectroelastic`` (``u``, ``phi``, ``psi``)
    ``sigma = C eps - e^T E - q^T H``, ``D = e eps + kappa E + alpha H``, ``B = q eps +
    alpha^T E + mu H``: ``L = [[C, e^T, q^T], [e, -kappa, -alpha], [q, -alpha^T, -mu]]``
    with the magnetoelectric tensor ``alpha`` (3 x 3). A composite of a piezoelectric
    phase and a piezomagnetic one is magnetoelectric though neither phase is (a product
    property).
``magnetoelectric`` (``phi``, ``psi``; rigid phases)
    ``D = kappa E + alpha H``, ``B = alpha^T E + mu H``: ``Z = (grad phi, grad psi)``,
    ``S = -(D, B)``, ``L = [[kappa, alpha], [alpha^T, mu]]``.
``thermoelectric`` (the electric potential ``V``, the temperature ``T``, linearized
    about the mean temperature ``T0``)
    ``j = -sigma (grad V + s grad T)`` (the current density; ``s``: the Seebeck
    coefficient) and the entropy flux ``j_s = q / T0 = s j - (k / T0) grad T`` (``k``: the
    thermal conductivity at zero current; Kelvin's relation ``Pi = s T0``): ``Z = (grad V,
    grad T)``, ``S = -(j, j_s)``, ``L = [[sigma, sigma s], [s^T sigma, k / T0 + s^T sigma
    s]]``, symmetric positive definite by Onsager's reciprocity.

The energy-conserving laws (the first three) are symmetric but not definite: their
potentials are saddle points of the electric enthalpy. :class:`CoupledProblem` solves the
periodic fluctuations of all the fields at once on the trilinear voxels of the FFT solver
(:mod:`morphogen.homogenization.fft`): the system ``K w = f`` of the element matrices
``int B^T L B`` (``B``: the generalized gradients of the shape functions) by MINRES, or by
conjugate gradients when every block of the law is positive, preconditioned block by
block by the inverse of a homogeneous reference medium in Fourier space. The unknowns of
each field are scaled by the square root of its reference so that the blocks have the
same magnitude whatever the units (stiffnesses of 1e11 Pa beside permittivities of 1e-9
F/m). The unit loadings of ``Z`` give the effective law ``L*`` (the mean fluxes), which
:func:`coupled_tensors` splits into the tensors of the coupling, with the constants
derived from them.

Uniform fields that load the phases without a gradient are the weak couplings of the
same problems: a change of temperature with the thermal stresses ``beta = C alpha`` and
the pyroelectric coefficients ``p`` of the phases, ``S = L Z - lambda dT``, gives the
effective ``lambda*`` (:meth:`CoupledProblem.eigen_flux`; the generalized relation of
Levin).

:func:`laminate` is the exact effective law of layers, for any of these laws: the checks
of the solver.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from morphogen.homogenization import kernels
from morphogen.homogenization.elements import hex_element
from morphogen.homogenization.fft import FourierPreconditioner, reference_moduli
from morphogen.homogenization.materials import Material
from morphogen.homogenization.solvers import NEGLIGIBLE, SolveInfo, conjugate_gradient, minres
from morphogen.image import VoxelImage
from morphogen.stopping import checkpoint

log = logging.getLogger("morphogen")

#: The notation of the tensors of the coupled laws
NOTATION = (
    "Voigt order 11 22 33 23 31 12 of strains (engineering shears) and stresses: "
    "sigma = C eps - e^T E - q^T H, D = e eps + kappa E + alpha H, B = q eps + alpha^T E + mu H "
    "(e, q: 3 x 6, rows 1 2 3; d = e C^-1; kappa, mu at constant strain); thermoelectric: "
    "j = -sigma (grad V + s grad T), q = T0 s j - k grad T"
)
#: The fields of a layout: the degrees of freedom of a node and the components of the
#: generalized gradient of each kind
FIELDS = {"displacement": (3, 6), "potential": (1, 3)}
#: The couplings and the fields of their layouts
COUPLINGS = {
    "piezoelectric": ("displacement", "potential"),
    "piezomagnetic": ("displacement", "potential"),
    "magnetoelectroelastic": ("displacement", "potential", "potential"),
    "magnetoelectric": ("potential", "potential"),
    "thermoelectric": ("potential", "potential"),
}
#: Voigt indices of the components of a strain normal to the layers of a laminate normal
#: to x3 (33, 23, 31) and in their plane (11, 22, 12); of a gradient likewise (3; 1, 2)
_NORMAL = {"displacement": (2, 3, 4), "potential": (2,)}
_PLANE = {"displacement": (0, 1, 5), "potential": (0, 1)}


def layout_sizes(layout) -> tuple[int, int]:
    """The degrees of freedom of a node and the components of the generalized gradient of
    a layout of fields (:data:`FIELDS`)."""
    unknown = [f for f in layout if f not in FIELDS]
    if unknown or not layout:
        raise ValueError(f"layout: fields among {tuple(FIELDS)}, got {tuple(layout)}")
    return sum(FIELDS[f][0] for f in layout), sum(FIELDS[f][1] for f in layout)


def _slices(layout) -> list[tuple[slice, slice]]:
    """The degrees of freedom of a node and the components of the gradient of each field."""
    out, dof, comp = [], 0, 0
    for f in layout:
        nd, nz = FIELDS[f]
        out.append((slice(dof, dof + nd), slice(comp, comp + nz)))
        dof, comp = dof + nd, comp + nz
    return out


# -- tensors of the couplings --------------------------------------------------------------


def _tensor3(e) -> np.ndarray:
    """The third-order tensor ``e_kij`` (3 x 3 x 3, symmetric in ij) of a piezoelectric
    matrix (3 x 6, Voigt order 11 22 33 23 31 12 of the strains, engineering shears)."""
    from morphogen.homogenization.materials import _VOIGT_INDEX

    return np.asarray(e, dtype=float)[..., _VOIGT_INDEX]


def rotate_piezo(e, euler) -> np.ndarray:
    """The piezoelectric (or piezomagnetic) matrix of a crystal (3 x 6, in its axes) in
    the axes of the sample, for one orientation (Bunge angles in degrees: shape (3, 6)) or
    several (shape (n, 3, 6)): ``e_kij = g_pk g_qi g_rj e'_pqr`` with ``crystal = g @
    sample``, as :func:`~morphogen.homogenization.materials.rotate_stiffness`."""
    from morphogen.homogenization.materials import _I0, _I1
    from morphogen.orientations import euler_to_matrix

    g = euler_to_matrix(euler)
    t = np.einsum("npk,nqi,nrj,pqr->nkij", g, g, g, _tensor3(e), optimize=True)
    out = t[:, :, _I0, _I1]
    return out[0] if np.ndim(euler) == 1 else out


def poled_ceramic(e31: float, e33: float, e15: float) -> np.ndarray:
    """The piezoelectric matrix ``e`` (3 x 6) of a ceramic poled along x3 (the symmetry
    6mm of PZT and BaTiO3; also the piezomagnetic matrix of CoFe2O4 magnetized along x3):
    ``e31 = e32``, ``e33`` and ``e15 = e24`` (Voigt order 11 22 33 23 31 12: ``e15`` is
    the entry (1, 31), ``e24`` the entry (2, 23))."""
    e = np.zeros((3, 6))
    e[2, 0] = e[2, 1] = e31
    e[2, 2] = e33
    e[0, 4] = e[1, 3] = e15
    return e


def _piezo(value, name: str):
    if value is None:
        return None
    e = np.asarray(value, dtype=float)
    if e.shape != (3, 6) or not np.all(np.isfinite(e)):
        raise ValueError(f"{name}: a finite 3 x 6 matrix (Voigt order 11 22 33 23 31 12)")
    return tuple(tuple(float(v) for v in row) for row in e)


def _second_order(value, name: str, *, symmetric: bool):
    """One value, three principal values or a 3 x 3 tensor, as a tuple (None: none)."""
    if value is None:
        return None
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = float(a) * np.eye(3)
    elif a.shape == (3,):
        a = np.diag(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: one value, three principal values or a 3 x 3 tensor")
    if symmetric and not np.allclose(a, a.T, rtol=1e-10, atol=1e-14 * np.abs(a).max()):
        raise ValueError(f"{name}: a symmetric tensor")
    return tuple(tuple(float(v) for v in row) for row in a)


@dataclass(frozen=True)
class Coupling:
    """The coupling coefficients of a material, in the axes of its crystal (those of the
    sample without an orientation): ``piezoelectric`` (``e``, 3 x 6: ``D = e eps`` at zero
    field, C/m^2), ``piezomagnetic`` (``q``, 3 x 6: ``B = q eps``, N/(A m)),
    ``magnetoelectric`` (``alpha``, 3 x 3 or one value: ``D = alpha H``), ``seebeck``
    (``s``, one value or a 3 x 3 tensor: the field ``E = s grad T`` at zero current, V/K)
    and ``pyroelectric`` (``p``, 3 components: ``D = p dT`` at constant strain and field,
    C/(m^2 K)). Matrices follow morphoGen's Voigt order 11 22 33 23 31 12 with engineering
    shears, which is that of the IEEE standard on piezoelectricity (``e15``: the entry of
    the row 1 and the column 31)."""

    piezoelectric: tuple | None = None
    piezomagnetic: tuple | None = None
    magnetoelectric: tuple | None = None
    seebeck: tuple | None = None
    pyroelectric: tuple | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "piezoelectric", _piezo(self.piezoelectric, "piezoelectric"))
        set_(self, "piezomagnetic", _piezo(self.piezomagnetic, "piezomagnetic"))
        alpha = _second_order(self.magnetoelectric, "magnetoelectric", symmetric=False)
        set_(self, "magnetoelectric", alpha)
        set_(self, "seebeck", _second_order(self.seebeck, "seebeck", symmetric=True))
        if self.pyroelectric is not None:
            p = np.asarray(self.pyroelectric, dtype=float)
            if p.shape != (3,) or not np.all(np.isfinite(p)):
                raise ValueError("pyroelectric: three finite components")
            set_(self, "pyroelectric", tuple(float(v) for v in p))

    @classmethod
    def from_config(cls, spec) -> Coupling:
        """A mapping of the coefficients (:class:`Coupling`); a poled ceramic gives ``e``
        (or ``q``) by its constants: ``piezoelectric: {e31: .., e33: .., e15: ..}``."""
        if isinstance(spec, Coupling):
            return spec
        if not isinstance(spec, dict):
            raise ValueError("coupling: a mapping of coupling coefficients")
        names = ("piezoelectric", "piezomagnetic", "magnetoelectric", "seebeck", "pyroelectric")
        unknown = sorted(set(spec) - set(names))
        if unknown:
            raise ValueError(f"coupling: unknown keys {unknown}; allowed: {list(names)}")
        values = dict(spec)
        for name in ("piezoelectric", "piezomagnetic"):
            value = values.get(name)
            if isinstance(value, dict):
                prefix = "e" if name == "piezoelectric" else "q"
                keys = {f"{prefix}31", f"{prefix}33", f"{prefix}15"}
                if set(value) != keys:
                    raise ValueError(f"coupling.{name}: a 3 x 6 matrix, or {sorted(keys)}")
                values[name] = poled_ceramic(*(float(value[f"{prefix}{k}"]) for k in (31, 33, 15)))
        return cls(**values)

    def to_config(self) -> dict:
        out = {}
        for name in ("piezoelectric", "piezomagnetic", "magnetoelectric", "seebeck",
                     "pyroelectric"):  # fmt: skip
            value = getattr(self, name)
            if value is not None:
                out[name] = np.array(value).tolist()
        return out

    def matrix(self, name: str, euler=None) -> np.ndarray:
        """The tensor ``name`` in the axes of the sample for the orientation ``euler``
        (Bunge angles, degrees), zero without it: 3 x 6 (``piezoelectric``,
        ``piezomagnetic``), 3 x 3 (``magnetoelectric``, ``seebeck``) or 3
        (``pyroelectric``)."""
        from morphogen.orientations import euler_to_matrix

        value = getattr(self, name)
        shape = {"piezoelectric": (3, 6), "piezomagnetic": (3, 6), "pyroelectric": (3,)}
        if value is None:
            return np.zeros(shape.get(name, (3, 3)))
        a = np.array(value)
        if euler is None:
            return a
        if a.shape == (3, 6):
            return rotate_piezo(a, euler)
        g = euler_to_matrix(euler)[0]
        return g.T @ a if a.shape == (3,) else g.T @ a @ g


def _dielectric(m: Material, physics: str, euler, name: str) -> np.ndarray:
    if not m.has(physics):
        raise ValueError(f"{name}: the material has no {physics} coefficient")
    return m.matrix(physics, euler)


def coupled_law(material: Material, coupling: str, euler=None, *, temperature: float = 300.0):
    """The law ``L`` of a material for a coupling of :data:`COUPLINGS` (see the module)
    in the axes of the sample (``euler``: the orientation of its crystal), and its
    thermal flux ``lambda`` of ``S = L Z - lambda dT`` (the thermal stress ``C alpha`` and
    minus the pyroelectric coefficients; zero for the transport couplings). A material
    without elastic parameters has no stiffness (a pore filled by a dielectric);
    ``temperature``: the mean temperature ``T0`` of the thermoelectric law (K)."""
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling: one of {tuple(COUPLINGS)}, got {coupling!r}")
    c = material.coupling or Coupling()
    _, nz = layout_sizes(COUPLINGS[coupling])
    law, eigen = np.zeros((nz, nz)), np.zeros(nz)
    if coupling == "thermoelectric":
        if not temperature > 0:
            raise ValueError("thermoelectric: a positive mean temperature (K)")
        sigma = _dielectric(material, "electrical", euler, coupling)
        k = _dielectric(material, "thermal", euler, coupling)
        s = c.matrix("seebeck", euler)
        law[:3, :3] = sigma
        law[:3, 3:] = sigma @ s
        law[3:, :3] = (sigma @ s).T
        law[3:, 3:] = k / temperature + s.T @ sigma @ s
        return law, eigen
    if coupling == "magnetoelectric":
        alpha = c.matrix("magnetoelectric", euler)
        law[:3, :3] = _dielectric(material, "dielectric", euler, coupling)
        law[3:, 3:] = _dielectric(material, "magnetic", euler, coupling)
        law[:3, 3:], law[3:, :3] = alpha, alpha.T
        return law, eigen
    if material.elastic:
        from morphogen.homogenization.elements import tensor_to_voigt

        law[:6, :6] = material.matrix("elastic", euler)
        strain = tensor_to_voigt(material.expansion_tensor(euler), strain=True)
        eigen[:6] = law[:6, :6] @ strain
    if coupling in ("piezoelectric", "magnetoelectroelastic"):
        e = c.matrix("piezoelectric", euler)
        law[6:9, :6], law[:6, 6:9] = e, e.T
        law[6:9, 6:9] = -_dielectric(material, "dielectric", euler, coupling)
        eigen[6:9] = -c.matrix("pyroelectric", euler)
    if coupling in ("piezomagnetic", "magnetoelectroelastic"):
        at = 9 if coupling == "magnetoelectroelastic" else 6
        q = c.matrix("piezomagnetic", euler)
        law[at : at + 3, :6], law[:6, at : at + 3] = q, q.T
        law[at : at + 3, at : at + 3] = -_dielectric(material, "magnetic", euler, coupling)
    if coupling == "magnetoelectroelastic":
        alpha = c.matrix("magnetoelectric", euler)
        law[6:9, 9:12], law[9:12, 6:9] = -alpha, -alpha.T
    return law, eigen


# -- the exact law of a laminate -----------------------------------------------------------


def laminate(laws, fractions, layout, eigen=None):
    """The exact effective law of layers normal to x3 of laws ``laws`` (symmetric
    matrices ``S = L Z`` of a layout of fields) in the volume ``fractions``: the
    components of ``Z`` in the plane of the layers and those of ``S`` across them are
    uniform (the continuity of the tangential gradients and of the normal fluxes), so
    that ``L*_nn = <L_nn^-1>^-1``, ``L*_np = L*_nn <L_nn^-1 L_np>`` and ``L*_pp = <L_pp -
    L_pn L_nn^-1 L_np> + <L_pn L_nn^-1> L*_nn <L_nn^-1 L_np>`` (Backus' averages of a
    stiffness, Maxwell's of a conductivity, and those of every coupled law of the module,
    definite or not; of complex symmetric laws too: those of a harmonic regime,
    :mod:`morphogen.homogenization.harmonic`). With the thermal fluxes ``eigen`` of the
    layers (``S = L Z - lambda dT``), returns the law and the effective ``lambda*``:
    ``lambda*_n = L*_nn <L_nn^-1
    lambda_n>`` and ``lambda*_p = <lambda_p - L_pn L_nn^-1 lambda_n> + <L_pn L_nn^-1>
    lambda*_n``."""
    _, nz = layout_sizes(layout)
    laws = np.asarray(laws)
    laws = laws.astype(complex if np.iscomplexobj(laws) else float)  # harmonic laws too
    f = np.asarray(fractions, dtype=float)
    if laws.shape != (f.size, nz, nz):
        raise ValueError(f"laminate: one {nz} x {nz} law per layer")
    if np.any(f < 0) or not np.isclose(f.sum(), 1.0):
        raise ValueError("laminate: fractions that sum to 1")
    n, p = [], []
    for kind, (_, comp) in zip(layout, _slices(layout), strict=True):
        n += [comp.start + i for i in _NORMAL[kind]]
        p += [comp.start + i for i in _PLANE[kind]]
    inv = [np.linalg.inv(m[np.ix_(n, n)]) for m in laws]
    lnn = np.linalg.inv(sum(fi * a for fi, a in zip(f, inv, strict=True)))
    a = sum(fi * i @ m[np.ix_(n, p)] for fi, i, m in zip(f, inv, laws, strict=True))
    lpp = sum(
        fi * (m[np.ix_(p, p)] - m[np.ix_(p, n)] @ i @ m[np.ix_(n, p)])
        for fi, i, m in zip(f, inv, laws, strict=True)
    )
    out = np.zeros((nz, nz), dtype=laws.dtype)
    out[np.ix_(n, n)] = lnn
    out[np.ix_(n, p)] = lnn @ a
    out[np.ix_(p, n)] = (lnn @ a).T
    out[np.ix_(p, p)] = lpp + a.T @ lnn @ a
    if eigen is None:
        return out
    lam = np.asarray(eigen, dtype=float).reshape(f.size, nz)
    star = np.zeros(nz)
    star[n] = lnn @ sum(fi * i @ v[n] for fi, i, v in zip(f, inv, lam, strict=True))
    star[p] = (
        sum(
            fi * (v[p] - m[np.ix_(p, n)] @ i @ v[n])
            for fi, i, m, v in zip(f, inv, laws, lam, strict=True)
        )
        + a.T @ star[n]
    )
    return out, star


# -- kernels -------------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _apply(nx, ny, nz, nd, group, mats, w, out):
    """``out = K w`` on the periodic grid of ``nd`` unknowns per node (the kernel of
    :func:`morphogen.homogenization.fft.apply_operator` for any ``nd``)."""
    corners = kernels.CORNERS
    size = 8 * nd
    for n in prange(nx * ny * nz):
        k = n % nz
        j = (n // nz) % ny
        i = n // (ny * nz)
        acc = np.zeros(nd)
        we = np.empty(size)
        for a in range(2):
            ei = i - a if i - a >= 0 else nx - 1
            for b in range(2):
                ej = j - b if j - b >= 0 else ny - 1
                for c in range(2):
                    ek = k - c if k - c >= 0 else nz - 1
                    g = group[(ei * ny + ej) * nz + ek]
                    if g < 0:
                        continue
                    ln = kernels.CORNER_INDEX[a, b, c]
                    for q in range(8):
                        qi = ei + corners[q, 0]
                        qj = ej + corners[q, 1]
                        qk = ek + corners[q, 2]
                        if qi == nx:
                            qi = 0
                        if qj == ny:
                            qj = 0
                        if qk == nz:
                            qk = 0
                        m = (qi * ny + qj) * nz + qk
                        for c2 in range(nd):
                            we[q * nd + c2] = w[m * nd + c2]
                    mat = mats[g]
                    for r in range(nd):
                        row = ln * nd + r
                        s = 0.0
                        for col in range(size):
                            s += mat[row, col] * we[col]
                        acc[r] += s
        for r in range(nd):
            out[n * nd + r] = acc[r]


@njit(parallel=True, cache=True)
def _sums(nx, ny, nz, nd, mat, stat, n_stat, laws, eigen, dt, w, aff, b_gauss, wt):
    """Integrals over each phase (``stat``) of the generalized gradients ``Z``, of the
    fluxes ``S = L Z - dt lambda``, of ``S . Z`` and of the volume: (phases, 2 nz + 2)."""
    n_el = nx * ny * nz
    ncomp = b_gauss.shape[1]
    size = 8 * nd
    n_chunks = min(n_el, 256)
    partial = np.zeros((n_chunks, n_stat, 2 * ncomp + 2))
    for ch in prange(n_chunks):
        nodes = np.empty(8, dtype=np.int64)
        we = np.empty(size)
        z = np.empty(ncomp)
        s = np.empty(ncomp)
        for e in range(ch * n_el // n_chunks, (ch + 1) * n_el // n_chunks):
            m = mat[e]
            if m < 0:
                continue
            kernels._element_nodes(e, nx, ny, nz, True, nodes)
            for q in range(8):
                for c in range(nd):
                    we[q * nd + c] = aff[q * nd + c] + w[nodes[q] * nd + c]
            acc = partial[ch, stat[e]]
            for gp in range(8):
                for r in range(ncomp):
                    v = 0.0
                    for col in range(size):
                        v += b_gauss[gp, r, col] * we[col]
                    z[r] = v
                dens = 0.0
                for r in range(ncomp):
                    v = -dt * eigen[m, r]
                    for col in range(ncomp):
                        v += laws[m, r, col] * z[col]
                    s[r] = v
                    dens += v * z[r]
                for r in range(ncomp):
                    acc[r] += wt * z[r]
                    acc[ncomp + r] += wt * s[r]
                acc[2 * ncomp] += wt * dens
                acc[2 * ncomp + 1] += wt
    return partial.sum(axis=0)


@njit(parallel=True, cache=True)
def _element_fields(nx, ny, nz, nd, mat, laws, eigen, dt, w, aff, b_gauss):
    """The means over each voxel of the generalized gradients and fluxes (zero in the
    voxels without a law)."""
    n_el = nx * ny * nz
    ncomp = b_gauss.shape[1]
    size = 8 * nd
    grad = np.zeros((n_el, ncomp))
    flux = np.zeros((n_el, ncomp))
    for e in prange(n_el):
        m = mat[e]
        if m < 0:
            continue
        nodes = np.empty(8, dtype=np.int64)
        we = np.empty(size)
        z = np.empty(ncomp)
        kernels._element_nodes(e, nx, ny, nz, True, nodes)
        for q in range(8):
            for c in range(nd):
                we[q * nd + c] = aff[q * nd + c] + w[nodes[q] * nd + c]
        for gp in range(8):
            for r in range(ncomp):
                v = 0.0
                for col in range(size):
                    v += b_gauss[gp, r, col] * we[col]
                z[r] = v
                grad[e, r] += 0.125 * v
            for r in range(ncomp):
                v = -dt * eigen[m, r]
                for col in range(ncomp):
                    v += laws[m, r, col] * z[col]
                flux[e, r] += 0.125 * v
    return grad, flux


# -- the problem ---------------------------------------------------------------------------


@dataclass
class CoupledSolution:
    """One solved loading of a :class:`CoupledProblem`: the macroscopic generalized
    gradient ``gradient`` and the mean flux ``flux`` (the components of the layout), the
    mean of ``S . Z``, the change of temperature, the volume fraction that carries a law,
    the solver data and, with ``fields``, the voxel means of the gradients and fluxes
    (``fields["gradient"]``, ``fields["flux"]``: shape + components) and the nodal
    fluctuation of each field (``fields["fluctuation"]``: shape + degrees of freedom)."""

    gradient: np.ndarray
    flux: np.ndarray
    energy: float
    temperature: float
    material_fraction: float
    info: SolveInfo
    phases: dict = field(default_factory=dict, repr=False)
    fields: dict | None = field(default=None, repr=False)


def conjugate_orthogonal_gradient(apply, precondition, rhs, tol, maxiter, nd: int):
    """The conjugate orthogonal conjugate gradient method (van der Vorst & Melissen 1990)
    for the real form of a complex symmetric system: the unknowns ``[x, y]`` of ``z = x +
    i y`` (``nd`` per node, the first half ``x``) and the operator ``apply([x, y]) =
    [Re(K z), -Im(K z)]`` of a complex symmetric ``K``; ``precondition`` acts alike on the
    two halves (a real symmetric positive operator). Conjugate gradients with the
    unconjugated product ``u^T v``: the iterations of the real problem when the imaginary
    part of ``K`` is small, where MINRES on the indefinite real form needs as many as the
    square of them. Returns (the solution, the iterations, the relative residual); the
    method has no minimization property and may stall (a residual above ``tol`` at the
    end: the caller falls back on MINRES)."""
    half = nd // 2

    def to_complex(v, sign=1.0):
        f = v.reshape(-1, nd)
        return (f[:, :half] + sign * 1j * f[:, half:]).ravel()

    def to_real(c, sign=1.0):
        out = np.empty((c.size // half, nd))
        f = c.reshape(-1, half)
        out[:, :half] = f.real
        out[:, half:] = sign * f.imag
        return out.ravel()

    def operator(p):
        return to_complex(apply(to_real(p)), -1.0)

    def inverse(r):
        return to_complex(precondition(to_real(r)))

    c = to_complex(rhs, -1.0)
    norm_c = float(np.linalg.norm(c))
    z = np.zeros_like(c)
    if norm_c == 0.0:
        return to_real(z), 0, 0.0
    r = c.copy()
    s = inverse(r)
    p = s.copy()
    rho = np.dot(r, s)  # unconjugated
    res, it = 1.0, 0
    for it in range(1, maxiter + 1):  # noqa: B007
        checkpoint()
        q = operator(p)
        curvature = np.dot(p, q)
        if curvature == 0.0 or not np.isfinite(curvature):
            break
        alpha = rho / curvature
        z += alpha * p
        r -= alpha * q
        res = float(np.linalg.norm(r)) / norm_c
        if res <= tol or not np.isfinite(res):
            break
        s = inverse(r)
        rho_new = np.dot(r, s)
        if rho == 0.0:
            break
        p = s + (rho_new / rho) * p
        rho = rho_new
    return to_real(z), it, res if np.isfinite(res) else float("inf")


class CoupledProblem:
    """Periodic problem of the coupled fields of a ``layout`` (:data:`FIELDS`) on a voxel
    image: ``laws[m]`` is the symmetric law ``S = L Z`` of the material ``m`` (the
    components of the layout), ``mat`` the material of each voxel (-1: none, a void) and
    ``eigen[m]`` its flux ``lambda`` per unit change of temperature (``S = L Z - lambda
    dT``; zero if not given). Each diagonal block of the laws must be definite, positive or
    negative, alike in all the materials that have it (a block may vanish in some: the
    stiffness of a pore filled by a dielectric).

    :meth:`solve` gives the mean flux of a macroscopic gradient ``Z`` (and a change of
    temperature), :meth:`tensor` the effective law ``L*`` (one solve per component, the
    unit solutions cached) and :meth:`eigen_flux` the effective ``lambda*``. ``tol``: the
    relative residual of the Krylov solver (MINRES; conjugate gradients when every block
    is positive); ``stat``: the phase of each voxel for the statistics (``mat`` if not
    given). ``pairs``: the layout is twice the same fields, the real and imaginary parts
    of a complex symmetric law (``[[L', -L''], [-L'', -L']]``: the harmonic problems of
    :mod:`morphogen.homogenization.harmonic`), solved by
    :func:`conjugate_orthogonal_gradient` (MINRES if it stalls)."""

    def __init__(self, image: VoxelImage, laws, layout, mat, *, eigen=None, stat=None,
                 tol: float = 1e-8, maxiter: int | None = None, pairs: bool = False):  # fmt: skip
        if not isinstance(image, VoxelImage):
            raise TypeError("the coupled solver needs a VoxelImage")
        self.layout = tuple(layout)
        self.nd, self.nz = layout_sizes(self.layout)
        self.image = image
        self.shape = image.shape
        self.box_volume = float(image.volume)
        laws = np.array(laws, dtype=float).reshape(-1, self.nz, self.nz)
        if not np.allclose(laws, np.swapaxes(laws, 1, 2), rtol=1e-10,
                           atol=1e-12 * max(np.abs(laws).max(initial=0.0), 1e-300)):  # fmt: skip
            raise ValueError("coupled laws: symmetric matrices S = L Z")
        self.laws = 0.5 * (laws + np.swapaxes(laws, 1, 2))
        self.mat = np.ascontiguousarray(np.asarray(mat).ravel(), dtype=np.int64)
        if self.mat.size != int(np.prod(self.shape)):
            raise ValueError("mat: the material of each voxel of the image")
        if self.mat.max(initial=-1) >= len(self.laws):
            raise ValueError("mat: indices of the laws")
        self.stat = self.mat if stat is None else np.ascontiguousarray(stat, dtype=np.int64)
        self.eigen = (
            np.zeros((len(self.laws), self.nz))
            if eigen is None
            else np.array(eigen, dtype=float).reshape(len(self.laws), self.nz)
        )
        self.tol = float(tol)
        self.maxiter = maxiter
        self.pairs = bool(pairs)
        middle = len(self.layout) // 2
        if self.pairs and self.layout[:middle] != self.layout[middle:]:
            raise ValueError("pairs: a layout of twice the same fields")
        self.counts = np.bincount(self.mat[self.mat >= 0], minlength=len(self.laws))
        self.empty = not np.any(self.counts) or not np.any(self.laws[self.counts > 0])
        self.element = hex_element(tuple(image.spacing))
        self.b_gauss = self._gradients()
        self._references()
        # the scaled problem: unknowns w = scale * w', gradients Z = scale * Z'
        d = self.scale_z
        self._laws = np.ascontiguousarray(self.laws * d[None, :, None] * d[None, None, :])
        self._eigen = np.ascontiguousarray(self.eigen * d[None, :])
        k = self.element.weight * np.einsum(
            "gri,mrs,gsj->mij", self.b_gauss, self._laws, self.b_gauss, optimize=True
        )
        self.element_matrices = np.ascontiguousarray(0.5 * (k + np.swapaxes(k, 1, 2)))
        self._preconditioners = None
        self._units: dict = {}

    # -- set-up ------------------------------------------------------------------------

    def _gradients(self) -> np.ndarray:
        """The generalized gradients of the shape functions at the Gauss points: (8, nz, 8
        nd), the unknowns of a voxel ordered corner by corner."""
        el, nd = self.element, self.nd
        b = np.zeros((8, self.nz, 8 * nd))
        for kind, (dof, comp) in zip(self.layout, _slices(self.layout), strict=True):
            for a in range(8):
                if kind == "displacement":
                    for c in range(3):
                        b[:, comp, a * nd + dof.start + c] = el.b_gauss[:, :, 3 * a + c]
                else:
                    b[:, comp, a * nd + dof.start] = el.g_gauss[:, :, a]
        return np.ascontiguousarray(b)

    def _references(self) -> None:
        """The sign and the homogeneous reference medium of each block, and the scales
        of the fields: the reference of a potential is the mean of the extreme
        eigenvalues of its block over the materials, that of a displacement the isotropic
        medium of :func:`~morphogen.homogenization.fft.reference_moduli`."""
        used = self.laws[self.counts > 0]
        self.signs, self.references = [], []
        scale_w, scale_z = np.ones(self.nd), np.ones(self.nz)
        for kind, (dof, comp) in zip(self.layout, _slices(self.layout), strict=True):
            blocks = used[:, comp, comp]
            blocks = blocks[np.abs(blocks).max(axis=(1, 2), initial=0.0) > 0]
            if not len(blocks):  # no material has the block: nothing to solve for it
                self.signs.append(1.0)
                self.references.append(None)
                continue
            eig = np.linalg.eigvalsh(blocks)
            sign = 1.0 if eig.min() > 0 else -1.0
            if not np.all(sign * eig > 0):
                raise ValueError(
                    f"coupled laws: the block of the field {kind!r} must be definite, "
                    "positive or negative, alike in every material"
                )
            if kind == "displacement":
                lam, mu = reference_moduli(sign * blocks, False)
                ref = (lam, mu)
                size = mu
            else:
                ref = 0.5 * (sign * eig).min() + 0.5 * (sign * eig).max()
                size = ref
            self.signs.append(sign)
            self.references.append(ref)
            scale_w[dof] = scale_z[comp] = 1.0 / np.sqrt(size)
        self.scale_w, self.scale_z = scale_w, scale_z
        self.definite = all(s > 0 for s in self.signs)

    def _preconditioner(self):
        """The inverse of the homogeneous reference medium of each block (scaled), in
        Fourier space; the identity for a block that no material has."""
        if self._preconditioners is None:
            el, out = self.element, []
            for kind, ref, (dof, _) in zip(self.layout, self.references, _slices(self.layout),
                                           strict=True):  # fmt: skip
                if ref is None:
                    out.append(None)
                elif kind == "displacement":
                    k = el.stiffness(*ref) * self.scale_w[dof.start] ** 2
                    out.append(FourierPreconditioner.from_element(self.shape, k, 3))
                else:
                    k = ref * el.k_thermal * self.scale_w[dof.start] ** 2
                    out.append(FourierPreconditioner.from_element(self.shape, k, 1))
            self._preconditioners = out
        parts = list(zip(self._preconditioners, _slices(self.layout), strict=True))
        nd = self.nd

        def precondition(r):
            field_r = r.reshape(-1, nd)
            z = np.empty_like(field_r)
            for p, (dof, _) in parts:
                block = np.ascontiguousarray(field_r[:, dof])
                z[:, dof] = block if p is None else p(block.ravel()).reshape(block.shape)
            return z.ravel()

        return precondition

    def _affine(self, gradient) -> np.ndarray:
        """The nodal values (8 nd) of the affine fields of a (scaled) macroscopic
        gradient at the corners of a voxel."""
        el, nd = self.element, self.nd
        out = np.zeros((8, nd))
        for kind, (dof, comp) in zip(self.layout, _slices(self.layout), strict=True):
            if kind == "displacement":
                out[:, dof] = el.affine_displacement(gradient[comp]).reshape(8, 3)
            else:
                out[:, dof.start] = el.affine_temperature(gradient[comp])
        return out.ravel()

    # -- solution ----------------------------------------------------------------------

    def _fluctuation(self, z: np.ndarray, dt: float):
        """The (scaled) periodic fluctuation of the scaled macroscopic gradient ``z`` and
        the change of temperature ``dt``."""
        nx, ny, nz = self.shape
        nd = self.nd
        loads = self.element_matrices @ self._affine(z)
        if dt:
            thermal = self.element.weight * np.einsum("grc,mr->mc", self.b_gauss, self._eigen)
            loads = loads - dt * thermal
        loads = np.ascontiguousarray(loads)
        dof = np.arange(nd * nx * ny * nz, dtype=np.int64)
        rhs = kernels.element_load(nx, ny, nz, True, self.mat, dof, nd, loads, dof.size)
        scale = float(np.sqrt(np.sum(self.counts * np.sum(loads * loads, axis=1))))
        if float(np.linalg.norm(rhs)) <= NEGLIGIBLE * scale:
            return np.zeros_like(rhs), SolveInfo("fft", rhs.size, 0, 0.0, 0.0, 0.0)
        mats = self.element_matrices

        def apply(v):
            out = np.empty_like(v)
            _apply(nx, ny, nz, nd, self.mat, mats, v, out)
            return out

        maxiter = self.maxiter or 40 * int(max(self.shape)) + 2000
        solve = conjugate_gradient if self.definite else minres
        method = "fft-cg" if self.definite else "fft-minres"
        t0 = time.perf_counter()
        w, it, res = None, 0, float("inf")
        if self.pairs:  # a complex symmetric law: conjugate gradients without conjugation
            w, it, res = conjugate_orthogonal_gradient(
                apply, self._preconditioner(), rhs, self.tol, maxiter, nd
            )
            method = "fft-cocg"
        if res > self.tol:  # (the method stalled: the real form, by MINRES)
            spent = it
            w, it, res = solve(apply, self._preconditioner(), rhs, self.tol, maxiter)
            it += spent
            method = "fft-cg" if self.definite else "fft-minres"
        if res > self.tol:
            log.warning(
                "coupled solver stopped at relative residual %.2e after %d iterations", res, it
            )
        return w, SolveInfo(method, rhs.size, it, res, 0.0, time.perf_counter() - t0)

    def _unit(self, j: int):
        """The fluctuation of the unit (scaled) gradient ``j`` (-1: a unit change of
        temperature), cached."""
        if j not in self._units:
            z = np.zeros(self.nz)
            if j >= 0:
                z[j] = 1.0
            self._units[j] = self._fluctuation(z, 1.0 if j < 0 else 0.0)
        return self._units[j]

    def solve(self, gradient, *, temperature: float = 0.0, fields: bool = False):
        """The solution (:class:`CoupledSolution`) of a macroscopic generalized gradient
        ``Z`` (the components of the layout) and a change of ``temperature``, by
        superposition of the unit solutions."""
        z = np.asarray(gradient, dtype=float).ravel()
        if z.shape != (self.nz,):
            raise ValueError(f"gradient: the {self.nz} components of the layout")
        t0 = time.perf_counter()
        dt = float(temperature)
        if self.empty:
            info = SolveInfo("none", 0, 0, 0.0, 0.0, 0.0)
            return CoupledSolution(z, np.zeros(self.nz), 0.0, dt, 0.0, info)
        zs = z / self.scale_z
        w = np.zeros(self.nd * int(np.prod(self.shape)))
        infos = []
        for j in [*np.flatnonzero(zs), *([-1] if dt else [])]:
            wj, info = self._unit(int(j))
            w += (dt if j < 0 else zs[j]) * wj
            infos.append(info)
        nx, ny, nz = self.shape
        aff = self._affine(zs)
        n_stat = int(self.stat.max(initial=0)) + 1
        sums = _sums(nx, ny, nz, self.nd, self.mat, self.stat, n_stat, self._laws,
                     self._eigen, dt, w, aff, self.b_gauss, self.element.weight)  # fmt: skip
        total = sums.sum(axis=0)
        ncomp = self.nz
        info = SolveInfo(
            infos[0].method if infos else "none",
            w.size,
            sum(i.iterations for i in infos),
            max((i.residual for i in infos), default=0.0),
            0.0,
            time.perf_counter() - t0,
        )
        phases = {}
        for g in range(n_stat):
            vol = sums[g, -1]
            if vol > 0:
                phases[g] = {
                    "fraction": float(vol / self.box_volume),
                    "gradient": sums[g, :ncomp] / vol * self.scale_z,
                    "flux": sums[g, ncomp : 2 * ncomp] / vol / self.scale_z,
                }
        out = CoupledSolution(
            gradient=z,
            flux=total[ncomp : 2 * ncomp] / self.box_volume / self.scale_z,
            energy=float(total[2 * ncomp] / self.box_volume),
            temperature=dt,
            material_fraction=float(total[-1] / self.box_volume),
            info=info,
            phases=phases,
        )
        if fields:
            grad, flux = _element_fields(nx, ny, nz, self.nd, self.mat, self._laws,
                                         self._eigen, dt, w, aff, self.b_gauss)  # fmt: skip
            out.fields = {
                "gradient": (grad * self.scale_z).reshape(*self.shape, ncomp),
                "flux": (flux / self.scale_z).reshape(*self.shape, ncomp),
                "fluctuation": (w.reshape(-1, self.nd) * self.scale_w).reshape(
                    *self.shape, self.nd
                ),
            }
        return out

    def tensor(self) -> np.ndarray:
        """The effective law ``L*`` (nz x nz): the mean fluxes of the unit gradients,
        symmetrized (it is symmetric to the tolerance of the solver)."""
        law = np.zeros((self.nz, self.nz))
        for j in range(self.nz):
            unit = np.zeros(self.nz)
            unit[j] = 1.0
            law[:, j] = self.solve(unit).flux
        return 0.5 * (law + law.T)

    def eigen_flux(self) -> np.ndarray:
        """The effective thermal flux ``lambda*`` of ``S = L* Z - lambda* dT``: minus the
        mean flux of a unit change of temperature without a macroscopic gradient."""
        return -self.solve(np.zeros(self.nz), temperature=1.0).flux

    @property
    def iterations(self) -> int:
        """The iterations of the unit solutions solved so far."""
        return sum(info.iterations for _, info in self._units.values())


def coupled_problem(image: VoxelImage, materials: dict[int, Material], coupling: str, *,
                    grains=None, orientations=None, temperature: float = 300.0,
                    tol: float = 1e-8, maxiter: int | None = None) -> CoupledProblem:  # fmt: skip
    """The :class:`CoupledProblem` of a voxel image whose phases have the ``materials``,
    for a coupling of :data:`COUPLINGS` (the laws of :func:`coupled_law`). The crystals
    are turned by the orientations of their grains (``grains``: the label of each voxel,
    ``orientations``: their Bunge angles), as in the other problems; ``temperature``: the
    mean temperature of a thermoelectric law (K)."""
    if not isinstance(image, VoxelImage):
        raise TypeError("coupled homogenization: a periodic VoxelImage (the FFT solver)")
    if image.data.dtype.kind not in "iub":
        raise ValueError("homogenization needs an integer phase image")
    phase = np.asarray(image.data).ravel().astype(np.int64)
    present = [int(p) for p in np.unique(phase)]
    missing = [p for p in present if p not in materials]
    if missing:
        raise ValueError(f"no material for phase(s) {missing}; phases present: {present}")
    oriented = grains is not None and orientations is not None
    if oriented:
        grains = np.asarray(grains.data if isinstance(grains, VoxelImage) else grains).ravel()
        if grains.size != phase.size:
            raise ValueError("grains: one grain label per voxel of the image")
    stat = np.searchsorted(present, phase)
    mat = np.full(phase.size, -1, dtype=np.int64)
    laws, eigen = [], []
    for p in present:
        m = materials[p]
        if m.void:
            continue
        sel = np.flatnonzero(phase == p)
        try:
            if oriented and _oriented(m):
                ids, inverse = np.unique(grains[sel], return_inverse=True)
                mat[sel] = len(laws) + inverse
                for g in ids:
                    law, lam = coupled_law(m, coupling, np.asarray(orientations, float)[g],
                                           temperature=temperature)  # fmt: skip
                    laws.append(law)
                    eigen.append(lam)
                continue
            law, lam = coupled_law(m, coupling, temperature=temperature)
        except ValueError as exc:
            raise ValueError(f"phase {p}: {exc}") from None
        mat[sel] = len(laws)
        laws.append(law)
        eigen.append(lam)
    layout = COUPLINGS[coupling]
    nz = layout_sizes(layout)[1]
    problem = CoupledProblem(
        image,
        np.array(laws).reshape(-1, nz, nz),
        layout,
        mat,
        eigen=np.array(eigen).reshape(-1, nz),
        stat=stat,
        tol=tol,
        maxiter=maxiter,
    )
    problem.coupling = coupling
    problem.phases = tuple(present)
    problem.temperature = float(temperature)
    return problem


def _oriented(m: Material) -> bool:
    """Whether a law of the material turns with the orientation of its grain."""
    from morphogen.homogenization.materials import PHYSICS

    return m.coupling is not None or any(m.has(p) and m.oriented(p) for p in PHYSICS)


# -- the tensors of a coupling ---------------------------------------------------------------


def split_law(law, coupling: str, *, temperature: float = 300.0) -> dict:
    """The tensors of a law ``L`` of a coupling (:data:`COUPLINGS`; see the module), with
    the constants derived from them.

    ``piezoelectric``: ``stiffness`` (``C^E``, at constant field), ``piezoelectric``
    (``e``, 3 x 6), ``permittivity`` (``kappa^eps``, at constant strain), then
    ``compliance`` (``S^E``), ``d`` (``d = e S^E``, the strain-charge form: 3 x 6, C/N),
    ``permittivity_stress`` (``kappa^sigma = kappa^eps + d e^T``), ``g33`` (``d33 /
    kappa^sigma_33``), the hydrostatic coefficients ``d_h = d31 + d32 + d33`` and ``g_h =
    d_h / kappa^sigma_33``, the coupling factors ``k33`` (``d33 / sqrt(S^E_33
    kappa^sigma_33)``) and ``k_t`` (thickness mode: ``e33 / sqrt(C^D_33 kappa^eps_33)``,
    ``C^D_33 = C^E_33 + e33^2 / kappa^eps_33``), and the entries ``e31``, ``e33``,
    ``e15``, ``d31``, ``d33``, ``d15``. ``piezomagnetic``: the same with
    ``piezomagnetic`` (``q``) and ``permeability``. ``magnetoelectroelastic``: both, and
    ``magnetoelectric`` (``alpha``, 3 x 3) with its entries ``alpha11``, ``alpha33``.
    ``magnetoelectric``: ``permittivity``, ``permeability``, ``magnetoelectric``.
    ``thermoelectric``: ``electrical_conductivity``, ``seebeck`` (``sigma^-1 L_12``),
    ``conductivity`` (thermal, at zero current) and, along each axis, the power factor
    ``sigma s^2`` and the figure of merit ``ZT = sigma s^2 T0 / k`` (``power_factor1``,
    ``zt1``...).
    """
    law = np.asarray(law, dtype=float)
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling: one of {tuple(COUPLINGS)}, got {coupling!r}")
    out: dict = {"tensor": law}
    if coupling == "thermoelectric":
        sigma = law[:3, :3]
        s = np.linalg.solve(sigma, law[:3, 3:])
        k = temperature * (law[3:, 3:] - law[3:, :3] @ s)
        out.update(electrical_conductivity=sigma, seebeck=s, conductivity=0.5 * (k + k.T))
        for i in range(3):
            pf = float(sigma[i, i] * s[i, i] ** 2)
            out[f"power_factor{i + 1}"] = pf
            out[f"zt{i + 1}"] = float(pf * temperature / k[i, i])
        return out
    if coupling == "magnetoelectric":
        out.update(permittivity=law[:3, :3], permeability=law[3:, 3:], magnetoelectric=law[:3, 3:])
        out.update(alpha11=float(law[0, 3]), alpha33=float(law[2, 5]))
        return out
    c = law[:6, :6]
    out["stiffness"] = c
    comp = np.linalg.inv(c) if np.linalg.matrix_rank(c) == 6 else None
    if comp is not None:
        out["compliance"] = comp
    blocks = []
    if coupling in ("piezoelectric", "magnetoelectroelastic"):
        blocks.append(("piezoelectric", "permittivity", "e", "d", slice(6, 9)))
    if coupling in ("piezomagnetic", "magnetoelectroelastic"):
        at = 9 if coupling == "magnetoelectroelastic" else 6
        blocks.append(("piezomagnetic", "permeability", "q", "dm", slice(at, at + 3)))
    for name, susceptibility, e_name, d_name, sl in blocks:
        e = law[sl, :6]
        kappa = -law[sl, sl]
        out[name], out[susceptibility] = e, kappa
        for key, (i, j) in (("31", (2, 0)), ("33", (2, 2)), ("15", (0, 4))):
            out[f"{e_name}{key}"] = float(e[i, j])
        if comp is None:
            continue
        d = e @ comp
        stress_free = kappa + d @ e.T
        out[d_name], out[f"{susceptibility}_stress"] = d, stress_free
        for key, (i, j) in (("31", (2, 0)), ("33", (2, 2)), ("15", (0, 4))):
            out[f"{d_name}{key}"] = float(d[i, j])
        if name == "piezoelectric":
            dh = float(d[2, 0] + d[2, 1] + d[2, 2])
            out["d_h"] = dh
            out["g33"] = float(d[2, 2] / stress_free[2, 2])
            out["g_h"] = float(dh / stress_free[2, 2])
            out["k33"] = float(abs(d[2, 2]) / np.sqrt(comp[2, 2] * stress_free[2, 2]))
            stiffened = c[2, 2] + e[2, 2] ** 2 / kappa[2, 2]
            out["k_t"] = float(abs(e[2, 2]) / np.sqrt(stiffened * kappa[2, 2]))
    if coupling == "magnetoelectroelastic":
        alpha = -law[6:9, 9:12]
        out["magnetoelectric"] = alpha
        out.update(alpha11=float(alpha[0, 0]), alpha33=float(alpha[2, 2]))
    return out


def coupled_tensors(image: VoxelImage, materials: dict[int, Material], coupling: str, *,
                    grains=None, orientations=None, temperature: float = 300.0,
                    tol: float = 1e-8, maxiter: int | None = None,
                    thermal: bool | None = None) -> dict:  # fmt: skip
    """The effective tensors of a coupling (:data:`COUPLINGS`) of a periodic voxel image:
    those of :func:`split_law` for its effective law (``"tensor"``: the law itself), the
    iterations and the largest residual of the solver (``"iterations"``, ``"residual"``)
    and, with ``thermal`` (by default: when a phase expands or is pyroelectric), the
    effective thermal flux (``"thermal_stress"``: ``beta* = C* alpha*``, Voigt;
    ``"thermal_expansion"``: ``alpha*`` at constant field, 3 x 3; ``"pyroelectric"``:
    ``p*`` at constant strain and field). See :func:`coupled_problem` for the options."""
    problem = coupled_problem(image, materials, coupling, grains=grains,
                              orientations=orientations, temperature=temperature, tol=tol,
                              maxiter=maxiter)  # fmt: skip
    law = problem.tensor()
    out = split_law(law, coupling, temperature=temperature)
    if thermal is None:
        thermal = bool(np.any(problem.eigen))
    if thermal and COUPLINGS[coupling][0] == "displacement":
        from morphogen.homogenization.elements import voigt_to_tensor

        lam = problem.eigen_flux()
        out["thermal_stress"] = lam[:6]
        if "compliance" in out:
            out["thermal_expansion"] = voigt_to_tensor(out["compliance"] @ lam[:6], strain=True)
        if coupling in ("piezoelectric", "magnetoelectroelastic"):
            out["pyroelectric"] = -lam[6:9]
    infos = [info for _, info in problem._units.values()]
    out["iterations"] = int(sum(i.iterations for i in infos))
    out["residual"] = float(max((i.residual for i in infos), default=0.0))
    return out


#: The prefix of the entries of each tensor of a coupled law in the tables of results
#: (:func:`coupled_entries`), and whether the tensor is symmetric (its upper triangle)
ENTRIES = {
    "stiffness": ("C", True),
    "piezoelectric": ("e", False),
    "d": ("d", False),
    "piezomagnetic": ("q", False),
    "dm": ("dm", False),
    "permittivity": ("eps", True),
    "permittivity_stress": ("eps_s", True),
    "permeability": ("mu", True),
    "permeability_stress": ("mu_s", True),
    "magnetoelectric": ("alpha", False),
    "electrical_conductivity": ("sigma", True),
    "seebeck": ("s", False),
    "conductivity": ("k", True),
    "thermal_expansion": ("A", True),
    "thermal_stress": ("beta", False),
    "pyroelectric": ("p", False),
}


def entry_names(coupling: str) -> tuple[str, ...]:
    """The names of the scalar results of a coupling (:data:`COUPLINGS`) in the tables of
    results and for the statistics of RVE studies: the entries of its tensors
    (:func:`coupled_entries`: ``C11``, ``e33``, ``eps11``...), those of its thermal flux
    with a displacement (``A11``, ``beta1``, ``p3``) and its constants (``d33``, ``k33``,
    ``zt1``...)."""
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling: one of {tuple(COUPLINGS)}, got {coupling!r}")
    layout = COUPLINGS[coupling]
    with np.errstate(invalid="ignore", divide="ignore"):  # a dummy law: only its names
        tensors = split_law(np.eye(layout_sizes(layout)[1]), coupling)
    if layout[0] == "displacement":
        tensors["thermal_stress"] = np.zeros(6)
        tensors["thermal_expansion"] = np.zeros((3, 3))
        if coupling in ("piezoelectric", "magnetoelectroelastic"):
            tensors["pyroelectric"] = np.zeros(3)
    constants = [k for k, v in tensors.items() if isinstance(v, float)]
    return (*coupled_entries(tensors), *constants)


def coupled_entries(tensors: dict) -> dict[str, float]:
    """The entries of the tensors of a coupled law (:func:`split_law`,
    :func:`coupled_tensors`) as the columns of a table of results: ``C11`` ... ``C66``,
    ``e11`` ... ``e36`` (rows 1 2 3, columns in the Voigt order 11 22 33 23 31 12 numbered
    1 to 6), ``d11`` ... ``d36``, ``eps11`` ... ``eps33``, ``mu11``, ``alpha11``,
    ``sigma11``, ``s11``, ``k11``, ``A11``, ``beta1`` ... ``beta6``, ``p1`` ... ``p3``
    (:data:`ENTRIES`: the upper triangles of the symmetric tensors)."""
    out: dict[str, float] = {}
    for name, (prefix, symmetric) in ENTRIES.items():
        value = tensors.get(name)
        if value is None:
            continue
        a = np.asarray(value, dtype=float)
        if a.ndim == 1:
            out.update({f"{prefix}{i + 1}": float(v) for i, v in enumerate(a)})
            continue
        for i in range(a.shape[0]):
            for j in range(i if symmetric else 0, a.shape[1]):
                out[f"{prefix}{i + 1}{j + 1}"] = float(a[i, j])
    return out
