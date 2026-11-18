"""Acoustics of porous media: the parameters of the equivalent fluid of a rigid porous
frame from its microstructure, and the sound absorption of a layer.

Sound in the air (or any fluid) that saturates a rigid porous frame, at wavelengths much
longer than the pores, is that of an *equivalent fluid* with a complex dynamic density
``rho(omega)`` (viscous and inertial effects) and bulk modulus ``K(omega)`` (thermal
effects). The semi-phenomenological model of Johnson, Koplik & Dashen (1987), Champoux &
Allard (1991) and Lafarge et al. (1997) — JCAL — gives them from six parameters of the
geometry of the pores, each the result of a static problem on the microstructure:

============================  ==============  =============================================
parameter                     symbol          problem
============================  ==============  =============================================
porosity                      ``phi``         the volume fraction of the pores
static permeability           ``k0``          Stokes flow (:mod:`~morphogen.homogenization.
                                              flow`): Darcy's law, the resistivity
                                              ``sigma = eta / k0``
tortuosity                    ``alpha_inf``   potential flow: the conduction of the pore
                                              fluid between insulating walls, ``alpha_inf
                                              = phi / s*`` (``s*``: its apparent
                                              conductivity for a unit one)
viscous length                ``Lambda``      the same field ``E``: ``2 int_V E^2 dV /
                                              int_S E^2 dS``, measured by moving the walls
thermal length                ``Lambda'``     ``2 V / S`` of the pores
static thermal permeability   ``k0'``         ``lap(theta) = -1`` in the pores, ``theta =
                                              0`` on the walls: ``k0' = <theta>``
============================  ==============  =============================================

(:func:`acoustic_parameters`; lengths in the unit of the spacing of the image,
permeabilities in its square). With the time dependence ``exp(i omega t)``, the viscosity
``eta``, density ``rho0``, Prandtl number ``Pr``, ratio of specific heats ``gamma`` and
pressure ``P0`` of the fluid,

    rho(omega) = (alpha_inf rho0 / phi) [1 + (eta phi / (i omega rho0 alpha_inf k0))
                  sqrt(1 + i 4 alpha_inf^2 k0^2 rho0 omega / (eta Lambda^2 phi^2))],
    K(omega) = (gamma P0 / phi) / (gamma - (gamma - 1) / [1 + (eta phi / (i omega rho0 Pr
                k0')) sqrt(1 + i 4 k0'^2 rho0 omega Pr / (eta Lambda'^2 phi^2))])

(:func:`equivalent_fluid`; without ``k0'``, the model of Johnson, Champoux and Allard:
``k0' = phi Lambda'^2 / 8``, the value of circular pores). The characteristic impedance
``Z_c = sqrt(rho K)`` and the wavenumber ``k = omega sqrt(rho / K)`` give the surface
impedance ``-i Z_c cot(k d)`` of a layer of thickness ``d`` on a rigid wall and its
absorption coefficient at normal incidence, ``1 - |(Z_s - Z_0) / (Z_s + Z_0)|^2``
(:func:`absorption`). The exact responses of slits and of circular tubes
(:func:`pore_response`; Zwikker & Kosten 1949, Stinson 1991) are the references of the
model.

The static thermal permeability is also the mean survival time of a species that
diffuses in the pores and is absorbed by their walls, times its diffusivity (the trapping
constant of diffusion-controlled reactions: Torquato 2002).

References: D. L. Johnson, J. Koplik, R. Dashen, J. Fluid Mech. 176 (1987) 379; Y.
Champoux, J.-F. Allard, J. Appl. Phys. 70 (1991) 1975; D. Lafarge, P. Lemarinier, J.-F.
Allard, V. Tarnow, J. Acoust. Soc. Am. 102 (1997) 1995; J.-F. Allard, N. Atalla,
Propagation of Sound in Porous Media, 2nd ed., Wiley (2009); M. R. Stinson, J. Acoust.
Soc. Am. 89 (1991) 550; C. Perrot, F. Chevillotte, R. Panneton, J. Appl. Phys. 103
(2008) 024909.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from morphogen.homogenization.core import VOIGT_NAMES, Loading
from morphogen.homogenization.elements import VOIGT
from morphogen.homogenization.flow import permeability_tensor, pore_mask
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage
from morphogen.stopping import checkpoint

__all__ = [
    "AIR",
    "PARAMETERS",
    "Fluid",
    "absorption",
    "acoustic_parameters",
    "along_axis",
    "equivalent_fluid",
    "pore_response",
    "thermal_permeability",
]

#: The scalar results of :func:`acoustic_parameters` (columns of the tables of results
#: and quantities of the statistics of RVE studies)
PARAMETERS = (
    "porosity",
    "permeability",
    "tortuosity",
    "viscous_length",
    "thermal_length",
    "thermal_permeability",
    "specific_surface",
    *(f"permeability{i}" for i in (1, 2, 3)),
    *(f"tortuosity{i}" for i in (1, 2, 3)),
    *(f"viscous_length{i}" for i in (1, 2, 3)),
    *(f"k{name}" for name in VOIGT_NAMES),
)


@dataclass(frozen=True)
class Fluid:
    """The fluid that saturates the pores (SI units): ``density``, dynamic ``viscosity``,
    ratio of specific heats ``heat_ratio``, Prandtl number ``prandtl`` and static
    ``pressure``. The default is air at 20 °C."""

    density: float = 1.204
    viscosity: float = 1.81e-5
    heat_ratio: float = 1.4
    prandtl: float = 0.71
    pressure: float = 101325.0

    def __post_init__(self) -> None:
        for name in ("density", "viscosity", "heat_ratio", "prandtl", "pressure"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"fluid.{name}: positive")
        if not self.heat_ratio >= 1.0:
            raise ValueError("fluid.heat_ratio: at least 1")

    @property
    def sound_speed(self) -> float:
        """The adiabatic speed of sound, ``sqrt(gamma P0 / rho0)``."""
        return float(np.sqrt(self.heat_ratio * self.pressure / self.density))

    @property
    def impedance(self) -> float:
        """The characteristic impedance ``rho0 c0``."""
        return self.density * self.sound_speed


#: Air at 20 °C and atmospheric pressure
AIR = Fluid()


@njit(cache=True, parallel=True)
def _wall_laplacian(theta, fluid, out):  # pragma: no cover
    """``out = sum over the six neighbours of (theta - neighbour)`` in the pores, the
    value beyond a wall the mirror ``-theta`` (zero on the wall, midway)."""
    nx, ny, nz = fluid.shape
    for x in prange(nx):
        xm, xp = (x - 1) % nx, (x + 1) % nx
        for y in range(ny):
            ym, yp = (y - 1) % ny, (y + 1) % ny
            for z in range(nz):
                if not fluid[x, y, z]:
                    out[x, y, z] = 0.0
                    continue
                zm, zp = (z - 1) % nz, (z + 1) % nz
                t = theta[x, y, z]
                s = 0.0
                s += t - theta[xm, y, z] if fluid[xm, y, z] else 2.0 * t
                s += t - theta[xp, y, z] if fluid[xp, y, z] else 2.0 * t
                s += t - theta[x, ym, z] if fluid[x, ym, z] else 2.0 * t
                s += t - theta[x, yp, z] if fluid[x, yp, z] else 2.0 * t
                s += t - theta[x, y, zm] if fluid[x, y, zm] else 2.0 * t
                s += t - theta[x, y, zp] if fluid[x, y, zp] else 2.0 * t
                out[x, y, z] = s


def _walls(fluid: np.ndarray) -> np.ndarray:
    """The number of faces of each voxel of the pores that are walls (0 in the solid)."""
    count = np.zeros(fluid.shape)
    for axis in range(3):
        if fluid.shape[axis] == 1:
            continue  # a planar image: its own neighbour along that axis
        for shift in (1, -1):
            count += ~np.roll(fluid, shift, axis=axis)
    return np.where(fluid, count, 0.0)


def _pores(image: VoxelImage, materials, pores) -> tuple[np.ndarray, float]:
    """The mask of the pores of an image of cubic voxels (the cell of a periodic medium),
    and their size."""
    fluid = np.ascontiguousarray(pore_mask(image, materials, pores))
    spacing = np.asarray(image.spacing, dtype=float)
    if not np.allclose(spacing, spacing[0], rtol=1e-9):
        raise ValueError(f"acoustics: cubic voxels, got the spacing {tuple(spacing)}")
    return fluid, float(spacing[0])


def thermal_permeability(
    image: VoxelImage, materials=None, *, pores=None, tol: float = 1e-8, maxiter: int | None = None
) -> float:
    """The static thermal permeability ``k0'`` of the pores of a periodic image (the void
    phases of ``materials``, or the phases ``pores``): the mean over the whole image of
    the solution of ``lap(theta) = -1`` in the pores with ``theta = 0`` on their walls
    (finite differences on the voxels, the walls midway between them, second order;
    conjugate gradients). In the squared unit of the spacing. It is the permeability of the flow
    along a pore of constant section (a slit of width ``H``: ``H^2 / 12``; a tube of radius
    ``R``: ``R^2 / 8``), and the mean survival time of a diffusing species that the walls
    absorb, times its diffusivity and the porosity."""
    fluid, h = _pores(image, materials, pores)
    if not fluid.any():
        return 0.0
    walls = _walls(fluid)
    if not walls.any():
        raise ValueError("thermal permeability: pores without a wall")
    # the diagonal of the operator: 1 per neighbour in the pores, 2 per wall
    diagonal = np.zeros(fluid.shape)
    for axis in range(3):
        if fluid.shape[axis] > 1:
            diagonal += 2.0
    diagonal = np.where(fluid, diagonal + walls, 1.0)
    rhs = fluid.astype(float)
    theta = np.zeros(fluid.shape)
    r = rhs.copy()
    z = r / diagonal
    p = z.copy()
    rz = float(np.vdot(r, z))
    norm = float(np.linalg.norm(rhs))
    ap = np.empty(fluid.shape)
    limit = int(maxiter) if maxiter else 20 * max(fluid.shape) + 2000
    for _ in range(limit):
        checkpoint()
        _wall_laplacian(p, fluid, ap)
        alpha = rz / float(np.vdot(p, ap))
        theta += alpha * p
        r -= alpha * ap
        if float(np.linalg.norm(r)) <= tol * norm:
            break
        z = r / diagonal
        rz_new = float(np.vdot(r, z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    # the mirror of a parabola leaves theta = 1 / 8 on a plane wall instead of zero: that
    # constant is taken out of the pores
    return float((theta.mean() - fluid.mean() / 8.0) * h * h)


def _distance(mask: np.ndarray) -> np.ndarray:
    """The distance (in voxels) of the centre of each voxel of ``mask`` to the nearest
    centre outside it, on the periodic image (zero outside)."""
    from scipy import ndimage

    pad = [(3, 3) if n > 1 else (0, 0) for n in mask.shape]
    distance = ndimage.distance_transform_edt(np.pad(mask, pad, mode="wrap"))
    inner = tuple(slice(3, -3) if n > 1 else slice(None) for n in mask.shape)
    return distance[inner]


def _potential_flow(image: VoxelImage, fluid: np.ndarray, tol: float):
    """The tortuosity and the viscous characteristic length along each axis, and the
    area of the walls per unit volume: the conduction of the pore fluid between
    insulating walls.

    The viscous length is ``2 int_V E^2 dV / int_S E^2 dS``; the integral over the walls
    is the rate at which the conductivity ``s*`` falls when the walls move into the
    pores, ``d ln s* / d delta = -2 / Lambda`` (Johnson, Koplik & Schwartz 1986), which
    stairs of voxels allow to measure where they do not allow a surface integral: the
    pores are dilated and eroded by one voxel (the voxels within reach of the other
    phase), the mean displacement of the walls between the two is their change of
    volume over the area of the walls, and ``Lambda = 2 delta / ln(s*+ / s*-)``. If the
    erosion closes the pores along an axis, the dilation alone is used."""
    from morphogen.analysis import specific_surface
    from morphogen.homogenization.apparent import make_problem

    mats = {1: Material(conductivity=1.0), 0: Material(void=True)}

    def conductivities(mask):
        binary = VoxelImage(mask.astype(np.uint8), image.spacing, periodic=True)
        if not mask.any():
            return np.zeros(3), binary
        problem = make_problem(binary, mats, "thermal", tol=tol)
        out = np.zeros(3)
        for axis in range(3):
            if mask.shape[axis] == 1:  # a planar image: a uniform field along its axis
                out[axis] = float(mask.mean())
                continue
            unit = np.zeros(3)
            unit[axis] = 1.0
            out[axis] = float(-problem.solve(Loading.gradient(unit)).stress[axis])
        return out, binary

    porosity = float(fluid.mean())
    sigma, binary = conductivities(fluid)
    surface = float(specific_surface(binary, 1))  # walls per unit volume (Crofton)
    tortuosity, length = np.ones(3), np.full(3, np.inf)
    if not surface > 0.0:  # no wall: a fluid alone
        return tortuosity, length, surface
    dilated = fluid | (_distance(~fluid) <= 1.0)
    eroded = fluid & (_distance(fluid) > 1.5)
    plus, _ = conductivities(dilated)
    minus, _ = conductivities(eroded)
    for axis in range(3):
        if not sigma[axis] > 1e-12 * porosity:  # no path along this axis
            tortuosity[axis], length[axis] = np.inf, 0.0
            continue
        tortuosity[axis] = porosity / sigma[axis]
        if fluid.shape[axis] == 1:
            length[axis] = 2.0 * porosity / surface
        elif minus[axis] > 1e-3 * sigma[axis]:
            shift = (float(dilated.mean()) - float(eroded.mean())) / surface
            length[axis] = 2.0 * shift / np.log(plus[axis] / minus[axis])
        else:  # the erosion closes the pores
            shift = (float(dilated.mean()) - porosity) / surface
            length[axis] = 2.0 * shift / np.log(plus[axis] / sigma[axis])
    return tortuosity, length, surface


def acoustic_parameters(
    image: VoxelImage,
    materials=None,
    *,
    pores=None,
    tol: float = 1e-5,
    thermal: bool = True,
    **flow_options,
) -> dict:
    """The parameters of the equivalent fluid of a porous image (see the module): the
    pores are the void phases of ``materials``, or the phases ``pores``; the frame is
    rigid and motionless. Periodic images of cubic voxels.

    Returns the ``porosity``; the static viscous ``permeability_tensor`` (3 x 3), its
    diagonal ``permeability1`` ... ``3`` and their mean ``permeability``; the
    ``tortuosity1`` ... ``3`` along the axes and their mean ``tortuosity``; the viscous
    characteristic lengths ``viscous_length1`` ... ``3`` and their mean
    ``viscous_length``; the ``thermal_length`` ``2 V / S``; the ``thermal_permeability``
    (``thermal=False``: left out); the ``specific_surface`` of the walls (per unit volume
    of the image). Lengths in the unit of the spacing, permeabilities in its square: an
    image in metres gives SI values. ``tol``: the tolerance of the flows (their mean
    velocity) and of the conduction problems; ``flow_options``: the other options of
    :func:`~morphogen.homogenization.flow.stokes_flow`."""
    fluid, _ = _pores(image, materials, pores)
    porosity = float(fluid.mean())
    if not 0.0 < porosity:
        raise ValueError("acoustics: no pore in the image")
    # the means: over the axes of the image (those in its plane for a planar one)
    axes = tuple(a for a in range(3) if fluid.shape[a] > 1) or (0, 1, 2)
    flow = permeability_tensor(image, materials, pores=pores, tol=tol, **flow_options)
    tortuosity, viscous, surface = _potential_flow(image, fluid, min(tol, 1e-6))
    out = {
        "porosity": porosity,
        "permeability_tensor": flow["tensor"],
        "specific_surface": surface,
        "thermal_length": float(2.0 * porosity / surface) if surface > 0.0 else float("inf"),
    }
    diagonal = np.diag(flow["tensor"])
    for name, values in (("permeability", diagonal), ("tortuosity", tortuosity),
                         ("viscous_length", viscous)):  # fmt: skip
        out[name] = float(np.mean([values[a] for a in axes]))
        out.update({f"{name}{a + 1}": float(values[a]) for a in range(3)})
    if thermal:
        out["thermal_permeability"] = thermal_permeability(image, materials, pores=pores)
    out.update(
        {f"k{n}": float(flow["tensor"][i, j]) for (i, j), n in zip(VOIGT, VOIGT_NAMES, strict=True)}
    )
    out["steps"] = flow["steps"]
    out["error"] = flow["error"]
    return out


def along_axis(parameters: dict, axis: int | None = None) -> dict:
    """The six parameters of :func:`equivalent_fluid` from the results of
    :func:`acoustic_parameters`: those along ``axis`` (0, 1 or 2: sound that travels
    along it), or their means over the axes (an isotropic medium)."""
    suffix = "" if axis is None else str(int(axis) + 1)
    out = {
        "porosity": parameters["porosity"],
        "permeability": parameters[f"permeability{suffix}"],
        "tortuosity": parameters[f"tortuosity{suffix}"],
        "viscous_length": parameters[f"viscous_length{suffix}"],
        "thermal_length": parameters["thermal_length"],
    }
    if "thermal_permeability" in parameters:
        out["thermal_permeability"] = parameters["thermal_permeability"]
    return out


def _hertz(frequencies) -> np.ndarray:
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if f.ndim != 1 or not f.size or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequencies: positive values (Hz)")
    return f


def equivalent_fluid(
    frequencies,
    *,
    porosity: float,
    permeability: float,
    tortuosity: float,
    viscous_length: float,
    thermal_length: float,
    thermal_permeability: float | None = None,
    fluid: Fluid = AIR,
) -> dict:
    """The equivalent fluid of a rigid porous frame at the ``frequencies`` (Hz), by the
    model of Johnson, Champoux, Allard and Lafarge (see the module; SI units: lengths in
    metres, permeabilities in square metres): its complex ``density`` and
    ``bulk_modulus`` (time dependence ``exp(i omega t)``), its characteristic
    ``impedance`` and ``wavenumber``, and the ``resistivity`` ``eta / k0`` (Pa s/m²).
    Without ``thermal_permeability``, that of circular pores, ``phi Lambda'^2 / 8``."""
    f = _hertz(frequencies)
    for name, value in (("porosity", porosity), ("permeability", permeability),
                        ("tortuosity", tortuosity), ("viscous_length", viscous_length),
                        ("thermal_length", thermal_length)):  # fmt: skip
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name}: a positive value")
    if not porosity <= 1.0 or not tortuosity >= 1.0 - 1e-9:
        raise ValueError("porosity at most 1, tortuosity at least 1")
    k0p = (
        porosity * thermal_length**2 / 8.0 if thermal_permeability is None else thermal_permeability
    )
    if not k0p > 0.0:
        raise ValueError("thermal_permeability: a positive value")
    omega = 2.0 * np.pi * f
    eta, rho0, pr = fluid.viscosity, fluid.density, fluid.prandtl
    gamma, p0 = fluid.heat_ratio, fluid.pressure
    viscous = np.sqrt(
        1.0 + 4j * tortuosity**2 * permeability**2 * rho0 * omega
        / (eta * viscous_length**2 * porosity**2)
    )  # fmt: skip
    density = (tortuosity * rho0 / porosity) * (
        1.0 + eta * porosity / (1j * omega * rho0 * tortuosity * permeability) * viscous
    )
    thermal = np.sqrt(
        1.0 + 4j * k0p**2 * rho0 * omega * pr / (eta * thermal_length**2 * porosity**2)
    )
    beta = 1.0 + eta * porosity / (1j * omega * rho0 * pr * k0p) * thermal
    bulk = (gamma * p0 / porosity) / (gamma - (gamma - 1.0) / beta)
    return {
        "frequencies": f,
        "density": density,
        "bulk_modulus": bulk,
        "impedance": np.sqrt(density * bulk),
        "wavenumber": omega * np.sqrt(density / bulk),
        "resistivity": float(eta / permeability),
    }


def pore_response(frequencies, shape: str, size: float, *, fluid: Fluid = AIR) -> dict:
    """The exact dynamic ``density`` and ``bulk_modulus`` of the fluid in straight pores
    of constant section at the ``frequencies`` (Hz): ``shape="slit"`` between two planes
    ``size`` apart, or ``"tube"`` of radius ``size`` (Zwikker & Kosten; Stinson 1991). A
    frame of such pores along the sound, of porosity ``phi``, has these divided by
    ``phi``; its parameters are ``k0 = k0' = phi size^2 / 12`` and ``Lambda = Lambda' =
    size`` (slits), ``phi size^2 / 8`` and ``size`` (tubes), with a tortuosity of 1."""
    from scipy.special import jv

    f = _hertz(frequencies)
    if not size > 0.0:
        raise ValueError("size: positive")
    omega = 2.0 * np.pi * f

    def mean_profile(q):  # the mean over the section of the response to a unit forcing
        if shape == "slit":
            s = 0.5 * size * np.sqrt(1j * q)
            return 1.0 - np.tanh(s) / s
        if shape == "tube":
            s = size * np.sqrt(-1j * q)
            return 1.0 - 2.0 * jv(1, s) / (s * jv(0, s))
        raise ValueError("shape: 'slit' or 'tube'")

    viscous = mean_profile(omega * fluid.density / fluid.viscosity)
    thermal = mean_profile(omega * fluid.density * fluid.prandtl / fluid.viscosity)
    gamma = fluid.heat_ratio
    return {
        "frequencies": f,
        "density": fluid.density / viscous,
        "bulk_modulus": gamma * fluid.pressure / (gamma - (gamma - 1.0) * thermal),
    }


def surface_absorption(impedance, incidence=0.0, *, fluid: Fluid = AIR, limit: float | None = None):
    """The absorption coefficient of a surface of normal ``impedance`` (a number or an
    array; a locally reacting surface) for a plane wave at the angle ``incidence`` from
    its normal (radians), ``1 - |(Z cos θ - Z0) / (Z cos θ + Z0)|²``, or in a diffuse
    field (``incidence="diffuse"``): Paris' average ``∫ α(θ) sin 2θ dθ`` over the angles
    up to ``limit`` (by default π/2), normalized. For a real impedance ``Z = ζ Z0`` it is
    ``8 / ζ² (1 + ζ - 1 / (1 + ζ) - 2 ln(1 + ζ))``."""
    z = np.asarray(impedance, dtype=complex) / fluid.impedance

    def at(theta):
        cosine = np.cos(theta)
        return 1.0 - np.abs((z * cosine - 1.0) / (z * cosine + 1.0)) ** 2

    if isinstance(incidence, str):
        return _diffuse(at, incidence, limit)[0]
    return at(_angle(incidence))


def _angle(incidence) -> float:
    theta = float(incidence)
    if not 0.0 <= theta < 0.5 * np.pi:
        raise ValueError("incidence: an angle from the normal in [0, π/2) (radians), or 'diffuse'")
    return theta


def _diffuse(at, incidence: str, limit):
    """Paris' average of the absorption ``at(θ)`` over the angles of a diffuse field, the
    angles of the quadrature and the absorption at each."""
    if incidence != "diffuse":
        raise ValueError("incidence: an angle from the normal (radians), or 'diffuse'")
    top = 0.5 * np.pi if limit is None else float(limit)
    if not 0.0 < top <= 0.5 * np.pi:
        raise ValueError("limit: an angle in (0, π/2]")
    nodes, weights = np.polynomial.legendre.leggauss(64)
    angles = 0.5 * top * (nodes + 1.0)
    weights = 0.5 * top * weights * np.sin(2.0 * angles)
    values = np.array([at(theta) for theta in angles])
    return np.tensordot(weights, values, axes=1) / weights.sum(), angles, values


def absorption(
    frequencies,
    thickness: float,
    *,
    incidence=0.0,
    transverse: dict | None = None,
    limit: float | None = None,
    fluid: Fluid = AIR,
    **parameters,
) -> dict:
    """The sound absorption of a layer of ``thickness`` (m) of a rigid porous frame on a
    rigid wall, at the ``frequencies`` (Hz): its ``absorption`` coefficient,
    ``surface_impedance`` and ``reflection`` coefficient, with the equivalent fluid of
    :func:`equivalent_fluid` (its ``parameters``: those of :func:`along_axis` across the
    layer).

    ``incidence``: the angle of the plane wave from the normal of the layer (radians; 0
    by default), or ``"diffuse"``: the absorption in a diffuse field, Paris' average of
    the absorption over the angles up to ``limit`` (by default π/2; ``absorption`` is
    then that average, with the ``angles`` of the quadrature and the
    ``absorption_by_angle``). The wave is refracted in the layer: with the wavenumber
    ``k0 sin θ`` along it, ``k_x² = ρ_n (ω² / K - k0² sin²θ / ρ_t)`` across it and ``Z_s =
    -i ω ρ_n / (k_x tan(k_x d))``. ``transverse``: the parameters along the layer of an
    anisotropic frame (those of :func:`along_axis` for an axis in its plane: its density
    ``ρ_t`` there differs from ``ρ_n`` across it; fibres, layered pores), by default
    those across it (an isotropic frame)."""
    if not thickness > 0.0:
        raise ValueError("thickness: positive")
    medium = equivalent_fluid(frequencies, fluid=fluid, **parameters)
    omega = 2.0 * np.pi * medium["frequencies"]
    density, bulk = medium["density"], medium["bulk_modulus"]
    along = density
    if transverse is not None:
        along = equivalent_fluid(frequencies, fluid=fluid, **transverse)["density"]

    def surface(theta):
        slip = (omega / fluid.sound_speed * np.sin(theta)) ** 2
        across = np.sqrt(density * (omega**2 / bulk - slip / along))
        return -1j * omega * density / (across * np.tan(across * thickness))

    def reflection(theta):
        z = surface(theta) * np.cos(theta)
        return (z - fluid.impedance) / (z + fluid.impedance)

    if isinstance(incidence, str):
        mean, angles, values = _diffuse(
            lambda theta: 1.0 - np.abs(reflection(theta)) ** 2, incidence, limit
        )
        return {
            "frequencies": medium["frequencies"],
            "absorption": mean,
            "angles": angles,
            "absorption_by_angle": values,
        }
    theta = _angle(incidence)
    return {
        "frequencies": medium["frequencies"],
        "absorption": 1.0 - np.abs(reflection(theta)) ** 2,
        "surface_impedance": surface(theta),
        "reflection": reflection(theta),
    }
