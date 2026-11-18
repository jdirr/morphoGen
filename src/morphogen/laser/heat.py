"""Heat conduction of the laser treatment of a sheet: a moving beam on a plate.

Units: lengths in millimetres, times in seconds, temperatures in degrees Celsius, powers
in watts. The thermal properties are given in SI units, as in the handbooks
(W m⁻¹ K⁻¹, kg m⁻³, J kg⁻¹ K⁻¹, W m⁻² K⁻¹), and converted.

**The quasi-steady track** (:func:`moving_source`, :class:`TrackKernel`). A beam of
absorbed power ``P`` moves at speed ``v`` along ``x`` on a plate of thickness ``h`` with
constant properties (conductivity ``k``, volumetric heat ``ρc``, diffusivity
``a = k/ρc``), on its upper face (``single``) or on both faces at once (``symmetric``:
two aligned beams, the double-laser platform of Wang et al. 2022). An instantaneous
source of energy ``dE`` on the upper face spreads as ``dE/ρc G(x, y; τ) Z(z; τ)
exp(−βτ)`` after a time ``τ``:

* in the plane, the beam profile convolved with the heat kernel: a Gaussian beam of
  1/e² radius ``w`` (standard deviation ``s0 = w/2``) gives
  ``G = exp(−r²/(2 s²)) / (2π s²)``, ``s² = s0² + 2aτ`` (Cline & Anthony 1977, Eagar &
  Tsai 1983); a rectangular top hat of half sizes ``bx``, ``by`` gives the product of
  ``(erf((x + b)/√(4aτ)) − erf((x − b)/√(4aτ))) / (4b)``;
* through the thickness, the adiabatic faces reflect the source: images at ``z = 2mh``,
  ``Z = Σ 2 exp(−(z − 2mh)²/(4aτ)) / √(4πaτ) = (1/h)(1 + 2 Σ exp(−n²π²aτ/h²)
  cos(nπz/h))`` (the series where it converges faster); a beam on the lower face adds
  ``Z(h − z)``;
* the faces lose heat by convection and linearized radiation, ``h_f (T − T0)`` each,
  approximated by the uniform loss rate ``β = 2 h_f/(ρc h)`` of a thin plate.

In the frame of the beam (``ξ = x − vt``) the quasi-steady field is
``T − T0 = P/ρc ∫₀^∞ G(ξ + vτ, y; τ) Z(z; τ) exp(−βτ) dτ``, integrated on a logarithmic
grid of ``τ``. Its limits are Rosenthal's (1946) solutions, checked in the tests: a point
source on a thick plate, ``T − T0 = P/(2πkR) exp(−v(R + ξ)/(2a))``
(:func:`rosenthal_point`), and a line source through a thin plate with surface losses,
``T − T0 = P/(2πkh) exp(−vξ/(2a)) K0(r √(v²/(4a²) + β/a))`` (Rykalin;
:func:`rosenthal_line`). A point at the distance ``d`` of the axis of a long straight
track sees the thermal cycle ``T(t) = T(ξ = −vt, d, z)``: :class:`TrackKernel` tabulates
these cycles over ``d`` and the depths of the layers, from which the peak temperatures,
the widths of the zones that reach a temperature (hardened, melted) and the kinetic
integrals follow.

**The transient sheet** (:class:`TransientHeat`). Arbitrary paths, their sequence, the
heat left by the previous tracks and the edges of the blank: finite volumes on a grid of
the blank (cells through the thickness for a single beam, one for a thin sheet),
locally one-dimensional implicit steps (Yanenko's splitting: unconditionally stable,
tridiagonal solves along x, y and z; Numba), adiabatic edges, the beam deposited on the
faces as its profile averaged over the positions it takes during a step, the faces
cooled by ``h_f``. Each step can update a metallurgical state (:mod:`morphogen.laser.metallurgy`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from numba import njit, prange
from scipy.special import erf, k0

#: Stefan–Boltzmann constant (W m⁻² K⁻⁴).
STEFAN_BOLTZMANN = 5.670374419e-8
#: Beam profiles.
PROFILES = ("gaussian", "tophat")
#: Laser configurations: one beam on the upper face, or two aligned beams on both faces.
CONFIGURATIONS = ("single", "symmetric")


@dataclass(frozen=True)
class ThermalProperties:
    """Thermal properties of the sheet, in SI units: conductivity (W m⁻¹ K⁻¹), density
    (kg m⁻³), mean specific heat over the thermal cycle (J kg⁻¹ K⁻¹), convection
    coefficient of each face (W m⁻² K⁻¹) and emissivity (radiation linearized about
    ``radiation_temperature``, °C)."""

    conductivity: float = 25.0
    density: float = 7700.0
    heat_capacity: float = 600.0
    convection: float = 20.0
    emissivity: float = 0.0
    radiation_temperature: float = 800.0

    def __post_init__(self):
        for name in ("conductivity", "density", "heat_capacity"):
            if not getattr(self, name) > 0:
                raise ValueError(f"thermal properties: {name} must be positive")
        if self.convection < 0 or not 0 <= self.emissivity <= 1:
            raise ValueError("thermal properties: convection >= 0, emissivity in [0, 1]")

    @property
    def k(self) -> float:
        """Conductivity in W mm⁻¹ K⁻¹."""
        return self.conductivity * 1e-3

    @property
    def rho_c(self) -> float:
        """Volumetric heat in J mm⁻³ K⁻¹."""
        return self.density * self.heat_capacity * 1e-9

    @property
    def diffusivity(self) -> float:
        """Thermal diffusivity in mm² s⁻¹."""
        return self.k / self.rho_c

    def film(self, ambient: float = 20.0) -> float:
        """Heat transfer coefficient of a face in W mm⁻² K⁻¹: convection plus the
        radiation ``εσ(T_r² + T0²)(T_r + T0)`` linearized about ``radiation_temperature``."""
        t_r, t_0 = self.radiation_temperature + 273.15, ambient + 273.15
        rad = self.emissivity * STEFAN_BOLTZMANN * (t_r**2 + t_0**2) * (t_r + t_0)
        return (self.convection + rad) * 1e-6

    def loss_rate(self, thickness: float, ambient: float = 20.0) -> float:
        """Loss rate ``β = 2 h_f / (ρc h)`` (s⁻¹) of a plate cooled on both faces."""
        return 2.0 * self.film(ambient) / (self.rho_c * float(thickness))

    @classmethod
    def from_config(cls, data) -> ThermalProperties:
        if isinstance(data, ThermalProperties):
            return data
        return cls(**dict(data or {}))


@dataclass(frozen=True)
class Beam:
    """A continuous laser beam: incident power per beam (W), speed (mm/s), radius (mm:
    the 1/e² radius of a Gaussian beam, the half side of a top hat), absorptivity, and
    for a rectangular top hat its half length along the motion (``length``, default the
    radius)."""

    power: float
    speed: float
    radius: float
    absorptivity: float = 0.35
    profile: str = "gaussian"
    length: float | None = None

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"beam profile: one of {PROFILES}")
        if not (self.power >= 0 and self.speed > 0 and self.radius > 0):
            raise ValueError("beam: power >= 0, speed > 0 and radius > 0")
        if not 0 < self.absorptivity <= 1:
            raise ValueError("beam: absorptivity in (0, 1]")
        if self.length is not None and not self.length > 0:
            raise ValueError("beam: length must be positive")

    @property
    def absorbed(self) -> float:
        """Absorbed power (W) of one beam."""
        return self.power * self.absorptivity

    @property
    def line_energy(self) -> float:
        """Linear energy density ``P/v`` (J/mm), incident, of one beam."""
        return self.power / self.speed

    @property
    def interaction_time(self) -> float:
        """Time (s) the beam takes to pass over a point: its diameter over its speed."""
        return 2.0 * (self.length or self.radius) / self.speed

    def with_(self, **changes) -> Beam:
        return replace(self, **changes)

    @classmethod
    def from_config(cls, data) -> Beam:
        if isinstance(data, Beam):
            return data
        return cls(**dict(data))


# -- closed forms --------------------------------------------------------------------------


def rosenthal_point(
    xi, y, z, power: float, speed: float, props: ThermalProperties, ambient: float = 20.0
):
    """Rosenthal's point source on the surface of a semi-infinite body (absorbed power
    ``power``): ``T0 + P/(2πkR) exp(−v(R + ξ)/(2a))``, ``R`` the distance to the source."""
    a, k = props.diffusivity, props.k
    xi, y, z = (np.asarray(v, dtype=float) for v in (xi, y, z))
    r = np.sqrt(xi * xi + y * y + z * z)
    return ambient + power / (2.0 * math.pi * k * r) * np.exp(-speed * (r + xi) / (2.0 * a))


def rosenthal_line(
    xi,
    y,
    power: float,
    speed: float,
    thickness: float,
    props: ThermalProperties,
    ambient: float = 20.0,
    *,
    losses: bool = True,
):
    """The line source through a thin plate (Rosenthal 1946; with the surface losses of
    Rykalin): ``T0 + P/(2πkh) exp(−vξ/(2a)) K0(r √(v²/(4a²) + β/a))``."""
    a, k, h = props.diffusivity, props.k, float(thickness)
    beta = props.loss_rate(h, ambient) if losses else 0.0
    xi, y = np.asarray(xi, dtype=float), np.asarray(y, dtype=float)
    r = np.sqrt(xi * xi + y * y)
    arg = r * math.sqrt(speed * speed / (4.0 * a * a) + beta / a)
    return ambient + power / (2.0 * math.pi * k * h) * np.exp(-speed * xi / (2.0 * a)) * k0(arg)


# -- the quasi-steady field of a moving beam -----------------------------------------------


def _depth(z, tau, a: float, h: float):
    """``Z(z; τ)`` of a source on the face ``z = 0`` of a plate ``0 <= z <= h`` with
    adiabatic faces (1/mm): images for short times, the cosine series for long ones."""
    u = a * tau / (h * h)
    out = np.empty(np.broadcast(z, tau).shape)
    short = np.broadcast_to(u < 0.2, out.shape)
    zz, tt = np.broadcast_arrays(z, tau)
    if short.any():
        zs, ts = zz[short], tt[short]
        s = np.zeros_like(zs)
        four_at = 4.0 * a * ts
        for m in range(-3, 4):
            d = zs - 2.0 * m * h
            s += np.exp(-d * d / four_at)
        out[short] = 2.0 * s / np.sqrt(math.pi * four_at)
    if (~short).any():
        zl, ul = zz[~short], np.broadcast_to(u, out.shape)[~short]
        s = np.ones_like(zl)
        for n in range(1, 9):
            s += 2.0 * np.exp(-((n * math.pi) ** 2) * ul) * np.cos(n * math.pi * zl / h)
        out[~short] = s / h
    return out


def _plane(x, y, tau, a: float, beam: Beam):
    """``G(x, y; τ)`` (1/mm²): the beam profile spread by the heat kernel."""
    if beam.profile == "gaussian":
        s2 = 0.25 * beam.radius**2 + 2.0 * a * tau
        return np.exp(-(x * x + y * y) / (2.0 * s2)) / (2.0 * math.pi * s2)
    bx, by = beam.length or beam.radius, beam.radius
    sq = np.sqrt(4.0 * a * tau)
    gx = (erf((x + bx) / sq) - erf((x - bx) / sq)) / (4.0 * bx)
    gy = (erf((y + by) / sq) - erf((y - by) / sq)) / (4.0 * by)
    return gx * gy


def _tau_grid(beam: Beam, a: float, h: float, beta: float, reach: float, nodes: int):
    """Logarithmic grid of the delays ``τ`` and its trapezoidal weights (in ``ln τ``)."""
    v = beam.speed
    s0 = max(0.5 * (beam.length or beam.radius), 1e-6)
    lo = 1e-6 * min(s0 * s0, h * h) / a
    hi = max(60.0 * 4.0 * a / (v * v), 4.0 * reach / v + 10.0 * s0 / v, 1e-3)
    if beta > 0:
        hi = min(hi, max(40.0 / beta, 4.0 * reach / v))
    tau = np.geomspace(lo, hi, int(nodes))
    ds = math.log(hi / lo) / (nodes - 1)
    w = np.full(nodes, ds)
    w[0] = w[-1] = 0.5 * ds
    return tau, w * tau


def _beams(beam: Beam, bottom, configuration: str):
    """The beams on the upper and lower faces: ``bottom`` if given, else the same beam
    for the ``symmetric`` configuration, none for ``single``."""
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"configuration: one of {CONFIGURATIONS}")
    if bottom is None and configuration == "symmetric":
        bottom = beam
    if bottom is not None and not math.isclose(bottom.speed, beam.speed):
        raise ValueError("the beams of both faces move together: the same speed")
    return beam, bottom


def moving_source(
    xi,
    y,
    z,
    beam: Beam,
    thickness: float,
    props: ThermalProperties,
    *,
    bottom: Beam | None = None,
    configuration: str = "single",
    ambient: float = 20.0,
    nodes: int = 1600,
    chunk: int = 4096,
):
    """Quasi-steady temperature (°C) at the points ``(ξ, y, z)`` (mm) in the frame of a
    beam moving along +x on a plate of thickness ``thickness``, its upper face at
    ``z = 0``: one beam on the upper face (``single``), the same beam on both faces
    (``symmetric``), or ``bottom``, another beam aligned with it on the lower face (a
    fibre laser above and a diode below, say)."""
    top, bottom = _beams(beam, bottom, configuration)
    h = float(thickness)
    if not h > 0:
        raise ValueError("thickness must be positive")
    xi, y, z = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (xi, y, z)))
    shape = xi.shape
    xi, y, z = xi.ravel(), y.ravel(), z.ravel()
    a = props.diffusivity
    beta = props.loss_rate(h, ambient)
    reach = float(np.max(np.abs(xi))) if xi.size else 0.0
    narrow = top if bottom is None else min((top, bottom), key=lambda b: b.radius)
    tau, w = _tau_grid(narrow, a, h, beta, reach, nodes)
    # the depth factor depends on z and tau only: once per depth
    zu, zi = np.unique(z, return_inverse=True)
    damp = (w * np.exp(-beta * tau))[None, :]
    faces = [(top, _depth(zu[:, None], tau[None, :], a, h) * damp)]
    if bottom is not None:
        zb = _depth(h - zu[:, None], tau[None, :], a, h) * damp
        same = (
            bottom.profile == top.profile
            and bottom.radius == top.radius
            and (bottom.length == top.length)
        )
        if same:  # one plane kernel: the powers weight the depth factors
            faces = [(top, faces[0][1] + zb * (bottom.absorbed / max(top.absorbed, 1e-300)))]
            if top.absorbed == 0:
                faces = [(bottom, zb)]
        else:
            faces.append((bottom, zb))
    out = np.zeros(xi.size)
    v = top.speed
    for b, ztab in faces:
        part = np.empty(xi.size)
        for start in range(0, xi.size, chunk):
            sl = slice(start, start + chunk)
            g = _plane(xi[sl, None] + v * tau[None, :], y[sl, None], tau[None, :], a, b)
            part[sl] = np.einsum("ij,ij->i", g, ztab[zi[sl]])
        out += b.absorbed * part
    return (ambient + out / props.rho_c).reshape(shape)


def moving_source_grid(
    xi,
    y,
    z,
    beam: Beam,
    thickness: float,
    props: ThermalProperties,
    *,
    bottom: Beam | None = None,
    configuration: str = "single",
    ambient: float = 20.0,
    nodes: int = 1600,
) -> np.ndarray:
    """:func:`moving_source` on the grid ``xi x y x z`` (1D arrays), shape
    ``(len(xi), len(y), len(z))``: the plane kernel is separable, so that the integral
    over the delays is a product of matrices for each depth (no exponential per point)."""
    top, bottom = _beams(beam, bottom, configuration)
    h = float(thickness)
    if not h > 0:
        raise ValueError("thickness must be positive")
    xi, y, z = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (xi, y, z))
    a = props.diffusivity
    beta = props.loss_rate(h, ambient)
    reach = float(np.max(np.abs(xi))) if xi.size else 0.0
    narrow = top if bottom is None else min((top, bottom), key=lambda b: b.radius)
    tau, w = _tau_grid(narrow, a, h, beta, reach, nodes)
    damp = w * np.exp(-beta * tau)
    out = np.zeros((xi.size, y.size, z.size))
    v = top.speed
    for b, zz in ((top, z), (bottom, h - z)):
        if b is None or b.absorbed == 0:
            continue
        ztab = _depth(zz[:, None], tau[None, :], a, h) * damp[None, :]
        gx, gy = _plane_factors_moving(xi, y, tau, v, a, b)
        for k in range(z.size):
            out[:, :, k] += b.absorbed * ((gx * ztab[k][None, :]) @ gy.T)
    return ambient + out / props.rho_c


def _plane_factors_moving(xi, y, tau, v: float, a: float, beam: Beam):
    """``gx(ξ + vτ; τ)`` (len(xi) x len(tau)) and ``gy(y; τ)`` (len(y) x len(tau))."""
    x = xi[:, None] + v * tau[None, :]
    if beam.profile == "gaussian":
        s2 = 0.25 * beam.radius**2 + 2.0 * a * tau[None, :]
        gx = np.exp(-(x * x) / (2.0 * s2)) / (2.0 * math.pi * s2)
        gy = np.exp(-(y[:, None] ** 2) / (2.0 * s2))
        return gx, gy
    bx, by = beam.length or beam.radius, beam.radius
    sq = np.sqrt(4.0 * a * tau)[None, :]
    gx = (erf((x + bx) / sq) - erf((x - bx) / sq)) / (4.0 * bx)
    gy = (erf((y[:, None] + by) / sq) - erf((y[:, None] - by) / sq)) / (4.0 * by)
    return gx, gy


@dataclass
class TrackKernel:
    """The thermal cycles that a long straight track imposes on the points of the sheet:
    ``temperature[i, j, k]`` at the distance ``distances[i]`` (mm) of its axis, the time
    ``times[j]`` (s; 0 when the beam passes the point's abscissa) and the depth
    ``depths[k]`` (mm, the middles of the layers), with their peaks; ``bottom`` is the
    beam of the lower face, if any."""

    beam: Beam
    bottom: Beam | None
    thickness: float
    ambient: float
    distances: np.ndarray
    depths: np.ndarray
    times: np.ndarray
    temperature: np.ndarray
    peak: np.ndarray = field(init=False)

    def __post_init__(self):
        self.peak = self.temperature.max(axis=1)

    @property
    def layers(self) -> int:
        return len(self.depths)

    @property
    def reach(self) -> float:
        """The largest distance tabulated: beyond it the track leaves the sheet as it is."""
        return float(self.distances[-1])

    @property
    def configuration(self) -> str:
        return "single" if self.bottom is None else "symmetric"

    def peak_at(self, distance, layer: int | None = None) -> np.ndarray:
        """Peak temperature at distances from the axis (``(..., layers)``, or one layer)."""
        d = np.abs(np.asarray(distance, dtype=float))
        cols = range(self.layers) if layer is None else [int(layer)]
        out = np.stack(
            [np.interp(d, self.distances, self.peak[:, k], right=self.ambient) for k in cols],
            axis=-1,
        )
        return out if layer is None else out[..., 0]

    def width(self, temperature: float, layer: int | None = None) -> float:
        """Width (mm) of the band whose peak temperature reaches ``temperature`` (in one
        layer, or in any)."""
        peak = self.peak.max(axis=1) if layer is None else self.peak[:, int(layer)]
        above = peak >= temperature
        if not above[0]:
            return 0.0
        if above.all():
            return 2.0 * self.reach
        i = int(np.argmin(above))  # the first distance below
        d0, d1, t0, t1 = self.distances[i - 1], self.distances[i], peak[i - 1], peak[i]
        return 2.0 * float(d0 + (t0 - temperature) / (t0 - t1) * (d1 - d0))

    def through_width(self, temperature: float) -> float:
        """Width (mm) of the band whose peak temperature reaches ``temperature`` in every
        layer: treated through the whole thickness."""
        return (
            self.width(temperature, None)
            if self.layers == 1
            else min(self.width(temperature, k) for k in range(self.layers))
        )

    def cycles(self, distance) -> np.ndarray:
        """Thermal cycles at distances from the axis: ``(points, times, layers)``,
        interpolated linearly between the tabulated distances (ambient beyond)."""
        d = np.abs(np.asarray(distance, dtype=float)).ravel()
        x = np.clip(d, 0.0, self.reach)
        i = np.clip(
            np.searchsorted(self.distances, x, side="right") - 1, 0, len(self.distances) - 2
        )
        d0, d1 = self.distances[i], self.distances[i + 1]
        f = ((x - d0) / (d1 - d0))[:, None, None]
        out = (1.0 - f) * self.temperature[i] + f * self.temperature[i + 1]
        out[d > self.reach] = self.ambient
        return out

    def kinetic_integral(self, activation: float, distance=0.0) -> np.ndarray:
        """``∫ exp(−Q/(R T(t))) dt`` (s) of the cycles at distances, ``Q`` in J/mol (per
        layer): the effective time of a thermally activated process; e.g. the diffusion
        length of a species of diffusivity ``D0 exp(−Q/RT)`` is ``√(D0 × it)``."""
        cyc = self.cycles(distance) + 273.15
        f = np.exp(-float(activation) / (8.314462618 * cyc))
        return np.trapezoid(f, self.times, axis=1)

    def cooling_time(self, distance=0.0, upper: float = 800.0, lower: float = 500.0) -> np.ndarray:
        """Time (s) to cool from ``upper`` to ``lower`` (°C) after the peak, ``t8/5`` by
        default (NaN where the peak stays below ``upper``)."""
        cyc = self.cycles(distance)
        out = np.full(cyc.shape[::2], np.nan)
        for p in range(cyc.shape[0]):
            for k in range(cyc.shape[2]):
                c = cyc[p, :, k]
                j = int(np.argmax(c))
                if c[j] < upper:
                    continue
                tail, tt = c[j:], self.times[j:]
                if tail[-1] > lower:
                    continue
                t_up = np.interp(-upper, -tail, tt)
                t_lo = np.interp(-lower, -tail, tt)
                out[p, k] = t_lo - t_up
        return out


def track_kernel(
    beam: Beam,
    thickness: float,
    props: ThermalProperties,
    *,
    bottom: Beam | None = None,
    configuration: str = "single",
    layers: int = 1,
    ambient: float = 20.0,
    cutoff: float | None = None,
    distances: int = 48,
    times: int = 240,
    nodes: int = 1200,
    max_reach: float | None = None,
) -> TrackKernel:
    """Tabulate the thermal cycles of a long straight track (see :class:`TrackKernel`):
    the distances up to where the peak falls to ``cutoff`` (°C; default: a tenth of the
    peak rise above ambient, at least 100 °C above it; at most ``max_reach`` mm), the
    times from ahead of the beam to a tail of cooling, ``layers`` layers through the
    thickness. The beams: as in :func:`moving_source`."""
    top, bottom = _beams(beam, bottom, configuration)
    if layers < 1:
        raise ValueError("layers: at least one")
    h = float(thickness)
    a = props.diffusivity
    big = top if bottom is None else max((top, bottom), key=lambda b: b.length or b.radius)
    s0 = 0.5 * (big.length or big.radius)
    depths = (np.arange(layers) + 0.5) * h / layers
    max_reach = float(max_reach) if max_reach is not None else max(30.0 * s0, 10.0 * h)
    kw = {"bottom": bottom, "ambient": ambient, "nodes": nodes}

    # where is the peak behind the beam, how far does the heat reach: a coarse look first
    far = max(4.0 * max_reach, 40.0 * s0)
    xs = np.concatenate(
        [np.linspace(3.0 * s0, 0.0, 12, endpoint=False), -np.geomspace(1e-3 * far, far, 40)]
    )
    coarse_d = np.concatenate([[0.0], np.geomspace(0.05 * s0, max_reach, 30)])
    pk = moving_source_grid(xs, coarse_d, depths, top, h, props, **kw).max(axis=0).max(axis=1)
    rise = pk[0] - ambient
    if cutoff is None:
        cutoff = ambient + max(0.1 * rise, 100.0)
    if pk[-1] >= cutoff:
        reach = max_reach
    else:
        reach = float(coarse_d[int(np.argmax(pk < cutoff))])
    reach = max(reach, 2.0 * s0)
    knee = min(reach, 3.0 * s0)
    dist = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, knee, max(2, distances // 2)),
                np.linspace(knee, reach, max(2, distances - distances // 2)),
            ]
        )
    )
    # times: dense around the passage, geometric afterwards (the slow tail of thin plates)
    t_int = 2.0 * s0 * 2.0 / top.speed
    t_ahead = 3.0 * t_int
    t_tail = max(40.0 * t_int, 4.0 * reach / top.speed + 4.0 * h * h / a)
    dense = np.linspace(-t_ahead, 4.0 * t_int, times // 2)
    tail = np.geomspace(4.0 * t_int, t_tail, times - times // 2 + 1)[1:]
    tt = np.concatenate([dense, tail])
    temp = moving_source_grid(-top.speed * tt, dist, depths, top, h, props, **kw)
    temp = np.ascontiguousarray(np.transpose(temp, (1, 0, 2)))  # distances x times x layers
    return TrackKernel(top, bottom, h, float(ambient), dist, depths, tt, temp)


# -- transient finite volumes --------------------------------------------------------------


@njit(cache=True)
def _thomas(lower, diag, upper, rhs, n):  # pragma: no cover - compiled
    """Tridiagonal solve in place (rhs holds the solution)."""
    c = np.empty(n)
    c[0] = upper[0] / diag[0]
    rhs[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / m
        rhs[i] = (rhs[i] - lower[i] * rhs[i - 1]) / m
    for i in range(n - 2, -1, -1):
        rhs[i] -= c[i] * rhs[i + 1]


@njit(parallel=True, cache=True)
def _sweep(t, axis, r, loss, ambient):  # pragma: no cover - compiled
    """Implicit 1D diffusion along ``axis`` of every line of ``t`` (nx, ny, nz), Neumann
    ends; ``loss`` (along z only) adds the cooling of the two outer faces."""
    nx, ny, nz = t.shape
    n = t.shape[axis]
    if axis == 0:
        m1, m2 = ny, nz
    elif axis == 1:
        m1, m2 = nx, nz
    else:
        m1, m2 = nx, ny
    for idx in prange(m1 * m2):
        p, q = idx // m2, idx % m2
        lower = np.full(n, -r)
        upper = np.full(n, -r)
        diag = np.full(n, 1.0 + 2.0 * r)
        rhs = np.empty(n)
        for i in range(n):
            if axis == 0:
                rhs[i] = t[i, p, q]
            elif axis == 1:
                rhs[i] = t[p, i, q]
            else:
                rhs[i] = t[p, q, i]
        lower[0] = 0.0
        upper[n - 1] = 0.0
        diag[0] -= r
        diag[n - 1] -= r
        if loss > 0.0:
            diag[0] += loss
            rhs[0] += loss * ambient
            diag[n - 1] += loss
            rhs[n - 1] += loss * ambient
        _thomas(lower, diag, upper, rhs, n)
        for i in range(n):
            if axis == 0:
                t[i, p, q] = rhs[i]
            elif axis == 1:
                t[p, i, q] = rhs[i]
            else:
                t[p, q, i] = rhs[i]


@dataclass
class TransientResult:
    """Fields of a transient run: the final and peak temperatures (nx, ny, nz), the
    time of the end of the run, and the metallurgical state when a model was given."""

    temperature: np.ndarray
    peak: np.ndarray
    time: float
    steps: int
    state: dict | None = None


class TransientHeat:
    """Finite volumes on a blank ``[x0, x0 + lx] x [y0, y0 + ly]`` of thickness
    ``thickness``: ``cell`` in the plane (mm), ``layers`` cells through the thickness
    (1 for a thin sheet treated on both faces)."""

    def __init__(
        self,
        size,
        thickness: float,
        props: ThermalProperties,
        *,
        cell: float,
        layers: int = 1,
        origin=(0.0, 0.0),
        ambient: float = 20.0,
    ):
        self.size = (float(size[0]), float(size[1]))
        self.origin = (float(origin[0]), float(origin[1]))
        self.thickness = float(thickness)
        self.props = props
        self.ambient = float(ambient)
        self.nx = max(1, round(self.size[0] / cell))
        self.ny = max(1, round(self.size[1] / cell))
        self.nz = int(layers)
        self.dx, self.dy = self.size[0] / self.nx, self.size[1] / self.ny
        self.dz = self.thickness / self.nz
        self.x = self.origin[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.y = self.origin[1] + (np.arange(self.ny) + 0.5) * self.dy
        self.z = (np.arange(self.nz) + 0.5) * self.dz

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    def _flux(self, beam: Beam, centres) -> np.ndarray:
        """Absorbed flux (W/mm²) of the cells of a face, averaged over beam centres. Each
        centre's profile is sampled at the cell centres of a window that may overhang the
        blank, normalized to unit energy over the window (the discrete profile of a
        coarse grid keeps the power), then cut by the blank: the part of the beam that
        falls off the sheet is lost."""
        q = np.zeros((self.nx, self.ny))
        if beam.profile == "gaussian":
            reach = 1.6 * beam.radius + 2.0 * max(self.dx, self.dy)
        else:
            reach = math.hypot(beam.length or beam.radius, beam.radius) + 2.0 * max(
                self.dx, self.dy
            )
        x0, y0 = self.origin
        for cx, cy, ux, uy in centres:
            ia = math.floor((cx - reach - x0) / self.dx)
            ib = math.ceil((cx + reach - x0) / self.dx) + 1
            ja = math.floor((cy - reach - y0) / self.dy)
            jb = math.ceil((cy + reach - y0) / self.dy) + 1
            gx, gy = np.meshgrid(
                x0 + (np.arange(ia, ib) + 0.5) * self.dx - cx,
                y0 + (np.arange(ja, jb) + 0.5) * self.dy - cy,
                indexing="ij",
            )
            if beam.profile == "gaussian":
                w = beam.radius
                prof = np.exp(-2.0 * (gx * gx + gy * gy) / (w * w))
            else:
                along = np.abs(gx * ux + gy * uy)
                across = np.abs(-gx * uy + gy * ux)
                prof = ((along <= (beam.length or beam.radius)) & (across <= beam.radius)).astype(
                    float
                )
            total = prof.sum() * self.dx * self.dy
            if total <= 0:  # a top hat narrower than the cells: its nearest cell
                prof[np.unravel_index(np.argmin(gx * gx + gy * gy), prof.shape)] = 1.0
                total = self.dx * self.dy
            prof /= total
            i0, i1 = max(ia, 0), min(ib, self.nx)
            j0, j1 = max(ja, 0), min(jb, self.ny)
            if i0 < i1 and j0 < j1:
                q[i0:i1, j0:j1] += prof[i0 - ia : i1 - ia, j0 - ja : j1 - ja]
        return q * (beam.absorbed / max(len(centres), 1))

    def run(
        self,
        segments,
        *,
        configuration: str = "single",
        model=None,
        state=None,
        dt_max: float = 0.5,
        cooling: float | None = None,
        substeps: int = 4,
        callback=None,
    ) -> TransientResult:
        """Run a sequence of laser moves. ``segments``: iterable of
        ``(points, beam, bottom, pause)``: a polyline (n x 2, mm) followed at the speed of
        ``beam`` (upper face) with ``bottom`` on the lower face (None: none, or the same
        beam in the ``symmetric`` configuration), then ``pause`` seconds with the lasers
        off (the move to the next track). ``model``: a metallurgical model updated at each
        step (its ``state`` returned). ``cooling``: after the last segment, cool until the
        hottest cell falls below this temperature (°C)."""
        if configuration not in CONFIGURATIONS:
            raise ValueError(f"configuration: one of {CONFIGURATIONS}")
        a = self.props.diffusivity
        film = self.props.film(self.ambient)
        rho_c = self.props.rho_c
        t = np.full(self.shape, self.ambient)
        peak = t.copy()
        if model is not None and state is None:
            state = model.initial_state(t.size)
        time, steps = 0.0, 0
        cell = min(self.dx, self.dy)

        def advance(dt, flux_top, flux_bottom):
            nonlocal time, steps
            old = t.copy()
            depth_top = self.thickness if self.nz == 1 else self.dz
            if flux_top is not None:
                t[:, :, 0] += flux_top * dt / (rho_c * depth_top)
            if flux_bottom is not None:
                t[:, :, -1] += flux_bottom * dt / (rho_c * depth_top)
            if self.nx > 1:
                _sweep(t, 0, a * dt / self.dx**2, 0.0, self.ambient)
            if self.ny > 1:
                _sweep(t, 1, a * dt / self.dy**2, 0.0, self.ambient)
            rz = a * dt / self.dz**2 if self.nz > 1 else 0.0
            _sweep(t, 2, rz, film * dt / (rho_c * depth_top), self.ambient)
            np.maximum(peak, t, out=peak)
            if model is not None:
                model.step(state, old.reshape(-1), t.reshape(-1), dt)
            time += dt
            steps += 1
            if callback is not None:
                callback(time, t)

        for points, beam, bottom, pause in segments:
            top, low = _beams(beam, bottom, configuration)
            pts = np.asarray(points, dtype=float).reshape(-1, 2)
            seg = np.diff(pts, axis=0)
            lengths = np.hypot(seg[:, 0], seg[:, 1])
            total = float(lengths.sum())
            if total > 0:
                cum = np.concatenate([[0.0], np.cumsum(lengths)])
                ds = min(0.5 * cell, 0.5 * top.radius)
                n = max(1, math.ceil(total / ds))
                dt = total / top.speed / n
                for s in range(n):
                    arc = (s + (np.arange(substeps) + 0.5) / substeps) * total / n
                    centres = []
                    for u in arc:
                        k = min(int(np.searchsorted(cum, u, side="right")) - 1, len(lengths) - 1)
                        f = (u - cum[k]) / max(lengths[k], 1e-300)
                        c = pts[k] + f * seg[k]
                        d = seg[k] / max(lengths[k], 1e-300)
                        centres.append((c[0], c[1], d[0], d[1]))
                    q_top = self._flux(top, centres) if top.absorbed > 0 else None
                    q_bot = self._flux(low, centres) if low is not None else None
                    advance(dt, q_top, q_bot)
            self._cool(advance, float(pause or 0.0), dt_max)
        if cooling is not None:
            span = 0.0
            while t.max() > cooling and span < 600.0:
                dt = min(dt_max, max(1e-3, 0.05 * max(span, 0.02)))
                advance(dt, None, None)
                span += dt
        if model is not None:
            model.finish(state)
        return TransientResult(t, peak, time, steps, state)

    @staticmethod
    def _cool(advance, pause: float, dt_max: float) -> None:
        """Steps with the laser off for ``pause`` seconds, growing geometrically."""
        span, dt = 0.0, min(1e-3, pause) if pause > 0 else 0.0
        while span < pause - 1e-12:
            dt = min(dt, pause - span, dt_max)
            advance(dt, None, None)
            span += dt
            dt *= 1.5
