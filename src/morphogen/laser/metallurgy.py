"""Thermo-metallurgical response of a steel sheet to the thermal cycles of laser tracks:
the phases and the hardness that each point keeps.

Two models share one interface: :meth:`initial_state` (arrays of the points),
:meth:`step` (one time step of their temperatures, vectorized), :meth:`finish` (the
transformations completed on the final cooling to ambient) and :meth:`hardness`.
Temperatures in °C, times in s, compositions in wt%.

:class:`TransformationHardening` (C–Mn and low-alloy steels). A cycle that exceeds
``Ac1`` forms austenite at its peak ``T_p`` by the lever rule of a hypo-eutectoid steel,
the carbon of the austenite following the ``A3`` line from the eutectoid ``0.77`` at
``Ac1`` to the steel's at ``Ac3``: ``f_γ = (C − C_α)/(C_γ(T_p) − C_α)`` (the pearlite at
once, the ferrite gradually); laser heating at 10³–10⁴ K/s shifts both temperatures up
by ``superheat``. On cooling the austenite turns into martensite when it cools fast
enough (the cooling time ``t8/5`` against a critical one, the rest into bainite or
ferrite–pearlite), at ambient by Koistinen and Marburger's law
``1 − exp(−0.011 (Ms − T))``, ``Ms`` of the austenite's carbon by Andrews's (1965)
linear formula (the rest retained). The hardnesses of martensite, bainite and
ferrite–pearlite are Maynier et al.'s (1978) functions of the composition and of the
cooling rate. Martensite below ``Ac1`` tempers: the Hollomon–Jaffe parameter
``P = T (C + log10 t)`` (K, hours) accumulated by additivity (the time that gives the
current ``P`` at the current temperature, extended by the step), its hardness falling
linearly from ``P0`` to ``P1`` (Grange et al. 1977). An annealed ferrite–pearlite sheet
thus gets tracks of martensite with tempered rings where later tracks pass (the
annealed MS1500 of Wang 2022); a martensitic or dual-phase sheet (DP1180, MS1500:
``martensite`` initial fraction) softens in rings that stay below ``Ac1`` and hardens
again where it is austenitized (Lapouge et al. 2019).

:class:`PeakHardening` (phenomenological). The hardness gained by a cycle is a
function of its peak temperature (a table), lost by Hollomon–Jaffe tempering; a new
cycle above the onset of the table re-transforms a share of the point that grows with
its peak, the rest keeping its tempered gain (the larger of the fresh gain and of this
mixture). For AISI 430, whose hardening by
grain-boundary martensite and chromium carbides near the melting point (Wang et al.
2022) has no simple phase model.

**Composition** (:class:`Alloying`). Laser carburizing from a graphite coating,
nitriding under nitrogen and alloying from a pre-placed coating happen in the melt
pool, stirred by Marangoni flows: the coating that a track melts (its areal mass on the
faces melted, times an efficiency) mixes with the melted metal (mass balance),
nitrogen dissolves up to Sieverts's solubility ``log[%N] = −188/T − 1.25 −
e_N^Cr [%Cr]`` (iron, ``e_N^Cr = −0.047``, at a partial pressure ``p``: times ``√p``)
times an efficiency. The solid-state diffusion of the same cycles reaches micrometres
(``√(∫D dt)``, :meth:`~morphogen.laser.heat.TrackKernel.kinetic_integral`) and is neglected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Elements of the compositions (wt%).
ELEMENTS = ("C", "N", "Si", "Mn", "Ni", "Cr", "Mo", "V")
#: Carbon of ferrite and of the eutectoid (wt%).
C_FERRITE, C_EUTECTOID = 0.02, 0.77


def _composition(data) -> dict[str, float]:
    comp = dict.fromkeys(ELEMENTS, 0.0)
    for k, v in dict(data or {}).items():
        if k not in ELEMENTS:
            raise ValueError(f"composition: unknown element {k!r} (one of {ELEMENTS})")
        comp[k] = float(v)
    return comp


def andrews_ac(comp) -> tuple[float, float]:
    """``Ac1`` and ``Ac3`` (°C) by Andrews's (1965) formulas."""
    c = _composition(comp)
    ac1 = 723.0 - 10.7 * c["Mn"] - 16.9 * c["Ni"] + 29.1 * c["Si"] + 16.9 * c["Cr"]
    ac3 = (
        910.0
        - 203.0 * math.sqrt(max(c["C"], 0.0))
        - 15.2 * c["Ni"]
        + 44.7 * c["Si"]
        + 104.0 * c["V"]
        + 31.5 * c["Mo"]
    )
    return ac1, ac3


def andrews_ms(carbon, comp, nitrogen=0.0):
    """``Ms`` (°C) by Andrews's (1965) linear formula, nitrogen counted as carbon."""
    c = _composition(comp)
    cn = np.asarray(carbon, dtype=float) + np.asarray(nitrogen, dtype=float)
    return 539.0 - 423.0 * cn - 30.4 * c["Mn"] - 17.7 * c["Ni"] - 12.1 * c["Cr"] - 7.5 * c["Mo"]


def maynier_hardness(carbon, comp, rate, *, nitrogen=0.0) -> tuple:
    """Hardnesses (HV) of martensite, bainite and ferrite–pearlite formed at the cooling
    rate ``rate`` (K/s at about 700 °C) by Maynier, Jungmann & Dollet (1978), the carbon
    of the transforming austenite given (nitrogen counted as carbon)."""
    c = _composition(comp)
    cc = np.asarray(carbon, dtype=float) + np.asarray(nitrogen, dtype=float)
    lv = np.log10(np.maximum(np.asarray(rate, dtype=float), 1e-3) * 3600.0)  # °C/h
    si, mn, ni, cr, mo, v = (c[k] for k in ("Si", "Mn", "Ni", "Cr", "Mo", "V"))
    hm = 127.0 + 949.0 * cc + 27.0 * si + 11.0 * mn + 8.0 * ni + 16.0 * cr + 21.0 * lv
    hb = (
        -323.0
        + 185.0 * cc
        + 330.0 * si
        + 153.0 * mn
        + 65.0 * ni
        + 144.0 * cr
        + 191.0 * mo
        + lv * (89.0 + 53.0 * cc - 55.0 * si - 22.0 * mn - 10.0 * ni - 20.0 * cr - 33.0 * mo)
    )
    hf = (
        42.0
        + 223.0 * cc
        + 53.0 * si
        + 30.0 * mn
        + 12.6 * ni
        + 7.0 * cr
        + 19.0 * mo
        + lv * (10.0 - 19.0 * si + 4.0 * ni + 8.0 * cr + 130.0 * v)
    )
    return hm, hb, hf


def sieverts_nitrogen(temperature: float, chromium: float = 0.0, pressure: float = 1.0) -> float:
    """Solubility of nitrogen (wt%) in liquid iron at ``temperature`` (°C) under a
    partial pressure ``pressure`` (atm): ``log[%N] = −188/T − 1.25`` (Sieverts's law),
    raised by chromium through ``log f_N = e_N^Cr [%Cr]``, ``e_N^Cr = −0.047``."""
    t = temperature + 273.15
    return 10.0 ** (-188.0 / t - 1.25 + 0.047 * chromium) * math.sqrt(max(pressure, 0.0))


def _tempering_step(p, t_new, dt, c_hj: float, t_min: float):
    """Hollomon–Jaffe parameter after ``dt`` seconds at ``t_new`` (°C), by additivity:
    the time that gives ``p`` at this temperature, plus ``dt``."""
    hot = t_new > t_min
    if not hot.any():
        return p
    tk = t_new[hot] + 273.15
    t_eq = 10.0 ** (np.minimum(p[hot] / tk - c_hj, 30.0))  # hours
    out = p.copy()
    out[hot] = np.maximum(p[hot], tk * (c_hj + np.log10(t_eq + dt / 3600.0)))
    return out


@dataclass
class Tempering:
    """Linear softening with the Hollomon–Jaffe parameter: hardness ``H0`` up to ``P0``,
    falling to ``floor`` (a fraction of ``H0``, or an absolute ``minimum`` HV, the larger)
    at ``P1``; ``constant`` C (hours); no tempering below ``t_min`` °C."""

    constant: float = 20.0
    start: float = 9000.0
    end: float = 20000.0
    floor: float = 0.4
    minimum: float = 0.0
    t_min: float = 150.0

    def factor(self, p, h0):
        """Hardness of a phase of as-quenched hardness ``h0`` at the parameter ``p``."""
        s = np.clip((np.asarray(p) - self.start) / (self.end - self.start), 0.0, 1.0)
        low = np.maximum(self.floor * np.asarray(h0), self.minimum)
        return h0 - s * np.maximum(h0 - low, 0.0)

    @classmethod
    def from_config(cls, data) -> Tempering:
        return data if isinstance(data, Tempering) else cls(**dict(data or {}))


class _Model:
    """Shared bookkeeping of the thermal excursions of the points."""

    ambient: float = 20.0
    #: An excursion starts above this temperature and ends below ``finish_below``.
    onset: float = 700.0
    finish_below: float = 300.0

    def _base_state(self, n: int) -> dict:
        return {
            "clock": np.zeros(1),
            "tmax": np.full(n, -273.15),
            "peak": np.full(n, -273.15),
            "excursion": np.zeros(n, dtype=bool),
            "t8": np.full(n, np.nan),
            "t85": np.full(n, np.nan),
            "melted": np.zeros(n, dtype=bool),
            "molten": np.zeros(n, dtype=bool),
        }

    def _track(self, state, t_old, t_new, dt):
        """Peaks, excursions and cooling times; returns the points whose excursion ended."""
        state["clock"] += dt
        clock = float(state["clock"][0])
        np.maximum(state["tmax"], t_new, out=state["tmax"])
        start = (~state["excursion"]) & (t_new >= self.onset)
        if start.any():
            state["excursion"][start] = True
            state["peak"][start] = t_new[start]
            state["t8"][start] = np.nan
            state["molten"][start] = False
        exc = state["excursion"]
        np.maximum(state["peak"], np.where(exc, t_new, -273.15), out=state["peak"])
        # cooling times 800 -> 500 °C (linear interpolation of the crossing within the step)
        for level, key in ((800.0, "t8"), (500.0, "t5")):
            cross = exc & (t_old >= level) & (t_new < level)
            if cross.any():
                f = (t_old[cross] - level) / np.maximum(t_old[cross] - t_new[cross], 1e-12)
                when = clock - dt + f * dt
                if key == "t8":
                    state["t8"][cross] = when
                else:
                    t8 = state["t8"][cross]
                    state["t85"][cross] = np.where(np.isfinite(t8), when - t8, state["t85"][cross])
        hot = t_new >= self.solidus
        state["melted"] |= hot
        state["molten"] |= exc & hot
        done = exc & (t_new < self.finish_below)
        return done

    def finish(self, state) -> None:
        """Complete the excursions still open (the final cooling to ambient)."""
        done = state["excursion"].copy()
        if done.any():
            self._complete(state, done)

    def peak_temperature(self, state) -> np.ndarray:
        return state["tmax"]


@dataclass
class TransformationHardening(_Model):
    """C–Mn or low-alloy steel (see the module): ``composition`` (wt%), ``Ac1``/``Ac3``
    (°C; Andrews's by default) plus ``superheat`` for laser heating rates, ``solidus``,
    the initial ``martensite`` fraction and the hardness of the initial microstructure's
    martensite (``martensite_hardness``; Maynier's at ``initial_rate`` by default) and of
    the rest (``matrix_hardness``; Maynier's ferrite–pearlite after a slow cooling by
    default), the critical cooling time ``t8/5`` up to which austenite turns into
    martensite (``critical_time``, s) and the time beyond which none does
    (``slow_time``), ``tempering``."""

    composition: dict = field(default_factory=lambda: {"C": 0.2, "Mn": 1.5, "Si": 0.25})
    ac1: float | None = None
    ac3: float | None = None
    superheat: float = 50.0
    solidus: float = 1480.0
    martensite: float = 0.0
    martensite_hardness: float | None = None
    matrix_hardness: float | None = None
    initial_rate: float = 1.0 / 30.0
    critical_time: float = 2.0
    slow_time: float = 30.0
    km: float = 0.011
    tempering: Tempering = field(default_factory=Tempering)
    ambient: float = 20.0
    finish_below: float = 300.0

    def __post_init__(self):
        self.composition = _composition(self.composition)
        self.tempering = Tempering.from_config(self.tempering)
        ac1, ac3 = andrews_ac(self.composition)
        self.ac1 = float(self.ac1 if self.ac1 is not None else ac1)
        self.ac3 = float(self.ac3 if self.ac3 is not None else ac3)
        if not self.ac1 < self.ac3:
            raise ValueError("Ac1 < Ac3")
        if not 0.0 <= self.martensite <= 1.0:
            raise ValueError("martensite: an initial fraction in [0, 1]")
        c = self.composition["C"]
        hm, _, _hf = maynier_hardness(c, self.composition, 1000.0)
        if self.martensite_hardness is None:
            self.martensite_hardness = float(hm)
        if self.matrix_hardness is None:
            self.matrix_hardness = float(
                maynier_hardness(c, self.composition, self.initial_rate)[2]
            )
        self.onset = self.ac1 + self.superheat

    @property
    def a1(self) -> float:
        return self.ac1 + self.superheat

    @property
    def a3(self) -> float:
        return self.ac3 + self.superheat

    def austenite(self, peak, carbon) -> tuple[np.ndarray, np.ndarray]:
        """Fraction and carbon (wt%) of the austenite at the peak temperature ``peak``
        of a steel of carbon ``carbon``: lever rule, the austenite carbon from 0.77 at
        ``A1`` to the steel's at ``A3`` (linear in temperature)."""
        peak = np.asarray(peak, dtype=float)
        c = np.clip(np.asarray(carbon, dtype=float), C_FERRITE + 1e-6, None)
        s = np.clip((peak - self.a1) / (self.a3 - self.a1), 0.0, 1.0)
        c_eut = np.maximum(C_EUTECTOID, c)
        c_gamma = c_eut + (c - c_eut) * s
        f = np.where(peak >= self.a1, (c - C_FERRITE) / np.maximum(c_gamma - C_FERRITE, 1e-9), 0.0)
        f = np.clip(np.where(peak >= self.a3, 1.0, f), 0.0, 1.0)
        return f, np.where(f > 0, c_gamma, c)

    def initial_state(self, n: int) -> dict:
        state = self._base_state(n)
        state.update(
            carbon=np.full(n, self.composition["C"]),
            nitrogen=np.full(n, self.composition["N"]),
            martensite=np.full(n, self.martensite),
            hm0=np.full(n, float(self.martensite_hardness)),
            ptemper=np.zeros(n),
            bainite=np.zeros(n),
            hb=np.zeros(n),
            retained=np.zeros(n),
        )
        return state

    def step(self, state, t_old, t_new, dt) -> None:
        """One step of the temperatures of the points (°C), ``dt`` seconds."""
        t_old = np.asarray(t_old, dtype=float)
        t_new = np.asarray(t_new, dtype=float)
        has_m = state["martensite"] > 0
        warm = has_m & (t_new > self.tempering.t_min)
        if warm.any():
            idx = np.flatnonzero(warm)
            state["ptemper"][idx] = _tempering_step(
                state["ptemper"][idx], t_new[idx], dt, self.tempering.constant, self.tempering.t_min
            )
        done = self._track(state, t_old, t_new, dt)
        if done.any():
            self._complete(state, done)

    def _complete(self, state, done) -> None:
        """End of the excursions of the points ``done``: transformations at the peak
        and on cooling to ambient."""
        idx = np.flatnonzero(done)
        peak = state["peak"][idx]
        carbon = state["carbon"][idx]
        nitrogen = state["nitrogen"][idx]
        fg, cg = self.austenite(np.where(state["molten"][idx], self.a3 + 1.0, peak), carbon)
        ng = np.where(fg > 0, nitrogen * np.minimum(1.0 / np.maximum(fg, 1e-9), 3.0), nitrogen)
        t85 = state["t85"][idx]
        t85 = np.where(np.isfinite(t85), t85, self.critical_time)  # unseen: as quenched
        rate = 300.0 / np.maximum(t85, 1e-6)
        lt = np.log(np.maximum(t85, 1e-9))
        share = np.clip(
            (math.log(self.slow_time) - lt)
            / (math.log(self.slow_time) - math.log(self.critical_time)),
            0.0,
            1.0,
        )
        ms = andrews_ms(cg, self.composition, ng)
        km = np.where(ms > self.ambient, 1.0 - np.exp(-self.km * (ms - self.ambient)), 0.0)
        hm, hb, hf = maynier_hardness(cg, self.composition, rate, nitrogen=ng)
        # the austenite consumes the martensite first, then the bainite, then the matrix
        m_old = state["martensite"][idx]
        b_old = state["bainite"][idx]
        m_left = np.maximum(m_old - fg, 0.0)
        b_left = np.maximum(b_old - np.maximum(fg - m_old, 0.0), 0.0)
        m_new = fg * share * km
        b_new = fg * (1.0 - share)
        retained = fg * share * (1.0 - km)
        m_tot = m_left + m_new
        w_old = np.divide(m_left, m_tot, out=np.zeros_like(m_tot), where=m_tot > 0)
        state["hm0"][idx] = np.where(
            m_tot > 0, w_old * state["hm0"][idx] + (1 - w_old) * hm, state["hm0"][idx]
        )
        state["ptemper"][idx] = w_old * state["ptemper"][idx]
        state["martensite"][idx] = m_tot
        b_tot = b_left + b_new
        wb = np.divide(b_left, b_tot, out=np.zeros_like(b_tot), where=b_tot > 0)
        state["hb"][idx] = np.where(
            b_tot > 0, wb * state["hb"][idx] + (1 - wb) * np.maximum(hb, hf), state["hb"][idx]
        )
        state["bainite"][idx] = b_tot
        state["retained"][idx] = np.where(fg > 0, retained, state["retained"][idx])
        state["excursion"][idx] = False
        state["peak"][idx] = -273.15
        state["t8"][idx] = np.nan

    def hardness(self, state) -> np.ndarray:
        """Vickers hardness of the points: martensite (tempered), bainite or
        ferrite–pearlite formed on cooling, retained austenite (as the matrix) and the
        initial matrix, by their fractions."""
        fm, fb = state["martensite"], state["bainite"]
        fr = state["retained"]
        hm = self.tempering.factor(state["ptemper"], state["hm0"])
        rest = np.clip(1.0 - fm - fb - fr, 0.0, 1.0)
        return fm * hm + fb * state["hb"] + (rest + fr) * float(self.matrix_hardness)

    def treated(self, state) -> np.ndarray:
        """Fraction of the points transformed by the tracks (martensite and bainite
        formed, minus the initial martensite)."""
        return np.clip(state["martensite"] + state["bainite"] - self.martensite, 0.0, 1.0)


@dataclass
class PeakHardening(_Model):
    """Hardness gained as a function of the peak temperature of a cycle (``table``:
    pairs ``(T_peak °C, ΔHV)``, increasing from 0 at its first temperature), on a matrix
    of hardness ``base_hardness``; ``full``: the peak above which a new cycle replaces
    the gain of the previous ones (re-transformed share ``(T_p − onset)/(full −
    onset)``); ``tempering`` of the gain; ``carbon_hardening``: HV gained per wt% of
    carbon and nitrogen dissolved in the melt (carburizing, nitriding)."""

    table: tuple = ((875.0, 0.0), (1080.0, 50.0), (1300.0, 120.0), (1425.0, 150.0), (1600.0, 150.0))
    base_hardness: float = 160.0
    full: float | None = None
    solidus: float = 1425.0
    composition: dict = field(default_factory=lambda: {"C": 0.026, "Cr": 18.7})
    carbon_hardening: float = 550.0
    tempering: Tempering = field(
        default_factory=lambda: Tempering(start=11000.0, end=19000.0, floor=0.0)
    )
    ambient: float = 20.0
    finish_below: float = 300.0

    def __post_init__(self):
        tab = np.asarray(self.table, dtype=float).reshape(-1, 2)
        if len(tab) < 2 or np.any(np.diff(tab[:, 0]) <= 0):
            raise ValueError("table: at least two rows of increasing temperatures")
        self.table = tuple(map(tuple, tab))
        self._t, self._dh = tab[:, 0], tab[:, 1]
        self.onset = float(self._t[0])
        self.full = float(self.full if self.full is not None else self._t[np.argmax(self._dh)])
        self.composition = _composition(self.composition)
        self.tempering = Tempering.from_config(self.tempering)

    def gain(self, peak) -> np.ndarray:
        """Hardness gained by a cycle of peak ``peak`` (°C) from the untreated state."""
        return np.interp(np.asarray(peak, dtype=float), self._t, self._dh, left=0.0)

    def initial_state(self, n: int) -> dict:
        state = self._base_state(n)
        state.update(
            carbon=np.full(n, self.composition["C"]),
            nitrogen=np.full(n, self.composition["N"]),
            gain=np.zeros(n),
            ptemper=np.zeros(n),
        )
        return state

    def step(self, state, t_old, t_new, dt) -> None:
        t_old = np.asarray(t_old, dtype=float)
        t_new = np.asarray(t_new, dtype=float)
        warm = (state["gain"] > 0) & (t_new > self.tempering.t_min)
        if warm.any():
            idx = np.flatnonzero(warm)
            state["ptemper"][idx] = _tempering_step(
                state["ptemper"][idx], t_new[idx], dt, self.tempering.constant, self.tempering.t_min
            )
        done = self._track(state, t_old, t_new, dt)
        if done.any():
            self._complete(state, done)

    def _complete(self, state, done) -> None:
        idx = np.flatnonzero(done)
        peak = state["peak"][idx]
        share = np.clip((peak - self.onset) / max(self.full - self.onset, 1e-9), 0.0, 1.0)
        fresh = self.gain(peak)
        chem = self.carbon_hardening * np.maximum(
            state["carbon"][idx]
            + state["nitrogen"][idx]
            - self.composition["C"]
            - self.composition["N"],
            0.0,
        )
        old = self.tempering.factor(state["ptemper"][idx], state["gain"][idx])
        new = fresh + chem
        state["gain"][idx] = np.maximum(new, share * new + (1.0 - share) * old)
        state["ptemper"][idx] = 0.0
        state["excursion"][idx] = False
        state["peak"][idx] = -273.15
        state["t8"][idx] = np.nan

    def hardness(self, state) -> np.ndarray:
        return self.base_hardness + self.tempering.factor(state["ptemper"], state["gain"])

    def treated(self, state) -> np.ndarray:
        """Share of the full gain of the table that the points keep."""
        return np.clip(
            (self.hardness(state) - self.base_hardness) / max(self._dh.max(), 1e-9), 0, 1
        )


@dataclass
class PeakSoftening(PeakHardening):
    """Hardness lost as a function of the peak temperature of a cycle — the dissolution
    of the hardening precipitates of an age-hardened aluminium alloy (Myhr & Grong 1991;
    Shercliff & Ashby): ``table`` pairs ``(T_peak °C, ΔHV ≤ 0)``, 0 at its first
    temperature, the hardness of the treated points after their natural ageing; a cycle
    keeps the larger of its loss and that of the previous cycles (the precipitates do
    not form again)."""

    table: tuple = (
        (200.0, 0.0),
        (250.0, -8.0),
        (300.0, -25.0),
        (350.0, -42.0),
        (400.0, -50.0),
        (450.0, -45.0),
        (520.0, -38.0),
        (600.0, -38.0),
    )
    base_hardness: float = 105.0
    solidus: float = 585.0
    composition: dict = field(default_factory=lambda: {"C": 0.0})
    carbon_hardening: float = 0.0
    tempering: Tempering = field(
        default_factory=lambda: Tempering(start=1e9, end=2e9, floor=1.0, minimum=0.0)
    )

    def _complete(self, state, done) -> None:
        idx = np.flatnonzero(done)
        fresh = self.gain(state["peak"][idx])
        state["gain"][idx] = np.minimum(state["gain"][idx], fresh)
        state["ptemper"][idx] = 0.0
        state["excursion"][idx] = False
        state["peak"][idx] = -273.15

    def treated(self, state) -> np.ndarray:
        """Share of the largest loss of the table that the points have."""
        loss = max(-float(np.min(self._dh)), 1e-9)
        return np.clip((self.base_hardness - self.hardness(state)) / loss, 0.0, 1.0)


MODELS = {
    "transformation": TransformationHardening,
    "peak": PeakHardening,
    "softening": PeakSoftening,
}


def model_from_config(data):
    """A metallurgical model from a mapping with a ``kind`` (``transformation``,
    ``peak`` or ``softening``) and the parameters of its class."""
    if isinstance(data, (TransformationHardening, PeakHardening)):
        return data
    data = dict(data)
    kind = data.pop("kind", "transformation")
    if kind not in MODELS:
        raise ValueError(f"metallurgy kind: one of {tuple(MODELS)}")
    if "tempering" in data:
        data["tempering"] = Tempering.from_config(data["tempering"])
    if "table" in data:
        data["table"] = tuple(tuple(row) for row in data["table"])
    return MODELS[kind](**data)


# -- composition changes in the melt pool --------------------------------------------------

TREATMENTS = ("hardening", "carburizing", "nitriding", "alloying")


@dataclass
class Alloying:
    """A thermochemical treatment in the melt pool of each track: ``carburizing`` (a
    graphite coating of ``coating`` g/m² on the treated faces), ``alloying`` (a coating
    of composition ``elements``, wt%), ``nitriding`` (nitrogen at the partial
    pressure ``pressure`` atm); ``efficiency``: the share of the coating (or of
    Sieverts's solubility) that the melt takes up, the rest burnt or evaporated."""

    kind: str = "carburizing"
    coating: float = 20.0
    elements: dict = field(default_factory=lambda: {"C": 100.0})
    efficiency: float = 0.5
    pressure: float = 1.0
    faces: str = "both"

    def __post_init__(self):
        if self.kind not in TREATMENTS[1:]:
            raise ValueError(f"alloying kind: one of {TREATMENTS[1:]}")
        if self.kind == "carburizing":
            self.elements = {"C": 100.0}
        if self.faces not in ("top", "bottom", "both"):
            raise ValueError("faces: top, bottom or both")
        if not 0 <= self.efficiency <= 1:
            raise ValueError("efficiency in [0, 1]")
        for k in self.elements:
            if k not in ELEMENTS:
                raise ValueError(f"elements: one of {ELEMENTS}")

    @classmethod
    def from_config(cls, data):
        if data is None or isinstance(data, Alloying):
            return data
        return cls(**dict(data))

    def mix(
        self,
        state,
        melted,
        weights,
        surface,
        *,
        density: float,
        melt_temperature: float,
        chromium: float = 0.0,
    ) -> dict:
        """Mix the melt of one track: ``melted`` (mask of the points), ``weights`` (their
        volumes, mm³), ``surface``: per face, the areas (mm²) of the points of that face
        whose coating the track melts (the coating is then gone: the caller clears it).
        Updates the carbon and nitrogen of the state; returns the amounts added."""
        out = {}
        if not melted.any():
            return out
        vol = weights[melted]
        mass = density * 1e-9 * vol.sum() * 1e3  # g (density kg/m3 -> g/mm3)
        for key, el in (("carbon", "C"), ("nitrogen", "N")):
            if key not in state:
                continue
            mean = float(np.average(state[key][melted], weights=vol))
            added = 0.0
            if self.kind in ("carburizing", "alloying") and el in self.elements:
                area = sum(surface.values())
                added = self.efficiency * self.coating * 1e-6 * area * self.elements[el] / 100.0
            new = 100.0 * (mean * mass / 100.0 + added) / (mass + added) if mass > 0 else mean
            if self.kind == "nitriding" and el == "N":
                sol = sieverts_nitrogen(melt_temperature, chromium, self.pressure)
                new = max(mean, self.efficiency * sol)
            state[key][melted] = new
            out[el] = new
        return out
