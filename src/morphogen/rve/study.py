"""Statistical RVE analysis of apparent properties (Dirrenberger et al. 2014, sec. 4-5).

A *series* gathers the realizations of one property under one boundary condition, at
several volumes. For every series the analysis gives:

* the number of realizations, the mean and the standard deviation ``D_Z(V)`` per volume,
  with the intervals ``[Zbar - 2 D_Z, Zbar + 2 D_Z]`` (Tables 3-4);
* the relative error of the sample, ``eps_rel = 2 D_Z(V) / (Zbar sqrt(n))`` (Table 8);
* the fit of ``D_Z^2(V) = K V^-gamma`` with gamma free (Table 9), and with gamma imposed
  when requested, e.g. the theoretical 2/3 of Poisson fibres (Table 10);
* the RVE volume ``V_RVE = (4 K / (eps^2 Zbar^2 n))^(1/gamma)`` (eq. 87) for each pair
  (eps_rel, n), its edge, the mean number of objects it contains (fibres, grains...) when
  the model provides it, and bootstrap confidence intervals;
* the number of realizations of each sampled volume needed for each precision;
* the statistical isotropy criterion of the mean response per volume (eqs. 72-74).

Across boundary conditions it gives the discrepancy ``D(Z) = (Zmax - Zmin) / Zmin``
between their mean values (eq. 69) and, when the phases are known, the bounds.
``Zbar`` in the RVE formula is the mean at the largest volume, as in the paper.

Stiffness and compliance (:class:`TensorStatistics`). When the realizations have their
apparent stiffness ``C`` (the columns ``stiffness.<bc>.C11`` ... ``C66`` of the results),
each boundary condition gets, per volume, the mean stiffness ``Cbar`` over the
realizations and the standard deviation of each of its components, and the compliance
``S = Cbar^-1``: the inverse of the mean stiffness, so that ``S Cbar = I``, not the mean
of the compliances of the realizations ``<C^-1>`` (kept as ``mean_compliance``; larger,
``<C^-1> >= <C>^-1`` by Jensen's inequality, by a term of the order of the relative
variance; under SUBC its inverse is Huet's lower estimate), with the standard deviation
of each component of the compliances ``C^-1`` of the realizations. Each component of
``C`` and of the compliances is also analysed as a series (variance fit, RVE sizes, no
bootstrap), the RVE sizes kept for the diagonal components and those whose mean the
realizations of the largest volume distinguish from zero (Student's t test at the level of
the study): the couplings that vanish on average have no relative precision. Notation:
morphoGen's Voigt order 11, 22, 33, 23, 31, 12 with engineering shears
(:data:`morphogen.homogenization.postprocess.NOTATION`). The conductivity tensor (the
columns ``conductivity_tensor.<bc>.C11`` ... ``C33``) has the same statistics, its inverse
the resistivity (:data:`CONDUCTIVITY_NOTATION`), and so has the tensor of every other
transport phenomenon (:data:`TRANSPORT_TENSORS`: ``electrical_conductivity_tensor``,
``permittivity_tensor``, ``permeability_tensor``, ``diffusivity_tensor``). By the window method
(:mod:`morphogen.rve.windows`) the windows of one realization are the realizations; with
tilings shifted around the periodic realization, those of all the tilings, their means
over all, their variances pooled within the tilings and their counts those of one tiling
(:mod:`morphogen.rve.scaling`). The squares of a planar image have areas for volumes, and
for edges their roots (``dimension`` 2).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from morphogen.homogenization.materials import TRANSPORT, TRANSPORT_LAWS
from morphogen.rve.scaling import (
    ScalingLaw,
    SizeStatistics,
    _moments,
    _pooled_variance,
    ensemble_statistics,
    fit_scaling_law,
    point_variance,
)

META_COLUMNS = {
    "size_index",
    "size",
    "volume",
    "realization",
    "seed",
    "time_s",
    "n_spheres",
    "n_segments",
    "n_cells",
    "n_modes",
}


def _is_meta(column: str) -> bool:
    """Columns not analysed by default: identifiers, counts, timings, mesh data, phase
    fractions of the homogenized samples and tensor components (``<p>.<bc>.<entry>``)."""
    return (
        column in META_COLUMNS or column.startswith(("mesh_", "fraction_")) or column.count(".") > 1
    )


def _isotropy(prop: str, response: np.ndarray) -> float:
    from morphogen.homogenization.apparent import bulk_isotropy, shear_isotropy, thermal_isotropy

    if prop.startswith("bulk"):
        return bulk_isotropy(response)
    if prop.startswith("shear"):
        return shear_isotropy(response)
    if prop.partition(".")[0] in TRANSPORT.values():  # a flux under a gradient along x
        return thermal_isotropy(response, 0)
    return math.nan


@dataclass
class Series:
    """Realizations of a property (under one boundary condition) at several volumes.

    ``response`` optionally holds the macroscopic response of every realization: the
    3 x 3 average stress (or strain under SUBC), or the average flux vector; it is used
    for the isotropy criteria.
    """

    property: str
    bc: str
    volume: np.ndarray
    value: np.ndarray
    response: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.volume = np.asarray(self.volume, dtype=float)
        self.value = np.asarray(self.value, dtype=float)
        if self.volume.shape != self.value.shape or self.volume.ndim != 1:
            raise ValueError("a series needs one volume per value")
        ok = np.isfinite(self.value) & np.isfinite(self.volume) & (self.volume > 0)
        if self.response is not None:
            self.response = np.asarray(self.response, dtype=float)
            if len(self.response) != len(self.value):
                raise ValueError("a series needs one response per value")
            self.response = self.response[ok]
        self.volume, self.value = self.volume[ok], self.value[ok]

    @property
    def key(self) -> str:
        return f"{self.property}.{self.bc}" if self.bc else self.property

    def statistics(self) -> SizeStatistics:
        return ensemble_statistics(self.volume, self.value)

    def mean_response(self, volume: float) -> np.ndarray | None:
        if self.response is None:
            return None
        sel = self.volume == volume
        return self.response[sel].mean(axis=0) if np.any(sel) else None


def _parse(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _volume(row: dict, volume_column: str) -> float:
    if volume_column in row and row[volume_column] not in ("", None):
        return _parse(row[volume_column])
    if "size" in row and row["size"] not in ("", None):
        return _parse(row["size"]) ** 3
    raise ValueError(f"results need a '{volume_column}' (or 'size') column")


def _responses(rows: list[dict]) -> np.ndarray | None:
    tensor = [f"R{n}" for n in ("11", "22", "33", "23", "31", "12")]
    vector = ["R1", "R2", "R3"]

    def present(keys):
        return all(k in rows[0] and math.isfinite(_parse(rows[0][k])) for k in keys)

    if present(tensor):
        out = np.empty((len(rows), 3, 3))
        pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
        for i, row in enumerate(rows):
            for name, (a, b) in zip(tensor, pairs, strict=True):
                out[i, a, b] = out[i, b, a] = _parse(row[name])
        return out
    if present(vector):
        return np.array([[_parse(row[k]) for k in vector] for row in rows])
    return None


def series_from_rows(
    rows: list[dict], *, properties=None, group_by: str | None = None, volume_column="volume"
) -> list[Series]:
    """Series of a results table.

    * Long format (columns ``property`` and ``value``, e.g. ``homogenization.csv``):
      one series per (property, ``bc`` or ``group_by`` column). Rows without a value
      (the summaries of the loading paths) are left out.
    * Wide format: one column per property. Columns named ``<property>.<bc>`` (e.g.
      ``bulk_modulus.kubc``) give one series per boundary condition; with ``group_by``
      the rows are split by the values of that column instead.
    """
    if not rows:
        raise ValueError("no results to analyse")
    wanted = None if properties is None else list(properties)
    series: list[Series] = []
    if any("property" in r and "value" in r for r in rows):
        rows = [r for r in rows if r.get("value") not in (None, "")]
        if not rows:
            raise ValueError("no results to analyse")
        group = group_by or ("bc" if "bc" in rows[0] else None)
        keys: dict[tuple, list[dict]] = {}
        for row in rows:
            g = str(row.get(group, "")) if group else ""
            keys.setdefault((str(row["property"]), g), []).append(row)
        for (prop, g), sub in keys.items():
            name = f"{prop}.{g}" if g else prop
            if wanted is not None and prop not in wanted and name not in wanted:
                continue
            series.append(
                Series(
                    prop,
                    g,
                    [_volume(r, volume_column) for r in sub],
                    [_parse(r["value"]) for r in sub],
                    _responses(sub),
                )
            )
        if not series:
            raise ValueError(f"no rows for the properties {wanted}")
        return series
    columns: list[str] = []
    numeric: set[str] = set()
    for row in rows:
        for c, v in row.items():
            if c not in columns:
                columns.append(c)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric.add(c)
    if wanted is None:
        wanted = [
            c
            for c in columns
            if not _is_meta(c) and c != volume_column and c != group_by and c in numeric
        ]
    missing = [p for p in wanted if p not in columns]
    if missing:
        raise ValueError(f"unknown result columns {missing}; available: {columns}")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_by, "")) if group_by else "", []).append(row)
    for g, sub in groups.items():
        volumes = [_volume(r, volume_column) for r in sub]
        for col in wanted:
            if group_by:
                prop, bc = col, g
            else:
                prop, _, bc = col.partition(".")
            series.append(Series(prop, bc, volumes, [_parse(r.get(col)) for r in sub]))
    return series


@dataclass
class SeriesAnalysis:
    """Results of the analysis of one series (see the module documentation)."""

    series: Series
    stats: SizeStatistics
    reference_mean: float
    free_fit: ScalingLaw | None
    fit: ScalingLaw | None
    rve: list[dict]
    needed: list[dict]
    isotropy: np.ndarray | None
    bootstrap: dict | None = None
    point_variance: float | None = None
    integral_range: float | None = None
    theory: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.series.key

    def statistics_rows(self) -> list[dict]:
        """The statistics of each volume (:meth:`SizeStatistics.rows`), with its edge (of a
        cube, or of the square of a planar image), the standard deviation and the interval
        ``Zbar +/- 2 D``."""
        rows = self.stats.rows()
        for i, row in enumerate(rows):
            std = math.sqrt(row["variance"]) if math.isfinite(row["variance"]) else math.nan
            row["edge"] = row["volume"] ** (1.0 / self.stats.dimension)
            row["std"] = std
            row["low"] = row["mean"] - 2.0 * std
            row["high"] = row["mean"] + 2.0 * std
            if self.isotropy is not None:
                row["isotropy"] = float(self.isotropy[i])
        return rows

    def to_dict(self) -> dict:
        def law(fit):
            if fit is None:
                return None
            d = fit.to_dict()
            lo, hi = fit.gamma_interval()
            klo, khi = fit.k_interval()
            d.update({"gamma_ci": [lo, hi], "K_ci": [klo, khi]})
            return {
                k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in d.items()
            }

        out = {
            "property": self.series.property,
            "bc": self.series.bc,
            "statistics": self.statistics_rows(),
            "mean": self.reference_mean,
            "free_fit": law(self.free_fit),
            "fit": law(self.fit),
            "rve": self.rve,
            "realizations_needed": self.needed,
        }
        if self.bootstrap is not None:
            out["bootstrap"] = self.bootstrap
        if self.point_variance is not None:
            out["point_variance"] = self.point_variance
            out["integral_range"] = self.integral_range
        if self.theory:
            out["theory"] = self.theory
        if self.stats.dimension != 3:  # the squares of a planar image: edges A^(1/2)
            out["dimension"] = self.stats.dimension
        return out


def _fit(stats, gamma, volume_range, weights) -> ScalingLaw | None:
    ok = np.isfinite(stats.variance) & (stats.variance > 0)
    if volume_range is not None:
        lo, hi = volume_range
        ok &= stats.volume >= (lo if lo is not None else -np.inf)
        ok &= stats.volume <= (hi if hi is not None else np.inf)
    need = 1 if gamma is not None else 2
    if np.count_nonzero(ok) < need or (gamma is None and np.unique(stats.volume[ok]).size < 2):
        return None
    w = None
    if weights == "n":
        w = np.maximum(stats.n - 1, 1) / 2.0
    return fit_scaling_law(
        stats.volume, stats.variance, gamma=gamma, volume_range=volume_range, weights=w
    )


def _rve_volume(fit, eps, mean, n) -> float:
    """V_RVE, or NaN when the variance does not decrease with the volume (gamma <= 0)."""
    if not fit.gamma > 0:
        return math.nan
    with np.errstate(over="ignore"):
        return float(fit.rve_volume(eps, mean, n))


def _rve_rows(fit, mean, eps, n, count, dimension: int = 3) -> list[dict]:
    """The RVE volume and its edge (of a cube, or of a square in ``dimension`` 2) for each
    pair (eps, n), with the count of objects that it holds."""
    rows = []
    for e in eps:
        for k in n:
            vol = _rve_volume(fit, e, mean, k)
            edge = vol ** (1.0 / dimension)
            row = {"eps_rel": float(e), "n": int(k), "volume": vol, "edge": edge}
            if count is not None and math.isfinite(vol):
                row["count"] = float(count(vol))
            rows.append(row)
    return rows


def _bootstrap(series, stats, options, gamma, volume_range, weights) -> dict | None:
    """Percentile intervals of gamma, K and V_RVE from resampled realizations."""
    n_boot = int(options.bootstrap)
    if n_boot <= 0:
        return None
    rng = np.random.default_rng(options.seed)
    groups = [series.value[series.volume == v] for v in stats.volume]
    gammas, ks, volumes = [], [], []
    largest = int(np.argmax(stats.volume))
    for _ in range(n_boot):
        means, variances = [], []
        for vals in groups:
            sample = vals[rng.integers(0, len(vals), len(vals))]
            means.append(sample.mean())
            variances.append(sample.var(ddof=1) if len(sample) > 1 else np.nan)
        boot = SizeStatistics(stats.volume, stats.n, np.array(means), np.array(variances))
        fit = _fit(boot, gamma, volume_range, weights)
        if fit is None or means[largest] == 0 or not fit.gamma > 0:
            continue
        gammas.append(fit.gamma)
        ks.append(fit.K)
        volumes.append(
            [
                _rve_volume(fit, e, means[largest], k)
                for e in options.relative_errors
                for k in options.n_realizations
            ]
        )
    if len(gammas) < max(10, n_boot // 10):
        return None
    q = [50 * (1 - options.level), 50 * (1 + options.level)]
    vol = np.array(volumes, dtype=float)
    vol[~np.isfinite(vol)] = np.inf
    return {
        "replicates": len(gammas),
        "level": options.level,
        "gamma": [float(x) for x in np.percentile(gammas, q)],
        "K": [float(x) for x in np.percentile(ks, q)],
        "volume": [
            [float(a), float(b)] for a, b in np.percentile(vol, q, axis=0, method="inverted_cdf").T
        ],
    }


@dataclass
class StudyOptions:
    """Settings of the analysis.

    ``gamma``: exponent imposed for the RVE sizes (None: the fitted one). ``min_volume``
    and ``max_volume`` restrict the fits (the paper keeps V >= 40^3 to avoid boundary
    layer effects). ``weights="n"`` weights each size by (n - 1) / 2, the inverse
    variance of ln D^2 for Gaussian data. ``bootstrap`` replicates (0: none).
    """

    gamma: float | None = None
    min_volume: float | None = None
    max_volume: float | None = None
    relative_errors: tuple = (0.01, 0.05)
    n_realizations: tuple = (1, 10)
    bootstrap: int = 1000
    seed: int = 0
    weights: str = "none"
    level: float = 0.95

    def __post_init__(self) -> None:
        if self.weights not in ("none", "n"):
            raise ValueError("weights: 'none' or 'n'")
        self.relative_errors = tuple(float(e) for e in self.relative_errors)
        self.n_realizations = tuple(int(k) for k in self.n_realizations)
        if any(e <= 0 for e in self.relative_errors) or any(k < 1 for k in self.n_realizations):
            raise ValueError("relative errors must be positive and n at least 1")


def analyze_series(
    series: Series,
    options: StudyOptions | None = None,
    *,
    count: Callable[[float], float] | None = None,
    theory: dict | None = None,
) -> SeriesAnalysis:
    """Analysis of one series (statistics, fits, RVE sizes, isotropy, bootstrap)."""
    return _analysis(series, series.statistics(), options, count, theory, resample=True)


def analyze_statistics(
    stats: SizeStatistics,
    prop: str = "volume_fraction",
    options: StudyOptions | None = None,
    *,
    theory: dict | None = None,
) -> SeriesAnalysis:
    """Analysis of precomputed statistics (e.g. the window method; no bootstrap)."""
    series = Series(prop, "", [], [])
    return _analysis(series, stats, options, None, theory, resample=False)


def _analysis(series, stats, options, count, theory, resample) -> SeriesAnalysis:
    options = options or StudyOptions()
    if stats.volume.size == 0:
        raise ValueError(f"series {series.key}: no finite values")
    reference = float(stats.mean[np.argmax(stats.volume)])
    volume_range = (options.min_volume, options.max_volume)
    free = _fit(stats, None, volume_range, options.weights)
    fit = free
    if options.gamma is not None:
        fit = _fit(stats, options.gamma, volume_range, options.weights)
    rve_rows, needed = [], []
    if fit is not None and reference != 0 and fit.gamma > 0:
        rve_rows = _rve_rows(fit, reference, options.relative_errors, options.n_realizations,
                             count, stats.dimension)  # fmt: skip
        for v in stats.volume:
            needed.append(
                {
                    "volume": float(v),
                    **{
                        f"eps={e:g}": float(math.ceil(fit.realizations(v, e, reference)))
                        for e in options.relative_errors
                    },
                }
            )
    boot = None
    if fit is not None and resample:
        boot = _bootstrap(series, stats, options, options.gamma, volume_range, options.weights)
        if boot is not None and len(rve_rows) == len(boot["volume"]):
            for row, (lo, hi) in zip(rve_rows, boot["volume"], strict=True):
                row["volume_ci"] = [lo, hi]
                row["edge_ci"] = [lo ** (1 / stats.dimension), hi ** (1 / stats.dimension)]
    iso = None
    if series.response is not None and series.value.size:
        iso = np.array([_isotropy(series.property, series.mean_response(v)) for v in stats.volume])
    pv = ir = None
    if series.property == "volume_fraction" and fit is not None and 0 < reference < 1:
        pv = point_variance(reference)
        ir = fit.integral_range(pv)
    return SeriesAnalysis(
        series=series,
        stats=stats,
        reference_mean=reference,
        free_fit=free,
        fit=fit,
        rve=rve_rows,
        needed=needed,
        isotropy=iso,
        bootstrap=boot,
        point_variance=pv,
        integral_range=ir,
        theory={k: v for k, v in (theory or {}).items() if isinstance(v, float)},
    )


def discrepancy(analyses: list[SeriesAnalysis]) -> dict[str, list[dict]]:
    """D(Z) = (Zmax - Zmin) / Zmin between boundary conditions, per volume (eq. 69)."""
    by_prop: dict[str, list[SeriesAnalysis]] = {}
    for a in analyses:
        if a.series.bc:
            by_prop.setdefault(a.series.property, []).append(a)
    out = {}
    for prop, group in by_prop.items():
        if len(group) < 2:
            continue
        common = set(group[0].stats.volume.tolist())
        for a in group[1:]:
            common &= set(a.stats.volume.tolist())
        rows = []
        dimension = group[0].stats.dimension
        for v in sorted(common):
            means = {a.series.bc: float(a.stats.mean[a.stats.volume == v][0]) for a in group}
            lo, hi = min(means.values()), max(means.values())
            rows.append(
                {
                    "volume": v,
                    "edge": v ** (1.0 / dimension),
                    "means": means,
                    "discrepancy": (hi - lo) / lo if lo != 0 else math.nan,
                }
            )
        out[prop] = rows
    return out


#: The tensors of the transport phenomena (3 x 3, in the axes 1 2 3) and their physics:
#: the conductivity, and those of electrical conduction, dielectrics, magnetostatics and
#: diffusion
TRANSPORT_TENSORS = {f"{name}_tensor": physics for physics, name in TRANSPORT.items()}
#: The tensors of the results whose mean a study gives per volume, with its inverse: the
#: stiffness and its compliance, the conductivity and its resistivity (the tensor of each
#: transport phenomenon and its inverse)
TENSOR_PROPERTIES = ("stiffness", *TRANSPORT_TENSORS)
#: A column of the results holding a component of a tensor: <property>.<bc>.C<i><j>
_TENSOR_ENTRY = re.compile(rf"^({'|'.join(TENSOR_PROPERTIES)})\.([^.]+)\.C([1-6])([1-6])$")
#: The off-diagonal components of an orthotropic tensor in its axes (stiffnesses in space
#: and in the plane; a conductivity has none)
_ORTHOTROPIC = {
    ("stiffness", 6): {(0, 1), (0, 2), (1, 2)},
    ("stiffness", 3): {(0, 1)},
}
#: The notation of a conductivity tensor and of its inverse in the studies (the components
#: C11 ... C33 of the columns of results)
CONDUCTIVITY_NOTATION = (
    "the matrix K_ij of the flux q_i = -K_ij dT/dx_j, i, j = 1 2 3 the axes (C_ij in the "
    "columns); S = K^-1, the resistivity, in the same notation"
)
#: What the matrices of :meth:`TensorStatistics.to_dict` are, for the stiffnesses of an
#: ensemble study (``to_dict`` states those of the windows and of the conductivities)
TENSOR_DEFINITIONS = {
    "stiffness": "Cbar, the mean of the apparent stiffnesses C of the realizations",
    "stiffness_std": "the standard deviation of each component of C over the realizations",
    "compliance": "S = Cbar^-1, the inverse of the mean stiffness (S Cbar = I), not the mean "
    "of the compliances of the realizations (mean_compliance)",
    "compliance_std": "the standard deviation of each component of the compliances C^-1 of "
    "the realizations",
    "mean_compliance": "<C^-1>, the mean of the compliances of the realizations (>= Cbar^-1)",
}
#: The names of a tensor and of its inverse, singular and plural
TENSOR_WORDS = {
    "stiffness": ("stiffness", "stiffnesses", "compliance", "compliances"),
    "conductivity_tensor": ("conductivity", "conductivities", "resistivity", "resistivities"),
    "electrical_conductivity_tensor": (
        "electrical conductivity",
        "electrical conductivities",
        "electrical resistivity",
        "electrical resistivities",
    ),
    "permittivity_tensor": (
        "permittivity",
        "permittivities",
        "dielectric impermeability",
        "dielectric impermeabilities",
    ),
    "permeability_tensor": ("permeability", "permeabilities", "reluctivity", "reluctivities"),
    "diffusivity_tensor": (
        "diffusivity",
        "diffusivities",
        "inverse diffusivity",
        "inverse diffusivities",
    ),
}
#: What the window tensors are (:mod:`morphogen.rve.windows`)
_WINDOW_TENSOR = {
    "stiffness": "C_kl = <eps_k : C : eps_l>_w, the window means of the mutual energy "
    "densities of the solutions of the whole sample under the unit macroscopic strains E_k "
    "(symmetric, positive semi-definite; tiling the sample, their mean is its apparent "
    "stiffness, by Hill's lemma)",
    "conductivity_tensor": "C_kl = <grad T_k . K grad T_l>_w, the window means of the mutual "
    "dissipations of the solutions of the whole sample under the unit macroscopic gradients "
    "G_k (symmetric, positive semi-definite; tiling the sample, their mean is its apparent "
    "conductivity, by Hill's lemma)",
}
for _prop in TRANSPORT_TENSORS:  # the other transports: the same tensors of their potentials
    _WINDOW_TENSOR.setdefault(
        _prop,
        _WINDOW_TENSOR["conductivity_tensor"].replace("conductivity", TENSOR_WORDS[_prop][0]),
    )


def _definitions(prop: str, method: str = "ensemble") -> dict[str, str]:
    """What the matrices of :meth:`TensorStatistics.to_dict` are, for the tensors ``prop``
    of the realizations of an ensemble study or of the windows of one realization."""
    if prop == "stiffness" and method == "ensemble":
        return dict(TENSOR_DEFINITIONS)
    one, many, inverse, inverses = TENSOR_WORDS.get(prop, TENSOR_WORDS["stiffness"])
    samples = "windows" if method == "windows" else "realizations"
    what = f"apparent {many} C of the realizations"
    if method == "windows":
        what = f"tensors C of the windows, {_WINDOW_TENSOR.get(prop, _WINDOW_TENSOR['stiffness'])}"
    return {
        "stiffness": f"Cbar, the mean of the {what}",
        "stiffness_std": f"the standard deviation of each component of C over the {samples}",
        "compliance": f"S = Cbar^-1, the inverse of the mean {one} (S Cbar = I), the "
        f"{inverse}, not the mean of the {inverses} of the {samples} (mean_compliance)",
        "compliance_std": f"the standard deviation of each component of the {inverses} C^-1 "
        f"of the {samples}",
        "mean_compliance": f"<C^-1>, the mean of the {inverses} of the {samples} (>= Cbar^-1)",
    }


def tensor_notation(prop: str, tensor) -> str:
    """The notation of a tensor ``prop`` of the studies and of its inverse: of a stiffness
    (:func:`morphogen.homogenization.postprocess.notation`: Voigt order 11 22 33 23 31 12,
    engineering shears; 11 22 12 for a 3 x 3 one), of a conductivity
    (:data:`CONDUCTIVITY_NOTATION`) or of the tensor of another transport phenomenon (the
    same, of its flux and potential)."""
    from morphogen.homogenization.postprocess import notation

    if prop == "conductivity_tensor":
        return CONDUCTIVITY_NOTATION
    if prop in TRANSPORT_TENSORS:
        potential, flux, _ = TRANSPORT_LAWS[TRANSPORT_TENSORS[prop]]
        inverse = TENSOR_WORDS[prop][2]
        return (
            f"the matrix K_ij of the {flux} q_i = -K_ij d({potential})/dx_j, i, j = 1 2 3 the "
            f"axes (C_ij in the columns); S = K^-1, the {inverse}, in the same notation"
        )
    return notation(tensor)


def is_tensor(name: str) -> bool:
    """Whether a property asked of a study names a tensor (:data:`TENSOR_PROPERTIES`):
    ``stiffness`` or ``stiffness.<bc>`` (``conductivity_tensor``, ``conductivity_tensor.<bc>``),
    whereas ``stiffness.<bc>.C44`` is the series of one of its components."""
    head, _, rest = str(name).partition(".")
    return head in TENSOR_PROPERTIES and "." not in rest


def _matrix(m) -> list:
    """A matrix for JSON: nested lists, None for undefined entries."""
    return [[float(x) if math.isfinite(x) else None for x in row] for row in np.asarray(m)]


#: The relative size, to the largest diagonal component of the tensor, under which a
#: component is zero to round-off (the significance of the couplings)
_ROUND_OFF = 1e-12


def _inverse(m: np.ndarray) -> np.ndarray:
    """The compliance of a stiffness (NaN when it is singular)."""
    from morphogen.homogenization.postprocess import compliance

    try:
        return compliance(m)
    except np.linalg.LinAlgError:
        return np.full_like(m, np.nan)


def _inverses(c: np.ndarray) -> np.ndarray:
    """The compliances of finite stiffnesses ``(k, d, d)`` at once, those of ``_inverse``:
    NaN for the singular ones, exactly or to round-off (the guard of
    :func:`morphogen.homogenization.postprocess.compliance`: its symmetric part not positive
    definite beyond round-off; a window within a pore)."""
    from morphogen.homogenization.postprocess import _ROUND_OFF as singular

    w = np.linalg.eigvalsh(0.5 * (c + np.swapaxes(c, 1, 2)))
    regular = w[:, 0] > singular * w[:, -1]  # so invertible
    s = np.full_like(c, np.nan)
    if regular.any():
        s[regular] = np.linalg.inv(c[regular])
    return s


@dataclass
class TensorStatistics:
    """The mean stiffness and compliance per volume of the realizations of one boundary
    condition (see the module documentation), or of the windows of one realization
    (``method="windows"``, :mod:`morphogen.rve.windows`); the conductivity and its
    resistivity likewise (``property="conductivity_tensor"``).

    Per volume (``volume``, sorted, with ``n`` realizations): ``mean`` the mean stiffness
    ``Cbar`` and ``std`` the standard deviation of each of its components (NaN for a
    single realization); ``compliance`` the inverse of the mean stiffness, ``S =
    Cbar^-1`` (``compliance @ mean = I``); ``compliance_std`` the standard deviation of the
    components of the compliances ``C^-1`` of the realizations and ``mean_compliance``
    their mean ``<C^-1>``. Arrays ``(m, 6, 6)`` (``(m, 3, 3)`` in the plane) in morphoGen's
    Voigt order 11 22 33 23 31 12 with engineering shears (:attr:`notation`; a
    conductivity ``(m, 3, 3)`` in the axes 1 2 3).
    ``components``: the analysis of each component of the stiffnesses (``C11`` ...
    ``C66``) and of the compliances (``S11`` ... ``S66``) of the realizations as a series,
    with RVE sizes for the diagonal ones and those that the realizations distinguish from
    zero (:meth:`significant`). ``dimension``: that of the samples (3, cubes; 2, the
    squares of a planar image, the volumes their areas), their edges ``V^(1/dimension)``
    (:meth:`edges`). ``tilings``: by the window method with shifted tilings, the number of
    tilings of each size, ``n`` then the windows of one tiling (the independent ones), the
    means those of all the windows, the standard deviations the roots of their variances
    pooled within the tilings (:mod:`morphogen.rve.scaling`); None: one tiling.
    """

    property: str
    bc: str
    volume: np.ndarray
    n: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    compliance: np.ndarray
    compliance_std: np.ndarray
    mean_compliance: np.ndarray
    components: dict[str, SeriesAnalysis] = field(default_factory=dict)
    level: float = 0.95
    method: str = "ensemble"
    dimension: int = 3
    tilings: np.ndarray | None = None

    @property
    def key(self) -> str:
        return f"{self.property}.{self.bc}" if self.bc else self.property

    def edges(self) -> np.ndarray:
        """The edges of the volumes, ``V^(1/dimension)``: of cubes, of squares."""
        return np.array([float(v) ** (1.0 / self.dimension) for v in self.volume])

    @property
    def notation(self) -> str:
        """The notation of the tensor and of its inverse (:func:`tensor_notation`:
        :data:`morphogen.homogenization.postprocess.NOTATION` for a stiffness)."""
        return tensor_notation(self.property, self.mean[0])

    @property
    def samples(self) -> str:
        """What its statistics are over: ``realizations`` or ``windows``."""
        return "windows" if self.method == "windows" else "realizations"

    @staticmethod
    def index(name: str) -> tuple[int, int]:
        """The row and column of a component (``C44``, ``S44``: ``(3, 3)``)."""
        return int(name[1]) - 1, int(name[2]) - 1

    def significant(self, name: str) -> bool:
        """Whether the mean of a component is distinguished from zero: always for the
        diagonal (positive, the stiffness and the compliance being positive definite); for
        the others, by the realizations of the largest volume that has several, Student's
        t test at the ``level`` of the study (``|Zbar| sqrt(n) / D`` above the quantile of
        n - 1 degrees of freedom), and with a single realization per volume, whether it
        is a component of an orthotropic stiffness in its axes (a conductivity in its axes
        has none). The couplings that vanish on average have no relative precision, hence
        no RVE size."""
        from scipy import stats as distributions

        i, j = self.index(name)
        if i == j:
            return True
        stats = self.components[name].stats
        several = np.flatnonzero(np.isfinite(stats.variance))
        if not several.size:
            return (i, j) in _ORTHOTROPIC.get((self.property, len(self.mean[0])), set())
        k = several[-1]
        mean, std, n = abs(stats.mean[k]), math.sqrt(stats.variance[k]), int(stats.n[k])
        # zero to round-off (an exact symmetry, as of laminates, whose couplings are 1e-17
        # of the diagonal with one sign): not distinguished from zero, whatever its t
        diagonal = np.abs(np.diag((self.compliance if name[0] == "S" else self.mean)[k]))
        diagonal = diagonal[np.isfinite(diagonal)]
        if not (diagonal.size and mean > _ROUND_OFF * diagonal.max()):
            return False
        if std == 0:
            return True
        quantile = distributions.t.ppf(0.5 + self.level / 2.0, n - 1)
        return bool(mean * math.sqrt(n) / std > quantile)

    def volume_rows(self) -> list[dict]:
        """One row per volume: its edge (of a cube, or of a square), its realizations
        (windows, and their tilings when shifted), the components of ``Cbar``, of ``S =
        Cbar^-1`` and their standard deviations (``C11_std`` ...), the rows of
        ``rve_<property>.csv``."""
        from morphogen.homogenization.postprocess import matrix_entries

        rows = []
        for k, (v, edge) in enumerate(zip(self.volume, self.edges(), strict=True)):
            row = {"bc": self.bc, "volume": float(v), "edge": float(edge)}
            row["n"] = int(self.n[k])
            if self.tilings is not None:
                row["tilings"] = int(self.tilings[k])
            row.update(matrix_entries(self.mean[k], "C"))
            row.update(matrix_entries(self.compliance[k], "S"))
            for prefix, m in (("C", self.std[k]), ("S", self.compliance_std[k])):
                row.update({f"{name}_std": x for name, x in matrix_entries(m, prefix).items()})
            rows.append(row)
        return rows

    def to_dict(self) -> dict:
        volumes = []
        for k, (v, edge) in enumerate(zip(self.volume, self.edges(), strict=True)):
            entry = {"volume": float(v), "edge": float(edge), "n": int(self.n[k])}
            if self.tilings is not None:
                entry["tilings"] = int(self.tilings[k])
            entry.update(
                stiffness=_matrix(self.mean[k]),
                stiffness_std=_matrix(self.std[k]),
                compliance=_matrix(self.compliance[k]),
                compliance_std=_matrix(self.compliance_std[k]),
                mean_compliance=_matrix(self.mean_compliance[k]),
            )
            volumes.append(entry)
        out = {
            "property": self.property,
            "bc": self.bc,
            "notation": self.notation,
            "method": self.method,
            "definitions": _definitions(self.property, self.method),
            "volumes": volumes,
            "components": {name: a.to_dict() for name, a in self.components.items()},
        }
        if self.dimension != 3:  # the squares of a planar image: edges A^(1/2)
            out["dimension"] = self.dimension
        return out


def _std(a: np.ndarray, tilings: np.ndarray | None = None) -> np.ndarray:
    """The standard deviation of each component of tensors ``(k, d, d)`` (NaN for one); of
    the windows of shifted tilings (``tilings`` the tiling of each), the root of their
    variance pooled within the tilings."""
    if tilings is not None and np.unique(tilings).size > 1:
        return np.sqrt(_pooled_variance(a, tilings))
    return a.std(axis=0, ddof=1) if len(a) > 1 else np.full(a.shape[1:], np.nan)


def _tensor_statistics(
    prop: str,
    bc: str,
    groups,
    options: StudyOptions | None,
    method: str = "ensemble",
    dimension: int = 3,
) -> TensorStatistics:
    """The statistics of tensors given by volume (:class:`TensorStatistics`): ``groups`` of
    (volume, finite tensors ``(k, d, d)``), the volumes increasing, one volume at a time (the
    windows of a size: no series of their components is kept, only their moments), or of
    (volume, tensors, the tiling of each) for the windows of shifted tilings: their means
    over all the windows, their variances pooled within the tilings, the counts those of
    one (:mod:`morphogen.rve.scaling`). The analysis of each component is that of a series
    of its values (:func:`analyze_series`, without bootstrap): its finite values per volume,
    their mean and unbiased variance. ``dimension``: that of the volumes (2: areas)."""
    options = replace(options or StudyOptions(), bootstrap=0)
    d = groups[0][1].shape[1]
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    names = [f"{x}{i + 1}{j + 1}" for x in "CS" for i, j in pairs]
    moments: dict[str, list[tuple]] = {name: [] for name in names}
    per_volume = []
    shifted = any(len(g) > 2 and g[2] is not None for g in groups)
    for v, c, *rest in groups:
        tilings = rest[0] if rest else None
        count = 1 if tilings is None else np.unique(tilings).size  # of tilings
        s = _inverses(c)  # the compliances of the realizations
        mean = c.mean(axis=0)
        per_volume.append((v, len(c) // count, mean, _std(c, tilings), _inverse(mean),
                           _std(s, tilings), s.mean(axis=0), count))  # fmt: skip
        for letter, values in (("C", c), ("S", s)):
            for i, j in pairs:
                z = np.ascontiguousarray(values[:, i, j])
                finite = np.isfinite(z)
                z = z[finite]
                if z.size:
                    of = None if tilings is None else tilings[finite]
                    k, m, var = _moments(z, of)
                    tiled = 1 if of is None else np.unique(of).size
                    moments[f"{letter}{i + 1}{j + 1}"].append((v, k, m, var, tiled))
    columns = (np.array(a) for a in zip(*per_volume, strict=True))
    volume, n, mean, std, comp, comp_std, mean_comp, tiled = columns
    out = TensorStatistics(
        property=prop,
        bc=bc,
        volume=volume.astype(float),
        n=n.astype(int),
        mean=mean,
        std=std,
        compliance=comp,
        compliance_std=comp_std,
        mean_compliance=mean_comp,
        level=options.level,
        method=method,
        dimension=dimension,
        tilings=tiled.astype(int) if shifted else None,
    )
    for name in names:
        if not moments[name]:  # singular stiffnesses
            continue
        v, k, m, var, t = (np.array(a, dtype=float) for a in zip(*moments[name], strict=True))
        t = t.astype(int) if shifted else None
        stats = SizeStatistics(v, k.astype(int), m, var, dimension=dimension, tilings=t)
        analysis = _analysis(Series(name, bc, [], []), stats, options, None, None, False)
        out.components[name] = analysis
        if not out.significant(name):
            analysis.rve, analysis.needed = [], []
    return out


def tensor_statistics(
    volumes,
    tensors,
    prop: str = "stiffness",
    bc: str = "",
    options: StudyOptions | None = None,
) -> TensorStatistics:
    """The mean stiffness and compliance per volume of realizations
    (:class:`TensorStatistics`): ``tensors`` their apparent stiffnesses (k x 6 x 6, or k x
    3 x 3 in the plane; Voigt order 11 22 33 23 31 12, engineering shears; or their
    conductivities, k x 3 x 3, ``prop="conductivity_tensor"``) at their ``volumes``;
    ``options`` those of the study (gamma, range of the fits, precisions), its components
    analysed without bootstrap."""
    v = np.asarray(volumes, dtype=float)
    c = np.asarray(tensors, dtype=float)
    if c.ndim != 3 or c.shape[1:] not in ((6, 6), (3, 3)) or len(c) != len(v):
        raise ValueError("tensor_statistics: one 6 x 6 (or 3 x 3) stiffness per volume")
    ok = np.isfinite(v) & (v > 0) & np.isfinite(c).all(axis=(1, 2))
    v, c = v[ok], c[ok]
    if not v.size:
        raise ValueError(f"{prop}.{bc}: no finite stiffness")
    return _tensor_statistics(prop, bc, [(x, c[v == x]) for x in np.unique(v)], options)


def tensors_from_rows(
    rows: list[dict],
    *,
    properties=None,
    volume_column: str = "volume",
    options: StudyOptions | None = None,
) -> list[TensorStatistics]:
    """The mean stiffness and compliance per volume (:func:`tensor_statistics`) of each
    boundary condition whose stiffness the rows of results hold, in the columns
    ``stiffness.<bc>.C11`` ... ``C66`` of the runs (the upper triangle; ``C11`` ...
    ``C33`` in the plane). ``properties``: the tensors kept, ``stiffness`` or
    ``stiffness.<bc>`` (None: all)."""
    found: dict[tuple[str, str], dict[tuple[int, int], str]] = {}
    for row in rows:
        for column in row:
            m = _TENSOR_ENTRY.match(str(column))
            if m:
                found.setdefault((m[1], m[2]), {})[(int(m[3]) - 1, int(m[4]) - 1)] = column
    out = []
    for (prop, bc), entries in found.items():
        if properties is not None and prop not in properties and f"{prop}.{bc}" not in properties:
            continue
        d = 6 if max(max(ij) for ij in entries) > 2 else 3
        upper = [(i, j) for i in range(d) for j in range(i, d)]
        if any(ij not in entries for ij in upper):
            continue
        tensors = np.full((len(rows), d, d), np.nan)
        for k, row in enumerate(rows):
            for i, j in upper:
                tensors[k, i, j] = tensors[k, j, i] = _parse(row.get(entries[(i, j)]))
        if not np.isfinite(tensors).all(axis=(1, 2)).any():
            continue
        volumes = [_volume(row, volume_column) for row in rows]
        out.append(tensor_statistics(volumes, tensors, prop, bc, options))
    return out


@dataclass
class StudyReport:
    """Analyses of all the series of a study, with cross-series results, and the mean
    stiffness and compliance per volume of each boundary condition (``tensors``)."""

    analyses: list[SeriesAnalysis]
    discrepancy: dict[str, list[dict]]
    options: StudyOptions
    bounds: dict = field(default_factory=dict)
    method: str = "ensemble"
    tensors: list[TensorStatistics] = field(default_factory=list)

    @property
    def dimension(self) -> int:
        """That of its samples, whose edges the tables print: 3 (cubes), or 2 (the squares
        of a planar image, the volumes their areas)."""
        if self.analyses:
            return self.analyses[0].stats.dimension
        return self.tensors[0].dimension if self.tensors else 3

    def __getitem__(self, key: str) -> SeriesAnalysis:
        for a in self.analyses:
            if a.key == key:
                return a
        raise KeyError(key)

    def to_dict(self) -> dict:
        out = {
            "method": self.method,
            "options": {
                "gamma": self.options.gamma,
                "min_volume": self.options.min_volume,
                "max_volume": self.options.max_volume,
                "relative_errors": list(self.options.relative_errors),
                "n_realizations": list(self.options.n_realizations),
                "bootstrap": self.options.bootstrap,
                "weights": self.options.weights,
                "level": self.options.level,
            },
            "properties": {a.key: a.to_dict() for a in self.analyses},
            "discrepancy": self.discrepancy,
            "bounds": self.bounds,
        }
        if self.tensors:
            out["tensors"] = {t.key: t.to_dict() for t in self.tensors}
        return out


def analyze_study(
    series: list[Series],
    options: StudyOptions | None = None,
    *,
    count: Callable[[float], float] | None = None,
    bounds: dict | None = None,
    theory: dict | None = None,
    method: str = "ensemble",
    tensors: list[TensorStatistics] | None = None,
) -> StudyReport:
    """Analyse every series and compare the boundary conditions; ``tensors``: the mean
    stiffness and compliance per volume (:func:`tensors_from_rows`) joining the report."""
    options = options or StudyOptions()
    analyses = [analyze_series(s, options, count=count, theory=theory) for s in series]
    return StudyReport(
        analyses=analyses,
        discrepancy=discrepancy(analyses),
        options=options,
        bounds=dict(bounds or {}),
        method=method,
        tensors=list(tensors or []),
    )
