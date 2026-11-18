"""Identification of the parameters of a behaviour on reference responses, as ``Zopt``
does for ``Zsim``: the parameters chosen are those whose material point
(:func:`~morphogen.homogenization.nonlinear.simulate_point`), under the loading path of
each reference, gives the stresses of the reference at its times in the least-squares
sense (a trust-region method on the logarithms of the parameters, which stay positive and
within ``span`` decades of their starting values). A reference is a response along a
path: the macroscopic curves of a loading path of a morphology (to give it a macroscopic
law) or the points of a test. Several references are followed together, each by its
weight whatever its number of points: a tension and a shear, cycles of several
amplitudes, which tell the isotropic hardening from the kinematic one.

The parameters are named after the keys of the configurations of the material
(:meth:`~morphogen.homogenization.materials.Material.from_config`): ``young``, ``poisson``
and, of its von Mises plasticity, ``yield``, ``hardening``, the isotropic terms ``Q``,
``b`` (``Q2``, ``b2`` for the second...), the kinematic terms ``C``, ``D`` (``C2``...)
and Norton's ``K``, ``n``; of its porous plasticity, those of its matrix (``yield``,
``hardening``, ``Q``, ``b``..., the kinematic terms ``C1``, ``D1``, ``C2``..., ``K``,
``n``), the initial ``porosity``, Gurson's ``q1``,
``q2``, ``q3`` and the coalescence ``fc``, ``fF``, Rousselier's ``sigma1``, ``D``, the
elliptic ``C``, ``F`` (numbers, not tables of the porosity) and the nucleation ``fN``,
``eN``, ``sN``, ``A``, ``B``, ``e0``: a porous morphology (a matrix with voids) gives
its macroscopic law, the classic calibration of Gurson's model on unit cells under
several triaxialities. Of the behaviours of :mod:`morphogen.homogenization.laws`: the
terms of a viscoelasticity, ``g1``, ``tau1``, ``g2``... (shear) and ``k1``, ``tauk1``...
(bulk), which give a viscoelastic composite its Prony series from its relaxation; the
positive parameters of a hyperelasticity (``mu``, ``c10``, ``c01``, ``c20``, ``c30``,
``lock``, ``bulk``; the material given by its law, without ``young`` and ``poisson``); and
those of a damage (``threshold``, ``strength``, ``ultimate``, ``fracture_energy``,
``ratio``, ``viscosity``)."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from morphogen.homogenization.behaviours import HILL_COMPONENTS
from morphogen.homogenization.materials import Material
from morphogen.homogenization.paths import PathResult

#: The parameters of von Mises plasticity that can be identified, beside young and poisson
PLASTIC = ("yield", "hardening", "Q", "b", "C", "D", "K", "n")
#: The coefficients of Hill's criterion (``hill11`` ... ``hill31``: those of the components)
HILL = tuple(f"hill{c}" for c in HILL_COMPONENTS)
#: Those of porous plasticity (its matrix, its criterion, its nucleation)
POROUS = ("yield", "hardening", "Q", "b", "K", "n", "porosity", "q1", "q2", "q3", "fc", "fF",
          "sigma1", "D", "C", "F", "fN", "eN", "sN", "A", "B", "e0")  # fmt: skip
#: The coefficients of the criteria of porous plasticity, beside its matrix
_CRITERION = ("porosity", "q1", "q2", "q3", "fc", "fF", "sigma1", "D", "C", "F")
#: The behaviours of :mod:`morphogen.homogenization.laws` in the configuration of a material
ANALYTIC = ("viscoelasticity", "hyperelasticity", "damage")
#: The parameters of a damage that can be identified
_DAMAGE = ("threshold", "strength", "ultimate", "fracture_energy", "ratio", "viscosity")
#: The residuals of a material point that fails (its Newton iterations diverge)
FAILED = 1e3


def _where(config: dict, name: str) -> tuple[dict, str]:
    """The mapping of the configuration of a material that holds the parameter ``name``,
    and its key there."""
    if name in ("young", "poisson"):
        if name not in config:
            raise ValueError(f"{name}: the material has no isotropic elasticity")
        return config, name
    if isinstance(config.get("porous_plasticity"), dict):
        return _where_porous(config["porous_plasticity"], name)
    if any(isinstance(config.get(key), dict) for key in ANALYTIC):
        return _where_analytic(config, name)
    match = re.fullmatch(r"(Q|b|C|D)(\d*)", name)
    plasticity = config.get("plasticity")
    if not isinstance(plasticity, dict):
        raise ValueError(f"{name}: the material has no von Mises plasticity")
    if name in ("yield", "hardening"):
        return plasticity, name
    if name in ("K", "n"):
        if not isinstance(plasticity.get("norton"), dict):
            raise ValueError(f"{name}: the plasticity has no Norton flow (give K and n)")
        return plasticity["norton"], name
    if name in HILL:
        # von Mises is Hill's criterion of coefficients 1: those not given are 1
        hill = plasticity.get("hill")
        if hill is None or isinstance(hill, (list, tuple)):
            hill = dict(zip(HILL_COMPONENTS, hill or (1.0,) * 6, strict=True))
            plasticity["hill"] = hill
        hill = {str(k): v for k, v in hill.items()}
        plasticity["hill"] = hill
        hill.setdefault(name[4:], 1.0)
        return hill, name[4:]
    if match is None:
        raise ValueError(
            f"parameter {name!r}: one of young, poisson, {', '.join(PLASTIC + HILL)} (Q2, "
            "b2... for the next terms)"
        )
    letter, rank = match.group(1), int(match.group(2) or 1)
    terms = plasticity.get("isotropic" if letter in "Qb" else "kinematic") or []
    if not 1 <= rank <= len(terms):
        kind = "isotropic" if letter in "Qb" else "kinematic"
        raise ValueError(f"{name}: the plasticity has {len(terms)} {kind} terms")
    return terms[rank - 1], letter


def _where_analytic(config: dict, name: str) -> tuple[dict, str]:
    """The mapping of the configuration of a material with a viscoelasticity, a
    hyperelasticity or a damage that holds ``name``."""
    visco, hyper, damage = (config.get(key) for key in ANALYTIC)
    match = re.fullmatch(r"(g|tau|k|tauk)(\d+)", name)
    if match is not None and isinstance(visco, dict):
        letter, rank = match.group(1), int(match.group(2))
        kind, key = ("shear", "g") if letter in ("g", "tau") else ("bulk", "k")
        terms = visco.get(kind) or []
        if not 1 <= rank <= len(terms):
            raise ValueError(f"{name}: the viscoelasticity has {len(terms)} {kind} terms")
        if not isinstance(terms[rank - 1], dict):  # pairs: mappings, to be set in place
            visco[kind] = terms = [
                t if isinstance(t, dict) else {key: t[0], "tau": t[1]} for t in terms
            ]
        return terms[rank - 1], key if letter in ("g", "k") else "tau"
    if isinstance(hyper, dict) and name in ("mu", "c10", "c01", "c20", "c30", "lock", "bulk"):
        return hyper, name
    if isinstance(damage, dict) and name in _DAMAGE:
        return damage, name
    raise ValueError(
        f"parameter {name!r}: one of young, poisson, g1, tau1... (shear terms) and k1, "
        "tauk1... (bulk terms) of a viscoelasticity, mu, c10, c01, c20, c30, lock, bulk of a "
        f"hyperelasticity, {', '.join(_DAMAGE)} of a damage"
    )


def _where_porous(porous: dict, name: str) -> tuple[dict, str]:
    """The mapping of the configuration of a porous plasticity that holds ``name``."""
    if name == "yield":
        return porous, "R0" if "R0" in porous else "yield"
    if name == "hardening":
        return porous, "H" if "H" in porous else "hardening"
    if name in ("K", "n"):
        if not isinstance(porous.get("norton"), dict):
            raise ValueError(f"{name}: the porous plasticity has no Norton flow (give K and n)")
        return porous["norton"], name
    if name in _CRITERION:
        if isinstance(porous.get(name), (list, tuple, dict)):
            raise ValueError(f"{name}: a table of the porosity, not identified")
        return porous, name
    if name in ("fN", "eN", "sN", "A", "B", "e0"):
        nucleation = porous.get("nucleation")
        if not isinstance(nucleation, dict) or name not in nucleation:
            raise ValueError(f"{name}: the porous plasticity has no nucleation with {name}")
        return nucleation, name
    # C and D alone: the coefficients of the criteria; C1, D1...: the kinematic terms
    match = re.fullmatch(r"(Q|b)(\d*)|(C|D)(\d+)", name)
    if match is None:
        raise ValueError(
            f"parameter {name!r} of porous plasticity: one of young, poisson, "
            f"{', '.join(POROUS)} (Q2, b2... for the next isotropic terms, C1, D1... for the "
            "kinematic ones)"
        )
    letter = match.group(1) or match.group(3)
    rank = int(match.group(2) or match.group(4) or 1)
    kind = "isotropic" if letter in "Qb" else "kinematic"
    terms = porous.get(kind) or []
    if not 1 <= rank <= len(terms):
        raise ValueError(f"{name}: the matrix has {len(terms)} {kind} terms")
    return terms[rank - 1], letter


def get_parameter(config: dict, name: str) -> float:
    """The value of a parameter in the configuration of a material."""
    mapping, key = _where(config, name)
    return float(mapping.get(key, 0.0))


def set_parameter(config: dict, name: str, value: float) -> None:
    """Set a parameter in the configuration of a material (in place)."""
    mapping, key = _where(config, name)
    mapping[key] = float(value)


def driven(result: PathResult) -> int:
    """The index of the stress component that varies most along a response."""
    return int(np.argmax(np.ptp(np.asarray(result.stress, dtype=float), axis=0)))


@dataclass
class Reference:
    """A response the identification follows: the ``stresses`` (n x 6 in the order of the
    paths, or the n values of the components compared) at the ``times`` of a loading
    ``path`` (its name or specification), the indices of the ``components`` compared (by
    default the one that varies most) and the ``weight`` of the reference in the gap."""

    path: object
    times: object
    stresses: object
    components: list[int] | None = None
    weight: float = 1.0


def _prepared(ref: Reference) -> tuple[Reference, np.ndarray]:
    """The reference checked, its components resolved, and the stresses compared."""
    times = np.asarray(ref.times, dtype=float)
    stresses = np.asarray(ref.stresses, dtype=float)
    components = ref.components
    if stresses.ndim == 1:
        if components is None or len(components) != 1:
            raise ValueError("a reference of one component: give its index (components)")
        stresses = stresses[:, None]
    elif components is None:
        components = [int(np.argmax(np.ptp(stresses, axis=0)))]
    components = [int(c) for c in components]
    if not components or any(not 0 <= c < 6 for c in components):
        raise ValueError("components: indices from 0 to 5 (11, 22, 33, 23, 31, 12)")
    if stresses.shape[1] == 6:
        compared = stresses[:, components]
    elif stresses.shape[1] == len(components):
        compared = stresses
    else:
        raise ValueError("the stresses of a reference: its 6 components, or those compared")
    if len(times) != len(compared) or len(times) < 2:
        raise ValueError("the reference: at least two times, one row of stresses at each")
    if not float(ref.weight) > 0:
        raise ValueError("the weight of a reference: positive")
    return Reference(ref.path, times, stresses, components, float(ref.weight)), compared


@dataclass
class Identification:
    """The parameters identified: their starting and final values, the material, the
    references followed (their components resolved), the responses of the material point
    under their paths with the parameters found and with the starting values (None where
    it fails), the relative root mean square of the gap to each reference (to its largest
    stress) and of the weighted gap to all, before and after, and the number of
    evaluations (a material point under the path of each reference)."""

    parameters: list[str]
    initial: dict
    values: dict
    material: dict
    references: list[Reference]
    responses: list[PathResult]
    initial_responses: list[PathResult | None]
    gaps: list[float]
    initial_gaps: list[float]
    error: float
    initial_error: float
    evaluations: int

    @property
    def response(self) -> PathResult:
        """The response under the path of the first reference."""
        return self.responses[0]

    @property
    def components(self) -> list[int]:
        """The components compared in the first reference."""
        return self.references[0].components


def identify(
    material: dict,
    path,
    times,
    stresses,
    parameters,
    *,
    components=None,
    kinematics: str = "small",
    span: float = 3.0,
    max_evaluations: int = 200,
    callback: Callable[[int, float], None] | None = None,
) -> Identification:
    """Identify ``parameters`` of ``material`` (a mapping of the configurations) so that
    its material point under ``path`` gives the ``stresses`` (n x 6 in the order of the
    paths, or n values of one component) of the reference at its ``times``
    (``components``: the indices compared, by default the component that varies most in
    the reference): :func:`identify_references` with one reference."""
    return identify_references(
        material,
        [Reference(path, times, stresses, components)],
        parameters,
        kinematics=kinematics,
        span=span,
        max_evaluations=max_evaluations,
        callback=callback,
    )


def identify_references(
    material: dict,
    references,
    parameters,
    *,
    kinematics: str = "small",
    span: float = 3.0,
    max_evaluations: int = 200,
    callback: Callable[[int, float], None] | None = None,
) -> Identification:
    """Identify ``parameters`` of ``material`` (a mapping of the configurations) so that
    its material point under the path of each of the ``references`` (:class:`Reference`)
    gives their stresses at their times: the gaps relative to the largest stress of each
    reference, the mean of their squares over each reference weighted by the references
    (each counts by its weight, whatever its number of points). ``callback(evaluations,
    error)`` after each evaluation (it may raise :class:`~morphogen.stopping.Stopped`)."""
    from scipy.optimize import least_squares

    from morphogen.homogenization.nonlinear import simulate_point

    parameters = list(dict.fromkeys(parameters))
    if not parameters:
        raise ValueError("parameters: at least one to identify")
    prepared = [_prepared(r) for r in references]
    if not prepared:
        raise ValueError("references: at least one response to follow")
    refs = [r for r, _ in prepared]
    compared = [c for _, c in prepared]
    scales = [float(np.abs(c).max()) or 1.0 for c in compared]
    sizes = np.array([c.size for c in compared], dtype=float)
    weights = np.array([r.weight for r in refs])
    # the mean of the squares is that of each reference weighted: one reference alone
    # gives the relative gaps of its points
    factors = np.sqrt(weights * sizes.sum() / (sizes * weights.sum()))
    base = copy.deepcopy(material)
    porous = base.get("porous_plasticity")
    if isinstance(porous, dict) and "q1" in parameters and "q3" not in parameters:
        q1, q3 = porous.get("q1"), porous.get("q3")
        if q1 is not None and q3 is not None and abs(q3 - q1**2) <= 1e-12 * q1**2:
            del porous["q3"]  # q3 = q1² (Tvergaard's, Z-mat's) follows q1
    start = np.array([get_parameter(base, p) for p in parameters])
    if np.any(start <= 0):
        zero = [p for p, v in zip(parameters, start, strict=True) if v <= 0]
        raise ValueError(f"{', '.join(zero)}: a positive starting value to identify")
    count = [0]

    def responses(x):
        config = copy.deepcopy(base)
        for name, value in zip(parameters, np.exp(x), strict=True):
            set_parameter(config, name, value)
        out = []
        for ref in refs:
            try:
                point = Material.from_config(config)
                out.append(simulate_point(point, ref.path, kinematics=kinematics))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                out.append(None)
        return config, out

    def gaps(results) -> list[np.ndarray]:
        out = []
        for ref, reference, scale, res in zip(refs, compared, scales, results, strict=True):
            if res is None:
                out.append(np.full(reference.size, FAILED))
                continue
            stress = np.asarray(res.stress)
            sim = np.column_stack([np.interp(ref.times, res.times, stress[:, c])
                                   for c in ref.components])  # fmt: skip
            out.append(((sim - reference) / scale).ravel())
        return out

    def residuals_of(results) -> np.ndarray:
        r = np.concatenate([g * f for g, f in zip(gaps(results), factors, strict=True)])
        count[0] += 1
        if callback is not None:
            callback(count[0], float(np.sqrt(np.mean(r**2))))
        return r

    x0 = np.log(start)
    _, initial = responses(x0)
    first = residuals_of(initial)
    fit = least_squares(
        lambda x: residuals_of(responses(x)[1]),
        x0,
        bounds=(x0 - span * np.log(10.0), x0 + span * np.log(10.0)),
        x_scale=1.0,
        max_nfev=int(max_evaluations),
        method="trf",
    )
    config, results = responses(fit.x)
    if any(res is None for res in results):
        raise ValueError("identification: the material point of the parameters found fails")
    final = gaps(results)
    return Identification(
        parameters=parameters,
        initial=dict(zip(parameters, start.tolist(), strict=True)),
        values={p: get_parameter(config, p) for p in parameters},
        material=config,
        references=refs,
        responses=results,
        initial_responses=initial,
        gaps=[float(np.sqrt(np.mean(g**2))) for g in final],
        initial_gaps=[float(np.sqrt(np.mean(g**2))) for g in gaps(initial)],
        error=float(np.sqrt(np.mean(fit.fun**2))),
        initial_error=float(np.sqrt(np.mean(first**2))),
        evaluations=count[0],
    )


#: The loadings of a macroscopic law: a tension along each axis, a shear in each plane
LOADINGS = ("11", "22", "33", "12", "23", "31")
#: Those of a higher stress triaxiality (porous morphologies): a tension along an axis whose
#: lateral strains are held, and a hydrostatic tension
TRIAXIAL = ("confined11", "confined22", "confined33", "hydrostatic")
_COLUMN = {"11": 0, "22": 1, "33": 2, "23": 3, "31": 4, "12": 5}


def loading_path(name: str, strain: float, n: int, cycle: bool = False) -> tuple[dict, list[int]]:
    """The path of a loading of a macroscopic law (:data:`LOADINGS`, :data:`TRIAXIAL`) to
    ``strain`` in ``n`` increments (with ``cycle``, then to ``-strain`` in ``2 n``), and the
    components of the stress compared."""
    ramp = [0.0, strain, -strain] if cycle else [0.0, strain]
    held = [0.0] * len(ramp)
    if cycle:
        timing = {"times": [0.0, 1.0, 3.0], "increments": [n, 2 * n]}
    else:
        timing = {"times": [0.0, 1.0], "increments": n}
    if name in LOADINGS:
        return {**timing, "strain": {name: ramp}}, [_COLUMN[name]]
    if name == "hydrostatic":
        return {**timing, "strain": {c: ramp for c in ("11", "22", "33")}}, [0, 1, 2]
    axis = name.removeprefix("confined")
    strains = {c: (ramp if c == axis else held) for c in ("11", "22", "33")}
    return {**timing, "strain": strains}, [0, 1, 2]


@dataclass
class MacroscopicLaw:
    """The macroscopic law of a morphology (:func:`macroscopic_law`): its responses under
    the loadings, as the references of the identification of the law."""

    loadings: tuple[str, ...]
    paths: list[dict]
    responses: list[PathResult]
    identification: Identification


def law_start(
    responses,
    loadings,
    law: str = "hill",
    *,
    kinematic: bool = False,
    matrix: dict | None = None,
    porosity: float | None = None,
) -> tuple[dict, list[str]]:
    """A starting law for the responses of a morphology under ``loadings`` (one loading
    each, :func:`macroscopic_law`: monotonic, or a cycle that first loads), and the
    parameters to fit: Young's modulus and Poisson's ratio of the first increment of the
    first tension (of the first shear without a tension: its shear modulus, ν = 0.3), the
    yield stress at 70 % of the stress at the peak of the loading (a shear times sqrt 3),
    the linear hardening of the slope of the second half of the loading; with
    ``kinematic``, a backstress ``(C, D)`` that takes half of that hardening and saturates
    over the strain of the loading; fitted, those and, for ``law="hill"``, the
    coefficients of the loadings run but the first (its coefficient stays 1: the
    coefficients and the yield stress have one scale). For ``law="gurson"``, Gurson's
    porous plasticity of the von Mises ``matrix`` (the configuration of a material) and of
    the ``porosity`` (the fraction of the voids), starting from Tvergaard's q1 = 1.5 and q2
    = 1: fitted, the elastic constants (from the first increment of a tension, or those of
    the matrix), q1 and q2."""
    if law not in ("hill", "mises", "gurson"):
        raise ValueError(f"law: 'hill', 'mises' or 'gurson', got {law!r}")
    loadings = [str(c) for c in loadings]
    if law == "gurson":
        return _gurson_start(responses, loadings, matrix, porosity)
    k = next((i for i, c in enumerate(loadings) if c in LOADINGS and c[0] == c[1]), 0)
    if loadings[k] not in LOADINGS:
        raise ValueError(f"{law} law: a tension or a shear among the loadings")
    c, res = loadings[k], responses[k]
    col = _COLUMN[c]
    strain, stress = np.asarray(res.strain), np.asarray(res.stress)
    slope = stress[1, col] / strain[1, col]
    if c[0] == c[1]:
        lateral = [j for j in range(3) if j != col]
        young = float(slope)
        poisson = float(np.clip(-np.mean(strain[1, lateral]) / strain[1, col], 0.0, 0.49))
        factor = 1.0
    else:  # a shear alone: the engineering strain, G = slope, E = 2 G (1 + 0.3)
        poisson = 0.3
        young = float(2.6 * slope)
        factor = np.sqrt(3.0)
    peak = int(np.argmax(np.abs(strain[:, col])))  # the end of the first loading
    end = factor * abs(stress[peak, col])
    half = peak // 2
    rise = factor * (abs(stress[peak, col]) - abs(stress[half, col]))
    run = abs(strain[peak, col] - strain[half, col]) / factor  # p: the shear over sqrt 3
    hardening = max(rise / run if run > 0 else 0.0, 1e-4 * young)
    plasticity = {"yield": 0.7 * end, "hardening": hardening}
    fitted = ["young", "poisson", "yield", "hardening"]
    if kinematic:
        plasticity["hardening"] = 0.5 * hardening
        span_p = abs(strain[peak, col]) / factor
        plasticity["kinematic"] = [{"C": 0.5 * hardening, "D": 1.0 / max(span_p, 1e-12)}]
        fitted += ["C", "D"]
    if law == "hill":
        plasticity["hill"] = {}
        fitted += [f"hill{d}" for d in loadings if d != c]
    return {"young": young, "poisson": poisson, "plasticity": plasticity}, fitted


def _gurson_start(responses, loadings, matrix, porosity) -> tuple[dict, list[str]]:
    """Gurson's law of a porous morphology: its matrix and porosity, q1 = 1.5, q2 = 1."""
    if matrix is None or porosity is None:
        raise ValueError("gurson law: the matrix (a von Mises material) and the porosity")
    if not 0.0 < float(porosity) < 1.0:
        raise ValueError(f"gurson law: a porosity between 0 and 1, got {porosity}")
    plasticity = matrix.get("plasticity")
    if not isinstance(plasticity, dict) or "young" not in matrix:
        raise ValueError("gurson law: a matrix of isotropic elasticity and von Mises plasticity")
    if plasticity.get("kinematic") or plasticity.get("hill") is not None:
        raise ValueError(
            "gurson law: a matrix without kinematic hardening (the backstresses of the porous "
            "law are not identified on the monotonic loadings of q1 and q2) or Hill's criterion"
        )
    young, poisson = float(matrix["young"]), float(matrix.get("poisson", 0.3))
    k = next((i for i, c in enumerate(loadings) if c in LOADINGS and c[0] == c[1]), None)
    if k is not None:  # the porous elasticity of the first increment of a tension
        col = _COLUMN[loadings[k]]
        strain = np.asarray(responses[k].strain)
        stress = np.asarray(responses[k].stress)
        lateral = [j for j in range(3) if j != col]
        young = float(stress[1, col] / strain[1, col])
        poisson = float(np.clip(-np.mean(strain[1, lateral]) / strain[1, col], 0.0, 0.49))
    porous = {"criterion": "gurson", "porosity": float(porosity), "q1": 1.5, "q2": 1.0}
    porous.update({key: copy.deepcopy(v) for key, v in plasticity.items()})
    start = {"young": young, "poisson": poisson, "porous_plasticity": porous}
    return start, ["young", "poisson", "q1", "q2"]


def macroscopic_law(
    respond: Callable[[dict], PathResult],
    start: dict | None = None,
    parameters=None,
    *,
    law: str = "hill",
    loadings=LOADINGS,
    strain: float = 0.01,
    increments: int = 20,
    cycle: bool = False,
    matrix: dict | None = None,
    porosity: float | None = None,
    kinematics: str = "small",
    span: float = 3.0,
    max_evaluations: int = 200,
    callback: Callable[[int, float], None] | None = None,
) -> MacroscopicLaw:
    """The law that gives a morphology its macroscopic behaviour: ``respond(path)``, the
    macroscopic response of the morphology along a loading path (a :class:`PathResult`:
    ``lambda path: NonlinearFFTProblem(image, materials).run(path)``), under a monotonic
    loading of each component of ``loadings`` to ``strain`` in ``increments`` steps (a
    tension along an axis, ``"11"``..., or a shear in a plane, ``"12"``...: the other
    stresses free; with ``cycle``, then back to ``-strain`` in twice as many, which shows
    the Bauschinger effect), then the identification of the law on these responses
    together (:func:`identify_references`, each response compared on its loaded
    component): the ``parameters`` of ``start`` (a mapping of the configurations of a
    material), or of the von Mises or Hill ``law`` that :func:`law_start` estimates from
    the responses, with a kinematic hardening when the loadings cycle, or Gurson's law of
    a porous morphology (``matrix``, ``porosity``), whose q1 and q2 need loadings of
    several stress triaxialities (:data:`TRIAXIAL`: a confined or hydrostatic tension,
    compared on its three normal stresses)."""
    loadings = tuple(str(c) for c in loadings)
    unknown = sorted(set(loadings) - set(LOADINGS + TRIAXIAL))
    if unknown or not loadings:
        known = ", ".join(LOADINGS + TRIAXIAL)
        raise ValueError(f"loadings: some of {known}, got {loadings}")
    if not strain > 0 or int(increments) < 1:
        raise ValueError("strain: positive, with at least one increment")
    paths, responses, references = [], [], []
    for c in loadings:
        path, components = loading_path(c, strain, int(increments), cycle)
        response = respond(path)
        paths.append(path)
        responses.append(response)
        references.append(Reference(path, response.times, response.stress, components))
    if start is None or parameters is None:
        estimated, fitted = law_start(responses, loadings, law, kinematic=cycle, matrix=matrix,
                                      porosity=porosity)  # fmt: skip
        start = estimated if start is None else start
        parameters = fitted if parameters is None else parameters
    fit = identify_references(start, references, parameters, kinematics=kinematics, span=span,
                              max_evaluations=max_evaluations, callback=callback)  # fmt: skip
    return MacroscopicLaw(loadings, paths, responses, fit)
