"""Alloys designed by thermodynamic and kinetic tools as microstructures of morphoGen: the
phases of an alloy, their fractions, sizes and elastic constants turned into the models of
its scales, homogenized, and compared with the properties the tool predicted.

Tools of alloy design compute, from a composition and a treatment, the phases at
equilibrium or after solidification (CALPHAD: Thermo-Calc, pycalphad, the equilibrium
and Scheil calculations of Leil-Mat), the precipitates of an ageing (KWN kinetics: their
fraction and the law of their radii), a grain size, and properties estimated by mixtures
of the phases (Voigt–Reuss–Hill moduli). An alloy is described here by a mapping (a JSON
or YAML file, :func:`read_alloy`) that such a tool fills:

.. code-block:: yaml

    alloy: AA7075-T6
    source: the tool that gave it
    composition: {Al: 89.9, Zn: 5.6, Mg: 2.5, Cu: 1.6, Cr: 0.23}   # normalized to 100
    basis: wt%                       # or at%
    temperature: 393.15              # K, of the state
    unit: um                         # of the lengths
    stress_unit: GPa                 # of the moduli
    grain_size: 50                   # mean equivalent diameter of the grains
    phases:
      - {name: FCC_A1, role: matrix, young: 70, poisson: 0.33}
      - {name: MgZn2, role: precipitates, fraction: 0.05, radius: {mean: 0.003, std: 0.001},
         young: 77, poisson: 0.3}
    properties: {young: 71.7}        # predicted by the tool, to compare

The roles: ``matrix`` (the grains of the polycrystal: one phase, the rest of the volume),
``grains`` (another phase of grains, a fraction of the cells: dual-phase steels),
``precipitates`` (spheres inside the matrix, far smaller than the grains) and
``particles`` (spheres among the grains, of their scale). The elastic constants of a phase
are its ``young`` and ``poisson``, or the ``cubic`` constants ``[C11, C12, C44]`` of its
crystals (the grains then turned by random orientations).

:func:`alloy_scales` gives the scales of the alloy as configurations of morphoGen: that of
the precipitates (a box of the matrix with about 60 of them, hard spheres of the law of
their radii), and that of the grains (a Voronoi polycrystal of about 64 grains, the phases
of its cells, the particles as inclusions), the matrix of the second being the first's
homogenized. :func:`homogenize_alloy` computes them by FFT and gives the effective moduli,
their Hashin–Shtrikman bounds and the departure from the moduli predicted.

Leil-Mat (mat.leil.app, Leil) gives these quantities on its canvas (the phases of its
equilibrium and Scheil calculations, the KWN trajectories of the precipitates, the moduli
of its property models); its programming interface is available on request only, so the
mapping is filled from its canvas or its reports until a client reads it directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROLES = ("matrix", "grains", "precipitates", "particles")
#: The objects of the box of each scale (precipitates, grains)
PRECIPITATES = 60
GRAINS = 64


@dataclass
class AlloyPhase:
    """A phase of an alloy: its name, role (see the module), volume fraction (the matrix:
    the rest), elastic constants (``young`` and ``poisson``, or the ``cubic`` constants of
    its crystals) and, for precipitates and particles, the law of their radii (a number,
    or ``{mean, std}``: a normal law truncated at three standard deviations)."""

    name: str
    role: str
    fraction: float | None = None
    young: float | None = None
    poisson: float | None = None
    cubic: tuple[float, float, float] | None = None
    radius: float | dict | None = None

    def material(self) -> dict:
        """The material of morphoGen's configurations."""
        if self.cubic is not None:
            return {"cubic": [float(c) for c in self.cubic]}
        return {"young": float(self.young), "poisson": float(self.poisson)}


@dataclass
class Alloy:
    """An alloy: its composition (normalized to 100, ``basis`` wt% or at%), the
    temperature of its state, its grain size and phases, and the properties predicted for
    it (see the module)."""

    alloy: str
    phases: list[AlloyPhase]
    grain_size: float
    composition: dict = field(default_factory=dict)
    basis: str = "wt%"
    temperature: float | None = None
    unit: str = "um"
    stress_unit: str = "GPa"
    properties: dict = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        if not self.grain_size > 0:
            raise ValueError("grain_size: the mean equivalent diameter of the grains, positive")
        if self.basis not in ("wt%", "at%"):
            raise ValueError("basis: wt% or at%")
        total = float(sum(self.composition.values()))
        if self.composition:
            if not total > 0 or any(v < 0 for v in self.composition.values()):
                raise ValueError("composition: non-negative amounts of the elements")
            if abs(total - 100.0) > 1e-9:  # normalized to 100 (once: read back as it is)
                self.composition = {
                    k: 100.0 * float(v) / total for k, v in self.composition.items()
                }
        self.phases = [p if isinstance(p, AlloyPhase) else AlloyPhase(**p) for p in self.phases]
        for p in self.phases:
            if p.role not in ROLES:
                raise ValueError(f"phase {p.name}: role one of {', '.join(ROLES)}")
            if p.cubic is None and (p.young is None or p.poisson is None):
                raise ValueError(f"phase {p.name}: its young and poisson, or its cubic constants")
            if p.cubic is not None:
                p.cubic = tuple(float(c) for c in p.cubic)
                if len(p.cubic) != 3:
                    raise ValueError(f"phase {p.name}: cubic, the three constants C11 C12 C44")
            if p.role in ("precipitates", "particles") and p.radius is None:
                raise ValueError(f"phase {p.name}: the radius of the {p.role}")
        matrices = [p for p in self.phases if p.role == "matrix"]
        if len(matrices) != 1:
            raise ValueError("phases: one matrix")
        given = sum(float(p.fraction) for p in self.phases if p.role != "matrix")
        if any(p.fraction is None or not 0 < p.fraction < 1 for p in self.phases
               if p.role != "matrix"):  # fmt: skip
            raise ValueError("phases: the fraction of each phase but the matrix, in (0, 1)")
        if not given < 1.0:
            raise ValueError("phases: their fractions leave no matrix")
        matrices[0].fraction = 1.0 - given

    def phase(self, role: str) -> list[AlloyPhase]:
        return [p for p in self.phases if p.role == role]

    @property
    def matrix(self) -> AlloyPhase:
        return self.phase("matrix")[0]

    def to_dict(self) -> dict:
        out = asdict(self)
        out["phases"] = [{k: v for k, v in asdict(p).items() if v is not None} for p in self.phases]
        return out


def read_alloy(source) -> Alloy:
    """An alloy from a mapping, or from a JSON or YAML file of it (see the module)."""
    if isinstance(source, Alloy):
        return source
    if isinstance(source, dict):
        data = dict(source)
    else:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
    unknown = set(data) - {f for f in Alloy.__dataclass_fields__}
    if unknown:
        raise ValueError(f"alloy: unknown keys {sorted(unknown)}")
    return Alloy(**data)


def write_alloy(path, alloy: Alloy) -> Path:
    """An alloy as a JSON file that :func:`read_alloy` reads back."""
    path = Path(path)
    path.write_text(json.dumps(alloy.to_dict(), indent=1), encoding="utf-8")
    return path


def _radius(value) -> dict | float:
    """The law of radii of the configurations: a number, or the normal law of a mean and a
    standard deviation truncated at three of them (a bounded law: the volume element must
    hold the largest particle, and its voxels resolve the mean one)."""
    if isinstance(value, dict):
        m, s = float(value["mean"]), float(value["std"])
        return {"type": "normal", "mean": m, "std": s, "low": max(0.0, m - 3 * s),
                "high": m + 3 * s}  # fmt: skip
    return float(value)


def _law(value):
    from morphogen.distributions import distribution_from_config

    return distribution_from_config(_radius(value))


def _mean_volume(value) -> float:
    return 4.0 / 3.0 * math.pi * _law(value).moment(3)


def _largest(radius) -> float:
    """The largest radius drawn from a law."""
    return float(_law(radius).upper())


def _spheres(radius, fraction: float) -> dict:
    """Hard spheres of a law of radii at a fraction (dense packings above 0.3)."""
    method = "lubachevsky_stillinger" if fraction > 0.3 else "rsa"
    return {"type": "hardcore_spheres", "radius": _radius(radius),
            "volume_fraction": float(fraction), "method": method}  # fmt: skip


def alloy_scales(alloy, *, matrix: dict | None = None) -> list[dict]:
    """The scales of an alloy as configurations (``name``, ``model``, ``domain``,
    ``materials``, ``phases``): the precipitates in the matrix (if it has any), then the
    grains with their phases and particles; ``matrix`` the material of the matrix of the
    grains (by default that of the matrix phase; the effective one of the precipitates
    when :func:`homogenize_alloy` computes them)."""
    alloy = read_alloy(alloy)
    out = []
    base = alloy.matrix
    fine = alloy.phase("precipitates")
    if fine:
        inside = sum(p.fraction for p in fine)
        local = inside / (base.fraction + inside)  # their fraction in the matrix
        volume = PRECIPITATES * sum(_mean_volume(p.radius) * p.fraction for p in fine) / inside
        edge = (volume / local) ** (1.0 / 3.0)
        edge = max(edge, 4.5 * max(_largest(p.radius) for p in fine))  # room for the largest
        materials = {0: base.material()}
        models = [_spheres(p.radius, local * p.fraction / inside) for p in fine]
        if len(fine) == 1:
            model = models[0]
            materials[1] = fine[0].material()
        else:
            model = {"type": "combination", "operation": "inclusions",
                     "models": [{"type": "tpms", "volume_fraction": 0.99}, *models]}  # fmt: skip
            materials = {0: base.material(), 1: base.material()}
            for k, p in enumerate(fine):
                materials[2 + k] = p.material()
        names = {0: base.name, **{1 + k: p.name for k, p in enumerate(fine)}}
        out.append({"name": "precipitates", "model": model,
                    "domain": {"size": edge, "periodic": True}, "materials": materials,
                    "phases": names})  # fmt: skip
    grains = alloy.phase("grains")
    coarse = alloy.phase("particles")
    cell_volume = math.pi / 6.0 * alloy.grain_size**3
    edge = (GRAINS * cell_volume) ** (1.0 / 3.0)
    if coarse:
        edge = max(edge, 4.5 * max(_largest(p.radius) for p in coarse))
    solid = 1.0 - sum(p.fraction for p in coarse)  # the grains, of the matrix and the others
    fractions = [base.fraction + sum(p.fraction for p in fine)] + [p.fraction for p in grains]
    fractions = [f / solid for f in fractions]
    crystals = any(p.cubic is not None for p in [base, *grains])
    polycrystal = {"type": "voronoi", "intensity": 1.0 / cell_volume,
                   "crystal_orientations": crystals, "phase_fractions": fractions}  # fmt: skip
    materials = {0: dict(matrix or base.material())}  # the cells of the matrix: phase 0
    names = {0: base.name}
    for k, p in enumerate(grains, start=1):
        materials[k], names[k] = p.material(), p.name
    model = polycrystal
    if coarse:
        first = len(materials)
        model = {"type": "combination", "operation": "inclusions",
                 "models": [polycrystal, *[_spheres(p.radius, p.fraction) for p in coarse]],
                 "inclusion_phases": list(range(first, first + len(coarse)))}  # fmt: skip
        for k, p in enumerate(coarse):
            materials[first + k], names[first + k] = p.material(), p.name
    out.append({"name": "grains", "model": model, "domain": {"size": edge, "periodic": True},
                "materials": materials, "phases": names})  # fmt: skip
    return out


def alloy_configuration(alloy, scale: str = "grains", *, shape: int = 32,
                        matrix: dict | None = None) -> dict:  # fmt: skip
    """The configuration of ``morphogen run`` of a scale of an alloy: its model and
    domain, ``shape`` voxels along an edge, the stiffness of the phases by FFT."""
    scales = {s["name"]: s for s in alloy_scales(alloy, matrix=matrix)}
    if scale not in scales:
        raise ValueError(f"scale: one of {', '.join(scales)}")
    s = scales[scale]
    return {
        "model": s["model"],
        "domain": s["domain"],
        "voxels": {"shape": int(shape)},
        "homogenization": {
            "method": "fft",
            "materials": {int(k): v for k, v in s["materials"].items()},
            "properties": ["stiffness"],
        },
        "outputs": {"directory": f"out/alloy_{scale}", "name": scale},
    }


def _isotropic(bulk: float, shear: float) -> dict:
    young = 9.0 * bulk * shear / (3.0 * bulk + shear)
    poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))
    return {"bulk_modulus": bulk, "shear_modulus": shear, "young": young, "poisson": poisson}


def homogenize_alloy(alloy, *, shape: int = 32, seed: int = 1) -> dict:
    """The effective elasticity of an alloy, scale by scale (FFT on ``shape`` voxels per
    edge, one realization of ``seed``): the stiffness tensor of each scale, its isotropic
    moduli (bulk, shear, Young's, Poisson's), the Hashin–Shtrikman bounds of the phases of
    the scale when they are isotropic, and at the end the departure of the effective
    Young's modulus from the one predicted (``properties["young"]``)."""
    from morphogen.config import load_config
    from morphogen.homogenization import homogenize
    from morphogen.homogenization.bounds import bounds
    from morphogen.seeding import derive_seed

    alloy = read_alloy(alloy)
    out: dict = {"alloy": alloy.alloy, "scales": {}}
    matrix = None
    for name in [s["name"] for s in alloy_scales(alloy)]:
        cfg = load_config(alloy_configuration(alloy, name, shape=shape, matrix=matrix))
        micro = cfg.model.generate(cfg.domain, derive_seed(seed, 0))
        image = micro.voxelize(**cfg.voxels.kwargs())
        materials = cfg.homogenization.materials
        grains = {}
        if any(getattr(m, "symmetry", "isotropic") != "isotropic" for m in materials.values()):
            grains = {"grains": micro.labels(grid=image.grid()), "orientations": micro.orientations}
        result = homogenize(image, materials, properties=["stiffness"], **grains)
        stiffness = result["stiffness.periodic"]
        entry = {
            "stiffness": np.asarray(stiffness["tensor"]).tolist(),
            **_isotropic(float(stiffness["bulk_modulus"]), float(stiffness["shear_modulus"])),
            "fractions": {int(k): float(np.mean(image.data == k)) for k in materials},
        }
        if all(getattr(m, "symmetry", "isotropic") == "isotropic" for m in materials.values()):
            entry["bounds"] = bounds(entry["fractions"], materials)
        out["scales"][name] = entry
        if name == "precipitates":  # the effective matrix of the grains
            matrix = {"young": entry["young"], "poisson": entry["poisson"]}
    last = out["scales"]["grains"]
    out["young"], out["poisson"] = last["young"], last["poisson"]
    predicted = alloy.properties.get("young")
    if predicted:
        out["predicted_young"] = float(predicted)
        out["departure"] = (last["young"] - float(predicted)) / float(predicted)
    return out
