"""MFront behaviours (TFEL, https://thelfer.github.io/tfel) in the solvers of morphoGen,
through MGIS (MFrontGenericInterfaceSupport).

A material whose ``mfront`` names a behaviour (:class:`MFrontBehaviour`: a library
compiled with the ``generic`` interface, or a ``.mfront`` source that morphoGen compiles
with ``mfront --obuild --interface=generic`` and caches) has its integration points
integrated by MGIS along loading paths, by finite elements and on voxels, in small and
finite strains: the stresses, the internal state variables and the consistent tangent of
every point, the rest of the solvers unchanged (the periodic element, Newton's method,
the outputs). Its elastic constants (``young`` and ``poisson``, or a crystal stiffness)
remain those of the reference medium of the FFT solver and of the linear problems.

Conventions. MGIS stores symmetric tensors in the order of TFEL, (11, 22, 33, 12, 13, 23)
(that of Cast3M, Code_Aster and Abaqus, not the usual order of Voigt, (11, 22, 33, 23,
13, 12)), with a factor sqrt(2) on their shear components (Mandel's normalization; the
tangents likewise), and the deformation gradient and the first Piola-Kirchhoff stress in
the order (11, 22, 33, 12, 21, 13, 31, 23, 32). morphoGen keeps the usual order (11, 22,
33, 23, 31, 12) with engineering shear strains, and tensors of 9 components row by row:
the components are converted at the calls of MGIS only. Finite strain
behaviours are loaded for the first Piola-Kirchhoff stress and its derivative with
respect to the deformation gradient. The gradients of the points of a crystal go to the
axes of its lattice (the orientation of its grain), the stresses and the tangents come
back to the axes of the sample. The tangents are symmetrized for the conjugate gradients
of the solvers whose laws all have symmetric tangents, and passed whole to GMRES
otherwise (and to Newton's method at a material point): a behaviour has a symmetric
tangent when it declares one (``@ProvidesSymmetricTangentOperator``, which MFront writes
in the traits of the behaviour, ``isConsistentTangentOperatorSymmetric``, in the header
that it generates next to the library: :func:`declared_symmetric`; the ``generic``
interface does not export it).

State. The internal state variables of MGIS live in morphoGen's arrays of the internal
variables of the points, after the change of temperature, the gradients and the
thermodynamic forces at the end of the step (the start of the next one); tensors of 9
components there hold their difference to the identity, so that the natural state is
zero. Every integration starts from them, so that divided steps and the iterations of
Newton's method need no special care. The cumulated plastic strain of the outputs is the
internal variable ``plastic_strain`` (by default ``EquivalentPlasticStrain`` or
``EquivalentViscoplasticStrain`` when the behaviour has one), and the plastic strain
tensor, in small strains, the strain minus ``ElasticStrain``.

The parameters of a behaviour (``parameters``) are global to its library in the process:
materials that share a behaviour share them.

MGIS and MFront are not on PyPI: ``conda install -c conda-forge mgis mfront`` (a C++
compiler, such as ``cxx-compiler``, builds the behaviours).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morphogen.homogenization.behaviours import MFRONT, STATE_P

INSTALL = "conda install -c conda-forge mgis mfront cxx-compiler"
_SQRT2 = math.sqrt(2.0)
#: Slots of the state of an MFront point: the plastic strain (tensor components) and the
#: cumulated plastic strain (STATE_P) of the outputs, the change of temperature, then
#: the gradients, the thermodynamic forces and the internal state variables of MGIS
_TEMP = STATE_P + 1
_FIRST = STATE_P + 2
#: MGIS's index of the component (i, j) of a symmetric tensor (TFEL's order: the shears
#: 12, 13, 23 after the diagonal), and the factor of that component (sqrt(2) off it)
_MGIS_INDEX = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])
_WEIGHT = np.where(np.eye(3, dtype=bool), 1.0, _SQRT2)
#: row-by-row index (3 i + j) of MGIS's components of tensors of 9 components
_ROWS = np.array([0, 4, 8, 1, 3, 2, 6, 5, 7])
#: morphoGen's Voigt pairs (11, 22, 33, 23, 31, 12)
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))


def _mgis():
    try:
        import mgis
        import mgis.behaviour
    except ImportError:
        raise ImportError(f"MFront behaviours need MGIS: {INSTALL}") from None
    return mgis


def available() -> bool:
    """Whether MGIS can be imported (MFront behaviours can be integrated)."""
    try:
        _mgis()
    except ImportError:
        return False
    return True


def info() -> dict:
    """MGIS and the MFront code generator (the version of TFEL), or why MFront behaviours
    are not available."""
    try:
        _mgis()
    except ImportError as exc:
        return {"mgis": False, "mfront": None, "tfel": None, "reason": str(exc)}
    exe = shutil.which("mfront")
    out = {"mgis": True, "mfront": exe, "tfel": None}
    if exe is not None:
        try:
            run = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return out
        for line in run.stdout.splitlines():
            if line.startswith("Version"):
                out["tfel"] = line.partition(":")[2].strip()
    return out


def _pairs(value, name: str) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        items = value.items()
    else:
        items = [tuple(v) for v in value]
    out = []
    for key, v in items:
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"mfront.{name}.{key}: a finite number")
        out.append((str(key), v))
    return tuple(sorted(out))


@dataclass(frozen=True)
class MFrontBehaviour:
    """An MFront behaviour of a material: ``library`` (compiled with the ``generic``
    interface) or ``source`` (a ``.mfront`` file, compiled on first use by
    :func:`compile_mfront`), ``behaviour`` (its name; by default the stem of the
    source), the values of its ``material_properties`` and ``parameters``, its
    ``external_state_variables`` other than the temperature (constants), the absolute
    ``temperature`` of the natural state (the changes of temperature of the loading paths
    add to it) and the internal variable that gives the cumulated plastic strain of the
    outputs (``plastic_strain``)."""

    library: str | None = None
    source: str | None = None
    behaviour: str | None = None
    material_properties: tuple = ()
    parameters: tuple = ()
    external_state_variables: tuple = ()
    temperature: float = 293.15
    plastic_strain: str | None = None

    def __post_init__(self) -> None:
        if (self.library is None) == (self.source is None):
            raise ValueError("mfront: give the library or the source of the behaviour")
        name = self.behaviour
        if name is None:
            if self.source is None:
                raise ValueError("mfront: the name of the behaviour of the library")
            name = Path(self.source).stem
        object.__setattr__(self, "behaviour", str(name))
        for key in ("library", "source"):
            if getattr(self, key) is not None:
                object.__setattr__(self, key, str(getattr(self, key)))
        object.__setattr__(self, "material_properties",
                           _pairs(self.material_properties, "material_properties"))  # fmt: skip
        object.__setattr__(self, "parameters", _pairs(self.parameters, "parameters"))
        esvs = _pairs(self.external_state_variables, "external_state_variables")
        object.__setattr__(self, "external_state_variables", esvs)
        t = float(self.temperature)
        if not (math.isfinite(t) and t > 0):
            raise ValueError("mfront.temperature: an absolute temperature (K)")
        object.__setattr__(self, "temperature", t)

    @classmethod
    def from_config(cls, spec) -> MFrontBehaviour:
        """A mapping with ``library`` or ``source``, and optionally ``behaviour``,
        ``material_properties``, ``parameters``, ``external_state_variables`` (mappings
        of names to values), ``temperature`` and ``plastic_strain``."""
        if isinstance(spec, MFrontBehaviour):
            return spec
        if isinstance(spec, str):
            spec = {"source" if spec.endswith(".mfront") else "library": spec}
        if not isinstance(spec, dict):
            raise ValueError("mfront: a mapping (library or source, behaviour, ...)")
        allowed = {"library", "source", "behaviour", "material_properties", "parameters",
                   "external_state_variables", "temperature", "plastic_strain"}  # fmt: skip
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError(f"mfront: unknown keys {unknown}; allowed: {sorted(allowed)}")
        return cls(**spec)

    def to_config(self) -> dict:
        out: dict = {"library": self.library} if self.library else {"source": self.source}
        out["behaviour"] = self.behaviour
        for key in ("material_properties", "parameters", "external_state_variables"):
            if getattr(self, key):
                out[key] = dict(getattr(self, key))
        out["temperature"] = self.temperature
        if self.plastic_strain is not None:
            out["plastic_strain"] = self.plastic_strain
        return out


# -- compilation -----------------------------------------------------------------------------


def _library_of(directory: Path) -> Path | None:
    for pattern in ("libBehaviour.so", "libBehaviour.dylib", "libBehaviour.dll", "Behaviour.dll"):
        found = sorted(directory.rglob(pattern))
        if found:
            return found[0]
    return None


def compile_mfront(source, *, directory=None) -> Path:
    """The library (``generic`` interface) of an MFront source, built by ``mfront
    --obuild --interface=generic`` in a cache directory named after its stem and the hash
    of its contents (``directory``; by default ``MORPHOGEN_MFRONT_CACHE`` or
    ``~/.cache/morphogen/mfront``), and reused while the source does not change."""
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"mfront: no source {source}")
    exe = shutil.which("mfront")
    if exe is None:
        raise RuntimeError(f"mfront: the MFront code generator is not on the PATH ({INSTALL})")
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    base = directory or os.environ.get("MORPHOGEN_MFRONT_CACHE")
    base = Path(base) if base else Path.home() / ".cache" / "morphogen" / "mfront"
    target = base / f"{source.stem}-{key}"
    library = _library_of(target) if target.exists() else None
    if library is not None:
        return library
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target / source.name)
    run = subprocess.run([exe, "--obuild", "--interface=generic", source.name], cwd=target,
                         capture_output=True, text=True, check=False)  # fmt: skip
    library = _library_of(target)
    if run.returncode != 0 or library is None:
        tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"mfront: {source.name} did not compile:\n{tail}")
    return library


# -- loading -------------------------------------------------------------------------------------


@dataclass
class MFrontLaw:
    """A behaviour loaded by MGIS for the integration points of a material: its sizes,
    the offsets of its internal state variables of 9 components (stored minus the
    identity) and of the variables of the outputs, whether it is orthotropic (its points
    are turned to the axes of their grains) and whether its consistent tangent is
    symmetric (:func:`declared_symmetric`)."""

    spec: MFrontBehaviour
    behaviour: object
    finite: bool
    n_grad: int
    n_force: int
    n_isv: int
    identity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    p_offset: int | None = None
    elastic_offset: int | None = None
    orthotropic: bool = False
    symmetric: bool = False

    @property
    def width(self) -> int:
        """Internal variables of a point in morphoGen's arrays."""
        return _FIRST + self.n_grad + self.n_force + self.n_isv


#: Parameters set on the behaviours of each library (the parameters are global)
_PARAMETERS: dict[tuple[str, str], tuple] = {}
#: The traits of the tangent of a behaviour in the header that MFront generates: its
#: class (the hypothesis of the specialization, or ``hypothesis`` for the general one)
#: and whether its tangent is symmetric
_TRAITS = r"class MechanicalBehaviourTraits<::tfel::material::{name}<([^,<>]+),[^>]*>\s*>"
_SYMMETRIC_TRAIT = re.compile(r"isConsistentTangentOperatorSymmetric\s*=\s*(true|false)")


def declared_symmetric(library, behaviour: str) -> bool | None:
    """Whether the behaviour ``behaviour`` of the MFront ``library`` declares a symmetric
    consistent tangent (``@ProvidesSymmetricTangentOperator``): the trait
    ``isConsistentTangentOperatorSymmetric`` of its tridimensional hypothesis in the header
    ``include/TFEL/Material/<behaviour>.hxx`` that ``mfront --obuild`` writes beside the
    directory of the library (that of :func:`compile_mfront`); None without that header."""
    library = Path(library).resolve()
    for root in (library.parent.parent, library.parent):
        header = root / "include" / "TFEL" / "Material" / f"{behaviour}.hxx"
        if header.is_file():
            break
    else:
        return None
    text = header.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(_TRAITS.format(name=re.escape(behaviour)))
    traits = {}
    matches = list(pattern.finditer(text))
    for k, match in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
        found = _SYMMETRIC_TRAIT.search(text, match.end(), end)
        if found:
            traits[match.group(1).strip()] = found.group(1) == "true"
    for hypothesis in ("ModellingHypothesis::TRIDIMENSIONAL", "hypothesis"):
        if hypothesis in traits:
            return traits[hypothesis]
    return None


def load_behaviour(spec: MFrontBehaviour, *, finite: bool) -> MFrontLaw:
    """The behaviour of ``spec`` loaded by MGIS for three-dimensional points (finite
    strains: for the first Piola-Kirchhoff stress and its derivative with respect to F),
    checked against the kinematics of the solver, its material properties and parameters
    set."""
    mgis = _mgis()
    bv = mgis.behaviour
    library = Path(spec.library) if spec.library else compile_mfront(spec.source)
    h = bv.Hypothesis.Tridimensional
    # each form of load refuses the behaviours of the other kinematics: the behaviour is
    # loaded as it is, its type checked against the solver below
    if bv.isStandardFiniteStrainBehaviour(str(library), spec.behaviour):
        opts = bv.FiniteStrainBehaviourOptions()
        opts.stress_measure = bv.FiniteStrainBehaviourOptionsStressMeasure.PK1
        opts.tangent_operator = bv.FiniteStrainBehaviourOptionsTangentOperator.DPK1_DF
        b = bv.load(opts, str(library), spec.behaviour, h)
    else:
        b = bv.load(str(library), spec.behaviour, h)
    standard = (bv.BehaviourType.STANDARDFINITESTRAINBEHAVIOUR if finite
                else bv.BehaviourType.STANDARDSTRAINBASEDBEHAVIOUR)  # fmt: skip
    if b.btype != standard:
        kind = "finite" if finite else "small"
        raise ValueError(
            f"mfront: {spec.behaviour} is a {b.btype.name} behaviour; the loading paths in "
            f"{kind} strains need a {standard.name} behaviour"
        )
    n_grad = bv.getArraySize(b.gradients, h)
    n_force = bv.getArraySize(b.thermodynamic_forces, h)
    if (n_grad, n_force) != ((9, 9) if finite else (6, 6)):
        raise ValueError(f"mfront: {spec.behaviour}: unexpected gradients ({n_grad})")
    given = dict(spec.material_properties)
    names = [v.name for v in b.mps]
    missing = sorted(set(names) - set(given))
    extra = sorted(set(given) - set(names))
    if missing or extra:
        raise ValueError(f"mfront: {spec.behaviour} has the material properties {names}; "
                         f"missing {missing}, unknown {extra}")  # fmt: skip
    params = dict(spec.parameters)
    unknown = sorted(set(params) - set(b.params))
    if unknown:
        raise ValueError(f"mfront: {spec.behaviour} has no parameters {unknown}; {b.params}")
    key = (str(library.resolve()), spec.behaviour)
    previous = _PARAMETERS.get(key)
    if previous is not None and previous != spec.parameters:
        raise ValueError(
            f"mfront: {spec.behaviour} of {library.name} with other parameters: the parameters "
            "of a behaviour are global to its library"
        )
    for name, value in params.items():
        b.setParameter(name, value)
    _PARAMETERS[key] = spec.parameters
    esvs = [v.name for v in b.esvs]
    other = dict(spec.external_state_variables)
    lacking = sorted(set(esvs) - {"Temperature"} - set(other))
    if lacking:
        raise ValueError(f"mfront: give the external state variables {lacking}")
    n_isv = bv.getArraySize(b.isvs, h)
    identity = []
    for v in b.isvs:
        if bv.getVariableSize(v, h) == 9 and v.type == bv.VariableType.Tensor:
            offset = bv.getVariableOffset(b.isvs, v.name, h)
            identity.extend(offset + k for k in (0, 1, 2))
    p_name = spec.plastic_strain
    isv_names = {v.name: v for v in b.isvs}
    if p_name is None:
        p_name = next((n for n in ("EquivalentPlasticStrain", "EquivalentViscoplasticStrain")
                       if n in isv_names), None)  # fmt: skip
    elif p_name not in isv_names:
        raise ValueError(f"mfront: {spec.behaviour} has no internal variable {p_name!r}")
    p_offset = None if p_name is None else bv.getVariableOffset(b.isvs, p_name, h)
    elastic = None
    if not finite and "ElasticStrain" in isv_names:
        elastic = bv.getVariableOffset(b.isvs, "ElasticStrain", h)
    orthotropic = b.symmetry == bv.BehaviourSymmetry.ORTHOTROPIC
    symmetric = bool(declared_symmetric(library, spec.behaviour))  # unknown: not symmetric
    return MFrontLaw(spec, b, finite, n_grad, n_force, n_isv, np.array(identity, dtype=int),
                     p_offset, elastic, orthotropic, symmetric)  # fmt: skip


# -- conventions -----------------------------------------------------------------------------


def _tensors(e) -> np.ndarray:
    """Symmetric tensors (n, 3, 3) of engineering Voigt strains (n, 6)."""
    t = np.empty((e.shape[0], 3, 3))
    for k, (i, j) in enumerate(_PAIRS):
        v = e[:, k] if i == j else 0.5 * e[:, k]
        t[:, i, j] = v
        t[:, j, i] = v
    return t


def _to_mgis(t) -> np.ndarray:
    """MGIS's components (n, 6) of symmetric tensors (n, 3, 3): 11, 22, 33, then
    sqrt(2) times 12, 13, 23."""
    out = np.empty((t.shape[0], 6))
    out[:, 0], out[:, 1], out[:, 2] = t[:, 0, 0], t[:, 1, 1], t[:, 2, 2]
    out[:, 3] = _SQRT2 * t[:, 0, 1]
    out[:, 4] = _SQRT2 * t[:, 0, 2]
    out[:, 5] = _SQRT2 * t[:, 1, 2]
    return out


def _from_mgis(m) -> np.ndarray:
    """Symmetric tensors (n, 3, 3) of MGIS's components (n, 6)."""
    return m[:, _MGIS_INDEX] / _WEIGHT


def _voigt(t) -> np.ndarray:
    """morphoGen's Voigt components (n, 6) of symmetric tensors (tensor shears)."""
    return np.stack([t[:, i, j] for i, j in _PAIRS], axis=1)


def _turn(t, g, back: bool):
    """Tensors (n, 3, 3) in the axes of the lattice (``g t g^T``) or back (``g^T t g``)."""
    if g is None:
        return t
    if back:
        return np.einsum("npi,npq,nqj->nij", g, t, g)
    return np.einsum("nip,npq,njq->nij", g, t, g)


def _turn4(a, g):
    """Fourth-order tensors (n, 3, 3, 3, 3) from the axes of the lattice to the sample."""
    if g is None:
        return a
    return np.einsum("npi,nqj,nrk,nsl,npqrs->nijkl", g, g, g, g, a, optimize=True)


def _strains_in(e, g) -> np.ndarray:
    return _to_mgis(_turn(_tensors(e), g, back=False))


def _gradients_in(hg, g) -> np.ndarray:
    f = np.eye(3) + hg.reshape(-1, 3, 3)
    return _turn(f, g, back=False).reshape(-1, 9)[:, _ROWS]


def _stress_out(m, g) -> np.ndarray:
    return _voigt(_turn(_from_mgis(m), g, back=True))


def _pk1_out(p, g) -> np.ndarray:
    rows = np.empty_like(p)
    rows[:, _ROWS] = p
    return _turn(rows.reshape(-1, 3, 3), g, back=True).reshape(-1, 9)


def _tangent_out(k, g, symmetric: bool = True) -> np.ndarray:
    """morphoGen's Voigt tangent (engineering strains, n x 6 x 6) of MGIS's (TFEL's order,
    factors sqrt(2)), symmetrized with ``symmetric``."""
    from morphogen.homogenization.materials import tensor4_to_voigt

    c = k[:, _MGIS_INDEX[:, :, None, None], _MGIS_INDEX[None, None, :, :]]
    c = c / (_WEIGHT[:, :, None, None] * _WEIGHT[None, None, :, :])
    d = tensor4_to_voigt(_turn4(c, g))
    return 0.5 * (d + np.swapaxes(d, 1, 2)) if symmetric else np.ascontiguousarray(d)


def _dpdf_out(k, g, symmetric: bool = True) -> np.ndarray:
    """dP/dF row by row (n x 9 x 9) of MGIS's (its order of the components), symmetrized
    with ``symmetric``."""
    a = np.empty_like(k)
    a[:, _ROWS[:, None], _ROWS[None, :]] = k
    if g is not None:
        a = _turn4(a.reshape(-1, 3, 3, 3, 3), g).reshape(-1, 9, 9)
    return 0.5 * (a + np.swapaxes(a, 1, 2)) if symmetric else a


# -- integration ---------------------------------------------------------------------------------


class MFrontPoints:
    """The integration points of the materials with an MFront behaviour, integrated by
    MGIS: one manager of the data of the points per material, the rotations of the points
    of crystals, a pool of threads. ``kind``, ``pmat`` and ``point_mat`` are those of the
    points of a solver, ``rotations`` the orientation matrix of each of its materials
    (``crystal = g @ sample``); ``symmetric``: the tangents symmetrized (for the conjugate
    gradients), or passed whole (for GMRES and at a material point)."""

    def __init__(self, laws, kind, pmat, point_mat, rotations, finite: bool, threads=None,
                 symmetric: bool = True):  # fmt: skip
        mgis = _mgis()
        bv = mgis.behaviour
        import numba

        self.finite = bool(finite)
        self.symmetric = bool(symmetric)
        self.points = np.flatnonzero(np.asarray(kind) == MFRONT)
        self.groups = []
        pmat = np.asarray(pmat)[self.points]
        point_mat = np.asarray(point_mat)[self.points]
        for c, law in enumerate(laws):
            if law.finite != self.finite:
                raise ValueError("mfront: the behaviour and the solver have other kinematics")
            local = np.flatnonzero(pmat == c)
            if local.size == 0:
                continue
            m = bv.MaterialDataManager(law.behaviour, local.size)
            for state in (m.s0, m.s1):
                for name, value in law.spec.material_properties:
                    bv.setMaterialProperty(state, name, value)
                for name, value in law.spec.external_state_variables:
                    bv.setExternalStateVariable(state, name, value)
            # orthotropic behaviours (crystals) in the axes of their grains; isotropic ones
            # are not turned
            g = np.asarray(rotations)[point_mat[local]]
            if not law.orthotropic or np.allclose(g, np.eye(3), rtol=0, atol=1e-14):
                g = None
            self.groups.append((law, local, m, g))
        self.pool = mgis.ThreadPool(int(threads or numba.get_num_threads()))
        self.integration = bv.IntegrationType.IntegrationWithConsistentTangentOperator
        self._bv = bv

    def integrate(self, grads, dtemp, dt, old, new, stress, tangent) -> int:
        """Integration of the MFront points of the arrays of all the points (see
        :meth:`~morphogen.homogenization.nonlinear._PointArrays.integrate`); returns the
        number of points of the materials whose integration failed."""
        idx = self.points
        out = [new[idx], stress[idx], tangent[idx]]
        failed = self.integrate_compact(grads[idx], dtemp, dt, old[idx], *out)
        new[idx], stress[idx], tangent[idx] = out
        return failed

    def integrate_compact(self, grads, dtemp, dt, old, new, stress, tangent) -> int:
        """The same for the arrays of the MFront points only (in the order of
        :attr:`points`), on the CPU."""
        bv = self._bv
        failed = 0
        for law, local, m, g in self.groups:
            o = old[local]
            a, b = _FIRST, _FIRST + law.n_grad
            c, d = b + law.n_force, b + law.n_force + law.n_isv
            start = o[:, a:b].copy()
            if self.finite:
                start[:, :3] += 1.0
            m.s0.gradients[:] = start
            m.s0.thermodynamic_forces[:] = o[:, b:c]
            isv = o[:, c:d].copy()
            if law.identity.size:
                isv[:, law.identity] += 1.0
            if law.n_isv:  # MGIS divides by the size in the views of the arrays
                m.s0.internal_state_variables[:] = isv
            t_ref = law.spec.temperature
            bv.setExternalStateVariable(m.s0, "Temperature", t_ref + float(o[0, _TEMP]))
            bv.setExternalStateVariable(m.s1, "Temperature", t_ref + float(dtemp))
            gi = _gradients_in(grads[local], g) if self.finite else _strains_in(grads[local], g)
            m.s1.gradients[:] = gi
            if bv.integrate(self.pool, m, self.integration, float(dt)) < 0:
                failed += local.size
                continue
            forces = np.array(m.s1.thermodynamic_forces)
            k = np.array(m.K).reshape(local.size, law.n_force, law.n_grad)
            if self.finite:
                stress[local] = _pk1_out(forces, g)
                tangent[local] = _dpdf_out(k, g, self.symmetric)
            else:
                stress[local] = _stress_out(forces, g)
                tangent[local] = _tangent_out(k, g, self.symmetric)
            if law.n_isv:
                isv = np.array(m.s1.internal_state_variables)
            state = np.zeros((local.size, new.shape[1]))
            if law.elastic_offset is not None:
                ee = _turn(_from_mgis(isv[:, law.elastic_offset : law.elastic_offset + 6]), g,
                           back=True)  # fmt: skip
                state[:, :6] = _voigt(_tensors(grads[local]) - ee)
            if law.p_offset is not None:
                state[:, STATE_P] = isv[:, law.p_offset]
            state[:, _TEMP] = dtemp
            state[:, a:b] = gi
            if self.finite:
                state[:, a : a + 3] -= 1.0
            state[:, b:c] = forces
            if law.identity.size:
                isv[:, law.identity] -= 1.0
            state[:, c:d] = isv
            new[local] = state
        return failed
