"""Material files of Z-mat (the material library of Z-set: ARMINES, ONERA, Transvalor) read
into morphoGen's materials, and what of that library morphoGen covers (:data:`COVERAGE`).

A material file of Z-mat assembles a behaviour from bricks (Besson, Cailletaud, Chaboche &
Forest, *Non-linear mechanics of materials*, 2010)::

    ***behavior gen_evp
     **elasticity isotropic
        young 200000.
        poisson 0.3
     **thermal_strain isotropic
        alpha 1.2e-5
     **potential gen_evp ev
       *criterion mises
       *flow norton
          n 7.
          K 400.
       *kinematic nonlinear
          C 15000.
          D 300.
       *isotropic constant
          R0 130.
    ***return

:func:`write_zmat` writes the elastic, von Mises and porous materials of morphoGen as such
files (a behaviour identified on the response of a morphology, for Z-set).

:func:`read_zmat` reads the behaviours ``gen_evp`` and ``reduced_plastic`` (an elasticity,
a thermal strain, potentials with their criterion, flow, isotropic and kinematic
hardening), ``porous_plastic`` (porous potentials: a porous criterion, a flow, an isotropic
hardening, a nucleation; coefficients that depend on the porosity as tables, ``fs = f``;
the initial porosity from ``*initialize_variable``) and ``linear_elastic``; comments start
with ``%``. :func:`zmat_material` turns one into a
:class:`~morphogen.homogenization.materials.Material`:

* integrated by morphoGen's own kernels (finite elements, FFT, GPU), as
  :class:`~morphogen.homogenization.behaviours.Plasticity`, when it is one potential with
  the von Mises criterion, a Norton flow or time-independent plasticity, an isotropic
  hardening ``constant``, ``linear``, ``nonlinear``, ``linear_nonlinear`` or
  ``nonlinear_sum`` (three saturating terms at most, without softening) and kinematic
  hardening ``linear`` (Prager) or ``nonlinear`` (Armstrong-Frederick, four at most), on
  an isotropic elasticity (elastic behaviours: any elasticity);
* ``porous_plastic`` by morphoGen's kernels (on the CPU), as
  :class:`~morphogen.homogenization.porous.PorousPlasticity`: one porous potential with
  the criterion ``gurson`` (``q1``, ``q2``, ``q3 = q1²`` as in Z-mat, the effective
  porosity ``fs``: the porosity, or a table of it, Tvergaard and Needleman's coalescence),
  ``rousselier``, ``modified_rousselier`` or ``elliptic`` (``C``, ``F``: numbers or tables
  of the porosity; Abouaf's compaction), a Norton or rate-independent flow, the isotropic
  hardenings of gen_evp and the nucleation ``constant``, ``gaussian`` or ``exponential``
  (with its threshold ``yield``), the kinematic hardening of gen_evp (``*kinematic linear``
  and ``nonlinear``, four at most) with the criteria ``gurson`` and ``elliptic``, on an
  isotropic elasticity; the modifiers of finite
  strains (``lagrange_rotate_no_J``...) give way to morphoGen's logarithmic strains, chosen
  by the solver;
* otherwise, as an MFront behaviour (:func:`zmat_mfront_source`, the brick
  ``StandardElastoViscoPlasticity``: Hill's criterion, the flows ``hyperbolic``,
  ``double_norton``, ``norton_exp``, ``strain_hardening`` (as user-defined
  viscoplastic flows), ``power_law`` and tabulated isotropic hardening, softening,
  ``nonlinear_evrad`` (Burlet-Cailletaud), several potentials), written in a cache folder
  and compiled by MFront, integrated by MGIS (:mod:`morphogen.homogenization.mfront`).

Conventions: the tensors of Z-set have the components (11, 22, 33, 12, 23, 31), so that the
coefficients ``hill1`` ... ``hill6`` of its Hill criterion, ``sqrt(3/2 s : M : s)`` with
``M`` diagonal on the deviator ``s`` (von Mises when they are all 1), become those of TFEL,
whose Hill stress is ``F (s11 - s22)^2 + G (s22 - s33)^2 + H (s33 - s11)^2 + 2 L s12^2 + 2 M
s13^2 + 2 N s23^2`` (not the order of Hill's paper): ``L = 3/2 hill4``, ``M = 3/2 hill6``,
``N = 3/2 hill5``, and ``F``, ``G``, ``H`` giving the same normal terms (checked by the yield
stresses in tension and in each shear); the reference temperature of the thermal strain is
morphoGen's natural
state (the loading paths give changes of temperature). Coefficients depending on the
temperature or other variables, interactions between potentials, damage and the other
models are not translated: :data:`COVERAGE` lists them with what morphoGen offers instead.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morphogen._version import __version__

#: Behaviours of Z-mat read by :func:`read_zmat`
BEHAVIOURS = ("gen_evp", "reduced_plastic", "linear_elastic", "porous_plastic")
#: Options of porous_plastic that change nothing in morphoGen (outputs and numerics)
_POROUS_HARMLESS = {"no_c_trick", "save_d", "save_l", "bifurcation_d", "bifurcation_l",
                    "perturbation", "additional_var_aux"}  # fmt: skip
#: Options of porous_plastic that morphoGen does not translate
_POROUS_OPTIONS = {"hf", "hp", "broken_behavior", "adiabatic_heating", "coefficient"}


@dataclass
class ZmatPotential:
    """A potential of a ``gen_evp`` behaviour, or a porous potential of a
    ``porous_plastic`` one: its name and bricks ``(type, coefficients)``: criterion (or
    porous criterion), flow, isotropic and kinematic hardening (lists), and for porous
    potentials the nucleation ``(type, coefficients, crack_like)`` and the shear anisotropy."""

    name: str
    criterion: tuple = ("mises", {})
    flow: tuple = ("plasticity", {})
    isotropic: list = field(default_factory=list)
    kinematic: list = field(default_factory=list)
    porous: bool = False
    nucleation: tuple | None = None
    shear_anisotropy: tuple | None = None


@dataclass
class ZmatBehaviour:
    """A behaviour of a material file of Z-mat: its type, elasticity and thermal strain
    ``(type, coefficients)`` and potentials; its modifier (``lagrange_rotate``...), the
    values of ``*initialize_variable`` and the other options (``**name`` -> lines)."""

    kind: str
    elasticity: tuple
    thermal_strain: tuple | None = None
    potentials: list = field(default_factory=list)
    modifier: str = ""
    initial: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)


def _number(token: str, where: str) -> float:
    try:
        return float(token.replace("D", "e").replace("d", "e"))
    except ValueError:
        raise ValueError(
            f"zmat: {where}: {token!r} is not a number (coefficients depending on the "
            "temperature, tables and functions are not translated)"
        ) from None


def _coefficients(lines: list[list[str]], where: str) -> dict:
    """``name value`` pairs (several per line allowed)."""
    out: dict = {}
    for tokens in lines:
        if len(tokens) % 2:
            raise ValueError(f"zmat: {where}: coefficients come as 'name value' pairs: {tokens}")
        for name, value in zip(tokens[0::2], tokens[1::2], strict=True):
            out[name] = _number(value, f"{where}.{name}")
    return out


def _is_number(token: str) -> bool:
    try:
        float(token.replace("D", "e").replace("d", "e"))
    except ValueError:
        return False
    return True


def _porous_coefficients(lines: list[list[str]], where: str) -> dict:
    """The coefficients of a brick of a porous potential: ``name value`` pairs, ``name =
    expression`` (kept as ``("=", expression)``) and tables, ``name variable`` followed by
    rows ``value variable_value`` (kept as ``("table", variable, ((x, value)...))``)."""
    out: dict = {}
    k = 0
    while k < len(lines):
        tokens = lines[k]
        text = " ".join(tokens)
        if "=" in text:
            name, _, expression = text.partition("=")
            out[name.strip()] = ("=", expression.strip().rstrip(";").strip())
            k += 1
            continue
        if len(tokens) == 2 and not _is_number(tokens[1]):
            name, variable = tokens
            rows = []
            k += 1
            while k < len(lines) and len(lines[k]) == 2 and all(map(_is_number, lines[k])):
                value, at = (_number(t, f"{where}.{name}") for t in lines[k])
                rows.append((at, value))
                k += 1
            if not rows:
                raise ValueError(f"zmat: {where}: the table of {name} has no rows")
            out[name] = ("table", variable, tuple(sorted(rows)))
            continue
        out.update(_coefficients([tokens], where))
        k += 1
    return out


def read_zmat(source) -> list[ZmatBehaviour]:
    """The behaviours of a material file of Z-mat (a path or its text): see the module."""
    text = (
        Path(source).read_text(encoding="utf-8", errors="replace")
        if (
            isinstance(source, Path)
            or (isinstance(source, str) and "\n" not in source and Path(source).is_file())
        )
        else str(source)
    )
    rows = []
    for raw in text.splitlines():
        line = raw.split("%", 1)[0].strip()
        if line:
            rows.append(line.split())
    out: list[ZmatBehaviour] = []
    behaviour = None
    section = None  # (level, keyword, type, name, lines)
    current_potential = None
    initial: dict = {}
    initializing = False

    def close_section():
        nonlocal section
        if section is None:
            return
        level, key, kind, _, lines = section
        where = f"{'*' * level}{key} {kind}".strip()
        if level == 2 and key == "elasticity":
            behaviour.elasticity = (kind or "isotropic", _coefficients(lines, where))
        elif level == 2 and key == "thermal_strain":
            behaviour.thermal_strain = (kind or "isotropic", _coefficients(lines, where))
        elif level == 2 and key == "option":
            behaviour.options[kind] = lines
        elif level == 1 and current_potential is not None and current_potential.porous:
            coefficients = _porous_coefficients(lines, where)
            if key == "porous_criterion":
                current_potential.criterion = (kind, coefficients)
            elif key == "flow":
                current_potential.flow = (kind or "plasticity", coefficients)
            elif key in ("isotropic_hardening", "isotropic"):
                current_potential.isotropic.append((kind or "constant", coefficients))
            elif key == "kinematic":
                current_potential.kinematic.append((kind, coefficients))
            elif key == "strain_nucleation":
                crack_like = kind.startswith("crack_like")
                current_potential.nucleation = (kind.removeprefix("crack_like").strip(),
                                                coefficients, crack_like)  # fmt: skip
            elif key == "shear_anisotropy":
                current_potential.shear_anisotropy = (kind, coefficients)
        elif level == 1:
            coefficients = _coefficients(lines, where)
            if key == "criterion":
                current_potential.criterion = (kind or "mises", coefficients)
            elif key == "flow":
                current_potential.flow = (kind or "plasticity", coefficients)
            elif key == "isotropic":
                current_potential.isotropic.append((kind or "constant", coefficients))
            elif key == "kinematic":
                current_potential.kinematic.append((kind or "nonlinear", coefficients))
        elif lines:
            raise ValueError(f"zmat: unexpected coefficients after {where}: {lines[0]}")
        section = None

    for tokens in rows:
        head = tokens[0]
        if head.lower() in ("*initialize_variable", "***initialize_variable"):
            initializing = True
            tokens = tokens[1:]
            if not tokens:
                continue
            head = tokens[0]
        if initializing:
            if head.startswith("*"):
                initializing = False
            else:
                initial.update(_coefficients([tokens], "*initialize_variable"))
                continue
        if head.startswith("***"):
            close_section()
            key = head[3:].lower()
            if key == "behavior" or key == "behaviour":
                kind = "_".join(tokens[1:2]).lower()
                if kind not in BEHAVIOURS:
                    raise ValueError(f"zmat: behaviour {kind!r}: one of {list(BEHAVIOURS)}")
                behaviour = ZmatBehaviour(kind, ("isotropic", {}))
                behaviour.modifier = tokens[2].lower() if len(tokens) > 2 else ""
                current_potential = None
            elif key == "return":
                if behaviour is not None:
                    out.append(behaviour)
                behaviour = None
            continue  # ***material and the other commands of the file
        if behaviour is None:
            continue
        if head.startswith("**") and not head.startswith("***"):
            close_section()
            key = head[2:].lower()
            kind = tokens[1].lower() if len(tokens) > 1 else ""
            if key == "potential":
                if kind not in ("gen_evp", "gen_evp2", ""):
                    raise ValueError(f"zmat: potential {kind!r}: only gen_evp potentials")
                name = tokens[2] if len(tokens) > 2 else f"p{len(behaviour.potentials) + 1}"
                current_potential = ZmatPotential(name)
                behaviour.potentials.append(current_potential)
                section = (2, key, kind, name, [])
            elif key == "porous_potential" and behaviour.kind == "porous_plastic":
                name = tokens[1] if len(tokens) > 1 else f"p{len(behaviour.potentials) + 1}"
                current_potential = ZmatPotential(name, criterion=("", {}), porous=True)
                behaviour.potentials.append(current_potential)
                section = (2, key, "", name, [])
            elif behaviour.kind == "porous_plastic" and key.lower() in (
                _POROUS_HARMLESS | _POROUS_OPTIONS
            ):
                current_potential = None
                section = (2, "option", key.lower(), "", [tokens[1:]] if len(tokens) > 1 else [])
            elif key in ("elasticity", "thermal_strain"):
                section = (2, key, kind, "", [])
            elif key in ("global_output", "verbose"):
                section = (2, key, kind, "", [])
            else:
                raise ValueError(
                    f"zmat: **{key} is not translated (elasticity, thermal_strain and "
                    "potentials only: interactions, damage, localization... are not)"
                )
            continue
        if head.startswith("*"):
            close_section()
            key = head[1:].lower()
            if key in ("integration", "initialize_variable"):
                section = None  # options of the integration of Z-set: skipped
                continue
            porous = current_potential is not None and current_potential.porous
            bricks = (
                ("porous_criterion", "flow", "isotropic_hardening", "isotropic", "kinematic",
                 "strain_nucleation", "shear_anisotropy")
                if porous
                else ("criterion", "flow", "isotropic", "kinematic")
            )  # fmt: skip
            if current_potential is None or key not in bricks:
                raise ValueError(f"zmat: *{key} outside a potential or not translated")
            kind = tokens[1].lower() if len(tokens) > 1 else ""
            extra = tokens[2:]
            lines = []
            if porous:
                if key == "strain_nucleation" and kind == "crack_like" and extra:
                    kind, extra = f"crack_like {extra[0].lower()}", extra[1:]
                # coefficients on the line of the brick (``*flow norton K .01 n 5.``)
                if len(extra) >= 2 and len(extra) % 2 == 0 and all(map(_is_number, extra[1::2])):
                    lines.append(extra)
            section = (1, key, kind, "", lines)
            continue
        if section is not None:
            section[4].append(tokens)
    close_section()
    if behaviour is not None:
        out.append(behaviour)
    if not out:
        raise ValueError("zmat: no ***behavior in the file")
    for b in out:
        b.initial = dict(initial)
    return out


# -- translation -------------------------------------------------------------------------


def _get(coefficients: dict, where: str, *names, default=None):
    """The first of ``names`` among the coefficients (aliases), or the default."""
    for name in names:
        if name in coefficients:
            return coefficients[name]
    if default is None:
        raise ValueError(f"zmat: {where}: missing coefficient {names[0]}")
    return default


def _check_known(coefficients: dict, where: str, known: set) -> None:
    unknown = sorted(set(coefficients) - known)
    if unknown:
        raise ValueError(f"zmat: {where}: unknown coefficients {unknown}; known: {sorted(known)}")


def _elastic_material_kwargs(behaviour: ZmatBehaviour) -> dict:
    """The elastic part of a material: ``young``/``poisson``, or a stiffness."""
    kind, c = behaviour.elasticity
    where = f"**elasticity {kind}"
    if kind == "isotropic":
        _check_known(c, where, {"young", "poisson", "E", "nu"})
        return {"young": _get(c, where, "young", "E"), "poisson": _get(c, where, "poisson", "nu")}
    if kind == "cubic":
        _check_known(c, where, {"y1111", "y1122", "y1212", "c11", "c12", "c44"})
        values = (
            _get(c, where, "y1111", "c11"),
            _get(c, where, "y1122", "c12"),
            _get(c, where, "y1212", "c44"),
        )
        return {"cubic": values}
    if kind in ("orthotropic", "transverse", "anisotropic"):
        return {"stiffness": _stiffness(kind, c, where)}
    raise ValueError(f"zmat: {where}: isotropic, cubic, orthotropic, transverse or anisotropic")


def _stiffness(kind: str, c: dict, where: str) -> np.ndarray:
    """The 6 x 6 stiffness (Voigt order 11, 22, 33, 23, 31, 12) of an anisotropic
    elasticity of Z-mat (``y<ijkl>`` coefficients, or the engineering constants of a
    transversely isotropic material, its axis 1)."""
    if kind == "transverse":
        _check_known(c, where, {"El", "Et", "Glt", "nult", "nutt"})
        el, et, glt = c["El"], c["Et"], c["Glt"]
        nult, nutt = c["nult"], c["nutt"]
        s = np.zeros((6, 6))
        s[0, 0], s[1, 1], s[2, 2] = 1 / el, 1 / et, 1 / et
        s[0, 1] = s[1, 0] = s[0, 2] = s[2, 0] = -nult / el
        s[1, 2] = s[2, 1] = -nutt / et
        s[3, 3] = 2 * (1 + nutt) / et  # the plane 23 is isotropic
        s[4, 4] = s[5, 5] = 1 / glt
        return np.linalg.inv(s)
    index = {"11": 0, "22": 1, "33": 2, "23": 3, "32": 3, "31": 4, "13": 4, "12": 5, "21": 5}
    out = np.zeros((6, 6))
    for name, value in c.items():
        m = re.fullmatch(r"y(\d\d)(\d\d)", name)
        if m is None or m.group(1) not in index or m.group(2) not in index:
            raise ValueError(
                f"zmat: {where}: coefficient {name!r}: give the components y1111, y1122... "
                "(the numbering c44, c55, c66 depends on the order of the components)"
            )
        i, j = index[m.group(1)], index[m.group(2)]
        out[i, j] = out[j, i] = value
    if kind == "orthotropic" and np.count_nonzero(out[:3, 3:]) + np.count_nonzero(
        out[3:, 3:] - np.diag(np.diag(out[3:, 3:]))
    ):
        raise ValueError(f"zmat: {where}: an orthotropic stiffness couples no shear")
    return out


def _native(behaviour: ZmatBehaviour):
    """The :class:`Plasticity` of a behaviour that morphoGen's kernels integrate, or
    ``None`` (with the reason) when it needs MFront."""
    from morphogen.homogenization.behaviours import MAX_ISOTROPIC, MAX_KINEMATIC, Plasticity

    if len(behaviour.potentials) != 1:
        return None, "several potentials"
    if behaviour.elasticity[0] != "isotropic":
        return None, "anisotropic elasticity"
    p = behaviour.potentials[0]
    if p.criterion[0] not in ("mises", "hill"):
        return None, f"criterion {p.criterion[0]}"
    hill = None
    if p.criterion[0] == "hill":
        c = p.criterion[1]
        if any(isinstance(v, tuple) for v in c.values()):
            return None, "coefficients of Hill's criterion depending on variables"
        # Z-set's hill1 ... hill6: the components 11, 22, 33, 12, 23, 31, morphoGen's order
        hill = tuple(_get(c, "*criterion hill", f"hill{i}", default=1.0) for i in range(1, 7))
    flow = _flow(p)
    if isinstance(flow, str):
        return None, flow
    k, n = flow
    hardening = _isotropic_hardening(p, MAX_ISOTROPIC)
    if isinstance(hardening, str):
        return None, hardening
    r0, h, terms = hardening
    kinematic = []
    for kind, c in p.kinematic:
        where = f"*kinematic {kind}"
        if kind == "linear":
            kinematic.append((_get(c, where, "C"), 0.0))
        elif kind == "nonlinear":
            kinematic.append((_get(c, where, "C"), _get(c, where, "D")))
        else:
            return None, f"kinematic hardening {kind}"
    if len(kinematic) > MAX_KINEMATIC:
        return None, "more than four kinematic terms"
    return (
        Plasticity(r0, h, tuple(terms), tuple(kinematic), viscosity=k, exponent=n, hill=hill),
        "",
    )


def _flow(p: ZmatPotential):
    """``(K, n)`` of a Norton flow, ``(None, None)`` for plasticity, or the reason why the
    flow is not native."""
    flow, fc = p.flow
    if flow == "plasticity":
        return None, None
    if flow == "norton":
        if any(isinstance(v, tuple) for v in fc.values()):
            return "coefficients of the flow depending on variables"
        k, n = _get(fc, "*flow norton", "K"), _get(fc, "*flow norton", "n")
        return (k, n) if n >= 1.0 else "a Norton exponent below 1"
    return f"flow {flow}"


def _isotropic_hardening(p: ZmatPotential, most: int):
    """``(R0, H, ((Q, b)...))`` of the isotropic hardening of a potential, or the reason
    why it is not native."""
    r0, h, terms = 0.0, 0.0, []
    if len(p.isotropic) > 1:
        return "several isotropic hardenings"
    for kind, c in p.isotropic:
        where = f"*isotropic {kind}"
        if any(isinstance(v, tuple) for v in c.values()):
            return "coefficients of the hardening depending on variables"
        if kind == "constant":
            r0 = _get(c, where, "R0")
        elif kind in ("linear", "iso_linear"):
            r0, h = _get(c, where, "R0"), _get(c, where, "H")
        elif kind == "nonlinear":
            r0 = _get(c, where, "R0")
            terms = [(_get(c, where, "Q"), _get(c, where, "b"))]
        elif kind == "linear_nonlinear":
            r0, h = _get(c, where, "R0"), _get(c, where, "H")
            terms = [(_get(c, where, "Q"), _get(c, where, "b"))]
        elif kind in ("nonlinear_sum", "nonlinear_double"):
            r0 = _get(c, where, "R0")
            k_terms = sorted(int(m.group(1)) for name in c if (m := re.fullmatch(r"Q(\d+)", name)))
            terms = [(c[f"Q{i}"], _get(c, where, f"b{i}")) for i in k_terms]
        else:
            return f"isotropic hardening {kind}"
    if len(terms) > most or any(q < 0 for q, _ in terms) or h < 0:
        return "softening or more than three saturating terms"
    return r0, h, tuple(terms)


def _porosity_table(value, where: str, name: str):
    """A coefficient of a porous criterion: a number, or a table of the porosity (``f``,
    ``ft``, ``fg``) as pairs ``(f, value)``."""
    from morphogen.homogenization.porous import MAX_TABLE

    if not isinstance(value, tuple):
        return float(value)
    if value[0] == "table" and value[1].lower() in ("f", "ft", "fg"):
        if len(value[2]) > MAX_TABLE:
            raise ValueError(f"zmat: {where}: {name}: a table of {MAX_TABLE} points at most")
        return [list(pair) for pair in value[2]]
    raise ValueError(
        f"zmat: {where}: {name} depends on {value[1]!r}: numbers, or tables of the porosity "
        f"({_see_coverage()})"
    )


def _native_porous(behaviour: ZmatBehaviour, porosity: float | None = None):
    """The :class:`~morphogen.homogenization.porous.PorousPlasticity` of a
    ``porous_plastic`` behaviour (see the module) and the notes of its translation."""
    from morphogen.homogenization.porous import MAX_ISOTROPIC, PorousPlasticity

    notes = []
    if behaviour.elasticity[0] != "isotropic":
        raise ValueError("zmat: porous_plastic: an isotropic elasticity (morphoGen's kernels)")
    if len(behaviour.potentials) != 1:
        raise ValueError(f"zmat: porous_plastic: one porous potential ({_see_coverage()})")
    for option in behaviour.options:
        if option in _POROUS_OPTIONS:
            raise ValueError(
                f"zmat: porous_plastic: **{option} is not translated ({_see_coverage()})"
            )
    if behaviour.modifier:
        notes.append(
            f"modifier {behaviour.modifier}: morphoGen's solvers choose small or finite strains "
            "(logarithmic strains)"
        )
    p = behaviour.potentials[0]
    kinematic = _porous_kinematic(p)
    if p.shear_anisotropy is not None and p.shear_anisotropy[0] not in ("mises", ""):
        raise ValueError(
            f"zmat: porous_plastic: *shear_anisotropy {p.shear_anisotropy[0]}: mises only "
            f"({_see_coverage()})"
        )
    flow = _flow(p)
    if isinstance(flow, str):
        raise ValueError(f"zmat: porous_plastic: {flow} ({_see_coverage()})")
    hardening = _isotropic_hardening(p, MAX_ISOTROPIC)
    if isinstance(hardening, str):
        raise ValueError(f"zmat: porous_plastic: {hardening} ({_see_coverage()})")
    r0, h, terms = hardening
    kind, c = p.criterion
    where = f"*porous_criterion {kind}"
    kwargs: dict = {}
    if kind == "gurson":
        _check_known(c, where, {"q1", "q2", "fs"})
        kwargs = {"q1": _get(c, where, "q1", default=1.5), "q2": _get(c, where, "q2", default=1.0)}
        fs = c.get("fs")
        if isinstance(fs, tuple) and fs[0] == "=":
            if fs[1].lower() not in ("f", "ft", "fg"):
                raise ValueError(f"zmat: {where}: fs = {fs[1]}: the porosity, or a table of it")
        elif fs is not None:
            table = _porosity_table(fs, where, "fs")
            kwargs["effective"] = [[0.0, table]] if isinstance(table, float) else table
        criterion = "gurson"
    elif kind in ("rousselier", "modified_rousselier"):
        known = {"sigma1", "D", "q2"} if kind == "modified_rousselier" else {"sigma1", "D"}
        _check_known(c, where, known)
        kwargs = {"sigma1": _get(c, where, "sigma1"), "D": _get(c, where, "D")}
        if kind == "modified_rousselier":
            kwargs["q2"] = _get(c, where, "q2")
        criterion = "rousselier"
    elif kind == "elliptic":
        _check_known(c, where, {"C", "F"})
        kwargs = {name: _porosity_table(c[name], where, name) for name in ("C", "F") if name in c}
        criterion = "elliptic"
    else:
        raise ValueError(f"zmat: porous criterion {kind!r} is not translated ({_see_coverage()})")
    nucleation = None
    if p.nucleation is not None:
        nkind, nc, crack_like = p.nucleation
        nwhere = f"*strain_nucleation {nkind}"
        if crack_like or "porosity_yield" in nc:
            raise ValueError(f"zmat: {nwhere}: crack_like and porosity_yield are not translated")
        if any(isinstance(v, tuple) for v in nc.values()):
            raise ValueError(f"zmat: {nwhere}: numbers only")
        names = {"constant": {"A": "A"}, "gaussian": {"fn": "fN", "en": "eN", "sn": "sN"},
                 "exponential": {"B": "B", "e0": "e0"}}  # fmt: skip
        if nkind not in names:
            raise ValueError(f"zmat: {nwhere}: constant, gaussian or exponential")
        _check_known(nc, nwhere, set(names[nkind]) | {"yield"})
        nucleation = {"type": nkind, "yield": nc.get("yield", 0.0)}
        nucleation.update({ours: _get(nc, nwhere, zmat) for zmat, ours in names[nkind].items()})
    if porosity is None:
        found = [behaviour.initial[k] for k in ("f", "ft", "fg", "f0") if k in behaviour.initial]
        porosity = found[0] if found else 0.0
        if not found:
            notes.append("initial porosity 0 (*initialize_variable f, or the option porosity)")
    pp = PorousPlasticity(
        criterion,
        r0,
        porosity=porosity,
        hardening=h,
        isotropic=terms,
        viscosity=flow[0],
        exponent=flow[1],
        nucleation=nucleation,
        kinematic=kinematic,
        **kwargs,
    )
    return pp, notes


def _porous_kinematic(p: ZmatPotential) -> tuple:
    """The kinematic terms ``(C, D)`` of a porous potential, with the bricks of gen_evp:
    ``*kinematic linear`` (``C``: Prager) and ``nonlinear`` (``C``, ``D``: Armstrong-Frederick;
    the type by default), numbers, four at most."""
    from morphogen.homogenization.porous import MAX_KINEMATIC

    out = []
    for kind, c in p.kinematic:
        kind = kind or "nonlinear"
        where = f"*kinematic {kind}"
        if kind not in ("linear", "nonlinear"):
            raise ValueError(
                f"zmat: porous_plastic: kinematic hardening {kind!r} is not translated "
                f"({_see_coverage()})"
            )
        if any(isinstance(v, tuple) for v in c.values()):
            raise ValueError(f"zmat: porous_plastic: {where}: numbers only")
        _check_known(c, where, {"C"} if kind == "linear" else {"C", "D"})
        out.append((_get(c, where, "C"), _get(c, where, "D") if kind == "nonlinear" else 0.0))
    if len(out) > MAX_KINEMATIC:
        raise ValueError(f"zmat: porous_plastic: {MAX_KINEMATIC} kinematic terms at most")
    return tuple(out)


def _real(value: float) -> str:
    return repr(float(value))


def _hill(c: dict) -> dict:
    """TFEL's Hill coefficients (``F (s11 - s22)^2 + G (s22 - s33)^2 + H (s33 - s11)^2 + 2 L
    s12^2 + 2 M s13^2 + 2 N s23^2``) of Z-mat's ``hill1`` ... ``hill6`` (on the deviator,
    components 11, 22, 33, 12, 23, 31)."""
    h = [_get(c, "*criterion hill", f"hill{i}", default=1.0) for i in range(1, 7)]
    # a (s22 - s33)^2 + b (s33 - s11)^2 + c (s11 - s22)^2 = 3/2 (h1 s11^2 + h2 s22^2 + h3
    # s33^2) for the deviators (s33 = -s11 - s22): three equations
    m = np.array([[1.0, 4.0, 1.0], [4.0, 1.0, 1.0], [4.0, 4.0, -2.0]])
    rhs = np.array([1.5 * (h[0] + h[2]), 1.5 * (h[1] + h[2]), 3.0 * h[2]])
    a, b, c = np.linalg.solve(m, rhs)
    return {"F": c, "G": a, "H": b, "L": 1.5 * h[3], "M": 1.5 * h[5], "N": 1.5 * h[4]}


def _brick_terms(items: dict) -> str:
    return ", ".join(f"{k} : {_real(v) if not isinstance(v, str) else v}" for k, v in items.items())


def _flow_lines(potential: ZmatPotential, index: int) -> tuple[list[str], str]:
    """The ``inelastic_flow`` entry of a potential and the name of its equivalent strain."""
    kind, c = potential.criterion
    if kind == "mises":
        criterion = '"Mises"'
    elif kind == "hill":
        criterion = f'"Hill" {{{_brick_terms(_hill(c))}}}'
    else:
        raise ValueError(f"zmat: criterion {kind!r}: mises or hill ({_see_coverage()})")
    flow, fc = potential.flow
    where = f"*flow {flow}"
    extra: dict = {}
    if flow == "plasticity":
        name, strain = '"Plastic"', "EquivalentPlasticStrain"
    elif flow == "norton":
        name, strain = '"Norton"', "EquivalentViscoplasticStrain"
        extra = {"K": _get(fc, where, "K"), "n": _get(fc, where, "n")}
    else:
        # the coefficients of the user-defined flows of the brick are renamed (MFront
        # reserves n, among others) and must all appear in the expression
        expressions = {
            "hyperbolic": (
                "zA * sinh((f / zK) ** zN) ** zM",
                {"zA": ("eps0", 1.0), "zK": ("K", None), "zN": ("n", 1.0), "zM": ("m", 1.0)},
            ),
            "double_norton": (
                "(f / zK) ** zN + (f / zK2) ** zN2",
                {"zK": ("K", None), "zN": ("n1", None), "zK2": ("K2", None), "zN2": ("n2", None)},
            ),
            "norton_exp": (
                "(f / zK) ** zN * exp(zA * (f / zK) ** (zN + 1))",
                {"zK": ("K", None), "zN": ("n", None), "zA": ("alpha", None)},
            ),
            "strain_hardening": (
                "(f / zK) ** zN * (p + zV) ** zM",
                {"zK": ("K", None), "zN": ("n", None), "zM": ("m", None), "zV": ("v0", None)},
            ),
        }
        if flow not in expressions:
            raise ValueError(f"zmat: flow {flow!r} is not translated ({_see_coverage()})")
        expression, names = expressions[flow]
        _check_known(fc, where, {zmat for zmat, _ in names.values()} | {"cutoff"})
        for key, (zmat, default) in names.items():
            extra[key] = _get(fc, where, zmat, default=default)
        extra["vp"] = f'"{expression}"'
        name, strain = '"UserDefinedViscoplasticity"', "EquivalentViscoplasticStrain"
    lines = [f"  inelastic_flow : {name} {{", f"    criterion : {criterion},"]
    for kind, c in potential.isotropic or [("constant", {"R0": 0.0})]:
        lines += [f"    isotropic_hardening : {term}," for term in _isotropic_terms(kind, c)]
    for kind, c in potential.kinematic:
        lines.append(f"    kinematic_hardening : {_kinematic_term(kind, c)},")
    for key, value in extra.items():
        lines.append(f"    {key} : {value if isinstance(value, str) else _real(value)},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("  },")
    return lines, strain


def _isotropic_terms(kind: str, c: dict) -> list[str]:
    where = f"*isotropic {kind}"
    r0 = _get(c, where, "R0", default=0.0)
    linear = f'"Linear" {{R0 : {_real(r0)}, H : {_real(c.get("H", 0.0))}}}'
    voce = '"Voce" {{R0 : 0, Rinf : {}, b : {}}}'
    if kind == "constant":
        return [f'"Linear" {{R0 : {_real(r0)}, H : 0}}']
    if kind in ("linear", "iso_linear"):
        return [linear]
    if kind in ("nonlinear", "linear_nonlinear"):
        return [linear, voce.format(_real(_get(c, where, "Q")), _real(_get(c, where, "b")))]
    if kind in ("nonlinear_sum", "nonlinear_double"):
        ids = sorted(int(m.group(1)) for name in c if (m := re.fullmatch(r"Q(\d+)", name)))
        return [linear] + [
            voce.format(_real(c[f"Q{i}"]), _real(_get(c, where, f"b{i}"))) for i in ids
        ]
    if kind == "power_law":
        k, e0, n = (_get(c, where, x) for x in ("K", "e0", "n"))
        return [
            f'"UserDefined" {{R : "zR + zK * (p + zE) ** zN", zR : {_real(r0)}, '
            f"zK : {_real(k)}, zE : {_real(max(e0, 0.0))}, zN : {_real(n)}}}"
        ]
    raise ValueError(f"zmat: isotropic hardening {kind!r} is not translated ({_see_coverage()})")


def _kinematic_term(kind: str, c: dict) -> str:
    where = f"*kinematic {kind}"
    if kind == "linear":
        return f'"Prager" {{C : {_real(_get(c, where, "C"))}}}'
    if kind == "nonlinear":
        return (
            f'"Armstrong-Frederick" {{C : {_real(_get(c, where, "C"))}, '
            f"D : {_real(_get(c, where, 'D'))}}}"
        )
    if kind == "nonlinear_evrad":
        return (
            f'"Burlet-Cailletaud" {{C : {_real(_get(c, where, "C"))}, '
            f"D : {_real(_get(c, where, 'D'))}, eta : {_real(_get(c, where, 'eta'))}}}"
        )
    raise ValueError(f"zmat: kinematic hardening {kind!r} is not translated ({_see_coverage()})")


def _see_coverage() -> str:
    return "see morphogen.homogenization.zmat.COVERAGE"


def zmat_mfront_source(behaviour: ZmatBehaviour, name: str = "ZmatBehaviour") -> str:
    """The MFront source of a ``gen_evp`` behaviour (the brick
    ``StandardElastoViscoPlasticity``, one inelastic flow per potential; see the
    module)."""
    if behaviour.elasticity[0] != "isotropic":
        raise ValueError("zmat: the plastic behaviours of MFront's brick need isotropic elasticity")
    el = _elastic_material_kwargs(behaviour)
    hooke = {"young_modulus": el["young"], "poisson_ratio": el["poisson"]}
    if behaviour.thermal_strain is not None:
        kind, c = behaviour.thermal_strain
        if kind != "isotropic":
            raise ValueError("zmat: **thermal_strain isotropic only")
        hooke["thermal_expansion"] = _get(c, "**thermal_strain", "alpha")
        hooke["thermal_expansion_reference_temperature"] = 293.15  # the natural state
    lines = [
        "@DSL Implicit;",
        f"@Behaviour {name};",
        "@Author morphoGen;",
        "@Description {",
        f'  "morphoGen {__version__}: a gen_evp behaviour of Z-mat, '
        f'{len(behaviour.potentials)} potential(s), with the brick StandardElastoViscoPlasticity."',
        "}",
        "@Algorithm NewtonRaphson;",
        "@Epsilon 1.e-14;",
        "@Theta 1;",
    ]
    if any(p.criterion[0] == "hill" for p in behaviour.potentials):
        # Hill's criterion is written in the axes of the material (those of the sample,
        # or of the grain of a crystal): an orthotropic behaviour for MFront
        lines.append("@OrthotropicBehaviour;")
    lines += [
        "@Brick StandardElastoViscoPlasticity{",
        f'  stress_potential : "Hooke" {{{_brick_terms(hooke)}}},',
    ]
    for k, potential in enumerate(behaviour.potentials):
        flow, _ = _flow_lines(potential, k)
        lines += flow
    lines[-1] = lines[-1].rstrip(",")
    lines.append("};")
    return "\n".join(lines) + "\n"


def _equivalent_strain(behaviour: ZmatBehaviour) -> str:
    first = behaviour.potentials[0]
    base = (
        "EquivalentPlasticStrain"
        if first.flow[0] == "plasticity"
        else ("EquivalentViscoplasticStrain")
    )
    return base + ("0" if len(behaviour.potentials) > 1 else "")


def zmat_material(
    source,
    *,
    name: str | None = None,
    directory=None,
    behaviour: int = 0,
    native: bool = True,
    porosity: float | None = None,
):
    """The :class:`~morphogen.homogenization.materials.Material` of the behaviour
    ``behaviour`` of a material file of Z-mat (path or text): morphoGen's kernels when they
    integrate it (unless ``native`` is false: MFront for all plastic behaviours, e.g. for
    the exports to other codes), otherwise an MFront behaviour whose source is written in
    ``directory`` (by default ``~/.cache/morphogen/zmat``) under the name ``name`` (by
    default the stem of the file); see the module. ``porosity``: the initial porosity of a
    ``porous_plastic`` behaviour (by default that of its ``*initialize_variable``)."""
    from morphogen.homogenization.materials import Material
    from morphogen.homogenization.mfront import MFrontBehaviour

    behaviours = read_zmat(source)
    if not 0 <= behaviour < len(behaviours):
        raise ValueError(f"zmat: behaviour {behaviour}: the file has {len(behaviours)}")
    b = behaviours[behaviour]
    kwargs = _elastic_material_kwargs(b)
    expansion = None
    if b.thermal_strain is not None:
        kind, c = b.thermal_strain
        if kind != "isotropic":
            raise ValueError("zmat: **thermal_strain isotropic only")
        expansion = _get(c, "**thermal_strain", "alpha")
    base = Material.from_config(kwargs)
    if expansion is not None:
        import dataclasses

        base = dataclasses.replace(base, expansion=expansion)
    if not b.potentials:
        return base
    import dataclasses

    if b.kind == "porous_plastic":
        if not native:
            raise ValueError("zmat: porous_plastic is translated for morphoGen's kernels only")
        porous, _ = _native_porous(b, porosity)
        return dataclasses.replace(base, porous_plasticity=porous)
    plasticity, _ = _native(b)
    if plasticity is not None and native:
        return dataclasses.replace(base, plasticity=plasticity)
    stem = mfront_name(name or (Path(source).stem if _is_file(source) else "ZmatBehaviour"))
    text = zmat_mfront_source(b, stem)
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    folder = Path(directory) if directory else Path.home() / ".cache" / "morphogen" / "zmat"
    folder = folder / key
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.mfront"
    path.write_text(text, encoding="utf-8")
    behaviour_spec = MFrontBehaviour(
        source=str(path), behaviour=stem, plastic_strain=_equivalent_strain(b)
    )
    return dataclasses.replace(base, mfront=behaviour_spec)


def mfront_name(stem: str) -> str:
    """A name of MFront behaviour from the stem of a file: ``Zmat`` and its letters and
    digits (the prefix avoids the names of TFEL's classes, such as ``Hill``)."""
    out = re.sub(r"[^A-Za-z0-9]", "", str(stem))
    out = out.removeprefix("Zmat")
    return "Zmat" + (out[:1].upper() + out[1:] if out else "Behaviour")


def _is_file(source) -> bool:
    return isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).is_file()
    )


def describe(source, behaviour: int = 0) -> dict:
    """How a behaviour of a Z-mat file is translated: ``"native"`` (morphoGen's kernels,
    with the configuration of the material) or ``"mfront"`` (and why)."""
    b = read_zmat(source)[behaviour]
    if not b.potentials:
        return {"translation": "native", "kind": "elastic", "material": _elastic_material_kwargs(b)}
    if b.kind == "porous_plastic":
        porous, notes = _native_porous(b)
        material = {**_elastic_material_kwargs(b), "porous_plasticity": porous.to_config()}
        return {"translation": "native", "kind": "porous", "material": material, "notes": notes}
    plasticity, reason = _native(b)
    if plasticity is not None:
        material = {**_elastic_material_kwargs(b), "plasticity": plasticity.to_config()}
        return {"translation": "native", "kind": "plastic", "material": material}
    return {"translation": "mfront", "reason": reason, "source": zmat_mfront_source(b)}


# -- writing -------------------------------------------------------------------------------


def _elasticity_lines(m, real) -> list[str]:
    """The ``**elasticity`` of a material: isotropic, cubic, or its stiffness component by
    component (named, so that the order of the shear components does not matter)."""
    if m.young is not None:
        return [" **elasticity isotropic", f"    young {real(m.young)}",
                f"    poisson {real(m.poisson)}"]  # fmt: skip
    if m.stiffness is None:
        raise ValueError("zmat: a material with elastic constants")
    c = np.asarray(m.stiffness, dtype=float)
    if m.symmetry == "cubic":
        return [" **elasticity cubic", f"    y1111 {real(c[0, 0])}",
                f"    y1122 {real(c[0, 1])}", f"    y1212 {real(c[3, 3])}"]  # fmt: skip
    names = ("11", "22", "33", "23", "31", "12")  # morphoGen's Voigt order
    out = [" **elasticity anisotropic"]
    for i in range(6):
        out += [f"    y{names[i]}{names[j]} {real(c[i, j])}" for j in range(i, 6) if c[i, j]]
    return out


def _written_flow(p, real) -> list[str]:
    """The flow of a von Mises or porous plasticity: Norton's, or plasticity."""
    if p.viscous:
        return ["   *flow norton", f"      n {real(p.exponent)}", f"      K {real(p.viscosity)}"]
    return ["   *flow plasticity"]


def _written_hardening(p, real, brick: str) -> list[str]:
    """The isotropic hardening of a von Mises or porous plasticity: one brick (``brick``:
    ``*isotropic`` or ``*isotropic_hardening``)."""
    r0, h, terms = p.yield_stress, p.hardening, p.isotropic
    if len(terms) > 1 and h:
        raise ValueError(
            "zmat: a linear hardening with several saturating terms has no isotropic brick "
            "of Z-mat (linear_nonlinear: one term; nonlinear_sum: no linear hardening)"
        )
    if not terms:
        kind = "linear" if h else "constant"
    elif len(terms) == 1:
        kind = "linear_nonlinear" if h else "nonlinear"
    else:
        kind = "nonlinear_sum"
    out = [f"   {brick} {kind}", f"      R0 {real(r0)}"]
    if h:
        out.append(f"      H {real(h)}")
    if len(terms) == 1:
        ((q, b),) = terms
        out += [f"      Q {real(q)}", f"      b {real(b)}"]
    else:
        for k, (q, b) in enumerate(terms, start=1):
            out += [f"      Q{k} {real(q)}", f"      b{k} {real(b)}"]
    return out


def _potential_lines(p, real) -> list[str]:
    """The potential of a :class:`~morphogen.homogenization.behaviours.Plasticity`: its
    criterion (von Mises, or Hill's with its coefficients ``hill1`` ... ``hill6`` of the
    components 11, 22, 33, 12, 23, 31), its flow, its kinematic terms (one brick each,
    named ``X1``... when several) and its isotropic hardening (one brick); ``real`` writes
    the numbers."""
    if p.hill is None:
        criterion = ["   *criterion mises"]
    else:
        criterion = ["   *criterion hill"]
        criterion += [f"      hill{i} {real(h)}" for i, h in enumerate(p.hill, start=1)]
    out = [" **potential gen_evp ev", *criterion, *_written_flow(p, real)]
    return out + _kinematic_lines(p.kinematic, real) + _written_hardening(p, real, "*isotropic")


def _kinematic_lines(kinematic, real) -> list[str]:
    """The kinematic terms ``(C, D)`` of a von Mises or porous potential: one brick each,
    ``linear`` when ``D = 0``, ``nonlinear`` otherwise, named ``X1``... when several."""
    out = []
    for k, (c, d) in enumerate(kinematic, start=1):
        name = f" X{k}" if len(kinematic) > 1 else ""
        if d == 0:
            out += [f"   *kinematic linear{name}", f"      C {real(c)}"]
        else:
            out += [f"   *kinematic nonlinear{name}", f"      C {real(c)}", f"      D {real(d)}"]
    return out


def _table_lines(name: str, table, real) -> list[str]:
    """A coefficient of a porous criterion: a number, or a table of the porosity (rows
    ``value f``)."""
    if len(table) == 1 and table[0][0] == 0.0:
        return [f"      {name} {real(table[0][1])}"]
    return [f"      {name} f", *[f"      {real(v)} {real(f)}" for f, v in table]]


def _porous_lines(pp, real) -> list[str]:
    """The porous potential of a :class:`~morphogen.homogenization.porous.PorousPlasticity`:
    its criterion, flow, kinematic terms (those of gen_evp), isotropic hardening and
    nucleation."""
    out = [" **porous_potential"]
    if pp.criterion == "gurson":
        if abs(pp.q3 - pp.q1**2) > 1e-12 * pp.q1**2:
            raise ValueError("zmat: Gurson's q3 is q1² in Z-mat (no coefficient q3)")
        out += ["   *porous_criterion gurson", f"      q1 {real(pp.q1)}", f"      q2 {real(pp.q2)}"]
        if pp.effective:  # the effective porosity: Tvergaard and Needleman's coalescence
            out += _table_lines("fs", pp.effective, real)
    elif pp.criterion == "rousselier":
        kind = "rousselier" if pp.q2 == 1.0 else "modified_rousselier"
        out += [f"   *porous_criterion {kind}", f"      sigma1 {real(pp.sigma1)}",
                f"      D {real(pp.D)}"]  # fmt: skip
        if kind == "modified_rousselier":
            out.append(f"      q2 {real(pp.q2)}")
    else:
        out += ["   *porous_criterion elliptic", *_table_lines("C", pp.C, real),
                *_table_lines("F", pp.F, real)]  # fmt: skip
    out += _written_flow(pp, real) + _kinematic_lines(pp.kinematic, real)
    out += _written_hardening(pp, real, "*isotropic_hardening")
    if pp.nucleation is not None:
        nucleation = dict(pp.nucleation)
        kind, threshold = nucleation.pop("type"), nucleation.pop("yield", 0.0)
        names = {"fN": "fn", "eN": "en", "sN": "sn"}  # Z-mat's names of the Gaussian
        out.append(f"   *strain_nucleation {kind}")
        out += [f"      {names.get(k, k)} {real(v)}" for k, v in nucleation.items()]
        if threshold:
            out.append(f"      yield {real(threshold)}")
    return out


def write_zmat(material, path=None, *, title: str = "", digits: int | None = None) -> str:
    """The material file of Z-mat of a material (a
    :class:`~morphogen.homogenization.materials.Material` or the mapping of its
    configuration), which :func:`read_zmat` and :func:`zmat_material` read back as the
    same material: a ``linear_elastic`` behaviour (isotropic, cubic or anisotropic
    elasticity); a ``gen_evp`` behaviour of one von Mises potential (a Norton flow or
    time-independent plasticity, one isotropic hardening ``constant``, ``linear``,
    ``nonlinear``, ``linear_nonlinear`` or ``nonlinear_sum``, the kinematic terms
    ``linear`` or ``nonlinear``) on an isotropic elasticity, with an isotropic thermal
    strain; or a ``porous_plastic`` behaviour of one porous potential (Gurson's criterion
    with the table of its effective porosity, Rousselier's or the elliptic one with the
    tables of its coefficients, the flow, kinematic terms and isotropic hardening of the
    matrix, the nucleation), its initial porosity in ``*initialize_variable`` (as in the examples of
    Z-mat's manual). The numbers are written in full (they read back exactly), or with
    ``digits`` significant digits. Written in ``path`` when it is given; ``title``: a
    comment at the top. Crystal plasticity and MFront behaviours are not written (MFront
    sources describe them)."""
    from morphogen.homogenization.materials import Material

    m = material if isinstance(material, Material) else Material.from_config(material)
    if m.void:
        raise ValueError("zmat: a void has no behaviour")
    for name, what in (
        ("crystal_plasticity", "crystal plasticity"),
        ("mfront", "an MFront behaviour"),
    ):
        if getattr(m, name) is not None:
            raise ValueError(f"zmat: {what} is not written as a material file (MFront is)")
    porous = m.porous_plasticity
    if (m.plasticity is not None or porous is not None) and m.young is None:
        raise ValueError("zmat: plasticity on an isotropic elasticity only")
    if m.analytic:
        raise ValueError(
            "zmat: the viscoelasticity, hyperelasticity and damage of "
            "morphogen.homogenization.laws are not written as a material file"
        )
    if digits is not None and int(digits) < 1:
        raise ValueError("zmat: digits, at least 1 (None: the numbers in full)")
    real = _real if digits is None else (lambda value: format(float(value), f".{int(digits)}g"))
    lines = [f"% {line}" for line in str(title).splitlines() if line.strip()]
    lines.append(f"% written by morphoGen {__version__}")
    if porous is not None:
        if m.expansion is not None:
            raise ValueError("zmat: porous_plastic without a thermal strain")
        lines += ["***material", "   *initialize_variable", f"      f {real(porous.porosity)}",
                  "***behavior porous_plastic"]  # fmt: skip
    else:
        kind = "gen_evp" if m.plasticity is not None else "linear_elastic"
        lines.append(f"***behavior {kind}")
    lines += _elasticity_lines(m, real)
    if m.expansion is not None:
        if not isinstance(m.expansion, float):
            raise ValueError("zmat: an isotropic thermal expansion only (**thermal_strain)")
        lines += [" **thermal_strain isotropic", f"    alpha {real(m.expansion)}"]
    if m.plasticity is not None:
        lines += _potential_lines(m.plasticity, real)
    if porous is not None:
        lines += _porous_lines(porous, real)
    lines.append("***return")
    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


#: What of Z-mat's library morphoGen covers: (category, Z-mat brick or model, status,
#: what morphoGen offers). Status: "native" (morphoGen's kernels), "mfront" (translated
#: by :func:`zmat_material` into an MFront behaviour), "elsewhere" (morphoGen has it under
#: another form), "user" (an MFront law of the user), "no".
COVERAGE = [
    ("model", "linear_elastic", "native", "isotropic, cubic, orthotropic, transverse, anisotropic"),
    ("model", "gen_evp", "native", "one potential of von Mises or Hill; the others through MFront"),
    ("model", "reduced_plastic", "native", "as gen_evp"),
    (
        "model",
        "finite_strain crystal (Méric-Cailletaud)",
        "elsewhere",
        "crystal_plasticity, small and finite strains (CrystalPlasticity)",
    ),
    (
        "model",
        "porous_plastic (one porous potential)",
        "native",
        "PorousPlasticity: small and finite strains, the CPU and the GPU",
    ),
    (
        "porous_criterion",
        "gurson, rousselier, modified_rousselier, elliptic",
        "native",
        "GTN (q3 = q1², fs = f or a table: coalescence), Rousselier, Abouaf's elliptic (C, F "
        "tables of f)",
    ),
    (
        "porous_criterion",
        "cam_clay, fkm, zhang_niemi, elliptic_aniso, elliptic_fiber",
        "no",
        "",
    ),
    (
        "kinematic (porous_plastic)",
        "linear, nonlinear",
        "native",
        "gurson and elliptic: the potential of σ - X (Besson & Guillemer-Neel 2003), four at "
        "most; not rousselier",
    ),
    (
        "strain_nucleation",
        "constant, gaussian, exponential (yield)",
        "native",
        "Chu-Needleman; not crack_like, porosity_yield",
    ),
    (
        "porous_plastic",
        "several porous potentials, Hf, Hp, *shear_anisotropy hill, broken_behavior, "
        "adiabatic_heating",
        "no",
        "",
    ),
    ("model", "damage elasticity, aniso_damage", "user", "an MFront law"),
    (
        "model",
        "hyper_elastic (Mooney, Ogden, Arruda-Boyce, neo-Hookean)",
        "user",
        "an MFront law; morphoGen's finite strains: Hencky's elasticity",
    ),
    ("model", "linear_viscoelastic, viscoelastic_spectral", "user", "an MFront law"),
    (
        "model",
        "needleman, crisfield, chaboche, generalized debonding",
        "elsewhere",
        "cohesive zone elements (morphogen.fracture.cohesive)",
    ),
    ("model", "diffusion, coefficient_diffusion", "elsewhere", "coupling.diffusion (transient)"),
    (
        "elasticity",
        "isotropic, cubic, orthotropic, transverse, anisotropic",
        "native",
        "crystals turned by the orientations of their grains",
    ),
    ("elasticity", "coefficients depending on the temperature", "no", ""),
    ("thermal_strain", "isotropic", "native", "expansion (crystals: anisotropic)"),
    ("criterion", "mises", "native", ""),
    (
        "criterion",
        "hill",
        "native",
        "hill1 ... hill6 (components 11, 22, 33, 12, 23, 31); Hill (F, G, H, L, M, N) "
        "for MFront with the models that need it",
    ),
    ("criterion", "tresca", "user", "MFront's Hosford criterion with a large exponent"),
    (
        "criterion",
        "linear_drucker_prager, cast_iron, karafillis_boyce, nouailhas, 2M1C",
        "user",
        "an MFront law",
    ),
    ("flow", "plasticity", "native", "rate-independent"),
    ("flow", "norton", "native", "dp/dt = <f/K>^n (n >= 1; MFront otherwise)"),
    (
        "flow",
        "hyperbolic, double_norton, norton_exp, strain_hardening",
        "mfront",
        "user-defined viscoplastic flows of the brick",
    ),
    (
        "flow",
        "sellars_tegart, gsell, modified_visco, function, flow_sum",
        "user",
        "UserDefinedViscoplasticity of MFront's brick",
    ),
    (
        "isotropic",
        "constant, linear, nonlinear, linear_nonlinear, nonlinear_sum",
        "native",
        "three saturating terms at most, without softening (MFront otherwise)",
    ),
    ("isotropic", "power_law", "mfront", "UserDefined isotropic hardening"),
    (
        "isotropic",
        "by_point, iso_table, linear_pp, nonlinear recovery",
        "user",
        "MFront's Data hardening (tables)",
    ),
    ("kinematic", "linear, nonlinear", "native", "Prager, Armstrong-Frederick (four at most)"),
    ("kinematic", "nonlinear_evrad", "mfront", "Burlet-Cailletaud"),
    ("kinematic", "ziegler, aniso_nonlinear, nonlinear with recovery", "user", "an MFront law"),
    ("potential", "several gen_evp potentials", "mfront", "one inelastic flow per potential"),
    ("potential", "interactions between potentials", "no", ""),
    (
        "modifier",
        "lagrange_rotate, lagrange_polar (finite strains)",
        "elsewhere",
        "finite strains: logarithmic strains (Hencky) of the loading paths",
    ),
    ("modifier", "plane_stress", "no", "morphoGen's volume elements are 3D"),
    (
        "simulation",
        "****simulate (material point)",
        "elsewhere",
        "loading paths of a single material point (PathResult) and of volume elements",
    ),
    (
        "optimization",
        "****optimize (identification)",
        "elsewhere",
        "homogenization.identification: von Mises parameters on references (material "
        "points), the behaviour found written back by write_zmat",
    ),
]


def coverage_markdown() -> str:
    """:data:`COVERAGE` as a Markdown table."""
    rows = ["| category | Z-mat | morphoGen | how |", "|---|---|---|---|"]
    rows += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in COVERAGE]
    return "\n".join(rows) + "\n"
