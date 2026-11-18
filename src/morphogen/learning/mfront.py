"""MFront sources of the learned laws: a material whose ``learned`` names a trained model
(:class:`~morphogen.learning.behaviour.LearnedBehaviour`) becomes an MFront behaviour, for
the codes that MFront serves (AMITEX_FFTP and Cast3M through its ``umat`` interface,
Abaqus, Code_Aster, MGIS), as :mod:`morphogen.io.mfront` writes morphoGen's own
behaviours: :func:`~morphogen.io.mfront.mfront_source`, ``morphogen mfront`` and the
AMITEX and Abaqus exports take the learned materials like the others.

The trained parameters are constants of the source (their shortest decimal forms, which
read back to the bit) and the forward pass of the model and its consistent tangent are
morphoGen's C++ port of the models (``learned.cpp``: :class:`~morphogen.learning.networks.
FeedForward`, :class:`~morphogen.learning.networks.Potential`, the gated recurrent unit of
:class:`~morphogen.learning.networks.Recurrent` and the Newton solve of the incremental
potential of :class:`~morphogen.learning.standard.Standard`, line by line), in a behaviour
of the ``Default`` DSL:

* small strains: the strain and the stress of the model;
* finite strains: the laws of the logarithmic strain (``strain="logarithmic"``) with
  ``@StrainMeasure Hencky``, whose strain is ``1/2 ln C`` and stress the ``T`` conjugate to
  it, the very ``T`` of the model (MFront turns it to the Cauchy stress and its tangent
  ``dT/dH`` to the operator of the interface); the laws of the Green-Lagrange strain with
  ``@StrainMeasure GreenLagrange`` (``E`` and the second Piola-Kirchhoff stress);
* the state of the recurrent network (its ``hidden`` values) and the internal variables of
  the standard model: an array of auxiliary state variables (``LearnedState``), zero at the
  natural state as in morphoGen; the strain at the start of a step, which the recurrent
  network takes, is MFront's (``eto``), the strain that morphoGen keeps in its internal
  variables;
* thermal expansion: the stress-free thermal strain ``alpha (T + dT - 293.15)`` of the
  material (its tensor in the axes of the material) taken out of the strain the model sees,
  that of the Hencky points in finite strains (``1/2 ln C - alpha dT``), as
  :mod:`morphogen.learning.behaviour` does; MFront's reference temperature, 293.15 K, is
  morphoGen's natural state. A law of the Green-Lagrange strain that expands is refused
  (its split of the Hencky points, ``E_m = (exp(2 H_m) - I) / 2``, is not written);
* orientations: a law in the axes of its material (``oriented``, the default) is an
  orthotropic behaviour (``@OrthotropicBehaviour``), which the solver gives in the axes of
  each grain (MGIS through morphoGen turns the strains, the umat interface takes the axes
  V1, V2, Abaqus its ``*ORIENTATION``; in finite strains, the first six state variables in
  Abaqus); a law in the axes of the sample is isotropic (not turned). A law in the axes of
  the sample with an anisotropic expansion, which the orientations of the grains would
  turn, is refused.

Order of the components at each interface: TFEL's symmetric tensors (11, 22, 33, then
``sqrt(2)`` times 12, 13, 23) in the behaviour (``eto``, ``sig``, ``Dt``); morphoGen's
Voigt order (11, 22, 33, 23, 31, 12, the engineering shears of the strains) in the model
and its C++ port: the strain goes to the model as ``e[VOIGT[i]] = c_i eto[i]``, the stress
comes back as ``sig[i] = c_i s[VOIGT[i]]`` and the tangent as ``Dt(i, j) = c_i c_j
K[VOIGT[i]][VOIGT[j]]``, ``VOIGT = (0, 1, 2, 5, 4, 3)`` and ``c = (1, 1, 1, sqrt(2),
sqrt(2), sqrt(2))``, as in the sources of morphoGen's porous kernel.

The tangents: symmetric for the potential and the standard model
(``@ProvidesSymmetricTangentOperator``), not for the perceptron and the recurrent network
(``@ProvidesTangentOperator``). MGIS through morphoGen reads that declaration (in the
header MFront generates, :func:`morphogen.homogenization.mfront.declared_symmetric`): its
solvers symmetrize the tangents of the first for the conjugate gradients and pass those
of the others whole, to GMRES in the volume elements and to Newton's method at a material
point, as they do for morphoGen's own learned points. A step of the standard model that
stops without converging keeps the last iterate of Newton's method, as morphoGen's points
do; a stress or a tangent that is not finite fails the integration.
"""

from __future__ import annotations

import re
from importlib import resources

import numpy as np

from morphogen._version import __version__

#: The strain measures of the learned laws in finite strains, by the option ``strain`` of
#: their models
MEASURES = {"logarithmic": "Hencky", "green-lagrange": "GreenLagrange"}
_SECTION = re.compile(r"^// ---- (\w+)\s*$", re.M)


def _real(value) -> str:
    return repr(float(value))


def _values(values, per_line: int = 6) -> str:
    """A C array initializer of reals, ``per_line`` values a line."""
    values = [_real(v) for v in np.ravel(values)] or ["0.0"]
    rows = [", ".join(values[k : k + per_line]) for k in range(0, len(values), per_line)]
    return "{\n    " + ",\n    ".join(rows) + "}"


def _reals(name: str, values) -> str:
    values = np.ravel(np.asarray(values, dtype=float))
    return f"constexpr double {name}[{max(values.size, 1)}] = {_values(values)};"


def _ints(name: str, values) -> str:
    values = [int(v) for v in values] or [0]
    return f"constexpr int {name}[{len(values)}] = {{{', '.join(map(str, values))}}};"


def _flag(name: str, value) -> str:
    return f"constexpr bool {name} = {'true' if value else 'false'};"


def learned_model(material, kinematics: str = "small"):
    """The model of the learned law of a material, checked for an MFront source of the
    kinematics: trained on data of its measure (small strains, or the Green-Lagrange
    measures in finite strains), its expansion one that the source takes (a law of the
    Green-Lagrange strain does not expand; a law in the axes of the sample has an isotropic
    expansion)."""
    from morphogen.learning.behaviour import learned_law

    if kinematics not in ("small", "finite"):
        raise ValueError("kinematics: 'small' or 'finite'")
    finite = kinematics == "finite"
    try:
        law = learned_law(material.learned, finite=finite)
    except ValueError as exc:
        raise ValueError(f"mfront: {exc}") from None
    model = law.model
    if material.expands:
        if finite and not model.logarithmic:
            raise ValueError(
                "mfront: a learned law of the Green-Lagrange strain that expands (its "
                "thermal split, that of the Hencky points, is not exported: train the law "
                "of the logarithmic strain, strain='logarithmic')"
            )
        if isinstance(material.expansion, tuple) and not material.learned.oriented:
            raise ValueError(
                "mfront: an anisotropic expansion of a learned law in the axes of the sample "
                "(the orientations of the grains turn the expansion, not the law)"
            )
    return model


def learned_strain_measure(material, kinematics: str = "small") -> str | None:
    """The strain measure of the MFront source of a learned material: none in small
    strains; ``Hencky`` (a law of the logarithmic strain) or ``GreenLagrange`` in finite
    strains."""
    model = learned_model(material, kinematics)
    if kinematics == "small":
        return None
    return MEASURES["logarithmic" if model.logarithmic else "green-lagrange"]


def _expansion(material) -> np.ndarray:
    """The thermal expansion of the material in the axes of its law (morphoGen's Voigt
    components, engineering shears: the thermal strain of a unit change of temperature)."""
    from morphogen.learning.data import _voigt

    if not material.expands:
        return np.zeros(6)
    a = np.asarray(material.expansion_tensor(), dtype=float)
    return _voigt(0.5 * (a + a.T), engineering=True)


def _constants(model) -> list[str]:
    """The C++ constants of a model: its scales, sizes and trained parameters (the values
    that its NumPy and Numba functions compute with)."""
    s = model._scales()
    lines = [
        "// the scales of the training set: x = strain / SCALE_STRAIN, y = stress / SCALE_STRESS",
        f"constexpr double SCALE_STRAIN = {_real(s.strain)};",
        f"constexpr double SCALE_STRESS = {_real(s.stress)};",
        f"constexpr double SCALE_INCREMENT = {_real(s.increment)};",
        f"constexpr double SCALE_TIME = {_real(s.time)};",
        f"constexpr int N_STATE = {model.n_state};",
    ]
    p = model.params
    if model.kind == "feedforward":
        sizes = list(model.options["hidden"])
        n = len(sizes)
        lines += [
            f"constexpr int N_LAYERS = {n};",
            _ints("SIZES", sizes),
            f"constexpr int WIDEST = {max(6, *sizes)};",
            "// the hidden layers (W_k: out x in, row by row), the output and the linear term",
            _reals("W", np.concatenate([p[f"W{k}"].ravel() for k in range(n)])),
            _reals("B", np.concatenate([p[f"b{k}"] for k in range(n)])),
            _reals("WOUT", p["Wout"]),
            _reals("BOUT", p["bout"]),
            _reals("LIN", p["L"]),
        ]
    elif model.kind == "potential":
        sizes = list(model.options["hidden"])
        n = len(sizes)
        w, wout, amat = model._weights()
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        wpos = [w[k].ravel() for k in range(1, n)]
        # the offsets of the W_k in WPOS (none for the first layer)
        woffsets = [0, 0, *np.cumsum([v.size for v in wpos])[:-1]] if n > 1 else [0]
        g0 = model._gradient(np.zeros((1, 6)))[0][0]
        lines += [
            f"constexpr int N_LAYERS = {n};",
            _ints("SIZES", sizes),
            _ints("OFFSETS", offsets),
            f"constexpr int TOTAL = {sum(sizes)};",
            f"constexpr int WIDEST = {max(6, *sizes)};",
            "// U_k (units x 6) and b_k of the layers; W_k = softplus(V_k) >= 0 of the layers",
            "// k > 0 (units x units below), w = softplus(v) >= 0, A = L L^T, dW/dx (0)",
            _reals("U", np.concatenate([p[f"U{k}"].ravel() for k in range(n)])),
            _reals("BIAS", np.concatenate([p[f"b{k}"] for k in range(n)])),
            _reals("WPOS", np.concatenate(wpos) if wpos else []),
            _ints("WOFFSETS", woffsets[:n]),
            _reals("WOUT", wout),
            _reals("AMAT", amat),
            _reals("G0", g0),
        ]
    elif model.kind == "recurrent":
        lines += [
            f"constexpr int HIDDEN = {model.options['hidden']};",
            f"constexpr int N_INPUTS = {model.n_inputs};",
            _flag("RATE", model.options["rate"]),
            _flag("LOGARITHMIC", model.logarithmic),
            "// the gates (W_i: inputs x 3 hidden, W_h: hidden x 3 hidden; reset, update, new)",
            "// and the output y = V h + Q x + c",
            _reals("WI", p["Wi"]),
            _reals("WH", p["Wh"]),
            _reals("BG", p["b"]),
            _reals("BN", p["bn"]),
            _reals("V", p["V"]),
            _reals("Q", p["Q"]),
            _reals("C", p["c"]),
        ]
    elif model.kind == "standard":
        from morphogen.learning import standard

        u, b, w, amat, g0, _, bm, gm, pm, cm, vm, delta, plastic, viscous, _, _ = (
            model._values())  # fmt: skip
        m = model.n_state
        lines += [
            f"constexpr int M = {m};",
            f"constexpr int D = {6 + m};",
            f"constexpr int NH = {b.size};",
            f"constexpr int N_NORMS = {bm.shape[0]};",
            f"constexpr int N_RIDGES = {cm.shape[0]};",
            f"constexpr double DELTA = {_real(delta)};",
            _flag("PLASTIC", plastic),
            _flag("VISCOUS", viscous),
            f"constexpr double NEWTON_TOL = {_real(standard.NEWTON_TOL)};",
            f"constexpr int NEWTON_MAX = {int(standard.NEWTON_MAX)};",
            f"constexpr double ARMIJO = {_real(standard._ARMIJO)};",
            "// the free energy: U (ridges x D), b, w = softplus >= 0, A = L L^T, dW/dz (0)",
            _reals("U", u),
            _reals("BIAS", b),
            _reals("WR", w),
            _reals("A", amat),
            _reals("G0", g0),
            "// the dissipation: B_j (norms x M x M), B_j^T B_j, P = softplus >= 0 (ridges x",
            "// norms), C (ridges x M), the viscous V = M M^T (M x M)",
            _reals("BM", bm),
            _reals("GM", gm),
            _reals("P", pm),
            _reals("CM", cm),
            _reals("VM", vm),
        ]
    else:
        raise ValueError(f"mfront: no source of a {model.kind!r} model")
    return lines


def _port(kind: str) -> str:
    """The common part of morphoGen's C++ port of the learned laws and the section of the
    model of ``kind`` (``learned.cpp``)."""
    text = (resources.files("morphogen.learning") / "learned.cpp").read_text(encoding="utf-8")
    parts = _SECTION.split(text)
    sections = dict(zip(parts[1::2], parts[2::2], strict=True))
    return "".join([parts[0], "// ---- common", sections["common"], f"// ---- {kind}",
                    sections[kind]]).strip("\n")  # fmt: skip


def _describe(model, material) -> str:
    """The model, its options and its file, for the description of a source."""
    options = model.options
    if model.kind in ("feedforward", "potential"):
        what = f"{model.kind} {'-'.join(str(h) for h in options['hidden'])}"
    elif model.kind == "recurrent":
        what = f"recurrent (GRU) of {options['hidden']} states" + (", rate" if options["rate"]
                                                                  else "")  # fmt: skip
    else:
        terms = [f"{options['internal']} internal variables", f"{options['hidden']} ridges"]
        terms += ["rate-independent dissipation"] if options["plastic"] else []
        terms += ["viscous dissipation"] if options["rate"] else []
        what = f"standard ({', '.join(terms)})"
    path = model.path
    where = f" of {path.name}" if path is not None else ""
    turned = "in the axes of the material" if material.learned.oriented else "isotropic"
    return f"{what}{where}, {turned}"


def learned_source(material, name: str, *, kinematics: str = "small") -> str:
    """The MFront source of the behaviour ``name`` of a material with a learned law (see the
    module): its model's parameters as constants, its forward pass and tangent by
    morphoGen's C++ port, in small strains or in finite strains (Hencky's measure for the
    laws of the logarithmic strain, Green-Lagrange's for the others)."""
    from morphogen.io.mfront import ABAQUS_AXES

    model = learned_model(material, kinematics)
    measure = learned_strain_measure(material, kinematics)
    oriented = material.learned.oriented
    n_state = model.n_state
    alpha = _expansion(material)
    symmetric = model.kind in ("potential", "standard")
    if measure is None:
        space = "in small strains"
    elif measure == "Hencky":
        space = ("in finite strains, of the logarithmic strain 1/2 ln C (Hencky), its stress "
                 "the conjugate T")  # fmt: skip
    else:
        space = ("in finite strains, of the Green-Lagrange strain, its stress the second "
                 "Piola-Kirchhoff stress")  # fmt: skip
    terms = [_describe(model, material)]
    if material.expands:
        terms.append("thermal expansion " + ", ".join(f"{v:g}" for v in alpha))
    space_name = f"morphogen_{name}"
    lines = [
        "@DSL Default;",
        f"@Behaviour {name};",
        "@Author morphoGen;",
        "@Description {",
        f'  "morphoGen {__version__}: a learned law {space}: {"; ".join(terms)}; its forward '
        "pass and consistent tangent by morphoGen's C++ port of the model (learned.cpp), "
        'the trained parameters constants of the source."',
        "}",
        "@ModellingHypothesis Tridimensional;",
    ]
    if oriented:
        lines.append("@OrthotropicBehaviour;")
        if measure is not None:
            lines.append(ABAQUS_AXES)
    if measure is not None:
        lines.append(f"@StrainMeasure {measure};")
    lines.append("@ProvidesSymmetricTangentOperator;" if symmetric
                 else "@ProvidesTangentOperator;")  # fmt: skip
    if n_state:
        lines += [f"@AuxiliaryStateVariable real lv[{n_state}];",
                  'lv.setEntryName("LearnedState");']  # fmt: skip
    lines += [
        "",
        "@Includes{",
        "// the learned law: its scales, sizes and trained parameters, then morphoGen's C++",
        "// port of its model (MFront includes this block in several headers of one",
        "// translation unit)",
        f"#ifndef {space_name.upper()}_LEARNED",
        f"#define {space_name.upper()}_LEARNED",
        "#include <cmath>",
        f"namespace {space_name} {{",
        *_constants(model),
        "// morphoGen's components (11 22 33 23 31 12) of TFEL's (11 22 33 12 13 23)",
        "constexpr int VOIGT[6] = {0, 1, 2, 5, 4, 3};",
        "// the thermal strain of a unit change of temperature, in the axes of the law",
        "// (morphoGen's components, engineering shears)",
        _reals("EXPANSION", alpha),
        _port(model.kind),
        f"}}  // end of namespace {space_name}",
        "#endif",
        "}",
        "",
        "@Integrator{",
        f"  namespace ml = {space_name};",
        "  // TFEL's components of tensors: sqrt(2) times the shears 12, 13, 23; the model's:",
        "  // morphoGen's 23, 31, 12, the engineering shears of the strains",
        "  const auto r2 = std::sqrt(2.);",
        "  const StrainStensor e = eto + deto;",
        "  double e1[6], e0[6], s6[6], k6[36];",
        "  for (int i = 0; i != 6; ++i) {",
        "    const auto c = i < 3 ? 1 : r2;",
        "    e1[ml::VOIGT[i]] = c * e[i];",
        "    e0[ml::VOIGT[i]] = c * eto[i];",
        "  }",
    ]
    if material.expands:
        lines += [
            "  // the mechanical strain: morphoGen's thermal strain alpha (T - 293.15) out, at",
            "  // the end and at the start of the step (logarithmic in finite strains)",
            "  for (int r = 0; r != 6; ++r) {",
            "    e1[r] -= ml::EXPANSION[r] * (T + dT - 293.15);",
            "    e0[r] -= ml::EXPANSION[r] * (T - 293.15);",
            "  }",
        ]
    if n_state:
        lines += [
            "  double s0[ml::N_STATE], s1[ml::N_STATE];",
            "  for (int r = 0; r != ml::N_STATE; ++r) {",
            "    s0[r] = lv[r];",
            "  }",
            "  if (ml::law(e1, e0, s0, dt, s6, k6, s1) != 0) {",
            "    return false;",
            "  }",
            "  for (int r = 0; r != ml::N_STATE; ++r) {",
            "    lv[r] = s1[r];",
            "  }",
        ]
    else:
        lines += [
            "  if (ml::law(e1, e0, nullptr, dt, s6, k6, nullptr) != 0) {",
            "    return false;",
            "  }",
        ]
    lines += [
        "  for (int i = 0; i != 6; ++i) {",
        "    sig[i] = (i < 3 ? 1 : r2) * s6[ml::VOIGT[i]];",
        "  }",
        "  if (computeTangentOperator_) {",
        "    for (int i = 0; i != 6; ++i) {",
        "      for (int j = 0; j != 6; ++j) {",
        "        const auto c = (i < 3 ? 1 : r2) * (j < 3 ? 1 : r2);",
        "        Dt(i, j) = c * k6[6 * ml::VOIGT[i] + ml::VOIGT[j]];",
        "      }",
        "    }",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"
