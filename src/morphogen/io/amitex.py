"""Runs of AMITEX_FFTP, the FFT solver of voxel microstructures of the CEA
(https://amitexfftp.github.io/AMITEX), on morphoGen images, and their results.

:func:`write_amitex` writes the files of the runs of a voxel image of phases:

* the material and zone numbers of the voxels (``<name>_materials.vtk``,
  ``<name>_zones.vtk``: legacy VTK with exactly ten header lines, then big-endian 32-bit
  integers, x varying fastest; numbers from 1). Each phase present is a material, in
  increasing order; the crystals of a phase turned by the orientations of their grains
  have one zone per grain, the other phases one zone;
* the material file (``<name>_material.xml``): the native law ``elasiso`` for isotropic
  phases (the Lamé coefficients; zero for voids) and ``elasaniso`` for crystals (the
  nine orthotropic constants C11, C12, C13, C22, C23, C33, C66, C55, C44 in the axes of
  the crystal, then the first two axes of the crystal, the first two rows of its
  orientation matrix, in the axes of the sample for each zone, in binary files of
  big-endian doubles); the reference medium of the solver is the mean of the extreme
  moduli, as in morphoGen's FFT solver. The nonlinear materials (von Mises and crystal
  plasticity) are MFront laws (:mod:`morphogen.io.mfront`, in ``mfront/``), compiled by
  the script with the ``umat`` interface (``mfront/src/libUmatBehaviour.so``, the law
  ``umat<behaviour>``), with the coefficients that this interface imposes: YOUN, NU,
  RHO, ALPH (von Mises), or the nine orthotropic constants YG1, YG2, YG3, NU12, NU23,
  NU13, G12, G23, G13, the axes V1, V2 of the crystal in each zone (as those of
  ``elasaniso``), RHO, ALP1, ALP2, ALP3 (crystals); their internal variables start at
  zero;
* a loading file per run (``<name>_<run>.xml``): the six unit strains of the stiffness
  tensor (one partial loading each; linear materials only), named macroscopic
  loadings, or loading paths (one partial loading per segment);
* the algorithm file (``<name>_algorithm.xml``), ``<name>_run.sh``, which compiles the
  MFront laws and runs them all with MPI, and ``<name>_amitex.json``, the description of
  the export that :func:`read_amitex_run` reads back with the results.

In finite strains (``kinematics="finite"``: ``Small_Perturbations`` false, loading paths
of the nine components of the displacement gradient and of the nominal stress), every
phase but the voids is an MFront law, Hencky's elasticity for the elastic ones (AMITEX's
native laws in finite strains are not morphoGen's): the laws of Hencky's strain measure
(elasticity, von Mises) are the functions ``umat<behaviour>_malls`` of the interface,
which compute the logarithmic strain from the deformation gradients that AMITEX gives
them, those of the deformation gradient (crystals: morphoGen's kernel) ``umat<behaviour>``.

AMITEX orders the components of symmetric tensors (xx, yy, zz, xy, xz, yz), with
engineering shear strains, and those of gradients (xx, yy, zz, xy, xz, yz, yx, zx, zy);
morphoGen's are (11, 22, 33, 23, 31, 12) and row by row.

:func:`read_amitex` reads the averages of a run over the unit cell (``.std``),
:func:`read_amitex_materials` and :func:`read_amitex_zones` those of its materials
(``.mstd``) and of the zones of a material (``.zstd``), and :func:`read_amitex_run` the
runs of an export, with the apparent stiffness of its unit strains;
:func:`morphogen.io.read_vtk` reads the fields of the runs (``.vtk``).
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morphogen._version import __version__
from morphogen.image import VoxelImage
from morphogen.io.mfront import (
    identifier,
    mfront_state_size,
    mfront_strain_measure,
    mfront_symmetry,
    orthotropic_constants,
    umat_function,
    user_behaviour,
    write_mfront,
    write_parameters,
)

#: AMITEX's names of the components of symmetric tensors, in morphoGen's Voigt order
#: (11, 22, 33, 23, 31, 12); in finite strains, of the macroscopic components (11, 22,
#: 33, 23, 13, 12) of gradients and nominal stresses
COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")
#: AMITEX's names of the nine components of gradients
GRADIENT = ("xx", "yy", "zz", "xy", "xz", "yz", "yx", "zx", "zy")
#: morphoGen's component k of a symmetric tensor is AMITEX's _FROM6[k]
_FROM6 = np.array([0, 1, 2, 5, 4, 3])
#: component (11, 12, 13, 21, 22, 23, 31, 32, 33) of a gradient: AMITEX's _FROM9[k]
_FROM9 = np.array([0, 3, 4, 6, 1, 5, 7, 8, 2])
#: the macroscopic components (11, 22, 33, 23, 13, 12) of tensors stored row by row
_MACRO = np.array([0, 4, 8, 5, 2, 1])
#: Laws of the material file
LAWS = ("elasiso", "elasaniso")
#: The library of the MFront laws of the nonlinear materials, compiled by the script of an
#: export with the umat interface (relative to its directory)
UMAT_LIBRARY = "mfront/src/libUmatBehaviour.so"
#: The material properties that the umat interface imposes (glossary names): of isotropic
#: behaviours, of orthotropic ones before the axes V1, V2 of their material, and after them
_UMAT_ISOTROPIC = ("YoungModulus", "PoissonRatio", "MassDensity", "ThermalExpansion")
_UMAT_ORTHOTROPIC = ("YoungModulus1", "YoungModulus2", "YoungModulus3", "PoissonRatio12",
                     "PoissonRatio23", "PoissonRatio13", "ShearModulus12", "ShearModulus23",
                     "ShearModulus13")  # fmt: skip
_UMAT_TAIL = ("MassDensity", "ThermalExpansion1", "ThermalExpansion2", "ThermalExpansion3")


def _umat_coefficients(m, user, kinematics: str) -> tuple[bool, list[float], list[float]]:
    """Whether the MFront law of a material is orthotropic, and its coefficients for the
    umat interface: those that the interface imposes (the elastic constants of the
    material, its density, no thermal expansion; the values of the material properties of
    the behaviour that have their glossary names), the axes of the zones of orthotropic
    behaviours, then the other material properties of the behaviour (``user``: a
    :class:`~morphogen.io.mfront.UserBehaviour`; None: a law written by morphoGen) in
    order. Returns the coefficients before the axes and those after them."""
    from morphogen.homogenization.materials import isotropic_matrix

    values = dict(m.mfront.material_properties) if user is not None else {}
    if user is not None:
        orthotropic = user.orthotropic
    else:
        orthotropic = mfront_symmetry(m, kinematics) == "orthotropic"
    if orthotropic:
        stiffness = m.stiffness if m.stiffness is not None else isotropic_matrix(*m.lame)
        default = dict(zip(_UMAT_ORTHOTROPIC, orthotropic_constants(stiffness), strict=True))
        default.update(dict.fromkeys(_UMAT_TAIL, 0.0), MassDensity=m.density or 0.0)
        head = [values.get(k, default[k]) for k in _UMAT_ORTHOTROPIC]
        rest = [values.get(k, default[k]) for k in _UMAT_TAIL]
        imposed = _UMAT_ORTHOTROPIC + _UMAT_TAIL
    else:
        e, nu = (m.young, m.poisson) if m.young is not None else (0.0, 0.0)
        default = {"YoungModulus": e, "PoissonRatio": nu, "MassDensity": m.density or 0.0,
                   "ThermalExpansion": 0.0}  # fmt: skip
        head = [values.get(k, default[k]) for k in _UMAT_ISOTROPIC]
        rest = []
        imposed = _UMAT_ISOTROPIC
    own = user.properties if user is not None else ()
    return orthotropic, head, rest + [values[k] for k in own if k not in imposed]


# -- inputs ----------------------------------------------------------------------------------


def write_amitex_field(path, values, spacing, origin=(0.0, 0.0, 0.0), name: str = "MaterialId"):
    """An integer field of the voxels as AMITEX_FFTP reads it: legacy VTK with exactly ten
    header lines (``DIMENSIONS``: the numbers of voxels plus one, cell data), then
    big-endian 32-bit integers, x varying fastest."""
    data = np.asarray(values)
    if data.ndim != 3 or data.dtype.kind not in "iub":
        raise ValueError("AMITEX fields: an integer field of the voxels (3D)")
    if data.size and (data.min() < 0 or data.max() >= 2**31):
        raise ValueError("AMITEX fields: values between 0 and 2^31 - 1")
    nx, ny, nz = data.shape
    s = [float(v) for v in spacing]
    o = [float(v) for v in origin]
    head = (
        "# vtk DataFile Version 4.5\n"
        "Materiau\n"
        "BINARY\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n"
        f"ORIGIN {o[0]:.10e} {o[1]:.10e} {o[2]:.10e}\n"
        f"SPACING {s[0]:.10e} {s[1]:.10e} {s[2]:.10e}\n"
        f"CELL_DATA {nx * ny * nz}\n"
        f"SCALARS {name} int\n"
        "LOOKUP_TABLE default\n"
    )
    raw = np.ascontiguousarray(data.transpose(2, 1, 0), dtype=">i4").tobytes()
    path = Path(path)
    path.write_bytes(head.encode("ascii") + raw)
    return path


def _write_bin(path: Path, values) -> Path:
    """Values per zone in AMITEX's binary format: their number, ``double``, then
    big-endian doubles."""
    v = np.ascontiguousarray(np.asarray(values, dtype=float).ravel(), dtype=">f8")
    path.write_bytes(f"{v.size}\ndouble\n".encode("ascii") + v.tobytes())
    return path


def _number(value: float) -> str:
    return repr(float(value))


def _xml(path: Path, root: ET.Element) -> Path:
    ET.indent(root, space="  ")
    text = ET.tostring(root, encoding="unicode")
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + text + "\n", encoding="utf-8")
    return path


def _orthotropic(stiffness, phase: int) -> list[float]:
    """The nine constants of ``elasaniso`` of a stiffness in the axes of the crystal
    (morphoGen's Voigt order): C11, C12, C13, C22, C23, C33, C66, C55, C44."""
    d = np.asarray(stiffness, dtype=float)
    coupling = np.concatenate([d[:3, 3:].ravel(), d[3:, :3].ravel(),
                               d[3:, 3:][~np.eye(3, dtype=bool)]])  # fmt: skip
    if np.abs(coupling).max() > 1e-12 * np.abs(d).max():
        raise ValueError(
            f"phase {phase}: AMITEX's elasaniso is orthotropic in the axes of the crystal; "
            "this stiffness couples normal and shear components"
        )
    return [d[0, 0], d[0, 1], d[0, 2], d[1, 1], d[1, 2], d[2, 2], d[5, 5], d[4, 4], d[3, 3]]


def _components(element: ET.Element, values, controls, finite: bool) -> None:
    """The driving of the components at the end of a partial loading: morphoGen's six
    components (strains where ``controls``), and in finite strains the lower triangle of
    the gradient, zero."""
    for name, value, strain in zip(COMPONENTS, values, controls, strict=True):
        ET.SubElement(element, name, Driving="Strain" if strain else "Stress",
                      Evolution="Linear", Value=_number(value))  # fmt: skip
    if finite:
        for name in GRADIENT[6:]:
            ET.SubElement(element, name, Driving="Strain", Evolution="Linear", Value="0")


def write_amitex_loading(path, segments, *, kinematics: str = "small", fields: bool = False):
    """A loading file of AMITEX_FFTP: ``segments`` lists the partial loadings, each a
    tuple (final time, number of increments, the six components at its end, which are
    strains (``controls``)). Components are morphoGen's: Voigt order with engineering
    shear strains in small strains, the macroscopic components (11, 22, 33, 23, 13, 12)
    of the gradient and of the nominal stress in finite strains, the gradient upper
    triangular (as morphoGen's). ``fields``: the stress and strain fields at every
    increment."""
    if kinematics not in ("small", "finite"):
        raise ValueError("kinematics: 'small' or 'finite'")
    root = ET.Element("Loading_Output")
    root.append(ET.Comment(f" morphoGen {__version__} "))
    output = ET.SubElement(root, "Output")
    flag = "1" if fields else "0"
    ET.SubElement(output, "vtk_StressStrain", Strain=flag, Stress=flag)
    for tag, (t_end, n_incr, values, controls) in enumerate(segments, start=1):
        loading = ET.SubElement(root, "Loading", Tag=str(tag))
        ET.SubElement(loading, "Time_Discretization", Discretization="Linear",
                      Nincr=str(int(n_incr)), Tfinal=_number(t_end))  # fmt: skip
        _components(loading, values, controls, kinematics == "finite")
    return _xml(Path(path), root)


def _segments(item) -> list[tuple]:
    """Partial loadings of a :class:`Loading` (one, from time 0 to 1) or of a
    :class:`LoadingPath` (one per segment)."""
    from morphogen.homogenization.core import Loading
    from morphogen.homogenization.paths import LoadingPath

    if isinstance(item, LoadingPath):
        if any(item.temperature):
            raise ValueError("AMITEX: the loadings written carry no change of temperature")
        return [(item.times[k + 1], item.increments[k], item.values[k + 1], item.controls)
                for k in range(len(item.times) - 1)]  # fmt: skip
    if isinstance(item, Loading):
        if item.temperature:
            raise ValueError("AMITEX: the loadings written carry no change of temperature")
        if item.physics != "elastic":
            raise ValueError("AMITEX: elastic loadings only")
        if item.kind == "mixed":
            controls = item.controls
        else:
            controls = (item.kind == "strain",) * 6
        return [(1.0, 1, item.value, controls)]
    raise TypeError(f"AMITEX: a Loading or a LoadingPath, got {type(item).__name__}")


def _stiffness_segments() -> list[tuple]:
    """The six unit strains (engineering shears), one partial loading each: in linear
    elasticity, the state at the end of loading k is the column k of the stiffness."""
    unit = np.eye(6)
    return [(float(k + 1), 1, unit[k], (True,) * 6) for k in range(6)]


@dataclass
class AmitexExport:
    """The files of the AMITEX runs of an image (:func:`write_amitex`): the directory,
    the name of the files, and the run of each loading (its loading file and the root of
    its results, relative to the directory)."""

    directory: Path
    name: str
    files: list[Path]
    runs: dict[str, dict]


def write_amitex(
    directory,
    image: VoxelImage,
    materials: dict,
    *,
    grains=None,
    orientations=None,
    loadings: dict | None = None,
    paths: dict | None = None,
    stiffness: bool | None = None,
    name: str = "amitex",
    fields: bool = False,
    tolerance: float | None = None,
    library: str = "",
    processes: int = 4,
    elastic_only: bool = False,
    kinematics: str = "small",
) -> AmitexExport:
    """The inputs of AMITEX_FFTP runs of a periodic voxel image of phases (see the
    module): ``materials`` (phase -> :class:`~morphogen.homogenization.Material`),
    crystals turned by the orientations of their grains (``grains``: label of each
    voxel, ``orientations``: Bunge angles of each label). The runs: the six unit strains
    of the apparent stiffness (``stiffness``; by default when no loading is given), the
    named ``loadings`` (:class:`~morphogen.homogenization.Loading`) and ``paths``
    (:class:`~morphogen.homogenization.LoadingPath`). ``fields``: stress and strain
    fields at every increment; ``tolerance``: the convergence criterion of AMITEX
    (``Default``: 1e-4); ``library``: the path of ``libUmatAmitex.so`` (empty: the
    native laws found through ``AMITEX_PATH``); ``processes``: MPI processes of the
    script. The nonlinear materials (von Mises and crystal plasticity) are MFront laws,
    which need loadings or paths (not the unit strains of the stiffness); ``elastic_only``
    exports their elasticity instead. ``kinematics``: small or finite strains (paths
    only; every phase but the voids an MFront law)."""
    from morphogen.homogenization.core import Loading
    from morphogen.homogenization.fft import reference_moduli
    from morphogen.homogenization.materials import material_table
    from morphogen.homogenization.paths import LoadingPath

    if kinematics not in ("small", "finite"):
        raise ValueError("kinematics: 'small' or 'finite'")
    finite = kinematics == "finite"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"name: letters, digits, '_', '-' and '.', got {name!r}")
    phase = np.asarray(image.data)
    if phase.ndim != 3 or phase.dtype.kind not in "iub":
        raise ValueError("AMITEX: a 3D image of integer phases")
    loadings = {str(k): Loading.from_spec(v) for k, v in (loadings or {}).items()}
    paths = {str(k): LoadingPath.from_spec(v) for k, v in (paths or {}).items()}
    if finite and (stiffness or loadings or not paths):
        raise ValueError(
            "AMITEX in finite strains: loading paths only (the unit strains of the stiffness "
            "and the loadings are linear problems)"
        )
    laws = []  # the phases of MFront laws: the nonlinear ones, all in finite strains
    for p in np.unique(phase):
        m = materials.get(int(p))
        if m is not None and not m.void and (finite or (m.nonlinear and not elastic_only)):
            laws.append(int(p))
    if laws and not finite:
        if stiffness or (stiffness is None and not loadings and not paths):
            raise ValueError(
                f"phases {laws}: the unit strains of the apparent stiffness need linear "
                "materials; give loading paths for the nonlinear materials (or elastic_only)"
            )
    table = material_table(phase, materials, "elastic", grains=grains, orientations=orientations)
    files: list[Path] = []
    # materials and zones of the voxels
    mat_ids = (table.stat + 1).reshape(phase.shape)
    zone_ids = np.ones(phase.size, dtype=np.int64)
    root = ET.Element("Materials")
    root.append(ET.Comment(f" morphoGen {__version__}: {len(table.phases)} materials "))
    used = np.unique(table.mat[table.mat >= 0])
    void = any(materials[p].void for p in table.phases)
    pool = table.matrices[used]
    if void:
        pool = np.concatenate([pool, np.zeros((1, 6, 6))])
    lam0, mu0 = reference_moduli(pool, table.isotropic) if pool.size else (0.0, 0.0)
    ET.SubElement(root, "Reference_Material", Lambda0=_number(lam0), Mu0=_number(mu0))
    description = []
    sources: list[str] = []
    parameters: dict[str, tuple] = {}
    for s, p in enumerate(table.phases):
        m = materials[p]
        sel = np.flatnonzero(table.stat == s)
        num = s + 1
        entry = {"material": num, "phase": int(p), "zones": 1}
        law = None
        if p in laws:
            # the MFront law of the material, compiled by the script (umat interface)
            if elastic_only and m.nonlinear:  # Hencky's elasticity (finite strains)
                m = dataclasses.replace(m, plasticity=None, crystal_plasticity=None, mfront=None,
                                        porous_plasticity=None, learned=None)  # fmt: skip
            user = user_behaviour(m, kinematics) if m.mfront is not None else None
            if user is None and m.expands:
                raise ValueError(
                    f"phase {p}: the thermal expansion of a phase written as an MFront law is "
                    "not exported to AMITEX_FFTP (its inputs carry no change of temperature)"
                )
            if user is None:
                behaviour = f"{identifier(name)}Phase{p}"
                source = write_mfront(m, directory / "mfront" / f"{behaviour}.mfront",
                                      name=behaviour, kinematics=kinematics)  # fmt: skip
                n_state = max(mfront_state_size(m, kinematics), 1)  # one at least, for Cast3M
                measure = mfront_strain_measure(m, kinematics)
            else:  # the user's law: its source, its parameters in the directory of the runs
                behaviour = user.name
                source = directory / "mfront" / user.source.name
                if source.name not in sources:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(user.source, source)
                n_state = max(user.internal_variables, 1)  # one at least, for Cast3M
                measure = user.strain_measure
                if parameters.setdefault(behaviour, m.mfront.parameters) != m.mfront.parameters:
                    raise ValueError(f"phase {p}: the parameters of {behaviour} are global")
                written = write_parameters(directory, behaviour, m.mfront.parameters)
                if written is not None and written not in files:
                    files.append(written)
            if source.name not in sources:
                sources.append(source.name)
                files.append(source)
            law = umat_function(behaviour, measure)
            orthotropic, head, rest = _umat_coefficients(m, user, kinematics)
            entry.update(mfront=f"mfront/{source.name}", library=UMAT_LIBRARY,
                         internal_variables=n_state)  # fmt: skip
        if m.void or (law is None and m.stiffness is None) or (law and not orthotropic):
            kind = "void" if m.void else "isotropic"
            root.append(ET.Comment(f" phase {p}: {kind}{', MFront law' if law else ''} "))
            if law:
                # the coefficients of the umat interface: YOUN, NU, RHO, ALPH, then the
                # other material properties of the behaviour
                el = ET.SubElement(root, "Material", numM=str(num), Lib=UMAT_LIBRARY, Law=law)
                coefficients = head + rest
            else:
                el = ET.SubElement(root, "Material", numM=str(num), Lib=library, Law="elasiso")
                coefficients = (0.0, 0.0) if m.void else m.lame
            for k, value in enumerate(coefficients, start=1):
                ET.SubElement(el, "Coeff", Index=str(k), Type="Constant", Value=_number(value))
            if law:
                for k in range(1, n_state + 1):
                    ET.SubElement(el, "IntVar", Index=str(k), Type="Constant", Value="0.")
            entry["law"] = law or "elasiso"
        else:
            ids = table.mat[sel]
            first = int(ids.min())
            zone_ids[sel] = ids - first + 1
            rotations = table.rotations[first : int(ids.max()) + 1]
            n_zones = len(rotations)
            what = "crystal, MFront law" if law else "crystal"
            root.append(ET.Comment(f" phase {p}: {what}, {n_zones} zones (grains) "))
            if law:
                # the coefficients of the umat interface: YG1, YG2, YG3, NU12, NU23, NU13,
                # G12, G23, G13, the axes V1 and V2 of each zone, RHO, ALP1, ALP2, ALP3,
                # then the other material properties of the behaviour
                el = ET.SubElement(root, "Material", numM=str(num), Lib=UMAT_LIBRARY, Law=law)
                constants = head
            else:
                el = ET.SubElement(root, "Material", numM=str(num), Lib=library, Law="elasaniso")
                constants = _orthotropic(m.stiffness, p)
            for k, value in enumerate(constants, start=1):
                ET.SubElement(el, "Coeff", Index=str(k), Type="Constant", Value=_number(value))
            for k, (row, axis) in enumerate([(r, a) for r in (0, 1) for a in range(3)]):
                target = directory / f"{name}_M{num}_N{row + 1}{'XYZ'[axis]}.bin"
                files.append(_write_bin(target, rotations[:, row, axis]))
                ET.SubElement(el, "Coeff", Index=str(10 + k), Type="Constant_Zone",
                              File=target.name, Format="binary")  # fmt: skip
            if law:
                for k, value in enumerate(rest, start=16):
                    ET.SubElement(el, "Coeff", Index=str(k), Type="Constant",
                                  Value=_number(value))  # fmt: skip
            for k in range(1, (n_state if law else 6) + 1):
                ET.SubElement(el, "IntVar", Index=str(k), Type="Constant", Value="0.")
            entry.update(law=law or "elasaniso", zones=n_zones)
            if grains is not None and orientations is not None and m.oriented("elastic"):
                labels = np.asarray(grains).ravel()[sel]
                order = np.argsort(ids, kind="stable")
                _, start = np.unique(ids[order], return_index=True)
                entry["grains"] = [int(g) for g in labels[order][start]]
        description.append(entry)
    files.append(write_amitex_field(directory / f"{name}_materials.vtk", mat_ids, image.spacing,
                                    image.origin, "MaterialId"))  # fmt: skip
    files.append(write_amitex_field(directory / f"{name}_zones.vtk",
                                    zone_ids.reshape(phase.shape), image.spacing, image.origin,
                                    "ZoneId"))  # fmt: skip
    files.append(_xml(directory / f"{name}_material.xml", root))
    # algorithm
    algo = ET.Element("Algorithm_Parameters")
    a = ET.SubElement(algo, "Algorithm", Type="Basic_Scheme")
    crit = "Default" if tolerance is None else _number(tolerance)
    ET.SubElement(a, "Convergence_Criterion", Value=crit)
    ET.SubElement(a, "Convergence_Acceleration", Value="true")
    mech = ET.SubElement(algo, "Mechanics")
    ET.SubElement(mech, "Filter", Type="Default")
    ET.SubElement(mech, "Small_Perturbations", Value="false" if finite else "true")
    files.append(_xml(directory / f"{name}_algorithm.xml", algo))
    # loadings
    runs: dict[str, dict] = {}
    if stiffness or (stiffness is None and not loadings and not paths):
        runs["stiffness"] = {"kind": "stiffness", "segments": _stiffness_segments()}
    for key, item in loadings.items():
        runs[str(key)] = {"kind": "loading", "segments": _segments(item)}
    for key, item in paths.items():
        runs[str(key)] = {"kind": "path", "segments": _segments(item)}
    lines = [
        "#!/bin/sh",
        f"# morphoGen {__version__}: the AMITEX_FFTP runs of {name} (set AMITEX_PATH for "
        "the native laws of the material file)",
        'cd "$(dirname "$0")" || exit 1',
        "mkdir -p results",
    ]
    if sources:
        lines += [
            "# the MFront laws of the materials (TFEL's mfront on the PATH)",
            f"(cd mfront && mfront --obuild --interface=umat {' '.join(sources)}) || exit 1",
        ]
    manifest_runs = {}
    for key, run in runs.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError(f"loading names: letters, digits, '_', '-' and '.', got {key!r}")
        loading = directory / f"{name}_{key}.xml"
        files.append(write_amitex_loading(loading, run["segments"], kinematics=kinematics,
                                          fields=fields))  # fmt: skip
        result = f"results/{name}_{key}"
        lines.append(
            f"mpirun -np ${{NP:-{int(processes)}}} amitex_fftp -nm {name}_materials.vtk "
            f"-nz {name}_zones.vtk -m {name}_material.xml -c {loading.name} "
            f"-a {name}_algorithm.xml -s {result}"
        )
        manifest_runs[key] = {
            "kind": run["kind"],
            "loading": loading.name,
            "results": result,
            "times": [float(seg[0]) for seg in run["segments"]],
        }
    script = directory / f"{name}_run.sh"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.append(script)
    manifest = {
        "morphogen": __version__,
        "name": name,
        "shape": [int(n) for n in phase.shape],
        "spacing": [float(v) for v in image.spacing],
        "origin": [float(v) for v in image.origin],
        "volume": float(np.prod(np.asarray(phase.shape) * np.asarray(image.spacing))),
        "kinematics": kinematics,
        "materials": description,
        "reference": {"lambda0": float(lam0), "mu0": float(mu0)},
        "runs": manifest_runs,
    }
    target = directory / f"{name}_amitex.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files.append(target)
    return AmitexExport(directory=directory, name=name, files=files, runs=manifest_runs)


# -- results ---------------------------------------------------------------------------------

#: A number written by Fortran without the letter of its exponent (0.49801817-310)
_FORTRAN = re.compile(r"^([+-]?(?:\d+\.\d*|\.\d+|\d+))([+-]\d+)$")


def _value(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        match = _FORTRAN.match(token)
        if match is None:
            raise ValueError(f"not a number: {token!r}") from None
        return float(f"{match.group(1)}e{match.group(2)}")


def _table(path) -> tuple[list[str], np.ndarray]:
    """The comment lines and the rows of numbers of an output file of AMITEX."""
    comments, rows = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            comments.append(text)
            continue
        rows.append([_value(t) for t in text.split()])
    if not rows:
        raise ValueError(f"{path}: no values")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f"{path}: rows of different lengths")
    return comments, np.array(rows)


@dataclass
class AmitexResult:
    """Averages over the unit cell, a material or a zone at the output increments of an
    AMITEX_FFTP run, in morphoGen's orders. In small strains, ``stress`` and ``strain``
    (Voigt 11, 22, 33, 23, 31, 12; engineering shear strains); in finite strains, the
    macroscopic components (11, 22, 33, 23, 13, 12) of the first Piola-Kirchhoff stress
    and of the displacement gradient, their nine components (row by row) in ``pk1`` and
    ``gradient``, the Cauchy stress and the Green-Lagrange strain (Voigt).
    ``stress_std``, ``strain_std``: the standard deviations over the voxels;
    ``iterations``: those of each increment (unit cell); ``extra``: the other columns
    (the internal variables of the zones)."""

    kinematics: str
    times: np.ndarray
    stress: np.ndarray
    strain: np.ndarray
    stress_std: np.ndarray
    strain_std: np.ndarray
    iterations: np.ndarray | None = None
    cauchy: np.ndarray | None = None
    green_lagrange: np.ndarray | None = None
    pk1: np.ndarray | None = None
    gradient: np.ndarray | None = None
    extra: np.ndarray | None = None


def _kinematics(comments, width: int, kinematics: str | None) -> str:
    if kinematics is not None:
        if kinematics not in ("small", "finite"):
            raise ValueError("kinematics: 'small' or 'finite'")
        return kinematics
    if any("xx,yy,zz,xy,xz,yz,yx,zx,zy" in c.replace(" ", "") for c in comments) or width >= 61:
        return "finite"
    return "small"


def _result(rows: np.ndarray, kinematics: str, *, unit_cell: bool) -> AmitexResult:
    """The quantities of rows of an output file: in small strains time, stress (6),
    strain (6) and their standard deviations; in finite strains time, Cauchy stress (6),
    first Piola-Kirchhoff stress (9), Green-Lagrange strain (6), displacement gradient
    (9) and their standard deviations; then, for the unit cell, the iterations."""
    width = rows.shape[1]
    times = rows[:, 0]
    if kinematics == "small":
        if width < 25:
            raise ValueError(f"AMITEX averages in small strains: 25 columns, got {width}")
        rest = rows[:, 25:]
        out = AmitexResult(
            kinematics, times, rows[:, 1:7][:, _FROM6], rows[:, 7:13][:, _FROM6],
            rows[:, 13:19][:, _FROM6], rows[:, 19:25][:, _FROM6],
        )  # fmt: skip
    else:
        if width < 61:
            raise ValueError(f"AMITEX averages in finite strains: 61 columns, got {width}")
        pk1, grad = rows[:, 7:16][:, _FROM9], rows[:, 22:31][:, _FROM9]
        rest = rows[:, 61:]
        out = AmitexResult(
            kinematics, times, pk1[:, _MACRO], grad[:, _MACRO],
            rows[:, 37:46][:, _FROM9][:, _MACRO], rows[:, 52:61][:, _FROM9][:, _MACRO],
            cauchy=rows[:, 1:7][:, _FROM6], green_lagrange=rows[:, 16:22][:, _FROM6],
            pk1=pk1, gradient=grad,
        )  # fmt: skip
    if unit_cell and rest.shape[1]:
        out.iterations = rest[:, 0].astype(int)
        rest = rest[:, 1:]
    if rest.shape[1]:
        out.extra = rest
    return out


def _results_path(prefix, suffix: str) -> Path:
    path = Path(prefix)
    return path if path.suffix == suffix else path.parent / f"{path.name}{suffix}"


def read_amitex(prefix, *, kinematics: str | None = None) -> AmitexResult:
    """The averages over the unit cell of an AMITEX_FFTP run (``<prefix>.std``, the root
    of its results); small or finite strains are recognized from the header or the
    number of columns, unless given."""
    path = _results_path(prefix, ".std")
    comments, rows = _table(path)
    return _result(rows, _kinematics(comments, rows.shape[1], kinematics), unit_cell=True)


def _grouped(path, kinematics: str | None) -> dict[int, AmitexResult]:
    """The averages of the materials or zones of a file whose increments have one row
    each, at the same time, numbered from 1."""
    comments, rows = _table(path)
    times = rows[:, 0]
    breaks = np.flatnonzero(np.diff(times) != 0) + 1
    sizes = np.diff(np.concatenate([[0], breaks, [len(times)]]))
    if np.any(sizes != sizes[0]):
        raise ValueError(f"{path}: increments with different numbers of rows")
    n = int(sizes[0])
    kin = _kinematics(comments, rows.shape[1], kinematics)
    return {k + 1: _result(rows[k::n], kin, unit_cell=False) for k in range(n)}


def read_amitex_materials(prefix, *, kinematics: str | None = None) -> dict[int, AmitexResult]:
    """The averages over each material (``<prefix>.mstd``), by material number."""
    return _grouped(_results_path(prefix, ".mstd"), kinematics)


def read_amitex_zones(prefix, material: int, *, kinematics: str | None = None):
    """The averages over each zone of a material (``<prefix>_<material>.zstd``), by zone
    number; their internal variables in ``extra``."""
    path = Path(prefix)
    path = path.parent / f"{path.name}_{int(material)}.zstd"
    return _grouped(path, kinematics)


@dataclass
class AmitexRun:
    """The runs of an export (:func:`write_amitex`) and their results: the description of
    the export, the averages of each run (``results``; runs without results are left
    out), the apparent stiffness of the unit strains and its properties."""

    manifest: dict
    results: dict[str, AmitexResult] = field(default_factory=dict)
    stiffness: np.ndarray | None = None
    properties: dict[str, float] = field(default_factory=dict)


def amitex_stiffness(result: AmitexResult, times=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)) -> np.ndarray:
    """The apparent stiffness (Voigt, engineering shears) of a run of the six unit strains:
    its column k is the stress at the end of the partial loading k (``times``)."""
    columns = []
    for t in times:
        hit = np.flatnonzero(np.isclose(result.times, t, rtol=1e-9, atol=1e-12))
        if hit.size == 0:
            raise ValueError(f"no output at the end of the unit strain at t = {t:g}")
        columns.append(result.stress[hit[-1]])
    return np.column_stack(columns)


def read_amitex_run(directory, name: str | None = None) -> AmitexRun:
    """The results of the runs of an export (``<name>_amitex.json`` in ``directory``;
    the only one there by default): the averages of each run whose results exist, and
    for the unit strains the apparent stiffness, the bulk modulus (``C_ij`` summed over
    the normal components, over 9), the shear modulus (``C66``, plane 12), the moduli of
    its isotropic projection and its engineering constants."""
    from morphogen.homogenization.apparent import isotropic_projection
    from morphogen.homogenization.postprocess import engineering_constants

    directory = Path(directory)
    if name is None:
        found = sorted(directory.glob("*_amitex.json"))
        if len(found) != 1:
            raise ValueError(f"{directory}: give the name of the export ({len(found)} found)")
        target = found[0]
    else:
        target = directory / f"{name}_amitex.json"
    manifest = json.loads(target.read_text(encoding="utf-8"))
    out = AmitexRun(manifest=manifest)
    for key, run in manifest["runs"].items():
        prefix = directory / run["results"]
        if not _results_path(prefix, ".std").exists():
            continue
        result = read_amitex(prefix, kinematics=manifest.get("kinematics", "small"))
        out.results[key] = result
        if run["kind"] == "stiffness":
            c = amitex_stiffness(result, run["times"])
            out.stiffness = c
            bulk, shear = isotropic_projection(c)
            out.properties.update(
                bulk_modulus=float(c[:3, :3].sum() / 9.0),
                shear_modulus=float(c[5, 5]),
                isotropic_bulk=float(bulk),
                isotropic_shear=float(shear),
                **engineering_constants(c),
            )
    return out
