"""A treated sheet in its volume: voxels of graded elastoplastic phases for the solvers of
three-dimensional fields.

The finite elements of the sheets (:mod:`morphogen.laser.mechanics`) are those of a
membrane in plane stress whose layers share the strain of the plane: no stress across the
thickness, no layer that holds its neighbours back. Here the periodic cell of the sheet
treated all over is drawn in its volume: an image of voxels ``(n1, n2, layers)`` along
the two vectors of its lattice and its thickness, whose phases are classes of hardness,
each with the flow law of its hardness, and whose two faces are free: a gap of void
voxels between the sheet and its periodic image across the thickness (the first layer
at ``z = 0``: the face of the beam of a laser treatment). The solvers of
:mod:`morphogen.homogenization` then take it as any microstructure: loading paths by FFT
(:func:`volume_tension`: a tensile test along an axis of the cell, the other stresses
free), its phases, fields and statistics, a conforming mesh, the other physics.

The flow laws of the sheets (:class:`~morphogen.laser.materials.FlowLaw`) become the
isotropic hardening of von Mises plasticity
(:class:`~morphogen.homogenization.behaviours.Plasticity`: ``R0 + H p + Σ Q_i (1 -
exp(-b_i p))``): Voce's law is one of its terms; Ludwik's and Swift's power laws are
fitted over a range of plastic strain by a linear term and three exponentials (their
amplitudes by non-negative least squares, their rates the best of a few: within one or
two percent).
"""

from __future__ import annotations

import numpy as np

from morphogen.homogenization.behaviours import Plasticity
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage
from morphogen.laser.materials import LAWS, FlowLaw

#: The rates of the three exponentials fitted to a power law, over the range of plastic
#: strain: the fastest, the middle and the slowest are chosen among these
RATES = ((3000.0, 1000.0, 300.0), (150.0, 60.0, 25.0), (10.0, 5.0, 2.0))


def flow_plasticity(kind, sigma0: float, a: float, b: float, *, strain: float = 0.3) -> Plasticity:
    """The von Mises plasticity of a flow law of the sheets (``kind``: its name, or its
    index in :data:`~morphogen.laser.materials.LAWS`; ``sigma0``, ``a``, ``b``: its
    parameters): exact for Voce's law, fitted over the plastic strains up to ``strain``
    for the power laws (see the module; the rates of the exponentials: those of
    :data:`RATES` that leave the least error)."""
    from itertools import product

    from scipy.optimize import nnls

    name = kind if isinstance(kind, str) else LAWS[int(kind)]
    law = FlowLaw(name, float(sigma0), float(a), float(b))
    if name == "voce":
        return Plasticity(yield_stress=law.sigma0, isotropic=((law.a, law.b),))
    if not strain > 0:
        raise ValueError("strain: the range of plastic strain of the fit, positive")
    p = strain * np.linspace(0.0, 1.0, 400) ** 2  # denser at the start, where the law bends
    first = float(law.stress(0.0))
    target = law.stress(p)
    best = None
    for rates in product(*RATES):
        rates = np.asarray(rates) / strain
        basis = np.column_stack([p, *(1.0 - np.exp(-rate * p) for rate in rates)])
        amplitudes, _ = nnls(basis, target - first)
        error = float(np.abs((first + basis @ amplitudes) / target - 1.0).max())
        if best is None or error < best[0]:
            best = (error, amplitudes, rates)
    _, amplitudes, rates = best
    terms = tuple((float(q), float(r)) for q, r in zip(amplitudes[1:], rates, strict=True) if q > 0)
    return Plasticity(yield_stress=first, hardening=float(amplitudes[0]), isotropic=terms)


def hardening_stress(plasticity: Plasticity, p) -> np.ndarray:
    """The yield stress ``R0 + H p + Σ Q_i (1 - exp(-b_i p))`` of a plasticity at the
    cumulated plastic strains ``p``."""
    p = np.asarray(p, dtype=float)
    out = plasticity.yield_stress + plasticity.hardening * p
    for q, rate in plasticity.isotropic:
        out = out + q * (1.0 - np.exp(-rate * p))
    return out


def sheet_volume(
    hardness,
    material,
    spacing,
    thickness: float,
    *,
    classes: int = 8,
    gap: int = 2,
    depth: int | None = None,
    strain: float = 0.3,
) -> tuple[VoxelImage, dict[int, Material]]:
    """The volume of a treated sheet (see the module): ``hardness`` (HV; ``(n1, n2,
    layers)``, on a grid of the periodic cell of steps ``spacing`` (mm) along its two
    axes, its layers through the ``thickness`` (mm)) of a sheet of ``material`` (a
    :class:`~morphogen.laser.materials.SheetMaterial`). Returns the image of its phases
    and their materials: phase 0, the void of the ``gap`` voxels above the sheet (its
    free faces); phases 1, 2, ...: the classes of hardness present among ``classes`` of
    equal width, in increasing hardness, each with the elastic constants of the sheet and
    the plasticity of the flow law of the mean hardness of its voxels (``strain``: the
    range of the fit of power laws). ``depth``: voxels through each layer (by default as
    many as make the voxels nearest cubes); the first layer is at ``z = 0``. The hardness
    of each phase is in the metadata of the image (``"hardness"``), with the number of
    voxels through the sheet (``"solid_voxels"``)."""
    hv = np.asarray(hardness, dtype=float)
    if hv.ndim == 2:
        hv = hv[:, :, None]
    if hv.ndim != 3 or not np.all(np.isfinite(hv)) or hv.min() <= 0:
        raise ValueError("hardness: positive values (n1, n2, layers)")
    dx, dy = (float(v) for v in spacing)
    if not (dx > 0 and dy > 0 and thickness > 0):
        raise ValueError("spacing and thickness: positive (mm)")
    if not 1 <= int(classes) <= 254 or int(gap) < 1:
        raise ValueError(
            "classes: from 1 to 254; gap: at least 1 (it frees the faces of the sheet)"
        )
    layer = float(thickness) / hv.shape[2]
    depth = max(1, round(layer / min(dx, dy))) if depth is None else int(depth)
    if depth < 1:
        raise ValueError("depth: at least one voxel through each layer")
    low, high = float(hv.min()), float(hv.max())
    count = 1 if high - low <= 1e-9 * high else int(classes)
    index = np.minimum((count * (hv - low) / max(high - low, 1e-300)).astype(np.int64), count - 1)
    present = np.unique(index)
    phase = np.searchsorted(present, index) + 1
    # the hardness of a class: the mean of its voxels (a sheet of two hardnesses keeps both)
    means = np.bincount(phase.ravel(), weights=hv.ravel())[1:] / np.bincount(phase.ravel())[1:]
    kind, sigma0, a, b = material.strength.flow(means, young=material.young)
    materials: dict[int, Material] = {0: Material(void=True)}
    for k in range(len(present)):
        law = flow_plasticity(int(kind[k]), sigma0[k], a[k], b[k], strain=strain)
        materials[k + 1] = Material(young=material.young, poisson=material.poisson, plasticity=law)
    data = np.zeros((hv.shape[0], hv.shape[1], hv.shape[2] * depth + int(gap)), dtype=np.uint8)
    data[:, :, : hv.shape[2] * depth] = np.repeat(phase, depth, axis=2)
    metadata = {
        "hardness": {k + 1: float(mean) for k, mean in enumerate(means)},
        "thickness": float(thickness),
        "solid_voxels": int(hv.shape[2] * depth),
    }
    image = VoxelImage(data, (dx, dy, layer / depth), periodic=True, metadata=metadata)
    return image, materials


def case_volume(case, resolution: float | None = None, **options):
    """The volume of the periodic cell of a sheet treated all over by the pattern of a
    :class:`~morphogen.laser.design.SheetCase` (its lattice, or the ``vectors`` of its
    loading: two vectors at right angles, the axes of the image): the hardness that its
    laser path leaves on a grid of ``resolution`` mm (by default the element of the
    case) in each of its layers (the first under the beam, the last on the face of the
    beam ``bottom``), then :func:`sheet_volume` with its ``options``. A lattice that is not
    rectangular (honeycombs, tracks shifted from one to the next) has a rectangular cell
    of several of its cells: give its two vectors to the loading."""
    from morphogen.laser.design import CELL_REACH
    from morphogen.laser.patterns import LaserPath, make_pattern

    cell = case.cell()
    a1, a2 = np.asarray(cell.a1, dtype=float), np.asarray(cell.a2, dtype=float)
    l1, l2 = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
    if abs(a1 @ a2) > 1e-9 * l1 * l2:
        raise ValueError(
            "the cell of this pattern is not rectangular: give two vectors at right angles "
            "(of a cell of several of its cells) to the loading (vectors)"
        )
    step = float(case.element if resolution is None else resolution)
    n1, n2 = max(2, round(l1 / step)), max(2, round(l2 / step))
    u, v = (np.arange(n1) + 0.5) / n1, (np.arange(n2) + 0.5) / n2
    points = cell.origin + u[:, None, None] * a1 + v[None, :, None] * a2
    params = dict(case.pattern)
    kind = params.pop("kind")
    path = (
        LaserPath([])
        if kind == "none"
        else make_pattern(kind, cell.box(CELL_REACH), beam=case.beam, bottom=case.bottom,
                          pause=case.pause, cover=True, **params)
    )  # fmt: skip
    treated = case.laser_treatment().apply(path, points.reshape(-1, 2))
    hardness = treated.hardness.reshape(n1, n2, -1)
    return sheet_volume(hardness, case.material, (l1 / n1, l2 / n2), case.specimen.thickness,
                        **options)  # fmt: skip


def volume_tension(image: VoxelImage, materials: dict, *, axis: int = 0, strain: float = 0.05,
                   increments: int = 20, kinematics: str = "small", **options) -> dict:  # fmt: skip
    """The tensile test of the volume of a sheet along the axis ``axis`` of its cell (0 or
    1), up to the macroscopic ``strain`` in ``increments`` steps, the other stresses of
    its plane free (:func:`~morphogen.homogenization.apparent.homogenize`, with its
    ``options``): the ``strain`` along the axis, the ``stress`` of the sheet (the mean
    stress of the cell over the fraction of its height that the sheet fills), the strain
    of its ``width`` and the path ``result`` (fields, phases). The strains of the cell
    across the sheet are those of its gap, which nothing holds (left free, they make the
    problem singular): they are prescribed, and the faces of the sheet stay free, its
    thickness changing in the gap. ``kinematics="finite"``: finite strains (the flow laws
    in the logarithmic strains); the strain is then the elongation over the initial
    length and the stress the force over the initial section, those of a tensile test
    and of the membrane of the sheets."""
    from morphogen.homogenization.apparent import homogenize

    if axis not in (0, 1):
        raise ValueError("axis: 0 or 1, in the plane of the sheet")
    solid = image.metadata.get("solid_voxels")
    if not solid:
        solid = int(np.any(np.asarray(image.data) > 0, axis=(0, 1)).sum())
    component = ("11", "22")[axis]
    still = [0.0, 0.0]
    path = {"times": [0.0, 1.0], "increments": int(increments),
            "strain": {component: [0.0, float(strain)], "33": still, "23": still,
                       "13": still}}  # fmt: skip
    nonlinear = {**(options.pop("nonlinear", None) or {}), "kinematics": kinematics}
    result = homogenize(image, materials, properties=[], paths={"tension": path},
                        nonlinear=nonlinear, **options)  # fmt: skip
    result = result["tension.periodic"]
    return {
        "strain": result.strain[:, axis],
        "stress": result.stress[:, axis] * image.shape[2] / solid,
        "width": result.strain[:, 1 - axis],
        "result": result,
    }
