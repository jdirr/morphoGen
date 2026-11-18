"""The phase-field fracture of a morphology of the workspace, a tool of the tab Fracture
of the interface: cracks that start, branch and merge in its phases along a loading path,
on its voxels or on a planar section of them
(:mod:`morphogen.fracture.phasefield`)."""

from __future__ import annotations

import numpy as np

from morphogen.gui.tools import Field, tool

COMPONENTS = (
    ("11", "tension along x"),
    ("22", "tension along y"),
    ("33", "tension along z"),
    ("12", "shear in the plane x-y"),
    ("23", "shear in the plane y-z"),
    ("31", "shear in the plane z-x"),
)
SPLITS = (
    ("spectral", "spectral: the positive principal strains (Miehe et al.)"),
    ("volumetric", "volumetric: dilatation and shear (Amor et al.)"),
    ("none", "none: all the energy breaks, as much in compression"),
)


@tool(
    "phase_field",
    "fracture",
    "Phase-field fracture of a morphology",
    "The brittle fracture of the phases of a morphology along a loading path, by the "
    "phase-field method on its voxels (periodic): a damage field regularizes the cracks "
    "over a length, the toughness of each phase is the energy of their unit area, and "
    "cracks start, branch and merge without being meshed (alternate minimization: the "
    "elastic problem by FFT, the damage under its irreversibility), in small strains or, "
    "for the rubbers and soft solids that stretch much before they break, in finite "
    "strains (the energy of the deformation gradient, nominal strains and stresses). The "
    "macroscopic stress against the strain with the strength of the morphology and its "
    "work of fracture, the damage field at the end, the regularized area of the cracks.",
    [
        Field("source", "item", label="morphology", required=True),
        Field(
            "materials",
            "materials",
            of="source",
            doc="isotropic elastic constants of the phases, or voids",
        ),
        Field(
            "toughness",
            "floats",
            [3e-5],
            label="toughness of the phases",
            doc="G_c of each phase, in the order of their numbers (the energy per unit "
            "area of crack; 0, or a phase beyond the list: a phase that does not break)",
        ),
        Field(
            "length",
            "float",
            label="regularization length",
            unit="length",
            doc="of the cracks (blank: two voxels, the least that the voxels resolve); "
            "the strength of a phase goes as sqrt(E G_c / length)",
        ),
        Field(
            "model",
            "choice",
            "AT1",
            choices=(
                ("AT1", "AT1: elastic up to a strength, cracks of finite width"),
                ("AT2", "AT2: damage from the first load"),
            ),
        ),
        Field("split", "choice", "spectral", label="energy that breaks", choices=SPLITS),
        Field(
            "kinematics",
            "choice",
            "small",
            choices=(
                ("small", "small strains"),
                ("finite", "finite strains (total Lagrangian; nominal strain and stress)"),
            ),
        ),
        Field(
            "energy",
            "choice",
            "hencky",
            label="strain energy of the phases",
            when={"kinematics": "finite"},
            doc="in finite strains, with the elastic constants of the phases at the natural "
            "state; a rubber breaks by its isochoric energy and its expansion (volumetric), "
            "or by all its energy (not split)",
            choices=(
                ("hencky", "elastic: the energy of Hencky of the logarithmic strain"),
                ("neo_hookean", "neo-Hookean (rubbers)"),
            ),
        ),
        Field(
            "planar",
            "bool",
            True,
            label="a planar section",
            doc="one layer of voxels through the middle of the morphology (normal to z; a "
            "plane strain): much faster than its volume",
        ),
        Field("voxels", "int", 32, label="voxels per edge"),
        Field("component", "choice", "11", label="loading", choices=COMPONENTS),
        Field(
            "amplitude",
            "float",
            0.01,
            label="strain at the end",
            doc="of the component of the loading; the other stresses are free",
        ),
        Field("increments", "int", 20),
    ],
    order=6,
    menu="Phase field",
)
def phase_field(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen.fracture.phasefield import PhaseField, PhaseFieldProblem
    from morphogen.fracture.phasefield_finite import FinitePhaseFieldProblem
    from morphogen.gui.tools.mechanics import _elastic
    from morphogen.homogenization.materials import materials_from_config
    from morphogen.image import VoxelImage

    n = p["voxels"]
    if not 8 <= n <= (256 if p["planar"] else 64):
        raise ValueError(
            "voxels per edge: from 8 to 256 for a planar section, to 64 for the volume"
        )
    increments = p["increments"]
    if not 2 <= increments <= 400:
        raise ValueError("increments: from 2 to 400")
    if not p["amplitude"] > 0.0:
        raise ValueError("strain at the end: positive")
    elastic = _elastic(p["source"], p.get("materials"))
    finite = p["kinematics"] == "finite"
    if finite and p["energy"] == "neo_hookean":  # rubbers of the constants of the phases
        law = {"hyperelasticity": {"model": "neo_hookean"}}
        elastic = {k: m if m == "void" else {**m, **law} for k, m in elastic.items()}
    materials = materials_from_config(elastic)
    phases = sorted(elastic)
    given = [float(g) for g in p["toughness"] or []]
    if any(g < 0.0 for g in given) or len(given) > len(phases):
        raise ValueError(f"toughness of the phases: at most one per phase ({phases}), positive")
    toughness = {k: g for k, g in zip(phases, given, strict=False) if g > 0.0}
    breakable = [k for k in toughness if not materials[k].void]
    if not breakable:
        raise ValueError("toughness of the phases: one solid phase at least must break")
    ctx.progress(0.03, "voxels")
    image = p["source"]["micro"].voxelize(shape=n)
    if p["planar"]:
        k = image.shape[2] // 2
        image = VoxelImage(np.asarray(image.data)[:, :, k : k + 1], image.spacing, periodic=True)
    h = float(image.spacing[0])
    length = p["length"] or 2.0 * h
    if not length >= h:
        raise ValueError(f"regularization length: at least a voxel ({h:g})")
    fracture = PhaseField({k: toughness[k] for k in breakable}, length, model=p["model"],
                          split=p["split"])  # fmt: skip
    problem = (FinitePhaseFieldProblem if finite else PhaseFieldProblem)(image, materials, fracture)
    path = {"times": [0.0, 1.0], "increments": increments,
            "strain": {p["component"]: [0.0, p["amplitude"]]}}  # fmt: skip

    def monitor(time, records) -> bool:
        ctx.progress(0.05 + 0.85 * float(time), f"step {len(records['t']) - 1} of {increments}")
        return False

    res = problem.run(path, fields=True, monitor=monitor)
    c = res.driven()
    strain, stress = res.strain[:, c], res.stress[:, c]
    rows = [
        {"time": float(t), "strain": float(e), "stress": float(s), "damage": float(d),
         "crack surface": float(g), "fracture energy": float(w)}
        for t, e, s, d, g, w in zip(res.times, strain, stress, res.damage, res.crack_surface,
                                    res.fracture_energy, strict=True)
    ]  # fmt: skip
    ctx.table(rows, "curve", "macroscopic curve of the loading")
    ctx.progress(0.92, "figures")
    fig = Figure(figsize=(9.4, 3.9), layout="constrained")
    left, right = fig.subplots(1, 2)
    left.plot(strain, stress, "o-", ms=3)
    left.set_xlabel(f"strain {p['component']}")
    left.set_ylabel(f"stress {p['component']}")
    left.grid(True, lw=0.4, alpha=0.5)
    damage = np.asarray(res.fields[float(res.times[-1])]["damage"])
    section = damage[:, :, damage.shape[2] // 2]
    lengths = np.asarray(image.size, dtype=float)
    shown = right.imshow(
        section.T,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        extent=(0.0, lengths[0], 0.0, lengths[1]),
        interpolation="nearest",
    )
    right.set_title("damage at the end (section x-y)", fontsize=9)
    fig.colorbar(shown, ax=right, shrink=0.85)
    ctx.figure(fig, "fracture", "curve of the loading and damage at the end")
    peak = int(np.argmax(np.abs(stress)))
    values = {
        "voxels": int(np.prod(image.shape)),
        "regularization length": float(length),
        "strength (largest stress)": res.strength(),
        "strain at the strength": float(strain[peak]),
        "stress at the end": float(stress[-1]),
        "work of the loading per unit volume": res.work(),
        "fracture energy per unit volume": float(res.fracture_energy[-1]),
        "regularized area of the cracks per unit volume": float(res.crack_surface[-1]),
        "mean damage of the phases that break": float(res.damage[-1]),
    }
    return {"values": values, "title": f"Phase-field fracture of {p['source']['title']}"}
