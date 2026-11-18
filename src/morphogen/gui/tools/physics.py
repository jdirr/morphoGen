"""The tools of the tab Physics of the interface: the physics of the homogenization
module beyond mechanics, on a morphology of the workspace — the transport of a scalar
potential (heat, electricity, dielectrics, magnetism, diffusion), the flow of a fluid
through its pores with the sound absorbed by a layer of it, the dynamic moduli of a
viscoelastic composite, and the band structure of light and of sound in a periodic
morphology (photonic and sonic crystals). Each runs the functions of
:mod:`morphogen.homogenization` that the guide to the physics (docs/physics.md)
describes, on the voxels of the morphology."""

from __future__ import annotations

import numpy as np

from morphogen.gui.tools import Field, tab, tool

tab(
    "physics",
    "Physics",
    "The other physics of a morphology of the workspace, on its voxels: its transport "
    "properties (thermal and electrical conductivities, permittivity, permeability, "
    "diffusivity) between their bounds, the flow through its pores (Darcy's permeability) "
    "with the acoustic parameters and the sound absorption of a layer of it, the "
    "storage and loss moduli of a viscoelastic composite over a range of frequencies, and "
    "the band structures of light and of sound in a periodic morphology.",
    order=62,
)

TRANSPORTS = (
    ("thermal", "heat conduction (conductivity)"),
    ("electrical", "electrical conduction (conductivity)"),
    ("dielectric", "dielectrics (permittivity)"),
    ("magnetic", "magnetostatics (permeability)"),
    ("diffusion", "diffusion (diffusivity)"),
)
DIRECTIONS = (
    ("mean", "the mean of the axes (an isotropic medium)"),
    ("x", "sound along x"),
    ("y", "sound along y"),
    ("z", "sound along z"),
)
WAVES = (
    ("electromagnetic", "light: electromagnetic waves (the permittivities of the phases)"),
    ("acoustic", "sound: the pressure of fluids (their densities and bulk moduli)"),
)
INCIDENCES = (
    ("normal", "normal: a plane wave across the layer"),
    ("oblique", "oblique: a plane wave at an angle from the normal"),
    ("diffuse", "diffuse field: every angle (Paris' average)"),
)


def _image(p: dict, lowest: int = 4, highest: int = 256):
    """The voxels of the morphology of the form."""
    n = p["voxels"]
    if not lowest <= n <= highest:
        raise ValueError(f"voxels per edge: from {lowest} to {highest}")
    return p["source"]["micro"].voxelize(shape=n)


def _tensor_rows(tensor) -> list[dict]:
    labels = ("1", "2", "3")
    return [
        {"": a, **{b: float(tensor[i, j]) for j, b in enumerate(labels)}}
        for i, a in enumerate(labels)
    ]


@tool(
    "transport",
    "physics",
    "Transport properties",
    "The apparent coefficient of a transport phenomenon of a scalar potential and its "
    "tensor, by FFT on the voxels of a morphology (periodic conditions): heat or electrical "
    "conduction, dielectrics, magnetostatics, diffusion — the same problem, div(a grad u) = "
    "0, with the coefficient a of each phase. A zero coefficient makes a phase inert (an "
    "insulator, an impermeable inclusion). Against the bounds of Wiener and of Hashin and "
    "Shtrikman for the phase fractions.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("physics", "choice", "thermal", choices=TRANSPORTS),
        Field(
            "coefficients",
            "floats",
            [1.0, 10.0],
            label="coefficients of the phases",
            doc="one per phase of the morphology, in the order of their numbers (0: a "
            "phase inert for this physics)",
        ),
        Field("voxels", "int", 32, label="voxels per edge"),
    ],
    order=1,
    menu="Transport",
)
def transport(p: dict, ctx) -> dict:
    from morphogen.analysis.basic import phase_fractions
    from morphogen.homogenization import Material, bounds, homogenize
    from morphogen.homogenization.apparent import TRANSPORT_PROPERTIES
    from morphogen.homogenization.materials import TRANSPORT

    ctx.progress(0.05, "voxels")
    image = _image(p)
    fractions = phase_fractions(image)
    phases = sorted(fractions)
    coefficients = [float(c) for c in p["coefficients"] or []]
    if len(coefficients) != len(phases):
        raise ValueError(
            f"coefficients of the phases: one per phase (the phases {phases} of the voxels)"
        )
    if min(coefficients) < 0.0 or max(coefficients) <= 0.0:
        raise ValueError("coefficients of the phases: non-negative, one at least positive")
    physics = p["physics"]
    scalar, tensor_name = TRANSPORT_PROPERTIES[physics]
    materials = {
        k: Material(**{TRANSPORT[physics]: c}) for k, c in zip(phases, coefficients, strict=True)
    }
    ctx.progress(0.2, "solving")
    out = homogenize(image, materials, properties=[scalar, tensor_name])
    tensor = np.asarray(out[f"{tensor_name}.periodic"]["tensor"])
    ctx.table(_tensor_rows(tensor), tensor_name, f"{tensor_name.replace('_', ' ')} (periodic)")
    values = {"voxels": int(np.prod(image.shape))}
    values.update({f"fraction of phase {k}": float(f) for k, f in fractions.items()})
    values[f"apparent {scalar.replace('_', ' ')}"] = float(out[f"{scalar}.periodic"].value)
    values.update({f"along {'xyz'[a]}": float(tensor[a, a]) for a in range(3)})
    try:
        limits = bounds(fractions, materials).get(scalar, {})
    except (ValueError, KeyError):
        limits = {}
    names = {
        "reuss": "Wiener lower bound (series)",
        "hs_lower": "Hashin-Shtrikman lower bound",
        "hs_upper": "Hashin-Shtrikman upper bound",
        "voigt": "Wiener upper bound (parallel)",
    }
    values.update({names[k]: float(limits[k]) for k in names if k in limits})
    return {"values": values, "title": f"{scalar.replace('_', ' ')} of {p['source']['title']}"}


@tool(
    "sound",
    "physics",
    "Flow and sound absorption",
    "The flow of a fluid through the pores of a morphology and the sound in them: the "
    "permeability of Darcy's law (a Stokes flow, by the lattice Boltzmann method on the "
    "voxels) and the parameters of the equivalent fluid of the rigid frame — porosity, "
    "resistivity, tortuosity, viscous and thermal characteristic lengths, thermal "
    "permeability — then the absorption coefficient of a layer of the material on a rigid "
    "wall, in air (the model of Johnson, Champoux, Allard and Lafarge): at normal "
    "incidence, at an angle (the wave refracted in the layer, whose frame along it is "
    "that of the two other axes when the sound goes along one) or in a diffuse field.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("pores", "int", 0, label="phase of the pores", doc="the fluid fills it"),
        Field("voxels", "int", 40, label="voxels per edge"),
        Field(
            "unit",
            "float",
            1e-3,
            label="unit of length of the morphology",
            unit="m",
            doc="in metres (1e-3: a morphology in millimetres): the parameters are given "
            "in SI units",
        ),
        Field("thickness", "float", 0.03, label="thickness of the layer", unit="m"),
        Field(
            "frequencies",
            "floats",
            [100.0, 6000.0],
            label="range of frequencies",
            unit="Hz",
            doc="the lowest and the highest; sixty between them",
        ),
        Field("direction", "choice", "mean", label="sound", choices=DIRECTIONS),
        Field("incidence", "choice", "normal", choices=INCIDENCES),
        Field(
            "angle",
            "float",
            45.0,
            unit="degree",
            when={"incidence": "oblique"},
            doc="of the wave from the normal of the layer",
        ),
    ],
    order=2,
    menu="Flow and acoustics",
)
def sound(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen.homogenization.acoustics import (
        AIR,
        absorption,
        acoustic_parameters,
        along_axis,
        equivalent_fluid,
    )

    scale = p["unit"]
    if not scale > 0.0:
        raise ValueError("unit of length of the morphology: positive (metres)")
    if not p["thickness"] > 0.0:
        raise ValueError("thickness of the layer: positive")
    band = sorted(float(f) for f in p["frequencies"] or [])
    if len(band) != 2 or not 0.0 < band[0] < band[1]:
        raise ValueError("range of frequencies: the lowest and the highest, positive")
    ctx.progress(0.05, "voxels")
    image = _image(p)
    if not np.any(image.data == p["pores"]):
        raise ValueError(f"phase of the pores: no voxel of the phase {p['pores']}")
    ctx.progress(0.15, "flow through the pores")
    res = acoustic_parameters(image, pores=p["pores"], tol=1e-4)
    axis = {"mean": None, "x": 0, "y": 1, "z": 2}[p["direction"]]
    six = along_axis(res, axis)
    if not (np.isfinite(six["tortuosity"]) and six["permeability"] > 0.0):
        raise ValueError(
            "the pores do not let the fluid through along this direction (no path): choose "
            "the phase of the pores, another direction or more voxels"
        )
    # in SI units: lengths times the unit, permeabilities times its square
    for name, power in (("permeability", 2), ("viscous_length", 1), ("thermal_length", 1),
                        ("thermal_permeability", 2)):  # fmt: skip
        six[name] *= scale**power
    ctx.progress(0.8, "absorption")
    f = np.geomspace(band[0], band[1], 60)
    normal = absorption(f, p["thickness"], **six)
    kind = p["incidence"] or "normal"
    if kind == "normal":
        layer, how = normal, "normal incidence"
    else:
        transverse = None
        if axis is not None:  # the frame along the layer: the mean of its two other axes
            others = [along_axis(res, a) for a in range(3) if a != axis]
            transverse = dict(six)
            for name, power in (("permeability", 2), ("viscous_length", 1)):
                transverse[name] = 0.5 * (others[0][name] + others[1][name]) * scale**power
            transverse["tortuosity"] = 0.5 * (others[0]["tortuosity"] + others[1]["tortuosity"])
            if not (np.isfinite(transverse["tortuosity"]) and transverse["permeability"] > 0.0):
                transverse = None  # no path along the layer: as across it
        if kind == "oblique":
            angle = float(p["angle"])
            if not 0.0 <= angle < 90.0:
                raise ValueError("angle: from 0 to 90 degrees from the normal")
            layer = absorption(f, p["thickness"], incidence=np.radians(angle),
                               transverse=transverse, **six)  # fmt: skip
            how = f"incidence at {angle:g}°"
        else:
            layer = absorption(f, p["thickness"], incidence="diffuse", transverse=transverse,
                               **six)  # fmt: skip
            how = "diffuse field"
    medium = equivalent_fluid(f, **six)
    rows = [
        {"axis": "xyz"[a], "permeability (m²)": res[f"permeability{a + 1}"] * scale**2,
         "tortuosity": res[f"tortuosity{a + 1}"],
         "viscous length (m)": res[f"viscous_length{a + 1}"] * scale}
        for a in range(3)
    ]  # fmt: skip
    ctx.table(rows, "parameters", "parameters along the axes")
    reduced = normal["surface_impedance"] / AIR.impedance
    ctx.table(
        [
            {"frequency (Hz)": float(x), "absorption": float(a),
             "absorption at normal incidence": float(b), "Re Zs / Z0": float(z.real),
             "Im Zs / Z0": float(z.imag)}
            for x, a, b, z in zip(f, layer["absorption"], normal["absorption"], reduced,
                                  strict=True)
        ],
        "absorption",
        f"absorption of the layer ({how}, rigid backing; Zs: at normal incidence)",
    )  # fmt: skip
    fig = Figure(figsize=(5.6, 3.8), layout="constrained")
    ax = fig.subplots()
    ax.semilogx(f, layer["absorption"], color="C0", lw=1.6, label=how)
    if kind != "normal":
        ax.semilogx(f, normal["absorption"], color="0.5", lw=1.0, ls="--",
                    label="normal incidence")  # fmt: skip
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("absorption coefficient")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(band[0], band[1])
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_title(f"{1e3 * p['thickness']:g} mm on a rigid wall, {how}", fontsize=10)
    ctx.figure(fig, "absorption", "sound absorption of the layer")
    values = {
        "voxels": int(np.prod(image.shape)),
        "porosity": six["porosity"],
        "permeability (m²)": six["permeability"],
        "resistivity (Pa s/m²)": medium["resistivity"],
        "tortuosity": six["tortuosity"],
        "viscous length (m)": six["viscous_length"],
        "thermal length (m)": six["thermal_length"],
        "thermal permeability (m²)": six["thermal_permeability"],
        "mean absorption over the range": float(np.mean(layer["absorption"])),
        "highest absorption": float(np.max(layer["absorption"])),
        "at the frequency (Hz)": float(f[int(np.argmax(layer["absorption"]))]),
    }
    return {"values": values, "title": f"Flow and sound in {p['source']['title']}"}


@tool(
    "dynamic",
    "physics",
    "Dynamic moduli",
    "The storage and loss moduli of a composite with a viscoelastic phase over a range of "
    "frequencies, by the correspondence principle: the complex moduli of the Prony series "
    "of the phase (a generalized Maxwell model) and the elastic constants of the others "
    "give the complex stiffness of the voxels at each frequency (FFT, periodic). The "
    "elastic constants of the viscoelastic phase are its instantaneous ones.",
    [
        Field("source", "item", label="morphology", required=True),
        Field(
            "materials",
            "materials",
            of="source",
            doc="isotropic elastic constants of the phases, or voids",
        ),
        Field(
            "phase",
            "int",
            label="viscoelastic phase",
            doc="blank: the first phase that is not a void",
        ),
        Field(
            "shear",
            "floats",
            [0.5, 0.01, 0.3, 1.0],
            label="Prony series of the shear modulus",
            doc="pairs g, τ: the relative moduli (their sum below 1) and the times",
        ),
        Field(
            "bulk",
            "floats",
            label="Prony series of the bulk modulus",
            doc="pairs k, τ (blank: an elastic bulk response)",
        ),
        Field(
            "frequencies",
            "floats",
            [0.01, 1000.0],
            label="range of angular frequencies",
            doc="the lowest and the highest (rad per unit of time)",
        ),
        Field("count", "int", 8, label="number of frequencies"),
        Field("voxels", "int", 20, label="voxels per edge"),
    ],
    order=3,
    menu="Dynamics and waves",
)
def dynamic(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen.gui.tools.mechanics import _elastic
    from morphogen.homogenization import homogenize
    from morphogen.homogenization.materials import materials_from_config

    def pairs(values, name):
        values = [float(v) for v in values or []]
        if len(values) % 2:
            raise ValueError(f"Prony series of the {name} modulus: pairs (modulus, time)")
        return [[values[i], values[i + 1]] for i in range(0, len(values), 2)]

    elastic = _elastic(p["source"], p.get("materials"))
    phase = p.get("phase")
    if phase is None:
        phase = next((k for k, m in elastic.items() if m != "void"), None)
    if phase not in elastic or elastic[phase] == "void":
        raise ValueError(f"viscoelastic phase: {phase} is a void or no phase of the morphology")
    series = {"shear": pairs(p.get("shear"), "shear"), "bulk": pairs(p.get("bulk"), "bulk")}
    series = {k: v for k, v in series.items() if v}
    if not series:
        raise ValueError("Prony series: at least one term (shear or bulk)")
    configs = dict(elastic)
    configs[phase] = {**elastic[phase], "viscoelasticity": series}
    materials = materials_from_config(configs)
    band = sorted(float(f) for f in p["frequencies"] or [])
    if len(band) != 2 or not 0.0 < band[0] < band[1]:
        raise ValueError("range of angular frequencies: the lowest and the highest, positive")
    count = p["count"]
    if not 2 <= count <= 60:
        raise ValueError("number of frequencies: from 2 to 60")
    ctx.progress(0.05, "voxels")
    image = _image(p, highest=128)
    omegas = np.geomspace(band[0], band[1], count)
    rows = []
    for i, omega in enumerate(omegas):
        ctx.progress(0.1 + 0.8 * i / count, f"frequency {i + 1} of {count}")
        res = homogenize(image, materials, properties=["dynamic_stiffness"],
                         frequencies=[float(omega)])["dynamic_stiffness.periodic"]  # fmt: skip
        rows.append({
            "angular frequency": float(omega),
            "bulk storage": res["bulk_storage_1"],
            "bulk loss": res["bulk_loss_1"],
            "shear storage": res["shear_storage_1"],
            "shear loss": res["shear_loss_1"],
            "tan δ (shear)": res["shear_tan_delta_1"],
            "Young storage (x)": res["young_storage_1"],
            "Young loss (x)": res["young_loss_1"],
        })  # fmt: skip
    ctx.table(rows, "moduli", "storage and loss moduli of the composite")
    fig = Figure(figsize=(9.2, 3.6), layout="constrained")
    left, right = fig.subplots(1, 2)
    left.loglog(omegas, [r["shear storage"] for r in rows], "o-", label="storage G'")
    left.loglog(omegas, [r["shear loss"] for r in rows], "s--", label="loss G''")
    left.set_xlabel("angular frequency")
    left.set_ylabel("shear modulus")
    left.legend(frameon=False)
    right.semilogx(omegas, [r["tan δ (shear)"] for r in rows], "o-", color="C3")
    right.set_xlabel("angular frequency")
    right.set_ylabel("loss factor tan δ")
    for ax in (left, right):
        ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ctx.figure(fig, "moduli", "shear moduli and loss factor of the composite")
    peak = int(np.argmax([r["tan δ (shear)"] for r in rows]))
    values = {
        "voxels": int(np.prod(image.shape)),
        "shear storage at the lowest frequency": rows[0]["shear storage"],
        "shear storage at the highest frequency": rows[-1]["shear storage"],
        "highest loss factor (shear)": rows[peak]["tan δ (shear)"],
        "at the angular frequency": rows[peak]["angular frequency"],
    }
    return {"values": values, "title": f"Dynamic moduli of {p['source']['title']}"}


@tool(
    "waves",
    "physics",
    "Band structure of light and sound",
    "The Bloch waves of a periodic morphology, in three dimensions: light in a photonic "
    "crystal (the vector waves of Maxwell's equations, by plane waves on a grid whose "
    "cells cut by the interfaces take an averaged permittivity) or sound in a sonic "
    "crystal (the pressure of fluids, finite elements on the voxels; a phase of zero "
    "density is left out: rigid scatterers). The frequencies along the edges of the "
    "Brillouin zone of the cell (Γ–X–M–Γ–R; Γ–X–M–Γ in the plane of a thin one), in units "
    "of c / L (the speed of light for unit coefficients, or of the sound of the first "
    "fluid, over the edge of the cell along x), and the band gaps: the ranges of "
    "frequencies in which no wave propagates along the path.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("waves", "choice", "electromagnetic", choices=WAVES),
        Field(
            "permittivity",
            "floats",
            [1.0, 11.56],
            label="permittivities of the phases",
            when={"waves": "electromagnetic"},
            doc="relative, in the order of their numbers (1: the vacuum, air)",
        ),
        Field(
            "density",
            "floats",
            [1.2, 0.0],
            label="densities of the phases",
            when={"waves": "acoustic"},
            doc="in the order of their numbers; 0: a rigid phase, which holds no sound",
        ),
        Field(
            "bulk",
            "floats",
            [1.42e5, 1.0],
            label="bulk moduli of the phases",
            when={"waves": "acoustic"},
        ),
        Field(
            "resolution",
            "int",
            12,
            label="plane waves (or voxels) per edge",
            doc="light: the plane waves; sound: the voxels, cut into tetrahedra",
        ),
        Field(
            "coarsen",
            "int",
            3,
            label="voxels per plane wave",
            when={"waves": "electromagnetic"},
            doc="the morphology is drawn that many times finer than the plane waves, and the "
            "cells that its interfaces cut are averaged (1: none)",
        ),
        Field(
            "path",
            "str",
            doc="corners of the Brillouin zone among G (Γ), X, Y, Z, M, R (blank: G X M G R; "
            "G X M G in the plane)",
        ),
        Field("points", "int", 4, doc="wave vectors per segment of the path"),
        Field("bands", "int", 8),
    ],
    order=4,
    menu="Dynamics and waves",
)
def waves(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen.gui.api import _planar
    from morphogen.gui.tools.stability import _corners
    from morphogen.homogenization.bloch import brillouin_path, scalar_band_structure
    from morphogen.homogenization.photonic import electromagnetic_band_structure
    from morphogen.image import VoxelImage

    item = p["source"]
    micro = item["micro"]
    if not micro.domain.periodic:
        raise ValueError(f"{item['title']} is not periodic: the Bloch waves are those of a cell")
    light = (p["waves"] or "electromagnetic") == "electromagnetic"
    n, coarsen = int(p["resolution"]), int(p["coarsen"] or 1) if light else 1
    points, bands = int(p["points"]), int(p["bands"])
    if not (4 <= n <= 32 and 1 <= coarsen <= 6):
        raise ValueError("plane waves per edge: from 4 to 32; voxels per plane wave: from 1 to 6")
    if not (1 <= points <= 24 and 1 <= bands <= 40):
        raise ValueError("points: from 1 to 24 per segment; bands: from 1 to 40")
    planar = _planar(micro)
    corners = _corners(p["path"], planar)
    ctx.progress(0.03, "voxels")
    lengths = np.asarray(micro.domain.lengths, dtype=float)
    edge = float(lengths[:2].min() if planar else lengths.min())
    shape = np.maximum(1, np.rint(lengths / edge * n)).astype(int) * coarsen
    if planar:
        shape[2] = 1
    data = np.asarray(micro.voxelize(shape=tuple(int(v) for v in shape)).data)
    spacing = lengths / shape
    if planar:
        spacing[2] = spacing[0]  # one layer: no wave vector across it
    image = VoxelImage(data, tuple(spacing), periodic=True)
    present = [int(k) for k in np.unique(data)]

    def per_phase(values, name):
        values = [float(v) for v in values or []]
        if len(values) <= max(present):
            raise ValueError(f"{name}: one per phase (the phases {present})")
        return {k: values[k] for k in present}

    ks, distance, labels = brillouin_path(image.size, corners, points)
    if light:
        eps = per_phase(p["permittivity"], "permittivities of the phases")
        if any(not v > 0 for v in eps.values()):
            raise ValueError("permittivities of the phases: positive (1 for the vacuum)")
        speed = 1.0
    else:
        rho = per_phase(p["density"], "densities of the phases")
        bulk = per_phase(p["bulk"], "bulk moduli of the phases")
        if any(v < 0 for v in rho.values()) or any(not v > 0 for v in bulk.values()):
            raise ValueError("densities: not negative (0: rigid); bulk moduli: positive")
        fluids = [k for k in present if rho[k] > 0]
        if not fluids:
            raise ValueError("densities of the phases: one fluid at least")
        a = {k: (1.0 / rho[k] if rho[k] > 0 else 0.0) for k in present}
        b = {k: 1.0 / bulk[k] for k in present}
        speed = float(np.sqrt(bulk[fluids[0]] / rho[fluids[0]]))
    frequencies = []
    from morphogen.homogenization.bloch import BandStructure

    for seg in range(len(corners) - 1):  # one segment at a time, for the progress
        ctx.progress(
            0.05 + 0.85 * seg / (len(corners) - 1), f"Bloch waves {corners[seg]}–{corners[seg + 1]}"
        )
        chunk = ks[(0 if seg == 0 else 1 + seg * points) : 1 + (seg + 1) * points]
        if light:
            part = electromagnetic_band_structure(image, eps, chunk, bands=bands, coarsen=coarsen)
        else:
            part = scalar_band_structure(image, a, b, chunk, bands=bands)
        frequencies.append(part.frequencies)
    result = BandStructure(ks, distance, np.vstack(frequencies), labels)
    length = float(lengths[0])
    scale = length / (2 * np.pi * speed)
    ctx.progress(0.92, "figures")
    fig = Figure(figsize=(5.6, 4.4), layout="constrained")
    ax = fig.subplots()
    result.plot(ax, scale=scale, ylabel="ω L / 2π c")
    ax.set_title("light" if light else f"sound (c of phase {fluids[0]})", fontsize=10)
    ctx.figure(fig, "bands", "band structure")
    gaps = result.gaps()
    rows = [
        {"bands": f"{k} – {k + 1}", "from": low * scale, "to": high * scale,
         "width (% of its middle)": 200.0 * (high - low) / (high + low)}
        for k, low, high in gaps
    ]  # fmt: skip
    if rows:
        ctx.table(rows, "gaps", "band gaps along the path (ω L / 2π c)")
    table = []
    for d, k, row in zip(distance, ks, result.frequencies, strict=True):
        entry = {"distance": float(d), "kx": float(k[0]), "ky": float(k[1]), "kz": float(k[2])}
        entry.update({f"band {i + 1}": float(v * scale) for i, v in enumerate(row)})
        table.append(entry)
    ctx.table(table, "bands", "the bands at the wave vectors (ω L / 2π c)")
    values = {
        "voxels": int(np.prod(shape)),
        "wave vectors": len(ks),
        "band gaps along the path": len(gaps),
        "highest frequency (ω L / 2π c)": float(result.frequencies.max() * scale),
    }
    if gaps:
        k, low, high = max(gaps, key=lambda g: (g[2] - g[1]) / (g[2] + g[1]))
        values["widest gap, above the band"] = int(k)
        values["widest gap, from (ω L / 2π c)"] = low * scale
        values["widest gap, to (ω L / 2π c)"] = high * scale
        values["widest gap (% of its middle)"] = 200.0 * (high - low) / (high + low)
    kind = "Light" if light else "Sound"
    return {"values": values, "title": f"{kind} in {item['title']}: band structure"}
