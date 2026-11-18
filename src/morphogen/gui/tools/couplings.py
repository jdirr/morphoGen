"""The tools of the tab Couplings: transport, chemistry and mechanics
(:mod:`morphogen.coupling`): the transient diffusion of a species (moisture, oxygen, a
solvent) into a morphology of the workspace, with its effective diffusivity; the thermal
oxidation of an epoxy limited by the diffusion of oxygen (the closed-loop scheme of Colin
and Verdu with the constants of Ernault et al. 2019), with the properties of its
oxidized layer, through the thickness of a plate or in the volume of a morphology
(around its fibres or fillers); and the bending of hygromorph multilayers as moisture
enters them (Mou, Dirrenberger & Le Duigou 2025), whose layers may be homogenized from
morphologies of the workspace (:meth:`~morphogen.coupling.Layer.from_microstructure`).

The FEniCSx solvers of :mod:`morphogen.coupling.fenicsx` (meshes swelling in 3D) need
dolfinx, which runs on Linux (or WSL) only; these tools use the voxel and plate solvers,
which run everywhere."""

from __future__ import annotations

import math

import numpy as np

from morphogen.gui.tools import Field, tab, tool

tab(
    "couplings",
    "Couplings",
    "Couplings of transport, chemistry and mechanics: transient diffusion in a morphology "
    "of the workspace and its effective diffusivity, the thermal oxidation of an epoxy "
    "and its oxidized layer (a plate, or the volume of a morphology), the bending and "
    "twisting of hygromorph multilayers whose "
    "layers may be homogenized from morphologies of the workspace.",
    order=65,
)

FACES = ("x-", "x+", "y-", "y+", "z-", "z+")
#: The diffusion properties of the phases when the form gives none (phase 1: a slower
#: and more soluble phase, as the fibre walls of templates/diffusion.yaml)
DEFAULT_PHASES = {
    "0": {"diffusivity": 1.0, "solubility": 1.0},
    "1": {"diffusivity": 0.1, "solubility": 3.0},
}
#: Output times of the diffusion (blank form): from the time the species takes to cross
#: a few voxels, at least this fraction of the time scale L² / D, to the time scale
FIRST_TIME = 1e-3
TIMES = 10


def _plot(
    curves: dict,
    xlabel: str,
    ylabel: str,
    title: str = "",
    *,
    logx=False,
    logy=False,
    ax=None,
    styles=None,
):
    """Curves name -> (x, y) on an axis (a new figure when none is given)."""
    from matplotlib.figure import Figure

    fig = None
    if ax is None:
        fig = Figure(figsize=(5.4, 3.8), layout="constrained")
        ax = fig.subplots()
    for name, (x, y) in curves.items():
        ax.plot(x, y, (styles or {}).get(name, "-"), lw=1.5, ms=3, label=name)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if len(curves) > 1:
        ax.legend(fontsize=8, frameon=False)
    return fig


def _phase_table(value, name: str, keys: tuple, required: tuple) -> dict[int, dict]:
    """A mapping phase -> properties of the form (a JSON object), checked."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name}: a mapping phase -> {{{', '.join(keys)}}}")
    out = {}
    for phase, props in value.items():
        if not isinstance(props, dict):
            raise ValueError(f"{name}: the properties of phase {phase}, a mapping")
        unknown = sorted(set(props) - set(keys))
        missing = [k for k in required if k not in props]
        if unknown or missing:
            raise ValueError(
                f"{name}, phase {phase}: "
                + (f"unknown {unknown}" if unknown else f"give {', '.join(missing)}")
            )
        try:
            out[int(phase)] = {k: float(v) for k, v in props.items()}
        except (TypeError, ValueError):
            raise ValueError(f"{name}, phase {phase}: numbers") from None
    return out


def _voxel_image(item: dict, resolution: int):
    from morphogen.gui.api import _voxels

    n = int(resolution)
    if not 4 <= n <= 256:
        raise ValueError("voxels per edge: from 4 to 256")
    micro = item["micro"]
    _voxels(micro.domain, n, 64.0, "diffusion")  # the operator and its factors
    image = micro.voxelize(shape=n)
    if np.asarray(image.data).dtype.kind not in "iub":
        raise ValueError(f"{item['title']} gives no phase image")
    return image


# -- transient diffusion ---------------------------------------------------------------------


@tool(
    "diffusion",
    "couplings",
    "Transient diffusion",
    "A species (moisture, oxygen, a solvent) diffusing into a morphology of the workspace "
    "from its exposed faces: each phase holds the concentration c = S u of the activity u "
    "(continuous across the interfaces) with its diffusivity D, solubility S and "
    "first-order consumption k, S du/dt = div(D S grad u) - k S u (finite volumes on the "
    "voxels, implicit Euler steps). The uptake curve, against Fick's √t at short times and "
    "Crank's slab with the effective diffusivity fitted to it; the concentration fields at "
    "the times and their video; the steady effective diffusivity of the homogenized "
    "permeability over the mean solubility, between its bounds.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("resolution", "int", 32, label="voxels per edge"),
        Field(
            "phases",
            "json",
            DEFAULT_PHASES,
            label="properties of the phases",
            doc="phase -> diffusivity, solubility (default 1), consumption (default 0)",
        ),
        Field(
            "exposed",
            "choices",
            ["x-", "x+"],
            choices=FACES,
            doc="the faces held at the activity; the others impermeable, or periodic",
        ),
        Field(
            "activity", "float", 1.0, doc="on the exposed faces (1: equilibrium with the outside)"
        ),
        Field("initial", "float", 0.0, doc="initial activity"),
        Field(
            "times",
            "floats",
            doc="output times (blank: ten, from the time to cross three voxels to the time "
            "scale L² / D, L the distance between the exposed faces)",
        ),
        Field("steps", "int", 20, doc="implicit Euler steps between two times"),
        Field(
            "field",
            "choice",
            "concentration",
            choices=(("concentration", "concentration c = S u"), ("activity", "activity u")),
        ),
        Field(
            "homogenize",
            "bool",
            True,
            label="steady effective diffusivity",
            doc="the homogenized permeability (FFT) over the mean solubility",
        ),
        Field("video", "bool", True),
    ],
    order=1,
    menu="Diffusion",
)
def diffusion(p: dict, ctx) -> dict:
    from morphogen.coupling import slab_uptake, transient_diffusion

    item = p["source"]
    image = _voxel_image(item, p["resolution"])
    labels = np.asarray(image.data).astype(np.int64)
    table = _phase_table(
        p["phases"] or DEFAULT_PHASES,
        "properties of the phases",
        ("diffusivity", "solubility", "consumption"),
        ("diffusivity",),
    )
    present = [int(v) for v in np.unique(labels)]
    missing = [k for k in present if k not in table]
    if missing:
        raise ValueError(
            f"properties of the phases: none for the phases {missing} of {item['title']}"
        )
    d = {k: table[k]["diffusivity"] for k in present}
    s = {k: table[k].get("solubility", 1.0) for k in present}
    k_ = {k: table[k].get("consumption", 0.0) for k in present}
    if any(v <= 0 for v in s.values()) or any(v < 0 for v in [*d.values(), *k_.values()]):
        raise ValueError(
            "properties of the phases: positive solubilities, non-negative "
            "diffusivities and consumptions"
        )
    exposed = list(p["exposed"] or ["x-", "x+"])
    axes = sorted({"xyz".index(f[0]) for f in exposed})
    lengths = np.asarray(image.size, dtype=float)
    fractions = {k: float(np.mean(labels == k)) for k in present}
    mean_s = sum(fractions[k] * s[k] for k in present)
    voigt = sum(fractions[k] * d[k] * s[k] for k in present) / mean_s
    reuss = 0.0
    if all(d[k] > 0 for k in present):
        reuss = 1.0 / sum(fractions[k] / (d[k] * s[k]) for k in present) / mean_s
    if not voigt > 0:
        raise ValueError("properties of the phases: no phase lets the species through")
    # the distance the species travels: between two opposite exposed faces, else twice
    # the length from one face (half of a slab exposed on both faces)
    one_axis = len(axes) == 1
    span = []
    for a in axes:
        both = f"{'xyz'[a]}-" in exposed and f"{'xyz'[a]}+" in exposed
        span.append(lengths[a] if both else 2.0 * lengths[a])
    distance = float(min(span))
    scale = distance**2 / voigt
    if p["times"]:
        times = np.asarray(p["times"], dtype=float)
    else:  # from the time to cross three voxels: the first voxels resolve the √t start
        voxel = max(float(image.spacing[a]) for a in axes)
        first = max(FIRST_TIME, (1.5 * voxel / distance) ** 2)
        times = np.geomspace(first, 1.0, TIMES) * scale
    if np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("times: positive and increasing")
    ctx.progress(0.1, f"diffusion in {labels.size} voxels")
    res = transient_diffusion(
        image,
        d,
        times,
        solubility=s,
        exposed=exposed,
        boundary=float(p["activity"]),
        initial=float(p["initial"]),
        consumption=k_,
        steps=int(p["steps"]),
    )
    # the fraction of the change of the content, (M - M0) / (M∞ - M0): the relative uptake
    # from a dry start, the fraction desorbed when drying
    start = mean_s * float(p["initial"])
    change = res.saturation - start
    relative = (res.uptake - start) / change if change else np.full_like(res.uptake, np.nan)
    rows = []
    for i, t in enumerate(res.times):
        row = {"time": float(t), "uptake": float(res.uptake[i]), "fraction": float(relative[i])}
        for k in present:
            row[f"c_{k}"] = float(res.fields[i][labels == k].mean())
        rows.append(row)
    ctx.table(rows, "uptake", "uptake curve (fraction: (M - M0) / (M∞ - M0))")
    values = {
        "saturation M∞ (mean S times the activity)": res.saturation,
        "final fraction (M - M0) / (M∞ - M0)": float(relative[-1]),
        "time scale L² / D (Voigt)": scale,
        "diffusivity, Voigt bound <D S> / <S>": voigt,
        "diffusivity, Reuss bound": reuss,
    }
    consuming = any(v > 0 for v in k_.values())
    early = np.flatnonzero((relative > 0) & (relative < 0.4))
    if len(early) >= 2 and not consuming:
        (t0, t1), (m0, m1) = res.times[early[:2]], relative[early[:2]]
        values["early slope of log M against log t (Fick: 1/2)"] = math.log(m1 / m0) / math.log(
            t1 / t0
        )
    fitted = None
    if one_axis and not consuming and change:
        from scipy.optimize import minimize_scalar

        def misfit(log_d):
            return float(np.sum((slab_uptake(res.times, distance, 10.0**log_d) - relative) ** 2))

        guess = math.log10(voigt)
        best = minimize_scalar(misfit, bounds=(guess - 6.0, guess + 2.0), method="bounded")
        fitted = 10.0**best.x
        values["effective diffusivity of the uptake (Crank)"] = fitted
        if 0 < relative[0] < 0.4:
            values["diffusivity of the short times, π (L M/M∞)² / 16 t"] = (
                math.pi * (distance * relative[0]) ** 2 / (16.0 * res.times[0])
            )
    if p["homogenize"]:
        ctx.progress(0.6, "homogenized permeability")
        from morphogen.homogenization.apparent import homogenize
        from morphogen.homogenization.materials import Material

        top = max(d[k] * s[k] for k in present)
        materials = {k: Material(conductivity=max(d[k] * s[k], 1e-9 * top)) for k in present}
        out = homogenize(image, materials, properties=("conductivity_tensor",))
        tensor = np.asarray(out["conductivity_tensor.periodic"]["tensor"])
        for a in axes:
            values[f"steady effective diffusivity along {'xyz'[a]}"] = float(tensor[a, a]) / mean_s
    ctx.progress(0.75, "figures")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9.6, 3.8), layout="constrained")
    left, right = fig.subplots(1, 2)
    curves = {"computed": (res.times, relative)}
    styles = {"computed": "o-", "Fick, √t": "k:", "Crank, fitted D": "--"}
    if len(early):  # √t through the first point of the short times, up to M / M∞ = 0.6
        i = int(early[0])
        fick = relative[i] * np.sqrt(res.times / res.times[i])
        curves["Fick, √t"] = (res.times[fick < 0.6], fick[fick < 0.6])
    if fitted is not None:
        dense = np.geomspace(res.times[0], res.times[-1], 120)
        curves["Crank, fitted D"] = (dense, slab_uptake(dense, distance, fitted))
    _plot(
        curves,
        "time",
        "M / M∞",
        "uptake (log scales)",
        logx=True,
        logy=True,
        ax=left,
        styles=styles,
    )
    _plot(curves, "time", "M / M∞", "uptake", ax=right, styles=styles)
    right.set_ylim(0.0, max(1.05, float(np.max(relative)) * 1.05))
    ctx.figure(fig, "uptake", "uptake curve")
    shown = list(res.fields)
    if p["field"] == "activity":  # u = c / S
        table_s = np.zeros(max(present) + 1)
        table_s[present] = [s[k] for k in present]
        shown = [f / table_s[labels] for f in shown]
    k = labels.shape[2] // 2
    chosen = np.unique(np.linspace(0, len(times) - 1, min(len(times), 6)).round().astype(int))
    fig = Figure(figsize=(2.6 * len(chosen) + 0.8, 3.0), layout="constrained")
    axes_ = np.atleast_1d(fig.subplots(1, len(chosen)))
    lo = min(float(np.min(f)) for f in shown)
    hi = max(float(np.max(f)) for f in shown)
    extent = (0.0, lengths[0], 0.0, lengths[1])
    for ax, i in zip(axes_, chosen, strict=True):
        im = ax.imshow(
            shown[i][:, :, k].T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
        )
        ax.set_title(f"t = {times[i]:.3g}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=list(axes_), shrink=0.85, label=p["field"])
    ctx.figure(fig, "fields", f"{p['field']} in the middle section (x-y) at the times")
    if p["video"]:
        from morphogen.viz.video import time_section_frames

        frames = time_section_frames(
            shown,
            2,
            k,
            size=420,
            cmap="viridis",
            vmin=lo,
            vmax=hi,
            captions=[f"t = {t:.3g}" for t in times],
            label=p["field"][:13],
        )
        ctx.video(frames, "diffusion", f"{p['field']} along the uptake", fps=3, mode="once")
    return {"values": values, "title": f"Transient diffusion in {item['title']}"}


# -- thermal oxidation -----------------------------------------------------------------------


@tool(
    "oxidation",
    "couplings",
    "Thermal oxidation",
    "The thermal oxidation of an epoxy plate exposed on both faces (the closed-loop scheme "
    "of Colin, Verdu and co-workers with the rate constants that Ernault et al. 2019 "
    "identified on a DGEBA/IPDA at 110 °C in air): oxygen enters by Fick's law at the "
    "concentration S p and is consumed by the radical chain, so that the oxidation "
    "(traced by the amides P=O) stays in a layer a few hundred micrometres thick. The "
    "profiles of the species through the thickness at the times, the thickness of the "
    "oxidized layer against time, the homogeneous oxidation of a thin film, and the "
    "profiles of the properties of an ageing law of the amide concentration (a modulus, a "
    "shrinkage).",
    [
        Field("thickness", "float", 3e-3, unit="m", doc="of the plate, exposed on both faces"),
        Field("times", "floats", [50.0, 111.0, 240.0, 400.0, 800.0], unit="h"),
        Field(
            "pressure", "float", unit="Pa", doc="partial pressure of oxygen (blank: air, 0.21 atm)"
        ),
        Field(
            "constants",
            "json",
            label="constants of the scheme",
            doc="changes of the defaults of Ernault et al. (2019): k1...k6 (s⁻¹, l/mol/s), "
            "gamma6, ph0, pooh0 (mol/l), solubility (mol/l/Pa), diffusivity (m²/s)",
        ),
        Field("nodes", "int", 61, doc="through half the plate (method of lines)"),
        Field(
            "fraction",
            "float",
            0.1,
            doc="the oxidized layer ends where the amides fall to "
            "this fraction of their value at the surface",
        ),
        Field(
            "law",
            "json",
            label="ageing law",
            doc="properties against the amide concentration (mol/l), piecewise linear, e.g. "
            '{"tracer": [0, 1], "young": [2400, 3200], "expansion": [0, -0.01]} (blank: none)',
        ),
    ],
    order=2,
    menu="Oxidation",
)
def oxidation(p: dict, ctx) -> dict:
    import dataclasses

    from morphogen.coupling import ClosedLoopOxidation, PropertyLaw

    changes = dict(p["constants"] or {})
    if not isinstance(changes, dict):
        raise ValueError("constants of the scheme: a mapping name -> value")
    known = {f.name for f in dataclasses.fields(ClosedLoopOxidation)}
    unknown = sorted(set(changes) - known)
    if unknown:
        raise ValueError(f"constants of the scheme: unknown {unknown}; among {sorted(known)}")
    if p["pressure"] is not None:
        changes["pressure"] = float(p["pressure"])
    model = ClosedLoopOxidation(**{k: float(v) for k, v in changes.items()})
    hours = np.asarray(p["times"] or [], dtype=float)
    if not len(hours) or np.any(hours <= 0) or np.any(np.diff(hours) <= 0):
        raise ValueError("times: positive and increasing")
    thickness = float(p["thickness"])
    nodes = int(p["nodes"])
    if not (thickness > 0 and 11 <= nodes <= 801):
        raise ValueError("thickness: positive; nodes: from 11 to 801")
    times = hours * 3600.0
    ctx.progress(0.05, "thin film")
    film = model.thin_film(times)
    ctx.progress(0.2, "profiles through the thickness")
    plate = model.profile(thickness, times, nodes=nodes)
    depth = plate["depth"]
    fraction = float(p["fraction"])
    layers = [model.oxidized_layer(depth, a, fraction) for a in plate["PO"]]
    ctx.progress(0.8, "figures")
    um = depth * 1e6
    rows = []
    for k, t in enumerate(hours):
        rows.append(
            {
                "time_h": float(t),
                "oxidized_layer_um": layers[k] * 1e6,
                "amide_surface": float(plate["PO"][k][0]),
                "amide_core": float(plate["PO"][k][-1]),
                "oxygen_core": float(plate["O2"][k][-1]),
                "amide_thin_film": float(film["PO"][k]),
            }
        )
    ctx.table(rows, "oxidized_layer", "the oxidized layer at the times")
    profile_rows = []
    for i, z in enumerate(um):
        row = {"depth_um": float(z)}
        row.update({f"amide_{t:g}h": float(plate["PO"][k][i]) for k, t in enumerate(hours)})
        profile_rows.append(row)
    ctx.table(profile_rows, "amide_profiles", "amide profiles (mol/l) through the thickness")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9.6, 3.8), layout="constrained")
    left, right = fig.subplots(1, 2)
    _plot(
        {f"{t:g} h": (um, plate["PO"][k]) for k, t in enumerate(hours)},
        "depth (µm)",
        "amides P=O (mol/l)",
        "oxidation profiles",
        ax=left,
    )
    for k, layer in enumerate(layers):
        left.axvline(layer * 1e6, color=left.lines[k].get_color(), ls=":", lw=1)
    _plot(
        {f"{t:g} h": (um, plate["O2"][k] * 1e3) for k, t in enumerate(hours)},
        "depth (µm)",
        "oxygen (mmol/l)",
        "dissolved oxygen",
        ax=right,
    )
    reach = min(float(um[-1]), 4e6 * max(layers))  # the core is not oxidized
    for ax in (left, right):
        ax.set_xlim(0.0, reach)
    ctx.figure(fig, "profiles", "profiles through the thickness")
    fig = Figure(figsize=(9.6, 3.8), layout="constrained")
    left, right = fig.subplots(1, 2)
    dense = np.linspace(0.0, times[-1], 200)
    kinetics = model.thin_film(dense[1:])
    _plot(
        {
            "thin film": (dense[1:] / 3600.0, kinetics["PO"]),
            "surface of the plate": (hours, plate["PO"][:, 0]),
            "core of the plate": (hours, plate["PO"][:, -1]),
        },
        "time (h)",
        "amides P=O (mol/l)",
        "kinetics",
        ax=left,
        styles={"surface of the plate": "o-", "core of the plate": "s-"},
    )
    _plot(
        {"oxidized layer": (hours, np.asarray(layers) * 1e6)},
        "time (h)",
        "thickness (µm)",
        f"oxidized layer (amides > {fraction:g} of the surface)",
        ax=right,
        styles={"oxidized layer": "o-"},
    )
    ctx.figure(fig, "kinetics", "kinetics and the oxidized layer")
    values = {
        "oxygen at the surface S p (mol/l)": model.surface_oxygen,
        "final oxidized layer (µm)": layers[-1] * 1e6,
        "final amides at the surface (mol/l)": float(plate["PO"][-1][0]),
        "final amides in the core (mol/l)": float(plate["PO"][-1][-1]),
        "final amides of a thin film (mol/l)": float(film["PO"][-1]),
    }
    if p["law"]:
        spec = dict(p["law"])
        if "tracer" not in spec:
            raise ValueError('ageing law: {"tracer": [...], "<property>": [...]}')
        law = PropertyLaw(tracer=tuple(spec.pop("tracer")), values=spec)
        names = sorted(law.values)
        fig = Figure(figsize=(4.0 * len(names) + 0.8, 3.6), layout="constrained")
        axes = np.atleast_1d(fig.subplots(1, len(names)))
        for ax, name in zip(axes, names, strict=True):
            _plot(
                {f"{t:g} h": (um, law(plate["PO"][k])[name]) for k, t in enumerate(hours)},
                "depth (µm)",
                name,
                f"{name} through the thickness",
                ax=ax,
            )
        ctx.figure(fig, "properties", "properties of the ageing law through the thickness")
        surface = law(plate["PO"][-1][0])
        core = law(plate["PO"][-1][-1])
        for name in names:
            values[f"final {name} at the surface"] = float(surface[name])
            values[f"final {name} in the core"] = float(core[name])
    return {"values": values, "title": f"Thermal oxidation of a plate of {thickness * 1e3:g} mm"}


# -- thermal oxidation of a morphology -------------------------------------------------------


@tool(
    "oxidation_morphology",
    "couplings",
    "Thermal oxidation of a morphology",
    "The thermal oxidation of the phases of a morphology of the workspace, in its volume: "
    "the closed-loop scheme in every voxel of the phases that oxidize, oxygen diffusing "
    "between the voxels from the exposed faces, around the phases that bar its way "
    "(fibres, fillers) and through the inert phases that let it pass. The oxidation "
    "products (amides P=O) in the volume at the times, their mean profile from the exposed "
    "face against that of a plate of the polymer alone, and the depth of the oxidized "
    "layer.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("resolution", "int", 24, label="voxels per edge"),
        Field(
            "edge",
            "float",
            0.6e-3,
            label="length of the morphology along x",
            unit="m",
            doc="its unit of length follows (the oxidized layer is a hundred micrometres thick)",
        ),
        Field(
            "oxidizing",
            "ints",
            [0],
            label="phases that oxidize",
            doc="with the constants of the scheme",
        ),
        Field(
            "permeable",
            "ints",
            label="inert phases that let oxygen through",
            doc="with the transport of oxygen of the scheme and no reactive site (blank: "
            "none); the other phases are impermeable and inert",
        ),
        Field(
            "exposed",
            "choices",
            ["x-"],
            choices=FACES,
            doc="the faces in the atmosphere; the others impermeable, or periodic",
        ),
        Field("times", "floats", [111.0, 400.0], unit="h"),
        Field(
            "pressure", "float", unit="Pa", doc="partial pressure of oxygen (blank: air, 0.21 atm)"
        ),
        Field(
            "constants",
            "json",
            label="constants of the scheme",
            doc="changes of the defaults of Ernault et al. (2019): k1...k6 (s⁻¹, l/mol/s), "
            "gamma6, ph0, pooh0 (mol/l), solubility (mol/l/Pa), diffusivity (m²/s)",
        ),
        Field(
            "fraction",
            "float",
            0.1,
            doc="the oxidized layer ends where the mean amides fall to this fraction of "
            "their value under the exposed face",
        ),
    ],
    order=3,
    menu="Oxidation",
)
def oxidation_morphology(p: dict, ctx) -> dict:
    import dataclasses

    from morphogen.coupling import ClosedLoopOxidation, oxidation_field

    item = p["source"]
    n = int(p["resolution"])
    if not 4 <= n <= 64:
        raise ValueError("voxels per edge: from 4 to 64")
    image = _voxel_image(item, n)
    labels = np.asarray(image.data).astype(np.int64)
    present = [int(v) for v in np.unique(labels)]
    changes = dict(p["constants"] or {})
    known = {f.name for f in dataclasses.fields(ClosedLoopOxidation)}
    unknown = sorted(set(changes) - known)
    if unknown:
        raise ValueError(f"constants of the scheme: unknown {unknown}; among {sorted(known)}")
    if p["pressure"] is not None:
        changes["pressure"] = float(p["pressure"])
    model = ClosedLoopOxidation(**{k: float(v) for k, v in changes.items()})
    oxidizing = sorted({int(k) for k in p["oxidizing"] or []})
    permeable = sorted({int(k) for k in p["permeable"] or []})
    if not set(oxidizing) & set(present):
        raise ValueError(f"phases that oxidize: one at least among the phases {present}")
    if set(oxidizing) & set(permeable):
        raise ValueError("a phase oxidizes or is inert, not both")
    inert = dataclasses.replace(model, ph0=0.0, pooh0=0.0)
    schemes = {k: model if k in oxidizing else inert if k in permeable else None for k in present}
    hours = np.asarray(p["times"] or [], dtype=float)
    if not len(hours) or np.any(hours <= 0) or np.any(np.diff(hours) <= 0):
        raise ValueError("times: positive and increasing")
    edge = float(p["edge"])
    if not edge > 0:
        raise ValueError("length of the morphology along x: positive")
    unit = edge / float(image.size[0])
    exposed = list(p["exposed"] or ["x-"])
    ctx.progress(0.05, f"oxidation in {labels.size} voxels")
    found = oxidation_field(
        image,
        schemes,
        hours * 3600.0,
        exposed=exposed,
        unit=unit,
        progress=lambda f: ctx.progress(0.05 + 0.8 * f, f"oxidation in {labels.size} voxels"),
    )
    amide = found.fields["PO"]
    reactive = np.isin(labels, oxidizing)
    # the mean profile of the phases that oxidize, from the first exposed face
    axis = "xyz".index(exposed[0][0])
    other = tuple(a for a in range(3) if a != axis)
    count = reactive.sum(axis=other)
    profiles = (amide * reactive).sum(axis=tuple(a + 1 for a in other)) / np.maximum(count, 1)
    if exposed[0][1] == "+":
        profiles, count = profiles[:, ::-1], count[::-1]
    both = f"{'xyz'[axis]}-" in exposed and f"{'xyz'[axis]}+" in exposed
    length = float(image.size[axis]) * unit
    depth = (np.arange(labels.shape[axis]) + 0.5) * length / labels.shape[axis]
    keep = (depth <= 0.5 * length if both else np.ones(len(depth), bool)) & (count > 0)
    fraction = float(p["fraction"])
    layers = [model.oxidized_layer(depth[keep], a[keep], fraction) for a in profiles]
    ctx.progress(0.88, "the plate of the polymer alone")
    plate = model.profile(length if both else 2.0 * length, hours * 3600.0, nodes=121)
    plate_layers = [model.oxidized_layer(plate["depth"], a, fraction) for a in plate["PO"]]
    rows = []
    for k, t in enumerate(hours):
        rows.append(
            {
                "time_h": float(t),
                "amide_mean": float(amide[k][reactive].mean()),
                "amide_max": float(amide[k][reactive].max()),
                "oxidized_layer_um": layers[k] * 1e6,
                "oxidized_layer_plate_um": plate_layers[k] * 1e6,
                "oxygen_mean": float(found.fields["O2"][k][reactive].mean()),
            }
        )
    ctx.table(rows, "oxidation", "the oxidation of the phases that oxidize at the times")
    um = depth[keep] * 1e6
    profile_rows = []
    for i, z in enumerate(um):
        row = {"depth_um": float(z)}
        row.update({f"amide_{t:g}h": float(profiles[k][keep][i]) for k, t in enumerate(hours)})
        profile_rows.append(row)
    ctx.table(profile_rows, "amide_profiles", "mean amide profiles (mol/l) from the exposed face")
    ctx.progress(0.92, "figures")
    from matplotlib.figure import Figure

    curves, styles = {}, {}
    for k, t in enumerate(hours):
        curves[f"{t:g} h"] = (um, profiles[k][keep])
        curves[f"{t:g} h, plate"] = (plate["depth"] * 1e6, plate["PO"][k])
        styles[f"{t:g} h"], styles[f"{t:g} h, plate"] = "o-", "k:"
    fig = _plot(
        curves,
        "depth (µm)",
        "amides P=O (mol/l)",
        "mean profiles (dotted: a plate of the polymer alone)",
        styles=styles,
    )
    fig.axes[0].set_xlim(0.0, float(um[-1]))
    ctx.figure(fig, "profiles", "mean oxidation profiles from the exposed face")
    k = labels.shape[2] // 2
    lengths = np.asarray(image.size, dtype=float) * unit * 1e6
    fig = Figure(figsize=(2.8 * len(hours) + 0.8, 3.0), layout="constrained")
    axes_ = np.atleast_1d(fig.subplots(1, len(hours)))
    top = float(amide.max())
    for ax, i in zip(axes_, range(len(hours)), strict=True):
        shown = np.where(reactive[:, :, k], amide[i][:, :, k], np.nan)
        im = ax.imshow(
            shown.T,
            origin="lower",
            extent=(0.0, lengths[0], 0.0, lengths[1]),
            cmap="inferno",
            vmin=0.0,
            vmax=top,
            interpolation="nearest",
        )
        ax.set_title(f"{hours[i]:g} h", fontsize=9)
        ax.set_xlabel("x (µm)")
    fig.colorbar(im, ax=list(axes_), shrink=0.85, label="amides (mol/l)")
    ctx.figure(fig, "fields", "amides in the middle section (x-y) at the times")
    values = {
        "voxels": int(labels.size),
        "time steps": int(found.steps),
        "fraction of the phases that oxidize": float(reactive.mean()),
        "oxygen at the surface S p (mol/l)": model.surface_oxygen,
        "final mean amides (mol/l)": float(amide[-1][reactive].mean()),
        "final oxidized layer (µm)": layers[-1] * 1e6,
        "final oxidized layer of a plate of the polymer alone (µm)": plate_layers[-1] * 1e6,
        "voxels across the final oxidized layer (four at least resolve it)": layers[-1]
        / (length / labels.shape[axis]),
    }
    return {"values": values, "title": f"Thermal oxidation of {item['title']}"}


# -- hygromorph multilayers ------------------------------------------------------------------


#: The layers of the default hygromorph: an active layer that swells, more along y, under a
#: passive one (test_coupling's hygromorph; SI units and MPa)
LAYERS = {
    "bottom": (
        "constants",
        0.5e-3,
        {
            "young": 2000.0,
            "poisson": 0.35,
            "beta_x": 0.2,
            "beta_y": 0.3,
            "diffusivity": 4e-12,
            "solubility": 0.1,
        },
    ),
    "middle": (
        "none",
        0.25e-3,
        {"young": 3000.0, "poisson": 0.35, "diffusivity": 2e-12, "solubility": 0.05},
    ),
    "top": (
        "constants",
        0.5e-3,
        {"young": 3500.0, "poisson": 0.35, "diffusivity": 1e-12, "solubility": 0.01},
    ),
}
#: The properties of the phases of a homogenized layer: a hygroscopic polymer matrix that
#: swells, and stiff fibres that hardly do (they restrain the swelling along them)
PHASES = {
    "0": {
        "young": 2000.0,
        "poisson": 0.35,
        "swelling": 0.2,
        "diffusivity": 4e-12,
        "solubility": 0.1,
    },
    "1": {
        "young": 30000.0,
        "poisson": 0.3,
        "swelling": 0.0,
        "diffusivity": 1e-12,
        "solubility": 0.01,
    },
}
LAYER_KEYS = ("young", "poisson", "beta_x", "beta_y", "beta_xy", "diffusivity", "solubility")
PHASE_KEYS = ("young", "poisson", "swelling", "diffusivity", "solubility")


def _layer_fields(name: str) -> list[Field]:
    kind, thickness, constants = LAYERS[name]
    choices = [("constants", "properties given"), ("morphology", "homogenized from a morphology")]
    if name == "middle":
        choices.insert(0, ("none", "no middle layer"))
    on = {name: "morphology"}
    return [
        Field(name, "choice", kind, label=f"{name} layer", choices=tuple(choices)),
        Field(
            f"{name}_thickness",
            "float",
            thickness,
            label="thickness",
            unit="m",
            when={name: ["constants", "morphology"]},
        ),
        Field(
            f"{name}_properties",
            "json",
            constants,
            label="properties",
            when={name: "constants"},
            doc="young, poisson, beta_x, beta_y, beta_xy (swelling per unit concentration), "
            "diffusivity, solubility",
        ),
        Field(f"{name}_morphology", "item", label="morphology", required=True, when=on),
        Field(
            f"{name}_phases",
            "json",
            PHASES,
            label="properties of its phases",
            when=on,
            doc="phase -> young, poisson, swelling, diffusivity, solubility",
        ),
    ]


@tool(
    "hygromorph",
    "couplings",
    "Hygromorph multilayer",
    "A free plate of two or three layers bending and twisting as moisture enters it "
    "(pine cone scales, 4D-printed hygromorphs; Mou, Dirrenberger & Le Duigou 2025): the "
    "moisture diffuses through the thickness (the activity continuous between layers), "
    "each layer swells by β c (orthotropic, with a shear for twisting layers) and the "
    "laminate theory gives the curvatures κx, κy and the twist κxy of the free plate at "
    "every time (positive: bending towards its lower face). A layer is given by its "
    "properties, or homogenized from a morphology of the workspace (its apparent "
    "stiffness, swelling and permeability, periodic FFT homogenization): stiff fibres "
    "aligned in a swelling matrix restrain it along them, and the layer bends across them.",
    [
        *_layer_fields("bottom"),
        *_layer_fields("middle"),
        *_layer_fields("top"),
        Field(
            "resolution", "int", 16, label="voxels per edge", doc="of the morphologies homogenized"
        ),
        Field(
            "normal",
            "choice",
            "z",
            choices=("x", "y", "z"),
            doc="the axis of the morphologies along the thickness of their layers",
        ),
        Field(
            "exposed",
            "choice",
            "bottom",
            choices=(("bottom", "the bottom face"), ("top", "the top face"), ("both", "both")),
        ),
        Field("activity", "float", 1.0, doc="on the exposed faces"),
        Field("initial", "float", 0.0, doc="initial activity (1 and an activity 0: drying)"),
        Field(
            "times",
            "floats",
            unit="s",
            doc="blank: up to three times the diffusion time of the stack",
        ),
        Field("steps", "int", 100, doc="implicit Euler steps between two times"),
    ],
    order=4,
    groups=(
        ("Bottom layer", "bottom"),
        ("Middle layer", "middle"),
        ("Top layer", "top"),
        ("Morphologies of the layers", "resolution", True),
        ("Moisture and time", "exposed"),
    ),
    menu="Moisture",
)
def hygromorph(p: dict, ctx) -> dict:
    from morphogen.coupling import Layer, Multilayer, timoshenko_curvature
    from morphogen.homogenization.materials import Material

    names = [n for n in ("bottom", "middle", "top") if p[n] != "none"]
    layers, rows = [], []
    for k, name in enumerate(names):
        thickness = float(p[f"{name}_thickness"])
        if not thickness > 0:
            raise ValueError(f"{name} layer: a positive thickness")
        if p[name] == "constants":
            props = _phase_table(
                {"0": p[f"{name}_properties"] or {}}, f"{name} layer", LAYER_KEYS, ("young",)
            )[0]
            layer = Layer(thickness, **props)
            origin = "properties given"
        else:
            item = p[f"{name}_morphology"]
            ctx.progress(0.05 + 0.5 * k / len(names), f"homogenizing the {name} layer")
            image = _voxel_image(item, p["resolution"])
            phases = _phase_table(
                p[f"{name}_phases"] or PHASES, f"{name} layer", PHASE_KEYS, ("young", "poisson")
            )
            present = [int(v) for v in np.unique(image.data)]
            missing = [v for v in present if v not in phases]
            if missing:
                raise ValueError(f"{name} layer: no properties for the phases {missing}")
            perm = {
                v: phases[v].get("diffusivity", 0.0) * phases[v].get("solubility", 1.0)
                for v in present
            }
            top = max(perm.values()) or 1.0
            materials = {
                v: Material(
                    young=phases[v]["young"],
                    poisson=phases[v]["poisson"],
                    conductivity=max(perm[v], 1e-9 * top),
                    expansion=phases[v].get("swelling", 0.0),
                )
                for v in present
            }
            layer = Layer.from_microstructure(
                thickness,
                image,
                materials,
                normal="xyz".index(p["normal"]),
                solubility={v: phases[v].get("solubility", 1.0) for v in present},
            )
            origin = f"homogenized from {item['title']}"
        layers.append(layer)
        q = layer.plane_stiffness()
        s = np.linalg.inv(q)  # its plane-stress compliance, in the same notation
        rows.append(
            {
                "layer": name,
                "origin": origin,
                "thickness": thickness,
                "Q_xx": q[0, 0],
                "Q_yy": q[1, 1],
                "Q_xy": q[0, 1],
                "Q_ss": q[2, 2],
                "Q_xs": q[0, 2],
                "Q_ys": q[1, 2],
                "S_xx": s[0, 0],
                "S_yy": s[1, 1],
                "S_xy": s[0, 1],
                "S_ss": s[2, 2],
                "S_xs": s[0, 2],
                "S_ys": s[1, 2],
                "beta_x": layer.beta_x,
                "beta_y": layer.beta_y,
                "beta_xy": layer.beta_xy,
                "diffusivity": layer.diffusivity,
                "solubility": layer.solubility,
            }
        )
    if len(layers) < 2:
        raise ValueError("a hygromorph needs two layers at least")
    ctx.table(rows, "layers", "the layers, from the bottom face: their plane-stress stiffness "
              "Q_ij and compliance S_ij = Q⁻¹ (Voigt xx, yy, xy = s, engineering shear; units of "
              "the moduli and their inverse)")  # fmt: skip
    plate = Multilayer(layers)
    total = plate.thickness
    diffusivities = [lay.diffusivity for lay in layers]
    if not all(v > 0 for v in diffusivities):
        raise ValueError("every layer needs a positive diffusivity (the moisture crosses them)")
    series = total / sum(lay.thickness / lay.diffusivity for lay in layers)
    scale = total**2 / series
    times = np.asarray(p["times"] or np.geomspace(1e-3, 3.0, 16) * scale, dtype=float)
    if np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("times: positive and increasing")
    ctx.progress(0.6, "moisture and curvatures")
    activity, initial = float(p["activity"]), float(p["initial"])
    c = plate.moisture(
        times, exposed=p["exposed"], activity=activity, initial=initial, steps=int(p["steps"])
    )
    strains, curvatures = plate.deformation(c, full=True)
    # at equilibrium every layer holds S times the activity (the reference: dry, c = 0)
    equilibrium = np.array([layers[i].solubility for i in plate.index]) * activity
    _, final = plate.deformation(equilibrium, full=True)
    history = []
    for i, t in enumerate(times):
        history.append(
            {
                "time": float(t),
                "kappa_x": curvatures[i, 0],
                "kappa_y": curvatures[i, 1],
                "kappa_xy": curvatures[i, 2],
                "strain_x": strains[i, 0],
                "strain_y": strains[i, 1],
                "mean_concentration": float(np.sum(c[i] * plate.dz) / total),
            }
        )
    ctx.table(history, "curvatures", "curvatures and mid-plane strains along the uptake")
    ctx.progress(0.85, "figures")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9.6, 3.8), layout="constrained")
    left, right = fig.subplots(1, 2)
    twist = np.abs(curvatures[:, 2]).max() > 1e-9 * np.abs(curvatures[:, :2]).max()
    curves = {"κx": (times, curvatures[:, 0]), "κy": (times, curvatures[:, 1])}
    if twist:
        curves["κxy (twist)"] = (times, curvatures[:, 2])
    _plot(
        curves,
        "time",
        "curvature (1/length)",
        "curvatures of the free plate",
        logx=True,
        ax=left,
        styles=dict.fromkeys(curves, "o-"),
    )
    for k, value in enumerate(final[: 3 if twist else 2]):
        left.axhline(value, color=left.lines[k].get_color(), ls=":", lw=1)
    shown = np.unique(np.linspace(0, len(times) - 1, min(len(times), 6)).round().astype(int))
    _plot(
        {f"t = {times[i]:.3g}": (c[i], plate.z) for i in shown},
        "concentration",
        "height z",
        "moisture through the thickness",
        ax=right,
    )
    for edge in np.cumsum([lay.thickness for lay in layers])[:-1]:
        right.axhline(edge, color="0.5", lw=0.8)
    ctx.figure(fig, "curvatures", "curvatures and moisture profiles")
    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    # a strip seen from its side (x-z), long enough to turn by 1.5 radians at the end
    bent = float(np.abs(curvatures[:, 0]).max())
    span = float(np.clip(1.5 / bent if bent > 0 else 0.0, 10.0 * total, 200.0 * total))
    s = np.linspace(-span / 2, span / 2, 101)
    for i in shown:
        k = curvatures[i, 0]
        x, z = (
            (s, -k * s**2 / 2)
            if abs(k * span) < 1e-6
            else (np.sin(k * s) / k, (np.cos(k * s) - 1.0) / k)
        )
        ax.plot(x, z, lw=1.5, label=f"t = {times[i]:.3g}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("a strip along x bending (side view)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    ctx.figure(fig, "strip", "a strip along x bending")
    values = {
        "final κx": float(curvatures[-1, 0]),
        "final κy": float(curvatures[-1, 1]),
        "final κxy (twist)": float(curvatures[-1, 2]),
        "equilibrium κx": float(final[0]),
        "equilibrium κy": float(final[1]),
        "equilibrium κxy": float(final[2]),
        "diffusion time of the stack H² / D": scale,
    }
    if np.abs(final[:2]).max() > 0:
        values["equilibrium radius of curvature"] = 1.0 / float(np.abs(final[:2]).max())
    if len(layers) == 2 and _isotropic_pair(layers):  # Timoshenko's bilayer (1925)
        low, up = layers
        swell = (up.beta_x * up.solubility - low.beta_x * low.solubility) * activity
        values["Timoshenko's curvature of the bilayer"] = timoshenko_curvature(
            swell, low.thickness, up.thickness, low.young, up.young
        )
    return {"values": values, "title": f"Hygromorph of {len(layers)} layers"}


def _isotropic_pair(layers) -> bool:
    """Whether two layers are isotropic, of one Poisson's ratio and with isotropic
    swellings (Timoshenko's formula is then exact)."""
    a, b = layers
    return (
        a.stiffness is None
        and b.stiffness is None
        and math.isclose(a.poisson, b.poisson)
        and all(math.isclose(x.beta_x, x.beta_y) and x.beta_xy == 0.0 for x in layers)
    )
