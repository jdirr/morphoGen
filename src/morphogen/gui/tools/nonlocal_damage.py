"""The tools of the tab Nonlocal damage: micromorphic continua, in which a length sets the
width of the bands where plasticity or damage localize, whatever the elements (Forest
2009, the micromorphic approach; :mod:`morphogen.micromorphic`): the microfoam model of
metallic foams (Dillard, Forest & Ienny 2006), the micromorphic ductile damage of
elastoplastic solids driven by the plastic strain weighted by the stress triaxiality
(Rice & Tracey 1969, Johnson & Cook 1985, Lemaitre 1985), and the micromorphic damage of
quasi-brittle solids (the implicit gradient model of Peerlings et al. 1996).

The specimens are made of voxels of six quadratic tetrahedra each
(:func:`~morphogen.homogenization.bloch.voxel_tet_mesh`): a notched plate, a bar, or a
morphology of the workspace (its middle section or its volume, without its holes and the
parts that do not join the grips), in plane strain (one layer of voxels, no displacement
across it), as a thin plate with free faces (nearly plane stress), or in 3D. Its face ``y =
0`` is held, its face ``y = height`` pulled (or pushed); the fields of the elements are
drawn on their voxels, in the middle section of a volume."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morphogen.gui.tools import Field, tab, tool

tab(
    "damage",
    "Nonlocal damage",
    "Micromorphic continua, whose length sets the width of the bands of localization "
    "independently of the elements: the microfoam model of metallic foams, the "
    "micromorphic ductile damage with the stress triaxiality, and the micromorphic damage "
    "of quasi-brittle solids, on a notched plate, a bar or a morphology of the workspace.",
    order=55,
)

#: The label of the voxels without material, and of a weaker zone of a bar
HOLE, WEAK = 255, 254
#: Phases taken for holes when the form names none
HOLE_NAMES = ("void", "pores", "pore", "porosity", "cells", "holes")
#: The largest problems of the interface (degrees of freedom): the direct solver of the
#: steps factorizes their tangent at every iteration
MAX_DOFS = 60_000
MAX_MICROMORPHIC_DOFS = 25_000

SPECIMENS = (
    ("notched_plate", "notched plate: a notch from its left edge at mid-height"),
    ("bar", "bar: notches on both sides at mid-height, or a weaker layer"),
    ("morphology", "a morphology of the workspace (its holes removed)"),
)
CONDITIONS = (
    ("plane_strain", "plane strain: one layer of voxels, no displacement across it"),
    ("plane_stress", "thin plate: one layer of voxels with free faces (nearly plane stress)"),
    ("3d", "3D: the plate with its thickness, a bar of square section, the volume"),
)
KINEMATICS = (("small", "small strains"), ("finite", "finite strains (total Lagrangian)"))
GRIPS = (
    ("rollers", "rollers: the gripped faces slide freely"),
    ("clamped", "clamped: the gripped faces do not move sideways"),
)
_SHAPED = {"specimen": ["notched_plate", "bar"]}


def _specimen_fields(
    default: str = "notched_plate",
    imperfection: float = 0.1,
    elements: int = 12,
    resolution: int = 24,
) -> list[Field]:
    """The fields of the specimen, its grips and its mesh (shared by the tools)."""
    return [
        Field("specimen", "choice", default, choices=SPECIMENS),
        Field("source", "item", label="morphology", required=True, when={"specimen": "morphology"}),
        Field(
            "holes",
            "ints",
            when={"specimen": "morphology"},
            doc="its phases removed (blank: those named void, pores or cells)",
        ),
        Field(
            "resolution",
            "int",
            resolution,
            label="voxels per edge",
            when={"specimen": "morphology"},
            doc="of its voxel image (six tetrahedra per voxel)",
        ),
        Field("condition", "choice", "plane_strain", choices=CONDITIONS),
        Field("width", "float", 1.0, unit="length", when=_SHAPED),
        Field(
            "height",
            "float",
            unit="length",
            when=_SHAPED,
            doc="along the load, y (blank: the width for a plate, twice for a bar)",
        ),
        Field("elements", "int", elements, label="voxels across the width", when=_SHAPED),
        Field(
            "thickness",
            "float",
            unit="length",
            when={"specimen": "notched_plate", "condition": "3d"},
            doc="blank: a quarter of the width",
        ),
        Field(
            "notch_depth",
            "float",
            0.3 if default == "notched_plate" else 0.0,
            when=_SHAPED,
            doc="fraction of the width (a bar: from each side)",
        ),
        Field(
            "notch_width",
            "float",
            0.1,
            when=_SHAPED,
            doc="fraction of the width (at least a row of voxels)",
        ),
        Field(
            "imperfection",
            "float",
            imperfection,
            when={"specimen": "bar"},
            doc="the strength of the layer at mid-height of the bar lowered by this fraction "
            "(where the band starts)",
        ),
        Field("grips", "choice", "rollers", choices=GRIPS),
    ]


# -- the specimens ---------------------------------------------------------------------------


@dataclass
class Specimen:
    """A voxel specimen pulled along y: the labels of its voxels (the phases, ``HOLE``
    without material, ``WEAK`` in the weaker layer of a bar), their edges, its condition
    (``plane_strain``, ``plane_stress`` or ``3d``), the depth of its notch (from ``x = 0``,
    0 for none) and its title."""

    labels: np.ndarray
    spacing: tuple
    condition: str
    title: str
    notch: float = 0.0

    @property
    def shape(self) -> tuple:
        return self.labels.shape

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray(self.shape, dtype=float) * np.asarray(self.spacing, dtype=float)

    @property
    def planar(self) -> bool:
        return self.condition != "3d"

    @property
    def section_area(self) -> float:
        """The gross cross-section normal to the load (width times thickness)."""
        return float(self.lengths[0] * self.lengths[2])

    def phases(self) -> list[int]:
        return sorted(int(v) for v in np.unique(self.labels) if v != HOLE)

    def estimate(self, generalized: int) -> int:
        """The degrees of freedom of its full box (an upper bound): three per node of
        the quadratic tetrahedra, ``generalized`` per corner node."""
        n = np.asarray(self.shape)
        return int(3 * np.prod(2 * n + 1) + generalized * np.prod(n + 1))

    def mesh(self):
        from morphogen.homogenization.bloch import voxel_tet_mesh
        from morphogen.image import VoxelImage

        image = VoxelImage(self.labels, spacing=self.spacing, periodic=False)
        return voxel_tet_mesh(image, phases=self.phases())


def _rows(centres: np.ndarray, middle: float, width: float, h: float) -> np.ndarray:
    """The voxels whose centres lie within half of ``width`` (at least half a voxel) of
    ``middle``."""
    return np.abs(centres - middle) <= max(width, h) / 2 + 1e-9 * h


def _shaped(p: dict, kind: str) -> Specimen:
    """A notched plate or a bar of the form."""
    width = float(p["width"])
    n = int(p["elements"])
    if not (width > 0 and 1 <= n <= 400):
        raise ValueError("width: positive; voxels across the width: from 1 to 400")
    h = width / n
    height = p["height"] or (width if kind == "notched_plate" else 2.0 * width)
    ny = max(2, round(float(height) / h))
    condition = p["condition"]
    if condition != "3d":
        nz = 1
    elif kind == "bar":
        nz = n
    else:
        nz = max(2, round(float(p["thickness"] or width / 4) / h))
    labels = np.zeros((n, ny, nz), np.uint8)
    xc = (np.arange(n) + 0.5) * h
    yc = (np.arange(ny) + 0.5) * h
    zc = (np.arange(nz) + 0.5) * h
    depth = float(p.get("notch_depth") or 0.0) * width
    rows = _rows(yc, ny * h / 2, float(p.get("notch_width") or 0.0) * width, h)
    if not 0.0 <= depth < (width if kind == "notched_plate" else width / 2):
        raise ValueError("notch depth: a fraction of the width (of its half for a bar)")
    if kind == "notched_plate":
        cut = xc < depth
        labels[np.ix_(cut, rows, np.ones(nz, bool))] = HOLE
        title = f"notched plate {width:g} x {ny * h:g}"
    else:
        cut = (xc < depth) | (xc > width - depth)
        labels[np.ix_(cut, rows, np.ones(nz, bool))] = HOLE
        if condition == "3d":  # a groove all around the square section
            side = (zc < depth) | (zc > nz * h - depth)
            labels[np.ix_(np.ones(n, bool), rows, side)] = HOLE
        weak = float(p.get("imperfection") or 0.0)
        if weak:
            if not 0.0 < weak < 1.0:
                raise ValueError("imperfection: a fraction between 0 and 1")
            layer = labels[:, rows, :]
            layer[layer != HOLE] = WEAK
            labels[:, rows, :] = layer
        title = f"bar {width:g} x {ny * h:g}" + (" notched" if depth else "")
    if np.all(labels == HOLE):
        raise ValueError("the notch leaves no material")
    return Specimen(labels, (h, h, h), condition, title, depth if kind == "notched_plate" else 0.0)


def _phase_names(item: dict) -> dict[int, str]:
    names = item.get("phases") or {}
    if not names and item.get("micro") is not None:
        names = item["micro"].phase_names
    return {int(k): str(v) for k, v in names.items()}


def _load_bearing(solid: np.ndarray) -> np.ndarray:
    """The voxels of ``solid`` joined by faces to both gripped faces (``y = 0`` and ``y =
    height``): the others would float, or hang by an edge (singular tangents)."""
    from scipy import ndimage

    labels, count = ndimage.label(solid)
    if count == 0:
        return np.zeros_like(solid)
    keep = np.intersect1d(np.unique(labels[:, 0, :]), np.unique(labels[:, -1, :]))
    return np.isin(labels, keep[keep > 0])


def _of_morphology(p: dict) -> Specimen:
    """The specimen of a morphology of the workspace: its voxel image (the middle x-y
    section in plane conditions), its holes removed and only the solid joining the grips."""
    from morphogen.gui.api import _voxels

    item = p["source"]
    micro = item["micro"]
    n = int(p["resolution"])
    if not 4 <= n <= 256:
        raise ValueError("voxels per edge: from 4 to 256")
    _voxels(micro.domain, n, 6.0, "voxel image")
    image = micro.voxelize(shape=n)
    data = np.asarray(image.data)
    if data.dtype.kind not in "iub":
        raise ValueError("the morphology gives no phase image")
    if p["condition"] != "3d":
        data = data[:, :, data.shape[2] // 2][:, :, None]
    names = _phase_names(item)
    holes = p["holes"]
    if not holes:
        holes = [k for k, v in names.items() if v.strip().lower() in HOLE_NAMES]
    solid = ~np.isin(data, list(holes))
    kept = _load_bearing(solid)
    if not kept.any():
        raise ValueError(
            f"the solid of {item['title']} (without the phases {sorted(holes)}) does not join "
            "its faces y = 0 and y = height: no path for the load"
        )
    labels = np.where(kept, data, HOLE).astype(np.uint8)
    section = "section" if p["condition"] != "3d" else "volume"
    return Specimen(labels, tuple(image.spacing), p["condition"], f"{section} of {item['title']}")


def _specimen(p: dict) -> Specimen:
    kind = p["specimen"]
    return _of_morphology(p) if kind == "morphology" else _shaped(p, kind)


def _check_size(specimen: Specimen, generalized: int, limit: int) -> None:
    dofs = specimen.estimate(generalized)
    if dofs > limit:
        raise ValueError(
            f"{specimen.title}: about {dofs} degrees of freedom, more than the {limit} of the "
            "interface (every iteration factorizes the tangent); choose fewer voxels, or a "
            "plane condition"
        )


def _supports(problem, specimen: Specimen, displacement: float, grips: str, extra=()) -> list:
    """The prescribed degrees of freedom: the face ``y = 0`` held along y, the face ``y =
    height`` moved by ``displacement`` (the last entry: its reaction is the force), the
    rigid motions stopped at two corners (rollers) or the faces held sideways (clamped);
    in plane strain no displacement along z; ``extra`` entries after the first ones."""
    x = problem.mesh.nodes
    tol = 1e-9 * float(specimen.lengths.max())
    bottom = np.flatnonzero(np.abs(x[:, 1]) < tol)
    top = np.flatnonzero(np.abs(x[:, 1] - specimen.lengths[1]) < tol)
    if not (len(bottom) and len(top)):
        raise ValueError(f"{specimen.title}: no material on a gripped face")
    plane = specimen.condition == "plane_strain"
    side = [0] if plane else [0, 2]
    entries = [(np.arange(len(x)), 2, 0.0)] if plane else []
    if grips == "clamped":
        entries += [(bottom, [1, *side], 0.0), (top, side, 0.0)]
    else:
        entries.append((bottom, 1, 0.0))
        pin = bottom[np.argmin(np.linalg.norm(x[bottom] - x[bottom].min(axis=0), axis=1))]
        entries.append((pin, side, 0.0))
        if not plane:  # no rotation about y
            corner = x[bottom].min(axis=0) + np.array([specimen.lengths[0], 0.0, 0.0])
            far = bottom[np.argmin(np.linalg.norm(x[bottom] - corner, axis=1))]
            if far != pin:
                entries.append((far, 2, 0.0))
    entries += list(extra)
    entries.append((top, 1, float(displacement)))
    return entries


def _run(problem, entries, ctx, *, steps: int, start: float, span: float, label: str, **options):
    """The loading path of a problem (:meth:`~morphogen.micromorphic.space.MixedProblem.run`
    with its fields), its progress reported from ``start`` to ``start + span`` as the load
    of the last entry grows.

    When the steps stop converging (a crack running unstably snaps back, which the growing
    displacement cannot follow), the converged steps are kept, with a warning: the
    assemblies are watched, and the internal variables passed to one that are those the
    last tangent returned mark the convergence of its step (the solver then takes its
    degrees of freedom, forces and internal variables)."""
    from morphogen.micromorphic.space import PathResult

    value = entries[-1][2]
    sets = [problem.space.dofs(np.atleast_1d(n), np.atleast_1d(c)) for n, c, _ in entries]
    dof = int(sets[-1][0, 0])
    full = float(value) or 1.0
    inner = problem._assemble
    path: dict = {"loads": [], "reactions": [], "fields": [], "states": [], "last": None}

    def assemble(values, state, tangent):
        if not path["states"]:  # the initial state
            path["loads"].append(0.0)
            path["reactions"].append([np.zeros(s.shape[1]) for s in sets])
            path["fields"].append(np.zeros_like(values))
            path["states"].append(state.copy())
        elif path["last"] is not None and state is path["last"][2]:  # a step converged
            u, r, kept = path["last"]
            path["loads"].append(float(u[dof]) / full)
            path["reactions"].append([r[s].sum(axis=0) for s in sets])
            path["fields"].append(u)
            path["states"].append(kept)
            path["last"] = None
        load = min(max(float(values[dof]) / full, 0.0), 1.0)
        ctx.progress(start + span * load, f"{label}: load {100 * load:.0f} %")
        out = inner(values, state, tangent)
        if tangent:
            path["last"] = (np.array(values), out[0], out[2])
        return out

    problem._assemble = assemble
    try:
        options.setdefault("max_cuts", 10)  # steps of a thousandth of the load at least
        return problem.run(entries, steps=int(steps), fields=True, **options)
    except RuntimeError as exc:
        if len(path["loads"]) < 2:
            raise RuntimeError(
                f"{label}: {exc} (more steps or a smaller strain may help)"
            ) from None
        import warnings

        warnings.warn(
            f"{label}: stopped at {100 * path['loads'][-1]:.0f} % of the load ({exc}: a crack "
            "running unstably, or steps too large); the converged steps are shown",
            stacklevel=2,
        )
        n = len(path["loads"])
        return PathResult(
            np.array(path["loads"]),
            path["reactions"],
            np.zeros(n, int),
            path["fields"][-1],
            path["states"][-1],
            path["fields"],
            path["states"],
        )
    finally:
        del problem._assemble


def _curve(result, specimen: Specimen, displacement: float) -> dict:
    """The displacement of the pulled face and its force (per unit thickness for a plate
    of one layer) at the steps, and the nominal stress (over the gross section)."""
    force = np.array([float(r[-1][0]) for r in result.reactions])
    u = displacement * np.asarray(result.loads)
    per = force / specimen.lengths[2] if specimen.planar else force
    return {"u": u, "force": per, "stress": force / specimen.section_area}


def _voxel_index(mesh, specimen: Specimen) -> np.ndarray:
    """The voxel of each element (by its centroid)."""
    centres = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    ijk = np.floor(centres / np.asarray(specimen.spacing)).astype(np.int64)
    ijk = np.clip(ijk, 0, np.asarray(specimen.shape) - 1)
    return np.ravel_multi_index(ijk.T, specimen.shape)


def _on_voxels(index: np.ndarray, values, shape) -> np.ndarray:
    """A field of the elements on their voxels (the mean of the six tetrahedra of equal
    volume), NaN without material."""
    n = int(np.prod(shape))
    count = np.bincount(index, minlength=n)
    total = np.bincount(index, weights=np.asarray(values, dtype=float), minlength=n)
    out = np.full(n, np.nan)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out.reshape(shape)


def _section(volume: np.ndarray) -> np.ndarray:
    """The middle x-y section of a voxel field (x, y)."""
    return volume[:, :, volume.shape[2] // 2]


def _cmap(name: str):
    import matplotlib

    return matplotlib.colormaps[name].with_extremes(bad="white")


#: The colormaps and ranges of the fields drawn
FIELD_STYLE = {
    "damage": ("inferno", (0.0, 1.0)),
    "triaxiality": ("RdBu_r", None),
}


def _maps_figure(maps: dict, specimen: Specimen, title: str = ""):
    """Maps of fields on the voxels of a specimen (its middle section): name -> (x, y)
    array, side by side."""
    from matplotlib.figure import Figure

    w, h = specimen.lengths[:2]
    cols = len(maps)
    # panels of the shape of the section (3 inches high), each with its colour bar
    panel = 3.0 * float(np.clip(w / h, 0.25, 2.5)) + 1.3
    fig = Figure(figsize=(cols * panel, 3.9), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, cols))
    for ax, (name, plane) in zip(axes, maps.items(), strict=True):
        key = next((k for k in FIELD_STYLE if name.startswith(k)), "")
        cmap, clim = FIELD_STYLE.get(key, ("viridis", None))
        finite = plane[np.isfinite(plane)]
        if clim is None and key == "triaxiality" and finite.size:
            top = float(np.abs(finite).max()) or 1.0
            clim = (-top, top)
        image = ax.imshow(
            np.asarray(plane, float).T,
            origin="lower",
            extent=(0, w, 0, h),
            cmap=_cmap(cmap),
            interpolation="nearest",
            **({"vmin": clim[0], "vmax": clim[1]} if clim else {}),
        )
        fig.colorbar(image, ax=ax, shrink=0.85, aspect=25)
        ax.set_title(name, fontsize=9, wrap=True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def _curves_figure(curves: dict, xlabel: str, ylabel: str, title: str = ""):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    for label, (x, y) in curves.items():
        ax.plot(x, y, "o-", ms=2.5, lw=1.5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if len(curves) > 1:
        ax.legend(fontsize=8, frameon=False)
    return fig


def _field_steps(ctx, specimen: Specimen, fields: dict, captions, name: str) -> None:
    """The fields on the voxels of the specimen at each step of the loading
    (``<name>_step<k>.vti``, one array per field, NaN without material) and their manifest:
    the 3D view of the page goes through them, the export makes their video."""
    from morphogen.image import VoxelImage
    from morphogen.io.vtk import write_vti

    count = len(next(iter(fields.values())))
    steps = []
    for k in range(count):
        file = f"{name}_step{k + 1:03d}.vti"
        images = [VoxelImage(np.asarray(v[k], dtype=np.float32), spacing=specimen.spacing,
                             name=field) for field, v in fields.items()]  # fmt: skip
        write_vti(ctx.path(file), images)
        label = captions[k] if k < len(captions) else f"step {k + 1}"
        steps.append({"file": file, "time": float(k + 1), "label": label})
    ctx.steps(name, steps)


def _field_video(ctx, volumes, specimen: Specimen, name: str, label: str, captions) -> None:
    """A video of a field on the middle section of a specimen along the steps."""
    from morphogen.viz.video import time_section_frames

    key = next((k for k in FIELD_STYLE if label.startswith(k)), "")
    cmap, clim = FIELD_STYLE.get(key, ("viridis", None))
    lo, hi = clim if clim else (None, None)
    frames = time_section_frames(
        volumes,
        2,
        specimen.shape[2] // 2,
        size=420,
        cmap=_cmap(cmap),
        vmin=lo,
        vmax=hi,
        captions=captions,
        label=label,
    )
    ctx.video(frames, name, f"{label} along the loading", fps=8, mode="once")


def _band_width(profile: np.ndarray, h: float) -> float:
    """The full width at half maximum of a profile along y (layers of thickness ``h``),
    interpolated linearly between the centres of the layers at both crossings (a band
    reaching a face ends there)."""
    profile = np.nan_to_num(np.asarray(profile, dtype=float))
    top = float(profile.max()) if profile.size else 0.0
    if not top > 0:
        return float("nan")
    half = 0.5 * top
    above = np.flatnonzero(profile > half)
    i, j = int(above[0]), int(above[-1])
    left = 0.0
    if i > 0:
        left = i - 0.5 + (half - profile[i - 1]) / (profile[i] - profile[i - 1])
    right = float(len(profile))
    if j < len(profile) - 1:
        right = j + 0.5 + (profile[j] - half) / (profile[j] - profile[j + 1])
    return float((right - left) * h)


def _crack(damage: np.ndarray, specimen: Specimen, level: float = 0.9) -> float:
    """The fraction of the ligament of a notched plate that the crack crosses: the reach
    along x of the voxels of its middle section damaged beyond ``level`` that join (by
    their sides) the voxels at the tip of the notch."""
    from scipy import ndimage

    section = np.nan_to_num(_section(damage)) > level
    h = specimen.spacing[0]
    tip = min(round(specimen.notch / h), section.shape[0] - 1)  # the column ahead of it
    rows = np.flatnonzero(_section(specimen.labels)[max(tip - 1, 0)] == HOLE)
    if not len(rows):  # a notch shorter than a voxel: the middle row
        rows = np.array([section.shape[1] // 2])
    labels, _ = ndimage.label(section)
    start = np.unique(labels[tip, rows])
    start = start[start > 0]
    if not len(start):
        return 0.0
    reach = np.flatnonzero(np.isin(labels, start).any(axis=1)).max() + 1
    return float((reach - tip) / (section.shape[0] - tip))


def _profile(volume: np.ndarray) -> np.ndarray:
    """The mean of a voxel field over each layer normal to y (0 without material)."""
    finite = np.isfinite(volume)
    total = np.where(finite, volume, 0.0).sum(axis=(0, 2))
    count = finite.sum(axis=(0, 2))
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def _overrides(value, name: str) -> dict[int, dict]:
    """Per-phase changes of the parameters of a material (a mapping phase -> mapping)."""
    if not value:
        return {}
    if not isinstance(value, dict) or not all(isinstance(v, dict) for v in value.values()):
        raise ValueError(f'{name}: a mapping phase -> parameters, e.g. {{"1": {{...}}}}')
    return {int(k): dict(v) for k, v in value.items()}


def _materials(specimen: Specimen, base, overrides: dict, weak: dict) -> dict:
    """The material of each label of a specimen: ``base`` (a dataclass) changed by the
    per-phase ``overrides`` and, in the weaker layer, by ``weak``."""
    import dataclasses

    out = {}
    for phase in specimen.phases():
        changes = dict(weak) if phase == WEAK else dict(overrides.get(phase, {}))
        try:
            out[phase] = dataclasses.replace(base, **changes) if changes else base
        except TypeError as exc:
            raise ValueError(f"phase {phase}: {exc}") from None
    return out


def _vtu(ctx, problem, values, name: str, cells: dict) -> None:
    """The fields of the elements and the displacement of the nodes, for ParaView."""
    from morphogen.io.meshes import write_vtu_tet

    u = problem.space.displacement(values)
    write_vtu_tet(
        ctx.path(f"{name}.vtu"),
        problem.mesh,
        {k: np.asarray(v, float) for k, v in cells.items()},
        {"displacement": u},
    )


def _view_3d(ctx, problem, values, field, name: str, title: str, cmap: str, clim=None) -> None:
    """A view of a 3D specimen deformed (true scale) and coloured by a field, when VTK
    renders off screen."""
    from morphogen.viz.video import deformed_view, vtk_can_render

    if not vtk_can_render():
        return
    rgb = deformed_view(
        problem.mesh,
        problem.space.displacement(values),
        np.asarray(field),
        magnification=1.0,
        cmap=cmap,
        clim=clim,
        title=title,
    )
    ctx.figure(rgb, name, f"{title} (3D view)")


def _captions(curve: dict) -> list[str]:
    return [f"u = {u:.3g}" for u in curve["u"]]


# -- microfoam -------------------------------------------------------------------------------


FOAMS = (
    ("isotropic", "isotropic foam: compressible plasticity, the parameters of the form"),
    ("nickel", "nickel foam of Dillard, Forest & Ienny (2006), MPa and mm"),
)


@tool(
    "microfoam",
    "damage",
    "Microfoam",
    "The microfoam model of metallic foams (Dillard, Forest & Ienny 2006): anisotropic "
    "compressible plasticity (the elliptic criterion σ_eq² = (3c/2) σ':H:σ' + F (P:σ)²) in "
    "a micromorphic continuum, whose microdeformation χ and characteristic length l_c set "
    "the width of the bands where plasticity localizes (with a softening: a negative "
    "hardening, or the softening branch beyond p_c); in small or finite strains (the "
    "densification of the foam, J). A specimen pulled or crushed: the force-displacement "
    "curve, the bands of the cumulated plastic strain p, the strain and the "
    "microdeformation, and their video; with the classical continuum for comparison "
    "(its bands one element wide).",
    [
        *_specimen_fields("bar", imperfection=0.1, elements=8, resolution=16),
        Field("loading", "choice", "compression", choices=("compression", "tension")),
        Field("strain", "float", 0.04, doc="nominal strain of the specimen at the end"),
        Field("steps", "int", 16),
        Field("kinematics", "choice", "small", choices=KINEMATICS),
        Field("material", "choice", "isotropic", choices=FOAMS),
        Field(
            "length",
            "float",
            0.02,
            label="length l_c",
            unit="length",
            doc="0: a classical continuum (the nickel foam: 0.1 mm in the article)",
        ),
        Field("young", "float", 100.0, when={"material": "isotropic"}),
        Field("poisson", "float", 0.3, when={"material": "isotropic"}),
        Field("yield_stress", "float", 1.0, when={"material": "isotropic"}),
        Field(
            "hardening",
            "float",
            -10.0,
            when={"material": "isotropic"},
            doc="linear hardening H (negative: a softening that localizes)",
        ),
        Field(
            "F",
            "float",
            0.3,
            when={"material": "isotropic"},
            doc="weight of the pressure in the criterion (0: von Mises)",
        ),
        Field(
            "micro",
            "floats",
            when={"material": "isotropic"},
            label="higher-order moduli λ, μ",
            doc="blank: 0 and young / 2",
        ),
        Field(
            "relative",
            "floats",
            when={"material": "isotropic"},
            label="relative moduli λ_s, μ_s, μ_c",
            doc="blank: 0, 10 young and 10 young",
        ),
        Field(
            "orientation",
            "choice",
            "TD",
            when={"material": "nickel"},
            choices=(("TD", "TD along the load"), ("RD", "RD along the load")),
        ),
        Field(
            "softening",
            "floats",
            doc="p_c, r, w: Y times r + (1 - r) exp(-((p - p_c)/w)²) beyond p_c (blank: none)",
        ),
        Field(
            "phases",
            "json",
            label="per-phase parameters",
            doc='changes of the material by phase, e.g. {"1": {"yield_stress": 2}}',
        ),
        Field("classical", "bool", False, label="compare with the classical continuum"),
        Field(
            "video",
            "choice",
            "none",
            choices=(
                ("p", "cumulated plastic strain"),
                ("strain", "strain along the load"),
                ("chi", "microdeformation along the load"),
                ("none", "no video"),
            ),
            doc="of a field along the loading (the fields of the steps are in the 3D view, and "
            "the export makes their video)",
        ),
    ],
    order=1,
    groups=(
        ("Specimen", "specimen"),
        ("Loading", "loading"),
        ("Material", "material"),
        ("Output", "classical", True),
    ),
)
def microfoam(p: dict, ctx) -> dict:
    from morphogen.micromorphic import CHI, Microfoam, MicrofoamProblem

    specimen = _specimen(p)
    length = float(p["length"] or 0.0)
    if length < 0:
        raise ValueError("length: non-negative (0: a classical continuum)")
    softening = tuple(p["softening"]) if p["softening"] else None
    if softening is not None and len(softening) != 3:
        raise ValueError("softening: p_c, r and w")
    if p["material"] == "nickel":
        axes = None if p["orientation"] == "TD" else ((0, 1, 0), (1, 0, 0), (0, 0, 1))
        foam = Microfoam.nickel_foam(length=length or None, axes=axes, softening=softening)
    else:
        young = float(p["young"])
        micro = tuple(p["micro"] or (0.0, young / 2))
        relative = tuple(p["relative"] or (0.0, 10 * young, 10 * young))
        if len(micro) != 2 or len(relative) != 3:
            raise ValueError("higher-order moduli: λ and μ; relative moduli: λ_s, μ_s and μ_c")
        foam = Microfoam(
            young=young,
            poisson=float(p["poisson"]),
            yield_stress=float(p["yield_stress"]),
            hardening=float(p["hardening"]),
            F=float(p["F"]),
            softening=softening,
            length=length or None,
            micro=micro,
            relative=relative,
        )
    weak = {"yield_stress": foam.yield_stress * (1.0 - float(p.get("imperfection") or 0.0))}
    overrides = _overrides(p["phases"], "per-phase parameters")
    sign = -1.0 if p["loading"] == "compression" else 1.0
    strain = abs(float(p["strain"]))
    displacement = sign * strain * specimen.lengths[1]
    cases = [("micromorphic" if length else "classical", foam)]
    if p["classical"] and length:
        import dataclasses

        cases.append(("classical", dataclasses.replace(foam, length=None)))
    for _, material in cases:
        _check_size(
            specimen,
            9 if material.length else 0,
            MAX_MICROMORPHIC_DOFS if material.length else MAX_DOFS,
        )
    ctx.progress(0.02, "mesh")
    mesh = specimen.mesh()
    runs = {}
    span = 0.9 / len(cases)
    for k, (name, material) in enumerate(cases):
        problem = MicrofoamProblem(
            mesh, _materials(specimen, material, overrides, weak), kinematics=p["kinematics"]
        )
        extra = []
        if problem.micromorphic and specimen.condition == "plane_strain":
            corners = np.flatnonzero(problem.space.corner)
            odd = [CHI[c] for c in ("chi13", "chi31", "chi23", "chi32", "chi33")]
            extra.append((corners, odd, 0.0))  # no microdeformation out of the plane
        entries = _supports(problem, specimen, displacement, p["grips"], extra)
        hardening = softening is None and all(m.hardening >= 0 for m in problem.materials.values())
        result = _run(
            problem,
            entries,
            ctx,
            steps=p["steps"],
            start=0.02 + k * span,
            span=span,
            label=f"{name} continuum",
            refactor=0.25 if hardening else 0.0,
            predictor=True,
        )
        runs[name] = (problem, result)
    ctx.progress(0.93, "figures")
    first = cases[0][0]
    problem, result = runs[first]
    index = _voxel_index(problem.mesh, specimen)
    curves = {name: _curve(res, specimen, displacement) for name, (_, res) in runs.items()}
    ylabel = "force per unit thickness" if specimen.planar else "force"
    ctx.figure(
        _curves_figure(
            {k: (sign * c["u"], sign * c["force"]) for k, c in curves.items()},
            "displacement of the pulled face" if sign > 0 else "shortening",
            ylabel,
            specimen.title,
        ),
        "force_displacement",
        "force-displacement",
    )
    rows = []
    for name, c in curves.items():
        rows += [
            {
                "continuum": name,
                "displacement": float(u),
                "force": float(f),
                "nominal_stress": float(s),
            }
            for u, f, s in zip(c["u"], c["force"], c["stress"], strict=True)
        ]
    ctx.table(rows, "force_displacement", "force-displacement")
    fields = problem.fields(result.values, result.state)
    finite = p["kinematics"] == "finite"
    maps = {
        "p": fields["p"],
        ("log strain yy" if finite else "strain yy"): fields["strain"][:, 1, 1],
    }
    if problem.micromorphic:
        maps["χyy - 1" if finite else "χyy"] = fields["chi"][:, 1, 1]
    if finite:
        maps["J"] = fields["J"]
    volumes = {k: _on_voxels(index, v, specimen.shape) for k, v in maps.items()}
    ctx.figure(
        _maps_figure(
            {k: _section(v) for k, v in volumes.items()},
            specimen,
            f"{first} continuum at the end: p the cumulated plastic strain, χ the "
            "microdeformation" + (", J the change of volume" if finite else ""),
        ),
        "fields",
        "fields at the end",
    )
    values = {
        "degrees of freedom": problem.space.n_dofs,
        "elements": problem.mesh.n_elements,
        "steps": len(result.loads) - 1,
        "Newton iterations": int(np.sum(result.iterations)),
    }
    p_profile = _profile(volumes["p"])
    for name, c in curves.items():
        stress = sign * c["stress"]
        values[f"peak nominal stress ({name})"] = float(stress.max())
        values[f"final nominal stress ({name})"] = float(stress[-1])
    values["largest p"] = float(np.max(fields["p"]))
    values["band width (p at half maximum, along y)"] = _band_width(p_profile, specimen.spacing[1])
    if finite:
        values["smallest J"] = float(np.min(fields["J"]))
    if "classical" in runs and first != "classical":
        cp, cr = runs["classical"]
        cvol = _on_voxels(
            _voxel_index(cp.mesh, specimen), cp.fields(cr.values, cr.state)["p"], specimen.shape
        )
        values["band width, classical continuum"] = _band_width(_profile(cvol), specimen.spacing[1])
        ctx.figure(
            _maps_figure(
                {
                    "p, micromorphic": _section(volumes["p"]),
                    "p, classical": _section(cvol),
                },
                specimen,
                "the bands of the two continua",
            ),
            "bands",
            "micromorphic and classical bands",
        )
    if not specimen.planar:
        _view_3d(ctx, problem, result.values, fields["p"], "view_p", "p", "viridis")
    _vtu(
        ctx,
        problem,
        result.values,
        "microfoam",
        {
            "p": fields["p"],
            "strain_yy": fields["strain"][:, 1, 1],
            "chi_yy": fields["chi"][:, 1, 1],
        },
    )
    ctx.progress(0.94, "fields of the steps")
    along: dict = {"p": [], "strain_yy": [], "chi_yy": []}
    for u, state in zip(result.fields, result.states, strict=True):
        f = problem.fields(u, state)
        for name, v in (("p", f["p"]), ("strain_yy", f["strain"][:, 1, 1]),
                        ("chi_yy", f["chi"][:, 1, 1])):  # fmt: skip
            along[name].append(_on_voxels(index, v, specimen.shape))
    _field_steps(ctx, specimen, along, _captions(curves[first]), "microfoam")
    if p["video"] != "none":
        ctx.progress(0.96, "video")
        chosen = {"p": "p", "strain": "strain_yy", "chi": "chi_yy"}[p["video"]]
        label = {"p": "p", "strain": "strain yy", "chi": "chi yy"}[p["video"]]
        _field_video(ctx, along[chosen], specimen, f"video_{p['video']}", label,
                     _captions(curves[first]))  # fmt: skip
    return {"values": values, "title": f"Microfoam: {specimen.title}",
            "fields3d": {"field": "p", "cmap": "viridis"}}  # fmt: skip


# -- ductile damage --------------------------------------------------------------------------


TRIAXIALITY_LAWS = (
    ("none", "none: the plastic strain alone"),
    ("rice_tracey", "Rice & Tracey (1969): exp(β (η - 1/3))"),
    ("johnson_cook", "Johnson & Cook (1985): ε_f = D1 + D2 exp(D3 η)"),
    ("lemaitre", "Lemaitre (1985): R_ν = 2/3 (1 + ν) + 3 (1 - 2ν) η²"),
)


@tool(
    "ductile",
    "damage",
    "Ductile damage with triaxiality",
    "Micromorphic ductile damage (Forest 2009; the implicit gradient ductile models of "
    "Engelen, Geers & Baaijens 2003): von Mises elastoplasticity of the effective stress, "
    "a damage d = d_max (1 - exp(-(κ - κ0)/κ_c)) driven by a micromorphic cumulated "
    "plastic strain χ - ℓ² Δχ = e, whose length ℓ sets the width of the band of damage; "
    "e the plastic strain weighted by the stress triaxiality η (ė = w(η) ṗ, w(1/3) = 1: "
    "Rice & Tracey's growth of voids, Johnson & Cook's strain to failure, Lemaitre's "
    "energy). A specimen pulled: the force-displacement curve, the fields of the damage, "
    "of the triaxiality and of the plastic strain, a video of the damage growing; with the "
    "plastic strain alone for comparison. In plane strain the triaxial stress ahead of a "
    "notch concentrates the damage there and the crack runs across the ligament.",
    [
        *_specimen_fields("notched_plate", imperfection=0.1),
        Field("strain", "float", 0.016, doc="nominal strain of the specimen at the end"),
        Field("steps", "int", 24),
        Field("kinematics", "choice", "small", choices=KINEMATICS),
        Field("young", "float", 200.0),
        Field("poisson", "float", 0.3),
        Field("yield_stress", "float", 0.4),
        Field("hardening", "float", 10.0, doc="linear hardening H"),
        Field("voce", "floats", doc="Q1, b1 (and Q2, b2) of the Voce hardening (blank: none)"),
        Field("onset", "float", 0.002, label="onset κ0", doc="of the damage"),
        Field("critical", "float", 0.02, label="scale κ_c", doc="of the damage law"),
        Field("max_damage", "float", 0.99),
        Field(
            "length",
            "float",
            label="length ℓ",
            unit="length",
            doc="blank: twice the voxels (at least two voxels)",
        ),
        Field("triaxiality", "choice", "rice_tracey", choices=TRIAXIALITY_LAWS),
        Field(
            "triaxiality_parameters",
            "floats",
            when={"triaxiality": ["rice_tracey", "johnson_cook"]},
            doc="Rice & Tracey: β (blank: 1.5); Johnson & Cook: D1, D2, D3 (blank: 0.05, "
            "3.44, -2.12, their 4340 steel)",
        ),
        Field(
            "compare",
            "bool",
            False,
            label="compare with the plastic strain alone",
            when={"triaxiality": ["rice_tracey", "johnson_cook", "lemaitre"]},
        ),
        Field(
            "phases",
            "json",
            label="per-phase parameters",
            doc='changes of the material by phase, e.g. {"1": {"yield_stress": 0.8}}',
        ),
        Field(
            "video",
            "choice",
            "none",
            doc="of a field along the loading (the fields of the steps are in the 3D view, and "
            "the export makes their video)",
            choices=(
                ("damage", "damage"),
                ("triaxiality", "triaxiality"),
                ("p", "cumulated plastic strain"),
                ("none", "no video"),
            ),
        ),
    ],
    order=2,
    groups=(
        ("Specimen", "specimen"),
        ("Loading", "strain"),
        ("Material", "young"),
        ("Output", "video", True),
    ),
)
def ductile(p: dict, ctx) -> dict:
    from morphogen.micromorphic import DuctileProblem, MicromorphicDuctile

    specimen = _specimen(p)
    h = float(min(specimen.spacing[:2]))
    length = float(p["length"] or 2.0 * h)
    law = p["triaxiality"]
    parameters = tuple(p["triaxiality_parameters"] or ())
    if law == "johnson_cook" and not parameters:
        parameters = (0.05, 3.44, -2.12)
    voce = tuple(p["voce"] or ())
    if len(voce) % 2 or len(voce) > 4:
        raise ValueError("voce: Q1, b1 (and Q2, b2)")
    common = dict(
        young=float(p["young"]),
        poisson=float(p["poisson"]),
        yield_stress=float(p["yield_stress"]),
        hardening=float(p["hardening"]),
        voce=tuple(zip(voce[::2], voce[1::2], strict=True)),
        length=length,
        onset=float(p["onset"]),
        critical=float(p["critical"]),
        max_damage=float(p["max_damage"]),
    )
    cases = [
        (law, MicromorphicDuctile(**common, triaxiality=law, triaxiality_parameters=parameters))
    ]
    if law != "none" and p["compare"]:
        cases.append(("none", MicromorphicDuctile(**common)))
    _check_size(specimen, 1, MAX_DOFS)
    weak = {"yield_stress": common["yield_stress"] * (1.0 - float(p.get("imperfection") or 0.0))}
    overrides = _overrides(p["phases"], "per-phase parameters")
    displacement = abs(float(p["strain"])) * specimen.lengths[1]
    ctx.progress(0.02, "mesh")
    mesh = specimen.mesh()
    runs = {}
    span = 0.9 / len(cases)
    for k, (name, material) in enumerate(cases):
        problem = DuctileProblem(
            mesh, _materials(specimen, material, overrides, weak), kinematics=p["kinematics"]
        )
        entries = _supports(problem, specimen, displacement, p["grips"])
        result = _run(
            problem,
            entries,
            ctx,
            steps=p["steps"],
            start=0.02 + k * span,
            span=span,
            label=f"triaxiality {name}",
            ordering="MMD_AT_PLUS_A",
        )
        runs[name] = (problem, result)
    ctx.progress(0.93, "figures")
    problem, result = runs[law]
    index = _voxel_index(problem.mesh, specimen)
    curves = {name: _curve(res, specimen, displacement) for name, (_, res) in runs.items()}
    label = {name: f"triaxiality: {name}" for name in runs}
    ylabel = "force per unit thickness" if specimen.planar else "force"
    ctx.figure(
        _curves_figure(
            {label[k]: (c["u"], c["force"]) for k, c in curves.items()},
            "displacement of the pulled face",
            ylabel,
            specimen.title,
        ),
        "force_displacement",
        "force-displacement",
    )
    rows = []
    for name, c in curves.items():
        rows += [
            {
                "triaxiality": name,
                "displacement": float(u),
                "force": float(f),
                "nominal_stress": float(s),
            }
            for u, f, s in zip(c["u"], c["force"], c["stress"], strict=True)
        ]
    ctx.table(rows, "force_displacement", "force-displacement")
    values = {
        "degrees of freedom": problem.space.n_dofs,
        "elements": problem.mesh.n_elements,
        "length ℓ": length,
        "steps": len(result.loads) - 1,
        "Newton iterations": int(np.sum(result.iterations)),
    }
    centres = problem.mesh.nodes[problem.mesh.elements[:, :4]].mean(axis=1)
    damages = {}
    for name, (prob, res) in runs.items():
        c = curves[name]
        damage = prob.damage(res.state)
        damages[name] = _on_voxels(index, damage, specimen.shape)
        values[f"peak force ({name})"] = float(c["force"].max())
        values[f"final force / peak ({name})"] = float(c["force"][-1] / c["force"].max())
        values[f"largest damage ({name})"] = float(damage.max())
        if specimen.notch:
            values[f"crack from the notch (d > 0.9), fraction of the ligament ({name})"] = _crack(
                damages[name], specimen
            )
    triax = problem.triaxiality(result.state)
    plastic = problem.plastic_strain(result.state)
    values["largest triaxiality"] = float(triax.max())
    if specimen.notch:  # ahead of the notch: within ℓ of its tip
        tip = np.hypot(centres[:, 0] - specimen.notch, centres[:, 1] - specimen.lengths[1] / 2)
        ahead = (tip < max(length, 1.5 * h)) & (centres[:, 0] > specimen.notch)
        if ahead.any():
            values["triaxiality ahead of the notch"] = float(triax[ahead].mean())
            for name, (prob, res) in runs.items():
                values[f"damage ahead of the notch / mean ({name})"] = float(
                    prob.damage(res.state)[ahead].mean() / max(prob.damage(res.state).mean(), 1e-30)
                )
    else:
        kappa = _on_voxels(index, result.state[:, :, 7].mean(axis=1), specimen.shape)
        values["band width (κ at half maximum, along y)"] = _band_width(_profile(kappa), h)
    maps = {
        f"damage ({law})": _section(damages[law]),
        "triaxiality": _section(_on_voxels(index, triax, specimen.shape)),
        "p": _section(_on_voxels(index, plastic, specimen.shape)),
    }
    if "none" in damages and law != "none":
        maps["damage (p alone)"] = _section(damages["none"])
    title = "end of the loading: damage, triaxiality η, cumulated plastic strain p"
    ctx.figure(_maps_figure(maps, specimen, title), "fields", "fields at the end")
    if not specimen.planar:
        _view_3d(
            ctx,
            problem,
            result.values,
            problem.damage(result.state),
            "view_damage",
            "damage",
            "inferno",
            (0.0, 1.0),
        )
    _vtu(
        ctx,
        problem,
        result.values,
        "ductile",
        {
            "damage": problem.damage(result.state),
            "triaxiality": triax,
            "p": plastic,
            "e": problem.driving(result.state),
        },
    )
    ctx.progress(0.94, "fields of the steps")
    picks = {"damage": problem.damage, "triaxiality": problem.triaxiality,
             "p": problem.plastic_strain}  # fmt: skip
    along = {name: [_on_voxels(index, pick(s), specimen.shape) for s in result.states]
             for name, pick in picks.items()}  # fmt: skip
    _field_steps(ctx, specimen, along, _captions(curves[law]), "ductile")
    if p["video"] != "none":
        ctx.progress(0.96, "video")
        _field_video(ctx, along[p["video"]], specimen, f"video_{p['video']}", p["video"],
                     _captions(curves[law]))  # fmt: skip
    return {"values": values, "title": f"Ductile damage: {specimen.title}",
            "fields3d": {"field": "damage", "cmap": "inferno"}}  # fmt: skip


# -- micromorphic damage (quasi-brittle) -----------------------------------------------------


@tool(
    "brittle",
    "damage",
    "Micromorphic damage",
    "The micromorphic damage of quasi-brittle solids (Forest 2009; the implicit gradient "
    "model of Peerlings, de Borst, Brekelmans & de Vree 1996): a micromorphic equivalent "
    "strain χ - ℓ² Δχ = ε_eq (the modified von Mises strain, k times weaker in tension than "
    "in compression), the damage d = 1 - (κ0/κ)(1 - α + α exp(-β (κ - κ0))) of its largest "
    "value κ, and σ = (1 - d) C : ε: a crack is a band of damage whose width the length ℓ "
    "sets. A specimen pulled: the force-displacement curve, the fields of the damage and "
    "of κ, a video of the damage growing.",
    [
        *_specimen_fields("bar", imperfection=0.2),
        Field("strain", "float", 0.006, doc="nominal strain of the specimen at the end"),
        Field("steps", "int", 30),
        Field("kinematics", "choice", "small", choices=KINEMATICS),
        Field("young", "float", 1000.0),
        Field("poisson", "float", 0.2),
        Field("threshold", "float", 1e-3, label="threshold κ0"),
        Field("alpha", "float", 0.99, label="α", doc="the residual stress of the law: 1 - α"),
        Field("beta", "float", 100.0, label="β", doc="its rate of softening"),
        Field("ratio", "float", 10.0, label="ratio k", doc="compressive / tensile strength"),
        Field("max_damage", "float", 0.9999),
        Field("length", "float", label="length ℓ", unit="length", doc="blank: twice the voxels"),
        Field(
            "variational",
            "bool",
            False,
            doc="the stress derived from the free energy (a penalty that does not degrade)",
        ),
        Field(
            "phases",
            "json",
            label="per-phase parameters",
            doc='changes of the material by phase, e.g. {"1": {"threshold": 0.002}}',
        ),
        Field(
            "video",
            "choice",
            "none",
            choices=(("damage", "damage"), ("kappa", "κ (largest χ)"), ("none", "no video")),
            doc="of a field along the loading (the fields of the steps are in the 3D view, and "
            "the export makes their video)",
        ),
    ],
    order=3,
    groups=(
        ("Specimen", "specimen"),
        ("Loading", "strain"),
        ("Material", "young"),
        ("Output", "video", True),
    ),
)
def brittle(p: dict, ctx) -> dict:
    from morphogen.micromorphic import DamageProblem, MicromorphicDamage

    specimen = _specimen(p)
    h = float(min(specimen.spacing[:2]))
    length = float(p["length"] or 2.0 * h)
    material = MicromorphicDamage(
        young=float(p["young"]),
        poisson=float(p["poisson"]),
        threshold=float(p["threshold"]),
        length=length,
        alpha=float(p["alpha"]),
        beta=float(p["beta"]),
        ratio=float(p["ratio"]),
        max_damage=float(p["max_damage"]),
        variational=bool(p["variational"]),
    )
    _check_size(specimen, 1, MAX_DOFS)
    weak = {"threshold": material.threshold * (1.0 - float(p.get("imperfection") or 0.0))}
    overrides = _overrides(p["phases"], "per-phase parameters")
    displacement = abs(float(p["strain"])) * specimen.lengths[1]
    ctx.progress(0.02, "mesh")
    problem = DamageProblem(
        specimen.mesh(), _materials(specimen, material, overrides, weak), kinematics=p["kinematics"]
    )
    entries = _supports(problem, specimen, displacement, p["grips"])
    result = _run(
        problem,
        entries,
        ctx,
        steps=p["steps"],
        start=0.02,
        span=0.9,
        label="micromorphic damage",
        ordering="MMD_AT_PLUS_A",
        predictor=True,
    )
    ctx.progress(0.93, "figures")
    index = _voxel_index(problem.mesh, specimen)
    curve = _curve(result, specimen, displacement)
    ylabel = "force per unit thickness" if specimen.planar else "force"
    ctx.figure(
        _curves_figure(
            {"micromorphic damage": (curve["u"], curve["force"])},
            "displacement of the pulled face",
            ylabel,
            specimen.title,
        ),
        "force_displacement",
        "force-displacement",
    )
    ctx.table(
        [
            {"displacement": float(u), "force": float(f), "nominal_stress": float(s)}
            for u, f, s in zip(curve["u"], curve["force"], curve["stress"], strict=True)
        ],
        "force_displacement",
        "force-displacement",
    )
    damage = problem.damage(result.state)
    kappa = result.state[:, :, 0].mean(axis=1)
    kappa_vox = _on_voxels(index, kappa, specimen.shape)
    damage_vox = _on_voxels(index, damage, specimen.shape)
    values = {
        "degrees of freedom": problem.space.n_dofs,
        "elements": problem.mesh.n_elements,
        "length ℓ": length,
        "steps": len(result.loads) - 1,
        "Newton iterations": int(np.sum(result.iterations)),
        "peak force": float(curve["force"].max()),
        "final force / peak": float(curve["force"][-1] / curve["force"].max()),
        "largest damage": float(damage.max()),
        "band width (κ at half maximum, along y)": _band_width(_profile(kappa_vox), h),
    }
    if specimen.notch:
        values["crack from the notch (d > 0.9), fraction of the ligament"] = _crack(
            damage_vox, specimen
        )
    ctx.figure(
        _maps_figure(
            {"damage": _section(damage_vox), "κ (largest χ)": _section(kappa_vox)},
            specimen,
            "end of the loading",
        ),
        "fields",
        "fields at the end",
    )
    if not specimen.planar:
        _view_3d(
            ctx, problem, result.values, damage, "view_damage", "damage", "inferno", (0.0, 1.0)
        )
    _vtu(ctx, problem, result.values, "damage", {"damage": damage, "kappa": kappa})
    ctx.progress(0.94, "fields of the steps")
    along = {name: [_on_voxels(index, s[:, :, column].mean(axis=1), specimen.shape)
                    for s in result.states]
             for name, column in (("damage", 1), ("kappa", 0))}  # fmt: skip
    _field_steps(ctx, specimen, along, _captions(curve), "brittle")
    if p["video"] != "none":
        ctx.progress(0.96, "video")
        _field_video(ctx, along[p["video"]], specimen, f"video_{p['video']}", p["video"],
                     _captions(curve))  # fmt: skip
    return {"values": values, "title": f"Micromorphic damage: {specimen.title}",
            "fields3d": {"field": "damage", "cmap": "inferno"}}  # fmt: skip
