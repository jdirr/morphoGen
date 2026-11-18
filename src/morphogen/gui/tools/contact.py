"""The tools of the tab Contact of the interface: a morphology of the workspace between
rigid tools (:mod:`morphogen.contact.experiments`). Its compression between two platens,
on which it rubs and under which the walls of its pores close on themselves (the nominal
stress against the strain, the areas in contact), and its indentation by a sphere or a
cone (the force against the depth, the modulus that it reads)."""

from __future__ import annotations

import numpy as np

from morphogen.gui.tools import Field, tab, tool

tab(
    "contact",
    "Contact",
    "A morphology between rigid tools, with the contact and Coulomb's friction of its "
    "surfaces: its compression between two platens until the walls of its pores close on "
    "themselves (a foam, a lattice or a porous solid that densifies), and its indentation "
    "by a sphere or a cone (the force against the depth and the modulus that it reads).",
    order=52,
)

AXES = (("2", "z"), ("1", "y"), ("0", "x"))
POINTS = (("gauss", "inside the faces"), ("nodes", "at the nodes of the faces"))
ELEMENTS = (
    ("1", "4-node tetrahedra (faster, stiffer)"),
    ("2", "10-node tetrahedra (curved faces on the interfaces)"),
)
KINEMATICS = (
    ("finite", "finite strains"),
    ("small", "small strains (the gaps are still those of the deformed surfaces)"),
)


def _sample_fields() -> list[Field]:
    """The fields of the sample of a test: the morphology, its materials and its mesh."""
    return [
        Field("source", "item", label="morphology", required=True),
        Field(
            "materials",
            "materials",
            of="source",
            doc="isotropic elastic constants of the phases, or voids: the mesh is that of "
            "the solid phases, the surface of the pores free to touch itself",
        ),
        Field(
            "size",
            "float",
            label="element size",
            unit="length",
            doc="of the conforming mesh of the sample (blank: a twelfth of its smallest "
            "edge); the tangent of the contacts is factorized: tens of thousands of nodes",
        ),
        Field("order", "choice", "1", label="elements", choices=ELEMENTS),
    ]


def _sample(p: dict, ctx):
    """The mesh of the solid of the morphology of a form and the materials of its phases."""
    from morphogen.config import MeshSpec
    from morphogen.gui.tools.mechanics import _elastic
    from morphogen.homogenization.bloch import voxel_tet_mesh
    from morphogen.homogenization.materials import materials_from_config
    from morphogen.mesh.conforming import conforming_mesh

    elastic = _elastic(p["source"], p.get("materials"))
    materials = materials_from_config(elastic)
    voids = {int(k) for k, m in materials.items() if m.void}
    if len(voids) == len(materials):
        raise ValueError("materials: every phase is a void")
    micro = p["source"]["micro"]
    lengths = np.asarray(micro.domain.lengths, dtype=float)
    size = p.get("size") or float(lengths.min()) / 12.0
    if not 0.0 < size <= 0.5 * float(lengths.min()):
        raise ValueError("element size: positive, at most half the smallest edge of the sample")
    order = int(p["order"])
    ctx.progress(0.03, "mesh")
    options = MeshSpec(size=size, order=order).kwargs(voids)
    options.update(hausdorff=size / 5.0, min_size=size / 3.0, periodic=False, cells="box")
    try:
        mesh = conforming_mesh(micro, **options)
    except ValueError as exc:  # no level set: six tetrahedra per voxel of its image
        if "level-set" not in str(exc):
            raise
        mesh = voxel_tet_mesh(micro.voxelize(voxel_size=size))
        mesh = mesh.quadratic() if order == 2 else mesh
    phases = {int(k) for k in np.unique(mesh.phase)}
    solid = sorted(phases - voids)
    if not solid:
        raise ValueError("the mesh of the sample has no solid phase")
    if len(solid) < len(phases):
        mesh = mesh.subset(solid)
    if mesh.n_nodes > 60000:
        raise ValueError(
            f"{mesh.n_nodes} nodes: the tangent of the contacts is factorized (a larger "
            "element size, or 4-node elements)"
        )
    return mesh, materials


def _curve(fig_axes, x, y, xlabel: str, ylabel: str) -> None:
    fig_axes.plot(x, y, "o-", ms=3)
    fig_axes.set_xlabel(xlabel)
    fig_axes.set_ylabel(ylabel)
    fig_axes.grid(True, lw=0.4, alpha=0.5)


@tool(
    "platen_compression",
    "contact",
    "Compression between two platens",
    "The compression of a morphology between two rigid platens across an axis, the lower "
    "one fixed, the upper one brought down: its solid presses on the platens with "
    "Coulomb's friction (none: lubricated platens, the sample widens freely; with friction "
    "it barrels) and, where the walls of its pores, its struts or its sides meet, on "
    "itself (a penalty on the integration points of the faces of the mesh, the nearest "
    "point of the surface that they face). The nominal stress (the force over the section "
    "of the box of the sample) against the nominal strain (the approach of the platens "
    "over its height), the apparent modulus and its rise as the walls close, the areas in "
    "contact and the energy that friction dissipated; the deformed sample with the "
    "pressure of its contacts.",
    [
        *_sample_fields(),
        Field("axis", "choice", "2", label="axis of the compression", choices=AXES),
        Field(
            "strain",
            "float",
            0.1,
            label="compression at the end",
            doc="nominal: the approach of the platens over the height of the sample (0.1: "
            "compressed by 10 %)",
        ),
        Field("steps", "int", 20),
        Field(
            "friction",
            "float",
            0.0,
            label="friction of the platens",
            doc="Coulomb's coefficient (0: lubricated)",
        ),
        Field(
            "self_contact",
            "bool",
            True,
            label="the sample touches itself",
            doc="the walls of its pores press on each other where they meet (with the "
            "friction below); otherwise they go through each other",
        ),
        Field(
            "self_friction",
            "float",
            label="friction of the sample on itself",
            when={"self_contact": [True]},
            doc="blank: that of the platens",
        ),
        Field(
            "points",
            "choice",
            "gauss",
            label="contact points",
            choices=POINTS,
            doc="of the faces of the sample: inside them, or at their nodes, for a sample "
            "whose corners and edges meet the platens, or its own walls, first (a lattice "
            "cut through its struts)",
        ),
        Field("kinematics", "choice", "finite", choices=KINEMATICS),
    ],
    order=1,
    menu="Tests",
    groups=(("Sample", "source"), ("Test", "axis"), ("Contact", "friction")),
)
def platen_compression(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen.contact import compression

    strain, steps = p["strain"], p["steps"]
    if not 0.0 < strain < 0.9:
        raise ValueError("compression at the end: between 0 and 0.9")
    if not 1 <= steps <= 400:
        raise ValueError("steps: from 1 to 400")
    if p["friction"] < 0 or (p.get("self_friction") or 0.0) < 0:
        raise ValueError("friction: 0 or more")
    mesh, materials = _sample(p, ctx)
    ctx.progress(0.1, f"compression ({mesh.n_nodes} nodes)")
    res = compression(
        mesh,
        materials,
        -strain,
        axis=int(p["axis"]),
        friction=p["friction"],
        self_contact=p["self_contact"],
        self_friction=p.get("self_friction"),
        points=p["points"],
        steps=steps,
        kinematics=p["kinematics"],
    )
    ctx.progress(0.9, "results")
    ctx.table(res.rows(), "curve", "curve of the compression")
    res.problem.write(ctx.path("compressed"), res.result)
    fig = Figure(figsize=(9.4, 3.9), layout="constrained")
    left, right = fig.subplots(1, 2)
    _curve(left, -res.strain, -res.stress, "nominal compression", "nominal stress")
    right.plot(-res.strain, res.area[:, 0], "o-", ms=3, label="fixed platen")
    right.plot(-res.strain, res.area[:, 1], "s-", ms=3, label="moving platen")
    if p["self_contact"]:
        right.plot(-res.strain, res.self_area, "^-", ms=3, label="the sample with itself")
    right.set_xlabel("nominal compression")
    right.set_ylabel("area in contact")
    right.grid(True, lw=0.4, alpha=0.5)
    right.legend(fontsize=8)
    ctx.figure(fig, "compression", "stress and areas in contact against the compression")
    tangent = res.tangent()
    values = {
        "elements": mesh.n_elements,
        "nodes": mesh.n_nodes,
        "height of the sample": res.height,
        "section of its box": res.section,
        "nominal stress at the end": float(res.stress[-1]),
        "apparent modulus (first step)": res.modulus,
        "tangent modulus at the end": float(tangent[-1]),
        "area in contact with the fixed platen": float(res.area[-1, 0]),
        "area in contact with the moving platen": float(res.area[-1, 1]),
        "area of the sample in contact with itself": float(res.self_area[-1]),
        "energy dissipated by friction": float(res.friction[-1]),
        "Newton iterations": int(np.sum(res.result.iterations)),
    }
    return {"values": values, "title": f"Compression of {p['source']['title']} between platens"}


@tool(
    "indentation_test",
    "contact",
    "Indentation",
    "The indentation of the upper face of a morphology by a rigid sphere or cone, to a "
    "depth and out again: the force against the depth, the area of the contact (that of "
    "the integration points that press), the mean pressure (the hardness of an elastic "
    "sample) and the reduced modulus E / (1 - ν²) that the force reads by Hertz's law (a "
    "sphere) or Sneddon's (a cone) — that of the sample while the contact is small against "
    "it and large against its heterogeneities; for a sphere, also with the rigid base "
    "under the sample (a layer of finite thickness: Dimitriadis et al.), and from the "
    "slope of the unloading (Oliver and Pharr).",
    [
        *_sample_fields(),
        Field("axis", "choice", "2", label="axis of the indenter", choices=AXES),
        Field(
            "indenter",
            "choice",
            "sphere",
            choices=(("sphere", "sphere"), ("cone", "cone")),
        ),
        Field(
            "radius",
            "float",
            label="radius of the sphere",
            unit="length",
            when={"indenter": ["sphere"]},
            doc="blank: a quarter of the smallest edge of the sample",
        ),
        Field(
            "half_angle",
            "float",
            70.3,
            label="half-angle of the cone (degrees)",
            when={"indenter": ["cone"]},
            doc="from its axis; 70.3: the cone of the area of the pyramids of Berkovich "
            "and Vickers",
        ),
        Field(
            "depth",
            "float",
            label="depth",
            unit="length",
            doc="of the indentation (blank: a hundredth of the radius of the sphere, a "
            "fortieth of the height of the sample for a cone)",
        ),
        Field(
            "point",
            "floats",
            label="point of the face",
            doc="its two coordinates along the face (blank: its middle)",
        ),
        Field("steps", "int", 8),
        Field(
            "unload",
            "bool",
            True,
            label="then unload",
            doc="withdraw the indenter in as many steps: the slope of the unloading",
        ),
        Field(
            "base",
            "choice",
            "fixed",
            label="lower face of the sample",
            choices=(("fixed", "glued to its support"), ("smooth", "sliding on it")),
        ),
        Field("friction", "float", 0.0, label="friction of the indenter"),
        Field(
            "poisson",
            "float",
            label="Poisson's ratio of the sample",
            when={"indenter": ["sphere"]},
            doc="to read the modulus with the rigid base under the sample (blank: not read)",
        ),
        Field("kinematics", "choice", "finite", choices=KINEMATICS),
    ],
    order=2,
    menu="Tests",
    groups=(("Sample", "source"), ("Indenter", "axis"), ("Test", "depth")),
)
def indentation_test(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen.contact import hertz_sphere, indentation, sneddon_cone

    steps = p["steps"]
    if not 1 <= steps <= 200:
        raise ValueError("steps: from 1 to 200")
    if p["friction"] < 0:
        raise ValueError("friction: 0 or more")
    mesh, materials = _sample(p, ctx)
    axis = int(p["axis"])
    edges = np.asarray(mesh.size, dtype=float)
    sphere = p["indenter"] == "sphere"
    radius = (p.get("radius") or 0.25 * float(edges.min())) if sphere else None
    depth = p.get("depth") or (0.01 * radius if sphere else edges[axis] / 40.0)
    point = p.get("point") or None
    if point is not None and len(point) != 2:
        raise ValueError("point of the face: its two coordinates along the face")
    poisson = p.get("poisson")
    if poisson is not None and not -1.0 < poisson <= 0.5:
        raise ValueError("Poisson's ratio of the sample: in (-1, 0.5]")
    ctx.progress(0.1, f"indentation ({mesh.n_nodes} nodes)")
    res = indentation(
        mesh,
        materials,
        depth,
        indenter=p["indenter"],
        radius=radius,
        half_angle=p["half_angle"],
        point=point,
        axis=axis,
        friction=p["friction"],
        steps=steps,
        unload=p["unload"],
        base=p["base"],
        kinematics=p["kinematics"],
    )
    ctx.progress(0.9, "results")
    ctx.table(res.rows(), "curve", "curve of the indentation")
    peak = int(np.argmax(res.depth))
    res.problem.write(ctx.path("indented"), res.result, step=peak)
    fig = Figure(figsize=(9.4, 3.9), layout="constrained")
    left, right = fig.subplots(1, 2)
    _curve(left, res.depth, res.force, "depth", "force")
    # the law of the half-space of the modulus read at the largest depth
    h = np.linspace(0.0, float(res.depth[peak]), 50)
    read = float(res.modulus[peak])
    law = [
        0.0 if d <= 0 else (
            hertz_sphere(radius, read, approach=d)["force"] if sphere
            else sneddon_cone(p["half_angle"], read, d)["force"]
        )
        for d in h
    ]  # fmt: skip
    left.plot(h, law, "--", lw=1.0, label="Hertz" if sphere else "Sneddon")
    left.legend(fontsize=8)
    _curve(right, res.depth[: peak + 1], res.modulus[: peak + 1], "depth", "reduced modulus read")
    ctx.figure(fig, "indentation", "force and modulus read against the depth")
    values = {
        "elements": mesh.n_elements,
        "nodes": mesh.n_nodes,
        "depth": float(res.depth[peak]),
        "force": float(res.force[peak]),
        "area of the contact": float(res.area[peak]),
        "radius of the contact": float(res.radius[peak]),
        "largest pressure": float(res.pressure[peak]),
        "mean pressure (hardness)": float(res.hardness[peak]),
        "reduced modulus read (a half-space)": read,
    }
    if sphere:
        values["radius of the sphere"] = float(radius)
        if poisson is not None:
            values["reduced modulus read (with the base of the sample)"] = float(
                res.layer_modulus(poisson)[peak]
            )
    if res.unloading_modulus is not None:
        values["stiffness of the unloading"] = res.stiffness
        values["reduced modulus of the unloading"] = res.unloading_modulus
    return {"values": values, "title": f"Indentation of {p['source']['title']}"}
