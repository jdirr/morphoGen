"""The growth of a crack front in three dimensions, a tool of the tab Fracture of the
interface: an elliptical crack inside a block, inclined to the tension, each point of
whose front advances by its own amount and kinks by its own angle
(:mod:`morphogen.fracture.surface`)."""

from __future__ import annotations

import math

import numpy as np

from morphogen.gui.tools import Field, tool


@tool(
    "front_growth",
    "fracture",
    "Growth of a crack front in 3D",
    "An elliptical crack inside a block in tension, its plane inclined to the load: the "
    "stress intensity factors K_I, K_II and K_III along its curved front (interaction "
    "integrals on a mesh conforming to the crack, refined around its front), against "
    "those of the infinite body (Irwin 1962; Kassir & Sih 1966), and its growth: each "
    "point of the front advances by its own amount, as its energy release rate to the "
    "power of Paris' law, in the direction of the largest hoop stress of its own factors "
    "(Erdogan & Sih). An ellipse grows towards a circle; an inclined crack turns its "
    "surface towards the plane normal to the tension, by a different angle at each point "
    "of its front. The mesh is rebuilt by MMG for each crack.",
    [
        Field("major", "float", 1.0, label="semi-axis a", unit="length", doc="along y"),
        Field(
            "minor",
            "float",
            label="semi-axis b",
            unit="length",
            doc="across y, in the plane of the crack (blank: a, a penny crack)",
        ),
        Field(
            "inclination",
            "float",
            30.0,
            unit="degree",
            doc="of the normal of the crack from the axis z of the tension, about y (0: "
            "mode I alone)",
        ),
        Field(
            "block",
            "float",
            label="edge of the block",
            unit="length",
            doc="a cube around the crack (blank: five times its largest semi-axis)",
        ),
        Field("young", "float", 1000.0, label="Young's modulus"),
        Field("poisson", "float", 0.3, label="Poisson's ratio"),
        Field("stress", "float", label="tension", doc="along z (blank: 0.001 E)"),
        Field(
            "element_size",
            "float",
            unit="length",
            doc="of the block (blank: an eighth of its edge)",
        ),
        Field(
            "front_size",
            "float",
            unit="length",
            doc="of the elements around the front (blank: an eighth of the smallest semi-axis)",
        ),
        Field("stations", "int", 12, doc="along the front, of the factors"),
        Field("steps", "int", 2, doc="advances of the front (0: the factors alone)"),
        Field(
            "increment",
            "float",
            unit="length",
            doc="the advance of the fastest point at each step (blank: 0.3 of the smallest "
            "semi-axis; four elements of the front at least)",
        ),
        Field(
            "exponent",
            "float",
            2.0,
            label="exponent of Paris' law",
            doc="the advance of a point: the increment times (G / G_max) to half of it",
        ),
        Field(
            "paris",
            "float",
            label="coefficient of Paris' law",
            doc="C of da/dN = C ΔK^m (blank: the cycles are not counted)",
        ),
    ],
    order=3,
    menu="Cracks",
)
def front_growth(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure
    from scipy.special import ellipe

    from morphogen.fracture import lefm
    from morphogen.fracture.surface import SurfaceCrack, propagate_front
    from morphogen.homogenization.materials import Material
    from morphogen.mesh.conforming import has_mmg

    if not has_mmg():
        raise ValueError("the meshes of the cracks are made by MMG (pip install pymmg)")
    a = float(p["major"])
    b = a if p["minor"] is None else float(p["minor"])
    if not (a > 0 and 0 < b <= a):
        raise ValueError("semi-axes: positive, b at most a")
    alpha = math.radians(float(p["inclination"]))
    if not 0.0 <= alpha < math.radians(80.0):
        raise ValueError("inclination: from 0 to 80 degrees")
    edge = 5.0 * a if p["block"] is None else float(p["block"])
    if edge < 2.5 * a:
        raise ValueError("edge of the block: at least 2.5 times the semi-axis a")
    young, nu = float(p["young"]), float(p["poisson"])
    if not (young > 0 and -1.0 < nu < 0.5):
        raise ValueError("Young's modulus: positive; Poisson's ratio: between -1 and 0.5")
    stress = 1e-3 * young if p["stress"] is None else float(p["stress"])
    size = edge / 8.0 if p["element_size"] is None else float(p["element_size"])
    front = b / 8.0 if p["front_size"] is None else float(p["front_size"])
    if not (0 < front <= size <= edge / 4.0):
        raise ValueError("element sizes: that of the front at most that of the block, itself "
                         "at most a quarter of its edge")  # fmt: skip
    stations, steps = int(p["stations"]), int(p["steps"])
    if not (4 <= stations <= 48 and 0 <= steps <= 12):
        raise ValueError("stations: from 4 to 48; steps: from 0 to 12")
    increment = 0.3 * b if p["increment"] is None else float(p["increment"])
    if increment < 3.0 * front:
        raise ValueError(
            f"increment: at least three elements of the front ({3 * front:.3g}), in which the "
            "factors of the kinked front are taken"
        )
    half = 0.5 * edge
    normal = (math.sin(alpha), 0.0, math.cos(alpha))
    crack = SurfaceCrack.ellipse((0.0, 0.0, 0.0), normal, a, b, major=(0.0, 1.0, 0.0), stations=64)
    material = Material(young=young, poisson=nu)

    def nearest(m, point):
        return int(np.argmin(np.linalg.norm(m.nodes - np.asarray(point), axis=1)))

    def forces(m):
        f = lefm.face_forces(m, lambda c: c[:, 2] > half - 1e-9 * edge, [0.0, 0.0, stress])
        return f + lefm.face_forces(m, lambda c: c[:, 2] < 1e-9 * edge - half, [0.0, 0.0, -stress])

    def supports(m):
        corners = ((-half, -half, -half), (half, -half, -half), (-half, half, -half))
        i, j, k = (nearest(m, c) for c in corners)
        return [(i, [0, 1, 2], 0.0), (j, [1, 2], 0.0), (k, 2, 0.0)]

    done = {"count": 0}

    def mesh_of(c):
        ctx.progress(
            0.05 + 0.85 * done["count"] / (steps + 1),
            f"crack {done['count']} of {steps}: mesh and solution",
        )
        done["count"] += 1
        return lefm.crack_mesh((-half,) * 3, (half,) * 3, c, size=size, front_size=front,
                               front_radius=4.0 * front)  # fmt: skip

    domain = {"r_in": 1.2 * front, "r_out": min(3.5 * front, 0.9 * increment)}
    growth = propagate_front(
        mesh_of,
        crack,
        material,
        supports,
        forces=forces,
        increment=increment,
        steps=steps,
        stations=stations,
        exponent=float(p["exponent"]),
        paris=None if p["paris"] is None else (float(p["paris"]), float(p["exponent"])),
        domain=domain,
        keep=True,
    )
    ctx.progress(0.92, "figures")
    ctx.table(growth.rows(), "growth", "the crack at each step")
    degrees = np.degrees(growth.s)
    along = []
    for k, (factors, angles) in enumerate(zip(growth.factors, growth.angles, strict=True)):
        for s, f, turn in zip(degrees, factors, angles, strict=True):
            along.append({"step": k, "s_deg": float(s), "K_I": float(f[0]), "K_II": float(f[1]),
                          "K_III": float(f[2]), "kink_deg": float(np.degrees(turn))})  # fmt: skip
    ctx.table(along, "factors", "the factors and the angle of kinking along the front")
    fig = Figure(figsize=(11.0, 3.6), layout="constrained")
    for ax, mode, name in zip(fig.subplots(1, 3), range(3), ("K_I", "K_II", "K_III"), strict=True):
        for k, factors in enumerate(growth.factors):
            ax.plot(degrees, factors[:, mode], "o-", ms=3, lw=1.2, label=f"step {k}")
        ax.set_xlabel("position along the front (degrees)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    fig.axes[0].legend(fontsize=8, frameon=False)
    ctx.figure(fig, "factors", "stress intensity factors along the front")
    # the fronts: in the plane of the first crack, and from its side
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.cross(normal, e1)
    fig = Figure(figsize=(9.2, 4.2), layout="constrained")
    top, side = fig.subplots(1, 2)
    for k, grown in enumerate(growth.cracks):
        loop = np.vstack([grown.points[-1], grown.points[-1][:1]])
        top.plot(loop @ e2, loop @ e1, lw=1.4, label=f"step {k}")
        side.plot(loop[:, 0], loop[:, 2], lw=1.4)
    top.set_title("fronts in the plane of the first crack", fontsize=9)
    side.set_title("fronts from the side (x, z: the tension along z)", fontsize=9)
    for ax in (top, side):
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    top.legend(fontsize=8, frameon=False)
    ctx.figure(fig, "fronts", "the successive fronts of the crack")
    first = growth.factors[0]
    values = {
        "nodes of the first mesh": int(growth.solutions[0].mesh.n_nodes),
        "element size at the front": growth.solutions[0].element_size(),
        "K_I along the first front (mean)": float(first[:, 0].mean()),
        "largest |K_II| along the first front": float(np.abs(first[:, 1]).max()),
        "largest |K_III| along the first front": float(np.abs(first[:, 2]).max()),
        "largest angle of kinking of the first front (degrees)": float(
            np.degrees(np.abs(growth.angles[0]).max())
        ),
    }
    # the infinite body
    unit = stress * math.sqrt(math.pi * b) / ellipe(1.0 - (b / a) ** 2)
    if alpha == 0.0:
        values["K_I of the infinite body, ends of the minor axis (Irwin)"] = unit
        values["K_I of the infinite body, ends of the major axis (Irwin)"] = unit * math.sqrt(b / a)
    elif b == a:
        tau = math.sin(alpha) * math.cos(alpha)
        base = 2.0 * stress * math.sqrt(a / math.pi)
        values["K_I of the infinite body (Kassir & Sih)"] = base * math.cos(alpha) ** 2
        values["largest K_II of the infinite body (Kassir & Sih)"] = 2.0 * base * tau / (2 - nu)
        values["largest K_III of the infinite body (Kassir & Sih)"] = (
            2.0 * base * tau * (1 - nu) / (2 - nu)
        )
    if steps:
        last = growth.cracks[-1]
        values["area of the last crack over that of the first"] = last.area() / crack.area()
        values["largest |K_II| along the last front"] = float(
            np.abs(growth.factors[-1][:, 1]).max()
        )
        if p["paris"] is not None:
            values["cycles"] = float(growth.cycles[-1])
    shape = "penny" if b == a else "elliptical"
    title = f"Growth of an {shape} crack" if b != a else "Growth of a penny crack"
    if alpha:
        title += f" inclined by {float(p['inclination']):g}°"
    return {"values": values, "title": title}
