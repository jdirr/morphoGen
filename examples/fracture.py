"""Fracture: cohesive zones along the mid-plane of a double cantilever beam against the
corrected beam theory, the energy release rate of an edge crack by XFEM against the
handbook, the same crack in tension and shear (mixed modes I and II), which kinks and
grows on meshes rebuilt by MMG at each step, and cracks in inelastic solids in finite
strains: the driving forces of a crack in a hardening plate and a cohesive crack in an
elasto-viscoplastic one; the ductile growth of a crack in a viscoplastic plate on meshes
rebuilt at each advance, its resistance curve; a crack growing through the grains of a
polycrystal of crystal plasticity, remeshed at each advance; a crack through a
morphology, its crack meshes conforming to its phases, turned towards a pore; and a crack
that feels the toughness of the phases and of their interfaces, deflected along a weak
interface and through a strong one (He & Hutchinson).

Usage: python examples/fracture.py [output folder]   (figures with matplotlib)
"""

import math
import sys
from pathlib import Path

import numpy as np

from morphogen.fracture import (
    XFEM,
    Bilinear,
    CohesiveProblem,
    Crack,
    KinkedCrack,
    Toughness,
    cohesive_zone_length,
    crack_mesh,
    dcb_beam_theory,
    dcb_specimen,
    insert_interfaces,
    lefm,
    propagate_ductile,
    propagate_kinked,
    release_rate,
    sent_factor,
    solve_crack,
)
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage
from morphogen.io.tables import write_table
from morphogen.mesh.conforming import has_mmg

out = Path(sys.argv[1] if len(sys.argv) > 1 else "fracture_output")
out.mkdir(parents=True, exist_ok=True)

# 1. A double cantilever beam (arms of 4, precrack of 20) opened at its loading line
young, poisson, energy, strength = 1000.0, 0.3, 0.5, 5.0
print(f"cohesive zone: {cohesive_zone_length(young, energy, strength):.1f} (elements of 1)")
mesh, interfaces, lines = dcb_specimen(60.0, 2.0, 4.0, 20.0, 1.0)
problem = CohesiveProblem(
    mesh,
    interfaces,
    {0: Material(young=young, poisson=poisson)},
    Bilinear(1e5, strength, strength, energy, energy),
)
result = problem.run(
    [
        (lines["top"], 2, 1.0),
        (lines["bottom"], 2, -1.0),
        (lines["top"], [0, 1], 0.0),
        (lines["bottom"], [0, 1], 0.0),
    ],
    load=3.0,
    steps=60,
    viscosity=1e-3,
)
opening = 2 * result.loads
load = result.reactions[:, 0, 2]
beam, crack = dcb_beam_theory(young, poisson, 2.0, 4.0, 20.0, energy, opening)
print(f"peak load {load.max():.3f} (beam theory {beam.max():.3f}), {result.time_s:.0f} s")
problem.write(out / "dcb", result)
try:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5, 3.6), layout="constrained")
    ax = fig.subplots()
    fine = np.linspace(0.0, opening.max(), 300)
    ax.plot(
        fine,
        dcb_beam_theory(young, poisson, 2.0, 4.0, 20.0, energy, fine)[0],
        color="0.3",
        label="corrected beam theory",
    )
    ax.plot(opening, load, "o", ms=3, label="cohesive zone elements")
    ax.set_xlabel("opening")
    ax.set_ylabel("load")
    ax.legend()
    fig.savefig(out / "dcb.png", dpi=150)
except ImportError:
    pass

# 2. An edge crack in a plate in tension: G by XFEM against Tada, Paris and Irwin
plate = voxel_tet_mesh(VoxelImage(np.zeros((16, 32, 2), np.uint8), periodic=False))
material = Material(young=1000.0, poisson=0.0)


def forces(m):
    top = lefm.face_forces(m, lambda c: c[:, 1] > 32 - 1e-9, [0, 1.0, 0])
    return top + lefm.face_forces(m, lambda c: c[:, 1] < 1e-9, [0, -1.0, 0])


def supports(m):
    x = m.nodes
    corner = (np.abs(x[:, 0] - 16) < 1e-9) & (x[:, 2] < 1e-9)
    a = np.flatnonzero(corner & (x[:, 1] < 1e-9))[0]
    b = np.flatnonzero(corner & (x[:, 1] > 32 - 1e-9))[0]
    return np.concatenate([[3 * a, 3 * a + 1, 3 * b], 3 * np.flatnonzero(x[:, 2] < 1e-9) + 2])


edge = Crack("straight", (0.0, 16.5, 0.0), (0, 1, 0), 4.5, direction=(1, 0, 0))
model = XFEM(plate, material, edge)
g = release_rate(lefm.xfem_work(plate, material, forces, supports), edge, 0.25)
exact = (math.sqrt(math.pi * 4.5) * sent_factor(4.5 / 16)) ** 2 / 1000.0
print(
    f"edge crack: {model.jump.sum()} jump and {model.tip.sum()} branch nodes, "
    f"G = {g:.5f} (handbook {exact:.5f}, {100 * (g / exact - 1):+.1f} %)"
)

# 3. The same crack in tension and shear (tau / sigma = 0.7): the G-theta method and the
# interaction integrals give G, K_I and K_II; the crack kinks in the direction of the
# largest hoop stress (Erdogan & Sih) and grows towards the normal of the largest
# principal stress, the body meshed again by MMG for each crack (the front on edges)
if has_mmg():

    def shear(m, tau=0.7):
        f = lefm.face_forces(m, lambda c: c[:, 1] > 32 - 1e-9, [tau, 0, 0])
        f += lefm.face_forces(m, lambda c: c[:, 1] < 1e-9, [-tau, 0, 0])
        f += lefm.face_forces(m, lambda c: c[:, 0] > 16 - 1e-9, [0, tau, 0])
        return f + lefm.face_forces(m, lambda c: c[:, 0] < 1e-9, [0, -tau, 0])

    def nearest(m, point):
        return int(np.argmin(np.linalg.norm(m.nodes - np.asarray(point), axis=1)))

    def rigid(m):  # the forces are self-equilibrated: the rigid motions only
        a, b = nearest(m, (16, 16, 0)), nearest(m, (16, 32, 0))
        return [(a, [0, 1], 0.0), (b, 0, 0.0), (np.flatnonzero(m.nodes[:, 2] < 1e-9), 2, 0.0)]

    growth = propagate_kinked(
        lambda c: crack_mesh((0, 0, 0), (16, 32, 1), c, size=0.5),
        KinkedCrack(((0.0, 16.0, 0.0), (4.5, 16.0, 0.0))),
        material,
        rigid,
        forces=lambda m: forces(m) + shear(m),
        increment=1.5,
        steps=4,
        paris=(1e-3, 3.0),
        keep=True,
    )
    write_table(out / "kinked_growth.csv", growth.rows())
    for row in growth.rows():
        print(
            f"kinked crack, step {row['step']}: K_I {row['K_I']:.3f}, K_II {row['K_II']:+.3f}, "
            f"kink {row['angle_deg']:+.1f} deg"
        )
    try:
        from matplotlib.figure import Figure

        fig = Figure(figsize=(5, 4), layout="constrained")
        ax = fig.subplots()
        path = np.asarray(growth.cracks[-1].points)
        ax.plot(path[:, 0], path[:, 1], "o-", color="C3", label="crack path")
        x = np.linspace(4.5, path[-1, 0], 10)
        ax.plot(x, 16 + np.tan(math.radians(-27.2)) * (x - 4.5), ":", color="0.4",
                label="normal to the largest principal stress")  # fmt: skip
        ax.set_aspect("equal")
        ax.set_xlim(0, 16)
        ax.set_ylim(8, 20)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="lower left")
        fig.savefig(out / "kinked_path.png", dpi=150)
    except ImportError:
        pass

# 4. Inelastic solids in finite strains. The edge crack in a plate of von Mises hardening
# (yield at 1/1000 of the modulus) pulled by its grips to three times that strain: the J of
# the work density (the deformation theory), of the free energy alone, and at the tip (the
# configurational forces of the gradients of the plastic strain and hardening added)
hardening = Material(young=1000.0, poisson=0.3, plasticity={"yield": 1.0, "hardening": 50.0})


def grips(m, u=0.05):
    x = m.nodes
    top, bottom = np.flatnonzero(x[:, 1] > 32 - 1e-9), np.flatnonzero(x[:, 1] < 1e-9)
    return [(top, [0, 1, 2], [0.0, u, 0.0]), (bottom, [0, 1, 2], [0.0, -u, 0.0])]


edge = Crack("straight", (0.0, 16.0, 0.0), (0, 1, 0), 4.5, direction=(1, 0, 0))
plastic = solve_crack(plate, edge, hardening, grips, kinematics="finite", steps=8)
for r_out in (3.0, 5.0):
    values = {
        e: plastic.release_rate(r_in=1.0, r_out=r_out, energy=e) for e in ("work", "free", "tip")
    }
    print(
        f"plastic plate, domain {r_out:g}: "
        + ", ".join(f"J_{k} {v:.4f}" for k, v in values.items())
    )

# A cohesive zone ahead of an edge crack, in an elastic and an elasto-viscoplastic plate
small = voxel_tet_mesh(VoxelImage(np.zeros((8, 8, 1), np.uint8), periodic=False))
opened, faces = insert_interfaces(small, lambda c: np.abs(c[:, 1] - 4.0) < 1e-9)
ahead = faces.subset(faces.centroids(opened)[:, 0] > 2.0)
law = Bilinear(1e5, 10.0, 10.0, 0.2, 0.2)
x = opened.nodes
top, bottom = np.flatnonzero(x[:, 1] > 8 - 1e-9), np.flatnonzero(x[:, 1] < 1e-9)
pull = [(top, [0, 1, 2], [0.0, 1.0, 0.0]), (bottom, [0, 1, 2], [0.0, -1.0, 0.0])]
viscoplastic = Material(
    young=1000.0,
    poisson=0.3,
    plasticity={"yield": 4.0, "hardening": 50.0, "norton": {"K": 5.0, "n": 5.0}},
)
for name, material in (
    ("elastic", Material(young=1000.0, poisson=0.3)),
    ("viscoplastic", viscoplastic),
):
    problem = CohesiveProblem(opened, ahead, {0: material}, law, kinematics="finite")
    result = problem.run(pull, 0.12, steps=60, viscosity=1e-4, duration=10.0)
    stored = 0.0 if problem.bulk is None else float(problem.bulk.free_energy() @ problem.bulk.wts)
    dissipated = result.work["bulk"][-1] - stored + result.work["interfaces"][-1]
    print(
        f"cohesive crack in the {name} plate: peak load {result.reactions[:, 0, 1].max():.2f}, "
        f"dissipation per unit of cracked area {dissipated / result.cracked[-1]:.3f} (G_c 0.2)"
    )
    write_table(out / f"cohesive_{name}.csv", result.rows())

# 5. Ductile growth on meshes rebuilt at each advance: the edge crack in a plate of a
# viscoplastic metal (Norton; the load factor is the time) pulled by its grips, growing
# straight by 1.5 once its driving force at the tip reaches 0.03, the state of the points
# carried from mesh to mesh by the nearest element: the far J at each advance draws the
# resistance curve (the plastic wake shields the tip)
if has_mmg():
    metal = Material(
        young=1000.0,
        poisson=0.3,
        plasticity={"yield": 1.0, "hardening": 50.0, "norton": {"K": 1.0, "n": 5.0}},
    )
    ductile = propagate_ductile(
        lambda c: crack_mesh((0, 0, 0), (16, 32, 1), c, size=1.0),
        KinkedCrack(((0.0, 16.0, 0.0), (4.5, 16.0, 0.0))),
        metal,
        lambda m: grips(m, 0.03),
        load_step=0.25,
        increment=1.5,
        steps=3,
        toughness=0.03,
        kink=False,
        transfer="nearest",
    )
    write_table(out / "resistance_curve.csv", ductile.rows())
    for row in ductile.rows():
        print(
            f"ductile growth, advance {row['delta_a']:.1f}: load {row['load']:.2f}, "
            f"J_tip {row['J_tip']:.4f}, far J {row['J_far']:.4f}"
        )
    try:
        from matplotlib.figure import Figure

        fig = Figure(figsize=(5, 3.6), layout="constrained")
        ax = fig.subplots()
        advance, far = ductile.resistance()
        ax.plot(advance, far, "o-", color="C0", label="far J (deformation theory)")
        ax.plot(advance, ductile.tip, "s--", color="C3", label="J at the tip")
        ax.axhline(0.03, color="0.5", lw=0.8, label="toughness")
        ax.set_xlabel("crack advance")
        ax.set_ylabel("driving force")
        ax.legend()
        fig.savefig(out / "resistance_curve.png", dpi=150)
    except ImportError:
        pass

# 6. A polycrystal: twelve columnar grains of a cubic crystal with octahedral slip (a
# Voronoi tessellation of germs in a plane, random orientations), a crack of 2.5 from the
# middle of the left edge growing by 1 once its driving force at the tip reaches 0.6, the
# crack meshes carrying the grains and the state moved grain by grain from mesh to mesh
if has_mmg():
    import dataclasses

    from morphogen.domain import Domain
    from morphogen.homogenization.crystal import CrystalPlasticity
    from morphogen.models.tessellation import Tessellation

    rng = np.random.default_rng(3)
    germs = np.column_stack([rng.uniform(0, 8, 12), rng.uniform(0, 8, 12), np.full(12, 0.25)])
    euler = rng.uniform(0, 1, (12, 3)) * [360.0, 180.0, 360.0]
    grains = Tessellation(Domain(size=(8.0, 8.0, 0.5), periodic=False), germs, orientations=euler)
    crystal = dataclasses.replace(
        Material.cubic(168e3, 121e3, 75e3),
        crystal_plasticity=CrystalPlasticity.from_config(
            {"tau0": 60.0, "Q": 40.0, "b": 5.0, "K": 5.0, "n": 20.0}
        ),
    )

    def pulled(m):  # the faces y = 0 and 8 pulled apart, plane strain
        x = m.nodes
        top, bottom = x[:, 1] > 8.0 - 1e-9, x[:, 1] < 1e-9
        faces = (x[:, 2] < 1e-9) | (x[:, 2] > 0.5 - 1e-9)
        corner = int(np.argmin(np.linalg.norm(x - [8.0, 4.0, 0.0], axis=1)))
        return [
            (np.flatnonzero(top & ~faces), 1, 1.0),
            (np.flatnonzero(bottom & ~faces), 1, -1.0),
            (np.flatnonzero(top & faces), [1, 2], [1.0, 0.0]),
            (np.flatnonzero(bottom & faces), [1, 2], [-1.0, 0.0]),
            (np.flatnonzero(faces & ~top & ~bottom), 2, 0.0),
            (corner, 0, 0.0),
        ]

    through = propagate_ductile(
        lambda c: crack_mesh((0, 0, 0), (8.0, 8.0, 0.5), c, size=0.4, grains=grains),
        KinkedCrack(((0.0, 4.0, 0.0), (2.5, 4.0, 0.0))),
        crystal,
        pulled,
        load_step=1e-3,
        increment=1.0,
        steps=3,
        toughness=0.6,
        max_load=0.03,
        transfer="nearest",
        duration=1.0,
    )
    write_table(out / "polycrystal_growth.csv", through.rows())
    for row, c in zip(through.rows(), through.cracks, strict=True):
        grain = int(grains.cell_of(np.atleast_2d(c.tip))[0])
        print(
            f"polycrystal, advance {row['advance']}: load {row['load']:.3f}, J_tip "
            f"{row['J_tip']:.3f}, far J {row['J_far']:.3f}, kink {row['angle_deg']:+.1f} deg, "
            f"tip in grain {grain}"
        )

# 7. A crack through a morphology: a plate of 16 x 16 x 1 in tension, a pore beside the path
# (a cylinder of radius 1.5, its centre 1 ahead of the tip and 2.5 above the crack); the mesh
# of its phases (cut by its level set, once) cut by each crack and remeshed by MMG, the pore
# left out: the crack turns towards the pore (the interaction integrals on the material at
# the front), which shields nothing but raises the G at the tip above that of the domain
if has_mmg():
    from morphogen.domain import Domain
    from morphogen.fracture import phase_mesh
    from morphogen.models.union import PrimitiveUnion
    from morphogen.primitives import Segments

    pore = Segments(np.array([[5.5, 10.5, -1.0]]), np.array([[5.5, 10.5, 2.0]]), np.array([1.5]))
    holed = PrimitiveUnion(Domain(size=(16.0, 16.0, 1.0), periodic=False), segments=pore)
    phases = phase_mesh(holed, (0, 0, 0), (16.0, 16.0, 1.0), 0.5)

    def tension(m):
        top = lefm.face_forces(m, lambda c: c[:, 1] > 16 - 1e-9, [0, 1.0, 0])
        return top + lefm.face_forces(m, lambda c: c[:, 1] < 1e-9, [0, -1.0, 0])

    def held(m):
        corner = int(np.argmin(np.linalg.norm(m.nodes - [16.0, 8.0, 0.0], axis=1)))
        top = int(np.argmin(np.linalg.norm(m.nodes - [16.0, 16.0, 0.0], axis=1)))
        return [
            (corner, [0, 1], 0.0),
            (top, 0, 0.0),
            (np.flatnonzero(m.nodes[:, 2] < 1e-9), 2, 0.0),
        ]

    attracted = propagate_kinked(
        lambda c: crack_mesh(
            (0, 0, 0), (16.0, 16.0, 1.0), c, size=0.5, morphology=phases, voids=[1]
        ),
        KinkedCrack(((0.0, 8.0, 0.0), (4.5, 8.0, 0.0))),
        {0: Material(young=1000.0, poisson=0.3), 1: Material(void=True)},
        held,
        forces=tension,
        increment=0.75,
        steps=2,
        keep=True,
    )
    write_table(out / "morphology_growth.csv", attracted.rows())
    for row, solution in zip(attracted.rows(), attracted.solutions, strict=True):
        print(
            f"through a pore, crack {row['step']}: tip ({row['x_tip']:.2f}, {row['y_tip']:.2f}), "
            f"K_II / K_I {row['K_II'] / row['K_I']:+.3f}, kink {row['angle_deg']:+.1f} deg, G at "
            f"the tip {solution.release_rate(energy='tip'):.4f}, of the domain "
            f"{solution.release_rate():.4f}"
        )

# 8. The toughness of the phases and of their interfaces: the plate in two halves of the same
# material, their interface normal to the crack 0.75 ahead of its tip. Each advance is the
# candidate of the largest G(theta) / toughness: the kink of the bulk, or along the interface
# (the first-order kinked factors of Cotterell & Rice give it G / 4 at 90 degrees). Below a
# quarter of the toughness of the phases (He & Hutchinson), the crack goes on to the interface
# and turns along it, at a lower critical load; above, it goes through
if has_mmg():

    def halves(points):
        return (np.asarray(points)[:, 0] > 5.25).astype(np.int64)

    alike = Material(young=1000.0, poisson=0.3)
    for ratio in (0.15, 0.4):
        weak = propagate_kinked(
            lambda c: crack_mesh((0, 0, 0), (16.0, 16.0, 1.0), c, size=0.5, morphology=halves),
            KinkedCrack(((0.0, 8.0, 0.0), (4.5, 8.0, 0.0))),
            {0: alike, 1: alike},
            held,
            forces=tension,
            increment=1.0,
            steps=2,
            toughness=Toughness(1.0, ratio, morphology=halves),
        )
        write_table(out / f"interface_{ratio}.csv", weak.rows())
        for row in weak.rows():
            print(
                f"interface of {ratio} times the toughness, crack {row['step']}: tip "
                f"({row['x_tip']:.2f}, {row['y_tip']:.2f}), {row['path']} ({row['met']}), "
                f"G / toughness {row['ratio']:.4f}, critical load {row['load_critical']:.3f}"
            )
