"""Micromorphic continua: the hole size effect of nickel foams with the microfoam model
(Dillard, Forest & Ienny 2006, its section 4 and figures 6 to 8), a crack propagated as a
band of damage by the micromorphic damage model on two meshes, the large compression of
the nickel foam in finite strains, and a ductile crack by micromorphic ductile damage on
two meshes.

A quarter of a nickel foam plate (half-width 6 R, a slab of half-thickness 0.4 R with
its mid-plane, symmetry planes and free faces: nearly plane stress) with a hole of
radius R is pulled along TD to an overall strain of 0.02. The classical continuum
concentrates the strain at the equator of the hole whatever its size; the microfoam
continuum (l_c = 0.1 mm) less and less as the hole shrinks to the size of the cells (R =
4, 1 and 0.5 mm): one mesh in units of R, and l_c / R.

A block of the foam compressed to half its height in finite strains densifies along TD,
whose plastic contraction ratios are small, and hardly along RD. The notched plate of a
hardening von Mises solid cracks by ductile damage driven by the micromorphic cumulated
plastic strain, in finite strains: two bands across the ligament.

Usage: python examples/microfoam.py [output folder] [--quick]   (figures with matplotlib)
"""

import sys
import time
from pathlib import Path

import numpy as np

from morphogen.domain import Domain
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.image import VoxelImage
from morphogen.io.tables import write_table
from morphogen.mesh.conforming import conforming_mesh, has_mmg
from morphogen.micromorphic import (
    CHI,
    DamageProblem,
    DuctileProblem,
    Microfoam,
    MicrofoamProblem,
    MicromorphicDamage,
    MicromorphicDuctile,
)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
quick = "--quick" in sys.argv
out = Path(args[0] if args else "microfoam_output")
out.mkdir(parents=True, exist_ok=True)


class QuarterPlate:
    """[0, half]² x [0, thickness] without the hole of radius 1 at the origin."""

    def __init__(self, half, thickness):
        self.domain = Domain(size=(half, half, thickness), origin=(0.0, 0.0, 0.0), periodic=False)

    def level_set(self, shape=None, *, voxel_size=None, grid=None):
        c = grid.center_points()
        values = 1.0 - np.hypot(c[:, 0], c[:, 1])  # negative in the plate
        return VoxelImage(
            values.reshape(grid.shape), spacing=tuple(grid.spacing), origin=tuple(grid.origin)
        )


def plate_supports(problem, half, strain):
    """Symmetry planes x1 = 0, x2 = 0 and the mid-plane x3 = 0 (the components of the
    microdeformation odd under each reflection vanish there), the face x2 = half pulled by
    rollers to the overall strain, the others free."""
    x = problem.mesh.nodes
    tol = 1e-9 * half
    sets = {
        "bottom": (np.flatnonzero(np.abs(x[:, 1]) < tol), 1, ("chi12", "chi21", "chi23", "chi32")),
        "left": (np.flatnonzero(np.abs(x[:, 0]) < tol), 0, ("chi12", "chi21", "chi13", "chi31")),
        "mid": (np.flatnonzero(np.abs(x[:, 2]) < tol), 2, ("chi13", "chi31", "chi23", "chi32")),
    }
    entries = [(nodes, comp, 0.0) for nodes, comp, _ in sets.values()]
    entries.append((np.flatnonzero(np.abs(x[:, 1] - half) < tol), 1, strain * half))
    if problem.micromorphic:
        held = {}
        for nodes, _, odd in sets.values():
            for n in nodes[problem.space.corner[nodes]]:
                held.setdefault(int(n), set()).update(CHI[c] for c in odd)
        for n, comps in held.items():
            entries.append((n, sorted(comps), 0.0))
    return entries


rows = []
if has_mmg():
    half, thickness, strain = 6.0, 0.4, 0.02
    size = 1.2 if quick else 0.8
    mesh = conforming_mesh(QuarterPlate(half, thickness), size=size, min_size=0.06, hausdorff=0.01,
                           gradation=1.3, phases="solid", periodic=False)  # fmt: skip
    print(f"quarter plate: {mesh.n_elements} tetrahedra")
    profiles = {}
    cases = [
        ("classical", None),
        ("R = 4 mm", 0.1 / 4),
        ("R = 1 mm", 0.1 / 1),
        ("R = 0.5 mm", 0.1 / 0.5),
    ]
    for name, length in cases:
        t0 = time.time()
        foam = Microfoam.nickel_foam(length=length)
        problem = MicrofoamProblem(mesh, {int(p): foam for p in np.unique(mesh.phase)})
        result = problem.run(
            plate_supports(problem, half, strain), steps=6 if quick else 10, refactor=0.25
        )
        fields = problem.fields(result.values, result.state)
        centre = problem.mesh.nodes[problem.mesh.elements[:, :4]].mean(axis=1)
        near = (centre[:, 1] < 0.2) & (centre[:, 2] < 0.25)
        bins = np.linspace(1.0, half, 21)
        which = np.digitize(centre[near, 0], bins) - 1
        inside = (which >= 0) & (which < 20)
        eps22 = fields["strain"][near, 1, 1]
        sig22 = fields["stress"][near, 1, 1]
        counts = np.bincount(which[inside], minlength=20)
        sums = np.bincount(which[inside], eps22[inside], minlength=20)
        # bins without an element are gaps of the curve
        profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan) / strain
        top = result.reactions[-1][3][0] / (half * thickness)  # the remote stress
        stress_ratio = sig22[which == 0].mean() / top
        profiles[name] = profile
        rows.append({"case": name, "l_c_over_R": length if length is not None else 0.0,
                     "strain_concentration": float(profile[0]),
                     "stress_concentration": float(stress_ratio), "remote_stress_MPa": float(top),
                     "time_s": time.time() - t0})  # fmt: skip
        print(
            f"{name}: strain concentration at the equator {profile[0]:.2f}, stress concentration "
            f"{stress_ratio:.2f} (remote stress {top:.3f} MPa), {time.time() - t0:.0f} s"
        )
    write_table(out / "hole_size_effect.csv", rows)
    try:
        from matplotlib.figure import Figure

        fig = Figure(figsize=(5, 3.6), layout="constrained")
        ax = fig.subplots()
        x = 0.5 * (bins[1:] + bins[:-1])
        for name, profile in profiles.items():
            ax.plot(x, profile, "-" if name == "classical" else "o-", ms=3, label=name)
        ax.set_xlabel("x1 / R along the ligament")
        ax.set_ylabel("strain 22 / overall strain")
        ax.legend()
        fig.savefig(out / "hole_size_effect.png", dpi=150)
    except ImportError:
        pass


def notched_plate(h):
    """The plate [0, 1]² of one layer of voxels of side h, a notch of depth 0.3 and width
    0.1 cut at mid-height."""
    n = round(1.0 / h)
    centre = (np.arange(n) + 0.5) * h
    data = np.zeros((n, n, 1), np.uint8)
    data[np.ix_(centre < 0.3, np.abs(centre - 0.5) < 0.05, [0])] = 1  # the notch, removed
    return voxel_tet_mesh(VoxelImage(data, spacing=(h, h, h), periodic=False), phases=[0])


def pulled(problem, pull):
    """The bottom held vertically (a node pinned), the top pulled by ``pull``."""
    x = problem.mesh.nodes
    bottom = np.flatnonzero(np.abs(x[:, 1]) < 1e-9)
    top = np.flatnonzero(np.abs(x[:, 1] - 1.0) < 1e-9)
    pin = bottom[np.argmin(np.linalg.norm(x[bottom], axis=1))]
    return [(bottom, 1, 0.0), (pin, [0, 2], 0.0), (top, 1, pull)]


def crack(problem, result, h, name, pull, t0):
    """The force-displacement curve, its row of the table and the reach of the crack."""
    force = np.array([r[2][0] for r in result.reactions]) / h  # per unit thickness
    damage = problem.damage(result.state)
    cracked = problem.mesh.nodes[problem.mesh.elements[:, :4]].mean(axis=1)[damage > 0.9]
    reach = cracked[:, 0].max() if len(cracked) else 0.0
    rows.append({"case": f"{name} h={h}", "peak_force": float(force.max()),
                 "final_force": float(force[-1]), "crack_tip_x": float(reach),
                 "time_s": time.time() - t0})  # fmt: skip
    print(
        f"{name}, h = {h}: peak force {force.max():.4f}, final {force[-1]:.4f}, "
        f"crack (d > 0.9) to x = {reach:.2f} (notch 0.3), {time.time() - t0:.0f} s"
    )
    return pull * result.loads, force


def figure(curves, xlabel, ylabel, name):
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return
    fig = Figure(figsize=(5, 3.6), layout="constrained")
    ax = fig.subplots()
    for label, (x, y) in curves.items():
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.savefig(out / name, dpi=150)


# A notched plate cracked by the micromorphic damage model: the band of damage runs from
# the notch across the ligament, its width set by the length; the force-displacement
# curves of two meshes agree
curves = {}
for h in (0.05,) if quick else (0.05, 0.04):
    t0 = time.time()
    material = MicromorphicDamage(young=1000.0, poisson=0.2, threshold=1e-3, length=0.08,
                                  beta=100.0, variational=False)  # fmt: skip
    problem = DamageProblem(notched_plate(h), material)
    result = problem.run(pulled(problem, 0.004), steps=40)
    curves[f"elements of {h}"] = crack(problem, result, h, "notched plate", 0.004, t0)
figure(curves, "displacement of the top", "force per unit thickness", "damage_band.png")

# A block of nickel foam (1 mm, symmetry planes, free faces) compressed to half its height
# in finite strains, along TD and along RD: the compressible plasticity of the article
# densifies the foam along TD, whose plastic contraction ratios are 0.17 and 0.22 (its
# volume falls with its height), and hardly along RD (0.49 and 0.47); the microdeformation
# follows the deformation in this homogeneous compression (χ = F); small strains for
# comparison
block = voxel_tet_mesh(VoxelImage(np.zeros((2, 2, 2), np.uint8), spacing=(0.5, 0.5, 0.5),
                                  periodic=False))  # fmt: skip
stresses, volumes = {}, {}
for direction, axis in (("TD", 1), ("RD", 0)):
    for kinematics in ("finite", "small"):
        t0 = time.time()
        problem = MicrofoamProblem(block, Microfoam.nickel_foam(length=0.1), kinematics=kinematics)
        x = problem.mesh.nodes
        entries = [(np.flatnonzero(np.abs(x[:, k]) < 1e-9), k, 0.0) for k in range(3)]
        entries.append((np.flatnonzero(np.abs(x[:, axis] - 1.0) < 1e-9), axis, -0.5))
        result = problem.run(entries, steps=10 if quick else 25, fields=True, predictor=True)
        strain = 0.5 * result.loads  # nominal, in compression
        stress = -np.array([r[3][0] for r in result.reactions])  # nominal (MPa), on 1 mm²
        corner = np.flatnonzero(np.all(np.abs(x - 1.0) < 1e-9, axis=1))[0]
        stretch = 1.0 + np.array([problem.space.displacement(v)[corner] for v in result.fields])
        # the change of volume: J, or 1 + tr ε in small strains
        volume = stretch.prod(axis=1) if kinematics == "finite" else stretch.sum(axis=1) - 2.0
        fields = problem.fields(result.values, result.state)
        follows = np.abs(fields["chi"] - (np.diag(stretch[-1]) - np.eye(3))).max()
        label = f"{direction}, {kinematics} strains"
        stresses[label] = (strain, stress)
        volumes[label] = (strain, volume)
        lateral = np.delete(stretch[-1], axis)
        rows.append({"case": f"compression along {direction}, {kinematics} strains",
                     "nominal_stress_MPa": float(stress[-1]), "volume_ratio": float(volume[-1]),
                     "lateral_stretches": "/".join(f"{s:.3f}" for s in lateral),
                     "p": float(fields["p"].mean()), "time_s": time.time() - t0})  # fmt: skip
        print(
            f"compression along {direction} to 50 % ({kinematics} strains): nominal stress "
            f"{stress[-1]:.2f} MPa, volume ratio {volume[-1]:.3f}, lateral stretches "
            f"{np.round(lateral, 3)}, relative density {0.035 / volume[-1]:.4f} (0.035 at "
            f"rest), |chi - (F - 1)| {follows:.1e}, {time.time() - t0:.0f} s"
        )
figure(stresses, "nominal compressive strain", "nominal stress (MPa)", "foam_compression.png")
figure(volumes, "nominal compressive strain", "volume ratio", "foam_densification.png")

# The notched plate of a hardening von Mises solid cracked by micromorphic ductile damage
# in finite strains: the damage follows the micromorphic cumulated plastic strain, whose
# length sets the width of the bands; two bands leave the notch tip at about 45° (the shear
# bands of a plate nearly in plane stress: the damage follows the plastic strain alone,
# without the triaxiality of void growth) and cross the ligament to the gripped corners,
# and the force-displacement curves of two meshes agree
ductile = {}
for h in (0.05,) if quick else (0.05, 0.04):
    t0 = time.time()
    material = MicromorphicDuctile(young=200.0, poisson=0.3, yield_stress=0.4, hardening=10.0,
                                   length=0.1, onset=0.002, critical=0.02)  # fmt: skip
    problem = DuctileProblem(notched_plate(h), material, kinematics="finite")
    result = problem.run(pulled(problem, 0.03), steps=40)
    ductile[f"elements of {h}"] = crack(problem, result, h, "ductile crack", 0.03, t0)
figure(ductile, "displacement of the top", "force per unit thickness", "ductile_crack.png")
write_table(out / "microfoam_examples.csv", rows)
