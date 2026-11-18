"""A stand-in for an installation of OpenRadioss, for the tests of morphogen.dynamics: its
Starter, Engine and converters are Python scripts that read the inputs morphoGen writes
(their header, their manifest and mesh) and write outputs shaped like OpenRadioss's: the
listings with their cycle lines and terminations, animation files that its anim_to_vtk
turns into legacy VTK files (displacements, reaction forces, von Mises stresses, the status
of the elements, the identifiers of the nodes and elements), a time history that its
th_to_csv turns into CSV. The Engine deforms the morphology as a bar of the apparent
modulus ``E phi`` (phi its solid fraction): the displacement of the driven face, linear
across the box, and the reaction forces of the faces that carry it. It tests the control
flow and the reading of the results, not OpenRadioss (POSIX only: the scripts run by their
shebang)."""

from __future__ import annotations

import sys
from pathlib import Path

COMMON = r"""
import json, sys, time
from pathlib import Path
import numpy as np

def option(name):
    args = sys.argv[1:]
    return args[args.index(name) + 1] if name in args else None

def manifest():
    return json.loads(Path("morphogen_radioss.json").read_text())
"""

STARTER = r"""
deck = Path(option("-i"))
name = deck.name[: -len("_0000.rad")]
text = deck.read_text()
out = Path(f"{name}_0000.out")
if not text.startswith("#RADIOSS STARTER") or "/BEGIN" not in text or "/END" not in text:
    out.write_text(" ERROR ID : 1\n ** ERROR: NOT A RADIOSS STARTER INPUT\n ** ERROR TERMINATION\n")
    sys.exit(3)
if "FAKE_STARTER_ERROR" in text:
    out.write_text(" ERROR ID :  55\n ** ERROR IN MATERIAL DEFINITION\n ** ERROR TERMINATION\n")
    sys.exit(2)
nodes = text.split("/NODE", 1)[1].split("#---1", 1)[0].strip().splitlines()[1:]
Path(f"{name}_0001.rst").write_text(f"restart of {len(nodes)} nodes, {option('-np')} domains")
out.write_text(f" {len(nodes)} NODES\n ** NORMAL TERMINATION\n")
print("starter done")
"""

ENGINE = r"""
deck = Path(option("-i"))
name = deck.name[: -len("_0001.rad")]
if not Path(f"{name}_0001.rst").is_file():
    Path(f"{name}_0001.out").write_text(" ** ERROR: NO RESTART FILE\n ** ERROR TERMINATION\n")
    sys.exit(4)
m = manifest()
from morphogen.dynamics.model import ExplicitLoading
loading = ExplicitLoading.from_config(m["loading"])
mesh = np.load(f"{name}_mesh.npz")
nodes, elements = mesh["nodes"], mesh["elements"]
axis, drive = m["axis"], m["driven"]
lengths, lower = np.array(m["lengths"]), np.array(m["lower"])
young = list(m["materials"].values())[0]["young"]
modulus = young * m["solid_volume"] / m["volume"]
end, frames, history = m["end_time"], m["frames"], m["history"]
listing = open(f"{name}_0001.out", "w")
listing.write("     CYCLE    TIME      TIME-STEP  ELEMENT          ERROR  I-ENERGY"
              "    K-ENERGY T  K-ENERGY R  EXT-WORK     MAS.ERR\n")
listing.flush()

def state(t):
    if loading.kind == "impact":
        v = loading.sign() * loading.speed(lengths)
        u = np.zeros_like(nodes)
        u[:, axis] = v * t
        return u, np.zeros_like(nodes), 0.0
    d = float(loading.displacement(np.array([t]), lengths)[0])
    u = np.zeros_like(nodes)
    u[:, drive] = d * (nodes[:, axis] - lower[axis]) / lengths[axis]
    force = modulus * m["area"] * d / lengths[axis]  # on the upper face, along the drive
    reac = np.zeros_like(nodes)
    up, low = mesh["upper_nodes"], mesh["lower_nodes"]
    reac[up, drive] = force / len(up)
    reac[low, drive] = -force / len(low)
    return u, reac, abs(modulus * d / lengths[axis])

steps = 20
for k in range(steps + 1):
    t = end * k / steps
    listing.write(f"{100 * k:10d}  {t:.4E}  {end / 1e4:.4E} SOLID     1   0.0%"
                  "   0.000   0.000   0.000   0.000   0.000\n")
    listing.flush()
    time.sleep(0.02)
for k in range(frames + 1):
    t = end * k / frames
    u, reac, vm = state(t)
    Path(f"{name}A{k + 1:03d}").write_text(json.dumps({
        "time": t, "points": (nodes + u).tolist(), "disp": u.tolist(), "reac": reac.tolist(),
        "elements": elements.tolist(), "numbers": mesh["numbers"].tolist(),
        "vonm": [vm] * len(elements), "off": [1.0] * len(elements)}))
times = np.linspace(0.0, end, history + 1)
columns = {"time": times.tolist()}
for t_index, t in enumerate(times):
    u, reac, vm = state(t)
    for face, prefix in (("lower_nodes", "L"), ("upper_nodes", "U")):
        for n in mesh[face]:
            name_ = f"NODE {prefix}{n + 1} REAC{'XYZ'[drive]}"
            columns.setdefault(name_, []).append(float(reac[n, drive]))
    volume = m["solid_volume"]
    energy = 0.5 * vm * vm / max(modulus, 1e-300) * volume
    columns.setdefault("PART 1 IE", []).append(energy)
    v = loading.sign() * loading.speed(lengths) if loading.kind == "impact" else 0.0
    columns.setdefault("PART 1 KE", []).append(0.5 * m["mass"] * v * v * (1 - t / end) ** 2)
    columns.setdefault(f"PART 1 {'XYZ'[axis]}MOM", []).append(m["mass"] * v * (1 - t / end))
Path(f"{name}T01").write_text(json.dumps(columns))
listing.write(" ** NORMAL TERMINATION\n")
listing.close()
"""

ANIM_TO_VTK = r"""
frame = json.loads(Path(sys.argv[1]).read_text())
points, elements = np.array(frame["points"]), np.array(frame["elements"])
k = elements.shape[1]
kind = {8: 12, 4: 10, 10: 24}[k]
lines = ["# vtk DataFile Version 3.0", f"anim TIME = {frame['time']!r}", "ASCII",
         "DATASET UNSTRUCTURED_GRID", f"POINTS {len(points)} float"]
lines += [" ".join(f"{v:.9g}" for v in p) for p in points]
lines.append(f"CELLS {len(elements)} {len(elements) * (k + 1)}")
lines += [f"{k} " + " ".join(str(n) for n in e) for e in elements]
lines.append(f"CELL_TYPES {len(elements)}")
lines += [str(kind)] * len(elements)
lines += [f"POINT_DATA {len(points)}", "FIELD FieldData 1", f"NODE_ID 1 {len(points)} int"]
lines += [" ".join(str(i + 1) for i in range(len(points)))]
lines.append("VECTORS Displacement float")
lines += [" ".join(f"{v:.9g}" for v in p) for p in frame["disp"]]
lines.append("VECTORS Reaction_Forces float")
lines += [" ".join(f"{v:.9g}" for v in p) for p in frame["reac"]]
lines += [f"CELL_DATA {len(elements)}", "FIELD FieldData 1", f"ELEMENT_ID 1 {len(elements)} int"]
lines += [" ".join(str(i) for i in frame["numbers"])]
lines += ["SCALARS Von_Mises float 1", "LOOKUP_TABLE default"]
lines += [" ".join(f"{v:.9g}" for v in frame["vonm"])]
lines += ["SCALARS Element_Status float 1", "LOOKUP_TABLE default"]
lines += [" ".join(f"{v:g}" for v in frame["off"])]
sys.stdout.write("\n".join(lines) + "\n")
"""

TH_TO_CSV = r"""
columns = json.loads(Path(sys.argv[1]).read_text())
names = list(columns)
with open(sys.argv[1] + ".csv", "w") as f:
    f.write(",".join(f'"{n}"' for n in names) + "\n")
    for row in zip(*columns.values()):
        f.write(",".join(f"{v:.9g}" for v in row) + "\n")
"""


def fake_installation(root, *, suffix: str = "linux64_gf", converters: bool = True) -> Path:
    """Write a stand-in installation of OpenRadioss in ``root`` (its exec, hm_cfg_files
    and extlib folders) and return ``root``."""
    root = Path(root)
    exec_dir = root / "exec"
    exec_dir.mkdir(parents=True, exist_ok=True)
    (root / "hm_cfg_files").mkdir(exist_ok=True)
    (root / "extlib" / "hm_reader" / suffix.split("_")[0]).mkdir(parents=True, exist_ok=True)
    (root / "extlib" / "h3d" / "lib" / suffix.split("_")[0]).mkdir(parents=True, exist_ok=True)
    programs = {"starter_": STARTER, "engine_": ENGINE}
    if converters:
        programs.update({"anim_to_vtk_": ANIM_TO_VTK, "th_to_csv_": TH_TO_CSV})
    for prefix, body in programs.items():
        path = exec_dir / f"{prefix}{suffix}"
        path.write_text(f"#!{sys.executable}\n{COMMON}\n{body}", encoding="utf-8")
        path.chmod(0o755)
    return root
