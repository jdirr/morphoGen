"""The results of a run of OpenRadioss read back (:func:`read_explicit_run`): the nominal
stress and strain of the morphology over time (the reaction forces of its loaded faces
over the area of the face of its box, the displacement of the driven face over the
length of the box), its energies, the fields of its animations on its mesh (the
displacement, the von Mises stress, the plastic strain, the energy density, the
pressure and the status of each element), and the values that sum a test up
(:func:`summary`: the peak stress, the absorbed energy, the plateau stress and the
densification strain of a crushing, the equilibrium of the faces, the rebound of an
impact).

The forces come from the time history (``th_to_csv``) when its columns name the nodes of
the loaded faces, else from the reaction forces of the animations (``anim_to_vtk``), on
the nodes that the manifest of the run names; :func:`write_frames` writes the fields of
each animation as a ``.vtu`` file of tetrahedra (the bricks split in six) for the 3D views
of morphoGen."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The fields of the animations, by the words of their names in the files of anim_to_vtk
#: (lower case), and the words that exclude a name
FIELDS = {
    "displacement": (("disp",), ()),
    "velocity": (("vel",), ("accel",)),
    "reaction": (("reac",), ()),
    "contact": (("cont",), ("contour",)),
    "von_mises": (("vonm", "von_mises", "von mises", "vonmises", "von-mises"), ()),
    "plastic_strain": (("epsp", "plastic"), ()),
    "energy_density": (("ener",), ("hourg",)),
    "pressure": (("pres",), ()),
    "status": (("off", "status", "delet"), ("offset",)),
}
#: The fields of the elements kept in the frames written for the 3D views
CELL_FIELDS = ("von_mises", "plastic_strain", "energy_density", "pressure")
#: The 6 tetrahedra of a hexahedron in VTK's order (along its diagonal 0-6; positive)
KUHN = np.array(
    [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
)
#: VTK's types of the solid cells
SOLID_TYPES = (12, 10, 24)


def _find(arrays: dict, kind: str):
    """The array of a field (:data:`FIELDS`) among those of a file, or None."""
    words, exclude = FIELDS[kind]
    for name, values in arrays.items():
        low = name.lower()
        if any(w in low for w in words) and not any(x in low for x in exclude):
            return values
    if kind == "pressure":
        for name, values in arrays.items():
            if name.strip().lower() == "p":
                return values
    return None


def _ids(arrays: dict, words=("node", "id")):
    """An array of identifiers (integers named like ``NODE_ID``), or None."""
    for name, values in arrays.items():
        low = name.lower().replace(" ", "_")
        values = np.asarray(values)
        if values.ndim == 1 and all(w in low for w in words) and np.all(np.mod(values, 1) == 0):
            return values.astype(np.int64)
    return None


@dataclass
class ExplicitFrame:
    """An animation of a run on the mesh of the run: its time, the displacement and
    reaction force of each node (None when the animation has none), the fields of the
    elements and their status (1 active, 0 deleted)."""

    time: float
    displacement: np.ndarray | None
    reaction: np.ndarray | None
    cells: dict[str, np.ndarray] = field(default_factory=dict)
    status: np.ndarray | None = None


@dataclass
class ExplicitResult:
    """The results of a run (see the module): its ``manifest``, its ``mesh`` (nodes,
    elements, phases and loaded faces), its ``curves`` (time, displacement, strain,
    forces and nominal stresses of the upper and lower faces), its ``energies``
    (time -> internal, kinetic... when the time history gives them), its ``frames``
    and the ``source`` of its forces."""

    directory: Path
    manifest: dict
    mesh: dict
    curves: dict[str, np.ndarray]
    energies: dict[str, np.ndarray]
    frames: list[ExplicitFrame]
    source: str
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.manifest["name"]


def _times_of(grids: list[dict], titles: list[str], manifest: dict) -> np.ndarray:
    """The times of the animations: those their titles give, else their rank times the
    interval of the animations (the first at 0)."""
    times = []
    for title in titles:
        m = re.search(r"time\s*[=:]?\s*([-+]?\d+\.?\d*(?:[eEdD][-+]?\d+)?)", title, re.I)
        times.append(float(m.group(1).replace("D", "E").replace("d", "e")) if m else None)
    if all(t is not None for t in times) and len(set(times)) == len(times):
        return np.array(times, dtype=float)
    step = float(manifest["end_time"]) / int(manifest["frames"])
    return np.arange(len(grids), dtype=float) * step


def _title(path: Path) -> str:
    with open(path, "rb") as f:
        f.readline()
        return f.readline().decode("latin-1", "replace").strip()


def read_frames(
    directory, manifest: dict, mesh: dict, paths
) -> tuple[list[ExplicitFrame], list[str]]:
    """The animations (legacy VTK files of anim_to_vtk) on the mesh of the run: their
    nodes matched by their identifiers (an array named like NODE_ID), by their order when
    they are as many, else by their positions in the first animation; their elements
    likewise (ELEMENT_ID)."""
    from scipy.spatial import cKDTree

    from morphogen.io.vtk import cells_of_type, read_vtk_grid

    notes: list[str] = []
    paths = list(paths)
    if not paths:
        return [], notes
    nodes = np.asarray(mesh["nodes"], dtype=float)
    n_nodes, n_elements = len(nodes), len(mesh["elements"])
    grids = [read_vtk_grid(p) for p in paths]
    times = _times_of(grids, [_title(Path(p)) for p in paths], manifest)
    first = grids[0]
    ids = _ids(first["point_data"]) if first["point_data"] else None
    if ids is not None and ids.max(initial=0) <= n_nodes and ids.min(initial=1) >= 1:
        node_index = ids - 1  # the point k of the files is the node node_index[k]
    elif len(first["points"]) == n_nodes:
        node_index = np.arange(n_nodes)
    else:
        u0 = _find(first["point_data"], "displacement")
        ref = first["points"] - (0.0 if u0 is None else np.asarray(u0, dtype=float))
        distance, node_index = cKDTree(nodes).query(ref)
        far = distance > 1e-6 * float(np.ptp(nodes, axis=0).max())
        if np.any(far):
            notes.append(f"{int(far.sum())} points of the animations are not nodes of the mesh")
            node_index = np.where(far, -1, node_index)
    cell_index = None
    for cell_type in SOLID_TYPES:
        _, picked = cells_of_type(first, cell_type)
        if len(picked):
            eids = _ids(
                {
                    k: np.asarray(v)[picked]
                    for k, v in first["cell_data"].items()
                    if np.ndim(v) == 1 and len(v) == len(first["types"])
                },
                ("elem",),
            )
            if eids is not None and eids.min(initial=1) >= 1:
                numbers = np.asarray(mesh["numbers"])  # element e has the number numbers[e]
                lookup = np.full(int(max(eids.max(initial=0), numbers.max(initial=0))) + 1, -1)
                lookup[numbers] = np.arange(n_elements)
                cell_index = (picked, lookup[eids])
            elif len(picked) == n_elements:
                numbers = np.asarray(mesh["numbers"])
                cell_index = (picked, np.argsort(numbers))  # the files in the order of the numbers
            else:
                notes.append(
                    f"{len(picked)} solid cells in the animations for {n_elements} elements: "
                    "their fields are left out"
                )
            break
    valid = node_index >= 0

    def on_nodes(values):
        """Values of the points of the files on the nodes of the mesh (0 elsewhere)."""
        if values is None:
            return None
        values = np.asarray(values, dtype=float)
        out = np.zeros((n_nodes, *values.shape[1:]))
        out[node_index[valid]] = values[: len(node_index)][valid]
        return out

    frames = []
    for t, grid in zip(times, grids, strict=True):
        point = grid["point_data"]
        u = on_nodes(_find(point, "displacement"))
        if u is None and len(grid["points"]) == len(first["points"]):
            u = on_nodes(grid["points"] - first["points"])
        frame = ExplicitFrame(float(t), u, on_nodes(_find(point, "reaction")))
        if cell_index is not None:
            picked, elements = cell_index
            data = {
                k: np.asarray(v)[picked]
                for k, v in grid["cell_data"].items()
                if np.ndim(v) >= 1 and len(v) == len(grid["types"])
            }
            for kind in (*CELL_FIELDS, "status"):
                values = _find(data, kind)
                if values is None:
                    continue
                values = np.asarray(values, dtype=float)
                if values.ndim > 1:  # a tensor or a vector: its norm
                    values = np.linalg.norm(values, axis=1)
                out = np.full(n_elements, np.nan)
                ok = elements >= 0
                out[elements[ok]] = values[ok]
                if kind == "status":
                    frame.status = np.where(np.isnan(out), 1.0, out)
                else:
                    frame.cells[kind] = out
        frames.append(frame)
    return frames, notes


def _face_forces_th(th: dict, manifest: dict, mesh: dict):
    """The forces of the loaded faces along the drive from the time history, when its
    columns name the nodes (``L<id>``, ``U<id>``, or their identifiers): (lower, upper),
    or None."""
    letter = "XYZ"[int(manifest["driven"])]
    lower_ids = set((np.asarray(mesh["lower_nodes"]) + 1).tolist())
    upper_ids = set((np.asarray(mesh["upper_nodes"]) + 1).tolist())
    sums = {"lower": None, "upper": None}
    counts = {"lower": 0, "upper": 0}
    for name, values in th.items():
        up = name.upper()
        if f"REAC{letter}" not in up:
            continue
        m = re.search(r"(?<![A-Z0-9])([LU])(\d+)(?![0-9])", up)
        if m:
            face = "lower" if m.group(1) == "L" else "upper"
        else:
            numbers = {int(x) for x in re.findall(r"\d+", up)}
            if numbers & lower_ids and not numbers & upper_ids:
                face = "lower"
            elif numbers & upper_ids and not numbers & lower_ids:
                face = "upper"
            else:
                continue
        sums[face] = values.copy() if sums[face] is None else sums[face] + values
        counts[face] += 1
    expected = {"lower": len(lower_ids), "upper": len(upper_ids)}
    if any(counts[f] != expected[f] for f in counts):
        return None
    return sums["lower"], sums["upper"]


def _energies_th(th: dict) -> dict[str, np.ndarray]:
    """The energies and momenta of the time history (its columns IE, KE, HE, XMOM...,
    summed over the parts; the global ones alone when it has them)."""
    kinds = {
        "IE": "internal",
        "KE": "kinetic",
        "HE": "hourglass",
        "XMOM": "momentum_x",
        "YMOM": "momentum_y",
        "ZMOM": "momentum_z",
    }
    columns: dict[str, list[tuple[bool, np.ndarray]]] = {}
    for name, values in th.items():
        tokens = re.split(r"[^A-Z0-9]+", name.upper())
        for token, kind in kinds.items():
            if token in tokens:
                columns.setdefault(kind, []).append(
                    ("GLOBAL" in tokens or "GLOB" in tokens, values)
                )
    out = {}
    for kind, found in columns.items():
        chosen = [v for g, v in found if g] or [v for _, v in found]
        out[kind] = np.sum(chosen, axis=0)
    if out:
        out["time"] = th["time"]
    return out


def read_explicit_run(directory) -> ExplicitResult:
    """The results of a run of OpenRadioss in ``directory`` (see the module), from its
    manifest and mesh (:func:`morphogen.io.write_radioss`), its animations converted to
    VTK and its time history converted to CSV (:func:`morphogen.dynamics.convert_outputs`)."""
    from morphogen.io.radioss import read_manifest, read_radioss_th

    directory = Path(directory)
    manifest = read_manifest(directory)
    name = manifest["name"]
    with np.load(directory / f"{name}_mesh.npz") as data:
        mesh = {k: data[k] for k in data.files}
    notes: list[str] = []
    vtk = sorted(directory.glob(f"{name}A*.vtk"), key=lambda p: p.name)
    frames, more = read_frames(directory, manifest, mesh, vtk)
    notes += more
    th, energies = {}, {}
    csvs = sorted(directory.glob(f"{name}T01*.csv"))
    if csvs:
        try:
            th = read_radioss_th(csvs[0])
            energies = _energies_th(th)
        except ValueError as exc:
            notes.append(f"time history not read: {exc}")
    lengths = np.asarray(manifest["lengths"], dtype=float)
    axis, drive, sign = int(manifest["axis"]), int(manifest["driven"]), float(manifest["sign"])
    area = float(manifest["area"])
    from morphogen.dynamics.model import ExplicitLoading

    loading = ExplicitLoading.from_config(manifest["loading"])
    curves: dict[str, np.ndarray] = {}
    source = "none"
    forces = _face_forces_th(th, manifest, mesh) if th and loading.kind != "impact" else None
    if forces is not None:
        times = np.asarray(th["time"], dtype=float)
        lower, upper = forces
        source = "time history"
    elif frames and loading.kind != "impact" and frames[0].reaction is not None:
        times = np.array([f.time for f in frames])
        upper = np.array([f.reaction[mesh["upper_nodes"], drive].sum() for f in frames])
        lower = np.array([f.reaction[mesh["lower_nodes"], drive].sum() for f in frames])
        source = "animations"
    else:
        times, lower, upper = None, None, None
    if times is not None:
        u = loading.displacement(times, lengths)
        curves = {
            "time": times,
            "displacement": u,
            "strain": sign * u / lengths[axis],
            "force_upper": sign * upper,
            "force_lower": -sign * lower,
        }
        curves["stress_upper"] = curves["force_upper"] / area
        curves["stress_lower"] = curves["force_lower"] / area
    elif loading.kind == "impact" and energies.get(f"momentum_{'xyz'[axis]}") is not None:
        t = np.asarray(energies["time"], dtype=float)
        p = np.asarray(energies[f"momentum_{'xyz'[axis]}"], dtype=float)
        force = -sign * np.gradient(p, t) if len(t) > 1 else np.zeros_like(p)
        curves = {"time": t, "momentum": p, "force_wall": force, "stress_wall": force / area}
        source = "momentum"
    if not curves and loading.kind != "impact":
        notes.append(
            "no forces of the faces: neither the time history nor the animations give "
            "them (are anim_to_vtk and th_to_csv in the installation?)"
        )
    return ExplicitResult(directory, manifest, mesh, curves, energies, frames, source, notes)


# -- summary -------------------------------------------------------------------------------


def _integral(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The cumulative integral of y dx (trapezoids), from 0."""
    out = np.zeros_like(np.asarray(y, dtype=float))
    if len(x) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def summary(result: ExplicitResult) -> dict:
    """The values that sum a run up, in the units of the run: the final and peak nominal
    stresses and strains, the energy absorbed per unit volume of the box (the integral of
    the nominal stress over the nominal strain) and per unit mass, the mean gap between
    the stresses of the two faces over the second half of the run (the equilibrium of a
    quasi-static test: small when the waves have settled), the plateau stress (the mean
    between 20 and 40 % strain, after Ashby) and the densification strain (where the
    energy efficiency, the absorbed energy over the stress, peaks: Avalle et al., 2001)
    of a crushing; for an impact, the peak force on the wall and the kinetic energies;
    the energies at the end when the time history gives them; the size of the run."""
    m = result.manifest
    out: dict = {
        "loading": m["loading"]["kind"],
        "elements": m["n_elements"],
        "nodes": m["n_nodes"],
        "end time": m["end_time"],
        "stable time step (estimate)": m["critical_time_step"],
        "cycles (estimate)": round(m["cycles"]),
        "mass": m["mass"],
        "solid fraction": m["solid_volume"] / m["volume"],
    }
    c = result.curves
    if "stress_upper" in c and len(c["time"]) > 1:
        strain, stress = np.asarray(c["strain"]), np.asarray(c["stress_upper"])
        lower = np.asarray(c["stress_lower"])
        k = int(np.argmax(stress))
        work = _integral(stress, strain)
        out.update(
            {
                "final strain": float(strain[-1]),
                "final stress": float(stress[-1]),
                "peak stress": float(stress[k]),
                "strain at the peak": float(strain[k]),
                "absorbed energy per volume": float(work[-1]),
                "absorbed energy per mass": float(work[-1] * m["volume"] / m["mass"]),
            }
        )
        half = strain >= 0.5 * strain[-1]
        scale = np.maximum(np.abs(stress[half]), 1e-12 * max(np.max(np.abs(stress)), 1e-300))
        out["face stress gap (second half)"] = float(
            np.mean(np.abs(stress[half] - lower[half]) / scale)
        )
        if m["loading"]["kind"] == "compression":
            if strain[-1] >= 0.4:
                window = (strain >= 0.2) & (strain <= 0.4)
                out["plateau stress (20-40 %)"] = float(np.mean(stress[window]))
            efficiency = np.where(stress > 0, work / np.maximum(stress, 1e-300), 0.0)
            j = int(np.argmax(efficiency))
            if 0 < j < len(efficiency) - 1 and strain[j] > 0.1:
                out["densification strain"] = float(strain[j])
    elif "force_wall" in c and len(c["time"]) > 1:
        out["peak force on the wall"] = float(np.max(c["force_wall"]))
    e = result.energies
    for kind in ("internal", "kinetic", "hourglass"):
        if kind in e and len(e[kind]):
            out[f"{kind} energy at the end"] = float(e[kind][-1])
    if m["loading"]["kind"] == "impact" and "kinetic" in e and len(e["kinetic"]) > 1:
        k0, k1 = float(e["kinetic"][0]), float(e["kinetic"][-1])
        out["initial kinetic energy"] = k0
        if k0 > 0:
            out["kinetic energy kept"] = k1 / k0
            out["rebound speed / impact speed"] = float(np.sqrt(max(k1, 0.0) / k0))
    return out


# -- frames for the 3D views ---------------------------------------------------------------


def display_cells(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The tetrahedra that draw a mesh (its tetrahedra, or six per brick) and the element
    of each."""
    elements = np.asarray(elements, dtype=np.int64)
    if elements.shape[1] != 8:
        return elements, np.arange(len(elements))
    tets = elements[:, KUHN].reshape(-1, 4)
    return tets, np.repeat(np.arange(len(elements)), len(KUHN))


def write_frames(result: ExplicitResult, folder, *, prefix: str = "frame") -> list[Path]:
    """Write each animation of a run as ``<prefix>_0001.vtu``... in ``folder``: the
    elements still active (the bricks split in six tetrahedra), their phase and fields,
    the displacement of the nodes, the box (morphoGen's 3D views draw the last one)."""
    from morphogen.io.vtk import write_vtu

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mesh = result.mesh
    nodes = np.asarray(mesh["nodes"], dtype=float)
    tets, owner = display_cells(mesh["elements"])
    cell_type = 10 if tets.shape[1] == 4 else 24
    box = {
        "box_lower": np.asarray(result.manifest["lower"], dtype=float),
        "box_upper": np.asarray(result.manifest["upper"], dtype=float),
    }
    written = []
    for k, frame in enumerate(result.frames, start=1):
        keep = np.ones(len(tets), dtype=bool)
        if frame.status is not None:
            keep = frame.status[owner] > 0
        if not keep.any():
            continue
        data = {"phase": np.asarray(mesh["phase"])[owner][keep].astype(np.int32)}
        for kind, values in frame.cells.items():
            data[kind] = np.nan_to_num(values[owner][keep])
        point = {}
        if frame.displacement is not None:
            point["displacement"] = frame.displacement
        path = folder / f"{prefix}_{k:04d}.vtu"
        write_vtu(
            path,
            nodes,
            tets[keep],
            cell_type,
            data,
            point_data=point,
            field_data={**box, "time": np.array([frame.time])},
        )
        written.append(path)
    return written


def frames_mesh(result: ExplicitResult):
    """The mesh of the frames as a :class:`~morphogen.mesh.tetra.TetMesh` (the bricks
    split in six), for the videos of :mod:`morphogen.viz.video`."""
    from morphogen.mesh.tetra import TetMesh

    tets, owner = display_cells(result.mesh["elements"])
    m = result.manifest
    return TetMesh(
        np.asarray(result.mesh["nodes"], dtype=float),
        tets,
        np.asarray(result.mesh["phase"])[owner],
        m["lower"],
        m["upper"],
    ), owner


def write_tables(result: ExplicitResult, folder) -> list[Path]:
    """The curves and energies of a run as ``curves.csv`` and ``energies.csv``."""
    from morphogen.io.tables import write_table

    folder = Path(folder)
    return [
        write_table(
            folder / f"{name}.csv", {k: np.asarray(v, dtype=float) for k, v in columns.items()}
        )
        for name, columns in (("curves", result.curves), ("energies", result.energies))
        if columns
    ]
