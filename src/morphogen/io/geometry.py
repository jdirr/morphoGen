"""Explicit ("vector") geometry: JSON descriptions, CSV tables and Gmsh scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from morphogen._version import __version__
from morphogen.models.base import Microstructure
from morphogen.models.tessellation import Tessellation
from morphogen.models.union import PrimitiveUnion

#: Relative enlargement of the spheres closing round-capped cylinders in Gmsh scripts.
CAP_ENLARGEMENT = 0.01


def write_json(path, micro: Microstructure) -> Path:
    """Full description of a realization; :func:`read_json` restores it exactly."""
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(micro.to_dict(), f)
        f.write("\n")
    return path


def read_json(path) -> Microstructure:
    """A microstructure read from a JSON file
    (:meth:`~morphogen.models.base.Microstructure.from_dict`)."""
    with open(Path(path), encoding="utf-8") as f:
        return Microstructure.from_dict(json.load(f))


def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> Path:
    names = list(columns)
    data = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(",".join(names) + "\n")
        np.savetxt(f, data, delimiter=",", fmt="%.12g")
    return path


def write_csv(path, micro: Microstructure) -> list[Path]:
    """Tables of primitives or cells: ``<stem>_spheres.csv``, ``<stem>_segments.csv``,
    ``<stem>_cells.csv`` (only the relevant ones are written)."""
    path = Path(path)
    stem = path.with_suffix("")
    out = []
    if isinstance(micro, PrimitiveUnion):
        s, g = micro.spheres, micro.segments
        if len(s):
            out.append(
                _write_csv(
                    Path(f"{stem}_spheres.csv"),
                    {
                        "x": s.centers[:, 0],
                        "y": s.centers[:, 1],
                        "z": s.centers[:, 2],
                        "radius": s.radii,
                    },
                )
            )
        if len(g):
            out.append(
                _write_csv(
                    Path(f"{stem}_segments.csv"),
                    {
                        "x0": g.start[:, 0],
                        "y0": g.start[:, 1],
                        "z0": g.start[:, 2],
                        "x1": g.end[:, 0],
                        "y1": g.end[:, 1],
                        "z1": g.end[:, 2],
                        "radius": g.radii,
                    },
                )
            )
    elif isinstance(micro, Tessellation):
        out.append(_write_csv(Path(f"{stem}_cells.csv"), micro.cell_table()))
    else:
        raise ValueError(f"no tabular description for {type(micro).__name__}")
    return out


def write_geo(path, micro: PrimitiveUnion, *, phase: int = 1, mesh_size: float | None = None):
    """Gmsh (OpenCASCADE) script building phase 1 (the union of the primitives clipped
    by the box) or phase 0 (the box minus the union), for conforming tetrahedral meshing
    with ``gmsh -3 file.geo``. Gmsh is an external program (GPL); nothing is linked.
    """
    if not isinstance(micro, PrimitiveUnion):
        raise ValueError("Gmsh scripts can only be written for unions of spheres and cylinders")
    if phase not in (0, 1):
        raise ValueError("phase: 0 or 1")
    path = Path(path)
    domain = micro.domain
    spheres, segments = micro._covering(0.0)
    lo, size = domain.lower, domain.lengths
    if mesh_size is None:
        radii = np.concatenate([spheres.radii, segments.radii])
        mesh_size = float(np.median(radii)) / 2.0 if len(radii) else float(size.min()) / 10.0
    model = micro.model.name if micro.model is not None else "primitives"

    def fmt(v):
        return ", ".join(f"{float(x):.12g}" for x in v)

    lines = [
        f"// morphoGen {__version__}: {model}, phase {phase}",
        "// Mesh with: gmsh -3 <this file>  (OpenCASCADE booleans on many objects can be slow)",
    ]
    if domain.periodic:
        lines.append("// The geometry is periodic; impose periodic meshes in Gmsh if needed.")
    lines += [
        'SetFactory("OpenCASCADE");',
        f"Mesh.MeshSizeMax = {mesh_size:.12g};",
        f"Mesh.MeshSizeMin = {mesh_size / 4:.12g};",
        f"Box(1) = {{{fmt(lo)}, {fmt(size)}}};",
    ]
    if segments.round_caps and len(segments):
        lines.append(
            f"// Round caps: spheres {1 + CAP_ENLARGEMENT:g} times the fibre radius, to avoid "
            "the tangency of equal spheres and cylinders that breaks OpenCASCADE booleans."
        )
    tag = 2
    for c, r in zip(spheres.centers, spheres.radii, strict=True):
        lines.append(f"Sphere({tag}) = {{{fmt(c)}, {r:.12g}}};")
        tag += 1
    for a, b, r in zip(segments.start, segments.end, segments.radii, strict=True):
        lines.append(f"Cylinder({tag}) = {{{fmt(a)}, {fmt(b - a)}, {r:.12g}}};")
        tag += 1
        if segments.round_caps:
            rc = r * (1.0 + CAP_ENLARGEMENT)
            for end in (a, b):
                # caps entirely outside the box do not change the clipped geometry
                if domain.distance_to_box(end)[0] > rc:
                    continue
                lines.append(f"Sphere({tag}) = {{{fmt(end)}, {rc:.12g}}};")
                tag += 1
    last = tag - 1
    if last < 2:
        raise ValueError("the microstructure has no primitive inside the box")
    if last > 2:
        lines.append(
            f"grains() = BooleanUnion{{ Volume{{2}}; Delete; }}{{ Volume{{3:{last}}}; Delete; }};"
        )
    else:
        lines.append("grains() = {2};")
    if phase == 1:
        lines.append(
            "solid() = BooleanIntersection{ Volume{grains()}; Delete; }{ Volume{1}; Delete; };"
        )
    else:
        lines.append(
            "solid() = BooleanDifference{ Volume{1}; Delete; }{ Volume{grains()}; Delete; };"
        )
    lines.append(f'Physical Volume("phase_{phase}", {phase + 1}) = {{solid()}};')
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path
