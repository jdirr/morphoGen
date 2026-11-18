"""Medit ASCII meshes (``.mesh``) and solutions (``.sol``), as used by MMG."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_BLOCKS = {
    # keyword: number of integer node columns (followed by a reference)
    "Tetrahedra": 4,
    "Triangles": 3,
    "Edges": 2,
}


def _write_block(f, name: str, data: np.ndarray, fmt: str) -> None:
    f.write(f"{name}\n{len(data)}\n")
    if len(data):
        np.savetxt(f, data, fmt=fmt)


def write_mesh(
    path,
    nodes,
    tetrahedra=None,
    triangles=None,
    *,
    edges=None,
    node_refs=None,
    tet_refs=None,
    tri_refs=None,
    edge_refs=None,
    required_triangles=None,
    required_edges=None,
    required_vertices=None,
    corners=None,
    ridges=None,
    comments=(),
) -> Path:
    """Write vertices, tetrahedra, triangles and edges (0-based indices) with references.

    Nodes with two coordinates make a 2D mesh (``Dimension 2``, for ``mmg2d``).
    ``required_*`` (0-based indices) are written as ``RequiredVertices``,
    ``RequiredEdges`` and ``RequiredTriangles``: MMG keeps them unchanged. ``corners``
    (vertices) and ``ridges`` (edges) mark sharp features. ``comments`` are written as
    ``#`` lines after the header.
    """
    path = Path(path)
    nodes = np.asarray(nodes, dtype=float)
    dim = nodes.shape[1]
    if dim not in (2, 3):
        raise ValueError("write_mesh: nodes with 2 or 3 coordinates")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"MeshVersionFormatted 2\nDimension {dim}\n")
        for line in comments:
            f.write(f"# {line}\n")
        refs = np.zeros(len(nodes), dtype=np.int64) if node_refs is None else node_refs
        fmt = " ".join(["%.17g"] * dim) + " %d"
        _write_block(f, "Vertices", np.column_stack([nodes, refs]), fmt)
        for name, cells, refs in (
            ("Tetrahedra", tetrahedra, tet_refs),
            ("Triangles", triangles, tri_refs),
            ("Edges", edges, edge_refs),
        ):
            if cells is None:
                continue
            cells = np.asarray(cells, dtype=np.int64)
            r = np.zeros(len(cells), dtype=np.int64) if refs is None else np.asarray(refs)
            _write_block(f, name, np.column_stack([cells + 1, r]), "%d")
        for name, req in (
            ("RequiredVertices", required_vertices),
            ("Corners", corners),
            ("RequiredEdges", required_edges),
            ("Ridges", ridges),
            ("RequiredTriangles", required_triangles),
        ):
            if req is not None:
                req = np.asarray(req, dtype=np.int64) + 1
                _write_block(f, name, req[:, None], "%d")
        f.write("End\n")
    return path


def write_sol(path, values, dimension: int = 3) -> Path:
    """Scalar solution at the vertices (e.g. a level-set function, or the sizes of the
    elements) of a mesh of ``dimension`` 3 or 2."""
    path = Path(path)
    values = np.asarray(values, dtype=float).ravel()
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"MeshVersionFormatted 2\nDimension {int(dimension)}\nSolAtVertices\n"
                f"{len(values)}\n1 1\n")  # fmt: skip
        np.savetxt(f, values, fmt="%.17g")
        f.write("End\n")
    return path


def read_sol(path) -> np.ndarray:
    """The scalar solution at the vertices of a ``.sol`` file (as written by MMG)."""
    tokens = Path(path).read_text(encoding="ascii", errors="replace").split()
    k = tokens.index("SolAtVertices")
    count, n_sol, kind = int(tokens[k + 1]), int(tokens[k + 2]), int(tokens[k + 3])
    if n_sol != 1 or kind != 1:
        raise ValueError(f"{path}: not a scalar solution")
    return np.asarray(tokens[k + 4 : k + 4 + count], dtype=float)


def read_mesh(path) -> dict[str, np.ndarray]:
    """Vertices (n, 3), vertex refs, and for each element block its 0-based
    connectivity and references (keys ``Tetrahedra``, ``Tetrahedra_refs``...); the
    entities that the file marks, as 0-based indices (``Ridges`` and ``RequiredEdges``
    among the edges, ``Corners`` and ``RequiredVertices`` among the vertices,
    ``RequiredTriangles``)."""
    text = Path(path).read_text(encoding="ascii", errors="replace")
    out: dict[str, np.ndarray] = {}
    dim_match = re.search(r"^\s*Dimension\s+(\d)", text, re.M)
    dim = int(dim_match.group(1)) if dim_match else 3
    for match in re.finditer(r"^\s*(Vertices|Tetrahedra|Triangles|Edges)\s*$", text, re.M):
        name = match.group(1)
        rest = text[match.end() :].lstrip()
        count_str, _, body = rest.partition("\n")
        count = int(count_str.split()[0])
        ncol = dim + 1 if name == "Vertices" else _BLOCKS[name] + 1
        if count == 0:
            values = np.zeros((0, ncol))
        else:
            lines = body.split("\n", count)[:count]
            values = np.array(" ".join(lines).split(), dtype=float).reshape(count, ncol)
        if name == "Vertices":
            out["Vertices"] = values[:, :dim]
            out["Vertices_refs"] = values[:, dim].astype(np.int64)
        else:
            out[name] = values[:, :-1].astype(np.int64) - 1
            out[f"{name}_refs"] = values[:, -1].astype(np.int64)
    # the entities that MMG marks: 0-based indices of vertices, edges or triangles
    for match in re.finditer(
        r"^\s*(Ridges|Corners|RequiredVertices|RequiredEdges|RequiredTriangles)\s*$", text, re.M
    ):
        words = text[match.end() :].split(None, 1)
        count = int(words[0])
        values = words[1].split(None, count)[:count] if count else []
        out[match.group(1)] = np.array(values, dtype=np.int64) - 1
    if "Vertices" not in out:
        raise ValueError(f"{path}: no vertices found")
    out["comments"] = re.findall(r"^#\s?(.*)$", text, re.M)
    return out
