"""A stand-in for laplacianFoam, for the tests of morphogen.io.openfoam: it reads a case
written by morphoGen with parsers of its own (the mesh of constant/polyMesh, the boundary
conditions of 0/T, the diffusivity of constant/transportProperties), solves the steady
diffusion by finite volumes with the flux of two points between the centres of the cells
(exact on the orthogonal meshes of voxels; the faces of a cyclic pair coupled face by face,
in their order), and writes what the function objects of the case write: the integral of
the Gauss gradient of T over the cells (postProcessing/gradientIntegral/0/volFieldValue.dat).
It tests the meshes and the boundary conditions that morphoGen writes, not OpenFOAM."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


def _strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"FoamFile\s*\{.*?\}", "", text, count=1, flags=re.S)


def _items(path: Path) -> str:
    body = _strip(path.read_text())
    start = re.search(r"\d+\s*\(", body).end()
    return body[start : body.rindex(")")]


def read_case(case) -> dict:
    """The mesh, the boundary conditions of T and the diffusivity of a case."""
    case = Path(case)
    mesh = case / "constant" / "polyMesh"
    points = np.array(
        [
            [float(v) for v in p.split()]
            for p in re.findall(r"\(([^()]*)\)", _items(mesh / "points"))
        ]
    )
    faces = [
        np.array(f.split(), dtype=int)
        for f in re.findall(r"\d+\(([^()]*)\)", _items(mesh / "faces"))
    ]
    owner = np.array(_items(mesh / "owner").split(), dtype=int)
    neighbour = np.array(_items(mesh / "neighbour").split(), dtype=int)
    patches = {}
    for name, block in re.findall(r"(\w+)\s*\{([^{}]*)\}", _items(mesh / "boundary")):
        entries = dict(re.findall(r"(\w+)\s+([^;]+);", block))
        patches[name] = entries
    field = _strip((case / "0" / "T").read_text())
    conditions = {}
    inner = field[field.index("boundaryField") :]
    for name, block in re.findall(r"(\w+)\s*\{([^{}]*)\}", inner):
        entries = dict(re.findall(r"(\w+)\s+([^;]+);", block))
        conditions[name] = entries
    dt = float(
        re.search(
            r"DT\s+\[[^\]]*\]\s+([-+.\deE]+);",
            (case / "constant" / "transportProperties").read_text(),
        ).group(1)
    )
    return {
        "points": points,
        "faces": faces,
        "owner": owner,
        "neighbour": neighbour,
        "patches": patches,
        "conditions": conditions,
        "DT": dt,
    }


def solve(case) -> dict:
    """The steady diffusion of a case: T in the cells, the flux through each patch (out
    of the domain), and the outputs of the function objects."""
    data = read_case(case)
    x, faces = data["points"], data["faces"]
    owner, neighbour = data["owner"], data["neighbour"]
    n_internal = len(neighbour)
    n_cells = int(max(owner.max(), neighbour.max(initial=-1)) + 1)
    centre = np.array([x[f].mean(axis=0) for f in faces])
    area = np.array(
        [
            0.5
            * np.cross(x[f] - x[f].mean(axis=0), np.roll(x[f], -1, axis=0) - x[f].mean(axis=0)).sum(
                axis=0
            )
            for f in faces
        ]
    )
    vol = np.zeros(n_cells)
    flux = np.einsum("ij,ij->i", centre, area) / 3.0
    np.add.at(vol, owner, flux)
    np.add.at(vol, neighbour, -flux[:n_internal])
    # the centroids of the cells: of the pyramids from the origin over their faces, of
    # volumes (1/3) c . S and centroids (3/4) c
    weighted = np.zeros((n_cells, 3))
    contrib = np.einsum("ij,ij->i", centre, area)[:, None] * centre / 4.0
    np.add.at(weighted, owner, contrib)
    np.add.at(weighted, neighbour, -contrib[:n_internal])
    cells = weighted / vol[:, None]
    dt = data["DT"]
    rows, cols, vals = [], [], []
    rhs = np.zeros(n_cells)
    mag = np.linalg.norm(area, axis=1)

    def couple(a, b, coef):
        rows.extend([a, b, a, b])
        cols.extend([a, b, b, a])
        vals.extend([coef, coef, -coef, -coef])

    for f in range(n_internal):
        couple(
            owner[f],
            neighbour[f],
            dt * mag[f] / np.linalg.norm(cells[neighbour[f]] - cells[owner[f]]),
        )
    boundary_value = {}
    for name, p in data["patches"].items():
        start, size = int(p["startFace"]), int(p["nFaces"])
        bc = data["conditions"][name]
        kind = bc["type"]
        if kind == "fixedValue":
            value = float(bc["value"].split()[-1])
            for f in range(start, start + size):
                coef = dt * mag[f] / np.linalg.norm(centre[f] - cells[owner[f]])
                rows.append(owner[f])
                cols.append(owner[f])
                vals.append(coef)
                rhs[owner[f]] += coef * value
                boundary_value[f] = (coef, value)
        elif kind == "cyclic" and name < p["neighbourPatch"]:
            other = data["patches"][p["neighbourPatch"]]
            o_start = int(other["startFace"])
            separation = np.array(p["separationVector"].strip("()").split(), dtype=float)
            for k in range(size):
                f, g = start + k, o_start + k
                assert (
                    np.abs(
                        x[faces[g]]
                        - (x[faces[f]] - separation)[[0, *range(len(faces[f]) - 1, 0, -1)]]
                    ).max()
                    < 1e-9
                )
                d = np.linalg.norm(cells[owner[g]] - (cells[owner[f]] - separation))
                couple(owner[f], owner[g], dt * mag[f] / d)
        else:
            assert kind in ("zeroGradient", "symmetryPlane", "cyclic"), kind
    k = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    t = splinalg.spsolve(k.tocsc(), rhs)
    out_flux = {}
    for name, p in data["patches"].items():
        start, size = int(p["startFace"]), int(p["nFaces"])
        out_flux[name] = sum(
            coef * (t[owner[f]] - value)
            for f, (coef, value) in boundary_value.items()
            if start <= f < start + size
        )
    # the Gauss gradient (linear interpolation on the faces) integrated over the cells
    face_t = np.empty(len(faces))
    w = np.linalg.norm(cells[neighbour] - centre[:n_internal], axis=1)
    w /= w + np.linalg.norm(cells[owner[:n_internal]] - centre[:n_internal], axis=1)
    face_t[:n_internal] = w * t[owner[:n_internal]] + (1 - w) * t[neighbour]
    face_t[n_internal:] = t[owner[n_internal:]]
    for f, (_, value) in boundary_value.items():
        face_t[f] = value
    for name, p in data["patches"].items():
        if data["conditions"][name]["type"] == "cyclic":
            other = data["patches"][p["neighbourPatch"]]
            for k_ in range(int(p["nFaces"])):
                f, g = int(p["startFace"]) + k_, int(other["startFace"]) + k_
                face_t[f] = 0.5 * (t[owner[f]] + t[owner[g]])
    integral = (face_t[n_internal:, None] * area[n_internal:]).sum(axis=0)
    folder = Path(case) / "postProcessing" / "gradientIntegral" / "0"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "volFieldValue.dat").write_text(
        "# Volume field value\n# Time volIntegrate(grad(T))\n"
        f"1 ({integral[0]:.15g} {integral[1]:.15g} {integral[2]:.15g})\n"
    )
    return {"T": t, "flux": out_flux, "volumes": vol, "centres": cells, "integral": integral}
