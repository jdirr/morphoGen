"""Reference implementations used by several test modules."""

from __future__ import annotations

import numpy as np


def axis_covariance_periodic(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Reference covariance along the axes of a periodic indicator (direct sums)."""
    x = x.astype(float)
    out = []
    for h in range(max_lag + 1):
        out.append(np.mean([np.mean(x * np.roll(x, -h, axis=a)) for a in range(3)]))
    return np.array(out)


def edge_face_counts(faces: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Number of faces sharing each edge, after merging coincident vertices."""
    _, inv = np.unique(np.round(vertices, 9), axis=0, return_inverse=True)
    f = inv.reshape(-1)[faces]
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return counts


def backus(stiffnesses, fractions):
    """Exact stiffness (Voigt notation) of a laminate normal to x3 (Backus averaging)."""
    p, n = [0, 1, 5], [2, 3, 4]
    inv = np.linalg.inv
    layers = list(zip(stiffnesses, fractions, strict=True))
    cnn = inv(sum(f * inv(c[np.ix_(n, n)]) for c, f in layers))
    a = sum(f * inv(c[np.ix_(n, n)]) @ c[np.ix_(n, p)] for c, f in layers)
    b = sum(f * c[np.ix_(p, n)] @ inv(c[np.ix_(n, n)]) for c, f in layers)
    cpp = sum(
        f * (c[np.ix_(p, p)] - c[np.ix_(p, n)] @ inv(c[np.ix_(n, n)]) @ c[np.ix_(n, p)])
        for c, f in layers
    )
    out = np.zeros((6, 6))
    out[np.ix_(n, n)] = cnn
    out[np.ix_(n, p)] = cnn @ a
    out[np.ix_(p, n)] = (cnn @ a).T
    out[np.ix_(p, p)] = cpp + b @ cnn @ a
    return out


def box_mesh(
    n=(3, 4, 5),
    lower=(0.0, 0.0, 0.0),
    upper=(1.0, 1.3, 0.8),
    phase=None,
    *,
    order=1,
    perturb=0.0,
    seed=0,
    periodic=True,
):
    """Structured tetrahedral mesh of a box (six tetrahedra per cell, matching faces).

    ``phase(centres)`` gives the phase of each element; ``perturb`` moves the interior
    nodes randomly (as a fraction of the cell size), keeping the faces plane.
    """
    from morphogen.mesh.conforming import background_mesh
    from morphogen.mesh.tetra import TetMesh

    nodes, tets, _, _ = background_mesh(n, lower, upper)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if perturb:
        h = (upper - lower) / np.asarray(n)
        interior = np.all((nodes > lower + 1e-9) & (nodes < upper - 1e-9), axis=1)
        rng = np.random.default_rng(seed)
        nodes = nodes.copy()
        nodes[interior] += perturb * h * rng.uniform(-1, 1, (int(interior.sum()), 3))
    centres = nodes[tets].mean(axis=1)
    ph = np.zeros(len(tets), dtype=int) if phase is None else np.asarray(phase(centres), int)
    mesh = TetMesh(nodes, tets, ph, lower, upper, periodic=periodic)
    return mesh.quadratic() if order == 2 else mesh


def abaqus_initial_conditions(text: str) -> dict:
    """The element sets of ``*INITIAL CONDITIONS, TYPE=SOLUTION`` of an Abaqus input: for
    each, its values and the number of values of each of its data lines."""
    block = text.split("*INITIAL CONDITIONS, TYPE=SOLUTION\n")[1].split("*")[0]
    out: dict = {}
    for line in block.splitlines():
        words = [w.strip() for w in line.split(",")]
        if words[0][:1].isalpha():
            label = words[0]
            out[label] = ([], [])
            words = words[1:]
        out[label][0].extend(float(w) for w in words)
        out[label][1].append(len(words))
    return out


def laminate_path(materials, fractions, path):
    """Exact response of a laminate of layers normal to x3 along a loading path: in each
    layer the strain and stress are uniform, the in-plane strains (11, 22, 12) are those
    of the laminate and the out-of-plane stresses (33, 23, 31) are the same in all the
    layers. Returns the macroscopic strains and stresses at the steps of the path."""
    from morphogen.homogenization.behaviours import update_points
    from morphogen.homogenization.materials import material_table
    from morphogen.homogenization.nonlinear import behaviour_table
    from morphogen.homogenization.paths import LoadingPath

    path = LoadingPath.from_spec(path)
    n = len(materials)
    mats = dict(enumerate(materials))
    phases = np.arange(n)
    table = material_table(phases, mats, "elastic")
    beh = behaviour_table(phases, mats, table)
    f = np.asarray(fractions, dtype=float)
    old, new = np.zeros((n, beh.width)), np.zeros((n, beh.width))
    sig, tan = np.zeros((n, 6)), np.zeros((n, 6, 6))
    eps = np.zeros((n, 6))
    inplane, normal = (0, 1, 5), (2, 3, 4)
    strains, stresses = [np.zeros(6)], [np.zeros(6)]
    t = path.times[0]
    for t_new in path.schedule():
        values, temp = path.at(t_new)
        for _ in range(50):
            update_points(beh.kind, table.mat, beh.pmat, beh.data, eps, temp, t_new - t, old,
                          new, sig, tan, 0, 1.0)  # fmt: skip
            res, jac = [], []
            for k in range(1, n):
                for c in inplane:  # equal in-plane strains
                    row = np.zeros(6 * n)
                    row[6 * k + c], row[c] = 1.0, -1.0
                    res.append(eps[k, c] - eps[0, c])
                    jac.append(row)
                for c in normal:  # equal out-of-plane stresses
                    row = np.zeros(6 * n)
                    row[6 * k : 6 * k + 6] = tan[k, c]
                    row[0:6] -= tan[0, c]
                    res.append(sig[k, c] - sig[0, c])
                    jac.append(row)
            for c in range(6):  # the macroscopic components
                row = np.zeros(6 * n)
                for k in range(n):
                    if path.controls[c]:
                        row[6 * k + c] = f[k]
                    else:
                        row[6 * k : 6 * k + 6] = f[k] * tan[k, c]
                jac.append(row)
                res.append((f @ (eps if path.controls[c] else sig))[c] - values[c])
            res = np.array(res)
            if np.abs(res).max() <= 1e-11 * (1.0 + np.abs(sig).max()):
                break
            eps -= np.linalg.solve(np.array(jac), res).reshape(n, 6)
        old, new = new, old
        strains.append(f @ eps)
        stresses.append(f @ sig)
        t = t_new
    return np.array(strains), np.array(stresses)
