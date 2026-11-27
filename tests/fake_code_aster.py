"""A stand-in for an installation of Code_Aster, for the tests of morphogen.io.code_aster:
it reads the export file of a study, the mesh in Code_Aster's format by a parser of its
own, and executes the command file with Python stand-ins of the commands it uses, which
record their arguments and do what Code_Aster does with them on 4-node tetrahedra:
LIRE_MAILLAGE, the materials (ELAS, ELAS_ORTH, THER, THER_ORTH), their frames
(AFFE_CARA_ELEM, MASSIF: ANGL_EULER turned into the matrix of Code_Aster's own routine
eulnau, ANGL_REP into that of matrot), the loadings (DDL_IMPO, TEMP_IMPO, FORMULE of the
coordinates, LIAISON_GROUP: each node of GROUP_NO_1 translated by TRAN paired with the
nearest node of GROUP_NO_2, as Code_Aster's pacoap pairs them), MECA_STATIQUE and
THER_LINEAIRE (the equations of the linear tetrahedra and the relations, by Lagrange
multipliers), CALC_CHAMP, POST_ELEM (INTEGRALE) and IMPR_TABLE (the format TABLEAU in the
files of the units of the export). It tests the files that morphoGen writes and their
reading back, not Code_Aster."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

#: the stress components of SIEF_ELGA and their tensor indices
SIEF = {
    "SIXX": (0, 0),
    "SIYY": (1, 1),
    "SIZZ": (2, 2),
    "SIXY": (0, 1),
    "SIXZ": (0, 2),
    "SIYZ": (1, 2),
}
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))


class Concept(dict):
    """A result of a command: its keywords, and what the command made of them."""

    def __init__(self, command: str, keywords: dict):
        super().__init__(keywords)
        self.command = command


def read_export(path) -> dict[int, Path]:
    """The files of the logical units of an export file."""
    path = Path(path)
    units = {}
    for line in path.read_text().splitlines():
        words = line.split()
        if words and words[0] == "F":
            units[int(words[4])] = path.parent / words[2]
    return units


def read_mail(path) -> dict:
    """Nodes, cells and groups of a mesh in Code_Aster's format (comments after %, words
    in blocks closed by FINSF, names of groups after NOM=)."""
    names, coords, cells, groups_ma, groups_no = {}, [], {}, {}, {}
    block, key, header = None, None, None
    for raw in Path(path).read_text().splitlines():
        line = raw.split("%", 1)[0]
        assert len(raw) <= 80, raw
        words = line.split()
        if not words:
            continue
        if key is None:
            key, header, block = words[0], words[1:], []
            if key == "FIN":
                break
            continue
        if words[0] == "FINSF":
            if key == "COOR_3D":
                for k in range(0, len(block), 4):
                    names[block[k]] = len(coords)
                    coords.append([float(v) for v in block[k + 1 : k + 4]])
            elif key in ("TETRA4", "TETRA10", "TRIA3", "TRIA6"):
                n = {"TETRA4": 4, "TETRA10": 10, "TRIA3": 3, "TRIA6": 6}[key]
                for k in range(0, len(block), n + 1):
                    cells[block[k]] = (key, [names[v] for v in block[k + 1 : k + 1 + n]])
            elif key in ("GROUP_MA", "GROUP_NO"):
                name = header[0].split("=")[1]
                if key == "GROUP_MA":
                    groups_ma[name] = list(block)
                else:
                    groups_no[name] = [names[v] for v in block]
            key = None
            continue
        block.extend(words)
    for name in list(names) + list(cells):
        assert len(name) <= 8, name
    return {
        "nodes": np.array(coords),
        "cells": cells,
        "groups_ma": groups_ma,
        "groups_no": groups_no,
    }


def euler_frame(angles) -> np.ndarray:
    """The local frame of ANGL_EULER (psi, theta, phi in degrees): the matrix of
    Code_Aster's eulnau, whose columns are the local axes in the global frame."""
    psi, theta, phi = np.radians(np.asarray(angles, dtype=float))
    cps, sps, cth, sth, cph, sph = (
        math.cos(psi),
        math.sin(psi),
        math.cos(theta),
        math.sin(theta),
        math.cos(phi),
        math.sin(phi),
    )
    pe = np.empty((3, 3))
    pe[0, 0] = cph * cps - sph * cth * sps
    pe[1, 0] = cph * sps + sph * cth * cps
    pe[2, 0] = sph * sth
    pe[0, 1] = -sph * cps - cph * cth * sps
    pe[1, 1] = -sph * sps + cph * cth * cps
    pe[2, 1] = cph * sth
    pe[0, 2] = sth * sps
    pe[1, 2] = -sth * cps
    pe[2, 2] = cth
    return pe


def nautical_frame(angles) -> np.ndarray:
    """The local frame of ANGL_REP (alpha, beta, gamma in degrees): the transpose of the
    matrix of Code_Aster's matrot (its rows are the local axes)."""
    a, b, g = np.radians(np.asarray(angles, dtype=float))
    ca, sa, cb, sb, cg, sg = (
        math.cos(a),
        math.sin(a),
        math.cos(b),
        math.sin(b),
        math.cos(g),
        math.sin(g),
    )
    pgl = np.array(
        [
            [cb * ca, cb * sa, -sb],
            [sg * sb * ca - cg * sa, cg * ca + sg * sb * sa, sg * cb],
            [sg * sa + cg * sb * ca, cg * sb * sa - ca * sg, cg * cb],
        ]
    )
    return pgl.T


def _stiffness(keywords: dict, frame: np.ndarray) -> np.ndarray:
    """The 6 x 6 stiffness (Voigt 11 22 33 23 31 12, engineering shears) in the global
    frame of the ELAS or ELAS_ORTH of a material."""
    if "ELAS" in keywords:
        e, nu = keywords["ELAS"]["E"], keywords["ELAS"]["NU"]
        lam, mu = e * nu / ((1 + nu) * (1 - 2 * nu)), e / (2 * (1 + nu))
        d = np.zeros((6, 6))
        d[:3, :3] = lam
        d[np.arange(3), np.arange(3)] += 2 * mu
        d[np.arange(3, 6), np.arange(3, 6)] = mu
        return d
    o = keywords["ELAS_ORTH"]
    s = np.zeros((6, 6))
    s[0, 0], s[1, 1], s[2, 2] = 1 / o["E_L"], 1 / o["E_T"], 1 / o["E_N"]
    s[0, 1] = s[1, 0] = -o["NU_LT"] / o["E_L"]
    s[0, 2] = s[2, 0] = -o["NU_LN"] / o["E_L"]
    s[1, 2] = s[2, 1] = -o["NU_TN"] / o["E_T"]
    s[3, 3], s[4, 4], s[5, 5] = 1 / o["G_TN"], 1 / o["G_LN"], 1 / o["G_LT"]
    local = np.linalg.inv(s)
    c = np.zeros((3, 3, 3, 3))
    for i, (a, b) in enumerate(VOIGT):
        for j, (p, q) in enumerate(VOIGT):
            for x, y in {(a, b), (b, a)}:
                for z, w in {(p, q), (q, p)}:
                    c[x, y, z, w] = local[i, j]
    r = frame
    c = np.einsum("ip,jq,kr,ls,pqrs->ijkl", r, r, r, r, c)
    return np.array([[c[a, b, p, q] for p, q in VOIGT] for a, b in VOIGT])


def _conductivity(keywords: dict, frame: np.ndarray) -> np.ndarray:
    if "THER" in keywords:
        return keywords["THER"]["LAMBDA"] * np.eye(3)
    o = keywords["THER_ORTH"]
    local = np.diag([o["LAMBDA_L"], o["LAMBDA_T"], o["LAMBDA_N"]])
    return frame @ local @ frame.T


def _gradients(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The gradients (n, 4, 3) of the shape functions of linear tetrahedra and their
    volumes."""
    p = nodes[tets]
    jac = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=1)
    inv = np.linalg.inv(jac)  # rows: d xi / d x
    grad = np.zeros((len(tets), 4, 3))
    grad[:, 1:] = np.transpose(inv, (0, 2, 1))
    grad[:, 0] = -grad[:, 1:].sum(axis=1)
    return grad, np.abs(np.linalg.det(jac)) / 6.0


def _strain_matrix(grad: np.ndarray) -> np.ndarray:
    """B (6 x 12): engineering strains (11 22 33 23 31 12) of the nodal displacements."""
    b = np.zeros((6, 12))
    for a in range(4):
        gx, gy, gz = grad[a]
        b[0, 3 * a], b[1, 3 * a + 1], b[2, 3 * a + 2] = gx, gy, gz
        b[3, 3 * a + 1], b[3, 3 * a + 2] = gz, gy
        b[4, 3 * a], b[4, 3 * a + 2] = gz, gx
        b[5, 3 * a], b[5, 3 * a + 1] = gy, gx
    return b


class FakeAster:
    """The commands of a study, run on its files (:meth:`run`)."""

    def __init__(self, export):
        self.export = Path(export)
        self.units = read_export(self.export)
        self.calls: list[tuple[str, Concept]] = []

    # -- the commands ------------------------------------------------------------------

    def _record(self, name, keywords) -> Concept:
        out = Concept(name, keywords)
        self.calls.append((name, out))
        return out

    def namespace(self) -> dict:
        ns = {"_F": lambda **k: dict(k)}
        for name in (
            "DEBUT",
            "FIN",
            "DEFI_MATERIAU",
            "AFFE_MODELE",
            "AFFE_CHAR_MECA",
            "AFFE_CHAR_MECA_F",
            "AFFE_CHAR_THER",
            "AFFE_CHAR_THER_F",
            "IMPR_RESU",
            "DEFI_LIST_REEL",
        ):
            ns[name] = lambda _n=name, **k: self._record(_n, k)
        ns.update(
            LIRE_MAILLAGE=self.lire_maillage,
            AFFE_MATERIAU=self.affe_materiau,
            AFFE_CARA_ELEM=self.affe_cara_elem,
            FORMULE=self.formule,
            MECA_STATIQUE=self.meca_statique,
            THER_LINEAIRE=self.ther_lineaire,
            CALC_CHAMP=self.calc_champ,
            POST_ELEM=self.post_elem,
            IMPR_TABLE=self.impr_table,
        )
        return ns

    def run(self) -> list[tuple[str, Concept]]:
        comm = self.units[1]
        exec(compile(comm.read_text(), str(comm), "exec"), self.namespace())
        return self.calls

    def lire_maillage(self, **k) -> Concept:
        out = self._record("LIRE_MAILLAGE", k)
        assert k["FORMAT"] == "ASTER", "the stand-in reads Code_Aster's format"
        out.mesh = read_mail(self.units[k["UNITE"]])
        return out

    def affe_materiau(self, **k) -> Concept:
        out = self._record("AFFE_MATERIAU", k)
        affe = k["AFFE"] if isinstance(k["AFFE"], (list, tuple)) else [k["AFFE"]]
        out.by_cell = {}
        mesh = k["MAILLAGE"].mesh
        for item in affe:
            groups = item["GROUP_MA"]
            for g in [groups] if isinstance(groups, str) else groups:
                for cell in mesh["groups_ma"][g]:
                    out.by_cell[cell] = item["MATER"]
        return out

    def affe_cara_elem(self, **k) -> Concept:
        out = self._record("AFFE_CARA_ELEM", k)
        massif = k["MASSIF"] if isinstance(k["MASSIF"], (list, tuple)) else [k["MASSIF"]]
        mesh = self._mesh()
        out.frames = {}
        for item in massif:
            if "ANGL_EULER" in item:
                frame = euler_frame(item["ANGL_EULER"])
            else:
                frame = nautical_frame(item["ANGL_REP"])
            for cell in mesh["groups_ma"][item["GROUP_MA"]]:
                out.frames[cell] = frame
        return out

    def formule(self, **k):
        self.calls.append(("FORMULE", Concept("FORMULE", k)))
        code = compile(k["VALE"], "<formule>", "eval")
        return lambda X, Y, Z: eval(code, {"__builtins__": {}}, {"X": X, "Y": Y, "Z": Z})

    def _mesh(self) -> dict:
        return next(c for n, c in self.calls if n == "LIRE_MAILLAGE").mesh

    # -- the solutions -------------------------------------------------------------------

    def _system(self, model, field: str):
        """The cells of the model (the 3D cells of its groups) and their element data."""
        mesh = self._mesh()
        groups = model["AFFE"]["GROUP_MA"]
        groups = [groups] if isinstance(groups, str) else groups
        cells = [c for g in groups for c in mesh["groups_ma"][g]]
        assert all(mesh["cells"][c][0] == "TETRA4" for c in cells), "4-node tetrahedra"
        tets = np.array([mesh["cells"][c][1] for c in cells])
        return mesh, cells, tets

    def _relations(self, mesh, loads, dofs, n_dof):
        """The rows of the imposed values and relations of the loadings: (coefficients
        by (node, component), value)."""
        rows = []
        index = {d: i for i, d in enumerate(dofs)}
        x = mesh["nodes"]
        for load in loads:
            for key in ("DDL_IMPO", "TEMP_IMPO"):
                items = load.get(key)
                if items is None:
                    continue
                for item in [items] if isinstance(items, dict) else items:
                    for node in mesh["groups_no"][item["GROUP_NO"]]:
                        for dof in dofs:
                            name = "TEMP" if dof == "TEMP" else dof
                            if name in item:
                                v = item[name]
                                v = v(*x[node]) if callable(v) else v
                                rows.append(({(node, index[dof]): 1.0}, float(v)))
            items = load.get("LIAISON_GROUP")
            if items is None:
                continue
            for item in [items] if isinstance(items, dict) else items:
                one = mesh["groups_no"][item["GROUP_NO_1"]]
                two = mesh["groups_no"][item["GROUP_NO_2"]]
                assert len(one) == len(two)
                moved = x[one] + np.asarray(item["TRAN"], dtype=float)
                dist = np.linalg.norm(moved[:, None, :] - x[two][None, :, :], axis=2)
                pair = np.asarray(two)[dist.argmin(axis=1)]
                assert dist.min(axis=1).max() < 1e-9, "a node without its image"
                assert len(set(pair.tolist())) == len(one), "an image of two nodes"
                d1 = [item["DDL_1"]] if isinstance(item["DDL_1"], str) else item["DDL_1"]
                d2 = [item["DDL_2"]] if isinstance(item["DDL_2"], str) else item["DDL_2"]
                c1 = np.atleast_1d(item["COEF_MULT_1"])
                c2 = np.atleast_1d(item["COEF_MULT_2"])
                for a, b in zip(one, pair, strict=True):
                    coef = {}
                    for d, c in zip(d1, c1, strict=True):
                        coef[(a, index[d])] = coef.get((a, index[d]), 0.0) + c
                    for d, c in zip(d2, c2, strict=True):
                        coef[(b, index[d])] = coef.get((b, index[d]), 0.0) + c
                    rows.append((coef, float(item["COEF_IMPO"])))
        return rows

    def _solve(self, k_matrix, rows, n_dof, nd):
        """The solution of ``K u = 0`` under the relations, by Lagrange multipliers."""
        r, c, v, rhs = [], [], [], []
        for i, (coef, value) in enumerate(rows):
            for (node, comp), w in coef.items():
                r.append(i)
                c.append(nd * node + comp)
                v.append(w)
            rhs.append(value)
        cmat = sparse.csr_matrix((v, (r, c)), shape=(len(rows), n_dof))
        kkt = sparse.bmat([[k_matrix, cmat.T], [cmat, None]]).tocsc()
        sol = splinalg.spsolve(kkt, np.concatenate([np.zeros(n_dof), rhs]))
        return sol[:n_dof]

    def meca_statique(self, **k) -> Concept:
        out = self._record("MECA_STATIQUE", k)
        mesh, cells, tets = self._system(k["MODELE"], "SIEF_ELGA")
        frames = k["CARA_ELEM"].frames if "CARA_ELEM" in k else {}
        mats = k["CHAM_MATER"].by_cell
        grad, vol = _gradients(mesh["nodes"], tets)
        n = len(mesh["nodes"])
        rows, cols, vals, d_cells, b_cells = [], [], [], [], []
        for e, cell in enumerate(cells):
            d = _stiffness(mats[cell], frames.get(cell, np.eye(3)))
            b = _strain_matrix(grad[e])
            ke = vol[e] * b.T @ d @ b
            dof = (3 * tets[e][:, None] + np.arange(3)).ravel()
            rows.append(np.repeat(dof, 12))
            cols.append(np.tile(dof, 12))
            vals.append(ke.ravel())
            d_cells.append(d)
            b_cells.append(b)
        kmat = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(3 * n, 3 * n),
        )
        excit = k["EXCIT"] if isinstance(k["EXCIT"], (list, tuple)) else [k["EXCIT"]]
        loads = [item["CHARGE"] for item in excit]
        u = self._solve(kmat, self._relations(mesh, loads, ("DX", "DY", "DZ"), 3 * n), 3 * n, 3)
        stress = []
        for e in range(len(cells)):
            eps = b_cells[e] @ u[(3 * tets[e][:, None] + np.arange(3)).ravel()]
            stress.append(d_cells[e] @ eps)
        out.fields = {
            "SIEF_ELGA": dict(
                zip(
                    ("SIXX", "SIYY", "SIZZ", "SIYZ", "SIXZ", "SIXY"),
                    np.array(stress).T,
                    strict=True,
                )
            )
        }
        out.volumes = vol
        return out

    def ther_lineaire(self, **k) -> Concept:
        out = self._record("THER_LINEAIRE", k)
        assert k.get("TYPE_CALCUL") == "STAT"
        assert k["INCREMENT"]["LIST_INST"].command == "DEFI_LIST_REEL"
        mesh, cells, tets = self._system(k["MODELE"], "FLUX_ELGA")
        frames = k["CARA_ELEM"].frames if "CARA_ELEM" in k else {}
        mats = k["CHAM_MATER"].by_cell
        grad, vol = _gradients(mesh["nodes"], tets)
        n = len(mesh["nodes"])
        rows, cols, vals, lam = [], [], [], []
        for e, cell in enumerate(cells):
            k_e = _conductivity(mats[cell], frames.get(cell, np.eye(3)))
            ke = vol[e] * grad[e] @ k_e @ grad[e].T
            rows.append(np.repeat(tets[e], 4))
            cols.append(np.tile(tets[e], 4))
            vals.append(ke.ravel())
            lam.append(k_e)
        kmat = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        )
        excit = k["EXCIT"] if isinstance(k["EXCIT"], (list, tuple)) else [k["EXCIT"]]
        loads = [item["CHARGE"] for item in excit]
        t = self._solve(kmat, self._relations(mesh, loads, ("TEMP",), n), n, 1)
        out.temperature = t
        out.flux = np.array([-lam[e] @ (grad[e].T @ t[tets[e]]) for e in range(len(cells))])
        out.volumes = vol
        return out

    def calc_champ(self, **k) -> Concept:
        self._record("CALC_CHAMP", k)
        resu = k["RESULTAT"]
        assert k["reuse"] is resu
        options = k["THERMIQUE"]
        assert "FLUX_ELGA" in (options if isinstance(options, (list, tuple)) else [options])
        resu.fields = {"FLUX_ELGA": dict(zip(("FLUX", "FLUY", "FLUZ"), resu.flux.T, strict=True))}
        return resu

    def post_elem(self, **k) -> Concept:
        out = self._record("POST_ELEM", k)
        inte = k["INTEGRALE"]
        assert inte["TOUT"] == "OUI" and inte["TYPE_MAILLE"] == "3D"
        resu = k["RESULTAT"]
        values = resu.fields[inte["NOM_CHAM"]]
        vol = resu.volumes
        out.row = {
            "NOM_CHAM": inte["NOM_CHAM"],
            "NUME_ORDRE": 1,
            "INST": 0.0,
            "VOL": float(vol.sum()),
            "TOUT": "00000001",
        }
        for c in inte["NOM_CMP"]:
            out.row[f"INTE_{c}"] = float(np.sum(values[c] * vol))
        for c in inte["NOM_CMP"]:
            out.row[f"MOYE_{c}"] = out.row[f"INTE_{c}"] / out.row["VOL"]
        return out

    def impr_table(self, **k) -> Concept:
        out = self._record("IMPR_TABLE", k)
        assert k["FORMAT"] == "TABLEAU"
        row = k["TABLE"].row
        cells = [f"{v:13.6E}" if isinstance(v, float) else str(v) for v in row.values()]
        text = (
            "#\n#ASTER stand-in: table of POST_ELEM\n#\n"
            + "-" * 60
            + "\n"
            + " ".join(row)
            + "\n"
            + " ".join(cells)
            + "\n"
        )
        self.units[k["UNITE"]].write_text(text)
        return out


def run(export) -> FakeAster:
    """The study of an export file run by the stand-in."""
    fake = FakeAster(export)
    fake.run()
    return fake
