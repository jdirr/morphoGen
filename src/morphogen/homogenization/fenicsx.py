"""The FEniCSx solver: morphoGen's finite element problems assembled and solved by
dolfinx (``solver="fenicsx"``).

``make_problem(mesh, materials, physics, bc, solver="fenicsx")`` (configurations:
``homogenization: solver: fenicsx``; the command line: ``--solver fenicsx``) returns
problems whose fields come from FEniCSx: the weak forms of linear (thermo-)elasticity
and conduction written in UFL on the same tetrahedra, with Lagrange elements of the
order of the mesh and the materials of the elements (oriented crystals included) as
cellwise constant tensors; the periodic fluctuations by the multi-point constraints of
dolfinx_mpc (the upper faces of the box slaved to the lower ones, a node held against
translations); the kinematic, static and mixed conditions on the very degrees of
freedom that morphoGen's own solver constrains; PETSc's sparse direct solver (MUMPS) or,
beyond ``DIRECT_LIMIT`` unknowns, conjugate gradients preconditioned by algebraic
multigrid (GAMG, with the rigid-body modes). The averages, energies and statistics of
the phases are then evaluated from the fields as for morphoGen's solver, so that the
two are interchangeable (``homogenize``, RVE studies, configurations, the command line)
and check each other.

FEniCSx runs on Linux (or WSL): ``conda install -c conda-forge fenics-dolfinx
dolfinx_mpc``; dolfinx 0.11 and dolfinx_mpc 0.11 were used.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from morphogen.homogenization.elements import voigt_to_tensor
from morphogen.homogenization.fe import FEElasticProblem, FEThermalProblem
from morphogen.homogenization.solvers import SolveInfo

#: Unknowns up to which PETSc's direct solver (MUMPS) is used.
DIRECT_LIMIT = 300_000


def available() -> bool:
    """Whether dolfinx, dolfinx_mpc and petsc4py can be imported."""
    try:
        _modules()
    except ImportError:
        return False
    return True


def _modules():
    try:
        import basix.ufl
        import dolfinx
        import dolfinx.fem.petsc
        import dolfinx_mpc
        import ufl
        from mpi4py import MPI
        from petsc4py import PETSc
    except ImportError as exc:  # pragma: no cover - depends on the platform
        raise ImportError(
            "solver 'fenicsx' needs FEniCSx: dolfinx, dolfinx_mpc and petsc4py "
            "(conda install -c conda-forge fenics-dolfinx dolfinx_mpc; Linux or WSL)"
        ) from exc
    return basix, dolfinx, dolfinx_mpc, ufl, MPI, PETSc


class _Fenicsx:
    """dolfinx mesh, spaces, materials and linear systems of a finite element problem
    (mixed into :class:`FEElasticProblem` and :class:`FEThermalProblem`)."""

    _fx = None

    def _fx_setup(self):
        if self._fx is not None:
            return self._fx
        basix, dolfinx, dolfinx_mpc, ufl, MPI, PETSc = _modules()
        m = self.mesh
        t0 = time.perf_counter()
        corners = m.elements[:, :4]
        used = np.unique(corners)
        index = np.full(m.n_nodes, -1, dtype=np.int64)
        index[used] = np.arange(len(used))
        coord = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
        msh = dolfinx.mesh.create_mesh(
            MPI.COMM_SELF, index[corners].astype(np.int64), ufl.Mesh(coord), m.nodes[used]
        )
        original = msh.topology.original_cell_index
        if isinstance(original, list):  # one array per cell type
            original = original[0]
        original = np.asarray(original)
        mat = self.mat[original]
        shape = (3,) if self.physics == "elastic" else ()
        space = dolfinx.fem.functionspace(msh, ("Lagrange", m.order, shape))
        # morphoGen's nodes -> degrees of freedom (block indices) of dolfinx
        dof_xyz = space.tabulate_dof_coordinates()
        dist, block = cKDTree(dof_xyz).query(m.nodes)
        size = float(np.max(m.upper - m.lower))
        if dist.max() > 1e-9 * size or len(np.unique(block)) != m.n_nodes:
            raise ValueError(
                "solver 'fenicsx': the nodes of the mesh are not those of dolfinx's elements"
            )
        d = 6 if self.physics == "elastic" else 3
        tensors = dolfinx.fem.functionspace(msh, ("DG", 0, (d, d)))
        stiff = dolfinx.fem.Function(tensors)
        cells = tensors.dofmap.list[:, 0]
        stiff.x.array.reshape(-1, d * d)[cells] = self.table.matrices[mat].reshape(-1, d * d)
        u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 2 * (m.order - 1)})
        ds = ufl.Measure("ds", domain=msh)
        n = ufl.FacetNormal(msh)
        if self.physics == "elastic":

            def voigt(w):
                e = ufl.sym(ufl.grad(w))
                return ufl.as_vector(
                    [e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[2, 0], 2 * e[0, 1]]
                )

            eig = dolfinx.fem.Function(dolfinx.fem.functionspace(msh, ("DG", 0, (6,))))
            eig.x.array.reshape(-1, 6)[eig.function_space.dofmap.list[:, 0]] = self.table.eigen[mat]
            load = dolfinx.fem.Constant(msh, np.zeros(6))  # macroscopic strain (Voigt)
            dtemp = dolfinx.fem.Constant(msh, 0.0)
            traction = dolfinx.fem.Constant(msh, np.zeros((3, 3)))  # macroscopic stress
            a = ufl.inner(ufl.dot(stiff, voigt(u)), voigt(v)) * dx
            rhs = -ufl.inner(ufl.dot(stiff, load - dtemp * eig), voigt(v)) * dx
            rhs += ufl.inner(ufl.dot(traction, n), v) * ds
            constants = {"load": load, "dt": dtemp, "boundary": traction}
        else:
            load = dolfinx.fem.Constant(msh, np.zeros(3))  # macroscopic gradient
            flux = dolfinx.fem.Constant(msh, np.zeros(3))  # macroscopic flux
            a = ufl.inner(ufl.dot(stiff, ufl.grad(u)), ufl.grad(v)) * dx
            rhs = -ufl.inner(ufl.dot(stiff, load), ufl.grad(v)) * dx - ufl.inner(flux, n) * v * ds
            constants = {"load": load, "boundary": flux}
        self._fx = {
            "modules": (dolfinx, dolfinx_mpc, PETSc),
            "mesh": msh,
            "space": space,
            "block": block,
            "a": a,
            "rhs": rhs,
            "constants": constants,
            "systems": {},
            "setup_s": time.perf_counter() - t0,
        }
        return self._fx

    def _fx_bcs(self, key):
        """Homogeneous Dirichlet conditions on the degrees of freedom that morphoGen's
        solver constrains (``_constraints``), or the node held in periodic problems."""
        fx = self._fx
        dolfinx = fx["modules"][0]
        space, block = fx["space"], fx["block"]
        nd = 3 if self.physics == "elastic" else 1
        if key[0] == "periodic":
            # a node inside the box: holding a master of the periodic constraints breaks
            # their condensation (dolfinx_mpc 0.11)
            inside = np.ones(self.mesh.n_nodes, dtype=bool)
            inside[self.mesh.boundary_nodes()] = False
            if not inside.any():
                raise ValueError("solver 'fenicsx': periodic meshes need a node inside the box")
            centre = 0.5 * (self.mesh.lower + self.mesh.upper)
            dist = np.linalg.norm(self.mesh.nodes - centre, axis=1)
            con = np.zeros((self.mesh.n_nodes, nd), dtype=bool)
            con[int(np.argmin(np.where(inside, dist, np.inf)))] = True
        else:
            con = np.asarray(self._constraints(key)).reshape(self.mesh.n_nodes, nd)
        if nd == 1:
            dofs = np.sort(block[con[:, 0]]).astype(np.int32)
            return [dolfinx.fem.dirichletbc(dolfinx.fem.Function(space), dofs)]
        bcs = []
        for c in range(3):
            if not con[:, c].any():
                continue
            sub, parent = space.sub(c).collapse()
            inverse = np.full(3 * len(block), -1, dtype=np.int64)
            inverse[np.asarray(parent)] = np.arange(len(parent))
            p = 3 * block[con[:, c]] + c
            order = np.argsort(p)
            dofs = [p[order].astype(np.int32), inverse[p[order]].astype(np.int32)]
            bcs.append(dolfinx.fem.dirichletbc(dolfinx.fem.Function(sub), dofs, space.sub(c)))
        return bcs

    def _fx_system(self, key):
        fx = self._fx_setup()
        if key in fx["systems"]:
            return fx["systems"][key]
        dolfinx, dolfinx_mpc, PETSc = fx["modules"]
        t0 = time.perf_counter()
        space = fx["space"]
        bcs = self._fx_bcs(key)
        a_form = dolfinx.fem.form(fx["a"])
        l_form = dolfinx.fem.form(fx["rhs"])
        mpc = None
        if key[0] == "periodic":
            lo, hi = self.mesh.lower, self.mesh.upper
            tol = 1e-8 * float(np.max(hi - lo))

            def upper(x):
                return np.logical_or.reduce([np.isclose(x[k], hi[k], atol=tol) for k in range(3)])

            def relation(x):
                out = x.copy()
                for k in range(3):
                    top = np.isclose(x[k], hi[k], atol=tol)
                    out[k, top] = x[k, top] - (hi[k] - lo[k])
                return out

            mpc = dolfinx_mpc.MultiPointConstraint(space)
            mpc.create_periodic_constraint_geometrical(space, upper, relation, bcs, tol=tol)
            mpc.finalize()
            matrix = dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=bcs)
        else:
            matrix = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
            matrix.assemble()
        n = matrix.getSize()[0]
        ksp = PETSc.KSP().create(fx["mesh"].comm)
        ksp.setOperators(matrix)
        if n <= DIRECT_LIMIT:
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            method = "fenicsx-mumps"
        else:  # pragma: no cover - large problems
            ksp.setType("cg")
            ksp.getPC().setType("gamg")
            ksp.setTolerances(rtol=self.tol, max_it=self.maxiter or 10_000)
            method = "fenicsx-cg-gamg"
        ksp.setFromOptions()
        system = (a_form, l_form, bcs, mpc, matrix, ksp, method, n, time.perf_counter() - t0)
        fx["systems"][key] = system
        return system

    def _fx_solve(self, key) -> tuple[np.ndarray, SolveInfo]:
        """Fluctuation (zero where constrained) in morphoGen's node order."""
        a_form, l_form, bcs, mpc, _, ksp, method, n, setup = self._fx_system(key)
        fx = self._fx
        dolfinx, dolfinx_mpc, PETSc = fx["modules"]
        t0 = time.perf_counter()
        if mpc is None:
            b = dolfinx.fem.petsc.assemble_vector(l_form)
            dolfinx.fem.petsc.apply_lifting(b, [a_form], bcs=[bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b, bcs)
            w = dolfinx.fem.Function(fx["space"])
        else:
            b = dolfinx_mpc.assemble_vector(l_form, mpc)
            dolfinx_mpc.apply_lifting(b, [a_form], [bcs], mpc)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b, bcs)
            w = dolfinx.fem.Function(mpc.function_space)
        ksp.solve(b, w.x.petsc_vec)
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(
                f"solver 'fenicsx': PETSc did not converge (reason {ksp.getConvergedReason()})"
            )
        w.x.scatter_forward()
        if mpc is not None:
            mpc.backsubstitution(w)
        values = np.asarray(w.x.array)
        if self.physics == "elastic":
            field = values.reshape(-1, 3)[fx["block"]].ravel()
        else:
            field = values[fx["block"]]
        residual = float(ksp.getResidualNorm())
        b.destroy()
        info = SolveInfo(
            method, int(n), int(ksp.getIterationNumber()), residual, fx["setup_s"] + setup, 0.0
        )
        info.solve_s = time.perf_counter() - t0
        return field, info


class FenicsxElasticProblem(_Fenicsx, FEElasticProblem):
    """Linear (thermo-)elasticity of a tetrahedral mesh solved by FEniCSx."""

    def _field(self, x: np.ndarray, dt: float):
        fx = self._fx_setup()
        c = fx["constants"]
        m = self.mesh
        if self.native == "strain":
            e = voigt_to_tensor(x, strain=True)
            u_aff = (m.nodes @ e.T).ravel()
            c["load"].value[:] = x
            c["boundary"].value[:] = 0.0
        else:
            u_aff = np.zeros(3 * m.n_nodes)
            c["load"].value[:] = 0.0
            c["boundary"].value[:] = voigt_to_tensor(x, strain=False)
        c["dt"].value = float(dt)
        w, info = self._fx_solve(self._key(x))
        return u_aff + w, info


class FenicsxThermalProblem(_Fenicsx, FEThermalProblem):
    """Steady linear conduction of a tetrahedral mesh solved by FEniCSx."""

    def _field(self, x: np.ndarray, dt: float = 0.0):
        fx = self._fx_setup()
        c = fx["constants"]
        m = self.mesh
        if self.native == "gradient":
            t_aff = m.nodes @ x
            c["load"].value[:] = x
            c["boundary"].value[:] = 0.0
        else:
            t_aff = np.zeros(m.n_nodes)
            c["load"].value[:] = 0.0
            c["boundary"].value[:] = x
        w, info = self._fx_solve(self._key(x))
        return t_aff + w, info
