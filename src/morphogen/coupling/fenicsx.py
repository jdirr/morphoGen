"""Diffusion and swelling of tetrahedral meshes with FEniCSx (dolfinx): moisture or a
solvent enters a sample through its exposed faces, and the sample swells and bends as
it does (Mou, Dirrenberger & Le Duigou 2025 model pine cone scales and 4D-printed
hygromorphs this way with FEniCS).

The activity ``u`` of the species is continuous across interfaces, each phase holds
``c = S u`` and lets it through with the permeability ``D S`` (as in
:mod:`morphogen.coupling.diffusion`): ``S du/dt = div(D S grad u)``, Lagrange elements of
the order of the mesh, the exposed faces held at the activity (a number or a function of
time), the other faces impermeable, implicit Euler steps. At each output time the
sample, free of tractions, deforms elastically under the swelling strains ``beta c I``
(small strains; isotropic phases), held only by 3-2-1 supports at three corners: its
displacement field, and the curvature of a strip fitted to the deflection of its
mid-plane (:func:`strip_curvature`). FEniCSx runs on Linux (or WSL).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

from morphogen.homogenization.fenicsx import _modules
from morphogen.mesh.tetra import TetMesh

FACES = ("x-", "x+", "y-", "y+", "z-", "z+")


@dataclasses.dataclass
class SwellingResult:
    """At the output ``times``: the mean concentration over the meshed volume
    (``uptake``; ``saturation`` at the activity of the last time), the concentration of
    the nodes (that of the phase of an element around each node, ``S u``) and their
    displacement (morphoGen's order of the nodes)."""

    times: np.ndarray
    uptake: np.ndarray
    saturation: float
    concentration: list
    displacement: list
    mesh: TetMesh = dataclasses.field(repr=False)

    @property
    def relative_uptake(self) -> np.ndarray:
        return self.uptake / self.saturation


def _dolfinx_mesh(mesh: TetMesh):
    basix, dolfinx, _, ufl, MPI, _ = _modules()
    corners = mesh.elements[:, :4]
    used = np.unique(corners)
    index = np.full(mesh.n_nodes, -1, dtype=np.int64)
    index[used] = np.arange(len(used))
    coord = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    msh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, index[corners], ufl.Mesh(coord), mesh.nodes[used])
    original = msh.topology.original_cell_index
    if isinstance(original, list):
        original = original[0]
    return msh, np.asarray(original)


def _node_map(space, mesh: TetMesh) -> np.ndarray:
    dist, block = cKDTree(space.tabulate_dof_coordinates()).query(mesh.nodes)
    if dist.max() > 1e-9 * float(np.max(mesh.upper - mesh.lower)):
        raise ValueError("the nodes of the mesh are not those of dolfinx's elements")
    return block


def _cellwise(dolfinx, msh, values):
    space = dolfinx.fem.functionspace(msh, ("DG", 0))
    f = dolfinx.fem.Function(space)
    f.x.array[space.dofmap.list[:, 0]] = values
    return f


def swelling(
    mesh: TetMesh,
    phases: dict,
    times,
    *,
    exposed=("z-",),
    activity: float | Callable[[float], float] = 1.0,
    initial: float = 0.0,
    steps: int = 20,
    kinematics: str = "small",
) -> SwellingResult:
    """Diffusion into ``mesh`` from its ``exposed`` faces and the free swelling of the
    sample at the output ``times``. ``phases``: phase -> mapping with ``diffusivity``,
    ``solubility`` (default 1), ``swelling`` (the strain per unit concentration,
    default 0), ``young`` and ``poisson``. Each output interval takes ``steps`` implicit
    Euler steps (the first one geometrically growing, for the sqrt(t) start).
    ``kinematics``: ``"small"`` (linear elasticity under the swelling strains) or
    ``"finite"`` (large rotations and strains: ``F = F_e F_s`` with the swelling stretch
    ``F_s = (1 + beta c) I``, Saint Venant-Kirchhoff's law of ``F_e`` per unit volume of
    the swollen state, so that ``P = (1 + beta c) F S_e``; Newton's method, each output
    reached by continuation of the concentration field, halved when it fails)."""
    if kinematics not in ("small", "finite"):
        raise ValueError('kinematics: "small" or "finite"')
    _, dolfinx, _, ufl, _, PETSc = _modules()

    for face in exposed:
        if face not in FACES:
            raise ValueError(f"exposed: faces among {', '.join(FACES)}")
    msh, original = _dolfinx_mesh(mesh)
    phase = np.asarray(mesh.phase)[original]
    missing = sorted(set(np.unique(phase).tolist()) - set(phases))
    if missing:
        raise ValueError(f"phases: no properties for {missing}")

    def prop(name, default=None):
        out = []
        for p in phase:
            v = phases[int(p)].get(name, default)
            if v is None:
                raise ValueError(f"phases: give '{name}' for phase {int(p)}")
            out.append(float(v))
        return _cellwise(dolfinx, msh, np.array(out))

    sol = prop("solubility", 1.0)
    perm = _cellwise(
        dolfinx, msh, np.asarray(prop("diffusivity").x.array) * np.asarray(sol.x.array)
    )
    beta = prop("swelling", 0.0)
    young, poisson = prop("young"), prop("poisson")
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young / (2 * (1 + poisson))
    order = mesh.order
    v_space = dolfinx.fem.functionspace(msh, ("Lagrange", order))
    w_space = dolfinx.fem.functionspace(msh, ("Lagrange", order, (3,)))
    node_s = _node_map(v_space, mesh)
    node_w = _node_map(w_space, mesh)
    # the exposed faces
    lo, hi, tol = mesh.lower, mesh.upper, mesh.tolerance()

    def on_faces(x):
        out = np.zeros(x.shape[1], dtype=bool)
        for face in exposed:
            axis, side = "xyz".index(face[0]), face[1]
            out |= np.abs(x[axis] - (lo[axis] if side == "-" else hi[axis])) <= tol
        return out

    facets = dolfinx.mesh.locate_entities_boundary(msh, 2, on_faces)
    exposed_dofs = dolfinx.fem.locate_dofs_topological(v_space, 2, facets)
    level = dolfinx.fem.Constant(msh, 0.0)
    bc = dolfinx.fem.dirichletbc(level, exposed_dofs, v_space)
    # diffusion: (S/dt) u + div(-D S grad u) = (S/dt) u_n
    u, v = ufl.TrialFunction(v_space), ufl.TestFunction(v_space)
    u_n = dolfinx.fem.Function(v_space)
    u_n.x.array[:] = initial
    inv_dt = dolfinx.fem.Constant(msh, 1.0)
    a_d = dolfinx.fem.form(
        (inv_dt * sol * u * v + perm * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    )
    l_d = dolfinx.fem.form(inv_dt * sol * u_n * v * ufl.dx)
    u_h = dolfinx.fem.Function(v_space)
    # mechanics: free swelling, 3-2-1 supports
    d, e = ufl.TrialFunction(w_space), ufl.TestFunction(w_space)

    def eps(z):
        return ufl.sym(ufl.grad(z))

    def sigma(z):
        return 2.0 * mu * eps(z) + lam * ufl.tr(eps(z)) * ufl.Identity(3)

    a_m = dolfinx.fem.form(ufl.inner(sigma(d), eps(e)) * ufl.dx)
    l_m = dolfinx.fem.form((3.0 * lam + 2.0 * mu) * beta * sol * u_h * ufl.div(e) * ufl.dx)
    tree = cKDTree(mesh.nodes)
    corners = [lo, np.array([hi[0], lo[1], lo[2]]), np.array([lo[0], hi[1], lo[2]])]
    supports = [int(tree.query(c)[1]) for c in corners]
    held = {0: [supports[0]], 1: [supports[0], supports[1]], 2: supports}
    bcs_m = []
    for c in range(3):
        sub, parent = w_space.sub(c).collapse()
        inverse = np.full(3 * len(node_w), -1, dtype=np.int64)
        inverse[np.asarray(parent)] = np.arange(len(parent))
        dofs = np.array(sorted({3 * int(node_w[n]) + c for n in held[c]}), dtype=np.int32)
        bcs_m.append(
            dolfinx.fem.dirichletbc(
                dolfinx.fem.Function(sub), [dofs, inverse[dofs].astype(np.int32)], w_space.sub(c)
            )
        )
    a_mech = dolfinx.fem.petsc.assemble_matrix(a_m, bcs=bcs_m)
    a_mech.assemble()
    ksp_m = PETSc.KSP().create(msh.comm)
    ksp_m.setOperators(a_mech)
    ksp_m.setType("preonly")
    ksp_m.getPC().setType("lu")
    ksp_m.getPC().setFactorSolverType("mumps")
    disp = dolfinx.fem.Function(w_space)
    if kinematics == "finite":
        u_load = dolfinx.fem.Function(v_space)  # the activity along the continuation
        identity = ufl.Identity(3)
        def_grad = identity + ufl.grad(disp)
        stretch = 1.0 + beta * sol * u_load
        fe = def_grad / stretch
        ee = 0.5 * (fe.T * fe - identity)
        se = lam * ufl.tr(ee) * identity + 2.0 * mu * ee
        residual = ufl.inner(stretch * def_grad * se, ufl.grad(e)) * ufl.dx
        newton = dolfinx.fem.petsc.NonlinearProblem(
            residual,
            disp,
            petsc_options_prefix="morphogen_swelling_",
            bcs=bcs_m,
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-10,
                "snes_atol": 1e-12,
                "snes_max_it": 15,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        u_prev = np.full(u_h.x.array.shape, float(initial))

        def finite_solve():
            target = np.asarray(u_h.x.array).copy()
            done, step = 0.0, 0.25
            saved = disp.x.array.copy()
            while done < 1.0 - 1e-12:
                s = min(1.0, done + step)
                u_load.x.array[:] = u_prev + s * (target - u_prev)
                newton.solve()
                if newton.solver.getConvergedReason() > 0:
                    done, saved = s, disp.x.array.copy()
                    step = min(1.5 * step, 1.0 - done) if done < 1.0 else step
                else:
                    disp.x.array[:] = saved
                    step *= 0.5
                    if step < 1e-4:
                        raise RuntimeError("swelling: Newton's method failed in finite strains")
            u_prev[:] = target

    value = activity if callable(activity) else (lambda t, a=float(activity): a)
    volume = float(np.abs(mesh.volumes()).sum())
    solvers: dict[float, object] = {}

    def diffusion_step(dt, t):
        key = float(dt)
        if key not in solvers:
            inv_dt.value = 1.0 / dt
            mat = dolfinx.fem.petsc.assemble_matrix(a_d, bcs=[bc])
            mat.assemble()
            ksp = PETSc.KSP().create(msh.comm)
            ksp.setOperators(mat)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            solvers.clear()
            solvers[key] = ksp
        inv_dt.value = 1.0 / dt
        level.value = value(t)
        b = dolfinx.fem.petsc.assemble_vector(l_d)
        dolfinx.fem.petsc.apply_lifting(b, [a_d], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, [bc])
        solvers[key].solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        u_n.x.array[:] = u_h.x.array
        b.destroy()

    times = np.asarray(times, dtype=float)
    if np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError("times: positive and increasing")
    uptake, conc, displacement = [], [], []
    mean_c = dolfinx.fem.form(sol * u_h * ufl.dx)
    # the solubility of an element around each node, for the nodal concentrations
    element_sol = np.array([float(phases[int(p)].get("solubility", 1.0)) for p in mesh.phase])
    node_sol = np.zeros(mesh.n_nodes)
    node_sol[mesh.elements.ravel()] = np.repeat(element_sol, mesh.elements.shape[1])
    t = 0.0
    u_h.x.array[:] = initial
    for t_out in times:
        span = t_out - t
        if t == 0.0:
            r = 1.25
            first = span * (r - 1.0) / (r**steps - 1.0)
            dts = first * r ** np.arange(steps)
        else:
            dts = np.full(steps, span / steps)
        for dt in dts:
            t += dt
            diffusion_step(dt, t)
        t = float(t_out)
        uptake.append(float(dolfinx.fem.assemble_scalar(mean_c)) / volume)
        conc.append(node_sol * np.asarray(u_h.x.array)[node_s])
        if kinematics == "finite":
            finite_solve()
        else:
            b = dolfinx.fem.petsc.assemble_vector(l_m)
            dolfinx.fem.petsc.apply_lifting(b, [a_m], bcs=[bcs_m])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b, bcs_m)
            ksp_m.solve(b, disp.x.petsc_vec)
            disp.x.scatter_forward()
            b.destroy()
        displacement.append(np.asarray(disp.x.array).reshape(-1, 3)[node_w].copy())
    s_mean = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(sol * ufl.dx))) / volume
    return SwellingResult(
        times, np.asarray(uptake), s_mean * float(value(times[-1])), conc, displacement, mesh
    )


def strip_curvature(mesh: TetMesh, displacement, *, axis: int = 0, normal: int = 2) -> float:
    """Curvature of a strip along ``axis`` bending in the plane of ``normal``: minus the
    second derivative of the deflection of the nodes of its mid-plane (least squares,
    ``w = -k x^2 / 2 + a x + b + c y``), so that it is positive when the strip bends
    towards its lower face (``-normal``), as :func:`~morphogen.coupling.timoshenko_curvature`
    and :class:`~morphogen.coupling.Multilayer`."""
    d = np.asarray(displacement)
    mid = 0.5 * (mesh.lower[normal] + mesh.upper[normal])
    z = mesh.nodes[:, normal]
    close = (
        np.abs(z - mid)
        <= 0.5 * np.min(np.abs(np.unique(z) - mid)[np.unique(z) != mid], initial=np.inf)
        + mesh.tolerance()
    )
    if not close.any():
        close = np.abs(z - mid) == np.abs(z - mid).min()
    x = mesh.nodes[close, axis]
    other = 3 - axis - normal
    y = mesh.nodes[close, other]
    w = d[close, normal]
    basis = np.column_stack([0.5 * x**2, x, np.ones_like(x), y])
    coef, *_ = np.linalg.lstsq(basis, w, rcond=None)
    return -float(coef[0])


def arc_curvature(mesh: TetMesh, displacement, *, axis: int = 0, normal: int = 2) -> float:
    """Curvature of a strip that bends into an arc (large rotations): the circle fitted
    (least squares, Kasa) to the deformed nodes of its mid-plane in the plane of ``axis``
    and ``normal``; positive when the strip bends towards its lower face."""
    d = np.asarray(displacement)
    mid = 0.5 * (mesh.lower[normal] + mesh.upper[normal])
    z0 = mesh.nodes[:, normal]
    close = np.abs(z0 - mid) == np.abs(z0 - mid).min()
    x = mesh.nodes[close, axis] + d[close, axis]
    z = z0[close] + d[close, normal]
    basis = np.column_stack([x, z, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(basis, -(x * x + z * z), rcond=None)
    cx, cz = -0.5 * coef[0], -0.5 * coef[1]
    radius = float(np.sqrt(cx * cx + cz * cz - coef[2]))
    return (1.0 if cz < z.mean() else -1.0) / radius


@dataclasses.dataclass
class OxidationResult:
    """At the ``times``: the oxidation tracer (amide concentration, mol/l) of each element,
    their stresses (Voigt order 11 22 33 23 31 12) and the displacement of the nodes."""

    times: np.ndarray
    tracer: list
    stress: list
    displacement: list
    mesh: TetMesh = dataclasses.field(repr=False)


def oxidation_stresses(
    mesh: TetMesh,
    law,
    times,
    *,
    oxidation=None,
    exposed=("z-", "z+"),
    poisson: float = 0.37,
    nodes: int = 81,
) -> OxidationResult:
    """Stresses of a sample oxidized through its ``exposed`` faces (Ernault et al. 2019):
    the amide profile of the diffusion-limited oxidation
    (:class:`~morphogen.coupling.ClosedLoopOxidation`, default: their DGEBA/IPDA at 110
    °C) at the depth of each element below the nearest exposed face, Young's modulus and
    the shrinkage (``expansion``, a strain) of ``law``
    (:class:`~morphogen.coupling.PropertyLaw` of the amide concentration), and the
    linear elastic response of the free sample (3-2-1 supports) to these eigenstrains
    (Poisson's ratio ``poisson``, 0.37 as theirs)."""
    from morphogen.coupling.oxidation import ClosedLoopOxidation

    _, dolfinx, _, ufl, _, PETSc = _modules()
    oxidation = oxidation or ClosedLoopOxidation()
    for face in exposed:
        if face not in FACES:
            raise ValueError(f"exposed: faces among {', '.join(FACES)}")
    times = np.asarray(times, dtype=float)
    centroids = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    depth = np.full(len(centroids), np.inf)
    for face in exposed:
        axis = "xyz".index(face[0])
        plane = mesh.lower[axis] if face[1] == "-" else mesh.upper[axis]
        depth = np.minimum(depth, np.abs(centroids[:, axis] - plane))
    profile = oxidation.profile(2.0 * float(depth.max()) * 1.02, times, nodes=nodes)
    surface = oxidation.thin_film(times)["PO"]
    msh, original = _dolfinx_mesh(mesh)
    w_space = dolfinx.fem.functionspace(msh, ("Lagrange", mesh.order, (3,)))
    node_w = _node_map(w_space, mesh)
    young = _cellwise(dolfinx, msh, np.ones(len(original)))
    shrink = _cellwise(dolfinx, msh, np.zeros(len(original)))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    d, e = ufl.TrialFunction(w_space), ufl.TestFunction(w_space)

    def eps(z):
        return ufl.sym(ufl.grad(z))

    identity = ufl.Identity(3)
    a_form = dolfinx.fem.form(
        ufl.inner(2.0 * mu * eps(d) + lam * ufl.tr(eps(d)) * identity, eps(e)) * ufl.dx
    )
    l_form = dolfinx.fem.form((3.0 * lam + 2.0 * mu) * shrink * ufl.div(e) * ufl.dx)
    tree = cKDTree(mesh.nodes)
    lo, hi = mesh.lower, mesh.upper
    corners = [lo, np.array([hi[0], lo[1], lo[2]]), np.array([lo[0], hi[1], lo[2]])]
    supports = [int(tree.query(c)[1]) for c in corners]
    held = {0: [supports[0]], 1: [supports[0], supports[1]], 2: supports}
    bcs = []
    for c in range(3):
        sub, parent = w_space.sub(c).collapse()
        inverse = np.full(3 * len(node_w), -1, dtype=np.int64)
        inverse[np.asarray(parent)] = np.arange(len(parent))
        dofs = np.array(sorted({3 * int(node_w[n]) + c for n in held[c]}), dtype=np.int32)
        bcs.append(
            dolfinx.fem.dirichletbc(
                dolfinx.fem.Function(sub), [dofs, inverse[dofs].astype(np.int32)], w_space.sub(c)
            )
        )
    disp = dolfinx.fem.Function(w_space)
    strain = eps(disp)
    sigma = (
        2.0 * mu * (strain - shrink * identity)
        + lam * ufl.tr(strain - shrink * identity) * identity
    )
    t_space = dolfinx.fem.functionspace(msh, ("DG", 0, (3, 3)))
    points = t_space.element.interpolation_points
    points = points() if callable(points) else points
    sigma_expr = dolfinx.fem.Expression(sigma, points)
    stress_fn = dolfinx.fem.Function(t_space)
    cells = t_space.dofmap.list[:, 0]
    order = np.empty(len(original), dtype=np.int64)
    order[original] = np.arange(len(original))  # morphoGen element -> dolfinx cell
    depths = np.concatenate([[0.0], profile["depth"]])
    tracers, stresses, displacements = [], [], []
    for k in range(len(times)):
        amide = np.concatenate([[surface[k]], profile["PO"][k]])
        tracer = np.interp(depth, depths, amide)
        props = law(tracer)
        young.x.array[young.function_space.dofmap.list[:, 0]] = props["young"][original]
        shrink.x.array[shrink.function_space.dofmap.list[:, 0]] = props.get(
            "expansion", np.zeros_like(tracer)
        )[original]
        mat = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
        mat.assemble()
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(mat)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        b = dolfinx.fem.petsc.assemble_vector(l_form)
        dolfinx.fem.petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, bcs)
        ksp.solve(b, disp.x.petsc_vec)
        disp.x.scatter_forward()
        stress_fn.interpolate(sigma_expr)
        tensors = np.asarray(stress_fn.x.array).reshape(-1, 9)[cells][order].reshape(-1, 3, 3)
        voigt = np.stack(
            [
                tensors[:, 0, 0],
                tensors[:, 1, 1],
                tensors[:, 2, 2],
                tensors[:, 1, 2],
                tensors[:, 2, 0],
                tensors[:, 0, 1],
            ],
            axis=1,
        )
        tracers.append(tracer)
        stresses.append(voigt)
        displacements.append(np.asarray(disp.x.array).reshape(-1, 3)[node_w].copy())
        b.destroy()
        ksp.destroy()
        mat.destroy()
    return OxidationResult(times, tracers, stresses, displacements, mesh)
