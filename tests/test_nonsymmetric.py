"""Nonsymmetric consistent tangents in morphoGen's own solvers: the laws whose tangents
are not symmetric (the Armstrong-Frederick hardening, the isotropic hardening of the slip
systems and the crystals of finite strains, porous plasticity, the learned perceptrons
and recurrent networks, the MFront behaviours that do not declare a symmetric tangent)
keep their full tangents, symmetrized again by the Krylov solvers where their measured
asymmetry is small, the other Newton corrections solved by GMRES with deflated restarts
and the Fourier preconditioner (by LU, or GMRES with the multigrid of the symmetric part,
by finite elements); the symmetric ones keep the symmetrized tangents and the conjugate
gradients, to the bit. On the CPU and on the GPU; the MGIS path: test_learning_mfront.py
and test_mfront.py."""

import dataclasses

import numpy as np
import pytest

from helpers import box_mesh
from morphogen import gpu
from morphogen.homogenization import (
    CrystalPlasticity,
    Material,
    NonlinearFFTProblem,
    Plasticity,
    simulate_point,
)
from morphogen.homogenization.finite import gradient_of, macro_part
from morphogen.homogenization.materials import material_table
from morphogen.homogenization.nonlinear import (
    ASYMMETRY,
    NonlinearFEProblem,
    _update,
    asymmetry,
    behaviour_table,
)
from morphogen.homogenization.solvers import (
    GMRES_DEFLATE,
    GMRES_RESTART,
    GMRES_STAGNATION,
    Recycling,
    conjugate_gradient,
    gmres,
)
from morphogen.image import VoxelImage
from morphogen.learning import Recurrent, generate, learned_material, random_paths, train
from morphogen.orientations import random_orientations

needs_gpu = pytest.mark.skipif(not gpu.available(), reason="no usable GPU (CuPy, CUDA)")
ELASTIC = {"young": 200000.0, "poisson": 0.3}
HILL = (1.3, 0.8, 1.0, 1.1, 0.9, 1.2)
STIFF = Material(young=400e3, poisson=0.2)


def _plastic(**kwargs) -> Material:
    spec = {"yield_stress": 100.0, "hardening": 500.0, **kwargs}
    return Material(young=70e3, poisson=0.3, plasticity=Plasticity(**spec))


def _crystal(**params) -> Material:
    spec = {"tau0": 60.0, "K": 10.0, "n": 15.0, **params}
    return dataclasses.replace(Material.cubic(168e3, 121e3, 75e3),
                               crystal_plasticity=CrystalPlasticity.from_config(spec))  # fmt: skip


GTN = Material(young=200e3, poisson=0.3, porous_plasticity={
    "criterion": "gurson", "yield": 300.0, "hardening": 1000.0, "porosity": 0.04})  # fmt: skip
#: The laws and whether their consistent tangents are symmetric, in small and in finite
#: strains
LAWS = {
    "elastic": (Material.cubic(168e3, 121e3, 75e3), True, True),
    "von Mises, Voce, Prager, Norton": (
        _plastic(isotropic=((50, 20),), kinematic=((5000, 0),), viscosity=50, exponent=5),
        True,
        True,
    ),
    "Hill, Norton": (_plastic(viscosity=50, exponent=5, hill=HILL), True, True),
    "von Mises, Armstrong-Frederick": (_plastic(kinematic=((20000, 200),)), False, False),
    "Hill, Armstrong-Frederick": (_plastic(kinematic=((20000, 200),), hill=HILL), False, False),
    "crystal": (_crystal(C=20000, D=300), True, False),
    "crystal, isotropic hardening": (_crystal(Q=20, b=10), False, False),
    "GTN": (GTN, False, False),
}


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_tangents_of_the_laws(kinematics):
    """Which laws have symmetric consistent tangents: the full tangents, along two steps
    (the second from backstresses, slips and porosities), are symmetric to round-off for
    those the behaviours declare symmetric (``Behaviours.symmetric``) and off symmetric by
    1e-4 to 1e-2 of their largest entries for the others; the symmetrized tangents are
    their symmetric parts, to the bit in small strains (the kernels only skip the
    symmetrization) and to round-off in finite strains (where the tangent of the
    logarithmic strain is symmetrized too, before ``dP/dF``)."""
    finite = kinematics == "finite"
    rng = np.random.default_rng(1)
    n, nc = 32, 9 if finite else 6
    e1 = 0.004 * rng.normal(size=(n, 6)) + np.array([0.006, -0.002, -0.002, 0, 0, 0])
    e2 = e1 + 0.004 * rng.normal(size=(n, 6))
    for name, (material, small, large) in LAWS.items():
        mats = {0: material}
        table = material_table([0], mats, "elastic", grains=[0],
                               orientations=np.array([[30.0, 40.0, 50.0]]))  # fmt: skip
        beh = behaviour_table([0], mats, table, finite=finite)
        assert beh.symmetric == (large if finite else small), name
        kind, mat, pmat = (np.repeat(a, n) for a in (beh.kind, table.mat, beh.pmat))
        tangents = {}
        for symmetric in (True, False):
            old, new = np.zeros((n, beh.width)), np.zeros((n, beh.width))
            stress, tangent = np.zeros((n, nc)), np.zeros((n, nc, nc))
            for e in (e1, e2):
                grads = gradient_of(e) if finite else e
                assert _update(finite, kind, mat, pmat, beh.data, grads, 0.0, 1.0, old, new,
                               stress, tangent, 0, 1.0, symmetric) == 0  # fmt: skip
                old, new = new, old
            tangents[symmetric] = tangent
        full = tangents[False]
        part = 0.5 * (full + np.swapaxes(full, 1, 2))
        if finite:
            assert np.abs(tangents[True] - part).max() < 1e-13 * np.abs(full).max(), name
        else:
            assert np.array_equal(tangents[True], part), name
        gap = np.abs(full - np.swapaxes(full, 1, 2)).max() / np.abs(full).max()
        if beh.symmetric:
            assert gap < 1e-12, (name, gap)
        else:
            assert 1e-5 < gap < 0.05, (name, gap)


def _system(n=200, skew=0.3, seed=0, negative=0):
    """A nonsymmetric system: a symmetric positive definite matrix (with ``negative`` small
    negative eigenvalues: indefinite, as the tangents of softening points) plus a skew part
    of ``skew`` times its norm, a diagonal preconditioner and the exact solution."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = rng.uniform(0.1, 2.0, n)
    if negative:
        lam[:negative] = -rng.uniform(1e-3, 0.05, negative)
    a = q @ np.diag(lam) @ q.T
    s = rng.normal(size=(n, n))
    a += skew * np.linalg.norm(a, 2) * (s - s.T) / np.linalg.norm(s - s.T, 2)
    d = np.abs(np.diag(a)) + 0.5
    b = rng.normal(size=n)
    return a, d, b, np.linalg.solve(a, b)


def test_gmres():
    """Restarted GMRES preconditioned on the right solves a nonsymmetric system to its
    tolerance (the residual of the system itself), with and without restarts, where the
    conjugate gradients do not; a zero right-hand side gives zero, and the iterations stop
    at ``maxiter``. On the GPU, the same iterations and solution."""
    a, d, b, exact = _system()
    for restart in (30, 8):
        x, it, res = gmres(lambda v: a @ v, lambda r: r / d, b, 1e-10, 2000, restart=restart)
        assert res < 1e-10 and np.linalg.norm(b - a @ x) < 2e-10 * np.linalg.norm(b)
        assert np.linalg.norm(x - exact) < 1e-8 * np.linalg.norm(exact)
        assert 0 < it < 2000
    y, _, _ = conjugate_gradient(lambda v: a @ v, lambda r: r / d, b, 1e-10, 2000)
    assert np.linalg.norm(y - exact) > 1e-3 * np.linalg.norm(exact)
    x, it, res = gmres(lambda v: a @ v, lambda r: r / d, np.zeros(b.size), 1e-10, 100)
    assert it == 0 and res == 0.0 and not np.any(x)
    _, it, res = gmres(lambda v: a @ v, lambda r: r / d, b, 1e-14, 7, restart=5)
    assert it == 7 and res > 1e-14
    if gpu.available():
        import cupy as cp

        from morphogen.homogenization import gpu as hgpu

        ag, dg = cp.asarray(a), cp.asarray(d)
        x0, it0, _ = gmres(lambda v: a @ v, lambda r: r / d, b, 1e-10, 2000, restart=8)
        x1, it1, res1 = hgpu.gmres(lambda v: (ag * v[None, :]).sum(axis=1),  # without cuBLAS
                                   lambda r: r / dg, cp.asarray(b), 1e-10, 2000,
                                   restart=8)  # fmt: skip
        assert abs(it1 - it0) <= 1 and res1 < 1e-10
        assert np.linalg.norm(cp.asnumpy(x1) - x0) < 1e-8 * np.linalg.norm(x0)


def test_gmres_with_deflated_restarts():
    """GMRES with deflated restarts (by default: the harmonic Ritz vectors of the 20 least
    harmonic Ritz values kept at each restart) against the plain restarted GMRES
    (``deflate=0``) on an indefinite nonsymmetric system, whose few small negative
    eigenvalues stall the plain restarts as those of the tangents of softening points do:
    the same solution in less than half the products (measured: 115 against 305 with 50
    vectors, 116 with 30 and 10 against 357); a solve that does not restart is the plain
    one to the bit. On the GPU, the same iterations and solution."""
    a, d, b, exact = _system(n=400, skew=0.02, negative=8)
    count = [0]

    def apply(v):
        count[0] += 1
        return a @ v

    runs = {}
    for restart, deflate in ((50, 0), (50, GMRES_DEFLATE), (30, 0), (30, 10)):
        count[0] = 0
        x, it, res = gmres(apply, lambda r: r / d, b, 1e-10, 5000, restart=restart,
                           deflate=deflate)  # fmt: skip
        assert res < 1e-10 and np.linalg.norm(b - a @ x) < 2e-10 * np.linalg.norm(b)
        assert np.linalg.norm(x - exact) < 1e-8 * np.linalg.norm(exact)
        runs[restart, deflate] = (x, it, count[0])
    assert runs[50, GMRES_DEFLATE][2] < 0.5 * runs[50, 0][2]
    assert runs[30, 10][2] < 0.5 * runs[30, 0][2]
    a2, d2, b2, _ = _system(n=300, skew=0.1)
    plain = gmres(lambda v: a2 @ v, lambda r: r / d2, b2, 1e-8, 2000, deflate=0)
    new = gmres(lambda v: a2 @ v, lambda r: r / d2, b2, 1e-8, 2000)
    assert new[1] < GMRES_RESTART and new[1] == plain[1] and np.array_equal(new[0], plain[0])
    if gpu.available():
        import cupy as cp

        from morphogen.homogenization import gpu as hgpu

        ag, dg = cp.asarray(a), cp.asarray(d)
        x1, it1, res1 = hgpu.gmres(lambda v: (ag * v[None, :]).sum(axis=1),  # without cuBLAS
                                   lambda r: r / dg, cp.asarray(b), 1e-10, 5000)  # fmt: skip
        x0, it0, _ = runs[50, GMRES_DEFLATE]
        assert abs(it1 - it0) <= 1 and res1 < 1e-10
        assert np.linalg.norm(cp.asnumpy(x1) - x0) < 1e-8 * np.linalg.norm(x0)


def _sequence(n=400, systems=8, skew=0.02, negative=8, change=1e-3, seed=0):
    """A sequence of systems whose operators change little, as the tangents of the Newton
    iterations of a path: that of :func:`_system` perturbed by ``change`` of its norm at
    each one, new right-hand sides; and the diagonal preconditioner."""
    a, d, _, _ = _system(n=n, skew=skew, negative=negative, seed=seed)
    rng = np.random.default_rng(seed + 1)
    p = rng.normal(size=(n, n))
    p *= change * np.linalg.norm(a, 2) / np.linalg.norm(p, 2)
    return [(a + i * p, rng.normal(size=n)) for i in range(systems)], d


def _solve_sequence(systems, d, recycle, tol=1e-10, solver=gmres, device=np.asarray):
    """The solutions, iterations and products of the systems solved one after the other,
    ``recycle`` told of each new operator."""
    count, out = [0], []
    for a, b in systems:
        if recycle is not None:
            recycle.renew()
        ad = device(a)

        def apply(v, ad=ad):
            count[0] += 1
            return (ad * v[None, :]).sum(axis=1)  # without BLAS or cuBLAS

        x, it, res = solver(apply, lambda r: r / device(d), device(b), tol, 5000,
                            recycle=recycle)  # fmt: skip
        x = np.asarray(x.get() if hasattr(x, "get") else x)
        assert res < tol and np.linalg.norm(b - a @ x) < 2 * tol * np.linalg.norm(b)
        out.append((x, it))
    return out, count[0]


def test_recycling_along_a_sequence_of_systems():
    """GCRO-DR (:class:`Recycling`) along a sequence of nonsymmetric indefinite systems whose
    operators change little, against GMRES with deflated restarts: the first solve, which
    restarts, is that of GMRES-DR to the bit and leaves the space of its last restart to
    the next ones, which reach the same solutions in fewer products (measured: 556 against
    901 for 8 systems, the 140 products of the images of the recycled vectors by the new
    operators included); a solve of the same operator reuses the images; a sequence whose
    solves never restart leaves the space empty and is that of GMRES-DR to the bit; a
    solve that fills no cycle has the next operator drop the space. On the GPU, the same
    iterations and solutions."""
    systems, d = _sequence()
    dr, products = _solve_sequence(systems, d, None)
    recycle = Recycling()
    gcro, recycled = _solve_sequence(systems, d, recycle)
    assert dr[0][1] > GMRES_RESTART and gcro[0][1] == dr[0][1]
    assert np.array_equal(gcro[0][0], dr[0][0])
    for (x1, _), (x0, _) in zip(gcro, dr, strict=True):
        assert np.linalg.norm(x1 - x0) < 1e-8 * np.linalg.norm(x0)
    assert recycled < 0.7 * products and all(it < 0.7 * dr[0][1] for _, it in gcro[1:])
    assert recycle.solves == len(systems) - 1 and recycle.products > 0
    a, b = systems[-1]
    images = recycle.products
    x, it, res = gmres(lambda v: a @ v, lambda r: r / d, b[::-1].copy(), 1e-10, 5000,
                       recycle=recycle)  # fmt: skip
    assert res < 1e-10 and recycle.products == images and it < 0.7 * dr[0][1]
    easy, d2 = _sequence(n=300, systems=3, skew=0.1, negative=0, seed=3)
    plain, _ = _solve_sequence(easy, d2, None, tol=1e-8)
    empty = Recycling()
    same, _ = _solve_sequence(easy, d2, empty, tol=1e-8)
    assert empty.u is None and all(it < GMRES_RESTART for _, it in plain)
    assert all(np.array_equal(x1, x0) for (x1, _), (x0, _) in zip(same, plain, strict=True))
    assert recycle.needed  # that solve filled a cycle: the next operator keeps the space
    recycle.renew()
    a3 = systems[0][0] + 0.5 * np.diag(np.abs(np.diag(systems[0][0])) + 1.0)
    x, it, res = gmres(lambda v: a3 @ v, lambda r: r / d, systems[0][1], 1e-6, 5000,
                       recycle=recycle)  # fmt: skip
    b3 = systems[0][1]
    assert res < 1e-6 and np.linalg.norm(b3 - a3 @ x) < 2e-6 * np.linalg.norm(b3)
    assert recycle.u is not None and not recycle.needed and recycle.solves == len(systems) + 1
    recycle.renew()  # a solve that filled no cycle: the space dropped
    assert recycle.u is None
    if gpu.available():
        import cupy as cp

        from morphogen.homogenization import gpu as hgpu

        on_gpu, _ = _solve_sequence(systems, d, Recycling(), solver=hgpu.gmres,
                                    device=cp.asarray)  # fmt: skip
        for (x1, it1), (x0, it0) in zip(on_gpu, gcro, strict=True):
            assert abs(it1 - it0) <= 1
            assert np.linalg.norm(x1 - x0) < 1e-8 * np.linalg.norm(x0)


def _stalling(n=300, negative=30, small=30, skew=0.02, seed=0):
    """A nonsymmetric system with more eigenvalues of both signs near zero than GMRES with
    deflated restarts keeps, as the tangents of the softening voxels of finer grids: 30
    negative ones in [-0.05, -1e-3] and 30 positive ones in [1e-3, 0.05] of 300, the others
    in [0.1, 2], the skew part 0.02 of the norm; the diagonal preconditioner, a right-hand
    side."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = rng.uniform(0.1, 2.0, n)
    lam[:negative] = -rng.uniform(1e-3, 0.05, negative)
    lam[negative : negative + small] = rng.uniform(1e-3, 0.05, small)
    a = q @ np.diag(lam) @ q.T
    s = rng.normal(size=(n, n))
    a += skew * np.linalg.norm(a, 2) * (s - s.T) / np.linalg.norm(s - s.T, 2)
    return a, np.abs(np.diag(a)) + 0.5, rng.normal(size=n)


def test_gmres_stops_where_it_stagnates():
    """On :func:`_stalling` GMRES with deflated restarts stagnates (2,000 iterations leave
    0.13 of the residual; without restarts it converges in 300, the size of the system), as
    it does on the softening voxels of 16^3 grids, where no Krylov method reaches the
    tolerance within its iterations. With ``stagnation`` (:data:`GMRES_STAGNATION` cycles
    without halving the residual, as the voxel paths ask) it stops after 290 iterations at
    0.14 of it, an inexact correction for Newton's method, and leaves the space of its last
    restart to the recycling; a solve that converges is unchanged to the bit. On the GPU,
    the same stop."""
    a, d, b = _stalling()
    stalled = gmres(lambda v: a @ v, lambda r: r / d, b, 1e-8, 2000)
    assert stalled[1] == 2000 and stalled[2] > 0.1
    recycle = Recycling()
    x, it, res = gmres(lambda v: a @ v, lambda r: r / d, b, 1e-8, 2000, recycle=recycle,
                       stagnation=GMRES_STAGNATION)  # fmt: skip
    assert it < 400 and res < 1.5 * stalled[2]
    assert np.linalg.norm(b - a @ x) / np.linalg.norm(b) == pytest.approx(res, rel=1e-6)
    assert recycle.u.shape[0] >= GMRES_DEFLATE and recycle.needed
    a2, d2, b2, _ = _system(n=400, skew=0.02, negative=8)
    plain = gmres(lambda v: a2 @ v, lambda r: r / d2, b2, 1e-10, 5000)
    stopped = gmres(lambda v: a2 @ v, lambda r: r / d2, b2, 1e-10, 5000,
                    stagnation=GMRES_STAGNATION)  # fmt: skip
    assert plain[1] > GMRES_RESTART and stopped[1] == plain[1]
    assert np.array_equal(stopped[0], plain[0])
    if gpu.available():
        import cupy as cp

        from morphogen.homogenization import gpu as hgpu

        ag, dg = cp.asarray(a), cp.asarray(d)
        _, it1, res1 = hgpu.gmres(lambda v: (ag * v[None, :]).sum(axis=1), lambda r: r / dg,
                                  cp.asarray(b), 1e-8, 2000,
                                  stagnation=GMRES_STAGNATION)  # fmt: skip
        assert abs(it1 - it) <= 30 and res1 < 1.5 * stalled[2]


def _state(problem, q, e, old):
    """Discretization, residual of the equilibrium, integral of the stress, tangents and
    weights of the points of ``problem`` at the fluctuation ``q`` and the macroscopic
    components ``e``, a step of unit time from the internal variables ``old``."""
    disc = problem._discretization()
    n_pts = problem.kind.size
    nc = 9 if problem.finite else 6
    grads, wts = np.zeros((n_pts, nc)), np.zeros(n_pts)
    sigma, tangent = np.zeros((n_pts, nc)), np.zeros((n_pts, nc, nc))
    new = np.zeros_like(old)
    disc.measures(q, e, grads, wts)
    assert disc.integrate(grads, 0.0, 1.0, old, new, sigma, tangent, 0, 1.0) == 0
    r_q, total, _ = disc.forces(sigma, wts)
    if problem.finite:
        total = macro_part(total)
    return disc, r_q, total, tangent, wts


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_full_tangent_of_the_voxels_and_elements(kinematics):
    """With the full tangents of a law that is not symmetric (GTN in a plastic state, an
    elastic inclusion beside it), the tangent operator of the voxels applied matrix-free,
    its coupling with the macroscopic strain and ``K_Ew``, the derivative of the integral
    of the stress by the fluctuation (``macro_response``), are the finite differences of
    the forces and of the integral of the stress; the operator and the coupling are not
    symmetric (``K_Ew`` is not ``K_wE^T``). Likewise ``K_Ew`` of the finite elements."""
    rng = np.random.default_rng(5)
    data = (rng.random((4, 3, 5)) < 0.3).astype(np.uint8)
    image = VoxelImage(data, spacing=(0.5, 0.7, 0.4))
    mats = {0: GTN, 1: STIFF}
    problem = NonlinearFFTProblem(image, mats, kinematics=kinematics)
    assert not problem.symmetrize
    n_q = 3 * data.size
    q = 2e-3 * rng.normal(size=n_q)
    e = np.array([0.012, -0.004, 0.006, 0.008, -0.005, 0.003])
    old = np.zeros((problem.kind.size, problem.behaviours.width))
    disc, _, _, tangent, wts = _state(problem, q, e, old)
    solver = disc.linearize(tangent, wts, True)
    u, v = rng.normal(size=n_q), rng.normal(size=n_q)
    kv = solver.apply(v)
    h = 1e-7
    up, down = _state(problem, q + h * v, e, old), _state(problem, q - h * v, e, old)
    fd = (up[1] - down[1]) / (2 * h)
    assert np.allclose(kv, fd, rtol=0, atol=1e-6 * np.abs(fd).max())
    assert abs(u @ kv - v @ solver.apply(u)) > 1e-6 * abs(u @ kv)
    dtotal = (up[2] - down[2]) / (2 * h)
    k_ew = disc.macro_response(tangent, wts, v)
    assert np.allclose(k_ew, dtotal, rtol=0, atol=1e-6 * np.abs(dtotal).max())
    unknown = np.array([1, 2, 4])
    g, _ = disc.coupling(tangent, wts, unknown)
    for k, j in enumerate(unknown):
        d = np.zeros(6)
        d[j] = h
        fd = (_state(problem, q, e + d, old)[1] - _state(problem, q, e - d, old)[1]) / (2 * h)
        assert np.allclose(g[:, k], fd, rtol=0, atol=1e-6 * np.abs(g).max())
    assert np.abs(g.T @ v - k_ew[unknown]).max() > 1e-6 * np.abs(k_ew).max()
    # the finite elements: K_Ew of their reduced unknowns
    mesh = box_mesh((2, 2, 2), phase=lambda c: (c[:, 0] > 0.6).astype(int), perturb=0.2)
    fe = NonlinearFEProblem(mesh, mats, kinematics=kinematics)
    disc = fe._discretization()
    old = np.zeros((fe.kind.size, fe.behaviours.width))
    qf = 2e-3 * rng.normal(size=disc.n_q)
    vf = rng.normal(size=disc.n_q)
    _, _, _, tangent, wts = _state(fe, qf, e, old)
    up, down = _state(fe, qf + h * vf, e, old), _state(fe, qf - h * vf, e, old)
    dtotal = (up[2] - down[2]) / (2 * h)
    k_ew = fe._discretization().macro_response(tangent, wts, vf)
    assert np.allclose(k_ew, dtotal, rtol=0, atol=1e-6 * np.abs(dtotal).max())


@needs_gpu
@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_full_tangents_on_the_gpu(kinematics):
    """The CUDA kernels of the behaviours give the full tangents of the Numba kernels to
    round-off, point by point, for every law of :data:`LAWS` along two steps (and their
    symmetric parts when they symmetrize)."""
    import cupy as cp

    rng = np.random.default_rng(3)
    shape = (4, 3, 5)
    phases = rng.integers(0, len(LAWS), size=shape).astype(np.uint8)
    image = VoxelImage(phases)
    mats = {k: m for k, (m, _, _) in enumerate(LAWS.values())}
    grains = np.arange(phases.size).reshape(shape)
    orientations = random_orientations(rng, phases.size)
    nc = 9 if kinematics == "finite" else 6
    out = {}
    for device in ("cpu", "gpu"):
        problem = NonlinearFFTProblem(image, mats, grains=grains, orientations=orientations,
                                      kinematics=kinematics, device=device)  # fmt: skip
        assert not problem.symmetrize
        disc = problem._discretization()
        n_pts = problem.kind.size
        g, _, sig, tan, old, new = disc.arrays(n_pts, nc, problem.behaviours.width)
        xp = cp if device == "gpu" else np
        e = 0.004 * np.random.default_rng(4).normal(size=(n_pts, 6))
        steps = []
        for factor in (1.0, 1.8):
            grads = gradient_of(factor * e) if kinematics == "finite" else factor * e
            g[...] = xp.asarray(grads)
            assert disc.integrate(g, 0.0, 1.0, old, new, sig, tan, 0, 1.0) == 0
            steps.append(disc.host(tan.copy())[0])
            old, new = new, old
        out[device] = steps
    for a, b in zip(out["cpu"], out["gpu"], strict=True):
        assert np.allclose(b, a, rtol=0, atol=1e-10 * np.abs(a).max())
        assert np.abs(a - np.swapaxes(a, 1, 2)).max() > 1e-6 * np.abs(a).max()


def _gap(a, b) -> float:
    """The largest gap of the stresses of two curves of the same steps, relative to the
    largest stress."""
    assert np.array_equal(a.times, b.times)
    return float(np.abs(a.stress - b.stress).max() / np.abs(a.stress).max())


@pytest.fixture(scope="module")
def recurrent():
    """A small recurrent network (6 states) trained in a second on von Mises data (20 epochs
    on 8 random walks): its consistent tangent is far from symmetric."""
    plastic = {**ELASTIC, "plasticity": {"yield": 200.0, "hardening": 10000.0}}
    data = generate(plastic, random_paths(8, segments=3, increments=5, seed=3))
    model = Recurrent(hidden=6)
    train(model, data, epochs=20, batch=4)
    return learned_material(model, **ELASTIC)


#: A tension and its partial return, the other stresses free
TENSION = {"times": [0, 1, 2], "increments": [8, 8], "strain": {"11": [0, 0.004, 0.001]}}


def test_learned_law_on_voxels(recurrent):
    """The recurrent network on voxels: on a homogeneous image (the Newton iterations of
    the macroscopic strain alone) its symmetrized tangent does not converge, where its full
    one follows the material point; beside stiff voxels the full tangent and GMRES take a
    third of the Newton and linear iterations of the symmetrized one and the conjugate
    gradients (measured: 47 against 148, 819 against 2631), to the same curve. By finite
    elements (LU) the symmetrized tangent does not converge either, the full one does."""
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    assert not NonlinearFFTProblem(image, {0: recurrent}).symmetrize
    point = simulate_point(recurrent, TENSION)
    with pytest.raises(RuntimeError, match="no convergence"):
        NonlinearFFTProblem(image, {0: recurrent}, tangent="symmetric", max_cuts=4).run(TENSION)
    full = NonlinearFFTProblem(image, {0: recurrent}).run(TENSION)
    assert _gap(full, point) < 1e-5 and full.info["cuts"] == 0
    rng = np.random.default_rng(0)
    image = VoxelImage((rng.random((4, 4, 4)) < 0.25).astype(np.uint8))
    mats = {0: recurrent, 1: STIFF}
    sym = NonlinearFFTProblem(image, mats, tangent="symmetric").run(TENSION)
    full = NonlinearFFTProblem(image, mats).run(TENSION)
    assert full.iterations.sum() < 0.5 * sym.iterations.sum()
    assert full.info["linear_iterations"] < 0.5 * sym.info["linear_iterations"]
    assert _gap(full, sym) < 1e-5
    mesh = box_mesh((2, 2, 2), phase=lambda c: (c[:, 0] > 0.6).astype(int), perturb=0.2)
    with pytest.raises(RuntimeError, match="no convergence"):
        NonlinearFEProblem(mesh, mats, tangent="symmetric", max_cuts=3).run(TENSION)
    full = NonlinearFEProblem(mesh, mats).run(TENSION)
    assert full.info["cuts"] == 0 and full.iterations.max() <= 6


#: Crystal plasticity whose slip systems harden each other strongly (Franciosi's
#: interactions, the collinear one 12.3), nearly rate-independent, and a non-proportional
#: biaxial path whose other stresses are free
LATENT = _crystal(tau0=40.0, Q=100, b=50, K=2.0, n=25.0,
                  interaction=[1.0, 1.0, 0.6, 12.3, 1.6, 1.3])  # fmt: skip
BIAXIAL = {"times": [0, 10, 20], "increments": [10, 10],
           "strain": {"11": [0, 0.01, 0.0], "22": [0, 0.0, 0.01]}}  # fmt: skip


def test_crystals_with_latent_hardening():
    """The crystals of :data:`LATENT` along :data:`BIAXIAL`: with their full tangents (the
    default) Newton's method takes a third fewer iterations than with their symmetric
    parts at a material point (measured: 44 and 47 against 72 and 79) and by finite
    elements, whose LU iterations are exact Newton ones (35 against 51 for a crystal, 53
    against 64 for two grains), to the same curves within its tolerance; on voxels (the
    quasi-Newton iterations of the macroscopic strain, inexact corrections) GMRES on the
    full tangents (forced: ``auto`` symmetrizes these, below :data:`ASYMMETRY`) gives the
    curve of the conjugate gradients within the tolerance."""
    for euler in ((10.0, 75.0, 20.0), (30.0, 40.0, 50.0)):
        exact = simulate_point(LATENT, BIAXIAL, orientation=euler)
        sym = simulate_point(LATENT, BIAXIAL, orientation=euler, tangent="symmetric")
        assert exact.iterations.sum() < 0.7 * sym.iterations.sum(), euler
        assert _gap(exact, sym) < 1e-8
    for n_grains in (1, 2):
        mesh = box_mesh((2, 2, 2), perturb=0.2)
        centres = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
        cell = (centres[:, 0] > 0.5).astype(int) if n_grains == 2 else np.zeros(
            mesh.n_elements, int)  # fmt: skip
        euler = np.array([[10.0, 75.0, 20.0], [30.0, 40.0, 50.0]])[:n_grains]
        mesh = dataclasses.replace(mesh, cell=cell, orientations=euler)
        problem = NonlinearFEProblem(mesh, {0: LATENT})
        assert not problem.symmetrize
        full = problem.run(BIAXIAL)
        sym = NonlinearFEProblem(mesh, {0: LATENT}, tangent="symmetric").run(BIAXIAL)
        assert full.iterations.sum() < 0.9 * sym.iterations.sum(), n_grains
        assert _gap(full, sym) < 1e-5
    rng = np.random.default_rng(1)
    grains = rng.integers(0, 3, size=(4, 4, 4))
    image = VoxelImage(np.zeros((4, 4, 4), dtype=np.uint8))
    hardening = _crystal(Q=40, b=5)
    kw = {"grains": grains, "orientations": random_orientations(rng, 3)}
    path = {"times": [0, 10], "increments": 5, "strain": {"33": 0.01}}
    full = NonlinearFFTProblem(image, {0: hardening}, tangent="full", **kw).run(path)
    sym = NonlinearFFTProblem(image, {0: hardening}, tangent="symmetric", **kw).run(path)
    assert _gap(full, sym) < 1e-6 and abs(full.iterations.sum() - sym.iterations.sum()) <= 2


def test_symmetric_laws_keep_the_conjugate_gradients():
    """Laws whose tangents are all symmetric keep the symmetrized tangents and the
    conjugate gradients: the default gives the curves of ``tangent="symmetric"`` to the
    bit, on voxels and by finite elements (the bitwise identity with the solvers before
    this option was measured on these and other cases); the full tangents with GMRES
    (forced) give them within the tolerance of Newton's method. A law that is not
    symmetric beside them makes the whole volume element keep its full tangents (which
    ``auto`` then measures); the option is checked."""
    rng = np.random.default_rng(2)
    image = VoxelImage((rng.random((5, 4, 6)) < 0.3).astype(np.uint8))
    voce = _plastic(isotropic=((50, 20),), kinematic=((5000, 0),), viscosity=50, exponent=5)
    mats = {0: voce, 1: STIFF}
    path = {"times": [0, 10, 20], "increments": [4, 4], "strain": {"11": [0, 0.01, 0.004]},
            "stress": {"22": 0, "33": 0}, "temperature": [0, 30, 30]}  # fmt: skip
    mesh = box_mesh((2, 2, 2), phase=lambda c: (c[:, 0] > 0.6).astype(int), perturb=0.2)
    makers = (lambda **kw: NonlinearFFTProblem(image, mats, **kw),
              lambda **kw: NonlinearFFTProblem(image, mats, kinematics="finite", **kw),
              lambda **kw: NonlinearFEProblem(mesh, mats, **kw))  # fmt: skip
    for make in makers:
        auto = make()
        assert auto.symmetrize
        a, b = auto.run(path), make(tangent="symmetric").run(path)
        assert np.array_equal(a.stress, b.stress) and np.array_equal(a.iterations, b.iterations)
        assert a.info["linear_iterations"] == b.info["linear_iterations"]
        full = make(tangent="full")
        assert not full.symmetrize
        assert _gap(full.run(path), a) < 1e-6
    assert not NonlinearFFTProblem(image, {0: voce, 1: GTN}).symmetrize
    with pytest.raises(ValueError, match="tangent"):
        NonlinearFFTProblem(image, mats, tangent="nonsymmetric")
    with pytest.raises(ValueError, match="tangent"):
        simulate_point(voce, path, tangent="exact")


def test_auto_decides_by_the_asymmetry(recurrent):
    """``tangent="auto"`` with laws that are not all symmetric: the Krylov solvers measure
    the asymmetry of the full tangents of the points of each phase at each linearization
    (:func:`asymmetry`: ``|C - C^T| / |C|`` over the points of a phase, here against its
    definition) and symmetrize them where the largest is below :data:`ASYMMETRY`. An
    Armstrong-Frederick composite stays far below it (measured: 4.5e-3 for its plastic
    phase, 1.3e-3 for the whole) and gives the curve of ``tangent="symmetric"`` to the bit,
    no linearization full; the recurrent network beside stiff voxels stays far above it
    (0.21; 0.14 for the whole), every linearization full, and gives the curve of
    ``tangent="full"`` to the bit. By finite elements LU keeps the full tangents without
    measuring them (its iterations are exact Newton ones), multigrid measures them."""
    rng = np.random.default_rng(0)
    sym = rng.normal(size=(7, 6, 6))
    sym += np.swapaxes(sym, 1, 2)
    skew = rng.normal(size=(7, 6, 6))
    skew -= np.swapaxes(skew, 1, 2)
    t, w = sym + 0.01 * skew, rng.uniform(0.5, 2.0, 7)
    t[5:] += 0.2 * skew[5:]
    norms = [np.einsum("pij,pij->p", a, a) for a in (t - np.swapaxes(t, 1, 2), t)]
    assert asymmetry(t, w) == pytest.approx(np.sqrt(w @ norms[0] / (w @ norms[1])), rel=1e-12)
    groups = np.array([0, 0, 0, 0, 0, 1, 1])
    phase = np.sqrt(w[5:] @ norms[0][5:] / (w[5:] @ norms[1][5:]))
    assert asymmetry(t, w, groups) == pytest.approx(phase, rel=1e-12)
    assert asymmetry(t, w, groups) > asymmetry(t, w) and asymmetry(sym, w, groups) == 0.0
    image = VoxelImage((np.random.default_rng(2).random((5, 4, 6)) < 0.3).astype(np.uint8))
    composite = {0: _plastic(kinematic=((20000, 200),)), 1: STIFF}
    path = {"times": [0, 10, 20], "increments": [4, 4], "strain": {"11": [0, 0.01, 0.004]},
            "stress": {"22": 0, "33": 0}}  # fmt: skip
    small = VoxelImage((np.random.default_rng(0).random((4, 4, 4)) < 0.25).astype(np.uint8))
    network = {0: recurrent, 1: STIFF}
    for img, mats, p, forced in ((image, composite, path, "symmetric"),
                                 (small, network, TENSION, "full")):  # fmt: skip
        problem = NonlinearFFTProblem(img, mats)
        assert problem.measured and not problem.symmetrize
        auto, other = problem.run(p), NonlinearFFTProblem(img, mats, tangent=forced).run(p)
        assert np.array_equal(auto.stress, other.stress)
        assert np.array_equal(auto.iterations, other.iterations)
        if forced == "symmetric":
            assert auto.info["full_tangents"] == 0 and auto.info["asymmetry"] < 0.3 * ASYMMETRY
        else:
            assert auto.info["full_tangents"] >= auto.iterations.sum()
            assert auto.info["asymmetry"] > 3 * ASYMMETRY
        assert "asymmetry" not in other.info
    mesh = box_mesh((2, 2, 2), phase=lambda c: (c[:, 0] > 0.6).astype(int), perturb=0.2)
    lu = NonlinearFEProblem(mesh, composite).run(path)
    full = NonlinearFEProblem(mesh, composite, tangent="full").run(path)
    assert "asymmetry" not in lu.info and np.array_equal(lu.stress, full.stress)
    mesh = box_mesh((4, 4, 4), phase=lambda c: (c[:, 0] > 0.6).astype(int), perturb=0.2)
    amg = NonlinearFEProblem(mesh, composite, solver="amg").run(path)
    sym = NonlinearFEProblem(mesh, composite, solver="amg", tangent="symmetric").run(path)
    assert amg.info["full_tangents"] == 0 and np.array_equal(amg.stress, sym.stress)


def test_auto_measures_each_phase(recurrent):
    """``tangent="auto"`` decides by the largest asymmetry of the phases: one voxel of the
    recurrent network among von Mises and stiff voxels (0.8 % of 5^3) keeps its full
    tangents (0.2 for its phase), where the asymmetry of the whole volume element (1.5e-2,
    below :data:`ASYMMETRY`) would symmetrize them (measured on its tension and return: 43
    Newton iterations against 60 with the symmetrized tangents), and gives the curve of
    ``tangent="full"`` to the bit."""
    rng = np.random.default_rng(0)
    data = np.where(rng.random((5, 5, 5)) < 0.25, 1, 2).astype(np.uint8)
    data.flat[np.flatnonzero(data.ravel() == 2)[0]] = 0
    mises = Material(**ELASTIC, plasticity=Plasticity(yield_stress=200.0, hardening=10000.0))
    image, mats = VoxelImage(data), {0: recurrent, 1: STIFF, 2: mises}
    path = {"times": [0, 1], "increments": 8, "strain": {"11": 0.004}}
    whole = []
    auto = NonlinearFFTProblem(image, mats).run(
        path, monitor=lambda state: whole.append(asymmetry(state.tangent, state.wts))
    )
    full = NonlinearFFTProblem(image, mats, tangent="full").run(path)
    assert max(whole) < ASYMMETRY < auto.info["asymmetry"]
    assert auto.info["full_tangents"] == auto.iterations.sum() > 0
    assert np.array_equal(auto.stress, full.stress)


def test_tangent_in_configurations():
    """``homogenization.nonlinear.tangent`` of a configuration reaches the options of the
    solvers (``homogenize``); other values are refused."""
    import yaml

    from morphogen import cli
    from morphogen.config import ConfigError, load_config
    from morphogen.homogenization import homogenize

    cfg = yaml.safe_load(cli._templates()["homogenization_crystal_plasticity_fft"])
    cfg["homogenization"]["nonlinear"]["tangent"] = "symmetric"
    assert load_config(cfg).homogenization.options()["nonlinear"]["tangent"] == "symmetric"
    cfg["homogenization"]["nonlinear"]["tangent"] = "exact"
    with pytest.raises(ConfigError, match="tangent"):
        load_config(cfg)
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    paths = {"tension": {"times": [0, 1], "increments": 2, "strain": {"33": 0.004}}}
    hardening = _crystal(Q=40, b=5)
    out = {t: homogenize(image, {0: hardening}, properties=[], paths=paths,
                         nonlinear={"tangent": t})["tension.periodic"]
           for t in ("auto", "symmetric")}  # fmt: skip
    assert _gap(out["auto"], out["symmetric"]) < 1e-6


#: Stiff spheres in a Rousselier matrix strained at a high triaxiality, whose softening
#: points make the tangent indefinite
TRIAXIAL = {"times": [0, 10], "increments": 10, "strain": {"11": 0.02, "22": 0.005, "33": 0.005}}


def _rousselier_composite():
    from morphogen.domain import Domain
    from morphogen.models import BooleanSpheres

    micro = BooleanSpheres(radius=1.0, volume_fraction=0.2).generate(Domain.cube(10.0), seed=2)
    rousselier = Material(young=200e3, poisson=0.3, porous_plasticity={
        "criterion": "rousselier", "yield": 300.0, "hardening": 800.0, "porosity": 0.01,
        "sigma1": 400.0, "D": 2.0})  # fmt: skip
    return micro.voxelize(shape=6), {0: rousselier, 1: STIFF}


def test_softening_points_with_full_tangents(caplog):
    """The softening points of the Rousselier composite (:data:`TRIAXIAL`, 6^3 voxels) with
    their full tangents (forced: ``auto`` symmetrizes these, below :data:`ASYMMETRY`):
    GMRES with deflated restarts solves every correction (no warning) and, from the first
    solve that restarts on, recycles its space from a solve to the next (GCRO-DR), in fewer
    products than without recycling, than the plain restarts and than the conjugate
    gradients and MINRES on the symmetrized tangents (measured: 3,367 iterations, the 121
    products of the images of the recycled vectors included, 23 solves recycled, against
    3,774, 6,841 and 4,432; on 8^3 voxels 8,654 against 11,039, 40,381 and 9,725)."""
    import logging

    image, mats = _rousselier_composite()
    with caplog.at_level(logging.WARNING, logger="morphogen"):
        full = NonlinearFFTProblem(image, mats, tangent="full").run(TRIAXIAL)
    assert not [r for r in caplog.records if "stopped at relative residual" in r.getMessage()]
    assert full.info["linear_iterations"] < 3600 and full.info["cuts"] <= 3
    assert full.info["recycled"] > 0


#: :data:`TRIAXIAL` in four increments: steps too large for the softening points
TRIAXIAL_4 = {**TRIAXIAL, "increments": 4}


@pytest.mark.slow
def test_stalls_of_large_steps():
    """The Rousselier composite of :func:`test_softening_points_with_full_tangents` along
    :data:`TRIAXIAL_4`, whose large steps make the tangents of the softening points beyond
    the Krylov solvers (more eigenvalues of both signs near zero than GMRES keeps, as on
    16^3 voxels with ten increments): without stopping where it stagnates GMRES ran six
    solves to their 1,120 iterations above their tolerance (19,028 products in all, the
    same 52 Newton iterations and 4 divided steps). With the stop (:data:`GMRES_STAGNATION`)
    and the steps whose corrections stagnate divided, the path takes the Newton
    iterations, the divided steps and the end of the curve of the conjugate gradients and
    MINRES (which stall there too) in fewer products than either (measured: 52 and 53
    Newton iterations, 4 divided steps, the stresses 1e-7 apart; 7,501 products against
    19,028 and 21,293; 8,522 without recycling, 12,740 with recycling alone)."""
    image, mats = _rousselier_composite()
    full = NonlinearFFTProblem(image, mats, tangent="full").run(TRIAXIAL_4)
    sym = NonlinearFFTProblem(image, mats, tangent="symmetric").run(TRIAXIAL_4)
    assert full.info["linear_iterations"] < 0.6 * sym.info["linear_iterations"]
    assert abs(full.iterations.sum() - sym.iterations.sum()) <= 5
    gap = np.abs(full.stress[-1] - sym.stress[-1]).max() / np.abs(sym.stress[-1]).max()
    assert gap < 1e-5 and full.times[-1] == sym.times[-1] == TRIAXIAL_4["times"][-1]


@needs_gpu
@pytest.mark.slow
def test_softening_points_with_full_tangents_on_the_gpu():
    """The path of :func:`test_softening_points_with_full_tangents` on the GPU, its GMRES
    with deflated restarts and its recycling those of the CPU: the same curve and about as
    many products (measured: the same 3,367 iterations and 23 solves recycled)."""
    image, mats = _rousselier_composite()
    full = NonlinearFFTProblem(image, mats, tangent="full").run(TRIAXIAL)
    on_gpu = NonlinearFFTProblem(image, mats, tangent="full", device="gpu").run(TRIAXIAL)
    assert _gap(on_gpu, full) < 1e-6 and on_gpu.info["recycled"] > 0
    assert abs(on_gpu.info["linear_iterations"] - full.info["linear_iterations"]) < 400


def test_multigrid_and_gmres_by_finite_elements():
    """By finite elements, a system solved iteratively (``solver="amg"``) with full
    tangents (forced: ``auto`` symmetrizes these, below :data:`ASYMMETRY`, as it does with
    multigrid) is solved by GMRES preconditioned by the multigrid of its symmetric part, to
    the curve of LU within the tolerance of Newton's method."""
    mesh = box_mesh((4, 4, 4), phase=lambda c: (c[:, 0] > 0.6).astype(int), perturb=0.2)
    mats = {0: _plastic(kinematic=((20000, 200),)), 1: STIFF}
    path = {"times": [0, 10, 20], "increments": [4, 4], "strain": {"11": [0, 0.01, -0.004]},
            "stress": {"22": 0, "33": 0}}  # fmt: skip
    lu = NonlinearFEProblem(mesh, mats, solver="direct").run(path)
    amg = NonlinearFEProblem(mesh, mats, solver="amg", tangent="full").run(path)
    assert amg.info["linear_iterations"] > 0 and _gap(amg, lu) < 1e-6


#: A composite with Armstrong-Frederick hardening and the mixed path of the GPU tests
AF_COMPOSITE = {0: _plastic(isotropic=((50, 20),), kinematic=((20000, 200), (5000, 0)),
                            viscosity=50, exponent=5), 1: STIFF}  # fmt: skip
MIXED = {"times": [0, 10, 20], "increments": [4, 4], "strain": {"11": [0, 0.01, 0.004]},
         "stress": {"22": 0, "33": 0}}  # fmt: skip


def _gpu_image():
    rng = np.random.default_rng(2)
    return VoxelImage((rng.random((5, 4, 6)) < 0.3).astype(np.uint8), spacing=(0.5,) * 3)


@needs_gpu
def test_full_tangent_paths_on_the_gpu(recurrent):
    """With full tangents the GPU follows the CPU along paths, its GMRES the CPU's: the
    composite with Armstrong-Frederick hardening along a mixed path (small strains with
    B-bar, finite strains with F-bar; forced to the full tangents), and the recurrent
    network beside stiff voxels (by ``auto``)."""
    image = _gpu_image()
    small = VoxelImage((np.random.default_rng(0).random((4, 4, 4)) < 0.25).astype(np.uint8))
    cases = [(image, AF_COMPOSITE, "small", MIXED, "full"),
             (image, AF_COMPOSITE, "finite", MIXED, "full"),
             (small, {0: recurrent, 1: STIFF}, "small", TENSION, "auto")]  # fmt: skip
    for img, mats, kinematics, p, tangent in cases:
        res = {}
        for device in ("cpu", "gpu"):
            problem = NonlinearFFTProblem(img, mats, kinematics=kinematics, device=device,
                                          tangent=tangent)  # fmt: skip
            assert not problem.symmetrize
            res[device] = problem.run(p)
        assert _gap(res["gpu"], res["cpu"]) < 1e-6
        assert abs(int(res["gpu"].iterations.sum()) - int(res["cpu"].iterations.sum())) <= 2
        if tangent == "auto":  # every linearization full on both
            assert all(r.info["full_tangents"] >= r.iterations.sum() > 0 for r in res.values())


@needs_gpu
def test_auto_on_the_gpu():
    """With ``tangent="auto"`` the GPU measures the asymmetry of the CPU (its norms on the
    GPU) and takes its decisions: the composite with Armstrong-Frederick hardening
    symmetrized at every linearization, the symmetrization on the GPU that of its kernels
    (the curve of ``tangent="symmetric"`` on the GPU to the bit)."""
    image = _gpu_image()
    cpu = NonlinearFFTProblem(image, AF_COMPOSITE).run(MIXED)
    auto = NonlinearFFTProblem(image, AF_COMPOSITE, device="gpu").run(MIXED)
    sym = NonlinearFFTProblem(image, AF_COMPOSITE, device="gpu", tangent="symmetric").run(MIXED)
    assert auto.info["asymmetry"] == pytest.approx(cpu.info["asymmetry"], rel=1e-6)
    assert auto.info["full_tangents"] == cpu.info["full_tangents"] == 0
    assert np.array_equal(auto.stress, sym.stress) and _gap(auto, cpu) < 1e-6
