"""Micromorphic continua: the microfoam model of nickel foams (Dillard, Forest & Ienny
2006) — its point against the article's plastic contraction ratios, the consistent
tangents of its point and elements, the one-dimensional characteristic lengths of its
Appendix A (boundary layers in elasticity and plasticity, softening bands of the
predicted sine shape whose width the length sets, not the elements), in finite strains
its tangents, objectivity and small-strain limit — the micromorphic damage model (exact
tangents in small and finite strains, bands of damage independent of the mesh) and the
micromorphic ductile damage of von Mises elasto-(visco)plasticity (tangents, ductile
bands independent of the mesh)."""

import math

import numpy as np
import pytest

from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.image import VoxelImage
from morphogen.micromorphic import (
    CHI,
    DamageProblem,
    DuctileProblem,
    Microfoam,
    MicrofoamProblem,
    MicromorphicDamage,
    MicromorphicDuctile,
    characteristic_length,
)
from morphogen.micromorphic.ductile import driving_update, point_update
from morphogen.micromorphic.microfoam import plastic_update

E, MU_S, MU_H = 100.0, 1000.0, 50.0
# isotropic elasticity with nu = 0 and the elliptic criterion with F = c/2: tension is
# one-dimensional (no lateral strain), as in the article's Appendix A
BAR = dict(young=E, poisson=0.0, c=1.0, F=0.5, relative=(0.0, MU_S, MU_S), micro=(0.0, MU_H))


def _bar(length, h, weak=None):
    """A bar along x2 of voxels of side h, the voxels of ``|x2 - length/2| < weak / 2`` in
    phase 1; its two ends."""
    n = round(length / h)
    data = np.zeros((1, n, 1), np.uint8)
    if weak is not None:
        centre = (np.arange(n) + 0.5) * h
        data[0, np.abs(centre - length / 2) < weak / 2] = 1
    return voxel_tet_mesh(VoxelImage(data, spacing=(h, h, h), periodic=False))


def _ends(problem, length):
    x = problem.mesh.nodes
    return np.flatnonzero(np.abs(x[:, 1]) < 1e-9), np.flatnonzero(np.abs(x[:, 1] - length) < 1e-9)


def _line(problem):
    """The corner nodes of the edge x1 = x3 = 0, along x2."""
    x = problem.mesh.nodes
    edge = (np.abs(x[:, 0]) < 1e-9) & (np.abs(x[:, 2]) < 1e-9)
    line = np.flatnonzero(edge & problem.space.corner)
    return line[np.argsort(x[line, 1])]


def _fd_tangent(point, eps, h=1e-7):
    out = np.empty((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = h
        out[:, j] = (point(eps + d)[0] - point(eps - d)[0]) / (2 * h)
    return out


def test_microfoam_point_of_the_nickel_foam():
    """The parameters of the article's Table 2 give its plastic contraction ratios (its eq.
    8: 0.49 and 0.47 in tension along RD, 0.17 and 0.22 along TD); the implicit update
    is consistent (σ : ε̇^p = σ_eq ṗ, f = 0) and its tangent exact, softening included."""
    foam = Microfoam.nickel_foam()
    for axis, lateral, thickness in ((0, 0.49, 0.47), (1, 0.17, 0.22)):
        stress = np.zeros((3, 3))
        stress[axis, axis] = 1.0
        n = foam.flow(stress)
        assert -n[1 - axis, 1 - axis] / n[axis, axis] == pytest.approx(lateral, abs=0.006)
        assert -n[2, 2] / n[axis, axis] == pytest.approx(thickness, abs=0.006)
        assert np.sum(n * stress) == pytest.approx(foam.equivalent(stress))
    cm, mm = foam.elastic_matrix(), foam.criterion_matrix()
    cinv = np.linalg.inv(cm)
    r2 = math.sqrt(2.0)
    eps = np.array([0.002, 0.012, -0.001, 0.0005 * r2, -0.001 * r2, 0.003 * r2])
    epsp0 = np.array([0.0005, 0.001, -0.0002, 0.0, 0.0, 0.0002])
    for material in (foam, Microfoam.nickel_foam(softening=(0.002, 0.05, 0.05))):
        h = material.hardening_parameters()

        def point(e, h=h):
            return plastic_update(cm, cinv, mm, h, e, epsp0, 0.003)

        sig, epsp, p, tan = point(eps)
        assert p > 0.003
        dp = p - 0.003
        assert sig @ (epsp - epsp0) == pytest.approx(math.sqrt(sig @ mm @ sig) * dp, rel=1e-9)
        np.testing.assert_allclose(_fd_tangent(point, eps), tan, atol=1e-6 * np.abs(tan).max())


def test_element_tangents():
    """The tangents of the elements against finite differences of their forces: microfoam
    (micromorphic, plastic) and the damage model (damage growing) in small and finite
    strains."""
    rng = np.random.default_rng(1)
    mesh = voxel_tet_mesh(VoxelImage(np.zeros((2, 2, 1), np.uint8), periodic=False))
    foam = Microfoam(young=100.0, poisson=0.2, yield_stress=0.5, hardening=5.0, F=0.3, length=0.3,
                     micro=(20.0, 40.0), relative=(200.0, 300.0, 150.0))  # fmt: skip
    cases = [(MicrofoamProblem(mesh, foam), 2e-2, None)]
    damage = MicromorphicDamage(young=1000.0, poisson=0.25, threshold=1e-3, length=0.5, beta=200.0)
    for kinematics in ("small", "finite"):
        for variational in (False, True):
            material = MicromorphicDamage(**{**damage.__dict__, "variational": variational})
            cases.append((DamageProblem(mesh, material, kinematics=kinematics), 3e-3, 1.5e-3))
    for problem, scale, kappa in cases:
        space = problem.space
        u = rng.normal(size=space.n_dofs) * scale
        state = problem._initial_state()
        if kappa is not None:  # chi above the history: damage grows
            chi = space.offset[:-1][space.corner] + 3
            u[chi] = 2e-3 + rng.random(len(chi)) * 1e-3
            state[:, :, 0] = kappa
        _, data, new = problem._assemble(u, state, True)
        du = rng.normal(size=space.n_dofs) * scale / 3
        rp, _, _ = problem._assemble(u + 1e-6 * du, state, False)
        rm, _, _ = problem._assemble(u - 1e-6 * du, state, False)
        fd = (rp - rm) / 2e-6
        assert np.linalg.norm(fd - space.matrix(data) @ du) < 1e-6 * np.linalg.norm(fd)
        if kappa is None:
            assert (new[:, :, 6] > 0).all()  # every point plastic
        else:
            assert (new[:, :, 1] > 0).all()  # every point damaged


def test_boundary_layers_of_appendix_a():
    """A bar pulled with its microdeformation held at one end: χ22 follows ``1 -
    exp(-x/ℓ)`` with the elastic ``ℓ`` of the one-dimensional analysis; in the plastic
    regime (linear hardening, χ22 forced at the end), its increments decay over the
    plastic ``ℓ`` of the article's Appendix A, three times longer here."""
    elastic = Microfoam(length=0.1, **BAR)
    problem = MicrofoamProblem(_bar(1.2, 0.02), elastic)
    start, end = _ends(problem, 1.2)
    held = start[problem.space.corner[start]]
    result = problem.run([(start, [0, 1, 2], 0.0), (held, CHI["chi22"], 0.0), (end, 1, 0.012)],
                         steps=1)  # fmt: skip
    line = _line(problem)
    x = problem.mesh.nodes[line, 1]
    chi = problem.space.generalized(result.values)[line, CHI["chi22"] - 3]
    length = characteristic_length(elastic, E)
    np.testing.assert_allclose(chi / chi[-1], 1 - np.exp(-x / length), atol=0.01)
    others = np.delete(problem.space.generalized(result.values)[line], CHI["chi22"] - 3, axis=1)
    assert np.abs(others).max() < 1e-3 * chi[-1]

    plastic = Microfoam(yield_stress=1.0, hardening=20.0, length=0.1, **BAR)
    tangent = E * 20.0 / (20.0 + 1.5 * E)
    length = characteristic_length(plastic, tangent)
    # the article's (A.6), its relative and higher-order moduli those of elasticity:
    # 1/ω = l_c sqrt((1 + H') / H'), H' = H / (3 c μ + H)
    moduli = {"relative": (0.0, E / 2, E / 2), "micro": (0.0, E / 2)}
    article = Microfoam(length=0.1, **{**BAR, **moduli})
    ratio = 20.0 / (1.5 * E + 20.0)
    expected = 0.1 * math.sqrt((1 + ratio) / ratio)
    assert characteristic_length(article, tangent) == pytest.approx(expected)
    problem = MicrofoamProblem(_bar(2.0, 0.025), plastic)
    start, end = _ends(problem, 2.0)
    held = start[problem.space.corner[start]]
    result = problem.run([(start, [0, 1, 2], 0.0), (held, CHI["chi22"], 0.075), (end, 1, 0.1)],
                         steps=6, fields=True)  # fmt: skip
    line = _line(problem)
    x = problem.mesh.nodes[line, 1]
    k = CHI["chi22"] - 3
    last, before = (problem.space.generalized(result.fields[i])[line, k] for i in (-1, -2))
    step = last - before
    fit = step[-1] + (step[0] - step[-1]) * np.exp(-x / length)
    assert np.abs(step - fit).max() < 0.01 * abs(step[0] - step[-1])
    assert result.state[:, :, 6].min() > 0  # plastic everywhere


def _softening_band(length, h, bar=0.6):
    """The width of the zone of the last increments of p in a softening bar with a weaker
    zone, and the stress at the end."""
    mesh = _bar(bar, h, weak=0.05)
    strong = Microfoam(yield_stress=1.0, hardening=-10.0, length=length, **BAR)
    weak = Microfoam(yield_stress=0.98, hardening=-10.0, length=length, **BAR)
    problem = MicrofoamProblem(mesh, {0: strong, 1: weak})
    start, end = _ends(problem, bar)
    result = problem.run([(start, [0, 1, 2], 0.0), (end, 1, 0.0125 * bar)], steps=15, fields=True)
    centre = problem.mesh.nodes[problem.mesh.elements[:, :4]].mean(axis=1)[:, 1]
    dp = (result.states[-1][:, :, 6] - result.states[-4][:, :, 6]).mean(axis=1)
    active = dp > 0.05 * dp.max()
    return centre[active].max() - centre[active].min() + h, result.reactions[-1][1][0] / h**2


@pytest.mark.slow
def test_softening_band_of_the_microfoam_model():
    """Softening (H < 0) localizes plasticity in a band that the length sets: the same
    width on two meshes, twice as wide for twice the length, and the width of the sine arc
    of Appendix A at 5 % of its top, ``2 acos(-0.9) / |ω|`` (0.857 of its wavelength)."""
    tangent = E * -10.0 / (-10.0 + 1.5 * E)
    coarse, stress_coarse = _softening_band(0.01, 0.025)
    fine, stress_fine = _softening_band(0.01, 0.0125)
    wider, _ = _softening_band(0.02, 0.025)
    assert fine == pytest.approx(coarse, rel=0.07)
    assert stress_fine == pytest.approx(stress_coarse, rel=0.03)
    assert wider / coarse == pytest.approx(2.0, rel=0.1)
    arc = 2 * math.acos(-0.9) * characteristic_length(Microfoam(length=0.01, **BAR), tangent)
    assert fine == pytest.approx(arc, rel=0.1)


def _half_width(profile, h):
    """The full width at half maximum of a profile of layers of thickness h (linear
    interpolation at both crossings)."""
    half = 0.5 * profile.max()
    above = np.flatnonzero(profile > half)
    i, j = above[0], above[-1]
    left = (i - 0.5 + (half - profile[i - 1]) / (profile[i] - profile[i - 1])) * h
    right = (j + 0.5 + (profile[j] - half) / (profile[j] - profile[j + 1])) * h
    return right - left


def _damage_band(length, h, kinematics):
    mesh = _bar(1.0, h, weak=0.1)
    kw = dict(young=1000.0, poisson=0.0, length=length, beta=100.0)
    materials = {
        0: MicromorphicDamage(threshold=1e-3, **kw),
        1: MicromorphicDamage(threshold=0.8e-3, **kw),  # damaging alone first
    }
    problem = DamageProblem(mesh, materials, kinematics=kinematics)
    start, end = _ends(problem, 1.0)
    result = problem.run([(start, [0, 1, 2], 0.0), (end, 1, 6e-3)], steps=20)
    centre = problem.mesh.nodes[problem.mesh.elements[:, :4]].mean(axis=1)[:, 1]
    layer = np.floor(centre / h).astype(int)
    kappa = np.bincount(layer, result.state[:, :, 0].mean(axis=1)) / np.bincount(layer)
    stress = np.array([r[1][0] for r in result.reactions]) / h**2
    return _half_width(kappa, h), np.interp([0.5, 0.75], result.loads, stress), kappa.max()


@pytest.mark.slow
def test_damage_band_independent_of_the_mesh():
    """A bar of the damage model with a weaker zone: the band of the history of the
    micromorphic strain (its width at half maximum) and the softening curve are the same
    on two meshes, wider for a longer length, the same in finite strains (strains of a
    few 10⁻³), and the damage reaches nearly 1 (a crack)."""
    band, stress, kappa_max = _damage_band(0.05, 0.025, "small")
    fine, stress_fine, _ = _damage_band(0.05, 0.0125, "small")
    wide, _, _ = _damage_band(0.1, 0.025, "small")
    finite, stress_finite, _ = _damage_band(0.05, 0.025, "finite")
    assert fine == pytest.approx(band, rel=0.08)
    np.testing.assert_allclose(stress_fine, stress, rtol=0.06)
    assert wide > 1.4 * band
    assert finite == pytest.approx(band, rel=0.08)
    np.testing.assert_allclose(stress_finite, stress, rtol=0.15)
    material = MicromorphicDamage(
        young=1000.0, poisson=0.0, threshold=1e-3, length=0.05, beta=100.0
    )
    assert material.damage(kappa_max) > 0.95 and stress[1] < 0.3 * 1000.0 * 1e-3


def _rotation(radians, axis):
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(radians) * cross + (1 - math.cos(radians)) * cross @ cross


def test_microfoam_in_finite_strains():
    """Microfoam in finite strains (logarithmic strains; the objective measures ``χ⁻¹ F``
    and ``χ⁻¹ Grad χ``), micromorphic and classical: the tangents of the elements against
    finite differences (every point plastic), the forces of the body turned by a
    superposed rotation (``x → Q x``, ``χ → Q χ``) turned with it and the same internal
    variables, and the forces of small strains for tiny displacements."""
    rng = np.random.default_rng(3)
    mesh = voxel_tet_mesh(VoxelImage(np.zeros((2, 2, 1), np.uint8), periodic=False))
    q = _rotation(0.9, (1.0, -2.0, 0.5))
    for length in (0.3, None):
        foam = Microfoam(young=100.0, poisson=0.2, yield_stress=0.5, hardening=5.0, F=0.3,
                         length=length, micro=(20.0, 40.0),
                         relative=(200.0, 300.0, 150.0))  # fmt: skip
        problem = MicrofoamProblem(mesh, foam, kinematics="finite")
        space = problem.space
        u = rng.normal(size=space.n_dofs) * 2e-2
        state = problem._initial_state()
        r, data, new = problem._assemble(u, state, True)
        assert (new[:, :, 6] > 0).all()  # every point plastic
        du = rng.normal(size=space.n_dofs) * 2e-2 / 3
        rp, _, _ = problem._assemble(u + 1e-6 * du, state, False)
        rm, _, _ = problem._assemble(u - 1e-6 * du, state, False)
        fd = (rp - rm) / 2e-6
        assert np.linalg.norm(fd - space.matrix(data) @ du) < 1e-6 * np.linalg.norm(fd)

        x = problem.mesh.nodes
        disp = space.offset[:-1, None] + np.arange(3)
        turned = u.copy()
        turned[disp] = (x + u[disp]) @ q.T - x
        micro = space.offset[:-1][space.corner][:, None] + 3 + np.arange(space.n_g)
        if space.n_g:
            chi = u[micro].reshape(-1, 3, 3) + np.eye(3)
            turned[micro] = (q @ chi - np.eye(3)).reshape(-1, 9)
        rq, _, new_q = problem._assemble(turned, state, False)
        scale = np.abs(r).max()
        np.testing.assert_allclose(rq[disp], r[disp] @ q.T, atol=1e-10 * scale)
        if space.n_g:
            forces = r[micro].reshape(-1, 3, 3)
            np.testing.assert_allclose(rq[micro].reshape(-1, 3, 3), q @ forces, atol=1e-10 * scale)
        np.testing.assert_allclose(new_q, new, atol=1e-12)

        tiny = rng.normal(size=space.n_dofs) * 1e-7
        rs, _, _ = MicrofoamProblem(mesh, foam)._assemble(tiny, state, False)
        rf, _, _ = problem._assemble(tiny, state, False)
        assert np.linalg.norm(rf - rs) < 1e-5 * np.linalg.norm(rs)


def test_ductile_tangents():
    """The micromorphic ductile damage model: the radial return of the effective stress
    (rate-independent and viscous, linear and Voce hardening) and its ``∂p/∂ε`` against
    finite differences; the tangents of the elements (not symmetric: the damage by ``χ``,
    the equation of ``χ`` by ``p``) with every point plastic and damage growing, in small
    and finite strains."""
    r2 = math.sqrt(2.0)
    eps = np.array([0.006, -0.002, 0.001, 0.003 * r2, -0.001 * r2, 0.002 * r2])
    epsp0 = np.array([0.001, -0.0005, -0.0005, 0.0, 0.0002, 0.0])
    for viscous in ({}, {"viscosity": 1.0, "exponent": 4.0}):
        material = MicromorphicDuctile(young=200.0, poisson=0.3, yield_stress=0.4, length=0.2,
                                       hardening=10.0, voce=((0.3, 30.0),), **viscous)  # fmt: skip
        par = material.parameters()

        def point(e, par=par):
            sig, _, p, _, _ = point_update(par, e, epsp0, 0.004, 0.01)
            return sig, p

        _, _, p, tan, dp = point_update(par, eps, epsp0, 0.004, 0.01)
        assert p > 0.004
        np.testing.assert_allclose(_fd_tangent(point, eps), tan, atol=1e-6 * np.abs(tan).max())
        fd = [(point(eps + d)[1] - point(eps - d)[1]) / 2e-7 for d in 1e-7 * np.eye(6)]
        np.testing.assert_allclose(fd, dp, atol=1e-6 * np.abs(dp).max())

    rng = np.random.default_rng(2)
    mesh = voxel_tet_mesh(VoxelImage(np.zeros((2, 2, 1), np.uint8), periodic=False))
    material = MicromorphicDuctile(young=200.0, poisson=0.3, yield_stress=0.4, length=0.3,
                                   hardening=10.0, onset=0.001, critical=0.02)  # fmt: skip
    for kinematics in ("small", "finite"):
        problem = DuctileProblem(mesh, material, kinematics=kinematics)
        space = problem.space
        u = rng.normal(size=space.n_dofs) * 2e-2
        chi = space.offset[:-1][space.corner] + 3
        u[chi] = 0.01 + rng.random(len(chi)) * 0.01  # above the history: damage grows
        state = problem._initial_state()
        state[:, :, 7] = 0.005
        _, data, new = problem._assemble(u, state, True)
        du = rng.normal(size=space.n_dofs) * 2e-2 / 3
        rp, _, _ = problem._assemble(u + 1e-7 * du, state, False)
        rm, _, _ = problem._assemble(u - 1e-7 * du, state, False)
        fd = (rp - rm) / 2e-7
        assert np.linalg.norm(fd - space.matrix(data) @ du) < 1e-6 * np.linalg.norm(fd)
        assert (new[:, :, 6] > 0).all() and (new[:, :, 8] > 0).all()


TRIAXIAL = {
    "rice_tracey": {"triaxiality": "rice_tracey"},
    "johnson_cook": {"triaxiality": "johnson_cook", "triaxiality_parameters": (0.05, 3.44, -2.12)},
    "lemaitre": {"triaxiality": "lemaitre"},
}


def test_triaxiality_of_the_ductile_damage():
    """The damage driven by the plastic strain weighted by the triaxiality: ``Δe = w(η) Δp``
    at the end of the step for each law (``w(1/3) = 1`` in uniaxial tension, ``w(0)`` in
    pure shear: 0.61 for Rice & Tracey, 0.50 for Johnson & Cook's steel, 0.87 for
    Lemaitre), ``∂e/∂ε`` (with ``∂η/∂ε``) against finite differences, and the tangents of
    the elements in small and finite strains."""
    r2 = math.sqrt(2.0)
    eps = np.array([0.006, -0.002, 0.004, 0.003 * r2, -0.001 * r2, 0.002 * r2])
    epsp0 = np.array([0.001, -0.0005, -0.0005, 0.0, 0.0002, 0.0])
    shear = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01 * r2])
    expected_shear = {"rice_tracey": math.exp(-0.5), "johnson_cook": 0.5005, "lemaitre": 2.6 / 3}
    for name, law in TRIAXIAL.items():
        material = MicromorphicDuctile(young=200.0, poisson=0.3, yield_stress=0.4, length=0.2,
                                       hardening=10.0, **law)  # fmt: skip
        par = material.parameters()
        assert material.weight(1.0 / 3.0) == pytest.approx(1.0)
        e, eta, de = driving_update(par, eps, epsp0, 0.004, 0.01, 0.01)
        p = point_update(par, eps, epsp0, 0.004, 0.01)[2]
        assert e - 0.01 == pytest.approx(material.weight(eta) * (p - 0.004), rel=1e-12)
        fd = [(driving_update(par, eps + d, epsp0, 0.004, 0.01, 0.01)[0]
               - driving_update(par, eps - d, epsp0, 0.004, 0.01, 0.01)[0]) / 2e-7
              for d in 1e-7 * np.eye(6)]  # fmt: skip
        np.testing.assert_allclose(fd, de, atol=1e-6 * np.abs(de).max())
        e_shear, eta_shear, _ = driving_update(par, shear, np.zeros(6), 0.0, 0.0, 0.01)
        p_shear = point_update(par, shear, np.zeros(6), 0.0, 0.01)[2]
        assert abs(eta_shear) < 1e-12
        assert e_shear / p_shear == pytest.approx(expected_shear[name], rel=1e-3)
    with pytest.raises(ValueError, match="triaxiality"):
        MicromorphicDuctile(young=1.0, poisson=0.3, yield_stress=1.0, length=1.0, triaxiality="x")

    rng = np.random.default_rng(3)
    mesh = voxel_tet_mesh(VoxelImage(np.zeros((2, 2, 1), np.uint8), periodic=False))
    material = MicromorphicDuctile(young=200.0, poisson=0.3, yield_stress=0.4, length=0.3,
                                   hardening=10.0, onset=0.001, critical=0.02,
                                   **TRIAXIAL["rice_tracey"])  # fmt: skip
    for kinematics in ("small", "finite"):
        problem = DuctileProblem(mesh, material, kinematics=kinematics)
        space = problem.space
        u = rng.normal(size=space.n_dofs) * 2e-2
        chi = space.offset[:-1][space.corner] + 3
        u[chi] = 0.01 + rng.random(len(chi)) * 0.01
        state = problem._initial_state()
        state[:, :, 7] = 0.005
        _, data, new = problem._assemble(u, state, True)
        du = rng.normal(size=space.n_dofs) * 2e-2 / 3
        rp, _, _ = problem._assemble(u + 1e-7 * du, state, False)
        rm, _, _ = problem._assemble(u - 1e-7 * du, state, False)
        fd = (rp - rm) / 2e-7
        assert np.linalg.norm(fd - space.matrix(data) @ du) < 1e-6 * np.linalg.norm(fd)
        assert (problem.driving(new) > 0).all()


def _ductile_band(length, h, kinematics, **triaxiality):
    mesh = _bar(1.0, h, weak=0.1)
    kw = dict(young=200.0, poisson=0.0, length=length, hardening=10.0, onset=0.002, critical=0.02,
              **triaxiality)  # fmt: skip
    materials = {
        0: MicromorphicDuctile(yield_stress=0.4, **kw),
        1: MicromorphicDuctile(yield_stress=0.36, **kw),  # damaging before the rest yields
    }
    problem = DuctileProblem(mesh, materials, kinematics=kinematics)
    start, end = _ends(problem, 1.0)
    result = problem.run([(start, [0, 1, 2], 0.0), (end, 1, 0.02)], steps=16)
    centre = problem.mesh.nodes[problem.mesh.elements[:, :4]].mean(axis=1)[:, 1]
    layer = np.floor(centre / h).astype(int)
    kappa = np.bincount(layer, result.state[:, :, 7].mean(axis=1)) / np.bincount(layer)
    damage = np.bincount(layer, problem.damage(result.state)) / np.bincount(layer)
    stress = np.array([r[1][0] for r in result.reactions]) / h**2
    curve = np.interp([0.5, 0.75, 1.0], result.loads, stress)
    return _half_width(kappa, h), curve, damage.max(), stress.max()


@pytest.mark.slow
def test_ductile_band_independent_of_the_mesh():
    """Ductile damage driven by the micromorphic cumulated plastic strain, in a bar with a
    weaker zone: plasticity and damage localize in a band whose width (that of ``κ`` at half
    maximum) and softening curve are the same on two meshes, nearly twice as wide for twice
    the length, nearly the same in finite strains (strains of 30 % in the band), and whose
    damage reaches ``d_max``: a crack across the bar, which no longer carries the load."""
    band, curve, damage, peak = _ductile_band(0.05, 0.025, "small")
    fine, curve_fine, _, _ = _ductile_band(0.05, 0.0125, "small")
    wide, _, _, _ = _ductile_band(0.1, 0.025, "small")
    finite, curve_finite, _, _ = _ductile_band(0.05, 0.025, "finite")
    assert fine == pytest.approx(band, rel=0.08)
    np.testing.assert_allclose(curve_fine, curve, rtol=0.06)
    assert wide > 1.6 * band
    assert finite == pytest.approx(band, rel=0.1)
    np.testing.assert_allclose(curve_finite[:2], curve[:2], rtol=0.1)
    assert damage > 0.98 and curve[-1] < 0.12 * peak
    # uniaxial tension (η = 1/3): the weight of Rice & Tracey is 1, the same band and curve
    tension, curve_tension, _, _ = _ductile_band(0.05, 0.025, "small", **TRIAXIAL["rice_tracey"])
    assert tension == pytest.approx(band, rel=0.05)
    np.testing.assert_allclose(curve_tension, curve, rtol=0.05)
