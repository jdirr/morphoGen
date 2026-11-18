"""Finite elements on tetrahedral meshes: exact cases, bounds, voids, solvers."""

import logging

import numpy as np
import pytest

import morphogen as mg
from helpers import backus, box_mesh
from morphogen.homogenization import (
    VOID,
    FEElasticProblem,
    FEThermalProblem,
    FFTElasticProblem,
    Loading,
    Material,
    apparent_bulk_modulus,
    apparent_conductivity,
    apparent_shear_modulus,
    bounds,
    conductivity_tensor,
    homogenize,
    isotropic_projection,
    isotropy_errors,
    make_problem,
    stiffness_tensor,
)
from morphogen.homogenization.apparent import bulk_isotropy, shear_isotropy, thermal_isotropy
from morphogen.homogenization.bounds import (
    hashin_shtrikman_bulk,
    hashin_shtrikman_conductivity,
    hashin_shtrikman_shear,
    reuss,
    voigt,
)
from morphogen.homogenization.elements import isotropic_stiffness, tensor_to_voigt, voigt_to_tensor
from morphogen.homogenization.fe import assemble, linear_interpolation, periodic_map
from morphogen.homogenization.materials import bulk_shear, isotropic_matrix, lame, young_poisson
from morphogen.mesh.tetra import TetMesh

SOFT = Material(young=1.0, poisson=0.2, conductivity=1.0)
STIFF = Material(young=20.0, poisson=0.35, conductivity=30.0)
MATS = {0: SOFT, 1: STIFF}


def random_phases(seed, fraction=0.4):
    rng = np.random.default_rng(seed)
    return lambda c: (rng.random(len(c)) < fraction).astype(int)


def test_material_conversions_and_errors():
    lam, mu = lame(210.0, 0.3)
    k, mu2 = bulk_shear(210.0, 0.3)
    assert mu == pytest.approx(mu2) and k == pytest.approx(lam + 2 * mu / 3)
    assert young_poisson(k, mu) == pytest.approx((210.0, 0.3))
    m = Material.from_config({"bulk": k, "shear": mu, "conductivity": 5.0})
    assert m.young == pytest.approx(210.0) and m.poisson == pytest.approx(0.3)
    assert Material.from_config("void") is VOID
    assert Material.from_config({"young": 1, "poisson": 0.25}).to_config() == {
        "young": 1.0,
        "poisson": 0.25,
    }
    for bad in (
        {"young": 1.0},
        {"young": -1.0, "poisson": 0.2},
        {"young": 1.0, "poisson": 0.5},
        {"conductivity": 0.0},
        {"young": 1, "poisson": 0.2, "bulk": 1},
        {"colour": 1},
        "plastic",
    ):
        with pytest.raises(ValueError):
            Material.from_config(bad)


@pytest.mark.parametrize("order", [1, 2])
def test_element_stiffness(order):
    nodes = np.array([[0.1, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 0.9, 0.1], [0.3, 0.2, 1.1]])
    mesh = TetMesh(nodes, [[0, 1, 2, 3]], [0], nodes.min(0), nodes.max(0))
    if order == 2:
        mesh = mesh.quadratic()
    one = np.zeros(mesh.n_elements, dtype=np.int64)
    k = assemble(mesh, 3, one, isotropic_matrix(1.3, 0.7)[None]).toarray()
    assert np.allclose(k, k.T)
    eig = np.linalg.eigvalsh(k)
    assert np.all(np.abs(eig[:6]) < 1e-12 * eig[-1]) and eig[6] > 1e-6 * eig[-1]
    x = mesh.nodes
    rotation = np.column_stack([-x[:, 1], x[:, 0], 0 * x[:, 0]]).ravel()
    assert np.abs(k @ rotation).max() < 1e-12 * np.abs(k).max()
    kt = assemble(mesh, 1, one, 2.0 * np.eye(3)[None]).toarray()
    np.testing.assert_allclose(kt @ np.ones(len(x)), 0.0, atol=1e-13)
    # stiffness of the linear part: P^T K2 P = K1 (exact integration)
    if order == 2:
        p, _ = linear_interpolation(mesh)
        k1 = assemble(mesh.linear(), 1, one, 2.0 * np.eye(3)[None]).toarray()
        np.testing.assert_allclose(p.T @ kt @ p, k1, atol=1e-12 * np.abs(k1).max())
    t = voigt_to_tensor(np.array([0.1, -0.2, 0.3, 0.05, -0.07, 0.11]), strain=True)
    assert t[0, 1] == pytest.approx(0.055)
    np.testing.assert_allclose(tensor_to_voigt(t, strain=True), [0.1, -0.2, 0.3, 0.05, -0.07, 0.11])


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("bc", ["kubc", "mixed", "periodic", "subc"])
def test_homogeneous_medium_is_exact(bc, order):
    mesh = box_mesh(order=order, perturb=0.25, seed=1)
    elastic = FEElasticProblem(mesh, MATS, bc, solver="direct")
    assert apparent_bulk_modulus(elastic).value == pytest.approx(SOFT.bulk, rel=1e-11)
    for plane in ((0, 1), (1, 2), (0, 2)):
        res = apparent_shear_modulus(elastic, plane)
        assert res.value == pytest.approx(SOFT.shear, rel=1e-11)
        assert res.energy == pytest.approx(SOFT.shear, rel=1e-11)
    thermal = FEThermalProblem(mesh, MATS, bc, solver="direct")
    for axis in range(3):
        res = apparent_conductivity(thermal, axis)
        assert res.value == pytest.approx(1.0, rel=1e-11)
        assert res.isotropy < 1e-11


@pytest.mark.parametrize("order", [1, 2])
def test_periodic_laminate_is_exact(order):
    mesh = box_mesh(
        (3, 3, 10),
        upper=(1.0, 1.3, 1.0),
        phase=lambda c: ((c[:, 2] > 0.3) & (c[:, 2] < 0.7)).astype(int),
        order=order,
    )
    exact = backus([isotropic_stiffness(*m.lame) for m in (SOFT, STIFF)], [0.6, 0.4])
    c, _ = stiffness_tensor(FEElasticProblem(mesh, MATS, "periodic", solver="direct"))
    np.testing.assert_allclose(c, exact, rtol=1e-10, atol=1e-12 * np.abs(exact).max())
    kappa, _ = conductivity_tensor(FEThermalProblem(mesh, MATS, "periodic", solver="direct"))
    parallel = 0.6 * 1.0 + 0.4 * 30.0
    series = 1.0 / (0.6 / 1.0 + 0.4 / 30.0)
    np.testing.assert_allclose(np.diag(kappa), [parallel, parallel, series], rtol=1e-10)
    # SUBC on the same laminate: the layers are in series along x3 for any loading
    c_subc, _ = stiffness_tensor(FEElasticProblem(mesh, MATS, "subc", solver="direct"))
    assert c_subc[2, 2] <= c[2, 2] * (1 + 1e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_energy_hierarchy_and_bounds(order):
    mesh = box_mesh((4, 4, 4), upper=(1.0, 1.0, 1.0), phase=random_phases(1), order=order)
    b = bounds(mesh.phase_fractions(), MATS)
    k = {}
    for bc in ("kubc", "periodic", "mixed", "subc"):
        res = apparent_bulk_modulus(FEElasticProblem(mesh, MATS, bc, solver="direct"))
        k[bc] = res.value
        # Hill's lemma: stress and energy estimates coincide
        assert res.energy == pytest.approx(res.value, rel=1e-9)
    kb = b["bulk_modulus"]
    eps = 1e-12
    assert kb["reuss"] <= k["subc"] * (1 + eps)
    assert k["subc"] <= k["periodic"] * (1 + eps) and k["subc"] <= k["mixed"] * (1 + eps)
    assert k["periodic"] <= k["kubc"] * (1 + eps) and k["mixed"] <= k["kubc"] * (1 + eps)
    assert k["kubc"] <= kb["voigt"] * (1 + eps)
    c = {}
    for bc in ("kubc", "periodic", "mixed", "subc"):
        c[bc] = apparent_conductivity(FEThermalProblem(mesh, MATS, bc, solver="direct")).value
    cb = b["conductivity"]
    assert cb["reuss"] <= c["subc"] <= c["periodic"] <= c["kubc"] <= cb["voigt"]
    assert c["subc"] <= c["mixed"] <= c["kubc"]


def test_volume_averages_and_fields():
    mesh = box_mesh((4, 3, 3), phase=random_phases(3, 0.5), order=2, perturb=0.2)
    problem = FEElasticProblem(mesh, MATS, "kubc", solver="direct")
    strain = np.array([[0.2, 0.05, 0.0], [0.05, -0.1, 0.03], [0.0, 0.03, 0.15]])
    sol = problem.solve(Loading.strain(strain), fields=True)
    vol = mesh.volumes()
    # with KUBC the average strain is exactly the imposed one
    mean_strain = (sol.fields["strain"] * vol[:, None]).sum(axis=0) / mesh.box_volume
    np.testing.assert_allclose(voigt_to_tensor(mean_strain, strain=False), strain, atol=1e-13)
    phase_mean = sum(p.fraction * p.mean["strain"] for p in sol.phases)
    np.testing.assert_allclose(phase_mean, mean_strain, atol=1e-13)
    mean_stress = (sol.fields["stress"] * vol[:, None]).sum(axis=0) / mesh.box_volume
    np.testing.assert_allclose(mean_stress, tensor_to_voigt(sol.stress, strain=False), rtol=1e-12)
    assert sol.energy == pytest.approx(np.tensordot(sol.stress, sol.strain), rel=1e-10)
    assert (sol.fields["energy"] * vol).sum() / mesh.box_volume == pytest.approx(sol.energy)
    assert sol.fields["displacement"].shape == (mesh.n_nodes, 3)
    assert sol.support is problem.mesh
    assert sum(p.fraction for p in sol.phases) == pytest.approx(1.0)
    assert all(np.all(p.std["stress"] >= 0) for p in sol.phases)
    thermal = FEThermalProblem(mesh, MATS, "periodic", solver="direct")
    tsol = thermal.solve(Loading.gradient([0.0, 1.0, 0.0]), fields=True)
    mean_flux = (tsol.fields["flux"] * vol[:, None]).sum(axis=0) / mesh.box_volume
    np.testing.assert_allclose(mean_flux, tsol.stress, rtol=1e-12, atol=1e-14)
    assert (tsol.fields["dissipation"] * vol).sum() / mesh.box_volume == pytest.approx(tsol.energy)
    assert tsol.fields["temperature"].shape == (mesh.n_nodes,)


def slab_with_island(order=1):
    """A slab of phase 1 spanning the box, an island of phase 1 inside phase 0."""

    def phase(c):
        slab = (c[:, 2] > 0.25) & (c[:, 2] < 0.5)
        island = np.all(np.abs(c - [0.5, 0.5, 0.75]) < 0.1, axis=1)
        return (slab | island).astype(int)

    return box_mesh((8, 8, 8), upper=(1.0, 1.0, 1.0), phase=phase, order=order)


def test_porous_media_floating_clusters_and_subc():
    mesh = slab_with_island()
    mats = {0: VOID, 1: STIFF}
    props = ["bulk_modulus", "shear_modulus", "conductivity"]
    a = homogenize(mesh, mats, bcs=["kubc", "mixed", "periodic"], properties=props)
    # meshing only the solid gives the same results; the island carries no load
    centres = mesh.nodes[mesh.elements].mean(axis=1)
    slab = mesh.subset([1])
    slab_only = TetMesh(
        mesh.nodes,
        mesh.elements[(mesh.phase == 1) & (centres[:, 2] < 0.6)],
        np.ones(int(((mesh.phase == 1) & (centres[:, 2] < 0.6)).sum()), int),
        mesh.lower,
        mesh.upper,
        periodic=True,
        metadata={"phases": "solid"},
    ).subset([1])
    b = homogenize(slab_only, mats, bcs=["kubc", "mixed", "periodic"], properties=props)
    for key, res in a.items():
        assert res.value == pytest.approx(b[key].value, rel=1e-9), key
        assert res.energy == pytest.approx(res.value, rel=1e-8), key
    problem = FEElasticProblem(slab, mats, "kubc")
    assert problem.floating_fraction > 0
    # a slab normal to x3: zero stiffness along x3 under PBC, as a void layer cuts it
    c, _ = stiffness_tensor(FEElasticProblem(slab, mats, "periodic", solver="direct"))
    assert abs(c[2, 2]) < 1e-6 * c[0, 0]
    with pytest.raises(ValueError, match="porous"):
        FEElasticProblem(mesh, mats, "subc")
    with pytest.raises(ValueError, match="porous"):
        FEThermalProblem(slab, mats, "uhf")


def test_loading_and_boundary_condition_errors():
    mesh = box_mesh((2, 2, 2), phase=random_phases(0))
    with pytest.raises(ValueError, match="unknown boundary condition"):
        FEElasticProblem(mesh, MATS, "free")
    problem = FEElasticProblem(mesh, MATS, "mbc", solver="direct")
    assert problem.bc == "mixed"
    with pytest.raises(ValueError, match="mixed boundary conditions"):
        problem.solve(Loading.strain([1, 0, 0, 0, 1, 1]))
    with pytest.raises(ValueError, match="strain"):
        problem.solve(Loading.stress(np.eye(3)))
    with pytest.raises(ValueError, match="one axis"):
        FEThermalProblem(mesh, MATS, "mtbc").solve(Loading.gradient([1, 1, 0]))
    with pytest.raises(ValueError, match="no material"):
        FEElasticProblem(mesh, {1: STIFF}, "kubc")
    with pytest.raises(ValueError, match="no elastic"):
        apparent_bulk_modulus(
            FEElasticProblem(mesh, {0: SOFT, 1: Material(conductivity=1.0)}, "kubc")
        )
    with pytest.raises(ValueError, match="kubc, periodic or subc"):
        stiffness_tensor(problem)
    with pytest.raises(ValueError):
        Loading("strain", [1, 2, 3])
    with pytest.raises(TypeError, match="TetMesh"):
        FEElasticProblem(mg.VoxelImage(np.ones((2, 2, 2), np.uint8)), MATS)
    with pytest.raises(ValueError, match="solver"):
        apparent_bulk_modulus(FEElasticProblem(mesh, MATS, "kubc", solver="magic"))


def test_problem_dispatch():
    mesh = box_mesh((2, 2, 2))
    image = mg.VoxelImage(np.zeros((3, 3, 3), np.uint8), periodic=True)
    assert isinstance(make_problem(mesh, MATS, "elastic", "kubc"), FEElasticProblem)
    assert isinstance(make_problem(image, MATS, "elastic"), FFTElasticProblem)
    with pytest.raises(ValueError, match="conforming mesh"):
        homogenize(image, MATS, bcs=["kubc"])
    with pytest.raises(TypeError, match="TetMesh"):
        make_problem(np.zeros(3), MATS, "elastic")
    with pytest.raises(ValueError, match="physics"):
        make_problem(mesh, MATS, "acoustic")


def test_isotropy_criteria_and_tensor_tools():
    assert bulk_isotropy(np.eye(3)) == 0.0
    shear = np.zeros((3, 3))
    shear[0, 1] = shear[1, 0] = 1.0
    assert shear_isotropy(shear) == 0.0
    t = np.diag([1.0, 1.0, 1.0]) + 0.1 * shear
    assert bulk_isotropy(t) == pytest.approx(np.sqrt(0.02 / 3.0))
    assert thermal_isotropy([2.0, -0.1, 0.3]) == pytest.approx(0.2)
    lam, mu = STIFF.lame
    c = isotropic_stiffness(lam, mu)
    assert isotropic_projection(c) == pytest.approx((STIFF.bulk, STIFF.shear))
    assert max(isotropy_errors(c).values()) < 1e-12
    c[0, 0] *= 1.1
    assert isotropy_errors(c)["normal"] > 0.03


def test_bounds_of_dirrenberger_2014_table4():
    # porous network of steel fibres (V_V = 16 %): bounds of Table 4
    steel = Material(young=210e3, poisson=0.3, conductivity=100.0)
    b = bounds({0: 0.84, 1: 0.16}, {0: VOID, 1: steel})
    assert b["bulk_modulus"]["hs_upper"] == pytest.approx(11839, rel=1e-3)
    assert b["shear_modulus"]["hs_upper"] == pytest.approx(7328, rel=1e-3)
    assert b["conductivity"]["hs_upper"] == pytest.approx(11.27, rel=1e-3)
    assert b["bulk_modulus"]["voigt"] == pytest.approx(28000, rel=1e-3)
    assert b["shear_modulus"]["voigt"] == pytest.approx(12923, rel=1e-3)
    assert b["conductivity"]["voigt"] == pytest.approx(16.00, rel=1e-3)
    assert b["bulk_modulus"]["hs_lower"] == 0.0 and b["shear_modulus"]["reuss"] == 0.0
    # two-phase checks: HS between Reuss and Voigt; equal phases give the phase value
    f, k, mu = [0.3, 0.7], [10.0, 2.0], [5.0, 1.0]
    lo, hi = hashin_shtrikman_bulk(f, k, mu)
    assert reuss(f, k) <= lo <= hi <= voigt(f, k)
    lo, hi = hashin_shtrikman_shear(f, k, mu)
    assert reuss(f, mu) <= lo <= hi <= voigt(f, mu)
    assert hashin_shtrikman_conductivity([0.5, 0.5], [3.0, 3.0]) == pytest.approx((3.0, 3.0))
    with pytest.raises(ValueError):
        voigt([0.5, 0.6], [1.0, 2.0])


@pytest.mark.parametrize("order", [1, 2])
def test_iterative_solvers_match_direct(order):
    mesh = box_mesh((7, 6, 6), upper=(1.0, 1.0, 1.0), phase=random_phases(4), order=order)
    props = ["bulk_modulus", "conductivity"]
    ref = homogenize(mesh, MATS, bcs=["kubc", "periodic"], properties=props, solver="direct")
    methods = ["cg", "amg"]
    for method in methods:
        res = homogenize(
            mesh, MATS, bcs=["kubc", "periodic"], properties=props, solver=method, tol=1e-10
        )
        for key, r in ref.items():
            assert res[key].value == pytest.approx(r.value, rel=1e-7), (method, key)
            info = res[key].solution.info
            assert info.iterations > 0 and info.residual < 1e-9
            if method == "amg" and order == 2:
                assert info.method == "amg-2level"
                assert info.iterations < 60


def test_homogenize_tensors():
    mesh = box_mesh((3, 3, 3), phase=random_phases(6))
    out = homogenize(
        mesh,
        MATS,
        bcs=["periodic", "mixed"],
        properties=["stiffness", "conductivity_tensor"],
        solver="direct",
    )
    assert set(out) == {"stiffness.periodic", "conductivity_tensor.periodic"}
    c = out["stiffness.periodic"]["tensor"]
    assert np.allclose(c, c.T) and np.all(np.linalg.eigvalsh(c) > 0)
    k_proj = out["stiffness.periodic"]["bulk_modulus"]
    assert STIFF.bulk > k_proj > SOFT.bulk


@pytest.mark.parametrize("order", [1, 2])
def test_against_scikit_fem(order):
    skfem = pytest.importorskip("skfem")
    from skfem.helpers import ddot, dot, eye, grad, sym_grad, trace

    upper = (0.9, 1.2, 1.1)
    mesh = box_mesh((3, 4, 3), upper=upper, phase=random_phases(2, 0.5), perturb=0.3, seed=2)
    strain = np.array([[0.3, 0.1, -0.05], [0.1, -0.2, 0.07], [-0.05, 0.07, 0.15]])
    gradient = np.array([0.3, -0.7, 0.2])
    fe_mesh = mesh.quadratic() if order == 2 else mesh
    ours = FEElasticProblem(fe_mesh, MATS, "kubc", solver="direct").solve(Loading.strain(strain))
    ours_t = FEThermalProblem(fe_mesh, MATS, "kubc", solver="direct").solve(
        Loading.gradient(gradient)
    )
    m = skfem.MeshTet(mesh.nodes.T.copy(), mesh.elements.T.copy())
    element = skfem.ElementTetP1() if order == 1 else skfem.ElementTetP2()
    lam = np.array([MATS[p].lame[0] for p in mesh.phase])
    mu = np.array([MATS[p].lame[1] for p in mesh.phase])
    cond = np.array([MATS[p].conductivity for p in mesh.phase])
    basis = skfem.Basis(m, skfem.ElementVector(element), intorder=2 * order)
    dg = basis.with_element(skfem.ElementTetP0())

    @skfem.BilinearForm
    def elasticity(u, v, w):
        e = sym_grad(u)
        return ddot(eye(w.lam * trace(e), 3) + 2 * w.mu * e, sym_grad(v))

    @skfem.BilinearForm
    def conduction(u, v, w):
        return w.k * dot(grad(u), grad(v))

    volume = mesh.box_volume
    k = elasticity.assemble(basis, lam=dg.interpolate(lam), mu=dg.interpolate(mu))
    affine = strain @ basis.doflocs  # component c of E x at the location of each dof
    u = np.zeros(basis.N)
    for c in range(3):
        dofs = [basis.nodal_dofs[c]] + ([basis.edge_dofs[c]] if order == 2 else [])
        for d in dofs:
            u[d] = affine[c, d]
    boundary = basis.get_dofs().all()
    sol = skfem.solve(*skfem.condense(k, np.zeros(basis.N), x=u, D=boundary))
    assert ours.energy == pytest.approx(sol @ (k @ sol) / volume, rel=1e-10)
    sbasis = skfem.Basis(m, element, intorder=2 * order)
    kt = conduction.assemble(sbasis, k=sbasis.with_element(skfem.ElementTetP0()).interpolate(cond))
    t = sbasis.doflocs.T @ gradient
    tsol = skfem.solve(*skfem.condense(kt, np.zeros(sbasis.N), x=t, D=sbasis.get_dofs().all()))
    assert ours_t.energy == pytest.approx(tsol @ (kt @ tsol) / volume, rel=1e-10)


def test_samples_without_load_carrying_material():
    # an empty sample, or a solid island touching no face, has zero apparent properties
    mats = {0: VOID, 1: STIFF}
    void = box_mesh((2, 2, 2))
    island = box_mesh(
        (6, 6, 6),
        upper=(1.0, 1.0, 1.0),
        phase=lambda c: np.all(np.abs(c - 0.5) < 0.2, axis=1).astype(int),
    )
    for mesh, bcs in ((void, ["kubc", "mixed"]), (island, ["periodic", "kubc"])):
        props = ["bulk_modulus", "shear_modulus", "conductivity"]
        res = homogenize(mesh, mats, bcs=bcs, properties=props, fields=True)
        assert len(res) == 6
        for key, r in res.items():
            assert r.value == 0.0 and r.energy == 0.0, key
            assert r.solution.info.n_dofs == 0 and np.isnan(r.isotropy)


def test_periodic_conditions_on_non_matching_faces(caplog):
    mesh = box_mesh((4, 4, 4), upper=(1.0, 1.0, 1.0))
    assert mesh.matching_faces()
    _, _, counts = periodic_map(mesh)
    assert counts["interpolated"] == 0 and counts["matched"] == 5**3 - 4**3  # upper faces
    # move the inner nodes of the face x = 1 within the face
    nodes = mesh.nodes.copy()
    face = (np.abs(nodes[:, 0] - 1.0) < 1e-12) & np.all(
        (nodes[:, 1:] > 1e-9) & (nodes[:, 1:] < 1 - 1e-9), axis=1
    )
    nodes[face, 1:] += np.random.default_rng(0).uniform(-0.06, 0.06, (int(face.sum()), 2))
    moved = TetMesh(nodes, mesh.elements, mesh.phase, mesh.lower, mesh.upper, periodic=True)
    assert not moved.matching_faces()
    t, _, counts = periodic_map(moved)
    assert counts["interpolated"] == int(face.sum()) and counts["outside"] == 0
    np.testing.assert_allclose(np.asarray(t.sum(axis=1)).ravel(), 1.0)
    with caplog.at_level(logging.WARNING, logger="morphogen"):
        k = apparent_bulk_modulus(FEElasticProblem(moved, MATS, "periodic", solver="direct"))
    assert "non-matching faces" in caplog.text
    # not exact (the patch test fails), but close for a homogeneous medium
    assert k.value == pytest.approx(SOFT.bulk, rel=0.02)


def test_periodic_conditions_on_porous_faces_that_differ():
    """A lattice of bars (voxels, its pores void) translated by two voxels is the same
    periodic medium, on the same Kuhn triangulation: its material then meets pores of the
    next cell on the upper faces, whose nodes are free (a free surface), not tied to the
    lower face; the moduli and the buckling loads are those of the cell before."""
    from morphogen.homogenization.bifurcation import post_bifurcation
    from morphogen.homogenization.bloch import voxel_tet_mesh
    from morphogen.image import VoxelImage

    x = np.arange(6)
    inside = (x >= 2) & (x < 4)
    a, b, c = np.meshgrid(inside, inside, inside, indexing="ij")
    bars = ((a & b) | (b & c) | (a & c)).astype(np.uint8)
    mats = {0: VOID, 1: SOFT}
    out = []
    for shift in (0, 2):
        image = VoxelImage(np.roll(bars, shift, axis=0), periodic=True)
        mesh = voxel_tet_mesh(image, phases=[1])
        t, _, counts = periodic_map(mesh)
        assert counts["interpolated"] == 0 and counts["free"] == (18 if shift else 0)
        np.testing.assert_allclose(np.asarray(t.sum(axis=1)).ravel(), 1.0)
        k = apparent_bulk_modulus(FEElasticProblem(mesh, mats, "periodic", solver="direct"))
        b = post_bifurcation(image, -np.eye(3), materials=mats, phases=[1], patterns="critical")
        out.append((k.value, *b.first[::2]))
    np.testing.assert_allclose(out[1], out[0], rtol=1e-9)
    # two voxels that meet across an edge of the box, through the periods: the nodes of
    # their common edge, (4, 0, z) and (0, 4, z), are one node of the periodic medium
    data = np.zeros((4, 4, 2), np.uint8)
    data[3, 0] = data[0, 3] = 1
    mesh = voxel_tet_mesh(VoxelImage(data, periodic=True), phases=[1])
    _, masters, counts = periodic_map(mesh)
    wrapped = np.mod(mesh.nodes[masters], mesh.size)
    wrapped[np.isclose(wrapped, mesh.size)] = 0.0
    assert len(np.unique(np.round(wrapped, 9), axis=0)) == len(masters)  # one per place
    assert counts["free"] > 0 and counts["interpolated"] == 0
