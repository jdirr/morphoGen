"""Linear elastic fracture mechanics of 3D cracks: the level sets of cracks, the cutting of
tetrahedra, the energy release rate of an edge-cracked plate by XFEM (against the
handbook factor of Tada, Paris & Irwin) and on conforming meshes (converging with their
refinement, and within a percent with 10-node elements whose mid-edge nodes stand at the
quarter of the edges that leave the front), the remeshing by MMG, and fatigue propagation
by Paris' law."""

import math

import numpy as np
import pytest

from morphogen.fracture import XFEM, Crack, lefm, propagate, release_rate, sent_factor
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage

W, H = 16.0, 32.0
PLATE = Material(young=1000.0, poisson=0.0)  # nu = 0: no difference of plane stress and strain


def _plate(h=1.0):
    shape = (round(W / h), round(H / h), 2)
    return voxel_tet_mesh(
        VoxelImage(np.zeros(shape, np.uint8), spacing=(h, h, 1.0), periodic=False)
    )


def _forces(m):
    top = lefm.face_forces(m, lambda c: c[:, 1] > H - 1e-9, [0, 1.0, 0])
    return top + lefm.face_forces(m, lambda c: c[:, 1] < 1e-9, [0, -1.0, 0])


def _supports(m):
    x = m.nodes
    corner = (np.abs(x[:, 0] - W) < 1e-9) & (x[:, 2] < 1e-9)
    a = np.flatnonzero(corner & (x[:, 1] < 1e-9))[0]
    b = np.flatnonzero(corner & (x[:, 1] > H - 1e-9))[0]
    return np.concatenate([[3 * a, 3 * a + 1, 3 * b], 3 * np.flatnonzero(x[:, 2] < 1e-9) + 2])


def _exact(a):
    return (math.sqrt(math.pi * a) * sent_factor(a / W)) ** 2 / PLATE.young


def test_cracks_and_cut_tetrahedra():
    straight = Crack("straight", (0, 2, 0), (0, 2, 0), 3.0, direction=(1, 1, 0))
    assert straight.normal == (0.0, 1.0, 0.0) and straight.direction == (1.0, 0.0, 0.0)
    x = np.array([[1.0, 2.5, 7.0], [4.0, 1.0, 0.0]])
    np.testing.assert_allclose(straight.phi(x), [0.5, -1.0])
    np.testing.assert_allclose(straight.psi(x), [-2.0, 1.0])
    assert straight.grown(1.5).size == 4.5
    penny = Crack("penny", (0, 0, 0), (0, 0, 1), 2.0)
    np.testing.assert_allclose(penny.psi(np.array([[3.0, 4.0, 9.0]])), [3.0])
    with pytest.raises(ValueError):
        Crack("straight", (0, 0, 0), (0, 0, 1), 1.0)
    # a tetrahedron cut by a plane: sub-tetrahedra on both sides filling its volume
    tet = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for phi in ([-0.3, 0.2, 0.2, 0.2], [-0.3, -0.1, 0.2, 0.4], [0.5, 0.5, 0.5, 0.5]):
        subs, sides = lefm._split(tet, np.array(phi))
        vols = [
            abs(np.linalg.det(np.stack([t[1] - t[0], t[2] - t[0], t[3] - t[0]]))) / 6 for t in subs
        ]
        assert sum(vols) == pytest.approx(1 / 6)
        for t, s in zip(subs, sides, strict=True):  # each on one side of the plane
            centre = t.mean(axis=0)
            weights = np.linalg.solve(np.vstack([tet.T, np.ones(4)]), np.append(centre, 1.0))
            assert np.sign(weights @ np.array(phi)) == s
    refined = lefm._refine(tet)
    assert len(refined) == 8 and all(
        abs(np.linalg.det(np.stack([t[1] - t[0], t[2] - t[0], t[3] - t[0]]))) / 6
        == pytest.approx(1 / 48)
        for t in refined
    )


def test_xfem_energy_release_rate_of_an_edge_crack():
    """An edge crack of 4.5 in a plate of 16 x 32 in tension, its plane through the
    elements: G within 4 % of the handbook (a/h = 4.5, branch functions within 2 h)."""
    mesh = _plate()
    crack = Crack("straight", (0.0, 16.5, 0.0), (0, 1, 0), 4.5, direction=(1, 0, 0))
    model = XFEM(mesh, PLATE, crack)
    assert model.jump.sum() > 0 and model.tip.sum() > 0
    assert model.crack_area() == pytest.approx(4.5 * 2.0)
    g = release_rate(lefm.xfem_work(mesh, PLATE, _forces, _supports), crack, 0.25)
    assert g == pytest.approx(_exact(4.5), rel=0.04)
    # without the crack (ahead of the plate), the stiffness is that of the plate
    outside = Crack("straight", (-10.0, 16.5, 0.0), (0, 1, 0), 1.0, direction=(1, 0, 0))
    intact = XFEM(mesh, PLATE, outside)
    assert intact.n_dofs == 3 * mesh.n_nodes
    w = intact.work(_forces(mesh), _supports(mesh))
    assert w == pytest.approx(W * 2.0 * H / PLATE.young, rel=1e-6)  # f.u = sigma^2 V / E


def test_conforming_meshes_converge():
    """The same crack on conforming meshes (six tetrahedra per voxel, faces on its plane):
    G approaches the handbook as the elements shrink (no singular elements)."""
    ratios = []
    for h in (1.0, 0.5):
        mesh = _plate(h)
        crack = Crack("straight", (0.0, 16.0, 0.0), (0, 1, 0), 4.5, direction=(1, 0, 0))
        g = release_rate(
            lefm.conforming_work_of(lambda c, m=mesh: m, PLATE, _forces, _supports),
            crack,
            max(h / 2, 0.5),
        )
        ratios.append(g / _exact(4.5))
    assert ratios[0] < ratios[1] < 1.0 and ratios[1] > 0.85
    opened, area = lefm.open_crack(
        _plate(), Crack("straight", (0.0, 16.0, 0.0), (0, 1, 0), 4.5, direction=(1, 0, 0))
    )
    assert area == pytest.approx(4.5 * 2.0) and opened.n_nodes > _plate().n_nodes  # half faces


def test_quadratic_elements_and_quarter_points():
    """The same crack with elements of 1: G is 24 % low with 4-node tetrahedra, 7 % low
    with 10-node ones, and within 2 % of the handbook when the mid-edge nodes of the edges
    that leave the front stand at their quarter (Barsoum 1976): the displacements then
    vary as the square root of the distance to the front."""
    mesh = _plate()
    crack = Crack("straight", (0.0, 16.0, 0.0), (0, 1, 0), 4.5, direction=(1, 0, 0))
    ratio = {}
    for name, options in (
        ("linear", {}),
        ("quadratic", {"order": 2, "quarter_point": False}),
        ("quarter", {"order": 2}),
    ):
        work = lefm.conforming_work_of(lambda c: mesh, PLATE, _forces, _supports, **options)
        ratio[name] = release_rate(work, crack, 0.5) / _exact(4.5)
    assert ratio["linear"] < 0.8 < ratio["quadratic"] < 0.95
    assert ratio["quarter"] == pytest.approx(1.0, abs=0.02)
    # the opened mesh of a crack of 4: its area; the front, the nodes that the lips share
    quad = mesh.quadratic()
    opened, faces = lefm.crack_faces(quad, crack.grown(-0.5))
    assert faces.order == 2 and faces.areas(opened).sum() == pytest.approx(4.0 * 2.0)
    front = np.intersect1d(faces.minus[:, :3], faces.plus[:, :3])
    assert len(front) == 3 and np.allclose(opened.nodes[front][:, :2], [4.0, 16.0])
    moved = lefm.quarter_points(opened, front)
    ends = moved.elements[:, lefm.TET_EDGES]
    leave = np.isin(ends[..., 0], front) ^ np.isin(ends[..., 1], front)
    tip = np.where(np.isin(ends[..., 0], front), ends[..., 0], ends[..., 1])[leave]
    other = np.where(np.isin(ends[..., 0], front), ends[..., 1], ends[..., 0])[leave]
    mids = moved.elements[:, 4:][leave]
    np.testing.assert_allclose(
        moved.nodes[mids], 0.75 * moved.nodes[tip] + 0.25 * moved.nodes[other], atol=1e-12
    )
    assert np.array_equal(moved.nodes[moved.elements[:, 4:][~leave]],
                          opened.nodes[opened.elements[:, 4:][~leave]])  # fmt: skip
    assert moved.volumes().sum() == pytest.approx(W * H * 2.0)
    with pytest.raises(ValueError, match="10-node"):
        lefm.quarter_points(mesh, front)
    # the consistent forces of a uniform traction on faces of six nodes: nothing at the
    # corners, a third of the face at each mid-edge node, the whole force in all
    f = lefm.face_forces(quad, lambda c: c[:, 1] > H - 1e-9, [0.0, 1.0, 0.0]).reshape(-1, 3)
    corners = np.unique(quad.elements[:, :4])
    assert f[:, 1].sum() == pytest.approx(W * 2.0) and np.abs(f[corners]).max() < 1e-12
    assert f[:, 1].max() == pytest.approx(2.0 * (0.5 / 3.0))  # two faces of 1/2 share an edge


def test_remeshing_conforming_to_the_crack():
    """MMG meshes the plate conforming to the plane of the crack; the faces behind the
    front open (the remeshing step of the conforming approach)."""
    from morphogen.mesh.conforming import has_mmg

    if not has_mmg():
        pytest.skip("MMG")
    crack = Crack("straight", (0.0, 16.3, 0.0), (0, 1, 0), 5.0, direction=(1, 0, 0))
    mesh = lefm.crack_mesh((0, 0, 0), (W, H, 2.0), crack, size=1.5)
    # two sides meshed by MMG, their interface on the plane (within its snapping)
    assert set(np.unique(mesh.phase)) == {0, 1}
    assert (np.abs(crack.phi(mesh.nodes)) < 0.1 * 1.5).sum() > 20
    opened, area = lefm.open_crack(mesh, crack)
    assert area == pytest.approx(5.0 * 2.0, rel=0.2) and opened.n_nodes > mesh.n_nodes
    g = release_rate(lefm.conforming_work_of(lambda c: mesh, PLATE, _forces, _supports), crack, 1.0)
    assert 0.5 * _exact(5.0) < g < 1.1 * _exact(5.0)


@pytest.mark.slow
def test_fatigue_propagation_by_paris_law():
    """Paris' law along the growth of the edge crack by XFEM: its factors within 4 % of the
    handbook, the cycles of the same forward steps within 8 %."""
    mesh = _plate()
    crack = Crack("straight", (0.0, 16.5, 0.0), (0, 1, 0), 4.5, direction=(1, 0, 0))
    c, m = 1e-3, 3.0
    # fronts in the middle of the elements: both sizes of each difference in one layer
    result = propagate(
        lefm.xfem_work(mesh, PLATE, _forces, _supports),
        crack,
        young=1000.0,
        poisson=0.0,
        increment=1.0,
        steps=3,
        delta=0.25,
        paris=(c, m),
        toughness=0.05,
    )
    assert np.allclose(result.sizes, [4.5, 5.5, 6.5, 7.5])
    assert np.all(np.diff(result.rates) > 0) and np.all(np.diff(result.cycles) > 0)
    np.testing.assert_allclose(result.factors**2 / 1000.0, result.rates)
    np.testing.assert_allclose(result.loads, np.sqrt(0.05 / result.rates))
    exact = [math.sqrt(1000.0 * _exact(a)) for a in result.sizes]
    np.testing.assert_allclose(result.factors, exact, rtol=0.04)
    # the cycles of Paris' law with the handbook factor (the same forward steps)
    n = np.concatenate([[0.0], np.cumsum([1.0 / (c * k**m) for k in exact[:-1]])])
    np.testing.assert_allclose(result.cycles[1:], n[1:], rtol=0.08)
    rows = result.rows()
    assert rows[-1]["a"] == 7.5 and set(rows[0]) == {"a", "G", "K", "N", "load"}
