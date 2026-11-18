"""Post-bifurcated patterns by group theory (Azulay & Combescure 2024): the symmetry group
of the regular honeycomb and its irreps (their Table 3), periodic frames of
Euler–Bernoulli beams and of Timoshenko beams (deflections with shear, Engesser's load,
the Euler–Bernoulli limit), the first bifurcations of the honeycomb under equibiaxial and
uniaxial compression (Azulay 2023), frames of struts in space (beams, lattices, stiffness
against the scaling laws of lattice materials, the bifurcations of the cubic lattice),
Schoenflies symbols, continua: the symmetries of meshes, a lattice of bars and a column
against Timoshenko; and along nonlinear paths in finite strains: the limit of linear
buckling, the pattern transformation of a square array of holes and its bifurcated
branch, an elastoplastic column."""

import numpy as np
import pytest
from scipy.linalg import eigh

from morphogen import symmetry as sy
from morphogen.domain import Domain
from morphogen.homogenization.behaviours import Plasticity
from morphogen.homogenization.bifurcation import (
    frame_representation,
    mesh_symmetries,
    path_bifurcation,
    post_bifurcation,
)
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.frames import (
    Frame,
    SpaceFrame,
    beam_matrices,
    cowper_coefficient,
    space_beam_matrices,
)
from morphogen.homogenization.materials import VOID, Material
from morphogen.image import VoxelImage
from morphogen.mesh import ImplicitBody, mirrored_mesh
from morphogen.mesh.conforming import has_mmg


@pytest.fixture(scope="module")
def honeycomb():
    return Frame.lattice_of("honeycomb", (2, 2), elements=4, thickness=0.05)


def _classes(group, irrep):
    """(order, point group, translations, conjugates) of each axial class of an irrep."""
    return sorted(
        (
            c.description["order"],
            c.description["point_group"],
            c.description["translations"],
            len(c.subgroups),
        )
        for c in sy.axial_subgroups(group, irrep)
    )


def test_symmetry_group_of_the_honeycomb(honeycomb):
    g = sy.Group.generate(honeycomb.generators, honeycomb.lattice, honeycomb.centre)
    assert g.order == 48  # D6 ⋉ (Z2 x Z2)
    irreps = g.irreps()
    assert [r.dim for r in irreps] == [1, 1, 1, 1, 2, 2, 3, 3, 3, 3]
    assert sum(r.dim**2 for r in irreps) == g.order and all(r.kind == 1 for r in irreps)
    rng = np.random.default_rng(0)
    for a, b in rng.integers(0, g.order, size=(20, 2)):
        for r in irreps:  # a representation: D(a) D(b) = D(ab)
            np.testing.assert_allclose(
                r.matrices[a] @ r.matrices[b], r.matrices[g.table[a, b]], atol=1e-10
            )
    # Table 3 of Azulay & Combescure: the axial isotropy subgroup classes of each irrep
    expected = [
        [(48, "D6", 4, 1)],  # A1, the symmetric branch
        [(24, "C6", 4, 1)],  # A2
        [(24, "D3", 4, 1)],  # B1
        [(24, "D3", 4, 1)],  # B2
        [(16, "D2", 4, 3)],  # E1
        [(8, "D1", 4, 3), (8, "D1", 4, 3)],  # E2
        [(8, "D2", 2, 3), (12, "D6", 1, 4)],  # T1
        [(4, "D2", 1, 6), (6, "C6", 1, 4), (8, "D2", 2, 3)],  # T2: modes II, III, I
        [(4, "D2", 1, 6), (6, "D3", 1, 4), (8, "D2", 2, 3)],  # T3
        [(4, "D2", 1, 6), (6, "D3", 1, 4), (8, "D2", 2, 3)],  # T4
    ]
    classes = [_classes(g, r) for r in irreps]
    assert sorted(classes) == sorted(expected)
    assert sum(len(c) for c in classes[1:]) == 17  # the classes of the symmetry-breaking irreps


def test_frames_of_euler_bernoulli_beams(honeycomb):
    # a cantilever of four beams: the Euler load pi^2 EI / 4 H^2
    ke, kg = np.zeros((15, 15)), np.zeros((15, 15))
    for k in range(4):
        e, s = beam_matrices(np.array([0.0, 0.25]), 100.0, 1.0, -1.0)
        idx = np.arange(3 * k, 3 * k + 6)
        ke[np.ix_(idx, idx)] += e
        kg[np.ix_(idx, idx)] += s
    mu = eigh(-kg[3:, 3:], ke[3:, 3:], eigvals_only=True)
    assert 1 / mu.max() == pytest.approx(np.pi**2 / 4, rel=1e-4)
    # an equibiaxial strain shortens every wall alike, without fluctuation
    forces, w = honeycomb.axial_forces(-np.eye(2))
    np.testing.assert_allclose(forces, -honeycomb.axial, rtol=1e-10)
    assert np.abs(w).max() < 1e-10
    # the stiffness commutes with the symmetries of the cell
    g = sy.Group.generate(honeycomb.generators, honeycomb.lattice, honeycomb.centre)
    k = honeycomb.stiffness()
    for t in frame_representation(honeycomb, g):
        assert np.abs(t @ k - k @ t).max() < 1e-10 * np.abs(k).max()
    # the primitive cell buckles with its joints turning alike: walls in double curvature
    cell = Frame.lattice_of("honeycomb", (1, 1), elements=8, thickness=0.05)
    lam, _ = cell.buckling(-1e-3 * np.eye(2), modes=1)
    assert lam[0] * 1e-3 * cell.axial[0] / cell.bending[0] == pytest.approx(np.pi**2, rel=1e-4)
    for kind, cells, order in (
        ("triangular", (3, 3), 108),
        ("square", (2, 2), 32),
        ("kagome", (2, 2), 48),
    ):
        frame = Frame.lattice_of(kind, cells, elements=2)
        group = sy.Group.generate(frame.generators, frame.lattice, frame.centre)
        assert group.order == order and group.action(frame.nodes).shape == (order, len(frame.nodes))


def _chain(count, step, matrices, size):
    """The elastic and geometric stiffness of a chain of ``count`` beams of vector
    ``step`` (``matrices(step)`` their pair), ``size`` dofs per node."""
    n = size * (count + 1)
    ke, kg = np.zeros((n, n)), np.zeros((n, n))
    e, g = matrices(step)
    for k in range(count):
        idx = np.arange(size * k, size * k + 2 * size)
        ke[np.ix_(idx, idx)] += e
        kg[np.ix_(idx, idx)] += g
    return ke, kg


def test_frames_of_timoshenko_beams():
    """Timoshenko beams (shear stiffness kGA): a cantilever deflects by P L^3 / 3EI + P L /
    kGA under a tip force and turns by P L^2 / 2EI, a fixed-fixed beam by P L^3 / 192EI + P
    L / 4kGA under a central force (exact for any number of elements); the buckling load
    tends to Engesser's P_E / (1 + P_E / kGA) as the elements refine; Euler-Bernoulli at
    Phi -> 0; in space, the deflection with shear in both planes; Cowper's coefficients."""
    ea, ei, kga = 100.0, 1.0, 3.0
    ke, _ = _chain(4, np.array([0.0, 0.25]), lambda d: beam_matrices(d, ea, ei, 0.0, kga), 3)
    u = np.linalg.solve(ke[3:, 3:], np.eye(12)[-3])  # tip force across the beam
    assert u[-3] == pytest.approx(1.0 / 3.0 + 1.0 / kga, rel=1e-13)
    assert u[-1] == pytest.approx(-0.5, rel=1e-13)  # the rotation of the section: no shear
    ke, _ = _chain(2, np.array([0.5, 0.0]), lambda d: beam_matrices(d, ea, ei, 0.0, kga), 3)
    middle = [3, 4, 5]
    u = np.linalg.solve(ke[np.ix_(middle, middle)], [0.0, 1.0, 0.0])
    assert u[1] == pytest.approx(1.0 / 192.0 + 0.25 / kga, rel=1e-13)
    euler = np.pi**2 / 4.0
    engesser = euler / (1.0 + euler / kga)
    for count, tol in ((16, 3e-4), (64, 2e-5)):
        ke, kg = _chain(count, np.array([0.0, 1.0 / count]),
                        lambda d: beam_matrices(d, ea, ei, -1.0, kga), 3)  # fmt: skip
        mu = eigh(-kg[3:, 3:], ke[3:, 3:], eigvals_only=True)
        assert 1.0 / mu.max() == pytest.approx(engesser, rel=tol)
    d = np.array([0.3, 0.4])
    e0, g0 = beam_matrices(d, ea, ei, -0.7)
    e1, g1 = beam_matrices(d, ea, ei, -0.7, np.inf)
    assert np.array_equal(e0, e1) and np.array_equal(g0, g1)
    e1, g1 = beam_matrices(d, ea, ei, -0.7, 1e14)  # Phi = 4.8e-13
    assert np.abs(e1 - e0).max() < 1e-12 * np.abs(e0).max()
    assert np.abs(g1 - g0).max() < 1e-12 * np.abs(g0).max()
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    ke, _ = _chain(8, axis / 8, lambda d: space_beam_matrices(d, 100.0, 1.0, 0.8, 0.0, 0.0, kga), 6)
    across = np.cross(axis, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across)
    for load in (across, np.cross(axis, across)):
        f = np.zeros(48)
        f[-6:-3] = load
        u = np.linalg.solve(ke[6:, 6:], f)
        assert u[-6:-3] @ load == pytest.approx(1.0 / 3.0 + 1.0 / kga, rel=1e-12)
    e0, g0 = space_beam_matrices(axis, 100.0, 1.0, 0.8, -0.7, 0.5)
    e1, g1 = space_beam_matrices(axis, 100.0, 1.0, 0.8, -0.7, 0.5, np.inf)
    assert np.array_equal(e0, e1) and np.array_equal(g0, g1)
    assert cowper_coefficient(0.3) == pytest.approx(7.8 / 8.8)
    assert cowper_coefficient(0.0, "rectangular") == pytest.approx(5.0 / 6.0)
    with pytest.raises(ValueError, match="section"):
        cowper_coefficient(0.3, "tube")


def test_first_bifurcations_of_the_honeycomb(honeycomb):
    b = post_bifurcation(honeycomb, -1e-3 * np.eye(2))
    load, irrep, count = b.first
    # equibiaxial: a triple point in T2, whose branches are modes I, II and III
    assert count == 3 and b.irreps[irrep].dim == 3
    assert _classes(b.group, b.irreps[irrep]) == [(4, "D2", 1, 6), (6, "C6", 1, 4), (8, "D2", 2, 3)]
    np.testing.assert_allclose(b.projections[irrep], 1.0, atol=1e-8)
    assert np.abs(np.delete(b.projections, irrep, axis=0)).max() < 1e-8
    lam, _ = honeycomb.buckling(-1e-3 * np.eye(2), modes=1)
    assert load == pytest.approx(lam[0], rel=1e-10) and np.nanmin(b.loads) == pytest.approx(
        lam[0], rel=1e-10
    )
    assert load * 1e-3 * honeycomb.axial[0] / honeycomb.bending[0] == pytest.approx(
        3.6508, rel=1e-4
    )
    # each pattern is fixed by its subgroup, lies in its isotypic subspace, has no translation
    t = frame_representation(honeycomb, b.group)
    for p in b.patterns:
        assert all(np.abs(t[h] @ p.mode - p.mode).max() < 1e-10 for h in p.subgroups[0])
        weights = b.irreps[p.irrep].projector_weights()
        projected = sum(w * (t[g] @ p.mode) for g, w in enumerate(weights))
        assert np.abs(projected - p.mode).max() < 1e-10
        assert np.abs(honeycomb.translations().T @ p.mode).max() < 1e-10
        assert np.linalg.norm(p.mode) == pytest.approx(1.0) and p.energy > 0
    assert sum(p.irrep > 0 for p in b.patterns) == 17
    # uniaxial normal to the vertical walls: D2 ⋉ (Z2 x Z2), a double point
    b = post_bifurcation(honeycomb, np.diag([-1e-3, 0.0]))
    assert b.group.order == 16 and len(b.irreps) == 10
    assert b.first[2] == 2 and b.irreps[b.first[1]].dim == 2
    pytest.importorskip("matplotlib")
    fig = b.figure(honeycomb, tiles=(2, 2), irreps=[b.first[1]])
    assert len([ax for ax in fig.axes if ax.collections]) == 2


def test_bloch_wave_stability_selects_the_cell():
    cell = Frame.lattice_of("honeycomb", (1, 1), elements=4, thickness=0.05)
    b = cell.reciprocal()
    np.testing.assert_allclose(b.T @ cell.lattice, 2 * np.pi * np.eye(2), atol=1e-12)
    strain = -1e-3 * np.eye(2)
    # the three M points: the triple first bifurcation of the 2 x 2 cell; Γ and its images:
    # the primitive cell
    m = cell.bloch_buckling(strain, [b[:, 0] / 2, b[:, 1] / 2, (b[:, 0] + b[:, 1]) / 2])[:, 0]
    big = Frame.lattice_of("honeycomb", (2, 2), elements=4, thickness=0.05)
    np.testing.assert_allclose(m, big.buckling(strain, modes=1)[0][0], rtol=1e-8)
    gamma = cell.bloch_buckling(strain, [np.zeros(2), b[:, 0]])[:, 0]
    np.testing.assert_allclose(gamma, cell.buckling(strain, modes=1)[0][0], rtol=1e-8)
    # over a grid of the Brillouin zone, the lowest load is at an M point
    grid = np.array([(i / 12, j / 12) for i in range(12) for j in range(12)])
    loads = cell.bloch_buckling(strain, grid @ b.T)[:, 0]
    assert np.nanmin(loads) == pytest.approx(m[0], rel=1e-8)
    assert tuple(grid[np.nanargmin(loads)] % 0.5) == (0.0, 0.0)


def test_frames_of_struts_in_space():
    # a cantilever of eight beams along an oblique axis: Euler's load pi^2 EI / 4 H^2, the
    # deflection P H^3 / 3 EI of its tip and its twist T H / GJ
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    n = 6 * 9
    ke, kg = np.zeros((n, n)), np.zeros((n, n))
    for k in range(8):
        e, g = space_beam_matrices(axis / 8, 100.0, 1.0, 0.8, -1.0)
        idx = np.arange(6 * k, 6 * k + 12)
        ke[np.ix_(idx, idx)] += e
        kg[np.ix_(idx, idx)] += g
    mu = eigh(-kg[6:, 6:], ke[6:, 6:], eigvals_only=True)
    assert 1 / mu.max() == pytest.approx(np.pi**2 / 4, rel=1e-5)
    across = np.cross(axis, [0.0, 0.0, 1.0])
    for load, dofs, expected in (
        (across / np.linalg.norm(across), slice(-6, -3), 1 / 3.0),
        (axis, slice(-3, None), 1 / 0.8),
    ):
        f = np.zeros(n - 6)
        f[dofs] = load
        u = np.linalg.solve(ke[6:, 6:], f)
        assert u[dofs] @ load == pytest.approx(expected, rel=1e-10)
    # the lattices: their groups (O_h and the translations of the centring), stiffness
    # commuting with them, a hydrostatic strain stretching every strut alike
    # (the diamond and pyrochlore lattices: the same factor group O_h, some of whose
    # operations come with a translation)
    lattices = (("cubic", 48), ("bcc", 96), ("octet", 192), ("kelvin", 96), ("diamond", 192),
                ("pyrochlore", 192))  # fmt: skip
    for kind, order in lattices:
        frame = SpaceFrame.lattice_of(kind, elements=2, radius=0.04)
        group = sy.Group.generate(frame.generators, frame.lattice, frame.centre)
        assert group.order == order
        k = frame.stiffness()
        for t in frame_representation(frame, group):
            assert np.abs(t.T @ k @ t - k).max() < 1e-10 * np.abs(k).max()
        forces, w = frame.axial_forces(1e-3 * np.eye(3))
        np.testing.assert_allclose(forces, 1e-3 * frame.axial, rtol=1e-8)
        assert np.abs(w).max() < 1e-10

    # stiffness against the scaling laws: octet E/Es = rho/9 and nu = 1/3 (stretching,
    # Deshpande, Fleck & Ashby 2001), cubic E/Es = rho/3 along its struts, Kelvin E/Es =
    # 1.009 rho^2 for Plateau borders (Warren & Kraynik 1997), about 0.6 rho^2 for circular
    # struts (I/A^2 = 1/4 pi instead of 0.1338), body-centred cubic bending-dominated
    def young(kind, radius):
        frame = SpaceFrame.lattice_of(kind, elements=2, radius=radius)
        s = np.linalg.inv(frame.macroscopic_stiffness())
        return 1 / s[0, 0], -s[0, 1] / s[0, 0], frame.relative_density()

    e, nu, rho = young("octet", 0.01)
    assert e / rho == pytest.approx(1 / 9, rel=0.01) and nu == pytest.approx(1 / 3, abs=1e-3)
    e, nu, rho = young("cubic", 0.02)
    assert e / rho == pytest.approx(1 / 3, rel=1e-3) and abs(nu) < 1e-10
    e, _, rho = young("kelvin", 0.01)
    assert e / rho**2 == pytest.approx(1.009 / 0.1338 / (4 * np.pi), rel=0.01)
    assert young("bcc", 0.02)[0] / young("bcc", 0.01)[0] == pytest.approx(16.0, rel=0.01)
    # the diamond lattice bends as the honeycomb (E as rho^2); the pyrochlore lattice
    # stretches as the octet truss, with half its struts: E/Es = rho/9, nu = 1/3
    assert young("diamond", 0.02)[0] / young("diamond", 0.01)[0] == pytest.approx(16.0, rel=0.02)
    e, nu, rho = young("pyrochlore", 0.005)
    assert e / rho == pytest.approx(1 / 9, rel=0.01) and nu == pytest.approx(1 / 3, abs=2e-3)


def test_bifurcations_of_a_frame_in_space():
    """The cubic lattice of struts in compression along z: its 2 x 2 x 2 cell buckles at
    Euler's load of its columns clamped with sway (wavelength of two cells along z), in the
    group D4h of the loading with the translations of the cell; the Bloch waves of the
    primitive cell agree."""
    strain = np.diag([0.0, 0.0, -1.0])
    frame = SpaceFrame.lattice_of("cubic", (2, 2, 2), elements=4, radius=0.04)
    b = post_bifurcation(frame, strain)
    load, irrep, count = b.first
    assert b.group.order == 128 and count == 2
    assert load * frame.axial[0] / frame.bending[0] == pytest.approx(np.pi**2, rel=1e-3)
    lam, _ = frame.buckling(strain, modes=1)
    assert load == pytest.approx(lam[0], rel=1e-10)
    critical = np.flatnonzero(np.abs(b.loads - load) <= 1e-6 * load)
    np.testing.assert_allclose(np.linalg.norm(b.projections[critical], axis=0), 1.0, atol=1e-8)
    assert np.abs(np.delete(b.projections, critical, axis=0)).max() < 1e-8
    t = frame_representation(frame, b.group)
    for p in b.patterns[:20]:
        assert all(np.abs(t[h] @ p.mode - p.mode).max() < 1e-10 for h in p.subgroups[0])
        assert np.abs(frame.translations().T @ p.mode).max() < 1e-10
    assert {p.description["point_group"] for p in b.patterns} >= {"D4h", "C4v"}
    cell = SpaceFrame.lattice_of("cubic", (1, 1, 1), elements=4, radius=0.04)
    corners = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]) / 2
    waves = cell.bloch_buckling(strain, corners @ cell.reciprocal().T)[:, 0]
    assert np.nanmin(waves) == pytest.approx(load, rel=1e-8)
    assert tuple(corners[np.nanargmin(waves)]) == (0.0, 0.0, 0.5)
    pytest.importorskip("matplotlib")
    fig = b.figure(frame, irreps=[irrep])
    assert any(ax.name == "3d" for ax in fig.axes)
    from morphogen.viz.video import frame_pattern_frames

    frames = frame_pattern_frames(frame, b.patterns[0].mode, frames=3, size=120)
    assert len(frames) == 3 and frames[0].shape[2] == 3


def test_schoenflies_symbols():
    def closure(generators):
        out, frontier = [np.eye(3)], [np.eye(3)]
        while frontier:
            new = []
            for a in generators:
                for b in frontier:
                    c = np.round(a @ b, 9)
                    if not any(np.allclose(c, x) for x in out):
                        out.append(c)
                        new.append(c)
            frontier = new
        return out

    c4 = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    c3 = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    s4 = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, -1]])
    c3z = np.array([[-0.5, -np.sqrt(3) / 2, 0], [np.sqrt(3) / 2, -0.5, 0], [0, 0, 1]])
    c2x, c2z = np.diag([1.0, -1, -1]), np.diag([-1.0, -1, 1])
    mz, mx, inv = np.diag([1.0, 1, -1]), np.diag([-1.0, 1, 1]), -np.eye(3)
    cases = {
        "Oh": [c4, c3, inv],
        "O": [c4, c3],
        "Td": [s4, c3],
        "Th": [c3, c2x, inv],
        "T": [c3, c2x],
        "D4h": [c4, mz, mx],
        "D4": [c4, c2x],
        "C4v": [c4, mx],
        "C4h": [c4, mz],
        "S4": [s4],
        "D2d": [s4, c2x],
        "D3h": [c3z, mz, mx],
        "C3v": [c3z, mx],
        "C3h": [c3z, mz],
        "S6": [c3z, inv],
        "D3d": [c3z, c2x, inv],
        "C2v": [c2z, mx],
        "Cs": [mz],
        "Ci": [inv],
        "C1": [],
        "C2": [c2z],
    }
    for name, generators in cases.items():
        assert sy.schoenflies(closure(generators)) == name, name


def _bars(cell=4, bar=2, cells=1):
    """A simple cubic lattice of square bars (voxels)."""
    x = np.arange(cell * cells) % cell
    inside = (x >= (cell - bar) // 2) & (x < (cell + bar) // 2)
    a, b, c = np.meshgrid(inside, inside, inside, indexing="ij")
    return VoxelImage(((a & b) | (b & c) | (a & c)).astype(np.uint8), periodic=True)


def test_post_bifurcation_of_a_continuum(tmp_path):
    # the symmetries of the meshes: the cube about a cell, the translations of a 2 x 2 x 2
    # cell, none (the identity) for a random image
    assert mesh_symmetries(voxel_tet_mesh(_bars(), phases=[1])).order == 48
    big = voxel_tet_mesh(_bars(cells=2), phases=[1])
    assert mesh_symmetries(big, (2, 2, 2)).order == 384
    noise = np.random.default_rng(0).random((4, 4, 4)) < 0.5
    assert (
        mesh_symmetries(voxel_tet_mesh(VoxelImage(noise.astype(np.uint8), periodic=True))).order
        == 1
    )
    materials = {0: VOID, 1: Material(young=1.0, poisson=0.3)}
    b = post_bifurcation(_bars(), -1e-3 * np.eye(3), materials=materials)
    _, irrep, count = b.first
    assert b.group.order == 48 and count == b.irreps[irrep].dim == 3
    np.testing.assert_allclose(b.projections[irrep], 1.0, atol=1e-8)
    assert np.abs(np.delete(b.projections, irrep, axis=0)).max() < 1e-8
    # a pattern for each axial class of the irreps present among the displacements (O_h has
    # 17 for its symmetry-breaking irreps; that of A1u, absent from this coarse mesh, has none)
    present = [k for k in range(len(b.irreps)) if b.multiplicities[k]]
    assert len(b.patterns) == sum(len(sy.axial_subgroups(b.group, b.irreps[k])) for k in present)
    assert sum(len(sy.axial_subgroups(b.group, r)) for r in b.irreps[1:]) == 17
    # each pattern is fixed by its subgroup, on every node of the mesh
    perm = b.group.action(b.mesh.nodes)
    for p in b.patterns:
        u = b.displacement(p.mode)
        for h in p.subgroups[0]:
            np.testing.assert_allclose(
                u[perm[h]], u @ b.group.rotations[h].T, atol=1e-8 * np.abs(u).max()
            )
    path = b.write(tmp_path / "patterns.vtu")
    assert path.stat().st_size > 1000 and b"pattern_" in path.read_bytes()
    assert all(b.wave_vectors(k) == [(0, 0, 0)] for k in range(len(b.irreps)))


def test_cells_of_eight_cells_split_by_their_translations():
    # the lattice of bars in a cell of 2 x 2 x 2 (384 operations), split first by its
    # translations (the wave vectors Γ, X, M and R) and then by the little co-groups of their
    # stars: the loads, multiplicities, first bifurcation and critical patterns of the dense
    # analysis, without its bases of the whole cell; the bars buckle first in the star of X,
    # an irrep of dimension 6 that doubles the period along one axis
    materials = {0: VOID, 1: Material(young=1.0, poisson=0.3)}
    image, strain = _bars(cells=2), -1e-3 * np.eye(3)
    options = {"materials": materials, "cells": (2, 2, 2), "patterns": "critical"}
    dense = post_bifurcation(image, strain, bloch=False, **options)
    split = post_bifurcation(image, strain, bloch=True, **options)
    assert split.group.order == 384 and split.projections is None
    np.testing.assert_array_equal(split.multiplicities, dense.multiplicities)
    np.testing.assert_allclose(split.loads, dense.loads, rtol=1e-9)
    load, irrep, count = split.first
    assert (irrep, count) == dense.first[1:] and load == pytest.approx(dense.first[0], rel=1e-9)
    assert split.irreps[irrep].dim == count == 6
    assert sorted(split.wave_vectors(irrep)) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    # the irreps of Γ repeat in every primitive cell: those of the single cell, which
    # buckles later (the period doubled lowers the first load)
    one = post_bifurcation(_bars(), strain, materials=materials, patterns="critical")
    gamma = [i for i in range(len(split.irreps)) if split.wave_vectors(i) == [(0, 0, 0)]]
    assert sorted((split.irreps[i].dim, split.multiplicities[i]) for i in gamma) == sorted(
        (r.dim, m) for r, m in zip(one.irreps, one.multiplicities, strict=True)
    )
    np.testing.assert_allclose(np.sort(split.loads[gamma]), np.sort(one.loads), rtol=1e-9)
    assert one.first[0] > 1.4 * load
    assert [(p.irrep, p.dimension, p.subgroups) for p in split.patterns] == [
        (p.irrep, p.dimension, p.subgroups) for p in dense.patterns
    ]
    np.testing.assert_allclose(
        [(p.energy, p.load) for p in split.patterns],
        [(p.energy, p.load) for p in dense.patterns],
        rtol=1e-9,
    )
    perm = split.group.action(split.mesh.nodes)
    for p in split.patterns:  # each fixed by its subgroup, on every node of the mesh
        for u in (split.displacement(p.mode), split.displacement(p.buckling_mode)):
            for h in p.subgroups[0]:
                np.testing.assert_allclose(
                    u[perm[h]], u @ split.group.rotations[h].T, atol=1e-8 * np.abs(u).max()
                )
    with pytest.raises(ValueError, match="patterns"):
        post_bifurcation(image, strain, materials=materials, patterns="some")


@pytest.mark.slow
def test_continuum_column_buckles_as_timoshenko():
    # a periodic square bar of 2 x 2 voxels, 12 long: the mode of one wavelength, in either
    # direction and phase, at Euler's load reduced by the shear (kappa = 5/6, G = E/2)
    data = np.zeros((4, 4, 12), np.uint8)
    data[1:3, 1:3, :] = 1
    materials = {0: VOID, 1: Material(young=1.0, poisson=0.0)}
    b = post_bifurcation(
        VoxelImage(data, periodic=True), np.diag([0.0, 0.0, -1e-3]), materials=materials, order=2
    )
    euler = (2 * np.pi / 12) ** 2 * 4 / 12
    timoshenko = euler / (1 + 2.4 * euler)
    assert b.group.order == 32 and b.first[2] == 4
    assert 0.92 * timoshenko < b.first[0] * 1e-3 < timoshenko


# -- along nonlinear paths -----------------------------------------------------------------


def _column(length):
    """A periodic square column of 2 x 2 voxels along z, ``length`` voxels long."""
    data = np.zeros((4, 4, length), np.uint8)
    data[1:3, 1:3, :] = 1
    return VoxelImage(data, periodic=True)


def _compression(axis, strain, *, free=(), increments=13):
    """A path of compression along ``axis`` to the nominal ``strain``, the components
    ``free`` without stress, the others held at zero strain."""
    names = ("11", "22", "33", "23", "31", "12")
    spec = {name: 0.0 for k, name in enumerate(names) if k not in free}
    spec[names[axis]] = strain
    return {"times": [0, 1], "increments": increments, "strain": spec}


def _ellipticities(nodes, u, cell, radius, band=1.0):
    """The ellipticity of each hole of a 2 x 2 array (the holes (0, 0), (0, 1), (1, 0),
    (1, 1)) under a displacement: its radial displacement along x less that along y (the
    nodes within ``band`` of its edge)."""
    out = []
    for hx in (0, 1):
        for hy in (0, 1):
            d = nodes[:, :2] - cell * np.array([hx + 0.5, hy + 0.5])
            rad = np.hypot(d[:, 0], d[:, 1])
            ring = np.abs(rad - radius) < band
            radial = (u[:, 0] * d[:, 0] + u[:, 1] * d[:, 1]) / np.maximum(rad, 1e-12)
            along_x = ring & (np.abs(d[:, 0]) > 0.9 * rad)
            along_y = ring & (np.abs(d[:, 1]) > 0.9 * rad)
            out.append(radial[along_x].mean() - radial[along_y].mean())
    return np.array(out)


def test_path_bifurcation_tends_to_linear_buckling():
    """A periodic column of 2 x 2 voxels, 96 long (a critical strain of 0.25 %): along the
    finite-strain path of a Hencky solid, the first bifurcation within 1 % of the linear
    buckling, in the same irreps (both directions and phases of one wavelength: four
    modes), the rotation of the column about its axis (a mechanism) left out."""
    image = _column(96)
    materials = {0: VOID, 1: Material(young=1.0, poisson=0.0)}
    linear = post_bifurcation(image, np.diag([0.0, 0.0, -1e-3]), materials=materials)
    eps = linear.first[0] * 1e-3
    b = path_bifurcation(image, _compression(2, -1.3 * eps), materials=materials)
    time, _, count = b.first
    assert time * 1.3 == pytest.approx(1.0, rel=0.01) and count == linear.first[2] == 4
    assert b.group.order == linear.group.order == 32
    critical = np.flatnonzero(np.abs(b.loads - time) <= 1e-3 * time)
    lowest = np.flatnonzero(np.abs(linear.loads - linear.first[0]) <= 1e-3 * linear.first[0])
    assert set(critical) == set(lowest)
    assert b.critical_strain[2] == pytest.approx(-time * 1.3 * eps)
    assert np.nanmin(b.spectra[0, :, 0]) > 0 and b.principal.times[-1] < 1.0  # stopped there
    for p in b.patterns:  # unit modes without translation
        assert np.linalg.norm(p.mode) == pytest.approx(1.0)
        assert abs(p.mode[0::3].sum()) < 1e-8 and abs(p.mode[2::3].sum()) < 1e-8


def test_pattern_transformation_of_a_square_array_of_holes():
    """Circular holes (porosity 0.5) in uniaxial compression, plane strain, the lateral
    stress free: the first bifurcation, in a one-dimensional irrep that doubles the period,
    turns the holes into alternating, mutually orthogonal ellipses (Mullin et al. 2007;
    Bertoldi et al. 2008); on the bifurcated branch the pattern grows, the stress levels
    off and the lateral strain turns negative (a negative Poisson's ratio)."""
    cell = 12
    radius = cell * np.sqrt(0.5 / np.pi)
    c = (np.arange(2 * cell) % cell) + 0.5 - cell / 2
    solid = (c[:, None] ** 2 + c[None, :] ** 2) > radius**2
    image = VoxelImage(solid.astype(np.uint8)[:, :, None], periodic=True)
    materials = {0: VOID, 1: Material(young=1.0, poisson=0.3)}
    path = _compression(1, -0.15, free=(0,), increments=24)
    b = path_bifurcation(image, path, materials=materials, cells=(2, 2, 1), branch=0.02)
    time, irrep, count = b.first
    assert b.irreps[irrep].dim == 1 and count == 1
    assert 0.03 < -b.critical_strain[1] < 0.09 and b.critical_strain[0] > 0
    pattern = min((p for p in b.patterns if p.irrep == irrep), key=lambda p: p.energy)
    e = _ellipticities(b.mesh.nodes, b.displacement(pattern.mode), cell, radius)
    assert e[0] * e[1] < 0  # a checkerboard of ellipses along x and along y
    np.testing.assert_allclose(e, e[0] * np.array([1, -1, -1, 1]), rtol=1e-6)
    a = np.abs(b.amplitudes)
    assert np.all(np.diff(a) > 0) and a[-1] > 50 * a[0]
    stress = b.branch.stress[b.branch.times > 1.6 * time, 1]
    assert np.ptp(stress) < 0.05 * abs(stress.mean())  # the plateau
    assert b.branch.strain[-1, 0] < 0  # auxetic once transformed
    u = b.branch.fields[1.0]["displacement"]
    w = u - b.mesh.nodes @ np.diag(b.branch.strain[-1, :3])  # the fluctuation
    e = _ellipticities(b.mesh.nodes, w, cell, radius)
    assert e[0] * e[1] < 0 and e[0] * e[3] > 0 and np.abs(e).min() > 1.0


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_pattern_transformation_on_a_conforming_mesh():
    """The square array of circular holes (porosity 0.5) on a conforming mesh of its level
    set, meshed in an eighth of the primitive cell and mirrored, so that the symmetries of
    the 2 x 2 cell are exact: the first bifurcation is simple, in a one-dimensional irrep
    that doubles the period, the holes turning into alternating ellipses, as on voxels."""
    radius = np.sqrt(0.5 / np.pi)

    def holes(points):
        d = points[:, :2] % 1.0 - 0.5
        return radius - np.hypot(d[:, 0], d[:, 1])

    body = ImplicitBody(Domain(size=(1.0, 1.0, 0.24), periodic=True), holes)
    mesh = mirrored_mesh(body, size=0.12, cells=(2, 2, 1), phases="solid")
    path = _compression(1, -0.12, free=(0,), increments=12)
    b = path_bifurcation(mesh, path, materials={1: Material(young=1.0, poisson=0.3)},
                         cells=(2, 2, 1))  # fmt: skip
    _, irrep, count = b.first
    assert b.group.order == 32 and b.irreps[irrep].dim == 1 and count == 1
    assert 0.05 < -b.critical_strain[1] < 0.11 and b.critical_strain[0] > 0
    pattern = min((p for p in b.patterns if p.irrep == irrep), key=lambda p: p.energy)
    e = _ellipticities(b.mesh.nodes, b.displacement(pattern.mode), 1.0, radius, band=0.06)
    assert e[0] * e[1] < 0
    np.testing.assert_allclose(e, e[0] * np.array([1, -1, -1, 1]), rtol=1e-6)


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_equibiaxial_compression_with_the_diagonal_mirrors():
    """The square array of holes on a mesh with its diagonal mirrors (4/mmm: 64 operations
    with the translations of the 2 x 2 cell), which equibiaxial compression keeps and
    uniaxial compression halves: equibiaxially, the first bifurcation, simple in a
    one-dimensional irrep, turns the holes into alternating ellipses, at about half the
    uniaxial strain, and the linear buckling of the same mesh finds the same irrep."""
    radius = np.sqrt(0.5 / np.pi)

    def holes(points):
        d = points[:, :2] % 1.0 - 0.5
        return radius - np.hypot(d[:, 0], d[:, 1])

    body = ImplicitBody(Domain(size=(1.0, 1.0, 0.3), periodic=True), holes)
    mesh = mirrored_mesh(body, size=0.15, diagonals=((0, 1),), cells=(2, 2, 1), phases="solid")
    group = mesh_symmetries(mesh, cells=(2, 2, 1))
    assert group.order == 64
    assert group.stabilizer_of(np.diag([-1.0, -1.0, 0.0])).sum() == 64
    assert group.stabilizer_of(np.diag([0.0, -1.0, 0.0])).sum() == 32
    materials = {1: Material(young=1.0, poisson=0.3)}
    spec = {"11": -0.06, "22": -0.06, "33": 0.0, "23": 0.0, "31": 0.0, "12": 0.0}
    b = path_bifurcation(mesh, {"times": [0, 1], "increments": 8, "strain": spec},
                         materials=materials, cells=(2, 2, 1))  # fmt: skip
    _, irrep, count = b.first
    assert b.group.order == 64 and b.irreps[irrep].dim == 1 and count == 1
    assert b.critical_strain[0] == pytest.approx(b.critical_strain[1])
    assert 0.03 < -b.critical_strain[0] < 0.06
    pattern = min((p for p in b.patterns if p.irrep == irrep), key=lambda p: p.energy)
    e = _ellipticities(b.mesh.nodes, b.displacement(pattern.mode), 1.0, radius, band=0.09)
    assert abs(e[0]) > 0
    np.testing.assert_allclose(e, e[0] * np.array([1, -1, -1, 1]), rtol=1e-6)
    linear = post_bifurcation(mesh, np.diag([-1e-3, -1e-3, 0.0]), materials=materials,
                              cells=(2, 2, 1))  # fmt: skip
    assert linear.group.order == 64 and linear.first[1] == irrep and linear.first[2] == 1


def test_plastic_bifurcation_comes_earlier():
    """The column of 2 x 2 voxels, 48 long, in compression: of a von Mises elastoplastic
    solid (yield strain 0.3 %, hardening E / 2) or of a copper crystal along [001]
    (octahedral slip from 100 MPa: yield at 0.37 % by Schmid's law), the plastic tangent
    bifurcates after yield but well before the elastic one, in the same irreps."""
    import dataclasses

    from morphogen.homogenization.crystal import CrystalPlasticity

    image = _column(48)
    path = _compression(2, -0.013, increments=26)
    copper = dataclasses.replace(
        Material.cubic(168e3, 121e3, 75e3),
        crystal_plasticity=CrystalPlasticity.from_config({"tau0": 100.0, "K": 5.0, "n": 20.0}),
    )
    out = {}
    for name, material in (
        ("elastic", Material(young=1.0, poisson=0.0)),
        ("plastic", Material(young=1.0, poisson=0.0, plasticity=Plasticity(0.003, hardening=0.5))),
        ("crystal", copper),
    ):
        b = path_bifurcation(image, path, materials={0: VOID, 1: material})
        out[name] = (-b.critical_strain[2], b.first[2], b.irreps[b.first[1]].dim)
    assert 0.003 < out["plastic"][0] < 0.8 * out["elastic"][0]
    assert 0.0037 < out["crystal"][0] < 0.5 * out["elastic"][0]
    assert out["plastic"][1:] == out["crystal"][1:] == out["elastic"][1:] == (4, 2)
