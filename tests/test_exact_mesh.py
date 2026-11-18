"""Exact meshes without a background grid: point-wise level sets, octrees, adaptive meshes
of level sets, meshes of convex pieces, and the route of each microstructure."""

import numpy as np
import pytest

import morphogen as mg
from morphogen.config import ConfigError, load_config
from morphogen.domain import Grid
from morphogen.mesh.conforming import ImplicitBody, conforming_mesh, mmg_executable
from morphogen.mesh.exact import (
    _to_split,
    adaptive_mesh,
    exact_mesh,
    exact_route,
    grid_boxes,
    polyhedral_mesh,
)
from morphogen.mesh.octree import Octree
from morphogen.mesh.tetra import TetMesh, tet_quality, tet_volumes
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Spheres

try:
    mmg_executable()
    HAS_MMG = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_MMG = False
needs_mmg = pytest.mark.skipif(not HAS_MMG, reason="MMG (pymmg) is not installed")


def _structures():
    cube = mg.Domain.cube(1.0, periodic=True)
    box = mg.Domain.cube(1.0, periodic=False)
    return {
        "spheres": mg.models.BooleanSpheres(radius=0.15, volume_fraction=0.2).generate(cube, 1),
        "gyroid": mg.models.TPMS(
            surface="gyroid", variant="sheet", cell_size=0.5, volume_fraction=0.3
        ).generate(cube),
        "gaussian": mg.models.GaussianExcursion(
            correlation_length=0.1, volume_fraction=0.4
        ).generate(cube, 2),
        "laminate": mg.models.RandomLaminate(
            thickness=0.1, spacing=0.25, hole_fraction=0.2, hole_radius=0.15
        ).generate(cube, 3),
        "fibres": mg.models.PoissonFibers(radius=0.04, volume_fraction=0.1).generate(box, 4),
    }


def test_level_set_at_matches_the_grid():
    """The point-wise level sets agree with the level sets on grids (at the voxel centres)
    and their signs with the phases."""
    for name, micro in _structures().items():
        if micro is None:
            continue
        grid = Grid(micro.domain, (20, 20, 20))
        pts = grid.center_points()
        reach = 4.0 * float(grid.spacing.max())
        exact = micro.level_set_at(pts, reach)
        sampled = micro.level_set(grid=grid).data.astype(float).ravel()
        scale = float(np.max(np.abs(sampled)))
        near = np.abs(sampled) < 0.5 * reach
        assert np.allclose(exact[near], sampled[near], atol=1e-5 * scale), name
        assert np.array_equal(exact < 0, micro.contains(pts)), name


def test_level_set_at_of_compositions():
    cube = mg.Domain.cube(1.0, periodic=True)
    a = mg.models.BooleanSpheres(radius=0.2, volume_fraction=0.2).generate(cube, 1)
    b = mg.models.TPMS(surface="schwarz_p", cell_size=1.0, volume_fraction=0.3).generate(cube)
    pts = np.random.default_rng(0).random((500, 3))
    from morphogen.models.combination import Combined

    union = Combined(cube, "union", [a, b])
    expected = np.minimum(a.level_set_at(pts, 0.5), b.level_set_at(pts, 0.5))
    assert np.allclose(union.level_set_at(pts, 0.5), expected)
    assert union.lipschitz == pytest.approx(max(a.lipschitz, b.lipschitz))
    difference = Combined(cube, "difference", [b, a])
    assert np.array_equal(difference.level_set_at(pts, 0.5) < 0, difference.contains(pts))
    # several phases: the function of phase 1, as on grids
    three = mg.models.DeadLeavesModel(radius=0.2, phase_fractions=[0.3, 0.3, 0.4])
    three = three.generate(cube, 1)
    assert np.array_equal(three.level_set_at(pts, 0.5), three.phase_level_sets_at(pts, 0.5)[1])


@pytest.mark.parametrize("periodic", [False, True])
def test_level_sets_of_partitions(periodic):
    """Dead leaves of three phases and osteons: one level set per phase but 0, negative
    exactly where the phase is."""
    domain = mg.Domain.cube(1.0, periodic=periodic)
    leaves = mg.models.DeadLeavesModel(radius=0.15, phase_fractions=[0.3, 0.3, 0.4])
    micro = leaves.generate(domain, 2)
    pts = Grid(domain, (24, 24, 24)).center_points()
    sets = micro.phase_level_sets_at(pts, 0.1)
    assert sorted(sets) == [0, 1, 2]
    found = np.zeros(len(pts), dtype=int)
    for k, values in sets.items():
        found[values < 0] = k
    labels = micro.cell_phases[micro.leaf_of(pts)]
    assert np.mean(found == labels) > 0.999  # uncovered points: phase 0 here, the nearest
    osteons = mg.models.Osteons().generate(mg.Domain(size=(1000.0, 1000.0, 10.0),
                                                     periodic=periodic), 1)  # fmt: skip
    pts = Grid(osteons.domain, (50, 50, 2)).center_points()
    sets = osteons.phase_level_sets_at(pts, 50.0)
    found = np.zeros(len(pts), dtype=int)
    for k, values in sets.items():
        found[values < 0] = k
    from morphogen.models.planar import plane_axes

    assert np.array_equal(found, osteons._phases(pts[:, list(plane_axes(osteons.axis))]))


def test_split_test():
    """Lines of three values: two crossings, a misplaced crossing and a bend toward zero
    are split; the kink of a distance in the middle of a sheet, a resolved crossing and a
    line far from the interface are not."""
    fa, fm, fb = np.array([[0.2, -0.1, 0.2],  # a sheet between the ends
                           [0.3, 0.3, -0.3],  # the crossing off the linear one
                           [0.2, 0.02, 0.2],  # bends toward zero
                           [-0.05, -0.2, -0.05],  # the middle of a sheet
                           [0.2, 0.0001, -0.2],  # linear
                           [2.0, 2.1, 2.2]]).T  # fmt: skip
    split = _to_split(fa, fm, fb, np.full(6, 0.5), hmin=0.01, hausd=0.01, lipschitz=1.0)
    assert split.tolist() == [True, True, True, False, False, False]
    # not below the smallest size
    assert not _to_split(fa, fm, fb, np.full(6, 0.01), hmin=0.01, hausd=0.01, lipschitz=1.0).any()


@pytest.mark.parametrize("axes", [(False, False, False), (True, True, True), (True, False, True)])
def test_octree_tetrahedra(axes):
    """A balanced octree split into tetrahedra: positive, conforming (the free faces are
    those of the box), the volume of the box, and matching faces along the periodic axes."""
    upper = np.array([1.0, 0.8, 1.2])
    tree = Octree(np.zeros(3), upper, (4, 3, 5), axes, 4)

    def test(points, sides):
        d = np.linalg.norm(points - np.array([0.1, 0.4, 0.6]), axis=2) - 0.35
        return np.any(np.abs(d) < 0.6 * sides.max(axis=1)[:, None], axis=1)

    tree.refine(test)
    tree.balance()
    nodes, tets, tri, refs, _ = tree.tetrahedra()
    assert tet_volumes(nodes, tets).min() > 0
    assert tet_volumes(nodes, tets).sum() == pytest.approx(float(np.prod(upper)))
    assert tet_quality(nodes, tets).min() > 0.5
    faces = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]],
                                    tets[:, [1, 2, 3]]]), axis=1)  # fmt: skip
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    assert counts.max() == 2
    assert np.array_equal(unique[counts == 1], np.unique(np.sort(tri, axis=1), axis=0))
    assert set(np.unique(refs).tolist()) == set(range(1, 7))
    for d in range(3):
        if axes[d]:
            other = [a for a in range(3) if a != d]
            lo = nodes[nodes[:, d] == 0.0][:, other]
            hi = nodes[nodes[:, d] == upper[d]][:, other]
            assert np.array_equal(lo[np.lexsort(lo.T)], hi[np.lexsort(hi.T)])


def _sphere(periodic: bool):
    domain = mg.Domain.cube(1.0, periodic=periodic)
    if periodic:  # across the corners: eight pieces
        return PrimitiveUnion(domain, Spheres(np.zeros((1, 3)), np.array([0.3])))
    return PrimitiveUnion(domain, Spheres(np.full((1, 3), 0.5), np.array([0.3])))


@needs_mmg
@pytest.mark.parametrize("periodic", [False, True])
def test_adaptive_mesh_of_a_sphere(periodic):
    """The nodes of the interface lie on the sphere (to rounding), the volume is that of
    the sphere within its chords, the elements are well shaped, and periodic faces
    match."""
    micro = _sphere(periodic)
    mesh = adaptive_mesh(micro, size=0.2, hausdorff=0.004)
    assert isinstance(mesh, TetMesh)
    shared = np.intersect1d(np.unique(mesh.elements[mesh.phase == 1]),
                            np.unique(mesh.elements[mesh.phase == 0]))  # fmt: skip
    assert np.abs(micro.level_set_at(mesh.nodes[shared], 0.5)).max() < 1e-12
    volume = mesh.volumes()[mesh.phase == 1].sum()
    exact = 4.0 / 3.0 * np.pi * 0.3**3
    assert -0.03 < volume / exact - 1.0 < 0.0  # inscribed facets
    assert mesh.volumes().sum() == pytest.approx(1.0)
    assert mesh.quality().min() > 0.05
    assert mesh.periodic == periodic
    if periodic:
        assert mesh.matching_faces()
    assert mesh.metadata["mesher"].startswith("adaptive level set")


@needs_mmg
def test_curved_quadratic_elements():
    """10-node elements, curved by :func:`morphogen.mesh.curved.curved_mesh` as those of
    the grid: the nodes of the interface on the sphere, every Jacobian positive, and the
    volume of the curved elements far closer to the sphere's than that of the straight
    ones."""
    from morphogen.mesh.exact import _p2_jacobians

    micro = _sphere(False)
    straight = adaptive_mesh(micro, size=0.25, hausdorff=0.01, order=2, project=False)
    mesh = adaptive_mesh(micro, size=0.25, hausdorff=0.01, order=2)
    curved = mesh.metadata["curved"]
    assert mesh.order == 2 and curved["sampling"] == 0.0  # the exact functions at the nodes
    assert curved["nodes"] == curved["interface_nodes"] > 100
    assert "curved" not in straight.metadata and not straight.curved
    assert (_p2_jacobians(mesh.nodes, mesh.elements) > 0).all()
    # the volume of the curved elements: a rule exact for the degree 3 of their Jacobians
    rule = np.array([[0.25] * 4] + [[0.5 if i == j else 1 / 6 for i in range(4)]
                                    for j in range(4)])  # fmt: skip
    weights = np.array([-4.0 / 30.0] + [9.0 / 120.0] * 4)

    def solid(m):
        jac = _p2_jacobians(m.nodes, m.elements[m.phase == 1], rule)
        return float((jac @ weights).sum())

    exact = 4.0 / 3.0 * np.pi * 0.3**3
    assert abs(solid(mesh) / exact - 1.0) < 0.25 * abs(solid(straight) / exact - 1.0)


@needs_mmg
def test_adaptive_mesh_finds_thin_sheets():
    """A sheet a tenth of the size thick, between the nodes of the first cells, is found."""
    domain = mg.Domain.cube(1.0, periodic=False)

    class Sheet:
        def __init__(self):
            self.domain = domain

        def level_set_at(self, points, reach=np.inf):
            return np.abs(np.asarray(points)[:, 2] - 0.4567) - 0.01

    mesh = adaptive_mesh(Sheet(), size=0.2, hausdorff=0.004)
    volume = mesh.volumes()[mesh.phase == 1].sum()
    assert volume == pytest.approx(0.02, rel=1e-6)  # planes: exact


@needs_mmg
def test_polyhedral_meshes():
    """Convex pieces: the volumes of the phases exact; periodic fractal sponges with
    matching faces."""
    sponge = mg.models.FractalSponge(pattern="menger", level=2, cell_size=1.0).generate(
        mg.Domain.cube(1.0, periodic=True)
    )
    pieces, phase, _ = sponge.convex_pieces()
    assert len(pieces) < 27**2 and set(phase.tolist()) == {0, 1}
    mesh = polyhedral_mesh(sponge, size=0.25)
    assert mesh.periodic and mesh.matching_faces()
    fraction = len(sponge.kept) ** 2 / 27**2
    assert mesh.volumes()[mesh.phase == 1].sum() == pytest.approx(fraction, rel=1e-9)
    plates = mg.models.ScaleRichNetwork(dimension=3, n_ligaments=25, decay=0.3,
                                        initial_thickness=0.03).generate(
        mg.Domain.cube(1.0, periodic=False), seed=1)  # fmt: skip
    mesh = exact_mesh(plates, size=0.2, hausdorff=0.01)  # the Hausdorff distance: level sets
    assert mesh.volumes()[mesh.phase == 1].sum() == pytest.approx(plates.volume_fraction,
                                                                  rel=1e-12)  # fmt: skip
    assert mesh.metadata["mesher"].startswith("exact convex pieces")
    # the bends of the facets kept by MMG: no node off the planes of its pieces
    pieces, _, cells = plates.convex_pieces()
    mesh = polyhedral_mesh(plates, size=0.2, by_piece=True)
    for poly, cell in zip(pieces, cells, strict=True):
        if cell > 0 and np.count_nonzero(cells == cell) == 1:  # a plate
            nodes = mesh.nodes[np.unique(mesh.elements[mesh.cell == cell])]
            assert (nodes @ poly[3].T - poly[4]).max() < 1e-12


@needs_mmg
@pytest.mark.parametrize("periodic", [False, True])
def test_exact_cells(periodic):
    """Tessellations: the exact cells (not regularized), their facets kept by MMG."""
    voronoi = mg.models.Voronoi(n_cells=20).generate(mg.Domain.cube(1.0, periodic=periodic), 3)
    mesh = exact_mesh(voronoi, size=0.25)
    exact = voronoi.cell_complex(whole=periodic).cell_volumes(voronoi.n_cells)
    meshed = np.bincount(mesh.cell, weights=mesh.volumes(), minlength=voronoi.n_cells)
    assert np.abs(meshed - exact).max() < 1e-12
    assert mesh.metadata["exact_facets"] and mesh.metadata["min_feature"] == 0.0


def test_grid_boxes():
    labels = np.zeros((4, 3, 2), dtype=np.int64)
    labels[1:3, :, 1] = 1
    edges = [np.linspace(0, 1, 5), np.linspace(0, 1, 4), np.linspace(0, 2, 3)]
    pieces, phase = grid_boxes(labels, edges)
    volumes = [np.prod(p[0].max(axis=0) - p[0].min(axis=0)) for p in pieces]
    assert sum(volumes) == pytest.approx(2.0)
    assert sum(v for v, f in zip(volumes, phase, strict=True) if f == 1) == pytest.approx(0.5)
    assert len(pieces) <= 4


def test_exact_routes():
    cube = mg.Domain.cube(1.0, periodic=True)
    assert exact_route(mg.models.Voronoi(n_cells=8).generate(cube, 1)) == "cells"
    assert exact_route(mg.models.BooleanSpheres(radius=0.2, volume_fraction=0.2)
                       .generate(cube, 1)) == "level_set"  # fmt: skip
    plates = mg.models.ScaleRichNetwork(dimension=3, n_ligaments=10, initial_thickness=0.02)
    assert exact_route(plates.generate(mg.Domain.cube(1.0, periodic=False), 1)) == "pieces"
    assert exact_route(mg.models.DeadLeavesModel(radius=0.2).generate(cube, 1)) == "level_set"
    transparent = mg.models.TransparentDeadLeavesModel(radius=0.2, volume_fraction=0.3)
    transparent = transparent.generate(cube, 1)
    assert exact_route(transparent) == ""
    # partitions of several phases, and the phases of tessellations without their grains
    three = mg.models.DeadLeavesModel(radius=0.2, phase_fractions=[0.3, 0.3, 0.4])
    assert exact_route(three.generate(cube, 1)) == "level_set"
    cells = mg.models.Voronoi(n_cells=8, phase_fractions=[0.5, 0.5]).generate(cube, 1)
    assert exact_route(cells) == "cells" and exact_route(cells, grains=False) == "level_set"
    one = mg.models.Voronoi(n_cells=8).generate(cube, 1)  # one phase: no interface
    assert exact_route(one, grains=False) == ""
    layers = mg.models.Voronoi(n_cells=8, boundary_thickness=0.05).generate(cube, 1)
    assert exact_route(layers) == "level_set"


def test_mesh_method_in_configurations():
    base = {"model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.2},
            "domain": {"size": 5.0}}  # fmt: skip
    cfg = load_config({**base, "mesh": {"size": 1.0, "method": "exact"}})
    assert cfg.mesh.method == "exact" and cfg.mesh.kwargs()["method"] == "exact"
    assert load_config({**base, "mesh": {"size": 1.0}}).mesh.method == "auto"  # the default
    assert load_config({**base, "mesh": {"size": 1.0, "method": "grid"}}).mesh.method == "grid"
    with pytest.raises(ConfigError, match=r"mesh\.method"):
        load_config({**base, "mesh": {"size": 1.0, "method": "voxels"}})


@needs_mmg
def test_conforming_mesh_exact_method():
    micro = _sphere(False)
    mesh = conforming_mesh(micro, size=0.25, hausdorff=0.01, method="exact")
    assert mesh.metadata["mesher"].startswith("adaptive level set")
    with pytest.raises(ValueError, match="method"):
        conforming_mesh(micro, size=0.25, method="voxels")
    # the default: exact when the microstructure has an exact route, the grid otherwise
    default = conforming_mesh(micro, size=0.25, hausdorff=0.01)
    assert default.metadata["mesher"].startswith("adaptive level set")
    assert conforming_mesh(micro, size=0.25, method="grid").metadata["mesher"] == "mmg3d -ls"
    # a level set on grids only: no exact route
    body = ImplicitBody(micro.domain, lambda p: np.linalg.norm(p - 0.5, axis=1) - 0.3)
    assert exact_route(body) == ""
    assert conforming_mesh(body, size=0.25).metadata["mesher"] == "mmg3d -ls"
    with pytest.raises(TypeError, match="no exact mesh"):
        conforming_mesh(body, size=0.25, method="exact")
    # no exact level set: on the grid, from the implicit functions of the phases (§75)
    leaves = mg.models.TransparentDeadLeavesModel(radius=0.3, volume_fraction=0.3)
    leaves = leaves.generate(mg.Domain.cube(1.0, periodic=False), 1)
    assert exact_route(leaves) == ""
    assert conforming_mesh(leaves, size=0.25).metadata["mesher"] == "mmg3d -ls"
