"""Exact cells of tessellations, their regularization, and conforming meshes of
polycrystals (one subdomain per grain)."""

import itertools

import numpy as np
import pytest
from scipy.spatial import cKDTree

import morphogen as mg
from morphogen.homogenization import Material, homogenize
from morphogen.io import meshes
from morphogen.mesh.conforming import conforming_mesh, has_mmg
from morphogen.mesh.polycrystal import _mesh_polygon, _orient, _triangulate_pslg, initial_mesh
from morphogen.mesh.polyhedra import CellComplex, power_complex, torus_classes
from morphogen.mesh.regularize import regularize

needs_mmg = pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")


def tessellation(n, periodic, seed=2, model="voronoi"):
    if model == "voronoi":
        m = mg.models.Voronoi(n_cells=n, crystal_orientations=True, phase_fractions=(0.3, 0.7))
    else:
        m = mg.models.Laguerre(
            n_cells=n,
            radius=mg.distributions.Uniform(0.02, 0.15),
            crystal_orientations=True,
            phase_fractions=(0.3, 0.7),
        )
    return m.generate(mg.Domain.cube(1.0, periodic=periodic), seed=seed)


def check_closed(cx: CellComplex):
    """Every piece is a closed polyhedron of positive volume, its facets oriented
    outwards (each edge used once in each direction)."""
    for p, pf in enumerate(cx.piece_facets):
        edges = {}
        for f in pf:
            loop = list(cx.facets[f]) if cx.facet_pieces[f, 0] == p else list(cx.facets[f])[::-1]
            for a, b in zip(loop, loop[1:] + loop[:1], strict=False):
                edges[(a, b)] = edges.get((a, b), 0) + 1
        assert all(c == 1 and edges.get((b, a)) == 1 for (a, b), c in edges.items()), p
    assert np.all(cx.piece_volumes() > 0)


def check_periodic_faces(cx: CellComplex):
    """The vertices on opposite faces are translates of each other."""
    v = cx.vertices
    size = cx.upper - cx.lower
    for d in range(3):
        lo = v[np.abs(v[:, d] - cx.lower[d]) < 1e-12]
        hi = v[np.abs(v[:, d] - cx.upper[d]) < 1e-12] - np.eye(3)[d] * size[d]
        assert len(lo) == len(hi) > 0
        assert np.max(cKDTree(hi).query(lo)[0]) < 1e-9


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("model", ["voronoi", "laguerre"])
def test_exact_cells(periodic, model):
    tess = tessellation(40, periodic, model=model)
    cx = tess.cell_complex()
    check_closed(cx)
    assert cx.piece_volumes().sum() == pytest.approx(1.0, abs=1e-10)
    # the vertices of a facet are equally close (in power distance) to its two germs,
    # and no germ is closer
    w = tess.weights if tess.weights is not None else np.zeros(tess.n_cells)
    shifts = [np.zeros(3)]
    if periodic:
        shifts = [np.array(s, float) for s in np.ndindex(3, 3, 3)]
        shifts = [s - 1 for s in shifts]
    for f in np.flatnonzero(cx.facet_pieces[:, 1] >= 0)[::7]:
        pts = cx.vertices[cx.facets[f]]
        power = (
            np.min(
                [np.sum((pts[:, None] - tess.germs[None] - s) ** 2, axis=2) for s in shifts], axis=0
            )
            - w[None, :]
        )
        a, b = cx.piece_cell[cx.facet_pieces[f]]
        best = power.min(axis=1)
        assert np.allclose(power[:, a], best, atol=1e-9) and np.allclose(
            power[:, b], best, atol=1e-9
        )
    # cell volumes against counts of voxels
    labels = tess.labels((64, 64, 64)).data
    counts = np.bincount(labels.ravel(), minlength=tess.n_cells) / labels.size
    vol = cx.cell_volumes(tess.n_cells)
    assert np.abs(counts - vol).sum() < 0.02
    if periodic:
        check_periodic_faces(cx)
        assert len(np.unique(cx.piece_cell)) == tess.n_cells


@pytest.mark.parametrize("periodic", [False, True])
def test_regularization_keeps_valid_cells(periodic):
    tess = tessellation(60, periodic, seed=5)
    cx = tess.cell_complex()
    delta = 0.02
    reg = regularize(cx, delta)
    check_closed(reg)
    assert reg.piece_volumes().sum() == pytest.approx(1.0, abs=1e-9)

    def short(c):
        e = c.edges()
        return int(
            (np.linalg.norm(c.vertices[e[:, 0]] - c.vertices[e[:, 1]], axis=1) < delta).sum()
        )

    assert short(cx) > 20 and short(reg) < 0.25 * short(cx)
    md = reg.metadata
    assert md["regularization"] == delta and md["contracted_edges"] > 0
    if periodic:
        check_periodic_faces(reg)
    # cells change by a small fraction of their volume
    v0, v1 = cx.cell_volumes(tess.n_cells), reg.cell_volumes(tess.n_cells)
    assert np.abs(v1 - v0).sum() < 0.03


@needs_mmg
@pytest.mark.parametrize("periodic", [False, True])
def test_polycrystal_mesh(periodic, tmp_path):
    tess = tessellation(25, periodic, seed=7)
    mesh = conforming_mesh(tess, size=0.1)
    md = mesh.metadata
    assert md["mesher"].startswith("polycrystal") and md["matching_faces"] == periodic
    assert mesh.volumes().min() > 0
    # MMG keeps the matching faces of periodic meshes, and remeshes the others
    assert mesh.volumes().sum() == pytest.approx(1.0, abs=1e-12 if periodic else 1e-5)
    q = mesh.quality()
    assert q.min() > 5e-4 and q.mean() > 0.6
    # grains: phases and orientations of their cells, volumes of the cells
    assert np.array_equal(mesh.phase, tess.cell_phases[mesh.cell])
    assert mesh.orientations is tess.orientations or np.array_equal(
        mesh.orientations, tess.orientations
    )
    exact = tess.cell_complex().cell_volumes(tess.n_cells)
    meshed = np.bincount(mesh.cell, weights=mesh.volumes(), minlength=tess.n_cells)
    big = exact > 0.01
    assert np.median(np.abs(meshed[big] - exact[big]) / exact[big]) < 0.02
    if periodic:
        assert mesh.matching_faces()
        assert np.unique(mesh.cell).size == tess.n_cells
    # exports keep the grains
    back = meshes.read_tet_mesh(meshes.write_vtu_tet(tmp_path / "m.vtu", mesh))
    assert np.array_equal(back.cell, mesh.cell)
    used = np.unique(mesh.cell)
    assert np.allclose(back.orientations[used], tess.orientations[used])
    back = meshes.read_tet_mesh(meshes.write_medit_tet(tmp_path / "m.mesh", mesh))
    assert np.array_equal(back.cell, mesh.cell) and np.array_equal(back.phase, mesh.phase)
    text = meshes.write_inp_tet(tmp_path / "m.inp", mesh, materials={0: "A", 1: "B"}).read_text(
        encoding="utf-8"
    )
    assert text.count("*ELSET, ELSET=CELL_") == used.size == text.count("*ORIENTATION")
    assert text.count("*SOLID SECTION") == used.size
    # the solid phase only, 10-node elements
    solid = conforming_mesh(tess, size=0.1, phases="solid", order=2)
    assert solid.order == 2 and set(np.unique(solid.phase)) == {1}
    assert solid.volumes().sum() == pytest.approx(exact[tess.cell_phases == 1].sum(), rel=0.03)


@needs_mmg
def test_polycrystal_mesh_of_exact_cells():
    """Without regularization, the mesh reproduces the exact cells."""
    tess = tessellation(8, True, seed=1)
    mesh = conforming_mesh(tess, size=0.15, min_feature=0)
    exact = tess.cell_complex().cell_volumes(tess.n_cells)
    meshed = np.bincount(mesh.cell, weights=mesh.volumes(), minlength=tess.n_cells)
    assert np.allclose(meshed, exact, rtol=1e-6, atol=1e-9)
    assert mesh.matching_faces()


def test_constrained_triangulation_of_faces():
    """Triangulations of planar straight-line graphs (the traces of the grain boundaries
    on a face of the box): every segment is an edge, the triangles tile the square
    counter-clockwise, the result does not depend on the order of the segments."""
    rng = np.random.default_rng(5)
    square = [(0, 1), (1, 2), (2, 3), (3, 0)]

    def check(points, segments):
        t = _triangulate_pslg(points, np.array(segments))
        u, v = points[t[:, 1]] - points[t[:, 0]], points[t[:, 2]] - points[t[:, 0]]
        area = 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        assert np.all(area > 0) and area.sum() == pytest.approx(1.0, abs=1e-12)
        edges = {tuple(sorted(e)) for tri in t.tolist() for e in itertools.combinations(tri, 2)}
        assert np.unique(t).size == len(points)
        return t, edges

    for _ in range(20):
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        points = np.concatenate([corners, rng.random((30, 2))])
        segments = list(square)
        for a, b in rng.permutation(list(itertools.combinations(range(4, 34), 2)))[:60]:
            if not any(_crossing(points, (a, b), s) for s in segments):
                segments.append((int(a), int(b)))
        t, edges = check(points, segments)
        assert all(tuple(sorted(s)) in edges for s in segments)
        again, _ = check(points, [s[::-1] for s in rng.permutation(segments).tolist()])
        assert np.array_equal(t, again)
    # a segment through points (split there), a tiny segment, nearly collinear points
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.2, 0.5], [0.5, 0.5], [0.8, 0.5],
                       [0.5, 0.5 + 1e-9], [0.8 + 1e-12, 0.2], [0.5, 0.2 - 1e-13]])  # fmt: skip
    _, edges = check(points, [*square, (4, 6), (5, 7), (8, 9)])
    assert {(4, 5), (5, 6), (5, 7), (8, 9)} <= edges


def _crossing(points, s, t) -> bool:
    if set(s) & set(t):
        return False
    a, b, c, d = points[[s[0], s[1], t[0], t[1]]]
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    return o1 * o2 <= 0 and o3 * o4 <= 0


@needs_mmg
@pytest.mark.parametrize(
    ("model", "n", "seed", "size"), [("voronoi", 25, 76, 0.1), ("laguerre", 40, 1, 0.08)]
)
def test_matching_faces_of_hard_cases(model, n, seed, size, tmp_path):
    """Tessellations whose faces mmg2d could not mesh from their traces alone."""
    tess = tessellation(n, True, seed=seed, model=model)
    cx = regularize(tess.cell_complex(), size / 4)
    sizing = ["-hmax", str(size), "-hmin", str(size / 10), "-hausd", str(size / 20), "-v", "-1"]
    xyz, _, _, tri, tri_refs, _ = initial_mesh(
        cx, size=size, sizing=sizing, periodic=True, folder=tmp_path, timeout=60
    )
    for axis in range(3):
        lo, hi = (xyz[tri[tri_refs == 2 * axis + s]].mean(axis=1) for s in (1, 2))
        lo[:, axis] += 1.0
        dist, _ = cKDTree(hi).query(lo)
        assert len(lo) == len(hi) and dist.max() < 1e-12


def test_power_complex_of_a_lattice():
    """Germs on a cubic lattice: cubic cells; weights equal: the same cells."""
    g = (np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) / 3
    rng = np.random.default_rng(0)
    g = g + 1e-3 * rng.standard_normal(g.shape)
    for weights in (None, np.full(len(g), 0.01)):
        cx = power_complex(g, weights, np.zeros(3), np.ones(3), periodic=True)
        vol = cx.cell_volumes(len(g))
        assert np.allclose(vol, 1 / 27, rtol=0.02) and vol.sum() == pytest.approx(1.0)


def outer_pairs(cx: CellComplex) -> list:
    """Facets bounding whole cells, grouped by translation (sorted classes of vertices)."""
    labels, _ = torus_classes(cx.vertices, cx.lower, cx.upper - cx.lower, 1e-9)
    groups = {}
    for f in np.flatnonzero(cx.facet_pieces[:, 1] < 0):
        groups.setdefault(tuple(sorted(labels[cx.facets[f]])), []).append(int(f))
    return list(groups.values())


@pytest.mark.parametrize("model", ["voronoi", "laguerre"])
def test_whole_periodic_cells(model):
    """Whole cells: one piece per cell, the same volumes as the cells cut by the box,
    filling its volume; the facets bounding them come in pairs of translates."""
    tess = tessellation(40, True, seed=3, model=model)
    whole = tess.cell_complex(whole=True)
    assert whole.whole and whole.periodic and tess.cell_complex(whole=True) is whole
    check_closed(whole)
    assert len(np.unique(whole.piece_cell)) == whole.n_pieces
    cut = tess.cell_complex().cell_volumes(tess.n_cells)
    assert np.allclose(whole.cell_volumes(tess.n_cells), cut, atol=1e-12)
    assert whole.piece_volumes().sum() == pytest.approx(1.0, abs=1e-12)
    pairs = outer_pairs(whole)
    assert pairs and all(len(p) == 2 for p in pairs)
    for f, g in pairs:
        a, b = whole.vertices[whole.facets[f]], whole.vertices[whole.facets[g]]
        shift = whole.facet_shift[f]
        assert np.array_equal(shift, -whole.facet_shift[g]) and np.any(shift)
        # beyond f lies the cell of g translated by a period: g is f translated back
        dist, _ = cKDTree(b).query(a - shift * (whole.upper - whole.lower))
        assert dist.max() < 1e-12
    # regularized: still pairs of translates, the same total volume
    reg = regularize(whole, 0.03)
    check_closed(reg)
    assert reg.whole and reg.metadata["contracted_edges"] > 0
    assert reg.piece_volumes().sum() == pytest.approx(1.0, abs=1e-12)
    assert all(len(p) == 2 for p in outer_pairs(reg))
    with pytest.raises(ValueError, match="periodic"):
        tessellation(10, False).cell_complex(whole=True)


def test_polygons_with_collinear_points():
    """Facets with subdivided sides (collinear points but for rounding): triangles of
    positive area tiling the polygon, every side kept, no flat triangle."""
    rng = np.random.default_rng(4)
    for _ in range(30):
        k = int(rng.integers(3, 9))
        angles = np.sort(rng.uniform(0, 2 * np.pi, k))
        radius = rng.uniform(0.05, 0.2, k)
        corners = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        hull = corners[_convex(corners)]
        frame = np.linalg.qr(rng.standard_normal((3, 3)))[0]
        loop = []
        for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True):
            n = int(rng.integers(1, 5))
            loop += [a + (b - a) * j / n for j in range(n)]
        loop = np.column_stack([np.array(loop), np.zeros(len(loop))]) @ frame.T + rng.random(3)
        lifted, tri = _mesh_polygon(loop, float(rng.uniform(0.02, 0.1)))
        pts = np.vstack([loop, lifted])
        cross = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
        normal = frame[:, 2] * np.sign(cross.sum(axis=0) @ frame[:, 2])
        area = 0.5 * cross @ normal
        u = corners[_convex(corners)]
        exact = 0.5 * abs(np.sum(u[:, 0] * np.roll(u[:, 1], -1) - np.roll(u[:, 0], -1) * u[:, 1]))
        assert area.min() > 1e-6 * exact and area.sum() == pytest.approx(exact, rel=1e-10)
        edges = {tuple(sorted(e)) for t in tri.tolist() for e in itertools.combinations(t, 2)}
        m = len(loop)
        assert all(tuple(sorted((i, (i + 1) % m))) in edges for i in range(m))


def _convex(points):
    from scipy.spatial import ConvexHull

    return ConvexHull(points).vertices


COPPER = Material.cubic(168.4, 121.4, 75.4, conductivity=400.0)


@pytest.mark.slow
@needs_mmg
def test_whole_polycrystal_mesh(tmp_path):
    """Meshes of whole periodic grains: they fill the volume of the box, their boundary is
    made of translated pairs of faces with the same triangles, every node of it has its
    translates; the bulk modulus of a cubic polycrystal is exact for every condition."""
    tess = tessellation(25, True, seed=7)
    mesh = conforming_mesh(tess, size=0.1)
    assert mesh.whole and mesh.metadata["cells"] == "whole" and mesh.matching_faces()
    assert mesh.volumes().min() > 0 and mesh.volumes().sum() == pytest.approx(1.0, abs=1e-12)
    slave, master, shift = mesh.periodic_links()
    assert len(slave) and np.abs(shift).max() == 1
    assert np.abs(mesh.nodes[slave] - mesh.nodes[master] - shift).max() < 1e-12
    outer = mesh.outer_faces()
    assert np.isin(np.unique(outer), np.union1d(slave, master)).all()
    # every node of the box faces of a mesh of the box is matched alike
    assert not conforming_mesh(tess, size=0.1, cells="box").whole
    # whole grains: the volumes of the cells (not cut by the box)
    exact = tess.cell_complex().cell_volumes(tess.n_cells)
    meshed = np.bincount(mesh.cell, weights=mesh.volumes(), minlength=tess.n_cells)
    big = exact > 0.01
    assert np.median(np.abs(meshed[big] - exact[big]) / exact[big]) < 0.02
    k = (168.4 + 2 * 121.4) / 3
    for m in (mesh, mesh.quadratic()):
        res = homogenize(
            m, {0: COPPER, 1: COPPER}, bcs=["kubc", "periodic", "subc"],
            properties=["bulk_modulus", "shear_modulus", "conductivity"],
        )  # fmt: skip
        for bc in ("kubc", "periodic", "subc"):
            assert res[f"bulk_modulus.{bc}"].value == pytest.approx(k, rel=1e-9)
            assert res[f"conductivity.{bc}"].value == pytest.approx(400.0, rel=1e-9)
        shear = [res[f"shear_modulus.{bc}"].value for bc in ("kubc", "periodic", "subc")]
        assert shear[0] > shear[1] > shear[2]
    with pytest.raises(ValueError, match="cells='box'"):
        homogenize(mesh, {0: COPPER, 1: COPPER}, bcs=["mixed"])
    # files keep the boundary of the whole cells
    for name in ("m.vtu", "m.mesh"):
        path = meshes.write_vtu_tet(tmp_path / name, mesh) if name.endswith("vtu") else (
            meshes.write_medit_tet(tmp_path / name, mesh))  # fmt: skip
        back = meshes.read_tet_mesh(path)
        assert back.whole and np.array_equal(back.outer, mesh.outer) and back.matching_faces()
    rows = (
        meshes.write_periodic_pairs(tmp_path / "p.csv", mesh)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert rows[0] == "node,image,shift_x,shift_y,shift_z" and len(rows) == len(slave) + 1
    text = meshes.write_inp_tet(tmp_path / "m.inp", mesh).read_text(encoding="utf-8")
    assert "*NSET, NSET=OUTER" in text and "NSET=X0" not in text
    # grains of phase 1 only: the faces facing the other grains are walls of pores
    solid = mesh.subset([1])
    assert solid.whole and not solid.matching_faces()
    assert len(solid.outer_faces()) < len(outer) and len(solid.periodic_links()[0])
    res = homogenize(solid, {1: COPPER}, bcs=["kubc", "periodic"], properties=["bulk_modulus"])
    assert 0 < res["bulk_modulus.periodic"].value < res["bulk_modulus.kubc"].value < k
