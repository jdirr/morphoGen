"""Automata on other lattices than the square grid: tilings, their neighbourhoods, cells,
rules and models (triangles, hexagons, Penrose rhombi, Ammann-Beenker tiles, prisms, BCC
and FCC cells, golden rhombohedra), reaction-diffusion on them, and the periodic
approximants of the quasi-periodic tilings."""

import hashlib
import math
import re

import numpy as np
import pytest
from scipy.spatial import ConvexHull, cKDTree

import morphogen as mg
from morphogen.automata import CellularAutomaton, LifeRule, make_tiling
from morphogen.automata.reaction import gray_scott, gray_scott_tiling, stable_substeps
from morphogen.automata.tilings import (
    GOLDEN,
    GOLDEN_RHOMBOHEDRA,
    ICOSAHEDRAL_STARS,
    LATTICES,
    PARALLELEPIPED_FACES,
    QUASIPERIODIC,
    SILVER,
    TilingContext,
    approximant_period,
    window_stars,
)
from morphogen.domain import Domain, Grid
from morphogen.models import ActivatorInhibitorModel, CellularAutomatonModel, TilingField
from morphogen.models.base import Microstructure


def _sizes(tiling, kind):
    return np.unique(np.diff(tiling.adjacency(kind)[0])).tolist()


def _symmetric(indptr, indices):
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    pairs = set(zip(rows.tolist(), indices.tolist(), strict=True))
    return all((j, i) in pairs for i, j in pairs) and all(i != j for i, j in pairs)


# -- neighbourhoods --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "dims", "faces", "edges", "vertices"),
    [
        ("square", 2, 4, 8, 8),
        ("triangular", 2, 3, 12, 12),
        ("hexagonal", 2, 6, 6, 6),
        ("square", 3, 6, 18, 26),
        ("triangular", 3, 5, 20, 38),
        ("hexagonal", 3, 8, 20, 20),  # hexagonal prisms
        ("bcc", 3, 14, 14, 14),  # truncated octahedra
        ("fcc", 3, 12, 12, 18),  # rhombic dodecahedra
    ],
)
def test_neighbour_counts(name, dims, faces, edges, vertices):
    size = (7, 7, 7) if dims == 3 else (13, 13, 1)
    t = make_tiling(name, size, 1.0, dims=dims)
    assert [_sizes(t, k) for k in ("faces", "edges", "vertices")] == [[faces], [edges], [vertices]]
    for kind in ("faces", "vertices"):
        assert _symmetric(*t.adjacency(kind))
    assert np.allclose(t.volumes.sum(), np.prod(size if dims == 3 else size[:2]))
    # the free edges of a non-periodic tiling: fewer neighbours, never more
    free = make_tiling(name, size, 1.0, dims=dims, periodic=False)
    assert max(_sizes(free, "vertices")) == vertices and min(_sizes(free, "faces")) < faces


def test_neighbour_shells_of_cubic_lattices():
    for name, near, far in (("bcc", 8, 6), ("fcc", 12, 6)):
        t = make_tiling(name, (6, 6, 6), 1.0, dims=3)
        a = (2 if name == "bcc" else 4) ** (1 / 3)
        kind = "faces" if name == "bcc" else "vertices"
        indptr, indices = t.adjacency(kind)
        d = t.centres[indices[: indptr[1]]] - t.centres[0]
        d -= t.size * np.round(d / t.size)  # minimum image
        r = np.linalg.norm(d / (t.size / (np.round(t.size / a) * a)), axis=1) / a
        first = math.sqrt(3) / 2 if name == "bcc" else 1 / math.sqrt(2)
        assert np.allclose(np.sort(r), [first] * near + [1.0] * far)
    # the Voronoi cells: 24 corners (truncated octahedron), 14 (rhombic dodecahedron)
    for name, corners, volume in (("bcc", 24, 0.5), ("fcc", 14, 0.25)):
        t = make_tiling(name, (4, 4, 4), 1.0, dims=3)
        a = (2 if name == "bcc" else 4) ** (1 / 3)
        hull = ConvexHull(t._solid["prototype"])
        assert len(hull.vertices) == corners
        assert hull.volume == pytest.approx(volume * a**3, rel=1e-12)


def test_neighbourhoods_and_errors():
    t = make_tiling("hexagonal", (16, 16, 1), 1.0)
    assert _sizes(t, "faces") == [6]
    assert np.unique(np.diff(t.neighbours("von_neumann", 2)[0])).tolist() == [18]
    assert np.unique(np.diff(t.neighbours("moore", 3)[0])).tolist() == [36]
    a = math.sqrt(2 / math.sqrt(3))  # centre spacing
    assert np.unique(np.diff(t.neighbours("euclidean", 1.01 * a)[0])).tolist() == [6]
    assert np.unique(np.diff(t.ball(1.01 * a)[0])).tolist() == [7]  # with the cell
    with pytest.raises(ValueError, match="lattice"):
        make_tiling("kagome", (8, 8, 1))
    with pytest.raises(ValueError, match="three-dimensional"):
        make_tiling("bcc", (8, 8, 1), dims=2)
    with pytest.raises(ValueError, match="quasi-periodic"):
        make_tiling("penrose", (8, 8, 1), periodic=True)
    with pytest.raises(ValueError, match="neighbourhood"):
        t.neighbours("hexagonal")
    # a Euclidean radius that reaches no centre would leave every cell unchanged (the
    # nearest centres 1.075, 1.091 and 1.122 apart in undistorted lattices, the box's here)
    for name, dims in (("hexagonal", 2), ("bcc", 3), ("fcc", 3)):
        lattice = make_tiling(name, (8, 8, 8), 1.0, dims=dims)
        with pytest.raises(ValueError, match="no cell centre within it") as error:
            lattice.neighbours("euclidean", 1)
        gap = float(re.search(r"the nearest ([0-9.]+) cell sizes", str(error.value)).group(1))
        assert gap > 1 and lattice.neighbours("euclidean", 1.002 * gap)[0][-1] > 0


# -- cells -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "dims", "per_cell", "tol"),
    [
        ("triangular", 2, 16, 0.06),
        ("hexagonal", 2, 16, 0.04),
        ("penrose", 2, 16, 0.08),
        ("ammann_beenker", 2, 16, 0.08),
        ("bcc", 3, 10, 0.05),
        ("fcc", 3, 10, 0.1),
        ("hexagonal", 3, 10, 0.08),
        ("icosahedral", 3, 10, 0.05),
    ],
)
def test_rasterized_cells(name, dims, per_cell, tol):
    """Each voxel takes the cell containing its centre: the voxels of a cell measure its
    area or volume."""
    size = (6, 6, 6) if dims == 3 else (12, 12, 1)
    periodic = name not in QUASIPERIODIC
    t = make_tiling(name, size, 1.0, dims=dims, periodic=periodic, rng=np.random.default_rng(4))
    shape = tuple(round(s * per_cell) if (a < 2 or dims == 3) else 1 for a, s in enumerate(size))
    grid = Grid(Domain(size=size, periodic=periodic), shape)
    idx = t.cell_index(grid)
    measured = np.bincount(idx.ravel(), minlength=t.n_cells) * grid.voxel_volume
    k = 2 if dims == 2 else 3
    c = np.asarray(size[:k]) / 2
    inner = np.all(np.abs(t.centres[:, :k] - c) < c - 1, axis=1) if not periodic else slice(None)
    rel = measured[inner] / t.volumes[inner] - 1
    assert np.abs(rel).max() < tol and np.abs(rel).mean() < tol / 2
    # the cell found contains the point: points are nearer their cell than any vertex span
    pts = grid.center_points()[::97]
    d = pts - t.centres[t.locate(pts)]
    if periodic:
        d -= t.size * np.round(d / t.size)  # minimum image
    if dims == 2:
        d = d[:, :2]
    assert np.linalg.norm(d, axis=1).max() <= t.circumradius * (1 + 1e-9)
    if dims == 2:  # the polygons: counterclockwise, the area of the cell
        for i in (0, t.n_cells // 2):
            x, y = t.cell_vertices(i).T
            area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
            assert area == pytest.approx(t.volumes[i], rel=1e-9)
    elif name == "icosahedral":  # the corners of the rhombohedra span their volumes
        for i in range(0, t.n_cells, 37):
            assert ConvexHull(t.cell_vertices(i)).volume == pytest.approx(t.volumes[i], rel=1e-12)


def test_penrose_tiling():
    t = make_tiling("penrose", (60, 60, 1), 1.0, periodic=False, rng=np.random.default_rng(1))
    inner = np.all(np.abs(t.centres[:, :2] - 30) < 25, axis=1)
    thick = np.count_nonzero(t.kinds[inner] == 0)
    assert thick / (inner.sum() - thick) == pytest.approx(GOLDEN, rel=0.02)
    assert t.volumes[inner].mean() == pytest.approx(1.0, rel=0.01)  # cells of mean area 1
    assert _sizes(t, "faces")[-1] == 4  # edge to edge: an edge is shared by two rhombi
    assert np.unique(np.diff(t.adjacency("vertices")[0])[inner]).tolist() == [7, 8, 9, 10, 11]
    assert set(t._plane["vertex_index"].tolist()) == {1, 2, 3, 4}  # sum of the offsets: 0
    # de Bruijn's eight vertex stars (S, S3, S4, S5, J, K, Q, D), angles in units of 36°,
    # and their frequencies, powers of the golden mean
    tau, r5 = GOLDEN, math.sqrt(5)
    eight = {
        (2, (3, 3, 4)): tau**-2,
        (2, (1, 2, 1, 3, 3)): tau**-3,
        (1, (2, 4, 4)): tau**-4,
        (1, (2, 2, 2, 4)): tau**-5,
        (2, (1, 1, 2, 1, 1, 2, 2)): tau**-6,
        (1, (2, 2, 2, 2, 2)): tau**-5 / r5,
        (2, (1, 1, 2, 2, 2, 2)): tau**-7,
        (2, (2, 2, 2, 2, 2)): tau**-7 / r5,
    }
    stars = t.vertex_stars()
    assert set(stars) == set(eight)
    total = sum(stars.values())
    for kind, freq in eight.items():
        assert stars[kind] / total == pytest.approx(freq, rel=0.15)
    # a pentagrid whose offsets do not sum to an integer: generalized Penrose tilings
    other = make_tiling(
        "penrose", (30, 30, 1), periodic=False, offsets=[0.1, 0.2, 0.3, 0.15, -0.25]
    )
    assert len(other.vertex_stars()) > 8
    # singular pentagrids, three lines through one point: their rhombi would overlap
    for offsets in ([0, 0, 0, 0, 0], [0.25, -0.25, 0, 0, 0], [0.5, 0.5, 0.5, 0.5, -2]):
        with pytest.raises(ValueError, match="singular pentagrid"):
            make_tiling("penrose", (16, 16, 1), periodic=False, offsets=offsets)


def _taper(points, centre, half):
    """Smooth window (cos^2 along each axis) over a cube: tile frequencies converge much
    faster than with a sharp window (the window's discrepancy)."""
    x = (np.asarray(points) - centre) / half
    return np.prod(np.where(np.abs(x) < 1, np.cos(np.pi * x / 2) ** 2, 0.0), axis=1)


def _clipped_area(poly, lo, hi):
    """Area of a convex polygon clipped to the rectangle [lo, hi] (Sutherland-Hodgman)."""
    for axis in (0, 1):
        for sign, bound in ((1, hi[axis]), (-1, lo[axis])):
            out = []
            for p, q in zip(poly, np.roll(poly, -1, axis=0), strict=True):
                fp, fq = sign * (p[axis] - bound), sign * (q[axis] - bound)
                if fp <= 0:
                    out.append(p)
                if fp * fq < 0:
                    out.append(p + fp / (fp - fq) * (q - p))
            if len(out) < 3:
                return 0.0
            poly = np.array(out)
    x, y = poly.T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def test_ammann_beenker_tiling():
    rng = np.random.default_rng(1)
    t = make_tiling("ammann_beenker", (80, 80, 1), 1.0, periodic=False, rng=rng)
    edge = mg.automata.tilings.tile_edge("ammann_beenker")
    assert np.allclose(np.unique(np.round(t.volumes, 12)), [edge**2 / math.sqrt(2), edge**2])
    # squares and rhombi in the ratio 1 : sqrt 2 (their areas 1 : 1/sqrt 2): equal areas
    w = _taper(t.centres[:, :2], 40.0, 37.0)
    squares = t.kinds == 0
    assert w[~squares].sum() / w[squares].sum() == pytest.approx(math.sqrt(2), rel=1e-3)
    assert (w * t.volumes)[~squares].sum() / (w * t.volumes)[squares].sum() == pytest.approx(
        1.0, rel=1e-3
    )
    assert np.sum(w * t.volumes) / w.sum() == pytest.approx(1.0, rel=1e-3)  # mean area 1
    assert _sizes(t, "faces")[-1] == 4  # edge to edge
    # no gap or overlap: the cells clipped to the box cover it exactly; a disc (V - E + F)
    lo, hi = np.zeros(2), np.full(2, 80.0)
    covered = sum(_clipped_area(t.cell_vertices(i), lo, hi) for i in range(t.n_cells))
    assert covered == pytest.approx(80.0**2, rel=1e-12)
    ids = t._plane["vertex_ids"].reshape(-1, 4)
    sides = np.sort(np.stack([ids, np.roll(ids, -1, axis=1)], axis=2).reshape(-1, 2), axis=1)
    assert len(np.unique(ids)) - len(np.unique(sides, axis=0)) + t.n_cells == 1
    # the six vertex stars (angles in units of 45°), one per coordination 3 to 8, with the
    # frequencies of the areas of their regions of the octagonal window (silver mean λ)
    lam = SILVER
    six = {
        (2, 3, 3): 1 / lam,
        (1, 2, 3, 2): 2 / lam**2,
        (1, 2, 1, 2, 2): 2 / lam**3,
        (1, 1, 1, 2, 1, 2): 2 / lam**4,
        (1, 1, 1, 1, 1, 1, 2): 1 / lam**5,
        (1, 1, 1, 1, 1, 1, 1, 1): 1 / lam**4,
    }
    assert sum(six.values()) == pytest.approx(1.0, rel=1e-12)
    stars = t.vertex_stars()
    assert set(stars) == set(six)
    total = sum(stars.values())
    for kind, freq in six.items():
        assert stars[kind] / total == pytest.approx(freq, rel=0.06)
    with pytest.raises(ValueError, match="singular multigrid"):
        make_tiling("ammann_beenker", (16, 16, 1), periodic=False, offsets=[0, 0, 0, 0])
    with pytest.raises(ValueError, match="4 numbers"):
        make_tiling("ammann_beenker", (16, 16, 1), periodic=False, offsets=[0.1, 0.2, 0.3])


#: The 12 edges of a parallelepiped: its corners whose bits differ by one.
EDGES = [(i, i | 1 << b) for b in range(3) for i in range(8) if not i >> b & 1]


def test_icosahedral_tiling():
    rng = np.random.default_rng(3)
    t = make_tiling("icosahedral", (32, 32, 32), 1.0, dims=3, periodic=False, rng=rng)
    edge = mg.automata.tilings.tile_edge("icosahedral")
    prolate = t.kinds == 0
    assert np.allclose(t.volumes[prolate], GOLDEN_RHOMBOHEDRA[0] * edge**3)
    assert np.allclose(t.volumes[~prolate], GOLDEN_RHOMBOHEDRA[1] * edge**3)
    assert GOLDEN_RHOMBOHEDRA[0] / GOLDEN_RHOMBOHEDRA[1] == pytest.approx(GOLDEN, rel=1e-12)
    w = _taper(t.centres, 16.0, 14.0)
    assert w[prolate].sum() / w[~prolate].sum() == pytest.approx(GOLDEN, rel=0.015)
    assert np.sum(w * t.volumes) / w.sum() == pytest.approx(1.0, rel=5e-3)  # mean volume 1
    # faces (4 vertices of Z^6), edges and vertices of the cells
    s = t._solid
    ids = s["vertex_ids"]
    cube = mg.automata.tilings._CUBE  # the corners, in the order of the vertex numbers
    corners = s["corner"][:, None] + np.einsum("vk,nkj->nvj", cube, s["frames"][s["triple"]])
    faces = np.sort(ids[:, PARALLELEPIPED_FACES].reshape(-1, 4), axis=1)
    face_at = corners[:, PARALLELEPIPED_FACES].reshape(-1, 4, 3).mean(axis=1)
    _, first, shared = np.unique(faces, axis=0, return_index=True, return_counts=True)
    assert shared.max() == 2  # never three cells on a face
    inner = np.all(np.abs(face_at[first] - 16) < 13, axis=1)
    assert np.all(shared[inner] == 2)  # every inner face between exactly two cells
    sides = np.unique(np.concatenate([np.sort(ids[:, e], axis=1) for e in EDGES]), axis=0)
    vertex_at = np.zeros((ids.max() + 1, 3))
    vertex_at[ids.ravel()] = corners.reshape(-1, 3)
    side_at = 0.5 * (vertex_at[sides[:, 0]] + vertex_at[sides[:, 1]])
    n_v, n_e, n_f, n_c = len(vertex_at), len(sides), len(first), t.n_cells
    assert n_v - n_e + n_f - n_c == 1  # a ball
    # densities: as many vertices as cells (the window over the covolume), 3 faces and 3
    # edges per cell (each cell has 6 faces, shared by two; Euler: V - E + F - C = 0)
    cells = w.sum()
    assert _taper(vertex_at, 16.0, 14.0).sum() / cells == pytest.approx(1.0, rel=1e-3)
    assert _taper(side_at, 16.0, 14.0).sum() / cells == pytest.approx(3.0, rel=1e-3)
    assert _taper(face_at[first], 16.0, 14.0).sum() / cells == pytest.approx(3.0, rel=1e-3)
    inside = np.all(np.abs(t.centres - 16) < 12, axis=1)
    assert np.all(np.diff(t.adjacency("faces")[0])[inside] == 6)
    # no gap or overlap: random points of the inner box lie in exactly one cell
    points = np.random.default_rng(0).uniform(6, 26, size=(4000, 3))
    found = t.locate(points)
    near = cKDTree(t.centres).query_ball_point(points, t.circumradius * (1 + 1e-9))
    for p, cand, c in zip(points, near, found, strict=True):
        cand = np.array(cand)
        u = np.einsum("kj,kji->ki", p - s["corner"][cand], s["inverse"][s["triple"][cand]])
        hit = cand[np.all((u > -1e-12) & (u < 1 + 1e-12), axis=1)]
        assert len(hit) == 1 and hit[0] == c
    with pytest.raises(ValueError, match="three-dimensional"):
        make_tiling("icosahedral", (8, 8, 8))
    with pytest.raises(ValueError, match="quasi-periodic"):
        make_tiling("icosahedral", (8, 8, 8), dims=3)
    with pytest.raises(ValueError, match="singular icosahedral grid"):
        make_tiling("icosahedral", (8, 8, 8), dims=3, periodic=False, offsets=np.zeros(6))


def _icosahedral_group() -> np.ndarray:
    """The 120 symmetries of the icosahedron as permutations of the twelve directions of
    its fivefold axes, numbered ``2 j + s`` (``+e_j``: s = 0; ``-e_j``: s = 1): the
    orthogonal maps that send three of them onto three with the same angles."""
    e = mg.automata.tilings._icosahedral_axes()
    d = np.stack([e, -e], axis=1).reshape(12, 3)
    base = d[[0, 2, 4]]
    perms = set()
    for a, b, c in np.ndindex(12, 12, 12):
        image = d[[a, b, c]]
        if not np.allclose(image @ image.T, base @ base.T, atol=1e-9):
            continue
        m = np.linalg.solve(base, image).T  # m @ base[k] = image[k]
        perms.add(tuple(int(i) for i in np.argmax((d @ m.T) @ d.T, axis=1)))
    return np.array(sorted(perms))


def _star_classes(tiling) -> dict[tuple, int]:
    """The stars of a periodic icosahedral tiling up to the symmetries of the icosahedron:
    for each vertex, the set of the corners of the cells that meet at it (three signed
    directions each), the smallest of its 120 images; the number of vertices of each."""
    edges, _ = mg.automata.tilings._icosahedral_corners()
    number = {tuple(sorted(row)): i for i, row in enumerate(edges.tolist())}
    action = np.array(
        [[number[tuple(sorted(g[row]))] for row in edges] for g in _icosahedral_group()]
    )
    s = tiling._solid
    ids = s["vertex_ids"].ravel()
    corner = (8 * s["triple"].astype(np.int64)[:, None] + np.arange(8)).ravel()
    order = np.lexsort((corner, ids))
    stars = np.split(corner[order], np.flatnonzero(np.diff(ids[order])) + 1)
    smallest: dict[tuple, tuple] = {}
    out: dict[tuple, int] = {}
    for star in stars:
        raw = tuple(star.tolist())
        if raw not in smallest:
            smallest[raw] = min(tuple(sorted(row)) for row in action[:, star].tolist())
        out[smallest[raw]] = out.get(smallest[raw], 0) + 1
    return out


def test_icosahedral_vertex_stars():
    """The 24 vertex stars of the icosahedral tiling (its local environments: Henley
    1986): their frequencies as expressions of the golden mean against the identities of the
    tiling and against the shares of its acceptance window (cut and projection, a
    construction independent of the dual grids); the stars of a patch and of the cubic
    approximants 5/3 and 8/5; the kinds against the stars up to the 120 symmetries of
    the icosahedron."""
    stars = ICOSAHEDRAL_STARS
    tau = GOLDEN
    assert len(stars) == 24
    assert {len(valences) for _, valences in stars} == {4, 5, 6, 7, 8, 9, 10, 12}
    for corners, valences in stars:  # the cells around a vertex triangulate a sphere
        assert sum(corners) == 2 * len(valences) - 4 and sum(valences) == 3 * sum(corners)
        assert min(valences) >= 3 and valences == tuple(sorted(valences))
    mean = lambda value: sum(f * value(*kind) for kind, f in stars.items())  # noqa: E731
    assert mean(lambda corners, valences: 1.0) == pytest.approx(1.0, abs=1e-14)
    assert mean(lambda corners, valences: len(valences)) == pytest.approx(6.0, abs=1e-13)
    # as many cells as vertices, prolate and oblate in the ratio τ; two tips for six
    # other corners in each
    per_vertex = np.array([mean(lambda corners, valences, k=k: corners[k]) for k in range(4)])
    assert per_vertex == pytest.approx([2 / tau, 6 / tau, 2 / tau**2, 6 / tau**2], abs=1e-13)
    twelve = sum(f for (_, valences), f in stars.items() if len(valences) == 12)
    assert twelve == pytest.approx(tau**-6, abs=1e-15)  # the twelve edges: W shrunk by τ²
    assert stars[(20, 0, 0, 0), (5,) * 12] == pytest.approx(tau**-9, abs=1e-15)

    # the shares of the acceptance window
    samples = 1 << 18
    window = window_stars(samples, np.random.default_rng(0))
    assert set(window) == set(stars)
    for kind, f in stars.items():
        assert abs(window[kind] - f) < 5.0 * math.sqrt(f * (1 - f) / (0.55 * samples))

    # a patch: these stars and no other
    patch = make_tiling(
        "icosahedral", (24, 24, 24), dims=3, periodic=False, rng=np.random.default_rng(3)
    )
    found = patch.vertex_stars()
    assert set(found) == set(stars)
    total = sum(found.values())
    assert 0.8 * patch.n_cells < total < patch.n_cells
    for kind, f in stars.items():
        if f > 0.05:  # (the stars of many cells are more often cut by the faces of the box)
            assert found[kind] / total == pytest.approx(f, rel=0.1)

    # the cubic approximants: every vertex has its star, six edges in the mean; the 24
    # stars at frequencies that converge, and a few vertices of other stars (the phason
    # strain)
    for order, cells, tol, others in ((4, 2440, 0.004, 0.006), (5, 10336, 0.0015, 0.003)):
        period = approximant_period("icosahedral", order)
        t = make_tiling(
            "icosahedral", period, dims=3, approximant=order, rng=np.random.default_rng(1)
        )
        found = t.vertex_stars()
        total = sum(found.values())
        assert t.n_cells == cells and total == cells  # V = C on the torus
        assert sum(len(valences) * n for (_, valences), n in found.items()) == 6 * cells
        assert sum(sum(corners) * n for (corners, _), n in found.items()) == 8 * cells
        for kind, f in stars.items():
            assert found.get(kind, 0) / total == pytest.approx(f, abs=tol)
        assert 0 < sum(n for kind, n in found.items() if kind not in stars) < others * cells
    # the kinds tell apart the stars up to the symmetries of the icosahedron; the
    # directions of their edges alone do not
    classes = _star_classes(t)
    assert len(_icosahedral_group()) == 120
    assert sorted(classes.values()) == sorted(found.values()) and len(classes) == len(found)
    edges, species = mg.automata.tilings._icosahedral_corners()
    directions = set()
    for star in classes:
        corners = tuple(np.bincount(species[list(star)], minlength=4).tolist())
        valences = np.bincount(edges[list(star)].ravel(), minlength=12)
        kind = (corners, tuple(sorted(valences[valences > 0].tolist())))
        assert found[kind] == classes[star]
        if kind in stars:
            masks = {
                sum(1 << int(g[e]) for e in np.flatnonzero(valences)) for g in _icosahedral_group()
            }
            directions.add(min(masks))
    assert len(directions) == 9
    with pytest.raises(ValueError, match="vertex stars"):
        make_tiling("bcc", (4, 4, 4), dims=3).vertex_stars()


# -- rules -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("neighbourhood", "radius", "dims", "rule"),
    [
        ("moore", 1, 3, "anneal"),
        ("von_neumann", 1, 2, "majority"),
        ("edges", 1, 3, "B5-7/S6-26"),
        ("moore", 1, 2, "life"),
        ("von_neumann", 2, 2, "anneal"),
        ("moore", 2, 2, "B9-14/S8-20"),
    ],
)
def test_square_tiling_reproduces_the_grid(rng, neighbourhood, radius, dims, rule):
    """The CSR engine on the square tiling: the grid automaton, to the bit."""
    shape = (9, 10, 7) if dims == 3 else (14, 12, 1)
    t = make_tiling("square", shape, 1.0, dims=dims)
    state = (rng.random(shape) < 0.5).astype(np.uint8)
    grid = CellularAutomaton(rule, neighbourhood=neighbourhood, radius=radius, dims=dims)
    tiled = CellularAutomaton(rule, neighbourhood=neighbourhood, radius=radius, tiling=t)
    a = grid.run(state, 6, np.random.default_rng(0))
    b = tiled.run(state.transpose(2, 0, 1).ravel(), 6, np.random.default_rng(0))
    assert np.array_equal(a.state.transpose(2, 0, 1).ravel(), b.state)
    assert np.allclose(a.history["interface"], b.history["interface"])


@pytest.mark.parametrize(
    ("name", "neighbourhood", "steps", "count", "corners"),
    [
        ("hexagonal", "von_neumann", 6, 127, 6),  # 3 t (t + 1) + 1 hexagons
        ("triangular", "moore", 5, 181, 6),  # hexagon of sides t + 1 and t
        ("square", "moore", 6, 169, 4),
    ],
)
def test_growth_from_one_seed(name, neighbourhood, steps, count, corners):
    """Any neighbour gives birth: the ball of the lattice, its characteristic shape."""
    t = make_tiling(name, (40, 40, 1), 1.0)
    state = np.zeros(t.n_cells, dtype=np.uint8)
    state[np.argmin(np.linalg.norm(t.centres - [20, 20, 0.5], axis=1))] = 1
    ca = CellularAutomaton("B1-12/S0-12", neighbourhood=neighbourhood, tiling=t)
    grown = np.flatnonzero(ca.run(state, steps, np.random.default_rng(0)).state)
    assert len(grown) == count
    points = t.centres[grown, :2] if name == "hexagonal" else np.concatenate(
        [t.cell_vertices(i) for i in grown]
    )  # fmt: skip
    hull = ConvexHull(points)
    assert len(hull.vertices) == corners
    if name != "hexagonal":  # the cells fill their convex hull
        assert hull.volume == pytest.approx(t.volumes[grown].sum(), rel=1e-9)
    else:  # a regular hexagon of circumradius t times the centre spacing
        a = math.sqrt(2 / math.sqrt(3))
        area = 1.5 * math.sqrt(3) * (steps * a) ** 2 * t.volumes[0]  # (scaled to the box)
        assert hull.volume == pytest.approx(area, rel=1e-9)
    micro = TilingField(Domain(size=(40, 40, 1)), t, ca.run(state, steps, None).state)
    image = micro.voxelize(grid=Grid(micro.domain, (160, 160, 1)))
    assert image.data.sum() / 16 == pytest.approx(t.volumes[grown].sum(), rel=0.03)


def test_growth_on_cubic_lattices():
    """Cuboctahedra on the FCC lattice, rhombic dodecahedra on the BCC lattice."""
    for name, count, corners in (("fcc", 147, 12), ("bcc", 175, 14)):
        t = make_tiling(name, (16, 16, 16), 1.0, dims=3)
        state = np.zeros(t.n_cells, dtype=np.uint8)
        state[np.argmin(np.linalg.norm(t.centres - 8, axis=1))] = 1
        ca = CellularAutomaton("B1-14/S0-14", neighbourhood="von_neumann", tiling=t)
        grown = ca.run(state, 3, None).state == 1
        assert grown.sum() == count  # (10t^3 + 15t^2 + 11t + 3) / 3, (2t + 1)(2t^2 + 2t + 1)
        assert len(ConvexHull(t.centres[grown]).vertices) == corners


def test_rules_on_irregular_neighbourhoods(rng):
    # triangles across their edges: 3 neighbours, the cell votes twice (ties keep it)
    t = make_tiling("triangular", (12, 12, 1), 1.0)
    ctx = TilingContext(t, "von_neumann")
    assert ctx.n_neighbours == 3
    state = (rng.random(t.n_cells) < 0.5).astype(np.uint8)
    new = CellularAutomaton("majority", neighbourhood="von_neumann", tiling=t).step(state, rng)
    expected = (ctx.counts(state) + 2 * state > 2).astype(np.uint8)
    assert np.array_equal(new, expected)
    # Penrose rhombi across their vertices: 7 to 11 neighbours, an array of counts
    p = make_tiling("penrose", (24, 24, 1), 1.0, periodic=False, rng=rng)
    ca = CellularAutomaton("conserved", fraction=0.3, tiling=p)
    assert isinstance(ca.lattice.n_neighbours, np.ndarray)
    s = (rng.random(p.n_cells) < 0.5).astype(np.uint8)
    evo = ca.run(s, 5, rng)
    assert np.all(np.array(evo.history["fraction"][1:]) == round(0.3 * p.n_cells) / p.n_cells)
    for rule in ("anneal", "life"):
        CellularAutomaton(rule, tiling=p).run(s, 3, rng)
    # sequential updates: colour classes of non-neighbours, any radius
    h = make_tiling("hexagonal", (12, 12, 6), 1.0, dims=3)
    ctx = TilingContext(h, "moore", 2)
    groups = ctx.sublattices()
    indptr, indices = ctx.indptr, ctx.indices
    rows = np.repeat(np.arange(h.n_cells), np.diff(indptr))
    colour = np.argmax(np.array(groups), axis=0)
    assert np.all(colour[rows] != colour[indices]) and sum(g.sum() for g in groups) == h.n_cells
    seq = CellularAutomaton("anneal", radius=2, tiling=h)
    seq.run((rng.random(h.n_cells) < 0.5).astype(np.uint8), 3, rng, update="sequential")


def test_activator_inhibitor_on_tilings(rng):
    from morphogen.automata import ActivatorInhibitorRule

    p = make_tiling("penrose", (30, 30, 1), 1.0, periodic=False, rng=rng)
    rule = ActivatorInhibitorRule(1.5, 4.0, dims=2)
    ctx = TilingContext(p)
    w = rule.tiling_kernels(ctx)[2]
    assert isinstance(w, np.ndarray)  # discs differ from cell to cell: balanced weights
    assert np.allclose(rule.score(np.ones(p.n_cells, np.uint8), ctx), 0.0)
    h = make_tiling("hexagonal", (20, 20, 1), 1.0)
    w = rule.tiling_kernels(TilingContext(h))[2]
    assert isinstance(w, float) and np.diff(rule.tiling_kernels(TilingContext(h))[0][0])[0] == 7


# -- models ----------------------------------------------------------------------------------

PLANE = Domain(size=(20, 20, 1))
WINDOW = Domain(size=(20, 20, 1), periodic=False)
CUBE = Domain.cube(8.0)
CUBE_WINDOW = Domain.cube(8.0, periodic=False)


@pytest.mark.parametrize(
    ("params", "domain"),
    [
        (
            {"lattice": "triangular", "dimension": 2, "rule": "conserved", "volume_fraction": 0.4},
            PLANE,
        ),
        (
            {"lattice": "hexagonal", "dimension": 2, "rule": "majority", "update": "sequential"},
            WINDOW,
        ),
        ({"lattice": "penrose", "dimension": 2, "steps": 8}, WINDOW),
        (
            {"lattice": "penrose", "dimension": 2, "rule": "conserved", "volume_fraction": 0.4},
            WINDOW,
        ),
        ({"lattice": "hexagonal", "rule": "conserved", "volume_fraction": 0.4, "steps": 6}, CUBE),
        ({"lattice": "bcc", "rule": "majority", "steps": 10}, CUBE),
        ({"lattice": "fcc", "rule": "majority", "update": "asynchronous"}, CUBE),
        ({"lattice": "ammann_beenker", "dimension": 2, "steps": 8}, WINDOW),
        ({"lattice": "ammann_beenker", "rule": "majority", "steps": 3}, CUBE_WINDOW),
        ({"lattice": "icosahedral", "rule": "majority", "steps": 3}, CUBE_WINDOW),
        (
            {"lattice": "icosahedral", "rule": "conserved", "volume_fraction": 0.4, "steps": 3},
            CUBE_WINDOW,
        ),
    ],
)
def test_tiling_models(params, domain):
    model = CellularAutomatonModel(**params)
    micro = model.generate(domain, seed=5)
    assert isinstance(micro, TilingField) and model.has_level_set
    image = micro.voxelize()
    assert image.shape == micro.default_grid.shape and image.data.dtype == np.uint8
    assert 0.02 < image.data.mean() < 0.98
    # the implicit function of the cells of phase 1: the side of the voxel centres, but
    # for those on a face between two cells
    level = micro.level_set().data
    assert np.all(((level < 0) == (image.data == 1)) | (np.abs(level) < 1e-9))
    assert np.array_equal(Microstructure.from_dict(micro.to_dict()).voxelize().data, image.data)
    pts = micro.default_grid.center_points()[::7]
    assert np.array_equal(micro.contains(pts), image.data.ravel()[::7] == 1)
    fine = micro.voxelize(shape=tuple(2 * n if n > 1 else 1 for n in image.shape))
    partial = micro.voxelize(supersampling=2)
    assert partial.data.mean() == pytest.approx(fine.data.mean(), abs=1e-6)
    frames = model.evolution(domain, seed=5, frames=3)
    assert np.array_equal(frames[-1], image.data)
    if not domain.periodic:  # a window of a larger tiling: the cells meeting the domain
        assert micro.cells is not None and len(micro.cells) < micro.tiling.n_cells
    if model.rule == "conserved":
        assert micro.voxelize().data.mean() == pytest.approx(0.4, abs=0.15)


def test_tiling_model_errors_and_activator_inhibitor():
    with pytest.raises(ValueError, match="lattice"):
        CellularAutomatonModel(lattice="kagome")
    with pytest.raises(ValueError, match="three-dimensional"):
        CellularAutomatonModel(lattice="fcc", dimension=2)
    with pytest.raises(ValueError, match="smoothing"):
        CellularAutomatonModel(lattice="hexagonal", smoothing=1.0)
    with pytest.raises(ValueError, match="quasi-periodic"):
        CellularAutomatonModel(lattice="penrose", dimension=2).generate(PLANE, seed=1)
    with pytest.raises(ValueError, match="no cell centre"):
        CellularAutomatonModel(
            lattice="hexagonal", dimension=2, neighbourhood="euclidean", radius=1
        ).generate(PLANE, seed=1)
    assert set(LATTICES) == {
        "square", "triangular", "hexagonal", "penrose", "ammann_beenker", "bcc", "fcc",
        "icosahedral",
    }  # fmt: skip
    with pytest.raises(ValueError, match="three-dimensional"):
        CellularAutomatonModel(lattice="icosahedral", dimension=2)
    with pytest.raises(ValueError, match="quasi-periodic"):
        CellularAutomatonModel(lattice="icosahedral").generate(CUBE, seed=1)
    model = ActivatorInhibitorModel(
        lattice="hexagonal", dimension=2, activation_radius=1.5, inhibition_radius=4.0, steps=6
    )
    micro = model.generate(Domain(size=(30, 30, 1)), seed=2)
    assert isinstance(micro, TilingField) and 0.2 < micro.voxelize().data.mean() < 0.8
    assert micro.attributes["inhibition_weight"] == pytest.approx(7 / 48)  # 7 cells, 48 around


def test_inverse_design_on_tilings():
    from morphogen.design import fit_rule

    model = CellularAutomatonModel(lattice="triangular", dimension=2, rule="majority", steps=6)
    target = model.generate(Domain(size=(16, 16, 1)), seed=1).voxelize(grid=None, shape=(16, 16, 1))
    fit = fit_rule(
        target.data, lattice="triangular", dims=2, shape=(16, 16, 1), steps=4,
        population=6, generations=1, rng=np.random.default_rng(0),
    )  # fmt: skip
    assert np.isfinite(fit.cost)
    rule = LifeRule.parse(fit.rule)  # counts of the 12 vertex neighbours of a triangle
    assert max(rule.birth | rule.survival, default=0) <= 12


SQUARE_DIGESTS = [
    ({}, Domain.cube(12.0), "9db7119ac1ff7c6c"),
    (
        {"rule": "conserved", "volume_fraction": 0.3, "steps": 8},
        Domain.cube(12.0),
        "3705b6d8e13d7989",
    ),
    ({"rule": "life", "dimension": 2, "steps": 12}, Domain(size=(40, 30, 1)), "5a32e79642051f42"),
    (
        {"rule": "majority", "neighbourhood": "von_neumann", "update": "asynchronous"},
        Domain.cube(12.0, periodic=False),
        "2c058976faab9724",
    ),
    (
        {"rule": "anneal", "update": "sequential", "dimension": 2},
        Domain(size=(32, 32, 1)),
        "f5400cefaf410c21",
    ),
    (
        {"rule": "B5-7/S6-26", "initial_fraction": 0.45, "neighbourhood": "edges"},
        Domain.cube(10.0),
        "1cf5bc9b351303bf",
    ),
]


@pytest.mark.parametrize(("params", "domain", "digest"), SQUARE_DIGESTS)
def test_square_lattice_unchanged(params, domain, digest):
    """Square grids give the realizations of morphoGen before the other lattices."""
    micro = CellularAutomatonModel(**params).generate(domain, seed=7)
    assert hashlib.sha256(micro.phases.tobytes()).hexdigest()[:16] == digest


def test_square_activator_inhibitor_unchanged():
    model = ActivatorInhibitorModel(activation_radius=1.5, inhibition_radius=3.5, steps=6)
    micro = model.generate(mg.Domain.cube(12.0), seed=3)
    assert hashlib.sha256(micro.phases.tobytes()).hexdigest()[:16] == "07ef460426c83909"


# -- reaction-diffusion ----------------------------------------------------------------------

A_HEX = math.sqrt(2 / math.sqrt(3))  # centre spacing of hexagons of unit area
B_TRI = 2 / 3**0.25  # edge of triangles of unit area


@pytest.mark.parametrize(
    ("name", "dims", "size"),
    [
        ("square", 2, (24, 20, 1)),
        ("hexagonal", 2, (24 * A_HEX, 12 * math.sqrt(3) * A_HEX, 1)),
        ("triangular", 2, (24 * B_TRI, 12 * math.sqrt(3) * B_TRI, 1)),
        ("triangular", 3, (24 * B_TRI, 4 * math.sqrt(3) * B_TRI, 4)),
        ("bcc", 3, (12 * 2 ** (1 / 3), 4 * 2 ** (1 / 3), 4 * 2 ** (1 / 3))),
        ("fcc", 3, (12 * 4 ** (1 / 3), 3 * 4 ** (1 / 3), 3 * 4 ** (1 / 3))),
    ],
)
def test_finite_volume_laplacian(name, dims, size):
    """Faces of Voronoi tilings fitted to their box: sum_j A_ij |d_ij| / 2 = d V_i (the
    pyramids on the faces fill the cell), and the Laplacian of a plane wave cos(k x) is
    -k^2 to second order in k."""
    t = make_tiling(name, size, 1.0, dims=dims)
    indptr, indices, measure = t.face_measures()
    w = t.diffusion_weights()[2]
    rows = np.repeat(np.arange(t.n_cells), np.diff(indptr))
    d = t.centres[indices] - t.centres[rows]
    d -= t.size * np.round(d / t.size)
    r = np.linalg.norm(d, axis=1)
    pyramids = np.bincount(rows, weights=measure * r / 2, minlength=t.n_cells)
    assert np.allclose(pyramids, dims * t.volumes, rtol=1e-12)
    assert np.allclose(w, measure / r)  # unit cells: A / d
    k = 2 * np.pi / size[0]
    u = np.cos(k * t.centres[:, 0])
    lap = np.bincount(rows, weights=w * (u[indices] - u[rows]), minlength=t.n_cells) / t.volumes
    rayleigh = np.sum(lap * u * t.volumes) / np.sum(u * u * t.volumes)
    assert rayleigh == pytest.approx(-(k**2), rel=0.01)  # k = 0.26 per cell size


def _homogenized_diffusivity(tiling, axis: int) -> float:
    """Diffusivity of the finite-volume graph at large scales: the discrete Laplace
    equation with u = x on the cells near the edge of the patch, the energy of the inner
    faces over the inner volume (|grad u| = 1)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    indptr, indices, w = tiling.diffusion_weights()
    n, k = tiling.n_cells, 2 if tiling.dims == 2 else 3
    c = tiling.size[:k] / 2
    x = tiling.centres[:, :k] - c
    a = sparse.csr_matrix((w, indices, indptr), shape=(n, n))
    a = sparse.diags(np.asarray(a.sum(axis=1)).ravel()) - a
    free = np.all(np.abs(x) < c - 4, axis=1)
    u = x[:, axis].copy()
    u[free] = spsolve(a[free][:, free].tocsc(), -a[free][:, ~free] @ u[~free])
    inner = np.all(np.abs(x) < c - 10, axis=1).astype(float)
    rows = np.repeat(np.arange(n), np.diff(indptr))
    # twice the energy: the pairs counted twice, the faces on the edge of the region half
    energy = 0.25 * np.sum((inner[rows] + inner[indices]) * w * (u[indices] - u[rows]) ** 2)
    return energy / np.sum(inner * tiling.volumes)


def test_finite_volumes_on_quasiperiodic_tilings():
    """Two-point fluxes across the edges of rhombi, whose centres lie off the normals: not
    exact for linear fields, but the graph diffuses like the plane at large scales."""
    for name in ("penrose", "ammann_beenker"):
        t = make_tiling(name, (80, 80, 1), 1.0, periodic=False, rng=np.random.default_rng(0))
        assert np.allclose(t.face_measures()[2], mg.automata.tilings.tile_edge(name))
        for axis in (0, 1):
            assert _homogenized_diffusivity(t, axis) == pytest.approx(1.0, abs=0.02)
        assert stable_substeps(t, 0.16, 0.08) == (2 if name == "penrose" else 1)
    ico = make_tiling(
        "icosahedral", (12, 12, 12), 1.0, dims=3, periodic=False, rng=np.random.default_rng(0)
    )
    a = mg.automata.tilings.tile_edge("icosahedral")
    assert np.allclose(ico.face_measures()[2], 2 / math.sqrt(5) * a**2)  # golden rhombi
    assert stable_substeps(ico, 0.16, 0.08) == 2


def test_gray_scott_on_the_square_tiling():
    """The finite-volume scheme on the square tiling is the grid's 5- or 7-point scheme,
    to round-off over thousands of steps."""
    for dims, shape in ((2, (48, 40, 1)), (3, (24, 26, 20))):
        t = make_tiling("square", shape, 1.0, dims=dims)
        assert stable_substeps(t, 0.16, 0.08) == 1
        u0, v0 = gray_scott(shape, np.random.default_rng(3), steps=0)

        def cells(field):  # the tiling's order: layer by layer
            return field.transpose(2, 0, 1).ravel()

        a = gray_scott(shape, None, steps=2000, initial=(u0, v0))
        b = gray_scott_tiling(t, None, steps=2000, initial=(cells(u0), cells(v0)))
        for grid, tiled in zip(a, b, strict=True):
            assert np.abs(cells(grid) - tiled).max() < 1e-12
        assert a[1].max() > 0.2  # a pattern, not the uniform state


def _wavelength(image) -> float:
    """Centroid of the radial power spectrum about its peak, as a wavelength (pixels)."""
    image = image - image.mean()
    n = image.shape[0]
    power = np.abs(np.fft.fft2(image)) ** 2
    k = np.fft.fftfreq(n)
    ring = np.rint(np.hypot(k[:, None], k[None, :]).ravel() * n).astype(int)
    profile = np.bincount(ring, weights=power.ravel()) / np.maximum(np.bincount(ring), 1)
    profile[:2] = 0
    peak = int(np.argmax(profile))
    band = np.arange(max(2, peak // 2), int(1.5 * peak) + 1)
    return n / (np.sum(band * profile[band]) / np.sum(profile[band]))


def test_turing_wavelength_on_tilings():
    """Labyrinths of the same wavelength on squares, hexagons and triangles."""
    n, px = 64, 2
    _, v = gray_scott((n, n, 1), np.random.default_rng(0), steps=3000)
    grid_wavelength = _wavelength(np.kron(v[:, :, 0], np.ones((px, px)))) / px
    assert 11 < grid_wavelength < 13.5  # about 12 cells
    for name in ("hexagonal", "triangular"):
        t = make_tiling(name, (n, n, 1), 1.0)
        _, v = gray_scott_tiling(t, np.random.default_rng(0), steps=3000)
        image = v[t.cell_index(Grid(Domain(size=(n, n, 1)), (n * px, n * px, 1)))][:, :, 0]
        assert _wavelength(image) / px == pytest.approx(grid_wavelength, rel=0.03)


@pytest.mark.parametrize(
    ("params", "domain", "substeps"),
    [
        ({"lattice": "hexagonal", "dimension": 2, "steps": 1500}, Domain(size=(36, 36, 1)), 1),
        (
            {"lattice": "penrose", "dimension": 2, "steps": 1500, "volume_fraction": 0.3},
            Domain(size=(36, 36, 1), periodic=False),
            2,
        ),
        (
            {"lattice": "icosahedral", "steps": 300, "pattern": "spots"},
            Domain.cube(8.0, periodic=False),
            2,
        ),
    ],
)
def test_reaction_diffusion_on_tilings(params, domain, substeps):
    from morphogen.models import ReactionDiffusionModel

    model = ReactionDiffusionModel(**params)
    micro = model.generate(domain, seed=2)
    assert isinstance(micro, TilingField) and model.has_level_set
    assert micro.attributes["substeps"] == substeps
    image = micro.voxelize()
    assert 0.05 < image.data.mean() < 0.95
    if model.volume_fraction is not None:  # by volume, over the cells of the window
        kept = micro.tiling.volumes[micro.cells]
        assert kept[micro.phases == 1].sum() / kept.sum() == pytest.approx(0.3, abs=0.01)
    assert np.array_equal(Microstructure.from_dict(micro.to_dict()).voxelize().data, image.data)
    frames = model.evolution(domain, seed=2, frames=3)
    assert np.array_equal(frames[-1] > micro.attributes["threshold"], image.data == 1)
    with pytest.raises(ValueError, match="three-dimensional"):
        ReactionDiffusionModel(lattice="fcc", dimension=2)
    with pytest.raises(ValueError, match="smoothing"):
        ReactionDiffusionModel(lattice="hexagonal", smoothing=1.0)


# -- periodic approximants -------------------------------------------------------------------


def _lucas(n: int) -> int:
    a, b = 2, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _pell(n: int) -> tuple[int, int]:
    """P_n and H_n = P_n + P_{n-1}."""
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, 2 * b + a
    return b, a + b


#: Tiles per period of the approximant of order k (the larger, then the other): thick and
#: thin rhombi 2 L_{2k+1}, 2 L_{2k} (Lucas numbers), squares and rhombi H_{2k}, 2 P_{2k},
#: prolate and oblate rhombohedra 4 F_{3k+2}, 4 F_{3k+1}: in the limit, ratios τ, √2, τ.
TILES_PER_PERIOD = {
    "penrose": lambda k: (2 * _lucas(2 * k + 1), 2 * _lucas(2 * k)),
    "ammann_beenker": lambda k: (_pell(2 * k)[1], 2 * _pell(2 * k)[0]),
    "icosahedral": lambda k: (4 * _fib(3 * k + 2), 4 * _fib(3 * k + 1)),
}


def _approximant(name, order, counts, seed=0):
    """The approximant of ``order`` filling ``counts`` periods exactly (unscaled tiles)."""
    period = approximant_period(name, order)
    dims = 3 if name == "icosahedral" else 2
    size = period * counts if dims == 3 else (*(period * counts), 1.0)
    rng = np.random.default_rng(seed)
    return make_tiling(name, size, 1.0, dims=dims, approximant=order, rng=rng), period


def _pairs(indptr, indices):
    return np.repeat(np.arange(len(indptr) - 1), np.diff(indptr)), np.asarray(indices)


def _torus_euler(t) -> int:
    """V - E + F (- C): vertices, edges, faces (and cells) of a tiling, by their vertex
    numbers (0 on the torus)."""
    if t.dims == 2:
        ids = t._plane["vertex_ids"].reshape(-1, 4)
        sides = np.sort(np.stack([ids, np.roll(ids, -1, axis=1)], axis=2).reshape(-1, 2), axis=1)
        return len(np.unique(ids)) - len(np.unique(sides, axis=0)) + t.n_cells
    ids = t._solid["vertex_ids"]
    sides = np.unique(np.concatenate([np.sort(ids[:, e], axis=1) for e in EDGES]), axis=0)
    faces = np.unique(np.sort(ids[:, PARALLELEPIPED_FACES].reshape(-1, 4), axis=1), axis=0)
    return len(np.unique(ids)) - len(sides) + len(faces) - t.n_cells


@pytest.mark.parametrize(
    ("name", "order", "counts"),
    [("penrose", 4, (2, 1)), ("ammann_beenker", 3, (1, 2)), ("icosahedral", 3, (2, 1, 1))],
)
def test_approximants_are_periodic_tilings(name, order, counts):
    """The same tiles, periodic: the adjacency wraps across every face of the box, the
    tiling translated by a period is itself, and it tiles the torus."""
    from scipy import sparse

    t, period = _approximant(name, order, counts)
    k = t.dims
    edge = mg.automata.tilings.tile_edge(name)
    star = mg.automata.tilings._star(name) * edge
    larger, other = TILES_PER_PERIOD[name](order)
    n = int(np.prod(counts))
    assert np.bincount(t.kinds).tolist() == [larger * n, other * n]
    # every cell is one of the tiles: edges ±e_j, the areas or volumes of the tiles
    if k == 2:
        poly = t._plane["poly"].reshape(-1, 4, 2)
        sides = (np.roll(poly, -1, axis=1) - poly).reshape(-1, 2)
        areas = np.sin([2 * np.pi / 5, np.pi / 5]) if name == "penrose" else [1, 1 / math.sqrt(2)]
    else:
        sides = t._solid["frames"][t._solid["triple"]].reshape(-1, 3)
        areas = GOLDEN_RHOMBOHEDRA
    gap = np.abs(sides[:, None, :] - np.concatenate([star, -star])[None]).max(axis=2).min(axis=1)
    assert gap.max() < 1e-12
    assert np.allclose(t.volumes, np.asarray(areas)[t.kinds] * edge**k, rtol=1e-12)
    assert t.volumes.sum() == pytest.approx(np.prod(t.size[:k]), rel=1e-12)  # they fill it
    # every edge (face) shared by two cells, across the faces of the box too: a torus
    assert _sizes(t, "faces") == [4 if k == 2 else 6] and _torus_euler(t) == 0
    rows, cols = _pairs(*t.adjacency("faces"))
    d = t.centres[cols, :k] - t.centres[rows, :k]
    assert np.all(np.any(np.abs(d) > t.size[:k] / 2, axis=0))  # pairs wrapping on each axis
    # the tiling translated by a period along the axis of two periods is itself
    shift = np.zeros(k)
    axis = int(np.argmax(counts))
    shift[axis] = period[axis]
    tree = cKDTree(np.mod(t.centres[:, :k], t.size[:k]), boxsize=t.size[:k])
    dist, image = tree.query(np.mod(t.centres[:, :k] + shift, t.size[:k]))
    assert dist.max() < 1e-9 and len(np.unique(image)) == t.n_cells
    assert np.array_equal(t.kinds[image], t.kinds)
    if k == 2:
        assert np.allclose(poly[image], poly)  # the same rhombi, in the same orientations
    else:
        assert np.array_equal(t._solid["triple"][image], t._solid["triple"])
    # no gap or overlap: random points in exactly one cell (the nearest images of the cells)
    points = np.random.default_rng(1).uniform(0, 1, (3000, k)) * t.size[:k]
    near = tree.query_ball_point(points, t.circumradius * (1 + 1e-9))
    found = t.locate(np.c_[points, np.full((len(points), 3 - k), 0.5)])
    for p, cand, c in zip(points, near, found, strict=True):
        cand = np.array(cand)
        u = p - t.centres[cand, :k]
        u -= t.size[:k] * np.round(u / t.size[:k])
        if k == 2:  # inside the counterclockwise polygon: left of every side
            v = poly[cand]
            w = np.roll(v, -1, axis=1) - v
            side = w[..., 0] * (u[:, None, 1] - v[..., 1]) - w[..., 1] * (u[:, None, 0] - v[..., 0])
            hit = cand[np.all(side > -1e-12, axis=1)]
        else:
            s = t._solid
            u = np.einsum("kj,kji->ki", u + t.centres[cand] - s["corner"][cand],
                          s["inverse"][s["triple"][cand]])  # fmt: skip
            hit = cand[np.all((u > -1e-12) & (u < 1 + 1e-12), axis=1)]
        assert len(hit) == 1 and hit[0] == c
    # rasterized, the cells wrap without a seam: the voxels facing each other across the
    # faces of the box lie in the same or in neighbouring cells
    shape = tuple(round(4 * s) if a < k else 1 for a, s in enumerate(t.size))
    idx = t.cell_index(Grid(Domain(size=t.size), shape))
    indptr, indices = t.adjacency("vertices")
    a = sparse.csr_matrix((np.ones(len(indices), bool), indices, indptr), shape=(t.n_cells,) * 2)
    for ax in range(k):
        lo, hi = np.take(idx, 0, axis=ax).ravel(), np.take(idx, -1, axis=ax).ravel()
        assert np.all((lo == hi) | np.asarray(a[lo, hi]).ravel())


def test_approximant_periods_and_boxes():
    edge = mg.automata.tilings.tile_edge
    tau, lam = GOLDEN, SILVER
    for k in (1, 2, 5):  # the periods, multiples of τ^k and λ^k
        assert np.allclose(
            approximant_period("penrose", k) / edge("penrose"),
            [math.sqrt(5) * tau**k, 2 * math.sin(2 * math.pi / 5) * tau**k],
        )
        assert np.allclose(approximant_period("ammann_beenker", k) / edge("ammann_beenker"), lam**k)
        cube = 2 * tau ** (k + 1) / math.sqrt(2 + tau)  # 1/1: 2.753, 2/1: 4.454, 3/2: 7.207
        assert np.allclose(approximant_period("icosahedral", k) / edge("icosahedral"), cube)
    # orders per axis: other rectangles (and cuboids)
    assert np.allclose(
        approximant_period("penrose", (4, 3)),
        [approximant_period("penrose", 4)[0], approximant_period("penrose", 3)[1]],
    )
    rng = np.random.default_rng(0)
    t = make_tiling("ammann_beenker", (15.5, 37.3, 1), approximant=(3, 4), rng=rng)
    assert _sizes(t, "faces") == [4] and _torus_euler(t) == 0
    # a box that is no multiple of the period: the nearest whole number of periods, fitted
    t = make_tiling("penrose", (20, 20, 1), 1.0, approximant=3, rng=rng)
    period = approximant_period("penrose", 3)
    scale = 20 / (np.rint(20 / period) * period)
    assert np.allclose(t._plane["scale"][:2], scale) and t.n_cells == 4 * 94
    assert t.volumes.sum() == pytest.approx(400.0, rel=1e-12) and _torus_euler(t) == 0
    # the shared edges measure the affine images of the e_j
    lengths = np.linalg.norm(mg.automata.tilings._star("penrose") * edge("penrose") * scale, axis=1)
    measure = t.face_measures()[2]
    assert np.abs(measure[:, None] - lengths[None]).min(axis=1).max() < 1e-12
    assert len(np.unique(np.round(measure, 9))) == 3  # e_1, e_4 and e_2, e_3: mirror images
    # the faces of rhombohedra fitted to a box: the areas of the scaled golden rhombi
    ico = make_tiling("icosahedral", (9, 8, 8), 1.0, dims=3, approximant=2, rng=rng)
    assert ico.volumes.sum() == pytest.approx(9 * 8 * 8, rel=1e-12) and _torus_euler(ico) == 0
    f = ico._solid["frames"]
    spans = np.linalg.norm(np.cross(f[:, [0, 0, 1]], f[:, [1, 2, 2]]), axis=2).ravel()
    assert np.abs(ico.face_measures()[2][:, None] - spans[None]).min(axis=1).max() < 1e-12
    # refusals
    with pytest.raises(ValueError, match="approximant=1"):
        make_tiling("penrose", (20, 20, 1))
    with pytest.raises(ValueError, match="quasi-periodic tilings only"):
        make_tiling("hexagonal", (20, 20, 1), approximant=2)
    with pytest.raises(ValueError, match="periodic tilings only"):
        make_tiling("penrose", (20, 20, 1), periodic=False, approximant=2)
    for order in (0, 1.5, (2, 3, 4), True):
        with pytest.raises(ValueError, match="an order of at least 1"):
            make_tiling("penrose", (20, 20, 1), approximant=order)
    with pytest.raises(ValueError, match="less than half its period"):
        make_tiling("ammann_beenker", (15, 15, 1), approximant=4)


def _sharing(ids, groups) -> set:
    """Pairs of cells sharing one of their sides, edges or faces (``groups``: their
    corners), by the vertex numbers."""
    from scipy import sparse

    keys = np.sort(ids[:, groups], axis=2).reshape(-1, groups.shape[1])
    number = np.unique(keys, axis=0, return_inverse=True)[1].ravel()
    owner = np.repeat(np.arange(len(ids)), len(groups))
    incidence = sparse.csr_matrix((np.ones(len(number)), (owner, number)))
    shared = (incidence @ incidence.T).tocoo()
    return {(i, j) for i, j in zip(shared.row.tolist(), shared.col.tolist(), strict=True) if i != j}


def test_degenerate_tori_are_refused():
    """Tori too small for their cells, whose neighbourhoods would be wrong (a cell meeting
    its own image, two cells meeting through several images), are refused with the
    remedy; more periods or a higher order give the neighbourhoods of the shared sides,
    edges and faces."""
    remedy = r"take a larger box \(more periods\) or a higher order"
    # order 1 with one period: rhombi sharing two vertices that are not a side (5 face
    # neighbours for 4 sides), also for orders (2, 1) of Penrose rhombi
    for name, order in (("penrose", 1), ("ammann_beenker", 1), ("penrose", (2, 1))):
        period = approximant_period(name, order)
        with pytest.raises(ValueError, match=f"several images.*{remedy}"):
            make_tiling(name, (*period, 1.0), approximant=order, rng=np.random.default_rng(0))
    # the cubic approximant 2/1 with one period: every cell has its 6 face neighbours, but
    # 46 pairs of cells share two vertices through two images (taken for an edge)
    with pytest.raises(ValueError, match=f"too small for its cells.*{remedy}"):
        _approximant("icosahedral", 2, (1, 1, 1))
    with pytest.raises(ValueError, match=remedy):  # the models refuse them too
        model = CellularAutomatonModel(lattice="ammann_beenker", dimension=2, approximant=1)
        model.generate(Domain(size=(2.65, 2.65, 1)), seed=1)
    for name, order, counts in (
        ("penrose", 1, (2, 1)),
        ("ammann_beenker", 1, (2, 2)),
        ("icosahedral", 1, (2, 2, 2)),
        ("icosahedral", 3, (1, 1, 1)),
    ):
        t, _ = _approximant(name, order, counts)
        assert _sizes(t, "faces") == [4 if t.dims == 2 else 6] and _torus_euler(t) == 0
        if t.dims == 2:
            ids, kinds = t._plane["vertex_ids"].reshape(-1, 4), {"faces": RHOMBUS_SIDES}
        else:
            ids, kinds = (
                t._solid["vertex_ids"],
                {"faces": PARALLELEPIPED_FACES, "edges": np.array(EDGES)},
            )
        for kind, groups in kinds.items():
            assert set(zip(*map(np.ndarray.tolist, _pairs(*t.adjacency(kind))), strict=True)) == (
                _sharing(ids, groups)
            )


RHOMBUS_SIDES = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])


def test_approximant_frequencies_converge():
    """Tile ratios and vertex stars of the approximants tend to those of the
    quasi-periodic tilings as the order grows."""
    tau, r5, lam = GOLDEN, math.sqrt(5), SILVER
    limits = {"penrose": tau, "ammann_beenker": math.sqrt(2), "icosahedral": tau}
    for name, orders in (("penrose", range(1, 8)), ("ammann_beenker", range(1, 6)),
                         ("icosahedral", range(1, 5))):  # fmt: skip
        errors = []
        for k in orders:  # two periods along each axis: no degenerate torus at order 1
            t, _ = _approximant(name, k, (2,) * (3 if name == "icosahedral" else 2))
            n = np.bincount(t.kinds)
            ratio = n[1] / n[0] if name == "ammann_beenker" else n[0] / n[1]
            errors.append(abs(ratio / limits[name] - 1))
        assert np.all(np.diff(np.log(errors)) < -1.5)  # divided by more than 4.5 per order
    # the vertex stars of Penrose approximants: de Bruijn's eight, and 8 defects per period
    eight = {
        (2, (3, 3, 4)): tau**-2, (2, (1, 2, 1, 3, 3)): tau**-3, (1, (2, 4, 4)): tau**-4,
        (1, (2, 2, 2, 4)): tau**-5, (2, (1, 1, 2, 1, 1, 2, 2)): tau**-6,
        (1, (2, 2, 2, 2, 2)): tau**-5 / r5, (2, (1, 1, 2, 2, 2, 2)): tau**-7,
        (2, (2, 2, 2, 2, 2)): tau**-7 / r5,
    }  # fmt: skip
    for k, tol in ((4, 0.25), (6, 0.04), (8, 0.005)):
        t, _ = _approximant("penrose", k, (1, 1))
        stars = t.vertex_stars()
        total = sum(stars.values())
        assert total == t.n_cells  # V = F on the torus (E = 2F)
        assert set(eight) <= set(stars)
        assert sum(v for key, v in stars.items() if key not in eight) == 8
        for kind, freq in eight.items():
            assert stars[kind] / total == pytest.approx(freq, rel=tol)
    # Ammann-Beenker approximants: the six stars and no other, at the window's frequencies
    six = {
        (2, 3, 3): 1 / lam, (1, 2, 3, 2): 2 / lam**2, (1, 2, 1, 2, 2): 2 / lam**3,
        (1, 1, 1, 2, 1, 2): 2 / lam**4, (1, 1, 1, 1, 1, 1, 2): 1 / lam**5,
        (1, 1, 1, 1, 1, 1, 1, 1): 1 / lam**4,
    }  # fmt: skip
    for k, tol in ((3, 0.03), (4, 1e-3), (5, 3e-5)):
        t, _ = _approximant("ammann_beenker", k, (1, 1), seed=k)
        stars = t.vertex_stars()
        total = sum(stars.values())
        assert set(stars) == set(six)
        for kind, freq in six.items():
            assert stars[kind] / total == pytest.approx(freq, rel=tol)


PENROSE_4 = 2 * approximant_period("penrose", 4)  # two periods: 34.0 x 28.9 cell sizes


@pytest.mark.parametrize(
    ("model", "params", "domain"),
    [
        (
            "cellular_automaton",
            {"lattice": "penrose", "dimension": 2, "approximant": 4, "steps": 8},
            Domain(size=(*PENROSE_4, 1)),
        ),
        (
            "cellular_automaton",
            {
                "lattice": "ammann_beenker",
                "dimension": 2,
                "approximant": 3,
                "rule": "conserved",
                "volume_fraction": 0.4,
            },
            Domain(size=(30, 30, 1)),  # two periods of 15.5 along each axis, fitted
        ),
        (
            "cellular_automaton",
            {"lattice": "penrose", "approximant": 2, "rule": "majority", "steps": 4},
            Domain.cube(12.0),  # prisms
        ),
        (
            "cellular_automaton",
            {"lattice": "icosahedral", "approximant": 2, "rule": "majority", "steps": 3},
            Domain.cube(10.0),  # 2 x 2 x 2 cubic cells 2/1
        ),
        (
            "activator_inhibitor",
            {
                "lattice": "ammann_beenker",
                "dimension": 2,
                "approximant": 3,
                "activation_radius": 1.5,
                "inhibition_radius": 4.0,
                "steps": 6,
            },
            Domain(size=(31, 31, 1)),
        ),
        (
            "cellular_automaton",
            {"lattice": "penrose", "dimension": 2, "approximant": 3, "steps": 5},
            Domain(size=(20, 20, 1), periodic=False),  # a window of a larger approximant
        ),
    ],
)
def test_models_on_approximants(model, params, domain):
    from morphogen.models.base import get_model

    m = get_model(model)(**params)
    micro = m.generate(domain, seed=3)
    assert isinstance(micro, TilingField) and micro.tiling.periodic
    assert micro.tiling.spec["approximant"] == params["approximant"]
    image = micro.voxelize()
    assert 0.05 < image.data.mean() < 0.95 and image.periodic == domain.periodic
    again = Microstructure.from_dict(micro.to_dict())  # the tiling rebuilt from its spec
    assert np.array_equal(again.voxelize().data, image.data)
    pts = micro.default_grid.center_points()[::7]
    assert np.array_equal(micro.contains(pts), image.data.ravel()[::7] == 1)
    frames = m.evolution(domain, seed=3, frames=2)
    assert np.array_equal(frames[-1], image.data)
    if domain.periodic:
        assert micro.cells is None and micro.tiling.n_cells == len(micro.phases)
    else:
        assert micro.cells is not None and len(micro.cells) < micro.tiling.n_cells


def test_approximant_model_errors():
    with pytest.raises(ValueError, match="quasi-periodic lattices only"):
        CellularAutomatonModel(lattice="hexagonal", approximant=2)
    with pytest.raises(ValueError, match="at least 1"):
        CellularAutomatonModel(lattice="penrose", dimension=2, approximant=0)
    with pytest.raises(ValueError, match="approximant"):  # the periodic domain needs one
        CellularAutomatonModel(lattice="penrose", dimension=2).generate(PLANE, seed=1)
    with pytest.raises(ValueError, match="less than half its period"):
        model = CellularAutomatonModel(lattice="penrose", dimension=2, approximant=6)
        model.generate(PLANE, seed=1)


def test_reaction_diffusion_and_fft_on_approximants():
    """Gray-Scott on the cells of a Penrose approximant in a periodic box, then the
    apparent bulk modulus of the pattern by the FFT solver; reaction-diffusion on a
    cubic approximant of the icosahedral tiling."""
    from morphogen.homogenization import Material, bounds, homogenize
    from morphogen.models import ReactionDiffusionModel

    model = ReactionDiffusionModel(
        lattice="penrose", dimension=2, approximant=4, steps=1500, volume_fraction=0.3
    )
    micro = model.generate(Domain(size=(*PENROSE_4, 1)), seed=2)
    assert micro.attributes["substeps"] == 2 and micro.cells is None
    volumes = micro.tiling.volumes
    assert volumes[micro.phases == 1].sum() / volumes.sum() == pytest.approx(0.3, abs=0.01)
    image = micro.voxelize()
    assert image.periodic and image.data.mean() == pytest.approx(0.3, abs=0.03)
    mats = {0: Material(young=1.0, poisson=0.2), 1: Material(young=20.0, poisson=0.35)}
    res = homogenize(image, mats, properties=["bulk_modulus"])["bulk_modulus.periodic"]
    b = bounds(mg.analysis.phase_fractions(image), mats)["bulk_modulus"]
    assert b["hs_lower"] < res.value < b["hs_upper"]
    assert res.energy == pytest.approx(res.value, rel=1e-6)
    assert res.solution.info.method == "fft"
    # 2 x 2 x 2 cubic cells 5/3 (27 cell sizes): room for the pattern
    cube = ReactionDiffusionModel(lattice="icosahedral", approximant=4, steps=300, pattern="spots")
    side = 2 * float(approximant_period("icosahedral", 4)[0])
    micro = cube.generate(Domain.cube(side), seed=2)
    assert micro.tiling.n_cells == 8 * 2440 and micro.attributes["substeps"] == 2
    assert 0.02 < micro.voxelize().data.mean() < 0.98
