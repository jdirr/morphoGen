"""Exact triangular meshes of honeycombs in their plane (``morphogen.mesh.cells``): the
lumens of the prismatic cells as their Voronoi polygons offset inward by half their walls,
against the walls of regular honeycombs in closed form, random ones against counts, and the
identities of prismatic structures in generalized plane strain."""

from __future__ import annotations

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import Material, homogenize
from morphogen.mesh.cells import cell_lumens, cells_mesh
from morphogen.mesh.conforming import mmg_executable
from morphogen.mesh.triangles import _shoelace, plane_mesh
from morphogen.models.base import model_from_config


def _has_mmg2d() -> bool:
    try:
        mmg_executable(None, "mmg2d")
    except (FileNotFoundError, RuntimeError, ImportError):
        return False
    return True


needs_mmg2d = pytest.mark.skipif(not _has_mmg2d(), reason="mmg2d (pymmg) is not installed")


def _honeycomb(arrangement: str, box, periodic: bool = True, wall: float = 0.1, **extra):
    model = model_from_config({"type": "honeycomb", "cell_size": 1.0, "wall_thickness": wall,
                               "arrangement": arrangement, **extra})  # fmt: skip
    return model, model.generate(mg.Domain(size=box, periodic=periodic), seed=1)


def test_lumens_of_the_cells():
    """The lumens of a square honeycomb (cells of 1, walls of 0.1): squares of 0.9, those
    the sides cut in pieces; prisms along another axis and wood with rays, which vary along
    the prisms, are not meshed by them."""
    _, micro = _honeycomb("square", (3.0, 3.0, 1.0))
    lumens = cell_lumens(micro)
    areas = [abs(_shoelace(p)) for _, p in lumens]
    assert sum(areas) == pytest.approx(9 * 0.81, rel=1e-12)
    whole = [a for a in areas if a == pytest.approx(0.81, rel=1e-12)]
    assert len(whole) == 4 and len(lumens) == 16  # (four whole, the others cut by the sides)
    _, other = _honeycomb("square", (3.0, 1.0, 3.0), axis=1)
    _, rays = _honeycomb("wood", (3.0, 3.0, 10.0), ray_fraction=0.1)
    assert cell_lumens(other) is None and cell_lumens(rays) is None
    assert cell_lumens(mg.models.TPMS(surface="gyroid", volume_fraction=0.3).generate(
        mg.Domain.cube(1.0, periodic=True), seed=1)) is None  # fmt: skip


@needs_mmg2d
@pytest.mark.parametrize("arrangement", ["square", "hexagonal"])
def test_regular_honeycombs(arrangement):
    """Square and hexagonal honeycombs in periodic boxes of whole cells: their walls
    1 - (1 - t/d)^2 to round-off (the theory of the model), their sides matching; meshed so
    by the meshes of the plane."""
    box = (3.0, 3.0, 1.0) if arrangement == "square" else (3.0, 3.0 * np.sqrt(3.0), 1.0)
    model, micro = _honeycomb(arrangement, box)
    exact = model.theory(micro.domain)["volume_fraction"]
    for order in (1, 2):
        mesh = cells_mesh(micro, size=0.1, order=order)
        assert mesh.metadata["mesher"] == "exact cells (mmg2d)" and mesh.matching_sides()
        assert mesh.phase_fractions()[1] == pytest.approx(exact, abs=1e-12)
    assert plane_mesh(micro, size=0.1).metadata["mesher"] == "exact cells (mmg2d)"


@needs_mmg2d
def test_random_honeycombs():
    """Voronoi honeycombs, in a periodic box and in a box with free sides (no walls along
    them): their walls within 1e-5 of counts of the model's own distance on a turned lattice
    of 2000 points across; the lumens' edges graded from their gaps to the sides."""
    for periodic in (True, False):
        _, micro = _honeycomb("voronoi", (4.0, 4.0, 1.0), periodic=periodic)
        mesh = cells_mesh(micro, size=0.1)
        assert mesh.metadata["mesher"] == "exact cells (mmg2d)" and mesh.periodic == periodic
        assert mesh.phase_fractions()[1] == pytest.approx(_rotated_walls(micro, 2000), abs=1e-5)
        assert mesh.min_angles().min() > 2.0


def _rotated_walls(micro, n: int, angle: float = 0.4142135623) -> float:
    """The fraction of the box in the walls, counted at the points of a lattice of ``n``
    across turned by ``angle`` (on a lattice along the sides, each straight side of a cell
    errs alike at every point along it)."""
    lower, lengths = micro.domain.lower[:2], micro.domain.lengths[:2]
    c, s = np.cos(angle), np.sin(angle)
    h = float(np.sqrt(lengths.prod())) / n
    m = int(np.ceil(np.linalg.norm(lengths) / h)) + 2
    total = inside = 0
    for rows in np.array_split(np.arange(-m, m), 40):
        u, v = np.meshgrid((rows + 0.5) * h, (np.arange(-m, m) + 0.5) * h, indexing="ij")
        x = lower[0] + 0.5 * lengths[0] + c * u.ravel() - s * v.ravel()
        y = lower[1] + 0.5 * lengths[1] + s * u.ravel() + c * v.ravel()
        keep = ((x > lower[0]) & (x < lower[0] + lengths[0])
                & (y > lower[1]) & (y < lower[1] + lengths[1]))  # fmt: skip
        total += int(keep.sum())
        inside += int((micro._signed(np.column_stack([x[keep], y[keep]]), 1.0) < 0).sum())
    return inside / total


@needs_mmg2d
def test_wood_in_its_plane():
    """Softwood without rays, its radial files of rectangular cells, earlywood and
    latewood, regular or displaced, in a periodic box and with free sides: the lumens of its
    cells rectangles shrunk by half their walls, the walls within 1e-5 of counts on a
    turned lattice of 2000 points across (along the sides, the counts err by 2e-4); a
    lumen that a side cuts 4e-6 from its edge, a sliver thinner than the elements (mmg2d
    failed), left out."""
    for disorder in (0.0, 0.2):
        for periodic in (True, False):
            _, wood = _honeycomb("wood", (0.6, 0.3, 0.1), periodic=periodic, wall=0.003,
                                 ring_width=0.3, disorder=disorder, cell_size=0.03)  # fmt: skip
            mesh = cells_mesh(wood, size=0.01)
            assert mesh is not None and mesh.metadata["mesher"] == "exact cells (mmg2d)"
            assert not periodic or mesh.matching_sides()
            assert mesh.phase_fractions()[1] == pytest.approx(_rotated_walls(wood, 2000), abs=1e-5)


@needs_mmg2d
def test_honeycombs_solved_in_their_plane():
    """A hexagonal honeycomb of a solid (walls of 0.1 about lumens of cells of 1) solved in
    generalized plane strain, its lumens empty: along its prisms the uniform strain is its
    field, its axial modulus and conductivity the fraction of the walls of those of the
    solid, to round-off; across them, isotropic in its plane as its hexagonal symmetry has
    it, within 1e-3, near the beams of Gibson and Ashby, 4/sqrt(3) (t/l)^3, for walls of a
    sixth of their length (bending only: 8 % below)."""
    _, micro = _honeycomb("hexagonal", (3.0, 3.0 * np.sqrt(3.0), 1.0))
    mesh = cells_mesh(micro, size=0.05, order=2)
    phi = mesh.phase_fractions()[1]
    solid = Material(young=70.0, poisson=0.3, conductivity=200.0)
    out = homogenize(mesh, {0: Material(void=True), 1: solid},
                     properties=["stiffness", "conductivity_tensor"])  # fmt: skip
    stiffness, conduction = out["stiffness.periodic"], out["conductivity_tensor.periodic"]
    assert stiffness["E3"] == pytest.approx(phi * 70.0, rel=1e-12)
    assert conduction["tensor"][2, 2] == pytest.approx(phi * 200.0, rel=1e-12)
    assert stiffness["E1"] == pytest.approx(stiffness["E2"], rel=1e-3)
    beams = 4.0 / np.sqrt(3.0) * (0.1 * np.sqrt(3.0)) ** 3  # (t / l, l the side of the hexagons)
    assert 1.0 < stiffness["E1"] / 70.0 / beams < 1.15
