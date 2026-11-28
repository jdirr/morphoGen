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


def _walls(micro, n: int) -> float:
    """The fraction of the box in the walls of a honeycomb, counted at the centres of the
    cells of an ``n`` x ``n`` grid by its own distance to the sides of the cells."""
    lower, lengths = micro.domain.lower[:2], micro.domain.lengths[:2]
    gx = lower[0] + (np.arange(n) + 0.5) / n * lengths[0]
    gy = lower[1] + (np.arange(n) + 0.5) / n * lengths[1]
    walls = 0
    for rows in np.array_split(np.arange(n), 8):
        x, y = np.meshgrid(gx[rows], gy, indexing="ij")
        walls += int((micro._signed(np.column_stack([x.ravel(), y.ravel()]), 1.0) < 0).sum())
    return walls / n**2


def test_lumens_of_the_cells():
    """The lumens of a square honeycomb (cells of 1, walls of 0.1): squares of 0.9, those
    the sides cut in pieces; prisms along another axis and the files of wood are not
    meshed by them."""
    _, micro = _honeycomb("square", (3.0, 3.0, 1.0))
    lumens = cell_lumens(micro)
    areas = [abs(_shoelace(p)) for _, p in lumens]
    assert sum(areas) == pytest.approx(9 * 0.81, rel=1e-12)
    whole = [a for a in areas if a == pytest.approx(0.81, rel=1e-12)]
    assert len(whole) == 4 and len(lumens) == 16  # (four whole, the others cut by the sides)
    _, other = _honeycomb("square", (3.0, 1.0, 3.0), axis=1)
    _, wood = _honeycomb("wood", (3.0, 3.0, 1.0), wall=0.05)
    assert cell_lumens(other) is None and cell_lumens(wood) is None
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
    them): their walls within 1e-5 of counts of the model's own distance on grids of 4000²
    points; the lumens' edges graded from their gaps to the sides."""
    for periodic in (True, False):
        _, micro = _honeycomb("voronoi", (4.0, 4.0, 1.0), periodic=periodic)
        mesh = cells_mesh(micro, size=0.1)
        assert mesh.metadata["mesher"] == "exact cells (mmg2d)" and mesh.periodic == periodic
        assert mesh.phase_fractions()[1] == pytest.approx(_walls(micro, 4000), abs=1e-5)
        assert mesh.min_angles().min() > 2.0


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
