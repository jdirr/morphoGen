"""morphoGen and Microgen (3MAH): the shapes of Microgen as a model of morphoGen against
Microgen's own fields, the phases of morphoGen as shapes of Microgen against morphoGen's
implicit functions, and the tetrahedral meshes of both through PyVista."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.mesh.tetra import TetMesh, tet_volumes
from morphogen.models import Microstructure, model_from_config
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Spheres

microgen = pytest.importorskip("microgen")
pv = pytest.importorskip("pyvista")
from morphogen.interop import microgen as imx  # noqa: E402

CUBE = mg.Domain.cube(1.0, periodic=True)
POINTS = np.random.default_rng(0).random((20000, 3))


def _sheet():
    return model_from_config({"type": "microgen", "shape": "Tpms", "part": "sheet",
                              "arguments": {"surface_function": "gyroid", "offset": 0.6,
                                             "cell_size": 1.0}})  # fmt: skip


def test_shapes_of_microgen_as_a_model():
    """A TPMS sheet of Microgen: the field of the structure is Microgen's (the same shape
    built by Microgen itself), its voxels its sign; saved and read back; a spinodoid takes
    the seed of the realization."""
    micro = _sheet().generate(CUBE, seed=1)
    own = microgen.Tpms(surface_function=microgen.surface_functions.gyroid, offset=0.6,
                        cell_size=1.0).as_sheet()  # fmt: skip
    x, y, z = POINTS.T
    assert np.allclose(micro.field(POINTS), own.evaluate(x, y, z))
    image = micro.voxelize(40)
    centres = image.grid().center_points()
    assert np.array_equal(image.data.ravel() == 1, own.evaluate(*centres.T) < 0)
    assert np.mean(image.data) == pytest.approx(np.mean(own.evaluate(x, y, z) < 0), abs=0.01)
    shift = np.array([1.0, -2.0, 3.0])
    assert np.allclose(micro.field(POINTS + shift), micro.field(POINTS))  # periodic
    back = Microstructure.from_dict(json.loads(json.dumps(micro.to_dict())))
    assert np.array_equal(back.voxelize(24).data, micro.voxelize(24).data)
    assert micro.lipschitz >= 1.0 and micro.summary()["shape"] == "Tpms"
    waves = {"k0": 30, "bandwidth": 5, "offset": 0.0, "resolution": 16}
    spinodoid = model_from_config({"type": "microgen", "shape": "Spinodoid", "arguments": waves})
    a, b, c = (spinodoid.generate(CUBE, s).voxelize(20).data for s in (1, 1, 2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    again = Microstructure.from_dict(json.loads(json.dumps(spinodoid.generate(CUBE, 1).to_dict())))
    assert np.array_equal(again.voxelize(20).data, a)


def test_a_sphere_of_microgen_placed_and_turned():
    """A sphere of Microgen moved to the centre of the box: its field the distance to it,
    its voxels the volume of the ball; a box turned by Bunge angles in degrees."""
    ball = {"radius": 0.3, "center": [0.5, 0.5, 0.5]}
    model = model_from_config({"type": "microgen", "shape": "Sphere", "arguments": ball})
    micro = model.generate(CUBE, 1)
    distance = np.linalg.norm(POINTS - 0.5, axis=1) - 0.3
    assert np.allclose(micro.level_set_at(POINTS), distance)
    fraction = float(np.mean(micro.voxelize(64).data))
    assert fraction == pytest.approx(4.0 / 3.0 * math.pi * 0.3**3, rel=0.01)
    turned = model_from_config({"type": "microgen", "shape": "Box", "arguments": {
        "dim": [0.6, 0.1, 0.1], "orientation": [90, 0, 0], "center": [0.5, 0.5, 0.5]}})  # fmt: skip
    box = turned.generate(CUBE, 1)
    assert box.contains([[0.5, 0.75, 0.5]])[0] and not box.contains([[0.75, 0.5, 0.5]])[0]


def test_errors_of_the_model():
    with pytest.raises(ValueError, match="part"):
        model_from_config({"type": "microgen", "shape": "Tpms", "part": "shell"})
    with pytest.raises(ValueError, match="name of a class"):
        model_from_config({"type": "microgen", "shape": "two words"})
    with pytest.raises(ValueError, match="no shape 'Teapot'"):
        model_from_config({"type": "microgen", "shape": "Teapot"}).generate(CUBE, 1)
    with pytest.raises(ValueError, match="no surface function"):
        nothing = {"surface_function": "nothing"}
        model_from_config({"type": "microgen", "shape": "Tpms", "arguments": nothing}).generate(
            CUBE, 1
        )
    with pytest.raises(ValueError, match="arguments of Sphere"):
        model_from_config({"type": "microgen", "shape": "Sphere",
                           "arguments": {"size": 1}}).generate(CUBE, 1)  # fmt: skip
    shape = microgen.Sphere(radius=0.2).translate((0.5, 0.5, 0.5))
    plain = imx.from_microgen(shape, CUBE)
    assert plain.contains([[0.5, 0.5, 0.5]])[0]
    with pytest.raises(ValueError, match="cannot be saved"):
        plain.to_dict()


def test_phases_of_morphogen_as_shapes_of_microgen():
    """The field of the shape is the implicit function of the phase (the distance to the
    spheres, periodic); the structures without point-wise functions give the grid's,
    interpolated; Microgen meshes them, and its volume mesh comes back to morphoGen with
    the volume of the phase."""
    spheres = mg.models.BooleanSpheres(radius=0.15, volume_fraction=0.2).generate(CUBE, 3)
    shape = imx.to_microgen(spheres)
    x, y, z = POINTS.T
    assert np.allclose(shape.evaluate(x, y, z), spheres.implicit_at(1, POINTS, 0.1))
    assert np.allclose(shape.evaluate(x + 1, y - 1, z), shape.evaluate(x, y, z))
    matrix = imx.to_microgen(spheres, phase=0)
    assert np.allclose(matrix.evaluate(x, y, z), -shape.evaluate(x, y, z))
    with pytest.raises(ValueError, match="phases of this structure"):
        imx.to_microgen(spheres, phase=3)
    leaves = model_from_config({"type": "transparent_dead_leaves", "radius": 0.2,
                                "volume_fraction": 0.4}).generate(CUBE, 2)  # fmt: skip
    assert leaves.implicit_at(1, POINTS[:3], 0.1) is None  # no point-wise function
    gridded = imx.to_microgen(leaves, shape=32)
    image = leaves.voxelize(32)
    centres = image.grid().center_points()
    assert np.array_equal(gridded.evaluate(*centres.T) < 0, image.data.ravel() == 1)
    ball = PrimitiveUnion(CUBE, spheres=Spheres(np.array([[0.5, 0.5, 0.5]]), np.array([0.3])))
    surface = imx.to_microgen(ball).generate_surface_mesh(bounds=(0, 1, 0, 1, 0, 1), resolution=40)
    assert surface.volume == pytest.approx(4.0 / 3.0 * math.pi * 0.027, rel=0.02)
    volume = imx.to_microgen(ball).generate_volume_mesh(bounds=(0, 1, 0, 1, 0, 1), resolution=32)
    mesh = imx.from_pyvista(volume)
    assert isinstance(mesh, TetMesh) and mesh.elements.shape[1] == 4
    assert np.abs(tet_volumes(mesh.nodes, mesh.elements)).sum() == pytest.approx(
        4.0 / 3.0 * math.pi * 0.027, rel=0.02
    )


def test_tetrahedral_meshes_through_pyvista():
    """A mesh of morphoGen and its PyVista grid: the same nodes, elements (4 or 10 nodes)
    and phases, both ways."""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    mesh = TetMesh(nodes, elements, np.array([0, 1]), nodes.min(0), nodes.max(0))
    grid = imx.to_pyvista(mesh)
    assert grid.n_cells == 2 and np.array_equal(grid.cell_data["phase"], [0, 1])
    back = imx.from_pyvista(grid)
    assert np.array_equal(back.elements, elements) and np.array_equal(back.phase, [0, 1])
    assert np.allclose(back.nodes, nodes)
    quadratic = mesh.quadratic()  # 10-node elements, in VTK's order both ways
    again = imx.from_pyvista(imx.to_pyvista(quadratic))
    assert again.elements.shape[1] == 10 and np.array_equal(again.elements, quadratic.elements)
    with pytest.raises(ValueError, match="tetrahedra"):
        imx.from_pyvista(pv.Sphere().cast_to_unstructured_grid())
