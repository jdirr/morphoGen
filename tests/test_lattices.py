"""Rotated and graded strut lattices (non-periodic domains): the struts along the turned
axes of the cell, the radii interpolated along the gradient, the volume fractions of
their theory."""

import numpy as np
import pytest

import morphogen as mg
from morphogen.models import StrutLattice, model_from_config
from morphogen.orientations import euler_to_matrix

WINDOW = mg.Domain.cube(8.0, periodic=False)


def test_rotated_lattice():
    euler = (30.0, 40.0, 50.0)
    model = StrutLattice(cell="cubic", cell_size=1.0, strut_radius=0.12, rotation=euler)
    seg = model.generate(WINDOW, seed=0).segments
    directions = (seg.end - seg.start) / seg.lengths[:, None]
    in_cell = np.abs(directions @ euler_to_matrix(np.array(euler))[0].T)  # lattice axes
    assert np.allclose(in_cell.max(axis=1), 1.0)  # each strut along an axis of the cell
    fraction = model.generate(WINDOW, seed=0).voxelize(shape=160, supersampling=2).data.mean()
    assert fraction == pytest.approx(model.theory()["volume_fraction"], rel=0.02)
    assert model_from_config(model.to_config()) == model
    with pytest.raises(ValueError, match="non-periodic"):
        model.generate(mg.Domain.cube(8.0, periodic=True), seed=0)
    with pytest.raises(ValueError, match="three Bunge angles"):
        StrutLattice(cell="cubic", strut_radius=0.1, rotation=(10.0, 20.0))


def test_graded_lattice():
    model = StrutLattice(cell="octet", cell_size=1.0, radius_gradient=(0.04, 0.12), gradient_axis=0)
    seg = model.generate(WINDOW, seed=0).segments
    s = np.clip((0.5 * (seg.start + seg.end))[:, 0] / 8.0, 0.0, 1.0)
    assert np.allclose(seg.radii, 0.04 + 0.08 * s)
    image = model.generate(WINDOW, seed=0).voxelize(shape=160, supersampling=2)
    theory = model.theory()
    assert image.data.mean() == pytest.approx(theory["volume_fraction"], rel=0.02)
    profile = image.data.mean(axis=(1, 2))
    first, last = profile[:20].mean(), profile[-20:].mean()  # the first and last cells
    assert theory["volume_fraction_lower"] < first < theory["volume_fraction"] < last
    assert last < theory["volume_fraction_upper"]
    assert model_from_config(model.to_config()) == model
    for kwargs, match in [
        ({"radius_gradient": (0.04, 0.12), "strut_radius": 0.1}, "radius_gradient, or"),
        ({"radius_gradient": (0.04, 0.12), "wall_thickness": 0.01}, "solid struts"),
        ({"radius_gradient": (0.04, 0.6)}, "half the cell size"),
        ({"radius_gradient": (0.04, 0.12), "gradient_axis": 3}, "gradient_axis"),
    ]:
        with pytest.raises(ValueError, match=match):
            StrutLattice(cell="octet", **kwargs)
