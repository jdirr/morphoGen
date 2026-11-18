"""Thermal oxidation in the voxels of an image: one voxel against the homogeneous
oxidation of a thin film, a plate against its one-dimensional profile (second order in
the voxels), a block and a composite of impermeable fibres along the diffusion against
the plate (exact), barriers and inert skins, units, the tracer of the ageing laws."""

import dataclasses

import numpy as np
import pytest

from morphogen.coupling import (
    ClosedLoopOxidation,
    OxidationField,
    PropertyLaw,
    graded_materials,
    oxidation_field,
)
from morphogen.coupling.oxidation import SPECIES
from morphogen.homogenization import Material
from morphogen.image import VoxelImage

HOUR = 3600.0
SCHEME = ClosedLoopOxidation()


def plate(n, thickness=3e-3, shape=(1, 1)):
    return VoxelImage(np.zeros((n, *shape), np.uint8), thickness / n)


def test_one_voxel_against_the_thin_film():
    """A voxel so small that diffusion keeps its oxygen at equilibrium: the homogeneous
    oxidation of a thin film, for every species over 400 hours."""
    times = np.array([50.0, 111.0, 240.0, 400.0]) * HOUR
    found = oxidation_field(VoxelImage(np.zeros((1, 1, 1), np.uint8), 1e-9), SCHEME, times)
    assert isinstance(found, OxidationField) and set(found.fields) == set(SPECIES)
    film = SCHEME.thin_film(times)
    for name in SPECIES:
        assert found.fields[name][:, 0, 0, 0] == pytest.approx(film[name], rel=1e-3)
    assert found.fields["O2"][-1, 0, 0, 0] == pytest.approx(SCHEME.surface_oxygen, rel=1e-8)
    assert found.steps < 150  # from the milliseconds of the radicals to hundreds of hours
    # a tighter tolerance of the steps: nearer the film
    tight = oxidation_field(
        VoxelImage(np.zeros((1, 1, 1), np.uint8), 1e-9), SCHEME, times, rtol=1e-6
    )
    loose = np.abs(found.fields["PO"][:, 0, 0, 0] / film["PO"] - 1.0).max()
    assert np.abs(tight.fields["PO"][:, 0, 0, 0] / film["PO"] - 1.0).max() < 0.2 * loose


def test_plate_against_its_profile():
    """A plate of 3 mm exposed on both faces: the profile of the oxidation products of
    the method of lines (241 nodes through its half), to the second order in the voxels;
    the oxidized layer, a hundred micrometres thick."""
    times = np.array([111.0, 400.0, 800.0]) * HOUR
    reference = SCHEME.profile(3e-3, times, nodes=241)
    errors, layers = {}, {}
    for n in (120, 240):
        found = oxidation_field(plate(n), SCHEME, times)
        amide = found.fields["PO"][:, :, 0, 0]
        assert amide == pytest.approx(amide[:, ::-1], rel=1e-12)  # both faces alike
        depth = (np.arange(n // 2) + 0.5) * 3e-3 / n
        worst = 0.0
        for k in range(len(times)):
            exact = np.interp(depth, reference["depth"], reference["PO"][k])
            worst = max(worst, np.abs(amide[k, : n // 2] - exact).max() / exact.max())
        errors[n] = worst
        layers[n] = SCHEME.oxidized_layer(depth, amide[-1, : n // 2])
    assert errors[120] < 0.05 and errors[240] < 0.015
    assert errors[240] < 0.4 * errors[120]
    layer = SCHEME.oxidized_layer(reference["depth"], reference["PO"][-1])
    assert layers[240] == pytest.approx(layer, rel=0.01) and 50e-6 < layer < 300e-6
    assert found.fields["PO"][-1, n // 2, 0, 0] < 0.05 * found.fields["PO"][-1, 0, 0, 0]


def test_block_and_fibres_along_the_diffusion():
    """A block uniform across the diffusion is the plate, in each of its voxels; with
    impermeable fibres along the diffusion the matrix oxidizes between them as the
    plate (they take no oxygen and bar none of its way)."""
    times = np.array([111.0, 400.0]) * HOUR
    n = 16
    slab = oxidation_field(plate(n, 0.6e-3), SCHEME, times, exposed=("x-",))
    block = oxidation_field(plate(n, 0.6e-3, (3, 2)), SCHEME, times, exposed=("x-",))
    for name in SPECIES:
        assert block.fields[name] == pytest.approx(
            np.broadcast_to(slab.fields[name], block.fields[name].shape), rel=1e-9, abs=1e-18
        )
    x = (np.arange(n) + 0.5) / n
    fibre = (x[:, None] - 0.5) ** 2 + (x[None, :] - 0.5) ** 2 < 0.3**2
    data = np.broadcast_to(fibre[None], (n, n, n)).astype(np.uint8)
    image = VoxelImage(data, 0.6e-3 / n, periodic=True)
    found = oxidation_field(image, {0: SCHEME, 1: None}, times, exposed=("x-",))
    amide = found.fields["PO"]
    assert np.all(amide[:, :, fibre] == 0.0) and np.all(found.fields["O2"][:, :, fibre] == 0.0)
    between = amide[:, :, ~fibre]
    assert between == pytest.approx(
        np.broadcast_to(slab.fields["PO"][:, :, 0, 0][:, :, None], between.shape), rel=1e-4
    )
    assert found.tracer().shape == (n, n, n) and found.tracer("PH", 0).max() < SCHEME.ph0
    # the tracer of an ageing law: the matrix graded by its oxidation products
    law = PropertyLaw(tracer=(0.0, 1.0), values={"young": (2.4e3, 3.2e3)})
    glass = Material(young=70e3, poisson=0.22)
    graded, materials = graded_materials(
        image, found.tracer(), {0: law}, {1: glass}, levels=4, poisson=0.37
    )
    assert graded.shape == image.shape and 3 <= len(materials) <= 5
    stiffest = max(m.young for k, m in materials.items() if k != 1)
    assert 2.4e3 < stiffest <= 3.2e3 and materials[1] is glass


def test_barriers_and_inert_skins():
    """Across the diffusion: behind an impermeable layer the polymer has the oxygen it
    was stored with and no more; an inert skin that lets oxygen through holds no
    oxidation product, and the oxygen that it does not consume reaches the polymer
    behind it."""
    times = np.array([400.0]) * HOUR
    n = 24
    labels = np.zeros((n, 1, 1), np.uint8)
    labels[1:3] = 1  # a layer of 50 micrometres under the first voxel
    image = VoxelImage(labels, 0.6e-3 / n)
    bare = oxidation_field(plate(n, 0.6e-3), SCHEME, times, exposed=("x-",)).fields["PO"][0]
    barred = oxidation_field(image, {0: SCHEME, 1: None}, times, exposed=("x-",))
    skin = dataclasses.replace(SCHEME, ph0=0.0, pooh0=0.0)
    covered = oxidation_field(image, {0: SCHEME, 1: skin}, times, exposed=("x-",))
    for found in (barred, covered):
        assert np.all(found.fields["PO"][0, 1:3] == 0.0)
    assert np.all(barred.fields["O2"][0, 1:3] == 0.0)
    assert covered.fields["O2"][0, 1:3].min() > 0.0  # oxygen crosses the skin
    # before the barrier the oxygen no longer leaves for the heart: more oxidation
    assert barred.fields["PO"][0, 0, 0, 0] > 1.1 * bare[0, 0, 0]
    # behind it: the oxygen of the storage alone, a little of that of the bare plate
    behind = [f[3:7].sum() for f in (barred.fields["PO"][0], covered.fields["PO"][0], bare)]
    assert behind[0] < 0.1 * behind[2] and behind[2] < behind[1]
    stored = SCHEME.surface_oxygen
    assert barred.fields["PO"][0, 3:, 0, 0] == pytest.approx(stored, rel=0.5)
    # a skin of another solubility: the oxygen at equilibrium on both sides of the face
    dense = dataclasses.replace(skin, solubility=4.0 * SCHEME.solubility)
    oxygen = oxidation_field(image, {0: SCHEME, 1: dense}, [1.0], exposed=("x-",)).fields["O2"]
    assert oxygen[0, 1, 0, 0] == pytest.approx(4.0 * stored, rel=1e-3)


def test_units_faces_and_refusals():
    times = [100.0 * HOUR]
    metres = oxidation_field(plate(12, 0.3e-3), SCHEME, times)
    microns = oxidation_field(
        VoxelImage(np.zeros((12, 1, 1), np.uint8), 300.0 / 12), SCHEME, times, unit=1e-6
    )
    assert microns.fields["PO"] == pytest.approx(metres.fields["PO"], rel=1e-9)
    # every face exposed: a cube oxidizes alike along its three axes
    cube = oxidation_field(
        VoxelImage(np.zeros((6, 6, 6), np.uint8), 0.6e-3 / 6),
        SCHEME,
        times,
        exposed=("x-", "x+", "y-", "y+", "z-", "z+"),
    ).fields["PO"][0]
    assert cube == pytest.approx(cube.transpose(1, 2, 0), rel=1e-9)
    assert cube == pytest.approx(cube[::-1], rel=1e-9)
    assert cube[0, 0, 0] > cube[0, 3, 3] > cube[3, 3, 3]  # corners, faces, heart
    # a lower pressure of oxygen: less oxidation
    low = oxidation_field(plate(12, 0.3e-3), SCHEME, times, pressure=0.5 * SCHEME.pressure)
    assert low.fields["PO"].max() < metres.fields["PO"].max()
    image = VoxelImage(np.array([0, 1], np.uint8).reshape(2, 1, 1), 1e-5)
    with pytest.raises(ValueError, match="none for the phases"):
        oxidation_field(image, {0: SCHEME}, times)
    with pytest.raises(ValueError, match="one phase at least"):
        oxidation_field(image, {0: None, 1: None}, times)
    with pytest.raises(ValueError, match="ClosedLoopOxidation"):
        oxidation_field(image, {0: SCHEME, 1: 3.0}, times)
    with pytest.raises(ValueError, match="the schemes differ"):
        oxidation_field(image, {0: SCHEME, 1: dataclasses.replace(SCHEME, pressure=1e4)}, times)
    with pytest.raises(ValueError, match="faces among"):
        oxidation_field(image, SCHEME, times, exposed=("w+",))
    with pytest.raises(ValueError, match="times"):
        oxidation_field(image, SCHEME, [2.0, 1.0])
    with pytest.raises(RuntimeError, match="steps before"):
        oxidation_field(image, SCHEME, times, max_steps=5)
