"""The ready-made tests of :mod:`morphogen.contact.experiments`: the compression of a block
between two platens (the uniaxial stress with the penalties of the platens in series;
stiffer with friction, below the modulus of the block held on its sides), of a block
with a slot (softer until the slot closes on itself, then stiffer), and the indentation
of a block by a sphere and by a cone (the force grows as Hertz's and Sneddon's laws, the
modulus that it reads, the unloading comes back along the loading), their rows, their
errors, and their command lines."""

import json

import numpy as np
import pytest

from morphogen import cli, io
from morphogen.contact import (
    CompressionResult,
    IndentationResult,
    Surface,
    compression,
    effective_modulus,
    indentation,
    layer_correction,
)
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage

STEEL = Material(young=1000.0, poisson=0.3)


def _block(shape, spacing, order=1):
    image = VoxelImage(np.zeros(shape, np.uint8), spacing=spacing, periodic=False)
    mesh = voxel_tet_mesh(image)
    return mesh.quadratic() if order == 2 else mesh


def _slotted():
    """A block of 4 x 2 x 2 with a slot of 2 x 2 x 0.125 through it, at mid-height."""
    data = np.ones((8, 4, 16), np.uint8)
    data[2:6, :, 8] = 0
    image = VoxelImage(data, spacing=(0.5, 0.5, 0.125), periodic=False)
    return voxel_tet_mesh(image).subset([1])


@pytest.mark.parametrize("order", [1, 2])
def test_block_between_lubricated_platens_is_in_uniaxial_stress(order):
    """Without friction the block widens freely: the stress of the uniaxial stress, with
    the penalties of the two platens in series."""
    mesh = _block((3, 3, 4), (1.0, 1.0, 1.0), order)
    res = compression(mesh, {0: STEEL}, -0.01, steps=4, kinematics="small", self_contact=False)
    assert isinstance(res, CompressionResult) and res.height == 4.0 and res.section == 9.0
    penalty = 10 * 1000.0 / Surface.of(mesh).size(mesh.nodes)
    exact = 1.0 / (1.0 / 1000.0 + 2.0 / (penalty * res.height))
    assert res.modulus == pytest.approx(exact, rel=2e-4)
    np.testing.assert_allclose(res.stress, exact * res.strain, rtol=2e-4, atol=1e-12)
    np.testing.assert_allclose(res.tangent(), exact, rtol=2e-4)
    np.testing.assert_allclose(res.area[1:], 9.0, rtol=1e-9)  # the whole of its two faces
    assert res.area[0].tolist() == [0.0, 0.0] and not res.self_area.any()
    np.testing.assert_allclose(res.force, -res.stress * res.section)
    # the block widens by Poisson's ratio: its sides move apart
    u = res.result.displacement
    wide = u[mesh.nodes[:, 0] > 3 - 1e-9, 0].mean() - u[mesh.nodes[:, 0] < 1e-9, 0].mean()
    assert wide / 3.0 == pytest.approx(-0.3 * res.stress[-1] / 1000.0, rel=1e-3)
    rows = res.rows()
    assert rows[-1]["strain"] == pytest.approx(-0.01)
    assert rows[-1]["area_moving"] == pytest.approx(9.0)
    assert set(rows[0]) == {"strain", "stress", "force", "area_fixed", "area_moving",
                            "self_area", "friction"}  # fmt: skip
    # the contact points at the nodes of the faces: the nodes of the edges of the block are
    # those of its sides as well, which the platens see edge-on and which carry nothing
    nodes = compression(mesh, {0: STEEL}, -0.01, steps=4, kinematics="small", points="nodes")
    np.testing.assert_allclose(nodes.stress, res.stress, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(nodes.area[1:], 9.0, rtol=1e-6)
    assert not nodes.self_area.any()


def test_block_between_platens_that_rub_is_stiffer():
    """With friction the faces of the block cannot widen: stiffer than in uniaxial
    stress, softer than the block held on all its sides; friction dissipates."""
    mesh = _block((3, 3, 4), (1.0, 1.0, 1.0))
    free = compression(mesh, {0: STEEL}, -0.01, steps=4, kinematics="small", self_contact=False)
    held = compression(mesh, {0: STEEL}, -0.01, steps=4, kinematics="small", self_contact=False,
                       friction=0.5)  # fmt: skip
    constrained = 1000.0 * 0.7 / (1.3 * 0.4)  # of the uniaxial strain
    assert 1.03 * free.modulus < held.modulus < constrained
    assert held.friction[-1] > 0 and free.friction[-1] == 0
    large = compression(mesh, {0: STEEL}, -0.01, steps=4, self_contact=False)  # finite strains
    assert large.modulus == pytest.approx(free.modulus, rel=0.01)


def test_block_with_a_slot_stiffens_when_the_slot_closes():
    """The slot makes the block softer until its walls meet (the surface of the sample
    against itself); then they press on each other and the block stiffens."""
    mesh = _slotted()
    res = compression(mesh, {1: STEEL}, -0.12, steps=12, friction=0.2)
    closed = res.self_area > 0
    assert not closed[:4].any() and closed[-3:].all()
    assert res.self_area[-1] >= 1.5  # the middle of the slot, of 2 x 2
    tangent = res.tangent()
    assert tangent[-1] > 1.3 * tangent[0] and tangent[0] < 1000.0
    solid = compression(mesh, {1: STEEL}, -0.12, steps=12, friction=0.2, self_contact=False)
    assert not solid.self_area.any()
    assert abs(solid.stress[-1]) < abs(res.stress[-1])  # its walls went through each other
    np.testing.assert_allclose(solid.stress[:4], res.stress[:4], rtol=1e-6)


def test_indentation_by_a_sphere_and_by_a_cone():
    """A coarse block: the force of the sphere grows as the depth to the power 3/2 and
    that of the cone as its square, the modulus read is that of the block within the
    coarseness of its faces, and the unloading comes back along the loading."""
    mesh = _block((8, 8, 4), (1.0, 1.0, 1.0), order=2)
    exact = effective_modulus(1000.0, 0.3)
    ball = indentation(mesh, {0: STEEL}, 0.08, radius=20.0, steps=3, unload=True)
    assert isinstance(ball, IndentationResult) and ball.indenter == "sphere"
    third = 0.08 / 3
    np.testing.assert_allclose(ball.depth, [0, third, 2 * third, 0.08, 2 * third, third, 0],
                               atol=1e-12)  # fmt: skip
    np.testing.assert_allclose(ball.force[4:], ball.force[2::-1], rtol=1e-5, atol=1e-9)
    slope = np.polyfit(np.log(ball.depth[1:4]), np.log(ball.force[1:4]), 1)[0]
    assert slope == pytest.approx(1.5, abs=0.2)
    # a contact of the size of the faces, on a block four of them thick: its base
    # stiffens it by a third, as that of a layer of this thickness (to 5 %)
    assert exact < ball.modulus[3] < 1.4 * exact and ball.modulus[0] == 0
    factor = layer_correction(20.0, 0.08, 4.0, 0.3)
    assert factor == pytest.approx(1.36, abs=0.005)
    assert ball.layer_modulus(0.3)[3] == pytest.approx(ball.modulus[3] / factor)
    assert ball.layer_modulus(0.3)[3] == pytest.approx(exact, rel=0.08)
    assert ball.area[3] > ball.area[1] > 0 and ball.hardness[3] == ball.force[3] / ball.area[3]
    assert ball.pressure[3] > ball.hardness[3]  # the largest pressure, and the mean one
    assert ball.radius[3] == pytest.approx(np.sqrt(20.0 * 0.08), rel=0.2)  # sqrt(R δ)
    assert ball.stiffness > 0 and 0.8 * exact < ball.unloading_modulus < 1.5 * exact
    rows = ball.rows()
    assert rows[3]["force"] == ball.force[3] and set(rows[0]) == {
        "depth", "force", "area", "radius", "pressure", "hardness", "modulus"}  # fmt: skip
    cone = indentation(mesh, {0: STEEL}, 0.4, indenter="cone", steps=3, base="smooth")
    assert cone.unloading_modulus is None and cone.stiffness is None
    with pytest.raises(ValueError, match="that of a sphere"):
        cone.layer_modulus(0.3)
    slope = np.polyfit(np.log(cone.depth[1:]), np.log(cone.force[1:]), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.4)
    assert 0.9 * exact < cone.modulus[-1] < 1.35 * exact
    # at the middle of the face, given: the same force in one step (an elastic block)
    other = indentation(mesh, {0: STEEL}, 0.08, radius=20.0, steps=1, point=(4.0, 4.0))
    assert other.force[-1] == pytest.approx(ball.force[3], rel=1e-5)


@pytest.mark.slow
def test_indentation_reads_the_modulus_of_a_block():
    """A sphere on a block of 12 x 12 x 6 faces, glued on its base or sliding on it: the
    reduced modulus of the block, read with the correction of a layer of its thickness,
    within 7 %."""
    mesh = _block((12, 12, 6), (1.0, 1.0, 1.0), order=2)
    exact = effective_modulus(1000.0, 0.3)
    for base in ("fixed", "smooth"):
        res = indentation(mesh, {0: STEEL}, 0.06, radius=30.0, steps=3, base=base)
        assert res.modulus[-1] > 1.1 * exact  # the half-space is softer than the layer
        assert res.layer_modulus(0.3)[-1] == pytest.approx(exact, rel=0.07)
        assert res.radius[-1] == pytest.approx(np.sqrt(30.0 * 0.06), rel=0.15)


def test_ready_made_tests_check_their_inputs():
    mesh = _block((2, 2, 2), (1.0, 1.0, 1.0))
    for strain in (0.1, -1.5, float("nan")):
        with pytest.raises(ValueError, match="nominal strain"):
            compression(mesh, {0: STEEL}, strain)
    with pytest.raises(ValueError, match="at least one step"):
        compression(mesh, {0: STEEL}, -0.1, steps=0)
    with pytest.raises(ValueError, match="axis"):
        compression(mesh, {0: STEEL}, -0.1, axis=3)
    with pytest.raises(ValueError, match="friction"):
        compression(mesh, {0: STEEL}, -0.1, friction=-1.0)
    with pytest.raises(ValueError, match="contact points among"):
        compression(mesh, {0: STEEL}, -0.1, points="edges")
    with pytest.raises(ValueError, match="indenter among"):
        indentation(mesh, {0: STEEL}, 0.1, indenter="pyramid")
    with pytest.raises(ValueError, match="positive depth"):
        indentation(mesh, {0: STEEL}, -0.1, radius=5.0)
    with pytest.raises(ValueError, match="radius of the sphere"):
        indentation(mesh, {0: STEEL}, 0.1)
    with pytest.raises(ValueError, match="radius of the sphere"):
        indentation(mesh, {0: STEEL}, 0.1, radius=0.05)
    with pytest.raises(ValueError, match="base"):
        indentation(mesh, {0: STEEL}, 0.1, radius=5.0, base="free")
    # along another axis: the same test, turned
    turned = compression(mesh, {0: STEEL}, -0.01, steps=1, axis=0, kinematics="small",
                         self_contact=False)  # fmt: skip
    upright = compression(mesh, {0: STEEL}, -0.01, steps=1, kinematics="small",
                          self_contact=False)  # fmt: skip
    assert turned.modulus == pytest.approx(upright.modulus, rel=1e-9)


def test_command_lines_of_the_ready_made_tests(tmp_path):
    mesh = _slotted()
    mesh_file = tmp_path / "slotted.vtu"
    io.save_mesh(mesh, mesh_file)
    (tmp_path / "mats.yaml").write_text("1: {young: 1000.0, poisson: 0.3}\n", encoding="utf-8")
    report = tmp_path / "compress.json"
    code = cli.main([
        "compress", str(mesh_file), "--materials", str(tmp_path / "mats.yaml"),
        "--strain", "-0.12", "--steps", "12", "--friction", "0.2",
        "--curve", str(tmp_path / "compress.csv"), "--fields", str(tmp_path / "crushed"),
        "-o", str(report),
    ])  # fmt: skip
    assert code == 0
    direct = compression(mesh, {1: STEEL}, -0.12, steps=12, friction=0.2)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["stress"] == pytest.approx(direct.stress[-1], rel=1e-6)
    assert data["modulus"] == pytest.approx(direct.modulus, rel=1e-6)
    assert data["self_area"] == pytest.approx(direct.self_area[-1], rel=1e-6)
    lines = (tmp_path / "compress.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("strain,stress,force") and len(lines) == 14
    assert b"contact_pressure" in (tmp_path / "crushed.vtu").read_bytes()
    block = tmp_path / "block.vtu"
    io.save_mesh(_block((8, 8, 4), (1.0, 1.0, 1.0)), block)
    report = tmp_path / "indent.json"
    code = cli.main([
        "indent", str(block), "--material", "0=young:1000,poisson:0.3", "--depth", "0.08",
        "--radius", "20", "--steps", "2", "--unload", "--small", "--poisson", "0.3",
        "--curve", str(tmp_path / "indent.csv"), "-o", str(report),
    ])  # fmt: skip
    assert code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["indenter"] == "sphere" and data["force"] > 0 and data["unloading_modulus"] > 0
    assert data["depth"] == pytest.approx(0.08) and 0 < data["layer_modulus"] < data["modulus"]
    assert len((tmp_path / "indent.csv").read_text(encoding="utf-8").splitlines()) == 6
    with pytest.raises(SystemExit):
        cli.main(["compress", str(mesh_file), "--materials", str(tmp_path / "mats.yaml")])


def test_correction_of_a_layer_on_a_rigid_base():
    """The series of Dimitriadis et al. (2002) at the two Poisson's ratios whose
    coefficients are published: 1 + 1.133 χ + 1.283 χ² + 0.769 χ³ - 0.0975 χ⁴ (bonded) and
    1 + 0.884 χ + 0.781 χ² + 0.386 χ³ + 0.0048 χ⁴ (free to slide) for 0.5; 1 + 1.112 χ +
    1.237 χ² + 0.712 χ³ - 0.133 χ⁴ (bonded) for 0.49."""
    published = [
        (0.5, True, (1.133, 1.283, 0.769, -0.0975)),
        (0.5, False, (0.884, 0.781, 0.386, 0.0048)),
        (0.49, True, (1.112, 1.237, 0.712, -0.133)),
    ]
    for poisson, bonded, coefficients in published:
        for chi in (0.1, 0.3, 0.6):
            series = 1.0 + sum(c * chi ** (k + 1) for k, c in enumerate(coefficients))
            got = layer_correction(1.0, chi**2, 1.0, poisson, bonded)  # sqrt(R δ) / h = χ
            assert got == pytest.approx(series, abs=2e-3 * chi)
    assert layer_correction(5.0, 0.0, 2.0, 0.3) == 1.0
    assert layer_correction(5.0, 0.1, 1e9, 0.3) == pytest.approx(1.0, abs=1e-8)
    assert layer_correction(5.0, 0.1, 2.0, 0.3) > layer_correction(5.0, 0.1, 2.0, 0.3, False) > 1
    with pytest.raises(ValueError, match="Poisson"):
        layer_correction(5.0, 0.1, 2.0, 0.6)
    with pytest.raises(ValueError, match="thickness"):
        layer_correction(5.0, 0.1, 0.0, 0.3)
