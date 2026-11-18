"""Crystal plasticity of Méric and Cailletaud: slip systems, interaction matrices, the
implicit integration (consistent tangent, large steps, substeps) in small and finite
strains, single crystals and polycrystals along loading paths."""

import dataclasses

import numpy as np
import pytest
from scipy.linalg import logm

from helpers import box_mesh
from morphogen.homogenization import Material
from morphogen.homogenization.behaviours import update_points
from morphogen.homogenization.crystal import (
    INTERACTIONS,
    CrystalPlasticity,
    interaction_types,
    slip_systems,
)
from morphogen.homogenization.elements import tensor_to_voigt
from morphogen.homogenization.finite import gradient_of, update_points_finite
from morphogen.homogenization.materials import material_table
from morphogen.homogenization.nonlinear import (
    NonlinearFEProblem,
    behaviour_table,
    simulate_point,
)
from morphogen.mesh.conforming import has_mmg
from morphogen.orientations import euler_to_matrix, random_orientations

#: Copper-like cubic elasticity (MPa)
CUBIC = (168e3, 121e3, 75e3)


def crystal(**params) -> Material:
    """A cubic crystal with octahedral slip."""
    spec = {"tau0": 100.0, "K": 5.0, "n": 20.0, **params}
    return dataclasses.replace(
        Material.cubic(*CUBIC), crystal_plasticity=CrystalPlasticity.from_config(spec)
    )


def test_slip_systems():
    for family, count in (("octahedral", 12), ("cube", 6), ("bcc110", 12), ("bcc112", 12)):
        n, d = slip_systems(family)
        assert n.shape == d.shape == (count, 3)
        assert np.allclose(np.linalg.norm(n, axis=1), 1) and np.allclose((n * d).sum(1), 0)
        pairs = {(tuple(np.round(a, 6)), tuple(np.round(b, 6))) for a, b in zip(n, d, strict=True)}
        assert len(pairs) == count
    n, d = slip_systems("octahedral")
    types = interaction_types(n, d)
    assert np.all(np.diag(types) == 0) and np.array_equal(types, types.T)
    # per system: 2 coplanar, 2 Hirth, 1 collinear, 4 glissile and 2 Lomer interactions
    for s in range(12):
        assert np.bincount(types[s], minlength=6).tolist() == [1, 2, 2, 1, 4, 2]
    cp = CrystalPlasticity.from_config(
        {"tau0": 1, "K": 1, "n": 5, "interaction": [1.0, 1.0, 0.6, 12.3, 1.6, 1.3]}
    )
    h = cp.interaction_matrix()
    assert np.allclose(np.diag(h), 1.0) and np.isclose(h.max(), 12.3)
    assert INTERACTIONS[3] == "collinear"


def test_crystal_plasticity_configuration():
    spec = {
        "families": {"octahedral": {}, "cube": {"tau0": 300.0}},
        "tau0": 120.0,
        "Q": 50.0,
        "b": 8.0,
        "K": 30.0,
        "n": 10.0,
        "interaction": [1.0, 0.5],
    }
    cp = CrystalPlasticity.from_config(spec)
    assert cp.n_systems == 18
    par = cp.params_array()
    assert np.all(par[:12, 0] == 120.0) and np.all(par[12:, 0] == 300.0)
    h = cp.interaction_matrix()
    assert np.allclose(np.diag(h), 1.0) and np.allclose(h[0, 1:], 0.5)
    assert CrystalPlasticity.from_config(cp.to_config()) == cp
    m = Material.from_config({"cubic": [168e3, 121e3, 75e3], "crystal_plasticity": spec})
    assert m.crystal_plastic and m.nonlinear and not m.plastic and m.oriented("elastic")
    assert Material.from_config(m.to_config()) == m
    iso = Material.from_config({"young": 2e5, "poisson": 0.3, "crystal_plasticity": spec})
    assert iso.oriented("elastic")  # its slip systems turn with the grains
    errors = [
        ({"K": 1, "n": 5}, "give"),
        ({"tau0": 1, "K": 1, "n": 5, "families": ["hcp"]}, "slip family"),
        ({"tau0": 1, "K": 1, "n": 5, "speed": 3}, "unknown key"),
        ({"tau0": 1, "K": 0, "n": 5}, "K > 0"),
        ({"tau0": 1, "K": 1, "n": 5, "interaction": [1, 2, 3]}, "interaction"),
        ({"tau0": 1, "K": 1, "n": 5, "interaction": np.ones((3, 3)).tolist()}, "12 x 12"),
        (
            {"tau0": 1, "K": 1, "n": 5, "families": ["bcc110"], "interaction": [1] * 6},
            "octahedral systems only",
        ),
    ]
    for bad, match in errors:
        with pytest.raises(ValueError, match=match):
            CrystalPlasticity.from_config(bad)
    with pytest.raises(ValueError, match="not both"):
        Material(young=1.0, poisson=0.3, plasticity={"yield": 1.0}, crystal_plasticity=cp)
    with pytest.raises(ValueError, match="no plasticity"):
        Material(void=True, crystal_plasticity=cp)


@pytest.mark.parametrize(
    "euler, axis, schmid, n_active",
    [
        ((0.0, 0.0, 0.0), (0, 0, 1), 1 / np.sqrt(6), 8),
        ((0.0, np.degrees(np.arccos(1 / np.sqrt(3))), 45.0), (1, 1, 1), 2 / (3 * np.sqrt(6)), 6),
    ],
)
def test_schmid_law(euler, axis, schmid, n_active):
    """Tension of a single crystal along [001] or [111]: the flow stress is the critical
    resolved shear stress plus the viscous stress of the slip rate, over the Schmid
    factor, with 8 or 6 equally active systems."""
    g = euler_to_matrix(euler)[0]  # crystal = g @ sample: the tensile axis in the crystal
    assert np.allclose(g[:, 2], np.array(axis) / np.linalg.norm(axis))
    m = crystal()
    rate = 1e-3
    res = simulate_point(
        m, {"times": [0, 20], "increments": 40, "strain": {"33": 20 * rate}}, orientation=euler
    )
    gamma_rate = rate / (n_active * schmid)
    expected = (100.0 + 5.0 * gamma_rate ** (1 / 20)) / schmid
    assert res.stress[-1, 2] == pytest.approx(expected, rel=1e-6)
    assert np.allclose(res.stress[-1, [0, 1, 3, 4, 5]], 0, atol=1e-6)
    assert res.info["cuts"] == 0


def _kernel(material, euler, finite):
    mats = {0: material}
    table = material_table([0], mats, "elastic", grains=[0], orientations=np.array([euler]))
    return table, behaviour_table([0], mats, table, finite=finite)


def _point(table, beh, grad, old, dt, finite, guess=0, new=None):
    nc = 9 if finite else 6
    new = np.zeros_like(old) if new is None else new.copy()
    s, a = np.zeros((1, nc)), np.zeros((1, nc, nc))
    update = update_points_finite if finite else update_points
    failed = update(beh.kind, table.mat, beh.pmat, beh.data, grad[None], 0.0, dt, old, new, s, a,
                    guess, 1.0)  # fmt: skip
    assert failed == 0
    return s[0], a[0], new


@pytest.mark.parametrize("finite", [False, True])
def test_consistent_tangent(finite):
    """The tangent of the implicit update against finite differences, with isotropic and
    kinematic hardening and a Franciosi interaction matrix; the same state from a warm
    start."""
    m = crystal(tau0=80.0, Q=40.0, b=8.0, C=3e4, D=300.0, K=30.0, n=8.0,
                interaction=[1.0, 1.0, 0.6, 12.3, 1.6, 1.3])  # fmt: skip
    table, beh = _kernel(m, (20.0, 35.0, 50.0), finite)
    rng = np.random.default_rng(3)
    old = np.zeros((1, beh.width))
    old[0, 7:19] = rng.normal(0, 1e-3, 12)
    old[0, 19:31] = np.abs(rng.normal(0, 3e-3, 12))
    if finite:
        old[0, 31:40] = rng.normal(0, 2e-3, 9)
        grad = rng.normal(0, 3e-3, 9)
        grad[8] += 5e-3
    else:
        grad = rng.normal(0, 3e-3, 6)
        grad[2] += 5e-3
    s, a, new = _point(table, beh, grad, old, 0.5, finite)
    assert new[0, 6] > old[0, 6]  # slip
    h = 1e-8
    fd = np.zeros_like(a)
    for j in range(grad.size):
        d = np.zeros(grad.size)
        d[j] = h
        up = _point(table, beh, grad + d, old, 0.5, finite)[0]
        down = _point(table, beh, grad - d, old, 0.5, finite)[0]
        fd[:, j] = (up - down) / (2 * h)
    assert np.allclose(a, 0.5 * (fd + fd.T), rtol=0, atol=1e-6 * np.abs(fd).max())
    s1, _, new1 = _point(table, beh, grad, old, 0.5, finite, guess=1, new=new)
    assert np.allclose(new1, new, rtol=0, atol=1e-12) and np.allclose(s1, s, atol=1e-8)


@pytest.mark.parametrize("finite", [False, True])
def test_large_steps(finite):
    """One step far beyond the elastic limit, with many redundant systems loaded at once:
    the local iteration converges (with substeps if need be) to the steady flow."""
    m = crystal(Q=30.0, b=10.0, interaction=[1.0, 1.0, 0.6, 12.3, 1.6, 1.3])
    table, beh = _kernel(m, (31.0, 47.0, 11.0), finite)
    old = np.zeros((1, beh.width))
    grad = np.zeros(9 if finite else 6)
    grad[8 if finite else 2] = 0.03  # uniaxial strain: a huge overstress
    _, a, new = _point(table, beh, grad, old, 0.5, finite)
    assert new[0, 6] > 0.005 and np.all(np.isfinite(a))


#: Crystals whose points keep more than their slips: cleavage planes that open, close and
#: rub, coupled with the slip; twins that turn the lattice.
IRON = {
    "cubic": [230000.0, 135000.0, 117000.0],
    "crystal_plasticity": {
        "K": 10.0,
        "n": 10.0,
        "families": {"bcc110": {"tau0": 150.0, "Q": 60.0, "b": 15.0}},
        "cleavage": {"planes": "cube", "strength": 500.0, "softening": 40.0, "K": 20.0,
                     "n": 1.0, "coupling": 3.0, "friction": 0.3, "closure": 0.5},
    },
}  # fmt: skip
MAGNESIUM = {
    "hexagonal": [59400.0, 25600.0, 21400.0, 61600.0, 16400.0],
    "crystal_plasticity": {
        "c_over_a": 1.624,
        "K": 5.0,
        "n": 8.0,
        "families": {"basal": {"tau0": 5.0}, "prismatic": {"tau0": 40.0},
                     "pyramidal2_ca": {"tau0": 60.0, "Q": 30.0, "b": 10.0}},
        "twins": {"twin1012": {"tau0": 15.0}},
        "twin_reorientation": 0.3,
    },
}  # fmt: skip


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("law", ["slip", "cleavage", "twins"])
def test_points_together_and_alone(finite, law):
    """The points of a volume element are integrated chunk by chunk, each chunk with work
    arrays of its own that keep what its points share (the Schmid tensors of a grain, the
    slips below which a system does not start, the derivatives of the last evaluation):
    the result of a point does not depend on the points integrated before it. Grains of
    several orientations, points that stay elastic among points that flow, over two steps
    (the second one from the slips of the first): the stresses, the tangents and the
    internal variables of the points integrated together are those of each point
    integrated alone, to the bit."""
    if law == "slip":
        m = crystal(tau0=80.0, Q=40.0, b=8.0, C=3e4, D=300.0, K=30.0, n=8.0,
                    interaction=[1.0, 1.0, 0.6, 12.3, 1.6, 1.3])  # fmt: skip
    else:
        m = Material.from_config(IRON if law == "cleavage" else MAGNESIUM)
    n, grains, level = 330, 5, 2.0e-3 if law == "twins" else 1.2e-3
    rng = np.random.default_rng(1)
    grain = np.repeat(rng.integers(0, grains, size=n // 16 + 1), 16)[:n]
    table = material_table(np.zeros(n, dtype=int), {0: m}, "elastic", grains=grain,
                           orientations=rng.uniform(0, 90, size=(grains, 3)))  # fmt: skip
    beh = behaviour_table(np.zeros(n, dtype=int), {0: m}, table, finite=finite)
    nc = 9 if finite else 6
    direction = rng.normal(size=(n, 6)) * level
    direction[:, 2] += 1.5 * level
    direction[: n // 3] *= 0.2  # a third of the points stay elastic longer
    update = update_points_finite if finite else update_points

    def integrate(points, step, old):
        e = step * direction[points]
        if finite:
            grad = np.hstack([e[:, :3], 0.5 * e[:, 3:], 0.5 * e[:, 3:]])
        else:
            grad = e
        new = np.zeros((len(points), beh.width))  # (the state before ``old``: none)
        s, a = np.zeros((len(points), nc)), np.zeros((len(points), nc, nc))
        failed = update(beh.kind[points], table.mat[points], beh.pmat[points], beh.data, grad,
                        0.0, 0.02, old, new, s, a, 0 if step == 1 else 2, 1.0)  # fmt: skip
        return failed, s, a, new

    everyone = np.arange(n)
    old = np.zeros((n, beh.width))
    flowing = 0
    for step in (1, 2):
        failed, s, a, new = integrate(everyone, step, old)
        alone = [integrate(everyone[i : i + 1], step, old[i : i + 1]) for i in range(n)]
        assert failed == sum(one[0] for one in alone) == 0
        assert np.array_equal(s, np.vstack([one[1] for one in alone]))
        assert np.array_equal(a, np.vstack([one[2] for one in alone]))
        assert np.array_equal(new, np.vstack([one[3] for one in alone]))
        flowing = np.count_nonzero(new[:, 6] > old[:, 6])
        old = new
    assert 0.3 * n < flowing < n  # some flow, some do not


def test_homogeneous_crystal_element():
    """A homogeneous single crystal element follows its material point, in small and
    finite strains, with linear and quadratic elements under periodic and kinematic
    conditions, along a cyclic path with a shear stress and a change of temperature."""
    cp = {"tau0": 80.0, "Q": 40.0, "b": 8.0, "C": 3e4, "D": 300.0, "K": 30.0, "n": 8.0}
    m = dataclasses.replace(
        Material.cubic(*CUBIC, expansion=1e-5), crystal_plasticity=CrystalPlasticity.from_config(cp)
    )
    euler = np.array([20.0, 35.0, 50.0])
    path = {"times": [0, 10, 20], "increments": [6, 6], "strain": {"11": [0.01, -0.005]},
            "stress": {"12": [40, -40]}, "temperature": [50, 0]}  # fmt: skip
    for kinematics in ("small", "finite"):
        ref = simulate_point(m, path, orientation=euler, kinematics=kinematics)
        assert ref.plastic[-1] > 0.01
        for order, bc in [(1, "periodic"), (1, "kubc"), (2, "periodic")]:
            mesh = box_mesh((1, 1, 1), order=order, perturb=0.2)
            mesh = dataclasses.replace(
                mesh, cell=np.zeros(mesh.n_elements, int), orientations=euler[None]
            )
            res = NonlinearFEProblem(mesh, {0: m}, bc, kinematics=kinematics).run(path)
            scale = np.abs(ref.stress).max()
            assert np.allclose(res.stress, ref.stress, rtol=0, atol=2e-6 * scale)
            assert np.allclose(res.strain, ref.strain, rtol=0, atol=1e-7)
            assert res.plastic[-1] == pytest.approx(ref.plastic[-1], rel=1e-6)
            if kinematics == "finite":
                assert np.allclose(res.cauchy, ref.cauchy, rtol=0, atol=2e-6 * scale)


def polycrystal(n_cells=6, n=(3, 3, 3), order=2, seed=1):
    """Periodic Voronoi grains on a structured mesh, random orientations."""
    rng = np.random.default_rng(seed)
    mesh = box_mesh(n, upper=(1.0, 1.0, 1.0), order=order)
    seeds = rng.random((n_cells, 3))
    centres = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    d = centres[:, None, :] - seeds[None, :, :]
    d -= np.round(d)
    cell = np.argmin((d**2).sum(-1), axis=1)
    return dataclasses.replace(mesh, cell=cell, orientations=random_orientations(rng, n_cells))


def test_polycrystal():
    """Tension of a polycrystal: the flow stress lies between the Sachs (largest Schmid
    factor, 1/2) and Taylor (3.06) estimates; in finite strains, the lattices rotate,
    and the response at small strains is that of the small strain theory."""
    m = crystal(tau0=60.0, K=10.0, n=15.0)
    mesh = polycrystal()
    path = {"times": [0, 10], "increments": 10, "strain": {"33": 0.01}}
    small = NonlinearFEProblem(mesh, {0: m}, "periodic").run(path)
    assert small.info["cuts"] == 0
    flow = small.stress[-1, 2]
    assert 2.0 * 60.0 < flow < 3.06 * 61.0 * 1.2  # viscous stress: a few MPa
    finite = NonlinearFEProblem(mesh, {0: m}, "periodic", kinematics="finite").run(
        path, fields=True
    )
    assert np.allclose(finite.stress[:, 2], small.stress[:, 2], rtol=0.03, atol=1.0)
    f = np.eye(3) + gradient_of(finite.strain[-1]).reshape(3, 3)
    assert f[1, 0] == f[2, 0] == f[2, 1] == 0.0  # upper triangular
    hencky = tensor_to_voigt(0.5 * logm(f.T @ f).real, strain=True)
    assert np.allclose(finite.log_strain[-1], hencky, rtol=0, atol=1e-12)
    # tension along 3 with free faces: a uniaxial Cauchy stress
    cauchy = finite.cauchy[-1]
    assert np.all(np.abs(cauchy[[0, 1, 3, 4, 5]]) < 1e-6 * cauchy[2])
    fields = finite.fields[10.0]
    turned = fields["orientation"]
    initial = mesh.orientations[mesh.cell]
    g0 = euler_to_matrix(initial)
    g1 = euler_to_matrix(turned)
    angle = np.degrees(np.arccos(np.clip((np.trace(g1 @ np.swapaxes(g0, 1, 2), axis1=1,
                                                            axis2=2) - 1) / 2, -1, 1)))  # fmt: skip
    assert 0.0 < angle.max() < 5.0  # lattice rotations of a few tenths of a degree
    assert fields["displacement"].shape == (mesh.n_nodes, 3)


@pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")
def test_crystal_plasticity_pipeline(tmp_path):
    """The template of a polycrystal in finite strains, scaled down: curves with the true
    strains and stresses, fields with the lattice orientations and the displacement; the
    command line in finite strains on its mesh."""
    import json

    import yaml

    from morphogen import cli, io, pipeline
    from morphogen.config import ConfigError, load_config

    cfg = yaml.safe_load(cli._templates()["homogenization_crystal_plasticity"])
    cfg["model"]["n_cells"] = 3
    cfg["mesh"].update({"size": 0.5, "min_feature": 0.1})
    path = {"times": [0, 6], "increments": 3, "strain": {"11": 0.006}}  # the template's rate
    cfg["homogenization"]["paths"]["tension"] = path
    cfg["outputs"].update({"directory": str(tmp_path / "out"), "conforming_mesh": ["vtu"]})
    spec = load_config(cfg).homogenization
    assert spec.options()["nonlinear"] == {"kinematics": "finite"}
    assert spec.materials[1].crystal_plastic
    outdir = pipeline.run(load_config(cfg))
    rows = pipeline.read_rows(outdir / "polycrystal_0000_tension_periodic.csv")
    assert len(rows) == 4 and {"LE11", "TS11", "TS11_1", "p"} <= set(rows[-1])
    assert float(rows[-1]["E11"]) == pytest.approx(0.006)
    assert float(rows[-1]["TS11"]) > float(rows[-1]["S11"]) > 0
    fields = io.vtk.read_vtu(outdir / "polycrystal_0000_tension_periodic_1.vtu")
    assert {"orientation", "euler", "stress", "p"} <= set(fields["cell_data"])
    assert fields["point_data"]["displacement"].shape == (len(fields["points"]), 3)
    data = json.loads(json.dumps(cfg))
    data["homogenization"]["nonlinear"] = {"kinematics": "large"}
    with pytest.raises(ConfigError, match="kinematics"):
        load_config(data)
    # command line, on the mesh of the realization
    mesh_file = next(outdir.glob("*_tet.vtu"))
    (tmp_path / "mats.yaml").write_text(
        yaml.safe_dump({1: spec.materials[1].to_config()}), encoding="utf-8"
    )
    report = tmp_path / "report.json"
    code = cli.main([
        "homogenize", str(mesh_file), "--materials", str(tmp_path / "mats.yaml"),
        "--path", "tension: t=0,6 n=3 E11=0.006", "--finite", "--curves", str(tmp_path),
        "--periodic", "-o", str(report),
    ])  # fmt: skip
    assert code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["paths"]["tension.periodic"]["TS11"] == pytest.approx(
        float(rows[-1]["TS11"]), rel=1e-3
    )
