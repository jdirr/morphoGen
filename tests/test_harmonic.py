"""Harmonic homogenization: complex laws of viscoelastic solids, of leaky dielectrics and
of absorbing optical media, in the quasi-static regime.

Exact references: laminates (the complex laws of layers in series and in parallel), the
two-dimensional checkerboard of Dykhne (the geometric mean of its phases, real or
complex), the static and instantaneous limits of the frequencies, and the harmonic
regime of the same viscoelastic layers followed in time.
"""

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import Material, homogenize
from morphogen.homogenization.coupled import laminate
from morphogen.homogenization.harmonic import (
    HarmonicResponse,
    _half_plane,
    complex_conductivity,
    dynamic_stiffness,
    effective_index,
    harmonic_tensor,
)
from morphogen.homogenization.materials import bulk_shear, isotropic_matrix

VISCO = Material(young=3.0, poisson=0.3, viscoelasticity={
    "shear": [[0.5, 1.0], [0.3, 10.0]], "bulk": [[0.2, 2.0]]})  # fmt: skip
ELASTIC = Material(young=30.0, poisson=0.2)
EPS0 = 8.854e-12


def layers(n=8, thickness=3):
    data = np.zeros((2, 2, n), dtype=np.uint8)
    data[:, :, :thickness] = 1
    return mg.VoxelImage(data, 1.0, periodic=True)


def complex_stiffness(material, omega):
    k, g = material.viscoelasticity.complex_moduli(omega, *bulk_shear(3.0, 0.3))
    return (k - 2.0 * g / 3.0) * isotropic_matrix(1.0, 0.0) + g * isotropic_matrix(0.0, 1.0)


def test_complex_stiffness_of_viscoelastic_layers():
    omega = np.array([1e-2, 1.0, 1e2])
    r = dynamic_stiffness(layers(), {0: VISCO, 1: ELASTIC}, omega, tol=1e-11)
    assert r.kind == "stiffness" and r.tensors.shape == (3, 6, 6)
    for i, w in enumerate(omega):
        exact = laminate([complex_stiffness(VISCO, w), ELASTIC.matrix("elastic")],
                         [5 / 8, 3 / 8], ("displacement",))  # fmt: skip
        assert np.allclose(r.tensors[i], exact, rtol=1e-9, atol=1e-10)
        assert np.allclose(r.tensors[i], r.tensors[i].T)  # complex symmetric
    assert np.all(r.loss[:, [0, 3, 5], [0, 3, 5]] > 0)  # dissipative
    # layers sheared across their plane are in series: the loss factor of the soft one
    # where the stiff one hardly strains
    _, g = VISCO.viscoelasticity.complex_moduli(omega, *bulk_shear(3.0, 0.3))
    series = 1.0 / (5 / 8 / g + 3 / 8 / 12.5)
    assert np.allclose(r.tensors[:, 3, 3], series, rtol=1e-9)
    assert np.allclose(HarmonicResponse.loss_factor(r.tensors[:, 3, 3]), series.imag / series.real)
    assert r.young(2).shape == (3,) and np.all(r.young(2).imag > 0)


def test_static_and_instantaneous_limits_of_the_frequencies():
    """At rest the relaxed moduli, at high frequencies the instantaneous ones: the real
    stiffnesses of the elastic problems."""
    from morphogen.homogenization import make_problem, stiffness_tensor

    rng = np.random.default_rng(1)
    img = mg.VoxelImage((rng.random((6, 6, 6)) < 0.4).astype(np.uint8), 1.0, periodic=True)
    r = dynamic_stiffness(img, {0: VISCO, 1: ELASTIC}, [1e-9, 1e9], tol=1e-10)
    k_inf, g_inf = VISCO.viscoelasticity.long_term(*bulk_shear(3.0, 0.3))
    relaxed = Material.from_moduli(k_inf, g_inf)
    for i, soft in enumerate((relaxed, Material(young=3.0, poisson=0.3))):
        c, _ = stiffness_tensor(make_problem(img, {0: soft, 1: ELASTIC}, "elastic", tol=1e-10))
        assert np.allclose(r.storage[i], c, rtol=1e-6, atol=1e-7 * c.max())
        assert np.abs(r.loss[i]).max() < 1e-6 * c.max()
    # between them the stiffness is complex, its isotropic moduli between those limits
    mid = dynamic_stiffness(img, {0: VISCO, 1: ELASTIC}, [0.5], tol=1e-9)
    assert r.shear[0].real < mid.shear[0].real < r.shear[1].real and mid.shear[0].imag > 0
    assert mid.bulk[0].imag > 0 and 0 < mid.loss_factor(mid.shear)[0] < 1
    entries = mid.entries()
    assert entries["shear_storage_1"] == pytest.approx(mid.shear[0].real)
    assert entries["shear_tan_delta_1"] == pytest.approx(mid.shear[0].imag / mid.shear[0].real)
    assert {"bulk_loss_1", "young_storage_1", "young_loss_1"} <= set(entries)


def test_the_harmonic_regime_followed_in_time():
    """The same layers sheared in their plane by a sinusoidal strain, in time: the stress
    of the last cycles has the amplitude and the phase of the complex stiffness."""
    omega, cycles, per = 2.0, 8, 60
    t = np.linspace(0.0, cycles * 2 * np.pi / omega, cycles * per + 1)
    strain = {c: [0.0] * t.size for c in ("11", "22", "33", "23", "31")}
    strain["12"] = (0.01 * np.sin(omega * t)).tolist()
    path = {"times": t.tolist(), "increments": 1, "strain": strain}
    out = homogenize(layers(), {0: VISCO, 1: ELASTIC}, properties=[], paths={"cycle": path})
    stress = out["cycle.periodic"].stress[-per - 1 :, 5]
    basis = np.column_stack([np.sin(omega * t[-per - 1 :]), np.cos(omega * t[-per - 1 :])])
    storage, loss = np.linalg.lstsq(basis, stress, rcond=None)[0] / 0.01
    c = dynamic_stiffness(layers(), {0: VISCO, 1: ELASTIC}, [omega]).tensors[0, 5, 5]
    assert storage == pytest.approx(c.real, rel=5e-3)
    assert loss == pytest.approx(c.imag, rel=5e-3)


def test_interfacial_polarization_of_maxwell_and_wagner():
    """A resistive layer in series with a conducting one: the complex conductivities add
    as impedances; the permittivity of the stack rises at low frequencies, where the
    charges of the interface screen the conducting layer."""
    a = Material(electrical_conductivity=1e-9, permittivity=3 * EPS0)
    b = Material(electrical_conductivity=1e-4, permittivity=80 * EPS0)
    omega = np.logspace(-2, 8, 6)
    r = complex_conductivity(layers(), {0: a, 1: b}, omega, tol=1e-11)
    sa = 1e-9 + 1j * omega * 3 * EPS0
    sb = 1e-4 + 1j * omega * 80 * EPS0
    assert np.allclose(r.tensors[:, 2, 2], 1.0 / (5 / 8 / sa + 3 / 8 / sb), rtol=1e-9)
    assert np.allclose(r.tensors[:, 0, 0], 5 / 8 * sa + 3 / 8 * sb, rtol=1e-12)
    eps = r.permittivity[:, 2, 2] / EPS0
    assert eps[0] == pytest.approx(3.0 / (5 / 8), rel=1e-4)  # the resistive layer alone
    assert eps[-1] == pytest.approx(1.0 / (5 / 8 / 3 + 3 / 8 / 80), rel=1e-6)  # capacitors
    assert np.all(np.diff(eps) < 1e-9)
    assert r.conductivity[0, 2, 2] == pytest.approx(1.0 / (5 / 8 / 1e-9 + 3 / 8 / 1e-4), rel=1e-6)
    # a perfect insulator and a conductor without a permittivity: the half plane is closed
    pure = {0: Material(permittivity=3 * EPS0), 1: Material(electrical_conductivity=1e-4)}
    ideal = complex_conductivity(layers(), pure, [1e3])
    series = 1.0 / (5 / 8 / (1j * 1e3 * 3 * EPS0) + 3 / 8 / 1e-4)
    assert ideal.tensors[0, 2, 2] == pytest.approx(series, rel=1e-8)
    entries = r.entries()
    assert entries["permittivity_1"] == pytest.approx(np.trace(r.permittivity[0]) / 3.0)


def test_effective_index_of_layers_and_of_a_checkerboard():
    # form birefringence of a metal-dielectric stack: the ordinary and extraordinary indices
    r = effective_index(layers(), {0: 1.5, 1: 0.2 + 3.0j}, [0.5], tol=1e-11)
    e0, e1 = 1.5**2, (0.2 + 3.0j) ** 2
    assert r.tensors[0, 0, 0] == pytest.approx(5 / 8 * e0 + 3 / 8 * e1, rel=1e-10)
    assert r.tensors[0, 2, 2] == pytest.approx(1.0 / (5 / 8 / e0 + 3 / 8 / e1), rel=1e-10)
    assert r.index[0, 0] == pytest.approx(np.sqrt(5 / 8 * e0 + 3 / 8 * e1), rel=1e-10)
    assert np.all(r.index.imag > 0)
    assert r.absorption[0, 2] == pytest.approx(4 * np.pi * r.index[0, 2].imag / 0.5)
    # Dykhne (1970): a checkerboard has the geometric mean of its phases, complex or not
    n = 32
    board = (np.add.outer(np.arange(n) // (n // 2), np.arange(n) // (n // 2)) % 2).astype(np.uint8)
    image = mg.VoxelImage(board[:, :, None], 1.0, periodic=True)
    r = effective_index(image, {0: 1.5, 1: 2.0 + 0.5j}, [1.0, 2.0], tol=1e-9)
    assert r.tensors[0, 0, 0] == pytest.approx(1.5 * (2.0 + 0.5j), rel=2e-3)
    assert r.index[0, 0] == pytest.approx(np.sqrt(1.5 * (2.0 + 0.5j)), rel=1e-3)
    assert r.index[0, 0] == pytest.approx(r.index[0, 1], rel=1e-9)  # the symmetry of the square
    assert np.allclose(r.tensors[0], r.tensors[1])  # indices without dispersion
    # indices that depend on the wavelength, from the materials
    glass = Material(refractive_index=[[1.50, 0.0], [1.52, 0.0]])
    dye = Material(refractive_index=[[1.7, 0.01], [1.8, 0.2]])
    assert glass.refractive_index == (1.50 + 0j, 1.52 + 0j)
    assert Material.from_config(dye.to_config()) == dye
    assert Material(refractive_index=[1.7, 0.01]).refractive_index == 1.7 + 0.01j
    r = effective_index(layers(), {0: glass, 1: dye}, [0.45, 0.65])
    assert r.tensors[1, 0, 0] == pytest.approx(5 / 8 * 1.52**2 + 3 / 8 * (1.8 + 0.2j) ** 2)
    with pytest.raises(ValueError, match="one per wavelength"):
        effective_index(layers(), {0: glass, 1: dye}, [0.4, 0.5, 0.6])
    with pytest.raises(ValueError, match="refractive_index"):
        effective_index(layers(), {0: glass, 1: Material(young=1.0, poisson=0.3)}, [0.4, 0.5])


def test_laws_in_a_half_plane():
    eye = np.eye(3)
    assert _half_plane([eye, 2.0 * eye]) % (2 * np.pi) == pytest.approx(0.0, abs=1e-12)
    theta = _half_plane([2.25 * eye, (-10.0 + 1.0j) * eye])  # a dielectric and a metal
    for value in (2.25, -10.0 + 1.0j):
        assert (np.exp(1j * theta) * value).real > 0
    with pytest.raises(ValueError, match="half"):
        _half_plane([eye, -eye])  # opposite signs without losses: resonances
    with pytest.raises(ValueError, match="half"):
        _half_plane([(1.0 + 1.0j) * eye, (1.0 - 1.0j) * eye, (-1.0 + 0.1j) * eye])
    img = layers()
    mat = img.data.ravel().astype(int)
    with pytest.raises(ValueError, match="complex symmetric"):
        harmonic_tensor(img, [np.array([[1, 1j, 0], [-1j, 1, 0], [0, 0, 1]]), eye], mat)
    # a plasmonic stack: the exact laminate, through the turned problem
    laws = np.array([2.25 * eye, (-10.0 + 1.0j) * eye])
    tensor, problem = harmonic_tensor(img, laws, mat, tol=1e-11)
    assert tensor[0, 0] == pytest.approx(5 / 8 * 2.25 + 3 / 8 * laws[1, 0, 0], rel=1e-9)
    assert tensor[2, 2] == pytest.approx(1 / (5 / 8 / 2.25 + 3 / 8 / laws[1, 0, 0]), rel=1e-8)
    assert not problem.definite


def test_complex_symmetric_systems_by_conjugate_orthogonal_gradients():
    """The solver of the complex laws: a dense complex symmetric system in its real form,
    then the same effective tensor as MINRES on that form in far fewer iterations."""
    from morphogen.homogenization.coupled import CoupledProblem, conjugate_orthogonal_gradient

    rng = np.random.default_rng(0)
    n = 12
    a = rng.standard_normal((n, n))
    k = a @ a.T + n * np.eye(n) + 1j * (lambda b: b + b.T)(rng.standard_normal((n, n)))
    c = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def apply(v):  # [x, y] -> [Re(K z), -Im(K z)], the unknowns of a node side by side
        f = v.reshape(-1, 2)
        out = k @ (f[:, 0] + 1j * f[:, 1])
        return np.column_stack([out.real, -out.imag]).ravel()

    rhs = np.column_stack([c.real, -c.imag]).ravel()
    x, its, res = conjugate_orthogonal_gradient(apply, lambda r: r.copy(), rhs, 1e-12, 200, 2)
    f = x.reshape(-1, 2)
    assert res <= 1e-12 and its <= 3 * n
    assert np.allclose(f[:, 0] + 1j * f[:, 1], np.linalg.solve(k, c), rtol=1e-9)
    # a random composite of a lossy and of an elastic phase
    data = (rng.random((6, 6, 6)) < 0.4).astype(np.uint8)
    image = mg.VoxelImage(data, 1.0, periodic=True)
    laws = np.array([complex_stiffness(VISCO, 1.0), ELASTIC.matrix("elastic")], dtype=complex)
    tensor, problem = harmonic_tensor(image, laws, data.ravel(), ("displacement",), tol=1e-9)
    infos = [info for _, info in problem._units.values()]
    assert {i.method for i in infos} == {"fft-cocg"}
    theta = _half_plane(list(laws))
    turned = np.exp(1j * theta) * laws
    real = np.zeros((2, 12, 12))
    real[:, :6, :6], real[:, 6:, 6:] = turned.real, -turned.real
    real[:, :6, 6:] = real[:, 6:, :6] = -turned.imag
    slow = CoupledProblem(image, real, ("displacement",) * 2, data.ravel(), tol=1e-9)
    unit = np.zeros(12)
    unit[0] = 1.0
    flux = slow.solve(unit).flux
    column = (flux[:6] - 1j * flux[6:]) * np.exp(-1j * theta)
    assert np.allclose(column, tensor[:, 0], rtol=1e-6, atol=1e-7 * abs(tensor[0, 0]))
    minres_its = next(iter(slow._units.values()))[1].iterations
    assert slow._units[0][1].method == "fft-minres" and infos[0].iterations < 0.5 * minres_its
    with pytest.raises(ValueError, match="twice the same fields"):
        CoupledProblem(image, real, ("displacement", "potential", "potential"), data.ravel(),
                       pairs=True)  # fmt: skip


def test_rve_statistics_of_harmonic_properties(tmp_path):
    """The entries of the harmonic properties, numbered by their frequencies, are
    properties of RVE studies: the storage and loss moduli of random layers."""
    import json

    from morphogen import pipeline
    from morphogen.config import ConfigError, load_config
    from morphogen.homogenization.harmonic import entry_names

    assert entry_names("dynamic_stiffness", 2)[:2] == ("bulk_storage_1", "bulk_loss_1")
    assert "shear_tan_delta_2" in entry_names("dynamic_stiffness", 2)
    assert entry_names("complex_conductivity", 1) == ("conductivity_1", "permittivity_1")
    assert entry_names("refractive_index", 2) == ("index_1", "extinction_1", "index_2",
                                                  "extinction_2")  # fmt: skip
    with pytest.raises(ValueError, match="harmonic property"):
        entry_names("stiffness", 1)
    data = {
        "model": {"type": "random_laminate", "thickness": 0.9, "spacing": 1.1},
        "domain": {"periodic": True},
        "seed": 3,
        "voxels": {"voxel_size": 1.0},
        "homogenization": {"method": "fft", "frequencies": [0.1, 10.0],
                           "properties": ["dynamic_stiffness"],
                           "materials": {0: VISCO.to_config(), 1: ELASTIC.to_config()}},
        "rve": {"sizes": [4.0, 6.0], "realizations": [3, 2], "bootstrap": 0,
                "properties": ["dynamic_stiffness.shear_loss_2",
                               "dynamic_stiffness.bulk_storage_1.periodic"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "stack"},
    }  # fmt: skip
    outdir = pipeline.run(load_config(data))
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    rows = pipeline.read_rows(outdir / "results.csv")
    for name in ("shear_loss_2", "bulk_storage_1"):
        stats = report["properties"][f"dynamic_stiffness.{name}.periodic"]["statistics"]
        assert [s["n"] for s in stats] == [3, 2]
        column = [r[f"dynamic_stiffness.periodic.{name}"] for r in rows
                  if r["volume"] == pytest.approx(64.0)]  # fmt: skip
        assert stats[0]["mean"] == pytest.approx(np.mean(column), rel=1e-12) and column[0] > 0
    for bad, match in (
        ("dynamic_stiffness.shear_loss_3", "unknown result 'shear_loss_3'"),
        ("dynamic_stiffness.index_1", "unknown result"),
    ):
        with pytest.raises(ConfigError, match=match):
            load_config({**data, "rve": {**data["rve"], "properties": [bad]}})


def test_harmonic_properties_in_homogenize_and_configurations(tmp_path, capsys):
    import json

    import yaml

    from morphogen import cli, pipeline
    from morphogen.config import ConfigError, load_config

    out = homogenize(layers(), {0: VISCO, 1: ELASTIC}, properties=["dynamic_stiffness"],
                     frequencies=[0.1, 10.0])  # fmt: skip
    res = out["dynamic_stiffness.periodic"]
    assert res["storage"].shape == (2, 6, 6) and res["shear_loss_2"] > 0
    with pytest.raises(ValueError, match="frequencies"):
        homogenize(layers(), {0: VISCO, 1: ELASTIC}, properties=["dynamic_stiffness"])
    materials = {0: {"refractive_index": [1.5, 0.0], "electrical_conductivity": 1e-9,
                     "permittivity": 3 * EPS0},
                 1: {"refractive_index": [0.2, 3.0], "electrical_conductivity": 1e-4,
                     "permittivity": 80 * EPS0}}  # fmt: skip
    data = {
        "model": {"type": "random_laminate", "thickness": 0.9, "spacing": 1.1},
        "domain": {"size": 6.0, "periodic": True},
        "seed": 3,
        "realizations": 2,
        "voxels": {"shape": [2, 2, 24]},
        "homogenization": {"method": "fft", "materials": materials, "wavelengths": [0.5],
                           "frequencies": [1.0, 1e6],
                           "properties": ["refractive_index", "complex_conductivity"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "stack"},
    }  # fmt: skip
    outdir = pipeline.run(load_config(data))
    rows = pipeline.read_rows(outdir / "results.csv")
    for row in rows:
        f1 = row["fraction_1"]
        ordinary = np.sqrt((1 - f1) * 1.5**2 + f1 * (0.2 + 3.0j) ** 2)
        extra = np.sqrt(1.0 / ((1 - f1) / 1.5**2 + f1 / (0.2 + 3.0j) ** 2))
        mean = (2 * ordinary + extra) / 3.0
        assert row["refractive_index.periodic.index_1"] == pytest.approx(mean.real, rel=1e-6)
        assert row["refractive_index.periodic.extinction_1"] == pytest.approx(mean.imag, rel=1e-6)
        assert (
            row["complex_conductivity.periodic.permittivity_1"]
            > (row["complex_conductivity.periodic.permittivity_2"])
        )
    del data["homogenization"]["wavelengths"]
    with pytest.raises(ConfigError, match="wavelengths"):
        load_config(data)
    data["homogenization"]["wavelengths"] = [-1.0]
    with pytest.raises(ConfigError, match="positive"):
        load_config(data)
    # the command line
    path = tmp_path / "img.vti"
    mg.io.save_image(layers(), path)
    (tmp_path / "materials.yaml").write_text(yaml.safe_dump(materials), encoding="utf-8")
    args = ["homogenize", str(path), "--materials", str(tmp_path / "materials.yaml"),
            "--property", "refractive_index", "complex_conductivity", "--wavelengths", "0.5",
            "0.6", "--frequencies", "1", "1e6"]  # fmt: skip
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "refractive_index.periodic" in text and "extinction" in text and "wavelength" in text
    assert cli.main([*args, "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert np.array(res["refractive_index.periodic"]["index"]).shape == (2, 3)
    assert len(res["complex_conductivity.periodic"]["frequencies"]) == 2
