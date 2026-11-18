"""The window method for tensors (rve.windows): the stiffness and the conductivity of the
windows of one periodic realization, their mean mutual energies under the unit loads."""

import itertools
import json
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen import pipeline, rve
from morphogen.config import ConfigError, load_config
from morphogen.domain import Domain
from morphogen.homogenization import (
    VOID,
    FFTElasticProblem,
    FFTThermalProblem,
    Material,
    conductivity_tensor,
    stiffness_tensor,
)
from morphogen.homogenization.postprocess import NOTATION, PLANE_NOTATION
from morphogen.rve import report as rve_report
from morphogen.seeding import derive_seed


def _orthotropic(young, shears, poisson=(0.3, 0.25, 0.2)) -> np.ndarray:
    """An orthotropic stiffness in its axes (Voigt 11 22 33 23 31 12, engineering shears),
    its shear moduli ``shears`` = (G23, G31, G12)."""
    s = np.diag([*(1.0 / e for e in young), *(1.0 / g for g in shears)])
    for (i, j), nu in zip(((0, 1), (0, 2), (1, 2)), poisson, strict=True):
        s[i, j] = s[j, i] = -nu / young[i]
    return np.linalg.inv(s)


# two orthotropic phases whose three shear moduli differ, in another order in each, as their
# conductivities: an exchange of the shear slots (23, 31, 12) or of the axes shows
C0 = _orthotropic((120.0, 80.0, 60.0), (7.0, 11.0, 17.0))
C1 = _orthotropic((300.0, 200.0, 250.0), (40.0, 25.0, 60.0))
K0, K1 = np.diag([1.0, 2.0, 3.0]), np.diag([10.0, 30.0, 20.0])
MATERIALS = {
    0: Material.anisotropic(C0, conductivity=K0),
    1: Material.anisotropic(C1, conductivity=K1),
}
TENSORS = {"stiffness": (C0, C1), "conductivity_tensor": (K0, K1)}


def _random_image(shape, seed, fraction=0.4, spacing=0.5):
    data = (np.random.default_rng(seed).random(shape) < fraction).astype(np.uint8)
    return mg.VoxelImage(data, spacing=spacing, periodic=True)


@pytest.mark.parametrize("prop", ["stiffness", "conductivity_tensor"])
def test_windows_tiling_the_box_give_its_tensor(prop):
    """The window tensors are symmetric to the bit and positive; the mean of the windows that
    tile the box is its mean energy to round-off and, by Hill's lemma, its apparent tensor
    (of its mean stresses, stiffness_tensor) to the tolerance of the solver; their
    statistics per size are those of the ensemble method, the windows as realizations."""
    image = _random_image((8, 8, 8), seed=1)
    windows = rve.window_tensors(image, MATERIALS, [4, 2, 3], property=prop, tol=1e-12)
    d = 6 if prop == "stiffness" else 3
    np.testing.assert_array_equal(windows.edges, [2, 3, 4])
    np.testing.assert_allclose(windows.volumes, (0.5 * windows.edges) ** 3, rtol=1e-15)
    assert [len(t) for t in windows.tensors] == [64, 8, 8]  # 3: 2 x 2 x 2, the rest left out
    scale = np.abs(windows.box).max()
    for edge, t in zip(windows.edges, windows.tensors, strict=True):
        assert t.shape[1:] == (d, d)
        np.testing.assert_array_equal(t, np.swapaxes(t, 1, 2))
        assert np.linalg.eigvalsh(t).min() > 0
        if 8 % edge == 0:  # tiling the box
            np.testing.assert_allclose(t.mean(axis=0), windows.box, rtol=0, atol=1e-14 * scale)
    np.testing.assert_allclose(windows.box, windows.apparent, rtol=0, atol=1e-10 * scale)
    if prop == "stiffness":
        apparent, _ = stiffness_tensor(FFTElasticProblem(image, MATERIALS, tol=1e-12))
        assert windows.notation == NOTATION
    else:
        apparent, _ = conductivity_tensor(FFTThermalProblem(image, MATERIALS, tol=1e-12))
        assert "K_ij" in windows.notation
    np.testing.assert_allclose(windows.apparent, apparent, rtol=1e-13, atol=1e-13 * scale)
    # the statistics of the windows: those of their tensors taken as realizations
    options = rve.StudyOptions(bootstrap=0)
    t = windows.statistics(options)
    assert t.method == "windows" and t.key == f"{prop}.periodic" and t.samples == "windows"
    np.testing.assert_array_equal(t.n, [64, 8, 8])
    counts = [len(x) for x in windows.tensors]
    ensemble = rve.tensor_statistics(np.repeat(windows.volumes, counts),
                                     np.concatenate(windows.tensors), prop, "periodic",
                                     options)  # fmt: skip
    for name in ("mean", "std", "compliance", "compliance_std", "mean_compliance"):
        np.testing.assert_array_equal(getattr(t, name), getattr(ensemble, name))
    assert t.components.keys() == ensemble.components.keys()
    for name, a in t.components.items():
        other = ensemble.components[name]
        np.testing.assert_array_equal(a.stats.variance, other.stats.variance)
        assert len(a.rve) == len(other.rve) and t.significant(name) == ensemble.significant(name)
        volumes = [[r["volume"] for r in x.rve] for x in (a, other)]
        np.testing.assert_array_equal(*volumes)
    np.testing.assert_allclose(t.compliance[-1] @ t.mean[-1], np.eye(d), atol=1e-12)
    definitions = t.to_dict()["definitions"]
    assert "windows" in definitions["stiffness"] and "mutual" in definitions["stiffness"]


def _laminate(tensors, fractions, inner, outer):
    """The concentration ``A_p`` of each layer of a laminate (``eps_p = A_p E``): the
    components ``inner`` of the strain (gradient) continuous across the layers, those
    ``outer`` of the stress (flux) too, the mean strain ``E``."""
    d = len(tensors[0])
    layers = list(zip(tensors, fractions, strict=True))
    h = sum(f * np.linalg.inv(c[np.ix_(outer, outer)]) for c, f in layers)
    out = [np.zeros((d, d)) for _ in tensors]
    for k in range(d):
        e = np.eye(d)[k]
        coupled = sum(
            f * np.linalg.solve(c[np.ix_(outer, outer)], c[np.ix_(outer, inner)] @ e[inner])
            for c, f in layers
        )
        traction = np.linalg.solve(h, e[outer] + coupled)
        for a, c in zip(out, tensors, strict=True):
            a[inner, k] = e[inner]
            local = traction - c[np.ix_(outer, inner)] @ e[inner]
            a[outer, k] = np.linalg.solve(c[np.ix_(outer, outer)], local)
    return out


@pytest.mark.parametrize("prop", ["stiffness", "conductivity_tensor"])
def test_window_tensors_of_a_laminate(prop):
    """A laminate of layers normal to z: the strains are uniform in each layer, so that the
    tensor of a window is f_1 A_1^T C_1 A_1 + f_0 A_0^T C_0 A_0 (f_p: the fraction of the
    layers p in the window); its mean the Backus stiffness, whose shear moduli across the
    layers are harmonic means (C44 of G23, C55 of G31) and along them arithmetic (C66 of
    G12): exchanged slots would not pass."""
    layers = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    data = np.broadcast_to(layers, (4, 4, 12)).astype(np.uint8)
    image = mg.VoxelImage(data, spacing=0.5, periodic=True)
    if prop == "stiffness":
        inner, outer = [0, 1, 5], [2, 3, 4]  # 11 22 12 continuous; the tractions 33 23 31
    else:
        inner, outer = [0, 1], [2]
    c = TENSORS[prop]
    f1 = layers.mean()
    a = _laminate(c, (1.0 - f1, f1), inner, outer)
    energies = [x.T @ m @ x for x, m in zip(a, c, strict=True)]  # per layer: A_p^T C_p A_p
    backus = (1.0 - f1) * c[0] @ a[0] + f1 * c[1] @ a[1]
    windows = rve.window_tensors(image, MATERIALS, [2, 3, 4], property=prop, tol=1e-12)
    scale = np.abs(backus).max()
    np.testing.assert_allclose(windows.apparent, backus, rtol=0, atol=1e-9 * scale)
    np.testing.assert_allclose(windows.box, backus, rtol=0, atol=1e-9 * scale)
    for edge, tensors in zip(windows.edges, windows.tensors, strict=True):
        counts = (4 // edge, 4 // edge, 12 // edge)
        assert len(tensors) == np.prod(counts)
        for m, t in enumerate(tensors):
            k = m % counts[2]  # the windows along z: their layers
            f = layers[k * edge : (k + 1) * edge].mean()
            expected = (1.0 - f) * energies[0] + f * energies[1]
            np.testing.assert_allclose(t, expected, rtol=0, atol=1e-9 * scale)
    if prop == "stiffness":
        g = np.array([[7.0, 11.0, 17.0], [40.0, 25.0, 60.0]])  # G23, G31, G12 of each phase
        fractions = np.array([1.0 - f1, f1])
        harmonic = 1.0 / (fractions @ (1.0 / g))
        np.testing.assert_allclose(np.diag(backus)[3:5], harmonic[:2], rtol=1e-12)
        assert backus[5, 5] == pytest.approx(fractions @ g[:, 2], rel=1e-12)
        t = windows.statistics(rve.StudyOptions(bootstrap=0))
        np.testing.assert_allclose(np.diag(t.compliance[-1])[3:], 1.0 / np.diag(backus)[3:],
                                   rtol=1e-8)  # fmt: skip
        # the windows of 2 voxels: C44 = C~44^2 <1/G23>_w, C55 = C~55^2 <1/G31>_w (the
        # tractions continuous) and C66 = <G12>_w (the strain), f the fraction of phase 1
        for m, w in enumerate(windows.tensors[0]):
            f = layers[2 * (m % 6) : 2 * (m % 6) + 2].mean()
            inverse = (1.0 - f) / g[0] + f / g[1]
            np.testing.assert_allclose(np.diag(w)[3:5], harmonic[:2] ** 2 * inverse[:2],
                                       rtol=1e-9)  # fmt: skip
            assert w[5, 5] == pytest.approx((1.0 - f) * g[0, 2] + f * g[1, 2], rel=1e-9)
    else:  # the conductivity across the layers, a harmonic mean; along them, arithmetic
        k_across = 1.0 / ((1.0 - f1) / 3.0 + f1 / 20.0)
        assert backus[2, 2] == pytest.approx(k_across, rel=1e-12)
        assert backus[0, 0] == pytest.approx((1.0 - f1) * 1.0 + f1 * 10.0, rel=1e-12)
        assert backus[1, 1] == pytest.approx((1.0 - f1) * 2.0 + f1 * 30.0, rel=1e-12)


def _by_tiling(windows, k):
    """The tensors of the windows of size ``k``, ``(tilings, n, d, d)``."""
    t = windows.tensors[k]
    return t.reshape(len(windows.offsets[k]), -1, *t.shape[1:])


@pytest.mark.parametrize("prop", ["stiffness", "conductivity_tensor"])
def test_shifted_windows_of_a_laminate(prop):
    """Tilings shifted around a periodic laminate normal to z (offsets k w // 2 along each
    axis, all their combinations): every window of every tiling is f_1 A_1^T C_1 A_1 + f_0
    A_0^T C_0 A_0, f_p the fraction of the layers p in it, wrapping around the image; their
    statistics per size the mean of all the windows, the root of their variance pooled
    within the tilings, the count of the windows of one tiling."""
    layers = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    data = np.broadcast_to(layers, (4, 4, 12)).astype(np.uint8)
    image = mg.VoxelImage(data, spacing=0.5, periodic=True)
    inner, outer = ([0, 1, 5], [2, 3, 4]) if prop == "stiffness" else ([0, 1], [2])
    c = TENSORS[prop]
    f1 = layers.mean()
    a = _laminate(c, (1.0 - f1, f1), inner, outer)
    energies = [x.T @ m @ x for x, m in zip(a, c, strict=True)]
    windows = rve.window_tensors(image, MATERIALS, [2, 3, 4], property=prop, tol=1e-12, shifts=2)
    np.testing.assert_array_equal(windows.tilings, [8, 8, 8])
    steps = {2: (0, 1), 3: (0, 1), 4: (0, 2)}
    scale = np.abs(windows.box).max()
    for k, edge in enumerate(windows.edges):
        offsets = windows.offsets[k]
        np.testing.assert_array_equal(offsets, list(itertools.product(*[steps[edge]] * 3)))
        counts = (4 // edge, 4 // edge, 12 // edge)
        tensors = _by_tiling(windows, k)
        assert tensors.shape[:2] == (8, np.prod(counts))
        for offset, tiling in zip(offsets, tensors, strict=True):
            for m, t in enumerate(tiling):
                z = offset[2] + (m % counts[2]) * edge + np.arange(edge)  # its layers
                f = layers[z % 12].mean()
                expected = (1.0 - f) * energies[0] + f * energies[1]
                np.testing.assert_allclose(t, expected, rtol=0, atol=1e-9 * scale)
    t = windows.statistics(rve.StudyOptions(bootstrap=0))
    np.testing.assert_array_equal(t.tilings, [8, 8, 8])
    np.testing.assert_array_equal(t.n, [24, 4, 3])  # the windows of one tiling
    for k in range(3):
        np.testing.assert_array_equal(t.mean[k], windows.tensors[k].mean(axis=0))
        tilings = _by_tiling(windows, k)
        squares = ((tilings - tilings.mean(axis=1, keepdims=True)) ** 2).sum(axis=(0, 1))
        pooled = squares / (8 * (t.n[k] - 1))
        np.testing.assert_allclose(t.std[k] ** 2, pooled, rtol=1e-12, atol=1e-30 * scale**2)
    stats = t.components["C33"].stats
    np.testing.assert_array_equal(stats.n, [24, 4, 3])
    np.testing.assert_array_equal(stats.tilings, [8, 8, 8])


def test_shifted_windows_of_a_random_image(monkeypatch):
    """Tilings shifted around a random periodic image take the solutions of the unshifted
    windows (no new solve: one per unit load); the first is the unshifted one, to the bit;
    each tiling of the box has the box's tensor for mean (round-off); the variance of each
    component pooled within the tilings agrees with that of one tiling within its error
    (relative sqrt(2 / (n - 1))); the rows and the dictionary count the tilings."""
    solves = []
    for cls in (FFTElasticProblem, FFTThermalProblem):
        monkeypatch.setattr(cls, "_field", _counted(cls._field, solves))
    image = _random_image((8, 8, 8), seed=2)
    shifted = rve.window_tensors(image, MATERIALS, [2, 4], tol=1e-12, shifts=2)
    assert len(solves) == 6  # the six unit strains, once each
    windows = rve.window_tensors(image, MATERIALS, [2, 4], tol=1e-12)
    assert len(solves) == 12 and windows.offsets is None
    np.testing.assert_array_equal(windows.tilings, [1, 1])
    scale = np.abs(windows.box).max()
    for k in range(len(windows.edges)):  # 2 and 4 voxels: the tilings cover the box
        tilings = _by_tiling(shifted, k)
        np.testing.assert_array_equal(tilings[0], windows.tensors[k])  # offset 0: unchanged
        assert np.linalg.eigvalsh(tilings).min() > 0
        box = np.broadcast_to(windows.box, tilings[:, 0].shape)
        np.testing.assert_allclose(tilings.mean(axis=1), box, rtol=0, atol=1e-14 * scale)
    options = rve.StudyOptions(bootstrap=0)
    one, t = windows.statistics(options), shifted.statistics(options)
    assert one.tilings is None and "tilings" not in one.volume_rows()[0]
    np.testing.assert_array_equal(t.n, one.n)  # 64 and 8 windows in each tiling
    error = np.sqrt(2.0 / (one.n[0] - 1))
    diagonal = np.diagonal(t.std[0] ** 2 / one.std[0] ** 2)
    assert np.all(np.abs(diagonal - 1.0) < 3.0 * error)
    row = t.volume_rows()[0]
    assert row["n"] == 64 and row["tilings"] == 8 and list(row)[:5] == [
        "bc", "volume", "edge", "n", "tilings"]  # fmt: skip
    assert [v["tilings"] for v in t.to_dict()["volumes"]] == [8, 8]
    with pytest.raises(ValueError, match="periodic"):
        rve.window_tensors(mg.VoxelImage(image.data, spacing=0.5), MATERIALS, [2], shifts=2)


def _counted(solve, calls):
    """The solve of a problem, counted in ``calls``."""

    def counted(self, *args, **kwargs):
        calls.append(type(self).__name__)
        return solve(self, *args, **kwargs)

    return counted


def test_planar_windows_and_pores():
    """Across a planar image the windows are squares, their measure their area, and the
    plane stiffness (Voigt 11 22 12) is the plane part of the stiffness of the image; a
    window within a pore has no stiffness, and no compliance: NaN in the mean of the
    compliances of its size, not in the inverse of the mean."""
    image = _random_image((16, 16, 1), seed=3)
    windows = rve.window_tensors(image, MATERIALS, [4, 8], tol=1e-12)
    assert [t.shape for t in windows.tensors] == [(16, 3, 3), (4, 3, 3)]
    np.testing.assert_allclose(windows.volumes, [4.0, 16.0], rtol=1e-15)  # (0.5 w)^2
    assert windows.notation == PLANE_NOTATION
    full, _ = stiffness_tensor(FFTElasticProblem(image, MATERIALS, tol=1e-12))
    plane = full[np.ix_([0, 1, 5], [0, 1, 5])]
    scale = np.abs(plane).max()
    np.testing.assert_allclose(windows.apparent, plane, rtol=0, atol=1e-13 * scale)
    np.testing.assert_allclose(windows.box, plane, rtol=0, atol=1e-10 * scale)
    for t in windows.tensors:
        np.testing.assert_allclose(t.mean(axis=0), windows.box, rtol=0, atol=1e-14 * scale)
    fraction = rve.window_statistics(image, [4, 8])
    np.testing.assert_allclose(fraction.volume, windows.volumes, rtol=1e-15)
    np.testing.assert_array_equal(fraction.n, [16, 4])
    # pores: phase 0 a void, in which the windows of one voxel have no stiffness
    porous = {0: VOID, 1: MATERIALS[1]}
    windows = rve.window_tensors(_random_image((6, 6, 6), seed=4, fraction=0.7), porous, [1, 3])
    empty = ~windows.tensors[0].any(axis=(1, 2))
    assert empty.any() and not empty.all()
    t = windows.statistics(rve.StudyOptions(bootstrap=0))
    assert np.isnan(t.mean_compliance[0]).all() and np.isfinite(t.compliance).all()
    assert np.count_nonzero(~empty) >= t.components["S11"].stats.n[0] > 0
    assert t.n[0] == 216 and np.isfinite(t.mean_compliance[1]).all()


def _config(tmp_path, name="windows", **rve_section):
    return {
        "model": {"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.3},
        "domain": {"periodic": True},
        "seed": 5,
        "voxels": {"voxel_size": 0.5},
        "homogenization": {
            "method": "fft",
            "materials": {0: {"stiffness": C0.tolist(), "conductivity": 1.0},
                          1: {"stiffness": C1.tolist(), "conductivity": 5.0}},
            "properties": ["stiffness", "conductivity_tensor"],
            "tolerance": 1e-10,
        },
        "rve": {"method": "windows", "size": 4.0, "window_sizes": [1, 2], **rve_section},
        "outputs": {"directory": str(tmp_path / name), "name": "spheres"},
    }  # fmt: skip


def test_window_study_of_tensors_in_the_pipeline(tmp_path):
    """A window study of the stiffness and the conductivity: results.csv has one row per
    window size with the mean tensors, their inverses and standard deviations; the mean of
    the windows tiling the realization is its apparent tensor (run.json); the report has the
    tables of each tensor per window size, rve_<tensor>.csv and the figures."""
    config = load_config(_config(tmp_path))
    assert config.rve.properties == [
        "volume_fraction",
        "stiffness.periodic",
        "conductivity_tensor.periodic",
    ]
    outdir = pipeline.run(config)
    rows = pipeline.read_rows(outdir / "results.csv")
    assert [r["n"] for r in rows] == [64, 8] and [r["window_edge"] for r in rows] == [1.0, 2.0]
    assert 0 < rows[0]["mean"] < 1 and "tilings" not in rows[0]  # one tiling by default
    run = json.loads((outdir / "run.json").read_text(encoding="utf-8"))
    assert run["stiffness_notation"] == NOTATION
    for key, d in (("stiffness.periodic", 6), ("conductivity_tensor.periodic", 3)):
        apparent = np.array(run["apparent"][key])
        mean = np.zeros((d, d))
        for i in range(d):
            for j in range(i, d):
                mean[i, j] = mean[j, i] = rows[-1][f"{key}.C{i + 1}{j + 1}"]
        np.testing.assert_allclose(mean, apparent, rtol=0, atol=1e-8 * np.abs(apparent).max())
        s = np.array([[rows[-1][f"{key}.S{min(i, j) + 1}{max(i, j) + 1}"] for j in range(d)]
                      for i in range(d)])  # fmt: skip
        np.testing.assert_allclose(s @ mean, np.eye(d), atol=1e-10)
        assert rows[0][f"{key}.C11_std"] > rows[1][f"{key}.C11_std"] > 0
    # the shears in their slots: G23 < G31 < G12 in the softer phase, the matrix
    c = np.array([[rows[-1][f"stiffness.periodic.C{min(i, j) + 1}{max(i, j) + 1}"]
                   for j in range(6)] for i in range(6)])  # fmt: skip
    assert c[3, 3] < c[4, 4] < c[5, 5]
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    assert report["method"] == "windows" and "volume_fraction" in report["properties"]
    stiffness = report["tensors"]["stiffness.periodic"]
    assert stiffness["method"] == "windows" and stiffness["notation"] == NOTATION
    assert "mutual energy" in stiffness["definitions"]["stiffness"]
    assert [v["n"] for v in stiffness["volumes"]] == [64, 8]
    assert "resistivity" in report["tensors"]["conductivity_tensor.periodic"]["notation"]
    files = {p.name for p in outdir.iterdir()}
    assert {"rve_stiffness.csv", "rve_conductivity_tensor.csv", "rve_report.tex"} <= files
    md = (outdir / "rve_report.md").read_text(encoding="utf-8")
    assert md.startswith("# RVE report (windows method)") and "## Windows per volume" in md
    assert "## Mean stiffness Cbar_ij +/- 2 D (PBC; Voigt 11 22 33 23 31 12" in md
    assert "| windows | 64 | 8 |" in md and "eps_k : C : eps_l" in md
    assert "## Mean conductivity Cbar_ij +/- 2 D (PBC; axes 1 2 3)" in md
    assert "## Resistivity S_ij = Cbar^-1 +/- 2 D (PBC; axes 1 2 3)" in md
    tex = (outdir / "rve_report.tex").read_text(encoding="utf-8")
    assert tex.count("{") == tex.count("}")
    # the stiffness alone: no volume fraction, the rows of the windows and their tensors
    config = load_config(_config(tmp_path, "alone", properties=["stiffness"]))
    assert config.rve.properties == ["stiffness.periodic"]
    outdir = pipeline.run(config)
    rows = pipeline.read_rows(outdir / "results.csv")
    assert "mean" not in rows[0] and "stiffness.periodic.C66" in rows[0]
    assert "conductivity_tensor.periodic.C11" not in rows[0]
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    assert report["properties"] == {} and list(report["tensors"]) == ["stiffness.periodic"]


def test_window_study_with_shifted_tilings(tmp_path, monkeypatch):
    """rve.shifts: the tilings shifted around the periodic realization (8 per size, offsets
    k w // 2 along each axis), from the same solutions: results.csv counts them (tilings,
    after n, the windows of one), its volume fraction that of window_statistics with the
    shifts on the same realization, its mean tensors the apparent ones of the realization;
    the report writes the counts n x T; refused for the ensemble method, a domain that is
    not periodic, or a count that is not a positive integer."""
    monkeypatch.setattr(rve_report, "plot", lambda report, outdir: [])  # no figures: faster
    config = load_config(_config(tmp_path, "shifted", shifts=2, properties=["volume_fraction",
                                 "stiffness"]))  # fmt: skip
    assert config.rve.shifts == 2
    outdir = pipeline.run(config)
    rows = pipeline.read_rows(outdir / "results.csv")
    assert list(rows[0])[:4] == ["window_edge", "volume", "n", "tilings"]
    assert [r["n"] for r in rows] == [64, 8] and [r["tilings"] for r in rows] == [8, 8]
    domain = Domain(size=(4.0, 4.0, 4.0), origin=config.domain.origin, periodic=True)
    image = config.model.generate(domain, derive_seed(5, 0)).voxelize(voxel_size=0.5)
    stats = rve.window_statistics(image, [2, 4], shifts=2)
    assert [r["mean"] for r in rows] == pytest.approx(stats.mean.tolist(), rel=1e-12)
    assert [r["variance"] for r in rows] == pytest.approx(stats.variance.tolist(), rel=1e-12)
    run = json.loads((outdir / "run.json").read_text(encoding="utf-8"))
    apparent = np.array(run["apparent"]["stiffness.periodic"])
    mean = np.array([[rows[-1][f"stiffness.periodic.C{min(i, j) + 1}{max(i, j) + 1}"]
                      for j in range(6)] for i in range(6)])  # fmt: skip
    np.testing.assert_allclose(mean, apparent, rtol=0, atol=1e-8 * np.abs(apparent).max())
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    stiffness = report["tensors"]["stiffness.periodic"]
    assert [(v["n"], v["tilings"]) for v in stiffness["volumes"]] == [(64, 8), (8, 8)]
    assert [r["tilings"] for r in report["properties"]["volume_fraction"]["statistics"]] == [8, 8]
    md = (outdir / "rve_report.md").read_text(encoding="utf-8")
    assert "| volume_fraction | 64 x 8 | 8 x 8 |" in md and "| windows | 64 x 8 | 8 x 8 |" in md
    assert "n x T: the n windows of each of T tilings" in md
    assert [r["tilings"] for r in pipeline.read_rows(outdir / "rve_stiffness.csv")] == [8, 8]
    data = _config(tmp_path, shifts=2)
    del data["homogenization"]
    data["domain"]["periodic"] = False
    cases = [
        (_config(tmp_path, shifts=0), "rve.shifts: expected a positive integer"),
        (_config(tmp_path, shifts=True), "rve.shifts: expected a positive integer"),
        (_config(tmp_path, method="ensemble", sizes=[2.0], realizations=2, shifts=2),
         "shifted tilings of the window method"),
        (data, "wrap around the realization"),
    ]  # fmt: skip
    for case, message in cases:
        with pytest.raises(ConfigError, match=message):
            load_config(case)


def test_planar_edges_in_the_report(tmp_path):
    """The windows of a planar image are squares: the report prints their areas as L^2 and
    their edges, and those of the RVE sizes, as the roots of their areas (a cube's V^(1/3)),
    in the tables, rve_report.json, rve_stiffness.csv and the RVE tool of the interface."""
    from morphogen.gui.tools import statistics as tool

    image = _random_image((32, 32, 1), seed=5)
    options = rve.StudyOptions(bootstrap=0)
    stats = rve.window_statistics(image, [2, 4, 8, 16])
    analysis = rve.analyze_statistics(stats, "volume_fraction", options)
    t = rve.window_tensors(image, MATERIALS, [2, 4, 8], tol=1e-10).statistics(options)
    assert stats.dimension == t.dimension == 2
    report = rve.StudyReport([analysis], {}, options, method="windows", tensors=[t])
    assert report.dimension == 2
    rve_report.write(report, tmp_path, figures=False)
    md = (tmp_path / "rve_report.md").read_text(encoding="utf-8")
    assert "| series | 1^2 | 2^2 | 4^2 | 8^2 |" in md  # the areas (0.5 w)^2
    assert "| component | 1^2 | 2^2 | 4^2 |" in md
    assert "^3" not in md and "the side of the square window" in md
    data = json.loads((tmp_path / "rve_report.json").read_text(encoding="utf-8"))
    fraction = data["properties"]["volume_fraction"]
    assert fraction["dimension"] == 2 and fraction["rve"]
    for row in fraction["statistics"] + fraction["rve"]:
        assert row["edge"] == pytest.approx(math.sqrt(row["volume"]), rel=1e-15)
    stiffness = data["tensors"]["stiffness.periodic"]
    assert stiffness["dimension"] == 2
    assert [v["edge"] for v in stiffness["volumes"]] == pytest.approx([1.0, 2.0, 4.0])
    for a in stiffness["components"].values():
        for row in a["rve"]:
            assert row["edge"] == pytest.approx(math.sqrt(row["volume"]), rel=1e-15)
    rows = pipeline.read_rows(tmp_path / "rve_stiffness.csv")
    assert [r["edge"] for r in rows] == pytest.approx([1.0, 2.0, 4.0])
    law = rve.ScalingLaw(K=0.01, gamma=1.0)
    for dimension, power in ((2, 0.5), (3, 1.0 / 3.0)):
        for row in rve.rve_table(law, 0.4, dimension=dimension):
            assert row["edge"] == pytest.approx(row["volume"] ** power, rel=1e-15)

    class Context:  # what the tool's tables need of its job
        folder, titles = tmp_path, []

        def table(self, rows, name, title):
            self.titles.append(title)

    context = Context()
    tool._mean_tensors(context)
    assert context.titles[0].startswith("stiffness C_ij at 4², C the mean of 16 windows (PBC)")


def test_window_studies_refused(tmp_path):
    """The window method computes the tensors of its windows by FFT: scalar apparent
    properties, loadings, paths, fields and finite elements are refused."""
    cases = [
        ({"properties": ["bulk_modulus"]}, {}, "window method computes the tensors"),
        ({"loadings": {"pull": {"strain": {"11": 0.01}}}}, {}, "not loadings or paths"),
        ({"fields": True}, {}, "writes no fields"),
        ({"method": "fe"}, {}, "window method solves its realization by the FFT"),
        ({}, {"properties": ["specific_surface"]}, "window method measures volume_fraction"),
    ]
    for homogenization, rve_section, message in cases:
        data = _config(tmp_path, **rve_section)
        data["homogenization"].update(homogenization)
        if homogenization.get("method") == "fe":
            data["mesh"] = {"size": 0.5}
        with pytest.raises(ConfigError, match=message):
            load_config(data)
    with pytest.raises(ValueError, match="property"):
        rve.window_tensors(_random_image((4, 4, 4), 0), MATERIALS, [2], property="shear_modulus")
    with pytest.raises(ValueError, match="does not fit"):
        rve.window_tensors(_random_image((4, 4, 4), 0), MATERIALS, [5])


def _tool_workspace(root, periodic=True):
    """A session of the interface whose workspace holds a Boolean model of spheres (its
    realization periodic or not)."""
    from morphogen.gui import api
    from morphogen.models.base import model_from_config

    session = api.Session(root)
    model = model_from_config({"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.3})
    micro = model.generate(Domain(size=4.0, periodic=periodic), derive_seed(7, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "boolean_spheres (seed 7)", micro=micro, seed=7,
                model=model.to_config(), domain={"size": 4.0, "periodic": periodic},
                phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    return session, key


def _tool_job(session, params):
    """The RVE tool run at once (in this thread): the job (its folder) and its status."""
    from morphogen.gui.tools import ToolJob

    job = ToolJob({"tool": "rve", "params": params}, session)
    job._run()
    return job, job.status()


def _tool_realization(folder):
    """The configuration of a study of the RVE tool by windows (its run.json) and the voxels
    of its realization, generated again as the pipeline does."""
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    config = load_config(run["config"])
    size = config.rve.size
    domain = Domain(size=(size, size, size), origin=config.domain.origin,
                    periodic=config.domain.periodic)  # fmt: skip
    micro = config.model.generate(domain, derive_seed(config.seed, 0))
    return config, micro.voxelize(**config.voxels.kwargs())


TOOL_WINDOWS = {"method": "windows", "size": 4.0, "window_sizes": [1.0, 2.0], "voxel_size": 0.5}


def test_rve_tool_by_windows(tmp_path):
    """The RVE tool of the interface by the window method: the stiffness of the windows of
    one realization, the mean C of the largest windows and its compliance (S C = I)."""
    session, key = _tool_workspace(tmp_path)
    params = {"source": key, **TOOL_WINDOWS, "property": "stiffness"}
    _, status = _tool_job(session, params)
    assert status["state"] == "done", status.get("error")
    stiffness, compliance = status["result"]["tables"][:2]
    assert stiffness["title"].startswith("stiffness C_ij at 2³, C the mean of 8 windows")
    c = np.array([row[1:] for row in stiffness["rows"]], dtype=float)
    s = np.array([row[1:] for row in compliance["rows"]], dtype=float)
    np.testing.assert_allclose(s @ c, np.eye(6), atol=1e-4)  # rounded to 6 digits
    titles = [t["title"] for t in status["result"]["tables"]]
    assert any(t.startswith("Mean stiffness Cbar_ij +/- 2 D (PBC") for t in titles)
    _, status = _tool_job(session, {**params, "property": "shear_modulus"})
    assert "window method" in str(status.get("error"))


def test_rve_tool_with_shifted_tilings(tmp_path):
    """The shifts of the RVE tool by windows (the field shown for the window method, 1 by
    default): with 2, its volume fraction and its mean stiffness those of window_statistics
    and window_tensors with the shifts on its realization (the 8 tilings of each size), the
    mean C of the largest windows titled with its tilings."""
    from morphogen.gui import tools

    fields = {f["name"]: f for f in tools.TOOLS["rve"].to_dict()["fields"]}
    assert fields["shifts"]["default"] == 1 and fields["shifts"]["when"] == {"method": ["windows"]}
    session, key = _tool_workspace(tmp_path)
    assert tools.TOOLS["rve"].parse({"source": key}, session)["shifts"] is None  # ensemble
    params = {"source": key, **TOOL_WINDOWS, "property": "stiffness", "shifts": "2"}
    job, status = _tool_job(session, params)
    assert status["state"] == "done", status.get("error")
    config, image = _tool_realization(job.folder)
    assert config.rve.shifts == 2
    rows = pipeline.read_rows(job.folder / "results.csv")
    assert [(r["n"], r["tilings"]) for r in rows] == [(64, 8), (8, 8)]
    stats = rve.window_statistics(image, [2, 4], shifts=2)
    assert [r["mean"] for r in rows] == stats.mean.tolist()
    assert [r["variance"] for r in rows] == stats.variance.tolist()
    homog = config.homogenization
    windows = rve.window_tensors(image, homog.materials, [2, 4], tol=homog.tolerance,
                                 maxiter=homog.max_iterations, device=homog.device,
                                 shifts=2)  # fmt: skip
    t = windows.statistics(rve.StudyOptions(bootstrap=0))
    np.testing.assert_array_equal(t.tilings, [8, 8])
    for k, row in enumerate(rows):
        mean = [[row[f"stiffness.periodic.C{min(i, j) + 1}{max(i, j) + 1}"] for j in range(6)]
                for i in range(6)]  # fmt: skip
        np.testing.assert_allclose(mean, t.mean[k], rtol=1e-12, atol=0)
    title = status["result"]["tables"][0]["title"]
    assert title.startswith("stiffness C_ij at 2³, C the mean of 8 windows x 8 tilings (PBC)")


def test_rve_tool_shifts_by_default_and_refused(tmp_path):
    """The shifts of the RVE tool: by default (or 1) the study of before, one tiling from
    the origin (the volume fraction of window_statistics without shifts to the bit, no
    tilings, the configuration without rve.shifts, the outputs equal byte for byte); more
    refused with the messages of the API, on a morphology that is not periodic (whose
    windows without shifts are still studied) and for a count that is not positive."""
    session, key = _tool_workspace(tmp_path / "periodic")
    params = {"source": key, **TOOL_WINDOWS}
    folders = []
    for shifts in ({}, {"shifts": 1}):
        job, status = _tool_job(session, {**params, **shifts})
        assert status["state"] == "done", status.get("error")
        folders.append(job.folder)
    config, image = _tool_realization(folders[0])
    assert "shifts" not in config.source["rve"] and config.rve.shifts == 1
    stats = rve.window_statistics(image, [2, 4])
    rows = pipeline.read_rows(folders[0] / "results.csv")
    assert list(rows[0])[:4] == ["window_edge", "volume", "n", "mean"]  # no tilings
    assert [r["n"] for r in rows] == stats.n.tolist() == [64, 8]
    assert [r["mean"] for r in rows] == stats.mean.tolist()
    assert [r["variance"] for r in rows] == stats.variance.tolist()
    for name in ("results.csv", "rve_report.json", "rve_report.md"):
        assert (folders[1] / name).read_bytes() == (folders[0] / name).read_bytes()
    for shifts in (0, -2):  # the count, as the API refuses it
        with pytest.raises(ValueError) as refused:
            rve.window_statistics(image, [2], shifts=shifts)
        _, status = _tool_job(session, {**params, "shifts": shifts})
        assert status["state"] == "error" and status["error"] == f"ValueError: {refused.value}"
    with pytest.raises(ValueError) as refused:  # the API on an image that is not periodic
        rve.window_statistics(mg.VoxelImage(image.data, spacing=0.5), [2], shifts=2)
    session, key = _tool_workspace(tmp_path / "open", periodic=False)
    params = {"source": key, **TOOL_WINDOWS}
    job, status = _tool_job(session, {**params, "shifts": 2, "property": "stiffness"})
    assert status["state"] == "error" and status["error"] == f"ValueError: {refused.value}"
    assert "periodic" in status["error"] and not (job.folder / "results.csv").exists()
    job, status = _tool_job(session, params)
    assert status["state"] == "done", status.get("error")
    assert "tilings" not in pipeline.read_rows(job.folder / "results.csv")[0]


def test_planar_axes_in_the_figures(tmp_path, monkeypatch):
    """The figures of the windows of a planar image draw them against their areas: their
    axes say area V where those of cubes say volume V (the variance, the mean, the tensors),
    the RVE sizes V_RVE^(1/2) and V_RVE^(1/3); the figures the RVE tool shows."""
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    labels = {}

    def record(fig, path, **kwargs):  # the axes of each figure, nothing written
        labels[path.name] = [(ax.get_xlabel(), ax.get_ylabel()) for ax in fig.axes]

    monkeypatch.setattr(Figure, "savefig", record)
    options = rve.StudyOptions(bootstrap=0)
    for shape, sizes, measure, root in (((32, 32, 1), [2, 4, 8, 16], "area V", 2),
                                        ((8, 8, 8), [1, 2, 4], "volume V", 3)):  # fmt: skip
        image = _random_image(shape, seed=5)
        stats = rve.window_statistics(image, sizes)
        analysis = rve.analyze_statistics(stats, "volume_fraction", options)
        t = rve.window_tensors(image, MATERIALS, sizes[:3], tol=1e-8).statistics(options)
        report = rve.StudyReport([analysis], {}, options, method="windows", tensors=[t])
        assert report.dimension == root
        labels.clear()
        rve_report.plot(report, tmp_path)
        assert labels["rve_volume_fraction.png"] == [(measure, "D^2(V) of volume_fraction")]
        assert labels["rve_volume_fraction_mean.png"] == [
            (measure, "volume_fraction: mean +/- 2 D")
        ]
        assert labels["rve_volume_fraction_size.png"] == [
            ("relative error eps_rel (%)", f"RVE edge V_RVE^(1/{root})")
        ]
        tensor = [x for x, _ in labels["rve_stiffness_periodic.png"]]
        assert tensor == ["", "", f"window {measure}", f"window {measure}"]
        if root == 2:  # nothing of a volume on the axes of the squares
            assert not any("volume" in x for axes in labels.values() for x, _ in axes)
