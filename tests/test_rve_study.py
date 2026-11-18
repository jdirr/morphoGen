"""Statistical RVE analysis following Dirrenberger et al. (2014)."""

import json
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen import rve
from morphogen.rve import report as rve_report

# Dirrenberger et al. (2014): Table 4 means at V = 100^3 and Table 10 (gamma = 2/3,
# eps = 5 %, n = 1): (K, Zbar, edge of V_RVE)
TABLE_10 = {
    "k KUBC": (5.0e8, 4763.0, 188),
    "k MBC": (3.5e8, 2527.0, 296),
    "mu KUBC": (4.0e8, 3270.0, 245),
    "mu MBC": (1.8e8, 1097.0, 489),
    "lambda UTG": (1.3e3, 7.07, 204),
    "lambda MTBC": (1.3e3, 6.07, 238),
}


def synthetic_rows(rng, specs, edges=(20, 30, 40, 50, 60, 80, 100), n=40):
    """Gaussian realizations with variance K V^-gamma (one row per realization)."""
    rows = []
    for (prop, bc), (mean, k, gamma) in specs.items():
        for edge in edges:
            v = float(edge) ** 3
            sd = math.sqrt(k * v**-gamma)
            for r in range(n):
                rows.append(
                    {
                        "property": prop,
                        "bc": bc,
                        "size": edge,
                        "volume": v,
                        "realization": r,
                        "value": mean + sd * rng.standard_normal(),
                    }
                )
    return rows


def test_table10_rve_sizes_and_fibre_counts():
    fibres = mg.models.PoissonFibers(radius=1.0, volume_fraction=0.16)
    for name, (k, mean, edge) in TABLE_10.items():
        vol = rve.ScalingLaw(K=k, gamma=2 / 3).rve_volume(0.05, mean)
        assert vol ** (1 / 3) == pytest.approx(edge, rel=0.01), name
        # number of fibres in V_RVE (eq. 9); Table 10 lists 2.8e3 ... 1.9e4
        count = fibres.expected_count(mg.Domain.cube(vol ** (1 / 3)))
        assert count == pytest.approx(0.0832 * vol ** (2 / 3), rel=1e-3)
    # n realizations of V reach the precision of one realization of V_RVE when
    # n = (V_RVE / V)^gamma: 14 for k (KUBC) at V = 50^3 (the paper's text says 18)
    k, mean, edge = TABLE_10["k KUBC"]
    law = rve.ScalingLaw(K=k, gamma=2 / 3)
    n = law.realizations(50.0**3, 0.05, mean)
    assert n == pytest.approx((law.rve_volume(0.05, mean) / 50.0**3) ** (2 / 3))
    assert n == pytest.approx(14.1, abs=0.1)


def test_discrepancy_between_boundary_conditions_eq69():
    rows = []
    for bc, k, mu in (("kubc", 4763.0, 3270.0), ("mixed", 2527.0, 1097.0)):
        for r in range(3):
            rows.append(
                {
                    "volume": 1e6,
                    "realization": r,
                    f"bulk_modulus.{bc}": k + r - 1,
                    f"shear_modulus.{bc}": mu + r - 1,
                }
            )
    series = rve.series_from_rows(rows)
    study = rve.analyze_study(series, rve.StudyOptions(bootstrap=0))
    d = study.discrepancy
    assert d["bulk_modulus"][0]["discrepancy"] == pytest.approx(0.88, abs=0.01)  # 88 %
    assert d["shear_modulus"][0]["discrepancy"] == pytest.approx(1.98, abs=0.01)  # 198 %


def test_relative_error_and_intervals():
    vals = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
    series = rve.Series("conductivity", "kubc", np.full(5, 8.0), vals)
    a = rve.analyze_series(series, rve.StudyOptions(bootstrap=0))
    row = a.statistics_rows()[0]
    d = np.std(vals, ddof=1)
    assert row["std"] == pytest.approx(d)
    assert row["relative_error"] == pytest.approx(2 * d / (vals.mean() * math.sqrt(5)))
    assert row["low"] == pytest.approx(vals.mean() - 2 * d)
    assert a.free_fit is None and a.rve == []  # one size: no fit


def test_analysis_recovers_the_scaling_law():
    rng = np.random.default_rng(1)
    specs = {
        ("bulk_modulus", "kubc"): (4763.0, 5.0e8, 2 / 3),
        ("bulk_modulus", "mixed"): (2527.0, 3.5e8, 0.8),
    }
    rows = synthetic_rows(rng, specs, n=60)
    series = rve.series_from_rows(rows)
    assert sorted(s.key for s in series) == ["bulk_modulus.kubc", "bulk_modulus.mixed"]
    free = rve.analyze_study(series, rve.StudyOptions(bootstrap=300, seed=2))
    for a in free.analyses:
        true_gamma = specs[(a.series.property, a.series.bc)][2]
        lo, hi = a.free_fit.gamma_interval()
        assert lo < true_gamma < hi, a.key
        blo, bhi = a.bootstrap["gamma"]
        assert blo < true_gamma < bhi, a.key
    imposed = rve.analyze_study(
        series,
        rve.StudyOptions(
            gamma=2 / 3,
            relative_errors=(0.05,),
            n_realizations=(1,),
            min_volume=40**3,
            bootstrap=300,
        ),
    )
    a = imposed["bulk_modulus.kubc"]
    assert a.fit.gamma_fixed and a.fit.K == pytest.approx(5.0e8, rel=0.3)
    assert a.fit.n_points == 5  # sizes >= 40^3
    row = a.rve[0]
    assert row["edge"] == pytest.approx(188, rel=0.1)
    assert row["edge_ci"][0] < row["edge"] < row["edge_ci"][1]
    needed = {r["volume"]: r["eps=0.05"] for r in a.needed}
    assert needed[100.0**3] < needed[20.0**3]


def test_isotropy_of_the_mean_response():
    rng = np.random.default_rng(3)
    n = 20
    stress = np.repeat(np.eye(3)[None], n, axis=0) * 9000.0
    stress += rng.normal(0.0, 50.0, size=(n, 3, 3))
    stress = 0.5 * (stress + stress.transpose(0, 2, 1))
    rows = []
    for i in range(n):
        row = {
            "property": "bulk_modulus",
            "bc": "kubc",
            "volume": 1000.0,
            "realization": i,
            "value": np.trace(stress[i]) / 9,
        }
        names = ("11", "22", "33", "23", "31", "12")
        pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
        for name, (a, b) in zip(names, pairs, strict=True):
            row[f"R{name}"] = stress[i, a, b]
        rows.append(row)
    a = rve.analyze_study(rve.series_from_rows(rows), rve.StudyOptions(bootstrap=0)).analyses[0]
    mean = stress.mean(axis=0)
    from morphogen.homogenization.apparent import bulk_isotropy

    assert a.isotropy[0] == pytest.approx(bulk_isotropy(mean))
    assert a.isotropy[0] < 0.01


def test_wide_tables_group_by_and_errors():
    rows = [{"size": 2.0, "bc": bc, "k_app": v} for bc, v in (("a", 1.0), ("b", 2.0))] * 3
    series = rve.series_from_rows(rows, group_by="bc")
    assert sorted(s.key for s in series) == ["k_app.a", "k_app.b"]
    assert series[0].volume[0] == 8.0
    with pytest.raises(ValueError, match="unknown result columns"):
        rve.series_from_rows(rows, properties=["nope"])
    with pytest.raises(ValueError, match="volume"):
        rve.series_from_rows([{"k_app": 1.0}])
    with pytest.raises(ValueError):
        rve.StudyOptions(weights="sqrt")
    with pytest.raises(ValueError):
        rve.StudyOptions(relative_errors=(0.0,))


def test_report_files(tmp_path):
    rng = np.random.default_rng(4)
    specs = {
        ("shear_modulus", "kubc"): (3270.0, 4.0e8, 2 / 3),
        ("shear_modulus", "mixed"): (1097.0, 1.8e8, 2 / 3),
    }
    series = rve.series_from_rows(synthetic_rows(rng, specs, edges=(10, 20, 40), n=10))
    study = rve.analyze_study(
        series,
        rve.StudyOptions(gamma=2 / 3, bootstrap=50),
        count=lambda v: 0.0832 * v ** (2 / 3),
        bounds={
            "shear_modulus": {"voigt": 12923.0, "reuss": 0.0, "hs_lower": 0.0, "hs_upper": 7328.0}
        },
    )
    files = rve_report.write(study, tmp_path)
    names = {f.name for f in files}
    assert {"rve_report.json", "rve_report.md", "rve_report.tex"} <= names
    md = (tmp_path / "rve_report.md").read_text(encoding="utf-8")
    for title in (
        "Realizations per volume",
        "Mean values",
        "Bounds",
        "Discrepancy",
        "Relative error",
        "Scaling of the variance",
        "RVE sizes",
        "Realizations needed",
    ):
        assert f"## {title}" in md
    assert "| shear_modulus (KUBC) |" in md and "20^3" in md
    tex = (tmp_path / "rve_report.tex").read_text(encoding="utf-8")
    assert tex.count("\\begin{table}") == tex.count("\\end{table}") >= 8
    assert tex.count("{") == tex.count("}")
    data = json.loads((tmp_path / "rve_report.json").read_text(encoding="utf-8"))
    assert data["options"]["gamma"] == pytest.approx(2 / 3)
    assert data["properties"]["shear_modulus.kubc"]["rve"][0]["count"] > 0
    if pytest.importorskip("matplotlib"):
        assert (tmp_path / "rve_shear_modulus_mean.png").exists()


def test_quantities_of_loading_paths():
    """A quantity of the curves of a path at its end or at the end of a step, and its
    checks against the configuration."""
    from morphogen.config import ConfigError, load_config
    from morphogen.homogenization import Material, NonlinearFFTProblem, Plasticity
    from morphogen.homogenization.paths import PathQuantity

    m = Material(young=200e3, poisson=0.3, plasticity=Plasticity(300.0, hardening=2000.0))
    image = mg.image.VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    path = {"times": [0, 1, 2], "increments": [2, 2], "strain": {"11": [0, 0.001, 0.01]}}
    res = NonlinearFFTProblem(image, {0: m}).run(path)
    assert res.value("S11", 1.0) == pytest.approx(200.0, rel=1e-9)  # elastic at 0.1 %
    assert res.value("S11") == res.stress[-1, 0] and res.value("p_0") == res.plastic[-1]
    with pytest.raises(ValueError, match="no step ends"):
        res.value("S11", 0.75)
    with pytest.raises(ValueError, match="unknown quantity"):
        res.value("X11")
    q = PathQuantity.parse("tension.S11@1e0.pbc")
    assert (q.name, q.key) == ("tension.S11@1", "tension.S11@1.periodic")
    assert q.of(res) == res.value("S11", 1.0)
    with pytest.raises(ValueError, match="a time after @"):
        PathQuantity.parse("tension.S11@one")
    base = {
        "model": {"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.2},
        "domain": {"periodic": True},
        "voxels": {"voxel_size": 0.25},
        "homogenization": {
            "method": "fft",
            "materials": {0: {"young": 7e4, "poisson": 0.33, "plasticity": {"yield": 100}},
                          1: {"young": 4e5, "poisson": 0.2}},
            "paths": {"tension": path},
        },
        "rve": {"sizes": [2, 3], "realizations": [3, 2]},
    }  # fmt: skip
    data = json.loads(json.dumps(base))
    data["rve"]["properties"] = ["tension.S11@1", "tension.p_0", "tension.S11.pbc"]
    cfg = load_config(data)
    assert cfg.rve.properties == ["tension.S11@1", "tension.p_0", "tension.S11.periodic"]
    for prop, match in [
        ("compression.S11", "unknown value"),
        ("tension.X11", "unknown quantity"),
        ("tension.TS11", "unknown quantity"),  # true stresses: finite strains only
        ("tension.S11@0.75", "no step"),  # the steps end at 0.5, 1, 1.5 and 2
        ("tension.S11.kubc", "solved under"),
    ]:
        data = json.loads(json.dumps(base))
        data["rve"]["properties"] = [prop]
        with pytest.raises(ConfigError, match=match):
            load_config(data)


@pytest.mark.slow
def test_rve_study_of_loading_paths(tmp_path):
    """An ensemble study of a plastic composite on voxels: the stresses of a path at the
    end of a step and at its end are analysed like the apparent moduli, in the report of
    the run and by morphogen fit."""
    from morphogen import cli, pipeline
    from morphogen.config import load_config

    cfg = {
        "model": {"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.25},
        "domain": {"periodic": True},
        "seed": 3,
        "voxels": {"voxel_size": 0.25},
        "homogenization": {
            "method": "fft",
            "materials": {
                0: {"young": 7e4, "poisson": 0.33, "plasticity": {"yield": 100, "hardening": 500}},
                1: {"young": 4e5, "poisson": 0.2},
            },
            "properties": ["bulk_modulus"],
            "paths": {"tension": {"times": [0, 1, 2], "increments": [1, 2],
                                  "strain": {"11": [0, 0.001, 0.004]}}},
        },
        "rve": {"sizes": [2, 3], "realizations": [4, 3], "bootstrap": 0,
                "properties": ["bulk_modulus", "tension.S11@1", "tension.S11"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "spheres"},
    }  # fmt: skip
    outdir = pipeline.run(load_config(cfg))
    rows = pipeline.read_rows(outdir / "homogenization.csv")
    at_yield = [r for r in rows if r["property"] == "tension.S11@1"]
    assert len(at_yield) == 7 and {r["bc"] for r in at_yield} == {"periodic"}
    curve = pipeline.read_rows(outdir / "spheres_L2_0000_tension_periodic.csv")
    first = next(r for r in at_yield if r["size_index"] == 0 and r["realization"] == 0)
    assert first["value"] == pytest.approx(curve[1]["S11"], rel=1e-12)  # the end of step 1
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    keys = set(report["properties"])
    assert keys == {"bulk_modulus.periodic", "tension.S11@1.periodic", "tension.S11.periodic"}
    assert (outdir / "rve_tension.S11@1_mean.png").exists() or not _has_matplotlib()
    assert cli.main(["fit", str(outdir), "--property", "tension.S11", "--bootstrap", "0",
                     "-o", str(tmp_path / "fit")]) == 0  # fmt: skip
    fitted = json.loads((tmp_path / "fit" / "rve_report.json").read_text(encoding="utf-8"))
    assert list(fitted["properties"]) == ["tension.S11.periodic"]
    # every series of the files, the summaries of the paths left out
    assert cli.main(["fit", str(outdir), "--bootstrap", "0"]) == 0


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _orthotropic_stiffness() -> np.ndarray:
    """An orthotropic stiffness whose three shear moduli differ (G23 = 7, G31 = 11, G12 =
    17), as its Young's moduli and Poisson's ratios, so that an exchange of components
    shows (Voigt 11 22 33 23 31 12, engineering shears)."""
    young = (120.0, 80.0, 60.0)
    s = np.diag([*(1.0 / e for e in young), 1.0 / 7.0, 1.0 / 11.0, 1.0 / 17.0])
    for (i, j), nu in (((0, 1), 0.3), ((0, 2), 0.25), ((1, 2), 0.2)):
        s[i, j] = s[j, i] = -nu / young[i]
    return np.linalg.inv(s)


def _tensors_of_rows(rows, prefix: str = "stiffness.periodic.C") -> np.ndarray:
    """The stiffness of each row of results from its columns C11 ... C66."""
    out = np.zeros((len(rows), 6, 6))
    for k, row in enumerate(rows):
        for i in range(6):
            for j in range(i, 6):
                out[k, i, j] = out[k, j, i] = row[f"{prefix}{i + 1}{j + 1}"]
    return out


def test_mean_stiffness_and_compliance_per_volume(tmp_path):
    """Per volume, the mean stiffness is the average of the tensors of the realizations, its
    compliance S = Cbar^-1 (S Cbar = I, the mean of the compliances apart), the standard
    deviations those of each component; the shear components keep their slots (Voigt 11 22
    33 23 31 12) in the tables, the CSV and the JSON; the components that do not vanish on
    average have RVE sizes, the couplings that do have none."""
    from morphogen import pipeline
    from morphogen.config import RVESpec
    from morphogen.homogenization.postprocess import NOTATION, matrix_entries

    c0 = _orthotropic_stiffness()
    rng = np.random.default_rng(3)
    rows = []
    for edge, n in ((10, 8), (20, 6), (40, 5)):
        for r in range(n):
            noise = rng.standard_normal((6, 6))
            c = c0 + 0.3 * (edge / 10.0) ** -1.5 * (noise + noise.T)  # D^2 ~ 1 / V
            row = {"size": float(edge), "volume": float(edge) ** 3, "realization": r}
            entries = matrix_entries(c, "C").items()
            rows.append(row | {f"stiffness.periodic.{k}": v for k, v in entries})
    (t,) = rve.tensors_from_rows(rows, options=rve.StudyOptions(bootstrap=0))
    assert t.key == "stiffness.periodic" and t.notation == NOTATION
    np.testing.assert_array_equal(t.n, [8, 6, 5])
    tensors, volumes = _tensors_of_rows(rows), np.array([r["volume"] for r in rows])
    for k, v in enumerate(t.volume):
        mine = tensors[volumes == v]
        np.testing.assert_allclose(t.mean[k], mine.mean(axis=0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(t.std[k], mine.std(axis=0, ddof=1), rtol=1e-10)
        np.testing.assert_allclose(t.compliance[k] @ t.mean[k], np.eye(6), atol=1e-12)
        inverses = np.linalg.inv(mine)
        np.testing.assert_allclose(t.mean_compliance[k], inverses.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(t.compliance_std[k], inverses.std(axis=0, ddof=1), rtol=1e-8)
    # the shears in their slots: C44 = G23, C55 = G31, C66 = G12 and S44 = 1 / G23 ...
    for slot, g in ((3, 7.0), (4, 11.0), (5, 17.0)):
        assert t.mean[-1][slot, slot] == pytest.approx(g, rel=0.01)
        assert t.compliance[-1][slot, slot] == pytest.approx(1.0 / g, rel=0.01)
    sized = {name for name, a in t.components.items() if a.rve}
    orthotropic = {f"{x}{ij}" for x in "CS" for ij in ("11", "22", "33", "12", "13", "23")}
    assert orthotropic | {"C44", "C55", "C66", "S44", "S55", "S66"} <= sized
    assert not sized & {"C14", "C15", "C16", "C24", "S14", "S56"}
    assert t.components["C11"].fit.gamma == pytest.approx(1.0, abs=0.3)
    # the report of a study of the stiffness alone: its tables, CSV and JSON
    report = pipeline.build_report(rows, RVESpec(bootstrap=0), ["stiffness"])
    assert report.analyses == [] and [x.key for x in report.tensors] == ["stiffness.periodic"]
    files = {f.name for f in rve_report.write(report, tmp_path)}
    assert {"rve_report.json", "rve_report.md", "rve_stiffness.csv"} <= files
    table = pipeline.read_rows(tmp_path / "rve_stiffness.csv")
    assert [r["n"] for r in table] == [8, 6, 5] and table[0]["bc"] == "periodic"
    assert table[-1]["C44"] == pytest.approx(t.mean[-1][3, 3], rel=1e-12)
    assert table[-1]["S66"] == pytest.approx(t.compliance[-1][5, 5], rel=1e-12)
    assert table[-1]["C55_std"] == pytest.approx(t.std[-1][4, 4], rel=1e-12)
    data = json.loads((tmp_path / "rve_report.json").read_text(encoding="utf-8"))
    data = data["tensors"]["stiffness.periodic"]
    assert data["notation"] == NOTATION
    assert "inverse of the mean" in data["definitions"]["compliance"]
    top = data["volumes"][-1]
    s_c = np.array(top["compliance"]) @ np.array(top["stiffness"])
    np.testing.assert_allclose(s_c, np.eye(6), atol=1e-12)
    assert top["stiffness"][3][3] == pytest.approx(t.mean[-1][3, 3], rel=1e-12)
    assert data["components"]["C44"]["rve"] and not data["components"]["C14"]["rve"]
    md = (tmp_path / "rve_report.md").read_text(encoding="utf-8")
    assert "## Mean stiffness Cbar_ij +/- 2 D (PBC; Voigt 11 22 33 23 31 12" in md
    assert "## Compliance S_ij = Cbar^-1 +/- 2 D (PBC" in md
    assert "## RVE sizes per component (PBC; fitted gamma)" in md and "| C44 | 7.0" in md
    tex = (tmp_path / "rve_report.tex").read_text(encoding="utf-8")
    assert "Cbar$^{-1}$" in tex and "^-" not in tex and tex.count("{") == tex.count("}")
    if _has_matplotlib():
        assert (tmp_path / "rve_stiffness_periodic.png").exists()
    # every series asked: the tensors join; other properties named: they do not
    assert pipeline.build_report(rows, RVESpec(bootstrap=0)).tensors
    others = [r | {"k": 1.0 + 0.01 * i} for i, r in enumerate(rows)]
    assert not pipeline.build_report(others, RVESpec(bootstrap=0), ["k"]).tensors
    with pytest.raises(ValueError, match="no stiffness"):
        pipeline.build_report([{"volume": 8.0, "k": 1.0}], RVESpec(), ["stiffness"])
    assert rve.is_tensor("stiffness") and rve.is_tensor("stiffness.kubc")
    assert not rve.is_tensor("stiffness.kubc.C44") and not rve.is_tensor("bulk_modulus")


def test_couplings_zero_to_round_off():
    """Exactly symmetric realizations (laminates: their couplings 1e-17 of the diagonal, of
    one sign, the round-off of the solver) have no coupling distinguished from zero, in C
    or in S, whatever its t statistic; the diagonal and the orthotropic components keep
    their RVE sizes."""
    from morphogen.homogenization.postprocess import matrix_entries

    c0 = _orthotropic_stiffness()
    rng = np.random.default_rng(5)
    coupled = np.ones((6, 6), dtype=bool)
    coupled[:3, :3] = False
    coupled[[3, 4, 5], [3, 4, 5]] = False
    rows = []
    for edge, n in ((10, 6), (20, 5), (40, 4)):
        for r in range(n):
            noise = rng.standard_normal((6, 6))
            c = c0 + 0.3 * (edge / 10.0) ** -1.5 * (noise + noise.T) * ~coupled
            off = 1e-17 * c0[0, 0] * (4.0 + 0.5 * rng.random((6, 6)))  # one sign
            c = c + np.where(coupled, off + off.T, 0.0)
            row = {"size": float(edge), "volume": float(edge) ** 3, "realization": r}
            entries = matrix_entries(c, "C").items()
            rows.append(row | {f"stiffness.periodic.{k}": v for k, v in entries})
    (t,) = rve.tensors_from_rows(rows, options=rve.StudyOptions(bootstrap=0))
    couplings = [f"{x}{i + 1}{j + 1}" for x in "CS" for i in range(6) for j in range(i, 6)
                 if coupled[i, j]]  # fmt: skip
    assert len(couplings) == 24
    assert not any(t.significant(name) for name in couplings)
    assert not any(t.components[name].rve for name in couplings)
    sized = {name for name, a in t.components.items() if a.rve}
    assert {"C11", "C12", "C44", "C66", "S11", "S23", "S55"} <= sized


def test_rve_study_of_the_stiffness(tmp_path, capsys):
    """An ensemble study of the apparent stiffness of two orthotropic phases on voxels (FFT):
    per volume, the mean C of rve_report.json and rve_stiffness.csv is the average of the
    tensors of the realizations in results.csv, S Cbar = I, the shear moduli in their slots;
    the configuration names it 'stiffness', morphogen fit analyses it again."""
    from morphogen import cli, pipeline
    from morphogen.config import load_config

    c0 = _orthotropic_stiffness()
    cfg = {
        "model": {"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.3},
        "domain": {"periodic": True},
        "seed": 5,
        "voxels": {"voxel_size": 0.5},
        "homogenization": {
            "method": "fft",
            "materials": {0: {"stiffness": c0.tolist()}, 1: {"stiffness": (3 * c0).tolist()}},
            "properties": ["stiffness"],
        },
        "rve": {"sizes": [2, 3], "realizations": [3, 2], "bootstrap": 0,
                "properties": ["stiffness"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "ortho"},
    }  # fmt: skip
    config = load_config(cfg)
    assert config.rve.properties == ["stiffness.periodic"]
    outdir = pipeline.run(config)
    rows = pipeline.read_rows(outdir / "results.csv")
    tensors, volumes = _tensors_of_rows(rows), np.array([r["volume"] for r in rows])
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    assert report["properties"] == {}  # the tensors alone were asked
    study = report["tensors"]["stiffness.periodic"]
    table = pipeline.read_rows(outdir / "rve_stiffness.csv")
    assert [v["n"] for v in study["volumes"]] == [3, 2]
    for entry, row in zip(study["volumes"], table, strict=True):
        mean = tensors[volumes == entry["volume"]].mean(axis=0)
        s = np.array(entry["compliance"])
        np.testing.assert_allclose(entry["stiffness"], mean, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(s @ mean, np.eye(6), atol=1e-9)
        assert row["C44"] == pytest.approx(mean[3, 3], rel=1e-10)
        assert row["S44"] == pytest.approx(s[3, 3], rel=1e-10)
        # G23 < G31 < G12 in both phases: so in the mean, in the slots 44, 55, 66
        assert mean[3, 3] < mean[4, 4] < mean[5, 5] and s[3, 3] > s[4, 4] > s[5, 5]
        assert mean[3, 3] / mean[4, 4] == pytest.approx(7.0 / 11.0, rel=0.1)
        assert mean[5, 5] / mean[4, 4] == pytest.approx(17.0 / 11.0, rel=0.1)
    md = (outdir / "rve_report.md").read_text(encoding="utf-8")
    assert "## Mean stiffness Cbar_ij +/- 2 D (PBC" in md and "| realizations | 3 | 2 |" in md
    assert (outdir / "rve_stiffness_periodic.png").exists() or not _has_matplotlib()
    assert cli.main(["fit", str(outdir), "--property", "stiffness", "--bootstrap", "0"]) == 0
    assert "## Compliance S_ij = Cbar^-1 +/- 2 D (PBC" in capsys.readouterr().out
