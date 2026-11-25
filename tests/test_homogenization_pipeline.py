"""Homogenization in configuration files, pipeline runs and the command line."""

import json

import numpy as np
import pytest
import yaml

import morphogen as mg
from morphogen import cli, pipeline
from morphogen.config import ConfigError, load_config

STEEL = {"young": 210000.0, "poisson": 0.3, "conductivity": 100.0}


needs_mmg = pytest.mark.skipif(
    not mg.mesh.conforming.has_mmg(), reason="MMG (pymmg) is not installed"
)


def fibre_study(tmp_path, **rve):
    return {
        "model": {"type": "poisson_fibers", "radius": 1.0, "volume_fraction": 0.16},
        "domain": {"periodic": False},
        "seed": 7,
        "mesh": {"size": 0.6},
        "homogenization": {
            "materials": {0: "void", 1: STEEL},
            "boundary_conditions": ["kubc", "mbc"],
            "properties": ["bulk_modulus", "conductivity"],
        },
        "rve": {"sizes": [3, 4], "realizations": [4, 3], "gamma": 2 / 3, "bootstrap": 50, **rve},
        "outputs": {"directory": str(tmp_path / "study"), "name": "fibres"},
    }


def test_homogenization_configuration(tmp_path):
    cfg = load_config(fibre_study(tmp_path), base_dir=tmp_path)
    h = cfg.homogenization
    assert h.method == "fe" and cfg.mesh.size == 0.6 and cfg.voxels is None
    assert cfg.mesh.kwargs(h.porous)["phases"] == "solid"
    assert h.boundary_conditions == ["kubc", "mixed"]
    assert h.series() == [
        "bulk_modulus.kubc",
        "bulk_modulus.mixed",
        "conductivity.kubc",
        "conductivity.mixed",
    ]
    assert cfg.rve.properties == ["volume_fraction", *h.series()]
    data = fibre_study(tmp_path, properties=["bulk_modulus", "conductivity.kubc"])
    assert load_config(data).rve.properties == [
        "bulk_modulus.kubc",
        "bulk_modulus.mixed",
        "conductivity.kubc",
    ]
    base = fibre_study(tmp_path)
    cases = [
        ({"materials": None}, "materials"),
        ({"boundary_conditions": ["free"]}, "unknown boundary condition"),
        ({"boundary_conditions": ["subc"]}, "porous"),
        ({"properties": ["toughness"]}, "unknown value"),
        ({"method": "voxel"}, "method"),
        ({"method": "fft"}, "FFT solver is periodic"),
        ({"void": "remove"}, "unknown keys"),
        ({"solver": "magic"}, "solver"),
        ({"materials": {1: STEEL}}, "no material for phase"),
        ({"materials": {0: "void", 1: {"young": 1.0}}}, "poisson"),
    ]
    for change, message in cases:
        data = json.loads(json.dumps(base))
        data["homogenization"].update(change)
        if change.get("materials", 0) is None:
            del data["homogenization"]["materials"]
        with pytest.raises(ConfigError, match=message):
            load_config(data)
    data = json.loads(json.dumps(base))
    data["rve"]["properties"] = ["shear_modulus"]
    with pytest.raises(ConfigError, match=r"rve\.properties"):
        load_config(data)
    data["rve"] = {"method": "windows", "size": 5.0, "window_sizes": [1, 2]}
    with pytest.raises(ConfigError, match="window method"):
        load_config(data)
    # meshes and methods
    data = json.loads(json.dumps(base))
    del data["mesh"]
    data["homogenization"]["method"] = "fe"
    with pytest.raises(ConfigError, match="'mesh' section"):
        load_config(data)
    data["homogenization"]["method"] = "auto"
    if mg.mesh.conforming.has_mmg():  # no mesh: finite elements on a light conforming mesh
        cfg = load_config(data)
        edge = float(np.min(cfg.domain.lengths))
        assert cfg.homogenization.method == "fe" and cfg.mesh.size == pytest.approx(edge / 10)
        assert cfg.mesh.hausdorff == pytest.approx(edge / 80) and cfg.mesh.background is None
    data["homogenization"]["method"] = "fft"
    data["homogenization"]["boundary_conditions"] = ["periodic"]
    with pytest.raises(ConfigError, match="voxels"):
        load_config(data)  # FFT needs voxel images
    data["voxels"] = {"voxel_size": 0.5}
    assert load_config(data).homogenization.method == "fft"
    data["homogenization"]["method"] = "auto"
    if mg.mesh.conforming.has_mmg():  # the level set sampled at the voxels described
        cfg = load_config(data)
        assert cfg.homogenization.method == "fe" and cfg.mesh.background == 0.5
    data["homogenization"]["properties"] = ["hydraulic_permeability"]  # only on voxels
    data["homogenization"]["materials"] = {0: "void", 1: STEEL}
    assert load_config(data).homogenization.method == "fft"
    for change, message in (
        ({"order": 3}, "order"),
        ({"size": -1}, "positive"),
        ({"phases": "pores"}, "phases"),
        ({"gradation": 0.5}, "gradation"),
        ({"colour": 1}, "unknown keys"),
    ):
        data = json.loads(json.dumps(base))
        data["mesh"].update(change)
        with pytest.raises(ConfigError, match=message):
            load_config(data)
    data = json.loads(json.dumps(base))
    data["model"] = {"type": "voronoi", "intensity": 1.0}
    data["homogenization"]["materials"] = {0: STEEL, 1: STEEL}
    assert load_config(data).mesh.grains is None  # tessellations: meshed by grains
    data["mesh"]["grains"] = False  # the interfaces of the phases of the cells alone
    assert load_config(data).mesh.grains is False
    for change, message in (
        ({"grains": "yes"}, "grains"),
        ({"min_feature": -0.1}, "min_feature"),
    ):
        data["mesh"] = {"size": 0.6, **change}
        with pytest.raises(ConfigError, match=message):
            load_config(data)
    data = json.loads(json.dumps(base))
    data["mesh"]["grains"] = True
    with pytest.raises(ConfigError, match="not tessellations"):
        load_config(data)
    data = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
        "domain": {"size": 2.0},
        "outputs": {"conforming_mesh": ["vtu"]},
    }
    with pytest.raises(ConfigError, match="conforming_mesh"):
        load_config(data)
    data["mesh"] = {"size": 0.5}
    assert load_config(data).mesh.kwargs(porous=False)["phases"] == "all"


def spheres_config(tmp_path, **homogenization):
    return {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
        "domain": {"size": 4.0, "periodic": True},
        "seed": 1,
        "realizations": 2,
        "homogenization": {
            "materials": {
                0: {"young": 1.0, "poisson": 0.3, "conductivity": 1.0},
                1: {"young": 10.0, "poisson": 0.25, "conductivity": 5.0},
            },
            "boundary_conditions": ["periodic"],
            "properties": ["bulk_modulus", "shear_modulus", "conductivity", "stiffness"],
            "fields": True,
            **homogenization,
        },
        "outputs": {"directory": str(tmp_path / "batch"), "name": "spheres"},
    }


def test_batch_run_with_fft(tmp_path):
    cfg = spheres_config(tmp_path, method="fft")
    cfg["voxels"] = {"shape": 12}
    outdir = pipeline.run(load_config(cfg))
    rows = pipeline.read_rows(outdir / "results.csv")
    assert len(rows) == 2
    for row in rows:
        assert row["stiffness.periodic.bulk_modulus"] == pytest.approx(
            row["bulk_modulus.periodic"], rel=1e-6
        )
        assert row["fraction_0"] + row["fraction_1"] == pytest.approx(1.0)
    fe = pipeline.read_rows(outdir / "homogenization.csv")
    assert len(fe) == 2 * 3
    assert {r["solver"] for r in fe} == {"fft"} and all(r["residual"] < 1e-8 for r in fe)
    images = mg.io.read_vti(outdir / "spheres_0001_bulk_modulus_periodic.vti")
    names = [im.name for im in images]
    assert "energy_density" in names and "von_mises" in names and "stress_12" in names
    energy = images[names.index("energy_density")].data
    row = next(r for r in fe if r["realization"] == 1 and r["property"] == "bulk_modulus")
    assert energy.mean() / 9.0 == pytest.approx(row["energy_value"], rel=1e-6)


def test_planar_section_of_a_volume(tmp_path):
    """``voxels.section``: a run studies one layer of voxels through the middle of its
    realizations (a planar study of a three-dimensional model): the image, the level set
    and the measures are those of the layer of the volume, and the FFT solver homogenizes
    the layer as it would the image given to it."""
    from morphogen.homogenization import homogenize
    from morphogen.homogenization.materials import materials_from_config
    from morphogen.seeding import derive_seed

    volume = mg.VoxelImage(np.arange(4 * 5 * 6).reshape(4, 5, 6), (0.5, 1.0, 2.0), (1.0, 2.0, 3.0),
                           periodic=True)  # fmt: skip
    middle = volume.layer()
    assert middle.shape == (4, 5, 1) and middle.periodic and middle.spacing == volume.spacing
    np.testing.assert_array_equal(middle.data[:, :, 0], volume.data[:, :, 3])
    assert middle.origin == (1.0, 2.0, 3.0 + 3 * 2.0)
    first = volume.layer(0, 0)
    assert first.shape == (1, 5, 6) and first.origin == volume.origin
    np.testing.assert_array_equal(first.data[0], volume.data[0])
    for axis, index in ((3, None), (1, 5), (1, -1)):
        with pytest.raises(ValueError, match="layer"):
            volume.layer(axis, index)
    materials = {0: {"young": 1.0, "poisson": 0.3}, 1: {"young": 10.0, "poisson": 0.2}}
    data = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
        "domain": {"size": 6.0, "periodic": True},
        "seed": 5,
        "voxels": {"shape": 16, "section": "y"},
        "homogenization": {"method": "fft", "materials": materials, "properties": ["stiffness"]},
        "analysis": ["volume_fraction"],
        "outputs": {"directory": str(tmp_path / "section"), "name": "cut", "image": ["vti"],
                    "level_set": ["vti"]},
    }  # fmt: skip
    cfg = load_config(data)
    assert cfg.voxels.section == 1
    outdir = pipeline.run(cfg)
    micro = cfg.model.generate(cfg.domain, derive_seed(5, 0))
    layer = micro.voxelize(shape=16).layer(1)
    written = mg.io.load_image(outdir / "cut_0000.vti")
    assert written.shape == (16, 1, 16)
    np.testing.assert_array_equal(written.data, layer.data)
    level = mg.io.load_image(outdir / "cut_0000_levelset.vti")
    assert level.shape == (16, 1, 16) and ((level.data < 0) == (layer.data == 1)).all()
    (row,) = pipeline.read_rows(outdir / "results.csv")
    assert row["volume_fraction"] == pytest.approx(float(np.mean(layer.data == 1)))
    direct = homogenize(layer, materials_from_config(materials), properties=["stiffness"])
    stiffness = direct["stiffness.periodic"]
    for key in ("E1", "E2", "E3", "G12"):
        assert row[f"stiffness.periodic.{key}"] == pytest.approx(stiffness[key], rel=1e-9)
    # a plane strain along y: stiffer along the fibres of the section than across them
    assert row["stiffness.periodic.E2"] > row["stiffness.periodic.E1"]
    for change, match in (
        ({"voxels": {"shape": 16, "section": "w"}}, "'x', 'y' or 'z'"),
        ({"homogenization": {**data["homogenization"], "method": "xfem"}}, "extended finite"),
    ):
        with pytest.raises(ConfigError, match=match):
            load_config({**data, **change})
    windows = {
        "model": data["model"],
        "domain": {"periodic": True},
        "voxels": {"voxel_size": 0.25, "section": "z"},
        "rve": {"method": "windows", "size": 8.0, "window_sizes": [1, 2]},
        "outputs": {"directory": str(tmp_path / "windows"), "name": "cut"},
    }
    with pytest.raises(ConfigError, match="window method"):
        load_config(windows)


@pytest.mark.slow
@needs_mmg
def test_batch_run_with_finite_elements(tmp_path):
    cfg = spheres_config(tmp_path, boundary_conditions=["kubc", "periodic"])
    cfg["mesh"] = {"size": 0.8, "order": 1}
    cfg["outputs"]["conforming_mesh"] = ["vtu", "inp"]
    cfg["outputs"]["periodic_pairs"] = True
    outdir = pipeline.run(load_config(cfg))
    rows = pipeline.read_rows(outdir / "results.csv")
    assert len(rows) == 2
    for row in rows:
        assert row["bulk_modulus.periodic"] <= row["bulk_modulus.kubc"]
        assert row["stiffness.periodic.bulk_modulus"] == pytest.approx(
            row["bulk_modulus.periodic"], rel=1e-6
        )
        assert row["fraction_0"] + row["fraction_1"] == pytest.approx(1.0)
        assert row["mesh_elements"] > 0 and row["mesh_quality_min"] > 0
    fe = pipeline.read_rows(outdir / "homogenization.csv")
    assert len(fe) == 2 * 3 * 2
    mesh = mg.io.read_tet_mesh(outdir / "spheres_0000_tet.vtu")
    assert mesh.n_elements == rows[0]["mesh_elements"]
    assert (outdir / "spheres_0000_tet.inp").exists()
    assert (outdir / "spheres_0000_tet_periodic.csv").exists()
    fields = mg.io.read_vtu(outdir / "spheres_0001_bulk_modulus_kubc.vtu")
    assert fields["point_data"]["displacement"].shape == (len(fields["points"]), 3)
    energy = fields["cell_data"]["energy_density"]
    vol = mg.mesh.TetMesh(
        fields["points"], fields["cells"], fields["cell_data"]["phase"], (0, 0, 0), (4, 4, 4)
    ).volumes()
    row = next(
        r
        for r in fe
        if r["realization"] == 1 and r["property"] == "bulk_modulus" and r["bc"] == "kubc"
    )
    assert (energy * vol).sum() / 64.0 / 9.0 == pytest.approx(row["energy_value"], rel=1e-5)
    assert (
        "temperature" in mg.io.read_vtu(outdir / "spheres_0000_conductivity_kubc.vtu")["point_data"]
    )


@pytest.mark.slow
@needs_mmg
def test_fe_rve_study_and_split_runs(tmp_path):
    data = fibre_study(tmp_path)
    outdir = pipeline.run(load_config(data))
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    props = report["properties"]
    assert {
        "volume_fraction",
        "bulk_modulus.kubc",
        "bulk_modulus.mixed",
        "conductivity.kubc",
        "conductivity.mixed",
    } <= set(props)
    kubc = props["bulk_modulus.kubc"]
    assert kubc["fit"]["gamma_fixed"] and kubc["rve"][0]["count"] > 0  # number of fibres
    assert "isotropy" in kubc["statistics"][0]
    d = report["discrepancy"]["bulk_modulus"]
    assert len(d) == 2 and all(r["discrepancy"] > 0 for r in d)
    assert report["bounds"]["bulk_modulus"]["hs_upper"] > 0
    for size in (0, 1):
        means = {
            bc: props[f"bulk_modulus.{bc}"]["statistics"][size]["mean"] for bc in ("kubc", "mixed")
        }
        assert means["mixed"] < means["kubc"]  # Dirrenberger 2014: MBC below KUBC
    md = (outdir / "rve_report.md").read_text(encoding="utf-8")
    assert "## Discrepancy between boundary conditions" in md
    assert "## Statistical isotropy" in md and "fibres" not in md.split("## Bounds")[0][:20]
    assert (outdir / "rve_report.tex").exists()
    # the same study split in three tasks, then merged by "morphogen fit"
    data["outputs"]["directory"] = str(tmp_path / "split")
    for task in range(3):
        pipeline.run(load_config(data), task_id=task, n_tasks=3)
    split = tmp_path / "split"
    assert len(list(split.glob("homogenization_task*of3.csv"))) == 3
    assert (
        cli.main(["fit", str(split), "--gamma", "0.6667", "--bootstrap", "0", "-o", str(split)])
        == 0
    )
    merged = json.loads((split / "rve_report.json").read_text(encoding="utf-8"))["properties"]
    for key in ("bulk_modulus.kubc", "conductivity.mixed"):
        a = [r["mean"] for r in props[key]["statistics"]]
        b = [r["mean"] for r in merged[key]["statistics"]]
        assert a == pytest.approx(b, rel=1e-12)


def test_cli_homogenize_image(tmp_path, capsys):
    rng = np.random.default_rng(0)
    image = mg.VoxelImage((rng.random((6, 6, 6)) < 0.5).astype(np.uint8), periodic=True)
    path = tmp_path / "img.vti"
    mg.io.save_image(image, path)
    args = [
        "homogenize",
        str(path),
        "--material",
        "0=young:1,poisson:0.3,conductivity:1",
        "1=young:10,poisson:0.3,conductivity:3",
        "--property",
        "bulk_modulus",
        "conductivity",
        "stiffness",
    ]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "FFT" in out and "periodic" in out and "bounds of bulk_modulus" in out
    assert (
        cli.main(
            [*args, "--json", "-o", str(tmp_path / "h.json"), "--fields", str(tmp_path / "fields")]
        )
        == 0
    )
    res = json.loads(capsys.readouterr().out)
    assert len(res["results"]) == 2 and "stiffness.periodic" in res
    assert (
        json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))["results"] == res["results"]
    )
    assert (tmp_path / "fields" / "img_bulk_modulus_periodic.vti").exists()
    assert cli.main([*args, "--bc", "kubc"]) == 2
    assert "conforming mesh" in capsys.readouterr().err
    assert cli.main(["homogenize", str(path), "--bc", "pbc"]) == 2
    assert "materials" in capsys.readouterr().err
    assert cli.main(["homogenize", str(path), "--material", "1=young"]) == 2


@needs_mmg
def test_cli_homogenize_geometry_and_mesh(tmp_path, capsys):
    micro = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3).generate(
        mg.Domain.cube(3.0), seed=2
    )
    geometry = tmp_path / "spheres.json"
    mg.io.save_microstructure(micro, geometry)
    materials = tmp_path / "mat.yaml"
    materials.write_text(yaml.safe_dump({"materials": {0: "void", 1: STEEL}}))
    args = ["homogenize", str(geometry), "--materials", str(materials), "--bc", "kubc", "mixed"]
    # no --size: finite elements all the same, on a light conforming mesh
    assert cli.main([*args, "--order", "1", "--json"]) == 0
    light = json.loads(capsys.readouterr().out)
    assert light["order"] == 1 and light["elements"] > 0
    assert {r["bc"] for r in light["results"]} == {"kubc", "mixed"}
    mesh_path = tmp_path / "spheres_tet.vtu"
    assert cli.main([*args, "--size", "0.8", "--save-mesh", str(mesh_path), "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert {r["bc"] for r in res["results"]} == {"kubc", "mixed"}
    assert res["elements"] > 0 and res["order"] == 2  # 10-node elements by default
    mesh = mg.io.read_tet_mesh(mesh_path)
    assert set(np.unique(mesh.phase)) == {1}  # the pores are not meshed
    assert mesh.order == 2 and mesh.curved  # their walls are curved
    assert cli.main([*args, "--size", "0.8", "--order", "1", "--json"]) == 0
    linear = json.loads(capsys.readouterr().out)
    assert linear["order"] == 1 and linear["elements"] == res["elements"]
    assert {r["bc"] for r in linear["results"]} == {"kubc", "mixed"}
    # the saved mesh gives the same results
    assert (
        cli.main(
            [
                "homogenize",
                str(mesh_path),
                "--materials",
                str(materials),
                "--bc",
                "kubc",
                "mixed",
                "--json",
            ]
        )
        == 0
    )
    again = json.loads(capsys.readouterr().out)
    for a, b in zip(res["results"], again["results"], strict=True):
        assert a["value"] == pytest.approx(b["value"], rel=1e-9)
    assert (
        cli.main(["homogenize", str(mesh_path), "--materials", str(materials), "--bc", "subc"]) == 2
    )
    assert "porous" in capsys.readouterr().err


COPPER = {"cubic": [168.4, 121.4, 75.4], "conductivity": 400.0}
K_COPPER = (168.4 + 2 * 121.4) / 3


def polycrystal_config(tmp_path, **sections):
    return {
        "model": {"type": "voronoi", "n_cells": 12, "crystal_orientations": True},
        "domain": {"size": 1.0, "periodic": True},
        "seed": 3,
        "homogenization": {
            "materials": {1: COPPER},
            "properties": ["bulk_modulus", "shear_modulus"],
        },
        "outputs": {"directory": str(tmp_path / "poly"), "name": "grains"},
        **sections,
    }


def test_polycrystal_runs(tmp_path):
    """Crystals turned by the orientations of the grains: the bulk modulus of a cubic
    polycrystal is (C11 + 2 C12) / 3 exactly, on voxels and on conforming meshes."""
    data = polycrystal_config(tmp_path, voxels={"shape": 12})
    data["homogenization"]["method"] = "fft"
    cfg = load_config(data)
    rows = pipeline.read_rows(pipeline.run(cfg) / "results.csv")
    assert rows[0]["bulk_modulus.periodic"] == pytest.approx(K_COPPER, rel=1e-9)
    assert 40.0 < rows[0]["shear_modulus.periodic"] < 54.7
    if not mg.mesh.conforming.has_mmg():
        return
    data = polycrystal_config(tmp_path, mesh={"size": 0.2, "min_feature": 0.05})
    data["homogenization"]["boundary_conditions"] = ["kubc", "periodic"]
    data["outputs"]["conforming_mesh"] = ["vtu", "inp"]
    cfg = load_config(data)
    assert cfg.mesh.kwargs(False)["min_feature"] == 0.05
    outdir = pipeline.run(cfg)
    rows = pipeline.read_rows(outdir / "results.csv")
    for bc in ("kubc", "periodic"):
        assert rows[0][f"bulk_modulus.{bc}"] == pytest.approx(K_COPPER, rel=1e-7)
    assert rows[0]["shear_modulus.periodic"] < rows[0]["shear_modulus.kubc"]
    mesh = mg.io.read_tet_mesh(outdir / "grains_0000_tet.vtu")
    assert mesh.cell is not None and mesh.orientations is not None and mesh.matching_faces()
    inp = (outdir / "grains_0000_tet.inp").read_text(encoding="utf-8")
    assert "*ORIENTATION, NAME=ORI_" in inp
    # the materials of the run, the copper in the axes of each grain
    assert "*MATERIAL, NAME=PHASE_1\n*ELASTIC, TYPE=ANISOTROPIC" in inp
    assert ", MATERIAL=PHASE_1, ORIENTATION=ORI_" in inp


def test_cli_homogenize_polycrystal(tmp_path, capsys):
    tess = mg.models.Voronoi(n_cells=10, crystal_orientations=True).generate(
        mg.Domain.cube(1.0, periodic=True), seed=4
    )
    geometry = tmp_path / "grains.json"
    mg.io.save_microstructure(tess, geometry)
    args = ["homogenize", str(geometry), "--material", "1=cubic:168.4/121.4/75.4", "--json"]
    assert cli.main([*args, "--voxels", "12"]) == 0
    res = json.loads(capsys.readouterr().out)
    bulk = next(r for r in res["results"] if r["property"] == "bulk_modulus")
    assert bulk["value"] == pytest.approx(K_COPPER, rel=1e-9) and res["shape"] == [12, 12, 12]
    if mg.mesh.conforming.has_mmg():
        assert cli.main([*args, "--size", "0.25", "--min-feature", "0.05", "--bc", "subc"]) == 0
        res = json.loads(capsys.readouterr().out)
        bulk = next(r for r in res["results"] if r["property"] == "bulk_modulus")
        assert bulk["value"] == pytest.approx(K_COPPER, rel=1e-7)


def test_cli_fit_with_groups(tmp_path, capsys):
    rng = np.random.default_rng(1)
    rows = []
    for bc, mean in (("KUBC", 10.0), ("SUBC", 6.0)):
        for edge in (5, 10, 20, 40):
            for _r in range(12):
                rows.append(
                    {
                        "size": edge,
                        "bc": bc,
                        "k": mean + np.sqrt(4.0 * edge**-2.0) * rng.standard_normal(),
                    }
                )
    pipeline.write_rows(tmp_path / "fe.csv", rows)
    assert (
        cli.main(
            [
                "fit",
                str(tmp_path / "fe.csv"),
                "--group-by",
                "bc",
                "--property",
                "k",
                "--eps",
                "0.01",
                "--n",
                "1",
                "--bootstrap",
                "100",
                "-o",
                str(tmp_path / "rep"),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "rep" / "rve_report.json").read_text(encoding="utf-8"))
    assert set(report["properties"]) == {"k.KUBC", "k.SUBC"}
    gamma = report["properties"]["k.KUBC"]["free_fit"]["gamma"]
    assert gamma == pytest.approx(2 / 3, abs=0.15)
    assert report["discrepancy"]["k"][0]["discrepancy"] == pytest.approx(4 / 6, abs=0.1)
    assert "| eps_rel |" in capsys.readouterr().out
