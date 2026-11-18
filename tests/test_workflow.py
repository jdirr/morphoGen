"""The whole chain in configurations: the ``visualization`` section (figures and videos of
the images, meshes and fields of the first realizations), ``outputs.tables``, the fields
of loading paths at every step, the template ``workflow`` and the command ``morphogen
export``."""

import copy
import importlib.util

import numpy as np
import pytest
import yaml

import morphogen as mg
from morphogen import cli, pipeline
from morphogen.config import ConfigError, load_config
from morphogen.io.tables import read_table
from morphogen.viz import video as vv
from morphogen.viz import workflow

pytest.importorskip("matplotlib")
HAS_VTK = importlib.util.find_spec("vtk") is not None
CAN_RENDER = vv.vtk_can_render()  # VTK, with a display, EGL or OSMesa

PLASTIC = {
    0: {
        "young": 70000,
        "poisson": 0.33,
        "plasticity": {"yield": 100, "isotropic": [{"Q": 80, "b": 20}]},
    },
    1: {"young": 400000, "poisson": 0.2},
}
PATH = {"times": [0, 1], "increments": [3], "strain": {"11": [0, 0.01]}}


def _base(tmp_path, method="fft") -> dict:
    cfg = {
        "model": {"type": "hardcore_spheres", "radius": 0.8, "volume_fraction": 0.2},
        "domain": {"size": 5.0, "periodic": True},
        "seed": 7,
        "voxels": {"shape": 12},
        "homogenization": {
            "materials": PLASTIC,
            "method": method,
            "boundary_conditions": ["periodic"],
            "properties": ["shear_modulus"],
            "paths": {"tension": PATH},
        },
        "outputs": {"directory": str(tmp_path / "out"), "name": "s"},
        "visualization": {"figures": [{"of": "image"}]},
    }
    if method == "fe":
        cfg["mesh"] = {"size": 1.4}
    return cfg


def _views(tmp_path, method="fft", **section):
    cfg = _base(tmp_path, method)
    cfg["visualization"] = section
    return load_config(cfg)


def test_visualization_configuration(tmp_path):
    cfg = _views(
        tmp_path,
        "fe",
        figures=["image", {"of": "mesh"}, {"of": "shear_modulus.periodic"}, "tension.periodic"],
        videos=[
            "image",
            {"of": "mesh", "kind": "sweep", "axis": 0, "frames": 5},
            {"of": "tension.periodic", "color": "p", "section": {"axis": 1}, "mode": "bounce"},
        ],
        size=[320, 240],
        cmap="inferno",
        elevation=10,
        azimuth=-45,
    )
    viz = cfg.visualization
    assert (viz.elevation, viz.azimuth) == (10.0, -45.0)
    image, mesh, field, curves = viz.figures
    assert (image.kind, image.formats, image.label, image.vtk) == (
        "sections",
        ["png"],
        "image_sections",
        False,
    )
    assert (mesh.kind, mesh.color, mesh.label, mesh.vtk) == ("view", "phase", "mesh", True)
    assert (field.group, field.kind, field.color, field.vtk) == (
        "result",
        "field",
        "von_mises",
        True,
    )
    assert field.label == "shear_modulus_periodic_von_mises"
    assert (curves.group, curves.kind, curves.label) == (
        "path",
        "curves",
        "tension_periodic_curves",
    )
    ortho, sweep, path = viz.videos
    assert (ortho.kind, ortho.formats, ortho.mode) == ("orthoslices", ["mp4"], "once")
    assert (sweep.label, sweep.axis, sweep.frames) == ("mesh_sweep", 0, 5)
    assert (path.section, path.mode, path.label) == ((1, 0.5), "bounce", "tension_periodic_p")
    assert viz.fields() == "steps" and viz.shows("mesh") and not viz.shows("labels")
    assert (viz.size, viz.cmap, viz.realizations) == ((320, 240), "inferno", 1)
    assert workflow.requirements(viz) == {"matplotlib", "vtk", "ffmpeg"}
    still = _views(tmp_path, figures=[{"of": "shear_modulus.periodic", "formats": ["svg", "tex"]}])
    assert still.visualization.fields() is True and not still.visualization.figures[0].vtk
    assert _views(tmp_path, figures=["tension.periodic"]).visualization.fields() is False
    errors = [
        ({"figures": [{"of": "grains"}]}, "result '<name>.<bc>'"),
        ({"figures": [{"of": "shear_modulus.kubc"}]}, "boundary conditions"),
        ({"figures": [{"of": "bulk_modulus.periodic"}]}, "none of the properties"),
        ({"figures": [{"of": "image", "kind": "orthoslices"}]}, "kind"),
        ({"figures": [{"of": "image", "frames": 3}]}, "unknown keys"),
        ({"videos": [{"of": "tension.periodic", "wave": "sine"}]}, "unknown keys"),
        ({"figures": [{"of": "shear_modulus.periodic", "color": "p"}]}, "color"),
        ({"figures": [{"of": "shear_modulus.periodic", "color": "displacement"}]}, "nodes"),
        ({"figures": [{"of": "tension.periodic", "section": [2, 0.5]}]}, "planes cut"),
        ({"videos": [{"of": "image", "mode": "sideways"}]}, "mode"),
        ({"videos": [{"of": "image", "formats": ["avi"]}]}, "unknown value"),
        ({"figures": ["image", {"of": "image", "kind": "sections"}]}, "same files"),
        ({"figures": [{"of": "image", "cmap": "nope"}]}, "colormap"),
        ({"figures": [{"of": "shear_modulus.periodic", "clim": [2, 1]}]}, "low < high"),
        ({"size": [8, 8], "figures": ["image"]}, "size"),
        ({"azimuth": "left", "figures": ["image"]}, "azimuth"),
        ({}, "figures or videos"),
    ]
    for section, match in errors:
        with pytest.raises(ConfigError, match=match):
            _views(tmp_path, **section)
    fe_errors = [
        ({"figures": [{"of": "mesh", "formats": ["svg"]}]}, "bitmaps"),
        ({"figures": [{"of": "mesh", "color": "von_mises"}]}, "'phase' or 'none'"),
        ({"videos": [{"of": "tension.periodic", "section": {"axis": 3}}]}, "section"),
    ]
    for section, match in fe_errors:
        with pytest.raises(ConfigError, match=match):
            _views(tmp_path, "fe", **section)
    cfg = _base(tmp_path)
    cfg["visualization"] = {"figures": ["mesh"]}
    with pytest.raises(ConfigError, match="'mesh' section"):
        load_config(cfg)
    del cfg["voxels"]
    cfg["homogenization"]["method"] = "fe"
    cfg["mesh"] = {"size": 1.0}
    cfg["visualization"] = {"figures": ["image"]}
    with pytest.raises(ConfigError, match="'voxels' section"):
        load_config(cfg)
    cfg = _base(tmp_path)
    cfg["homogenization"]["materials"] = {
        0: {"young": 1.0, "poisson": 0.3, "conductivity": 1.0},
        1: {"young": 2.0, "poisson": 0.3, "conductivity": 2.0},
    }
    cfg["homogenization"]["properties"] = ["conductivity"]
    del cfg["homogenization"]["paths"]
    cfg["visualization"] = {"figures": [{"of": "conductivity.periodic", "part": "fluctuation"}]}
    with pytest.raises(ConfigError, match="mechanical"):
        load_config(cfg)
    cfg["visualization"] = {"figures": ["conductivity.periodic"]}
    assert load_config(cfg).visualization.figures[0].color == "flux"


def test_requirements_are_checked_before_the_run(tmp_path, monkeypatch):
    cfg = _views(tmp_path, videos=[{"of": "image", "formats": ["mp4"]}])
    monkeypatch.setattr(vv, "ffmpeg_path", lambda: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        pipeline.run(cfg)
    assert not (tmp_path / "out" / "results.csv").exists()
    gif = _views(tmp_path, videos=[{"of": "image", "formats": ["gif"]}])
    workflow.check(gif.visualization)  # no ffmpeg needed


def test_3d_views_need_a_display(tmp_path, monkeypatch):
    """Without a display, EGL or OSMesa, VTK would abort the process at its first render:
    the workflow says so before the run, and a scene raises an error instead."""
    monkeypatch.setattr(vv, "vtk_can_render", lambda: False)
    cfg = _views(tmp_path, "fe", figures=[{"of": "mesh"}])
    with pytest.raises(RuntimeError, match="a display" if HAS_VTK else "VTK for the 3D views"):
        workflow.check(cfg.visualization)
    if HAS_VTK:
        with pytest.raises(RuntimeError, match="cannot render"):
            vv.Scene()


@pytest.mark.slow
def test_workflow_on_voxels(tmp_path):
    """FFT solver: sections of the image, fields and curves, videos of sections, the
    fields of the loading path at every step, the tables as .txt and .xlsx."""
    cfg = _base(tmp_path)
    cfg["realizations"] = 2
    cfg["outputs"]["tables"] = ["txt", "xlsx", "csv"]
    cfg["homogenization"]["fields"] = True
    cfg["visualization"] = {
        "size": [160, 120],
        "figures": [
            {"of": "image", "formats": ["png", "svg", "tex"]},
            {"of": "shear_modulus.periodic", "color": "stress_12", "formats": ["png", "pdf"]},
            {"of": "tension.periodic", "formats": ["png"]},
            {"of": "tension.periodic", "kind": "field", "color": "p", "name": "plastic"},
        ],
        "videos": [
            {"of": "image", "frames": 3, "formats": ["gif"]},
            {"of": "image", "kind": "sections", "axis": 0, "formats": ["frames"]},
            {"of": "shear_modulus.periodic", "formats": ["gif"]},
            {"of": "tension.periodic", "color": "von_mises", "formats": ["gif"], "mode": "bounce"},
        ],
    }
    out = pipeline.run(load_config(cfg))
    names = {p.name for p in out.iterdir()}
    stills = {
        "s_0000_image_sections.png",
        "s_0000_image_sections.svg",
        "s_0000_image_sections.tex",
        "s_0000_shear_modulus_periodic_stress_12.png",
        "s_0000_shear_modulus_periodic_stress_12.pdf",
        "s_0000_tension_periodic_curves.png",
        "s_0000_plastic.png",
    }
    videos = {
        "s_0000_image_orthoslices.gif",
        "s_0000_image_sections_frames",
        "s_0000_shear_modulus_periodic_von_mises.gif",
        "s_0000_tension_periodic_von_mises.gif",
    }
    assert stills | videos <= names
    assert not any(
        n.startswith("s_0001_") and "periodic_" in n and n.endswith(".gif") for n in names
    )
    from PIL import Image

    # the natural state and the 3 steps, back and forth
    assert Image.open(out / "s_0000_tension_periodic_von_mises.gif").n_frames == 6
    assert len(list((out / "s_0000_image_sections_frames").glob("*.png"))) == 12
    # the files of the fields: the end of the segment only, though every step was kept
    assert (out / "s_0000_tension_periodic_1.vti").exists()
    assert not (out / "s_0000_tension_periodic_2.vti").exists()
    for stem in ("results", "homogenization", "s_0000_tension_periodic", "s_0001_tension_periodic"):
        csv = read_table(out / f"{stem}.csv")
        as_text = {k: [str(v) for v in values] for k, values in csv.items()}
        for fmt in ("txt", "xlsx"):
            other = read_table(out / f"{stem}.{fmt}")
            assert {k: [str(v) for v in values] for k, values in other.items()} == as_text
    assert len(read_table(out / "results.xlsx")["seed"]) == 2


@pytest.mark.slow
@pytest.mark.skipif(not CAN_RENDER, reason="VTK cannot render here")
def test_template_workflow(tmp_path):
    """The template of the whole chain, scaled down: finite elements on the conforming
    mesh, 3D views of the mesh and of the fields on the deformed mesh, videos."""
    data = yaml.safe_load(cli._templates()["workflow"])
    data["outputs"]["directory"] = str(tmp_path / "out")
    data["voxels"]["shape"] = 16
    data["mesh"]["size"] = 1.4
    data["homogenization"]["paths"]["tension"]["increments"] = [2]
    viz = data["visualization"]
    viz["size"] = [200, 150]
    fmt = ["mp4"] if vv.ffmpeg_path() else ["gif"]
    for video in viz["videos"]:
        video["formats"] = fmt
        video["frames"] = 4
    out = pipeline.run(load_config(data))
    stem = out / "composite_0000"
    for name in (
        "image_sections.png",
        "image_sections.tex",
        "mesh.png",
        "shear_modulus_periodic_von_mises.png",
        "tension_periodic_curves.pdf",
        "tension_periodic_curves.tex",
        "tension_periodic_p.png",
        f"image_orthoslices.{fmt[0]}",
        f"mesh.{fmt[0]}",
        f"mesh_sweep.{fmt[0]}",
        f"shear_modulus_periodic_von_mises.{fmt[0]}",
        f"tension_periodic_von_mises.{fmt[0]}",
    ):
        assert (out / f"composite_0000_{name}").stat().st_size > 100, name
    assert (out / "composite_0000_tet.vtu").exists() and (out / "results.xlsx").exists()
    from PIL import Image

    view = np.asarray(Image.open(f"{stem}_shear_modulus_periodic_von_mises.png"))
    assert view.shape == (150, 200, 3) and len(np.unique(view.reshape(-1, 3), axis=0)) > 50


@pytest.mark.skipif(not CAN_RENDER, reason="VTK cannot render here")
def test_views_of_meshes(tmp_path):
    from morphogen.homogenization.bloch import voxel_tet_mesh

    data = np.zeros((4, 4, 4), np.uint8)
    data[1:3, 1:3, :] = 1
    mesh = voxel_tet_mesh(mg.VoxelImage(data, periodic=True))
    u = np.zeros((mesh.n_nodes, 3))
    u[:, 0] = 0.1 * mesh.nodes[:, 2]
    phases = vv.deformed_view(mesh, None, "phase", size=(200, 150))
    assert phases.shape == (150, 200, 3)
    turned = vv.deformed_view(mesh, None, "phase", size=(200, 150), azimuth=120.0, elevation=60.0)
    assert np.abs(turned.astype(int) - phases.astype(int)).max() > 0  # another camera
    # the colour bar of the phases: one box per phase, in the colours of the sections
    colours = {tuple(c) for c in phases[:, -vv.BAR + 8 : -vv.BAR + 22].reshape(-1, 3)}
    assert {tuple(c) for c in mg.viz.png.label_colors(np.arange(2))} <= colours
    moved = vv.deformed_view(mesh, u, np.linalg.norm(u, axis=1), magnification=3.0, size=(200, 150))
    assert np.abs(moved.astype(int) - phases.astype(int)).max() > 0
    history = vv.history_frames(
        mesh, [0 * u, u, 2 * u], frames=5, size=(120, 90), captions=["t = 0", "t = 1", "t = 2"]
    )
    assert len(history) == 5 and history[0].shape == (90, 120, 3)
    assert np.abs(history[-1].astype(int) - history[0].astype(int)).max() > 0
    with pytest.raises(ValueError, match="one field per displacement"):
        vv.history_frames(mesh, [u, u], [np.zeros(mesh.n_elements)])


def test_sections_with_colour_bars():
    rng = np.random.default_rng(0)
    fields = [rng.random((6, 7, 8)) * k for k in (1.0, 2.0)]
    frames = vv.time_section_frames(fields, 2, size=64, captions=["t = 0", "t = 1"], label="p")
    assert len(frames) == 2 and frames[0].shape[1] == frames[0].shape[1] and frames[0].shape[1] > 64
    plain = vv.section_frames(fields[0], 1, size=64, labels=False)
    barred = vv.section_frames(fields[0], 1, size=64, labels=False, bar=True)
    assert barred[0].shape[1] == plain[0].shape[1] + vv.BAR
    assert np.array_equal(barred[0][:, : plain[0].shape[1]], plain[0])
    phases = vv.section_frames(np.ones((3, 3, 3), np.uint8), 0, size=32, bar=True)
    assert phases[0].shape[1] == 32  # no bar for phases


def test_fields_of_paths_at_every_step():
    image = mg.VoxelImage(
        (np.indices((6, 6, 6)).sum(axis=0) % 3 == 0).astype(np.uint8), periodic=True
    )
    mats = mg.homogenization.materials_from_config(PLASTIC)
    path = mg.homogenization.LoadingPath.from_spec(
        {"times": [0, 1, 2], "increments": [2, 1], "strain": {"11": [0, 0.01, 0.0]}}
    )
    res = mg.homogenization.homogenize(
        image, mats, properties=[], paths={"t": path}, fields="steps"
    )["t.periodic"]
    assert sorted(res.fields) == pytest.approx([0.5, 1.0, 2.0])
    named = workflow.path_fields(res.fields[1.0])
    assert {"von_mises", "p", "stress_12", "plastic_strain_11", "displacement"} <= set(named)
    assert workflow.field_values(named, "displacement").shape == image.data.shape
    with pytest.raises(ValueError, match="no field"):
        workflow.field_values(named, "temperature")
    from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem

    with pytest.raises(ValueError, match="'steps'"):
        NonlinearFFTProblem(image, mats, "periodic").run(path, fields="every")


def test_export_command(tmp_path, capsys):
    model = mg.models.BooleanSpheres(radius=0.2, volume_fraction=0.2)
    micro = model.generate(mg.Domain.cube(1.0, periodic=True), seed=1)
    mg.io.save_microstructure(micro, tmp_path / "m.json")
    mg.io.save_image(micro.voxelize(shape=10), tmp_path / "m.vti")
    mg.io.write_table(tmp_path / "t.csv", {"x": [1.0, 2.0], "y": [3.0, 4.0]})
    runs = [
        ["m.json", "m.svg", "--shape", "12"],
        ["m.vti", "o.gif", "--kind", "orthoslices", "--frames", "3", "--size", "64"],
        ["m.vti", "s", "--format", "frames", "--axis", "1"],
        ["t.csv", "t.xlsx"],
    ]
    for argv in runs:
        assert (
            cli.main(["export", *(str(tmp_path / a) if "." in a or a == "s" else a for a in argv)])
            == 0
        )
    assert (tmp_path / "m.svg").exists() and (tmp_path / "o.gif").exists()
    assert len(list((tmp_path / "s").glob("*.png"))) == 10
    assert read_table(tmp_path / "t.xlsx")["y"] == [3.0, 4.0]
    if CAN_RENDER:
        from morphogen.homogenization.bloch import voxel_tet_mesh

        mesh = voxel_tet_mesh(micro.voxelize(shape=4))
        u = np.tile([0.0, 0.0, 0.01], (mesh.n_nodes, 1))
        mg.io.write_vtu_tet(
            tmp_path / "f.vtu", mesh, {"s": np.arange(mesh.n_elements, dtype=float)}, {"u": u}
        )
        common = ["--size", "120", "90", "--frames", "3"]
        assert (
            cli.main(
                [
                    "export",
                    str(tmp_path / "f.vtu"),
                    str(tmp_path / "f.gif"),
                    "--field",
                    "s",
                    "--displacement",
                    "u",
                    *common,
                ]
            )
            == 0
        )
        assert (
            cli.main(
                ["export", str(tmp_path / "f.vtu"), str(tmp_path / "f.png"), "--field", "phase"]
            )
            == 0
        )
        capsys.readouterr()
        assert (
            cli.main(
                ["export", str(tmp_path / "f.vtu"), str(tmp_path / "g.gif"), "--field", "missing"]
            )
            == 2
        )
        assert "no array 'missing'" in capsys.readouterr().err


def test_template_is_listed():
    data = yaml.safe_load(cli._templates()["workflow"])
    cfg = load_config(copy.deepcopy(data))
    assert cfg.visualization is not None and cfg.outputs.tables == ["txt", "xlsx"]
    assert {v.of for v in cfg.visualization.views} == {
        "image",
        "mesh",
        "shear_modulus.periodic",
        "tension.periodic",
    }
