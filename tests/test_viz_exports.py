"""Exports: tables (csv, txt, xlsx), figures (bitmaps, vector graphics, TikZ compiled by
LaTeX when present), videos (sections, orthoslices, 3D turntables, sweeps and fields by
VTK, mp4 by ffmpeg, gif) and the entry point ``morphogen.export``."""

import shutil
import subprocess
import zipfile

import numpy as np
import pytest

import morphogen as mg
from morphogen import homogenization as hz
from morphogen.io.tables import read_table, write_table, write_tables
from morphogen.viz import video as vv
from morphogen.viz.export import save_figure, tikz

pytest.importorskip("matplotlib")
CAN_RENDER = vv.vtk_can_render()  # VTK, with a display, EGL or OSMesa


def _figure():
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6, 3))
    ax1, ax2 = fig.subplots(1, 2)
    x = np.linspace(0.1, 10, 30)
    ax1.plot(x, np.sqrt(x), label=r"$\sqrt{x}$")
    ax1.plot(x, np.log(x), "--o", label="log & co")
    ax1.axhspan(1.0, 1.5, color="0.9")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"strain $\varepsilon$")
    ax1.set_title("50% of it")
    ax1.legend()
    ax2.imshow(np.arange(12.0).reshape(3, 4), origin="lower", extent=(0, 4, 0, 3))
    ax2.set_xticks([0, 2, 4], [r"$\Gamma$", "X", "M"])
    ax2.add_collection(LineCollection([[(0, 0), (1, 1)], [(1, 1), (2, 0.5)]], colors="k"))
    ax2.scatter([0.5, 1.0], [1.5, 1.2])
    return fig


def test_tables(tmp_path):
    records = [{"k": i, "x": 0.25 * i, "name": f"p<{i}>", "ok": i % 2 == 0} for i in range(4)]
    records[1]["x"] = float("nan")
    for suffix in (".csv", ".txt", ".xlsx"):
        back = read_table(write_table(tmp_path / f"t{suffix}", records))
        assert list(back) == ["k", "x", "name", "ok"] and back["k"] == [0, 1, 2, 3]
        assert back["name"][2] == "p<2>" and back["x"][3] == 0.75
    assert read_table(tmp_path / "t.xlsx")["ok"][:2] == [True, False]
    assert np.loadtxt(tmp_path / "t.txt", usecols=(0, 1)).shape == (4, 2)
    write_tables(
        tmp_path / "book.xlsx", {"curve": {"t": [0, 1], "y": [1.5, 2.5]}, "eye": np.eye(2)}
    )
    with zipfile.ZipFile(tmp_path / "book.xlsx") as z:
        assert {"xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml", "xl/styles.xml"} <= set(
            z.namelist()
        )
    assert read_table(tmp_path / "book.xlsx", sheet=1) == {"c1": [1.0, 0.0], "c2": [0.0, 1.0]}
    paths = write_tables(tmp_path / "many.csv", {"a": [[1, 2]], "b": [[3, 4]]})
    assert [p.name for p in paths] == ["many_a.csv", "many_b.csv"]


def test_figures_and_tikz(tmp_path):
    fig = _figure()
    for suffix in (".png", ".tif", ".svg", ".pdf"):
        assert save_figure(fig, tmp_path / f"f{suffix}").stat().st_size > 1000
    code = tikz(fig, tmp_path / "f.tex")
    assert code.count(r"\begin{axis}") == 2 and r"xmode=log" in code
    assert r"\addlegendentry{$\sqrt{x}$}" in code and r"\addlegendentry{log \& co}" in code
    assert r"{$\Gamma$}" in code and r"\addplot graphics" in code and r"\fill[" in code
    assert r"50\% of it" in code and "unbounded coords=jump" in code
    assert (tmp_path / "f-1-0.png").exists()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tikz_compiles(tmp_path):
    save_figure(_figure(), tmp_path / "f.tex")
    (tmp_path / "doc.tex").write_text(
        r"\documentclass{standalone}\usepackage{pgfplots}\pgfplotsset{compat=1.18}"
        r"\begin{document}\input{f.tex}\end{document}"
    )
    done = subprocess.run(  # by its path: Windows does not look the bare name up in PATH
        [shutil.which("pdflatex"), "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
        cwd=tmp_path,
        capture_output=True,
        timeout=600,
    )
    assert done.returncode == 0 and (tmp_path / "doc.pdf").exists()


def test_video_sequences_and_writers(tmp_path):
    frames = [np.full((5, 7, 3), k, np.uint8) for k in range(4)]
    assert len(vv.sequence(frames)) == 4
    assert [int(f[0, 0, 0]) for f in vv.sequence(frames, "bounce")] == [0, 1, 2, 3, 2, 1]
    assert len(vv.sequence(frames, "loop", repeat=3)) == 12
    with pytest.raises(ValueError, match="mode"):
        vv.sequence(frames, "sideways")
    gif = vv.write_video(frames, tmp_path / "a.gif", mode="loop", repeat=2)
    from PIL import Image

    assert Image.open(gif).n_frames == 8
    folder = vv.write_video(frames, tmp_path / "frames", mode="bounce")
    assert len(list(folder.glob("frame_*.png"))) == 6
    if vv.ffmpeg_path():
        assert (
            vv.write_video(frames, tmp_path / "a.mp4", fps=4).stat().st_size > 100
        )  # odd sizes padded
    image = mg.VoxelImage(np.indices((6, 8, 10)).sum(axis=0).astype(np.uint8) % 3)
    f = vv.section_frames(image, axis=1, size=64)
    assert len(f) == 8 and max(f[0].shape[:2]) == 64
    f = vv.orthoslice_frames(
        image.astype(float) if hasattr(image, "astype") else image, frames=5, size=40
    )
    assert len(f) == 5 and f[0].shape == (40, 120, 3)


@pytest.mark.skipif(not CAN_RENDER, reason="VTK cannot render here")
def test_3d_videos(tmp_path):
    from morphogen.homogenization.bloch import voxel_tet_mesh

    data = np.zeros((4, 4, 4), np.uint8)
    data[1:3, 1:3, :] = 1
    image = mg.VoxelImage(data, periodic=True)
    turn = vv.turntable_frames(image, frames=4, size=(120, 90))
    assert (
        len(turn) == 4 and turn[0].shape == (90, 120, 3) and turn[0].min() < 200
    )  # something drawn
    mesh = voxel_tet_mesh(image, phases=[1])
    u = np.zeros((mesh.n_nodes, 3))
    u[:, 0] = np.sin(np.pi * mesh.nodes[:, 2] / 4.0)
    for frames in (
        vv.turntable_frames(mesh, field=u, frames=3, size=(120, 90)),
        vv.section_sweep_frames(mesh, u[:, 0], axis=2, frames=3, size=(120, 90)),
        vv.field_frames(mesh, u, frames=4, size=(120, 90), section=(1, 0.5), wave="ramp"),
    ):
        assert len(frames) in (3, 4) and frames[0].shape == (90, 120, 3)
    moving = vv.field_frames(mesh, u, frames=4, size=(120, 90))
    assert np.abs(moving[1].astype(int) - moving[0].astype(int)).max() > 0  # the mesh moves
    # the colours follow a growing load (the mesh held still): the field of a linear
    # solution grows from zero; a field that does not follow the load keeps its colours
    still = {"magnification": 0.0, "frames": 4, "size": (120, 90), "wave": "ramp"}
    grows = vv.field_frames(mesh, u, u[:, 0] + 1.0, **still)
    assert np.abs(grows[-1].astype(int) - grows[0].astype(int)).max() > 50
    fixed = vv.field_frames(mesh, u, u[:, 0] + 1.0, scaling="none", **still)
    assert np.array_equal(fixed[-1], fixed[0])
    assert [vv.field_scaling(n) for n in ("von_mises", "energy_density", "stress_12", None)] == [
        "magnitude",
        "quadratic",
        "linear",
        "magnitude",
    ]


def test_export_entry_point(tmp_path):
    image = mg.VoxelImage((np.indices((6, 6, 6)).sum(axis=0) % 2).astype(np.uint8), periodic=True)
    for suffix in (".vti", ".tif", ".npy", ".png", ".csv"):
        assert mg.export(image, tmp_path / f"img{suffix}").exists()
    assert len(read_table(tmp_path / "img.csv")["value"]) == 216
    assert mg.export(image, tmp_path / "img.gif", kind="orthoslices", frames=3, size=32).exists()
    frame = hz.Frame.lattice_of("honeycomb", (2, 2), elements=2, thickness=0.05)
    bif = hz.post_bifurcation(frame, -1e-3 * np.eye(2))
    mg.export(bif, tmp_path / "bif.xlsx")
    table = read_table(tmp_path / "bif.xlsx")
    assert {"irrep", "dim", "load", "point_group", "conjugates"} <= set(table)
    assert mg.export(bif, tmp_path / "bif.png", frame=frame, irreps=[bif.first[1]]).exists()
    pattern = next(p.mode for p in bif.patterns if p.irrep == bif.first[1])
    assert mg.export(
        frame, tmp_path / "pattern.gif", pattern=pattern, frames=4, mode="bounce"
    ).exists()
    bands = hz.band_structure(
        mg.VoxelImage(np.zeros((2, 2, 2), np.uint8), periodic=True),
        {0: hz.Material(young=1.0, poisson=0.25, density=1.0)},
        ("G", "X"),
        bands=3,
        points=2,
        order=1,
    )
    mg.export(bands, tmp_path / "bands.csv")
    assert set(read_table(tmp_path / "bands.csv")) >= {"distance", "kx", "band_1", "band_3"}
    assert mg.export(bands, tmp_path / "bands.svg").exists()
    model = mg.models.BooleanSpheres(radius=0.2, volume_fraction=0.2)
    micro = model.generate(mg.Domain.cube(1.0, periodic=True), seed=1)
    assert mg.export(micro, tmp_path / "micro.json").exists()
    assert mg.export(micro, tmp_path / "micro.vti", shape=(8, 8, 8)).exists()
    assert mg.export({"a": [1, 2]}, tmp_path / "plain.txt").exists()
    with pytest.raises(TypeError, match="no writer"):
        mg.export(object(), tmp_path / "x.csv")
