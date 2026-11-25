"""Visualization (PNG, figures, animations) and the web interface."""

import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from importlib import resources

import numpy as np
import pytest
import yaml

import morphogen as mg
from morphogen import cli
from morphogen.gui import api
from morphogen.gui.server import GuiServer
from morphogen.viz import decode_png, encode_png, slice_of, to_rgb
from morphogen.viz.video import vtk_can_render

# -- images without dependencies -----------------------------------------------------------


def test_png_round_trip_and_colours():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (7, 5, 3), dtype=np.uint8)
    data = encode_png(rgb)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.array_equal(decode_png(data), rgb)
    assert decode_png(encode_png(rgb[..., 0])).shape == (7, 5, 3)  # grey levels
    phases = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    colours = to_rgb(phases)
    assert np.array_equal(colours[0, 1], colours[1, 1]) and not np.array_equal(
        colours[0, 0], colours[0, 1]
    )
    field = to_rgb(np.array([[0.0, 1.0]]))
    assert np.array_equal(field[0, 0], [68, 1, 84]) and np.array_equal(field[0, 1], [253, 231, 37])
    labels = to_rgb(np.array([[-1, 3]]), "labels")
    assert np.array_equal(labels[0, 0], [0, 0, 0]) and labels[0, 1].sum() > 0
    vol = np.arange(24).reshape(2, 3, 4)
    s = slice_of(vol, 2, 1)  # x horizontal, y vertical, top row = largest y
    assert s.shape == (3, 2) and s[0, 0] == vol[0, 2, 1] and s[-1, -1] == vol[1, 0, 1]
    with pytest.raises(ValueError):
        encode_png(np.zeros((2, 2, 4)))


# -- evolution of structures ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "domain"),
    [
        (mg.models.CellularAutomatonModel(steps=10), mg.Domain.cube(12.0)),
        (mg.models.ActivatorInhibitorModel(steps=5), mg.Domain.cube(12.0, periodic=False)),
        (mg.models.ReactionDiffusionModel(steps=400, cell_size=0.5), mg.Domain.cube(16.0)),
        (mg.models.NucleationGrowthModel(nucleus_density=0.01, time=3.0), mg.Domain.cube(12.0)),
        (mg.models.AggregationModel(n_seeds=3), mg.Domain.cube(12.0)),
    ],
)
def test_evolution_ends_with_the_realization(model, domain):
    frames = model.evolution(domain, seed=4, frames=5)
    assert 3 <= len(frames) <= 7
    micro = model.generate(domain, seed=4)
    last = frames[-1]
    if last.dtype == np.int32:
        assert np.array_equal(last, micro.cell_labels)
    elif last.dtype == np.uint8:
        assert np.array_equal(last, micro.phases)
    else:
        np.testing.assert_allclose(last.astype(np.float32), micro.field)
    assert all(f.shape == last.shape for f in frames)


# -- figures and the view command ----------------------------------------------------------


def test_figures_and_view_command(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from morphogen.viz import figures

    micro = mg.models.CellularAutomatonModel(steps=8).generate(mg.Domain.cube(12.0), seed=1)
    for fig in (
        figures.slices_figure(micro),
        figures.tiled_figure(micro, tiles=2),
        figures.tiled_figure(micro, tiles=2, axis=0),
        figures.surface_figure(micro),
        figures.history_figure(micro.attributes["history"]),
    ):
        fig.savefig(tmp_path / "f.png")
        assert (tmp_path / "f.png").stat().st_size > 1000
    # a cell tiled in its three sections, a planar one in its plane
    assert len(figures.tiled_figure(micro, tiles=2).axes) == 3
    assert len(figures.tiled_figure(micro, tiles=2, axis=1).axes) == 1
    assert len(figures.tiled_figure(np.zeros((6, 6, 1), np.uint8), tiles=2).axes) == 1
    with pytest.raises(ValueError, match="axis"):
        figures.tiled_figure(micro, axis=3)
    grains = mg.models.NucleationGrowthModel(nucleus_density=0.01, time=3.0).generate(
        mg.Domain.cube(12.0), seed=1
    )
    figures.kinetics_figure(grains).savefig(tmp_path / "k.png")
    figures.slices_figure(grains, labels=True).savefig(tmp_path / "l.png")
    with pytest.raises(ValueError, match="kinetics"):
        figures.kinetics_figure(micro)
    cfg = yaml.safe_load(cli._templates()["cellular_automaton"])
    cfg["domain"]["size"] = 4.0
    cfg["voxels"]["shape"] = 16
    path = tmp_path / "ca.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert cli.main(["view", str(path), "-o", str(tmp_path / "ca.png")]) == 0
    assert (
        cli.main(["view", str(path), "--animate", str(tmp_path / "ca.gif"), "--frames", "4"]) == 0
    )
    assert (tmp_path / "ca.gif").read_bytes()[:6] in (b"GIF89a", b"GIF87a")
    mg.io.save_microstructure(micro, tmp_path / "m.json")
    assert cli.main(["view", str(tmp_path / "m.json"), "--tile", "2"]) == 0
    assert (tmp_path / "m_view.png").exists()
    mg.io.save_image(micro.voxelize(), tmp_path / "m.vti")
    assert cli.main(["view", str(tmp_path / "m.vti"), "--history"]) == 2  # no history
    assert "history" in capsys.readouterr().err


# -- web interface -------------------------------------------------------------------------


def test_cut_of_a_mesh_by_a_plane():
    """The cut of a tetrahedral mesh by a plane: polygons of 3 or 4 points whose areas sum
    to the section of the box, on a plane through the elements as on one that holds their
    faces (each face cut once: by the element below it); its figure, in the colours of
    the phases or of values per element."""
    from helpers import box_mesh
    from morphogen.viz.figures import mesh_figure, mesh_section

    mesh = box_mesh()  # a box 1 x 1.3 x 0.8 of regular tetrahedra
    lower, upper = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)

    def area(polygon):
        x, y = polygon[:, 0], polygon[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    layers = np.unique(mesh.nodes[:, 2])
    assert len(layers) > 2
    for axis in range(3):
        u, v = (a for a in range(3) if a != axis)
        section = (upper[u] - lower[u]) * (upper[v] - lower[v])
        through = lower[axis] + 0.37 * (upper[axis] - lower[axis])
        for position in (through, None, upper[axis]):
            polygons, elements = mesh_section(mesh, axis, position)
            assert {len(p) for p in polygons} <= {3, 4} and len(polygons) == len(elements)
            assert sum(area(p) for p in polygons) == pytest.approx(section, rel=1e-12)
            assert len(set(elements.tolist())) == len(elements)
    on_nodes, _ = mesh_section(mesh, 2, float(layers[1]))  # a plane of faces of elements
    assert sum(area(p) for p in on_nodes) == pytest.approx(1.3, rel=1e-12)
    assert mesh_section(mesh, 2, lower[2] - 1.0)[0] == []
    with pytest.raises(ValueError, match="axis"):
        mesh_section(mesh, 3)
    fig = mesh_figure(mesh, title="a box")
    (ax,) = fig.axes
    assert ax.get_title() == "a box" and len(ax.collections[0].get_paths()) == len(
        mesh_section(mesh)[0]
    )
    coloured = mesh_figure(mesh, axis=0, values=np.arange(mesh.n_elements), label="element")
    assert len(coloured.axes) == 2  # with its colour bar


def test_models_open_with_their_examples():
    """Every model comes with the parameters and the domain of its example configuration
    (its template), which fill the forms of the interface and generate as they are."""
    models = {m["name"]: m for m in api.models()}
    assert all(m["example"] for m in models.values())
    assert models["tpms"]["example"]["volume_fraction"] == 0.2
    assert models["voronoi"]["example_domain"]["size"] == 1.0
    for name in ("tpms", "hardcore_spheres", "voronoi", "dead_leaves"):
        m = models[name]
        request = {"model": name, "params": m["example"], "domain": m["example_domain"]}
        out = api.generate({**request, "seed": 1, "resolution": 24})
        assert out["resolution"] == 24 and len(out["sections"]) in (1, 3)


def test_models_by_family():
    """The menu of the models groups them by family, in the order of the families: every
    model has one (a new model must be given its family)."""
    families = [m["family"] for m in api.models()]
    assert set(families) == set(api.MODEL_FAMILIES)
    runs = [f for k, f in enumerate(families) if k == 0 or f != families[k - 1]]
    assert runs == list(api.MODEL_FAMILIES)  # each family in one piece, in order
    assert {m["name"]: m["family"] for m in api.models()}["nacre"] == "Bio-inspired"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runs the functions of the page")
def test_forms_read_back_the_examples():
    """The page writes each example value in its field and reads it back as it is (its own
    functions, run by Node): a list typed in a JSON field ("0.012, 0.012", the tube radii of
    hierarchical_lattice) is a list again, not text that the model refuses."""
    page = (resources.files("morphogen") / "gui" / "index.html").read_text(encoding="utf-8")
    script = "\n".join(
        re.search(rf"^function {name}\(.*?^}}", page, re.M | re.S).group(0)
        for name in ("numbers", "exampleText", "readForm", "jsonless")
    )
    script += """
const fields = JSON.parse(require("fs").readFileSync(0, "utf8"));
const read = fields.map((f) => readForm({querySelectorAll: () =>
  [{dataset: {kind: f.kind, name: f.name}, value: exampleText(f.value)}]})[f.name]);
process.stdout.write(JSON.stringify(read));
"""
    fields = []
    for m in api.models():
        kinds = {p["name"]: p["kind"] for p in m["parameters"]}
        fields += [
            {"model": m["name"], "name": k, "kind": kinds[k], "value": v}
            for k, v in m["example"].items()
            if kinds[k] != "bool"
        ]
    run = subprocess.run(
        [shutil.which("node"), "-e", script],
        input=json.dumps(fields),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    for field, value in zip(fields, json.loads(run.stdout), strict=True):
        assert value == field["value"], field
    assert ["hierarchical_lattice", "tube_radius", "json"] in [
        [f["model"], f["name"], f["kind"]] for f in fields if isinstance(f["value"], list)
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runs the functions of the page")
def test_distributions_monodisperse_or_polydisperse():
    """Every parameter that is a distribution (radii, lengths, thicknesses) opens as
    monodisperse (one size) or as the law of its example (polydisperse), and the editor
    writes back the value of the example: a number, or the mapping of its law."""
    page = (resources.files("morphogen") / "gui" / "index.html").read_text(encoding="utf-8")
    script = _page_functions("esc", "numbers", "distOf", "distJSON", "distEditor", "distFields")
    script += "\n" + re.search(r"^const DIST_LAWS = \{.*?^\};", page, re.M | re.S).group(0)
    script += """
const fields = JSON.parse(require("fs").readFileSync(0, "utf8"));
const unescape = (s) => s.replace(/&quot;/g, '"').replace(/&lt;/g, "<")
  .replace(/&gt;/g, ">").replace(/&amp;/g, "&");
process.stdout.write(JSON.stringify(fields.map((f) => {
  const html = distEditor(f.name, f.value, f.default);
  const value = unescape(/type="hidden"[^>]*value="([^"]*)"/.exec(html)[1]);
  const law = /<option value="(\\w+)" selected>/.exec(html)[1];
  return [value ? JSON.parse(value) : null, law];
})));
"""
    fields = []
    for m in api.models():
        for p in m["parameters"]:
            if p["kind"] == "distribution":
                fields.append({"model": m["name"], "name": p["name"], "default": p["default"],
                               "value": m["example"].get(p["name"])})  # fmt: skip
    fields.append({"model": "-", "name": "r", "default": 1.0,
                   "value": {"type": "normal", "mean": 2.0, "std": 0.5, "low": 1.0}})  # fmt: skip
    sizes = {"type": "discrete", "values": [1.0, 2.0], "weights": [0.7, 0.3]}
    fields.append({"model": "-", "name": "r", "default": 1.0, "value": sizes})
    run = subprocess.run(
        [shutil.which("node"), "-e", script],
        input=json.dumps(fields),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    read = json.loads(run.stdout)
    assert len(fields) > 20
    for field, (value, law) in zip(fields, read, strict=True):
        expected = field["value"]
        assert value == expected, field
        if isinstance(expected, dict):
            assert law == expected["type"], field
        else:
            assert law == "constant", field
    assert {f["value"]["type"] for f in fields if isinstance(f["value"], dict)} >= {
        "normal",
        "discrete",
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runs the functions of the page")
def test_long_explanations_shortened():
    """The page shows the start of the long explanations of the tools and models (their
    first sentence, or their first words), the whole of them on demand."""
    from morphogen.gui import tools

    page = (resources.files("morphogen") / "gui" / "index.html").read_text(encoding="utf-8")
    script = re.search(r"^function shortDoc\(.*?^}", page, re.M | re.S).group(0)
    script += """
const docs = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(docs.map((d) => shortDoc(d))));
"""
    docs = [t["doc"] for t in tools.catalog()["tools"]] + [m["doc"] for m in api.models()]
    run = subprocess.run(
        [shutil.which("node"), "-e", script],
        input=json.dumps(docs),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    shortened = 0
    for doc, short in zip(docs, json.loads(run.stdout), strict=True):
        if len(doc) <= 280:
            assert short == doc
            continue
        shortened += 1
        assert len(short) <= 221 and doc.startswith(short.removesuffix("…"))
        assert short.endswith((".", "…"))
    assert shortened > 10


def _page_functions(*names: str) -> str:
    """The source of functions of the page (each from ``function name(`` at the start of a
    line to the first line that is a closing brace)."""
    page = (resources.files("morphogen") / "gui" / "index.html").read_text(encoding="utf-8")
    return "\n".join(
        re.search(rf"^function {name}\(.*?^}}", page, re.M | re.S).group(0) for name in names
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runs the functions of the page")
def test_search_of_the_page():
    """The search of the header (the functions of the page, run by Node) finds every tool
    by its title and every model by its name, first; an option of a form (a strain
    energy, the kinematics, a file format) with the field and the value to set; words by
    their start only ("paris" is not in "Comparisons"); nothing for what is not there."""
    from morphogen.gui import tools

    page = (resources.files("morphogen") / "gui" / "index.html").read_text(encoding="utf-8")
    script = _page_functions("fold", "searchIndex", "searchMatches")
    for constant in (r"^const SECTIONS = \[.*?^\];", r"^const TABS = \[.*?^\];",
                     r"^const FIXED_TABS = \{.*?^\};"):  # fmt: skip
        script += "\n" + re.search(constant, page, re.M | re.S).group(0)
    script += """
function tabOf(name) { return TABS.find((t) => t.name === name); }
const {catalog, models, queries} = JSON.parse(require("fs").readFileSync(0, "utf8"));
for (const t of catalog.tabs) {
  TABS.push({name: t.name, title: t.title, section: t.section, order: t.order});
}
const index = searchIndex(catalog, models, {export: ["surface (.stl)", "Kind"]});
const value = (d) => (d && d.value !== undefined ? d.value : null);
const found = queries.map((q) => searchMatches(index, q).map((h) =>
  [h.entry.address, h.detail ? h.detail.field || h.detail.text : null, value(h.detail)]));
process.stdout.write(JSON.stringify(found));
"""
    catalog, models = tools.catalog(), api.models()
    titles = [t["title"] for t in catalog["tools"]]
    names = [m["name"] for m in models]
    asked = {
        "arruda boyce": ["mechanics/path", "hyper_model", "arruda_boyce"],
        "Mooney-Rivlin": ["mechanics/path", "hyper_model", "mooney_rivlin"],
        "ogden": ["mechanics/path", "hyper_model", "ogden"],
        "finite strains": ["mechanics/path", "kinematics", "finite"],
        "hyperel": ["mechanics/path", "behaviour", "hyperelastic"],
        "neper": ["interop/mesh_formats", "formats", "neper"],
        "gyroid": ["generate/tpms", "surface", None],
        "PHASE  field": ["fracture/phase_field", None, None],
        "abaqus": ["interop/abaqus_input", None, None],
        "paris": ["fracture/crack_growth", "paris_C", None],
        "stl": ["export", "surface (.stl)", None],
        "export": ["export", None, None],
    }
    nothing = ["xyzzy", "", "  ", "omparisons"]
    queries = titles + names + list(asked) + nothing
    run = subprocess.run(
        [shutil.which("node"), "-e", script],
        input=json.dumps({"catalog": catalog, "models": models, "queries": queries}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    found = json.loads(run.stdout)
    assert len(found) == len(queries)
    for t, hits in zip(catalog["tools"], found, strict=False):
        assert hits[0][0] == f"{t['tab']}/{t['name']}", t["title"]
    for name, hits in zip(names, found[len(titles) :], strict=False):
        assert hits[0][0] == f"generate/{name}", name
    for (query, first), hits in zip(asked.items(), found[len(titles) + len(names) :], strict=False):
        assert hits[0] == first, query
    assert found[-len(nothing) :] == [[]] * len(nothing)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runs the functions of the page")
def test_docstrings_as_running_text():
    """The page shows the documentation of a model as running text: the lines of a
    paragraph joined, the paragraphs kept, the marks of code and the roles removed."""
    script = _page_functions("plainDoc")
    script += """
const docs = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(docs.map(plainDoc)));
"""
    docs = [
        "Boolean model: a union\nof ``grains``.\n\n    Give the fraction\n    or the intensity.\n",
        "See :func:`~morphogen.models.x.y` and :class:`Model`\n \nthen ``a``.",
    ] + [m["doc"] for m in api.models()]
    run = subprocess.run(
        [shutil.which("node"), "-e", script],
        input=json.dumps(docs),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    plain = json.loads(run.stdout)
    assert plain[0] == "Boolean model: a union of grains.\n\nGive the fraction or the intensity."
    assert plain[1] == "See morphogen.models.x.y and Model\n\nthen a."
    for doc, text in zip(docs[2:], plain[2:], strict=True):
        assert "``" not in text and ":`" not in text
        assert all(line == line.strip() and line for line in text.split("\n\n"))
        assert "\n\n\n" not in text and not re.search(r"[^\n]\n[^\n]", text)
        assert (
            text.replace("\n\n", " ").split()
            == re.sub(r":\w+:`~?([^`]*)`", r"\1", doc.replace("``", "")).split()
        )
    # the summary of a model: the first paragraph of its documentation, on one line
    for m in api.models():
        assert m["summary"] and "\n" not in m["summary"] and "``" not in m["summary"]
        assert m["summary"].split() == m["doc"].split("\n\n")[0].replace("``", "").split()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runs the functions of the page")
def test_figures_given_as_data_urls_are_shown():
    """The page shows the files of a result inline ("?inline") and the images that tools give
    as data URLs as they are: the sections of the morphologies made by Reconstruction and
    Automaton rule were broken images, their data URLs ended by "?inline"."""
    page = (resources.files("morphogen") / "gui" / "index.html").read_text(encoding="utf-8")
    script = "\n".join(
        re.search(rf"^function {name}\(.*?^}}", page, re.M | re.S).group(0)
        for name in ("esc", "media")
    )
    script += "process.stdout.write(media(JSON.parse(require('fs').readFileSync(0, 'utf8'))));"
    data = "data:image/png;base64,iVBORw0KGgo="
    figures = [
        {"title": "x-y section", "url": data},
        {"title": "covariance", "url": "/files/k1/covariance.png"},
        {"title": "growth", "url": "/files/k1/growth.mp4", "format": "mp4"},
    ]
    run = subprocess.run(
        [shutil.which("node"), "-e", script],
        input=json.dumps(figures),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    html = run.stdout
    assert f'<img src="{data}"' in html and f"{data}?inline" not in html
    assert '<img src="/files/k1/covariance.png?inline"' in html
    assert '<video src="/files/k1/growth.mp4?inline"' in html


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runs the functions of the page")
def test_chips_of_the_workspace_close():
    """Each chip of the workspace has its close button (the cross, named for readers of the
    screen), which the strip tells from a click on the chip itself; the item just closed
    stays in the strip with the undo of its close."""
    page = (resources.files("morphogen") / "gui" / "index.html").read_text(encoding="utf-8")
    script = _page_functions("esc", "chip", "closedChip")
    script += "\n" + re.search(r"^const KIND = \{.*?\};", page, re.M).group(0)
    script += """
const items = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify([...items.map(chip), closedChip(items[0])]));
"""
    items = [
        {"key": "a1", "kind": "morphology", "title": "#1 <voronoi>", "thumbnail": None},
        {"key": "b2", "kind": "simulation", "title": "#2 tension", "thumbnail": "data:x"},
    ]
    run = subprocess.run(
        [shutil.which("node"), "-e", script],
        input=json.dumps(items),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    first, second, closed = json.loads(run.stdout)
    assert 'data-key="a1"' in first and 'data-close="a1"' in first
    assert 'aria-label="close #1 &lt;voronoi&gt;"' in first and "<voronoi>" not in first
    assert first.count("<button") == 1 and 'type="button" class="close"' in first
    assert 'data-close="b2"' in second and '<img src="data:x"' in second
    assert 'id="ws-undo"' in closed and "#1 &lt;voronoi&gt; closed" in closed
    # the clicks of the strip: on the cross, a close; elsewhere on the chip, its menu
    assert 'e.target.closest("button.close")' in page and "closeItem(c.dataset.key)" in page
    assert '"/api/workspace/restore"' in page


def test_interface_actions(tmp_path):
    names = {m["name"] for m in api.models()}
    assert names == set(mg.list_models())
    session = api.Session(tmp_path)
    out = api.generate(
        {
            "model": "boolean_spheres",
            "params": {"radius": 1.0, "volume_fraction": 0.3, "dummy_empty": ""},
            "domain": {"size": 6.0},
            "seed": 1,
            "resolution": 64,
        },
        session,
    )
    # the sections of the level set, its measures, the morphology kept in the workspace
    assert len(out["sections"]) == 3 and out["sections"][0]["png"].startswith("data:image/png")
    assert out["level_set"] and "volume_fraction" in out["theory"]
    assert 0.15 < out["measures"]["volume_fraction (level set)"] < 0.45
    assert out["item"]["kind"] == "morphology" and out["item"]["title"].startswith("#1 ")
    key = out["files"]["geometry"].split("/")[2]
    assert session.file(key, "microstructure.json").exists()
    assert [i["key"] for i in session.workspace()] == [key]
    with pytest.raises(FileNotFoundError):
        session.file(key, "../../etc/passwd")
    # a large section is a file of the session, without a voxel image of the volume
    big = api.generate(
        {"model": "tpms", "params": {"volume_fraction": 0.3}, "domain": {"size": 1.0},
         "resolution": 1024},
        session,
    )  # fmt: skip
    assert big["sections"][0]["png"].endswith("?inline")
    assert session.file(big["item"]["key"], "section_yz.png").stat().st_size > 1000
    grains = api.generate(
        {"model": "voronoi", "params": {"n_cells": 10}, "domain": {"size": 4.0}, "resolution": 32}
    )
    assert len(grains["labels"]) == 3 and grains["warnings"] == []
    # the warnings of a generation are shown with its results
    plywood = api.generate(
        {"model": "bouligand", "params": {}, "domain": {"size": 1.0}, "seed": 1, "resolution": 32}
    )
    assert "half-turns" in plywood["warnings"][0]
    evo = api.formation(
        {
            "model": "cellular_automaton",
            "params": {"steps": 6},
            "domain": {"size": [16, 16, 1]},
            "frames": 3,
            "seed": 2,
        }
    )
    assert 2 <= len(evo["frames"]) <= 5 and len(evo["changes"]) == len(evo["frames"]) - 1
    assert api.evolve is api.formation  # the former name
    with pytest.raises(ValueError, match="formation"):
        api.formation(
            {
                "model": "boolean_spheres",
                "params": {"radius": 1.0, "volume_fraction": 0.2},
                "domain": {"size": 4.0},
            }
        )
    count = api.count({"tiles": "polarized", "shape": [2, 2, 2]})
    assert count["count"] == "450" and len(count["assembly"]) == 3
    with pytest.raises(ValueError, match="resolution"):
        api.generate(
            {
                "model": "boolean_spheres",
                "params": {"radius": 1.0, "volume_fraction": 0.2},
                "domain": {"size": 4.0},
                "resolution": 5000,
            }
        )
    job = api.OptimizationJob(
        {
            "grid": [16, 16, 1],
            "volume_fraction": 0.4,
            "settings": {"method": "beso", "iterations": 15},
        },
        session,
    )
    job._run()  # synchronously
    status = job.status()
    assert status["state"] == "done" and status["result"]["properties"]["bulk_modulus"] > 0
    assert status["result"]["refined_properties"]["bulk_modulus"] > 0
    assert status["result"]["hinges"] >= 0
    assert status["png"].startswith("data:image/png") and len(status["objective"]) >= 2
    assert status["result"]["item"]["kind"] == "morphology"  # the design, in the workspace
    with pytest.raises(ValueError, match="grid"):
        api.OptimizationJob({"grid": [128, 128, 32], "volume_fraction": 0.4})
    with pytest.raises(ValueError, match="free memory"):
        api.check_memory(1e18, "a voxel image")


#: The output of macOS's vm_stat on an arm64 machine (its first lines)
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                3711.
Pages active:                            185612.
Pages inactive:                          181927.
Pages speculative:                         3226.
Pages throttled:                              0.
Pages wired down:                        120473.
Pages purgeable:                           7071.
"Translation faults":                 437468712.
Pages copy-on-write:                   15462337.
"""


def test_morphology_of_several_phases(tmp_path):
    """A morphology of several phases in the interface: the sections of the implicit
    functions of its phases, and the export of the function and of the surface of each
    phase (the surface of the last being that of the others together)."""
    from morphogen import io
    from morphogen.models.implicit import implicit_functions

    session = api.Session(tmp_path)
    inner = {"type": "hardcore_spheres", "radius": 0.2, "volume_fraction": 0.2}
    out = api.generate(
        {"model": "coating", "params": {"model": inner, "thickness": 0.05},
         "domain": {"size": 1.0}, "seed": 3, "resolution": 64},
        session,
    )  # fmt: skip
    assert out["level_set"] and out["implicit_phases"] == 3
    info = out["item"]["info"]
    assert info["implicit functions"] == "of its 3 phases" and "level set" not in info
    assert len(out["item"]["phases"]) == 3
    assert {"fraction_0 (64 voxels)", "fraction_2 (64 voxels)"} <= set(out["measures"])
    key = out["item"]["key"]
    micro = session.items[key]["micro"]
    exported = api.export(
        {"source": key, "outputs": ["level_set", "surface"], "shape": 24}, session
    )
    files = {f["name"] for f in exported["files"]}
    assert files == {"export.zip", "surface_phase0.stl", "surface_phase1.stl"} | {
        f"level_set_phase{k}.vti" for k in range(3)
    }
    folder = session.file(exported["archive"].split("/")[2], "export.zip").parent
    image = micro.voxelize(shape=24).data
    for k in range(3):  # negative in its phase, at the centres of the voxels
        level = io.load_image(folder / f"level_set_phase{k}.vti").data
        assert level.shape == image.shape and ((level < 0) == (image == k)).all()
    for k in range(2):  # closed on the faces of the box: it encloses the volume of the phase
        surface = io.read_stl(folder / f"surface_phase{k}.stl")
        assert len(surface.faces) > 100
        assert abs(surface.volume() - float(np.mean(image == k))) < 0.02
    # a morphology of two phases: its level set alone
    spheres = api.generate(
        {"model": "boolean_spheres", "params": {"radius": 0.2, "volume_fraction": 0.3},
         "domain": {"size": 1.0}, "seed": 1, "resolution": 32},
        session,
    )  # fmt: skip
    assert spheres["level_set"] and spheres["implicit_phases"] == 2
    assert spheres["item"]["info"]["level set"] is True
    two = api.export({"source": spheres["item"]["key"], "outputs": ["level_set", "surface"],
                      "shape": 16}, session)  # fmt: skip
    assert {f["name"] for f in two["files"]} == {"export.zip", "level_set.vti", "surface.stl"}
    both = session.items[spheres["item"]["key"]]["micro"]
    assert [suffix for suffix, _ in implicit_functions(both, shape=8)] == [""]
    assert [suffix for suffix, _ in implicit_functions(micro, shape=8)] == [
        "_phase0", "_phase1", "_phase2"
    ]  # fmt: skip


def test_free_memory_of_vm_stat(monkeypatch):
    """macOS has no SC_AVPHYS_PAGES: its free memory is that of vm_stat (the free,
    speculative and inactive pages, as psutil counts them), which the guard of the
    interface compares; an output without them, or vm_stat failing, leaves it unknown."""
    available = 16384.0 * (3711 + 181927 + 3226)
    assert api._vm_stat_available(VM_STAT) == available
    assert api._vm_stat_available("Mach Virtual Memory Statistics:") is None

    def failing():
        raise subprocess.CalledProcessError(1, ["vm_stat"])

    with monkeypatch.context() as m:
        m.setattr(api.sys, "platform", "darwin")
        m.setattr(api, "_vm_stat", lambda: VM_STAT)
        assert api.available_memory() == available
        with pytest.raises(ValueError, match="free memory"):
            api.check_memory(1e18, "a voxel image")
        m.setattr(api, "_vm_stat", failing)
        assert api.available_memory() is None


def test_interface_server(tmp_path):
    server = GuiServer(("127.0.0.1", 0), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=60) as r:
            return r.status, r.read()

    def post(path, data):
        req = urllib.request.Request(
            base + path,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    try:
        status, page = get("/")
        assert status == 200 and b"morphoGen" in page and b"/api/models" in page
        assert get("/favicon.ico")[0] == 204
        models = json.loads(get("/api/models")[1])
        assert any(m["name"] == "tpms" for m in models)
        out = post(
            "/api/generate",
            {
                "model": "tpms",
                "params": {"surface": "gyroid", "cell_size": 2.0, "volume_fraction": 0.3},
                "domain": {"size": 2.0},
                "resolution": 32,
            },
        )
        status, data = get(out["files"]["geometry"])
        assert status == 200 and json.loads(data)["kind"]
        items = json.loads(get("/api/workspace")[1])
        assert [i["key"] for i in items] == [out["item"]["key"]]
        renamed = post("/api/workspace/rename", {"key": out["item"]["key"], "title": "gyroid"})
        assert renamed["title"] == "#1 gyroid" and renamed["kind"] == "morphology"
        exported = post(
            "/api/export",
            {"source": out["item"]["key"], "outputs": ["geometry", "image"], "shape": 16},
        )
        status, data = get(exported["archive"])
        assert status == 200 and data[:2] == b"PK"
        sim = post(
            "/api/simulate",
            {"source": out["item"]["key"], "method": "fft", "shape": 12,
             "materials": {"0": "void", "1": {"young": 1.0, "poisson": 0.3}}},
        )  # fmt: skip
        server.jobs[sim["job"]].thread.join(timeout=300)
        state = json.loads(get(f"/api/job/{sim['job']}")[1])
        assert state["state"] == "done", state.get("error")
        assert len(json.loads(get("/api/workspace")[1])) == 2
        job = post(
            "/api/optimize",
            {
                "grid": [12, 12, 1],
                "volume_fraction": 0.4,
                "settings": {"method": "hca", "iterations": 5},
            },
        )
        server.jobs[job["job"]].thread.join(timeout=120)
        state = json.loads(get(f"/api/job/{job['job']}")[1])
        assert state["state"] == "done"
        with pytest.raises(urllib.error.HTTPError) as err:
            post("/api/generate", {"model": "nope"})
        assert err.value.code == 400
        sim = post("/api/simulate", {**SIMULATION, "video": "none"})
        server.jobs[sim["job"]].thread.join(timeout=300)
        state = json.loads(get(f"/api/job/{sim['job']}")[1])
        assert state["state"] == "done", state.get("error")
        figure = state["result"]["figures"][0]["url"]
        with urllib.request.urlopen(base + figure + "?inline", timeout=60) as r:
            assert r.headers["Content-Type"] == "image/png"
            assert r.headers["Content-Disposition"].startswith("inline")
        with urllib.request.urlopen(base + figure, timeout=60) as r:
            assert r.headers["Content-Disposition"].startswith("attachment")
        with pytest.raises(urllib.error.HTTPError) as err:
            post("/api/simulate", {**SIMULATION, "method": "bem"})
        assert err.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as err:
            get("/api/job/unknown")
        assert err.value.code == 404
        # a form of another site (text/plain) is refused
        form = urllib.request.Request(
            base + "/api/workspace/save", data=b'{"directory": "x"}',
            headers={"Content-Type": "text/plain"},
        )  # fmt: skip
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(form, timeout=30)
        assert err.value.code == 415
    finally:
        server.shutdown()
        server.server_close()


CAN_RENDER = vtk_can_render()  # VTK, with a display, EGL or OSMesa
SIMULATION = {
    "model": "hardcore_spheres",
    "params": {"radius": 0.8, "volume_fraction": 0.2},
    "domain": {"size": 5.0},
    "seed": 3,
    "method": "fft",
    "property": "shear_modulus",
    "shape": 12,
    "materials": {"0": {"young": 1.0, "poisson": 0.3}, "1": {"young": 20.0, "poisson": 0.3}},
    "field": "stress_12",
    "cmap": "coolwarm",
    "video": "field",
    "frames": 4,
}


def test_interface_simulation(tmp_path):
    """The whole chain from the form: a configuration run, its views, video and files,
    the configuration downloadable, and views redrawn without solving again."""
    pytest.importorskip("matplotlib")
    from morphogen.config import ConfigError, load_config

    session = api.Session(tmp_path)
    job = api.SimulationJob(SIMULATION, session)
    job._run()  # synchronously
    status = job.status()
    assert status["state"] == "done", status.get("error")
    r = status["result"]
    assert 1.0 / 2.6 < r["values"]["shear_modulus.periodic"] < 20.0 / 2.6  # between the phases
    titles = [f["title"] for f in r["figures"]]
    assert titles == ["image sections", "shear modulus periodic stress 12"]
    assert len(r["videos"]) == 1 and r["videos"][0]["format"] in ("mp4", "gif")
    assert {"data", "table", "figure", "video", "configuration"} <= {f["kind"] for f in r["files"]}
    folder = session.file(r["key"], "configuration.yaml").parent
    rerun = load_config(folder / "configuration.yaml")  # rerun from the command line
    assert rerun.outputs.directory == folder / "simulation" and rerun.visualization is not None
    again = api.render(
        {"key": r["key"], "field": "von_mises", "cmap": "inferno", "video": "field", "frames": 3},
        session,
    )
    assert again["figure"]["url"].endswith("/render_1.png")
    assert again["video"]["title"] == "von_mises (field)"
    with pytest.raises(ValueError, match="nodes of finite element"):
        api.render({"key": r["key"], "field": "displacement"}, session)
    with pytest.raises(ValueError, match="property"):
        api.simulation_config({**SIMULATION, "property": "stiffness"}, tmp_path)
    with pytest.raises(ValueError, match="colormap"):
        api.simulation_config({**SIMULATION, "cmap": "rainbowish"}, tmp_path)
    with pytest.raises(ConfigError):
        api.SimulationJob({**SIMULATION, "params": {"radius": -1.0}}, session)
    if CAN_RENDER:  # finite elements: the mesh, the field on the deformed mesh, a turntable
        fe = api.SimulationJob(
            {
                **SIMULATION,
                "method": "fe",
                "element_size": 1.6,
                "video": "turntable",
                "section_axis": "z",
                "magnification": 0.2,
            },
            session,
        )
        fe._run()
        done = fe.status()
        assert done["state"] == "done", done.get("error")
        assert [f["title"] for f in done["result"]["figures"]] == [
            "mesh",
            "shear modulus periodic stress 12",
        ]  # no voxels: the conforming mesh of the level set
        cut = api.render(
            {
                "key": done["result"]["key"],
                "field": "displacement",
                "section_axis": "x",
                "section_fraction": 0.3,
            },
            session,
        )
        assert cut["figure"]["title"] == "displacement"


def test_interface_workspace(tmp_path):
    """The workflow of the interface: a morphology generated, simulated from the workspace
    (its voxels; its conforming mesh), exported (its geometry, sections, voxel images,
    surface and a video of its sections; the view, tables and configuration of a
    simulation) and optimized from a simulation, whose design joins the workspace and is
    simulated in turn."""
    from morphogen.mesh.conforming import has_mmg

    session = api.Session(tmp_path)
    spheres = api.generate(
        {"model": "hardcore_spheres", "params": {"radius": 0.8, "volume_fraction": 0.2},
         "domain": {"size": 5.0}, "seed": 3, "resolution": 48},
        session,
    )["item"]  # fmt: skip
    materials = {"0": {"young": 1.0, "poisson": 0.3}, "1": {"young": 20.0, "poisson": 0.3}}
    job = api.SimulationJob(
        {"source": spheres["key"], "method": "fft", "property": "bulk_modulus", "shape": 16,
         "materials": materials},
        session,
    )  # fmt: skip
    job._run()
    done = job.status()
    assert done["state"] == "done", done.get("error")
    simulation = done["result"]["item"]
    assert simulation["title"] == "#2 bulk modulus of #1, FFT"
    assert simulation["source"] == spheres["key"] and simulation["property"] == "bulk_modulus"
    assert 1.0 < done["result"]["values"]["bulk_modulus.periodic"] < 20.0
    # the fields that the export draws: those of the files of the simulation
    fields = api.simulation_fields({"key": simulation["key"]}, session)["fields"]
    assert fields[:2] == ["von_mises", "energy_density"] and "stress_12" in fields
    # the simulations of the session in a database, for the Ashby charts
    db = tmp_path / "session.db"
    assert api.database_add_session({"database": str(db)}, session) == {"added": 1, "runs": 1}
    assert api.ashby({"database": str(db), "x": "solid_fraction",
                      "y": "bulk_modulus.periodic.relative"})["matched"] == 1  # fmt: skip
    # the configuration reruns the same realization (the realization 0 of its seed)
    from morphogen.config import load_config
    from morphogen.seeding import derive_seed

    rerun = load_config(job.folder / "configuration.yaml")
    again = rerun.model.generate(rerun.domain, derive_seed(rerun.seed, 0))
    np.testing.assert_array_equal(
        again.voxelize(shape=16).data, session.item(spheres["key"])["micro"].voxelize(shape=16).data
    )
    pytest.importorskip("skimage")
    files = api.export(
        {"source": spheres["key"], "resolution": 64, "shape": 24, "video": "sweep", "frames": 3,
         "outputs": ["geometry", "sections", "level_set", "image", "surface"]},
        session,
    )  # fmt: skip
    names = {f["name"] for f in files["files"]}
    assert {"microstructure.json", "level_set.vti", "image.vti", "surface.stl"} <= names
    assert {"section_xy.png", "export.zip"} <= names and len(files["videos"]) == 1
    table = api.export(
        {"source": simulation["key"], "outputs": ["figure", "tables", "configuration"],
         "field": "stress_11", "formats": ["png", "svg"]},
        session,
    )  # fmt: skip
    names = {f["name"] for f in table["files"]}
    assert {"view.png", "view.svg", "results.csv", "results.xlsx", "configuration.yaml"} <= names
    with pytest.raises(ValueError, match="outputs"):
        api.export({"source": simulation["key"], "outputs": ["stl"]}, session)
    with pytest.raises(ValueError, match="workspace"):
        api.export({"source": "nothing"}, session)
    # an optimization from the simulation: its morphology, its solid material, its fraction
    opt = api.OptimizationJob(
        {"start": simulation["key"], "grid": [16, 16, 1],
         "settings": {"method": "oc", "iterations": 6, "symmetry": "none"}},
        session,
    )  # fmt: skip
    assert opt.optimization.material.young == 20.0
    assert 0.05 < opt.optimization.volume_fraction < 0.4
    opt._run()
    design = opt.status()["result"]["item"]
    assert design["title"] == "#3 design (bulk modulus) from #2"
    stiffness, compliance = opt.status()["result"]["tables"]  # C and S = C⁻¹ of the design
    assert stiffness["title"].startswith("stiffness C_ij of the thresholded design")
    assert compliance["title"].startswith("compliance S_ij = C⁻¹ of the thresholded design")
    files = [session.root / opt.status()["result"]["item"]["key"] / f"{t}.csv"
             for t in ("stiffness", "compliance")]  # fmt: skip
    c, s = (np.loadtxt(f, delimiter=",", skiprows=1, usecols=range(1, 7)) for f in files)
    assert np.allclose(s @ c, np.eye(6), atol=1e-8) and c[0, 0] == pytest.approx(
        stiffness["rows"][0][1], rel=1e-5
    )
    assert design["phases"] == {"0": "void", "1": "solid"} and design["planar"]
    method = "fe" if has_mmg() and CAN_RENDER else "fft"  # the first view of a mesh: VTK
    sim = api.SimulationJob(
        {"source": design["key"], "method": method, "property": "bulk_modulus",
         "element_size": 1.0, "shape": 16,
         "materials": {"0": "void", "1": {"young": 20.0, "poisson": 0.3}}},
        session,
    )  # fmt: skip
    sim._run()
    assert sim.status()["state"] == "done", sim.status().get("error")
    assert not (sim.folder / "configuration.yaml").exists()  # a design is not a model
    assert [i["number"] for i in session.workspace()] == [4, 3, 2, 1]


def test_interface_database_and_ashby(tmp_path):
    from morphogen.results import ResultsDatabase

    db = tmp_path / "r.db"
    with ResultsDatabase(db) as d:
        for family, n in (("octet", 1.0), ("kelvin", 2.0)):
            rows = [
                {"realization": k, "solid_fraction": phi, "bulk_modulus.periodic.relative": phi**n}
                for k, phi in enumerate((0.1, 0.2, 0.4))
            ]
            d.add_records(rows, name=family, config={"model": {"type": family}})
    info = api.database_info({"database": str(db)})
    assert len(info["runs"]) == 2 and info["records"] == 6
    assert "solid_fraction" in info["numeric"] and "model.type" in info["text"]
    out = api.ashby({"database": str(db), "by": "model.type", "where": "solid_fraction > 0.15"})
    assert out["x"] == "solid_fraction" and out["guides"] == [1.0, 2.0] and out["matched"] == 4
    assert {g["name"] for g in out["groups"]} == {"octet", "kelvin"}
    assert out["y_label"].startswith("Relative bulk modulus")
    linear = api.ashby({"database": str(db), "log": False, "guides": [1]})
    assert linear["guides"] == [1.0] and len(linear["groups"]) == 2
    with pytest.raises(ValueError, match="no record"):
        api.ashby({"database": str(db), "where": ["solid_fraction > 1"]})
    with pytest.raises(ValueError, match="no database"):
        api.database_info({"database": str(tmp_path / "none.db")})
    with pytest.raises(ValueError, match="directory"):
        api.database_add({"database": str(db), "directory": ""})
    with pytest.raises(ValueError, match="database"):
        api.database_info({})
    # through the server
    server = GuiServer(("127.0.0.1", 0), tmp_path / "session")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        req = urllib.request.Request(
            base + "/api/ashby",
            data=json.dumps({"database": str(db)}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            assert len(json.loads(r.read())["groups"]) == 2  # families: the models
        with urllib.request.urlopen(base + "/", timeout=60) as r:
            page = r.read()
        assert b"tab-ashby" in page and b"prefers-color-scheme: dark" in page
    finally:
        server.shutdown()


def test_fft_simulation_on_the_gpu(tmp_path):
    """The option of the tab Simulate: the FFT solver on the GPU gives the modulus of the
    CPU (skipped without CuPy and a CUDA device)."""
    from morphogen import gpu

    if not gpu.available():
        pytest.skip("no GPU backend (CuPy, CUDA)")
    session = api.Session(tmp_path)
    key = api.generate(
        {"model": "hardcore_spheres", "params": {"radius": 0.8, "volume_fraction": 0.2},
         "domain": {"size": 5.0}, "seed": 3, "resolution": 32},
        session,
    )["item"]["key"]  # fmt: skip
    values = {}
    for device in ("cpu", "gpu"):
        job = api.SimulationJob(
            {"source": key, "method": "fft", "property": "bulk_modulus", "shape": 24,
             "device": device, "materials": {"0": {"young": 1.0, "poisson": 0.3},
                                             "1": {"young": 20.0, "poisson": 0.3}}},
            session,
        )  # fmt: skip
        job._run()
        assert job.status()["state"] == "done", job.status().get("error")
        values[device] = job.status()["result"]["values"]["bulk_modulus.periodic"]
    assert values["gpu"] == pytest.approx(values["cpu"], rel=1e-6)
    with pytest.raises(ValueError, match="device"):
        api.simulation_config({**SIMULATION, "device": "tpu"}, tmp_path)


def test_formation_video_export(tmp_path):
    """The export of a structure grown by a process: the video of its formation; other
    models have none."""
    session = api.Session(tmp_path)
    grown = api.generate(
        {"model": "cellular_automaton", "params": {"steps": 6, "cell_size": 0.5},
         "domain": {"size": [8, 8, 0.5]}, "seed": 2, "resolution": 32},
        session,
    )["item"]["key"]  # fmt: skip
    out = api.export({"source": grown, "video": "formation", "frames": 4}, session)
    assert len(out["videos"]) == 1 and out["videos"][0]["title"] == "formation"
    plain = api.generate({"model": "tpms", "params": {"volume_fraction": 0.3},
                          "domain": {"size": 1.0}, "resolution": 24}, session)  # fmt: skip
    with pytest.raises(ValueError, match="formation"):
        api.export({"source": plain["item"]["key"], "video": "formation"}, session)


def test_workspace_saved_and_opened(tmp_path):
    """A workspace written to a folder and opened in another session: its morphologies
    (their microstructures), its simulations (their files) and their links come back
    under new keys, and the simulation exports in the new session."""
    first = api.Session(tmp_path / "first")
    key = api.generate(
        {"model": "hardcore_spheres", "params": {"radius": 0.8, "volume_fraction": 0.2},
         "domain": {"size": 5.0}, "seed": 3, "resolution": 32},
        first,
    )["item"]["key"]  # fmt: skip
    job = api.SimulationJob(
        {"source": key, "method": "fft", "property": "bulk_modulus", "shape": 12,
         "materials": {"0": {"young": 1.0, "poisson": 0.3}, "1": {"young": 20.0, "poisson": 0.3}}},
        first,
    )  # fmt: skip
    job._run()
    saved = api.save_workspace({"directory": str(tmp_path / "saved")}, first)
    assert saved["saved"] == 2 and (tmp_path / "saved" / "workspace.json").is_file()
    second = api.Session(tmp_path / "second")
    opened = api.open_workspace({"directory": saved["directory"]}, second)
    morph, sim = opened["items"]
    assert morph["title"] == "#1 hardcore_spheres (seed 3)" and morph["key"] != key
    assert sim["source"] == morph["key"] and sim["title"] == "#2 bulk modulus of #1, FFT"
    micro = second.item(morph["key"])["micro"]
    np.testing.assert_array_equal(
        micro.voxelize(shape=16).data, first.item(key)["micro"].voxelize(shape=16).data
    )
    out = api.export({"source": sim["key"], "outputs": ["tables"]}, second)
    assert "results.csv" in {f["name"] for f in out["files"]}
    with pytest.raises(ValueError, match="no saved workspace"):
        api.open_workspace({"directory": str(tmp_path)}, second)
    with pytest.raises(ValueError, match="empty"):
        api.save_workspace({"directory": str(tmp_path / "none")}, api.Session(tmp_path / "x"))
