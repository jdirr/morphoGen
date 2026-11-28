"""The tutorials of the documentation (docs/tutorial.md, docs/tutorial_rve.md,
docs/tutorial_rubber.md, docs/tutorial_geometry.md, docs/tutorial_iga.md,
docs/tutorial_contact.md, docs/tutorial_fracture.md, docs/tutorial_texture.md,
docs/tutorial_morphology.md, docs/tutorial_plane.md, docs/tutorial_fibres.md) stay correct:
the Python snippets of each run one after another in one session, at smaller sizes; the
configuration of the
first runs with ``morphogen run``; each of their ``morphogen`` commands is one that the
command line parses; the figures that the pages show are there
(docs/tutorials/make_figures.py)."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import numpy as np
import pytest
import yaml

from morphogen import cli

DOCS = Path(__file__).resolve().parents[1] / "docs"
TUTORIAL = DOCS / "tutorial.md"
RVE = DOCS / "tutorial_rve.md"
RUBBER = DOCS / "tutorial_rubber.md"
GEOMETRY = DOCS / "tutorial_geometry.md"
IGA = DOCS / "tutorial_iga.md"
CONTACT = DOCS / "tutorial_contact.md"
FRACTURE = DOCS / "tutorial_fracture.md"
TEXTURE = DOCS / "tutorial_texture.md"
MORPHOLOGY = DOCS / "tutorial_morphology.md"
PLANE = DOCS / "tutorial_plane.md"
FIBRES = DOCS / "tutorial_fibres.md"
if not DOCS.is_dir():  # an installed package without its sources
    pytest.skip("the documentation is not here", allow_module_level=True)

#: The sizes of the snippets made smaller for the test; each text appears once in the
#: tutorial, so that a change of the snippets shows here
SMALLER = {
    "image = micro.voxelize(shape=64)": "image = micro.voxelize(shape=24)",
    '"voxels": {"shape": 64}': '"voxels": {"shape": 24}',
    "percolation_threshold(family, 0.15, 0.45, seed=0)": (
        "percolation_threshold(family, 0.15, 0.45, realizations=3, tol=0.05, seed=0)"
    ),
    "for size in (4.0, 6.0, 8.0):": "for size in (3.0, 4.0):",
    "for r in range(16):": "for r in range(3):",
    'LoadingPath.parse("t=0,1 n=10 E11=0,0.01")': 'LoadingPath.parse("t=0,1 n=3 E11=0,0.01")',
    "coarse = micro.voxelize(shape=16)": "coarse = micro.voxelize(shape=8)",
    "tiny = micro.voxelize(shape=8)": "tiny = micro.voxelize(shape=4)",
    "learning.random_paths(32, segments=2, increments=4, seed=0)": (
        "learning.random_paths(4, segments=1, increments=2, seed=0)"
    ),
    "epochs=100, batch=8": "epochs=2, batch=8",
}
#: Those of the tutorial of the rubber: a coarser composite, a shorter strip
SMALLER_RUBBER = {
    "image = micro.voxelize(shape=16)": "image = micro.voxelize(shape=8)",
    '"increments": 10, "strain": {"11": [0, 0.5]}': '"increments": 3, "strain": {"11": [0, 0.5]}',
    '"increments": 200,': '"increments": 50,',
    "nx, ny, critical = 36, 12, 1.6": "nx, ny, critical = 24, 8, 1.6",
}
#: Those of the tutorial of the representative volume: a smaller study of its
#: configuration (the statistics of the volume fractions take seconds as they are)
SMALLER_RVE = {
    "sizes: [8.0, 12.0, 16.0]": "sizes: [6.0, 8.0, 10.0]",
    "realizations: [32, 16, 8]": "realizations: [6, 4, 3]",
}
#: Those of the tutorial of the geometry: coarser sections, surfaces, meshes and grids
SMALLER_GEOMETRY = {
    "resolution=900": "resolution=120",
    "for n in (24, 48, 96):": "for n in (12, 24):",
    'mg.mesh.conforming_mesh(micro, size=0.5, method="grid")': (
        'mg.mesh.conforming_mesh(micro, size=0.9, method="grid")'
    ),
    "for n in (32, 64)}": "for n in (12, 16)}",
    "LevelSetSample.of(spheres_alone, 16)": "LevelSetSample.of(spheres_alone, 8)",
    "spheres_alone.voxelize(shape=64)": "spheres_alone.voxelize(shape=16)",
}
#: Those of the tutorial of the isogeometric analysis: fewer sheets, a coarser design
SMALLER_IGA = {
    "for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9):": "for a in (0.3, 0.6, 0.9):",
    "design = opt.ShapeDesign(grains, subdivisions=4, elevate=(1, 2))": (
        "design = opt.ShapeDesign(grains, subdivisions=3, elevate=(0, 1))"
    ),
    "auxetic.design.voxelize((256, 256),": "auxetic.design.voxelize((128, 128),",
}
#: Those of the tutorial of contact and damage: a coarser composite, fewer steps
SMALLER_CONTACT = {
    "image = micro.voxelize(shape=10)": "image = micro.voxelize(shape=8)",
    '"tension": {"times": [0, 1], "increments": 15,': (
        '"tension": {"times": [0, 1], "increments": 6,'
    ),
}
#: Those of the tutorial of the textures: a shorter tension, the first grains of the
#: polycrystal of Taylor, a smaller polycrystal of space
SMALLER_TEXTURE = {
    'path = {"times": [0, 20], "increments": 10, "strain": {"11": 0.02}}': (
        'path = {"times": [0, 4], "increments": 3, "strain": {"11": 0.004}}'
    ),
    "    for e in grains.orientations\n": "    for e in grains.orientations[:6]\n",
    "after = texture.inverse_pole_figure(final, axis, weights=grains.areas)": (
        "after = texture.inverse_pole_figure(final, axis, weights=grains.areas[:6])"
    ),
    "    n_cells=300,": "    n_cells=60,",
    'cells = {"grains": polycrystal.labels(shape=24).data,': (
        'cells = {"grains": polycrystal.labels(shape=12).data,'
    ),
    "found = homogenize(polycrystal.voxelize(shape=24),": (
        "found = homogenize(polycrystal.voxelize(shape=12),"
    ),
}
#: Those of the tutorial of mathematical morphology: fewer probes, a coarser image
SMALLER_MORPHOLOGY = {
    'balls = mm.capacity(micro, "ball", radii, points=2**16)': (
        'balls = mm.capacity(micro, "ball", radii, points=2**14)'
    ),
    'segments = mm.capacity(micro, "segment", lengths, points=2**14)': (
        'segments = mm.capacity(micro, "segment", lengths, points=2**12)'
    ),
    "image = micro.voxelize(shape=64)": "image = micro.voxelize(shape=40)",
    "for s in range(8)\n": "for s in range(3)\n",
}
#: Those of the tutorial of the plane: a coarser sheet, a shorter path
SMALLER_PLANE = {
    "mesh = mg.mesh.plane_mesh(sheet, size=0.025)": "mesh = mg.mesh.plane_mesh(sheet, size=0.05)",
    '"increments": 5, "strain": {"11": 0.01}': '"increments": 2, "strain": {"11": 0.01}',
}
#: Those of the tutorial of the fibres: a smaller ply, coarser triangles
SMALLER_FIBRES = {
    "size=(50.0, 50.0, 10.0)": "size=(25.0, 25.0, 10.0)",
    "plane_mesh(ply, size=1.0, order=2)": "plane_mesh(ply, size=2.0, order=2)",
}
PAGES = {
    PLANE: SMALLER_PLANE,
    FIBRES: SMALLER_FIBRES,
    CONTACT: SMALLER_CONTACT,
    FRACTURE: {},
    TEXTURE: SMALLER_TEXTURE,
    TUTORIAL: SMALLER,
    RVE: SMALLER_RVE,
    RUBBER: SMALLER_RUBBER,
    GEOMETRY: SMALLER_GEOMETRY,
    IGA: SMALLER_IGA,
    MORPHOLOGY: SMALLER_MORPHOLOGY,
}


def _blocks(language: str, page: Path = TUTORIAL) -> list[str]:
    """The fenced code blocks of a language of a page, in their order."""
    text = page.read_text(encoding="utf-8")
    return re.findall(rf"^```{language}\n(.*?)^```$", text, flags=re.M | re.S)


def _smaller(code: str, sizes: dict = SMALLER) -> str:
    if not sizes:  # a page run as it is
        return code
    pattern = "|".join(re.escape(k) for k in sizes)
    return re.sub(pattern, lambda m: sizes[m.group(0)], code)


def _run(page: Path, tmp_path, monkeypatch) -> dict:
    """The snippets of a page run in a session, at the smaller sizes: its namespace."""
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "tutorial"}
    for block in _blocks("python", page):
        exec(compile(_smaller(block, PAGES[page]), str(page), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("page", list(PAGES), ids=lambda p: p.stem)
def test_the_sizes_replaced_are_in_the_tutorial(page):
    code = "\n".join(_blocks("python", page))
    for text in PAGES[page]:
        assert code.count(text) == 1, text


def test_the_figures_of_the_tutorials_are_there():
    """Every figure that a tutorial shows is in docs/_static/tutorial, written by the
    snippets of its page (docs/tutorials/make_figures.py), and no other is."""
    folder = DOCS / "_static" / "tutorial"
    shown = set()
    for page in sorted(DOCS.glob("tutorial*.md")):
        text = page.read_text(encoding="utf-8")
        names = re.findall(r"_static/tutorial/([\w.-]+\.png)", text)
        saved = re.findall(r'savefig\(\s*"([\w.-]+\.png)"|Path\("([\w.-]+\.png)"\)', text)
        saved = {a or b for a, b in saved}
        for name in names:
            path = folder / name
            assert path.is_file(), f"{path}: python docs/tutorials/make_figures.py {page.stem}"
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size < 400_000
            if page in PAGES:  # drawn by a snippet of the page, or by a study that it runs
                by_run = "rve_report" in text and re.fullmatch(r"rve_\w+\.png", name)
                assert name in saved or by_run, f"{page.name}: no snippet saves {name}"
        shown.update(names)
    assert len(shown) >= 8
    assert {p.name for p in folder.glob("*.png")} == shown


@pytest.mark.slow
def test_tutorial_snippets(tmp_path, monkeypatch):
    blocks = _blocks("python")
    assert len(blocks) >= 8  # generate, characterize, percolation, moduli, RVE, path...
    ns = _run(TUTORIAL, tmp_path, monkeypatch)
    assert 0.15 <= ns["found"].threshold <= 0.45
    reuss = 1.0 / sum(f / m.shear for f, m in zip(*_phases(ns), strict=True))
    voigt = sum(f * m.shear for f, m in zip(*_phases(ns), strict=True))
    assert reuss <= ns["mu"] <= voigt
    assert np.isfinite(ns["law"].K) and ns["law"].K > 0
    curve = ns["curve"]
    assert curve.stress[-1, 0] > 0.0 and np.isfinite(ns["gaps"]["tension"]["stress"])
    for name in ("spheres.vti", "laws/Alloy.mfront", "composite.inp",
                 "amitex/composite_run.sh", "composite_law.npz", "study_sections.png",
                 "study_covariance.png", "study_tension.png"):  # fmt: skip
        assert (tmp_path / name).is_file(), name


@pytest.mark.slow
def test_rve_tutorial_snippets(tmp_path, monkeypatch):
    """The size of the representative volume: the variance of the volume fraction of
    the spheres follows the integral range of their model, that of the fibres on Poisson
    lines falls as slowly as their theory says, the windows of one image reach the same
    law as they grow, and the study of the configuration writes its report."""
    ns = _run(RVE, tmp_path, monkeypatch)
    p, a3 = ns["p"], ns["a3"]
    assert ns["law"].K == pytest.approx(p * (1 - p) * a3, rel=0.15)  # 0.51 for 0.53
    low, high = ns["free"].gamma_interval()
    assert low < 1.0 < high and high - low < 0.4
    low, high = ns["slow"].gamma_interval()
    assert low < 2.0 / 3.0 < high < 0.95  # the fibres: 2/3, and not the 1 of the spheres
    assert ns["fibres"].theory()["variance_exponent"] == pytest.approx(2.0 / 3.0)
    # the windows of one image: below the asymptotic law when small, on it when large
    windows = ns["windows"]
    ratio = windows.volume * windows.variance / (p * (1 - p) * a3)
    assert 0.6 < ratio[0] < 0.9 and ratio[-2:] == pytest.approx(1.0, abs=0.15)
    # the study of the configuration: a law for each property, the figures of the run
    report, outdir = ns["report"], ns["outdir"]
    assert set(report["properties"]) == {
        "bulk_modulus.periodic", "conductivity.periodic", "volume_fraction"
    }  # fmt: skip
    for series in report["properties"].values():
        assert series["fit"]["gamma"] == 1.0 and series["fit"]["K"] > 0.0
        assert series["rve"][0]["eps_rel"] == 0.01 and series["rve"][0]["n"] == 1
    for name in ("rve_variance.png", "rve.yaml", outdir / "rve_report.md",
                 outdir / "rve_conductivity.png"):  # fmt: skip
        assert (tmp_path / name).is_file(), name


@pytest.mark.slow
def test_rubber_tutorial_snippets(tmp_path, monkeypatch):
    """A rubber from its tests to its tearing: the laws that can stiffen follow the
    tests and the others do not, the second loading is softer, the particles stiffen the
    rubber, and the cracked strip breaks about the stretch of Rivlin and Thomas."""
    ns = _run(RUBBER, tmp_path, monkeypatch)
    errors = {}
    for model, law in ns["fitted"].items():
        found = law.law(bulk=1000.0)
        gaps = np.concatenate(
            [found.nominal_stress(m, lam, incompressible=True) / s - 1.0
             for m, (lam, s) in ns["tests"].items()]
        )  # fmt: skip
        errors[model] = float(np.sqrt(np.mean(gaps**2)))
    assert errors["yeoh"] < 0.06 and errors["eight_chain"] < 0.06
    assert errors["neo_hookean"] > 0.2 and errors["mooney_rivlin"] > 0.2
    chains = dict(ns["fitted"]["eight_chain"].parameters)
    assert chains["mu"] == pytest.approx(0.4, rel=0.05) and chains["lock"] == pytest.approx(
        2.5, rel=0.05
    )
    assert 0.3 * ns["first"] < ns["again"] < 0.7 * ns["first"]  # softened by the first loading
    assert ns["point"].strain[20, 1] == pytest.approx(-0.31, abs=0.01)
    filled, alone = ns["filled"], ns["alone"]
    assert filled.finite and 1.2 < filled.stress[-1, 0] / alone.stress[-1, 0] < 4.0
    torn = ns["torn"]
    assert torn.finite and ns["broken"] > ns["peak"] > 0
    assert 1.2 < 1 + torn.strain[ns["peak"], 1] < 1.8  # for 1.6: coarse voxels here
    assert torn.crack_surface[-1] == pytest.approx(1 + 3 * ns["h"] / (8 * ns["length"]), rel=0.1)
    for name in ("rubber_tests.png", "rubber_filled.png", "rubber_torn.png"):
        assert (tmp_path / name).is_file(), name


@pytest.mark.slow
def test_geometry_tutorial_snippets(tmp_path, monkeypatch):
    """Exact geometry: the signs of the implicit functions are the phases, the surfaces
    and the mesh hold the volumes of the phases within their chords, the mesh follows
    the grooves of the coatings, the curved elements hold the volumes of the spheres and
    of their interphase, and the three solvers agree on the same geometry."""
    import morphogen as mg

    if not mg.mesh.conforming.has_mmg():
        pytest.skip("MMG (pymmg) is not installed")
    ns = _run(GEOMETRY, tmp_path, monkeypatch)
    micro, exact = ns["micro"], ns["exact"]
    assert micro.implicit_phases() == [0, 1, 2] and ns["count"] == 6
    assert ((ns["level"].data < 0) == (ns["image"].data == 2)).all()
    assert sorted(ns["cut"].coverages) == [0, 1, 2]
    assert ns["spheres"].volume() / 125 == pytest.approx(exact[1], rel=0.05)
    assert exact[2] == pytest.approx(0.2300, abs=1e-4)  # the shells less their lenses
    flat, mesh, fractions = ns["flat"], ns["mesh"], ns["fractions"]
    assert flat.order == 1 and mesh.order == 2 and mesh.curved and not flat.curved
    assert mesh.matching_faces() and set(np.unique(mesh.phase)) == {0, 1, 2}
    assert mesh.min_jacobian() >= 0.3
    assert 0.5 * ns["on"]["interface_nodes"] < ns["on"]["nodes"] <= ns["on"]["interface_nodes"]
    assert fractions[1] == pytest.approx(exact[1], rel=0.03)
    lost = flat.phase_fractions()[1] / exact[1] - 1  # the chords of the flat faces
    assert lost < -0.05 and abs(fractions[1] / exact[1] - 1) < 0.3 * abs(lost)
    assert fractions[2] == pytest.approx(exact[2], rel=0.03)
    # the grooves of the coatings that these coarse elements resolve: sharp edges of the mesh
    assert flat.metadata["sharp_edges"]["pieces"] >= 3 and len(flat.sharp) > 50
    assert mesh.sharp.tolist() == flat.sharp.tolist() and ns["on"]["sharp_nodes"] > 100
    # the three solvers: within a few percent on these coarse grids
    fe = ns["fe"]
    assert fe["10 nodes"] == pytest.approx(fe["4 nodes"], rel=0.05)
    assert ns["fft"][16] == pytest.approx(fe["10 nodes"], rel=0.06)
    assert ns["xfem"] == pytest.approx(ns["voxels"], rel=0.06)
    for name in ("interphase.vti", "geometry_section.png", "spheres.stl", "coated.vtu",
                 "geometry_mesh.png", "coated.inp", "coated.msh",
                 "coated_equations.inp"):  # fmt: skip
        assert (tmp_path / name).is_file(), name


@pytest.mark.slow
def test_iga_tutorial_snippets(tmp_path, monkeypatch):
    """Isogeometric analysis: the third order of quadratic functions on Lamé's cylinder,
    the arrays of holes against the series of Perrins et al., the hole of Vigdergauz on
    the bound of Hashin and Shtrikman, the auxetic sheet at its bound of stiffness, and
    the holes found again from their effective constants."""
    ns = _run(IGA, tmp_path, monkeypatch)
    errors = ns["errors"]
    assert 7.0 < errors[2] / errors[4] < 10.0 and 7.0 < errors[4] / errors[8] < 10.0
    assert ns["solid"].fraction == pytest.approx(1 - np.pi * 0.09, rel=1e-8)
    perrins = ns["perrins"]
    assert ns["square"].tensor[0, 0] / 2 == pytest.approx(perrins(np.pi * 0.09, "square"), rel=2e-6)
    np.testing.assert_allclose(np.diag(ns["hexagonal"].tensor) / 2,
                               perrins(ns["holes"], "hexagonal"), rtol=3e-5)  # fmt: skip
    assert ns["circle"]["bulk_modulus"] / ns["bound"] == pytest.approx(0.986, abs=1e-3)
    assert ns["best"]["bulk_modulus"] / ns["bound"] == pytest.approx(1.0, abs=5e-4)
    assert ns["stiffest"].converged and ns["stiffest"].fraction == pytest.approx(0.6, abs=1e-4)
    ratios = ns["ratios"]
    assert ratios[0] > 0.15 and ratios[-1] < -0.95 and np.all(np.diff(ratios) < 0)
    auxetic, found = ns["auxetic"], ns["found"]
    assert auxetic.converged and found["young_1"] == pytest.approx(50.0, rel=1e-6)
    assert found["poisson_12"] == pytest.approx(-0.814, abs=3e-3)
    np.testing.assert_allclose(ns["a"] + auxetic.x[0], 0.752, atol=5e-3)
    assert ns["b"] + auxetic.x[1] == pytest.approx(0.03, abs=1e-9)
    check = ns["check"].constants()
    assert check["poisson_12"] == pytest.approx(found["poisson_12"], abs=1e-2)
    assert check["young_1"] == pytest.approx(50.0, rel=0.06)
    # the image of the sheet for the FFT solver: coarse here, and auxetic all the same
    assert ns["voxels"].periodic and set(np.unique(ns["voxels"].data)) == {0, 1}
    np.testing.assert_allclose(ns["by_fft"], ns["by_iga"], rtol=0.12, atol=3.0)
    assert ns["by_fft"][0, 1] < -100.0
    np.testing.assert_allclose(ns["inverse"].x, [0.1, -0.03], atol=1e-5)
    assert ns["inverse"].objective < 1e-10
    for name in ("iga_hole.png", "iga_auxetic.png"):
        assert (tmp_path / name).is_file(), name


@pytest.mark.slow
def test_contact_tutorial_snippets(tmp_path, monkeypatch):
    """Contact and damage: the block with a slot stiffens when the slot closes on itself
    and stays soft without self-contact; the indenter reads the modulus of the block with
    its base; the walls of a periodic cell touch along a path; the damage laws follow
    their closed forms at a material point; the composite passes its strength."""
    ns = _run(CONTACT, tmp_path, monkeypatch)
    crush, through = ns["crush"], ns["through"]
    assert crush.modulus == pytest.approx(673.0, rel=5e-3)
    assert crush.strain[ns["closed"]] == pytest.approx(-0.07)
    assert crush.tangent()[-1] == pytest.approx(1194.0, rel=0.01) and crush.self_area[-1] == 2.0
    assert through.stress[-1] / crush.stress[-1] == pytest.approx(0.82, abs=0.01)
    assert not through.self_area.any()
    assert (tmp_path / "crushed.vtu").is_file()
    dent, peak = ns["dent"], ns["peak"]
    exact = 1000.0 / (1 - 0.3**2)
    assert dent.modulus[peak] == pytest.approx(1426.0, rel=5e-3)
    assert dent.layer_modulus(0.3)[peak] == pytest.approx(exact, rel=0.06)
    press = ns["press"]
    np.testing.assert_allclose(press.series["contact_area"][[3, -1]], [0.0, 0.375])
    assert press.stress[-1, 2] < -200.0
    assert np.abs(ns["bar"].plastic - ns["exact"]).max() < 1e-10
    creep, t_r = ns["creep"], ns["t_r"]
    assert t_r == pytest.approx(5.5804, rel=1e-4)
    assert creep.plastic[-1] == pytest.approx(1 - 0.1 ** (1 / 7), rel=0.03)
    assert ns["pull"].stress[:, 0].max() == pytest.approx(3.0, rel=0.02)
    assert -ns["push"].stress[:, 0].min() == pytest.approx(34.1, rel=0.01)
    tension = ns["tension"]
    assert 100.0 < tension.strength() < 500.0  # coarser here: later and higher
    assert 0.005 < tension.phases[0]["plastic"][-1] < 0.5
    assert tension.phases[1]["plastic"][-1] == 0.0
    for name in ("contact_crush.png", "damage_laws.png"):
        assert (tmp_path / name).is_file(), name


@pytest.mark.slow
def test_fracture_tutorial_snippets(tmp_path, monkeypatch):
    """How a microstructure breaks: two layers debond at the strength of their interfaces
    and the work is their fracture energy; a pore nearly doubles the utilization of a
    cycle; a crystal of magnesium twins when pulled along c and not when compressed, and
    a textured polycrystal keeps the asymmetry; a plane of iron follows its law and
    breaks with its energy; a polycrystal cleaves across one section; the probability of
    Beremin rises along the path."""
    ns = _run(FRACTURE, tmp_path, monkeypatch)
    pull = ns["pull"]
    assert pull.stress[:, 2].max() == pytest.approx(30.0, rel=1e-3)
    work = np.trapezoid(pull.stress[:, 2], pull.strain[:, 2]) * 2.0
    assert work == pytest.approx(0.8, rel=1e-3)
    assert pull.series["debonded_area"][-1] == pytest.approx(0.5)
    judged = ns["judged"]
    assert judged.largest == pytest.approx(0.70, abs=0.01) and judged.fraction() == 0.0
    assert judged.largest / (40.0 / 110.0) == pytest.approx(1.93, abs=0.02)
    assert np.isnan(judged.utilization[ns["pore"]]).all()
    assert ns["shear"] == pytest.approx(0.1289, abs=1e-4)
    pulled, pushed = ns["pulled"], ns["pushed"]
    assert pulled.stress[-1, 2] == pytest.approx(87.5, rel=0.01)
    assert pulled.series["twin_fraction"][-1] == pytest.approx(0.44, abs=0.01)
    assert pushed.stress[-1, 2] == pytest.approx(-231.0, rel=0.01)
    assert pushed.series["twin_fraction"][-1] == 0.0
    sheet = ns["sheet"]
    assert sheet["tension"].stress[-1, 2] == pytest.approx(77.0, rel=0.03)
    assert sheet["tension"].series["twin_fraction"][-1] == pytest.approx(0.20, abs=0.02)
    assert sheet["compression"].stress[-1, 2] == pytest.approx(-153.0, rel=0.03)
    cleaved = ns["cleaved"]
    assert np.abs(cleaved.stress[20:, 2] - ns["exact"][20:]).max() < 1e-3
    assert np.trapezoid(cleaved.stress[:, 2], cleaved.strain[:, 2]) == pytest.approx(5.0, rel=0.01)
    broken, damage = ns["broken"], ns["damage"]
    assert broken.stress[:, 2].max() == pytest.approx(592.0, rel=0.03)
    assert broken.stress[-1, 2] < 0.15 * broken.stress[:, 2].max()
    assert 64 <= int((damage > 0.9).sum()) < 100
    found = ns["found"]
    assert found.weibull[[2, 7, 14]] == pytest.approx([1185.0, 1459.0, 1683.0], rel=0.01)
    assert found.probability[2] < 0.01 and found.probability[7] == pytest.approx(0.23, abs=0.03)
    assert found.probability[-1] > 0.99
    for name in ("fracture_twins.png", "fracture_cleavage.png"):
        assert (tmp_path / name).is_file(), name


@pytest.mark.slow
def test_texture_tutorial_snippets(tmp_path, monkeypatch):
    """From an orientation map to a textured polycrystal: the grains and the texture of
    the map that comes with morphoGen; the exact bulk modulus of its polycrystal and its
    soft cube axes; the lattices that a tension turns; the tensile axis of the grains of
    a Taylor polycrystal leaves <110>; a polycrystal of space with the texture of the
    map."""
    ns = _run(TEXTURE, tmp_path, monkeypatch)
    grains = ns["grains"]
    assert grains.count == 56 and grains.diameters.mean() == pytest.approx(3.95, abs=0.01)
    assert [round(p.maximum, 1) for p in ns["poles"]] == [5.7, 3.6, 4.6]
    assert len(ns["pairs"]) == 130
    mean = float(np.sum(ns["lengths"] * ns["angles"]) / ns["lengths"].sum())
    assert mean == pytest.approx(38.4, abs=0.1)
    elastic = ns["elastic"]
    assert elastic["bulk_modulus"] == pytest.approx((246.5 + 2 * 147.3) / 3, rel=1e-6)
    assert elastic["E1"] == pytest.approx(188.5, abs=0.2)
    assert elastic["E3"] == pytest.approx(197.8, abs=0.2)
    # (a shorter tension: the lattices have turned, by less than at 2 %)
    pulled, turned = ns["pulled"], ns["turned"]
    assert pulled.info["cuts"] == 0 and 100.0 < pulled.cauchy[-1, 0] < 251.0
    assert 0.0 < turned.mean() < 0.71 and len(ns["euler"]) == 64 * 48
    # (the first grains of the polycrystal of Taylor: none stays near <110>)
    final, after = ns["final"], ns["after"]
    assert final.shape == (6, 3) and after.at([0, 1, 1]) < 0.8
    solid = ns["solid"]
    assert solid["bulk_modulus"] == pytest.approx((246.5 + 2 * 147.3) / 3, rel=1e-6)
    assert solid["E1"] < 215.0  # (226 with orientations at random)
    for name in ("texture_map.png", "texture_poles.png", "texture_fibres.png"):
        assert (tmp_path / name).is_file(), name


def _phases(ns):
    """The fractions and the elastic materials of the phases of the image of the
    tutorial."""
    from morphogen import analysis

    fractions = analysis.phase_fractions(ns["image"])
    materials = ns["materials"]
    return [fractions[p] for p in sorted(materials)], [materials[p] for p in sorted(materials)]


def test_tutorial_configuration_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (config,) = _blocks("yaml")
    spec = yaml.safe_load(config)
    spec["voxels"]["shape"] = 16
    (tmp_path / "spheres.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    assert cli.main(["run", "spheres.yaml", "-q"]) == 0
    out = tmp_path / spec["outputs"]["directory"]
    assert (out / "spheres_0000.vti").is_file() and (out / "results.csv").is_file()
    assert (out / "spheres_0000.json").is_file()


@pytest.mark.parametrize("page", [TUTORIAL, RVE], ids=lambda p: p.stem)
def test_tutorial_commands_parse(page):
    parser = cli.build_parser()
    commands = []
    for block in _blocks("bash", page):
        for line in block.splitlines():
            words = shlex.split(line, comments=True)
            if words and words[0] == "morphogen":
                commands.append(words)
    names = {words[1] for words in commands}
    expected = {"info", "models", "template", "run", "view", "percolation", "mfront", "amitex"}
    assert (expected if page == TUTORIAL else {"run"}) <= names
    for words in commands:
        args = parser.parse_args(words[1:])
        assert callable(args.func), words


@pytest.mark.slow
def test_morphology_tutorial_snippets(tmp_path, monkeypatch):
    """Mathematical morphology: the probes of the exact geometry follow the closed forms of
    the Boolean model, the straight lines of Matheron's tests give back its radius, the
    digital operations keep their order, the exact functionals and the sections agree with
    the stereological identities, and the image is identified as hard spheres."""
    ns = _run(MORPHOLOGY, tmp_path, monkeypatch)
    model, balls = ns["model"], ns["balls"]
    assert np.all(np.abs(balls.values - model.choquet_capacity("ball", ns["radii"])) < 0.03)
    assert ns["radius"] == pytest.approx(1.0, rel=0.1)
    assert ns["eroded"].mean() < ns["opened"].mean() < ns["image"].data.mean()
    assert ns["image"].data.mean() < ns["closed"].mean() < ns["dilated"].mean()
    exact, theory = ns["exact"], ns["theory"]
    assert exact.specific_surface == pytest.approx(theory["specific_surface"], rel=0.03)
    sections = ns["sections"]
    la = np.mean([s["boundary_density"] for s in sections])
    assert la == pytest.approx(np.pi / 4 * exact.specific_surface, rel=0.05)
    assert ns["found"].best.name == "hardcore_spheres"
    for name in ("morphology_capacities.png", "morphology_operations.png",
                 "morphology_identification.png"):  # fmt: skip
        assert (tmp_path / name).is_file(), name


@pytest.mark.slow
def test_plane_tutorial_snippets(tmp_path, monkeypatch):
    """Polycrystals of the plane: the exact polygons of seeds of the plane and their small
    features removed, the bulk modulus of cubic crystals under the three conditions and
    their ordering, the layer in space giving the tensor of the plane, the exact section of
    a polycrystal of space, the bounds of an orientation map, a plastic path, the plane
    elements of Abaqus."""
    ns = _run(PLANE, tmp_path, monkeypatch)
    assert ns["mesh"].n_elements < ns["exact"].n_elements and ns["mesh"].matching_sides()
    out = ns["out"]
    e1 = [out[f"stiffness.{bc}"]["E1"] for bc in ("kubc", "periodic", "subc")]
    assert e1[0] >= e1[1] >= e1[2]
    for bc in ("kubc", "periodic", "subc"):
        assert out[f"stiffness.{bc}"]["bulk_modulus"] == pytest.approx(
            (168.4 + 2 * 121.4) / 3.0, rel=1e-9)  # fmt: skip
    assert np.abs(ns["layer"]["tensor"] - ns["plane"]["tensor"]).max() < 1e-9 * 168.4
    assert ns["section"].areas().sum() == pytest.approx(1.0, rel=1e-12)
    bounds = ns["bounds"]
    assert bounds["stiffness.kubc"]["E1"] > bounds["stiffness.subc"]["E1"]
    assert ns["path"]["pull.periodic"].components()["S11"] > 100.0
    assert (tmp_path / "sheet.inp").read_text().count("*SOLID SECTION") == ns["sheet"].n_grains
    assert (tmp_path / "plane_grains.png").is_file()


@pytest.mark.slow
def test_fibres_tutorial_snippets(tmp_path, monkeypatch):
    """Fibre composites in their plane: the fibres meshed by their exact discs, the axial
    conductivity the arithmetic mean of the phases under every condition, the connection of
    Hill to round-off, the transverse bulk modulus between the bounds of Hashin and
    Shtrikman, the conditions ordered, the pores of the preform in one part and its axial
    permeability above the transverse ones, the plane elements of Abaqus."""
    ns = _run(FIBRES, tmp_path, monkeypatch)
    assert ns["mesh"].metadata["mesher"] == "exact discs (mmg2d)" and ns["mesh"].matching_sides()
    c, out = ns["c"], ns["out"]
    for bc in ("kubc", "periodic", "subc"):
        k = out[f"conductivity_tensor.{bc}"]["tensor"]
        assert k[2, 2] == pytest.approx(c * 10.0 + (1 - c) * 0.2, rel=1e-9)
    e1 = [out[f"stiffness.{bc}"]["E1"] for bc in ("kubc", "periodic", "subc")]
    assert e1[0] >= e1[1] >= e1[2]
    a, b, km, lm = ns["a"], ns["b"], ns["km"], ns["lm"]
    assert ns["stress"][0] == pytest.approx(2 * km * a + lm * b, rel=1e-12)
    assert ns["lower"] < ns["k23"] < ns["upper"]
    preform = ns["preform"]
    assert preform["parts"] == 1 and preform["k33"] > 10 * max(preform["k11"], preform["k22"])
    assert (tmp_path / "ply.inp").read_text().count("*SOLID SECTION") == 2
    assert (tmp_path / "fibres_section.png").is_file()
