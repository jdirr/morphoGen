"""The recipes of the documentation (docs/howto.md) and the first study of its landing
page (docs/index.md) stay correct: their Python snippets run one after another in one
session, as they are written (they are small by design), and give what the pages say;
the README shows the same first study, and each ``morphogen`` command of the pages of
the documentation and of the README is one that the command line parses."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import numpy as np
import pytest

import morphogen as mg
from morphogen import cli

DOCS = Path(__file__).resolve().parents[1] / "docs"
README = DOCS.parent / "README.md"
HOWTO = DOCS / "howto.md"
if not DOCS.is_dir():  # an installed package without its sources
    pytest.skip("the documentation is not here", allow_module_level=True)


def _blocks(page: Path, language: str) -> list[str]:
    text = page.read_text(encoding="utf-8")
    return re.findall(rf"^```{language}\n(.*?)^```$", text, flags=re.M | re.S)


@pytest.mark.slow
def test_recipes_run(tmp_path, monkeypatch):
    if not mg.mesh.conforming.has_mmg():
        pytest.skip("MMG (pymmg) is not installed")
    monkeypatch.chdir(tmp_path)
    blocks = _blocks(HOWTO, "python")
    assert len(blocks) >= 18
    ns: dict = {"__name__": "recipes"}
    for block in blocks:
        exec(compile(block, str(HOWTO), "exec"), ns)
    # the realization read back is the one saved; the section is a layer of the image
    assert np.array_equal(ns["same"].voxelize(shape=24).data, ns["micro"].voxelize(shape=24).data)
    assert ns["section"].shape == (32, 32, 1) and ns["mixed"].implicit_phases() == [0, 1, 2]
    fraction = float(np.mean(ns["image"].data == 1))
    assert ns["surface"].volume() / ns["domain"].volume == pytest.approx(fraction, rel=0.05)
    assert set(np.unique(ns["solid"].phase)) == {0}
    # the grains of the map that comes with morphoGen; the exact bulk modulus of its crystals
    assert ns["grains"].count == 56 and round(ns["figure"].maximum, 1) == 5.7
    assert len(ns["polycrystal"].orientations) == 200 and ns["poles"].at([0, 0, 1]) > 4.0
    exact = (246.5 + 2 * 147.3) / 3
    assert ns["moduli"]["bulk_modulus.periodic"].value == pytest.approx(exact, rel=1e-6)
    # a uniaxial stress: the strain is the compliance, the lateral stress zero
    response = ns["response"]
    assert response["E11"] == pytest.approx(1.0 / ns["stiffness"]["E1"], rel=1e-6)
    assert response["S11"] == pytest.approx(1.0) and abs(response["S22"]) < 1e-9
    limits = ns["limits"]
    assert limits["reuss"] <= limits["hs_lower"] <= limits["hs_upper"] <= limits["voigt"]
    assert ns["point"].stress[5, 0] > 100.0 > 0.0 > ns["point"].stress[10, 0]
    assert ns["large"].finite and ns["large"].log_strain[-1, 0] == pytest.approx(np.log(1.5))
    broken = ns["broken"]
    assert broken.stress[-1, 0] < 0.05 * broken.strength() and broken.crack_surface[-1] > 0.05
    assert ns["law"].K == pytest.approx(4.0, rel=0.2) and ns["law"].gamma == 1.0
    for name in ("spheres.json", "spheres.stl", "composite.inp", "composite_equations.inp",
                 "voxels.inp", "amitex/spheres_run.sh", "laws/Alloy.mfront", "grains.stl",
                 "grains.data", "grains.txt", "grains.msh", "bodies.msh"):  # fmt: skip
        assert (tmp_path / name).is_file(), name
    assert (tmp_path / "settle" / "morphogen_blender.py").is_file()
    assert ns["stream"].objects == ["phase_1"] and (tmp_path / "tunnel" / "scene.json").is_file()
    assert ns["scene"].kinds == {"sphere": len(ns["micro"].spheres)}
    # the clusters of the spheres, whole but the one that winds around the periodic box
    assert len(ns["bodies"]) == 4 and [p.whole for p in ns["bodies"]].count(False) == 1
    assert np.shares_memory(ns["shared"], ns["u"].data) and ns["u"].data.dtype == np.float32


def test_first_study_of_the_landing_page(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    page = DOCS / "index.md"
    blocks = _blocks(page, "python")
    assert len(blocks) == 1
    ns: dict = {"__name__": "landing"}
    exec(compile(blocks[0], str(page), "exec"), ns)
    # the values that the comments of the snippet give
    assert float(np.mean(ns["image"].data == 1)) == pytest.approx(0.29, abs=0.005)
    assert ns["model"].theory()["volume_fraction"] == pytest.approx(0.3)
    values = {name: result.value for name, result in ns["out"].items()}
    assert values["bulk_modulus.periodic"] == pytest.approx(5.89, rel=0.01)
    assert values["shear_modulus.periodic"] == pytest.approx(2.59, rel=0.01)
    assert values["conductivity.periodic"] == pytest.approx(0.33, rel=0.02)
    # the template of its commands is one of the templates
    assert cli.main(["template", "homogenization_spheres", "-o", "study.yaml"]) == 0
    assert (tmp_path / "study.yaml").is_file()


def test_first_study_of_the_readme_is_that_of_the_landing_page():
    assert _blocks(README, "python") == _blocks(DOCS / "index.md", "python")


#: The pages that give ``morphogen`` commands (the README among them)
PAGES_WITH_COMMANDS = [
    "index.md", "installation.md", "tutorial.md", "tutorial_rve.md",
    "tutorial_interface.md", "howto.md", "gallery.md", "measures.md",
    "properties.md", "physics.md", "behaviours.md", "fracture.md", "rve.md",
    "laser.md", "dynamics.md", "running.md", "interop.md", "configuration.md",
    "../README.md",
]  # fmt: skip


@pytest.mark.parametrize("page", PAGES_WITH_COMMANDS)
def test_commands_of_the_pages_parse(page):
    parser = cli.build_parser()
    commands = []
    for block in _blocks(DOCS / page, "bash"):
        for line in block.replace("\\\n", " ").splitlines():  # continuations joined
            for part in line.split("&&"):
                words = shlex.split(part, comments=True)
                if words and words[0] == "morphogen":
                    commands.append(words)
    assert commands, page
    for words in commands:
        args = parser.parse_args(words[1:])
        assert callable(args.func), words
    if page == "howto.md":
        assert {"run", "status", "submit", "merge", "db", "ashby"} <= {w[1] for w in commands}
