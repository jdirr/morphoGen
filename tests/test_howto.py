"""The recipes of the documentation (docs/howto.md) and the first study of its landing
page (docs/index.md) stay correct: their Python snippets run one after another in one
session, as they are written (they are small by design), and give what the pages say;
each ``morphogen`` command of these pages and of the tour of the interface
(docs/tutorial_interface.md) is one that the command line parses."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import numpy as np
import pytest

import morphogen as mg
from morphogen import cli

DOCS = Path(__file__).resolve().parents[1] / "docs"
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
                 "voxels.inp", "amitex/spheres_run.sh", "laws/Alloy.mfront"):  # fmt: skip
        assert (tmp_path / name).is_file(), name


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


@pytest.mark.parametrize("page", ["howto.md", "tutorial_interface.md", "index.md"])
def test_commands_of_the_pages_parse(page):
    parser = cli.build_parser()
    commands = []
    for block in _blocks(DOCS / page, "bash"):
        for line in block.splitlines():
            words = shlex.split(line, comments=True)
            if words and words[0] == "morphogen":
                commands.append(words)
    assert commands, page
    for words in commands:
        args = parser.parse_args(words[1:])
        assert callable(args.func), words
    if page == "howto.md":
        assert {"run", "status", "submit", "merge", "db", "ashby"} <= {w[1] for w in commands}
