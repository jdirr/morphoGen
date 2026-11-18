"""The files of a release (docs/release.md) in step with the package: the conda-forge
recipe with pyproject.toml (its dependencies, its extras as constraints, its entry point,
the packages of the public interface as its import tests), CITATION.cff with the version,
and the JOSS draft with its bibliography and the length JOSS asks for."""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

import morphogen
from morphogen._public import public_modules

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "packaging" / "conda-forge" / "recipe.yaml"
PAPER = ROOT / "paper"

#: The conda-forge packages of the PyPI packages of the extras that differ by name; None:
#: none (pymmg: the package mmg puts mmg3d on the PATH; cupy brings its CUDA libraries)
CONDA = {
    "matplotlib": "matplotlib-base",
    "cupy-cuda13x": "cupy",
    "pymmg": None,
    "cuda-toolkit": None,
}
#: The extras of the tools of development, not of the uses of the package
TOOLS = {"test", "docs", "dev", "all"}


def requirement(text: str) -> tuple[str, str]:
    """The name and version constraint of a requirement of pyproject.toml (its extras and
    environment marker dropped) or of the recipe."""
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)(\[[^\]]*\])?\s*(.*)", text.split(";")[0].strip())
    return match.group(1).lower(), match.group(3).replace(" ", "")


@pytest.fixture(scope="module")
def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


@pytest.fixture(scope="module")
def recipe():
    if not RECIPE.exists():
        pytest.skip("the recipe is in the repository, not in the source archive")
    return yaml.safe_load(RECIPE.read_text(encoding="utf-8"))


def test_recipe_follows_pyproject(project, recipe):
    requirements = recipe["requirements"]
    python, *run = requirements["run"]
    assert python == "python >=${{ python_min }}"
    assert project["requires-python"] == ">=" + recipe["context"]["python_min"]
    assert dict(map(requirement, run)) == dict(map(requirement, project["dependencies"]))
    extras = {}
    for extra, packages in project["optional-dependencies"].items():
        if extra not in TOOLS:
            for name, version in map(requirement, packages):
                if CONDA.get(name, name) is not None:
                    extras[CONDA.get(name, name)] = version
    assert dict(map(requirement, requirements["run_constraints"])) == extras
    scripts = [f"{name} = {target}" for name, target in project["scripts"].items()]
    assert recipe["build"]["python"]["entry_points"] == scripts
    assert recipe["about"]["summary"] == project["description"]
    assert recipe["about"]["license"] == project["license"]


def test_recipe_imports_the_public_packages(recipe):
    packages = [name for name, is_package in public_modules() if is_package]
    assert recipe["tests"][0]["python"]["imports"] == packages


def test_citation_version():
    path = ROOT / "CITATION.cff"
    if not path.exists():
        pytest.skip("no CITATION.cff")
    citation = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert citation["version"] == morphogen.__version__
    assert citation["license"] == "GPL-3.0-or-later"


def test_paper_cites_its_bibliography():
    if not PAPER.exists():
        pytest.skip("the paper is in the repository, not in the source archive")
    text = (PAPER / "paper.md").read_text(encoding="utf-8")
    body = re.sub(r"<!--.*?-->", "", text.split("\n---\n", 1)[1], flags=re.S)
    bibliography = (PAPER / "paper.bib").read_text(encoding="utf-8")
    keys = re.findall(r"^@\w+\{([^,]+),", bibliography, flags=re.M)
    assert len(keys) == len(set(keys))
    assert set(re.findall(r"@([A-Za-z][A-Za-z0-9_]*)", body)) == set(keys)
    for entry in re.split(r"\n(?=@)", bibliography):
        assert entry.count("{") == entry.count("}"), entry.splitlines()[0]
    # the sections and length of JOSS's guidelines (docs/release.md), the headings left out
    sections = re.findall(r"^# (.+)$", body, flags=re.M)
    assert sections[:6] == [
        "Summary",
        "Statement of need",
        "State of the field",
        "Software design",
        "Research impact statement",
        "AI usage disclosure",
    ]
    words = len(re.sub(r"^# .+$", "", body, flags=re.M).split())
    assert 750 <= words <= 1750
