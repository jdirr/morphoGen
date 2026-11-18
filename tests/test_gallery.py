"""The gallery of examples of the documentation (docs/gallery.md) in step with the templates:
every template has its card and its image, every card its template, the templates that
show a result of their run have its figure, and the page is the one that its script
(docs/gallery/make_gallery.py) writes."""

from __future__ import annotations

import importlib.util
import re
from importlib import resources
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
if not DOCS.is_dir():  # an installed package without its sources
    pytest.skip("the documentation is not here", allow_module_level=True)
PAGE = DOCS / "gallery.md"
IMAGES = DOCS / "_static" / "gallery"
CARD = ":::{container} gallery-card"


def _templates() -> list[str]:
    folder = resources.files("morphogen") / "templates"
    return sorted(p.name[: -len(".yaml")] for p in folder.iterdir() if p.name.endswith(".yaml"))


def _cards() -> dict[str, str]:
    """The text of each card of the page, by the template of its image."""
    text = PAGE.read_text(encoding="utf-8")
    cards = {}
    for block in text.split(CARD)[1:]:
        block = block.split("\n:::\n")[0]
        found = re.search(r"_static/gallery/(\w+)\.png", block)
        assert found, block[:200]
        assert found.group(1) not in cards, f"two cards of {found.group(1)}"
        cards[found.group(1)] = block
    return cards


def test_every_template_has_its_card_and_every_card_its_template():
    cards = _cards()
    templates = _templates()
    assert sorted(cards) == templates, (
        f"templates without a card: {sorted(set(templates) - set(cards))}; cards without "
        f"a template: {sorted(set(cards) - set(templates))} (add them to FAMILIES in "
        "docs/gallery/make_gallery.py and run it)"
    )
    for name, block in cards.items():  # the card shows its own image, file and commands
        assert f":name: gallery-{name.replace('_', '-')}" in block
        assert f"../src/morphogen/templates/{name}.yaml" in block
        assert (DOCS / f"../src/morphogen/templates/{name}.yaml").is_file()
        assert f"morphogen template {name} -o {name}.yaml" in block
        assert any(f"morphogen {c} {name}.yaml" in block for c in ("run", "optimize", "laser"))
        assert f"**`{name}`** — " in block  # its name and description


def _script():
    path = DOCS / "gallery" / "make_gallery.py"
    spec = importlib.util.spec_from_file_location("make_gallery", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_every_template_has_its_image():
    total = 0
    results = set(_script().RESULTS)
    assert results <= set(_templates())
    wanted = {name: f"python docs/gallery/make_gallery.py {name}" for name in _templates()}
    wanted.update(
        {f"{name}_result": f"python docs/gallery/make_gallery.py --results {name}"
         for name in results}
    )  # fmt: skip
    for name, command in wanted.items():
        path = IMAGES / f"{name}.png"
        assert path.is_file(), f"{path} missing: {command}"
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(data) < 100_000, f"{name}.png: {len(data)} bytes"
        total += len(data)
    assert total < 5_000_000
    extra = {p.stem for p in IMAGES.glob("*.png")} - set(wanted)
    assert not extra, f"images without a template: {sorted(extra)}"
    # the card of a template with a result shows it below its image
    cards = _cards()
    for name in _templates():
        assert (f"_static/gallery/{name}_result.png" in cards[name]) == (name in results)


def test_page_is_written_by_its_script():
    script = _script()
    assert sorted(script.gallery_names()) == _templates()
    assert PAGE.read_text(encoding="utf-8") == script.page(), (
        "docs/gallery.md is not the page of docs/gallery/make_gallery.py: run "
        "python docs/gallery/make_gallery.py --page"
    )
