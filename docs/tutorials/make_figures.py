"""The figures of the tutorials of the documentation (docs/_static/tutorial/).

    python docs/tutorials/make_figures.py                    # every tutorial
    python docs/tutorials/make_figures.py tutorial_rubber    # that one

The Python snippets of a tutorial page are run one after another, as they are written,
in a temporary folder (as tests/test_tutorial.py runs them at smaller sizes): the
figures that they save there (``fig.savefig("name.png")``), or that a run of theirs
writes in a folder below (the report of an RVE study), and that the page shows
(``_static/tutorial/name.png``) are copied to docs/_static/tutorial/, reduced to a
palette of colours. A page thus shows the figures of its own snippets; run this script
again when they change. The tutorials take from ten seconds to a few minutes each.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FIGURES = DOCS / "_static" / "tutorial"
#: The pages of the tutorials whose snippets save figures
PAGES = ("tutorial", "tutorial_rubber", "tutorial_geometry", "tutorial_rve", "tutorial_iga",
         "tutorial_contact", "tutorial_fracture", "tutorial_texture",
         "tutorial_morphology")  # fmt: skip
#: Largest number of colours of a figure
COLOURS = 96


def snippets(page: str) -> list[str]:
    """The Python snippets of a page, in their order."""
    text = (DOCS / f"{page}.md").read_text(encoding="utf-8")
    return re.findall(r"^```python\n(.*?)^```$", text, flags=re.M | re.S)


def shown(page: str) -> list[str]:
    """The names of the figures that a page shows."""
    text = (DOCS / f"{page}.md").read_text(encoding="utf-8")
    return re.findall(r"_static/tutorial/([\w.-]+\.png)", text)


def run(page: str, folder: Path) -> dict:
    """The snippets of a page run in ``folder``: their namespace."""
    namespace: dict = {"__name__": page}
    here = Path.cwd()
    os.chdir(folder)
    try:
        for block in snippets(page):
            exec(compile(block, f"{page}.md", "exec"), namespace)
    finally:
        os.chdir(here)
    return namespace


def keep(source: Path, target: Path) -> int:
    """A figure copied with a palette of at most :data:`COLOURS` colours; its size."""
    from PIL import Image

    picture = Image.open(source).convert("RGB")
    picture = picture.quantize(COLOURS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    target.parent.mkdir(parents=True, exist_ok=True)
    picture.save(target, optimize=True)
    return target.stat().st_size


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pages", nargs="*", help="the tutorials whose figures to write (all)")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.pages) - set(PAGES))
    if unknown:
        print(f"unknown tutorials: {unknown} (among {', '.join(PAGES)})", file=sys.stderr)
        return 1
    for page in args.pages or PAGES:
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as folder:
            run(page, Path(folder))
            for name in shown(page):
                found = sorted(Path(folder).rglob(name), key=lambda path: len(path.parts))
                if not found:
                    print(f"{page}: its snippets do not save {name}", file=sys.stderr)
                    return 1
                size = keep(found[0], FIGURES / name)  # the one nearest to the folder
                print(f"{page:20s} {name:28s} {size / 1024:6.1f} kB")
        print(f"{page}: {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
