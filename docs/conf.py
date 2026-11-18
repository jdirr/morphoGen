"""Configuration of the documentation site (Sphinx): the Markdown pages of docs/ (MyST),
the README in parts, the change log, and the API reference written from the docstrings.

    python -m pip install ".[docs]"
    python -m sphinx -b html docs site          # the site in site/, from site/index.html

The pages of the API reference (docs/api/) are written at each build, one per module of
the public interface (``morphogen._public``, docs/stability.md): a package lists its
public names, linked to the module that defines them, and its modules; a module
documents what it defines, so that each object is documented once.
"""

import importlib
import inspect
from pathlib import Path

import morphogen
from morphogen._platform import full_speed
from morphogen._public import is_public_module, public_modules

full_speed()  # Windows: the build kept off the efficiency cores (EcoQoS)

project = "morphoGen"
author = "Justin Dirrenberger"
copyright = "2026, Justin Dirrenberger"
release = morphogen.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "site"]
suppress_warnings = ["myst.header"]  # the README's sections start at level 2

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_dmath_allow_digits = False  # "$5 and $6" is not math
myst_heading_anchors = 4  # links to the sections of a page (theory.md#references)

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
napoleon_google_docstring = False

html_theme = "furo"
html_title = f"morphoGen {release}"
html_show_sourcelink = False
html_logo = "_static/logo.svg"  # a periodic cell of grains; the icon of the tabs too
html_favicon = "_static/logo.svg"
html_theme_options = {
    "navigation_with_keys": True,  # the arrows of the keyboard turn the pages
    # the blue of the phase 1 of the figures (viz.PALETTE), lighter on the dark theme
    "light_css_variables": {"color-brand-primary": "#1f4e79", "color-brand-content": "#1f4e79"},
    "dark_css_variables": {"color-brand-primary": "#8fbde6", "color-brand-content": "#8fbde6"},
}
# the styles and the script of the site: the cards of the gallery; the landing page, the
# figures of the tutorials, the copy button of the code blocks and the filter of the
# gallery (the images, in _static/gallery and _static/tutorial, are those of their pages:
# copied once, with the pages)
html_static_path = ["_static/gallery.css", "_static/site.css", "_static/site.js"]
html_css_files = ["gallery.css", "site.css"]
html_js_files = ["site.js"]


def _summary(obj) -> str:
    """The first paragraph of a docstring, on one line (without the "::" of a literal
    block that follows it)."""
    doc = inspect.getdoc(obj) or ""
    summary = " ".join(doc.split("\n\n")[0].split())
    return summary[:-1] if summary.endswith("::") else summary


def _title(name: str) -> str:
    return f"{name}\n{'=' * len(name)}\n"


def _package_page(name: str, children: list[str]) -> str:
    package = importlib.import_module(name)
    lines = [_title(name), f".. automodule:: {name}\n"]
    links, here = [], []
    for attr in getattr(package, "__all__", []):
        obj = getattr(package, attr)
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue  # the modules (below) and the data
        home = obj.__module__
        if home != name and is_public_module(home):
            summary = _summary(obj)
            target = f"{home}.{obj.__qualname__}"
            links.append(f"* :py:obj:`{attr} <{target}>`" + (f" — {summary}" if summary else ""))
        else:  # defined here, or in a module left out: documented on this page
            kind = "class" if inspect.isclass(obj) else "function"
            here.append(f".. auto{kind}:: {name}.{attr}\n")
    if links or here:
        lines.append("Public names\n------------\n")
        lines.extend([*links, ""] if links else [])
        lines.extend(here)
    if children:
        lines.append("Modules\n-------\n")
        lines.append(".. toctree::\n   :maxdepth: 1\n")
        lines.extend(f"   {child}" for child in children)
        lines.append("")
    return "\n".join(lines) + "\n"


def _module_page(name: str) -> str:
    return (
        _title(name)
        + f"\n.. automodule:: {name}\n   :members:\n   :ignore-module-all:\n"
        + "   :show-inheritance:\n"
    )


def write_api_pages(app) -> None:
    """The pages of docs/api/, rewritten only where they change (Sphinx rebuilds what
    changed)."""
    folder = Path(app.srcdir) / "api"
    folder.mkdir(exist_ok=True)
    modules = public_modules()
    pages = {}
    for name, is_package in modules:
        if is_package:
            depth = name.count(".") + 1
            children = [m for m, _ in modules if m.startswith(name + ".") and m.count(".") == depth]
            pages[name] = _package_page(name, children)
        else:
            pages[name] = _module_page(name)
    for path in folder.glob("*.rst"):
        if path.stem not in pages:
            path.unlink()
    for name, text in pages.items():
        path = folder / f"{name}.rst"
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8", newline="\n")


def keep_undocumented(app, what, name, obj, skip, options):
    """The public functions, classes, methods and properties without a docstring shown
    with their signature, rather than left out (autodoc's default); what a module imports
    stays out."""
    if skip and not name.startswith("_"):
        target = obj.__func__ if isinstance(obj, (classmethod, staticmethod)) else obj
        if inspect.isfunction(target) or inspect.isclass(target) or isinstance(obj, property):
            return False
    return None


def setup(app):
    app.connect("builder-inited", write_api_pages)
    app.connect("autodoc-skip-member", keep_undocumented)
