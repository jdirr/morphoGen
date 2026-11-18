"""The public interface of morphoGen (docs/stability.md): its modules and their public
names, for the API reference of the documentation site (docs/conf.py) and for the test
that keeps the list of these names (tests/test_public_api.py)."""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
from pathlib import PurePath

import morphogen

#: The modules left out of the public interface: the command line (its commands are
#: public, not the functions behind them), the graphical interface (a web page and its
#: server) and the compiled kernels
INTERNAL = ("morphogen.cli", "morphogen.gui", "morphogen.kernels")

#: The types of the module attributes counted as public data
DATA = (bool, int, float, complex, str, bytes, tuple, list, dict, set, frozenset)


def is_public_module(name: str) -> bool:
    """Whether a module of morphoGen is part of the public interface."""
    if any(part.startswith("_") for part in name.split(".")):
        return False
    return not any(name == m or name.startswith(m + ".") for m in INTERNAL)


def public_modules() -> list[tuple[str, bool]]:
    """The modules of the public interface, ``(name, is a package)``, sorted by name."""
    found = [("morphogen", True)]
    for info in pkgutil.walk_packages(morphogen.__path__, "morphogen."):
        if is_public_module(info.name):
            found.append((info.name, info.ispkg))
    return sorted(found)


def _compiled(obj) -> bool:  # a Numba or CUDA kernel
    return type(obj).__module__.split(".")[0] in ("numba", "cupy")


def _assigned(module) -> set[str] | None:
    """The names that a module assigns at its top level (in its branches too), from its
    source; None without a source."""
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError, SyntaxError):
        return None
    names = set()
    nodes = list(tree.body)
    while nodes:
        node = nodes.pop()
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            nodes.extend(node.body)
            nodes.extend(getattr(node, "orelse", []))
            nodes.extend(getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                nodes.extend(handler.body)
    return names


def defined_names(module) -> list[tuple[str, object]]:
    """The public names that a module defines: its functions and classes, and the data
    that it assigns (not what it imports), sorted by name."""
    names = []
    assigned = _assigned(module)
    for attr, obj in vars(module).items():
        if attr.startswith("_") or inspect.ismodule(obj) or _compiled(obj):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            if obj.__module__ == module.__name__:
                names.append((attr, obj))
        elif isinstance(obj, DATA) and (assigned is None or attr in assigned):
            names.append((attr, obj))
    return sorted(names, key=lambda item: item[0])


def _default(value) -> str:
    if isinstance(value, PurePath):  # the same on every system (not WindowsPath, PosixPath)
        return f"Path({value.as_posix()!r})"
    return re.sub(r" at 0x[0-9A-Fa-f]+", "", repr(value))


def parameters(obj) -> str:
    """The parameters of a function or class, without their annotations: their names,
    kinds and defaults (``(a, b=1, *, c=None, **options)``)."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    parts, positional_only, keyword_only = [], False, False
    for p in signature.parameters.values():
        if p.kind is p.POSITIONAL_ONLY:
            positional_only = True
        elif positional_only:
            parts.append("/")
            positional_only = False
        if p.kind is p.KEYWORD_ONLY and not keyword_only:
            parts.append("*")
            keyword_only = True
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
            keyword_only = True
        elif p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
        elif p.default is p.empty:
            parts.append(p.name)
        else:
            parts.append(f"{p.name}={_default(p.default)}")
    if positional_only:
        parts.append("/")
    return "(" + ", ".join(parts) + ")"


def _members(cls) -> list[str]:
    """The public methods and properties that a class defines."""
    lines = []
    for attr, raw in sorted(vars(cls).items()):
        if attr.startswith("_"):
            continue
        if isinstance(raw, property):
            lines.append(f".{attr}")
        elif isinstance(raw, (classmethod, staticmethod)):
            lines.append(f".{attr}{parameters(raw.__func__)}")
        elif inspect.isfunction(raw):
            lines.append(f".{attr}{parameters(raw)}")
    return lines


def public_interface() -> list[str]:
    """One line per public name: the names of the ``__all__`` of each package, then the
    functions (with their parameters), classes (with their public methods and
    properties) and data that each module defines."""
    lines = []
    for name, is_package in public_modules():
        module = importlib.import_module(name)
        if is_package:
            lines.extend(f"{name}: {attr}" for attr in getattr(module, "__all__", []))
        for attr, obj in defined_names(module):
            if inspect.isclass(obj):
                lines.append(f"{name}.{attr}{parameters(obj)}")
                lines.extend(f"{name}.{attr}{member}" for member in _members(obj))
            elif inspect.isfunction(obj):
                lines.append(f"{name}.{attr}{parameters(obj)}")
            else:
                lines.append(f"{name}.{attr}")
    return lines
