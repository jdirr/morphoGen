"""Graphical interface in the web browser (``morphogen gui``), served locally with the
standard library: generate realizations of any model from forms built on the
parameters' metadata, watch automata and growth processes form structures, optimize
periodic cells live, and explore the design space of tile assemblies."""

from morphogen.gui.server import serve

__all__ = ["serve"]
