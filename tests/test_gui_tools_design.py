"""The tools of the tab Design: the automaton rule of a morphology, and the rigidity of
strut lattices along their dilution."""

import pytest

from morphogen.domain import Domain
from morphogen.gui import api
from morphogen.gui.tools import ToolJob
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed


def _run(session, name, params):
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()
    status = job.status()
    assert status["state"] == "done", status.get("error")
    return status["result"]


def test_automaton_rule_of_a_morphology(tmp_path):
    """A structure of the majority rule: the rule found reproduces its descriptors well
    enough, and the structure of the rule joins the workspace."""
    session = api.Session(tmp_path)
    model = model_from_config(
        {"type": "cellular_automaton", "rule": "majority", "steps": 10, "cell_size": 0.25}
    )
    micro = model.generate(Domain(size=6.0, periodic=True), derive_seed(3, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "majority", micro=micro, model=model.to_config(), seed=3)
    out = _run(session, "automaton_rule",
               {"source": key, "cells": 16, "generations": 4, "population": 8})  # fmt: skip
    assert out["values"]["rule"].startswith("B") and out["values"]["cost"] < 1.0
    (item,) = out["items"]
    assert session.item(item["key"])["model"]["rule"] == out["values"]["rule"]
    target = next(t for t in out["tables"] if t["title"].startswith("descriptors"))
    assert target["rows"][0][0] == "volume fraction"


def test_automaton_rule_on_a_tiling(tmp_path):
    """The rule of a planar morphology searched on the hexagonal lattice: the structure of
    the rule found is grown on hexagons."""
    session = api.Session(tmp_path)
    model = model_from_config(
        {"type": "cellular_automaton", "lattice": "hexagonal", "dimension": 2, "steps": 8}
    )
    micro = model.generate(Domain(size=(16.0, 16.0, 1.0), periodic=True), derive_seed(3, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "hexagons", micro=micro, model=model.to_config(), seed=3,
                planar=True)  # fmt: skip
    out = _run(session, "automaton_rule",
               {"source": key, "cells": 16, "generations": 2, "population": 6,
                "lattice": "hexagonal"})  # fmt: skip
    (item,) = out["items"]
    assert session.item(item["key"])["model"]["lattice"] == "hexagonal"
    assert out["values"]["rule"].startswith("B")


def test_rigidity_of_a_strut_lattice(tmp_path):
    """The octet lattice is stretching-dominated (Z = 12) and keeps no mechanism while
    diluted to Z = 6.7, where Maxwell's count leaves states of self-stress; below 6 it
    has mechanisms and floppy strains."""
    session = api.Session(tmp_path)
    out = _run(session, "rigidity", {"lattice": "octet", "cells": 2})
    rows = out["tables"][0]["rows"]
    columns = out["tables"][0]["columns"]
    col = {c: k for k, c in enumerate(columns)}
    assert rows[0][col["coordination"]] == pytest.approx(12.0)
    assert rows[0][col["mechanisms"]] == 0 and rows[0][col["floppy_strains"]] == 0
    low = rows[-1]
    assert low[col["coordination"]] < 6 and low[col["mechanisms"]] > 0
    assert out["values"]["stretching-dominated (full)"] is True
    assert 6.0 < out["values"]["lowest stretching-dominated coordination"] < 12.0
