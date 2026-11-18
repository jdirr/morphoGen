"""Database of the configurations tested and their results, filters and Ashby charts."""

import json

import numpy as np
import pytest

from morphogen import cli, pipeline
from morphogen.config import load_config
from morphogen.results import (
    OTHER,
    ResultsDatabase,
    ashby_chart,
    ashby_points,
    axis_label,
    default_axes,
    flatten,
    matches,
    parse_condition,
    write_records,
)


def test_flatten_and_conditions():
    flat = flatten({"model": {"type": "tpms", "cell_size": [1, 2]}, "seed": 3})
    assert flat == {"model.type": "tpms", "model.cell_size": "[1, 2]", "seed": 3}
    assert parse_condition("solid_fraction <= 0.3") == ("solid_fraction", "<=", 0.3)
    assert parse_condition("model.type in gyroid, octet") == (
        "model.type",
        "in",
        ["gyroid", "octet"],
    )
    assert parse_condition("domain.periodic == true")[2] is True
    record = {"model.type": "tpms", "solid_fraction": 0.2, "domain.periodic": True, "n": 3.0}
    assert matches(record, ["model.type == tpms", "solid_fraction < 0.25", "n == 3"])
    assert matches(record, ["domain.periodic == true", "model.type ~ tp*", "missing != 1"])
    assert not matches(record, ["model.type in voronoi,laguerre"])
    assert not matches(record, ["missing == 1"]) and not matches(record, ["model.type > 1"])
    assert matches(record, ["model.type not in voronoi", "missing not in 1"])
    with pytest.raises(ValueError, match="key OP value"):
        parse_condition("solid_fraction")


def lattice_run(tmp_path, name, cell, fraction, database):
    cfg = {
        "model": {"type": "strut_lattice", "cell": cell, "volume_fraction": fraction},
        "domain": {"size": 1.0, "periodic": True},
        "seed": 1,
        "voxels": {"shape": 12},
        "homogenization": {
            "materials": {0: "void", 1: {"young": 200.0, "poisson": 0.3, "density": 7.8}},
            "properties": ["stiffness"],
            "tolerance": 1e-6,
        },
        "outputs": {"directory": str(tmp_path / name), "name": cell, "database": str(database)},
    }
    return pipeline.run(load_config(cfg))


def test_database_of_runs(tmp_path):
    db_path = tmp_path / "results.db"
    for cell, phi in (("octet", 0.2), ("octet", 0.35), ("kelvin", 0.3)):
        lattice_run(tmp_path, f"{cell}_{phi}", cell, phi, db_path)
    with ResultsDatabase(db_path) as db:
        runs = db.runs()
        assert [r["name"] for r in runs] == ["octet", "octet", "kelvin"]
        records = db.records()
        assert len(records) == 3 and all(r["model.type"] == "strut_lattice" for r in records)
        octet = db.records(["model.cell == octet", "solid_fraction > 0.25"])
        assert len(octet) == 1 and octet[0]["model.volume_fraction"] == 0.35
        e = octet[0]["stiffness.periodic.E1.relative"]
        assert 0 < e < octet[0]["solid_fraction"]  # below the rule of mixtures
        assert octet[0]["density"] == pytest.approx(7.8 * octet[0]["solid_fraction"])
        keys = db.keys()
        assert keys["solid_fraction"]["numeric"] and not keys["model.cell"]["numeric"]
        # adding a run again replaces it; runs are removed by id, path or name
        db.add_run(tmp_path / "octet_0.2", name="again")
        assert len(db.runs()) == 3 and db.runs()[-1]["name"] == "again"
        assert db.remove("kelvin") == 1 and db.remove(runs[1]["id"]) == 1
        assert len(db.records()) == 1
        n = db.add_records([{"realization": 0, "x": 1.0}], name="external")
        assert n == 1 and db.records(["run_name == external"])[0]["x"] == 1.0
        with pytest.raises(ValueError, match=r"run\.json"):
            db.add_run(tmp_path)
    out = write_records(records, tmp_path / "table.csv", ["run_name", "solid_fraction"])
    assert out.read_text(encoding="utf-8").splitlines()[0] == "run_name,solid_fraction"


def synthetic(n_families=4, per=5, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for f in range(n_families):
        for k in range(per + f):  # families of different sizes
            phi = float(rng.uniform(0.05, 0.5))
            records.append(
                {
                    "run_name": f"run{f}",
                    "realization": k,
                    "model.type": f"family{f}",
                    "solid_fraction": phi,
                    "stiffness.periodic.E1.relative": phi ** (1 + f / 3),
                }
            )
    records.append({"model.type": "broken", "solid_fraction": 0.1})  # no property: left out
    return records


def test_ashby_points():
    records = synthetic()
    assert default_axes(records) == ("solid_fraction", "stiffness.periodic.E1.relative")
    data = ashby_points(records)
    names = [g["name"] for g in data["groups"]]
    # the three largest families in colour, the others folded
    assert names == ["family3", "family2", "family1", OTHER]
    assert data["groups"][-1]["members"] == ["family0"]
    assert sum(len(g["points"]) for g in data["groups"]) == len(records) - 1
    data = ashby_points(records, groups=["family0"], by="model.type")
    assert data["groups"][0]["name"] == "family0"
    one = ashby_points(records, by=None)
    assert len(one["groups"]) == 1 and one["groups"][0]["slot"] == 0
    linear = ashby_points([{"a": -1.0, "b": 2.0}], "a", "b", None, log=False)
    assert len(linear["groups"][0]["points"]) == 1
    assert ashby_points([{"a": -1.0, "b": 2.0}], "a", "b", None)["groups"] == []
    with pytest.raises(ValueError, match="relative property"):
        default_axes([{"x": 1.0}])
    assert axis_label("stiffness.periodic.E1.relative") == (
        "Relative Young's modulus E1 (E / Es, periodic)"
    )
    assert axis_label("bulk_modulus.kubc.normalized").startswith("Normalized bulk modulus")
    assert axis_label("solid_fraction").startswith("Relative density")


def test_ashby_charts(tmp_path):
    pytest.importorskip("matplotlib")
    records = synthetic()
    for name, theme in (("chart.png", "light"), ("chart.svg", "dark")):
        path = ashby_chart(records, tmp_path / name, theme=theme, title="Lattices")
        assert path.stat().st_size > 1000
    svg = (tmp_path / "chart.svg").read_text(encoding="utf-8")
    assert "Lattices" in svg and "n = 1" in svg and "family3" in svg
    ashby_chart(records, tmp_path / "one.png", by=None, log=False)
    with pytest.raises(ValueError, match="no record"):
        ashby_chart(records, tmp_path / "x.png", "missing", "solid_fraction")
    with pytest.raises(ValueError, match="theme"):
        ashby_chart(records, tmp_path / "x.png", theme="blue")


def test_cli_database_and_chart(tmp_path, capsys):
    db = tmp_path / "r.db"
    run = lattice_run(tmp_path, "octet", "octet", 0.25, tmp_path / "unused.db")
    assert cli.main(["db", "add", str(run), "--db", str(db)]) == 0
    assert cli.main(["db", "list", "--db", str(db), "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["db", "list", "--db", str(db), "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["records"] == 1 and listing["runs"][0]["name"] == "octet"
    assert cli.main(["db", "keys", "--db", str(db), "--where", "model.cell == octet"]) == 0
    assert "solid_fraction" in capsys.readouterr().out
    out = tmp_path / "all.csv"
    assert cli.main(["db", "export", "--db", str(db), "-o", str(out)]) == 0
    assert "model.cell" in out.read_text(encoding="utf-8").splitlines()[0]
    if pytest.importorskip("matplotlib"):
        chart, table = tmp_path / "a.svg", tmp_path / "points.csv"
        args = ["ashby", "--db", str(db), "-o", str(chart), "--table", str(table)]
        assert cli.main(args) == 0
        assert chart.exists() and table.read_text(encoding="utf-8").count("\n") == 2
    assert cli.main(["db", "remove", "1", "--db", str(db)]) == 0
    assert cli.main(["db", "remove", "1", "--db", str(db)]) == 2
    assert cli.main(["ashby", "--db", str(db), "-o", str(tmp_path / "b.png")]) == 2
    assert "no record" in capsys.readouterr().err
    # a run recorded through the command line
    cfg = tmp_path / "c.yaml"
    run_cfg = json.loads((run / "run.json").read_text(encoding="utf-8"))["config"]
    run_cfg["outputs"] = {"directory": str(tmp_path / "again"), "name": "again"}
    cfg.write_text(json.dumps(run_cfg))
    assert cli.main(["run", str(cfg), "--db", str(db), "-q"]) == 0
    with ResultsDatabase(db) as d:
        assert [r["name"] for r in d.runs()] == ["again"]
