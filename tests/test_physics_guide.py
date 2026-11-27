"""The guide to the physics of the documentation (docs/physics.md) stays correct: its
Python snippets run one after another in one session and give the physics they announce;
its configuration of an RVE study is valid, and runs at a smaller size; each of its
``morphogen`` commands is one that the command line parses."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import numpy as np
import pytest
import yaml

from morphogen import cli

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "physics.md"
if not GUIDE.parent.is_dir():  # an installed package without its sources
    pytest.skip("the documentation is not here", allow_module_level=True)


def _blocks(language: str) -> list[str]:
    """The fenced code blocks of a language, in their order."""
    text = GUIDE.read_text(encoding="utf-8")
    return re.findall(rf"^```{language}\n(.*?)^```$", text, flags=re.M | re.S)


@pytest.mark.slow
def test_guide_snippets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    blocks = _blocks("python")
    assert len(blocks) >= 11  # a sample, elasticity, transport, couplings, dynamics, paths...
    ns: dict = {"__name__": "guide"}
    for block in blocks:
        exec(compile(block, str(GUIDE), "exec"), ns)
    f = float(ns["image"].data.mean())
    # elasticity and transport between the bounds of Reuss and Voigt (Wiener)
    k0, k1 = (m.moduli()[0] for m in (ns["matrix"], ns["sphere"]))
    assert 1 / ((1 - f) / k0 + f / k1) < ns["bulk"] < (1 - f) * k0 + f * k1
    assert 1 / ((1 - f) / 0.2 + f / 30.0) < ns["k"] < (1 - f) * 0.2 + f * 30.0
    assert np.allclose(ns["eps"], ns["eps"].T) and 3.0 < ns["eps"][0, 0] < 10.0
    assert 1.0 < ns["tortuosity"] < 2.0  # impermeable spheres: about 1 + f / 2
    # weak couplings: the glass restrains the resin; a porous rock
    assert 5e-6 < ns["alpha"][0, 0] < 60e-6 and 0.0 < ns["beta"] < 0.4
    porous = ns["porous"]
    assert f < porous["biot_coefficient"] < 1.0 and 0.0 < porous["skempton_coefficient"] < 1.0
    # in time: the moisture taken up and the swelling of the cell grow, free of mean stress
    wet = ns["wet"]
    assert np.all(np.diff(wet.history["uptake"]) > 0) and wet.history["uptake"][-1] < 1.0
    assert np.all(np.diff(wet.mean_strain[:, 1]) > 0) and wet.mean_strain[0, 1] > 0
    assert np.abs(wet.history["mean_stress"][:, 1:3]).max() < 1e-3
    # the piezoelectric composite: a weaker coupling than the ceramic, of the same sign
    assert ns["law"]["piezoelectric"].shape == (3, 6) and 0.0 < ns["law"]["d33"] < 1.9e-10
    # dynamics: a loss peak between the two relaxation times, an absorbing index
    assert ns["dma"]["shear_tan_delta_2"] > ns["dma"]["shear_tan_delta_3"] > 0.0
    assert 1.5 < ns["index"]["index_1"] < 2.4 and ns["index"]["extinction_1"] > 0.0
    (low, high), *_ = ns["gaps"]
    assert low == pytest.approx(0.32, abs=0.02) and high == pytest.approx(0.44, abs=0.02)
    # a mat of fibres: a permeability of the order of the squared gap between the fibres,
    # and a layer of 4 cm that absorbs most of the sound above 1 kHz
    sound, layer = ns["sound"], ns["layer"]["absorption"]
    assert 1e-10 < sound["permeability1"] < 1e-8 and 1.0 < sound["tortuosity1"] < 1.2
    assert layer[0] < layer[1] < layer[2] and 0.3 < layer[0] and layer[3] > 0.9
    relax = ns["relax"]
    assert relax.stress[4, 0] > relax.stress[-1, 0] > 0.0  # relaxation at a held strain
    # softening responses: a peak, then less stress; the phase field breaks the section
    curve, broken = ns["curve"], ns["broken"]
    assert curve.strength() > 1.2 * curve.stress[-1, 0] and 0.0 < curve.plastic[-1] < 1.0
    assert broken.stress[-1, 0] < 0.05 * broken.strength()
    assert broken.fracture_energy[-1] == pytest.approx(broken.work(), rel=0.6)
    # finite strains: the logarithmic strain of a stretch of 1.5, a true stress above the
    # nominal one, a filled rubber about twice as stiff as its matrix
    alone, reinforced = ns["alone"], ns["reinforced"]
    assert alone.log_strain[-1, 0] == pytest.approx(np.log(1.5))
    assert alone.cauchy[-1, 0] > alone.stress[-1, 0] > 0.0
    assert 1.6 < reinforced.stress[-1, 0] / alone.stress[-1, 0] < 2.4
    # the notched strip tears near the stretch of Rivlin and Thomas (1.5), its crack as
    # large as the section of the strip (regularized on voxels: 1 + 3 h / (8 l))
    torn = ns["torn"]
    stretch, stress = 1.0 + torn.strain[:, 1], torn.stress[:, 1]
    peak = int(np.argmax(stress))
    gone = np.flatnonzero(stress[peak:] < 0.1 * stress[peak])
    assert gone.size and 1.4 < stretch[peak + gone[0]] < 1.65
    assert 1.0 < torn.crack_surface[-1] < 1.3
    # the level set cut by extended finite elements: the modulus of the voxels within 3 %
    assert ns["cut"]["bulk_modulus.periodic"].value == pytest.approx(ns["bulk"], rel=0.03)


def test_guide_configuration_runs(tmp_path, monkeypatch):
    from morphogen.config import load_config

    monkeypatch.chdir(tmp_path)
    (config,) = _blocks("yaml")
    spec = yaml.safe_load(config)
    cfg = load_config(spec)  # as written
    assert "dynamic_stiffness.shear_loss_1" in cfg.rve.properties
    assert {"pull.strength", "swelling.mean", "pull.S11"} <= set(cfg.rve.properties)
    # at a smaller size
    spec["voxels"]["voxel_size"] = 1.0
    spec["rve"].update(sizes=[4.0, 5.0], realizations=[3, 2], bootstrap=0)
    (tmp_path / "rve.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    assert cli.main(["run", "rve.yaml", "-q"]) == 0
    out = tmp_path / spec["outputs"]["directory"]
    report = (out / "rve_report.md").read_text(encoding="utf-8")
    for name in ("bulk_modulus", "swelling.mean", "dynamic_stiffness.shear_loss_1",
                 "pull.strength"):  # fmt: skip
        assert name in report, name


def test_guide_commands_parse():
    parser = cli.build_parser()
    commands = []
    for block in _blocks("bash"):
        for line in block.splitlines():
            words = shlex.split(line, comments=True)
            if words and words[0] == "morphogen":
                commands.append(words)
    assert {words[1] for words in commands} == {"run", "homogenize", "bands"}
    properties = set()
    for words in commands:
        args = parser.parse_args(words[1:])
        assert callable(args.func), words
        properties |= set(getattr(args, "property", []))
    assert {"stiffness", "conductivity", "piezoelectric", "dynamic_stiffness",
            "poroelastic", "hydraulic_permeability", "acoustic"} <= properties  # fmt: skip
