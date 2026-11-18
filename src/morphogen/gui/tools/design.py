"""The tools of the tab Design: the rule of a cellular automaton whose structures match a
morphology (inverse design by a genetic algorithm over the birth and survival sets), and
the rigidity of strut lattices with defects (the counts of Maxwell and Calladine:
mechanisms, states of self-stress, floppy macroscopic strains)."""

from __future__ import annotations

import numpy as np

from morphogen.automata.tilings import LATTICES, PLANAR, QUASIPERIODIC
from morphogen.gui.tools import Field, tab, tool

tab(
    "design",
    "Inverse design",
    "Inverse design: the automaton rule whose structures match a morphology of the "
    "workspace, and the rigidity of strut lattices (stretching- or bending-dominated) as "
    "their struts are removed.",
    order=75,
)


@tool(
    "automaton_rule",
    "design",
    "Automaton rule of a morphology",
    "The rule B.../S... of a cellular automaton, and the density of its initial noise, "
    "whose structures best match a morphology of the workspace: its volume fraction, "
    "interface density and two-point correlation at a few lags, searched by a genetic "
    "algorithm over the birth and survival sets (all candidates run from the same noise). "
    "The structure of the rule found joins the workspace.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("cells", "int", 24, label="cells per edge", doc="of the target and of the runs"),
        Field("neighbourhood", "choice", "moore", choices=("moore", "von_neumann", "edges")),
        Field("lattice", "choice", "square", choices=LATTICES,
              doc="cells of the automaton: squares or cubes, triangles, hexagons, Penrose "
              "rhombi, Ammann-Beenker squares and rhombi (prisms in 3D), BCC or FCC Voronoi "
              "cells, golden rhombohedra (3D)"),
        Field("steps", "int", 15, doc="updates of each candidate"),
        Field("population", "int", 16),
        Field("generations", "int", 10),
        Field("seed", "int", 1),
    ],
    order=1,
)  # fmt: skip
def automaton_rule(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen.design.search import fit_rule
    from morphogen.domain import Domain
    from morphogen.gui import api
    from morphogen.models.base import model_from_config
    from morphogen.seeding import derive_seed

    item = p["source"]
    micro = item["micro"]
    lengths = np.asarray(micro.domain.lengths, dtype=float)
    n = int(p["cells"])
    if not 8 <= n <= 64:
        raise ValueError("cells per edge: from 8 to 64")
    planar = bool(item.get("planar")) or lengths[2] < 0.05 * lengths[:2].max()
    cell = float(lengths.max()) / n
    target = micro.voxelize(voxel_size=cell)
    if planar:
        mid = target.data.shape[2] // 2
        target_data = target.data[:, :, mid : mid + 1]
    else:
        target_data = target.data
    shape = target_data.shape
    total = int(p["generations"])

    def progress(g, best, cost):
        ctx.progress((g + 1) / max(total, 1), f"generation {g + 1} / {total}, cost {cost:.3g}")

    lattice = p.get("lattice") or "square"
    if planar and lattice not in PLANAR:
        raise ValueError(f"lattice {lattice}: three-dimensional morphologies only")
    periodic = lattice not in QUASIPERIODIC
    fit = fit_rule(
        (target_data == 1).astype(np.uint8),
        neighbourhood=p["neighbourhood"],
        dims=2 if planar else 3,
        lattice=lattice,
        shape=shape,
        steps=int(p["steps"]),
        population=int(p["population"]),
        generations=total,
        rng=np.random.default_rng(p["seed"]),
        callback=progress,
    )
    spec = {"type": "cellular_automaton", "rule": fit.rule, "steps": int(p["steps"]),
            "neighbourhood": p["neighbourhood"], "initial_fraction": float(fit.initial_fraction),
            "cell_size": cell}  # fmt: skip
    if planar:
        spec["dimension"] = 2
    if lattice != "square":
        spec["lattice"] = lattice
    model = model_from_config(spec)
    domain = Domain(size=tuple(float(v) for v in lengths), periodic=periodic)
    grown = model.generate(domain, derive_seed(p["seed"], 0))
    views = api.morphology_views(grown, 384)
    names = ["volume fraction", "interface density"] + [
        f"correlation, lag {k}" for k in range(1, len(fit.target) - 1)
    ]
    ctx.table(
        [{"descriptor": name, "target": float(t), "rule": float(d)}
         for name, t, d in zip(names, fit.target, fit.descriptors, strict=True)],
        "descriptors", "descriptors of the target and of the rule found",
    )  # fmt: skip
    history = fit.history.get("best") or fit.history.get("cost") or []
    if history:
        fig = Figure(figsize=(5.4, 3.4), layout="constrained")
        ax = fig.subplots()
        ax.plot(np.arange(1, len(history) + 1), history, lw=1.8)
        ax.set_xlabel("generation")
        ax.set_ylabel("cost of the best rule")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ctx.figure(fig, "search", "the genetic search")
    for s in views["sections"]:
        ctx.figures.append({"title": f"rule {fit.rule}, {s['title']}", "url": s["png"]})
    ctx.add(
        "morphology",
        f"automaton {fit.rule} (from #{item['number']})",
        micro=grown,
        origin="design",
        source=item["key"],
        model=model.to_config(),
        domain={"size": [float(v) for v in lengths], "periodic": periodic},
        seed=p["seed"],
        phases={"0": "phase 0", "1": "phase 1"},
        thumbnail=views["thumbnail"],
        fraction=views["fraction"],
        planar=planar,
        info={"model": "cellular_automaton", "rule": fit.rule},
    )
    return {
        "values": {
            "rule": fit.rule,
            "initial fraction": float(fit.initial_fraction),
            "cost": float(fit.cost),
            "evaluations": int(fit.evaluations),
        }
    }


LATTICES = ("cubic", "bcc", "fcc", "octet", "kelvin", "diamond", "pyrochlore")


@tool(
    "rigidity",
    "design",
    "Rigidity of a strut lattice",
    "The periodic pin-jointed framework of a strut lattice (cubic, bcc, fcc, octet, "
    "Kelvin, diamond, pyrochlore), its struts removed at random (to a mean coordination, "
    "or a fraction of them) "
    "and its nodes perturbed: the mechanisms, the states of self-stress and the floppy "
    "macroscopic strains of its compatibility matrix (Maxwell 1864, Calladine 1978, "
    "Pellegrino & Calladine 1986; Guest & Hutchinson 2003), whether it is stretching- or "
    "bending-dominated, along a series of dilutions.",
    [
        Field("lattice", "choice", "octet", choices=LATTICES),
        Field("cells", "int", 2, label="cells per edge"),
        Field("dilution", "choice", "coordination", choices=(
            ("coordination", "to a mean coordination"), ("fraction", "a fraction of the struts"))),
        Field("targets", "floats", label="coordinations or fractions",
              doc="blank: from the full lattice down to coordination 4 (or fractions 0 to 0.5)"),
        Field("perturbation", "float", 0.0, label="node perturbation", unit="length",
              doc="radius of the random displacement of the nodes (a fraction of the cell)"),
        Field("seed", "int", 1),
    ],
    order=2,
)  # fmt: skip
def rigidity(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen.design.frames import Framework

    base = Framework.lattice(p["lattice"], int(p["cells"]))
    rng = np.random.default_rng(p["seed"])
    full = base.coordination()
    if p["dilution"] == "coordination":
        targets = p["targets"] or [float(z) for z in np.linspace(full, 4.0, 7)]
    else:
        targets = p["targets"] or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    for k, target in enumerate(targets):
        ctx.progress(k / max(len(targets), 1), f"dilution {k + 1} / {len(targets)}")
        frame = base
        if p["dilution"] == "coordination" and target < full:
            frame = base.diluted(rng, coordination=float(target))
        elif p["dilution"] == "fraction" and target > 0:
            frame = base.diluted(rng, fraction=float(target))
        if p["perturbation"]:
            frame = frame.perturbed(rng, float(p["perturbation"]))
        r = frame.rigidity()
        rows.append({"target": float(target), **r})
    ctx.table(rows, "rigidity", "rigidity along the dilution")
    fig = Figure(figsize=(5.6, 3.8), layout="constrained")
    ax = fig.subplots()
    z = [r["coordination"] for r in rows]
    ax.plot(z, [r["mechanisms"] for r in rows], "o-", label="mechanisms")
    ax.plot(z, [r["self_stresses"] for r in rows], "s-", label="states of self-stress")
    ax.plot(z, [r["floppy_strains"] for r in rows], "^-", label="floppy macroscopic strains")
    ax.axvline(6.0, color="0.5", lw=1, ls="--")
    ax.text(6.0, ax.get_ylim()[1] * 0.95, " Z = 6", va="top", fontsize=8, color="0.4")
    ax.set_xlabel("mean coordination Z")
    ax.set_ylabel("count")
    ax.set_title(f"{p['lattice']} lattice, {int(p['cells'])}³ cells", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ctx.figure(fig, "rigidity", "Maxwell–Calladine counts")
    first = rows[0]
    return {
        "values": {
            "lattice": p["lattice"],
            "coordination of the full lattice": float(full),
            "stretching-dominated (full)": bool(first["stretching_dominated"]),
            "lowest stretching-dominated coordination": min(
                (r["coordination"] for r in rows if r["stretching_dominated"]), default=float("nan")
            ),
        }
    }
