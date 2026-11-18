"""Explicit dynamics with OpenRadioss: crushing, tension, shear and impacts of a
morphology at the rates of crashes, shocks and drops.

morphoGen writes the inputs of the explicit finite element solver OpenRadioss for the
mesh of a morphology (its voxels or a conforming mesh of its level set), the materials of
its phases and a loading (:mod:`morphogen.io.radioss`), runs them (:func:`run_openradioss`,
an installation found or installed by :func:`install_openradioss`) and reads the results
back (:func:`read_explicit_run`): the nominal stress-strain curves of its faces, its
energies and its fields over time. :func:`explicit_test` does it all.

>>> from morphogen import dynamics
>>> test = dynamics.explicit_test(micro, {0: "void", 1: {"young": 70e3, "poisson": 0.3,
...     "density": 2.7e-9, "plasticity": {"yield": 200, "hardening": 500}}},
...     {"kind": "compression", "strain": 0.5, "strain_rate": 500}, "crush")  # doctest: +SKIP
>>> dynamics.summary(test.result)["plateau stress (20-40 %)"]  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from morphogen.dynamics.model import (
    LOADINGS,
    ExplicitLoading,
    ExplicitMaterial,
    JohnsonCook,
    UnitSystem,
    critical_time_step,
    element_lengths,
    explicit_materials,
    explicit_mesh,
)
from morphogen.dynamics.openradioss import (
    OpenRadioss,
    RadiossRun,
    convert_outputs,
    default_home,
    find_openradioss,
    has_openradioss,
    install_openradioss,
    release_asset,
    release_info,
    run_openradioss,
)
from morphogen.dynamics.results import (
    ExplicitFrame,
    ExplicitResult,
    read_explicit_run,
    summary,
    write_frames,
    write_tables,
)


@dataclass
class ExplicitTest:
    """An explicit dynamic test of :func:`explicit_test`: its inputs
    (:class:`~morphogen.io.radioss.RadiossInputs`), its run (None when not run) and its
    results read back (None when not run)."""

    inputs: object
    run: RadiossRun | None = None
    result: ExplicitResult | None = None

    @property
    def directory(self) -> Path:
        return Path(self.inputs.directory)


def explicit_test(
    micro,
    materials,
    loading,
    directory,
    *,
    mesh: str = "voxels",
    resolution: int = 24,
    element_size: float | None = None,
    order: int = 1,
    units=None,
    density: float | None = None,
    name: str | None = None,
    frames: int = 50,
    history: int = 1000,
    time_step: float | None = None,
    length_scale: float = 1.0,
    run: bool = True,
    threads: int | None = None,
    mpi: int | None = None,
    installation=None,
    progress=None,
    phase_names: dict | None = None,
) -> ExplicitTest:
    """An explicit dynamic test of a morphology in ``directory``: its mesh
    (:func:`explicit_mesh`: ``mesh`` ``"voxels"`` of ``resolution`` voxels along its
    longest edge, or ``"conforming"`` tetrahedra of ``element_size`` and ``order``), the
    inputs of OpenRadioss (:func:`morphogen.io.write_radioss`: ``materials`` a mapping
    phase -> material, :func:`explicit_materials`, ``density`` that of the materials
    that give none; ``loading`` an :class:`ExplicitLoading` or its configuration;
    every value in the ``units``, a :class:`UnitSystem`, Mg mm s by default, the lengths
    of the morphology times ``length_scale``), then, with ``run``, their run
    (:func:`run_openradioss`)
    and their results (:func:`read_explicit_run`). ``progress(fraction, message)``
    follows the run."""
    from morphogen.io.radioss import run_name, write_radioss

    spec = explicit_materials(materials, density=density)
    loading = ExplicitLoading.from_config(loading)
    if progress:
        progress(0.0, "mesh")
    shape = explicit_mesh(
        micro, spec, mesh=mesh, resolution=resolution, element_size=element_size, order=order
    )
    model = getattr(micro, "kind", None) or type(micro).__name__.lower()
    names = phase_names
    if names is None:
        names = {int(k): str(v) for k, v in (getattr(micro, "phase_names", None) or {}).items()}
    inputs = write_radioss(
        directory,
        shape,
        {q: m for q, m in spec.items() if m is not None},
        loading,
        name=run_name(name or model),
        units=units,
        frames=frames,
        history=history,
        time_step=time_step,
        length_scale=length_scale,
        phase_names=names,
        title=f"{loading.kind} of {model}",
    )
    test = ExplicitTest(inputs)
    if not run:
        return test

    def scaled(fraction, message=""):
        if progress:
            progress(0.05 + 0.9 * fraction, message)

    test.run = run_openradioss(
        directory, inputs.name, threads=threads, mpi=mpi, installation=installation, progress=scaled
    )
    test.result = read_explicit_run(directory)
    test.result.warnings[:0] = test.run.warnings
    if progress:
        progress(1.0, "done")
    return test


__all__ = [
    "LOADINGS",
    "ExplicitFrame",
    "ExplicitLoading",
    "ExplicitMaterial",
    "ExplicitResult",
    "ExplicitTest",
    "JohnsonCook",
    "OpenRadioss",
    "RadiossRun",
    "UnitSystem",
    "convert_outputs",
    "critical_time_step",
    "default_home",
    "element_lengths",
    "explicit_materials",
    "explicit_mesh",
    "explicit_test",
    "find_openradioss",
    "has_openradioss",
    "install_openradioss",
    "read_explicit_run",
    "release_asset",
    "release_info",
    "run_openradioss",
    "summary",
    "write_frames",
    "write_tables",
]
