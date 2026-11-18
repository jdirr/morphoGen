"""Configuration files of topology optimizations (``morphogen optimize``).

Example::

    optimization:
      objective: poisson_ratio      # bulk_modulus, shear_modulus, conductivity, target
      volume_fraction: 0.35
      grid: [64, 64, 1]             # voxels of the cell (nz = 1: planar, prismatic cell)
      cell_size: 1.0                # edge of the cell along x
      material: {young: 1.0, poisson: 0.3}
      method: mma                   # oc, mma, beso, hca
      filter_radius: 3.0            # voxels
      projection: 8.0               # final sharpness of the Heaviside projection
      symmetry: cubic               # none, orthotropic, cubic (square in 2D)
      iterations: 300
    seed: 1
    outputs:
      directory: out/auxetic
      name: cell
      image: [vti]                  # thresholded design
      density: [vti]                # density field
      geometry: [json]              # lattice microstructure: re-voxelize, mesh, homogenize
      check_refinement: 2           # re-evaluate the design on a grid twice as fine (0: no)

Results: ``optimization.json`` (settings, homogenized tensors of the density and of the
thresholded designs, ``tensor`` and ``binary_tensor``, with their compliances
``compliance`` and ``binary_compliance`` for the elastic designs, ``S = C^-1`` in the
notation stated, engineering constants, convergence, hinges, and the constants of the
design on the finer grid) and ``history.csv``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from morphogen import io
from morphogen._utils import to_jsonable
from morphogen.config import ConfigError, _check_keys, _list_of, read_config_file
from morphogen.homogenization.materials import Material
from morphogen.optimization.topology import CellDesign, CellOptimization

IMAGE_FORMATS = ("vti", "vtk", "mhd", "npz", "npy", "tif", "tiff", "raw")
SETTINGS = {
    "objective",
    "volume_fraction",
    "grid",
    "cell_size",
    "material",
    "method",
    "penalty",
    "filter_radius",
    "projection",
    "symmetry",
    "stiffness_floor",
    "iterations",
    "tolerance",
    "move",
    "target",
    "min_stiffness",
    "evolution_rate",
    "initial",
}


@dataclass
class OptimizationConfig:
    optimization: CellOptimization
    seed: int | None
    directory: Path
    name: str
    image: list[str]
    density: list[str]
    geometry: list[str]
    source: dict
    check_refinement: int = 2


def load_optimization_config(source, *, base_dir=None) -> OptimizationConfig:
    """Validate an optimization configuration (file path or mapping)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = read_config_file(path)
        base = path.parent if base_dir is None else Path(base_dir)
    else:
        data = dict(source)
        base = Path.cwd() if base_dir is None else Path(base_dir)
    _check_keys(
        "configuration",
        data,
        {"optimization", "seed", "outputs", "threads", "version"},
        {"optimization"},
    )
    opt = dict(data["optimization"] or {})
    _check_keys("optimization", opt, SETTINGS, {"volume_fraction", "grid"})
    grid = opt.pop("grid")
    grid = [grid] * 3 if np.ndim(grid) == 0 else list(grid)
    if len(grid) != 3 or any(int(n) != n or n < 1 for n in grid):
        raise ConfigError("optimization.grid: three positive integers (nz = 1 for planar cells)")
    cell_size = float(opt.pop("cell_size", 1.0))
    if not cell_size > 0:
        raise ConfigError("optimization.cell_size: must be positive")
    material = opt.pop("material", None)
    try:
        mat = Material.from_config(material) if material is not None else None
        optimization = CellOptimization(
            tuple(int(n) for n in grid), material=mat, voxel_size=cell_size / grid[0], **opt
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"optimization: {exc}") from None
    seed = data.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ConfigError(f"seed: expected a non-negative integer, got {seed!r}")
    out = dict(data.get("outputs") or {})
    _check_keys(
        "outputs", out, {"directory", "name", "image", "density", "geometry", "check_refinement"}
    )
    check = out.get("check_refinement", 2)
    if isinstance(check, bool) or not isinstance(check, int) or check < 0:
        raise ConfigError(f"outputs.check_refinement: a non-negative integer, got {check!r}")
    directory = Path(out.get("directory", "morphogen_optimization"))
    if not directory.is_absolute():
        directory = base / directory
    return OptimizationConfig(
        optimization=optimization,
        seed=seed,
        directory=directory,
        name=str(out.get("name", "cell")),
        image=_list_of("outputs.image", out.get("image"), IMAGE_FORMATS),
        density=_list_of("outputs.density", out.get("density"), IMAGE_FORMATS),
        geometry=_list_of("outputs.geometry", out.get("geometry"), ("json",)),
        source=data,
        check_refinement=check,
    )


def _compliances(design: CellDesign) -> dict:
    """The compliances ``S = C^-1`` of the elastic tensors of a design (of its densities and
    thresholded), in their notation; None where a tensor is singular or incomplete (the
    tensor of the densities of a planar cell has its in-plane entries only); none for the
    conductivity."""
    from morphogen.homogenization.postprocess import NOTATION, compliance

    if np.shape(design.tensor) != (6, 6):
        return {}
    out = {"notation": NOTATION}
    for key, tensor in (("compliance", design.tensor), ("binary_compliance", design.binary_tensor)):
        out[key] = None
        if np.all(np.isfinite(tensor)):
            try:
                out[key] = compliance(tensor)
            except np.linalg.LinAlgError:  # a thresholded design that falls apart
                pass
    return out


def write_results(design: CellDesign, cfg: OptimizationConfig) -> Path:
    """Write the design files, ``optimization.json`` and ``history.csv``."""
    from morphogen import pipeline

    outdir = cfg.directory
    outdir.mkdir(parents=True, exist_ok=True)
    micro = design.to_microstructure()
    image = micro.voxelize()
    for fmt in cfg.image:
        io.save_image(image, outdir / f"{cfg.name}.{fmt}")
    if cfg.density:
        dens = image.with_data(design.density.astype(np.float32), name="density")
        for fmt in cfg.density:
            io.save_image(dens, outdir / f"{cfg.name}_density.{fmt}")
    for fmt in cfg.geometry:
        io.save_microstructure(micro, outdir / f"{cfg.name}.{fmt}")
    rows = [
        dict(zip(design.history, values, strict=True))
        for values in zip(*design.history.values(), strict=True)
    ]
    pipeline.write_rows(outdir / "history.csv", rows)
    data = {
        "versions": pipeline.versions(),
        "config": cfg.source,
        "settings": design.settings,
        "seed": cfg.seed,
        "iterations": design.iterations,
        "converged": design.converged,
        "objective": design.objective,
        "volume_fraction": design.volume_fraction,
        "binary_fraction": design.binary_fraction,
        "tensor": design.tensor,
        "binary_tensor": design.binary_tensor,
        **_compliances(design),
        "properties": design.properties(),
        "density_properties": design.properties(design.tensor),
        "hinges": design.hinges,
        "check": design.check,
        "time_s": design.time_s,
    }
    (outdir / "optimization.json").write_text(
        json.dumps(to_jsonable(data), indent=2) + "\n", encoding="utf-8"
    )
    return outdir


def run_optimization(cfg: OptimizationConfig, callback=None) -> tuple[CellDesign, Path]:
    """The topology optimization of a configuration: the optimized cell (checked on a refined grid
    with ``check_refinement``) and the folder of its results; ``callback`` follows the
    iterations."""
    t0 = time.time()
    design = cfg.optimization.run(np.random.default_rng(cfg.seed), callback=callback)
    design.time_s = time.time() - t0
    if cfg.check_refinement > 1:
        design.check = {
            "refinement": cfg.check_refinement,
            "properties": design.refined_properties(cfg.check_refinement),
        }
    return design, write_results(design, cfg)
