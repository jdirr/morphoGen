"""An orientation map from end to end: the map that an acquisition would give of a section
of a polycrystal, its grains, its texture, and the stiffness of windows of it.

The map is the demonstration map of the template ``ebsd``
(src/morphogen/templates/ebsd_map.ctf), which this script writes:

    python examples/ebsd_map.py            # the analysis of the map
    python examples/ebsd_map.py --write    # writes the demonstration map again first

With a measured map, only its path changes: ``read_ebsd("my_map.ang")``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import morphogen as mg
from morphogen import texture as tx
from morphogen.homogenization import Material, homogenize
from morphogen.io import ebsd

NICKEL = ebsd.Phase("Nickel", "cubic", (3.524, 3.524, 3.524, 90.0, 90.0, 90.0))
#: The elastic constants of nickel (GPa)
C11, C12, C44 = 246.5, 147.3, 124.7


def demonstration_map() -> ebsd.EBSDMap:
    """A section of a Poisson-Voronoi polycrystal of nickel, 32 um by 24 um, measured
    every 0.5 um: a third of its grains within a few degrees of the cube orientation
    (the axes of the crystal along those of the sample), the others at random; the
    orientations scattered by 0.4 degrees, 1.5 % of the points and a quarter of those of
    the boundaries not indexed."""
    domain = mg.Domain((32.0, 24.0, 0.5), periodic=False, unit="um")
    model = mg.models.Voronoi(intensity=0.0128, crystal_orientations=True)
    micro = model.generate(domain, seed=2026)
    rng = np.random.default_rng(2026)
    n = len(micro.orientations)
    near_cube = tx.sample_texture(rng, np.zeros((1, 3)), n, spread=6.0)
    orientations = np.where(rng.random(n)[:, None] < 1.0 / 3.0, near_cube, micro.orientations)
    return ebsd.synthetic_map(micro, (64, 48, 1), orientations=orientations, phases=(NICKEL,),
                              scatter=0.4, missing=0.015, boundaries=0.25, seed=2026)  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="write the demonstration map again")
    args = parser.parse_args()
    if args.write:
        target = Path(__file__).resolve().parents[1] / "src/morphogen/templates/ebsd_map.ctf"
        print("written:", ebsd.write_ebsd(target, demonstration_map()))

    # the map, its orientations in the axes of its points
    measured = ebsd.read_ebsd(ebsd.example_map())
    ny, nx = measured.shape
    indexed = 100 * measured.indexed.mean()
    print(f"map: {nx} x {ny} points every {measured.step[0]:g} {measured.unit}, "
          f"{indexed:.1f} % indexed, phase {measured.phases[0].name}")  # fmt: skip

    # its grains: neighbouring points less than 5 degrees apart
    grains = measured.grains(threshold=5.0, min_size=5)
    print(f"grains: {grains.count}, mean diameter {grains.diameters.mean():.2f} {grains.unit}, "
          f"spread inside the grains {grains.spread.mean():.2f} degrees")  # fmt: skip

    # the texture, weighted by the areas of the grains: pole figures, in multiples of a
    # random texture, and the share of the grains within 15 degrees of the cube orientation
    for indices in ((1, 0, 0), (1, 1, 1)):
        figure = tx.pole_figure(grains.orientations, indices, weights=grains.areas)
        print(f"pole figure {{{''.join(map(str, indices))}}}: maximum {figure.maximum:.1f}")
    cube = tx.disorientation(grains.orientations, np.zeros(3)) < 15.0
    share = 100 * grains.areas[cube].sum() / grains.areas.sum()
    print(f"cube component: {share:.0f} % of the area")

    # the boundaries: their disorientations against those of crystals at random
    _, lengths, angles = tx.boundary_disorientations(grains.labels, grains.orientations)
    mean = float(np.sum(lengths * angles) / lengths.sum())
    print(f"boundaries: {len(angles)}, mean disorientation {mean:.1f} degrees "
          "(40.7 for crystals at random: Mackenzie)")  # fmt: skip

    # the stiffness of the polycrystal of the map: each grain with its orientation
    crystal = grains.microstructure()
    shape = (nx, ny, 1)
    result = homogenize(crystal.voxelize(shape), {1: Material.cubic(C11, C12, C44)},
                        grains=crystal.labels(shape).data, orientations=crystal.orientations,
                        properties=["stiffness"])["stiffness.periodic"]  # fmt: skip
    print(f"stiffness: E1 = {result['E1']:.1f}, E2 = {result['E2']:.1f}, "
          f"E3 = {result['E3']:.1f} GPa; bulk modulus {result['bulk_modulus']:.2f} "
          f"(exact: {(C11 + 2 * C12) / 3:.2f})")  # fmt: skip

    # windows of the map as the realizations of a study: the model of the configurations
    model = mg.models.EBSD(file="ebsd_map.ctf", window="random")
    domain = mg.Domain((16.0, 12.0, 0.5), periodic=False, unit="um")
    counts = [len(model.generate(domain, seed).orientations) for seed in range(4)]
    print(f"windows of 16 x 12 um: {counts} grains")

    # the map of the grains, written for the tools that read such files
    with tempfile.TemporaryDirectory() as folder:
        cleaned = ebsd.EBSDMap.from_grains(grains.labels, grains.orientations, grains.step,
                                           phases=grains.phases)  # fmt: skip
        path = ebsd.write_ebsd(Path(folder) / "grains.ang", cleaned)
        print(f"written and read back: {ebsd.read_ebsd(path).grains().count} grains")


if __name__ == "__main__":
    main()
