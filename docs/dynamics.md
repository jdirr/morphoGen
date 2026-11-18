# Explicit dynamics with OpenRadioss

`morphogen.dynamics` simulates the crushing, tension, shear and impact of a morphology
at the rates of crashes, shocks and drops with
[OpenRadioss](https://github.com/OpenRadioss/OpenRadioss), the open-source explicit
finite element solver (AGPL-3.0; run as a separate program, installed from its
releases by `morphogen radioss install`, see [Installation](installation.md#optional-components)): its voxels as stabilized 8-node bricks or a
conforming mesh of its level set as tetrahedra, its phases elastic, von Mises
elastoplastic (the linear, Voce and Norton hardenings of morphoGen tabulated in LAW36)
or Johnson–Cook (LAW2), with failure by element deletion, the self contact of its
folding struts and closing cells (TYPE24), its faces held and driven after a smooth
start, or the whole thrown against a rigid wall; its nominal stress–strain curves at
both faces (their gap telling a quasi-static test from a dynamic one), energies, peak
and plateau stresses, densification strain, absorbed energy per volume and mass, and
its fields over time (`morphogen radioss export|run|read`, the tab Explicit
dynamics). The inputs are the Radioss Block Format of the reference guide, written
with their manifest so that a run made elsewhere (a cluster) is read back.

```python
import morphogen as mg
from morphogen import dynamics

lattice = mg.model_from_config(
    {"type": "tpms", "surface": "gyroid", "volume_fraction": 0.2, "cell_size": 5.0}
)  # an aluminium gyroid, its lengths in mm
micro = lattice.generate(mg.Domain.cube(10.0), seed=1)
materials = {
    0: "void",
    1: {
        "young": 70e3,  # MPa
        "poisson": 0.33,
        "density": 2.7e-9,  # Mg/mm³
        "plasticity": {"yield": 150.0, "hardening": 600.0},
        "failure_strain": 0.4,
    },
}
test = dynamics.explicit_test(
    micro,
    materials,
    {"kind": "compression", "strain": 0.6, "strain_rate": 500.0},
    "gyroid_crush",
    resolution=32,
)  # units: Mg, mm, s by default
values = dynamics.summary(test.result)  # peak and plateau stresses (MPa), densification...
```

The loadings are `compression`, `tension` and `shear` (the lower face held, the upper
one driven at the nominal strain rate after a smooth start, laterally `fixed` or
`free`) and `impact` (the morphology thrown at a `velocity` against a rigid wall), with
the self contact of the outer surface by default. The values are in the consistent unit
system of `units` (`dynamics.UnitSystem`: Mg, mm, s by default, i.e. MPa, Mg/mm³, mm/s, N,
mJ; `length_scale` converts the lengths of the morphology). The time step of an explicit
run is that of the smallest element (`dynamics.critical_time_step`; the manifest of the
inputs gives the number of cycles before the run): a coarser mesh, a higher rate or a
minimum time step (`time_step`, mass scaling) shortens it. The command line does the
same from a geometry or an image, and reads back a run made elsewhere:

```bash
morphogen radioss install                       # the latest release, from GitHub
morphogen radioss export gyroid.json --materials aluminium.yaml --voxels 32 \
    --loading compression --strain 0.6 --rate 500 -o crush
morphogen radioss run crush --threads 8         # Starter, Engine, conversions, summary
morphogen radioss read crush --frames-vtu       # curves.csv, energies.csv, frames/*.vtu
```

The inputs follow the fields of the Radioss Reference Guide for the input version 2022
(`/BEGIN`, `/NODE`, `/BRICK`, `/TETRA4`, `/TETRA10`, `/PART`, `/PROP/SOLID`, `/MAT/LAW1`,
`/MAT/LAW2`, `/MAT/LAW36`, `/FUNCT`, `/GRNOD`, `/BCS`, `/IMPVEL`, `/INIVEL`, `/RWALL`,
`/SURF/PART/EXT`, `/INTER/TYPE24`, `/TH/NODE`, `/TH/PART`, `/TH/RWALL`; `/RUN`, `/TFILE`,
`/ANIM`, `/DT/NODA/CST`); the forces of the faces come from the reactions of their nodes
in the time history (`th_to_csv`) or in the animations (`anim_to_vtk`), the forces on a
rigid wall from the change of the momentum.
