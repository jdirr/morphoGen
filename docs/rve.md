# Size of the representative volume

How large must a volume be to represent a random medium? morphoGen answers with the
statistical method of Kanit et al. (2003) and Dirrenberger et al. (2014): the variance of
a property over realizations of several sizes follows the generalized scaling law
D²(V) = K V^-γ, from which come the RVE sizes with confidence intervals, for a precision
and a number of realizations, the discrepancy between boundary conditions and the
isotropy. Realizations run in parallel on a workstation and as job arrays on clusters
([parallel and cluster runs](running.md#parallel-and-cluster-runs)). The
[tutorial](tutorial_rve.md) compares spheres, whose correlations are short, with fibres
on Poisson lines, whose correlations never stop.

## The method

* **Two methods**: the ensemble method (realizations at several sizes) and the window
  method (sub-volumes of one large sample, or of any scalar field; the stiffness and
  conductivity tensors of the windows of one periodic realization, their mean mutual
  energies by FFT; tilings shifted around a periodic sample, their variances pooled
  within the tilings).
* **Statistics**, per property and boundary condition: means and intervals, relative
  errors, fit of K and γ (free or imposed), RVE sizes and the number of objects (fibres,
  grains) they contain, bootstrap confidence intervals, discrepancy between conditions.
* **Every physical quantity**: the scalar properties and tensors of each physics, the
  named results of the couplings and of the harmonic properties (`piezoelectric.e33`,
  `dynamic_stiffness.shear_loss_1`, `poroelastic.biot_coefficient`), and the quantities
  of nonlinear loading paths (a stress at a time, the strength, the work of fracture).
* **Reports** in Markdown, LaTeX and JSON, with figures.
* **Results of other codes**: the fit of external tables, e.g. apparent moduli computed
  with another FE/FFT code.

## A study

The template `rve_poisson_fibers_fe` is the study of Dirrenberger et al. (2014) on fibres
on Poisson lines, by finite elements:

```bash
morphogen template rve_poisson_fibers_fe -o study.yaml   # or rve_poisson_fibers (morphology)
morphogen run study.yaml
```

The run writes:

* `results.csv`: one row per realization (measures and apparent properties);
* `homogenization.csv`: one row per realization, property and boundary condition;
* `rve_report.md`, `rve_report.tex` and `rve_report.json`: for each property and
  condition, the statistics per size, K, γ (free and imposed), R², the RVE sizes for
  each (ε_rel, n) with confidence intervals, the discrepancy between conditions, the
  isotropy criteria and the bounds;
* when the homogenization computes the `stiffness`, for each condition and size the mean
  stiffness C̄_ij and its compliance S_ij = C̄⁻¹ (the inverse of the mean C, not the mean
  of the inverses), each component ± 2 standard deviations, and the RVE size of each
  component (Voigt order 11 22 33 23 31 12, engineering shears): in the report, in
  `rve_stiffness.csv` (one row per condition and size: `C11` ... `C66`, `S11` ... `S66`,
  `C11_std` ...) and in `rve_stiffness_<bc>.png`;
* figures per property.

To analyse apparent properties computed elsewhere (CSV with a `volume` column, one
row per realization; `--group-by` splits the rows by boundary condition):

```bash
morphogen fit fe_results.csv --property k_app mu_app --group-by bc --gamma 0.6667 \
    --eps 0.01 0.05 --n 1 10 -o report/
```

Nonlinear responses are analysed in the same way: `rve.properties` may name quantities
of the curves of the loading paths, `<path>.<quantity>` at the end of the path or
`<path>.<quantity>@<time>` at the end of one of its steps (`tension.S11@3`: the stress
of the path `tension` at the time 3; also `TS11` in finite strains, `p`, or the mean of
a phase, `S11_1`), followed by a boundary condition to keep only that one
(`tension.S11@3.periodic`). The template `rve_crystal_plasticity_fft`, after Yang et al.
(2019), compares the RVE sizes of the elastic moduli of a polycrystal with those of its
stress at 0.1, 0.3 and 1 % of a tension (crystal plasticity, nonlinear FFT solver):

```bash
morphogen template rve_crystal_plasticity_fft -o study.yaml && morphogen run study.yaml
morphogen fit out/rve_crystal_plasticity_fft --property tension.S11@3   # again, from the files
```

The template `rve_multiphysics` gives the RVE sizes of several physics at once.
