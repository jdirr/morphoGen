# Z-mat in morphoGen

morphoGen reads the material files of Z-mat, the material library of Z-set (ARMINES,
ONERA, Transvalor; *Materials manual*, version 9.1), and translates their behaviours
(`morphogen.homogenization.zmat`):

* a `gen_evp` (or `reduced_plastic`) behaviour of one potential with the von Mises or
  Hill criterion (its coefficients `hill1` ... `hill6`), a Norton flow or time-independent
  plasticity, an isotropic hardening `constant`, `linear`, `nonlinear`, `linear_nonlinear`
  or `nonlinear_sum` and kinematic hardening `linear` or `nonlinear`, on an isotropic
  elasticity, runs on morphoGen's own kernels (finite elements, FFT, GPU), as the
  plasticity of its materials;
* the other `gen_evp` behaviours that MFront's brick `StandardElastoViscoPlasticity`
  expresses (Hill's criterion with them, the flows `hyperbolic`, `double_norton`, `norton_exp`,
  `strain_hardening`, the isotropic hardening `power_law`, softening, `nonlinear_evrad`,
  several potentials) become an MFront behaviour, compiled by MFront and integrated by
  MGIS along the loading paths of the volume elements (Linux, or WSL on Windows);
* a `porous_plastic` behaviour of one porous potential (`*porous_criterion gurson`,
  `rousselier`, `modified_rousselier` or `elliptic`, a Norton or rate-independent flow, the
  isotropic hardenings of gen_evp, the nucleation `constant`, `gaussian` or `exponential`,
  and with `gurson` and `elliptic` the kinematic hardening of gen_evp: `*kinematic nonlinear`
  with `C` and `D` (Armstrong-Frederick), `*kinematic linear` with `C` (Prager), named `X1`,
  `X2`... when several, four at most, the potential that of `σ - X` as in Besson and
  Guillemer-Neel (2003), the backstresses with their hydrostatic part; Z-set itself was not
  at hand to check its own `porous_plastic` files with backstresses, whose syntax here is
  that of gen_evp; Rousselier's criterion takes none) on an isotropic elasticity runs on
  morphoGen's porous kernels (finite elements, FFT, on the CPU and the GPU), in small or
  finite strains (a modifier such as `lagrange_rotate_no_J` gives
  way to morphoGen's logarithmic strains, which the solver chooses): coefficients as
  numbers, `fs = f`, or tables of the porosity (`C f` followed by rows `value f`: the
  coalescence of Tvergaard and Needleman as the effective porosity `fs` of GTN, Abouaf's
  `C` and `F`), the initial porosity from `*initialize_variable f` (or the option
  `porosity` of the material: `{zmat: {file: powder.mat, porosity: 0.4}}`);
* `linear_elastic` behaviours: any elasticity (isotropic, cubic, orthotropic,
  transverse, anisotropic).

In a configuration, a material is `{zmat: alloy.mat}` (a path relative to the
configuration), or `{zmat: {file: alloy.mat, behaviour: 0, native: false}}` to have
MFront integrate even the behaviours of the native kernels (for the exports to other
codes); `morphogen zmat alloy.mat` shows the translation, `-o laws/` writes the MFront
source, `morphogen zmat --coverage` prints the table below.

Checks: the native kernels and MFront agree on the same files (Chaboche's viscoplasticity
with two backstresses, Voce and Armstrong-Frederick plasticity: relative gaps below
1e-10; Hill's criterion of distinct coefficients: below 1e-8); Hill's criterion with all
its coefficients 1 is von Mises; its yield stresses in
tension along x and y and in the three shears (distinct coefficients) are the analytical
ones, which fixes the order of the shear components between Z-set (11, 22, 33, 12, 23,
31), TFEL (its Hill tensor: `F (s11 - s22)^2 + G (s22 - s33)^2 + H (s33 - s11)^2 + 2 L
s12^2 + 2 M s13^2 + 2 N s23^2`) and morphoGen (11, 22, 33, 23, 31, 12).

## Material files written

`write_zmat` writes the elastic, von Mises, Hill and porous materials of morphoGen as material
files of Z-mat, which read back as the same materials (the numbers in full, or to the
significant digits asked for): a `linear_elastic` behaviour (isotropic, cubic or
anisotropic elasticity, each component of the stiffness named, `y2323`, `y3131`,
`y1212`...), or a `gen_evp` behaviour of one von Mises potential on an isotropic
elasticity: `*flow norton` (`n`, `K`) or `plasticity`, one isotropic hardening
(`constant`, `linear`, `nonlinear`, `linear_nonlinear`, or `nonlinear_sum` for several
saturating terms; a linear hardening with several terms has no brick and is refused), one
brick per kinematic term (`linear` when `D = 0`, `nonlinear` otherwise, named `X1`,
`X2`... when several), and `**thermal_strain isotropic`. A behaviour identified on the
response of a morphology (`homogenization.identification`, as Zopt does) thus goes back
to Z-set; in the interface, the tool "Z-set file of a behaviour" of the tab Solvers &
formats writes the behaviours of the workspace. Porous plasticity is written as a
`porous_plastic` behaviour of one porous potential: `*porous_criterion gurson` (`q1`,
`q2`; `q3 = q1²`, as in Z-mat, otherwise refused) with the table `fs f` of its effective
porosity (Tvergaard and Needleman's coalescence), `rousselier` (or `modified_rousselier`
with `q2`) or `elliptic` (`C`, `F`, numbers or tables of `f`), the flow, the kinematic
terms (the bricks of gen_evp) and `*isotropic_hardening` of the matrix, `*strain_nucleation` (`fn`, `en`, `sn`; `A`; `B`,
`e0`; the threshold `yield`), and the initial porosity in `*initialize_variable f` before
the behaviour (as in the examples of the manual). Crystal plasticity and MFront
behaviours are not written (their MFront sources are).

## Coverage of Z-mat's library

Status: *native* (morphoGen's kernels), *mfront* (translated into an MFront behaviour),
*elsewhere* (morphoGen has it in another form), *user* (an MFront law written by the
user), *no*.

| category | Z-mat | morphoGen | how |
|---|---|---|---|
| model | linear_elastic | native | isotropic, cubic, orthotropic, transverse, anisotropic |
| model | gen_evp | native | one potential of von Mises or Hill; the others through MFront |
| model | reduced_plastic | native | as gen_evp |
| model | finite_strain crystal (Méric-Cailletaud) | elsewhere | crystal_plasticity, small and finite strains (CrystalPlasticity) |
| model | porous_plastic (one porous potential) | native | PorousPlasticity: small and finite strains, the CPU and the GPU |
| porous_criterion | gurson, rousselier, modified_rousselier, elliptic | native | GTN (q3 = q1², fs = f or a table: coalescence), Rousselier, Abouaf's elliptic (C, F tables of f) |
| porous_criterion | cam_clay, fkm, zhang_niemi, elliptic_aniso, elliptic_fiber | no |  |
| kinematic (porous_plastic) | linear, nonlinear | native | gurson and elliptic: the potential of σ - X (Besson & Guillemer-Neel 2003), four at most; not rousselier |
| strain_nucleation | constant, gaussian, exponential (yield) | native | Chu-Needleman; not crack_like, porosity_yield |
| porous_plastic | several porous potentials, Hf, Hp, *shear_anisotropy hill, broken_behavior, adiabatic_heating | no |  |
| model | damage elasticity, aniso_damage | user | an MFront law |
| model | hyper_elastic (Mooney, Ogden, Arruda-Boyce, neo-Hookean) | user | an MFront law; morphoGen's finite strains: Hencky's elasticity |
| model | linear_viscoelastic, viscoelastic_spectral | user | an MFront law |
| model | needleman, crisfield, chaboche, generalized debonding | elsewhere | cohesive zone elements (morphogen.fracture.cohesive) |
| model | diffusion, coefficient_diffusion | elsewhere | coupling.diffusion (transient) |
| elasticity | isotropic, cubic, orthotropic, transverse, anisotropic | native | crystals turned by the orientations of their grains |
| elasticity | coefficients depending on the temperature | no |  |
| thermal_strain | isotropic | native | expansion (crystals: anisotropic) |
| criterion | mises | native |  |
| criterion | hill | native | hill1 ... hill6 (components 11, 22, 33, 12, 23, 31); Hill (F, G, H, L, M, N) for MFront with the models that need it |
| criterion | tresca | user | MFront's Hosford criterion with a large exponent |
| criterion | linear_drucker_prager, cast_iron, karafillis_boyce, nouailhas, 2M1C | user | an MFront law |
| flow | plasticity | native | rate-independent |
| flow | norton | native | dp/dt = <f/K>^n (n >= 1; MFront otherwise) |
| flow | hyperbolic, double_norton, norton_exp, strain_hardening | mfront | user-defined viscoplastic flows of the brick |
| flow | sellars_tegart, gsell, modified_visco, function, flow_sum | user | UserDefinedViscoplasticity of MFront's brick |
| isotropic | constant, linear, nonlinear, linear_nonlinear, nonlinear_sum | native | three saturating terms at most, without softening (MFront otherwise) |
| isotropic | power_law | mfront | UserDefined isotropic hardening |
| isotropic | by_point, iso_table, linear_pp, nonlinear recovery | user | MFront's Data hardening (tables) |
| kinematic | linear, nonlinear | native | Prager, Armstrong-Frederick (four at most) |
| kinematic | nonlinear_evrad | mfront | Burlet-Cailletaud |
| kinematic | ziegler, aniso_nonlinear, nonlinear with recovery | user | an MFront law |
| potential | several gen_evp potentials | mfront | one inelastic flow per potential |
| potential | interactions between potentials | no |  |
| modifier | lagrange_rotate, lagrange_polar (finite strains) | elsewhere | finite strains: logarithmic strains (Hencky) of the loading paths |
| modifier | plane_stress | no | morphoGen's volume elements are 3D |
| simulation | ****simulate (material point) | elsewhere | loading paths of a single material point (PathResult) and of volume elements |
| optimization | ****optimize (identification) | elsewhere | homogenization.identification: von Mises parameters on references (material points), the behaviour found written back by write_zmat |
