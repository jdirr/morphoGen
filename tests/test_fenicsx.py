"""The FEniCSx solver against morphoGen's own finite element solver: the same meshes,
elements and constrained degrees of freedom give the same apparent properties."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("dolfinx_mpc")

from helpers import box_mesh
from morphogen.homogenization.apparent import (
    conductivity_tensor,
    homogenize,
    make_problem,
    stiffness_tensor,
    thermal_expansion,
)
from morphogen.homogenization.core import Loading
from morphogen.homogenization.materials import Material


def random_phases(seed, fraction=0.4):
    rng = np.random.default_rng(seed)
    return lambda c: (rng.random(len(c)) < fraction).astype(int)


MATS = {
    0: Material(young=1.0, poisson=0.3, conductivity=1.0, expansion=1e-5),
    1: Material(young=10.0, poisson=0.25, conductivity=20.0, expansion=3e-5),
}


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("bc", ["periodic", "kubc", "subc", "mixed"])
def test_elasticity_as_morphogen(bc, order):
    mesh = box_mesh((4, 5, 4), phase=random_phases(2), perturb=0.2, order=order)
    own = make_problem(mesh, MATS, "elastic", bc, tol=1e-12)
    fx = make_problem(mesh, MATS, "elastic", bc, solver="fenicsx")
    if bc == "mixed":  # defined for triaxial strains and single shears
        for load in (Loading.strain([1e-3, 0, 0, 0, 0, 0]), Loading.strain([0, 0, 0, 0, 0, 2e-3])):
            a, b = own.solve(load), fx.solve(load)
            np.testing.assert_allclose(b.stress, a.stress, rtol=1e-8, atol=1e-14)
        return
    c_own, _ = stiffness_tensor(own)
    c_fx, sols = stiffness_tensor(fx)
    np.testing.assert_allclose(c_fx, c_own, rtol=1e-8, atol=1e-12)
    assert sols[0].info.method.startswith("fenicsx")
    a_own, _ = thermal_expansion(own)
    a_fx, _ = thermal_expansion(fx)
    np.testing.assert_allclose(a_fx, a_own, rtol=1e-8, atol=1e-16)
    # the statistics of the phases follow from the same fields
    s_own = own.solve(Loading.strain([1e-3, 0, 0, 0, 5e-4, 0]))
    s_fx = fx.solve(Loading.strain([1e-3, 0, 0, 0, 5e-4, 0]))
    for p, q in zip(s_own.phases, s_fx.phases, strict=True):
        np.testing.assert_allclose(q.mean["stress"], p.mean["stress"], rtol=1e-7, atol=1e-14)


@pytest.mark.parametrize("bc", ["periodic", "kubc", "subc"])
def test_conduction_as_morphogen(bc):
    mesh = box_mesh((4, 4, 5), phase=random_phases(2), perturb=0.2, order=2)
    k_own, _ = conductivity_tensor(make_problem(mesh, MATS, "thermal", bc, tol=1e-12))
    k_fx, _ = conductivity_tensor(make_problem(mesh, MATS, "thermal", bc, solver="fenicsx"))
    np.testing.assert_allclose(k_fx, k_own, rtol=1e-8, atol=1e-12)


def test_homogenize_with_fenicsx():
    mesh = box_mesh((4, 4, 4), phase=random_phases(2), perturb=0.1)
    props = ("bulk_modulus", "shear_modulus", "conductivity")
    own = homogenize(mesh, MATS, bcs=("periodic", "kubc"), properties=props)
    fx = homogenize(mesh, MATS, bcs=("periodic", "kubc"), properties=props, solver="fenicsx")
    for key in own:
        assert fx[key].value == pytest.approx(own[key].value, rel=1e-8)


def test_oriented_crystals_as_morphogen():
    """Cubic grains turned by their orientations (the material table of the mesh)."""
    import dataclasses

    mesh = box_mesh((4, 4, 4), phase=random_phases(5, 0.5), perturb=0.15, order=2)
    centres = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    grains = (centres[:, 0] > 0.5).astype(int) + 2 * (centres[:, 2] > 0.4).astype(int)
    angles = np.array([[0.0, 0.0, 0.0], [30.0, 45.0, 10.0], [80.0, 20.0, 60.0], [10.0, 70.0, 35.0]])
    mesh = dataclasses.replace(
        mesh, phase=np.zeros_like(mesh.phase), cell=grains, orientations=angles
    )
    copper = {0: Material.cubic(168.4, 121.4, 75.4)}
    for bc in ("periodic", "kubc"):
        c_own, _ = stiffness_tensor(make_problem(mesh, copper, "elastic", bc, tol=1e-12))
        c_fx, _ = stiffness_tensor(make_problem(mesh, copper, "elastic", bc, solver="fenicsx"))
        np.testing.assert_allclose(c_fx, c_own, rtol=1e-8, atol=1e-9)
