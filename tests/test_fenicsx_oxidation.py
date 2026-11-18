"""Stresses of an oxidized epoxy plate with FEniCSx against the free laminated plate."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("dolfinx_mpc")

from helpers import box_mesh
from morphogen.coupling import ClosedLoopOxidation, PropertyLaw
from morphogen.coupling.fenicsx import oxidation_stresses


def test_oxidized_plate_against_laminate_theory():
    """A plate 3 mm thick oxidized on both faces at 110 °C in air for 240 h: the amide
    profile sets the modulus and the shrinkage of each element; far from the edges the
    plate is a free symmetric laminate, sigma(z) = E(z)/(1 - nu) (eps0 - eps*(z)) with
    eps0 = <E eps*> / <E>: tension in the oxidized skin, compression in the core."""
    thickness, side, nu = 3.0e-3, 8.0e-3, 0.37
    mesh = box_mesh((6, 6, 60), upper=(side, side, thickness), periodic=False)
    law = PropertyLaw(
        tracer=(0.0, 2.0), values={"young": (2.4e9, 4.0e9), "expansion": (0.0, -0.02)}
    )
    model = ClosedLoopOxidation()
    res = oxidation_stresses(mesh, law, [240 * 3600.0], oxidation=model, poisson=nu)
    tracer, stress = res.tracer[0], res.stress[0]
    centroids = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    column = (np.abs(centroids[:, 0] - side / 2) < 0.7e-3) & (
        np.abs(centroids[:, 1] - side / 2) < 0.7e-3
    )
    # the laminate: the same profile on a fine grid of the thickness
    z = np.linspace(0.0, thickness, 6001)
    depth = np.minimum(z, thickness - z)
    elem_depth = np.minimum(centroids[:, 2], thickness - centroids[:, 2])
    order = np.argsort(elem_depth)
    q = np.interp(depth, elem_depth[order], tracer[order])
    props = law(q)
    e, eig = props["young"], props["expansion"]
    eps0 = np.trapezoid(e * eig, z) / np.trapezoid(e, z)
    elem_props = law(tracer[column])
    expected = elem_props["young"] / (1.0 - nu) * (eps0 - elem_props["expansion"])
    measured = 0.5 * (stress[column, 0] + stress[column, 1])
    skin = elem_depth[column] < 0.05e-3
    assert np.all(measured[skin] > 0) and np.all(measured[elem_depth[column] > 1.0e-3] < 0)
    np.testing.assert_allclose(measured, expected, rtol=0.05, atol=0.02 * np.abs(expected).max())
    assert tracer.max() > 0.3 and tracer.min() < 0.05 * tracer.max()
