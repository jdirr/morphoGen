"""Diffusion and swelling of tetrahedral meshes with FEniCSx: Crank's slab, uniform free
swelling, the bending of a bilayer strip (Timoshenko) and its sequence in time, and its
curling into an arc in finite strains."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("dolfinx_mpc")

from helpers import box_mesh
from morphogen.coupling import slab_uptake, timoshenko_curvature
from morphogen.coupling.fenicsx import arc_curvature, strip_curvature, swelling


def test_slab_uptake_against_crank():
    thickness, d = 1.0, 1.0
    mesh = box_mesh((2, 2, 40), upper=(0.1, 0.1, thickness), periodic=False)
    phases = {0: {"diffusivity": d, "solubility": 0.3, "young": 1.0, "poisson": 0.3}}
    times = np.array([0.005, 0.02, 0.1, 0.4])
    res = swelling(mesh, phases, times, exposed=("z-", "z+"), steps=60)
    np.testing.assert_allclose(res.relative_uptake, slab_uptake(times, thickness, d), rtol=0.02)


def test_uniform_free_swelling():
    mesh = box_mesh((3, 3, 3), upper=(1.0, 1.2, 0.8), periodic=False, perturb=0.2, seed=1)
    beta, s = 0.01, 0.5
    phases = {
        0: {"diffusivity": 1.0, "solubility": s, "swelling": beta, "young": 10.0, "poisson": 0.3}
    }
    res = swelling(mesh, phases, [50.0], exposed=("x-", "x+", "y-", "y+", "z-", "z+"), steps=10)
    assert res.relative_uptake[-1] == pytest.approx(1.0, abs=1e-8)
    # a uniform swelling strain beta S, without stresses: u = beta S (x - x_A) + rigid motion
    u = res.displacement[-1]
    x = mesh.nodes - mesh.nodes[np.argmin(np.linalg.norm(mesh.nodes - mesh.lower, axis=1))]
    np.testing.assert_allclose(u, beta * s * x, atol=1e-10)


@pytest.mark.parametrize("order", [2])
def test_bilayer_bending(order):
    """A strip of two layers, the upper one swelling: Timoshenko's curvature at
    saturation (equal Poisson's ratios: the plate bends as the beam), reached in time."""
    h1, h2, e1, e2, beta = 0.04, 0.06, 3.0, 1.0, 0.01
    length, width = 1.0, 0.1
    mesh = box_mesh(
        (40, 4, 6), upper=(length, width, h1 + h2), periodic=False, order=order,
        phase=lambda c: (c[:, 2] > h1).astype(int),
    )  # fmt: skip
    phases = {
        0: {"diffusivity": 1.0, "solubility": 1.0, "swelling": 0.0, "young": e1, "poisson": 0.3},
        1: {"diffusivity": 1.0, "solubility": 1.0, "swelling": beta, "young": e2, "poisson": 0.3},
    }
    times = [1e-4, 1e-3, 1e-2, 1.0]
    res = swelling(mesh, phases, times, exposed=("z+",), steps=20)
    kappa = [strip_curvature(mesh, u) for u in res.displacement]
    assert np.all(np.diff(kappa) > 0)  # the upper layer swells first: the strip bends down
    expected = timoshenko_curvature(beta, h1, h2, e1, e2)
    assert kappa[-1] == pytest.approx(expected, rel=0.01)  # 0.3 % (quadratic tetrahedra)


def test_large_bending_in_finite_strains():
    """A long bilayer strip curls into an arc: in finite strains, the curvature of the
    arc fitted to its mid-plane is Timoshenko's (small strains, large rotations) and its
    end comes closer by the shortening of the chord; in small strains it does not."""
    h1, h2, e1, e2, beta, length = 0.04, 0.06, 3.0, 1.0, 0.03, 2.0
    mesh = box_mesh(
        (32, 2, 4), upper=(length, 0.1, h1 + h2), periodic=False, order=2,
        phase=lambda c: (c[:, 2] > h1).astype(int),
    )  # fmt: skip
    phases = {
        0: {"diffusivity": 1.0, "solubility": 1.0, "swelling": 0.0, "young": e1, "poisson": 0.3},
        1: {"diffusivity": 1.0, "solubility": 1.0, "swelling": beta, "young": e2, "poisson": 0.3},
    }
    kappa = timoshenko_curvature(beta, h1, h2, e1, e2)
    line = (np.abs(mesh.nodes[:, 2] - 0.5 * (h1 + h2)) < 1e-9) & (np.abs(mesh.nodes[:, 1]) < 1e-9)
    order = np.argsort(mesh.nodes[line, 0])

    def chord_and_arc(u):
        """Chord and length of the deformed axis of the mid-plane (in the x-z plane)."""
        pts = (mesh.nodes[line] + u[line])[order][:, [0, 2]]
        return float(np.linalg.norm(pts[-1] - pts[0])), float(
            np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        )

    finite = swelling(mesh, phases, [1.0], exposed=("z+",), steps=10, kinematics="finite")
    u = finite.displacement[-1]
    k = arc_curvature(mesh, u)
    assert k == pytest.approx(kappa, rel=0.07)
    chord, arc = chord_and_arc(u)
    assert chord == pytest.approx(2.0 / k * np.sin(0.5 * k * arc), rel=0.005)  # a circular arc
    assert chord / arc < 0.98  # curled: the chord shortens

    def span(u):
        x = mesh.nodes[line, 0] + u[line, 0]
        return float(x.max() - x.min())

    assert span(u) < 0.99 * length  # the ends come closer (the mid-plane stretches 1.5 %)
    small = swelling(mesh, phases, [1.0], exposed=("z+",), steps=10)
    assert span(small.displacement[-1]) > length  # small strains: they do not
