"""The effective stiffness of random periodic frameworks as frames of beams
(:mod:`morphogen.homogenization.networks`): the sparse periodic problem against the dense
one of the lattices of struts and the closed forms of the honeycomb and triangular frames,
of Euler-Bernoulli and of Timoshenko beams, frames built from frameworks,
stretching-dominated Delaunay networks (an independent truss, their affinity and Poisson's
ratio), bending-dominated open foams (E ~ phi^2), the softening of networks of
non-identical beams (Ban et al. 2016: the exact derivatives against finite differences,
for Timoshenko beams too, the second-order theory against replicas), the models that keep
their frame, the frames of their realizations and the command morphogen frame-stiffness,
and the errors."""

import dataclasses
import json

import numpy as np
import pytest

import morphogen as mg
from morphogen.design.frames import Framework
from morphogen.homogenization.frames import FLEXIBLE_LIMIT, Frame, SpaceFrame
from morphogen.homogenization.networks import (
    fibre_scales,
    frame_stiffness,
    network_frame,
    network_stiffness,
    scaled_frame,
    softening,
    softening_coefficients,
)
from morphogen.models.foam import DelaunayNetwork, periodic_edges, periodic_framework

CUBE = mg.Domain.cube(1.0)


def _delaunay(n, seed, dimension=3):
    model = DelaunayNetwork(n_points=n, radius=0.01, dimension=dimension)
    return model.framework(CUBE, np.random.default_rng(seed))


def test_lattices_against_the_dense_periodic_problem():
    """The sparse path gives the stiffness of the dense one (SpaceFrame.macroscopic_
    stiffness) to round-off, for the four lattices of struts and from their frameworks."""
    for kind in ("cubic", "bcc", "octet", "kelvin", "diamond", "pyrochlore"):
        frame = SpaceFrame.lattice_of(kind, elements=2, radius=0.03)
        dense = frame.macroscopic_stiffness()
        result = frame.effective_stiffness()
        assert np.abs(result.stiffness - dense).max() < 1e-12 * np.abs(dense).max()
        assert result.density == pytest.approx(frame.relative_density(), rel=1e-12)
        parts = result.stretching + result.bending + result.torsion
        np.testing.assert_allclose(parts, result.stiffness, atol=1e-14 * np.abs(dense).max())
    for cell in ("cubic", "bcc", "octet", "kelvin", "diamond", "pyrochlore"):
        frame = SpaceFrame.from_framework(Framework.lattice(cell, 2), 0.03, poisson=0.3)
        dense = SpaceFrame.lattice_of(cell, (2, 2, 2), elements=1, radius=0.03)
        dense = dense.macroscopic_stiffness()
        assert np.abs(frame_stiffness(frame).stiffness - dense).max() < 1e-12 * np.abs(dense).max()
    # the cubic lattice: stretching alone, E = rho / 3 along its struts
    s = SpaceFrame.lattice_of("cubic", radius=0.02).effective_stiffness()
    assert s.split()["stretching"] == pytest.approx(1.0) and s.nonaffinity() < 1e-24
    assert s.young() == pytest.approx(s.density / 3.0, rel=1e-12)
    # the pyrochlore lattice: its struts along the diagonals of the faces, as those of the
    # octet truss with half as many: the nodes move with the strain (straight lines of
    # struts), and the stretching of the struts gives C11 = rho / 6, C12 = C44 = rho / 12
    s = SpaceFrame.lattice_of("pyrochlore", elements=1, radius=0.02).effective_stiffness()
    stretching = np.zeros((6, 6))
    stretching[:3, :3] = 1.0 / 12.0 + np.eye(3) / 12.0
    stretching[3:, 3:] = np.eye(3) / 12.0
    assert s.nonaffinity() < 1e-24
    np.testing.assert_allclose(s.stretching / s.density, stretching, atol=1e-12)
    octet = SpaceFrame.lattice_of("octet", radius=0.02).effective_stiffness()
    np.testing.assert_allclose(octet.stretching / octet.density, stretching, atol=1e-12)
    assert s.split()["stretching"] > 0.99  # the joints bend the struts a little
    # the diamond lattice: a hydrostatic strain stretches its struts alike, K = rho / 9
    # whatever their bending, which carries the shears
    s = SpaceFrame.lattice_of("diamond", radius=0.02).effective_stiffness()
    bulk = (s.stiffness[:3, :3].sum()) / 9.0
    assert bulk == pytest.approx(s.density / 9.0, rel=1e-12)
    assert s.split()["bending"] > 0.99


@pytest.mark.parametrize("t", [0.02, 0.1, 0.2])
def test_plane_lattices_against_their_closed_forms(t):
    """Euler-Bernoulli walls with rigid joints, exactly: the regular honeycomb, E = 4/sqrt(3)
    E_s t^3 / (1 + 3 t^2), nu = (1 - t^2) / (1 + 3 t^2) (Masters & Evans 1996), a fraction
    3 t^2 / (1 + 3 t^2) of stretching energy; the triangular lattice deformed affinely, its
    chords turned against the joints: E = 2/sqrt(3) E_s t 3 (1 + t^2) / (3 + t^2), nu = (1 -
    t^2) / (3 + t^2) (walls of unit length and depth)."""
    s = Frame.lattice_of("honeycomb", (1, 1), elements=2, thickness=t).effective_stiffness()
    e = s.engineering()
    young = 4.0 / np.sqrt(3.0) * t**3 / (1.0 + 3.0 * t**2)
    assert e["E1"] == pytest.approx(young, rel=1e-12) and e["E2"] == pytest.approx(young, rel=1e-12)
    assert e["nu12"] == pytest.approx((1.0 - t**2) / (1.0 + 3.0 * t**2), rel=1e-12)
    assert s.split()["stretching"] == pytest.approx(3.0 * t**2 / (1.0 + 3.0 * t**2), rel=1e-10)
    assert s.density == pytest.approx(2.0 * t / np.sqrt(3.0), rel=1e-12)
    s = Frame.lattice_of("triangular", (2, 2), elements=1, thickness=t).effective_stiffness()
    e = s.engineering()
    young = 2.0 / np.sqrt(3.0) * t * 3.0 * (1.0 + t**2) / (3.0 + t**2)
    assert e["E1"] == pytest.approx(young, rel=1e-12) and e["G12"] == pytest.approx(
        young / (2.0 * (1.0 + e["nu12"])), rel=1e-12
    )
    assert e["nu12"] == pytest.approx((1.0 - t**2) / (3.0 + t**2), rel=1e-12)
    assert s.nonaffinity() < 1e-20  # every node is a centre of symmetry


@pytest.mark.parametrize("t", [0.02, 0.1, 0.2])
def test_thick_lattices_of_timoshenko_beams(t):
    """Timoshenko walls (shear coefficient kappa): the flexure of the honeycomb walls gains
    the shear compliance, its share Phi = 12 EI / (kappa G A l^2) = E t^2 / (kappa G) of the
    bending one, E = 4/sqrt(3) E_s t^3 / (1 + Phi + 3 t^2), nu = (1 + Phi - t^2) / (1 + Phi +
    3 t^2) (Masters & Evans with the shear deflection of the walls, as Gibson & Ashby's
    correction); the chords of the triangular lattice turn against the joints with the
    stiffness 12 EI / l (1 + Phi): t^2 becomes t^2 / (1 + Phi) in its closed form; the
    energy split between stretching, bending and shear; the struts in space against the
    dense problem, Euler-Bernoulli unchanged when the shear is off."""
    for kappa, value in ((5.0 / 6.0, 5.0 / 6.0), ("cowper", 13.0 / 15.3)):
        phi = 2.6 * t**2 / value  # E / G = 2 (1 + nu) = 2.6
        frame = Frame.lattice_of("honeycomb", (1, 1), elements=2, thickness=t,
                                 shear_coefficient=kappa)  # fmt: skip
        assert frame.shear == pytest.approx(value * t / 2.6, rel=1e-14)
        s = frame.effective_stiffness()
        e = s.engineering()
        flexure = 1.0 + phi + 3.0 * t**2
        young = 4.0 / np.sqrt(3.0) * t**3 / flexure
        assert e["E1"] == pytest.approx(young, rel=1e-12)
        assert e["E2"] == pytest.approx(young, rel=1e-12)
        assert e["nu12"] == pytest.approx((1.0 + phi - t**2) / flexure, rel=1e-12)
        split = s.split()
        assert split["stretching"] == pytest.approx(3.0 * t**2 / flexure, rel=1e-10)
        assert split["shear"] == pytest.approx(phi / flexure, rel=1e-10)
        frame = Frame.lattice_of("triangular", (2, 2), elements=1, thickness=t,
                                 shear_coefficient=kappa)  # fmt: skip
        e = frame.effective_stiffness().engineering()
        tau = t**2 / (1.0 + phi)
        young = 2.0 / np.sqrt(3.0) * t * 3.0 * (1.0 + tau) / (3.0 + tau)
        assert e["E1"] == pytest.approx(young, rel=1e-12)
        assert e["nu12"] == pytest.approx((1.0 - tau) / (3.0 + tau), rel=1e-12)
    if t != 0.1:
        return
    for kind in ("bcc", "kelvin"):
        frame = SpaceFrame.lattice_of(kind, elements=2, radius=t / 2, shear_coefficient="cowper")
        dense = frame.macroscopic_stiffness()
        result = frame.effective_stiffness()
        assert np.abs(result.stiffness - dense).max() < 1e-12 * np.abs(dense).max()
        parts = result.stretching + result.bending + result.torsion + result.shear
        np.testing.assert_allclose(parts, result.stiffness, atol=1e-14 * np.abs(dense).max())
        plain = SpaceFrame.lattice_of(kind, elements=2, radius=t / 2)
        rigid = SpaceFrame.lattice_of(kind, elements=2, radius=t / 2)
        rigid.shear = np.full(len(rigid.beams), np.inf)
        assert np.array_equal(rigid.macroscopic_stiffness(), plain.macroscopic_stiffness())
        assert result.young() < plain.effective_stiffness().young()  # softer in shear
    fw = _delaunay(60, 4)
    frame = SpaceFrame.from_framework(fw, 0.02, young=2.0, poisson=0.25, shear_coefficient=0.9)
    assert frame.shear == pytest.approx(0.9 * 2.0 / 2.5 * np.pi * 4e-4)
    cowper = SpaceFrame.from_framework(fw, 0.02, shear_coefficient="cowper").shear
    assert cowper == pytest.approx(6 * 1.3 / 8.8 / 2.6 * np.pi * 4e-4)
    with pytest.raises(ValueError, match="shear_coefficient"):
        SpaceFrame.from_framework(fw, 0.02, shear_coefficient="timoshenko")
    with pytest.raises(ValueError, match="shear_coefficient"):
        Frame.from_framework(_delaunay(30, 2, dimension=2), 0.01, shear_coefficient=-1.0)


def _split_with_stiff_ends(frame, stiff=1e7):
    """Each beam of a frame with rigid ends split in three, its end segments ``stiff``
    times stiffer: the rigid ends computed independently."""
    v = frame.vectors()
    lengths = np.linalg.norm(v, axis=1)
    e = v / lengths[:, None]
    n0, m = len(frame.nodes), len(frame.beams)
    x0 = frame.nodes[frame.beams[:, 0]]
    p = x0 + frame.rigid[:, :1] * e
    q = x0 + (lengths - frame.rigid[:, 1])[:, None] * e
    ip, iq = n0 + np.arange(m), n0 + m + np.arange(m)
    beams = np.vstack([np.column_stack([frame.beams[:, 0], ip]), np.column_stack([ip, iq]),
                       np.column_stack([iq, frame.beams[:, 1]])])  # fmt: skip
    zero = np.zeros_like(frame.shifts)

    def three(x, s=stiff):
        return None if x is None else np.concatenate([x * s, x, x * s])

    fields = {"nodes": np.vstack([frame.nodes, p, q]), "beams": beams,
              "shifts": np.vstack([zero, zero, frame.shifts]), "axial": three(frame.axial),
              "bending": three(frame.bending), "area": three(frame.area, 1.0),
              "shear": three(frame.shear), "rigid": None}  # fmt: skip
    if isinstance(frame, SpaceFrame):
        fields.update(torsion=three(frame.torsion), polar=three(frame.polar, 1.0))
    return dataclasses.replace(frame, **fields)


@pytest.mark.parametrize("t", [0.02, 0.1, 0.2])
def test_rigid_joints(t):
    """Beams with rigid ends of length a at their joints, flexible over l' = l - 2a: the
    triangular lattice, deformed affinely, takes EA/l' and 12EI/l'^3 (1 + Phi'), E = 2/sqrt(3)
    E_s t (l/l') 3 (1 + tau) / (3 + tau) with tau = t^2 (l/l')^2 / (1 + Phi'), to round-off;
    random frames against their beams split in three, the end segments 1e7 times stiffer
    (space and plane, Euler-Bernoulli and Timoshenko); the cubic lattice stiffer by l/l',
    affine still; the lengths from rigid_joints radii, the shortest beams kept flexible
    over FLEXIBLE_LIMIT, the errors and the dense paths refused."""
    for a in (0.05, 0.2):
        for kappa in (None, 5.0 / 6.0):
            frame = Frame.lattice_of("triangular", (2, 2), elements=1, thickness=t,
                                     shear_coefficient=kappa)  # fmt: skip
            frame.rigid = np.full((len(frame.beams), 2), a)
            e = frame_stiffness(frame).engineering()
            flexible = 1.0 - 2.0 * a
            tau = t**2 / flexible**2
            if kappa is not None:
                tau /= 1.0 + 2.6 * t**2 / (kappa * flexible**2)
            young = 2.0 / np.sqrt(3.0) * t / flexible * 3.0 * (1.0 + tau) / (3.0 + tau)
            assert e["E1"] == pytest.approx(young, rel=1e-12)
            assert e["nu12"] == pytest.approx((1.0 - tau) / (3.0 + tau), rel=1e-12)
    if t != 0.1:
        return
    voronoi = periodic_framework(np.random.default_rng(4).random((20, 3)), np.ones(3))
    for fw in (_delaunay(40, 3), voronoi):
        for kappa in (None, "cowper"):
            frame = SpaceFrame.from_framework(fw, 0.02, shear_coefficient=kappa, rigid_joints=1.5)
            exact = frame_stiffness(frame).stiffness
            split = frame_stiffness(_split_with_stiff_ends(frame)).stiffness
            assert np.abs(exact - split).max() < 2e-6 * np.abs(split).max()
            plain = frame_stiffness(dataclasses.replace(frame, rigid=None)).stiffness
            assert np.abs(exact - plain).max() > 0.1 * np.abs(split).max()
    plane = Frame.from_framework(_delaunay(40, 5, dimension=2), 0.01, rigid_joints=2.0,
                                 shear_coefficient="cowper")  # fmt: skip
    split = frame_stiffness(_split_with_stiff_ends(plane)).stiffness
    assert np.abs(frame_stiffness(plane).stiffness - split).max() < 2e-6 * np.abs(split).max()
    cubic = Framework.lattice("cubic", 2)
    strut = float(np.linalg.norm(cubic.vectors(), axis=1).max())
    plain = network_stiffness(cubic, 0.05)
    rigid = network_stiffness(cubic, 0.05, rigid_joints=1.0)  # a = 0.05 at both ends
    assert rigid.young() == pytest.approx(plain.young() / (1.0 - 0.1 / strut), rel=1e-12)
    assert rigid.affine[0, 0] == pytest.approx(rigid.stiffness[0, 0], rel=1e-12)
    fw = _delaunay(60, 2)
    frame = SpaceFrame.from_framework(fw, 0.01, rigid_joints=0.5)
    lengths = np.linalg.norm(fw.vectors(), axis=1)
    long = lengths >= 0.01 / (1.0 - FLEXIBLE_LIMIT)
    assert long.sum() > 0.9 * len(long)
    np.testing.assert_allclose(frame.rigid[long], 0.005)
    short = SpaceFrame.from_framework(fw, 0.01, rigid_joints=100.0)
    np.testing.assert_allclose(short.rigid.sum(axis=1), (1.0 - FLEXIBLE_LIMIT) * lengths)
    with pytest.raises(ValueError, match="rigid_joints"):
        SpaceFrame.from_framework(fw, 0.01, rigid_joints=-1.0)
    none = SpaceFrame.from_framework(fw, 0.01, rigid_joints=0.0)  # the frame without them
    assert none.rigid is None and none.macroscopic_stiffness().shape == (6, 6)
    with pytest.raises(ValueError, match="rigid joints"):
        frame.macroscopic_stiffness()
    with pytest.raises(ValueError, match="rigid joints"):
        plane.buckling(np.diag([-1.0, 0.0]))


def test_frames_of_frameworks():
    """Frames of beams of circular sections from frameworks or their dictionaries, the
    open foams' Voronoi framework (the edges of the struts), the errors."""
    fw = _delaunay(80, 1)
    frame = SpaceFrame.from_framework(fw.to_dict(), np.full(fw.n_bars, 0.01), young=2.0)
    assert frame.axial == pytest.approx(2.0 * np.pi * 1e-4)
    assert frame.bending == pytest.approx(2.0 * np.pi * 1e-8 / 4.0)
    assert frame.torsion == pytest.approx(frame.bending / 1.3)  # G = E / 2.6, J = 2 I
    np.testing.assert_allclose(frame.vectors(), fw.vectors())
    plane = Frame.from_framework(_delaunay(60, 2, dimension=2), 0.01)
    assert plane.nodes.shape[1] == 2 and plane.n_dofs == 180
    germs = np.random.default_rng(3).random((30, 3)) * [1.0, 2.0, 1.5]
    voronoi = periodic_framework(germs, np.array([1.0, 2.0, 1.5]))
    ends, _ = periodic_edges(germs, np.array([1.0, 2.0, 1.5]))
    lengths = np.sort(np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1))
    np.testing.assert_allclose(np.sort(np.linalg.norm(voronoi.vectors(), axis=1)), lengths)
    assert voronoi.coordination() == 4.0 and np.all(voronoi.degrees() == 4)
    with pytest.raises(ValueError, match="along z"):
        Frame.from_framework(fw, 0.01)
    with pytest.raises(ValueError, match="several heights"):  # no bar across z, but in space
        tilted = Framework([[0.2, 0.2, 0.2], [0.6, 0.5, 0.7]], [[0, 1], [0, 1]],
                           [[0, 0, 0], [1, 0, 0]], [1, 1, 1])  # fmt: skip
        Frame.from_framework(tilted, 0.01)
    with pytest.raises(ValueError, match="positive"):
        SpaceFrame.from_framework(fw, -0.01)
    with pytest.raises(ValueError, match="isolated"):
        lone = Framework(np.zeros((3, 3)), [[0, 1]], [[1, 0, 0]], [1, 1, 1])
        SpaceFrame.from_framework(lone, 0.1)
    truss = SpaceFrame.from_framework(voronoi, 0.01)
    truss.bending[:] = truss.torsion[:] = 0.0
    with pytest.raises(ValueError, match="mechanisms"):
        frame_stiffness(truss)  # four bars per node: floppy as a truss


def test_delaunay_networks_are_stretching_dominated_not_affine():
    """Slender Delaunay networks store their energy in stretching, as a truss computed
    independently (the compatibility matrix of the framework, dense), with the Poisson's
    ratio of the affine estimate (1/4 in space) but 0.82 of its modulus: the non-affine
    relaxation of a disordered network of central forces (Z = 15.5); in the plane 0.76 and
    nu = 0.38."""
    fw = _delaunay(150, 5)
    s = network_stiffness(fw, 1e-3)
    v = fw.vectors()
    lengths = np.linalg.norm(v, axis=1)
    n = v / lengths[:, None]
    a, _ = fw.compatibility()
    b = np.column_stack([n[:, i] * v[:, j] * (1.0 if i == j else 0.5) +
                         n[:, j] * v[:, i] * (0.0 if i == j else 0.5)
                         for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))])  # fmt: skip
    k = np.pi * 1e-6 / lengths
    w = -np.linalg.lstsq(a.T @ (k[:, None] * a), a.T @ (k[:, None] * b), rcond=None)[0]
    extension = a @ w + b
    truss = extension.T @ (k[:, None] * extension)
    assert np.abs(s.stiffness - truss).max() < 1e-3 * np.abs(truss).max()  # beams: + bending
    pinned = SpaceFrame.from_framework(fw, 1e-3)
    pinned.bending[:] = pinned.torsion[:] = 0.0
    assert np.abs(frame_stiffness(pinned).stiffness - truss).max() < 1e-12 * np.abs(truss).max()
    assert s.split()["stretching"] > 0.999
    assert np.linalg.eigvalsh(s.affine - frame_stiffness(pinned).stiffness).min() > -1e-14
    ratios = [s.affinity(axis) for axis in range(3)]
    assert 0.75 < np.mean(ratios) < 0.88
    nu = np.mean([s.poisson(i, (i + 1) % 3) for i in range(3)])
    assert nu == pytest.approx(0.25, abs=0.015)
    plane = network_stiffness(_delaunay(500, 6, dimension=2), 1e-4, dimension=2)
    assert 0.7 < plane.affinity() < 0.82 and plane.split()["stretching"] > 0.999
    assert plane.poisson() == pytest.approx(0.38, abs=0.03)


def test_open_foams_are_bending_dominated():
    """Voronoi open foams: E ~ phi^2 (bending, rigid joints), E / (E_s phi^2) about 1
    (Gibson & Ashby), Poisson's ratio about 0.43, the stretching fraction growing with the
    density."""
    fw = periodic_framework(np.random.default_rng(11).random((60, 3)), np.ones(3))
    thin, thick = (network_stiffness(fw, r) for r in (0.002, 0.004))
    e_thin = np.mean([thin.young(a) for a in range(3)])
    e_thick = np.mean([thick.young(a) for a in range(3)])
    assert e_thick / e_thin == pytest.approx(16.0, rel=0.1)  # r^4: phi^2
    assert 0.8 < e_thin / thin.density**2 < 1.5
    assert thin.split()["bending"] > 0.9
    assert thick.split()["stretching"] > thin.split()["stretching"]
    nu = np.mean([thin.poisson(i, j) for i in range(3) for j in range(3) if i != j])
    assert thin.affinity() < 0.1 and 0.3 < nu < 0.6


def test_softening_of_non_identical_beams():
    """Ban et al. (2016): the exact derivatives of the Young's modulus in the factor of a
    beam against finite differences (concave: d2E < 0), the softening and scatter of
    replicas against the second-order theory (s CV^2, v CV^2), the laws of the factors."""
    frame = SpaceFrame.from_framework(_delaunay(40, 7), 0.02)
    for mode in ("axial", "bending", "modulus"):
        c = softening_coefficients(frame, mode)
        assert c["young"] == pytest.approx(frame_stiffness(frame).young(), rel=1e-12)
        assert np.all(c["d2E"] <= 1e-15 * c["young"])
        h = 1e-3
        for beam in (0, 17, 101):
            values = []
            for step in (-h, h):
                scale = np.ones(len(frame.beams))
                scale[beam] += step
                values.append(frame_stiffness(scaled_frame(frame, scale, mode)).young())
            first = (values[1] - values[0]) / (2 * h)
            assert first == pytest.approx(c["dE"][beam], rel=1e-5, abs=1e-9 * c["young"])
            second = (values[1] - 2.0 * c["young"] + values[0]) / h**2
            assert second == pytest.approx(c["d2E"][beam], rel=2e-3, abs=1e-7 * c["young"])
    followed = []
    r = softening(frame, "modulus", (0.15, 0.3), replicas=40, seed=1,
                  callback=lambda done, total: followed.append((done, total)))  # fmt: skip
    for cv, mean, error, std in zip(r.cv, r.corrected, r.error, r.std, strict=True):
        assert 1.0 - mean == pytest.approx(r.coefficients["mean"] * cv**2, abs=3.5 * error)
        assert std == pytest.approx(np.sqrt(r.coefficients["variance"]) * cv, rel=0.35)
    assert np.all(r.softening() > 0) and r.predicted()[1] == pytest.approx(4 * r.predicted()[0])
    assert followed == [(k, 80) for k in range(1, 81)]
    assert (r.mode, r.law, r.replicas, r.axis) == ("modulus", "normal", 40, 0)
    rows = r.rows()  # the table of the replicas
    assert [row["cv"] for row in rows] == [0.15, 0.3] and rows[1]["replicas"] == 40
    assert rows[1]["theory"] == pytest.approx(1.0 - r.predicted()[1])
    assert rows[0]["corrected_error"] == r.error[0] and rows[0]["std"] == r.std[0]
    assert rows[0]["mean_error"] == pytest.approx(r.std[0] / np.sqrt(40))
    assert rows[1]["scatter"] == pytest.approx(r.scatter()[1]) == pytest.approx(r.std[1] ** 2)
    assert rows[1]["scatter_error"] == pytest.approx(r.std[1] ** 2 * np.sqrt(2 / 39))
    assert rows[1]["theory_scatter"] == pytest.approx(r.coefficients["variance"] * 0.09)
    for kwargs, message in (({"law": "uniform"}, "law"), ({"law": "bimodal", "cvs": (0.5, 1.0)},
                            "below 1"), ({"cvs": ()}, "at least one")):  # fmt: skip
        with pytest.raises(ValueError, match=message):
            softening(frame, **{"replicas": 2, **kwargs})
    for law in ("normal", "lognormal", "bimodal"):
        x = fibre_scales(200_000, 0.3, law, seed=2)
        assert x.mean() == pytest.approx(1.0, abs=3e-3) and x.std() == pytest.approx(0.3, rel=0.01)
    with pytest.raises(ValueError, match="mode"):
        softening_coefficients(frame, "shear")
    with pytest.raises(ValueError, match="law"):
        fibre_scales(10, 0.1, "uniform")


def test_softening_of_timoshenko_beams():
    """Timoshenko beams whose shear stiffness follows the area of a general section (Ban et
    al.'s axial stiffness lowered to 1e-3 (EA)_0: Phi of order 10-100): the stiffness of a
    beam is concave in its EI and kGA, which adds U . K'' U / V to the second derivatives,
    exact against finite differences for the four modes; replicas of the area against s
    CV^2."""
    frame = SpaceFrame.from_framework(_delaunay(40, 7), 0.02, shear_coefficient="cowper")
    frame = scaled_frame(frame, 1e-3, "area")
    phi = 12.0 * frame.bending / (frame.shear * np.linalg.norm(frame.vectors(), axis=1) ** 2)
    assert np.median(phi) > 10.0
    assert frame_stiffness(frame).split()["shear"] > 0.2
    for mode in ("axial", "bending", "area", "modulus"):
        c = softening_coefficients(frame, mode)
        assert np.all(c["d2E"] <= 1e-15 * c["young"])
        h = 1e-3
        for beam in (0, 17, 101):
            values = []
            for step in (-h, h):
                scale = np.ones(len(frame.beams))
                scale[beam] += step
                values.append(frame_stiffness(scaled_frame(frame, scale, mode)).young())
            first = (values[1] - values[0]) / (2 * h)
            assert first == pytest.approx(c["dE"][beam], rel=1e-5, abs=1e-9 * c["young"])
            second = (values[1] - 2.0 * c["young"] + values[0]) / h**2
            assert second == pytest.approx(c["d2E"][beam], rel=2e-3, abs=1e-7 * c["young"])
    r = softening(frame, "area", (0.2,), replicas=30, seed=3)
    expected = r.coefficients["mean"] * 0.04
    assert 1.0 - r.corrected[0] == pytest.approx(expected, abs=3.5 * r.error[0])


def test_forces_and_energies_of_the_beams():
    """beam_forces: the axial forces of the dense path (Frame and SpaceFrame.axial_forces)
    to round-off under any strain; under a uniaxial stress, the energies of the beams sum
    to E . C E / 2 = 1 / 2E, split as FrameStiffness.split, the stretching energy of each
    beam is N^2 l / 2EA (rigid ends and Timoshenko beams too), and the virial of the forces
    of a truss gives the stress back; the frame drawn in its cell (frame_figure: its pieces
    as long as its beams, inside the cell) and the replicas drawn (softening_figure)."""
    from morphogen.homogenization.networks import beam_forces
    from morphogen.viz.figures import _cell_segments, frame_figure, softening_figure

    rng = np.random.default_rng(3)
    for frame in (Frame.lattice_of("kagome", (2, 2), elements=2),
                  SpaceFrame.lattice_of("octet", elements=2, radius=0.03)):  # fmt: skip
        dim = len(frame.lattice)
        strain = rng.standard_normal((dim, dim))
        strain = strain + strain.T
        dense, _ = frame.axial_forces(strain)
        loads = beam_forces(frame, strain=strain)
        assert np.abs(loads["force"] - dense).max() < 1e-12 * np.abs(dense).max()
    frame = SpaceFrame.from_framework(_delaunay(40, 7), 0.02, shear_coefficient="cowper",
                                      rigid_joints=0.5)  # fmt: skip
    s = frame_stiffness(frame)
    loads = beam_forces(frame, 1)
    parts = ("stretching", "bending", "torsion", "shear")
    total = sum(loads[k].sum() for k in parts)
    assert total == pytest.approx(0.5 / s.young(1), rel=1e-11)
    strain = s.uniaxial(1) / s.young(1)
    np.testing.assert_allclose(loads["strain"], strain, atol=1e-11 * np.abs(strain).max())
    split = s.split(s.uniaxial(1))
    for k in parts:
        assert loads[k].sum() / total == pytest.approx(split[k], rel=1e-11)
    flexible = np.linalg.norm(frame.vectors(), axis=1) - frame.rigid.sum(axis=1)
    volume = abs(np.linalg.det(frame.lattice))
    stretching = 0.5 * loads["force"] ** 2 * flexible / (frame.axial * volume)
    np.testing.assert_allclose(loads["stretching"], stretching, rtol=1e-11,
                               atol=1e-12 * stretching.max())  # fmt: skip
    truss = SpaceFrame.from_framework(_delaunay(40, 7), 0.02)
    truss.bending[:] = truss.torsion[:] = 0.0
    loads = beam_forces(truss, 0)
    d = truss.vectors()
    virial = np.einsum("m,mi,mj->ij", loads["force"] / np.linalg.norm(d, axis=1), d, d) / volume
    np.testing.assert_allclose(virial, np.diag([1.0, 0.0, 0.0]), atol=1e-11)
    assert loads["force"].max() > 0 > loads["force"].min()
    with pytest.raises(ValueError, match="axis"):
        beam_forces(truss, 3)
    # the frame drawn: the pieces of its beams in the cell, as long as them
    plane = network_frame(DelaunayNetwork(n_points=60, radius=0.003, dimension=2).generate(
        mg.Domain(size=(1.0, 1.0, 0.1)), seed=3))  # fmt: skip
    for drawn in (plane, frame):
        segments, owners = _cell_segments(drawn)
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        beams = np.linalg.norm(drawn.vectors(), axis=1)
        np.testing.assert_allclose(np.bincount(owners, lengths), beams, rtol=1e-12)
        reduced = segments @ np.linalg.inv(drawn.lattice).T
        assert reduced.min() > -1e-12 and reduced.max() < 1 + 1e-12
        assert len(owners) > len(drawn.beams)  # some cross the faces of the cell
    loads = beam_forces(plane, 1)
    fig = frame_figure(plane, loads["force"], scale="symmetric", label="N", title="plane")
    (ax, bar) = fig.axes
    assert bar.get_ylabel() == "N" and ax.get_title() == "plane"
    assert ax.collections[1].norm.vmin == -ax.collections[1].norm.vmax
    fig = frame_figure(frame, beam_forces(frame)["bending"], scale="log")
    assert fig.axes[0].name == "3d" and fig.axes[0].collections[1].norm.vmin > 0
    assert len(frame_figure(frame).axes) == 1  # the beams alone, without a colour bar
    with pytest.raises(ValueError, match="one finite value per beam"):
        frame_figure(frame, np.ones(3))
    with pytest.raises(ValueError, match="scale"):
        frame_figure(frame, scale="square")
    study = softening(frame, "bending", (0.1, 0.2), replicas=3, seed=2)
    fig = softening_figure([study, dataclasses.replace(study, mode="other")], title="replicas")
    left, right = fig.axes
    assert left.get_ylabel() == "⟨E⟩ / E₀" and right.get_xlabel().startswith("v CV²")
    assert len(right.get_legend().get_texts()) == 3  # the diagonal and the two modes


def test_network_models_keep_their_frame():
    """open_foam keeps its framework (the same struts as before), and the network models
    add the stiffness of their frame to the summary on request."""
    domain = mg.Domain.cube(1.0)
    plain = mg.models.OpenFoam(n_cells=30, strut_radius=0.01).generate(domain, seed=3)
    model = mg.models.OpenFoam(n_cells=30, strut_radius=0.01, frame_stiffness=True)
    micro = model.generate(domain, seed=3)
    assert np.array_equal(plain.segments.start, micro.segments.start)
    fw = Framework.from_dict(micro.metadata["framework"])
    assert fw.n_bars == len(micro.segments) and fw.coordination() == 4.0
    summary = micro.metadata["summary"]
    assert "frame_young" not in plain.metadata["summary"]
    expected = network_stiffness(fw, 0.01).summary()["frame_young"]
    assert summary["frame_young"] == pytest.approx(expected, rel=1e-6)
    assert summary["frame_stretching"] < 0.5 and summary["frame_affinity"] < 0.5
    c, s = (np.array([[summary[f"frame_{x}{min(i, j)}{max(i, j)}"] for j in range(1, 7)]
                      for i in range(1, 7)]) for x in "CS")  # fmt: skip
    assert np.allclose(s @ c, np.eye(6), atol=1e-6)  # its stiffness and compliance
    assert c[0, 0] == pytest.approx(network_stiffness(fw, 0.01).stiffness[0, 0], rel=1e-7)
    bone = mg.models.OpenFoam(n_cells=30, strut_radius=0.01, anisotropy=2.0, axis=2)
    stiff = network_stiffness(bone.generate(domain, seed=3).metadata["framework"], 0.01)
    assert stiff.young(2) > 3.0 * stiff.young(0)  # cells elongated along z
    d = mg.models.DelaunayNetwork(n_points=200, radius=0.002, dimension=2, frame_stiffness=True)
    s = d.generate(mg.Domain(size=(1.0, 1.0, 0.1)), seed=1).metadata["summary"]
    assert s["frame_stretching"] > 0.98 and 0.6 < s["frame_affinity"] < 0.9
    lattice = mg.models.RandomLattice(cell="octet", strut_radius=0.02, frame_stiffness=True)
    s = lattice.generate(mg.Domain.cube(2.0), seed=2).metadata["summary"]
    assert s["frame_stretching"] > 0.95 and s["frame_poisson"] == pytest.approx(1.0 / 3.0, abs=0.01)


def test_frames_of_realizations():
    """network_frame: the frame of a realization from its metadata, the radius of its
    summary (solved for a volume fraction) or of its model, a plane frame for a planar
    network, the options of network_stiffness; by_axis and the means of summary; a
    microstructure without a framework refused plainly."""
    domain = mg.Domain.cube(1.0)
    foam = mg.models.OpenFoam(n_cells=20, volume_fraction=0.05).generate(domain, seed=1)
    radius = foam.metadata["summary"]["strut_radius"]
    frame = network_frame(foam, shear_coefficient="cowper", rigid_joints=0.33)
    assert isinstance(frame, SpaceFrame) and frame.area == pytest.approx(np.pi * radius**2)
    expected = network_stiffness(foam.metadata["framework"], radius, shear_coefficient="cowper",
                                 rigid_joints=0.33)  # fmt: skip
    s = frame_stiffness(frame)
    assert np.array_equal(s.stiffness, expected.stiffness)
    rows = s.by_axis()
    assert [r["axis"] for r in rows] == ["x", "y", "z"]
    assert rows[2]["young"] == s.young(2) and rows[1]["poisson"] == s.poisson(1, 2)
    assert rows[0]["affinity"] == s.affinity(0) and rows[0]["shear"] == s.split()["shear"]
    parts = ("stretching", "bending", "torsion", "shear")
    assert sum(rows[1][k] for k in parts) == pytest.approx(1.0)
    assert s.summary(2.0)["frame_young"] == pytest.approx(np.mean([r["young"] for r in rows]) / 2)
    lattice = mg.models.RandomLattice(cell="bcc", strut_radius=0.05).generate(domain, seed=2)
    assert network_frame(lattice).area == pytest.approx(np.pi * 0.05**2)  # of its model
    plane = DelaunayNetwork(n_points=60, radius=0.003, dimension=2)
    planar = plane.generate(mg.Domain(size=(1.0, 1.0, 0.1)), seed=3)
    frame = network_frame(planar)
    assert isinstance(frame, Frame) and frame_stiffness(frame).by_axis()[1]["axis"] == "y"
    spheres = mg.models.BooleanSpheres(radius=0.1, volume_fraction=0.2).generate(domain, seed=1)
    with pytest.raises(ValueError, match="boolean_spheres has no framework of fibres or struts"):
        network_frame(spheres)


def test_frame_stiffness_command(tmp_path, capsys):
    """morphogen frame-stiffness: the frame of a realization of a configuration (seeded as
    the realizations of its batch run) or of a geometry, Euler-Bernoulli or Timoshenko beams
    (Cowper's coefficient or a given one), rigid joints, the modulus and Poisson's ratio of
    the fibres, Ban et al.'s softening coefficients, the text and JSON outputs, the errors."""
    import yaml

    from morphogen import cli
    from morphogen.homogenization.frames import cowper_coefficient
    from morphogen.seeding import derive_seed

    cfg = {"model": {"type": "open_foam", "n_cells": 20, "strut_radius": 0.01},
           "domain": {"size": 1.0, "periodic": True}, "seed": 5}  # fmt: skip
    path = tmp_path / "foam.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model = mg.models.OpenFoam(n_cells=20, strut_radius=0.01, frame_stiffness=True)
    first = model.generate(CUBE, derive_seed(5, 0))
    framework = first.metadata["framework"]

    def run(*args) -> dict:
        assert cli.main(["frame-stiffness", *map(str, args), "--json"]) == 0
        return json.loads(capsys.readouterr().out)

    def gap(out, expected) -> float:
        c = expected.stiffness
        return np.abs(np.array(out["stiffness"]) - c).max() / c.max()

    out = run(path)
    plain = network_stiffness(framework, 0.01)
    assert gap(out, plain) < 1e-12
    assert out["summary"]["frame_young"] == pytest.approx(
        first.metadata["summary"]["frame_young"], rel=1e-7
    )  # that of the model's summary: the first realization of the run
    assert out["model"] == "open_foam" and out["seed"] == 5 and out["realization"] == 0
    assert out["theory"] == "euler-bernoulli" and out["shear_coefficient"] is None
    assert out["beams"] == len(framework["bars"]) and out["radius"] == pytest.approx(0.01)
    assert [r["axis"] for r in out["axes"]] == ["x", "y", "z"] and "softening" not in out
    assert out["engineering"]["E2"] == pytest.approx(plain.young(1), rel=1e-12)
    s = np.array(out["compliance"])  # S = C^-1, in the notation stated
    assert np.allclose(s @ np.array(out["stiffness"]), np.eye(6), atol=1e-9)
    assert s[1, 1] == pytest.approx(1 / out["engineering"]["E2"]) and "23 31 12" in out["notation"]
    second = run(path, "--realization", "1")
    other = model.generate(CUBE, derive_seed(5, 1)).metadata["summary"]["frame_young"]
    assert second["summary"]["frame_young"] == pytest.approx(other, rel=1e-7)
    out = run(path, "--beams", "timoshenko", "--rigid-joints", "0.33", "--young", "2",
              "--poisson", "0.25", "--softening", "modulus", "area", "--axis", "y")  # fmt: skip
    exact = network_stiffness(framework, 0.01, young=2.0, poisson=0.25, shear_coefficient="cowper",
                              rigid_joints=0.33)  # fmt: skip
    assert gap(out, exact) < 1e-12
    assert out["shear_coefficient"] == pytest.approx(cowper_coefficient(0.25))
    assert out["summary"]["frame_young"] == pytest.approx(exact.summary(2.0)["frame_young"])
    assert out["axes"][0]["shear"] > 0 and out["rigid_joints"] == 0.33
    frame = SpaceFrame.from_framework(framework, 0.01, young=2.0, poisson=0.25,
                                      shear_coefficient="cowper", rigid_joints=0.33)  # fmt: skip
    coefficients = softening_coefficients(frame, "area", axis=1)
    assert out["softening"]["axis"] == "y" and set(out["softening"]) == {"axis", "modulus", "area"}
    assert out["softening"]["area"]["s"] == pytest.approx(coefficients["mean"], rel=1e-10)
    assert out["softening"]["area"]["v"] == pytest.approx(coefficients["variance"], rel=1e-10)
    out = run(path, "--beams", "timoshenko", "--shear-coefficient", "0.9")
    given = network_stiffness(framework, 0.01, shear_coefficient=0.9)
    assert out["shear_coefficient"] == 0.9
    assert out["summary"]["frame_young"] == pytest.approx(given.summary()["frame_young"])
    assert cli.main(["frame-stiffness", str(path), "--softening", "bending"]) == 0
    text = capsys.readouterr().out
    assert "open_foam, realization 0 of seed 5: a frame of" in text and "Euler-Bernoulli" in text
    assert "along z" in text and "frame_young" in text and "bending" in text.splitlines()[-1]
    assert "stiffness C (Voigt, order 11 22 33 23 31 12" in text and "compliance S = C^-1" in text
    # a geometry: a planar network, a plane frame
    planar = DelaunayNetwork(n_points=60, radius=0.003, dimension=2)
    micro = planar.generate(mg.Domain(size=(1.0, 1.0, 0.1), periodic=True), seed=2)
    mg.io.save_microstructure(micro, tmp_path / "planar.json")
    out = run(tmp_path / "planar.json")
    expected = network_stiffness(micro.metadata["framework"], 0.003, dimension=2)
    assert out["dimension"] == 2 and np.allclose(out["stiffness"], expected.stiffness, rtol=1e-12)
    assert set(out["engineering"]) == {"E1", "E2", "nu12", "nu21", "G12"}
    assert out["seed"] is None and out["realization"] is None
    assert "torsion" not in out["axes"][0] and len(out["axes"]) == 2
    assert cli.main(["frame-stiffness", str(tmp_path / "planar.json")]) == 0
    assert "a plane frame of" in capsys.readouterr().out
    # the errors, plainly
    spheres = {"model": {"type": "boolean_spheres", "radius": 0.1, "volume_fraction": 0.2},
               "domain": {"size": 1.0}, "seed": 1}  # fmt: skip
    (tmp_path / "spheres.yaml").write_text(yaml.safe_dump(spheres), encoding="utf-8")
    for args, message in (
        ([tmp_path / "spheres.yaml"], "boolean_spheres has no framework"),
        ([path, "--shear-coefficient", "0.9"], "Timoshenko"),
        ([tmp_path / "planar.json", "--softening", "axial", "--axis", "z"], "x or y"),
        ([tmp_path / "planar.json", "--seed", "3"], "a configuration"),
        ([path, "--rigid-joints", "-1"], "rigid_joints"),
    ):
        assert cli.main(["frame-stiffness", *map(str, args)]) == 2
        assert message in capsys.readouterr().err


def test_frame_stiffness_command_replicas(tmp_path, capsys):
    """morphogen frame-stiffness --replicas: Ban et al.'s replicas of each mode of
    --softening (--cv, --distribution, --replica-seed) against softening of the same frame
    and seed, in the JSON output, the text, the table (--table, .csv) and the figure
    (--figure); the errors."""
    import csv

    import yaml

    from morphogen import cli
    from morphogen.seeding import derive_seed

    cfg = {"model": {"type": "open_foam", "n_cells": 20, "strut_radius": 0.01},
           "domain": {"size": 1.0, "periodic": True}, "seed": 5}  # fmt: skip
    path = tmp_path / "foam.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    micro = mg.models.OpenFoam(n_cells=20, strut_radius=0.01).generate(CUBE, derive_seed(5, 0))
    frame = network_frame(micro)
    table, figure = tmp_path / "replicas.csv", tmp_path / "replicas.png"
    args = ["frame-stiffness", str(path), "--softening", "modulus", "bending", "--axis", "y",
            "--replicas", "5", "--cv", "0.1", "0.25", "--distribution", "lognormal",
            "--replica-seed", "3"]  # fmt: skip
    assert cli.main([*args, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    r = out["replicas"]
    assert (r["axis"], r["distribution"], r["replicas"], r["seed"]) == ("y", "lognormal", 5, 3)
    assert r["cv"] == [0.1, 0.25] and len(r["rows"]) == 4
    for k, mode in enumerate(("modulus", "bending")):
        expected = softening(frame, mode, (0.1, 0.25), replicas=5, law="lognormal", axis=1, seed=3)
        for got, row in zip(r["rows"][2 * k : 2 * k + 2], expected.rows(), strict=True):
            assert got["mode"] == mode and got["law"] == "lognormal"
            for key in ("mean", "corrected", "corrected_error", "theory", "scatter"):
                assert got[key] == pytest.approx(row[key], rel=1e-9)
        assert out["softening"][mode]["s"] == pytest.approx(expected.coefficients["mean"])
    assert cli.main([*args, "--table", str(table), "--figure", str(figure)]) == 0
    text = capsys.readouterr().out
    assert "replicas (lognormal law, 5 per CV, seed 3)" in text and "1 - s CV^2" in text
    assert f"written to {table}" in text and figure.stat().st_size > 1000
    with open(table, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["mode"] for row in rows] == ["modulus", "modulus", "bending", "bending"]
    assert float(rows[3]["corrected"]) == pytest.approx(r["rows"][3]["corrected"], rel=1e-12)
    for extra, message in (
        (["--replicas", "5"], "of the modes of --softening"),
        (["--softening", "axial", "--replicas", "1"], "at least 2"),
        (["--table", str(table)], "of the replicas"),
        (["--softening", "axial", "--replicas", "3", "--distribution", "bimodal", "--cv", "1.2"],
         "below 1"),
    ):  # fmt: skip
        assert cli.main(["frame-stiffness", str(path), *extra]) == 2
        assert message in capsys.readouterr().err
