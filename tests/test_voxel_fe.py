"""The trilinear voxels of the FFT solvers assembled on the voxels of the materials and solved
by conjugate gradients preconditioned by a sparse factorization
(:mod:`morphogen.homogenization.voxel_fe`): the operator and the solutions of the FFT
solvers, in a few iterations where these take hundreds near the rigidity threshold."""

import numpy as np
import pytest

from morphogen.homogenization import (
    VOID,
    Loading,
    Material,
    conductivity_tensor,
    stiffness_tensor,
)
from morphogen.homogenization.fft import FFTElasticProblem, FFTThermalProblem, apply_operator
from morphogen.homogenization.voxel_fe import (
    VoxelFEElasticProblem,
    VoxelFEThermalProblem,
    load_bearing,
    voxel_stiffness,
)
from morphogen.image import VoxelImage

SOLID = Material(young=1.0, poisson=0.3, conductivity=1.0)
SOFT = Material(young=0.2, poisson=0.1, conductivity=0.3)
UNIAXIAL = Loading.strain([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _sites(shape, fraction, seed, spacing=(1.0, 1.0, 1.0)) -> VoxelImage:
    mask = np.random.default_rng(seed).random(shape) < fraction
    return VoxelImage(mask.astype(np.uint8), spacing=spacing, periodic=True)


@pytest.mark.parametrize("shape", [(6, 5, 4), (7, 6, 1), (1, 5, 4), (2, 3, 4)])
@pytest.mark.parametrize("cls", [FFTElasticProblem, FFTThermalProblem])
def test_the_matrix_is_the_operator_of_the_fft_solver(shape, cls):
    """On the nodes of the voxels of the materials the matrix is the operator of the FFT
    solver, which vanishes on the others (those that only void voxels touch), the two nodes
    of an axis one voxel thick folded into one, as the periodic grid makes them, and those of
    an axis two voxels long kept apart."""
    problem = cls(_sites(shape, 0.5, 1, (1.0, 1.3, 0.8)), {0: VOID, 1: SOLID}, device="cpu")
    nd, mats = problem.nd, problem.element_matrices
    matrix, dofs = voxel_stiffness(problem.shape, problem.mat, mats, nd)
    v = np.random.default_rng(2).standard_normal(int(np.prod(shape)) * nd)
    out = np.empty_like(v)
    apply_operator(*problem.shape, nd, problem.mat, mats, v, out)
    np.testing.assert_allclose(matrix @ v[dofs], out[dofs], rtol=0, atol=1e-12 * abs(out).max())
    rest = np.ones(v.size, dtype=bool)
    rest[dofs] = False
    assert not np.any(out[rest]) and np.all(np.diff(dofs) > 0)
    assert abs(matrix - matrix.T).max() < 1e-14 * abs(matrix).max()


@pytest.mark.parametrize(
    ("shape", "fraction", "materials"),
    [
        ((16, 12, 1), 0.5, {0: SOFT, 1: SOLID}),
        ((20, 20, 1), 0.6, {0: VOID, 1: SOLID}),
        ((6, 8, 7), 0.4, {0: SOFT, 1: SOLID}),
        ((8, 8, 8), 0.3, {0: VOID, 1: SOLID}),
    ],
)
def test_the_tensors_of_the_fft_solver(shape, fraction, materials):
    """The stiffness (6 x 6) and conductivity tensors of planar and 3-D images, the other
    phase soft or void, voxels of three spacings: those of the FFT solver to the tolerance, in
    two iterations per loading."""
    image = _sites(shape, fraction, 3, (1.0, 1.3, 0.8))
    kw = {"tol": 1e-10, "maxiter": 20000}
    c_fft, _ = stiffness_tensor(FFTElasticProblem(image, materials, device="cpu", **kw))
    c_fe, solutions = stiffness_tensor(VoxelFEElasticProblem(image, materials, **kw))
    np.testing.assert_allclose(c_fe, c_fft, rtol=0, atol=1e-8 * abs(c_fft).max())
    assert max(s.info.iterations for s in solutions) <= 3
    assert {s.info.method for s in solutions} == {"fe-voxels"}
    k_fft, _ = conductivity_tensor(FFTThermalProblem(image, materials, device="cpu", **kw))
    k_fe, _ = conductivity_tensor(VoxelFEThermalProblem(image, materials, **kw))
    np.testing.assert_allclose(k_fe, k_fft, rtol=0, atol=1e-8 * abs(k_fft).max())


@pytest.mark.parametrize(
    ("shape", "fraction", "seed", "bears", "iterations"),
    [
        ((24, 24, 1), 0.4647, 0, False, 300),
        ((24, 24, 1), 0.4647, 1, True, 300),
        ((10, 10, 10), 0.135, 0, False, 600),
        ((10, 10, 10), 0.135, 1, True, 800),
    ],
)
def test_the_moduli_near_the_rigidity_threshold(shape, fraction, seed, bears, iterations):
    """Sites of the square and cubic lattices at their rigidity thresholds, void around: the
    stresses of a uniaxial strain of the FFT solver, those of an image that bears to the
    tolerance and those of a floppy one at the round-off, in 2 iterations where the FFT
    solver takes 300 to 900."""
    image = _sites(shape, fraction, seed)
    fft = FFTElasticProblem(image, {0: VOID, 1: SOLID}, tol=1e-10, maxiter=20000, device="cpu")
    fe = VoxelFEElasticProblem(image, {0: VOID, 1: SOLID}, tol=1e-10)
    a, b = fft.solve(UNIAXIAL), fe.solve(UNIAXIAL)
    assert b.info.iterations <= 3 < iterations < a.info.iterations
    assert b.info.residual <= 1e-10 and b.info.n_dofs == 3 * np.unique(_nodes(image)).size
    if bears:
        assert a.stress[0, 0] > 1e-4
        np.testing.assert_allclose(b.stress, a.stress, rtol=0, atol=1e-6 * a.stress[0, 0])
    else:
        assert abs(a.stress).max() < 1e-11 and abs(b.stress).max() < 1e-10
    assert b.material_fraction == a.material_fraction


def _nodes(image: VoxelImage) -> np.ndarray:
    """The nodes of the voxels of phase 1 on the periodic grid of the image."""
    i, j, k = np.nonzero(image.data == 1)
    nx, ny, nz = image.shape
    return np.concatenate([((i + a) % nx * ny + (j + b) % ny) * nz + (k + c) % nz
                           for a in (0, 1) for b in (0, 1) for c in (0, 1)])  # fmt: skip


def test_uniform_and_void_images_and_bad_arguments():
    """A uniform material bears the affine field without iterations (its loads cancel), a
    void image nothing; the shift is a fraction of the diagonal."""
    solid = VoxelImage(np.ones((4, 4, 4), dtype=np.uint8), periodic=True)
    s = VoxelFEElasticProblem(solid, {1: SOLID}).solve(UNIAXIAL)
    lam = 0.3 / (1.3 * 0.4)
    assert s.info.iterations == 0 and abs(s.stress[0, 0] - (lam + 1 / 1.3)) < 1e-12
    void = VoxelImage(np.zeros((4, 4, 4), dtype=np.uint8), periodic=True)
    assert not np.any(VoxelFEElasticProblem(void, {0: VOID}).solve(UNIAXIAL).stress)
    for shift in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError, match="shift"):
            VoxelFEElasticProblem(solid, {1: SOLID}, shift=shift)


def test_the_voxels_that_bear():
    """load_bearing keeps the parts of the voxels, cut at the nodes that alone join them,
    through which a cycle winds around the box along the axis: the diagonal chain of pins
    (winding along x and y) stays, a pixel hanging from one of its corners and a block of
    pixels pinned to it at one node go; a column of pixels (winding along y) stays for y
    and goes for x; in 3-D a chain of voxels along the body diagonal stays and a voxel
    hinged to it at one edge, which may bear as a bar, stays with it."""
    image = np.zeros((12, 12, 1), dtype=bool)
    k = np.arange(12)
    image[k, k, 0] = True  # the chain of pins
    image[4, 2, 0] = True  # its corner (4, 3) only
    image[8:10, 5:7, 0] = True  # its node (8, 7) only
    kept = load_bearing(image, 0)
    expected = np.zeros_like(image)
    expected[k, k, 0] = True
    np.testing.assert_array_equal(kept, expected)
    np.testing.assert_array_equal(load_bearing(image, 1), expected)
    column = np.zeros((12, 12, 1), dtype=bool)
    column[3, :, 0] = True
    assert not load_bearing(column, 0).any()
    np.testing.assert_array_equal(load_bearing(column, 1), column)
    cube = np.zeros((8, 8, 8), dtype=bool)
    k = np.arange(8)
    cube[k, k, k] = True
    cube[3, 4, 2] = True  # hinged at the edge (3, 4, 3)-(4, 4, 3) of the voxel (3, 3, 3)
    assert np.array_equal(load_bearing(cube, 0), cube)
    cube[6, 1, 1] = True  # alone: floats
    assert load_bearing(cube, 0).sum() == 9


def test_the_parts_merged_by_their_voxels():
    """The blocks of the incidences of the voxels merged where they share a voxel (the
    union-find of load_bearing, by size with the paths halved) are the connected components
    of the graph of the blocks joined by their voxels, numbered in the order of their first
    voxels: on random blocks and on a chain of 20,000 blocks visited from its far end,
    whose plain forest grows as long as the chain."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    from morphogen.homogenization.voxel_fe import _parts

    rng = np.random.default_rng(3)
    chain = np.arange(20_000)[::-1]
    cases = ((rng.integers(0, 9000, 3000 * 4), 9000, 4),
             (np.stack([chain[1:], chain[:-1]], axis=1).ravel(), 20_000, 2))  # fmt: skip
    for block, nb, nc in cases:
        part, count = _parts(block.astype(np.int64), nb, nc)
        pairs = block.reshape(-1, nc)
        rows = np.repeat(pairs[:, 0], nc)
        graph = sp.coo_matrix((np.ones(rows.size), (rows, pairs.ravel())), shape=(nb, nb))
        _, label = connected_components(graph, directed=False)
        first = label[pairs[:, 0]]
        _, order = np.unique(first, return_index=True)
        renumber = np.empty(label.max() + 1, np.int64)
        renumber[first[np.sort(order)]] = np.arange(order.size)
        np.testing.assert_array_equal(part, renumber[first])
        assert count == order.size


@pytest.mark.parametrize(("shape", "fraction"), [((32, 32, 1), 0.4627), ((12, 12, 12), 0.1307)])
def test_the_voxels_that_bear_have_the_moduli_of_the_image(shape, fraction):
    """Sites of the square and cubic lattices at their rigidity thresholds, void around: the
    voxels that bear along x and y alone, fewer degrees of freedom, have the moduli of the
    whole image (bearing or floppy) and its conductivities, to the round-off."""
    solved = 0
    for seed in range(4):
        mask = np.random.default_rng(seed).random(shape) < fraction
        for axis in (0, 1):
            kept = load_bearing(mask, axis)
            assert np.all(kept <= mask) and kept.sum() < mask.sum()
            strain = np.zeros(6)
            strain[axis] = 1.0
            loads = (Loading.strain(strain), Loading.gradient(np.eye(3)[axis]))
            for cls, loading in zip((VoxelFEElasticProblem, VoxelFEThermalProblem), loads,
                                    strict=True):  # fmt: skip
                a, b = (cls(VoxelImage(m.astype(np.uint8), periodic=True), {0: VOID, 1: SOLID},
                            tol=1e-12).solve(loading) for m in (mask, kept))  # fmt: skip
                assert b.info.n_dofs < a.info.n_dofs
                assert abs(b.energy - a.energy) <= 1e-9 * max(a.energy, 1e-3)
                solved += a.energy > 1e-6
    assert solved >= 4
