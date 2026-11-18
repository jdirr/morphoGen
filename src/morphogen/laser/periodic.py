"""The periodic cell of a laser-architectured sheet and its homogenized tensile tests.

A pattern repeated over a whole sheet — a family of straight tracks at its spacing,
crossed families, corrugations, a honeycomb — is periodic: its *lattice*
(:func:`lattice`) is spanned by two vectors ``a1``, ``a2`` (mm; reduced to the most
compact pair, :func:`reduce_basis`). The *cell* (:class:`PeriodicCell`) is the
parallelogram they span, meshed by quadrilaterals whose nodes match on opposite edges.
Its displacements are periodic up to the macroscopic displacement gradient ``H``: a
node of the right (top) edge moves as its image on the left (bottom) edge plus ``H a1``
(``H a2``); one node is held (no translation), ``H`` is symmetric (no rotation): the
computational homogenization of periodic media (Michel, Moulinec & Suquet 1999). The
unknowns are the displacements of the independent nodes and ``H`` (:meth:`PeriodicCell.map`).

**Tensile test of the sheet** (:func:`cell_test`) in the direction ``n`` at ``angle``
degrees from ``x`` (the rolling direction), ``t`` across: ``H = a n⊗n + b t⊗t + c (n⊗t
+ t⊗n)``, ``a`` prescribed, ``b`` and ``c`` free — their conjugate forces, the volume of
the cell times the mean nominal stresses ``P_tt`` and ``P_nt + P_tn``, vanish: a
uniaxial stress. The mean nominal stress ``P_nn`` against the engineering strain ``|(I
+ H) n| − 1`` gives the yield and tensile strengths and the uniform elongation (the
maximum of the nominal stress: Considère's, the sheet's own, no specimen necking); the
width strain (the distance between two lines of the sheet along ``n``, the edges of a
strip: ``ln (det(I + H) / |(I + H) n|)``) over the thickness strain (the mean of ``ln
λ3`` over the cell) gives Lankford's coefficient — those of the sheet, free of the edges
of a specimen and of where its tracks fall in its gauge (Hill 1963: the representative
volume).

**Biaxial stretching** (``ratio``): ``b`` prescribed with ``a`` so that the true
strains across and along ``n`` keep the ratio, ``ln(1 + b) = ratio ln(1 + a)``, ``c``
free: the linear strain paths of sheet forming, from uniaxial tension (−½) through plane
strain (0) to equibiaxial stretching (1); the result's stress is then the mean nominal
stress ``P_nn`` (its maximum: the diffuse necking along ``n``), ``fields["nominal"]``
holds ``P_nn``, ``P_tt`` and ``P_nt`` (:func:`mean_nominal`).

**The shear** (``free_shear``): free by default, its conjugate stress vanishing, so that
a sheet whose tracks are oblique to ``n`` shears as it is stretched; held at zero, the
principal axes of the mean stretch stay along ``n`` and ``t``, the mean shear stress
then free: the path of the strains is prescribed in full, as by the tools of a forming
operation and in the analysis of Marciniak and Kuczyński.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

from morphogen.laser.mechanics import Membrane, TensileResult, curve_measures
from morphogen.laser.specimen import QuadMesh
from morphogen.laser.structures import load_test

__all__ = ["PeriodicCell", "cell_test", "lattice", "mean_nominal", "reduce_basis"]


def reduce_basis(a1, a2) -> tuple[np.ndarray, np.ndarray]:
    """The Lagrange–Gauss reduction of a basis of a plane lattice: the shortest pair of
    vectors spanning it (at 60° to 120° from each other), counter-clockwise."""
    a1, a2 = np.asarray(a1, dtype=float), np.asarray(a2, dtype=float)
    if abs(a1[0] * a2[1] - a1[1] * a2[0]) < 1e-12 * max(a1 @ a1, a2 @ a2, 1e-300):
        raise ValueError("lattice: the vectors are parallel")
    for _ in range(64):
        if a1 @ a1 > a2 @ a2:
            a1, a2 = a2, a1
        mu = round(float(a1 @ a2) / float(a1 @ a1))
        if mu == 0:
            break
        a2 = a2 - mu * a1
    if a1[0] * a2[1] - a1[1] * a2[0] < 0:
        a2 = -a2
    return a1, a2


def _turn(v, degrees: float) -> np.ndarray:
    th = math.radians(degrees)
    c, s = math.cos(th), math.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def lattice(pattern: dict, along: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The lattice vectors (mm, reduced) of a pattern repeated over the sheet (see the
    module): ``straight`` (``spacing`` across the tracks, ``along`` them: by default the
    spacing, 2 mm at most, the treatment being the same all along a track),
    ``corrugated`` and ``fourier`` (``period`` along, ``spacing`` across, shifted by the
    ``phase`` between neighbours), ``crossed`` (along each family, from a track of the
    other family to the next: ``spacing2 / sin 2θ`` along the first, ``spacing / sin
    2θ`` along the second), ``honeycomb`` (the centres of the hexagons), ``none`` (a
    square of 2 mm); all turned by the pattern's ``rotation``."""
    p = dict(pattern)
    kind = p.get("kind")
    if kind == "none":
        a1, a2 = np.array([2.0, 0.0]), np.array([0.0, 2.0])
    elif kind == "straight":
        s = p.get("spacing")
        if not s:
            raise ValueError("the cell of a straight family: give its spacing")
        th = float(p.get("angle", 0.0))
        along = min(float(s), 2.0) if along is None else float(along)
        a1, a2 = _turn([along, 0.0], th), _turn([0.0, float(s)], th)
    elif kind in ("corrugated", "fourier"):
        s = p.get("spacing")
        if not s:
            raise ValueError(f"the cell of {kind} tracks: give their spacing")
        period = float(p.get("period", 10.0))
        a1 = np.array([period, 0.0])
        a2 = np.array([-float(p.get("phase", 0.0)) / 360.0 * period, float(s)])
    elif kind == "crossed":
        th = math.radians(float(p.get("angle", 45.0)))
        s1 = float(p.get("spacing", 3.0))
        s2 = float(p.get("spacing2") or s1)
        sin2 = abs(math.sin(2.0 * th))
        if sin2 < 1e-6:
            raise ValueError("crossed: the families are parallel")
        a1 = s2 / sin2 * np.array([math.cos(th), math.sin(th)])
        a2 = s1 / sin2 * np.array([math.cos(th), -math.sin(th)])
    elif kind == "honeycomb":
        c = float(p.get("cell", 3.0))
        th = float(p.get("angle", 0.0))
        a1 = _turn([1.5 * c, 0.5 * math.sqrt(3.0) * c], th)
        a2 = _turn([0.0, math.sqrt(3.0) * c], th)
    else:
        raise ValueError(f"no lattice for the pattern {kind!r}: give the cell's vectors")
    rot = float(p.get("rotation", 0.0))
    return reduce_basis(_turn(a1, rot), _turn(a2, rot))


@dataclass
class PeriodicCell:
    """The parallelogram spanned by ``a1`` and ``a2`` from ``origin`` (mm), meshed by
    ``n1 x n2`` quadrilaterals (``mesh``); ``master``: the independent node each node
    moves with (itself for the independent ones, ``i < n1`` and ``j < n2``); ``shift``:
    the multiples ``(k1, k2)`` of ``a1`` and ``a2`` between them."""

    a1: np.ndarray
    a2: np.ndarray
    origin: np.ndarray
    mesh: QuadMesh
    master: np.ndarray
    shift: np.ndarray

    @classmethod
    def build(cls, a1, a2, *, element: float = 0.25, centre=(0.0, 0.0), shape=None) -> PeriodicCell:
        """The cell of a lattice centred at ``centre``, its elements about ``element``
        mm, or ``shape = (n1, n2)`` elements along ``a1`` and ``a2``."""
        a1, a2 = np.asarray(a1, dtype=float), np.asarray(a2, dtype=float)
        if a1[0] * a2[1] - a1[1] * a2[0] <= 0:
            raise ValueError("cell: a1, a2 counter-clockwise (reduce_basis)")
        if shape is None:
            n1 = max(2, math.ceil(np.linalg.norm(a1) / element - 1e-9))
            n2 = max(2, math.ceil(np.linalg.norm(a2) / element - 1e-9))
        else:
            n1, n2 = (int(v) for v in shape)
            if min(n1, n2) < 2:
                raise ValueError("cell: two elements at least along each vector")
        origin = np.asarray(centre, dtype=float) - 0.5 * (a1 + a2)
        i, j = np.meshgrid(np.arange(n1 + 1), np.arange(n2 + 1), indexing="ij")
        i, j = i.ravel(), j.ravel()
        nodes = origin + (i / n1)[:, None] * a1 + (j / n2)[:, None] * a2

        def ids(ii, jj):
            return ii * (n2 + 1) + jj

        qi, qj = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        qi, qj = qi.ravel(), qj.ravel()
        quads = np.stack(
            [ids(qi, qj), ids(qi + 1, qj), ids(qi + 1, qj + 1), ids(qi, qj + 1)], axis=1
        )
        master = ids(i % n1, j % n2)
        shift = np.stack([i // n1, j // n2], axis=1)
        mesh = QuadMesh(
            nodes,
            quads.astype(np.int64),
            np.flatnonzero(i == 0),
            np.flatnonzero(i == n1),
            np.flatnonzero(j == 0),
            np.flatnonzero(j == n2),
            (n1, n2),
        )
        return cls(a1, a2, origin, mesh, master.astype(np.int64), shift)

    @property
    def area(self) -> float:
        """The area of the cell (mm²)."""
        return float(abs(self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]))

    @property
    def independent(self) -> np.ndarray:
        """The independent nodes (``i < n1``, ``j < n2``), in the order of the unknowns."""
        return np.flatnonzero(self.master == np.arange(len(self.master)))

    def box(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        """The box ``(x0, x1, y0, y1)`` around the cell, widened by ``margin``."""
        lo = self.mesh.nodes.min(axis=0) - margin
        hi = self.mesh.nodes.max(axis=0) + margin
        return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))

    def map(self, angle: float = 0.0) -> sparse.csr_matrix:
        """The map ``T`` of the unknowns ``q`` (the displacements of the independent
        nodes, then ``a``, ``b``, ``c`` of ``H``, see the module, for the direction at
        ``angle`` degrees) to the nodal displacements ``u = T q``."""
        th = math.radians(angle)
        c, s = math.cos(th), math.sin(th)
        bases = (
            np.array([[c * c, c * s], [c * s, s * s]]),  # n ⊗ n
            np.array([[s * s, -c * s], [-c * s, c * c]]),  # t ⊗ t
            np.array([[-2 * c * s, c * c - s * s], [c * c - s * s, 2 * c * s]]),  # n⊗t + t⊗n
        )
        ind = self.independent
        col = np.full(len(self.master), -1, dtype=np.int64)
        col[ind] = np.arange(len(ind))
        m = col[self.master]
        nn = len(self.master)
        rows = [np.arange(2 * nn)]
        cols = [np.stack([2 * m, 2 * m + 1], axis=1).ravel()]
        vals = [np.ones(2 * nn)]
        d = self.shift[:, :1] * self.a1 + self.shift[:, 1:] * self.a2  # nodes x 2
        dep = np.flatnonzero(np.any(self.shift != 0, axis=1))
        for k, B in enumerate(bases):
            v = d[dep] @ B.T  # B d for each dependent node
            rows.append(np.stack([2 * dep, 2 * dep + 1], axis=1).ravel())
            cols.append(np.full(2 * len(dep), 2 * len(ind) + k))
            vals.append(v.ravel())
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(2 * nn, 2 * len(ind) + 3),
        )


def mean_nominal(membrane: Membrane) -> np.ndarray:
    """The mean nominal stress of a membrane in its current state (2 x 2, MPa): ``J σ
    F⁻ᵀ`` (``J`` the change of volume, ``det F λ3``) averaged over its layers and its
    area; over a periodic cell, the macroscopic nominal stress (Hill–Mandel)."""
    F = membrane.Fn
    sig = membrane.sig_n  # (elements, 4, layers, 3): xx, yy, xy
    S = np.empty((*sig.shape[:-1], 2, 2))
    S[..., 0, 0], S[..., 1, 1] = sig[..., 0], sig[..., 1]
    S[..., 0, 1] = S[..., 1, 0] = sig[..., 2]
    inv_t = np.linalg.inv(F).swapaxes(-1, -2)[:, :, None]
    J = np.linalg.det(F)[..., None] * membrane.lam_n
    P = (J[..., None, None] * (S @ inv_t)).mean(axis=2)
    w = membrane.w0
    return np.einsum("eg,egij->ij", w, P) / float(w.sum())


class _Reduced:
    """A membrane seen through the periodic map ``u = T q`` (forces ``Tᵀ f``, tangent
    ``Tᵀ K T``); its other attributes are those of the membrane."""

    def __init__(self, membrane: Membrane, T):
        self.inner = membrane
        self.T = T.tocsr()
        self.Tt = self.T.T.tocsr()
        self.ndof = self.T.shape[1]
        self.u = np.zeros(self.ndof)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def full(self, q) -> np.ndarray:
        return self.T @ q

    def reset(self) -> None:
        self.inner.reset()
        self.u = np.zeros(self.ndof)

    def evaluate(self, q):
        f, K, state, bad = self.inner.evaluate(self.T @ q)
        return self.Tt @ f, (self.Tt @ K @ self.T).tocsr(), state, bad

    def commit(self, q, state) -> None:
        self.inner.commit(self.T @ q, state)
        self.u = np.asarray(q, dtype=float).copy()


def cell_test(
    membrane: Membrane,
    cell: PeriodicCell,
    angle: float = 0.0,
    *,
    max_strain: float = 0.25,
    step: float = 0.004,
    max_step: float = 0.02,
    drop: float = 0.05,
    tol: float = 1e-5,
    max_iterations: int = 20,
    max_steps: int = 400,
    failure_margin: float | None = None,
    record_history: bool = False,
    keep_fields: bool = False,
    monitors: dict | None = None,
    ratio: float | None = None,
    until=None,
    free_shear: bool = True,
) -> TensileResult:
    """The tensile test of the sheet in the direction at ``angle`` degrees from x (see
    the module) on the membrane of the cell: the engineering strain along it up to
    ``max_strain`` (the steps and stops of
    :func:`~morphogen.laser.structures.load_test`); the result's stress is the mean
    nominal stress along the direction, its width and thickness strains give
    Lankford's coefficient (:meth:`~morphogen.laser.mechanics.TensileResult.r_value`).
    ``monitors``: more functions of the unknowns recorded at each step, after the
    membrane has taken its new state (``fields["monitors"]``). ``ratio``: biaxial
    stretching, the true strain across the direction prescribed at ``ratio`` times the
    true strain along it (see the module); ``until``: the test stops when this function
    of the unknowns returns true (after the monitors of the step); ``free_shear``: false
    holds the shear ``c`` at zero (see the module)."""
    T = cell.map(angle)
    red = _Reduced(membrane, T)
    na = T.shape[1] - 3  # the unknown a, then b and c
    w = membrane.w0
    wsum = float(w.sum())
    th = math.radians(angle)
    n, t = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])

    def thickness(_q):
        return float((np.log(membrane.lam_n).mean(axis=2) * w).sum() / wsum)

    def nominal(u, v):
        return lambda _q: float(u @ mean_nominal(membrane) @ v)

    monitors = {
        **(monitors or {}),
        "c": lambda q: float(q[na + 2]),
        "b": lambda q: float(q[na + 1]),
        "thickness_strain": thickness,
    }
    if ratio is not None:
        monitors.update({"P_nn": nominal(n, n), "P_tt": nominal(t, t), "P_nt": nominal(n, t)})
    moved, direction = [na], [1.0]
    if ratio is not None:  # a linear path of the true strains: ln(1 + b) = ratio ln(1 + a)
        rho = float(ratio)
        moved = [na, na + 1]

        def direction(d):
            return (
                np.array([d, (1.0 + d) ** rho - 1.0]),
                np.array([1.0, rho * (1.0 + d) ** (rho - 1.0)]),
            )

    # the first independent node, the origin of the cell, and with fixed axes the shear
    fixed = [0, 1] if free_shear else [0, 1, na + 2]
    res = load_test(
        red,
        fixed,
        moved,
        direction,
        length=1.0,
        section=cell.area * membrane.thickness,
        max_disp=max_strain,
        step=step,
        max_step=max_step,
        drop=drop,
        tol=tol,
        max_iterations=max_iterations,
        max_steps=max_steps,
        failure_margin=failure_margin,
        record_history=record_history,
        monitors=monitors,
        keep_fields=keep_fields,
        until=until,
    )
    mon = res.fields["monitors"]
    a, b, c = res.displacement, mon["b"], mon["c"]
    strain = np.sqrt((1.0 + a) ** 2 + c**2) - 1.0
    stress = res.stress if ratio is None else np.asarray(mon["P_nn"])
    # the proof stress on the elastic slope: the phases of a treated steel share it
    ys, uts, eu, energy, necked = curve_measures(strain, stress, young=membrane.young)
    failure = res.failure_strain
    if math.isfinite(failure):  # from a (the controlled unknown) to the strain along n
        failure = float(np.interp(failure, a, strain))
    fields = dict(res.fields)
    fields["cell"] = (cell.a1.copy(), cell.a2.copy())
    fields["macroscopic"] = {"a": a.copy(), "b": np.asarray(b), "c": np.asarray(c)}
    if ratio is not None:
        fields["nominal"] = {k: np.asarray(mon[f"P_{k}"]) for k in ("nn", "tt", "nt")}
    return replace(
        res,
        strain=strain,
        stress=stress,
        yield_strength=ys,
        tensile_strength=uts,
        uniform_elongation=eu,
        energy=energy,
        necked=necked,
        failure_strain=failure,
        # the distance between the edges of a strip along n: det(I + H) / |(I + H) n|
        width_strain=np.log(
            ((1.0 + a) * (1.0 + np.asarray(b)) - np.asarray(c) ** 2) / (1.0 + strain)
        ),
        thickness_strain=np.asarray(mon["thickness_strain"]),
        fields=fields,
    )
