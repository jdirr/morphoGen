"""Hygromorph plates: multilayers that bend as moisture diffuses through them.

Pine cone scales and 4D-printed hygromorphs bend because their layers swell
differently and because the moisture profile through the thickness evolves (Mou,
Dirrenberger & Le Duigou 2025). Here a free multilayer plate, each layer isotropic in
elasticity (plane stress) with orthotropic swelling ``beta_x``, ``beta_y`` (strain per
unit concentration), a diffusivity and a solubility: moisture enters by Fick's law
through the thickness (finite volumes, continuity of the activity between layers,
implicit Euler), the swelling strains ``beta c(z, t)`` are eigenstrains, and the
classical laminate theory gives the mid-plane strains and curvatures of the free plate
(``[A B; B D] [eps0; kappa] = [N*; M*]``) at every time: the sequence of bendings along x
and y. The geometric nonlinearity of large deflections (Föppl–von Kármán, which selects
cylindrical shapes beyond a critical curvature) is left to finite element models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

#: Voigt components of the in-plane quantities of a plate normal to each axis.
_IN_PLANE = {2: ((0, 1), (0, 1, 5)), 0: ((1, 2), (1, 2, 3)), 1: ((2, 0), (2, 0, 4))}


@dataclass(frozen=True)
class Layer:
    """A layer of a hygromorph: thickness, Young's modulus and Poisson's ratio (or the
    in-plane ``stiffness``: the 3 x 3 plane-stress matrix of the components ``xx``,
    ``yy``, ``xy``, engineering shear, of an orthotropic or anisotropic layer), swelling
    strains per unit concentration along x and y (and the engineering shear ``beta_xy``),
    the diffusivity through the thickness and the solubility (the concentration at
    equilibrium with the outside activity 1)."""

    thickness: float
    young: float
    poisson: float = 0.3
    beta_x: float = 0.0
    beta_y: float = 0.0
    diffusivity: float = 0.0
    solubility: float = 1.0
    stiffness: tuple | None = None
    beta_xy: float = 0.0

    def plane_stiffness(self) -> np.ndarray:
        """The plane-stress stiffness ``Q`` (3 x 3: xx, yy, xy)."""
        if self.stiffness is not None:
            return np.asarray(self.stiffness, dtype=float).reshape(3, 3)
        e, nu = self.young, self.poisson
        q = e / (1.0 - nu * nu)
        return np.array([[q, nu * q, 0.0], [nu * q, q, 0.0], [0.0, 0.0, 0.5 * e / (1.0 + nu)]])

    @classmethod
    def from_microstructure(
        cls, thickness: float, sample, materials, *, solubility=None, normal: int = 2, **options
    ) -> Layer:
        """A layer whose properties are the apparent ones of a microstructure (a voxel
        image or a conforming mesh, periodic homogenization by
        :func:`~morphogen.homogenization.homogenize`): the plane-stress stiffness of the
        apparent stiffness tensor in the plane normal to ``normal``, the swelling of its
        apparent expansion tensor (give each material its swelling strain per unit
        concentration as ``expansion``), the diffusivity through the thickness, the
        apparent conductivity across it (give each material its permeability, diffusivity
        times solubility, as ``conductivity``) over the mean ``solubility`` of the phases
        (a mapping phase -> solubility, default 1)."""
        from morphogen.homogenization.apparent import homogenize

        props = ("stiffness", "conductivity_tensor", "thermal_expansion")
        out = homogenize(sample, materials, bcs=("periodic",), properties=props, **options)
        c = np.asarray(out["stiffness.periodic"]["tensor"])
        k = np.asarray(out["conductivity_tensor.periodic"]["tensor"])
        a = np.asarray(out["thermal_expansion.periodic"]["tensor"])
        (i, j), voigt = _IN_PLANE[int(normal)]
        compliance = np.linalg.inv(c)[np.ix_(voigt, voigt)]
        q = np.linalg.inv(compliance)
        fractions = _fractions(sample)
        solubility = solubility or {}
        mean_solubility = sum(f * float(solubility.get(r, 1.0)) for r, f in fractions.items())
        return cls(
            thickness=float(thickness),
            young=float(1.0 / compliance[0, 0]),
            poisson=float(-compliance[0, 1] / compliance[0, 0]),
            beta_x=float(a[i, i]),
            beta_y=float(a[j, j]),
            beta_xy=float(2.0 * a[i, j]),
            diffusivity=float(k[normal, normal] / mean_solubility),
            solubility=float(mean_solubility),
            stiffness=tuple(map(tuple, q)),
        )


def _fractions(sample) -> dict[int, float]:
    """Volume fractions of the phases of an image or a mesh."""
    from morphogen.image import VoxelImage

    if isinstance(sample, VoxelImage):
        labels, counts = np.unique(np.asarray(sample.data), return_counts=True)
        return {int(r): float(n) / counts.sum() for r, n in zip(labels, counts, strict=True)}
    volumes = np.abs(sample.volumes())
    return {
        int(r): float(volumes[sample.phase == r].sum() / volumes.sum())
        for r in np.unique(sample.phase)
    }


class Multilayer:
    """A free plate of layers stacked from ``z = 0`` (the bottom face) upwards."""

    def __init__(self, layers, nodes_per_layer: int = 40) -> None:
        self.layers = tuple(layers)
        if not self.layers:
            raise ValueError("Multilayer: at least one layer")
        edges = np.concatenate([[0.0], np.cumsum([lay.thickness for lay in self.layers])])
        self.thickness = float(edges[-1])
        cells = []
        for i in range(len(self.layers)):
            e = np.linspace(edges[i], edges[i + 1], nodes_per_layer + 1)
            cells.append(np.column_stack([e[:-1], e[1:], np.full(nodes_per_layer, i)]))
        cells = np.vstack(cells)
        self.lower, self.upper, self.index = cells[:, 0], cells[:, 1], cells[:, 2].astype(int)
        self.z = 0.5 * (self.lower + self.upper)
        self.dz = self.upper - self.lower

    def _prop(self, name: str) -> np.ndarray:
        return np.array([getattr(self.layers[i], name) for i in self.index], dtype=float)

    def _q(self) -> np.ndarray:
        """Plane-stress stiffness of each cell (cells x 3 x 3)."""
        per_layer = np.stack([lay.plane_stiffness() for lay in self.layers])
        return per_layer[self.index]

    def stiffness(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``A``, ``B`` and ``D`` (3 x 3: xx, yy, xy) about ``z = 0``."""
        q = self._q()
        z0, z1 = self.lower, self.upper
        a = np.einsum("nij,n->ij", q, z1 - z0)
        b = np.einsum("nij,n->ij", q, (z1**2 - z0**2) / 2.0)
        d = np.einsum("nij,n->ij", q, (z1**3 - z0**3) / 3.0)
        return a, b, d

    def deformation(self, concentration, *, full: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Mid-plane strains (about ``z = 0``) and curvatures ``(kappa_x, kappa_y)`` of
        the free plate under the swelling of the concentrations of the cells (a profile,
        or an array of profiles: times x cells); ``full``: also the shear strain and the
        twist (``kappa_xy``) of anisotropic layers."""
        c = np.atleast_2d(np.asarray(concentration, dtype=float))
        q = self._q()
        beta = np.stack([self._prop("beta_x"), self._prop("beta_y"), self._prop("beta_xy")], axis=1)
        sig = np.einsum("nij,nj->ni", q, beta)  # Q : beta of each cell
        w0, w1 = self.dz, (self.upper**2 - self.lower**2) / 2.0
        n_star = (c * w0) @ sig  # times x 3
        m_star = (c * w1) @ sig
        a, b, d = self.stiffness()
        big = np.block([[a, b], [b, d]])
        sol = np.linalg.solve(big, np.concatenate([n_star, m_star], axis=-1).T).T
        strains, curvatures = sol[:, :3], sol[:, 3:]
        if not full:
            strains, curvatures = strains[:, :2], curvatures[:, :2]
        if np.ndim(concentration) == 1:
            return strains[0], curvatures[0]
        return strains, curvatures

    def moisture(
        self,
        times,
        *,
        exposed: str = "bottom",
        activity: float | Callable[[float], float] = 1.0,
        initial: float = 0.0,
        steps: int = 200,
    ) -> np.ndarray:
        """Concentration profiles (times x cells) as moisture enters through the
        ``exposed`` face (``bottom``, ``top`` or ``both``) held at ``activity`` (a number
        or a function of time); the activity is continuous between layers, the
        concentration is the solubility times it."""
        if exposed not in ("bottom", "top", "both"):
            raise ValueError('exposed: "bottom", "top" or "both"')
        d, s = self._prop("diffusivity"), self._prop("solubility")
        perm = d * s
        h = self.dz
        # conductances between neighbouring cells (half-cells in series)
        g = 1.0 / (
            0.5 * h[:-1] / np.maximum(perm[:-1], 1e-300)
            + 0.5 * h[1:] / np.maximum(perm[1:], 1e-300)
        )
        g_bot = perm[0] / (0.5 * h[0]) if exposed in ("bottom", "both") else 0.0
        g_top = perm[-1] / (0.5 * h[-1]) if exposed in ("top", "both") else 0.0
        diag = np.zeros(len(h))
        diag[:-1] += g
        diag[1:] += g
        diag[0] += g_bot
        diag[-1] += g_top
        value = activity if callable(activity) else (lambda t, a=float(activity): a)
        cap = s * h
        u = np.full(len(h), float(initial))
        times = np.asarray(times, dtype=float)
        out, t = [], 0.0
        for t_out in times:
            span = t_out - t
            if span > 0:
                if t == 0.0:
                    r = 1.05
                    first = span * (r - 1.0) / (r**steps - 1.0)
                    dts = first * r ** np.arange(steps)
                else:
                    dts = np.full(steps, span / steps)
                for dt in dts:
                    t += dt
                    ab = np.zeros((3, len(h)))
                    ab[0, 1:] = -g
                    ab[1] = cap / dt + diag
                    ab[2, :-1] = -g
                    rhs = cap / dt * u
                    rhs[0] += g_bot * value(t)
                    rhs[-1] += g_top * value(t)
                    u = solve_banded((1, 1), ab, rhs)
                t = float(t_out)
            out.append(s * u)
        return np.array(out)

    def curvature_history(self, times, **moisture) -> dict[str, np.ndarray]:
        """Moisture profiles, mid-plane strains and curvatures of the free plate at
        ``times`` (keywords: those of :meth:`moisture`)."""
        c = self.moisture(times, **moisture)
        strains, curvatures = self.deformation(c)
        return {
            "times": np.asarray(times, float),
            "concentration": c,
            "strains": strains,
            "curvatures": curvatures,
        }


def timoshenko_curvature(strain: float, h1: float, h2: float, e1: float, e2: float) -> float:
    """Curvature of a free bilayer whose upper layer (2) swells by ``strain`` relative to
    the lower one (Timoshenko 1925): ``6 de (1 + m)^2 / (h (3 (1 + m)^2 + (1 + m n)
    (m^2 + 1/(m n))))``, ``m = h1/h2``, ``n = E1/E2``; positive: the plate bends towards
    its lower face (centre of curvature below)."""
    m, n, h = h1 / h2, e1 / e2, h1 + h2
    return (
        6.0
        * strain
        * (1.0 + m) ** 2
        / (h * (3.0 * (1.0 + m) ** 2 + (1.0 + m * n) * (m * m + 1.0 / (m * n))))
    )
