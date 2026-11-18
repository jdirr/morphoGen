"""Specimens of laser-architectured sheets and their meshes.

The sheet's coordinates: ``x`` along the tensile axis, ``y`` across; the parallel
(gauge) part of a specimen occupies ``[0, length] x [0, width]``, the band where the
patterns are drawn (:meth:`Specimen.band`). Kinds:

``dogbone``
    a flat tensile specimen: the parallel part, circular fillets of radius ``radius``
    to grips of width ``grip_width`` and length ``grip_length`` (the part between the
    fillet and the jaws), the strain read by a virtual extensometer of length
    ``extensometer`` centred on the parallel part (DIC);
``strip``
    the parallel part alone, its ends held by the grips (``clamped``: no lateral
    contraction there) or free to contract;
``cell``
    a periodic cell of the sheet, ``length`` x ``width`` (a period of the pattern along
    x, the spacing of its tracks along y): an infinite architectured sheet.

Meshes are structured quadrilaterals mapped on the outline, about ``element`` mm.
Defaults: the specimens of Wang (2022, thesis ch. 5): a parallel part 30 mm long and
5 mm wide, 1 mm thick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KINDS = ("dogbone", "strip", "cell")
#: Gauss points of the 2 x 2 rule on [-1, 1]².
_GP = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) / math.sqrt(3.0)


@dataclass
class QuadMesh:
    """Quadrilaterals: ``nodes`` (n x 2), ``quads`` (e x 4, counter-clockwise), and the
    node sets of the outline (``left``, ``right``, ``bottom``, ``top`` edges of the
    mapped grid)."""

    nodes: np.ndarray
    quads: np.ndarray
    left: np.ndarray
    right: np.ndarray
    bottom: np.ndarray
    top: np.ndarray
    shape: tuple

    def gauss(self):
        """The Gauss points ``(e x 4 x 2)`` coordinates, the derivatives of the shape
        functions in the reference configuration ``(e x 4 x 4 x 2)`` and the weights
        ``(e x 4)`` (areas)."""
        xe = self.nodes[self.quads]  # e x 4 x 2
        pts = np.empty((len(self.quads), 4, 2))
        dndx = np.empty((len(self.quads), 4, 4, 2))
        wts = np.empty((len(self.quads), 4))
        for g, (r, s) in enumerate(_GP):
            n = 0.25 * np.array(
                [(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)]
            )
            dr = 0.25 * np.array([-(1 - s), (1 - s), (1 + s), -(1 + s)])
            ds = 0.25 * np.array([-(1 - r), -(1 + r), (1 + r), (1 - r)])
            pts[:, g] = np.einsum("a,eai->ei", n, xe)
            jac = np.stack([np.einsum("a,eai->ei", dr, xe), np.einsum("a,eai->ei", ds, xe)], axis=1)
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            if np.any(det <= 0):
                raise ValueError("mesh: an inverted element")
            inv = (
                np.stack(
                    [
                        np.stack([jac[:, 1, 1], -jac[:, 0, 1]], -1),
                        np.stack([-jac[:, 1, 0], jac[:, 0, 0]], -1),
                    ],
                    axis=1,
                )
                / det[:, None, None]
            )
            dnat = np.stack([dr, ds], axis=1)  # 4 x 2
            # ∂N/∂x_i = Σ_k (J⁻¹)_ik ∂N/∂ξ_k, J_ki = ∂x_i/∂ξ_k
            dndx[:, g] = np.einsum("ak,eik->eai", dnat, inv)
            wts[:, g] = det
        return pts, dndx, wts

    @property
    def area(self) -> float:
        return float(self.gauss()[2].sum())


@dataclass
class Specimen:
    """A specimen (see the module); lengths in mm."""

    kind: str = "dogbone"
    length: float = 30.0
    width: float = 5.0
    thickness: float = 1.0
    radius: float = 6.0
    grip_width: float = 10.0
    grip_length: float = 10.0
    extensometer: float | None = None
    clamped: bool = True
    element: float = 0.25

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"specimen kind: one of {KINDS}")
        for name in ("length", "width", "thickness", "element"):
            if not getattr(self, name) > 0:
                raise ValueError(f"specimen: {name} must be positive")
        if self.kind == "dogbone":
            if not self.grip_width > self.width:
                raise ValueError("specimen: grip_width > width")
            if not self.radius > 0:
                raise ValueError("specimen: radius must be positive")
        if self.extensometer is None:  # away from the fillets or the clamped ends
            if self.kind == "dogbone":
                self.extensometer = max(self.length - self.width, 0.5 * self.length)
            elif self.kind == "strip" and self.clamped:
                self.extensometer = max(self.length - 2.0 * self.width, 0.5 * self.length)
            else:
                self.extensometer = self.length
        if not 0 < self.extensometer <= self.length + 1e-12:
            raise ValueError("specimen: 0 < extensometer <= length")

    @classmethod
    def from_config(cls, data) -> Specimen:
        if isinstance(data, Specimen):
            return data
        return cls(**dict(data or {}))

    def band(self) -> tuple[float, float, float, float]:
        """The parallel part ``(x0, x1, y0, y1)`` where patterns are drawn."""
        return (0.0, self.length, 0.0, self.width)

    @property
    def fillet(self) -> float:
        """Length along x of each fillet."""
        if self.kind != "dogbone":
            return 0.0
        dw = 0.5 * (self.grip_width - self.width)
        r = self.radius
        return math.sqrt(max(r * r - (r - min(dw, r)) ** 2, 0.0)) + max(dw - r, 0.0)

    def _width_at(self, x):
        """Width of the outline at abscissae ``x``."""
        x = np.asarray(x, dtype=float)
        if self.kind != "dogbone":
            return np.full_like(x, self.width)
        r, dw, lf = self.radius, 0.5 * (self.grip_width - self.width), self.fillet
        d = np.maximum(np.maximum(-x, x - self.length), 0.0)  # distance beyond the parallel part
        if dw <= r:  # a circular arc tangent to the parallel part
            u = np.minimum(d, lf)
            rise = r - np.sqrt(np.maximum(r * r - u * u, 0.0))
        else:  # an arc of a quarter circle then a straight shoulder
            rise = np.where(
                d <= r, r - np.sqrt(np.maximum(r * r - np.minimum(d, r) ** 2, 0.0)), r + (d - r)
            )
        return self.width + 2.0 * np.minimum(rise, dw)

    def extent(self) -> tuple[float, float]:
        """Abscissae of the ends of the specimen."""
        if self.kind != "dogbone":
            return 0.0, self.length
        e = self.fillet + self.grip_length
        return -e, self.length + e

    def mesh(self, element: float | None = None) -> QuadMesh:
        """A structured mesh of the specimen (elements of about ``element`` mm)."""
        h = float(element or self.element)
        x0, x1 = self.extent()
        ny = max(2, math.ceil(self.width / h))
        if self.kind == "dogbone":
            # the parallel part at h, the fillets and grips coarser
            nx_g = max(4, math.ceil(self.length / h))
            xs_g = np.linspace(0.0, self.length, nx_g + 1)
            lf = self.fillet + self.grip_length
            nf = max(3, math.ceil(lf / (1.5 * h)))
            t = np.linspace(0.0, 1.0, nf + 1)[1:] ** 1.4
            xs = np.concatenate([-lf * t[::-1], xs_g, self.length + lf * t])
        else:
            nx = max(2, math.ceil(self.length / h))
            xs = np.linspace(x0, x1, nx + 1)
        w = self._width_at(xs)
        yc = 0.5 * self.width
        frac = np.linspace(-0.5, 0.5, ny + 1)
        X = np.repeat(xs[:, None], ny + 1, axis=1)
        Y = yc + w[:, None] * frac[None, :]
        nodes = np.column_stack([X.ravel(), Y.ravel()])
        nxe, nye = len(xs) - 1, ny
        idx = np.arange(len(xs) * (ny + 1)).reshape(len(xs), ny + 1)
        quads = np.column_stack(
            [idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()]
        )
        return QuadMesh(nodes, quads, idx[0], idx[-1], idx[:, 0], idx[:, -1], (nxe, nye))

    def extensometer_ends(self) -> tuple[float, float]:
        """Abscissae of the ends of the extensometer."""
        c = 0.5 * self.length
        return c - 0.5 * self.extensometer, c + 0.5 * self.extensometer
