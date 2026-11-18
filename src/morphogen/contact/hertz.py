"""Closed forms of the contact of elastic bodies, against which the solver is checked:
Hertz's sphere and cylinder, Sneddon's cone and flat punch, the stick zone of Cattaneo and
Mindlin (Johnson 1985), and the sphere on a layer of finite thickness (Dimitriadis et
al. 2002)."""

from __future__ import annotations

import math


def effective_modulus(young: float, poisson: float, young_other: float | None = None,
                      poisson_other: float = 0.0) -> float:  # fmt: skip
    """The modulus of the contact of two isotropic bodies, ``1/E* = (1 - ν1²)/E1 + (1 -
    ν2²)/E2``; of one body against a rigid one without ``young_other``."""
    inverse = (1.0 - poisson**2) / young
    if young_other is not None:
        inverse += (1.0 - poisson_other**2) / young_other
    return 1.0 / inverse


def hertz_sphere(radius: float, modulus: float, *, force: float | None = None,
                 approach: float | None = None) -> dict:  # fmt: skip
    """Hertz's contact of a sphere of ``radius`` and a plane (or of two spheres: ``1/R =
    1/R1 + 1/R2``) of effective ``modulus`` E*, under a ``force`` or an ``approach`` of the
    two bodies: ``F = 4/3 E* sqrt(R) δ^(3/2)``, the radius of the contact ``a = sqrt(R δ)``
    and the largest pressure ``p0 = 3 F / (2 π a²)``, with the pressure ``p0 sqrt(1 - r² /
    a²)``."""
    if (force is None) == (approach is None):
        raise ValueError("hertz_sphere: a force or an approach")
    if approach is None:
        approach = (3.0 * force / (4.0 * modulus * math.sqrt(radius))) ** (2.0 / 3.0)
    force = 4.0 / 3.0 * modulus * math.sqrt(radius) * approach**1.5
    contact = math.sqrt(radius * approach)
    return {"force": force, "approach": approach, "radius": contact,
            "pressure": 3.0 * force / (2.0 * math.pi * contact**2)}  # fmt: skip


def layer_correction(radius: float, depth: float, thickness: float, poisson: float,
                     bonded: bool = True) -> float:  # fmt: skip
    """The factor by which the rigid base of a layer of ``thickness`` multiplies Hertz's
    force of a sphere of ``radius`` pressed to a ``depth`` into it (Dimitriadis et al.
    2002): with ``χ = sqrt(R δ) / h``, the radius of the contact over the thickness,

    ``1 - (2 α0/π) χ + (4 α0²/π²) χ² - (8/π³)(α0³ + 4π² β0/15) χ³ + (16 α0/π⁴)(α0³ + 3π²
    β0/5) χ⁴``

    (the fourth order with the sign corrected by Garcia and Garcia 2018), where, for a
    layer ``bonded`` to its base, ``α0 = -(1.2876 - 1.4678 ν + 1.3442 ν²) / (1 - ν)`` and
    ``β0 = (0.6387 - 1.0277 ν + 1.5164 ν²) / (1 - ν)``, and for one free to slide on it,
    ``α0 = -0.347 (3 - 2 ν) / (1 - ν)`` and ``β0 = 0.056 (5 - 2 ν) / (1 - ν)``. A layer
    wide against its thickness, a contact smaller than it (``χ`` below 1)."""
    if not (radius > 0 and depth >= 0 and thickness > 0):
        raise ValueError("layer_correction: a radius, a depth and a thickness")
    if not -1.0 < poisson <= 0.5:
        raise ValueError("layer_correction: a Poisson's ratio in (-1, 0.5]")
    chi = math.sqrt(radius * depth) / thickness
    if bonded:
        alpha = -(1.2876 - 1.4678 * poisson + 1.3442 * poisson**2) / (1.0 - poisson)
        beta = (0.6387 - 1.0277 * poisson + 1.5164 * poisson**2) / (1.0 - poisson)
    else:
        alpha = -0.347 * (3.0 - 2.0 * poisson) / (1.0 - poisson)
        beta = 0.056 * (5.0 - 2.0 * poisson) / (1.0 - poisson)
    pi = math.pi
    return (
        1.0
        - 2.0 * alpha / pi * chi
        + 4.0 * alpha**2 / pi**2 * chi**2
        - 8.0 / pi**3 * (alpha**3 + 4.0 * pi**2 * beta / 15.0) * chi**3
        + 16.0 * alpha / pi**4 * (alpha**3 + 3.0 * pi**2 * beta / 5.0) * chi**4
    )


def hertz_cylinder(radius: float, modulus: float, force: float) -> dict:
    """Hertz's contact of a cylinder of ``radius`` and a plane along a line, under a
    ``force`` per unit length: the half-width of the contact ``b = sqrt(4 F R / (π E*))``
    and the largest pressure ``p0 = 2 F / (π b)``, with the pressure ``p0 sqrt(1 - x² /
    b²)``."""
    width = math.sqrt(4.0 * force * radius / (math.pi * modulus))
    return {"force": force, "half_width": width, "pressure": 2.0 * force / (math.pi * width)}


def sneddon_cone(half_angle: float, modulus: float, depth: float) -> dict:
    """Sneddon's indentation of a half-space of effective ``modulus`` by a rigid cone of
    ``half_angle`` (degrees, from its axis) to a ``depth``: ``F = 2/π E* tan(α) h²`` and
    the radius of the contact ``a = 2/π h tan(α)``."""
    tangent = math.tan(math.radians(half_angle))
    return {"force": 2.0 / math.pi * modulus * tangent * depth**2, "depth": depth,
            "radius": 2.0 / math.pi * depth * tangent}  # fmt: skip


def flat_punch(radius: float, modulus: float, depth: float) -> dict:
    """The indentation of a half-space by a rigid flat punch of ``radius`` to a ``depth``:
    ``F = 2 a E* h``."""
    return {"force": 2.0 * radius * modulus * depth, "depth": depth, "radius": radius}


def mindlin_stick_radius(radius: float, friction: float, normal: float, tangential: float):
    """The radius of the stick zone of a Hertz contact of radius ``radius`` under a
    tangential force below ``μ N`` (Cattaneo 1938; Mindlin 1949): ``c = a (1 - T / (μ
    N))^(1/3)``, a ring of slip around it."""
    ratio = tangential / (friction * normal)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("mindlin_stick_radius: a tangential force of at most μ N")
    return radius * (1.0 - ratio) ** (1.0 / 3.0)
