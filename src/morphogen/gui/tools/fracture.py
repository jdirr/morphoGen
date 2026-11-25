"""The tools of the tab Fracture (:mod:`morphogen.fracture`): the stress intensity factors
of a crack in a plate, a block or a morphology of the workspace (the G-θ method and the
interaction integrals, in small and finite strains, and the J of inelastic bulks), its
propagation on meshes rebuilt by MMG at each advance, in a plate or through a morphology
(kinked after Erdogan & Sih, at a fixed load with Paris' law, or once its driving force
reaches a toughness: G_c, the J_tip of an inelastic bulk, or the porosity of GTN points at
the tip; through a morphology, the toughness of each phase and of the interfaces between
them deflects it along weak interfaces and holds it back at tough phases), cohesive zones
(a ligament ahead of a notch, the interfaces of the phases or the
grain boundaries of a morphology) and the growth of a crack through the grains of a
polycrystal of crystal plasticity.

The specimens are boxes ``[lower, upper]`` (a plate of 16 x 32 x 1 by default, the domain
of a morphology, or a slab of it about the middle of z for the propagation), loaded along y:
tension on the faces y = 0 and y = H, in-plane shear on
the four lateral faces, antiplane shear along z, by tractions (the rigid motions held) or by
grips (the faces y = 0 and H clamped and moved). Straight cracks run from the face x = 0
(their front along z, through the thickness), at the height ``position``, inclined by
``angle`` towards y; a penny crack is centred in the box, its normal along y (turned by the
same angle). The crack meshes of a morphology carry its phases (each with the material of
the form, E and ν, or void: the pores, left out of the meshes), conforming to their
interfaces or as staircases, and the sides of the crack apart."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from morphogen.gui.tools import Field, tab, tool

tab(
    "fracture",
    "Fracture",
    "Cracks and cohesive zones: the stress intensity factors of a crack (G-θ, interaction "
    "integrals, J of inelastic bulks), its propagation on meshes rebuilt at each advance "
    "(kinking, Paris' law, toughness, ductile growth), cohesive interfaces (force-displacement, "
    "dissipated energies, broken interfaces) and crack growth through a polycrystal.",
    order=50,
)

#: The elements of the meshes, at most (an estimate of 16 V / h³ for MMG)
MAX_ELEMENTS = 300_000
#: The default element size keeps the meshes below this many elements
DEFAULT_ELEMENTS = 50_000
#: The materials of the phases of a morphology when the form gives none (a contrast of 3)
DEFAULT_YOUNG, DEFAULT_POISSON, DEFAULT_CONTRAST = 1000.0, 0.3, 3.0

LOADINGS = (
    ("tension", "tension along y (mode I)"),
    ("shear", "in-plane shear xy (mode II)"),
    ("mixed", "tension and shear (modes I and II)"),
    ("antiplane", "antiplane shear yz (mode III)"),
)
CONTROLS = (
    ("tractions", "tractions on the faces (the rigid motions held)"),
    ("grips", "grips: the faces y = 0 and y = H clamped and moved"),
)
BEHAVIOURS = (
    ("elastic", "elastic (neo-Hookean in finite strains)"),
    ("plastic", "von Mises elasto-(visco)plasticity"),
    ("porous", "porous plasticity (Gurson-Tvergaard-Needleman)"),
    ("custom", "a material as JSON (plasticity, porous_plasticity, mfront...)"),
    ("workspace", "a behaviour of the workspace (B: a material point, a Z-set file)"),
)
KINEMATICS = (("small", "small strains"), ("finite", "finite strains"))


# -- the fields shared by the tools ------------------------------------------------------------


def _behaviour_fields(when: dict | None = None) -> list[Field]:
    """The fields of the isotropic material of a specimen and of its behaviour (``when``:
    the conditions of them all; the page reads the hidden fields too, so that the fields of
    a behaviour repeat them)."""
    isotropic = {**(when or {}), "behaviour": ["elastic", "plastic", "porous"]}
    inelastic = {**(when or {}), "behaviour": ["plastic", "porous"]}
    porous = {**(when or {}), "behaviour": "porous"}
    custom = {**(when or {}), "behaviour": "custom"}
    kept = {**(when or {}), "behaviour": "workspace"}
    return [
        Field("behaviour", "choice", "elastic", choices=BEHAVIOURS, when=when,
              doc="the behaviour of the points of the bulk (morphogen.fracture.bulk)"),
        Field("kept", "item", label="behaviour of the workspace", items=("behaviour",),
              required=True, when=kept, doc="with its own elasticity"),
        Field("young", "float", DEFAULT_YOUNG, label="Young's modulus E", unit="stress",
              when=isotropic),
        Field("poisson", "float", DEFAULT_POISSON, label="Poisson's ratio ν", when=isotropic),
        Field("yield_stress", "float", label="yield stress R0", unit="stress", when=inelastic,
              doc="blank: E / 1000"),
        Field("hardening", "float", label="linear hardening H", unit="stress", when=inelastic,
              doc="blank: E / 20"),
        Field("viscosity", "float", label="Norton's K", unit="stress", when=inelastic,
              doc="viscoplasticity, the load factor as the time (blank: rate-independent)"),
        Field("exponent", "float", 5.0, label="Norton's n", when=inelastic),
        Field("porosity", "float", 0.01, label="initial porosity f0", when=porous),
        Field("material", "json", label="material (JSON)", when=custom,
              doc='the keys of a Material: young, poisson (or stiffness), plasticity, '
              'porous_plasticity, crystal_plasticity, mfront; e.g. {"young": 1000, "poisson": '
              '0.3, "plasticity": {"yield": 1, "hardening": 50, "isotropic": [{"Q": 5, "b": '
              '20}]}}'),
    ]  # fmt: skip


def _plate_fields(when: dict | None = None, penny: bool = True) -> list[Field]:
    blanks = (" (blank: 16, or 4 for a penny)", " (blank: 32, or 4)", " (blank: 1, or 4)")
    blanks = blanks if penny else ("", "", "")
    return [
        Field("width", "float", None if penny else 16.0, unit="length", when=when,
              doc="along x" + blanks[0]),
        Field("height", "float", None if penny else 32.0, unit="length", when=when,
              doc="along y, the direction of the load" + blanks[1]),
        Field("thickness", "float", None if penny else 1.0, unit="length", when=when,
              doc="along z, the front of a straight crack" + blanks[2]),
    ]  # fmt: skip


def _crack_fields(kinds) -> list[Field]:
    choices = {
        "edge": ("edge", "edge: from the face x = 0, its front through the thickness"),
        "through": ("through", "through: a centre crack of 2a in a plate of 2W, by its half "
                    "(the plane x = 0 of symmetry; tension)"),
        "penny": ("penny", "penny: a circular crack inside a block"),
    }  # fmt: skip
    return [
        Field("crack", "choice", kinds[0], choices=tuple(choices[k] for k in kinds)),
        Field("length", "float", unit="length",
              doc="of a straight crack, or the radius of a penny (blank: 0.28 W for an edge "
              "crack, 0.25 W through, a quarter of the smallest edge for a penny)"),
        Field("position", "float", unit="length", doc="the height y of the crack (blank: the "
              "middle)"),
        Field("angle", "float", 0.0, unit="degree",
              doc="the plane of the crack turned about z, from x towards y"),
    ]  # fmt: skip


def _loading_fields(loadings=LOADINGS) -> list[Field]:
    return [
        Field("loading", "choice", "tension", choices=loadings),
        Field("shear_ratio", "float", 0.7, label="shear / tension", when={"loading": "mixed"}),
        Field("control", "choice", "tractions", choices=CONTROLS),
        Field("load", "float", label="load",
              doc="the traction (tractions) or the displacement of each grip (grips) at the "
              "load factor 1 (blank: a nominal strain of 0.1 %: 10⁻³ E, or 5·10⁻⁴ H)"),
        Field("plane_strain", "bool", True, label="plane strain",
              doc="the faces z = 0 and z = T held along z (straight cracks, in-plane loads: the "
              "auxiliary fields of the interaction integrals are those of plane strain); "
              "otherwise the face z = 0 only (a mirror plane)"),
        Field("kinematics", "choice", "small", choices=KINEMATICS,
              doc="finite strains: total Lagrangian, the modes as estimates (turned back by "
              "the rotation of the mean deformation gradient)"),
        Field("element_size", "float", unit="length",
              doc="the largest edge of the tetrahedra meshed by MMG (blank: a / 4.5 for a "
              "straight crack, a / 3 for a penny, a / 9 for the growth through a morphology; "
              "at most about 50 000 elements)"),
    ]  # fmt: skip


# -- materials -----------------------------------------------------------------------------------


def _material(p: dict):
    """The isotropic material of a specimen with its behaviour (the fields of
    :func:`_behaviour_fields`)."""
    from morphogen.homogenization.materials import Material
    from morphogen.homogenization.porous import PorousPlasticity

    kind = p.get("behaviour") or "elastic"
    if kind == "workspace":
        return Material.from_config(p["kept"]["material"])
    if kind == "custom":
        spec = p.get("material")
        if not isinstance(spec, dict) or not spec:
            raise ValueError('material: a JSON object, e.g. {"young": 1000, "poisson": 0.3, '
                             '"plasticity": {"yield": 1, "hardening": 50}}')  # fmt: skip
        try:
            return Material(**spec)
        except TypeError as exc:
            raise ValueError(f"material: {exc}") from None
    young, poisson = float(p["young"]), float(p["poisson"])
    if kind == "elastic":
        return Material(young=young, poisson=poisson)
    r0 = p["yield_stress"] if p.get("yield_stress") is not None else young / 1000.0
    h = p["hardening"] if p.get("hardening") is not None else young / 20.0
    k, n = p.get("viscosity"), p.get("exponent") or 5.0
    if kind == "plastic":
        plasticity: dict = {"yield": r0, "hardening": h}
        if k:
            plasticity["norton"] = {"K": float(k), "n": float(n)}
        return Material(young=young, poisson=poisson, plasticity=plasticity)
    if kind == "porous":
        gtn = PorousPlasticity("gurson", yield_stress=r0, porosity=float(p["porosity"] or 0.0),
                               hardening=h, viscosity=float(k) if k else None,
                               exponent=float(n) if k else None)  # fmt: skip
        return Material(young=young, poisson=poisson, porous_plasticity=gtn)
    raise ValueError(f"behaviour: one of {', '.join(b[0] for b in BEHAVIOURS)}")


def _phase_materials(item: dict, given, present=()) -> dict:
    """The isotropic materials of the phases of a morphology and of those ``present`` in its
    image (``given``: those of the form, by phase, or ``"void"``: pores, whose elements the
    meshes leave out; the others E = 1000 for the phase 0 and 3000 for the others, ν =
    0.3)."""
    from morphogen.homogenization.materials import Material

    phases = {int(k) for k in (item.get("phases") or {"0": "", "1": ""})}
    phases = sorted(phases | {int(k) for k in present})
    out = {}
    for k in phases:
        spec = (given or {}).get(k)
        if spec == "void":
            out[k] = Material(void=True)
            continue
        if spec is None:
            young = DEFAULT_YOUNG * (1.0 if k == 0 else DEFAULT_CONTRAST)
            spec = {"young": young, "poisson": DEFAULT_POISSON}
        out[k] = Material(young=float(spec["young"]), poisson=float(spec["poisson"]))
    present = {int(k) for k in present} or set(out)
    if all(out[k].void for k in present if k in out):
        raise ValueError("every phase of the morphology is void")
    return out


def _check_solid(data, voids, slab: bool = False) -> None:
    """That the phases of a phase image other than ``voids`` make one body (parts below 1 %
    of the largest aside, as :func:`~morphogen.fracture.lefm.without_voids` drops them;
    ``slab``: the image is that of a slab of the morphology)."""
    if not len(voids):
        return
    from scipy import ndimage

    labels, count = ndimage.label(~np.isin(data, list(voids)))
    if count > 1:
        sizes = np.bincount(labels.ravel())[1:]
        parts = int(np.count_nonzero(sizes >= 0.01 * sizes.max()))
        if parts > 1:
            hint = "; a thicker slab may hold it together" if slab else ""
            raise ValueError(f"void phases {list(voids)}: the body left falls apart into {parts} "
                             f"parts (pores are phases enclosed by the others{hint})")  # fmt: skip


def _stiffest(materials) -> float:
    """The largest Young's modulus of materials (a crystal: its largest normal stiffness)."""
    from morphogen.fracture.mixed import _stiffest

    return max(_stiffest(m) for m in materials if not m.void)


def _phase_lookup(micro, spacing: float, lower=None, upper=None):
    """The phase of a morphology at points (``phase_of(points)``), from a voxel image of
    it (or of the box ``[lower, upper]`` in its domain) whose voxels are about ``spacing``
    (at most 256 per edge), the phases of the image and the image (its phases)."""
    from morphogen.domain import Domain, Grid

    lengths = np.asarray(micro.domain.lengths, dtype=float)
    spacing = max(float(spacing), float(lengths.max()) / 256.0)
    whole = lower is None or (np.allclose(lower, micro.domain.lower)
                              and np.allclose(upper, micro.domain.upper))  # fmt: skip
    if whole:
        image = micro.voxelize(voxel_size=spacing)
    else:
        box = Domain(size=tuple(np.asarray(upper, dtype=float) - lower), origin=tuple(lower),
                     periodic=False)  # fmt: skip
        image = micro.voxelize(grid=Grid.for_domain(box, voxel_size=spacing))
    data = np.asarray(image.data)
    if data.dtype.kind not in "iub":
        raise ValueError("the morphology gives no phase image")
    origin = np.asarray(image.origin, dtype=float)
    steps = np.asarray(image.spacing, dtype=float)
    shape = np.asarray(data.shape)

    def phase_of(points):
        index = np.floor((np.asarray(points, dtype=float) - origin) / steps).astype(np.int64)
        index = np.clip(index, 0, shape - 1)
        return data[index[:, 0], index[:, 1], index[:, 2]].astype(np.int64)

    return phase_of, np.unique(data), data


def _load_value(p: dict, lower, upper, young: float) -> float:
    """The load of the form at the load factor 1: the traction or the displacement of each
    grip (blank: a nominal strain of 0.1 %)."""
    if p.get("load") is not None:
        return float(p["load"])
    if p["control"] == "tractions":
        return 1e-3 * young
    return 5e-4 * float(upper[1] - lower[1])


# -- specimens -----------------------------------------------------------------------------------


@dataclass
class Specimen:
    """A cracked specimen: its box, its crack, the materials of the phases of its crack
    meshes (``mesh_of(crack)``: MMG conforming to the crack; those of a morphology carry
    its phases and the sides of the crack apart, ``mesh.side``, its pores left out), the
    sides of the crack of the elements of a mesh (``sides(mesh)``, None: those of the mesh)
    and its loading."""

    lower: np.ndarray
    upper: np.ndarray
    crack: Any
    kind: str
    materials: dict
    mesh_of: Any
    sides: Any
    prescribed: Any
    forces: Any
    size: float
    young: float
    names: dict
    morphology: bool = False
    handbook: dict | None = None
    voids: tuple = ()

    @property
    def interfaces(self) -> str:
        """The kind of the interfaces of the phases of a morphology on its crack meshes."""
        return getattr(self.mesh_of, "kind", "")

    def solve(self, crack=None, mesh=None, materials=None, **options):
        """The :class:`~morphogen.fracture.mixed.CrackSolution` of a crack of the specimen
        (on its crack mesh, or on ``mesh``; of its materials, or of ``materials``) at the
        load factor 1."""
        from morphogen.fracture.mixed import solve_crack

        crack = self.crack if crack is None else crack
        mesh = self.mesh_of(crack) if mesh is None else mesh
        sides = None if self.sides is None else self.sides(mesh)
        materials = self.materials if materials is None else materials
        return solve_crack(mesh, crack, materials, self.prescribed, 1.0, forces=self.forces,
                           sides=sides, **options)  # fmt: skip

    def labels(self, mesh) -> np.ndarray:
        """The phase of the bulk of each element of a crack mesh (0 for a homogeneous one)."""
        if self.morphology:
            return np.asarray(mesh.phase)
        return np.zeros(mesh.n_elements, dtype=np.int64)

    def solids(self) -> list:
        """The materials of the phases that are not void."""
        return [m for m in self.materials.values() if not m.void]


def _size(value, crack_size: float, kind: str, volume: float) -> float:
    """The element size of the form, or the default (see :func:`_loading_fields`); checked
    against :data:`MAX_ELEMENTS`."""
    if value is None:
        size = crack_size / (3.0 if kind == "penny" else 4.5)
        size = max(size, (16.0 * volume / DEFAULT_ELEMENTS) ** (1.0 / 3.0))
    else:
        size = float(value)
    if not size > 0:
        raise ValueError("element size: positive")
    count = 16.0 * volume / size**3
    if count > MAX_ELEMENTS:
        raise ValueError(
            f"element size {size:g}: about {count:,.0f} elements, more than {MAX_ELEMENTS:,} "
            "in the interface (a larger element size, or a smaller specimen)"
        )
    return size


def _crack_of(p: dict, lower, upper):
    """The crack of the form in the box: a :class:`~morphogen.fracture.mixed.KinkedCrack`
    (straight, from the face x = lower) or a penny :class:`~morphogen.fracture.lefm.Crack`."""
    from morphogen.fracture.lefm import Crack
    from morphogen.fracture.mixed import KinkedCrack

    kind = p["crack"]
    lengths = upper - lower
    angle = math.radians(p.get("angle") or 0.0)
    normal = np.array([-math.sin(angle), math.cos(angle), 0.0])
    direction = np.array([math.cos(angle), math.sin(angle), 0.0])
    y = lower[1] + 0.5 * lengths[1] if p.get("position") is None else float(p["position"])
    if not lower[1] < y < upper[1]:
        raise ValueError(f"position: the height of the crack inside the specimen "
                         f"({lower[1]:g} to {upper[1]:g})")  # fmt: skip
    if kind == "penny":
        a = p["length"] if p.get("length") is not None else 0.25 * float(lengths.min())
        centre = np.array([lower[0] + 0.5 * lengths[0], y, lower[2] + 0.5 * lengths[2]])
        reach = np.min(np.minimum(centre - lower, upper - centre))
        if not 0 < a < reach:
            raise ValueError(f"penny: a radius below {reach:g} (the crack inside the specimen)")
        return Crack("penny", tuple(centre), tuple(normal), float(a))
    default = 0.25 if kind == "through" else 0.28
    a = p["length"] if p.get("length") is not None else default * float(lengths[0])
    if kind == "through" and abs(angle) > 1e-12:
        raise ValueError("through: a centre crack by its symmetric half, normal to y (angle 0)")
    mouth = np.array([lower[0], y, lower[2]])
    tip = mouth + float(a) * direction
    if not (a > 0 and np.all(tip[:2] > lower[:2]) and np.all(tip[:2] < upper[:2])):
        raise ValueError("length: the front of the crack inside the specimen")
    return KinkedCrack((tuple(mouth), tuple(tip)))


def _nearest(mesh, point) -> int:
    return int(np.argmin(np.linalg.norm(mesh.nodes - np.asarray(point, dtype=float), axis=1)))


def _loads(p: dict, kind: str, lower, upper, young: float):
    """The prescribed displacements ``prescribed(mesh)`` and the nodal forces ``forces(mesh)``
    (None for grips) of the loading of the form at the load factor 1 (see the module)."""
    from morphogen.fracture.lefm import face_forces

    loading, control = p["loading"], p["control"]
    lo, up = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    tol = 1e-9 * float(np.max(up - lo))
    mid = 0.5 * (lo + up)
    straight = kind != "penny"
    if kind == "through" and loading != "tension":
        raise ValueError("through: a centre crack by its symmetric half, in tension")
    plane_strain = bool(p.get("plane_strain")) and straight and loading != "antiplane"
    value = _load_value(p, lo, up, young)
    ratio = float(p.get("shear_ratio") or 0.0) if loading == "mixed" else 0.0
    normal = value if loading in ("tension", "mixed") else 0.0
    shear = value if loading == "shear" else ratio * value
    anti = value if loading == "antiplane" else 0.0

    def faces(m):
        x = m.nodes
        return {"top": x[:, 1] > up[1] - tol, "bottom": x[:, 1] < lo[1] + tol,
                "left": x[:, 0] < lo[0] + tol, "z0": x[:, 2] < lo[2] + tol,
                "z1": x[:, 2] > up[2] - tol}  # fmt: skip

    if control == "grips":
        top = np.array([shear, normal, anti])

        def prescribed(m):
            f = faces(m)
            grips = f["top"] | f["bottom"]
            out = [(np.flatnonzero(f["top"]), [0, 1, 2], top),
                   (np.flatnonzero(f["bottom"]), [0, 1, 2], -top)]  # fmt: skip
            if anti and straight:  # pure antiplane shear: the in-plane motion held
                out.append((np.flatnonzero(~grips), [0, 1], 0.0))
            elif plane_strain:
                out.append((np.flatnonzero((f["z0"] | f["z1"]) & ~grips), 2, 0.0))
            if kind == "through":  # the plane of symmetry
                out.append((np.flatnonzero(f["left"] & ~grips), 0, 0.0))
            return out

        return prescribed, None

    def forces(m):
        def on(axis, side):
            if side > 0:
                return lambda c: c[:, axis] > up[axis] - tol
            return lambda c: c[:, axis] < lo[axis] + tol

        f = np.zeros(3 * m.n_nodes)
        if normal:
            f += face_forces(m, on(1, 1), [0.0, normal, 0.0])
            f += face_forces(m, on(1, -1), [0.0, -normal, 0.0])
        if shear:  # uniform in-plane shear: the four lateral faces
            f += face_forces(m, on(1, 1), [shear, 0.0, 0.0])
            f += face_forces(m, on(1, -1), [-shear, 0.0, 0.0])
            f += face_forces(m, on(0, 1), [0.0, shear, 0.0])
            f += face_forces(m, on(0, -1), [0.0, -shear, 0.0])
        if anti:
            f += face_forces(m, on(1, 1), [0.0, 0.0, anti])
            f += face_forces(m, on(1, -1), [0.0, 0.0, -anti])
            if not straight:  # uniform antiplane shear of a block: the faces z too
                f += face_forces(m, on(2, 1), [0.0, anti, 0.0])
                f += face_forces(m, on(2, -1), [0.0, -anti, 0.0])
        return f

    def prescribed(m):
        f = faces(m)
        if not straight:  # a block: three corners hold the rigid motions
            a = _nearest(m, lo)
            b = _nearest(m, (up[0], lo[1], lo[2]))
            c = _nearest(m, (lo[0], up[1], lo[2]))
            return [(a, [0, 1, 2], 0.0), (b, [1, 2], 0.0), (c, 2, 0.0)]
        if anti:  # pure antiplane shear (the in-plane motion held), as the strip solution
            corner = _nearest(m, (up[0], lo[1], lo[2]))
            return [(np.arange(m.n_nodes), [0, 1], 0.0), (corner, 2, 0.0)]
        faces_z = f["z0"] | f["z1"] if plane_strain else f["z0"]
        a = _nearest(m, (up[0], mid[1], lo[2]))
        out = [(np.flatnonzero(faces_z), 2, 0.0)]
        if kind == "through":  # the plane of symmetry, and one node along y
            return [*out, (np.flatnonzero(f["left"]), 0, 0.0), (a, 1, 0.0)]
        b = _nearest(m, (up[0], up[1], lo[2]))
        return [*out, (a, [0, 1], 0.0), (b, 0, 0.0)]

    return prescribed, forces


def _handbook(p: dict, kind: str, crack, lower, upper, young: float) -> dict | None:
    """The handbook factor of the specimen when it has one: tension by tractions of an edge
    crack (Tada, Paris & Irwin), of a centre crack (Tada) and of a penny (Sneddon, in an
    infinite body), antiplane shear of an edge crack (the strip); straight cracks normal to
    y, small strains."""
    from morphogen.fracture.lefm import sent_factor

    if p["control"] != "tractions" or p.get("kinematics") == "finite" or (p.get("angle") or 0):
        return None
    value = _load_value(p, lower, upper, young)
    width = float(upper[0] - lower[0])
    a = float(crack.size)
    if p["loading"] == "tension":
        if kind == "edge":
            factor = value * math.sqrt(math.pi * a) * sent_factor(a / width)
            return {"mode": "K_I", "value": factor, "source": "Tada, Paris & Irwin (SENT)"}
        if kind == "through":
            r = a / width
            shape = (1 - 0.025 * r**2 + 0.06 * r**4) * math.sqrt(1.0 / math.cos(0.5 * math.pi * r))
            return {"mode": "K_I", "value": value * math.sqrt(math.pi * a) * shape,
                    "source": "Tada (centre crack in a long strip)"}  # fmt: skip
        return {"mode": "K_I", "value": 2.0 * value * math.sqrt(a / math.pi),
                "source": "Sneddon (infinite body)"}  # fmt: skip
    if p["loading"] == "antiplane" and kind == "edge":
        strip = math.sqrt(2 * width / (math.pi * a) * math.tan(math.pi * a / (2 * width)))
        return {"mode": "K_III", "value": value * math.sqrt(math.pi * a) * strip,
                "source": "antiplane strip"}  # fmt: skip
    return None


INTERFACES = (
    ("conforming", "conforming: the crack meshes follow the interfaces of the phases (MMG "
     "cuts the mesh of the phases by each crack; a level set needed)"),
    ("staircase", "staircase: each element of the crack meshes the phase of its centroid"),
)  # fmt: skip


class _MorphologyMeshes:
    """The crack meshes of a morphology (``self(crack)``: :func:`~morphogen.fracture.lefm.
    crack_mesh`), its void phases left out, and the ``kind`` of their interfaces: conforming
    to its phases (the mesh of its phases made at the first crack, then cut by each one), or
    staircases (each element the phase ``phase_of`` its centroid); a morphology without a
    level set, or of more than two phases, has staircases."""

    def __init__(self, micro, lower, upper, size: float, phase_of, present, voids, kind: str):
        self.micro, self.lower, self.upper, self.size = micro, lower, upper, float(size)
        self.phase_of, self.voids, self.phases = phase_of, tuple(voids), None
        self.kind = kind
        if kind == "conforming" and not {int(k) for k in present} <= {0, 1}:
            self._staircase("more than two phases")

    def _staircase(self, why: str) -> None:
        import warnings

        warnings.warn(f"{why}: staircase interfaces on the crack meshes", stacklevel=3)
        self.kind = "staircase"

    def __call__(self, crack):
        from morphogen.fracture.lefm import crack_mesh, phase_mesh

        if self.kind == "conforming" and self.phases is None:
            try:
                self.phases = phase_mesh(self.micro, self.lower, self.upper, self.size)
            except ValueError as exc:  # no level set
                if "level-set" not in str(exc):
                    raise
                self._staircase("the morphology has no level set")
        if self.kind == "conforming":  # the interfaces followed coarsely (h / 5, h / 3)
            return crack_mesh(self.lower, self.upper, crack, size=self.size,
                              morphology=self.phases, voids=self.voids,
                              hausdorff=self.size / 5.0, min_size=self.size / 3.0)  # fmt: skip
        return crack_mesh(self.lower, self.upper, crack, size=self.size, morphology=self.phase_of,
                          voids=self.voids)  # fmt: skip


def specimen(p: dict, *, morphology_bulk: bool = True, slab: bool = False) -> Specimen:
    """The cracked specimen of the form of the tools Stress intensity factors and Crack
    propagation: a homogeneous box of the material of :func:`_material`, or the domain of a
    morphology whose phases the crack meshes carry (their materials; its void phases, the
    pores, left out): conforming to their interfaces, or as staircases (each element the
    phase of its centroid). ``slab``: of a morphology, a slab about the middle of z (the
    field ``slab``, blank: the whole depth, at most a sixteenth of the width), in plane
    strain the path of a growth that of its section, meshed by elements of a / 9."""
    from morphogen.fracture.lefm import crack_mesh

    kind = p["crack"]
    morphology = morphology_bulk and p.get("bulk") == "morphology"
    if morphology:
        item = p.get("source")
        if item is None:
            raise ValueError("morphology: choose a morphology of the workspace")
        micro = item["micro"]
        lower = np.asarray(micro.domain.lower, dtype=float).copy()
        upper = np.asarray(micro.domain.upper, dtype=float).copy()
        if kind == "through":
            raise ValueError("through: a centre crack by the symmetry of a homogeneous plate")
        names = {int(k): v for k, v in (item.get("phases") or {}).items()}
        if slab:
            depth = float(upper[2] - lower[2])
            thickness = p.get("slab") or min(depth, float(upper[0] - lower[0]) / 16.0)
            if not 0 < thickness <= depth * (1 + 1e-12):
                raise ValueError(f"slab: from 0 to the depth of the morphology ({depth:g})")
            middle = 0.5 * (lower[2] + upper[2])
            lower[2], upper[2] = middle - 0.5 * thickness, middle + 0.5 * thickness
    else:
        penny = kind == "penny"
        dims = [p.get("width") or (4.0 if penny else 16.0),
                p.get("height") or (4.0 if penny else 32.0),
                p.get("thickness") or (4.0 if penny else 1.0)]  # fmt: skip
        if min(dims) <= 0:
            raise ValueError("width, height, thickness: positive")
        lower, upper = np.zeros(3), np.asarray(dims, dtype=float)
        names = {0: "bulk"}
    crack = _crack_of(p, lower, upper)
    scale = 2.0 if morphology and slab else 1.0  # the growth through a morphology: a / 9
    size = _size(p.get("element_size"), crack.size / scale, kind, float(np.prod(upper - lower)))
    voids = ()
    if morphology:
        phase_of, present, image = _phase_lookup(micro, 0.5 * size, lower, upper)
        materials = _phase_materials(item, p.get("materials"), present)
        if p.get("phase_kept"):  # a behaviour of the workspace for one phase
            from morphogen.homogenization.materials import Material

            phase = p.get("kept_phase")
            if phase not in materials or materials[phase].void:
                solid = [k for k, m in materials.items() if not m.void]
                raise ValueError(f"its phase: one of the solid phases {solid}")
            materials[phase] = Material.from_config(p["phase_kept"]["material"])
        voids = tuple(k for k, m in materials.items() if m.void)
        _check_solid(image, voids, slab and upper[2] - lower[2] < micro.domain.lengths[2])
        front, _ = crack.front(lower, upper, 32)
        if voids and len(front) and np.mean(np.isin(phase_of(front), voids)) > 0.5:
            raise ValueError("the front of the crack lies in a pore (a void phase): another "
                             "length or position")  # fmt: skip
        mesh_of = _MorphologyMeshes(micro, lower, upper, size, phase_of, present, voids,
                                    p.get("interfaces") or "staircase")  # fmt: skip
    else:
        material = _material(p)

        def mesh_of(c):
            return crack_mesh(lower, upper, c, size=size)

        materials = {0: material, 1: material}
    young = _stiffest(materials.values())
    prescribed, forces = _loads(p, kind, lower, upper, young)
    handbook = None if morphology else _handbook(p, kind, crack, lower, upper, young)
    return Specimen(lower, upper, crack, kind, materials, mesh_of, None, prescribed, forces,
                    size, young, names, morphology, handbook, voids)  # fmt: skip


# -- views -----------------------------------------------------------------------------------------


def section_cut(mesh, axis: int = 2, at=None):
    """The cut of a tetrahedral mesh by the plane ``x[axis] = at`` (by default the middle of
    the mesh): for the four corners of each polygon (a triangle repeats its last one), the
    two nodes of the edge it lies on and its position along it, and the element of each
    polygon."""
    x = mesh.nodes
    tets = mesh.elements[:, :4]
    if at is None:
        at = 0.5 * (x[:, axis].min() + x[:, axis].max())
    d = x[tets, axis] - float(at)
    positive = d > 0
    count = positive.sum(axis=1)
    order = np.argsort(~positive, axis=1, kind="stable")  # the positive nodes first
    nodes = np.take_along_axis(tets, order, axis=1)
    dist = np.take_along_axis(d, order, axis=1)
    starts, ends, where, owners = [], [], [], []
    for n, edges in ((1, ((0, 1), (0, 2), (0, 3), (0, 3))), (3, ((3, 0), (3, 1), (3, 2), (3, 2))),
                     (2, ((0, 2), (0, 3), (1, 3), (1, 2)))):  # fmt: skip
        rows = np.flatnonzero(count == n)
        if not len(rows):
            continue
        i, j = (np.array(k) for k in zip(*edges, strict=True))
        di, dj = dist[rows][:, i], dist[rows][:, j]
        starts.append(nodes[rows][:, i])
        ends.append(nodes[rows][:, j])
        where.append(di / (di - dj))
        owners.append(rows)
    if not owners:
        empty = np.zeros((0, 4), dtype=np.int64)
        return empty, empty, np.zeros((0, 4)), np.zeros(0, dtype=np.int64), axis
    return (np.concatenate(starts), np.concatenate(ends), np.concatenate(where),
            np.concatenate(owners), axis)  # fmt: skip


def section(mesh, displacement=None, scale: float = 0.0, axis: int = 2, at=None, cut=None):
    """The section of a tetrahedral mesh by the plane ``x[axis] = at`` (by default the middle
    of the mesh; ``cut``: that of :func:`section_cut`), in the configuration ``x + scale u``:
    its polygons (cells x 4 x 2, in the two other coordinates) and the element of each. The
    lips of an opened crack, whose nodes are duplicated, open with ``u``."""
    a, b, t, owners, axis = section_cut(mesh, axis, at) if cut is None else cut
    x = mesh.nodes
    moved = x if displacement is None else x + scale * np.asarray(displacement).reshape(-1, 3)
    keep = [i for i in range(3) if i != axis]
    points = moved[a] + t[..., None] * (moved[b] - moved[a])
    return points[..., keep], owners


def _canvas(width: float = 6.4, height: float = 4.8, layout: str | None = "constrained"):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), dpi=100, layout=layout)
    FigureCanvasAgg(fig)
    return fig


class _Frames:
    """The frames of a video drawn on one figure of fixed axes, whose cells (a section of a
    mesh coloured by a field), lines and black outlines (the interfaces of phases) are
    updated from frame to frame."""

    def __init__(self, window, clim, label: str, *, cmap="viridis", lines=None, width=5.6):
        from matplotlib.collections import LineCollection, PolyCollection

        aspect = (window[1][1] - window[1][0]) / max(window[0][1] - window[0][0], 1e-300)
        height = float(np.clip((width - 1.4) * aspect + 1.0, 3.0, 8.0))
        self.fig = _canvas(width, height, layout=None)
        self.ax = self.fig.add_axes((0.11, 0.09, 0.66, 0.83))
        self.cells = PolyCollection([], cmap=cmap, edgecolors="face", linewidths=0.3)
        self.cells.set_clim(*clim)
        self.ax.add_collection(self.cells)
        self.lines = None
        source = self.cells
        if lines is not None:  # (cmap, clim, label) of lines over the cells
            self.lines = LineCollection([], cmap=lines[0], linewidths=1.8)
            self.lines.set_clim(*lines[1])
            self.ax.add_collection(self.lines)
            source, label = self.lines, lines[2]
        self.outlines = LineCollection([], colors="0.08", linewidths=0.9)
        self.ax.add_collection(self.outlines)
        self.fig.colorbar(source, cax=self.fig.add_axes((0.82, 0.15, 0.03, 0.7)), label=label)
        self.ax.set_xlim(*window[0])
        self.ax.set_ylim(*window[1])
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.title = self.ax.set_title("", fontsize=9)

    def frame(self, polygons, values, caption: str, segments=None, colours=None, *,
              outlines=None) -> np.ndarray:  # fmt: skip
        self.cells.set_verts(polygons)
        self.cells.set_array(np.asarray(values, dtype=float))
        if self.lines is not None:
            self.lines.set_segments(segments)
            self.lines.set_array(np.asarray(colours, dtype=float))
        self.outlines.set_segments([] if outlines is None else outlines)
        self.title.set_text(caption)
        return _rgb(self.fig)


def _rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _draw_section(ax, mesh, values, *, displacement=None, scale=0.0, cmap="viridis", clim=None,
                  edges=False, window=None, label="", colorbar=True):  # fmt: skip
    """The mid-section z of a mesh on axes, its cells coloured by ``values`` (one per
    element), within ``window`` ((x0, x1), (y0, y1)) when given."""
    from matplotlib.collections import PolyCollection

    polygons, owners = section(mesh, displacement, scale)
    values = np.asarray(values, dtype=float)[owners]
    cells = PolyCollection(polygons, array=values, cmap=cmap, rasterized=True,
                           edgecolors="0.25" if edges else "face",
                           linewidths=0.15 if edges else 0.3)  # fmt: skip
    if clim is not None:
        cells.set_clim(*clim)
    ax.add_collection(cells)
    ax.set_aspect("equal")
    ax.autoscale_view()
    if window is not None:
        ax.set_xlim(*window[0])
        ax.set_ylim(*window[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if colorbar:
        ax.figure.colorbar(cells, ax=ax, shrink=0.85, label=label)
    return cells


def _labels_cmap(count: int):
    from matplotlib.colors import ListedColormap

    from morphogen.viz.png import label_colors

    if count <= 1:
        return ListedColormap([(0.80, 0.84, 0.90)])
    if count <= 10:
        palette = ((0.80, 0.84, 0.90), (0.93, 0.68, 0.38), (0.45, 0.70, 0.55),
                   (0.62, 0.55, 0.80), (0.90, 0.50, 0.55), (0.55, 0.75, 0.90),
                   (0.85, 0.80, 0.45), (0.65, 0.65, 0.65), (0.95, 0.75, 0.80),
                   (0.50, 0.55, 0.40))  # fmt: skip
        return ListedColormap(palette[:count])
    return ListedColormap(label_colors(np.arange(count)) / 255.0)


def _path_xy(crack) -> np.ndarray:
    """The trace of a crack in the plane z: its polyline (a penny: its diameter along the
    trace of its plane)."""
    if hasattr(crack, "points"):
        return np.asarray(crack.points)[:, :2]
    centre, normal = np.asarray(crack.point), np.asarray(crack.normal)
    along = np.array([normal[1], -normal[0], 0.0])
    along /= max(np.linalg.norm(along), 1e-300)
    return np.array([centre - crack.size * along, centre + crack.size * along])[:, :2]


def _tips_xy(crack) -> np.ndarray:
    path = _path_xy(crack)
    return path[-1:] if hasattr(crack, "points") else path


def _window(crack, size: float, lower, upper, reach: float = 6.0):
    """The box around the front of a crack, ``reach`` elements wide on each side."""
    tips = _tips_xy(crack)
    half = reach * size
    lo = np.maximum(tips.min(axis=0) - half, np.asarray(lower)[:2])
    hi = np.minimum(tips.max(axis=0) + half, np.asarray(upper)[:2])
    return (lo[0], hi[0]), (lo[1], hi[1])


def von_mises(solution) -> np.ndarray:
    """The von Mises stress of each element of a solved body (Cauchy stresses)."""
    h, p, _, _, _ = solution.stresses()
    s = p
    if solution.finite:
        f = np.eye(3) + h
        s = np.einsum("eij,ekj->eik", p, f) / np.linalg.det(f)[:, None, None]
    dev = s - np.einsum("eii->e", s)[:, None, None] / 3.0 * np.eye(3)
    return np.sqrt(1.5 * np.einsum("eij,eij->e", dev, dev))


def _state_field(solution):
    """The field shown on a solved body: the cumulated plastic strain of the points of an
    inelastic bulk (the porosity of porous points), else the von Mises stress."""
    bulk = solution.bulk
    if bulk is None:
        return von_mises(solution), "von Mises stress"
    from morphogen.homogenization.behaviours import POROUS, STATE_P

    if np.any(bulk.kind == POROUS):
        from morphogen.homogenization.porous import porosity

        return porosity(bulk.old, bulk.behaviours.data[2], bulk.pmat), "porosity"
    return np.asarray(bulk.old[:, STATE_P], dtype=float), "cumulated plastic strain p"


def _magnification(solutions, lower, upper, fraction: float = 0.05) -> float:
    """The magnification drawing the largest displacement as ``fraction`` of the box."""
    peak = max(float(np.abs(s.displacement).max()) for s in solutions)
    return fraction * float(np.max(np.asarray(upper) - np.asarray(lower))) / max(peak, 1e-300)


def _mesh_figure(mesh, crack, size, lower, upper, names: dict, labels=None, title=""):
    """The mid-section of a crack mesh (its phases or grains) and the trace of the crack, with
    the zone around the front."""
    fig = _canvas(10.0, 4.6)
    left, right = fig.subplots(1, 2)
    if labels is None:
        labels = mesh.phase
    values = np.asarray(labels)
    count = int(values.max()) + 1 if values.size else 1
    for ax, window in ((left, None), (right, _window(crack, size, lower, upper))):
        _draw_section(ax, mesh, values, cmap=_labels_cmap(count), clim=(-0.5, count - 0.5),
                      edges=True, window=window, colorbar=False)  # fmt: skip
        path = _path_xy(crack)
        ax.plot(path[:, 0], path[:, 1], "-", color="crimson", lw=1.8)
        ax.plot(*_tips_xy(crack).T, "o", color="crimson", ms=4)
    left.set_title(title or f"mesh: {mesh.n_elements} tetrahedra (section z)", fontsize=10)
    right.set_title("around the front", fontsize=10)
    if names and len(names) > 1:
        from matplotlib.patches import Patch

        cmap = _labels_cmap(count)
        handles = [Patch(color=cmap((k + 0.5) / count), label=f"{k}: {name}")
                   for k, name in sorted(names.items()) if k < count]  # fmt: skip
        right.legend(handles=handles, fontsize=7, loc="upper right", frameon=True)
    return fig


def _field_figure(solution, crack, size, lower, upper, values, label: str, title: str = ""):
    """The field ``values`` of a solved body on its deformed mid-section (magnified), whole
    and around the front."""
    fig = _canvas(10.0, 4.6)
    left, right = fig.subplots(1, 2)
    m = _magnification([solution], lower, upper)
    clim = (float(np.min(values)), float(np.percentile(values, 99.5)))
    if not clim[1] > clim[0]:
        clim = (clim[0], clim[0] + 1e-300 + abs(clim[0]))
    window = _window(crack, size, lower, upper)
    _draw_section(left, solution.mesh, values, displacement=solution.displacement, scale=m,
                  clim=clim, colorbar=False)  # fmt: skip
    _draw_section(right, solution.mesh, values, displacement=solution.displacement, scale=m,
                  clim=clim, window=window, label=label)  # fmt: skip
    left.set_title(title or f"{label} (displacements x {m:.3g})", fontsize=10)
    right.set_title("around the front", fontsize=10)
    return fig


def _curves(x, ys: dict, xlabel: str, ylabel: str, title: str = "", *, lines=None,
            styles=None, twin=None):  # fmt: skip
    """A figure of curves ``ys`` (label: values) against ``x``; ``lines``: horizontal lines
    (label: value); ``twin``: (label, values) on a second axis."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.6, 3.9), layout="constrained")
    ax = fig.subplots()
    for k, (label, y) in enumerate(ys.items()):
        style = (styles or {}).get(label, "o-" if len(x) < 30 else "-")
        ax.plot(x, y, style, lw=1.6, ms=4, label=label, color=f"C{k}")
    for label, value in (lines or {}).items():
        ax.axhline(value, color="0.45", lw=0.9, ls="--", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    handles = ax.get_legend_handles_labels()[0]
    if twin is not None:
        other = ax.twinx()
        handles += other.plot(x, twin[1], ":", color="0.3", lw=1.4, label=twin[0])
        other.set_ylabel(twin[0])
    if len(handles) > 1:
        ax.legend(handles=handles, fontsize=8, frameon=False)
    return fig


def _write_vtu(ctx, name: str, solution, fields: dict) -> None:
    """The opened mesh of a solved body, its displacement and element fields (.vtu)."""
    from morphogen.io.meshes import write_vtu_tet

    write_vtu_tet(ctx.path(f"{name}.vtu"), solution.mesh,
                  cell_data={k: np.asarray(v, dtype=float) for k, v in fields.items()},
                  point_data={"displacement": np.asarray(solution.displacement)})  # fmt: skip


def _view3d(solution, values, label: str):
    """A 3D view (VTK, when it renders) of the deformed body cut through the middle of z."""
    from morphogen.viz.video import deformed_view, vtk_can_render

    if not vtk_can_render():
        return None
    return deformed_view(solution.mesh, solution.displacement, np.asarray(values, dtype=float),
                         section=(2, 0.5), title=label, cmap="viridis")  # fmt: skip


def _growth_frames(solutions, lower, upper, captions, outlines: bool = False) -> list[np.ndarray]:
    """The frames of a growth: the deformed mid-section of each solution (the magnification,
    the colours and the axes of the whole growth), coloured by :func:`_state_field`, with
    ``outlines`` the interfaces of the phases and the walls of the pores of a morphology."""
    fields = [_state_field(s) for s in solutions]
    label = fields[0][1]
    lo = min(float(np.min(f)) for f, _ in fields)
    hi = max(float(np.percentile(f, 99.5)) for f, _ in fields)
    hi = hi if hi > lo else lo + 1e-300 + abs(lo)
    m = _magnification(solutions, lower, upper)
    moved = np.concatenate([s.mesh.nodes + m * s.displacement for s in solutions])
    pad = 0.02 * float(np.max(np.asarray(upper) - np.asarray(lower)))
    window = ((moved[:, 0].min() - pad, moved[:, 0].max() + pad),
              (moved[:, 1].min() - pad, moved[:, 1].max() + pad))  # fmt: skip
    video = _Frames(window, (lo, hi), label)
    frames = []
    for s, (f, _), caption in zip(solutions, fields, captions, strict=True):
        polygons, owners = section(s.mesh, s.displacement, m)
        lines = _outlines(s, m) if outlines else None
        frames.append(video.frame(polygons, f[owners], f"{caption} (displacements x {m:.3g})",
                                  outlines=lines))  # fmt: skip
    return frames


def _deflections(cracks, advances) -> tuple[list, np.ndarray]:
    """The parts of the path of a growth along interfaces (polylines in the plane z, one per
    advance along an interface) and the points where the crack turned into an interface
    (from the bulk, or by more than 20° from another interface)."""
    lines, turns, before = [], [], None
    for k, advance in enumerate(advances[: len(cracks) - 1]):
        old, new = cracks[k], cracks[k + 1]
        if advance.kind == "interface":
            tail = np.vstack([old.tip, np.asarray(new.points[len(old.points) - 1 :])])[:, :2]
            lines.append(tail)
            if before is None or before.kind != "interface" or abs(advance.angle) > 0.35:
                turns.append(tail[-2] if len(tail) > 2 else tail[0])
        before = advance
    return lines, np.asarray(turns, dtype=float).reshape(-1, 2)


def _growth_figure(solution, cracks, size, lower, upper, title: str, names=None, advances=None):
    """The path of a growing crack over the field of its last configuration (undeformed),
    whole and around the path; with the ``names`` of the phases of a morphology, the path
    over its phases (whole), the outlines of their interfaces and of the pores; with the
    ``advances`` chosen by a toughness, the parts of the path along interfaces and the
    deflections into them marked."""
    from matplotlib.collections import LineCollection

    values, label = _state_field(solution)
    fig = _canvas(10.0, 4.6)
    left, right = fig.subplots(1, 2)
    path = _path_xy(cracks[-1])
    tips = np.array([_tips_xy(c)[0] for c in cracks])
    half = 4.0 * size
    lo = np.maximum(np.minimum(path.min(axis=0), tips.min(axis=0)) - half, lower[:2])
    hi = np.minimum(np.maximum(path.max(axis=0), tips.max(axis=0)) + half, upper[:2])
    clim = (float(np.min(values)), float(np.percentile(values, 99.5)))
    if not clim[1] > clim[0]:
        clim = (clim[0], clim[0] + 1e-300 + abs(clim[0]))
    outlines = _outlines(solution) if names is not None else None
    for ax, window in ((left, None), (right, ((lo[0], hi[0]), (lo[1], hi[1])))):
        if ax is left and names is not None:  # the phases of the morphology
            phases = np.asarray(solution.mesh.phase)
            count = int(phases.max()) + 1
            _draw_section(ax, solution.mesh, phases, cmap=_labels_cmap(count),
                          clim=(-0.5, count - 0.5), colorbar=False)  # fmt: skip
            ax.set_xlim(lower[0], upper[0])
            ax.set_ylim(lower[1], upper[1])
            if len(names) > 1:
                from matplotlib.patches import Patch

                cmap = _labels_cmap(count)
                handles = [Patch(color=cmap((k + 0.5) / count), label=f"{k}: {name}")
                           for k, name in sorted(names.items()) if k < count]  # fmt: skip
                ax.legend(handles=handles, fontsize=7, loc="upper right", frameon=True)
        else:
            _draw_section(ax, solution.mesh, values, clim=clim, window=window, label=label,
                          colorbar=ax is right)  # fmt: skip
        if outlines is not None:
            ax.add_collection(LineCollection(outlines, colors="0.08", linewidths=0.8))
        ax.plot(path[:, 0], path[:, 1], "-", color="white", lw=3.0)
        ax.plot(path[:, 0], path[:, 1], "-", color="crimson", lw=1.6)
        ax.plot(tips[:, 0], tips[:, 1], "o", color="crimson", ms=4, mec="white", mew=0.6)
        if advances:
            along, turns = _deflections(cracks, advances)
            if along:
                ax.add_collection(LineCollection(along, colors="gold", linewidths=2.2,
                                                 label="along an interface"))  # fmt: skip
            if len(turns):
                ax.plot(turns[:, 0], turns[:, 1], "D", color="gold", ms=6, mec="black", mew=0.8,
                        label="deflection")  # fmt: skip
            if ax is right and (along or len(turns)):
                ax.legend(fontsize=7, loc="lower right", frameon=True)
    left.set_title(title, fontsize=10)
    right.set_title(f"the path ({label} of the last crack)", fontsize=10)
    return fig


def _reach_of(crack) -> float:
    from morphogen.fracture.mixed import _reach

    return _reach(crack)


def _front_length(crack, lower, upper) -> float:
    return float(crack.front(lower, upper)[1].sum())


def _auto_stations(crack, size: float, lower, upper) -> int:
    """As many stations along the front as keep 2.5 elements between them (none below 3)."""
    length = _front_length(crack, lower, upper)
    if getattr(crack, "kind", None) == "penny":
        count = min(16, int(length / (2.5 * size)))
        return count if count >= 4 else 0
    count = min(12, int(length / (2.5 * size)) + 1)
    return count if count >= 3 else 0


def _elastic_twin(material):
    """The elastic material of an inelastic one (its elastic constants)."""
    from morphogen.homogenization.materials import Material

    if material.young is not None:
        return Material(young=material.young, poisson=material.poisson)
    return Material(stiffness=material.stiffness)


# -- the tools -----------------------------------------------------------------------------------

_BULK = {"bulk": "homogeneous"}


_MORPHOLOGY = {"bulk": "morphology"}


def _morphology_fields(interfaces: str) -> list[Field]:
    """The fields of a specimen that is a morphology of the workspace (``interfaces``: the
    default kind of the interfaces of its crack meshes)."""
    return [
        Field("bulk", "choice", "homogeneous", choices=(
            ("homogeneous", "a homogeneous plate or block"),
            ("morphology", "a morphology of the workspace: its domain, its phases"))),
        Field("source", "item", label="morphology", when=_MORPHOLOGY,
              doc="its domain is the specimen, its phases the bulk"),
        Field("materials", "materials", of="source", when=_MORPHOLOGY,
              doc="E and ν of each phase; void phases are pores, left out of the meshes"),
        Field("phase_kept", "item", label="behaviour of a phase", items=("behaviour",),
              none="none: the elastic phases above", when=_MORPHOLOGY,
              doc="a behaviour of the workspace (B: a material point, a Z-set file) given to "
              "the phase below, with its own elasticity"),
        Field("kept_phase", "int", 1, label="its phase", when=_MORPHOLOGY),
        Field("interfaces", "choice", interfaces, choices=INTERFACES, when=_MORPHOLOGY,
              doc="of the phases on the crack meshes"),
    ]  # fmt: skip


@tool(
    "crack_factors",
    "fracture",
    "Stress intensity factors of a crack",
    "A crack in a homogeneous plate or block, or in a morphology of the workspace (its phases "
    "on the crack mesh, conforming to their interfaces or as staircases, its pores left out), "
    "meshed by MMG conforming to it and opened: "
    "the factors K_I, K_II and K_III by the interaction integrals and the energy release rate "
    "G by the G-θ method (Destuynder & Djaoua; the J of Rice on a wider ring), along the "
    "front at stations or as their means, the angle of kinking (Erdogan & Sih), in small or "
    "finite strains; an inelastic bulk gives the J of the work density, of the free energy "
    "and at the tip (configurational forces). The handbook factor of the specimen when it has "
    "one, and the energy of the body as a check (G = -dΠ/dA).",
    [
        *_morphology_fields("staircase"),
        *_plate_fields(_BULK),
        *_behaviour_fields(_BULK),
        *_crack_fields(("edge", "through", "penny")),
        *_loading_fields(),
        Field(
            "stations",
            "int",
            doc="points along the front for the profiles of K and G, at least two elements "
            "apart (blank: as many as fit; 0: the means only)",
        ),
        Field(
            "contact",
            "bool",
            False,
            label="contact of the lips",
            doc="a penalty: lips closed by compression or shear",
        ),
        Field(
            "friction",
            "float",
            0.0,
            label="friction of the lips μ",
            when={"contact": [True]},
            doc="Coulomb's coefficient: lips in contact stick or slide (0: frictionless); "
            "linear elements",
        ),
        Field(
            "energy_check",
            "bool",
            False,
            label="energy check",
            doc="G by the energy of the body for the crack shorter and longer by an element "
            "(two more solutions; at fixed loads)",
        ),
    ],
    order=1,
    groups=(
        ("Specimen", "bulk"),
        ("Material", "behaviour"),
        ("Crack", "crack"),
        ("Loading", "loading"),
        ("Mesh and integrals", "element_size", True),
    ),
    menu="Cracks",
)
def crack_factors(p: dict, ctx) -> dict:
    from morphogen.fracture.mixed import energy_release_rate

    ctx.progress(0.02, "specimen")
    spec = specimen(p)
    inelastic = any(m.nonlinear for m in spec.materials.values())
    ctx.progress(0.05, "meshing conforming to the crack (MMG)")
    mesh = spec.mesh_of(spec.crack)
    ctx.progress(0.2, f"solving ({mesh.n_elements} tetrahedra)")
    options = {"kinematics": p["kinematics"], "contact": True if p.get("contact") else None}
    if p.get("contact") and float(p.get("friction") or 0.0) > 0:
        options["friction"] = float(p["friction"])
    solution = spec.solve(mesh=mesh, **options)
    ctx.progress(0.55, "G-θ and interaction integrals")
    h = solution.element_size()
    reach = _reach_of(spec.crack)
    values: dict = {"elements": mesh.n_elements, "element_size": h, "crack_area": solution.area(),
                    "front_length": _front_length(spec.crack, spec.lower, spec.upper)}  # fmt: skip
    values.update(_morphology_values(spec, solution))
    modes = solution.modes()
    young, nu = _isotropic_front(solution)
    e_prime = young / (1.0 - nu**2)
    for key in ("K_I", "K_II", "K_III", "G", "G_I", "G_II", "G_III"):
        values[key] = modes[key]
    values["kink_angle_deg"] = math.degrees(modes["angle"])
    r_out = min(8.0 * h, 0.95 * reach)
    r_in = min(2.0 * h, 0.5 * r_out)
    energies = ("work", "free", "tip") if inelastic else ("work",)
    rows = []
    for k in (2, 3, 4, 6, 8):
        if k * h > 0.95 * reach:
            break
        row = {"r_in": min(h, 0.5 * k * h), "r_out": k * h}
        for energy in energies:
            name = "G" if not inelastic else f"J_{energy}"
            try:
                row[name] = solution.release_rate(row["r_in"], row["r_out"], energy=energy)
            except ValueError:  # no free energy (MFront)
                row[name] = math.nan
        rows.append(row)
    if inelastic:
        for energy in energies:
            try:
                values[f"J_{energy}"] = solution.release_rate(energy=energy)
            except ValueError:
                values[f"J_{energy}"] = math.nan
        values["J_work_wide_ring"] = solution.release_rate(r_in, r_out, energy="work")
    else:
        values["J"] = solution.release_rate(r_in, r_out)
        values["K_equivalent"] = math.sqrt(max(e_prime * modes["G"], 0.0))  # sqrt(E' G)
    if spec.handbook is not None:
        name = spec.handbook["mode"]
        values[f"{name}_handbook"] = spec.handbook["value"]
        values[f"{name}_ratio"] = modes[name] / spec.handbook["value"]
    if p.get("energy_check"):
        ctx.progress(0.65, "energy check: two more cracks")
        delta = min(h, 0.5 * reach)
        values["G_energy"] = energy_release_rate(lambda c: spec.solve(c, **options), spec.crack,
                                                 delta)  # fmt: skip
        values["G_ratio_energy"] = modes["G"] / values["G_energy"]
    ctx.progress(0.8, "figures")
    ctx.table(rows, "domains", "independence of the domain (rings of the G-θ method)")
    stations = p.get("stations")
    if stations is None:
        stations = _auto_stations(spec.crack, h, spec.lower, spec.upper)
    if stations:
        try:
            profile = solution.modes(stations=int(stations))
        except ValueError as exc:
            import warnings

            warnings.warn(f"stations: {exc}", stacklevel=2)
            profile = None
        if profile is not None:
            keys = ("K_I", "K_II", "K_III", "G")
            table = [{"s": float(s), **{k: float(profile[k][i]) for k in keys},
                      "angle_deg": math.degrees(float(profile["angle"][i]))}
                     for i, s in enumerate(profile["s"])]  # fmt: skip
            ctx.table(table, "front", "along the front (s: a length, or an angle on a penny)")
            ys = {k: profile[k] for k in ("K_I", "K_II", "K_III")}
            ctx.figure(_curves(profile["s"], ys, "s along the front", "K",
                               "stress intensity factors along the front"), "front")  # fmt: skip
    labels = spec.labels(solution.mesh)
    names = spec.names if spec.morphology else {}
    ctx.figure(_rgb(_mesh_figure(solution.mesh, spec.crack, h, spec.lower, spec.upper, names,
                                 labels)), "mesh", "mesh conforming to the crack")  # fmt: skip
    field, label = _state_field(solution)
    ctx.figure(_rgb(_field_figure(solution, spec.crack, h, spec.lower, spec.upper, field, label)),
               "fields", f"{label} near the front")  # fmt: skip
    lengths = spec.upper - spec.lower
    if spec.kind == "penny" or lengths[2] > 0.25 * lengths[0]:  # a block: a 3D view
        view = _view3d(solution, field, label)
        if view is not None:
            ctx.figure(view, "view3d", f"{label} (3D, cut at the middle of z)")
    _write_vtu(ctx, "crack", solution, {label.replace(" ", "_"): field, "bulk_phase": labels})
    text = ""
    if spec.handbook is not None:
        name = spec.handbook["mode"]
        text = (f"{name} = {modes[name]:.5g} against {spec.handbook['value']:.5g} "
                f"({spec.handbook['source']}): {100 * (values[name + '_ratio'] - 1):+.1f} %. "
                "Linear tetrahedra without singular elements approach the exact factor from "
                "below as they shrink at the front; the handbook solution may assume a longer "
                "strip or an infinite body.")  # fmt: skip
    return {"values": values, "text": text}


def _phase_name(spec, phase: int) -> str:
    return f"{phase}: {spec.names[phase]}" if phase in spec.names else str(phase)


def _morphology_values(spec, solution) -> dict:
    """The values of a specimen that is a morphology: the interfaces of its crack meshes,
    its pores and the phase at the front."""
    if not spec.morphology:
        return {}
    out = {"interfaces": spec.interfaces,
           "phase_at_front": _phase_name(spec, solution.front_phase())}  # fmt: skip
    if spec.voids:
        out["pores"] = ", ".join(_phase_name(spec, k) for k in spec.voids)
    return out


def _isotropic_front(solution) -> tuple[float, float]:
    """Young's modulus and Poisson's ratio of the material at the front (the Voigt-Reuss-Hill
    estimate of a crystal), as the interaction integrals take them."""
    from morphogen.fracture.mixed import _isotropic_moduli

    return _isotropic_moduli(solution.materials[solution.front_phase()])


GROWTHS = (
    ("fixed", "at a fixed load: an increment per step (Paris' law; elastic bulks)"),
    ("toughness", "once the driving force at the tip reaches the toughness (G_c, J_tip)"),
    ("porosity", "once the porosity near the tip reaches a critical value (GTN points)"),
)
TRANSFERS = (
    ("nearest", "nearest: the values of the old element (the identity onto the same mesh)"),
    ("projection", "projection: through the nodes of the old mesh (smoother)"),
)
_CRITERION = {"growth": ["toughness", "porosity"]}


def _counting_distinct(mesh_of, ctx, steps: int, start: float = 0.05, span: float = 0.8):
    """``mesh_of`` reporting the progress of a growth, one mesh per new crack; the same
    crack asked again (an elastic estimate, then the growth) takes the same mesh."""
    state: dict = {"count": 0, "crack": None, "mesh": None}

    def counted(crack):
        if state["crack"] is not None and state["crack"] == crack:
            return state["mesh"]
        k = min(state["count"], steps)
        ctx.progress(start + span * k / (steps + 1), f"crack {k} of {steps}: meshing and solving")
        mesh = mesh_of(crack)
        state.update(count=state["count"] + 1, crack=crack, mesh=mesh)
        return mesh

    return counted


def _growth_outputs(ctx, lower, upper, size, cracks, solutions, captions, title: str,
                    names=None, advances=None, video: bool = False) -> None:  # fmt: skip
    """The figures, frames and file of a growth: the path over the last field (and over the
    phases of a morphology, with the ``names`` of its phases; the deflections along
    interfaces of the ``advances`` of a toughness), the frames of every crack (the interfaces
    of the phases outlined: their video if asked, else kept for the export), the opened mesh
    of the last one."""
    ctx.figure(_rgb(_growth_figure(solutions[-1], cracks, size, lower, upper, title, names,
                                   advances)), "path", "crack path")  # fmt: skip
    ctx.progress(0.92, "frames of the growth")
    frames = _growth_frames(solutions, lower, upper, captions, outlines=names is not None)
    ctx.video(frames, "growth", "the growth, crack by crack", fps=1.5, mode="loop", wanted=video)
    field, label = _state_field(solutions[-1])
    fields = {label.replace(" ", "_"): field}
    if names is not None:
        fields["bulk_phase"] = solutions[-1].mesh.phase
    _write_vtu(ctx, "last_crack", solutions[-1], fields)


@tool(
    "crack_growth",
    "fracture",
    "Crack propagation",
    "A straight crack in a homogeneous plate, or in a morphology of the workspace (its phases "
    "on the crack meshes, conforming to their interfaces or as staircases, with the materials "
    "of its phases; its void phases, the pores, left out), grows on meshes rebuilt by MMG at "
    "each advance, kinking in the direction of the largest hoop stress (Erdogan & Sih) by the "
    "factors of the interaction integrals (on the material at the front, the auxiliary fields "
    "of its moduli): deflected by stiff inclusions, attracted by soft ones and pores, through "
    "which it runs: at a fixed load, an increment per step (Paris' law counts the "
    "cycles, Griffith's criterion gives the critical load of each length), or, the load raised "
    "by steps, once the driving force at the tip reaches the toughness (G of an elastic bulk, "
    "J_tip of an inelastic one: von Mises elasto-viscoplasticity, GTN; MFront laws have no free "
    "energy, hence no J_tip) or the porosity of the GTN points near the tip a critical value "
    "(ductile growth), the state of the points "
    "carried from mesh to mesh. Through a morphology, a toughness for each phase and for the "
    "interfaces between them makes each advance the candidate of the largest G(θ) / Γ(θ): the "
    "kink of the bulk, against the toughness of the phase met, or along an interface within "
    "an increment ahead (the first-order kinked factors of Cotterell & Rice): deflected along "
    "weak interfaces (He & Hutchinson), through which it follows a brick-and-mortar, and held "
    "back by tough phases (the critical load of each advance). The path (over the phases of a "
    "morphology, its deflections marked), the loads and "
    "driving forces of each advance, the resistance curve (J_far = J_ψ on a ring around the "
    "front, Simha et al. 2008, with the configurational forces of the inclusions and pores in "
    "it) and the frames of the growth (their video on demand).",
    [
        *_morphology_fields("conforming"),
        Field(
            "slab",
            "float",
            unit="length",
            when=_MORPHOLOGY,
            doc="the thickness of the slab of the morphology about the middle of z, in plane "
            "strain: the path that of its section (blank: the whole depth, at most a "
            "sixteenth of the width)",
        ),
        *_plate_fields(_BULK, penny=False),
        *_behaviour_fields(_BULK),
        *_crack_fields(("edge", "through")),
        *_loading_fields(LOADINGS[:3]),
        Field("growth", "choice", "fixed", choices=GROWTHS),
        Field("steps", "int", doc="advances of the front (blank: 3, or 6 through a morphology)"),
        Field(
            "increment",
            "float",
            unit="length",
            doc="advance of the front at each step (blank: 1.5 elements)",
        ),
        Field(
            "kink",
            "bool",
            True,
            when=_CRITERION,
            doc="in the direction of the largest hoop stress (Erdogan & Sih), else straight",
        ),
        Field(
            "paris_C",
            "float",
            label="Paris' C",
            when={"growth": "fixed"},
            doc="da/dN = C ΔK^m with ΔK = sqrt(E' G) at the load (blank: no cycles)",
        ),
        Field("paris_m", "float", 3.0, label="Paris' m", when={"growth": "fixed"}),
        Field(
            "toughness",
            "float",
            unit="energy / area",
            when={"growth": ["fixed", "toughness"]},
            doc="G_c of an elastic bulk, J_tip of an inelastic one; at a fixed load, the "
            "critical load of each length is sqrt(G_c / G) (blank: none); by the toughness, "
            "blank: 1.2 times the elastic G of the first crack at the load factor 1",
        ),
        Field("critical_porosity", "float", when={"growth": "porosity"}, doc="blank: 1.05 f0"),
        Field(
            "radius",
            "float",
            unit="length",
            when={"growth": "porosity"},
            doc="the points within this distance of the front (blank: 1.5 element)",
        ),
        Field(
            "load_step",
            "float",
            0.25,
            when=_CRITERION,
            doc="of the load factor between the checks of the criterion",
        ),
        Field("max_load", "float", 4.0, when=_CRITERION, doc="the loading stops there"),
        Field(
            "transfer",
            "choice",
            "nearest",
            choices=TRANSFERS,
            when=_CRITERION,
            doc="of the state of the points onto the mesh of the next crack",
        ),
        Field(
            "contact",
            "bool",
            False,
            label="contact of the lips",
            doc="a penalty 50 E / h",
        ),
        Field(
            "friction",
            "float",
            0.0,
            label="friction of the lips μ",
            when={"contact": [True], "growth": ["fixed"]},
            doc="Coulomb's coefficient: lips in contact stick or slide (0: frictionless)",
        ),
        Field(
            "toughness_phases",
            "floats",
            label="toughness of the phases",
            unit="energy / area",
            when={**_MORPHOLOGY, "growth": ["fixed", "toughness"]},
            doc="Γ of each phase of the morphology, in the order of its phases (0, 1, ...; the "
            "void phases may be left out): the advance into the bulk meets that of the "
            "toughest phase along it, and an elastic crack grows at sqrt(Γ / G), so that tough "
            "phases hold it back (blank: the toughness for all the phases)",
        ),
        Field(
            "interface_toughness",
            "float",
            label="toughness of the interfaces",
            unit="energy / area",
            when={**_MORPHOLOGY, "growth": ["fixed", "toughness"]},
            doc="Γ of the interfaces between the phases: where an interface lies within an "
            "increment ahead of the front, the crack goes along it when its G(θ) / Γ is larger "
            "than that of the bulk (a crack normal to the interface of two alike phases "
            "deflects below Γ / Γ_bulk = 1/4, He & Hutchinson 1989; blank: no interface "
            "criterion, the crack crosses the interfaces by the kink of the bulk)",
        ),
        Field("video", "bool", False, doc="a video of the growth; the export makes it too"),
    ],
    order=2,
    groups=(
        ("Specimen", "bulk"),
        ("Material", "behaviour"),
        ("Crack", "crack"),
        ("Loading", "loading"),
        ("Growth", "growth"),
    ),
    menu="Cracks",
)
def crack_growth(p: dict, ctx) -> dict:
    from morphogen.fracture.ductile import propagate_ductile
    from morphogen.fracture.mixed import propagate_kinked

    ctx.progress(0.01, "specimen")
    spec = specimen(p, slab=True)
    solids = spec.solids()
    growth, steps = p["growth"], p.get("steps")
    steps = int(steps) if steps is not None else 6 if spec.morphology else 3
    if steps < 1:
        raise ValueError("steps: at least one advance")
    if growth == "fixed" and any(m.nonlinear for m in solids):
        raise ValueError("a fixed load: an elastic bulk (an inelastic one grows by its toughness "
                         "or its porosity, its state carried along)")  # fmt: skip
    if growth == "porosity" and not any(m.porous for m in solids):
        if spec.morphology:
            raise ValueError("porosity: a porous bulk, not the elastic phases of a morphology")
        raise ValueError("porosity: a porous bulk (behaviour: porous plasticity)")
    increment = float(p.get("increment") or 1.5 * spec.size)
    if not increment > 0:
        raise ValueError("increment: positive")
    mesh_of = _counting_distinct(spec.mesh_of, ctx, steps)
    contact = 50.0 * spec.young / spec.size if p.get("contact") else None
    kinematics = p["kinematics"]
    values: dict = {"increment": increment, "element_size": spec.size}
    advances = None  # chosen by a toughness of the phases and of their interfaces
    if growth == "fixed":
        paris = (float(p["paris_C"]), float(p["paris_m"])) if p.get("paris_C") else None
        toughness = p.get("toughness")
        rule = _toughness_of(p, spec, toughness)
        if rule is not None and rule.uniform:  # one value for all the phases: as without
            toughness, rule = rule.value, None
        rubbing = {}
        if contact and float(p.get("friction") or 0.0) > 0:
            rubbing["friction"] = float(p["friction"])
        result = propagate_kinked(mesh_of, spec.crack, spec.materials, spec.prescribed, 1.0,
                                  increment=increment, steps=steps, kinematics=kinematics,
                                  paris=paris, contact=contact, toughness=rule, keep=True,
                                  forces=spec.forces, **rubbing)  # fmt: skip
        rows = result.rows()
        for row in rows:
            row.pop("z_tip", None)
            if paris is None:
                row.pop("N", None)
            if toughness and rule is None:
                row["load_critical"] = math.sqrt(toughness / row["G"]) if row["G"] > 0 else math.nan
        cracks, solutions = result.cracks, result.solutions
        captions = [f"crack {k}: a = {c.size:.3g}, G = {g:.3g}"
                    for k, (c, g) in enumerate(zip(cracks, result.rates, strict=True))]  # fmt: skip
        _tip_phases(spec, rows, solutions, captions)
        ctx.table(rows, "advances", "the cracks: lengths, tips, factors, kinks")
        lengths = np.array([c.size for c in cracks])
        lines = {"G_c": toughness} if toughness and rule is None else None
        twin = ("cycles N", result.cycles) if paris is not None else None
        ctx.figure(_curves(lengths, {"G": result.rates}, "crack length", "G",
                           "energy release rate along the growth", lines=lines, twin=twin),
                   "release_rate", "G along the growth")  # fmt: skip
        values.update(final_length=cracks[-1].size, G_first=float(result.rates[0]),
                      G_last=float(result.rates[-1]),
                      kink_first_deg=math.degrees(float(result.angles[0])))  # fmt: skip
        if paris is not None:
            values["cycles"] = float(result.cycles[-1])
        if rule is not None:
            advances = result.advances
            _toughness_outputs(values, spec, rule, cracks, advances)
            ctx.figure(_curves(lengths, {"critical load": result.loads}, "crack length",
                               "load factor", "the critical load of each crack, sqrt(Γ / G) of "
                               "its advance"), "critical_load", "critical loads")  # fmt: skip
            values["load_critical_first"] = float(result.loads[0])
            values["load_critical_largest"] = float(np.max(result.loads))
    else:
        toughness = porosity = rule = None
        if growth == "toughness":
            toughness = p.get("toughness")
            if toughness is None and not p.get("toughness_phases"):  # the elastic G at the tip
                ctx.progress(0.03, "the elastic G of the first crack")
                twins = {k: m if m.void else _elastic_twin(m) for k, m in spec.materials.items()}
                first = spec.solve(spec.crack, mesh_of(spec.crack), twins, kinematics=kinematics)
                toughness = 1.2 * first.release_rate(energy="tip")
            rule = _toughness_of(p, spec, toughness)
            if rule is not None and rule.uniform:  # one value for all the phases: as without
                toughness, rule = rule.value, None
            if toughness is not None:
                values["toughness"] = toughness
        else:
            porosity = p.get("critical_porosity")
            if porosity is None:
                porous = next(m for m in solids if m.porous)
                porosity = 1.05 * porous.porous_plasticity.porosity
            values["critical_porosity"] = porosity
        result = propagate_ductile(mesh_of, spec.crack, spec.materials, spec.prescribed,
                                   load_step=float(p["load_step"]), increment=increment,
                                   steps=steps, porosity=porosity,
                                   toughness=toughness if rule is None else rule,
                                   max_load=float(p["max_load"]), kinematics=kinematics,
                                   transfer=p["transfer"], radius=p.get("radius"),
                                   forces=spec.forces, contact=contact, kink=bool(p["kink"]),
                                   keep=True)  # fmt: skip
        cracks, solutions = result.cracks, result.solutions
        rows = result.rows()
        for row, c in zip(rows, cracks, strict=True):
            row["x_tip"], row["y_tip"] = float(c.tip[0]), float(c.tip[1])
        captions = [f"advance {k}: load {lam:.3g}, J_tip {jt:.3g}" for k, (lam, jt) in
                    enumerate(zip(result.loads, result.tip, strict=True))]  # fmt: skip
        _tip_phases(spec, rows, solutions, captions)
        ctx.table(rows, "advances", "the advances: loads, driving forces, factors, kinks")
        ctx.table(result.history, "loading", "every load step: driving forces, porosity at the tip")
        last = result.history[-1]
        if rule is not None:  # the ratio G(θ) / Γ(θ) of the advance
            broken = last.get("ratio", 0.0) >= 1.0
        else:
            broken = toughness is not None and last["J_tip"] >= toughness
        met = broken or (porosity is not None and last["porosity_tip"] >= porosity)
        values.update(advances=len(cracks) - 1, final_length=cracks[-1].size,
                      last_load=float(result.loads[-1]),
                      stopped="after the last step" if met else "at the maximum load")  # fmt: skip
        delta, far = result.resistance()
        ys = {"J_far (J_ψ, ring)": far, "J_tip": result.tip, "J_work": result.work}
        if result.advances:  # the toughness met by each advance
            ys["Γ of the advance"] = np.array([a.toughness for a in result.advances])
        lines = {"toughness": toughness} if toughness is not None and rule is None else None
        curve = _curves(delta, ys, "crack advance Δa", "driving force", "resistance curve",
                        lines=lines, twin=("load factor", result.loads))  # fmt: skip
        ctx.figure(curve, "resistance", "R-curve")
        loads = np.array([h["load"] for h in result.history])
        if result.advances:
            key, limit = "ratio", {"G(θ) / Γ(θ) = 1": 1.0}
        elif porosity is None:
            key, limit = "J_tip", {"toughness": toughness}
        else:
            key, limit = "porosity_tip", {"critical": porosity}
        trace = np.array([h[key] for h in result.history])
        ctx.figure(_curves(np.arange(len(loads)), {key: trace}, "load step", key,
                           "the criterion at each load step", lines=limit,
                           twin=("load factor", loads)), "loading", "criterion")  # fmt: skip
        if result.advances:
            advances = result.advances
            _toughness_outputs(values, spec, rule, cracks, advances)
    values["cracks"] = len(cracks)
    if spec.morphology:
        values.update(_morphology_values(spec, solutions[-1]))
        values["phases_at_tips"] = " / ".join(str(row["phase"]) for row in rows)
        values["largest_kink_deg"] = max(abs(row["angle_deg"]) for row in rows)
        values["tip_deflection"] = float(cracks[-1].tip[1] - cracks[0].tip[1])
    names = spec.names if spec.morphology else None
    _growth_outputs(ctx, spec.lower, spec.upper, spec.size, cracks, solutions, captions,
                    f"{len(cracks)} cracks: the path", names, advances,
                    video=p["video"])  # fmt: skip
    return {"values": values}


def _toughness_of(p: dict, spec, toughness):
    """The toughness of the phases and of their interfaces of the form (the fields
    ``toughness_phases``, in the order of the phases, and ``interface_toughness``;
    ``toughness``: that of every phase when the form gives none per phase), or None when it
    gives neither (one toughness for the whole body, the growth without interfaces). The
    interfaces are those of the crack meshes (conforming), or of the image of the
    morphology (staircases)."""
    from morphogen.fracture.toughness import Toughness

    given, interface = p.get("toughness_phases"), p.get("interface_toughness")
    if not spec.morphology or (not given and interface is None):
        return None
    phases = sorted(spec.materials)
    solid = [k for k in phases if not spec.materials[k].void]
    if given:
        if len(given) == len(phases):
            table = {k: float(v) for k, v in zip(phases, given, strict=True) if k in solid}
        elif len(given) == len(solid):
            table = {k: float(v) for k, v in zip(solid, given, strict=True)}
        else:
            raise ValueError(f"toughness of the phases: one value per phase, in the order of "
                             f"the phases {', '.join(map(str, phases))}")  # fmt: skip
    elif toughness is None:
        raise ValueError("toughness of the interfaces: a toughness of the phases too (the "
                         "toughness, or the toughness of the phases)")  # fmt: skip
    else:
        table = dict.fromkeys(solid, float(toughness))
    morphology = spec.mesh_of.phase_of if spec.interfaces == "staircase" else None
    return Toughness(table, interface, morphology=morphology)


def _toughness_outputs(values: dict, spec, rule, cracks, advances) -> None:
    """The values of a growth by a toughness of the phases and of their interfaces: that of
    each phase and of the interfaces, the advances along interfaces and the deflections into
    them, the length of the path against its extent along the first direction."""
    values["toughness_phases"] = ", ".join(f"{_phase_name(spec, k)}: {v:.4g}"
                                           for k, v in sorted(rule.phases.items()))  # fmt: skip
    if rule.interfaces is not None:
        values["interface_toughness"] = rule.interfaces
    taken = advances[: len(cracks) - 1]
    values["along_interfaces"] = sum(a.kind == "interface" for a in taken)
    values["deflections"] = len(_deflections(cracks, advances)[1])
    first = cracks[0]
    direction = np.asarray(first.points[-1]) - np.asarray(first.points[-2])
    direction /= np.linalg.norm(direction)
    reach = float((np.asarray(cracks[-1].tip) - np.asarray(first.tip)) @ direction)
    values["path_length"] = float(cracks[-1].size - first.size)
    values["path_extent"] = reach


def _tip_phases(spec, rows, solutions, captions) -> None:
    """The phase at the front of each crack of a growth through a morphology, in the rows of
    its table and the captions of its frames."""
    if not spec.morphology:
        return
    for k, (row, solution) in enumerate(zip(rows, solutions, strict=True)):
        phase = solution.front_phase()
        row["phase"] = phase
        captions[k] = f"{captions[k]}, tip in {_phase_name(spec, phase)}"


# -- cohesive zones ------------------------------------------------------------------------------

COHESIVE_SPECIMENS = (
    ("ligament", "a notched plate: a cohesive ligament ahead of the notch, pulled by its grips"),
    ("interfaces", "a morphology of the workspace: the interfaces of its phases or its grain "
     "boundaries, in tension along y"),
    ("dcb", "a double cantilever beam (mode I), against the corrected beam theory"),
)  # fmt: skip
LAWS = (
    ("bilinear", "bilinear (Camanho & Dávila; Benzeggagh-Kenane mixity)"),
    ("exponential", "exponential (Ortiz & Pandolfi)"),
)
_PLATES = {"specimen": ["ligament", "dcb"]}


def _cubic(p: dict, plastic: bool = False):
    """The cubic crystal of the form (its crystal plasticity, Méric-Cailletaud, with
    ``plastic``)."""
    import dataclasses

    from morphogen.homogenization.crystal import CrystalPlasticity
    from morphogen.homogenization.materials import Material

    crystal = Material.cubic(float(p["c11"]), float(p["c12"]), float(p["c44"]))
    if not plastic:
        return crystal
    slip = CrystalPlasticity.from_config({"families": p.get("families") or ["octahedral"],
                                          "tau0": p["tau0"], "Q": p["Q"], "b": p["b"],
                                          "K": p["K"], "n": p["n"]})  # fmt: skip
    return dataclasses.replace(crystal, crystal_plasticity=slip)


def _crystal_fields(when: dict | None = None) -> list[Field]:
    return [
        Field("c11", "float", 168e3, label="C11", unit="stress", when=when,
              doc="cubic crystal (copper, MPa)"),
        Field("c12", "float", 121e3, label="C12", unit="stress", when=when),
        Field("c44", "float", 75e3, label="C44", unit="stress", when=when),
    ]  # fmt: skip


def _law(p: dict, young: float, size: float, beam: bool = False):
    """The traction-separation law of the form; blank values scaled by the stiffest Young's
    modulus E and the element size h: strengths E / 100, energies giving a cohesive zone of
    four elements (G = 4 h N² / E), a penalty 100 E / h; for the beam, the strength E / 200
    and the energy E / 2000 (the case checked against the beam theory: a zone of 20 elements
    of 1 for E = 1000)."""
    from morphogen.fracture.cohesive import Bilinear, Exponential

    strength = young / (200.0 if beam else 100.0)

    def energy_of(n):
        return young / 2000.0 if beam else 4.0 * size * n**2 / young

    if p["law"] == "exponential":
        sigma = p.get("strength") or strength
        return Exponential(strength=sigma, energy=p.get("energy") or energy_of(sigma),
                           beta=float(p.get("beta") or 1.0))  # fmt: skip
    n = p.get("normal_strength") or strength
    s = p.get("shear_strength") or n
    g1 = p.get("mode_I") or energy_of(n)
    g2 = p.get("mode_II") or g1 * (s / n) ** 2
    k = p.get("stiffness") or 100.0 * young / size
    return Bilinear(k, n, s, g1, g2, float(p.get("eta") or 2.0))  # fmt: skip


def _law_values(law) -> tuple[float, float, float]:
    """The mode I strength, fracture energy and separation at failure of a law."""
    if hasattr(law, "normal_strength"):
        return law.normal_strength, law.mode_I, 2.0 * law.mode_I / law.normal_strength
    return law.strength, law.energy, 10.0 * law.critical_opening


def _damage_now(law, mesh, faces, u) -> np.ndarray:
    """The damage of each interface element (its mean separation) under the displacement
    ``u`` from an intact history: the damage of a monotonic opening, for the frames."""
    from morphogen.fracture.cohesive import traction

    x = mesh.nodes[faces.minus]
    normal = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
    jump = (u[faces.plus] - u[faces.minus]).mean(axis=1)
    dn = np.einsum("ij,ij->i", jump, normal)
    ds = np.linalg.norm(jump - dn[:, None] * normal, axis=1)
    out = np.empty(len(dn))
    for e in range(len(dn)):
        _, h = traction(law, (dn[e], ds[e], 0.0), 0.0)
        out[e] = h
    if not hasattr(law, "normal_strength"):  # exponential: the history is the largest opening
        out = 1.0 - np.exp(-out / law.critical_opening)
    return np.clip(out, 0.0, 1.0)


def _interface_segments(mesh, faces, u, scale: float, at: float):
    """The traces of the interface elements (their faces of side -) on the plane z = ``at``,
    in the configuration ``x + scale u``: segments (n x 2 x 2) and the element of each."""
    return _traces(mesh.nodes, faces.minus, mesh.nodes + scale * u, at)


def _traces(nodes, tri, moved, at: float):
    """The traces of triangles (rows of three nodes) on the plane z = ``at`` of the positions
    ``nodes``, drawn at the positions ``moved``: segments (n x 2 x 2) and the triangle of
    each."""
    tri = np.asarray(tri, dtype=np.int64).reshape(-1, 3)
    d = nodes[tri, 2] - at
    positive = d > 0
    count = positive.sum(axis=1)
    rows = np.flatnonzero((count == 1) | (count == 2))
    if not len(rows):
        return np.zeros((0, 2, 2)), rows
    lone = np.where(count[rows] == 1, np.argmax(positive[rows], axis=1),
                    np.argmin(positive[rows], axis=1))  # fmt: skip
    ends = []
    for shift in (1, 2):
        other = (lone + shift) % 3
        a, b = tri[rows, lone], tri[rows, other]
        da, db = d[rows, lone], d[rows, other]
        t = (da / (da - db))[:, None]
        ends.append((moved[a] + t * (moved[b] - moved[a]))[:, :2])
    return np.stack(ends, axis=1), rows


def _phase_faces(mesh, lips=None) -> np.ndarray:
    """The faces (rows of three nodes) of the interfaces between the phases of a mesh and of
    the walls of its pores (its free faces off its box and off the ``lips`` of its crack)."""
    from morphogen.fracture.lefm import boundary_faces
    from morphogen.mesh.tetra import TET_FACES

    tets = mesh.elements[:, :4]
    faces = tets[:, TET_FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(mesh.n_elements), 4)
    keys = np.sort(faces, axis=1)
    order = np.lexsort(keys.T[::-1])
    same = np.all(keys[order][1:] == keys[order][:-1], axis=1)
    a, b = order[:-1][same], order[1:][same]
    between = faces[a][mesh.phase[owner[a]] != mesh.phase[owner[b]]]
    free = boundary_faces(mesh)
    x, tol = mesh.nodes[free], mesh.tolerance()
    wall = np.ones(len(free), dtype=bool)
    for axis in range(3):
        for value in (mesh.lower[axis], mesh.upper[axis]):
            wall &= ~np.all(np.abs(x[:, :, axis] - value) <= tol, axis=1)
    if lips is not None and lips.n_elements:
        crack = {tuple(k) for k in np.sort(np.vstack([lips.minus, lips.plus]), axis=1).tolist()}
        wall &= ~np.array([tuple(k) in crack for k in np.sort(free, axis=1).tolist()], dtype=bool)
    return np.vstack([between, free[wall]])


def _outlines(solution, scale: float = 0.0, faces=None):
    """The traces on the mid-plane z of the interfaces of the phases and of the walls of the
    pores of a solved body (``faces``: those of :func:`_phase_faces`), in the configuration
    ``x + scale u``: segments (n x 2 x 2)."""
    mesh = solution.mesh
    faces = _phase_faces(mesh, solution.faces) if faces is None else faces
    at = 0.5 * (mesh.nodes[:, 2].min() + mesh.nodes[:, 2].max())
    moved = mesh.nodes + scale * np.asarray(solution.displacement).reshape(-1, 3)
    return _traces(mesh.nodes, faces, moved, at)[0]


def _cohesive_figure(mesh, faces, u, damage, scale, title, window=None, *, colorbar=True,
                     size=(6.4, 5.0)):  # fmt: skip
    """The deformed mid-section of a body with cohesive interfaces: the bulk in grey, the
    traces of the interfaces coloured by their damage."""
    from matplotlib.collections import LineCollection

    fig = _canvas(*size)
    ax = fig.subplots()
    labels = mesh.cell if mesh.cell is not None else mesh.phase
    _draw_section(ax, mesh, np.zeros(mesh.n_elements) if mesh.cell is None else labels,
                  displacement=u, scale=scale, cmap=_labels_cmap(1), colorbar=False,
                  window=window)  # fmt: skip
    at = 0.5 * (mesh.nodes[:, 2].min() + mesh.nodes[:, 2].max())
    segments, rows = _interface_segments(mesh, faces, u, scale, at)
    lines = LineCollection(segments, array=np.asarray(damage)[rows], cmap="inferno_r",
                           linewidths=1.8)  # fmt: skip
    lines.set_clim(0.0, 1.0)
    ax.add_collection(lines)
    if colorbar:
        fig.colorbar(lines, ax=ax, shrink=0.85, label="damage of the interfaces")
    ax.set_title(title, fontsize=9)
    return fig


def _report_newton(problem, ctx, load: float, start: float = 0.1, span: float = 0.8) -> None:
    """Report the load factor of each step of a cohesive problem as its progress (the load
    factor is an argument of its Newton solver; nothing is reported if it changes)."""
    newton = problem._newton

    def reported(*args, **kwargs):
        target = args[5] if len(args) > 5 else kwargs.get("target")
        if isinstance(target, float):
            ctx.progress(start + span * min(target / load, 1.0),
                         f"load factor {target:.4g} of {load:.4g}")  # fmt: skip
        return newton(*args, **kwargs)

    problem._newton = reported


def _box_mesh(lower, upper, size: float, plane_y=None):
    """Six tetrahedra per voxel of the box (about ``size``; ``plane_y`` on the faces of the
    voxels)."""
    from morphogen.homogenization.bloch import voxel_tet_mesh
    from morphogen.image import VoxelImage

    lengths = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    shape = np.maximum(1, np.rint(lengths / size)).astype(int)
    if plane_y is not None:  # an even count along y, the plane on a layer of faces
        below = max(1, round((plane_y - lower[1]) / size))
        above = max(1, round((upper[1] - plane_y) / size))
        if not np.isclose((plane_y - lower[1]) / below, (upper[1] - plane_y) / above):
            raise ValueError("the plane of the ligament: on a layer of voxel faces (its height "
                             "a multiple of the element size)")  # fmt: skip
        shape[1] = below + above
    image = VoxelImage(np.zeros(tuple(shape), np.uint8), spacing=tuple(lengths / shape),
                       origin=tuple(lower), periodic=False)  # fmt: skip
    return voxel_tet_mesh(image)


def _morphology_mesh(micro, size: float, grains: bool):
    """A mesh of a morphology conforming to its phases (MMG) or grains (a tessellation, cut
    by its box), its curved interfaces followed coarsely (a Hausdorff distance of h / 5,
    elements of h / 3 at least: every interface element is a cohesive one; the regularized
    cells of tessellations: on the grid, not exact); without a level set, six tetrahedra per
    voxel of its image (the cells of its labels as grains)."""
    from morphogen.homogenization.bloch import voxel_tet_mesh
    from morphogen.mesh.conforming import conforming_mesh
    from morphogen.mesh.tetra import TetMesh

    tessellation = hasattr(micro, "cell_of") and not getattr(micro, "boundary_thickness", 0)
    if grains and not tessellation:
        raise ValueError("grain boundaries: a Voronoi or Laguerre tessellation (without a "
                         "grain-boundary phase)")  # fmt: skip
    coarse = {"size": size, "hausdorff": size / 5.0, "min_size": size / 3.0, "periodic": False,
              "method": "grid"}  # fmt: skip
    try:
        if grains:
            return conforming_mesh(micro, cells="box", **coarse)
        return conforming_mesh(micro, grains=False, **coarse)
    except ValueError as exc:  # no level set: the voxels
        if "level-set" not in str(exc):
            raise
        image = micro.voxelize(voxel_size=size)
        mesh = voxel_tet_mesh(image)
        if grains:
            cells = np.asarray(micro.labels(grid=image.grid()).data).ravel()
            mesh = TetMesh(mesh.nodes, mesh.elements, mesh.phase, mesh.lower, mesh.upper,
                           cell=np.tile(cells, 6), orientations=micro.orientations)  # fmt: skip
        return mesh


@tool(
    "cohesive_zones",
    "fracture",
    "Cohesive zones",
    "Cohesive interface elements (six nodes, zero thickness) inserted along given surfaces "
    "and loaded by the displacement of grips, in small or finite strains: the ligament ahead "
    "of the notch of a plate, the interfaces of the phases of a morphology (debonding) or the "
    "grain boundaries of a tessellation (intergranular cracking), or a double cantilever beam "
    "against the corrected beam theory. Traction-separation laws: bilinear (Camanho & Dávila, "
    "Turon et al.) or exponential (Ortiz & Pandolfi), the instabilities passed by the viscous "
    "regularization of Gao & Bower. The force-displacement curve, the energies (of the loads, "
    "stored and dissipated by the bulk, dissipated by the interfaces: about G_c per unit of "
    "cracked area in mode I), the cohesive zone length and the broken interfaces.",
    [
        Field("specimen", "choice", "ligament", choices=COHESIVE_SPECIMENS),
        Field("source", "item", label="morphology", when={"specimen": "interfaces"}),
        Field(
            "interfaces",
            "choice",
            "phases",
            when={"specimen": "interfaces"},
            choices=(
                ("phases", "between the phases (the debonding of inclusions)"),
                ("grains", "every grain boundary of a tessellation"),
            ),
        ),
        Field(
            "materials",
            "materials",
            of="source",
            when={"specimen": "interfaces"},
            doc="E and ν of each phase; void phases are pores, left out of the mesh",
        ),
        Field(
            "crystal",
            "bool",
            False,
            label="cubic crystal grains",
            when={"specimen": "interfaces", "interfaces": "grains"},
            doc="crystal elasticity in each grain, turned by its orientation (random when the "
            "tessellation has none)",
        ),
        *_crystal_fields({"specimen": "interfaces", "interfaces": "grains", "crystal": True}),
        Field("width", "float", 8.0, unit="length", when={"specimen": "ligament"}),
        Field(
            "height",
            "float",
            8.0,
            unit="length",
            when={"specimen": "ligament"},
            doc="the ligament on its mid-plane y = H / 2",
        ),
        Field("thickness", "float", 1.0, unit="length", when={"specimen": "ligament"}),
        Field(
            "notch",
            "float",
            2.0,
            unit="length",
            when={"specimen": "ligament"},
            doc="the free crack from x = 0, the cohesive ligament beyond",
        ),
        Field("beam_length", "float", 60.0, unit="length", when={"specimen": "dcb"}),
        Field("beam_width", "float", 2.0, unit="length", when={"specimen": "dcb"}),
        Field(
            "arm",
            "float",
            4.0,
            unit="length",
            when={"specimen": "dcb"},
            doc="the thickness of each arm",
        ),
        Field("precrack", "float", 20.0, unit="length", when={"specimen": "dcb"}),
        *_behaviour_fields(_PLATES),
        Field(
            "element_size",
            "float",
            unit="length",
            doc="blank: 0.5 (ligament), 1 (beam), a tenth of the largest edge (morphology; a "
            "3D one takes minutes)",
        ),
        Field("law", "choice", "bilinear", choices=LAWS),
        Field(
            "normal_strength",
            "float",
            label="normal strength N",
            unit="stress",
            when={"law": "bilinear"},
            doc="blank: E / 100 (E / 200 for the beam)",
        ),
        Field(
            "shear_strength",
            "float",
            label="shear strength S",
            unit="stress",
            when={"law": "bilinear"},
            doc="blank: N",
        ),
        Field(
            "mode_I",
            "float",
            label="G_Ic",
            unit="energy / area",
            when={"law": "bilinear"},
            doc="blank: 4 h N² / E, a cohesive zone of four elements (E / 2000 for the beam)",
        ),
        Field(
            "mode_II",
            "float",
            label="G_IIc",
            unit="energy / area",
            when={"law": "bilinear"},
            doc="blank: G_Ic (S / N)²",
        ),
        Field("eta", "float", 2.0, label="η (Benzeggagh-Kenane)", when={"law": "bilinear"}),
        Field(
            "stiffness",
            "float",
            label="penalty K",
            unit="stress / length",
            when={"law": "bilinear"},
            doc="blank: 100 E / h",
        ),
        Field(
            "strength",
            "float",
            label="strength σc",
            unit="stress",
            when={"law": "exponential"},
            doc="blank: E / 100 (E / 200 for the beam)",
        ),
        Field(
            "energy",
            "float",
            label="G_c",
            unit="energy / area",
            when={"law": "exponential"},
            doc="blank: 4 h σc² / E (E / 2000 for the beam)",
        ),
        Field("beta", "float", 1.0, label="β (shear / normal)", when={"law": "exponential"}),
        Field(
            "displacement",
            "float",
            unit="length",
            doc="of each grip (of the loading line of each arm) at the end (blank: enough to "
            "break the interfaces in tension)",
        ),
        Field("steps", "int", 60, doc="steps of the loading (divided when Newton fails)"),
        Field(
            "regularization",
            "float",
            label="viscous regularization ζ",
            doc="of Gao & Bower, in units of the "
            "displacement (blank: 10⁻³ of the displacement; 0: none)",
        ),
        Field("kinematics", "choice", "small", choices=KINEMATICS),
        Field("video", "bool", False, doc="a video of the opening; the export makes it too"),
    ],
    order=5,
    groups=(
        ("Specimen", "specimen"),
        ("Bulk", "behaviour"),
        ("Cohesive law", "law"),
        ("Loading", "displacement"),
    ),
    menu="Cohesive zones",
)
def cohesive_zones(p: dict, ctx) -> dict:
    from morphogen.fracture.cohesive import (
        CohesiveProblem,
        cohesive_zone_length,
        dcb_beam_theory,
        dcb_specimen,
        insert_interfaces,
        turon_strength,
    )
    from morphogen.fracture.lefm import without_voids

    kind = p["specimen"]
    ctx.progress(0.02, "meshing")
    names: dict = {}
    if kind == "interfaces":
        item = p.get("source")
        if item is None:
            raise ValueError("interfaces: choose a morphology of the workspace")
        micro = item["micro"]
        lower = np.asarray(micro.domain.lower, dtype=float)
        upper = np.asarray(micro.domain.upper, dtype=float)
        size = float(p.get("element_size") or float(np.max(upper - lower)) / 10.0)
        _size(size, size, "edge", float(np.prod(upper - lower)))
        grains = p["interfaces"] == "grains"
        mesh = _morphology_mesh(micro, size, grains)
        if grains:
            if p.get("crystal"):
                orientations = mesh.orientations
                if orientations is None:
                    from morphogen.orientations import random_orientations

                    count = int(mesh.cell.max()) + 1
                    orientations = random_orientations(np.random.default_rng(item.get("seed")),
                                                       count)  # fmt: skip
                from morphogen.mesh.tetra import TetMesh

                mesh = TetMesh(mesh.nodes, mesh.elements, np.zeros(mesh.n_elements, np.int64),
                               mesh.lower, mesh.upper, metadata=dict(mesh.metadata),
                               cell=mesh.cell, orientations=orientations)  # fmt: skip
                materials = {0: _cubic(p)}
            else:
                materials = _phase_materials(item, p.get("materials"), np.unique(mesh.phase))
                mesh = without_voids(mesh, [k for k, m in materials.items() if m.void])
            opened, faces = insert_interfaces(mesh, lambda c: np.ones(len(c), dtype=bool),
                                              sides=mesh.cell)  # fmt: skip
        else:  # the pores (void phases) left out: their walls free
            materials = _phase_materials(item, p.get("materials"), np.unique(mesh.phase))
            names = {int(k): v for k, v in (item.get("phases") or {}).items()}
            mesh = without_voids(mesh, [k for k, m in materials.items() if m.void])
            opened, faces = insert_interfaces(mesh, "phases")
        cohesive = faces
        if not cohesive.n_elements:
            raise ValueError("no interface in this morphology (one phase, or one grain)")
        x = opened.nodes
        tol = 1e-9 * float(np.max(upper - lower))
        top = np.flatnonzero(x[:, 1] > upper[1] - tol)
        bottom = np.flatnonzero(x[:, 1] < lower[1] + tol)
        a = bottom[np.argmin(np.linalg.norm(x[bottom] - lower, axis=1))]
        b = bottom[np.argmin(np.linalg.norm(x[bottom] - [upper[0], lower[1], lower[2]], axis=1))]
        prescribed = [(top, 1, 1.0), (bottom, 1, 0.0), (a, [0, 2], 0.0), (b, 2, 0.0)]
        component, factor, span = 1, 1.0, float(upper[1] - lower[1])
    elif kind == "ligament":
        width, height = float(p["width"]), float(p["height"])
        lower, upper = np.zeros(3), np.array([width, height, float(p["thickness"])])
        size = float(p.get("element_size") or 0.5)
        mesh = _box_mesh(lower, upper, size, plane_y=0.5 * height)
        plane = 0.5 * height
        opened, faces = insert_interfaces(mesh, lambda c: np.abs(c[:, 1] - plane) < 1e-6 * size)
        notch = float(p["notch"])
        if not 0 <= notch < width:
            raise ValueError("notch: from 0 to the width")
        cohesive = faces.subset(faces.centroids(opened)[:, 0] > notch)
        materials = {0: _material(p)}
        x = opened.nodes
        tol = 1e-9 * float(np.max(upper))
        top, bottom = np.flatnonzero(x[:, 1] > height - tol), np.flatnonzero(x[:, 1] < tol)
        prescribed = [(top, [0, 1, 2], [0.0, 1.0, 0.0]), (bottom, [0, 1, 2], [0.0, -1.0, 0.0])]
        component, factor, span = 1, 2.0, height
    else:
        length, width = float(p["beam_length"]), float(p["beam_width"])
        arm, precrack = float(p["arm"]), float(p["precrack"])
        size = float(p.get("element_size") or 1.0)
        opened, cohesive, sets = dcb_specimen(length, width, arm, precrack, size)
        lower, upper = np.zeros(3), np.array([length, width, 2 * arm])
        materials = {0: _material(p)}
        top, bottom = sets["top"], sets["bottom"]
        prescribed = [(top, 2, 1.0), (bottom, 2, -1.0), (top, [0, 1], 0.0),
                      (bottom, [0, 1], 0.0)]  # fmt: skip
        component, factor, span = 2, 2.0, 2 * arm
    young = _stiffest(materials.values())
    law = _law(p, young, size, beam=kind == "dcb")
    strength, energy, failure = _law_values(law)
    load = p.get("displacement")
    if load is None:  # twice the elastic stretch at the strength and the opening at failure
        if kind == "dcb":
            load = 3.0 * (float(p["beam_length"]) / 60.0)
        elif kind == "ligament":
            load = 2.0 * (strength / young * span + failure) / factor
        else:
            load = 2.0 * (strength / young * span + 2.0 * failure) / factor
    steps = int(p["steps"])
    viscosity = p.get("regularization")
    viscosity = 1e-3 * load if viscosity is None else float(viscosity)
    problem = CohesiveProblem(opened, cohesive, materials, law, kinematics=p["kinematics"])
    _report_newton(problem, ctx, float(load))
    ctx.progress(0.1, f"solving: {opened.n_elements} tetrahedra, {cohesive.n_elements} "
                      "interface elements")  # fmt: skip
    result = problem.run(prescribed, float(load), steps=steps, viscosity=viscosity, fields=True)
    ctx.progress(0.9, "figures")
    opening = factor * result.loads
    force = result.reactions[:, 0, component]
    rows = result.rows()
    for row, d in zip(rows, opening, strict=True):
        row["displacement"] = float(d)
    ctx.table(rows, "steps", "the steps: load factor, reactions, cracked area, works")
    area = float(cohesive.areas(opened).sum())
    work = {k: float(v[-1]) for k, v in result.work.items()}
    if problem.bulk is None:  # an elastic bulk stores the work of its forces
        stored = work.get("bulk", math.nan)
    else:
        stored = float(problem.bulk.free_energy() @ problem.bulk.wts)
    cracked = float(result.cracked[-1])
    values = {
        "elements": opened.n_elements,
        "interface_elements": cohesive.n_elements,
        "interface_area": area,
        "peak_force": float(np.max(force)),
        "displacement_at_peak": float(opening[int(np.argmax(force))]),
        "cracked_area": cracked,
        "broken_fraction": cracked / area if area > 0 else 0.0,
        "work_external": work.get("external", math.nan),
        "work_interfaces": work.get("interfaces", math.nan),
        "work_bulk": work.get("bulk", math.nan),
        "stored_bulk": stored,
        "dissipated_bulk": work.get("bulk", math.nan) - stored,
        "cohesive_zone_length": cohesive_zone_length(young, energy, strength),
        "zone_elements": cohesive_zone_length(young, energy, strength) / size,
        "strength_for_3_elements": turon_strength(young, energy, size, 3),
        "G_c": energy,
    }
    if cracked > 0:  # G_c, the process zones of unbroken interfaces, energy released unstably
        values["work_interfaces_per_cracked_area"] = work.get("interfaces", math.nan) / cracked
    ys = {"force": force}
    title = f"{cohesive.n_elements} interface elements, {opened.n_elements} tetrahedra"
    if kind == "dcb":
        beam, _ = dcb_beam_theory(materials[0].young, materials[0].poisson,
                                  float(p["beam_width"]), float(p["arm"]), float(p["precrack"]),
                                  energy, opening)  # fmt: skip
        ys["corrected beam theory"] = beam
        values["peak_beam_theory"] = float(np.max(beam))
    curve = _curves(opening, ys, "displacement" if kind == "interfaces" else "opening", "force",
                    title, styles={"force": "o-", "corrected beam theory": "-"},
                    twin=("cracked area", result.cracked))  # fmt: skip
    ctx.figure(curve, "force", "force-displacement")
    energies = {f"work of the {k}": v for k, v in result.work.items()}
    ctx.figure(_curves(opening, energies, "displacement", "energy", "works: of the loads = "
                       "of the bulk + of the interfaces"), "energies", "energies")  # fmt: skip
    u = result.displacement
    scale = _magnification([result], lower, upper, 0.04)
    ctx.figure(_rgb(_cohesive_figure(opened, cohesive, u, result.damage.mean(axis=1), scale,
                                     f"broken interfaces (displacements x {scale:.3g})")),
               "interfaces", "broken interfaces")  # fmt: skip
    ctx.progress(0.95, "frames of the opening")
    chosen = np.unique(np.linspace(0, len(result.fields) - 1, min(24, len(result.fields)))
                       .astype(int))  # fmt: skip
    ctx.video(_opening_frames(opened, cohesive, law, [result.fields[k] for k in chosen], scale,
                              [f"displacement {opening[k]:.3g}, force {force[k]:.3g}"
                               for k in chosen]),
              "opening", "the interfaces opening", fps=8, mode="once",
              wanted=p["video"])  # fmt: skip
    problem.write(ctx.path("cohesive"), result)
    if names:
        values["phases"] = ", ".join(f"{k}: {v}" for k, v in sorted(names.items()))
    return {"values": values}


def _opening_frames(mesh, faces, law, fields, scale: float, captions) -> list[np.ndarray]:
    """The frames of a cohesive problem: the deformed mid-section of the bulk and the traces
    of the interfaces coloured by their damage (the largest so far)."""
    moved = np.concatenate([mesh.nodes + scale * np.asarray(u) for u in fields])
    pad = 0.02 * float(np.ptp(mesh.nodes, axis=0).max())
    window = ((moved[:, 0].min() - pad, moved[:, 0].max() + pad),
              (moved[:, 1].min() - pad, moved[:, 1].max() + pad))  # fmt: skip
    labels = np.zeros(mesh.n_elements)
    video = _Frames(window, (-0.5, 0.5), "", cmap=_labels_cmap(1),
                    lines=("inferno_r", (0.0, 1.0), "damage of the interfaces"))  # fmt: skip
    cut = section_cut(mesh)
    at = 0.5 * (mesh.nodes[:, 2].min() + mesh.nodes[:, 2].max())
    worst = np.zeros(faces.n_elements)
    frames = []
    for u, caption in zip(fields, captions, strict=True):
        worst = np.maximum(worst, _damage_now(law, mesh, faces, u))
        polygons, owners = section(mesh, u, scale, cut=cut)
        segments, rows = _interface_segments(mesh, faces, u, scale, at)
        frames.append(video.frame(polygons, labels[owners], caption, segments, worst[rows]))
    return frames


# -- polycrystals ----------------------------------------------------------------------------------


@dataclass
class Grains:
    """The grains of a tessellation for the crack meshes: the cell of points and the crystal
    orientation of each cell (:func:`~morphogen.fracture.lefm.with_grains`)."""

    tessellation: Any
    orientations: np.ndarray

    def cell_of(self, points) -> np.ndarray:
        return self.tessellation.cell_of(points)


def _grains_of(item: dict, seed) -> Grains:
    micro = item["micro"]
    if not hasattr(micro, "cell_of") or not hasattr(micro, "germs"):
        raise ValueError(f"{item['title']}: a polycrystal, a Voronoi or Laguerre tessellation of "
                         "the workspace")  # fmt: skip
    orientations = getattr(micro, "orientations", None)
    if orientations is None:
        from morphogen.orientations import random_orientations

        rng = np.random.default_rng(seed if seed is not None else item.get("seed"))
        orientations = random_orientations(rng, len(micro.germs))
    return Grains(micro, np.asarray(orientations, dtype=float))


def _grain_map(grains: Grains, lower, upper, at: float, cracks, title: str, count: int = 360):
    """The grains of the plane z = ``at`` (their cells on a grid), their boundaries, the path
    of the crack and its successive tips."""
    from morphogen.viz.png import label_colors

    lengths = np.asarray(upper) - np.asarray(lower)
    n = np.maximum(2, np.rint(count * lengths[:2] / lengths[:2].max())).astype(int)
    xs = lower[0] + (np.arange(n[0]) + 0.5) * lengths[0] / n[0]
    ys = lower[1] + (np.arange(n[1]) + 0.5) * lengths[1] / n[1]
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, at)])
    cells = grains.cell_of(points).reshape(n[0], n[1])
    rgb = label_colors(cells).astype(float) / 255.0
    edge = np.zeros(cells.shape, dtype=bool)
    edge[1:, :] |= cells[1:, :] != cells[:-1, :]
    edge[:, 1:] |= cells[:, 1:] != cells[:, :-1]
    rgb[edge] = 0.15
    fig = _canvas(6.4, float(np.clip(6.4 * lengths[1] / lengths[0], 3.5, 8.0)))
    ax = fig.subplots()
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", interpolation="nearest",
              extent=(lower[0], upper[0], lower[1], upper[1]))  # fmt: skip
    path = _path_xy(cracks[-1])
    tips = np.array([_tips_xy(c)[0] for c in cracks])
    ax.plot(path[:, 0], path[:, 1], "-", color="white", lw=3.2)
    ax.plot(path[:, 0], path[:, 1], "-", color="crimson", lw=1.8)
    ax.plot(tips[:, 0], tips[:, 1], "o", color="crimson", ms=4, mec="white", mew=0.6)
    for tip in tips:
        ax.annotate(str(int(grains.cell_of(np.array([[tip[0], tip[1], at]]))[0])), tip,
                    textcoords="offset points", xytext=(4, 4), fontsize=7)  # fmt: skip
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=9)
    return fig


@tool(
    "polycrystal_crack",
    "fracture",
    "Polycrystal crack growth",
    "An edge crack grows through the grains of a polycrystal of the workspace (a Voronoi or "
    "Laguerre tessellation: a slab of it, pulled by grips in plane strain), each grain a cubic "
    "crystal turned by its orientation, elastic or slipping on its systems (Méric-Cailletaud "
    "crystal plasticity, in small or finite strains): the body is meshed again by MMG at each "
    "advance, its elements taking the grain of their centroids (staircase grain boundaries), "
    "and the state of the points moved grain by grain; the front advances once the driving "
    "force at the tip (J_tip, the configurational forces of the slips) reaches the toughness, "
    "kinking after Erdogan & Sih (the Voigt-Reuss-Hill moduli of the crystal). The path over "
    "the grains, the loads and driving forces of each advance, the resistance curve and the "
    "frames of the growth (their video on demand).",
    [
        Field(
            "source",
            "item",
            label="polycrystal",
            required=True,
            needs=("grains",),
            doc="a Voronoi or Laguerre tessellation (its orientations, or random ones)",
        ),
        Field(
            "thickness",
            "float",
            unit="length",
            doc="of the slab about the middle of z (blank: the whole depth, at most a "
            "sixteenth of the width)",
        ),
        *_crystal_fields(),
        Field(
            "plasticity",
            "choice",
            "crystal",
            choices=(
                ("crystal", "crystal plasticity (slip systems)"),
                ("elastic", "crystal elasticity only"),
            ),
        ),
        Field(
            "families",
            "choices",
            ["octahedral"],
            when={"plasticity": "crystal"},
            choices=(
                ("octahedral", "octahedral {111}<110> (FCC)"),
                ("cube", "cube {100}<110>"),
                ("bcc110", "BCC {110}<111>"),
                ("bcc112", "BCC {112}<111>"),
            ),
        ),
        Field(
            "tau0",
            "float",
            60.0,
            label="τ0",
            unit="stress",
            when={"plasticity": "crystal"},
            doc="critical resolved shear stress",
        ),
        Field(
            "Q",
            "float",
            40.0,
            unit="stress",
            when={"plasticity": "crystal"},
            doc="isotropic hardening (Q, b)",
        ),
        Field("b", "float", 5.0, when={"plasticity": "crystal"}),
        Field(
            "K",
            "float",
            5.0,
            unit="stress",
            when={"plasticity": "crystal"},
            doc="Norton's viscosity (K, n)",
        ),
        Field("n", "float", 20.0, when={"plasticity": "crystal"}),
        Field("length", "float", unit="length", doc="of the crack (blank: 0.3 W)"),
        Field("position", "float", unit="length", doc="its height y (blank: the middle)"),
        Field("angle", "float", 0.0, unit="degree"),
        Field("steps", "int", 3, doc="advances of the front"),
        Field("increment", "float", unit="length", doc="blank: W / 8"),
        Field(
            "toughness",
            "float",
            0.6,
            unit="energy / area",
            doc="J_tip at which the front advances (N/mm with MPa and mm)",
        ),
        Field(
            "strain_step",
            "float",
            2.5e-4,
            doc="of the nominal strain of the grips, between the checks of the criterion",
        ),
        Field("max_strain", "float", 7.5e-3, doc="the loading stops there"),
        Field(
            "strain_rate",
            "float",
            1.0,
            unit="1/time",
            when={"plasticity": "crystal"},
            doc="of the grips (the time of the viscous slips)",
        ),
        Field("kinematics", "choice", "small", choices=KINEMATICS),
        Field("element_size", "float", unit="length", doc="blank: W / 20"),
        Field("transfer", "choice", "nearest", choices=TRANSFERS),
        Field(
            "seed",
            "int",
            doc="of the random orientations of a tessellation without any "
            "(blank: the seed of the morphology)",
        ),
        Field("video", "bool", False, doc="a video of the growth; the export makes it too"),
    ],
    order=4,
    groups=(
        ("Polycrystal", "source"),
        ("Crack", "length"),
        ("Growth", "steps"),
    ),
    menu="Cracks",
)
def polycrystal_crack(p: dict, ctx) -> dict:
    from morphogen.fracture.ductile import propagate_ductile
    from morphogen.fracture.lefm import crack_mesh
    from morphogen.fracture.mixed import KinkedCrack

    item = p["source"]
    grains = _grains_of(item, p.get("seed"))
    domain = item["micro"].domain
    lower = np.asarray(domain.lower, dtype=float).copy()
    upper = np.asarray(domain.upper, dtype=float).copy()
    width, height = float(upper[0] - lower[0]), float(upper[1] - lower[1])
    depth = float(upper[2] - lower[2])
    thickness = p.get("thickness") or min(depth, width / 16.0)
    if not 0 < thickness <= depth * (1 + 1e-12):
        raise ValueError(f"thickness: from 0 to the depth of the polycrystal ({depth:g})")
    middle = 0.5 * (lower[2] + upper[2])
    lower[2], upper[2] = middle - 0.5 * thickness, middle + 0.5 * thickness
    size = _size(p.get("element_size") or width / 20.0, 1.0, "edge", width * height * thickness)
    a = float(p.get("length") or 0.3 * width)
    y = lower[1] + 0.5 * height if p.get("position") is None else float(p["position"])
    angle = math.radians(p.get("angle") or 0.0)
    mouth = np.array([lower[0], y, lower[2]])
    tip = mouth + a * np.array([math.cos(angle), math.sin(angle), 0.0])
    if not (a > 0 and lower[1] < y < upper[1] and np.all(tip[:2] > lower[:2])
            and np.all(tip[:2] < upper[:2])):  # fmt: skip
        raise ValueError("length, position, angle: the crack inside the polycrystal")
    crack = KinkedCrack((tuple(mouth), tuple(tip)))
    material = _cubic(p, plastic=p["plasticity"] == "crystal")
    steps = int(p["steps"])
    if steps < 1:
        raise ValueError("steps: at least one advance")
    increment = float(p.get("increment") or width / 8.0)
    tol = 1e-9 * max(width, height)

    def pulled(m):  # the grips move by ±H/2 per unit of the load factor: the nominal strain
        x = m.nodes
        top, bottom = x[:, 1] > upper[1] - tol, x[:, 1] < lower[1] + tol
        faces = (x[:, 2] < lower[2] + tol) | (x[:, 2] > upper[2] - tol)
        corner = _nearest(m, (upper[0], 0.5 * (lower[1] + upper[1]), lower[2]))
        u = 0.5 * height
        return [(np.flatnonzero(top & ~faces), 1, u), (np.flatnonzero(bottom & ~faces), 1, -u),
                (np.flatnonzero(top & faces), [1, 2], [u, 0.0]),
                (np.flatnonzero(bottom & faces), [1, 2], [-u, 0.0]),
                (np.flatnonzero(faces & ~top & ~bottom), 2, 0.0), (corner, 0, 0.0)]  # fmt: skip

    def meshed(c):
        return crack_mesh(lower, upper, c, size=size, grains=grains)

    mesh_of = _counting_distinct(meshed, ctx, steps)
    options = {}
    if p["plasticity"] == "crystal":
        options["duration"] = 1.0 / float(p.get("strain_rate") or 1.0)
    result = propagate_ductile(mesh_of, crack, material, pulled,
                               load_step=float(p["strain_step"]), increment=increment,
                               steps=steps, toughness=float(p["toughness"]),
                               max_load=float(p["max_strain"]), kinematics=p["kinematics"],
                               transfer=p["transfer"], keep=True, **options)  # fmt: skip
    cracks, solutions = result.cracks, result.solutions
    at = 0.5 * (lower[2] + upper[2])
    rows = result.rows()
    for row, c in zip(rows, cracks, strict=True):
        row["x_tip"], row["y_tip"] = float(c.tip[0]), float(c.tip[1])
        row["grain"] = int(grains.cell_of(np.array([[c.tip[0], c.tip[1], at]]))[0])
    for row in rows:
        row["strain"] = row.pop("load")
    ctx.table(rows, "advances", "the advances: strains, driving forces, kinks, grains of the tips")
    ctx.table(result.history, "loading", "every load step (the load: the nominal strain)")
    last = result.history[-1]
    met = last["J_tip"] >= float(p["toughness"])
    tips = [row["grain"] for row in rows]
    values = {
        "grains": len(grains.orientations),
        "advances": len(cracks) - 1,
        "final_length": cracks[-1].size,
        "last_strain": float(result.loads[-1]),
        "stopped": "after the last step" if met else "at the largest strain",
        "grains_crossed": len(set(tips)) - 1,
        "largest_kink_deg": float(np.degrees(np.abs(result.angles)).max()),
        "elements": solutions[-1].mesh.n_elements,
        "orientations": "of the tessellation"
        if getattr(item["micro"], "orientations", None) is not None
        else "random",
    }
    ctx.progress(0.88, "figures")
    ctx.figure(_rgb(_grain_map(grains, lower, upper, at, cracks,
                               f"the path over the grains (tips in grains {tips})")),
               "grains", "crack path over the grains")  # fmt: skip
    delta, far = result.resistance()
    ys = {"J_far (J_ψ, ring)": far, "J_tip": result.tip, "J_work": result.work}
    ctx.figure(_curves(delta, ys, "crack advance Δa", "driving force", "resistance curve",
                       lines={"toughness": float(p["toughness"])},
                       twin=("nominal strain", result.loads)), "resistance", "R-curve")  # fmt: skip
    captions = [f"advance {k}: strain {lam:.3g}, J_tip {jt:.3g}, grain {g}" for k, (lam, jt, g)
                in enumerate(zip(result.loads, result.tip, tips, strict=True))]  # fmt: skip
    _growth_outputs(ctx, lower, upper, size, cracks, solutions, captions,
                    f"{len(cracks)} cracks through the grains", video=p["video"])  # fmt: skip
    return {"values": values}
