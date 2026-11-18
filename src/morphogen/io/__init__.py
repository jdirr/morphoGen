"""Reading and writing images, meshes and geometries (format chosen by file extension).

=============  ======================================================================
Images         ``.vti`` (VTK XML), ``.vtk`` (legacy VTK), ``.mhd`` (MetaImage),
               ``.xdmf`` (XDMF with raw binary data), ``.npz`` (NumPy, round-trips
               metadata), ``.npy``, ``.tif`` (tifffile)
Meshes         ``.inp`` (Abaqus/CalculiX), ``.msh`` (Gmsh 2.2), ``.vtu``, ``.stl``,
               ``.mesh`` (Medit, tetrahedra), ``.xdmf`` (tetrahedra with their cell
               and facet tags for FEniCSx), other extensions through meshio
Geometries     ``.json`` (full description), ``.csv`` (tables), ``.geo`` (Gmsh script)
Solvers        periodic equations (Abaqus/CalculiX), DAMASK grids and materials, Neper
               seeds (:mod:`morphogen.io.solvers`); the runs of AMITEX_FFTP and their
               results (:mod:`morphogen.io.amitex`); MFront sources of the nonlinear
               materials (:mod:`morphogen.io.mfront`); the inputs of OpenRadioss, the
               explicit solver, and its listings and time histories
               (:mod:`morphogen.io.radioss`, run by :mod:`morphogen.dynamics`)
Orientations   ``.ang`` (EDAX/TSL) and ``.ctf`` (Oxford Instruments/HKL) orientation
               maps of EBSD, and their grains (:mod:`morphogen.io.ebsd`)
=============  ======================================================================

Images larger than memory are written slab by slab by :func:`voxelize_to_file`.
"""

from __future__ import annotations

from pathlib import Path

from morphogen.image import VoxelImage
from morphogen.io.amitex import (
    amitex_stiffness,
    read_amitex,
    read_amitex_materials,
    read_amitex_run,
    read_amitex_zones,
    write_amitex,
    write_amitex_field,
    write_amitex_loading,
)
from morphogen.io.ebsd import EBSDMap, read_ebsd, write_ebsd
from morphogen.io.export import export
from morphogen.io.geometry import read_json, write_csv, write_geo, write_json
from morphogen.io.meshes import (
    read_msh,
    read_stl,
    read_tet_mesh,
    write_inp,
    write_inp_tet,
    write_medit_tet,
    write_meshio,
    write_msh,
    write_msh_tet,
    write_periodic_pairs,
    write_stl,
    write_vtu_mesh,
    write_vtu_tet,
)
from morphogen.io.metaimage import read_mhd, write_mhd
from morphogen.io.mfront import mfront_source, write_mfront
from morphogen.io.native import read_npy, read_npz, write_npy, write_npz
from morphogen.io.radioss import read_radioss_listing, read_radioss_th, write_radioss
from morphogen.io.solvers import (
    write_damask,
    write_fenicsx,
    write_neper,
    write_periodic_equations,
)
from morphogen.io.tables import read_table, write_table, write_tables
from morphogen.io.tiff import read_tiff, write_tiff
from morphogen.io.vtk import (
    read_vti,
    read_vtk,
    read_vtk_grid,
    read_vtu,
    write_vti,
    write_vtk,
    write_vtu,
)
from morphogen.io.xdmf import read_xdmf, voxelize_to_file, write_xdmf
from morphogen.mesh.surface import SurfaceMesh
from morphogen.mesh.tetra import TetMesh
from morphogen.mesh.voxel import VoxelMesh
from morphogen.models.base import Microstructure

IMAGE_FORMATS = ("vti", "vtk", "mhd", "xdmf", "npz", "npy", "tif")
MESH_FORMATS = ("inp", "msh", "vtu")
GEOMETRY_FORMATS = ("json", "csv", "geo")


def _ext(path) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return "tif" if ext == "tiff" else ext


def save_image(images, path, **options) -> Path:
    """Write an image, or several images on the same grid (formats .vti and .vtk)."""
    ext = _ext(path)
    many = not isinstance(images, VoxelImage)
    if ext == "vti":
        return write_vti(path, images, **options)
    if ext == "vtk":
        return write_vtk(path, images, **options)
    if ext == "xdmf":
        return write_xdmf(path, images)
    if many:
        if len(images) != 1:
            raise ValueError(f".{ext} files hold a single array; use .vti or .vtk")
        images = images[0]
    writers = {"mhd": write_mhd, "npz": write_npz, "npy": write_npy, "tif": write_tiff}
    if ext not in writers:
        raise ValueError(f"unknown image format .{ext}; use one of {IMAGE_FORMATS}")
    return writers[ext](path, images, **options)


def load_image(path, name: str | None = None) -> VoxelImage:
    """Read an image (the array called ``name``, or the first one)."""
    ext = _ext(path)
    if ext in ("vti", "vtk"):
        images = read_vti(path) if ext == "vti" else read_vtk(path)
        if name is None:
            return images[0]
        for img in images:
            if img.name == name:
                return img
        raise ValueError(f"{path}: no array named {name!r}")
    if ext == "xdmf":
        return read_xdmf(path, name)
    readers = {"mhd": read_mhd, "npz": read_npz, "npy": read_npy, "tif": read_tiff}
    if ext not in readers:
        raise ValueError(f"unknown image format .{ext}; use one of {IMAGE_FORMATS}")
    return readers[ext](path)


def save_mesh(mesh, path, **options) -> Path:
    """A mesh written to a file whose extension gives the format: a surface mesh as .stl, a
    tetrahedral mesh as .inp, .msh, .vtu, .mesh or .xdmf (FEniCSx), a voxel mesh as .inp, .msh or
    .vtu, and any of them in the other formats of meshio."""
    ext = _ext(path)
    if isinstance(mesh, SurfaceMesh):
        if ext == "stl":
            return write_stl(path, mesh, **options)
        return write_meshio(path, mesh)
    if isinstance(mesh, TetMesh):
        writers = {
            "inp": write_inp_tet,
            "msh": write_msh_tet,
            "vtu": write_vtu_tet,
            "mesh": write_medit_tet,
            "xdmf": write_fenicsx,
        }
        if ext in writers:
            return writers[ext](path, mesh, **options)
        return write_meshio(path, mesh)
    if not isinstance(mesh, VoxelMesh):
        raise TypeError("save_mesh expects a VoxelMesh, a TetMesh or a SurfaceMesh")
    writers = {"inp": write_inp, "msh": write_msh, "vtu": write_vtu_mesh}
    if ext in writers:
        return writers[ext](path, mesh, **options)
    return write_meshio(path, mesh)


def save_microstructure(micro: Microstructure, path, **options):
    """Write the explicit geometry of a realization (.json, .csv tables or .geo script)."""
    ext = _ext(path)
    if ext == "json":
        return write_json(path, micro)
    if ext == "csv":
        return write_csv(path, micro)
    if ext == "geo":
        return write_geo(path, micro, **options)
    raise ValueError(f"unknown geometry format .{ext}; use one of {GEOMETRY_FORMATS}")


def load_microstructure(path) -> Microstructure:
    """A microstructure read back from the .json file that :func:`save_microstructure` wrote."""
    if _ext(path) != "json":
        raise ValueError("microstructures are read back from .json files")
    return read_json(path)


__all__ = [
    "GEOMETRY_FORMATS",
    "IMAGE_FORMATS",
    "MESH_FORMATS",
    "EBSDMap",
    "amitex_stiffness",
    "export",
    "load_image",
    "load_microstructure",
    "mfront_source",
    "read_amitex",
    "read_amitex_materials",
    "read_amitex_run",
    "read_amitex_zones",
    "read_ebsd",
    "read_json",
    "read_mhd",
    "read_msh",
    "read_npy",
    "read_npz",
    "read_radioss_listing",
    "read_radioss_th",
    "read_stl",
    "read_table",
    "read_tet_mesh",
    "read_tiff",
    "read_vti",
    "read_vtk",
    "read_vtk_grid",
    "read_vtu",
    "read_xdmf",
    "save_image",
    "save_mesh",
    "save_microstructure",
    "voxelize_to_file",
    "write_amitex",
    "write_amitex_field",
    "write_amitex_loading",
    "write_csv",
    "write_damask",
    "write_ebsd",
    "write_fenicsx",
    "write_geo",
    "write_inp",
    "write_inp_tet",
    "write_json",
    "write_medit_tet",
    "write_mfront",
    "write_mhd",
    "write_msh",
    "write_msh_tet",
    "write_neper",
    "write_npy",
    "write_npz",
    "write_periodic_equations",
    "write_periodic_pairs",
    "write_radioss",
    "write_stl",
    "write_table",
    "write_tables",
    "write_tiff",
    "write_vti",
    "write_vtk",
    "write_vtu",
    "write_vtu_tet",
    "write_xdmf",
]
