"""Options of the FEniCSx solver that need no FEniCSx: configurations, errors."""

from __future__ import annotations

import numpy as np
import pytest

from morphogen import config
from morphogen.homogenization import fenicsx
from morphogen.homogenization.apparent import make_problem
from morphogen.homogenization.materials import Material
from morphogen.homogenization.solvers import METHODS, SOLVERS
from morphogen.image import VoxelImage


def test_solver_names():
    assert "fenicsx" in SOLVERS and "fenicsx" not in METHODS
    assert isinstance(fenicsx.available(), bool)
    with pytest.raises(ValueError, match="tetrahedral meshes"):
        make_problem(
            VoxelImage(np.zeros((4, 4, 4), np.uint8)),
            {0: Material(1.0, 0.3)},
            "elastic",
            solver="fenicsx",
        )


@pytest.mark.skipif(fenicsx.available(), reason="FEniCSx is installed")
def test_missing_fenicsx_is_explained():
    from helpers import box_mesh

    problem = make_problem(
        box_mesh((2, 2, 2)), {0: Material(1.0, 0.3)}, "elastic", "kubc", solver="fenicsx"
    )
    with pytest.raises(ImportError, match="conda install"):
        problem.solve(problem_loading())


def problem_loading():
    from morphogen.homogenization.core import Loading

    return Loading.strain([1e-3, 0, 0, 0, 0, 0])


def test_configuration_accepts_fenicsx():
    data = {"materials": {0: {"young": 1.0, "poisson": 0.3}}, "solver": "fenicsx"}
    assert config._homogenization(data).solver == "fenicsx"
    with pytest.raises(config.ConfigError, match="fenicsx"):
        config._homogenization({**data, "solver": "petsc"})
