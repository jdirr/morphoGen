"""Shared fixtures.

Statistical tests compare measurements on realizations with closed-form results.
Seeds are fixed so that the tests are deterministic; tolerances are a few standard
deviations of the estimator (computed from the theory where possible).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from morphogen import settings
from morphogen._platform import full_speed

# Tests run in parallel (pytest-xdist, `-n`): each worker takes its share of the cores for
# the compiled kernels and SciPy rather than all of them, or MORPHOGEN_TEST_THREADS
# threads (the local CI runs two Pythons at once).
_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
_threads = os.environ.get("MORPHOGEN_TEST_THREADS")
if _threads or _workers > 1:
    settings.set_num_threads(int(_threads or max(1, (os.cpu_count() or 1) // _workers)))

#: The longest files of tests (several minutes each), run first so that the workers of a
#: parallel run (`-n`, one file each: `--dist loadfile`) finish together
LONGEST = [
    "test_gui_tools_fracture.py",
    "test_toughness.py",
    "test_bifurcation.py",
    "test_ntfa.py",
    "test_micromorphic.py",
    "test_mixed.py",
    "test_ductile.py",
    "test_levelset_foam.py",
    "test_fracture_morphology.py",
    "test_fft_nonlinear.py",
    "test_conforming.py",
    "test_homogenization_pipeline.py",
    "test_fracture_inelastic.py",
    "test_polycrystal.py",
    "test_gui_behaviours.py",
]


def pytest_configure(config):
    # Windows moves the test processes started by an agent or a terminal out of focus to
    # its efficiency cores at a low clock (EcoQoS), and the setting that keeps them on
    # the performance cores is not inherited: this process and each pytest-xdist worker
    # make it (nothing elsewhere)
    full_speed()


def pytest_collection_modifyitems(config, items):
    rank = {name: i for i, name in enumerate(LONGEST)}
    items.sort(key=lambda item: rank.get(item.path.name, len(LONGEST)))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
