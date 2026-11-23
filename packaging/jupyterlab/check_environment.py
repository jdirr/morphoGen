"""Checks an environment of the JupyterLab installers, run with its own Python, as installed:

    <environment>/python/python.exe check_environment.py <environment>     (Windows)
    <environment>/python/bin/python3 check_environment.py <environment>    (macOS)

The packages import from the environment and not from elsewhere; its programs run from where
it is (``morphogen info``, ``jupyter --version``); morphoGen computes (a realization, the FFT
solver, a conforming mesh and the finite elements under kinematic and periodic conditions, a
plastic loading path); the first notebook runs through the kernel of the environment; and the
launcher opens JupyterLab on a folder of notebooks of its own, made with the first notebook,
whose server answers. Exits with 0 when all went well.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PROBLEMS: list[str] = []


def check(ok: bool, message: str) -> bool:
    print(("ok:      " if ok else "problem: ") + message, flush=True)
    if not ok:
        PROBLEMS.append(message)
    return ok


def programs(environment: Path) -> None:
    """The programs of the packages, through the PATH that the launcher gives JupyterLab."""
    python = Path(sys.executable).parent
    folders = [python, python / "Scripts"] if os.name == "nt" else [python]
    env = dict(os.environ, PATH=os.pathsep.join([*map(str, folders), os.environ["PATH"]]))
    for command in (["morphogen", "info"], ["jupyter", "--version"], ["pip", "--version"]):
        done = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=600, shell=os.name == "nt"
        )
        check(done.returncode == 0, f"the program {' '.join(command)} runs from {environment}")
        if done.returncode:
            print(done.stdout[-2000:], done.stderr[-2000:])


def computations() -> None:
    import numpy as np

    import morphogen as mg
    from morphogen.homogenization import LoadingPath, Material, homogenize

    phases = {
        0: Material(young=3.0, poisson=0.35, conductivity=0.2),
        1: Material(young=70.0, poisson=0.22, conductivity=1.0),
    }
    model = mg.models.HardcoreSpheres(radius=1.0, volume_fraction=0.2, min_gap=0.1)
    micro = model.generate(mg.Domain.cube(5.0, periodic=True), seed=1)
    start = time.perf_counter()
    fft = homogenize(micro.voxelize(shape=20), phases, properties=["bulk_modulus", "conductivity"])
    k_fft = fft["bulk_modulus.periodic"].value
    check(
        3.33 < k_fft < 41.7,
        f"the FFT solver: k = {k_fft:.3f} ({time.perf_counter() - start:.0f} s)",
    )
    check(mg.mesh.conforming.has_mmg(), "the mesher MMG of pymmg is found")
    start = time.perf_counter()
    mesh = mg.mesh.conforming_mesh(micro, size=1.0, hausdorff=0.2, min_size=0.2)
    fe = homogenize(mesh, phases, bcs=["kubc", "periodic"], properties=["bulk_modulus"])
    k_kubc, k_pbc = fe["bulk_modulus.kubc"].value, fe["bulk_modulus.periodic"].value
    check(
        k_kubc >= k_pbc > 3.33,
        f"the finite elements: k = {k_kubc:.3f} (KUBC) >= {k_pbc:.3f} "
        f"(PBC) on {len(mesh.elements)} tetrahedra ({time.perf_counter() - start:.0f} s)",
    )
    plastic = {
        0: Material(young=70e3, poisson=0.33, plasticity={"yield": 100.0, "hardening": 500.0}),
        1: Material(young=400e3, poisson=0.2),
    }
    path = LoadingPath.parse("t=0,10 n=5 E11=0,0.01")
    curve = homogenize(micro.voxelize(shape=12), plastic, properties=[], paths={"t": path})
    stress = curve["t.periodic"].stress[:, 0]
    check(
        bool(np.all(np.diff(stress) > 0)) and stress[-1] > 100.0,
        f"a plastic loading path: Sigma_11 = {stress[-1]:.0f} MPa at 1 %",
    )


def notebook(environment: Path, folder: Path) -> None:
    """The first notebook, run by the kernel of the environment."""
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    spec = KernelSpecManager().get_kernel_spec("python3")
    check(
        "launch_kernel.py" in " ".join(spec.argv)
        and Path(spec.resource_dir).resolve().is_relative_to(environment),  # long names (8.3)
        f"the kernel python3 is that of the environment ({spec.resource_dir})",
    )
    nb = nbformat.read(environment / "welcome.ipynb", as_version=4)
    start = time.perf_counter()
    try:
        NotebookClient(
            nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(folder)}}
        ).execute()
        figures = sum(
            1 for c in nb.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})
        )
        check(
            figures >= 1,
            f"the first notebook runs ({time.perf_counter() - start:.0f} s, {figures} figure)",
        )
    except Exception as error:  # the error of the cell that failed
        check(False, f"the first notebook fails: {error}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def jupyterlab(environment: Path, folder: Path) -> None:
    """The launcher, as the shortcut starts it, without a browser: the folder of the notebooks
    is made with the first notebook, and the server answers."""
    port, token = _free_port(), "check-environment"
    notebooks = folder / "notebooks"
    env = dict(os.environ, MORPHOGEN_NOTEBOOKS=str(notebooks))
    log = folder / "jupyterlab.log"
    root = ["--allow-root"] if hasattr(os, "geteuid") and os.geteuid() == 0 else []  # containers
    with log.open("w") as out:
        server = subprocess.Popen(
            [
                sys.executable,
                str(environment / "launcher.py"),
                "--no-browser",
                *root,
                f"--port={port}",
                f"--IdentityProvider.token={token}",
            ],
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        try:
            status = None
            for _ in range(240):
                if server.poll() is not None:
                    break
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/status",
                        headers={"Authorization": f"token {token}"},
                    )
                    with urllib.request.urlopen(request, timeout=5) as answer:
                        status = json.load(answer)
                    break
                except OSError:
                    time.sleep(0.5)
            check(status is not None, f"JupyterLab answers on port {port} (log: {log})")
            if status is not None:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/lab", headers={"Authorization": f"token {token}"}
                )
                with urllib.request.urlopen(request, timeout=30) as answer:
                    check(answer.status == 200 and b"JupyterLab" in answer.read(), "its page /lab")
            check(
                (notebooks / "welcome.ipynb").is_file(), f"the folder of the notebooks {notebooks}"
            )
        finally:
            server.terminate()
            server.wait(timeout=60)
    if PROBLEMS:
        print(log.read_text(errors="replace")[-3000:])


def main() -> int:
    environment = Path(sys.argv[1]).resolve()
    import morphogen
    from morphogen._platform import full_speed

    full_speed()  # Windows: not throttled as a process in the background
    where = Path(morphogen.__file__).resolve()
    check(
        where.is_relative_to(environment), f"morphoGen {morphogen.__version__} from {where.parent}"
    )
    check(
        Path(sys.executable).resolve().is_relative_to(environment),
        f"Python {sys.version.split()[0]}",
    )
    with tempfile.TemporaryDirectory(prefix="morphogen_check_") as tmp:
        folder = Path(tmp)
        programs(environment)
        computations()
        notebook(environment, folder)
        jupyterlab(environment, folder)
    print(f"{len(PROBLEMS)} problem(s)")
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    sys.exit(main())
