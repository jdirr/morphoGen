"""The screenshots of the tour of the graphical interface (docs/tutorial_interface.md).

    python docs/tutorials/make_screenshots.py

Starts ``morphogen gui`` on a free port, drives a small study in its page with
docs/tutorials/screenshots.js (Node.js and Playwright, with its Chromium: a developer's
tool, not a dependency of morphoGen) and keeps the screenshots in docs/_static/tutorial/
(interface_<step>.png, reduced to a palette of colours). A few minutes: the study runs a
loading path and a phase-field fracture.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "_static" / "tutorial"
SCRIPT = Path(__file__).with_name("screenshots.js")
#: Largest number of colours of a screenshot
COLOURS = 160


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def keep(source: Path, target: Path) -> int:
    """A screenshot copied with a palette of at most :data:`COLOURS` colours; its size."""
    from PIL import Image

    picture = Image.open(source).convert("RGB")
    picture = picture.quantize(COLOURS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    target.parent.mkdir(parents=True, exist_ok=True)
    picture.save(target, optimize=True)
    return target.stat().st_size


def main() -> int:
    node = shutil.which("node")
    if node is None:
        print("Node.js (with Playwright) drives the page: not found", file=sys.stderr)
        return 1
    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    command = [sys.executable, "-m", "morphogen", "gui", "--port", str(port), "--no-browser"]
    server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):  # until the server answers
            try:
                urllib.request.urlopen(url + "api/info", timeout=2).read()
                break
            except OSError:
                time.sleep(0.5)
        else:
            print("the interface did not start", file=sys.stderr)
            return 1
        with tempfile.TemporaryDirectory() as folder:
            done = subprocess.run([node, str(SCRIPT), url, folder], check=False)
            if done.returncode:
                return done.returncode
            for source in sorted(Path(folder).glob("interface_*.png")):
                size = keep(source, FIGURES / source.name)
                print(f"{source.name:28s} {size / 1024:6.1f} kB")
    finally:
        server.terminate()
        server.wait(timeout=20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
