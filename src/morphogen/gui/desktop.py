"""The stand-alone application of morphoGen: the graphical interface in a window of its
own rather than a tab of the web browser.

``morphogen desktop`` (or ``morphogen-desktop``, an application of the system once
installed; a bundle that needs no Python with ``packaging/desktop``) starts the server of
the interface on a free port of the local machine, private to the application, and opens
the page in a window of the web engine of the system through pywebview (WebView2 on
Windows, WebKit on macOS, Qt WebEngine or WebKitGTK on Linux: ``pip install
'morphogen[desktop]'``). The page then uses the dialogs of the system
(:class:`DesktopApi`, ``window.pywebview.api`` in the page): the folders of the workspaces
it saves and opens, the folders and files that the forms of the tools take (their
**Browse…** buttons), the files of the results saved where one chooses and opened by
their applications, the folder of a result shown in the file manager. A menu holds the
workspace, the folder of the session and the help; closing the window ends the session
(its files with it, unless saved; the computations still running stopped, the programs
they run killed), after a confirmation when the workspace holds items not saved.

Without pywebview, the application falls back on the web browser, as ``morphogen gui``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import webbrowser
from pathlib import Path

from morphogen import __version__

#: The address of the project, for the menu Help
REPOSITORY = "https://github.com/jdirr/morphoGen"
#: The size of the window (logical pixels) and its smallest size
SIZE = (1440, 920)
MIN_SIZE = (960, 640)


def icon_path() -> Path | None:
    """The icon of the application (a PNG of the package)."""
    from importlib import resources

    path = Path(str(resources.files("morphogen") / "gui" / "icon.png"))
    return path if path.is_file() else None


def open_with_system(path) -> bool:
    """Open a file or a folder with the application the system gives it."""
    path = str(path)
    try:
        if sys.platform == "win32":
            os.startfile(path)  # the user's own file
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(
                ["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except (OSError, AttributeError):
        return False
    return True


def reveal_in_system(path) -> bool:
    """Show a file in the file manager (its folder, the file selected where the system
    can), or open a folder."""
    path = Path(path)
    try:
        if sys.platform == "win32" and path.is_file():
            subprocess.Popen(["explorer", "/select,", str(path)])
            return True
        if sys.platform == "darwin" and path.is_file():
            subprocess.Popen(["open", "-R", str(path)])
            return True
    except OSError:
        return False
    return open_with_system(path if path.is_dir() else path.parent)


class DesktopApi:
    """What the page of the application asks of the system (``window.pywebview.api``):
    dialogs, files and folders. ``session`` is the session of the server
    (:class:`morphogen.gui.api.Session`); the window is set once created. Each method
    returns plain values (a path, None when the dialog is cancelled, a boolean)."""

    def __init__(
        self,
        session,
        window=None,
        *,
        dialogs=None,
        opener=open_with_system,
        revealer=reveal_in_system,
    ):
        self._session = session
        self._window = window
        self._dialogs = dialogs  # the module of the constants of the dialogs (webview)
        self._open = opener
        self._reveal = revealer
        self._saved = None  # the number given to the last item when the workspace was saved

    def _dialog(self, kind: str, title: str = "", start: str = "", name: str = "", types=()):
        if self._window is None:
            return None
        dialog = getattr(self._dialogs.FileDialog, kind)
        start = str(Path(start).expanduser()) if start else str(Path.home())
        options = {"directory": start, "allow_multiple": False}
        if kind == "SAVE":
            options["save_filename"] = name
        if types:
            options["file_types"] = tuple(types)
        chosen = self._window.create_file_dialog(dialog, **options)
        if not chosen:
            return None
        return str(chosen[0] if isinstance(chosen, (list, tuple)) else chosen)

    def info(self) -> dict:
        """The application and its session, for the page."""
        return {
            "desktop": True,
            "version": __version__,
            "session": str(self._session.root),
            "platform": sys.platform,
        }

    def choose_folder(self, title: str = "", start: str = "") -> str | None:
        """A folder chosen in the dialog of the system."""
        return self._dialog("FOLDER", title, start)

    def choose_file(self, title: str = "", start: str = "") -> str | None:
        """A file chosen in the dialog of the system."""
        return self._dialog("OPEN", title, start)

    def _file(self, url: str) -> Path:
        """The file of the session at the address of the page ``/files/<key>/<name>``."""
        path = urllib.parse.unquote(urllib.parse.urlparse(str(url)).path)
        parts = path.split("/", 3)
        if len(parts) != 4 or parts[1] != "files":
            raise ValueError(f"not a file of the session: {url}")
        return self._session.file(parts[2], parts[3])

    def save_file(self, url: str) -> str | None:
        """Copy a file of the session (its address in the page) where the dialog of the
        system says; the path written."""
        source = self._file(url)
        start = str(Path.home() / "Downloads") if (Path.home() / "Downloads").is_dir() else ""
        target = self._dialog("SAVE", "Save the file", start, name=source.name)
        if not target:
            return None
        shutil.copyfile(source, target)
        return target

    def open_file(self, url: str) -> bool:
        """Open a file of the session (a figure, a video, a table) with its application."""
        return self._open(self._file(url))

    def reveal(self, key: str) -> bool:
        """Show the folder of an item of the workspace (a result, a simulation) in the file
        manager."""
        folder = (self._session.root / str(key)).resolve()
        if self._session.root.resolve() not in folder.parents or not folder.is_dir():
            raise ValueError(f"no folder of the session for {key!r}")
        return self._reveal(folder)

    def open_session_folder(self) -> bool:
        """Show the folder of the session (every result of the session) in the file manager."""
        return self._reveal(self._session.root)

    def workspace_saved(self) -> None:
        """The page saved the workspace (closing then asks nothing, until another item
        joins it)."""
        self._saved = self._session._count

    def unsaved(self) -> bool:
        """Whether the workspace holds items, some joined since it was last saved."""
        return bool(self._session.items) and self._session._count != self._saved


def _menu(webview, window, api: DesktopApi) -> list:
    """The menus of the window: the workspace, the session, the help."""
    try:
        from webview.menu import Menu, MenuAction, MenuSeparator
    except ImportError:  # pragma: no cover - older pywebview
        return []

    def click(element: str):
        return lambda: window.evaluate_js(f"document.getElementById({element!r}).click()")

    def about():
        window.evaluate_js(
            f"alert('morphoGen {__version__}: morphological modelling and generation of virtual "
            "microstructures, their homogenization, fracture and explicit dynamics. "
            "GPL-3.0-or-later.')"
        )

    return [
        Menu(
            "File",
            [
                MenuAction("Open a workspace…", click("ws-open")),
                MenuAction("Save the workspace…", click("ws-save")),
                MenuSeparator(),
                MenuAction("Show the folder of the session", api.open_session_folder),
                MenuSeparator(),
                MenuAction("Quit", window.destroy),
            ],
        ),
        Menu(
            "Help",
            [
                MenuAction("What each tab does", click("help-toggle")),
                MenuAction("morphoGen on GitHub", lambda: webbrowser.open(REPOSITORY)),
                MenuAction(
                    "OpenRadioss on GitHub",
                    lambda: webbrowser.open("https://github.com/OpenRadioss/OpenRadioss"),
                ),
                MenuSeparator(),
                MenuAction("About morphoGen", about),
            ],
        ),
    ]


def stop_jobs(server, timeout: float = 10.0) -> int:
    """Stop the jobs still running on a server (at their next checkpoint: the programs
    they run, OpenRadioss, are killed) and wait for them, ``timeout`` seconds at most;
    the jobs stopped."""
    import time

    with server.jobs_lock:
        running = [j for j in server.jobs.values() if j.status()["state"] == "running"]
    for job in running:
        job.stop()
    end = time.monotonic() + timeout
    for job in running:
        thread = getattr(job, "thread", None)
        if thread is not None and thread.is_alive():
            thread.join(max(0.0, end - time.monotonic()))
    return len(running)


def start_server(session_dir):
    """The server of the interface on a free port of 127.0.0.1, serving from a thread;
    returns it and its address."""
    from morphogen.gui.server import GuiServer

    server = GuiServer(("127.0.0.1", 0), Path(session_dir))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def check(session_dir=None) -> int:
    """Check that the application runs, without a window (for the bundles): the server
    serves the page and its API, and a model is generated (the compiled kernels). Prints
    what it checked; 0 when all went well."""
    import json
    import urllib.request

    with tempfile.TemporaryDirectory(prefix="morphogen_desktop_") as tmp:
        server, url = start_server(session_dir or tmp)
        try:
            page = urllib.request.urlopen(url, timeout=30).read()
            if b"morphoGen" not in page:
                raise RuntimeError("the page is not served")
            info = json.loads(urllib.request.urlopen(url + "api/info", timeout=30).read())
            models = json.loads(urllib.request.urlopen(url + "api/models", timeout=60).read())
            body = json.dumps(
                {
                    "model": "boolean_spheres",
                    "params": {"radius": 1.0, "volume_fraction": 0.3},
                    "domain": {"size": 8, "periodic": True},
                    "seed": 1,
                    "resolution": 64,
                }
            ).encode()
            request = urllib.request.Request(
                url + "api/generate", data=body, headers={"Content-Type": "application/json"}
            )
            generated = json.loads(urllib.request.urlopen(request, timeout=600).read())
            if "error" in generated:
                raise RuntimeError(generated["error"])
            catalog = json.loads(urllib.request.urlopen(url + "api/tools", timeout=60).read())
        except Exception as exc:  # reported, the check failed
            print(f"morphoGen {__version__}: the check failed: {type(exc).__name__}: {exc}")
            return 1
        finally:
            server.shutdown()
            server.server_close()
    tabs = ", ".join(t["title"] for t in catalog["tabs"])
    print(
        f"morphoGen {info['version']}: the page, {len(models)} models and the tabs {tabs}; a "
        "realization generated"
    )
    return 0


def run(
    session_dir=None, *, debug: bool = False, browser: bool = False, gui: str | None = None
) -> int:
    """Run the application until its window closes (see the module); ``browser``: in the
    web browser instead (as ``morphogen gui``); ``gui``: the renderer of pywebview (``qt``,
    ``gtk``, ``edgechromium``, ``cocoa``; by default the one of the system)."""
    try:
        if browser:
            raise ImportError("asked for the browser")
        import webview
    except ImportError as exc:
        from morphogen.gui.server import serve

        if not browser:
            print(
                f"no window ({exc}): pip install 'morphogen[desktop]' for the application; "
                "the interface opens in the web browser instead"
            )
        serve(0, open_browser=True, session_dir=session_dir)
        return 0
    temporary = None
    if session_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="morphogen_desktop_")
        session_dir = temporary.name
    server, url = start_server(session_dir)
    api = DesktopApi(server.session, dialogs=webview)
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    webview.settings["ALLOW_DOWNLOADS"] = True
    options = {
        "js_api": api,
        "width": SIZE[0],
        "height": SIZE[1],
        "min_size": MIN_SIZE,
        "text_select": True,
        "background_color": "#f6f6f2",
    }
    window = webview.create_window(f"morphoGen {__version__}", url, **options)
    api._window = window
    try:
        window.menu = _menu(webview, window, api)  # read when the window is shown
    except (AttributeError, TypeError):  # pragma: no cover - older pywebview
        pass

    def closing():
        if not api.unsaved():
            return True
        return window.create_confirmation_dialog(
            "Quit morphoGen",
            "The items of the workspace are lost with the session unless saved (Save… beside "
            "the workspace). Quit anyway?",
        )

    window.events.closing += closing
    icon = icon_path()
    try:
        webview.start(gui=gui, debug=debug, icon=str(icon) if icon else None)
    finally:
        stop_jobs(server)
        server.shutdown()
        server.server_close()
        if temporary is not None:
            temporary.cleanup()
    return 0


def main(argv=None) -> int:
    """The program ``morphogen-desktop``: the application (``--check``: check it without a
    window)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="morphogen-desktop", description="morphoGen, the stand-alone application"
    )
    parser.add_argument("--version", action="version", version=f"morphoGen {__version__}")
    parser.add_argument("--session", help="folder of the files of the session")
    parser.add_argument("--debug", action="store_true", help="the developer tools of the page")
    parser.add_argument("--browser", action="store_true", help="in the web browser")
    parser.add_argument("--gui", help="the renderer of pywebview (qt, gtk, edgechromium, cocoa)")
    parser.add_argument("--check", action="store_true", help="check the application, no window")
    args = parser.parse_args(argv)
    if args.check:
        return check(args.session)
    return run(args.session, debug=args.debug, browser=args.browser, gui=args.gui)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
