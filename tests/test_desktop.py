"""The stand-alone application (morphogen.gui.desktop): what the page asks of the system
(the dialogs, the files of the results saved and opened, the folders shown) with a
stand-in of the window of pywebview, the confirmation of closing an unsaved workspace,
the check of a bundle without a window, the fallback on the web browser without
pywebview, and the hooks of the page. With MORPHOGEN_DESKTOP_TEST=1 and pywebview, the
real window opens (a display is needed: xvfb-run on a server)."""

from __future__ import annotations

import os
import sys
from importlib import resources
from pathlib import Path

import pytest

from morphogen.gui import api, desktop


class _Dialogs:
    """The constants of pywebview's dialogs."""

    class FileDialog:
        OPEN, FOLDER, SAVE = 10, 20, 30


class _Window:
    """The window of pywebview: its file dialogs answer ``answer``."""

    def __init__(self, answer=None):
        self.answer = answer
        self.calls = []

    def create_file_dialog(self, kind, **options):
        self.calls.append((kind, options))
        return self.answer


@pytest.fixture
def session(tmp_path):
    s = api.Session(tmp_path / "session")
    key, folder = s.folder()
    (folder / "curves.csv").write_text("time,stress\n0,0\n1,2\n")
    s.add(key, "result", "a run", tool="radioss_test")
    return s, key


def _api(session, answer=None):
    opened, revealed = [], []
    a = desktop.DesktopApi(
        session, _Window(answer), dialogs=_Dialogs, opener=opened.append, revealer=revealed.append
    )
    return a, opened, revealed


def test_dialogs_and_files(session, tmp_path):
    s, key = session
    a, opened, revealed = _api(s, answer=(str(tmp_path / "chosen"),))
    assert a.info()["desktop"] is True and a.info()["session"] == str(s.root)
    assert a.choose_folder("a title", str(tmp_path)) == str(tmp_path / "chosen")
    kind, options = a._window.calls[-1]
    assert kind == _Dialogs.FileDialog.FOLDER and options["directory"] == str(tmp_path)
    assert a.choose_file() == str(tmp_path / "chosen")
    assert a._window.calls[-1][0] == _Dialogs.FileDialog.OPEN
    target = tmp_path / "saved.csv"
    a._window.answer = str(target)  # a dialog may answer a plain path
    assert a.save_file(f"/files/{key}/curves.csv") == str(target)
    assert target.read_text().startswith("time,stress")
    assert a._window.calls[-1][1]["save_filename"] == "curves.csv"
    a.open_file(f"/files/{key}/curves.csv?inline")
    assert opened == [s.root.resolve() / key / "curves.csv"]
    a.reveal(key)
    a.open_session_folder()
    assert revealed == [(s.root / key).resolve(), s.root]
    a._window.answer = None  # cancelled
    assert a.choose_folder() is None and a.save_file(f"/files/{key}/curves.csv") is None
    with pytest.raises(ValueError, match="not a file of the session"):
        a.open_file("/api/info")
    with pytest.raises(FileNotFoundError):
        a.open_file(f"/files/{key}/../../outside.txt")
    with pytest.raises(ValueError, match="no folder"):
        a.reveal("../..")
    assert desktop.DesktopApi(s).choose_folder() is None  # no window yet


def test_closing_asks_while_unsaved(session):
    s, key = session
    a, _, _ = _api(s)
    assert a.unsaved()
    a.workspace_saved()
    assert not a.unsaved()
    other, _ = s.folder()
    s.add(other, "morphology", "another")
    assert a.unsaved()
    s.remove(other)
    s.remove(key)
    assert not a.unsaved()  # empty


def test_check_without_a_window(tmp_path, capsys):
    assert desktop.main(["--check", "--session", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Explicit dynamics" in out and "a realization generated" in out
    with pytest.raises(SystemExit):
        desktop.main(["--version"])


def test_fallback_on_the_browser(monkeypatch, capsys):
    served = []
    monkeypatch.setitem(sys.modules, "webview", None)  # pywebview missing
    monkeypatch.setattr(
        "morphogen.gui.server.serve",
        lambda port, open_browser, session_dir: served.append((port, session_dir)),
    )
    assert desktop.run() == 0 and served == [(0, None)]
    assert "morphogen[desktop]" in capsys.readouterr().out
    assert desktop.run(browser=True) == 0 and len(served) == 2


def test_hooks_of_the_page_and_the_package():
    page = (resources.files("morphogen") / "gui" / "index.html").read_text(encoding="utf-8")
    for hook in (
        "pywebviewready",
        "DESKTOP.choose_folder",
        "DESKTOP.save_file",
        "DESKTOP.open_file",
        "DESKTOP.reveal",
        'data-browse="file"',
        "data-reveal",
    ):
        assert hook in page
    icon = desktop.icon_path()  # Windows Forms takes no PNG for the icon of a window
    head = b"\x00\x00\x01\x00" if sys.platform == "win32" else b"\x89PNG\r\n\x1a\n"
    assert icon is not None and icon.read_bytes()[: len(head)] == head
    from morphogen.cli import build_parser

    args = build_parser().parse_args(["desktop", "--browser"])
    assert args.browser and args.func.__name__ == "cmd_desktop"
    root = Path(__file__).resolve().parents[1]
    spec = root / "packaging" / "desktop" / "morphogen-desktop.spec"
    if spec.is_file():  # in the sources (not in the wheel)
        assert "morphogen_desktop.py" in spec.read_text(encoding="utf-8")


@pytest.mark.skipif(
    os.environ.get("MORPHOGEN_DESKTOP_TEST") != "1",
    reason="the real window: MORPHOGEN_DESKTOP_TEST=1, pywebview and a display",
)
def test_real_window(tmp_path):
    webview = pytest.importorskip("webview")
    import time

    server, url = desktop.start_server(tmp_path)
    a = desktop.DesktopApi(server.session, dialogs=webview)
    window = webview.create_window("morphoGen", url, js_api=a)
    a._window = window
    window.menu = desktop._menu(webview, window, a)
    seen = {}

    def check():
        window.events.loaded.wait(60)
        for _ in range(100):
            if window.evaluate_js("document.body.classList.contains('desktop')"):
                break
            time.sleep(0.2)
        seen["desktop"] = window.evaluate_js("document.body.classList.contains('desktop')")
        for _ in range(300):  # the tabs of the tools, once the catalog has come
            seen["tabs"] = window.evaluate_js("TABS.map((t) => t.name).join(',')")
            if "dynamics" in seen["tabs"]:
                break
            time.sleep(0.2)
        window.destroy()

    try:
        webview.start(check, icon=str(desktop.icon_path()))  # as the application
    finally:
        server.shutdown()
        server.server_close()
    assert seen["desktop"] is True and "dynamics" in seen["tabs"]


def test_jobs_stopped_when_the_window_closes(tmp_path):
    import threading

    server, _ = desktop.start_server(tmp_path)

    class Job:
        def __init__(self):
            self.event = threading.Event()
            self.thread = threading.Thread(target=self.event.wait, daemon=True)
            self.thread.start()

        def status(self):
            return {"state": "running" if self.thread.is_alive() else "stopped"}

        def stop(self):
            self.event.set()

    try:
        jobs = [Job(), Job()]
        server.jobs.update(a=jobs[0], b=jobs[1])
        assert desktop.stop_jobs(server, timeout=5) == 2
        assert not any(j.thread.is_alive() for j in jobs)
        assert desktop.stop_jobs(server) == 0
    finally:
        server.shutdown()
        server.server_close()


def test_browse_fields():
    from morphogen.gui.tools import Field

    assert Field("folder", "str", browse="folder").to_dict()["browse"] == "folder"
    assert "browse" not in Field("name", "str").to_dict()
    with pytest.raises(ValueError, match="browse"):
        Field("x", "str", browse="directory")
