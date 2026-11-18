"""Local web server of the graphical interface (standard library only).

``morphogen gui`` serves a single page on http://127.0.0.1:<port>/ and a JSON API; the
computations run in this process, with results written to a temporary session folder.
The server listens on the local machine only.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import tempfile
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs

from morphogen import __version__
from morphogen._utils import to_jsonable
from morphogen.gui import api, tools

log = logging.getLogger("morphogen")

MAX_BODY = 1 << 20
#: Files shown in the page (images, videos) rather than downloaded
INLINE = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
    ".pdf": "application/pdf",
}


def page() -> bytes:
    return (resources.files("morphogen") / "gui" / "index.html").read_bytes()


def viewer() -> bytes:
    """The script of the 3D view of the page (WebGL, no library)."""
    return (resources.files("morphogen") / "gui" / "viewer.js").read_bytes()


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, session_dir: Path):
        super().__init__(address, Handler)
        self.session = api.Session(session_dir)
        self.jobs: dict[str, api.OptimizationJob | api.SimulationJob | tools.ToolJob] = {}
        self.jobs_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: GuiServer

    def log_message(self, fmt, *args):  # quiet: use the morphogen logger
        log.debug("gui: " + fmt, *args)

    def _send(self, code: int, body: bytes, content_type: str, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200):
        body = json.dumps(to_jsonable(data)).encode("utf-8")
        self._send(code, body, "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request too large")
        data = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def do_GET(self):
        path, _, query = self.path.partition("?")
        try:
            if path in ("/", "/index.html"):
                self._send(200, page(), "text/html; charset=utf-8")
            elif path == "/viewer.js":
                self._send(200, viewer(), "text/javascript; charset=utf-8")
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif path == "/api/info":
                from morphogen import gpu

                self._json({"version": __version__, "gpu": gpu.available()})
            elif path == "/api/models":
                self._json(api.models())
            elif path == "/api/workspace":
                self._json(self.server.session.workspace())
            elif path == "/api/tools":
                self._json(tools.catalog())
            elif path == "/api/fields":
                key = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("key")
                self._json(api.simulation_fields({"key": key}, self.server.session))
            elif path == "/api/result":
                key = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("key")
                self._json(api.tool_result(key, self.server.session))
            elif path == "/api/surface":  # the 3D view: key, resolution, field, budget
                request = {k: v[-1] for k, v in parse_qs(query).items()}
                self._json(api.surface(request, self.server.session))
            elif path.startswith("/api/job/"):
                key = path.rsplit("/", 1)[-1]
                with self.server.jobs_lock:
                    job = self.server.jobs.get(key)
                if job is None:
                    self._json({"error": "unknown job"}, 404)
                else:
                    self._json(job.status())
            elif path.startswith("/files/"):
                _, _, key, name = path.split("/", 3)
                file = self.server.session.file(key, name)
                inline = "inline" in query.split("&") and file.suffix in INLINE
                kind = INLINE.get(file.suffix) if inline else None
                self._send(
                    200,
                    file.read_bytes(),
                    kind or mimetypes.guess_type(file.name)[0] or "application/octet-stream",
                    {
                        "Content-Disposition": (
                            f'{"inline" if inline else "attachment"}; filename="{file.name}"'
                        )
                    },
                )
            else:
                self._json({"error": "not found"}, 404)
        except ConnectionError:  # the page went away during a large answer (a surface)
            return
        except FileNotFoundError:
            self._json({"error": "not found"}, 404)
        except (ValueError, TypeError, KeyError) as exc:  # an item that is not in the workspace
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug(traceback.format_exc())
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        # JSON only: a page of another site cannot post it without the preflight of its
        # browser, which this server never answers (forms post text/plain or form data)
        kind = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if kind != "application/json":
            self._json({"error": "requests of the interface are JSON (application/json)"}, 415)
            return
        try:
            request = self._body()
            if path == "/api/generate":
                self._json(api.generate(request, self.server.session))
            elif path in ("/api/formation", "/api/evolve"):
                self._json(api.formation(request))
            elif path == "/api/count":
                self._json(api.count(request))
            elif path == "/api/database":
                self._json(api.database_info(request))
            elif path == "/api/database/add":
                self._json(api.database_add(request))
            elif path == "/api/workspace/remove":
                self.server.session.remove(request.get("key"))
                self._json({"removed": request.get("key")})
            elif path == "/api/workspace/rename":
                item = self.server.session.rename(request.get("key"), request.get("title"))
                self._json(self.server.session.public(item))
            elif path == "/api/workspace/save":
                self._json(api.save_workspace(request, self.server.session))
            elif path == "/api/workspace/open":
                self._json(api.open_workspace(request, self.server.session))
            elif path == "/api/database/session":
                self._json(api.database_add_session(request, self.server.session))
            elif path == "/api/ashby":
                self._json(api.ashby(request))
            elif path in ("/api/optimize", "/api/simulate", "/api/tool"):
                kind = {
                    "/api/optimize": api.OptimizationJob,
                    "/api/simulate": api.SimulationJob,
                    "/api/tool": tools.ToolJob,
                }[path]
                job = kind(request, self.server.session).start()
                key = uuid.uuid4().hex[:12]
                with self.server.jobs_lock:
                    self.server.jobs[key] = job
                self._json({"job": key})
            elif path == "/api/job/stop":  # at its next checkpoint (morphogen.stopping)
                with self.server.jobs_lock:
                    job = self.server.jobs.get(str(request.get("job")))
                if job is None:
                    raise ValueError(f"no job {request.get('job')!r} running in this session")
                job.stop()
                self._json({"stopping": request["job"]})
            elif path == "/api/render":
                self._json(api.render(request, self.server.session))
            elif path == "/api/export":
                self._json(api.export(request, self.server.session))
            else:
                self._json({"error": "not found"}, 404)
        except (ValueError, TypeError, KeyError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            log.debug(traceback.format_exc())
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def serve(port: int = 8765, *, open_browser: bool = True, session_dir=None) -> None:
    """Run the interface until interrupted (Ctrl+C)."""
    with tempfile.TemporaryDirectory(prefix="morphogen_gui_") as tmp:
        server = GuiServer(("127.0.0.1", int(port)), Path(session_dir or tmp))
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"morphoGen interface on {url} (Ctrl+C to stop)")
        if open_browser:
            threading.Timer(0.5, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("stopped")
        finally:
            server.server_close()
