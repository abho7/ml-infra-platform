"""
HTTP control plane.

Built on `http.server` rather than FastAPI or Flask, for the same reason the
rest of this project has no dependencies beyond numpy: the three engines it
composes are all from-scratch, and a web framework here would be the only
vendored abstraction in the stack. The surface needed is small -- a dozen JSON
routes and a static file -- and stdlib covers it.

THREADING. The control plane is not thread-safe: it owns a Raft cluster whose
state advances by explicit ticks, and two requests advancing it concurrently
would interleave ticks in a way no real cluster could reproduce. A single lock
serialises every request. That makes the API slow under load and completely
deterministic, which is the right trade for a system whose whole value is
being able to say exactly what happened.

The background ticker is what makes the dashboard live: without it the cluster
would only advance when someone poked it, and a "live" view would show a frozen
system between clicks.
"""

from __future__ import annotations

import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mlplat.control.plane import ControlPlane
from mlplat.training.job import JobSpec

DASHBOARD = Path(__file__).resolve().parent.parent.parent / "dashboard"


class Ticker(threading.Thread):
    """Advances the platform on a wall-clock cadence so the dashboard is live.

    Daemon so it never keeps the process alive on its own. Holds the same lock
    the request handlers use, so a tick can never interleave with a request.
    """

    def __init__(self, plane: ControlPlane, lock: threading.Lock, hz: float = 10.0):
        super().__init__(daemon=True)
        self.plane = plane
        self.lock = lock
        self.period = 1.0 / hz
        self._stop = threading.Event()
        self.paused = False

    def run(self) -> None:
        while not self._stop.wait(self.period):
            if self.paused:
                continue
            with self.lock:
                try:
                    self.plane.advance()
                except Exception:  # pragma: no cover - a wedged tick must not
                    pass          # kill the server; the API still reports state

    def stop(self) -> None:
        self._stop.set()


class Handler(BaseHTTPRequestHandler):
    server_version = "mlplat/1.0"

    def __init__(self, plane, lock, ticker, *args, **kwargs):
        self.plane = plane
        self.lock = lock
        self.ticker = ticker
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------- plumbing

    def log_message(self, *_args) -> None:
        pass  # the event stream is the log; access lines would drown it

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _static(self, rel: str) -> None:
        path = (DASHBOARD / rel).resolve()
        if not path.is_file() or DASHBOARD.resolve() not in path.parents:
            self._json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        ctype = {"html": "text/html", "js": "application/javascript",
                 "css": "text/css", "json": "application/json"}.get(
                     path.suffix.lstrip("."), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------- GET

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path.rstrip("/") or "/"

        if p == "/":
            return self._static("index.html")
        if p.startswith("/static/"):
            return self._static(p[len("/static/"):])

        with self.lock:
            if p == "/api/state":
                return self._json(self.plane.state())
            if p == "/api/events":
                limit = int(q.get("limit", ["50"])[0])
                return self._json({"events": self.plane.recent_events(limit)})
            if p == "/api/jobs":
                return self._json({"jobs": [m.to_json() for m in self.plane.jobs.values()]})
            if p.startswith("/api/jobs/"):
                jid = p.split("/")[-1]
                try:
                    return self._json(self.plane.job(jid).to_json())
                except KeyError as e:
                    return self._json({"error": str(e)}, 404)
            if p == "/api/health":
                return self._json(self.plane.state()["health"])
        return self._json({"error": f"no route {p}"}, 404)

    # ------------------------------------------------------------------ POST

    def do_POST(self) -> None:
        p = urlparse(self.path).path.rstrip("/") or "/"
        try:
            body = self._body()
        except json.JSONDecodeError as e:
            return self._json({"error": f"bad json: {e}"}, 400)

        with self.lock:
            try:
                if p == "/api/jobs":
                    spec = JobSpec(**body)
                    return self._json(self.plane.submit(spec).to_json(), 201)
                if p == "/api/search":
                    return self._json(self.plane.search(
                        body["vector"], int(body.get("k", 5)),
                        mode=body.get("mode", "linearizable"),
                    ))
                if p == "/api/advance":
                    return self._json(self.plane.advance(
                        storage_ticks=int(body.get("storage_ticks", 1)),
                        job_steps=int(body.get("job_steps", 1)),
                    ))
                if p == "/api/pause":
                    self.ticker.paused = bool(body.get("paused", True))
                    return self._json({"paused": self.ticker.paused})

                # --- fault injection, deliberately part of the real API ---
                if p == "/api/nodes/kill":
                    return self._json(self.plane.kill_node(body["node"]))
                if p == "/api/nodes/restart":
                    return self._json(self.plane.restart_node(body["node"]))
                if p == "/api/partition":
                    return self._json(self.plane.partition(*body["groups"]))
                if p == "/api/heal":
                    return self._json(self.plane.heal())
                if p == "/api/workers/kill":
                    return self._json(self.plane.kill_worker(body["job"], body.get("worker")))
            except (KeyError, ValueError, TypeError) as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
        return self._json({"error": f"no route {p}"}, 404)


def serve(plane: ControlPlane, *, host: str = "127.0.0.1", port: int = 8950,
          hz: float = 10.0) -> tuple[ThreadingHTTPServer, Ticker]:
    lock = threading.Lock()
    ticker = Ticker(plane, lock, hz=hz)
    ticker.start()
    handler = partial(Handler, plane, lock, ticker)
    httpd = ThreadingHTTPServer((host, port), handler)
    return httpd, ticker
