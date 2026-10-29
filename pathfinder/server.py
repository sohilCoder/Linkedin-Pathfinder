"""A small JSON API + web page, built only on the standard library.

Endpoints
    GET    /                         the web UI
    GET    /api/stats                network size, matching stats, top companies
    GET    /api/members              members and their upload sizes
    POST   /api/members              join / replace: {"profile": {...}, "csv": "<text>"}
    DELETE /api/members/<member_id>  leave: deletes the member and their rows
    GET    /api/search?member=&company=&title=&max_hops=
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, unquote, urlparse

from .loader import ExportFormatError, MemberProfile
from .network import Network

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def make_handler(network: Network):
    page = resources.files("pathfinder").joinpath("static/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "Pathfinder/1.0"

        # quieter logs
        def log_message(self, fmt, *args):  # noqa: N802
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if url.path == "/api/stats":
                return self._json(200, network.stats())
            if url.path == "/api/members":
                return self._json(200, network.members())
            if url.path == "/api/search":
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                if not q.get("member") or not q.get("company"):
                    return self._error(400, "member and company are required")
                try:
                    result = network.search(
                        q["member"], q["company"],
                        title=q.get("title", ""),
                        max_hops=int(q.get("max_hops", 3)),
                    )
                except KeyError as exc:
                    return self._error(404, str(exc).strip("'\""))
                except ValueError:
                    return self._error(400, "max_hops must be a number")
                return self._json(200, result)
            return self._error(404, "not found")

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/members":
                return self._error(404, "not found")
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                return self._error(413, "upload must be between 1 byte and 5 MB")
            try:
                payload = json.loads(self.rfile.read(length))
                profile = MemberProfile.from_dict(payload.get("profile") or {})
                network.add_member(profile, str(payload.get("csv") or ""))
            except ExportFormatError as exc:
                return self._error(422, str(exc))
            except (ValueError, TypeError, AttributeError) as exc:
                return self._error(400, f"bad request: {exc}")
            return self._json(201, {"member_id": profile.member_id})

        def do_DELETE(self):  # noqa: N802
            path = urlparse(self.path).path
            prefix = "/api/members/"
            if not path.startswith(prefix):
                return self._error(404, "not found")
            member_id = unquote(path[len(prefix):])
            if network.remove_member(member_id):
                return self._json(200, {"deleted": member_id})
            return self._error(404, "unknown member")

    return Handler


def make_server(network: Network, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(network))


def serve(network: Network, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = make_server(network, host, port)
    print(f"Warm Intro Pathfinder running at http://{host}:{server.server_port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
