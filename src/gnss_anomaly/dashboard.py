"""Loopback-only read-only dashboard; no arbitrary file serving or execution endpoints."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .figures import FORMATS, figure_bytes, paper_bundle, summary_csv
from .reporting import History


def handler_for(history: History):
    assets = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, mime="application/json; charset=utf-8", status=200, filename=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'",
            )
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            host = self.headers.get("Host", "")
            port = self.server.server_port
            if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return self.send({"error": "仅支持本机访问"}, status=403)
            origin = self.headers.get("Origin")
            if (origin and origin != f"http://{host}") or self.headers.get(
                "Sec-Fetch-Site"
            ) == "cross-site":
                return self.send({"error": "拒绝跨站读取"}, status=403)
            route = urlparse(self.path)
            query = parse_qs(route.query)

            def arg(name, default=None):
                return query.get(name, [default])[0]

            try:
                if route.path in ("/", "/app.js", "/style.css"):
                    name, mime = {
                        "/": ("index.html", "text/html; charset=utf-8"),
                        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                        "/style.css": ("style.css", "text/css; charset=utf-8"),
                    }[route.path]
                    return self.send((assets / name).read_bytes(), mime)
                if route.path == "/api/runs":
                    return self.send(
                        {**history.index(), "capabilities": ["injection_label_gallery_v1"]}
                    )
                if route.path == "/api/run":
                    return self.send(history.detail(arg("run")))
                if route.path == "/api/data":
                    return self.send(history.data_index(arg("view", "prepared")))
                if route.path == "/api/criteria":
                    path = history.root / "docs/metrics-criteria.md"
                    return self.send({"text": path.read_text(encoding="utf-8")})
                if route.path == "/api/case":
                    window, truth, prediction, row = history.case(arg("run"), arg("result"))
                    return self.send(
                        {
                            "window": window.model_dump(mode="json"),
                            "truth": truth.model_dump(),
                            "prediction": prediction.model_dump(),
                            "metrics": row["metrics"],
                            "diagnostics": row.get("diagnostics", {}),
                            "tools": row["tools"],
                            "model_calls": [
                                {
                                    k: c.get(k)
                                    for k in (
                                        "model",
                                        "response_model",
                                        "response_id",
                                        "request_settings",
                                        "finish_reason",
                                        "reasoning_present",
                                        "status",
                                        "usage",
                                        "seconds",
                                        "response",
                                        "attempt",
                                        "logical_call",
                                        "error_type",
                                        "prediction_status",
                                    )
                                }
                                for c in row.get("model_calls", [])
                            ],
                        }
                    )
                if route.path == "/api/label":
                    window, truth, _, _ = history.case(arg("run"), arg("result"))
                    return self.send(
                        {"window": window.model_dump(mode="json"), "truth": truth.model_dump()}
                    )
                if route.path in ("/chart", "/export"):
                    fmt = arg("format", "svg")
                    if fmt not in FORMATS:
                        raise ValueError("未知导出格式")
                    chart = arg("chart", "overview")
                    detail = None
                    case = None
                    bounds = None
                    if arg("start") is not None or arg("end") is not None:
                        bounds = (int(arg("start")), int(arg("end")))
                    if chart == "snapshot":
                        window, _ = history.snapshot(arg("snapshot"))
                        case = (window, None, None)
                    else:
                        detail = history.detail(arg("run"))
                        if chart in ("case", "labels"):
                            window, truth, prediction, row = history.case(arg("run"), arg("result"))
                            if chart == "case":
                                detail["case_info"] = {
                                    "method": row["method"],
                                    "repeat": row["repeat"],
                                }
                            case = (window, truth, prediction if chart == "case" else None)
                    content = figure_bytes(
                        detail,
                        chart,
                        fmt,
                        arg("metric", "f1"),
                        arg("dimension", "kind"),
                        arg("methods", "").split(",") if arg("methods") else None,
                        case,
                        bounds,
                        trend=arg("trend") == "daily",
                    )
                    return self.send(
                        content,
                        FORMATS[fmt],
                        filename=f"gnss-{chart}.{fmt}" if route.path == "/export" else None,
                    )
                if route.path == "/metrics.csv":
                    return self.send(
                        summary_csv(history.detail(arg("run"))),
                        "text/csv; charset=utf-8",
                        filename="gnss-metrics.csv",
                    )
                if route.path == "/paper.zip":
                    return self.send(
                        paper_bundle(history.detail(arg("run"))),
                        "application/zip",
                        filename="gnss-paper-figures.zip",
                    )
                return self.send({"error": "未找到页面"}, status=404)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                message = (
                    str(exc)
                    if isinstance(exc, ValueError)
                    else "记录缺失或格式不支持，请检查本地文件"
                )
                return self.send({"error": message}, status=400)

    return Handler


def serve(workspace: Path, port: int = 8765):
    if not 1 <= port <= 65535:
        raise ValueError("端口必须在 1–65535 范围内")
    history = History(workspace)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(history))
    print(f"实验工作台：http://127.0.0.1:{port}（Ctrl+C 停止；只读本地运行与快照）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
