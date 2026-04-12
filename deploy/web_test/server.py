#!/usr/bin/env python3
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HOST = "0.0.0.0"
PORT = 8090
AUDIT_API = "http://127.0.0.1:8000/audit"


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def _run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return completed.stdout.strip()
    except FileNotFoundError:
        return ""


def _kill_pids(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    time.sleep(0.5)

    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def cleanup_port(port: int) -> None:
    lsof = shutil.which("lsof")
    if lsof:
        output = _run_command([lsof, "-t", f"-iTCP:{port}", "-sTCP:LISTEN"])
        if output:
            pids = [int(item) for item in output.splitlines() if item.strip().isdigit()]
            if pids:
                print(f"[web_test] 释放端口 {port}: {pids}")
                _kill_pids(pids)
        return

    fuser = shutil.which("fuser")
    if fuser:
        print(f"[web_test] 使用 fuser 释放端口 {port}")
        subprocess.run([fuser, "-k", f"{port}/tcp"], check=False)
        return

    print(f"[web_test] 未找到 lsof/fuser，跳过端口清理: {port}")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html_path = Path(__file__).with_name("index.html")
            self._send_html(html_path.read_text(encoding="utf-8"))
            return

        if self.path == "/health":
            self._send_json(200, {"ok": True, "proxy": True, "target": AUDIT_API})
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/audit":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        text = str(payload.get("text", "")).strip()
        if not text:
            self._send_json(400, {"error": "text is required"})
            return

        req_data = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            AUDIT_API,
            data=req_data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read()
                code = resp.getcode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            self._send_json(502, {"error": "upstream http error", "detail": detail, "status": e.code})
        except Exception as e:
            self._send_json(502, {"error": "upstream unavailable", "detail": str(e)})


def main():
    cleanup_port(PORT)
    print(f"[web_test] open: http://{HOST}:{PORT}")
    print(f"[web_test] proxy to: {AUDIT_API}")
    ReusableHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
