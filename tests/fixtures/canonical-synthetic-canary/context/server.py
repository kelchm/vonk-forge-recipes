#!/usr/bin/env python3
"""Deterministic OpenAI-shaped HTTP smoke service for contract tests."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "canonical-synthetic-canary"
HEALTH = {
    "status": "ok",
    "service": MODEL,
    "model": MODEL,
    "healthy": True,
}
EXPECTED_REQUEST = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "ping"}],
    "stream": False,
    "max_tokens": 16,
}
EXPECTED_RESPONSE = {
    "id": "chatcmpl-canonical-synthetic-canary",
    "object": "chat.completion",
    "created": 1735689600,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "canonical synthetic ok",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 3,
        "total_tokens": 4,
    },
}


class Handler(BaseHTTPRequestHandler):
    server_version = "canonical-synthetic-canary/1.0"

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._write_json(200, HEALTH)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}"
            )
        except (ValueError, json.JSONDecodeError):
            payload = None
        # OpenAI defaults stream to false; LiteLLM removes it on this path.
        if isinstance(payload, dict) and payload.get("stream", False) is False:
            payload = {**payload, "stream": False}
        if payload != EXPECTED_REQUEST or payload.get("stream") is not False:
            self._write_json(
                400,
                {
                    "error": {
                        "message": "unexpected request",
                        "type": "invalid_request_error",
                    }
                },
            )
            return
        self._write_json(200, EXPECTED_RESPONSE)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _write_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    host = os.environ.get("VONK_LISTEN_HOST", "0.0.0.0")
    port = int(os.environ.get("VONK_LISTEN_PORT", "8000"))
    ThreadingHTTPServer((host, port), Handler).serve_forever()
