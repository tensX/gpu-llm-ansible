"""Заглушка vLLM для CI без GPU.

Отвечает как OpenAI-совместимый API vLLM на /health, /v1/models и /v1/chat/completions
и, как vLLM, требует для /v1/* ключ из переменной окружения VLLM_API_KEY.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_KEY = os.environ["VLLM_API_KEY"]
MODEL = os.environ.get("MOCK_MODEL", "qwen2.5-1.5b-instruct")
PORT = int(os.environ.get("MOCK_PORT", "8000"))


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if self.headers.get("Authorization") == f"Bearer {API_KEY}":
            return True
        self.send_json(401, {"error": "Unauthorized"})
        return False

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {})
        elif self.path == "/v1/models":
            if self.authorized():
                self.send_json(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self.send_json(404, {"error": "Not Found"})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "Not Found"})
            return
        if not self.authorized():
            return
        request = json.loads(body or b"{}")
        if request.get("model") != MODEL:
            self.send_json(404, {"error": f"The model `{request.get('model')}` does not exist."})
            return
        self.send_json(200, {
            "object": "chat.completion",
            "model": MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Париж"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
        })

    def log_message(self, fmt, *args):
        sys.stderr.write("mock-vllm: " + fmt % args + "\n")


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
