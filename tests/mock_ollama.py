"""A stand-in Ollama server for testing the extraction loop without a GPU.

Implements just enough of /api/tags and /api/generate to exercise the client,
worker, checkpointing and collection paths. Can inject failures and latency so
the retry and resume logic is tested against something that actually misbehaves.

    python tests/mock_ollama.py --port 11500 --fail-rate 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock-model:latest"
AGENTS = [("NVDA", "stock"), ("AAPL", "stock"), ("the Fed", "central_bank"),
          ("USD", "currency"), ("crude oil", "commodity"), ("S&P 500", "index")]
RELATIONS = ["invests in", "supplies", "competes with", "acquires", "regulates", "raises rates on"]


class Handler(BaseHTTPRequestHandler):
    fail_rate = 0.0
    latency = 0.01

    def log_message(self, *a):  # silence per-request logging
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._json(200, {"models": [{"name": MODEL}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not (self.path.startswith("/api/generate") or self.path.startswith("/api/chat")):
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        time.sleep(self.latency)

        if random.random() < self.fail_rate:
            self._json(503, {"error": "mock overload"})
            return

        msgs = req.get("messages") or []
        prompt = req.get("prompt") or (msgs[-1]["content"] if msgs else "")
        # Deterministic per article so re-runs are comparable.
        rng = random.Random(hash(prompt) & 0xFFFFFFFF)
        k = rng.randint(0, 3)
        tuples = []
        for _ in range(k):
            (a, at), (b, bt) = rng.sample(AGENTS, 2)
            tuples.append({
                "agent_a": a,
                "agent_b": b,
                "relation_type": rng.choice(RELATIONS),
                "direction": rng.choice(["positive", "negative"]),
                # Fora da faixa de propósito: o sink precisa normalizar.
                "strength": round(rng.uniform(-0.1, 1.1), 2),
            })
        text = json.dumps({"tuples": tuples})
        self._json(200, {
            "model": req.get("model", MODEL),
            "message": {"role": "assistant", "content": text},
            "response": text,
            "done": True,
            "prompt_eval_count": max(1, len(prompt) // 4),
            "eval_count": max(1, len(text) // 4),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--fail-rate", type=float, default=0.0)
    ap.add_argument("--latency", type=float, default=0.01)
    a = ap.parse_args()
    Handler.fail_rate, Handler.latency = a.fail_rate, a.latency
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"mock ollama on http://127.0.0.1:{a.port} model={MODEL} "
          f"fail_rate={a.fail_rate} latency={a.latency}s")
    srv.serve_forever()


if __name__ == "__main__":
    main()
