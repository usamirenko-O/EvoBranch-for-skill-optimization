"""Serve local text embeddings through an OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_PORT = 8318
#: what the proxy / clients call this model; the HF id is what we load
SERVED_NAME = "local-bge-small-en-v1.5"


class _State:
    model = None
    model_name = DEFAULT_MODEL
    lock = threading.Lock()


def _load(model_name: str):
    from sentence_transformers import SentenceTransformer

    _State.model = SentenceTransformer(model_name, device="cpu")
    _State.model_name = model_name


def _embed(texts: list) -> tuple:
    with _State.lock:
        vectors = _State.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        tokenizer = _State.model.tokenizer
        n_tokens = sum(len(tokenizer.encode(t, add_special_tokens=True)) for t in texts)
    return [[float(x) for x in v] for v in vectors], int(n_tokens)


class _Handler(BaseHTTPRequestHandler):
    server_version = "LocalEmbeddings/1.0"

    def log_message(self, fmt, *args):  # noqa: D102 - quiet
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "local",
                                                   "hf_model": _State.model_name}]})
            return
        if self.path.rstrip("/").endswith("/health"):
            self._send({"status": "ok", "model": _State.model_name})
            return
        self._send({"error": "not found"}, status=404)

    def do_POST(self):  # noqa: N802
        if not self.path.rstrip("/").endswith("/embeddings"):
            self._send({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8") if length else "{}")
        except json.JSONDecodeError:
            self._send({"error": {"message": "bad json"}}, status=400)
            return
        inputs = request.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not all(isinstance(t, str) for t in inputs):
            self._send({"error": {"message": "input must be a string or a list of strings"}}, status=400)
            return
        if not inputs:
            self._send({"object": "list", "data": [], "model": SERVED_NAME,
                        "usage": {"prompt_tokens": 0, "total_tokens": 0}})
            return
        try:
            vectors, n_tokens = _embed(inputs)
        except Exception as exc:  # noqa: BLE001
            self._send({"error": {"message": f"embedding failed: {exc}"}}, status=500)
            return
        self._send({
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
            "model": SERVED_NAME,
            "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
        })


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"sentence-transformers model id (default {DEFAULT_MODEL})")
    args = p.parse_args()
    print(f"loading {args.model} on cpu ...", flush=True)
    _load(args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    server.daemon_threads = True
    print(f"local embeddings ({args.model}) on http://127.0.0.1:{args.port}/v1  served as '{SERVED_NAME}'", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
