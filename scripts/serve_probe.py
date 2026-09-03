#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Queue
import socket
import uuid

from sae_bench.probe import make_worker, validate_request


MAX_BODY_BYTES = 1_000_000


class ProbeServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler, *, ray, workers, top_k: int):
        super().__init__(server_address, handler)
        self.ray = ray
        self.workers = Queue()
        for worker in workers:
            self.workers.put(worker)
        self.top_k = top_k


class Handler(BaseHTTPRequestHandler):
    server: ProbeServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {"status": "ok", "workers": self.server.workers.qsize()})

    def do_POST(self) -> None:
        if self.path != "/probe":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("invalid request body size")
            value = json.loads(self.rfile.read(length))
            value.setdefault("id", uuid.uuid4().hex)
            request = validate_request(value, self.server.top_k)
            worker = self.server.workers.get()
            try:
                result = self.server.ray.get(worker.probe.remote(request))
            finally:
                self.server.workers.put(worker)
            self._send(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _send(self, status: int, value: dict) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve queued SAE probes over HTTP.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sae-path", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--host", default="::")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in [1, 8]")

    import ray

    ray.init(address=args.address)
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if args.workers > available_gpus:
        raise RuntimeError(
            f"requested {args.workers} workers but cluster exposes {available_gpus} GPUs"
        )
    worker_class = make_worker(
        ray,
        model_path=str(args.model_path.resolve(strict=True)),
        sae_path=str(args.sae_path.resolve(strict=True)),
        layer_index=args.layer,
    )
    workers = [worker_class.remote() for _ in range(args.workers)]
    ray.get([worker.probe.remote({"id": "warmup", "texts": ["warmup"], "top_k": 1, "feature_ids": []}) for worker in workers])
    if ":" in args.host:
        ProbeServer.address_family = socket.AF_INET6
    server = ProbeServer((args.host, args.port), Handler, ray=ray, workers=workers, top_k=args.top_k)
    print(
        json.dumps(
            {
                "status": "ready",
                "host": args.host,
                "node_ip": ray.util.get_node_ip_address(),
                "port": args.port,
                "workers": args.workers,
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

