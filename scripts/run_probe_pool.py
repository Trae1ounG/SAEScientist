#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sae_scientist.probe import MAX_TEXTS_PER_REQUEST, make_worker, validate_request


def read_requests(path: Path, default_top_k: int) -> list[dict[str, Any]]:
    requests = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        request = validate_request(json.loads(line), default_top_k)
        if request["id"] in seen:
            raise ValueError(f"duplicate request id on line {line_number}: {request['id']}")
        seen.add(request["id"])
        requests.append(request)
    if not requests:
        raise ValueError("input contains no requests")
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a queued SAE activation probe pool with one Ray actor per GPU."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sae-path", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--address", default="auto")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in [1, 8]")
    if not 1 <= args.top_k <= 1024:
        parser.error("top-k must be in [1, 1024]")

    requests = read_requests(args.input, args.top_k)
    import ray

    ray.init(address=args.address)
    resources = ray.cluster_resources()
    available_gpus = int(resources.get("GPU", 0))
    if args.workers > available_gpus:
        raise RuntimeError(
            f"requested {args.workers} workers but cluster exposes {available_gpus} GPUs"
        )
    model_path = args.model_path.resolve(strict=True)
    sae_path = args.sae_path.resolve(strict=True)
    worker_class = make_worker(
        ray,
        model_path=str(model_path),
        sae_path=str(sae_path),
        layer_index=args.layer,
    )
    workers = [worker_class.remote() for _ in range(args.workers)]

    pending = {}
    request_iter = iter(requests)
    for worker in workers:
        request = next(request_iter, None)
        if request is not None:
            pending[worker.probe.remote(request)] = worker

    results = []
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        future = ready[0]
        worker = pending.pop(future)
        results.append(ray.get(future))
        request = next(request_iter, None)
        if request is not None:
            pending[worker.probe.remote(request)] = worker

    order = {request["id"]: index for index, request in enumerate(requests)}
    results.sort(key=lambda result: order[result["id"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "requests": len(results),
                "workers": args.workers,
                "cluster_gpus": available_gpus,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
