#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stop this benchmark's live Ray probe actors and release their GPUs."
    )
    parser.add_argument("--address", default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from ray.util.state import list_actors

    ray.init(address=args.address)
    actors = list_actors(
        filters=[
            ("class_name", "=", "make_worker.<locals>.ProbeWorker"),
            ("state", "=", "ALIVE"),
        ],
        detail=True,
    )
    if len(actors) > args.max_workers:
        raise RuntimeError(
            f"refusing to stop {len(actors)} actors; max-workers={args.max_workers}"
        )
    if not args.execute:
        print({"matched": len(actors), "executed": False})
        return

    @ray.remote(num_cpus=0)
    def terminate(pid: int) -> int:
        os.kill(pid, signal.SIGTERM)
        return pid

    futures = [
        terminate.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                actor.node_id, soft=False
            )
        ).remote(actor.pid)
        for actor in actors
    ]
    stopped = ray.get(futures)
    print({"matched": len(actors), "stopped": len(stopped), "executed": True})


if __name__ == "__main__":
    main()

