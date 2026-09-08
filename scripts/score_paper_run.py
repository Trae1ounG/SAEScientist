#!/usr/bin/env python3
"""Compute current paper scores without model inference or API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sae_scientist.paper_scores import paper_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--judgment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = paper_scores(
        json.loads(args.activation.read_text()), json.loads(args.judgment.read_text())
    )
    result["sources"] = {"activation": str(args.activation), "judgment": str(args.judgment)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["scores"]))


if __name__ == "__main__":
    main()
