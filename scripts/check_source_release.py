#!/usr/bin/env python3
"""Check publishable files for result/trace paths and evaluation output fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {"results", "outputs", "runs", "logs", "traces", "transcripts", "checkpoints", "artifacts"}
OUTPUT_FIELDS = {"activation_rank", "expert_activation_rank", "judgment_summary", "agent_results",
                 "scores", "transcript", "transcripts", "trace", "traces", "leaderboard"}
OUTPUT_FILES = {"activation.json", "steering.json", "judgment_summary.json", "paper_scores.json",
                "leaderboard.json", "replicates.json"}


def output_fields(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in OUTPUT_FIELDS:
                yield key
            yield from output_fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from output_fields(child)


def inspect_paths(root: Path, paths: list[str]) -> list[str]:
    errors = []
    for name in sorted(set(paths)):
        path = root / name
        # A tracked deletion is already absent from the proposed working tree.
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            errors.append(f"{name}: publish regular source files, not symlinks")
            continue
        parts = Path(name).parts
        if (PRIVATE_DIRS.intersection(parts) or path.name in OUTPUT_FILES
                or path.name == ".env" or path.name.startswith(".env.")):
            errors.append(f"{name}: private output or credential path")
        if parts[0] in {"data", "examples"} and path.suffix in {".jsonl", ".csv", ".tsv", ".gz", ".npz"}:
            errors.append(f"{name}: unexpected measurement/archive file in public inputs")
        if parts[0] in {"data", "examples"} and path.suffix == ".json":
            try:
                keys = sorted(set(output_fields(json.loads(path.read_text(encoding="utf-8")))))
                if keys:
                    errors.append(f"{name}: evaluation output fields: {', '.join(keys)}")
            except (ValueError, UnicodeError):
                errors.append(f"{name}: invalid JSON input")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=args.root,
    ).decode().split("\0")
    errors = inspect_paths(args.root, [p for p in paths if p])
    print(json.dumps({"ok": not errors, "errors": errors,
                      "scope": "Existing tracked and non-ignored files. Git history and arbitrary secrets are not scanned."}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
