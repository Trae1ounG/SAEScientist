"""Record the runtime and source used by a new local experiment."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import subprocess


def environment_record(root: Path) -> dict:
    packages = {}
    for name in ("torch", "transformers", "numpy", "ray", "openai", "accelerate",
                 "safetensors", "huggingface-hub", "sae-scientist"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    sources = {}
    for folder in ("src", "scripts", "agents"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".md", ".toml"}:
                sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "git_revision": revision, "source_sha256": sources,
    }
