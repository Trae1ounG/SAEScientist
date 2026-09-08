"""Local setup checks performed before starting a paid benchmark experiment."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import socket
from urllib.parse import urlsplit


def check_environment(jobs: list[dict]) -> dict:
    """Inspect local configuration without inference, network requests, or writes.

    This does not authenticate either service or estimate peak GPU memory.
    Credential values are never included in diagnostics.
    """
    config = jobs[0]["config"]
    errors = []
    if platform.system() != "Linux":
        errors.append("Full experiments require Linux and an NVIDIA CUDA GPU.")
    model = Path(config["model_path"])
    if not model.is_dir() or not (model / "config.json").is_file():
        errors.append("model_path must contain a downloaded model with config.json.")
    else:
        try:
            spec = json.loads((model / "config.json").read_text())
            if spec.get("model_type") != "gemma2" or spec.get("hidden_size") != 3584:
                errors.append("model_path must use the Gemma-2-9B architecture with hidden_size 3584.")
        except (ValueError, OSError, AttributeError):
            errors.append("Model config.json could not be read as a model configuration.")
    weights = list(model.glob("*.safetensors")) + list(model.glob("pytorch_model*.bin"))
    if not weights:
        errors.append("Model weights are missing from model_path.")
    elif any(not path.is_file() or path.stat().st_size == 0 for path in weights):
        errors.append("Model weights contain an empty or invalid file.")
    for index in (model / "model.safetensors.index.json", model / "pytorch_model.bin.index.json"):
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if not shards or any(Path(name).name != name or not (model / name).is_file()
                                     or (model / name).stat().st_size == 0 for name in shards):
                    errors.append("Model shard index references missing, empty, or invalid weight files.")
            except (ValueError, KeyError, TypeError, AttributeError, OSError):
                errors.append("Model weight index could not be read.")
    if not any((model / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        errors.append("Tokenizer files are missing from model_path.")
    for path in sorted({j["config"]["sae"]["path"] for j in jobs}):
        sae = Path(path)
        if not sae.is_file() or sae.stat().st_size == 0:
            errors.append(f"Missing or empty SAE checkpoint: {sae}")
    if not shutil.which(config["agent"]["cli"]):
        errors.append("agent.cli is not an executable on PATH.")
    judge = config["judge"]
    if judge["model"].startswith("YOUR_"):
        errors.append("Set judge.model to your Azure deployment name.")
    key_name = judge.get("api_key_env", "AZURE_OPENAI_API_KEY")
    if not (os.environ.get(key_name, "").strip() or os.environ.get("AZURE_OPENAI_API_KEY", "").strip()):
        errors.append(f"Missing judge credential environment variable: {key_name}")
    for name in ("AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"):
        if not os.environ.get(name, "").strip():
            errors.append(f"Missing environment variable: {name}")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    if endpoint:
        try:
            url = urlsplit(endpoint)
            valid = url.scheme == "https" and url.hostname and not url.username and not url.password
        except ValueError:
            valid = False
        if not valid:
            errors.append("AZURE_OPENAI_ENDPOINT must be an HTTPS URL without embedded credentials.")
    packages = ("torch", "transformers", "numpy", "ray", "openai", "accelerate", "safetensors")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        errors.append("Missing Python packages: " + ", ".join(missing))
    if "torch" not in missing:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("PyTorch cannot access a CUDA GPU.")
        except (ImportError, OSError, RuntimeError):
            errors.append("PyTorch could not initialize; check the PyTorch/CUDA installation.")
    probe = config.get("probe", {})
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((probe.get("host", "127.0.0.1"), probe.get("port", 8765)))
        except OSError:
            errors.append("The probe port is unavailable; stop its current service or choose another port.")
    return {"ok": not errors, "errors": errors,
            "note": "Local checks only. Service authentication, CLI sandbox compatibility, "
                    "checkpoint integrity, and sufficient GPU memory require a smoke run."}
