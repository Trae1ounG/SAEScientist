from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .remote_scoring import query_features, score_probe_results
from .suites import load_suite


PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "scripts"


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"model_path", "sae", "task", "suite", "expert_feature_id", "agent", "steering", "judge", "output_dir"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing: {', '.join(missing)}")
    return config


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def run_script(name: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / name), *arguments],
        cwd=PROJECT,
        check=True,
    )


def serve_command(config: dict[str, Any]) -> list[str]:
    sae = config["sae"]
    probe = config.get("probe", {})
    return [
        sys.executable,
        str(SCRIPTS / "serve_probe.py"),
        "--model-path",
        str(project_path(config["model_path"])),
        "--sae-path",
        str(project_path(sae["path"])),
        "--layer",
        str(sae["layer"]),
        "--workers",
        str(probe.get("workers", 1)),
        "--address",
        str(probe.get("ray_address", "auto")),
        "--host",
        str(probe.get("host", "127.0.0.1")),
        "--port",
        str(probe.get("port", 8765)),
    ]


def wait_for_probe(process: subprocess.Popen, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"probe service exited with code {process.returncode}")
        try:
            with urlopen(url.rstrip("/") + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(1)
    raise TimeoutError("probe service did not become ready")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def serve(config_path: Path) -> None:
    config = read_config(config_path)
    raise SystemExit(subprocess.call(serve_command(config), cwd=PROJECT))


def reproduce(config_path: Path) -> None:
    config = read_config(config_path)
    output_dir = project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config["agent"]["run_id"]
    runs_root = output_dir / "runs"
    probe = config.get("probe", {})
    probe_url = f"http://{probe.get('host', '127.0.0.1')}:{probe.get('port', 8765)}"

    import ray

    ray_address = probe.get("ray_address")
    if ray_address:
        ray.init(address=ray_address)
    else:
        ray.init(num_gpus=int(probe.get("workers", 1)))
    server = subprocess.Popen(serve_command(config), cwd=PROJECT)
    try:
        wait_for_probe(server, probe_url, float(probe.get("start_timeout_seconds", 600)))
        agent = config["agent"]
        cli_path = shutil.which(agent["cli"])
        if not cli_path:
            raise FileNotFoundError(f"agent CLI not found: {agent['cli']}")
        arguments = [
            "--task", str(project_path(config["task"])),
            "--run-id", run_id,
            "--runs-root", str(runs_root),
            "--harness", agent["harness"],
            "--model", agent["model"],
            "--cli-path", cli_path,
            "--probe-url", probe_url,
            "--timeout-minutes", str(agent.get("timeout_minutes", 60)),
        ]
        if agent.get("reasoning_effort"):
            arguments.extend(["--reasoning-effort", agent["reasoning_effort"]])
        run_script("run_agent.py", *arguments)

        task = json.loads(project_path(config["task"]).read_text(encoding="utf-8"))
        suite = load_suite(project_path(config["suite"]), config.get("concept_id"))
        submission_path = runs_root / run_id / "workspace" / "submission.json"
        feature_id = int(json.loads(submission_path.read_text(encoding="utf-8"))["feature_id"])
        expert_id = int(config["expert_feature_id"])
        probed = query_features(
            probe_url,
            [case["text"] for case in suite["activation_cases"]],
            [feature_id, expert_id],
        )
        activation = {
            "schema": 1,
            "feature_id": feature_id,
            "expert_feature_id": expert_id,
            "exact_match": feature_id == expert_id,
            **score_probe_results(
                suite["activation_cases"], probed, int(task["sae"]["feature_count"])
            ),
        }
        (output_dir / "activation.json").write_text(
            json.dumps(activation, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        stop_process(server)
        ray.shutdown()

    sae = config["sae"]
    run_script(
        "fetch_gemma_feature.py",
        "--layer", str(sae["layer"]),
        "--width", str(sae["width"]),
        "--average-l0", str(sae["average_l0"]),
        "--feature-id", str(feature_id),
        "--params", str(project_path(sae["path"])),
        "--resolved-revision", str(sae["resolved_revision"]),
        "--output-dir", str(output_dir / "features"),
    )
    feature = output_dir / "features" / f"gemma2_9b_it_l{sae['layer']}_w{sae['width']}_feature_{feature_id}.npz"
    steering = config["steering"]
    run_script(
        "evaluate_gemma_feature.py",
        "--model-path", str(project_path(config["model_path"])),
        "--feature", str(feature),
        "--suite", str(project_path(config["suite"])),
        "--alphas", str(steering["alphas"]),
        "--max-new-tokens", str(steering["max_new_tokens"]),
        "--output", str(output_dir / "steering.json"),
        *(["--concept-id", config["concept_id"]] if config.get("concept_id") else []),
    )
    judge = config["judge"]
    run_script(
        "judge_feature_steering.py",
        "--result", str(output_dir / "steering.json"),
        "--suite", str(project_path(config["suite"])),
        "--output-prefix", str(output_dir / "judgment"),
        "--provider", judge["provider"],
        "--model-name", judge["model"],
        "--api-key-env", judge["api_key_env"],
        "--repeats", str(judge.get("repeats", 2)),
        *(["--concept-id", config["concept_id"]] if config.get("concept_id") else []),
    )
    print(json.dumps({"run_id": run_id, "feature_id": feature_id, "output_dir": str(output_dir)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="sae-scientist")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.config)
    else:
        reproduce(args.config)


if __name__ == "__main__":
    main()
