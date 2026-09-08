import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from sae_scientist.cli import read_config, serve_command
from sae_scientist import cli


def test_config_requires_complete_pipeline(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_path": "model"}))
    with pytest.raises(ValueError, match="config is missing"):
        read_config(path)


def test_serve_command_uses_configured_paths(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model_path": "model",
                "sae": {"path": "sae.npz", "layer": 9},
                "task": "task.json",
                "suite": "suite.json",
                "expert_feature_id": 1,
                "agent": {},
                "steering": {},
                "judge": {},
                "output_dir": "outputs",
                "probe": {"host": "127.0.0.1", "port": 9000, "workers": 2},
            }
        )
    )
    command = serve_command(read_config(path))
    assert command[command.index("--layer") + 1] == "9"
    assert command[command.index("--workers") + 1] == "2"
    assert command[command.index("--port") + 1] == "9000"


def pipeline_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/cat.json").read_text())
    config["output_dir"] = str(tmp_path / "output")
    config["agent"]["run_id"] = "synthetic-run"
    config["steering"] = {"policy": "expert_centered", "expert_alpha": 160, "max_new_tokens": 192}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return config, path


def test_pipeline_releases_ray_when_probe_cannot_start(tmp_path, monkeypatch):
    _, path = pipeline_config(tmp_path)
    events = []
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        init=lambda **_: events.append("init"), shutdown=lambda: events.append("shutdown")))

    def fail(*args, **kwargs):
        raise OSError("synthetic launch failure")

    monkeypatch.setattr(cli.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="synthetic launch failure"):
        cli.reproduce(path)
    assert events == ["init", "shutdown"]


def test_pipeline_wires_discovery_evaluation_and_scoring(tmp_path, monkeypatch):
    """Exercise orchestration with synthetic services, without inference or API calls."""
    config, path = pipeline_config(tmp_path)
    output = Path(config["output_dir"])
    events, commands = [], []
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        init=lambda **_: events.append("ray init"), shutdown=lambda: events.append("ray shutdown")))
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_, **__: object())
    monkeypatch.setattr(cli, "wait_for_probe", lambda *_: events.append("probe ready"))
    monkeypatch.setattr(cli, "stop_process", lambda *_: events.append("probe stopped"))
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/synthetic/agent")

    def run_script(name, *args):
        commands.append((name, args))
        events.append(name)
        if name == "run_agent.py":
            workspace = output / "runs" / "synthetic-run" / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "submission.json").write_text('{"feature_id": 123}')
        if name == "audit_agent_runs.py":
            (output / "audit.json").write_text(json.dumps({"runs": [
                {"run_id": "synthetic-run", "eligible": True}]}))

    def query(url, texts, ids):
        assert ids == [123, config["expert_feature_id"]]
        return [{"selected_features": [
            {"feature_id": feature, "activation": 1.0, "rank": 10}
            for feature in ids]} for _ in texts]

    monkeypatch.setattr(cli, "run_script", run_script)
    monkeypatch.setattr(cli, "query_features", query)
    cli.reproduce(path)
    assert [name for name, _ in commands] == [
        "run_agent.py", "audit_agent_runs.py", "fetch_gemma_feature.py",
        "evaluate_gemma_feature.py", "judge_feature_steering.py", "score_paper_run.py"]
    assert events.index("ray shutdown") < events.index("evaluate_gemma_feature.py")
    args = dict(zip(commands[3][1][::2], commands[3][1][1::2]))
    assert args["--alphas"] == "80.0,120.0,160.0,200.0,240.0"
    assert args["--fallback-alphas"] == "10.0,20.0,40.0,60.0"
    assert args["--max-new-tokens"] == "192"
    activation = json.loads((output / "activation.json").read_text())
    assert activation["feature_id"] == 123
    assert activation["activation_rank"]["activation_auroc"] == 0.5
