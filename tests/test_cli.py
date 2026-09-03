import json
from pathlib import Path

import pytest

from sae_bench.cli import read_config, serve_command


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
