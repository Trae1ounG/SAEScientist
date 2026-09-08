import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inspect_dataset", ROOT / "scripts/inspect_dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_all_published_tasks_resolve():
    rows = module.inspect_dataset(ROOT / "data/benchmark.json")
    assert len(rows) == 20
    assert {r["layer"] for r in rows} == {9, 20}
    assert {r["feature_count"] for r in rows} == {131072}
    assert {r["evaluation_prompts"] for r in rows} == {20}
    assert {r["calibration_prompts"] for r in rows} == {5}


def test_original_prompt_overlap_is_reported():
    rows = module.inspect_dataset(ROOT / "data/benchmark.json")
    shared = {r["task_id"]: r["shared_calibration_evaluation_prompts"] for r in rows
              if r["shared_calibration_evaluation_prompts"]}
    assert shared == {name: 1 for name in (
        "gemma_french_005", "gemma_spanish_006", "gemma_portuguese_007", "gemma_german_008")}


def test_duplicate_prompt_ids_fail(monkeypatch):
    original = module.steering_sets

    def duplicate(*args):
        calibration, evaluation = original(*args)
        return calibration, evaluation + [evaluation[0]]

    monkeypatch.setattr(module, "steering_sets", duplicate)
    with pytest.raises(ValueError, match="duplicate evaluation prompt IDs"):
        module.inspect_dataset(ROOT / "data/benchmark.json")


def test_missing_suite_is_an_error(tmp_path):
    benchmark = json.loads((ROOT / "data/benchmark.json").read_text())
    benchmark["tasks"][0]["suite"] = "data/missing.json"
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(benchmark))
    with pytest.raises(FileNotFoundError):
        module.inspect_dataset(path)


def test_duplicate_task_is_an_error(tmp_path):
    benchmark = json.loads((ROOT / "data/benchmark.json").read_text())
    benchmark["tasks"].append(benchmark["tasks"][0])
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(benchmark))
    with pytest.raises(ValueError, match="duplicate task"):
        module.inspect_dataset(path)


def test_wrong_expert_is_an_error(tmp_path):
    benchmark = json.loads((ROOT / "data/benchmark.json").read_text())
    benchmark["tasks"][0]["expert_feature_id"] = -1
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(benchmark))
    with pytest.raises(ValueError, match="Expert ID mismatch"):
        module.inspect_dataset(path)
