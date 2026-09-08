"""Runner tests use synthetic measurements only, never author experiment records."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from sae_scientist.cli import steering_parameters

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_benchmark", ROOT / "scripts/run_benchmark.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def settings(tmp_path):
    config = module.read(ROOT / "configs/benchmark.json")
    config["output_dir"] = str(tmp_path / "experiment")
    config["judge"]["model"] = "test-deployment"
    return config


def synthetic_outputs(job, feature_target=3):
    cfg = job["config"]
    folder = Path(cfg["output_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    records = {
        "config.json": cfg,
        "activation.json": {
            "task_id": job["task_id"], "run_id": cfg["agent"]["run_id"],
            "model": cfg["agent"]["model"], "harness": cfg["agent"]["harness"],
            "feature_id": 123,
            "activation_rank": {"positive": {"mean_rank": 10}, "activation_auroc": 0.9},
            "expert_activation_rank": {"positive": {"mean_rank": 10}},
        },
        "audit.json": {"runs": [{"run_id": cfg["agent"]["run_id"], "eligible": True}]},
        "steering.json": {"feature": {"feature_id": 123}},
        "judgment_summary.json": {
            "result": str(folder / "steering.json"),
            "expected_rows": 40, "valid_rows": 40, "error_rows": 0,
            "conditions": {
                "feature": {"target_relevance": feature_target},
                "baseline": {"target_relevance": 1},
                "random": {"target_relevance": 0},
            },
        },
    }
    for name, data in records.items():
        (folder / name).write_text(json.dumps(data))


def test_full_plan_and_no_writes(tmp_path):
    config = settings(tmp_path)
    jobs = module.plan(config)
    assert len(jobs) == 60
    assert len({j["task_id"] for j in jobs}) == 20
    assert {j["config"]["sae"]["layer"] for j in jobs} == {9, 20}
    assert len({j["config"]["output_dir"] for j in jobs}) == 60
    assert not Path(config["output_dir"]).exists()


def test_subset_keeps_matching_expert_protocol(tmp_path):
    jobs = module.plan(settings(tmp_path), ["gemma_french_005"], 1)
    assert len(jobs) == 1
    cfg = jobs[0]["config"]
    assert cfg["expert_feature_id"] == 105738
    assert cfg["steering"]["max_new_tokens"] == 128
    assert cfg["steering"]["expert_alpha"] == 120
    assert cfg["sae"]["layer"] == 9


def test_input_content_changes_invalidate_saved_run(tmp_path, monkeypatch):
    config = settings(tmp_path)
    jobs = module.plan(config, ["gemma_cat_001"], 1)
    synthetic_outputs(jobs[0])
    original = module.steering_sets

    def changed_inputs(*args):
        calibration, evaluation = copy.deepcopy(original(*args))
        evaluation[0]["prompt"] += " Changed input."
        return calibration, evaluation

    monkeypatch.setattr(module, "steering_sets", changed_inputs)
    changed_jobs = module.plan(config, ["gemma_cat_001"], 1)
    with pytest.raises(ValueError, match="configuration changed"):
        module.summarize(changed_jobs)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_replicate_count(tmp_path, count):
    with pytest.raises(ValueError, match="replicates"):
        module.plan(settings(tmp_path), replicates=count)


def test_unknown_task(tmp_path):
    with pytest.raises(ValueError, match="unknown task"):
        module.plan(settings(tmp_path), ["no-such-task"])


def test_candidate_and_expert_calibration():
    cfg = {"policy": "expert_centered", "expert_alpha": 160, "max_new_tokens": 192}
    before = copy.deepcopy(cfg)
    candidate = steering_parameters(cfg, False)
    assert candidate["alphas"] == "80.0,120.0,160.0,200.0,240.0"
    assert candidate["fallback_alphas"] == "10.0,20.0,40.0,60.0"
    expert = steering_parameters(cfg, True)
    assert expert["alphas"] == "160.0"
    assert expert["fallback_alphas"] == ""
    assert cfg == before


def test_mean_then_sample_std(tmp_path):
    jobs = module.plan(settings(tmp_path), ["gemma_cat_001", "gemma_french_005"], 2)
    for job in jobs:
        synthetic_outputs(job, 2 if job["replicate"] == 1 else 4)
    result = module.summarize(jobs)
    assert result["task_count"] == 2
    assert result["replicates"] == 2
    assert result["scores"]["Steering"]["mean"] == 50
    assert result["scores"]["Steering"]["sample_std"] == pytest.approx(50 / 2**0.5)
    assert result["scores"]["Overall"]["mean"] == pytest.approx(230 / 3)


def test_single_run_std_is_null(tmp_path):
    jobs = module.plan(settings(tmp_path), ["gemma_cat_001"], 1)
    synthetic_outputs(jobs[0])
    assert module.summarize(jobs)["scores"]["Overall"]["sample_std"] is None


@pytest.mark.parametrize("filename", ["activation.json", "judgment_summary.json", "audit.json"])
def test_missing_input_prevents_summary(tmp_path, filename):
    jobs = module.plan(settings(tmp_path), ["gemma_cat_001"], 1)
    synthetic_outputs(jobs[0])
    (Path(jobs[0]["config"]["output_dir"]) / filename).unlink()
    with pytest.raises(FileNotFoundError):
        module.summarize(jobs)


@pytest.mark.parametrize("change", ["config", "task", "run", "model", "audit", "feature", "judge", "coverage"])
def test_mismatched_or_incomplete_outputs_rejected(tmp_path, change):
    jobs = module.plan(settings(tmp_path), ["gemma_cat_001"], 1)
    synthetic_outputs(jobs[0])
    folder = Path(jobs[0]["config"]["output_dir"])
    if change == "config":
        path = folder / "config.json"; data = module.read(path); data["steering"]["expert_alpha"] = 10
    elif change in ("task", "run", "model"):
        path = folder / "activation.json"; data = module.read(path)
        data[{"task": "task_id", "run": "run_id", "model": "model"}[change]] = "different"
    elif change == "audit":
        path = folder / "audit.json"; data = module.read(path); data["runs"][0]["eligible"] = False
    elif change == "feature":
        path = folder / "steering.json"; data = module.read(path); data["feature"]["feature_id"] = 99
    elif change == "judge":
        path = folder / "judgment_summary.json"; data = module.read(path); data["result"] = "wrong.json"
    else:
        path = folder / "judgment_summary.json"; data = module.read(path); data["valid_rows"] = 39
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.summarize(jobs)


def test_dry_run_and_score_only(tmp_path, monkeypatch, capsys):
    cfg = settings(tmp_path)
    path = tmp_path / "config.json"; path.write_text(json.dumps(cfg))
    base = ["run_benchmark.py", "--config", str(path), "--task-id", "gemma_cat_001", "--replicates", "1"]
    monkeypatch.setattr(sys, "argv", base + ["--dry-run"])
    module.main()
    assert json.loads(capsys.readouterr().out)["planned_runs"] == 1
    assert not Path(cfg["output_dir"]).exists()
    jobs = module.plan(cfg, ["gemma_cat_001"], 1)
    synthetic_outputs(jobs[0])
    monkeypatch.setattr(module, "reproduce", lambda *_: pytest.fail("score-only executed inference"))
    monkeypatch.setattr(sys, "argv", base + ["--score-only"])
    module.main()
    assert (Path(cfg["output_dir"]) / "summary.json").exists()


def test_execute_orchestration_without_external_services(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "check_environment", lambda _: {"ok": True})
    cfg = settings(tmp_path)
    cfg["model_path"] = str(tmp_path / "model")
    cfg["sae_root"] = str(tmp_path / "saes")
    jobs = module.plan(cfg, ["gemma_cat_001"], 2)
    Path(cfg["model_path"]).mkdir()
    sae = Path(jobs[0]["config"]["sae"]["path"])
    sae.parent.mkdir(parents=True); sae.touch()
    path = tmp_path / "config.json"; path.write_text(json.dumps(cfg))
    calls = []
    def simulate(config_path):
        job = jobs[len(calls)]
        assert module.read(config_path) == job["config"]
        calls.append(config_path)
        synthetic_outputs(job)
    monkeypatch.setattr(module, "reproduce", simulate)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--config", str(path),
                                     "--task-id", "gemma_cat_001", "--replicates", "2"])
    module.main()
    assert len(calls) == 2
    with pytest.raises(FileExistsError):
        module.main()


def test_output_override_in_dry_run_and_scoring(tmp_path, monkeypatch, capsys):
    cfg = settings(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    override = tmp_path / "smoke"
    args = ["run_benchmark.py", "--config", str(config_path),
            "--task-id", "gemma_cat_001", "--replicates", "1",
            "--output-dir", str(override)]
    monkeypatch.setattr(sys, "argv", args + ["--dry-run"])
    module.main()
    assert json.loads(capsys.readouterr().out)["output_dir"] == str(override)
    assert not override.exists()
    resolved = copy.deepcopy(cfg)
    resolved["output_dir"] = str(override)
    synthetic_outputs(module.plan(resolved, ["gemma_cat_001"], 1)[0])
    monkeypatch.setattr(module, "reproduce", lambda *_: pytest.fail("unexpected inference"))
    monkeypatch.setattr(sys, "argv", args + ["--score-only"])
    module.main()
    assert (override / "summary.json").is_file()
    assert not Path(cfg["output_dir"]).exists()


@pytest.mark.parametrize("section,key,value", [
    ("probe", "host", "0.0.0.0"), ("probe", "port", True),
    ("probe", "port", 65536), ("probe", "workers", 1.5),
    ("agent", "model", ""), ("agent", "timeout_minutes", -1),
    ("agent", "timeout_minutes", float("nan")),
])
def test_invalid_runtime_config_rejected(tmp_path, section, key, value):
    cfg = settings(tmp_path)
    cfg[section][key] = value
    with pytest.raises(ValueError):
        module.plan(cfg)


def test_self_consistent_but_incomplete_judge_rejected(tmp_path):
    jobs = module.plan(settings(tmp_path), ["gemma_cat_001"], 1)
    synthetic_outputs(jobs[0])
    path = Path(jobs[0]["config"]["output_dir"]) / "judgment_summary.json"
    data = module.read(path)
    data["expected_rows"] = data["valid_rows"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="evaluation protocol"):
        module.summarize(jobs)


def test_failed_preflight_creates_no_experiment(tmp_path, monkeypatch):
    cfg = settings(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(module, "check_environment", lambda _: {"ok": False, "errors": ["missing credentials"]})
    monkeypatch.setattr(module, "reproduce", lambda _: pytest.fail("inference called"))
    base = ["run_benchmark.py", "--config", str(path), "--replicates", "1"]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(ValueError, match="missing credentials"):
        module.main()
    monkeypatch.setattr(sys, "argv", base + ["--check"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    assert not Path(cfg["output_dir"]).exists()


def test_atomic_summary_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    path.write_text('{"previous": true}\n')
    def fail(*_):
        raise OSError("synthetic replacement failure")
    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        module.write_summary(path, {"new": True})
    assert json.loads(path.read_text()) == {"previous": True}
    assert list(tmp_path.iterdir()) == [path]
