import json
from types import SimpleNamespace

from sae_scientist import preflight


def setup_config(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "gemma2", "hidden_size": 3584}))
    (model / "model.safetensors").write_bytes(b"synthetic")
    (model / "tokenizer.json").write_text("{}")
    sae = tmp_path / "params.npz"
    sae.write_bytes(b"synthetic file; no real weights")
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight.shutil, "which", lambda _: "/synthetic/cli")
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "synthetic-secret-do-not-print")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("OPENAI_API_VERSION", "synthetic-version")
    return [{"config": {"model_path": str(model), "sae": {"path": str(sae)},
                        "agent": {"cli": "synthetic-agent"}, "judge": {"model": "test-deployment"},
                        "probe": {"host": "127.0.0.1", "port": 0}}}]


def test_environment_check_is_local_and_redacts_credentials(tmp_path, monkeypatch):
    jobs = setup_config(tmp_path, monkeypatch)
    before = sorted(tmp_path.rglob("*"))
    report = preflight.check_environment(jobs)
    assert report["ok"]
    assert "synthetic-secret" not in json.dumps(report)
    assert sorted(tmp_path.rglob("*")) == before


def test_all_setup_errors_reported_together(tmp_path, monkeypatch):
    jobs = setup_config(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(preflight.shutil, "which", lambda _: None)
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _: None)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://user:synthetic-secret@example.com")
    jobs[0]["config"]["sae"]["path"] = str(tmp_path / "missing.npz")
    report = preflight.check_environment(jobs)
    assert not report["ok"]
    assert len(report["errors"]) >= 6
    assert "synthetic-secret" not in json.dumps(report)


def test_configured_key_follows_judge_fallback(tmp_path, monkeypatch):
    jobs = setup_config(tmp_path, monkeypatch)
    jobs[0]["config"]["judge"]["api_key_env"] = "CUSTOM_JUDGE_KEY"
    monkeypatch.delenv("CUSTOM_JUDGE_KEY", raising=False)
    assert preflight.check_environment(jobs)["ok"]


def test_missing_shard_and_wrong_architecture_fail_before_inference(tmp_path, monkeypatch):
    jobs = setup_config(tmp_path, monkeypatch)
    model = tmp_path / "model"
    (model / "config.json").write_text(json.dumps({"model_type": "other", "hidden_size": 4096}))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
    report = preflight.check_environment(jobs)
    assert not report["ok"]
    assert any("architecture" in error for error in report["errors"])
    assert any("shard" in error for error in report["errors"])
