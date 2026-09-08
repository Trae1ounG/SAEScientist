"""The public source tree contains inputs and synthetic tests, not experiment outputs."""
import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_source_release", ROOT / "scripts/check_source_release.py")
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


def test_reference_files_contain_only_configuration():
    allowed = {"schema", "status", "feature_case", "suite", "concept_id", "protocol"}
    paths = list((ROOT / "data/eval").glob("*_reference.json"))
    assert len(paths) == 20
    for path in paths:
        assert set(json.loads(path.read_text())) <= allowed, path


def test_no_bundled_author_results():
    # Local reviewer runs in ignored outputs/ are intentionally permitted.
    assert not list((ROOT / "results").rglob("*.json"))
    assert not list((ROOT / "data").rglob("*.jsonl"))
    assert not list((ROOT / "assets").rglob("*.csv"))


def test_release_check_detects_outputs_and_credentials(tmp_path):
    names = ["results/measurements.json", ".env.local", "data/hidden.json"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"nested": [{"scores": {"fake": 1}}]}))
    errors = RELEASE.inspect_paths(tmp_path, names)
    assert len(errors) == 3
    assert "evaluation output fields: scores" in errors[-1] or any("fields: scores" in e for e in errors)


def test_release_check_accepts_inputs_and_pending_deletions(tmp_path):
    path = tmp_path / "data/task.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"feature_id": 1, "protocol": {"alpha": 80}}))
    assert RELEASE.inspect_paths(tmp_path, ["data/task.json", "results/deleted.json"]) == []


def test_release_check_rejects_symlinks(tmp_path):
    target = tmp_path / "target"
    target.write_text("synthetic")
    (tmp_path / "link").symlink_to(target)
    assert "symlinks" in RELEASE.inspect_paths(tmp_path, ["link"])[0]
